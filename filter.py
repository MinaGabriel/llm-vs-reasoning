# filter.py: knowledge filtering, closed-book (Hugging Face Transformers). Dataset-agnostic library + CLI.
#   A question is KEPT if the model answers it wrong. An answer counts as correct if it string-matches a gold
#   answer OR the same model judges it equivalent ("Prince Philip" = "Philip Mountbatten").
#
# Library (used by pipeline.py and facts.py): load_llm, unload_llm, generate_batched, run_pass, summarize, save_stats
# CLI:  python filter.py --input MuSiQue.pkl --model Qwen/Qwen2.5-7B-Instruct --output MuSiQue_f.pkl
import argparse
import datetime
import gc
import json
import os
import pickle
import re
import time
import unicodedata

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

INSTRUCTION = """Answer with only the answer, in as few words as possible. No sentences, no explanations.

Question: Who directed Jaws?
Answer: Steven Spielberg
Question: What is the capital of the country where the Eiffel Tower is?
Answer: Paris"""

JUDGE = """Question: {question}
Gold answer: {gold}
Model answer: {answer}
Does the model answer refer to the same thing as the gold answer (e.g. "Prince Philip" = "Philip Mountbatten")?
Reply with only yes or no."""


# ---------- grading ----------

def normalize(text):
    """Lowercase, no accents, no punctuation or articles, no spaces: 'The Fiorello La Guardia.' -> 'fiorellolaguardia'"""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^a-z0-9 ]", " ", text))
    return text.replace(" ", "")


def is_correct(prediction, golds):
    """Correct if the prediction and any gold answer contain each other after normalization."""
    p = normalize(prediction)
    return bool(p) and any(normalize(g) and (normalize(g) in p or p in normalize(g)) for g in golds)


# ---------- models ----------

def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def load_llm(model_id, quant="none"):
    """Model + left-padded tokenizer. Pre-quantized AWQ/GPTQ ids load on GPU 0 in fp16; others in bf16 or bitsandbytes."""
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    is_prequantized = any(q in model_id.lower() for q in ("awq", "gptq"))
    if is_prequantized:
        # AWQ/GPTQ kernels cannot be offloaded to CPU/meta; pin directly to GPU 0 in fp16
        kwargs = {"device_map": {"": 0}, "torch_dtype": torch.float16}
    else:
        kwargs = {"device_map": "auto"}
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return model, tok

def unload_llm(llm, model_id=None, delete_cache=False):
    """Free GPU memory; optionally delete the model's files from the Hugging Face cache (saves disk)."""
    model, tok = llm
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if delete_cache and model_id:
        try:
            from huggingface_hub import scan_cache_dir
            info = scan_cache_dir()
            for repo in info.repos:
                if repo.repo_id == model_id:
                    info.delete_revisions(*[r.commit_hash for r in repo.revisions]).execute()
                    print(f"Deleted {model_id} from the Hugging Face cache")
        except Exception as err:
            print(f"Could not delete {model_id} from the cache: {err}")


def batch_size_for(model_id):
    """A sensible batch size from the model size in its name (e.g. '72B' -> 16, '7B' -> 64)."""
    m = re.search(r"(\d+(?:\.\d+)?)B", model_id)
    size = float(m.group(1)) if m else 8
    return 16 if size >= 60 else 32 if size >= 25 else 48 if size >= 10 else 96


def generate_batched(llm, prompts, max_new_tokens=32, batch_size=16, desc="Generating"):
    """Greedy generation in batches. Prompts are sorted by length (less padding), results come back in order."""
    model, tok = llm
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results = [None] * len(prompts)
    for start in tqdm(range(0, len(order), batch_size), desc=desc):
        idx = order[start:start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, text in zip(idx, tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
            results[i] = text.strip()
    return results


# ---------- one filter pass ----------

def run_pass(llm, records, model_id, role="filter", judge=True, batch_size=16):
    """
    Description: The model answers every question closed-book; answers that fail string matching are judged by the
                 same model. Returns one log entry per record (same order).
    Output:      [{"model", "role", "answer", "raw_answer", "string_match", "judge_match", "judge_raw", "correct"}, ...]
    """
    _, tok = llm
    raws = generate_batched(llm, [chat(tok, f"{INSTRUCTION}\n\nQuestion: {r['question']}\nAnswer:") for r in records],
                            max_new_tokens=32, batch_size=batch_size, desc=f"[{model_id.split('/')[-1]}] answering")
    answers = [raw.split("\n")[0].strip() for raw in raws]
    logs = [{"model": model_id, "role": role, "answer": a, "raw_answer": raw,
             "string_match": is_correct(a, r["gold_answers"]), "judge_match": None, "judge_raw": None}
            for r, a, raw in zip(records, answers, raws)]
    if judge:
        idx = [i for i, l in enumerate(logs) if not l["string_match"]]
        verdicts = generate_batched(llm, [chat(tok, JUDGE.format(question=records[i]["question"],
                                                                 gold=" / ".join(records[i]["gold_answers"]),
                                                                 answer=logs[i]["answer"])) for i in idx],
                                    max_new_tokens=3, batch_size=batch_size,
                                    desc=f"[{model_id.split('/')[-1]}] judging")
        for i, v in zip(idx, verdicts):
            logs[i]["judge_raw"] = v
            logs[i]["judge_match"] = v.lower().startswith("yes")
    for l in logs:
        l["correct"] = bool(l["string_match"] or l["judge_match"])
    return logs


def apply_pass(records, logs, model_id):
    """Attach the log to every record; return the records the model answered wrong (failed_by updated)."""
    name = model_id.split("/")[-1]
    wrong = []
    for r, l in zip(records, logs):
        r.setdefault("filter_log", []).append(l)
        if not l["correct"]:
            r["failed_by"] = r.get("failed_by", []) + [model_id]
            r[f"answer_{name}"] = l["answer"]
            wrong.append(r)
    return wrong


# ---------- logging ----------

def summarize(records, logs, model_id, role, elapsed, kept_count, extra=None):
    """Statistics of one pass, for the paper."""
    n = len(logs)
    n_sm = sum(bool(l["string_match"]) for l in logs)
    n_jm = sum(bool(l.get("judge_match")) for l in logs)
    flipped = [{"question": r["question"], "gold": r["gold_answers"], "answer": l["answer"]}
               for r, l in zip(records, logs) if l.get("judge_match")]
    entry = {
        "model": model_id,
        "role": role,
        "evaluated": n,
        "string_match_correct": n_sm,
        "judge_correct": n_jm,
        "total_correct": n_sm + n_jm,
        "answered_wrong": n - n_sm - n_jm,
        "kept": kept_count,
        "pct_known": round(100 * (n_sm + n_jm) / n, 1) if n else None,
        "pct_judge_among_string_wrong": round(100 * n_jm / (n - n_sm), 1) if n > n_sm else None,
        "elapsed_sec": round(elapsed, 1),
        "judge_flipped_examples": flipped[:20],
    }
    entry.update(extra or {})
    return entry


def save_stats(dataset, entry):
    """Append one entry to <dataset>_stats.json (the full history of the run)."""
    path = f"{dataset}_stats.json"
    history = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            history = json.load(f)
    history.append({"step": len(history) + 1, "time": datetime.datetime.now().isoformat(timespec="seconds"), **entry})
    with open(path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    return path


def print_summary(entry):
    print(f"[{entry['role']} | {entry['model']}] evaluated {entry['evaluated']} | string match {entry['string_match_correct']} | "
          f"judge {entry['judge_correct']} | kept {entry['kept']} | {entry['elapsed_sec']} s")


# ---------- CLI (one filter pass on a file) ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=None, help="keep at most n (default: all answered wrong)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    args = parser.parse_args()
    if os.path.abspath(args.output) == os.path.abspath(args.input):
        raise SystemExit("--output must differ from --input")

    records = pickle.load(open(args.input, "rb"))
    dataset = records[0]["dataset"]
    start = time.time()
    llm = load_llm(args.model, args.quant)
    logs = run_pass(llm, records, args.model, batch_size=args.batch_size or batch_size_for(args.model))
    kept = apply_pass(records, logs, args.model)
    kept = kept[:args.n] if args.n else kept
    pickle.dump(kept, open(args.output, "wb"))
    entry = summarize(records, logs, args.model, "filter", time.time() - start, len(kept),
                      {"input_file": args.input, "output_file": args.output})
    save_stats(dataset, entry)
    print_summary(entry)


if __name__ == "__main__":
    main()