# filter.py: keep only the questions a second (filter) model ALSO answers wrong, closed-book.
#   Replaces vLLM with Hugging Face Transformers, BitsAndBytes quantization (4-bit/8-bit), and batching.
#
# Usage: python filter.py --input MuSiQue.pkl --model meta-llama/Llama-3.3-70B-Instruct --n 500 --quant 4bit --batch-size 16
import argparse
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


def normalize(text):
    """Lowercase, no accents, no punctuation or articles, no spaces."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^a-z0-9 ]", " ", text))
    return text.replace(" ", "")


def is_correct(prediction, golds):
    """Correct if the prediction and any gold answer contain each other after normalization."""
    p = normalize(prediction)
    return bool(p) and any(normalize(g) and (normalize(g) in p or p in normalize(g)) for g in golds)


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def load_llm(model_id, quant="none"):
    """Load model and left-padded tokenizer with optional 4-bit/8-bit quantization or pre-quantized AWQ/GPTQ."""
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"device_map": "auto"}
    is_prequantized = any(q in model_id.lower() for q in ("awq", "gptq", "gguf"))

    if not is_prequantized and torch.cuda.is_available():
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16
    elif not is_prequantized:
        kwargs["torch_dtype"] = torch.float16 if torch.backends.mps.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return model, tok


def generate_batched(llm, prompts, max_new_tokens=32, batch_size=16, desc="Generating"):
    """Run batched greedy generation with left-padding and tqdm progress."""
    model, tok = llm
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tok(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen_ids = outs[:, inputs.input_ids.shape[1] :]
        decoded = tok.batch_decode(gen_ids, skip_special_tokens=True)
        results.extend([d.strip() for d in decoded])
    return results


def run_pass(llm, candidates, model_name, role="filter", judge=True, batch_size=16):
    """Run closed-book QA and optional LLM judge over a list of candidate dicts."""
    _, tok = llm
    prompts = [chat(tok, f"{INSTRUCTION}\n\nQuestion: {ex['question']}\nAnswer:") for ex in candidates]
    raws = generate_batched(llm, prompts, max_new_tokens=32, batch_size=batch_size, desc=f"[{role}] Answering")
    answers = [r.split("\n")[0].strip() for r in raws]

    str_matches = [is_correct(a, ex["gold_answers"]) for ex, a in zip(candidates, answers)]
    judged_correct = set()

    if judge:
        to_judge = [i for i, m in enumerate(str_matches) if not m]
        if to_judge:
            judge_prompts = [
                chat(
                    tok,
                    JUDGE.format(
                        question=candidates[i]["question"],
                        gold=" / ".join(candidates[i]["gold_answers"]),
                        answer=answers[i],
                    ),
                )
                for i in to_judge
            ]
            verdicts = generate_batched(
                llm, judge_prompts, max_new_tokens=3, batch_size=batch_size, desc=f"[{role}] Judging"
            )
            judged_correct = {i for i, v in zip(to_judge, verdicts) if v.lower().startswith("yes")}

    logs = []
    for i, (ans, raw, sm) in enumerate(zip(answers, raws, str_matches)):
        jc = i in judged_correct
        logs.append(
            {
                "model": model_name,
                "role": role,
                "answer": ans,
                "raw_answer": raw,
                "string_match": sm,
                "judge_match": jc if judge else None,
                "correct": sm or jc,
            }
        )
    return logs


def summarize(records, logs, model_name, role, elapsed, kept_count, extra=None):
    """Build summary statistics dictionary."""
    n_total = len(logs)
    n_sm = sum(1 for l in logs if l["string_match"])
    n_jm = sum(1 for l in logs if l.get("judge_match"))
    entry = {
        "model": model_name,
        "role": role,
        "evaluated": n_total,
        "string_match_correct": n_sm,
        "judge_correct": n_jm,
        "total_correct": n_sm + n_jm,
        "kept_wrong": kept_count,
        "elapsed_sec": round(elapsed, 2),
    }
    if extra:
        entry.update(extra)
    return entry


def save_stats(dataset, entry):
    """Append pass statistics to <dataset>_stats.json."""
    path = f"{dataset}_stats.json"
    history = []
    if os.path.exists(path):
        with open(path, "r") as f:
            history = json.load(f)
    history.append(entry)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    return path


def print_summary(entry):
    print(
        f"[{entry['role']} | {entry['model']}] Evaluated: {entry['evaluated']} | "
        f"String match: {entry['string_match_correct']} | Judge: {entry['judge_correct']} | "
        f"Kept (failed): {entry['kept_wrong']} | Time: {entry['elapsed_sec']}s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="candidate pool .pkl from MuSiQue.py")
    parser.add_argument("--model", required=True, help="filter model, e.g. meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--n", type=int, default=500, help="number of questions to keep")
    parser.add_argument("--batch-size", type=int, default=16, help="inference batch size")
    parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="4bit", help="BitsAndBytes quantization")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        candidates = pickle.load(f)

    start = time.time()
    llm = load_llm(args.model, quant=args.quant)
    name = args.model.split("/")[-1]

    logs = run_pass(llm, candidates, args.model, role="filter", judge=True, batch_size=args.batch_size)

    kept = []
    for ex, l in zip(candidates, logs):
        ex.setdefault("filter_log", []).append(l)
        if not l["correct"]:
            ex["failed_by"] = ex.get("failed_by", [ex["model"]] if "model" in ex else []) + [args.model]
            ex[f"answer_{name}"] = l["answer"]
            ex[f"raw_answer_{name}"] = l["raw_answer"]
            kept.append(ex)
            if len(kept) == args.n:
                break

    dataset = candidates[0].get("dataset", "MuSiQue") if candidates else "MuSiQue"
    out_file = args.output if args.output else f"{dataset}_filtered.pkl"

    if not out_file.endswith(".pkl"):
        out_file += ".pkl"

    if os.path.abspath(out_file) == os.path.abspath(args.input):
        raise ValueError(
            "Output file matches input file. Specify --output to avoid overwriting the input."
        )
    with open(out_file, "wb") as f:
        pickle.dump(kept, f)

    entry = summarize(candidates, logs, args.model, "filter", time.time() - start, len(kept), {"output_file": out_file})
    save_stats(dataset, entry)
    print_summary(entry)
    print(f"Saved {len(kept)} kept questions to {out_file}")


if __name__ == "__main__":
    main()