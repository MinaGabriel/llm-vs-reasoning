# facts.py: fact extraction + fact judging (P(True)). Dataset-agnostic library + CLI.
#   Paragraphs can be dicts or plain strings, so the same code serves MuSiQue now and retrieved Wikipedia later.
#
#   1. Fact extraction: the model lists the atomic facts stated in EACH paragraph (one paragraph at a time, so every
#      fact keeps its source). The question is NOT shown, so extraction is neutral.
#   2. Fact judging: the same model judges each fact against its paragraph, (A) True / (B) False;
#      P(True) = p(A) / (p(A) + p(B)) from the next-token probabilities. One judgment -> c = 0.5 in NARS.
#   Facts with P(True) < min_p_true are moved to "facts_dropped" (both systems get the same kept facts).
#   Speed: each unique paragraph is processed once; prompts are sorted by length before batching.
#
# CLI:  python facts.py --input MuSiQue_final.pkl --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
import argparse
import pickle
import time

import torch
from tqdm import tqdm

from filter import chat, generate_batched, load_llm, normalize, save_stats

EXTRACT = """Extract every fact stated in the paragraph below.
Rules:
- One fact per line, starting with "- ".
- Each fact is one short, self-contained sentence with a single claim.
- Use full names; never use pronouns ("he", "she", "it", "the film").
- Only facts stated in the paragraph; do not add anything you know.

Title: {title}
Paragraph: {text}

Facts:"""

JUDGE_FACT = """Title: {title}
Paragraph: {text}

Statement: {fact}
Is the statement true according to the paragraph?
(A) True
(B) False
Answer with A or B."""


def as_paragraphs(paragraphs):
    """Plain strings or dicts -> [{"title": ..., "text": ...}]"""
    out = []
    for p in paragraphs or []:
        if isinstance(p, str):
            out.append({"title": "", "text": p})
        else:
            out.append({"title": p.get("title", ""), "text": p.get("text") or p.get("paragraph_text", "")})
    return out


def parse_facts(text):
    """'- Rumi was born in Balkh.\\n- ...' -> ['Rumi was born in Balkh.', ...] (no duplicates, >= 3 words)."""
    facts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("-", "*", "•")):
            fact = line.lstrip("-*• ").strip()
            if len(fact.split()) >= 3:
                facts.append(fact)
    return list(dict.fromkeys(facts))


@torch.no_grad()
def p_true_batched(llm, prompts, batch_size=16, desc="Judging facts"):
    """P(True) per prompt from the first answer token: p(A) / (p(A) + p(B)), plus the mass p(A) + p(B)."""
    model, tok = llm
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for start in tqdm(range(0, len(order), batch_size), desc=desc):
        idx = order[start:start + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        probs = torch.softmax(model(**enc).logits[:, -1, :].float(), dim=-1)      # left padding: last position
        for i, pa, pb in zip(idx, probs[:, a_id].tolist(), probs[:, b_id].tolist()):
            out[i] = {"p_true": pa / (pa + pb) if pa + pb > 0 else 0.5, "mass": pa + pb}
    return out


def extract_and_judge(llm, paragraph_lists, model_id, batch_size=16):
    """
    Description: Facts with P(True) for many questions at once. Each unique paragraph is processed once.
    Input:       [[paragraph, ...], ...]   one list of paragraphs per question (strings or dicts)
    Output:      ([[{"text", "paragraph_idx", "paragraph_title", "p_true", "mass", "c", "model"}, ...], ...], n_unique)
    """
    _, tok = llm
    per_question = [as_paragraphs(plist) for plist in paragraph_lists]
    unique = list({(p["title"], p["text"]): p for plist in per_question for p in plist}.values())
    if not unique:
        return [[] for _ in paragraph_lists], 0

    raw = generate_batched(llm, [chat(tok, EXTRACT.format(title=p["title"], text=p["text"])) for p in unique],
                           max_new_tokens=384, batch_size=batch_size, desc="Extracting facts")
    facts_of = {(p["title"], p["text"]): parse_facts(r) for p, r in zip(unique, raw)}

    pairs = [(key, f) for key, fs in facts_of.items() for f in fs]
    scores = p_true_batched(llm, [chat(tok, JUDGE_FACT.format(title=k[0], text=k[1], fact=f)) for k, f in pairs],
                            batch_size=batch_size)
    score_of = dict(zip(pairs, scores))

    out = []
    for plist in per_question:
        facts = []
        for j, p in enumerate(plist):
            key = (p["title"], p["text"])
            for f in facts_of[key]:
                s = score_of[(key, f)]
                facts.append({"text": f, "paragraph_idx": j, "paragraph_title": p["title"],
                              "p_true": s["p_true"], "mass": s["mass"], "c": 0.5, "model": model_id})
        out.append(facts)
    return out, len(unique)


def attach_facts(records, all_facts, n_unique, model_id, min_p_true, elapsed, extra=None):
    """Store facts in the records (low P(True) moved to facts_dropped); return the stats entry for the paper."""
    n_extracted = n_dropped = 0
    dropped_examples = []
    for r, facts in zip(records, all_facts):
        paragraphs = r.get("paragraphs") or []
        for f in facts:                                           # analysis only: never shown to the systems
            p = paragraphs[f["paragraph_idx"]] if f["paragraph_idx"] < len(paragraphs) else {}
            f["is_supporting"] = p.get("is_supporting") if isinstance(p, dict) else None
        low = [f for f in facts if f["p_true"] < min_p_true]
        n_extracted += len(facts)
        n_dropped += len(low)
        dropped_examples += [{"question": r["question"], "fact": f["text"], "p_true": round(f["p_true"], 3)} for f in low]
        r["facts"] = [f for f in facts if f["p_true"] >= min_p_true]
        r["facts_dropped"] = low
        r["hop_coverage"] = [any(normalize(h["answer"]) in normalize(f["text"]) for f in r["facts"])
                             for h in r.get("decomposition") or [] if normalize(h.get("answer", ""))]
    sup = [f for r in records for f in r["facts"] if f.get("is_supporting")]
    dis = [f for r in records for f in r["facts"] if f.get("is_supporting") is False]
    hops = [c for r in records for c in r["hop_coverage"]]
    entry = {
        "model": model_id,
        "role": "facts",
        "questions": len(records),
        "unique_paragraphs": n_unique,
        "facts_extracted": n_extracted,
        "facts_dropped_low_p_true": n_dropped,
        "min_p_true": min_p_true,
        "facts_kept": n_extracted - n_dropped,
        "facts_per_question": round((n_extracted - n_dropped) / max(1, len(records)), 1),
        "supporting_facts": len(sup),
        "distractor_facts": len(dis),
        "mean_p_true_supporting": round(sum(f["p_true"] for f in sup) / max(1, len(sup)), 3),
        "mean_p_true_distractor": round(sum(f["p_true"] for f in dis) / max(1, len(dis)), 3),
        "hop_coverage": f"{sum(hops)}/{len(hops)}",
        "dropped_examples": dropped_examples[:20],
        "extract_prompt": EXTRACT,
        "judge_prompt": JUDGE_FACT,
        "elapsed_sec": round(elapsed, 1),
    }
    entry.update(extra or {})
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=None, help="only the first n questions")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    parser.add_argument("--min-p-true", type=float, default=0.5)
    args = parser.parse_args()

    records = pickle.load(open(args.input, "rb"))
    records = records[:args.n] if args.n else records
    start = time.time()
    llm = load_llm(args.model, args.quant)
    all_facts, n_unique = extract_and_judge(llm, [r.get("paragraphs") for r in records], args.model, args.batch_size)
    entry = attach_facts(records, all_facts, n_unique, args.model, args.min_p_true, time.time() - start,
                         {"input_file": args.input})
    out_file = args.input.removesuffix(".pkl") + "_facts.pkl"
    pickle.dump(records, open(out_file, "wb"))
    save_stats(records[0]["dataset"], {**entry, "output_file": out_file})
    print(f"{entry['questions']} questions, {entry['facts_kept']} facts kept, hop coverage {entry['hop_coverage']}. Saved to {out_file}")


if __name__ == "__main__":
    main()