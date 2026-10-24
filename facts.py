# facts.py: fact extraction without a separate judging pass. Dataset-agnostic library + CLI.
#   Paragraphs can be dicts or plain strings, so the same code serves MuSiQue now and retrieved Wikipedia later.
#
#   1. Fact extraction: the model lists the atomic facts stated in EACH paragraph (one paragraph at a time, so every
#      fact keeps its source). The question is NOT shown, so extraction is neutral.
#   All extracted facts are kept. No P(True) or token-probability mass is computed.
#   Facts store only text and paragraph provenance, with no truth scores or per-fact model metadata.
#   ONA supplies its own default truth values when converting facts to Narsese.
#   Speed: each unique paragraph is processed once; prompts are sorted by length before batching.
#
# CLI:  python facts.py --input runs/MuSiQue_final_questions.pkl --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
import argparse
import pickle
import time
from pathlib import Path

from filter import chat, generate_batched, load_llm, normalize, save_stats, unload_llm

EXTRACT = """Extract every fact stated in the paragraph below.
Rules:
- One fact per line, starting with "- ".
- Each fact is one short, self-contained sentence with a single claim.
- Use full names; never use pronouns ("he", "she", "it", "the film").
- Only facts stated in the paragraph; do not add anything you know.

Title: {title}
Paragraph: {text}

Facts:"""

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


def extract_facts(llm, paragraph_lists, batch_size=16):
    """
    Description: Extract facts for many questions at once. Each unique paragraph is processed once; no judging.
    Input:       [[paragraph, ...], ...]   one list of paragraphs per question (strings or dicts)
    Output:      ([[{"text", "paragraph_idx", "paragraph_title"}, ...], ...], n_unique)
    """
    _, tok = llm
    per_question = [as_paragraphs(plist) for plist in paragraph_lists]
    unique = list({(p["title"], p["text"]): p for plist in per_question for p in plist}.values())
    if not unique:
        return [[] for _ in paragraph_lists], 0

    raw = generate_batched(llm, [chat(tok, EXTRACT.format(title=p["title"], text=p["text"])) for p in unique],
                           max_new_tokens=384, batch_size=batch_size, desc="Extracting facts")
    facts_of = {(p["title"], p["text"]): parse_facts(r) for p, r in zip(unique, raw)}

    out = []
    for plist in per_question:
        facts = []
        for j, p in enumerate(plist):
            key = (p["title"], p["text"])
            for f in facts_of[key]:
                facts.append({"text": f, "paragraph_idx": j, "paragraph_title": p["title"]})
        out.append(facts)
    return out, len(unique)


def attach_facts(records, all_facts, n_unique, model_id, elapsed, extra=None):
    """Store every extracted fact, copy paragraph labels, and return extraction statistics."""
    if len(records) != len(all_facts):
        raise ValueError("Expected one fact list per question")
    n_extracted = 0
    for r, facts in zip(records, all_facts):
        paragraphs = r.get("paragraphs") or []
        for f in facts:                                           # analysis only: never shown to the systems
            p = paragraphs[f["paragraph_idx"]] if f["paragraph_idx"] < len(paragraphs) else {}
            f["is_supporting"] = p.get("is_supporting") if isinstance(p, dict) else None
        n_extracted += len(facts)
        r["facts"] = facts
        r["facts_dropped"] = []
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
        "judging_enabled": False,
        "facts_dropped": 0,
        "facts_kept": n_extracted,
        "facts_per_question": round(n_extracted / max(1, len(records)), 1),
        "supporting_facts": len(sup),
        "distractor_facts": len(dis),
        "hop_coverage": f"{sum(hops)}/{len(hops)}",
        "extract_prompt": EXTRACT,
        "elapsed_sec": round(elapsed, 1),
    }
    entry.update(extra or {})
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="defaults to the input path with _facts.pkl appended")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=None, help="only the first n questions")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    args = parser.parse_args()

    records = pickle.load(open(args.input, "rb"))
    records = records[:args.n] if args.n else records
    if not records:
        parser.error("The selected input contains no questions")
    start = time.time()
    llm = load_llm(args.model, args.quant)
    try:
        all_facts, n_unique = extract_facts(llm, [r.get("paragraphs") for r in records], args.batch_size)
    finally:
        unload_llm(llm)
    entry = attach_facts(records, all_facts, n_unique, args.model, time.time() - start,
                         {"input_file": args.input})
    out_file = args.output or args.input.removesuffix(".pkl") + "_facts.pkl"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(records, open(out_file, "wb"))
    save_stats(records[0]["dataset"], {**entry, "output_file": out_file})
    print(f"{entry['questions']} questions, {entry['facts_kept']} facts kept, hop coverage {entry['hop_coverage']}. Saved to {out_file}")


if __name__ == "__main__":
    main()
