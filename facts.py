# facts.py: fact extraction + fact judging (P(True)). Dataset-agnostic: works on any list of paragraphs,
#   so the same code serves MuSiQue now and retrieved Wikipedia passages later.
#
#   1. Fact extraction: the model lists the atomic facts stated in EACH paragraph (one paragraph at a time,
#      so every fact keeps its source). The question is NOT shown, so extraction is neutral: the model does
#      not know which facts matter.
#   2. Fact judging: the same model judges each fact against its paragraph, (A) True / (B) False;
#      P(True) = p(A) / (p(A) + p(B)) from the next-token probabilities. One judgment -> c = 0.5 in NARS.
#
# Usage:  python facts.py --input MuSiQue_filtered_Qwen2.5-7B.pkl --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --n 20
# Output: <input>_facts.pkl  (each record gets a "facts" list)
import argparse
import pickle

import torch
from tqdm import tqdm

from filter import chat, generate_batched, load_llm, normalize

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
    """
    Description: Accept plain strings or dicts, so retrieved text can be passed directly.
    Input:       ["Rumi was born in Balkh.", {"title": "Kabul", "text": "Kabul is the capital..."}]
    Output:      [{"title": "", "text": "Rumi was born in Balkh."}, {"title": "Kabul", "text": "Kabul is ..."}]
    """
    out = []
    for p in paragraphs:
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
    """
    Description: P(True) for each judging prompt from the first answer token: p(A) / (p(A) + p(B)).
                 Also returns the mass p(A) + p(B) (how much the model stayed on the two options).
    Output:      [{"p_true": 0.97, "mass": 0.99}, ...]
    """
    model, tok = llm
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]
    out = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        probs = torch.softmax(model(**enc).logits[:, -1, :].float(), dim=-1)    # left padding: last position
        for pa, pb in zip(probs[:, a_id].tolist(), probs[:, b_id].tolist()):
            out.append({"p_true": pa / (pa + pb) if pa + pb > 0 else 0.5, "mass": pa + pb})
    return out


def extract_and_judge(llm, paragraph_lists, model_id, batch_size=16):
    """
    Description: Facts with P(True) for many questions at once (all paragraphs batched together).
    Input:       [[paragraph, ...], ...]   one list of paragraphs per question (strings or dicts)
    Output:      [[{"text", "paragraph_idx", "paragraph_title", "p_true", "mass", "c", "model"}, ...], ...]
    """
    _, tok = llm
    jobs = [(q, j, p) for q, plist in enumerate(paragraph_lists) for j, p in enumerate(as_paragraphs(plist))]

    # 1. Fact extraction, one paragraph per prompt
    raw = generate_batched(llm, [chat(tok, EXTRACT.format(title=p["title"], text=p["text"])) for _, _, p in jobs],
                           max_new_tokens=384, batch_size=batch_size, desc="Extracting facts")
    facts = [(q, j, p, f) for (q, j, p), r in zip(jobs, raw) for f in parse_facts(r)]

    # 2. Fact judging, each fact against its own paragraph
    scores = p_true_batched(llm, [chat(tok, JUDGE_FACT.format(title=p["title"], text=p["text"], fact=f))
                                  for _, _, p, f in facts], batch_size=batch_size)

    out = [[] for _ in paragraph_lists]
    for (q, j, p, f), s in zip(facts, scores):
        out[q].append({"text": f, "paragraph_idx": j, "paragraph_title": p["title"],
                       "p_true": s["p_true"], "mass": s["mass"], "c": 0.5, "model": model_id})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="a dataset file in the shared schema")
    parser.add_argument("--model", required=True, help="fact extractor and judge, e.g. a 70B AWQ model")
    parser.add_argument("--n", type=int, default=None, help="only the first n questions (pilot)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="4bit")
    args = parser.parse_args()

    records = pickle.load(open(args.input, "rb"))
    records = records[:args.n] if args.n else records
    llm = load_llm(args.model, quant=args.quant)

    all_facts = extract_and_judge(llm, [r["paragraphs"] for r in records], args.model, args.batch_size)
    for r, facts in zip(records, all_facts):
        for f in facts:                                       # analysis only: never shown to the systems
            f["is_supporting"] = r["paragraphs"][f["paragraph_idx"]].get("is_supporting")
        r["facts"] = facts
        # Coverage check: does some fact contain each hop's gold answer?
        r["hop_coverage"] = [any(normalize(h["answer"]) in normalize(f["text"]) for f in facts)
                             for h in r.get("decomposition", []) if normalize(h["answer"])]

    out_file = args.input.removesuffix(".pkl") + "_facts.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(records, f)

    # Summary for the pilot check
    n_facts = [len(r["facts"]) for r in records]
    sup = [f for r in records for f in r["facts"] if f.get("is_supporting")]
    dis = [f for r in records for f in r["facts"] if f.get("is_supporting") is False]
    hops = [c for r in records for c in r["hop_coverage"]]
    print(f"\n{len(records)} questions, {sum(n_facts)} facts ({sum(n_facts) / len(records):.1f} per question)")
    print(f"Supporting-paragraph facts: {len(sup)}, mean P(True) {sum(f['p_true'] for f in sup) / max(1, len(sup)):.2f}")
    print(f"Distractor facts: {len(dis)}, mean P(True) {sum(f['p_true'] for f in dis) / max(1, len(dis)):.2f}")
    print(f"Hop answers found in the facts: {sum(hops)}/{len(hops)} ({100 * sum(hops) / max(1, len(hops)):.0f}%)")
    print(f"Facts judged unlikely (P(True) < 0.5): {sum(f['p_true'] < 0.5 for r in records for f in r['facts'])}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()