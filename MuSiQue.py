# MuSiQue.py: build MuSiQue candidates in the shared schema, screened by a (small) model, closed-book.
#   Keeps the questions the screening model answers WRONG. Screening uses string matching only
#   (a correct-but-differently-phrased answer just stays a candidate; the filter passes judge it later).
#   If --model none is passed, it exports the pool directly without screening.
#
# Usage:  python MuSiQue.py --model Qwen/Qwen2.5-0.5B-Instruct --n 1000
#         python MuSiQue.py --model none --n 1000
# Output: MuSiQue.pkl (candidates) and MuSiQue_stats.json (pass 1 = screening, if screened)
import argparse
import os
import pickle
import random
import time

from datasets import load_dataset

from filter import load_llm, print_summary, run_pass, save_stats, summarize

DATASET = "MuSiQue"


def to_schema(ex):
    """One MuSiQue example in the shared schema."""
    return {
        "id": ex["id"],
        "dataset": DATASET,
        "question": ex["question"],
        "answer": ex["answer"],
        "gold_answers": [ex["answer"]] + list(ex["answer_aliases"] or []),
        "decomposition": [
            {
                "question": s["question"],
                "answer": s["answer"],
                "support_idx": s["paragraph_support_idx"],
            }
            for s in ex["question_decomposition"]
        ],
        "paragraphs": [
            {
                "idx": p["idx"],
                "title": p["title"],
                "text": p["paragraph_text"],
                "is_supporting": p["is_supporting"],
            }
            for p in ex["paragraphs"]
        ],
        "source": "Salesforce/ContextualBench (MuSiQue, validation)",
        "failed_by": [],
        "filter_log": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="screening model, e.g. Qwen/Qwen2.5-0.5B-Instruct, or 'none' to skip screening",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="number of candidates to collect (questions it fails or pool size)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for old in (f"{DATASET}.pkl", f"{DATASET}_filtered.pkl", f"{DATASET}_stats.json"):
        if os.path.exists(old):
            os.remove(old)  # a new screening starts a new history

    # Answerable two-hop questions whose second step refers to the first answer (#1)
    data = load_dataset("Salesforce/ContextualBench", "MuSiQue", split="validation")
    pool = [
        to_schema(ex)
        for ex in data
        if ex["answerable"]
        and len(ex["question_decomposition"]) == 2
        and "#1" in ex["question_decomposition"][1]["question"]
    ]
    random.Random(args.seed).shuffle(pool)

    # Export directly if no screening model is requested
    if args.model.lower() == "none":
        kept = pool[: args.n]
        with open(f"{DATASET}.pkl", "wb") as f:
            pickle.dump(kept, f)
        print(f"Saved {len(kept)} questions to {DATASET}.pkl (no screening)")
        return

    print(f"{len(pool)} eligible two-hop questions; collecting {args.n} the screening model fails")

    llm = load_llm(args.model)
    start, kept, checked, all_records, all_logs = time.time(), [], 0, [], []
    chunk = max(2 * args.n, 256)
    while len(kept) < args.n and checked < len(pool):
        batch = pool[checked : checked + chunk]
        logs = run_pass(llm, batch, args.model, role="screening", judge=False)
        for r, l in zip(batch, logs):
            if len(kept) == args.n:
                break
            checked += 1
            all_records.append(r)
            all_logs.append(l)
            r["filter_log"] = [l]
            if not l["correct"]:
                r["failed_by"] = [args.model]
                kept.append(r)

    with open(f"{DATASET}.pkl", "wb") as f:
        pickle.dump(kept, f)
    entry = summarize(
        all_records,
        all_logs,
        args.model,
        "screening",
        time.time() - start,
        len(kept),
        {
            "pool_size": len(pool),
            "n_checked": checked,
            "seed": args.seed,
            "output_file": f"{DATASET}.pkl",
        },
    )
    stats = save_stats(DATASET, entry)
    print_summary(entry)
    print(f"Saved {len(kept)} candidates to {DATASET}.pkl; filtering history in {stats}")


if __name__ == "__main__":
    main()