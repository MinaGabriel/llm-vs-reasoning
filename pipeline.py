# pipeline.py: the whole dataset construction in one run (one A100).
#   1. Pool: questions in the shared schema from <dataset>.py (DATASET, load_pool).
#   2. Filter chain: each model answers closed-book; questions it answers correctly (string match) are dropped.
#      The evidence model also answers with ONLY the gold paragraphs (evidence check).
#      Each model is loaded, used, unloaded, and deleted from the disk cache before the next one.
#   3. Final model, ONE load: answers closed-book, then JUDGES every saved answer of every model (one judge for all),
#      keeps questions all models got wrong AND the evidence check got right, drops time-sensitive questions and
#      duplicates, takes --final, and extracts facts without a separate fact-judging pass.
#   Every step is saved in runs/ (resume with --resume) and appended to <dataset>_stats.json.
#
# Usage:  python pipeline.py --dataset MuSiQue --pool 800 --final 100
# Output: <dataset>_final.pkl and <dataset>_stats.json
import argparse
import importlib
import json
import os
import pickle
import re
import time

from prettytable import PrettyTable

import facts as FA
import filter as F

CHAIN = [
    "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]
EVIDENCE_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
FINAL_MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
TIME_SENSITIVE = re.compile(r"\b(this season|current|currently|now|this year|latest|recent|recently|today)\b", re.I)


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MuSiQue", help="module name of the dataset file, e.g. MuSiQue")
    parser.add_argument("--pool", type=int, default=800)
    parser.add_argument("--final", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chain", nargs="*", default=CHAIN)
    parser.add_argument("--evidence-model", default=EVIDENCE_MODEL, help="must be in --chain")
    parser.add_argument("--final-model", default=FINAL_MODEL, help="last filter, judge of all answers, fact extractor")
    parser.add_argument("--facts-batch", type=int, default=16)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ds = importlib.import_module(args.dataset)
    name = ds.DATASET
    os.makedirs("runs", exist_ok=True)
    pool_file = f"runs/{name}_00_pool.pkl"

    # 1. Pool
    if args.resume and os.path.exists(pool_file):
        records = pickle.load(open(pool_file, "rb"))
    else:
        all_pool = ds.load_pool(args.seed)
        records = all_pool[:args.pool]
        save(records, pool_file)
        F.save_stats(name, {"role": "pool", "model": None, "eligible": len(all_pool), "pool": len(records),
                            "seed": args.seed, "chain": args.chain, "evidence_model": args.evidence_model,
                            "final_model": args.final_model, "final_target": args.final,
                            "answer_prompt": F.INSTRUCTION, "evidence_prompt": F.EVIDENCE, "judge_prompt": F.JUDGE})
    print(f"Pool: {len(records)} questions")

    # 2. Filter chain (answer + string match; evidence check in the evidence model's pass)
    for i, model_id in enumerate(args.chain, 1):
        step_file = f"runs/{name}_{i:02d}_{model_id.split('/')[-1]}.pkl"
        if args.resume and os.path.exists(step_file):
            records = pickle.load(open(step_file, "rb"))
            print(f"Skipping {model_id} (done): {len(records)} questions")
            continue
        start = time.time()
        llm = F.load_llm(model_id)
        bs = F.batch_size_for(model_id)
        extra = {"output_file": step_file}
        if model_id == args.evidence_model:
            n_ev, n_ev_sm = F.evidence_check(llm, records, model_id, bs)
            extra.update({"evidence_checked": n_ev, "evidence_correct_string_match": n_ev_sm})
        logs = F.run_pass(llm, records, model_id, batch_size=bs)
        kept = F.apply_pass(records, logs, model_id)
        entry = F.summarize(records, logs, model_id, "filter", time.time() - start, len(kept), extra)
        F.unload_llm(llm, model_id, delete_cache=not args.keep_cache)
        records = kept
        save(records, step_file)
        F.save_stats(name, entry)
        F.print_summary(entry)

    # 3. Final model: answer, judge everything, select, extract facts (one load)
    start = time.time()
    llm = F.load_llm(args.final_model)
    bs = F.batch_size_for(args.final_model)
    logs = F.run_pass(llm, records, args.final_model, batch_size=bs)
    entry = F.summarize(records, logs, args.final_model, "filter", time.time() - start,
                        sum(not l["string_match"] for l in logs))
    records = F.apply_pass(records, logs, args.final_model)
    F.save_stats(name, entry)
    F.print_summary(entry)

    start = time.time()
    judge_stats = F.judge_all(llm, records, args.final_model, bs)
    unknown = [r for r in records if not any(l["correct"] for l in r["filter_log"])]
    evidenced = [r for r in unknown if r.get("evidence_check") is None or r["evidence_check"]["correct"]]
    timely = [r for r in evidenced if not TIME_SENSITIVE.search(r["question"])]
    unique = F.dedupe(timely)
    final = unique[:args.final]
    for r in final:
        r["failed_by"] = [l["model"] for l in r["filter_log"]]
    F.save_stats(name, {
        "role": "judge_and_select", "model": args.final_model,
        "candidates": len(records),
        "known_after_judge": len(records) - len(unknown),
        "evidence_insufficient": len(unknown) - len(evidenced),
        "time_sensitive_removed": len(evidenced) - len(timely),
        "duplicates_removed": len(timely) - len(unique),
        "available": len(unique),
        "final_questions": len(final),
        "judge_per_model": {m: {"judged": s["judged"], "flipped": s["flipped"]} for m, s in judge_stats.items()},
        "judge_flipped_examples": {m: s["examples"] for m, s in judge_stats.items()},
        "elapsed_sec": round(time.time() - start, 1),
    })
    print(f"Judge + selection: {len(records)} candidates -> {len(records) - len(unknown)} known after judging, "
          f"{len(unknown) - len(evidenced)} evidence-insufficient, {len(evidenced) - len(timely)} time-sensitive, "
          f"{len(timely) - len(unique)} duplicates -> {len(unique)} available, {len(final)} kept")
    if len(final) < args.final:
        print(f"WARNING: only {len(final)} questions (target {args.final}); increase --pool")
    save(final, f"runs/{name}_final_questions.pkl")

    start = time.time()
    all_facts, n_unique = FA.extract_facts(llm, [r.get("paragraphs") for r in final], args.facts_batch)
    facts_entry = FA.attach_facts(final, all_facts, n_unique, args.final_model, time.time() - start)
    F.unload_llm(llm, args.final_model, delete_cache=not args.keep_cache)
    out_file = f"{name}_final.pkl"
    save(final, out_file)
    F.save_stats(name, {**facts_entry, "output_file": out_file})

    # Summary
    t = PrettyTable(["Model", "Judged", "Flipped to correct"])
    t.align = "l"
    for m, s in judge_stats.items():
        t.add_row([m.split("/")[-1], s["judged"], s["flipped"]])
    print(t)
    print(f"Facts: {facts_entry['facts_kept']} kept ({facts_entry['facts_per_question']} per question), "
          f"hop coverage {facts_entry['hop_coverage']}")
    print(f"Final dataset: {out_file} ({len(final)} questions); full log: {name}_stats.json")


if __name__ == "__main__":
    main()
