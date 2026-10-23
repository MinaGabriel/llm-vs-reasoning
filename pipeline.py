# pipeline.py: the whole dataset construction in one run.
#   1. Pool: the dataset's questions in the shared schema (from <dataset>.py: DATASET, load_pool).
#   2. Filter chain: each model answers closed-book (+ judge); only questions it answers WRONG continue.
#      Each model is loaded, used, unloaded, and deleted from the disk cache before the next one.
#   3. Final model, ONE load: filter pass, time-sensitive questions removed, first --final kept, then fact
#      extraction + P(True) on the paragraphs of those questions.
#   Every step is saved in runs/ (so a disconnected run resumes with --resume) and appended to <dataset>_stats.json.
#
# Usage: python pipeline.py --dataset MuSiQue --pool 500 --final 100
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
    "Qwen/Qwen2.5-1.5B-Instruct"
]
FINAL_MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
TIME_SENSITIVE = re.compile(r"\b(this season|current|currently|now|this year|latest|recent|recently|today)\b", re.I)


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MuSiQue", help="module name of the dataset file, e.g. MuSiQue")
    parser.add_argument("--pool", type=int, default=500, help="questions taken from the dataset")
    parser.add_argument("--final", type=int, default=100, help="questions in the final dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chain", nargs="*", default=CHAIN, help="filter models, run in this order")
    parser.add_argument("--final-model", default=FINAL_MODEL, help="last filter + fact extractor and judge")
    parser.add_argument("--facts-batch", type=int, default=16)
    parser.add_argument("--min-p-true", type=float, default=0.5)
    parser.add_argument("--keep-cache", action="store_true", help="do not delete models from the disk cache")
    parser.add_argument("--resume", action="store_true", help="reuse finished steps saved in runs/")
    args = parser.parse_args()

    ds = importlib.import_module(args.dataset)
    os.makedirs("runs", exist_ok=True)
    pool_file = f"runs/{ds.DATASET}_00_pool.pkl"

    # 1. Pool
    if args.resume and os.path.exists(pool_file):
        records = pickle.load(open(pool_file, "rb"))
    else:
        all_pool = ds.load_pool(args.seed)
        records = all_pool[:args.pool]
        save(records, pool_file)
        F.save_stats(ds.DATASET, {"role": "pool", "model": None, "eligible": len(all_pool), "pool": len(records),
                                  "seed": args.seed, "chain": args.chain, "final_model": args.final_model,
                                  "final_target": args.final, "answer_prompt": F.INSTRUCTION, "judge_prompt": F.JUDGE})
    print(f"Pool: {len(records)} questions")

    # 2. Filter chain
    for i, model_id in enumerate(args.chain, 1):
        step_file = f"runs/{ds.DATASET}_{i:02d}_{model_id.split('/')[-1]}.pkl"
        if args.resume and os.path.exists(step_file):
            records = pickle.load(open(step_file, "rb"))
            print(f"Skipping {model_id} (done): {len(records)} questions")
            continue
        start = time.time()
        llm = F.load_llm(model_id)
        logs = F.run_pass(llm, records, model_id, batch_size=F.batch_size_for(model_id))
        wrong = F.apply_pass(records, logs, model_id)
        entry = F.summarize(records, logs, model_id, "filter", time.time() - start, len(wrong),
                            {"output_file": step_file})
        F.unload_llm(llm, model_id, delete_cache=not args.keep_cache)
        records = wrong
        save(records, step_file)
        F.save_stats(ds.DATASET, entry)
        F.print_summary(entry)

    # 3. Final model: filter + facts in one load
    start = time.time()
    llm = F.load_llm(args.final_model)
    logs = F.run_pass(llm, records, args.final_model, batch_size=F.batch_size_for(args.final_model))
    wrong = F.apply_pass(records, logs, args.final_model)
    timely = [r for r in wrong if not TIME_SENSITIVE.search(r["question"])]
    final = timely[:args.final]
    entry = F.summarize(records, logs, args.final_model, "filter", time.time() - start, len(final),
                        {"time_sensitive_removed": len(wrong) - len(timely), "final_questions": len(final)})
    F.save_stats(ds.DATASET, entry)
    F.print_summary(entry)
    if len(final) < args.final:
        print(f"WARNING: only {len(final)} questions survived (target {args.final}); increase --pool")
    save(final, f"runs/{ds.DATASET}_final_questions.pkl")

    start = time.time()
    all_facts, n_unique = FA.extract_and_judge(llm, [r.get("paragraphs") for r in final], args.final_model,
                                               args.facts_batch)
    facts_entry = FA.attach_facts(final, all_facts, n_unique, args.final_model, args.min_p_true, time.time() - start)
    F.unload_llm(llm, args.final_model, delete_cache=not args.keep_cache)

    out_file = f"{ds.DATASET}_final.pkl"
    save(final, out_file)
    F.save_stats(ds.DATASET, {**facts_entry, "output_file": out_file})

    # Summary
    t = PrettyTable(["Model", "In", "Correct (string)", "Correct (judge)", "Kept"])
    t.align = "l"
    for e in json.load(open(f"{ds.DATASET}_stats.json")):
        if e.get("role") == "filter":
            t.add_row([e["model"].split("/")[-1], e["evaluated"], e["string_match_correct"], e["judge_correct"], e["kept"]])
    print(t)
    print(f"Facts: {facts_entry['facts_kept']} kept ({facts_entry['facts_per_question']} per question), "
          f"hop coverage {facts_entry['hop_coverage']}")
    print(f"Final dataset: {out_file} ({len(final)} questions); full log: {ds.DATASET}_stats.json")


if __name__ == "__main__":
    main()