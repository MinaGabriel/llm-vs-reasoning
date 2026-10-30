"""Re-judge the pilot's non-exact answers with a stricter, consistent judge, and rebuild the tables.

The pilot's judge (Qwen2.5-72B, short prompt) was inconsistent on long answers: it accepted "The Washington Post" for
"Washington, D.C." and rejected "FIFA (Fédération Internationale de Football Association)". Every answer that is not an
exact match is graded again, with the question, all gold answers and explicit rules; exact matches stay right.

    python rejudge_pilot.py                  # grades once (cached in judge2.jsonl), then prints the tables
"""
import json
import os
import pickle
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from glob import glob

from openai import OpenAI

from keys import deepinfra_key

RUN_DIR = "runs/pilot_dev_30q_seed0"
JUDGE = "Qwen/Qwen3-235B-A22B-Instruct-2507"     # not one of the compared systems
CACHE = f"{RUN_DIR}/judge2.jsonl"
SYSTEMS = ["8B direct", "NARS + DeepSeek-V3", "NARS + 8B + LoRA", "NARS + 8B"]
DATASETS = ["MuSiQue", "HotpotQA", "2Wiki"]

PROMPT = """You grade one answer to a question against the gold answer. Be strict and consistent.

Question: {question}
Gold answer: {gold}
Model answer: {answer}

YES if the model answer gives the same entity, date, number or place as the gold answer, even with extra words around
it, a fuller or shorter name, or an abbreviation:
  "IOC president Thomas Bach" for "Thomas Bach"; "UN (United Nations)" for "United Nations";
  "Marie Curie won it" for "Marie Curie"; "the 1990s" for "1990s"
NO if it gives anything else, even when it shares words with the gold answer or is related to it:
  a different thing with a similar name ("The Boston Globe" for "Boston"); a part or a larger whole ("Oregon" for
  "Portland"; "Portland" for "Oregon"); something about the answer instead of it ("Einstein's theory" for "Einstein");
  several candidates ("Paris or Lyon" for "Paris"); a different date or number; no answer ("not stated").
Reply with only yes or no."""


def load():
    rows = []
    for path in sorted(glob(f"{RUN_DIR}/*.jsonl")):
        parts = os.path.basename(path)[:-6].split("__")
        if len(parts) < 2:
            continue
        dataset, k = parts[0], int(parts[1][1:]) if len(parts) == 3 else 0
        system = {"8B_direct": "8B direct", "NARS_DeepSeek_V3": "NARS + DeepSeek-V3",
                  "NARS_8B_LoRA": "NARS + 8B + LoRA", "NARS_8B": "NARS + 8B"}[parts[-1]]
        for r in map(json.loads, open(path)):
            if not r.get("error"):
                rows.append({**r, "dataset": dataset, "k": k, "system": system})
    return rows


def golds(q):
    return list(dict.fromkeys([q["answer"], *(q.get("gold_answers") or [])]))


def grade_all(rows):
    """judge2 for every answered, non-exact record (cached by dataset, index, answer)."""
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE):
            c = json.loads(line)
            cache[(c["dataset"], c["index"], c["answer"])] = c["yes"]
    questions = {d: pickle.load(open(f"runs/{d}_dev_entity.pkl", "rb")) for d in DATASETS}
    todo = {(r["dataset"], r["index"], r["answer"]) for r in rows if r["answer"] and not r["exact"]} - cache.keys()
    client = OpenAI(api_key=deepinfra_key(), base_url="https://api.deepinfra.com/v1/openai", max_retries=3, timeout=60)

    def ask(key):
        dataset, index, answer = key
        q = questions[dataset][index]
        reply = client.chat.completions.create(model=JUDGE, temperature=0, max_tokens=3, messages=[{
            "role": "user", "content": PROMPT.format(question=q["question"], gold=" / ".join(golds(q)), answer=answer)}])
        assert reply.model == JUDGE, f"judge served as {reply.model}"
        return key, (reply.choices[0].message.content or "").strip().lower().startswith("yes")

    with ThreadPoolExecutor(16) as pool, open(CACHE, "a") as f:
        for (dataset, index, answer), yes in pool.map(ask, sorted(todo)):
            cache[(dataset, index, answer)] = yes
            f.write(json.dumps({"dataset": dataset, "index": index, "answer": answer, "yes": yes}, ensure_ascii=False) + "\n")
    print(f"graded {len(todo)} new answers ({len(cache)} cached in total)")
    for r in rows:
        r["judge2"] = bool(r["exact"] or (r["answer"] and cache[(r["dataset"], r["index"], r["answer"])]))


def table(rows, metric):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["system"], r["k"]][r["dataset"]].append(r[metric])
    print(f"\n{metric.upper()} · MuSiQue / HotpotQA / 2Wiki → all (30 questions per dataset)")
    print(f"{'System':<20}" + "".join(f"{'k = ' + str(k):<30}" for k in (0, 4, 8)) + "change 0→8")
    for s in SYSTEMS:
        cells, pct = [], {}
        for k in (0, 4, 8):
            d = by[s, k]
            if not d:
                cells.append("-")
                continue
            right, n = sum(sum(v) for v in d.values()), sum(len(v) for v in d.values())
            pct[k] = 100 * right / n
            cells.append(" / ".join(str(sum(d[x])) for x in DATASETS) + f" → {right}/{n} = {pct[k]:.0f}%")
        change = f"{pct[8] - pct[0]:+.0f} points" if 0 in pct and 8 in pct else "-"
        print(f"{s:<20}" + "".join(f"{c:<30}" for c in cells) + change)


def drop_difference(rows, metric, a="NARS + 8B + LoRA", b="8B direct", draws=10000):
    """How much less A loses than B from k = 0 to k = 8, per question, with a bootstrap 95% interval over questions."""
    val = {(r["system"], r["k"], r["dataset"], r["index"]): r[metric] for r in rows}
    qs = sorted({(d, i) for (s, k, d, i) in val if s == a and k == 8} & {(d, i) for (s, k, d, i) in val if s == b and k == 8})
    qs = [q for q in qs if all((s, k, *q) in val for s in (a, b) for k in (0, 8))]
    diff = [(val[a, 8, *q] - val[a, 0, *q]) - (val[b, 8, *q] - val[b, 0, *q]) for q in qs]
    mean = sum(diff) / len(diff)
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diff) for _ in diff) / len(diff) for _ in range(draws))
    lo, hi = boots[int(0.025 * draws)], boots[int(0.975 * draws)]
    print(f"   {metric:<7} {a} loses {100 * mean:+.0f} points less than {b} (k 0→8, {len(qs)} questions); "
          f"95% CI [{100 * lo:+.0f}, {100 * hi:+.0f}]")


if __name__ == "__main__":
    rows = load()
    grade_all(rows)
    changed = [r for r in rows if r["judge"] != r["judge2"]]
    print(f"{len(changed)} of {len(rows)} verdicts changed:")
    for r in sorted(changed, key=lambda r: (r["system"], r["k"])):
        print(f"   {'✗→✓' if r['judge2'] else '✓→✗'}  k={r['k']} {r['system']:<18} {str(r['answer'])[:45]!r:<48} gold {r['gold'][:30]!r}")
    for metric in ("judge", "judge2", "exact"):
        table(rows, metric)
    print("\nINTERACTION (the robustness claim): change k=0→8 of A minus change of B, same questions")
    for metric in ("judge2", "exact"):
        drop_difference(rows, metric)
        drop_difference(rows, metric, a="NARS + DeepSeek-V3")
