import json
import pickle
import re
import unicodedata
from pathlib import Path

def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(a|an|the)\b", " ", text))

qs = pickle.load(open("runs/MuSiQue_test_entity.pkl", "rb"))
run_dir = Path("runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations")
nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}

for idx in [0, 14, 34, 42, 81, 87, 106, 111, 113, 121, 137, 160, 169, 199, 207]:
    q = qs[idx]
    r = nars[idx]
    ps = q["supporting"]
    titles = [p["title"] for p in ps]
    p_facts = [" ".join(p["facts"]) for p in ps]
    n_ans = r["answer"]
    # sample wrong/right LLM answers
    sample_b = [base[idx]["answers"][m]["answer"] for m in base[idx]["answers"] if normalize(base[idx]["answers"][m]["answer"]) != normalize(n_ans)][:3]
    print(f"Q{idx}: nars={n_ans!r} (J={r['judge']}) | sample_llm={sample_b}")
    print(f"   titles={titles}")
    for pi, pf in enumerate(p_facts):
        print(f"   P{pi} ({ps[pi]['title']}): nars_in_P={normalize(n_ans) in normalize(pf)} | facts={ps[pi]['facts'][:2]}")
