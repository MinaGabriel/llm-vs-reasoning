import json
import pickle
import re
import unicodedata
from pathlib import Path

def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(a|an|the)\b", " ", text))

def lenient(answer, golds):
    a = normalize(answer or "")
    for g in map(normalize, golds):
        short, long_ = sorted((a, g), key=len)
        if short and short in long_ and 2 * len(short) >= len(long_):
            return True
    return False

def in_para(ans, p_facts_str):
    na = normalize(ans)
    np = normalize(p_facts_str)
    return bool(na and na in np)

for ds in ["MuSiQue", "HotpotQA", "2Wiki"]:
    qs = pickle.load(open(f"runs/{ds}_test_entity.pkl", "rb"))
    run_dir = Path(f"runs/{ds}_2hop_100q_seed0_plan_votes3_merge_relations")
    nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
    base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}
    print(f"================ {ds} ================")
    # Check when LLM answer is in BOTH paragraphs (bridge entity) while NARS answer is in exactly ONE paragraph!
    bridge_helped = 0
    bridge_hurt = 0
    for i, r in sorted(nars.items()):
        q = qs[i]
        ps = [p for p in q["supporting"] if p["facts"]]
        if len(ps) != 2 or not r.get("answer"):
            continue
        p0_str = " ".join(ps[0]["facts"])
        p1_str = " ".join(ps[1]["facts"])
        n_ans = r["answer"]
        n_in_both = in_para(n_ans, p0_str) and in_para(n_ans, p1_str)
        n_in_one = (in_para(n_ans, p0_str) ^ in_para(n_ans, p1_str))
        for m, b_rec in base[i]["answers"].items():
            b_ans = b_rec["answer"] or ""
            b_in_both = in_para(b_ans, p0_str) and in_para(b_ans, p1_str)
            if b_in_both and n_in_one:
                if r["judge"] and not b_rec["judge"]:
                    bridge_helped += 1
                elif b_rec["judge"] and not r["judge"]:
                    bridge_hurt += 1
                    print(f"  Bridge hurt Q{i} ({m}): b={b_ans!r} -> n={n_ans!r} (gold={r['gold']!r})")
    print(f"  LLM in both paragraphs (bridge) vs NARS in 1 paragraph: +{bridge_helped} helped, -{bridge_hurt} hurt!")
