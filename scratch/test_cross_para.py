import json
import pickle
import re
import unicodedata
from pathlib import Path

def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(a|an|the)\b", " ", text))

def in_para(ans, p_facts_str):
    na = normalize(ans)
    np = normalize(p_facts_str)
    return bool(na and na in np)

for ds in ["MuSiQue", "HotpotQA", "2Wiki"]:
    qs = pickle.load(open(f"runs/{ds}_test_entity.pkl", "rb"))
    run_dir = Path(f"runs/{ds}_2hop_100q_seed0_plan_votes3_merge_relations")
    nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
    same_para_nars = []
    bridge_nars = []
    for i, r in sorted(nars.items()):
        q = qs[i]
        ps = [p for p in q["supporting"] if p["facts"]]
        if len(ps) != 2 or not r.get("answer") or not r.get("rules"):
            continue
        p0_str = " ".join(ps[0]["facts"])
        p1_str = " ".join(ps[1]["facts"])
        n_ans = r["answer"]
        # Check if n_ans is in BOTH paragraphs (bridge entity)
        if in_para(n_ans, p0_str) and in_para(n_ans, p1_str):
            bridge_nars.append((i, r["judge"], n_ans, r["gold"]))
        # Check if all rule constants are in only P0 (not P1) and n_ans is also only in P0 (not P1), or vice versa
        consts = [x.replace("_", " ") for rule in r["rules"] for c in rule["conditions"] for x in (c[0], c[2]) if not x.startswith("?")]
        if consts:
            c_in_0 = any(in_para(c, p0_str) for c in consts)
            c_in_1 = any(in_para(c, p1_str) for c in consts)
            if c_in_0 and not c_in_1 and not in_para(n_ans, p1_str):
                same_para_nars.append((i, r["judge"], n_ans, r["gold"]))
            elif c_in_1 and not c_in_0 and not in_para(n_ans, p0_str):
                same_para_nars.append((i, r["judge"], n_ans, r["gold"]))
    print(f"=== {ds} ===")
    print(f"  NARS answer in BOTH paragraphs (Hop-1 bridge): {bridge_nars}")
    print(f"  NARS stayed in single paragraph (never crossed to P_other): {same_para_nars}")
