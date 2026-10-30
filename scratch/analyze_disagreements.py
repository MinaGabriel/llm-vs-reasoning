import json
import re
import unicodedata
from pathlib import Path

def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(a|an|the)\b", " ", text))

ABSTAIN_RE = re.compile(
    r"\b(not contain|no information|not mention|not state|not provide|cannot|can't|don't have|do not have|unknown|none|no answer|not specified|does not say|unable to|misunderstanding)\b",
    re.I,
)

for ds in ["MuSiQue", "HotpotQA", "2Wiki"]:
    run_dir = Path(f"runs/{ds}_2hop_100q_seed0_plan_votes3_merge_relations")
    if not run_dir.exists():
        continue
    nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
    base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}
    print(f"================ {ds} ================")
    models = list(next(iter(base.values()))["answers"].keys())
    for m in models:
        helped_abstain = 0
        hurt_abstain = 0
        helped_verbose = 0
        hurt_verbose = 0
        for i, r in nars.items():
            b_rec = base[i]["answers"][m]
            b_ans = b_rec["answer"] or ""
            n_ans = r["answer"] or ""
            if not n_ans:
                continue
            is_abstain = bool(ABSTAIN_RE.search(b_ans)) or not b_ans.strip()
            # verbose: b_ans contains n_ans (normalized) and len(b_ans) > len(n_ans) * 2
            nb, nn = normalize(b_ans), normalize(n_ans)
            is_verbose_match = bool(nn and nn in nb and len(nb) > 2 * len(nn))
            if is_abstain:
                if r["judge"] and not b_rec["judge"]:
                    helped_abstain += 1
                elif b_rec["judge"] and not r["judge"]:
                    hurt_abstain += 1
            elif is_verbose_match:
                if r["exact"] and not b_rec["exact"]:
                    helped_verbose += 1
                elif b_rec["judge"] and not r["judge"]:
                    hurt_verbose += 1
        print(f"{m:<28} | abstain: +{helped_abstain} -{hurt_abstain} | verbose_extract: +{helped_verbose}E -{hurt_verbose}J")
