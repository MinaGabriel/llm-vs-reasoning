import json
import re
from pathlib import Path

for ds in ["MuSiQue", "HotpotQA", "2Wiki"]:
    run_dir = Path(f"runs/{ds}_2hop_100q_seed0_plan_votes3_merge_relations")
    if not run_dir.exists():
        continue
    nars = [json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip()]
    print(f"=== {ds} (total={len(nars)}, nars_judge={sum(r['judge'] for r in nars)}) ===")
    for r in nars:
        q = r["question"]
        rules = r.get("rules") or []
        ans = r.get("answer")
        votes = r.get("votes") or []
        fq = r.get("from_question", False)
        repairs = r.get("auto_repairs") or {}
        # Check flags
        has_neg = bool(re.search(r"\b(not|never|neither|except|besides|other than)\b", q.lower()))
        none_vote = (None in votes) or len([v for v in votes if v]) < 2
        disagree_vote = len(set(v for v in votes if v)) > 1
        over_hops = any(len(rule["conditions"]) >= 4 for rule in rules)
        # Check if answer variable is directly in a condition with a constant (1-hop answer)
        ans_is_hop1 = False
        for rule in rules:
            ans_var = rule.get("answer")
            conds = rule.get("conditions") or []
            # variables in conditions that have a constant
            const_vars = {x for c in conds for x in (c[0], c[2]) if x.startswith("?") and any(not y.startswith("?") for y in (c[0], c[2]))}
            all_vars = {x for c in conds for x in (c[0], c[2]) if x.startswith("?")}
            if len(all_vars) < 2 or (ans_var in const_vars and len(all_vars) >= 2 and len(conds) >= 2):
                # wait: if ans_var is directly linked to a constant in a 2-condition rule where the other condition has no constant, it returned hop 1!
                non_const_conds = [c for c in conds if all(y.startswith("?") for y in (c[0], c[2]))]
                if non_const_conds and all(ans_var not in (c[0], c[2]) for c in non_const_conds):
                    ans_is_hop1 = True
                if len(all_vars) < 2:
                    ans_is_hop1 = True

        HighConf = bool(ans) and fq and not has_neg and not none_vote and not disagree_vote and not repairs and not over_hops and not ans_is_hop1
        if HighConf and not r["judge"]:
            print(f"  FalsePositive HighConf Q{r['index']}: ans={ans!r} gold={r['gold']!r} | Q={q}")
            print(f"     rules={rules}")
    hc_all = [r for r in nars if bool(r.get("answer")) and r.get("from_question") and not bool(re.search(r"\b(not|never|neither|except|besides|other than)\b", r["question"].lower())) and not ((None in (r.get("votes") or [])) or len([v for v in (r.get("votes") or []) if v]) < 2) and not (len(set(v for v in (r.get("votes") or []) if v)) > 1) and not r.get("auto_repairs") and not any(len(rule["conditions"]) >= 4 for rule in (r.get("rules") or []))]
    print(f"  HighConf count: {len(hc_all)}, correct: {sum(r['judge'] for r in hc_all)} ({100*sum(r['judge'] for r in hc_all)/max(1,len(hc_all)):.1f}%)")
