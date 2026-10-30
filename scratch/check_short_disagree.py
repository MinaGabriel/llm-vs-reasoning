import json
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

ABSTAIN_RE = re.compile(
    r"\b(not contain|no information|not mention|not state|not provide|cannot|can't|don't have|do not have|unknown|none|no answer|not specified|does not say|unable to|misunderstanding|not include|no role)\b",
    re.I,
)
NEG_Q_RE = re.compile(r"\b(not|never|neither|except|besides|other than)\b", re.I)

def is_verified_nars(r, question):
    ans = r.get("answer")
    rules = r.get("rules") or []
    votes = r.get("votes") or []
    repairs = r.get("auto_repairs") or {}
    if not ans or not rules:
        return False
    if NEG_Q_RE.search(question):
        return False
    if None in votes or len([v for v in votes if v]) < 2 or len(set(votes)) > 1:
        return False
    if r.get("ona_top") and normalize(r["ona_top"]) != normalize(ans):
        return False
    if repairs.get("pruned_unlinked") or repairs.get("inverse_relation"):
        return False
    for rule in rules:
        conds = rule.get("conditions") or []
        ans_var = rule.get("answer")
        if not (2 <= len(conds) <= 3):
            return False
        vars_all = {x for c in conds for x in (c[0], c[2]) if x.startswith("?")}
        consts_all = {x for c in conds for x in (c[0], c[2]) if not x.startswith("?")}
        if len(vars_all) < 2 or not consts_all:
            return False
        if ans_var in (conds[0][0], conds[0][2]) and ans_var not in (conds[-1][0], conds[-1][2]):
            return False
    return True

for ds in ["MuSiQue", "HotpotQA", "2Wiki"]:
    run_dir = Path(f"runs/{ds}_2hop_100q_seed0_plan_votes3_merge_relations")
    nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
    base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}
    models = list(next(iter(base.values()))["answers"].keys())
    print(f"================ {ds} ================")
    for i, r in sorted(nars.items()):
        if not is_verified_nars(r, r["question"]):
            continue
        n_ans = r["answer"]
        disagree_right_nars = []
        disagree_wrong_nars = []
        for m in models:
            b_rec = base[i]["answers"][m]
            b_ans = b_rec["answer"] or ""
            if ABSTAIN_RE.search(b_ans) or not b_ans.strip():
                continue
            if not lenient(b_ans, [n_ans]):
                if r["judge"] and not b_rec["judge"]:
                    disagree_right_nars.append((m, b_ans))
                elif b_rec["judge"] and not r["judge"]:
                    disagree_wrong_nars.append((m, b_ans))
        if disagree_right_nars or disagree_wrong_nars:
            print(f"Q{i} (fq={r['from_question']}, nars_J={r['judge']}): nars={n_ans!r} | gold={r['gold']!r} | +{len(disagree_right_nars)} -{len(disagree_wrong_nars)}")
            print(f"   Q: {r['question']}")
            print(f"   rules: {r['rules']}")
            if disagree_right_nars:
                print(f"   NARS RIGHT, LLM WRONG: {disagree_right_nars[:4]}")
            if disagree_wrong_nars:
                print(f"   NARS WRONG, LLM RIGHT: {disagree_wrong_nars[:4]}")
