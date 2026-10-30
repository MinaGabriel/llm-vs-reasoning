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

NEG_Q_RE = re.compile(r"\b(not|never|neither|except|besides|other than)\b", re.I)

def is_verified_nars(r, question):
    ans = r.get("answer")
    rules = r.get("rules") or []
    votes = r.get("votes") or []
    repairs = r.get("auto_repairs") or {}
    if not ans or not rules or not r.get("from_question", False):
        return False
    if NEG_Q_RE.search(question):
        return False
    if None in votes or len([v for v in votes if v]) < 2 or len(set(votes)) > 1:
        return False
    if r.get("ona_top") and normalize(r["ona_top"]) != normalize(ans):
        return False
    if repairs:
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
        # Check if ans_var is bound in a condition with a constant while another variable is only in var-var conditions (Hop 1)
        const_cond_vars = {x for c in conds if any(not y.startswith("?") for y in (c[0], c[2])) for x in (c[0], c[2]) if x.startswith("?")}
        if ans_var in const_cond_vars and len(vars_all - const_cond_vars) >= 1:
            return False
    return True

for ds in ["MuSiQue", "HotpotQA", "2Wiki"]:
    run_dir = Path(f"runs/{ds}_2hop_100q_seed0_plan_votes3_merge_relations")
    if not run_dir.exists():
        continue
    nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
    verified = [i for i, r in nars.items() if is_verified_nars(r, r["question"])]
    v_acc = sum(nars[i]["judge"] for i in verified)
    print(f"=== {ds}: verified={len(verified)}/100, verified_judge_acc={v_acc}/{len(verified)} ({100*v_acc/max(1,len(verified)):.1f}%) ===")
    for i in verified:
        if not nars[i]["judge"]:
            print(f"   Wrong Q{i}: ans={nars[i]['answer']!r} gold={nars[i]['gold']!r} | Q={nars[i]['question']}")
            print(f"      rules={nars[i]['rules']}")
