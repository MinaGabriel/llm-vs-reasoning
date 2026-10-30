import json
import pickle
import re
import unicodedata
from pathlib import Path

def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(a|an|the)\b", " ", text))

def normalize_words(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower().replace("_", " ")

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

ABSTAIN_RE = re.compile(
    r"\b(not contain|no information|not mention|not state|not provide|cannot|can't|don't have|do not have|unknown|none|no answer|not specified|does not say|unable to|misunderstanding|not include|no role)\b",
    re.I,
)
NEG_Q_RE = re.compile(r"\b(not|never|neither|except|besides|other than)\b", re.I)

def is_verified_nars(r, q):
    question = q["question"]
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
    ps = [p for p in q["supporting"] if p["facts"]]
    if len(ps) == 2:
        p0_str = " ".join(ps[0]["facts"])
        p1_str = " ".join(ps[1]["facts"])
        consts = [x.replace("_", " ") for rule in rules for c in rule["conditions"] for x in (c[0], c[2]) if not x.startswith("?")]
        c_in_0 = any(in_para(c, p0_str) for c in consts)
        c_in_1 = any(in_para(c, p1_str) for c in consts)
        if not (c_in_0 and c_in_1):
            if in_para(ans, p0_str) and in_para(ans, p1_str):
                return False
            if c_in_0 and not c_in_1 and not in_para(ans, p1_str):
                return False
            if c_in_1 and not c_in_0 and not in_para(ans, p0_str):
                return False
    return True

def should_consult_nars(q, r, draft):
    n_ans = r.get("answer") or ""
    if not n_ans and not r.get("rules"):
        return False
    if ABSTAIN_RE.search(draft) or not draft.strip():
        return True
    verified = is_verified_nars(r, q)
    # Verbose full-sentence draft
    if len(draft.split()) > 6:
        return True
    # Missing unit / qualifier completion (e.g. "17.037" -> "17.037 square miles", "Bay" -> "Bay County")
    if n_ans and normalize(draft) and normalize(n_ans).startswith(normalize(draft)) and normalize(n_ans) != normalize(draft):
        if r.get("votes") and None not in r.get("votes") and len(set(r["votes"])) == 1:
            return True
    # Verified NARS disagrees with short draft
    if verified and not lenient(draft, [n_ans]):
        n_words = set(re.findall(r"[a-z0-9]+", normalize_words(n_ans)))
        d_words = set(re.findall(r"[a-z0-9]+", normalize_words(draft)))
        if " and " in draft.lower() and n_words and n_words <= d_words:
            return False
        return True
    return False

for ds in ["MuSiQue", "HotpotQA", "2Wiki"]:
    qs = pickle.load(open(f"runs/{ds}_test_entity.pkl", "rb"))
    run_dir = Path(f"runs/{ds}_2hop_100q_seed0_plan_votes3_merge_relations")
    nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
    base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}
    total_calls = 0
    q_triggered = set()
    for i, r in nars.items():
        q = qs[i]
        for m, b_rec in base[i]["answers"].items():
            if should_consult_nars(q, r, b_rec["answer"] or ""):
                total_calls += 1
                q_triggered.add(i)
    print(f"{ds}: {total_calls}/1200 model calls ({len(q_triggered)}/100 questions)")

print("\n--- Verbose drafts in MuSiQue ---")
qs = pickle.load(open("runs/MuSiQue_test_entity.pkl", "rb"))
nars = {r["index"]: r for r in map(json.loads, open("runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations/nars.jsonl"))}
base = {r["index"]: r for r in map(json.loads, open("runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations/baselines.jsonl"))}
nllm = {r["index"]: r for r in map(json.loads, open("runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations/nars_llm.jsonl"))}
for i, r in nars.items():
    v = is_verified_nars(r, qs[i])
    for m, b in base[i]["answers"].items():
        if m not in nllm[i]["answers"]:
            continue
        d = b["answer"] or ""
        if len(d.split()) > 6 and not ABSTAIN_RE.search(d):
            nl = nllm[i]["answers"][m]
            be, nle, bj, nlj, nla, na = b["exact"], nl["exact"], b["judge"], nl["judge"], nl["answer"], r["answer"]
            if be != nle or bj != nlj:
                print(f"Q{i} ({m}) v={v}: E {be}->{nle} J {bj}->{nlj} | d={d!r} -> nl={nla!r} (nars={na!r})")
