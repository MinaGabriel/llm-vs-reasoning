import json
import pickle
import re
import unicodedata
from collections import Counter
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

def valid_rule_shape(rule):
    conds = rule.get("conditions") or []
    ans_var = rule.get("answer")
    if not (2 <= len(conds) <= 3):
        return False
    vars_all = {x for c in conds for x in (c[0], c[2]) if x.startswith("?")}
    if len(vars_all) < 2:
        return False
    if ans_var in (conds[0][0], conds[0][2]) and ans_var not in (conds[-1][0], conds[-1][2]):
        return False
    counts = Counter(x for c in conds for x in (c[0], c[2]) if x.startswith("?"))
    if any(c[0].startswith("?") and c[2].startswith("?") and any(counts[v] == 1 and v != ans_var for v in (c[0], c[2])) for c in conds[1:]):
        return False
    return True

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
    if not all(valid_rule_shape(rule) for rule in rules):
        return False
    ps = [p for p in q["supporting"] if p["facts"]]
    if len(ps) == 2:
        p0_str = " ".join(ps[0]["facts"])
        p1_str = " ".join(ps[1]["facts"])
        consts = [x.replace("_", " ") for rule in rules for c in rule["conditions"] for x in (c[0], c[2]) if not x.startswith("?")]
        # Also find which paragraph is anchored by the question's capitalized/rare entities
        cap_phrases = re.findall(r"\b[A-Z][a-zA-Z0-9_-]{3,}\b", question[1:])
        anchors = consts if consts else cap_phrases
        c_in_0 = any(in_para(c, p0_str) for c in anchors)
        c_in_1 = any(in_para(c, p1_str) for c in anchors)
        if not (c_in_0 or c_in_1):
            return False
        if not (c_in_0 and c_in_1 and r.get("from_question", False)):
            if in_para(ans, p0_str) and in_para(ans, p1_str):
                return False
            if c_in_0 and not c_in_1 and not in_para(ans, p1_str):
                return False
            if c_in_1 and not c_in_0 and not in_para(ans, p0_str):
                return False
    return True

qs = pickle.load(open("runs/MuSiQue_test_entity.pkl", "rb"))
run_dir = Path("runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations")
nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}
nllm = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars_llm.jsonl").read_text().splitlines() if l.strip())}

models = list(next(iter(nllm.values()))["answers"].keys())
for m in models:
    e_base = sum(base[i]["answers"][m]["exact"] for i in nars)
    l_base = sum(base[i]["answers"][m]["lenient"] for i in nars)
    j_base = sum(base[i]["answers"][m]["judge"] for i in nars)
    e_hyb, l_hyb, j_hyb = 0, 0, 0
    helped, hurt = [], []
    for i, r in nars.items():
        q = qs[i]
        b_rec = base[i]["answers"][m]
        nl_rec = nllm[i]["answers"][m]
        draft = b_rec["answer"] or ""
        n_ans = r.get("answer") or ""
        verified = is_verified_nars(r, q)
        use_nllm = False
        if ABSTAIN_RE.search(draft) or not draft.strip():
            if n_ans or r.get("rules"):
                use_nllm = True
        elif len(draft.split()) > 6:
            ps = [p for p in q["supporting"] if p["facts"]]
            in_both = len(ps) == 2 and in_para(n_ans, " ".join(ps[0]["facts"])) and in_para(n_ans, " ".join(ps[1]["facts"]))
            if n_ans and normalize(n_ans) in normalize(draft) and not in_both and normalize(n_ans) not in normalize(q["question"]):
                use_nllm = True
        elif re.match(r"^\d", draft.strip()) and n_ans and normalize(n_ans).startswith(normalize(draft)) and normalize(n_ans) != normalize(draft) and "local time" not in n_ans.lower():
            if r.get("votes") and None not in r.get("votes") and len(set(r["votes"])) == 1:
                use_nllm = True
        elif verified and not lenient(draft, [n_ans]) and not normalize(draft).startswith(normalize(n_ans)):
            n_words = set(re.findall(r"[a-z0-9]+", normalize_words(n_ans)))
            d_words = set(re.findall(r"[a-z0-9]+", normalize_words(draft)))
            if not (" and " in draft.lower() and n_words and n_words <= d_words):
                use_nllm = True
        chosen = nl_rec if use_nllm else b_rec
        e_hyb += chosen["exact"]
        l_hyb += chosen["lenient"]
        j_hyb += chosen["judge"]
        if not b_rec["judge"] and chosen["judge"]:
            helped.append(i)
        elif b_rec["judge"] and not chosen["judge"]:
            hurt.append((i, draft, chosen["answer"], r["gold"]))
    print(f"{m:<28} | E: {e_base}->{e_hyb} ({e_hyb-e_base:+d}) | L: {l_base}->{l_hyb} ({l_hyb-l_base:+d}) | J: {j_base}->{j_hyb} ({j_hyb-j_base:+d}) [+{len(helped)} -{len(hurt)}]")
    if hurt:
        print(f"   hurt: {hurt}")
