import json, pickle, re, unicodedata
from collections import Counter

RUN_DIR = "runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations"
with open("runs/MuSiQue_test_entity.pkl", "rb") as f:
    questions = pickle.load(f)

nars = {r["index"]: r for r in map(json.loads, open(f"{RUN_DIR}/nars.jsonl"))}
direct = {r["index"]: r for r in map(json.loads, open(f"{RUN_DIR}/baselines.jsonl"))}
cot = {r["index"]: r for r in map(json.loads, open(f"{RUN_DIR}/cot.jsonl"))}
nars_llm_prev = {r["index"]: r for r in map(json.loads, open(f"{RUN_DIR}/nars_llm.jsonl"))}

MODELS = [
    "gemma-3-4b", "Llama-3.1-8B", "Mistral-Nemo-2407", "gemma-3-12b",
    "phi-4", "Mistral-Small-3.2-24B-2506", "gemma-3-27b", "gemma-4-31B",
    "Llama-3.3-70B", "Qwen3-Next-80B-A3B", "Qwen3-235B-A22B-2507", "DeepSeek-V3"
]

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

ABSTAIN_RE = re.compile(
    r"\b(not contain|no information|not mention|not state|not provide|cannot|can't|don't have|do not have|"
    r"unknown|none|no answer|not specified|does not say|unable to|misunderstanding|not include|no role|not enough)\b",
    re.I,
)
NEG_Q_RE = re.compile(r"\b(not|never|neither|except|besides|other than)\b", re.I)

def paragraphs(q):
    return [p for p in q["supporting"] if p["facts"]]

def in_para(ans, p_facts_str):
    na, np_ = normalize(ans), normalize(p_facts_str)
    return bool(na and na in np_)

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

def has_valid_nars_chain(r, q):
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
    ps = paragraphs(q)
    if len(ps) == 2:
        p0_str, p1_str = " ".join(ps[0]["facts"]), " ".join(ps[1]["facts"])
        consts = [x.replace("_", " ") for rule in rules for c in rule["conditions"] for x in (c[0], c[2]) if not x.startswith("?")]
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

# Print all cases where Direct and CoT disagree in Judge
for m in MODELS:
    d_only, c_only = [], []
    for i in sorted(nars):
        dj = direct[i]["answers"][m]["judge"]
        cj = cot[i]["answers"][m]["judge"]
        if dj != cj:
            if dj:
                d_only.append(i)
            else:
                c_only.append(i)
    print(f"{m:28s}: CoT>Direct on {len(c_only):2d} qs {c_only}, Direct>CoT on {len(d_only):2d} qs {d_only}")
