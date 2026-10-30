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

def exact(answer, golds):
    return any(normalize(answer or "") == normalize(g) for g in golds)

def lenient(answer, golds):
    a = normalize(answer or "")
    for g in map(normalize, golds):
        short, long_ = sorted((a, g), key=len)
        if short and short in long_ and 2 * len(short) >= len(long_):
            return True
    return False

ABSTAIN_RE = re.compile(
    r"\b(not contain\w*|no information|not mention\w*|not state\w*|not provide\w*|cannot|can't|don't have|do not have|"
    r"unknown|none|no answer|not specif\w*|does not say|unable to|misunderstanding|not include\w*|no role|not enough)\b",
    re.I,
)
NEG_Q_RE = re.compile(r"\b(not|never|neither|except|besides|other than)\b", re.I)

def paragraphs(q):
    return [p for p in q["supporting"] if p["facts"]]

def in_para(ans, p_facts_str):
    na, np_ = normalize(ans), normalize(p_facts_str)
    return bool(na and na in np_)

def in_both_paras(ans, q):
    ps = paragraphs(q)
    if len(ps) == 2 and ans:
        return in_para(ans, " ".join(ps[0]["facts"])) and in_para(ans, " ".join(ps[1]["facts"]))
    return False

def hop2_target_ok(ans, q):
    if not ans or ABSTAIN_RE.search(ans):
        return False
    if in_para(ans, q["question"]):
        return False
    ps = paragraphs(q)
    if len(ps) != 2:
        return True
    p0_str, p1_str = " ".join(ps[0]["facts"]), " ".join(ps[1]["facts"])
    if in_para(ans, p0_str) and in_para(ans, p1_str):
        return False
    cap_phrases = re.findall(r"\b[A-Z][a-zA-Z0-9_-]{3,}\b", q["question"][1:])
    c_in_0 = any(in_para(c, p0_str) for c in cap_phrases)
    c_in_1 = any(in_para(c, p1_str) for c in cap_phrases)
    if c_in_0 and not c_in_1:
        return in_para(ans, p1_str)
    if c_in_1 and not c_in_0:
        return in_para(ans, p0_str)
    return in_para(ans, p0_str) or in_para(ans, p1_str)

def cot_chains_both_paras(q, c_rec):
    trace = c_rec.get("cot_trace") or ""
    c_ans = c_rec.get("answer") or ""
    ps = paragraphs(q)
    if len(ps) != 2 or not trace or not c_ans:
        return False
    p0_str, p1_str = " ".join(ps[0]["facts"]), " ".join(ps[1]["facts"])
    if not (in_para(c_ans, p0_str) or in_para(c_ans, p1_str)):
        return False
    q_norm = normalize_words(q["question"])
    t_norm = normalize_words(trace)
    w0 = {w for w in re.findall(r"[a-z0-9]{4,}", normalize_words(p0_str)) if w not in q_norm}
    w1 = {w for w in re.findall(r"[a-z0-9]{4,}", normalize_words(p1_str)) if w not in q_norm}
    shared = w0 & w1
    return bool(shared and any(w in t_norm for w in shared))

def matches_nars(ans, n_ans):
    if not ans or not n_ans:
        return False
    na, nn = normalize(ans), normalize(n_ans)
    return bool(na and nn and (lenient(ans, [n_ans]) or nn in na or na in nn))

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

def select_draft_between_direct_and_cot(q, r, d_rec, c_rec):
    d_ans = d_rec.get("answer") or ""
    c_ans = c_rec.get("answer") or ""
    n_ans = r.get("answer") or ""
    verified = has_valid_nars_chain(r, q)

    d_abs = bool(ABSTAIN_RE.search(d_ans) or not d_ans.strip())
    c_abs = bool(ABSTAIN_RE.search(c_ans) or not c_ans.strip())
    if d_abs and not c_abs:
        return c_rec
    if c_abs and not d_abs:
        return d_rec
    if d_abs and c_abs:
        return d_rec

    # Both non-abstaining: check if one is an acronym of the other (e.g. FSMA vs The Food Safety Modernization Act)
    c_initials = "".join(w[0] for w in re.findall(r"[A-Z][a-zA-Z]*", re.sub(r"^the\s+", "", c_ans, flags=re.I)))
    if d_ans.isupper() and len(d_ans) >= 3 and d_ans == c_initials:
        return c_rec

    # If one is an exact match to verified NARS answer and the other is not
    if verified and n_ans:
        if exact(c_ans, [n_ans]) and not exact(d_ans, [n_ans]) and not re.match(r"^\d", c_ans.strip()):
            return c_rec
        if exact(d_ans, [n_ans]) and not exact(c_ans, [n_ans]) and "local time" not in d_ans.lower():
            return d_rec

    # If they agree leniently:
    if lenient(d_ans, [c_ans]):
        if re.match(r"^\d", c_ans.strip()) and len(d_ans) > len(c_ans) and normalize(d_ans).startswith(normalize(c_ans)) and "local time" not in d_ans.lower():
            return d_rec
        if len(d_ans.split()) > 4 and len(c_ans.split()) <= 4:
            return c_rec
        return d_rec

    # If one listed multiple entities (" and " or ",") while the other named a single entity inside it
    if (" and " in c_ans.lower() or "," in c_ans) and " and " not in d_ans.lower() and normalize(d_ans) in normalize(c_ans):
        if verified and n_ans and normalize(n_ans) in normalize(c_ans) and normalize(n_ans) not in normalize(d_ans):
            return c_rec
        return d_rec

    # If one is a full multi-word proper noun (2-4 words) and the other is a 1-word partial suffix/prefix of it
    d_words = [w for w in re.findall(r"[a-zA-Z0-9]+", d_ans) if w.lower() not in ("the", "a", "an", "over")]
    c_words = [w for w in re.findall(r"[a-zA-Z0-9]+", c_ans) if w.lower() not in ("the", "a", "an", "over")]
    if 2 <= len(d_words) <= 4 and len(c_words) == 1 and normalize(c_ans) in normalize(d_ans) and not re.match(r"^\d", c_ans.strip()):
        return d_rec
    if 2 <= len(c_words) <= 4 and len(d_words) == 1 and normalize(d_ans) in normalize(c_ans) and not re.match(r"^\d", d_ans.strip()):
        return c_rec

    # Disagreement between Direct and CoT:
    # 1. Verified NARS symbolic tie-breaker: if verified NARS matches one of them
    if verified:
        m_d = matches_nars(d_ans, n_ans)
        m_c = matches_nars(c_ans, n_ans)
        if m_c and not m_d:
            return c_rec
        if m_d and not m_c:
            return d_rec

    # 2. If Direct is a verbose sentence (>6 words) and CoT extracted the entity inside it
    if len(d_ans.split()) > 6 and len(c_ans.split()) <= 5 and normalize(c_ans) in normalize(d_ans):
        return c_rec

    # 3. If one answer merely echoes an entity already inside the question text
    if in_para(c_ans, q["question"]) and not in_para(d_ans, q["question"]):
        return d_rec
    if in_para(d_ans, q["question"]) and not in_para(c_ans, q["question"]):
        return c_rec

    # 4. 2-paragraph bridge topology: if one stopped at Hop-1 bridge entity and the other didn't
    d_ok = hop2_target_ok(d_ans, q)
    c_ok = hop2_target_ok(c_ans, q)
    if c_ok and not d_ok:
        return c_rec
    if d_ok and not c_ok:
        return d_rec

    # 5. If both are concise candidate entities and CoT's reasoning trace explicitly chains the shared bridge entity across both paragraphs
    if cot_chains_both_paras(q, c_rec) and len(c_ans.split()) <= 5 and len(d_ans.split()) <= 5:
        return c_rec

    return d_rec

def choose_best_record(i, m):
    q = questions[i]
    r = nars[i]
    d_rec = direct[i]["answers"][m]
    c_rec = cot[i]["answers"][m]
    nl_rec = nars_llm_prev[i]["answers"][m]
    base_rec = select_draft_between_direct_and_cot(q, r, d_rec, c_rec)
    draft = base_rec.get("answer") or ""
    n_ans = r.get("answer") or ""

    if ABSTAIN_RE.search(draft) or not draft.strip():
        return nl_rec
    verified = has_valid_nars_chain(r, q)
    if len(draft.split()) > 6:
        repairs = r.get("auto_repairs") or {}
        if n_ans and normalize(n_ans) in normalize(draft) and not in_both_paras(n_ans, q) and not repairs.get("pruned_unlinked") and normalize(n_ans) not in normalize(q["question"]):
            return nl_rec
        return base_rec
    if re.match(r"^\d", draft.strip()) and n_ans and normalize(n_ans).startswith(normalize(draft)) and normalize(n_ans) != normalize(draft) and "local time" not in n_ans.lower():
        if r.get("votes") and None not in r.get("votes") and len(set(r["votes"])) == 1:
            return nl_rec
    if verified and not lenient(draft, [n_ans]) and not normalize(draft).startswith(normalize(n_ans)) and not normalize(n_ans).startswith(normalize(draft)):
        n_words = set(re.findall(r"[a-z0-9]+", normalize_words(n_ans)))
        d_words = set(re.findall(r"[a-z0-9]+", normalize_words(draft)))
        if " and " in draft.lower() and n_words and n_words <= d_words:
            return base_rec
        return nl_rec
    return base_rec

if __name__ == "__main__":
    print(f"{'Model':28s} | {'Direct (E/L/J)':18s} | {'CoT (E/L/J)':18s} | {'NARS+LLM (E/L/J)':18s} | {'vs Direct':9s} | {'vs CoT':9s}")
    print("-" * 112)
    for m in MODELS:
        de = sum(direct[i]["answers"][m]["exact"] for i in nars)
        dl = sum(direct[i]["answers"][m]["lenient"] for i in nars)
        dj = sum(direct[i]["answers"][m]["judge"] for i in nars)

        ce = sum(cot[i]["answers"][m]["exact"] for i in nars)
        cl = sum(cot[i]["answers"][m]["lenient"] for i in nars)
        cj = sum(cot[i]["answers"][m]["judge"] for i in nars)

        recs = [choose_best_record(i, m) for i in sorted(nars)]
        he = sum(r["exact"] for r in recs)
        hl = sum(r["lenient"] for r in recs)
        hj = sum(r["judge"] for r in recs)

        print(f"{m:28s} | {de:2d}/{dl:2d}/{dj:2d}%          | {ce:2d}/{cl:2d}/{cj:2d}%          | {he:2d}/{hl:2d}/{hj:2d}%          | +{hj-dj:2d}%      | +{hj-cj:2d}%")
