from test_nars_cot_hybrid import *

def cot_chains_both_paras(q, c_rec):
    trace = c_rec.get("cot_trace") or ""
    c_ans = c_rec.get("answer") or ""
    ps = paragraphs(q)
    if len(ps) != 2 or not trace or not c_ans:
        return False
    p0_str, p1_str = " ".join(ps[0]["facts"]), " ".join(ps[1]["facts"])
    # Check if trace mentions capitalized words/phrases from BOTH p0 and p1, and c_ans is in one of the paragraphs
    if not (in_para(c_ans, p0_str) or in_para(c_ans, p1_str)):
        return False
    # Check if any 4+ char word in p0 (not in question) and any 4+ char word in p1 (not in question) appear in trace
    q_norm = normalize_words(q["question"])
    t_norm = normalize_words(trace)
    w0 = {w for w in re.findall(r"[a-z0-9]{4,}", normalize_words(p0_str)) if w not in q_norm}
    w1 = {w for w in re.findall(r"[a-z0-9]{4,}", normalize_words(p1_str)) if w not in q_norm}
    shared = w0 & w1
    return bool(shared and any(w in t_norm for w in shared))

for m in MODELS:
    for i in sorted(nars):
        q = questions[i]
        r = nars[i]
        d_rec = direct[i]["answers"][m]
        c_rec = cot[i]["answers"][m]
        if d_rec["judge"] != c_rec["judge"]:
            chosen = select_draft_between_direct_and_cot(q, r, d_rec, c_rec)
            chains = cot_chains_both_paras(q, c_rec)
            if chosen["judge"] == 0:
                print(f"{m:26s} Q{i:<3d} | D({int(d_rec['judge'])})={d_rec['answer']!r:30s} | C({int(c_rec['judge'])})={c_rec['answer']!r:30s} | chains={chains}")
