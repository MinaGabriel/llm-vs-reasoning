from test_nars_cot_hybrid import *

for m in MODELS:
    for i in sorted(nars):
        q = questions[i]
        r = nars[i]
        d_rec = direct[i]["answers"][m]
        c_rec = cot[i]["answers"][m]
        if d_rec["judge"] != c_rec["judge"]:
            chosen = choose_best_record(i, m)
            if not chosen["judge"]:
                print(f"{m:26s} Q{i:<3d} | gold={q['answer']!r:25s} | D({int(d_rec['judge'])})={d_rec['answer']!r:35s} | C({int(c_rec['judge'])})={c_rec['answer']!r:35s} | N={r['answer']!r}")
