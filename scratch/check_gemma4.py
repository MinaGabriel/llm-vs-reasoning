from test_nars_cot_hybrid import *

for i in sorted(nars):
    chosen = choose_best_record(i, "gemma-4-31B")
    nl = nars_llm_prev[i]["answers"]["gemma-4-31B"]
    d = direct[i]["answers"]["gemma-4-31B"]
    c = cot[i]["answers"]["gemma-4-31B"]
    if chosen["exact"] != nl["exact"] or chosen["judge"] != nl["judge"]:
        print(f"Q{i:<3d} | gold={questions[i]['answer']!r:25s} | chosen({int(chosen['exact'])},{int(chosen['judge'])})={chosen['answer']!r} | D({int(d['exact'])},{int(d['judge'])})={d['answer']!r} | C({int(c['exact'])},{int(c['judge'])})={c['answer']!r} | NL({int(nl['exact'])},{int(nl['judge'])})={nl['answer']!r}")
