import json, pickle
from test_nars_cot_hybrid import choose_best_record, nars, direct, cot, nars_llm_prev, questions, MODELS

for m in ["DeepSeek-V3", "Qwen3-Next-80B-A3B", "Llama-3.3-70B"]:
    print(f"=== {m} ===")
    for i in sorted(nars):
        chosen = choose_best_record(i, m)
        d = direct[i]["answers"][m]
        c = cot[i]["answers"][m]
        nl = nars_llm_prev[i]["answers"][m]
        if chosen["judge"] != (d["judge"] or c["judge"] or nl["judge"]) or chosen["exact"] != (d["exact"] or c["exact"] or nl["exact"]):
            print(f"  Q{i:<3d} | gold={questions[i]['answer']!r:25s} | chosen({int(chosen['exact'])},{int(chosen['judge'])})={chosen['answer']!r} | D({int(d['exact'])},{int(d['judge'])})={d['answer']!r} | C({int(c['exact'])},{int(c['judge'])})={c['answer']!r} | NL({int(nl['exact'])},{int(nl['judge'])})={nl['answer']!r}")
