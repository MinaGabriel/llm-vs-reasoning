import json
from pathlib import Path

run_dir = Path("runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations")
nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}
nllm = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars_llm.jsonl").read_text().splitlines() if l.strip())}

for idx, r in sorted(nars.items()):
    b = base[idx]["answers"]
    nl = nllm[idx]["answers"]
    models = [m for m in nl if m in b]
    helped = [m for m in models if not b[m]["judge"] and nl[m]["judge"]]
    hurt = [m for m in models if b[m]["judge"] and not nl[m]["judge"]]
    if len(hurt) >= 1 or len(helped) >= 2:
        nj = r["judge"]
        ans = r["answer"]
        gold = r["gold"]
        q = r["question"]
        votes = r.get("votes")
        repairs = r.get("auto_repairs")
        fq = r.get("from_question")
        rules = r.get("rules")
        print(f"Q{idx}: +{len(helped)} -{len(hurt)} | nars_judge={nj} | ans={ans!r} | gold={gold!r}")
        print(f"  Q: {q}")
        print(f"  votes={votes} | repairs={repairs} | from_q={fq}")
        print(f"  rules={rules}")
        for m in hurt[:3]:
            print(f"    HURT {m}: base={b[m]['answer']!r} -> nllm={nl[m]['answer']!r}")
        for m in helped[:3]:
            print(f"    HELP {m}: base={b[m]['answer']!r} -> nllm={nl[m]['answer']!r}")
