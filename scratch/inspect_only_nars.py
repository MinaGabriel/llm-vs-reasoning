import json
from pathlib import Path

run_dir = Path("runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations")
nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}
nllm = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars_llm.jsonl").read_text().splitlines() if l.strip())}

print("=== QUESTIONS WHERE NARS IS RIGHT AND AT LEAST 2 LLMS ARE WRONG ===")
models = list(next(iter(nllm.values()))["answers"].keys())
for i, r in sorted(nars.items()):
    if r["judge"]:
        wrong_llms = [m for m in models if not base[i]["answers"][m]["judge"]]
        if len(wrong_llms) >= 1:
            print(f"Q{i}: nars={r['answer']!r} (gold={r['gold']!r}) | fq={r['from_question']} votes={r['votes']} repairs={r['auto_repairs']}")
            for m in wrong_llms[:5]:
                print(f"   LLM {m}: {base[i]['answers'][m]['answer']!r} -> NARS+LLM: {nllm[i]['answers'][m]['answer']!r} (judge={nllm[i]['answers'][m]['judge']})")
