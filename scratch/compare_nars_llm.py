import json
from pathlib import Path

run_dir = Path("runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations")
nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}
nllm = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars_llm.jsonl").read_text().splitlines() if l.strip())}

models = list(next(iter(nllm.values()))["answers"].keys())
for m in models:
    both = sum(base[i]["answers"][m]["judge"] and nars[i]["judge"] for i in nars)
    only_nars = sum((not base[i]["answers"][m]["judge"]) and nars[i]["judge"] for i in nars)
    only_llm = sum(base[i]["answers"][m]["judge"] and (not nars[i]["judge"]) for i in nars)
    neither = sum((not base[i]["answers"][m]["judge"]) and (not nars[i]["judge"]) for i in nars)
    print(f"{m:<28} | LLM={both+only_llm}% | NARS={both+only_nars}% | OracleUnion={both+only_nars+only_llm}% (only_nars={only_nars}, only_llm={only_llm})")
