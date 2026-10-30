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

for m in MODELS:
    print(f"=== {m} ===")
    for i in sorted(nars):
        da = direct[i]["answers"][m]
        ca = cot[i]["answers"][m]
        nla = nars_llm_prev[i]["answers"][m]
        if da["judge"] != ca["judge"]:
            print(f"  Q{i:<3d} | gold={questions[i]['answer']!r:25s} | D({int(da['judge'])})={da['answer']!r:35s} | C({int(ca['judge'])})={ca['answer']!r:35s} | N({int(nars[i]['judge'])})={nars[i]['answer']!r:25s} | NL({int(nla['judge'])})={nla['answer']!r}")
