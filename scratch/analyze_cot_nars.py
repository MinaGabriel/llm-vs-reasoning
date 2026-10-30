import json, pickle, re, unicodedata
from collections import Counter

RUN_DIR = "runs/MuSiQue_2hop_100q_seed0_plan_votes3_merge_relations"
with open("runs/MuSiQue_test_entity.pkl", "rb") as f:
    questions = pickle.load(f)

nars = {r["index"]: r for r in map(json.loads, open(f"{RUN_DIR}/nars.jsonl"))}
direct = {r["index"]: r for r in map(json.loads, open(f"{RUN_DIR}/baselines.jsonl"))}
cot = {r["index"]: r for r in map(json.loads, open(f"{RUN_DIR}/cot.jsonl"))}
nars_llm_prev = {r["index"]: r for r in map(json.loads, open(f"{RUN_DIR}/nars_llm.jsonl"))}

def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(a|an|the)\b", " ", text))

def lenient(answer, golds):
    a = normalize(answer or "")
    for g in map(normalize, golds):
        short, long_ = sorted((a, g), key=len)
        if short and short in long_ and 2 * len(short) >= len(long_):
            return True
    return False

models = list(direct[15]["answers"].keys())
for m in models:
    d_j = sum(direct[i]["answers"][m]["judge"] for i in direct)
    c_j = sum(cot[i]["answers"][m]["judge"] for i in cot)
    nl_j = sum(nars_llm_prev[i]["answers"][m]["judge"] for i in nars_llm_prev)
    oracle_dc = sum(direct[i]["answers"][m]["judge"] or cot[i]["answers"][m]["judge"] for i in direct)
    oracle_dcn = sum(direct[i]["answers"][m]["judge"] or cot[i]["answers"][m]["judge"] or nars[i]["judge"] for i in direct)
    print(f"{m:28s} | Direct: {d_j:2d} | CoT: {c_j:2d} | Prev NARS+LLM: {nl_j:2d} | D|C: {oracle_dc:2d} | D|C|N: {oracle_dcn:2d}")
