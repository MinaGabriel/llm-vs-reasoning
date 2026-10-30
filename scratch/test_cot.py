import json, pickle, re
from openai import OpenAI

import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from keys import deepinfra_key
API_KEY = deepinfra_key()  # $DEEPINFRA_API_KEY or the git-ignored .deepinfra_key
client = OpenAI(api_key=API_KEY, base_url="https://api.deepinfra.com/v1/openai", max_retries=2, timeout=60)

BASELINES = [
    "google/gemma-3-4b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-3-12b-it",
    "microsoft/phi-4",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "google/gemma-3-27b-it",
    "google/gemma-4-31B-it",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "deepseek-ai/DeepSeek-V3",
]

COT_PROMPT = """Answer the question using only the facts below.
Think step by step in 1-2 short sentences to connect the facts, then on the final line write "Answer: " followed by only the short answer.

Facts:
{facts}

Question: {question}"""

def parse_cot_answer(text):
    if not text:
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    for ln in reversed(lines):
        m = re.search(r"(?:final\s+)?answer\s*[:\-]\s*(.+)$", ln, re.I)
        if m:
            ans = m.group(1).strip().strip("*\"'`").rstrip(".")
            if ans:
                return ans
    return lines[-1].strip("*\"'`").rstrip(".")

with open("runs/MuSiQue_test_entity.pkl", "rb") as f:
    questions = pickle.load(f)

q = questions[15]
facts_str = "\n".join(f"- {fact}" for p in q["supporting"] if p["facts"] for fact in p["facts"])
for m in BASELINES:
    resp = client.chat.completions.create(
        model=m, temperature=0, max_tokens=160,
        messages=[{"role": "user", "content": COT_PROMPT.format(facts=facts_str, question=q["question"])}]
    )
    raw = resp.choices[0].message.content or ""
    parsed = parse_cot_answer(raw)
    print(f"{m.split('/')[-1]:35s} -> parsed: {parsed!r} | gold: {q['answer']!r}")
