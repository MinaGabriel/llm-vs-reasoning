import json, os, pickle, random, re, time, unicodedata, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from prettytable import PrettyTable

import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from keys import deepinfra_key
API_KEY = deepinfra_key()  # $DEEPINFRA_API_KEY or the git-ignored .deepinfra_key
DATASET = "MuSiQue"
SPLIT = "test"
HOPS = 2
N_QUESTIONS = 100
DISTRACTORS = False
SEED = 0
WORKERS = 8
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
ONA_OPTIONS = {"plan": True, "votes": 3, "merge_relations": True}

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

client = OpenAI(api_key=API_KEY, base_url="https://api.deepinfra.com/v1/openai", max_retries=2, timeout=60)

def label(model):
    return model.split("/")[-1].replace("Meta-", "").replace("-Instruct-Turbo", "").replace("-Instruct", "").replace("-it", "")

def load_prices():
    for attempt in range(3):
        try:
            with urllib.request.urlopen("https://api.deepinfra.com/models/list", timeout=20) as response:
                return {m["model_name"]: (m["pricing"]["cents_per_input_token"] * 1e4, m["pricing"]["cents_per_output_token"] * 1e4)
                        for m in json.load(response) if (m.get("pricing") or {}).get("type") == "tokens"}
        except Exception:
            time.sleep(1)
    return {}

PRICES = load_prices()

def cost(input_tokens, output_tokens, model):
    price = PRICES.get(model, (0, 0))
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000

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

def exact(answer, golds):
    return any(normalize(answer or "") == normalize(g) for g in golds)

JUDGE_PROMPT = """You grade answers to a question. Be strict.

Question: {question}
Gold answer: {gold}
Model answer: {answer}

The model answer is correct ONLY if it names the same entity, date, number or place as the gold answer.
Different wording, a full name, or extra detail around the same thing is fine:
  "Philip Mountbatten" for "Prince Philip" -> yes
  "12,000 people" for "12,000" -> yes
  "4 June 1990" for "1990" -> yes
It is WRONG if it names something else, even something related: a part of the gold answer, a larger thing that
contains it, a person connected to it, or a vaguer or partial answer:
  "Battle of Hastings" for "Norman Conquest" -> no   (a part, not the same thing)
  "Henry VIII" for "Edward VI" -> no                 (a related person)
  "Europe" for "France" -> no                        (broader)
  "200 km" for "200 km south of Paris" -> no         (partial)
  "The text does not say" -> no                      (no answer)
Reply with only yes or no."""

METRICS = ["exact", "lenient", "judge"]

def golds(q):
    return [q["answer"], *(q.get("gold_answers") or [])]

def paragraphs(q):
    return [p for p in (q["context"] if DISTRACTORS else q["supporting"]) if p["facts"]]

def facts(q):
    return [f for p in paragraphs(q) for f in p["facts"]]

def metrics(q, answer):
    g = golds(q)
    result = {"exact": exact(answer, g), "lenient": lenient(answer, g), "judge_cost": 0.0}
    result["judge"] = result["lenient"]
    if answer and not result["lenient"]:
        try:
            response = client.chat.completions.create(model=JUDGE_MODEL, temperature=0, max_tokens=3, messages=[{
                "role": "user", "content": JUDGE_PROMPT.format(question=q["question"], gold=q["answer"], answer=answer)}])
            result["judge"] = (response.choices[0].message.content or "").strip().lower().startswith("yes")
            result["judge_cost"] = cost(response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0, JUDGE_MODEL)
        except Exception as error:
            print(f"   ! judge failed ({type(error).__name__}), using the lenient match")
    return result

with open(f"runs/{DATASET}_{SPLIT}_entity.pkl", "rb") as f:
    questions = pickle.load(f)
pool = [i for i, q in enumerate(questions) if q["hops"] == HOPS]
INDICES = sorted(random.Random(SEED).sample(pool, min(N_QUESTIONS, len(pool))))
OPTIONS_TAG = "_".join(k if v is True else f"{k}{v}" for k, v in ONA_OPTIONS.items() if v is True or (type(v) is int and v > 1))
RUN_DIR = f"runs/{DATASET}{'_dev' if SPLIT == 'dev' else ''}_{HOPS}hop_{len(INDICES)}q_seed{SEED}" + (f"_{OPTIONS_TAG}" if OPTIONS_TAG else "")

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

def ask_cot(q, model):
    answer, raw, tokens = None, None, (0, 0)
    try:
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=160, messages=[{"role": "user", "content": COT_PROMPT.format(
                facts="\n".join(f"- {f}" for f in facts(q)), question=q["question"])}])
        raw = (response.choices[0].message.content or "").strip()
        answer = parse_cot_answer(raw)
        tokens = (response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0)
    except Exception as error:
        print(f"   ! Q{q['id'][:8]} CoT {label(model)}: {type(error).__name__}")
    return {"answer": answer, "cot_trace": raw, **metrics(q, answer), "cost": cost(*tokens, model)}

def run_cot(index):
    q = questions[index]
    answers = {label(m): ask_cot(q, m) for m in BASELINES}
    return {"index": index, "question": q["question"], "gold": q["answer"], "answers": answers,
            "cost": sum(a["cost"] for a in answers.values()),
            "judge_cost": sum(a["judge_cost"] for a in answers.values())}

path = f"{RUN_DIR}/cot.jsonl"
records, start = [], time.time()
open(path, "w").close()
with ThreadPoolExecutor(WORKERS) as executor:
    futures = [executor.submit(run_cot, i) for i in INDICES]
    for done, future in enumerate(as_completed(futures), 1):
        record = future.result()
        records.append(record)
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if done % 10 == 0 or done == len(INDICES):
            print(f"[{done:>3}/{len(INDICES)}] spent ${sum(r['cost'] + r['judge_cost'] for r in records):.4f}   {time.time() - start:.0f}s", flush=True)

n = len(records)
table = PrettyTable(["Model (LLM CoT)", "Exact", "Lenient", "Judge", "Cost USD"])
for m in BASELINES:
    answers = [r["answers"][label(m)] for r in records]
    table.add_row([f"{label(m)} (CoT)", *[f"{100 * sum(a[k] for a in answers) / n:.1f}%" for k in METRICS],
                   f"${sum(a['cost'] for a in answers):.4f}"])
print(table)
