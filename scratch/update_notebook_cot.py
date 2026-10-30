import json

nb_path = "/Users/mina/Developer/llm-vs-reasoning/nars_with_llm.ipynb"
with open(nb_path, "r") as f:
    nb = json.load(f)

# Update intro markdown cell
nb["cells"][0]["source"] = [
    "# NARS Alone vs. LLM (Direct) vs. LLM (CoT) vs. NARS + LLM\n",
    "Every system reads the same 70B-extracted facts. Set the dataset, split, hops and sample size in **Settings**.\n",
    "\n",
    "1. **Run NARS**: translates facts into triples and rules, executes ONA, and extracts ONA's `Stamp=[...]` symbolic proof chain and candidate bindings (`nars.jsonl`).\n",
    "2. **Run Baselines — LLM Direct**: every model answers directly from the extracted facts (`baselines.jsonl`).\n",
    "3. **Run Baselines — LLM CoT**: every model reasons step-by-step (Chain-of-Thought) across the facts before outputting `Answer: <short answer>` (`cot.jsonl`).\n",
    "4. **Run NARS + LLM**: every model answers using its initial reasoning **plus** NARS's verified symbolic proof trace (`nars_llm.jsonl`).\n",
    "5. **Final Report**: compares **NARS Alone**, **LLM (Direct)**, **LLM (CoT)**, and **NARS + LLM** across all three metrics (`Exact`, `Lenient`, `Judge`) and saves `report_nars_llm.png`."
]

# Create CoT cells
cot_md_cell = {
    "cell_type": "markdown",
    "id": "run_cot_md",
    "metadata": {},
    "source": [
        "## Run Baselines (LLM Chain-of-Thought)\n",
        "Every model reasons step-by-step across the extracted facts before outputting `Answer: <short answer>` (`cot.jsonl`)."
    ]
}

cot_code_source = """# ── Chain-of-Thought Baselines: step-by-step reasoning from the same extracted facts ───────────────
COT_PROMPT = \"\"\"Answer the question using only the facts below.
Think step by step in 1-2 short sentences to connect the facts, then on the final line write "Answer: " followed by only the short answer.

Facts:
{facts}

Question: {question}\"\"\"


def parse_cot_answer(text):
    if not text:
        return None
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    for ln in reversed(lines):
        m = re.search(r"(?:final\\s+)?answer\\s*[:\\-]\\s*(.+)$", ln, re.I)
        if m:
            ans = m.group(1).strip().strip("*\\"'`").rstrip(".")
            if ans:
                return ans
    return lines[-1].strip("*\\"'`").rstrip(".")


def ask_cot(q, model):
    answer, raw, tokens = None, None, (0, 0)
    try:
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=160, messages=[{"role": "user", "content": COT_PROMPT.format(
                facts="\\n".join(f"- {f}" for f in facts(q)), question=q["question"])}])
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


def show_cot(r, records):
    return "  ".join(f"{name} {'✓' if a['judge'] else '✗'} {sum(x['answers'][name]['judge'] for x in records)}"
                     for name, a in r["answers"].items()) + "   (CoT judge)"


cot_baselines = run_all(run_cot, "cot.jsonl", show_cot)

n = len(cot_baselines)
table = PrettyTable(["Model (LLM CoT)", "Exact", "Lenient", "Judge", "Cost USD"])
for m in BASELINES:
    answers = [r["answers"][label(m)] for r in cot_baselines]
    table.add_row([f"{label(m)} (CoT)", *[f"{100 * sum(a[k] for a in answers) / n:.1f}%" for k in METRICS],
                   f"${sum(a['cost'] for a in answers):.4f}"])
print(f"\\nLLM (CoT) on {n} {DATASET} {HOPS}-hop questions (from the extracted facts)"
      f"   judge ${sum(r['judge_cost'] for r in cot_baselines):.4f}")
print(table)
"""

cot_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "run_cot_code",
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in cot_code_source.rstrip("\n").split("\n")]
}
cot_code_cell["source"][-1] = cot_code_cell["source"][-1].rstrip("\n")

# Keep cells up to run_baselines_code, insert cot cells if not already present
new_cells = []
for c in nb["cells"]:
    if c.get("id") in ("run_cot_md", "run_cot_code"):
        continue
    new_cells.append(c)
    if c.get("id") == "run_baselines_code":
        new_cells.append(cot_md_cell)
        new_cells.append(cot_code_cell)

nb["cells"] = new_cells
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Successfully inserted CoT cells into nars_with_llm.ipynb")
