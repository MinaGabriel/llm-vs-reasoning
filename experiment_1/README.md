# Experiment 1 · MuSiQue, 0 to 16 distractors

**Question.** As distractor paragraphs are added, does NARS (with a distilled 8B rule writer) lose less accuracy than
the same 8B answering directly, and how does that compare with LLMs from 4B to 671B?

**Fixed before running (2026-10-30).** Nothing below changes after the first result is seen.

| | |
|---|---|
| Questions | 200 MuSiQue 2-hop questions, seed 0: every fresh one from the validation split (74 pass the filters; 126 more come from train), then the train split, using only questions never tried during the LoRA's training-data generation; none shares a question or supporting paragraph with the test set, the dev set or the LoRA training data. Filtered as every earlier set: the 70B cannot answer them closed-book, can answer them from the 2 supporting paragraphs, single-term gold answer. Each records its `source_split`, checked as a covariate |
| Input | facts extracted by Llama-3.1-70B from the 2 supporting paragraphs plus the first k of 16 distractor paragraphs (the dataset's own), in a fixed reading order; the sets nest across k |
| k | 0, 4, 8, 16 |
| NARS systems | NARS + 8B + LoRA (adapter `MinaGabriel/nars-llama31-8b-lora`, revision recorded in config.json); NARS + DeepSeek-V3 (the teacher). ONA options: plan, 3 votes, merge_relations; ONA commit dc4efd0 |
| LLM baselines (direct answer, same prompt) | Llama-3.1-8B (the student's own weights, vLLM bf16; the reference), gemma-3-4b, gemma-3-12b, phi-4, Mistral-Small-3.2-24B, gemma-3-27b, Llama-3.3-70B, Qwen3-Next-80B-A3B, DeepSeek-V3 |
| Metrics | judge (Qwen3-235B-A22B-Instruct-2507, strict prompt in the notebook; exact matches count as right) and exact match |
| Main test | NARS + 8B + LoRA vs Llama-3.1-8B: system × log2(1 + k) interaction in a logistic regression with questions as clusters (GEE), judge metric; exact match as a robustness check |
| Secondary | the same interaction for every other system vs Llama-3.1-8B; change from k = 0 to 16 minus that of Llama-3.1-8B, per question, bootstrap 95% CI; paired McNemar at each k; accuracy by source split |

**Files.**
- `build_questions.ipynb` (local): builds `questions/MuSiQue_exp1.pkl` and its stats.
- `run_experiment_1.ipynb` (Colab, A100): serves the 8B and the adapter with vLLM, runs all 11 systems at every k, and
  writes to `MyDrive/nars_lora/experiment_1/`: `config.json` (everything that defines the run), `answers/` (one line per
  answer: inputs read, raw reply, answer, tokens, cost, time, metrics), `traces/` (NARS rows: triples, plan, rules,
  votes, Narsese, ONA output, every LLM call), and from the Results cell `results.csv`, `report.txt`, `accuracy_vs_k.png`.
- `ONA.py`, `pipeline.py`, `llm.py`, `MuSiQue.py`, `keys.py`: frozen copies of the code this experiment uses.
