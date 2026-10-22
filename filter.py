# filter.py: keep only the questions a second (filter) model ALSO answers wrong, closed-book (vLLM).
#   An answer counts as correct if it string-matches a gold answer OR the model judges it equivalent.
# Usage: python filter.py --input MuSiQue_Qwen2.5-0.5B-Instruct_1000.pkl --model meta-llama/Llama-3.3-70B-Instruct --n 500
import argparse
import pickle

from vllm import LLM, SamplingParams

from MuSiQue import INSTRUCTION, is_correct

JUDGE = """Question: {question}
Gold answer: {gold}
Model answer: {answer}
Does the model answer refer to the same thing as the gold answer (e.g. "Prince Philip" = "Philip Mountbatten")?
Reply with only yes or no."""


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="candidate pool .pkl from MuSiQue.py")
    parser.add_argument("--model", required=True, help="filter model, e.g. meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--n", type=int, default=500, help="number of questions to keep")
    args = parser.parse_args()

    candidates = pickle.load(open(args.input, "rb"))
    llm = LLM(model=args.model, max_model_len=2048, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    name = args.model.split("/")[-1]

    # 1. Closed-book answers (same prompt as MuSiQue.py), greedy
    outs = llm.generate([chat(tok, f"{INSTRUCTION}\n\nQuestion: {ex['question']}\nAnswer:") for ex in candidates],
                        SamplingParams(temperature=0, max_tokens=32))
    raws = [o.outputs[0].text.strip() for o in outs]
    answers = [r.split("\n")[0] for r in raws]

    # 2. Judge every answer that string matching calls wrong
    to_judge = [i for i, (ex, a) in enumerate(zip(candidates, answers)) if not is_correct(a, ex["gold_answers"])]
    verdicts = llm.generate([chat(tok, JUDGE.format(question=candidates[i]["question"],
                                                    gold=" / ".join(candidates[i]["gold_answers"]),
                                                    answer=answers[i])) for i in to_judge],
                            SamplingParams(temperature=0, max_tokens=3))
    judged_correct = {i for i, v in zip(to_judge, verdicts) if v.outputs[0].text.strip().lower().startswith("yes")}

    # 3. Keep the questions that are wrong by BOTH checks
    kept = []
    for i in to_judge:
        if i in judged_correct:
            continue
        ex = candidates[i]
        ex["failed_by"] = ex.get("failed_by", [ex["model"]]) + [args.model]
        ex[f"answer_{name}"] = answers[i]
        ex[f"raw_answer_{name}"] = raws[i]
        kept.append(ex)
        if len(kept) == args.n:
            break

    out_file = f"{args.input.removesuffix('.pkl')}__filtered_{name}_{len(kept)}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(kept, f)
    print(f"{len(candidates)} candidates: {len(candidates) - len(to_judge)} correct by string match, "
          f"{len(judged_correct)} correct by the judge, {len(kept)} kept. Saved to {out_file}")


if __name__ == "__main__":
    main()