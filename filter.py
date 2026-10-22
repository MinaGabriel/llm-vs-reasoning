# filter.py: keep only the questions a second (filter) model ALSO answers wrong, closed-book (vLLM).
# Usage: python filter.py --input MuSiQue_Qwen2.5-0.5B-Instruct_1000.pkl --model Qwen/Qwen2.5-72B-Instruct-AWQ --n 500
# Output: MuSiQue_Qwen2.5-0.5B-Instruct_1000__filtered_Qwen2.5-72B-Instruct-AWQ_500.pkl
import argparse
import pickle

from vllm import LLM, SamplingParams

from MuSiQue import INSTRUCTION, is_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="candidate pool .pkl from MuSiQue.py")
    parser.add_argument("--model", required=True, help="filter model, e.g. Qwen/Qwen2.5-72B-Instruct-AWQ")
    parser.add_argument("--n", type=int, default=500, help="number of questions to keep")
    args = parser.parse_args()

    candidates = pickle.load(open(args.input, "rb"))
    llm = LLM(model=args.model, max_model_len=2048, gpu_memory_utilization=0.90)   # AWQ models are detected automatically
    tok = llm.get_tokenizer()
    name = args.model.split("/")[-1]

    # Same prompt as MuSiQue.py, through the model's chat template; greedy decoding
    prompts = [tok.apply_chat_template([{"role": "user", "content": f"{INSTRUCTION}\n\nQuestion: {ex['question']}\nAnswer:"}],
                                       tokenize=False, add_generation_prompt=True) for ex in candidates]
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32))

    kept, checked = [], 0
    for ex, out in zip(candidates, outputs):
        checked += 1
        raw = out.outputs[0].text.strip()
        answer = raw.split("\n")[0]
        if not is_correct(answer, ex["gold_answers"]):
            ex["failed_by"] = ex.get("failed_by", [ex["model"]]) + [args.model]
            ex[f"answer_{name}"] = answer
            ex[f"raw_answer_{name}"] = raw
            kept.append(ex)
        if len(kept) == args.n:
            break

    out_file = f"{args.input.removesuffix('.pkl')}__filtered_{name}_{len(kept)}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(kept, f)
    print(f"Checked {checked} of {len(candidates)} candidates; {name} failed {len(kept)}. Saved to {out_file}")


if __name__ == "__main__":
    main()