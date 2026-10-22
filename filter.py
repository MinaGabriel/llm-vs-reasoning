# filter.py: keep only the questions a second (filter) model ALSO answers wrong, closed-book.
# Usage: python filter.py --input MuSiQue_Qwen2.5-0.5B-Instruct_1000.pkl --model meta-llama/Llama-3.3-70B-Instruct --n 500 --4bit
# Output: MuSiQue_Qwen2.5-0.5B-Instruct_1000__filtered_Llama-3.3-70B-Instruct_500.pkl
import argparse
import pickle

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from MuSiQue import ask, is_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="candidate pool .pkl from MuSiQue.py")
    parser.add_argument("--model", required=True, help="filter model, e.g. meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--n", type=int, default=500, help="number of questions to keep")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="load in 4-bit (needed for 70B on 80 GB)")
    args = parser.parse_args()

    candidates = pickle.load(open(args.input, "rb"))
    tok = AutoTokenizer.from_pretrained(args.model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16) if args.four_bit else None
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16,
                                                 quantization_config=quant).eval()
    name = args.model.split("/")[-1]

    kept = []
    bar = tqdm(candidates, desc=f"Filtering with {name}")
    for ex in bar:
        answer = ask(model, tok, ex["question"])
        if not is_correct(answer, ex["gold_answers"]):
            ex["failed_by"] = ex.get("failed_by", [ex["model"]]) + [args.model]
            ex[f"answer_{name}"] = answer
            kept.append(ex)
        bar.set_postfix(kept=len(kept))
        if len(kept) == args.n:
            break

    out_file = f"{args.input.removesuffix('.pkl')}__filtered_{name}_{len(kept)}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(kept, f)
    print(f"Checked {bar.n} of {len(candidates)} candidates; {name} failed {len(kept)}. Saved to {out_file}")


if __name__ == "__main__":
    main()