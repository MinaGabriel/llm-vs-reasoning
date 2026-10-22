# MuSiQue.py: collect MuSiQue two-hop questions that a model CANNOT answer closed-book.
# Usage: python MuSiQue.py --model Qwen/Qwen2.5-0.5B-Instruct --n 10
# Output: MuSiQue_<model name>.pkl, e.g. MuSiQue_Qwen2.5-0.5B-Instruct.pkl
import argparse
import pickle
import random
import re
import unicodedata

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = """Answer with only the answer, in as few words as possible. No sentences, no explanations.

Question: Who directed Jaws?
Answer: Steven Spielberg
Question: What is the capital of the country where the Eiffel Tower is?
Answer: Paris"""

def normalize(text):
    """Lowercase, no accents, no punctuation or articles, no spaces: 'The Fiorello La Guardia.' -> 'fiorellolaguardia'"""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^a-z0-9 ]", " ", text))
    return text.replace(" ", "")


def is_correct(prediction, golds):
    """Correct if the prediction and any gold answer contain each other after normalization."""
    p = normalize(prediction)
    return bool(p) and any(normalize(g) and (normalize(g) in p or p in normalize(g)) for g in golds)


def ask(model, tok, question):
    """The model's short greedy answer to a question, with no documents (closed-book)."""
    prompt = tok.apply_chat_template([{"role": "user", "content": f"{INSTRUCTION}\n\nQuestion: {question}\nAnswer:"}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n", type=int, default=10, help="number of questions the model fails")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda:0", torch_dtype=torch.bfloat16).eval()

    # Answerable two-hop questions whose second step refers to the first answer (#1)
    data = load_dataset("Salesforce/ContextualBench", "MuSiQue", split="validation")
    pool = [ex for ex in data if ex["answerable"] and len(ex["question_decomposition"]) == 2
            and "#1" in ex["question_decomposition"][1]["question"]]
    random.Random(args.seed).shuffle(pool)

    # Keep the questions the model answers WRONG without documents (no prior knowledge)
    kept, checked = [], 0
    for ex in pool:
        golds = [ex["answer"]] + list(ex["answer_aliases"] or [])
        answer = ask(model, tok, ex["question"])
        checked += 1
        if not is_correct(answer, golds):
            kept.append({**ex, "gold_answers": golds, "model": args.model, "model_answer": answer})
        if len(kept) == args.n:
            break

    out_file = f"MuSiQue_{args.model.split('/')[-1]}_{args.n}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(kept, f)
    print(f"Checked {checked} questions; the model failed {len(kept)}. Saved to {out_file}")


if __name__ == "__main__":
    main()