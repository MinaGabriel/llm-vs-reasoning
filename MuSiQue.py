# MuSiQue.py: the MuSiQue dataset in the shared schema. Every dataset file provides DATASET and load_pool(seed).
#   Shared schema: id, dataset, question, answer, gold_answers, decomposition, paragraphs, source, failed_by, filter_log
#   (datasets without paragraphs or decompositions use empty lists).
import random

from datasets import load_dataset

DATASET = "MuSiQue"


def to_schema(ex):
    return {
        "id": ex["id"],
        "dataset": DATASET,
        "question": ex["question"],
        "answer": ex["answer"],
        "gold_answers": [ex["answer"]] + list(ex["answer_aliases"] or []),
        "decomposition": [{"question": s["question"], "answer": s["answer"], "support_idx": s["paragraph_support_idx"]}
                          for s in ex["question_decomposition"]],
        "paragraphs": [{"idx": p["idx"], "title": p["title"], "text": p["paragraph_text"],
                        "is_supporting": p["is_supporting"]} for p in ex["paragraphs"]],
        "source": "Salesforce/ContextualBench (MuSiQue, validation)",
        "failed_by": [],
        "filter_log": [],
    }


def load_pool(seed=0):
    """All answerable two-hop questions whose second step refers to the first answer (#1), shuffled."""
    data = load_dataset("Salesforce/ContextualBench", "MuSiQue", split="validation")
    pool = [to_schema(ex) for ex in data if ex["answerable"] and len(ex["question_decomposition"]) == 2
            and "#1" in ex["question_decomposition"][1]["question"]]
    random.Random(seed).shuffle(pool)
    return pool


if __name__ == "__main__":
    print(f"{len(load_pool())} eligible {DATASET} questions")