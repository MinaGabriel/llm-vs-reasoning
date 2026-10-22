import random
import re
import unicodedata
import torch
from datasets import load_dataset


class MuSiQue:
    def __init__(self, n, tok, model):
        self.n = n
        self.tok = tok
        self.model = model
        self.questions = self._get_samples()

    def _norm(self, s):
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())

    def _is_correct(self, pred, golds):
        p = self._norm(pred)
        return bool(p) and any(self._norm(g) in p or p in self._norm(g) for g in golds if self._norm(g))

    def _ask(self, question):
        msg = [{"role": "user", "content": f"Answer in 1 to 5 words only.\nQuestion: {question}"}]
        prompt = self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=20, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def _get_samples(self):
        ds = list(load_dataset("Salesforce/ContextualBench", "MuSiQue", split="validation"))
        random.shuffle(ds)

        samples = []
        for ex in ds:
            if not ex["answerable"]:
                continue

            golds = [ex["answer"]] + list(ex["answer_aliases"] or [])
            ans = self._ask(ex["question"])

            if not self._is_correct(ans, golds):
                ex["model_answer"] = ans
                samples.append(ex)

            if len(samples) == self.n:
                break

        return samples

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]

    def __iter__(self):
        return iter(self.questions)