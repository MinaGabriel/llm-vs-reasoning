# nars_ona.py: the formal NARS branch, end to end (ONA = OpenNARS for Applications).
#   Input : records with "question", "gold_answers", "facts" (from facts.py: text and paragraph provenance).
#           Translation supplies f = 1.0, c = 0.5 by default; legacy facts may supply truth values.
#   Output: record["nars"] = answer, truth (f, c), expectation e, correct, status, and the full Narsese program.
#
#   1. Triples (LLM): each fact -> 1-3 triples (subject, relation, object). Batched; each unique fact once.
#   2. Rules (LLM):   question + this question's relations/entities -> up to 3 rules (conditions -> answer variable).
#   3. Narsese (code): facts  <(s * o) --> r>. {f c}   with asserted f = 1.0, c = 0.5 by default.
#                      Older scored files remain supported through their p_true field.
#                      rules  <((A && B) && C) ==> <$x --> answer>>. {1.0 0.9}   (named variables $a, $b, ...)
#                      query  <?1 --> answer>?   then N inference cycles
#   4. ONA (code): run ./NAR shell, take the last "Answer:" line -> atom, f, c; e = c (f - 0.5) + 0.5; grade vs gold.
#   Status: ok | no_answer (ONA found nothing) | translation_error (no valid rule).
#
# Setup:  git clone https://github.com/opennars/OpenNARS-for-Applications ONA && cd ONA && sh build.sh
# Usage:  python nars_ona.py --input MuSiQue_final.pkl --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
import argparse
import json
import pickle
import re
import subprocess
import time
import unicodedata

from filter import batch_size_for, chat, generate_batched, is_correct, load_llm, save_stats, unload_llm

TRIPLES = """Convert the fact into 1 to 3 triples (subject, relation, object).
- Copy entity names exactly as written in the fact.
- Relation: short snake_case phrase, read as "subject relation object" (e.g. wife_of, born_in, directed, played).
- Reply with JSON only: a list of triples.

Fact: Maria Shriver is the wife of Arnold Schwarzenegger.
[{{"subject": "Maria Shriver", "relation": "wife_of", "object": "Arnold Schwarzenegger"}}]
Fact: Dennis Quaid played Jerry Lee Lewis in Great Balls of Fire.
[{{"subject": "Dennis Quaid", "relation": "played", "object": "Jerry Lee Lewis"}}, {{"subject": "Dennis Quaid", "relation": "acted_in", "object": "Great Balls of Fire"}}]

Fact: {fact}
"""

RULES = """Translate the question into a rule using the available triples.
- Most triples are distractors. Identify the named subject, work, place, or event in the question and anchor the rule to its matching entity.
- Follow a connected chain of relations from that anchor to the requested answer. Shared words such as "child" or "performer" alone do not establish relevance.
- Use ONLY relations and entities in the triples, spelled exactly as listed. Preserve each relation's subject/object direction.
- Replace intermediate entities and the requested answer with variables ?A, ?B, ?C. Keep the question's anchor as a constant.
- The "answer" field must name a variable used in the conditions, not a known entity.
- Each condition is [subject, relation, object]. All conditions must hold together.
- Return 1 to 3 rules only when supported by connected triples; otherwise return an empty rules list. Do not invent missing links.
- Reply with JSON only, shaped as {{"rules": [{{"conditions": [["?A", "relation", "anchor"], ["?B", "relation", "?A"]], "answer": "?B"}}]}}.

Available triples [subject, relation, object]:
{triples}

Question: {question}
Answer:"""


def atom(name):
    """'Arnold Schwarzenegger' -> 'arnold_schwarzenegger' (ONA-safe term)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s if s and s[0].isalpha() else f"e_{s}"


def parse_json(text):
    """First JSON object or list in the text, or None."""
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None


class OnaReasoner:
    def __init__(self, llm, model_id, ona_path="ONA/NAR", cycles=500, batch_size=None, timeout=120):
        self.llm, self.model_id = llm, model_id
        self.ona_path, self.cycles, self.timeout = ona_path, cycles, timeout
        self.batch_size = batch_size or batch_size_for(model_id)

    # ---------- 1. facts -> triples (LLM) ----------
    def triples(self, records):
        """Adds r["triples"]: [{"subject", "relation", "object", "f", "c", "fact"}, ...]. Each unique fact once."""
        _, tok = self.llm
        texts = list(dict.fromkeys(f["text"] for r in records for f in r["facts"]))
        raws = generate_batched(self.llm, [chat(tok, TRIPLES.format(fact=t)) for t in texts], max_new_tokens=160,
                                batch_size=self.batch_size, desc="[nars] facts -> triples")
        parsed = {}
        for t, raw in zip(texts, raws):
            out = parse_json(raw)
            out = out if isinstance(out, list) else [out] if isinstance(out, dict) else []
            parsed[t] = [x for x in out if isinstance(x, dict) and all(x.get(k) for k in ("subject", "relation", "object"))]
        for r in records:
            r["triples"] = [{"subject": x["subject"], "relation": x["relation"], "object": x["object"],
                             "f": f.get("f", f.get("p_true", 1.0)), "c": f.get("c", 0.5), "fact": f["text"]}
                            for f in r["facts"] for x in parsed.get(f["text"], [])]

    # ---------- 2. question -> rules (LLM) ----------
    def rules(self, records):
        """Adds r["rules_raw"] (LLM output) and r["rules"] (validated rules)."""
        _, tok = self.llm
        prompts = []
        for r in records:
            triples = [[t["subject"], atom(t["relation"]), t["object"]] for t in r["triples"]]
            prompts.append(chat(tok, RULES.format(question=r["question"],
                                                  triples=json.dumps(triples, ensure_ascii=False))))
        raws = generate_batched(self.llm, prompts, max_new_tokens=300, batch_size=max(1, self.batch_size // 2),
                                desc="[nars] question -> rules")
        for r, raw in zip(records, raws):
            r["rules_raw"] = raw
            out = parse_json(raw)
            r["rules"] = self._validate(out.get("rules", []) if isinstance(out, dict) else [], r["triples"])

    @staticmethod
    def _validate(rules, triples):
        """Keep rules whose relations and constants exist in the triples and whose answer variable is used."""
        rels = {atom(t["relation"]) for t in triples}
        ents = {atom(t["subject"]) for t in triples} | {atom(t["object"]) for t in triples}
        valid = []
        for rule in rules[:3]:
            conds, ans = rule.get("conditions") or [], str(rule.get("answer", ""))
            ok = bool(conds) and ans.startswith("?") and all(
                isinstance(c, list) and len(c) == 3 and atom(c[1]) in rels
                and all(str(x).startswith("?") or atom(x) in ents for x in (c[0], c[2])) for c in conds)
            if ok and any(ans in (c[0], c[2]) for c in conds):
                valid.append({"conditions": conds, "answer": ans})
        return valid

    # ---------- 3. Narsese program (code) ----------
    def program(self, r):
        """The full ONA input for one question, and the atom -> name map for reading the answer back."""
        names, lines = {}, ["*volume=0"]
        for t in r["triples"]:
            s, o, rel = atom(t["subject"]), atom(t["object"]), atom(t["relation"])
            names.setdefault(s, t["subject"])
            names.setdefault(o, t["object"])
            lines.append(f"<({s} * {o}) --> {rel}>. {{{t['f']:.3f} {t['c']:.3f}}}")
        for rule in r["rules"]:
            var = {}
            term = lambda x: f"${var.setdefault(x, 'abcdefgh'[len(var)])}" if str(x).startswith("?") else atom(x)
            conds = [f"<({term(c[0])} * {term(c[2])}) --> {atom(c[1])}>" for c in rule["conditions"]]
            body = conds[0]
            for c in conds[1:]:
                body = f"({body} && {c})"
            lines.append(f"<{body} ==> <{term(rule['answer'])} --> answer>>. {{1.0 0.9}}")
        lines += ["<?1 --> answer>?", str(self.cycles)]
        return "\n".join(lines) + "\n", names

    # ---------- 4. run ONA (code) ----------
    def ona(self, program):
        """Run ONA; the last 'Answer:' line -> (atom, f, c), or None."""
        out = subprocess.run([self.ona_path, "shell"], input=program, capture_output=True, text=True,
                             timeout=self.timeout).stdout
        best = None
        for line in out.splitlines():
            m = re.match(r"Answer: <(\S+) --> answer>\..*frequency=([\d.]+), confidence=([\d.]+)", line)
            if m:
                best = (m.group(1), float(m.group(2)), float(m.group(3)))
        return best

    # ---------- end to end ----------
    def run(self, records):
        self.triples(records)
        self.rules(records)
        for r in records:
            prog, names = self.program(r)
            res = None if not r["rules"] else self.ona(prog)
            status = "translation_error" if not r["rules"] else "no_answer" if res is None else "ok"
            a, f, c = res if res else (None, None, None)
            text = names.get(a, a.replace("_", " ") if a else None)
            r["nars"] = {"status": status, "answer_atom": a, "answer": text, "f": f, "c": c,
                         "e": c * (f - 0.5) + 0.5 if res else None,
                         "correct": bool(text) and is_correct(text, r["gold_answers"]), "program": prog}
        return records

    @staticmethod
    def summary(records):
        n = len(records)
        st = [r["nars"]["status"] for r in records]
        return {"questions": n, "ok": st.count("ok"), "no_answer": st.count("no_answer"),
                "translation_error": st.count("translation_error"),
                "correct": sum(r["nars"]["correct"] for r in records),
                "accuracy": round(sum(r["nars"]["correct"] for r in records) / max(1, n), 3),
                "triples_per_question": round(sum(len(r["triples"]) for r in records) / max(1, n), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="e.g. MuSiQue_final.pkl (with facts)")
    parser.add_argument("--model", required=True, help="formalizer, e.g. the 70B AWQ model")
    parser.add_argument("--ona", default="ONA/NAR")
    parser.add_argument("--cycles", type=int, default=500)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    records = pickle.load(open(args.input, "rb"))
    records = records[:args.n] if args.n else records
    start = time.time()
    llm = load_llm(args.model)
    nars = OnaReasoner(llm, args.model, args.ona, args.cycles)
    nars.run(records)
    unload_llm(llm)
    out_file = args.input.removesuffix(".pkl") + "_nars.pkl"
    pickle.dump(records, open(out_file, "wb"))
    entry = {"role": "nars", "model": args.model, "cycles": args.cycles, **OnaReasoner.summary(records),
             "elapsed_sec": round(time.time() - start, 1), "output_file": out_file}
    save_stats(records[0]["dataset"], entry)
    print(entry)


if __name__ == "__main__":
    main()
