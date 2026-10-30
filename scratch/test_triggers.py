import json
import pickle
import re
import unicodedata
from pathlib import Path
import ONA

def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\b(a|an|the)\b", " ", text))

ABSTAIN_RE = re.compile(
    r"\b(not contain|no information|not mention|not state|not provide|cannot|can't|don't have|do not have|unknown|none|no answer|not specified|does not say|unable to|misunderstanding|not include)\b",
    re.I,
)

def get_intermediate_entities(r, err_traces):
    """Return normalized names of intermediate (non-answer) variables bound in NARS rules."""
    ans_norm = normalize(r.get("answer") or "")
    if not ans_norm:
        return set()
    inter = set()
    err = err_traces.get(r["index"])
    rules = r.get("rules") or []
    ans_vars = {rule.get("answer") for rule in rules if rule.get("answer")}
    if err and err.get("binding_chains"):
        for chain in err["binding_chains"]:
            for part in chain.split(", "):
                if " = " in part:
                    v, val = part.split(" = ", 1)
                    if v.strip() not in ans_vars and normalize(val) != ans_norm:
                        inter.add(normalize(val))
    # Also from rule conditions constants or matched proof triples in trace_text / error_traces
    if err and err.get("proof_lines"):
        # every entity in proof_lines except the final answer and question constants
        for line in err["proof_lines"][:-1]:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                for ent in (parts[0], parts[2]):
                    en = normalize(ent)
                    if en and en != ans_norm:
                        inter.add(en)
    return inter

for ds in ["MuSiQue", "HotpotQA", "2Wiki"]:
    run_dir = Path(f"runs/{ds}_2hop_100q_seed0_plan_votes3_merge_relations")
    if not run_dir.exists():
        continue
    nars = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "nars.jsonl").read_text().splitlines() if l.strip())}
    base = {r["index"]: r for r in (json.loads(l) for l in (run_dir / "baselines.jsonl").read_text().splitlines() if l.strip())}
    # parse error_nars.txt
    err_path = run_dir / "error_nars.txt"
    err_traces = {}
    if err_path.exists():
        for block in err_path.read_text().split("=" * 100):
            m_q = re.search(r"^Q(\d+):", block.strip(), flags=re.M)
            if not m_q:
                continue
            idx = int(m_q.group(1))
            names, fact_set = {}, set()
            m_triples = re.search(r"-- Triples after aliases and inverses\n((?:   .+\n)+)", block)
            if m_triples:
                for line in m_triples.group(1).splitlines():
                    clean = re.sub(r"\s*\(inverse\)\s*$", "", line.strip())
                    parts_t = [p.strip() for p in clean.split("|")]
                    if len(parts_t) == 3:
                        s, rel, o = parts_t
                        sa, ra, oa = ONA.atom(s), ONA.atom(rel), ONA.atom(o)
                        names.setdefault(sa, s)
                        names.setdefault(oa, o)
                        fact_set.add((sa, ra, oa))
            err_traces[idx] = {"names": names, "fact_set": fact_set}

    print(f"================ {ds} ================")
    for i, r in sorted(nars.items()):
        if not r.get("answer"):
            continue
        # check if any model in base had a Hop-1 answer or abstention or unit strip
        for m, b_rec in base[i]["answers"].items():
            b_ans = b_rec["answer"] or ""
            n_ans = r["answer"] or ""
            nb, nn = normalize(b_ans), normalize(n_ans)
            # unit strip check: nb is digits and nn starts with nb and has letters
            if nb and nb != nn and nb.isdigit() and nn.startswith(nb) and any(c.isalpha() for c in nn):
                print(f"  UnitRestore Q{i} ({m}): {b_ans!r} -> {n_ans!r} (nars_J={r['judge']}, base_J={b_rec['judge']})")
