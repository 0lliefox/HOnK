#!/usr/bin/env python3
"""match_baseline.py — transparent automated matchers for the "match stage"
experiment: can an off-the-shelf matcher recover HOnK's manual schema mappings?

Two matchers, both aligning every source class to its best canonical class:
  lexical   : string similarity (char ratio + token Jaccard) on the labels.
  embedding : cosine similarity between the source label and the canonical
              label+gloss, using all-MiniLM-L6-v2 (the model already used in
              the downstream evaluation).

Each source label has exactly one gold canonical target, so the natural metric
is top-1 recovery: for how many source labels does the matcher's best-ranked
canonical equal the manually-chosen one. We report it overall, on the
non-identity subset (the correspondences that require real semantic bridging
rather than a trivial string identity), and broken down by the hard curator
cases (POS granularity, argument swaps, negations).

Usage (from tools/boomer, main .venv):
    python match_baseline.py [--top-k 3]
"""

import argparse
import csv
import difflib
import os
import re
from collections import defaultdict

from rdflib import Graph, RDFS

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH = os.path.join(HERE, "match")


def load_classes(path):
    g = Graph(); g.parse(path)
    out = {}
    for s, _, lbl in g.triples((None, RDFS.label, None)):
        out.setdefault(str(s), {})["label"] = str(lbl)
    for s, _, com in g.triples((None, RDFS.comment, None)):
        out.setdefault(str(s), {})["gloss"] = str(com)
    return out


def load_reference(path):
    gold, flags = {}, {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            gold[row["source_iri"]] = row["canonical_iri"]
            flags[row["source_iri"]] = row
    return gold, flags


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_sim(a, b):
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    ta, tb = tokens(a), tokens(b)
    jacc = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return max(ratio, jacc)


def rank_lexical(sources, canon, top_k):
    preds = {}
    citems = list(canon.items())
    for s_iri, s in sources.items():
        scored = sorted(
            ((lexical_sim(s["label"], c["label"]), c_iri) for c_iri, c in citems),
            reverse=True)
        preds[s_iri] = [(iri, sc) for sc, iri in scored[:top_k]]
    return preds


def rank_embedding(sources, canon, top_k):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    s_iris = list(sources)
    c_iris = list(canon)
    s_text = [sources[i]["label"] for i in s_iris]
    c_text = [f'{canon[i]["label"]}: {canon[i].get("gloss","")}' for i in c_iris]
    se = model.encode(s_text, convert_to_tensor=True, normalize_embeddings=True)
    ce = model.encode(c_text, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(se, ce)
    preds = {}
    for i, s_iri in enumerate(s_iris):
        row = sim[i]
        order = sorted(range(len(c_iris)), key=lambda j: float(row[j]), reverse=True)
        preds[s_iri] = [(c_iris[j], float(row[j])) for j in order[:top_k]]
    return preds


def score(preds, gold, flags, name):
    subsets = defaultdict(lambda: [0, 0])   # key -> [correct, total]
    top3 = [0, 0]
    misses = []
    for s_iri, ranked in preds.items():
        best = ranked[0][0]
        correct = int(best == gold[s_iri])
        in3 = int(any(iri == gold[s_iri] for iri, _ in ranked))
        fl = flags[s_iri]
        for key in ("all", fl["class"], fl["kind"]):
            subsets[key][0] += correct; subsets[key][1] += 1
        if fl["swap"]:
            subsets["swap"][0] += correct; subsets["swap"][1] += 1
        if fl["neg"]:
            subsets["neg"][0] += correct; subsets["neg"][1] += 1
        top3[0] += in3; top3[1] += 1
        if not correct and fl["class"] == "nonidentity":
            misses.append((s_iri.split("#")[-1], gold[s_iri].split("#")[-1], best.split("#")[-1]))

    def pct(k):
        c, t = subsets[k]
        return f"{100*c/t:5.1f}% ({c}/{t})" if t else "  n/a"

    print(f"\n=== {name} ===")
    print(f"  overall top-1 recovery      : {pct('all')}")
    print(f"  non-identity top-1 recovery : {pct('nonidentity')}")
    print(f"  identity (trivial) top-1    : {pct('identity')}")
    print(f"  edge-relation mappings      : {pct('edge')}")
    print(f"  POS mappings                : {pct('pos')}")
    print(f"  argument-swap cases         : {pct('swap')}")
    print(f"  negation cases              : {pct('neg')}")
    print(f"  overall top-3 recovery      : {100*top3[0]/top3[1]:5.1f}% ({top3[0]}/{top3[1]})")
    return subsets, misses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    sources = load_classes(os.path.join(MATCH, "source.owl"))
    canon = load_classes(os.path.join(MATCH, "canonical.owl"))
    gold, flags = load_reference(os.path.join(MATCH, "reference.tsv"))
    print(f"sources={len(sources)} canon={len(canon)} gold={len(gold)}")

    lex = rank_lexical(sources, canon, args.top_k)
    emb = rank_embedding(sources, canon, args.top_k)
    _, lex_miss = score(lex, gold, flags, "Lexical baseline (label string similarity)")
    _, emb_miss = score(emb, gold, flags, "Embedding baseline (MiniLM, label+gloss)")

    with open(os.path.join(MATCH, "baseline_predictions.tsv"), "w", encoding="utf-8") as fh:
        fh.write("matcher\tsource\tgold\tpredicted\tcorrect\n")
        for name, preds in (("lexical", lex), ("embedding", emb)):
            for s_iri, ranked in preds.items():
                best = ranked[0][0]
                fh.write(f'{name}\t{s_iri.split("#")[-1]}\t{gold[s_iri].split("#")[-1]}'
                         f'\t{best.split("#")[-1]}\t{int(best==gold[s_iri])}\n')

    print(f"\nEmbedding non-identity misses (first 20 of {len(emb_miss)}):")
    for s, g, p in emb_miss[:20]:
        print(f"   {s:26s} gold={g:16s} got={p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
