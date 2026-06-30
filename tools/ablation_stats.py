#!/usr/bin/env python3
"""Per-arm intrinsic stats for the method-vs-data ablation (item 1).

Streams each N-Triples build once and reports triples, distinct nodes, distinct
relations (predicates), distinct rdf:type classes (POS richness), density and
average degree. Complements benchmarking/compare_graphs.py (which also does
pairwise overlap + navigability); this one is a fast single-pass per-arm
summary for the decomposition table.

Usage:  python tools/ablation_stats.py [file1.nt name1] [file2.nt name2] ...
        (no args -> the four ablation arms under ontologies/)
"""

import os
import re
import sys

RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
# HOnK reifies each relation occurrence as honk#<Type><N> (Desires1, Desires2,
# DistinctFrom100...). Strip the trailing index to recover the relation *type*
# so we report relation diversity rather than the reified-instance count.
REIF_INDEX = re.compile(r"\d+>$")


def _base_relation(predicate):
    if "honk#" in predicate:
        return REIF_INDEX.sub(">", predicate)
    return predicate


def arm_stats(path):
    nodes = set()
    rel_types = set()
    type_classes = set()
    triples = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            body = line.rstrip()
            if not body.endswith("."):
                continue
            body = body[:-1].strip()
            parts = body.split(" ", 2)
            if len(parts) != 3:
                continue
            s, p, o = parts
            triples += 1
            nodes.add(s)
            nodes.add(o)
            if "honk#" in p:
                rel_types.add(_base_relation(p))
            if p == RDF_TYPE and o.startswith("<") and "owl#" not in o and "rdf-schema#" not in o:
                type_classes.add(o)
    n = len(nodes)
    density = triples / (n * (n - 1)) if n > 1 else 0.0
    degree = (2 * triples) / n if n else 0.0
    return {"triples": triples, "nodes": n, "relations": len(rel_types),
            "type_classes": len(type_classes), "density": density, "degree": degree}


def main():
    default = [
        ("ontologies/ablation_a_conceptnet_raw.nt", "a_CN_raw"),
        ("ontologies/ablation_b1_union_raw.nt", "b1_union_raw"),
        ("ontologies/ablation_b2_union_canonicalise.nt", "b2_union_canon"),
        ("ontologies/ablation_c_full_unclustered.nt", "c_full_norm"),
    ]
    args = sys.argv[1:]
    pairs = default if not args else [(args[i], args[i + 1]) for i in range(0, len(args), 2)]

    hdr = f"{'arm':16} {'triples':>12} {'nodes':>12} {'rels':>5} {'types':>6} {'density':>12} {'degree':>7}"
    print(hdr)
    print("-" * len(hdr))
    for path, name in pairs:
        if not os.path.exists(path):
            print(f"{name:16} MISSING ({path})")
            continue
        st = arm_stats(path)
        print(f"{name:16} {st['triples']:>12,} {st['nodes']:>12,} {st['relations']:>5} "
              f"{st['type_classes']:>6} {st['density']:>12.3e} {st['degree']:>7.3f}")


if __name__ == "__main__":
    main()
