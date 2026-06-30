#!/usr/bin/env python3
"""compare_alignment.py — compare HOnK's deterministic merge against Boomer's
probabilistic ontology merge on their shared alignment axioms (item 7 /
reviewers R1-W2, R2-W5).

Both inputs must be N-Triples (convert the Boomer .ofn with
`robot convert --format nt`). Each graph is reduced to a canonical set of
alignment triples (subject local-name, canonical relation, object local-name);
symmetric relations (eq, neq) are stored order-independently, so the two merges
are compared on a common vocabulary rather than on raw, differently-namespaced
triples. Streaming line parsing keeps memory flat for the large HOnK output.

Honest framing for the paper: Boomer is fed HOnK's own candidate ptable, so this
measures *agreement with a principled probabilistic merger given the same
candidate mappings*, not agreement with an independent gold standard.

Usage (from tools/boomer):
    python compare_alignment.py --boomer boomer_output.nt \
        --honk ../../ontologies/ontology_db_oxi_clustered_final.nt
"""

import argparse

# Symmetric relations are compared without regard to subject/object order.
SYMMETRIC = {"eq", "neq"}

# Canonical relation -> lowercase substrings of the predicate URI that denote it.
PRED_TOKENS = [
    ("eq", ["#eq>", "equivalentclass", "sameas"]),
    ("neq", ["neqto", "disjointwith", "antonym"]),
    ("isa", ["#isa>", "subclassof", "instanceof", "#type>"]),
    ("partof", ["partof"]),
    ("formof", ["formof"]),
]


def canon_pred(predicate: str):
    pl = predicate.lower()
    for canon, tokens in PRED_TOKENS:
        if any(token in pl for token in tokens):
            return canon
    return None


def local(uri: str) -> str:
    u = uri.strip()
    if u.startswith("<") and u.endswith(">"):
        u = u[1:-1]
    if "#" in u:
        u = u.rsplit("#", 1)[1]
    elif "/" in u:
        u = u.rsplit("/", 1)[1]
    return u.strip().lower()


def extract_alignment(path: str):
    """Stream N-Triples and return the canonical alignment-triple set."""
    out = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or not line.endswith("."):
                continue
            parts = line[:-1].strip().split(None, 2)
            if len(parts) != 3:
                continue
            subj, pred, obj = parts
            canon = canon_pred(pred)
            if canon is None:
                continue
            sl, ol = local(subj), local(obj)
            if not sl or not ol or sl == ol:
                continue
            if canon in SYMMETRIC and sl > ol:
                sl, ol = ol, sl
            out.add((sl, canon, ol))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boomer", default="boomer_output.nt", help="Boomer output, N-Triples.")
    parser.add_argument("--honk", default="../../ontologies/ontology_db_oxi_clustered_final.nt",
                        help="HOnK output, N-Triples.")
    parser.add_argument("--examples", type=int, default=10, help="Disagreement examples to print.")
    args = parser.parse_args()

    boomer = extract_alignment(args.boomer)
    honk = extract_alignment(args.honk)

    inter = boomer & honk
    union = boomer | honk
    jaccard = 100.0 * len(inter) / len(union) if union else 0.0
    precision = 100.0 * len(inter) / len(honk) if honk else 0.0    # HOnK axioms Boomer also has
    recall = 100.0 * len(inter) / len(boomer) if boomer else 0.0   # Boomer axioms HOnK also has

    print("--- Alignment-axiom comparison (HOnK vs Boomer) ---")
    print(f"HOnK alignment axioms:            {len(honk)}")
    print(f"Boomer alignment axioms:          {len(boomer)}")
    print(f"Shared:                           {len(inter)}")
    print(f"Jaccard:                          {jaccard:.2f}%")
    print(f"Agreement with Boomer (recall):   {recall:.2f}%")
    print(f"HOnK axioms Boomer agrees (prec):  {precision:.2f}%")

    only_boomer = sorted(boomer - honk)
    only_honk = sorted(honk - boomer)
    print(f"\nIn Boomer, not HOnK ({len(only_boomer)}); showing {min(args.examples, len(only_boomer))}:")
    for triple in only_boomer[:args.examples]:
        print(f"  {triple[0]} -[{triple[1]}]- {triple[2]}")
    print(f"\nIn HOnK, not Boomer ({len(only_honk)}); showing {min(args.examples, len(only_honk))}:")
    for triple in only_honk[:args.examples]:
        print(f"  {triple[0]} -[{triple[1]}]- {triple[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
