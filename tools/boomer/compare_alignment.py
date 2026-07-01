#!/usr/bin/env python3
"""compare_alignment.py — compare HOnK's deterministic merge against Boomer's
probabilistic merge on the SAME candidate mappings (item 7 / reviewers R1-W2,
R2-W5).

HOnK applies every mapping in the ptable deterministically (source -> canonical,
e.g. Synonym -> eq). Boomer, given the same ptable + ontology, resolves which
alignments to accept probabilistically and emits them as owl class axioms
(``<SourceLabel> rdfs:subClassOf <Canonical>``). This script measures how much
of HOnK's deterministic mapping Boomer's probabilistic merge agrees with, and
where they differ.

The comparison is on local names, so it is namespace-agnostic: a Boomer solution
in the parmenides# namespace is directly comparable to a honk# ptable (the
relation/POS labels are identical).

Honest framing: Boomer is fed HOnK's own ptable, so this measures agreement with
a principled probabilistic merger given the same candidates, not against an
independent gold standard.

Usage (from tools/boomer):
    python compare_alignment.py --boomer boomer_output.nt --ptable mappings.tsv
"""

import argparse

SUBCLASS = "<http://www.w3.org/2000/01/rdf-schema#subClassOf>"
# Boomer's own grouping vocabulary / structural namespaces to skip.
SKIP_SUBJECT = ("DisjointSibling", "urn:unnamed", "www.w3.org")


def local(uri: str) -> str:
    u = uri.strip()
    if u.startswith("<") and u.endswith(">"):
        u = u[1:-1]
    if "#" in u:
        u = u.rsplit("#", 1)[1]
    elif "/" in u:
        u = u.rsplit("/", 1)[1]
    elif ":" in u:  # CURIE form (e.g. honk:a in the ptable)
        u = u.rsplit(":", 1)[1]
    return u.strip().lower()


def load_boomer(path):
    """Boomer's accepted alignments: real-class subClassOf real-class."""
    out = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if SUBCLASS not in line or not line.endswith("."):
                continue
            parts = line[:-1].strip().split(" ", 2)
            if len(parts) != 3:
                continue
            s, p, o = parts
            if p != SUBCLASS:
                continue
            if any(tok in s for tok in SKIP_SUBJECT) or "www.w3.org" in o:
                continue
            sl, ol = local(s), local(o)
            if sl and ol and sl != ol:
                out.add((sl, ol))
    return out


def load_ptable(path):
    """HOnK's deterministic mapping decisions: source -> target (per the ptable)."""
    out = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2 or not cols[0] or not cols[1]:
                continue
            sl, ol = local(cols[0]), local(cols[1])
            if sl and ol and sl != ol:
                out.add((sl, ol))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boomer", default="boomer_output.nt", help="Boomer output, N-Triples.")
    parser.add_argument("--ptable", default="mappings.tsv", help="HOnK ptable (mappings.tsv).")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    boomer = load_boomer(args.boomer)
    honk = load_ptable(args.ptable)

    shared = boomer & honk
    jaccard = 100.0 * len(shared) / len(boomer | honk) if (boomer | honk) else 0.0
    recall = 100.0 * len(shared) / len(honk) if honk else 0.0     # HOnK mappings Boomer accepts
    precision = 100.0 * len(shared) / len(boomer) if boomer else 0.0  # Boomer alignments in HOnK's ptable

    print("--- HOnK (deterministic) vs Boomer (probabilistic) on the shared ptable ---")
    print(f"HOnK ptable mappings:               {len(honk)}")
    print(f"Boomer accepted alignments:         {len(boomer)}")
    print(f"Shared:                             {len(shared)}")
    print(f"Jaccard:                            {jaccard:.2f}%")
    print(f"Boomer accepts of HOnK (recall):    {recall:.2f}%")
    print(f"Boomer alignments in HOnK (prec):   {precision:.2f}%")

    honk_only = sorted(honk - boomer)
    boomer_only = sorted(boomer - honk)
    print(f"\nHOnK maps, Boomer did NOT accept ({len(honk_only)}); showing {min(args.examples, len(honk_only))}:")
    for a, b in honk_only[:args.examples]:
        print(f"  {a} -> {b}")
    print(f"\nBoomer aligned, not in HOnK's ptable ({len(boomer_only)}); showing {min(args.examples, len(boomer_only))}:")
    for a, b in boomer_only[:args.examples]:
        print(f"  {a} -> {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
