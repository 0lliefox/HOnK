#!/usr/bin/env python3
"""decompose_type_classes.py — decompose a graph's distinct rdf:type classes.

The intrinsic-comparison tables report the number of distinct ``rdf:type``
classes instantiated in a serialised graph (the column historically labelled
``#POS``). Because relationships are reified, that count mixes several kinds
of class. This tool splits it, per graph, into:

  grammatical    lexical--grammatical classes from the ontology hierarchy
                 (GrammaticalFunction subtree of ontology_classes.json)
  penn-tag       Penn Treebank tags added by post-processing
                 (keys of pos_tag_mappings.json)
  pos-target     targets of the per-source POS mappings not already above
  relation       canonical relation classes (Property subtree)
  relation-raw   raw source relation labels (keys of *_edge_mappings.json;
                 present in unnormalised/raw builds)
  logical        Parmenides logical classes (LogicalFunction /
                 LogicalRewritingRule / Dependency subtrees)
  dimension      entity-typing classes (Dimensions subtree, e.g. LOC/GPE)
  UNKNOWN        anything else (raw source tags such as WordNet lexical
                 domains, or classes added outside ontology_classes.json)

Counts use the same filter as tools/ablation_stats.py: objects of rdf:type
triples, excluding owl# and rdf-schema# classes.

Usage (from the repo root):
    ./.venv/bin/python tools/decompose_type_classes.py ontologies/<graph>.nt [...]
    ./.venv/bin/python tools/decompose_type_classes.py --diff A.nt B.nt
"""

import argparse
import collections
import json
import os
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAPPINGS = os.path.join(ROOT, "supporting_files", "mappings")

RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"

CATEGORY_ORDER = ["grammatical", "penn-tag", "pos-target", "relation",
                  "relation-raw", "logical", "dimension", "bookkeeping",
                  "UNKNOWN"]


def _walk_hierarchy(node, root, branch, cat_of):
    if not isinstance(node, dict):
        return
    classes = node.get("classes", {}) if ("classes" in node or "properties" in node) else node
    for name, sub in classes.items():
        if root == "MetaGrammaticalFunction":
            b = branch or name
            cat = {"GrammaticalFunction": "grammatical",
                   "LogicalFunction": "logical",
                   "LogicalRewritingRule": "logical",
                   "Dependency": "logical"}.get(b, "UNKNOWN")
        elif root == "Property":
            cat = "relation"
        elif root == "Dimensions":
            cat = "dimension"
        elif root == "GraphParse":
            cat = "bookkeeping"
        else:
            cat = "UNKNOWN"
        cat_of.setdefault(name, cat)
        _walk_hierarchy(sub, root,
                        branch if root != "MetaGrammaticalFunction" else (branch or name),
                        cat_of)


def build_classifier():
    with open(os.path.join(MAPPINGS, "ontology_classes.json"), encoding="utf-8") as fh:
        hierarchy = json.load(fh)
    cat_of = {}
    for root, sub in hierarchy.items():
        _walk_hierarchy(sub, root, None, cat_of)

    with open(os.path.join(MAPPINGS, "pos_tag_mappings.json"), encoding="utf-8") as fh:
        penn = set(json.load(fh).keys())

    pos_targets = set()
    for name in ("cn_pos_mappings.json", "wk_pos_mappings.json", "wn_pos_mappings.json"):
        with open(os.path.join(MAPPINGS, name), encoding="utf-8") as fh:
            for value in json.load(fh).values():
                if isinstance(value, str):
                    pos_targets.add(value)
                elif isinstance(value, list):
                    pos_targets.update(v for v in value if isinstance(v, str))
                elif isinstance(value, dict):
                    for key in ("class", "pos", "target"):
                        if isinstance(value.get(key), str):
                            pos_targets.add(value[key])

    raw_edges = set()
    for name in ("cn_edge_mappings.json", "wk_edge_mappings.json", "wn_edge_mappings.json"):
        with open(os.path.join(MAPPINGS, name), encoding="utf-8") as fh:
            raw_edges.update(k.lower() for k in json.load(fh).keys())

    def classify(local):
        decoded = urllib.parse.unquote(local)
        if local in cat_of:
            return cat_of[local]
        if local in penn or decoded in penn:
            return "penn-tag"
        if local in pos_targets:
            return "pos-target"
        if local.lower() in raw_edges:
            return "relation-raw"
        return "UNKNOWN"

    return classify


def graph_classes(path):
    classes = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split(" ", 2)
            if len(parts) < 3 or parts[1] != RDF_TYPE:
                continue
            obj = parts[2].rstrip(" .\n")
            if obj.startswith("<") and "owl#" not in obj and "rdf-schema#" not in obj:
                classes.add(obj.rstrip(">").split("#")[-1].split("/")[-1])
    return classes


def report(path, classify):
    classes = graph_classes(path)
    counts = collections.Counter(classify(c) for c in classes)
    print(f"{os.path.basename(path)}: total={len(classes)} | "
          + " ".join(f"{k}={counts[k]}" for k in CATEGORY_ORDER if counts[k]))
    unknown = sorted(c for c in classes if classify(c) == "UNKNOWN")
    if unknown:
        print(f"  UNKNOWN: {unknown}")
    return classes


def main() -> int:
    ap = argparse.ArgumentParser(description="Decompose rdf:type class counts.")
    ap.add_argument("graphs", nargs="+", help="N-Triples graph files")
    ap.add_argument("--diff", action="store_true",
                    help="with exactly two graphs, also print the class-set diff")
    args = ap.parse_args()

    classify = build_classifier()
    sets = [report(p, classify) for p in args.graphs]

    if args.diff:
        if len(sets) != 2:
            print("--diff needs exactly two graphs", file=sys.stderr)
            return 1
        a, b = sets
        for label, extra in ((f"only in {os.path.basename(args.graphs[0])}", a - b),
                             (f"only in {os.path.basename(args.graphs[1])}", b - a)):
            print(f"{label} ({len(extra)}):")
            for c in sorted(extra):
                print(f"  {c} [{classify(c)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
