"""Validate the manual source->HOnK mappings against the canonical ontology.

Each cross-source mapping file aligns a knowledge-base label with a HOnK term:
  * ``*_edge_mappings.json``  map a source relation -> a HOnK ``Property`` relation
  * ``*_pos_mappings.json``   map a source POS tag  -> a HOnK grammatical class

This script checks that every mapped *target* still exists in
``ontology_classes.json``. A target that no longer resolves is a "dangling"
mapping: it will silently degrade the graph if the ontology is renamed, which is
exactly the maintenance fragility reviewers asked us to quantify (item 6).

Usage:
    python tools/validate_mappings.py [--mappings-dir DIR]

Exit code is non-zero if any dangling targets are found, so it can gate CI.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Set, Tuple

# Keys inside ontology_classes.json that are structural, not class names.
RESERVED_KEYS = {"properties", "classes", "sameAs"}

EDGE_FILES = ["cn_edge_mappings.json", "wk_edge_mappings.json", "wn_edge_mappings.json"]
POS_FILES = ["cn_pos_mappings.json", "wk_pos_mappings.json", "wn_pos_mappings.json"]


def _collect_class_names(node: dict, acc: Set[str]) -> None:
    """Recursively gather every class name from a (possibly irregular) subtree.

    The hierarchy mixes two shapes: children under an explicit ``"classes"`` dict,
    and children placed directly as sibling keys. Both are handled here.
    """
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key in RESERVED_KEYS:
            if key == "classes" and isinstance(value, dict):
                _collect_class_names(value, acc)
            continue
        acc.add(key)
        if isinstance(value, dict):
            _collect_class_names(value, acc)


def load_ontology_terms(mappings_dir: str) -> Tuple[Set[str], Set[str]]:
    """Return (grammatical_class_names, property_relation_names)."""
    with open(os.path.join(mappings_dir, "ontology_classes.json"), encoding="utf-8") as handle:
        ontology = json.load(handle)

    gram_classes: Set[str] = set()
    _collect_class_names(ontology.get("MetaGrammaticalFunction", {}), gram_classes)

    properties: Set[str] = set(
        ontology.get("Property", {}).get("classes", {}).keys()
    )
    return gram_classes, properties


def _edge_target(value) -> str:
    """Edge values are either a bare string or a dict carrying ``rel``."""
    if isinstance(value, dict):
        return str(value.get("rel", "")).strip()
    return str(value).strip()


def validate(mappings_dir: str) -> Tuple[List[dict], int]:
    """Validate every mapping; return (dangling_rows, total_rules)."""
    gram_classes, properties = load_ontology_terms(mappings_dir)
    dangling: List[dict] = []
    total = 0

    for filename in EDGE_FILES + POS_FILES:
        path = os.path.join(mappings_dir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            mappings: Dict[str, object] = json.load(handle)

        is_edge = filename.endswith("_edge_mappings.json")
        valid_targets = properties if is_edge else gram_classes
        for source_label, value in mappings.items():
            total += 1
            target = _edge_target(value) if is_edge else str(value).strip()
            if target not in valid_targets:
                dangling.append(
                    {
                        "file": filename,
                        "source_label": source_label,
                        "target": target,
                        "kind": "relation" if is_edge else "class",
                    }
                )
    return dangling, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(here, "..", "supporting_files", "mappings"))
    parser.add_argument("--mappings-dir", default=default_dir)
    args = parser.parse_args()

    dangling, total = validate(args.mappings_dir)
    resolved = total - len(dangling)
    print(f"Checked {total} cross-source mapping rules in {args.mappings_dir}")
    print(f"Resolved: {resolved}/{total} ({100.0 * resolved / total:.1f}%)")
    if dangling:
        print(f"\nDANGLING TARGETS ({len(dangling)}):")
        for row in dangling:
            print(f"  [{row['file']}] {row['source_label']!r} -> {row['target']!r} ({row['kind']} not in ontology)")
        return 1
    print("All mapping targets resolve against ontology_classes.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
