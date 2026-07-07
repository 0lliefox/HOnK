"""Linguistic/logical soundness guards for the source->HOnK mapping files.

These assert the Fix E corrections and, more importantly, that every POS-class
and relation target used by a mapping actually exists in ontology_classes.json
(catches typos and drift). The edge_mapping fidelity stratum only ever sampled
AtLocation, so these invariants were previously unchecked.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = os.path.join(HERE, os.pardir, "supporting_files", "mappings")


def _load(name):
    with open(os.path.join(MAP, name), encoding="utf-8") as fh:
        return json.load(fh)


# Keys in ontology_classes.json that are structural, not class/relation names.
_STRUCTURAL_KEYS = {"properties", "classes", "sameAs"}


def _collect_names(node, acc):
    """Recursively gather every class/relation name in ontology_classes.json.

    The tree is irregular (some children nest under a "classes" dict, some are
    inline), so we walk all dict keys and skip the structural ones. List-valued
    "properties" entries are attribute names, not classes, so they are ignored.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _STRUCTURAL_KEYS:
                if key == "classes":
                    _collect_names(value, acc)
                continue
            acc.add(key)
            _collect_names(value, acc)


def _ontology_names():
    acc = set()
    _collect_names(_load("ontology_classes.json"), acc)
    return acc


def test_wiktionary_name_maps_to_proper_noun():
    # Fix E: a proper-noun POS tag must not be typed as a Pronoun.
    assert _load("wk_pos_mappings.json")["name"] == "ProperNoun"


def test_instanceof_relations_are_harmonised():
    # Fix E: all instance-of relations preserve the instance/subclass distinction.
    cn = _load("cn_edge_mappings.json")
    wn = _load("wn_edge_mappings.json")
    assert cn["InstanceOf"]["rel"] == "instanceOf"
    assert wn["instance_hypernym"]["rel"] == "instanceOf"
    assert wn["instance_hyponym"]["rel"] == "instanceOf"


def test_all_pos_mapping_targets_exist_in_ontology():
    names = _ontology_names()
    missing = {}
    for f in ("cn_pos_mappings.json", "wn_pos_mappings.json", "wk_pos_mappings.json"):
        for src, target in _load(f).items():
            if target not in names:
                missing.setdefault(f, []).append(f"{src} -> {target}")
    assert not missing, f"POS targets absent from ontology_classes.json: {missing}"


def test_all_edge_mapping_targets_exist_in_ontology():
    names = _ontology_names()
    missing = {}
    for f in ("cn_edge_mappings.json", "wn_edge_mappings.json", "wk_edge_mappings.json"):
        for src, spec in _load(f).items():
            target = spec["rel"]
            if target not in names:
                missing.setdefault(f, []).append(f"{src} -> {target}")
    assert not missing, f"Relation targets absent from ontology_classes.json: {missing}"
