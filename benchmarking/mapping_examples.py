#!/usr/bin/env python3
"""
mapping_examples.py — show one real labelled example for each relation in the edge mapping files.

Reads directly from the raw source files — no database required.

Usage (from project root):
    python benchmarking/mapping_examples.py [--config config.yaml]

Covers:
  WordNet     — streams the .nt file; labels come from rdfs:label triples
  Wiktionary  — streams the JSON file; labels are the 'word' fields
  ConceptNet  — streams the CSV file; labels are the surface-form URI segments

Architecture note:
  Loaders store the RAW relation name (e.g. 'hyponyms', 'DerivedFrom').
  The actual mapping (renamed rel + swap) is applied in add_relation_to_graph()
  inside graph_funcs_oxi.py / graph_funcs_rdf.py, but ONLY when
  config['turtle_export']['normalise_pos'] is True.
  Each entry below shows both possible outcomes so the semantics can be judged.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_mappings(path: str) -> Dict:
    """Load a JSON mapping file; keys lowercased to match loader behaviour."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k.lower(): v for k, v in data.items()}
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return {}


def load_json_mappings_raw(path: str) -> Dict:
    """Load a JSON mapping file preserving original key casing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

W = 82
SEP  = "═" * W
THIN = "─" * W


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def show_entry(
    rel_name:   str,
    subj:       str,
    obj:        str,
    mapped_rel: str,
    swap:       object,   # raw value from JSON (bool, str "true", or None/absent)
) -> None:
    """
    Print one relation example with both possible outcomes.
    swap is applied by add_relation_to_graph when normalise_pos=True.
    """
    # Normalise swap to a bool for logic; keep original for display
    swap_bool = bool(swap) if swap is not None else False
    if isinstance(swap, str):
        swap_display = f'"{swap}"  ← string, not bool — type inconsistency'
    elif swap is None:
        swap_display = "(absent — defaults to False)"
    else:
        swap_display = repr(swap)

    print(f"\n  [{rel_name}]  →  {mapped_rel}   swap={swap_display}")
    print(f"  {THIN}")
    print(f"    Source data     :  {subj!r}  -[{rel_name}]->  {obj!r}")
    if swap_bool:
        print(f"    swap        :  {obj!r}  -[{mapped_rel}]->  {subj!r}"
              f"  ← subject/object swapped")
    else:
        print(f"    no swap     :  {subj!r}  -[{mapped_rel}]->  {obj!r}")


def no_example(rel_name: str, reason: str = "no English example found in file") -> None:
    print(f"\n  [{rel_name}]  — {reason}")


# ---------------------------------------------------------------------------
# WordNet  (two-pass streaming .nt scanner)
# ---------------------------------------------------------------------------

WNP   = "http://wordnet-rdf.princeton.edu/ontology#"
RDFS  = "http://www.w3.org/2000/01/rdf-schema#"
LABEL_PRED = f"<{RDFS}label>"


def _parse_nt_uri_triple(line: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse an N-Triples line of the form  <S> <P> <O> .
    Returns (subject_uri, pred_uri, object_uri) or None.
    """
    line = line.rstrip()
    if not line.startswith("<"):
        return None
    try:
        parts = line.split("> <")
        if len(parts) < 3:
            return None
        subj = parts[0][1:]                          # strip leading <
        pred = parts[1]
        obj  = parts[2].split(">")[0]               # strip trailing > .
        return subj, pred, obj
    except Exception:
        return None


def _extract_eng_label(line: str) -> Optional[str]:
    """Extract the English (@eng) label value from a rdfs:label triple line."""
    m = re.search(r'"((?:[^"\\]|\\.)+)"@eng', line)
    return m.group(1) if m else None


def wordnet_examples(wn_path: str, mappings: Dict, mappings_raw: Dict) -> None:
    section("WordNet")

    # Derive relation list from the mapping file (original case for predicate URI matching)
    suspect_rels = list(mappings_raw.keys())

    # Pre-build lookup: predicate_uri_string -> rel_name (original case)
    pred_to_rel: Dict[str, str] = {
        f"{WNP}{r}": r for r in suspect_rels
    }

    # ── Pass 1: find one example per relation ──────────────────────────────
    examples: Dict[str, Tuple[str, str]] = {}  # rel_name -> (subj_uri, obj_uri)

    print(f"\n  Scanning {wn_path} for examples…", flush=True)
    try:
        with open(wn_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = _parse_nt_uri_triple(line)
                if parsed is None:
                    continue
                subj, pred, obj = parsed
                if pred in pred_to_rel and subj != obj:
                    rel_name = pred_to_rel[pred]
                    if rel_name not in examples:
                        examples[rel_name] = (subj, obj)
                if len(examples) == len(suspect_rels):
                    break
    except FileNotFoundError:
        print(f"  ERROR: WordNet file not found at {wn_path!r}")
        return

    if not examples:
        print("  No relation triples found.")
        return

    # ── Pass 2: find English labels for all needed URIs ────────────────────
    needed: set = {uri for pair in examples.values() for uri in pair}
    uri_labels: Dict[str, str] = {}

    with open(wn_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if LABEL_PRED not in line:
                continue
            # Subject URI is up to the first >
            close = line.find(">")
            if close == -1:
                continue
            subj = line[1:close]
            if subj in needed and subj not in uri_labels:
                label = _extract_eng_label(line)
                if label:
                    uri_labels[subj] = label
                    if uri_labels.keys() >= needed:
                        break

    # ── Display ────────────────────────────────────────────────────────────
    for rel_name in suspect_rels:
        mapping = mappings.get(rel_name.lower())
        if mapping is None:
            no_example(rel_name, "not in mapping file")
            continue
        mapped_rel = mapping.get("rel", "?")
        swap       = mapping.get("swap", None)

        if rel_name not in examples:
            no_example(rel_name)
            continue

        subj_uri, obj_uri = examples[rel_name]
        subj_label = uri_labels.get(subj_uri, subj_uri.split("/")[-1])
        obj_label  = uri_labels.get(obj_uri,  obj_uri.split("/")[-1])

        if subj_label == obj_label:
            no_example(rel_name, "only self-loop examples found in file")
            continue

        show_entry(rel_name, subj_label, obj_label, mapped_rel, swap)


# ---------------------------------------------------------------------------
# Wiktionary  (JSON streaming scan; swap flag IGNORED by loader)
# ---------------------------------------------------------------------------

def wiktionary_examples(wk_path: str, mappings: Dict, mappings_raw: Dict) -> None:
    section("Wiktionary")

    # Derive relation list from the mapping file (original case = field names in entries)
    suspect_rels = list(mappings_raw.keys())
    found: Dict[str, Optional[Tuple[str, str]]] = {r: None for r in suspect_rels}

    print(f"\n  Scanning {wk_path} for examples…", flush=True)
    try:
        with open(wk_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        print(f"  ERROR: Wiktionary file not found at {wk_path!r}")
        return
    except Exception as e:
        print(f"  ERROR loading Wiktionary: {e}")
        return

    for entry in entries:
        if entry.get("lang_code") != "en":
            continue
        word = entry.get("word", "")
        for rel_name in suspect_rels:
            if found[rel_name] is not None:
                continue
            items = entry.get(rel_name, [])
            if not items:
                continue
            for item in items:
                linked = item.get("word") if isinstance(item, dict) else (
                    item if isinstance(item, str) else None
                )
                if linked and linked != word:
                    found[rel_name] = (word, linked)
                    break
        if all(v is not None for v in found.values()):
            break

    for rel_name in suspect_rels:
        mapping = mappings.get(rel_name.lower())
        if mapping is None:
            no_example(rel_name, "not in mapping file")
            continue
        mapped_rel = mapping.get("rel", "?")
        swap       = mapping.get("swap", None)

        if found[rel_name] is None:
            no_example(rel_name)
            continue

        subj, obj = found[rel_name]
        show_entry(rel_name, subj, obj, mapped_rel, swap)


# ---------------------------------------------------------------------------
# ConceptNet  (TSV streaming scan)
# ---------------------------------------------------------------------------

def _cn_surface(uri_with_spaces: str) -> str:
    """Extract the surface-form segment from a ConceptNet URI (already _ -> space)."""
    parts = uri_with_spaces.split("/")
    return parts[3] if len(parts) > 3 else parts[-1]


def _cn_lang(uri_with_spaces: str) -> str:
    parts = uri_with_spaces.split("/")
    return parts[2] if len(parts) > 2 else ""


def conceptnet_examples(cn_path: str, mappings: Dict, mappings_raw: Dict) -> None:
    section("ConceptNet")

    # Derive relation list from the mapping file (original case matches /r/RelName in TSV)
    suspect_rels = list(mappings_raw.keys())
    found: Dict[str, Optional[Tuple[str, str]]] = {r: None for r in suspect_rels}

    print(f"\n  Scanning {cn_path} for examples…", flush=True)
    try:
        with open(cn_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 4:
                    continue
                rel_raw = row[1].replace("/r/", "")
                if rel_raw not in found or found[rel_raw] is not None:
                    continue

                start = row[2].replace("_", " ")
                end   = row[3].replace("_", " ")

                if _cn_lang(start) != "en" or _cn_lang(end) != "en":
                    continue

                surf_s = _cn_surface(start)
                surf_e = _cn_surface(end)

                # Skip numeric / empty surface forms for readability
                if surf_s and surf_e and surf_s != surf_e and not surf_s[0].isdigit() and not surf_e[0].isdigit():
                    found[rel_raw] = (surf_s, surf_e)

                if all(v is not None for v in found.values()):
                    break
    except FileNotFoundError:
        print(f"  ERROR: ConceptNet file not found at {cn_path!r}")
        return

    for rel_name in suspect_rels:
        mapping = mappings.get(rel_name.lower())
        if mapping is None:
            no_example(rel_name, "not in mapping file")
            continue
        mapped_rel = mapping.get("rel", "?")
        swap       = mapping.get("swap", None)

        if found[rel_name] is None:
            no_example(rel_name)
            continue

        subj, obj = found[rel_name]
        show_entry(rel_name, subj, obj, mapped_rel, swap)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show one labelled example per relation in the edge mapping files."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        for candidate in ["config.yaml", "../config.yaml"]:
            if Path(candidate).is_file():
                config_path = candidate
                break
    if config_path is None:
        print("ERROR: config.yaml not found.", file=sys.stderr)
        sys.exit(1)

    cfg   = load_config(config_path)
    files = cfg.get("local_files", {})

    wn_mappings     = load_json_mappings(files.get("wordnet_edge_mappings_file", ""))
    wk_mappings     = load_json_mappings(files.get("wiktionary_edge_mappings_file", ""))
    cn_mappings     = load_json_mappings(files.get("conceptnet_edge_mappings_file", ""))
    wn_mappings_raw = load_json_mappings_raw(files.get("wordnet_edge_mappings_file", ""))
    wk_mappings_raw = load_json_mappings_raw(files.get("wiktionary_edge_mappings_file", ""))
    cn_mappings_raw = load_json_mappings_raw(files.get("conceptnet_edge_mappings_file", ""))

    print(f"\n{'═' * W}")
    print("  HOnK — Edge-Mapping Examples")
    print("  Mapping + swap applied in add_relation_to_graph when normalise_pos=True.")
    print("  Each entry shows: raw source data / result without swap / result with swap.")
    print(f"{'═' * W}")

    wordnet_examples(files.get("wordnet", ""), wn_mappings, wn_mappings_raw)
    wiktionary_examples(files.get("wiktionary", ""), wk_mappings, wk_mappings_raw)
    conceptnet_examples(files.get("conceptnet", ""), cn_mappings, cn_mappings_raw)

    print(f"\n{SEP}\nDone.\n")


if __name__ == "__main__":
    main()
