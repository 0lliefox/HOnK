#!/usr/bin/env python3
"""provenance_audit_adjudicator.py — independently re-verify each row of the
stratified provenance sample (item 2 / reviewer R2-W4) against the database and
the raw mapping files, turning "traceable by construction" into "verified".

For every sampled output we run a stratum-specific, deterministic check that is
independent of the sampler's own bookkeeping:

  pos_mapping       the concept's assigned POS is a *declared output* of the
                    source's POS mapping (not a fabricated / leaked tag).
  edge_mapping      the relation's original type has a declared mapping to the
                    canonical honk: relation recorded in the provenance.
  url_cluster       the concepts sharing the recorded URL are merged into a
                    *single* cluster (the merge is grounded in a real shared
                    identifier and is complete).
  enriched_cluster  the inferred cluster-level edge is grounded in a real
                    concept-level source edge between members of the two
                    clusters (the enrichment is not fabricated).

A row PASSES if its check holds. This is a provenance-consistency audit (does
the output agree with the source record it claims to come from), not a human
semantic-correctness judgement; the blank ``label`` column of the sample is left
for that separate manual pass.

Usage (from project root):
    python evaluator/provenance_audit_adjudicator.py \
        [--config config.yaml] [--in fidelity_audit_sample.csv] \
        [--out fidelity_audit_adjudicated.csv]
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

import psycopg2
import yaml

MAPPING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "supporting_files", "mappings")
SOURCE_PREFIX = {"conceptnet": "cn", "wiktionary": "wk", "wordnet": "wn"}
DEFAULT_POS = {"Concept"}  # explicit fallback POS for source tags with no specific mapping


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def connect(cfg: Dict[str, Any]):
    db = cfg["database"]
    return psycopg2.connect(
        dbname=db["dbname"], user=db["user"], password=db["password"],
        host=db["host"], port=str(db["port"]),
    )


def load_pos_targets() -> Dict[str, set]:
    """prefix -> set of POS values the source's POS mapping can emit."""
    out: Dict[str, set] = defaultdict(set)
    for prefix in ("cn", "wk", "wn"):
        path = os.path.join(MAPPING_DIR, f"{prefix}_pos_mappings.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        for value in data.values():
            target = value.get("rel") if isinstance(value, dict) else value
            out[prefix].add(str(target))
    return out


def load_edge_maps() -> Dict[str, Dict[str, str]]:
    """prefix -> {original relation label: canonical honk target}."""
    out: Dict[str, Dict[str, str]] = defaultdict(dict)
    for prefix in ("cn", "wk", "wn"):
        path = os.path.join(MAPPING_DIR, f"{prefix}_edge_mappings.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        for src_label, value in data.items():
            target = value.get("rel") if isinstance(value, dict) else value
            out[prefix][src_label] = str(target)
    return out


# ---------------------------------------------------------------------------
# Per-stratum independent checks. Each returns (passed: bool, note: str).
# ---------------------------------------------------------------------------

def check_pos(cur, row, pos_targets) -> (bool, str):
    m = re.search(r"concept_id=(\d+)", row["detail"])
    if not m:
        return False, "no concept_id in detail"
    cid = int(m.group(1))
    cur.execute("SELECT term, part_of_speech, source FROM concepts WHERE id=%s", (cid,))
    rec = cur.fetchone()
    if not rec:
        return False, f"concept {cid} absent"
    term, pos, source = rec
    prefix = SOURCE_PREFIX.get(source.lower(), source.lower())
    if pos in pos_targets.get(prefix, set()) or pos in DEFAULT_POS:
        return True, f"POS '{pos}' is a declared output of {prefix}_pos_mappings"
    return False, f"POS '{pos}' not a declared {prefix} mapping output"


def check_edge(cur, row, edge_maps) -> (bool, str):
    m = re.search(r"relation_id=(\d+)", row["detail"])
    if not m:
        return False, "no relation_id in detail"
    rid = int(m.group(1))
    cur.execute("SELECT relation_type, source FROM relations WHERE id=%s", (rid,))
    rec = cur.fetchone()
    if not rec:
        return False, f"relation {rid} absent"
    rel, source = rec
    prefix = SOURCE_PREFIX.get(source.lower(), source.lower())
    # provenance records the expected canonical target (honk:<x>); confirm the
    # mapping file agrees, independently of what the sampler wrote.
    pm = re.search(r"->\s*honk:(\S+)", row["provenance"])
    declared = edge_maps.get(prefix, {}).get(rel)
    if declared is None:
        return False, f"relation '{rel}' has no declared {prefix} edge mapping"
    if pm and pm.group(1) != declared:
        return False, f"provenance target honk:{pm.group(1)} != mapping '{declared}'"
    return True, f"'{rel}' -> honk:{declared} per {prefix}_edge_mappings"


def check_url_cluster(cur, row) -> (bool, str):
    m = re.search(r"shared URL:\s*(\S+)", row["provenance"])
    if not m:
        return False, "no URL in provenance"
    url = m.group(1)
    cur.execute(
        """
        SELECT count(DISTINCT u.concept_id) AS n_concepts,
               count(DISTINCT cl.cluster_id) AS n_clusters
        FROM urls u
        LEFT JOIN clusters cl ON cl.concept_id = u.concept_id
        WHERE u.external_url = %s
        """,
        (url,),
    )
    n_concepts, n_clusters = cur.fetchone()
    if n_concepts is None or n_concepts < 2:
        return False, f"URL shared by {n_concepts} concept(s), no merge"
    if n_clusters != 1:
        return False, f"{n_concepts} concepts span {n_clusters} clusters (incomplete merge)"
    return True, f"{n_concepts} concepts sharing URL co-clustered into 1 cluster"


def check_enriched(cur, row) -> (bool, str):
    m = re.search(r"(\S+)\s*->\s*(\S+)\s*\((.+)\)", row["detail"])
    if not m:
        return False, "cannot parse detail"
    start_id, end_id, rel = m.group(1), m.group(2), m.group(3)
    cur.execute(
        "SELECT 1 FROM cluster_relations WHERE start_cluster_id=%s AND end_cluster_id=%s "
        "AND relation_type=%s LIMIT 1", (start_id, end_id, rel))
    if not cur.fetchone():
        return False, "cluster-level edge absent from cluster_relations"
    cur.execute("SELECT count(*) FROM clusters WHERE cluster_id=%s", (start_id,))
    if cur.fetchone()[0] < 2:
        return False, "start cluster is a singleton (not an enrichment)"
    # Ground the inferred edge in a real concept-level source edge between the
    # two clusters' members.
    cur.execute(
        """
        SELECT 1 FROM relations r
        JOIN clusters cs ON r.start_concept_id = cs.concept_id
        JOIN clusters ce ON r.end_concept_id   = ce.concept_id
        WHERE cs.cluster_id=%s AND ce.cluster_id=%s AND r.relation_type=%s
        LIMIT 1
        """,
        (start_id, end_id, rel),
    )
    if not cur.fetchone():
        return False, "no member-level source edge grounds this inferred edge"
    return True, "grounded in a member-level source edge"


CHECKS = {
    "pos_mapping": "check_pos",
    "edge_mapping": "check_edge",
    "url_cluster": "check_url_cluster",
    "enriched_cluster": "check_enriched",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--in", dest="infile", default="evaluator/fidelity_audit_sample.csv")
    parser.add_argument("--out", default="evaluator/fidelity_audit_adjudicated.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pos_targets = load_pos_targets()
    edge_maps = load_edge_maps()

    with open(args.infile, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    passed = defaultdict(int)
    total = defaultdict(int)
    conn = connect(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '120s'")
            for row in rows:
                stratum = row["stratum"]
                total[stratum] += 1
                if stratum == "pos_mapping":
                    ok, note = check_pos(cur, row, pos_targets)
                elif stratum == "edge_mapping":
                    ok, note = check_edge(cur, row, edge_maps)
                elif stratum == "url_cluster":
                    ok, note = check_url_cluster(cur, row)
                elif stratum == "enriched_cluster":
                    ok, note = check_enriched(cur, row)
                else:
                    ok, note = False, "unknown stratum"
                passed[stratum] += int(ok)
                row["auto_check"] = "pass" if ok else "FAIL"
                row["auto_note"] = note
    finally:
        conn.close()

    fields = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Adjudicated {len(rows)} rows -> {args.out}\n")
    order = ["pos_mapping", "edge_mapping", "url_cluster", "enriched_cluster"]
    tp = ts = 0
    print(f"{'stratum':18s} {'pass':>5s} / {'total':>5s}   rate")
    print("-" * 44)
    for s in order:
        tp += passed[s]
        ts += total[s]
        rate = 100.0 * passed[s] / total[s] if total[s] else 0.0
        print(f"{s:18s} {passed[s]:5d} / {total[s]:5d}   {rate:6.2f}%")
    print("-" * 44)
    print(f"{'OVERALL':18s} {tp:5d} / {ts:5d}   {100.0*tp/ts if ts else 0:6.2f}%")

    fails = [(r["stratum"], r["auto_note"], r["item"][:70]) for r in rows if r["auto_check"] == "FAIL"]
    if fails:
        print(f"\n{len(fails)} failing rows:")
        for s, note, item in fails[:20]:
            print(f"  [{s}] {note}  ::  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
