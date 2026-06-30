#!/usr/bin/env python3
"""
word_relations.py — find all relationships between two words in the HOnK database.

Usage (from project root):
    python benchmarking/word_relations.py <word1> <word2> [--config config.yaml]

Reports two separate tables:
  1. Original relations from the `relations` table  (direct KB edges)
  2. Cluster-derived relations from the `cluster_relations` table
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import psycopg2
import yaml


# ---------------------------------------------------------------------------
# Config / DB helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect(cfg: Dict[str, Any]):
    db = cfg["database"]
    return psycopg2.connect(
        dbname=db["dbname"],
        user=db["user"],
        password=db["password"],
        host=db["host"],
        port=str(db["port"]),
    )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def find_concepts(cur, term: str) -> List[Tuple[int, str, str, str]]:
    """Return all (id, term, part_of_speech, source) rows matching *term* (case-insensitive)."""
    cur.execute(
        "SELECT id, term, part_of_speech, source FROM concepts WHERE LOWER(term) = LOWER(%s)",
        (term,),
    )
    return cur.fetchall()


def find_direct_relations(
    cur, ids1: List[int], ids2: List[int]
) -> List[Dict[str, Any]]:
    """
    All rows in `relations` where one endpoint is in ids1 and the other in ids2.
    Returns dicts with keys: direction, from_term, from_pos, from_source,
    relation_type, weight, rel_source, to_term, to_pos, to_source.
    direction is '→' when word1 is the subject, '←' when word2 is the subject.
    """
    if not ids1 or not ids2:
        return []

    ids1_set = set(ids1)
    cur.execute(
        """
        SELECT
            r.start_concept_id,
            c1.term             AS from_term,
            c1.part_of_speech   AS from_pos,
            c1.source           AS from_source,
            r.relation_type,
            r.weight,
            r.source            AS rel_source,
            c2.term             AS to_term,
            c2.part_of_speech   AS to_pos,
            c2.source           AS to_source
        FROM relations r
        JOIN concepts c1 ON r.start_concept_id = c1.id
        JOIN concepts c2 ON r.end_concept_id   = c2.id
        WHERE (r.start_concept_id = ANY(%s) AND r.end_concept_id = ANY(%s))
           OR (r.start_concept_id = ANY(%s) AND r.end_concept_id = ANY(%s))
        ORDER BY r.relation_type, r.source, c1.term, c2.term
        """,
        (ids1, ids2, ids2, ids1),
    )
    cols = [d[0] for d in cur.description]
    rows = []
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        d["direction"] = "→" if d["start_concept_id"] in ids1_set else "←"
        del d["start_concept_id"]
        rows.append(d)
    return rows


def find_cluster_ids_for_concepts(
    cur, concept_ids: List[int]
) -> Dict[int, List[str]]:
    """Return {concept_id: [cluster_id, ...]} for the given concept IDs."""
    if not concept_ids:
        return {}
    cur.execute(
        "SELECT concept_id, cluster_id FROM clusters WHERE concept_id = ANY(%s)",
        (concept_ids,),
    )
    mapping: Dict[int, List[str]] = {}
    for cid, clid in cur.fetchall():
        mapping.setdefault(cid, []).append(clid)
    return mapping


def find_cluster_relations(
    cur,
    ids1: List[int],
    ids2: List[int],
    concepts1: List[Tuple],
    concepts2: List[Tuple],
) -> List[Dict[str, Any]]:
    """
    All rows in `cluster_relations` where one endpoint-cluster contains a
    concept from ids1 and the other contains a concept from ids2.

    Also annotates which word's concept is in each cluster.
    """
    if not ids1 or not ids2:
        return []

    cluster_map1 = find_cluster_ids_for_concepts(cur, ids1)
    cluster_map2 = find_cluster_ids_for_concepts(cur, ids2)

    clusters1 = list({cid for cids in cluster_map1.values() for cid in cids})
    clusters2 = list({cid for cids in cluster_map2.values() for cid in cids})

    if not clusters1 or not clusters2:
        return []

    cur.execute(
        """
        SELECT
            cr.start_cluster_id,
            cr.end_cluster_id,
            cr.relation_type,
            cr.weight,
            cr.source AS rel_source
        FROM cluster_relations cr
        WHERE (cr.start_cluster_id = ANY(%s) AND cr.end_cluster_id = ANY(%s))
           OR (cr.start_cluster_id = ANY(%s) AND cr.end_cluster_id = ANY(%s))
        ORDER BY cr.relation_type, cr.source, cr.start_cluster_id
        """,
        (clusters1, clusters2, clusters2, clusters1),
    )
    raw_rows = cur.fetchall()

    # Build reverse maps: cluster_id -> list of (term, pos, source) for word1/word2
    def cluster_members(cluster_map: Dict[int, List[str]], concepts: List[Tuple]) -> Dict[str, List[str]]:
        """cluster_id -> ["term(pos,src)", ...]"""
        c_by_id = {c[0]: c for c in concepts}  # concept_id -> row
        out: Dict[str, List[str]] = {}
        for cid, clids in cluster_map.items():
            c = c_by_id.get(cid)
            if c is None:
                continue
            label = f"{c[1]}({c[2]},{c[3]})"
            for clid in clids:
                out.setdefault(clid, []).append(label)
        return out

    members1 = cluster_members(cluster_map1, concepts1)
    members2 = cluster_members(cluster_map2, concepts2)

    rows = []
    for start_cid, end_cid, rel_type, weight, rel_source in raw_rows:
        # Determine direction relative to word1/word2
        if start_cid in clusters1 and end_cid in clusters2:
            word1_cluster, word2_cluster = start_cid, end_cid
            direction = "→"
        else:
            word1_cluster, word2_cluster = end_cid, start_cid
            direction = "←"

        rows.append({
            "direction":     direction,
            "word1_cluster": word1_cluster,
            "word1_concepts": ", ".join(members1.get(word1_cluster, ["?"])),
            "relation_type": rel_type,
            "weight":        weight,
            "rel_source":    rel_source,
            "word2_cluster": word2_cluster,
            "word2_concepts": ", ".join(members2.get(word2_cluster, ["?"])),
        })

    return rows


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def print_table(title: str, rows: List[Dict], columns: List[str]) -> None:
    SEP  = "=" * 110
    THIN = "-" * 110
    print(f"\n{SEP}")
    print(f"  {title}  ({len(rows)} result(s))")
    print(SEP)

    if not rows:
        print("  (none)\n")
        return

    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    fmt = "  " + "  ".join(f"{{:<{widths[c]}}}" for c in columns)
    print(fmt.format(*columns))
    print(THIN)
    for row in rows:
        print(fmt.format(*[str(row.get(c, "")) for c in columns]))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find all relationships between two words in the HOnK database."
    )
    parser.add_argument("word1", help="First word")
    parser.add_argument("word2", help="Second word")
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (auto-detected if omitted)",
    )
    args = parser.parse_args()

    # Locate config.yaml
    config_path = args.config
    if config_path is None:
        for candidate in ["config.yaml", "../config.yaml"]:
            if Path(candidate).is_file():
                config_path = candidate
                break
    if config_path is None:
        print(
            "ERROR: config.yaml not found. Run from the project root or benchmarking/ directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = load_config(config_path)
    conn = connect(cfg)
    cur = conn.cursor()

    word1, word2 = args.word1.strip(), args.word2.strip()
    print(f"\nSearching for relationships between '{word1}' and '{word2}'...\n")

    concepts1 = find_concepts(cur, word1)
    concepts2 = find_concepts(cur, word2)

    if not concepts1:
        print(f"  WARNING: '{word1}' not found in the concepts table.")
    else:
        print(f"  '{word1}' — {len(concepts1)} concept(s):")
        for cid, term, pos, src in concepts1:
            print(f"    id={cid}  pos={pos}  source={src}")

    if not concepts2:
        print(f"  WARNING: '{word2}' not found in the concepts table.")
    else:
        print(f"  '{word2}' — {len(concepts2)} concept(s):")
        for cid, term, pos, src in concepts2:
            print(f"    id={cid}  pos={pos}  source={src}")

    ids1 = [r[0] for r in concepts1]
    ids2 = [r[0] for r in concepts2]

    # ---- Table 1: original relations ----------------------------------------
    direct = find_direct_relations(cur, ids1, ids2)
    print_table(
        f"ORIGINAL RELATIONS  (relations table)  '{word1}' ↔ '{word2}'",
        direct,
        [
            "direction",
            "from_term", "from_pos", "from_source",
            "relation_type", "weight", "rel_source",
            "to_term",   "to_pos",   "to_source",
        ],
    )

    # ---- Table 2: cluster-derived relations ---------------------------------
    clustered = find_cluster_relations(cur, ids1, ids2, concepts1, concepts2)
    print_table(
        f"CLUSTER-DERIVED RELATIONS  (cluster_relations table)  '{word1}' ↔ '{word2}'",
        clustered,
        [
            "direction",
            "word1_cluster", "word1_concepts",
            "relation_type", "weight", "rel_source",
            "word2_cluster", "word2_concepts",
        ],
    )

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
