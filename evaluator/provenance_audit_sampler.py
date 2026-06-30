#!/usr/bin/env python3
"""provenance_audit_sampler.py — stratified, provenance-linked sample of HOnK
outputs for a manual semantic-fidelity audit (item 2 / reviewer R2-W4).

The paper claims HOnK *preserves* source fidelity; this tool lets us *measure*
it. It draws ~150 outputs across four strata, each row carrying a back-pointer
to the originating source-KB evidence, and writes a CSV with a blank ``label``
column for an annotator to mark ``correct`` / ``partial`` / ``incorrect``.

Strata:
  pos_mapping           - a concept whose POS came from a source->HOnK mapping
  edge_mapping          - a relation whose type came from a source->HOnK mapping
  url_cluster           - concepts merged into one cluster via a shared URL
  inferred_cluster_rel  - a cluster relation with no direct underlying relation

Sampling is deterministic (ORDER BY md5(key || salt)) so the same sheet can be
regenerated. Requires a populated DB-mode build (run the full+clustered ablation
config in db mode first).

Usage (from project root):
    python evaluator/provenance_audit_sampler.py \
        [--config config.yaml] [--out fidelity_audit_sample.csv] \
        [--per-stratum 40 40 35 35] [--salt honk-audit-v1]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
import yaml

# DB source value -> mapping-file prefix.
SOURCE_PREFIX = {"conceptnet": "cn", "wiktionary": "wk", "wordnet": "wn"}
MAPPED_SOURCES = tuple(SOURCE_PREFIX.keys())
MAPPING_DIR = Path(__file__).resolve().parent.parent / "supporting_files" / "mappings"

CSV_FIELDS = ["stratum", "item", "source", "provenance", "detail", "label", "notes"]


# ---------------------------------------------------------------------------
# Config / DB
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def connect(cfg: Dict[str, Any]):
    db = cfg["database"]
    return psycopg2.connect(
        dbname=db["dbname"],
        user=db["user"],
        password=db["password"],
        host=db["host"],
        port=str(db["port"]),
    )


def load_reverse_mappings():
    """Build reverse lookups: HOnK target -> {prefix: [original source labels]}
    for both POS and edge mappings, so each sampled output can show the source
    label(s) it was derived from."""
    rev_pos: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    rev_edge: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for prefix in ("cn", "wk", "wn"):
        for kind, rev in (("pos", rev_pos), ("edge", rev_edge)):
            path = MAPPING_DIR / f"{prefix}_{kind}_mappings.json"
            if not path.is_file():
                continue
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            for src_label, value in data.items():
                target = value.get("rel") if isinstance(value, dict) else value
                rev[str(target)][prefix].append(src_label)
    return rev_pos, rev_edge


def _originals(rev, target: str, prefix: str) -> str:
    return ", ".join(rev.get(target, {}).get(prefix, [])) or "<unmapped/identity>"


# ---------------------------------------------------------------------------
# Per-stratum sampling
# ---------------------------------------------------------------------------

def sample_pos(cur, quota: int, salt: str, rev_pos) -> List[Dict[str, str]]:
    cur.execute(
        """
        SELECT id, term, part_of_speech, source
        FROM concepts
        WHERE source = ANY(%s)
        ORDER BY md5(CAST(id AS text) || %s)
        LIMIT %s
        """,
        (list(MAPPED_SOURCES), salt, quota),
    )
    rows = []
    for cid, term, pos, source in cur.fetchall():
        prefix = SOURCE_PREFIX.get(source, source)
        rows.append({
            "stratum": "pos_mapping",
            "item": f"{term} [{pos}]",
            "source": source,
            "provenance": f"{prefix} POS {_originals(rev_pos, pos, prefix)} -> {pos}",
            "detail": f"concept_id={cid}; file={prefix}_pos_mappings.json",
            "label": "",
            "notes": "",
        })
    return rows


def sample_edge(cur, quota: int, salt: str, rev_edge) -> List[Dict[str, str]]:
    cur.execute(
        """
        SELECT r.id, c1.term, r.relation_type, c2.term, r.source
        FROM relations r
        JOIN concepts c1 ON r.start_concept_id = c1.id
        JOIN concepts c2 ON r.end_concept_id   = c2.id
        WHERE r.source = ANY(%s)
        ORDER BY md5(CAST(r.id AS text) || %s)
        LIMIT %s
        """,
        (list(MAPPED_SOURCES), salt, quota),
    )
    rows = []
    for rid, s_term, rel, o_term, source in cur.fetchall():
        prefix = SOURCE_PREFIX.get(source, source)
        rows.append({
            "stratum": "edge_mapping",
            "item": f"{s_term} -[{rel}]-> {o_term}",
            "source": source,
            "provenance": f"{prefix} relation {_originals(rev_edge, rel, prefix)} -> {rel}",
            "detail": f"relation_id={rid}; file={prefix}_edge_mappings.json",
            "label": "",
            "notes": "",
        })
    return rows


def sample_url_cluster(cur, quota: int, salt: str) -> List[Dict[str, str]]:
    cur.execute(
        """
        SELECT cluster_id
        FROM clusters
        GROUP BY cluster_id
        HAVING COUNT(*) >= 2
        ORDER BY md5(cluster_id || %s)
        LIMIT %s
        """,
        (salt, quota),
    )
    cluster_ids = [row[0] for row in cur.fetchall()]
    rows = []
    for cluster_id in cluster_ids:
        cur.execute(
            """
            SELECT c.term, c.source, c.part_of_speech,
                   STRING_AGG(DISTINCT u.external_url, ' | ') AS urls
            FROM clusters cl
            JOIN concepts c ON cl.concept_id = c.id
            LEFT JOIN urls u ON u.concept_id = c.id
            WHERE cl.cluster_id = %s
            GROUP BY c.id, c.term, c.source, c.part_of_speech
            ORDER BY c.source, c.term
            """,
            (cluster_id,),
        )
        members = cur.fetchall()
        terms = "; ".join(f"{t} [{src}]" for t, src, _pos, _u in members)
        shared = sorted({u for _t, _s, _p, urls in members if urls for u in urls.split(" | ")})
        rows.append({
            "stratum": "url_cluster",
            "item": f"merged ({len(members)}): {terms}",
            "source": ", ".join(sorted({src for _t, src, _p, _u in members})),
            "provenance": "shared URL(s): " + (" | ".join(shared[:3]) if shared else "<none recorded>"),
            "detail": f"cluster_id={cluster_id}; members={len(members)}",
            "label": "",
            "notes": "",
        })
    return rows


def sample_inferred(cur, quota: int, salt: str) -> List[Dict[str, str]]:
    cur.execute(
        """
        SELECT cr.start_cluster_id, cr.end_cluster_id, cr.relation_type, cr.source
        FROM cluster_relations cr
        WHERE NOT EXISTS (
            SELECT 1
            FROM relations r
            JOIN clusters c1 ON r.start_concept_id = c1.concept_id AND c1.cluster_id = cr.start_cluster_id
            JOIN clusters c2 ON r.end_concept_id   = c2.concept_id AND c2.cluster_id = cr.end_cluster_id
            WHERE r.relation_type = cr.relation_type
        )
        ORDER BY md5(cr.start_cluster_id || cr.end_cluster_id || cr.relation_type || %s)
        LIMIT %s
        """,
        (salt, quota),
    )
    inferred = cur.fetchall()

    def rep_term(cluster_id: str) -> str:
        cur.execute(
            """
            SELECT c.term FROM clusters cl JOIN concepts c ON cl.concept_id = c.id
            WHERE cl.cluster_id = %s ORDER BY c.term LIMIT 1
            """,
            (cluster_id,),
        )
        row = cur.fetchone()
        return row[0] if row else cluster_id

    rows = []
    for start_id, end_id, rel, source in inferred:
        rows.append({
            "stratum": "inferred_cluster_rel",
            "item": f"{rep_term(start_id)} -[{rel}]-> {rep_term(end_id)}",
            "source": source or "",
            "provenance": "inferred via clustering; no direct relation underlies this edge",
            "detail": f"{start_id} -> {end_id} ({rel})",
            "label": "",
            "notes": "",
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="fidelity_audit_sample.csv")
    parser.add_argument("--salt", default="honk-audit-v1", help="Deterministic sampling salt.")
    parser.add_argument(
        "--per-stratum", type=int, nargs=4, default=[40, 40, 35, 35],
        metavar=("POS", "EDGE", "URL_CLUSTER", "INFERRED"),
        help="Quota per stratum (default 40 40 35 35 = 150).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    rev_pos, rev_edge = load_reverse_mappings()
    q_pos, q_edge, q_url, q_inf = args.per_stratum

    rows: List[Dict[str, str]] = []
    conn = connect(cfg)
    try:
        with conn.cursor() as cur:
            rows += sample_pos(cur, q_pos, args.salt, rev_pos)
            rows += sample_edge(cur, q_edge, args.salt, rev_edge)
            rows += sample_url_cluster(cur, q_url, args.salt)
            rows += sample_inferred(cur, q_inf, args.salt)
    finally:
        conn.close()

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(int)
    for row in rows:
        counts[row["stratum"]] += 1
    print(f"Wrote {len(rows)} audit rows to {args.out}")
    for stratum in ("pos_mapping", "edge_mapping", "url_cluster", "inferred_cluster_rel"):
        print(f"  {stratum}: {counts[stratum]}")
    if len(rows) < sum(args.per_stratum):
        print("NOTE: fewer rows than requested — some strata had limited candidates "
              "(is the DB populated with a full+clustered build?).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
