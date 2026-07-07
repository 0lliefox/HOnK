#!/usr/bin/env python3
"""provenance_audit_sampler.py — stratified, provenance-linked sample of HOnK
outputs for a manual semantic-fidelity audit (item 2 / reviewer R2-W4).

The paper claims HOnK *preserves* source fidelity; this tool lets us *measure*
it. It draws ~150 outputs across four strata, each row carrying a back-pointer
to the originating source-KB evidence, and writes a CSV with a blank ``label``
column for an annotator to mark ``correct`` / ``partial`` / ``incorrect``.

Strata:
  pos_mapping       - a concept whose POS came from a source->HOnK mapping
  edge_mapping      - a relation whose type came from a source->HOnK mapping
  url_cluster       - concepts merged into one cluster via a shared URL
  enriched_cluster  - a cluster-level relation between clusters with >=2 members
                      (the connectivity clustering adds; members inherit it)

Sampling is deterministic and scale-safe on the 12M-row tables: TABLESAMPLE
REPEATABLE for concepts/relations, and a hashtext-modulo prefilter for the URL
and cluster strata (no full sort / no correlated NOT EXISTS). Requires a
populated DB-mode build.

Usage (from project root):
    python evaluator/provenance_audit_sampler.py \
        [--config config.yaml] [--out fidelity_audit_sample.csv] \
        [--per-stratum 40 40 35 35] [--seed 42]
"""

import argparse
import csv
import json
from collections import defaultdict
from typing import Any, Dict, List

import psycopg2
import yaml

SOURCE_PREFIX = {"conceptnet": "cn", "wiktionary": "wk", "wordnet": "wn"}
MAPPED_SOURCES = list(SOURCE_PREFIX.keys())
import os
MAPPING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "supporting_files", "mappings")

CSV_FIELDS = ["stratum", "item", "source", "provenance", "detail", "label", "notes"]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def connect(cfg: Dict[str, Any]):
    db = cfg["database"]
    return psycopg2.connect(
        dbname=db["dbname"], user=db["user"], password=db["password"],
        host=db["host"], port=str(db["port"]),
    )


def load_reverse_mappings():
    """HOnK target -> {prefix: [original source labels]} for POS and edge maps."""
    rev_pos: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    rev_edge: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    # Concepts store the MAPPED POS (e.g. Adjective), so POS uses a reverse map;
    # relations store the ORIGINAL name (e.g. AtLocation) and are mapped only at
    # graph-serialisation time, so edges need a forward map (original -> target).
    fwd_edge: Dict[str, Dict[str, str]] = defaultdict(dict)
    for prefix in ("cn", "wk", "wn"):
        for kind, rev in (("pos", rev_pos), ("edge", rev_edge)):
            path = os.path.join(MAPPING_DIR, f"{prefix}_{kind}_mappings.json")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            for src_label, value in data.items():
                target = value.get("rel") if isinstance(value, dict) else value
                rev[str(target)][prefix].append(src_label)
                if kind == "edge":
                    fwd_edge[prefix][src_label] = str(target)
    return rev_pos, rev_edge, fwd_edge


def _originals(rev, target: str, prefix: str) -> str:
    return ", ".join(rev.get(target, {}).get(prefix, [])) or "<unmapped/identity>"


# ---------------------------------------------------------------------------
# Strata (deterministic, scale-safe)
# ---------------------------------------------------------------------------

def sample_pos(cur, quota, seed, rev_pos, fraction=0.5):
    cur.execute(
        f"""
        SELECT id, term, part_of_speech, source
        FROM concepts TABLESAMPLE SYSTEM (%s) REPEATABLE (%s)
        WHERE LOWER(source) = ANY(%s)
        LIMIT %s
        """,
        (fraction, seed, MAPPED_SOURCES, quota),
    )
    rows = []
    for cid, term, pos, source in cur.fetchall():
        prefix = SOURCE_PREFIX.get(source.lower(), source)
        rows.append({
            "stratum": "pos_mapping", "item": f"{term} [{pos}]", "source": source,
            "provenance": f"{prefix} POS {_originals(rev_pos, pos, prefix)} -> {pos}",
            "detail": f"concept_id={cid}; file={prefix}_pos_mappings.json",
            "label": "", "notes": "",
        })
    return rows


def sample_edge(cur, quota, seed, fwd_edge, fraction=0.5):
    # Stratify by relation_type so the sample is not dominated by the most frequent
    # type (previously the whole edge_mapping stratum was AtLocation, leaving the
    # meronym/holonym/hypernym mappings unaudited). A deterministic hashtext
    # prefilter keeps it scale-safe; ROW_NUMBER per type then ORDER BY rn draws one
    # of each type before a second of any, maximising relation-type coverage.
    cur.execute(
        """
        SELECT id, s_term, relation_type, o_term, source FROM (
            SELECT r.id, c1.term AS s_term, r.relation_type, c2.term AS o_term, r.source,
                   ROW_NUMBER() OVER (PARTITION BY r.relation_type
                                      ORDER BY abs(hashtext(r.id::text))) AS rn
            FROM relations r
            JOIN concepts c1 ON r.start_concept_id = c1.id
            JOIN concepts c2 ON r.end_concept_id   = c2.id
            WHERE LOWER(r.source) = ANY(%s)
              AND (abs(hashtext(r.id::text)) %% 50) = (%s %% 50)
        ) t
        WHERE rn <= 5
        ORDER BY rn, relation_type
        LIMIT %s
        """,
        (MAPPED_SOURCES, seed, quota),
    )
    rows = []
    for rid, s_term, rel, o_term, source in cur.fetchall():
        prefix = SOURCE_PREFIX.get(source.lower(), source)
        target = fwd_edge.get(prefix, {}).get(rel)
        mapped = f"honk:{target}" if target else "<identity/unmapped>"
        rows.append({
            "stratum": "edge_mapping", "item": f"{s_term} -[{rel}]-> {o_term}", "source": source,
            "provenance": f"{prefix} relation '{rel}' -> {mapped}",
            "detail": f"relation_id={rid}; file={prefix}_edge_mappings.json",
            "label": "", "notes": "",
        })
    return rows


def sample_url_cluster(cur, quota, seed):
    # Deterministic ~0.5% prefilter of the urls table (no full sort), keep URLs
    # shared by >=2 concepts, then show the concepts merged through each.
    cur.execute(
        """
        WITH shared AS (
            SELECT external_url
            FROM urls
            WHERE (abs(hashtext(external_url)) %% 200) = (%s %% 200)
            GROUP BY external_url HAVING count(*) >= 2
            LIMIT %s
        )
        SELECT s.external_url, c.term, c.source, c.part_of_speech, c.sense
        FROM shared s
        JOIN urls u ON u.external_url = s.external_url
        JOIN concepts c ON c.id = u.concept_id
        ORDER BY s.external_url, c.source
        """,
        (seed, quota),
    )
    by_url = defaultdict(list)
    for url, term, source, pos, sense in cur.fetchall():
        # Show the sense discriminator so a sense-scoped merge (e.g. the *battle*
        # sense of magenta with the Battle of Magenta synset) is distinguishable
        # from the pre-fix false merge that conflated it with the colour.
        label = f"{term} ({sense})" if sense else term
        by_url[url].append((label, source))
    rows = []
    for url, members in by_url.items():
        terms = "; ".join(f"{t} [{s}]" for t, s in members)
        rows.append({
            "stratum": "url_cluster", "item": f"merged ({len(members)}): {terms}",
            "source": ", ".join(sorted({s for _t, s in members})),
            "provenance": f"shared URL: {url}",
            "detail": f"members={len(members)}",
            "label": "", "notes": "",
        })
    return rows


def sample_enriched(cur, quota, seed):
    cur.execute(
        """
        WITH multi AS (
            SELECT cluster_id FROM clusters GROUP BY cluster_id HAVING count(*) >= 2
        )
        SELECT cr.start_cluster_id, cr.end_cluster_id, cr.relation_type, cr.source
        FROM cluster_relations cr
        JOIN multi m ON cr.start_cluster_id = m.cluster_id
        WHERE (abs(hashtext(cr.start_cluster_id || cr.end_cluster_id || cr.relation_type)) %% 500) = (%s %% 500)
        LIMIT %s
        """,
        (seed, quota),
    )
    inferred = cur.fetchall()

    def rep_term(cluster_id):
        cur.execute(
            "SELECT c.term FROM clusters cl JOIN concepts c ON cl.concept_id=c.id "
            "WHERE cl.cluster_id=%s ORDER BY c.term LIMIT 1", (cluster_id,))
        row = cur.fetchone()
        return row[0] if row else cluster_id

    rows = []
    for start_id, end_id, rel, source in inferred:
        rows.append({
            "stratum": "enriched_cluster",
            "item": f"{rep_term(start_id)} -[{rel}]-> {rep_term(end_id)}",
            "source": source or "",
            "provenance": "cluster-level relation (members of the start cluster inherit it via URL-based merging)",
            "detail": f"{start_id} -> {end_id} ({rel})",
            "label": "", "notes": "",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="fidelity_audit_sample.csv")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--per-stratum", type=int, nargs=4, default=[40, 40, 35, 35],
        metavar=("POS", "EDGE", "URL_CLUSTER", "ENRICHED"),
        help="Quota per stratum (default 40 40 35 35 = 150).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rev_pos, rev_edge, fwd_edge = load_reverse_mappings()
    q_pos, q_edge, q_url, q_enr = args.per_stratum

    rows: List[Dict[str, str]] = []
    conn = connect(cfg)
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '120s'")
            rows += sample_pos(cur, q_pos, args.seed, rev_pos)
            rows += sample_edge(cur, q_edge, args.seed, fwd_edge)
            rows += sample_url_cluster(cur, q_url, args.seed)
            rows += sample_enriched(cur, q_enr, args.seed)
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
    for stratum in ("pos_mapping", "edge_mapping", "url_cluster", "enriched_cluster"):
        print(f"  {stratum}: {counts[stratum]}")
    if len(rows) < sum(args.per_stratum):
        print("NOTE: fewer rows than requested for some strata (limited candidates "
              "or sample fraction); re-run with a different --seed if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
