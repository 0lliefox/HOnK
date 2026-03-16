import psycopg2
import psycopg2.extras
import yaml
import sys
import urllib.parse
import json
import os
from collections import defaultdict, deque

BASE_URI = "https://logds.github.io/parmenides#"


def load_db_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config['database']
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        sys.exit(1)


def load_mappings(mappings_dir='.'):
    """
    Loads edge and POS mappings from JSON files to reverse-engineer the original
    data representation prior to ingestion, accounting for direction swaps.
    """
    sources = {
        'Wiktionary': 'wk',
        'WordNet':    'wn',
        'ConceptNet': 'cn',
    }
    reverse_edges = {s: {} for s in sources}
    reverse_pos   = {s: {} for s in sources}

    for source_name, prefix in sources.items():
        edge_file = os.path.join(mappings_dir, f"{prefix}_edge_mappings.json")
        if os.path.exists(edge_file):
            with open(edge_file, 'r') as f:
                try:
                    data = json.load(f)
                    for orig_key, map_val in data.items():
                        if not isinstance(map_val, dict):
                            continue
                        mapped_rel = map_val.get('rel')
                        is_swapped = str(map_val.get('swap', False)).lower() == 'true'
                        if mapped_rel:
                            reverse_edges[source_name].setdefault(mapped_rel, []).append(
                                (orig_key, is_swapped)
                            )
                except json.JSONDecodeError:
                    print(f"Warning: could not parse {edge_file}, skipping.")

        pos_file = os.path.join(mappings_dir, f"{prefix}_pos_mappings.json")
        if os.path.exists(pos_file):
            with open(pos_file, 'r') as f:
                try:
                    data = json.load(f)
                    for orig_key, mapped_pos in data.items():
                        reverse_pos[source_name].setdefault(mapped_pos, []).append(orig_key)
                except json.JSONDecodeError:
                    print(f"Warning: could not parse {pos_file}, skipping.")

    return reverse_edges, reverse_pos


def get_cluster_id_by_term(conn, term, pos=None):
    """Returns all cluster IDs associated with a given term, optionally filtered by POS."""
    query = """
        SELECT DISTINCT cl.cluster_id, c.part_of_speech, c.source
        FROM concepts c
        JOIN clusters cl ON c.id = cl.concept_id
        WHERE LOWER(c.term) = LOWER(%s)
    """
    params = [term]
    if pos:
        query += " AND c.part_of_speech = %s"
        params.append(pos)

    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(query, params)
        return cur.fetchall()


def trace_cluster_formation(conn, cluster_id):
    """Finds the shared URLs that caused concepts to be grouped into a cluster."""
    query = """
        SELECT DISTINCT
            c1.id             AS id1,
            c1.term           AS term1,
            c1.source         AS source1,
            c1.part_of_speech AS pos1,
            c2.id             AS id2,
            c2.term           AS term2,
            c2.source         AS source2,
            c2.part_of_speech AS pos2,
            u1.external_url   AS shared_url
        FROM clusters cl1
        JOIN clusters cl2 ON cl1.cluster_id = cl2.cluster_id
        JOIN concepts  c1 ON cl1.concept_id = c1.id
        JOIN concepts  c2 ON cl2.concept_id = c2.id
        JOIN urls      u1 ON c1.id = u1.concept_id
        JOIN urls      u2 ON c2.id = u2.concept_id
        WHERE cl1.cluster_id = %s
          AND cl1.concept_id < cl2.concept_id
          AND u1.external_url = u2.external_url
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(query, (cluster_id,))
        return cur.fetchall()


def find_cluster_members(conn, cluster_id):
    query = """
        SELECT c.id, c.term, c.part_of_speech, c.source
        FROM clusters cl
        JOIN concepts c ON cl.concept_id = c.id
        WHERE cl.cluster_id = %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(query, (cluster_id,))
        return cur.fetchall()


def find_cluster_relations(conn, cluster_id):
    """
    Returns all cluster-level relations involving this cluster, with a representative
    concept (lowest ID) used to identify each cluster's "label" for display.
    """
    query = """
        WITH ClusterReps AS (
            SELECT cluster_id, MIN(concept_id) AS rep_id
            FROM clusters
            GROUP BY cluster_id
        )
        SELECT DISTINCT
            r1.rep_id         AS start_rep_id,
            c1.term           AS start_term,
            cr.relation_type,
            r2.rep_id         AS end_rep_id,
            c2.term           AS end_term,
            cr.start_cluster_id,
            cr.end_cluster_id
        FROM cluster_relations cr
        JOIN ClusterReps r1 ON cr.start_cluster_id = r1.cluster_id
        JOIN concepts    c1 ON r1.rep_id = c1.id
        JOIN ClusterReps r2 ON cr.end_cluster_id = r2.cluster_id
        JOIN concepts    c2 ON r2.rep_id = c2.id
        WHERE cr.start_cluster_id = %s
           OR cr.end_cluster_id   = %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(query, (cluster_id, cluster_id))
        return cur.fetchall()


def trace_relation_origin(conn, start_cluster_id, end_cluster_id, relation_type):
    """
    Traces which pre-clustering concept pairs gave rise to a cluster-level relation.
    No source filter is applied — all ingested sources are considered.
    """
    query = """
        SELECT DISTINCT
            r.source          AS original_source,
            c1.id             AS start_concept_id,
            c1.term           AS start_term,
            c1.source         AS start_concept_source,
            c2.id             AS end_concept_id,
            c2.term           AS end_term,
            c2.source         AS end_concept_source,
            r.relation_type
        FROM relations r
        JOIN clusters cl1 ON r.start_concept_id = cl1.concept_id
        JOIN clusters cl2 ON r.end_concept_id   = cl2.concept_id
        JOIN concepts c1  ON r.start_concept_id = c1.id
        JOIN concepts c2  ON r.end_concept_id   = c2.id
        WHERE cl1.cluster_id = %s
          AND cl2.cluster_id = %s
          AND r.relation_type = %s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(query, (start_cluster_id, end_cluster_id, relation_type))
        return cur.fetchall()


def build_cluster_graph(formation_edges):
    """Builds an adjacency graph from shared-URL formation edges."""
    graph     = defaultdict(list)
    edge_info = {}
    for row in formation_edges:
        id1, id2 = row['id1'], row['id2']
        url = row['shared_url']
        graph[id1].append((id2, url))
        graph[id2].append((id1, url))
        edge_info[(id1, id2)] = url
        edge_info[(id2, id1)] = url
    return graph, edge_info


def get_path(graph, u, v):
    """BFS shortest path between two concept IDs in the formation graph."""
    if u == v:
        return [u]
    queue   = deque([[u]])
    visited = {u}
    while queue:
        path = queue.popleft()
        curr = path[-1]
        for neighbour, _ in graph[curr]:
            if neighbour == v:
                return path + [neighbour]
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(path + [neighbour])
    return []


def get_cluster_path_string(conn, cluster_id, from_id, to_id, m_dict=None):
    """
    Returns a human-readable string showing the URL hops connecting two concepts
    within a cluster. Returns None if the two IDs are the same.
    """
    if from_id == to_id:
        return None

    edges = trace_cluster_formation(conn, cluster_id)

    # m_dict is optional — only fetch if not already provided by the caller
    if m_dict is None:
        members = find_cluster_members(conn, cluster_id)
        m_dict  = {m['id']: m for m in members}

    graph, edge_info = build_cluster_graph(edges)
    path = get_path(graph, from_id, to_id)

    if not path:
        return "No URL path found (isolated node?)"

    hop_strs = []
    for k in range(len(path) - 1):
        n1  = m_dict.get(path[k],     {'term': str(path[k]),     'source': 'Unknown'})
        n2  = m_dict.get(path[k + 1], {'term': str(path[k + 1]), 'source': 'Unknown'})
        url = edge_info.get((path[k], path[k + 1]), 'Unknown URL')
        hop_strs.append(
            f"[{n1['source']}] '{n1['term']}' -[Shared URL: {url}]-> [{n2['source']}] '{n2['term']}'"
        )
    return "\n             ".join(hop_strs)


def auto_discover_transitive(conn, reverse_edges, reverse_pos, limit=5, exclude_sources=None):
    """
    Scans clusters to find examples where a relation was coalesced onto the
    cluster representative via a transitive (non-direct) URL path.
    """
    if exclude_sources is None:
        exclude_sources = []

    print(f"\n[Auto-Discovery] Looking for {limit} transitive closure example(s)...")
    if exclude_sources:
        print(f"  Excluding source(s): {', '.join(exclude_sources)}")

    query = """
        SELECT cluster_id
        FROM clusters
        GROUP BY cluster_id
        HAVING COUNT(concept_id) >= 4
    """
    with conn.cursor() as cur:
        cur.execute(query)
        candidate_clusters = [row[0] for row in cur.fetchall()]

    print(f"  {len(candidate_clusters)} candidate clusters found. Scanning...\n")

    examples_found = 0

    for idx, cid in enumerate(candidate_clusters):
        if examples_found >= limit:
            break

        sys.stdout.write(
            f"\r  Cluster {idx + 1}/{len(candidate_clusters)}  |  examples found: {examples_found}"
        )
        sys.stdout.flush()

        edges = trace_cluster_formation(conn, cid)
        if not edges:
            continue

        members = find_cluster_members(conn, cid)
        if not members:
            continue

        m_dict = {m['id']: m for m in members}
        rep_id = min(m_dict.keys())

        graph, _ = build_cluster_graph(edges)

        # Nodes that reach the representative in more than one hop
        transitive_nodes = [
            n_id for n_id in m_dict
            if n_id != rep_id and len(get_path(graph, n_id, rep_id)) > 2
        ]

        if not transitive_nodes:
            continue

        all_cluster_edges = find_cluster_relations(conn, cid)

        for rel in all_cluster_edges:
            origins = trace_relation_origin(
                conn, rel['start_cluster_id'], rel['end_cluster_id'], rel['relation_type']
            )

            existed_for_rep   = False
            transitive_origins = []

            for org in origins:
                if (org['start_concept_id'] == rel['start_rep_id']
                        and org['end_concept_id'] == rel['end_rep_id']):
                    existed_for_rep = True

                if exclude_sources and org['original_source'] in exclude_sources:
                    continue

                if (org['start_concept_id'] in transitive_nodes
                        or org['end_concept_id'] in transitive_nodes):
                    transitive_origins.append(org)

            if not existed_for_rep and transitive_origins:
                examples_found += 1

                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()

                subj_uri = f"<{BASE_URI}{urllib.parse.quote(rel['start_term'], safe='')}>"
                pred_uri = f"<{BASE_URI}{urllib.parse.quote(rel['relation_type'], safe='')}>"
                obj_uri  = f"<{BASE_URI}{urllib.parse.quote(rel['end_term'], safe='')}>"

                print(f"{'=' * 50}")
                print(f"Example {examples_found}  |  Cluster: {cid}")
                print(f"Representative: [{m_dict[rep_id]['source']}] '{m_dict[rep_id]['term']}' (ID: {rep_id})")
                print(f"{'=' * 50}")
                print(f"  {rel['start_term']} [{rel['relation_type']}] {rel['end_term']}")
                print(f"  {subj_uri} {pred_uri} {obj_uri}")
                print(f"\n  Originating from transitive member(s):")

                for org in transitive_origins:
                    _print_origin(org, rel, reverse_edges)

                    if org['start_concept_id'] in transitive_nodes:
                        path_str = get_cluster_path_string(
                            conn, cid, org['start_concept_id'], rel['start_rep_id'], m_dict
                        )
                        if path_str:
                            print(f"    Path ({org['start_term']} -> rep):\n      {path_str}")

                    elif org['end_concept_id'] in transitive_nodes:
                        path_str = get_cluster_path_string(
                            conn, cid, org['end_concept_id'], rel['end_rep_id'], m_dict
                        )
                        if path_str:
                            print(f"    Path (rep -> {org['end_term']}):\n      {path_str}")

                print()
                break  # one example per cluster keeps output diverse

    if examples_found == 0:
        print("\nNo transitive closure examples found in the scanned clusters.")
    else:
        print(f"\nDone — {examples_found} example(s) shown.")


def _print_origin(org, rel, reverse_edges):
    """Prints a single relation origin with reverse-mapped edge names where available."""
    source    = org['original_source']
    start_src = org['start_concept_source']
    end_src   = org['end_concept_source']
    mapped_rel = org['relation_type']
    start_t   = org['start_term']
    end_t     = org['end_term']

    rev_edges = reverse_edges.get(source, {}).get(mapped_rel, [])

    if not rev_edges:
        print(f"    [{source}] '{start_t}' -[{mapped_rel}]-> '{end_t}'")
        return

    statements = []
    for orig_rel, is_swapped in rev_edges:
        if is_swapped:
            statements.append(
                f"[{end_src}] '{end_t}' -[{orig_rel}]-> [{start_src}] '{start_t}' (swapped)"
            )
        else:
            statements.append(
                f"[{start_src}] '{start_t}' -[{orig_rel}]-> [{end_src}] '{end_t}'"
            )

    print(f"    [Edge source: {source}] {' OR '.join(statements)}  ->  ingested as '{mapped_rel}'")


def main():
    db_config            = load_db_config()
    conn                 = psycopg2.connect(**db_config)
    reverse_edges, reverse_pos = load_mappings()

    if len(sys.argv) == 1 or sys.argv[1].lower() == 'auto':
        exclude_sources = []
        if len(sys.argv) > 2:
            exclude_sources = [s.strip() for s in sys.argv[2].split(',')]
        auto_discover_transitive(conn, reverse_edges, reverse_pos, limit=5,
                                 exclude_sources=exclude_sources)
        conn.close()
        return

    # Term lookup mode
    raw_input = sys.argv[1]
    if raw_input.startswith('<') and raw_input.endswith('>'):
        raw_input = raw_input[1:-1]
    if '#' in raw_input:
        raw_input = raw_input.split('#')[-1]
    term_to_find = urllib.parse.unquote(raw_input)

    print(f"Looking up clusters for: '{term_to_find}'")
    results = get_cluster_id_by_term(conn, term_to_find)

    if not results:
        print("No clusters found for that term.")
        conn.close()
        return

    for res in results:
        cluster_id = res['cluster_id']
        print(f"\n{'=' * 50}")
        print(f"Cluster: {cluster_id}  |  POS: {res['part_of_speech']}  |  Source: {res['source']}")
        print(f"{'=' * 50}")

        members = find_cluster_members(conn, cluster_id)
        m_dict  = {m['id']: m for m in members}

        print(f"\nMembers ({len(members)})")
        for m in members:
            orig_pos_list = reverse_pos.get(m['source'], {}).get(m['part_of_speech'], [])
            orig_pos_str  = f"  [orig POS: {', '.join(orig_pos_list)}]" if orig_pos_list else ""
            print(f"  ID:{m['id']:>10}  [{m['source']}]  {m['term']}  ({m['part_of_speech']}){orig_pos_str}")

        formation_edges = trace_cluster_formation(conn, cluster_id)
        print(f"\nFormation evidence (shared URLs)")
        if not formation_edges:
            print("  (Isolated node — no shared URL intersections found)")
        else:
            for edge in formation_edges:
                print(f"  [{edge['source1']}] '{edge['term1']}'")
                print(f"    -[{edge['shared_url']}]->")
                print(f"  [{edge['source2']}] '{edge['term2']}'")

        all_cluster_edges = find_cluster_relations(conn, cluster_id)
        new_relations = []

        for rel in all_cluster_edges:
            origins = trace_relation_origin(
                conn, rel['start_cluster_id'], rel['end_cluster_id'], rel['relation_type']
            )
            existed_between_reps = any(
                org['start_concept_id'] == rel['start_rep_id']
                and org['end_concept_id'] == rel['end_rep_id']
                for org in origins
            )
            if not existed_between_reps and origins:
                rel_dict            = dict(rel)
                rel_dict['origins'] = origins
                new_relations.append(rel_dict)

        print(f"\nNew relations formed by clustering ({len(new_relations)})")
        if not new_relations:
            print("  (No new relations were generated for this cluster)")
        else:
            for rel in new_relations:
                subj_uri = f"<{BASE_URI}{urllib.parse.quote(rel['start_term'], safe='')}>"
                pred_uri = f"<{BASE_URI}{urllib.parse.quote(rel['relation_type'], safe='')}>"
                obj_uri  = f"<{BASE_URI}{urllib.parse.quote(rel['end_term'], safe='')}>"

                print(f"\n  {rel['start_term']} [{rel['relation_type']}] {rel['end_term']}")
                print(f"  {subj_uri} {pred_uri} {obj_uri}")
                print("  Originating from:")

                for org in rel['origins']:
                    _print_origin(org, rel, reverse_edges)

                    if org['start_concept_id'] != rel['start_rep_id']:
                        pass_dict = m_dict if rel['start_cluster_id'] == cluster_id else None
                        path_str  = get_cluster_path_string(
                            conn, rel['start_cluster_id'], org['start_concept_id'],
                            rel['start_rep_id'], pass_dict
                        )
                        if path_str:
                            print(f"    Path ({org['start_term']} -> rep):\n      {path_str}")

                    if org['end_concept_id'] != rel['end_rep_id']:
                        pass_dict = m_dict if rel['end_cluster_id'] == cluster_id else None
                        path_str  = get_cluster_path_string(
                            conn, rel['end_cluster_id'], org['end_concept_id'],
                            rel['end_rep_id'], pass_dict
                        )
                        if path_str:
                            print(f"    Path (rep -> {org['end_term']}):\n      {path_str}")

    conn.close()


if __name__ == "__main__":
    main()