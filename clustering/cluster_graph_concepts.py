import json
import logging
import os
from collections import defaultdict

from tqdm import tqdm
from pyoxigraph import NamedNode, Quad, DefaultGraph

from tools.timer import timer
from tools.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")


class ConceptGraphClusterer:
    def __init__(self, builder, config):
        self.builder = builder
        self.graph_manager = self.builder.graph_manager
        self.g = self.graph_manager.g if hasattr(self.graph_manager, 'g') else self.graph_manager
        self.ns = self.graph_manager.ns
        self.config = config
        self.verbose = self.config['general']['verbose']
        self.base_uri = self.config['turtle_export']['base_uri']

    @timer
    def run(self):
        logging.info("Starting graph clustering...")
        cluster_mappings = self.find_and_store_clusters()
        self.coalesce_relationships(cluster_mappings)
        self.cleanup_graph()
        logging.info("Concept clustering finished")
        return self.g

    def find_and_store_clusters(self):
        cache_path = f"{self.config['local_files']['cache']}/adj_list_graph.json"

        # Adjacency list built from Graph
        db = self.build_adj_list()

        if os.path.exists(cache_path) and self.config['general']['should_cache']:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False, indent=4)

        # Transitive closure
        self.transitive_closure(db)

        # Build clusters
        cluster_mappings, visited_nodes = self.build_clusters_from_adj(db)

        # Create clusters for unclustered concepts
        self.add_unclustered_concepts(cluster_mappings, visited_nodes)

        logging.info("Cluster identification and storage complete.")
        return cluster_mappings

    @timer
    def build_adj_list(self):
        logging.info("  - Building adjacency list from Graph URLs")

        db = defaultdict(set)
        for quad in tqdm(self.g.quads_for_pattern(None, self.ns.hasURL, None), desc="Fetching URL triples",
                         disable=not self.verbose):
            s_uri = quad.subject.value
            o_val = quad.object.value

            o, pos = o_val.split('==')
            s = f"{s_uri}=={pos}"

            db[s].add(o)
            db[o].add(s)

        return {k: sorted(list(v)) for k, v in db.items()}

    @timer
    def add_unclustered_concepts(self, cluster_mappings, visited_nodes):
        logging.info("  - Fetching all concept IDs from graph...")

        unvisited_candidates = set()

        for quad in self.g.quads_for_pattern(None, RDF_TYPE, None):
            s = quad.subject.value
            t_str = quad.object.value

            if self.base_uri in s and self.base_uri in t_str:
                if '#' in t_str:
                    pos = t_str.split('#')[-1]
                else:
                    pos = t_str.split('/')[-1]

                candidate = f"{s}=={pos}"

                if candidate not in visited_nodes:
                    unvisited_candidates.add(candidate)

        cluster_id = 0
        if cluster_mappings:
            last_cluster_label = cluster_mappings[-1][1]
            try:
                cluster_id = int(last_cluster_label.replace('c', ''))
            except ValueError:
                cluster_id = 0

        for node in tqdm(unvisited_candidates, desc="Handling isolated nodes", disable=not self.verbose):
            cluster_id += 1
            cluster_mappings.append((node, f"c{cluster_id}"))

    @timer
    def build_clusters_from_adj(self, db):
        logging.info("  - Building clusters from adjacency list...")
        cluster_mappings = []
        visited_nodes = set()

        visited_clusters = dict()
        for key_node, adjacency_list in tqdm(db.items(), desc="Building clusters", disable=not self.verbose):
            cluster_nodes = set(adjacency_list)
            cluster_nodes.add(key_node)

            c_key = tuple(sorted(tuple(map(str, cluster_nodes))))
            if not (c_key in visited_clusters.keys()):
                cluster_id = len(visited_clusters)
                visited_clusters[c_key] = cluster_id

                for node in cluster_nodes:
                    if node.startswith(self.base_uri):
                        cluster_mappings.append((node, f"c{cluster_id}"))

                    visited_nodes.add(node)

        return cluster_mappings, visited_nodes

    @timer
    def transitive_closure(self, adjacency_db):
        logging.info("  - Calculating transitive closure on adjacency list...")
        for i in tqdm(adjacency_db.keys(), desc="Transitive closure", disable=not self.verbose):
            adjacency_list = adjacency_db[i]
            j_idx = 0
            while j_idx < len(adjacency_list):
                j = adjacency_list[j_idx]
                adjacency_list_j = adjacency_db[j]
                k_idx = 0
                while k_idx < len(adjacency_list_j):
                    k = adjacency_list_j[k_idx]
                    if i != k and k not in adjacency_list:
                        adjacency_list.append(k)
                    k_idx += 1
                j_idx += 1
                adjacency_db[j] = adjacency_list_j
            adjacency_db[i] = adjacency_list

    @timer
    def coalesce_relationships(self, cluster_mappings):
        logging.info("Coalescing relationships between concepts based on clusters...")

        logging.info("  - Building in-memory cluster maps...")
        raw_uri_to_cluster_info = defaultdict(list)
        cluster_to_raw_uris = defaultdict(set)

        for key, cluster_label in cluster_mappings:
            parts = key.split('==')
            raw_uri = parts[0]
            member_pos = parts[1] if len(parts) > 1 else None

            raw_uri_to_cluster_info[raw_uri].append((cluster_label, member_pos))
            cluster_to_raw_uris[cluster_label].add(raw_uri)

        cluster_relations = set()
        predicate_constraints = {}
        excluded_predicates = {self.ns.hasURL.value, RDF_TYPE.value}

        logging.info(f"  - Scanning {len(raw_uri_to_cluster_info)} unique concepts for existing relations...")

        for s_raw in tqdm(raw_uri_to_cluster_info.keys(), desc="Scanning graph", disable=not self.verbose):
            for quad in self.g.quads_for_pattern(NamedNode(s_raw), None, None):
                p_node = quad.predicate
                o_node = quad.object
                p_val = p_node.value

                if self.base_uri not in p_val or p_val in excluded_predicates:
                    continue

                if p_node not in predicate_constraints:
                    s_constraint = next(self.g.quads_for_pattern(p_node, self.ns.source_pos, None), None)
                    t_constraint = next(self.g.quads_for_pattern(p_node, self.ns.target_pos, None), None)

                    predicate_constraints[p_node] = (
                        s_constraint.object.value if s_constraint else None,
                        t_constraint.object.value if t_constraint else None
                    )

                s_constraint_pos, t_constraint_pos = predicate_constraints[p_node]
                o_str = o_node.value

                if o_str in raw_uri_to_cluster_info:
                    s_clusters_info = raw_uri_to_cluster_info[s_raw]
                    o_clusters_info = raw_uri_to_cluster_info[o_str]

                    for c_s_label, s_member_pos in s_clusters_info:
                        if s_constraint_pos and s_member_pos and s_constraint_pos != s_member_pos:
                            continue

                        for c_o_label, o_member_pos in o_clusters_info:
                            if t_constraint_pos and o_member_pos and t_constraint_pos != o_member_pos:
                                continue

                            cluster_relations.add((c_s_label, p_node, c_o_label))

        logging.info(f"  - Found {len(cluster_relations)} unique cluster-level relationships.")
        logging.info("  - Optimizing and propagating relationships to all equivalent concepts...")

        cluster_to_named_nodes = {
            label: [NamedNode(uri) for uri in uris]
            for label, uris in cluster_to_raw_uris.items()
        }

        grouped_targets = defaultdict(set)
        for c_start, p_node, c_end in tqdm(cluster_relations, desc="Grouping relations", disable=not self.verbose):
            grouped_targets[(c_start, p_node)].update(cluster_to_named_nodes[c_end])

        logging.info(f"  - Grouped into {len(grouped_targets)} unique (start_cluster, predicate) pairs.")

        for (c_start, p_node), target_uris in tqdm(grouped_targets.items(), desc="Materialising triples",
                                                   disable=not self.verbose):
            start_uris = cluster_to_named_nodes[c_start]
            for s in start_uris:
                for o in target_uris:
                    if s != o:
                        self.g.add(Quad(s, p_node, o, DefaultGraph()))

        logging.info("Relationship coalescing complete.")

    @timer
    def cleanup_graph(self):
        logging.info("Cleaning up intermediate graph data...")

        # We must cast iterators to list() before calling remove() in Pyoxigraph
        def bulk_remove(p_node):
            to_remove = list(self.g.quads_for_pattern(None, p_node, None))
            for q in to_remove:
                self.g.remove(q)

        bulk_remove(self.ns.hasURL)
        bulk_remove(self.ns.source_pos)
        bulk_remove(self.ns.target_pos)

        logging.info("Merging duplicate relationship predicates...")

        weights = {q.subject: q.object.value for q in self.g.quads_for_pattern(None, self.ns.weight, None)}

        if not weights:
            return

        negations = {}
        for q in self.g.quads_for_pattern(None, self.ns.is_negated, None):
            if q.subject in weights:
                negations[q.subject] = q.object.value

        props_map = defaultdict(list)
        for rel_uri, weight_val in tqdm(weights.items(), desc="Analyzing predicates", disable=not self.verbose):
            is_negated_val = negations.get(rel_uri)
            if is_negated_val is not None:
                base_type_quad = next(self.g.quads_for_pattern(rel_uri, RDF_TYPE, None), None)
                if base_type_quad:
                    sig = (base_type_quad.object.value, is_negated_val, weight_val)
                    props_map[sig].append(rel_uri)

        merged_count = 0
        for sig, uris in tqdm(props_map.items(), desc="Merging predicates", disable=not self.verbose):
            if len(uris) > 1:
                uris.sort(key=lambda u: u.value)
                canonical = uris[0]
                duplicates = uris[1:]

                for dup in duplicates:
                    triples_to_move = list(self.g.quads_for_pattern(None, dup, None))
                    if triples_to_move:
                        for q in triples_to_move:
                            self.g.add(Quad(q.subject, canonical, q.object, DefaultGraph()))
                            self.g.remove(q)

                    # Remove any remaining metadata referencing the duplicate predicate
                    for q in list(self.g.quads_for_pattern(dup, None, None)):
                        self.g.remove(q)

                    merged_count += 1

        logging.info(
            f"Cleanup complete: removed hasURL/POS annotations and merged {merged_count} duplicate predicates.")


def main():
    os.chdir('../')
    config = get_config()

    from build_ontology import OntologyBuilder
    from benchmarking.benchmark import Benchmark

    builder = OntologyBuilder(config, 0, Benchmark("scalability"))
    clusterer = ConceptGraphClusterer(builder, config)
    clusterer.run()


if __name__ == "__main__":
    main()