import itertools
import json
import logging
import os
from collections import defaultdict

from tqdm import tqdm
from rdflib import RDF, URIRef

from tools.timer import timer
from tools.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        for s, o in tqdm(self.g.subject_objects(self.ns.hasURL), desc="Fetching URL triples", disable=not self.verbose):
            o, pos = o.split('==')
            s = f"{s}=={pos}"

            db[s].add(o)
            db[o].add(s)

        return {k: sorted(list(v)) for k, v in db.items()}

    @timer
    def add_unclustered_concepts(self, cluster_mappings, visited_nodes):
        logging.info("  - Fetching all concept IDs from graph...")

        unvisited_candidates = set()

        # We iterate over all unique subjects in the graph
        for s, t in self.g.subject_objects(RDF.type):
            if self.base_uri in str(s):
                # Extract POS from type URI
                t_str = str(t)

                # We only consider types that are within our namespace
                if self.base_uri in t_str:
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

        # Build lookups
        logging.info("  - Building in-memory cluster maps...")

        raw_uri_to_cluster_info = defaultdict(list)
        cluster_to_raw_uris = defaultdict(set)

        for key, cluster_label in cluster_mappings:
            parts = key.split('==')
            raw_uri = parts[0]
            member_pos = parts[1] if len(parts) > 1 else None

            raw_uri_to_cluster_info[raw_uri].append((cluster_label, member_pos))
            cluster_to_raw_uris[cluster_label].add(raw_uri)

        # Identify relationships between clusters
        cluster_relations = set()
        predicate_constraints = {}
        excluded_predicates = {self.ns.hasURL, RDF.type}

        logging.info(f"  - Scanning {len(raw_uri_to_cluster_info)} unique concepts for existing relations...")

        for s_raw in tqdm(raw_uri_to_cluster_info.keys(), desc="Scanning graph", disable=not self.verbose):
            for p, o in self.g.predicate_objects(URIRef(s_raw)):
                if self.base_uri not in str(p) or p in excluded_predicates:
                    continue

                if p not in predicate_constraints:
                    s_constraint = self.g.value(p, self.ns.source_pos)
                    t_constraint = self.g.value(p, self.ns.target_pos)
                    predicate_constraints[p] = (
                        str(s_constraint) if s_constraint else None,
                        str(t_constraint) if t_constraint else None
                    )

                s_constraint_pos, t_constraint_pos = predicate_constraints[p]
                o_str = str(o)

                if o_str in raw_uri_to_cluster_info:
                    s_clusters_info = raw_uri_to_cluster_info[s_raw]
                    o_clusters_info = raw_uri_to_cluster_info[o_str]

                    for c_s_label, s_member_pos in s_clusters_info:
                        # Subject must match source POS
                        if s_constraint_pos and s_member_pos:
                            if s_constraint_pos != s_member_pos:
                                continue

                        for c_o_label, o_member_pos in o_clusters_info:
                            # Object must match target POS
                            if t_constraint_pos and o_member_pos:
                                if t_constraint_pos != o_member_pos:
                                    continue

                            cluster_relations.add((c_s_label, p, c_o_label))

        # Materialise relationships
        logging.info(f"  - Found {len(cluster_relations)} unique cluster-level relationships.")
        logging.info("  - Optimizing and propagating relationships to all equivalent concepts...")

        cluster_to_urirefs = {
            label: [URIRef(uri) for uri in uris]
            for label, uris in cluster_to_raw_uris.items()
        }

        # Group targets by (start_cluster, predicate) to avoid repeated lookups and batch additions
        grouped_targets = defaultdict(set)
        for c_start, p, c_end in tqdm(cluster_relations, desc="Grouping relations", disable=not self.verbose):
            # if c_start == c_end:
            #     continue
            grouped_targets[(c_start, p)].update(cluster_to_urirefs[c_end])

        logging.info(f"  - Grouped into {len(grouped_targets)} unique (start_cluster, predicate) pairs.")

        def triples_generator():
            for (c_start, p), target_uris in tqdm(grouped_targets.items(), desc="Materialising triples", disable=not self.verbose):
                start_uris = cluster_to_urirefs[c_start]
                for s in start_uris:
                    for o in target_uris:
                        if s != o:
                            yield (s, p, o, self.g)

        self.g.addN(triples_generator())

        logging.info("Relationship coalescing complete.")

    @timer
    def cleanup_graph(self):
        logging.info("Cleaning up intermediate graph data...")
        self.g.remove((None, self.ns.hasURL, None))
        self.g.remove((None, self.ns.source_pos, None))
        self.g.remove((None, self.ns.target_pos, None))

        logging.info("Merging duplicate relationship predicates...")
        weights = {s: str(o) for s, o in self.g.subject_objects(self.ns.weight)}

        if not weights:
            return

        negations = {}
        for s, o in self.g.subject_objects(self.ns.is_negated):
            if s in weights:
                negations[s] = str(o)

        props_map = defaultdict(list)
        for rel_uri, weight_val in tqdm(weights.items(), desc="Analyzing predicates", disable=not self.verbose):
            is_negated_val = negations.get(rel_uri)
            if is_negated_val is not None:
                base_type = self.g.value(rel_uri, RDF.type)
                if base_type:
                    sig = (str(base_type), is_negated_val, weight_val)
                    props_map[sig].append(rel_uri)

        merged_count = 0
        for sig, uris in tqdm(props_map.items(), desc="Merging predicates", disable=not self.verbose):
            if len(uris) > 1:
                uris.sort(key=lambda u: str(u))
                canonical = uris[0]
                duplicates = uris[1:]
                for dup in duplicates:
                    triples_to_move = list(self.g.triples((None, dup, None)))
                    if triples_to_move:
                        for s, _, o in triples_to_move:
                            self.g.add((s, canonical, o))
                        self.g.remove((None, dup, None))
                    self.g.remove((dup, None, None))
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
