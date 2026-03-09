import json
import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict

from tqdm import tqdm

from tools.timer import timer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AbstractConceptGraphClusterer(ABC):
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

        if self.config['general']['should_cache']:
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
            seen = set(adjacency_list)
            j_idx = 0
            while j_idx < len(adjacency_list):
                j = adjacency_list[j_idx]
                adjacency_list_j = adjacency_db[j]
                for k in adjacency_list_j:
                    if i != k and k not in seen:
                        seen.add(k)
                        adjacency_list.append(k)
                j_idx += 1
            adjacency_db[i] = adjacency_list

    @abstractmethod
    def build_adj_list(self):
        pass

    @abstractmethod
    def add_unclustered_concepts(self, cluster_mappings, visited_nodes):
        pass

    @abstractmethod
    def coalesce_relationships(self, cluster_mappings):
        pass

    @abstractmethod
    def cleanup_graph(self):
        pass