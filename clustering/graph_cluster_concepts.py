import logging
from collections import defaultdict
from tqdm import tqdm
from rdflib import Graph, URIRef, OWL, RDF

from knowledge_bases.abstract_loader import timer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphConceptClusterer:
    def __init__(self, builder, graph: Graph):
        self.builder = builder
        self.g = graph
        self.cluster_mappings = {} # concept_uri -> representative_uri

    @timer
    def run(self):
        # 1. Build Adjacency List from owl:sameAs
        adj_list = self.build_adj_list()
        
        # 2. Transitive Closure
        self.transitive_closure(adj_list)
        
        # 3. Build Clusters (Pick Representatives)
        self.build_clusters(adj_list)
        
        # 4. Coalesce Relationships (Append new triples)
        self.coalesce_relationships()
        
        logging.info("Graph-based clustering finished.")
        return self.g

    @timer
    def build_adj_list(self):
        logging.info("  - Building adjacency list from owl:sameAs...")
        adj_list = defaultdict(set)
        
        # Find all owl:sameAs triples
        for s, p, o in tqdm(self.g.triples((None, OWL.sameAs, None)), desc="Processing sameAs"):
            adj_list[s].add(o)
            adj_list[o].add(s)
            
        # Convert sets to lists for transitive closure processing
        return {k: list(v) for k, v in adj_list.items()}

    @timer
    def transitive_closure(self, adjacency_db):
        logging.info("  - Calculating transitive closure on adjacency list...")
        for i in tqdm(adjacency_db.keys(), desc="Transitive closure"):
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
    def build_clusters(self, db):
        logging.info("  - Building clusters and picking representatives...")
        visited_nodes = set()

        # Pre-fetch nodes with rdf:type to identify internal concepts
        internal_concepts = set(self.g.subjects(RDF.type, None))

        for key_node, adjacency_list in tqdm(db.items(), desc="Building clusters"):
            if key_node in visited_nodes:
                continue
                
            cluster_nodes = set(adjacency_list)
            cluster_nodes.add(key_node)
            
            if not cluster_nodes.isdisjoint(visited_nodes):
                continue

            candidates = list(cluster_nodes)
            
            def sort_key(n):
                is_internal = n in internal_concepts
                s = str(n)
                return (not is_internal, len(s), s) # False < True, so internal comes first
            
            candidates.sort(key=sort_key)
            representative = candidates[0]
            
            for node in cluster_nodes:
                if node != representative:
                    self.cluster_mappings[node] = representative
                visited_nodes.add(node)

    @timer
    def coalesce_relationships(self):
        logging.info("  - Appending new triples from coalesced relationships...")
        
        new_triples = []
        
        # Iterate over all triples and generate new triples with representative URIs
        for s, p, o in tqdm(self.g, desc="Generating new triples"):
            # Skip sameAs triples as they are now resolved
            if p == OWL.sameAs:
                continue
                
            new_s = self.cluster_mappings.get(s, s)
            new_o = self.cluster_mappings.get(o, o)

            if (new_s != s or new_o != o) and new_s != new_o:
                new_triples.append((new_s, p, new_o))
        
        logging.info(f"  - Adding {len(new_triples)} new triples to the graph...")
        for t in tqdm(new_triples, desc="Appending new triples"):
            self.g.add(t)

        logging.info("  - Removing owl:sameAs triples...")
        self.g.remove((None, OWL.sameAs, None))
