import logging
from collections import defaultdict
from tqdm import tqdm
from rdflib import Graph, URIRef, OWL, RDF

from clustering.cluster_concepts import ConceptClusterer
from knowledge_bases.abstract_loader import timer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphConceptClusterer:
    def __init__(self, builder, graph: Graph):
        self.builder = builder
        self.g = graph
        self.cluster_mappings = {} # concept_uri -> representative_uri

    @timer
    def run(self):
        logging.info("Starting graph-based clustering...")
        
        # 1. Build Adjacency List from owl:sameAs
        adj_list = self.build_adj_list()
        
        # 2. Transitive Closure
        ConceptClusterer.transitive_closure(adj_list)
        
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
    def build_clusters(self, db):
        logging.info("  - Building clusters and picking representatives...")
        visited_nodes = set()

        for key_node, adjacency_list in tqdm(db.items(), desc="Building clusters"):
            if key_node in visited_nodes:
                continue
                
            cluster_nodes = set(adjacency_list)
            cluster_nodes.add(key_node)
            
            if not cluster_nodes.isdisjoint(visited_nodes):
                continue

            # Pick a representative for the cluster
            # Heuristic: Pick the shortest URI, or alphabetically first
            # Converting to string to sort ensures determinism
            sorted_nodes = sorted(list(cluster_nodes), key=lambda n: str(n))
            representative = sorted_nodes[0]
            
            for node in cluster_nodes:
                if node != representative:
                    self.cluster_mappings[node] = representative
                visited_nodes.add(node)
        
        # Singleton clusters don't need mapping (they map to themselves implicitly)

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
            
            # If a new triple is formed (i.e., at least one of s or o was mapped)
            # and it's not a self-loop
            if (new_s != s or new_o != o) and new_s != new_o:
                new_triples.append((new_s, p, new_o))
        
        logging.info(f"  - Adding {len(new_triples)} new triples to the graph...")
        for t in tqdm(new_triples, desc="Appending new triples"):
            self.g.add(t)

        logging.info("  - Removing owl:sameAs triples...")
        self.g.remove((None, OWL.sameAs, None))
