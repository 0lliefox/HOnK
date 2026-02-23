import logging
import os
import sys
import argparse
import random
import copy
import yaml
from rdflib import Graph, Namespace

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarking.benchmark import Benchmark
from benchmarking.plot_results import ResultsPlotter
from clustering.cluster_graph_concepts import ConceptGraphClusterer
from tools.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEST_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

class MockGraphManager:
    def __init__(self, graph, ns):
        self.g = graph
        self.ns = ns

class MockBuilder:
    def __init__(self, graph, ns, run_id):
        self.graph_manager = MockGraphManager(graph, ns)
        self.benchmarking = Benchmark("graph_scalability")
        self.memory_benchmarking = Benchmark("graph_memory")
        self.run_id = run_id

def create_subset_graph(full_graph, fraction):
    logging.info(f"--- Creating subset graph for {fraction * 100:.0f}% of data ---")
    
    # Identify all unique subjects (concepts)
    # Note: This includes reified relation nodes if they are subjects in the graph.
    # But we want to sample "Concepts".
    # Concepts usually have a specific type or are subjects of hasURL.
    # Let's assume all subjects are candidates, or we can filter if needed.
    # If we sample reified nodes as "subjects", it might be weird.
    # But for simplicity, let's sample all subjects.
    
    all_subjects = list(set(full_graph.subjects()))
    
    if not all_subjects:
        return Graph(), 0
        
    sample_size = int(len(all_subjects) * fraction)
    sampled_subjects = set(random.sample(all_subjects, sample_size))
    
    logging.info(f"Sampling {sample_size} out of {len(all_subjects)} subjects...")
    
    subset_graph = Graph()
    # Copy namespaces
    for prefix, uri in full_graph.namespaces():
        subset_graph.bind(prefix, uri)
        
    # Add triples where subject is in sampled_subjects
    predicates_seen = set()
    
    for s in sampled_subjects:
        for p, o in full_graph.predicate_objects(s):
            subset_graph.add((s, p, o))
            predicates_seen.add(p)
            
    # Also include properties of the predicates (reified relations)
    # because ConceptGraphClusterer needs source_pos/target_pos attached to predicates
    for p in predicates_seen:
        # If p is a URIRef (not a literal, which it shouldn't be), check if it has properties
        # We add all triples where p is the subject
        for attr, val in full_graph.predicate_objects(p):
            subset_graph.add((p, attr, val))
            
    logging.info(f"Created subset with {len(subset_graph)} triples.")
    return subset_graph, len(sampled_subjects)

def run_experiment(input_file, config, fractions):
    logging.info(f"Loading full graph from {input_file}...")
    full_graph = Graph()
    full_graph.parse(input_file, format='nt') 
    
    base_uri = config['turtle_export']['base_uri']
    ns = Namespace(base_uri)
    
    for frac in fractions:
        subset_graph, concept_count = create_subset_graph(full_graph, frac)
        
        if concept_count == 0:
            continue
            
        id_percentage = int(frac * 100)
        
        mock_builder = MockBuilder(subset_graph, ns, id_percentage)
        mock_builder.benchmarking.add_row(id_percentage, 'concept_count', concept_count)
        
        clusterer = ConceptGraphClusterer(mock_builder, config)
        
        logging.info(f"Running test for {id_percentage}% ({concept_count} subjects)")
        
        try:
            clusterer.run()
            mock_builder.benchmarking.to_csv("graph_clustering_benchmark", data_length=False, append=True)
            if hasattr(mock_builder, 'memory_benchmarking'):
                 mock_builder.memory_benchmarking.to_csv("graph_memory_benchmark", data_length=False, append=True)

        except Exception as e:
            logging.error(f"Error during clustering: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test scalability of graph clustering.")
    parser.add_argument("input_file", help="Path to the input .nt file")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    args = parser.parse_args()

    config = get_config()
    
    for i in range(args.runs):
        run_experiment(args.input_file, config, TEST_FRACTIONS)

    results_file = "graph_clustering_benchmark.csv"
    if os.path.exists(results_file):
        plotter = ResultsPlotter(results_file)
        plotter.run()
    elif os.path.exists(os.path.join("benchmarking", results_file)):
         plotter = ResultsPlotter(os.path.join("benchmarking", results_file))
         plotter.run()

if __name__ == "__main__":
    main()
