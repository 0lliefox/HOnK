import pytest
import yaml
import copy
import os
import shutil
import tempfile
import logging
from benchmarking.benchmark import Benchmark
import build_ontology
from benchmarking.compare_graphs import GraphComparator

@pytest.fixture
def config_setup():
    with open('config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    temp_dir = tempfile.mkdtemp()
    
    yield base_config, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)
    if os.path.exists('temp_config_test.yaml'):
        os.remove('temp_config_test.yaml')

def test_graph_db_equivalence(config_setup):
    base_config, temp_dir = config_setup
    
    graph_filename = 'ontology_graph_clustered_test.nt'
    db_filename = 'ontology_db_clustered_test.nt'
    
    variations = [
        {
            'mode': 'graph',
            'output_file': graph_filename
        },
        {
            'mode': 'db',
            'output_file': db_filename
        }
    ]
    
    benchmarking = Benchmark("timing")
    
    for i, variation in enumerate(variations):
        logging.info(f"Running {variation['mode']} mode...")
        
        iteration_config = copy.deepcopy(base_config)
        
        # Apply settings
        iteration_config['general']['mode'] = variation['mode']
        iteration_config['general']['should_cache'] = True
        iteration_config['db_config']['confirm_clear_db'] = False
        
        # En/disable clustering
        if 'clustering' not in iteration_config:
            iteration_config['clustering'] = {}
        iteration_config['clustering']['enabled'] = True

        if 'turtle_export' not in iteration_config:
            iteration_config['turtle_export'] = {}
        iteration_config['turtle_export']['convert'] = False
        iteration_config['turtle_export']['normalise_pos'] = True
        iteration_config['turtle_export']['output_file'] = variation['output_file']
        
        # Write temp config
        with open('temp_config_test.yaml', 'w') as f:
            yaml.dump(iteration_config, f)
            
        # Run build
        build_ontology.main('temp_config_test.yaml', run_id=i, benchmarking=benchmarking)

    path_graph = os.path.join('ontologies', graph_filename)
    path_db = os.path.join('ontologies', db_filename)
    
    assert os.path.exists(path_graph), f"Graph mode output not found at {path_graph}"
    assert os.path.exists(path_db), f"DB mode output not found at {path_db}"
    
    # Compare graphs
    comparator = GraphComparator(path_graph, path_db, format1='nt', format2='nt', name1='Graph', name2='Database')

    comparator._process_graph(comparator.graph1_path, comparator.format1, 'g1', 1)
    comparator._process_graph(comparator.graph2_path, comparator.format2, 'g2', 2)
    
    # Check basic stats equality
    stats1 = comparator.stats['g1']
    stats2 = comparator.stats['g2']
    
    assert stats1['triples'] == stats2['triples'], f"Triple counts differ: {stats1['triples']} vs {stats2['triples']}"
    assert stats1['nodes'] == stats2['nodes'], f"Node counts differ: {stats1['nodes']} vs {stats2['nodes']}"
    assert stats1['relations'] == stats2['relations'], f"Relation counts differ: {stats1['relations']} vs {stats2['relations']}"
    
    # Check unique samples
    comparator._find_unique_samples()
    
    unique_g1 = len(comparator.unique_samples['g1'])
    unique_g2 = len(comparator.unique_samples['g2'])
    
    if unique_g1 > 0:
        print("Unique to Graph Mode:")
        for t in comparator.unique_samples['g1']:
            print(t)
            
    if unique_g2 > 0:
        print("Unique to DB Mode:")
        for t in comparator.unique_samples['g2']:
            print(t)
            
    assert unique_g1 == 0, f"Found {unique_g1} triples unique to Graph mode"
    assert unique_g2 == 0, f"Found {unique_g2} triples unique to DB mode"
    
    print("Graphs are equivalent.")
