import yaml
import copy
import os

import build_ontology
from benchmarking.benchmark import Benchmark


def main():
    with open('config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    variations = [
        {
            'general': {
                'mode': 'graph',
                'should_cache': False
            },
            'clustering': {
                'enabled': True
            },
            'turtle_export': {
                'convert': False,
                'normalise_pos': True,
                'output_file': 'ontology_norm_graph.nt'
            },
        },
        {
            'general': {
                'mode': 'graph',
                'should_cache': False
            },
            'clustering': {
                'enabled': True
            },
            'turtle_export': {
                'convert': False,
                'normalise_pos': False,
                'output_file': 'ontology_orig_graph.nt'
            },
        },
        {
            'general': {
                'mode': 'db',
                'should_cache': False
            },
            'clustering': {
                'enabled': True
            },
            'turtle_export': {
                'convert': False,
                'normalise_pos': True,
                'output_file': 'ontology_norm_db_clust.nt'
            },
        },
        {
            'general': {
                'mode': 'db',
                'should_cache': False
            },
            'clustering': {
                'enabled': False
            },
            'turtle_export': {
                'convert': False,
                'normalise_pos': True,
                'output_file': 'ontology_norm_db_unclust.nt'
            },
        },
        {
            'general': {
                'mode': 'db',
                'should_cache': False
            },
            'clustering': {
                'enabled': True
            },
            'turtle_export': {
                'convert': False,
                'normalise_pos': True,
                'output_file': 'ontology_norm_db_clust.ttl'
            },
        },
        {
            'general': {
                'mode': 'db',
                'should_cache': False
            },
            'clustering': {
                'enabled': False
            },
            'turtle_export': {
                'convert': False,
                'normalise_pos': True,
                'output_file': 'ontology_norm_db_unclust.ttl'
            },
        },
    ]

    benchmarking = Benchmark("timing")
    for i, variation in enumerate(variations):
        print(f"Running iteration {i+1}/{len(variations)}")

        iteration_config = copy.deepcopy(base_config)
        for key, value in variation.items():
            if isinstance(value, dict) and key in iteration_config:
                iteration_config[key].update(value)
            else:
                iteration_config[key] = value

        if 'general' not in iteration_config:
            iteration_config['general'] = {}
        iteration_config['general']['confirm_clear_db'] = False

        with open('temp_config.yaml', 'w') as f:
            yaml.dump(iteration_config, f)

        try:
            build_ontology.main('temp_config.yaml', run_id=i, benchmarking=benchmarking)
        finally:
            os.remove('temp_config.yaml')

if __name__ == '__main__':
    main()
