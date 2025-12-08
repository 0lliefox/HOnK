import yaml
import subprocess
import copy
import os

import build_ontology


def run_script(config):
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    try:
        build_ontology.main('temp_config.yaml')
    finally:
        os.remove('temp_config.yaml')

def main():
    with open('config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    variations = [
        {
            'general': {
                'mode': 'graph'
            },
            'clustering': {
                'enabled': 'true'
            },
            'turtle_export': {
                'normalise_pos': 'true',
                'output_file': 'ontology_norm_graph.nt'
            },
        },
        {
            'general': {
                'mode': 'graph'
            },
            'clustering': {
                'enabled': 'true'
            },
            'turtle_export': {
                'normalise_pos': 'false',
                'output_file': 'ontology_orig_graph.nt'
            },
        },
        {
            'general': {
                'mode': 'db'
            },
            'clustering': {
                'enabled': 'true'
            },
            'turtle_export': {
                'normalise_pos': 'true',
                'output_file': 'ontology_norm_db_clust.nt'
            },
        },
        {
            'general': {
                'mode': 'db'
            },
            'clustering': {
                'enabled': 'false'
            },
            'turtle_export': {
                'normalise_pos': 'true',
                'output_file': 'ontology_norm_db_unclust.nt'
            },
        },
    ]

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

        run_script(iteration_config)

if __name__ == '__main__':
    main()
