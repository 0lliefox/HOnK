import json
import yaml
import os
import sys
import argparse
import copy

CONFIG_FILE = "config.yaml"
CONFIGURATIONS_FILE = "configurations_combined.json.json"

def deep_merge(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base

def main():
    parser = argparse.ArgumentParser(description='Generate configuration for a specific run.')
    parser.add_argument('--iteration', type=int, required=True, help='Current iteration index (0-based).')
    parser.add_argument('--config-index', type=int, required=True, help='Index of the configuration in configurations.json.')
    args = parser.parse_args()

    if not os.path.exists(CONFIGURATIONS_FILE):
        print(f"Error: {CONFIGURATIONS_FILE} not found.")
        sys.exit(1)

    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} not found.")
        sys.exit(1)

    with open(CONFIGURATIONS_FILE, 'r') as f:
        configurations = json.load(f)

    with open(CONFIG_FILE, 'r') as f:
        base_config = yaml.safe_load(f)

    if args.config_index >= len(configurations):
        print(f"Error: Config index {args.config_index} out of range.")
        sys.exit(1)

    config_item = configurations[args.config_index]
    
    # Auto-generate run_id based on index and iteration
    unique_run_id = (args.iteration * len(configurations)) + args.config_index
    
    overrides = config_item.get('config_overrides', {})
    
    current_config = copy.deepcopy(base_config)
    deep_merge(current_config, overrides)
        
    temp_config_file = f"temp_config_{unique_run_id}.yaml"
    
    with open(temp_config_file, 'w') as f:
        yaml.dump(current_config, f)
        
    # Print the generated config file path and run_id to stdout so the shell script can capture it
    print(f"{temp_config_file} {unique_run_id}")

if __name__ == "__main__":
    main()
