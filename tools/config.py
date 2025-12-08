import logging
import yaml

def get_config(config_file='config.yaml'):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_file}' not found")
        exit(1)
    return config
