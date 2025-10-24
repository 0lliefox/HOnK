import logging

import yaml


def get_config():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Configuration file 'config.yaml' not found")
        exit(1)
    return config