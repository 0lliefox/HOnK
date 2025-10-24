import logging
import os
import pickle

import yaml

from tools.config import get_config


def save_to_pickle(filepath, data):
    cache_dir = get_cache_dir(filepath)
    with open(cache_dir, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Saved to pickle file, '{cache_dir}'")


def load_from_pickle(filepath):
    cache_dir = get_cache_dir(filepath)
    if os.path.exists(cache_dir):
        logging.info(f"Loading file from cache, '{cache_dir}'")
        with open(cache_dir, 'rb') as f:
            data = pickle.load(f)
        logging.info("Finished loading graph from cache")
        return data
    else:
        return None


def get_cache_dir(filepath):
    config = get_config()
    file_name = os.path.splitext(filepath)[0].split('/')[-1]
    cache_dir = config['local_files']['cache']
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    return f"{cache_dir}/{file_name}.pkl"
