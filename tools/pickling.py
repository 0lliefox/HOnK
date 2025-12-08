import logging
import os
import pickle

from tools.config import get_config


class PickleManager:
    def __init__(self, should_cache, pause_timer=None, resume_timer=None):
        self.should_cache = should_cache
        self.pause_timer = pause_timer
        self.resume_timer = resume_timer

    def save(self, filepath, data):
        if self.should_cache:
            if self.pause_timer:
                self.pause_timer()

            cache_dir = self.get_cache_dir(filepath)
            with open(cache_dir, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"Saved to pickle file, '{cache_dir}'")

            if self.resume_timer:
                self.resume_timer()

    def load(self, filepath):
        if self.should_cache:
            cache_dir = self.get_cache_dir(filepath)
            if os.path.exists(cache_dir):
                logging.info(f"Loading file from cache, '{cache_dir}'")
                with open(cache_dir, 'rb') as f:
                    data = pickle.load(f)
                logging.info("Finished loading from cache")
                return data
        return None

    def get_cache_dir(self, filepath):
        config = get_config()
        file_name = os.path.splitext(filepath)[0].split('/')[-1]
        cache_dir = config['local_files']['cache']
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        return f"{cache_dir}/{file_name}.pkl"

