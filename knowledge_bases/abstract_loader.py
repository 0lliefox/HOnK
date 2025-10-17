import logging
import os
import pickle
import time
from abc import abstractmethod, ABC
from functools import lru_cache, wraps


def timer(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        class_name = self.__class__.__name__
        logging.info(f"Starting execution of {class_name}.load_data...")
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Finished execution of {class_name}.load_data in {duration:.2f} seconds.")
        self.builder.benchmarking.add_row(self.builder.run_id, class_name, duration)
        return result
    return wrapper


class AbstractLoader(ABC):
    def __init__(self, builder):
        self.builder = builder
        self.config = builder.config
        self.conn = builder.conn
        self.mappings = builder.full_mappings

    @timer
    def load_data(self):
        self._load_data_implementation()

    @abstractmethod
    def _load_data_implementation(self):
        pass

    def _is_valid_term_for_language(self, lang):
        return self.config['general']['language'] == lang

    def _get_mapped_pos(self, pos_tag):
        return self.builder.full_mappings.get(pos_tag, pos_tag)

    @lru_cache(maxsize=1024)
    def get_or_create_concept(self, term, pos, source, cursor=None):
        term, pos = term.replace('_', ' '), self._get_mapped_pos(pos)

        if term != ' ':  # ' ' is added as 'Punctuation', so keeping this
            term = term.strip()

        standalone = cursor is None
        if standalone: cursor = self.conn.cursor()
        try:
            cursor.execute("""
                           INSERT INTO concepts (term, part_of_speech, source)
                           VALUES (%s, %s, %s)
                           ON CONFLICT (term, part_of_speech) DO NOTHING
                           RETURNING id;
                           """, (term, pos, source))
            result = cursor.fetchone()
            if result:
                # if standalone: self.conn.commit()
                return result[0]
            else:
                # If the insert did nothing (due to conflict), fetch the existing ID
                cursor.execute("SELECT id FROM concepts WHERE term = %s AND part_of_speech = %s", (term, pos))
                return cursor.fetchone()[0]
        finally:
            if standalone: cursor.close()

    def add_relation(self, start_id, end_id, rel_type, weight, source, cursor):
        if start_id == end_id: return
        cursor.execute(
            "INSERT INTO relations (start_concept_id, end_concept_id, relation_type, weight, source) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (start_concept_id, end_concept_id, relation_type) DO NOTHING",
            (start_id, end_id, rel_type, weight, source)
        )

    def add_property(self, c_id, c_type, c_value, source, cursor):
        cursor.execute(
            "INSERT INTO properties (concept_id, type, value, source) VALUES (%s, %s, %s, %s) ON CONFLICT (concept_id, type, value) DO NOTHING",
            (c_id, c_type, c_value, source)
        )

    def add_url(self, c_id, e_url, source, cursor):
        cursor.execute(
            "INSERT INTO urls (concept_id, external_url, source) VALUES (%s, %s, %s) ON CONFLICT (concept_id, external_url) DO NOTHING",
            (c_id, e_url, source)
        )

    def save_to_pickle(self, filepath, data):
        cache_dir = self.get_cache_dir(filepath)
        with open(cache_dir, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Saved to pickle file, '{cache_dir}'")

    def load_from_pickle(self, filepath):
        cache_dir = self.get_cache_dir(filepath)
        if os.path.exists(cache_dir):
            logging.info(f"Loading file from cache, '{cache_dir}'")
            with open(cache_dir, 'rb') as f:
                data = pickle.load(f)
            logging.info("Finished loading graph from cache")
            return data
        else:
            return None

    def get_cache_dir(self, filepath):
        file_name = os.path.splitext(filepath)[0].split('/')[-1]
        return f"{self.config['local_files']['cache']}/{file_name}.pkl"