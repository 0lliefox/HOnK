import logging
import os
import time
from abc import abstractmethod, ABC
from functools import lru_cache, wraps

import psutil
from rdflib import OWL, Literal

from tools.pickling import PickleManager


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def timer(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.__class__.__name__ == 'ConceptClusterer':
            class_name = func.__name__
        else:
            class_name = self.__class__.__name__
        
        start_mem = get_memory_usage()
        logging.info(f"Starting execution of {class_name}... (Memory: {start_mem:.2f} MB)")
        
        self._start_time = time.time()
        self._paused_time = 0
        self._is_paused = False

        result = func(self, *args, **kwargs)

        end_time = time.time()
        end_mem = get_memory_usage()
        duration = end_time - self._start_time - self._paused_time
        mem_diff = end_mem - start_mem
        
        logging.info(f"Finished execution of {class_name} in {duration:.2f} seconds. (Memory change: {mem_diff:+.2f} MB, Final: {end_mem:.2f} MB)")
        
        self.builder.benchmarking.add_row(self.builder.run_id, class_name, duration)
        self.builder.benchmarking.add_row(self.builder.run_id, f"{class_name}_memory_mb", end_mem)
        
        return result
    return wrapper


class AbstractLoader(ABC):
    def __init__(self, builder):
        self.builder = builder
        self.config = builder.config
        self.conn = builder.conn
        self.mode = builder.mode
        self.source = None
        self.mappings = self.get_mappings(["edge", "pos"])
        self.cc_graph = self.builder.cc_graph
        self.g = self.cc_graph.g
        self.ns = self.cc_graph.ns

        self._start_time = 0
        self._paused_time = 0
        self._pause_start_time = 0
        self._is_paused = False
        self.pickle_manager = PickleManager(self.builder.should_cache, self.pause_timer, self.resume_timer)

    def pause_timer(self):
        if not self._is_paused:
            self._pause_start_time = time.time()
            self._is_paused = True

    def resume_timer(self):
        if self._is_paused:
            self._paused_time += time.time() - self._pause_start_time
            self._is_paused = False

    def get_mappings(self, mapping_types):
        if self.source:
            return {
                k.lower(): v
                for source in [self.source.lower()]
                for m_type in mapping_types
                for k, v in self.cc_graph.load_mappings(self.config['local_files'][f"{source}_{m_type}_mappings_file"]).items()
            }
        return None

    @timer
    def load_data_with_timer(self):
        self.load_data()

    @abstractmethod
    def load_data(self):
        pass

    def does_term_matches_language(self, lang):
        return self.config['general']['language'] == lang

    def get_mapped_pos(self, pos_tag):
        if self.config['turtle_export']['normalise_pos']:
            return self.cc_graph.full_mappings.get(pos_tag, pos_tag)
        else:
            return pos_tag

    @lru_cache(maxsize=1024)
    def add_or_get_concept_from_db(self, term, pos, cursor=None):
        standalone = cursor is None
        if standalone: cursor = self.conn.cursor()
        try:
            if not self.config['general']['unique_source']:
                cursor.execute("""
                   INSERT INTO concepts (term, part_of_speech, source)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (term, part_of_speech) DO NOTHING
                   RETURNING id;
                   """,
                   (term, pos, self.source)
               )
            else:
                cursor.execute("""
                   INSERT INTO concepts (term, part_of_speech, source)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (term, part_of_speech, source) DO NOTHING
                   RETURNING id;
                   """,
                   (term, pos, self.source)
                )
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

    def get_or_create_concept(self, term, pos, cursor=None):
        term, pos = term.replace('_', ' ').replace('"', ''), self.get_mapped_pos(pos) if self.cc_graph.normalise_pos else pos

        if term != ' ':  # ' ' is added as 'Punctuation', so keeping this
            term = term.strip()

        if self.mode == 'db':
            return self.add_or_get_concept_from_db(term, pos, cursor)
        elif self.mode == 'graph':
            self.cc_graph.add_concept_to_graph(term, pos)
        return None

    def add_relation_to_db(self, start_id, end_id, rel_type, weight, cursor):
        if start_id == end_id: return

        if not self.config['general']['unique_source']:
            cursor.execute(
                "INSERT INTO relations (start_concept_id, end_concept_id, relation_type, weight, source) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (start_concept_id, end_concept_id, relation_type) DO NOTHING",
                (start_id, end_id, rel_type, weight, self.source)
            )
        else:
            cursor.execute(
                "INSERT INTO relations (start_concept_id, end_concept_id, relation_type, weight, source) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (start_concept_id, end_concept_id, relation_type, source) DO NOTHING",
                (start_id, end_id, rel_type, weight, self.source)
            )

    def add_relation(self, start, end, rel_type, weight, cursor):
        if self.mode == 'db':
            self.add_relation_to_db(start['id'], end['id'], rel_type, weight, cursor)
        elif self.mode == 'graph':
            self.cc_graph.add_relation_to_graph(start['term'], end['term'], rel_type, weight)

    def add_property_to_db(self, c_id, c_type, c_value, cursor):
        if not self.config['general']['unique_source']:
            cursor.execute(
                "INSERT INTO properties (concept_id, type, value, source) VALUES (%s, %s, %s, %s) ON CONFLICT (concept_id, type, value) DO NOTHING",
                (c_id, c_type, c_value, self.source)
            )
        else:
            cursor.execute(
                "INSERT INTO properties (concept_id, type, value, source) VALUES (%s, %s, %s, %s) ON CONFLICT (concept_id, type, value, source) DO NOTHING",
                (c_id, c_type, c_value, self.source)
            )

    def add_property(self, concept, c_type, c_value, cursor):
        if self.mode == 'db':
            self.add_property_to_db(concept['id'], c_type, c_value, cursor)
        elif self.mode == 'graph':
            self.cc_graph.add_property_to_graph(c_type, c_value, term=concept['term'])

    def add_url(self, concept, e_url, cursor):
        if self.mode == 'db':
            cursor.execute(
                "INSERT INTO urls (concept_id, external_url, source) VALUES (%s, %s, %s) ON CONFLICT (concept_id, external_url) DO NOTHING",
                (concept['id'], e_url, self.source)
            )
        elif self.mode == 'graph':
            concept_uri = self.cc_graph.get_safe_uri(concept['term'])
            self.g.add((concept_uri, OWL.sameAs, Literal(e_url)))
