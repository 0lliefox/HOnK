import logging
import time
from abc import abstractmethod, ABC
from functools import lru_cache, wraps

from rdflib import Graph, OWL, RDFS, XSD, Literal, RDF

from tools.pickling import PickleManager


def timer(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.__class__.__name__ == 'ConceptClusterer':
            class_name = func.__name__
        else:
            class_name = self.__class__.__name__
        logging.info(f"Starting execution of {class_name}...")
        self._start_time = time.time()
        self._paused_time = 0
        self._is_paused = False

        result = func(self, *args, **kwargs)

        end_time = time.time()
        duration = end_time - self._start_time - self._paused_time
        logging.info(f"Finished execution of {class_name} in {duration:.2f} seconds.")
        self.builder.benchmarking.add_row(self.builder.run_id, class_name, duration)
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
        self.g = self.init_graph()

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

    def init_graph(self):
        g = Graph()
        g.bind(self.config['turtle_export']['base_prefix'], self.builder.ns)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)

        return g

    def get_mappings(self, mapping_types):
        if self.source:
            return {
                k.lower(): v
                for source in [self.source.lower()]
                for m_type in mapping_types
                for k, v in self.builder.load_mappings(self.config['local_files'][f"{source}_{m_type}_mappings_file"]).items()
            }
        return None

    @timer
    def load_data_with_timer(self):
        self.load_data()
        if self.mode == 'graph':
            self.builder.serialise_graph(f"{self.source}_{self.config['turtle_export']['output_file']}", f"{self.config['turtle_export']['output_file'].split('.')[-1]}", self.g)

    @abstractmethod
    def load_data(self):
        pass

    def does_term_matches_language(self, lang):
        return self.config['general']['language'] == lang

    def get_mapped_pos(self, pos_tag):
        if self.config['turtle_export']['normalise_pos']:
            return self.builder.full_mappings.get(pos_tag, pos_tag)
        else:
            return pos_tag

    def add_concept_to_graph(self, term, pos):
        uri = self.builder.get_safe_uri(term)
        self.g.add((uri, RDF.type, self.builder.ns[pos]))
        self.g.add((uri, RDFS.label, Literal(term, datatype=XSD.string)))

    @lru_cache(maxsize=1024)
    def add_or_get_concept_from_db(self, term, pos, cursor=None):
        term, pos = term.replace('_', ' ').replace('"', ''), self.get_mapped_pos(pos)

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
                           """, (term, pos, self.source))
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
        if self.mode == 'db':
            return self.add_or_get_concept_from_db(term, pos, cursor)
        elif self.mode == 'graph':
            self.add_concept_to_graph(term, pos)
        return None

    def add_relation_to_graph(self, start, end, rel_type):
        self.g.add(
            (
                self.builder.get_safe_uri(start),
                self.builder.get_safe_uri(rel_type),
                self.builder.get_safe_uri(end)
            )
        )

    def add_relation_to_db(self, start_id, end_id, rel_type, weight, cursor):
        if start_id == end_id: return
        cursor.execute(
            "INSERT INTO relations (start_concept_id, end_concept_id, relation_type, weight, source) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (start_concept_id, end_concept_id, relation_type) DO NOTHING",
            (start_id, end_id, rel_type, weight, self.source)
        )

    def add_relation(self, start, end, rel_type, weight, cursor):
        if self.mode == 'db':
            self.add_relation_to_db(start['id'], end['id'], rel_type, weight, cursor)
        elif self.mode == 'graph':
            self.add_relation_to_graph(start['term'], end['term'], rel_type)

    def add_property_to_graph(self, term, c_type, c_value):
        prop_uri = self.builder.get_safe_uri(c_type)
        self.g.add((self.builder.get_safe_uri(term), prop_uri, Literal(c_value)))
        self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))

    def add_property_to_db(self, c_id, c_type, c_value, cursor):
        cursor.execute(
            "INSERT INTO properties (concept_id, type, value, source) VALUES (%s, %s, %s, %s) ON CONFLICT (concept_id, type, value) DO NOTHING",
            (c_id, c_type, c_value, self.source)
        )

    def add_property(self, concept, c_type, c_value, cursor):
        if self.mode == 'db':
            self.add_property_to_db(concept['id'], c_type, c_value, cursor)
        elif self.mode == 'graph':
            self.add_property_to_graph(concept['term'], c_type, c_value)

    def add_url(self, c_id, e_url, cursor):
        if self.mode == 'db':
            cursor.execute(
                "INSERT INTO urls (concept_id, external_url, source) VALUES (%s, %s, %s) ON CONFLICT (concept_id, external_url) DO NOTHING",
                (c_id, e_url, self.source)
            )