from abc import abstractmethod, ABC
from functools import lru_cache

from rdflib import OWL, Literal

from tools.pickling import PickleManager
from tools.timer import timer
from tools.db_funcs import DBManager


class AbstractLoader(ABC):
    def __init__(self, builder):
        self.builder = builder
        self.config = builder.config
        self.mode = builder.mode
        self.conn = builder.conn if self.mode == 'db' else None
        self.source = None
        self.mappings = self.get_mappings(["edge", "pos"])
        self.graph_manager = self.builder.graph_manager
        self.g = self.graph_manager.g
        self.ns = self.graph_manager.ns
        
        if self.mode == 'db':
            self.db_manager = DBManager(builder)
        else:
            self.db_manager = None

        self._timer_stack = []
        self.pickle_manager = PickleManager(self.builder.should_cache, self.pause_timer, self.resume_timer)

    def pause_timer(self):
        if hasattr(self, '_timer_stack') and self._timer_stack:
            self._timer_stack[-1].pause()

    def resume_timer(self):
        if hasattr(self, '_timer_stack') and self._timer_stack:
            self._timer_stack[-1].resume()

    def get_mappings(self, mapping_types):
        if self.source:
            return {
                k.lower(): v
                for source in [self.source.lower()]
                for m_type in mapping_types
                for k, v in self.graph_manager.load_mappings(self.config['local_files'][f"{source}_{m_type}_mappings_file"]).items()
            }
        return None

    def load_data(self):
        if self.db_manager:
            self.db_manager.source = self.source
        data = self._parse_data_timed()
        self._store_data_timed(data)

    @timer
    def _parse_data_timed(self):
        return self.parse_data()

    @timer
    def _store_data_timed(self, data):
        self.store_data(data)

    @abstractmethod
    def parse_data(self):
        pass

    @abstractmethod
    def store_data(self, data):
        pass

    def does_term_matches_language(self, lang):
        return self.config['general']['language'] == lang

    @lru_cache(maxsize=1024)
    @timer(log=False, threaded=False, independent=True)
    def normalise_data(self, term, pos):
        if self.graph_manager.normalise_pos:
            term = self.normalise_term(term)
            pos = self.get_mapped_pos(pos)
        return term, pos

    @lru_cache(maxsize=1024)
    def get_mapped_pos(self, pos_tag):
        if self.config['turtle_export']['normalise_pos']:
            return self.graph_manager.full_mappings.get(pos_tag, pos_tag)
        else:
            return pos_tag

    def get_or_create_concept(self, term, pos, cursor=None):
        term, pos = self.normalise_data(term, pos)

        if self.mode == 'db':
            return self.db_manager.add_or_get_concept_from_db(term, pos, cursor)
        elif self.mode == 'graph':
            self.graph_manager.add_concept_to_graph(term, pos)
        return None

    def add_relation(self, start, end, rel_type, weight, cursor):
        # Unnecessary to add relation between same terms (e.g. A isRelated A), we know that already?
        if self.mode == 'db':
            if start['id'] != end['id']:
                self.db_manager.add_relation_to_db(start['id'], end['id'], rel_type, weight, cursor)
        elif self.mode == 'graph':
            start_term, start_pos = self.normalise_data(start['term'], start['pos'])
            end_term, end_pos = self.normalise_data(end['term'], end['pos'])
            if start_term != end_term:
                self.graph_manager.add_relation_to_graph(start_term, end_term, rel_type, weight, start_pos, end_pos)

    def add_property(self, concept, c_type, c_value, cursor):
        if self.mode == 'db':
            self.db_manager.add_property_to_db(concept['id'], c_type, c_value, cursor)
        elif self.mode == 'graph':
            term, _ = self.normalise_data(concept['term'], None)
            self.graph_manager.add_property_to_graph(c_type, c_value, term=term)

    def add_url(self, concept, e_url, cursor):
        if self.builder.should_cluster:
            if self.mode == 'db':
                self.db_manager.add_url_to_db(concept['id'], e_url, cursor)
            elif self.mode == 'graph':
                term, pos = self.normalise_data(concept['term'], concept['pos'])
                concept_uri = self.graph_manager.get_safe_uri(term)
                self.graph_manager.add_url_to_graph(concept_uri, e_url, pos)


    @lru_cache(maxsize=1024)
    @timer(log=False, threaded=False, independent=True)
    def normalise_term(self, term):
        term = term.replace('_', ' ').replace('"', '')

        if term != ' ':  # ' ' is added as 'Punctuation', so keeping this
            term = term.strip()

        return term
