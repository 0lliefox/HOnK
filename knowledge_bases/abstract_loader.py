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

        self.batch_size = self.config['general']['batch_size']
        self._clear_batches()

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

    @timer(log=False, threaded=False, independent=True, memory=False)
    @lru_cache(maxsize=1024)
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
    def normalise_term(self, term):
        term = term.replace('_', ' ').replace('"', '')

        if term != ' ':  # ' ' is added as 'Punctuation', so keeping this
            term = term.strip()

        return term

    def _clear_batches(self):
        self.batch_concepts = set()
        self.batch_relations = []
        self.batch_urls = []
        self.batch_properties = []

    def queue_concept(self, term, pos):
        term, pos = self.normalise_data(term, pos)
        self.batch_concepts.add((term, pos))
        return term, pos

    def queue_relation(self, start_term, start_pos, end_term, end_pos, rel_type, weight):
        self.batch_relations.append((start_term, start_pos, end_term, end_pos, rel_type, weight))

    def queue_url(self, term, pos, e_url):
        if self.builder.should_cluster:
            self.batch_urls.append((term, pos, e_url))

    def queue_property(self, term, pos, c_type, c_value):
        self.batch_properties.append((term, pos, c_type, c_value))

    @timer(log=False, threaded=False, independent=False, memory=False)
    def flush_batch(self, cursor=None):
        if not self.batch_concepts and not self.batch_relations and not self.batch_urls and not self.batch_properties:
            return

        if self.mode == 'db':
            # Bulk resolve concepts
            concept_map = self.db_manager.get_or_create_concepts_bulk(self.batch_concepts, cursor)

            db_relations = []
            db_urls = []
            db_props = []

            # Map IDs for Relations
            for start_term, start_pos, end_term, end_pos, rel_type, weight in self.batch_relations:
                start_id = concept_map.get((start_term, start_pos))
                end_id = concept_map.get((end_term, end_pos))
                if start_id and end_id and start_id != end_id:
                    db_relations.append((start_id, end_id, rel_type, weight, self.source))

            # Map IDs for URLs
            for term, pos, url in self.batch_urls:
                start_id = concept_map.get((term, pos))
                if start_id:
                    db_urls.append((start_id, url, self.source))

            # Map IDs for Properties
            for term, pos, c_type, c_value in self.batch_properties:
                start_id = concept_map.get((term, pos))
                if start_id:
                    db_props.append((start_id, c_type, c_value, self.source))

            # Execute bulk inserts
            if db_relations: self.db_manager.add_relations_bulk(db_relations, cursor)
            if db_urls: self.db_manager.add_urls_bulk(db_urls, cursor)
            if db_props: self.db_manager.add_properties_bulk(db_props, cursor)

            # Commit the transaction immediately to free up server RAM
            self.conn.commit()

        elif self.mode == 'graph':
            for term, pos in self.batch_concepts:
                self.graph_manager.add_concept_to_graph(term, pos)
            for st, sp, et, ep, rel, w in self.batch_relations:
                if st != et:
                    self.graph_manager.add_relation_to_graph(st, et, rel, w, sp, ep)
            for term, pos, url in self.batch_urls:
                concept_uri = self.graph_manager.get_safe_uri(term)
                self.graph_manager.add_url_to_graph(concept_uri, url, pos)
            for term, pos, c_type, c_value in self.batch_properties:
                self.graph_manager.add_property_to_graph(c_type, c_value, term=term)

        # Reset buffers
        self._clear_batches()