from abc import abstractmethod, ABC
from functools import lru_cache

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
                for k, v in
                self.graph_manager.load_mappings(self.config['local_files'][f"{source}_{m_type}_mappings_file"]).items()
            }
        return None

    def load_data(self):
        if self.db_manager:
            self.db_manager.source = self.source

        data = self._parse_data_timed()
        self._store_data_timed(data)

        # Flush remaining bulk
        if self.mode == 'db' and getattr(self.db_manager, 'use_bulk', False):
            self.db_manager.flush_all()

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
        # Basic surface hygiene applies in every mode so labels are well-formed.
        term = self.normalise_term(term)
        mode = self.graph_manager.normalisation_mode
        if mode == 'full':
            pos = self.get_mapped_pos(pos) if pos is not None else None
        elif mode == 'canonicalise':
            term = self.canonicalise_term(term, pos)
        # 'raw': keep the cleaned source label and original POS as-is.
        return term, pos

    @lru_cache(maxsize=1024)
    def get_mapped_pos(self, pos_tag):
        if self.graph_manager.normalise_pos:
            return self.graph_manager.full_mappings.get(pos_tag, pos_tag)
        else:
            return pos_tag

    def get_or_create_concept(self, term, pos, cursor=None, sense=None):
        # `sense` is an optional source-provided discriminator (ConceptNet subsense
        # or WordNet lexical_domain) that scopes a distinct sense node; when None the
        # concept resolves to the bare-lemma hub (Fix A). It is deliberately not run
        # through normalise_data so sense identity is stable across ablation modes.
        # `sense` is keyword-last so existing positional `cursor` callers are unaffected.
        term, pos = self.normalise_data(term, pos)

        if self.mode == 'db':
            return self.db_manager.add_or_get_concept_from_db(term, pos, cursor, sense)
        elif self.mode == 'graph':
            self.graph_manager.add_concept_to_graph(term, pos, sense)
        return None

    def add_relation(self, start, end, rel_type, weight, cursor):
        # Unnecessary to add relation between same terms (e.g. A isRelated A), we know that already?
        if self.mode == 'db':
            if start['id'] != end['id']:
                self.db_manager.add_relation_to_db(start['id'], end['id'], rel_type, weight, cursor)
        elif self.mode == 'graph':
            start_term, start_pos = self.normalise_data(start['term'], start['pos'])
            end_term, end_pos = self.normalise_data(end['term'], end['pos'])
            start_sense, end_sense = start.get('sense'), end.get('sense')
            # Distinct sense nodes of the same lemma are distinct endpoints, so the
            # loop-avoidance guard must also consider the sense discriminator.
            if start_term != end_term or start_sense != end_sense \
                    or (self.builder.should_cluster and start_pos != end_pos):
                self.graph_manager.add_relation_to_graph(
                    start_term, end_term, rel_type, weight, start_pos, end_pos, start_sense, end_sense)

    def add_property(self, concept, c_type, c_value, cursor):
        if self.mode == 'db':
            self.db_manager.add_property_to_db(concept['id'], c_type, c_value, cursor)
        elif self.mode == 'graph':
            term, _ = self.normalise_data(concept['term'], None)
            self.graph_manager.add_property_to_graph(c_type, c_value, term=term, sense=concept.get('sense'))

    def add_url(self, concept, e_url, cursor):
        if self.builder.should_cluster:
            if self.mode == 'db':
                self.db_manager.add_url_to_db(concept['id'], e_url, cursor)
            elif self.mode == 'graph':
                term, pos = self.normalise_data(concept['term'], concept['pos'])
                concept_uri = self.graph_manager.get_safe_uri(term, concept.get('sense'))
                self.graph_manager.add_url_to_graph(concept_uri, e_url, pos)

    @lru_cache(maxsize=1024)
    def normalise_term(self, term):
        term = term.replace('_', ' ').replace('"', '')

        if term != ' ':  # ' ' is added as 'Punctuation', so keeping this
            term = term.strip()

        return term

    @lru_cache(maxsize=1024)
    def canonicalise_term(self, term, pos):
        """Lightweight surface canonicalisation for the 'canonicalise' ablation
        mode: lowercase + WordNet lemmatisation (matching only, no HOnK mappings).
        Isolates trivial string variation from HOnK's semantic alignment."""
        term = term.lower()
        if not hasattr(self, '_lemmatizer'):
            import nltk
            from nltk.stem import WordNetLemmatizer
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            self._lemmatizer = WordNetLemmatizer()
        nltk_pos = 'n'
        if pos:
            p = pos.lower()
            if 'verb' in p:
                nltk_pos = 'v'
            elif 'adj' in p:
                nltk_pos = 'a'
            elif 'adv' in p:
                nltk_pos = 'r'
        try:
            return self._lemmatizer.lemmatize(term, pos=nltk_pos)
        except Exception:
            return term