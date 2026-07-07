import json
from abc import ABC, abstractmethod

class GraphManager(ABC):
    # Base relations (after mapping) whose same-label endpoints are always
    # degenerate: nothing is a proper part/subtype/instance of itself. Used by the
    # reflexive self-loop filter (Fix C) to drop e.g. `Alabama partOf Alabama`
    # while leaving non-structural same-label edges (eq, relatedTo, the cross-POS
    # `bob_up isA come_up`, which has *different* endpoint labels) untouched.
    STRUCTURAL_IRREFLEXIVE_RELS = frozenset({'partOf', 'isA', 'instanceOf'})

    @staticmethod
    def _normalise_label(label):
        return label.strip().lower() if label is not None else None

    @staticmethod
    def _norm_sense(sense):
        """URI-safe form of a sense discriminator (ConceptNet subsense or WordNet
        lexical_domain), e.g. 'n/wn/act' -> 'n-wn-act', 'paris rer' -> 'paris-rer'."""
        import re
        return re.sub(r'[^a-zA-Z0-9_-]', '-', sense)

    _CN_POS_TO_WORD = {'n': 'noun', 'v': 'verb', 'a': 'adj', 's': 'adj', 'r': 'adv'}

    @classmethod
    def supersense_of_sense(cls, sense):
        """WordNet supersense (lexical domain) for a sense discriminator, used by the
        type-coherence merge guard (Fix B). Returns a comparable `pos.domain` string,
        or None when no clean WordNet supersense is available (bare/POS-only/DBpedia
        senses are left unguarded to avoid false splits).

          WordNet lexical_domain 'noun.location' -> 'noun.location'
          ConceptNet subsense    'n/wn/act'      -> 'noun.act'
          'a/wn', 'n/wp/paris rer', None          -> None
        """
        if not sense:
            return None
        if '.' in sense and '/' not in sense:      # WordNet lexical_domain
            return sense
        parts = sense.split('/')                    # ConceptNet subsense
        if len(parts) >= 3 and parts[1] == 'wn':
            return f"{cls._CN_POS_TO_WORD.get(parts[0], parts[0])}.{parts[2]}"
        return None

    def __init__(self, builder, config):
        self.config = config
        self.builder = builder
        self.mode = config['general']['mode']
        self.should_cluster = config['clustering']['enabled']
        # Lazily built caches for the self-loop filter (Fix C).
        self._label_index = None      # uri.value -> normalised rdfs:label
        self._rel_base_cache = {}     # relation-instance uri.value -> base relation name
        # uri.value -> WordNet supersense, for the type-coherence merge guard (Fix B),
        # populated during ingestion (graph mode). DB mode derives it from the sense column.
        self.node_supersense = {}

        self.equivalent_classes = {}
        self.id_to_uri = {}
        self.annotation_property_list = {}
        self.prop_id = {}
        # Normalisation mode: 'full' (HOnK POS+edge mappings), 'raw' (no mappings),
        # 'canonicalise' (lowercase+lemma only). Falls back to the legacy
        # turtle_export.normalise_pos boolean when 'normalisation' is unset.
        _mode = config['general'].get('normalisation')
        if _mode is None:
            _mode = 'full' if config['turtle_export']['normalise_pos'] else 'raw'
        self.normalisation_mode = _mode
        self.normalise_pos = (_mode == 'full')  # gates POS + edge mapping application
        self.declared_base_properties = set()

        self.full_mappings = {
            k.lower(): v
            for source in self.config['general']['mapping_sources']
            for m_type in ["edge", "pos"]
            for k, v in self.load_mappings(config['local_files'][f"{source}_{m_type}_mappings_file"]).items()
        }

        with open(self.config['local_files']['ontology_classes'], 'r') as f:
            self.ontology_classes = json.load(f)
            self.lemma_mappings = self.ontology_classes['MetaGrammaticalFunction']['classes']['GrammaticalFunction']['classes']

        with open(self.config['local_files']['rejected_classes'], 'r') as f:
            self.rejected_classes = json.load(f)

        with open(self.config['local_files']['pos_tag_classes'], 'r') as f:
            self.pos_tag_mappings = json.load(f)

    def load_mappings(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {k.lower(): v for k, v in data.items()}

    @abstractmethod
    def init_graph(self):
        pass

    @abstractmethod
    def create_classes(self, g):
        pass

    @abstractmethod
    def add_concept_to_graph(self, term, pos, sense=None, cid=None):
        pass

    @abstractmethod
    def add_relation_to_graph(self, start_id, end_id, rel_type, weight, source_pos=None, target_pos=None,
                              start_sense=None, end_sense=None):
        pass

    @abstractmethod
    def add_property_to_graph(self, c_type, c_value, term=None, cid=None, sense=None):
        pass

    @abstractmethod
    def add_url_to_graph(self, concept_uri, url, pos):
        pass

    @abstractmethod
    def add_pos_tag_classes(self, g):
        pass