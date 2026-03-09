import json
from abc import ABC, abstractmethod

class GraphManager(ABC):
    def __init__(self, builder, config):
        self.config = config
        self.builder = builder
        self.mode = config['general']['mode']
        self.should_cluster = config['clustering']['enabled']

        self.equivalent_classes = {}
        self.id_to_uri = {}
        self.annotation_property_list = {}
        self.prop_id = {}
        self.normalise_pos = config['turtle_export']['normalise_pos']
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
    def add_concept_to_graph(self, term, pos, cid=None):
        pass

    @abstractmethod
    def add_relation_to_graph(self, start_id, end_id, rel_type, weight, source_pos=None, target_pos=None):
        pass

    @abstractmethod
    def add_property_to_graph(self, c_type, c_value, term=None, cid=None):
        pass

    @abstractmethod
    def add_url_to_graph(self, concept_uri, url, pos):
        pass

    @abstractmethod
    def add_pos_tag_classes(self, g):
        pass