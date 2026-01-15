import json
import logging
import re
from functools import lru_cache
from urllib.parse import quote

from rdflib import Literal, URIRef, Namespace, Graph, OWL, RDFS, XSD, RDF
from tqdm import tqdm

from knowledge_bases import ParmenidesLoader


class CCGraph:
    def __init__(self, config):
        self.config = config
        self.ns = Namespace(self.config['turtle_export']['base_uri'])

        self.equivalent_classes = {}  # Map of classes that should be added to a concept as equivalent classes
        self.id_to_uri = {}
        self.annotation_property_list = {}
        self.prop_id = {}  # Starting ID for reified relationships
        self.normalise_pos = config['turtle_export']['normalise_pos']
        self.declared_base_properties = set()

        sources = self.config['general']['sources']
        mapping_types = ["edge", "pos"]
        self.full_mappings = {
            k.lower(): v
            for source in sources
            for m_type in mapping_types
            for k, v in self.load_mappings(config['local_files'][f"{source}_{m_type}_mappings_file"]).items()
        }

        # Class mappings
        with open(self.config['local_files']['ontology_classes'], 'r') as f:
            self.ontology_classes = json.load(f)
            self.lemma_mappings = self.ontology_classes['MetaGrammaticalFunction']['classes']['GrammaticalFunction']['classes']

        # Rejected classes
        with open(self.config['local_files']['rejected_classes'], 'r') as f:
            self.rejected_classes = json.load(f)

        # POS tag mappings
        with open(self.config['local_files']['pos_tag_classes'], 'r') as f:
            self.pos_tag_mappings = json.load(f)

        self.g = self.init_graph()

    def init_graph(self):
        g = Graph()
        g.bind(self.config['turtle_export']['base_prefix'], self.ns)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)

        # Add logical functions
        ParmenidesLoader.add_logical_functions(self.config, g)

        # Setup classes from JSON
        self.create_classes(g)

        return g

    @lru_cache(maxsize=1024)
    def get_safe_uri(self, term):
        return URIRef(self.ns + quote(term)) if re.search(r'[^a-zA-Z0-9_-]', term) else self.ns[term]

    def load_mappings(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {k.lower(): v for k, v in data.items()}

    def create_classes(self, g):
        logging.info("Creating ontology classes")
        for parent, children in self.ontology_classes.items():
            children = children['classes']
            parent_uri = self.ns[parent]
            g.add((parent_uri, RDF.type, OWL.Class))
            self.create_sub_classes(parent, children, g, parent_uri)

    def create_sub_classes(self, parent, children, g: Graph, parent_uri: URIRef):
        ignored_keys = ['classes', 'sameAs']
        for child in children:
            child_uri = self.ns[child]
            g.add((child_uri, RDF.type, OWL.Class))
            g.add((child_uri, RDFS.subClassOf, parent_uri))
            if 'classes' in children[child]:
                self.create_sub_classes(parent, children[child]['classes'], g, child_uri)
            elif child not in ignored_keys and len(children[child]) > 0:
                self.create_sub_classes(parent, children[child], g, child_uri)

            if len(children[child]) > 0 and 'sameAs' in children[child]:
                self.equivalent_classes[child] = [child, children[child]['sameAs']]

    def add_concept_to_graph(self, term, pos, cid=None):
        if pos.lower() in self.rejected_classes:
            return

        uri = self.get_safe_uri(term)

        if cid is not None:
            self.id_to_uri[cid] = uri

        for i_pos in self.equivalent_classes.get(pos, [pos]):
            self.g.add((uri, RDF.type, self.ns[i_pos]))
        self.g.add((uri, RDFS.label, Literal(term, datatype=XSD.string)))

    def add_relation_to_graph(self, start_id, end_id, rel_type, weight):
        try:
            start_uri, end_uri = self.id_to_uri[start_id], self.id_to_uri[end_id]
        except KeyError as e:
            if isinstance(start_id, int):
                return
            start_uri, end_uri = self.get_safe_uri(start_id), self.get_safe_uri(end_id)

        if self.normalise_pos:
            mapping = self.full_mappings.get(rel_type.lower(), {'rel': rel_type, 'relNegated': False, 'swap': False})
        else:
            mapping = {'rel': rel_type, 'relNegated': False, 'swap': False}

        rel, is_negated, swap = mapping.get('rel'), mapping.get('relNegated', False), mapping.get('swap', False)

        sub_property_list = {'rel': rel, 'is_negated': Literal(is_negated), 'weight': Literal(weight)}
        sub_list_key = frozenset(sub_property_list.items())

        if sub_list_key in self.annotation_property_list:
            rel_uri = self.annotation_property_list[sub_list_key]
            new_annotation_instance = False
        else:
            counter = self.prop_id.get(rel, 1)
            rel_uri = self.get_safe_uri(f"{rel}{counter}")
            self.annotation_property_list[sub_list_key] = rel_uri
            self.prop_id[rel] = counter if counter == 1 else counter + 1
            new_annotation_instance = True

        base_prop_uri = self.ns[rel]
        if base_prop_uri not in self.declared_base_properties:
            self.g.add((base_prop_uri, RDF.type, OWL.AnnotationProperty))
            self.declared_base_properties.add(base_prop_uri)

        if new_annotation_instance:
            self.g.add((rel_uri, RDF.type, base_prop_uri))
            self.g.add((rel_uri, self.ns['is_negated'], Literal(is_negated)))
            self.g.add((rel_uri, self.ns['weight'], Literal(float(weight))))

        s, t = (end_uri, start_uri) if swap else (start_uri, end_uri)
        self.g.add((s, rel_uri, t))

    def add_property_to_graph(self, c_type, c_value, term=None, cid=None):
        if cid is not None and cid in self.id_to_uri:
            concept_uri = self.id_to_uri[cid]
        elif term is not None:
            concept_uri = self.get_safe_uri(term)
        else:
            concept_uri = None

        if concept_uri is not None:
            prop_uri = self.get_safe_uri(c_type)
            self.g.add((concept_uri, prop_uri, Literal(c_value)))
            self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))

    def add_pos_tag_classes(self, g):
        logging.info("Adding POS tag classes")
        g.add((self.ns['POSTag'], RDF.type, OWL.Class))

        new_triples = set()
        for pos_tag, mapping in tqdm(self.pos_tag_mappings.items(), desc="Processing POS tagging"):
            pos_uri = self.get_safe_uri(pos_tag)
            g.add((pos_uri, RDFS.subClassOf, self.ns['POSTag']))

            if "classes" not in mapping:
                continue

            for class_name, class_details in mapping["classes"].items():
                class_uri = self.ns[class_name]

                required_property_uris = [
                    (self.get_safe_uri(prop), Literal(True))
                    for prop in class_details.get("properties", [])
                ]

                candidate_subjects = g.subjects(RDF.type, class_uri)
                for subject_uri in candidate_subjects:
                    has_all_properties = all(
                        (subject_uri, prop_uri, prop_val) in g
                        for prop_uri, prop_val in required_property_uris
                    )

                    if has_all_properties:
                        new_triples.add((subject_uri, RDF.type, pos_uri))

        logging.info(f"Identified {len(new_triples)} new POS tag classifications to add.")
        for triple in new_triples:
            if triple not in g:
                g.add(triple)