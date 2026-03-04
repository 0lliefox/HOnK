import json
import logging
import re
from functools import lru_cache
from urllib.parse import quote

from pyoxigraph import Store, NamedNode, Literal, Quad, DefaultGraph
from tqdm import tqdm

from knowledge_bases import ParmenidesLoader
from tools.timer import timer

# Pre-define core RDF/OWL/RDFS/XSD NamedNodes for speed
RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
OWL_CLASS = NamedNode("http://www.w3.org/2002/07/owl#Class")
RDFS_SUBCLASSOF = NamedNode("http://www.w3.org/2000/01/rdf-schema#subClassOf")
RDFS_LABEL = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
XSD_STRING = NamedNode("http://www.w3.org/2001/XMLSchema#string")
XSD_BOOLEAN = NamedNode("http://www.w3.org/2001/XMLSchema#boolean")
XSD_FLOAT = NamedNode("http://www.w3.org/2001/XMLSchema#float")
OWL_ANNOTATIONPROPERTY = NamedNode("http://www.w3.org/2002/07/owl#AnnotationProperty")
OWL_DATATYPEPROPERTY = NamedNode("http://www.w3.org/2002/07/owl#DatatypeProperty")


class Namespace:
    def __init__(self, base_uri):
        self.base_uri = base_uri

    def __getitem__(self, key):
        return NamedNode(f"{self.base_uri}{key}")

    def __getattr__(self, name):
        # This catches dot notation (e.g., self.ns.hasURL)
        return NamedNode(f"{self.base_uri}{name}")

    def __add__(self, other):
        return NamedNode(f"{self.base_uri}{other}")


def pyoxi_literal(val):
    if isinstance(val, bool):
        return Literal("true" if val else "false", datatype=XSD_BOOLEAN)
    elif isinstance(val, float):
        return Literal(str(val), datatype=XSD_FLOAT)
    return Literal(str(val), datatype=XSD_STRING)


class GraphManager:
    def __init__(self, builder, config):
        self.config = config
        self.builder = builder
        self.ns = Namespace(self.config['turtle_export']['base_uri'])
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
            self.lemma_mappings = self.ontology_classes['MetaGrammaticalFunction']['classes']['GrammaticalFunction'][
                'classes']

        with open(self.config['local_files']['rejected_classes'], 'r') as f:
            self.rejected_classes = json.load(f)

        with open(self.config['local_files']['pos_tag_classes'], 'r') as f:
            self.pos_tag_mappings = json.load(f)

        self.g = self.init_graph()

    def init_graph(self):
        g = Store()

        ParmenidesLoader.add_logical_functions(self.config, g)

        self.create_classes(g)
        return g

    @lru_cache(maxsize=1024)
    def get_safe_uri(self, term):
        return NamedNode(self.ns.base_uri + quote(term)) if re.search(r'[^a-zA-Z0-9_-]', term) else self.ns[term]

    def load_mappings(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {k.lower(): v for k, v in data.items()}

    def create_classes(self, g):
        logging.info("Creating ontology classes")
        for parent, children in self.ontology_classes.items():
            children = children['classes']
            parent_uri = self.ns[parent]
            g.add(Quad(parent_uri, RDF_TYPE, OWL_CLASS, DefaultGraph()))
            self.create_sub_classes(parent, children, g, parent_uri)

    def create_sub_classes(self, parent, children, g: Store, parent_uri: NamedNode):
        ignored_keys = ['classes', 'sameAs']
        for child in children:
            child_uri = self.ns[child]
            g.add(Quad(child_uri, RDF_TYPE, OWL_CLASS, DefaultGraph()))
            g.add(Quad(child_uri, RDFS_SUBCLASSOF, parent_uri, DefaultGraph()))
            if 'classes' in children[child]:
                self.create_sub_classes(parent, children[child]['classes'], g, child_uri)
            elif child not in ignored_keys and len(children[child]) > 0:
                self.create_sub_classes(parent, children[child], g, child_uri)

            if len(children[child]) > 0 and 'sameAs' in children[child]:
                self.equivalent_classes[child] = [child, children[child]['sameAs']]

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_concept_to_graph(self, term, pos, cid=None):
        if pos.lower() in self.rejected_classes:
            return

        uri = self.get_safe_uri(term)
        if cid is not None:
            self.id_to_uri[cid] = uri

        for i_pos in self.equivalent_classes.get(pos, [pos]):
            self.g.add(Quad(uri, RDF_TYPE, self.ns[i_pos], DefaultGraph()))
        self.g.add(Quad(uri, RDFS_LABEL, pyoxi_literal(term), DefaultGraph()))

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_relation_to_graph(self, start_id, end_id, rel_type, weight, source_pos=None, target_pos=None):
        try:
            start_uri, end_uri = self.id_to_uri[start_id], self.id_to_uri[end_id]
        except KeyError:
            if isinstance(start_id, int): return
            start_uri, end_uri = self.get_safe_uri(start_id), self.get_safe_uri(end_id)

        if start_uri == end_uri: return

        mapping = self.full_mappings.get(rel_type.lower(), {'rel': rel_type, 'relNegated': False,
                                                            'swap': False}) if self.normalise_pos else {'rel': rel_type,
                                                                                                        'relNegated': False,
                                                                                                        'swap': False}
        rel, is_negated, swap = mapping.get('rel'), mapping.get('relNegated', False), mapping.get('swap', False)

        final_source_pos, final_target_pos = (target_pos, source_pos) if swap else (source_pos, target_pos)
        weight = float(weight)

        # Pyoxigraph Nodes are not natively hashable in the same way, we cache the raw python types instead
        sub_property_list = {
            'rel': rel,
            'is_negated': is_negated,
            'weight': weight
        }
        if self.mode == 'graph' and self.should_cluster:
            sub_property_list['source_pos'] = final_source_pos
            sub_property_list['target_pos'] = final_target_pos

        sub_list_key = frozenset(sub_property_list.items())

        if sub_list_key in self.annotation_property_list:
            rel_uri = self.annotation_property_list[sub_list_key]
            new_annotation_instance = False
        else:
            counter = self.prop_id.get(rel, 0) + 1
            rel_uri = self.get_safe_uri(f"{rel}{counter}")
            self.annotation_property_list[sub_list_key] = rel_uri
            self.prop_id[rel] = counter
            new_annotation_instance = True

        base_prop_uri = self.ns[rel]
        if base_prop_uri.value not in self.declared_base_properties:
            self.g.add(Quad(base_prop_uri, RDF_TYPE, OWL_ANNOTATIONPROPERTY, DefaultGraph()))
            self.declared_base_properties.add(base_prop_uri.value)

        if new_annotation_instance:
            self.g.add(Quad(rel_uri, RDF_TYPE, base_prop_uri, DefaultGraph()))
            self.g.add(Quad(rel_uri, self.ns['is_negated'], pyoxi_literal(is_negated), DefaultGraph()))
            self.g.add(Quad(rel_uri, self.ns['weight'], pyoxi_literal(weight), DefaultGraph()))

            if self.mode == 'graph' and self.should_cluster:
                if final_source_pos: self.g.add(
                    Quad(rel_uri, self.ns['source_pos'], pyoxi_literal(final_source_pos), DefaultGraph()))
                if final_target_pos: self.g.add(
                    Quad(rel_uri, self.ns['target_pos'], pyoxi_literal(final_target_pos), DefaultGraph()))

        s, t = (end_uri, start_uri) if swap else (start_uri, end_uri)
        self.g.add(Quad(s, rel_uri, t, DefaultGraph()))

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_property_to_graph(self, c_type, c_value, term=None, cid=None):
        if cid is not None and cid in self.id_to_uri:
            concept_uri = self.id_to_uri[cid]
        elif term is not None:
            concept_uri = self.get_safe_uri(term)
        else:
            concept_uri = None

        if concept_uri is not None:
            prop_uri = self.get_safe_uri(c_type)
            if isinstance(c_value, str):
                if c_value.lower() == "true":
                    c_value = True
                elif c_value.lower() == "false":
                    c_value = False

            self.g.add(Quad(concept_uri, prop_uri, pyoxi_literal(c_value), DefaultGraph()))
            self.g.add(Quad(prop_uri, RDF_TYPE, OWL_DATATYPEPROPERTY, DefaultGraph()))

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_url_to_graph(self, concept_uri, url, pos):
        self.g.add(Quad(concept_uri, self.ns.hasURL, pyoxi_literal(f"{url}=={pos}"), DefaultGraph()))

    def add_pos_tag_classes(self, g):
        logging.info("Adding POS tag classes")
        g.add(Quad(self.ns['POSTag'], RDF_TYPE, OWL_CLASS, DefaultGraph()))

        new_triples = set()
        for pos_tag, mapping in tqdm(self.pos_tag_mappings.items(), desc="Processing POS tagging"):
            pos_uri = self.get_safe_uri(pos_tag)
            g.add(Quad(pos_uri, RDFS_SUBCLASSOF, self.ns['POSTag'], DefaultGraph()))

            if "classes" not in mapping: continue

            for class_name, class_details in mapping["classes"].items():
                class_uri = self.ns[class_name]
                required_property_uris = [(self.get_safe_uri(prop), pyoxi_literal(True)) for prop in
                                          class_details.get("properties", [])]

                for quad in g.quads_for_pattern(None, RDF_TYPE, class_uri):
                    subject_uri = quad.subject
                    has_all_properties = True

                    for prop_uri, prop_val in required_property_uris:
                        # Check if the required quad exists; if next() hits None, it's missing
                        if next(g.quads_for_pattern(subject_uri, prop_uri, prop_val), None) is None:
                            has_all_properties = False
                            break

                    if has_all_properties:
                        new_triples.add((subject_uri, RDF_TYPE, pos_uri))

        logging.info(f"Identified {len(new_triples)} new POS tag classifications to add.")
        for s, p, o in new_triples:
            if next(g.quads_for_pattern(s, p, o), None) is None:
                g.add(Quad(s, p, o, DefaultGraph()))