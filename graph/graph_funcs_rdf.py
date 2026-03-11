import logging
import re
from functools import lru_cache
from urllib.parse import quote
from tqdm import tqdm

from rdflib import Literal as RDFLiteral, URIRef, Namespace as RDFNamespace, Graph, OWL, RDFS, XSD, RDF
from knowledge_bases import ParmenidesLoader
from tools.timer import timer
from graph.graph_funcs import GraphManager


class RDFGraphManager(GraphManager):
    def __init__(self, builder, config):
        super().__init__(builder, config)
        self.ns = RDFNamespace(self.config['turtle_export']['base_uri'])
        # Stores (uri, rel_uri, source_pos, target_pos) for same-term cross-POS relations
        # that cannot be written as self-loop triples. Used by the clusterer to recover
        # the equivalent cluster-level relations (e.g. come_up StativeVerb isA come_up MotionVerb).
        self.cross_pos_hints = []
        self.g = self.init_graph()

    def init_graph(self):
        g = Graph()
        g.bind(self.config['turtle_export']['base_prefix'], self.ns)
        g.bind("owl", OWL)
        g.bind("rdfs", RDFS)
        g.bind("xsd", XSD)

        if 'parmenides' in self.config['general']['sources']:
            ParmenidesLoader.add_logical_functions(self.config, g)

        self.create_classes(g)
        return g

    @lru_cache(maxsize=1024)
    def get_safe_uri(self, term):
        return URIRef(self.ns + quote(term)) if re.search(r'[^a-zA-Z0-9_-]', term) else self.ns[term]

    def create_classes(self, g):
        logging.info("Creating ontology classes (RDFLib)")
        for parent, children in self.ontology_classes.items():
            children = children['classes']
            parent_uri = self.ns[parent]
            g.add((parent_uri, RDF.type, OWL.Class))
            self.create_sub_classes(parent, children, g, parent_uri)

    def create_sub_classes(self, parent, children, g, parent_uri):
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

    @timer(log=False, threaded=False, independent=True, memory=False)
    def add_concept_to_graph(self, term, pos, cid=None):
        if pos.lower() in self.rejected_classes:
            return

        uri = self.get_safe_uri(term)
        if cid is not None:
            self.id_to_uri[cid] = uri

        for i_pos in self.equivalent_classes.get(pos, [pos]):
            self.g.add((uri, RDF.type, self.ns[i_pos]))
        self.g.add((uri, RDFS.label, RDFLiteral(term, datatype=XSD.string)))

    @timer(log=False, threaded=False, independent=True, memory=False)
    def add_relation_to_graph(self, start_id, end_id, rel_type, weight, source_pos=None, target_pos=None):
        try:
            start_uri, end_uri = self.id_to_uri[start_id], self.id_to_uri[end_id]
        except KeyError:
            if isinstance(start_id, int):
                return
            start_uri, end_uri = self.get_safe_uri(start_id), self.get_safe_uri(end_id)

        mapping = self.full_mappings.get(rel_type.lower(), {'rel': rel_type, 'relNegated': False,
                                                            'swap': False}) if self.normalise_pos else {'rel': rel_type,
                                                                                                        'relNegated': False,
                                                                                                        'swap': False}
        rel, is_negated, swap = mapping.get('rel'), mapping.get('relNegated', False), mapping.get('swap', False)

        final_source_pos, final_target_pos = (target_pos, source_pos) if swap else (source_pos, target_pos)

        weight = float(weight)
        formatted_weight = str(int(weight)) if weight.is_integer() else str(weight)

        sub_property_list = {
            'rel': rel,
            'is_negated': RDFLiteral(is_negated),
            'weight': RDFLiteral(formatted_weight, datatype=XSD.double)
        }
        if self.mode == 'graph' and self.should_cluster:
            sub_property_list['source_pos'] = RDFLiteral(final_source_pos)
            sub_property_list['target_pos'] = RDFLiteral(final_target_pos)

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
        if base_prop_uri not in self.declared_base_properties:
            self.g.add((base_prop_uri, RDF.type, OWL.AnnotationProperty))
            self.declared_base_properties.add(base_prop_uri)

        if new_annotation_instance:
            self.g.add((rel_uri, RDF.type, base_prop_uri))
            self.g.add((rel_uri, self.ns['is_negated'], RDFLiteral(is_negated)))
            self.g.add((rel_uri, self.ns['weight'], RDFLiteral(formatted_weight, datatype=XSD.double)))

            if self.mode == 'graph' and self.should_cluster:
                if final_source_pos:
                    self.g.add((rel_uri, self.ns['source_pos'], RDFLiteral(final_source_pos)))
                if final_target_pos:
                    self.g.add((rel_uri, self.ns['target_pos'], RDFLiteral(final_target_pos)))

        if start_uri == end_uri:
            # A same-term cross-POS relation (e.g. come_up StativeVerb isA come_up MotionVerb)
            # cannot be written as a self-loop triple. In graph+cluster mode, store a hint so
            # the clusterer can materialise the equivalent cluster-level relation instead,
            # matching the behaviour of DB mode where these are separate concept IDs.
            if self.mode == 'graph' and self.should_cluster \
                    and final_source_pos and final_target_pos \
                    and final_source_pos != final_target_pos:
                self.cross_pos_hints.append((start_uri, rel_uri, final_source_pos, final_target_pos))
            return

        s, t = (end_uri, start_uri) if swap else (start_uri, end_uri)
        self.g.add((s, rel_uri, t))

    @timer(log=False, threaded=False, independent=True, memory=False)
    def add_property_to_graph(self, c_type, c_value, term=None, cid=None):
        concept_uri = self.id_to_uri.get(cid) if cid is not None else (
            self.get_safe_uri(term) if term is not None else None)

        if concept_uri is not None:
            prop_uri = self.get_safe_uri(c_type)
            if isinstance(c_value, str):
                c_value = True if c_value.lower() == "true" else (False if c_value.lower() == "false" else c_value)

            self.g.add((concept_uri, prop_uri, RDFLiteral(c_value)))
            self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))

    @timer(log=False, threaded=False, independent=True, memory=False)
    def add_url_to_graph(self, concept_uri, url, pos):
        self.g.add((concept_uri, self.ns.hasURL, RDFLiteral(f"{url}=={pos}")))

    def add_pos_tag_classes(self, g):
        logging.info("Adding POS tag classes (RDFLib)")
        g.add((self.ns['POSTag'], RDF.type, OWL.Class))

        new_triples = set()
        for pos_tag, mapping in tqdm(self.pos_tag_mappings.items(), desc="Processing POS tagging"):
            pos_uri = self.get_safe_uri(pos_tag)
            g.add((pos_uri, RDFS.subClassOf, self.ns['POSTag']))

            if "classes" not in mapping:
                continue

            for class_name, class_details in mapping["classes"].items():
                class_uri = self.ns[class_name]
                required_property_uris = [(self.get_safe_uri(prop), RDFLiteral(True)) for prop in
                                          class_details.get("properties", [])]

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