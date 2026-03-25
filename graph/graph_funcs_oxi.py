import logging
import re
from functools import lru_cache
from urllib.parse import quote
from tqdm import tqdm

from pyoxigraph import Store, NamedNode, Literal as OxiLiteral, Quad, DefaultGraph
from knowledge_bases import ParmenidesLoader
from tools.timer import timer
from graph.graph_funcs import GraphManager

# Pre-define Pyoxigraph constants for maximum parsing speed
OXI_RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
OXI_OWL_CLASS = NamedNode("http://www.w3.org/2002/07/owl#Class")
OXI_RDFS_SUBCLASSOF = NamedNode("http://www.w3.org/2000/01/rdf-schema#subClassOf")
OXI_RDFS_LABEL = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
OXI_XSD_STRING = NamedNode("http://www.w3.org/2001/XMLSchema#string")
OXI_XSD_BOOLEAN = NamedNode("http://www.w3.org/2001/XMLSchema#boolean")
OXI_XSD_DOUBLE = NamedNode("http://www.w3.org/2001/XMLSchema#double")
OXI_XSD_INTEGER = NamedNode("http://www.w3.org/2001/XMLSchema#integer")
OXI_OWL_ANNOTATIONPROPERTY = NamedNode("http://www.w3.org/2002/07/owl#AnnotationProperty")
OXI_OWL_DATATYPEPROPERTY = NamedNode("http://www.w3.org/2002/07/owl#DatatypeProperty")


class OxiNamespace:
    """Wrapper to mimic RDFLib's Namespace dynamic NamedNode generation"""

    def __init__(self, base_uri):
        self.base_uri = base_uri

    def __getitem__(self, key):
        return NamedNode(f"{self.base_uri}{key}")

    def __getattr__(self, name):
        return NamedNode(f"{self.base_uri}{name}")

    def __add__(self, other):
        return NamedNode(f"{self.base_uri}{other}")


def pyoxi_literal(val):
    """Mimics RDFLib's strict native datatype mapping logic"""
    if isinstance(val, bool):
        return OxiLiteral("true" if val else "false", datatype=OXI_XSD_BOOLEAN)
    elif isinstance(val, int):
        return OxiLiteral(str(val), datatype=OXI_XSD_INTEGER)
    elif isinstance(val, float):
        return OxiLiteral(str(val), datatype=OXI_XSD_DOUBLE)
    return OxiLiteral(str(val))


class OxiGraphManager(GraphManager):
    def __init__(self, builder, config):
        super().__init__(builder, config)
        self.ns = OxiNamespace(self.config['turtle_export']['base_uri'])
        # Stores (uri, rel_uri, source_pos, target_pos) for same-term cross-POS relations
        # that cannot be written as self-loop quads. Used by the clusterer to recover
        # the equivalent cluster-level relations (e.g. come_up StativeVerb isA come_up MotionVerb).
        self.cross_pos_hints = []
        self.g = self.init_graph()

    def init_graph(self):
        g = Store()
        if 'parmenides' in self.config['general']['sources']:
            ParmenidesLoader.add_logical_functions(self.config, g)
        self.create_classes(g)
        return g

    def _declare_property(self, prop_uri, owl_type):
        if prop_uri.value not in self.declared_base_properties:
            self.g.add(Quad(prop_uri, OXI_RDF_TYPE, owl_type, DefaultGraph()))
            self.declared_base_properties.add(prop_uri.value)

    @lru_cache(maxsize=1024)
    def get_safe_uri(self, term):
        return NamedNode(self.ns.base_uri + quote(term)) if re.search(r'[^a-zA-Z0-9_-]', term) else self.ns[term]

    def create_classes(self, g):
        logging.info("Creating ontology classes (Oxigraph)")
        for parent, children in self.ontology_classes.items():
            children = children['classes']
            parent_uri = self.ns[parent]
            g.add(Quad(parent_uri, OXI_RDF_TYPE, OXI_OWL_CLASS, DefaultGraph()))
            self.create_sub_classes(parent, children, g, parent_uri)

    def create_sub_classes(self, parent, children, g, parent_uri):
        ignored_keys = ['classes', 'sameAs']
        for child in children:
            child_uri = self.ns[child]
            g.add(Quad(child_uri, OXI_RDF_TYPE, OXI_OWL_CLASS, DefaultGraph()))
            g.add(Quad(child_uri, OXI_RDFS_SUBCLASSOF, parent_uri, DefaultGraph()))
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
            self.g.add(Quad(uri, OXI_RDF_TYPE, self.ns[i_pos], DefaultGraph()))

        self.g.add(Quad(uri, OXI_RDFS_LABEL, OxiLiteral(str(term), datatype=OXI_XSD_STRING), DefaultGraph()))

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_relation_to_graph(self, start_id, end_id, rel_type, weight, source_pos=None, target_pos=None):
        try:
            start_uri, end_uri = self.id_to_uri[start_id], self.id_to_uri[end_id]
        except KeyError:
            if isinstance(start_id, int):
                return
            start_uri, end_uri = self.get_safe_uri(start_id), self.get_safe_uri(end_id)

        mapping = self.full_mappings.get(
            rel_type.lower(),
            {
                'rel': rel_type,
                'relNegated': False,
                'swap': False
            }
        ) if self.normalise_pos else \
            {
                'rel': rel_type,
                'relNegated': False,
                'swap': False
            }

        rel, is_negated, swap = mapping.get('rel'), mapping.get('relNegated', False), mapping.get('swap', False)

        final_source_pos, final_target_pos = (target_pos, source_pos) if swap else (source_pos, target_pos)
        weight = float(weight)

        sub_property_list = {'rel': rel, 'is_negated': is_negated, 'weight': weight}
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
        self._declare_property(base_prop_uri, OXI_OWL_ANNOTATIONPROPERTY)

        if new_annotation_instance:
            self.g.add(Quad(rel_uri, OXI_RDF_TYPE, base_prop_uri, DefaultGraph()))
            self.g.add(Quad(rel_uri, self.ns['is_negated'], pyoxi_literal(is_negated), DefaultGraph()))
            self.g.add(Quad(rel_uri, self.ns['weight'], pyoxi_literal(weight), DefaultGraph()))

            if self.mode == 'graph' and self.should_cluster:
                if final_source_pos:
                    self.g.add(Quad(rel_uri, self.ns['source_pos'], pyoxi_literal(final_source_pos), DefaultGraph()))
                if final_target_pos:
                    self.g.add(Quad(rel_uri, self.ns['target_pos'], pyoxi_literal(final_target_pos), DefaultGraph()))

        if start_uri == end_uri:
            # A same-term cross-POS relation (e.g. come_up StativeVerb isA come_up MotionVerb)
            # cannot be written as a self-loop quad. In graph+cluster mode, store a hint so
            # the clusterer can materialise the equivalent cluster-level relation instead,
            # matching the behaviour of DB mode where these are separate concept IDs.
            if self.mode == 'graph' and self.should_cluster \
                    and final_source_pos and final_target_pos \
                    and final_source_pos != final_target_pos:
                self.cross_pos_hints.append((start_uri, rel_uri, final_source_pos, final_target_pos))
            return

        s, t = (end_uri, start_uri) if swap else (start_uri, end_uri)
        self.g.add(Quad(s, rel_uri, t, DefaultGraph()))

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_property_to_graph(self, c_type, c_value, term=None, cid=None):
        concept_uri = self.id_to_uri.get(cid) if cid is not None else (
            self.get_safe_uri(term) if term is not None else None)

        if concept_uri is not None:
            prop_uri = self.get_safe_uri(c_type)
            if isinstance(c_value, str):
                c_value = True if c_value.lower() == "true" else (False if c_value.lower() == "false" else c_value)

            self.g.add(Quad(concept_uri, prop_uri, pyoxi_literal(c_value), DefaultGraph()))
            self._declare_property(prop_uri, OXI_OWL_DATATYPEPROPERTY)

    @timer(log=False, threaded=False, independent=False, memory=False)
    def add_url_to_graph(self, concept_uri, url, pos):
        self.g.add(Quad(concept_uri, self.ns.hasURL, pyoxi_literal(f"{url}=={pos}"), DefaultGraph()))

    def add_pos_tag_classes(self, g):
        logging.info("Adding POS tag classes (Oxigraph)")
        g.add(Quad(self.ns['POSTag'], OXI_RDF_TYPE, OXI_OWL_CLASS, DefaultGraph()))

        new_triples = set()
        for pos_tag, mapping in tqdm(self.pos_tag_mappings.items(), desc="Processing POS tagging"):
            pos_uri = self.get_safe_uri(pos_tag)
            g.add(Quad(pos_uri, OXI_RDFS_SUBCLASSOF, self.ns['POSTag'], DefaultGraph()))

            if "classes" not in mapping:
                continue

            for class_name, class_details in mapping["classes"].items():
                class_uri = self.ns[class_name]
                required_properties = [self.get_safe_uri(prop) for prop in class_details.get("properties", [])]

                candidate_subjects = [quad.subject for quad in g.quads_for_pattern(None, OXI_RDF_TYPE, class_uri)]
                for subject_uri in candidate_subjects:
                    has_all_properties = True
                    for prop_uri in required_properties:
                        match_found = False
                        for quad in g.quads_for_pattern(subject_uri, prop_uri, None):
                            if isinstance(quad.object, OxiLiteral) and quad.object.value.lower() in ('true', '1'):
                                match_found = True
                                break
                        if not match_found:
                            has_all_properties = False
                            break

                    if has_all_properties:
                        new_triples.add((subject_uri, OXI_RDF_TYPE, pos_uri))

        logging.info(f"Identified {len(new_triples)} new POS tag classifications to add.")
        for s, p, o in new_triples:
            if next(g.quads_for_pattern(s, p, o), None) is None:
                g.add(Quad(s, p, o, DefaultGraph()))