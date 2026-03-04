import urllib

import pyoxigraph
from pyoxigraph import NamedNode, Literal, BlankNode, Quad, DefaultGraph

# Pre-define core RDF/OWL/RDFS/XSD NamedNodes
RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
OWL_CLASS = NamedNode("http://www.w3.org/2002/07/owl#Class")
OWL_OBJECTPROPERTY = NamedNode("http://www.w3.org/2002/07/owl#ObjectProperty")
RDFS_COMMENT = NamedNode("http://www.w3.org/2000/01/rdf-schema#comment")
RDFS_LABEL = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
RDFS_SUBCLASSOF = NamedNode("http://www.w3.org/2000/01/rdf-schema#subClassOf")
XSD_STRING = NamedNode("http://www.w3.org/2001/XMLSchema#string")
XSD_BOOLEAN = NamedNode("http://www.w3.org/2001/XMLSchema#boolean")
XSD_INTEGER = NamedNode("http://www.w3.org/2001/XMLSchema#integer")
XSD_DOUBLE = NamedNode("http://www.w3.org/2001/XMLSchema#double")


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


def literal(s: str):
    return Literal(str(s), datatype=XSD_STRING)


def boolean(s: bool):
    return Literal("true" if s else "false", datatype=XSD_BOOLEAN)


def integer(s: int):
    return Literal(str(s), datatype=XSD_INTEGER)


def double(s: float):
    return Literal(str(s), datatype=XSD_DOUBLE)


def onta(ns, s: str):
    return NamedNode(f"{ns.base_uri}{urllib.parse.quote_plus(s)}")


class ParmenidesBuild:
    parmenides_ns = Namespace("https://logds.github.io/parmenides#")

    def create_property(self, name, comment=None):
        if name not in self.relationships:
            d_object = ParmenidesBuild.parmenides_ns[name]
            self.g.add(Quad(d_object, RDF_TYPE, OWL_OBJECTPROPERTY, DefaultGraph()))
            self.relationships[name] = d_object
            if comment is not None:
                self.g.add(Quad(d_object, RDFS_COMMENT, literal(comment), DefaultGraph()))
        return self.relationships[name]

    def create_relationship(self, name, comment=None):
        if name not in self.relationships:
            self.relationships[name] = ParmenidesBuild.parmenides_ns[name]
            if comment is not None:
                self.g.add(Quad(self.relationships[name], RDFS_COMMENT, literal(comment), DefaultGraph()))
        return self.relationships[name]

    def __init__(self, g):
        self.names = dict()
        self.relationships = dict()
        self.g = g
        self.classes = dict()

        self.create_property("hasAdjective")
        self.create_property("subject")
        self.create_property("d_object")
        self.create_property("composite_form_with")
        self.create_property("attachTo")
        self.create_property("argument")
        self.create_property("logicalConstructProperty")
        self.create_property("logicalConstructName")
        self.create_relationship("hasProperty")
        self.create_relationship("formOf")
        self.create_relationship("entryPoint")
        self.create_relationship("partOf")
        self.create_relationship("isA")
        self.create_relationship("relatedTo")
        self.create_relationship("capableOf")
        self.create_relationship("adjectivalForm")
        self.create_relationship("adverbialForm")
        self.create_relationship("eqTo")
        self.create_relationship("neqTo")

    def create_concept(self, full_name, type_uri,
                       hasAdjective=None,
                       entryPoint=None,
                       subject=None,
                       d_object=None,
                       entity_name=None,
                       composite_with=None, comment=None,
                       **kwargs):
        if entity_name is None:
            entity_name = full_name
        ref = self.create_entity(full_name, type_uri, label=entity_name)

        if entryPoint is None:
            entryPoint = ref
        else:
            assert entryPoint in self.names
            entryPoint = self.names[entryPoint]

        self.g.add(Quad(ref, self.relationships["entryPoint"], entryPoint, DefaultGraph()))

        from collections.abc import Iterable
        if (hasAdjective is not None) and (isinstance(hasAdjective, Iterable)):
            assert hasAdjective in self.names
            self.g.add(Quad(ref, self.relationships["hasAdjective"], self.names[hasAdjective], DefaultGraph()))

        if (d_object is not None):
            assert subject is not None

        if composite_with is not None:
            assert isinstance(composite_with, list)
            for composite in composite_with:
                assert composite in self.names
                self.g.add(Quad(ref, self.relationships["composite_form_with"], self.names[composite], DefaultGraph()))

        if subject is not None:
            assert subject in self.names
            self.g.add(Quad(ref, self.relationships["subject"], self.names[subject], DefaultGraph()))
            if d_object is not None:
                self.g.add(Quad(ref, self.relationships["d_object"], self.names[d_object], DefaultGraph()))

        for k, val in kwargs.items():
            if k not in self.relationships:
                d_obj = ParmenidesBuild.parmenides_ns[k]
                self.g.add(Quad(d_obj, RDF_TYPE, OWL_OBJECTPROPERTY, DefaultGraph()))
                self.relationships[k] = d_obj

            if isinstance(val, bool):
                result = boolean(val)
            elif isinstance(val, float):
                result = double(val)
            else:
                result = literal(val)

            self.g.add(Quad(ref, self.relationships[k], result, DefaultGraph()))

        if comment is not None:
            self.g.add(Quad(ref, RDFS_COMMENT, literal(comment), DefaultGraph()))
        return ref

    def create_relationship_instance(self, src: str, rel: str, dst: str, refl=False):
        assert src in self.names
        assert dst in self.names
        rel_uri = self.create_relationship(rel)
        self.g.add(Quad(self.names[src], rel_uri, self.names[dst], DefaultGraph()))
        if refl:
            self.g.add(Quad(self.names[dst], rel_uri, self.names[src], DefaultGraph()))

    def create_entity(self, name: str, clazzL=None, label=None, comment=None, **kwargs):
        if label is None:
            label = name
        if name not in self.names:
            self.names[name] = onta(ParmenidesBuild.parmenides_ns, name)

        if clazzL is not None:
            if isinstance(clazzL, list):
                for clazz in clazzL:
                    assert clazz in self.classes
                    clazz_uri = self.classes[clazz]
                    self.g.add(Quad(self.names[name], RDF_TYPE, clazz_uri, DefaultGraph()))
            elif isinstance(clazzL, str):
                self.g.add(Quad(self.names[name], RDF_TYPE, ParmenidesBuild.parmenides_ns[clazzL], DefaultGraph()))

            self.g.add(Quad(self.names[name], RDFS_LABEL, literal(label), DefaultGraph()))

        self.extract_properties(self.names[name], kwargs)

        if comment is not None:
            self.g.add(Quad(self.names[name], RDFS_COMMENT, literal(comment), DefaultGraph()))

        return self.names[name]

    def extract_properties(self, obj_src, kwargs):
        for k, val in kwargs.items():
            if val is not None:
                if k not in self.relationships:
                    rel = ParmenidesBuild.parmenides_ns[k]
                    self.g.add(Quad(rel, RDF_TYPE, OWL_OBJECTPROPERTY, DefaultGraph()))
                    self.relationships[k] = rel

                if isinstance(val, dict):
                    src_bnode = BlankNode()
                    self.g.add(Quad(obj_src, self.relationships[k], src_bnode, DefaultGraph()))
                    self.extract_properties(src_bnode, val)
                elif isinstance(val, list) or isinstance(val, tuple):
                    for x in val:
                        if isinstance(x, bool):
                            result = boolean(x)
                        elif isinstance(x, float):
                            result = double(x)
                        elif isinstance(x, int):
                            result = integer(x)
                        else:
                            result = literal(x)
                        self.g.add(Quad(obj_src, self.relationships[k], result, DefaultGraph()))
                else:
                    if isinstance(val, bool):
                        result = boolean(val)
                    elif isinstance(val, float):
                        result = double(val)
                    elif isinstance(val, int):
                        result = integer(val)
                    else:
                        result = literal(val)
                    self.g.add(Quad(obj_src, self.relationships[k], result, DefaultGraph()))

    def create_class(self, name, subclazzOf=None, comment=None):
        if name not in self.classes:
            clazz = onta(ParmenidesBuild.parmenides_ns, name)
            self.g.add(Quad(clazz, RDF_TYPE, OWL_CLASS, DefaultGraph()))

            if subclazzOf is not None:
                if isinstance(subclazzOf, str):
                    subclazz_uri = self.create_class(subclazzOf)
                    self.g.add(Quad(clazz, RDFS_SUBCLASSOF, subclazz_uri, DefaultGraph()))
                elif isinstance(subclazzOf, list):
                    for x in subclazzOf:
                        x_uri = self.create_class(x)
                        self.g.add(Quad(clazz, RDFS_SUBCLASSOF, x_uri, DefaultGraph()))

            self.classes[name] = clazz

        if comment is not None:
            self.g.add(Quad(self.classes[name], RDFS_COMMENT, literal(comment), DefaultGraph()))

        return self.classes[name]

    def serialize(self, filename):
        with open(filename, 'wb') as f:
            self.g.dump(f, format=pyoxigraph.RdfFormat.TURTLE, from_graph=pyoxigraph.DefaultGraph())