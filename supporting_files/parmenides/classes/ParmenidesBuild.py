import urllib.parse
from collections.abc import Iterable

from pyoxigraph import NamedNode as OxiNamedNode, Literal as OxiLiteral, Quad, DefaultGraph, BlankNode as OxiBlankNode
from rdflib import Namespace as RDFNamespace, URIRef as RDFURIRef, RDF, OWL, RDFS, Literal as RDFLiteral, Graph, XSD, \
    BNode as RDFBNode

OXI_RDF_TYPE = OxiNamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
OXI_OWL_CLASS = OxiNamedNode("http://www.w3.org/2002/07/owl#Class")
OXI_OWL_OBJECTPROPERTY = OxiNamedNode("http://www.w3.org/2002/07/owl#ObjectProperty")
OXI_RDFS_SUBCLASSOF = OxiNamedNode("http://www.w3.org/2000/01/rdf-schema#subClassOf")
OXI_RDFS_LABEL = OxiNamedNode("http://www.w3.org/2000/01/rdf-schema#label")
OXI_RDFS_COMMENT = OxiNamedNode("http://www.w3.org/2000/01/rdf-schema#comment")
OXI_XSD_STRING = OxiNamedNode("http://www.w3.org/2001/XMLSchema#string")
OXI_XSD_BOOLEAN = OxiNamedNode("http://www.w3.org/2001/XMLSchema#boolean")
OXI_XSD_DOUBLE = OxiNamedNode("http://www.w3.org/2001/XMLSchema#double")
OXI_XSD_INTEGER = OxiNamedNode("http://www.w3.org/2001/XMLSchema#integer")


class OxiNamespace:
    def __init__(self, base_uri):
        self.base_uri = base_uri

    def __getitem__(self, key):
        return OxiNamedNode(f"{self.base_uri}{key}")

    def __getattr__(self, name):
        return OxiNamedNode(f"{self.base_uri}{name}")

    def __add__(self, other):
        return OxiNamedNode(f"{self.base_uri}{other}")


class ParmenidesBuild:
    honk_uri = "https://ofox.co.uk/honk#"

    def __init__(self, g):
        self.g = g

        # Automatically infer the graph type based on the object passed
        if type(g).__name__ == 'Store' or hasattr(g, 'quads_for_pattern'):
            self.graph_type = 'oxigraph'
            self.ns = OxiNamespace(self.honk_uri)
        else:
            self.graph_type = 'rdf'
            self.ns = RDFNamespace(self.honk_uri)
            self.g.bind("honk", self.ns)
            self.g.bind("rdfs", RDF)

        self.names = dict()
        self.relationships = dict()
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

    def _literal(self, s: str):
        if self.graph_type == 'oxigraph':
            return OxiLiteral(str(s), datatype=OXI_XSD_STRING)
        return RDFLiteral(s, datatype=XSD.string)

    def _boolean(self, s: bool):
        if self.graph_type == 'oxigraph':
            return OxiLiteral("true" if s else "false", datatype=OXI_XSD_BOOLEAN)
        return RDFLiteral(s, datatype=XSD.boolean)

    def _integer(self, s):
        if self.graph_type == 'oxigraph':
            return OxiLiteral(str(s), datatype=OXI_XSD_INTEGER)
        return RDFLiteral(s, datatype=XSD.integer)

    def _double(self, s: float):
        if self.graph_type == 'oxigraph':
            return OxiLiteral(str(s), datatype=OXI_XSD_DOUBLE)
        return RDFLiteral(s, datatype=XSD.double)

    def _onta(self, s: str):
        if self.graph_type == 'oxigraph':
            return self.ns[urllib.parse.quote_plus(s)]
        return RDFURIRef(self.ns[urllib.parse.quote_plus(s)])

    def _add_triple(self, s, p, o):
        """Standardizes triple insertion logic safely"""
        if self.graph_type == 'oxigraph':
            self.g.add(Quad(s, p, o, DefaultGraph()))
        else:
            self.g.add((s, p, o))

    def create_property(self, name, comment=None):
        if name not in self.relationships:
            if self.graph_type == 'oxigraph':
                d_object = self.ns[name]
                self._add_triple(d_object, OXI_RDF_TYPE, OXI_OWL_OBJECTPROPERTY)
            else:
                d_object = RDFURIRef(self.ns[name])
                self._add_triple(d_object, RDF.type, OWL.ObjectProperty)

            self.relationships[name] = d_object
            if comment is not None:
                if self.graph_type == 'oxigraph':
                    self._add_triple(d_object, OXI_RDFS_COMMENT, OxiLiteral(str(comment)))
                else:
                    self._add_triple(d_object, RDFS.comment, RDFLiteral(comment))
        return self.relationships[name]

    def create_relationship(self, name, comment=None):
        if name not in self.relationships:
            if self.graph_type == 'oxigraph':
                self.relationships[name] = self.ns[name]
                if comment is not None:
                    self._add_triple(self.relationships[name], OXI_RDFS_COMMENT, OxiLiteral(str(comment)))
            else:
                self.relationships[name] = RDFURIRef(self.ns[name])
                if comment is not None:
                    self._add_triple(self.relationships[name], RDFS.comment, RDFLiteral(comment))
        return self.relationships[name]

    def create_concept(self, full_name, type_val,
                       hasAdjective=None,
                       entryPoint=None,
                       subject=None,
                       d_object=None,
                       entity_name=None,
                       composite_with=None, comment=None,
                       **kwargs):
        if entity_name is None:
            entity_name = full_name
        ref = self.create_entity(full_name, type_val, label=entity_name)

        if entryPoint is None:
            entryPoint = ref
        else:
            assert entryPoint in self.names
            entryPoint = self.names[entryPoint]

        self._add_triple(ref, self.relationships["entryPoint"], entryPoint)

        if (hasAdjective is not None) and (isinstance(hasAdjective, Iterable)):
            assert hasAdjective in self.names
            self._add_triple(ref, self.relationships["hasAdjective"], self.names[hasAdjective])

        if d_object is not None:
            assert subject is not None

        if composite_with is not None:
            assert isinstance(composite_with, list)
            for composite in composite_with:
                assert composite in self.names
                self._add_triple(ref, self.relationships["composite_form_with"], self.names[composite])

        if subject is not None:
            assert subject in self.names
            self._add_triple(ref, self.relationships["subject"], self.names[subject])
            if d_object is not None:
                self._add_triple(ref, self.relationships["d_object"], self.names[d_object])

        for k, val in kwargs.items():
            if k not in self.relationships:
                if self.graph_type == 'oxigraph':
                    d_obj = self.ns[k]
                    self._add_triple(d_obj, OXI_RDF_TYPE, OXI_OWL_OBJECTPROPERTY)
                else:
                    d_obj = RDFURIRef(self.ns[k])
                    self._add_triple(d_obj, RDF.type, OWL.ObjectProperty)
                self.relationships[k] = d_obj

            if isinstance(val, bool):
                result = self._boolean(val)
            elif isinstance(val, str):
                result = self._literal(val)
            elif isinstance(val, float):
                result = self._double(val)
            else:
                result = self._literal(str(val))

            self._add_triple(ref, self.relationships[k], result)

        if comment is not None:
            if self.graph_type == 'oxigraph':
                self._add_triple(ref, OXI_RDFS_COMMENT, OxiLiteral(str(comment)))
            else:
                self._add_triple(ref, RDFS.comment, RDFLiteral(comment))

        return ref

    def create_relationship_instance(self, src: str, rel: str, dst: str, refl=False):
        assert src in self.names
        assert dst in self.names
        rel_node = self.create_relationship(rel)
        self._add_triple(self.names[src], rel_node, self.names[dst])
        if refl:
            self._add_triple(self.names[dst], rel_node, self.names[src])

    def create_entity(self, name: str, clazzL=None, label=None, comment=None,
                      **kwargs):
        if label is None:
            label = name
        if name not in self.names:
            self.names[name] = self._onta(name)

        if clazzL is not None:
            if isinstance(clazzL, list):
                for clazz in clazzL:
                    assert clazz in self.classes
                    clazz_node = self.classes[clazz]
                    if self.graph_type == 'oxigraph':
                        self._add_triple(self.names[name], OXI_RDF_TYPE, clazz_node)
                    else:
                        self._add_triple(self.names[name], RDF.type, clazz_node)
            elif isinstance(clazzL, str):
                if self.graph_type == 'oxigraph':
                    self._add_triple(self.names[name], OXI_RDF_TYPE, self.ns[clazzL])
                else:
                    self._add_triple(self.names[name], RDF.type, self.ns[clazzL])

            if self.graph_type == 'oxigraph':
                self._add_triple(self.names[name], OXI_RDFS_LABEL, self._literal(label))
            else:
                self._add_triple(self.names[name], RDFS.label, self._literal(label))

        self.extract_properties(self.names[name], kwargs)

        if comment is not None:
            if self.graph_type == 'oxigraph':
                self._add_triple(self.names[name], OXI_RDFS_COMMENT, OxiLiteral(str(comment)))
            else:
                self._add_triple(self.names[name], RDFS.comment, RDFLiteral(comment))

        return self.names[name]

    def extract_properties(self, obj_src, kwargs):
        for k, val in kwargs.items():
            if val is not None:
                if k not in self.relationships:
                    if self.graph_type == 'oxigraph':
                        rel = self.ns[k]
                        self._add_triple(rel, OXI_RDF_TYPE, OXI_OWL_OBJECTPROPERTY)
                    else:
                        rel = RDFURIRef(self.ns[k])
                        self._add_triple(rel, RDF.type, OWL.ObjectProperty)
                    self.relationships[k] = rel

                if isinstance(val, dict):
                    src_bnode = OxiBlankNode() if self.graph_type == 'oxigraph' else RDFBNode()
                    self._add_triple(obj_src, self.relationships[k], src_bnode)
                    self.extract_properties(src_bnode, val)
                elif isinstance(val, (list, tuple)):
                    for x in val:
                        if isinstance(x, bool):
                            result = self._boolean(x)
                        elif isinstance(x, str):
                            result = self._literal(x)
                        elif isinstance(x, int):
                            result = self._integer(x)
                        elif isinstance(x, float):
                            result = self._double(x)
                        else:
                            result = self._literal(str(x))
                        self._add_triple(obj_src, self.relationships[k], result)
                else:
                    if isinstance(val, bool):
                        result = self._boolean(val)
                    elif isinstance(val, str):
                        result = self._literal(val)
                    elif isinstance(val, int):
                        result = self._integer(val)
                    elif isinstance(val, float):
                        result = self._double(val)
                    else:
                        result = self._literal(str(val))
                    self._add_triple(obj_src, self.relationships[k], result)

    def create_class(self, name, subclazzOf=None, comment=None):
        if name not in self.classes:
            clazz = self._onta(name)

            if self.graph_type == 'oxigraph':
                self._add_triple(clazz, OXI_RDF_TYPE, OXI_OWL_CLASS)
            else:
                self._add_triple(clazz, RDF.type, OWL.Class)

            if subclazzOf is not None:
                if isinstance(subclazzOf, str):
                    subclazzOf_node = self.create_class(subclazzOf)
                    if self.graph_type == 'oxigraph':
                        self._add_triple(clazz, OXI_RDFS_SUBCLASSOF, subclazzOf_node)
                    else:
                        self._add_triple(clazz, RDFS.subClassOf, subclazzOf_node)
                elif isinstance(subclazzOf, list):
                    for x in subclazzOf:
                        x_node = self.create_class(x)
                        if self.graph_type == 'oxigraph':
                            self._add_triple(clazz, OXI_RDFS_SUBCLASSOF, x_node)
                        else:
                            self._add_triple(clazz, RDFS.subClassOf, x_node)
            self.classes[name] = clazz

        if comment is not None:
            if self.graph_type == 'oxigraph':
                self._add_triple(self.classes[name], OXI_RDFS_COMMENT, OxiLiteral(str(comment)))
            else:
                self._add_triple(self.classes[name], RDFS.comment, RDFLiteral(comment))

        return self.classes[name]

    def serialize(self, filename):
        if self.graph_type == 'oxigraph':
            import pyoxigraph
            with open(filename, 'wb') as f:
                self.g.dump(
                    f, 
                    pyoxigraph.RdfFormat.TURTLE, 
                    from_graph=pyoxigraph.DefaultGraph(),
                    prefixes={
                        "honk": self.honk_uri,
                        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                        "owl": "http://www.w3.org/2002/07/owl#"
                    }
                )
        else:
            self.g.serialize(destination=filename)