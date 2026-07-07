"""Fix C: reflexive self-loop filter.

Builds a tiny Oxigraph store directly and exercises the graph manager's helpers
and the final `remove_structural_self_loops` pass, which is the mode-agnostic
correctness guarantee that keeps DB and graph output triple-equivalent.
"""

from pyoxigraph import Store, NamedNode, Literal, Quad, DefaultGraph

from graph.graph_funcs_oxi import (
    OxiGraphManager,
    OxiNamespace,
    OXI_RDF_TYPE,
    OXI_RDFS_LABEL,
    OXI_XSD_STRING,
)

BASE = "http://example.org/honk#"


def _mgr():
    m = OxiGraphManager.__new__(OxiGraphManager)  # skip heavy __init__
    m.g = Store()
    m.ns = OxiNamespace(BASE)
    m._label_index = None
    m._rel_base_cache = {}
    return m


def _label(m, uri, lab):
    m.g.add(Quad(uri, OXI_RDFS_LABEL, Literal(lab, datatype=OXI_XSD_STRING), DefaultGraph()))


def _rel_instance(m, inst, base):
    m.g.add(Quad(m.ns[inst], OXI_RDF_TYPE, m.ns[base], DefaultGraph()))
    return m.ns[inst]


def test_same_label_and_rel_base_helpers():
    m = _mgr()
    a, b, c = m.ns["alabama--n-location"], m.ns["alabama--n-object"], m.ns["bob_up"]
    _label(m, a, "Alabama")
    _label(m, b, "alabama")  # case-insensitive match
    _label(m, c, "bob_up")
    partOf1 = _rel_instance(m, "partOf1", "partOf")
    assert m._same_label(a.value, b.value) is True
    assert m._same_label(a.value, c.value) is False
    assert m.rel_base_of(partOf1) == "partOf"
    assert m.STRUCTURAL_IRREFLEXIVE_RELS == frozenset({"partOf", "isA", "instanceOf"})


def test_removes_same_label_structural_loop_only():
    m = _mgr()
    N = m.ns
    state, river = N["alabama--n-location"], N["alabama--n-object"]
    bob, come = N["bob_up"], N["come_up"]
    mag_a, mag_n = N["magenta--a"], N["magenta--n"]
    _label(m, state, "alabama"); _label(m, river, "alabama")
    _label(m, bob, "bob_up"); _label(m, come, "come_up")
    _label(m, mag_a, "magenta"); _label(m, mag_n, "magenta")

    partOf1 = _rel_instance(m, "partOf1", "partOf")
    isA1 = _rel_instance(m, "isA1", "isA")
    rel1 = _rel_instance(m, "relatedTo1", "relatedTo")

    m.g.add(Quad(state, partOf1, river, DefaultGraph()))  # same label + structural -> DROP
    m.g.add(Quad(bob, isA1, come, DefaultGraph()))        # diff label + structural -> KEEP (bob_up isA come_up)
    m.g.add(Quad(mag_a, rel1, mag_n, DefaultGraph()))     # same label + non-structural -> KEEP

    removed = m.remove_structural_self_loops()

    assert removed == 1
    assert next(m.g.quads_for_pattern(state, partOf1, river), None) is None
    assert next(m.g.quads_for_pattern(bob, isA1, come), None) is not None
    assert next(m.g.quads_for_pattern(mag_a, rel1, mag_n), None) is not None
