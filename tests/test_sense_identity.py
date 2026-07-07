"""Fix A: sense-aware concept identity in the graph backend.

Distinct ConceptNet senses of a lemma (magenta the colour = /a/wn vs the Battle
of Magenta = /n/wn/act) become distinct sense-scoped URIs whose external URLs no
longer collapse onto one node, while a bare-lemma hub keeps the shared label and
is reachable via `withSense`. Uses a bare Oxigraph manager with a fake builder.
"""

from pyoxigraph import Store

from graph.graph_funcs_oxi import (
    OxiGraphManager,
    OxiNamespace,
    OXI_RDF_TYPE,
    OXI_RDFS_LABEL,
)

BASE = "http://example.org/honk#"


class _Bench:
    def add_row(self, *a, **k):
        pass


class _Builder:
    benchmarking = _Bench()
    memory_benchmarking = _Bench()
    run_id = 0


def _mgr():
    m = OxiGraphManager.__new__(OxiGraphManager)
    m.builder = _Builder()
    m.g = Store()
    m.ns = OxiNamespace(BASE)
    m.rejected_classes = {}
    m.equivalent_classes = {}
    m.id_to_uri = {}
    m._label_index = None
    m._rel_base_cache = {}
    m.node_supersense = {}
    return m


def test_sense_uris_are_distinct_with_hub_linkage():
    m = _mgr()
    m.add_concept_to_graph("magenta", "Adjective", sense="a/wn")   # colour
    m.add_concept_to_graph("magenta", "Noun", sense="n/wn/act")    # battle

    colour = m.get_safe_uri("magenta", "a/wn")
    battle = m.get_safe_uri("magenta", "n/wn/act")
    hub = m.get_safe_uri("magenta")

    assert colour.value == BASE + "magenta--a-wn"
    assert battle.value == BASE + "magenta--n-wn-act"
    assert hub.value == BASE + "magenta"
    assert colour.value != battle.value

    # each sense -> hub via withSense (navigational, not a clustering edge)
    assert next(m.g.quads_for_pattern(colour, m.ns.withSense, hub), None) is not None
    assert next(m.g.quads_for_pattern(battle, m.ns.withSense, hub), None) is not None

    # every node carries the bare surface label (LaSSI keys off this)
    for u in (colour, battle, hub):
        labels = [q.object.value for q in m.g.quads_for_pattern(u, OXI_RDFS_LABEL, None)]
        assert labels == ["magenta"]

    # rdf:type reflects the POS on the sense nodes
    assert next(m.g.quads_for_pattern(colour, OXI_RDF_TYPE, m.ns["Adjective"]), None) is not None
    assert next(m.g.quads_for_pattern(battle, OXI_RDF_TYPE, m.ns["Noun"]), None) is not None


def test_url_scopes_to_sense_node_not_hub():
    m = _mgr()
    colour = m.get_safe_uri("magenta", "a/wn")
    m.add_url_to_graph(colour, "http://wordnet-rdf.princeton.edu/wn31/300378586-s", "Adjective")

    hub = m.get_safe_uri("magenta")
    assert next(m.g.quads_for_pattern(colour, m.ns.hasURL, None), None) is not None
    assert next(m.g.quads_for_pattern(hub, m.ns.hasURL, None), None) is None


def test_bare_lemma_is_hub_without_sense():
    m = _mgr()
    m.add_concept_to_graph("magenta", "Concept")  # bare ConceptNet edge, sense=None
    hub = m.get_safe_uri("magenta")
    assert hub.value == BASE + "magenta"
    assert next(m.g.quads_for_pattern(hub, m.ns.withSense, None), None) is None
