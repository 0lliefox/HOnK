"""Fix B: type-coherence merge guard — supersense derivation.

The guard partitions a URL-connected cluster by WordNet supersense so a URL
collision cannot merge disjoint senses (a colour with a battle). The derivation
must map WordNet lexical_domains and ConceptNet /wn/ subsenses to the *same*
comparable key, and return None for senses with no clean WordNet supersense
(so they are never falsely split).
"""

from graph.graph_funcs import GraphManager

ss = GraphManager.supersense_of_sense


def test_wordnet_lexical_domain_passthrough():
    assert ss("noun.location") == "noun.location"
    assert ss("verb.motion") == "verb.motion"
    assert ss("adj.all") == "adj.all"


def test_conceptnet_wn_subsense_normalised():
    assert ss("n/wn/act") == "noun.act"
    assert ss("v/wn/motion") == "verb.motion"


def test_cross_source_same_supersense_matches():
    # A WordNet concept and a ConceptNet concept of the same domain must agree,
    # so the guard does NOT split legitimately co-clustered cross-source senses.
    assert ss("n/wn/location") == ss("noun.location") == "noun.location"


def test_disjoint_supersenses_differ():
    assert ss("n/wn/act") != ss("n/wn/communication")   # battle vs song
    assert ss("noun.location") != ss("noun.person")      # state vs person


def test_ambiguous_senses_are_unguarded():
    # No clean WordNet supersense -> None -> never split (avoids false positives).
    assert ss("a/wn") is None            # adjective sense without a domain
    assert ss("n/wp/paris rer") is None  # DBpedia sense
    assert ss(None) is None
    assert ss("") is None
