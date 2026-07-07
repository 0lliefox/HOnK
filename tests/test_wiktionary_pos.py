"""Fix D: Wiktionary relation-target POS resolution.

The loader used to force the source word's POS onto every relation target
(except related/derived), producing wrong-POS edges/phantom nodes. These pin
down the corrected behaviour without a DB or graph store.
"""

from knowledge_bases.wiktionary_loader import WiktionaryLoader


def _loader():
    # resolve_target_pos uses no instance state; skip the heavy __init__.
    return WiktionaryLoader.__new__(WiktionaryLoader)


def test_synonym_uses_target_pos_when_source_pos_absent():
    # nonsense(Noun) --synonym--> unreasoning, which Wiktionary lists only as adj.
    w2p = {"unreasoning": {"Adjective"}}
    assert _loader().resolve_target_pos("unreasoning", "Noun", "synonyms", w2p) == ["Adjective"]


def test_synonym_is_pos_preserving_when_target_has_source_pos():
    # liberate(Verb) --synonym--> free (adj/adv/verb/noun): keep just Verb, no fan-out.
    w2p = {"free": {"Adjective", "Adverb", "Verb", "Noun"}}
    assert _loader().resolve_target_pos("free", "Verb", "synonyms", w2p) == ["Verb"]


def test_synonym_unknown_target_falls_back_to_source_pos():
    assert _loader().resolve_target_pos("zzzunknown", "Noun", "synonyms", {}) == ["Noun"]


def test_derived_uses_target_pos_cross_pos():
    # happy(Adjective) --derived--> happiness(Noun): derivation is cross-POS.
    w2p = {"happiness": {"Noun"}}
    assert _loader().resolve_target_pos("happiness", "Adjective", "derived", w2p) == ["Noun"]


def test_derived_unknown_target_is_dropped():
    assert _loader().resolve_target_pos("zzzunknown", "Adjective", "derived", {}) == []
