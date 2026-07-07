"""Fix A (Stage 0): the ConceptNet Relation parser retains the sense suffix.

Distinct senses of a lemma carry a POS + a WordNet/Wikipedia disambiguation
suffix (e.g. magenta the colour = /a/wn vs the Battle of Magenta = /n/wn/act).
The suffix used to be discarded; it is now captured for sense-aware identity.
"""

import json

from knowledge_bases.conceptnet_loader import Relation


def _rel(start, end, rel="/r/IsA", data=None):
    data = data or {"weight": 1.0}
    return Relation(["/a/x", rel, start, end, json.dumps(data)])


def test_full_sense_suffix_captured():
    r = _rel("/c/en/magenta/n/wn/act", "/c/en/colour/n")
    assert r.surfaceStart == "magenta"
    assert r.startPOS == "n"
    assert r.startSubSense == "wn/act"
    assert r.endSubSense is None  # colour/n is POS-only


def test_adjective_sense_suffix():
    r = _rel("/c/en/magenta/a/wn", "/c/en/red/a")
    assert r.startPOS == "a"
    assert r.startSubSense == "wn"


def test_dbpedia_sense_suffix():
    # The loader converts '_' to ' ' across the URI; the suffix stays a distinct
    # discriminator (normalised again when the sense URI is built).
    r = _rel("/c/en/magenta/n/wp/paris_rer", "/c/en/station/n")
    assert r.startPOS == "n"
    assert r.startSubSense == "wp/paris rer"


def test_bare_lemma_has_no_subsense():
    r = _rel("/c/en/magenta", "/c/en/red")
    assert r.surfaceStart == "magenta"
    assert r.startSubSense is None


def test_pos_only_has_no_subsense():
    r = _rel("/c/en/dog/n", "/c/en/animal/n")
    assert r.startPOS == "n"
    assert r.startSubSense is None


def test_external_url_keeps_start_sense():
    r = _rel(
        "/c/en/magenta/n/wn/act",
        "http://wordnet-rdf.princeton.edu/wn31/101288277-n",
        rel="/r/ExternalURL",
    )
    assert r.startPOS == "n"
    assert r.startSubSense == "wn/act"
