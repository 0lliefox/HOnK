"""Unit tests for the method-vs-data ablation normalisation modes (item 1).

Verifies that `general.normalisation` ('full' | 'raw' | 'canonicalise') drives
``AbstractLoader.normalise_data`` correctly:

  * full          -> HOnK POS mapping applied (e.g. noun.act -> AgentNoun)
  * raw           -> source labels kept (basic hygiene only, no mappings)
  * canonicalise  -> lowercase + WordNet lemmatisation, no HOnK mappings

These run without a database or a real graph store by faking the graph manager
and builder (the only collaborators ``normalise_data`` touches).
"""

import pytest

from knowledge_bases.abstract_loader import AbstractLoader


class _FakeBenchmark:
    def add_row(self, *args, **kwargs):
        pass


class _FakeBuilder:
    def __init__(self):
        self.benchmarking = _FakeBenchmark()
        self.memory_benchmarking = _FakeBenchmark()
        self.run_id = 0
        self.should_cache = False


class _FakeGraphManager:
    def __init__(self, mode):
        self.normalisation_mode = mode
        self.normalise_pos = (mode == "full")
        # Mapping keys are lowercased by load_mappings in the real code.
        self.full_mappings = {
            "synonym": {"rel": "eq", "relNegated": False},
            "noun.act": "AgentNoun",
        }


class _Loader(AbstractLoader):
    """Concrete loader that bypasses the heavy AbstractLoader.__init__."""

    def __init__(self, mode):
        self.builder = _FakeBuilder()
        self.config = {}
        self.graph_manager = _FakeGraphManager(mode)
        self._timer_stack = []

    def parse_data(self):  # abstract method, unused here
        pass

    def store_data(self, data):  # abstract method, unused here
        pass


def test_full_mode_applies_pos_mapping():
    loader = _Loader("full")
    term, pos = loader.normalise_data("running_dog", "noun.act")
    assert term == "running dog"   # underscore hygiene
    assert pos == "AgentNoun"      # POS mapped to the HOnK class


def test_raw_mode_keeps_source_labels():
    loader = _Loader("raw")
    term, pos = loader.normalise_data("running_dog", "noun.act")
    assert term == "running dog"   # basic hygiene only
    assert pos == "noun.act"       # NOT mapped -> isolates "more data" from "the method"


def test_canonicalise_mode_lowercases():
    loader = _Loader("canonicalise")
    term, pos = loader.normalise_data("Running", "verb")
    assert term == term.lower()    # always lowercased
    assert pos == "verb"           # POS left untouched (no HOnK mapping)


def test_canonicalise_mode_lemmatises_when_wordnet_available():
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            nltk.data.find("corpora/wordnet")
        except Exception:
            pytest.skip("WordNet corpus unavailable offline")
    loader = _Loader("canonicalise")
    term, _ = loader.normalise_data("Cats", "noun")
    assert term == "cat"           # plural lemmatised to singular
