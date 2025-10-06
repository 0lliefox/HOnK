from knowledge_bases.abstract_loader import AbstractLoader
from knowledge_bases.parmenides_loader import ParmenidesLoader
from knowledge_bases.wordnet_loader import WordNetLoader
from knowledge_bases.conceptnet_loader import ConceptNetLoader
from knowledge_bases.wiktionary_loader import WiktionaryLoader

__all__ = [
    "AbstractLoader",
    "WordNetLoader",
    "ConceptNetLoader",
    "WiktionaryLoader",
    "ParmenidesLoader"
]
