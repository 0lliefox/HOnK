import itertools
import logging
from typing import List, Set, Tuple

import pyoxigraph

try:
    from .store_loader import shorten
except ImportError:
    from store_loader import shorten  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def extract_subgraph(
    store: pyoxigraph.Store,
    keywords: List[str],
) -> Set[Tuple[str, str, str]]:
    if not keywords:
        return set()

    branches = []
    for kw in keywords:
        kw_norm = kw.lower().replace(' ', '_')
        kw_space = kw.lower()
        f = (
            f'CONTAINS(LCASE(STR(?s)), "{kw_norm}") || CONTAINS(LCASE(STR(?o)), "{kw_norm}")'
            f' || CONTAINS(LCASE(STR(?s)), "{kw_space}") || CONTAINS(LCASE(STR(?o)), "{kw_space}")'
        )
        branches.append(f'{{ ?s ?p ?o . FILTER( {f} ) }}')

    query = "SELECT ?s ?p ?o WHERE { " + " UNION ".join(branches) + " }"
    triples: Set[Tuple[str, str, str]] = set()
    try:
        for r in store.query(query):
            triples.add((
                shorten(r['s'].value),
                shorten(r['p'].value),
                shorten(r['o'].value),
            ))
    except Exception as e:
        logger.error("SPARQL subgraph query failed: %s", e)

    return triples


def extract_bridging_triples(
    store: pyoxigraph.Store,
    keywords: List[str],
    bridging_limit: int = 100,  # kept for API compatibility; no longer applied in SPARQL
) -> List[Tuple[str, str, str]]:
    if len(keywords) < 2:
        return []

    branches = []
    for kw_a, kw_b in itertools.combinations(keywords, 2):
        kw_a_norm = kw_a.lower().replace(' ', '_')
        kw_b_norm = kw_b.lower().replace(' ', '_')
        kw_a_space = kw_a.lower()
        kw_b_space = kw_b.lower()
        f = (
            f'( (CONTAINS(LCASE(STR(?s)), "{kw_a_norm}") || CONTAINS(LCASE(STR(?s)), "{kw_a_space}"))'
            f' && (CONTAINS(LCASE(STR(?o)), "{kw_b_norm}") || CONTAINS(LCASE(STR(?o)), "{kw_b_space}")) )'
            f' || '
            f'( (CONTAINS(LCASE(STR(?s)), "{kw_b_norm}") || CONTAINS(LCASE(STR(?s)), "{kw_b_space}"))'
            f' && (CONTAINS(LCASE(STR(?o)), "{kw_a_norm}") || CONTAINS(LCASE(STR(?o)), "{kw_a_space}")) )'
        )
        # No LIMIT per pair — a per-pair limit fills the result bucket in pyoxigraph scan order,
        # placing GeoNames/DBpedia triples before semantic bridges from ConceptNet/WordNet.
        branches.append(f'{{ ?s ?p ?o . FILTER( {f} ) }}')

    query = "SELECT ?s ?p ?o WHERE { " + " UNION ".join(branches) + " }"
    bridging: Set[Tuple[str, str, str]] = set()
    try:
        for r in store.query(query):
            bridging.add((
                shorten(r['s'].value),
                shorten(r['p'].value),
                shorten(r['o'].value),
            ))
    except Exception as e:
        logger.error("SPARQL bridging query failed: %s", e)

    return list(bridging)


def compute_differential_triples(
    baseline_triples: Set[Tuple[str, str, str]],
    honk_triples: Set[Tuple[str, str, str]],
) -> Set[Tuple[str, str, str]]:
    return honk_triples - baseline_triples
