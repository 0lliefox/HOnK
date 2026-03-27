import itertools
import logging
from typing import List, Set, Tuple

import pyoxigraph

try:
    from .store_loader import shorten
except ImportError:
    from store_loader import shorten  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def _build_regex_filter(kw: str, var: str) -> str:
    safe_kw = kw.replace('"', '')
    return f'REGEX(STR({var}), "(^|[^a-zA-Z0-9]){safe_kw}([^a-zA-Z0-9]|$)", "i")'


def extract_subgraph(
        store: pyoxigraph.Store,
        keywords: List[str],
) -> Set[Tuple[str, str, str]]:
    if not keywords:
        return set()

    branches = []
    for kw in keywords:
        kw_norm = kw.replace(' ', '_')  # preserve original case

        f = f'{_build_regex_filter(kw_norm, "?s")} || {_build_regex_filter(kw_norm, "?o")}'

        if kw_norm != kw:  # multi-word keyword: also match space form
            f += f' || {_build_regex_filter(kw, "?s")} || {_build_regex_filter(kw, "?o")}'

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

    def _kw_filter(kw: str, var: str) -> str:
        norm = kw.replace(' ', '_')
        if norm != kw:
            return f'({_build_regex_filter(norm, var)} || {_build_regex_filter(kw, var)})'
        return _build_regex_filter(norm, var)

    branches = []
    for kw_a, kw_b in itertools.combinations(keywords, 2):
        fa_s, fa_o = _kw_filter(kw_a, '?s'), _kw_filter(kw_a, '?o')
        fb_s, fb_o = _kw_filter(kw_b, '?s'), _kw_filter(kw_b, '?o')
        f = (
            f'( {fa_s} && {fb_o} )'
            f' || '
            f'( {fb_s} && {fa_o} )'
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