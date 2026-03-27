import logging
import re
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


def format_context(
        triples: Set[Tuple[str, str, str]],
        keywords: List[str],
        bridging_triples: List[Tuple[str, str, str]],
        max_context_triples: int = 75,
        max_bridge_triples: int = 20,
) -> str:
    if not triples and not bridging_triples:
        return ""

    kw_regexes = {}
    for k in keywords:
        pattern = rf'(?:^|[^a-zA-Z0-9]){re.escape(k)}(?:[^a-zA-Z0-9]|$)'
        kw_regexes[k] = re.compile(pattern, re.IGNORECASE)

    # Generic hub nodes that bloat context without adding specific semantic value
    generic_hubs = {'entity', 'thing', 'concept', 'node', 'item'}

    def _relevance(triple: Tuple[str, str, str]) -> float:
        s, p, o = triple
        score = 0.0

        for k in keywords:
            k_norm = k.replace(' ', '_')
            kw_len = max(len(k), len(k_norm))
            regex = kw_regexes[k]

            for label in (s, o):
                # Exact matches score highest
                if label.lower() == k.lower() or label.lower() == k_norm.lower():
                    score += 3.0
                    # Strict boundary regex match
                elif regex.search(label):
                    score += kw_len / max(len(label), 1)

                    # Penalise multi-word or hyphenated nodes if the keyword is a single word
                    if " " not in k and (" " in label or "-" in label):
                        score -= 0.5

        # Give priority to strong semantic relationships
        if p in ['eq', 'synonym', 'equivalentTo', 'sameAs']:
            score += 1.5
        elif p in ['isA', 'partOf', 'instanceOf']:
            score += 1.0
        elif p in ['relatedTo', 'hasContext']:
            score += 0.2

        # Heavily penalise morphological/etymological derivations
        if p in ['derivedFrom', 'etymologicallyRelatedTo', 'formOf']:
            score -= 1.0

        # Slightly penalise highly generic hub nodes to filter out noise
        if s.lower() in generic_hubs or o.lower() in generic_hubs:
            score -= 0.5

        return score

    lines = []

    # Bridging triples sorted by relevance so semantic bridges appear before
    # geographic coincidences regardless of pyoxigraph storage order.
    bridges_sorted = sorted(bridging_triples, key=_relevance, reverse=True)
    bridges_shown = bridges_sorted[:max_bridge_triples]
    if bridges_shown:
        lines.append("--- Direct Keyword Connections ---")
        for s, p, o in bridges_shown:
            lines.append(f"({s} -> {p} -> {o})")

    # Remaining budget filled with relevance-positive general triples.
    general_cap = max(0, max_context_triples - len(bridges_shown))
    bridge_set = set(bridging_triples)
    general = sorted(
        (t for t in triples if t not in bridge_set and _relevance(t) > 0),
        key=_relevance,
        reverse=True,
    )
    if general and general_cap > 0:
        lines.append("--- General Semantic Context ---")
        for s, p, o in general[:general_cap]:
            lines.append(f"({s} -> {p} -> {o})")

    return "\n".join(lines)