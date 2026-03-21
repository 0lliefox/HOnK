import logging
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

    kw_lower = [k.lower() for k in keywords]

    def _relevance(triple: Tuple[str, str, str]) -> float:
        s, _, o = triple
        score = 0.0
        for k in kw_lower:
            k_norm = k.replace(' ', '_')
            kw_len = max(len(k), len(k_norm))
            for label in (s.lower(), o.lower()):
                if label == k or label == k_norm:
                    score += 3.0  # exact match
                elif k in label or k_norm in label:
                    # Coverage ratio: penalises long geographic named-entity labels
                    # (e.g. "Thornbury_Neighbourhood_Centre") while preserving full
                    # weight for short concept labels (e.g. "brightness").
                    score += kw_len / max(len(label), 1)
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
