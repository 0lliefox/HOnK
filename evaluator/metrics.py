import logging
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


def compute_deterministic_metrics(
    triples: Set[Tuple[str, str, str]],
    keywords: List[str],
    bridging_triples: List[Tuple[str, str, str]],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "hit_rate": 0.0,
        "subgraph_volume": len(triples),
        "avg_path_length": float('inf'),
        "bridging_triple_count": len(bridging_triples),
        "relation_diversity": 0,
        "keyword_connectivity": 0.0,
    }
    if not triples:
        return metrics

    G = nx.Graph()
    for s, _, o in triples:
        G.add_edge(s, o)

    metrics["relation_diversity"] = len({p for _, p, _ in triples})

    def _kw_in_node(kw: str, node: str) -> bool:
        kw_norm = kw.replace(' ', '_')
        return kw in node or kw_norm in node

    matched = [kw for kw in keywords if any(_kw_in_node(kw, node) for node in G.nodes())]
    metrics["hit_rate"] = (len(matched) / len(keywords)) * 100 if keywords else 0.0

    target_nodes_per_kw = {
        kw: [n for n in G.nodes() if _kw_in_node(kw, n)]
        for kw in matched
    }

    if len(matched) >= 2:
        pairs = list(combinations(matched, 2))
        connected_pairs = 0
        path_lengths = []
        for kw_u, kw_v in pairs:
            nodes_u = target_nodes_per_kw[kw_u]
            nodes_v = target_nodes_per_kw[kw_v]
            pair_connected = False
            for nu in nodes_u:
                for nv in nodes_v:
                    if nx.has_path(G, nu, nv):
                        pair_connected = True
                        path_lengths.append(nx.shortest_path_length(G, nu, nv))
                        break
                if pair_connected:
                    break
            if pair_connected:
                connected_pairs += 1
        metrics["keyword_connectivity"] = (connected_pairs / len(pairs)) * 100 if pairs else 0.0
        if path_lengths:
            metrics["avg_path_length"] = sum(path_lengths) / len(path_lengths)

    return metrics
