"""Unit test for the graph navigability metric (item 8a, reviewer R1-D5).

Exercises the union-find connectivity over a small node-pair file:
component {a,b,c} (size 3), component {d,e} (size 2), and two isolated nodes
(f, g) implied by num_nodes=7.
"""

import os
import tempfile

from benchmarking.compare_graphs import GraphComparator, SEPARATOR


def test_navigability_components_lcc_and_reachability():
    content = f"a{SEPARATOR}b\nb{SEPARATOR}c\nd{SEPARATOR}e\n"
    fd, path = tempfile.mkstemp(suffix=".pairs")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        # _compute_navigability does not use `self`, so None is a fine receiver.
        nav = GraphComparator._compute_navigability(None, path, num_nodes=7)
    finally:
        os.remove(path)

    # {a,b,c}=3, {d,e}=2, isolated f,g => 4 components total.
    assert nav["components"] == 4
    # Largest component is 3 of 7 nodes.
    assert abs(nav["lcc_fraction"] - 3 / 7) < 1e-9
    # Reachability = (3^2 + 2^2 + 1 + 1) / 7^2 = 15/49.
    assert abs(nav["reachability"] - 15 / 49) < 1e-9
