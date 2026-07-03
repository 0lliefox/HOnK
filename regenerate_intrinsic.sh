#!/usr/bin/env bash
# Regenerate the six builds behind the paper's intrinsic-comparison tables
# (table:concept-net-comparison, table:clustering-comparison,
#  table:normalised-comparison) and re-run the pairwise comparisons plus the
# rdf:type class decomposition.
#
# Notes
# -----
# * Builds run in GRAPH mode (Pyoxigraph, in-memory), so PostgreSQL is
#   untouched; the historical artifacts were DB-mode HPC runs, and mode
#   equivalence is covered by tests/test_graph_db_equivalence.py.
# * "Unnormalised" is expressed as `normalisation: raw`.
# * Outputs are regen_-prefixed so the historical db_*_final.nt artifacts in
#   ontologies/ are never overwritten; compare or swap them deliberately.
# * Full-graph builds are large (~6-7 GB each) and take hours; run overnight.
set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python

for i in 0 1 2 3 4 5; do
  $PY run_experiments.py --iteration 0 --config-index "$i" \
      --configurations-file configurations_intrinsic.json
done

$PY benchmarking/compare_graphs.py \
    "ontologies/regen_db_CN_oxi_clustered.nt" \
    "ontologies/regen_db_CN+WK+WN_oxi_clustered.nt" \
    --name1 "CN Only" --name2 "CN+WK+WN"

$PY benchmarking/compare_graphs.py \
    "ontologies/regen_db_oxi_clustered.nt" \
    "ontologies/regen_db_oxi_unclustered.nt" \
    --name1 "Clustered" --name2 "Unclustered"

$PY benchmarking/compare_graphs.py \
    "ontologies/regen_db_oxi_unnormalised_clustered.nt" \
    "ontologies/regen_db_oxi_unnormalised_unclustered.nt" \
    --name1 "Unnormalised Clustered" --name2 "Unnormalised Unclustered"

$PY tools/decompose_type_classes.py ontologies/regen_*.nt
