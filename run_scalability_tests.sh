#!/usr/bin/env bash
# Usage: ./run_scalability_tests.sh <input.nt> [--runs N]
#
# Runs the DB clustering scalability test followed by the graph clustering
# scalability test (both RDF and Oxigraph backends).
#
# Prerequisites:
#   - DB test: the database must already be populated with a full pipeline run
#     (clusters table present). The pipeline must have been run in db mode.
#   - Graph test: <input.nt> must be a pre-clustering snapshot with hasURL triples
#     present. To produce one, set keep_url_triples: true in config.yaml before
#     running the pipeline in graph mode, then pass the output .nt here.
#
# Results are written to benchmarking/results/:
#   clustering_benchmark.csv          (DB mode)
#   graph_rdf_clustering_benchmark.csv
#   graph_oxi_clustering_benchmark.csv
#   graph_rdf_memory_benchmark.csv
#   graph_oxi_memory_benchmark.csv

set -euo pipefail

RUNS=10
INPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) INPUT_FILE="$1"; shift ;;
    esac
done

if [[ -z "$INPUT_FILE" ]]; then
    echo "Usage: $0 <input.nt> [--runs N]"
    exit 1
fi

# Run from the project root regardless of where the script is called from
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "============================================"
echo " DB clustering scalability test"
echo " Runs: $RUNS"
echo "============================================"
python -m benchmarking.scalability_test --runs "$RUNS"

echo ""
echo "============================================"
echo " Graph clustering scalability test (RDF + Oxigraph)"
echo " Input: $INPUT_FILE"
echo " Runs: $RUNS"
echo "============================================"
python benchmarking/graph_scalability_test.py "$INPUT_FILE" --runs "$RUNS"

echo ""
echo "All scalability tests complete. Results in benchmarking/results/"
