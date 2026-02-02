#!/bin/bash
ITERATIONS=${1:-1}
BENCHMARK_FILE="benchmarking/results/timing_final_ontology.csv"

# Remove existing benchmark file to start fresh
if [ -f "$BENCHMARK_FILE" ]; then
    echo "Removing existing benchmark file: $BENCHMARK_FILE"
    rm "$BENCHMARK_FILE"
fi

echo "Starting $ITERATIONS iterations..."

for ((i=0; i<ITERATIONS; i++))
do
    echo "Running iteration $((i+1))/$ITERATIONS (Run ID: $i)"

    python build_ontology.py --run-id $i

    if [ $? -ne 0 ]; then
        echo "Error in iteration $((i+1)). Aborting."
        exit 1
    fi

     sleep 1
done

echo "All iterations completed, saved to $BENCHMARK_FILE"
