#!/bin/bash
CONFIGURATIONS_FILE="configurations.json"
BENCHMARK_FILE="benchmarking/results/timing_final_ontology.csv"
ITERATIONS=${1:-1}
START_GLOBAL_ID=${2:-0}

# Remove existing benchmark file to start fresh
if [ -f "$BENCHMARK_FILE" ] && [ "$START_GLOBAL_ID" -eq 0 ]; then
    echo "Removing existing benchmark file: $BENCHMARK_FILE"
    rm "$BENCHMARK_FILE"
fi

echo "Starting iterations based on $CONFIGURATIONS_FILE..."

# Get number of configurations
NUM_CONFIGS=$(python3 -c "import json; print(len(json.load(open('$CONFIGURATIONS_FILE'))))")

GLOBAL_COUNTER=0

for ((i=0; i<ITERATIONS; i++))
do
    echo "--- Starting Iteration Set $((i+1))/$ITERATIONS ---"

    for ((j=0; j<NUM_CONFIGS; j++))
    do
        if [ "$GLOBAL_COUNTER" -lt "$START_GLOBAL_ID" ]; then
            echo "Skipping experiment $GLOBAL_COUNTER (Target: $START_GLOBAL_ID)"
            ((GLOBAL_COUNTER++))
            continue
        fi

        # Generate config and get filename/run_id
        OUTPUT=$(python3 run_experiments.py --iteration $i --config-index $j)

        if [ $? -ne 0 ]; then
            echo "Error generating configuration."
            exit 1
        fi

        read -r TEMP_CONFIG_FILE RUN_ID <<< "$OUTPUT"

        echo "Running experiment $j (Global Run ID: $RUN_ID, Counter: $GLOBAL_COUNTER) using $TEMP_CONFIG_FILE"

        python3 build_ontology.py --config "$TEMP_CONFIG_FILE" --run-id "$RUN_ID"

        if [ $? -ne 0 ]; then
            echo "Error in run $RUN_ID. Aborting."
            rm "$TEMP_CONFIG_FILE"
            exit 1
        fi

        rm "$TEMP_CONFIG_FILE"
        ((GLOBAL_COUNTER++))
    done
done

echo "All iterations completed, saved to $BENCHMARK_FILE"
