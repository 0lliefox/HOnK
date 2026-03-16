#!/bin/bash
#SBATCH --account=comet_honkstar
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/honk_array_%A_%a.out
#SBATCH --error=logs/honk_array_%A_%a.err
#SBATCH --array=0-124
#SBATCH --mail-type=ALL
#SBATCH --mail-user=o.fox3@newcastle.ac.uk

echo "=== Processing Task ID: $SLURM_ARRAY_TASK_ID ==="

# 1. Environment Setup
module load Miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pg_env

export PYTHONNOUSERSITE=1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 2. Dynamically calculate Iteration and Config Index
NUM_CONFIGS=25
ITERATION=$(( SLURM_ARRAY_TASK_ID / NUM_CONFIGS ))
CONFIG_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_CONFIGS ))

echo "Mapped Task $SLURM_ARRAY_TASK_ID -> Iteration: $ITERATION, Config Index: $CONFIG_INDEX"

# 3. Database Setup for Parallel Tasks
PORT=$((15432 + SLURM_ARRAY_TASK_ID))
PG_DATA="/nobackup/proj/comet_honkstar/postgres_data_$SLURM_ARRAY_TASK_ID"

# Initialize a fresh database cluster for this specific task
initdb -D "$PG_DATA" -U fox

# Ensure the temporary database is destroyed when the task finishes to save quota
trap 'echo "Cleaning up DB..."; pg_ctl -D "$PG_DATA" stop; rm -rf "$PG_DATA"' EXIT

# Start the unique database
pg_ctl -D "$PG_DATA" -l "$PG_DATA/pg_log.txt" -o "-p $PORT" start
sleep 5

# Create the specific ontology database for this run
psql -h localhost -p $PORT -d postgres -U fox -c "CREATE DATABASE ontology_db;"

# 4. Generate the specific config for this run
# Ensure run_experiments.py handles prepending the iteration to the output_file!
OUTPUT=$(python3 run_experiments.py --iteration $ITERATION --config-index $CONFIG_INDEX)
read -r TEMP_CONFIG_FILE RUN_ID <<< "$OUTPUT"

echo "Running Global Run ID: $RUN_ID using $TEMP_CONFIG_FILE"

# 5. Inject the unique dynamic port into the generated YAML config
sed -i "s/port: '15432'/port: '$PORT'/g" "$TEMP_CONFIG_FILE"

# 6. Execute the pipeline (-u prevents output buffering)
python3 -u build_ontology.py --config "$TEMP_CONFIG_FILE" --run-id "$RUN_ID"

# Clean up temporary config file
rm "$TEMP_CONFIG_FILE"
echo "=== Task $SLURM_ARRAY_TASK_ID Completed ==="
