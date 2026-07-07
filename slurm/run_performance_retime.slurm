#!/bin/bash
# =============================================================================
# Performance re-time for the v1.5.0 (post-fidelity-fix) build.
#
# The fidelity fixes (sense-aware identity, type-coherence guard, self-loop
# filter, GeoNames geonameid keying) enlarged the graph by ~75% (50.6M -> 88.4M
# triples), so the paper's performance/timing tables must be re-measured on the
# current code. This job re-runs the full 25-config benchmark matrix
# (configurations.json) x 5 iterations = 125 tasks, exactly as honk_array.slurm
# did for the original numbers, but on the fixed branch.
#
# It regenerates the CSVs behind:
#   - table:pipeline-performance   (runtime, memory, throughput, nodes/triples)
#   - tab:serialisation_times      (RDFLib vs Oxigraph, NT/Turtle/Rapper)
#   - tab:benchmark-batch-size-times
#   - table:unique-source-execution-times
#   - Supplementary direct-to-graph + dataset-execution-time figures
#
# The scalability figure (fig:clustering-scalability) is produced separately by
# setup_scalability.slurm -> run_scalability_tests.slurm.
#
# Submit order:
#   sbatch slurm/run_performance_retime.slurm
#   sbatch slurm/setup_scalability.slurm            # builds DB dump + .nt snapshot
#   sbatch --dependency=afterok:<setup_jobid> slurm/run_scalability_tests.slurm
#
# Pull results back with:  slurm/pull_results.sh
# =============================================================================
#SBATCH --account=comet_honkstar
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/perf_retime_%A_%a.out
#SBATCH --error=logs/perf_retime_%A_%a.err
#SBATCH --array=0-124
#SBATCH --mail-type=ALL
#SBATCH --mail-user=o.fox3@newcastle.ac.uk

echo "=== Performance re-time: Task ID $SLURM_ARRAY_TASK_ID ==="

# 1. Environment
module load Miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pg_env

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 2. Map task -> (iteration, config) over the 25-config matrix, 5 iterations
NUM_CONFIGS=25
ITERATION=$(( SLURM_ARRAY_TASK_ID / NUM_CONFIGS ))
CONFIG_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_CONFIGS ))
echo "Task $SLURM_ARRAY_TASK_ID -> Iteration $ITERATION, Config $CONFIG_INDEX"

# 3. Per-task PostgreSQL on a unique port (one ontology per process)
PORT=$((15432 + SLURM_ARRAY_TASK_ID))
PG_DATA="/nobackup/proj/comet_honkstar/postgres_data_$SLURM_ARRAY_TASK_ID"

# On a preemptible partition a killed task can leave $PG_DATA behind, which makes
# initdb fail on requeue; clean any stale cluster first so the task is re-runnable.
rm -rf "$PG_DATA"
initdb -D "$PG_DATA" -U fox
trap 'echo "Cleaning up DB..."; pg_ctl -D "$PG_DATA" stop; rm -rf "$PG_DATA"' EXIT

pg_ctl -D "$PG_DATA" -l "$PG_DATA/pg_log.txt" -o "-p $PORT" start
sleep 5
psql -h localhost -p $PORT -d postgres -U fox -c "CREATE DATABASE ontology_db;"

# 4. Generate this run's config and capture the global run id
OUTPUT=$(python3 run_experiments.py --iteration $ITERATION --config-index $CONFIG_INDEX)
read -r TEMP_CONFIG_FILE RUN_ID <<< "$OUTPUT"
echo "Global Run ID: $RUN_ID using $TEMP_CONFIG_FILE"

# 5. Inject the unique dynamic port into the generated YAML (robust to whatever
#    port the base config carries, e.g. 5432 or 15432)
sed -i -E "s/port: *'[0-9]+'/port: '$PORT'/g" "$TEMP_CONFIG_FILE"

# 6. Execute the pipeline (-u prevents output buffering)
python3 -u build_ontology.py --config "$TEMP_CONFIG_FILE" --run-id "$RUN_ID"

rm "$TEMP_CONFIG_FILE"
echo "=== Task $SLURM_ARRAY_TASK_ID Completed ==="
