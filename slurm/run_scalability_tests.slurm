#!/bin/bash
#SBATCH --account=comet_honkstar
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/scalability_%A_%a.out
#SBATCH --error=logs/scalability_%A_%a.err
#SBATCH --array=0-9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=o.fox3@newcastle.ac.uk

echo "=== Processing Task ID: $SLURM_ARRAY_TASK_ID ==="

# ── Environment Setup ─────────────────────────────────────────────────────────
module load Miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pg_env

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# ── Paths set by setup_scalability.slurm ─────────────────────────────────────
DB_DUMP="/nobackup/proj/comet_honkstar/scalability_db.dump"
INPUT_FILE="${SLURM_SUBMIT_DIR}/ontologies/ontology_graph_oxi_clustered_urlskept_final.nt"

if [[ ! -f "$DB_DUMP" ]]; then
    echo "ERROR: DB dump not found at ${DB_DUMP}. Run setup_scalability.slurm first."
    exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Graph .nt file not found at ${INPUT_FILE}. Run setup_scalability.slurm first."
    exit 1
fi

# ── Per-task database (restored from shared dump) ────────────────────────────
# Each task gets its own PostgreSQL so concurrent scalability tests do not
# conflict when creating/dropping the _test tables.
PORT=$((15432 + SLURM_ARRAY_TASK_ID))
PG_DATA="/nobackup/proj/comet_honkstar/postgres_scalability_${SLURM_ARRAY_TASK_ID}"

initdb -D "$PG_DATA" -U fox

trap 'echo "Cleaning up DB..."; pg_ctl -D "$PG_DATA" stop; rm -rf "$PG_DATA"' EXIT

pg_ctl -D "$PG_DATA" -l "$PG_DATA/pg_log.txt" -o "-p $PORT" start
sleep 5

psql -h localhost -p $PORT -d postgres -U fox -c "CREATE DATABASE ontology_db;"

echo "Restoring database from dump..."
pg_restore -h localhost -p $PORT -U fox -d ontology_db "$DB_DUMP"
echo "Restore complete."

# ── Per-task working directory ────────────────────────────────────────────────
# Each task writes benchmark CSVs to its own subdirectory to avoid file conflicts.
# After all tasks complete, merge with:
#   python3 benchmarking/merge_scalability_results.py
TASK_WORKDIR="${SLURM_SUBMIT_DIR}/benchmarking/results/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TASK_WORKDIR}/benchmarking/results"

# Place a config.yaml in the task workdir pointing at this task's DB port.
# get_config() reads config.yaml from the current working directory.
cp "${SLURM_SUBMIT_DIR}/slurm/config_finaldb.yaml" "${TASK_WORKDIR}/config.yaml"
sed -i "s/port: '5432'/port: '$PORT'/g" "${TASK_WORKDIR}/config.yaml"

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

# ── Step 1: DB clustering scalability test ────────────────────────────────────
echo ""
echo "--- Step 1: DB clustering scalability test (task $SLURM_ARRAY_TASK_ID) ---"
cd "${TASK_WORKDIR}"
python3 -m benchmarking.scalability_test --runs 1

# ── Step 2: Graph clustering scalability test ─────────────────────────────────
echo ""
echo "--- Step 2: Graph clustering scalability test (task $SLURM_ARRAY_TASK_ID) ---"
python3 -m benchmarking.graph_scalability_test "${INPUT_FILE}" --runs 1

echo ""
echo "=== Task $SLURM_ARRAY_TASK_ID Completed ==="
