#!/bin/bash
#SBATCH --account=comet_honkstar
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/db_scalability_%A_%a.out
#SBATCH --error=logs/db_scalability_%A_%a.err
#SBATCH --array=0-9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=o.fox3@newcastle.ac.uk

echo "=== DB Scalability Task ID: $SLURM_ARRAY_TASK_ID ==="

module load Miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pg_env

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

DB_DUMP="/nobackup/proj/comet_honkstar/scalability_db.dump"

if [[ ! -f "$DB_DUMP" ]]; then
    echo "ERROR: DB dump not found at ${DB_DUMP}. Run setup_scalability.slurm first."
    exit 1
fi

# Each task gets its own postgres instance to avoid conflicts on _test tables
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

# Each task writes to its own subdir; merge afterwards with merge_scalability_results.py
TASK_WORKDIR="${SLURM_SUBMIT_DIR}/benchmarking/results/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TASK_WORKDIR}/benchmarking/results"

# Copy config and patch port so get_config() picks up the right instance
cp "${SLURM_SUBMIT_DIR}/slurm/config_finaldb.yaml" "${TASK_WORKDIR}/config.yaml"
sed -i "s/port: '15432'/port: '$PORT'/g" "${TASK_WORKDIR}/config.yaml"

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "--- DB clustering scalability test (task $SLURM_ARRAY_TASK_ID) ---"
cd "${TASK_WORKDIR}"
python3 "${SLURM_SUBMIT_DIR}/benchmarking/scalability_test.py" --runs 1

echo "=== Task $SLURM_ARRAY_TASK_ID Completed ==="
