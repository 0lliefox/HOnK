#!/bin/bash
#SBATCH --account=comet_honkstar
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/scalability_setup_%j.out
#SBATCH --error=logs/scalability_setup_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=o.fox3@newcastle.ac.uk

echo "=== Scalability Setup Job ==="

# ── Environment Setup ─────────────────────────────────────────────────────────
module load Miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pg_env

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p "${SLURM_SUBMIT_DIR}/logs"
mkdir -p "${SLURM_SUBMIT_DIR}/ontologies"

# ── Database Setup ────────────────────────────────────────────────────────────
PORT=15432
PG_DATA="/nobackup/proj/comet_honkstar/postgres_scalability_setup"
DB_DUMP="/nobackup/proj/comet_honkstar/scalability_db.dump"

rm -rf "$PG_DATA"   # clean any stale cluster so initdb does not fail on re-run
initdb -D "$PG_DATA" -U fox

trap 'echo "Stopping setup DB..."; pg_ctl -D "$PG_DATA" stop' EXIT

pg_ctl -D "$PG_DATA" -l "$PG_DATA/pg_log.txt" -o "-p $PORT" start
sleep 5

psql -h localhost -p $PORT -d postgres -U fox -c "CREATE DATABASE ontology_db;"

# Inject the setup port into a temp copy of the DB config
DB_CONFIG="${SLURM_SUBMIT_DIR}/slurm/config_finaldb_setup.yaml"
cp "${SLURM_SUBMIT_DIR}/slurm/config_finaldb.yaml" "$DB_CONFIG"
sed -i "s/port: '15432'/port: '$PORT'/g" "$DB_CONFIG"

trap 'echo "Stopping setup DB..."; pg_ctl -D "$PG_DATA" stop; rm -f "$DB_CONFIG"' EXIT

cd "${SLURM_SUBMIT_DIR}"

# ── Step 1: Run DB pipeline to populate the database ─────────────────────────
echo ""
echo "--- Step 1: Building DB pipeline ---"
python3 -u build_ontology.py --config "$DB_CONFIG" --run-id 0

# ── Step 2: Dump the populated database ──────────────────────────────────────
echo ""
echo "--- Step 2: Dumping database to ${DB_DUMP} ---"
pg_dump -h localhost -p $PORT -U fox -Fc ontology_db -f "$DB_DUMP"
echo "Dump complete: $(du -sh "$DB_DUMP" | cut -f1)"

# ── Step 3: Run graph pipeline to produce the .nt snapshot ───────────────────
echo ""
echo "--- Step 3: Building graph pipeline (.nt with URL triples) ---"
python3 -u build_ontology.py --config "${SLURM_SUBMIT_DIR}/slurm/config_finalgraphcluster.yaml" --run-id 0

GRAPH_NT="${SLURM_SUBMIT_DIR}/ontologies/ontology_graph_oxi_clustered_urlskept_final.nt"
if [[ ! -f "$GRAPH_NT" ]]; then
    echo "ERROR: Graph .nt file not produced at ${GRAPH_NT}"
    exit 1
fi
echo "Graph snapshot ready: $(du -sh "$GRAPH_NT" | cut -f1)"

echo ""
echo "=== Setup complete. Submit the array job with: ==="
echo "  sbatch --dependency=afterok:\${SLURM_JOB_ID} slurm/run_scalability_tests.slurm"
