#!/bin/bash
#SBATCH --account=comet_honkstar
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/graph_scalability_%A_%a.out
#SBATCH --error=logs/graph_scalability_%A_%a.err
#SBATCH --array=0-9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=o.fox3@newcastle.ac.uk

echo "=== Graph Scalability Task ID: $SLURM_ARRAY_TASK_ID ==="

module load Miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pg_env

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

INPUT_FILE="${SLURM_SUBMIT_DIR}/ontologies/ontology_graph_oxi_clustered_urlskept_final.nt"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Graph .nt file not found at ${INPUT_FILE}. Run setup_scalability.slurm first."
    exit 1
fi

# Each task writes to its own subdir; merge afterwards with merge_scalability_results.py
TASK_WORKDIR="${SLURM_SUBMIT_DIR}/benchmarking/results/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TASK_WORKDIR}/benchmarking/results"

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "--- Graph clustering scalability test (task $SLURM_ARRAY_TASK_ID) ---"
python3 "${SLURM_SUBMIT_DIR}/benchmarking/graph_scalability_test.py" "${INPUT_FILE}" --runs 1

echo "=== Task $SLURM_ARRAY_TASK_ID Completed ==="
