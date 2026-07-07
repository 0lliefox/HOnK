#!/usr/bin/env bash
# =============================================================================
# pull_results.sh — wait for the HPC performance re-time to finish, then pull the
# benchmark outputs back into this repo so the paper tables/figures can be rebuilt.
#
# Fill in your connection details either via environment variables or a local
# untracked config file: slurm/hpc.env  (copied from slurm/hpc.env.example).
#
# Required:
#   HPC_HOST         ssh target for the compute host, e.g.
#                    "b9063849@comet.hpc.ncl.ac.uk" or an ssh config alias
#   HPC_REMOTE_DIR   absolute path to the HOnK checkout on the cluster,
#                    e.g. "/nobackup/proj/comet_honkstar/HOnK"
# Optional:
#   HPC_PROXY_JUMP   jump host if the compute node is reached via a gateway,
#                    e.g. "b9063849@unix.ncl.ac.uk" (adds -o ProxyJump=...)
#   HPC_JOBID        slurm job/array id to wait on (e.g. "12345"); if unset,
#                    --no-wait is assumed and results are pulled immediately
#   HPC_SSH_KEY      path to a private key (e.g. ~/.ssh/honk_hpc); if unset,
#                    ssh uses its defaults/agent
#   POLL_SECONDS     squeue poll interval (default 120)
#
# Usage:
#   slurm/pull_results.sh            # wait on HPC_JOBID (if set), then pull
#   slurm/pull_results.sh --no-wait  # pull immediately
#   HPC_JOBID=12345 slurm/pull_results.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load optional config file
if [[ -f "${SCRIPT_DIR}/hpc.env" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/hpc.env"
fi

WAIT=1
[[ "${1:-}" == "--no-wait" ]] && WAIT=0

: "${HPC_HOST:?Set HPC_HOST (ssh target) in the environment or slurm/hpc.env}"
: "${HPC_REMOTE_DIR:?Set HPC_REMOTE_DIR (remote repo path) in the environment or slurm/hpc.env}"
POLL_SECONDS="${POLL_SECONDS:-120}"

SSH_ARGS=(-o BatchMode=yes)
[[ -n "${HPC_SSH_KEY:-}" ]]    && SSH_ARGS+=(-i "${HPC_SSH_KEY}")
[[ -n "${HPC_PROXY_JUMP:-}" ]] && SSH_ARGS+=(-o "ProxyJump=${HPC_PROXY_JUMP}")

ssh_cmd() { ssh "${SSH_ARGS[@]}" "${HPC_HOST}" "$@"; }

# ---- 1. Wait for the job to finish (if requested and a job id is known) ------
if [[ "${WAIT}" -eq 1 && -n "${HPC_JOBID:-}" ]]; then
    echo "Waiting for slurm job ${HPC_JOBID} on ${HPC_HOST} (poll ${POLL_SECONDS}s)..."
    while true; do
        # squeue prints one line per still-running array element; empty => done
        remaining="$(ssh_cmd "squeue -j ${HPC_JOBID} -h -o %A 2>/dev/null | wc -l" || echo 0)"
        remaining="${remaining//[[:space:]]/}"
        if [[ "${remaining}" == "0" ]]; then
            echo "Job ${HPC_JOBID} no longer in the queue; proceeding to pull."
            break
        fi
        echo "  $(date '+%H:%M:%S')  ${remaining} array task(s) still queued/running..."
        sleep "${POLL_SECONDS}"
    done
elif [[ "${WAIT}" -eq 1 ]]; then
    echo "No HPC_JOBID set; skipping wait (pass --no-wait to silence this)."
fi

# ---- 2. Pull the artifacts back ---------------------------------------------
RSYNC_OPTS=(-avz --info=progress2 -e "ssh ${SSH_ARGS[*]}")

pull() {  # $1 = remote relative path (dir or glob), $2 = local dest dir
    local remote="$1" dest="$2"
    mkdir -p "${dest}"
    echo "==> ${remote}  ->  ${dest}"
    rsync "${RSYNC_OPTS[@]}" "${HPC_HOST}:${HPC_REMOTE_DIR}/${remote}" "${dest}" || \
        echo "   (nothing to pull for ${remote}, continuing)"
}

pull "benchmarking/results/"        "${REPO_DIR}/benchmarking/results/"
pull "ontologies/comparison_*.csv"  "${REPO_DIR}/ontologies/"
pull "logs/"                        "${REPO_DIR}/logs/"

echo ""
echo "Done. Next: rebuild the performance tables/figures from benchmarking/results/"
echo "  (benchmarking/plot.py, benchmarking/plot_results.py) and update the paper."
