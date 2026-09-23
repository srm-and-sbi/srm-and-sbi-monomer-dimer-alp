#!/bin/bash
# =============================================================================
# Slurm HPC submitter: MAP optimization benchmark (detector workflow).
# =============================================================================
# Compares the production MAP optimizer with new-loop settings on the real trained flow, from
# identical starting candidates, on two independent candidate pools per recording, for a subset of
# the EVAL tier (see Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_MAP_Benchmark.md).
#
# SHARDED: one rank per GPU of the node (srun tasks, not torchrun), each writing a shard; one
# --merge pass then combines the shards (it requires every rank and the exact recording list) and
# writes the report. The invocation id is created here, before the ranks start.
#
# This is a validation ANALYSIS, not a canonical stage: it is deliberately NOT wired into
# SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Submit.sh and is submitted directly with sbatch.
# Dry run first: `python <shim> ... --dry-run` on the login node, then `sbatch --test-only`.
#
# Overridable via --export: CONDITION (FAB|INLB), TOTAL_TIME, TASKS ("0 1"), MAX_SIMS, ARTIFACT_TAG,
#   GROUP_SIZE, EXTRA (extra CLI flags).
#
# Example (EVAL tasks 0 and 1, 2,000 recordings):
#   cd /path/to/srm-and-sbi-monomer-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_MAP_Benchmark \
#          --export=ALL,REPO=$PWD,CONDITION=FAB,TASKS="0 1" \
#          Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_MAP_Benchmark.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_MAP_Benchmark
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%j.out

set -eo pipefail

_find_repo() {
    local c
    for c in "${REPO:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/../..}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
        [ -n "$c" ] || continue
        if [ -f "$c/pyproject.toml" ] && [ -d "$c/srm_and_sbi_monomer_dimer_alp" ]; then
            (cd "$c" && pwd); return 0
        fi
    done
    return 1
}
REPO="$(_find_repo)" || {
    echo "FATAL: cannot locate the srm-and-sbi-monomer-dimer-alp repo root. Submit with an" >&2
    echo "  explicit REPO, e.g. --export=ALL,REPO=\$PWD,..." >&2; exit 1; }
cd "$REPO"

HPC_ENV="${HPC_ENV:-$REPO/Script_Bank/HPC/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE (via hpc_local.env or --export)}"

case "${CONDITION:-}" in FAB|INLB) ;; *) echo "FATAL: CONDITION='${CONDITION:-}' (use FAB|INLB)." >&2; exit 1;; esac
TOTAL_TIME="${TOTAL_TIME:-2.0}"
TASKS="${TASKS:-0 1}"
MAX_SIMS="${MAX_SIMS:-0}"
GROUP_SIZE="${GROUP_SIZE:-64}"
ARTIFACT_TAG="${ARTIFACT_TAG:-}"
TAG_ARG=(); [ -n "$ARTIFACT_TAG" ] && TAG_ARG=(--artifact-tag "$ARTIFACT_TAG")

# ABSOLUTE script path: hpc_local.env may cd elsewhere after this script's own cd.
PY="$REPO/Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_MAP_Benchmark.py"
# shellcheck disable=SC2206
ARGS=( --condition "$CONDITION" --total-time-seconds "$TOTAL_TIME" --tasks $TASKS
       --max-sims "$MAX_SIMS" --group-size "$GROUP_SIZE" "${TAG_ARG[@]}" )

GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
NNODES="${SLURM_NNODES:-1}"
echo "=== MAP_Benchmark | condition=${CONDITION} time=${TOTAL_TIME}s tasks=${TASKS} max_sims=${MAX_SIMS} tag=${ARTIFACT_TAG:-none} nodes=${NNODES} gpus_per_node=${GPUS} | node $(hostname) ==="

# One invocation identifier for EVERY rank of this launch, created before the ranks start; the
# shards record it and --merge refuses shards whose identifiers differ.
export SRM_AND_SBI_INVOCATION_ID="${SRM_AND_SBI_INVOCATION_ID:-$(python -c 'import uuid; print(uuid.uuid4())')}"
echo "    invocation id: ${SRM_AND_SBI_INVOCATION_ID}"
WORLD=$((NNODES * GPUS))
if [ "$WORLD" -gt 1 ]; then
    CPT_PER_TASK=$(( ${SLURM_CPUS_ON_NODE:-$((GPUS * 4))} / GPUS ))
    echo "    sharded: nodes=$NNODES tasks_per_node=$GPUS world_size=$WORLD cpus_per_task=$CPT_PER_TASK"
    srun --nodes="$NNODES" --ntasks="$WORLD" --ntasks-per-node="$GPUS" \
         --cpus-per-task="$CPT_PER_TASK" --cpu-bind=none \
         python -u "$PY" "${ARGS[@]}" ${EXTRA:-}
    python -u "$PY" "${ARGS[@]}" --merge ${EXTRA:-}
else
    python -u "$PY" "${ARGS[@]}" ${EXTRA:-}
fi
echo "=== MAP_Benchmark complete ==="
