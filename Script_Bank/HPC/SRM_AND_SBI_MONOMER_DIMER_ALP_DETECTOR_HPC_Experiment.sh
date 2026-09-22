#!/bin/bash
# =============================================================================
# Slurm HPC application submitter: MAP estimation on real microscopy videos.
# =============================================================================
# Adapts to the allocation: with >1 node it shards the (kind, cell) work across one
# worker per GPU on EVERY node (one Slurm task per GPU on every node), each
# writing its own shard to the shared filesystem, then a single --merge pass
# combines them into one report; with 1 node and >1 GPU it shards across that
# node's GPUs (one Slurm task per GPU) then merges; with 1 GPU it is the original
# single-GPU path (writes the report directly, no merge). --gres is per node, so
# --nodes=N --gres=gpu:G gives N*G shard workers (world_size = N*G). The sharding is
# embarrassingly parallel (no cross-rank communication, hence no torchrun and no
# rendezvous); workers just need the shared output dir for the merge. Node
# count comes from the allocation (Submit.sh NODES -> sbatch --nodes; SLURM_NNODES),
# not an --export knob. Reads the trained posterior + the .tif recordings under
# <data_bank>/Experiment/, writes inferred-parameter distributions per condition
# (Posit/..._MAP_Experiment/).
# Overridable via --export: CONDITION (FAB|INLB, required), KINDS (default = CONDITION), MAX_CELLS (0=all),
#   VERBOSE (0|1, per-window optimizer trace incl. the current learning rate), SHOW_PROGRESS (that trace's step cadence),
#   CHUNK_STEP (seconds; unset -> model-window default, non-overlapping), SUMMARY (map|posterior|both), POOL_MODE, TOTAL_TIME,
#   SRM_AND_SBI_GPUS (cap the GPUs used; default = all allocated),
#   EXIT_BARRIER (seconds; raises torch-elastic's 300 s exit barrier so straggler
#     ranks are not killed; default 3600 -- the job wall time is the real bound).
#   ARTIFACT_TAG (SCREAMING_SNAKE token, e.g. CAP256; appended to the timing label of every
#     PRODUCT of this stage and to the estimator it loads -- Paths.product_label -- so a named experiment lives beside the
#     canonical run; the shared inputs are read under the plain timing label; unset = canonical),
#   A worker that draws no cells writes no shard.
#   Non-deterministic (no seed).
# Submit from the repo root and forward REPO: Slurm spools this script to
# /var/spool, so the child must be told where the repo is (--export=ALL,REPO=$PWD).
# --job-name follows the data-file naming convention
# SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_<timing_label>_Experiment; with no TOTAL_TIME set the
# launcher default (2.0 s) gives timing_label 2S_50FPS, so swap the token (e.g.
# 5S_50FPS) whenever you pass TOTAL_TIME=5.0.
# CAVEAT: Slurm's --export splits its value on commas, so a multi-value KINDS
# CANNOT go inside the --export string (--export=ALL,KINDS=FAB,INLB would parse as
# KINDS=FAB plus a stray, value-less INLB). Either leave KINDS at the script default (the run's CONDITION)
# or pre-export it in the submitting shell and let --export=ALL carry it:
# Example (default KINDS):
#   cd /path/to/srm-and-sbi-monomer-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Experiment --export=ALL,REPO=$PWD,CONDITION=FAB,SUMMARY=both Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Experiment.sh
# Example (multi-value KINDS via the environment, NOT inside --export):
#   cd /path/to/srm-and-sbi-monomer-dimer-alp
#   export KINDS=FAB,INLB   # a deliberate cross-condition application, not the default
#   sbatch --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Experiment --export=ALL,REPO=$PWD,CONDITION=FAB,SUMMARY=both Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Experiment.sh
# Example (two nodes, (kind, cell) work sharded across both -- add --nodes=N; --gres is per node):
#   sbatch --nodes=2 --gres=gpu:4 --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Experiment --export=ALL,REPO=$PWD,CONDITION=FAB,SUMMARY=both Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Experiment.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment   # fallback; per-run --job-name (with timing_label) overrides this
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%j.out   # submit-directory; the controller overrides this via MON_OUT for packed jobs

set -eo pipefail

# Locate the repo root robustly. Slurm runs a SPOOLED COPY of this batch script
# from /var/spool, so BASH_SOURCE is unreliable for a directly-submitted job.
# Resolve REPO from, in order: an explicit REPO (e.g. --export=ALL,REPO=...), the
# Slurm submit directory, or this script's own location (for a non-Slurm
# `bash <script>`); accept the first that actually contains the package, and fail
# loud otherwise rather than crashing cryptically on a /var/spool path.
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
    echo "FATAL: cannot locate the srm-and-sbi-monomer-dimer-alp repo root (Slurm spools this" >&2
    echo "  script, so its own path is unreliable). Submit with an explicit REPO, e.g.:" >&2
    echo "    cd /path/to/srm-and-sbi-monomer-dimer-alp && sbatch --export=ALL,REPO=\$PWD,... <this-script>" >&2
    exit 1
}
cd "$REPO"

# Per-machine HPC config (gitignored; copy from hpc_local.env.example): sets
# MACHINE_PROFILE / CONDA_SETUP / etc. Sourced via the resolved REPO so it is
# found even under Slurm spooling. Falls back to the defaults below if absent.
HPC_ENV="${HPC_ENV:-$REPO/Script_Bank/HPC/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE (via hpc_local.env or --export) to a profile in your machine_profiles.toml}"

KINDS="${KINDS:-$CONDITION}"   # default: the run's condition; override only for a deliberate cross-condition application
MAX_CELLS="${MAX_CELLS:-0}"
# Leave CHUNK_STEP unset by default so the Experiment entry point applies its own
# default (step = the integer model window -> non-overlapping tiling), which is
# valid for ANY --total-time-seconds. A fixed literal here would divide only some
# windows (e.g. 2 divides a 2 s window but not a 5 s one). Set CHUNK_STEP to force
# overlapping chunks (e.g. 1 = 1 s stride).
CHUNK_STEP="${CHUNK_STEP:-}"
SUMMARY="${SUMMARY:-both}"
POOL_MODE="${POOL_MODE:-bounded}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"

# GPUs PER NODE for sharding: SRM_AND_SBI_GPUS override, else the node's allocation,
# else 1. NODE COUNT comes from the Slurm allocation (SLURM_NNODES; 1 for a
# non-Slurm/local run), so the shard-worker count is world_size = NNODES * GPUS.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
NNODES="${SLURM_NNODES:-1}"
EXP_PY="$REPO/Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment.py"
# CONDITION (FAB|INLB): every product of this stage is condition-specific (the labeling law
# re-images the trajectories per condition), so the token is required and forwarded.
case "${CONDITION:-}" in FAB|INLB) ;; *) echo "FATAL: CONDITION='${CONDITION:-}' (use FAB|INLB)." >&2; exit 1;; esac

ARTIFACT_TAG="${ARTIFACT_TAG:-}"   # empty -> canonical product names; else e.g. CAP256 (Paths.product_label)
TAG_ARG=()
[ -n "$ARTIFACT_TAG" ] && TAG_ARG=(--artifact-tag "$ARTIFACT_TAG")
EXP_ARGS=( --condition "$CONDITION" --kinds "$KINDS" --max-cells "$MAX_CELLS"
           --summary "$SUMMARY" --pool-mode "$POOL_MODE" --total-time-seconds "$TOTAL_TIME" "${TAG_ARG[@]}" )
# Forward --chunk-step-seconds only when explicitly set; otherwise let the entry
# point default it to the model window (see the CHUNK_STEP note above).
[ -n "$CHUNK_STEP" ] && EXP_ARGS+=( --chunk-step-seconds "$CHUNK_STEP" )

# VERBOSE=1 turns on the per-window optimizer trace (--verbose -> show=True): the pool/score/elite
# shapes, the per-step progress line carrying the CURRENT learning rate and the running optimum, and
# the stop line naming 'early' or 'full-run' and the step reached. SHOW_PROGRESS sets that line's
# cadence in steps. Both are diagnostics of the MAP optimization itself, off by default because they
# multiply the log volume by the window count.
VERBOSE="${VERBOSE:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-}"
[ "$VERBOSE" = "1" ] && EXP_ARGS+=( --verbose )
[ -n "$SHOW_PROGRESS" ] && EXP_ARGS+=( --show-progress-steps "$SHOW_PROGRESS" )

echo "=== Experiment | kinds=${KINDS} max_cells=${MAX_CELLS} chunk_step=${CHUNK_STEP:-window-default} summary=${SUMMARY} pool=${POOL_MODE} time=${TOTAL_TIME}s tag=${ARTIFACT_TAG:-none} verbose=${VERBOSE} progress_every=${SHOW_PROGRESS:-default} nodes=${NNODES} gpus_per_node=${GPUS} world_size=$((NNODES * GPUS)) seed=None | node $(hostname) ==="

# The sharded stages are embarrassingly parallel: every rank draws its own share and writes
# its own shard, and one --merge pass combines them. They are therefore launched as plain
# Slurm tasks -- one per GPU on every allocated node -- and NOT through torchrun: torchrun's
# elastic agent enforces a 300 s exit barrier that no launcher setting changes (torch 2.9
# ignores TORCHELASTIC_EXIT_BARRIER_TIMEOUT), so when ranks finish more than five minutes
# apart the first node's agent tears down the rendezvous, the other nodes' agents die with a
# connection error and kill any rank still working, and that rank's shard is lost.
# resolve_topology() reads SLURM_NTASKS / SLURM_PROCID / SLURM_LOCALID, so every task knows
# its rank and binds its own GPU; there is no rendezvous, no barrier, and nothing to time out.
# --merge refuses to combine an incomplete shard set (see --allow-partial in the stage's --help).
WORLD=$((NNODES * GPUS))
if [ "$WORLD" -gt 1 ]; then
    CPT_PER_TASK=$(( ${SLURM_CPUS_ON_NODE:-$((GPUS * 4))} / GPUS ))
    echo "    sharded: nodes=$NNODES tasks_per_node=$GPUS world_size=$WORLD cpus_per_task=$CPT_PER_TASK"
    srun --nodes="$NNODES" --ntasks="$WORLD" --ntasks-per-node="$GPUS" \
         --cpus-per-task="$CPT_PER_TASK" --cpu-bind=none \
         python -u "$EXP_PY" "${EXP_ARGS[@]}"
    python -u "$EXP_PY" "${EXP_ARGS[@]}" --merge
else
    # Single GPU: the original path (writes the report directly; no merge).
    python -u "$EXP_PY" "${EXP_ARGS[@]}"
fi

echo "=== Experiment complete ==="
