#!/bin/bash
# =============================================================================
# Slurm HPC validation submitter: MAP recovery on the held-out EVAL set.
# =============================================================================
# Adapts to the allocation: with >1 node it shards the EVAL set across one worker
# per GPU on EVERY node (one Slurm task per GPU on every node), each writing
# its own shard to the shared filesystem, then a single --merge pass combines them
# into one report; with 1 node and >1 GPU it shards across that node's GPUs
# (one Slurm task per GPU) then merges; with 1 GPU it is the original single-GPU
# path (writes the report directly, no merge). --gres is per node, so --nodes=N
# --gres=gpu:G gives N*G shard workers (world_size = N*G). The sharding is
# embarrassingly parallel (no cross-rank communication, hence no torchrun and no
# rendezvous); workers just need the shared output dir for the merge. Node
# count comes from the allocation (Submit.sh NODES -> sbatch --nodes; SLURM_NNODES),
# not an --export knob. Reads the trained posterior + EVAL data, writes the
# recovery report (Posit/..._MAP_Recovery/).
# Overridable via --export: EVAL_TASKS, SUMMARY (map|posterior|both), POOL_MODE
#   (bounded|unrestricted), TOTAL_TIME, SRM_AND_SBI_GPUS (cap the GPUs used;
#   default = all allocated). Sharding is at video granularity, so every allocated
#   GPU gets work regardless of EVAL_TASKS (even EVAL_TASKS=1 spreads its videos).
#   Non-deterministic (no seed).
# Submit from the repo root and forward REPO: Slurm spools this script to
# /var/spool, so the child must be told where the repo is (--export=ALL,REPO=$PWD).
# --job-name follows the data-file naming convention
# SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_<timing_label>_Evaluation; with no TOTAL_TIME set the
# launcher default (2.0 s) gives timing_label 2S_50FPS, so swap the token (e.g.
# 5S_50FPS) whenever you pass TOTAL_TIME=5.0. POOL_MODE defaults to bounded (a
# trained posterior); use POOL_MODE=unrestricted for an undertrained/smoke posterior.
# Example (single node):
#   cd /path/to/srm-and-sbi-monomer-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Evaluation --export=ALL,REPO=$PWD,CONDITION=FAB,EVAL_TASKS=1,SUMMARY=both Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Evaluation.sh
# Example (two nodes, EVAL set sharded across both -- add --nodes=N; --gres is per node):
#   sbatch --nodes=2 --gres=gpu:4 --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Evaluation --export=ALL,REPO=$PWD,CONDITION=FAB,EVAL_TASKS=10,SUMMARY=both Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Evaluation.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Evaluation   # fallback; per-run --job-name (with timing_label) overrides this
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
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

EVAL_TASKS="${EVAL_TASKS:-1}"
SUMMARY="${SUMMARY:-both}"
POOL_MODE="${POOL_MODE:-bounded}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"

# GPUs PER NODE for sharding: SRM_AND_SBI_GPUS override, else the node's allocation,
# else 1. NODE COUNT comes from the Slurm allocation (SLURM_NNODES; 1 for a
# non-Slurm/local run), so the shard-worker count is world_size = NNODES * GPUS.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
NNODES="${SLURM_NNODES:-1}"
# No worker cap by task count: sharding is at VIDEO granularity, so even EVAL_TASKS=1
# spreads its thousands of (task, sim) videos across every allocated GPU. Capping GPUS
# at EVAL_TASKS here (a task-sharding-era rule) would wrongly run a 1-task, 10^4-video
# set on a single GPU. An over-provisioned rank with no videos writes no shard and
# --merge tolerates the missing file.
EVAL_PY="$REPO/Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Evaluation.py"
# CONDITION (FAB|INLB): every product of this stage is condition-specific (the labeling law
# re-images the trajectories per condition), so the token is required and forwarded.
case "${CONDITION:-}" in FAB|INLB) ;; *) echo "FATAL: CONDITION='${CONDITION:-}' (use FAB|INLB)." >&2; exit 1;; esac

EVAL_ARGS=( --condition "$CONDITION" --eval-tasks "$EVAL_TASKS" --summary "$SUMMARY" --pool-mode "$POOL_MODE"
            --total-time-seconds "$TOTAL_TIME" )

echo "=== Evaluation | eval_tasks=${EVAL_TASKS} summary=${SUMMARY} pool=${POOL_MODE} time=${TOTAL_TIME}s nodes=${NNODES} gpus_per_node=${GPUS} world_size=$((NNODES * GPUS)) seed=None | node $(hostname) ==="

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
         python -u "$EVAL_PY" "${EVAL_ARGS[@]}"
    python -u "$EVAL_PY" "${EVAL_ARGS[@]}" --merge
else
    # Single GPU: the original path (writes the report directly; no merge).
    python -u "$EVAL_PY" "${EVAL_ARGS[@]}"
fi

echo "=== Evaluation complete ==="
