#!/bin/bash
# =============================================================================
# Slurm HPC submitter: the direct imaging estimators on one CPU node (detector workflow).
# =============================================================================
# Runs one of the direct (non-neural) estimator utilities of Script_Bank/Analysis -- PSF width,
# flicker rate, fluorescence loss, or the flicker mismatch study -- as one process with a pool of
# workers on one whole CPU node. The utilities read a synthetic tier (or render in-memory scenes)
# and write a NEW run folder under the Data_Bank Posit tier with report.md, the per-recording
# arrays, summary.json and provenance.json; an existing folder is refused before anything is read.
# DETECTOR_WORKFLOW.md sec. 9.6 records how their results are judged and which EVAL tasks are
# development and which reserved; a tier run declares its PURPOSE, and the utility refuses tasks
# that purpose may not read, before it reads a recording.
#
# These are analysis UTILITIES, not canonical stages: deliberately NOT wired into
# SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Submit.sh and submitted directly with sbatch.
# Dry run first: run the utility with --dry-run on the login node (it resolves paths, applies every
# refusal the run would apply, and prints what it would read and write, touching nothing), then sbatch.
#
# Overridable via --export:
#   ESTIMATOR   PSF_Width | Flicker_Rate | Fluorescence_Loss | Flicker_Mismatch   (required)
#   CONDITION   FAB | INLB (tier runs)          TOTAL_TIME  recording length in s (default 2.0)
#   TASKS       EVAL task indices, e.g. "2 3 4 5 6 7 8 9" (tier runs)
#   MAX_VIDEOS  cap on recordings scored (default: every recording of TASKS)
#   EXPECT      recordings expected per task (default 1000; the 20 s tier holds 100)
#   PURPOSE     development | verdict   (required for a tier run; the folder name carries DEV or VERDICT)
#   RUN_SUFFIX  appended to the run folder name after the purpose token, e.g. the commit -> _DEV_<commit>
#   CODE_COMMIT recorded in provenance.json (a synced tree without .git cannot report it itself)
#   WORKERS     worker processes (default 48, one per physical core of a JUWELS batch node)
#   EXTRA       further CLI flags passed verbatim (e.g. "--replicates 4" for the mismatch study)
#
# Example (development PSF run on the 2 s tier, EVAL tasks 2-9, JUWELS):
#   cd /path/to/srm-and-sbi-monomer-dimer-alp
#   sbatch --partition=batch --account=chkf10 \
#          --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Direct_PSF_Width_DEV_<commit> \
#          --export=ALL,REPO=$PWD,ESTIMATOR=PSF_Width,CONDITION=FAB,TOTAL_TIME=2.0,TASKS="2 3 4 5 6 7 8 9",PURPOSE=development,RUN_SUFFIX=<commit>,CODE_COMMIT=<commit> \
#          Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Direct_Estimator.sh
# The job name mirrors the run folder the utility writes
# (<alias>_<CONDITION>_<timing>_Direct_<ESTIMATOR>_<DEV|VERDICT>[_<RUN_SUFFIX>]).
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Estimator
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48   # one per physical core; JUWELS counts cores here and allocates the whole node (96 threads)
#SBATCH --time=06:00:00
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
# One thread per worker: the pool supplies the parallelism, and threaded BLAS inside every worker
# would oversubscribe the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SRM_AND_SBI_CODE_COMMIT="${CODE_COMMIT:-}"

case "${ESTIMATOR:-}" in
    PSF_Width|Flicker_Rate|Fluorescence_Loss|Flicker_Mismatch) ;;
    *) echo "FATAL: ESTIMATOR='${ESTIMATOR:-}' (use PSF_Width|Flicker_Rate|Fluorescence_Loss|Flicker_Mismatch)." >&2; exit 1;;
esac
TOTAL_TIME="${TOTAL_TIME:-2.0}"
WORKERS="${WORKERS:-48}"
# ABSOLUTE script path: hpc_local.env may cd elsewhere after this script's own cd.
PY="$REPO/Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_${ESTIMATOR}.py"

ARGS=( --total-time-seconds "$TOTAL_TIME" --workers "$WORKERS" )
if [ "$ESTIMATOR" != "Flicker_Mismatch" ]; then
    case "${CONDITION:-}" in FAB|INLB) ;; *) echo "FATAL: CONDITION='${CONDITION:-}' (use FAB|INLB)." >&2; exit 1;; esac
    [ -n "${TASKS:-}" ] || { echo "FATAL: TASKS is required for a tier run (e.g. TASKS=\"2 3\")." >&2; exit 1; }
    # shellcheck disable=SC2206
    case "${PURPOSE:-}" in development|verdict) ;; *) echo "FATAL: PURPOSE='${PURPOSE:-}' -- a tier run must declare its purpose (development|verdict)." >&2; exit 1;; esac
    ARGS+=( --condition "$CONDITION" --tasks $TASKS --expect-videos-per-task "${EXPECT:-1000}"
            --max-videos "${MAX_VIDEOS:-100000000}" --purpose "$PURPOSE" )
fi
[ -n "${RUN_SUFFIX:-}" ] && ARGS+=( --run-suffix "$RUN_SUFFIX" )

echo "=== Direct_${ESTIMATOR} | condition=${CONDITION:-n/a} time=${TOTAL_TIME}s tasks=${TASKS:-n/a} purpose=${PURPOSE:-n/a} workers=${WORKERS} suffix=${RUN_SUFFIX:-none} commit=${CODE_COMMIT:-undeclared} | node $(hostname) | $(date -u +%FT%TZ) ==="
echo "    version: $(grep -m1 '^version' "$REPO/pyproject.toml")"
# Exit status of the utilities: 0 nothing failed, 1 a FAIL verdict, 2 insufficient evidence only,
# 3 the implementation changed during the run (results invalid for acceptance). Python also exits 1
# on an uncaught exception and argparse 2 on an argument error or a refusal (an existing run folder,
# a task the purpose may not read), so 1 and 2 are not verdicts alone: the log above says which.
# The status is recorded rather than letting `set -e` cut the log short.
rc=0
python -u "$PY" "${ARGS[@]}" ${EXTRA:-} || rc=$?
echo "=== Direct_${ESTIMATOR} complete (rc=${rc}: 0 = no FAIL verdict; 1 = a FAIL verdict or an uncaught exception; 2 = insufficient evidence, or an argument error or refusal; 3 = implementation changed during the run, invalid for acceptance -- the log says which) $(date -u +%FT%TZ) ==="
exit "$rc"
