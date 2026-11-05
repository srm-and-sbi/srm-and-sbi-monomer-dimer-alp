#!/usr/bin/env bash
# Retained-draw collection of the cells of one condition that have no validated collection yet (2026-11-05).
# One companion invocation per cell (output-only label COHORTCELL<n>, seed SEED_BASE + n), restartable: a cell whose
# collection folder already exists and validates is skipped; an existing invalid folder stops the stream (never overwritten).
# Streams: STREAM of N_STREAMS takes every N_STREAMS-th cell of the missing list, so several streams share one GPU.
#
#   MACHINE_PROFILE=<p> PYTHON=<python> bash collect_cohort.sh <FAB|INLB> <TAG> <SEED_BASE> <STREAM> <N_STREAMS> <LOG_DIR>
#
# Run from the repository root (PYTHONPATH=$PWD is set here). Logs: <LOG_DIR>/draws_<COND>_cell<n>.log and .time.
set -euo pipefail
COND=$1; TAG=$2; SEED_BASE=$3; STREAM=${4:-0}; N_STREAMS=${5:-1}; LOG_DIR=$6
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMPANION=Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Draws.py
PYTHON=${PYTHON:-python}
export PYTHONPATH=$PWD
mkdir -p "$LOG_DIR"
missing=$($PYTHON "$HERE/missing_cells.py" --condition "$COND" --artifact-tag "$TAG" 2>>"$LOG_DIR/missing_${COND}_stream${STREAM}.log")
IFS=',' read -r -a cells <<< "$missing"
posit=$($PYTHON - <<'PY'
import os, sys; sys.path.insert(0, os.getcwd())
from pathlib import Path
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow
print(Path(PARAMETERS.machine.data_bank_root) / detector_workflow().paths.posit_subdir)
PY
)
echo "[$(date -u +%FT%TZ)] $COND stream $STREAM/$N_STREAMS: missing cells ${missing:-none}; seed base $SEED_BASE; posit $posit" | tee -a "$LOG_DIR/stream_${COND}_${STREAM}.log"
i=0
for cell in "${cells[@]}"; do
  [ -z "$cell" ] && continue
  if [ $((i % N_STREAMS)) -ne "$STREAM" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  label="COHORTCELL${cell}"
  dest="$posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_${COND}_2S_50FPS_${TAG}_Window_Posterior_Draws_${label}"
  if [ -d "$dest" ]; then
    if $PYTHON "$HERE/validate_collection.py" --condition "$COND" --artifact-tag "$TAG" "$dest/$(basename "$dest").npz" >>"$LOG_DIR/stream_${COND}_${STREAM}.log" 2>&1; then
      echo "[$(date -u +%FT%TZ)] cell $cell: existing valid collection, skipped" | tee -a "$LOG_DIR/stream_${COND}_${STREAM}.log"; continue
    else
      echo "[$(date -u +%FT%TZ)] cell $cell: existing folder does NOT validate; stopping this stream (nothing overwritten): $dest" | tee -a "$LOG_DIR/stream_${COND}_${STREAM}.log"; exit 2
    fi
  fi
  seed=$((SEED_BASE + cell))
  echo "[$(date -u +%FT%TZ)] cell $cell: collecting (seed $seed, label $label)" | tee -a "$LOG_DIR/stream_${COND}_${STREAM}.log"
  /usr/bin/time -v -o "$LOG_DIR/draws_${COND}_cell${cell}.time" \
    $PYTHON "$COMPANION" --total-time-seconds 2.0 --condition "$COND" --artifact-tag "$TAG" --cells "$cell" \
      --posterior-samples 10000 --seed "$seed" --collection-label "$label" > "$LOG_DIR/draws_${COND}_cell${cell}.log" 2>&1 \
    || { echo "[$(date -u +%FT%TZ)] cell $cell: companion FAILED (see draws_${COND}_cell${cell}.log); stopping this stream" | tee -a "$LOG_DIR/stream_${COND}_${STREAM}.log"; exit 3; }
  $PYTHON "$HERE/validate_collection.py" --condition "$COND" --artifact-tag "$TAG" "$dest/$(basename "$dest").npz" >>"$LOG_DIR/stream_${COND}_${STREAM}.log" 2>&1 \
    || { echo "[$(date -u +%FT%TZ)] cell $cell: collection written but FAILED validation; stopping this stream" | tee -a "$LOG_DIR/stream_${COND}_${STREAM}.log"; exit 4; }
  echo "[$(date -u +%FT%TZ)] cell $cell: done ($(grep 'Elapsed' "$LOG_DIR/draws_${COND}_cell${cell}.time" | sed 's/.*): //'))" | tee -a "$LOG_DIR/stream_${COND}_${STREAM}.log"
done
echo "[$(date -u +%FT%TZ)] $COND stream $STREAM/$N_STREAMS: finished" | tee -a "$LOG_DIR/stream_${COND}_${STREAM}.log"
