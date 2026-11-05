#!/usr/bin/env bash
# Pull the rcl01 cohort collections and their logs back to the PC (the source of truth), validate the union of all
# collections per condition against the adopted product, and regenerate both adopted Experiment reports on the CPU
# with every retained collection under the review label COHORT60PLOTS (2026-11-05). Run from the repository root on
# the PC: bash Script_Bank/Analysis/Evidence/Experiment_Aggregation_Cohort_2026-11-05/pull_and_regenerate.sh [pull|validate|regenerate]
set -euo pipefail
STEP=${1:-all}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=/home/mars-fias/anaconda3/envs/SRM_AND_SBI_ENVY_V0/bin/python
export MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD
PC_POSIT=/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit
RC_POSIT=/media/rcl_fias/RCL_FIAS_C/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit
RC_LOGS=/home/rcl_fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp/Script_Bank/Analysis/Evidence/Experiment_Aggregation_Cohort_2026-11-05/rcl01_logs
HOST=rcl_fias@10.83.255.103
LABEL=COHORT60PLOTS
if [ "$STEP" = all ] || [ "$STEP" = pull ]; then
  # new folders only: an existing PC folder is never overwritten (the pilot collections stay as they are)
  rsync -a --ignore-existing --itemize-changes --include='SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_*_Window_Posterior_Draws_COHORTCELL*/***' --exclude='*' \
      "$HOST:$RC_POSIT/" "$PC_POSIT/" | grep -c '^>f' | sed 's/^/files pulled: /'
  rsync -a --itemize-changes "$HOST:$RC_LOGS/" "$HERE/rcl01_logs/" | grep -c '^>f' | sed 's/^/log files pulled: /'
fi
if [ "$STEP" = all ] || [ "$STEP" = validate ]; then
  for pair in "FAB C128V42A" "INLB C128V42AP3LR32"; do
    set -- $pair
    $PY "$HERE/validate_collection.py" --condition "$1" --artifact-tag "$2" --union | tee "$HERE/validate_union_$1.log" | tail -1
  done
fi
if [ "$STEP" = all ] || [ "$STEP" = regenerate ]; then
  for pair in "FAB C128V42A" "INLB C128V42AP3LR32"; do
    set -- $pair
    prefix="$PC_POSIT/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_${1}_2S_50FPS_${2}"
    sources=$(ls -d ${prefix}_Window_Posterior_Draws*/ | sed 's#/$##' | while read d; do echo "$d/$(basename $d).npz"; done | paste -sd, -)
    echo "$1: $(echo "$sources" | tr ',' '\n' | wc -l) collections"
    /usr/bin/time -v -o "$HERE/regenerate_${1}_${LABEL}.time" $PY Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment.py \
        --condition "$1" --total-time-seconds 2.0 --artifact-tag "$2" --pool-mode unrestricted \
        --regenerate-report --review-label "$LABEL" --cloud-sources "$sources" > "$HERE/regenerate_${1}_${LABEL}.log" 2>&1 \
      && echo "$1: regenerated ($(grep Elapsed "$HERE/regenerate_${1}_${LABEL}.time" | sed 's/.*): //'), $(grep 'Maximum resident' "$HERE/regenerate_${1}_${LABEL}.time" | awk '{printf "%.2f GB", $NF/1048576}'))" \
      || { echo "$1: regeneration FAILED (see regenerate_${1}_${LABEL}.log)"; exit 1; }
  done
fi
