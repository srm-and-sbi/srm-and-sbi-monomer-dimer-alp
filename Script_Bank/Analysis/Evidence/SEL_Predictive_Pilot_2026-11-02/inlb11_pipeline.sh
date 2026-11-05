#!/bin/bash
# InlB pilot: stage the five candidate recordings from the PC's complete MET movie bank (read-only source) under the
# sixty-recording catalog rule validated on the ten Fab working files (coverslip-and-cell order: CS3 0-19, CS3_160205
# 20-22 = cells 07/08/16, CS4 23-42, CS5 43-59), then render InlB cell 11 three times under SEL and analyze.
set -u
cd /home/mars-fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp || exit 1
PY=/home/mars-fias/anaconda3/envs/SRM_AND_SBI_ENVY_V0/bin/python
E=Script_Bank/Analysis/Evidence/SEL_Predictive_Pilot_2026-11-02
D=/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Experiment/SPT_Data_MET_FAB_INLB_S-BSST712
B=/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Projects/SRM_AND_SBI/SPT_Movies_Complete_Data_Bank/MET/InlB
map_inlb() { i=$1; if [ $i -lt 20 ]; then cs=CS3; n=$((i+1)); elif [ $i -lt 23 ]; then cs=CS3_160205; case $i in 20) n=7;; 21) n=8;; 22) n=16;; esac; elif [ $i -lt 43 ]; then cs=CS4; n=$((i-22)); else cs=CS5; n=$((i-42)); fi; printf "%s/%s/tifs/InlB_%s_cell%02d.tif" $B $cs $cs $n; }
NOTE=$D/INLB_STAGED_FROM_PC_BANK_2026-11-03.md
{ echo "# InlB recordings staged on 2026-11-03 from the PC's complete MET movie bank"; echo; echo "Source (read-only): $B/<coverslip>/tifs/. Index rule = the sixty-recording catalog's coverslip-and-cell order"; echo "(CS3 cells 01-20 -> 0-19; CS3_160205 cells 07, 08, 16 -> 20-22; CS4 cells 01-20 -> 23-42; CS5 cells 01-17 -> 43-59),"; echo "validated on 2026-11-03 against the ten Fab working files of this folder (10 of 10 byte-identical to the bank's"; echo "Fab_CS2/CS5/CS6 movies under the same rule). To be cross-checked against Catalog_Note_Experiment.tsv on JUWELS."; echo; echo "| working file | source movie | SHA-256 |"; echo "|---|---|---|"; } > $NOTE
for i in 53 43 11 42 22; do
  src=$(map_inlb $i); dst=$D/Experiment_INLB_Cell_${i}_20S_RAW.tif
  [ -e "$dst" ] && { echo "exists: $dst"; } || cp "$src" "$dst" || exit 1
  a=$(sha256sum "$src" | cut -c1-64); b=$(sha256sum "$dst" | cut -c1-64)
  [ "$a" = "$b" ] || { echo "CHECKSUM MISMATCH cell $i"; exit 1; }
  echo "| $(basename $dst) | $(basename $src) | $a |" >> $NOTE; echo "staged cell $i <- $(basename $src) ${a:0:16}"
done
echo "===== INLB STAGING VERIFIED $(date -Is)"
k=0
for seed in 20261102 20261103 20261104; do
  k=$((k+1)); echo "===== INLB cell 11 realization R$k seed $seed start $(date -Is)"
  MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD $PY Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.py \
    --total-time-seconds 2.0 --kind MET-INLB --cell 11 --nuisance-tag SEL \
    --declared-rds Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Declared_RDS_INLB.toml \
    --run-label SEL_R$k --seed $seed 2>&1 | tr '\r' '\n' | grep -v "%|"
  echo "===== R$k rc=${PIPESTATUS[0]} end $(date -Is)"
done
for L in SEL_R1 SEL_R2 SEL_R3; do
  echo "===== Spot_Count $L $(date -Is)"
  MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD $PY Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Spot_Count.py --total-time-seconds 2.0 --kind MET-INLB --cell 11 --experiment-span-seconds 20 --source-label $L --workers 4 2>&1 | grep -v Warning | tail -5
  echo "===== Tail_Structure $L $(date -Is)"
  MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD $PY Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Brightness_Tail_Structure.py --total-time-seconds 2.0 --kind MET-INLB --cell 11 --experiment-span-seconds 20 --source-label $L 2>&1 | grep -v Warning | tail -5
done
echo "===== width/background $(date -Is)"
MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD $PY $E/pilot_width_background.py MET-INLB 11 SEL_R1 SEL_R2 SEL_R3 2>&1 | grep -v Warning > $E/pilot_width_background_INLB11.out
echo "===== INLB11 PIPELINE DONE $(date -Is)"
