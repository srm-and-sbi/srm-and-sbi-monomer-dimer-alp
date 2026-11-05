#!/bin/bash
cd /home/mars-fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp || exit 1
PY=/home/mars-fias/anaconda3/envs/SRM_AND_SBI_ENVY_V0/bin/python
S=/tmp/claude-1000/-home-mars-fias-Documents-Projects-RCL-Agent-Projects/db645050-aff7-47b3-8133-140fa02ba488/scratchpad/sel_build
n=0; while ! grep -q "FAB26 PILOT DONE" $S/render_fab26.log; do sleep 10; n=$((n+1)); [ $n -gt 180 ] && { echo "timeout waiting for renders"; exit 1; }; done
for L in SEL_R1 SEL_R2 SEL_R3; do
  echo "===== Spot_Count $L $(date -Is)"
  MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD $PY Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Spot_Count.py --total-time-seconds 2.0 --kind MET-FAB --cell 26 --experiment-span-seconds 20 --source-label $L --workers 4 2>&1 | grep -v Warning | tail -5
  echo "===== Tail_Structure $L $(date -Is)"
  MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD $PY Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Brightness_Tail_Structure.py --total-time-seconds 2.0 --kind MET-FAB --cell 26 --experiment-span-seconds 20 --source-label $L 2>&1 | grep -v Warning | tail -5
done
echo "===== FAB26 ANALYSIS DONE $(date -Is)"
