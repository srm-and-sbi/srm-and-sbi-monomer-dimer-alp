#!/bin/bash
cd /home/mars-fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp || exit 1
PY=/home/mars-fias/anaconda3/envs/SRM_AND_SBI_ENVY_V0/bin/python
k=0
for seed in 20261102 20261103 20261104; do
  k=$((k+1))
  echo "===== FAB cell 26 realization R$k seed $seed start $(date -Is)"
  MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD $PY Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.py \
    --total-time-seconds 2.0 --kind MET-FAB --cell 26 --nuisance-tag SEL \
    --declared-rds Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Declared_RDS_FAB.toml \
    --run-label SEL_R$k --seed $seed
  echo "===== R$k rc=$? end $(date -Is)"
done
echo "===== FAB26 PILOT DONE"
