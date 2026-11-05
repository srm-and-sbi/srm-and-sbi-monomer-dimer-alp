#!/usr/bin/env bash
# sigma_pc-only arms (reviewer brief via the lead, 2026-11-03): mu_pc restored to the SEL values (Fab 105, InlB 152.9);
# sigma_pc raised alone on the saved higher-abundance renders (trajectory, dye counts, seed, lambda kept).
set -euo pipefail
cd /home/mars-fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp
PY=/home/mars-fias/anaconda3/envs/SRM_AND_SBI_ENVY_V0/bin/python
export MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD
EV=Script_Bank/Analysis/Evidence/SEL_Predictive_Pilot_2026-11-02
$PY $EV/brightness_arm.py --total-time-seconds 2 --kind MET-FAB --cell 26 --source-label SEL_N30000_SINGLE --set-imaging sigma_pc=0.8 --arm-label SIGPC08
$PY $EV/brightness_arm.py --total-time-seconds 2 --kind MET-INLB --cell 11 --source-label SEL_N20000_SINGLE --set-imaging sigma_pc=1.0 --arm-label SIGPC10
D=/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit
F=$D/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Posterior_Predictive_Video/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Declared_RDS_MET-FAB_Cell_26_20S_SEL_N30000_SINGLE
I=$D/SRM_AND_SBI_MONOMER_DIMER_ALP_INLB_2S_50FPS_Posterior_Predictive_Video/SRM_AND_SBI_MONOMER_DIMER_ALP_INLB_2S_50FPS_Declared_RDS_MET-INLB_Cell_11_20S_SEL_N20000_SINGLE
$PY $EV/brightness_histograms.py ${F}_Synthetic_Video.npz ${F}_SIGPC08_Synthetic_Video.npz ${F}_SIGPC08_Brightness_Histograms
$PY $EV/brightness_histograms.py ${I}_Synthetic_Video.npz ${I}_SIGPC10_Synthetic_Video.npz ${I}_SIGPC10_Brightness_Histograms
echo SIGMA_ARMS_DONE
