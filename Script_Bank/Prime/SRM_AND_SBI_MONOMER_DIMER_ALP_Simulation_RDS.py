"""Entry-point script: generate one condition's ReaDDy reaction-diffusion trajectory tier.

Samples the eleven reaction-diffusion parameters of the separated stoichiometry-mobility
model (two molecular species A monomer / B dimer x three mobility modes f / s / i = six
particle types; the reaction channels generated from the model blocks and the run's
condition) from the box-uniform prior in the estimator space (log10 for log rows, the
value itself for the linear initial dimer fraction), maps them to physical values with
``parameterization.to_physical``, runs ReaDDy simulations for each (task, simulation)
pair, and saves the resulting particle trajectories (with their reaction records) as .h5
files and the parameter samples as a .zarr (compressed) or .npy (uncompressed) theta set.

This is the ONE RDS entry point of the pipeline, run once PER CONDITION (``--condition``):
the association ratio is a declared per-condition constant (MET-FAB 0: no association
channel, eleven channels; MET-INLB 1: the reference convention, seventeen channels), so the
two conditions' trajectories differ. Within a condition the tier is shared by both
workflows, which re-image it at their DLI stages (the biology under the calibrated imaging
nuisance, the detector with the imaging drawn as its inference target) under the
condition's labeling law, so it carries the sibling alias plus the condition token and no
workflow qualifier. To the biology workflow the ``Theta_Set`` is the learnable label; to
the detector workflow the same file is the record of the reaction-diffusion nuisance it
marginalizes. The engine is ``srm_and_sbi_monomer_dimer_alp.simulation_rds_runner.run_rds``;
this shim builds the biology ``WorkflowConfig`` (unqualified alias) and hands it over.

Outputs (the ``{timing_label}`` token, e.g. ``2S_50FPS``, is rendered from
``PARAMETERS.simulation.timing.label`` to namespace files by duration + fps):

    <data_bank>/<video_subdir>/<trajectory_repo>/<sibling_alias>_<CONDITION>_{timing_label}_TASK_{n}_{split}/
        <sibling_alias>_<CONDITION>_{timing_label}_TASK_{n}_SIM_{m}_{split}.h5     -- per-simulation trajectory
    <data_bank>/<theta_subdir>/
        <sibling_alias>_<CONDITION>_{timing_label}_Theta_Set_TASK_{n}_{split}.zarr -- per-task theta sample set

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py \\
        --condition FAB --total-time-seconds 2.0 --tasks 2 --task-simulations 5 --seed None
    (repeat with --condition INLB for the other condition's tier)
    (then render per workflow and condition with SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py
    and SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Simulation_DLI.py, or drive everything with
    SRM_AND_SBI_MONOMER_DIMER_ALP_Generate_Datasets.py)

Diagnostics:
    --probe logs the process resource limits (RLIMIT_NPROC / RLIMIT_NOFILE) at
    startup and a per-simulation line (thread count, open file descriptors,
    resident memory). Logging only; it does not change generation behavior. It
    is the instrumentation used to diagnose the per-simulation ReaDDy-kernel
    resource leak fixed in the shared runner.
"""

import sys

from srm_and_sbi_monomer_dimer_alp.simulation_rds_runner import build_rds_parser, run_rds
from srm_and_sbi_monomer_dimer_alp.utils import console_log_context
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow


if __name__ == "__main__":
    cli_args = build_rds_parser().parse_args(sys.argv[1:])
    cfg = biology_workflow()
    with console_log_context(cli_args, "RDS", paths=cfg.console_log_paths):
        run_rds(cfg, cli_args)
