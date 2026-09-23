"""Analysis entry point (detector workflow): point-estimate validation, phase 1 (the median).

ROLE. Establish the per-observation marginal median as the reference point estimate before the SGM
and MAP phases are judged against it. For a fixed checkpoint, a fixed list of synthetic EVAL
recordings and bounded sampling, it draws five independent repeats at the production draw count and
one larger reference run per recording, stores every draw with its quantiles, and reports whether
the production calculation matches an independent recomputation and whether its Monte Carlo
variability is small relative to the smallest accuracy difference the program will interpret.

WHERE TO RUN. Where the checkpoint and the EVAL tier are: a GPU node for the full list; any machine
for a ``--pilot`` with a scratch data bank (``--data-bank-root``) holding only the needed inputs. It
is a validation utility in Script_Bank/Analysis, NOT a canonical stage, and never wired into the
stage dispatcher.

WHAT IT DOES NOT DO. It does not assess calibration and cannot remove posterior bias; it validates
this checkpoint's summaries, and the observations are development data, not untouched adoption
evidence. It computes no SGM and no MAP: those phases follow once this one passes.

Reads  <data_bank>/<posit>/<alias>_<timing>[_<TAG>]_Estimator.npz
       <data_bank>/<theta>|<video>/<alias>_<timing>_{Theta,Video}_Set_TASK_<t>_EVAL.zarr
Writes <data_bank>/<posit>/<alias>_<timing>[_<TAG>]_Point_Estimate_Validation_Median/
       (the validation artifact, report.md, per_recording_exceedances.csv)

Usage:
    MACHINE_PROFILE=<p> python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Point_Estimate_Validation.py \\
        --condition FAB --total-time-seconds 2.0 [--pilot 10 --data-bank-root D --output-dir O]
        [--merge] [--dry-run]
"""
import sys

from srm_and_sbi_monomer_dimer_alp.point_estimate_validation_runner import (
    build_parser, run_point_estimate_validation)
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow


def main(argv=None):
    parser = build_parser("Point-estimate validation, phase 1: the marginal median as the "
                          "reference (detector workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_point_estimate_validation(detector_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())
