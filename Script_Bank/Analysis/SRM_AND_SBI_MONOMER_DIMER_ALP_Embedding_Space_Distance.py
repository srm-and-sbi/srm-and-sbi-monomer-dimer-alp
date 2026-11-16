"""Analysis entry point (biology workflow): experimental-versus-synthetic embedding distance.

ROLE. Ask whether the trained network places the experimental recordings where it places the synthetic
data it was trained on. The embedding is the only representation the posterior ever sees, so a window
the network places outside its synthetic cloud is a window on which every reaction-diffusion estimate
is an extrapolation -- confidently reported, but outside the region the network learned. This
analysis measures the difference two ways that fail differently: a kernel two-sample test (MMD with a
permutation null) and a classifier two-sample test (C2ST, held-out AUC with a unit-level permutation
null). Both are blocked by recording and repeated on one window per recording, where the permutation
null is well-founded for independent recordings. A per-window typicality says which windows lie
outside the synthetic cloud.

HOW TO READ IT FOR THIS WORKFLOW. A detected difference between the two populations is not by itself a
failure of the forward model: the synthetic set covers the whole training prior, and the recordings
may occupy a narrow part of it. The biology workflow holds the imaging parameters FIXED and infers the
reaction-diffusion parameters, so a window OUTSIDE the synthetic cloud is not primarily a statement
about the imaging. It carries structure the synthetic prior never generated: biological variation
outside the prior's reach, or an imaging mismatch that the frozen imaging vector cannot express.
Neither is fixed by training longer; the remedies are a wider prior or a recalibrated imaging vector.
(In the detector workflow the imaging IS the inference target, so a window outside the cloud points at
the imaging forward model or at the prior ranges of its parameters.)

WHERE TO RUN. GPU strongly preferred -- it embeds every experimental window and every synthetic EVAL
video through the trained network. It is a post-hoc, user-driven analysis in Script_Bank/Analysis,
NOT a canonical pipeline stage and never wired into the stage dispatcher.

WHAT IT NEEDS.
  - the trained biology estimator for the run's timing;
  - the staged experimental recordings (Experiment_<KIND>_Cell_<n>_<span>S_RAW.tif);
  - the synthetic EVAL video sets to serve as the reference distribution.

WHAT IT DOES NOT DO.
  - It does NOT recover parameters, and it does NOT say the estimates are wrong. It measures whether
    the network is being asked to work outside the region it was trained on.

Writes <data_bank>/<posit>/<alias>_<timing>_Embedding_Space_Distance/   (report + figures)

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Embedding_Space_Distance.py \
        --condition FAB --total-time-seconds 2.0 [--artifact-tag TAG] [--eval-tasks N] [--dry-run]
"""
import sys

from srm_and_sbi_monomer_dimer_alp.embedding_space_distance_runner import (
    build_parser, run_embedding_space_distance)
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow


def main(argv=None):
    parser = build_parser(
        "Experimental-versus-synthetic embedding distance (biology workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_embedding_space_distance(biology_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())
