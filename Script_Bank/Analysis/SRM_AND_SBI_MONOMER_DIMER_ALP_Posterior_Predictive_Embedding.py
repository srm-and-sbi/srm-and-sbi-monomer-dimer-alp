"""Analysis entry point (biology workflow): posterior-predictive check in the embedding space.

For every experimental window: videos are simulated at joint draws of the window's reaction-diffusion
posterior (imaging as the biology training data have it: the condition's Nuisance_DLI artifact and the
camera box), embedded through the trained biology network, and the window is ranked among them. A
window inside its own predictive cloud is one the model reproduces at the parameters inferred for it.

The steps, the options, the outputs and the reading are documented once, for both workflows, in

    SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Embedding.md

Usage (three invocations; see the companion):
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Embedding.py \
        --step plan|simulate|analyze --condition FAB --total-time-seconds 2.0 [--artifact-tag TAG] [--dry-run]
"""
import sys

from srm_and_sbi_monomer_dimer_alp.posterior_predictive_embedding_runner import (
    build_parser, run_posterior_predictive_embedding)
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow


def main(argv=None):
    parser = build_parser("Posterior-predictive check in the embedding space (biology workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_posterior_predictive_embedding(biology_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())
