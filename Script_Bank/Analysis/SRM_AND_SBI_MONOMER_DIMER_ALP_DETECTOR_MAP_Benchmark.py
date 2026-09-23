"""Analysis entry point (detector workflow): the MAP optimization benchmark.

ROLE. Resolve the MAP optimizer's settings on the real trained flow before any analysis is
regenerated. For a fixed checkpoint and a fixed list of synthetic EVAL recordings, it compares the
best starting candidate, the production optimizer and new-loop settings (absolute versus
scale-aware steps, production versus extended budget and patience, a bracket of learning rates),
all from identical starting candidates, on two independent candidate pools per recording, against
a polished reference optimum. See ``srm_and_sbi_monomer_dimer_alp.map_benchmark``.

WHERE TO RUN. Where the checkpoint and the EVAL tier are: a GPU node for a large subset, sharded
across GPUs (one rank per GPU; ``--merge`` afterwards); any machine for a small ``--pilot`` with a
scratch data bank. A validation utility, NOT a canonical stage, never wired into the dispatcher.

Reads  <data_bank>/<posit>/<alias>_<timing>[_<TAG>]_Estimator.npz and the EVAL theta/video stores
Writes <data_bank>/<posit>/<alias>_<timing>[_<TAG>]_MAP_Benchmark/ (artifact, report.md)

Usage:
    MACHINE_PROFILE=<p> python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_MAP_Benchmark.py \\
        --condition FAB --total-time-seconds 2.0 [--tasks 0 1] [--pilot N] [--merge] [--dry-run]
"""
import sys

from srm_and_sbi_monomer_dimer_alp.map_benchmark_runner import build_parser, run_map_benchmark
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow


def main(argv=None):
    parser = build_parser("MAP optimization benchmark on the real trained flow (detector workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_map_benchmark(detector_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())
