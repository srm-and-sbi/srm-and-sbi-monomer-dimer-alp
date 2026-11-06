"""Illustrative rank-weighted constructor build, written INTO this evidence folder (never a production name, never the
Posit directory): one component, MET-FAB, all ten windows, one SGM per cell, member_weighting = posterior_rank_median,
read from the regenerated COHORT60RANKS archive. Exercises the companion's dry-run console path and its build/report path
through the companion's own functions. Not a production selection.

Usage (repository root):
    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python Script_Bank/Analysis/Evidence/Experiment_Compatibility_2026-11-06/illustrative_build.py \
        --archive <FAB COHORT60RANKS>/aggregation_candidates.npz --out <folder>
"""
import argparse
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "Script_Bank" / "Analysis"))
import SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI as companion      # noqa: E402
from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli     # noqa: E402
from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det  # noqa: E402

SPEC = """# Illustrative empirical_representatives spec for the 0.1.47 implementation checkpoint: NOT a production selection.
[block]
posterior_sample_pool_choice = "empirical_representatives"
candidate_archive = "{archive}"
candidate_archive_sha256 = "{sha}"
rationale = "Illustrative interface example: one SGM per cell over all ten 2 s windows with consensus (median-rank) member weighting, to show the dry-run, the report and the exports. No acquisition period, construction, weighting or treatment is selected by this file."

[[component]]
source_experiment = "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A"
condition = "FAB"
window_set = "all"
grouping = "per_cell"
statistic = "sgm"
weight = 1.0
member_weighting = "posterior_rank_median"
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    spec = args.out / "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_ILLUSTRATIVERANK_Nuisance_DLI_Spec.toml"
    spec.write_text(SPEC.format(archive=str(args.archive), sha=sha))
    art = args.out / "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_ILLUSTRATIVERANK_Nuisance_DLI.npz"
    if art.exists():
        raise SystemExit(f"refusing to overwrite {art}")
    R = companion._resolve(2.0, "FAB")
    spec_dict = ndli.load_spec(spec, R["imaging_keys"], R["plo"], R["phi"], table=det.DETECTOR_PARAMETERIZATION)
    block = spec_dict["block"]
    print("==== dry run (the companion's own console path) ====", flush=True)
    companion._build_representatives(SimpleNamespace(condition="FAB", dry_run=True), R, block, spec_dict, art, spec)
    print("==== build ====", flush=True)
    companion._build_representatives(SimpleNamespace(condition="FAB", dry_run=False), R, block, spec_dict, art, spec)
    nu = ndli.NuisanceDLI.load(art)
    sel = nu.selection
    print(f"stored rows {nu.samples.shape[0]}; zero rows {int((nu.probabilities == 0).sum())}; effective rows {sel['effective_rows']:.4f}; "
          f"baseline effective rows {sel['equal_member_baseline']['effective_rows']:.4f}; max probability {nu.probabilities.max():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
