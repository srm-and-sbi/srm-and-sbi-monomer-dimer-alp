"""ILLUSTRATIVE build of an ``empirical_representatives`` Nuisance_DLI from the Fab PILOT2CELLS candidate archive.

Evidence driver, not a production step: it writes the spec, the artifact and the report into this Evidence
folder (``example_build/``), never into the data bank, and its component weights are an EXAMPLE of the
interface, not a scientific selection. It calls the analysis script's own functions (the build route and the
report writer) so the example exercises the production code path.
"""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MACHINE_PROFILE", "mars_pc")
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli          # noqa: E402
from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det       # noqa: E402
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg          # noqa: E402

spec_path = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py"
spec = importlib.util.spec_from_file_location("nuisance_dli_analysis", spec_path)
analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)

out = Path(__file__).resolve().parent / "example_build"
if out.exists():
    raise SystemExit(f"refusing to overwrite the existing example build: {out}")
out.mkdir()
archive_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit/"
    "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment_PILOT2CELLSR1/aggregation_candidates.npz")
archive = agg.load_candidate_archive(archive_path)
R = analysis._resolve(2.0, "FAB")
ref, _ = analysis._reference_physical(R)
target = analysis._target(SimpleNamespace(condition="FAB"), R)      # the binding every build states explicitly
tpl = out / "EXAMPLE_Nuisance_DLI_Spec_template.toml"
ndli.emit_representatives_template(tpl, archive, reference=ref, target=target,
                                   provenance={"timing_label": R["timing_label"], "condition": "FAB", "nuisance_tag": "EXAMPLE",
                                               "choice": "empirical_representatives", "candidate_archive": archive["path"],
                                               "candidate_archive_sha256": archive["sha256"], "note": "illustrative; not a selection"})
# the filled example: two components with EXAMPLE weights (no scientific decision), no override
filled = out / "EXAMPLE_Nuisance_DLI_Spec.toml"
filled.write_text(f'''# ILLUSTRATIVE spec (Evidence only): the weights below are an EXAMPLE of the interface, not a selection.
[block]
posterior_sample_pool_choice = "empirical_representatives"
candidate_archive = "{archive['path']}"
candidate_archive_sha256 = "{archive['sha256']}"
rationale = "Illustration of the interface on the two-cell pilot archive: the per-cell representatives over all ten windows beside the global representative; weights are an example, not a decision."

[[component]]
source_experiment = "{archive['manifest']['product_label']}"
condition = "FAB"
window_set = "all"
grouping = "per_cell"
statistic = "sgm"
weight = 0.75

[[component]]
source_experiment = "{archive['manifest']['product_label']}"
condition = "FAB"
window_set = "all"
grouping = "global"
statistic = "sgm"
weight = 0.25
''')
spec_dict = ndli.load_spec(filled, R["imaging_keys"], R["plo"], R["phi"], table=det.DETECTOR_PARAMETERIZATION)
nu = ndli.build_nuisance_dli(spec_dict, R["imaging_keys"], R["plo"], R["phi"], table=det.DETECTOR_PARAMETERIZATION,
                             archive=archive, spec_path=filled, target=target)
art = out / "EXAMPLE_Nuisance_DLI.npz"
nu.flush(art)
report_dir = analysis._write_nuisance_report(nu, R, art, archive=archive)
back = ndli.NuisanceDLI.load(art)
print(f"example artifact: {art} ({back.samples.shape[0]} vectors; probabilities {back.probabilities.round(4).tolist()}; "
      f"pool_mode {back.pool_mode}; components {len(back.selection['components'])}); report {report_dir}")
print("identity:", {k: v for k, v in back.identity(art).items() if k != 'selection'})
