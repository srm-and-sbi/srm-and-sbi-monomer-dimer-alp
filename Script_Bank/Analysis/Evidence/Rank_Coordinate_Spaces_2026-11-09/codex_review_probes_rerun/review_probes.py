"""Read-only production inspection and isolated fixtures for the 0.1.48 review.

Run from the successor repository with its environment; no production outputs are written.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
REPO = WORKSPACE / "srm-and-sbi" / "srm-and-sbi-monomer-dimer-alp"
sys.path[:0] = [str(REPO), str(REPO / "tests")]
import test_experiment_compatibility as fixture
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg
from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli


def availability_probe():
    ids = [(0, c, w) for c in range(3) for w in range(4)]
    clouds = []
    for _, c, w in ids:
        clouds.append(np.column_stack([
            np.full(51, -0.8 + 0.05 * w + 0.001 * c),
            np.linspace(0.0, 1.0, 51), np.full(51, 0.5),
        ]))
    coll = fixture._collection(np.asarray(clouds), ids)
    with tempfile.TemporaryDirectory() as tmp:
        arc, *_ = fixture._archive_with_spaces(Path(tmp), coll, {"all": None}, {"a_only": ("a",)})
        comp = agg.archive_component(arc, kind="FAB", window_set="all", construction="per_cell_window")
        advertised = ndli.component_weighting_availability(arc)[("FAB", "all", "per_cell_window")]
        result = {"availability": advertised, "spaces": {}}
        for coords in (None, ["a"]):
            label = "all" if coords is None else "a_only"
            try:
                p, _, _ = ndli.member_probabilities("posterior_rank_median", 1.0, comp, rank_coordinates=coords)
                result["spaces"][label] = {"accepted": True, "total": float(p.sum())}
            except ValueError as exc:
                result["spaces"][label] = {"accepted": False, "reason": str(exc)}
        return result


def hpc_probe():
    import test_hpc_submit_forwarding as hf
    result = {}
    value = "spatial_brightness=mu_r,sigma_r,mu_pc,sigma_pc"
    for name in hf.LAUNCHERS:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            calls = root / "calls.jsonl"
            hf._fake(bin_dir / "python", 'if [ "$1" = "-c" ]; then echo fake-id; exit 0; fi\n'
                     'python3 - "$@" <<\'PY\'\nimport json,sys\n'
                     f'with open({str(calls)!r}, "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nPY\n')
            hf._fake(bin_dir / "srun", 'while [ "${1#--}" != "$1" ]; do shift; done; "$@"\n')
            conda = hf._fake(root / "conda.sh", "conda() { :; }\n")
            env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", HPC_ENV="/dev/null",
                       CONDA_SETUP=str(conda), MACHINE_PROFILE="mars_pc", REPO=str(REPO), CONDITION="FAB",
                       COMPATIBILITY_COORDINATES=value, SRM_AND_SBI_GPUS="2", SLURM_NNODES="1", SLURM_CPUS_ON_NODE="8")
            proc = subprocess.run(["bash", str(REPO / "Script_Bank/HPC" / name)], env=env, cwd=REPO,
                                  text=True, capture_output=True, timeout=30)
            records = [json.loads(s) for s in calls.read_text().splitlines()] if calls.exists() else []
            result[name] = {"exit_code": proc.returncode, "calls": records,
                            "rank_option_forwarded": bool(records) and all("--compatibility-coordinates" in x for x in records)}
    return result


def cohort_checks():
    evidence = REPO / "Script_Bank/Analysis/Evidence/Rank_Coordinate_Spaces_2026-11-09"
    output = {}
    for path in sorted((evidence / "review_copies").glob("*/aggregation_candidates.npz")):
        arc = agg.load_candidate_archive(path)
        cond = arc["manifest"]["target"]["condition"]
        a = arc["arrays"]
        comp = agg.archive_component(arc, kind=cond, window_set="all", construction="per_cell_window")
        stats = {"groups": len(a["construction"]), "spaces": {}}
        for coords in (None, ["mu_r", "sigma_r", "mu_pc", "sigma_pc"]):
            label = "all" if coords is None else "spatial_brightness"
            p, _, s = ndli.member_probabilities("posterior_rank_median", 1.0, comp, rank_coordinates=coords)
            ids = comp["identities"]
            cells = sorted({x["cell"] for x in ids})
            windows = sorted({x["chunk"] for x in ids})
            mass = [float(sum(p[i] for i, x in enumerate(ids) if x["cell"] == c)) for c in cells]
            stats["spaces"][label] = {
                "n": len(p), "zero": int((p == 0).sum()), "effective_rows": float(1 / (p*p).sum()),
                "cell_mass_range": [min(mass), max(mass)],
                "window_mass": {str(w): float(sum(p[i] for i, x in enumerate(ids) if x["chunk"] == w)) for w in windows},
                "zero_by_window": {str(w): sum(int(p[i] == 0) for i, x in enumerate(ids) if x["chunk"] == w) for w in windows},
                "scores": s,
            }
        six = stats["spaces"]["all"].pop("scores")
        four = stats["spaces"]["spatial_brightness"].pop("scores")
        stats["four_score_higher_than_six"] = int((four > six).sum())
        stats["four_score_lower_than_six"] = int((four < six).sum())
        stats["four_score_equal_to_six"] = int((four == six).sum())
        stats["proposal_overrides"] = {}
        for space in ("all", "spatial_brightness"):
            p = next((HERE.parent / f"{cond}_{space}").glob("*Nuisance_DLI.npz"))
            nu = ndli.NuisanceDLI.load(p)
            stats["proposal_overrides"][space] = nu.selection["overrides"]
        output[cond] = stats
    return output


if __name__ == "__main__":
    data = {"availability": availability_probe(), "hpc_forwarding": hpc_probe(), "cohorts": cohort_checks()}
    (HERE / "review_probes.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))
