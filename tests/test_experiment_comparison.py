"""The Experiment comparison (Script_Bank/Analysis/..._DETECTOR_Experiment_Comparison.py) on synthetic,
schema-valid Experiment products: windows are aligned by identifier across products stored in different
orders; the location, prior-box counts and shares, the reach beyond the box, whole intervals beyond it,
widths, the typical and largest discrepancies of the three point estimates, the ranges of window and
per-recording medians, the movement between windows, the drift and the pairwise agreement reproduce what the
synthetic products were built to have; a constant series has an undefined correlation; the selected vector and the localization references
are carried beside the estimates; MAP stops are totaled from a job log; mismatched windows, window geometry or
pool mode, a checkpoint that is not the estimator artifact's weights, an artifact without one selected vector
and a log for an estimator not compared are refused; the dry run compares nothing and an existing record is
never overwritten. CPU only; no trained flow."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from srm_and_sbi_monomer_dimer_alp import artifact_schema as S                       # noqa: E402
from srm_and_sbi_monomer_dimer_alp.detector_nuisance_dli import NuisanceDLI, artifact_path  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming      # noqa: E402
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow, parameter_keys  # noqa: E402
import _product_fixtures as fx                                                        # noqa: E402


def _companion():
    path = REPO / "Script_Bank" / "Analysis" / "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Comparison.py"
    spec = importlib.util.spec_from_file_location("detector_experiment_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EC = _companion()
CFG = detector_workflow()
KEYS = parameter_keys(CFG)
LO = np.array(CFG.param_module.theta_lower_bound(), float)
HI = np.array(CFG.param_module.theta_upper_bound(), float)
D = len(KEYS)
CELLS, CHUNKS = 12, 10
N = CELLS * CHUNKS
I_SR, I_PC, I_PB, I_LAM = (KEYS.index(k) for k in ("sigma_r", "mu_pc", "prob_photo_bleach", "lambda_rate"))
OFFSETS = np.array([-1.6449, -0.6745, 0.0, 0.6745, 1.6449])     # calibrated normal quantiles at the canonical levels
WIDTH = {"baseline": 0.05, "CAP256": 0.05, "CAND": 0.02}
VECTOR = (LO + HI) / 2 + 0.01


def _paths_and_label(tag):
    paths = CFG.paths.with_condition("FAB")
    timing = RunTiming(total_time_seconds=2.0, frames=PARAMETERS.simulation.timing)
    return paths, paths.product_label(timing.label, tag)


def _stem(tag):
    paths, label = _paths_and_label(tag)
    return f"{paths.project_alias}_{label}"


def _write_experiment(posit, name, median, map_, sgm, width, ids, rng, **manifest_over):
    stem = _stem(None if name == "baseline" else name)
    quantiles = median[:, :, None] + OFFSETS[None, None, :] * width
    perm = rng.permutation(N)                                              # this product's own row order
    a = dict(map_estimate=map_[perm], posterior_quantiles=quantiles[perm], posterior_sgm=sgm[perm],
             scores=np.zeros(N), kind_index=ids[perm, 0].astype(np.int64), cell=ids[perm, 1].astype(np.int64),
             chunk=ids[perm, 2].astype(np.int64), kinds=np.asarray(["FAB"]))
    over = dict(parameter_keys=KEYS, sgm_scale=[1.0] * D, pool_mode="unrestricted", job_id=f"job-{name}")
    over.update(manifest_over)
    m = fx.valid_manifest("experiment", N, kinds=["FAB"], **over)
    a[S.MANIFEST_KEY] = S.encode_manifest(m)
    folder = posit / f"{stem}_MAP_Experiment"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{stem}_MAP_Experiment.npz", **a)


def _write_estimator(posit, name, weights):
    manifest = dict(weights_sha256=weights, parameter_keys=KEYS, rebuild_spec={}, metadata={})
    np.savez(posit / f"{_stem(name)}_Estimator.npz", manifest=np.asarray(json.dumps(manifest)), prior_low=LO, prior_high=HI)


def _write_vector(posit, choice="selection_user"):
    paths, label = _paths_and_label(None)
    path = artifact_path(posit, paths.project_alias, label, "REF")
    if choice == "selection_user":
        nd = NuisanceDLI.from_selection(KEYS, VECTOR, {"kind": "user", "values": {}}, prior_low=LO, prior_high=HI)
    else:
        nd = NuisanceDLI.from_box(KEYS, LO, HI, prior_low=LO, prior_high=HI)
    nd.flush(path)
    return path


def synthetic_posit(root, rng, *, cand_over=None):
    """Three products on the same 12 recordings x 10 windows. baseline: a per-recording level plus small noise,
    the three estimates nearly equal. CAP256 (base): baseline's estimates with mu_pc shifted by exactly +0.1 dex.
    CAND: bleaching drifting by +0.05 dex per window in every recording; mu_pc below the prior floor in the last
    five windows of every recording; lambda_rate constant; its sigma_r MAP three intervals' half-widths above its
    median; its bleaching MAP stepping by +0.5 dex at window 5 of recording 0."""
    posit = Path(root) / "Posit"
    posit.mkdir()
    cells = np.repeat(np.arange(CELLS), CHUNKS)
    chunks = np.tile(np.arange(CHUNKS), CELLS)
    ids = np.stack([np.zeros(N, int), cells, chunks], axis=1)
    level = LO + (0.25 + 0.5 * rng.random((CELLS, D))) * (HI - LO)
    base = level[cells] + rng.normal(0, 0.01, (N, D))
    est = {}
    est["baseline"] = dict(median=base, map_=base.copy(), sgm=base + 0.002)
    cap = base.copy()
    cap[:, I_PC] += 0.1
    est["CAP256"] = dict(median=cap, map_=cap.copy(), sgm=cap + 0.002)
    cand = base.copy()
    cand[:, I_PB] = level[cells, I_PB] + 0.05 * chunks
    cand[:, I_PC] = np.where(chunks >= 5, LO[I_PC] - 0.05, base[:, I_PC])
    cand[:, I_LAM] = 0.5
    cmap = cand.copy()
    cmap[:, I_SR] += 3 * WIDTH["CAND"]
    cmap[(cells == 0) & (chunks >= 5), I_PB] += 0.5
    est["CAND"] = dict(median=cand, map_=cmap, sgm=cand + 0.002)
    for name in ("baseline", "CAP256", "CAND"):
        e = est[name]
        _write_experiment(posit, name, e["median"], e["map_"], e["sgm"], WIDTH[name], ids, rng,
                          **((cand_over or {}) if name == "CAND" else {}))
    _write_estimator(posit, "CAP256", fx.SHA_C)                        # the checkpoint the fixture manifests record
    _write_vector(posit)
    log = Path(root) / "cap256_experiment.out"
    log.write_text("[00:00:01] MAP stops: early 60\nnoise\n[00:00:02] MAP stops: early 58, budget 2\n", encoding="utf-8")
    return posit, est, log


def _run(posit, out, extra=(), log=None):
    argv = ["--total-time-seconds", "2", "--base", "CAP256", "--control", "baseline", "--candidates", "CAND",
            "--vector-tag", "REF", "--posit", str(posit), "--out-dir", str(out), *extra]
    if log is not None:
        argv += ["--log", f"CAP256={log}"]
    return EC.main(argv)


def _refused(fn, needle):
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"not refused: {needle}")


def test_the_comparison_aligns_windows_and_reproduces_the_synthetic_products():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        posit, est, log = synthetic_posit(tmp, rng)
        out = Path(tmp) / "comparison"
        assert _run(posit, out, log=log) == 0
        files = ["comparison.md", "comparison.json", "comparison.npz", "PROVENANCE.md", "README.md"]
        files += [f"figures/distribution_{k}.png" for k in KEYS] + [f"figures/pairs_{k}.png" for k in KEYS]
        files += [f"figures/window_drift_{n}.png" for n in ("baseline", "CAP256", "CAND")]
        for f in files:
            assert (out / f).exists(), f
        r = json.loads((out / "comparison.json").read_text())
        assert r["order"] == ["baseline", "CAP256", "CAND"] and r["base"] == "CAP256" and r["control"] == "baseline"
        assert r["n_windows"] == N and r["n_recordings"] == CELLS and r["chunks"] == list(range(CHUNKS))
        # alignment: the stored arrays, put back in (cell, chunk) order, are the generated ones
        z = np.load(out / "comparison.npz")
        order = np.lexsort((z["chunk"], z["cell"]))
        for name in ("baseline", "CAP256", "CAND"):
            assert np.allclose(z[f"{name}_median"][order], est[name]["median"]), name
            assert np.allclose(z[f"{name}_MAP"][order], est[name]["map_"]), name
        # an exact shift and identical series
        p = r["pairs"]["median"]["CAP256 - baseline"]
        assert abs(p["mu_pc"]["median_shift_log10"] - 0.1) < 1e-9 and p["mu_pc"]["frac_later_higher"] == 1.0
        assert abs(p["sigma_r"]["pearson_r"] - 1.0) < 1e-12 and p["sigma_r"]["median_shift_log10"] == 0.0
        assert abs(p["mu_pc"]["recording_median_shift_log10"] - 0.1) < 1e-9
        # a constant series: the correlation is undefined, not zero
        assert r["pairs"]["median"]["CAND - CAP256"]["lambda_rate"]["pearson_r"] is None
        assert r["pairs"]["median"]["CAND - baseline"]["lambda_rate"]["spearman_r"] is None
        # the prior-box shares
        loc = r["location"]["median"]["CAND"]["mu_pc"]
        assert loc["below_prior_frac"] == 0.5 and loc["above_prior_frac"] == 0.0 and loc["outside_prior_frac"] == 0.5
        assert loc["n_below_prior"] == N // 2 and loc["n_above_prior"] == 0 and loc["n_outside_prior"] == N // 2
        assert abs(loc["below_extent_log10"] - 0.05) < 1e-9 and loc["above_extent_log10"] == 0.0
        assert abs(loc["min_physical"] - 10 ** (LO[I_PC] - 0.05)) < 1e-6
        assert r["location"]["median"]["baseline"]["mu_pc"]["outside_prior_frac"] == 0.0
        assert r["location"]["median"]["baseline"]["mu_pc"]["n_outside_prior"] == 0
        # whole intervals beyond the box: CAND's mu_pc medians sit 0.05 dex below the floor with a 90 % half-width of
        # 1.6449 x 0.02 = 0.033 dex, so the whole 90 % interval of every such window lies below the floor
        bd = r["boundary"]["CAND"]["mu_pc"]
        assert bd["n_interval90_below"] == N // 2 and bd["frac_interval90_below"] == 0.5 and bd["n_interval90_above"] == 0
        assert bd["n_interval50_below"] == N // 2 and r["boundary"]["baseline"]["mu_pc"]["n_interval90_below"] == 0
        # the ranges of the window medians and of the per-recording medians
        br = r["between_recordings"]["median"]["CAND"]["mu_pc"]
        per_rec = np.array([np.median(est["CAND"]["median"][c * CHUNKS:(c + 1) * CHUNKS, I_PC]) for c in range(CELLS)])
        assert abs(br["recording_median_min_log10"] - per_rec.min()) < 1e-12 and abs(br["recording_median_max_log10"] - per_rec.max()) < 1e-12
        assert abs(br["recording_median_max_physical"] - 10 ** per_rec.max()) < 1e-6
        # widths from the stored quantiles
        w = r["widths"]["CAND"]["mu_r"]
        assert abs(w["median_width_90_log10"] - 2 * 1.6449 * WIDTH["CAND"]) < 1e-9
        assert abs(w["median_width_50_log10"] - 2 * 0.6745 * WIDTH["CAND"]) < 1e-9
        assert abs(w["median_width_90_over_prior"] - 2 * 1.6449 * WIDTH["CAND"] / (HI[0] - LO[0])) < 1e-9
        # the three point estimates of one posterior
        a = r["within_run"]["CAND"]["sigma_r"]
        assert a["MAP_inside_90_frac"] == 0.0 and a["MAP_inside_50_frac"] == 0.0
        assert abs(a["median_signed_MAP_minus_median_over_iqr"] - 3 / (2 * 0.6745)) < 1e-6
        assert r["within_run"]["baseline"]["mu_r"]["MAP_inside_50_frac"] == 1.0
        assert abs(r["within_run"]["baseline"]["mu_r"]["median_abs_SGM_minus_median_log10"] - 0.002) < 1e-9
        # the largest discrepancy: CAND's bleaching MAP steps by 0.5 dex in five windows, its median does not
        assert abs(r["within_run"]["CAND"]["prob_photo_bleach"]["max_abs_MAP_minus_median_log10"] - 0.5) < 1e-9
        assert abs(r["within_run"]["CAND"]["prob_photo_bleach"]["median_abs_MAP_minus_median_log10"]) < 1e-9
        assert abs(r["within_run"]["CAND"]["sigma_r"]["max_abs_MAP_minus_median_log10"] - 3 * WIDTH["CAND"]) < 1e-9
        assert abs(r["within_run"]["baseline"]["mu_r"]["max_abs_SGM_minus_median_log10"] - 0.002) < 1e-9
        # movement between windows: one step of 0.5 dex in one recording, in the MAP only
        f = r["fluctuation"]["MAP"]["CAND"]["prob_photo_bleach"]
        assert f["jumps_over_threshold"] == 1 and f["recordings_with_a_jump"] == 1
        assert r["fluctuation"]["median"]["CAND"]["prob_photo_bleach"]["jumps_over_threshold"] == 0
        # drift: +0.05 dex per window over nine steps, in every recording
        rows = [x for x in r["drift"]["CAND"] if x["parameter"] == "prob_photo_bleach" and x["estimate"] == "posterior median"]
        assert len(rows) == 1 and abs(rows[0]["change_median"] - 0.45) < 1e-9
        assert rows[0]["sign_consistency"] == 1.0 and rows[0]["material_fraction"] == 1.0 and rows[0]["n_cells"] == CELLS
        # between recordings: the SD of the per-recording means and the median SD within a recording
        b = r["between_recordings"]["median"]["baseline"]["mu_r"]
        per = est["baseline"]["median"][:, 0].reshape(CELLS, CHUNKS)          # generated in (cell, chunk) order
        assert abs(b["between_recording_sd_log10"] - np.std(per.mean(axis=1), ddof=1)) < 1e-12
        assert abs(b["median_within_recording_sd_log10"] - np.median(per.std(axis=1, ddof=1))) < 1e-12
        assert b["between_recording_sd_log10"] > b["median_within_recording_sd_log10"]
        # the selected vector and the references
        assert np.allclose(r["vector"]["log10"], VECTOR)
        s = r["vector_shift"]["median"]["CAP256"]["mu_pc"]
        assert abs(s["median_minus_vector_log10"] - (np.median(est["CAP256"]["median"][:, I_PC]) - VECTOR[I_PC])) < 1e-9
        assert abs(s["median_minus_vector_percent"] - 100 * (10 ** s["median_minus_vector_log10"] - 1)) < 1e-9
        assert s["n_windows_above_vector"] == int(np.sum(est["CAP256"]["median"][:, I_PC] > VECTOR[I_PC]))
        assert abs(s["frac_windows_above_vector"] - s["n_windows_above_vector"] / N) < 1e-12
        assert set(r["references"]) == {"mu_r", "sigma_r", "mu_pc", "sigma_pc", "lambda_rate"}
        assert r["references"]["mu_pc"]["values"][0]["value"] == 386.0
        assert r["references"]["sigma_r"]["upper_biased"]["value"] == 0.37
        # identity and the MAP stops
        assert r["columns"]["CAP256"]["identity"]["status"] == "the product's checkpoint is the estimator artifact's weights"
        assert r["columns"]["baseline"]["identity"]["status"].startswith("estimator artifact not present")
        assert r["columns"]["CAP256"]["map_stops"] == {"budget": 2, "early": 118} and r["columns"]["CAND"]["map_stops"] is None
        assert r["columns"]["CAND"]["job"] == "job-CAND" and r["columns"]["CAP256"]["pool_mode"] == "unrestricted"
        assert r["code_identity"]["script_sha256"] and r["missing_columns"] == []
        md = (out / "comparison.md").read_text()
        assert "not read (no log given)" in md and "| – |" in md and "selected vector" in md
        assert "386 (Fab per-detection)" in md and "never recovery or accuracy" in md
        assert "## Range and boundary" in md and "typical / max" in md and f"{N // 2} (50 %)" in md
        assert "script sha256" in (out / "PROVENANCE.md").read_text()
        # a second comparison into the same folder is refused
        _refused(lambda: _run(posit, out), "refusing to overwrite")


def test_mismatched_windows_geometry_or_pool_mode_are_refused():
    rng = np.random.default_rng(8)
    with tempfile.TemporaryDirectory() as tmp:
        posit, _, _ = synthetic_posit(tmp, rng, cand_over=dict(pool_mode="bounded"))
        _refused(lambda: _run(posit, Path(tmp) / "a"), "CAND: pool_mode bounded differs")
    with tempfile.TemporaryDirectory() as tmp:
        geometry = S.window_geometry(n_frames=100, step_frames=50, span_frames=1000)
        posit, _, _ = synthetic_posit(tmp, rng, cand_over=dict(window_geometry=geometry))
        _refused(lambda: _run(posit, Path(tmp) / "b"), "CAND: window_geometry")
    with tempfile.TemporaryDirectory() as tmp:
        posit, _, _ = synthetic_posit(tmp, rng)
        stem = _stem("CAND")
        path = posit / f"{stem}_MAP_Experiment" / f"{stem}_MAP_Experiment.npz"
        z = dict(np.load(path, allow_pickle=False))
        z["cell"] = z["cell"].copy()
        z["cell"][z["cell"] == CELLS - 1] = CELLS + 3                      # one recording the base does not have
        np.savez_compressed(path, **z)
        _refused(lambda: _run(posit, Path(tmp) / "c"), "CAND: windows differ from the base's")


def test_a_checkpoint_that_is_not_the_estimator_weights_is_refused():
    rng = np.random.default_rng(13)
    with tempfile.TemporaryDirectory() as tmp:
        posit, _, _ = synthetic_posit(tmp, rng)
        _write_estimator(posit, "CAND", "d" * 64)
        _refused(lambda: _run(posit, Path(tmp) / "a"), "CAND: the Experiment product records checkpoint")


def test_the_vector_must_be_one_selected_vector_and_logs_must_name_compared_estimators():
    rng = np.random.default_rng(21)
    with tempfile.TemporaryDirectory() as tmp:
        posit, _, log = synthetic_posit(tmp, rng)
        _write_vector(posit, choice="box")
        _refused(lambda: _run(posit, Path(tmp) / "a"), "holds no single selected vector")
    with tempfile.TemporaryDirectory() as tmp:
        posit, _, log = synthetic_posit(tmp, rng)
        _refused(lambda: EC.main(["--total-time-seconds", "2", "--base", "CAP256", "--posit", str(posit),
                                  "--out-dir", str(Path(tmp) / "b"), "--log", f"NOPE={log}"]), "not compared: ['NOPE']")
        _refused(lambda: EC.main(["--total-time-seconds", "2", "--base", "CAP256", "--posit", str(posit),
                                  "--out-dir", str(Path(tmp) / "c"), "--log", "CAP256"]), "--log expects NAME=PATH")


def test_the_dry_run_compares_nothing_a_missing_base_is_fatal_and_a_missing_candidate_is_listed():
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        posit, _, _ = synthetic_posit(tmp, rng)
        out = Path(tmp) / "dry"
        assert _run(posit, out, extra=("--dry-run",)) == 0
        assert not out.exists()
        _refused(lambda: EC.main(["--total-time-seconds", "2", "--base", "NOPE", "--posit", str(posit),
                                  "--out-dir", str(out)]), "base column")
        out2 = Path(tmp) / "with_missing"
        assert EC.main(["--total-time-seconds", "2", "--base", "CAP256", "--candidates", "CAND", "NOPE",
                        "--posit", str(posit), "--out-dir", str(out2)]) == 0
        r = json.loads((out2 / "comparison.json").read_text())
        assert r["order"] == ["CAP256", "CAND"] and r["missing_columns"] == ["NOPE"] and r["vector"] is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
