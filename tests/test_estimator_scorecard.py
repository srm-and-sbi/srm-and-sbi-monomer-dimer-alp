"""The estimator scorecard compiler (Script_Bank/Analysis/..._DETECTOR_Estimator_Scorecard.py) on synthetic,
schema-valid records: rows are aligned by identifier across products stored in different orders and the
calibration clouds by truth match; the recovery, coverage and paired log-density numbers reproduce what the
synthetic estimators were built to have; the flags fire on the conditions they name (below the floor, a
large correlation drop, a collapse, better points with worse uncertainty, a difference under the threshold);
records of one column from different weights are refused, as is a calibration product that does not hold
the Evaluation's recordings or a report that does not describe its arrays; absent optional records read as
"not run" and a missing candidate is listed, not fatal; the dry run compiles nothing; mismatched recordings
are refused. CPU only; no trained flow."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from srm_and_sbi_monomer_dimer_alp import artifact_schema as S                    # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming   # noqa: E402
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow, parameter_keys  # noqa: E402
import _product_fixtures as fx                                                     # noqa: E402


def _companion():
    path = REPO / "Script_Bank" / "Analysis" / "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Estimator_Scorecard.py"
    spec = importlib.util.spec_from_file_location("detector_estimator_scorecard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SC = _companion()
CFG = detector_workflow()
KEYS = parameter_keys(CFG)
LO = np.array(CFG.param_module.theta_lower_bound(), float)
HI = np.array(CFG.param_module.theta_upper_bound(), float)
D = len(KEYS)
N = 600
NOISE = {"baseline": 0.08, "CAP256": 0.05, "CAND": 0.02}
Z90, Z50 = 1.6449, 0.6745                                  # normal quantiles for calibrated 90 % / 50 % intervals
LOG_DENSITY_SHIFT = {"baseline": 0.0, "CAP256": 0.0, "CAND": 0.5}
WEIGHTS = fx.SHA_C                                          # the checkpoint the fixture's Evaluation manifests record


def calibration_report(n_videos=N):
    return f"""# Posterior_Calibration diagnostics - synthetic

## Quantitative

| metric | value | expected | meaning |
|---|---|---|---|
| calibration_videos | {n_videos} | - | - |
| coverage_max_gap | 0.0654  (at nominal 0.55) | < 0.05 | gap |
| tarp_ATC | -0.0247 | - | atc |
| lc2st_reject_fraction | 0.000 | - | frac |
| marginal_1d_worst | 0.0283 | - | worst |ATC| over the one-dimensional marginals |

## SBC rank uniformity (per parameter)

| parameter | label | KS D (effect size) | KS p-value | C2ST(rank) | verdict |
|---|---|---|---|---|---|
""" + "\n".join(f"| {k} | $x$ | 0.0{i + 1} | 0.00e+00 | - | ok |" for i, k in enumerate(KEYS)) + """

## Diagnosis: location error versus width error (per parameter)

| parameter | label | bias (z) | spread (z) | sharpness | physical bias | defect |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {k} | $x$ | +0.0{i} | 1.1{i} | 7.1% | 1.001x | ok |" for i, k in enumerate(KEYS)) + "\n"


def _stem(paths, tag):
    timing = RunTiming(total_time_seconds=2.0, frames=PARAMETERS.simulation.timing)
    return f"{paths.project_alias}_{paths.product_label(timing.label, tag)}"


def _write_evaluation(posit, stem, truth, ids, est, quantiles, job):
    a = dict(map_estimate=est["MAP"], posterior_quantiles=quantiles, posterior_sgm=est["SGM"],
             scores=np.zeros(truth.shape[0]), true_log10=truth,
             task_index=ids[:, 0].astype(np.int64), sim_index=ids[:, 1].astype(np.int64))
    m = fx.valid_manifest("evaluation", truth.shape[0], parameter_keys=KEYS, sgm_scale=[1.0] * D, job_id=job)
    a[S.MANIFEST_KEY] = S.encode_manifest(m)
    folder = posit / f"{stem}_MAP_Recovery"
    folder.mkdir(parents=True)
    np.savez_compressed(folder / f"{stem}_MAP_Recovery.npz", **a)


def _write_estimator(posit, stem, weights):
    manifest = dict(weights_sha256=weights, parameter_keys=KEYS,
                    rebuild_spec=dict(embedding_args=dict(start_channels=16, spatial_pooling="stats"),
                                      maf_args=dict(hidden_features=128, num_transforms=8, num_blocks=2,
                                                    dropout_probability=0.1)),
                    metadata=dict(network_preset="capacity256"))
    np.savez(posit / f"{stem}_Estimator.npz", manifest=np.asarray(json.dumps(manifest)), prior_low=LO, prior_high=HI)


def _write_probe(posit, stem, weights):
    probe = posit / f"{stem}_Embedding_Probe"
    probe.mkdir()
    (probe / f"{stem}_Embedding_Probe.json").write_text(json.dumps(dict(
        meta=dict(fit_tasks=[0], held_out_tasks=[1], weights_sha256=weights),
        result={k: dict(held_out=dict(mae=0.01 * (i + 1), bias=0.0, slope=0.9, corr=0.95, n=200), null_mae=0.1)
                for i, k in enumerate(KEYS)})), encoding="utf-8")


def synthetic_posit(root, rng):
    """Three estimators on the same 600 recordings, each product stored in its own row order.
    baseline (control): noise 0.08; CAP256 (base): noise 0.05 with prob_photo_bleach estimated as a constant
    (a collapse); CAND: noise 0.02 everywhere, but its mu_r intervals are five times too narrow (better points,
    worse uncertainty). Calibration: CAP256 and CAND have clouds (CAND's truth log-density 0.5 higher on every
    recording); baseline and CAP256 have reports (the baseline's without arrays, so it is not lifted); baseline
    has a probe record and CAP256 an estimator artifact, both recording the Evaluation's checkpoint. NOPE is a
    candidate with nothing."""
    paths = CFG.paths.with_condition("FAB")
    posit = Path(root) / "Posit"
    posit.mkdir()
    tasks = np.repeat(np.arange(3), N // 3)
    sims = np.tile(np.arange(N // 3), 3)
    ids = np.stack([tasks, sims], axis=1)
    truth = LO + rng.random((N, D)) * (HI - LO)
    truth_ld = rng.normal(6.0, 1.0, N)
    for name in ("baseline", "CAP256", "CAND"):
        sigma = NOISE[name]
        centre = truth + rng.normal(0.0, sigma, (N, D))
        if name == "CAP256":
            centre[:, KEYS.index("prob_photo_bleach")] = (LO[4] + HI[4]) / 2       # constant: no recovery at all
        scale = np.full(D, sigma)
        if name == "CAND":
            scale[KEYS.index("mu_r")] = sigma / 5                                 # intervals far too narrow
        offsets = np.array([-Z90, -Z50, 0.0, Z50, Z90])
        quantiles = centre[:, :, None] + offsets[None, None, :] * scale[None, :, None]
        est = {"MAP": centre + rng.normal(0, 1e-3, (N, D)), "SGM": centre + rng.normal(0, 1e-3, (N, D))}
        perm = rng.permutation(N)                                                  # this product's own row order
        stem = _stem(paths, None if name == "baseline" else name)
        _write_evaluation(posit, stem, truth[perm], ids[perm], {v: a[perm] for v, a in est.items()},
                          quantiles[perm], job=f"job-{name}")
        cal = posit / f"{stem}_Posterior_Calibration"
        cal.mkdir()
        if name in ("CAP256", "CAND"):
            cperm = rng.permutation(N)
            np.savez_compressed(cal / f"{stem}_Posterior_Calibration.npz", truths=truth[cperm],
                                samples=rng.normal(size=(N, 10, D)), truth_log_probs=(truth_ld + LOG_DENSITY_SHIFT[name])[cperm],
                                sample_log_probs=rng.normal(size=(N, 10)), parameter_keys=np.array(KEYS))
        if name in ("baseline", "CAP256"):
            (cal / "report.md").write_text(calibration_report(), encoding="utf-8")
        if name == "baseline":
            _write_probe(posit, stem, WEIGHTS)
        if name == "CAP256":
            _write_estimator(posit, stem, WEIGHTS)
    return posit, truth


def _run(posit, out, extra=()):
    return SC.main(["--total-time-seconds", "2", "--base", "CAP256", "--control", "baseline",
                    "--candidates", "CAND", "NOPE", "--posit", str(posit), "--out-dir", str(out),
                    "--bootstrap", "200", "--seed", "5", *extra])


def _refused(fn, needle):
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"not refused: {needle}")


def test_the_scorecard_aligns_rows_and_reproduces_the_synthetic_estimators():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        posit, truth = synthetic_posit(tmp, rng)
        out = Path(tmp) / "scorecard"
        assert _run(posit, out) == 0
        for f in ("scorecard.md", "scorecard.json", "scorecard.npz", "PROVENANCE.md", "README.md",
                  "figures/recovery_and_coverage.png", "figures/truth_log_density_paired.png"):
            assert (out / f).exists(), f
        r = json.loads((out / "scorecard.json").read_text())
        assert r["order"] == ["baseline", "CAP256", "CAND"] and r["base"] == "CAP256" and r["control"] == "baseline"
        assert r["missing_candidates"] == ["NOPE"] and r["n_recordings"] == N
        assert r["bootstrap"]["resamples"] == 200 and r["bootstrap"]["seed"] == 5
        # alignment: the stored arrays are in the base's order and the truths are the generated ones
        z = np.load(out / "scorecard.npz")
        order = np.lexsort((z["sim_index"], z["task_index"]))
        assert np.allclose(z["true_log10"][order], truth)
        # recovery reproduces the noise each estimator was given (MAE of a normal error = sigma * sqrt(2 / pi))
        for name, sigma in NOISE.items():
            for k in KEYS:
                if name == "CAP256" and k == "prob_photo_bleach":
                    continue
                s = r["recovery"][k][name]["median"]["all"]
                assert abs(s["mae_dex"] - sigma * np.sqrt(2 / np.pi)) < 0.012, (name, k, s["mae_dex"])
                assert s["corr"] > 0.5 and abs(s["slope"] - 1) < 0.25, (name, k, s)
        # the constant estimate: zero correlation by convention, below the floor, a collapse against the control
        assert r["recovery"]["prob_photo_bleach"]["CAP256"]["median"]["all"]["corr"] == 0.0
        assert r["summary"]["CAP256"]["below_floor_parameters"] == ["prob_photo_bleach"]
        f = r["flags"]["CAP256"]["baseline"]["prob_photo_bleach"]
        assert f["below_floor"] and f["large_drop"] and f["collapse"]
        # calibrated intervals cover at their nominal rate; the narrowed ones do not
        assert abs(r["coverage"]["sigma_r"]["CAND"]["all"]["cover90"] - 0.9) < 0.05
        assert r["coverage"]["mu_r"]["CAND"]["all"]["cover90"] < 0.5
        # flags of the candidate against the base
        f = r["flags"]["CAND"]["CAP256"]
        assert f["mu_r"]["category"] == "points better, uncertainty worse", f["mu_r"]
        for k in ("sigma_r", "mu_pc", "sigma_pc", "lambda_rate"):
            assert f[k]["category"] == "improved", (k, f[k])
        assert f["prob_photo_bleach"]["category"] == "improved" and not f["prob_photo_bleach"]["large_drop"]
        vb = r["summary"]["CAND"]["versus_base"]
        assert vb["points_better_uncertainty_worse"] == ["mu_r"] and vb["collapse"] == [] and vb["large_drop"] == []
        # paired truth log-density: CAND is 0.5 above the base on every recording
        p = r["joint_paired"]["CAND"]["CAP256"]["all"]
        assert abs(p["mean"] - 0.5) < 1e-9 and p["ci_low"] > 0.49 and p["share_higher"] == 1.0
        assert r["joint_paired"]["CAND"].get("baseline") is None            # the control has no cloud
        assert r["joint"]["baseline"] == {} and r["joint"]["CAP256"]["all"]["n"] == N
        # lifted reports: CAP256 (arrays + report) lifted, the baseline's report present but not lifted, CAND none
        assert r["lifted"]["CAP256"]["quantitative"]["tarp_ATC"] == "-0.0247"
        assert r["lifted"]["CAP256"]["quantitative"]["marginal_1d_worst"] == "0.0283"
        assert r["lifted"]["CAP256"]["sbc_ks_d"]["sigma_r"] == 0.02
        assert r["lifted"]["CAP256"]["diagnosis"]["mu_pc"]["spread_z"] == 1.12
        assert r["lifted"]["baseline"] is None and r["lifted"]["CAND"] is None
        assert r["columns"]["baseline"]["lifted_status"].startswith("report present, not lifted")
        assert r["columns"]["CAP256"]["lifted_status"].startswith("lifted; the report's 600 videos")
        # identity: the records of a column are compared; the calibration product is unverified by construction
        assert r["columns"]["CAP256"]["identity"]["estimator_weights"] == WEIGHTS
        assert r["columns"]["CAP256"]["identity"]["status"].startswith("one checkpoint across evaluation_checkpoint, estimator_weights")
        assert r["columns"]["baseline"]["identity"]["probe_weights"] == WEIGHTS
        assert r["columns"]["CAND"]["identity"]["status"].startswith("a single record carries a checkpoint")
        for n in r["order"]:
            assert r["columns"][n]["identity"]["calibration"].startswith("unverified")
        assert r["columns"]["CAP256"]["encoder"]["spatial_pooling"] == "stats" and r["columns"]["CAP256"]["flow"]["num_transforms"] == 8
        # the probe row
        assert r["probe"]["baseline"]["parameters"]["lambda_rate"]["held_out_mae"] == 0.06
        assert r["probe"]["CAND"] is None and r["probe"]["CAP256"] is None
        md = (out / "scorecard.md").read_text()
        assert "not run" in md and "COLLAPSE (large drop, below floor)" in md and "points better, uncertainty worse" in md
        assert "200 resamples, seed 5" in md and "not the variation between training runs" in md
        assert "unverified" in (out / "PROVENANCE.md").read_text()
        # regimes partition the recordings and follow the prior midpoints
        reg = r["regimes"]
        assert reg["mu_r_low"]["n"] + reg["mu_r_high"]["n"] == N and reg["sigma_r_low"]["n"] + reg["sigma_r_high"]["n"] == N
        assert reg["operating_subgroup"]["n"] == int(np.sum(truth[:, KEYS.index("mu_pc")] < (LO[2] + HI[2]) / 2))
        assert reg["operating_subgroup"]["definition"] == "true log10 mu_pc in [2, 2.375)"
        assert r["thresholds"] == SC.DEFAULT_THRESHOLDS
        # a second compilation into the same folder is refused
        _refused(lambda: _run(posit, out), "refusing to overwrite")


def test_records_of_one_column_from_different_weights_are_refused():
    rng = np.random.default_rng(3)
    paths = CFG.paths.with_condition("FAB")
    with tempfile.TemporaryDirectory() as tmp:
        posit, _ = synthetic_posit(tmp, rng)
        _write_probe(posit, _stem(paths, "CAND"), "a" * 64)                 # a probe of other weights under CAND's tag
        _refused(lambda: _run(posit, Path(tmp) / "a"), "CAND: the records under this tag come from different weights")
    with tempfile.TemporaryDirectory() as tmp:
        posit, _ = synthetic_posit(tmp, rng)
        _write_estimator(posit, _stem(paths, "CAP256"), "d" * 64)          # an estimator artifact of other weights
        _refused(lambda: _run(posit, Path(tmp) / "b"), "CAP256: the records under this tag come from different weights")


def test_the_calibration_product_must_hold_the_evaluation_recordings_and_its_report_must_count_them():
    rng = np.random.default_rng(9)
    paths = CFG.paths.with_condition("FAB")
    stem = _stem(paths, "CAP256")
    with tempfile.TemporaryDirectory() as tmp:
        posit, truth = synthetic_posit(tmp, rng)
        cal = posit / f"{stem}_Posterior_Calibration"
        z = dict(np.load(cal / f"{stem}_Posterior_Calibration.npz"))
        extra = (LO + HI) / 2 + 1e-3                                        # one recording the Evaluation does not have
        z["truths"] = np.vstack([z["truths"], extra[None, :]])
        z["truth_log_probs"] = np.append(z["truth_log_probs"], 1.0)
        z["samples"] = np.vstack([z["samples"], z["samples"][:1]])
        z["sample_log_probs"] = np.vstack([z["sample_log_probs"], z["sample_log_probs"][:1]])
        (cal / "report.md").write_text(calibration_report(N + 1), encoding="utf-8")
        np.savez_compressed(cal / f"{stem}_Posterior_Calibration.npz", **z)
        _refused(lambda: _run(posit, Path(tmp) / "a"), "CAP256: the calibration product holds 601 recordings, the Evaluation 600")
    with tempfile.TemporaryDirectory() as tmp:
        posit, _ = synthetic_posit(tmp, rng)
        cal = posit / f"{stem}_Posterior_Calibration"
        (cal / "report.md").write_text(calibration_report(N - 1), encoding="utf-8")   # a report of another run
        _refused(lambda: _run(posit, Path(tmp) / "b"), "the report counts '599' calibration videos; the calibration arrays hold 600")


def test_the_dry_run_compiles_nothing_and_a_missing_base_is_fatal():
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        posit, _ = synthetic_posit(tmp, rng)
        out = Path(tmp) / "dry"
        assert _run(posit, out, extra=("--dry-run",)) == 0
        assert not out.exists()
        _refused(lambda: SC.main(["--total-time-seconds", "2", "--base", "NOPE", "--posit", str(posit), "--out-dir", str(out)]),
                 "base column")


def test_the_flag_rules():
    def column(mae, corr, gap90):
        return {"recovery": {k: {"median": {"all": {"mae_dex": mae[i], "corr": corr[i]}}} for i, k in enumerate(KEYS)},
                "coverage": {k: {"all": {"gap90": gap90[i]}} for i, k in enumerate(KEYS)}}
    ref = column([0.100] * D, [0.90] * D, [0.02] * D)
    cand = column([0.100, 0.080, 0.130, 0.080, 0.100, 0.080], [0.90, 0.90, 0.90, 0.90, 0.25, 0.55], [0.02, 0.02, 0.02, 0.09, 0.02, 0.02])
    f = SC.flag(cand, ref, SC.DEFAULT_THRESHOLDS)
    assert f["mu_r"]["category"] == "under threshold"
    assert f["sigma_r"]["category"] == "improved"
    assert f["mu_pc"]["category"] == "worsened"
    assert f["sigma_pc"]["category"] == "points better, uncertainty worse"
    # 0.90 -> 0.25: a large drop that leaves the parameter below the floor = a collapse
    assert f["prob_photo_bleach"]["below_floor"] and f["prob_photo_bleach"]["large_drop"] and f["prob_photo_bleach"]["collapse"]
    # 0.90 -> 0.55: a large drop with useful recovery remaining = not a collapse
    assert not f["lambda_rate"]["below_floor"] and f["lambda_rate"]["large_drop"] and not f["lambda_rate"]["collapse"]
    assert not f["sigma_r"]["large_drop"] and not f["sigma_r"]["below_floor"] and not f["sigma_r"]["collapse"]
    strict = dict(SC.DEFAULT_THRESHOLDS, min_effect=0.03)
    assert SC.flag(cand, ref, strict)["sigma_r"]["category"] == "under threshold"
    # a constant estimate scores zero correlation and zero slope by convention
    s = SC.recovery_stats(np.full(50, 0.2), np.linspace(0.0, 0.3, 50), 0.0, 0.3)
    assert s["corr"] == 0.0 and s["slope"] == 0.0 and s["n"] == 50


def test_mismatched_recordings_are_refused():
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as tmp:
        posit, _ = synthetic_posit(tmp, rng)
        paths = CFG.paths.with_condition("FAB")
        stem = _stem(paths, "CAND")
        folder = posit / f"{stem}_MAP_Recovery"
        z = dict(np.load(folder / f"{stem}_MAP_Recovery.npz", allow_pickle=True))
        # same identifiers, one truth altered
        z["true_log10"] = z["true_log10"].copy()
        z["true_log10"][0, 0] += 1e-6
        np.savez_compressed(folder / f"{stem}_MAP_Recovery.npz", **z)
        _refused(lambda: _run(posit, Path(tmp) / "a"), "truths differ")
        # a recording the base does not have
        z["true_log10"][0, 0] -= 1e-6
        z["sim_index"] = z["sim_index"].copy()
        z["sim_index"][0] = 10_000
        np.savez_compressed(folder / f"{stem}_MAP_Recovery.npz", **z)
        _refused(lambda: _run(posit, Path(tmp) / "b"), "recordings differ")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
