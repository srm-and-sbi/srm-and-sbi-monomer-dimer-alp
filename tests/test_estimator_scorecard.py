"""The estimator scorecard compiler (Script_Bank/Analysis/..._DETECTOR_Estimator_Scorecard.py) on synthetic,
schema-valid records: rows are aligned by identifier across products stored in different orders and the
calibration clouds by exact truth match; the recovery, coverage and paired log-density numbers reproduce
what the synthetic estimators were built to have; the flags fire on the conditions they name (a collapsed
parameter, better points with worse uncertainty, an inconclusive difference); missing optional records read
as "not run" and a missing candidate is listed, not fatal; the dry run compiles nothing; mismatched
recordings are refused. CPU only; no trained flow."""
import importlib.util
import json
import shutil
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

CALIBRATION_REPORT = """# Posterior_Calibration diagnostics - synthetic

## Quantitative

| metric | value | expected | meaning |
|---|---|---|---|
| calibration_videos | 600 | - | - |
| coverage_max_gap | 0.0654  (at nominal 0.55) | < 0.05 | gap |
| tarp_ATC | -0.0247 | - | atc |
| lc2st_reject_fraction | 0.000 | - | frac |

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


def synthetic_posit(root, rng):
    """Three estimators on the same 600 recordings, each product stored in its own row order.
    baseline (control): noise 0.08; CAP256 (base): noise 0.05 with prob_photo_bleach estimated as a constant
    (a collapse); CAND: noise 0.02 everywhere, but its mu_r intervals are five times too narrow (better points,
    worse uncertainty). Calibration: CAP256 and CAND have clouds (CAND's truth log-density 0.5 higher on every
    recording); baseline and CAP256 have reports; baseline has a probe record. NOPE is a candidate with nothing."""
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
            (cal / "report.md").write_text(CALIBRATION_REPORT, encoding="utf-8")
        if name == "baseline":
            probe = posit / f"{stem}_Embedding_Probe"
            probe.mkdir()
            (probe / f"{stem}_Embedding_Probe.json").write_text(json.dumps(dict(
                meta=dict(fit_tasks=[0], held_out_tasks=[1], weights_sha256="a" * 64),
                result={k: dict(held_out=dict(mae=0.01 * (i + 1), bias=0.0, slope=0.9, corr=0.95, n=200),
                                null_mae=0.1) for i, k in enumerate(KEYS)})), encoding="utf-8")
    return posit, truth


def _run(posit, out, extra=()):
    return SC.main(["--total-time-seconds", "2", "--base", "CAP256", "--control", "baseline",
                    "--candidates", "CAND", "NOPE", "--posit", str(posit), "--out-dir", str(out),
                    "--bootstrap", "200", *extra])


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
        # the collapsed parameter: no correlation, flagged under the floor and as a drop against the control
        assert abs(r["recovery"]["prob_photo_bleach"]["CAP256"]["median"]["all"]["corr"]) < 0.15
        assert r["summary"]["CAP256"]["below_floor_parameters"] == ["prob_photo_bleach"]
        assert r["flags"]["CAP256"]["baseline"]["prob_photo_bleach"]["collapse"]
        assert r["flags"]["CAP256"]["baseline"]["prob_photo_bleach"]["below_floor"]
        # calibrated intervals cover at their nominal rate; the narrowed ones do not
        assert abs(r["coverage"]["sigma_r"]["CAND"]["all"]["cover90"] - 0.9) < 0.05
        assert r["coverage"]["mu_r"]["CAND"]["all"]["cover90"] < 0.5
        # flags of the candidate against the base
        f = r["flags"]["CAND"]["CAP256"]
        assert f["mu_r"]["category"] == "points better, uncertainty worse", f["mu_r"]
        for k in ("sigma_r", "mu_pc", "sigma_pc", "lambda_rate"):
            assert f[k]["category"] == "improved", (k, f[k])
        assert f["prob_photo_bleach"]["category"] == "improved" and not f["prob_photo_bleach"]["collapse"]
        vb = r["summary"]["CAND"]["versus_base"]
        assert vb["points_better_uncertainty_worse"] == ["mu_r"] and vb["collapse"] == []
        # paired truth log-density: CAND is 0.5 above the base on every recording
        p = r["joint_paired"]["CAND"]["CAP256"]["all"]
        assert abs(p["mean"] - 0.5) < 1e-9 and p["ci_low"] > 0.49 and p["share_higher"] == 1.0
        assert r["joint_paired"]["CAND"].get("baseline") is None            # the control has no cloud
        assert r["joint"]["baseline"] == {} and r["joint"]["CAP256"]["all"]["n"] == N
        # lifted reports and the probe row: present where written, "not run" elsewhere
        assert r["lifted"]["CAP256"]["quantitative"]["tarp_ATC"] == "-0.0247"
        assert r["lifted"]["CAP256"]["sbc_ks_d"]["sigma_r"] == 0.02
        assert r["lifted"]["CAP256"]["diagnosis"]["mu_pc"]["spread_z"] == 1.12
        assert r["lifted"]["baseline"]["quantitative"]["coverage_max_gap"].startswith("0.0654")
        assert r["lifted"]["CAND"] is None
        assert r["probe"]["baseline"]["parameters"]["lambda_rate"]["held_out_mae"] == 0.06
        assert r["probe"]["CAND"] is None and r["probe"]["CAP256"] is None
        md = (out / "scorecard.md").read_text()
        assert "not run" in md and "below floor; COLLAPSE (drop)" in md and "points better, uncertainty worse" in md
        # regimes partition the recordings and follow the prior midpoints
        reg = r["regimes"]
        assert reg["mu_r_low"]["n"] + reg["mu_r_high"]["n"] == N and reg["sigma_r_low"]["n"] + reg["sigma_r_high"]["n"] == N
        assert reg["operating_subgroup"]["n"] == int(np.sum(truth[:, KEYS.index("mu_pc")] < (LO[2] + HI[2]) / 2))
        assert reg["operating_subgroup"]["definition"] == "true log10 mu_pc in [2, 2.375)"
        assert r["thresholds"] == SC.DEFAULT_THRESHOLDS
        # a second compilation into the same folder is refused
        try:
            _run(posit, out)
        except SystemExit as e:
            assert "refusing to overwrite" in str(e)
        else:
            raise AssertionError("overwrote an existing scorecard")


def test_the_dry_run_compiles_nothing_and_a_missing_base_is_fatal():
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        posit, _ = synthetic_posit(tmp, rng)
        out = Path(tmp) / "dry"
        assert _run(posit, out, extra=("--dry-run",)) == 0
        assert not out.exists()
        try:
            SC.main(["--total-time-seconds", "2", "--base", "NOPE", "--posit", str(posit), "--out-dir", str(out)])
        except SystemExit as e:
            assert "base column" in str(e)
        else:
            raise AssertionError("compiled against a base without an Evaluation product")


def test_the_flag_rules():
    def column(mae, corr, gap90):
        return {"recovery": {k: {"median": {"all": {"mae_dex": mae[i], "corr": corr[i]}}} for i, k in enumerate(KEYS)},
                "coverage": {k: {"all": {"gap90": gap90[i]}} for i, k in enumerate(KEYS)}}
    ref = column([0.100] * D, [0.90] * D, [0.02] * D)
    cand = column([0.100, 0.080, 0.130, 0.080, 0.100, 0.080], [0.90, 0.90, 0.90, 0.90, 0.25, 0.55], [0.02, 0.02, 0.02, 0.09, 0.02, 0.02])
    f = SC.flag(cand, ref, SC.DEFAULT_THRESHOLDS)
    assert f["mu_r"]["category"] == "inconclusive"
    assert f["sigma_r"]["category"] == "improved"
    assert f["mu_pc"]["category"] == "worsened"
    assert f["sigma_pc"]["category"] == "points better, uncertainty worse"
    assert f["prob_photo_bleach"]["below_floor"] and f["prob_photo_bleach"]["collapse"]
    assert not f["lambda_rate"]["below_floor"] and f["lambda_rate"]["collapse"]
    assert not f["sigma_r"]["collapse"] and not f["sigma_r"]["below_floor"]
    strict = dict(SC.DEFAULT_THRESHOLDS, min_effect=0.03)
    assert SC.flag(cand, ref, strict)["sigma_r"]["category"] == "inconclusive"


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
        try:
            _run(posit, Path(tmp) / "a")
        except SystemExit as e:
            assert "truths differ" in str(e)
        else:
            raise AssertionError("accepted altered truths")
        # a recording the base does not have
        z["true_log10"][0, 0] -= 1e-6
        z["sim_index"] = z["sim_index"].copy()
        z["sim_index"][0] = 10_000
        np.savez_compressed(folder / f"{stem}_MAP_Recovery.npz", **z)
        try:
            _run(posit, Path(tmp) / "b")
        except SystemExit as e:
            assert "recordings differ" in str(e)
        else:
            raise AssertionError("accepted a foreign recording")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
