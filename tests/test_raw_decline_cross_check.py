"""The raw-domain decline cross-check (Script_Bank/Analysis/..._DETECTOR_Raw_Decline_Cross_Check.py) on a
synthetic, schema-valid Experiment product and a synthetic raw-domain record built to known relations: the
recordings and windows are aligned by identifier and frame start across different storage orders; the local
direct rates equal the estimator's window values except in three windows whose field statistic did not fall
(local rate below zero), which every count, median and correlation includes and every logarithmic shift
excludes, with its own count; the analytical dye retention at the estimator's value equals the field retention
by construction; the free-offset fitted rate is twice the estimator's value on the usable recordings (shift
−log10 2); the late-phase rate is half of it except in one recording where it is below zero; the opening field
excess ranks with the estimator's window-0 mu_pc while the background level ranks against it, so the brightness
pairing must read the excess; a missing recording, a window start the raw record lacks, the wrong domain and an
existing record are refused; the dry run computes nothing. CPU only; no trained flow."""
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
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming      # noqa: E402
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow, parameter_keys  # noqa: E402
import _product_fixtures as fx                                                        # noqa: E402


def _companion():
    path = REPO / "Script_Bank" / "Analysis" / "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Raw_Decline_Cross_Check.py"
    spec = importlib.util.spec_from_file_location("detector_raw_decline_cross_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CC = _companion()
CFG = detector_workflow()
KEYS = parameter_keys(CFG)
LO = np.array(CFG.param_module.theta_lower_bound(), float)
HI = np.array(CFG.param_module.theta_upper_bound(), float)
D = len(KEYS)
CELLS, CHUNKS = 12, 10
N = CELLS * CHUNKS
I_PC, I_PB = KEYS.index("mu_pc"), KEYS.index("prob_photo_bleach")
OFFSETS = np.array([-1.6449, -0.6745, 0.0, 0.6745, 1.6449])
TAG = "CAND"
SEPARATION = 950.0
NONFALLING = ((0, 3), (0, 7), (1, 5))   # (cell, window) whose field statistic did not fall: local rate below zero
LATE_NEGATIVE_CELL = 2                  # the recording whose late-phase rate is below zero


def _stem(tag):
    paths = CFG.paths.with_condition("FAB")
    timing = RunTiming(total_time_seconds=2.0, frames=PARAMETERS.simulation.timing)
    return f"{paths.project_alias}_{paths.product_label(timing.label, tag)}"


def _write_experiment(posit, median, ids, rng):
    stem = _stem(TAG)
    quantiles = median[:, :, None] + OFFSETS[None, None, :] * 0.02
    perm = rng.permutation(N)
    a = dict(map_estimate=median[perm] + 0.001, posterior_quantiles=quantiles[perm], posterior_sgm=median[perm] + 0.002,
             scores=np.zeros(N), kind_index=ids[perm, 0].astype(np.int64), cell=ids[perm, 1].astype(np.int64),
             chunk=ids[perm, 2].astype(np.int64), kinds=np.asarray(["FAB"]))
    m = fx.valid_manifest("experiment", N, kinds=["FAB"], parameter_keys=KEYS, sgm_scale=[1.0] * D,
                          pool_mode="unrestricted", job_id="job-cand")
    a[S.MANIFEST_KEY] = S.encode_manifest(m)
    folder = posit / f"{stem}_MAP_Experiment"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{stem}_MAP_Experiment.npz", **a)
    return m["window_geometry"]


def _write_raw(folder, cells, direct_window, p_recording, late, pc_window0, starts, rng, *, domain="raw", drop_cell=None):
    """A raw record in its own recording order: local rates as given; whole rate = the estimator's recording
    value; field retention = the analytical expectation at it; fitted = 2x on usable recordings; late rate as
    given; opening field excess increasing with the estimator's window-0 mu_pc; background level decreasing."""
    folder.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(cells))
    keep = np.array([c != drop_cell for c in cells[order]])
    o = order[keep]
    usable = np.ones(len(o), bool)
    usable[0] = False
    arrays = dict(cell=cells[o].astype(np.int64), estimate=2.0 * p_recording[o], usable=usable,
                  decline_p_whole=p_recording[o], decline_p_late=late[o],
                  decline_ratio_whole=(1.0 - p_recording[o]) ** (SEPARATION / 100.0),
                  decline_ratio_late=(1.0 - 0.5 * p_recording[o]) ** (500.0 / 100.0),
                  center_separation_frames=np.full(len(o), SEPARATION), opening_excess=20.0 + 0.1 * pc_window0[o],
                  opening_median=1300.0 - 0.1 * pc_window0[o], closing_median=1200.0 - 0.1 * pc_window0[o],
                  window_rate_2s=direct_window[o], window_starts_2s=starts,
                  domain=np.asarray([domain]), kinds=np.asarray(["FAB"]))
    np.savez_compressed(folder / CC.RAW_ARRAYS, **arrays)
    (folder / "summary.json").write_text(json.dumps({"condition": "FAB", "domain": domain, "compare_values": [0.05, 0.03]}),
                                         encoding="utf-8")
    return folder / CC.RAW_ARRAYS


def synthetic(root, rng, **raw_over):
    posit = Path(root) / "Posit"
    posit.mkdir()
    cells = np.repeat(np.arange(CELLS), CHUNKS)
    chunks = np.tile(np.arange(CHUNKS), CELLS)
    ids = np.stack([np.zeros(N, int), cells, chunks], axis=1)
    level = LO + (0.2 + 0.6 * rng.random((CELLS, D))) * (HI - LO)
    median = level[cells] + rng.normal(0, 0.02, (N, D))
    geometry = _write_experiment(posit, median, ids, rng)
    p_window = 10.0 ** median[:, I_PB].reshape(CELLS, CHUNKS)
    p_recording = 10.0 ** np.median(median[:, I_PB].reshape(CELLS, CHUNKS), axis=1)
    pc_window0 = 10.0 ** median[:, I_PC].reshape(CELLS, CHUNKS)[:, 0]
    direct = p_window.copy()
    for c, w in NONFALLING:
        direct[c, w] = -0.01
    late = 0.5 * p_recording
    late[LATE_NEGATIVE_CELL] = -0.002
    starts = np.arange(CHUNKS) * int(geometry["step_frames"])
    raw = _write_raw(posit / "raw_record", np.arange(CELLS), direct, p_recording, late, pc_window0, starts, rng,
                     **raw_over)
    return posit, raw, dict(p_window=p_window, direct=direct, p_recording=p_recording, pc_window0=pc_window0)


def _run(posit, raw, out, extra=()):
    return CC.main(["--total-time-seconds", "2", "--estimator", TAG, "--posit", str(posit), "--raw-record", str(raw),
                    "--out-dir", str(out), "--authorization", "test", *extra])


def _refused(fn, needle):
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"not refused: {needle}")


def test_the_cross_check_aligns_recordings_and_windows_and_reproduces_the_built_relations():
    rng = np.random.default_rng(11)
    k = len(NONFALLING)
    with tempfile.TemporaryDirectory() as tmp:
        posit, raw, truth = synthetic(tmp, rng)
        out = Path(tmp) / "cc"
        assert _run(posit, raw, out) == 0
        for f in ("cross_check.md", "cross_check.json", "cross_check.npz", "PROVENANCE.md", "README.md",
                  "figures/rates_per_recording.png", "figures/retention.png", "figures/window_positions.png",
                  "figures/brightness.png"):
            assert (out / f).exists(), f
        r = json.loads((out / "cross_check.json").read_text())
        assert r["estimator"] == TAG and r["n_recordings"] == CELLS and r["n_windows_per_recording"] == CHUNKS
        assert r["n_usable_fits"] == CELLS - 1 and r["compare_values"] == [0.05, 0.03]
        P = r["per_view"]["median"]
        # aligned arrays, in (cell, chunk) order whatever the storage orders were
        z = np.load(out / "cross_check.npz")
        assert np.allclose(z["direct_window_rate"], truth["direct"]) and np.allclose(z["median_bleaching_window"], truth["p_window"])
        assert np.allclose(z["median_bleaching_recording"], truth["p_recording"]) and z["usable"].sum() == CELLS - 1
        assert np.allclose(z["free_offset_fit_rate"], 2.0 * truth["p_recording"])
        # every finite window counts; the shift uses the positive local rates only, with its own count
        a = P["window_positions"]["all_windows"]
        assert a["n"] == N and a["n_positive"] == N - k and a["n_raw_nonpositive"] == k
        assert a["n_nn_higher"] == k and a["n_nn_higher_positive"] == 0
        assert abs(a["median_log10_ratio_nn_over_raw"]) < 1e-9 and 0.0 < a["spearman_r"] < 1.0
        assert abs(P["window_positions"]["within_recording_spearman_median"] - 1.0) < 1e-12
        assert P["window_positions"]["within_recording_spearman_n_positive"] == CELLS
        pos = P["window_positions"]["per_position"]
        assert [p["chunk"] for p in pos] == list(range(CHUNKS)) and all(p["n"] == CELLS for p in pos)
        assert sum(p["n_positive"] for p in pos) == N - k and sum(p["n_direct_nonpositive"] for p in pos) == k
        assert sum(p["n_nn_higher"] for p in pos) == k and all(p["n_nn_higher_positive"] == 0 for p in pos)
        # whole rate = the estimator's recording value; fitted = 2x on the usable recordings; late = half or below zero
        w = P["rates_per_recording"]["whole_recording"]
        assert w["n"] == CELLS and w["n_positive"] == CELLS and abs(w["spearman_r"] - 1.0) < 1e-12
        assert abs(w["median_log10_ratio_nn_over_raw"]) < 1e-9
        f = P["rates_per_recording"]["free_offset_fit"]
        assert f["n"] == CELLS - 1 and abs(f["median_log10_ratio_nn_over_raw"] + np.log10(2)) < 1e-9 and f["n_nn_higher"] == 0
        lt = P["rates_per_recording"]["late_phase"]
        assert lt["n"] == CELLS and lt["n_positive"] == CELLS - 1 and lt["n_raw_nonpositive"] == 1
        assert abs(lt["median_log10_ratio_nn_over_raw"] - np.log10(2)) < 1e-9 and lt["n_nn_higher"] == CELLS
        # retention: the analytical dye retention beside the field retention, no difference or test reported
        ret = P["retention"]
        assert ret["n"] == CELLS and abs(ret["analytical_dye_retention_median"] - ret["field_retention_median"]) < 1e-12
        assert abs(ret["spearman_r"] - 1.0) < 1e-12 and abs(ret["nn_bleaching_vs_field_retention"]["spearman_r"] + 1.0) < 1e-12
        assert abs(ret["compare_values_analytical"]["0.05"] - 0.95 ** 9.5) < 1e-12
        assert "median_predicted_minus_measured" not in ret and "n_predicted_below_measured" not in ret
        # brightness: the opening field excess (not the background level) ranks with the estimator's window-0 mu_pc
        b = P["brightness"]
        assert abs(b["mu_pc_window0_vs_opening_excess"]["spearman_r"] - 1.0) < 1e-12
        assert b["mu_pc_window0_vs_opening_excess"]["n"] == CELLS
        md = (out / "cross_check.md").read_text()
        for needle in ("never an equality", "## Window by window", "## Brightness beside the decline",
                       "## Analytical dye retention beside experimental field retention", "not a test",
                       "opening field excess", f"field statistic in {k} of {N} matched windows",
                       f"In {N - k} windows the local rate is positive", f"in the other {k} the field statistic did not fall",
                       f"| late-phase rate (mid to closing window) | {CELLS} |"):
            assert needle in md, needle
        assert "All matched windows" not in md and "fitted component" not in md
        assert "Authorization: test." in (out / "PROVENANCE.md").read_text()
        _refused(lambda: _run(posit, raw, out), "refusing to overwrite")


def test_mismatched_recordings_window_starts_and_domain_are_refused_and_the_dry_run_computes_nothing():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        posit, raw, _ = synthetic(tmp, rng, drop_cell=3)
        _refused(lambda: _run(posit, raw, Path(tmp) / "a"), "recordings differ")
    with tempfile.TemporaryDirectory() as tmp:
        posit, raw, _ = synthetic(tmp, rng)
        z = dict(np.load(raw))
        z["window_starts_2s"] = z["window_starts_2s"] + 50
        np.savez_compressed(raw, **z)
        _refused(lambda: _run(posit, raw, Path(tmp) / "b"), "are not window starts of the raw record")
    with tempfile.TemporaryDirectory() as tmp:
        posit, raw, _ = synthetic(tmp, rng, domain="stored")
        _refused(lambda: _run(posit, raw, Path(tmp) / "c"), "is not the raw domain")
    with tempfile.TemporaryDirectory() as tmp:
        posit, raw, _ = synthetic(tmp, rng)
        out = Path(tmp) / "dry"
        assert _run(posit, raw, out, extra=("--dry-run",)) == 0
        assert not out.exists()
        _refused(lambda: CC.main(["--total-time-seconds", "2", "--estimator", "NOPE", "--posit", str(posit),
                                  "--raw-record", str(raw), "--out-dir", str(out)]), "no Experiment product")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
