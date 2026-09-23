"""Point-estimate validation, phase 1: the kernel's definitions, checks and gates, and the runner end
to end on a tiny synthetic data bank with a stubbed sampler (no trained flow, no real tier).

The runner tests check what the pilot and the full run depend on: the artifact it writes passes its
own validation; the stored quantiles match an independent recomputation; seeds depend only on
(task, sim, stream), so a two-rank sharded run merged in the frozen order reproduces the
single-process artifact draw for draw; a pilot selects its recordings deterministically and issues
no verdict; and a video that reads as all zeros stops the run.
"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import zarr

from srm_and_sbi_monomer_dimer_alp import artifacts
from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
from srm_and_sbi_monomer_dimer_alp import point_estimate_validation as pev
from srm_and_sbi_monomer_dimer_alp import point_estimate_validation_runner as runner
from srm_and_sbi_monomer_dimer_alp.artifact_schema import INVOCATION_ID_ENV
from srm_and_sbi_monomer_dimer_alp.evaluation import QUANTILE_LEVELS
from srm_and_sbi_monomer_dimer_alp.io import theta_set_schema
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming, to_physical
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow

KEYS = list(det.DETECTOR_PARAMETER_KEYS)
D = len(KEYS)
MU_PC = KEYS.index("mu_pc")


# ---- kernel --------------------------------------------------------------------------------------

def test_stream_seeds_are_deterministic_distinct_and_order_free():
    a = pev.stream_seed(7, 0, 3, 0)
    assert a == pev.stream_seed(7, 0, 3, 0)
    seeds = {pev.stream_seed(7, t, s, k) for t in range(2) for s in range(20) for k in range(6)}
    assert len(seeds) == 2 * 20 * 6                                   # one stream, one seed
    assert pev.stream_seed(8, 0, 3, 0) != a                           # the base seed matters
    assert 0 <= a < 2 ** 63


def test_pilot_selection_is_deterministic_and_covers_both_subgroups_and_their_ends():
    rng = np.random.default_rng(1)
    n = 200
    truth = rng.uniform(-0.5, 0.5, size=(n, D))
    truth[:, MU_PC] = rng.uniform(2.0, 2.75, size=n)
    task = np.repeat([0, 1], n // 2); sim = np.tile(np.arange(n // 2), 2)
    sel = pev.select_pilot(task, sim, truth, KEYS, 10)
    assert np.array_equal(sel, pev.select_pilot(task, sim, truth, KEYS, 10))
    dim = pev.dim_mask(truth, KEYS)
    assert sel.size == 10 and dim[sel].sum() == 5
    for mask in (dim, ~dim):                                         # both ends of each subgroup
        vals = truth[mask, MU_PC]
        chosen = truth[sel[mask[sel]], MU_PC]
        assert chosen.min() == vals.min() and chosen.max() == vals.max()
    try:
        pev.select_pilot(task[:3], sim[:3], truth[:3], KEYS, 10)
    except pev.ValidationError:
        pass
    else:
        raise AssertionError("an undersized subgroup was accepted")


def test_type7_recomputation_matches_numpy_and_detects_a_wrong_stored_value():
    rng = np.random.default_rng(2)
    draws = rng.normal(size=(1000, D)).astype(np.float32)
    q = np.quantile(draws, list(QUANTILE_LEVELS), axis=0).T
    assert np.abs(pev.type7_quantiles(draws, QUANTILE_LEVELS) - q).max() < 1e-6
    arrays = {"repeat_draws": draws[None, None], "repeat_quantiles": q[None, None].astype(float),
              "reference_draws": draws[None], "reference_quantiles": q[None].astype(float)}
    worst, _ = pev.recomputation_max_abs_diff(arrays, QUANTILE_LEVELS)
    assert worst < 1e-6
    arrays["reference_quantiles"] = arrays["reference_quantiles"].copy()
    arrays["reference_quantiles"][0, 2, 2] += 1e-5
    worst, where = pev.recomputation_max_abs_diff(arrays, QUANTILE_LEVELS)
    assert worst > 9e-6 and where == (0, "reference", 2)


def test_accuracy_and_across_repeat_summaries_on_a_hand_example():
    truth = np.zeros((3, 1))
    medians = np.array([[[0.1], [0.3]], [[-0.1], [0.5]], [[0.0], [0.1]]])      # (N=3, R=2, D=1)
    mae, bias, large = pev.accuracy(medians, truth, np.ones(3, bool), large_dex=0.3)
    assert np.allclose(mae, [[0.2 / 3], [0.9 / 3]]) and np.allclose(bias, [[0.0], [0.3]])
    assert np.allclose(large, [[0.0], [1 / 3]])
    s = pev.across_repeats(mae)
    assert np.isclose(s["sd"][0], np.std([0.2 / 3, 0.9 / 3], ddof=1))
    assert np.isclose(s["min"][0], 0.2 / 3) and np.isclose(s["max"][0], 0.3)


def test_per_recording_ratio_uses_sd_over_reference_iqr_and_keeps_zero_iqr_cases():
    rep = np.array([[[1.0, 5.0], [1.2, 5.0], [0.8, 5.0]],
                    [[2.0, 7.0], [2.0, 7.1], [2.0, 6.9]]])          # (N=2, R=3, D=2)
    ref_q = np.zeros((2, 2, 5))
    ref_q[:, :, 1], ref_q[:, :, 3] = 0.0, 2.0                        # IQR 2 everywhere ...
    ref_q[1, :, 3] = 0.0                                             # ... except recording 1: IQR 0
    ratio, sd, iqr, status = pev.per_recording_ratio(rep, ref_q, QUANTILE_LEVELS)
    assert np.isclose(ratio[0, 0], np.std([1.0, 1.2, 0.8], ddof=1) / 2.0) and ratio[0, 1] == 0.0
    assert status[1, 0] == "zero_iqr_zero_sd" and ratio[1, 0] == 0.0
    assert status[1, 1] == "zero_iqr_nonzero_sd" and np.isinf(ratio[1, 1])


def test_gates_apply_the_frozen_thresholds():
    ok = {"recompute": (5e-7, None), "subgroups": {"overall": {
        "mae": {"sd": np.array([0.0009, 0.0011])}, "bias": {"sd": np.array([0.0, 0.0])},
        "mae_ref": np.array([0.1, 0.1]), "bias_ref": np.array([0.0, 0.0]),
        "ratio_share": np.array([0.95, 0.94])}}}
    ok["subgroups"]["overall"]["mae"]["mean"] = np.array([0.1009, 0.1])
    ok["subgroups"]["overall"]["bias"]["mean"] = np.array([0.0, 0.0011])
    g = pev.evaluate_gates(ok)
    assert g["recompute"]
    assert list(g["overall"]["subset_sd"]) == [True, False]
    assert list(g["overall"]["reference_diff"]) == [True, False]
    assert list(g["overall"]["per_recording"]) == [True, False]
    ok["recompute"] = (2e-6, None)
    assert not pev.evaluate_gates(ok)["recompute"]


# ---- runner on a tiny synthetic data bank --------------------------------------------------------

TASKS, SIMS = (0, 1), 6


def _bank(root):
    """Two EVAL tasks of six recordings: theta stores with the real schema attributes, small
    nonzero videos. Returns the frozen ids and log10 truths in task-then-sim order."""
    cfg = detector_workflow()
    paths = cfg.paths.with_condition("FAB")
    timing = RunTiming(total_time_seconds=2.0, frames=PARAMETERS.simulation.timing)
    rng = np.random.default_rng(3)
    lo = np.array([e["PRIOR_RANGE"][0] for e in det.DETECTOR_PARAMETERIZATION])
    hi = np.array([e["PRIOR_RANGE"][1] for e in det.DETECTOR_PARAMETERIZATION])
    truths = []
    for t in TASKS:
        log10 = rng.uniform(lo, hi, size=(SIMS, D))
        log10[:, MU_PC] = np.linspace(2.02, 2.72, SIMS) + 0.001 * t          # dim and bright
        tpath = paths.theta_set_path(t, root, timing.label, True, "EVAL")
        tpath.parent.mkdir(parents=True, exist_ok=True)
        z = zarr.open_array(store=str(tpath), mode="w", shape=(SIMS, D), chunks=(1, D), dtype="f8")
        z[:] = to_physical(log10, det.DETECTOR_PARAMETERIZATION)
        z.attrs.update(theta_set_schema(det.DETECTOR_PARAMETERIZATION, condition="FAB",
                                        timing_label=timing.label, generator="dli_detector"))
        vpath = paths.video_set_path(t, root, timing.label, True, "EVAL")
        vpath.parent.mkdir(parents=True, exist_ok=True)
        v = zarr.open_array(store=str(vpath), mode="w", shape=(SIMS, 4, 8, 8), chunks=(1, 4, 8, 8),
                            dtype="u1")
        v[:] = rng.integers(1, 255, size=(SIMS, 4, 8, 8), dtype=np.uint8)
        truths.append(log10)
    est = paths.estimator_path(root, paths.product_label(timing.label, None))
    est.parent.mkdir(parents=True, exist_ok=True)
    est.write_bytes(b"stand-in estimator bytes")
    return paths, timing


def _stub_summary(posterior, video, device, vista, n, batch, pool_mode="bounded", quantiles=(),
                  return_samples=False, return_sgm=False, sgm_scale=None):
    """Draws depend on the video and on torch's global generator, which the runner seeds per stream."""
    assert pool_mode == "bounded" and return_samples and not return_sgm
    center = float(np.asarray(video).mean()) / 255.0
    draws = (center + 0.05 * torch.randn(n, D, dtype=torch.float64)).numpy().astype(np.float32)
    return np.quantile(draws, list(quantiles), axis=0).T, draws


def _patch(rank=0, world=1):
    saved = (runner.posterior_summary, runner.resolve_topology, artifacts.load_estimator)
    runner.posterior_summary = _stub_summary
    runner.resolve_topology = lambda: SimpleNamespace(
        world_size=world, rank=rank, local_rank=0, device=torch.device("cpu"), backend="CPU",
        is_distributed=world > 1, is_main=rank == 0)
    artifacts.load_estimator = lambda *a, **k: SimpleNamespace(
        weights_sha256="a" * 64, posterior_estimator=SimpleNamespace(to=lambda d: None), prior=None)
    return saved


def _unpatch(saved):
    runner.posterior_summary, runner.resolve_topology, artifacts.load_estimator = saved


def _args(root, out, *extra):
    return runner.build_parser("t").parse_args(
        ["--condition", "FAB", "--total-time-seconds", "2.0", "--tasks", "0", "1", "--repeats", "3",
         "--draws", "40", "--reference-draws", "120", "--data-bank-root", str(root),
         "--output-dir", str(out), *extra])


def _load(out):
    npz = next(Path(out).glob("*_Point_Estimate_Validation_Median.npz"))
    with np.load(str(npz), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def test_runner_writes_a_valid_artifact_and_a_sharded_run_reproduces_it():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "bank"
        _bank(root)
        saved = _patch()
        try:
            assert runner.run_point_estimate_validation(detector_workflow(), _args(root, Path(tmp) / "single")) == 0
        finally:
            _unpatch(saved)
        single = _load(Path(tmp) / "single")
        m = pev.validate_artifact(single)
        assert single["repeat_draws"].shape == (12, 3, 40, D)
        assert single["reference_draws"].shape == (12, 120, D)
        assert m["checkpoint"]["weights_sha256"] == "a" * 64 and len(m["checkpoint"]["file_sha256"]) == 64
        assert m["sampling"]["pool_mode"] == "bounded" and m["parameter_keys"] == KEYS
        assert pev.recomputation_max_abs_diff(single, QUANTILE_LEVELS)[0] <= 1e-6
        assert (Path(tmp) / "single" / "report.md").exists()
        assert (Path(tmp) / "single" / "per_recording_exceedances.csv").exists()
        # two ranks, then --merge: identical arrays, in the frozen order
        os.environ[INVOCATION_ID_ENV] = "inv-test"
        try:
            for r in (0, 1):
                saved = _patch(rank=r, world=2)
                try:
                    runner.run_point_estimate_validation(detector_workflow(), _args(root, Path(tmp) / "shard"))
                finally:
                    _unpatch(saved)
            assert len(list((Path(tmp) / "shard").glob("_shard_*_of_02.npz"))) == 2
            runner.run_point_estimate_validation(detector_workflow(), _args(root, Path(tmp) / "shard", "--merge"))
        finally:
            os.environ.pop(INVOCATION_ID_ENV, None)
        merged = _load(Path(tmp) / "shard")
        pev.validate_artifact(merged)
        assert not list((Path(tmp) / "shard").glob("_shard_*"))                  # shards removed
        for key in ("task_index", "sim_index", "repeat_draws", "reference_draws", "repeat_quantiles",
                    "reference_quantiles", "stream_seeds", "true_log10", "dim_subgroup"):
            assert np.array_equal(single[key], merged[key]), key


def test_pilot_is_deterministic_issues_no_verdict_and_a_zero_video_stops_the_run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "bank"
        paths, timing = _bank(root)
        saved = _patch()
        try:
            runner.run_point_estimate_validation(detector_workflow(), _args(root, Path(tmp) / "pilot", "--pilot", "4"))
            pilot = _load(Path(tmp) / "pilot")
            m = pev.validate_artifact(pilot)
            assert m["observations"]["pilot"] == 4 and pilot["task_index"].size == 4
            assert int(pilot["dim_subgroup"].sum()) == 2
            report = (Path(tmp) / "pilot" / "report.md").read_text()
            assert "no acceptance verdict" in report and "Phase-1 gates" not in report
            # a chunk that reads as zeros (the fill value of a missing chunk) stops the run
            vpath = paths.video_set_path(0, root, timing.label, True, "EVAL")
            v = zarr.open_array(store=str(vpath), mode="r+")
            v[0] = 0
            try:
                runner.run_point_estimate_validation(detector_workflow(), _args(root, Path(tmp) / "z"))
            except SystemExit as exc:
                assert "all zeros" in str(exc)
            else:
                raise AssertionError("an all-zero video was summarized")
        finally:
            _unpatch(saved)


def test_merge_refuses_mixed_runs_and_incomplete_lists():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "bank"
        _bank(root)
        saved = _patch()
        try:
            runner.run_point_estimate_validation(detector_workflow(), _args(root, Path(tmp) / "a"))
        finally:
            _unpatch(saved)
        a = _load(Path(tmp) / "a")
        m = pev.decode_manifest(a["manifest_json"])
        ids = list(zip(a["task_index"].tolist(), a["sim_index"].tolist()))
        half = {k: v[:6] for k, v in a.items() if k != "manifest_json"}
        rest = {k: v[6:] for k, v in a.items() if k != "manifest_json"}
        merged, _ = pev.merge_shards([(rest, m), (half, m)], ids)                   # order restored
        assert np.array_equal(merged["sim_index"], a["sim_index"])
        try:
            pev.merge_shards([(half, m)], ids)
        except pev.ValidationError as exc:
            assert "missing" in str(exc)
        else:
            raise AssertionError("an incomplete list was merged")
        other = dict(m); other["seeds"] = dict(m["seeds"], base_seed=1)
        try:
            pev.merge_shards([(half, m), (rest, other)], ids)
        except pev.ValidationError as exc:
            assert "seeds" in str(exc)
        else:
            raise AssertionError("shards of two runs were merged")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
