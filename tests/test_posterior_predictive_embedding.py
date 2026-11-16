"""The posterior-predictive embedding check, with stand-ins for the simulator and the network.

No ReaDDy run, no estimator, no GPU: the renderer, the embedding and the recordings are replaced by
small deterministic functions, so the three steps run end to end in a temporary folder. Checked: the
draw selection (whole rows, without replacement, the two pools); the plan (every video's full
parameter set, the nuisances drawn inside their boxes, a plan never replaced); the simulation step (the
training data's storage conversion, one file per window, resumable, split by shard); the analysis
(a window the stand-in model reproduces is inside its predictive cloud, a window it cannot produce is
outside; the report, the table and the stored embeddings are written; the EVAL reference is used only
for the estimator it was computed with); the window layout (taken from the source of the posterior
draws, a conflicting option refused); the plan checksum (videos of another plan refused at resume and
at analysis, although their inferred draws are the same); the pixel-level statistics.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
from srm_and_sbi_monomer_dimer_alp import parameterization as bio
from srm_and_sbi_monomer_dimer_alp import posterior_predictive_embedding_runner as ppe
from srm_and_sbi_monomer_dimer_alp.experiment_support import CONDITION_DISPLAY
from srm_and_sbi_monomer_dimer_alp.io import convert_video_dtype
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow

KINDS = list(CONDITION_DISPLAY)[:2]                # FAB, INLB
CONDITION = "FAB"
N_FRAMES = 100                                  # the estimator window of the condition
KEYS = list(det.DETECTOR_PARAMETER_KEYS)
LOW, HIGH = np.array(det.theta_lower_bound()), np.array(det.theta_upper_bound())
CENTER = 0.5 * (LOW + HIGH)
MU_PC = KEYS.index("mu_pc")
CHECKPOINT = "cd" * 32


def test_select_draws_takes_whole_rows_inside_the_box():
    rng = np.random.default_rng(0)
    low, high = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    linear = np.array([False, False, True])
    cloud = rng.uniform(0.1, 0.9, size=(30, 3))
    cloud[0, 0] = 1.5                                   # a log row outside the box
    cloud[1, 2] = 1.2                                   # the linear row outside [0, 1]
    rows, index, eligible = ppe.select_draws(cloud, 10, low, high, linear, "bounded", np.random.default_rng(1))
    assert eligible == 28 and rows.shape == (10, 3) and len(set(index.tolist())) == 10
    assert 0 not in index and 1 not in index
    assert np.array_equal(rows, cloud[index])            # complete rows, never recombined
    rows_u, index_u, eligible_u = ppe.select_draws(cloud, 29, low, high, linear, "unrestricted",
                                                   np.random.default_rng(1))
    assert eligible_u == 29 and 0 in index_u and 1 not in index_u
    few, _, n = ppe.select_draws(cloud, 100, low, high, linear, "bounded", np.random.default_rng(1))
    assert n == 28 and few.shape[0] == 28                # fewer than asked: the caller decides
    try:
        ppe.select_draws(cloud, 5, low, high, linear, "everything", np.random.default_rng(1))
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown pool was accepted")


def test_window_reading_names_the_four_cases():
    assert ppe.window_reading(0.4, 0.5, 0.05) == "inside both"
    assert ppe.window_reading(0.01, 0.5, 0.05) == "inside the EVAL cloud, outside its predictive cloud"
    assert ppe.window_reading(0.4, 0.01, 0.05) == "outside the EVAL cloud, inside its predictive cloud"
    assert ppe.window_reading(0.01, 0.01, 0.05) == "outside both"
    assert ppe.window_reading(0.4, None, 0.05) == "inside its predictive cloud"
    assert ppe.window_reading(0.01, float("nan"), 0.05) == "outside its predictive cloud"


def _collection(path, rng, recordings=2, windows=2, draws=120, step_frames=None, span_frames=None):
    """A retained draw collection: tight clouds around the box center, a few draws outside the box."""
    n = len(KINDS) * recordings * windows
    cloud = (CENTER + 0.02 * (HIGH - LOW) * rng.normal(size=(n, draws, len(KEYS)))).astype(np.float32)
    cloud[:, :3, 0] = HIGH[0] + 0.5                              # three draws per window outside the box
    kind_index = np.repeat(np.arange(len(KINDS)), recordings * windows)
    cell = np.tile(np.repeat(np.arange(1, recordings + 1), windows), len(KINDS))
    chunk = np.tile(np.arange(windows), len(KINDS) * recordings)
    sgm_index = np.full(n, 10, dtype=np.int64)
    scale = (HIGH - LOW).tolist()
    manifest = {"parameter_keys": KEYS, "coordinate_transform": "log10", "sgm_scale": scale,
                "pool_mode": "unrestricted", "n_summary_draws": draws, "checkpoint_sha256": CHECKPOINT,
                "window_geometry": {"n_frames": N_FRAMES, "step_frames": step_frames or N_FRAMES,
                                    "span_frames": span_frames}, "collection_label": "TEST"}
    np.savez_compressed(str(path), posterior_samples_cloud=cloud, kind_index=kind_index, cell=cell, chunk=chunk,
                        kinds=np.array(KINDS), fresh_sgm=cloud[np.arange(n), sgm_index].astype(float),
                        fresh_sgm_index=sgm_index, parameter_keys=np.array(KEYS), sgm_scale=np.array(scale),
                        collection_manifest_json=json.dumps(manifest))


def _level(mu_pc):
    """The stand-in model: the pixel level a dye brightness gives (16-bit counts)."""
    return 40.0 * float(mu_pc)


def _renderer(condition, rds_vector, imaging_physical, labeling_plan, n_frames, trajectory_path):
    assert condition == CONDITION and len(imaging_physical) == len(det.DETECTOR_IMAGING_KEYS)
    assert len(rds_vector) == len(bio.parameter_keys_of(CONDITION))
    rng = np.random.default_rng(int(1e6 * float(imaging_physical[0])) % (2 ** 31))
    Path(trajectory_path).write_bytes(b"trajectory")                # the step removes its scratch file
    synth = _level(imaging_physical[MU_PC]) + 60.0 * rng.normal(size=(n_frames, 8, 8))
    return {"synth": synth, "dye_counts": np.array([1, 2]), "n_subunits": 3}


def _embedder(videos):
    arr = np.asarray(videos, dtype=float)
    return np.stack([arr.mean(axis=(1, 2, 3)), arr.std(axis=(1, 2, 3)), np.median(arr, axis=(1, 2, 3))], axis=1)


def _window_reader(kind, cell):
    """Recordings of the first group follow the stand-in model at the posterior's center; those of the
    second are far brighter than any video the posterior produces."""
    rng = np.random.default_rng(100 * KINDS.index(kind) + cell)
    mu_pc = 10.0 ** CENTER[MU_PC] * (1.0 if kind == KINDS[0] else 4.0)
    return [convert_video_dtype(_level(mu_pc) + 60.0 * rng.normal(size=(N_FRAMES, 8, 8)), bits_from=16, bits_to=8)
            for _ in range(2)]


def _setup(tmp, extra=()):
    source = Path(tmp) / "draws.npz"
    if not source.exists():
        _collection(source, np.random.default_rng(2))
    argv = ["--step", "plan", "--condition", CONDITION, "--total-time-seconds", "2.0", "--draws-source", str(source),
            "--kinds", ",".join(KINDS),
            "--draws-per-window", "40", "--knn-k-cloud", "5", "--knn-k", "5", *extra]
    args = ppe.build_parser("test").parse_args(argv)
    cfg = detector_workflow()
    R = ppe._spec(cfg, args)
    out = Path(tmp) / "out"
    R.update(out_dir=out, plan_path=out / "plan.npz", video_dir=out / "videos",
             reference_embeddings=Path(tmp) / "reference.npz")
    return cfg, args, R


def test_plan_fixes_every_parameter_of_every_video():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, args, R = _setup(tmp)
        assert ppe.run_plan(cfg, args, R) == 0
        plan = ppe.load_plan(R["plan_path"])
        theta = plan["theta_estimator"]
        assert theta.shape == (8, 40, 6) and plan["imaging_physical"].shape == (8, 40, 11)
        assert plan["kind"].tolist() == [KINDS[0]] * 4 + [KINDS[1]] * 4
        assert plan["cell"].tolist() == [1, 1, 2, 2] * 2 and plan["chunk"].tolist() == [0, 1] * 4
        assert (theta >= LOW - 1e-9).all() and (theta <= HIGH + 1e-9).all()      # bounded pool
        assert plan["n_eligible"].tolist() == [117] * 8                          # 120 minus the three outside
        assert (plan["row_index"] >= 3).all()
        assert np.allclose(plan["imaging_physical"][..., :6], 10.0 ** theta)     # the posterior draw itself
        scope = np.log10(plan["imaging_physical"][..., 6:])
        assert (scope >= np.array(det.scope_lower_bound()) - 1e-9).all()
        assert (scope <= np.array(det.scope_upper_bound()) + 1e-9).all()
        rds = plan["rds_physical"]
        low = np.asarray(bio.to_physical(np.array(bio.theta_lower_bound(CONDITION)), bio.parameterization_of(CONDITION)))
        high = np.asarray(bio.to_physical(np.array(bio.theta_upper_bound(CONDITION)), bio.parameterization_of(CONDITION)))
        assert rds.shape == (8, 40, len(low)) and (rds >= low * (1 - 1e-9)).all() and (rds <= high * (1 + 1e-9)).all()
        assert len({tuple(v) for v in rds.reshape(-1, len(low)).round(12).tolist()}) == 320   # a fresh nuisance per video
        man = plan["manifest"]
        assert man["checkpoint_sha256"] == CHECKPOINT and man["pool"] == "bounded" and man["n_frames"] == N_FRAMES
        assert man["workflow"] == "detector" and man["draws_source"]["sha256"] == ppe._sha256(R["draws_source"])
        try:
            ppe.run_plan(cfg, args, R)
        except SystemExit as exc:
            assert "never replaced" in str(exc)
        else:
            raise AssertionError("an existing plan was replaced")


def _refused(fn, needle):
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(f"not refused (expected a message with {needle!r})")


def test_plan_follows_the_window_layout_of_the_draws():
    """The posteriors of the source were computed on windows of a recorded step; the plan takes that
    step whatever this run's default is, and refuses an option that contradicts it."""
    half = N_FRAMES // 2                                         # windows stepped by one second
    span = 2 * ppe.DEFAULT_SPAN_SECONDS                          # recordings twice the project's default length
    with tempfile.TemporaryDirectory() as tmp:
        _collection(Path(tmp) / "draws.npz", np.random.default_rng(2), step_frames=half,
                    span_frames=span * half)
        cfg, args, R = _setup(tmp)
        assert ppe.run_plan(cfg, args, R) == 0
        man = ppe.load_plan(R["plan_path"])["manifest"]
        assert (man["n_frames"], man["step_frames"], man["span_frames"]) == (N_FRAMES, half, span * half)
        assert args.experiment_span_seconds is None and man["experiment_span_seconds"] == span    # the stored span
        contract = {"window_geometry": {"n_frames": N_FRAMES, "step_frames": half, "span_frames": span * half}}
        assert ppe.plan_geometry(contract, R["timing"], 1, span) == (N_FRAMES, half, span * half, span)   # agreeing
        assert ppe.plan_geometry(contract, R["timing"], None, None) == (N_FRAMES, half, span * half, span)
        odd = {"window_geometry": {"n_frames": N_FRAMES, "step_frames": half, "span_frames": span * half + 1}}
        _refused(lambda: ppe.plan_geometry(odd, R["timing"], None, None), "whole number of seconds")
        bare = {"window_geometry": {"n_frames": N_FRAMES, "step_frames": half}}
        assert ppe.plan_geometry(bare, R["timing"], None, None)[2:] == (None, ppe.DEFAULT_SPAN_SECONDS)
        _refused(lambda: ppe.plan_geometry(contract, R["timing"], 2, span), "stepped by")
        _refused(lambda: ppe.plan_geometry(contract, R["timing"], None, span - 1), "recordings of")
        other = {"window_geometry": {"n_frames": N_FRAMES + 1, "step_frames": half}}
        _refused(lambda: ppe.plan_geometry(other, R["timing"], None, span), "the estimator of this run reads")
        _refused(lambda: ppe.plan_geometry({}, R["timing"], None, span), "no window geometry")
    with tempfile.TemporaryDirectory() as tmp:                   # draws on non-overlapping windows
        cfg, args, R = _setup(tmp, extra=["--chunk-step-seconds", "1"])
        _refused(lambda: ppe.run_plan(cfg, args, R), "stepped by")
        assert not R["plan_path"].exists()
        cfg, args, R = _setup(tmp)
        ppe.run_plan(cfg, args, R)
        assert ppe.load_plan(R["plan_path"])["manifest"]["step_frames"] == N_FRAMES


def test_videos_of_another_plan_are_refused():
    """A plan with the same inferred draws but other nuisance draws is another plan: its checksum
    differs, and videos simulated from the first are refused at resume and at analysis."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, args, R = _setup(tmp)
        ppe.run_plan(cfg, args, R)
        ppe.run_simulate(cfg, args, R, renderer=_renderer)
        with np.load(R["plan_path"], allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["rds_physical"] = arrays["rds_physical"] * 1.0001                 # other nuisance draws
        np.savez_compressed(str(R["plan_path"]), **arrays)
        assert np.array_equal(ppe.load_plan(R["plan_path"])["theta_estimator"], arrays["theta_estimator"])
        _refused(lambda: ppe.run_simulate(cfg, args, R, renderer=_renderer), "another plan")
        _refused(lambda: ppe.run_analyze(cfg, args, R, embedder=_embedder, window_reader=_window_reader),
                 "another plan")


def test_pixel_statistics_are_the_stated_summaries():
    video = np.full((1, 4, 10, 10), 10, dtype=np.uint8)
    video[0, 1::2] = 30                                          # frame means 10, 30, 10, 30
    video[0, 0, 0, :4] = 255                                     # four clipped pixels of 400
    stats = ppe.pixel_statistics(video)
    assert stats.shape == (1, len(ppe.PIXEL_STATISTICS))
    median, _, _, maximum, clipped, variation = stats[0]
    assert median == 30.0 and maximum == 255.0 and abs(clipped - 0.01) < 1e-12      # 196 tens, 200 thirties
    means = np.array([10 + 4 * 245 / 100.0, 30.0, 10.0, 30.0])
    assert abs(variation - means.std() / means.mean()) < 1e-12
    cloud = np.tile(np.arange(1.0, 101.0)[:, None], (1, len(ppe.PIXEL_STATISTICS)))   # values 1 .. 100
    low, mid, high, below = ppe.pixel_comparison(np.full(len(ppe.PIXEL_STATISTICS), 50.5), cloud)
    assert abs(low[0] - 5.95) < 1e-9 and abs(mid[0] - 50.5) < 1e-9 and abs(high[0] - 95.05) < 1e-9
    assert np.allclose(below, 0.5)


def test_plan_refuses_a_window_with_too_few_admitted_draws():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, args, R = _setup(tmp, extra=["--draws-per-window", "118"])
        try:
            ppe.run_plan(cfg, args, R)
        except SystemExit as exc:
            assert "only 117 of 120" in str(exc), str(exc)
        else:
            raise AssertionError("118 draws were taken from 117 admitted ones")
        assert not R["plan_path"].exists()


def test_simulate_stores_like_the_training_data_and_resumes():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, args, R = _setup(tmp)
        ppe.run_plan(cfg, args, R)
        plan = ppe.load_plan(R["plan_path"])
        args.shard = "0/2"
        ppe.run_simulate(cfg, args, R, renderer=_renderer)
        assert len(list(R["video_dir"].glob("Window_*.npz"))) == 4               # windows 0, 2, 4, 6
        calls = []

        def counting(*a):
            calls.append(1)
            return _renderer(*a)
        args.shard = None
        ppe.run_simulate(cfg, args, R, renderer=counting)
        assert len(calls) == 4 * 40                                              # only the four missing windows
        files = sorted(R["video_dir"].glob("Window_*.npz"))
        assert len(files) == 8 and not list(R["video_dir"].glob("*.partial.npz"))
        assert not list(R["video_dir"].glob("trajectories_*"))                   # scratch removed
        first = ppe.window_video_path(R["video_dir"], plan["kind"][0], plan["cell"][0], plan["chunk"][0])
        with np.load(first) as data:
            assert data["videos"].shape == (40, N_FRAMES, 8, 8) and data["videos"].dtype == np.uint8
            expected = convert_video_dtype(
                _renderer(CONDITION, plan["rds_physical"][0, 0], plan["imaging_physical"][0, 0], None, N_FRAMES,
                          Path(tmp) / "t.h5")["synth"], bits_from=16, bits_to=8)
            assert np.array_equal(data["videos"][0], expected)                   # the storage step, bit for bit
            assert np.array_equal(data["theta_estimator"], plan["theta_estimator"][0])
        calls.clear()
        ppe.run_simulate(cfg, args, R, renderer=counting)
        assert not calls                                                         # nothing is simulated twice


def test_analyze_separates_reproduced_windows_from_the_others():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, args, R = _setup(tmp)
        ppe.run_plan(cfg, args, R)
        ppe.run_simulate(cfg, args, R, renderer=_renderer)
        try:                                                                     # one window missing: refused
            first = sorted(R["video_dir"].glob("Window_*.npz"))[0]
            held = first.with_suffix(".held")
            first.rename(held)
            ppe.run_analyze(cfg, args, R, embedder=_embedder, window_reader=_window_reader)
        except SystemExit as exc:
            assert "missing" in str(exc)
            held.rename(first)
        else:
            raise AssertionError("an incomplete set of windows was analyzed without --allow-partial")
        assert ppe.run_analyze(cfg, args, R, embedder=_embedder, window_reader=_window_reader) == 0
        with np.load(R["out_dir"] / "embeddings.npz") as data:
            readings, kinds = data["readings"].tolist(), data["kind"].tolist()
            assert data["predictive"].shape == (8, 40, 3) and data["experimental"].shape == (8, 3)
        inside = [r for r, k in zip(readings, kinds) if k == KINDS[0]]
        outside = [r for r, k in zip(readings, kinds) if k == KINDS[1]]
        assert outside == ["outside its predictive cloud"] * 4, outside
        assert inside.count("inside its predictive cloud") >= 3, inside
        text = (R["out_dir"] / "report.md").read_text()
        assert "by group" in text and "by window" in text and "none (no stored embeddings" in text
        for needle in ("not calibrated p-values", "Posterior draws excluded by the pool", "Pixel-level comparison",
                       "does not by itself indicate a fault", "averaged over that distribution"):
            assert needle in text, needle
        assert "expected if the model reproduced" not in text and "points at the predictive" not in text
        pixels = (R["out_dir"] / "window_pixel_statistics.csv").read_text().splitlines()
        assert len(pixels) == 1 + 8 * len(ppe.PIXEL_STATISTICS)
        bright = [row.split(",") for row in pixels[1:] if row.split(",")[3] == "pixel median"][4:]
        assert [row[-1] for row in bright] == ["outside"] * 4, bright            # the four bright windows
        with np.load(R["out_dir"] / "embeddings.npz") as data:
            assert str(data["plan_sha256"]) == ppe._sha256(R["plan_path"])
            assert np.allclose(data["excluded_share"], 3.0 / 120.0)
            assert data["predictive_pixel_statistics"].shape == (8, 40, len(ppe.PIXEL_STATISTICS))
        rows = (R["out_dir"] / "window_predictive_typicality.csv").read_text().splitlines()
        assert len(rows) == 9 and rows[0].count(",") == len(ppe.WINDOW_HEADERS) - 1


def test_analyze_uses_an_eval_reference_only_of_the_same_estimator():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, args, R = _setup(tmp)
        ppe.run_plan(cfg, args, R)
        ppe.run_simulate(cfg, args, R, renderer=_renderer)
        # A reference cloud around the reproduced windows' level: the bright windows lie outside it too.
        level = _embedder(np.stack(_window_reader(KINDS[0], 1)))[0]
        reference = level + np.random.default_rng(3).normal(size=(300, 3)) * np.array([1.5, 0.3, 1.5])
        np.savez_compressed(str(R["reference_embeddings"]), synthetic=reference, estimator_sha256="")
        ppe.run_analyze(cfg, args, R, embedder=_embedder, window_reader=_window_reader)
        with np.load(R["out_dir"] / "embeddings.npz") as data:
            readings, kinds = data["readings"].tolist(), data["kind"].tolist()
        assert all(r == "outside both" for r, k in zip(readings, kinds) if k == KINDS[1]), readings
        assert all(r == "inside both" for r, k in zip(readings, kinds) if k == KINDS[0]), readings
        np.savez_compressed(str(R["reference_embeddings"]), synthetic=reference, estimator_sha256="ef" * 32)
        try:
            ppe.run_analyze(cfg, args, R, embedder=_embedder, window_reader=_window_reader)
        except SystemExit as exc:
            assert "another estimator" in str(exc)
        else:
            raise AssertionError("a reference cloud of another estimator was used")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
