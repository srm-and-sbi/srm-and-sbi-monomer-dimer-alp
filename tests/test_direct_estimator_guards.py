"""Guards of the direct-estimator utilities: purpose, run folders, harness contrasts, observables.

Small regression checks, one per defect a review of the development tooling found. A tier run must
declare its purpose and is held to the declared split of the EVAL tiers before it reads a recording
(DETECTOR_WORKFLOW.md sec. 9.6); a run never reuses a run folder, and the flicker-harness variants resolve to different
folders; the harness's linking contrast reads the same detections under both groupings; the PSF
observable `track_length_median` describes the tracks the population estimate keeps, and
`tracks_per_spot` reads 2 for two unfragmented tracks; the selftest decomposition table shows every
additive term; a run whose implementation changed while it executed is invalid for acceptance.

Runnable with ``python -m pytest`` or directly:
``MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_direct_estimator_guards.py``.
"""
import contextlib
import functools
import importlib.util
import io
import pathlib
import tempfile

import numpy as np

from srm_and_sbi_monomer_dimer_alp import direct_acceptance as da
from srm_and_sbi_monomer_dimer_alp import direct_imaging_estimates as die
from srm_and_sbi_monomer_dimer_alp import provenance as prov

REPO = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "Script_Bank" / "Analysis"


@functools.lru_cache(maxsize=None)
def _load(name):
    path = ANALYSIS / f"SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{name}.py"
    spec = importlib.util.spec_from_file_location(name.lower(), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _purpose(purpose, estimator="Direct_PSF_Width", condition="FAB", n_frames=100, split="EVAL",
             tasks=(2, 3), max_videos=10 ** 8, expect=1000):
    return da.check_run_purpose(purpose, estimator=estimator, condition=condition, n_frames=n_frames,
                                split=split, tasks=list(tasks), max_videos=max_videos,
                                expect_videos_per_task=expect)


def _refused(fn, *words):
    try:
        fn()
    except ValueError as exc:
        for w in words:
            assert w in str(exc), f"{w!r} not in the refusal: {exc}"
        return
    raise AssertionError("expected a refusal")


def _cli_refusal(mod, argv):
    """Run a utility's main() and return (exit status, stderr); it must refuse."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            mod.main(argv)
        except SystemExit as exc:
            return exc.code, err.getvalue()
    raise AssertionError(f"{argv} was not refused")


# ---- purpose and the declared split --------------------------------------------------------

def test_development_run_refuses_reserved_tasks():
    rec = _purpose("development", tasks=range(0, 20))
    assert rec["split_declared"] and rec["reserved_tasks"] == [20, 21, 22, 23, 24]
    _refused(lambda: _purpose("development", tasks=(2, 3, 20)), "[20]", "reserved")
    _refused(lambda: _purpose("development", estimator="Direct_Fluorescence_Loss", n_frames=1000,
                              tasks=(9, 10), expect=100), "[10]", "reserved")
    assert _purpose("development", estimator="Direct_Fluorescence_Loss", n_frames=1000,
                    tasks=range(10), expect=100)["tasks"] == list(range(10))


def test_verdict_run_reads_exactly_the_reserved_tasks():
    assert _purpose("verdict", tasks=range(20, 25))["tasks"] == [20, 21, 22, 23, 24]
    _purpose("verdict", estimator="Direct_Flicker_Rate", tasks=range(20, 25))
    _purpose("verdict", estimator="Direct_Fluorescence_Loss", n_frames=1000, tasks=range(10, 20), expect=100)
    _refused(lambda: _purpose("verdict", tasks=range(20, 24)), "exactly the reserved tasks")
    _refused(lambda: _purpose("verdict", estimator="Direct_Fluorescence_Loss", tasks=range(20, 25)),
             "kept for the verdict")
    _refused(lambda: _purpose("verdict", n_frames=1000, tasks=range(10, 20), expect=100),
             "kept for the verdict")
    _refused(lambda: _purpose("verdict", tasks=range(20, 25), max_videos=200), "cut the verdict short")
    _refused(lambda: _purpose("verdict", split="TEST", tasks=range(20, 25)), "holds none")
    _refused(lambda: _purpose("verdict", condition="INLB", tasks=range(20, 25)), "no reserved set")


def test_undeclared_tiers_and_other_splits_hold_no_reserved_task():
    rec = _purpose("development", condition="INLB", tasks=(0, 21))
    assert not rec["split_declared"] and rec["reserved_tasks"] is None
    assert not _purpose("development", n_frames=250, tasks=(0, 24))["split_declared"]
    assert not _purpose("development", split="TRAIN", tasks=(0, 21, 150))["split_declared"]


def test_each_task_is_read_once():
    _refused(lambda: _purpose("development", tasks=(2, 2)), "read once")


def test_tier_run_refuses_a_reserved_task_before_reading():
    for name in ("Direct_PSF_Width", "Direct_Flicker_Rate"):
        code, err = _cli_refusal(_load(name), ["--condition", "FAB", "--total-time-seconds", "2.0",
                                               "--tasks", "2", "20", "--purpose", "development",
                                               "--dry-run"])
        assert code == 2 and "reserved for a verdict" in err, (name, err)
    code, err = _cli_refusal(_load("Direct_Fluorescence_Loss"),
                             ["--condition", "FAB", "--total-time-seconds", "20.0", "--tasks", "10",
                              "--expect-videos-per-task", "100", "--purpose", "development", "--dry-run"])
    assert code == 2 and "reserved for a verdict" in err, err


def test_tier_run_requires_a_purpose():
    for name in ("Direct_PSF_Width", "Direct_Flicker_Rate", "Direct_Fluorescence_Loss"):
        code, err = _cli_refusal(_load(name), ["--condition", "FAB", "--total-time-seconds", "2.0",
                                               "--tasks", "2", "--dry-run"])
        assert code == 2 and "--purpose is required" in err, (name, err)


def test_a_run_without_eval_tasks_refuses_a_purpose():
    for name in ("Direct_PSF_Width", "Direct_Flicker_Rate", "Direct_Fluorescence_Loss"):
        code, err = _cli_refusal(_load(name), ["--selftest", "--purpose", "development", "--dry-run"])
        assert code == 2 and "applies to a tier run" in err, (name, err)
    code, err = _cli_refusal(_load("Direct_PSF_Width"), ["--experiment", "--condition", "FAB",
                                                         "--total-time-seconds", "2.0",
                                                         "--purpose", "verdict", "--dry-run"])
    assert code == 2 and "applies to a tier run" in err, err


def test_purpose_token_is_not_repeated_in_the_suffix():
    code, err = _cli_refusal(_load("Direct_PSF_Width"),
                             ["--selftest", "--run-suffix", "DEV_a148931", "--dry-run"])
    assert code == 2 and "added automatically" in err, err


# ---- run folders ---------------------------------------------------------------------------

def test_run_folder_is_never_reused():
    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "Posit" / "RUN_DEV_a148931"
        prov.reserve_run_folder(target)
        assert target.is_dir()
        try:
            prov.reserve_run_folder(target)
        except FileExistsError:
            return
    raise AssertionError("an existing run folder was accepted")


def test_harness_variants_resolve_to_different_folders():
    hm = _load("Direct_Flicker_Mismatch")
    names = {hm.run_descriptor("FAB", hm.BLEACH_OFF), hm.run_descriptor("single", hm.BLEACH_OFF),
             hm.run_descriptor("FAB", 0.056), hm.run_descriptor("INLB", hm.BLEACH_OFF)}
    assert len(names) == 4, names
    assert hm.run_descriptor("FAB", 0.056) == "Direct_Flicker_Mismatch_FAB_LAW_BLEACH_0p056"
    assert hm.run_descriptor("FAB", hm.BLEACH_OFF, "a148931") == "Direct_Flicker_Mismatch_FAB_LAW_a148931"


def test_every_utility_refuses_an_existing_run_folder_before_computing():
    with tempfile.TemporaryDirectory() as tmp:
        for name, extra in (("Direct_Flicker_Mismatch", []), ("Direct_PSF_Width", ["--selftest"]),
                            ("Direct_Flicker_Rate", ["--selftest"]),
                            ("Direct_Fluorescence_Loss", ["--selftest"])):
            code, err = _cli_refusal(_load(name), extra + ["--out-dir", tmp, "--dry-run"])
            assert code == 2 and "exists already" in err, (name, err)


# ---- flicker harness: the linking contrast reads the same detections -------------------------

def _harness_case(n_frames=60):
    """Two visible subunits detected in every frame; an emitter with no visible subunit nearby
    detected in frames 10-20 (unmatched); a second fit beside subunit 0 in frame 30."""
    true_xy = np.zeros((n_frames, 2, 2))
    true_xy[:, 0], true_xy[:, 1] = (20.0, 20.0), (60.0, 60.0)
    rng = np.random.default_rng(7)
    rows = [(t, true_xy[t, s, 0] + 0.1, true_xy[t, s, 1], 100.0 + rng.random())
            for t in range(n_frames) for s in (0, 1)]
    rows += [(t, 120.0, 120.0, 50.0 + rng.random()) for t in range(10, 21)]
    rows += [(30, 20.9, 20.0, 66.0)]
    rows.sort(key=lambda r: r[0])
    f, x, y, a = (np.asarray(v) for v in zip(*rows))
    m = dict(frame_index=f.astype(np.int64), x=x, y=y, amplitude=a, sqrt2sigma=np.ones(f.size))
    return m, true_xy, np.array([True, True])


def _points(traces):
    return sorted((int(fr), float(am)) for f, a in traces for fr, am in zip(f, a))


def test_linking_contrast_reads_the_same_detections():
    hm = _load("Direct_Flicker_Mismatch")
    m, true_xy, visible = _harness_case()
    matched = hm.match_to_subunits(m, true_xy, visible)
    assert (matched == hm.UNMATCHED).sum() == 11 and (matched == hm.SECOND_FIT).sum() == 1
    tid = die.link_spot_tracks(m["frame_index"], m["x"], m["y"], frame_stride=1)
    scene = dict(spot_photons=np.zeros((60, 2)), dye_photons=np.zeros((60, 2)), visible=visible)
    lv = {level: hm.level_traces(level, scene, m, tid, matched, 5)
          for level in ("fitted_oracle", "fitted_linked", "production")}
    assert len(lv["fitted_oracle"]) == len(lv["fitted_linked"]) == 2
    assert _points(lv["fitted_oracle"]) == _points(lv["fitted_linked"])
    # production adds the unmatched emitter's track: detections outside the matched set, which the
    # linking contrast no longer mixes in
    assert len(lv["production"]) == 3
    extra = set(_points(lv["production"])) - set(_points(lv["fitted_linked"]))
    assert len(extra) == 11 and all(10 <= fr <= 20 for fr, _ in extra)


def test_harness_records_why_a_level_has_no_estimate():
    hm = _load("Direct_Flicker_Mismatch")
    few = [(np.arange(50), np.full(50, 100.0))] * 3
    r = hm.estimate_level(few, n_model_traces=10)
    assert r["reason"] == "too_few_traces" and not np.isfinite(r["lambda_rate"])
    ok = dict(lambda_rate=2.0, n_traces=9, span_median=60.0, refined=True, at_grid_edge=False, reason=None)
    bad = dict(lambda_rate=float("nan"), n_traces=3, span_median=float("nan"), refined=False,
               at_grid_edge=False, reason="too_few_traces")
    results = [dict(lambda_rate=2.0, mu_pc=120.0, levels=[ok] * (len(hm.LEVELS) - 1) + [bad]),
               dict(lambda_rate=2.0, mu_pc=120.0, levels=[ok] * len(hm.LEVELS))]
    c = hm.collect_levels(results)
    assert c["reason"][0, -1] == "too_few_traces" and c["reason"][1, -1] == ""
    last = hm.step_rows(c["err"])[-1]
    assert last[-1] == "1 of 2"                          # the failed scene leaves the pair, visibly
    assert hm.mean_cell(c["err"][:, -1], np.ones(2, bool)).endswith("(1)")


# ---- PSF observables ------------------------------------------------------------------------

def _psf_measurements(tracks):
    """Spot fits from ``tracks`` = [(n_fits, n_good), ...]: consecutive frames, the first n_good fits
    with a small relative error and the rest with one above the population filter's 0.5."""
    frame, x, w, se, tid = [], [], [], [], []
    for k, (n_fits, n_good) in enumerate(tracks):
        for i in range(n_fits):
            frame.append(i); x.append(10.0 * k); w.append(1.4 + 0.01 * k)
            se.append(0.05 if i < n_good else 1.3); tid.append(k)
    n = len(frame)
    m = dict(frame_index=np.asarray(frame), x=np.asarray(x), y=np.zeros(n), sqrt2sigma=np.asarray(w),
             sqrt2sigma_se=np.asarray(se), amplitude=np.full(n, 500.0),
             n_frames_used=max(nf for nf, _ in tracks))
    return m, np.asarray(tid)


def test_track_length_median_describes_the_retained_tracks():
    psf = _load("Direct_PSF_Width")
    # nine tracks of five good fits; nine of thirty fits with four good ones, which the length floor
    # drops once the relative-error filter has removed their poor fits
    m, tid = _psf_measurements([(5, 5)] * 9 + [(30, 4)] * 9)
    r = die.psf_width_population(m["sqrt2sigma"], m["sqrt2sigma_se"], track_id=tid, min_track_length=5)
    assert r["n_tracks"] == 9 and list(r["track_lengths"]) == [5] * 9
    obs = psf.recording_observables(m, psf._scope_center(), track_lengths=r["track_lengths"],
                                    n_tracks=r["n_tracks"])
    assert obs["track_length_median"] == 5.0            # counting every fit would give 17.5


def test_tracks_per_spot_reads_two_for_two_unfragmented_tracks():
    """The review's counterexample: two emitters, each seen in a different half of the recording and
    perfectly linked. No track is fragmented, and the observable reads 2."""
    psf = _load("Direct_PSF_Width")
    m = dict(frame_index=np.arange(100), x=np.zeros(100), y=np.zeros(100),
             sqrt2sigma=np.full(100, 1.4), sqrt2sigma_se=np.full(100, 0.05),
             amplitude=np.full(100, 500.0), n_frames_used=100)
    obs = psf.recording_observables(m, psf._scope_center(), track_lengths=np.array([50, 50]), n_tracks=2)
    assert obs["spots_per_frame"] == 1.0 and obs["tracks_per_spot"] == 2.0


# ---- PSF selftest decomposition --------------------------------------------------------------

def test_decomposition_terms_sum_to_the_total():
    psf = _load("Direct_PSF_Width")
    n_sub, n_frames = 12, 8
    true_w = np.linspace(1.2, 1.7, n_sub)
    xy = np.zeros((n_frames, n_sub, 2))
    xy[:, :, 0] = np.arange(n_sub)[None, :] * 10.0
    frame, sub = np.repeat(np.arange(n_frames), n_sub), np.tile(np.arange(n_sub), n_frames)
    m = dict(frame_index=frame, x=xy[frame, sub, 0] + 0.2, y=xy[frame, sub, 1],
             sqrt2sigma=true_w[sub] * 1.02, sqrt2sigma_se=np.full(frame.size, 0.05))
    dec = psf.truth_decomposition(m, sub, true_w, xy, np.ones(n_sub, bool), 1.30, 1.47,
                                  min_track_length=5)
    assert abs(sum(dec[k] for k in psf.DECOMPOSITION_TERMS) - dec["total"]) < 1e-12
    assert dec["sample"] > 0.03                          # a real finite-population term


def test_decomposition_table_shows_every_additive_term():
    psf = _load("Direct_PSF_Width")
    col = {k: j for j, k in enumerate(psf.DECOMPOSITION_KEYS)}
    dec = np.zeros((1, len(psf.DECOMPOSITION_KEYS)))
    dec[0, col["total"]] = dec[0, col["sample"]] = 0.301      # the review's case: all of it sample
    dec[0, col["detected_share"]], dec[0, col["tracks_per_subunit"]], dec[0, col["spread_ratio"]] = 0.9, 1.1, 1.0
    headers, rows = psf.decomposition_table(dec, [dict(mu_r=1.41)])
    assert set(psf.DECOMPOSITION_TERMS) <= set(headers)
    for row in rows:
        shown = dict(zip(headers, row))
        terms = sum(float(shown[k]) for k in psf.DECOMPOSITION_TERMS)
        assert abs(terms - float(shown["total"])) <= 1e-4 * (len(psf.DECOMPOSITION_TERMS) + 1), row


# ---- a changed implementation invalidates acceptance -----------------------------------------

def _code_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "kernel.py").write_text("x = 1\n")
        startup = prov.code_provenance(root, files=("kernel.py",))
        same = prov.finalize_code_provenance(startup, root, files=("kernel.py",))
        (root / "kernel.py").write_text("x = 2\n")
        changed = prov.finalize_code_provenance(startup, root, files=("kernel.py",))
    return same, changed


def test_changed_implementation_invalidates_the_run():
    same, changed = _code_blocks()
    assert not same["changed_during_run"] and changed["changed_during_run"]
    for base_exit in (0, 1, 2):
        res = dict(verdicts={"accuracy": "PASS"}, exit_code=base_exit)
        da.apply_code_provenance(res, same)
        assert res["exit_code"] == base_exit and res["valid_for_acceptance"]
        assert res["verdicts"]["implementation"].startswith("PASS")
        res = dict(verdicts={"accuracy": "PASS"}, exit_code=base_exit)
        da.apply_code_provenance(res, changed)
        assert res["exit_code"] == da.EXIT_IMPLEMENTATION_CHANGED and not res["valid_for_acceptance"]
        assert res["verdicts"]["implementation"].startswith("INVALID")
        assert res["verdicts"]["accuracy"] == "PASS"             # kept, as a diagnostic


def test_verdict_table_leads_with_the_implementation_check():
    class Reporter:
        def __init__(self):
            self.tables = []

        def table(self, title, headers, rows, note=""):
            self.tables.append((title, rows))

    _, changed = _code_blocks()
    res = da.evaluate(np.full((3, 6), 10.0), np.ones(3, bool), [None] * 3, lambda mask: {},
                      target_key="mu_r", selftest=True)
    da.apply_code_provenance(res, changed)
    rep = Reporter()
    da.render(rep, res, estimator="test", target_key="mu_r")
    title, rows = rep.tables[0]
    assert title.startswith("Verdicts") and rows[0][0].startswith("0. implementation")
    assert rows[0][1].startswith("INVALID")


if __name__ == "__main__":
    import sys
    import time
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as exc:                      # noqa: BLE001 -- report every failure
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
