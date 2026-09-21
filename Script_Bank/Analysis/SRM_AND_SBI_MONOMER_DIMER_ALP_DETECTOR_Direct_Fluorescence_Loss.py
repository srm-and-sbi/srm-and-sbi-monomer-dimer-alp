"""Direct fluorescence-loss estimator: prob_photo_bleach from the decay of spot flux.

The neural posterior estimator infers all six imaging parameters jointly from whole videos.
This utility estimates one of them -- the photobleaching probability -- without a network, by
measuring how the total background-subtracted fluorescence inside spot apertures decays over
the recording. Like the PSF-width estimator it is simulation-based inference, validated
against known ground truth under acceptance criteria fixed before the run, and direct in that
nothing is learned.

Why the observable is total fluorescence, and why the recording must be long

    Total flux is linear in the number of surviving dyes, which is what the parameter
    controls. A count of visible spots is NOT a substitute: a two-dye spot stays visible when
    one of its dyes bleaches, so counting spots understates the loss and does so in a way that
    depends on the labeling law. `DETECTOR_WORKFLOW.md` sec. 9.4 makes the same point.

    The parameter is a probability over a FIXED hundred-frame reference window, not over the
    clip, so a 2 s recording spans exactly one window and sees the decay once. Three effects
    compound from there.

    First, the brightness flicker is a stationary Ornstein-Uhlenbeck process with a
    correlation time of roughly fifteen frames, so consecutive frames of a fluorescence curve
    are not independent samples of the decay: a 100-frame recording carries about THREE
    effectively independent samples, not a hundred. Second, decay-rate information grows as
    the cube of the duration. Third, and decisively, the amplitude and the offset of the curve
    are unknown and must be fitted alongside the rate; where the decay is shallow, the
    exponential is close to a straight line over the observed window and only the PRODUCT of
    amplitude and rate is determined, so the rate itself is barely constrained.

    The information budget, with amplitude and offset profiled out, gives the following bounds
    in dex at the MET-FAB emitter density (entries in bold are inside the 0.10 dex acceptance
    threshold; the prior is 1.5 dex wide, so a bound above that is no constraint at all):

| recording | p = 0.01 | p = 0.032 | p = 0.10 | p = 0.32 |
|---|---|---|---|---|
| 2 s (100 frames) | 1817 | 178 | 16.5 | 1.26 |
| 20 s (1000 frames) | 6.01 | 0.65 | **0.083** | **0.019** |
| 60 s (3000 frames) | 0.43 | **0.057** | **0.014** | **0.011** |

    So this parameter is NOT identifiable anywhere in its prior from a 2 s clip, and at 20 s it
    is identifiable only in the upper half. That is a property of the recordings, not of this
    estimator, and it applies to the neural posterior equally. The acceptance threshold is
    therefore stated at 1000 frames AND restricted to the range where the bound permits it;
    outside that range the report records the estimate and states that no threshold applies.

Prespecified acceptance (fixed before the first run)

    prob_photo_bleach   mean absolute log10 error <= 0.10 dex at 1000 frames

    That is 6.7% of its 1.5 dex prior width. No threshold is set at 100 frames, because the
    information budget shows the bound there already exceeds it.

Usage (from the repo root):
    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Fluorescence_Loss.py \\
        --condition FAB --total-time-seconds 2.0 --tasks 0 --max-videos 200

    ... --selftest --selftest-frames 1000   # in-memory recordings at known bleaching rates
    ... --dry-run                           # resolve settings; read and compute nothing

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/<alias>_<CONDITION>_<timing>_Direct_Fluorescence_Loss/
        report.md, direct_fluorescence_loss.npz, summary.json, figures/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import direct_imaging_estimates as die  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import information_budget as ib  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import io as sio  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import labeling as lab  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import (  # noqa: E402
    PARAMETERS, RunTiming,
)

assert os.path.abspath(die.__file__).startswith(REPO_ROOT), (
    "Imported srm_and_sbi_monomer_dimer_alp from outside this repo checkout: "
    + die.__file__ + " -- run with PYTHONPATH set to the repo root."
)

STAGE = "Direct_Fluorescence_Loss"

# Fixed before the first run. 0.10 dex is 6.7% of the 1.5 dex prior width of
# prob_photo_bleach. It applies at 1000 frames; at 100 frames the information budget bound
# already exceeds it, so no threshold is set there and the report says so.
ACCEPTANCE = {"prob_bleach_mae_dex": 0.10, "acceptance_frames": 1000}
# Relative noise of the total-fluorescence curve per frame at the MET-FAB emitter density,
# measured on rendered recordings. Used only to state the information bound beside the result.
RELATIVE_NOISE = 0.028
SELFTEST_SEED = 20260918


def _scope_center() -> dict:
    return {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
            for e in det.DETECTOR_NUISANCE_SCOPE}


def estimate_one(video_levels, scope: dict, *, lambda_rate=None,
                 n_sigma: float = 4.0, detect_frames: int = 5,
                 observable: str = "field") -> dict:
    """Estimate ``prob_photo_bleach`` from one stored video.

    ``observable`` selects how the flux curve is formed. ``"field"`` sums the whole frame and
    removes the background by a per-frame quantile; it is immune to emitter motion and is the
    default. ``"apertures"`` sums inside apertures pinned to the opening frames; it is kept
    only for diagnosis, because on diffusing emitters the spots walk out of their apertures
    and the resulting decay is dominated by motion rather than bleaching.
    """
    if observable == "apertures":
        curve = die.spot_flux_curve(video_levels, scope, n_sigma=n_sigma,
                                    detect_frames=detect_frames)
        n_ap = curve["n_apertures"]
        if n_ap == 0:
            return dict(prob_photo_bleach=np.nan, prob_se=np.nan, n_apertures=0, n_eff=np.nan)
    else:
        curve = die.frame_flux_curve(video_levels)
        n_ap = -1
    fit = die.fit_fluorescence_loss(
        curve["flux"], lambda_rate=lambda_rate,
        frame_time_seconds=PARAMETERS.simulation.timing.frame_time_seconds)
    return dict(prob_photo_bleach=fit["prob_photo_bleach"], prob_se=fit["prob_se"],
                n_apertures=n_ap, n_eff=fit["n_eff"])


_STORE_CACHE: dict = {}


def _worker(job):
    video_path, index, scope, lam, opts = job
    handle = _STORE_CACHE.get(video_path)
    if handle is None:
        handle = sio.load_data(video_path)
        _STORE_CACHE[video_path] = handle
    out = estimate_one(np.asarray(handle[index]), scope, lambda_rate=lam, **opts)
    out["index"] = int(index)
    return out


def run_selftest(n_subunits: int, n_frames: int, n_sigma: float, detect_frames: int,
                 observable: str = "field") -> dict:
    """Render single-dye recordings at known bleaching probabilities and recover them."""
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video

    stem = PARAMETERS.simulation.stem
    npx, px_nm = stem.root_size_px, stem.pixel_size_nm
    dt = PARAMETERS.simulation.timing.frame_time_seconds
    scope = _scope_center()
    grid = [0.01, 0.0316, 0.1, 0.316]

    truths, ests = [], []
    for i, p in enumerate(grid):
        rng = np.random.default_rng(SELFTEST_SEED + i)
        step_nm = np.sqrt(2.0 * 0.05 * 1e6 * dt)
        box = npx * px_nm
        pos = rng.uniform(0.1 * box, 0.9 * box, size=(n_subunits, 2))
        poses = np.zeros((n_frames, n_subunits, 3))
        for t in range(n_frames):
            poses[t, :, :2] = pos
            pos = np.clip(pos + rng.normal(0.0, step_nm, (n_subunits, 2)),
                          0.02 * box, 0.98 * box)
        host = np.tile(np.arange(n_subunits)[None, :], (n_frames, 1))

        img = {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
               for e in det.DETECTOR_IMAGING}
        img["prob_photo_bleach"] = p
        vec = np.array([img[k] for k in det.DETECTOR_IMAGING_KEYS])
        frames = render_dli_video(poses, host, np.ones(n_subunits, dtype=np.int64), vec,
                                  seed=SELFTEST_SEED + i)
        video = sio.convert_video_dtype(np.moveaxis(frames, 2, 0), bits_from=16, bits_to=8)
        r = estimate_one(video, scope, lambda_rate=img["lambda_rate"],
                         n_sigma=n_sigma, detect_frames=detect_frames,
                         observable=observable)
        truths.append(p)
        ests.append(r["prob_photo_bleach"])
        err = (np.log10(max(r["prob_photo_bleach"], 1e-9)) - np.log10(p)
               if np.isfinite(r["prob_photo_bleach"]) else np.nan)
        print(f"  [{i + 1}/{len(grid)}] p {p:.4f} -> {r['prob_photo_bleach']:.4f}   "
              f"err {err:+.4f} dex   ({r['n_apertures']} apertures, "
              f"n_eff {r['n_eff']:.1f})", flush=True)
    return dict(truth=np.asarray(truths, float), estimate=np.asarray(ests, float))


def score(truth, estimate, n_frames, reporter) -> dict:
    ok = np.isfinite(estimate) & np.isfinite(truth) & (estimate > 0) & (truth > 0)
    t, e = truth[ok], estimate[ok]
    err = np.log10(e) - np.log10(t)
    mae, bias = float(np.abs(err).mean()), float(err.mean())
    corr = (float(np.corrcoef(np.log10(t), np.log10(e))[0, 1])
            if t.size > 2 and np.std(np.log10(t)) > 0 else float("nan"))

    # The bound this recording length supports AT EACH TRUTH. It varies enormously across the
    # prior -- from well inside the threshold at the top to hundreds of dex at the bottom --
    # so a single pooled error would average the measurable range together with a range where
    # no estimator can say anything, and would report the mixture as a failure of this one.
    lam = 10 ** (0.5 * sum(det.DETECTOR_PARAMETERIZATION[det.DETECTOR_FIND["lambda_rate"]]["PRIOR_RANGE"]))
    bounds = np.array([ib.crb_prob_bleach_dex(float(p), int(n_frames), RELATIVE_NOISE,
                                              lambda_rate=lam,
                                              frame_time_seconds=PARAMETERS.simulation.timing.frame_time_seconds)["sd_dex"]
                       for p in t])
    mean_bound = float(np.mean(bounds)) if bounds.size else float("nan")

    # Acceptance applies only where the recording can support it.
    within = bounds <= ACCEPTANCE["prob_bleach_mae_dex"]
    mae_within = float(np.abs(err[within]).mean()) if within.any() else float("nan")
    mae_outside = float(np.abs(err[~within]).mean()) if (~within).any() else float("nan")

    reporter.stat("recordings scored", int(ok.sum()))
    reporter.stat("recordings dropped", int((~ok).sum()),
                  note="no apertures found, or a non-positive estimate")
    reporter.stat("prob_photo_bleach MAE (dex)", mae,
                  expected=f"<= {ACCEPTANCE['prob_bleach_mae_dex']} at "
                           f"{ACCEPTANCE['acceptance_frames']} frames",
                  note="mean absolute log10 error; 0.10 dex is 6.7% of the prior width")
    reporter.stat("prob_photo_bleach bias (dex)", bias,
                  note="mean signed log10 error; positive means the estimate runs high")
    reporter.stat("correlation", corr, note="Pearson r of log10 estimate against log10 truth")
    reporter.stat("mean information bound (dex)", mean_bound,
                  note="Cramer-Rao floor at this recording length, including the flicker "
                       "correlation; no estimator can do better than this")
    reporter.stat("frames", int(n_frames),
                  note="recording length; decay-rate information grows as its cube")

    reporter.stat("recordings inside the identifiable range", int(within.sum()),
                  note="those whose information bound is itself below the threshold; only "
                       "these can be held to it")
    reporter.stat("MAE inside the identifiable range (dex)", mae_within,
                  expected=f"<= {ACCEPTANCE['prob_bleach_mae_dex']}",
                  note="the acceptance quantity")
    reporter.stat("MAE outside the identifiable range (dex)", mae_outside,
                  note="reported for information only; no estimator can meet the threshold "
                       "here, so a large value is a property of the recording")

    at_acceptance_length = int(n_frames) >= ACCEPTANCE["acceptance_frames"]
    testable = bool(at_acceptance_length and within.any())
    passed = reporter.check(
        "prob_photo_bleach MAE within acceptance, where identifiable",
        bool(testable and mae_within <= ACCEPTANCE["prob_bleach_mae_dex"]),
        (f"{mae_within:.4f} dex over {int(within.sum())} of {int(ok.sum())} recordings "
         f"vs <= {ACCEPTANCE['prob_bleach_mae_dex']}"
         if testable else
         (f"no recording is inside the identifiable range at {n_frames} frames"
          if at_acceptance_length else
          f"{n_frames} frames is below the {ACCEPTANCE['acceptance_frames']}-frame length "
          f"the threshold is stated at")),
        fatal=False,
        note="Accuracy of the photobleaching probability, over the part of the prior the "
             "recording can actually support. Outside that part the information bound exceeds "
             "the threshold, so a miss there measures the recording rather than the estimator "
             "and is reported separately instead of being pooled in.")
    # A real comparison, not a formality. For an unbiased estimator the mean absolute error of
    # a normal variate is sqrt(2/pi) ~ 0.8 of its standard deviation, so the bound on the sd
    # implies a floor of about 0.8 * bound on the MAE. Falling clearly under that floor is
    # worth inspecting; the factor below leaves room for the small-sample scatter of an MAE
    # taken over a handful of recordings.
    bound_within = float(np.mean(bounds[within])) if within.any() else float("nan")
    mae_floor = 0.8 * bound_within
    respects_bound = bool(not np.isfinite(mae_within) or not np.isfinite(mae_floor)
                          or mae_within >= 0.5 * mae_floor)
    reporter.check(
        "measurement respects the information bound", respects_bound,
        (f"MAE inside the identifiable range {mae_within:.4f} dex against an implied MAE floor "
         f"of {mae_floor:.4f} dex (from a mean bound of {bound_within:.4f} dex on the standard "
         f"deviation), over {int(within.sum())} recording(s)"),
        fatal=False,
        note="A measured error far BELOW the Cramer-Rao floor points to a defect -- ground truth "
             "leaking into the estimate, or a mis-stated bound -- rather than an unusually good "
             "estimator. Two caveats keep this a prompt to look rather than an automatic "
             "failure: the bound constrains an UNBIASED estimator, and a fit with bounded "
             "parameters can legitimately beat it by shrinking toward the middle of its range; "
             "and an MAE over few recordings is itself noisy.")

    return dict(mae_dex=mae, mae_within_dex=mae_within, mae_outside_dex=mae_outside,
                bound_within_dex=bound_within, respects_bound=bool(respects_bound),
                bias_dex=bias, corr=corr, mean_bound_dex=mean_bound,
                bounds_dex=[float(b) for b in bounds],
                n_within=int(within.sum()), n_frames=int(n_frames), n_scored=int(ok.sum()),
                at_acceptance_length=at_acceptance_length, testable=testable,
                acceptance=ACCEPTANCE, passed=bool(passed))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Direct (non-neural) estimate of prob_photo_bleach from fluorescence loss.")
    ap.add_argument("--condition", default=None, choices=list(lab.LABELING_CONDITIONS))
    ap.add_argument("--total-time-seconds", type=float, default=None)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0])
    ap.add_argument("--split", default="EVAL", choices=["EVAL", "TEST", "TRAIN"])
    ap.add_argument("--max-videos", type=int, default=200)
    ap.add_argument("--expect-videos-per-task", type=int, default=1000)
    ap.add_argument("--n-sigma", type=float, default=4.0)
    ap.add_argument("--detect-frames", type=int, default=5,
                    help="frames whose detections seed the fixed aperture set (apertures only).")
    ap.add_argument("--observable", default="field", choices=["field", "apertures"],
                    help="how the flux curve is formed. 'field' sums the whole frame and is "
                         "immune to emitter motion; 'apertures' pins apertures to the opening "
                         "frames and is kept for diagnosis only -- on diffusing emitters the "
                         "spots walk out of their apertures and the fitted decay measures "
                         "motion rather than bleaching.")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-subunits", type=int, default=150)
    ap.add_argument("--selftest-frames", type=int, default=1000,
                    help="the acceptance threshold is stated at 1000 frames.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    if not args.selftest and (args.condition is None or args.total_time_seconds is None):
        ap.error("--condition and --total-time-seconds are required (or use --selftest)")

    data_bank_root = PARAMETERS.machine.data_bank_root
    dt = PARAMETERS.simulation.timing.frame_time_seconds
    if args.selftest:
        timing_label = RunTiming(total_time_seconds=args.selftest_frames * dt).label
        # No condition token: the selftest renders condition-free single-dye recordings, and
        # the slot before the timing label is reserved for FAB/INLB.
        run_alias = f"{det.detector_paths(PARAMETERS.paths).project_alias}_{timing_label}"
        n_frames = args.selftest_frames
    else:
        timing = RunTiming(total_time_seconds=args.total_time_seconds)
        timing_label = timing.label
        n_frames = timing.frame_count
        paths = det.detector_paths(PARAMETERS.paths).with_condition(args.condition)
        run_alias = f"{paths.project_alias}_{timing_label}"
        video_paths = [(t, paths.video_set_path(t, data_bank_root, timing_label, True, args.split))
                       for t in args.tasks]
        theta_paths = [(t, paths.theta_set_path(t, data_bank_root, timing_label, True, args.split))
                       for t in args.tasks]
        scope_paths = [(t, paths.record_set_path("Nuisance_SCOPE_Theta_Set", t, data_bank_root,
                                                 timing_label, True, args.split))
                       for t in args.tasks]

    descriptor = f"{STAGE}_SELFTEST" if args.selftest else STAGE
    out_dir = args.out_dir or os.path.join(str(data_bank_root), "Posit",
                                           f"{run_alias}_{descriptor}")

    if args.dry_run:
        print(f"[{STAGE}] DRY RUN -- nothing is read and nothing is written.")
        print(f"  machine profile : {os.environ.get('MACHINE_PROFILE', '(unset)')}")
        print(f"  mode            : {'selftest' if args.selftest else args.split}")
        print(f"  run alias       : {run_alias}")
        print(f"  out dir         : {out_dir}")
        print(f"  frames          : {n_frames}   (acceptance stated at "
              f"{ACCEPTANCE['acceptance_frames']})")
        print(f"  observable      : {args.observable}")
        if not args.selftest:
            for t, p in video_paths:
                print(f"  reads video T{t:<3d}: {p}   exists={os.path.exists(p)}")
            for t, p in theta_paths:
                print(f"  reads theta T{t:<3d}: {p}   exists={os.path.exists(p)}")
            for t, p in scope_paths:
                print(f"  reads scope T{t:<3d}: {p}   exists={os.path.exists(p)}")
        else:
            print(f"  selftest        : 4 recordings, {args.selftest_subunits} subunits x "
                  f"{n_frames} frames")
        print(f"  acceptance      : {ACCEPTANCE}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    reporter = DiagnosticReporter(
        stage=STAGE, enabled=True, dump=True, dump_dir=out_dir, run_label=run_alias,
        run_note=("Direct, non-neural estimate of the photobleaching probability. The "
                  "acceptance threshold applies at the full recording length; a 2 s tier is "
                  "reported for information only, because the information bound there "
                  "already exceeds the threshold."))

    if args.selftest:
        reporter.checkpoint("selftest", subunits=args.selftest_subunits, frames=n_frames,
                            recordings=4)
        res = run_selftest(args.selftest_subunits, n_frames, args.n_sigma,
                           args.detect_frames, observable=args.observable)
        truth, estimate = res["truth"], res["estimate"]
        extra = dict(n_apertures=np.array([]), n_eff=np.array([]))
    else:
        truth_rows, jobs = [], []
        opts = dict(n_sigma=args.n_sigma, detect_frames=args.detect_frames,
                    observable=args.observable)
        for (t, vpath), (_, tpath), (_, spath) in zip(video_paths, theta_paths, scope_paths):
            reporter.check_file(f"video task {t}", vpath)
            reporter.check_file(f"theta task {t}", tpath)
            # The camera record is only read by the aperture observable; the default field
            # observable never touches it. Requiring it unconditionally aborted runs that did
            # not need it, so it is fatal only on the path that uses it.
            needs_scope = (args.observable == "apertures")
            reporter.check_file(f"scope task {t}", spath, fatal=needs_scope)
            theta = np.asarray(sio.load_data(tpath))
            scope_arr = (np.asarray(sio.load_data(spath)) if os.path.exists(spath)
                         else np.full((theta.shape[0], len(det.DETECTOR_SCOPE_KEYS)), np.nan))
            reporter.check(
                f"task {t} tier is production-sized",
                theta.shape[0] >= args.expect_videos_per_task,
                f"{theta.shape[0]} videos vs expected {args.expect_videos_per_task}",
                fatal=False,
                note="A stale development tier can carry production filenames while holding "
                     "only a couple of videos; scoring it would produce a confident-looking "
                     "but meaningless error.")
            n_here = min(theta.shape[0], args.max_videos - len(jobs))
            for i in range(n_here):
                scope_row = {k: float(scope_arr[i, j])
                             for j, k in enumerate(det.DETECTOR_SCOPE_KEYS)}
                lam = float(theta[i, det.DETECTOR_FIND["lambda_rate"]])
                jobs.append((str(vpath), i, scope_row, lam, opts))
                truth_rows.append(theta[i, det.DETECTOR_FIND["prob_photo_bleach"]])
            if len(jobs) >= args.max_videos:
                break
        reporter.stat("videos queued", len(jobs))
        if args.workers and args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                out = list(pool.map(_worker, jobs, chunksize=1))
        else:
            out = [_worker(j) for j in jobs]
        truth = np.asarray(truth_rows, dtype=float)
        estimate = np.asarray([o["prob_photo_bleach"] for o in out], dtype=float)
        extra = dict(n_apertures=np.array([o["n_apertures"] for o in out]),
                     n_eff=np.array([o["n_eff"] for o in out]))

    summary = score(truth, estimate, n_frames, reporter)

    reporter.table(
        "Acceptance", ["criterion", "threshold", "observed", "verdict"],
        [["prob_photo_bleach MAE (dex), where identifiable",
          f"<= {ACCEPTANCE['prob_bleach_mae_dex']} at {ACCEPTANCE['acceptance_frames']} frames",
          (f"{summary['mae_within_dex']:.4f} over {summary['n_within']} of "
           f"{summary['n_scored']} at {n_frames} frames"
           if summary["testable"] else f"n/a at {n_frames} frames"),
          "PASS" if summary["passed"] else
          ("FAIL" if summary["testable"] else "NOT APPLICABLE")],
         ["prob_photo_bleach MAE (dex), outside", "-- (no threshold)",
          f"{summary['mae_outside_dex']:.4f}", "informational"]],
        note="The threshold is stated at the full recording length and applied only where the "
             "information bound is itself below it. A 2 s clip cannot support this parameter "
             "anywhere in its prior, and a 20 s recording only in its upper half, for any "
             "estimator -- see DETECTOR_WORKFLOW.md sec. 9.5.")

    np.savez_compressed(os.path.join(out_dir, "direct_fluorescence_loss.npz"),
                        truth=truth, estimate=estimate, **extra)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    reporter.summary()
    path = reporter.write_report()
    print(f"\n[{STAGE}] report -> {path}")
    # A failed acceptance must reach a caller that reads the exit status. A run that is not
    # testable at this recording length is not a failure -- it is reported as not applicable.
    if summary["testable"] and not summary["passed"]:
        print(f"[{STAGE}] ACCEPTANCE NOT MET: prob_photo_bleach MAE where identifiable")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
