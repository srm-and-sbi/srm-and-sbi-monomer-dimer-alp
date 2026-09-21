"""Direct flicker-rate estimator: lambda_rate from the autocorrelation of spot brightness.

This utility estimates the brightness-flicker rate without a network, from video alone. It
completes the set of direct estimators alongside the PSF-width and fluorescence-loss ones,
and like them it is simulation-based inference -- validated against known ground truth on
simulated recordings, under acceptance criteria fixed before the run -- and direct in that
nothing is learned.

Relation to the derivation of record. `Flicker_Rate_Derivation` fixed the `lambda_rate` PRIOR
from ThunderSTORM localization tables, giving 5.1 (MET-Fab) and 4.7 (MET-InlB). That utility
consumes ThunderSTORM's fitted per-localization photon counts, so it is unavailable under the
acquisition-information premise of `DETECTOR_WORKFLOW.md` sec. 9.4, which assumes those outputs
are not to hand. This utility measures the same quantity from the frames themselves, using a
deliberately identical method -- per-trace log, per-trace linear detrend, gap-aware pooled
autocorrelation, normalization at LAG ONE, and a shape match against a matched-length
Ornstein-Uhlenbeck model arm -- so the two numbers are directly comparable and a disagreement
between them is informative rather than a difference of technique.

Why the model arm simulates a single dye, and what that costs

    In the single-dye case the method is EXACTLY free of the brightness parameters: `mu_pc`
    shifts ln-brightness additively and `sigma_pc` scales it linearly, so both vanish under
    the detrend and the normalization. That is a property worth protecting. Under
    `FAB_POISSON` a labeled subunit carries about two dyes whose photons sum before the
    logarithm, which breaks the exactness -- ln(sum of k lognormals) is not an
    Ornstein-Uhlenbeck process, and its normalized autocorrelation acquires a dependence on
    `sigma_pc`.

    Modeling that multiplicity would remove a small bias at the price of making this
    estimator depend on `sigma_pc`, which is itself one of the six parameters under
    inference. In a campaign whose purpose is to decide which parameters can be pinned
    INDEPENDENTLY, that is circular, and the small bias is the better trade. It is therefore
    left in and bounded: measured against the renderer, the shape displacement from one dye
    to two is 0.0077 at `sigma_pc = 0.42` against roughly 0.035 for a ten percent change in
    `lambda_rate` -- about 2%, or 0.009 dex -- rising to about 0.05 dex at the top corner
    (`sigma_pc = 1.0`, three dyes). The report states that band, computed across the whole
    `sigma_pc` prior, without ever needing a `sigma_pc` value.

Prespecified acceptance (fixed before the first run)

    lambda_rate   Pearson correlation with truth >= 0.8 AND mean absolute log10 error <= 0.08 dex

    0.08 dex is 8% of the 1.0 dex prior width. The multiplicity systematic above sits inside
    it at the center of the `sigma_pc` prior and consumes most of it at the top corner, which
    the report says explicitly rather than leaving the reader to infer.

Usage (from the repo root):
    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Rate.py \\
        --condition FAB --total-time-seconds 2.0 --tasks 0 --max-videos 100

    ... --selftest     # in-memory recordings at known lambda_rate; needs no data tier
    ... --dry-run      # resolve settings; read and compute nothing

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/<alias>_<CONDITION>_<timing>_Direct_Flicker_Rate/
        report.md, direct_flicker_rate.npz, summary.json, figures/
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

STAGE = "Direct_Flicker_Rate"

ACCEPTANCE = {"lambda_corr": 0.80, "lambda_mae_dex": 0.08}
# A 2 s clip is 100 frames, so a track cannot be long; the derivation of record required 40
# localizations, which is kept here as the floor for a usable trace.
DEFAULT_MIN_TRACK = 40
SELFTEST_SEED = 20260918


def _scope_center() -> dict:
    return {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
            for e in det.DETECTOR_NUISANCE_SCOPE}


def estimate_one(video_levels, scope: dict, *, min_track_length: int, n_sigma: float,
                 half_px: int, n_model_traces: int) -> dict:
    """Estimate ``lambda_rate`` from one stored video.

    The measurement pass is the same one the width estimator runs; its fitted per-spot
    amplitudes are the photometric series the autocorrelation needs. ``frame_stride`` is
    pinned to 1 because the autocorrelation is indexed in frames and a stride would rescale
    every lag.
    """
    m = die.measure_spot_widths(video_levels, scope, frame_stride=1, n_sigma=n_sigma,
                                half_px=half_px)
    if m["sqrt2sigma"].size == 0:
        return dict(lambda_rate=np.nan, n_traces=0, tau_seconds=np.nan, residual=np.nan)
    tid = die.link_spot_tracks(m["frame_index"], m["x"], m["y"], frame_stride=1)
    traces, spans = die.spot_intensity_traces(m, tid, min_length=min_track_length)
    if len(traces) < 5:
        return dict(lambda_rate=np.nan, n_traces=len(traces), tau_seconds=np.nan,
                    residual=np.nan)
    shape, _ = die.flicker_data_shape(traces)
    if shape is None:
        return dict(lambda_rate=np.nan, n_traces=len(traces), tau_seconds=np.nan,
                    residual=np.nan)
    r = die.match_flicker_rate(
        shape, spans, frame_time_seconds=PARAMETERS.simulation.timing.frame_time_seconds,
        n_traces=n_model_traces)
    return dict(lambda_rate=r["lambda_rate"], n_traces=len(traces),
                tau_seconds=r["tau_seconds"], residual=r["residual"])


_STORE_CACHE: dict = {}


def _worker(job):
    video_path, index, scope, opts = job
    handle = _STORE_CACHE.get(video_path)
    if handle is None:
        handle = sio.load_data(video_path)
        _STORE_CACHE[video_path] = handle
    out = estimate_one(np.asarray(handle[index]), scope, **opts)
    out["index"] = int(index)
    return out


def run_selftest(n_subunits: int, n_frames: int, opts: dict) -> dict:
    """Render single-dye recordings at known flicker rates and recover them.

    Single dye per subunit deliberately: it isolates the estimator from the multiplicity
    systematic, which is quantified separately and reported as a band. A tier run exercises
    the multi-dye case.
    """
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video

    stem = PARAMETERS.simulation.stem
    npx, px_nm = stem.root_size_px, stem.pixel_size_nm
    dt = PARAMETERS.simulation.timing.frame_time_seconds
    scope = _scope_center()
    grid = [1.5, 3.0, 5.0, 8.0]

    truths, ests, extra = [], [], []
    for i, lam in enumerate(grid):
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
        img.update(lambda_rate=lam, prob_photo_bleach=1e-12)
        vec = np.array([img[k] for k in det.DETECTOR_IMAGING_KEYS])
        frames = render_dli_video(poses, host, np.ones(n_subunits, dtype=np.int64), vec,
                                  seed=SELFTEST_SEED + i)
        video = sio.convert_video_dtype(np.moveaxis(frames, 2, 0), bits_from=16, bits_to=8)
        r = estimate_one(video, scope, **opts)
        truths.append(lam)
        ests.append(r["lambda_rate"])
        extra.append(r["n_traces"])
        err = (np.log10(r["lambda_rate"]) - np.log10(lam)
               if np.isfinite(r["lambda_rate"]) and r["lambda_rate"] > 0 else np.nan)
        print(f"  [{i + 1}/{len(grid)}] lambda {lam:5.2f} -> {r['lambda_rate']:7.3f}   "
              f"err {err:+.4f} dex   ({r['n_traces']} traces, "
              f"tau {r['tau_seconds']:.4f} s)", flush=True)
    return dict(truth=np.asarray(truths, float), estimate=np.asarray(ests, float),
                n_traces=np.asarray(extra))


def score(truth, estimate, reporter) -> dict:
    ok = np.isfinite(estimate) & np.isfinite(truth) & (estimate > 0) & (truth > 0)
    t, e = truth[ok], estimate[ok]
    err = np.log10(e) - np.log10(t)
    mae, bias = float(np.abs(err).mean()), float(err.mean())
    corr = (float(np.corrcoef(np.log10(t), np.log10(e))[0, 1])
            if t.size > 2 and np.std(np.log10(t)) > 0 else float("nan"))
    typ, worst = die.flicker_multiplicity_band(float(np.median(e)) if e.size else np.nan)

    reporter.stat("recordings scored", int(ok.sum()))
    reporter.stat("recordings dropped", int((~ok).sum()),
                  note="too few usable traces, or too few pooled autocorrelation pairs")
    reporter.stat("lambda_rate MAE (dex)", mae,
                  expected=f"<= {ACCEPTANCE['lambda_mae_dex']}",
                  note="mean absolute log10 error; 0.08 dex is 8% of the 1.0 dex prior width")
    reporter.stat("lambda_rate bias (dex)", bias,
                  note="mean signed log10 error; positive means the estimate runs fast")
    reporter.stat("lambda_rate correlation", corr,
                  expected=f">= {ACCEPTANCE['lambda_corr']}",
                  note="Pearson r of log10 estimate against log10 truth")
    reporter.stat("multiplicity systematic (dex)", typ,
                  expected=f"up to {worst} at the top of the sigma_pc prior",
                  note="the price of keeping the estimator free of mu_pc and sigma_pc; a band, "
                       "not a correction, so that no sigma_pc value is needed")

    passes = {
        "corr": reporter.check(
            "lambda_rate correlation within acceptance",
            bool(np.isfinite(corr) and corr >= ACCEPTANCE["lambda_corr"]),
            f"{corr:.4f} vs >= {ACCEPTANCE['lambda_corr']}", fatal=False,
            note="Whether the estimate tracks the true flicker rate at all."),
        "mae": reporter.check(
            "lambda_rate MAE within acceptance", mae <= ACCEPTANCE["lambda_mae_dex"],
            f"{mae:.4f} dex vs <= {ACCEPTANCE['lambda_mae_dex']}", fatal=False,
            note="Accuracy of the flicker rate. The multiplicity systematic reported above is "
                 "inside this threshold at the center of the sigma_pc prior and consumes most "
                 "of it at the top corner."),
    }
    return dict(mae_dex=mae, bias_dex=bias, corr=corr, n_scored=int(ok.sum()),
                multiplicity_typical_dex=typ, multiplicity_worst_dex=worst,
                acceptance=ACCEPTANCE, passes={k: bool(v) for k, v in passes.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Direct (non-neural) estimate of lambda_rate from spot-brightness flicker.")
    ap.add_argument("--condition", default=None, choices=list(lab.LABELING_CONDITIONS))
    ap.add_argument("--total-time-seconds", type=float, default=None)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0])
    ap.add_argument("--split", default="EVAL", choices=["EVAL", "TEST", "TRAIN"])
    ap.add_argument("--max-videos", type=int, default=100)
    ap.add_argument("--expect-videos-per-task", type=int, default=1000)
    ap.add_argument("--min-track-length", type=int, default=DEFAULT_MIN_TRACK)
    ap.add_argument("--n-sigma", type=float, default=4.0)
    ap.add_argument("--half-px", type=int, default=14)
    ap.add_argument("--model-traces", type=int, default=4000,
                    help="traces in the Ornstein-Uhlenbeck model arm, per grid point.")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker processes for a tier run; the selftest renders serially.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-subunits", type=int, default=150)
    ap.add_argument("--selftest-frames", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    if not args.selftest and (args.condition is None or args.total_time_seconds is None):
        ap.error("--condition and --total-time-seconds are required (or use --selftest)")

    data_bank_root = PARAMETERS.machine.data_bank_root
    dt = PARAMETERS.simulation.timing.frame_time_seconds
    opts = dict(min_track_length=args.min_track_length, n_sigma=args.n_sigma,
                half_px=args.half_px, n_model_traces=args.model_traces)

    if args.selftest:
        timing_label = RunTiming(total_time_seconds=args.selftest_frames * dt).label
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
        print(f"  out dir         : {out_dir}")
        print(f"  frames          : {n_frames}   min track {args.min_track_length}")
        print(f"  model arm       : single dye, {args.model_traces} traces per grid point")
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
        run_note=("Direct, non-neural estimate of the brightness-flicker rate. The model arm "
                  "is single-dye, which keeps the estimator free of mu_pc and sigma_pc; the "
                  "resulting multiplicity systematic is reported as a band rather than "
                  "corrected, so no value of an inferred parameter is required."))

    if args.selftest:
        reporter.checkpoint("selftest", subunits=args.selftest_subunits, frames=n_frames,
                            recordings=4)
        res = run_selftest(args.selftest_subunits, n_frames, opts)
        truth, estimate, n_traces = res["truth"], res["estimate"], res["n_traces"]
    else:
        truth_rows, jobs = [], []
        for (t, vpath), (_, tpath), (_, spath) in zip(video_paths, theta_paths, scope_paths):
            reporter.check_file(f"video task {t}", vpath)
            reporter.check_file(f"theta task {t}", tpath)
            reporter.check_file(f"scope task {t}", spath)
            theta = np.asarray(sio.load_data(tpath))
            scope_arr = np.asarray(sio.load_data(spath))
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
                jobs.append((str(vpath), i, scope_row, opts))
                truth_rows.append(theta[i, det.DETECTOR_FIND["lambda_rate"]])
            if len(jobs) >= args.max_videos:
                break
        reporter.stat("videos queued", len(jobs))
        if args.workers and args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                out = list(pool.map(_worker, jobs, chunksize=1))
        else:
            out = [_worker(j) for j in jobs]
        truth = np.asarray(truth_rows, dtype=float)
        estimate = np.asarray([o["lambda_rate"] for o in out], dtype=float)
        n_traces = np.asarray([o["n_traces"] for o in out])

    summary = score(truth, estimate, reporter)
    if n_traces.size:
        reporter.stat("median usable traces per recording", float(np.median(n_traces)),
                      note="linked tracks long enough to enter the pooled autocorrelation")

    reporter.table(
        "Acceptance", ["criterion", "threshold", "observed", "verdict"],
        [["lambda_rate correlation", f">= {ACCEPTANCE['lambda_corr']}",
          f"{summary['corr']:.4f}", "PASS" if summary["passes"]["corr"] else "FAIL"],
         ["lambda_rate MAE (dex)", f"<= {ACCEPTANCE['lambda_mae_dex']}",
          f"{summary['mae_dex']:.4f}", "PASS" if summary["passes"]["mae"] else "FAIL"],
         ["multiplicity systematic (dex)", "-- (reported, not thresholded)",
          f"{summary['multiplicity_typical_dex']} typical, "
          f"{summary['multiplicity_worst_dex']} worst", "informational"]],
        note="The systematic is the price of a model arm that needs no value of mu_pc or "
             "sigma_pc. Modeling the dye multiplicity would remove it, at the cost of making "
             "this estimator depend on a parameter that is itself under inference.")

    np.savez_compressed(os.path.join(out_dir, "direct_flicker_rate.npz"),
                        truth=truth, estimate=estimate, n_traces=n_traces)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    reporter.summary()
    path = reporter.write_report()
    print(f"\n[{STAGE}] report -> {path}")
    failed = [k for k, v in summary["passes"].items() if not v]
    if failed:
        print(f"[{STAGE}] ACCEPTANCE NOT MET: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
