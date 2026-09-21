"""Direct PSF-width estimator: mu_r and sigma_r read off the spots of a recording.

The neural posterior estimator infers all six imaging parameters jointly from whole
videos. This utility estimates two of them -- the PSF-width population parameters --
without a network, by fitting the renderer's own pixel-integrated Gaussian to individual
spots and summarizing the fitted widths. It is simulation-based inference in the same
sense the flow is: it is validated against known ground truth on simulated recordings,
against acceptance criteria fixed before the run. What differs is that nothing is
learned; each number is read off a closed-form property of the forward model.

The purpose is the acquisition-information contract of `DETECTOR_WORKFLOW.md` sec. 9.4 --
deciding which parameters must stay in the inferred block and which can be supplied from
a cheaper measurement. Section 9.4 is a proposal and is not in force; this utility
produces the evidence that proposal is gated on, and does not itself change any stage.

What it measures, and why those estimators

    mu_r, sigma_r   `sample_psf_width` draws sqrt(2)*sigma in pixels from a lognormal with
                    scale mu_r and shape sigma_r, ONE draw per subunit, carried by every
                    dye of that subunit for the whole recording. So mu_r is the median of
                    sqrt(2)*sigma and sigma_r the standard deviation of its logarithm, and
                    both are population parameters of the spots in one video.

    Three design choices are forced by measurements against the renderer, each recorded in
    `direct_imaging_estimates`:

      - the local background is SUPPLIED from the SCOPE camera block, not fitted. A free
        background is nearly degenerate with a broad faint Gaussian and pulls the fitted
        width down by 6% at the bottom of the mu_r box and 25% at the top -- a
        width-dependent bias, which compresses the very spread sigma_r measures.
      - the fit is FLAT-weighted, and the EMCCD variance law enters only the standard
        error, through a sandwich covariance. Weighting the fit by the model variance
        down-weights the bright core that carries the width information and biased mu_r
        high by 0.06 dex.
      - spots are LINKED across frames before the population spread is taken, because a
        track is repeat measurement of one subunit's single width. Averaging within a
        track first reduces the per-fit noise by the square root of the track length and
        demotes the errors-in-variables correction from dominant term to perturbation.

Acceptance is governed by DETECTOR_WORKFLOW.md sec. 9.6 (frozen 2026-09-21): evidence adequacy,
operational success, accuracy overall AND in the operating subgroup, and coverage of the nominal
90 % ranges, evaluated in that order by the shared `direct_acceptance` kernel. The accuracy
thresholds below are that rule's step 4a.

Prespecified acceptance (fixed before the first run; see ACCEPTANCE below)

    mu_r       mean absolute log10 error <= 0.02 dex AND |mean signed error| <= 0.01 dex
    sigma_r    Pearson correlation with truth >= 0.8 AND mean absolute error <= 0.08

    The mu_r thresholds are 6.7% and 3.3% of its 0.3 dex prior width, both being log10
    errors against a log10 width. The sigma_r threshold is stated in LINEAR units, because
    sigma_r is itself a spread and a log error in a near-zero spread is not interpretable;
    it must therefore not be compared against the 0.75 dex prior width, which is a different
    unit. Against the linear prior range [0.10, 0.56] it is 16% of the span, and it is about
    a third of sigma_r at the bottom of that range and a seventh at the top. A parameter that
    clears its thresholds is a candidate to leave the inferred block; one that does not, stays.

Usage (from the repo root):
    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_PSF_Width.py \\
        --condition FAB --total-time-seconds 2.0 --tasks 0 1 --max-videos 200

    ... --selftest     # in-memory scenes rendered at known widths; needs no data tier
    ... --dry-run      # resolve profile, paths and settings; read and compute nothing

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/<alias>_<CONDITION>_<timing>_Direct_PSF_Width/
        report.md                       (the diagnostic report)
        direct_psf_width.npz            (per-video estimates and truths)
        summary.json                    (every number the report quotes)
        figures/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import direct_imaging_estimates as die  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import direct_acceptance as da  # noqa: E402
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

STAGE = "Direct_PSF_Width"

# ---- prespecified acceptance ------------------------------------------------------------
# Fixed before the first run and quoted in the report whatever the outcome. The mu_r
# thresholds are 6.7% (MAE) and 3.3% (bias) of its 0.3 dex prior width -- log10 errors
# against a log10 width, so the ratio is meaningful. sigma_r's threshold is in LINEAR units
# (a log error in a near-zero spread is not interpretable), so it must NOT be divided by the
# 0.75 dex prior width; against the linear range [0.10, 0.56] it is 16% of the span.
ACCEPTANCE = {
    "mu_r_mae_dex": 0.02,
    "mu_r_abs_bias_dex": 0.01,
    "sigma_r_corr": 0.80,
    "sigma_r_mae": 0.08,
}

# ---- measurement operating point ---------------------------------------------------------
DEFAULT_FRAME_STRIDE = 2      # every other frame: linking tolerates a stride, and the cost is linear
DEFAULT_MIN_TRACK = 5         # tracks shorter than this carry too little averaging to help
SELFTEST_SEED = 20260918


def _scope_center() -> dict:
    """SCOPE camera values at the center of their box (used by the selftest only)."""
    return {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
            for e in det.DETECTOR_NUISANCE_SCOPE}


# ==========================================================================================
# One video -> one (mu_r, sigma_r) estimate
# ==========================================================================================

def estimate_one(video_levels, scope: dict, *, frame_stride: int, min_track_length: int,
                 n_sigma: float, half_px: int) -> dict:
    """Estimate ``(mu_r, sigma_r)`` from one stored video.

    Args:
        video_levels: ``(n_frames, n_x, n_y)`` stored 8-bit video.
        scope: the five SCOPE camera values in physical units.
        frame_stride: use every n-th frame.
        min_track_length: minimum linked-track length kept.
        n_sigma: spot-detection threshold.
        half_px: fit patch half-width.

    Returns:
        ``dict`` with ``mu_r``, ``sigma_r``, ``sigma_r_raw``, ``log_mean_se`` (standard error of
        ``ln mu_r``), ``sigma_r_se``, ``n_tracks``, ``n_spots``, and ``reason`` -- ``None`` for a
        valid estimate, else the code for why none was produced (``no_spots``,
        ``too_few_tracks``).
    """
    m = die.measure_spot_widths(video_levels, scope, frame_stride=frame_stride,
                                n_sigma=n_sigma, half_px=half_px)
    if m["sqrt2sigma"].size == 0:
        return dict(mu_r=np.nan, sigma_r=np.nan, sigma_r_raw=np.nan, log_mean_se=np.nan,
                    sigma_r_se=np.nan, n_tracks=0, n_spots=0, reason="no_spots")
    tid = die.link_spot_tracks(m["frame_index"], m["x"], m["y"],
                               frame_stride=frame_stride)
    r = die.psf_width_population(m["sqrt2sigma"], m["sqrt2sigma_se"],
                                 track_id=tid, min_track_length=min_track_length)
    valid = bool(np.isfinite(r["mu_r"]) and np.isfinite(r["sigma_r"]))
    return dict(mu_r=r["mu_r"], sigma_r=r["sigma_r"], sigma_r_raw=r["sigma_r_raw"],
                log_mean_se=r["log_mean_se"], sigma_r_se=r["sigma_r_se"],
                n_tracks=int(r.get("n_tracks", 0)), n_spots=int(m["sqrt2sigma"].size),
                reason=None if valid else "too_few_tracks")


# -- worker-side lazy handle, so a pool process opens each store once ----------------------
_STORE_CACHE: dict = {}


def _worker(job):
    """Pool worker: open (once) and measure one video. Must be top-level to pickle."""
    video_path, index, scope, opts = job
    handle = _STORE_CACHE.get(video_path)
    if handle is None:
        handle = sio.load_data(video_path)
        _STORE_CACHE[video_path] = handle
    video = np.asarray(handle[index])
    out = estimate_one(video, scope, **opts)
    out["index"] = int(index)
    return out


# ==========================================================================================
# Selftest: in-memory scenes rendered at known widths
# ==========================================================================================

def run_selftest(reporter: DiagnosticReporter, *, n_subunits: int, n_frames: int,
                 frame_stride: int, min_track_length: int, n_sigma: float,
                 half_px: int) -> dict:
    """Render diffusing single-dye scenes at a grid of known widths and recover them.

    Single-dye subunits on Brownian paths, rendered through the production renderer at
    the center of every other imaging parameter, so the only thing varying is the PSF
    width population. This isolates the estimator from the labeling law and from the
    reaction-diffusion tier: a failure here is the estimator's, not the model's.
    """
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video

    stem = PARAMETERS.simulation.stem
    npx, px_nm = stem.root_size_px, stem.pixel_size_nm
    dt = PARAMETERS.simulation.timing.frame_time_seconds
    scope = _scope_center()

    grid = [(mu_r, sigma_r)
            for mu_r in (1.10, 1.41, 1.85)
            for sigma_r in (0.10, 0.237, 0.50)]

    def scene(seed):
        rng = np.random.default_rng(seed)
        step_nm = np.sqrt(2.0 * 0.05 * 1e6 * dt)        # D = 0.05 um^2/s, a typical receptor
        box = npx * px_nm
        pos = rng.uniform(0.1 * box, 0.9 * box, size=(n_subunits, 2))
        poses = np.zeros((n_frames, n_subunits, 3))
        for t in range(n_frames):
            poses[t, :, :2] = pos
            pos = np.clip(pos + rng.normal(0.0, step_nm, (n_subunits, 2)),
                          0.02 * box, 0.98 * box)
        return poses, np.tile(np.arange(n_subunits)[None, :], (n_frames, 1))

    truths, ests, ses, reasons, thetas = [], [], [], [], []
    for i, (mu_r, sigma_r) in enumerate(grid):
        img = {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
               for e in det.DETECTOR_IMAGING}
        img.update(mu_r=mu_r, sigma_r=sigma_r, prob_photo_bleach=1e-12)
        thetas.append([img[k] for k in det.DETECTOR_FIND])
        vec = np.array([img[k] for k in det.DETECTOR_IMAGING_KEYS])
        poses, host = scene(SELFTEST_SEED + i)
        frames = render_dli_video(poses, host, np.ones(n_subunits, dtype=np.int64), vec,
                                  seed=SELFTEST_SEED + i)
        video = sio.convert_video_dtype(np.moveaxis(frames, 2, 0), bits_from=16, bits_to=8)
        r = estimate_one(video, scope, frame_stride=frame_stride,
                         min_track_length=min_track_length, n_sigma=n_sigma,
                         half_px=half_px)
        truths.append((mu_r, sigma_r))
        ests.append((r["mu_r"], r["sigma_r"]))
        ses.append((r["log_mean_se"], r["sigma_r_se"]))
        reasons.append(r["reason"])
        print(f"  [{i + 1}/{len(grid)}] mu_r {mu_r:.3f} -> {r['mu_r']:.4f}   "
              f"sigma_r {sigma_r:.3f} -> {r['sigma_r']:.4f}   "
              f"({r['n_tracks']} tracks, {r['n_spots']} spot-frames)", flush=True)

    truths = np.asarray(truths, dtype=float)
    ests = np.asarray(ests, dtype=float)
    return dict(truth=truths, estimate=ests, se=np.asarray(ses, dtype=float), reasons=reasons,
                theta=np.asarray(thetas, dtype=float), grid=grid)


# ==========================================================================================
# Scoring against the prespecified acceptance
# ==========================================================================================

def make_figure(truth, estimate, out_name, reporter):
    """Truth-versus-estimate scatter for both parameters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = np.isfinite(estimate).all(axis=1) & np.isfinite(truth).all(axis=1)
    t, e = truth[ok], estimate[ok]
    if t.shape[0] == 0:
        # Every video failed to yield an estimate. That is exactly when the report matters
        # most -- it carries the checks that say why -- so skip the figure rather than raise
        # on an empty reduction and discard the report, the arrays and the summary with it.
        reporter.check("figure has data to plot", False,
                       "no video produced a finite estimate; figure skipped", fatal=False,
                       note="The scatter needs at least one scored recording. The report, "
                            "arrays and summary are still written.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, col, name, logscale in ((axes[0], 0, r"$\mu_{r}$", True),
                                    (axes[1], 1, r"$\sigma_{r}$", False)):
        ax.scatter(t[:, col], e[:, col], s=14, alpha=0.55, edgecolor="none")
        lo = float(min(t[:, col].min(), e[:, col].min()))
        hi = float(max(t[:, col].max(), e[:, col].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
        if logscale:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"true {name}"); ax.set_ylabel(f"estimated {name}")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    reporter.save_figure(out_name, fig,
                         caption="Direct estimate against ground truth, one point per "
                                 "recording. The dashed line is equality; vertical scatter "
                                 "is estimator error and a vertical offset is bias.")
    plt.close(fig)


# ==========================================================================================
# Entry point
# ==========================================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Direct (non-neural) estimate of the PSF-width parameters mu_r and sigma_r.")
    ap.add_argument("--condition", default=None, choices=list(lab.LABELING_CONDITIONS),
                    help="labeling condition of the data tier (not needed for --selftest).")
    ap.add_argument("--total-time-seconds", type=float, default=None,
                    help="recording length of the tier (not needed for --selftest).")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0],
                    help="EVAL task indices to read.")
    ap.add_argument("--split", default="EVAL", choices=["EVAL", "TEST", "TRAIN"],
                    help="data split to read (EVAL is the held-out namespace with ground truth).")
    ap.add_argument("--max-videos", type=int, default=200,
                    help="cap on the number of videos scored, across all tasks.")
    ap.add_argument("--expect-videos-per-task", type=int, default=1000,
                    help="expected videos in a production task; a smaller tier is flagged. "
                         "Guards against a stale local tier that carries production filenames "
                         "but only a handful of videos.")
    ap.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    ap.add_argument("--min-track-length", type=int, default=DEFAULT_MIN_TRACK)
    ap.add_argument("--n-sigma", type=float, default=4.0, help="spot-detection threshold.")
    ap.add_argument("--half-px", type=int, default=14, help="fit patch half-width in pixels.")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker processes for a tier run; 0 runs in this process. The "
                         "selftest ignores it and renders its scenes serially.")
    ap.add_argument("--selftest", action="store_true",
                    help="in-memory scenes at known widths; needs no data tier.")
    ap.add_argument("--selftest-subunits", type=int, default=150)
    ap.add_argument("--selftest-frames", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve profile, paths and settings; read and compute nothing.")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory (default: the Data_Bank Posit tier).")
    args = ap.parse_args(argv)

    if not args.selftest and (args.condition is None or args.total_time_seconds is None):
        ap.error("--condition and --total-time-seconds are required (or use --selftest)")

    # ---- resolve identity and paths ------------------------------------------------------
    data_bank_root = PARAMETERS.machine.data_bank_root
    if args.selftest:
        timing_label = RunTiming(total_time_seconds=args.selftest_frames
                                 * PARAMETERS.simulation.timing.frame_time_seconds).label
        # No condition token: the selftest renders condition-free single-dye scenes, and
        # CLAUDE.md reserves the slot before the timing label for FAB/INLB. SELFTEST belongs
        # in the descriptor after the timing label, not in that slot.
        run_alias = f"{det.detector_paths(PARAMETERS.paths).project_alias}_{timing_label}"
        video_paths = []
    else:
        timing = RunTiming(total_time_seconds=args.total_time_seconds)
        timing_label = timing.label
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

    # ---- dry run -------------------------------------------------------------------------
    if args.dry_run:
        print(f"[{STAGE}] DRY RUN -- nothing is read and nothing is written.")
        print(f"  machine profile   : {os.environ.get('MACHINE_PROFILE', '(unset)')}")
        print(f"  data_bank_root    : {data_bank_root}")
        print(f"  mode              : {'selftest' if args.selftest else args.split}")
        print(f"  run alias         : {run_alias}")
        print(f"  out dir           : {out_dir}")
        print(f"  frame stride      : {args.frame_stride}   min track {args.min_track_length}"
              f"   n_sigma {args.n_sigma}   half_px {args.half_px}")
        print(f"  workers           : {args.workers}")
        if args.selftest:
            print(f"  selftest scenes   : 9 (3 mu_r x 3 sigma_r), "
                  f"{args.selftest_subunits} subunits x {args.selftest_frames} frames")
        else:
            print(f"  max videos        : {args.max_videos}")
            for t, p in video_paths:
                print(f"  reads video  T{t:<3d}: {p}   exists={os.path.exists(p)}")
            for t, p in theta_paths:
                print(f"  reads theta  T{t:<3d}: {p}   exists={os.path.exists(p)}")
            for t, p in scope_paths:
                print(f"  reads scope  T{t:<3d}: {p}   exists={os.path.exists(p)}")
        print("  acceptance        : " + ", ".join(f"{k}={v}" for k, v in ACCEPTANCE.items()))
        return 0

    os.makedirs(out_dir, exist_ok=True)
    reporter = DiagnosticReporter(
        stage=STAGE, enabled=True, dump=True, dump_dir=out_dir, run_label=run_alias,
        run_note=("Direct, non-neural estimate of the PSF-width population parameters. "
                  "Acceptance thresholds were fixed before the run and are quoted whatever "
                  "the outcome."))

    opts = dict(frame_stride=args.frame_stride, min_track_length=args.min_track_length,
                n_sigma=args.n_sigma, half_px=args.half_px)

    # ---- measure -------------------------------------------------------------------------
    if args.selftest:
        reporter.checkpoint("selftest", subunits=args.selftest_subunits,
                            frames=args.selftest_frames, scenes=9)
        res = run_selftest(reporter, n_subunits=args.selftest_subunits,
                           n_frames=args.selftest_frames, **opts)
        truth, estimate, se, reasons, theta_all = (res["truth"], res["estimate"], res["se"],
                                                   res["reasons"], res["theta"])
        per_video = dict(n_tracks=np.array([]), n_spots=np.array([]))
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
                     "but meaningless error. This check names that case rather than hiding it.")
            n_here = min(theta.shape[0], args.max_videos - len(jobs))
            for i in range(n_here):
                scope_row = {k: float(scope_arr[i, j])
                             for j, k in enumerate(det.DETECTOR_SCOPE_KEYS)}
                jobs.append((str(vpath), i, scope_row, opts))
                truth_rows.append(theta[i, :len(det.DETECTOR_FIND)])
            if len(jobs) >= args.max_videos:
                break

        reporter.stat("videos queued", len(jobs), note="videos the estimator will read")
        if args.workers and args.workers > 1:
            # Ordered map (results stay aligned with truth_rows) with a progress line every ~5 %.
            out = []
            every = max(1, len(jobs) // 20)
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for i, o in enumerate(pool.map(_worker, jobs, chunksize=1), 1):
                    out.append(o)
                    if i % every == 0 or i == len(jobs):
                        el = time.time() - t0
                        print(f"  [progress] {i}/{len(jobs)} recordings  elapsed {el/60:.1f} min  "
                              f"ETA {el/i*(len(jobs)-i)/60:.1f} min", flush=True)
        else:
            out = [_worker(j) for j in jobs]
        theta_all = np.asarray(truth_rows, dtype=float)          # PHYSICAL units, all six columns
        truth = theta_all[:, [det.DETECTOR_FIND["mu_r"], det.DETECTOR_FIND["sigma_r"]]]
        estimate = np.asarray([[o["mu_r"], o["sigma_r"]] for o in out], dtype=float)
        se = np.asarray([[o["log_mean_se"], o["sigma_r_se"]] for o in out], dtype=float)
        reasons = [o["reason"] for o in out]
        per_video = dict(n_tracks=np.array([o["n_tracks"] for o in out]),
                         n_spots=np.array([o["n_spots"] for o in out]))

    # ---- score ---------------------------------------------------------------------------
    # Not `truth.shape == estimate.shape`: those are built in the same loop and always agree,
    # so comparing them can never fail. What can fail is the pairing surviving the run.
    reporter.check("every queued recording produced a row",
                   truth.shape[0] == estimate.shape[0] and truth.shape[0] > 0,
                   f"{truth.shape[0]} truths against {estimate.shape[0]} estimates", fatal=False,
                   note="A mismatch would mean the worker pool dropped or reordered results, "
                        "which would silently pair each estimate with the wrong ground truth.")
    # ---- the frozen rules of DETECTOR_WORKFLOW.md sec. 9.6, evaluated by the shared kernel ----
    valid = np.isfinite(estimate).all(axis=1) & np.isfinite(truth).all(axis=1)

    def accuracy(mask):
        tm, em = truth[mask], estimate[mask]
        mu_err = np.log10(em[:, 0]) - np.log10(tm[:, 0])
        sg_err = em[:, 1] - tm[:, 1]
        sg_corr = (float(np.corrcoef(tm[:, 1], em[:, 1])[0, 1])
                   if tm.shape[0] > 2 and np.std(tm[:, 1]) > 0 and np.std(em[:, 1]) > 0 else None)
        mu_mae, mu_bias = float(np.abs(mu_err).mean()), float(mu_err.mean())
        sg_mae = float(np.abs(sg_err).mean())
        return {
            "mu_r MAE (dex)": (mu_mae, f"<= {ACCEPTANCE['mu_r_mae_dex']}", mu_mae <= ACCEPTANCE["mu_r_mae_dex"]),
            "mu_r |bias| (dex)": (abs(mu_bias), f"<= {ACCEPTANCE['mu_r_abs_bias_dex']}",
                                  abs(mu_bias) <= ACCEPTANCE["mu_r_abs_bias_dex"]),
            "sigma_r corr (linear)": (sg_corr if sg_corr is not None else float("nan"),
                                      f">= {ACCEPTANCE['sigma_r_corr']}",
                                      None if sg_corr is None else sg_corr >= ACCEPTANCE["sigma_r_corr"]),
            "sigma_r MAE (linear)": (sg_mae, f"<= {ACCEPTANCE['sigma_r_mae']}", sg_mae <= ACCEPTANCE["sigma_r_mae"]),
            "sigma_r bias (linear, reported)": (float(sg_err.mean()), "--", True),
        }

    # Nominal 90 % ranges, constructed as the companion note specifies: mu_r from the standard
    # error of the mean log width (normal, in log10); sigma_r from its own delta-method standard
    # error (normal, linear, floored at zero). Coverage against the truth is what validates them.
    z = 1.6448536269514722
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_c, mu_se10 = np.log10(estimate[:, 0]), se[:, 0] / np.log(10.0)
        mu_lo, mu_hi = mu_c - z * mu_se10, mu_c + z * mu_se10
        sg_lo, sg_hi = np.maximum(estimate[:, 1] - z * se[:, 1], 0.0), estimate[:, 1] + z * se[:, 1]
        mu_cov = (np.log10(truth[:, 0]) >= mu_lo) & (np.log10(truth[:, 0]) <= mu_hi)
        sg_cov = (truth[:, 1] >= sg_lo) & (truth[:, 1] <= sg_hi)
    lo_r, hi_r = da.prior_range("mu_r")
    lo_s, hi_s = da.prior_range("sigma_r")
    ranges = {"mu_r (log10)": (mu_cov, mu_hi - mu_lo, hi_r - lo_r),
              "sigma_r (linear)": (sg_cov, sg_hi - sg_lo, 10 ** hi_s - 10 ** lo_s)}

    result = da.evaluate(theta_all, valid, reasons, accuracy, target_key="mu_r",
                         ranges=ranges, selftest=bool(args.selftest))
    da.render(reporter, result, estimator=STAGE, target_key="mu_r")
    if per_video["n_tracks"].size:
        reporter.stat("median tracks per video", float(np.median(per_video["n_tracks"])),
                      note="linked spot tracks entering the population estimate")
        reporter.stat("median spot-frames per video", float(np.median(per_video["n_spots"])),
                      note="individual spot fits before linking")

    make_figure(truth, estimate, "direct_psf_width_truth_vs_estimate", reporter)

    np.savez_compressed(os.path.join(out_dir, "direct_psf_width.npz"),
                        truth=truth, estimate=estimate, theta=theta_all, valid=valid,
                        reasons=np.asarray([r or "" for r in reasons]), se=se,
                        range_mu_r_log10=np.column_stack([mu_lo, mu_hi]),
                        range_sigma_r=np.column_stack([sg_lo, sg_hi]), **per_video)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(dict(acceptance=ACCEPTANCE, evaluation=result), fh, indent=2, default=float)

    reporter.summary()
    path = reporter.write_report()
    print(f"\n[{STAGE}] report -> {path}")
    # The verdicts of sec. 9.6 must reach a caller that reads the exit status, not only a reader
    # of report.md: 0 = nothing failed, 1 = a FAIL verdict, 2 = insufficient evidence only.
    print(f"[{STAGE}] verdicts: " + "; ".join(f"{k}: {v}" for k, v in result["verdicts"].items()))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
