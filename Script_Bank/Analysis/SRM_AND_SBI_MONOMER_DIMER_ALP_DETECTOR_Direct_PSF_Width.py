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
        --condition FAB --total-time-seconds 2.0 --tasks 2 3 --max-videos 200 \\
        --purpose development --run-suffix <commit>

    ... --purpose verdict   # required for a tier run: development, or verdict for the reserved EVAL
                            # tasks of the tier in full (sec. 9.6)
    ... --selftest          # in-memory scenes rendered at known widths; needs no data tier
    ... --dry-run           # resolve profile, paths and settings, apply every refusal; read nothing

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/<alias>_<CONDITION>_<timing>_Direct_PSF_Width_<DEV|VERDICT>[_<suffix>]/
        report.md                       (the diagnostic report)
        direct_psf_width.npz            (per-video estimates, truths, recording identifiers and
                                         the observable per-recording quantities)
        summary.json                    (every number the report quotes, the purpose record)
        provenance.json                 (command, host, versions, implementation hash)
        figures/
    A tier run's folder carries its purpose (DEV for development, VERDICT for a verdict run), and
    --run-suffix appends to it, e.g. the commit: _DEV_<commit>. A run never reuses a folder: an
    existing one is refused before anything is read, so an earlier run is never overwritten.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
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
from srm_and_sbi_monomer_dimer_alp import provenance as prov  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import io as sio  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import labeling as lab  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.experiment_support import (  # noqa: E402
    discover_cells, read_cell_chunks,
)
from srm_and_sbi_monomer_dimer_alp import temporal_dynamics as tdk  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.visualization_inference import figure_window_drift  # noqa: E402
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
    """SCOPE camera values at the center of their box.

    Used by the selftest and by ``--experiment``: the box centers are the MET acquisition values
    of ``DETECTOR_WORKFLOW.md`` section 6.3 (gain-conversion ratio, optical offset, baseline,
    read noise, quantum efficiency), the same values the neural estimator marginalizes around.
    """
    return {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
            for e in det.DETECTOR_NUISANCE_SCOPE}


# ==========================================================================================
# One video -> one (mu_r, sigma_r) estimate
# ==========================================================================================

OBSERVABLE_KEYS = ("spot_photons_median", "spot_snr_median", "spots_per_frame",
                   "track_length_median", "tracks_per_spot")


def recording_observables(m: dict, scope: dict, *, track_lengths, n_tracks: int) -> dict:
    """Per-recording quantities that an experimental recording provides as well.

    ``DETECTOR_WORKFLOW.md`` sec. 9.6 allows a correction of the PSF estimates, or of their
    ranges, only through quantities available on experimental recordings, calibrated on
    development data and validated on the reserved EVAL tasks. These are candidate predictors for
    that, and none of them reads a true parameter:

      spot_photons_median   median fitted total signal of the accepted spot fits, in incident photons
      spot_snr_median       median fitted total signal over the per-pixel background noise
      spots_per_frame       accepted spot fits per used frame
      track_length_median   median length of the tracks the population estimate keeps, counted in
                            fits that pass its relative-error filter (``track_lengths``, as
                            `psf_width_population` returns them)
      tracks_per_spot       tracks kept (``n_tracks``) over the mean number of accepted fits per used
                            frame. Not a measure of fragmentation: it rises when tracks break, but
                            equally when emitters are visible for only part of the recording, bleach
                            or turn over -- two emitters each seen, perfectly linked, in a different
                            half of a recording give 2.
    """
    n_spots = int(np.asarray(m["sqrt2sigma"]).size)
    nan = float("nan")
    if n_spots == 0:
        return dict(spot_photons_median=nan, spot_snr_median=nan, spots_per_frame=0.0,
                    track_length_median=nan, tracks_per_spot=nan)
    noise = die.background_sigma_adu(scope["gamma"], scope["kappa_q"], scope["kappa_o"],
                                     scope["kappa_s"])
    amplitude = np.asarray(m["amplitude"], dtype=float)
    photons = die.adu_to_photons(amplitude, scope["gamma"], scope["kappa_q"])
    kept = np.asarray(track_lengths if track_lengths is not None else [], dtype=float)
    spots_per_frame = n_spots / max(int(m["n_frames_used"]), 1)
    return dict(spot_photons_median=float(np.median(photons)),
                spot_snr_median=float(np.median(amplitude) / noise),
                spots_per_frame=float(spots_per_frame),
                track_length_median=float(np.median(kept)) if kept.size else nan,
                tracks_per_spot=float(n_tracks / spots_per_frame) if n_tracks > 0 else nan)


def estimate_one(video_levels, scope: dict, *, frame_stride: int, min_track_length: int,
                 n_sigma: float, half_px: int, return_measurements: bool = False) -> dict:
    """Estimate ``(mu_r, sigma_r)`` from one stored video.

    Args:
        video_levels: ``(n_frames, n_x, n_y)`` stored 8-bit video.
        scope: the five SCOPE camera values in physical units.
        frame_stride: use every n-th frame.
        min_track_length: minimum linked-track length kept.
        n_sigma: spot-detection threshold.
        half_px: fit patch half-width.
        return_measurements: also return the per-spot measurements and track labels (the
            selftest's decomposition against the truth needs them; a tier run does not).

    Returns:
        ``dict`` with ``mu_r``, ``sigma_r``, ``sigma_r_raw``, ``log_mean_se`` (standard error of
        ``ln mu_r``), ``sigma_r_se``, ``noise_variance`` (the subtracted per-fit term),
        ``n_tracks``, ``n_spots``, the observable quantities of `recording_observables`, and
        ``reason`` -- ``None`` for a valid estimate, else the code for why none was produced
        (``no_spots``, ``too_few_tracks``).
    """
    m = die.measure_spot_widths(video_levels, scope, frame_stride=frame_stride,
                                n_sigma=n_sigma, half_px=half_px)
    if m["sqrt2sigma"].size == 0:
        tid = np.zeros(0, dtype=np.int64)
        out = dict(mu_r=np.nan, sigma_r=np.nan, sigma_r_raw=np.nan, log_mean_se=np.nan,
                   sigma_r_se=np.nan, noise_variance=np.nan, n_tracks=0, n_spots=0,
                   reason="no_spots",
                   **recording_observables(m, scope, track_lengths=None, n_tracks=0))
    else:
        tid = die.link_spot_tracks(m["frame_index"], m["x"], m["y"],
                                   frame_stride=frame_stride)
        r = die.psf_width_population(m["sqrt2sigma"], m["sqrt2sigma_se"],
                                     track_id=tid, min_track_length=min_track_length)
        valid = bool(np.isfinite(r["mu_r"]) and np.isfinite(r["sigma_r"]))
        n_tracks = int(r.get("n_tracks", 0))
        out = dict(mu_r=r["mu_r"], sigma_r=r["sigma_r"], sigma_r_raw=r["sigma_r_raw"],
                   log_mean_se=r["log_mean_se"], sigma_r_se=r["sigma_r_se"],
                   noise_variance=float(r["noise_variance"]), n_tracks=n_tracks,
                   n_spots=int(m["sqrt2sigma"].size),
                   reason=None if valid else "too_few_tracks",
                   **recording_observables(m, scope, track_lengths=r.get("track_lengths"),
                                           n_tracks=n_tracks))
    if return_measurements:
        out.update(measurements=m, track_id=tid)
    return out


# -- worker-side lazy handle, so a pool process opens each store once ----------------------
_STORE_CACHE: dict = {}


def _worker(job):
    """Pool worker: open (once) and measure one video. Must be top-level to pickle."""
    video_path, task, index, scope, opts = job
    handle = _STORE_CACHE.get(video_path)
    if handle is None:
        handle = sio.load_data(video_path)
        _STORE_CACHE[video_path] = handle
    video = np.asarray(handle[index])
    out = estimate_one(video, scope, **opts)
    out["task"] = int(task)
    out["index"] = int(index)
    return out


def _worker_array(job):
    """Pool worker for an in-memory window (the ``--experiment`` path). Top-level to pickle."""
    video, scope, opts = job
    return estimate_one(np.asarray(video), scope, **opts)


# ==========================================================================================
# Experimental recordings: one estimate per (cell, window), no ground truth
# ==========================================================================================

def run_experiment_mode(args, reporter, out_dir, paths, timing, data_bank_root, opts,
                        startup_code):
    """Estimate ``(mu_r, sigma_r)`` in every model-length window of every experimental recording.

    Mirrors the Experiment stage's windowing exactly (``read_cell_chunks``: 16-bit raw -> 8-bit,
    non-overlapping windows unless ``--chunk-step-seconds`` says otherwise) and the direct
    estimator's own range construction. The camera block is the section 6.3 acquisition value
    set (``_scope_center``). The recordings have no ground truth, so no acceptance verdict is
    reached; the report carries the per-window estimates, their nominal 90 % ranges, and the
    within-recording drift table and figure for the direct estimate, with its own range as the
    band. That range under-covers on synthetic recordings (about 66 % and 63 % at nominal 90 %,
    ``DETECTOR_WORKFLOW.md`` section 9.6), so the band is drawn as reported, not as validated.
    """
    span = args.experiment_span_seconds
    experiment_dir = (pathlib.Path(args.experiment_dir) if args.experiment_dir
                      else data_bank_root / paths.experiment_subdir)
    n_frames = timing.frame_count
    step_seconds = args.chunk_step_seconds if args.chunk_step_seconds else timing.total_time_seconds
    step_frames = int(round(step_seconds / timing.frame_time_seconds))
    cells = ([int(c) for c in args.cells.split(",")] if args.cells
             else discover_cells(experiment_dir, args.condition, span))
    if args.max_cells > 0:
        cells = cells[:args.max_cells]
    scope = _scope_center()
    reporter.checkpoint("experiment", condition=args.condition, cells=len(cells), span_s=span,
                        window_s=timing.total_time_seconds, step_s=step_seconds)
    reporter.check("experimental recordings found", len(cells) > 0,
                   f"{len(cells)} recordings under {experiment_dir}")
    for k, v in scope.items():
        reporter.stat(f"camera {k}", float(v),
                      note="section 6.3 acquisition value (box center), supplied, not fitted")

    jobs, meta = [], []
    for cell in cells:
        tif = experiment_dir / f"Experiment_{args.condition}_Cell_{cell}_{span}S_RAW.tif"
        if not tif.exists():
            reporter.check(f"recording cell {cell}", False, f"missing: {tif.name}", fatal=False)
            continue
        windows = read_cell_chunks(tif, n_frames, step_frames)
        for ci, w in enumerate(windows):
            jobs.append((w, scope, opts))
            meta.append((cell, ci))
    reporter.stat("windows queued", len(jobs), note="(cell, window) pairs the estimator reads")
    if not jobs:
        reporter.summary()
        reporter.write_report()
        return 1
    if args.workers and args.workers > 1:
        out, every, t0 = [], max(1, len(jobs) // 20), time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, o in enumerate(pool.map(_worker_array, jobs, chunksize=1), 1):
                out.append(o)
                if i % every == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"  [progress] {i}/{len(jobs)} windows  elapsed {el/60:.1f} min  "
                          f"ETA {el/i*(len(jobs)-i)/60:.1f} min", flush=True)
    else:
        out = [_worker_array(j) for j in jobs]

    estimate = np.asarray([[o["mu_r"], o["sigma_r"]] for o in out], dtype=float)
    se = np.asarray([[o["log_mean_se"], o["sigma_r_se"]] for o in out], dtype=float)
    reasons = np.asarray([o["reason"] or "" for o in out])
    n_tracks = np.asarray([o["n_tracks"] for o in out])
    n_spots = np.asarray([o["n_spots"] for o in out])
    cell_arr = np.asarray([m[0] for m in meta], dtype=int)
    chunk_arr = np.asarray([m[1] for m in meta], dtype=int)
    kind_index = np.zeros(len(meta), dtype=int)
    valid = np.isfinite(estimate).all(axis=1)
    z = 1.6448536269514722
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_c, mu_se10 = np.log10(estimate[:, 0]), se[:, 0] / np.log(10.0)
        mu_lo, mu_hi = mu_c - z * mu_se10, mu_c + z * mu_se10
        sg_lo = np.maximum(estimate[:, 1] - z * se[:, 1], 0.0)
        sg_hi = estimate[:, 1] + z * se[:, 1]

    # ---- report: distribution over windows, then drift across the windows of a recording ----
    lo_r, hi_r = da.prior_range("mu_r")
    lo_s, hi_s = da.prior_range("sigma_r")
    reporter.stat("valid windows", int(valid.sum()), expected=f"of {len(valid)}",
                  note="reason codes for the rest: " + (", ".join(
                      f"{r}={int((reasons == r).sum())}" for r in sorted(set(reasons[reasons != ""])))
                      or "none"))
    rows = []
    for name, vals, lo, hi, unit in (("mu_r", mu_c[valid], lo_r, hi_r, "log10"),
                                    ("sigma_r", estimate[valid, 1], 10 ** lo_s, 10 ** hi_s, "linear")):
        q25, q50, q75 = np.percentile(vals, [25, 50, 75]) if vals.size else (np.nan,) * 3
        outside = float(np.mean((vals < lo) | (vals > hi))) if vals.size else np.nan
        rows.append([name, unit, str(int(vals.size)), f"{q50:+.3f}" if unit == "log10" else f"{q50:.4f}",
                     f"{q75 - q25:.3f}", f"{100 * outside:.1f}%"])
    reporter.table("Direct PSF-width estimates over all (cell, window) pairs",
                   ["parameter", "units", "n", "median", "IQR", "outside prior"], rows,
                   note="no ground truth on experimental recordings: these are the estimator's "
                        "point values pooled over recordings and windows. mu_r in log10, sigma_r "
                        "in linear units, matching the frozen acceptance criteria.")
    est_log = np.column_stack([mu_c, estimate[:, 1]])          # mu_r log10; sigma_r linear
    grid, n_cells, n_chunks = tdk.reshape_to_grid(est_log, kind_index, cell_arr, chunk_arr, 1)
    if n_chunks >= 2:
        log_rows = np.array([True, False])
        to_phys = lambda u: np.stack([10 ** np.asarray(u)[..., 0], np.asarray(u)[..., 1]], -1)
        drows = tdk.window_drift_rows({"direct": grid}, [f"MET-{args.condition}"],
                                      ["mu_r", "sigma_r"], to_phys, log_rows)
        h, r = tdk.format_drift_rows(drows)
        reporter.table("Within-recording drift across windows (direct estimate)", h, r,
                       note="same construction as the Experiment stage's drift table: per "
                            "recording, a line fitted to the estimate against the window index, "
                            "first-to-last change aggregated across recordings. mu_r in dex, "
                            "sigma_r absolute (linear units).")
        # Band = the estimator's own nominal 90 % range (median across cells); no 50 % level.
        rng = np.full((len(meta), 2, 5), np.nan)
        rng[:, 0, 0], rng[:, 0, 4] = mu_lo, mu_hi
        rng[:, 1, 0], rng[:, 1, 4] = sg_lo, sg_hi
        bands = tdk.shared_posterior_bands(rng, kind_index, cell_arr, chunk_arr, 1)[0]
        fig = figure_window_drift(
            ["mu_r", "sigma_r"], [r"$\mu_{r}$", r"$\sigma_{r}$"],
            {"direct": tdk.window_medians(grid)[0]}, bands=bands,
            prior_ranges=[(lo_r, hi_r), (10 ** lo_s, 10 ** hi_s)],
            title=f"MET-{args.condition}: direct PSF-width estimate against window position "
                  f"(line = median across recordings; band = the estimator's own nominal 90 % "
                  f"range, median across recordings, drawn as reported, not as validated)",
            y_label="estimate (mu_r log10; sigma_r linear)", band_label="nominal range")
        reporter.save_figure(
            f"window_drift_direct_{args.condition}", fig,
            caption="Direct estimate against window position with its own nominal 90 % range as "
                    "the band. The range is a standard-error construction, not a posterior "
                    "interval, and under-covers on synthetic recordings (section 9.6); it is "
                    "drawn as reported.")

    np.savez_compressed(os.path.join(out_dir, "direct_psf_width_experiment.npz"),
                        estimate=estimate, se=se, valid=valid, reasons=reasons,
                        range_mu_r_log10=np.column_stack([mu_lo, mu_hi]),
                        range_sigma_r=np.column_stack([sg_lo, sg_hi]),
                        n_tracks=n_tracks, n_spots=n_spots, cell=cell_arr, chunk=chunk_arr,
                        kind_index=kind_index, kinds=np.asarray([args.condition]),
                        camera=np.asarray([scope[k] for k in det.DETECTOR_SCOPE_KEYS]),
                        camera_keys=np.asarray(list(det.DETECTOR_SCOPE_KEYS)))
    code = prov.finalize_code_provenance(startup_code, files=prov.DIRECT_ESTIMATOR_FILES)
    changed = bool(code["changed_during_run"])
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(dict(mode="experiment", condition=args.condition, span_seconds=span,
                       window_seconds=timing.total_time_seconds, step_seconds=step_seconds,
                       cells=[int(c) for c in cells], n_windows=int(len(meta)),
                       n_valid=int(valid.sum()), camera=scope,
                       implementation=("INVALID (implementation changed during the run)" if changed
                                       else "unchanged during the run"),
                       note="no ground truth; no acceptance verdict"), fh, indent=2, default=float)
    record = prov.analysis_run_record(
        startup_code, argv=sys.argv, code=code,
        extra=dict(stage=STAGE, mode="experiment", condition=args.condition, cells=[int(c) for c in cells],
                   span_seconds=span, step_seconds=step_seconds, windows=int(len(meta)),
                   run_suffix=args.run_suffix, out_dir=str(out_dir), settings=opts))
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    reporter.check("implementation unchanged during the run", not changed,
                   "implementation hash identical at startup and at write", fatal=False,
                   note="A file edited while the run executed leaves estimates that describe no single "
                        "implementation; the run exits with status 3 and provenance.json records both hashes.")
    reporter.summary()
    path = reporter.write_report()
    print(f"\n[{STAGE}] experiment report -> {path}")
    return da.EXIT_IMPLEMENTATION_CHANGED if changed else 0


# ==========================================================================================
# Selftest: in-memory scenes rendered at known widths
# ==========================================================================================

SELFTEST_GRIDS = ("widths", "brightness")
SELFTEST_LABELINGS = ("single", "FAB")
MATCH_RADIUS_PX = 1.5          # a spot fit within this distance of a true emitter is that emitter


def selftest_scenes(grid: str) -> list:
    """The selftest's scenes: the imaging values each one moves away from the prior center.

    ``widths`` (the default) is the 3 x 3 grid of ``mu_r`` and ``sigma_r``. ``brightness`` puts
    ``mu_pc`` on the five edges of its prior quarters (100 to 562 photons per dye), each at three
    PSF spreads with ``mu_r`` at its prior center: the dim-subgroup bias of
    ``DETECTOR_WORKFLOW.md`` sec. 9.6 runs with the true brightness, and this grid is where the
    decomposition against the truth can say why.
    """
    if grid == "widths":
        return [dict(mu_r=mu_r, sigma_r=sigma_r)
                for mu_r in (1.10, 1.41, 1.85) for sigma_r in (0.10, 0.237, 0.50)]
    if grid == "brightness":
        return [dict(mu_pc=float(10 ** q), sigma_r=sigma_r)
                for q in da.prior_quarters("mu_pc") for sigma_r in (0.10, 0.237, 0.50)]
    raise ValueError(f"unknown selftest grid {grid!r}; use one of {SELFTEST_GRIDS}")


def truth_decomposition(m: dict, track_id, true_sqrt2sigma, true_xy, visible, mu_r_true: float,
                        mu_r_estimate: float, *, min_track_length: int,
                        max_relative_se: float = 0.5) -> dict:
    """Account for one selftest scene's ``mu_r`` error against the truth, term by term.

    Each accepted spot fit is matched to the nearest visible subunit in its frame (within
    ``MATCH_RADIUS_PX``), and each track kept by the population step (same relative-error filter,
    same length floor) to the subunit most of its fits matched. The five terms below sum to the
    total by construction, because ``other`` is the remainder. It is diagnostic accounting,
    conditional on that matching and on the order in which the terms are taken, not a uniquely
    established causal decomposition: another matching rule or another order would split the same
    total differently. In log10 units:

      sample       mean true log width of all visible subunits, minus log10 of the true mu_r:
                   the finite population the scene happened to draw
      selection    mean true log width of the subunits the kept tracks found, minus that of all
                   visible subunits: which subunits detection and the length floor let through
      duplication  mean true log width over tracks, minus that over the distinct subunits found:
                   a subunit split into several tracks counts several times
      fitting      mean over matched tracks of the fitted minus the true log width
      other        the remainder of the total error: the tail trim, unmatched tracks, and the
                   difference between these track means and the population step's own

    ``detected_share`` is the fraction of visible subunits found, ``tracks_per_subunit`` how many
    kept tracks each found subunit produced, and ``spread_ratio`` the standard deviation of the
    true log widths of the found subunits over that of all visible ones (below one, selection
    compresses the spread that ``sigma_r`` measures).
    """
    nan = float("nan")
    true_lw = np.log10(np.asarray(true_sqrt2sigma, dtype=float))
    vis_idx = np.nonzero(np.asarray(visible, dtype=bool))[0]
    frames = np.asarray(m["frame_index"])
    xs, ys = np.asarray(m["x"], dtype=float), np.asarray(m["y"], dtype=float)
    matched = np.full(frames.size, -1, dtype=np.int64)
    for t in np.unique(frames):
        sel = np.nonzero(frames == t)[0]
        pos = np.asarray(true_xy)[t][vis_idx]
        good = np.isfinite(pos).all(axis=1)
        if not good.any():
            continue
        cand, pos = vis_idx[good], pos[good]
        d = np.hypot(xs[sel, None] - pos[None, :, 0], ys[sel, None] - pos[None, :, 1])
        j = np.argmin(d, axis=1)
        close = d[np.arange(sel.size), j] <= MATCH_RADIUS_PX
        matched[sel[close]] = cand[j[close]]

    w = np.asarray(m["sqrt2sigma"], dtype=float)
    se = np.asarray(m["sqrt2sigma_se"], dtype=float)
    ok = np.isfinite(w) & np.isfinite(se) & (w > 0) & (se > 0)
    ok &= (se / np.maximum(w, 1e-12)) <= float(max_relative_se)
    tid = np.asarray(track_id, dtype=np.int64)
    track_lw, track_sub = [], []
    for k in np.unique(tid[ok]):
        sel = ok & (tid == k)
        if sel.sum() < int(min_track_length):
            continue
        weight = 1.0 / np.maximum((se[sel] / w[sel]) ** 2, 1e-12)
        track_lw.append(float(np.sum(weight * np.log10(w[sel])) / np.sum(weight)))
        ids = matched[sel]
        ids = ids[ids >= 0]
        track_sub.append(int(np.bincount(ids).argmax()) if ids.size else -1)
    track_lw, track_sub = np.asarray(track_lw), np.asarray(track_sub, dtype=np.int64)
    has = track_sub >= 0
    found = np.unique(track_sub[has])

    total = (float(np.log10(mu_r_estimate) - np.log10(mu_r_true))
             if np.isfinite(mu_r_estimate) and mu_r_estimate > 0 else nan)
    lw_vis = float(true_lw[vis_idx].mean()) if vis_idx.size else nan
    sample = lw_vis - float(np.log10(mu_r_true))
    if found.size:
        selection = float(true_lw[found].mean()) - lw_vis
        duplication = float(true_lw[track_sub[has]].mean() - true_lw[found].mean())
        fitting = float(np.mean(track_lw[has] - true_lw[track_sub[has]]))
        sd_vis = float(np.std(true_lw[vis_idx], ddof=1)) if vis_idx.size > 1 else nan
        spread_ratio = (float(np.std(true_lw[found], ddof=1)) / sd_vis
                        if found.size > 1 and sd_vis > 0 else nan)
    else:
        selection = duplication = fitting = spread_ratio = nan
    parts = (sample, selection, duplication, fitting)
    other = (total - sum(parts)) if all(np.isfinite(v) for v in (total, *parts)) else nan
    return dict(total=total, sample=sample, selection=selection, duplication=duplication,
                fitting=fitting, other=other,
                detected_share=float(found.size / vis_idx.size) if vis_idx.size else nan,
                tracks_per_subunit=float(has.sum() / found.size) if found.size else nan,
                unmatched_track_share=float((~has).mean()) if track_sub.size else nan,
                spread_ratio=spread_ratio, n_visible=int(vis_idx.size),
                n_found=int(found.size), n_tracks=int(track_sub.size))


DECOMPOSITION_KEYS = ("total", "sample", "selection", "duplication", "fitting", "other",
                      "detected_share", "tracks_per_subunit", "unmatched_track_share",
                      "spread_ratio")
# The additive terms of `truth_decomposition`: total = sample + selection + duplication + fitting + other.
DECOMPOSITION_TERMS = ("sample", "selection", "duplication", "fitting", "other")


def decomposition_table(dec: np.ndarray, scenes: list) -> tuple:
    """Headers and rows of the selftest's decomposition table: per scene and as a mean over scenes,
    the total, every additive term (so the displayed terms sum to the displayed total, up to
    rounding), and the matching diagnostics."""
    col = {k: j for j, k in enumerate(DECOMPOSITION_KEYS)}
    headers = (["scene", "total"] + list(DECOMPOSITION_TERMS)
               + ["subunits found", "tracks per subunit", "spread ratio"])

    def cells(v):
        return ([f"{v[col['total']]:+.4f}"] + [f"{v[col[k]]:+.4f}" for k in DECOMPOSITION_TERMS]
                + [f"{v[col['detected_share']]:.0%}", f"{v[col['tracks_per_subunit']]:.2f}",
                   f"{v[col['spread_ratio']]:.2f}"])

    rows = [[", ".join(f"{k} {v:.4g}" for k, v in sc.items())] + cells(dec[i])
            for i, sc in enumerate(scenes)]
    with np.errstate(invalid="ignore"):
        rows.append(["mean over scenes"] + cells(np.nanmean(dec, axis=0)))
    return headers, rows


def run_selftest(reporter: DiagnosticReporter, *, n_subunits: int, n_frames: int,
                 frame_stride: int, min_track_length: int, n_sigma: float,
                 half_px: int, grid: str = "widths", labeling: str = "single") -> dict:
    """Render diffusing scenes at known imaging values and recover the PSF parameters.

    Subunits on Brownian paths, rendered through the production renderer with every imaging
    parameter at its prior center except the ones the grid moves (`selftest_scenes`) and with
    bleaching off. ``labeling="single"`` gives every subunit one dye, which isolates the
    estimator from the labeling law; ``"FAB"`` draws dye counts from the bare FAB dye-count law at
    probe occupancy 1 (about 81 % of subunits carry a dye). A tier also applies the condition's
    probe occupancy (0.155 for MET-FAB), which thins the visible subunits without changing a
    visible one's dye count; here the density of visible spots is set by ``n_subunits`` instead.
    Every scene is also decomposed against its truth (`truth_decomposition`): the widths are
    reproduced from the renderer's own seeded draw and the positions are the scene's.
    """
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import (
        render_dli_video, sample_psf_width,
    )

    stem = PARAMETERS.simulation.stem
    npx, px_nm = stem.root_size_px, stem.pixel_size_nm
    dt = PARAMETERS.simulation.timing.frame_time_seconds
    scope = _scope_center()
    scenes = selftest_scenes(grid)
    if labeling not in SELFTEST_LABELINGS:
        raise ValueError(f"unknown selftest labeling {labeling!r}; use one of {SELFTEST_LABELINGS}")
    law = lab.resolve_labeling_law("FAB")[1] if labeling == "FAB" else None
    center = {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
              for e in det.DETECTOR_IMAGING}

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

    truths, ests, ses, reasons, thetas, decomp, observables = [], [], [], [], [], [], []
    for i, overrides in enumerate(scenes):
        seed = SELFTEST_SEED + i
        img = dict(center)
        img.update(prob_photo_bleach=1e-12, **overrides)
        thetas.append([img[k] for k in det.DETECTOR_FIND])
        vec = np.array([img[k] for k in det.DETECTOR_IMAGING_KEYS])
        poses, host = scene(seed)
        dye_counts = (np.ones(n_subunits, dtype=np.int64) if law is None
                      else lab.draw_dye_counts(law, n_subunits, np.random.default_rng(seed + 1000)))
        frames = render_dli_video(poses, host, dye_counts, vec, seed=seed)
        video = sio.convert_video_dtype(np.moveaxis(frames, 2, 0), bits_from=16, bits_to=8)
        r = estimate_one(video, scope, frame_stride=frame_stride,
                         min_track_length=min_track_length, n_sigma=n_sigma,
                         half_px=half_px, return_measurements=True)
        true_w = sample_psf_width(n_subunits, PARAMETERS.simulation.dli.sqrt_2sigma_dist_label,
                                  keyword_args={"mu_r": img["mu_r"], "sigma_r": img["sigma_r"]},
                                  seed=seed)
        dec = truth_decomposition(r["measurements"], r["track_id"], true_w,
                                  poses[:, :, :2] / px_nm, dye_counts > 0, img["mu_r"],
                                  r["mu_r"], min_track_length=min_track_length)
        truths.append((img["mu_r"], img["sigma_r"]))
        ests.append((r["mu_r"], r["sigma_r"]))
        ses.append((r["log_mean_se"], r["sigma_r_se"]))
        reasons.append(r["reason"])
        decomp.append([dec[k] for k in DECOMPOSITION_KEYS])
        observables.append([r[k] for k in OBSERVABLE_KEYS])
        label = ", ".join(f"{k} {v:.4g}" for k, v in overrides.items())
        print(f"  [{i + 1}/{len(scenes)}] {label}: mu_r {img['mu_r']:.3f} -> {r['mu_r']:.4f} "
              f"(total {dec['total']:+.4f} dex = " + " + ".join(
                  f"{k} {dec[k]:+.4f}" for k in DECOMPOSITION_TERMS)
              + f"), sigma_r {img['sigma_r']:.3f} -> {r['sigma_r']:.4f}; "
              f"found {dec['detected_share']:.0%} of {dec['n_visible']} visible, "
              f"{dec['tracks_per_subunit']:.2f} tracks each", flush=True)

    return dict(truth=np.asarray(truths, dtype=float), estimate=np.asarray(ests, dtype=float),
                se=np.asarray(ses, dtype=float), reasons=reasons,
                theta=np.asarray(thetas, dtype=float), scenes=scenes,
                decomposition=np.asarray(decomp, dtype=float),
                observables=np.asarray(observables, dtype=float))


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
    ap.add_argument("--experiment", action="store_true",
                    help="estimate on the EXPERIMENTAL recordings of --condition instead of a "
                         "synthetic tier: every model-length window of every recording, windowed "
                         "exactly as the Experiment stage windows them, camera from the section "
                         "6.3 acquisition values. No ground truth, so no acceptance verdict.")
    ap.add_argument("--experiment-span-seconds", type=int, default=20,
                    help="length of the experimental recordings (selects the RAW .tif files).")
    ap.add_argument("--chunk-step-seconds", type=float, default=None,
                    help="window step for --experiment; default = the window length "
                         "(non-overlapping), the Experiment stage's default.")
    ap.add_argument("--cells", type=str, default=None,
                    help="comma-separated recording indices for --experiment; default: all.")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="cap on the number of recordings for --experiment (0 = all).")
    ap.add_argument("--experiment-dir", default=None,
                    help="override the recordings directory for --experiment (default: "
                         "<data_bank_root>/<experiment_subdir>, where the Experiment stage reads).")
    ap.add_argument("--selftest-grid", default="widths", choices=SELFTEST_GRIDS,
                    help="selftest scenes: 'widths' (3 x 3 in mu_r and sigma_r) or 'brightness' "
                         "(mu_pc on its prior-quarter edges x 3 sigma_r).")
    ap.add_argument("--selftest-labeling", default="single", choices=SELFTEST_LABELINGS,
                    help="selftest dye counts: one dye per subunit, or drawn from the FAB law.")
    ap.add_argument("--purpose", default=None, choices=list(da.PURPOSES),
                    help="required for a tier run: why it reads its tasks (DETECTOR_WORKFLOW.md "
                         "sec. 9.6). 'development' reads development tasks only; 'verdict' reads "
                         "exactly the reserved EVAL tasks of the tier, in full, once per estimator. "
                         "Checked before any recording is read; the folder name carries DEV or "
                         "VERDICT. Not accepted by a run that reads no EVAL task.")
    ap.add_argument("--run-suffix", default=None,
                    help="token appended to the run folder name after the purpose token, e.g. the "
                         "commit (-> _DEV_<commit>); letters, digits and underscores.")
    args = ap.parse_args(argv)

    if not args.selftest and (args.condition is None or args.total_time_seconds is None):
        ap.error("--condition and --total-time-seconds are required (or use --selftest)")
    if args.run_suffix is not None and not re.fullmatch(r"[A-Za-z0-9_]+", args.run_suffix):
        ap.error(f"--run-suffix {args.run_suffix!r}: use letters, digits and underscores only")
    if args.run_suffix is not None and re.match(r"(?i)(DEV|VERDICT|SELFTEST)(_|$)", args.run_suffix):
        ap.error(f"--run-suffix {args.run_suffix!r}: the purpose token (DEV, VERDICT, SELFTEST) is added "
                 f"automatically; pass only what follows it, e.g. the commit")
    tier_run = not (args.selftest or args.experiment)
    if tier_run and args.purpose is None:
        ap.error("--purpose is required for a tier run: development or verdict (DETECTOR_WORKFLOW.md sec. 9.6)")
    if not tier_run and args.purpose is not None:
        ap.error("--purpose applies to a tier run; a selftest or an experimental run reads no EVAL task")

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
    elif args.experiment:
        timing = RunTiming(total_time_seconds=args.total_time_seconds)
        timing_label = timing.label
        paths = det.detector_paths(PARAMETERS.paths).with_condition(args.condition)
        run_alias = f"{paths.project_alias}_{timing_label}"
        video_paths, theta_paths, scope_paths = [], [], []
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

    # ---- purpose: held to the declared split before any recording is read ------------------
    purpose_record = dict(purpose="selftest" if args.selftest else "experiment")
    if tier_run:
        try:
            purpose_record = da.check_run_purpose(
                args.purpose, estimator=STAGE, condition=args.condition, n_frames=timing.frame_count,
                split=args.split, tasks=args.tasks, max_videos=args.max_videos,
                expect_videos_per_task=args.expect_videos_per_task)
        except ValueError as exc:
            ap.error(str(exc))

    descriptor = (f"{STAGE}_SELFTEST" if args.selftest
                  else f"{STAGE}_Experiment" if args.experiment
                  else f"{STAGE}_{da.PURPOSE_TOKENS[args.purpose]}")
    if args.selftest and args.selftest_grid != "widths":
        descriptor += f"_{args.selftest_grid.upper()}"
    if args.selftest and args.selftest_labeling != "single":
        descriptor += f"_{args.selftest_labeling}_LAW"
    if args.run_suffix:
        descriptor += f"_{args.run_suffix}"
    posit_dir = os.path.join(str(data_bank_root), "Posit")
    out_dir = args.out_dir or os.path.join(posit_dir, f"{run_alias}_{descriptor}")
    if tier_run and args.purpose == "verdict":
        # The reserved tasks are judged once per estimator (sec. 9.6): a verdict run writes to its
        # fixed Posit location, where an earlier verdict folder of the same estimator and tier is seen.
        if args.out_dir:
            ap.error("a verdict run writes to its fixed Posit folder; --out-dir is for development runs")
        earlier = sorted(glob.glob(os.path.join(posit_dir, f"{run_alias}_{STAGE}_VERDICT*")))
        if earlier:
            ap.error(f"a verdict folder of {STAGE} on this tier exists already ({earlier[0]}); the "
                     f"reserved tasks are judged once (DETECTOR_WORKFLOW.md sec. 9.6)")
    if os.path.exists(out_dir):
        ap.error(f"the run folder exists already: {out_dir}. A run never reuses a folder: pass another "
                 f"--run-suffix, or move the earlier run aside deliberately.")

    # ---- dry run -------------------------------------------------------------------------
    if args.dry_run:
        print(f"[{STAGE}] DRY RUN -- nothing is read and nothing is written.")
        print(f"  machine profile   : {os.environ.get('MACHINE_PROFILE', '(unset)')}")
        print(f"  data_bank_root    : {data_bank_root}")
        print(f"  mode              : "
              f"{'selftest' if args.selftest else 'experiment' if args.experiment else args.split}")
        print(f"  run alias         : {run_alias}")
        print(f"  out dir           : {out_dir}   (new; created at the start of the run)")
        if tier_run:
            pr = purpose_record
            print(f"  purpose           : {pr['purpose']} -- tasks {pr['tasks']}; "
                  + (f"declared split: development {pr['development_tasks'][0]}-{pr['development_tasks'][-1]}, "
                     f"reserved {pr['reserved_tasks'][0]}-{pr['reserved_tasks'][-1]}"
                     if pr["split_declared"] else "no declared split for this tier (no reserved task)"))
        print(f"  frame stride      : {args.frame_stride}   min track {args.min_track_length}"
              f"   n_sigma {args.n_sigma}   half_px {args.half_px}")
        print(f"  workers           : {args.workers}")
        if args.selftest:
            print(f"  selftest scenes   : {len(selftest_scenes(args.selftest_grid))} "
                  f"({args.selftest_grid} grid, {args.selftest_labeling} labeling), "
                  f"{args.selftest_subunits} subunits x {args.selftest_frames} frames")
        elif args.experiment:
            exp_dir = (pathlib.Path(args.experiment_dir) if args.experiment_dir
                       else data_bank_root / paths.experiment_subdir)
            span = args.experiment_span_seconds
            cells = ([int(c) for c in args.cells.split(",")] if args.cells
                     else discover_cells(exp_dir, args.condition, span))
            step = args.chunk_step_seconds or args.total_time_seconds
            n_win = int((span - args.total_time_seconds) // step) + 1
            print(f"  recordings dir    : {exp_dir}   exists={exp_dir.exists()}")
            print(f"  recordings        : {len(cells)} x {span} s -> {n_win} windows of "
                  f"{args.total_time_seconds:g} s each (step {step:g} s)")
            print("  camera (sec. 6.3) : " + ", ".join(f"{k}={v:.4g}" for k, v in _scope_center().items()))
            print("  ground truth      : none -- no acceptance verdict; drift table + figure only")
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

    startup_code = prov.code_provenance(files=prov.DIRECT_ESTIMATOR_FILES)
    try:
        prov.reserve_run_folder(out_dir)
    except FileExistsError:
        ap.error(f"the run folder appeared after the check (a concurrent run?): {out_dir}")
    reporter = DiagnosticReporter(
        stage=STAGE, enabled=True, dump=True, dump_dir=out_dir, run_label=run_alias,
        run_note=("Direct, non-neural estimate of the PSF-width population parameters. "
                  "Acceptance thresholds were fixed before the run and are quoted whatever "
                  "the outcome."))

    opts = dict(frame_stride=args.frame_stride, min_track_length=args.min_track_length,
                n_sigma=args.n_sigma, half_px=args.half_px)
    if args.experiment:
        return run_experiment_mode(args, reporter, out_dir, paths, timing, data_bank_root, opts,
                                   startup_code)
    if tier_run:
        reporter.stat("run purpose", args.purpose,
                      note=(f"tasks {purpose_record['tasks']}; " + (
                          f"declared split: development {purpose_record['development_tasks'][0]}-"
                          f"{purpose_record['development_tasks'][-1]}, reserved "
                          f"{purpose_record['reserved_tasks'][0]}-{purpose_record['reserved_tasks'][-1]} "
                          f"(DETECTOR_WORKFLOW.md sec. 9.6)" if purpose_record["split_declared"]
                          else "no declared split for this tier, so no reserved task")))

    # ---- measure -------------------------------------------------------------------------
    if args.selftest:
        reporter.checkpoint("selftest", subunits=args.selftest_subunits,
                            frames=args.selftest_frames, grid=args.selftest_grid,
                            labeling=args.selftest_labeling,
                            scenes=len(selftest_scenes(args.selftest_grid)))
        res = run_selftest(reporter, n_subunits=args.selftest_subunits,
                           n_frames=args.selftest_frames, grid=args.selftest_grid,
                           labeling=args.selftest_labeling, **opts)
        truth, estimate, se, reasons, theta_all = (res["truth"], res["estimate"], res["se"],
                                                   res["reasons"], res["theta"])
        per_video = dict(n_tracks=np.array([]), n_spots=np.array([]),
                         scene=np.arange(truth.shape[0]),
                         decomposition=res["decomposition"],
                         decomposition_keys=np.asarray(DECOMPOSITION_KEYS),
                         **{k: res["observables"][:, j] for j, k in enumerate(OBSERVABLE_KEYS)})
        headers, rows = decomposition_table(res["decomposition"], res["scenes"])
        reporter.table(
            "mu_r error accounted against the truth (log10 units)", headers, rows,
            note="total = sample + selection + duplication + fitting + other. sample: the finite "
                 "population the scene drew (part of the total, not a property of the estimator); "
                 "selection: the found subunits' true widths against all visible subunits'; "
                 "duplication: a subunit whose detections formed several tracks counts several "
                 "times; fitting: fitted against true width within matched tracks; other: the "
                 "remainder (tail trim, unmatched tracks, the population step's own averaging). "
                 "Diagnostic accounting conditional on nearest-subunit matching within "
                 f"{MATCH_RADIUS_PX} px and on this order of terms, not a uniquely established causal "
                 "decomposition. A spread ratio below one means the found subunits span a narrower "
                 "range of true widths than all visible ones.")
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
                jobs.append((str(vpath), t, i, scope_row, opts))
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
                         n_spots=np.array([o["n_spots"] for o in out]),
                         task=np.array([o["task"] for o in out], dtype=np.int64),
                         index=np.array([o["index"] for o in out], dtype=np.int64),
                         sigma_r_raw=np.array([o["sigma_r_raw"] for o in out], dtype=float),
                         noise_variance=np.array([o["noise_variance"] for o in out], dtype=float),
                         **{k: np.array([o[k] for o in out], dtype=float)
                            for k in OBSERVABLE_KEYS})

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
    # The implementation hash is compared once every estimate exists and before anything is
    # written: a run whose code changed while it executed is invalid for acceptance (exit 3).
    code = prov.finalize_code_provenance(startup_code, files=prov.DIRECT_ESTIMATOR_FILES)
    da.apply_code_provenance(result, code)
    da.render(reporter, result, estimator=STAGE, target_key="mu_r")
    if per_video["n_tracks"].size:
        reporter.stat("median tracks per video", float(np.median(per_video["n_tracks"])),
                      note="linked spot tracks entering the population estimate")
        reporter.stat("median spot-frames per video", float(np.median(per_video["n_spots"])),
                      note="individual spot fits before linking")
    if per_video.get("spot_photons_median") is not None and per_video["spot_photons_median"].size:
        with np.errstate(invalid="ignore"):
            reporter.table(
                "Observable per-recording quantities (medians over recordings)",
                ["quantity", "median"],
                [[k, f"{np.nanmedian(per_video[k]):.4g}"] for k in OBSERVABLE_KEYS],
                note="Candidates for a correction or a range construction calibrated on development "
                     "data (DETECTOR_WORKFLOW.md sec. 9.6); every one is available on an experimental "
                     "recording, and none reads a true parameter. Per recording in the arrays.")

    make_figure(truth, estimate, "direct_psf_width_truth_vs_estimate", reporter)

    np.savez_compressed(os.path.join(out_dir, "direct_psf_width.npz"),
                        truth=truth, estimate=estimate, theta=theta_all, valid=valid,
                        reasons=np.asarray([r or "" for r in reasons]), se=se,
                        range_mu_r_log10=np.column_stack([mu_lo, mu_hi]),
                        range_sigma_r=np.column_stack([sg_lo, sg_hi]), **per_video)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(dict(acceptance=ACCEPTANCE, purpose=purpose_record, evaluation=result), fh,
                  indent=2, default=float)
    record = prov.analysis_run_record(
        startup_code, argv=sys.argv, code=code,
        extra=dict(stage=STAGE, mode="selftest" if args.selftest else args.split,
                   purpose=purpose_record, tasks=None if args.selftest else list(args.tasks),
                   selftest_grid=args.selftest_grid if args.selftest else None,
                   selftest_labeling=args.selftest_labeling if args.selftest else None,
                   recordings=int(truth.shape[0]), run_suffix=args.run_suffix,
                   out_dir=str(out_dir), settings=opts))
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    reporter.check("implementation unchanged during the run", not code["changed_during_run"],
                   "implementation hash identical at startup and at write", fatal=False,
                   note="A file edited while the run executed leaves results that describe no single "
                        "implementation: the run is INVALID for acceptance, exits with status 3, and "
                        "provenance.json records both hashes. The arrays are kept as diagnostics.")

    reporter.summary()
    path = reporter.write_report()
    print(f"\n[{STAGE}] report -> {path}")
    # The verdicts of sec. 9.6 must reach a caller that reads the exit status, not only a reader
    # of report.md: 0 = nothing failed, 1 = a FAIL verdict, 2 = insufficient evidence only,
    # 3 = the implementation changed during the run (invalid for acceptance).
    print(f"[{STAGE}] verdicts: " + "; ".join(f"{k}: {v}" for k, v in result["verdicts"].items()))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
