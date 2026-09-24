"""Flicker mismatch study: where along the observation chain the direct flicker estimator's bias arises.

The direct flicker-rate estimator (`..._DETECTOR_Direct_Flicker_Rate.py`) reads ``lambda_rate`` off
the pooled, detrended, lag-1-normalized ln-intensity autocorrelation of linked spot traces, matched
against a single-dye Ornstein-Uhlenbeck model arm cut to the observed spans and detrended
identically. On the two-second multiple-dye development recordings it runs fast: +0.11 dex overall,
+0.26 dex at the slowest rates and +0.16 dex in the dim half (`DETECTOR_WORKFLOW.md` sec. 9.6). The
model arm reproduces the spans and the detrend, but not the rest of the observation chain the traces
pass through, and which part of that chain contributes the bias is not isolated.

This harness looks for it. It renders scenes with known truth through the production renderer,
reproduces every dye's true photon series from the renderer's own seeded draw, runs the production
measurement (detection, spot fits, linking) on the rendered video, and applies the SAME shape match
to seven versions of the traces. Each version changes one thing against the one before it:

    dye_truth_full       every dye's true photon series over the whole recording: what the
                         single-dye model arm assumes, so its error is the method's own floor
    spot_truth_full      every visible subunit's summed true photons over the whole recording:
                         adds dye multiplicity (a spot is the sum of its dyes)
    spot_truth_span      that series over the span from the subunit's first to its last
                         detection, without gaps: adds the selection of spans
    spot_truth_detected  that series at the detected frames only: adds the detection gaps, i.e.
                         censoring of the dim frames
    fitted_oracle        the fitted amplitudes at those detections, grouped by true identity:
                         adds photometry noise
    fitted_linked        the SAME detections grouped by the production linker: changes the
                         grouping only, so the contrast with fitted_oracle isolates linking
    production           every accepted detection grouped by the production linker, the estimator
                         as it runs on a tier: adds the detections outside the matched set (fits with
                         no visible subunit within the match radius, and second fits near a subunit
                         already matched in that frame)

A contrast between adjacent levels, paired within scenes, suggests a contribution of that step under
these scenes and this ordering. The steps are added in one fixed order, and an effect measured after
one step can differ in size, even in sign, when taken in another; the contrasts are diagnostic
accounting, not a causal decomposition. Every level records why it produced no estimate where it
produced none, and every mean is reported with the number of scenes behind it.

Dye counts follow the bare dye-count law of the chosen condition at probe occupancy 1 (for FAB about
81 % of subunits carry a dye); a tier also applies the condition's probe occupancy (0.155 for MET-FAB),
which thins the visible subunits without changing a visible one's dye count. Here the density of
visible spots is set by ``--n-subunits`` instead.

The harness reads no EVAL task, so it consumes none of the development or reserved data of sec. 9.6;
it tunes nothing and reaches no verdict. It is a diagnostic utility, never wired into the stage
dispatcher.

Usage (from the repo root):
    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Mismatch.py \\
        --workers 16

    ... --lambda-grid 1.25 2 4 8 --mu-pc-grid 120 240 480 --replicates 2   # the default scene grid
    ... --labeling single        # one dye per subunit instead of the FAB law
    ... --prob-photo-bleach 0.056  # bleaching on (off by default)
    ... --dry-run                # print the plan and apply the refusals; render and compute nothing

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/<alias>_<timing>_Direct_Flicker_Mismatch_<labeling>[_BLEACH_<p>][_<suffix>]/
        report.md, direct_flicker_mismatch.npz, summary.json, provenance.json, figures/
    The labeling (FAB_LAW, INLB_LAW or SINGLE_DYE) and, when bleaching is on, its probability are part
    of the folder name, so the documented variants never share a folder. A run never reuses a folder:
    an existing one is refused before anything is rendered; any other change needs its own
    --run-suffix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import direct_acceptance as da  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import direct_imaging_estimates as die  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import io as sio  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import labeling as lab  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import provenance as prov  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import simulation_dli_support as dli  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import (  # noqa: E402
    PARAMETERS, RunTiming,
)

assert os.path.abspath(die.__file__).startswith(REPO_ROOT), (
    "Imported srm_and_sbi_monomer_dimer_alp from outside this repo checkout: "
    + die.__file__ + " -- run with PYTHONPATH set to the repo root."
)

STAGE = "Direct_Flicker_Mismatch"
LEVELS = ("dye_truth_full", "spot_truth_full", "spot_truth_span", "spot_truth_detected",
          "fitted_oracle", "fitted_linked", "production")
LEVEL_NOTES = {
    "dye_truth_full": "each dye's true photons, whole recording (the model arm's own assumption)",
    "spot_truth_full": "each visible subunit's summed true photons, whole recording",
    "spot_truth_span": "that series from first to last detection, no gaps",
    "spot_truth_detected": "that series at the detected frames only",
    "fitted_oracle": "fitted amplitudes of the matched detections, grouped by true identity",
    "fitted_linked": "the same matched detections, grouped by the production linker",
    "production": "every accepted detection, grouped by the production linker: the estimator on a tier",
}
# What changes between each level and the one before it.
STEP_NOTES = {
    "spot_truth_full": "dye multiplicity: a spot is the sum of its dyes",
    "spot_truth_span": "span selection: each subunit's detected span",
    "spot_truth_detected": "detection gaps: undetected frames become gaps",
    "fitted_oracle": "photometry: fitted amplitudes replace the true photons",
    "fitted_linked": "linking only: the same detections, grouped by the linker",
    "production": "detections outside the matched set: unmatched fits and second fits near a matched subunit",
}
FAILURE_REASONS = ("too_few_traces", "too_few_pairs")
MATCH_RADIUS_PX = 1.5       # a spot fit within this distance of a true emitter is that emitter
DIFFUSION_UM2_PER_S = 0.05  # a typical receptor, as in the direct estimators' selftests
SEED_BASE = 20260924
BLEACH_OFF = 1e-12          # the renderer's stand-in for no bleaching
UNMATCHED, SECOND_FIT = -1, -2   # codes of the detections outside the matched set


def _scope_center() -> dict:
    return {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
            for e in det.DETECTOR_NUISANCE_SCOPE}


def _imaging_center() -> dict:
    return {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
            for e in det.DETECTOR_IMAGING}


def run_descriptor(labeling: str, prob_photo_bleach: float, run_suffix=None) -> str:
    """The run folder's descriptor. The labeling and, when bleaching is on, its probability are part
    of it, so the documented variants never write to the same folder; any other change of settings
    needs its own ``--run-suffix``, since an existing folder is refused."""
    descriptor = STAGE + ("_SINGLE_DYE" if labeling == "single" else f"_{labeling}_LAW")
    if prob_photo_bleach > 1e3 * BLEACH_OFF:
        descriptor += "_BLEACH_" + f"{prob_photo_bleach:g}".replace(".", "p").replace("-", "m").replace("+", "")
    if run_suffix:
        descriptor += f"_{run_suffix}"
    return descriptor


# ==========================================================================================
# One scene: render it and reproduce its truth
# ==========================================================================================

def render_scene(lambda_rate: float, mu_pc: float, seed: int, *, n_subunits: int, n_frames: int,
                 labeling: str, prob_photo_bleach: float):
    """Render one diffusing scene and reproduce its true per-dye photons from the same seed.

    `render_dli_video` draws the per-dye brightness with ``generate_brightness_photons(...,
    seed=seed)``; calling that function again with the same arguments and seed returns the very
    photons the video was rendered from, so the truth needs no instrumented renderer. Dye counts
    come from the bare law at probe occupancy 1 (module docstring).
    """
    stem = PARAMETERS.simulation.stem
    npx, px_nm = stem.root_size_px, stem.pixel_size_nm
    dt = PARAMETERS.simulation.timing.frame_time_seconds
    rng = np.random.default_rng(seed)
    step_nm = np.sqrt(2.0 * DIFFUSION_UM2_PER_S * 1e6 * dt)
    box = npx * px_nm
    pos = rng.uniform(0.1 * box, 0.9 * box, size=(n_subunits, 2))
    poses = np.zeros((n_frames, n_subunits, 3))
    for t in range(n_frames):
        poses[t, :, :2] = pos
        pos = np.clip(pos + rng.normal(0.0, step_nm, (n_subunits, 2)), 0.02 * box, 0.98 * box)
    host = np.tile(np.arange(n_subunits)[None, :], (n_frames, 1))
    if labeling == "single":
        dye_counts = np.ones(n_subunits, dtype=np.int64)
    else:
        law = lab.resolve_labeling_law(labeling)[1]
        dye_counts = lab.draw_dye_counts(law, n_subunits, np.random.default_rng(seed + 1000))

    img = _imaging_center()
    img.update(lambda_rate=float(lambda_rate), mu_pc=float(mu_pc),
               prob_photo_bleach=float(prob_photo_bleach))
    vec = np.array([img[k] for k in det.DETECTOR_IMAGING_KEYS])
    frames = dli.render_dli_video(poses, host, dye_counts, vec, seed=seed)
    video = sio.convert_video_dtype(np.moveaxis(frames, 2, 0), bits_from=16, bits_to=8)

    _, dye_subunit = dli.build_dye_tracks(poses, host, dye_counts)
    dye_photons = dli.generate_brightness_photons(
        nframes=n_frames, nemitters=int(dye_subunit.shape[0]), mu_pc=img["mu_pc"],
        sigma_pc=img["sigma_pc"], lambda_rate=img["lambda_rate"],
        prob_photo_bleach=img["prob_photo_bleach"],
        numb_photo_bleach=dli._fixed("numb_photo_bleach"), delta_frame=dt, seed=seed)
    spot_photons = np.zeros((n_frames, n_subunits))
    np.add.at(spot_photons.T, dye_subunit, dye_photons.T)     # a spot is the sum of its dyes
    return dict(video=video, dye_photons=dye_photons, spot_photons=spot_photons,
                true_xy=poses[:, :, :2] / px_nm, visible=dye_counts > 0, img=img)


def match_to_subunits(m: dict, true_xy: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Code every accepted spot fit by the visible subunit it belongs to.

    The code is the index of the nearest visible subunit of the fit's frame when it lies within
    ``MATCH_RADIUS_PX``; ``UNMATCHED`` (-1) when none does; ``SECOND_FIT`` (-2) when a nearer fit of
    the same frame already took that subunit. A subunit thus keeps at most one fit per frame, so an
    oracle trace never holds two values in one frame, and the negative codes are the detections
    outside the matched set.
    """
    frames = np.asarray(m["frame_index"])
    xs, ys = np.asarray(m["x"], dtype=float), np.asarray(m["y"], dtype=float)
    vis_idx = np.nonzero(visible)[0]
    matched = np.full(frames.size, UNMATCHED, dtype=np.int64)
    if vis_idx.size == 0:
        return matched
    for t in np.unique(frames):
        sel = np.nonzero(frames == t)[0]
        pos = true_xy[t][vis_idx]
        d = np.hypot(xs[sel, None] - pos[None, :, 0], ys[sel, None] - pos[None, :, 1])
        j = np.argmin(d, axis=1)
        dist = d[np.arange(sel.size), j]
        taken = set()
        for k in np.argsort(dist):
            if dist[k] > MATCH_RADIUS_PX:
                break
            cand = int(vis_idx[j[k]])
            if cand in taken:
                matched[sel[k]] = SECOND_FIT
            else:
                matched[sel[k]] = cand
                taken.add(cand)
    return matched


def level_traces(level: str, scene: dict, m: dict, tid: np.ndarray, matched: np.ndarray,
                 min_length: int) -> list:
    """The ``(frame_index, value)`` traces of one observation level.

    Every level that uses detections applies the estimator's own trace rule (at least three points
    and a span of at least ``min_length`` frames). ``fitted_oracle`` and ``fitted_linked`` read the
    same detections -- the matched ones -- and differ only in how they are grouped.
    """
    n_frames = scene["spot_photons"].shape[0]
    all_frames = np.arange(n_frames)
    if level == "dye_truth_full":
        return [(all_frames, scene["dye_photons"][:, d]) for d in range(scene["dye_photons"].shape[1])]
    if level == "spot_truth_full":
        return [(all_frames, scene["spot_photons"][:, s]) for s in np.nonzero(scene["visible"])[0]]
    if level == "production":
        traces, _ = die.spot_intensity_traces(m, tid, min_length=min_length)
        return traces
    frames = np.asarray(m["frame_index"])
    amps = np.asarray(m["amplitude"], dtype=float)
    inside = matched >= 0
    if level == "fitted_linked":
        kept = dict(frame_index=frames[inside], amplitude=amps[inside])
        traces, _ = die.spot_intensity_traces(kept, np.asarray(tid)[inside], min_length=min_length)
        return traces
    out = []
    for s in np.unique(matched[inside]):
        sel = matched == s
        f = frames[sel]
        order = np.argsort(f)
        f = f[order]
        if f.size < 3 or int(f[-1] - f[0] + 1) < int(min_length):
            continue
        if level == "spot_truth_span":
            span = np.arange(int(f[0]), int(f[-1]) + 1)
            out.append((span, scene["spot_photons"][span, s]))
        elif level == "spot_truth_detected":
            out.append((f, scene["spot_photons"][f, s]))
        elif level == "fitted_oracle":
            out.append((f, amps[sel][order]))
        else:
            raise ValueError(f"unknown level {level!r}")
    return out


def estimate_level(traces: list, *, n_model_traces: int) -> dict:
    """The production shape match on one level's traces, with the estimator's own refusals."""
    nan = float("nan")
    none = dict(lambda_rate=nan, n_traces=len(traces), span_median=nan, refined=False,
                at_grid_edge=False)
    if len(traces) < 5:
        return dict(none, reason="too_few_traces")
    shape, _ = die.flicker_data_shape(traces)
    if shape is None:
        return dict(none, reason="too_few_pairs")
    spans = np.asarray([int(f.max() - f.min() + 1) for f, _ in traces], dtype=np.int64)
    grid, shapes = die.flicker_model_shapes(
        spans, frame_time_seconds=PARAMETERS.simulation.timing.frame_time_seconds,
        n_traces=n_model_traces)
    r = die.match_shapes(shape, grid, shapes)
    return dict(lambda_rate=float(r["lambda_rate"]), n_traces=len(traces),
                span_median=float(np.median(spans)), refined=bool(r["refined"]),
                at_grid_edge=bool(r["at_grid_edge"]), reason=None)


def run_scene(job) -> dict:
    """Pool worker: one scene through every level. Top-level so it pickles."""
    (i, lam, mu_pc, rep, seed, n_subunits, n_frames, labeling, prob_bleach, n_sigma, half_px,
     min_length, n_model_traces) = job
    t0 = time.time()
    scene = render_scene(lam, mu_pc, seed, n_subunits=n_subunits, n_frames=n_frames,
                         labeling=labeling, prob_photo_bleach=prob_bleach)
    scope = _scope_center()
    m = die.measure_spot_widths(scene["video"], scope, frame_stride=1, n_sigma=n_sigma,
                                half_px=half_px)
    tid = (die.link_spot_tracks(m["frame_index"], m["x"], m["y"], frame_stride=1)
           if m["sqrt2sigma"].size else np.zeros(0, dtype=np.int64))
    matched = match_to_subunits(m, scene["true_xy"], scene["visible"])
    rows = [estimate_level(level_traces(level, scene, m, tid, matched, min_length),
                           n_model_traces=n_model_traces) for level in LEVELS]
    n_vis = int(scene["visible"].sum())
    found = np.unique(matched[matched >= 0]).size
    return dict(scene=i, lambda_rate=lam, mu_pc=mu_pc, replicate=rep, seed=seed,
                levels=rows, n_visible=n_vis, detected_share=found / max(n_vis, 1),
                spot_fits=int(m["sqrt2sigma"].size), n_matched=int((matched >= 0).sum()),
                n_unmatched=int((matched == UNMATCHED).sum()),
                n_second_fit=int((matched == SECOND_FIT).sum()), seconds=time.time() - t0)


# ==========================================================================================
# Collection and report
# ==========================================================================================

def collect_levels(results: list) -> dict:
    """Per scene and level: the estimate, its signed log10 error, and why a level produced none."""
    lam = np.array([r["lambda_rate"] for r in results], dtype=float)
    est = np.array([[x["lambda_rate"] for x in r["levels"]] for r in results], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.log10(est) - np.log10(lam)[:, None]
    return dict(
        lam=lam, mu=np.array([r["mu_pc"] for r in results], dtype=float), est=est, err=err,
        reason=np.array([[x["reason"] or "" for x in r["levels"]] for r in results], dtype="<U32"),
        at_grid_edge=np.array([[x["at_grid_edge"] for x in r["levels"]] for r in results], dtype=bool),
        refined=np.array([[x["refined"] for x in r["levels"]] for r in results], dtype=bool),
        n_traces=np.array([[x["n_traces"] for x in r["levels"]] for r in results], dtype=float),
        span=np.array([[x["span_median"] for x in r["levels"]] for r in results], dtype=float))


def mean_cell(err_column: np.ndarray, mask: np.ndarray) -> str:
    """Mean signed error of the scenes in ``mask`` that produced an estimate, with their count."""
    v = err_column[mask]
    v = v[np.isfinite(v)]
    return f"{v.mean():+.3f} ({v.size})" if v.size else f"n/a (0 of {int(mask.sum())})"


def step_rows(err: np.ndarray) -> list:
    """The contrast of every level with the one before it, paired within scenes: mean, median,
    and how many scenes produced an estimate at both levels."""
    rows = []
    for li in range(1, len(LEVELS)):
        d = err[:, li] - err[:, li - 1]
        d = d[np.isfinite(d)]
        rows.append([f"{LEVELS[li - 1]} -> {LEVELS[li]}", STEP_NOTES[LEVELS[li]],
                     f"{d.mean():+.3f}" if d.size else "n/a",
                     f"{np.median(d):+.3f}" if d.size else "n/a",
                     f"{d.size} of {err.shape[0]}"])
    return rows


def make_figure(lam, mu, err, common, reporter):
    """Signed error against the true rate, one line per level, one panel per brightness, over the
    scenes where every level produced an estimate (so the lines compare the same scenes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mus = np.unique(mu)
    fig, axes = plt.subplots(1, len(mus), figsize=(4.2 * len(mus), 3.8), sharey=True, squeeze=False)
    lams = np.unique(lam)
    for ax, mv in zip(axes[0], mus):
        for li, level in enumerate(LEVELS):
            y = []
            for lv in lams:
                v = err[(lam == lv) & (mu == mv) & common, li]
                y.append(float(v.mean()) if v.size else np.nan)
            ax.plot(lams, y, marker="o", ms=4, lw=1.4, label=level)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("true lambda_rate (1/s)")
        ax.set_title(f"mu_pc = {mv:.0f} photons per dye", fontsize=10)
    axes[0][0].set_ylabel("mean signed error (dex)")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    reporter.save_figure("flicker_mismatch_error_by_level", fig,
                         caption="Mean signed log10 error of the shape match at each observation "
                                 "level against the true rate, one panel per brightness, over the "
                                 "scenes where every level produced an estimate. A departure between "
                                 "adjacent levels suggests a contribution of that step under these "
                                 "scenes and this ordering; the report gives the counts.")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Where along the observation chain the direct flicker estimator's bias arises.")
    ap.add_argument("--lambda-grid", type=float, nargs="+", default=[1.25, 2.0, 4.0, 8.0],
                    help="true rates; kept inside the model grid [1, 14] so an estimate can fall "
                         "on either side of the truth.")
    ap.add_argument("--mu-pc-grid", type=float, nargs="+", default=[120.0, 240.0, 480.0],
                    help="photons per dye; 120 lies in the dim operating subgroup of sec. 9.6.")
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--n-subunits", type=int, default=200)
    ap.add_argument("--total-time-seconds", type=float, default=2.0)
    ap.add_argument("--labeling", default="FAB", choices=("single",) + tuple(lab.LABELING_CONDITIONS),
                    help="dye counts per subunit: one dye, or drawn from a condition's bare law "
                         "(probe occupancy 1).")
    ap.add_argument("--prob-photo-bleach", type=float, default=BLEACH_OFF,
                    help="bleaching probability per 100 frames; off by default.")
    ap.add_argument("--min-track-length", type=int, default=40)
    ap.add_argument("--n-sigma", type=float, default=4.0)
    ap.add_argument("--half-px", type=int, default=14)
    ap.add_argument("--model-traces", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-suffix", default=None,
                    help="token appended to the run folder name; letters, digits, underscores.")
    args = ap.parse_args(argv)
    if args.run_suffix is not None and not re.fullmatch(r"[A-Za-z0-9_]+", args.run_suffix):
        ap.error(f"--run-suffix {args.run_suffix!r}: use letters, digits and underscores only")

    timing = RunTiming(total_time_seconds=args.total_time_seconds)
    n_frames = timing.frame_count
    run_alias = f"{det.detector_paths(PARAMETERS.paths).project_alias}_{timing.label}"
    descriptor = run_descriptor(args.labeling, args.prob_photo_bleach, args.run_suffix)
    out_dir = args.out_dir or os.path.join(str(PARAMETERS.machine.data_bank_root), "Posit",
                                           f"{run_alias}_{descriptor}")
    if os.path.exists(out_dir):
        ap.error(f"the run folder exists already: {out_dir}. A run never reuses a folder: pass another "
                 f"--run-suffix, or move the earlier run aside deliberately.")
    jobs = []
    for lam in args.lambda_grid:
        for mu_pc in args.mu_pc_grid:
            for rep in range(args.replicates):
                i = len(jobs)
                jobs.append((i, float(lam), float(mu_pc), rep, SEED_BASE + i, args.n_subunits,
                             n_frames, args.labeling, args.prob_photo_bleach, args.n_sigma,
                             args.half_px, args.min_track_length, args.model_traces))

    if args.dry_run:
        print(f"[{STAGE}] DRY RUN -- nothing is rendered and nothing is written.")
        print(f"  machine profile : {os.environ.get('MACHINE_PROFILE', '(unset)')}")
        print(f"  out dir         : {out_dir}   (new; created at the start of the run)")
        print(f"  scenes          : {len(jobs)} = lambda {args.lambda_grid} x mu_pc "
              f"{args.mu_pc_grid} x {args.replicates} replicates")
        print(f"  each scene      : {args.n_subunits} subunits, {n_frames} frames, labeling "
              f"{args.labeling} (occupancy 1), prob_photo_bleach {args.prob_photo_bleach:g}")
        print(f"  levels          : {', '.join(LEVELS)}")
        print(f"  workers         : {args.workers}")
        return 0

    startup_code = prov.code_provenance(files=prov.DIRECT_ESTIMATOR_FILES)
    try:
        prov.reserve_run_folder(out_dir)
    except FileExistsError:
        ap.error(f"the run folder appeared after the check (a concurrent run?): {out_dir}")
    reporter = DiagnosticReporter(
        stage=STAGE, enabled=True, dump=True, dump_dir=out_dir, run_label=run_alias,
        run_note=("Development diagnostic on rendered scenes with known truth: where along the "
                  "observation chain the direct flicker estimator's bias arises. It reads no EVAL "
                  "task, tunes nothing and reaches no verdict."))
    reporter.checkpoint("scenes", n=len(jobs), frames=n_frames, subunits=args.n_subunits,
                        labeling=args.labeling, prob_photo_bleach=args.prob_photo_bleach)

    t0 = time.time()
    results = []
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, r in enumerate(pool.map(run_scene, jobs, chunksize=1), 1):
                results.append(r)
                print(f"  [scene {k}/{len(jobs)}] lambda {r['lambda_rate']:g} mu_pc {r['mu_pc']:g} "
                      f"rep {r['replicate']}: production {r['levels'][-1]['lambda_rate']:.3f}  "
                      f"({r['seconds']:.0f} s)", flush=True)
    else:
        for k, j in enumerate(jobs, 1):
            r = run_scene(j)
            results.append(r)
            print(f"  [scene {k}/{len(jobs)}] lambda {r['lambda_rate']:g} mu_pc {r['mu_pc']:g} "
                  f"rep {r['replicate']}: " + "  ".join(
                      f"{lv} {x['lambda_rate']:.3f}" for lv, x in zip(LEVELS, r["levels"])), flush=True)
    reporter.stat("wall time (min)", round((time.time() - t0) / 60.0, 1))

    c = collect_levels(results)
    lam, mu, est, err = c["lam"], c["mu"], c["est"], c["err"]
    common = np.isfinite(err).all(axis=1)          # scenes where every level produced an estimate
    n_common = int(common.sum())
    lams, mus = np.unique(lam), np.unique(mu)
    every = np.ones_like(lam, dtype=bool)
    reporter.table(
        "Mean signed error (dex) by level and true rate (scenes with an estimate in parentheses)",
        ["level"] + [f"lambda {v:g}" for v in lams] + ["all", f"every level ({n_common})"],
        [[lv] + [mean_cell(err[:, li], lam == v) for v in lams] + [mean_cell(err[:, li], every),
                                                                     mean_cell(err[:, li], common)]
         for li, lv in enumerate(LEVELS)],
        note="Each level changes one thing against the one above it. A cell averages the scenes where "
             "that level produced an estimate, so cells of different levels can average different "
             "scenes; the last column averages only the scenes where every level produced one. "
             + " ".join(f"{k}: {v}." for k, v in LEVEL_NOTES.items()))
    reporter.table(
        "Mean signed error (dex) by level and brightness (scenes with an estimate in parentheses)",
        ["level"] + [f"mu_pc {v:g}" for v in mus],
        [[lv] + [mean_cell(err[:, li], mu == v) for v in mus] for li, lv in enumerate(LEVELS)],
        note="mu_pc in photons per dye. The development recordings ran fast by +0.16 dex in their "
             "dim half and +0.06 dex in their bright half (DETECTOR_WORKFLOW.md sec. 9.6).")
    reporter.table(
        "Contrast between adjacent levels (dex), paired within scenes",
        ["step", "what changes", "mean", "median", "scenes paired"], step_rows(err),
        note="Paired within scenes, so the scene-to-scene scatter of the truth cancels; a scene enters a "
             "contrast only when both of its levels produced an estimate. A contrast suggests a "
             "contribution of that step under these scenes and this ordering: the steps are added in one "
             "fixed order, and an effect can differ in size, even in sign, when taken in another. It is "
             "diagnostic accounting, not a causal decomposition.")
    reporter.table(
        "Scenes without an estimate, by level and reason",
        ["level", "estimate produced"] + list(FAILURE_REASONS) + ["estimate at a grid edge"],
        [[lv, f"{int(np.isfinite(err[:, li]).sum())} of {len(results)}"]
         + [str(int((c["reason"][:, li] == rs).sum())) for rs in FAILURE_REASONS]
         + [str(int(c["at_grid_edge"][:, li].sum()))]
         for li, lv in enumerate(LEVELS)],
        note="too_few_traces: fewer than five traces met the trace rule; too_few_pairs: fewer pooled "
             "lag-one pairs than the estimator requires. The estimator refuses both cases the same way. "
             "A grid-edge estimate (lambda 1 or 14) is unrefined and may lie beyond the grid.")
    reporter.table(
        "Traces and spans per level (medians over scenes)",
        ["level", "traces", "median span (frames)"],
        [[lv, f"{np.nanmedian(c['n_traces'][:, li]):.0f}",
          f"{np.nanmedian(c['span'][:, li]):.0f}" if np.isfinite(c['span'][:, li]).any() else "n/a"]
         for li, lv in enumerate(LEVELS)])
    reporter.stat("median share of visible subunits detected",
                  float(np.median([r["detected_share"] for r in results])),
                  note="subunits with at least one accepted fit matched to them")
    reporter.stat("median detections outside the matched set per scene",
                  f"{np.median([r['n_unmatched'] for r in results]):.0f} with no visible subunit within "
                  f"{MATCH_RADIUS_PX} px; {np.median([r['n_second_fit'] for r in results]):.0f} second fits "
                  f"near a matched subunit",
                  note="what production reads beyond fitted_linked; the matched detections per scene "
                       f"have a median of {np.median([r['n_matched'] for r in results]):.0f}")
    reporter.stat("dye counts", f"{args.labeling} (bare law, probe occupancy 1)",
                  note="a tier also applies the condition's probe occupancy, which thins the visible "
                       "subunits without changing a visible one's dye count; here --n-subunits sets "
                       "the density of visible spots")
    make_figure(lam, mu, err, common, reporter)

    code = prov.finalize_code_provenance(startup_code, files=prov.DIRECT_ESTIMATOR_FILES)
    changed = bool(code["changed_during_run"])
    np.savez_compressed(os.path.join(out_dir, "direct_flicker_mismatch.npz"),
                        levels=np.asarray(LEVELS), lambda_true=lam, mu_pc=mu,
                        replicate=np.array([r["replicate"] for r in results]),
                        seed=np.array([r["seed"] for r in results]), estimate=est, error_dex=err,
                        reason=c["reason"], at_grid_edge=c["at_grid_edge"], refined=c["refined"],
                        n_traces=c["n_traces"], span_median=c["span"],
                        detected_share=np.array([r["detected_share"] for r in results]),
                        n_visible=np.array([r["n_visible"] for r in results]),
                        n_matched=np.array([r["n_matched"] for r in results]),
                        n_unmatched=np.array([r["n_unmatched"] for r in results]),
                        n_second_fit=np.array([r["n_second_fit"] for r in results]))
    per_level = {}
    for li, lv in enumerate(LEVELS):
        ok = np.isfinite(err[:, li])
        per_level[lv] = dict(
            scenes=len(results), estimates=int(ok.sum()),
            reasons={rs: int((c["reason"][:, li] == rs).sum()) for rs in FAILURE_REASONS},
            at_grid_edge=int(c["at_grid_edge"][:, li].sum()),
            mean_error_dex=float(err[ok, li].mean()) if ok.any() else None,
            mean_error_dex_every_level=float(err[common, li].mean()) if n_common else None)
    steps = {}
    for li in range(1, len(LEVELS)):
        d = err[:, li] - err[:, li - 1]
        d = d[np.isfinite(d)]
        steps[f"{LEVELS[li - 1]} -> {LEVELS[li]}"] = dict(
            changes=STEP_NOTES[LEVELS[li]], paired=int(d.size),
            mean_dex=float(d.mean()) if d.size else None,
            median_dex=float(np.median(d)) if d.size else None)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(dict(levels=list(LEVELS), level_notes=LEVEL_NOTES, per_level=per_level, steps=steps,
                       scenes=len(results), scenes_every_level=n_common,
                       labeling=f"{args.labeling} (bare law, probe occupancy 1)",
                       prob_photo_bleach=args.prob_photo_bleach,
                       implementation=("INVALID (implementation changed during the run)" if changed
                                       else "unchanged during the run")),
                  fh, indent=2)
    record = prov.analysis_run_record(
        startup_code, argv=sys.argv, code=code,
        extra=dict(stage=STAGE, purpose="diagnostic (rendered scenes; reads no EVAL task)",
                   scenes=len(jobs), lambda_grid=args.lambda_grid,
                   mu_pc_grid=args.mu_pc_grid, replicates=args.replicates,
                   n_subunits=args.n_subunits, frames=n_frames, labeling=args.labeling,
                   occupancy=1.0, prob_photo_bleach=args.prob_photo_bleach, seed_base=SEED_BASE,
                   run_suffix=args.run_suffix, out_dir=str(out_dir)))
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    reporter.check("implementation unchanged during the run", not changed,
                   "implementation hash identical at startup and at write", fatal=False,
                   note="A file edited while the run executed leaves results that describe no single "
                        "implementation: the run is INVALID, exits with status 3, and provenance.json "
                        "records both hashes. The arrays are kept as diagnostics.")
    reporter.summary()
    path = reporter.write_report()
    print(f"\n[{STAGE}] report -> {path}")
    return da.EXIT_IMPLEMENTATION_CHANGED if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
