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

    First, the log-brightness flicker is a stationary Ornstein-Uhlenbeck process with a
    correlation time of roughly fifteen frames, so consecutive frames of a fluorescence curve
    are not independent samples of the decay: a 100-frame recording carries about THREE
    effectively independent samples, not a hundred. Second, a longer recording sees more of
    the decay; in the short-window, shallow-decay limit with known amplitude and offset and
    independent constant-variance noise, rate information would grow approximately as the cube
    of the duration. Third, and decisively, the amplitude and the offset of the curve
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

    Under this reduced model the benchmark exceeds the 0.10 dex threshold by more than an order
    of magnitude everywhere at 2 s and falls inside it only in the upper half of the prior at
    20 s. It is an approximate benchmark (an effective-sample-size adjustment stands in for the
    correlated decay likelihood) and it constrains an unbiased estimator, so it is explanatory
    context, not an eligibility rule. Eligibility is OBSERVABLE (DETECTOR_WORKFLOW.md sec. 9.6):
    a converged fit whose fitted decay is visible above the residual scatter and whose own
    standard error on log10 p is small enough to report classes a recording as usable; the
    accuracy threshold, stated at 1000 frames, applies to usable recordings, and the recovery of
    rejected recordings is reported beside them so that selection cannot hide a failure.

Prespecified acceptance (fixed before the first run)

    prob_photo_bleach   mean absolute log10 error <= 0.10 dex at 1000 frames

    That is 6.7% of its 1.5 dex prior width. No threshold is set at 100 frames, because the
    information budget shows the bound there already exceeds it.

Usage (from the repo root):
    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Fluorescence_Loss.py \\
        --condition FAB --total-time-seconds 20.0 --tasks 0 1 --expect-videos-per-task 100 \\
        --max-videos 200 --purpose development --run-suffix <commit>

    ... --purpose verdict                   # required for a tier run: development, or verdict for the
                                            # reserved EVAL tasks of the tier in full (sec. 9.6)
    ... --selftest --selftest-frames 1000   # in-memory recordings at known bleaching rates
    ... --dry-run                           # resolve settings and apply every refusal; read nothing

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/<alias>_<CONDITION>_<timing>_Direct_Fluorescence_Loss_<DEV|VERDICT>[_<suffix>]/
        report.md, direct_fluorescence_loss.npz, summary.json, provenance.json, figures/
    A run never reuses a folder: an existing one is refused before anything is read.
"""

from __future__ import annotations

import argparse
import glob
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
from srm_and_sbi_monomer_dimer_alp import direct_imaging_estimates as die  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import direct_acceptance as da  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import provenance as prov  # noqa: E402
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
# Observable eligibility diagnostic (DETECTOR_WORKFLOW.md sec. 9.6, bleaching): a recording is
# USABLE when its fit converged, the fitted decay is visible above the residual scatter, and the
# fit's own standard error on log10 p is small enough to be worth reporting. It never reads the
# true value. Frozen before the full-length validation; calibrated only on the four self-test
# scenes (2026-09-21): a standard-error cap of 0.15 dex rejected the p = 0.1 scene that was
# recovered to 0.025 dex (its flicker-inflated standard error is 0.195 dex); 0.25 dex accepts the
# two scenes recovered within 0.03 dex and rejects the two with errors of 0.3 and 0.6 dex, whose
# standard errors are 5 and 6 dex. The decay-visibility floor of 3 alone would have accepted the
# p = 0.0316 scene (signal-to-noise 4.2, error -0.61 dex), so both criteria are needed.
ELIGIBILITY = {"min_decay_snr": 3.0, "max_se_log10": 0.25}
# Flicker rate used for the effective-sample-size correction of the fit's standard error when the
# caller supplies none: the prior center, a deployable value. The true per-recording rate is never
# used -- it is unavailable on an experimental recording.
LAMBDA_RATE_DEFAULT = 10 ** (0.5 * sum(det.DETECTOR_PARAMETERIZATION[det.DETECTOR_FIND["lambda_rate"]]["PRIOR_RANGE"]))
ACCEPTANCE = {"prob_bleach_mae_dex": 0.10, "acceptance_frames": 1000}
# Relative noise of the total-fluorescence curve per frame at the MET-FAB emitter density, taken
# from the kernel so this script and the budget utility cannot quote different values for the same
# quantity. Used only to state the information bound beside the result.
RELATIVE_NOISE = ib.FIELD_RELATIVE_NOISE_MET_FAB
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
    def _out(prob=np.nan, se=np.nan, n_ap=-1, n_eff=np.nan, snr=np.nan, se10=np.nan,
             outcome="failed", reason=None, low=np.nan, high=np.nan):
        return dict(prob_photo_bleach=prob, prob_se=se, n_apertures=n_ap, n_eff=n_eff,
                    decay_snr=snr, se_log10=se10, outcome=outcome, reason=reason,
                    low=low, high=high)

    if observable == "apertures":
        curve = die.spot_flux_curve(video_levels, scope, n_sigma=n_sigma,
                                    detect_frames=detect_frames)
        n_ap = curve["n_apertures"]
        if n_ap == 0:
            return _out(n_ap=0, reason="no_apertures")
    else:
        curve = die.frame_flux_curve(video_levels)
        n_ap = -1
    lam = LAMBDA_RATE_DEFAULT if lambda_rate is None else float(lambda_rate)
    fit = die.fit_fluorescence_loss(
        curve["flux"], lambda_rate=lam,
        frame_time_seconds=PARAMETERS.simulation.timing.frame_time_seconds)
    prob, se = fit["prob_photo_bleach"], fit["prob_se"]
    if not fit["success"]:
        return _out(prob, se, n_ap, fit["n_eff"], reason="fit_failed")
    if not (np.isfinite(prob) and prob > 0):
        return _out(prob, se, n_ap, fit["n_eff"], reason="nonpositive_estimate")
    # Three outcomes (sec. 9.6): a VALID estimate is classed usable or uninformative by two
    # observable diagnostics -- the visibility of the fitted decay above the residual scatter,
    # and the fit's own standard error on log10 p. Neither reads the true value.
    total_drop = fit["amplitude"] * (1.0 - np.exp(-fit["rate_per_frame"] * fit["n_frames"]))
    snr = float(total_drop / fit["resid_sd"]) if fit["resid_sd"] > 0 else float("inf")
    se10 = float(se / (prob * np.log(10.0))) if np.isfinite(se) else float("nan")
    usable = bool(snr >= ELIGIBILITY["min_decay_snr"] and np.isfinite(se10)
                  and se10 <= ELIGIBILITY["max_se_log10"])
    z = 1.6448536269514722
    if np.isfinite(se10):
        # In log10, with the exponent clipped: an uninformative fit can carry a standard error of
        # hundreds of dex, and 10 ** that overflows a float. Such a range is reported as-is (it
        # simply spans the whole prior and beyond); the recording is uninformative, not failed.
        log_c = float(np.log10(prob))
        low = 10.0 ** max(log_c - z * se10, -300.0)
        high = 10.0 ** min(log_c + z * se10, 300.0)
    else:
        low, high = np.nan, np.nan
    return _out(prob, se, n_ap, fit["n_eff"], snr, se10,
                outcome="usable" if usable else "uninformative", reason=None, low=low, high=high)


_STORE_CACHE: dict = {}


def _worker(job):
    video_path, task, index, scope, lam, opts = job
    handle = _STORE_CACHE.get(video_path)
    if handle is None:
        handle = sio.load_data(video_path)
        _STORE_CACHE[video_path] = handle
    out = estimate_one(np.asarray(handle[index]), scope, lambda_rate=lam, **opts)
    out["task"] = int(task)
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

    truths, ests, outs, thetas = [], [], [], []
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
        thetas.append([img[k] for k in det.DETECTOR_FIND])
        vec = np.array([img[k] for k in det.DETECTOR_IMAGING_KEYS])
        frames = render_dli_video(poses, host, np.ones(n_subunits, dtype=np.int64), vec,
                                  seed=SELFTEST_SEED + i)
        video = sio.convert_video_dtype(np.moveaxis(frames, 2, 0), bits_from=16, bits_to=8)
        r = estimate_one(video, scope, lambda_rate=None,          # never the scene's true rate
                         n_sigma=n_sigma, detect_frames=detect_frames,
                         observable=observable)
        truths.append(p)
        ests.append(r["prob_photo_bleach"])
        outs.append(r)
        err = (np.log10(max(r["prob_photo_bleach"], 1e-9)) - np.log10(p)
               if np.isfinite(r["prob_photo_bleach"]) else np.nan)
        print(f"  [{i + 1}/{len(grid)}] p {p:.4f} -> {r['prob_photo_bleach']:.4f}   "
              f"err {err:+.4f} dex   ({r['n_apertures']} apertures, "
              f"n_eff {r['n_eff']:.1f})", flush=True)
    return dict(truth=np.asarray(truths, float), estimate=np.asarray(ests, float),
                out=outs, theta=np.asarray(thetas, float))


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
    ap.add_argument("--lambda-rate", type=float, default=None,
                    help="flicker rate for the standard-error correction (default: the prior "
                         "center); a measured value from the direct flicker estimator may be "
                         "supplied. The true per-recording rate is never used.")
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
    if not args.selftest and args.purpose is None:
        ap.error("--purpose is required for a tier run: development or verdict (DETECTOR_WORKFLOW.md sec. 9.6)")
    if args.selftest and args.purpose is not None:
        ap.error("--purpose applies to a tier run; a selftest reads no EVAL task")

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

    # ---- purpose: held to the declared split before any recording is read ------------------
    purpose_record = dict(purpose="selftest")
    if not args.selftest:
        try:
            purpose_record = da.check_run_purpose(
                args.purpose, estimator=STAGE, condition=args.condition, n_frames=n_frames,
                split=args.split, tasks=args.tasks, max_videos=args.max_videos,
                expect_videos_per_task=args.expect_videos_per_task)
        except ValueError as exc:
            ap.error(str(exc))

    descriptor = (f"{STAGE}_SELFTEST" if args.selftest
                  else f"{STAGE}_{da.PURPOSE_TOKENS[args.purpose]}")
    if args.run_suffix:
        descriptor += f"_{args.run_suffix}"
    posit_dir = os.path.join(str(data_bank_root), "Posit")
    out_dir = args.out_dir or os.path.join(posit_dir, f"{run_alias}_{descriptor}")
    if not args.selftest and args.purpose == "verdict":
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

    if args.dry_run:
        print(f"[{STAGE}] DRY RUN -- nothing is read and nothing is written.")
        print(f"  machine profile : {os.environ.get('MACHINE_PROFILE', '(unset)')}")
        print(f"  mode            : {'selftest' if args.selftest else args.split}")
        print(f"  run alias       : {run_alias}")
        print(f"  out dir         : {out_dir}   (new; created at the start of the run)")
        if not args.selftest:
            pr = purpose_record
            print(f"  purpose         : {pr['purpose']} -- tasks {pr['tasks']}; "
                  + (f"declared split: development {pr['development_tasks'][0]}-{pr['development_tasks'][-1]}, "
                     f"reserved {pr['reserved_tasks'][0]}-{pr['reserved_tasks'][-1]}"
                     if pr["split_declared"] else "no declared split for this tier (no reserved task)"))
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

    startup_code = prov.code_provenance(files=prov.DIRECT_ESTIMATOR_FILES)
    try:
        prov.reserve_run_folder(out_dir)
    except FileExistsError:
        ap.error(f"the run folder appeared after the check (a concurrent run?): {out_dir}")
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
        truth, estimate, out, theta_all = res["truth"], res["estimate"], res["out"], res["theta"]
        ids = dict(scene=np.arange(np.asarray(truth).shape[0]))
    else:
        reporter.stat("run purpose", args.purpose,
                      note=(f"tasks {purpose_record['tasks']}; " + (
                          f"declared split: development {purpose_record['development_tasks'][0]}-"
                          f"{purpose_record['development_tasks'][-1]}, reserved "
                          f"{purpose_record['reserved_tasks'][0]}-{purpose_record['reserved_tasks'][-1]} "
                          f"(DETECTOR_WORKFLOW.md sec. 9.6)" if purpose_record["split_declared"]
                          else "no declared split for this tier, so no reserved task")))
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
                jobs.append((str(vpath), t, i, scope_row, args.lambda_rate, opts))
                truth_rows.append(theta[i, :len(det.DETECTOR_FIND)])
            if len(jobs) >= args.max_videos:
                break
        reporter.stat("videos queued", len(jobs))
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
        truth = theta_all[:, det.DETECTOR_FIND["prob_photo_bleach"]]
        estimate = np.asarray([o["prob_photo_bleach"] for o in out], dtype=float)
        ids = dict(task=np.array([o["task"] for o in out], dtype=np.int64),
                   index=np.array([o["index"] for o in out], dtype=np.int64))

    extra = dict(n_apertures=np.array([o["n_apertures"] for o in out]),
                 n_eff=np.array([o["n_eff"] for o in out]),
                 decay_snr=np.array([o["decay_snr"] for o in out], dtype=float),
                 se_log10=np.array([o["se_log10"] for o in out], dtype=float),
                 range_low=np.array([o["low"] for o in out], dtype=float),
                 range_high=np.array([o["high"] for o in out], dtype=float))
    reasons = [o["reason"] for o in out]
    outcome = np.array([o["outcome"] for o in out])
    valid = np.isfinite(estimate) & (estimate > 0) & np.isfinite(truth) & (truth > 0) & (outcome != "failed")
    usable = valid & (outcome == "usable")

    # ---- the frozen rules of DETECTOR_WORKFLOW.md sec. 9.6, evaluated by the shared kernel ----
    def accuracy(mask):
        lt, le = np.log10(truth[mask]), np.log10(estimate[mask])
        err = le - lt
        corr = (float(np.corrcoef(lt, le)[0, 1])
                if lt.size > 2 and np.std(lt) > 0 and np.std(le) > 0 else None)
        mae = float(np.abs(err).mean())
        return {
            "prob_photo_bleach MAE (dex)": (mae, f"<= {ACCEPTANCE['prob_bleach_mae_dex']}",
                                            mae <= ACCEPTANCE["prob_bleach_mae_dex"]),
            "prob_photo_bleach bias (dex, reported)": (float(err.mean()), "--", True),
            "prob_photo_bleach corr (log10, reported)": (corr if corr is not None else float("nan"), "--", True),
        }

    at_length = int(n_frames) >= ACCEPTANCE["acceptance_frames"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (truth >= extra["range_low"]) & (truth <= extra["range_high"])
        width = np.log10(extra["range_high"]) - np.log10(extra["range_low"])
    lo_p, hi_p = da.prior_range("prob_photo_bleach")
    result = da.evaluate(theta_all, valid, reasons, accuracy, target_key="prob_photo_bleach",
                         ranges={"prob_photo_bleach (log10)": (cov, width, hi_p - lo_p)},
                         usable=usable, selftest=bool(args.selftest) or not at_length)
    if not at_length and not args.selftest:
        for k in list(result["verdicts"]):
            result["verdicts"][k] = (f"NOT APPLICABLE ({n_frames} frames is below the "
                                     f"{ACCEPTANCE['acceptance_frames']}-frame length the threshold is stated at)")
        result["exit_code"] = 0
    # The implementation hash is compared once every estimate exists and before anything is
    # written: a run whose code changed while it executed is invalid for acceptance (exit 3).
    code = prov.finalize_code_provenance(startup_code, files=prov.DIRECT_ESTIMATOR_FILES)
    da.apply_code_provenance(result, code)
    da.render(reporter, result, estimator=STAGE, target_key="prob_photo_bleach")
    reporter.table(
        "Bleaching outcomes against all attempted recordings", ["outcome", "count", "share of attempted", "rule"],
        [["failed measurement", str(int((outcome == "failed").sum())),
          f"{100 * (outcome == 'failed').mean():.1f} %", "counted against operational success"],
         ["valid but uninformative", str(int((outcome == "uninformative").sum())),
          f"{100 * (outcome == 'uninformative').mean():.1f} %", "reported; recovery shown beside usable"],
         ["usable", str(int((outcome == "usable").sum())),
          f"{100 * (outcome == 'usable').mean():.1f} %",
          f">= {da.RULES['bleach_usable_min']:.0f} overall, {da.RULES['bleach_usable_operating_min']:.0f} operating"]],
        note=f"Eligibility is observable and never reads the true value: a converged fit, a fitted total decay at "
             f"least {ELIGIBILITY['min_decay_snr']:.0f}x the residual scatter, and a fit standard error on log10 p "
             f"of at most {ELIGIBILITY['max_se_log10']} dex. The flicker correction uses the supplied rate "
             f"({LAMBDA_RATE_DEFAULT:.2f} at the prior center unless --lambda-rate is given).")
    lam_ctx = LAMBDA_RATE_DEFAULT if args.lambda_rate is None else args.lambda_rate
    p_ctx = 10 ** (0.5 * (lo_p + hi_p))
    bound_ctx = ib.crb_prob_bleach_dex(float(p_ctx), int(n_frames), RELATIVE_NOISE, lambda_rate=lam_ctx,
                                       frame_time_seconds=PARAMETERS.simulation.timing.frame_time_seconds)["sd_dex"]
    reporter.stat("information-budget benchmark at the prior center (dex)", float(bound_ctx),
                  note="explanatory context only, not an eligibility rule: the approximate standard deviation an "
                       "unbiased total-fluorescence estimator is not expected to beat at this recording length "
                       "(DETECTOR_WORKFLOW.md sec. 9.5)")

    np.savez_compressed(os.path.join(out_dir, "direct_fluorescence_loss.npz"),
                        truth=truth, estimate=estimate, theta=theta_all, valid=valid, usable=usable,
                        outcome=outcome, reasons=np.asarray([r or "" for r in reasons]), **extra, **ids)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(dict(acceptance=ACCEPTANCE, eligibility=ELIGIBILITY, purpose=purpose_record,
                       evaluation=result), fh, indent=2, default=float)
    record = prov.analysis_run_record(
        startup_code, argv=sys.argv, code=code,
        extra=dict(stage=STAGE, mode="selftest" if args.selftest else args.split,
                   purpose=purpose_record, tasks=None if args.selftest else list(args.tasks),
                   recordings=int(np.asarray(truth).shape[0]), run_suffix=args.run_suffix,
                   out_dir=str(out_dir), observable=args.observable, lambda_rate=args.lambda_rate))
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
    print(f"[{STAGE}] verdicts: " + "; ".join(f"{k}: {v}" for k, v in result["verdicts"].items()))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
