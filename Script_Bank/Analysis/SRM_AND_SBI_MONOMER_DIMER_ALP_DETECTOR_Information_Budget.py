"""Information budget: approximate precision benchmarks under a simplified observation model.

An estimator that misses a parameter can be failing for two very different reasons: it may
be a poor estimator, or the recordings may not carry the information. Those call for
opposite responses -- improve the estimator, or stop trying to infer the parameter -- and a
measured error alone cannot tell them apart. This utility computes an approximate benchmark
for the second question: a Cramer-Rao bound on the standard deviation of an unbiased estimator
of each imaging parameter under a simplified observation model, from one recording, placed
beside the two measured numbers.

    bound     an approximate benchmark under the reduced model (computed here)
    direct    what the direct estimators achieve      (their reports)
    neural    the posterior width of the amortized flow (the calibration report)

Reading the three together grades each parameter:

    neural near the bound       little to gain from a different estimator under the reduced
                                model; not a proof that the data are exhausted
    neural far from the bound   headroom worth looking for; the estimator or its training is
                                the first place to look
    bound wider than the prior  the reduced model expects one recording to constrain the
                                parameter poorly: a candidate to leave the inferred block,
                                not a proof of non-identifiability (the bound is approximate,
                                treats a reduced observable, and constrains an unbiased
                                estimator's standard deviation)

This is the quantitative criterion behind `DETECTOR_WORKFLOW.md` sec. 9.4, which proposes a
reduced inferred block. That section is a proposal and is not in force; this utility supplies
evidence for it and changes no stage.

What the bounds do and do not include. They use the exact EMCCD variance law, the exact pixel
integration, and the temporal correlation of the brightness flicker -- which is the largest
single term in the photobleaching budget and the one most easily missed. They do NOT include
emitters entering and leaving the field, reactions changing a spot's dye multiplicity,
overlapping spots, or the truncation of the observable population by detectability. Every
bound here is therefore OPTIMISTIC for its reduced statistic, never a prediction of achievable
error and never a limit on inference from the full video. A measured scatter well below one of
them is a prompt to check the measurement and the benchmark, not an automatic failure: the
bound concerns an unbiased estimator's standard deviation, and a biased or Bayesian estimator
can legitimately do better.

Usage (from the repo root):
    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Information_Budget.py \\
        --total-time-seconds 2.0 --n-emitters 154

    ... --measured measured.json   # overlay direct/neural numbers into the comparison table
    ... --dry-run                  # resolve settings and print what it would compute

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/<alias>_<timing>_Information_Budget/
        report.md, information_budget.npz, summary.json, figures/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import information_budget as ib  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import labeling as lab  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import (  # noqa: E402
    PARAMETERS, RunTiming,
)

assert os.path.abspath(ib.__file__).startswith(REPO_ROOT), (
    "Imported srm_and_sbi_monomer_dimer_alp from outside this repo checkout: "
    + ib.__file__ + " -- run with PYTHONPATH set to the repo root."
)

STAGE = "Information_Budget"

# Operating point of a MET-FAB recording, from the labeling audit and the dye-multiplicity
# stratification: about 1233 subunits per recording, a visible fraction of 0.125, and a mean
# of 2.03 dyes on a labeled subunit under FAB_POISSON.
DEFAULT_N_EMITTERS = 154
DEFAULT_DYES_PER_SPOT = 2.03
# Relative noise of the whole-field fluorescence curve, from the kernel so the budget
# and the estimator cannot quote different values for the same quantity.
DEFAULT_RELATIVE_NOISE = ib.FIELD_RELATIVE_NOISE_MET_FAB


def prior_box() -> dict:
    """Prior box of the six learnable imaging parameters, in log10 and physical units."""
    out = {}
    for e in det.DETECTOR_PARAMETERIZATION:
        lo, hi = e["PRIOR_RANGE"]
        out[e["KEY"]] = dict(log_low=lo, log_high=hi, width_dex=hi - lo,
                             low=10 ** lo, high=10 ** hi, center=10 ** (0.5 * (lo + hi)))
    return out


def scope_center() -> dict:
    return {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
            for e in det.DETECTOR_NUISANCE_SCOPE}


def compute_budget(n_frames: int, n_emitters: int, dyes_per_spot: float,
                   relative_noise: float, frame_time_seconds: float) -> dict:
    """Compute the bound for every learnable imaging parameter at the prior center."""
    box = prior_box()
    scope = scope_center()
    cam = dict(gamma=scope["gamma"], kappa_q=scope["kappa_q"], kappa_o=scope["kappa_o"],
               kappa_b=scope["kappa_b"], kappa_s=scope["kappa_s"])

    mu_r = box["mu_r"]["center"]
    sigma_px = mu_r / np.sqrt(2.0)
    mu_pc = box["mu_pc"]["center"]
    spot_photons = mu_pc * dyes_per_spot
    lam = box["lambda_rate"]["center"]

    # --- per-spot width precision, and its per-track value after averaging -------------
    per_spot = ib.crb_log_width_one_spot(spot_photons, sigma_px, **cam)
    track_len = max(int(n_frames) // 2, 1)          # a track is rarely the full recording
    v_track = (per_spot ** 2) / track_len

    pop_center = ib.crb_population_lognormal(n_emitters, box["sigma_r"]["center"], v_track)
    pop_low = ib.crb_population_lognormal(n_emitters, box["sigma_r"]["low"], v_track)
    pop_high = ib.crb_population_lognormal(n_emitters, box["sigma_r"]["high"], v_track)

    # --- brightness population: NOT the same structure as the width ----------------------
    # The PSF width is drawn once per subunit and held for the whole recording, so a track is
    # repeat measurement of one number and its variance divides by the track length. Brightness
    # is not: it is a stationary Ornstein-Uhlenbeck process, so every frame is a FRESH draw
    # from the very population whose spread sigma_pc describes. Averaging a track therefore
    # estimates that track's time-average, not a constant, and dividing the per-spot variance
    # by the track length -- as the width bound legitimately does -- would understate this
    # bound. Verified against the renderer: the within-track variance of ln brightness is 0.145
    # at sigma_pc = 0.42, against a population variance of 0.176, so most of the spread lives
    # WITHIN a track rather than between tracks.
    #
    # The units are therefore spot-FRAMES, not tracks, with the measurement variance left at
    # its per-frame value; and because consecutive frames of one dye are OU-correlated, the
    # count is discounted by the same effective-sample-size factor.
    fisher = ib.spot_fisher_information(spot_photons, sigma_px, **cam)
    cov = np.linalg.inv(fisher)
    per_spot_logA = float(np.sqrt(cov[0, 0]) / spot_photons)
    rho_dye = float(np.exp(-lam * float(frame_time_seconds)))
    eff_frames_per_track = ib.effective_sample_size_ar1(track_len, rho_dye)
    n_bright_units = max(int(round(n_emitters * eff_frames_per_track)), 2)
    v_bright = per_spot_logA ** 2
    bright_center = ib.crb_population_lognormal(n_bright_units, box["sigma_pc"]["center"],
                                                v_bright)

    # --- photobleaching, with and without the flicker correlation ----------------------
    bleach = {}
    for name, p in (("low", box["prob_photo_bleach"]["low"]),
                    ("center", box["prob_photo_bleach"]["center"]),
                    ("high", box["prob_photo_bleach"]["high"])):
        bleach[name] = dict(
            independent=ib.crb_prob_bleach_dex(p, n_frames, relative_noise),
            correlated=ib.crb_prob_bleach_dex(p, n_frames, relative_noise, lambda_rate=lam,
                                              frame_time_seconds=frame_time_seconds),
            value=p)

    # --- flicker rate -------------------------------------------------------------------
    flicker = {name: ib.crb_lambda_rate_dex(box["lambda_rate"][key], n_emitters, track_len,
                                            frame_time_seconds=frame_time_seconds)
               for name, key in (("low", "low"), ("center", "center"), ("high", "high"))}

    return dict(box=box, scope=scope, per_spot_log_width=per_spot,
                per_spot_log_amplitude=per_spot_logA, track_length=track_len,
                v_track=v_track, v_bright=v_bright,
                width_population=dict(center=pop_center, low=pop_low, high=pop_high),
                brightness_population=bright_center, bleach=bleach, flicker=flicker,
                n_bright_units=n_bright_units, eff_frames_per_track=eff_frames_per_track,
                spot_photons=spot_photons, n_emitters=n_emitters, n_frames=n_frames)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Cramer-Rao information budget for the six learnable imaging parameters.")
    ap.add_argument("--condition", default="FAB", choices=list(lab.LABELING_CONDITIONS),
                    help="labeling condition the operating point refers to. It names the "
                         "output, because the emitter density and the dyes per spot differ "
                         "between conditions and two runs would otherwise overwrite one another.")
    ap.add_argument("--total-time-seconds", type=float, default=2.0,
                    help="recording length the budget is computed for.")
    ap.add_argument("--compare-seconds", type=float, nargs="*", default=[2.0, 20.0],
                    help="additional recording lengths to tabulate, for the duration scaling.")
    ap.add_argument("--n-emitters", type=int, default=DEFAULT_N_EMITTERS,
                    help="visible emitters (spots) per recording.")
    ap.add_argument("--dyes-per-spot", type=float, default=DEFAULT_DYES_PER_SPOT,
                    help="mean dyes on a labeled subunit under the condition's labeling law.")
    ap.add_argument("--relative-noise", type=float, default=DEFAULT_RELATIVE_NOISE,
                    help="relative noise of the total-fluorescence curve, per frame.")
    ap.add_argument("--measured", default=None,
                    help="optional JSON with measured direct/neural spreads to overlay.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    data_bank_root = PARAMETERS.machine.data_bank_root
    dt = PARAMETERS.simulation.timing.frame_time_seconds
    timing = RunTiming(total_time_seconds=args.total_time_seconds)
    alias = det.detector_paths(PARAMETERS.paths).with_condition(args.condition).project_alias
    run_alias = f"{alias}_{timing.label}"
    out_dir = args.out_dir or os.path.join(str(data_bank_root), "Posit",
                                           f"{run_alias}_{STAGE}")

    if args.dry_run:
        print(f"[{STAGE}] DRY RUN -- nothing is read and nothing is written.")
        print(f"  machine profile : {os.environ.get('MACHINE_PROFILE', '(unset)')}")
        print(f"  condition       : {args.condition}")
        print(f"  out dir         : {out_dir}")
        print(f"  frames          : {timing.frame_count} ({args.total_time_seconds} s at "
              f"{int(round(1 / dt))} fps)")
        print(f"  emitters        : {args.n_emitters}   dyes/spot {args.dyes_per_spot}")
        print(f"  relative noise  : {args.relative_noise}")
        print(f"  compare lengths : {args.compare_seconds} s")
        print(f"  measured overlay: {args.measured or '(none)'}")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    reporter = DiagnosticReporter(
        stage=STAGE, enabled=True, dump=True, dump_dir=out_dir, run_label=run_alias,
        run_note=("Cramer-Rao lower bounds on the imaging parameters. These are OPTIMISTIC: "
                  "they omit emitter turnover, dye multiplicity changes, spot overlap and "
                  "detection truncation, all of which a real estimator faces."))

    budget = compute_budget(timing.frame_count, args.n_emitters, args.dyes_per_spot,
                            args.relative_noise, dt)
    box = budget["box"]

    reporter.stat("frames", timing.frame_count, note="frames in the recording the budget assumes")
    reporter.stat("visible emitters", args.n_emitters, note="spots available to any estimator")
    reporter.stat("photons per spot per frame", budget["spot_photons"],
                  note="median brightness times the mean dyes on a labeled subunit")
    reporter.stat("per-spot sd(ln sigma), one frame", budget["per_spot_log_width"],
                  note="Cramer-Rao bound on one spot's width from one frame")
    reporter.stat("assumed track length", budget["track_length"],
                  note="frames a spot is followed for; averaging divides its variance by this")

    # ---- the headline table ---------------------------------------------------------------
    w = budget["width_population"]
    rows = [
        ["mu_r", f"{box['mu_r']['width_dex']:.2f}",
         f"{w['center']['sd_log_scale_dex']:.4f}",
         f"{w['center']['sd_log_scale_dex'] / box['mu_r']['width_dex'] * 100:.1f}%"],
        ["sigma_r (at prior center)", f"{box['sigma_r']['width_dex']:.2f}",
         f"{w['center']['sd_shape']:.4f} (linear)",
         "-"],
        ["sigma_pc (at prior center)", f"{box['sigma_pc']['width_dex']:.2f}",
         f"{budget['brightness_population']['sd_shape']:.4f} (linear)", "-"],
        ["prob_photo_bleach (center)", f"{box['prob_photo_bleach']['width_dex']:.2f}",
         f"{budget['bleach']['center']['correlated']['sd_dex']:.4f}",
         f"{budget['bleach']['center']['correlated']['sd_dex'] / box['prob_photo_bleach']['width_dex'] * 100:.1f}%"],
        ["lambda_rate (center)", f"{box['lambda_rate']['width_dex']:.2f}",
         f"{budget['flicker']['center']['sd_dex']:.4f}",
         f"{budget['flicker']['center']['sd_dex'] / box['lambda_rate']['width_dex'] * 100:.1f}%"],
    ]
    reporter.table("Information budget at the prior center",
                   ["parameter", "prior width (dex)", "bound (sd)", "bound / prior"], rows,
                   note="A bound approaching the prior width means one recording carries almost "
                        "no information about that parameter.")

    # ---- sigma_r: where the measurement noise starts to matter -----------------------------
    # The bound is (sigma_r^2 + v) / (sigma_r sqrt(2(n-1))). For sigma_r well above sqrt(v) it
    # reduces to sigma_r / sqrt(2(n-1)) -- a CONSTANT relative precision, no degradation. Only
    # below the crossover sigma_r ~ sqrt(v) does the noise term take over and the bound grow.
    # Tabulating the linked and unlinked cases side by side locates that crossover against the
    # prior, which is the decision-relevant fact: it is what linking buys.
    v_unlinked = budget["per_spot_log_width"] ** 2
    sig_rows = []
    for key in ("low", "center", "high"):
        val = box["sigma_r"][key]
        linked = ib.crb_population_lognormal(args.n_emitters, val, budget["v_track"])
        unlinked = ib.crb_population_lognormal(args.n_emitters, val, v_unlinked)
        sig_rows.append([f"{val:.3f}", f"{linked['sd_shape']:.4f}",
                         f"{linked['sd_shape'] / val:.3f}x",
                         f"{unlinked['sd_shape']:.4f}",
                         f"{unlinked['sd_shape'] / val:.3f}x"])
    reporter.table(
        "sigma_r bound, linked against unlinked",
        ["sigma_r", "bound (linked)", "relative", "bound (unlinked)", "relative"], sig_rows,
        note=f"The bound is (sigma_r^2 + v)/(sigma_r sqrt(2(n-1))), so it degrades only once the "
             f"per-unit measurement variance v approaches sigma_r^2 -- below the crossover at "
             f"sigma_r ~ sqrt(v). Linking puts sqrt(v) at {np.sqrt(budget['v_track']):.3f}, well "
             f"under the prior floor of {box['sigma_r']['low']:.2f}, so the relative precision is "
             f"constant across the whole prior and no part of it is information-starved. Without "
             f"linking sqrt(v) is {np.sqrt(v_unlinked):.3f}, comparable to that floor, and the "
             f"bottom of the prior does sit in the degraded regime. That, and not the fitting, is "
             f"what averaging within a track buys.")

    # ---- the duration scaling, and the flicker-correlation term ----------------------------
    dur_rows = []
    for secs in args.compare_seconds:
        nf = int(secs / dt)
        for pname in ("low", "center", "high"):
            p = box["prob_photo_bleach"][{"low": "low", "center": "center", "high": "high"}[pname]]
            ind = ib.crb_prob_bleach_dex(p, nf, args.relative_noise)
            cor = ib.crb_prob_bleach_dex(p, nf, args.relative_noise,
                                         lambda_rate=box["lambda_rate"]["center"],
                                         frame_time_seconds=dt)
            dur_rows.append([f"{secs:g}", f"{nf}", f"{p:.4f}", f"{cor['n_eff']:.1f}",
                             f"{ind['sd_dex']:.4f}", f"{cor['sd_dex']:.4f}"])
    reporter.table(
        "prob_photo_bleach: duration and the flicker-correlation cost",
        ["seconds", "frames", "p", "effective frames", "bound if independent (dex)",
         "bound with flicker (dex)"], dur_rows,
        note="With the amplitude and offset profiled out, the bound falls with duration faster "
             "than the cubic law (which holds approximately in the short-window, shallow-decay limit "
             "with known amplitude and offset and independent constant-variance noise) where the "
             "decay is shallow, and more "
             "slowly where it completes within the window; the table, not a scaling law, carries "
             "the duration dependence. The last two columns separate that from the flicker: the "
             "log-brightness is an Ornstein-Uhlenbeck process with a correlation time of roughly "
             "fifteen frames, so consecutive frames of a fluorescence curve are not independent "
             "samples of the decay; the effective-frames column approximates that cost (a "
             "mean-estimation result applied to a fitted rate). Under this reduced model the "
             "parameter is unmeasurable in a two-second clip at the bottom of its prior.")

    # ---- optional overlay of measured numbers ----------------------------------------------
    measured = {}
    if args.measured:
        with open(args.measured) as fh:
            measured = json.load(fh)
        cmp_rows = []
        for key, entry in measured.items():
            bound = entry.get("bound")
            for source in ("direct", "neural"):
                val = entry.get(source)
                if val is None or bound in (None, 0):
                    continue
                cmp_rows.append([key, source, f"{bound:.4f}", f"{val:.4f}",
                                 f"{val / bound:.1f}x"])
        if cmp_rows:
            reporter.table("Measured spread against the bound",
                           ["parameter", "source", "bound", "measured", "ratio"], cmp_rows,
                           note="A ratio near one suggests little to gain from a different "
                                "estimator under the reduced model; a large ratio suggests "
                                "headroom worth looking for. The benchmark constrains an "
                                "unbiased estimator's scatter under a reduced statistic, so "
                                "neither reading is a proof, and a ratio below one is a prompt "
                                "to check the measurement and the benchmark.")

    # ---- invariants the algebra must satisfy ----------------------------------------------
    # A bound is easy to get subtly wrong and hard to notice, because a wrong number still
    # looks like a number. These check properties that follow from the structure rather than
    # from any particular value, so they fail loudly if a derivative, an inversion or a
    # scaling is mistaken.
    reporter.check(
        "bounds are finite at the prior center",
        all(np.isfinite(v) for v in [w["center"]["sd_log_scale_dex"], w["center"]["sd_shape"],
                                     budget["bleach"]["center"]["correlated"]["sd_dex"],
                                     budget["flicker"]["center"]["sd_dex"]]),
        "every parameter has a computable bound at the center of its prior", fatal=False,
        note="An infinite bound would mean the parameter does not enter the likelihood at all.")

    more = ib.crb_population_lognormal(4 * args.n_emitters, box["sigma_r"]["center"],
                                       budget["v_track"])
    reporter.check(
        "more emitters tighten the population bound",
        more["sd_log_scale_dex"] < w["center"]["sd_log_scale_dex"],
        f"{more['sd_log_scale_dex']:.5f} at 4x emitters vs "
        f"{w['center']['sd_log_scale_dex']:.5f}", fatal=False,
        note="More independent units must never widen a bound; a violation would mean the "
             "sample size enters the information with the wrong sign or power.")

    p_mid = box["prob_photo_bleach"]["center"]
    short = ib.crb_prob_bleach_dex(p_mid, 100, DEFAULT_RELATIVE_NOISE)["sd_dex"]
    long_ = ib.crb_prob_bleach_dex(p_mid, 1000, DEFAULT_RELATIVE_NOISE)["sd_dex"]
    reporter.check(
        "a longer recording tightens the decay bound", long_ < short,
        f"{long_:.5f} dex at 1000 frames vs {short:.5f} dex at 100", fatal=False,
        note="Decay-rate information grows with duration, so the bound must shrink. A "
             "violation would mean the time axis enters the Fisher sum incorrectly.")

    correlated = ib.crb_prob_bleach_dex(p_mid, 1000, DEFAULT_RELATIVE_NOISE,
                                        lambda_rate=box["lambda_rate"]["center"],
                                        frame_time_seconds=dt)["sd_dex"]
    reporter.check(
        "correlated noise widens the decay bound", correlated >= long_,
        f"{correlated:.5f} dex with flicker vs {long_:.5f} dex if independent", fatal=False,
        note="Correlated samples carry less information than independent ones, so folding the "
             "flicker in can only widen the bound. A violation would mean the effective sample "
             "size is applied as a variance where it should be a standard deviation, or "
             "inverted.")

    reporter.check(
        "the shallow-decay degeneracy widens the bound most at small p",
        (ib.crb_prob_bleach_dex(box["prob_photo_bleach"]["low"], 1000, DEFAULT_RELATIVE_NOISE)["sd_dex"]
         > ib.crb_prob_bleach_dex(box["prob_photo_bleach"]["high"], 1000, DEFAULT_RELATIVE_NOISE)["sd_dex"]),
        "the bound at the bottom of the prior exceeds the bound at the top", fatal=False,
        note="A shallower decay is closer to a straight line, so the amplitude, rate and offset "
             "are more degenerate and the rate is less constrained. If this ordering reversed, "
             "the amplitude and offset are not being profiled out of the information matrix.")

    np.savez_compressed(os.path.join(out_dir, "information_budget.npz"),
                        per_spot_log_width=budget["per_spot_log_width"],
                        per_spot_log_amplitude=budget["per_spot_log_amplitude"],
                        spot_photons=budget["spot_photons"],
                        n_emitters=budget["n_emitters"], n_frames=budget["n_frames"])
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(dict(
            n_frames=timing.frame_count, n_emitters=args.n_emitters,
            spot_photons=budget["spot_photons"],
            per_spot_log_width=budget["per_spot_log_width"],
            mu_r_bound_dex=w["center"]["sd_log_scale_dex"],
            sigma_r_bound=w["center"]["sd_shape"],
            sigma_pc_bound=budget["brightness_population"]["sd_shape"],
            prob_bleach_bound_dex=budget["bleach"]["center"]["correlated"]["sd_dex"],
            lambda_rate_bound_dex=budget["flicker"]["center"]["sd_dex"],
            measured=measured,
        ), fh, indent=2, default=float)

    reporter.summary()
    path = reporter.write_report()
    print(f"\n[{STAGE}] report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
