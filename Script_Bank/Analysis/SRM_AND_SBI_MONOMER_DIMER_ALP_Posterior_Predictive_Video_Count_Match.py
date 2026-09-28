"""Posterior-predictive video check, companion script: the receptor total of a declared-configuration render.

Part of the posterior-predictive video check; the files of the check and their order are listed in
``SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md`` ("The files of the posterior-predictive video
check"). The value chosen approximates one recording's opening spot density. OPTIONAL: the visual check
declares the receptor total (one value for every recording) and does not use this script; it serves a
density-matched render when one is wanted.

ROLE. A posterior-predictive render at a declared reaction-diffusion configuration
(``--declared-rds``) needs a receptor total ``N_R``, and the recordings do not state one. The value
chosen here is a RENDERING SETTING that approximates the recording's opening spot density, not a
validated biological estimate. Dividing a spot count by the per-subunit visibility would ignore the
dimers, the finite field, the missed detections and the counting saturation, so the approximation is
made through the forward model itself: short clips are rendered by the production chain
(``posterior_predictive_video_runner.simulate_and_render``: the RDS stage's system, the DLI stage's
labeling at the condition's declared occupancy, its renderer with the tagged ``Nuisance_DLI`` imaging
vector), and spots are counted on the recording and on each render by the direct estimators'
detection (``direct_imaging_estimates.measure_spot_widths`` on the stored 8-bit levels: 4 sigma,
fit half-width 14 px, SCOPE box center) over the first ``--match-seconds``.

METHOD. A render's detected count per frame is written ``c = eta * V``: ``V`` is its number of
visible hosts at frame 0 (its labeling record) and ``eta`` the detected-per-visible-host ratio, which
carries the detection losses, the overlaps, the bleaching within the window and the exits from the
field at that density. With the plan's per-subunit visibility ``a`` and the declared initial ratio
``r``, the expected number of visible hosts is ``N_R * v`` with
``v = (a_A + r (1 - (1 - a_B)^2)) / (1 + 2 r)``. Each iteration renders at the current ``N_R``,
measures ``eta``, and proposes ``N_R = c_exp / (eta * v)``; a step is limited to a factor of two.
Every render -- the first, each update and the check renders -- first passes one count test
(``count_problem``): no render exceeds ``--max-count``, and none lies outside the prior box of ``N_R``
(the training support) unless ``--allow-outside-prior`` records the user's decision to render there.
Normalizing by the realized initial visibility reduces the labeling-count
variability of a single render; the remaining simulation variability, and the dependence of
``eta`` on density, arrangement, overlap, dye multiplicity and motion, are not removed
(``E[eta V]`` is not ``E[eta] E[V]`` in general). The tolerance (default 3 %) is an ITERATION
criterion, not an uncertainty of ``N_R``. After acceptance, ``--check-renders`` independent renders at
the accepted value report the detected count they produce, beside the recording's.

FAILURE PATHS, all explicit, recorded and returned as a non-zero exit status (the diagnostics are
kept; only an accepted status carries a count):

    undefined_no_experimental_detections   the recording's opening window has no accepted spot
    undefined_no_visibility                 the plan and composition give no visible host
    undefined_no_visible_hosts              a render has no visible host at frame 0 (eta undefined)
    undefined_no_synthetic_detections       a render has no accepted spot (eta = 0)
    undefined_nonfinite_update              the proposal is not a finite positive count
    unresolved_saturation                   the detected count no longer responds to N_R (its
                                            elasticity, from the eta trend over a step of at least
                                            25 %, is below --min-elasticity while a larger count is
                                            proposed): count matching unresolved under detector
                                            saturation; no count is reported
    unresolved_max_count                    the next count (the first included) or the converged
                                            one exceeds --max-count; nothing is rendered above it
    user_decision_outside_prior             the next count (the first included) or the converged
                                            one lies outside the prior box: nothing is rendered
                                            there, and the user decides (--allow-outside-prior)
    not_converged                           the iteration budget is exhausted

Accepted: ``matched`` (inside the prior box) and ``matched_outside_prior`` (only under
``--allow-outside-prior``, the user's decision; flagged as outside the training support).

WHAT IT DOES NOT DO. A render at the accepted count has the recording's opening density by
construction; that density is not evidence of realism (later-window densities remain checks, since
only the opening window is used). Nothing is clipped. The accepted value is entered by hand in the
declared configuration's ``per_cell`` table.

OUTPUTS never overwrite: every run writes into an attempt folder named by the configuration's
scenario, the imaging tag and ``--attempt-label``, and the run refuses, before any computation, to
start when a requested cell's output already exists there.

WHERE TO RUN. Any machine with ReaDDy and the staged recordings (CPU, about 30 s per render).
Post-hoc, user-driven analysis in Script_Bank/Analysis, NOT a canonical pipeline stage.

Writes <data_bank>/<posit>/<alias>_<timing>_Posterior_Predictive_Video/Count_Match/
           <SCENARIO>_<TAG>[_<attempt>]/<alias>_<timing>_Count_Match_<KIND>_Cell_<cell>.{json,md}

Usage:
    MACHINE_PROFILE=<p> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Count_Match.py \\
        --total-time-seconds 2 --kind MET-FAB --cell 0 5 --nuisance-tag REF \\
        --declared-rds Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Declared_RDS_FAB.toml
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

STATUS_ACCEPTED = ("matched", "matched_outside_prior")

# The direct estimators' detection (DETECTOR_WORKFLOW.md sec. 9.6), applied to both videos.
N_SIGMA = 4.0
HALF_PX = 14
MIN_ELASTICITY_STEP = np.log(1.25)       # the elasticity is read only across a step of >= 25 %
MAX_STEP_FACTOR = 2.0                    # one update moves N_R by at most this factor


# ==========================================================================================
# The iteration (pure: renders are supplied by a callable, so every path is testable)
# ==========================================================================================

def count_problem(n, *, max_count, prior_box, allow_outside_prior, what="N_R"):
    """Why a render at ``N_R = n`` may not be made, as ``(status, reason)``, or None when it may.

    Every render passes this test before it is made: the first, each update and the check renders.
    No render exceeds ``max_count`` (a resource and saturation guard), and none lies outside
    ``prior_box`` (the training support) unless ``allow_outside_prior`` records the user's decision.
    """
    if not np.isfinite(n) or round(n) < 1:
        return "undefined_nonfinite_update", f"{what} {n!r} is not a finite positive count"
    n = int(round(n))
    if n > max_count:
        return "unresolved_max_count", f"{what} {n} exceeds --max-count {max_count}"
    low, high = prior_box
    if not allow_outside_prior and not low - 1e-9 <= n <= high + 1e-9:
        return ("user_decision_outside_prior",
                f"{what} {n} lies outside the prior box {low:.0f}-{high:.0f} (the training support); "
                f"rendering there is the user's decision (--allow-outside-prior)")
    return None


def guarded_render(render_count, *, max_count, prior_box, allow_outside_prior):
    """``render_count`` behind :func:`count_problem`: a render the limits exclude raises before it is
    made. The iteration stops before it reaches one; the guard also covers the check renders."""
    def render(n_r, tag):
        problem = count_problem(n_r, max_count=max_count, prior_box=prior_box,
                                allow_outside_prior=allow_outside_prior, what="the render at N_R")
        if problem:
            raise RuntimeError(f"refusing to render: {problem[1]}")
        return render_count(n_r, tag)
    return render


def iterate_count(c_exp, v, render_count, *, initial, tolerance, max_iterations, max_count,
                  min_elasticity, prior_box, allow_outside_prior=False):
    """Damped fixed-point iteration for the receptor total, with every failure path explicit.

    Args:
        c_exp: the recording's detected spots per frame over the match window.
        v: expected visible hosts per receptor subunit at frame 0.
        render_count: ``(n_r, iteration) -> (visible_hosts_0, detected_per_frame)`` for one render.
        initial, tolerance, max_iterations, max_count, min_elasticity: see the parser.
        prior_box: ``(low, high)`` of ``N_R`` in physical units (the training support).
        allow_outside_prior: the user's decision to render outside ``prior_box`` (default: no).

    Returns:
        dict with ``status`` (one of the module docstring's), ``reason``, ``iterations``,
        ``matched_count_total`` (an int only for an accepted status, else None) and
        ``last_count_total`` (the last count rendered, a diagnostic).
    """
    iterations = []
    limits = dict(max_count=max_count, prior_box=prior_box, allow_outside_prior=allow_outside_prior)

    def result(status, reason, matched=None):
        last = iterations[-1]["count_total"] if iterations else None
        return dict(status=status, reason=reason, iterations=iterations,
                    matched_count_total=matched if status in STATUS_ACCEPTED else None,
                    last_count_total=last)

    if not np.isfinite(c_exp) or c_exp <= 0:
        return result("undefined_no_experimental_detections",
                      f"the recording's match window has {c_exp!r} detected spots per frame")
    if not np.isfinite(v) or v <= 0:
        return result("undefined_no_visibility", f"expected visible hosts per subunit is {v!r}")
    n = float(initial)
    for it in range(int(max_iterations)):
        problem = count_problem(n, **limits, what="the initial N_R" if it == 0 else "the next N_R")
        if problem:                                   # checked before the render, never after it
            return result(*problem)
        n_rendered = int(round(n))
        visible, c_syn = render_count(n_rendered, it)
        rec = dict(iteration=it, count_total=n_rendered, visible_hosts_0=int(visible),
                   expected_visible_hosts=n_rendered * v, detected_per_frame=float(c_syn))
        iterations.append(rec)
        if visible <= 0:
            return result("undefined_no_visible_hosts",
                          f"the render at N_R {n_rendered} has no visible host at frame 0")
        if not np.isfinite(c_syn) or c_syn <= 0:
            return result("undefined_no_synthetic_detections",
                          f"the render at N_R {n_rendered} has {c_syn!r} detected spots per frame")
        eta = c_syn / visible
        proposal = c_exp / (eta * v)
        rec.update(eta=eta, proposed_count_total=proposal)
        if not np.isfinite(proposal) or proposal <= 0:
            return result("undefined_nonfinite_update", f"the proposal is {proposal!r}")
        if len(iterations) >= 2:
            prev = iterations[-2]
            step = np.log(n_rendered / prev["count_total"])
            if abs(step) >= MIN_ELASTICITY_STEP:
                elasticity = 1.0 + np.log(eta / prev["eta"]) / step
                rec["elasticity"] = float(elasticity)
                if elasticity < min_elasticity and proposal > n_rendered:
                    return result("unresolved_saturation",
                                  f"count matching unresolved under detector saturation: from N_R "
                                  f"{prev['count_total']} to {n_rendered} the detected count responds with "
                                  f"elasticity {elasticity:.2f} (< {min_elasticity:g}) while a larger count "
                                  f"({proposal:.0f}) is proposed")
        if abs(proposal - n_rendered) / n_rendered <= tolerance:
            matched = int(round(proposal))
            problem = count_problem(matched, **limits, what="the converged N_R")
            if problem:                               # the check renders would be made at it
                return result(*problem)
            inside = prior_box[0] - 1e-9 <= matched <= prior_box[1] + 1e-9
            return result("matched" if inside else "matched_outside_prior",
                          f"successive values within {tolerance:.0%} (an iteration criterion)"
                          + ("" if inside else f"; outside the prior box {prior_box[0]:.0f}-"
                                               f"{prior_box[1]:.0f}, rendered by the user's decision"),
                          matched)
        n_next = min(max(proposal, n_rendered / MAX_STEP_FACTOR), MAX_STEP_FACTOR * n_rendered)
        if n_next > max_count:
            if n_rendered >= max_count:
                return result("unresolved_max_count",
                              f"the proposal {proposal:.0f} stays above --max-count {max_count}")
            n_next = float(max_count)
        rec["next_count_total"] = n_next
        n = n_next
    return result("not_converged", f"no two successive values within {tolerance:.0%} after "
                                   f"{int(max_iterations)} iterations")


def visible_hosts_per_subunit(plan, ratio: float) -> float:
    """Expected visible hosts per receptor subunit at frame 0 for the initial ratio ``r = n_B / n_A``."""
    a_mono, a_dim = plan.visible_per_subunit
    return (a_mono + ratio * (1.0 - (1.0 - a_dim) ** 2)) / (1.0 + 2.0 * ratio)


def attempt_folder_name(scenario_name, nuisance_tag, attempt_label=None) -> str:
    """``<SCENARIO>_<TAG>[_<attempt>]``: the identity of one set of match results."""
    safe = lambda t: "".join(c if c.isalnum() else "_" for c in str(t)).strip("_")
    parts = [safe(scenario_name), safe(nuisance_tag) if nuisance_tag else "CANONICAL"]
    if attempt_label:
        parts.append(safe(attempt_label))
    return "_".join(parts)


def preflight_outputs(paths_by_cell) -> None:
    """Refuse, before any computation, to write over an existing result."""
    existing = [str(p) for paths in paths_by_cell.values() for p in paths if Path(p).exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing count-match results:\n  "
                         + "\n  ".join(existing)
                         + "\npass --attempt-label to write a distinct attempt folder.")


# ==========================================================================================
# One recording
# ==========================================================================================

def _scope_center(det) -> dict:
    """SCOPE camera values at the center of their box: the detection rule of the direct estimators."""
    return {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]))
            for e in det.DETECTOR_NUISANCE_SCOPE}


def detected_per_frame(video8):
    """Accepted spot-frames per frame of a stored 8-bit video under the direct estimators' rules."""
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp import direct_imaging_estimates as die
    m = die.measure_spot_widths(video8, _scope_center(det), frame_stride=1, n_sigma=N_SIGMA,
                                half_px=HALF_PX)
    return np.bincount(m["frame_index"], minlength=video8.shape[0]).astype(float)


def _context(args):
    from srm_and_sbi_monomer_dimer_alp import labeling as lab
    from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv
    from srm_and_sbi_monomer_dimer_alp.experiment_support import KIND_OF_CONDITION
    from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERIZATION, PARAMETERS, RunTiming
    from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow
    kind = KIND_OF_CONDITION.get(args.kind, args.kind)
    paths = biology_workflow().paths.with_condition(kind)
    data_bank_root = PARAMETERS.machine.data_bank_root
    map_label = RunTiming(total_time_seconds=args.total_time_seconds,
                          frames=PARAMETERS.simulation.timing).label
    first_cell = args.cell[0]
    _, record = ppv.load_declared_rds(args.declared_rds, kind, first_cell,
                                      overrides={"count_total": args.initial_count})
    folder = (data_bank_root / paths.posit_subdir
              / f"{paths.project_alias}_{map_label}_Posterior_Predictive_Video" / "Count_Match"
              / attempt_folder_name(record["scenario"]["name"], args.nuisance_tag, args.attempt_label))
    stems = {c: folder / f"{paths.project_alias}_{map_label}_Count_Match_{args.kind}_Cell_{c}"
             for c in args.cell}
    count_entry = next(e for e in PARAMETERIZATION if e["KEY"] == "count_total")
    prior_box = tuple(10.0 ** np.asarray(count_entry["PRIOR_RANGE"], dtype=float))
    limits = dict(max_count=args.max_count, prior_box=prior_box,
                  allow_outside_prior=args.allow_outside_prior)
    return dict(kind=kind, paths=paths, data_bank_root=data_bank_root, map_label=map_label,
                folder=folder, stems=stems, limits=limits, lab=lab, ppv=ppv, PARAMETERS=PARAMETERS)


def match_one(args, ctx, cell: int) -> dict:
    from srm_and_sbi_monomer_dimer_alp import __version__ as package_version
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp.experiment_support import read_recording
    from srm_and_sbi_monomer_dimer_alp.io import convert_video_dtype
    lab, ppv, PARAMETERS = ctx["lab"], ctx["ppv"], ctx["PARAMETERS"]
    kind = ctx["kind"]
    frame_time = PARAMETERS.simulation.timing.frame_time_seconds
    n_match = int(round(args.match_seconds / frame_time))
    folder = ctx["folder"]
    folder.mkdir(parents=True, exist_ok=True)

    plan = lab.resolve_labeling(kind, args.labeling_law, args.occupancy)
    imaging = ppv.resolve_biology_imaging(PARAMETERS.machine.root_for("EVAL"), ctx["map_label"], kind,
                                          nuisance_tag=args.nuisance_tag)
    tif = ctx["paths"].experiment_video_path(kind, cell, args.experiment_span_seconds,
                                             ctx["data_bank_root"])
    raw = read_recording(tif)
    if raw.shape[0] < n_match:
        raise ValueError(f"{tif} holds {raw.shape[0]} frames; the match window needs {n_match}.")
    t0 = time.time()
    c_exp = float(detected_per_frame(convert_video_dtype(raw[:n_match], bits_from=16, bits_to=8)).mean())
    print(f"[cell {cell}] experimental: {c_exp:.2f} detected spots per frame over the first "
          f"{args.match_seconds:g} s ({time.time() - t0:.0f} s)", flush=True)
    _, base_record = ppv.load_declared_rds(args.declared_rds, kind, cell,
                                           overrides={"count_total": args.initial_count})
    ratio = base_record["parameters"]["ratio_dimer_monomer_initial"]["value"]
    v = visible_hosts_per_subunit(plan, ratio)
    limits = ctx["limits"]
    prior_box = limits["prior_box"]

    with tempfile.TemporaryDirectory(dir=str(folder), prefix="_count_match_") as work:
        def render_one(n_r, tag):
            vec, _ = ppv.load_declared_rds(args.declared_rds, kind, cell, overrides={"count_total": n_r})
            seed = None if args.seed is None else args.seed + 1000 * cell + tag
            t1 = time.time()
            clip = ppv.simulate_and_render(kind, vec, imaging["vector"], plan, n_match,
                                           Path(work) / f"cell_{cell}_{tag}.h5", seed=seed)
            row = dict(zip(lab.LABELING_SET_COLUMNS, clip["labeling_row"]))
            visible = row["monomers_visible_0"] + row["dimers_visible_0"]
            c_syn = float(detected_per_frame(convert_video_dtype(clip["synth"], bits_from=16, bits_to=8)).mean())
            print(f"[cell {cell}] render {tag}: N_R {n_r} -> visible {int(visible)} (expected {n_r * v:.0f}), "
                  f"detected {c_syn:.2f}/frame ({time.time() - t1:.0f} s)", flush=True)
            return visible, c_syn

        render = guarded_render(render_one, **limits)          # the iteration's and the check renders'
        out = iterate_count(c_exp, v, render, initial=args.initial_count, tolerance=args.tolerance,
                            max_iterations=args.max_iterations,
                            min_elasticity=args.min_elasticity, **limits)
        checks = []
        if out["status"] in STATUS_ACCEPTED:
            for k in range(args.check_renders):
                visible, c_syn = render(out["matched_count_total"], 100 + k)
                checks.append(dict(seed_offset=100 + k, visible_hosts_0=int(visible),
                                   detected_per_frame=c_syn,
                                   eta=(c_syn / visible) if visible > 0 else None))

    resolved_n = out["matched_count_total"] if out["matched_count_total"] is not None else out["last_count_total"]
    resolved = None
    if resolved_n is not None:
        vec, rec = ppv.load_declared_rds(args.declared_rds, kind, cell, overrides={"count_total": resolved_n})
        resolved = {k: row["value"] for k, row in rec["parameters"].items()}
    result = dict(
        kind=args.kind, condition=kind, cell=cell, recording=str(tif),
        status=out["status"], reason=out["reason"],
        matched_count_total=out["matched_count_total"], last_count_total=out["last_count_total"],
        interpretation=("a rendering setting approximating the recording's opening spot density, "
                        "matched by construction; not a validated biological estimate; the tolerance "
                        "is an iteration criterion, not an uncertainty of N_R"),
        prior_box_count_total=list(prior_box),
        match_seconds=args.match_seconds, match_frames=n_match,
        detection=dict(rule="direct_imaging_estimates.measure_spot_widths on stored 8-bit levels",
                       n_sigma=N_SIGMA, half_px=HALF_PX, scope="SCOPE box center"),
        experimental_detected_per_frame=c_exp,
        labeling=plan.record(), visible_hosts_per_subunit=v,
        imaging=dict(description=imaging["description"], identity=imaging["identity"],
                     vector=imaging["vector"].tolist(), keys=det.DETECTOR_IMAGING_KEYS),
        declared_rds=dict(path=base_record["path"], sha256=base_record["sha256"],
                          scenario=base_record["scenario"]["name"]),
        resolved_parameters=resolved,
        iterations=out["iterations"], check_renders=checks,
        settings=dict(initial=args.initial_count, tolerance=args.tolerance,
                      max_iterations=args.max_iterations, max_count=args.max_count,
                      allow_outside_prior=args.allow_outside_prior,
                      min_elasticity=args.min_elasticity, max_step_factor=MAX_STEP_FACTOR,
                      seed=args.seed),
        package_version=package_version, argv=sys.argv[1:],
    )
    stem = ctx["stems"][cell]
    Path(f"{stem}.json").write_text(json.dumps(result, indent=1, default=str))
    _write_md(Path(f"{stem}.md"), result, tif, plan, imaging, base_record)
    verdict = (f"accepted N_R = {out['matched_count_total']} ({out['status']})"
               if out["status"] in STATUS_ACCEPTED else f"NO count ({out['status']}: {out['reason']})")
    print(f"[cell {cell}] {verdict}; wrote {stem}.json", flush=True)
    return result


def _write_md(path, r, tif, plan, imaging, record):
    lines = [f"# Receptor total for a posterior-predictive render: {r['kind']} cell {r['cell']}", "",
             f"**Status: {r['status']}** -- {r['reason']}.", "",
             f"The value is a rendering setting that approximates the recording's opening spot density "
             f"through the production chain (see the script's docstring). A render at it has that density "
             f"by construction, which is not evidence of realism; the tolerance is an iteration criterion, "
             f"not an uncertainty of N_R.", "",
             f"- recording: `{tif.name}`, first {r['match_seconds']:g} s ({r['match_frames']} frames): "
             f"{r['experimental_detected_per_frame']:.2f} detected spots per frame (4 sigma, half-width 14 px, "
             f"SCOPE box center, 8-bit levels)",
             f"- labeling: {plan.describe()}; expected visible hosts per subunit "
             f"{r['visible_hosts_per_subunit']:.4f}",
             f"- imaging: {imaging['description']}",
             f"- declared configuration: `{Path(record['path']).name}` (sha256 {record['sha256'][:12]}), "
             f"scenario {record['scenario']['name']}", "",
             "| iteration | N_R rendered | visible hosts (frame 0) | expected | detected per frame | eta | proposed N_R |",
             "|---|---|---|---|---|---|---|"]
    for it in r["iterations"]:
        eta = it.get("eta")
        prop = it.get("proposed_count_total")
        lines.append(f"| {it['iteration']} | {it['count_total']} | {it['visible_hosts_0']} | "
                     f"{it['expected_visible_hosts']:.0f} | {it['detected_per_frame']:.2f} | "
                     f"{'-' if eta is None else f'{eta:.3f}'} | {'-' if prop is None else f'{prop:.0f}'} |")
    if r["check_renders"]:
        dets = [c["detected_per_frame"] for c in r["check_renders"]]
        lines += ["", f"Independent check render(s) at the accepted value: detected "
                      f"{', '.join(f'{d:.2f}' for d in dets)} spots per frame (recording "
                      f"{r['experimental_detected_per_frame']:.2f})."]
    lo, hi = r["prior_box_count_total"]
    if r["matched_count_total"] is not None:
        lines += ["", f"**Accepted N_R = {r['matched_count_total']}**; prior box {lo:.0f}-{hi:.0f}: "
                      + ("inside." if r["status"] == "matched" else
                         "OUTSIDE the training support (rendered under --allow-outside-prior, the "
                         "user's decision; flagged).")]
    elif r["status"] == "user_decision_outside_prior":
        lines += ["", f"**No count accepted: the user's decision is required** (prior box {lo:.0f}-{hi:.0f}; "
                      f"last rendered {r['last_count_total']}). Nothing was rendered outside the box."]
    else:
        lines += ["", f"**No count accepted** (last rendered {r['last_count_total']})."]
    path.write_text("\n".join(lines) + "\n")


def build_parser():
    from srm_and_sbi_monomer_dimer_alp.experiment_support import KIND_OF_CONDITION
    p = argparse.ArgumentParser(description="Choose a receptor total that approximates recordings' opening "
                                            "spot density (see the module docstring).")
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window of the imaging calibration; locates the tagged Nuisance_DLI.")
    p.add_argument("--kind", default="MET-FAB", choices=tuple(KIND_OF_CONDITION))
    p.add_argument("--cell", type=int, nargs="+", required=True, help="one or more recording indices.")
    p.add_argument("--declared-rds", required=True, help="the declared configuration (TOML).")
    p.add_argument("--nuisance-tag", default=None, help="the tagged Nuisance_DLI artifact (e.g. REF).")
    p.add_argument("--attempt-label", default=None,
                   help="names a distinct attempt folder; results are never overwritten.")
    p.add_argument("--labeling-law", default=None, help="as at the DLI stage (default: the baseline).")
    p.add_argument("--occupancy", default=None, help="as at the DLI stage (default: declared/derived).")
    p.add_argument("--experiment-span-seconds", type=int, default=20)
    p.add_argument("--match-seconds", type=float, default=2.0,
                   help="opening window over which the counts are compared (default 2 s).")
    p.add_argument("--initial-count", type=float, default=1000.0, help="first N_R rendered.")
    p.add_argument("--tolerance", type=float, default=0.03,
                   help="iteration criterion: relative agreement of the rendered and proposed N_R.")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--max-count", type=int, default=10000,
                   help="no render above this receptor total, the first and the check renders "
                        "included (a resource and saturation guard).")
    p.add_argument("--allow-outside-prior", action="store_true",
                   help="the user's decision to render outside the prior box of N_R (the training "
                        "support). Without it nothing is rendered there: a count outside the box, the "
                        "first included, ends the cell as user_decision_outside_prior.")
    p.add_argument("--min-elasticity", type=float, default=0.3,
                   help="below this response of the detected count to N_R, matching is unresolved "
                        "(detector saturation).")
    p.add_argument("--check-renders", type=int, default=1,
                   help="independent renders at the accepted value, reported beside the recording.")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed; render k of cell c uses seed + 1000 c + k.")
    return p


def main(argv=None):
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    ctx = _context(args)
    preflight_outputs({c: (Path(f"{s}.json"), Path(f"{s}.md")) for c, s in ctx["stems"].items()})
    problem = count_problem(args.initial_count, **ctx["limits"], what="--initial-count")
    if problem:                                        # before any recording is read or render made
        raise SystemExit(f"{problem[1]}.")
    results = [match_one(args, ctx, cell) for cell in args.cell]
    failed = [r["cell"] for r in results if r["status"] not in STATUS_ACCEPTED]
    if failed:
        print(f"no count accepted for cell(s) {failed}; see their reports.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
