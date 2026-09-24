"""Frozen acceptance rules for the direct estimators (``DETECTOR_WORKFLOW.md`` §9.6).

One implementation of the four evaluation steps, shared by the three direct-estimator utilities
so that a rule cannot drift between them:

1. evidence adequacy of the run          (recordings attempted, overall and in the operating subgroup)
2. operational success                   (fraction of attempted recordings with a valid estimate)
3. evidence adequacy for accuracy         (successful estimates remaining, overall / operating / quartile)
4. accuracy, and uncertainty coverage     (the per-estimator thresholds; the 90 % range's coverage)

Verdicts are kept separate and reported side by side: ``PASS``, ``FAIL (operational)``,
``FAIL (protocol)`` (a drop without a reason code: a reporting failure of the run, never a
measurement failure), ``FAIL (accuracy)``, ``FAIL (uncertainty)``, ``INSUFFICIENT EVIDENCE (run)``,
``INSUFFICIENT EVIDENCE (accuracy)``. One does not absorb another. A selftest of a few in-memory
scenes never reaches a verdict; its accuracy numbers are reported as ``SELFTEST (informational)``.

Two guards surround the steps. Before any recording is read, `check_run_purpose` holds a tier run
to the declared split of the EVAL tiers: a development run reads development tasks only, a verdict
run exactly the reserved ones. After the steps, `apply_code_provenance` makes a run whose
implementation changed while it executed invalid for acceptance, whatever its verdicts.

The module is a pure kernel: no file access, no machine profile, no printing except through the
``DiagnosticReporter`` handed to `render`. Subgroup boundaries come from the detector prior and are
fixed here, never recomputed from the recordings that happened to succeed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import detector_parameterization as det

# The numbers of §9.6. Changing any of them is a documented, versioned decision, not a tuning.
RULES: Dict[str, float] = dict(
    attempted_min=1000,
    attempted_operating_min=400,
    success_fraction_overall=0.95,
    success_fraction_operating=0.90,
    successful_min=900,
    successful_operating_min=250,
    quartile_min=100,
    nominal_coverage=0.90,
    coverage_min=0.85,
    bleach_usable_min=100,
    bleach_usable_operating_min=50,
)

# The operating subgroup: true log10 mu_pc in the lower half of its prior (about 100-237 photons
# per dye). Motivated by the experimental estimates of §6.11; it does not assert experimental truth.
OPERATING_KEY = "mu_pc"
OPERATING_RANGE = (2.0, 2.375)

# Quartiles are reported for these parameters (each estimator adds its own target if absent).
STRATIFY_KEYS: Tuple[str, ...] = ("mu_pc", "mu_r", "sigma_r")

_Z90 = 1.6448536269514722   # two-sided 90 % normal quantile, for the nominal ranges
_Z95 = 1.959963984540054    # for the Wilson interval on a measured coverage

# The declared split of the MET-FAB EVAL tiers (DETECTOR_WORKFLOW.md sec. 9.6, "Declared split
# (2026-09-24)"). Per (condition, frames per recording): the development tasks, the tasks reserved
# for a verdict, and the estimators whose verdict the reserved tasks are kept for. A tier not listed
# has no reserved task yet. Changing an entry is a documented, versioned decision, like RULES.
DECLARED_SPLIT: Dict[Tuple[str, int], dict] = {
    ("FAB", 100): dict(development=tuple(range(0, 20)), reserved=tuple(range(20, 25)),
                       verdict_estimators=("Direct_PSF_Width", "Direct_Flicker_Rate")),
    ("FAB", 1000): dict(development=tuple(range(0, 10)), reserved=tuple(range(10, 20)),
                        verdict_estimators=("Direct_Fluorescence_Loss",)),
}
PURPOSES: Tuple[str, ...] = ("development", "verdict")
# The token a tier run's folder name carries after the utility name: <...>_Direct_PSF_Width_DEV_<commit>.
PURPOSE_TOKENS: Dict[str, str] = {"development": "DEV", "verdict": "VERDICT"}

# Exit status of the utilities: 0 nothing failed, 1 a FAIL verdict, 2 insufficient evidence only,
# 3 the implementation changed during the run. Python itself exits 1 on an uncaught exception and
# argparse exits 2 on an argument error or a refusal, so 1 and 2 are not verdicts alone: the log says which.
EXIT_IMPLEMENTATION_CHANGED = 3


# ------------------------------------------------------------------------------------------
# The purpose of a tier run, checked before any recording is read
# ------------------------------------------------------------------------------------------
def check_run_purpose(purpose: str, *, estimator: str, condition: str, n_frames: int, split: str,
                      tasks: Sequence[int], max_videos: int, expect_videos_per_task: int) -> dict:
    """Hold a tier run's tasks to its declared purpose; a pure check, run before any read.

    A development run reads development tasks only: on a tier with a declared split every task must
    be one of its development tasks, because a reserved task read for any other purpose leaves the
    reserved set. A verdict run reads exactly the reserved EVAL tasks of its tier, every recording of
    them, with an estimator the set is kept for. TRAIN and TEST tasks, and EVAL tiers without a
    declared split, hold no reserved task: development runs may read them and verdict runs may not.

    Returns the record the run folder stores (purpose, tier, tasks, and the declared development and
    reserved tasks of the tier); raises ``ValueError`` naming the reason for a refusal.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose {purpose!r}: use one of {', '.join(PURPOSES)}")
    tasks = [int(t) for t in tasks]
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"tasks {tasks}: each task is read once")
    declared = DECLARED_SPLIT.get((str(condition), int(n_frames))) if split == "EVAL" else None
    tier = f"the {condition} {split} tier at {int(n_frames)} frames"
    record = dict(purpose=purpose, estimator=estimator, condition=condition, n_frames=int(n_frames),
                  split=split, tasks=sorted(tasks), split_declared=declared is not None,
                  development_tasks=list(declared["development"]) if declared else None,
                  reserved_tasks=list(declared["reserved"]) if declared else None)
    if purpose == "development":
        if declared:
            outside = sorted(set(tasks) - set(declared["development"]))
            if outside:
                reserved = sorted(set(outside) & set(declared["reserved"]))
                raise ValueError(
                    f"a development run reads development tasks only; tasks {outside} of {tier} are not "
                    f"development tasks" + (f" ({reserved} are reserved for a verdict)" if reserved else "")
                    + f". Development tasks: {declared['development'][0]} to {declared['development'][-1]} "
                    f"(DETECTOR_WORKFLOW.md sec. 9.6, declared split).")
        return record
    if split != "EVAL":
        raise ValueError(f"a verdict run reads the reserved EVAL tasks; split {split} holds none")
    if not declared:
        raise ValueError(f"no reserved set is declared for {tier} (DETECTOR_WORKFLOW.md sec. 9.6)")
    if estimator not in declared["verdict_estimators"]:
        raise ValueError(f"the reserved tasks of {tier} are kept for the verdict of "
                         f"{', '.join(declared['verdict_estimators'])}, not of {estimator}")
    if sorted(tasks) != list(declared["reserved"]):
        raise ValueError(f"a verdict run reads exactly the reserved tasks of {tier}, "
                         f"{list(declared['reserved'])}; got {sorted(tasks)}")
    need = len(declared["reserved"]) * int(expect_videos_per_task)
    if int(max_videos) < need:
        raise ValueError(f"--max-videos {max_videos} would cut the verdict short: the reserved tasks hold "
                         f"{need} recordings at {expect_videos_per_task} per task")
    return record


# ------------------------------------------------------------------------------------------
# Prior-fixed subgroups
# ------------------------------------------------------------------------------------------
def prior_range(key: str) -> Tuple[float, float]:
    """The log10 prior range of a detector imaging parameter."""
    lo, hi = det.DETECTOR_PARAMETERIZATION[det.DETECTOR_FIND[key]]["PRIOR_RANGE"]
    return float(lo), float(hi)


def prior_quarters(key: str) -> np.ndarray:
    """Five edges dividing the log10 prior of ``key`` into its four fixed quarters."""
    lo, hi = prior_range(key)
    return np.linspace(lo, hi, 5)


def operating_mask(theta_log10: np.ndarray) -> np.ndarray:
    """Rows whose true ``mu_pc`` lies in the operating subgroup."""
    col = theta_log10[:, det.DETECTOR_FIND[OPERATING_KEY]]
    return (col >= OPERATING_RANGE[0]) & (col < OPERATING_RANGE[1])


def quartile_masks(theta_log10: np.ndarray, key: str) -> List[Tuple[str, np.ndarray]]:
    """``(label, mask)`` for the four prior-fixed quarters of ``key``; the top edge is inclusive."""
    edges = prior_quarters(key)
    col = theta_log10[:, det.DETECTOR_FIND[key]]
    out = []
    for i in range(4):
        hi_incl = (col <= edges[i + 1]) if i == 3 else (col < edges[i + 1])
        out.append((f"{key} Q{i + 1} [{edges[i]:.4g}, {edges[i + 1]:.4g})",
                    (col >= edges[i]) & hi_incl))
    return out


def wilson_interval(k: int, n: int, z: float = _Z95) -> Tuple[float, float]:
    """Wilson score interval for a proportion ``k/n``; ``(nan, nan)`` when ``n == 0``."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def normal_range(center: float, se: float, *, z: float = _Z90) -> Tuple[float, float]:
    """Symmetric nominal range ``center ± z·se``; ``(nan, nan)`` when the SE is not finite."""
    if not (np.isfinite(center) and np.isfinite(se)):
        return float("nan"), float("nan")
    return float(center - z * se), float(center + z * se)


# ------------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------------
# An accuracy function receives the boolean row mask to score (already restricted to valid rows)
# and returns {criterion: (observed, threshold_text, passed_or_None)}. ``passed`` is None when the
# criterion could not be computed (e.g. correlation on a constant sample).
AccuracyFn = Callable[[np.ndarray], Dict[str, Tuple[float, str, Optional[bool]]]]


def theta_to_log10(theta_physical: np.ndarray) -> np.ndarray:
    """Detector ``Theta_Set`` rows hold PHYSICAL values; every detector prior is stated in log10.

    All six learnable imaging parameters are log-scale rows, so the conversion is a plain
    log10 of every column. Non-positive entries (none are expected) become NaN and fall
    outside every subgroup rather than raising.
    """
    th = np.asarray(theta_physical, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(th > 0, np.log10(np.where(th > 0, th, 1.0)), np.nan)


def evaluate(theta: np.ndarray,
             valid: np.ndarray,
             reasons: Sequence[Optional[str]],
             accuracy: AccuracyFn,
             *,
             target_key: str,
             theta_is_log10: bool = False,
             ranges: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = None,
             usable: Optional[np.ndarray] = None,
             selftest: bool = False) -> dict:
    """Apply the four §9.6 steps and return every number and verdict, nothing printed.

    Args:
        theta: ``(n, 6)`` true detector imaging parameters, one row per ATTEMPTED recording, in
            the order of the detector parameterization. PHYSICAL units as stored in a
            ``Theta_Set`` unless ``theta_is_log10`` is True.
        valid: ``(n,)`` bool, True where the estimator returned a valid estimate.
        reasons: ``(n,)`` reason code per recording (``None`` where valid).
        accuracy: callable scoring a row mask; see `AccuracyFn`.
        target_key: the parameter this estimator targets (its quartiles are reported too).
        ranges: per emitted quantity, ``name -> (covered, width, prior_width)``: ``covered`` is
            ``(n,)`` bool, True where the nominal 90 % range contains the truth; ``width`` is the
            range width in that quantity's own coordinates; ``prior_width`` the prior width in the
            same coordinates. ``None`` if the estimator emits no range yet. Every named range must
            reach the coverage minimum for the uncertainty step to pass.
        usable: bleaching only — ``(n,)`` bool, True where the observable eligibility diagnostic
            declares the recording usable. Accuracy is then scored on ``valid & usable`` and the
            frozen usable minima apply; rejected recordings are scored beside them.
        selftest: a few in-memory scenes; report accuracy, reach no verdict.
    """
    theta_log10 = (np.asarray(theta, dtype=float) if theta_is_log10 else theta_to_log10(theta))
    valid = np.asarray(valid, dtype=bool)
    n_att = int(valid.size)
    op = operating_mask(theta_log10) if n_att else np.zeros(0, bool)
    res: dict = dict(rules=dict(RULES), operating=dict(key=OPERATING_KEY, range=OPERATING_RANGE),
                     selftest=bool(selftest), verdicts={})

    # -- reason codes -------------------------------------------------------------------
    reason_counts: Dict[str, int] = {}
    missing_reason = 0
    for v, r in zip(valid, reasons):
        if v:
            continue
        if not r:
            missing_reason += 1
            r = "(no reason recorded)"
        reason_counts[str(r)] = reason_counts.get(str(r), 0) + 1
    res["reasons"] = reason_counts
    res["dropped_without_reason"] = int(missing_reason)

    # -- step 1: evidence adequacy of the run ------------------------------------------
    n_att_op = int(op.sum())
    res["attempted"] = dict(overall=n_att, operating=n_att_op)
    run_ok = n_att >= RULES["attempted_min"] and n_att_op >= RULES["attempted_operating_min"]
    if selftest:
        res["verdicts"]["run_evidence"] = "SELFTEST (informational)"
    else:
        res["verdicts"]["run_evidence"] = "ADEQUATE" if run_ok else "INSUFFICIENT EVIDENCE (run)"

    # -- step 2: operational success ---------------------------------------------------
    n_ok = int(valid.sum())
    n_ok_op = int((valid & op).sum())
    frac = n_ok / n_att if n_att else float("nan")
    frac_op = n_ok_op / n_att_op if n_att_op else float("nan")
    res["success"] = dict(overall=n_ok, operating=n_ok_op,
                          fraction_overall=frac, fraction_operating=frac_op,
                          ci_overall=wilson_interval(n_ok, n_att),
                          ci_operating=wilson_interval(n_ok_op, n_att_op))
    if selftest or not run_ok:
        res["verdicts"]["operational"] = ("SELFTEST (informational)" if selftest
                                          else "NOT EVALUATED (run evidence insufficient)")
    else:
        # The measured success fraction is the estimator's operational record; a drop without a
        # reason code is a REPORTING failure of the run (the rule 'dropped without a reason = 0')
        # and is stated as its own verdict so it never masquerades as a measurement failure.
        op_ok = (frac >= RULES["success_fraction_overall"]
                 and frac_op >= RULES["success_fraction_operating"])
        res["verdicts"]["operational"] = "PASS" if op_ok else "FAIL (operational)"
    res["verdicts"]["reporting"] = ("PASS" if missing_reason == 0
                                    else f"FAIL (protocol: {missing_reason} drop(s) without a reason code)")

    # -- step 3: evidence adequacy for accuracy ----------------------------------------
    scored = valid.copy()
    if usable is not None:
        usable = np.asarray(usable, dtype=bool)
        scored = valid & usable
        res["usable"] = dict(overall=int(scored.sum()), operating=int((scored & op).sum()),
                             rejected_valid=int((valid & ~usable).sum()),
                             fraction_of_attempted=float(scored.sum() / n_att) if n_att else float("nan"))
        need, need_op = RULES["bleach_usable_min"], RULES["bleach_usable_operating_min"]
    else:
        need, need_op = RULES["successful_min"], RULES["successful_operating_min"]
    n_sc, n_sc_op = int(scored.sum()), int((scored & op).sum())
    res["scored"] = dict(overall=n_sc, operating=n_sc_op, required=dict(overall=need, operating=need_op))
    acc_evidence_ok = n_sc >= need and n_sc_op >= need_op
    if selftest:
        res["verdicts"]["accuracy_evidence"] = "SELFTEST (informational)"
    else:
        res["verdicts"]["accuracy_evidence"] = ("ADEQUATE" if acc_evidence_ok
                                                else "INSUFFICIENT EVIDENCE (accuracy)")

    # -- step 4a: accuracy, overall and operating (required), quartiles (reported) -------
    def _score(mask: np.ndarray) -> dict:
        m = mask & scored
        if int(m.sum()) == 0:
            return dict(n=0, criteria={})
        return dict(n=int(m.sum()), criteria=accuracy(m))

    res["accuracy"] = dict(overall=_score(np.ones(n_att, bool)), operating=_score(op))
    strat_keys = list(STRATIFY_KEYS) + ([target_key] if target_key not in STRATIFY_KEYS else [])
    quart = {}
    for key in strat_keys:
        for label, mask in quartile_masks(theta_log10, key):
            block = _score(mask)
            block["n_attempted"] = int(mask.sum())
            block["n_valid"] = int((mask & valid).sum())
            block["reported"] = block["n"] >= RULES["quartile_min"]
            quart[label] = block
    res["accuracy"]["quartiles"] = quart
    if usable is not None:
        rej = valid & ~usable
        res["accuracy"]["rejected"] = (dict(n=int(rej.sum()), criteria=accuracy(rej))
                                       if rej.any() else dict(n=0, criteria={}))

    def _all_pass(block: dict) -> Optional[bool]:
        flags = [p for (_, _, p) in block["criteria"].values()]
        if not flags or any(p is None for p in flags):
            return None
        return all(flags)

    if selftest:
        res["verdicts"]["accuracy"] = "SELFTEST (informational)"
    elif not acc_evidence_ok:
        res["verdicts"]["accuracy"] = "INSUFFICIENT EVIDENCE (accuracy)"
    else:
        a_all, a_op = _all_pass(res["accuracy"]["overall"]), _all_pass(res["accuracy"]["operating"])
        if a_all is None or a_op is None:
            res["verdicts"]["accuracy"] = "INSUFFICIENT EVIDENCE (accuracy)"
        else:
            res["verdicts"]["accuracy"] = "PASS" if (a_all and a_op) else "FAIL (accuracy)"

    # -- step 4b: uncertainty — coverage (reliability) and width (informativeness) -------
    if not ranges:
        res["uncertainty"] = None
        res["verdicts"]["uncertainty"] = "NOT EMITTED (no per-recording range)"
    else:
        unc, all_ok, any_n = {}, True, False
        for name, (covered, width, prior_width) in ranges.items():
            covered = np.asarray(covered, dtype=bool)
            width = np.asarray(width, dtype=float)
            has_range = scored & np.isfinite(width)
            k, n = int((covered & has_range).sum()), int(has_range.sum())
            k_op, n_op = int((covered & has_range & op).sum()), int((has_range & op).sum())
            cov = k / n if n else float("nan")
            cov_op = k_op / n_op if n_op else float("nan")
            w = width[has_range]
            width_rel = (dict(median=float(np.median(w) / prior_width),
                              p10=float(np.percentile(w, 10) / prior_width),
                              p90=float(np.percentile(w, 90) / prior_width))
                         if (w.size and prior_width) else None)
            unc[name] = dict(n=n, n_operating=n_op, coverage_overall=cov, ci_overall=wilson_interval(k, n),
                             coverage_operating=cov_op, ci_operating=wilson_interval(k_op, n_op),
                             width_relative_to_prior=width_rel,
                             n_without_range=int((scored & ~np.isfinite(width)).sum()))
            any_n = any_n or n > 0
            all_ok = all_ok and n > 0 and cov >= RULES["coverage_min"] and cov_op >= RULES["coverage_min"]
        res["uncertainty"] = dict(nominal=RULES["nominal_coverage"], ranges=unc)
        if selftest:
            res["verdicts"]["uncertainty"] = "SELFTEST (informational)"
        elif not acc_evidence_ok or not any_n:
            res["verdicts"]["uncertainty"] = "INSUFFICIENT EVIDENCE (accuracy)"
        else:
            res["verdicts"]["uncertainty"] = "PASS" if all_ok else "FAIL (uncertainty)"

    # -- exit status: 0 PASS / nothing failed; 1 any FAIL; 2 insufficient evidence only ----
    # (3, the implementation changed during the run, is set by `apply_code_provenance`.)
    v = list(res["verdicts"].values())
    if any(s.startswith("FAIL") for s in v):
        res["exit_code"] = 1
    elif any(s.startswith("INSUFFICIENT") for s in v) or selftest:
        res["exit_code"] = 2 if not selftest else 0
    else:
        res["exit_code"] = 0
    return res


def apply_code_provenance(res: dict, code: dict) -> dict:
    """Make a run whose implementation changed while it executed invalid for acceptance.

    ``code`` is the block `provenance.finalize_code_provenance` returns: the implementation hash at
    startup and at write. When they differ, part of the run may have executed one version and part
    another, so its results describe no single implementation. The verdicts stay in ``res`` as
    diagnostics, but the ``implementation`` verdict reads ``INVALID`` and the exit status becomes 3
    whatever the other verdicts were. Called after every other change to the verdicts.
    """
    changed = bool(code["changed_during_run"])
    res["verdicts"]["implementation"] = (
        "INVALID (implementation changed during the run; no verdict of this run may be used)" if changed
        else "PASS (implementation unchanged during the run)")
    res["valid_for_acceptance"] = not changed
    if changed:
        res["exit_code"] = EXIT_IMPLEMENTATION_CHANGED
    return res


# ------------------------------------------------------------------------------------------
# Rendering into a DiagnosticReporter
# ------------------------------------------------------------------------------------------
def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}"
    return str(x)


def _pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.1f} %"


def _ci(ci: Tuple[float, float]) -> str:
    lo, hi = ci
    return "n/a" if not (np.isfinite(lo) and np.isfinite(hi)) else f"[{100 * lo:.1f}, {100 * hi:.1f}] %"


def render(reporter, res: dict, *, estimator: str, target_key: str) -> None:
    """Write the §9.6 evaluation into the report: verdicts, the four steps, and the strata."""
    R = res["rules"]
    reporter.table(
        "Verdicts (DETECTOR_WORKFLOW.md §9.6)", ["step", "verdict"],
        [["0. implementation unchanged during the run", res["verdicts"].get("implementation", "NOT CHECKED")],
         ["1. evidence adequacy of the run", res["verdicts"]["run_evidence"]],
         ["2. operational success", res["verdicts"]["operational"]],
         ["2'. reporting protocol (reason code on every drop)", res["verdicts"]["reporting"]],
         ["3. evidence adequacy for accuracy and coverage", res["verdicts"]["accuracy_evidence"]],
         ["4a. accuracy (overall AND operating subgroup)", res["verdicts"]["accuracy"]],
         ["4b. uncertainty coverage (overall AND operating subgroup)", res["verdicts"]["uncertainty"]]],
        note="Verdicts are separate and none absorbs another: a large run with many failed estimates is an "
             "operational FAIL even when its accuracy step is also INSUFFICIENT EVIDENCE. Step 0 is the "
             "exception: a run whose implementation changed while it executed is INVALID, and none of its "
             "verdicts may be used. The operating "
             f"subgroup is true log10 {res['operating']['key']} in [{res['operating']['range'][0]}, "
             f"{res['operating']['range'][1]}), about 100-237 photons per dye; quartile boundaries are the "
             "prior's own quarters, fixed, never recomputed from the successful recordings.")

    att, suc, sc = res["attempted"], res["success"], res["scored"]
    rows = [
        ["recordings attempted, overall", str(att["overall"]), f">= {R['attempted_min']:.0f}"],
        ["recordings attempted, operating subgroup", str(att["operating"]), f">= {R['attempted_operating_min']:.0f}"],
        ["valid estimates, overall", f"{suc['overall']}  ({_pct(suc['fraction_overall'])}, CI {_ci(suc['ci_overall'])})",
         f">= {100 * R['success_fraction_overall']:.0f} % of attempted"],
        ["valid estimates, operating subgroup", f"{suc['operating']}  ({_pct(suc['fraction_operating'])}, CI {_ci(suc['ci_operating'])})",
         f">= {100 * R['success_fraction_operating']:.0f} % of attempted"],
        ["dropped without a reason code", str(res["dropped_without_reason"]), "= 0"],
    ]
    if res.get("usable") is not None:
        u = res["usable"]
        rows += [["usable measurements, overall (of attempted)", f"{u['overall']}  ({_pct(u['fraction_of_attempted'])})",
                  f">= {R['bleach_usable_min']:.0f}"],
                 ["usable measurements, operating subgroup", str(u["operating"]), f">= {R['bleach_usable_operating_min']:.0f}"],
                 ["valid but uninformative (rejected by the diagnostic)", str(u["rejected_valid"]), "reported"]]
    else:
        rows += [["scored for accuracy, overall", str(sc["overall"]), f">= {sc['required']['overall']:.0f}"],
                 ["scored for accuracy, operating subgroup", str(sc["operating"]), f">= {sc['required']['operating']:.0f}"]]
    reporter.table("Evidence and operational success (steps 1-3)", ["quantity", "observed", "rule"], rows,
                   note="A valid estimate is finite and, where a model is fitted, comes from a fit whose "
                        "optimizer reported success. Confidence intervals are Wilson 95 %.")
    if res["reasons"]:
        reporter.table("Dropped recordings by reason", ["reason code", "count"],
                       [[k, str(v)] for k, v in sorted(res["reasons"].items(), key=lambda kv: -kv[1])])

    def acc_rows(block: dict, label: str) -> list:
        out = []
        for crit, (obs, thr, passed) in block["criteria"].items():
            verdict = "-" if passed is None else ("meets" if passed else "misses")
            out.append([label, crit, _fmt(obs), thr, verdict, str(block["n"])])
        return out

    rows = acc_rows(res["accuracy"]["overall"], "overall") + acc_rows(res["accuracy"]["operating"], "operating subgroup")
    if res["accuracy"].get("rejected") and res["accuracy"]["rejected"]["n"]:
        rows += acc_rows(res["accuracy"]["rejected"], "rejected by diagnostic (reported)")
    reporter.table(f"Accuracy (step 4a) — {estimator}", ["stratum", "criterion", "observed", "threshold", "against threshold", "n"],
                   rows, note="Required overall and in the operating subgroup. 'meets'/'misses' are per criterion; the "
                              "step verdict above combines them. Correlation is in log10 coordinates for mu_r and "
                              "lambda_rate, linear for sigma_r; sigma_r MAE is in LINEAR units.")

    qrows = []
    for label, block in res["accuracy"]["quartiles"].items():
        cells = [label, str(block["n_attempted"]), str(block["n_valid"]), str(block["n"])]
        if block["n"] and block["reported"]:
            cells.append("; ".join(f"{c} {_fmt(o)}" for c, (o, _, _) in block["criteria"].items()))
            cells.append("reported")
        elif block["n"]:
            cells.append("; ".join(f"{c} {_fmt(o)}" for c, (o, _, _) in block["criteria"].items()))
            cells.append(f"< {R['quartile_min']:.0f} scored: no verdict")
        else:
            cells += ["-", "none scored"]
        qrows.append(cells)
    reporter.table("Accuracy by prior-fixed quartile (reported, no verdict)",
                   ["quartile (true value, log10)", "attempted", "valid", "scored", "criteria", "status"], qrows,
                   note="Localizes where an estimator degrades. A quartile with fewer than the frozen minimum of "
                        "scored recordings is shown without a verdict.")

    u = res.get("uncertainty")
    if u:
        rows = []
        for name, r in u["ranges"].items():
            wr = r["width_relative_to_prior"]
            rows += [[name, f"coverage of the nominal {100 * u['nominal']:.0f} % range, overall",
                      f"{_pct(r['coverage_overall'])}  (CI {_ci(r['ci_overall'])}, n={r['n']})", f">= {100 * R['coverage_min']:.0f} %"],
                     [name, "coverage, operating subgroup",
                      f"{_pct(r['coverage_operating'])}  (CI {_ci(r['ci_operating'])}, n={r['n_operating']})", f">= {100 * R['coverage_min']:.0f} %"],
                     [name, "median range width / prior width (same coordinates)",
                      "n/a" if wr is None else f"{wr['median']:.3f}  (p10 {wr['p10']:.3f}, p90 {wr['p90']:.3f})", "reported"],
                     [name, "scored recordings without a finite range", str(r["n_without_range"]), "reported"]]
        reporter.table(
            "Uncertainty (step 4b): reliability and informativeness", ["range", "quantity", "observed", "rule"], rows,
            note="Coverage establishes uncertainty reliability; interval width describes informativeness. A broad "
                 "range can cover adequately without measuring anything: returning nearly the whole prior is not "
                 "successful, precise recovery, however well it covers.")
