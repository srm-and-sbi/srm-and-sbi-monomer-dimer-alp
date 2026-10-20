"""Shared Estimator-Comparison diagnostic engine for both DIMER workflows.

``run_estimator_comparison(cfg, args)`` decides whether one trained estimator generalizes
better than another by the paired log-score on the shared ``(task, sim)`` subset of the
held-out TEST set (Diebold-Mariano + Wilcoxon + paired bootstrap; the ``estimator_comparison``
kernel). The two entry-point shims shrink to: build the ``WorkflowConfig``, parse args,
call ``run_estimator_comparison`` -- the same pattern as the pipeline stages.

The inputs are two Test-Loss-Distribution artifacts (``TestLossDistribution``), which store
the per-video loss keyed by ``(task, sim)``; these are produced by both workflows, so the
diagnostic is workflow-agnostic and the only per-workflow difference is the alias-qualified
``paths`` namespace the artifacts live under (biology's or detector's). No GPU, no sampling:
it reads two finished artifacts and runs the statistics.

Two log-scores compare only within one parameterization: before the kernel runs,
``_assert_same_parameterization`` refuses a pair whose learnable keys (and order), per-row
scale flags, prior boxes or condition differ. What the guard does not establish: pairing by
``(task, sim)`` assumes both estimators scored the same TEST examples, which the manifest's
``test_set_id`` (``TEST/<timing>``) does not prove; shared-example identity is established
separately (same tier path and timing).

An Analysis diagnostic -- read-only, never wired into the ``Submit.sh`` dispatcher -- whose
report lands in the ``Posit/`` tier alongside the estimators it compares.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp import estimator_comparison as ecomp
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_monomer_dimer_alp.test_loss_distribution import TestLossDistribution
from srm_and_sbi_monomer_dimer_alp.workflow import WorkflowConfig


def _comparison_dir(paths, data_bank_root: Path, timing_label: str) -> Path:
    """Directory holding the comparison report + figure + arrays (Posit tier)."""
    return (data_bank_root / paths.posit_subdir /
            f"{paths.project_alias}_{timing_label}_Estimator_Comparison")


def _resolve_tld(spec: str, paths, data_bank_root: Path, timing_label: str):
    """Resolve one estimator spec to a Test-Loss-Distribution path + a short label.

    ``spec`` is one of: ``"canonical"`` (the current best-epoch TLD), a loss tag such as
    ``"-6.91"`` (globs the workflow's ``Posit/`` for the matching provenance backup), or an
    explicit ``.npz`` path.
    """
    if spec == "canonical":
        return paths.test_loss_distribution_path(data_bank_root, timing_label), "canonical"
    p = Path(spec)
    if p.suffix == ".npz":
        label = p.stem.split("TEST_LOSS_")[-1] if "TEST_LOSS_" in p.stem else p.stem
        return p, label
    posit = data_bank_root / paths.posit_subdir
    matches = sorted(posit.glob(
        f"{paths.project_alias}_{timing_label}_Test_Loss_Distribution_*TEST_LOSS_{spec}.npz"))
    if not matches:
        raise SystemExit(
            f"estimator comparison: no TLD backup matching loss tag '{spec}' in {posit} "
            f"(use 'canonical', a loss tag like -6.91, or an explicit .npz path)")
    return matches[-1], spec


def _learnable_rows(manifest):
    """The learnable rows of a TLD manifest's parameter table, keyed by ``key``."""
    return [e for e in manifest.get("parameter_table", []) if e.get("role") == "learnable"]


def _assert_same_parameterization(manifest_a, manifest_b, *, label_a="A", label_b="B") -> None:
    """Refuse two test-loss manifests that are not one parameterization: the learnable keys
    (and their order), the per-row scale flags (``theta_log_flag``, else the table's
    ``log_flag``), the prior boxes of the learnable rows, and the condition must agree. The
    refusal (``ValueError``) names the first difference. It does not establish that both
    estimators scored the same TEST examples (see the module docstring)."""
    def keys(m):
        return [str(k) for k in m.get("theta_keys", [])]

    def flags(m):
        if m.get("theta_log_flag") is not None:
            return [bool(v) for v in m["theta_log_flag"]]
        return [bool(e.get("log_flag")) for e in _learnable_rows(m)]

    def boxes(m):
        return [tuple(float(v) for v in e.get("prior_range") or ()) for e in _learnable_rows(m)]

    ka, kb = keys(manifest_a), keys(manifest_b)
    if len(ka) != len(kb):
        raise ValueError(f"estimator comparison: {label_a} has {len(ka)} learnable keys and {label_b} "
                         f"{len(kb)} ({ka} vs {kb}); two parameterizations are not compared.")
    for i, (x, y) in enumerate(zip(ka, kb)):
        if x != y:
            raise ValueError(f"estimator comparison: learnable key at index {i} differs: {label_a} "
                             f"{x!r} vs {label_b} {y!r}; two parameterizations are not compared.")
    fa, fb = flags(manifest_a), flags(manifest_b)
    for i, (x, y) in enumerate(zip(fa, fb)):
        if x != y:
            raise ValueError(f"estimator comparison: the scale of {ka[i]!r} differs: {label_a} "
                             f"{'log' if x else 'linear'} vs {label_b} {'log' if y else 'linear'}; "
                             f"the densities are over different spaces.")
    if len(fa) != len(fb):
        raise ValueError(f"estimator comparison: {label_a} records {len(fa)} scale flags and {label_b} "
                         f"{len(fb)}.")
    ba, bb = boxes(manifest_a), boxes(manifest_b)
    if ba != bb:
        for i, (x, y) in enumerate(zip(ba, bb)):
            if x != y:
                raise ValueError(f"estimator comparison: the prior box of {ka[i]!r} differs: {label_a} "
                                 f"{x} vs {label_b} {y}; the prior volumes differ.")
        raise ValueError(f"estimator comparison: the learnable prior boxes differ ({len(ba)} vs {len(bb)} rows).")
    ca, cb = manifest_a.get("condition"), manifest_b.get("condition")
    if ca != cb:
        raise ValueError(f"estimator comparison: the condition differs: {label_a} {ca!r} vs {label_b} {cb!r}.")


def _figure_improvement(result):
    """Histogram of the per-video improvement (loss_B - loss_A) with mean, CI, and zero."""
    from matplotlib.figure import Figure

    BLUE, ORANGE, REF, GRID, INK, GOOD = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#52514e", "#0ca30c"
    d = np.asarray(result.improvement)
    # Clip the heavy tail for display only (the stats used the full data).
    lo, hi = np.quantile(d, [0.005, 0.995])
    span = max(hi - lo, 1e-9)
    fig = Figure(figsize=(6.4, 4.2), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(REF)
    ax.tick_params(colors=INK, labelsize=8)
    ax.hist(np.clip(d, lo - 0.02 * span, hi + 0.02 * span), bins=60, color=BLUE, zorder=3)
    ax.axvline(0.0, color=REF, linewidth=1.2, linestyle="--", zorder=4, label="no difference")
    ax.axvline(result.mean_improvement, color=ORANGE, linewidth=2.0, zorder=5,
               label=f"mean {result.mean_improvement:+.4f}")
    ax.axvspan(result.boot_ci_low, result.boot_ci_high, color=ORANGE, alpha=0.15, zorder=1,
               label=f"{int((1-result.alpha)*100)}% CI")
    ax.set_xlabel(f"per-video improvement  (loss_{result.label_b} - loss_{result.label_a};"
                  f"  > 0 => {result.label_a} better)", fontsize=9, color=INK)
    ax.set_ylabel("videos", fontsize=9, color=INK)
    ax.set_title(f"{result.label_a}  vs  {result.label_b}   —   {result.verdict}",
                 fontsize=10, color=(GOOD if "better" in result.verdict else INK))
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def run_estimator_comparison(cfg: WorkflowConfig, args: argparse.Namespace) -> None:
    """Run the full paired estimator comparison for the given workflow + CLI args."""
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = cfg.paths.with_condition(args.condition)   # condition-specific namespace
    timing_label = timing.label
    comp_dir = _comparison_dir(paths, data_bank_root, timing_label)

    path_a, label_a = _resolve_tld(args.a, paths, data_bank_root, timing_label)
    path_b, label_b = _resolve_tld(args.b, paths, data_bank_root, timing_label)

    machine = PARAMETERS.machine
    div = "=" * 72
    print(div)
    print(f" {paths.project_alias} — Estimator Comparison (paired log-score)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print(f"\nMachine profile   : {machine.name}")
    print("\nRun configuration:")
    print(f"  --total-time-seconds : {args.total_time_seconds}")
    print(f"  estimator A ({label_a}) : {path_a}")
    print(f"  estimator B ({label_b}) : {path_b}")
    print(f"  --n-boot / --seed / --alpha : {args.n_boot} / {args.seed} / {args.alpha}")
    print(f"  writes report   : {comp_dir}")
    print(f"\n{div}\n")

    if args.dry_run:
        missing = 0
        for role, path in [("estimator A TLD", path_a), ("estimator B TLD", path_b)]:
            ok = Path(path).exists()
            missing += 0 if ok else 1
            print(f"  reads {role}: {path}  [{'OK' if ok else 'MISSING'}]")
        print(f"\n[DRY RUN] configuration validated; "
              f"{'all inputs present' if not missing else f'{missing} input(s) MISSING'}.")
        print("[DRY RUN] no comparison performed.")
        return

    run_start = time.time()
    reporter = DiagnosticReporter(
        stage="Estimator_Comparison", enabled=True, dump=True, dump_dir=comp_dir,
        run_label=f"{paths.project_alias}_{timing_label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    reporter.check_file("estimator A TLD", path_a)
    reporter.check_file("estimator B TLD", path_b)

    tld_a = TestLossDistribution.load(path_a)
    tld_b = TestLossDistribution.load(path_b)
    try:
        _assert_same_parameterization(tld_a.manifest, tld_b.manifest, label_a=label_a, label_b=label_b)
    except ValueError as exc:
        raise SystemExit(str(exc))
    reporter.check("same_parameterization", True,
                   f"{len(tld_a.manifest.get('theta_keys', []))} learnable keys, scales, boxes and "
                   f"condition {tld_a.manifest.get('condition')!r} agree",
                   note="log-scores compare only within one parameterization; the guard compares "
                        "keys and order, per-row scale flags, prior boxes and the condition.")
    result = ecomp.compare(
        tld_a.keys, tld_a.loss, tld_b.keys, tld_b.loss,
        label_a=label_a, label_b=label_b,
        n_boot=args.n_boot, seed=(args.seed or 0), alpha=args.alpha,
    )

    # ---- Save the paired improvement (reproducible re-analysis) ----------
    comp_dir.mkdir(parents=True, exist_ok=True)
    array_path = comp_dir / f"{paths.project_alias}_{timing_label}_Estimator_Comparison.npz"
    np.savez_compressed(str(array_path), improvement=result.improvement,
                        label_a=np.asarray(label_a), label_b=np.asarray(label_b))
    print(f"\nComparison arrays saved to {array_path}")

    # ---- Report ----------------------------------------------------------
    reporter.check("shared_nonempty", result.n_shared > 0,
                   f"{result.n_shared} shared (task, sim) videos",
                   note="the two TEST-loss sets share at least one keyed video (the paired sample).")
    reporter.stat("estimator_A", f"{label_a}  (n={result.n_a})")
    reporter.stat("estimator_B", f"{label_b}  (n={result.n_b})")
    reporter.stat("shared_videos", result.n_shared,
                  note="videos present in BOTH sets; the comparison is paired over exactly these.")
    headers = ["metric", "value"]
    rows = [[r["metric"], r["value"]] for r in ecomp.summarize(result)]
    reporter.table(
        "Paired log-score comparison", headers, rows,
        note="improvement = loss_B - loss_A per shared video; > 0 means A has the lower loss. "
             "Pairing cancels the intrinsic per-video entropy floor, so the difference isolates "
             "the two estimators' KL gap (Amisano & Giacomini 2007). Diebold-Mariano is the "
             "mean-based test; Wilcoxon + bootstrap are the heavy-tail-robust companions; the "
             "verdict follows the Diebold-Mariano p-value at alpha. Pairing by (task, sim) "
             "assumes both estimators scored the same TEST examples; the manifest's test_set_id "
             f"({tld_a.manifest.get('test_set_id')!r} / {tld_b.manifest.get('test_set_id')!r}) "
             "names the split and timing only and does not prove it -- shared-example identity "
             "is established separately (same tier path and timing).")

    if reporter.dump:
        reporter.save_figure(
            "improvement_distribution", _figure_improvement(result),
            caption=f"Per-video paired improvement (loss_{label_b} - loss_{label_a}). Mass to "
                    f"the right of zero favors {label_a}; the orange line is the mean and the "
                    "band its bootstrap CI. The distribution is centered by pairing, so its "
                    "location (not the raw loss scale) is the signal.")

    reporter.summary()
    reporter.write_report()
    print(f"\nVerdict: {result.verdict}. Total elapsed: {time.time() - run_start:.1f}s")


def build_estimator_comparison_parser() -> argparse.ArgumentParser:
    """Construct the Estimator-Comparison CLI parser (shared by both workflow shims)."""
    parser = argparse.ArgumentParser(
        description="Compare two trained estimators by the paired log-score on the shared "
                    "(task, sim) TEST subset (Diebold-Mariano / Wilcoxon / bootstrap).")
    parser.add_argument(
        "--condition", required=True, choices=LABELING_CONDITIONS,
        help="Experimental condition of the run (FAB = MET-FAB, INLB = MET-INLB): selects the "
             "condition-specific data and estimator namespace (the condition slot of the "
             "runtime grammar).",
    )
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
        help="Video duration in seconds; selects the timing_label namespace (e.g. 2.0 -> 2S_50FPS).")
    parser.add_argument(
        "--a", type=str, required=True,
        help="Estimator A: 'canonical' (current best-epoch TLD), a loss tag like -6.91 "
             "(globs the matching provenance backup), or an explicit .npz path.")
    parser.add_argument(
        "--b", type=str, required=True,
        help="Estimator B: same forms as --a.")
    parser.add_argument(
        "--n-boot", type=int, default=10000,
        help="Paired-bootstrap resamples for the mean-improvement CI (default: 10000).")
    parser.add_argument(
        "--seed", type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v),
        default=0, help="Bootstrap RNG seed (default: 0).")
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration + resolve/verify the two TLD inputs, then exit without comparing.")
    return parser
