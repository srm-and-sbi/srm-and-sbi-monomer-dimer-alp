"""Shared Evaluation-stage engine for both DIMER workflows (biology + detector).

``run_evaluation(cfg, args)`` holds the entire MAP-recovery orchestration -- banner,
dry-run probe, estimator load, the per-video seed-then-optimize MAP loop, the live
progress log, multi-GPU round-robin video sharding, the ``--merge`` combine step, and
the recovery report + figures. The two entry-point scripts shrink to: build the
``WorkflowConfig``, parse args, call ``run_evaluation``.

The numeric engine already lives in ``evaluation`` (``map_estimate``,
``recovery_table``, ``posterior_summary``, ``posterior_coverage_table``), shared by
both workflows. The only per-workflow differences are which parameterization supplies
the learnable table (for the recovery/coverage tables + per-parameter figures), the
``parameter_keys`` schema guard, the device ``build_prior``, and the alias-qualified
``paths`` -- resolved from the config in ``_evaluation_spec(cfg)``. The estimator path
is resolved through the shared ``paths.estimator_path`` method for both workflows.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch._dynamo

from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp import artifacts
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from srm_and_sbi_monomer_dimer_alp.evaluation import (
    ESTIMATE_DEFINITIONS_VERSION, QUANTILE_LEVELS,
    point_estimates_compared_table,
    band_label,
    draw_label,
    map_estimate,
    optimizer_contract,
    optimizer_summary_lines,
    point_estimate_rows,
    posterior_coverage_table,
    STOP_REASONS,
    posterior_summary,
    recovery_table,
    short_labels,
    _theta_repr,
    point_estimate_agreement_table, prior_scale,
)
from srm_and_sbi_monomer_dimer_alp.experiment_support import (
    RetiredOption, assert_complete_shard_set, merge_validated_shards, save_shard, shard_by_rank)
from srm_and_sbi_monomer_dimer_alp.provenance import code_provenance, finalize_code_provenance
from srm_and_sbi_monomer_dimer_alp.inference_support import resolve_topology
from srm_and_sbi_monomer_dimer_alp.io import load_data, load_theta_set, theta_set_status
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming, to_flow, to_physical
from srm_and_sbi_monomer_dimer_alp.utils import console_log_context  # noqa: F401  (entry points import via this module's siblings)
from srm_and_sbi_monomer_dimer_alp.visualization_inference import (
    figure_recovery_combined, figure_point_estimates_vs_truth)
from srm_and_sbi_monomer_dimer_alp.workflow import WorkflowConfig


@dataclass(frozen=True)
class _EvaluationSpec:
    """Per-workflow Evaluation specializations resolved from a ``WorkflowConfig``."""
    draw_spec: list          # learnable table for the recovery/coverage tables + figures
    parameter_keys: list     # load_estimator schema guard
    build_prior: Callable    # device prior rebuild for bounded rejection sampling


def _evaluation_spec(cfg: WorkflowConfig) -> _EvaluationSpec:
    """Resolve the Evaluation stage's per-workflow specializations from the workflow config."""
    m = cfg.param_module
    if cfg.tag == "detector":
        return _EvaluationSpec(
            draw_spec=m.DETECTOR_PARAMETERIZATION,
            parameter_keys=[e["KEY"] for e in m.DETECTOR_PARAMETERIZATION],
            build_prior=m.build_prior,
        )
    return _EvaluationSpec(
        draw_spec=m.PARAMETERIZATION,
        parameter_keys=m.PARAMETER_KEYS,
        build_prior=m.build_prior,
    )


def write_recovery_outputs(reporter, args, eval_cfg, draw_spec, recovery_array_path: Path,
                           arrays: dict, run_start, persist_arrays: bool = True) -> dict:
    """Validate a recovery product, persist it and write its report + figures.

    Shared by the single-process path and the ``--merge`` step so both emit an identical
    report. ``arrays`` is the complete product (every field of
    ``artifact_schema.STAGE_FIELDS["evaluation"]`` plus the encoded manifest). It is validated
    HERE, on entry, through :func:`artifact_schema.validate_product` -- the writer is the
    boundary, so no caller (and no independent report-generation code) can persist or render an
    invalid product -- and the manifest the report reads is decoded from the arrays themselves,
    never supplied separately. The three point estimates are read by their stored fields --
    ``map_estimate``, the 0.50 level of ``posterior_quantiles`` located through the manifest, and
    ``posterior_sgm`` -- so the report never re-derives or reinterprets a product with defaults
    of its own. With ``persist_arrays=False`` (report-only rendering) the arrays on disk are left
    untouched. Returns the validated manifest.
    """
    manifest = schema.validate_product(arrays, stage="evaluation",
                                       source=f"evaluation product {recovery_array_path.name}")
    bin_mode = args.bin_mode
    n_bins = args.n_bins or eval_cfg.quantile_bins
    min_count = args.min_count or eval_cfg.quantile_min_count
    tok = short_labels()                       # {"map": "MAP", "median": "median", "sgm": "SGM"}
    qi = schema.median_level_index(manifest)
    scores = np.asarray(arrays["scores"], dtype=float)
    map_est = np.asarray(arrays["map_estimate"], dtype=float)
    true_log10 = np.asarray(arrays["true_log10"], dtype=float)
    post_q = np.asarray(arrays["posterior_quantiles"], dtype=float)
    median = post_q[:, :, qi]
    sgm = np.asarray(arrays["posterior_sgm"], dtype=float)
    n_samples = map_est.shape[0]
    estimates = {tok["map"]: map_est, tok["median"]: median, tok["sgm"]: sgm}

    if persist_arrays:
        np.savez_compressed(str(recovery_array_path), **arrays)
        print(f"\nRecovery arrays saved to {recovery_array_path}")
    else:
        # Report-only rendering from an existing product: the arrays on disk are the record and
        # are left untouched (a re-save would also drop any field this writer does not know).
        print(f"\nRecovery arrays left as stored at {recovery_array_path} (report-only rendering)")

    # ---- Recovery report -------------------------------------------------
    guide = eval_cfg.error_guide                   # 0.3 ~= log10(2): within a factor of 2
    guide_tight = eval_cfg.error_guide_tight        # 0.15 ~= log10(sqrt(2)): within a factor of ~1.41
    reporter.check("eval_set_nonempty", n_samples > 0,
                   f"{n_samples} EVAL samples recovered",
                   note="the held-out EVAL namespace yielded at least one recovered video.")
    reporter.check("three_point_estimates_present", True,
                   "map_estimate, posterior_quantiles (median at level 0.50) and posterior_sgm "
                   "stored for every observation",
                   note="the product contract: no estimate is reported without the other two.")
    reporter.check_no_nan_inf("true_log10", true_log10)
    reporter.stat("eval_tasks", args.eval_tasks)
    reporter.stat("eval_samples", n_samples,
                  note="number of held-out videos whose parameters were recovered.")
    reporter.stat("artifact_schema", f"v{manifest['artifact_schema_version']} / estimate "
                  f"definitions v{manifest['estimate_definitions_version']}",
                  note="the stored computation contract; the manifest inside the .npz records the "
                       "optimizer settings, draw count, quantile levels, SGM scaling, code and "
                       "checkpoint identity this product was made under.")
    reporter.stat("summary_draws", manifest["n_summary_draws"],
                  note=f"{manifest['draw_label']}s per video that the median and the SGM summarize "
                       f"(pool mode {manifest['pool_mode']}).")
    reporter.stat("mean_log_prob", float(np.mean(scores)),
                  note="mean log-density at the returned MAP candidate -- the objective the "
                       "seed-then-optimize step maximizes (larger = sharper peak). Computed in "
                       "the estimator's z-scored space, so its absolute scale is a relative "
                       "optimization diagnostic, not a calibration/quality metric; the "
                       "per-parameter recovery tables below are the quality measure.")
    if n_samples < eval_cfg.quantile_min_count:
        reporter.stat(
            "quantile_bands", "sparse",
            note=f"fewer than {eval_cfg.quantile_min_count} samples per bin: the "
                 "report shows the scatter and the error table; conditional "
                 "quantile bands populate only with a larger EVAL set.")

    reporter.table(
        "Point estimates: definitions", ["key", "stored field", "definition"],
        point_estimate_rows(manifest["pool_mode"], qi),
        note="one row per point estimate (evaluation.POINT_ESTIMATES); every table and figure "
             "below uses these three tokens. The three are read together; no one replaces the "
             "others.")

    band_note = (f"The two 'within' columns are the fractions of EVAL videos recovered inside each "
                 f"nested tolerance band, stated as the multiplicative range the band permits: "
                 f"{band_label(guide)} is +/-{guide:g} in log10 (a factor of two) and "
                 f"{band_label(guide_tight)} is +/-{guide_tight:g} (a factor of the square root of "
                 f"two). 'outside prior' is the share beyond the row's prior box. 'corr(inf, true)' "
                 f"is the correlation between inferred and true values -- near zero when the "
                 f"estimator's output does not depend on its input.")
    for key, arr in estimates.items():
        headers, rows = recovery_table(draw_spec, true_log10, arr, guide, guide_tight)
        extra = ("" if key != tok["map"] else
                 " The gradient ascent is unconstrained, so an outside-prior value happens under "
                 "either pool mode -- --pool-mode bounds the candidate pool, not the optimizer's "
                 "steps -- and is a flow optimum rather than a MAP of the prior-supported posterior.")
        reporter.table(
            f"{key} recovery (per parameter, log10 units)", headers, rows,
            note=f"error = inferred - true in log10 units for the {key} estimate. {band_note}{extra} "
                 f"The three recovery tables are read together.")

    cov_headers, cov_rows = posterior_coverage_table(draw_spec, true_log10, post_q)
    reporter.table(
        "Posterior calibration (per parameter)", cov_headers, cov_rows,
        note="fraction of truths inside the per-video credible intervals of the draws; a "
             "calibrated posterior covers ~50% (IQR) and ~90%.")
    cmp_headers, cmp_rows = point_estimates_compared_table(draw_spec, true_log10, estimates)
    reporter.table(
        "Point estimates compared (per parameter, log10 units)", cmp_headers, cmp_rows,
        note=f"the three point estimates on the same videos, columns grouped by statistic with the "
             f"estimates consecutive: correlation with the truth, MAE, signed bias (mean of "
             f"inferred - true) and the share outside the prior box, each for {tok['map']}, "
             f"{tok['median']} and {tok['sgm']} (definitions above). A conclusion about a parameter "
             f"is drawn from the row as a whole, never from one column.")
    agr_headers, agr_rows = point_estimate_agreement_table(draw_spec, map_est, post_q, sgm)
    reporter.table(
        "Point-estimate agreement (per parameter, log10 units)", agr_headers, agr_rows,
        note="each 'X vs Y' column is the median over videos of the absolute difference between "
             "the two point estimates (log10). 'MAP outside 90%' is the share of videos whose MAP "
             "falls outside the central 90% interval of the draws. Large gaps with a high outside "
             "share establish that the optimized candidate and the draw summaries disagree; the "
             "cause is not decided by this table and needs separate checks.")

    if reporter.dump:
        for i, para in enumerate(draw_spec):
            key = para["KEY"]
            label = para.get("LABEL") or key
            prior_range = para["PRIOR_RANGE"]
            reporter.save_figure(
                f"recovery_{key}",
                figure_recovery_combined(
                    true_log10[:, i], map_est[:, i], post_q[:, i, :], qi,
                    prior_range, label, n_bins=n_bins, min_count=min_count,
                    error_guide=guide, error_guide_tight=guide_tight,
                    error_ylim_floor=eval_cfg.error_ylim_floor,
                    error_ylim_quantile=eval_cfg.error_ylim_quantile, bin_mode=bin_mode),
                caption=f"{key} ({label}). Panels 1-2: the {tok['map']} against the truth and its "
                        f"residual error, with {bin_mode}-binned conditional-quantile bands (drawn "
                        f"where a bin has >= {min_count} points). Panel 3: the truth against the "
                        f"{tok['median']} with IQR bars (per-video spread of the draws), the "
                        f"{tok['map']} overlaid.",
            )
            fig = figure_point_estimates_vs_truth(
                true_log10[:, i], {k: v[:, i] for k, v in estimates.items()},
                post_q[:, i, :], prior_range, label, n_bins=n_bins, min_count=min_count)
            if fig is not None:
                reporter.save_figure(
                    f"point_estimates_{key}", fig,
                    caption=f"{key} ({label}). The three point estimates against the truth, one "
                            f"panel each ({tok['map']}, {tok['median']}, {tok['sgm']}): grey density "
                            f"of all videos, the estimate's median over equal-count bins of the truth "
                            f"as the line, and the IQR of the draws (median over the bin of the "
                            f"per-video Q25 and Q75) as the band, drawn once and identical on every "
                            f"panel because the three come from the same draws. The estimate axis is "
                            f"clipped to the prior box widened by 40 %; values beyond it are counted "
                            f"in the tables, not drawn.")

    reporter.summary()
    reporter.write_report()
    print(f"\nTotal elapsed: {time.time() - run_start:.1f}s")
    return manifest


def _save_shard(topo, recovery_dir: Path, arrays: dict, run_start) -> None:
    """Write this worker's validated partial product (multi-GPU sharded run). The report is
    produced by the ``--merge`` step once every shard exists. A worker that drew no videos
    still writes a valid zero-observation shard, so the merge's rank-coverage check sees every
    rank account for itself."""
    n = int(np.asarray(arrays["scores"]).shape[0])
    path = save_shard(recovery_dir, topo, arrays, count=n, write_empty=True)
    print(f"\n[rank {topo.rank}/{topo.world_size}] shard saved: {path} "
          f"({n} videos{'; no videos were assigned to this rank' if n == 0 else ''}) in "
          f"{time.time() - run_start:.1f}s. Run the --merge step once all shards finish.",
          flush=True)


def _merge_shards(reporter, args, eval_cfg, draw_spec, recovery_dir: Path,
                  recovery_array_path: Path, run_start, expected_ids) -> None:
    """Combine every per-shard product into the final report (no recovery).

    Every shard is validated, the shard manifests must describe one computation, the merged
    ``(task_index, sim_index)`` set must equal ``expected_ids`` exactly (no duplicates, no
    missing, no extra), and rank coverage must be complete. Then the merged product is written
    through :func:`write_recovery_outputs` and the shard files are removed.
    """
    shard_paths = sorted(recovery_dir.glob("_shard_*_of_*.npz"))
    if not shard_paths:
        raise SystemExit(
            f"--merge: no shard files (_shard_*_of_*.npz) found in {recovery_dir}")
    try:
        world_size = assert_complete_shard_set(shard_paths, partial_option=None)
        merged, _manifest, n_used = merge_validated_shards(
            shard_paths, stage="evaluation",
            concat_keys=["scores", "map_estimate", "true_log10", "posterior_quantiles",
                         "posterior_sgm", "task_index", "sim_index"],
            expected_ids=expected_ids)
    except (ValueError, schema.SchemaError) as exc:
        raise SystemExit(f"--merge: {exc}")
    reporter.stat("shards_merged", f"{n_used}/{world_size}",
                  note="per-rank shards combined into this report; the merge requires every rank "
                       "and every expected (task, sim) observation, so this is always complete.")
    print(f"Merged {merged['scores'].shape[0]} videos from {n_used} shard(s).", flush=True)
    write_recovery_outputs(reporter, args, eval_cfg, draw_spec, recovery_array_path,
                           merged, run_start)
    for shard_path in shard_paths:
        shard_path.unlink()
    print(f"Removed {len(shard_paths)} shard file(s).", flush=True)


def _expected_observations(paths, args, data_bank_root, timing_label, compress, draw_spec):
    """The EVAL inventory this run must cover: ``[(task, sim), ...]`` over every EVAL task and
    every simulation it holds (capped by ``--max-sims``). Needs only the theta sets, so the
    ``--merge`` step can compute it without a GPU."""
    per_task_sims = {}
    for task in range(args.eval_tasks):
        theta_set = load_theta_set(
            paths.theta_set_path(task, data_bank_root, timing_label, compress, "EVAL"),
            draw_spec, condition=args.condition)
        n_sims = theta_set.shape[0]
        per_task_sims[task] = min(n_sims, args.max_sims) if args.max_sims > 0 else n_sims
    return [(t, s) for t in range(args.eval_tasks) for s in range(per_task_sims[t])]


def run_evaluation(cfg: WorkflowConfig, args: argparse.Namespace) -> None:
    """Run the full MAP-recovery evaluation for the given workflow + CLI args."""
    spec = _evaluation_spec(cfg)

    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    data_bank_root = PARAMETERS.machine.data_bank_root
    compress = True  # EVAL video/theta sets are read from .zarr, as in inference
    paths = cfg.paths.with_condition(args.condition)   # condition-specific namespace
    eval_cfg = PARAMETERS.inference.evaluation

    # ---- Global RNG / precision settings ---------------------------------
    if args.seed is not None:   # None -> non-deterministic (consistent with generation)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    timing_label = timing.label
    # The estimator and every product derived from it carry the optional artifact tag
    # (Paths.product_label); the EVAL inputs never do.
    product_label = paths.product_label(timing_label, args.artifact_tag)
    estimator_path = paths.estimator_path(data_bank_root, product_label)
    recovery_dir = paths.map_recovery_dir(data_bank_root, product_label)
    recovery_array_path = paths.map_recovery_array_path(data_bank_root, product_label)

    # Resolve effective hyperparameters (None -> config default).
    lr = args.learning_rate if args.learning_rate is not None else eval_cfg.learning_rate
    tolerance = eval_cfg.tolerance
    theta_prex_size = args.theta_prex_size or eval_cfg.theta_prex_size
    elite_prex_size = args.elite_prex_size or eval_cfg.elite_prex_size
    numb_steps = args.numb_steps or eval_cfg.numb_steps
    show_progress_steps = args.show_progress_steps or eval_cfg.show_progress_steps
    pool_mode = args.pool_mode or eval_cfg.pool_mode
    # The settings the banner prints are the settings the manifest records: one object for both.
    opt_contract = optimizer_contract(eval_cfg, learning_rate=lr, tolerance=tolerance,
                                      theta_prex_size=theta_prex_size,
                                      elite_prex_size=elite_prex_size, numb_steps=numb_steps,
                                      pool_mode=pool_mode)

    # Verbosity tiers (the show / verbose flags).
    show = args.verbose or args.debug or args.debug_dump
    verbose_deep = args.debug or args.debug_dump
    debug_log = args.debug or args.debug_dump   # write per-step/theta detail to progress.log

    bin_mode = args.bin_mode
    posterior_samples = args.posterior_samples or eval_cfg.posterior_samples

    # ---- Pre-run banner --------------------------------------------------
    machine = PARAMETERS.machine
    div = "=" * 72
    print(div)
    print(f" {paths.project_alias} — Evaluation (MAP recovery)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nMachine profile:")
    print(f"  name              : {machine.name}")
    print(f"  compute_backend   : {machine.compute_backend}")
    print("\nRun configuration (CLI args):")
    print(f"  --total-time-seconds : {args.total_time_seconds}")
    print(f"  --eval-tasks         : {args.eval_tasks}        (EVAL-namespace tasks; held out)")
    print(f"  --max-sims           : {args.max_sims}        (per task; 0 = all)")
    print(f"  --seed               : {args.seed}")
    print(f"  verbosity            : "
          f"{'debug-dump' if args.debug_dump else ('debug' if args.debug else ('verbose' if args.verbose else 'normal'))}")
    print("\nMAP estimate hyperparameters (effective):")
    print(f"  pool_mode            : {pool_mode}   "
          f"({'rejection within prior' if pool_mode == 'bounded' else 'flow direct, no rejection'})")
    for line in optimizer_summary_lines(opt_contract):
        print(f"  {line}")
    print(f"  point estimates      : map, median, sgm  (all three, always; "
          f"{draw_label(pool_mode)}s summarized)")
    print(f"  --bin-mode           : {bin_mode}")
    print(f"  --posterior-samples  : {posterior_samples}")
    progress_path = recovery_dir / "progress.log"
    print("\nOutput destinations:")
    print(f"  reads estimator : {estimator_path}")
    print(f"  reads EVAL      : <data_bank>/{paths.video_subdir}/"
          f"{paths.project_alias}_{timing_label}_Video_Set_TASK_{{0..{args.eval_tasks - 1}}}_EVAL.zarr")
    print(f"  writes report   : {recovery_dir}")
    print(f"  live progress   : {progress_path}   (tail -f to monitor)")
    print(f"\n{div}\n")

    # ---- Dry run: validate inputs and exit before the reporter, GPU, posterior,
    # or topology -- so it creates no output directory and needs no torchrun.
    if args.dry_run:
        eval_video_path = paths.video_set_path(0, data_bank_root, timing_label, compress, "EVAL")
        eval_theta_path = paths.theta_set_path(0, data_bank_root, timing_label, compress, "EVAL")
        inputs = [
            ("estimator artifact", estimator_path),
            ("EVAL video set (task 0)", eval_video_path),
            ("EVAL theta set (task 0)", eval_theta_path),
        ]
        missing = 0
        for role, path in inputs:
            status = (theta_set_status(path, spec.draw_spec, condition=args.condition) if "theta" in role
                      else ("OK" if Path(path).exists() else "MISSING"))
            if status != "OK":
                missing += 1
            print(f"  reads {role}: {path}  [{status}]")
        if missing:
            print(f"\n[DRY RUN] configuration validated; {missing} input(s) MISSING.")
        else:
            print(f"\n[DRY RUN] configuration validated; all inputs present.")
        print("[DRY RUN] no MAP-recovery evaluation performed.")
        return

    run_start = time.time()

    # ---- Diagnostics reporter (the recovery report is the deliverable) ----
    reporter = DiagnosticReporter(
        stage="Evaluation", enabled=True, dump=True, dump_dir=recovery_dir,
        run_label=f"{paths.project_alias}_{product_label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    # --merge: combine the per-shard arrays from a multi-GPU sharded run into the
    # final report, then exit (no recovery work, no GPU, no posterior needed).
    if args.merge:
        expected = _expected_observations(paths, args, data_bank_root, timing_label, compress,
                                          spec.draw_spec)
        _merge_shards(reporter, args, eval_cfg, spec.draw_spec, recovery_dir,
                      recovery_array_path, run_start, expected)
        return

    reporter.check_file("estimator artifact", estimator_path)

    topo = resolve_topology()
    device = topo.device
    vista_device = torch.device("cpu")
    posterior = artifacts.load_estimator(estimator_path, device=str(device),
                                          expected_parameter_keys=spec.parameter_keys)
    posterior.posterior_estimator.to(device)
    if device.type == "cuda":
        # Rebuild the prior on THIS worker's device for bounded rejection sampling.
        posterior.prior = spec.build_prior(device=str(device))

    # ---- Provenance captured at STARTUP: the code as loaded, the checkpoint actually loaded
    # (its verified weights checksum, not the path re-read later), the invocation identity the
    # launcher assigned and this rank's execution attempt. The manifest is built from these, and
    # the implementation records are compared again at write time: a file edited during the run
    # makes the product fail closed at publication (it is not written).
    code_at_start = code_provenance()
    checkpoint_sha256 = posterior.weights_sha256
    run_ident = schema.run_identity(product_label, distributed=topo.is_distributed)
    exec_ident = schema.execution_identity()   # this attempt (its Slurm job, or None): recorded, never compared

    # ---- Probe the EVAL namespace, then take this worker's VIDEO shard ----
    # Sharding is at video granularity, not task granularity. Splitting whole tasks across
    # ranks leaves them with unequal task counts whenever the worker count does not divide
    # the task count, and since every video costs about the same, the wall clock is then set
    # by the heaviest rank while the rest sit idle -- e.g. 10 tasks over 8 ranks gives two
    # ranks twice the work of the other six, so the run costs what 16 tasks would. Dealing
    # the individual (task, sim) videos round-robin instead balances any task count over any
    # worker count to within a single video. Each rank still opens each task's store at most
    # once (the pairs stay grouped by task), and zarr reads are lazy, so the finer split
    # costs only a few extra metadata reads.
    all_videos = _expected_observations(paths, args, data_bank_root, timing_label, compress,
                                        spec.draw_spec)
    my_videos = shard_by_rank(all_videos, topo)     # ordered by task, then sim
    my_tasks = sorted({t for t, _ in my_videos})
    total_sims = len(my_videos)

    # ---- Live progress log (so a long MAP run can be monitored) ----------
    recovery_dir.mkdir(parents=True, exist_ok=True)
    if topo.is_main:
        shutil.rmtree(recovery_dir / "figures", ignore_errors=True)

    def log_progress(progress_fh, message: str) -> None:
        """Append a timestamped line to the progress log and flush immediately."""
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        if progress_fh is not None:
            progress_fh.write(line + "\n")
            progress_fh.flush()

    def log_file_only(progress_fh, message: str) -> None:
        """Write a line to the progress log only (console already has it via --debug)."""
        if progress_fh is not None:
            progress_fh.write(f"           {message}\n")
            progress_fh.flush()

    if topo.is_distributed and not topo.is_main:
        progress_fh = None
    else:
        try:
            progress_fh = open(progress_path, "w", encoding="utf-8")
        except OSError as exc:
            print(f"WARNING: cannot open progress log {progress_path} ({exc}); "
                  f"continuing without it.", flush=True)
            progress_fh = None

    step_log = (lambda m: log_file_only(progress_fh, m)) if debug_log else None

    # ---- The three point estimates over the EVAL namespace ---------------
    scores, map_est, true_log10, post_quantiles, post_sgm = [], [], [], [], []
    task_index, sim_index = [], []
    sgm_scale = prior_scale(spec.draw_spec)
    shard_note = (f" [shard rank {topo.rank}/{topo.world_size}: {len(my_tasks)} of "
                  f"{args.eval_tasks} tasks]" if topo.is_distributed else "")
    log_progress(progress_fh,
                 f"START MAP recovery: {len(my_tasks)} EVAL task(s){shard_note}, "
                 f"{total_sims} videos total (pool={theta_prex_size}, "
                 f"elites={elite_prex_size}, steps={numb_steps}, "
                 f"patience={eval_cfg.optimizer_patience}/{eval_cfg.scheduler_patience}, "
                 f"lr={lr:g}..{eval_cfg.learning_rate_minimum:g} in pool-IQR units, "
                 f"tolerance={tolerance:g} nats; "
                 f"estimates=map,median,sgm over {posterior_samples} draws).")
    map_stops = Counter()
    loop_start = time.time()
    done = 0
    try:
        # Group this rank's videos by task so each store is opened once, in order.
        for task, task_videos in groupby(my_videos, key=lambda pair: pair[0]):
            video_set = load_data(
                paths.video_set_path(task, data_bank_root, timing_label, compress, "EVAL"))
            theta_set = load_theta_set(
                paths.theta_set_path(task, data_bank_root, timing_label, compress, "EVAL"),
                spec.draw_spec, condition=args.condition)
            task_start = time.time()
            task_count = 0
            for _, sim in task_videos:
                task_count += 1
                video_chunk = np.asarray(video_set[sim])
                if show:
                    print(f"\n######## MAP estimate: task {task} sim {sim} ########",
                          flush=True)
                if debug_log:
                    log_file_only(progress_fh, f"-- task {task} sim {sim} --")
                score, theta_log, map_info = map_estimate(
                    posterior, video_chunk, device, vista_device,
                    theta_prex_size, eval_cfg.theta_prex_batch_size,
                    eval_cfg.score_prex_batch_size, elite_prex_size,
                    numb_steps, eval_cfg.optimizer_patience,
                    eval_cfg.scheduler_patience, show_progress_steps,
                    eval_cfg.learning_rate_minimum, eval_cfg.learning_rate_factor,
                    lr, tolerance, pool_mode=pool_mode, show=show, verbose=verbose_deep,
                    log_fn=step_log, return_info=True,
                )
                map_stops[map_info["stop"]] += 1
                true_log = to_flow(np.asarray(theta_set[sim], dtype=float), spec.draw_spec)
                scores.append(score)
                map_est.append(theta_log)
                true_log10.append(true_log)
                task_index.append(task)
                sim_index.append(sim)
                # The two draw-derived estimates, from ONE draw set per video: the quantile
                # summary (its 0.50 level is the marginal median) and the SGM of the same draws.
                summary, sgm_vec = posterior_summary(
                    posterior, video_chunk, device, vista_device,
                    posterior_samples, eval_cfg.theta_prex_batch_size,
                    pool_mode=pool_mode, quantiles=QUANTILE_LEVELS,
                    return_sgm=True, sgm_scale=sgm_scale)
                post_quantiles.append(summary)
                post_sgm.append(sgm_vec)
                if show:
                    print(f"          original theta [LOG] {_theta_repr(true_log)}",
                          flush=True)
                    print(f"          original theta [ABS] "
                          f"{_theta_repr(to_physical(true_log, spec.draw_spec))}", flush=True)
                done += 1
                elapsed = time.time() - loop_start
                avg = elapsed / done
                eta = avg * (total_sims - done)
                log_progress(
                    progress_fh,
                    f"task {task} sim {sim} | {done}/{total_sims} | "
                    f"log_prob={score:+.3f} | stop={map_info['stop']}@{map_info['steps']} | "
                    f"elapsed={elapsed:.1f}s | avg={avg:.1f}s/video | ETA={eta:.0f}s")
                if debug_log:
                    log_file_only(progress_fh,
                                  f"pool IQR [LOG] {_theta_repr(map_info['scale'], 4)}")
                    log_file_only(progress_fh,
                                  f"inferred [LOG] {_theta_repr(theta_log)}")
                    log_file_only(progress_fh,
                                  f"original [LOG] {_theta_repr(true_log)}")
            log_progress(progress_fh,
                         f"task {task} complete ({task_count} videos on this rank in "
                         f"{time.time() - task_start:.1f}s).")
        log_progress(progress_fh,
                     f"DONE: {done}/{total_sims} videos in "
                     f"{time.time() - loop_start:.1f}s.")
        log_progress(progress_fh, "MAP stops: " + (", ".join(
            f"{k} {map_stops[k]}" for k in STOP_REASONS if map_stops[k]) or "none"))
    finally:
        if progress_fh is not None:
            progress_fh.close()

    # ---- Assemble, validate, write ---------------------------------------
    # The product is one dict: the three estimates, truth, scores, observation ids and the
    # manifest. It is validated BEFORE anything is written, so an incomplete or non-finite
    # product never reaches disk. One worker writes the final report directly; multiple workers
    # each write a validated shard, and the ``--merge`` step combines them into the report.
    manifest = schema.build_manifest(
        stage="evaluation", parameter_keys=spec.parameter_keys,
        coordinate_transform="parameterization.to_flow (log10 for log rows, linear rows as-is)",
        pool_mode=pool_mode, draw_label=draw_label(pool_mode), n_summary_draws=posterior_samples,
        quantile_levels=QUANTILE_LEVELS, sgm_scale=sgm_scale, sgm_coordinates="estimator",
        optimizer=opt_contract,
        code=finalize_code_provenance(code_at_start), checkpoint_sha256=checkpoint_sha256,
        run_identity=run_ident, execution=exec_ident, seed_policy=schema.seed_policy(args.seed),
        window_geometry=None, condition_labels=None, stored_optional_fields=[],
        n_observations=len(scores), rank=topo.rank if topo.is_distributed else None,
        world_size=topo.world_size if topo.is_distributed else None,
        estimate_definitions_version=ESTIMATE_DEFINITIONS_VERSION)
    if scores:
        arrays = dict(
            scores=np.asarray(scores, dtype=float), map_estimate=np.asarray(map_est, dtype=float),
            true_log10=np.asarray(true_log10, dtype=float),
            posterior_quantiles=np.asarray(post_quantiles, dtype=float),
            posterior_sgm=np.asarray(post_sgm, dtype=float),
            task_index=np.asarray(task_index, dtype=np.int64),
            sim_index=np.asarray(sim_index, dtype=np.int64))
    else:
        # A rank that drew no videos: correctly shaped (0, D) / (0, D, Q) / (0,) arrays, so the
        # shard is valid and mergeable (an empty list converted to an array would be (0,)).
        arrays = schema.empty_product_arrays("evaluation", n_parameters=len(spec.parameter_keys),
                                             n_levels=len(QUANTILE_LEVELS))
    arrays[schema.MANIFEST_KEY] = schema.encode_manifest(manifest)
    # Validated BEFORE anything is written. An empty product is admitted only as a shard of a
    # sharded run; a single-process run over nothing is an error, not a report.
    try:
        schema.validate_product(arrays, stage="evaluation",
                                source=f"rank {topo.rank} product", allow_empty=topo.is_distributed)
        if not topo.is_distributed:
            schema.assert_unique_observations(schema.observation_ids(arrays, "evaluation"),
                                              source="product", expected=all_videos)
    except schema.SchemaError as exc:
        raise SystemExit(f"Evaluation product failed the contract: {exc}")
    if topo.is_distributed:
        _save_shard(topo, recovery_dir, arrays, run_start)
    else:
        write_recovery_outputs(reporter, args, eval_cfg, spec.draw_spec, recovery_array_path,
                               arrays, run_start)


def build_evaluation_parser() -> argparse.ArgumentParser:
    """Construct the Evaluation CLI parser (identical for both workflows)."""
    eval_cfg = PARAMETERS.inference.evaluation
    parser = argparse.ArgumentParser(
        description="Evaluate a trained posterior by MAP recovery on the EVAL namespace.",
    )
    parser.add_argument(
        "--condition", required=True, choices=LABELING_CONDITIONS,
        help="Experimental condition of the run (FAB = MET-FAB, INLB = MET-INLB): selects the "
             "condition-specific data and estimator namespace (the condition slot of the "
             "runtime grammar).",
    )
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
        help="Video duration in seconds; must match the trained posterior's runs.",
    )
    parser.add_argument(
        "--eval-tasks", type=int, required=True,
        help="Number of EVAL-namespace tasks to recover (held-out data).",
    )
    parser.add_argument(
        "--max-sims", type=int, default=0,
        help="Cap on simulations recovered per EVAL task (0 = all; useful for "
             "quick checks).",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Combine-only mode: read the per-shard recovery .npz files written by "
             "a multi-GPU sharded run (in this run's MAP_Recovery directory), "
             "concatenate them, and write the final report + figures + combined "
             ".npz, then exit. Does no recovery and needs no GPU; the launcher runs "
             "it once after the sharded workers finish. Single-GPU runs never use it.",
    )
    parser.add_argument(
        "--artifact-tag", default=None,
        help="Optional SCREAMING_SNAKE token ([A-Z0-9]+, e.g. CAP256) appended to the timing "
             "label of every PRODUCT this stage reads and writes (the estimator it loads, the MAP-recovery directory and arrays), so a named experiment lives "
             "beside the canonical run instead of overwriting it (Paths.product_label). The "
             "shared inputs (video/theta sets, recordings) are always read under the plain "
             "timing label. Default: no tag (canonical names).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and inputs, print what would be read/written, then exit "
             "without running the stage (no GPU, no compute). Use before a queue submission or a long local run.",
    )
    parser.add_argument(
        "--seed", type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v), default=None,
        help="Master RNG seed (PyTorch + numpy + Python random). Default None "
             "-> non-deterministic (consistent with generation); pass an int for a "
             "reproducible run.",
    )
    parser.add_argument(
        "--pool-mode", choices=("bounded", "unrestricted"), default=None,
        help="Candidate-pool sampler: 'bounded' (rejection-sample within the "
             "prior; correct for a trained posterior) or 'unrestricted' (sample "
             "the flow directly, no rejection; for smoke tests / undertrained "
             f"posteriors that would stall). Default: {eval_cfg.pool_mode}.",
    )
    parser.add_argument(
        "--summary", action=RetiredOption,      # retired in 0.1.16: an explicit error, hidden from --help
    )
    parser.add_argument(
        "--bin-mode", choices=("prior", "quantile"), default="quantile",
        help="Bin edges for the MAP conditional-quantile bands: 'quantile' "
             "(equal-count data-quantile bins; default) or 'prior' (equal-width "
             "bins across the prior range).",
    )
    parser.add_argument(
        "--posterior-samples", type=int, default=None,
        help=f"Draws per video summarized by the quantiles, the median and the SGM "
             f"(default: {eval_cfg.posterior_samples}).",
    )
    parser.add_argument(
        "--n-bins", type=int, default=None,
        help=f"Number of bins for the MAP conditional-quantile bands "
             f"(default: {eval_cfg.quantile_bins}). Lower it for a small smoke set.",
    )
    parser.add_argument(
        "--min-count", type=int, default=None,
        help=f"Minimum points per bin to draw a MAP band "
             f"(default: {eval_cfg.quantile_min_count}). Set to 1 to force bands "
             f"on a minimal smoke set.",
    )
    parser.add_argument(
        "--theta-prex-size", type=int, default=None,
        help=f"Candidate-pool size per video (default: {eval_cfg.theta_prex_size}).",
    )
    parser.add_argument(
        "--elite-prex-size", type=int, default=None,
        help=f"Number of optimization seeds / top-K (default: {eval_cfg.elite_prex_size}).",
    )
    parser.add_argument(
        "--numb-steps", type=int, default=None,
        help=f"Max gradient-ascent steps (default: {eval_cfg.numb_steps}). The "
             "optimization may stop earlier via the patience criterion.",
    )
    parser.add_argument(
        "--show-progress-steps", type=int, default=None,
        help="Cadence (in steps) for the per-step log-prob progress line "
             f"(default: {eval_cfg.show_progress_steps}). Lower = more frequent.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Initial Adam learning rate of the MAP ascent, in pool-IQR units (a fraction of "
             f"each parameter's candidate-pool spread per step; default {eval_cfg.learning_rate}).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Rich per-video console diagnostics: per-stage shapes/timings, "
             "per-step optimization progress, stopping reason, and the optimal "
             "vs. original theta in log10 and physical units.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Implies --verbose, and goes deeper (the 'verbose' "
             "level): sbi sampling progress bars + optimizer-config on the "
             "console, plus per-video / per-step detail written into progress.log.",
    )
    parser.add_argument(
        "--debug-dump", action="store_true",
        help="Implies --debug; additionally tees the full console transcript to "
             "<data_bank>/Labor/Debug/<run>/Evaluation/console.log (the recovery "
             "report itself always lands in Posit/).",
    )
    return parser
