"""Shared Experiment-stage engine for both DIMER workflows (biology + detector).

``run_experiment(cfg, args)`` MAP-estimates model parameters from real microscopy
videos (no ground truth): each cell's long recording is windowed into model-length
chunks, the MAP theta is estimated per chunk, and the report shows the distribution
of inferred parameters per experimental condition (kind) so conditions can be
compared. The two entry-point scripts shrink to: build the ``WorkflowConfig``,
parse args, call ``run_experiment``.

Both workflows route through the shared ``experiment_support`` module for the
low-level pieces they share -- cell discovery, per-cell read+window, rank
round-robin, and the shard save/load/merge -- so they discover, window, shard, and
merge byte-for-byte identically. (The 0.4.3 refactor moved the detector Experiment
onto that module; unifying the engine moves the biology Experiment onto it too.)
The report machinery (``_aggregate_by_kind`` / ``write_experiment_outputs``) lives
here, shared. The only per-workflow differences are which parameterization supplies
the learnable table (report tables + figures), the ``parameter_keys`` schema guard,
the device ``build_prior``, and the alias-qualified ``paths`` -- resolved from the
config in ``_experiment_spec(cfg)``. The estimator path is resolved through the
shared ``paths.estimator_path`` method for both workflows.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch._dynamo

from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp import artifacts
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_monomer_dimer_alp.evaluation import (
    map_estimate,
    experiment_table, experiment_estimates_compared_table,
    posterior_summary,
    _theta_repr,
    point_estimate_agreement_table, prior_scale,
)
from srm_and_sbi_monomer_dimer_alp.experiment_support import (
    assert_complete_shard_set,
    condition_display,
    discover_cells,
    load_shards,
    merge_shard_arrays,
    read_cell_chunks,
    save_shard,
    shard_by_rank,
)
from srm_and_sbi_monomer_dimer_alp.inference_support import resolve_topology
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_monomer_dimer_alp import temporal_dynamics as tdk
from srm_and_sbi_monomer_dimer_alp.visualization_inference import (
    figure_experiment_combined, figure_window_drift, figure_point_estimates_by_cell)
from srm_and_sbi_monomer_dimer_alp.workflow import WorkflowConfig


@dataclass(frozen=True)
class _ExperimentSpec:
    """Per-workflow Experiment specializations resolved from a ``WorkflowConfig``."""
    draw_spec: list          # learnable table for the report tables + figures
    parameter_keys: list     # load_estimator schema guard
    build_prior: Callable    # device prior rebuild for bounded rejection sampling


def _experiment_spec(cfg: WorkflowConfig) -> _ExperimentSpec:
    """Resolve the Experiment stage's per-workflow specializations from the workflow config."""
    m = cfg.param_module
    if cfg.tag == "detector":
        return _ExperimentSpec(
            draw_spec=m.DETECTOR_PARAMETERIZATION,
            parameter_keys=[e["KEY"] for e in m.DETECTOR_PARAMETERIZATION],
            build_prior=m.build_prior,
        )
    return _ExperimentSpec(
        draw_spec=m.PARAMETERIZATION,
        parameter_keys=m.PARAMETER_KEYS,
        build_prior=m.build_prior,
    )


def _aggregate_by_kind(inferred_log10, kind_index, cell_arr, kinds, mode, n_params):
    """Group inferred theta by condition for the report, per ``mode``.

    ``"pooled"``      -- every (cell, chunk) estimate is one sample, pooled per
                         kind (mixes temporal + biological variation).
    ``"cell-median"`` -- collapse each cell to its median across its chunks first,
                         so each kind's distribution is one sample per cell.
    """
    out = {}
    for ki, kind in enumerate(kinds):
        kmask = kind_index == ki
        if not np.any(kmask):
            out[kind] = np.empty((0, n_params))
            continue
        kinf = inferred_log10[kmask]
        if mode == "cell-median":
            kcells = cell_arr[kmask]
            out[kind] = np.asarray(
                [np.median(kinf[kcells == c], axis=0) for c in np.unique(kcells)])
        else:  # pooled
            out[kind] = kinf
    return out


def write_experiment_outputs(reporter, args, eval_cfg, draw_spec, array_path: Path,
                             scores, inferred_log10, kind_index, cell_of, chunk_of,
                             post_quantiles, post_samples, kinds, run_start,
                             post_sgm=None, persist_arrays: bool = True) -> None:
    """Save the inferred-theta arrays and write the report + figures.

    Shared by the single-process path and the ``--merge`` combine step, so both
    emit an identical report. ``post_quantiles`` may be a list (built by the
    estimation loop) or an array (concatenated by a merge); both are handled.
    ``post_sgm`` (optional, ``(N, D)``) is the per-window sample geometric median of the
    same posterior draws; when present the inferred-theta table is repeated for it and
    for the 1-D posterior median, and the point-estimate agreement table is written.
    ``draw_spec`` is the workflow's learnable table.
    """
    do_map = args.summary in ("map", "both")
    do_posterior = args.summary in ("posterior", "both")
    posterior_samples = args.posterior_samples or eval_cfg.posterior_samples

    scores = np.asarray(scores)
    inferred_log10 = np.asarray(inferred_log10)
    kind_index = np.asarray(kind_index)
    cell_of = np.asarray(cell_of)
    chunk_of = np.asarray(chunk_of)
    n_estimates = inferred_log10.shape[0]
    # (N, D, 5) posterior quantiles [Q05,Q25,Q50,Q75,Q95] when View B was requested.
    post_arr = np.asarray(post_quantiles)
    post_q = post_arr if (do_posterior and post_arr.size > 0) else None
    # (N, n_samples, D) raw draws, only when the run was asked to keep them.
    post_s_arr = np.asarray(post_samples)
    post_s = post_s_arr if (do_posterior and post_s_arr.size > 0) else None
    sgm_arr = np.asarray(post_sgm if post_sgm is not None else [])
    sgm = sgm_arr if (post_q is not None and sgm_arr.size > 0) else None

    # ---- Save the inferred-theta arrays ----------------------------------
    save_arrays = dict(
        inferred_log10=inferred_log10, scores=scores, kind_index=kind_index,
        cell=cell_of, chunk=chunk_of, kinds=np.asarray(kinds))
    if post_q is not None:
        save_arrays["posterior_quantiles"] = post_q
    if post_s is not None:
        save_arrays["posterior_samples_cloud"] = post_s
    if sgm is not None:
        save_arrays["posterior_sgm"] = sgm              # (N, D) sample geometric median
    if persist_arrays:
        np.savez_compressed(str(array_path), **save_arrays)
        print(f"\nExperiment MAP arrays saved to {array_path}")
    else:
        print(f"\nExperiment arrays left as stored at {array_path} (report-only rendering)")

    # ---- Report ----------------------------------------------------------
    inferred_by_kind = _aggregate_by_kind(
        inferred_log10, kind_index, cell_of, kinds,
        args.aggregation, len(draw_spec))
    # Everything a reader sees carries the scientific condition name (MET-FAB / MET-INLB). The
    # stored `kinds` field keeps the "FAB"/"INLB" tokens: the data-file namespace and the schema
    # downstream analyses key on; presentation prepends the receptor.
    shown = [condition_display(k) for k in kinds]
    inferred_shown = {condition_display(k): v for k, v in inferred_by_kind.items()}
    agg_desc = ("pooled over (cell x chunk)" if args.aggregation == "pooled"
                else "one point per cell (median over its chunks)")
    reporter.check("estimates_nonempty", n_estimates > 0,
                   f"{n_estimates} MAP estimates over real videos",
                   note="at least one experimental chunk was MAP-estimated.")
    reporter.stat("conditions", len(kinds))
    reporter.stat("total_estimates", n_estimates,
                  note="number of (cell, chunk) windows MAP-estimated across all conditions.")
    reporter.stat("aggregation", args.aggregation, note=f"report distribution view: {agg_desc}.")
    for kind, name in zip(kinds, shown):
        reporter.stat(f"n[{name}]", int(inferred_by_kind[kind].shape[0]),
                      note=f"estimates for condition {name} (stored as '{kind}').")
    if n_estimates:
        reporter.stat("mean_log_prob", float(np.mean(scores)),
                      note="mean MAP log-density at the optimized mode (optimization "
                           "diagnostic; not a calibration/quality metric).")

    headers, rows = experiment_table(draw_spec, inferred_shown, shown)
    reporter.table("MAP theta by condition (log10 units)", headers, rows,
                   note=f"no ground truth for real data; values are the distribution "
                        f"of the MAP (the optimizer's mode) per condition ({agg_desc}); the "
                        f"posterior-median and SGM tables below repeat it for the other two "
                        f"point estimates, and the three are read together. Compare "
                        f"conditions to read out parameter differences. 'outside prior' "
                        f"is the share of estimates beyond the row's prior box (possible "
                        f"only under --pool-mode unrestricted).")

    if post_q is not None:
        reporter.stat("posterior_samples", posterior_samples,
                      note="samples per chunk used to summarize the posterior (View B).")
        # The same per-condition table for the two posterior-derived point estimates, so
        # the MAP, the 1-D posterior median and the SGM stand side by side.
        med_by_kind = _aggregate_by_kind(post_q[:, :, 2], kind_index, cell_of, kinds,
                                         args.aggregation, len(draw_spec))
        med_shown = {condition_display(k): v for k, v in med_by_kind.items()}
        med_headers, med_rows = experiment_table(draw_spec, med_shown, shown)
        reporter.table("Posterior-median theta by condition (log10 units)",
                       med_headers, med_rows,
                       note=f"same table for the 1-D posterior median (Q50 of each "
                            f"marginal) of every window ({agg_desc}).")
        if sgm is not None:
            sgm_by_kind = _aggregate_by_kind(sgm, kind_index, cell_of, kinds,
                                             args.aggregation, len(draw_spec))
            sgm_shown = {condition_display(k): v for k, v in sgm_by_kind.items()}
            sgm_headers, sgm_rows = experiment_table(draw_spec, sgm_shown, shown)
            reporter.table("SGM theta by condition (log10 units)", sgm_headers, sgm_rows,
                           note=f"same table for the sample geometric median (SGM) of "
                                f"every window's posterior cloud -- the sample closest, in "
                                f"prior-width-scaled log10 distance, to all other samples, "
                                f"so a joint point estimate that is itself a probable point "
                                f"({agg_desc}).")
        # The three estimates in one row per (parameter, condition): the view in which they
        # are read against each other (the tables above carry each one's full distribution).
        cmp_estimates = {"MAP": inferred_shown, "median": med_shown}
        if sgm is not None:
            cmp_estimates["SGM"] = sgm_shown
        cmp_headers, cmp_rows = experiment_estimates_compared_table(
            draw_spec, cmp_estimates, shown)
        reporter.table("Point estimates compared (per parameter, log10 units)",
                       cmp_headers, cmp_rows,
                       note=f"the three point estimates on the same windows ({agg_desc}), columns "
                            f"grouped by statistic with the estimates consecutive: the median over "
                            f"windows, the IQR over windows and the share outside the prior box, each "
                            f"for MAP, median and SGM. 'median' is the 1-D posterior median (Q50 of "
                            f"each marginal); 'SGM' the sample geometric median of the posterior cloud. "
                            f"A statement about a parameter is read from the row as a whole, never "
                            f"from one column.")
        agr_headers, agr_rows = point_estimate_agreement_table(
            draw_spec, inferred_log10, post_q, sgm,
            groups=[condition_display(kinds[k]) for k in kind_index])
        reporter.table("Point-estimate agreement (per parameter, log10 units)",
                       agr_headers, agr_rows,
                       note="each 'X vs Y' column is the median over windows of the absolute "
                            "difference between the two point estimates (log10); 'median' is the "
                            "1-D posterior median, 'SGM' the sample geometric median. 'MAP outside "
                            "90%' is the share of windows whose MAP falls outside the posterior's "
                            "central 90% interval. Large gaps with a high outside share establish "
                            "that the optimized mode and the posterior summaries disagree; whether "
                            "that is a density spike the samples do not visit, an optimizer that "
                            "stopped short, or another feature of the posterior's shape is not "
                            "decided by this table and needs separate checks. The three point "
                            "estimates are read together, and no single one replaces the others.")

    # ---- Within-recording drift: every stored point estimate against window position ----
    # The windows of one recording share the same acquisition, so a trend across them is either
    # dynamics or an acquisition confound; this table and figure measure it for the MAP, the
    # posterior median and the SGM together, against the posterior's own per-window interval.
    if n_estimates:
        grids, _, n_chunks_g = tdk.point_estimate_grids(
            inferred_log10, kind_index, cell_of, chunk_of, len(kinds), post_q, sgm)
        if n_chunks_g >= 2:
            keys = [para["KEY"] for para in draw_spec]
            labels = [para.get("LABEL") or para["KEY"] for para in draw_spec]
            log_rows = np.array([bool(para.get("LOG_FLAG")) for para in draw_spec], dtype=bool)
            rows = tdk.window_drift_rows(grids, [condition_display(k) for k in kinds], keys,
                                         lambda u: u, log_rows)
            d_headers, d_rows = tdk.format_drift_rows(rows)
            reporter.table(
                "Within-recording drift across windows (per point estimate, stored coordinates)",
                d_headers, d_rows,
                note="per recording, a line is fitted to each estimate against the window index "
                     "and the fitted first-to-last change is taken; the row aggregates those "
                     "changes across recordings. Rows exist for every point estimate the run "
                     "stored (MAP; posterior median and SGM under --summary posterior|both), "
                     "and they are read together: the three come from the same posterior, so a "
                     "difference between them describes the posterior's shape along the "
                     "recording. 'cells over 0.3 dex' is the share whose change exceeds a "
                     "factor of two. The signed-rank p states detectability, not magnitude, and "
                     "no row attributes a cause: the biology block is marginalized here, so an "
                     "imaging trend and a biological one are not separable in this stage alone.")
            if reporter.dump:
                bands_all = (tdk.shared_posterior_bands(post_q, kind_index, cell_of, chunk_of,
                                                        len(kinds)) if post_q is not None else None)
                for ki, kind in enumerate(kinds):
                    med = {name: tdk.window_medians(g)[ki] for name, g in grids.items()}
                    fig = figure_window_drift(
                        keys, labels, med,
                        bands=None if bands_all is None else bands_all[ki],
                        prior_ranges=[para["PRIOR_RANGE"] for para in draw_spec],
                        title=f"{condition_display(kind)}: point estimates against window "
                              f"position (lines = median across recordings; bands = median "
                              f"per-window posterior 50 % and 90 % intervals)")
                    reporter.save_figure(
                        f"window_drift_{kind}", fig,
                        caption=f"Within-recording drift, condition {condition_display(kind)}: "
                                f"the MAP, posterior median and SGM (where stored) as the median "
                                f"across recordings at each window, over the posterior's own "
                                f"per-window central 50 % and 90 % intervals (median across "
                                f"recordings). Dotted red lines are the prior bounds. The three "
                                f"lines share one posterior, so the band is drawn once.")

    if reporter.dump and n_estimates:
        for i, para in enumerate(draw_spec):
            key = para["KEY"]
            label = para.get("LABEL") or key
            prior_range = para["PRIOR_RANGE"]
            values = ({condition_display(kind): inferred_by_kind[kind][:, i]
                       for kind in kinds} if do_map else None)
            # View B: per chunk, [MAP, posterior median, q25, q75] by kind.
            by_kind_post = None
            if post_q is not None:
                by_kind_post = {}
                for ki, kind in enumerate(kinds):
                    m = kind_index == ki
                    map_col = inferred_log10[m][:, i:i + 1]               # (n, 1)
                    q_cols = post_q[m][:, i][:, [2, 1, 3]]                # (n, 3): med,q25,q75
                    by_kind_post[condition_display(kind)] = np.hstack(
                        [map_col, q_cols])                                # (n, 4)
            reporter.save_figure(
                f"experiment_{key}",
                figure_experiment_combined(
                    values, by_kind_post, prior_range, label, seed=args.seed,
                    show_map=do_map, show_posterior=(post_q is not None)),
                caption=f"{key} ({label}). View A (MAP): per-condition distribution of "
                        f"the MAP point estimate ({agg_desc}). View B (posterior): each "
                        f"chunk's posterior median +/- IQR per condition (within-chunk "
                        f"uncertainty)"
                        + (f"; the SGM is tabulated above and its gap to the median is in the "
                           f"agreement table" if sgm is not None else
                           f"; this run stored no SGM")
                        + f". A panel stamped 'not computed' marks a view the --summary option "
                          f"omitted.",
            )

    if reporter.dump and n_estimates and post_q is not None:
        # The point-estimate view for experimental recordings: no truth exists, so the three
        # estimates are laid out per recording over the posterior's own IQR (one shared band),
        # recordings ordered by their posterior-median value. This is the figure a statement about
        # the inference on real recordings is read from, beside the tables above.
        for ki, kind in enumerate(kinds):
            kmask = kind_index == ki
            if not np.any(kmask):
                continue
            for i, para in enumerate(draw_spec):
                key = para["KEY"]
                label = para.get("LABEL") or key
                fig = figure_point_estimates_by_cell(
                    {"MAP": inferred_log10[kmask, i], "median": post_q[kmask, i, 2],
                     "SGM": (sgm[kmask, i] if sgm is not None else None)},
                    post_q[kmask, i, :], cell_of[kmask], para["PRIOR_RANGE"],
                    f"{label} ({condition_display(kind)})")
                if fig is not None:
                    reporter.save_figure(
                        f"point_estimates_{kind}_{key}",
                        fig,
                        caption=f"{key} ({label}), {condition_display(kind)}. The three point estimates "
                                f"per window, one panel each (MAP; median = 1-D posterior median; SGM): "
                                f"recordings along the x axis ordered by their posterior-median value, "
                                f"one point per window, the recording's median of that estimate as the "
                                f"line, and the posterior IQR (median over the recording's windows of the "
                                f"per-window Q25 and Q75) as the band, drawn once and identical on every "
                                f"panel. The estimate axis is clipped to the prior box widened by 40 %; "
                                f"values beyond it are counted in the tables, not drawn.")

    reporter.summary()
    reporter.write_report()
    print(f"\nTotal elapsed: {time.time() - run_start:.1f}s")


def _save_shard(topo, out_dir: Path, scores, inferred_log10, kind_index, cell_of,
                chunk_of, post_quantiles, post_samples, kinds, run_start,
                post_sgm=None) -> None:
    """Write this worker's partial experiment arrays (multi-GPU sharded run)."""
    arrays = dict(
        scores=np.asarray(scores),
        inferred_log10=np.asarray(inferred_log10),
        kind_index=np.asarray(kind_index),
        cell=np.asarray(cell_of),
        chunk=np.asarray(chunk_of),
        kinds=np.asarray(kinds),
    )
    if post_quantiles:
        arrays["posterior_quantiles"] = np.asarray(post_quantiles)
    if post_samples:
        arrays["posterior_samples_cloud"] = np.asarray(post_samples)
    if post_sgm:
        arrays["posterior_sgm"] = np.asarray(post_sgm)
    path = save_shard(out_dir, topo, arrays, count=len(scores))
    if path is None:
        print(f"\n[rank {topo.rank}/{topo.world_size}] no cells assigned -- "
              f"no shard written.", flush=True)
    else:
        print(f"\n[rank {topo.rank}/{topo.world_size}] shard saved: {path} "
              f"({arrays['scores'].shape[0]} estimates) in {time.time() - run_start:.1f}s. "
              f"Run the --merge step once all shards finish.", flush=True)


def _merge_shards(reporter, args, eval_cfg, draw_spec, out_dir: Path,
                  array_path: Path, run_start) -> None:
    """Combine all per-shard experiment arrays into the final report (no estimation)."""
    shard_paths = load_shards(out_dir)
    if not shard_paths:
        raise SystemExit(
            f"--merge: no shard files (_shard_*_of_*.npz) found in {out_dir}")
    try:
        world_size = assert_complete_shard_set(shard_paths, allow_partial=args.allow_partial)
    except ValueError as exc:
        raise SystemExit(f"--merge: {exc}")
    reporter.stat("shards_merged", f"{len(shard_paths)}/{world_size}",
                  note="per-rank shards combined into this report; fewer than world_size means "
                       "--allow-partial was used and some (kind, cell) work is missing.")
    print(f"Merging {len(shard_paths)} shard file(s) from {out_dir}", flush=True)
    merged, n_used = merge_shard_arrays(
        shard_paths,
        concat_keys=["scores", "inferred_log10", "kind_index", "cell", "chunk"],
        first_keys=["kinds"],
        optional_concat_keys=["posterior_quantiles", "posterior_samples_cloud",
                              "posterior_sgm"])
    kinds = [str(k) for k in merged["kinds"]]
    # An absent posterior_quantiles key -> empty array signals "not computed" to
    # write_experiment_outputs.
    post_quantiles = merged.get("posterior_quantiles", np.asarray([]))
    post_samples = merged.get("posterior_samples_cloud", np.asarray([]))
    post_sgm = merged.get("posterior_sgm")
    print(f"Merged {merged['scores'].shape[0]} estimates from {n_used} shard(s).", flush=True)
    write_experiment_outputs(reporter, args, eval_cfg, draw_spec, array_path,
                             merged["scores"], merged["inferred_log10"],
                             merged["kind_index"], merged["cell"], merged["chunk"],
                             post_quantiles, post_samples, kinds, run_start,
                             post_sgm=post_sgm)
    for shard_path in shard_paths:
        shard_path.unlink()
    print(f"Removed {len(shard_paths)} shard file(s).", flush=True)


def run_experiment(cfg: WorkflowConfig, args: argparse.Namespace) -> None:
    """Run MAP estimation over the experimental videos for the given workflow + CLI args."""
    spec = _experiment_spec(cfg)

    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = cfg.paths.with_condition(args.condition)   # condition-specific namespace
    eval_cfg = PARAMETERS.inference.evaluation
    span = args.experiment_span_seconds

    # ---- Global RNG / precision settings ---------------------------------
    if args.seed is not None:   # None -> non-deterministic (consistent with generation)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    timing_label = timing.label
    # The estimator and every product derived from it carry the optional artifact tag
    # (Paths.product_label); the experimental recordings never do.
    product_label = paths.product_label(timing_label, args.artifact_tag)
    estimator_path = paths.estimator_path(data_bank_root, product_label)
    experiment_dir = data_bank_root / paths.experiment_subdir
    out_dir = paths.experiment_recovery_dir(data_bank_root, product_label)
    array_path = out_dir / (out_dir.name + ".npz")
    progress_path = out_dir / "progress.log"

    # Effective hyperparameters (None -> config default).
    lr = (args.learning_rate if args.learning_rate is not None
          else eval_cfg.learning_rate_minimum * eval_cfg.learning_rate_maximum_factor)
    tolerance = eval_cfg.learning_rate_minimum * eval_cfg.tolerance_factor
    theta_prex_size = args.theta_prex_size or eval_cfg.theta_prex_size
    elite_prex_size = args.elite_prex_size or eval_cfg.elite_prex_size
    numb_steps = args.numb_steps or eval_cfg.numb_steps
    show_progress_steps = args.show_progress_steps or eval_cfg.show_progress_steps
    pool_mode = args.pool_mode or eval_cfg.pool_mode

    show = args.verbose or args.debug or args.debug_dump
    verbose_deep = args.debug or args.debug_dump
    debug_log = args.debug or args.debug_dump

    # Summary views (A = MAP-point distribution; B = posterior credible summary).
    do_map = args.summary in ("map", "both")
    do_posterior = args.summary in ("posterior", "both")
    posterior_samples = args.posterior_samples or eval_cfg.posterior_samples
    dump_samples = args.dump_posterior_samples
    if dump_samples and not do_posterior:
        raise SystemExit(
            "--dump-posterior-samples needs the posterior view: pass --summary posterior "
            "or --summary both. The draws it stores are the ones the posterior summary "
            "makes, so there is nothing to keep when only the MAP view runs.")

    # Video geometry: model-length windows stepped across each long recording.
    n_frames = timing.frame_count
    window_seconds = timing.total_time_seconds
    step_seconds = args.chunk_step_seconds if args.chunk_step_seconds else int(window_seconds)
    if (window_seconds != int(window_seconds) or step_seconds < 1
            or step_seconds > int(window_seconds)
            or int(window_seconds) % int(step_seconds) != 0):
        raise SystemExit(
            f"--chunk-step-seconds={step_seconds} is invalid: it must be a positive "
            f"integer that divides the model window ({window_seconds:g} s) and does "
            f"not exceed it (valid steps: integer divisors of {int(window_seconds)}).")
    step_frames = int(round(step_seconds / timing.frame_time_seconds))
    exp_frames = int(round(span / timing.frame_time_seconds))
    n_chunks = (exp_frames - n_frames) // step_frames + 1

    # ---- Resolve the (kind, cells) work list -----------------------------
    kinds = ([args.condition] if args.kinds is None
             else [k.strip() for k in args.kinds.split(",") if k.strip()])
    explicit_cells = ([int(c) for c in args.cells.split(",")] if args.cells else None)
    cells_by_kind = {}
    for kind in kinds:
        cells = explicit_cells if explicit_cells is not None else discover_cells(
            experiment_dir, kind, span)
        if args.max_cells > 0:
            cells = cells[:args.max_cells]
        cells_by_kind[kind] = cells
    total_estimates = sum(len(c) for c in cells_by_kind.values()) * n_chunks

    # ---- Pre-run banner --------------------------------------------------
    machine = PARAMETERS.machine
    div = "=" * 72
    print(div)
    print(f" {paths.project_alias} — Experiment (MAP estimation on real data)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nMachine profile:")
    print(f"  name              : {machine.name}")
    print(f"  compute_backend   : {machine.compute_backend}")
    print("\nRun configuration (CLI args):")
    print(f"  --total-time-seconds      : {args.total_time_seconds}  (model window -> {n_frames} frames)")
    print(f"  --experiment-span-seconds : {span}  ({n_chunks} chunks/video, step {step_seconds}s)")
    print(f"  --kinds                   : {kinds}")
    for kind in kinds:
        print(f"      {kind}: cells {cells_by_kind[kind]}")
    print(f"  --max-cells               : {args.max_cells}  (0 = all discovered)")
    print(f"  --aggregation             : {args.aggregation}")
    print(f"  --summary                 : {args.summary}   (View A map={do_map}, View B posterior={do_posterior})")
    if do_posterior:
        print(f"  --posterior-samples       : {posterior_samples}")
        n_windows_total = sum(len(cells_by_kind[k]) for k in kinds) * n_chunks
        mb = n_windows_total * posterior_samples * len(spec.draw_spec) * 4 / 1e6
        print(f"  --dump-posterior-samples  : {dump_samples}"
              + (f"   (+{mb:.0f} MB of raw draws: "
                 f"{n_windows_total} windows x {posterior_samples} x "
                 f"{len(spec.draw_spec)})" if dump_samples else ""))
    print(f"  --seed                    : {args.seed}")
    print(f"  total MAP estimates       : {total_estimates}")
    print("\nMAP estimate hyperparameters (effective):")
    print(f"  pool_mode            : {pool_mode}")
    print(f"  theta_prex_size      : {theta_prex_size}    elite_prex_size: {elite_prex_size}")
    print(f"  numb_steps           : {numb_steps}    learning_rate: {lr:.3e}")
    print(f"  verbosity            : "
          f"{'debug-dump' if args.debug_dump else ('debug' if args.debug else ('verbose' if args.verbose else 'normal'))}")
    print("\nOutput destinations:")
    print(f"  reads estimator  : {estimator_path}")
    print(f"  reads videos     : {experiment_dir}/Experiment_<KIND>_Cell_<n>_{span}S_RAW.tif")
    print(f"  writes report    : {out_dir}")
    print(f"  live progress    : {progress_path}   (tail -f to monitor)")
    print(f"\n{div}\n")

    # ---- Dry-run preview (no GPU, no compute) ----------------------------
    if args.dry_run:
        print("[DRY RUN] validating configuration and inputs:")
        checks = [
            ("estimator artifact", estimator_path),
            ("experiment dir", experiment_dir),
        ]
        missing = 0
        for role, path in checks:
            ok = Path(path).exists()
            missing += not ok
            print(f"  reads {role}: {path}  [{'OK' if ok else 'MISSING'}]")
        for kind in kinds:
            discovered = discover_cells(experiment_dir, kind, span)
            print(f"  discovered {kind} recordings: {len(discovered)} cell(s) "
                  f"({span}S_RAW.tif)")
        if missing:
            print(f"[DRY RUN] configuration validated; {missing} input(s) MISSING.")
        else:
            print("[DRY RUN] configuration validated; all inputs present.")
        print("[DRY RUN] no MAP estimation performed.")
        return

    run_start = time.time()

    # ---- Diagnostics reporter (the experiment report is the deliverable) -
    reporter = DiagnosticReporter(
        stage="Experiment", enabled=True, dump=True, dump_dir=out_dir,
        run_label=f"{paths.project_alias}_{product_label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    # --merge: combine the per-shard arrays from a multi-GPU sharded run into the
    # final report, then exit (no estimation, no GPU, no posterior needed).
    if args.merge:
        _merge_shards(reporter, args, eval_cfg, spec.draw_spec, out_dir, array_path, run_start)
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

    # ---- Output dir + progress log + stale-figure clear ------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    if topo.is_main:
        shutil.rmtree(out_dir / "figures", ignore_errors=True)

    def log_progress(progress_fh, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        if progress_fh is not None:
            progress_fh.write(line + "\n")
            progress_fh.flush()

    def log_file_only(progress_fh, message: str) -> None:
        if progress_fh is not None:
            progress_fh.write(f"           {message}\n")
            progress_fh.flush()

    if topo.is_distributed and not topo.is_main:
        progress_fh = None
    else:
        try:
            progress_fh = open(progress_path, "w", encoding="utf-8")
        except OSError as exc:
            print(f"WARNING: cannot open progress log {progress_path} ({exc}).", flush=True)
            progress_fh = None
    step_log = (lambda m: log_file_only(progress_fh, m)) if debug_log else None

    # ---- This worker's (kind, cell) shard --------------------------------
    flat_work = [(ki, cell) for ki, kind in enumerate(kinds)
                 for cell in cells_by_kind[kind]]
    my_work = set(shard_by_rank(flat_work, topo))
    my_estimates = len(my_work) * n_chunks

    # ---- MAP estimation over (kind, cell, chunk) -------------------------
    scores, inferred_log10, post_quantiles = [], [], []
    post_sgm = []                # sample geometric median per window (View B)
    sgm_scale = prior_scale(spec.draw_spec)
    post_samples = []            # raw draws per window, only under --dump-posterior-samples
    kind_index, cell_of, chunk_of = [], [], []
    shard_note = (f" [shard rank {topo.rank}/{topo.world_size}: {len(my_work)} of "
                  f"{len(flat_work)} cells]" if topo.is_distributed else "")
    log_progress(progress_fh,
                 f"START Experiment MAP: {my_estimates} estimates{shard_note} "
                 f"({len(kinds)} kinds, {n_chunks} chunks/video; pool={theta_prex_size}, "
                 f"elites={elite_prex_size}, steps={numb_steps}; summary={args.summary}).")
    loop_start = time.time()
    done = 0
    try:
        for ki, kind in enumerate(kinds):
            for cell in cells_by_kind[kind]:
                if (ki, cell) not in my_work:
                    continue
                tif_path = paths.experiment_video_path(kind, cell, span, data_bank_root)
                if not tif_path.exists():
                    log_progress(progress_fh, f"SKIP {kind} cell {cell}: file missing "
                                              f"({tif_path.name}).")
                    continue
                chunks = read_cell_chunks(tif_path, n_frames, step_frames)
                n_cell_chunks = len(chunks)
                cell_start = time.time()
                for c, chunk in enumerate(chunks):
                    if show:
                        print(f"\n######## MAP estimate: {kind} cell {cell} chunk {c} ########",
                              flush=True)
                    if debug_log:
                        log_file_only(progress_fh, f"-- {kind} cell {cell} chunk {c} --")
                    score, theta_log = map_estimate(
                        posterior, chunk, device, vista_device,
                        theta_prex_size, eval_cfg.theta_prex_batch_size,
                        eval_cfg.score_prex_batch_size, elite_prex_size,
                        numb_steps, eval_cfg.optimizer_patience,
                        eval_cfg.scheduler_patience, show_progress_steps,
                        eval_cfg.learning_rate_minimum, eval_cfg.learning_rate_factor,
                        lr, tolerance, pool_mode=pool_mode, show=show,
                        verbose=verbose_deep, log_fn=step_log,
                    )
                    scores.append(score)
                    inferred_log10.append(theta_log)
                    if do_posterior:
                        out = posterior_summary(
                            posterior, chunk, device, vista_device,
                            posterior_samples, eval_cfg.theta_prex_batch_size,
                            pool_mode=pool_mode, return_samples=dump_samples,
                            return_sgm=True, sgm_scale=sgm_scale)
                        if dump_samples:
                            summary, cloud, sgm_vec = out
                            post_samples.append(cloud)
                        else:
                            summary, sgm_vec = out
                        post_quantiles.append(summary)
                        post_sgm.append(sgm_vec)
                    kind_index.append(ki)
                    cell_of.append(cell)
                    chunk_of.append(c)
                    done += 1
                    elapsed = time.time() - loop_start
                    avg = elapsed / done
                    eta = avg * (my_estimates - done)
                    log_progress(progress_fh,
                                 f"{kind} cell {cell} chunk {c} | {done}/{my_estimates} | "
                                 f"log_prob={score:+.3f} | elapsed={elapsed:.1f}s | "
                                 f"avg={avg:.1f}s | ETA={eta:.0f}s")
                    if debug_log:
                        log_file_only(progress_fh, f"inferred [LOG] {_theta_repr(theta_log)}")
                log_progress(progress_fh, f"{kind} cell {cell} complete "
                                          f"({n_cell_chunks} chunks in "
                                          f"{time.time() - cell_start:.1f}s).")
        log_progress(progress_fh, f"DONE: {done}/{my_estimates} estimates in "
                                  f"{time.time() - loop_start:.1f}s.")
    finally:
        if progress_fh is not None:
            progress_fh.close()

    # ---- Write outputs ---------------------------------------------------
    if topo.is_distributed:
        _save_shard(topo, out_dir, scores, inferred_log10, kind_index, cell_of,
                    chunk_of, post_quantiles, post_samples, kinds, run_start,
                    post_sgm=post_sgm)
    else:
        write_experiment_outputs(reporter, args, eval_cfg, spec.draw_spec, array_path,
                                 scores, inferred_log10, kind_index, cell_of,
                                 chunk_of, post_quantiles, post_samples, kinds,
                                 run_start, post_sgm=post_sgm)


def build_experiment_parser() -> argparse.ArgumentParser:
    """Construct the Experiment CLI parser (identical for both workflows)."""
    eval_cfg = PARAMETERS.inference.evaluation
    parser = argparse.ArgumentParser(
        description="MAP-estimate parameters from real experimental videos (no ground truth).",
    )
    parser.add_argument(
        "--condition", required=True, choices=LABELING_CONDITIONS,
        help="Experimental condition of the run (FAB = MET-FAB, INLB = MET-INLB): selects the "
             "condition-specific data and estimator namespace (the condition slot of the "
             "runtime grammar).",
    )
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
        help="Model window duration in seconds; must match the trained posterior.",
    )
    parser.add_argument(
        "--experiment-span-seconds", type=int, default=20,
        help="Duration of each raw experimental recording in seconds (default: 20).",
    )
    parser.add_argument(
        "--chunk-step-seconds", type=int, default=None,
        help="Step (seconds) between consecutive model-length windows; integer that "
             "divides the window and is <= it. 1 = maximal overlap; window (default) "
             "= non-overlapping. Smaller steps yield more (overlapping) chunks.",
    )
    parser.add_argument(
        "--kinds", type=str, default=None,
        help="Comma-separated experimental conditions to analyze (default: the run's --condition, "
             "the recordings this estimator was calibrated for). Naming the other condition is a "
             "deliberate cross-condition application, a misspecification test, not the default.",
    )
    parser.add_argument(
        "--cells", type=str, default=None,
        help="Comma-separated explicit cell indices (default: discover all on disk).",
    )
    parser.add_argument(
        "--max-cells", type=int, default=0,
        help="Cap on cells per kind (0 = all; useful for quick checks).",
    )
    parser.add_argument(
        "--summary", choices=("map", "posterior", "both"), default="map",
        help="Which views to render: 'map' (View A: MAP-point distribution per "
             "condition; default), 'posterior' (View B: per-chunk posterior median "
             "+/- IQR per condition), or 'both'. View B draws --posterior-samples per chunk.",
    )
    parser.add_argument(
        "--dump-posterior-samples", action="store_true",
        help="Additionally store the raw posterior draws for every window, as "
             "posterior_samples_cloud (n_windows, --posterior-samples, D) in the saved "
             "npz. Quantiles keep only per-parameter marginals; the raw draws keep the "
             "joint structure, which is what pooling posteriors across a recording "
             "needs. Requires --summary posterior or both. Adds roughly "
             "n_windows * samples * D * 4 bytes.",
    )
    parser.add_argument(
        "--posterior-samples", type=int, default=None,
        help=f"Samples per chunk for View B posterior summary "
             f"(default: {eval_cfg.posterior_samples}).",
    )
    parser.add_argument(
        "--aggregation", choices=("pooled", "cell-median"), default="pooled",
        help="Report distribution view: 'pooled' (every cell x chunk estimate is a "
             "sample; mixes temporal + biological variation) or 'cell-median' (one "
             "sample per cell, median over its chunks; biological spread only). "
             "Default: pooled. The .npz keeps raw per-(cell,chunk) data either way.",
    )
    parser.add_argument(
        "--seed", type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v), default=None,
        help="Master RNG seed (PyTorch + numpy + Python random). Default None "
             "-> non-deterministic (consistent with generation); pass an int for a "
             "reproducible run.",
    )
    parser.add_argument(
        "--pool-mode", choices=("bounded", "unrestricted"), default=None,
        help=f"Candidate-pool sampler (default: {eval_cfg.pool_mode}). See Evaluation.",
    )
    parser.add_argument("--theta-prex-size", type=int, default=None,
                        help=f"Candidate-pool size per video (default: {eval_cfg.theta_prex_size}).")
    parser.add_argument("--elite-prex-size", type=int, default=None,
                        help=f"Number of optimization seeds (default: {eval_cfg.elite_prex_size}).")
    parser.add_argument("--numb-steps", type=int, default=None,
                        help=f"Max gradient-ascent steps (default: {eval_cfg.numb_steps}).")
    parser.add_argument("--show-progress-steps", type=int, default=None,
                        help=f"Per-step progress cadence (default: {eval_cfg.show_progress_steps}).")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Adam learning rate for the MAP optimization.")
    parser.add_argument("--verbose", action="store_true",
                        help="Rich per-chunk console diagnostics (see Evaluation).")
    parser.add_argument("--debug", action="store_true",
                        help="Implies --verbose; deeper console + per-step detail in progress.log.")
    parser.add_argument("--debug-dump", action="store_true",
                        help="Implies --debug; tees the console transcript to "
                             "Labor/Debug/<run>/Experiment/console.log.")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="With --merge: combine the shards that exist even when some ranks never saved "
             "theirs (the report then covers only the present shards and says so). Without "
             "it an incomplete shard set aborts the merge, naming the missing ranks.")
    parser.add_argument(
        "--merge", action="store_true",
        help="Combine-only mode: read the per-shard .npz files written by a "
             "multi-GPU sharded run (in this run's experiment output directory), "
             "concatenate them, and write the final report + figures + combined "
             ".npz, then exit. Does no estimation and needs no GPU; the launcher "
             "runs it once after the sharded workers finish. Single-GPU runs never use it.")
    parser.add_argument(
        "--artifact-tag", default=None,
        help="Optional SCREAMING_SNAKE token ([A-Z0-9]+, e.g. CAP256) appended to the timing "
             "label of every PRODUCT this stage reads and writes (the estimator it loads, the experiment report directory and arrays), so a named experiment lives "
             "beside the canonical run instead of overwriting it (Paths.product_label). The "
             "shared inputs (video/theta sets, recordings) are always read under the plain "
             "timing label. Default: no tag (canonical names).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration and inputs, print what would be read/written, then exit "
                             "without running the stage (no GPU, no compute). Use before a queue submission or a long local run.")
    return parser
