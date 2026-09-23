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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch._dynamo

from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from srm_and_sbi_monomer_dimer_alp import artifacts
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_monomer_dimer_alp.evaluation import (
    ESTIMATE_DEFINITIONS_VERSION, QUANTILE_LEVELS,
    draw_label,
    map_estimate,
    optimizer_contract,
    optimizer_summary_lines,
    STOP_REASONS,
    experiment_table, experiment_estimates_compared_table,
    point_estimate_rows,
    posterior_summary,
    short_labels,
    _theta_repr,
    point_estimate_agreement_table, prior_scale,
)
from srm_and_sbi_monomer_dimer_alp.experiment_support import (
    RecordingLayoutError,
    RetiredOption,
    assert_complete_shard_set,
    chunk_count,
    condition_display,
    discover_cells,
    inspect_recording,
    load_shards,
    merge_validated_shards,
    preflight_recordings,
    read_cell_chunks,
    save_shard,
    shard_by_rank,
)
from srm_and_sbi_monomer_dimer_alp.provenance import code_provenance, finalize_code_provenance
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


def _aggregate_by_kind(estimate, kind_index, cell_arr, kinds, mode, n_params):
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
        kinf = estimate[kmask]
        if mode == "cell-median":
            kcells = cell_arr[kmask]
            out[kind] = np.asarray(
                [np.median(kinf[kcells == c], axis=0) for c in np.unique(kcells)])
        else:  # pooled
            out[kind] = kinf
    return out


def write_experiment_outputs(reporter, args, eval_cfg, draw_spec, array_path: Path,
                             arrays: dict, run_start, persist_arrays: bool = True) -> dict:
    """Validate an experiment product, persist it and write the report + figures.

    ``arrays`` is the complete product (every field of
    ``artifact_schema.STAGE_FIELDS["experiment"]``, ``kinds``, optionally
    ``posterior_samples_cloud``, plus the encoded manifest). It is validated HERE, on entry,
    through :func:`artifact_schema.validate_product` -- the writer is the boundary, so no caller
    can persist or render an invalid product -- and the manifest the report reads is decoded
    from the arrays themselves, never supplied separately. The three point estimates are read
    by their stored fields -- ``map_estimate``, the 0.50 level of ``posterior_quantiles`` located
    through the manifest, and ``posterior_sgm``. With ``persist_arrays=False`` (report-only
    rendering) the arrays on disk are left untouched. Returns the validated manifest.
    """
    manifest = schema.validate_product(arrays, stage="experiment",
                                       source=f"experiment product {array_path.name}")
    tok = short_labels()
    qi = schema.median_level_index(manifest)
    scores = np.asarray(arrays["scores"], dtype=float)
    map_est = np.asarray(arrays["map_estimate"], dtype=float)
    kind_index = np.asarray(arrays["kind_index"]).astype(int)
    cell_of = np.asarray(arrays["cell"]).astype(int)
    chunk_of = np.asarray(arrays["chunk"]).astype(int)
    kinds = [str(k) for k in arrays["kinds"]]
    post_q = np.asarray(arrays["posterior_quantiles"], dtype=float)
    median = post_q[:, :, qi]
    sgm = np.asarray(arrays["posterior_sgm"], dtype=float)
    n_estimates = map_est.shape[0]

    if persist_arrays:
        np.savez_compressed(str(array_path), **arrays)
        print(f"\nExperiment arrays saved to {array_path}")
    else:
        print(f"\nExperiment arrays left as stored at {array_path} (report-only rendering)")

    # ---- Report ----------------------------------------------------------
    # Everything a reader sees carries the scientific condition name (MET-FAB / MET-INLB). The
    # stored `kinds` field keeps the "FAB"/"INLB" tokens: the data-file namespace and the schema
    # downstream analyses key on; presentation prepends the receptor.
    shown = [condition_display(k) for k in kinds]
    agg_desc = ("pooled over (cell x chunk)" if args.aggregation == "pooled"
                else "one point per cell (median over its chunks)")
    by_kind = {}
    for key, arr in ((tok["map"], map_est), (tok["median"], median), (tok["sgm"], sgm)):
        grouped = _aggregate_by_kind(arr, kind_index, cell_of, kinds, args.aggregation,
                                     len(draw_spec))
        by_kind[key] = {condition_display(k): v for k, v in grouped.items()}
    reporter.check("estimates_nonempty", n_estimates > 0,
                   f"{n_estimates} windows estimated over experimental recordings",
                   note="at least one experimental window was estimated.")
    reporter.check("three_point_estimates_present", True,
                   "map_estimate, posterior_quantiles (median at level 0.50) and posterior_sgm "
                   "stored for every window",
                   note="the product contract: no estimate is reported without the other two.")
    reporter.stat("conditions", len(kinds))
    reporter.stat("total_estimates", n_estimates,
                  note="number of (cell, chunk) windows estimated across all conditions.")
    reporter.stat("aggregation", args.aggregation, note=f"report distribution view: {agg_desc}.")
    reporter.stat("artifact_schema", f"v{manifest['artifact_schema_version']} / estimate "
                  f"definitions v{manifest['estimate_definitions_version']}",
                  note="the stored computation contract; the manifest inside the .npz records the "
                       "optimizer settings, draw count, quantile levels, SGM scaling, code and "
                       "checkpoint identity this product was made under.")
    reporter.stat("summary_draws", manifest["n_summary_draws"],
                  note=f"{manifest['draw_label']}s per window that the median and the SGM "
                       f"summarize (pool mode {manifest['pool_mode']}).")
    for kind, name in zip(kinds, shown):
        reporter.stat(f"n[{name}]", int(by_kind[tok["map"]][name].shape[0]),
                      note=f"estimates for condition {name} (stored as '{kind}').")
    if n_estimates:
        reporter.stat("mean_log_prob", float(np.mean(scores)),
                      note="mean log-density at the returned MAP candidate (optimization "
                           "diagnostic; not a calibration/quality metric).")

    reporter.table(
        "Point estimates: definitions", ["key", "stored field", "definition"],
        point_estimate_rows(manifest["pool_mode"], qi),
        note="one row per point estimate (evaluation.POINT_ESTIMATES); every table and figure "
             "below uses these three tokens. The three are read together; no one replaces the "
             "others.")

    for key in (tok["map"], tok["median"], tok["sgm"]):
        headers, rows = experiment_table(draw_spec, by_kind[key], shown)
        extra = ("" if key != tok["map"] else
                 " The gradient ascent is unconstrained, so an outside-prior value happens under "
                 "either pool mode -- --pool-mode bounds the candidate pool, not the optimizer's "
                 "steps -- and is a flow optimum rather than a MAP of the prior-supported posterior.")
        reporter.table(f"{key} theta by condition (log10 units)", headers, rows,
                       note=f"no ground truth for experimental recordings; values are the "
                            f"distribution of the {key} estimate per condition ({agg_desc}). "
                            f"'outside prior' is the share beyond the row's prior box.{extra} The "
                            f"three tables are read together.")
    cmp_headers, cmp_rows = experiment_estimates_compared_table(draw_spec, by_kind, shown)
    reporter.table("Point estimates compared (per parameter, log10 units)",
                   cmp_headers, cmp_rows,
                   note=f"the three point estimates on the same windows ({agg_desc}), columns "
                        f"grouped by statistic with the estimates consecutive: the median over "
                        f"windows, the IQR over windows and the share outside the prior box, each "
                        f"for {tok['map']}, {tok['median']} and {tok['sgm']} (definitions above). A "
                        f"statement about a parameter is read from the row as a whole, never from "
                        f"one column.")
    agr_headers, agr_rows = point_estimate_agreement_table(
        draw_spec, map_est, post_q, sgm,
        groups=[condition_display(kinds[k]) for k in kind_index])
    reporter.table("Point-estimate agreement (per parameter, log10 units)",
                   agr_headers, agr_rows,
                   note="each 'X vs Y' column is the median over windows of the absolute "
                        "difference between the two point estimates (log10). 'MAP outside 90%' is "
                        "the share of windows whose MAP falls outside the central 90% interval of "
                        "the draws. Large gaps with a high outside share establish that the "
                        "optimized candidate and the draw summaries disagree; the cause is not "
                        "decided by this table and needs separate checks.")

    # ---- Within-recording drift: the three point estimates against window position --------
    # The windows of one recording share the same acquisition, so a trend across them is either
    # dynamics or an acquisition confound; measured for all three estimates together, against the
    # per-window interval of the draws.
    if n_estimates:
        grids, _, n_chunks_g = tdk.point_estimate_grids(
            map_est, kind_index, cell_of, chunk_of, len(kinds), post_q, sgm, median_index=qi)
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
                     "changes across recordings. One row per point estimate (MAP, posterior "
                     "median, SGM), read together: the three come from the same draws, so a "
                     "difference between them describes the shape of the density along the "
                     "recording. 'cells over 0.3 dex' is the share whose change exceeds a factor "
                     "of two. The signed-rank p states detectability, not magnitude, and no row "
                     "attributes a cause: the biology block is marginalized here, so an imaging "
                     "trend and a biological one are not separable in this stage alone.")
            if reporter.dump:
                bands_all = tdk.shared_posterior_bands(post_q, kind_index, cell_of, chunk_of,
                                                       len(kinds))
                for ki, kind in enumerate(kinds):
                    med = {name: tdk.window_medians(g)[ki] for name, g in grids.items()}
                    fig = figure_window_drift(
                        keys, labels, med, bands=bands_all[ki],
                        prior_ranges=[para["PRIOR_RANGE"] for para in draw_spec],
                        title=f"{condition_display(kind)}: point estimates against window "
                              f"position (lines = median across recordings; bands = median "
                              f"per-window 50 % and 90 % intervals of the draws)")
                    reporter.save_figure(
                        f"window_drift_{kind}", fig,
                        caption=f"Within-recording drift, condition {condition_display(kind)}: "
                                f"the MAP, posterior median and SGM as the median across "
                                f"recordings at each window, over the per-window central 50 % and "
                                f"90 % intervals of the draws (median across recordings). Dotted "
                                f"red lines are the prior bounds. The three lines share one draw "
                                f"set, so the band is drawn once.")

    if reporter.dump and n_estimates:
        for i, para in enumerate(draw_spec):
            key = para["KEY"]
            label = para.get("LABEL") or key
            prior_range = para["PRIOR_RANGE"]
            values = {name: by_kind[tok["map"]][name][:, i] for name in shown}
            by_kind_post = {}
            for ki, kind in enumerate(kinds):
                m = kind_index == ki
                map_col = map_est[m][:, i:i + 1]                                 # (n, 1)
                q_cols = post_q[m][:, i][:, [qi, qi - 1, qi + 1]]                # (n, 3): med,q25,q75
                by_kind_post[condition_display(kind)] = np.hstack([map_col, q_cols])
            reporter.save_figure(
                f"experiment_{key}",
                figure_experiment_combined(values, by_kind_post, prior_range, label,
                                           seed=args.seed),
                caption=f"{key} ({label}). Left: per-condition distribution of the {tok['map']} "
                        f"({agg_desc}). Right: each window's {tok['median']} +/- IQR of its draws "
                        f"per condition (within-window spread), the {tok['map']} overlaid. The "
                        f"{tok['sgm']} is tabulated above and its gap to the median is in the "
                        f"agreement table.",
            )
        # The point-estimate view for experimental recordings: no truth exists, so the three
        # estimates are laid out per recording over the IQR of the draws (one shared band),
        # recordings ordered by their median value.
        for ki, kind in enumerate(kinds):
            kmask = kind_index == ki
            if not np.any(kmask):
                continue
            for i, para in enumerate(draw_spec):
                key = para["KEY"]
                label = para.get("LABEL") or key
                fig = figure_point_estimates_by_cell(
                    {tok["map"]: map_est[kmask, i], tok["median"]: median[kmask, i],
                     tok["sgm"]: sgm[kmask, i]},
                    post_q[kmask, i, :], cell_of[kmask], para["PRIOR_RANGE"],
                    f"{label} ({condition_display(kind)})")
                if fig is not None:
                    reporter.save_figure(
                        f"point_estimates_{kind}_{key}", fig,
                        caption=f"{key} ({label}), {condition_display(kind)}. The three point "
                                f"estimates per window, one panel each ({tok['map']}, "
                                f"{tok['median']}, {tok['sgm']}): recordings along the x axis "
                                f"ordered by their median value, one point per window, the "
                                f"recording's median of that estimate as the line, and the IQR of "
                                f"the draws (median over the recording's windows of the per-window "
                                f"Q25 and Q75) as the band, drawn once and identical on every panel. "
                                f"The estimate axis is clipped to the prior box widened by 40 %; "
                                f"values beyond it are counted in the tables, not drawn.")

    reporter.summary()
    reporter.write_report()
    print(f"\nTotal elapsed: {time.time() - run_start:.1f}s")
    return manifest


def _save_shard(topo, out_dir: Path, arrays: dict, run_start) -> None:
    """Write this worker's validated partial product (multi-GPU sharded run). A worker that drew
    no cells still writes a valid zero-observation shard, so the merge's rank-coverage check sees
    every rank account for itself."""
    n = int(np.asarray(arrays["scores"]).shape[0])
    path = save_shard(out_dir, topo, arrays, count=n, write_empty=True)
    print(f"\n[rank {topo.rank}/{topo.world_size}] shard saved: {path} "
          f"({n} estimates{'; no cells were assigned to this rank' if n == 0 else ''}) in "
          f"{time.time() - run_start:.1f}s. Run the --merge step once all shards finish.",
          flush=True)


def _merge_shards(reporter, args, eval_cfg, draw_spec, out_dir: Path,
                  array_path: Path, run_start, expected_ids) -> None:
    """Combine every per-shard product into the final report (no estimation). Shards are
    validated and must describe one computation; the merged ``(kind_index, cell, chunk)`` set
    must equal ``expected_ids`` exactly; rank coverage must be complete."""
    shard_paths = load_shards(out_dir)
    if not shard_paths:
        raise SystemExit(
            f"--merge: no shard files (_shard_*_of_*.npz) found in {out_dir}")
    try:
        world_size = assert_complete_shard_set(shard_paths, partial_option=None)
        merged, _manifest, n_used = merge_validated_shards(
            shard_paths, stage="experiment",
            concat_keys=["scores", "map_estimate", "kind_index", "cell", "chunk",
                         "posterior_quantiles", "posterior_sgm"],
            first_keys=["kinds"], optional_concat_keys=["posterior_samples_cloud"],
            expected_ids=expected_ids)
    except (ValueError, schema.SchemaError) as exc:
        raise SystemExit(f"--merge: {exc}")
    reporter.stat("shards_merged", f"{n_used}/{world_size}",
                  note="per-rank shards combined into this report; the merge requires every rank "
                       "and every expected (kind, cell, chunk) window, so this is always complete.")
    print(f"Merged {merged['scores'].shape[0]} estimates from {n_used} shard(s).", flush=True)
    write_experiment_outputs(reporter, args, eval_cfg, draw_spec, array_path,
                             merged, run_start)
    for shard_path in shard_paths:
        shard_path.unlink()
    print(f"Removed {len(shard_paths)} shard file(s).", flush=True)


def _selected_recordings(paths, kinds, cells_by_kind, span, data_bank_root):
    """``[(kind_index, cell, path), ...]`` for every selected recording present on disk. A
    recording missing on disk is not part of the run, matching the estimation loop's SKIP rule."""
    out = []
    for ki, kind in enumerate(kinds):
        for cell in cells_by_kind[kind]:
            tif_path = paths.experiment_video_path(kind, cell, span, data_bank_root)
            if tif_path.exists():
                out.append((ki, int(cell), tif_path))
    return out


def _expected_observations(paths, kinds, cells_by_kind, span, data_bank_root, n_frames,
                           step_frames):
    """The window inventory this run must cover: ``[(kind_index, cell, chunk), ...]`` for every
    recording present on disk, with the chunk count each recording yields under the window/step
    geometry. The frame count comes from :func:`inspect_recording` -- the SAME layout
    interpretation :func:`read_cell_chunks` cuts windows with -- so inventory and windows agree
    by construction; an unsupported layout raises here (no frames are loaded)."""
    expected = []
    for ki, cell, tif_path in _selected_recordings(paths, kinds, cells_by_kind, span,
                                                   data_bank_root):
        n_avail = inspect_recording(tif_path)[0]
        expected.extend((ki, cell, c) for c in range(chunk_count(n_avail, n_frames, step_frames)))
    return expected


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

    show = args.verbose or args.debug or args.debug_dump
    verbose_deep = args.debug or args.debug_dump
    debug_log = args.debug or args.debug_dump

    posterior_samples = args.posterior_samples or eval_cfg.posterior_samples
    dump_samples = args.dump_posterior_samples

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
    print(f"  point estimates           : map, median, sgm  (all three, always; "
          f"{draw_label(pool_mode)}s summarized)")
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
    for line in optimizer_summary_lines(opt_contract):
        print(f"  {line}")
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
        try:
            expected = _expected_observations(paths, kinds, cells_by_kind, span, data_bank_root,
                                              n_frames, step_frames)
        except RecordingLayoutError as exc:
            raise SystemExit(f"--merge: {exc}")
        _merge_shards(reporter, args, eval_cfg, spec.draw_spec, out_dir, array_path, run_start,
                      expected)
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
    # launcher assigned and this rank's execution attempt. The implementation records are
    # compared again at write time; a file edited during the run makes the product fail closed
    # at publication (it is not written).
    code_at_start = code_provenance()
    checkpoint_sha256 = posterior.weights_sha256
    run_ident = schema.run_identity(product_label, distributed=topo.is_distributed)
    exec_ident = schema.execution_identity()   # this attempt (its Slurm job, or None): recorded, never compared

    # ---- Preflight every selected recording's TIFF layout BEFORE estimation -----------------
    # One layout rule (experiment_support.inspect_recording) serves the inventory and the reader;
    # an unsupported file is reported now, for all files at once, not near the end of a long run.
    selected = _selected_recordings(paths, kinds, cells_by_kind, span, data_bank_root)
    try:
        recording_shapes = preflight_recordings([p for _, _, p in selected])
    except RecordingLayoutError as exc:
        raise SystemExit(str(exc))
    print(f"Preflight: {len(recording_shapes)} recording(s) inspected; layouts supported.",
          flush=True)

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

    # ---- The three point estimates over (kind, cell, chunk) --------------
    scores, map_est, post_quantiles = [], [], []
    post_sgm = []                # SGM of each window's draws
    sgm_scale = prior_scale(spec.draw_spec)
    post_samples = []            # raw draws per window, only under --dump-posterior-samples
    kind_index, cell_of, chunk_of = [], [], []
    shard_note = (f" [shard rank {topo.rank}/{topo.world_size}: {len(my_work)} of "
                  f"{len(flat_work)} cells]" if topo.is_distributed else "")
    log_progress(progress_fh,
                 f"START Experiment MAP: {my_estimates} estimates{shard_note} "
                 f"({len(kinds)} kinds, {n_chunks} chunks/video; pool={theta_prex_size}, "
                 f"elites={elite_prex_size}, steps={numb_steps}, "
                 f"patience={eval_cfg.optimizer_patience}/{eval_cfg.scheduler_patience}, "
                 f"lr={lr:g}..{eval_cfg.learning_rate_minimum:g} in pool-IQR units, "
                 f"tolerance={tolerance:g} nats; "
                 f"estimates=map,median,sgm over {posterior_samples} draws).")
    map_stops = Counter()
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
                    score, theta_log, map_info = map_estimate(
                        posterior, chunk, device, vista_device,
                        theta_prex_size, eval_cfg.theta_prex_batch_size,
                        eval_cfg.score_prex_batch_size, elite_prex_size,
                        numb_steps, eval_cfg.optimizer_patience,
                        eval_cfg.scheduler_patience, show_progress_steps,
                        eval_cfg.learning_rate_minimum, eval_cfg.learning_rate_factor,
                        lr, tolerance, pool_mode=pool_mode, show=show,
                        verbose=verbose_deep, log_fn=step_log, return_info=True,
                    )
                    map_stops[map_info["stop"]] += 1
                    scores.append(score)
                    map_est.append(theta_log)
                    # The two draw-derived estimates, from ONE draw set per window.
                    out = posterior_summary(
                        posterior, chunk, device, vista_device,
                        posterior_samples, eval_cfg.theta_prex_batch_size,
                        pool_mode=pool_mode, quantiles=QUANTILE_LEVELS,
                        return_samples=dump_samples, return_sgm=True, sgm_scale=sgm_scale)
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
                                 f"log_prob={score:+.3f} | "
                                 f"stop={map_info['stop']}@{map_info['steps']} | "
                                 f"elapsed={elapsed:.1f}s | avg={avg:.1f}s | ETA={eta:.0f}s")
                    if debug_log:
                        log_file_only(progress_fh,
                                      f"pool IQR [LOG] {_theta_repr(map_info['scale'], 4)}")
                        log_file_only(progress_fh, f"inferred [LOG] {_theta_repr(theta_log)}")
                log_progress(progress_fh, f"{kind} cell {cell} complete "
                                          f"({n_cell_chunks} chunks in "
                                          f"{time.time() - cell_start:.1f}s).")
        log_progress(progress_fh, f"DONE: {done}/{my_estimates} estimates in "
                                  f"{time.time() - loop_start:.1f}s.")
        log_progress(progress_fh, "MAP stops: " + (", ".join(
            f"{k} {map_stops[k]}" for k in STOP_REASONS if map_stops[k]) or "none"))
    finally:
        if progress_fh is not None:
            progress_fh.close()

    # ---- Assemble, validate, write ---------------------------------------
    manifest = schema.build_manifest(
        stage="experiment", parameter_keys=spec.parameter_keys,
        coordinate_transform="parameterization.to_flow (log10 for log rows, linear rows as-is)",
        pool_mode=pool_mode, draw_label=draw_label(pool_mode), n_summary_draws=posterior_samples,
        quantile_levels=QUANTILE_LEVELS, sgm_scale=sgm_scale, sgm_coordinates="estimator",
        optimizer=opt_contract,
        code=finalize_code_provenance(code_at_start), checkpoint_sha256=checkpoint_sha256,
        run_identity=run_ident, execution=exec_ident, seed_policy=schema.seed_policy(args.seed),
        window_geometry=schema.window_geometry(n_frames=n_frames, step_frames=step_frames,
                                               span_frames=exp_frames),
        condition_labels=kinds,
        # Raw-draw storage is a run-level setting: every shard of the run declares it the same way.
        stored_optional_fields=(["posterior_samples_cloud"] if dump_samples else []),
        n_observations=len(scores), rank=topo.rank if topo.is_distributed else None,
        world_size=topo.world_size if topo.is_distributed else None,
        estimate_definitions_version=ESTIMATE_DEFINITIONS_VERSION)
    if scores:
        arrays = dict(
            scores=np.asarray(scores, dtype=float), map_estimate=np.asarray(map_est, dtype=float),
            kind_index=np.asarray(kind_index, dtype=np.int64),
            cell=np.asarray(cell_of, dtype=np.int64),
            chunk=np.asarray(chunk_of, dtype=np.int64), kinds=np.asarray(kinds),
            posterior_quantiles=np.asarray(post_quantiles, dtype=float),
            posterior_sgm=np.asarray(post_sgm, dtype=float))
        if dump_samples:
            arrays["posterior_samples_cloud"] = np.asarray(post_samples, dtype=np.float32)
    else:
        # A rank that drew no windows: correctly shaped (0, D) / (0, D, Q) / (0,) arrays, so the
        # shard is valid and mergeable (an empty list converted to an array would be (0,)).
        arrays = schema.empty_product_arrays("experiment", n_parameters=len(spec.parameter_keys),
                                             n_levels=len(QUANTILE_LEVELS),
                                             run_fields={"kinds": np.asarray(kinds)})
        if dump_samples:
            arrays["posterior_samples_cloud"] = np.empty(
                (0, posterior_samples, len(spec.parameter_keys)), dtype=np.float32)
    arrays[schema.MANIFEST_KEY] = schema.encode_manifest(manifest)
    # Validated BEFORE anything is written. An empty product is admitted only as a shard of a
    # sharded run; a single-process run over nothing is an error, not a report.
    try:
        schema.validate_product(arrays, stage="experiment",
                                source=f"rank {topo.rank} product", allow_empty=topo.is_distributed)
        if not topo.is_distributed:
            expected = _expected_observations(paths, kinds, cells_by_kind, span,
                                              data_bank_root, n_frames, step_frames)
            schema.assert_unique_observations(schema.observation_ids(arrays, "experiment"),
                                              source="product", expected=expected)
    except schema.SchemaError as exc:
        raise SystemExit(f"Experiment product failed the contract: {exc}")
    if topo.is_distributed:
        _save_shard(topo, out_dir, arrays, run_start)
    else:
        write_experiment_outputs(reporter, args, eval_cfg, spec.draw_spec, array_path,
                                 arrays, run_start)


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
        "--summary", action=RetiredOption,      # retired in 0.1.16: an explicit error, hidden from --help
    )
    parser.add_argument(
        "--dump-posterior-samples", action="store_true",
        help="Additionally store the raw draws for every window, as posterior_samples_cloud "
             "(n_windows, --posterior-samples, D) in the saved npz. Quantiles keep only "
             "per-parameter marginals; the raw draws keep the joint structure, which is what "
             "pooling across a recording needs. Adds roughly n_windows * samples * D * 4 bytes.",
    )
    parser.add_argument(
        "--posterior-samples", type=int, default=None,
        help=f"Draws per window summarized by the quantiles, the median and the SGM "
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
                        help="Initial Adam learning rate of the MAP ascent, in pool-IQR units "
                             f"(default {eval_cfg.learning_rate}).")
    parser.add_argument("--verbose", action="store_true",
                        help="Rich per-chunk console diagnostics (see Evaluation).")
    parser.add_argument("--debug", action="store_true",
                        help="Implies --verbose; deeper console + per-step detail in progress.log.")
    parser.add_argument("--debug-dump", action="store_true",
                        help="Implies --debug; tees the console transcript to "
                             "Labor/Debug/<run>/Experiment/console.log.")
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
