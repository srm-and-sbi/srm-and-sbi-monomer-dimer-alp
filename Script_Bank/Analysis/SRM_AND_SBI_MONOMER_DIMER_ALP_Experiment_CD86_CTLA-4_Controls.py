"""Special-scope entry point: MET-posterior MAP estimation on the CD86 / CTLA-4
control receptors (preliminary, cross-receptor guidance).

A NEAR-VERBATIM CLONE of the MET ``Experiment.py`` stage. It applies the SAME
seed-then-optimize ``map_estimate`` to real single-particle-tracking videos, but of
two *control* receptors of known oligomeric state -- CD86 (monomeric) and CTLA-4 (a
constitutive dimer) -- reusing the MET-trained posterior with NO retraining. It
differs from the MET stage in ONLY the dataset folder it reads, the output directory
it writes, and the ``--kinds`` default (plus this docstring); every estimation and
reporting behavior is identical, so the canonical MET ``Experiment.py`` and its
outputs are never touched. Lives in ``Script_Bank/Analysis``; NOT a canonical stage.
Both conditions run in one pass; ``kind_index`` differentiates them in the output.

CAVEAT (full reasoning in the companion analysis): the control data use an
EXCHANGEABLE SiR-S5 label whose blinking corrupts the brightness cue the posterior
relies on to separate monomer (1 dye) from dimer (2 dyes), and the true monomer/dimer
diffusion gap is small (~13%). So the per-species split (D_A/D_B/counts) is NOT a
trustworthy oligomeric decomposition here -- the robust read-out is the diffusion
SCALE (compared against the reported mobile diffusion), not monomer-vs-dimer identity.

Inputs (real microscopy, copied into the data bank by the user):
    <data_bank>/Experiment/SPT_Data_CD86_CTLA-4_CONTROLS_S-BIAD1369/
        Experiment_{CD86|CTLA-4}_Cell_{n}_{span}S_RAW.tif   -- 16-bit raw SPT video.
    EMBL-EBI BioImage Archive S-BIAD1369 (Catapano et al., Angew. Chem. Int. Ed.
    2025, 64, e202413117).

Outputs (under <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_MAP_Experiment_CD86_CTLA-4_CONTROLS/):
    report.md, figures/, <...>.npz (map_estimate, posterior_quantiles, posterior_sgm, scores,
    kind/cell/chunk, manifest_json), progress.log

Usage (multi-GPU, both conditions in one pass; then merge the shards):
    MACHINE_PROFILE=<profile> torchrun --nproc_per_node=4 \\
        SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py \\
        --total-time-seconds 2.0 --kinds CD86,CTLA-4 --pool-mode unrestricted
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py \\
        --total-time-seconds 2.0 --kinds CD86,CTLA-4 --merge
"""

import argparse
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch._dynamo

from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_monomer_dimer_alp.evaluation import (
    ESTIMATE_DEFINITIONS_VERSION, QUANTILE_LEVELS,
    draw_label,
    map_estimate,
    optimizer_contract,
    experiment_table, experiment_estimates_compared_table,
    point_estimate_agreement_table, point_estimate_rows,
    posterior_summary, prior_scale, short_labels,
    _theta_repr,
)
from srm_and_sbi_monomer_dimer_alp import artifacts
from srm_and_sbi_monomer_dimer_alp.experiment_support import (
    RecordingLayoutError, RetiredOption, assert_complete_shard_set, chunk_count,
    inspect_recording, merge_validated_shards, preflight_recordings, read_cell_chunks, save_shard)
from srm_and_sbi_monomer_dimer_alp.provenance import code_provenance, finalize_code_provenance
from srm_and_sbi_monomer_dimer_alp.inference_support import resolve_topology
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, PARAMETERIZATION, PARAMETER_KEYS, RunTiming, build_prior
from srm_and_sbi_monomer_dimer_alp.utils import console_log_context
from srm_and_sbi_monomer_dimer_alp.visualization_inference import figure_experiment_combined

# --- Control-receptor identity: the ONLY substantive divergence from MET Experiment.py.
# This special-scope clone reads a different dataset folder and writes a distinct output
# directory; everything else is byte-identical to the MET Experiment stage, so MET's
# files and outputs are never touched. Kind labels CD86 / CTLA-4 name the receptors
# themselves (the axis that varies here); the receptor meaning is carried in the analysis.
CONTROLS_EXPERIMENT_SUBDIR = "Experiment/SPT_Data_CD86_CTLA-4_CONTROLS_S-BIAD1369"
CONTROLS_RECOVERY_PATTERN = "{project_alias}_{timing_label}_MAP_Experiment_CD86_CTLA-4_CONTROLS"


def _discover_cells(experiment_dir: Path, kind: str, span: int) -> list:
    """Return sorted cell indices available on disk for a given kind."""
    cells = []
    for path in experiment_dir.glob(f"Experiment_{kind}_Cell_*_{span}S_RAW.tif"):
        match = re.search(rf"Cell_(\d+)_{span}S_RAW", path.name)
        if match:
            cells.append(int(match.group(1)))
    return sorted(cells)


def _aggregate_by_kind(estimate, kind_index, cell_arr, kinds, mode, n_params):
    """Group inferred theta by condition for the report, per ``mode``.

    ``"pooled"``      -- every (cell, chunk) estimate is one sample, pooled per
                         kind (mixes temporal + biological variation).
    ``"cell-median"`` -- collapse each cell to its median across its chunks first,
                         so each kind's distribution is one sample per cell
                         (biological spread; within-cell temporal noise averaged out).

    Returns ``{kind: (N, D) array}`` of inferred log10 theta. The raw per-(cell,
    chunk) arrays are saved to the .npz regardless, so either view is reproducible.
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


def write_experiment_outputs(reporter, args, eval_cfg, array_path: Path,
                             arrays: dict, run_start, persist_arrays: bool = True) -> dict:
    """Validate the product on entry, persist it and write the report + figures (the same
    three-estimate contract as the MET Experiment stage: ``map_estimate``, the 0.50 level of
    ``posterior_quantiles`` located through the manifest, ``posterior_sgm``). The manifest is
    decoded from the arrays, never supplied separately; the writer is the validation boundary.
    ``persist_arrays=False`` renders the report without touching the arrays on disk. Returns
    the validated manifest."""
    manifest = schema.validate_product(arrays, stage="experiment",
                                       source=f"controls product {array_path.name}")
    tok = short_labels()
    qi = schema.median_level_index(manifest)
    scores = np.asarray(arrays["scores"], dtype=float)
    map_est = np.asarray(arrays["map_estimate"], dtype=float)
    kind_index = np.asarray(arrays["kind_index"]).astype(int)
    cell_of = np.asarray(arrays["cell"]).astype(int)
    kinds = [str(k) for k in arrays["kinds"]]
    post_q = np.asarray(arrays["posterior_quantiles"], dtype=float)
    median = post_q[:, :, qi]
    sgm = np.asarray(arrays["posterior_sgm"], dtype=float)
    n_estimates = map_est.shape[0]

    if persist_arrays:
        np.savez_compressed(str(array_path), **arrays)
        print(f"\nExperiment arrays saved to {array_path}")
    else:
        print(f"\nReport-only rendering: arrays at {array_path} left untouched.")

    # ---- Report ----------------------------------------------------------
    agg_desc = ("pooled over (cell x chunk)" if args.aggregation == "pooled"
                else "one point per cell (median over its chunks)")
    by_kind = {key: _aggregate_by_kind(arr, kind_index, cell_of, kinds, args.aggregation,
                                       len(PARAMETERIZATION))
               for key, arr in ((tok["map"], map_est), (tok["median"], median), (tok["sgm"], sgm))}
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
                  note="the stored computation contract (optimizer settings, draw count, quantile "
                       "levels, SGM scaling, code and checkpoint identity) is in the .npz manifest.")
    reporter.stat("summary_draws", manifest["n_summary_draws"],
                  note=f"{manifest['draw_label']}s per window that the median and the SGM "
                       f"summarize (pool mode {manifest['pool_mode']}).")
    for kind in kinds:
        reporter.stat(f"n[{kind}]", int(by_kind[tok["map"]][kind].shape[0]),
                      note=f"estimates for condition {kind}.")
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
        headers, rows = experiment_table(PARAMETERIZATION, by_kind[key], kinds)
        reporter.table(f"{key} theta by condition (log10 units)", headers, rows,
                       note=f"no ground truth for experimental recordings; values are the "
                            f"distribution of the {key} estimate per condition ({agg_desc}). The "
                            f"three tables are read together.")
    cmp_headers, cmp_rows = experiment_estimates_compared_table(PARAMETERIZATION, by_kind, kinds)
    reporter.table("Point estimates compared (per parameter, log10 units)", cmp_headers, cmp_rows,
                   note=f"the three point estimates on the same windows ({agg_desc}), columns "
                        f"grouped by statistic with the estimates consecutive, each for "
                        f"{tok['map']}, {tok['median']} and {tok['sgm']} (definitions above). A "
                        f"statement about a parameter is read from the row as a whole.")
    agr_headers, agr_rows = point_estimate_agreement_table(
        PARAMETERIZATION, map_est, post_q, sgm, groups=[kinds[k] for k in kind_index])
    reporter.table("Point-estimate agreement (per parameter, log10 units)", agr_headers, agr_rows,
                   note="median over windows of the absolute difference between two point "
                        "estimates (log10), and the share of windows whose MAP falls outside the "
                        "central 90% interval of the draws.")

    if reporter.dump and n_estimates:
        for i, para in enumerate(PARAMETERIZATION):
            key = para["KEY"]
            label = para.get("LABEL") or key
            prior_range = para["PRIOR_RANGE"]
            values = {kind: by_kind[tok["map"]][kind][:, i] for kind in kinds}
            by_kind_post = {}
            for ki, kind in enumerate(kinds):
                m = kind_index == ki
                map_col = map_est[m][:, i:i + 1]                                 # (n, 1)
                q_cols = post_q[m][:, i][:, [qi, qi - 1, qi + 1]]                # (n, 3): med,q25,q75
                by_kind_post[kind] = np.hstack([map_col, q_cols])                # (n, 4)
            reporter.save_figure(
                f"experiment_{key}",
                figure_experiment_combined(values, by_kind_post, prior_range, label,
                                           seed=args.seed),
                caption=f"{key} ({label}). Left: per-condition distribution of the {tok['map']} "
                        f"({agg_desc}). Right: each window's {tok['median']} +/- IQR of its draws "
                        f"per condition, the {tok['map']} overlaid. The {tok['sgm']} is tabulated "
                        f"above and its gap to the median is in the agreement table.",
            )

    reporter.summary()
    reporter.write_report()
    print(f"\nTotal elapsed: {time.time() - run_start:.1f}s")
    return manifest


def _save_shard(topo, out_dir: Path, arrays: dict, run_start) -> None:
    """Write this worker's validated partial product (multi-GPU sharded run). A worker that drew
    no cells still writes a valid zero-observation shard, so the merge's rank-coverage check
    sees every rank account for itself."""
    n = int(np.asarray(arrays["scores"]).shape[0])
    path = save_shard(out_dir, topo, arrays, count=n, write_empty=True)
    print(f"\n[rank {topo.rank}/{topo.world_size}] shard saved: {path} "
          f"({n} estimates{'; no cells were assigned to this rank' if n == 0 else ''}) in "
          f"{time.time() - run_start:.1f}s. Run the --merge step once all shards finish.",
          flush=True)


def _merge_shards(reporter, args, eval_cfg, out_dir: Path, array_path: Path, run_start,
                  expected_ids) -> None:
    """Combine every validated per-shard product into the final report (no estimation); the
    shards must describe one computation and cover exactly the expected windows."""
    shard_paths = sorted(out_dir.glob("_shard_*_of_*.npz"))
    if not shard_paths:
        raise SystemExit(
            f"--merge: no shard files (_shard_*_of_*.npz) found in {out_dir}")
    try:
        world_size = assert_complete_shard_set(shard_paths, partial_option=None)
        merged, _manifest, n_used = merge_validated_shards(
            shard_paths, stage="experiment",
            concat_keys=["scores", "map_estimate", "kind_index", "cell", "chunk",
                         "posterior_quantiles", "posterior_sgm"],
            first_keys=["kinds"], expected_ids=expected_ids)
    except (ValueError, schema.SchemaError) as exc:
        raise SystemExit(f"--merge: {exc}")
    reporter.stat("shards_merged", f"{n_used}/{world_size}",
                  note="per-rank shards combined; the merge requires every rank and every "
                       "expected (kind, cell, chunk) window.")
    print(f"Merged {merged['scores'].shape[0]} estimates from {n_used} shard(s).", flush=True)
    write_experiment_outputs(reporter, args, eval_cfg, array_path, merged, run_start)
    for shard_path in shard_paths:
        shard_path.unlink()
    print(f"Removed {len(shard_paths)} shard file(s).", flush=True)


def _selected_recordings(experiment_dir, paths, kinds, cells_by_kind, span):
    """``[(kind_index, cell, path), ...]`` for every selected recording present on disk."""
    out = []
    for ki, kind in enumerate(kinds):
        for cell in cells_by_kind[kind]:
            tif_path = experiment_dir / paths.experiment_pattern.format(kind=kind, cell=cell,
                                                                        span=span)
            if tif_path.exists():
                out.append((ki, int(cell), tif_path))
    return out


def _expected_observations(experiment_dir, paths, kinds, cells_by_kind, span, n_frames,
                           step_frames):
    """``[(kind_index, cell, chunk), ...]`` for every recording on disk, with the chunk count it
    yields under the window/step geometry. The frame count comes from
    ``experiment_support.inspect_recording`` -- the same layout rule ``read_cell_chunks`` cuts
    windows with -- so inventory and windows agree by construction (no frames loaded)."""
    expected = []
    for ki, cell, tif_path in _selected_recordings(experiment_dir, paths, kinds, cells_by_kind,
                                                   span):
        n_avail = inspect_recording(tif_path)[0]
        expected.extend((ki, cell, c) for c in range(chunk_count(n_avail, n_frames, step_frames)))
    return expected


def main(args: argparse.Namespace) -> None:
    """Run MAP estimation over the experimental videos per the CLI args."""
    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = PARAMETERS.paths.with_condition(args.condition)   # the MET estimator's condition namespace
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
    estimator_path = paths.estimator_path(data_bank_root, timing_label)
    experiment_dir = data_bank_root / CONTROLS_EXPERIMENT_SUBDIR
    out_dir = (data_bank_root / paths.posit_subdir /
               CONTROLS_RECOVERY_PATTERN.format(
                   project_alias=paths.project_alias, timing_label=timing_label))
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

    posterior_samples = args.posterior_samples or eval_cfg.posterior_samples

    # Video geometry: model-length windows stepped across each long recording.
    # The window length equals the synthetic (training) video length, keeping the
    # chunks fully compatible with the model. The step between consecutive windows
    # is --chunk-step-seconds: an integer that divides the window and is <= it
    # (1 s -> maximal overlap; window -> non-overlapping). Smaller steps yield more
    # (overlapping) chunks per recording.
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
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    explicit_cells = ([int(c) for c in args.cells.split(",")] if args.cells else None)
    cells_by_kind = {}
    for kind in kinds:
        cells = explicit_cells if explicit_cells is not None else _discover_cells(
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
            discovered = _discover_cells(experiment_dir, kind, span)
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
        run_label=f"{paths.project_alias}_{timing_label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    # --merge: combine the per-shard arrays from a multi-GPU sharded run into the
    # final report, then exit (no estimation, no GPU, no posterior needed).
    if args.merge:
        try:
            expected = _expected_observations(experiment_dir, paths, kinds, cells_by_kind, span,
                                              n_frames, step_frames)
        except RecordingLayoutError as exc:
            raise SystemExit(f"--merge: {exc}")
        _merge_shards(reporter, args, eval_cfg, out_dir, array_path, run_start, expected)
        return

    reporter.check_file("estimator artifact", estimator_path)

    topo = resolve_topology()
    device = topo.device
    vista_device = torch.device("cpu")
    posterior = artifacts.load_estimator(estimator_path, device=str(device),
                                          expected_parameter_keys=PARAMETER_KEYS)
    posterior.posterior_estimator.to(device)
    if device.type == "cuda":
        # Rebuild the prior on THIS worker's device for bounded rejection sampling.
        # Under multi-GPU sharding each rank binds its own cuda:local_rank, so reusing
        # the pickled _prior (saved on cuda:0) would mix devices; rebuilding is
        # equivalent on the single-GPU path. Mirrors Evaluation.py.
        posterior.prior = build_prior(device=str(device))

    # ---- Provenance captured at STARTUP (code as loaded, checkpoint actually loaded, the
    # launcher's invocation identity, this rank's execution attempt); the implementation records
    # are compared again at write time and a change makes the product fail closed at publication.
    code_at_start = code_provenance()
    checkpoint_sha256 = posterior.weights_sha256
    run_ident = schema.run_identity(out_dir.name, distributed=topo.is_distributed)
    exec_ident = schema.execution_identity()   # this attempt (its Slurm job, or None): recorded, never compared

    # ---- Preflight every selected recording's TIFF layout BEFORE estimation ---------------
    selected = _selected_recordings(experiment_dir, paths, kinds, cells_by_kind, span)
    try:
        recording_shapes = preflight_recordings([p for _, _, p in selected])
    except RecordingLayoutError as exc:
        raise SystemExit(str(exc))
    print(f"Preflight: {len(recording_shapes)} recording(s) inspected; layouts supported.",
          flush=True)

    # ---- Output dir + progress log + stale-figure clear ------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only rank 0 / a single worker clears the shared figures dir (avoid a race).
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

    # Under sharding all workers share out_dir; only rank 0 writes the progress
    # log (concurrent truncating writers would clobber each other). Other ranks
    # still echo progress to their own stdout (captured in the Slurm log).
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
    # Flatten the (kind, cell) work items and split across workers round-robin
    # (a single worker takes them all). Each cell's chunks stay on one worker; the
    # report aggregates by kind, so per-shard results merge order-independently.
    flat_work = [(ki, cell) for ki, kind in enumerate(kinds)
                 for cell in cells_by_kind[kind]]
    my_work = set(w for i, w in enumerate(flat_work)
                  if i % topo.world_size == topo.rank)
    my_estimates = len(my_work) * n_chunks

    # ---- The three point estimates over (kind, cell, chunk) --------------
    scores, map_est, post_quantiles, post_sgm = [], [], [], []
    sgm_scale = prior_scale(PARAMETERIZATION)
    kind_index, cell_of, chunk_of = [], [], []
    shard_note = (f" [shard rank {topo.rank}/{topo.world_size}: {len(my_work)} of "
                  f"{len(flat_work)} cells]" if topo.is_distributed else "")
    log_progress(progress_fh,
                 f"START Experiment MAP: {my_estimates} estimates{shard_note} "
                 f"({len(kinds)} kinds, {n_chunks} chunks/video; pool={theta_prex_size}, "
                 f"elites={elite_prex_size}, steps={numb_steps}; "
                 f"estimates=map,median,sgm over {posterior_samples} draws).")
    loop_start = time.time()
    done = 0
    try:
        for ki, kind in enumerate(kinds):
            for cell in cells_by_kind[kind]:
                if (ki, cell) not in my_work:
                    continue
                tif_path = experiment_dir / paths.experiment_pattern.format(
                    kind=kind, cell=cell, span=span)
                if not tif_path.exists():
                    log_progress(progress_fh, f"SKIP {kind} cell {cell}: file missing "
                                              f"({tif_path.name}).")
                    continue
                chunks = read_cell_chunks(tif_path, n_frames, step_frames)   # shared layout rule
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
                    map_est.append(theta_log)
                    # The two draw-derived estimates, from ONE draw set per window.
                    summary, sgm_vec = posterior_summary(
                        posterior, chunk, device, vista_device,
                        posterior_samples, eval_cfg.theta_prex_batch_size,
                        pool_mode=pool_mode, quantiles=QUANTILE_LEVELS,
                        return_sgm=True, sgm_scale=sgm_scale)
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

    # ---- Assemble, validate, write ---------------------------------------
    manifest = schema.build_manifest(
        stage="experiment", parameter_keys=PARAMETER_KEYS,
        coordinate_transform="parameterization.to_flow (log10 for log rows, linear rows as-is)",
        pool_mode=pool_mode, draw_label=draw_label(pool_mode), n_summary_draws=posterior_samples,
        quantile_levels=QUANTILE_LEVELS, sgm_scale=sgm_scale, sgm_coordinates="estimator",
        optimizer=optimizer_contract(eval_cfg, learning_rate=lr, tolerance=tolerance,
                                     theta_prex_size=theta_prex_size,
                                     elite_prex_size=elite_prex_size, numb_steps=numb_steps,
                                     pool_mode=pool_mode),
        code=finalize_code_provenance(code_at_start), checkpoint_sha256=checkpoint_sha256,
        run_identity=run_ident, execution=exec_ident, seed_policy=schema.seed_policy(args.seed),
        window_geometry=schema.window_geometry(n_frames=n_frames, step_frames=step_frames,
                                               span_frames=exp_frames),
        condition_labels=kinds, stored_optional_fields=[],
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
    else:
        arrays = schema.empty_product_arrays("experiment", n_parameters=len(PARAMETER_KEYS),
                                             n_levels=len(QUANTILE_LEVELS),
                                             run_fields={"kinds": np.asarray(kinds)})
    arrays[schema.MANIFEST_KEY] = schema.encode_manifest(manifest)
    try:
        schema.validate_product(arrays, stage="experiment",
                                source=f"rank {topo.rank} product", allow_empty=topo.is_distributed)
        if not topo.is_distributed:
            expected = _expected_observations(experiment_dir, paths, kinds, cells_by_kind,
                                              span, n_frames, step_frames)
            schema.assert_unique_observations(schema.observation_ids(arrays, "experiment"),
                                              source="product", expected=expected)
    except schema.SchemaError as exc:
        raise SystemExit(f"Experiment product failed the contract: {exc}")
    if topo.is_distributed:
        _save_shard(topo, out_dir, arrays, run_start)
    else:
        write_experiment_outputs(reporter, args, eval_cfg, array_path, arrays, run_start)


def parse_args(argv=None) -> argparse.Namespace:
    """Construct the CLI parser and parse argv."""
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
        "--total-time-seconds", type=float,
        required=True,
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
        "--kinds", type=str, default="CD86,CTLA-4",
        help="Comma-separated control receptors run together in one pass "
             "(default: 'CD86,CTLA-4'); differentiated in the output by kind_index.",
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
        "--seed", type=int, default=None,
        help="Master RNG seed (PyTorch + numpy + Python random). Default None "
             "-> non-deterministic (consistent with generation); pass an int for a "
             "reproducible run.",
    )
    parser.add_argument(
        "--pool-mode", choices=("bounded", "unrestricted"), default=None,
        help=f"Candidate-pool sampler (default: {eval_cfg.pool_mode}). See Evaluation.py.",
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
                        help="Rich per-chunk console diagnostics (see Evaluation.py).")
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration and inputs, print what would be read/written, then exit "
                             "without running the stage (no GPU, no compute). Use before a queue submission or a long local run.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args(sys.argv[1:])
    with console_log_context(cli_args, "Experiment"):
        main(cli_args)
