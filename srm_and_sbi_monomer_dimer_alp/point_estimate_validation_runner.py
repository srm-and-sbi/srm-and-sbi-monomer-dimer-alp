"""Shared engine of the point-estimate validation program (phase 1: the median reference).

For a FIXED checkpoint, a FIXED list of synthetic EVAL observations and bounded posterior sampling,
it draws, per observation, five independent repeats at the production draw count and one larger
reference run -- each from its own random stream -- through the production ``posterior_summary``
(``return_samples=True``, ``return_sgm=False``: no quadratic-cost medoid over the reference draws),
stores every draw with its quantiles in a separate validation artifact (never a schema-1 stage
product), and writes a report of the phase-1 checks defined in ``point_estimate_validation``.

It is a validation utility, not a canonical stage: it is never wired into the stage dispatcher.
A multi-GPU run shards observations across ranks (``i % world_size == rank``); ``--merge``
combines the shards, requires the exact frozen observation list, and writes the report.

``--pilot N`` selects N recordings deterministically (half dim, half bright; see
``point_estimate_validation.select_pilot``) to validate the utility and to measure throughput and
memory. A pilot's report carries no acceptance verdict.
"""
from __future__ import annotations

import argparse
import csv
import resource
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import artifact_schema as schema
from . import artifacts
from . import point_estimate_validation as pev
from .diagnostics import DiagnosticReporter
from .evaluation import QUANTILE_LEVELS, draw_label, posterior_summary
from .experiment_support import assert_complete_shard_set, load_shards, shard_path
from .inference_support import resolve_topology
from .io import load_data, load_theta_set, theta_set_status
from .parameterization import PARAMETERS, RunTiming, to_flow
from .provenance import IMPLEMENTATION_FILES, code_provenance, file_sha256, finalize_code_provenance
from .workflow import WorkflowConfig

#: Files whose contents determine this utility's numbers: the estimate implementation plus the
#: utility itself. Hashed at startup and again at write time.
VALIDATION_FILES = IMPLEMENTATION_FILES + (
    "srm_and_sbi_monomer_dimer_alp/point_estimate_validation.py",
    "srm_and_sbi_monomer_dimer_alp/point_estimate_validation_runner.py",
    "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Point_Estimate_Validation.py",
)
REPORT_EXCEEDANCE_ROWS = 100      # exceedances listed in the report; the full list goes to a CSV


@dataclass(frozen=True)
class _ValidationSpec:
    """Per-workflow specializations, resolved from the workflow config."""
    draw_spec: list
    parameter_keys: list
    build_prior: Callable


def _validation_spec(cfg: WorkflowConfig) -> _ValidationSpec:
    m = cfg.param_module
    if cfg.tag == "detector":
        return _ValidationSpec(draw_spec=m.DETECTOR_PARAMETERIZATION,
                               parameter_keys=[e["KEY"] for e in m.DETECTOR_PARAMETERIZATION],
                               build_prior=m.build_prior)
    return _ValidationSpec(draw_spec=m.PARAMETERIZATION, parameter_keys=list(m.PARAMETER_KEYS),
                           build_prior=m.build_prior)


def _frozen_observations(theta_paths, spec, condition, max_sims):
    """The frozen observation list: every simulation of every listed task (capped per task by
    ``max_sims`` when positive), in task-then-sim order, with its truth in estimator coordinates."""
    task_index, sim_index, truth = [], [], []
    for task, path in theta_paths.items():
        theta = np.asarray(load_theta_set(path, spec.draw_spec, condition=condition)[:], dtype=float)
        n = theta.shape[0] if max_sims <= 0 else min(theta.shape[0], max_sims)
        task_index.extend([task] * n)
        sim_index.extend(range(n))
        truth.append(to_flow(theta[:n], spec.draw_spec))
    return (np.asarray(task_index, dtype=np.int64), np.asarray(sim_index, dtype=np.int64),
            np.concatenate(truth, axis=0))


def _environment() -> dict:
    """Library versions the draws were produced with (informational; not compared across shards)."""
    import platform
    import sbi
    return {"python": platform.python_version(), "torch": torch.__version__, "sbi": sbi.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda}


def _peak_bytes(device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024     # Linux: kilobytes


def run_point_estimate_validation(cfg: WorkflowConfig, args: argparse.Namespace) -> int:
    """Run (or merge, or dry-run) phase 1 of the point-estimate validation."""
    if args.phase not in pev.PHASES:
        raise SystemExit(f"--phase {args.phase}: only {pev.PHASES} is implemented; the SGM and MAP "
                         f"phases are added when their protocols are frozen.")
    spec = _validation_spec(cfg)
    eval_cfg = PARAMETERS.inference.evaluation
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    data_bank_root = Path(args.data_bank_root) if args.data_bank_root else PARAMETERS.machine.data_bank_root
    paths = cfg.paths.with_condition(args.condition)
    product_label = paths.product_label(timing.label, args.artifact_tag)
    estimator_path = paths.estimator_path(data_bank_root, product_label)
    stem = f"{paths.project_alias}_{product_label}_Point_Estimate_Validation_{args.phase.capitalize()}"
    out_dir = Path(args.output_dir) if args.output_dir else data_bank_root / paths.posit_subdir / stem
    array_path = out_dir / f"{stem}.npz"
    batch = args.batch_size or eval_cfg.theta_prex_batch_size
    theta_paths = {t: paths.theta_set_path(t, data_bank_root, timing.label, True, "EVAL")
                   for t in args.tasks}
    video_paths = {t: paths.video_set_path(t, data_bank_root, timing.label, True, "EVAL")
                   for t in args.tasks}

    print("=" * 72)
    print(f" {paths.project_alias} -- point-estimate validation, phase {args.phase}"
          + (f" (PILOT, {args.pilot} recordings)" if args.pilot else ""))
    print(f" checkpoint   : {estimator_path}")
    print(f" observations : EVAL tasks {args.tasks}" + (f", {args.max_sims} sims each"
                                                         if args.max_sims > 0 else ", all sims"))
    print(f" sampling     : {pev.SAMPLING_MODE}; {args.repeats} x {args.draws} draws + reference "
          f"{args.reference_draws}; batch {batch}; base seed {args.base_seed}")
    print(f" writes       : {out_dir}")
    print("=" * 72)

    if args.dry_run:
        missing = 0
        ok = Path(estimator_path).exists(); missing += not ok
        print(f"  estimator: [{'OK' if ok else 'MISSING'}]")
        for t in args.tasks:
            status = theta_set_status(theta_paths[t], spec.draw_spec, condition=args.condition)
            missing += status != "OK"
            ok = Path(video_paths[t]).exists(); missing += not ok
            print(f"  task {t}: theta [{status}]  video [{'OK' if ok else 'MISSING'}]")
        if all(theta_set_status(p, spec.draw_spec, condition=args.condition) == "OK"
               for p in theta_paths.values()):
            task_i, sim_i, truth = _frozen_observations(theta_paths, spec, args.condition, args.max_sims)
            dim = pev.dim_mask(truth, spec.parameter_keys)
            print(f"  frozen list: {task_i.size} observations, {int(dim.sum())} in the dim subgroup")
            if args.pilot:
                sel = pev.select_pilot(task_i, sim_i, truth, spec.parameter_keys, args.pilot)
                j = spec.parameter_keys.index(pev.DIM_SUBGROUP["key"])
                for i in sel:
                    print(f"    pilot: task {task_i[i]} sim {sim_i[i]}  true log10 "
                          f"{pev.DIM_SUBGROUP['key']} {truth[i, j]:.4f}  "
                          f"{'dim' if dim[i] else 'bright'}")
        print(f"[DRY RUN] {missing} input(s) missing; nothing computed.")
        return 0

    run_start = time.time()
    task_i, sim_i, truth = _frozen_observations(theta_paths, spec, args.condition, args.max_sims)
    frozen_size = int(task_i.size)
    if args.pilot:
        sel = pev.select_pilot(task_i, sim_i, truth, spec.parameter_keys, args.pilot)
        task_i, sim_i, truth = task_i[sel], sim_i[sel], truth[sel]
    dim = pev.dim_mask(truth, spec.parameter_keys)
    expected_ids = list(zip(task_i.tolist(), sim_i.tolist()))
    selection = {"tasks": [int(t) for t in args.tasks], "max_sims": int(args.max_sims),
                 "frozen_list_size": frozen_size, "pilot": int(args.pilot) or None,
                 "rule": ("every simulation of every listed EVAL task, task then sim order"
                          + ("; pilot: select_pilot (half dim, half bright, evenly spaced ranks of "
                             "the true subgroup key)" if args.pilot else "")),
                 "expected_ids": [[int(a), int(b)] for a, b in expected_ids]}
    note = (f"PILOT of {len(expected_ids)} recordings: code, throughput and memory only; no "
            f"acceptance verdict." if args.pilot else "")
    reporter = DiagnosticReporter(stage="PointEstimateValidation", enabled=True, dump=True,
                                  dump_dir=out_dir, run_label=stem, run_note=note)

    if args.merge:
        shard_paths = load_shards(out_dir)
        if not shard_paths:
            raise SystemExit(f"--merge: no shard files in {out_dir}")
        try:
            assert_complete_shard_set(shard_paths, partial_option=None)
            loaded = []
            for p in shard_paths:
                with np.load(str(p), allow_pickle=False) as d:
                    a = {k: d[k] for k in d.files}
                loaded.append((a, pev.validate_artifact(a, source=str(p), allow_empty=True)))
            arrays, manifest = pev.merge_shards(loaded, expected_ids)
            manifest["written_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            manifest["execution"] = schema.execution_identity()
            arrays["manifest_json"] = pev.encode_manifest(manifest)
            pev.validate_artifact(arrays, source="merged artifact")
        except (ValueError, pev.ValidationError) as exc:
            raise SystemExit(f"--merge: {exc}")
        np.savez(str(array_path), **arrays)
        write_median_report(reporter, arrays, manifest, out_dir)
        for p in shard_paths:
            p.unlink()
        return 0

    reporter.check_file("estimator artifact", estimator_path)
    topo = resolve_topology()
    device, vista = topo.device, torch.device("cpu")
    posterior = artifacts.load_estimator(estimator_path, device=str(device),
                                          expected_parameter_keys=spec.parameter_keys)
    posterior.posterior_estimator.to(device)
    if device.type == "cuda":
        posterior.prior = spec.build_prior(device=str(device))
    # Identity captured at startup: the checkpoint actually loaded (weights checksum verified by the
    # loader, plus the file checksum) and the code as loaded.
    checkpoint = {"path_name": Path(estimator_path).name, "weights_sha256": posterior.weights_sha256,
                  "file_sha256": file_sha256(estimator_path)}
    code_at_start = code_provenance(files=VALIDATION_FILES)
    run_ident = schema.run_identity(stem, distributed=topo.is_distributed)
    exec_ident = schema.execution_identity()

    mine = [i for i in range(len(expected_ids)) if i % topo.world_size == topo.rank]
    r_count, n_rep, n_ref, d = args.repeats, args.draws, args.reference_draws, len(spec.parameter_keys)
    q = len(QUANTILE_LEVELS)
    rep_draws = np.empty((len(mine), r_count, n_rep, d), dtype=np.float32)
    rep_q = np.empty((len(mine), r_count, d, q))
    ref_draws = np.empty((len(mine), n_ref, d), dtype=np.float32)
    ref_q = np.empty((len(mine), d, q))
    seeds = np.empty((len(mine), r_count + 1), dtype=np.int64)
    seconds = np.empty((len(mine), r_count + 1))
    peaks = np.empty((len(mine), r_count + 1), dtype=np.int64)
    videos = {}
    loop_start = time.time()
    for row, i in enumerate(mine):
        task, sim = int(task_i[i]), int(sim_i[i])
        if task not in videos:
            videos[task] = load_data(video_paths[task])
        video = np.asarray(videos[task][sim])
        if not np.any(video):
            raise SystemExit(f"EVAL video task {task} sim {sim} is all zeros: the store is missing "
                             f"that chunk or the video is empty. Refusing to summarize it.")
        for stream in range(r_count + 1):
            n = n_rep if stream < r_count else n_ref
            seed = pev.stream_seed(args.base_seed, task, sim, stream)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            summary, cloud = posterior_summary(
                posterior, video, device, vista, n, batch, pool_mode=pev.SAMPLING_MODE,
                quantiles=QUANTILE_LEVELS, return_samples=True, return_sgm=False)
            seconds[row, stream] = time.perf_counter() - t0
            peaks[row, stream] = _peak_bytes(device)
            seeds[row, stream] = seed
            if stream < r_count:
                rep_draws[row, stream], rep_q[row, stream] = cloud, summary
            else:
                ref_draws[row], ref_q[row] = cloud, summary
        done = row + 1
        avg = (time.time() - loop_start) / done
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] task {task} sim {sim} | "
              f"{done}/{len(mine)} | {seconds[row].sum():.1f}s this observation | "
              f"ETA {avg * (len(mine) - done):.0f}s", flush=True)

    sel_rows = np.asarray(mine, dtype=int)
    manifest = {
        "kind": pev.VALIDATION_KIND, "validation_schema_version": pev.VALIDATION_SCHEMA_VERSION,
        "phase": args.phase, "workflow": cfg.tag, "condition": args.condition,
        "product_label": product_label, "parameter_keys": list(spec.parameter_keys),
        "coordinate_transform": "parameterization.to_flow (log10 for log rows, linear rows as-is)",
        "sampling": {"pool_mode": pev.SAMPLING_MODE, "draw_label": draw_label(pev.SAMPLING_MODE),
                     "n_repeats": int(r_count), "repeat_draws": int(n_rep),
                     "reference_draws": int(n_ref), "batch_size": int(batch),
                     "function": "evaluation.posterior_summary(return_samples=True, return_sgm=False)"},
        "quantiles": {"levels": [float(v) for v in QUANTILE_LEVELS],
                      "interpolation": pev.INTERPOLATION},
        "checkpoint": checkpoint,
        "code": finalize_code_provenance(code_at_start, files=VALIDATION_FILES),
        "seeds": {"base_seed": int(args.base_seed),
                  "rule": "point_estimate_validation.stream_seed(base_seed, task, sim, stream): "
                          "SeedSequence(entropy=base_seed, spawn_key=(task, sim, stream)); streams "
                          "0..R-1 are the repeats, stream R the reference; torch.manual_seed per stream"},
        "observations": selection, "dim_subgroup": pev.DIM_SUBGROUP, "thresholds": pev.THRESHOLDS,
        "run_identity": run_ident, "execution": exec_ident,
        "environment": _environment(),
        "device": str(device), "memory_measure": ("cuda max_memory_allocated per stream"
                                                  if device.type == "cuda"
                                                  else "process peak RSS (monotone) after each stream"),
        "data_bank_root": str(data_bank_root), "n_observations": len(mine),
        "rank": topo.rank if topo.is_distributed else None,
        "world_size": topo.world_size if topo.is_distributed else None,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    arrays = {"task_index": task_i[sel_rows], "sim_index": sim_i[sel_rows],
              "true_log10": truth[sel_rows], "dim_subgroup": dim[sel_rows],
              "repeat_draws": rep_draws, "repeat_quantiles": rep_q, "reference_draws": ref_draws,
              "reference_quantiles": ref_q, "stream_seeds": seeds, "stream_seconds": seconds,
              "stream_peak_bytes": peaks, "manifest_json": pev.encode_manifest(manifest)}
    try:
        pev.validate_artifact(arrays, source=f"rank {topo.rank} artifact",
                              allow_empty=topo.is_distributed)
    except pev.ValidationError as exc:
        raise SystemExit(f"validation artifact refused at publication: {exc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if topo.is_distributed:
        np.savez(str(shard_path(out_dir, topo.rank, topo.world_size)), **arrays)
        print(f"[rank {topo.rank}/{topo.world_size}] shard written ({len(mine)} observations) in "
              f"{time.time() - run_start:.0f}s; run --merge once every rank has finished.")
        return 0
    np.savez(str(array_path), **arrays)
    write_median_report(reporter, arrays, manifest, out_dir)
    print(f"Total elapsed: {time.time() - run_start:.1f}s")
    return 0


def _fmt(v, digits=4):
    return f"{v:.{digits}f}" if np.isfinite(v) else str(v)


def write_median_report(reporter, arrays, manifest, out_dir):
    """The phase-1 report, computed from the artifact alone (so it can be re-rendered from disk)."""
    keys = manifest["parameter_keys"]
    thr = manifest["thresholds"]
    obs = manifest["observations"]
    pilot = obs.get("pilot")
    samp = manifest["sampling"]
    metrics = pev.median_metrics(arrays, manifest)
    n = int(np.asarray(arrays["task_index"]).shape[0])
    reporter.stat("checkpoint", manifest["checkpoint"]["path_name"],
                  note=f"weights sha256 {manifest['checkpoint']['weights_sha256']}; file sha256 "
                       f"{manifest['checkpoint']['file_sha256']}")
    reporter.stat("observations", n, note=f"{obs['rule']}; frozen list {obs['frozen_list_size']}")
    reporter.stat("dim subgroup", int(np.asarray(arrays["dim_subgroup"]).sum()),
                  note=f"true log10 {manifest['dim_subgroup']['key']} in "
                       f"[{manifest['dim_subgroup']['log10_range'][0]}, "
                       f"{manifest['dim_subgroup']['log10_range'][1]}); {manifest['dim_subgroup']['source']}")
    reporter.stat("sampling", samp["pool_mode"],
                  note=f"{samp['draw_label']}s; {samp['n_repeats']} repeats x {samp['repeat_draws']} "
                       f"draws and a reference of {samp['reference_draws']}, each its own random "
                       f"stream; {samp['function']}; batch {samp['batch_size']}")
    reporter.stat("median convention", "estimator coordinates", note=manifest["quantiles"]["interpolation"])
    reporter.stat("resolution", f"{thr['resolution_dex']} dex",
                  note="smallest subset-level MAE or bias difference the program interprets; an "
                       "analysis-resolution choice, not a biological tolerance or significance level")

    worst, where = metrics["recompute"]
    reporter.check("recomputation_agreement", worst <= thr["recompute_abs_dex"],
                   f"max |stored - recomputed| = {worst:.2e} dex at {where}", fatal=False,
                   note=f"every stored quantile against a sort-and-interpolate recomputation from its "
                        f"own stored draws; tolerance {thr['recompute_abs_dex']} dex absolute.")

    stab_rows, res_rows, rec_rows = [], [], []
    for name in pev.SUBGROUPS:
        s = metrics["subgroups"].get(name)
        if s is None:
            continue
        for j, k in enumerate(keys):
            stab_rows.append([k, name, s["n"],
                              _fmt(s["mae"]["mean"][j]), _fmt(s["mae"]["sd"][j], 5),
                              f"{_fmt(s['mae']['min'][j])} .. {_fmt(s['mae']['max'][j])}",
                              f"{s['bias']['mean'][j]:+.4f}", _fmt(s["bias"]["sd"][j], 5),
                              f"{s['bias']['min'][j]:+.4f} .. {s['bias']['max'][j]:+.4f}",
                              f"{100 * s['large']['mean'][j]:.1f}%"])
            res_rows.append([k, name, _fmt(s["mae_ref"][j]), f"{s['bias_ref'][j]:+.4f}",
                             _fmt(abs(s["mae"]["mean"][j] - s["mae_ref"][j]), 5),
                             _fmt(abs(s["bias"]["mean"][j] - s["bias_ref"][j]), 5),
                             f"{100 * s['large_ref'][j]:.1f}%"])
            rec_rows.append([k, name, f"{100 * s['ratio_share'][j]:.1f}%", int(s["exceedances"][j]),
                             _fmt(s["ratio_max"][j], 3)])
    reporter.table("Subset accuracy of the median across repeats (dex)",
                   ["parameter", "subgroup", "n", "MAE mean", "MAE sd", "MAE min .. max",
                    "bias mean", "bias sd", "bias min .. max", "large error"], stab_rows,
                   note=f"MAE and signed bias (median - truth) computed separately for each repeat; "
                        f"'sd' is the sample standard deviation across repeats (criterion <= "
                        f"{thr['subset_sd_dex']} dex). Large error = |median - truth| > "
                        f"{thr['large_error_dex']} dex, mean share over repeats. Five repeats estimate "
                        f"the Monte Carlo variability; they do not bound it.")
    reporter.table("Resolution check against the reference run (dex)",
                   ["parameter", "subgroup", "MAE ref", "bias ref", "|MAE mean - ref|",
                    "|bias mean - ref|", "large error ref"], res_rows,
                   note=f"criterion <= {thr['reference_diff_dex']} dex for both columns. The reference "
                        f"({samp['reference_draws']} draws) is a finite sample, not exact truth; a "
                        f"narrow failure is assessed against its own sampling variability first.")
    reporter.table("Per-recording stability of the median",
                   ["parameter", "subgroup", f"share <= {thr['ratio_max']}", "exceedances",
                    "max ratio"], rec_rows,
                   note=f"ratio = standard deviation of the {samp['n_repeats']} repeat medians / "
                        f"posterior IQR of the reference run; criterion: share >= "
                        f"{100 * thr['ratio_share_min']:.0f}%. A normal posterior gives about 0.03 "
                        f"at 1,000 draws.")

    ratio, status = metrics["ratio"], metrics["ratio_status"]
    task, sim = np.asarray(arrays["task_index"]), np.asarray(arrays["sim_index"])
    exceed = np.argwhere(~(ratio <= thr["ratio_max"]))
    exc_rows = [[int(task[i]), int(sim[i]), keys[j], _fmt(ratio[i, j], 3),
                 _fmt(metrics["ratio_sd"][i, j], 5), _fmt(metrics["ratio_iqr"][i, j], 5),
                 status[i, j]] for i, j in exceed]
    with open(Path(out_dir) / "per_recording_exceedances.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "sim", "parameter", "ratio", "sd_repeat_medians", "reference_iqr", "status"])
        w.writerows(exc_rows)
    reporter.table("Per-recording exceedances", ["task", "sim", "parameter", "ratio", "sd",
                                                 "reference IQR", "status"],
                   exc_rows[:REPORT_EXCEEDANCE_ROWS] or [["-", "-", "-", "-", "-", "-", "none"]],
                   note=f"{len(exc_rows)} exceedance(s); the first {REPORT_EXCEEDANCE_ROWS} are listed, "
                        f"all are in per_recording_exceedances.csv.")
    zero = {s: int((status == s).sum()) for s in ("zero_iqr_zero_sd", "zero_iqr_nonzero_sd")}
    reporter.stat("zero-IQR cases", sum(zero.values()),
                  note=f"{zero['zero_iqr_zero_sd']} with zero spread (counted as meeting the "
                       f"criterion), {zero['zero_iqr_nonzero_sd']} with nonzero spread (exceedances); "
                       f"none is dropped.")

    if pilot:
        reporter.stat("acceptance verdict", "none",
                      note=f"pilot of {n} recordings: code, throughput and memory only.")
    else:
        gates = pev.evaluate_gates(metrics, thr)
        rows = []
        for name in pev.SUBGROUPS:
            g = gates.get(name)
            if g is None:
                continue
            for j, k in enumerate(keys):
                rows.append([k, name] + ["PASS" if bool(g[c][j]) else "FAIL"
                                         for c in ("subset_sd", "reference_diff", "per_recording")])
        reporter.table("Phase-1 gates", ["parameter", "subgroup", "subset sd", "reference diff",
                                         "per recording"], rows,
                       note="recomputation agreement is the check above; all four gates must pass.")
        reporter.check("phase1_gates", gates["recompute"] and all(
            all(bool(v) for v in np.ravel(g[c])) for name, g in gates.items() if name != "recompute"
            for c in ("subset_sd", "reference_diff", "per_recording")),
            "all phase-1 gates", fatal=False,
            note="recomputation, subset stability, resolution check and per-recording stability.")

    secs = np.asarray(arrays["stream_seconds"], dtype=float)
    peaks = np.asarray(arrays["stream_peak_bytes"], dtype=float)
    r = samp["n_repeats"]
    rep_s, ref_s = np.median(secs[:, :r]), np.median(secs[:, r])
    per_obs = float(np.median(secs.sum(axis=1)))
    reporter.table("Throughput and memory", ["stream", "draws", "median seconds", "draws per second",
                                             "peak memory (MB)"],
                   [["repeat", samp["repeat_draws"], f"{rep_s:.2f}", f"{samp['repeat_draws'] / rep_s:.0f}",
                     f"{peaks[:, :r].max() / 2**20:.0f}"],
                    ["reference", samp["reference_draws"], f"{ref_s:.2f}",
                     f"{samp['reference_draws'] / ref_s:.0f}", f"{peaks[:, r].max() / 2**20:.0f}"]],
                   note=f"device {manifest['device']}, torch {manifest['environment']['torch']}; memory = "
                        f"{manifest['memory_measure']}. Median "
                        f"{per_obs:.1f} s per observation for all streams; at that pace the frozen "
                        f"list of {obs['frozen_list_size']} would take "
                        f"{per_obs * obs['frozen_list_size'] / 3600:.1f} h on one such device.")
    reporter.summary()
    reporter.write_report()


def build_parser(description: str) -> argparse.ArgumentParser:
    eval_cfg = PARAMETERS.inference.evaluation
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--condition", required=True, choices=("FAB", "INLB"),
                    help="Experimental condition: selects the condition-specific estimator and EVAL tier.")
    ap.add_argument("--total-time-seconds", type=float, required=True,
                    help="Recording duration of the tier (selects the timing label).")
    ap.add_argument("--artifact-tag", default=None,
                    help="Estimator artifact tag (none = the estimator of record).")
    ap.add_argument("--phase", default="median", choices=pev.PHASES,
                    help="Validation phase. Only the median reference is implemented.")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1],
                    help="EVAL tasks forming the frozen observation list (default: 0 1).")
    ap.add_argument("--max-sims", type=int, default=0,
                    help="Per-task cap on simulations (0 = all; the frozen protocol uses all).")
    ap.add_argument("--pilot", type=int, default=0,
                    help="Select this many recordings deterministically (half dim, half bright) to "
                         "measure throughput and memory; the report then issues no verdict.")
    ap.add_argument("--repeats", type=int, default=5, help="Independent repeats per observation.")
    ap.add_argument("--draws", type=int, default=eval_cfg.posterior_samples,
                    help=f"Draws per repeat (default: the production {eval_cfg.posterior_samples}).")
    ap.add_argument("--reference-draws", type=int, default=10000,
                    help="Draws in the reference run (default 10000).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help=f"Sampling batch (default: production {eval_cfg.theta_prex_batch_size}).")
    ap.add_argument("--base-seed", type=int, default=20260923,
                    help="Base seed from which every stream's seed is derived (recorded).")
    ap.add_argument("--data-bank-root", default=None,
                    help="Read inputs from this data-bank root instead of the machine profile's "
                         "(for a pilot on a machine that holds only the needed inputs).")
    ap.add_argument("--output-dir", default=None,
                    help="Write here instead of the Posit tier (for a pilot, use a scratch directory).")
    ap.add_argument("--merge", action="store_true",
                    help="Merge the per-rank shards of a sharded run and write the report.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve inputs, list the frozen observations (and the pilot), compute nothing.")
    return ap
