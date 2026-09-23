"""Shared engine of the MAP optimization benchmark (see ``map_benchmark`` for the design).

For a fixed checkpoint and a fixed list of synthetic EVAL recordings, per recording: two independent
bounded candidate pools of the production size; the production number of top candidates as seeds;
configuration A (best candidate), B (the production ``optimize_elite``, unchanged, one recording at a
time) and the new-loop configurations C..H (all chains of a group of recordings batched through
``map_benchmark.run_chains``); a reference optimum (the best vector found by any configuration on
either pool, polished with L-BFGS); and the density re-evaluated at every returned vector. Results
go to a benchmark artifact; a multi-GPU run writes one shard per rank and ``--merge`` combines them
and writes the report. It is a validation utility, never wired into the stage dispatcher.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import artifact_schema as schema
from . import artifacts
from . import map_benchmark as mb
from . import point_estimate_validation as pev
from .diagnostics import DiagnosticReporter
from .evaluation import (collect_score_prex, collect_theta_prex, extract_elite_prex, optimize_elite,
                         optimizer_contract, pool_scale, prior_scale)
from .experiment_support import assert_complete_shard_set, load_shards, shard_path
from .inference_support import normalize_video, resolve_topology
from .io import load_data, theta_set_status
from .parameterization import PARAMETERS, RunTiming
from .point_estimate_validation_runner import _environment, _frozen_observations, _validation_spec
from .provenance import IMPLEMENTATION_FILES, code_provenance, file_sha256, finalize_code_provenance
from .workflow import WorkflowConfig

BENCHMARK_KIND, BENCHMARK_VERSION = "map_benchmark", 1
BENCHMARK_FILES = IMPLEMENTATION_FILES + (
    "srm_and_sbi_monomer_dimer_alp/map_benchmark.py",
    "srm_and_sbi_monomer_dimer_alp/map_benchmark_runner.py",
    "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_MAP_Benchmark.py",
)
POOLS = 2
CONTRACT = ("kind", "version", "workflow", "condition", "product_label", "parameter_keys", "checkpoint",
            "code", "configs", "pool_size", "elites", "base_seed", "observations", "run_identity")
ARRAY_KEYS = ("task_index", "sim_index", "true_log10", "dim_subgroup", "best_candidate", "iqr", "scale",
              "scale_raised", "scale_capped", "score", "reeval", "theta", "stop_code", "steps",
              "overshoot", "max_deviation", "seconds_production", "seconds_pool", "reference_score",
              "reference_theta", "reference_polish_gain")


@contextmanager
def identity_embedding(flow, embedding_shape):
    """The production ``map_estimate`` substitution: the flow scores against a precomputed
    embedding. Restored on exit, so sampling (which embeds the video) is never done inside."""
    holder = flow.net if (hasattr(flow, "net") and hasattr(flow.net, "_embedding_net")) else flow
    original, shape = holder._embedding_net, flow.condition_shape
    holder._embedding_net = torch.nn.Identity()
    flow._condition_shape = tuple(embedding_shape)
    try:
        yield
    finally:
        holder._embedding_net = original
        flow._condition_shape = shape


def _lbfgs_polish(flow, emb, theta0, center, scale, max_iter):
    """L-BFGS (strong Wolfe) on the same objective from ``theta0``, in scaled coordinates."""
    u = ((theta0 - center) / scale).detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([u], lr=1.0, max_iter=max_iter, tolerance_grad=1e-10,
                            tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")

    def score(theta):
        return flow.log_prob(input=theta.view(1, 1, -1), condition=emb).view(())

    def closure():
        opt.zero_grad()
        loss = -score(center + scale * u)
        loss.backward()
        return loss
    opt.step(closure)
    theta = (center + scale * u).detach()
    with torch.no_grad():
        return float(score(theta).item()), theta


def _process_group(flow, post, rows, videos, truth_rows, dev, args, prior_w, n_seeds):
    """Everything for one group of recordings; returns per-recording result arrays."""
    cpu = torch.device("cpu")
    ev = PARAMETERS.inference.evaluation
    g, dim = len(rows), prior_w.shape[0]
    k_all = len(mb.CONFIG_ORDER)
    out = {"best_candidate": np.zeros((g, POOLS)), "iqr": np.zeros((g, POOLS, dim)),
           "scale": np.zeros((g, POOLS, dim)), "scale_raised": np.zeros((g, POOLS, dim), bool),
           "scale_capped": np.zeros((g, POOLS, dim), bool),
           "score": np.zeros((g, POOLS, k_all)), "reeval": np.zeros((g, POOLS, k_all)),
           "theta": np.zeros((g, POOLS, k_all, dim)),
           "stop_code": np.zeros((g, POOLS, k_all, n_seeds), np.int64),
           "steps": np.zeros((g, POOLS, k_all, n_seeds), np.int64),
           "overshoot": np.full((g, POOLS, k_all, n_seeds), np.nan),
           "max_deviation": np.full((g, POOLS, k_all, n_seeds), np.nan),
           "seconds_production": np.zeros((g, POOLS)), "seconds_pool": np.zeros((g, POOLS)),
           "reference_score": np.zeros(g), "reference_theta": np.zeros((g, dim)),
           "reference_polish_gain": np.zeros(g)}
    embs, pools = [], []
    for j, (task, sim) in enumerate(rows):
        cond = torch.tensor(normalize_video(videos[j]), dtype=torch.float32, device=dev).unsqueeze(0)
        post.set_default_x(cond)
        rec_pools = []
        for p in range(POOLS):
            torch.manual_seed(pev.stream_seed(args.base_seed, task, sim, 100 + p))
            t0 = time.perf_counter()
            rec_pools.append(collect_theta_prex(post, flow, cpu, cond, args.pool_size, args.pool_size,
                                                "bounded"))
            out["seconds_pool"][j, p] = time.perf_counter() - t0
        with torch.no_grad():
            embs.append(flow.embedding_net(cond).detach())
        pools.append(rec_pools)
    w_t = torch.tensor(prior_w, dtype=torch.float32, device=dev)
    with identity_embedding(flow, embs[0].shape[1:]):
        # A and B per recording and pool; seeds and scales for the chains
        chain = {k: [] for k in ("theta0", "center", "scale", "lr", "lr_min", "max_steps",
                                 "stop_patience", "sched_patience", "guard", "dscale", "rec")}
        slots = []                                          # (row, pool, config index, seed index)
        for j, (task, sim) in enumerate(rows):
            for p in range(POOLS):
                pool = pools[j][p]
                sc = collect_score_prex(flow, dev, cpu, embs[j], pool, args.pool_size)
                seeds = extract_elite_prex(pool, sc, n_seeds).to(dev)
                center, iqr, scale, raised, capped = mb.scaled_coordinates(pool.numpy(), prior_w)
                out["iqr"][j, p], out["scale"][j, p] = iqr, scale
                out["scale_raised"][j, p], out["scale_capped"][j, p] = raised, capped
                out["best_candidate"][j, p] = float(sc.max().item())
                ka = mb.CONFIG_ORDER.index("A")
                out["score"][j, p, ka] = out["best_candidate"][j, p]
                out["theta"][j, p, ka] = pool[int(sc.argmax())].numpy()
                out["stop_code"][j, p, ka] = mb.STOP_NONE
                kb = mb.CONFIG_ORDER.index("B")
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                # B is the production optimizer AS CONFIGURED when the benchmark runs (the
                # recorded runs used the 0.1.16 absolute-step optimizer; from 0.1.17 it steps in
                # pool-IQR units). A pool whose IQR cannot set the step returns its best candidate.
                p_center, p_iqr = pool_scale(pool)
                if np.all(np.isfinite(p_iqr) & (p_iqr > 0)):
                    s_b, th_b, info_b = optimize_elite(
                        flow, dev, cpu, embs[j], seeds, ev.numb_steps, ev.optimizer_patience,
                        ev.scheduler_patience, 10 ** 9, ev.learning_rate_minimum,
                        ev.learning_rate_factor, ev.learning_rate, ev.tolerance,
                        center=p_center, scale=p_iqr)
                else:
                    s_b, th_b = out["best_candidate"][j, p], pool[int(sc.argmax())].numpy()
                    info_b = {"stop": "invalid-scale", "steps": 0}
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                out["seconds_production"][j, p] = time.perf_counter() - t0
                out["score"][j, p, kb], out["theta"][j, p, kb] = float(s_b), np.asarray(th_b)
                out["stop_code"][j, p, kb] = mb.PRODUCTION_STOP_CODES[info_b["stop"]]
                out["steps"][j, p, kb] = int(info_b["steps"])
                c_t = torch.tensor(center, dtype=torch.float32, device=dev)
                s_t = torch.tensor(scale, dtype=torch.float32, device=dev)
                iqr_t = torch.tensor(np.maximum(iqr, 1e-12), dtype=torch.float32, device=dev)
                for name, (coords, lr, lr_min, budget) in mb.CHAIN_CONFIGS.items():
                    kc = mb.CONFIG_ORDER.index(name)
                    for q in range(n_seeds):
                        chain["theta0"].append(seeds[q])
                        chain["center"].append(c_t if coords == "scaled" else torch.zeros_like(c_t))
                        chain["scale"].append(s_t if coords == "scaled" else torch.ones_like(s_t))
                        chain["lr"].append(lr); chain["lr_min"].append(lr_min)
                        for key in ("max_steps", "stop_patience", "sched_patience", "guard"):
                            chain[key].append(budget[key])
                        chain["dscale"].append(iqr_t); chain["rec"].append(j)
                        slots.append((j, p, kc, q))
        emb_rows = torch.cat(embs, dim=0)                   # (g, E)
        rec_of = torch.tensor(chain["rec"], dtype=torch.long, device=dev)

        def score_fn(theta, idx):
            return flow.log_prob(input=theta.unsqueeze(0), condition=emb_rows[rec_of[idx]]).squeeze(0)

        as_t = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=dev)
        res = mb.run_chains(score_fn, torch.stack(chain["theta0"]), torch.stack(chain["center"]),
                            torch.stack(chain["scale"]), as_t(chain["lr"]), as_t(chain["lr_min"]),
                            as_t(chain["max_steps"], torch.long), as_t(chain["stop_patience"], torch.long),
                            as_t(chain["sched_patience"], torch.long), as_t(chain["guard"], torch.long),
                            deviation_scale=torch.stack(chain["dscale"]))
        best = res["best"].cpu().numpy(); best_th = res["best_theta"].cpu().numpy()
        for m, (j, p, kc, q) in enumerate(slots):
            out["stop_code"][j, p, kc, q] = int(res["stop"][m]); out["steps"][j, p, kc, q] = int(res["steps"][m])
            out["overshoot"][j, p, kc, q] = float(res["overshoot"][m])
            out["max_deviation"][j, p, kc, q] = float(res["max_deviation"][m])
            if q == 0 or best[m] > out["score"][j, p, kc]:
                out["score"][j, p, kc], out["theta"][j, p, kc] = best[m], best_th[m]
        # consistency: the density at every returned vector
        th_all = torch.tensor(out["theta"].reshape(-1, dim), dtype=torch.float32, device=dev)
        rec_all = torch.arange(g, device=dev).repeat_interleave(POOLS * k_all)
        with torch.no_grad():
            re = flow.log_prob(input=th_all.unsqueeze(0), condition=emb_rows[rec_all]).squeeze(0)
        out["reeval"] = re.cpu().numpy().reshape(g, POOLS, k_all)
        # reference: the best vector found anywhere for the recording, polished
        for j in range(g):
            flat = out["score"][j].ravel()
            best_idx = int(np.argmax(flat))
            p_best = best_idx // k_all
            th0 = torch.tensor(out["theta"][j].reshape(-1, dim)[best_idx], dtype=torch.float32, device=dev)
            c = torch.tensor(np.median(pools[j][p_best].numpy(), axis=0), dtype=torch.float32, device=dev)
            s = torch.tensor(out["scale"][j, p_best], dtype=torch.float32, device=dev)
            pol, th_pol = _lbfgs_polish(flow, embs[j], th0, c, s, args.polish_iterations)
            top = float(flat[best_idx])
            out["reference_score"][j] = max(pol, top)
            out["reference_theta"][j] = (th_pol if pol >= top else th0).cpu().numpy()
            out["reference_polish_gain"][j] = pol - top
    return out


def run_map_benchmark(cfg: WorkflowConfig, args: argparse.Namespace) -> int:
    spec = _validation_spec(cfg)
    ev = PARAMETERS.inference.evaluation
    timing = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing)
    root = Path(args.data_bank_root) if args.data_bank_root else PARAMETERS.machine.data_bank_root
    paths = cfg.paths.with_condition(args.condition)
    product_label = paths.product_label(timing.label, args.artifact_tag)
    estimator_path = paths.estimator_path(root, product_label)
    stem = f"{paths.project_alias}_{product_label}_MAP_Benchmark"
    out_dir = Path(args.output_dir) if args.output_dir else root / paths.posit_subdir / stem
    theta_paths = {t: paths.theta_set_path(t, root, timing.label, True, "EVAL") for t in args.tasks}
    video_paths = {t: paths.video_set_path(t, root, timing.label, True, "EVAL") for t in args.tasks}
    print("=" * 72)
    print(f" {paths.project_alias} -- MAP optimization benchmark")
    print(f" checkpoint : {estimator_path}")
    print(f" recordings : EVAL tasks {args.tasks}" + (f", first {args.max_sims} sims each" if args.max_sims > 0 else ", all sims")
          + (f"; pilot {args.pilot}" if args.pilot else ""))
    print(f" pools      : {POOLS} x {args.pool_size} bounded candidates; {ev.elite_prex_size} seeds; configs "
          + ", ".join(mb.CONFIG_ORDER))
    print(f" writes     : {out_dir}")
    print("=" * 72)
    if args.dry_run:
        ok = Path(estimator_path).exists()
        print(f"  estimator: [{'OK' if ok else 'MISSING'}]")
        for t in args.tasks:
            print(f"  task {t}: theta [{theta_set_status(theta_paths[t], spec.draw_spec, condition=args.condition)}]"
                  f"  video [{'OK' if Path(video_paths[t]).exists() else 'MISSING'}]")
        print("[DRY RUN] nothing computed.")
        return 0

    run_start = time.time()
    task_i, sim_i, truth = _frozen_observations(theta_paths, spec, args.condition, args.max_sims)
    if args.pilot:
        sel = pev.select_pilot(task_i, sim_i, truth, spec.parameter_keys, args.pilot)
        task_i, sim_i, truth = task_i[sel], sim_i[sel], truth[sel]
    dim_flags = pev.dim_mask(truth, spec.parameter_keys)
    expected = [[int(a), int(b)] for a, b in zip(task_i, sim_i)]
    reporter = DiagnosticReporter(stage="MAPBenchmark", enabled=True, dump=True, dump_dir=out_dir,
                                  run_label=stem)
    if args.merge:
        shard_paths = load_shards(out_dir)
        if not shard_paths:
            raise SystemExit(f"--merge: no shards in {out_dir}")
        assert_complete_shard_set(shard_paths, partial_option=None)
        loaded = []
        for p in shard_paths:
            with np.load(str(p), allow_pickle=False) as d:
                loaded.append(({k: d[k] for k in d.files}, json.loads(str(d["manifest_json"]))))
        ref = loaded[0][1]
        for _, m in loaded[1:]:
            for key in CONTRACT:
                if json.dumps(m[key], sort_keys=True) != json.dumps(ref[key], sort_keys=True):
                    raise SystemExit(f"--merge: shards disagree on {key!r}")
        arrays = {k: np.concatenate([a[k] for a, _ in loaded], axis=0) for k in ARRAY_KEYS}
        ids = list(zip(arrays["task_index"].tolist(), arrays["sim_index"].tolist()))
        want = [tuple(e) for e in ref["observations"]["expected_ids"]]
        if sorted(ids) != sorted(want) or len(set(ids)) != len(ids):
            raise SystemExit("--merge: merged recordings differ from the frozen list")
        order = np.array([ids.index(e) for e in want])
        arrays = {k: v[order] for k, v in arrays.items()}
        manifest = dict(ref, rank=None, world_size=None, execution=schema.execution_identity(),
                        shards=[{"rank": m["rank"], "n": m["n_observations"], "execution": m["execution"]}
                                for _, m in loaded])
        np.savez(str(out_dir / f"{stem}.npz"), manifest_json=np.asarray(json.dumps(manifest)), **arrays)
        write_report(reporter, arrays, manifest)
        for p in shard_paths:
            p.unlink()
        return 0

    reporter.check_file("estimator artifact", estimator_path)
    topo = resolve_topology()
    dev = topo.device
    post = artifacts.load_estimator(estimator_path, device=str(dev), expected_parameter_keys=spec.parameter_keys)
    post.posterior_estimator.to(dev)
    if dev.type == "cuda":
        post.prior = spec.build_prior(device=str(dev))
    flow = post.posterior_estimator
    checkpoint = {"path_name": Path(estimator_path).name, "weights_sha256": post.weights_sha256,
                  "file_sha256": file_sha256(estimator_path)}
    code_at_start = code_provenance(files=BENCHMARK_FILES)
    run_ident = schema.run_identity(stem, distributed=topo.is_distributed)
    prior_w = prior_scale(spec.draw_spec)
    mine = [i for i in range(len(expected)) if i % topo.world_size == topo.rank]
    parts, stores = [], {}
    for start in range(0, len(mine), args.group_size):
        group = mine[start:start + args.group_size]
        rows = [(int(task_i[i]), int(sim_i[i])) for i in group]
        vids = []
        for task, sim in rows:
            if task not in stores:
                stores[task] = load_data(video_paths[task])
            v = np.asarray(stores[task][sim])
            if not np.any(v):
                raise SystemExit(f"EVAL video task {task} sim {sim} is all zeros; refusing it.")
            vids.append(v)
        t0 = time.time()
        res = _process_group(flow, post, rows, vids, truth[group], dev, args, prior_w, ev.elite_prex_size)
        parts.append((group, res))
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] rank {topo.rank}: "
              f"{start + len(group)}/{len(mine)} recordings ({time.time() - t0:.1f}s for this group)", flush=True)
    sel = np.array([i for g, _ in parts for i in g], dtype=int)
    arrays = {k: np.concatenate([r[k] for _, r in parts], axis=0) for k in parts[0][1]} if parts else {}
    arrays.update(task_index=task_i[sel], sim_index=sim_i[sel], true_log10=truth[sel],
                  dim_subgroup=dim_flags[sel])
    manifest = {"kind": BENCHMARK_KIND, "version": BENCHMARK_VERSION, "workflow": cfg.tag,
                "condition": args.condition, "product_label": product_label,
                "parameter_keys": list(spec.parameter_keys), "checkpoint": checkpoint,
                "code": finalize_code_provenance(code_at_start, files=BENCHMARK_FILES),
                "configs": {"order": list(mb.CONFIG_ORDER), "labels": mb.CONFIG_LABELS,
                            "chains": {k: [v[0], v[1], v[2], v[3]] for k, v in mb.CHAIN_CONFIGS.items()},
                            "improve_tol": mb.IMPROVE_TOL, "scale_eps": mb.SCALE_EPS,
                            "production": optimizer_contract(
                                ev, learning_rate=ev.learning_rate, tolerance=ev.tolerance,
                                theta_prex_size=args.pool_size, elite_prex_size=ev.elite_prex_size,
                                numb_steps=ev.numb_steps, pool_mode="bounded"),
                            "polish_iterations": args.polish_iterations},
                "pool_size": args.pool_size, "elites": ev.elite_prex_size, "sampling": "bounded",
                "base_seed": args.base_seed,
                "observations": {"tasks": list(args.tasks), "max_sims": args.max_sims, "pilot": args.pilot or None,
                                 "expected_ids": expected},
                "run_identity": run_ident, "execution": schema.execution_identity(),
                "environment": _environment(), "device": str(dev), "n_observations": int(sel.size),
                "rank": topo.rank if topo.is_distributed else None,
                "world_size": topo.world_size if topo.is_distributed else None,
                "seconds_total": time.time() - run_start}
    if manifest["code"]["changed_during_run"]:
        raise SystemExit("implementation files changed during the run; the benchmark is refused.")
    out_dir.mkdir(parents=True, exist_ok=True)
    packed = {"manifest_json": np.asarray(json.dumps(manifest)), **arrays}
    if topo.is_distributed:
        np.savez(str(shard_path(out_dir, topo.rank, topo.world_size)), **packed)
        print(f"[rank {topo.rank}] shard written; run --merge when every rank has finished.")
        return 0
    np.savez(str(out_dir / f"{stem}.npz"), **packed)
    write_report(reporter, arrays, manifest)
    print(f"Total elapsed: {time.time() - run_start:.1f}s")
    return 0


def write_report(reporter, arrays, manifest):
    """The benchmark report, computed from the artifact alone."""
    keys = manifest["parameter_keys"]
    n = int(np.asarray(arrays["task_index"]).shape[0])
    reporter.stat("checkpoint", manifest["checkpoint"]["path_name"],
                  note=f"weights sha256 {manifest['checkpoint']['weights_sha256']}")
    reporter.stat("recordings", n, note=f"dim subgroup {int(np.asarray(arrays['dim_subgroup']).sum())}; "
                                        f"{POOLS} independent bounded pools of {manifest['pool_size']} each; "
                                        f"{manifest['elites']} seeds per pool, identical for every configuration")
    reporter.stat("objective", "unconstrained flow log-density, estimator coordinates",
                  note="a returned point is a numerical MAP candidate")
    rows = mb.summary_rows(arrays)
    reporter.table("Density gain and gap to the reference optimum (nats)",
                   ["config", "setting", "gain median", "gain mean", "improved", "gap median", "gap p90",
                    "gap max", "within 1e-3", "within 1e-2", "max |score - reeval|"],
                   [[r["config"], r["label"], f"{r['gain_median']:.4f}", f"{r['gain_mean']:.4f}",
                     f"{100 * r['improved_share']:.0f}%", f"{r['gap_median']:.4f}", f"{r['gap_p90']:.4f}",
                     f"{r['gap_max']:.4f}", f"{100 * r['within_1e-3']:.0f}%", f"{100 * r['within_1e-2']:.0f}%",
                     f"{r['inconsistency_max']:.1e}"] for r in rows],
                   note="gain = returned score minus the best starting candidate's; gap = reference score "
                        "minus returned score, the reference being the best vector found anywhere, polished "
                        "with L-BFGS; improved = gain above 1e-3 nats. Every configuration of a pool starts "
                        "from the same seeds.")
    reporter.table("Stopping and cost (per seed; B from 0.1.17 on)",
                   ["config", "budget hits", "non-finite", "steps median", "steps p90"],
                   [[r["config"], f"{100 * r['budget_share']:.0f}%", f"{100 * r['nonfinite_share']:.0f}%",
                     f"{r['steps_median']:.0f}", f"{r['steps_p90']:.0f}"] for r in rows if np.isfinite(r["steps_median"])],
                   note="a budget hit means the chain was still improving meaningfully when it ran out of steps.")
    ag = mb.pool_agreement_rows(arrays)
    reporter.table("Sensitivity to initialization (pool 1 versus pool 2)",
                   ["config", "|score diff| median", "p90", "max", "max |theta diff| / IQR median", "p90", "max"],
                   [[r["config"], f"{r['score_diff_median']:.4f}", f"{r['score_diff_p90']:.4f}",
                     f"{r['score_diff_max']:.4f}", f"{r['theta_diff_iqr_median']:.3f}",
                     f"{r['theta_diff_iqr_p90']:.3f}", f"{r['theta_diff_iqr_max']:.3f}"] for r in ag],
                   note="an optimizer that reaches the mode returns nearly the same point from both pools.")
    over = np.asarray(arrays["overshoot"]); dev_ = np.asarray(arrays["max_deviation"])
    reporter.table("Path diagnostics (new-loop configurations, per seed)",
                   ["config", "overshoot median (nats)", "overshoot p90", "max deviation median (IQR)", "p90"],
                   [[name, f"{np.nanmedian(over[:, :, k]):.3f}", f"{np.nanquantile(over[:, :, k], 0.9):.3f}",
                     f"{np.nanmedian(dev_[:, :, k]):.2f}", f"{np.nanquantile(dev_[:, :, k], 0.9):.2f}"]
                    for k, name in enumerate(mb.CONFIG_ORDER) if name in mb.CHAIN_CONFIGS],
                   note="overshoot = start score minus the lowest score visited; deviation = largest distance "
                        "from the start, in units of the pool IQR.")
    truth = np.asarray(arrays["true_log10"]); theta = np.asarray(arrays["theta"])[:, 0]      # pool 1
    dim = np.asarray(arrays["dim_subgroup"], bool)
    rec_rows = []
    for k, name in enumerate(mb.CONFIG_ORDER):
        err = theta[:, k, :] - truth
        rec_rows.append([name] + [f"{np.mean(np.abs(err[:, j])):.4f}" for j in range(len(keys))]
                        + [f"{np.mean(np.abs(err[dim, j])):.4f}" if dim.any() else "-" for j in range(len(keys))])
    reporter.table("Recovery against truth, pool 1 (MAE, dex) -- descriptive",
                   ["config"] + keys + [f"{k} (dim)" for k in keys], rec_rows,
                   note="a higher density does not imply a lower parameter error; settings are not chosen "
                        "from this table.")
    raised = np.asarray(arrays["scale_raised"]); capped = np.asarray(arrays["scale_capped"])
    reporter.stat("scale safeguard", f"raised {int(raised.sum())}, capped {int(capped.sum())}",
                  note=f"of {raised.size} (recording, pool, parameter) scales: IQR below eps x prior width "
                       f"(raised) or above the prior width (capped)")
    reporter.stat("production optimizer seconds", f"{np.median(arrays['seconds_production']):.2f} median",
                  note="per pool, the production optimize_elite as is")
    reporter.stat("reference polish", f"{np.median(arrays['reference_polish_gain']):.2e} median gain",
                  note="L-BFGS gain over the best vector found by any configuration; near zero means the "
                       "best configuration already sat at a stationary point")
    reporter.summary()
    reporter.write_report()


def build_parser(description: str) -> argparse.ArgumentParser:
    ev = PARAMETERS.inference.evaluation
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--condition", required=True, choices=("FAB", "INLB"))
    ap.add_argument("--total-time-seconds", type=float, required=True)
    ap.add_argument("--artifact-tag", default=None, help="Estimator tag (none = estimator of record).")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1], help="EVAL tasks (default 0 1).")
    ap.add_argument("--max-sims", type=int, default=0, help="Per-task cap (0 = all).")
    ap.add_argument("--pilot", type=int, default=0, help="Deterministic half-dim, half-bright subset.")
    ap.add_argument("--pool-size", type=int, default=ev.theta_prex_size,
                    help=f"Candidates per pool (default: production {ev.theta_prex_size}).")
    ap.add_argument("--group-size", type=int, default=64, help="Recordings batched per engine call.")
    ap.add_argument("--polish-iterations", type=int, default=500, help="L-BFGS iterations for the reference.")
    ap.add_argument("--base-seed", type=int, default=20260923)
    ap.add_argument("--data-bank-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap
