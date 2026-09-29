#!/usr/bin/env python
"""Linear probe of a frozen detector encoder: how much of each imaging parameter its embedding carries.

Part of the encoder screening (DETECTOR_WORKFLOW.md, the section on the encoder screening; companion note
``SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Embedding_Probe.md``). The persisted detector estimator of the
label given (the canonical one, or a tagged one such as ``CAP256`` through ``--artifact-tag``) embeds
synthetic EVAL videos through its trained ``Complex3DCNN``, exactly as the flow would see them; a ridge
regression from that embedding to the log10 of each imaging parameter is fitted on the fit tasks, its
regularization is selected on a development set, and its held-out error, bias, slope and correlation are
reported per parameter, beside the error of predicting the fit set's mean. A parameter the probe predicts
well is linearly accessible in the embedding; a parameter it does not predict is not shown to be absent.
The probe is a diagnostic of the encoder as trained, not an estimator: nothing is adopted, and the flow is
not touched.

The split is by task (task-disjoint fit, development and held-out sets) when enough EVAL tasks are given;
with no development tasks, a fraction of the fit videos, by simulation index, serves as the development
set (recorded): held-out scoring is then task-disjoint and fit versus development simulation-disjoint.
Each set must hold at least ``MIN_VIDEOS`` videos, or the probe refuses before fitting. Videos previously inspected in an Evaluation are development evidence, not a fresh
verdict.

Outputs, under ``<data_bank>/<posit>/<alias>_<timing>[_<TAG>]_Embedding_Probe/`` (never overwritten):
the report (``.md``), the numbers (``.json``), the embeddings with their theta and split (``.npz``, so a
second probe needs no GPU pass) and a predicted-versus-true figure (``.png``).

Usage::

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Embedding_Probe.py \
        --total-time-seconds 2 --fit-tasks 0 --held-out-tasks 1 [--artifact-tag CAP256] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import __version__ as _PACKAGE_VERSION
from srm_and_sbi_monomer_dimer_alp import artifacts
from srm_and_sbi_monomer_dimer_alp import embedding_space_distance as esd
from srm_and_sbi_monomer_dimer_alp.inference_support import resolve_topology
from srm_and_sbi_monomer_dimer_alp.io import load_theta_set
from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_monomer_dimer_alp.visualization_dli import load_video_set
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow, parameter_keys, parameter_table

MIN_VIDEOS = 20
"""Fewest videos accepted in each of the fit, development and held-out sets; fewer cannot select a penalty
or score a regression meaningfully, and the probe refuses before fitting."""
ALPHAS = tuple(float(a) for a in np.logspace(-3, 4, 15))
"""Ridge penalties tried, relative to the standardized embedding's per-feature variance (the penalty is
multiplied by the number of fit videos, so the grid is comparable across fit-set sizes)."""


# --- data -------------------------------------------------------------------------------------------

def resolve(cfg, args):
    timing = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing)
    root = PARAMETERS.machine.data_bank_root
    paths = cfg.paths.with_condition(args.condition)
    posit = root / paths.posit_subdir
    label = paths.product_label(timing.label, args.artifact_tag)
    return dict(timing=timing, root=root, paths=paths, posit=posit, label=label,
                estimator=posit / f"{paths.project_alias}_{label}_Estimator.npz",
                out_dir=Path(args.out_dir) if args.out_dir else posit / f"{paths.project_alias}_{label}_Embedding_Probe",
                keys=parameter_keys(cfg), table=parameter_table(cfg))


def task_paths(R, task):
    return (R["paths"].video_set_path(task, R["root"], R["timing"].label, compress=True, split="EVAL"),
            R["paths"].theta_set_path(task, R["root"], R["timing"].label, compress=True, split="EVAL"))


def embed_tasks(R, posterior, device, tasks, batch_size):
    """Embeddings ``[N, D]``, log10 theta ``[N, P]``, task and simulation indices, over the tasks given."""
    n_frames = R["timing"].frame_count
    embs, thetas, task_ix, sim_ix = [], [], [], []
    for task in tasks:
        video_path, theta_path = task_paths(R, task)
        videos = load_video_set(str(video_path))
        theta = np.asarray(load_theta_set(str(theta_path), R["table"]), dtype=float)
        if theta.shape[0] != videos.shape[0]:
            raise SystemExit(f"task {task}: {videos.shape[0]} videos but {theta.shape[0]} theta rows")
        if theta.shape[1] != len(R["keys"]):
            raise SystemExit(f"task {task}: theta has {theta.shape[1]} columns for {len(R['keys'])} parameters")
        if (theta <= 0).any():
            raise SystemExit(f"task {task}: a non-positive parameter value; the probe targets log10 values")
        t0 = time.time()
        embs.append(esd.embed_videos(posterior, videos, device=str(device), batch_size=batch_size,
                                     expected_frames=n_frames))
        thetas.append(np.log10(theta))
        task_ix.append(np.full(theta.shape[0], task)); sim_ix.append(np.arange(theta.shape[0]))
        print(f"  task {task}: {theta.shape[0]} videos embedded in {time.time() - t0:.0f} s")
    return np.vstack(embs), np.vstack(thetas), np.concatenate(task_ix), np.concatenate(sim_ix)


def split_masks(task_ix, sim_ix, fit_tasks, dev_tasks, held_out_tasks, dev_fraction):
    """Boolean masks (fit, dev, held-out). Without development tasks, the last ``dev_fraction`` of each fit
    task's videos, by simulation index, is the development set."""
    fit = np.isin(task_ix, fit_tasks); held = np.isin(task_ix, held_out_tasks)
    if dev_tasks:
        dev = np.isin(task_ix, dev_tasks)
    else:
        dev = np.zeros_like(fit)
        for task in fit_tasks:
            sims = sim_ix[task_ix == task]
            cut = int(round(len(sims) * (1.0 - dev_fraction)))
            dev |= (task_ix == task) & (sim_ix >= cut)
        fit = fit & ~dev
    if (fit & dev).any() or (fit & held).any() or (dev & held).any():
        raise SystemExit("the fit, development and held-out sets overlap")
    return fit, dev, held


# --- the probe --------------------------------------------------------------------------------------

def ridge_fit(X, y, alpha):
    """Weights and intercept of ridge regression on standardized ``X`` (penalty ``alpha * n`` on the weights)."""
    n, d = X.shape
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + alpha * n * np.eye(d), Xc.T @ yc)
    return w, ym - xm @ w


def ridge_predict(X, w, b):
    return X @ w + b


def scores(pred, true):
    """Error statistics of a group; a group of fewer than two videos gets NaN statistics and its count."""
    if true.size < 2:
        return dict(mae=float("nan"), bias=float("nan"), rmse=float("nan"), slope=float("nan"),
                    corr=float("nan"), n=int(true.size))
    err = pred - true
    slope = float(np.polyfit(true, pred, 1)[0]) if np.ptp(true) > 0 else float("nan")
    corr = float(np.corrcoef(true, pred)[0, 1]) if np.ptp(true) > 0 and np.ptp(pred) > 0 else float("nan")
    return dict(mae=float(np.abs(err).mean()), bias=float(err.mean()), rmse=float(np.sqrt((err ** 2).mean())),
                slope=slope, corr=corr, n=int(true.size))


def probe(emb, theta, fit, dev, held, keys, alphas=ALPHAS):
    """Per parameter: alpha selected on the development set, refit on fit + development, scored held-out."""
    sizes = dict(fit=int(fit.sum()), development=int(dev.sum()), held_out=int(held.sum()))
    small = {k: v for k, v in sizes.items() if v < MIN_VIDEOS}
    if small:
        raise SystemExit(f"too few videos to probe (at least {MIN_VIDEOS} per set): " +
                         ", ".join(f"{k} {v}" for k, v in small.items()) + f"; sets {sizes}")
    mu, sd = emb[fit].mean(0), emb[fit].std(0) + 1e-12
    X = (emb - mu) / sd
    out = {}
    for j, key in enumerate(keys):
        y = theta[:, j]
        dev_mae = {}
        for alpha in alphas:
            w, b = ridge_fit(X[fit], y[fit], alpha)
            dev_mae[alpha] = float(np.abs(ridge_predict(X[dev], w, b) - y[dev]).mean())
        alpha = min(dev_mae, key=dev_mae.get)
        refit = fit | dev
        w, b = ridge_fit(X[refit], y[refit], alpha)
        pred = ridge_predict(X[held], w, b); true = y[held]
        null_mae = float(np.abs(y[refit].mean() - true).mean())
        median = float(np.median(true))
        low, high = true <= median, true > median
        out[key] = dict(alpha=alpha, dev_mae=dev_mae[alpha], held_out=scores(pred, true), null_mae=null_mae,
                        span_dex=float(np.ptp(y)), by_regime={"low_half": scores(pred[low], true[low]),
                                                              "high_half": scores(pred[high], true[high])},
                        pred=pred, true=true)
    return out


# --- outputs ----------------------------------------------------------------------------------------

def write_figure(result, keys, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(keys); cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    for ax, key, label in zip(np.ravel(axes), keys, labels):
        r = result[key]; t, p = r["true"], r["pred"]
        ax.scatter(t, p, s=4, alpha=0.35, lw=0)
        lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        s = r["held_out"]
        ax.set_title(f"{label}\nMAE {s['mae']:.3f} dex (null {r['null_mae']:.3f}); slope {s['slope']:.2f}; "
                     f"r {s['corr']:.2f}", fontsize=9)
        ax.set_xlabel("true log10"); ax.set_ylabel("probe log10")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Linear probe of the frozen encoder (held-out EVAL videos)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(result, keys, labels, meta, path):
    lines = [f"# Embedding probe of the {meta['label']} detector encoder", "",
             f"Estimator: `{meta['estimator']}` (weights sha256 `{meta['weights_sha256'][:12]}...`, embedding "
             f"width {meta['width']}). Package {meta['package_version']}. Device {meta['device']}.", "",
             f"EVAL tasks: fit {meta['fit_tasks']} ({meta['n_fit']} videos), development "
             f"{meta['dev_tasks'] or f'{meta['dev_fraction']:.0%} of each fit task by simulation index'} "
             f"({meta['n_dev']} videos), held-out {meta['held_out_tasks']} ({meta['n_held']} videos). "
             "Targets are log10 of the physical values, as the recovery metrics of the Evaluation stage.", "",
             "A ridge regression from the standardized embedding to each parameter; its penalty is chosen on "
             "the development set and the model refitted on fit + development. `null` is the error of "
             "predicting the refit set's mean. `span` is the range of the true values over all videos. A "
             "well-predicted parameter is linearly accessible in the embedding; a poorly predicted one is "
             "not thereby shown to be absent from it.", "",
             "| parameter | held-out MAE (dex) | null MAE | bias | slope | corr | span (dex) | alpha | "
             "MAE low half | MAE high half |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, label in zip(keys, labels):
        r = result[key]; s = r["held_out"]
        lines.append(f"| {label} | {s['mae']:.3f} | {r['null_mae']:.3f} | {s['bias']:+.3f} | {s['slope']:.2f} | "
                     f"{s['corr']:.2f} | {r['span_dex']:.2f} | {r['alpha']:.3g} | "
                     f"{r['by_regime']['low_half']['mae']:.3f} | {r['by_regime']['high_half']['mae']:.3f} |")
    lines += ["", "The low and high halves split the held-out videos at the median true value of that "
              "parameter, with the same predictions.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True, help="model window of the estimator (s).")
    ap.add_argument("--condition", default="FAB", choices=LABELING_CONDITIONS)
    ap.add_argument("--artifact-tag", default=None, help="probe the tagged estimator (e.g. CAP256); default canonical.")
    ap.add_argument("--fit-tasks", type=int, nargs="+", required=True, help="EVAL tasks the probe is fitted on.")
    ap.add_argument("--dev-tasks", type=int, nargs="*", default=[], help="EVAL tasks for selecting the penalty.")
    ap.add_argument("--held-out-tasks", type=int, nargs="+", required=True, help="EVAL tasks scored, never fitted.")
    ap.add_argument("--dev-fraction", type=float, default=0.2,
                    help="without --dev-tasks: the last fraction of each fit task's videos, by simulation "
                         "index, forms the development set (default 0.2).")
    ap.add_argument("--batch-size", type=int, default=16, help="videos per embedding forward pass.")
    ap.add_argument("--out-dir", default=None, help="write here instead of the estimator's Posit folder.")
    ap.add_argument("--dry-run", action="store_true", help="resolve inputs and outputs; embed nothing.")
    args = ap.parse_args(argv)
    if not 0.0 < args.dev_fraction < 1.0:
        raise SystemExit("--dev-fraction must lie in (0, 1)")

    cfg = detector_workflow()
    R = resolve(cfg, args)
    tasks = sorted(set(args.fit_tasks) | set(args.dev_tasks) | set(args.held_out_tasks))
    overlap = (set(args.fit_tasks) & set(args.held_out_tasks)) | (set(args.dev_tasks) & set(args.held_out_tasks)) \
        | (set(args.fit_tasks) & set(args.dev_tasks))
    if overlap:
        raise SystemExit(f"tasks in more than one set: {sorted(overlap)}")
    missing = [p for t in tasks for p in task_paths(R, t) if not p.exists()]
    if not R["estimator"].exists():
        missing.insert(0, R["estimator"])
    if R["out_dir"].exists():
        raise SystemExit(f"refusing to overwrite an existing probe: {R['out_dir']}")

    print(f"embedding probe of {R['label']}: estimator {R['estimator'].name}; EVAL tasks fit {args.fit_tasks}, "
          f"dev {args.dev_tasks or f'{args.dev_fraction:.0%} of fit by simulation index'}, held-out {args.held_out_tasks}")
    print(f"  writes: {R['out_dir']}/")
    if missing:
        print("  MISSING:\n    " + "\n    ".join(str(p) for p in missing))
    if args.dry_run:
        print("[DRY RUN] no compute performed." + (" Inputs missing." if missing else ""))
        return 1 if missing else 0
    if missing:
        raise SystemExit("inputs missing (listed above)")

    device = resolve_topology().device
    posterior = artifacts.load_estimator(str(R["estimator"]), device=str(device), expected_parameter_keys=R["keys"])
    posterior.posterior_estimator.to(device)
    print(f"  embedding on {device} ...")
    emb, theta, task_ix, sim_ix = embed_tasks(R, posterior, device, tasks, args.batch_size)
    fit, dev, held = split_masks(task_ix, sim_ix, args.fit_tasks, args.dev_tasks, args.held_out_tasks, args.dev_fraction)
    result = probe(emb, theta, fit, dev, held, R["keys"])

    labmap = {e["KEY"]: (e.get("LABEL") or e["KEY"]) for e in R["table"]}
    labels = [labmap.get(k, k) for k in R["keys"]]
    meta = dict(label=R["label"], estimator=str(R["estimator"]), weights_sha256=posterior.weights_sha256,
                width=int(emb.shape[1]), package_version=_PACKAGE_VERSION, device=str(device),
                fit_tasks=args.fit_tasks, dev_tasks=args.dev_tasks, held_out_tasks=args.held_out_tasks,
                dev_fraction=args.dev_fraction, n_fit=int(fit.sum()), n_dev=int(dev.sum()), n_held=int(held.sum()),
                alphas=list(ALPHAS), keys=list(R["keys"]))
    R["out_dir"].mkdir(parents=True, exist_ok=False)
    stem = R["out_dir"] / f"{R['paths'].project_alias}_{R['label']}_Embedding_Probe"
    np.savez_compressed(f"{stem}.npz", embedding=emb, theta_log10=theta, task=task_ix, sim=sim_ix,
                        fit=fit, dev=dev, held_out=held, keys=np.array(R["keys"]))
    numbers = {k: {kk: vv for kk, vv in v.items() if kk not in ("pred", "true")} for k, v in result.items()}
    Path(f"{stem}.json").write_text(json.dumps(dict(meta=meta, result=numbers), indent=2), encoding="utf-8")
    write_figure(result, R["keys"], labels, f"{stem}.png")
    write_report(result, R["keys"], labels, meta, f"{stem}.md")
    print(f"\n{Path(f'{stem}.md').read_text()}")
    print(f"written: {R['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
