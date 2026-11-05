"""Analysis entry point (detector workflow): the per-window posteriors of the experimental recordings,
read from an existing Experiment product.

ROLE. The detector Experiment stage stores, for every 2 s window of every experimental recording, three
point summaries of ONE posterior (the numerical MAP, the sample geometric median of the draws, the vector
of marginal medians) and the marginal quantiles of the draws. A condition-wide selection such as the
median over the 600 per-window medians conceals the differences between windows and between recordings,
and composes coordinates that were never jointly drawn. This analysis exports every window's complete
MAP, SGM and median vectors with their 50 % and 90 % marginal intervals in physical units, draws the
empirical distributions and the time course of the pilot recordings, inventories all cells, and
reproduces the condition-wide interval summaries. It reads only the saved product: no estimator, no GPU,
no new draws. The joint draws of the pilot windows are collected by the companion
``..._Experiment_Window_Posterior_Draws.py``.

THREE OBJECTS THAT ARE NOT INTERCHANGEABLE. The marginal interval of one window (its posterior width);
the joint relationships inside that one posterior (not visible here: quantiles are marginal); the change
of the point summaries between windows and between cells (what the time series and the pairwise plots of
complete vectors show). The three point summaries are summaries of the SAME conditional distribution,
never three independent measurements, and are never pooled as thirty observations.

CONVENTIONS KEPT. Stored coordinates are the estimator's (log10 of the physical value for every detector
row); a physical value is ten to the power of the stored value, applied to each stored quantile
(``parameterization.to_physical``); the quantile convention of the product (numpy linear interpolation
between order statistics over the window's draws) is never recomputed here. An interval width is computed
INSIDE its window (10^q95 - 10^q05) before any summary over windows; the median of widths need not equal
the difference of separately summarized endpoints. The SGM is the product's own (exact sample medoid in
estimator coordinates, each coordinate divided by its prior width).

WHERE TO RUN. Any machine (CPU only). Post-hoc, user-driven analysis, NOT a canonical pipeline stage and
never wired into the stage dispatcher.

Reads  <data_bank>/<posit>/<alias>_<COND>_<timing>[_<TAG>]_MAP_Experiment/<same>.npz   (+ report.md checksum)
       <data_bank>/<posit>/<alias>_<COND>_<timing>_<SEL tag>_Nuisance_DLI.npz             (the displayed reference)
Writes <data_bank>/<posit>/<alias>_<COND>_<timing>[_<TAG>]_Window_Posterior_Review/       (new folder; refuses an existing one)
       window_estimates.csv (long: one row per window x parameter), window_vectors.csv (wide: one row per window),
       cell_inventory.csv, condition_interval_summary.csv, figures/, report.md, provenance.json

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Review.py \
        --total-time-seconds 2.0 --condition FAB --artifact-tag C128V42A --pilot-cells 26 \
        [--nuisance-tag SEL] [--verified-csv <handoff CSV>] [--experiment-span-seconds 20] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import __version__ as PACKAGE_VERSION
from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from srm_and_sbi_monomer_dimer_alp import provenance
from srm_and_sbi_monomer_dimer_alp.detector_nuisance_dli import artifact_path as nuisance_artifact_path
from srm_and_sbi_monomer_dimer_alp.detector_nuisance_dli import require_nuisance_dli, spec_path as nuisance_spec_path
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming, is_log_row, to_physical
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow, parameter_keys, parameter_table

UNITS = {"mu_r": "pixels (sqrt(2) times Gaussian sigma)", "sigma_r": "dimensionless ln-spread",
         "mu_pc": "photons per active dye per frame", "sigma_pc": "dimensionless ln-spread",
         "prob_photo_bleach": "probability per 100 frames", "lambda_rate": "per second"}
STATS = ("map", "sgm", "median")
STAT_STYLE = {"map": ("MAP", "tab:red", "P"), "sgm": ("SGM", "tab:green", "s"), "median": ("median", "tab:blue", "o")}
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
REVIEW_SUFFIX = "Window_Posterior_Review"
DRAWS_SUFFIX = "Window_Posterior_Draws"


# ---- shared helpers (the draws companion imports these) -------------------------------------------

def sha256_of(path) -> str:
    return provenance.file_sha256(path)


def resolve_context(args):
    """Paths, table and labels shared by the two companions."""
    cfg = detector_workflow()
    condition = str(args.condition)
    keys = parameter_keys(cfg, condition)
    table = parameter_table(cfg, condition)
    if any(not is_log_row(e) for e in table):
        raise SystemExit("every detector row is a log row; a linear row would need a different width rule.")
    timing = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing)
    paths = cfg.paths.with_condition(condition)
    product_label = paths.product_label(timing.label, args.artifact_tag)
    root = PARAMETERS.machine.data_bank_root
    posit = root / paths.posit_subdir
    exp_dir = paths.experiment_recovery_dir(root, product_label)
    return dict(cfg=cfg, condition=condition, keys=keys, table=table, timing=timing, paths=paths,
                product_label=product_label, root=root, posit=posit, exp_dir=exp_dir,
                exp_npz=exp_dir / (exp_dir.name + ".npz"), exp_report=exp_dir / "report.md",
                prior_low=np.array([e["PRIOR_RANGE"][0] for e in table], float),
                prior_high=np.array([e["PRIOR_RANGE"][1] for e in table], float),
                estimator=paths.estimator_path(root, product_label))


def load_experiment_product(ctx):
    """The adopted Experiment product, validated against the artifact schema; returns (arrays, manifest)."""
    if not ctx["exp_npz"].exists():
        raise SystemExit(f"Experiment product missing: {ctx['exp_npz']}")
    with np.load(ctx["exp_npz"], allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    manifest = schema.validate_product(arrays, stage="experiment", source=ctx["exp_npz"].name)
    if list(manifest["parameter_keys"]) != list(ctx["keys"]):
        raise SystemExit(f"parameter order {manifest['parameter_keys']} != {ctx['keys']}")
    if tuple(manifest["quantile_levels"]) != LEVELS:
        raise SystemExit(f"quantile levels {manifest['quantile_levels']} != {LEVELS}")
    kinds = [str(k) for k in arrays["kinds"]]
    if kinds != [ctx["condition"]]:
        raise SystemExit(f"product kinds {kinds} != [{ctx['condition']}]: not this condition's own recordings")
    return arrays, manifest


def load_sel(ctx, nuisance_tag):
    """The displayed reference: the selection_user Nuisance_DLI vector (physical) and its identity."""
    nu = require_nuisance_dli(ctx["posit"], ctx["paths"].project_alias, ctx["timing"].label, nuisance_tag=nuisance_tag)
    if list(nu.parameter_keys) != list(ctx["keys"]) or nu.fixed_vector_log10 is None:
        raise SystemExit(f"Nuisance_DLI {nuisance_tag}: not a selection_user artifact over {ctx['keys']}")
    ap = nuisance_artifact_path(ctx["posit"], ctx["paths"].project_alias, ctx["timing"].label, nuisance_tag)
    sp = nuisance_spec_path(ctx["posit"], ctx["paths"].project_alias, ctx["timing"].label, nuisance_tag)
    log10 = np.asarray(nu.fixed_vector_log10, float)
    return dict(tag=nuisance_tag, log10=log10, physical=to_physical(log10, ctx["table"]),
                artifact=str(ap), artifact_sha256=sha256_of(ap),
                spec=str(sp) if sp.exists() else None, spec_sha256=sha256_of(sp) if sp.exists() else None)


def window_times(manifest, timing):
    g = manifest["window_geometry"]
    n_frames, step = int(g["n_frames"]), int(g["step_frames"])
    frame_s = timing.total_time_seconds / timing.frame_count
    return n_frames, step, frame_s


def physical_quantiles(q_log10, table):
    """(N, D, Q) stored quantiles -> physical, level by level (the transform of the quantile)."""
    out = np.empty_like(q_log10, dtype=float)
    for j in range(q_log10.shape[-1]):
        out[..., j] = to_physical(q_log10[..., j], table)
    return out


def check(checks, name, ok, detail):
    checks.append((name, "PASS" if ok else "FAIL", detail))
    return ok


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v):
    v = float(v)
    if v == 0 or not np.isfinite(v):
        return f"{v:g}"
    a = abs(v)
    return f"{v:.4f}" if a < 1 else (f"{v:.3f}" if a < 10 else (f"{v:.2f}" if a < 1000 else f"{v:.0f}"))


def write_csv(path, headers, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)


# ---- the review ------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True)
    ap.add_argument("--condition", required=True, choices=("FAB", "INLB"))
    ap.add_argument("--artifact-tag", default=None, help="Estimator tag of the Experiment product (none = estimator of record).")
    ap.add_argument("--experiment-span-seconds", type=int, default=20)
    ap.add_argument("--pilot-cells", default="", help="Comma-separated cells drawn in detail (all cells enter the tables).")
    ap.add_argument("--nuisance-tag", default="SEL", help="The displayed reference vector (a selection_user Nuisance_DLI).")
    ap.add_argument("--verified-csv", default=None, help="Optional condition-wide interval CSV to reproduce (handoff).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    t_start = time.time()
    ctx = resolve_context(args)
    keys, table = ctx["keys"], ctx["table"]
    D = len(keys)
    out_dir = ctx["posit"] / f"{ctx['paths'].project_alias}_{ctx['product_label']}_{REVIEW_SUFFIX}"
    pilot_cells = [int(c) for c in args.pilot_cells.split(",") if c.strip()]
    print(f"Window-posterior review ({ctx['condition']}, {ctx['product_label']})\n"
          f"  reads   : {ctx['exp_npz']}\n  reference: Nuisance_DLI tag {args.nuisance_tag}\n"
          f"  pilot cells: {pilot_cells or 'none (tables only)'}\n  writes  : {out_dir}")
    if args.dry_run:
        print("[DRY RUN] inputs resolved; nothing read beyond this point, nothing written.")
        return 0
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite an existing review folder: {out_dir}")
    startup = provenance.code_provenance()
    arrays, manifest = load_experiment_product(ctx)
    sel = load_sel(ctx, args.nuisance_tag)
    n_frames, step, frame_s = window_times(manifest, ctx["timing"])
    checks = []

    # ---- identity and inventory checks --------------------------------------------------------------
    cell = arrays["cell"].astype(int); chunk = arrays["chunk"].astype(int); kind_index = arrays["kind_index"].astype(int)
    N = cell.shape[0]
    ids = schema.observation_ids(arrays, "experiment")
    try:
        schema.assert_unique_observations(ids, source=ctx["exp_npz"].name)
        unique_ok = True
    except schema.SchemaError as exc:
        unique_ok = False; print(exc)
    check(checks, "unique_observation_ids", unique_ok, f"{N} (kind_index, cell, chunk) tuples, all distinct")
    cells = sorted(set(cell.tolist()))
    per_cell_chunks = {c: sorted(chunk[cell == c].tolist()) for c in cells}
    ten_ok = all(v == list(range(10)) for v in per_cell_chunks.values())
    check(checks, "ten_chunks_per_cell", ten_ok, f"{len(cells)} cells; every cell carries chunks 0..9 exactly" if ten_ok
          else f"cells with other chunk sets: {[c for c, v in per_cell_chunks.items() if v != list(range(10))]}")
    check(checks, "single_kind", bool(np.all(kind_index == 0)), f"kind_index all 0 = {ctx['condition']}")
    check(checks, "parameter_order", list(manifest["parameter_keys"]) == keys, ", ".join(keys))
    check(checks, "window_geometry", manifest["window_geometry"] == {"n_frames": n_frames, "step_frames": step, "span_frames": 1000}
          and n_frames == ctx["timing"].frame_count, f"{manifest['window_geometry']}; model window {ctx['timing'].frame_count} frames")
    check(checks, "pilot_cells_present", all(c in cells for c in pilot_cells), f"{pilot_cells} in the product")
    est_sha = sha256_of(ctx["estimator"]) if ctx["estimator"].exists() else None
    check(checks, "estimator_artifact_present", est_sha is not None, f"{ctx['estimator'].name} sha256 {est_sha}")
    finite_ok = all(np.all(np.isfinite(arrays[k])) for k in ("map_estimate", "posterior_sgm", "posterior_quantiles", "scores"))
    check(checks, "finite_estimates", finite_ok, "map_estimate, posterior_sgm, posterior_quantiles, scores all finite")
    mono_ok = bool(np.all(np.diff(arrays["posterior_quantiles"], axis=-1) >= 0))
    check(checks, "quantiles_monotone", mono_ok, "q05 <= q25 <= q50 <= q75 <= q95 in every window and parameter")

    # ---- physical-unit tables ----------------------------------------------------------------------------
    map_l, sgm_l, q_l = arrays["map_estimate"].astype(float), arrays["posterior_sgm"].astype(float), arrays["posterior_quantiles"].astype(float)
    map_p, sgm_p, q_p = to_physical(map_l, table), to_physical(sgm_l, table), physical_quantiles(q_l, table)
    med_p = q_p[:, :, 2]; med_l = q_l[:, :, 2]
    w50 = q_p[:, :, 3] - q_p[:, :, 1]; w90 = q_p[:, :, 4] - q_p[:, :, 0]
    lo, hi = ctx["prior_low"], ctx["prior_high"]
    sel_in90 = (q_l[:, :, 0] <= sel["log10"]) & (sel["log10"] <= q_l[:, :, 4])
    map_in90 = (q_l[:, :, 0] <= map_l) & (map_l <= q_l[:, :, 4])
    start_f = chunk * step; stop_f = start_f + n_frames
    order = np.lexsort((chunk, cell))
    long_rows, wide_rows = [], []
    for i in order:
        base = [ctx["condition"], int(cell[i]), int(chunk[i]), int(start_f[i]), int(stop_f[i]), f"{start_f[i]*frame_s:g}", f"{stop_f[i]*frame_s:g}"]
        wide = list(base) + [float(arrays["scores"][i])]
        for j, k in enumerate(keys):
            long_rows.append(base + [k, UNITS[k], sel["physical"][j], map_p[i, j], sgm_p[i, j], q_p[i, j, 0], q_p[i, j, 1], med_p[i, j], q_p[i, j, 3], q_p[i, j, 4],
                                     w50[i, j], w90[i, j], map_l[i, j], sgm_l[i, j], med_l[i, j], int(sel_in90[i, j]), int(map_in90[i, j]),
                                     int(map_l[i, j] < lo[j] or map_l[i, j] > hi[j]), int(med_l[i, j] < lo[j] or med_l[i, j] > hi[j]), float(arrays["scores"][i])])
        for stat, arr in (("map", map_p), ("sgm", sgm_p), ("median", med_p), ("q05", q_p[:, :, 0]), ("q95", q_p[:, :, 4])):
            wide += [arr[i, j] for j in range(D)]
        wide_rows.append(wide)
    out_dir.mkdir(parents=True); fig_dir = out_dir / "figures"; fig_dir.mkdir()
    write_csv(out_dir / "window_estimates.csv",
              ["condition", "cell", "chunk", "start_frame", "stop_frame_exclusive", "start_seconds", "stop_seconds", "parameter", "unit", "sel",
               "map", "sgm", "q05", "q25", "median", "q75", "q95", "width50", "width90", "map_log10", "sgm_log10", "median_log10",
               "sel_inside90", "map_inside90", "map_outside_prior", "median_outside_prior", "map_score_estimator_coordinates"], long_rows)
    write_csv(out_dir / "window_vectors.csv",
              ["condition", "cell", "chunk", "start_frame", "stop_frame_exclusive", "start_seconds", "stop_seconds", "map_score"]
              + [f"{s}_{k}" for s in ("map", "sgm", "median", "q05", "q95") for k in keys], wide_rows)

    # ---- condition-wide interval summary (reproduction of the verified CSV) ---------------------------------
    summary_rows, repro = [], []
    verified = {}
    if args.verified_csv:
        with open(args.verified_csv) as fh:
            for r in csv.DictReader(fh):
                if r["condition"] == ctx["condition"]:
                    verified[r["parameter"]] = r
    for j, k in enumerate(keys):
        row = {"condition": ctx["condition"], "parameter": k, "unit": UNITS[k], "n_windows": N, "sel": sel["physical"][j],
               "median_lower90": np.median(q_p[:, j, 0]), "median_lower50": np.median(q_p[:, j, 1]), "median_posterior_median": np.median(med_p[:, j]),
               "median_upper50": np.median(q_p[:, j, 3]), "median_upper90": np.median(q_p[:, j, 4]), "median_width90": np.median(w90[:, j]),
               "median_width50": np.median(w50[:, j]), "median_sgm": np.median(sgm_p[:, j]), "median_map": np.median(map_p[:, j])}
        for p in (5, 25, 50, 75, 95):
            row[f"window_median_p{p:02d}"] = np.percentile(med_p[:, j], p)
        row["windows_with_sel_inside90"] = int(sel_in90[:, j].sum()); row["windows_with_map_inside90"] = int(map_in90[:, j].sum())
        row["windows_median_outside_prior"] = int(((med_l[:, j] < lo[j]) | (med_l[:, j] > hi[j])).sum())
        summary_rows.append(row)
        if k in verified:
            v = verified[k]; worst = 0.0
            for col in ("median_lower90", "median_lower50", "median_posterior_median", "median_upper50", "median_upper90", "median_width90", "median_sgm",
                        "window_median_p05", "window_median_p25", "window_median_p50", "window_median_p75", "window_median_p95"):
                worst = max(worst, abs(float(v[col]) - float(row[col])) / max(abs(float(v[col])), 1e-12))
            cnt_ok = int(v["windows_with_sel_inside90"]) == row["windows_with_sel_inside90"]
            repro.append((k, worst, cnt_ok))
    if repro:
        check(checks, "verified_interval_summary_reproduced", all(w < 1e-6 and c for _, w, c in repro),
              "; ".join(f"{k}: max relative deviation {w:.1e}, SEL-inside count {'equal' if c else 'DIFFERS'}" for k, w, c in repro))
    write_csv(out_dir / "condition_interval_summary.csv", list(summary_rows[0].keys()), [list(r.values()) for r in summary_rows])

    # ---- cell inventory (every cell; identity preserved) --------------------------------------------------------
    inv_rows = []
    for c in cells:
        m = cell == c
        for j, k in enumerate(keys):
            mm, sg, me = map_p[m, j], sgm_p[m, j], med_p[m, j]
            first, last = med_l[m][np.argmin(chunk[m]), j], med_l[m][np.argmax(chunk[m]), j]
            inv_rows.append([ctx["condition"], c, k, int(m.sum()), np.median(mm), np.median(sg), np.median(me), me.min(), me.max(),
                             np.median(w50[m, j]), np.median(w90[m, j]), last - first, int(sel_in90[m, j].sum()),
                             int(((med_l[m, j] < lo[j]) | (med_l[m, j] > hi[j])).sum())])
    write_csv(out_dir / "cell_inventory.csv",
              ["condition", "cell", "parameter", "n_windows", "cell_median_of_map", "cell_median_of_sgm", "cell_median_of_median",
               "min_window_median", "max_window_median", "cell_median_width50", "cell_median_width90",
               "median_change_first_to_last_dex", "windows_sel_inside90", "windows_median_outside_prior"], inv_rows)

    # ---- figures ----------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [e["LABEL"] for e in table]
    cmap = matplotlib.colormaps["viridis"]
    figures = []

    def axis_setup(ax, j, physical=True):
        ax.set_title(f"{keys[j]}  {labels[j]}", fontsize=10)
        if physical:
            ax.axvline(sel["physical"][j], color="k", ls="--", lw=1, label=f"{sel['tag']} {fmt(sel['physical'][j])}")
            for e in (10 ** lo[j], 10 ** hi[j]):
                ax.axvline(e, color="gray", ls=":", lw=0.8)

    for c in pilot_cells:
        m = np.where(cell == c)[0]; m = m[np.argsort(chunk[m])]
        t_mid = (start_f[m] + n_frames / 2) * frame_s
        # (a) ten-point empirical distributions of each statistic, per parameter
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for j, ax in enumerate(axes.ravel()):
            for stat, arr in (("map", map_p), ("sgm", sgm_p), ("median", med_p)):
                name, color, marker = STAT_STYLE[stat]
                v = np.sort(arr[m, j]); ax.step(v, np.arange(1, 11) / 10, where="post", color=color, lw=1)
                ax.scatter(arr[m, j], np.arange(1, 11)[np.argsort(np.argsort(arr[m, j]))] / 10, color=color, marker=marker, s=28, label=f"{name} (10 windows)")
            axis_setup(ax, j); ax.set_ylim(0, 1.05); ax.set_ylabel("empirical CDF over the 10 windows"); ax.set_xlabel(UNITS[keys[j]], fontsize=8)
            if j == 0: ax.legend(fontsize=7)
        fig.suptitle(f"{ctx['condition']} cell {c}: ten-window empirical distributions of the three point summaries, physical units "
                     f"(dotted: prior box edges; each statistic is ten values, one per window; the three are summaries of the same posterior, not pooled)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); p = fig_dir / f"ecdf_cell_{c}.png"; fig.savefig(p, dpi=130); plt.close(fig); figures.append(p.name)
        # (b) time series with the shared within-window intervals
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for j, ax in enumerate(axes.ravel()):
            ax.fill_between(t_mid, q_p[m, j, 0], q_p[m, j, 4], color="tab:blue", alpha=0.12, label="90 % marginal interval (draws)")
            ax.fill_between(t_mid, q_p[m, j, 1], q_p[m, j, 3], color="tab:blue", alpha=0.25, label="50 % marginal interval (draws)")
            for stat, arr in (("map", map_p), ("sgm", sgm_p), ("median", med_p)):
                name, color, marker = STAT_STYLE[stat]
                ax.plot(t_mid, arr[m, j], color=color, marker=marker, ms=5, lw=1, label=name)
            ax.axhline(sel["physical"][j], color="k", ls="--", lw=1, label=f"{sel['tag']} {fmt(sel['physical'][j])}")
            for e in (10 ** lo[j], 10 ** hi[j]):
                ax.axhline(e, color="gray", ls=":", lw=0.8)
            ax.set_title(f"{keys[j]}  {labels[j]}", fontsize=10); ax.set_xlabel("window midpoint (s)"); ax.set_ylabel(UNITS[keys[j]], fontsize=8)
            if keys[j] in ("prob_photo_bleach", "lambda_rate", "mu_pc"): ax.set_yscale("log")
            if j == 0: ax.legend(fontsize=7)
        fig.suptitle(f"{ctx['condition']} cell {c}: the three point summaries over the ten windows with each window's own 50 % and 90 % marginal intervals "
                     f"(physical units; dotted: prior box edges; the bands are marginal, per window, and say nothing about joint structure)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); p = fig_dir / f"timeseries_cell_{c}.png"; fig.savefig(p, dpi=130); plt.close(fig); figures.append(p.name)
        # (c) pairwise plots of the complete vectors, colored by window time
        fig, axes = plt.subplots(D - 1, D - 1, figsize=(16, 16))
        for a in range(D):
            for b in range(D):
                if b >= a: 
                    if a < D - 1 and b < D - 1 and b > a: axes[a, b].axis("off")
                    continue
                ax = axes[a - 1, b]
                for stat, arr in (("map", map_p), ("sgm", sgm_p), ("median", med_p)):
                    name, color, marker = STAT_STYLE[stat]
                    ax.scatter(arr[m, b], arr[m, a], c=t_mid, cmap=cmap, marker=marker, s=34, edgecolors=color, linewidths=0.8, vmin=0, vmax=t_mid.max())
                    ax.plot(arr[m, b], arr[m, a], color=color, lw=0.5, alpha=0.5)
                ax.scatter([sel["physical"][b]], [sel["physical"][a]], marker="*", s=120, color="k", zorder=5)
                ax.set_xlabel(keys[b], fontsize=8); ax.set_ylabel(keys[a], fontsize=8); ax.tick_params(labelsize=7)
        fig.suptitle(f"{ctx['condition']} cell {c}: pairwise plots of the complete per-window vectors (MAP plus, SGM square, median circle; fill = window time, "
                     f"dark early to bright late; black star = {sel['tag']}).\nThese show how the point estimates move together ACROSS windows; "
                     f"they do not show within-window joint uncertainty.", fontsize=10)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, t_mid.max()))
        fig.colorbar(sm, cax=fig.add_axes([0.72, 0.62, 0.02, 0.25]), label="window midpoint (s)")
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.04, top=0.93, hspace=0.35, wspace=0.35)
        p = fig_dir / f"pairwise_cell_{c}.png"; fig.savefig(p, dpi=110); plt.close(fig); figures.append(p.name)
    # (d) between-cell overview: every cell's median of window medians with its within-cell range
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, ax in enumerate(axes.ravel()):
        cm_ = np.array([np.median(med_p[cell == c, j]) for c in cells]); mn = np.array([med_p[cell == c, j].min() for c in cells]); mx = np.array([med_p[cell == c, j].max() for c in cells])
        ax.vlines(cells, mn, mx, color="tab:blue", lw=1, alpha=0.6, label="range of the 10 window medians")
        ax.scatter(cells, cm_, color="tab:blue", s=14, zorder=3, label="cell median of the window medians")
        ax.axhline(sel["physical"][j], color="k", ls="--", lw=1, label=f"{sel['tag']}")
        for e in (10 ** lo[j], 10 ** hi[j]): ax.axhline(e, color="gray", ls=":", lw=0.8)
        for c in pilot_cells: ax.axvline(c, color="tab:red", lw=0.8, alpha=0.5)
        ax.set_title(f"{keys[j]}  {labels[j]}", fontsize=10); ax.set_xlabel("cell index"); ax.set_ylabel(UNITS[keys[j]], fontsize=8)
        if keys[j] in ("prob_photo_bleach", "lambda_rate", "mu_pc"): ax.set_yscale("log")
        if j == 0: ax.legend(fontsize=7)
    fig.suptitle(f"{ctx['condition']}: between-cell overview of the per-window posterior medians (one marker and range per cell; red lines: pilot cells; "
                 f"no pooling across cells is implied)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p = fig_dir / "cells_overview.png"; fig.savefig(p, dpi=130); plt.close(fig); figures.append(p.name)
    # (e) descriptive pooled histograms, labeled as a mixture
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for j, ax in enumerate(axes.ravel()):
        for stat, arr in (("map", map_p), ("sgm", sgm_p), ("median", med_p)):
            name, color, _ = STAT_STYLE[stat]
            ax.hist(np.log10(arr[:, j]), bins=40, histtype="step", color=color, label=name)
        ax.axvline(sel["log10"][j], color="k", ls="--", lw=1); ax.axvspan(lo[j], hi[j], color="gray", alpha=0.08)
        ax.set_title(f"{keys[j]}  {labels[j]}", fontsize=10); ax.set_xlabel("log10 physical value"); ax.set_ylabel("windows")
        if j == 0: ax.legend(fontsize=7)
    fig.suptitle(f"{ctx['condition']}: DESCRIPTIVE MIXTURE over {N} windows of {len(cells)} cells and ten times each (not one posterior for a shared parameter); "
                 f"gray: prior box; dashed: {sel['tag']}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); p = fig_dir / "pooled_mixture.png"; fig.savefig(p, dpi=130); plt.close(fig); figures.append(p.name)

    # ---- report ----------------------------------------------------------------------------------------------------
    rep = [f"# Window-posterior review ({ctx['condition']}, {ctx['product_label']})", "",
           f"Source product `{ctx['exp_npz'].name}` (sha256 {sha256_of(ctx['exp_npz'])}; report.md sha256 {sha256_of(ctx['exp_report']) if ctx['exp_report'].exists() else 'absent'}); "
           f"estimator checkpoint sha256 recorded by the product {manifest['checkpoint_sha256']}; pool mode {manifest['pool_mode']} ({manifest['draw_label']}s); "
           f"{manifest['n_summary_draws']} summary draws per window; SGM scale {manifest['sgm_scale']} in {manifest['sgm_coordinates']} coordinates. "
           f"Reference {sel['tag']}: `{Path(sel['artifact']).name}` (sha256 {sel['artifact_sha256']}). No new draws were made; every value below is read from the saved "
           f"product and transformed to physical units (10 to the power of each stored quantile). Package {PACKAGE_VERSION}.", "",
           "## Checks", "", md_table(["check", "result", "detail"], checks), "",
           "## Condition-wide interval summary (over all windows; a summary of summaries, not one posterior)", "",
           md_table(["parameter", "SEL", "median q05", "median q25", "median of medians", "median q75", "median q95", "median width90", "median SGM", "median MAP", "SEL inside 90 %", "MAP inside 90 %", "median outside prior"],
                    [[r["parameter"], fmt(r["sel"]), fmt(r["median_lower90"]), fmt(r["median_lower50"]), fmt(r["median_posterior_median"]), fmt(r["median_upper50"]), fmt(r["median_upper90"]),
                      fmt(r["median_width90"]), fmt(r["median_sgm"]), fmt(r["median_map"]), f"{r['windows_with_sel_inside90']}/{N}", f"{r['windows_with_map_inside90']}/{N}", f"{r['windows_median_outside_prior']}/{N}"] for r in summary_rows]), ""]
    for c in pilot_cells:
        m = np.where(cell == c)[0]; m = m[np.argsort(chunk[m])]
        rep += [f"## Pilot cell {c}: per-window values (physical units)", ""]
        for j, k in enumerate(keys):
            rows = [[int(chunk[i]), f"{start_f[i]*frame_s:g}-{stop_f[i]*frame_s:g} s", fmt(map_p[i, j]), fmt(sgm_p[i, j]), fmt(med_p[i, j]),
                     f"[{fmt(q_p[i, j, 1])}, {fmt(q_p[i, j, 3])}]", f"[{fmt(q_p[i, j, 0])}, {fmt(q_p[i, j, 4])}]", fmt(w90[i, j]), "yes" if sel_in90[i, j] else "no"] for i in m]
            rep += [f"**{k}** ({UNITS[k]}; {sel['tag']} {fmt(sel['physical'][j])}; prior [{fmt(10**lo[j])}, {fmt(10**hi[j])}])", "",
                    md_table(["chunk", "window", "MAP", "SGM", "median", "50 % interval", "90 % interval", "width90", "SEL inside 90 %"], rows), ""]
        rep += [f"Figures: `figures/ecdf_cell_{c}.png`, `figures/timeseries_cell_{c}.png`, `figures/pairwise_cell_{c}.png`.", ""]
    rep += ["## Files", "", "- `window_estimates.csv`: one row per window and parameter, every cell (physical values, stored log10 values, widths, flags).",
            "- `window_vectors.csv`: one row per window, the complete MAP, SGM, median, q05 and q95 vectors.",
            "- `cell_inventory.csv`: one row per cell and parameter (cell identity preserved).",
            "- `condition_interval_summary.csv`: the condition-wide summary above" + (" (reproduces the handoff CSV; see the check)" if repro else "") + ".",
            f"- `figures/`: {', '.join(figures)}.", f"- `provenance.json`: sources, checksums, versions, command.", "",
            "## Reading rules", "", "- Each window's interval is marginal and per window; the time series shows changes between windows; neither shows joint structure within a window (see the draws companion).",
            "- The three point summaries describe one conditional distribution each; they are never pooled as thirty observations.",
            "- The pooled histogram is a descriptive mixture of cells and times, not a posterior for a shared parameter.",
            f"- Runtime {time.time() - t_start:.1f} s (CPU)."]
    (out_dir / "report.md").write_text("\n".join(rep) + "\n")
    prov = provenance.analysis_run_record(startup, argv=sys.argv, files=provenance.IMPLEMENTATION_FILES + (str(Path(__file__).relative_to(provenance.repo_root())),),
                                          extra={"purpose": "window-posterior review of an existing detector Experiment product (no new draws)",
                                                 "sources": {"experiment_npz": str(ctx["exp_npz"]), "experiment_npz_sha256": sha256_of(ctx["exp_npz"]),
                                                             "experiment_report_sha256": sha256_of(ctx["exp_report"]) if ctx["exp_report"].exists() else None,
                                                             "estimator": str(ctx["estimator"]), "estimator_sha256": est_sha, "experiment_checkpoint_sha256": manifest["checkpoint_sha256"],
                                                             "nuisance_dli": sel["artifact"], "nuisance_dli_sha256": sel["artifact_sha256"], "nuisance_spec": sel["spec"], "nuisance_spec_sha256": sel["spec_sha256"],
                                                             "verified_csv": args.verified_csv},
                                                 "sel_physical": dict(zip(keys, map(float, sel["physical"]))), "n_windows": int(N), "cells": cells, "pilot_cells": pilot_cells,
                                                 "checks": [{"name": n, "result": r, "detail": d} for n, r, d in checks], "runtime_seconds": time.time() - t_start})
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    print("\n".join(f"  {n:40s} {r}  {d[:110]}" for n, r, d in checks))
    print(f"wrote {out_dir} ({len(figures)} figures) in {time.time() - t_start:.1f} s")
    return 0 if all(r == "PASS" for _, r, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
