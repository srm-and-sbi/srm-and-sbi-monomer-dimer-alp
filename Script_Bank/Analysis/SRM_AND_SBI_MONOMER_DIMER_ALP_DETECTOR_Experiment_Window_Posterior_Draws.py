"""Analysis entry point (detector workflow): retain the complete joint posterior draws of every window of
selected experimental recordings, and examine the joint structure inside each window.

ROLE. The adopted Experiment products keep, per window, three point summaries and the marginal quantiles
of 10,000 draws; the draws themselves were not stored and cannot be reconstructed from their quantiles.
This companion draws a FRESH cloud per window with the same trained estimator, the same input conversion
and normalization, the same window geometry and the same sampling mode as the source Experiment, and
preserves every complete draw (``posterior_samples_cloud``, windows x draws x parameters, float32,
estimator coordinates). It recomputes the draw-derived summaries (marginal quantiles, median vector, SGM)
FROM THE RETAINED CLOUD, and joins the source product's MAP, quantiles and SGM by (kind, cell, chunk) as
references from the source run -- they are never mixed with the fresh draws as if they shared one Monte
Carlo sample. No MAP is re-optimized: the adopted MAP already exists and this collection is a different
product with its own name, so the adopted Experiment directory is untouched.

ROUTE. A collection companion calling the engine's own functions (``experiment_support.read_cell_chunks``,
``artifacts.load_estimator``, ``evaluation.posterior_summary`` with ``return_samples`` and ``return_sgm``,
``evaluation.sample_geometric_median`` with the prior-width scale), not a rerun of the Experiment stage,
whose ``--artifact-tag`` selects both its estimator and its output directory.

WHAT IT EXAMINES. Per window: Pearson and Spearman correlation matrices of the cloud in estimator
coordinates, Pearson in physical coordinates; joint plots of named parameter pairs for chunks 0, 4 and 9
(physical axes, a documented fixed-seed subset of the draws for the scatter, the full cloud for every
number); where the reference vector (``SEL``) and the source MAP sit relative to the cloud, as marginal
membership and as a joint depth (the share of draws farther from the window's SGM than the reference, in
prior-scaled estimator coordinates). Three objects stay apart: marginal uncertainty within a window, joint
relationships within that same posterior, changes between windows or cells.

SAMPLING MODE. The source mode is kept (``unrestricted``: direct flow draws that may fall outside the
prior box). Outside-prior draws are counted and labeled, never rejected, clipped or trimmed. A seed
documents the attempt; it does not promise bitwise reproducibility across devices or library versions.

WHERE TO RUN. Locally on a GPU workstation (the PC profile ``mars_pc``) or rcl01: one estimator pass
per window. Post-hoc, user-driven analysis, NOT a canonical pipeline stage and never wired into the
stage dispatcher. No HPC submission.

Reads  <data_bank>/<posit>/<alias>_<COND>_<timing>[_<TAG>]_MAP_Experiment/<same>.npz   (source references)
       <data_bank>/<posit>/<alias>_<COND>_<timing>[_<TAG>]_Estimator.npz              (the adopted estimator)
       <data_bank>/<experiment>/Experiment_<COND>_Cell_<n>_<span>S_RAW.tif             (the recordings)
       <data_bank>/<posit>/<alias>_<COND>_<timing>_<SEL tag>_Nuisance_DLI.npz           (the displayed reference)
Writes <data_bank>/<posit>/<alias>_<COND>_<timing>[_<TAG>]_Window_Posterior_Draws[_<LABEL>]/   (new folder; refuses an existing one;
       <LABEL> = the optional output-only --collection-label that keeps several collections of one estimator apart)
       <same>.npz (the retained clouds + fresh summaries + joined source references + collection manifest),
       window_summaries.csv, correlations.csv, joint_depth.csv, figures/, report.md, provenance.json

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Draws.py \
        --total-time-seconds 2.0 --condition FAB --artifact-tag C128V42A --cells 26 --seed 20261103 \
        [--posterior-samples 10000] [--nuisance-tag SEL] [--scatter-subset 2000] [--collection-label CELL4] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Review import (   # noqa: E402
    DRAWS_SUFFIX, LEVELS, UNITS, check, fmt, load_experiment_product, load_sel, md_table, physical_quantiles,
    resolve_context, sha256_of, window_times, write_csv)
from srm_and_sbi_monomer_dimer_alp import __version__ as PACKAGE_VERSION                    # noqa: E402
from srm_and_sbi_monomer_dimer_alp import artifacts, provenance                              # noqa: E402
from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema                          # noqa: E402
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg                      # noqa: E402
from srm_and_sbi_monomer_dimer_alp.evaluation import (QUANTILE_LEVELS, draw_label, posterior_summary,   # noqa: E402
                                                      prior_scale, sample_geometric_median)
from srm_and_sbi_monomer_dimer_alp.experiment_support import preflight_recordings, read_cell_chunks      # noqa: E402
from srm_and_sbi_monomer_dimer_alp.inference_support import resolve_topology               # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, to_physical          # noqa: E402

PAIRS = (("mu_r", "mu_pc"), ("mu_pc", "sigma_pc"), ("mu_r", "sigma_r"), ("sigma_pc", "lambda_rate"),
         ("prob_photo_bleach", "lambda_rate"), ("mu_pc", "prob_photo_bleach"))
JOINT_CHUNKS = (0, 4, 9)
COLLECTION_STAGE = "window_posterior_draws"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True)
    ap.add_argument("--condition", required=True, choices=("FAB", "INLB"))
    ap.add_argument("--artifact-tag", default=None)
    ap.add_argument("--cells", required=True, help="Comma-separated cell indices whose windows are sampled.")
    ap.add_argument("--experiment-span-seconds", type=int, default=20)
    ap.add_argument("--posterior-samples", type=int, default=None, help="Draws per window (default: the source product's count).")
    ap.add_argument("--seed", type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v), default=None)
    ap.add_argument("--nuisance-tag", default="SEL")
    ap.add_argument("--collection-label", default=None,
                    help="Output-only label appended to the collection folder and file names (e.g. CELL4), so several "
                         "collections of the same estimator and source Experiment coexist; it never changes the inputs.")
    ap.add_argument("--scatter-subset", type=int, default=2000, help="Draws shown per scatter panel (display only; numbers use the full cloud).")
    ap.add_argument("--scatter-seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    t_start = time.time()
    ctx = resolve_context(args)
    keys, table = ctx["keys"], ctx["table"]
    D = len(keys)
    cells = [int(c) for c in args.cells.split(",") if c.strip()]
    label = (args.collection_label or "").strip()
    if label and not re.fullmatch(r"[A-Za-z0-9]+", label):
        raise SystemExit(f"--collection-label must be alphanumeric (got {label!r})")
    out_dir = ctx["posit"] / (f"{ctx['paths'].project_alias}_{ctx['product_label']}_{DRAWS_SUFFIX}" + (f"_{label}" if label else ""))
    npz_path = out_dir / (out_dir.name + ".npz")
    arrays, manifest = load_experiment_product(ctx)
    src_sha_before = sha256_of(ctx["exp_npz"])
    n_frames, step, frame_s = window_times(manifest, ctx["timing"])
    n_draws = int(args.posterior_samples or manifest["n_summary_draws"])
    pool_mode = str(manifest["pool_mode"])
    sgm_scale = prior_scale(table)
    if not np.allclose(sgm_scale, manifest["sgm_scale"]):
        raise SystemExit(f"prior widths {sgm_scale.tolist()} != the source SGM scale {manifest['sgm_scale']}")
    tifs = {c: ctx["paths"].experiment_video_path(ctx["condition"], c, args.experiment_span_seconds, ctx["root"]) for c in cells}
    missing = [str(p) for p in tifs.values() if not p.exists()]
    if missing:
        raise SystemExit("recording(s) missing:\n  " + "\n  ".join(missing))
    shapes = preflight_recordings(list(tifs.values()))
    topo = resolve_topology()
    import torch
    print(f"Window-posterior draws ({ctx['condition']}, {ctx['product_label']})\n"
          f"  source product : {ctx['exp_npz']}\n    sha256 {src_sha_before}; checkpoint {manifest['checkpoint_sha256']}; pool mode {pool_mode}; "
          f"window geometry {manifest['window_geometry']}\n  estimator      : {ctx['estimator']}\n"
          f"  recordings     : " + ", ".join(f"cell {c} {shapes[p]}" for c, p in tifs.items()) + "\n"
          f"  draws/window   : {n_draws} ({draw_label(pool_mode)}s); SGM scale {sgm_scale.tolist()} (estimator coordinates)\n"
          f"  seed           : {args.seed}\n  device         : {topo.device} ({topo.backend}); torch {torch.__version__}\n"
          f"  label          : {label or '(none)'} (output-only)\n  writes         : {out_dir}")
    if out_dir.exists():
        raise SystemExit(f"refusing to write into an existing collection folder: {out_dir}")
    if args.dry_run:
        print("[DRY RUN] inputs and destination checked; nothing sampled, nothing written.")
        return 0
    startup = provenance.code_provenance()
    sel = load_sel(ctx, args.nuisance_tag)
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = topo.device
    posterior = artifacts.load_estimator(ctx["estimator"], device=str(device), expected_parameter_keys=keys)
    posterior.posterior_estimator.to(device)
    if device.type == "cuda":
        posterior.prior = ctx["cfg"].param_module.build_prior(device=str(device))
    checks = []
    check(checks, "checkpoint_matches_source", posterior.weights_sha256 == manifest["checkpoint_sha256"],
          f"loaded weights sha256 {posterior.weights_sha256}; source product recorded {manifest['checkpoint_sha256']}")
    if posterior.weights_sha256 != manifest["checkpoint_sha256"]:
        raise SystemExit("the estimator on disk is not the one the source Experiment ran with; stopping before sampling.")
    eval_cfg = PARAMETERS.inference.evaluation
    out_dir = provenance.reserve_run_folder(out_dir); fig_dir = out_dir / "figures"; fig_dir.mkdir()

    # ---- sampling ------------------------------------------------------------------------------------------
    clouds, fq, fsgm, fidx, cell_of, chunk_of, elapsed, n_out = [], [], [], [], [], [], [], []
    src_cell, src_chunk = arrays["cell"].astype(int), arrays["chunk"].astype(int)
    src_rows = []
    lo, hi = ctx["prior_low"], ctx["prior_high"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for c in cells:
        chunks = read_cell_chunks(tifs[c], n_frames, step)
        expected_windows = (int(manifest["window_geometry"]["span_frames"]) - n_frames) // step + 1
        if len(chunks) != expected_windows:
            raise SystemExit(f"cell {c}: {len(chunks)} windows; the source geometry gives {expected_windows}")
        for w, chunk in enumerate(chunks):
            t0 = time.time()
            summary, cloud, sgm_vec = posterior_summary(posterior, chunk, device, torch.device("cpu"), n_draws,
                                                        eval_cfg.theta_prex_batch_size, pool_mode=pool_mode, quantiles=QUANTILE_LEVELS,
                                                        return_samples=True, return_sgm=True, sgm_scale=sgm_scale)
            sgm_again, idx = sample_geometric_median(cloud, scale=sgm_scale)
            if not np.array_equal(sgm_again, sgm_vec):
                raise SystemExit(f"cell {c} chunk {w}: SGM recomputation disagrees with posterior_summary")
            dt = time.time() - t0
            rows = np.where((src_cell == c) & (src_chunk == w) & (arrays["kind_index"] == 0))[0]
            if rows.size != 1:
                raise SystemExit(f"cell {c} chunk {w}: {rows.size} source rows (expected 1)")
            outside = int(np.any((cloud < lo) | (cloud > hi), axis=1).sum())
            clouds.append(cloud.astype(np.float32)); fq.append(summary); fsgm.append(sgm_vec); fidx.append(idx)
            cell_of.append(c); chunk_of.append(w); elapsed.append(dt); n_out.append(outside); src_rows.append(int(rows[0]))
            print(f"  {ctx['condition']} cell {c} chunk {w}: {n_draws} draws in {dt:.1f} s; SGM row {idx}; {outside} draws outside the prior box", flush=True)
    peak_gpu = float(torch.cuda.max_memory_allocated(device)) / 2**20 if device.type == "cuda" else None
    W = len(clouds)
    cloud_arr = np.stack(clouds); fq = np.stack(fq); fsgm = np.stack(fsgm); fidx = np.asarray(fidx, np.int64)
    cell_arr = np.asarray(cell_of, np.int64); chunk_arr = np.asarray(chunk_of, np.int64); src_rows = np.asarray(src_rows, np.int64)
    code = provenance.finalize_code_provenance(startup)
    collection_manifest = {
        "stage": COLLECTION_STAGE, "collection_note": "retained joint draws of selected windows; NOT an Experiment-stage product (no fresh MAP)",
        "parameter_keys": keys, "coordinate_transform": str(manifest["coordinate_transform"]), "pool_mode": pool_mode, "draw_label": draw_label(pool_mode),
        "n_summary_draws": n_draws, "quantile_levels": list(QUANTILE_LEVELS), "sgm_scale": sgm_scale.tolist(), "sgm_coordinates": "estimator",
        "checkpoint_sha256": posterior.weights_sha256, "estimator_path": str(ctx["estimator"]), "estimator_sha256": sha256_of(ctx["estimator"]),
        "window_geometry": dict(manifest["window_geometry"]), "condition_labels": [ctx["condition"]], "observation_id_fields": ["kind_index", "cell", "chunk"],
        "n_observations": W, "cells": cells, "collection_label": label or None, "seed_policy": schema.seed_policy(args.seed), "input_conversion": "read_cell_chunks(domain='stored'): 16-bit raw frames to 8-bit, normalize_video in the sampler",
        "source_product": {"path": str(ctx["exp_npz"]), "sha256": src_sha_before, "checkpoint_sha256": manifest["checkpoint_sha256"], "execution": manifest.get("execution"),
                           "n_summary_draws": manifest["n_summary_draws"], "written_at": manifest.get("written_at"),
                           "joined_fields": ["source_map_estimate", "source_posterior_quantiles", "source_posterior_sgm", "source_scores"],
                           "join_rule": "by (kind_index, cell, chunk); source_row_index records the matched row"},
        "reference": {"nuisance_tag": sel["tag"], "artifact": sel["artifact"], "artifact_sha256": sel["artifact_sha256"], "physical": dict(zip(keys, map(float, sel["physical"])))},
        "execution": {"host": platform.node(), "device": str(device), "backend": topo.backend, "torch": torch.__version__, "numpy": np.__version__,
                      "python": sys.version.split()[0], "package_version": PACKAGE_VERSION, "seconds_per_window": [round(e, 2) for e in elapsed],
                      "peak_gpu_memory_mib": peak_gpu, "argv": sys.argv},
        "code": code, "written_at": schema._now(),
    }
    np.savez_compressed(npz_path, posterior_samples_cloud=cloud_arr, kind_index=np.zeros(W, np.int64), cell=cell_arr, chunk=chunk_arr,
                        kinds=np.asarray([ctx["condition"]]), fresh_quantiles=fq, fresh_sgm=fsgm, fresh_sgm_index=fidx,
                        draws_outside_prior=np.asarray(n_out, np.int64), quantile_levels=np.asarray(QUANTILE_LEVELS),
                        prior_low=lo, prior_high=hi, sgm_scale=sgm_scale, parameter_keys=np.asarray(keys),
                        source_map_estimate=arrays["map_estimate"][src_rows], source_posterior_quantiles=arrays["posterior_quantiles"][src_rows],
                        source_posterior_sgm=arrays["posterior_sgm"][src_rows], source_scores=arrays["scores"][src_rows], source_row_index=src_rows,
                        collection_manifest_json=np.asarray(json.dumps(collection_manifest, sort_keys=True, default=str)))
    print(f"wrote {npz_path} ({npz_path.stat().st_size / 2**20:.1f} MiB)")

    # ---- validation from the saved file ------------------------------------------------------------------------
    with np.load(npz_path, allow_pickle=False) as saved:
        S = {k: saved[k] for k in saved.files}
    cl = S["posterior_samples_cloud"]
    check(checks, "cloud_shape", cl.shape == (len(cells) * 10, n_draws, D) and cl.dtype == np.float32, f"{cl.shape} {cl.dtype}; expected ({len(cells) * 10}, {n_draws}, {D}) float32")
    check(checks, "cloud_finite", bool(np.all(np.isfinite(cl))), "every draw finite")
    ids = [(int(a), int(b), int(c_)) for a, b, c_ in zip(S["kind_index"], S["cell"], S["chunk"])]
    check(checks, "window_identities", len(set(ids)) == len(ids) and all(sorted(S["chunk"][S["cell"] == c].tolist()) == list(range(10)) for c in cells)
          and set(S["cell"].tolist()) == set(cells), f"{len(ids)} windows; each cell carries chunks 0..9 once; cells {sorted(set(S['cell'].tolist()))}")
    check(checks, "parameter_order_and_transform", [str(k) for k in S["parameter_keys"]] == keys and json.loads(str(S["collection_manifest_json"]))["coordinate_transform"] == manifest["coordinate_transform"],
          f"{keys}; {manifest['coordinate_transform']}")
    recomputed = np.stack([np.quantile(cl[i].astype(np.float64), list(QUANTILE_LEVELS), axis=0).T for i in range(len(cl))])
    recomputed32 = np.stack([np.quantile(cl[i], list(QUANTILE_LEVELS), axis=0).T for i in range(len(cl))])
    dq = float(np.max(np.abs(recomputed - S["fresh_quantiles"])))
    check(checks, "quantiles_recompute", np.array_equal(recomputed32, S["fresh_quantiles"].astype(np.float32)) or dq < 1e-5,
          f"quantiles recomputed from the saved cloud agree with the saved ones (max |diff| {dq:.1e} in log10; the saved cloud is float32, the summary was taken on the same float32 draws)")
    sgm_ok = all(np.array_equal(cl[i, S["fresh_sgm_index"][i]].astype(np.float64), S["fresh_sgm"][i].astype(np.float32).astype(np.float64)) for i in range(len(cl)))
    sgm_re = all(sample_geometric_median(cl[i], scale=S["sgm_scale"])[1] == S["fresh_sgm_index"][i] for i in range(len(cl)))
    check(checks, "sgm_is_a_cloud_row", sgm_ok and sgm_re, "the saved SGM equals the cloud row at its index, and recomputing the medoid with the prior-width scale returns the same row")
    q_phys = physical_quantiles(S["fresh_quantiles"].astype(float), table)
    check(checks, "physical_endpoints_reproduce", bool(np.allclose(q_phys, 10 ** S["fresh_quantiles"].astype(float))),
          "physical endpoints = 10 ** stored log10 quantiles; widths are computed inside each window from these")
    join_ok = all(src_cell[r] == c_ and src_chunk[r] == w for r, c_, w in zip(S["source_row_index"], S["cell"], S["chunk"]))
    check(checks, "source_join_by_identity", join_ok and np.array_equal(S["source_map_estimate"], arrays["map_estimate"][S["source_row_index"]]),
          "source MAP/quantiles/SGM joined by (kind_index, cell, chunk); row indices recorded")
    check(checks, "checkpoint_hash_recorded", json.loads(str(S["collection_manifest_json"]))["checkpoint_sha256"] == manifest["checkpoint_sha256"], manifest["checkpoint_sha256"])
    src_sha_after = sha256_of(ctx["exp_npz"])
    check(checks, "source_product_unchanged", src_sha_after == src_sha_before, f"source npz sha256 {src_sha_after}")
    check(checks, "reference_artifact_unchanged", sha256_of(sel["artifact"]) == sel["artifact_sha256"], f"{Path(sel['artifact']).name} sha256 {sel['artifact_sha256']}")
    check(checks, "code_unchanged_during_run", not code["changed_during_run"], f"implementation hash {code['implementation']['sha256'][:16]}")

    # ---- summaries, correlations, joint depth -----------------------------------------------------------------------
    f_phys = q_phys; f_med = f_phys[:, :, 2]; f_sgm_p = to_physical(fsgm, table)
    s_q = physical_quantiles(S["source_posterior_quantiles"].astype(float), table); s_map_p = to_physical(S["source_map_estimate"], table); s_sgm_p = to_physical(S["source_posterior_sgm"], table)
    w90 = f_phys[:, :, 4] - f_phys[:, :, 0]; w50 = f_phys[:, :, 3] - f_phys[:, :, 1]
    sum_rows, corr_rows, depth_rows, corr_store = [], [], [], {}
    for i in range(W):
        c, w = int(cell_arr[i]), int(chunk_arr[i]); start, stop = w * step, w * step + n_frames
        z = cl[i].astype(float)
        for j, k in enumerate(keys):
            sum_rows.append([ctx["condition"], c, w, start, stop, f"{start*frame_s:g}", f"{stop*frame_s:g}", k, UNITS[k], sel["physical"][j],
                             f_phys[i, j, 0], f_phys[i, j, 1], f_med[i, j], f_phys[i, j, 3], f_phys[i, j, 4], w50[i, j], w90[i, j], f_sgm_p[i, j],
                             s_map_p[i, j], s_q[i, j, 2], s_sgm_p[i, j], s_q[i, j, 0], s_q[i, j, 4], fq[i, j, 2] - S["source_posterior_quantiles"][i, j, 2],
                             fsgm[i, j] - S["source_posterior_sgm"][i, j], int(fq[i, j, 0] <= sel["log10"][j] <= fq[i, j, 4]),
                             int(fq[i, j, 0] <= S["source_map_estimate"][i, j] <= fq[i, j, 4]), int(np.mean((z[:, j] < lo[j]) | (z[:, j] > hi[j])) * 1e4) / 1e4])
        pear = np.corrcoef(z, rowvar=False)
        from scipy.stats import spearmanr
        spear = spearmanr(z).statistic
        pear_phys = np.corrcoef(to_physical(z, table), rowvar=False)
        corr_store[(c, w)] = {"pearson_estimator": pear, "spearman": spear, "pearson_physical": pear_phys}
        for a in range(D):
            for b in range(a + 1, D):
                corr_rows.append([ctx["condition"], c, w, keys[a], keys[b], pear[a, b], spear[a, b], pear_phys[a, b]])
        # joint depth: distance to the window's SGM in prior-scaled estimator coordinates, and the share of draws strictly
        # farther than the reference (the shared distance rank of experiment_aggregation.reference_distance_ranks, the same
        # definition Experiment's compatibility diagnostics use; the output contract of joint_depth.csv is unchanged)
        refs = (("SEL", sel["log10"]), ("source MAP", S["source_map_estimate"][i]), ("source SGM", S["source_posterior_sgm"][i]), ("fresh median vector", fq[i, :, 2]))
        ranks = agg.reference_distance_ranks(z, fsgm[i], np.asarray([v for _, v in refs], float), sgm_scale)
        for (name, vec), rank in zip(refs, ranks):
            d_ref = float(np.sqrt((((vec - fsgm[i]) / sgm_scale) ** 2).sum()))
            depth_rows.append([ctx["condition"], c, w, name, d_ref, float(rank), int(np.all((fq[i, :, 0] <= vec) & (vec <= fq[i, :, 4])))])
    write_csv(out_dir / "window_summaries.csv",
              ["condition", "cell", "chunk", "start_frame", "stop_frame_exclusive", "start_seconds", "stop_seconds", "parameter", "unit", "sel",
               "fresh_q05", "fresh_q25", "fresh_median", "fresh_q75", "fresh_q95", "fresh_width50", "fresh_width90", "fresh_sgm",
               "source_map", "source_median", "source_sgm", "source_q05", "source_q95", "median_fresh_minus_source_dex", "sgm_fresh_minus_source_dex",
               "sel_inside_fresh90", "source_map_inside_fresh90", "fraction_draws_outside_prior"], sum_rows)
    write_csv(out_dir / "correlations.csv", ["condition", "cell", "chunk", "parameter_a", "parameter_b", "pearson_estimator_log10", "spearman", "pearson_physical"], corr_rows)
    write_csv(out_dir / "joint_depth.csv", ["condition", "cell", "chunk", "reference", "prior_scaled_distance_to_fresh_sgm", "share_of_draws_farther_than_reference", "inside_every_marginal_90"], depth_rows)

    # ---- figures ----------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(args.scatter_seed)
    labels = [e["LABEL"] for e in table]; figures = []
    kidx = {k: j for j, k in enumerate(keys)}
    for c in cells:
        for w in JOINT_CHUNKS:
            i = int(np.where((cell_arr == c) & (chunk_arr == w))[0][0]); z = cl[i].astype(float)
            sub = rng.choice(n_draws, size=min(args.scatter_subset, n_draws), replace=False)
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            for ax, (ka, kb) in zip(axes.ravel(), PAIRS):
                a, b = kidx[ka], kidx[kb]
                ax.scatter(10 ** z[sub, a], 10 ** z[sub, b], s=4, alpha=0.25, color="tab:blue", label=f"{len(sub)} of {n_draws} draws (display subset)")
                H, xe, ye = np.histogram2d(z[:, a], z[:, b], bins=40)
                ax.contour(10 ** (0.5 * (xe[1:] + xe[:-1])), 10 ** (0.5 * (ye[1:] + ye[:-1])), H.T, levels=5, colors="k", linewidths=0.6, alpha=0.6)
                ax.scatter([f_sgm_p[i, a]], [f_sgm_p[i, b]], marker="s", s=70, color="tab:green", edgecolors="k", zorder=5, label="fresh SGM (a draw)")
                ax.scatter([f_med[i, a]], [f_med[i, b]], marker="o", s=70, color="tab:blue", edgecolors="k", zorder=5, label="fresh median vector (composite)")
                ax.scatter([s_map_p[i, a]], [s_map_p[i, b]], marker="P", s=80, color="tab:red", edgecolors="k", zorder=5, label="source MAP (reference)")
                ax.scatter([sel["physical"][a]], [sel["physical"][b]], marker="*", s=160, color="k", zorder=6, label=sel["tag"])
                for e in (10 ** lo[a], 10 ** hi[a]): ax.axvline(e, color="gray", ls=":", lw=0.8)
                for e in (10 ** lo[b], 10 ** hi[b]): ax.axhline(e, color="gray", ls=":", lw=0.8)
                ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(f"{ka} ({UNITS[ka]})", fontsize=8); ax.set_ylabel(f"{kb} ({UNITS[kb]})", fontsize=8)
                st = corr_store[(c, w)]
                ax.set_title(f"Pearson(log10) {st['pearson_estimator'][a, b]:+.2f}, Spearman {st['spearman'][a, b]:+.2f}, Pearson(physical) {st['pearson_physical'][a, b]:+.2f}", fontsize=9)
            axes[0, 0].legend(fontsize=7, loc="best")
            fig.suptitle(f"{ctx['condition']} cell {c} chunk {w} ({w*step*frame_s:g}-{(w*step+n_frames)*frame_s:g} s): joint structure of ONE window's {n_draws} {draw_label(pool_mode)}s "
                         f"(physical axes, log scales; contours: 2D histogram of the full cloud in log10; dotted: prior box; scatter is a fixed-seed subset)", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.95)); p = fig_dir / f"joint_cell_{c}_chunk_{w}.png"; fig.savefig(p, dpi=120); plt.close(fig); figures.append(p.name)
        # marginals over the ten windows
        fig, axes = plt.subplots(2, 3, figsize=(15, 8)); cmap = matplotlib.colormaps["viridis"]
        for j, ax in enumerate(axes.ravel()):
            for w in range(10):
                i = int(np.where((cell_arr == c) & (chunk_arr == w))[0][0])
                ax.hist(cl[i, :, j], bins=60, histtype="step", density=True, color=cmap(w / 9), lw=1, label=f"chunk {w}" if j == 0 else None)
            ax.axvline(sel["log10"][j], color="k", ls="--", lw=1); ax.axvspan(lo[j], hi[j], color="gray", alpha=0.08)
            ax.set_title(f"{keys[j]}  {labels[j]}", fontsize=10); ax.set_xlabel("log10 physical value (estimator coordinate)"); ax.set_ylabel("density")
            if j == 0: ax.legend(fontsize=6, ncol=2)
        fig.suptitle(f"{ctx['condition']} cell {c}: marginal densities of the fresh draws, one curve per window (dark early to bright late); dashed: {sel['tag']}; gray: prior box", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); p = fig_dir / f"marginals_cell_{c}.png"; fig.savefig(p, dpi=130); plt.close(fig); figures.append(p.name)
        # correlation heat map: windows x pairs
        pairs = [(a, b) for a in range(D) for b in range(a + 1, D)]
        M = np.array([[corr_store[(c, w)]["pearson_estimator"][a, b] for (a, b) in pairs] for w in range(10)])
        Ms = np.array([[corr_store[(c, w)]["spearman"][a, b] for (a, b) in pairs] for w in range(10)])
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, mat, name in ((axes[0], M, "Pearson (estimator coordinates, log10)"), (axes[1], Ms, "Spearman")):
            im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(pairs))); ax.set_xticklabels([f"{keys[a]} x {keys[b]}" for a, b in pairs], fontsize=7, rotation=60, ha="right"); ax.set_yticks(range(10)); ax.set_ylabel("chunk")
            for (y, x), v in np.ndenumerate(mat): ax.text(x, y, f"{v:+.2f}", ha="center", va="center", fontsize=6)
            ax.set_title(f"{ctx['condition']} cell {c}: within-window {name} correlations of the fresh draws", fontsize=9)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8); p = fig_dir / f"correlations_cell_{c}.png"; fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); figures.append(p.name)

    # ---- report ------------------------------------------------------------------------------------------------------
    rep = [f"# Window-posterior draws ({ctx['condition']}, {ctx['product_label']})", "",
           f"Fresh joint draws for cells {cells}, ten windows each, {n_draws} complete draws per window ({draw_label(pool_mode)}s, pool mode {pool_mode} as in the source), "
           f"with the adopted estimator (weights sha256 {posterior.weights_sha256}) on {device} ({topo.backend}), seed {args.seed}. Source references joined from "
           f"`{ctx['exp_npz'].name}` (sha256 {src_sha_before}). Reference {sel['tag']}: `{Path(sel['artifact']).name}`. Stored coordinates are the estimator's (log10); physical values are "
           f"ten to the power of the stored values. The fresh summaries are computed from the retained cloud; the source MAP is a reference from the source run, not a result of this cloud. "
           f"Package {PACKAGE_VERSION}; torch {torch.__version__}." + (f" Collection label `{label}` (output-only; same estimator and source as the unlabeled collection)." if label else ""), "",
           "## Checks", "", md_table(["check", "result", "detail"], checks), "",
           "## Runtime and storage", "", f"- seconds per window: {', '.join(f'{e:.1f}' for e in elapsed)} (total sampling {sum(elapsed):.0f} s; wall {time.time() - t_start:.0f} s)",
           f"- peak GPU memory {peak_gpu:.0f} MiB" if peak_gpu is not None else "- CPU run", f"- `{npz_path.name}`: {npz_path.stat().st_size / 2**20:.1f} MiB compressed "
           f"({cl.nbytes / 2**20:.1f} MiB of float32 draws)", ""]
    for c in cells:
        rep += [f"## Cell {c}: fresh summaries per window (physical units) and the source references", ""]
        for j, k in enumerate(keys):
            rows = []
            for w in range(10):
                i = int(np.where((cell_arr == c) & (chunk_arr == w))[0][0])
                rows.append([w, fmt(f_med[i, j]), fmt(f_sgm_p[i, j]), f"[{fmt(f_phys[i, j, 0])}, {fmt(f_phys[i, j, 4])}]", fmt(w90[i, j]), fmt(s_map_p[i, j]), fmt(s_q[i, j, 2]),
                             f"{fq[i, j, 2] - S['source_posterior_quantiles'][i, j, 2]:+.3f}", "yes" if fq[i, j, 0] <= sel["log10"][j] <= fq[i, j, 4] else "no",
                             f"{np.mean((cl[i, :, j] < lo[j]) | (cl[i, :, j] > hi[j])) * 100:.1f} %"])
            rep += [f"**{k}** ({UNITS[k]}; {sel['tag']} {fmt(sel['physical'][j])}; prior [{fmt(10**lo[j])}, {fmt(10**hi[j])}])", "",
                    md_table(["chunk", "fresh median", "fresh SGM", "fresh 90 %", "fresh width90", "source MAP", "source median", "median fresh-source (dex)", "SEL inside fresh 90 %", "draws outside prior"], rows), ""]
        rep += [f"### Cell {c}: joint depth of the reference vectors", "",
                md_table(["chunk", "reference", "prior-scaled distance to the fresh SGM", "share of draws farther", "inside every marginal 90 %"],
                         [[r[2], r[3], f"{r[4]:.3f}", f"{r[5]:.3f}", "yes" if r[6] else "no"] for r in depth_rows if r[1] == c]), "",
                "The share of draws farther from the window's SGM than the reference is a distance rank relative to the SGM in prior-scaled estimator coordinates: near 1 means "
                "most draws lie farther from the SGM than the reference does; near 0 means almost every draw is closer to the SGM than the reference is. It is a distance rank, "
                "not a posterior-density or joint-credible-region statement. The same definition (strict comparison, prior-width scale; "
                "`experiment_aggregation.reference_distance_ranks`) is the rank of the Experiment report's posterior-compatibility diagnostics; this companion "
                "places fixed reference vectors in one window's cloud, the official report compares the SGM representatives of groups with one another.", "",
                f"### Cell {c}: within-window correlations (Pearson, estimator coordinates), chunks {JOINT_CHUNKS}", ""]
        for w in JOINT_CHUNKS:
            P_ = corr_store[(c, w)]["pearson_estimator"]; Sp = corr_store[(c, w)]["spearman"]
            rep += [f"chunk {w}:", "", md_table(["", *keys], [[keys[a]] + [f"{P_[a, b]:+.2f} / {Sp[a, b]:+.2f}" for b in range(D)] for a in range(D)]), "(Pearson / Spearman)", ""]
        rep += [f"Figures: " + ", ".join(f"`figures/joint_cell_{c}_chunk_{w}.png`" for w in JOINT_CHUNKS) + f", `figures/marginals_cell_{c}.png`, `figures/correlations_cell_{c}.png`.", ""]
    rep += ["## Files", "", f"- `{npz_path.name}`: posterior_samples_cloud (windows x draws x parameters, float32, log10), kind_index/cell/chunk, fresh_quantiles, fresh_sgm, fresh_sgm_index, "
            "draws_outside_prior, source_map_estimate / source_posterior_quantiles / source_posterior_sgm / source_scores / source_row_index (joined references), prior box, sgm_scale, collection_manifest_json.",
            "- `window_summaries.csv`, `correlations.csv`, `joint_depth.csv`, `figures/`, `provenance.json`.", "",
            "## Reading rules", "", "- Every row of the cloud is one complete joint draw; columns are never resampled or rank-matched independently.",
            "- Outside-prior draws are retained and counted (unrestricted flow draws, as in the source).",
            "- Correlations are reported in estimator coordinates (log10) with Spearman beside Pearson; the physical-coordinate Pearson is in `correlations.csv`.",
            "- Fresh and source summaries come from different Monte Carlo samples; their differences (dex) measure sampling reproducibility, not a change of the posterior."]
    (out_dir / "report.md").write_text("\n".join(rep) + "\n")
    prov = provenance.analysis_run_record(startup, argv=sys.argv, files=provenance.IMPLEMENTATION_FILES, code=code,
                                          extra={"purpose": "retained joint posterior draws of selected experimental windows (detector)", "collection_label": label or None, "collection_manifest": collection_manifest,
                                                 "output_npz": str(npz_path), "output_npz_sha256": sha256_of(npz_path),
                                                 "checks": [{"name": n, "result": r, "detail": d} for n, r, d in checks], "wall_seconds": time.time() - t_start})
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    print("\n".join(f"  {n:36s} {r}  {d[:110]}" for n, r, d in checks))
    print(f"wrote {out_dir} ({len(figures)} figures) in {time.time() - t_start:.0f} s")
    return 0 if all(r == "PASS" for _, r, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
