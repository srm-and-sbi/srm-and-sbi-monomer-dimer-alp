"""Cell-agnostic equal-window aggregation comparison of one retained window-posterior draw collection.

Reads ONE ``*_Window_Posterior_Draws[_<LABEL>].npz`` collection (condition, cell, window identities, parameter
order, prior box, SGM scale, reference vector and checkpoint all from its own metadata) and repeats the
calculation of ``Special_Analyses/FAB26_AGGREGATION_COMPARISON_2026-11-03/reproduce.py`` (and the InlB-11 twin):

- pool the ten retained window clouds (10,000 complete joint draws each): 100,000 vectors, exactly 10 % mixture
  weight per window; identities (window, draw index) preserved, no filtering, no inference;
- primary metric: log10 estimator coordinates divided by the prior widths (the per-window SGM convention);
  the pooled representative is the member returned by the repository's collection kernel
  ``sample_geometric_median.sample_geometric_median`` (``weiszfeld_snap`` above 20,000 members: the member
  nearest the continuous geometric median, an approximation to the exact sample medoid), an actual saved draw;
- pooled marginal quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the full pooled log10 sample, endpoints transformed to
  physical units: actual mixture quantiles;
- the physical-coordinate normalization (physical values divided by physical prior ranges) as the same SEPARATE
  sensitivity check the completed reviews recorded; neither convention is adopted as a policy;
- relative difference of the pooled representative to the reference vector (percent and dex), per parameter.

Writes into a NEW ``--out-dir``: comparison.csv, window_sgms.csv, result.json, comparison.png, README.md, checks.json.
CPU only; reads the collection, changes nothing else. Usage:

    python aggregation_comparison.py --source <collection.npz> --out-dir <new folder> [--repo <repository root>]
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
UNITS = {"mu_r": "px (sqrt(2) times Gaussian sigma)", "sigma_r": "dimensionless ln-spread",
         "mu_pc": "photons per active dye per frame", "sigma_pc": "dimensionless ln-spread",
         "prob_photo_bleach": "probability per 100 frames", "lambda_rate": "per second"}
SHORT = {"mu_r": "mu_r (px)", "sigma_r": "sigma_r", "mu_pc": "mu_pc (photons/dye/frame)", "sigma_pc": "sigma_pc",
         "prob_photo_bleach": "bleaching probability / 100 frames", "lambda_rate": "lambda (1/s)"}
DISPLAY = {"FAB": "Fab", "INLB": "InlB"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="the *_Window_Posterior_Draws[_<LABEL>].npz collection")
    ap.add_argument("--out-dir", type=Path, required=True, help="a NEW folder for the outputs")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4],
                    help="repository root holding srm_and_sbi_monomer_dimer_alp/sample_geometric_median.py")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise SystemExit(f"refusing to write into an existing folder: {out}")
    source = args.source.resolve()
    kernel_path = args.repo / "srm_and_sbi_monomer_dimer_alp" / "sample_geometric_median.py"
    if not kernel_path.exists():
        raise SystemExit(f"collection kernel not found: {kernel_path}")

    with np.load(source, allow_pickle=False) as a:
        cloud = a["posterior_samples_cloud"].astype(np.float64)
        chunks = a["chunk"].copy(); cells = a["cell"].copy(); kinds = [str(k) for k in a["kinds"]]
        keys = [str(k) for k in a["parameter_keys"]]
        scale = a["sgm_scale"].astype(np.float64)
        window_sgms = a["fresh_sgm"].copy(); window_sgm_index = a["fresh_sgm_index"].copy()
        window_quantiles = a["fresh_quantiles"].copy(); levels = a["quantile_levels"].astype(np.float64)
        low, high = a["prior_low"].astype(np.float64), a["prior_high"].astype(np.float64)
        manifest = json.loads(a["collection_manifest_json"].item())
    condition = kinds[0]
    cell = int(np.unique(cells)[0])
    checks = {}

    def check(name, ok, detail):
        checks[name] = {"result": "PASS" if ok else "FAIL", "detail": detail}

    W, S, D = cloud.shape
    check("one_cell_one_condition", len(kinds) == 1 and np.unique(cells).size == 1, f"condition {condition}, cell {cell}")
    check("cloud_shape", cloud.shape == (10, 10000, 6), f"{cloud.shape}")
    check("window_identities", np.array_equal(np.sort(chunks), np.arange(10)), f"chunks {chunks.tolist()}")
    check("cloud_finite", bool(np.isfinite(cloud).all()), "every draw finite")
    check("pool_mode_unrestricted", manifest["pool_mode"] == "unrestricted", manifest["pool_mode"])
    check("parameter_order", keys == ["mu_r", "sigma_r", "mu_pc", "sigma_pc", "prob_photo_bleach", "lambda_rate"], str(keys))
    check("scale_equals_prior_widths", np.allclose(scale, high - low) and np.allclose(scale, manifest["sgm_scale"]),
          f"sgm_scale {scale.tolist()} = prior widths {(high - low).tolist()}")
    check("quantile_levels", np.allclose(levels, LEVELS), levels.tolist())
    flat = cloud.reshape(-1, D)
    window_of = np.repeat(np.arange(W), S)            # the window (row of the cloud) each pooled vector came from
    draw_of = np.tile(np.arange(S), W)
    counts = np.bincount(window_of, minlength=W)
    check("equal_window_weights", bool(np.all(counts == S)) and flat.shape[0] == W * S,
          f"{W} windows x {S} draws = {flat.shape[0]} vectors; weight per window {S / flat.shape[0]:.3f}")
    q = np.quantile(flat, LEVELS, axis=0).T                       # (D, 5) log10
    rng = np.random.default_rng(0)
    perm = rng.permutation(flat.shape[0])
    q_again = np.quantile(flat[perm], LEVELS, axis=0).T
    check("pooled_quantiles_reproducible", np.array_equal(q, q_again), "identical after a row permutation (order-free mixture quantiles)")
    physical_q = np.power(10.0, q)
    sel = np.array([manifest["reference"]["physical"][k] for k in keys], dtype=np.float64)
    ref_tag = manifest["reference"]["nuisance_tag"]

    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("review_sgm_kernel", kernel_path)
    kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)
    selected, method = kernel.sample_geometric_median(flat, scale)
    source_window, draw_index = int(window_of[selected]), int(draw_of[selected])
    check("representative_is_a_complete_vector", np.array_equal(flat[selected], cloud[source_window, draw_index])
          and (source_window, draw_index) == divmod(int(selected), S),
          f"flat row {int(selected)} = cloud[{source_window}, {draw_index}] (chunk {int(chunks[source_window])})")
    pooled = np.power(10.0, flat[selected])
    summed = np.linalg.norm((flat - flat[selected]) / scale, axis=1).sum()
    selected_physical, method_physical = kernel.sample_geometric_median(np.power(10.0, flat), np.power(10.0, high) - np.power(10.0, low))
    opening = int(np.where(chunks == 0)[0][0])
    check("opening_window_sgm_is_a_cloud_row", np.allclose(window_sgms[opening], cloud[opening, window_sgm_index[opening]]), f"row {int(window_sgm_index[opening])} of chunk 0")

    rows = []
    for j, key in enumerate(keys):
        rows.append(dict(parameter=key, unit=UNITS[key], sel=sel[j], window_1_sgm=10.0 ** window_sgms[opening, j],
                         pooled_representative=pooled[j], pooled_q05=physical_q[j, 0], pooled_q25=physical_q[j, 1],
                         pooled_median=physical_q[j, 2], pooled_q75=physical_q[j, 3], pooled_q95=physical_q[j, 4],
                         pooled_width90=physical_q[j, 4] - physical_q[j, 0], pooled_width50=physical_q[j, 3] - physical_q[j, 1],
                         rel_diff_pooled_vs_sel_percent=(pooled[j] / sel[j] - 1.0) * 100.0,
                         diff_pooled_vs_sel_dex=np.log10(pooled[j] / sel[j]),
                         rel_diff_opening_sgm_vs_sel_percent=(10.0 ** window_sgms[opening, j] / sel[j] - 1.0) * 100.0,
                         sel_inside_pooled_90=bool(physical_q[j, 0] <= sel[j] <= physical_q[j, 4]),
                         sel_inside_pooled_50=bool(physical_q[j, 1] <= sel[j] <= physical_q[j, 3])))
    out.mkdir(parents=True)
    with (out / "comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out / "window_sgms.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cell", "chunk", "window_number", "start_s", "stop_s", "sgm_row_index"] + keys)
        for i in np.argsort(chunks):
            c = int(chunks[i]); w.writerow([cell, c, c + 1, 2 * c, 2 * (c + 1), int(window_sgm_index[i])] + np.power(10.0, window_sgms[i]).tolist())
    result = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  condition=condition, cell=cell, collection_label=manifest.get("collection_label"),
                  cloud_shape=list(cloud.shape), total_draws=int(flat.shape[0]), window_weights=[S / flat.shape[0]] * W,
                  parameter_keys=keys, pool_mode=manifest["pool_mode"], checkpoint_sha256=manifest["checkpoint_sha256"],
                  reference=dict(tag=ref_tag, physical=dict(zip(keys, sel.tolist())), artifact=manifest["reference"]["artifact"]),
                  metric="log10 estimator coordinates divided by prior widths", scale=scale.tolist(), method=method,
                  method_note="Existing large-pool Weiszfeld geometric median, snapped to an actual draw; not an exact sample medoid.",
                  selected_flat_index=int(selected), selected_source_chunk=int(chunks[source_window]),
                  selected_source_window_number=int(chunks[source_window]) + 1, selected_source_draw_index=draw_index,
                  selected_summed_normalized_distance=float(summed),
                  selected_log10=flat[selected].tolist(), selected_physical=pooled.tolist(),
                  opening_window_sgm_physical=np.power(10.0, window_sgms[opening]).tolist(),
                  pooled_quantile_levels=LEVELS, pooled_quantiles_log10=q.tolist(), pooled_quantiles_physical=physical_q.tolist(),
                  alternate_metric=dict(metric="physical coordinates divided by physical prior ranges", method=method_physical,
                                        selected_flat_index=int(selected_physical),
                                        selected_source_chunk=int(chunks[int(window_of[selected_physical])]),
                                        selected_source_draw_index=int(draw_of[selected_physical]),
                                        selected_physical=np.power(10.0, flat[selected_physical]).tolist()),
                  checks=checks, source_manifest=manifest)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pool_review_mpl"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(chunks); times = chunks[order] * 2
    p_window_q = np.power(10.0, window_quantiles[order]); p_sgm = np.power(10.0, window_sgms[order])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        ax.fill_between(times, p_window_q[:, j, 0], p_window_q[:, j, 4], color="#4878a8", alpha=.12, label="Within-window 90% interval")
        ax.fill_between(times, p_window_q[:, j, 1], p_window_q[:, j, 3], color="#4878a8", alpha=.24, label="Within-window 50% interval")
        ax.plot(times, p_sgm[:, j], "o-", color="#28659b", markersize=4, linewidth=1.2, label="Per-window SGM")
        ax.axhline(sel[j], color="#603696", linestyle="--", linewidth=1.5, label=ref_tag)
        ax.axhline(pooled[j], color="#d87916", linewidth=1.5, label="Pooled representative")
        ax.set_title(SHORT[keys[j]], fontsize=10); ax.set_xticks([0, 4, 8, 12, 16, 18]); ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Start of 2 s window (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, .045), frameon=False)
    fig.suptitle(f"{DISPLAY.get(condition, condition)} cell {cell}: per-window joint summaries, equal-window pooling, and {ref_tag}", fontsize=13)
    fig.text(.5, .015, f"{flat.shape[0]:,} saved joint draws; pooled representative uses the existing large-pool approximation in normalized log10 coordinates",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .94)); fig.savefig(out / "comparison.png", dpi=160); plt.close(fig)

    name = f"{DISPLAY.get(condition, condition)} cell {cell}"
    text = [f"# {name}: aggregation comparison", "",
            f"Exploratory CPU comparison of saved draws (`{source.name}`). {ref_tag} is the existing reference; no replacement vector or aggregation rule is adopted here.", "",
            "## Population and method", "",
            f"- Ten 2 s windows, {S:,} joint draws each: {flat.shape[0]:,} complete vectors. Every window receives {100 * S / flat.shape[0]:.0f}% weight.",
            "- The draws remain unrestricted and unfiltered. No inference is rerun.",
            "- Distances use log10 estimator coordinates normalized by the existing prior widths, matching the per-window definition.",
            f"- The existing collection kernel uses {method} above 20,000 vectors. The pooled representative is an actual saved draw, but an approximation to the exact sample medoid.",
            f"- The selected draw is from window {int(chunks[source_window]) + 1} (chunk {int(chunks[source_window])}), draw index {draw_index}. This origin identifies a pool member; it does not give that window extra mixture weight.",
            "- Quantiles are computed on the full pooled log10 sample and transformed to original units. These are actual mixture quantiles, not medians of per-window interval bounds.",
            "- A physical-coordinate normalization sensitivity result is recorded separately in result.json; it is not substituted for the primary comparison.",
            "- The relative difference is (pooled representative / reference - 1) in percent, with the same difference in dex beside it.", "",
            "## Numerical comparison", "",
            f"| Parameter | {ref_tag} | Window 1 SGM | Pooled representative | Pooled median | Pooled 50% interval | Pooled 90% interval | Pooled rep. vs {ref_tag} |",
            "|---|---:|---:|---:|---:|---|---|---:|"]
    for r in rows:
        text.append("| {} | {:.5g} | {:.5g} | {:.5g} | {:.5g} | {:.5g}–{:.5g} | {:.5g}–{:.5g} | {:+.1f} % ({:+.3f} dex) |".format(
            r["parameter"], r["sel"], r["window_1_sgm"], r["pooled_representative"], r["pooled_median"], r["pooled_q25"], r["pooled_q75"],
            r["pooled_q05"], r["pooled_q95"], r["rel_diff_pooled_vs_sel_percent"], r["diff_pooled_vs_sel_dex"]))
    text += ["", "![Comparison](comparison.png)", "", "## Checks", "", "| check | result | detail |", "|---|---|---|"]
    text += [f"| {k} | {v['result']} | {v['detail']} |" for k, v in checks.items()]
    text += ["", "## Reading rules", "",
             "- The pooled bleaching value is a descriptive statistic of the learned cloud. It does not replace the separately documented 0.05 reference.",
             "- The pooled representative is selected in the full six-dimensional cloud; pairwise correlations are diagnostics, and no separate sigma_pc-lambda relationship is imposed.",
             "- The opening window contributes 10% and the nine later windows 90%: the descriptive weighting of this comparison, not a claim that later windows are more accurate.", "",
             "## Files", "",
             "- comparison.csv: all six parameters, the reference, the opening-window SGM, the pooled representative, pooled marginal medians, 50% and 90% intervals, relative differences, in original units.",
             "- window_sgms.csv: the complete per-window SGM vectors and their cloud row indices.",
             "- result.json: source checksum, coordinate metric, method, selected draw identity, alternate-metric sensitivity, checks and the source manifest.",
             "- checks.json: the validation results.",
             "- Produced by `aggregation_comparison.py` (cell-agnostic; `Script_Bank/Analysis/Evidence/Window_Posterior_Additional_Cells_2026-11-03/`); use a fresh --out-dir to reproduce.", ""]
    (out / "README.md").write_text("\n".join(text))
    print(json.dumps({k: result[k] for k in ["condition", "cell", "total_draws", "method", "selected_source_window_number", "selected_source_draw_index", "selected_physical"]}, indent=2))
    print("\n".join(f"  {k:36s} {v['result']}  {v['detail'][:100]}" for k, v in checks.items()))
    print("Review outputs:", out)
    return 0 if all(v["result"] == "PASS" for v in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
