"""Matched comparison of several per-cell aggregation results (one row per cell x parameter).

Inputs are the ``result.json`` files written by ``aggregation_comparison.py`` (or the identical-format
``reproduce.py`` files of the Fab-26 / InlB-11 comparisons) plus each collection's ``joint_depth.csv`` and
``window_summaries.csv`` (from the draw collection folders). Nothing is recomputed from the draws: the script
tabulates the existing per-cell results side by side, adds two marginal-compatibility diagnostics per cell (how many
of the ten windows hold SEL, and how many hold the cell's own pooled vector, inside every marginal 90 % interval; and,
per parameter, the intersection of the ten windows' 90 % intervals, empty or not, with the membership of SEL and of
the pooled vector in it), the SEL distance rank relative to each window's SGM, and the per-window SGM range, and draws
one six-panel figure with the four cells side by side. The membership counts and intersections are marginal
compatibility diagnostics, not joint credible-region tests and not an independent validation.
Descriptive only: no aggregation rule or nuisance selection is adopted.

Usage:
    python four_cell_comparison.py --out-dir <new folder> \
        --cell "Fab 26=<result.json>:<joint_depth.csv>:<window_summaries.csv>" [--cell ...]
"""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

KEYS = ["mu_r", "sigma_r", "mu_pc", "sigma_pc", "prob_photo_bleach", "lambda_rate"]
SHORT = {"mu_r": "mu_r (px)", "sigma_r": "sigma_r", "mu_pc": "mu_pc (photons/dye/frame)", "sigma_pc": "sigma_pc",
         "prob_photo_bleach": "bleaching probability / 100 frames", "lambda_rate": "lambda (1/s)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cell", action="append", required=True, help='"<label>=<result.json>:<joint_depth.csv>:<window_summaries.csv>"')
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise SystemExit(f"refusing to write into an existing folder: {out}")
    cells = []
    for spec in args.cell:
        label, paths = spec.split("=", 1)
        r_path, d_path, w_path = (Path(p) for p in paths.split(":"))
        res = json.loads(r_path.read_text())
        depth = [r for r in csv.DictReader(d_path.open()) if r["reference"] == "SEL"]
        wins = list(csv.DictReader(w_path.open()))
        assert len(depth) == 10 and res["parameter_keys"] == KEYS and len(res["window_weights"]) == 10
        assert np.allclose(res["window_weights"], 0.1)
        sel = res.get("reference", {}).get("physical") or {k: v for k, v in zip(KEYS, [float(w["sel"]) for w in wins[:6]])}
        q = np.asarray(res["pooled_quantiles_log10"], float)                       # (6, 5)
        pq = np.power(10.0, q)
        opening = res.get("opening_window_sgm_physical")
        if opening is None:                                                        # the pilot reproduce.py result.json
            opening = [float(w["fresh_sgm"]) for w in wins if int(w["chunk"]) == 0]
        sgm_range = {k: (min(float(w["fresh_sgm"]) for w in wins if w["parameter"] == k),
                         max(float(w["fresh_sgm"]) for w in wins if w["parameter"] == k)) for k in KEYS}
        # per-window fresh 90 % intervals (physical units), chunk-ordered: (10, 6, 2)
        w90 = np.array([[[float(w["fresh_q05"]), float(w["fresh_q95"])] for k in KEYS for w in wins
                         if w["parameter"] == k and int(w["chunk"]) == c] for c in range(10)]).reshape(10, 6, 2)
        pooled_vec = np.asarray(res["selected_physical"], float); sel_vec = np.array([float(sel[k]) for k in KEYS])
        pooled_inside = int(np.all((w90[:, :, 0] <= pooled_vec) & (pooled_vec <= w90[:, :, 1]), axis=1).sum())
        sel_inside_recount = int(np.all((w90[:, :, 0] <= sel_vec) & (sel_vec <= w90[:, :, 1]), axis=1).sum())
        inter_lo, inter_hi = w90[:, :, 0].max(0), w90[:, :, 1].min(0)
        intersection = {k: dict(empty=bool(inter_lo[j] > inter_hi[j]), low=float(inter_lo[j]), high=float(inter_hi[j]),
                                pooled_inside=bool(inter_lo[j] <= pooled_vec[j] <= inter_hi[j]),
                                sel_inside=bool(inter_lo[j] <= sel_vec[j] <= inter_hi[j])) for j, k in enumerate(KEYS)}
        cells.append(dict(label=label, condition=res["condition"], cell=int(res["cell"]), sel=[float(sel[k]) for k in KEYS],
                          opening=[float(v) for v in opening], pooled=[float(v) for v in res["selected_physical"]], pq=pq,
                          method=res["method"], source_chunk=int(res["selected_source_chunk"]), draw=int(res["selected_source_draw_index"]),
                          source_sha256=res["source_sha256"], total=int(res["total_draws"]),
                          n_inside=sum(int(r["inside_every_marginal_90"]) for r in depth),
                          depth_min=min(float(r["share_of_draws_farther_than_reference"]) for r in depth),
                          depth_max=max(float(r["share_of_draws_farther_than_reference"]) for r in depth),
                          depth_by_chunk=[float(r["share_of_draws_farther_than_reference"]) for r in sorted(depth, key=lambda r: int(r["chunk"]))],
                          sgm_range=sgm_range, pooled_inside=pooled_inside, sel_inside_recount=sel_inside_recount, intersection=intersection))
        assert sel_inside_recount == cells[-1]["n_inside"], (label, sel_inside_recount, cells[-1]["n_inside"])
    out.mkdir(parents=True)
    rows = []
    for c in cells:
        for j, k in enumerate(KEYS):
            rows.append(dict(cell=c["label"], condition=c["condition"], cell_index=c["cell"], parameter=k, sel=c["sel"][j],
                             opening_window_sgm=c["opening"][j], pooled_representative=c["pooled"][j], pooled_median=c["pq"][j, 2],
                             pooled_q25=c["pq"][j, 1], pooled_q75=c["pq"][j, 3], pooled_q05=c["pq"][j, 0], pooled_q95=c["pq"][j, 4],
                             rel_diff_pooled_vs_sel_percent=(c["pooled"][j] / c["sel"][j] - 1) * 100, diff_pooled_vs_sel_dex=np.log10(c["pooled"][j] / c["sel"][j]),
                             rel_diff_opening_vs_sel_percent=(c["opening"][j] / c["sel"][j] - 1) * 100,
                             sel_inside_pooled_90=bool(c["pq"][j, 0] <= c["sel"][j] <= c["pq"][j, 4]),
                             window_sgm_min=c["sgm_range"][k][0], window_sgm_max=c["sgm_range"][k][1],
                             windows90_intersection_empty=c["intersection"][k]["empty"],
                             windows90_intersection_low=c["intersection"][k]["low"], windows90_intersection_high=c["intersection"][k]["high"],
                             pooled_in_intersection=c["intersection"][k]["pooled_inside"], sel_in_intersection=c["intersection"][k]["sel_inside"]))
    with (out / "four_cell_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out / "per_cell_depth.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cell", "condition", "cell_index", "windows_with_sel_inside_every_marginal_90", "windows_with_pooled_vector_inside_every_marginal_90",
                                        "parameters_with_nonempty_windows90_intersection", "sel_depth_min", "sel_depth_max"] + [f"sel_depth_chunk_{i}" for i in range(10)])
        for c in cells:
            w.writerow([c["label"], c["condition"], c["cell"], c["n_inside"], c["pooled_inside"], sum(not v["empty"] for v in c["intersection"].values()),
                        c["depth_min"], c["depth_max"]] + c["depth_by_chunk"])
    (out / "result.json").write_text(json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in cells], indent=2, default=str) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pool_review_mpl"))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    x = np.arange(len(cells))
    for j, ax in enumerate(axes.ravel()):
        for i, c in enumerate(cells):
            ax.plot([i, i], [c["pq"][j, 0], c["pq"][j, 4]], color="#4878a8", alpha=.45, linewidth=2, solid_capstyle="butt", label="Pooled 90% interval" if i == 0 else None)
            ax.plot([i, i], [c["pq"][j, 1], c["pq"][j, 3]], color="#4878a8", alpha=.9, linewidth=6, solid_capstyle="butt", label="Pooled 50% interval" if i == 0 else None)
            ax.plot([i - .22, i + .22], [c["pq"][j, 2]] * 2, color="#28659b", linewidth=1.6, label="Pooled median" if i == 0 else None)
            ax.plot(i + .28, c["pooled"][j], marker="D", color="#d87916", markersize=7, linestyle="none", label="Pooled representative (a draw)" if i == 0 else None)
            ax.plot(i - .28, c["opening"][j], marker="s", color="#2a9d8f", markersize=6, linestyle="none", label="Opening-window SGM (a draw)" if i == 0 else None)
            ax.plot([i - .45, i + .45], [c["sel"][j]] * 2, color="#603696", linestyle="--", linewidth=1.5, label="SEL (condition reference)" if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels([c["label"] for c in cells], fontsize=9)
        ax.set_title(SHORT[KEYS[j]], fontsize=10); ax.grid(alpha=.18, axis="y"); ax.spines[["top", "right"]].set_visible(False)
        if KEYS[j] in ("mu_pc", "prob_photo_bleach", "lambda_rate"):
            ax.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, .03), frameon=False)
    fig.suptitle("Equal-window pooling per cell: pooled mixture intervals, representatives and the SEL references", fontsize=13)
    fig.text(.5, .005, "100,000 retained joint draws per cell (ten 2 s windows x 10,000); representative = large-pool Weiszfeld member in normalized log10 coordinates; log axes for mu_pc, bleaching, lambda",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .95)); fig.savefig(out / "four_cell_comparison.png", dpi=160); plt.close(fig)

    md = ["# Four-cell aggregation comparison", "",
          "Side-by-side tabulation of the per-cell equal-window aggregation results (10 windows x 10,000 retained joint draws per cell; "
          "window weight 0.1 each; representative = the repository's large-pool Weiszfeld member in prior-width-normalized log10 coordinates; "
          "pooled quantiles = mixture quantiles in log10, endpoints in physical units). Nothing is recomputed from the draws here. "
          "Descriptive; no aggregation rule or nuisance selection is adopted.", "",
          "| cell | condition | representative from | SEL inside every marginal 90 % (windows of 10) | pooled vector inside every marginal 90 % (windows of 10) | parameters whose ten 90 % intervals share a non-empty intersection (of 6) | SEL distance rank vs window SGM (min–max over windows) | source collection sha256 |", "|---|---|---|---:|---:|---:|---|---|"]
    md += [f"| {c['label']} | {c['condition']} | chunk {c['source_chunk']}, draw {c['draw']} ({c['method']}) | {c['n_inside']} | {c['pooled_inside']} | {sum(not v['empty'] for v in c['intersection'].values())} | {c['depth_min']:.2f}–{c['depth_max']:.2f} | {c['source_sha256'][:16]}… |" for c in cells]
    md += ["", "The membership counts and the intersections are marginal-compatibility diagnostics, not joint credible-region tests and not an independent "
           "validation (the pooled vector is selected from the same draws). The distance rank is the share of a window's draws farther from that window's "
           "SGM than SEL is, in prior-scaled estimator coordinates: a distance rank relative to the SGM, not a posterior-density statement.", "",
           "## Intersection of the ten windows' 90 % intervals (physical units; empty when the windows share no common value)", "",
           "| cell | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for c in cells:
        cellsrow = []
        for k in KEYS:
            v = c["intersection"][k]
            cellsrow.append("empty" if v["empty"] else "{:.4g}–{:.4g} (pooled {}, SEL {})".format(v["low"], v["high"], "in" if v["pooled_inside"] else "out", "in" if v["sel_inside"] else "out"))
        md.append(f"| {c['label']} | " + " | ".join(cellsrow) + " |")
    md.append("")
    for j, k in enumerate(KEYS):
        md += [f"## {k}", "", "| cell | SEL | opening-window SGM | pooled representative | pooled median | pooled 50 % | pooled 90 % | pooled rep. vs SEL | window SGM range |", "|---|---:|---:|---:|---:|---|---|---:|---|"]
        for c in cells:
            r = c["sgm_range"][k]
            md.append("| {} | {:.5g} | {:.5g} | {:.5g} | {:.5g} | {:.5g}–{:.5g} | {:.5g}–{:.5g} | {:+.1f} % ({:+.3f} dex) | {:.4g}–{:.4g} |".format(
                c["label"], c["sel"][j], c["opening"][j], c["pooled"][j], c["pq"][j, 2], c["pq"][j, 1], c["pq"][j, 3], c["pq"][j, 0], c["pq"][j, 4],
                (c["pooled"][j] / c["sel"][j] - 1) * 100, np.log10(c["pooled"][j] / c["sel"][j]), r[0], r[1]))
        md.append("")
    md += ["![Four-cell comparison](four_cell_comparison.png)", "", "Files: `four_cell_comparison.csv` (one row per cell x parameter), `per_cell_depth.csv`, `result.json`, `four_cell_comparison.png`. "
           "Produced by `four_cell_comparison.py` (`Script_Bank/Analysis/Evidence/Window_Posterior_Additional_Cells_2026-11-03/`).", ""]
    (out / "README.md").write_text("\n".join(md))
    print("\n".join(md[:12])); print("wrote", out)


if __name__ == "__main__":
    main()
