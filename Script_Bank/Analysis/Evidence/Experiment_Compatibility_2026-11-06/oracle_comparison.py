"""Full-data numerical oracle: compare the production compatibility output (the regenerated review-copy archives (label COHORT60RANKS or a successor),
format 2) with the independent review calculation (POSTERIOR_RANK_WEIGHTING_REVIEW_2026-11-06: comparison_matrices.npz,
candidate_weights.csv), and the constructor's posterior_rank_median probabilities with the review's median-rank weights.

Usage (repository root, PC profile):
    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python Script_Bank/Analysis/Evidence/Experiment_Compatibility_2026-11-06/oracle_comparison.py \
        --review <POSTERIOR_RANK_WEIGHTING_REVIEW_2026-11-06 dir> --fab <FAB COHORT60RANKS dir> --inlb <INLB COHORT60RANKS dir> --out <json>

Read-only on every input. Writes one JSON with the comparisons and exits non-zero on any disagreement beyond the stated
tolerances (exact equality is the target; a one-draw rank difference would be listed pair by pair).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg

TARGET_EFFECTIVE = {("FAB", "all"): 58.49, ("FAB", "opening"): 56.64, ("INLB", "all"): 49.08, ("INLB", "opening"): 35.49}


def block_matrix(archive, kind, window_set, construction, cell=None):
    """The (n, n) rank matrix of one block in stable identity order, with its ordered identities."""
    a = archive["arrays"]
    for b in archive["compatibility"]["blocks"]:
        if b["kind"] == kind and b["window_set"] == window_set and b["construction"] == construction and (cell is None or int(b["cell"]) == int(cell)):
            gi = b["group_index"]
            ids = [(int(a["cell"][i]), int(a["chunk"][i])) for i in gi]
            order = np.argsort([(c, w) for c, w in ids], axis=0)[:, 0] if False else np.lexsort(([w for _, w in ids], [c for c, _ in ids]))
            return b["ranks"][np.ix_(order, order)], [ids[o] for o in order], b
    raise KeyError((kind, window_set, construction, cell))


def compare_matrices(name, ours, theirs, report):
    ours = np.asarray(ours, float); theirs = np.asarray(theirs, float)
    assert ours.shape == theirs.shape, (name, ours.shape, theirs.shape)
    mask = ~np.eye(ours.shape[-1], dtype=bool)
    if ours.ndim == 3:
        mask = np.broadcast_to(mask, ours.shape)
    o, t = ours[mask], theirs[mask]
    diff = np.abs(o - t)
    n_draw = None
    entry = {"entries": int(o.size), "exact_equal": bool(np.array_equal(o, t)), "max_abs_difference": float(diff.max()),
             "n_differing": int(np.sum(diff > 0)), "diagonal_masked_ours": bool(np.all(np.isnan(ours[~mask] if ours.ndim == 2 else ours[~mask]))),
             "diagonal_masked_theirs": bool(np.all(np.isnan(theirs[~mask] if theirs.ndim == 2 else theirs[~mask])))}
    if entry["n_differing"]:
        idx = np.argwhere(mask & (np.abs(ours - theirs) > 0))
        entry["differing_pairs"] = [{"index": [int(v) for v in i], "ours": float(ours[tuple(i)]), "review": float(theirs[tuple(i)])} for i in idx[:50]]
    report[name] = entry
    return entry["exact_equal"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--fab", type=Path, required=True)
    ap.add_argument("--inlb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    review = np.load(args.review / "comparison_matrices.npz")
    weights = {}
    with (args.review / "candidate_weights.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["method"] != "median_rank":
                continue
            key = (row["condition"], row["window_set"], row["construction"], int(row["cell_or_position"]), int(row["chunk"]) if row["chunk"] else -1)
            weights[key] = (float(row["score"]), float(row["sampling_probability_within_construction_or_cell"]))
    report = {"review_dir": str(args.review), "archives": {}, "matrices": {}, "scores": {}, "probabilities": {}, "named_facts": {}, "ok": True}
    ok = True
    for cond, folder in (("FAB", args.fab), ("INLB", args.inlb)):
        arc = agg.load_candidate_archive(folder / "aggregation_candidates.npz")
        report["archives"][cond] = {"path": str(folder / "aggregation_candidates.npz"), "sha256": arc["sha256"], "format": arc["format_version"],
                                   "n_blocks": len(arc["compatibility"]["blocks"]), "status_counts": arc["manifest"]["compatibility"]["status_counts"],
                                   "n_zero_scores": arc["manifest"]["compatibility"]["n_zero_scores"],
                                   "seconds": arc["manifest"]["compatibility"]["seconds"]}
        a = arc["arrays"]
        # cell matrices, all and opening
        for ws in ("all", "opening"):
            R, ids, blk = block_matrix(arc, cond, ws, "per_cell")
            assert [c for c, _ in ids] == list(range(60)), ids
            ok &= compare_matrices(f"{cond}_{ws}_cells", R, review[f"{cond}_{ws}_rank"], report["matrices"])
            # scores and probabilities (per_cell, posterior_rank_median)
            comp = agg.archive_component(arc, kind=cond, window_set=ws, construction="per_cell")
            p, norm, s = ndli.member_probabilities("posterior_rank_median", 1.0, comp, where=f"{cond}/{ws}/per_cell")
            cells = [d["cell"] for d in comp["identities"]]
            rev_s = np.array([weights[(cond, ws, "per_cell", c, -1)][0] for c in cells]); rev_p = np.array([weights[(cond, ws, "per_cell", c, -1)][1] for c in cells])
            report["scores"][f"{cond}_{ws}_per_cell"] = {"exact_equal": bool(np.array_equal(s, rev_s)), "max_abs_difference": float(np.abs(s - rev_s).max()),
                                                        "n_zero_ours": int(np.sum(s == 0)), "zero_cells_ours": [c for c, v in zip(cells, s) if v == 0],
                                                        "n_zero_review": int(np.sum(rev_s == 0))}
            report["probabilities"][f"{cond}_{ws}_per_cell"] = {"max_abs_difference": float(np.abs(p - rev_p).max()), "allclose_1e-12": bool(np.allclose(p, rev_p, atol=1e-12, rtol=0)),
                                                               "effective_number_ours": float(1 / np.sum(p ** 2)), "effective_number_review": float(1 / np.sum(rev_p ** 2)),
                                                               "effective_number_plan_target": TARGET_EFFECTIVE[(cond, ws)],
                                                               "max_probability": float(p.max()), "n_zero": int(np.sum(p == 0))}
            ok &= report["scores"][f"{cond}_{ws}_per_cell"]["exact_equal"] and report["probabilities"][f"{cond}_{ws}_per_cell"]["allclose_1e-12"]
            ok &= abs(report["probabilities"][f"{cond}_{ws}_per_cell"]["effective_number_ours"] - TARGET_EFFECTIVE[(cond, ws)]) < 0.01
        # within-cell window matrices, all
        W = np.full((60, 10, 10), np.nan)
        for c in range(60):
            R, ids, blk = block_matrix(arc, cond, "all", "per_cell_window", cell=c)
            assert [w for _, w in ids] == list(range(10)), ids
            W[c] = R
        ok &= compare_matrices(f"{cond}_window_matrices", W, review[f"{cond}_window_rank"], report["matrices"])
        comp = agg.archive_component(arc, kind=cond, window_set="all", construction="per_cell_window")
        p, norm, s = ndli.member_probabilities("posterior_rank_median", 1.0, comp, where=f"{cond}/all/per_cell_window")
        idents = [(d["cell"], d["chunk"]) for d in comp["identities"]]
        rev_s = np.array([weights[(cond, "all", "per_cell_window", c, w)][0] for c, w in idents])
        rev_p = np.array([weights[(cond, "all", "per_cell_window", c, w)][1] for c, w in idents])      # within-cell normalized in the review
        ours_within = p * 60.0                                                                           # ours carries the 1/60 cell mass
        cell_mass = {c: float(p[[i for i, (cc, _) in enumerate(idents) if cc == c]].sum()) for c in range(60)}
        report["scores"][f"{cond}_all_per_cell_window"] = {"exact_equal": bool(np.array_equal(s, rev_s)), "max_abs_difference": float(np.abs(s - rev_s).max()),
                                                           "n_zero_ours": int(np.sum(s == 0)), "zero_windows_ours": [list(i) for i, v in zip(idents, s) if v == 0],
                                                           "n_zero_review": int(np.sum(rev_s == 0))}
        report["probabilities"][f"{cond}_all_per_cell_window"] = {"within_cell_max_abs_difference": float(np.abs(ours_within - rev_p).max()),
                                                                  "allclose_1e-12": bool(np.allclose(ours_within, rev_p, atol=1e-12, rtol=0)),
                                                                  "every_cell_mass_is_1_over_60": bool(all(abs(v - 1 / 60) < 1e-12 for v in cell_mass.values())),
                                                                  "n_zero": int(np.sum(p == 0))}
        ok &= report["scores"][f"{cond}_all_per_cell_window"]["exact_equal"] and report["probabilities"][f"{cond}_all_per_cell_window"]["allclose_1e-12"]
        ok &= report["probabilities"][f"{cond}_all_per_cell_window"]["every_cell_mass_is_1_over_60"]
        # the existing candidate vectors and identities must be the COHORT60PLOTS ones (exact)
        plots_dir = Path(re.sub(r"_[A-Z0-9]+$", "_COHORT60PLOTS", str(folder).rstrip("/")))     # the review label is the last token
        plots = agg.load_candidate_archive(plots_dir / "aggregation_candidates.npz")
        same = all(np.array_equal(plots["arrays"][k], a[k]) for k in ("sgm_log10", "median_log10", "quantiles_log10", "cell", "chunk", "construction", "window_set",
                                                                      "sgm_source_cell", "sgm_source_chunk", "sgm_source_draw", "sgm_method", "equivalent_to",
                                                                      "outside_box_fraction", "members_json"))
        report["archives"][cond]["v1_arrays_identical_to_COHORT60PLOTS"] = bool(same); ok &= same
    # the named facts of the brief (INLB)
    inlb = agg.load_candidate_archive(args.inlb / "aggregation_candidates.npz")
    R34, ids34, _ = block_matrix(inlb, "INLB", "all", "per_cell_window", cell=34)
    comp = agg.archive_component(inlb, kind="INLB", window_set="all", construction="per_cell_window")
    sc = {(d["cell"], d["chunk"]): d["compatibility"]["score"] for d in comp["identities"]}
    report["named_facts"] = {
        "INLB_cell34_window8_score": sc[(34, 8)], "INLB_cell34_window9_score": sc[(34, 9)],
        "INLB_cell34_rank_8_vs_9": float(R34[8, 9]), "INLB_cell34_rank_9_vs_8": float(R34[9, 8]),
        "INLB_cell34_mean_row_8": float(np.nanmean(R34[8])), "INLB_cell34_mean_row_9": float(np.nanmean(R34[9])),
        "INLB_cell54_window0_score": sc[(54, 0)], "INLB_cell54_window1_score": sc[(54, 1)],
        "INLB_opening_zero_cells": report["scores"]["INLB_opening_per_cell"]["zero_cells_ours"],
        "FAB_all_zero_cells": report["scores"]["FAB_all_per_cell"]["zero_cells_ours"], "INLB_all_zero_cells": report["scores"]["INLB_all_per_cell"]["zero_cells_ours"],
    }
    nf = report["named_facts"]
    ok &= nf["INLB_cell34_window8_score"] == 0.0 and nf["INLB_cell34_window9_score"] == 0.0 and abs(nf["INLB_cell34_rank_8_vs_9"] - 0.9995) < 5e-4 \
        and abs(nf["INLB_cell34_rank_9_vs_8"] - 0.9999) < 5e-4 and nf["INLB_cell54_window0_score"] == 0.0 and nf["INLB_cell54_window1_score"] == 0.0 \
        and nf["INLB_opening_zero_cells"] == [24, 49, 58] and nf["FAB_all_zero_cells"] == [] and nf["INLB_all_zero_cells"] == []
    report["ok"] = bool(ok)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k in ("matrices", "scores", "probabilities", "named_facts", "ok")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
