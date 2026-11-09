"""Compare the proposal's member weights under the two rank spaces (all six coordinates against the four retained
coordinates) for one condition, from the two review artifacts and their member CSV exports: per-row probabilities,
effective rows, within-cell effective windows, zero rows, the largest within-cell share changes, the rank correlation
of the two weight vectors, and the weighted marginal summaries. Writes JSON and a Markdown table.

Usage: python compare_spaces.py --condition FAB --all <PROPOSAL600ALL dir> --sb4 <PROPOSAL600SB4 dir> --out <folder>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/mars-fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp")
sys.path.insert(0, str(REPO))
from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli            # noqa: E402
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg            # noqa: E402


def members(folder):
    csv_path = next(folder.glob("*_Analysis/representatives_members.csv"))
    rows = list(csv.DictReader(csv_path.open()))
    return {(int(r["cell"]), int(r["chunk"])): r for r in rows}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True); ap.add_argument("--all", type=Path, required=True)
    ap.add_argument("--sb4", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    nu_all = ndli.NuisanceDLI.load(next(a.all.glob("*_Nuisance_DLI.npz")))
    nu_sb4 = ndli.NuisanceDLI.load(next(a.sb4.glob("*_Nuisance_DLI.npz")))
    assert np.array_equal(nu_all.samples, nu_sb4.samples), "the two artifacts must store the same rows"
    m_all, m_sb4 = members(a.all), members(a.sb4)
    assert set(m_all) == set(m_sb4)
    keys_cw = sorted(m_all)
    p_all = np.array([float(m_all[k]["contribution"]) for k in keys_cw]); p_sb4 = np.array([float(m_sb4[k]["contribution"]) for k in keys_cw])
    s_all = np.array([float(m_all[k]["score_used"]) for k in keys_cw]); s_sb4 = np.array([float(m_sb4[k]["score_used"]) for k in keys_cw])
    cells = np.array([k[0] for k in keys_cw]); chunks = np.array([k[1] for k in keys_cw])
    C = np.unique(cells)
    def within(p):
        eff = {}; share = np.zeros_like(p)
        for c in C:
            m = cells == c; s = p[m].sum(); share[m] = p[m] / s if s > 0 else 0.0
            eff[int(c)] = float(1.0 / np.sum((p[m] / s) ** 2)) if s > 0 else None
        return eff, share
    eff_all, sh_all = within(p_all); eff_sb4, sh_sb4 = within(p_sb4)
    d = sh_sb4 - sh_all
    order = np.argsort(-np.abs(d))
    largest = [{"cell": int(cells[i]), "window": int(chunks[i]), "share_all": float(sh_all[i]), "share_sb4": float(sh_sb4[i]),
                "score_all": float(s_all[i]), "score_sb4": float(s_sb4[i])} for i in order[:15]]
    zeros_all = [(int(cells[i]), int(chunks[i])) for i in np.flatnonzero(p_all == 0)]
    zeros_sb4 = [(int(cells[i]), int(chunks[i])) for i in np.flatnonzero(p_sb4 == 0)]
    keys = list(nu_all.parameter_keys)
    def marg(nu):
        out = {}
        for j, k in enumerate(keys):
            q = [10 ** agg.weighted_quantile(nu.samples[:, j], nu.probabilities, p) for p in (0.05, 0.5, 0.95)]
            out[k] = {"q05": q[0], "median": q[1], "q95": q[2]}
        return out
    ea = np.array(list(eff_all.values()), float); es = np.array(list(eff_sb4.values()), float)
    res = {"condition": a.condition, "n_rows": int(nu_all.samples.shape[0]),
           "effective_rows": {"all": nu_all.selection["effective_rows"], "spatial_brightness": nu_sb4.selection["effective_rows"]},
           "zero_rows": {"all": zeros_all, "spatial_brightness": zeros_sb4},
           "within_cell_effective_windows": {"all": {"min": float(ea.min()), "median": float(np.median(ea)), "max": float(ea.max()), "n_below_8": int((ea < 8).sum())},
                                             "spatial_brightness": {"min": float(es.min()), "median": float(np.median(es)), "max": float(es.max()), "n_below_8": int((es < 8).sum())}},
           "spearman_rank_correlation_of_row_probabilities": spearman(p_all, p_sb4),
           "max_abs_within_cell_share_change": float(np.abs(d).max()), "median_abs_within_cell_share_change": float(np.median(np.abs(d))),
           "n_rows_share_change_over_0_02": int((np.abs(d) > 0.02).sum()),
           "largest_within_cell_share_changes": largest,
           "weighted_marginals_physical": {"all": marg(nu_all), "spatial_brightness": marg(nu_sb4)},
           "rank_coordinates_disjoint_from_overrides": {"all": nu_all.selection.get("rank_coordinates_disjoint_from_overrides"),
                                                        "spatial_brightness": nu_sb4.selection.get("rank_coordinates_disjoint_from_overrides")}}
    (a.out / f"comparison_{a.condition}.json").write_text(json.dumps(res, indent=2) + "\n")
    lines = [f"# {a.condition}: member weights in the two rank spaces", "",
             "| quantity | all six coordinates | four retained coordinates |", "|---|---|---|",
             f"| effective rows (baseline 600) | {res['effective_rows']['all']:.2f} | {res['effective_rows']['spatial_brightness']:.2f} |",
             f"| zero-probability rows | {len(zeros_all)} {zeros_all} | {len(zeros_sb4)} {zeros_sb4} |",
             f"| within-cell effective windows min / median / max | {ea.min():.2f} / {np.median(ea):.2f} / {ea.max():.2f} | {es.min():.2f} / {np.median(es):.2f} / {es.max():.2f} |",
             f"| cells below 8 effective windows | {int((ea < 8).sum())} | {int((es < 8).sum())} |",
             f"| ranked coordinates disjoint from the overrides | {res['rank_coordinates_disjoint_from_overrides']['all']} | {res['rank_coordinates_disjoint_from_overrides']['spatial_brightness']} |",
             "", f"Spearman rank correlation of the 600 row probabilities: {res['spearman_rank_correlation_of_row_probabilities']:.4f}; "
             f"within-cell share changes: max {np.abs(d).max():.4f}, median {np.median(np.abs(d)):.4f}, {int((np.abs(d) > 0.02).sum())} rows change by more than 0.02 "
             f"(equal weighting would give every window 1/10).", "", "| parameter | median (all) | median (four) | 90 % range (all) | 90 % range (four) |", "|---|---|---|---|---|"]
    for k in keys:
        A = res["weighted_marginals_physical"]["all"][k]; B = res["weighted_marginals_physical"]["spatial_brightness"][k]
        lines.append(f"| {k} | {A['median']:.5g} | {B['median']:.5g} | {A['q05']:.5g} – {A['q95']:.5g} | {B['q05']:.5g} – {B['q95']:.5g} |")
    lines += ["", "Largest within-cell share changes (four-coordinate minus all-coordinate):", "", "| cell | window | share (all) | share (four) | score (all) | score (four) |", "|---|---|---|---|---|---|"]
    for r in largest:
        lines.append(f"| {r['cell']} | {r['window']} | {r['share_all']:.4f} | {r['share_sb4']:.4f} | {r['score_all']:.4f} | {r['score_sb4']:.4f} |")
    (a.out / f"comparison_{a.condition}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
