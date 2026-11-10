"""Weighting-revision review builds (this folder only, tag PROPOSALCW4; never a production name, never the Posit directory).

The proposed revised selection: the SAME 600 SGM representatives per condition as the adopted RANK4 (one per cell and 2 s
window over all ten windows), the SAME four ranked coordinates (rank space ``spatial_brightness``) and the SAME bleaching
and flicker overrides, with ONE change: the comparison scope. Each representative's score is its median distance rank
among the other 599 cell-window groups of the condition (``rank_scope = "construction"``; archive format 4, review label
COHORT60RANKSR4), and the scores are normalized over all 600; no cell keeps a reserved mass. RANK4 (within-cell scores,
equal cell mass; archive COHORT60RANKSR3) is read for comparison only and is never written.

The spec is derived from the adopted RANK4 spec text (the override tables are carried verbatim). The script runs the
companion's own dry-run and build paths, validates the artifact through the biology loader gate
(``require_nuisance_dli``) pointed at this folder, compares against RANK4, the equal-member baseline and the reviewer's
exploratory probabilities, and writes ``comparison_<COND>.json``.

Usage (repository root):
    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python <this file> --condition FAB|INLB
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path("/home/mars-fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp")
POSIT = Path("/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit")
EXPLORATORY = HERE.parent / "WEIGHTING_SCOPE_REVIEW_2026-11-10"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Script_Bank" / "Analysis"))
import SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI as companion      # noqa: E402
from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli     # noqa: E402
from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg     # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import to_physical      # noqa: E402

TAG = "PROPOSALCW4"
TAGS = {"FAB": "C128V42A", "INLB": "C128V42AP3LR32"}
OLD_LABEL, NEW_LABEL = "COHORT60RANKSR3", "COHORT60RANKSR4"
RATIONALE = (
    "Review proposal {tag} (2026-11-10; NOT adopted, built for review only): one SGM representative per cell and 2 s window over all ten "
    "windows of the sixty {cond} recordings (600 vectors), so that the between-cell and the within-recording variation of the effective "
    "imaging conditions across the full acquisition both enter the simulated population; member weighting posterior_rank_median under "
    "rank_scope construction: each representative's probability is proportional to the median cross-posterior distance rank of its SGM "
    "among the other 599 cell-window groups of the condition (the other windows of its own cell are nine of those references, with no "
    "separate treatment), normalized over all 600 representatives. This is consensus over the complete representative population of the "
    "aggregation; no cell keeps a reserved mass, so a cell whose windows lie away from the population receives less total probability, "
    "possibly zero (a majority-zero window receives exactly zero; both are sampling outcomes of the policy, documented, not data-quality "
    "verdicts). It revises the scope of the adopted RANK4 selection (within-cell ranks with equal cell mass), which stays unchanged under "
    "its own tag. The four spatial and brightness coordinates of each SGM are retained together; the bleaching probability is overridden "
    "by the recorded value 0.05 (empirically anchored for Fab, declared for InlB) and the flicker rate by the direct "
    "single-dye-equivalent reference of the committed derivation, because the estimator does not recover the former and only partially "
    "recovers the latter. One complete vector is sampled per synthetic 2 s video and held fixed during it. Member weights from the ranks "
    "over the four retained coordinates only (rank space spatial_brightness), so the weights rest on exactly the coordinates the final "
    "vectors carry.")
HEADER = (
    "# REVIEW Nuisance_DLI spec {tag} (2026-11-10; not adopted, not for generation): one SGM per cell and 2 s window over all ten windows,\n"
    "# posterior_rank_median over mu_r, sigma_r, mu_pc, sigma_pc with rank_scope construction (every representative against the other 599,\n"
    "# normalized over all 600, no reserved cell mass); bleaching and flicker overridden exactly as in RANK4 (reasons below, verbatim).\n"
    "# The adopted RANK4 (within-cell scope) and the historical fixed reference SEL are unchanged.\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(samples, p, cells, chunks, table, plo, phi):
    keys = [e["KEY"] for e in table]
    out = {"probability_sum": float(p.sum()), "effective_rows": float(1.0 / np.sum(p ** 2)), "zero_probability_rows": int(np.sum(p == 0)),
           "max_row_probability": float(p.max()), "opening_window_mass": float(p[chunks == 0].sum()),
           "window_mass_by_start_chunk": [float(p[chunks == w].sum()) for w in sorted(set(chunks.tolist()))]}
    mass = np.asarray([p[cells == c].sum() for c in sorted(set(cells.tolist()))])
    out.update({"cell_total_min": float(mass.min()), "cell_total_max": float(mass.max()), "zero_total_cells": int(np.sum(mass == 0)),
                "effective_cells": float(1.0 / np.sum(mass ** 2))})
    outside = np.any((samples < plo) | (samples > phi), axis=1)
    out["outside_prior_mass"] = float(p[outside].sum())
    out["outside_prior_by_key"] = {k: float(p[(samples[:, j] < plo[j]) | (samples[:, j] > phi[j])].sum()) for j, k in enumerate(keys)}
    q = {}
    for j, k in enumerate(keys):
        vals = [agg.weighted_quantile(samples[:, j], p, lev) for lev in (0.05, 0.5, 0.95)]
        q[k] = [float(v) for v in to_physical(np.asarray(vals)[:, None], [table[j]])[:, 0]]
    out["marginal_p05_median_p95_physical"] = q
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--condition", required=True, choices=("FAB", "INLB"))
    args = ap.parse_args()
    cond = args.condition
    out = HERE / cond
    out.mkdir(parents=True, exist_ok=True)
    alias = f"SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{cond}"
    rank4_spec = POSIT / f"{alias}_2S_50FPS_RANK4_Nuisance_DLI_Spec.toml"
    rank4_art = POSIT / f"{alias}_2S_50FPS_RANK4_Nuisance_DLI.npz"
    rank4_sha_before = sha(rank4_art)
    new_archive = POSIT / f"{alias}_2S_50FPS_{TAGS[cond]}_MAP_Experiment_{NEW_LABEL}" / "aggregation_candidates.npz"
    # ---- the review spec: the adopted spec text with the archive, the scope and the rationale changed; overrides verbatim ----
    text = rank4_spec.read_text()
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#")).lstrip("\n")
    old_arch = re.search(r'^candidate_archive = "(.*)"$', body, re.M).group(1)
    assert OLD_LABEL in old_arch
    body = body.replace(old_arch, str(new_archive))
    body = re.sub(r'^candidate_archive_sha256 = ".*"$', f'candidate_archive_sha256 = "{sha(new_archive)}"', body, flags=re.M)
    assert body.count('member_weighting = "posterior_rank_median"\n') == 1 and "rank_scope" not in body      # the adopted spec predates the scope field
    body, n = re.subn(r'^rationale = ".*"$', lambda m: 'rationale = "' + RATIONALE.format(tag=TAG, cond=cond) + '"', body, flags=re.M)
    assert n == 1
    body = body.replace('member_weighting = "posterior_rank_median"\n', 'member_weighting = "posterior_rank_median"\nrank_scope = "construction"\n')
    over_old = text[text.index("[override.prob_photo_bleach]"):]; over_new = body[body.index("[override.prob_photo_bleach]"):]
    assert over_old.strip() == over_new.strip()                       # the override tables travel verbatim
    stem = f"{alias}_2S_50FPS_{TAG}_Nuisance_DLI"
    spec = out / f"{stem}_Spec.toml"
    spec.write_text(HEADER.format(tag=TAG) + "\n" + body + "\n")
    art = out / f"{stem}.npz"
    if art.exists():
        raise SystemExit(f"refusing to overwrite {art}")
    R = companion._resolve(2.0, cond)
    table = det.DETECTOR_PARAMETERIZATION; keys = list(R["imaging_keys"])
    plo, phi = np.asarray(R["plo"], float), np.asarray(R["phi"], float)
    spec_dict = ndli.load_spec(spec, R["imaging_keys"], R["plo"], R["phi"], table=table)
    block = spec_dict["block"]
    print("==== dry run (the companion's own console path) ====", flush=True)
    companion._build_representatives(SimpleNamespace(condition=cond, dry_run=True), R, block, spec_dict, art, spec)
    print("==== review build + report ====", flush=True)
    companion._build_representatives(SimpleNamespace(condition=cond, dry_run=False), R, block, spec_dict, art, spec)
    # ---- loader validation: the biology gate, pointed at this folder (the Posit directory holds no such tag) ----
    assert not (POSIT / art.name).exists()
    nu = ndli.require_nuisance_dli(out, alias, "2S_50FPS", TAG)
    sel = nu.selection; comp = sel["components"][0]
    draws = nu.sample(20000, seed=20261110)
    stored = {tuple(r): k for k, r in enumerate(nu.samples)}
    idx = np.asarray([stored[tuple(d)] for d in draws])                # KeyError if a draw is not a stored row
    assert np.all(nu.probabilities[idx] > 0)
    # ---- identities, by recorded membership (never by array position) ----
    cells = np.empty(nu.samples.shape[0], np.int64); chunks = np.empty_like(cells); score = np.empty(nu.samples.shape[0])
    for m in comp["members"]:
        r = m["stored_row"]; cells[r], chunks[r], score[r] = m["cell"], m["chunk"], m["score_used"]
        assert abs(m["probability"] - nu.probabilities[r]) < 1e-15
    old = ndli.NuisanceDLI.load(rank4_art)
    ocomp = old.selection["components"][0]
    ocells = np.empty(old.samples.shape[0], np.int64); ochunks = np.empty_like(ocells)
    for m in ocomp["members"]:
        ocells[m["stored_row"]], ochunks[m["stored_row"]] = m["cell"], m["chunk"]
    same_rows = bool(np.array_equal(old.samples, nu.samples) and np.array_equal(ocells, cells) and np.array_equal(ochunks, chunks))
    base = np.asarray(sel["equal_member_baseline"]["probabilities"], float)
    # ---- the reviewer's exploratory probabilities, matched through the archive's group identities ----
    with np.load(EXPLORATORY / f"{cond}_construction_wide_ranks.npz", allow_pickle=False) as z:
        xi, xp, xs = z["per_cell_window_group_index"], z["per_cell_window_probability"], z["per_cell_window_score"]
    arc = agg.load_candidate_archive(new_archive)
    A = arc["arrays"]
    row_of = {(int(c), int(w)): r for r, (c, w) in enumerate(zip(cells, chunks))}
    rows = np.asarray([row_of[(int(A["cell"][g]), int(A["chunk"][g]))] for g in xi])
    result = {
        "condition": cond, "tag": TAG, "status": "review product; not adopted; not in the Posit directory",
        "artifact": {"path": str(art), "sha256": sha(art), "spec": str(spec), "spec_sha256": sha(spec)},
        "archive": {"path": str(new_archive), "sha256": arc["sha256"], "format": arc["format_version"],
                    "comparison_scope": arc["compatibility"]["comparison_scope"], "protocol": arc["compatibility"]["protocol"]["name"]},
        "selection": {"grouping": comp["grouping"], "window_set": comp["window_set"], "member_weighting": comp["member_weighting"],
                      "rank_scope": comp["rank_scope"], "reference_population": comp["reference_population"], "rank_space": comp["rank_space"],
                      "rank_coordinates": comp["rank_coordinates"], "references_per_member": comp["normalizers"]["references_per_member"],
                      "normalization_rule": comp["normalizers"]["rule"], "sum_of_scores": comp["normalizers"]["sum_of_scores"],
                      "zero_mass_cells": comp["normalizers"]["zero_mass_cells"],
                      "rank_coordinates_disjoint_from_overrides": sel["rank_coordinates_disjoint_from_overrides"],
                      "overrides": {k: v["value_physical"] for k, v in sel["overrides"].items()}},
        "loader_validation": {"loaded_through": "detector_nuisance_dli.require_nuisance_dli(<this folder>, alias, 2S_50FPS, PROPOSALCW4)",
                              "choice": nu.posterior_sample_pool_choice, "stored_rows": int(nu.samples.shape[0]),
                              "draws": int(draws.shape[0]), "draws_are_stored_positive_probability_rows": True,
                              "zero_rows_drawn": int(np.sum(nu.probabilities[idx] == 0)),
                              "max_abs_frequency_minus_probability": float(np.max(np.abs(np.bincount(idx, minlength=nu.samples.shape[0]) / draws.shape[0] - nu.probabilities)))},
        "against_exploratory_calculation": {"scores_identical": bool(np.array_equal(score[rows], xs)),
                                            "max_abs_probability_difference": float(np.max(np.abs(nu.probabilities[rows] - xp)))},
        "against_rank4": {"rank4_artifact": str(rank4_art), "rank4_sha256": rank4_sha_before, "rank4_unchanged_after_build": sha(rank4_art) == rank4_sha_before,
                          "rank4_scope": companion._scope_of(ocomp), "same_stored_vectors_and_identities": same_rows,
                          "overrides_identical": {k: v["value_physical"] for k, v in old.selection["overrides"].items()} == {k: v["value_physical"] for k, v in sel["overrides"].items()},
                          "total_variation_distance": float(0.5 * np.sum(np.abs(nu.probabilities - old.probabilities))),
                          "zero_rows_in_both": int(np.sum((nu.probabilities == 0) & (old.probabilities == 0))),
                          "zero_only_here": int(np.sum((nu.probabilities == 0) & (old.probabilities > 0))),
                          "zero_only_in_rank4": int(np.sum((nu.probabilities > 0) & (old.probabilities == 0)))},
        "distributions": {"construction_wide_PROPOSALCW4": stats(nu.samples, nu.probabilities, cells, chunks, table, plo, phi),
                          "within_cell_RANK4": stats(old.samples, old.probabilities, ocells, ochunks, table, plo, phi),
                          "equal_member_baseline": stats(nu.samples, base, cells, chunks, table, plo, phi)},
    }
    assert same_rows and result["against_rank4"]["rank4_unchanged_after_build"] and result["against_rank4"]["overrides_identical"]
    assert result["against_exploratory_calculation"]["scores_identical"] and result["against_exploratory_calculation"]["max_abs_probability_difference"] < 1e-15
    assert comp["rank_scope"] == "construction" and comp["normalizers"]["references_per_member"] == [599] and abs(nu.probabilities.sum() - 1.0) < 1e-12
    (HERE / f"comparison_{cond}.json").write_text(json.dumps(result, indent=1) + "\n")
    d = result["distributions"]
    print(f"{cond}: stored rows {nu.samples.shape[0]}; scope {comp['rank_scope']}; "
          + "; ".join(f"{name}: effective {v['effective_rows']:.2f}, zeros {v['zero_probability_rows']}, opening {100 * v['opening_window_mass']:.2f} %, "
                      f"cell totals {100 * v['cell_total_min']:.3f}..{100 * v['cell_total_max']:.2f} %, outside prior {100 * v['outside_prior_mass']:.1f} %"
                      for name, v in d.items())
          + f"; TV distance to RANK4 {result['against_rank4']['total_variation_distance']:.4f}; exploratory max |dp| "
            f"{result['against_exploratory_calculation']['max_abs_probability_difference']:.2e}; RANK4 unchanged {result['against_rank4']['rank4_unchanged_after_build']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
