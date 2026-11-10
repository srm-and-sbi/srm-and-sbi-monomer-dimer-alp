"""Verify the regenerated complete-cohort archives (review label COHORT60RANKSR4, comparison scope ``construction``)
against (1) the reviewer's exploratory calculation, (2) an independent plain-counting oracle over the retained clouds,
(3) the source archives of the previous review (COHORT60RANKSR3: vectors, identities, within-cell blocks), and confirm
that the earlier products are byte-identical to their recorded checksums. Read-only on every input.

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python Script_Bank/Analysis/Evidence/Comparison_Scope_2026-11-10/verify_construction_wide.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg

HERE = Path(__file__).resolve().parent
POSIT = Path("/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit")
EXPLORATORY = Path("/home/mars-fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/Special_Analyses/"
                   "NUISANCE_DLI_PRODUCTION_PROPOSAL_2026-11-09/WEIGHTING_SCOPE_REVIEW_2026-11-10")
TAGS = {"FAB": "C128V42A", "INLB": "C128V42AP3LR32"}
SPACE = "spatial_brightness"
RANK_KEYS = ("mu_r", "sigma_r", "mu_pc", "sigma_pc")
RECORDED = {  # checksums recorded at adoption (DETECTOR_WORKFLOW 7.6 / the 2026-11-09 evidence folders)
    "RANK4_FAB": "3b89edab3bb4498abfaf5a54c2fdf31e85f8ac5eff110c8a3a95a83bbb59c801",
    "RANK4_INLB": "50a7678ea858aa10f5df950f70d4301278059a10981c208ca3b0996245e661c8",
}
N_ORACLE = 1500


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def product(cond, label):
    return POSIT / f"SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{cond}_2S_50FPS_{TAGS[cond]}_MAP_Experiment_{label}" / "aggregation_candidates.npz"


def clouds_of(manifest, cond):
    out = {}
    for source in manifest["sources"]:
        with np.load(source["path"], allow_pickle=False) as z:
            for n, (ki, cell, chunk) in enumerate(zip(z["kind_index"], z["cell"], z["chunk"])):
                assert str(z["kinds"][ki]) == cond
                out[(int(cell), int(chunk))] = z["posterior_samples_cloud"][n].astype(np.float64)
    return out


def direct_rank(pool, v_ref, v_cand, scale):
    radius = np.sqrt((((pool - v_ref) / scale) ** 2).sum(axis=1))
    reach = np.sqrt((((v_cand - v_ref) / scale) ** 2).sum())
    return int(np.sum(radius > reach)) / radius.size


def one(cond):
    out = {}
    new_path, old_path = product(cond, "COHORT60RANKSR4"), product(cond, "COHORT60RANKSR3")
    new = agg.load_candidate_archive(new_path)            # the reader re-validates every block of every space under the declared scope
    old = agg.load_candidate_archive(old_path)
    A, B = new["arrays"], old["arrays"]
    out["archive"] = {"path": str(new_path), "sha256": new["sha256"], "format": new["format_version"],
                      "comparison_scope": new["compatibility"]["comparison_scope"], "protocol": new["compatibility"]["protocol"]["name"],
                      "rank_spaces": {k: v["coordinates"] for k, v in new["compatibility"]["spaces"].items()}}
    out["previous_archive"] = {"path": str(old_path), "sha256": old["sha256"], "format": old["format_version"],
                               "comparison_scope": old["compatibility"]["comparison_scope"], "protocol": old["compatibility"]["protocol"]["name"]}
    assert new["format_version"] == 4 and new["compatibility"]["comparison_scope"] == "construction"
    assert old["format_version"] == 3 and old["compatibility"]["comparison_scope"] == "within_cell"
    # ---- (3) the groups, vectors, identities and provenance are those of the source archive --------------------
    same = ["construction", "window_set", "kind", "cell", "chunk", "n_draws", "n_windows", "members_json", "sgm_log10", "sgm_source_cell",
            "sgm_source_chunk", "sgm_source_draw", "sgm_source_label", "sgm_method", "equivalent_to", "median_log10", "quantiles_log10",
            "outside_box_fraction", "outside_box_by_key", "parameter_keys", "sgm_scale", "prior_low", "prior_high", "sgm_outside_prior_keys"]
    out["groups_identical_to_previous_archive"] = {k: bool(np.array_equal(A[k], B[k])) for k in same}
    assert all(out["groups_identical_to_previous_archive"].values())
    ms, mo = new["manifest"], old["manifest"]
    out["sources_identical"] = [s["sha256"] for s in ms["sources"]] == [s["sha256"] for s in mo["sources"]] and len(ms["sources"]) == 60
    out["contract_identical"] = ms["contract"] == mo["contract"] and ms["target"] == mo["target"]
    assert out["sources_identical"] and out["contract_identical"]
    keys = [str(k) for k in A["parameter_keys"]]
    G = A["construction"].shape[0]
    sel = lambda c, s="all": np.flatnonzero((A["kind"] == cond) & (A["window_set"] == s) & (A["construction"] == c))   # noqa: E731
    blocks = {lab: {(b["window_set"], b["construction"]): b for b in sp["blocks"]} for lab, sp in new["compatibility"]["spaces"].items()}
    # ---- (1) the reviewer's exploratory calculation, entry by entry ------------------------------------------------
    with np.load(EXPLORATORY / f"{cond}_construction_wide_ranks.npz", allow_pickle=False) as z:
        X = {k: z[k] for k in z.files}
    rep = {}
    for c in ("per_cell", "per_window_position", "per_cell_window"):
        blk = blocks[SPACE][("all", c)]
        gi = np.asarray(blk["group_index"]); xi = X[f"{c}_group_index"]
        assert sorted(gi.tolist()) == sorted(xi.tolist()) == sel(c).tolist()
        order = np.asarray([int(np.flatnonzero(gi == g)[0]) for g in xi])
        R = blk["ranks"][np.ix_(order, order)]
        score = A[f"compatibility_{SPACE}_score"][xi]
        comp = agg.archive_component(new, kind=cond, window_set="all", construction=c)
        assert comp["group_indices"] == xi.tolist()
        p, norm, s = ndli.member_probabilities("posterior_rank_median", 1.0, comp, rank_coordinates=list(RANK_KEYS), rank_scope="construction")
        rep[c] = {"n": int(gi.size), "references_per_candidate": int(gi.size - 1),
                  "ranks_identical": bool(np.array_equal(R, X[f"{c}_ranks"], equal_nan=True)),
                  "off_diagonal_entries": int(gi.size * (gi.size - 1)),
                  "scores_identical": bool(np.array_equal(score, X[f"{c}_score"])),
                  "constructor_scores_identical": bool(np.array_equal(s, X[f"{c}_score"])),
                  "max_abs_probability_difference": float(np.max(np.abs(p - X[f"{c}_probability"]))),
                  "probability_sum": float(p.sum()), "zero_probability_rows": int(np.sum(p == 0)),
                  "effective_representatives": float(1.0 / np.sum(p ** 2)), "normalization_rule": norm["rule"],
                  "comparison_scope": norm["comparison_scope"], "references_per_member": norm["references_per_member"]}
        assert rep[c]["ranks_identical"] and rep[c]["scores_identical"] and rep[c]["constructor_scores_identical"]
        assert rep[c]["max_abs_probability_difference"] < 1e-15 and abs(rep[c]["probability_sum"] - 1.0) < 1e-12
        if c == "per_cell_window":
            cells = A["cell"][xi]; chunks = A["chunk"][xi]
            mass = {int(cc): float(p[cells == cc].sum()) for cc in sorted(set(cells.tolist()))}
            rep[c].update({"opening_window_mass": float(p[chunks == 0].sum()), "cell_total_min": min(mass.values()), "cell_total_max": max(mass.values()),
                           "zero_total_cells": [cc for cc, m in mass.items() if m == 0.0], "n_cells": len(mass),
                           "equal_member_cell_total": 1.0 / len(mass)})
    out["reproduction_of_exploratory_calculation"] = rep
    # ---- (3b) the previous within-cell matrices are the same-cell blocks of the population matrices (both spaces) ---
    wc = {}
    for lab in ("all", SPACE):
        blk = blocks[lab][("all", "per_cell_window")]
        pos = {int(g): k for k, g in enumerate(blk["group_index"])}
        n = 0; ok = True
        for ob in old["compatibility"]["spaces"][lab]["blocks"]:
            if ob["construction"] != "per_cell_window" or ob["window_set"] != "all":
                continue
            idx = [pos[int(g)] for g in ob["group_index"]]
            sub = blk["ranks"][np.ix_(idx, idx)]
            ok &= bool(np.array_equal(sub, ob["ranks"], equal_nan=True)); n += len(idx) * (len(idx) - 1)
        pc_new = A[f"{agg._space_prefix(lab)}_score"][sel("per_cell")]; pc_old = B[f"{agg._space_prefix(lab)}_score"][sel("per_cell")]
        wc[lab] = {"within_cell_entries_checked": n, "identical_to_same_cell_blocks": ok, "per_cell_scores_identical_to_previous": bool(np.array_equal(pc_new, pc_old))}
        assert ok and n == 5400 and wc[lab]["per_cell_scores_identical_to_previous"]
    out["previous_within_cell_matrices_inside_the_population_matrices"] = wc
    # opening only: the per-cell and the cell-window populations coincide -> equal scores, each computed on its own
    eq = {}
    for lab in ("all", SPACE):
        a = A[f"{agg._space_prefix(lab)}_score"][sel("per_cell", "opening")]; b = A[f"{agg._space_prefix(lab)}_score"][sel("per_cell_window", "opening")]
        eq[lab] = bool(np.array_equal(a, b)) and a.size == 60
        st = A[f"{agg._space_prefix(lab)}_status"][sel("per_window_position", "opening")]
        assert eq[lab] and st.tolist() == ["insufficient_peers"]
    out["opening_per_cell_equals_per_cell_window_scores"] = eq
    # ---- (2) independent plain-counting oracle over the retained clouds (no sort, no lookup) ------------------------
    clouds = clouds_of(ms, cond)
    members = [json.loads(str(x)) for x in A["members_json"]]
    scale_all = A["sgm_scale"].astype(np.float64)
    rng = np.random.default_rng(20261110)
    oracle = {}
    for lab, coords in (("all", keys), (SPACE, list(RANK_KEYS))):
        ki = [keys.index(k) for k in coords]; scale = scale_all[ki]
        V = A["sgm_log10"][:, ki].astype(np.float64)
        counts = {}
        for c, n_pairs in (("per_cell", 300), ("per_window_position", 90), ("per_cell_window", N_ORACLE)):
            blk = blocks[lab][("all", c)]; gi = np.asarray(blk["group_index"]); n = gi.size
            pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
            pick = pairs if len(pairs) <= n_pairs else [pairs[k] for k in rng.choice(len(pairs), size=n_pairs, replace=False)]
            pools = {}
            bad = 0
            for a, b in pick:
                j = int(gi[b])
                if j not in pools:
                    pools[j] = np.concatenate([clouds[(int(cc), int(w))][:, ki] for cc, w in members[j]], axis=0)
                    assert pools[j].shape[0] == int(A["n_draws"][j])
                bad += int(blk["ranks"][a, b] != direct_rank(pools[j], V[j], V[int(gi[a])], scale))
            counts[c] = {"pairs_checked": len(pick), "mismatches": bad}
            assert bad == 0
        oracle[lab] = counts
    out["independent_plain_counting_oracle"] = oracle
    return out


if __name__ == "__main__":
    result = {"conditions": {c: one(c) for c in ("FAB", "INLB")}}
    unchanged = {}
    for c in ("FAB", "INLB"):
        art = POSIT / f"SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{c}_2S_50FPS_RANK4_Nuisance_DLI.npz"
        unchanged[f"RANK4_{c}"] = {"sha256": sha(art), "equals_recorded": sha(art) == RECORDED[f"RANK4_{c}"]}
        sel = POSIT / f"SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{c}_2S_50FPS_SEL_Nuisance_DLI.npz"
        unchanged[f"SEL_{c}"] = {"sha256": sha(sel) if sel.exists() else None, "exists": sel.exists()}
        assert unchanged[f"RANK4_{c}"]["equals_recorded"]
    result["earlier_products_unchanged"] = unchanged
    result["all_ok"] = True
    (HERE / "verification_construction_wide.json").write_text(json.dumps(result, indent=1) + "\n")
    for c, r in result["conditions"].items():
        cw = r["reproduction_of_exploratory_calculation"]["per_cell_window"]
        print(f"{c}: exploratory ranks/scores identical for all three constructions; per_cell_window effective {cw['effective_representatives']:.2f}, "
              f"zeros {cw['zero_probability_rows']}, opening mass {100 * cw['opening_window_mass']:.2f} %, cell totals "
              f"{100 * cw['cell_total_min']:.3f} % .. {100 * cw['cell_total_max']:.2f} %, zero-total cells {cw['zero_total_cells']}; "
              f"oracle {r['independent_plain_counting_oracle']}")
    print("ALL OK")
