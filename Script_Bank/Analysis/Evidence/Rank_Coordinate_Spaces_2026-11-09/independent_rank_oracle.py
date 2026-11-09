"""Independent recomputation of the compatibility ranks of a candidate archive, in every rank space it declares, from the
retained draw collections: plain distance counting (no call into experiment_aggregation.reference_distance_ranks), compared
EXACTLY (bitwise after float64 rounding tolerance 0) to the stored block matrices and scores. Also checks that the primary
('all') arrays of the archive equal the reference archive's (the R2 review copy).

Usage: python independent_rank_oracle.py --archive <R3 aggregation_candidates.npz> --reference <R2 aggregation_candidates.npz>
       --sources <cloud_sources_<COND>.txt> --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_archive(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    manifest = json.loads(str(arrays.pop("manifest_json")))
    return arrays, manifest


def load_clouds(paths):
    """{(kind_label, cell, chunk): cloud (n_draws, D)} and the SGM per window from the collections (the stored window SGMs)."""
    clouds, sgms = {}, {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            m = json.loads(str(z["manifest_json"])) if "manifest_json" in z.files else {}
            keys = [k for k in z.files]
            # the collection schema: posterior_samples_cloud (n_windows, n_draws, D), kind_index, cell, chunk, kinds
            cloud = np.asarray(z["posterior_samples_cloud"], np.float64)
            kinds = [str(k) for k in z["kinds"]]
            ki = np.asarray(z["kind_index"]).astype(int); cell = np.asarray(z["cell"]).astype(int); chunk = np.asarray(z["chunk"]).astype(int)
            for r in range(cloud.shape[0]):
                clouds[(kinds[ki[r]], int(cell[r]), int(chunk[r]))] = cloud[r]
    return clouds


def rank_counting(ref_cloud, ref_sgm, cand, scale):
    """Share of reference draws strictly farther from the reference SGM than the candidate (explicit comparison, no sorting)."""
    dz = np.sqrt((((ref_cloud - ref_sgm) / scale) ** 2).sum(axis=1))
    dc = np.sqrt((((cand - ref_sgm) / scale) ** 2).sum())
    return float(np.count_nonzero(dz > dc)) / dz.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True); ap.add_argument("--reference", required=True)
    ap.add_argument("--sources", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arrays, manifest = load_archive(a.archive)
    ref_arrays, ref_manifest = load_archive(a.reference)
    paths = [p for p in Path(a.sources).read_text().strip().split(",") if p]
    clouds = load_clouds(paths)
    keys = [str(k) for k in arrays["parameter_keys"]]
    scale_all = np.asarray(arrays["sgm_scale"], np.float64)
    out = {"archive": a.archive, "reference": a.reference, "n_collections": len(paths), "format_version": manifest["archive_format_version"],
           "primary_equals_reference": {}, "spaces": {}}
    # 1. the primary arrays equal the R2 archive's
    for k in sorted(ref_arrays):
        if k == "manifest_json" or k not in arrays:
            out["primary_equals_reference"][k] = "missing in archive" if k != "manifest_json" else None
            continue
        x, y = arrays[k], ref_arrays[k]
        eq = x.shape == y.shape and (np.array_equal(x, y) if x.dtype.kind not in "f" else np.array_equal(np.nan_to_num(x, nan=-7.0), np.nan_to_num(y, nan=-7.0)))
        out["primary_equals_reference"][k] = bool(eq)
    out["primary_all_equal"] = all(v is True for v in out["primary_equals_reference"].values())
    # 2. every rank space recomputed by counting
    constr = arrays["construction"].astype(str); kind = arrays["kind"].astype(str); cell = arrays["cell"].astype(int); chunk = arrays["chunk"].astype(int)
    members = [json.loads(str(s)) for s in arrays["members_json"]]
    sgm = np.asarray(arrays["sgm_log10"], np.float64)
    spaces = manifest["compatibility"].get("rank_spaces") or {"all": {"coordinates": keys, "blocks": manifest["compatibility"]["blocks"],
                                                                      "arrays": manifest["compatibility"]["arrays"]}}
    for label, decl in spaces.items():
        coords = [str(c) for c in decl["coordinates"]]
        idx = np.asarray([keys.index(c) for c in coords])
        scale = scale_all[idx]
        blocks = decl["blocks"] if "blocks" in decl else manifest["compatibility"]["blocks"]
        score_name = decl["arrays"]["score"] if "arrays" in decl else "compatibility_score"
        score = np.asarray(arrays[score_name], np.float64)
        n_pairs = 0; max_diff = 0.0; n_score_diff = 0; max_score_diff = 0.0; exact_pairs = 0
        for blk in blocks:
            gi = np.asarray(arrays[blk["arrays"]["group_index"]]).astype(int)
            R = np.asarray(arrays[blk["arrays"]["ranks"]], np.float64)
            pools = []
            for j in gi:
                rows = [clouds[(kind[j], int(c), int(w))] for c, w in members[j]]
                pools.append(np.concatenate(rows, axis=0)[:, idx])
            for jj, j in enumerate(gi):
                ref_sgm = sgm[j][idx]
                for ii, i in enumerate(gi):
                    if ii == jj:
                        assert np.isnan(R[ii, jj]); continue
                    r = rank_counting(pools[jj], ref_sgm, sgm[i][idx], scale)
                    d = abs(r - R[ii, jj]); max_diff = max(max_diff, d); n_pairs += 1; exact_pairs += int(d == 0.0)
            for ii, i in enumerate(gi):
                med = float(np.median(np.delete(R[ii], ii)))
                d = abs(med - score[i]); max_score_diff = max(max_score_diff, d); n_score_diff += int(d != 0.0)
        out["spaces"][label] = {"coordinates": coords, "n_blocks": len(blocks), "n_off_diagonal_pairs": n_pairs, "exact_pairs": exact_pairs,
                                "max_abs_rank_difference": max_diff, "n_scores_not_equal_to_row_median": n_score_diff, "max_abs_score_difference": max_score_diff,
                                "n_zero_scores": int(np.sum(score == 0.0)), "all_exact": bool(exact_pairs == n_pairs and n_score_diff == 0)}
    out["ok"] = bool(out["primary_all_equal"] and all(s["all_exact"] for s in out["spaces"].values()))
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "primary_equals_reference"}, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
