"""Verify the production Nuisance_DLI artifacts (canonical, no tag) against the reviewed proposal PROPOSALCW4 and load
them through the biology gate. Read-only.

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python Script_Bank/Analysis/Evidence/Nuisance_DLI_Production_2026-11-10/verify_production.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli

HERE = Path(__file__).resolve().parent
POSIT = Path("/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit")
REVIEW = Path("/home/mars-fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/Special_Analyses/"
              "NUISANCE_DLI_PRODUCTION_PROPOSAL_2026-11-09/WEIGHTING_REVISION_2026-11-10")
EXPECT = {"FAB": {"effective": 427.30, "zeros": 1, "lambda": 5.1}, "INLB": {"effective": 317.97, "zeros": 64, "lambda": 4.7}}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


out = {}
for cond in ("FAB", "INLB"):
    alias = f"SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{cond}"
    nu = ndli.require_nuisance_dli(POSIT, alias, "2S_50FPS")                      # the biology gate, canonical artifact (no tag)
    art = ndli.artifact_path(POSIT, alias, "2S_50FPS"); spec = ndli.spec_path(POSIT, alias, "2S_50FPS")
    ref = ndli.NuisanceDLI.load(REVIEW / cond / f"{alias}_2S_50FPS_PROPOSALCW4_Nuisance_DLI.npz")
    sel = nu.selection; comp = sel["components"][0]; keys = list(nu.parameter_keys)
    same_members = [(m["cell"], m["chunk"], m["stored_row"], m["score_used"], m["probability"]) for m in comp["members"]] == \
                   [(m["cell"], m["chunk"], m["stored_row"], m["score_used"], m["probability"]) for m in ref.selection["components"][0]["members"]]
    d = nu.sample(20000, seed=20261110)
    stored = {tuple(r): k for k, r in enumerate(nu.samples)}
    idx = np.asarray([stored[tuple(x)] for x in d])
    j = keys.index("lambda_rate"); b = keys.index("prob_photo_bleach")
    r = {"artifact": str(art), "artifact_sha256": sha(art), "spec": str(spec), "spec_sha256": sha(spec), "tag": None,
         "choice": nu.posterior_sample_pool_choice, "stored_rows": int(nu.samples.shape[0]),
         "samples_identical_to_reviewed_proposal": bool(np.array_equal(nu.samples, ref.samples)),
         "probabilities_identical_to_reviewed_proposal": bool(np.array_equal(nu.probabilities, ref.probabilities)),
         "members_identical_to_reviewed_proposal": bool(same_members),
         "archive_sha256": sel["archive"]["sha256"], "archive_identical_to_reviewed_proposal": sel["archive"]["sha256"] == ref.selection["archive"]["sha256"],
         "overrides": {k: v["value_physical"] for k, v in sel["overrides"].items()},
         "overrides_identical_to_reviewed_proposal": {k: (v["value_physical"], v["reason"]) for k, v in sel["overrides"].items()}
                                                     == {k: (v["value_physical"], v["reason"]) for k, v in ref.selection["overrides"].items()},
         "rank_scope": comp["rank_scope"], "rank_space": comp["rank_space"], "rank_coordinates": comp["rank_coordinates"],
         "references_per_member": comp["normalizers"]["references_per_member"], "normalization_rule": comp["normalizers"]["rule"],
         "effective_rows": sel["effective_rows"], "zero_probability_rows": sel["n_zero_probability_rows"],
         "probability_sum": float(nu.probabilities.sum()), "outside_prior_mass": sel["outside_prior_mass"],
         "outside_prior_by_key": sel["outside_prior_by_key"],
         "bleaching_in_every_row": float(10 ** nu.samples[0, b]) if np.unique(nu.samples[:, b]).size == 1 else None,
         "flicker_in_every_row": float(10 ** nu.samples[0, j]) if np.unique(nu.samples[:, j]).size == 1 else None,
         "sampled_draws": int(d.shape[0]), "draws_are_stored_rows": True, "zero_rows_drawn": int(np.sum(nu.probabilities[idx] == 0)),
         "max_abs_frequency_minus_probability": float(np.max(np.abs(np.bincount(idx, minlength=nu.samples.shape[0]) / d.shape[0] - nu.probabilities)))}
    e = EXPECT[cond]
    assert r["samples_identical_to_reviewed_proposal"] and r["probabilities_identical_to_reviewed_proposal"] and r["members_identical_to_reviewed_proposal"]
    assert r["archive_identical_to_reviewed_proposal"] and r["overrides_identical_to_reviewed_proposal"] and r["rank_scope"] == "construction"
    assert r["references_per_member"] == [599] and abs(r["effective_rows"] - e["effective"]) < 0.005 and r["zero_probability_rows"] == e["zeros"]
    assert abs(r["bleaching_in_every_row"] - 0.05) < 1e-12 and abs(r["flicker_in_every_row"] - e["lambda"]) < 1e-12 and r["zero_rows_drawn"] == 0
    out[cond] = r
out["all_ok"] = True
(HERE / "verification_production.json").write_text(json.dumps(out, indent=1) + "\n")
for c in ("FAB", "INLB"):
    r = out[c]
    print(f"{c}: sha256 {r['artifact_sha256']}; identical to the reviewed proposal (vectors, probabilities, members, archive, overrides); "
          f"effective {r['effective_rows']:.2f}, zeros {r['zero_probability_rows']}, bleaching {r['bleaching_in_every_row']:.3g}, flicker "
          f"{r['flicker_in_every_row']:.3g}, outside prior {100 * r['outside_prior_mass']:.1f} %")
print("ALL OK")
