"""Technical validation of the two SEL selection_user Nuisance_DLI artifacts (PC Posit)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from srm_and_sbi_monomer_dimer_alp.detector_nuisance_dli import require_nuisance_dli, artifact_path, spec_path
try:
    from srm_and_sbi_monomer_dimer_alp.detector_parameterization import DETECTOR_PARAMETER_KEYS as KEYS
except Exception:
    from srm_and_sbi_monomer_dimer_alp.posterior_predictive_video_runner import DETECTOR_PARAMETER_KEYS as KEYS
POSIT = Path("/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit")
APPROVED = {
  "FAB":  {"mu_r": 1.734, "sigma_r": 0.177, "mu_pc": 105.0, "sigma_pc": 0.5848, "prob_photo_bleach": 0.05, "lambda_rate": 8.0},
  "INLB": {"mu_r": 2.256, "sigma_r": 0.4029, "mu_pc": 152.9, "sigma_pc": 0.8588, "prob_photo_bleach": 0.05, "lambda_rate": 2.0},
}
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
fails = 0
def check(name, ok, detail=""):
    global fails
    fails += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
print("expected production key order:", list(KEYS))
for cond in ("FAB", "INLB"):
    alias = f"SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{cond}"
    ap = artifact_path(POSIT, alias, "2S_50FPS", "SEL"); sp = spec_path(POSIT, alias, "2S_50FPS", "SEL")
    print(f"\n=== {cond}  artifact {ap.name}\n    spec     {sp.name}")
    nu = require_nuisance_dli(POSIT, alias, "2S_50FPS", nuisance_tag="SEL")
    m = nu.manifest()
    check("choice selection_user", m["posterior_sample_pool_choice"] == "selection_user", m["posterior_sample_pool_choice"])
    check("parameter_keys == production order", list(nu.parameter_keys) == list(KEYS), str(list(nu.parameter_keys)))
    check("exactly one stored vector", nu.samples.shape == (1, 6), str(nu.samples.shape))
    fixed = nu.fixed_vector_log10
    phys = {k: float(10 ** v) for k, v in zip(nu.parameter_keys, fixed)}
    for k in KEYS:
        exp = APPROVED[cond][k]
        check(f"{k} physical {phys[k]:.6g} == approved {exp}", abs(phys[k] - exp) <= 1e-9 * max(1.0, abs(exp)), f"log10 {fixed[list(KEYS).index(k)]:+.4f}")
    draws = nu.sample(1000)
    check("1000 draws identical to the fixed vector", draws.shape == (1000, 6) and np.all(draws == fixed[None, :]))
    draws2 = nu.sample(7)
    check("repeated sampling identical", np.all(draws2 == fixed[None, :]))
    sel = nu.selection
    check("selection record present", isinstance(sel, dict) and len(sel) > 0, str(sorted(sel.keys())) if isinstance(sel, dict) else str(type(sel)))
    # sources / limitations per key (selection record field 'sources')
    per = sel.get("sources", {})
    if cond == "FAB":
        print("    sources[mu_r] =", json.dumps(per.get("mu_r"))[:160])
    for k in KEYS:
        rec = per.get(k, {})
        if isinstance(rec, dict):
            ok = bool(rec.get("source")) and bool(rec.get("limitation"))
        else:
            ok = bool(rec)
        check(f"{k}: non-empty source + limitation in the record", ok)
    vp = sel.get("values_physical", {})
    check("values_physical in the record == approved table", all(abs(float(vp[k]) - APPROVED[cond][k]) <= 1e-9 for k in KEYS), str(vp))
    out = sel.get("outside_prior")
    just = sel.get("outside_prior_justification", "")
    if cond == "FAB":
        check("FAB: nothing outside the prior box", not out, str(out))
        check("FAB: pool_mode bounded", m["pool_mode"] == "bounded", m["pool_mode"])
    else:
        check("INLB: mu_r acknowledged outside the prior box", list(out) == ["mu_r"], str(out))
        check("INLB: acknowledgement text survives", "extrapolation" in (just or "") and "2.256" in (just or ""), (just or "")[:80])
        check("INLB: pool_mode unrestricted (records the acknowledged extrapolation)", m["pool_mode"] == "unrestricted", m["pool_mode"])
        low, high = nu.prior_low, nu.prior_high
        inside = (fixed >= low) & (fixed <= high)
        check("INLB: only mu_r outside the box", [k for k, i in zip(KEYS, inside) if not i] == ["mu_r"])
    ident = nu.identity(ap)
    check("identity() fixed=True n_samples=1", ident["fixed"] is True and ident["n_samples"] == 1)
    print(f"    SHA-256 spec     {sha(sp)}")
    print(f"    SHA-256 artifact {sha(ap)}")
    print(f"    sizes: spec {sp.stat().st_size} B, artifact {ap.stat().st_size} B")
print(f"\nRESULT: {'ALL PASS' if fails == 0 else str(fails) + ' FAIL'}")
sys.exit(1 if fails else 0)
