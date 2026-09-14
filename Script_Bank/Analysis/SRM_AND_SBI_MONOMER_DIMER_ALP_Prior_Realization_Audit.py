"""Prior-realization audit: do the GENERATED products realize what the prior and the declared inputs define?

Read-only over the PRODUCTS of one condition, split, and recording length: it writes only its report;
nothing is simulated or rendered. The
structure audit checks the generator's CODE against the specification; this audit checks the
generator's OUTPUT -- a trajectory tier's ``Theta_Set``, optionally its trajectories, and the DLI
stage's ``Labeling_Set`` -- against the prior box, the composition rule, the stationary mode law,
and the declared occupancies. Run it after any tier or DLI pass; it costs seconds.

Checks:
    P0 schema          Every Theta_Set of the tier carries the schema of the current biology table
                       (ordered keys, prior bounds, scales, condition); the loader refuses one that
                       does not, so a tier drawn under another table never reaches P1-P5. The
                       report records the stored generator, package version, and write dates.
    P1 prior box       Every stored theta (physical) maps back inside the estimator box; per row the
                       empirical marginal in estimator space against Uniform(lo, hi): the
                       Kolmogorov-Smirnov p-value and the share of draws per quarter of the box
                       (expected 25% each). A flagged row is evidence to investigate (sampler, conversion,
                       or an undersized tier), not a verdict.
    P2 composition     From the stored (N_R, r): every draw realizes at least one dimer; the realized
                       receptor fraction follows x_B = 2r/(1+2r) up to rounding; the median of the
                       requested log10 r lies within three standard errors of the box center (the
                       tolerance scales as 1/sqrt(n)), so the realized prior is even-handed between
                       mostly-monomer and mostly-dimer populations; the realized f_B median is reported.
    P3 trajectories    (--trajectories K) K trajectory files, evenly spaced over the tier: the frame-0
                       particle counts per species equal the composition realized from the stored
                       theta; the per-frame subunit total is constant (conservation); the pooled
                       frame-0 mode occupancies agree with the stationary law averaged over the K
                       thetas (the initial-mode law the generator seeds with).
    P4 labeling        The workflow's ``Labeling_Set``: n_subunits equals round(N_R) per simulation;
                       the recorded occupancy columns equal the condition's declared or derived value
                       (a differing value is reported as an override; whether it was intended is for
                       the reader); the pooled per-subunit visibility
                       n_labeled_subunits / n_subunits against a = p_occ P(dye >= 1); the pooled
                       visible fractions against a for monomers and 1 - (1 - a)^2 for dimers; the
                       both-labeled share (visible dimers whose two subunits are both labeled; for INLB the two-dye
                       share) against a / (2 - a); the emitters per
                       subunit against p_occ E[dye].
    P5 predictive      Visible hosts at frame 0 (monomers_visible_0 + dimers_visible_0) per simulation
                       against the per-recording first-2 s spot counts of the 60 deposited recordings
                       of the condition (Special_Analyses A9): quantiles side by side and the share of
                       recordings whose spot count lies inside the simulated range. DESCRIPTIVE: the
                       simulated count is before bleaching, detection, and field-of-view effects, so
                       it is judged only for gross mismatch (the empirical median inside the simulated
                       5-95% band).
    P6 visibility      The theoretical visibility chain corroborated through the DLI stage's OWN
                       labeling functions (draw_dye_counts, occupancy_per_subunit, labeling_summary)
                       on a synthetic frame-0 lineage of 10^5 monomers and 10^5 dimers per condition:
                       per-subunit visibility a, visible monomers, visible dimers, both-labeled share,
                       dark-partner share 2(1-a)/(2-a), emitters per subunit and per visible subunit,
                       and the dye-count law among visible subunits (zero-truncated); each within
                       four standard errors. Both conditions always, so the FAB/INLB visibility
                       ratio is checked against the DECLARED visibility ratio and shown beside the
                       deposited spot-count ratios (A9). Theory, code path, and -- when a
                       Labeling_Set is present -- products appear side by side in the report.

Usage (from the repo root; MACHINE_PROFILE set):
    PYTHONPATH=$PWD python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py \\
        --condition FAB --split train --total-time-seconds 2 [--workflow biology|detector] \\
        [--tasks all|0,1,2] [--trajectories 8]
    ... --selftest      # in-memory synthetic products drawn from the prior and labeled through the
                        # DLI stage's functions at the declared occupancies; verifies the checks
                        # themselves (P1, P2, P4, P5, P6) without any tier on disk
    ... --visibility    # P6 only, both conditions: theory vs code path (no tier needed)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from srm_and_sbi_monomer_dimer_alp import labeling as lab  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import parameterization as par  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import simulation_rds_support as rds  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.io import load_data, load_theta_set, read_theta_set_schema  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERIZATION, PARAMETERS, RunTiming  # noqa: E402

RDS = PARAMETERS.simulation.rds
KEYS = list(par.PARAMETER_KEYS)
SEED = 20260914
P2_SIGMAS = 3.0               # P2: |median log10 r - box center| <= P2_SIGMAS standard errors of a uniform's sample median
P2_MIN_DRAWS = 20             # P2: below this the symmetry verdict is not formed (reported only)
TOL_VISIBLE_FLOOR = 0.02      # P4: absolute tolerance floor on pooled fractions
TOL_P6_FLOOR = 0.002          # P6: absolute tolerance floor on Monte Carlo fractions (n = 3e5 subunits)
TOL_RATIO = 0.05              # P6: |realized FAB/INLB visibility ratio - declared| (products and code path)
N_P6_HOSTS = 100_000          # P6: monomers AND dimers per condition on the synthetic lineage
TOL_MODE = 0.03               # P3: absolute tolerance on pooled frame-0 mode occupancies
KS_ALPHA = 1e-3               # P1: a row is flagged when its KS p-value falls below this
A9_CSV = os.path.join(REPO_ROOT, "..", "Special_Analyses", "MET_NEXT_MODEL_DESIGN_PLAN", "results",
                      "A9_per_recording.csv")
# Fallback reference (A9, 2026-09-14): first-2 s localizations per frame, per recording, 60 per condition.
A9_FALLBACK = {"Fab": dict(min=17, q25=79, median=143, q75=218, max=307),
               "InlB": dict(min=72, q25=251, median=348, q75=448, max=794)}
A9_TOKEN = {"FAB": "Fab", "INLB": "InlB"}


def passed(ok) -> str:
    return "PASS" if ok else "FAIL"


def quantiles(x) -> dict:
    x = np.asarray(x, dtype=float)
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return dict(n=int(x.size), min=float(q[0]), q05=float(q[1]), q25=float(q[2]), median=float(q[3]),
                q75=float(q[4]), q95=float(q[5]), max=float(q[6]))


# ----------------------------------------------------------------------------------------------
# Checks on arrays (file-agnostic, so the selftest exercises the same code)
# ----------------------------------------------------------------------------------------------

def p1_prior_box(theta_physical: np.ndarray) -> dict:
    from scipy import stats
    flow = par.to_flow(theta_physical)
    low, high = np.array(par.theta_lower_bound()), np.array(par.theta_upper_bound())
    inside = np.all((flow >= low - 1e-9) & (flow <= high + 1e-9), axis=1)
    rows = {}
    for j, key in enumerate(KEYS):
        u = (flow[:, j] - low[j]) / (high[j] - low[j])
        p_value = float(stats.kstest(u, "uniform").pvalue)
        quarters = np.histogram(u, bins=[0, 0.25, 0.5, 0.75, 1.0 + 1e-12])[0] / max(u.size, 1)
        rows[key] = dict(ks_pvalue=p_value, quarter_shares=[round(float(q), 3) for q in quarters],
                         flagged=p_value < KS_ALPHA)
    return dict(n_draws=int(flow.shape[0]), all_inside_box=bool(inside.all()),
                n_outside=int((~inside).sum()), rows=rows,
                ok=bool(inside.all()) and not any(r["flagged"] for r in rows.values()))


def realized_compositions(theta_physical: np.ndarray):
    i_n, i_r = KEYS.index(RDS.stoichiometry.count_total_key), KEYS.index(RDS.stoichiometry.composition_ratio_key)
    return [par.realize_initial_composition(t[i_n], t[i_r]) for t in theta_physical]


def p2_composition(theta_physical: np.ndarray) -> dict:
    comps = realized_compositions(theta_physical)
    n_dimers = np.array([c.n_dimers for c in comps])
    f_b = np.array([c.complex_fraction for c in comps])
    x_real = np.array([c.fraction_realized for c in comps])
    x_req = np.array([c.fraction_requested for c in comps])
    n_tot = np.array([c.n_total for c in comps])
    rounding = np.abs(x_real - x_req) * n_tot / 2.0          # in units of dimers: must be <= 0.5 + cap effects
    at_least_one = bool((n_dimers >= 1).all())
    median_fb = float(np.median(f_b))
    # Symmetry of the REQUESTED composition prior: the sample median of the box-uniform log10 ratio against
    # the box center, within P2_SIGMAS standard errors of the sample median of a UNIFORM of width w,
    # SE = 1 / (2 f(m) sqrt(n)) = w / (2 sqrt(n)) (the normal-median formula 1.2533 sigma/sqrt(n) would be
    # 28% too tight here). The tolerance scales with n; a fixed tolerance in f_B would fail a correct sampler
    # at small n, since f_B changes by ~0.58 per dex near an even split.
    i_r = KEYS.index(RDS.stoichiometry.composition_ratio_key)
    lo, hi = par.theta_lower_bound()[i_r], par.theta_upper_bound()[i_r]
    u = np.log10(np.asarray(theta_physical, dtype=float)[:, i_r])
    median_u, center = float(np.median(u)), 0.5 * (lo + hi)
    tol_u = float(P2_SIGMAS * (hi - lo) / (2.0 * np.sqrt(max(u.size, 1))))
    symmetric = abs(median_u - center) <= tol_u if u.size >= P2_MIN_DRAWS else None
    return dict(n_draws=int(f_b.size), every_draw_has_a_dimer=at_least_one, min_dimers=int(n_dimers.min()),
                complex_fraction=quantiles(f_b), receptor_fraction=quantiles(x_real),
                median_complex_fraction=median_fb, median_log10_ratio=median_u, box_center_log10_ratio=center,
                tolerance_log10_ratio=tol_u, symmetric_about_center=symmetric,
                max_rounding_deviation_dimers=float(rounding.max()),
                ok=at_least_one and (symmetric is not False) and float(rounding.max()) <= 0.5 + 1e-9)


def p4_labeling(labeling_rows: np.ndarray, condition: str, theta_physical: np.ndarray | None) -> dict:
    col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
    L = np.asarray(labeling_rows, dtype=float)
    _, law = lab.resolve_labeling_law(condition)
    p_declared = par.occupancy_of(condition)
    occ_m, occ_d = L[:, col["occupancy_monomer"]], L[:, col["occupancy_dimer"]]
    recorded = np.isfinite(occ_m).all() and np.isfinite(occ_d).all()
    matches_declared = bool(recorded and np.allclose(occ_m, p_declared) and np.allclose(occ_d, p_declared))
    override = bool(recorded and not matches_declared)
    a_m = (np.nanmean(occ_m) if recorded else p_declared) * law.visible_probability
    a_d = (np.nanmean(occ_d) if recorded else p_declared) * law.visible_probability
    n_sub = max(L[:, col["n_subunits"]].sum(), 1)
    n_m, n_d = max(L[:, col["monomers_0"]].sum(), 1), max(L[:, col["dimers_0"]].sum(), 1)
    p_used = (n_m * (np.nanmean(occ_m) if recorded else p_declared)
              + 2 * n_d * (np.nanmean(occ_d) if recorded else p_declared)) / (n_m + 2 * n_d)
    a_sub = p_used * law.visible_probability
    sub = L[:, col["n_labeled_subunits"]].sum() / n_sub
    emit = L[:, col["n_dyes"]].sum() / n_sub
    exp_emit = p_used * law.mean
    sd_emit = np.sqrt(max(p_used * (law.variance + law.mean ** 2) - exp_emit ** 2, 0.0))
    tol_sub = max(TOL_VISIBLE_FLOOR, 3 * np.sqrt(a_sub * (1 - a_sub) / n_sub))
    tol_emit = max(TOL_VISIBLE_FLOOR, 3 * sd_emit / np.sqrt(n_sub))
    mono = L[:, col["monomers_visible_0"]].sum() / n_m
    dim = L[:, col["dimers_visible_0"]].sum() / n_d
    two = L[:, col["dimers_two_labeled_0"]].sum() / max(L[:, col["dimers_visible_0"]].sum(), 1)
    exp_mono, exp_dim, exp_two = a_m, 1 - (1 - a_d) ** 2, a_d / (2 - a_d)
    tol_m = max(TOL_VISIBLE_FLOOR, 3 * np.sqrt(exp_mono * (1 - exp_mono) / n_m))
    tol_d = max(TOL_VISIBLE_FLOOR, 3 * np.sqrt(exp_dim * (1 - exp_dim) / n_d))
    n_vis_d = max(L[:, col["dimers_visible_0"]].sum(), 1)
    tol_two = max(TOL_VISIBLE_FLOOR, 3 * np.sqrt(exp_two * (1 - exp_two) / n_vis_d))
    subunits_ok = None
    if theta_physical is not None:
        i_n = KEYS.index(RDS.stoichiometry.count_total_key)
        n_expected = np.array([max(1, round(float(t[i_n]))) for t in theta_physical])
        subunits_ok = bool(np.array_equal(L[:, col["n_subunits"]].astype(int), n_expected))
    return dict(n_simulations=int(L.shape[0]), occupancy_declared=p_declared,
                occupancy_source=par.occupancy_source_of(condition), occupancy_recorded=bool(recorded),
                occupancy_matches_declared=matches_declared, occupancy_override=override,
                occupancy_used=(float(np.nanmean(occ_m)), float(np.nanmean(occ_d))) if recorded else None,
                visible_per_subunit=(float(sub), float(a_sub), float(tol_sub)),
                visible_monomer=(float(mono), float(exp_mono), float(tol_m)),
                visible_dimer=(float(dim), float(exp_dim), float(tol_d)),
                both_labeled_share=(float(two), float(exp_two), float(tol_two)),
                emitters_per_subunit=(float(emit), float(exp_emit), float(tol_emit)),
                n_subunits_equals_round_N_R=subunits_ok,
                ok=bool(recorded and abs(sub - a_sub) <= tol_sub and abs(mono - exp_mono) <= tol_m
                        and abs(dim - exp_dim) <= tol_d and abs(two - exp_two) <= tol_two
                        and abs(emit - exp_emit) <= tol_emit and (subunits_ok is not False)))


# ----------------------------------------------------------------------------------------------
# P6: the theoretical visibility chain, corroborated through the DLI stage's own functions
# ----------------------------------------------------------------------------------------------

def dye_law_pmf(law: lab.LabelingLaw, k: np.ndarray) -> np.ndarray:
    """Exact ``P(dye = k)`` of a labeling law (the four families of ``labeling.LabelingLaw``)."""
    from scipy import stats
    k = np.asarray(k)
    if law.family == "bernoulli":
        return np.where(k == 0, 1 - law.mean, np.where(k == 1, law.mean, 0.0)).astype(float)
    if law.family == "poisson":
        return stats.poisson(law.mean).pmf(k)
    if law.family == "binomial":
        return stats.binom(int(law.shape), law.mean / law.shape).pmf(k)
    r, p = law._negative_binomial_parameters()
    return stats.nbinom(r, p).pmf(k)


def visibility_theory(condition: str) -> dict:
    """Closed-form visibility quantities from the condition's declared settings and dye law."""
    _, law = lab.resolve_labeling_law(condition)
    p = par.occupancy_of(condition)
    q = law.visible_probability
    a = p * q
    # dye-count law among VISIBLE subunits: the zero-truncated law, binned 1 / 2 / >= 3 (exact pmf)
    p1, p2 = (dye_law_pmf(law, np.array([1, 2])) / q).tolist()
    return dict(occupancy=float(p), occupancy_source=par.occupancy_source_of(condition), dye_law=law.describe(),
                p_dye_ge1=float(q), visible_per_subunit=float(a), visible_monomer=float(a),
                visible_dimer=float(1 - (1 - a) ** 2), both_labeled_share=float(a / (2 - a)),
                dark_partner_share=float(2 * (1 - a) / (2 - a)), emitters_per_subunit=float(p * law.mean),
                emitters_per_visible_subunit=float(law.mean / q),
                dyes_among_visible_1_2_3plus=(float(p1), float(p2), float(1 - p1 - p2)))


def labeled_lineage(condition: str, n_monomers: int, n_dimers: int, rng: np.random.Generator,
                    occupancy=None) -> tuple:
    """Label a synthetic frame-0 lineage through the DLI stage's functions.

    Returns ``(dye_counts, labeling_summary_row)``. Monomers are hosts of one subunit at rank 0,
    dimers hosts of two subunits at rank 1; the species map is the model's ('A', 'B').
    """
    _, law = lab.resolve_labeling_law(condition)
    occupancy = par.occupancy_of(condition) if occupancy is None else occupancy
    species = RDS.molecular_species_names
    host_index_0 = np.concatenate([np.arange(n_monomers), n_monomers + np.repeat(np.arange(n_dimers), 2)]).astype(int)
    host_rank_0 = np.concatenate([np.zeros(n_monomers, int), np.ones(2 * n_dimers, int)])
    initial_species = [species[0]] * n_monomers + [species[1]] * (2 * n_dimers)
    dye = lab.draw_dye_counts(law, host_index_0.size, rng,
                              occupancy=lab.occupancy_per_subunit(occupancy, initial_species))
    row = lab.labeling_summary(dye, host_index_0, host_rank_0, monomer_ranks=[0],
                               occupancy_by_species_values=lab.occupancy_by_species(occupancy, species))
    return dye, row


def p6_visibility(condition: str, n_hosts: int = N_P6_HOSTS, seed: int = SEED) -> dict:
    th = visibility_theory(condition)
    rng = np.random.default_rng([seed, 6])
    dye, row = labeled_lineage(condition, n_hosts, n_hosts, rng)
    col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
    n_sub, n_vis = row[col["n_subunits"]], max(row[col["n_labeled_subunits"]], 1)
    vis = dye[dye >= 1]
    realized = dict(
        visible_per_subunit=row[col["n_labeled_subunits"]] / n_sub,
        visible_monomer=row[col["monomers_visible_0"]] / n_hosts,
        visible_dimer=row[col["dimers_visible_0"]] / n_hosts,
        both_labeled_share=row[col["dimers_two_labeled_0"]] / max(row[col["dimers_visible_0"]], 1),
        dark_partner_share=1 - row[col["dimers_two_labeled_0"]] / max(row[col["dimers_visible_0"]], 1),
        emitters_per_subunit=row[col["n_dyes"]] / n_sub,
        emitters_per_visible_subunit=row[col["n_dyes"]] / n_vis,
        dyes_among_visible_1_2_3plus=(float(np.mean(vis == 1)), float(np.mean(vis == 2)), float(np.mean(vis >= 3))))
    # denominators for the standard errors of each fraction
    denom = dict(visible_per_subunit=n_sub, visible_monomer=n_hosts, visible_dimer=n_hosts,
                 both_labeled_share=max(row[col["dimers_visible_0"]], 1), dark_partner_share=max(row[col["dimers_visible_0"]], 1))
    comparisons, ok = {}, True
    for k, n in denom.items():
        e, r = th[k], realized[k]
        tol = max(TOL_P6_FLOOR, 4 * np.sqrt(e * (1 - e) / n))
        comparisons[k] = (float(r), float(e), float(tol)); ok &= abs(r - e) <= tol
    _, law = lab.resolve_labeling_law(condition)
    for k, n, sd in (("emitters_per_subunit", n_sub,
                      np.sqrt(max(th["occupancy"] * (law.variance + law.mean ** 2) - th["emitters_per_subunit"] ** 2, 0))),
                     ("emitters_per_visible_subunit", n_vis, np.sqrt(max(law.variance + law.mean ** 2 - law.mean ** 2 / th["p_dye_ge1"], 0)))):
        e, r = th[k], realized[k]
        tol = max(TOL_P6_FLOOR, 4 * sd / np.sqrt(n))
        comparisons[k] = (float(r), float(e), float(tol)); ok &= abs(r - e) <= tol
    for i, name in enumerate(("dyes_among_visible_1", "dyes_among_visible_2", "dyes_among_visible_3plus")):
        e, r = th["dyes_among_visible_1_2_3plus"][i], realized["dyes_among_visible_1_2_3plus"][i]
        tol = max(TOL_P6_FLOOR, 4 * np.sqrt(max(e * (1 - e), 0.0) / n_vis))
        comparisons[name] = (float(r), float(e), float(tol)); ok &= abs(r - e) <= tol
    # the occupancy the code path actually applied must be the declared one
    occ_ok = bool(np.isclose(row[col["occupancy_monomer"]], th["occupancy"]) and np.isclose(row[col["occupancy_dimer"]], th["occupancy"]))
    return dict(condition=condition, n_monomers=n_hosts, n_dimers=n_hosts, n_subunits=int(n_sub), theory=th,
                comparisons=comparisons, occupancy_recorded_equals_declared=occ_ok, ok=bool(ok and occ_ok))


def declared_visibility_ratio() -> tuple:
    """(numerator condition, anchor condition, declared ratio) from the condition settings."""
    for s in par.CONDITION_SETTINGS:
        if s.visibility_ratio is not None:
            return s.token, s.visibility_ratio_to, float(s.visibility_ratio)
    return None


def a9_spot_ratio() -> dict:
    """Deposited Fab/InlB spot-count ratios (descriptive companion to the visibility ratio)."""
    if not os.path.exists(A9_CSV):
        return dict(source="A9 fallback (2026-09-14)", first2s_median=143 / 348, whole_recording_mean=0.48,
                    per_area_first2s_means=0.43, per_area_first2s_medians=0.38)
    import pandas as pd
    d = pd.read_csv(A9_CSV)
    fab, inlb = d[d["cond"] == "Fab"], d[d["cond"] == "InlB"]
    return dict(source=A9_CSV, first2s_median=float(fab["spots_first2s"].median() / inlb["spots_first2s"].median()),
                whole_recording_mean=float(fab["spots_mean"].mean() / inlb["spots_mean"].mean()),
                per_area_first2s_means=float((fab["spots_first2s"] / fab["area_um2"]).mean()
                                             / (inlb["spots_first2s"] / inlb["area_um2"]).mean()),
                per_area_first2s_medians=float((fab["spots_first2s"] / fab["area_um2"]).median()
                                               / (inlb["spots_first2s"] / inlb["area_um2"]).median()))


def visibility_ratio_check(per_condition: dict, key: str) -> dict:
    """FAB/INLB per-subunit visibility ratio (from ``per_condition[cond][key]``) against the declared one."""
    num, anchor, declared = declared_visibility_ratio()
    if num not in per_condition or anchor not in per_condition:
        return None
    r = per_condition[num][key] / per_condition[anchor][key]
    return dict(numerator=num, anchor=anchor, realized=float(r), declared=declared, tolerance=TOL_RATIO,
                deposited_spot_ratios=a9_spot_ratio(), ok=bool(abs(r - declared) <= TOL_RATIO))


def a9_reference(condition: str) -> dict:
    token = A9_TOKEN[condition]
    if os.path.exists(A9_CSV):
        import pandas as pd
        d = pd.read_csv(A9_CSV)
        v = d.loc[d["cond"] == token, "spots_first2s"].to_numpy(dtype=float)
        if v.size:
            out = quantiles(v); out["source"] = A9_CSV; out["values"] = v.tolist(); return out
    q = dict(A9_FALLBACK[token]); q["source"] = "A9 fallback quantiles (2026-09-14)"; q["n"] = 60; return q


def p5_predictive(labeling_rows: np.ndarray, condition: str) -> dict:
    col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
    L = np.asarray(labeling_rows, dtype=float)
    visible = L[:, col["monomers_visible_0"]] + L[:, col["dimers_visible_0"]]
    sim = quantiles(visible)
    ref = a9_reference(condition)
    inside = None
    if "values" in ref:
        v = np.asarray(ref["values"]); inside = float(np.mean((v >= sim["min"]) & (v <= sim["max"])))
    gross_ok = sim["q05"] <= ref["median"] <= sim["q95"]
    return dict(simulated_visible_hosts_frame0=sim, deposited_first2s_spots=
                {k: ref[k] for k in ("n", "min", "q25", "median", "q75", "max", "source") if k in ref},
                share_of_recordings_inside_simulated_range=inside,
                empirical_median_inside_simulated_5_95=bool(gross_ok), ok=bool(gross_ok))


def p3_trajectories(paths, data_bank_root, timing, split, tasks, theta_by_task, k: int) -> dict:
    import readdy
    files = []
    for task in tasks:
        n_sims = theta_by_task[task].shape[0]
        for sim in range(n_sims):
            files.append((task, sim, paths.trajectory_path(task, sim, data_bank_root, timing.label, split)))
    files = [f for f in files if os.path.exists(f[2])]
    if not files:
        return dict(n_files=0, ok=None, note="no trajectory file found")
    pick = [files[i] for i in np.linspace(0, len(files) - 1, min(k, len(files))).astype(int)]
    mode_counts = np.zeros(len(RDS.mobility.modes)); n_particles = 0
    law_sum = np.zeros(len(RDS.mobility.modes))
    problems = []
    for task, sim, path in pick:
        theta = np.asarray(theta_by_task[task][sim], dtype=float)
        comp = rds.initial_composition_of(theta)
        tray = readdy.Trajectory(str(path))
        _, types, _, _ = tray.read_observable_particles()
        name_of_id = {v: k for k, v in tray.particle_types.items()}
        names0 = [name_of_id[int(t)] for t in types[0]]
        n_a = sum(1 for n in names0 if RDS.species_of_type[n] == RDS.stoichiometry.monomer.name)
        n_b = sum(1 for n in names0 if RDS.species_of_type[n] == RDS.stoichiometry.dimer.name)
        if (n_a, n_b) != (comp.n_monomers, comp.n_dimers):
            problems.append(f"task {task} sim {sim}: frame-0 counts ({n_a}, {n_b}) != realized ({comp.n_monomers}, {comp.n_dimers})")
        totals = [sum(RDS.subunit_counts_per_type[RDS.particle_type_names.index(name_of_id[int(t)])] for t in frame)
                  for frame in types]
        if len(set(totals)) != 1 or totals[0] != comp.n_total:
            problems.append(f"task {task} sim {sim}: subunit total not conserved ({min(totals)}..{max(totals)} vs {comp.n_total})")
        for n in names0:
            mode_counts[RDS.mobility.mode_index(RDS.mode_of_type[n])] += 1
        n_particles += len(names0)
        law_sum += rds.stationary_mode_law(theta) * len(names0)
    pooled = mode_counts / max(n_particles, 1)
    law = law_sum / max(n_particles, 1)
    dev = float(np.max(np.abs(pooled - law)))
    return dict(n_files=len(pick), frame0_counts_match=not any("counts" in p for p in problems),
                subunit_total_conserved=not any("conserved" in p for p in problems),
                pooled_frame0_mode_occupancy=[round(float(v), 4) for v in pooled],
                stationary_law_weighted=[round(float(v), 4) for v in law], max_abs_deviation=dev,
                problems=problems, ok=not problems and dev <= TOL_MODE)


# ----------------------------------------------------------------------------------------------
# Products on disk
# ----------------------------------------------------------------------------------------------

def discover_tasks(paths, data_bank_root, timing_label, split) -> list:
    pattern = str(paths.theta_set_path("*", data_bank_root, timing_label, True, split))
    found = sorted(glob.glob(pattern))
    tasks = []
    for f in found:
        stem = os.path.basename(f)
        token = stem.split("_TASK_")[1].split("_")[0]
        tasks.append(int(token))
    return sorted(set(tasks))


def labeling_rows_of(condition: str, args) -> np.ndarray | None:
    """Stacked Labeling_Set rows of a condition at the run's timing/split/workflow, or None when absent."""
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    base = PARAMETERS.paths.with_condition(condition)
    paths = det.detector_paths(PARAMETERS.paths).with_condition(condition) if args.workflow == "detector" else base
    root = PARAMETERS.machine.root_for(args.split)          # TRAIN/TEST may live on the scratch tier
    timing = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing)
    rows = []
    for t in discover_tasks(base, root, timing.label, args.split.upper()):
        p = paths.record_set_path("Labeling_Set", t, root, timing.label, True, args.split.upper())
        if os.path.exists(p):
            rows.append(np.asarray(load_data(p)[:], dtype=float))
    if not rows or rows[0].shape[1] != len(lab.LABELING_SET_COLUMNS):
        return None
    return np.vstack(rows)


def load_products(args):
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    condition = args.condition
    base = PARAMETERS.paths.with_condition(condition)
    paths = det.detector_paths(PARAMETERS.paths).with_condition(condition) if args.workflow == "detector" else base
    root = PARAMETERS.machine.root_for(args.split)          # products: TRAIN/TEST may live on the scratch tier
    timing = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing)
    split = args.split.upper()
    tasks = discover_tasks(base, root, timing.label, split)
    if args.tasks != "all":
        wanted = {int(t) for t in args.tasks.split(",")}
        tasks = [t for t in tasks if t in wanted]
    if not tasks:
        sys.exit(f"no Theta_Set found for {condition} {split} {timing.label} under {root}")
    theta_by_task, schemas = {}, {}
    for t in tasks:
        p = base.theta_set_path(t, root, timing.label, True, split)
        theta_by_task[t] = np.asarray(load_theta_set(p, par.PARAMETERIZATION, condition=condition)[:], dtype=float)
        schemas[t] = read_theta_set_schema(p)
    theta = np.vstack([theta_by_task[t] for t in tasks])
    p0 = dict(n_theta_sets=len(tasks), schema_checked=True,
              generators=sorted({sc.get("generator") for sc in schemas.values()}),
              package_versions=sorted({sc.get("package_version") for sc in schemas.values()}),
              schema_versions=sorted({sc.get("schema_version") for sc in schemas.values()}),
              written_utc_range=(min(sc.get("written_utc", "") for sc in schemas.values()),
                                 max(sc.get("written_utc", "") for sc in schemas.values())),
              ok=True)
    labeling = []
    for t in tasks:
        p = paths.record_set_path("Labeling_Set", t, root, timing.label, True, split)
        if os.path.exists(p):
            labeling.append(np.asarray(load_data(p)[:], dtype=float))
    labeling = np.vstack(labeling) if labeling else None
    if labeling is not None and labeling.shape[1] != len(lab.LABELING_SET_COLUMNS):
        sys.exit(f"Labeling_Set has {labeling.shape[1]} columns; this audit expects the "
                 f"{len(lab.LABELING_SET_COLUMNS)}-column record of 0.1.4 (re-render the DLI pass).")
    out_dir = (PARAMETERS.machine.data_bank_root / paths.posit_subdir      # the report stays on the permanent tier
               / f"{paths.project_alias}_{timing.label}_Prior_Realization_Audit_{split}")
    return dict(condition=condition, split=split, timing=timing, tasks=tasks, base=base, paths=paths, root=root,
                theta=theta, theta_by_task=theta_by_task, labeling=labeling, out_dir=out_dir, p0=p0)


def synthetic_products(condition: str, n: int = 2000, seed: int = SEED) -> dict:
    """In-memory products drawn from the prior and the declared occupancy (the selftest)."""
    rng = np.random.default_rng(seed)
    low, high = np.array(par.theta_lower_bound()), np.array(par.theta_upper_bound())
    theta = par.to_physical(rng.uniform(low, high, size=(n, len(low))))
    # Each composition is labeled through the DLI stage's own functions (labeled_lineage), so the
    # synthetic Labeling_Set is produced by the code path the products come from, not by a re-implementation.
    rows = [labeled_lineage(condition, c.n_monomers, c.n_dimers, rng)[1] for c in realized_compositions(theta)]
    return dict(condition=condition, theta=theta, labeling=np.array(rows, dtype=float))


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

def write_report(out_dir, header: dict, res: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    L = ["# Prior-realization audit", "", *[f"- {k}: {v}" for k, v in header.items()], "",
         "| check | verdict | detail |", "|---|---|---|"]
    p0 = res.get("p0")
    if p0 is not None:
        L.append(f"| P0 schema | {passed(p0['ok'])} | {p0['n_theta_sets']} Theta_Set(s), every one carrying the current table's schema "
                 f"(keys, bounds, scales, condition); generator {p0['generators']}, package {p0['package_versions']}, "
                 f"schema version {p0['schema_versions']}, written {p0['written_utc_range'][0]} .. {p0['written_utc_range'][1]} |")
    p1 = res.get("p1")
    if p1 is not None:
        flagged = [k for k, r in p1["rows"].items() if r["flagged"]]
        L.append(f"| P1 prior box | {passed(p1['ok'])} | {p1['n_draws']} draws, {p1['n_outside']} outside the box; "
                 f"rows with KS p < {KS_ALPHA}: {flagged or 'none'} |")
    p2 = res.get("p2")
    if p2 is not None:
        L.append(f"| P2 composition | {passed(p2['ok'])} | every draw has a dimer: {p2['every_draw_has_a_dimer']} (min {p2['min_dimers']}); "
                 f"requested log10 r median {p2['median_log10_ratio']:.3f} vs box center {p2['box_center_log10_ratio']:.1f} "
                 f"(tol {p2['tolerance_log10_ratio']:.3f} = {P2_SIGMAS:g} SE at n = {p2['n_draws']}; symmetric: {p2['symmetric_about_center']}); "
                 f"realized f_B median {p2['median_complex_fraction']:.3f}; "
                 f"f_B quartiles {p2['complex_fraction']['q25']:.3f} / {p2['complex_fraction']['q75']:.3f}; "
                 f"max rounding deviation {p2['max_rounding_deviation_dimers']:.2f} dimers |")
    if res.get("p3") is not None:
        p3 = res["p3"]
        L.append(f"| P3 trajectories | {passed(p3['ok']) if p3['ok'] is not None else 'n/a'} | {p3.get('n_files', 0)} files; frame-0 counts match: "
                 f"{p3.get('frame0_counts_match')}; subunit total conserved: {p3.get('subunit_total_conserved')}; pooled frame-0 mode "
                 f"occupancy {p3.get('pooled_frame0_mode_occupancy')} vs law {p3.get('stationary_law_weighted')} "
                 f"(max dev {p3.get('max_abs_deviation', float('nan')):.3f}, tol {TOL_MODE}) |")
    if res.get("p4") is not None:
        p4 = res["p4"]
        L.append(f"| P4 labeling | {passed(p4['ok'])} | {p4['n_simulations']} simulations; occupancy declared {p4['occupancy_declared']:.3f} "
                 f"({p4['occupancy_source']}), recorded: {p4['occupancy_recorded']}, matches: {p4['occupancy_matches_declared']}, "
                 f"override: {p4['occupancy_override']}; visible per subunit {p4['visible_per_subunit'][0]:.4f} vs a = "
                 f"{p4['visible_per_subunit'][1]:.4f} (tol {p4['visible_per_subunit'][2]:.4f}); visible monomers "
                 f"{p4['visible_monomer'][0]:.3f} vs {p4['visible_monomer'][1]:.3f} (tol {p4['visible_monomer'][2]:.3f}); "
                 f"visible dimers {p4['visible_dimer'][0]:.3f} vs {p4['visible_dimer'][1]:.3f}; "
                 f"both-labeled share {p4['both_labeled_share'][0]:.3f} vs {p4['both_labeled_share'][1]:.3f}; emitters per subunit "
                 f"{p4['emitters_per_subunit'][0]:.4f} vs {p4['emitters_per_subunit'][1]:.4f}; n_subunits = round(N_R): "
                 f"{p4['n_subunits_equals_round_N_R']} |")
        p5 = res["p5"]
        s, r = p5["simulated_visible_hosts_frame0"], p5["deposited_first2s_spots"]
        L.append(f"| P5 predictive visible counts | {passed(p5['ok'])} (descriptive) | simulated visible hosts at frame 0: "
                 f"min {s['min']:.0f}, q25 {s['q25']:.0f}, median {s['median']:.0f}, q75 {s['q75']:.0f}, max {s['max']:.0f}; "
                 f"deposited first-2 s spots ({r['n']} recordings): min {r['min']:.0f}, q25 {r['q25']:.0f}, median {r['median']:.0f}, "
                 f"q75 {r['q75']:.0f}, max {r['max']:.0f}; share of recordings inside the simulated range: "
                 f"{p5['share_of_recordings_inside_simulated_range']}; empirical median inside simulated 5-95%: "
                 f"{p5['empirical_median_inside_simulated_5_95']} |")
    if res.get("p6") is not None:
        p6 = res["p6"]
        L.append(f"| P6 visibility (code path) | {passed(p6['ok'])} | {p6['n_monomers']} monomers + {p6['n_dimers']} dimers labeled through "
                 f"draw_dye_counts / occupancy_per_subunit / labeling_summary at occupancy {p6['theory']['occupancy']:.4f} "
                 f"({p6['theory']['occupancy_source']}), {p6['theory']['dye_law']}; recorded occupancy equals declared: "
                 f"{p6['occupancy_recorded_equals_declared']}; all quantities within tolerance: {p6['ok']} (table below) |")
    if res.get("ratio") is not None:
        rt = res["ratio"]; dep = rt["deposited_spot_ratios"]
        L.append(f"| P6 visibility ratio {rt['numerator']}/{rt['anchor']} | {passed(rt['ok'])} | realized {rt['realized']:.3f} vs declared "
                 f"{rt['declared']:.3f} (tol {rt['tolerance']}); deposited spot-count ratios (descriptive): "
                 f"{', '.join(f'{k} {v:.2f}' for k, v in dep.items() if k != 'source')} |")
    if res.get("p6") is not None:
        p6, p4 = res["p6"], res.get("p4")
        L += ["", "Visibility chain: theory from the declared settings, the DLI code path on the synthetic lineage, and "
              "(when present) the generated products. Tolerances: four standard errors (code path), max(0.02, three "
              "standard errors) (products).", "",
              "| quantity | theory | code path (realized, tol) | products (realized, tol) |", "|---|---|---|---|"]
        prod_keys = dict(visible_per_subunit="visible_per_subunit", visible_monomer="visible_monomer",
                         visible_dimer="visible_dimer", both_labeled_share="both_labeled_share", emitters_per_subunit="emitters_per_subunit")
        for k, (r, e, tol) in p6["comparisons"].items():
            prod = "" if (p4 is None or k not in prod_keys) else f"{p4[prod_keys[k]][0]:.4f} ({p4[prod_keys[k]][2]:.4f})"
            L.append(f"| {k} | {e:.4f} | {r:.4f} ({tol:.4f}) | {prod} |")
    if p1 is not None:
        L += ["", "P1 per-row quarter shares (expected 0.25 each) and KS p-values:", "", "| row | quarters | KS p |", "|---|---|---|"]
        for k, r in p1["rows"].items():
            L.append(f"| {k} | {r['quarter_shares']} | {r['ks_pvalue']:.3g} |")
    L.append("")
    report = os.path.join(out_dir, "Prior_Realization_Audit.md")
    with open(report, "w") as h:
        h.write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, "prior_realization_summary.json"), "w") as h:
        json.dump(dict(header=header, results=res), h, indent=2, default=str)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--condition", choices=list(lab.LABELING_CONDITIONS))
    ap.add_argument("--split", default="train", choices=["train", "test", "eval"])
    ap.add_argument("--total-time-seconds", type=float)
    ap.add_argument("--workflow", default="biology", choices=["biology", "detector"],
                    help="whose Labeling_Set to read (the tier's Theta_Set is shared)")
    ap.add_argument("--tasks", default="all", help="'all' or a comma-separated list of task indices")
    ap.add_argument("--trajectories", type=int, default=0, help="number of trajectory files to open for P3 (0 = skip)")
    ap.add_argument("--selftest", action="store_true", help="synthetic in-memory products; no tier needed")
    ap.add_argument("--visibility", action="store_true", help="P6 only, both conditions: theory vs the DLI code path; no tier needed")
    ap.add_argument("--out-dir", default=None, help="selftest only: where to write the report (default: repo scratch is NOT used; "
                                                  "the Posit dir of the biology alias)")
    args = ap.parse_args()

    if args.selftest or args.visibility:
        mode = "selftest" if args.selftest else "visibility"
        results = {}
        for condition in lab.LABELING_CONDITIONS:
            res = dict(p6=p6_visibility(condition))
            if args.selftest:
                prod = synthetic_products(condition)
                res.update(p1=p1_prior_box(prod["theta"]), p2=p2_composition(prod["theta"]),
                           p4=p4_labeling(prod["labeling"], condition, prod["theta"]), p5=p5_predictive(prod["labeling"], condition))
            results[condition] = res
            for k, v in res.items():
                print(f"  [{passed(v['ok'])}] {condition} {k}")
        ratio = visibility_ratio_check({c: dict(v=r["p6"]["comparisons"]["visible_per_subunit"][0]) for c, r in results.items()}, "v")
        print(f"  [{passed(ratio['ok'])}] visibility ratio {ratio['numerator']}/{ratio['anchor']}: realized {ratio['realized']:.3f} "
              f"vs declared {ratio['declared']:.3f}")
        out_dir = args.out_dir or str(PARAMETERS.machine.data_bank_root / PARAMETERS.paths.posit_subdir
                                      / f"{PARAMETERS.paths.project_alias}_Prior_Realization_Audit_{mode.upper()}")
        for condition, res in results.items():
            res = dict(res, ratio=ratio)
            header = dict(mode=("selftest (synthetic products from the prior, labeled through the DLI code path at the declared occupancy)"
                                if args.selftest else "visibility (theory vs the DLI code path; no products)"), condition=condition)
            if "p1" in res:
                header["n"] = int(res["p1"]["n_draws"])
            print("report:", write_report(os.path.join(out_dir, condition), header, res))
        ok = ratio["ok"] and all(v["ok"] for res in results.values() for v in res.values())
        print("OVERALL:", passed(ok)); sys.exit(0 if ok else 1)

    if args.condition is None or args.total_time_seconds is None:
        ap.error("--condition and --total-time-seconds are required (or use --selftest)")
    prod = load_products(args)
    res = dict(p0=prod["p0"], p1=p1_prior_box(prod["theta"]), p2=p2_composition(prod["theta"]))
    if args.trajectories > 0:
        res["p3"] = p3_trajectories(prod["base"], prod["root"], prod["timing"], prod["split"], prod["tasks"],
                                    prod["theta_by_task"], args.trajectories)
    if prod["labeling"] is not None:
        res["p4"] = p4_labeling(prod["labeling"], prod["condition"], prod["theta"] if prod["labeling"].shape[0] == prod["theta"].shape[0] else None)
        res["p5"] = p5_predictive(prod["labeling"], prod["condition"])
    res["p6"] = p6_visibility(prod["condition"])
    # cross-condition visibility ratio on PRODUCTS: needs the other condition's Labeling_Set at the same timing/split/workflow
    if prod["labeling"] is not None:
        other = [c for c in lab.LABELING_CONDITIONS if c != prod["condition"]][0]
        other_rows = labeling_rows_of(other, args)
        if other_rows is not None:
            col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
            per = {c: dict(v=R[:, col["n_labeled_subunits"]].sum() / max(R[:, col["n_subunits"]].sum(), 1))
                   for c, R in ((prod["condition"], prod["labeling"]), (other, other_rows))}
            res["ratio"] = visibility_ratio_check(per, "v")
    for k, v in res.items():
        print(f"  [{passed(v['ok']) if v.get('ok') is not None else 'n/a'}] {k}")
    header = dict(condition=prod["condition"], split=prod["split"], timing=prod["timing"].label, workflow=args.workflow,
                  tasks=prod["tasks"], n_theta=int(prod["theta"].shape[0]),
                  labeling_set=("present" if prod["labeling"] is not None else "absent (run the DLI pass for P4/P5)"))
    report = write_report(prod["out_dir"], header, res)
    ok = all(v["ok"] for v in res.values() if v.get("ok") is not None)
    print("report:", report); print("OVERALL:", passed(ok)); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
