"""Prior-realization audit: do the GENERATED products realize what the prior and the declared inputs define?

Read-only over the PRODUCTS of one condition, split, and recording length: it writes only its report;
nothing is simulated or rendered. The
structure audit checks the generator's CODE against the specification; this audit checks the
generator's OUTPUT -- a trajectory tier's ``Theta_Set``, optionally its trajectories, and the DLI
stage's ``Labeling_Set`` -- against the prior box, the composition rule, the stationary mode law,
and the declared occupancies. Run it after any tier or DLI pass; it costs seconds.

Checks:
    P0 schema          Every Theta_Set of the tier carries the schema of the condition's biology table
                       (ordered keys, prior bounds, scales, condition); the loader refuses one that
                       does not, so a tier drawn under another table never reaches P1-P5. The
                       report records the stored generator, package version, and write dates.
    P1 prior box       Every stored theta (physical) maps back inside the condition's estimator box
                       (eleven rows under MET-FAB, twelve under MET-INLB); per row, in the ESTIMATOR
                       coordinate (log10 for the log rows, the value for the two linear fraction
                       rows), the empirical marginal against Uniform(lo, hi): the Kolmogorov-Smirnov
                       p-value and the share of draws per quarter of the box (expected 25% each). A
                       flagged row is evidence to investigate (sampler, conversion, or an undersized
                       tier), not a verdict.
    P2 composition     The window-start composition realized from the stored (N_total, f) and, under
                       MET-INLB, R_active through the canonical helper (``population_from_theta``):
                       the represented receptor count N_R lies within 1.5 subunits of its expectation
                       (three roundings); N_R = A_ref + B1 + 2 B2_ref = A_start + B1 + 2 B2_start
                       (INLB) or A + 2 D_sim (FAB) EXACTLY, with A_start = A_ref - 2c and
                       B2_start = B2_ref + c; B1 and the reference counts are unchanged by the
                       allocation; 0 <= c <= capacity with c = round(R_active x capacity); c = 0 for
                       every MET-FAB draw; M_start + 2 D_start = N_total; the median of each LINEAR
                       row lies at 0.5 within 3 / (2 sqrt n) (a uniform's sample median); reported:
                       quantiles of f, the capacity, c, the share of draws without a pair (capacity 0)
                       and with an allocation, the derived starting fraction f_start and the active
                       share c / D_start.
    P3 trajectories    (--trajectories K) K trajectory files, evenly spaced over the tier: the frame-0
                       particle counts per stoichiometric class (and per probe class B1 / B2 under
                       INLB) equal the STARTING counts the helper realizes from the stored theta (the
                       allocation included; the files with c > 0 are counted); the per-frame
                       represented subunit total is constant (conservation through fusion, fission,
                       B1 -> A and mode switches); the pooled frame-0 mode occupancies agree with the
                       stationary law averaged over the K thetas.
    P4 labeling        The workflow's ``Labeling_Set``: n_subunits equals the realized N_R per
                       simulation and the frame-0 class counts equal the helper's (INLB: A_start, B1,
                       B2_start; FAB: A and the dimers at reference, whose classes are drawn at
                       labeling); the recorded probe-probability columns equal the model's rule (1 for
                       monomer subunits; (1 + s_2)/2 under FAB and 1 under INLB for dimer subunits; a
                       differing value is reported as an override); the probe classes (one- and
                       two-probe dimers) split the dimers, with the FAB two-probe share at the
                       declared s_2 = p/(2-p) and the INLB classes exhaustive; ``two_probe_share`` is
                       the declared share of the REFERENCE dimers (the share Fab labeling draws with;
                       under INLB the starting B2 share also carries the allocation and is read from
                       the counts); the pooled visible fractions against q = P(dye >= 1) for monomers
                       and the class mix for dimers; the two-labeled share of visible dimers against
                       the two-probe dimers' q^2; the emitters per subunit against E[dye] x the bound
                       share.
    P5 predictive      Visible hosts at frame 0 (monomers_visible_0 + dimers_visible_0) per simulation
                       against the per-recording first-2 s spot counts of the 60 deposited recordings
                       of the condition (Special_Analyses A9): quantiles side by side and the share of
                       recordings whose spot count lies inside the simulated range. DESCRIPTIVE: the
                       simulated count is before bleaching, detection, and field-of-view effects, so
                       it is judged only for gross mismatch (the empirical median inside the simulated
                       5-95% band).
    P6 visibility      The theoretical visibility chain of the RETAINED population corroborated through
                       the DLI stage's OWN labeling path (resolve_labeling, label_subunits:
                       assign_probes, draw_dye_counts_bound, labeling_summary) on a synthetic frame-0
                       lineage of 10^5 monomers and 10^5 dimers per condition (B1 / B2 at the declared
                       two-probe share under INLB; B hosts classed at labeling under FAB): visible
                       monomers q, visible one-probe dimers q, visible two-probe dimers 1 - (1-q)^2,
                       both-probes-fluorescent share among visible two-probe dimers q/(2-q) (both
                       probes carry at least one dye; a two-dye share only under the one-dye-per-probe
                       InlB law; the JSON key keeps its name), the FAB two-probe share
                       s_2, emitters per bound probe E[dye], and the dye-count law among visible
                       subunits (zero-truncated); each within four standard errors. The FAB/INLB
                       visibility ratio of the TRUE population, p q, is checked against the DECLARED
                       visibility ratio (the derivation of the Fab occupancy) and shown beside the
                       deposited spot-count ratios (A9). Theory, code path, and -- when a
                       Labeling_Set is present -- products appear side by side in the report.

Usage (from the repo root; MACHINE_PROFILE set):
    PYTHONPATH=$PWD python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py \\
        --condition FAB --split train --total-time-seconds 2 [--workflow biology|detector] \\
        [--tasks all|0,1,2] [--trajectories 8]
    ... --selftest      # in-memory synthetic products drawn from each condition's prior, realized
                        # through the canonical helper (allocation included) and labeled through
                        # the DLI stage's path from the starting counts; verifies the checks
                        # themselves (P1, P2, P4, P5, P6) without any tier on disk, and that some
                        # MET-INLB draws carry an allocation (c > 0) and no MET-FAB draw does
    ... --visibility    # P6 only, both conditions: theory vs code path (no tier needed)
    ... --campaign-version <package version> --expected-tasks 200 --task-simulations 1000 \\
        --campaign-window <iso start>..<iso end>
                        # P7 campaign membership, run after the render pass: every expected task
                        # present; the tier's Theta_Set and, under --workflow detector, the
                        # detector's own Theta_Set carrying in their schemas the campaign's package
                        # version, condition, timing, generator and a written time inside the
                        # window, with the row count; the workflow's Labeling_Set (which carries no
                        # version or time) checked for its columns, row count and the declared
                        # probe rule; the schema guard does not establish membership
                        # (DETECTOR_WORKFLOW.md 9.10)
"""
from __future__ import annotations

import argparse
import datetime
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
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming  # noqa: E402

RDS = PARAMETERS.simulation.rds
SEED = 20260914
P2_SIGMAS = 3.0               # P2: |median of a linear row - 0.5| <= P2_SIGMAS standard errors of a uniform's sample median
P2_MIN_DRAWS = 20             # P2: below this the symmetry verdict is not formed (reported only)
P2_ROUNDING_BOUND = 1.5       # P2: |N_R - expectation| <= three roundings of half a subunit
TOL_VISIBLE_FLOOR = 0.02      # P4: absolute tolerance floor on pooled fractions
TOL_P6_FLOOR = 0.002          # P6: absolute tolerance floor on Monte Carlo fractions (n = 3e5 subunits)
TOL_RATIO = 0.05              # P6: |realized FAB/INLB visibility ratio - declared| (products and code path)
N_P6_HOSTS = 100_000          # P6: monomers AND dimers per condition on the synthetic lineage
TOL_MODE = 0.03               # P3: absolute tolerance on pooled frame-0 mode occupancies
KS_ALPHA = 1e-3               # P1: a row is flagged when its KS p-value falls below this
A9_CSV = os.path.join(REPO_ROOT, "..", "Special_Analyses", "MET_NEXT_MODEL_DESIGN_PLAN", "results",
                      "A9_per_recording.csv")
# Embedded A9 reference (2026-09-14): first-2 s localizations per frame, per deposited recording,
# 60 per condition, copied from A9_per_recording.csv (column spots_first2s) so that P5 computes the
# per-recording share on a machine without the Special_Analyses tree (the CSV, when present, takes
# precedence). Quantiles min/q25/median/q75/max: Fab 17/79/143/218/307, InlB 72/251/348/448/794.
A9_FIRST2S_SPOTS = {"Fab": [
    28.84, 23.72, 35.37, 61.64, 66.81, 76.89, 59.95, 62.87, 70.93, 96.81, 133.4, 96.47, 138.56,
    141.82, 157.21, 169.48, 278.96, 148.79, 266.97, 166.81, 24.37, 16.67, 57.97, 64.9, 93.82,
    91.86, 104.98, 220.27, 143.79, 226.38, 299.46, 238.24, 303.73, 192.91, 138.2, 189.15, 306.72,
    231.26, 273.34, 279.09, 55.1, 74.39, 86.11, 91.55, 120.75, 79.06, 143.86, 180.24, 162.49,
    221.96, 291.87, 270.94, 216.63, 237.44, 198.49, 111.99, 81.83, 156.4, 188.08, 214.37,
], "InlB": [
    177.54, 190.63, 209.26, 184.61, 312.84, 296.55, 254.85, 382.74, 313.87, 365.92, 247.53, 219.96,
    355.04, 461.56, 348.47, 436.91, 323.01, 293.89, 364.43, 643.17, 191.39, 251.95, 415.58, 72.27,
    241.41, 577.55, 375.95, 521.05, 793.64, 604.5, 379.54, 716.7, 649.84, 663.03, 641.85, 521,
    546.54, 279.02, 490.77, 365.32, 443.92, 417.68, 329.73, 107.76, 167.13, 102.15, 247.29, 255.03,
    408.82, 313.7, 520.92, 408.53, 252.87, 375.04, 487.89, 347.83, 291.04, 246.19, 301.29, 153.04,
]}
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

def table_of(condition: str) -> list:
    """The condition's learnable table: its theta schema (eleven rows under MET-FAB, twelve under MET-INLB)."""
    return par.parameterization_of(condition)


def p1_prior_box(theta_physical: np.ndarray, condition: str) -> dict:
    """Every stored theta maps back inside the condition's estimator box; per row (the twelfth row
    included under MET-INLB) the marginal in the ESTIMATOR coordinate against Uniform(lo, hi)."""
    from scipy import stats
    table = table_of(condition)
    flow = par.to_flow(theta_physical, table)
    low, high = np.array(par.theta_lower_bound(condition)), np.array(par.theta_upper_bound(condition))
    inside = np.all((flow >= low - 1e-9) & (flow <= high + 1e-9), axis=1)
    rows = {}
    for j, entry in enumerate(table):
        u = (flow[:, j] - low[j]) / (high[j] - low[j])
        p_value = float(stats.kstest(u, "uniform").pvalue)
        quarters = np.histogram(u, bins=[0, 0.25, 0.5, 0.75, 1.0 + 1e-12])[0] / max(u.size, 1)
        rows[entry["KEY"]] = dict(scale=("log10" if par.is_log_row(entry) else "linear"), ks_pvalue=p_value,
                                  quarter_shares=[round(float(q), 3) for q in quarters], flagged=p_value < KS_ALPHA)
    return dict(condition=condition, n_draws=int(flow.shape[0]), n_rows=len(table), all_inside_box=bool(inside.all()),
                n_outside=int((~inside).sum()), rows=rows,
                ok=bool(inside.all()) and not any(r["flagged"] for r in rows.values()))


def realized_population(theta_physical: np.ndarray, condition: str):
    """The window-start composition of every stored theta (the canonical helper, by key)."""
    return par.population_from_theta(np.asarray(theta_physical, dtype=float), condition)


def p2_composition(theta_physical: np.ndarray, condition: str) -> dict:
    """The window-start composition realized from the stored (N_total, f[, R_active]) through the
    canonical helper: the reference population within rounding of its expectation, exact
    conservation through the allocation, the Fab allocation zero, the linear rows centered."""
    theta = np.asarray(theta_physical, dtype=float)
    pop = realized_population(theta, condition)
    classes = pop.ligand_classes
    # the same reference population without the allocation (only where the condition infers it): the
    # allocation must leave B1, A_ref and N_R untouched
    inferred = RDS.stoichiometry.active_allocation_key not in par.declared_row_values(condition)
    zero = (par.realize_population(count_total=pop.count_total, fraction_basal_dimers_reference=pop.fraction_basal_dimers_reference,
                                   relative_count_active_dimers_initial=np.zeros_like(pop.count_total), condition=condition)
            if classes and inferred else None)
    n_sub, n_exp = pop.n_subunits.astype(float), pop.expected_subunits
    rounding = np.abs(n_sub - n_exp)                      # up to three roundings of half a subunit
    n_r_reference = (pop.n_monomers_reference + pop.n_dimers_one_probe + 2 * pop.n_dimers_two_probe_reference if classes
                     else pop.n_monomers_reference + 2 * pop.n_dimers_reference)
    n_r_start = (pop.n_monomers + pop.n_dimers_one_probe + 2 * pop.n_dimers_two_probe if classes
                 else pop.n_monomers + 2 * pop.n_dimers)
    conserved = bool(np.array_equal(pop.n_subunits, n_r_reference) and np.array_equal(pop.n_subunits, n_r_start)
                     and np.array_equal(pop.n_dimers, pop.n_dimers_one_probe + pop.n_dimers_two_probe)
                     and np.array_equal(pop.n_monomers, pop.n_monomers_reference - 2 * pop.n_pairs_allocated)
                     and np.array_equal(pop.n_dimers_two_probe, pop.n_dimers_two_probe_reference + pop.n_pairs_allocated))
    # None = not applicable: no inferred allocation (MET-FAB) for the first, ligand classes (MET-INLB) for the second
    b1_unchanged = None if zero is None else bool(np.array_equal(pop.n_dimers_one_probe, zero.n_dimers_one_probe)
                                                  and np.array_equal(pop.n_monomers_reference, zero.n_monomers_reference)
                                                  and np.array_equal(pop.n_subunits, zero.n_subunits))
    c, cap = pop.n_pairs_allocated, pop.capacity
    c_bounds = bool(np.all(c >= 0) and np.all(c <= cap) and np.array_equal(c, np.round(pop.relative_count_active_dimers_initial * cap)))
    fab_zero = None if classes else bool(np.all(c == 0) and np.all(cap == 0))
    underlying = bool(np.allclose(pop.n_monomers_underlying_start + 2.0 * pop.n_dimers_underlying_start, pop.count_total, rtol=1e-9))
    # the LINEAR rows are uniform on (0, 1): their sample median lies at 0.5 within P2_SIGMAS standard
    # errors of a uniform's sample median, SE = width / (2 sqrt n)
    index = par.parameter_index_of(condition)
    linear = {}
    n = theta.shape[0]
    for entry in table_of(condition):
        if par.is_log_row(entry):
            continue
        lo, hi = entry["PRIOR_RANGE"]
        col = theta[:, index[entry["KEY"]]]
        tol = float(P2_SIGMAS * (hi - lo) / (2.0 * np.sqrt(max(n, 1))))
        median, center = float(np.median(col)), 0.5 * (lo + hi)
        linear[entry["KEY"]] = dict(median=median, center=center, tolerance=tol,
                                    centered=(abs(median - center) <= tol) if n >= P2_MIN_DRAWS else None,
                                    quantiles=quantiles(col))
    centered_ok = all(v["centered"] is not False for v in linear.values())
    with np.errstate(invalid="ignore"):
        share = pop.active_share
    return dict(condition=condition, occupancy=float(pop.occupancy), ligand_classes=classes, n_draws=int(n),
                max_reference_count_deviation=float(rounding.max()), rounding_bound=P2_ROUNDING_BOUND,
                conservation_exact=conserved, b1_and_reference_unchanged_by_allocation=b1_unchanged,
                c_within_capacity=c_bounds, fab_allocation_zero=fab_zero, underlying_population_conserved=underlying,
                linear_rows=linear, represented_receptors=quantiles(n_sub),
                basal_fraction=quantiles(pop.fraction_basal_dimers_reference), capacity=quantiles(cap),
                pairs_allocated=quantiles(c), share_without_a_pair=float(np.mean(cap == 0)),
                share_with_allocation=float(np.mean(c > 0)),
                starting_fraction=quantiles(pop.fraction_basal_dimers_start[np.isfinite(pop.fraction_basal_dimers_start)])
                if np.isfinite(pop.fraction_basal_dimers_start).any() else None,
                active_share=quantiles(share[np.isfinite(share)]) if np.isfinite(share).any() else None,
                min_monomers=int(pop.n_monomers.min()), min_dimers=int(pop.n_dimers.min()),
                ok=bool(float(rounding.max()) <= P2_ROUNDING_BOUND + 1e-9 and conserved and (b1_unchanged is not False)
                        and c_bounds and (fab_zero is not False) and underlying and centered_ok))


def p4_labeling(labeling_rows: np.ndarray, condition: str, theta_physical: np.ndarray | None) -> dict:
    col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
    L = np.asarray(labeling_rows, dtype=float)
    plan = lab.resolve_labeling(condition)
    law, q = plan.law, plan.law.visible_probability
    occ_m, occ_d = L[:, col["occupancy_monomer"]], L[:, col["occupancy_dimer"]]
    recorded = bool(np.isfinite(occ_m).all() and np.isfinite(occ_d).all())
    matches_rule = bool(recorded and np.allclose(occ_m, plan.occupancy_pair[0]) and np.allclose(occ_d, plan.occupancy_pair[1]))
    override = bool(recorded and not matches_rule)
    n_sub = max(L[:, col["n_subunits"]].sum(), 1)
    n_m, n_d = max(L[:, col["monomers_0"]].sum(), 1), max(L[:, col["dimers_0"]].sum(), 1)
    one, two = L[:, col["dimers_one_probe_0"]].sum(), L[:, col["dimers_two_probe_0"]].sum()
    classes_recorded = bool(np.isfinite(L[:, col["dimers_one_probe_0"]]).all() and np.isfinite(L[:, col["dimers_two_probe_0"]]).all())
    share_recorded = L[:, col["two_probe_share"]]
    if not override and classes_recorded:
        bound_sub = n_m + one + 2 * two
        exp_sub = q * bound_sub / n_sub
        exp_mono = q
        exp_dim = (one * q + two * (1 - (1 - q) ** 2)) / n_d
        exp_two = two * q * q / max(one * q + two * (1 - (1 - q) ** 2), 1e-12)
        exp_emit = law.mean * bound_sub / n_sub
        classes_split_dimers = bool(np.isclose(one + two, n_d))
        if plan.ligand_classes:
            class_share_ok = classes_split_dimers
        else:
            s2 = plan.two_probe_share
            class_share_ok = classes_split_dimers and abs(two / n_d - s2) <= max(TOL_VISIBLE_FLOOR, 3 * np.sqrt(s2 * (1 - s2) / n_d))
        share_ok = bool(np.all(np.isclose(share_recorded, plan.two_probe_share)))
    else:                                               # an override's coins: the per-class visibilities
        a_m = (np.nanmean(occ_m) if recorded else 1.0) * q
        a_d = (np.nanmean(occ_d) if recorded else 1.0) * q
        exp_sub = (n_m * a_m + 2 * n_d * a_d) / (n_m + 2 * n_d)
        exp_mono, exp_dim, exp_two = a_m, 1 - (1 - a_d) ** 2, a_d / (2 - a_d)
        exp_emit = exp_sub / q * law.mean
        class_share_ok = share_ok = True                # not defined under coins
    sub_vis = L[:, col["n_labeled_subunits"]].sum() / n_sub
    emit = L[:, col["n_dyes"]].sum() / n_sub
    sd_emit = np.sqrt(max((exp_emit / max(law.mean, 1e-12)) * (law.variance + law.mean ** 2) - exp_emit ** 2, 0.0))
    tol_sub = max(TOL_VISIBLE_FLOOR, 3 * np.sqrt(exp_sub * (1 - exp_sub) / n_sub))
    tol_emit = max(TOL_VISIBLE_FLOOR, 3 * sd_emit / np.sqrt(n_sub))
    mono = L[:, col["monomers_visible_0"]].sum() / n_m
    dim = L[:, col["dimers_visible_0"]].sum() / n_d
    n_vis_d = max(L[:, col["dimers_visible_0"]].sum(), 1)
    two_lab = L[:, col["dimers_two_labeled_0"]].sum() / n_vis_d
    tol_m = max(TOL_VISIBLE_FLOOR, 3 * np.sqrt(exp_mono * (1 - exp_mono) / n_m))
    tol_d = max(TOL_VISIBLE_FLOOR, 3 * np.sqrt(exp_dim * (1 - exp_dim) / n_d))
    tol_two = max(TOL_VISIBLE_FLOOR, 3 * np.sqrt(max(exp_two * (1 - exp_two), 0.0) / n_vis_d))
    subunits_ok = classes_ok = None
    n_with_allocation = None
    if theta_physical is not None:
        pop = realized_population(theta_physical, condition)
        subunits_ok = bool(np.array_equal(L[:, col["n_subunits"]].astype(int), pop.n_subunits))
        # the frame-0 class counts against the helper's STARTING counts, per simulation
        mono_ok = np.array_equal(L[:, col["monomers_0"]].astype(int), pop.n_monomers)
        dim_ok = np.array_equal(L[:, col["dimers_0"]].astype(int), pop.n_dimers)
        if plan.ligand_classes and classes_recorded:
            class_ok = (np.array_equal(L[:, col["dimers_one_probe_0"]].astype(int), pop.n_dimers_one_probe)
                        and np.array_equal(L[:, col["dimers_two_probe_0"]].astype(int), pop.n_dimers_two_probe))
        else:
            class_ok = True                              # FAB classes are drawn at labeling (checked by share above)
        classes_ok = bool(mono_ok and dim_ok and class_ok)
        n_with_allocation = int((pop.n_pairs_allocated > 0).sum())
    return dict(n_simulations=int(L.shape[0]), probe_rule=plan.probe_rule, occupancy_declared=float(plan.occupancy),
                occupancy_source=plan.occupancy_source, probe_probability_rule=tuple(plan.occupancy_pair),
                two_probe_share_declared=plan.two_probe_share,
                two_probe_share_meaning="the declared share of the REFERENCE dimers (Fab labeling draws with it; "
                                        "under INLB the starting B2 share also carries the allocation)",
                occupancy_recorded=recorded,
                occupancy_matches_declared=matches_rule, occupancy_override=override,
                occupancy_used=(float(np.nanmean(occ_m)), float(np.nanmean(occ_d))) if recorded else None,
                probe_classes_recorded=classes_recorded, one_probe_dimers=float(one), two_probe_dimers=float(two),
                two_probe_share_realized=float(two / n_d) if classes_recorded else None, class_share_ok=bool(class_share_ok),
                two_probe_share_recorded_ok=bool(share_ok),
                visible_per_subunit=(float(sub_vis), float(exp_sub), float(tol_sub)),
                visible_monomer=(float(mono), float(exp_mono), float(tol_m)),
                visible_dimer=(float(dim), float(exp_dim), float(tol_d)),
                both_labeled_share=(float(two_lab), float(exp_two), float(tol_two)),
                emitters_per_subunit=(float(emit), float(exp_emit), float(tol_emit)),
                n_subunits_equals_realized_N_R=subunits_ok, frame0_counts_equal_starting_counts=classes_ok,
                simulations_with_allocation=n_with_allocation,
                ok=bool(recorded and abs(sub_vis - exp_sub) <= tol_sub and abs(mono - exp_mono) <= tol_m
                        and abs(dim - exp_dim) <= tol_d and abs(two_lab - exp_two) <= tol_two
                        and abs(emit - exp_emit) <= tol_emit and (subunits_ok is not False) and (classes_ok is not False)
                        and class_share_ok and share_ok))


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
    """Closed-form visibility quantities of the RETAINED population from the condition's settings and dye law."""
    plan = lab.resolve_labeling(condition)
    law, q = plan.law, plan.law.visible_probability
    p, s2 = float(plan.occupancy), float(plan.two_probe_share)
    p1, p2 = (dye_law_pmf(law, np.array([1, 2])) / q).tolist()
    return dict(occupancy=p, occupancy_source=plan.occupancy_source, two_probe_share=s2, ligand_classes=plan.ligand_classes,
                dye_law=law.describe(), p_dye_ge1=float(q),
                visible_per_true_subunit=float(p * q),                      # the A9 reading: spots / (p q) = N_total
                visible_monomer=float(q), visible_one_probe_dimer=float(q),
                visible_two_probe_dimer=float(1 - (1 - q) ** 2), two_dye_share_among_visible_two_probe=float(q / (2 - q)),
                emitters_per_bound_probe=float(law.mean), emitters_per_visible_subunit=float(law.mean / q),
                dyes_among_visible_1_2_3plus=(float(p1), float(p2), float(1 - p1 - p2)))


def synthetic_lineage(condition: str, n_monomers: int, n_dimers: int, n_two_probe: int | None = None) -> tuple:
    """A frame-0 lineage of the retained population: monomers at rank 0 (one subunit each); under FAB dimers are
    ``B`` hosts of two subunits at rank 1; under INLB ``n_two_probe`` hosts are ``B2`` (two subunits, rank 2) and
    the rest ``B1`` (one subunit, rank 1). Returns ``(host_index_0, host_rank_0, species_of_rank, monomer_ranks)``."""
    classes = RDS.condition_setting(condition).ligand_classes
    hosts = [(0, 1)] * n_monomers
    if classes:
        n_two = int(round(par.two_probe_share_of(condition) * n_dimers)) if n_two_probe is None else int(n_two_probe)
        hosts += [(1, 1)] * (n_dimers - n_two) + [(2, 2)] * n_two
    else:
        hosts += [(1, 2)] * n_dimers
    ranks = np.array([r for r, _ in hosts], dtype=int)
    sizes = np.array([n for _, n in hosts], dtype=int)
    host_index_0 = np.repeat(np.arange(len(hosts)), sizes)
    host_rank_0 = np.repeat(ranks, sizes)
    species = RDS.molecular_species_names
    return host_index_0, host_rank_0, {0: species[0], 1: species[1], 2: species[1]}, [0]


def labeled_lineage(condition: str, n_monomers: int, n_dimers: int, rng: np.random.Generator,
                    n_two_probe: int | None = None) -> tuple:
    """Label a synthetic frame-0 lineage through the DLI stage's path (``resolve_labeling`` + ``label_subunits``).
    Returns ``(dye_counts, labeling_summary_row, probe_bound, host_index_0, host_rank_0)``."""
    plan = lab.resolve_labeling(condition)
    host_index_0, host_rank_0, species_of_rank, mono_ranks = synthetic_lineage(condition, n_monomers, n_dimers, n_two_probe)
    bound = lab.assign_probes(plan, host_index_0, host_rank_0, species_of_rank, np.random.default_rng(rng.integers(2**32)))
    # the same two draws, in the stage's order, through the stage's function
    dye, row = lab.label_subunits(plan, host_index_0, host_rank_0, species_of_rank, mono_ranks, rng)
    return dye, row, bound, host_index_0, host_rank_0


def p6_visibility(condition: str, n_hosts: int = N_P6_HOSTS, seed: int = SEED) -> dict:
    th = visibility_theory(condition)
    plan = lab.resolve_labeling(condition)
    rng = np.random.default_rng([seed, 6])
    dye, row, _, host_index_0, host_rank_0 = labeled_lineage(condition, n_hosts, n_hosts, rng)
    col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
    # per-host bookkeeping from the arrays themselves: subunits, bound-by-record, labeled
    is_dimer = host_rank_0 != 0
    n_total_hosts = int(host_index_0.max()) + 1
    labeled = (dye >= 1).astype(float)
    labeled_per_host = np.bincount(host_index_0, weights=labeled, minlength=n_total_hosts)
    subunits_per_host = np.bincount(host_index_0, minlength=n_total_hosts)
    dimer_hosts = np.flatnonzero(np.bincount(host_index_0, weights=is_dimer.astype(float), minlength=n_total_hosts) > 0)
    # probe classes: under INLB by host size (B1 one subunit, B2 two); under FAB from the record's counts and the
    # realized dye pattern cannot tell a dark two-Fab from a one-Fab, so the class split is read from the record
    one_probe, two_probe = row[col["dimers_one_probe_0"]], row[col["dimers_two_probe_0"]]
    if plan.ligand_classes:
        two_hosts = dimer_hosts[subunits_per_host[dimer_hosts] == 2]
        one_hosts = dimer_hosts[subunits_per_host[dimer_hosts] == 1]
        vis_one = float(np.mean(labeled_per_host[one_hosts] >= 1)) if one_hosts.size else float("nan")
        vis_two = float(np.mean(labeled_per_host[two_hosts] >= 1)) if two_hosts.size else float("nan")
        two_dye = float(np.mean(labeled_per_host[two_hosts] >= 2) / max(np.mean(labeled_per_host[two_hosts] >= 1), 1e-12)) if two_hosts.size else float("nan")
        n_one, n_two = int(one_hosts.size), int(two_hosts.size)
        realized = dict(visible_monomer=row[col["monomers_visible_0"]] / n_hosts,
                        visible_one_probe_dimer=vis_one, visible_two_probe_dimer=vis_two,
                        two_dye_share_among_visible_two_probe=two_dye)
        denom = dict(visible_monomer=n_hosts, visible_one_probe_dimer=max(n_one, 1), visible_two_probe_dimer=max(n_two, 1),
                     two_dye_share_among_visible_two_probe=max(int(np.sum(labeled_per_host[two_hosts] >= 1)), 1))
    else:
        q = th["p_dye_ge1"]
        realized = dict(visible_monomer=row[col["monomers_visible_0"]] / n_hosts)
        denom = dict(visible_monomer=n_hosts)
        # the FAB dimer visibility against the realized class mix, and the two-probe share against s_2
        exp_dim_mix = (one_probe * q + two_probe * (1 - (1 - q) ** 2)) / n_hosts
        realized["visible_dimer_class_mix"] = row[col["dimers_visible_0"]] / n_hosts
        th = dict(th, visible_dimer_class_mix=float(exp_dim_mix))
        denom["visible_dimer_class_mix"] = n_hosts
        realized["two_probe_share_realized"] = two_probe / n_hosts
        th = dict(th, two_probe_share_realized=th["two_probe_share"])
        denom["two_probe_share_realized"] = n_hosts
    n_sub = int(row[col["n_subunits"]])
    n_bound = int(row[col["monomers_0"]] + one_probe + 2 * two_probe)
    n_vis = max(int(row[col["n_labeled_subunits"]]), 1)
    vis = dye[dye >= 1]
    realized.update(emitters_per_bound_probe=row[col["n_dyes"]] / max(n_bound, 1),
                    emitters_per_visible_subunit=row[col["n_dyes"]] / n_vis,
                    dyes_among_visible_1_2_3plus=(float(np.mean(vis == 1)), float(np.mean(vis == 2)), float(np.mean(vis >= 3))))
    comparisons, ok = {}, True
    for k, n in denom.items():
        e, r = th[k], realized[k]
        tol = max(TOL_P6_FLOOR, 4 * np.sqrt(max(e * (1 - e), 0.0) / n))
        comparisons[k] = (float(r), float(e), float(tol)); ok &= (np.isnan(r) and n <= 1) or abs(r - e) <= tol
    law = plan.law
    for k, n, sd in (("emitters_per_bound_probe", max(n_bound, 1), np.sqrt(law.variance)),
                     ("emitters_per_visible_subunit", n_vis, np.sqrt(max(law.variance + law.mean ** 2 - law.mean ** 2 / th["p_dye_ge1"], 0)))):
        e, r = th[k], realized[k]
        tol = max(TOL_P6_FLOOR, 4 * sd / np.sqrt(n))
        comparisons[k] = (float(r), float(e), float(tol)); ok &= abs(r - e) <= tol
    for i, name in enumerate(("dyes_among_visible_1", "dyes_among_visible_2", "dyes_among_visible_3plus")):
        e, r = th["dyes_among_visible_1_2_3plus"][i], realized["dyes_among_visible_1_2_3plus"][i]
        tol = max(TOL_P6_FLOOR, 4 * np.sqrt(max(e * (1 - e), 0.0) / n_vis))
        comparisons[name] = (float(r), float(e), float(tol)); ok &= abs(r - e) <= tol
    # the rule the code path applied must be the model's, and recorded as such
    occ_ok = bool(np.isclose(row[col["occupancy_monomer"]], plan.occupancy_pair[0])
                  and np.isclose(row[col["occupancy_dimer"]], plan.occupancy_pair[1])
                  and np.isclose(row[col["two_probe_share"]], plan.two_probe_share)
                  and np.isclose(one_probe + two_probe, n_hosts))
    return dict(condition=condition, n_monomers=n_hosts, n_dimers=n_hosts, n_subunits=n_sub, theory=th,
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
    v = np.asarray(A9_FIRST2S_SPOTS[token], dtype=float)
    out = quantiles(v)
    out["source"] = "A9 per-recording values embedded in this script (A9_per_recording.csv, 2026-09-14)"
    out["values"] = v.tolist()
    return out


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


def p3_trajectories(paths, data_bank_root, timing, split, tasks, theta_by_task, k: int, condition: str) -> dict:
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
    n_with_allocation = 0
    for task, sim, path in pick:
        theta = np.asarray(theta_by_task[task][sim], dtype=float)
        comp = rds.initial_composition_of(theta, condition)          # the STARTING counts, allocation included
        n_with_allocation += int(comp.n_pairs_allocated > 0)
        tray = readdy.Trajectory(str(path))
        _, types, _, _ = tray.read_observable_particles()
        name_of_id = {v: k for k, v in tray.particle_types.items()}
        names0 = [name_of_id[int(t)] for t in types[0]]
        n_a = sum(1 for n in names0 if RDS.species_of_type[n] == RDS.stoichiometry.monomer.name)
        n_b = sum(1 for n in names0 if RDS.species_of_type[n] == RDS.stoichiometry.dimer.name)
        if (n_a, n_b) != (comp.n_monomers, comp.n_dimers):
            problems.append(f"task {task} sim {sim}: frame-0 counts ({n_a}, {n_b}) != starting ({comp.n_monomers}, {comp.n_dimers})")
        if comp.ligand_classes:
            n_b1 = sum(1 for n in names0 if RDS.molecule_of_type[n] == RDS.stoichiometry.dimer_one_probe.name)
            n_b2 = sum(1 for n in names0 if RDS.molecule_of_type[n] == RDS.stoichiometry.dimer_two_probe.name)
            if (n_b1, n_b2) != (comp.n_dimers_one_probe, comp.n_dimers_two_probe):
                problems.append(f"task {task} sim {sim}: frame-0 counts B1/B2 ({n_b1}, {n_b2}) != starting "
                                f"({comp.n_dimers_one_probe}, {comp.n_dimers_two_probe})")
        sub = dict(zip(RDS.particle_type_names, RDS.subunit_counts_per_type))
        totals = [sum(sub[name_of_id[int(t)]] for t in frame) for frame in types]
        if len(set(totals)) != 1 or totals[0] != comp.n_subunits:
            problems.append(f"task {task} sim {sim}: represented subunit total not conserved ({min(totals)}..{max(totals)} vs {comp.n_subunits})")
        for n in names0:
            mode_counts[RDS.mobility.mode_index(RDS.mode_of_type[n])] += 1
        n_particles += len(names0)
        law_sum += rds.stationary_mode_law(theta, condition) * len(names0)
    pooled = mode_counts / max(n_particles, 1)
    law = law_sum / max(n_particles, 1)
    dev = float(np.max(np.abs(pooled - law)))
    return dict(n_files=len(pick), files_with_allocation=n_with_allocation,
                frame0_counts_match=not any("counts" in p for p in problems),
                subunit_total_conserved=not any("conserved" in p for p in problems),
                pooled_frame0_mode_occupancy=[round(float(v), 4) for v in pooled],
                stationary_law_weighted=[round(float(v), 4) for v in law], max_abs_deviation=dev,
                problems=problems, ok=not problems and dev <= TOL_MODE)


# ----------------------------------------------------------------------------------------------
# P7: campaign membership (DETECTOR_WORKFLOW.md section 9.10)
# ----------------------------------------------------------------------------------------------

def parse_campaign_window(spec: str | None) -> tuple:
    """``"<iso>..<iso>"`` (timezone-aware) -> (start, end) datetimes; (None, None) when absent."""
    if not spec:
        return (None, None)
    a, b = spec.split("..")
    start, end = (datetime.datetime.fromisoformat(a), datetime.datetime.fromisoformat(b))
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError(f"--campaign-window must be two timezone-aware ISO times, start before end: {spec!r}")
    return (start, end)


def _in_window(written: str | None, window: tuple) -> bool | None:
    start, end = window
    if start is None:
        return None
    if not written:
        return False
    try:
        t = datetime.datetime.fromisoformat(written)
    except ValueError:
        return False
    if t.tzinfo is None:
        t = t.replace(tzinfo=datetime.timezone.utc)
    return bool(start <= t <= end)


def _check_schema(reasons: list, label: str, schema: dict | None, *, version: str, condition: str,
                  timing_label: str, generator: str, window: tuple, rows: int | None, task_simulations: int | None):
    if schema is None:
        reasons.append(f"{label}: no schema")
        return
    if schema.get("package_version") != version:
        reasons.append(f"{label}: package_version {schema.get('package_version')!r} != campaign {version!r}")
    if schema.get("condition") != condition:
        reasons.append(f"{label}: condition {schema.get('condition')!r} != {condition!r}")
    if schema.get("timing_label") != timing_label:
        reasons.append(f"{label}: timing_label {schema.get('timing_label')!r} != {timing_label!r}")
    if schema.get("generator") != generator:
        reasons.append(f"{label}: generator {schema.get('generator')!r} != {generator!r}")
    if _in_window(schema.get("written_utc"), window) is False:
        reasons.append(f"{label}: written_utc {schema.get('written_utc')!r} outside the campaign window")
    if task_simulations is not None and rows is not None and rows != task_simulations:
        reasons.append(f"{label}: {rows} rows != {task_simulations}")


def p7_campaign(provenance: list, *, condition: str, timing_label: str, workflow: str, version: str,
                expected_tasks: int | None, task_simulations: int | None, window: tuple) -> dict:
    """Campaign membership of every task against the frozen specification.

    ``provenance`` is one dict per task found: ``task``, ``rds_schema``, ``rds_rows``, and, under the
    detector workflow, ``det_schema`` / ``det_rows``, plus ``labeling`` (the task's Labeling_Set rows
    or None). The schema guard compares keys, bounds, scales and condition only, so a dataset of an
    earlier generator with the same imaging box passes it; membership is this inventory instead.
    """
    plan = lab.resolve_labeling(condition)
    col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
    found = sorted(int(e["task"]) for e in provenance)
    missing = [] if expected_tasks is None else [t for t in range(int(expected_tasks)) if t not in set(found)]
    unexpected = [] if expected_tasks is None else [t for t in found if t >= int(expected_tasks)]
    failures = {}
    for e in provenance:
        reasons = []
        _check_schema(reasons, "tier Theta_Set", e.get("rds_schema"), version=version, condition=condition,
                      timing_label=timing_label, generator="rds", window=window, rows=e.get("rds_rows"),
                      task_simulations=task_simulations)
        if workflow == "detector":
            _check_schema(reasons, "detector Theta_Set", e.get("det_schema"), version=version, condition=condition,
                          timing_label=timing_label, generator="dli_detector", window=window, rows=e.get("det_rows"),
                          task_simulations=task_simulations)
        L = e.get("labeling")
        if L is None:
            reasons.append("Labeling_Set: absent")
        else:
            L = np.asarray(L, dtype=float)
            if L.ndim != 2 or L.shape[1] != len(lab.LABELING_SET_COLUMNS):
                reasons.append(f"Labeling_Set: {L.shape[1] if L.ndim == 2 else '?'} columns != {len(lab.LABELING_SET_COLUMNS)}")
            else:
                if task_simulations is not None and L.shape[0] != task_simulations:
                    reasons.append(f"Labeling_Set: {L.shape[0]} rows != {task_simulations}")
                occ_m, occ_d, share = L[:, col["occupancy_monomer"]], L[:, col["occupancy_dimer"]], L[:, col["two_probe_share"]]
                if not (np.isfinite(occ_m).all() and np.isfinite(occ_d).all()
                        and np.allclose(occ_m, plan.occupancy_pair[0]) and np.allclose(occ_d, plan.occupancy_pair[1])):
                    reasons.append(f"Labeling_Set: probe probabilities per class {(float(np.nanmean(occ_m)), float(np.nanmean(occ_d)))} "
                                   f"!= the declared rule {tuple(plan.occupancy_pair)} (an override, or another model)")
                if not (np.isfinite(share).all() and np.allclose(share, plan.two_probe_share)):
                    reasons.append(f"Labeling_Set: two_probe_share != declared {plan.two_probe_share:.4f}")
        if reasons:
            failures[int(e["task"])] = reasons
    ok = bool(not missing and not failures)
    return dict(campaign_version=version, expected_tasks=expected_tasks, task_simulations=task_simulations,
                window=tuple(w.isoformat() if w is not None else None for w in window), workflow=workflow,
                n_tasks_found=len(found), missing_tasks=missing, unexpected_tasks=unexpected,
                n_tasks_failing=len(failures), failures=failures, ok=ok)


def collect_provenance(base, paths, root, timing_label, split, tasks, workflow, theta_by_task) -> list:
    """Per task: the tier's Theta_Set schema and rows, the detector's own Theta_Set (detector workflow),
    and the workflow's Labeling_Set rows -- the inputs of P7; nothing is validated here."""
    out = []
    for t in tasks:
        e = dict(task=int(t), rds_schema=read_theta_set_schema(base.theta_set_path(t, root, timing_label, True, split)),
                 rds_rows=int(theta_by_task[t].shape[0]), det_schema=None, det_rows=None, labeling=None)
        if workflow == "detector":
            dp = paths.theta_set_path(t, root, timing_label, True, split)
            if os.path.exists(dp):
                e["det_schema"] = read_theta_set_schema(dp)
                e["det_rows"] = int(np.asarray(load_data(dp)).shape[0])
        lp = paths.record_set_path("Labeling_Set", t, root, timing_label, True, split)
        if os.path.exists(lp):
            e["labeling"] = np.asarray(load_data(lp)[:], dtype=float)
        out.append(e)
    return out


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
        theta_by_task[t] = np.asarray(load_theta_set(p, table_of(condition), condition=condition)[:], dtype=float)
        schemas[t] = read_theta_set_schema(p)
    theta = np.vstack([theta_by_task[t] for t in tasks])
    p0 = dict(n_theta_sets=len(tasks), schema_checked=True,
              generators=sorted({sc.get("generator") for sc in schemas.values()}),
              package_versions=sorted({sc.get("package_version") for sc in schemas.values()}),
              schema_versions=sorted({sc.get("schema_version") for sc in schemas.values()}),
              written_utc_range=(min(sc.get("written_utc", "") for sc in schemas.values()),
                                 max(sc.get("written_utc", "") for sc in schemas.values())),
              ok=True)
    provenance = collect_provenance(base, paths, root, timing.label, split, tasks, args.workflow, theta_by_task)
    labeling = [e["labeling"] for e in provenance if e["labeling"] is not None]
    labeling = np.vstack(labeling) if labeling else None
    if labeling is not None and labeling.shape[1] != len(lab.LABELING_SET_COLUMNS):
        sys.exit(f"Labeling_Set has {labeling.shape[1]} columns; this audit expects the "
                 f"{len(lab.LABELING_SET_COLUMNS)}-column record of the retained-population model (re-render the DLI pass).")
    out_dir = (PARAMETERS.machine.data_bank_root / paths.posit_subdir      # the report stays on the permanent tier
               / f"{paths.project_alias}_{timing.label}_Prior_Realization_Audit_{split}")
    return dict(condition=condition, split=split, timing=timing, tasks=tasks, base=base, paths=paths, root=root,
                theta=theta, theta_by_task=theta_by_task, labeling=labeling, out_dir=out_dir, p0=p0,
                provenance=provenance)


def synthetic_products(condition: str, n: int = 2000, seed: int = SEED) -> dict:
    """In-memory products drawn from the condition's prior and the declared occupancy (the selftest)."""
    rng = np.random.default_rng(seed)
    low, high = np.array(par.theta_lower_bound(condition)), np.array(par.theta_upper_bound(condition))
    theta = par.to_physical(rng.uniform(low, high, size=(n, len(low))), table_of(condition))
    # Each composition is labeled through the DLI stage's own functions (labeled_lineage) from the
    # STARTING counts (the placed-count fields, allocation included), so the synthetic Labeling_Set is
    # produced by the code path the products come from, not by a re-implementation.
    pop = realized_population(theta, condition)
    rows = [labeled_lineage(condition, int(pop.n_monomers[i]), int(pop.n_dimers[i]), rng, int(pop.n_dimers_two_probe[i]))[1]
            for i in range(n)]
    return dict(condition=condition, theta=theta, labeling=np.array(rows, dtype=float),
                n_with_allocation=int((pop.n_pairs_allocated > 0).sum()), ligand_classes=bool(pop.ligand_classes))


def selftest_allocation(prod: dict) -> dict:
    """The selftest exercises the allocation: some MET-INLB draws have c > 0, no MET-FAB draw does."""
    n = prod["n_with_allocation"]
    ok = (n > 0) if prod["ligand_classes"] else (n == 0)
    return dict(condition=prod["condition"], n_draws=int(prod["theta"].shape[0]), draws_with_allocation=n,
                expected=("some" if prod["ligand_classes"] else "none"), ok=bool(ok))


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
        L.append(f"| P1 prior box | {passed(p1['ok'])} | {p1['n_draws']} draws over {p1['n_rows']} rows ({p1['condition']}), "
                 f"{p1['n_outside']} outside the box; rows with KS p < {KS_ALPHA}: {flagged or 'none'} |")
    p2 = res.get("p2")
    if p2 is not None:
        lin = "; ".join(f"{k} median {v['median']:.3f} vs {v['center']:.1f} (tol {v['tolerance']:.3f} = {P2_SIGMAS:g} SE at "
                        f"n = {p2['n_draws']}; centered: {v['centered']})" for k, v in p2["linear_rows"].items())
        fs = "n/a" if p2["starting_fraction"] is None else f"{p2['starting_fraction']['median']:.3f}"
        sh = "n/a" if p2["active_share"] is None else f"{p2['active_share']['median']:.3f}"
        na = lambda v: "n/a" if v is None else v      # noqa: E731
        L.append(f"| P2 composition | {passed(p2['ok'])} | {p2['n_draws']} draws at occupancy {p2['occupancy']:.4f}; N_R within "
                 f"{p2['rounding_bound']} of its expectation: max deviation {p2['max_reference_count_deviation']:.2f}; conservation "
                 f"exact through the allocation: {p2['conservation_exact']}; B1 and the reference unchanged: "
                 f"{na(p2['b1_and_reference_unchanged_by_allocation'])}; 0 <= c <= capacity: {p2['c_within_capacity']}; Fab "
                 f"allocation zero: {na(p2['fab_allocation_zero'])}; M_start + 2 D_start = N_total: "
                 f"{p2['underlying_population_conserved']}; linear "
                 f"rows: {lin}; represented receptors median {p2['represented_receptors']['median']:.0f} "
                 f"(min {p2['min_monomers']} monomers / {p2['min_dimers']} dimers placed); f quantiles q05/median/q95 "
                 f"{p2['basal_fraction']['q05']:.3f}/{p2['basal_fraction']['median']:.3f}/{p2['basal_fraction']['q95']:.3f}; capacity "
                 f"median {p2['capacity']['median']:.0f} (max {p2['capacity']['max']:.0f}); c median {p2['pairs_allocated']['median']:.0f} "
                 f"(max {p2['pairs_allocated']['max']:.0f}); share without a pair {p2['share_without_a_pair']:.3f}, with an allocation "
                 f"{p2['share_with_allocation']:.3f}; f_start median {fs}; active share median {sh} |")
    if res.get("p3") is not None:
        p3 = res["p3"]
        L.append(f"| P3 trajectories | {passed(p3['ok']) if p3['ok'] is not None else 'n/a'} | {p3.get('n_files', 0)} files "
                 f"({p3.get('files_with_allocation', 0)} with c > 0); frame-0 counts = starting counts: "
                 f"{p3.get('frame0_counts_match')}; subunit total conserved: {p3.get('subunit_total_conserved')}; pooled frame-0 mode "
                 f"occupancy {p3.get('pooled_frame0_mode_occupancy')} vs law {p3.get('stationary_law_weighted')} "
                 f"(max dev {p3.get('max_abs_deviation', float('nan')):.3f}, tol {TOL_MODE}) |")
    if res.get("p4") is not None:
        p4 = res["p4"]
        L.append(f"| P4 labeling | {passed(p4['ok'])} | {p4['n_simulations']} simulations; probe rule {p4['probe_rule']} at occupancy "
                 f"{p4['occupancy_declared']:.4f} ({p4['occupancy_source']}), two-probe share declared for the reference dimers "
                 f"{p4['two_probe_share_declared']}; frame-0 class counts = the helper's starting counts: "
                 f"{p4['frame0_counts_equal_starting_counts']} ({p4['simulations_with_allocation']} simulations with c > 0); "
                 f"recorded: {p4['occupancy_recorded']}, matches the rule: {p4['occupancy_matches_declared']}, "
                 f"override: {p4['occupancy_override']}; probe classes recorded: {p4['probe_classes_recorded']} "
                 f"(two-probe share realized {p4['two_probe_share_realized']}, ok {p4['class_share_ok']}); visible per subunit "
                 f"{p4['visible_per_subunit'][0]:.4f} vs {p4['visible_per_subunit'][1]:.4f} (tol {p4['visible_per_subunit'][2]:.4f}); "
                 f"visible monomers {p4['visible_monomer'][0]:.3f} vs {p4['visible_monomer'][1]:.3f}; visible dimers "
                 f"{p4['visible_dimer'][0]:.3f} vs {p4['visible_dimer'][1]:.3f}; two-labeled share {p4['both_labeled_share'][0]:.3f} vs "
                 f"{p4['both_labeled_share'][1]:.3f}; emitters per subunit {p4['emitters_per_subunit'][0]:.4f} vs "
                 f"{p4['emitters_per_subunit'][1]:.4f}; n_subunits = realized N_R (represented receptors): "
                 f"{p4['n_subunits_equals_realized_N_R']} |")
        p5 = res["p5"]
        s, r = p5["simulated_visible_hosts_frame0"], p5["deposited_first2s_spots"]
        L.append(f"| P5 predictive visible counts | {passed(p5['ok'])} (descriptive) | simulated visible hosts at frame 0: "
                 f"min {s['min']:.0f}, q25 {s['q25']:.0f}, median {s['median']:.0f}, q75 {s['q75']:.0f}, max {s['max']:.0f}; "
                 f"deposited first-2 s spots ({r['n']} recordings): min {r['min']:.0f}, q25 {r['q25']:.0f}, median {r['median']:.0f}, "
                 f"q75 {r['q75']:.0f}, max {r['max']:.0f}; share of recordings inside the simulated range: "
                 f"{p5['share_of_recordings_inside_simulated_range']}; empirical median inside simulated 5-95%: "
                 f"{p5['empirical_median_inside_simulated_5_95']} |")
    if res.get("allocation_exercised") is not None:
        ae = res["allocation_exercised"]
        L.append(f"| selftest: allocation exercised | {passed(ae['ok'])} | {ae['draws_with_allocation']} of {ae['n_draws']} synthetic "
                 f"{ae['condition']} draws carry c > 0 (expected {ae['expected']}) |")
    if res.get("p7") is not None:
        p7 = res["p7"]
        fails = "; ".join(f"task {t}: " + " / ".join(r) for t, r in sorted(p7["failures"].items())) or "none"
        L.append(f"| P7 campaign membership | {passed(p7['ok'])} | package {p7['campaign_version']}, {p7['workflow']} workflow, "
                 f"{p7['n_tasks_found']} task(s) found, expected {p7['expected_tasks']} (missing {p7['missing_tasks'] or 'none'}, "
                 f"beyond the expected range {p7['unexpected_tasks'] or 'none'}), {p7['task_simulations']} rows per set, "
                 f"window {p7['window'][0]} .. {p7['window'][1]}; failing tasks {p7['n_tasks_failing']}: {fails} |")
    if res.get("p6") is not None:
        p6 = res["p6"]
        L.append(f"| P6 visibility (code path) | {passed(p6['ok'])} | {p6['n_monomers']} monomers + {p6['n_dimers']} dimers of the retained "
                 f"population labeled through resolve_labeling / label_subunits (assign_probes, draw_dye_counts_bound, labeling_summary); "
                 f"occupancy {p6['theory']['occupancy']:.4f} ({p6['theory']['occupancy_source']}), two-probe share "
                 f"{p6['theory']['two_probe_share']:.4f}, {p6['theory']['dye_law']}; the record carries the rule: "
                 f"{p6['occupancy_recorded_equals_declared']}; all quantities within tolerance: {p6['ok']} (table below) |")
    if res.get("ratio") is not None:
        rt = res["ratio"]; dep = rt["deposited_spot_ratios"]
        L.append(f"| P6 visibility ratio {rt['numerator']}/{rt['anchor']} (true population, p q) | {passed(rt['ok'])} | "
                 f"{rt['realized']:.3f} vs declared {rt['declared']:.3f} (tol {rt['tolerance']}); deposited spot-count ratios (descriptive): "
                 f"{', '.join(f'{k} {v:.2f}' for k, v in dep.items() if k != 'source')} |")
    if res.get("p6") is not None:
        p6, p4 = res["p6"], res.get("p4")
        L += ["", "Visibility chain: theory from the declared settings, the DLI code path on the synthetic lineage, and "
              "(when present) the generated products. Tolerances: four standard errors (code path), max(0.02, three "
              "standard errors) (products).", "",
              "| quantity | theory | code path (realized, tol) | products (realized, tol) |", "|---|---|---|---|"]
        prod_keys = dict(visible_monomer="visible_monomer", visible_dimer_class_mix="visible_dimer")
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
    ap.add_argument("--campaign-version", default=None,
                    help="P7: the package version every product of the campaign must carry (DETECTOR_WORKFLOW.md section 9.10)")
    ap.add_argument("--expected-tasks", type=int, default=None, help="P7: tasks 0..N-1 must all be present")
    ap.add_argument("--task-simulations", type=int, default=None, help="P7: rows per Theta_Set and Labeling_Set")
    ap.add_argument("--campaign-window", default=None,
                    help="P7: '<iso>..<iso>' timezone-aware bounds of the campaign's written times")
    args = ap.parse_args()
    if any(v is not None for v in (args.expected_tasks, args.task_simulations, args.campaign_window)) \
            and args.campaign_version is None:
        ap.error("--campaign-version is required with --expected-tasks / --task-simulations / --campaign-window")

    if args.selftest or args.visibility:
        mode = "selftest" if args.selftest else "visibility"
        results = {}
        for condition in lab.LABELING_CONDITIONS:
            res = dict(p6=p6_visibility(condition))
            if args.selftest:
                prod = synthetic_products(condition)
                res.update(p1=p1_prior_box(prod["theta"], condition), p2=p2_composition(prod["theta"], condition),
                           p4=p4_labeling(prod["labeling"], condition, prod["theta"]), p5=p5_predictive(prod["labeling"], condition),
                           allocation_exercised=selftest_allocation(prod))
            results[condition] = res
            for k, v in res.items():
                print(f"  [{passed(v['ok'])}] {condition} {k}")
        ratio = visibility_ratio_check({c: dict(v=r["p6"]["theory"]["visible_per_true_subunit"]) for c, r in results.items()}, "v")
        print(f"  [{passed(ratio['ok'])}] visibility ratio {ratio['numerator']}/{ratio['anchor']} of the true population (p q): "
              f"{ratio['realized']:.3f} vs declared {ratio['declared']:.3f}")
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
    res = dict(p0=prod["p0"], p1=p1_prior_box(prod["theta"], prod["condition"]), p2=p2_composition(prod["theta"], prod["condition"]))
    if args.trajectories > 0:
        res["p3"] = p3_trajectories(prod["base"], prod["root"], prod["timing"], prod["split"], prod["tasks"],
                                    prod["theta_by_task"], args.trajectories, prod["condition"])
    if prod["labeling"] is not None:
        res["p4"] = p4_labeling(prod["labeling"], prod["condition"], prod["theta"] if prod["labeling"].shape[0] == prod["theta"].shape[0] else None)
        res["p5"] = p5_predictive(prod["labeling"], prod["condition"])
    if args.campaign_version is not None:
        res["p7"] = p7_campaign(prod["provenance"], condition=prod["condition"], timing_label=prod["timing"].label,
                                workflow=args.workflow, version=args.campaign_version, expected_tasks=args.expected_tasks,
                                task_simulations=args.task_simulations, window=parse_campaign_window(args.campaign_window))
    res["p6"] = p6_visibility(prod["condition"])
    # the FAB/INLB visibility ratio of the TRUE population (p q), the derivation of the Fab occupancy; the retained
    # records carry only the retained population's visibility, so the check is on the declared settings
    res["ratio"] = visibility_ratio_check({c: dict(v=visibility_theory(c)["visible_per_true_subunit"])
                                           for c in lab.LABELING_CONDITIONS}, "v")
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
