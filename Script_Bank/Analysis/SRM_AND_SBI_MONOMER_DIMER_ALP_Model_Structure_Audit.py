"""Model-structure audit: the separated stoichiometry-mobility generator against its specification.

Two tiers. The DETERMINISTIC tier runs without a simulation, imports only the package, and is
the correctness check of the model STRUCTURE (the model specification's implementation
increment, 2026-09-09). The RUN tier (``--run``) executes tiny, short simulations and renders;
it is compute and runs only when explicitly requested after approval.

Deterministic tier (always):
    D1 channels        `reaction_channels` yields, PER CONDITION, exactly the declared channels:
                       MET-INLB (association ratio 1) seventeen -- six association fusions (one
                       per unordered pair of monomer modes, product = dimer in the SLOWER
                       parent's mode, one shared microscopic rate), three dissociation fissions
                       (dimer -> two monomers in the dimer's mode), eight switching conversions
                       (the four shared rates, once per species, adjacent modes only); MET-FAB
                       (association ratio 0) eleven -- NO fusion channel at all (structural, not
                       a tiny rate), the same fissions and conversions. `build_system` registers
                       exactly these names under each condition.
    D2 diffusion       D[X, m] = D_A * species_factor * mode_factor for every type; for every
                       prior draw D_immobile < D_slow <= D_fast within a species and
                       D_dimer <= D_monomer within a mode (the disjoint-range guarantee).
    D3 transforms      `to_flow(to_physical(u)) == u` on prior draws INCLUDING the box corners;
                       every row of the decided table is a log row, so a blanket 10**u equals
                       `to_physical` everywhere (the per-row rule stays the contract); every
                       table VALUE is its row's prior center.
    D4 composition     `realize_initial_composition` from the dimer-to-monomer ratio r for totals
                       1..40 and the box floor/ceiling (316, 3162) at r in {0, 0.01, 1, 100, huge}:
                       conservation N_R = n_A + 2 n_B, n_B <= floor(N_R/2), monotone in r, the
                       derived x_B = 2r/(1+2r) and f_B = r/(1+r) as documented, at least one dimer
                       at the box floor, a negative ratio rejected; the association reference equals the
                       Smoluchowski expression 4 pi (2 D_A) r / ((4/3) pi r^3) = 6 D_A / r^2, and
                       under MET-INLB every fusion fires at exactly lambda_ref (R_ON = 1).
    D5 stationary law  `stationary_mode_law` sums to one and satisfies detailed balance on
                       every link of the chain.
    D6 occupancy       Occupancy is applied by MOLECULAR species: `rank_to_species` maps every
                       monomer type to A and every dimer type to B, so a per-species occupancy
                       gives identical probabilities to A_f, A_s, A_i (and to B_f, B_s, B_i),
                       and an occupancy keyed by a particle type is rejected.
    D7 fission place   Fission daughters are placed OUTSIDE the fusion radius, at
                       `SimulationStem.fission_product_distance_nm` = 2 x the reaction distance,
                       and `build_system` passes that value (declared convention of 2026-09-09:
                       at the 2 ms sub-step an eligible pair reacts with probability ~1, so
                       daughters placed AT the radius would re-fuse deterministically, ~50% per
                       step for immobile daughters). The one-step rebinding probability at the
                       prior center is reported for fast and immobile daughters under the OLD
                       placement, as the documented reason.
    D8 conditions      The condition registry: the model's tokens equal the labeling and the
                       experiment registries' tokens; MET-FAB association ratio is exactly 0.0
                       and MET-INLB exactly 1.0; the retired association-ratio row is absent from
                       the biology table and the detector's RDS nuisance; `rds_alias` carries the
                       condition token and refuses to resolve without one. Declared occupancies:
                       MET-INLB 0.5 (declared), MET-FAB 0.155 (DERIVED from the declared Fab/InlB
                       visibility ratio 0.5 and the InlB anchor), visibilities 0.25 / 0.125; a
                       setting with both or neither of occupancy and visibility_ratio is refused.
    D9 ranges          The decided prior ranges (2026-09-14) as a literal table equal the
                       parameterization's; the immobile factor keeps D_i = R_i D_A below the
                       pipelines' looser immobility threshold (0.0065 um^2/s) at the D_A center
                       for every R_i in range (immobility by construction) and a full decade
                       below the slow factor; the dissociation floor loses <= 2% of dimers over
                       20 s; the composition box is symmetric about an even split.
    D10 theta schema   A Theta_Set written by `io.write_theta_set` (.zarr and .npy) carries the
                       table's schema and reads back through `io.load_theta_set`; a table with a
                       renamed key, a table with a shifted prior bound, a different condition, a
                       schema-less .zarr, and a .npy without its sidecar are each REFUSED
                       (`ThetaSetSchemaError`), and `theta_set_status` names the reason.

Run tier (``--run``; tiny; after approval only):
    R1 conservation    One 2 s run per condition at the prior center (the reference recording length) (MET-INLB: the full
                       reactive network; MET-FAB: dissociation and switching only): the lineage
                       replays with every subunit covered once per frame (fail-loud extractor)
                       and its subunit count equals the realized N_R.
    R2 stationarity    The isolated switching chain under the MET-FAB configuration (no
                       association channel by construction) with monomers only, 10 s: the
                       time-averaged mode occupancies over the second half agree with
                       `stationary_mode_law` within a prespecified tolerance.
    R3 visible         Static labeling draws on the MET-INLB lineage (every channel exercised)
                       at each condition's DECLARED occupancy reproduce the visible fractions
                       a = p_occ (1 - P0) (monomers), 1 - (1 - a)^2 (dimers) and the both-labeled
                       share a / (2 - a) among visible dimers under the condition's law.
    R4 both conditions One short video per condition rendered from THAT condition's own
                       trajectory through the production renderer; both carry signal.
    R5 boundary        In the MET-INLB run, particles may lie outside the imaged box (open
                       lateral boundary) while the per-frame subunit total stays constant;
                       the fraction outside is reported.
    R6 timing          One 2 s MET-INLB run at the count ceiling (N_R = 3162, r = 0.01, i.e. the
                       most particles the box allows, D_A at its ceiling): wall time reported as
                       the compute check of the count range; descriptive, never a failure.

Usage (from the repo root):
    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.py [--run]

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit/
        SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.md   (report)
        audit_summary.json                                        (every number the report quotes)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from srm_and_sbi_monomer_dimer_alp import labeling as lab  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import parameterization as par  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import simulation_rds_support as rds  # noqa: E402

assert os.path.abspath(par.__file__).startswith(REPO_ROOT), (
    "Imported srm_and_sbi_monomer_dimer_alp from outside this repo checkout: " + par.__file__
    + " -- run with PYTHONPATH set to the repo root.")

PARAMETERS, PARAMETERIZATION = par.PARAMETERS, par.PARAMETERIZATION
RDS = PARAMETERS.simulation.rds

OUT_DIR = os.path.join(str(PARAMETERS.machine.data_bank_root), "Posit",
                       "SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit")
REPORT = os.path.join(OUT_DIR, "SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.md")

SEED = 20260909
N_PRIOR_DRAWS = 20_000        # D2/D3: prior draws for the ordering and round-trip checks
TOL_ROUNDTRIP = 1e-9
TOL_STATIONARY = 0.03         # R2: absolute tolerance on time-averaged mode occupancies
TOL_VISIBLE = 0.02            # R3: absolute tolerance on visible fractions
N_LABEL_DRAWS = 400           # R3: independent labelings


def passed(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def prior_center_theta() -> np.ndarray:
    return np.array([par.prior_center(e) for e in PARAMETERIZATION], dtype=float)


def key_index(key: str) -> int:
    return par.parameter_find(key)


# ----------------------------------------------------------------------------------------------
# Deterministic tier
# ----------------------------------------------------------------------------------------------

def d1_channels() -> dict:
    """Per condition: the generated channels against the declared network (17 under INLB, 11 under FAB)."""
    out = {c: _d1_condition(c) for c in RDS.condition_tokens}
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def _d1_condition(condition: str) -> dict:
    theta = prior_center_theta()
    channels = rds.reaction_channels(theta, condition)
    sto, mob = RDS.stoichiometry, RDS.mobility
    mono, dim = sto.monomer.name, sto.dimer.name
    names = [c.name for c in channels]
    kinds = {k: sum(c.kind == k for c in channels) for k in ("fusion", "fission", "conversion")}
    n_modes = len(mob.modes)
    r_on = RDS.association_ratio_of(condition)
    expected_fusions = n_modes * (n_modes + 1) // 2 if r_on > 0.0 else 0
    expected_conversions = len(mob.switching) * len(sto.species)
    problems = []
    seen_pairs = set()
    lamb_on = {c.rate for c in channels if c.kind == "fusion"}
    for c in channels:
        if c.kind == "fusion":
            m1, m2 = (RDS.mode_of_type[e] for e in c.educts)
            if any(RDS.species_of_type[e] != mono for e in c.educts):
                problems.append(f"{c.name}: an association educt is not a monomer")
            if RDS.species_of_type[c.products[0]] != dim:
                problems.append(f"{c.name}: association product is not the dimer")
            if RDS.mode_of_type[c.products[0]] != mob.slower(m1, m2):
                problems.append(f"{c.name}: product mode is not the slower parent's")
            seen_pairs.add(frozenset((m1, m2)) if m1 != m2 else frozenset((m1,)))
        elif c.kind == "fission":
            m = RDS.mode_of_type[c.educts[0]]
            if RDS.species_of_type[c.educts[0]] != dim:
                problems.append(f"{c.name}: dissociation educt is not the dimer")
            if any(RDS.species_of_type[p] != mono or RDS.mode_of_type[p] != m for p in c.products):
                problems.append(f"{c.name}: daughters are not monomers in the dimer's mode")
        elif c.kind == "conversion":
            e, p = c.educts[0], c.products[0]
            if RDS.species_of_type[e] != RDS.species_of_type[p]:
                problems.append(f"{c.name}: a mobility switch changed the species")
            if abs(mob.mode_index(RDS.mode_of_type[e]) - mob.mode_index(RDS.mode_of_type[p])) != 1:
                problems.append(f"{c.name}: a mobility switch skipped a mode")
        else:
            problems.append(f"{c.name}: unknown kind {c.kind}")
    # ReaDDy registration carries exactly these names.
    stem = rds.build_system(theta, condition)
    registered = None
    try:
        registered = sorted(r.name for r in stem.reactions.registered_reactions) if hasattr(
            stem.reactions, "registered_reactions") else None
    except Exception:  # pragma: no cover -- API surface differs across ReaDDy builds
        registered = None
    out = dict(
        condition=condition, association_ratio=r_on,
        n_channels=len(channels), n_unique_names=len(set(names)), kinds=kinds,
        expected=dict(fusion=expected_fusions, fission=n_modes, conversion=expected_conversions,
                      total=expected_fusions + n_modes + expected_conversions),
        unordered_monomer_pairs_covered=len(seen_pairs),
        one_shared_association_rate=(len(lamb_on) == 1 if r_on > 0.0 else len(lamb_on) == 0),
        problems=problems, names=names,
        registered_matches=(registered is None or registered == sorted(names)),
        registered_available=registered is not None,
    )
    out["ok"] = (len(channels) == out["expected"]["total"] and len(set(names)) == len(channels)
                 and kinds == {k: v for k, v in out["expected"].items() if k != "total"}
                 and len(seen_pairs) == expected_fusions and not problems
                 and out["one_shared_association_rate"] and out["registered_matches"])
    return out


def d2_diffusion() -> dict:
    rng = np.random.default_rng(SEED)
    low, high = np.array(par.theta_lower_bound()), np.array(par.theta_upper_bound())
    flow = rng.uniform(low, high, size=(N_PRIOR_DRAWS, len(low)))
    corners = np.array(list(itertools.product(*zip(low, high)))) if len(low) <= 12 else None
    if corners is not None:
        flow = np.vstack([flow, corners])
    phys = par.to_physical(flow)
    sto, mob = RDS.stoichiometry, RDS.mobility
    within_species_ok, within_mode_ok, formula_ok = True, True, True
    # Vectorized check of the ordering through diffusion_coefficients on a subsample, and the
    # closed-form on all draws (the function is per-theta; the closed form is what it computes).
    d_a = phys[:, key_index(mob.diffusivity_key)]
    r_b = phys[:, key_index(mob.dimer_ratio_key)]
    mode_factors = [np.ones_like(d_a)] + [phys[:, key_index(k)] for k in mob.mode_ratio_keys[1:]]
    # R_slow <= 1 (equality admissible: no slowdown), and every later mode STRICTLY slower
    # than the previous one (the disjoint-range guarantee R_immobile < R_slow).
    for j in range(1, len(mode_factors)):
        if j == 1:
            if not np.all(mode_factors[j] <= mode_factors[j - 1]):
                within_species_ok = False
        elif not np.all(mode_factors[j] < mode_factors[j - 1]):
            within_species_ok = False
    if not np.all((r_b > 0) & (r_b <= 1.0)):
        within_mode_ok = False
    for idx in rng.choice(phys.shape[0], size=200, replace=False):
        d = rds.diffusion_coefficients(phys[idx])
        for sp in sto.species_names:
            f = 1.0 if sp == sto.monomer.name else r_b[idx]
            for mode, mf in zip(mob.modes, mode_factors):
                expected = d_a[idx] * f * mf[idx]
                if not np.isclose(d[RDS.type_name(sp, mode)], expected, rtol=1e-12, atol=0):
                    formula_ok = False
        for mode in mob.modes:
            if not d[RDS.type_name(sto.dimer.name, mode)] <= d[RDS.type_name(sto.monomer.name, mode)]:
                within_mode_ok = False
    return dict(n_draws=int(phys.shape[0]), within_species_ordering=within_species_ok,
                dimer_not_faster_than_monomer=within_mode_ok, closed_form_matches=formula_ok,
                ok=within_species_ok and within_mode_ok and formula_ok)


def d3_transforms() -> dict:
    rng = np.random.default_rng(SEED + 1)
    low, high = np.array(par.theta_lower_bound()), np.array(par.theta_upper_bound())
    flow = rng.uniform(low, high, size=(N_PRIOR_DRAWS, len(low)))
    flow = np.vstack([flow, low[None], high[None]])
    phys = par.to_physical(flow)
    back = par.to_flow(phys)
    max_err = float(np.max(np.abs(back - flow)))
    all_log = all(par.is_log_row(e) for e in PARAMETERIZATION)
    blanket = np.power(10.0, flow)
    blanket_equals = bool(np.allclose(blanket, phys, rtol=1e-12))
    per_entry_ok = all(
        np.isclose(par.entry_to_flow(e, par.entry_to_physical(e, 0.3)), 0.3) for e in PARAMETERIZATION)
    center_ok = all(np.isclose(par.prior_center(e), e["VALUE"], rtol=1e-6) for e in PARAMETERIZATION)
    # The per-row rule stays the contract: a synthetic linear row must pass through untouched.
    linear_entry = dict(KEY="synthetic_linear", PRIOR_RANGE=(0.0, 1.0), LOG_FLAG=False, LOG_BASE=None)
    linear_rule_ok = (par.entry_to_physical(linear_entry, 0.3) == 0.3 and par.entry_to_flow(linear_entry, 0.3) == 0.3)
    return dict(max_roundtrip_error=max_err, all_rows_log=all_log, blanket_power_equals_to_physical=blanket_equals,
                per_entry_roundtrip=per_entry_ok, table_values_are_prior_centers=center_ok,
                linear_rule_still_honored=bool(linear_rule_ok),
                ok=(max_err < TOL_ROUNDTRIP and all_log and blanket_equals and per_entry_ok and center_ok
                    and linear_rule_ok))


def d4_composition() -> dict:
    rows, problems = [], []
    ratios = (0.0, 0.01, 1.0, 100.0, 1e9)
    for n in list(range(1, 41)) + [316, 3162]:
        previous = -1
        for r in ratios:
            c = par.realize_initial_composition(float(n), r)
            rows.append((n, r, c.n_monomers, c.n_dimers, c.fraction_realized))
            if c.n_total != n or c.n_monomers + 2 * c.n_dimers != n:
                problems.append(f"N={n}, r={r}: conservation broken ({c})")
            if c.n_dimers > n // 2 or c.n_monomers < 0:
                problems.append(f"N={n}, r={r}: dimer cap violated ({c})")
            if c.n_dimers < previous:
                problems.append(f"N={n}, r={r}: dimers not monotone in r")
            previous = c.n_dimers
            if r == 0.0 and c.n_dimers != 0:
                problems.append(f"N={n}, r=0: dimers present")
            if r == 1e9 and c.n_dimers != n // 2:
                problems.append(f"N={n}, r huge: not all pairs dimerized")
            if not np.isclose(c.fraction_realized, 2 * c.n_dimers / n):
                problems.append(f"N={n}, r={r}: realized fraction wrong")
            if not np.isclose(c.fraction_requested, 2 * r / (1 + 2 * r)):
                problems.append(f"N={n}, r={r}: derived x_B wrong")
    if not np.isclose(float(par.ratio_to_complex_fraction(1.0)), 0.5) or \
            not np.isclose(float(par.ratio_to_receptor_fraction(1.0)), 2.0 / 3.0) or \
            not np.isclose(float(par.receptor_fraction_to_ratio(2.0 / 3.0)), 1.0):
        problems.append("ratio <-> fraction helpers disagree with r=1 <-> f_B=1/2 <-> x_B=2/3")
    if par.realize_initial_composition(316, 0.01).n_dimers < 1:
        problems.append("no dimer realized at the count floor and the ratio floor")
    # rounding of a non-integer total, and the lower guard
    c = par.realize_initial_composition(10.4, 1.0)
    if c.n_total != 10:
        problems.append("N=10.4 did not round to 10")
    c = par.realize_initial_composition(0.2, 1.0)
    if c.n_total != 1 or c.n_dimers != 0:
        problems.append("N=0.2 did not floor to one monomer")
    bad = False
    try:
        par.realize_initial_composition(10, -0.5)
    except ValueError:
        bad = True
    if not bad:
        problems.append("a negative dimer-to-monomer ratio was accepted")
    # association reference
    d_a, r_nm = 0.2, PARAMETERS.simulation.stem.particle_diameter_nm
    r = r_nm * 1e-3
    smol = 4 * np.pi * (2 * d_a) * r / ((4 / 3) * np.pi * r ** 3)
    ref_ok = bool(np.isclose(rds.association_reference_rate(d_a, r_nm), smol, rtol=1e-12)
                  and np.isclose(rds.association_reference_rate(d_a, r_nm), 6 * d_a / r ** 2, rtol=1e-12))
    theta = prior_center_theta()
    lamb_ref = rds.association_reference_rate(theta[key_index(RDS.mobility.diffusivity_key)], r_nm)
    lamb_inlb = {c.rate for c in rds.reaction_channels(theta, "INLB") if c.kind == "fusion"}
    lamb_ok = bool(len(lamb_inlb) == 1 and np.isclose(lamb_inlb.pop(), RDS.association_ratio_of("INLB") * lamb_ref)
                   and RDS.association_ratio_of("INLB") == 1.0)
    fab_fusions = [c for c in rds.reaction_channels(theta, "FAB") if c.kind == "fusion"]
    fab_ok = len(fab_fusions) == 0          # structural: no channel, not a tiny rate
    return dict(n_cases=len(rows), problems=problems, association_reference_is_smoluchowski=ref_ok,
                lambda_on_equals_reference_under_INLB=lamb_ok, no_fusion_channel_under_FAB=fab_ok,
                ok=not problems and ref_ok and lamb_ok and fab_ok)


def d5_stationary() -> dict:
    rng = np.random.default_rng(SEED + 2)
    low, high = np.array(par.theta_lower_bound()), np.array(par.theta_upper_bound())
    mob = RDS.mobility
    ok_sum, ok_balance = True, True
    for _ in range(200):
        theta = par.to_physical(rng.uniform(low, high))
        pi = rds.stationary_mode_law(theta)
        if not np.isclose(pi.sum(), 1.0) or np.any(pi <= 0):
            ok_sum = False
        values = rds.theta_by_key(theta)
        fwd = {(a, b): values[k] for a, b, k in mob.switching}
        for i in range(len(mob.modes) - 1):
            a, b = mob.modes[i], mob.modes[i + 1]
            if not np.isclose(pi[i] * fwd[(a, b)], pi[i + 1] * fwd[(b, a)], rtol=1e-10):
                ok_balance = False
    return dict(sums_to_one=ok_sum, detailed_balance=ok_balance, ok=ok_sum and ok_balance)


def d6_occupancy() -> dict:
    fake_tray = SimpleNamespace(particle_types={name: i for i, name in enumerate(RDS.particle_type_names)})
    species_of_rank = rds.rank_to_species(fake_tray)
    mono_ranks = rds.monomer_ranks(fake_tray)
    sto = RDS.stoichiometry
    by_species_ok = all(
        species_of_rank[fake_tray.particle_types[RDS.type_name(sp, m)]] == sp
        for sp in sto.species_names for m in RDS.mobility.modes)
    mono_ok = sorted(mono_ranks) == sorted(
        fake_tray.particle_types[n] for n in RDS.particle_type_names if RDS.species_of_type[n] == sto.monomer.name)
    occ = lab.parse_occupancy(f"{sto.monomer.name}=0.9,{sto.dimer.name}=0.3")
    ranks = [fake_tray.particle_types[n] for n in RDS.particle_type_names]
    initial_species = [species_of_rank[r] for r in ranks]
    p = lab.occupancy_per_subunit(occ, initial_species)
    same_within_species = bool(np.all(p[:3] == 0.9) and np.all(p[3:] == 0.3))
    rejected = False
    try:
        lab.occupancy_per_subunit(lab.parse_occupancy("A_f=0.9,A_s=0.9,A_i=0.9,B_f=0.3,B_s=0.3,B_i=0.3"),
                                  initial_species)
    except ValueError:
        rejected = True
    unknown_type_rejected = False
    try:
        rds.rank_to_species(SimpleNamespace(particle_types={"A": 0, "B": 1, "C": 2}))
    except ValueError:
        unknown_type_rejected = True
    return dict(rank_to_species_by_species=by_species_ok, monomer_ranks_are_all_monomer_modes=mono_ok,
                occupancy_identical_within_species=same_within_species,
                occupancy_keyed_by_particle_type_rejected=rejected,
                legacy_species_trajectory_rejected=unknown_type_rejected,
                ok=by_species_ok and mono_ok and same_within_species and rejected and unknown_type_rejected)


def d7_fission_placement() -> dict:
    import inspect
    geom = par.PARAMETERS.simulation.stem
    r_nm = float(geom.particle_diameter_nm)
    d_prod = float(geom.fission_product_distance_nm)
    outside = d_prod > r_nm
    is_twice = abs(d_prod - 2.0 * r_nm) < 1e-9
    src = inspect.getsource(rds.build_system)
    passes_field = ("fission_product_distance_nm" in src
                    and "product_distance=product_distance_nm" in src
                    and "product_distance=reaction_distance_nm" not in src)
    # Documented reason: one-step rebinding probability under placement AT the radius, for a
    # daughter pair with relative displacement variance 2 * (2 D) dt per axis (2D Brownian).
    theta = prior_center_theta()
    coeff = rds.diffusion_coefficients(theta)
    dt_s = par.PARAMETERS.simulation.timing.frame_time_seconds / par.PARAMETERS.simulation.timing.steps_per_frame
    mono = RDS.stoichiometry.monomer.name
    rebind = {}
    rng = np.random.default_rng(0)
    for mode in RDS.mobility.modes:
        d_um2_s = coeff[RDS.type_name(mono, mode)]
        sigma_nm = np.sqrt(4.0 * d_um2_s * dt_s) * 1e3
        xi = rng.normal(0.0, sigma_nm, (200_000, 2)); xi[:, 0] += r_nm
        rebind[mode] = float(np.mean(np.hypot(xi[:, 0], xi[:, 1]) <= r_nm))
    return dict(reaction_distance_nm=r_nm, product_distance_nm=d_prod, outside_fusion_radius=outside,
                twice_reaction_distance=is_twice, build_system_uses_field=passes_field,
                sub_step_s=dt_s, one_step_rebind_probability_if_placed_at_radius=rebind,
                ok=outside and is_twice and passes_field)


def d8_conditions() -> dict:
    """The condition registry and the retirement of the association-ratio row."""
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp.experiment_support import CONDITION_DISPLAY
    tokens = RDS.condition_tokens
    tokens_agree = (set(tokens) == set(lab.LABELING_CONDITIONS) == set(CONDITION_DISPLAY)
                    and len(set(tokens)) == len(tokens))
    fab_zero = RDS.association_ratio_of("FAB") == 0.0
    inlb_one = RDS.association_ratio_of("INLB") == 1.0
    retired = "relative_rate_dimerization"
    row_absent = (retired not in par.PARAMETER_KEYS
                  and retired not in [e["KEY"] for e in det.DETECTOR_NUISANCE]
                  and not hasattr(RDS.stoichiometry, "association_ratio_key"))
    epsilon_refused = False
    try:
        par.ConditionSetting("X", 1e-6)
    except ValueError:
        epsilon_refused = True
    bare_refused = False
    try:
        _ = par.PARAMETERS.paths.rds_alias
    except ValueError:
        bare_refused = True
    aliases = {c: par.PARAMETERS.paths.with_condition(c).rds_alias for c in tokens}
    alias_ok = all(aliases[c] == f"{par.PARAMETERS.paths.sibling_alias}_{c}" for c in tokens)
    det_alias_ok = all(det.detector_paths(par.PARAMETERS.paths).with_condition(c).rds_alias == aliases[c]
                       for c in tokens)
    # Declared occupancies (2026-09-14): INLB 0.5 declared; FAB derived = 0.5 x (0.5 x 0.5) / 0.806 = 0.155.
    occ = {c: par.occupancy_of(c) for c in tokens}
    vis = {c: par.visibility_of(c) for c in tokens}
    src = {c: par.occupancy_source_of(c) for c in tokens}
    fab_setting = RDS.condition_setting("FAB")
    fab_derived_ok = (src["FAB"] == "derived" and fab_setting.visibility_ratio == 0.5
                      and fab_setting.visibility_ratio_to == "INLB"
                      and abs(occ["FAB"] - 0.5 * vis["INLB"] / RDS.dye_probability_of("FAB")) < 1e-12
                      and abs(occ["FAB"] - 0.155) < 5e-4 and abs(vis["FAB"] - 0.125) < 1e-12)
    inlb_declared_ok = src["INLB"] == "declared" and occ["INLB"] == 0.5 and abs(vis["INLB"] - 0.25) < 1e-12
    both_refused = neither_refused = False
    try:
        par.ConditionSetting("X", 1.0, occupancy=0.5, visibility_ratio=0.5, visibility_ratio_to="INLB")
    except ValueError:
        both_refused = True
    try:
        par.ConditionSetting("X", 1.0)
    except ValueError:
        neither_refused = True
    return dict(tokens=list(tokens), tokens_agree_across_registries=tokens_agree, fab_ratio_is_zero=fab_zero,
                inlb_ratio_is_one=inlb_one, association_row_absent=row_absent, disguised_zero_refused=epsilon_refused,
                bare_rds_alias_refused=bare_refused, tier_aliases=aliases, detector_reads_same_tier=det_alias_ok,
                occupancy={c: round(v, 4) for c, v in occ.items()}, visibility=vis, occupancy_source=src,
                fab_occupancy_derived_ok=fab_derived_ok, inlb_occupancy_declared_ok=inlb_declared_ok,
                both_and_neither_refused=both_refused and neither_refused,
                ok=tokens_agree and fab_zero and inlb_one and row_absent and epsilon_refused and bare_refused
                and alias_ok and det_alias_ok and fab_derived_ok and inlb_declared_ok and both_refused
                and neither_refused)


# The decided prior ranges (2026-09-14), as a second, literal copy: a table edit that is not a
# decision fails D9. Estimator coordinate = log10 of the physical value for every row.
DECIDED_RANGES = {
    "count_total": (2.5, 3.5), "ratio_dimer_monomer_initial": (-2.0, 2.0), "rate_dissociation": (-3.0, 1.0),
    "diffusivity_alp": (-1.25, -0.25), "relative_diffusivity_dimer": (-1.0, 0.0),
    "relative_diffusivity_slow": (-1.0, 0.0), "relative_diffusivity_immobile": (-3.0, -2.0),
    "rate_fast_slow": (-1.0, 1.0), "rate_slow_fast": (-1.0, 1.0), "rate_slow_immobile": (-1.0, 1.0),
    "rate_immobile_slow": (-1.0, 1.0),
}
IMMOBILE_THRESHOLDS_UM2_S = (0.0028, 0.0065)     # the tracking pipelines' immobility thresholds


def d9_ranges() -> dict:
    """The decided ranges, immobility by construction, the dissociation floor, the symmetric composition box."""
    table = {e["KEY"]: tuple(float(v) for v in e["PRIOR_RANGE"]) for e in PARAMETERIZATION}
    ranges_match = table == {k: tuple(float(x) for x in v) for k, v in DECIDED_RANGES.items()}
    e_da = PARAMETERIZATION[key_index(RDS.mobility.diffusivity_key)]
    e_ri = PARAMETERIZATION[key_index("relative_diffusivity_immobile")]
    e_rs = PARAMETERIZATION[key_index("relative_diffusivity_slow")]
    d_a_center, d_a_max = par.prior_center(e_da), float(par.entry_to_physical(e_da, e_da["PRIOR_RANGE"][1]))
    r_i_max = float(par.entry_to_physical(e_ri, e_ri["PRIOR_RANGE"][1]))
    r_s_min = float(par.entry_to_physical(e_rs, e_rs["PRIOR_RANGE"][0]))
    immobile_at_center = r_i_max * d_a_center <= IMMOBILE_THRESHOLDS_UM2_S[1]
    immobile_at_ceiling_strict = r_i_max * d_a_max <= IMMOBILE_THRESHOLDS_UM2_S[0]   # reported, not required
    decade_gap = r_s_min / r_i_max >= 10.0 - 1e-9
    e_off = PARAMETERIZATION[key_index(RDS.stoichiometry.dissociation_rate_key)]
    k_off_min = float(par.entry_to_physical(e_off, e_off["PRIOR_RANGE"][0]))
    loss_20s = 1.0 - np.exp(-k_off_min * 20.0)
    e_r = PARAMETERIZATION[key_index(RDS.stoichiometry.composition_ratio_key)]
    symmetric = abs(e_r["PRIOR_RANGE"][0] + e_r["PRIOR_RANGE"][1]) < 1e-12
    return dict(ranges_match_decided_table=ranges_match, d_i_max_at_d_a_center=r_i_max * d_a_center,
                immobile_below_looser_threshold_at_center=bool(immobile_at_center),
                immobile_below_stricter_threshold_at_d_a_ceiling=bool(immobile_at_ceiling_strict),
                slow_over_immobile_gap=r_s_min / r_i_max, decade_gap=bool(decade_gap),
                dissociation_floor_loss_over_20s=float(loss_20s), composition_box_symmetric=symmetric,
                ok=ranges_match and immobile_at_center and decade_gap and loss_20s <= 0.02 and symmetric)


# ----------------------------------------------------------------------------------------------
# Run tier (tiny simulations; only with --run)
# ----------------------------------------------------------------------------------------------

R1_SECONDS = 2.0       # the reference recording length: the run tier is exercised at the main case


def _run(theta: np.ndarray, condition: str, seconds: float, workdir: str, name: str):
    import readdy
    from srm_and_sbi_monomer_dimer_alp.parameterization import RunTiming
    timing = RunTiming(total_time_seconds=seconds, frames=PARAMETERS.simulation.timing)
    stem = rds.build_system(theta, condition)
    smut = rds.build_simulation(stem, theta, seed=SEED)
    path = os.path.join(workdir, name)
    smut.output_file = path
    smut.progress_output_stride = timing.total_steps
    smut.run(n_steps=timing.total_steps,
             timestep=timing.delta_time_nanoseconds * readdy.units.nanosecond, show_summary=False)
    del smut, stem
    return readdy.Trajectory(path), timing


def r1_r5_reactive(workdir: str, condition: str) -> tuple:
    theta = prior_center_theta()
    tray, timing = _run(theta, condition, R1_SECONDS, workdir, f"structure_audit_center_{condition}.h5")
    poses = rds.extract_trajectory_poses(tray)
    lineage = rds.extract_subunit_lineage(tray)
    comp = rds.initial_composition_of(theta)
    species_of_rank = rds.rank_to_species(tray)
    # per-frame subunit total from the particles observable (independent of the lineage)
    _, types, ids, _ = tray.read_observable_particles()
    subunits_of_rank = {}
    for name, n_sub in zip(RDS.particle_type_names, RDS.subunit_counts_per_type):
        if name in tray.particle_types:
            subunits_of_rank[int(tray.particle_types[name])] = n_sub
    totals = np.array([sum(subunits_of_rank[int(t)] for t in types[f]) for f in range(lineage.n_frames)])
    box = PARAMETERS.simulation.stem.box_size
    xy = rds.collapse_species_axis(poses)[..., :2]
    present = np.isfinite(xy).all(axis=-1)
    outside = ((xy[..., 0] < 0) | (xy[..., 0] > box[0]) | (xy[..., 1] < 0) | (xy[..., 1] > box[1])) & present
    r1 = dict(condition=condition, n_subunits=lineage.n_subunits, n_total_realized=comp.n_total,
              subunit_total_constant=bool(np.all(totals == totals[0])), first_total=int(totals[0]),
              ok=lineage.n_subunits == comp.n_total and bool(np.all(totals == comp.n_total)))
    r5 = dict(condition=condition, frames=int(present.shape[0]), particles_outside_last_frame=int(outside[-1].sum()),
              fraction_outside_mean=float(outside.sum() / max(present.sum(), 1)),
              subunit_total_constant=r1["subunit_total_constant"],
              note="the open lateral boundary keeps particles simulated outside the imaged field; "
                   "the total is conserved regardless (in-field count != N_R).",
              ok=r1["subunit_total_constant"])
    return r1, r5, lineage, tray, species_of_rank


def r2_stationarity(workdir: str) -> dict:
    theta = prior_center_theta()
    theta[key_index(RDS.stoichiometry.composition_ratio_key)] = 0.0    # ratio 0 -> monomers only
    theta[key_index(RDS.stoichiometry.count_total_key)] = 600.0
    # MET-FAB has no association channel by construction, so with monomers only the run IS the
    # isolated switching chain (no reaction can create a dimer; fissions have no educt).
    tray, timing = _run(theta, "FAB", 10.0, workdir, "structure_audit_chain.h5")
    poses = rds.extract_trajectory_poses(tray)
    present = np.isfinite(poses).any(axis=2)                          # (frames, particles, types)
    per_type = present.sum(axis=1)                                     # (frames, types)
    mode_of_rank = {int(r): RDS.mode_of_type[n] for n, r in tray.particle_types.items()}
    modes = RDS.mobility.modes
    per_mode = np.zeros((per_type.shape[0], len(modes)))
    for r, m in mode_of_rank.items():
        per_mode[:, modes.index(m)] += per_type[:, r]
    half = per_mode.shape[0] // 2
    frac = per_mode[half:].sum(axis=0) / per_mode[half:].sum()
    pi = rds.stationary_mode_law(theta)
    return dict(stationary_law=pi.tolist(), time_averaged_second_half=frac.tolist(),
                max_abs_deviation=float(np.max(np.abs(frac - pi))), n_particles=int(per_mode[0].sum()),
                ok=bool(np.max(np.abs(frac - pi)) <= TOL_STATIONARY))


def r3_visible(lineage, tray, species_of_rank) -> dict:
    """Visible fractions and the both-labeled share (visible dimers with both subunits labeled) at each condition's DECLARED occupancy."""
    rng = np.random.default_rng(SEED + 3)
    initial_species = [species_of_rank[int(r)] for r in lineage.host_rank[0]]
    mono_ranks = rds.monomer_ranks(tray)
    out = {}
    for condition in lab.LABELING_CONDITIONS:
        name, law = lab.resolve_labeling_law(condition)
        p_occ = par.occupancy_of(condition)
        p_sub = lab.occupancy_per_subunit(p_occ, initial_species)
        occ_pair = lab.occupancy_by_species(p_occ, RDS.molecular_species_names)
        rows = np.stack([lab.labeling_summary(
            lab.draw_dye_counts(law, lineage.n_subunits, rng, occupancy=p_sub),
            lineage.host_index[0], lineage.host_rank[0], mono_ranks, occupancy_by_species_values=occ_pair)
            for _ in range(N_LABEL_DRAWS)])
        col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
        mono_vis = rows[:, col["monomers_visible_0"]].sum() / max(rows[:, col["monomers_0"]].sum(), 1)
        dim_vis = rows[:, col["dimers_visible_0"]].sum() / max(rows[:, col["dimers_0"]].sum(), 1)
        both_labeled = rows[:, col["dimers_two_labeled_0"]].sum() / max(rows[:, col["dimers_visible_0"]].sum(), 1)
        a = p_occ * law.visible_probability
        expect_mono, expect_dim, expect_two = a, 1 - (1 - a) ** 2, a / (2 - a)
        recorded = bool(np.all(rows[:, col["occupancy_monomer"]] == p_occ) and np.all(rows[:, col["occupancy_dimer"]] == p_occ))
        out[condition] = dict(law=name, occupancy=p_occ, source=par.occupancy_source_of(condition),
                              monomers_0=int(rows[0, col["monomers_0"]]), dimers_0=int(rows[0, col["dimers_0"]]),
                              visible_monomer_emp=float(mono_vis), visible_monomer=expect_mono,
                              visible_dimer_emp=float(dim_vis), visible_dimer=expect_dim,
                              both_labeled_share_emp=float(both_labeled), both_labeled_share=expect_two, occupancy_recorded=recorded,
                              ok=bool(abs(mono_vis - expect_mono) <= TOL_VISIBLE and abs(dim_vis - expect_dim) <= TOL_VISIBLE
                                      and abs(both_labeled - expect_two) <= TOL_VISIBLE and recorded))
    out["ok"] = all(v["ok"] for k, v in out.items() if k != "ok")
    return out


def r4_both_conditions(runs: dict) -> dict:
    """``runs``: condition -> (lineage, tray, species_of_rank) of THAT condition's own trajectory.

    Labels the lineage exactly as the DLI stage does -- ``draw_dye_counts`` with the per-subunit
    occupancy of the condition's DECLARED probe occupancy -- and renders 20 frames through the
    production renderer; reports labeled subunits beside the dye total.
    """
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video
    rng = np.random.default_rng(SEED + 4)
    imaging = np.array([10 ** ((e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2) for e in det.DETECTOR_IMAGING])
    out = {}
    for condition in lab.LABELING_CONDITIONS:
        lineage, tray, species_of_rank = runs[condition]
        poses = rds.collapse_species_axis(rds.extract_trajectory_poses(tray))[:20]
        _, law = lab.resolve_labeling_law(condition)
        p_occ = par.occupancy_of(condition)
        initial_species = [species_of_rank[int(r)] for r in lineage.host_rank[0]]
        dyes = lab.draw_dye_counts(law, lineage.n_subunits, rng,
                                   occupancy=lab.occupancy_per_subunit(p_occ, initial_species))
        frames = render_dli_video(poses, lineage.host_index[:20], dyes, imaging, seed=SEED)
        n_labeled = int((dyes >= 1).sum())
        expected_labeled = p_occ * law.visible_probability * lineage.n_subunits
        out[condition] = dict(frames=int(frames.shape[2]), n_subunits=int(lineage.n_subunits),
                              occupancy=float(p_occ), n_labeled=n_labeled,
                              expected_labeled=float(expected_labeled), n_dyes=int(dyes.sum()),
                              pixel_max=float(frames.max()), own_trajectory=True,
                              ok=bool(frames.shape[2] == 20 and frames.max() > 0 and np.isfinite(frames).all()
                                      and n_labeled > 0))
    out["ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict))
    return out


def r6_timing(workdir: str) -> dict:
    """Wall time of one 2 s MET-INLB run at the most particles the decided box allows (descriptive)."""
    import time
    theta = prior_center_theta()
    e_n = PARAMETERIZATION[key_index(RDS.stoichiometry.count_total_key)]
    e_r = PARAMETERIZATION[key_index(RDS.stoichiometry.composition_ratio_key)]
    e_d = PARAMETERIZATION[key_index(RDS.mobility.diffusivity_key)]
    theta[key_index(e_n["KEY"])] = float(par.entry_to_physical(e_n, e_n["PRIOR_RANGE"][1]))   # 3162 subunits
    theta[key_index(e_r["KEY"])] = float(par.entry_to_physical(e_r, e_r["PRIOR_RANGE"][0]))   # r = 0.01 -> most particles
    theta[key_index(e_d["KEY"])] = float(par.entry_to_physical(e_d, e_d["PRIOR_RANGE"][1]))   # fastest diffusion
    comp = rds.initial_composition_of(theta)
    t0 = time.perf_counter()
    tray, timing = _run(theta, "INLB", 2.0, workdir, "structure_audit_timing.h5")
    wall = time.perf_counter() - t0
    del tray
    return dict(condition="INLB", seconds_simulated=2.0, n_subunits=comp.n_total, n_particles=comp.n_complexes,
                n_monomers=comp.n_monomers, n_dimers=comp.n_dimers, wall_seconds=float(wall),
                wall_seconds_per_simulated_second=float(wall / 2.0), ok=bool(np.isfinite(wall)))


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

def write_report(det_out: dict, run_out: dict | None) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    L = ["# Model-structure audit: separated stoichiometry-mobility generator", "",
         f"Repository `srm-and-sbi-monomer-dimer-alp`; particle types {RDS.particle_type_names}; "
         f"parameters {par.PARAMETER_KEYS}; condition association ratios "
         f"{ {c: RDS.association_ratio_of(c) for c in RDS.condition_tokens} }.", "",
         "## Deterministic tier", "", "| check | verdict | detail |", "|---|---|---|"]
    d = det_out
    for c in RDS.condition_tokens:
        dc = d["d1"][c]
        L.append(f"| D1 channels under {c} (R_ON = {dc['association_ratio']:g}) | {passed(dc['ok'])} | {dc['n_channels']} channels "
                 f"(expected {dc['expected']['total']}), {dc['n_unique_names']} unique names, kinds {dc['kinds']}, "
                 f"association rate shared/absent as declared: {dc['one_shared_association_rate']}, "
                 f"ReaDDy names match: {dc['registered_matches']} |")
    L.append(f"| D2 diffusion ordering | {passed(d['d2']['ok'])} | {d['d2']['n_draws']} prior draws incl. box corners; "
             f"immobile < slow <= fast within species: {d['d2']['within_species_ordering']}; dimer <= monomer within "
             f"mode: {d['d2']['dimer_not_faster_than_monomer']}; closed form: {d['d2']['closed_form_matches']} |")
    L.append(f"| D3 transforms | {passed(d['d3']['ok'])} | max round-trip error {d['d3']['max_roundtrip_error']:.2e}; "
             f"all rows log: {d['d3']['all_rows_log']}; blanket 10**u equals to_physical: "
             f"{d['d3']['blanket_power_equals_to_physical']}; per-row rule still honors a linear row: "
             f"{d['d3']['linear_rule_still_honored']}; table values are prior centers: {d['d3']['table_values_are_prior_centers']} |")
    L.append(f"| D4 initial composition and association reference | {passed(d['d4']['ok'])} | {d['d4']['n_cases']} "
             f"(N, r) cases incl. the box floor/ceiling and r in {{0, 0.01, 1, 100, huge}}; problems: {d['d4']['problems'] or 'none'}; "
             f"lambda_ref = 6 D_A / r^2 = Smoluchowski/volume: {d['d4']['association_reference_is_smoluchowski']}; "
             f"INLB fusions at exactly lambda_ref: {d['d4']['lambda_on_equals_reference_under_INLB']}; "
             f"no fusion channel under FAB: {d['d4']['no_fusion_channel_under_FAB']} |")
    L.append(f"| D5 stationary mode law | {passed(d['d5']['ok'])} | sums to one: {d['d5']['sums_to_one']}; "
             f"detailed balance on every link: {d['d5']['detailed_balance']} |")
    L.append(f"| D6 occupancy by molecular species | {passed(d['d6']['ok'])} | identical within species: "
             f"{d['d6']['occupancy_identical_within_species']}; keyed by particle type rejected: "
             f"{d['d6']['occupancy_keyed_by_particle_type_rejected']}; legacy A/B/C trajectory rejected: "
             f"{d['d6']['legacy_species_trajectory_rejected']} |")
    L.append(f"| D7 fission placement | {passed(d['d7']['ok'])} | daughters at {d['d7']['product_distance_nm']:.0f} nm = 2 x reaction distance {d['d7']['reaction_distance_nm']:.0f} nm: {d['d7']['twice_reaction_distance']}; outside the fusion radius: "
             f"{d['d7']['outside_fusion_radius']}; build_system passes the field: {d['d7']['build_system_uses_field']}; one-step rebinding "
             f"probability if placed AT the radius (sub-step {d['d7']['sub_step_s']*1e3:.0f} ms, prior center) by daughter mode: "
             f"{ {k: round(v, 3) for k, v in d['d7']['one_step_rebind_probability_if_placed_at_radius'].items()} } |")
    L.append(f"| D8 condition registry | {passed(d['d8']['ok'])} | tokens {d['d8']['tokens']} agree across registries: "
             f"{d['d8']['tokens_agree_across_registries']}; FAB ratio 0.0: {d['d8']['fab_ratio_is_zero']}; INLB ratio 1.0: "
             f"{d['d8']['inlb_ratio_is_one']}; retired association-ratio row absent: {d['d8']['association_row_absent']}; "
             f"disguised zero refused: {d['d8']['disguised_zero_refused']}; bare rds_alias refused: "
             f"{d['d8']['bare_rds_alias_refused']}; tier aliases {d['d8']['tier_aliases']}; detector reads the same tier: "
             f"{d['d8']['detector_reads_same_tier']}; occupancies {d['d8']['occupancy']} ({d['d8']['occupancy_source']}), "
             f"visibilities { {c: round(v, 4) for c, v in d['d8']['visibility'].items()} }; FAB derived from the "
             f"declared ratio and the INLB anchor: {d['d8']['fab_occupancy_derived_ok']}; INLB declared: "
             f"{d['d8']['inlb_occupancy_declared_ok']}; both/neither refused: {d['d8']['both_and_neither_refused']} |")
    L.append(f"| D9 decided ranges | {passed(d['d9']['ok'])} | table equals the decided ranges: {d['d9']['ranges_match_decided_table']}; "
             f"D_i max at the D_A center {d['d9']['d_i_max_at_d_a_center']:.4f} um^2/s below the looser immobility threshold: "
             f"{d['d9']['immobile_below_looser_threshold_at_center']} (below the stricter one at the D_A ceiling: "
             f"{d['d9']['immobile_below_stricter_threshold_at_d_a_ceiling']}, reported); slow/immobile gap "
             f"{d['d9']['slow_over_immobile_gap']:.1f}x (>= one decade: {d['d9']['decade_gap']}); dissociation floor loses "
             f"{100 * d['d9']['dissociation_floor_loss_over_20s']:.1f}% over 20 s; composition box symmetric: "
             f"{d['d9']['composition_box_symmetric']} |")
    L.append(f"| D10 theta schema | {passed(d['d10']['ok'])} | round trip .zarr/.npy: {d['d10']['roundtrip.zarr']}/{d['d10']['roundtrip.npy']}; "
             f"refuses renamed key: {d['d10']['refuses_renamed_key']}, shifted bound: {d['d10']['refuses_shifted_bound']}, "
             f"other condition: {d['d10']['refuses_other_condition']}, schema-less .zarr: {d['d10']['refuses_schemaless_zarr']}, "
             f".npy without sidecar: {d['d10']['refuses_npy_without_sidecar']}; status names the reason: "
             f"{d['d10']['status_names_reason']}; package {d['d10']['schema_package_version']} |")
    for c in RDS.condition_tokens:
        L += ["", f"Channels under {c}:", ""] + [f"- `{n}`" for n in d["d1"][c]["names"]] + [""]
    if run_out is None:
        L += ["## Run tier", "", "Not executed (`--run` not given). The run tier needs explicit approval.", ""]
    else:
        r = run_out
        L += ["## Run tier (tiny simulations)", "", "| check | verdict | detail |", "|---|---|---|"]
        for c in RDS.condition_tokens:
            rc = r["r1"][c]
            L.append(f"| R1 conservation {c} | {passed(rc['ok'])} | lineage {rc['n_subunits']} subunits = realized "
                     f"N_R {rc['n_total_realized']}; per-frame subunit total constant: {rc['subunit_total_constant']} |")
        L.append(f"| R2 stationary occupancies | {passed(r['r2']['ok'])} | law {np.round(r['r2']['stationary_law'], 3).tolist()} "
                 f"vs time average {np.round(r['r2']['time_averaged_second_half'], 3).tolist()}; max |dev| "
                 f"{r['r2']['max_abs_deviation']:.3f} (tol {TOL_STATIONARY}) |")
        for c in lab.LABELING_CONDITIONS:
            v = r["r3"][c]
            L.append(f"| R3 visible fractions {c} (occupancy {v['occupancy']:.3f}, {v['source']}) | {passed(v['ok'])} | "
                     f"monomer {v['visible_monomer_emp']:.3f} vs {v['visible_monomer']:.3f}; dimer {v['visible_dimer_emp']:.3f} vs "
                     f"{v['visible_dimer']:.3f}; both-labeled share among visible dimers {v['both_labeled_share_emp']:.3f} vs "
                     f"{v['both_labeled_share']:.3f}; occupancy recorded in the labeling row: {v['occupancy_recorded']} |")
        L.append(f"| R4 both conditions, each from its own trajectory, at the declared occupancy | {passed(r['r4']['ok'])} | "
                 + "; ".join(f"{c}: {r['r4'][c]['frames']} frames, {r['r4'][c]['n_labeled']} labeled subunits of "
                             f"{r['r4'][c]['n_subunits']} at occupancy {r['r4'][c]['occupancy']:.3f} "
                             f"(expected {r['r4'][c]['expected_labeled']:.0f}; {r['r4'][c]['n_dyes']} dyes), "
                             f"pixel max {r['r4'][c]['pixel_max']:.0f}"
                             for c in lab.LABELING_CONDITIONS) + " |")
        L.append(f"| R5 boundary | {passed(r['r5']['ok'])} | particles outside the field at the last frame: "
                 f"{r['r5']['particles_outside_last_frame']}; mean fraction outside {r['r5']['fraction_outside_mean']:.4f}; "
                 f"subunit total constant: {r['r5']['subunit_total_constant']} |")
        L.append(f"| R6 timing at the count ceiling | {passed(r['r6']['ok'])} | {r['r6']['condition']}, {r['r6']['n_subunits']} subunits = "
                 f"{r['r6']['n_particles']} particles ({r['r6']['n_monomers']} monomers + {r['r6']['n_dimers']} dimers), D_A at its ceiling, "
                 f"{r['r6']['seconds_simulated']:.0f} s simulated in {r['r6']['wall_seconds']:.1f} s wall "
                 f"({r['r6']['wall_seconds_per_simulated_second']:.1f} s per simulated second); descriptive |")
        L.append("")
    with open(REPORT, "w") as handle:
        handle.write("\n".join(L) + "\n")
    with open(os.path.join(OUT_DIR, "audit_summary.json"), "w") as handle:
        json.dump(dict(deterministic=det_out, run=run_out), handle, indent=2, default=str)


def d10_theta_schema() -> dict:
    """Theta_Set schema round trip and refusals (io.write_theta_set / load_theta_set / theta_set_status)."""
    from srm_and_sbi_monomer_dimer_alp import io as sio
    import zarr
    table = par.PARAMETERIZATION
    rng = np.random.default_rng(20260914)
    low, high = np.array(par.theta_lower_bound()), np.array(par.theta_upper_bound())
    theta = par.to_physical(rng.uniform(low, high, size=(5, len(low))))
    schema = sio.theta_set_schema(table, condition="FAB", timing_label="2S_50FPS", generator="audit")
    out = dict(schema_keys=list(schema["parameter_keys"]) == list(par.PARAMETER_KEYS),
               schema_package_version=schema["package_version"])

    def refused(path, tbl, condition=None):
        try:
            sio.load_theta_set(path, tbl, condition=condition)
            return False
        except sio.ThetaSetSchemaError:
            return True

    with tempfile.TemporaryDirectory() as d:
        for ext in (".zarr", ".npy"):
            p = os.path.join(d, "Theta_Set" + ext)
            sio.write_theta_set(p, theta, schema)
            back = np.asarray(sio.load_theta_set(p, table, condition="FAB")[:])
            out["roundtrip" + ext] = bool(np.allclose(back, theta))
            out["status_ok" + ext] = sio.theta_set_status(p, table, condition="FAB") == "OK"
        p = os.path.join(d, "Theta_Set.zarr")
        renamed = [dict(e) for e in table]; renamed[1]["KEY"] = "fraction_dimer_initial"
        shifted = [dict(e) for e in table]; shifted[0]["PRIOR_RANGE"] = (2.0, 3.5)
        out["refuses_renamed_key"] = refused(p, renamed)
        out["refuses_shifted_bound"] = refused(p, shifted)
        out["refuses_other_condition"] = refused(p, table, condition="INLB")
        out["status_names_reason"] = sio.theta_set_status(p, renamed).startswith("SCHEMA MISMATCH")
        bare = os.path.join(d, "Bare_Theta_Set.zarr")
        z = zarr.open(store=bare, mode="w", shape=theta.shape, chunks=(1, theta.shape[1]), dtype=np.float64)
        z[:, :] = theta
        out["refuses_schemaless_zarr"] = refused(bare, table)
        bare_npy = os.path.join(d, "Bare_Theta_Set.npy")
        np.save(bare_npy, theta)
        out["refuses_npy_without_sidecar"] = refused(bare_npy, table)
        out["status_missing_file"] = sio.theta_set_status(os.path.join(d, "none.zarr"), table) == "MISSING"
    out["ok"] = all(v for k, v in out.items() if isinstance(v, bool))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true",
                        help="also execute the run tier (tiny simulations and renders); needs approval")
    args = parser.parse_args()
    det_out = dict(d1=d1_channels(), d2=d2_diffusion(), d3=d3_transforms(),
                   d4=d4_composition(), d5=d5_stationary(), d6=d6_occupancy(),
                   d7=d7_fission_placement(), d8=d8_conditions(), d9=d9_ranges(),
                   d10=d10_theta_schema())
    for k, v in det_out.items():
        print(f"  [{passed(v['ok'])}] {k}")
    run_out = None
    if args.run:
        with tempfile.TemporaryDirectory() as workdir:
            r1, runs = {}, {}
            for condition in RDS.condition_tokens:      # one 2 s run per condition (its own network)
                r1_c, r5_c, lineage_c, tray_c, species_c = r1_r5_reactive(workdir, condition)
                r1[condition] = r1_c
                runs[condition] = (lineage_c, tray_c, species_c)
                if condition == "INLB":
                    r5, lineage, tray, species_of_rank = r5_c, lineage_c, tray_c, species_c
            r1["ok"] = all(v["ok"] for k, v in r1.items() if k != "ok")
            r3 = r3_visible(lineage, tray, species_of_rank)      # the INLB lineage exercises every channel
            r4 = r4_both_conditions(runs)
            r2 = r2_stationarity(workdir)
            r6 = r6_timing(workdir)
            run_out = dict(r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6)
            for k in ("r1", "r2", "r3", "r4", "r5", "r6"):
                print(f"  [{passed(run_out[k]['ok'])}] {k}")
    write_report(det_out, run_out)
    all_ok = all(v["ok"] for v in det_out.values()) and (run_out is None or all(v["ok"] for v in run_out.values()))
    print("report:", REPORT)
    print("OVERALL:", passed(all_ok))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
