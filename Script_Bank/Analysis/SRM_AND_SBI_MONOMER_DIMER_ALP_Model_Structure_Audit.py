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
                       the linear initial dimer fraction passes through untouched at 0 and 1;
                       a blanket 10**u would differ from `to_physical` exactly on that row.
    D4 composition     `realize_initial_composition` at x_B = 0 and 1 for totals 1..40 (odd and
                       even): conservation N_R = n_A + 2 n_B, n_B <= floor(N_R/2), requested vs
                       realized fraction as documented; the association reference equals the
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
                       condition token and refuses to resolve without one.

Run tier (``--run``; tiny; after approval only):
    R1 conservation    One 1 s run per condition at the prior center (MET-INLB: the full
                       reactive network; MET-FAB: dissociation and switching only): the lineage
                       replays with every subunit covered once per frame (fail-loud extractor)
                       and its subunit count equals the realized N_R.
    R2 stationarity    The isolated switching chain under the MET-FAB configuration (no
                       association channel by construction) with monomers only, 10 s: the
                       time-averaged mode occupancies over the second half agree with
                       `stationary_mode_law` within a prespecified tolerance.
    R3 visible         Static labeling draws on the MET-INLB lineage (every channel exercised)
                       at a declared per-species occupancy reproduce the visible fractions
                       p_occ (1 - P0) (monomers) and 1 - (1 - p_occ (1 - P0))^2 (dimers) under
                       both labeling laws.
    R4 both conditions One short video per condition rendered from THAT condition's own
                       trajectory through the production renderer; both carry signal.
    R5 boundary        In the MET-INLB run, particles may lie outside the imaged box (open
                       lateral boundary) while the per-frame subunit total stays constant;
                       the fraction outside is reported.

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
    j = key_index(RDS.stoichiometry.fraction_dimer_key)
    linear_untouched = bool(np.array_equal(phys[:, j], flow[:, j]))
    endpoints = [float(par.to_physical(np.where(np.arange(len(low)) == j, v, low))[j]) for v in (0.0, 1.0)]
    blanket = np.power(10.0, flow)
    differs_only_on_linear_rows = bool(
        np.all(np.isclose(blanket[:, [i for i in range(len(low)) if i != j]],
                          phys[:, [i for i in range(len(low)) if i != j]], rtol=1e-12))
        and not np.allclose(blanket[:, j], phys[:, j]))
    per_entry_ok = all(
        np.isclose(par.entry_to_flow(e, par.entry_to_physical(e, 0.3)), 0.3) for e in PARAMETERIZATION)
    center_ok = all(np.isclose(par.prior_center(e), e["VALUE"], rtol=1e-6) for e in PARAMETERIZATION)
    return dict(max_roundtrip_error=max_err, linear_row_untouched=linear_untouched,
                linear_endpoints=endpoints, blanket_power_differs_only_on_linear_row=differs_only_on_linear_rows,
                per_entry_roundtrip=per_entry_ok, table_values_are_prior_centers=center_ok,
                ok=(max_err < TOL_ROUNDTRIP and linear_untouched and endpoints == [0.0, 1.0]
                    and differs_only_on_linear_rows and per_entry_ok and center_ok))


def d4_composition() -> dict:
    rows, problems = [], []
    for n in range(1, 41):
        for x in (0.0, 1.0, 0.5, 1.0 / 3.0):
            c = par.realize_initial_composition(float(n), x)
            rows.append((n, x, c.n_monomers, c.n_dimers, c.fraction_realized))
            if c.n_total != n or c.n_monomers + 2 * c.n_dimers != n:
                problems.append(f"N={n}, x={x}: conservation broken ({c})")
            if c.n_dimers > n // 2 or c.n_monomers < 0:
                problems.append(f"N={n}, x={x}: dimer cap violated ({c})")
            if x == 0.0 and c.n_dimers != 0:
                problems.append(f"N={n}, x=0: dimers present")
            if x == 1.0 and c.n_dimers != n // 2:
                problems.append(f"N={n}, x=1: not all pairs dimerized")
            if x == 1.0 and (n % 2 == 1) and c.n_monomers != 1:
                problems.append(f"N={n} odd, x=1: expected one leftover monomer")
            if not np.isclose(c.fraction_realized, 2 * c.n_dimers / n):
                problems.append(f"N={n}, x={x}: realized fraction wrong")
    # rounding of a non-integer total, and the lower guard
    c = par.realize_initial_composition(10.4, 0.5)
    if c.n_total != 10:
        problems.append("N=10.4 did not round to 10")
    c = par.realize_initial_composition(0.2, 0.5)
    if c.n_total != 1 or c.n_dimers != 0:
        problems.append("N=0.2 did not floor to one monomer")
    bad = False
    try:
        par.realize_initial_composition(10, 1.2)
    except ValueError:
        bad = True
    if not bad:
        problems.append("x_B outside [0, 1] was accepted")
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
    return dict(tokens=list(tokens), tokens_agree_across_registries=tokens_agree, fab_ratio_is_zero=fab_zero,
                inlb_ratio_is_one=inlb_one, association_row_absent=row_absent, disguised_zero_refused=epsilon_refused,
                bare_rds_alias_refused=bare_refused, tier_aliases=aliases, detector_reads_same_tier=det_alias_ok,
                ok=tokens_agree and fab_zero and inlb_one and row_absent and epsilon_refused and bare_refused
                and alias_ok and det_alias_ok)


# ----------------------------------------------------------------------------------------------
# Run tier (tiny simulations; only with --run)
# ----------------------------------------------------------------------------------------------

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
    tray, timing = _run(theta, condition, 1.0, workdir, f"structure_audit_center_{condition}.h5")
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
    theta[key_index(RDS.stoichiometry.fraction_dimer_key)] = 0.0       # monomers only
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
    rng = np.random.default_rng(SEED + 3)
    sto = RDS.stoichiometry
    occ_map = {sto.monomer.name: 0.8, sto.dimer.name: 0.6}
    initial_species = [species_of_rank[int(r)] for r in lineage.host_rank[0]]
    p_sub = lab.occupancy_per_subunit(occ_map, initial_species)
    mono_ranks = rds.monomer_ranks(tray)
    out = {}
    for condition in lab.LABELING_CONDITIONS:
        name, law = lab.resolve_labeling_law(condition)
        rows = np.stack([lab.labeling_summary(
            lab.draw_dye_counts(law, lineage.n_subunits, rng, occupancy=p_sub),
            lineage.host_index[0], lineage.host_rank[0], mono_ranks) for _ in range(N_LABEL_DRAWS)])
        col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
        mono_vis = rows[:, col["monomers_visible_0"]].sum() / max(rows[:, col["monomers_0"]].sum(), 1)
        dim_vis = rows[:, col["dimers_visible_0"]].sum() / max(rows[:, col["dimers_0"]].sum(), 1)
        q_mono = occ_map[sto.monomer.name] * law.visible_probability
        q_dim_sub = occ_map[sto.dimer.name] * law.visible_probability
        expect_mono, expect_dim = q_mono, 1 - (1 - q_dim_sub) ** 2
        out[condition] = dict(law=name, monomers_0=int(rows[0, col["monomers_0"]]), dimers_0=int(rows[0, col["dimers_0"]]),
                              visible_monomer_emp=float(mono_vis), visible_monomer=expect_mono,
                              visible_dimer_emp=float(dim_vis), visible_dimer=expect_dim,
                              ok=bool(abs(mono_vis - expect_mono) <= TOL_VISIBLE and abs(dim_vis - expect_dim) <= TOL_VISIBLE))
    out["ok"] = all(v["ok"] for k, v in out.items() if k != "ok")
    return out


def r4_both_conditions(runs: dict) -> dict:
    """``runs``: condition -> (lineage, tray) of THAT condition's own trajectory."""
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video
    rng = np.random.default_rng(SEED + 4)
    imaging = np.array([10 ** ((e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2) for e in det.DETECTOR_IMAGING])
    out = {}
    for condition in lab.LABELING_CONDITIONS:
        lineage, tray = runs[condition]
        poses = rds.collapse_species_axis(rds.extract_trajectory_poses(tray))[:20]
        _, law = lab.resolve_labeling_law(condition)
        dyes = lab.draw_dye_counts(law, lineage.n_subunits, rng)
        frames = render_dli_video(poses, lineage.host_index[:20], dyes, imaging, seed=SEED)
        out[condition] = dict(frames=int(frames.shape[2]), n_dyes=int(dyes.sum()), pixel_max=float(frames.max()),
                              own_trajectory=True,
                              ok=bool(frames.shape[2] == 20 and frames.max() > 0 and np.isfinite(frames).all()))
    out["ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict))
    return out


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
             f"linear row untouched: {d['d3']['linear_row_untouched']} (endpoints {d['d3']['linear_endpoints']}); "
             f"blanket 10**u differs only on the linear row: {d['d3']['blanket_power_differs_only_on_linear_row']} |")
    L.append(f"| D4 initial composition and association reference | {passed(d['d4']['ok'])} | {d['d4']['n_cases']} "
             f"(N, x_B) cases incl. odd totals at x_B = 0 and 1; problems: {d['d4']['problems'] or 'none'}; "
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
             f"{d['d8']['detector_reads_same_tier']} |")
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
            L.append(f"| R3 visible fractions {c} | {passed(v['ok'])} | monomer {v['visible_monomer_emp']:.3f} vs "
                     f"{v['visible_monomer']:.3f}; dimer {v['visible_dimer_emp']:.3f} vs {v['visible_dimer']:.3f} |")
        L.append(f"| R4 both conditions, each from its own trajectory | {passed(r['r4']['ok'])} | "
                 + "; ".join(f"{c}: {r['r4'][c]['frames']} frames, {r['r4'][c]['n_dyes']} dyes, max {r['r4'][c]['pixel_max']:.0f}"
                             for c in lab.LABELING_CONDITIONS) + " |")
        L.append(f"| R5 boundary | {passed(r['r5']['ok'])} | particles outside the field at the last frame: "
                 f"{r['r5']['particles_outside_last_frame']}; mean fraction outside {r['r5']['fraction_outside_mean']:.4f}; "
                 f"subunit total constant: {r['r5']['subunit_total_constant']} |")
        L.append("")
    with open(REPORT, "w") as handle:
        handle.write("\n".join(L) + "\n")
    with open(os.path.join(OUT_DIR, "audit_summary.json"), "w") as handle:
        json.dump(dict(deterministic=det_out, run=run_out), handle, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true",
                        help="also execute the run tier (tiny simulations and renders); needs approval")
    args = parser.parse_args()
    det_out = dict(d1=d1_channels(), d2=d2_diffusion(), d3=d3_transforms(),
                   d4=d4_composition(), d5=d5_stationary(), d6=d6_occupancy(),
                   d7=d7_fission_placement(), d8=d8_conditions())
    for k, v in det_out.items():
        print(f"  [{passed(v['ok'])}] {k}")
    run_out = None
    if args.run:
        with tempfile.TemporaryDirectory() as workdir:
            r1, runs = {}, {}
            for condition in RDS.condition_tokens:      # one 1 s run per condition (its own network)
                r1_c, r5_c, lineage_c, tray_c, species_c = r1_r5_reactive(workdir, condition)
                r1[condition] = r1_c
                runs[condition] = (lineage_c, tray_c)
                if condition == "INLB":
                    r5, lineage, tray, species_of_rank = r5_c, lineage_c, tray_c, species_c
            r1["ok"] = all(v["ok"] for k, v in r1.items() if k != "ok")
            r3 = r3_visible(lineage, tray, species_of_rank)      # the INLB lineage exercises every channel
            r4 = r4_both_conditions(runs)
            r2 = r2_stationarity(workdir)
            run_out = dict(r1=r1, r2=r2, r3=r3, r4=r4, r5=r5)
            for k in ("r1", "r2", "r3", "r4", "r5"):
                print(f"  [{passed(run_out[k]['ok'])}] {k}")
    write_report(det_out, run_out)
    all_ok = all(v["ok"] for v in det_out.values()) and (run_out is None or all(v["ok"] for v in run_out.values()))
    print("report:", REPORT)
    print("OVERALL:", passed(all_ok))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
