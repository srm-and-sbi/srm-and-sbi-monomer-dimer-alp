"""Model-structure audit: the separated stoichiometry-mobility generator against its specification.

Two tiers. The DETERMINISTIC tier runs without a simulation, imports only the package, and is
the correctness check of the model STRUCTURE (the model specification's implementation
increment, 2026-09-09). The RUN tier (``--run``) executes tiny, short simulations and renders;
it is compute and runs only when explicitly requested after approval.

Deterministic tier (always):
    D1 channels        `reaction_channels` yields, PER CONDITION, exactly the declared channels of
                       the retained-population model: MET-INLB (association ratio 1, ligand classes)
                       twenty-four -- six association fusions A + A -> B2 (one per unordered pair of
                       monomer modes, product in the SLOWER parent's mode, one shared microscopic
                       rate), three fissions B2 -> A + A, three conversions B1 -> A at kappa_OFF (the
                       probe-free partner is not represented), twelve switching conversions (the
                       four shared rates, once per species A, B1, B2, adjacent modes only); MET-FAB
                       (association ratio 0) eleven -- NO fusion channel at all (structural, not a
                       tiny rate), three fissions B -> A + A, eight switching conversions. Every
                       channel conserves the retained subunit count. `build_system` registers
                       exactly these names under each condition.
    D2 diffusion       D[X, m] = D_A * species_factor * mode_factor for every type; for every
                       prior draw D_immobile < D_slow <= D_fast within a species and
                       D_dimer <= D_monomer within a mode (the disjoint-range guarantee).
    D3 transforms      `to_flow(to_physical(u)) == u` on prior draws INCLUDING the box corners;
                       every row of the decided table is a log row, so a blanket 10**u equals
                       `to_physical` everywhere (the per-row rule stays the contract); every
                       table VALUE is its row's prior center.
    D4 composition     `realize_initial_composition` from the TRUE total N_total and the true
                       dimer-to-monomer ratio r, per condition, over the box (1,000 .. 100,000) and
                       the band (f_B 5-25 %) plus r in {0, huge}: the retained integers equal the
                       rounded expectations, dimers and classes first (n_B = round(p (2-p) n_B,true),
                       n_B2 = round(p^2 n_B,true), n_A = round(p n_A,true)); the retained count is
                       n_A + 2 n_B (FAB) or n_A + n_B1 + 2 n_B2 (INLB) and lies within rounding of
                       p N_total [1 + 2(1-p) f_B/(1+f_B)] (FAB) or p N_total (INLB); monotone in r;
                       the true x_B = 2r/(1+2r) and f_B = r/(1+r) as documented; at least one
                       monomer and one dimer at the count floor and the band's lower edge; the
                       ceiling places about 4,762 (INLB) and 1,600-2,100 (FAB) retained subunits; a
                       negative ratio or a non-positive total rejected; the association reference
                       equals the Smoluchowski expression 4 pi (2 D_A) r / ((4/3) pi r^3) =
                       6 D_A / r^2, and under MET-INLB every fusion fires at exactly lambda_ref.
    D5 stationary law  `stationary_mode_law` sums to one and satisfies detailed balance on
                       every link of the chain.
    D6 probe rule      `rank_to_species` maps every monomer type to the stoichiometric class A and
                       every dimer type (B, B1, B2) to B, `rank_to_molecule` keeps the ligand class;
                       the probe rule of `labeling.assign_probes` binds every retained monomer, every
                       subunit of a B1 / B2 host, and one or two subunits of a FAB dimer at the
                       declared two-probe share; an override's per-class coins give identical
                       probabilities within a class, and a coin keyed by a particle type is rejected.
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
                       condition token and refuses to resolve without one. Occupancies of the TRUE
                       population: MET-INLB 0.0476 (EQUILIBRIUM at 0.25 nM with K_D 5 nM), MET-FAB
                       0.0148 (DERIVED from the declared Fab/InlB visibility ratio 0.5 and the InlB
                       anchor), visibilities 0.0238 / 0.0119, two-probe shares 0.0244 / 0.0074;
                       MET-INLB carries the ligand classes and MET-FAB does not; a setting with more
                       or fewer than one occupancy source, or with association but no ligand
                       classes, is refused.
    D9 ranges          The decided prior ranges (2026-09-14; count and composition rows 2026-10-05)
                       as a literal table equal the parameterization's; the immobile factor keeps
                       D_i = R_i D_A below the pipelines' looser immobility threshold (0.0065 um^2/s)
                       at the D_A center for every R_i in range (immobility by construction) and a
                       full decade below the slow factor; the dissociation floor loses <= 2% of
                       dimers over 20 s; the composition box maps to the 5-25 % band of the true
                       basal complex fraction and the count box is [3.0, 5.0].
    D10 theta schema   A Theta_Set written by `io.write_theta_set` (.zarr and .npy) carries the
                       table's schema and reads back through `io.load_theta_set`; a table with a
                       renamed key, a table with a shifted prior bound, a different condition, a
                       schema-less .zarr, and a .npy without its sidecar are each REFUSED
                       (`ThetaSetSchemaError`), and `theta_set_status` names the reason.
    D11 conversion     The `B1 -> A` release is a CONVERSION for every mode (the product keeps the
        displacement    educt's position) and no channel re-forms `B1`, so a `B1` lineage sees at most
                       one such event; the per-event difference from a fission placement is one
                       daughter kick of half the product distance (one reaction distance, 10 nm, at a
                       fixed 3D distance in a random direction). That kick is compared with the
                       per-frame diffusive RMS displacement of a monomer in each mode at the prior
                       FLOOR of `D_A` and of the relative factors, with the localization precision of
                       the MET recordings (29 nm) and with the pixel; it passes when it is below the
                       precision and below a tenth of a pixel. R7 measures both placements.

Run tier (``--run``; tiny; after approval only):
    R1 conservation    One 2 s run per condition at the prior center (the reference recording length) (MET-INLB: the full
                       reactive network with the ligand classes; MET-FAB: dissociation and switching
                       only): the lineage replays with every retained subunit covered once per frame
                       (fail-loud extractor) and its subunit count equals the realized retained N_R.
    R2 stationarity    The isolated switching chain under the MET-FAB configuration (no
                       association channel by construction) with monomers only, 10 s: the
                       time-averaged mode occupancies over the second half agree with
                       `stationary_mode_law` within a prespecified tolerance.
    R3 visible         Static labeling draws on each condition's OWN lineage through the shared
                       path (`resolve_labeling` + `label_subunits`) reproduce the retained-population
                       visible fractions: monomers q = 1 - P0; dimers (1 - s_2) q + s_2 (1 - (1-q)^2)
                       under FAB, and under INLB q for B1 hosts and 1 - (1-q)^2 for B2 hosts, with
                       the probe classes recorded.
    R4 both conditions One short video per condition rendered from THAT condition's own
                       trajectory through the production renderer; both carry signal.
    R5 boundary        In the MET-INLB run, particles may lie outside the imaged box (open
                       lateral boundary) while the per-frame subunit total stays constant;
                       the fraction outside is reported.
    R6 timing          Wall time and peak memory of one 2 s run per condition at the simulated-population
                       ceiling (N_total = 100,000: about 4,762 retained subunits under MET-INLB and
                       about 2,100 under MET-FAB), descriptive, plus a 20-frame render of each.
    R7 placement       One 0.2 s MET-INLB run with diffusion switched almost off (D_A = 1e-6 um^2/s)
                       and kappa_OFF at the prior ceiling, N_total at the box ceiling and r at the
                       band ceiling (most B1): the in-plane displacement of every `B1 -> A` product
                       from its educt's position at the frame before (expected zero: in place) and of
                       every fission daughter from its parent's (expected a fixed 10 nm in 3D, pi/4 x
                       10 nm in-plane on average), read from the reaction records and the frames.

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
    """Per condition: the generated channels against the declared network (24 under INLB, 11 under FAB)."""
    out = {c: _d1_condition(c) for c in RDS.condition_tokens}
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def _d1_condition(condition: str) -> dict:
    theta = prior_center_theta()
    channels = rds.reaction_channels(theta, condition)
    sto, mob = RDS.stoichiometry, RDS.mobility
    mono, dim = sto.monomer.name, sto.dimer.name
    species = RDS.species_for(condition)
    dimers = [s for s in species if s.stoichiometry == "dimer"]
    product_dimer = next(s.name for s in dimers if s.dissociation == "fission")
    one_probe = [s.name for s in dimers if s.dissociation == "conversion"]
    names = [c.name for c in channels]
    kinds = {k: sum(c.kind == k for c in channels) for k in ("fusion", "fission", "conversion")}
    n_modes = len(mob.modes)
    r_on = RDS.association_ratio_of(condition)
    expected_fusions = n_modes * (n_modes + 1) // 2 if r_on > 0.0 else 0
    expected_fissions = n_modes * sum(s.dissociation == "fission" for s in dimers)
    expected_conversions = len(mob.switching) * len(species) + n_modes * len(one_probe)
    sub = dict(zip(RDS.particle_type_names, RDS.subunit_counts_per_type))
    problems = []
    seen_pairs = set()
    lamb_on = {c.rate for c in channels if c.kind == "fusion"}
    kappa_off = rds.theta_by_key(theta)[sto.dissociation_rate_key]
    for c in channels:
        if sum(sub[e] for e in c.educts) != sum(sub[p] for p in c.products):
            problems.append(f"{c.name}: the retained subunit count is not conserved")
        if c.kind == "fusion":
            m1, m2 = (RDS.mode_of_type[e] for e in c.educts)
            if any(RDS.species_of_type[e] != mono for e in c.educts):
                problems.append(f"{c.name}: an association educt is not a monomer")
            if RDS.molecule_of_type[c.products[0]] != product_dimer:
                problems.append(f"{c.name}: association product is not the two-probe dimer class")
            if RDS.mode_of_type[c.products[0]] != mob.slower(m1, m2):
                problems.append(f"{c.name}: product mode is not the slower parent's")
            seen_pairs.add(frozenset((m1, m2)) if m1 != m2 else frozenset((m1,)))
        elif c.kind == "fission":
            m = RDS.mode_of_type[c.educts[0]]
            if RDS.species_of_type[c.educts[0]] != dim or RDS.molecule_of_type[c.educts[0]] in one_probe:
                problems.append(f"{c.name}: dissociation educt is not a two-subunit dimer class")
            if any(RDS.species_of_type[p] != mono or RDS.mode_of_type[p] != m for p in c.products):
                problems.append(f"{c.name}: daughters are not monomers in the dimer's mode")
        elif c.kind == "conversion":
            e, p = c.educts[0], c.products[0]
            if RDS.molecule_of_type[e] in one_probe and RDS.species_of_type[p] == mono:
                # B1 -> A: one eligible daughter at kappa_OFF, the mode conserved
                if RDS.mode_of_type[e] != RDS.mode_of_type[p] or c.rate_key != sto.dissociation_rate_key \
                        or not np.isclose(c.rate, kappa_off):
                    problems.append(f"{c.name}: the one-probe dissociation must keep the mode at kappa_OFF")
                continue
            if RDS.molecule_of_type[e] != RDS.molecule_of_type[p]:
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
        expected=dict(fusion=expected_fusions, fission=expected_fissions, conversion=expected_conversions,
                      total=expected_fusions + expected_fissions + expected_conversions),
        particle_types=list(RDS.particle_type_names_for(condition)),
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
    band_r = tuple(f / (1 - f) for f in par.COMPOSITION_BAND)
    for condition in RDS.condition_tokens:
        p = par.occupancy_of(condition)
        classes = RDS.condition_setting(condition).ligand_classes
        for n in (1e3, 3.16e3, 1e4, 3.16e4, 1e5):
            previous = -1
            for r in (0.0, band_r[0], 0.1325, band_r[1], 1e9):
                c = par.realize_initial_composition(n, r, condition)
                n_a_true, n_b_true = n / (1 + 2 * r), r * n / (1 + 2 * r)
                rows.append((condition, n, r, c.n_monomers, c.n_dimers, c.n_dimers_one_probe, c.n_dimers_two_probe,
                             c.n_subunits))
                if c.n_dimers != round(p * (2 - p) * n_b_true) or c.n_monomers != round(p * n_a_true):
                    problems.append(f"{condition} N={n}, r={r}: integers are not the rounded expectations ({c})")
                if c.n_dimers_two_probe != min(round(p * p * n_b_true), c.n_dimers) or \
                        c.n_dimers_one_probe + c.n_dimers_two_probe != c.n_dimers:
                    problems.append(f"{condition} N={n}, r={r}: probe classes do not split the dimers ({c})")
                expected_subunits = (c.n_monomers + c.n_dimers_one_probe + 2 * c.n_dimers_two_probe if classes
                                     else c.n_monomers + 2 * c.n_dimers)
                if c.n_subunits != expected_subunits:
                    problems.append(f"{condition} N={n}, r={r}: retained count rule broken ({c})")
                if abs(c.n_subunits - c.expected_subunits) > 2.5:
                    problems.append(f"{condition} N={n}, r={r}: retained count {c.n_subunits} far from its "
                                    f"expectation {c.expected_subunits:.1f}")
                if c.n_dimers < previous:
                    problems.append(f"{condition} N={n}, r={r}: dimers not monotone in r")
                previous = c.n_dimers
                if r == 0.0 and c.n_dimers != 0:
                    problems.append(f"{condition} N={n}, r=0: dimers present")
                if not np.isclose(c.receptor_fraction_true, 2 * r / (1 + 2 * r)) or \
                        not np.isclose(c.complex_fraction_true, r / (1 + r)):
                    problems.append(f"{condition} N={n}, r={r}: derived true fractions wrong")
                if not np.isclose(c.ratio_retained_expected, (2 - p) * r) or not np.isclose(c.two_probe_share, p / (2 - p)):
                    problems.append(f"{condition} N={n}, r={r}: retained ratio or two-probe share wrong")
        floor = par.realize_initial_composition(1e3, band_r[0], condition)
        if floor.n_dimers < 1 or floor.n_monomers < 1:
            problems.append(f"{condition}: the count floor and the band's lower edge place no monomer or no dimer")
        ceiling = par.realize_initial_composition(1e5, band_r[1], condition).n_subunits
        lo, hi = (4740, 4785) if classes else (2000, 2100)
        if not lo <= ceiling <= hi:
            problems.append(f"{condition}: {ceiling} retained subunits at the ceiling, outside {lo}-{hi}")
    if not np.isclose(float(par.ratio_to_complex_fraction(1.0)), 0.5) or \
            not np.isclose(float(par.ratio_to_receptor_fraction(1.0)), 2.0 / 3.0) or \
            not np.isclose(float(par.receptor_fraction_to_ratio(2.0 / 3.0)), 1.0):
        problems.append("ratio <-> fraction helpers disagree with r=1 <-> f_B=1/2 <-> x_B=2/3")
    for bad in ((10, -0.5, "FAB"), (0.0, 0.1, "FAB"), (100, 0.1, "XYZ")):
        try:
            par.realize_initial_composition(*bad)
        except (ValueError, KeyError):
            continue
        problems.append(f"{bad} was accepted")
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
                ceiling_retained_subunits={c: par.realize_initial_composition(1e5, band_r[1], c).n_subunits
                                           for c in RDS.condition_tokens},
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
    molecule_of_rank = rds.rank_to_molecule(fake_tray)
    mono_ranks = rds.monomer_ranks(fake_tray)
    sto = RDS.stoichiometry
    by_species_ok = all(
        species_of_rank[fake_tray.particle_types[RDS.type_name(sp.name, m)]] == (sto.monomer.name if sp.stoichiometry == "monomer" else sto.dimer.name)
        and molecule_of_rank[fake_tray.particle_types[RDS.type_name(sp.name, m)]] == sp.name
        for sp in sto.species for m in RDS.mobility.modes)
    mono_ok = sorted(mono_ranks) == sorted(
        fake_tray.particle_types[n] for n in RDS.particle_type_names if RDS.species_of_type[n] == sto.monomer.name)
    # the model's probe rule on a synthetic frame-0 lineage: 60 A, 40 B (FAB) / 30 B1 + 10 B2 (INLB)
    rng = np.random.default_rng(SEED + 6)
    rule_ok = True
    details = {}
    for condition in RDS.condition_tokens:
        plan = lab.resolve_labeling(condition)
        if plan.ligand_classes:
            hosts = [("A_f", 1)] * 60 + [("B1_s", 1)] * 30 + [("B2_i", 2)] * 10
        else:
            hosts = [("A_f", 1)] * 60 + [("B_s", 2)] * 40
        host_index = np.repeat(np.arange(len(hosts)), [n for _, n in hosts])
        host_rank = np.repeat([fake_tray.particle_types[name] for name, _ in hosts], [n for _, n in hosts])
        bound = lab.assign_probes(plan, host_index, host_rank, species_of_rank, rng)
        per_host = np.bincount(host_index, weights=bound.astype(float))
        monomers_bound = bool(np.all(per_host[:60] == 1))
        if plan.ligand_classes:
            classes_ok = bool(np.all(per_host[60:90] == 1) and np.all(per_host[90:] == 2))
        else:
            classes_ok = bool(np.all((per_host[60:] == 1) | (per_host[60:] == 2)))
        details[condition] = dict(monomers_bound=monomers_bound, classes_ok=classes_ok,
                                  two_probe_share=plan.two_probe_share, probe_rule=plan.probe_rule)
        rule_ok &= monomers_bound and classes_ok and plan.probe_rule == "classes"
    # an override's coins: identical within a stoichiometric class; keyed by a particle type refused
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
    return dict(rank_to_species_by_class=by_species_ok, monomer_ranks_are_all_monomer_modes=mono_ok,
                probe_rule=details, probe_rule_ok=bool(rule_ok),
                occupancy_identical_within_species=same_within_species,
                occupancy_keyed_by_particle_type_rejected=rejected,
                legacy_species_trajectory_rejected=unknown_type_rejected,
                ok=by_species_ok and mono_ok and rule_ok and same_within_species and rejected and unknown_type_rejected)


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
    # Occupancies of the true population (2026-10-05): INLB = the equilibrium 0.25 / (0.25 + 5) = 0.0476;
    # FAB derived = 0.5 x (0.0476 x 0.5) / 0.806 = 0.0148. Ligand classes under INLB only.
    occ = {c: par.occupancy_of(c) for c in tokens}
    vis = {c: par.visibility_of(c) for c in tokens}
    src = {c: par.occupancy_source_of(c) for c in tokens}
    s2 = {c: par.two_probe_share_of(c) for c in tokens}
    fab_setting, inlb_setting = RDS.condition_setting("FAB"), RDS.condition_setting("INLB")
    fab_derived_ok = (src["FAB"] == "derived" and fab_setting.visibility_ratio == 0.5
                      and fab_setting.visibility_ratio_to == "INLB" and not fab_setting.ligand_classes
                      and abs(occ["FAB"] - 0.5 * vis["INLB"] / RDS.dye_probability_of("FAB")) < 1e-12
                      and abs(occ["FAB"] - 0.0148) < 5e-5 and abs(vis["FAB"] - occ["FAB"] * RDS.dye_probability_of("FAB")) < 1e-12
                      and abs(s2["FAB"] - occ["FAB"] / (2 - occ["FAB"])) < 1e-12)
    inlb_declared_ok = (src["INLB"] == "equilibrium" and inlb_setting.ligand_classes
                        and inlb_setting.probe_concentration_nm == 0.25 and inlb_setting.dissociation_constant_nm == 5.0
                        and abs(occ["INLB"] - 0.25 / 5.25) < 1e-12 and abs(vis["INLB"] - 0.5 * 0.25 / 5.25) < 1e-12)
    both_refused = neither_refused = False
    try:
        par.ConditionSetting("X", 1.0, ligand_classes=True, occupancy=0.5, visibility_ratio=0.5, visibility_ratio_to="INLB")
    except ValueError:
        both_refused = True
    try:
        par.ConditionSetting("X", 1.0, ligand_classes=True)
    except ValueError:
        neither_refused = True
    no_classes_refused = False
    try:
        par.ConditionSetting("X", 1.0, dissociation_constant_nm=5.0)
    except ValueError:
        no_classes_refused = True
    return dict(tokens=list(tokens), tokens_agree_across_registries=tokens_agree, fab_ratio_is_zero=fab_zero,
                inlb_ratio_is_one=inlb_one, association_row_absent=row_absent, disguised_zero_refused=epsilon_refused,
                bare_rds_alias_refused=bare_refused, tier_aliases=aliases, detector_reads_same_tier=det_alias_ok,
                occupancy={c: round(v, 5) for c, v in occ.items()}, visibility=vis, occupancy_source=src,
                two_probe_share={c: round(v, 5) for c, v in s2.items()},
                ligand_classes={c: RDS.condition_setting(c).ligand_classes for c in tokens},
                fab_occupancy_derived_ok=fab_derived_ok, inlb_occupancy_equilibrium_ok=inlb_declared_ok,
                both_and_neither_refused=both_refused and neither_refused,
                association_without_classes_refused=no_classes_refused,
                ok=tokens_agree and fab_zero and inlb_one and row_absent and epsilon_refused and bare_refused
                and alias_ok and det_alias_ok and fab_derived_ok and inlb_declared_ok and both_refused
                and neither_refused and no_classes_refused)


# The decided prior ranges (2026-09-14; count and composition rows 2026-10-05), as a second, literal
# copy: a table edit that is not a decision fails D9. Estimator coordinate = log10 of the physical
# value for every row. The composition box is log10(0.05/0.95) .. log10(0.25/0.75), the 5-25 % band.
DECIDED_RANGES = {
    "count_total": (3.0, 5.0), "ratio_dimer_monomer_initial": (-1.2787536009528289, -0.47712125471966244),
    "rate_dissociation": (-3.0, 1.0),
    "diffusivity_alp": (-1.25, -0.25), "relative_diffusivity_dimer": (-1.0, 0.0),
    "relative_diffusivity_slow": (-1.0, 0.0), "relative_diffusivity_immobile": (-3.0, -2.0),
    "rate_fast_slow": (-1.0, 1.0), "rate_slow_fast": (-1.0, 1.0), "rate_slow_immobile": (-1.0, 1.0),
    "rate_immobile_slow": (-1.0, 1.0),
}
IMMOBILE_THRESHOLDS_UM2_S = (0.0028, 0.0065)     # the tracking pipelines' immobility thresholds


def d9_ranges() -> dict:
    """The decided ranges, immobility by construction, the dissociation floor, the composition band."""
    table = {e["KEY"]: tuple(float(v) for v in e["PRIOR_RANGE"]) for e in PARAMETERIZATION}
    decided = {k: tuple(float(x) for x in v) for k, v in DECIDED_RANGES.items()}
    ranges_match = (set(table) == set(decided)
                    and all(np.allclose(table[k], decided[k], atol=1e-12) for k in decided))
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
    band = tuple(float(par.ratio_to_complex_fraction(par.entry_to_physical(e_r, b))) for b in e_r["PRIOR_RANGE"])
    band_ok = bool(np.allclose(band, par.COMPOSITION_BAND, atol=1e-9))
    return dict(ranges_match_decided_table=ranges_match, d_i_max_at_d_a_center=r_i_max * d_a_center,
                immobile_below_looser_threshold_at_center=bool(immobile_at_center),
                immobile_below_stricter_threshold_at_d_a_ceiling=bool(immobile_at_ceiling_strict),
                slow_over_immobile_gap=r_s_min / r_i_max, decade_gap=bool(decade_gap),
                dissociation_floor_loss_over_20s=float(loss_20s), composition_band=band, composition_band_ok=band_ok,
                ok=ranges_match and immobile_at_center and decade_gap and loss_20s <= 0.02 and band_ok)


# ----------------------------------------------------------------------------------------------
# Run tier (tiny simulations; only with --run)
# ----------------------------------------------------------------------------------------------

R1_SECONDS = 2.0       # the reference recording length: the run tier is exercised at the main case


def _run(theta: np.ndarray, condition: str, seconds: float, workdir: str, name: str):
    import readdy
    from srm_and_sbi_monomer_dimer_alp.parameterization import RunTiming
    timing = RunTiming(total_time_seconds=seconds, frames=PARAMETERS.simulation.timing)
    stem = rds.build_system(theta, condition)
    smut = rds.build_simulation(stem, theta, condition, seed=SEED)
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
    comp = rds.initial_composition_of(theta, condition)
    species_of_rank = rds.rank_to_species(tray)
    # per-frame retained subunit total from the particles observable (independent of the lineage)
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
    # about 600 RETAINED monomers: the true total that the FAB occupancy thins to 600 (a monomer-only
    # chain needs the particles, not the box; the ratio 0 lies outside the band on purpose)
    theta[key_index(RDS.stoichiometry.count_total_key)] = 600.0 / par.occupancy_of("FAB")
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


def r3_visible(runs: dict) -> dict:
    """Visible fractions of the retained population on each condition's OWN lineage through the shared path."""
    rng = np.random.default_rng(SEED + 3)
    out = {}
    col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
    for condition in lab.LABELING_CONDITIONS:
        lineage, tray, species_of_rank = runs[condition]
        plan = lab.resolve_labeling(condition)
        mono_ranks = rds.monomer_ranks(tray)
        q = plan.law.visible_probability
        rows = np.stack([lab.label_subunits(plan, lineage.host_index[0], lineage.host_rank[0], species_of_rank,
                                            mono_ranks, rng)[1] for _ in range(N_LABEL_DRAWS)])
        n_m, n_d = max(rows[:, col["monomers_0"]].sum(), 1), max(rows[:, col["dimers_0"]].sum(), 1)
        mono_vis = rows[:, col["monomers_visible_0"]].sum() / n_m
        dim_vis = rows[:, col["dimers_visible_0"]].sum() / n_d
        one_probe = rows[:, col["dimers_one_probe_0"]].sum() / n_d
        two_probe = rows[:, col["dimers_two_probe_0"]].sum() / n_d
        expect_mono = q
        expect_dim = one_probe * q + two_probe * (1 - (1 - q) ** 2)        # by the realized class mix
        recorded = bool(np.all(rows[:, col["occupancy_monomer"]] == 1.0)
                        and np.all(rows[:, col["two_probe_share"]] == plan.two_probe_share))
        if plan.ligand_classes:
            classes_ok = bool(np.isclose(one_probe + two_probe, 1.0))      # B1 / B2 by species, every host classed
        else:
            classes_ok = abs(two_probe - plan.two_probe_share) <= max(TOL_VISIBLE, 4 * np.sqrt(plan.two_probe_share / n_d))
        out[condition] = dict(law=plan.law_name, probe_rule=plan.probe_rule, two_probe_share=plan.two_probe_share,
                              monomers_0=int(rows[0, col["monomers_0"]]), dimers_0=int(rows[0, col["dimers_0"]]),
                              one_probe_share_emp=float(one_probe), two_probe_share_emp=float(two_probe),
                              visible_monomer_emp=float(mono_vis), visible_monomer=float(expect_mono),
                              visible_dimer_emp=float(dim_vis), visible_dimer=float(expect_dim), recorded=recorded,
                              ok=bool(abs(mono_vis - expect_mono) <= TOL_VISIBLE and abs(dim_vis - expect_dim) <= TOL_VISIBLE
                                      and recorded and classes_ok))
    out["ok"] = all(v["ok"] for k, v in out.items() if k != "ok")
    return out


def r4_both_conditions(runs: dict) -> dict:
    """``runs``: condition -> (lineage, tray, species_of_rank) of THAT condition's own trajectory.

    Labels the lineage exactly as the DLI stage does (``resolve_labeling`` + ``label_subunits``) and
    renders 20 frames through the production renderer; reports labeled subunits beside the dye total.
    """
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video
    rng = np.random.default_rng(SEED + 4)
    imaging = np.array([10 ** ((e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2) for e in det.DETECTOR_IMAGING])
    out = {}
    for condition in lab.LABELING_CONDITIONS:
        lineage, tray, species_of_rank = runs[condition]
        poses = rds.collapse_species_axis(rds.extract_trajectory_poses(tray))[:20]
        plan = lab.resolve_labeling(condition)
        dyes, row = lab.label_subunits(plan, lineage.host_index[0], lineage.host_rank[0], species_of_rank,
                                       rds.monomer_ranks(tray), rng)
        frames = render_dli_video(poses, lineage.host_index[:20], dyes, imaging, seed=SEED)
        n_labeled = int((dyes >= 1).sum())
        col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
        q = plan.law.visible_probability
        expected_labeled = q * (row[col["monomers_0"]] + row[col["dimers_one_probe_0"]] + 2 * row[col["dimers_two_probe_0"]])
        out[condition] = dict(frames=int(frames.shape[2]), n_subunits=int(lineage.n_subunits),
                              probe_rule=plan.probe_rule, n_labeled=n_labeled,
                              expected_labeled=float(expected_labeled), n_dyes=int(dyes.sum()),
                              pixel_max=float(frames.max()), own_trajectory=True,
                              ok=bool(frames.shape[2] == 20 and frames.max() > 0 and np.isfinite(frames).all()
                                      and n_labeled > 0))
    out["ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict))
    return out


def r6_timing(workdir: str) -> dict:
    """Wall time and peak memory of one 2 s run per condition at the simulated-population ceiling
    (N_total = 100,000, the band's upper edge), plus a 20-frame render of each (descriptive)."""
    import resource
    import time
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video
    imaging = np.array([10 ** ((e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2) for e in det.DETECTOR_IMAGING])
    out = {}
    for condition in RDS.condition_tokens:
        theta = prior_center_theta()
        e_n = PARAMETERIZATION[key_index(RDS.stoichiometry.count_total_key)]
        e_r = PARAMETERIZATION[key_index(RDS.stoichiometry.composition_ratio_key)]
        e_d = PARAMETERIZATION[key_index(RDS.mobility.diffusivity_key)]
        theta[key_index(e_n["KEY"])] = float(par.entry_to_physical(e_n, e_n["PRIOR_RANGE"][1]))   # 100,000 true subunits
        theta[key_index(e_r["KEY"])] = float(par.entry_to_physical(e_r, e_r["PRIOR_RANGE"][1]))   # f_B = 25 %: most retained subunits
        theta[key_index(e_d["KEY"])] = float(par.entry_to_physical(e_d, e_d["PRIOR_RANGE"][1]))   # fastest diffusion
        comp = rds.initial_composition_of(theta, condition)
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        tray, timing = _run(theta, condition, 2.0, workdir, f"structure_audit_timing_{condition}.h5")
        wall = time.perf_counter() - t0
        t1 = time.perf_counter()
        lineage = rds.extract_subunit_lineage(tray)
        positions = rds.extract_subunit_positions(tray, lineage)
        plan = lab.resolve_labeling(condition)
        dyes, _ = lab.label_subunits(plan, lineage.host_index[0], lineage.host_rank[0], rds.rank_to_species(tray),
                                     rds.monomer_ranks(tray), np.random.default_rng(SEED))
        frames = render_dli_video(None, None, dyes, imaging, seed=SEED, subunit_positions=positions[:20])
        render_wall = time.perf_counter() - t1
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        del tray, positions, frames
        out[condition] = dict(condition=condition, seconds_simulated=2.0, n_total=float(comp.n_total_true),
                              n_subunits=comp.n_subunits, n_particles=comp.n_complexes, n_monomers=comp.n_monomers,
                              n_dimers=comp.n_dimers, wall_seconds=float(wall),
                              wall_seconds_per_simulated_second=float(wall / 2.0),
                              lineage_and_render_20_frames_seconds=float(render_wall),
                              peak_rss_mib_before=rss_before / 1024.0, peak_rss_mib_after=rss_after / 1024.0,
                              ok=bool(np.isfinite(wall)))
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def r7_conversion_placement(workdir: str) -> dict:
    """Measured placement of the `B1 -> A` conversion product and of the fission daughters.

    One 0.2 s MET-INLB run with diffusion switched almost off (D_A = 1e-6 um^2/s: 0.3 nm per frame),
    kappa_OFF at the prior ceiling (events within the run), N_total at the box ceiling and r at the
    band ceiling (most B1). For every `B1 -> A` record and every fission record the product's position
    at the first frame at or after the step is compared with the educt's at the last frame before it
    (in-plane distance, nm). Events whose educt or product is not visible at those frames (a mode
    switch within the same frame) are skipped and counted.
    """
    theta = prior_center_theta()
    e_n = PARAMETERIZATION[key_index(RDS.stoichiometry.count_total_key)]
    e_r = PARAMETERIZATION[key_index(RDS.stoichiometry.composition_ratio_key)]
    e_k = PARAMETERIZATION[key_index(RDS.stoichiometry.dissociation_rate_key)]
    theta[key_index(e_n["KEY"])] = float(par.entry_to_physical(e_n, e_n["PRIOR_RANGE"][1]))
    theta[key_index(e_r["KEY"])] = float(par.entry_to_physical(e_r, e_r["PRIOR_RANGE"][1]))
    theta[key_index(e_k["KEY"])] = float(par.entry_to_physical(e_k, e_k["PRIOR_RANGE"][1]))
    theta[key_index(RDS.mobility.diffusivity_key)] = 1e-6
    condition = "INLB"
    comp = rds.initial_composition_of(theta, condition)
    tray, timing = _run(theta, condition, 0.2, workdir, "structure_audit_placement_INLB.h5")
    _times, _types, ids, poses = tray.read_observable_particles()
    rec_times, rec_lists = tray.read_observable_reactions()
    steps_per_frame = PARAMETERS.simulation.timing.steps_per_frame
    pos_by_frame = [dict(zip((int(i) for i in np.atleast_1d(fid)), np.asarray(fpos, dtype=float)))
                    for fid, fpos in zip(ids, poses)]
    n_frames = len(pos_by_frame)
    b1, mono = RDS.stoichiometry.dimer_one_probe.name, RDS.stoichiometry.monomer.name
    dist = {"fission": [], "conversion": []}
    skipped = {"fission": 0, "conversion": 0}
    for t, records in zip(rec_times, rec_lists):
        step = int(t)
        f_before = (step - 1) // steps_per_frame
        f_after = step // steps_per_frame + (1 if step % steps_per_frame else 0)
        if f_before < 0 or f_after >= n_frames:
            continue
        for rec in records:
            kind, label = str(rec.type), str(rec.reaction_label)
            if kind == "fission":
                key = "fission"
            elif kind == "conversion" and " => " in label:
                educt_species, product_species = (s.split("_")[0] for s in label.split(" => "))
                if educt_species != b1 or product_species != mono:
                    continue                                   # a mode switch, not the release
                key = "conversion"
            else:
                continue
            educt = int(np.atleast_1d(rec.educts)[0])
            products = [int(x) for x in np.atleast_1d(rec.products)]
            e_pos = pos_by_frame[f_before].get(educt)
            p_pos = [pos_by_frame[f_after].get(p) for p in products]
            if e_pos is None or any(p is None for p in p_pos):
                skipped[key] += 1
                continue
            dist[key].extend(float(np.linalg.norm((p - e_pos)[:2])) for p in p_pos)
    del tray
    kick = float(PARAMETERS.simulation.stem.fission_product_distance_nm) / 2.0
    out = dict(condition=condition, seconds_simulated=0.2, n_b1_initial=comp.n_dimers_one_probe,
               n_b2_initial=comp.n_dimers_two_probe, expected_fission_in_plane_mean_nm=float(np.pi / 4 * kick),
               expected_fission_3d_nm=kick)
    for key, vals in dist.items():
        v = np.asarray(vals, dtype=float)
        out[key] = dict(n=int(v.size), skipped=skipped[key],
                        mean_nm=float(v.mean()) if v.size else float("nan"),
                        sd_nm=float(v.std()) if v.size else float("nan"),
                        max_nm=float(v.max()) if v.size else float("nan"))
    f, c = out["fission"], out["conversion"]
    enough = f["n"] >= 20 and c["n"] >= 100
    in_place = enough and c["max_nm"] < 1.0
    tol = 5.0 * f["sd_nm"] / np.sqrt(f["n"]) if f["n"] else float("inf")
    fission_as_declared = enough and (abs(f["mean_nm"] - out["expected_fission_in_plane_mean_nm"]) <= tol
                                      and f["max_nm"] <= kick + 1.0)
    out.update(enough_events=enough, conversion_in_place=in_place, fission_as_declared=fission_as_declared,
               ok=bool(enough and in_place and fission_as_declared))
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
             f"visibilities { {c: round(v, 5) for c, v in d['d8']['visibility'].items()} }, two-probe shares "
             f"{d['d8']['two_probe_share']}, ligand classes {d['d8']['ligand_classes']}; FAB derived from the "
             f"declared ratio and the INLB anchor: {d['d8']['fab_occupancy_derived_ok']}; INLB equilibrium at 0.25 nM / K_D 5 nM: "
             f"{d['d8']['inlb_occupancy_equilibrium_ok']}; both/neither refused: {d['d8']['both_and_neither_refused']}; "
             f"association without classes refused: {d['d8']['association_without_classes_refused']} |")
    L.append(f"| D9 decided ranges | {passed(d['d9']['ok'])} | table equals the decided ranges: {d['d9']['ranges_match_decided_table']}; "
             f"D_i max at the D_A center {d['d9']['d_i_max_at_d_a_center']:.4f} um^2/s below the looser immobility threshold: "
             f"{d['d9']['immobile_below_looser_threshold_at_center']} (below the stricter one at the D_A ceiling: "
             f"{d['d9']['immobile_below_stricter_threshold_at_d_a_ceiling']}, reported); slow/immobile gap "
             f"{d['d9']['slow_over_immobile_gap']:.1f}x (>= one decade: {d['d9']['decade_gap']}); dissociation floor loses "
             f"{100 * d['d9']['dissociation_floor_loss_over_20s']:.1f}% over 20 s; composition box maps to the band "
             f"{tuple(round(b, 4) for b in d['d9']['composition_band'])}: {d['d9']['composition_band_ok']} |")
    L.append(f"| D10 theta schema | {passed(d['d10']['ok'])} | round trip .zarr/.npy: {d['d10']['roundtrip.zarr']}/{d['d10']['roundtrip.npy']}; "
             f"refuses renamed key: {d['d10']['refuses_renamed_key']}, shifted bound: {d['d10']['refuses_shifted_bound']}, "
             f"other condition: {d['d10']['refuses_other_condition']}, schema-less .zarr: {d['d10']['refuses_schemaless_zarr']}, "
             f".npy without sidecar: {d['d10']['refuses_npy_without_sidecar']}; status names the reason: "
             f"{d['d10']['status_names_reason']}; package {d['d10']['schema_package_version']} |")
    L.append(f"| D11 conversion displacement | {passed(d['d11']['ok'])} | B1 -> A is a conversion for every mode "
             f"(product in place): {d['d11']['conversion_in_place']}; no channel re-forms B1 (at most one release per B1 "
             f"lineage): {d['d11']['b1_never_formed']}; a fission places each daughter {d['d11']['fission_daughter_kick_nm']:.0f} nm "
             f"from the parent (fixed 3D distance, random direction; {d['d11']['in_plane_mean_kick_nm']:.1f} nm in-plane on average) "
             f"-- the per-event difference between the two placements; against the per-frame diffusive RMS of a monomer at the "
             f"prior floor: " + ", ".join(f"{m} {v:.1f} nm ({d['d11']['kick_over_rms'][m]:.2f}x)" for m, v in d['d11']['per_frame_rms_at_prior_floor_nm'].items())
             + f"; against the localization precision {d['d11']['localization_precision_nm']:.0f} nm: {d['d11']['kick_over_precision']:.2f}x; "
             f"against the pixel {d['d11']['pixel_nm']:.0f} nm: {d['d11']['kick_over_pixel']:.3f}x |")
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
            L.append(f"| R3 visible fractions {c} ({v['probe_rule']} rule, two-probe share {v['two_probe_share']:.4f}) | "
                     f"{passed(v['ok'])} | {v['monomers_0']} monomers + {v['dimers_0']} dimers at frame 0; monomer "
                     f"{v['visible_monomer_emp']:.3f} vs {v['visible_monomer']:.3f}; dimer {v['visible_dimer_emp']:.3f} vs "
                     f"{v['visible_dimer']:.3f} (by the realized class mix: one-probe {v['one_probe_share_emp']:.3f}, two-probe "
                     f"{v['two_probe_share_emp']:.3f}); the rule recorded in the labeling row: {v['recorded']} |")
        L.append(f"| R4 both conditions, each from its own trajectory, through the shared labeling path | {passed(r['r4']['ok'])} | "
                 + "; ".join(f"{c}: {r['r4'][c]['frames']} frames, {r['r4'][c]['n_labeled']} labeled subunits of "
                             f"{r['r4'][c]['n_subunits']} under the {r['r4'][c]['probe_rule']} rule "
                             f"(expected {r['r4'][c]['expected_labeled']:.0f}; {r['r4'][c]['n_dyes']} dyes), "
                             f"pixel max {r['r4'][c]['pixel_max']:.0f}"
                             for c in lab.LABELING_CONDITIONS) + " |")
        L.append(f"| R5 boundary | {passed(r['r5']['ok'])} | particles outside the field at the last frame: "
                 f"{r['r5']['particles_outside_last_frame']}; mean fraction outside {r['r5']['fraction_outside_mean']:.4f}; "
                 f"subunit total constant: {r['r5']['subunit_total_constant']} |")
        L.append(f"| R6 timing and memory at the simulated-population ceiling | {passed(r['r6']['ok'])} | "
                 + "; ".join(f"{c}: N_total {x['n_total']:.0f} -> {x['n_subunits']} retained subunits = {x['n_particles']} particles "
                             f"({x['n_monomers']} monomers + {x['n_dimers']} dimers), D_A at its ceiling, {x['seconds_simulated']:.0f} s "
                             f"simulated in {x['wall_seconds']:.1f} s wall ({x['wall_seconds_per_simulated_second']:.1f} s per simulated "
                             f"second), lineage + 20-frame render {x['lineage_and_render_20_frames_seconds']:.1f} s, peak RSS "
                             f"{x['peak_rss_mib_before']:.0f} -> {x['peak_rss_mib_after']:.0f} MiB"
                             for c, x in r['r6'].items() if isinstance(x, dict)) + "; descriptive |")
        p7 = r["r7"]
        L.append(f"| R7 conversion and fission placement (measured) | {passed(p7['ok'])} | MET-INLB, {p7['seconds_simulated']:g} s, "
                 f"D_A = 1e-6 um^2/s, kappa_OFF at the ceiling, {p7['n_b1_initial']} B1 + {p7['n_b2_initial']} B2 at onset; "
                 f"B1 -> A products: n = {p7['conversion']['n']} ({p7['conversion']['skipped']} skipped), in-plane displacement "
                 f"mean {p7['conversion']['mean_nm']:.2f} nm, max {p7['conversion']['max_nm']:.2f} nm (in place: {p7['conversion_in_place']}); "
                 f"fission daughters: n = {p7['fission']['n']} ({p7['fission']['skipped']} skipped), mean {p7['fission']['mean_nm']:.2f} nm "
                 f"(expected pi/4 x {p7['expected_fission_3d_nm']:.0f} = {p7['expected_fission_in_plane_mean_nm']:.2f}), sd "
                 f"{p7['fission']['sd_nm']:.2f}, max {p7['fission']['max_nm']:.2f} nm (<= {p7['expected_fission_3d_nm']:.0f} nm fixed 3D distance: "
                 f"{p7['fission_as_declared']}) |")
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


LOCALIZATION_PRECISION_NM = 29.0   # MET recordings (Rahm et al. 2021, Front. Comput. Sci. 3, 757653)


def d11_conversion_displacement() -> dict:
    """The `B1 -> A` release as an in-place conversion against the fission placement and the
    observable scales (arithmetic from the parameterization; R7 measures the two placements)."""
    geom = par.PARAMETERS.simulation.stem
    timing = par.PARAMETERS.simulation.timing
    channels = rds.reaction_channels(prior_center_theta(), "INLB")
    b1, mono = RDS.stoichiometry.dimer_one_probe.name, RDS.stoichiometry.monomer.name
    species = lambda type_name: type_name.split("_")[0]                                   # noqa: E731
    release = [c for c in channels if species(c.educts[0]) == b1 and species(c.products[0]) == mono]
    conversion_in_place = (len(release) == len(RDS.mobility.modes)
                           and all(c.kind == "conversion" and len(c.educts) == 1 and len(c.products) == 1
                                   for c in release))
    b1_never_formed = not any(species(p) == b1 for c in channels for p in c.products
                              if not all(species(e) == b1 for e in c.educts))
    kick_nm = float(geom.fission_product_distance_nm) / 2.0            # each daughter, weights (0.5, 0.5)
    # Per-frame diffusive RMS displacement (2D) of a monomer in each mode at the prior FLOOR of D_A
    # and of the relative factors: the smallest diffusive step the prior allows per mode.
    theta_floor = prior_center_theta()
    for key in (RDS.mobility.diffusivity_key,) + tuple(k for k in RDS.mobility.mode_ratio_keys if k):
        entry = PARAMETERIZATION[key_index(key)]
        theta_floor[key_index(key)] = float(par.entry_to_physical(entry, entry["PRIOR_RANGE"][0]))
    coeff = rds.diffusion_coefficients(theta_floor)
    rms_nm = {mode: float(np.sqrt(4.0 * coeff[RDS.type_name(mono, mode)] * timing.frame_time_seconds) * 1e3)
              for mode in RDS.mobility.modes}
    pixel_nm = float(geom.pixel_size_nm)
    return dict(conversion_in_place=conversion_in_place, b1_never_formed=b1_never_formed,
                n_release_channels=len(release), fission_daughter_kick_nm=kick_nm,
                conversion_kick_nm=0.0, in_plane_mean_kick_nm=float(np.pi / 4 * kick_nm),
                per_frame_rms_at_prior_floor_nm=rms_nm,
                kick_over_rms={m: kick_nm / v for m, v in rms_nm.items()},
                localization_precision_nm=LOCALIZATION_PRECISION_NM,
                kick_over_precision=kick_nm / LOCALIZATION_PRECISION_NM,
                pixel_nm=pixel_nm, kick_over_pixel=kick_nm / pixel_nm,
                ok=bool(conversion_in_place and b1_never_formed
                        and kick_nm <= LOCALIZATION_PRECISION_NM and kick_nm <= pixel_nm / 10.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true",
                        help="also execute the run tier (tiny simulations and renders); needs approval")
    args = parser.parse_args()
    det_out = dict(d1=d1_channels(), d2=d2_diffusion(), d3=d3_transforms(),
                   d4=d4_composition(), d5=d5_stationary(), d6=d6_occupancy(),
                   d7=d7_fission_placement(), d8=d8_conditions(), d9=d9_ranges(),
                   d10=d10_theta_schema(), d11=d11_conversion_displacement())
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
            r3 = r3_visible(runs)                                 # each condition's own lineage through the shared path
            r4 = r4_both_conditions(runs)
            r2 = r2_stationarity(workdir)
            r6 = r6_timing(workdir)
            r7 = r7_conversion_placement(workdir)
            run_out = dict(r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6, r7=r7)
            for k in ("r1", "r2", "r3", "r4", "r5", "r6", "r7"):
                print(f"  [{passed(run_out[k]['ok'])}] {k}")
    write_report(det_out, run_out)
    all_ok = all(v["ok"] for v in det_out.values()) and (run_out is None or all(v["ok"] for v in run_out.values()))
    print("report:", REPORT)
    print("OVERALL:", passed(all_ok))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
