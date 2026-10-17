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
    D3 transforms      Per condition: `to_flow(to_physical(u, table), table) == u` on prior draws
                       INCLUDING the box corners; the LINEAR rows of the condition's table are
                       exactly {f} (MET-FAB) and {f, R_active} (MET-INLB), so a blanket 10**u differs
                       from `to_physical` on those columns and equals it on the log columns; a
                       conversion without a table raises; every table VALUE is its row's prior
                       center.
    D4 reference       The reference probe-bound population, per condition, from the receptor
       population      total N_total and the basal reference fraction f: at zero allocation the
                       integers of `realize_population` equal an AUDIT-LOCAL literal of the ratio
                       form (n_A,true = N/(1 + 2r), n_B,true = rN/(1 + 2r) with r = f/(1 - f);
                       D_sim = round(p (2-p) n_B,true), B2 = min(round(p^2 n_B,true), D_sim),
                       A = round(p n_A,true)) EXACTLY on a grid over the box (1,000 .. 100,000) x
                       f in [0, 1] (both endpoints) and on 1e5 random draws; N_R = A + 2 D_sim
                       (FAB) or A + B1 + 2 B2 (INLB) lies within 1.5 of its expectation; D_sim is
                       monotone in f; an invalid input (N <= 0, f or R outside [0, 1], a nonzero
                       Fab allocation, an unknown condition, a positional call) is refused and
                       nothing is returned. The ceilings at N_total = 100,000 are REPORTED (no
                       window): under FAB the represented receptors N_R and the probe-bound
                       receptors apart. The association reference equals the Smoluchowski
                       expression 4 pi (2 D_A) r / ((4/3) pi r^3) = 6 D_A / r^2, under MET-INLB
                       every fusion fires at exactly lambda_ref, and MET-FAB has no fusion channel.
    D5 stationary law  `stationary_mode_law` sums to one and satisfies detailed balance on
                       every link of the chain.
    D6 probe rule      `rank_to_species` maps every monomer type to the stoichiometric class A and
                       every dimer type (B, B1, B2) to B, `rank_to_molecule` keeps the ligand class;
                       the probe rule of `labeling.assign_probes` binds every retained monomer, every
                       subunit of a B1 / B2 host, and one or two subunits of a FAB dimer at the
                       declared two-probe share; an override's per-class coins give identical
                       probabilities within a class, and a coin keyed by a particle type is rejected.
    D7 fission place   Fission daughters are placed `SimulationStem.fission_product_distance_nm`
                       = 4 x the reaction distance apart (40 nm; 2 x until 2026-10-13) and
                       `build_system` passes that value. ReaDDy orients the daughter axis
                       uniformly in three dimensions inside the 2 nm periodic slab, so a fraction
                       q = 1 - sqrt(1 - (r/d)^2) of the pairs lies within the reaction distance in
                       the plane: 13.4% at 20 nm, 3.2% at 40 nm, the accepted residual (the check
                       requires the geometric fraction q <= 4%). Reported beside it: the next-step re-fusion fraction per
                       daughter mode at the prior center (the placement geometry plus one diffusive
                       step) and the one-step rebinding probability under placement AT the radius,
                       the reason for placing the daughters apart at all.
    D8 conditions      The condition registry: the model's tokens equal the labeling and the
                       experiment registries' tokens; MET-FAB association ratio is exactly 0.0
                       and MET-INLB exactly 1.0; the retired rows (the association ratio and the
                       dimer-to-monomer ratio) are absent from both conditions' tables, from the
                       master table and from the detector's RDS-nuisance mirror, which equals the
                       master's twelve keys; the Fab keys equal the InlB keys [:11] (alignment by
                       KEY, the shared prefix a safety margin); MET-FAB declares the allocation 0
                       and MET-INLB infers it; the removed symbols are absent; `rds_alias` carries
                       the condition token and refuses to resolve without one. Occupancies of the
                       underlying population: MET-INLB 0.0476 (EQUILIBRIUM at 0.25 nM with K_D
                       5 nM), MET-FAB 0.0148 (DERIVED from the declared Fab/InlB visibility ratio
                       0.5 and the InlB anchor), visibilities 0.0238 / 0.0119, two-probe shares
                       0.0244 / 0.0074; MET-INLB carries the ligand classes and MET-FAB does not; a
                       setting with more or fewer than one occupancy source, with association but
                       no ligand classes, with a declared allocation outside [0, 1], or with an
                       inferred or nonzero allocation but no ligand classes, is refused.
    D9 ranges          The decided prior ranges and scales as a literal table (both fraction rows
                       linear on (0, 1); the allocation row only under MET-INLB) equal each
                       condition's table; the immobile factor keeps D_i = R_i D_A below the
                       pipelines' looser immobility threshold (0.0065 um^2/s) at the D_A center for
                       every R_i in range (immobility by construction) and a full decade below the
                       slow factor; the dissociation floor loses <= 2% of dimers over 20 s; the
                       count box is [3.0, 5.0].
    D10 theta schema   Per condition: a Theta_Set written by `io.write_theta_set` (.zarr and .npy)
                       carries the condition's table and reads back through `io.load_theta_set`; a
                       table with a renamed key, a table with a shifted prior bound, the other
                       condition, the other condition's table, an audit-local table carrying the
                       retired ratio row, a schema-less .zarr, and a .npy without its sidecar are
                       each REFUSED (`ThetaSetSchemaError`), and `theta_set_status` names the
                       reason.
    D11 conversion     The `B1 -> A` release is a CONVERSION for every mode (the product keeps the
        displacement    educt's position) and no channel re-forms `B1`, so a `B1` lineage sees at most
                       one such event; the per-event difference from a fission placement is one
                       daughter kick of half the product distance (two reaction distances, 20 nm, at a
                       fixed 3D distance in a random direction). That kick is compared with the
                       per-frame diffusive RMS displacement of a monomer in each mode at the prior
                       FLOOR of `D_A` and of the relative factors, with the localization precision of
                       the MET recordings (29 nm) and with the pixel; it passes when it is below the
                       precision and below a fifth of a pixel. R7 measures both placements.
    D12 allocation     The window-start allocation, under MET-INLB, on a grid over N_total x f x
                       R_active with half-integer products R_active x capacity included (rounded
                       half to even): the capacity is A_ref // 2, c = round(R_active cap), the
                       starting counts are A_start = A_ref - 2c and B2_start = B2_ref + c with B1
                       unchanged, N_R = A_start + B1 + 2 B2_start exactly (conservation), an odd
                       A_ref leaves one monomer at R_active = 1, a zero capacity (f = 1, or a total
                       too small for a pair) allocates nothing, the derived underlying quantities
                       satisfy M_start = M_ref - 2c, D_start = D_ref + c, M_start + 2 D_start = N,
                       f_start = D_start / (M_start + D_start) and the active share c / D_start
                       (NaN at D_start = 0), c is monotone in R_active, the worked literals hold
                       (N = 1e4, f = 0.2: A_ref 317, B1 151, B2 4, cap 158, c 79 at R_active 0.5
                       and 158 at 1; the prior center 79 A + 302 B1 + 48 B2); under MET-FAB the
                       starting counts equal the reference counts with capacity and c zero; the
                       vectorized helper (`population_from_theta`) equals the one-draw initializer
                       (`initial_composition_of`, by key) on 1e4 prior draws per condition.

Run tier (``--run``; tiny; after approval only):
    R1 conservation    One 2 s run per condition at the prior center (the reference recording length) (MET-INLB: the full
                       reactive network with the ligand classes, c > 0 at the center: 79 A + 302 B1
                       + 48 B2 at N_total = 10,000; MET-FAB: dissociation and switching only): the
                       frame-0 particle counts per class equal the STARTING counts of the helper,
                       the lineage replays with every represented subunit covered once per frame
                       (fail-loud extractor) and its subunit count equals the realized N_R.
    R2 stationarity    The isolated switching chain under the MET-FAB configuration (no
                       association channel by construction) with monomers only (f = 0, inside the
                       prior), 10 s: the time-averaged mode occupancies over the second half agree
                       with `stationary_mode_law` within a prespecified tolerance.
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
    R6 timing          Wall time of one 2 s run per condition at the simulated-population
                       ceiling (N_total = 100,000, D_A at its ceiling; MET-FAB at f = 1 with
                       kappa_OFF at its ceiling, so its 1,466 initial dimers split within the run
                       toward 2,932 particles; MET-INLB at f = 0 and R_active = 0: 4,762 monomers),
                       with the particle counts the run reaches and the process-wide peak memory
                       before and after it (cumulative, not isolated), plus a 20-frame render of
                       each; descriptive, a high-load benchmark, not a runtime bound.
    R7 placement       One 0.2 s MET-INLB run with diffusion switched almost off (D_A = 1e-6 um^2/s)
                       and kappa_OFF at the prior ceiling, N_total at the box ceiling and f = 1
                       (most B1): the in-plane displacement of every `B1 -> A` product
                       from its educt's position at the frame before (expected zero: in place) and of
                       every fission daughter from its parent's (expected a fixed 20 nm in 3D, pi/4 x
                       20 nm in-plane on average), read from the reaction records and the frames.
    R8 geometry        Four 2 ms steps of 20,000 immobile dimers in the production box and slab with
                       the production fission distance and a fusion at the reaction distance: the
                       in-plane separation of every daughter pair at the frame after the fission and
                       the fraction within the reaction distance, and the fusions at the following
                       step whose educts are the two daughters of one fission (chance encounters of
                       daughters of different dimers counted apart) over the fissions: the next-step
                       re-fusion of an immobile daughter pair; both within four sampling standard
                       deviations (about 0.5 %) of the geometric fraction q of D7.

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

PARAMETERS = par.PARAMETERS
RDS = PARAMETERS.simulation.rds
STO = RDS.stoichiometry
N_KEY, F_KEY, R_KEY = STO.count_total_key, STO.basal_fraction_key, STO.active_allocation_key

OUT_DIR = os.path.join(str(PARAMETERS.machine.data_bank_root), "Posit",
                       "SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit")
REPORT = os.path.join(OUT_DIR, "SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.md")

SEED = 20260909
N_PRIOR_DRAWS = 20_000        # D2/D3: prior draws for the ordering and round-trip checks
N_D4_RANDOM = 100_000         # D4: random (N_total, f) draws per condition against the ratio-form literal
N_D12_DRAWS = 10_000          # D12: prior draws per condition, vectorized helper vs one-draw initializer
TOL_ROUNDTRIP = 1e-9
TOL_SUBUNITS = 1.5            # D4: |N_R - expectation| <= three roundings of half a subunit
TOL_STATIONARY = 0.03         # R2: absolute tolerance on time-averaged mode occupancies
TOL_VISIBLE = 0.02            # R3: absolute tolerance on visible fractions
N_LABEL_DRAWS = 400           # R3: independent labelings

# The retired composition row, as an AUDIT-LOCAL literal (D8 asserts its absence, D10 that a Theta_Set
# stored under a table carrying it is refused): the log10 dimer-to-monomer ratio on the 5-25 % band.
RETIRED_RATIO_ROW = dict(KEY="ratio_dimer_monomer_initial",
                         PRIOR_RANGE=(-1.2787536009528289, -0.47712125471966244), LOG_FLAG=True, LOG_BASE=10,
                         UNIT="Ratio", LABEL=r"$r_{B/A}$", NOTE="Retired")
RETIRED_KEYS = ("relative_rate_dimerization", RETIRED_RATIO_ROW["KEY"])


def passed(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def table_of(condition: str) -> list:
    """The condition's learnable table (eleven rows under MET-FAB, twelve under MET-INLB)."""
    return par.parameterization_of(condition)


def prior_center_theta(condition: str) -> np.ndarray:
    """Physical prior center of the condition's table, in its theta order."""
    return np.array([par.prior_center(e) for e in table_of(condition)], dtype=float)


def key_index(key: str, condition: str) -> int:
    return par.parameter_find(key, condition)


def prior_box(condition: str) -> tuple:
    return np.array(par.theta_lower_bound(condition)), np.array(par.theta_upper_bound(condition))


def reference_population_ratio_form(count_total: float, fraction: float, occupancy: float,
                                    ligand_classes: bool) -> tuple:
    """AUDIT-LOCAL literal of the reference population in the ratio form that preceded the
    allocation: with r = f / (1 - f), n_A,true = N / (1 + 2r), n_B,true = r N / (1 + 2r);
    D_sim = round(p (2 - p) n_B,true), B2 = min(round(p^2 n_B,true), D_sim), B1 = D_sim - B2,
    A = round(p n_A,true); N_R = A + 2 D_sim without ligand classes, A + B1 + 2 B2 with them. At
    f = 1 the ratio is unbounded and the limits apply: n_A,true = 0, n_B,true = N / 2. Python
    scalars and ``round`` (half to even), transcribed independently of the package helper.
    Returns ``(A, D_sim, B1, B2, N_R)``."""
    n, p = float(count_total), float(occupancy)
    if fraction >= 1.0:
        n_a_true, n_b_true = 0.0, n / 2.0
    else:
        r = fraction / (1.0 - fraction)
        n_a_true, n_b_true = n / (1.0 + 2.0 * r), r * n / (1.0 + 2.0 * r)
    d_sim = int(round(p * (2.0 - p) * n_b_true))
    b2 = int(min(round(p * p * n_b_true), d_sim))
    b1 = d_sim - b2
    a = int(round(p * n_a_true))
    n_r = a + b1 + 2 * b2 if ligand_classes else a + 2 * d_sim
    return a, d_sim, b1, b2, n_r


def realize(condition: str, count_total, fraction, allocation):
    """``realize_population`` under ``condition`` on broadcast inputs (keyword-only helper)."""
    return par.realize_population(count_total=count_total, fraction_basal_dimers_reference=fraction,
                                  relative_count_active_dimers_initial=allocation, condition=condition)


# ----------------------------------------------------------------------------------------------
# Deterministic tier
# ----------------------------------------------------------------------------------------------

def d1_channels() -> dict:
    """Per condition: the generated channels against the declared network (24 under INLB, 11 under FAB)."""
    out = {c: _d1_condition(c) for c in RDS.condition_tokens}
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def _d1_condition(condition: str) -> dict:
    theta = prior_center_theta(condition)
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
    kappa_off = rds.theta_by_key(theta, condition)[sto.dissociation_rate_key]
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
    out = {c: _d2_condition(c) for c in RDS.condition_tokens}
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def _d2_condition(condition: str) -> dict:
    rng = np.random.default_rng(SEED)
    low, high = prior_box(condition)
    flow = rng.uniform(low, high, size=(N_PRIOR_DRAWS, len(low)))
    corners = np.array(list(itertools.product(*zip(low, high)))) if len(low) <= 12 else None
    if corners is not None:
        flow = np.vstack([flow, corners])
    phys = par.to_physical(flow, table_of(condition))
    sto, mob = RDS.stoichiometry, RDS.mobility
    within_species_ok, within_mode_ok, formula_ok = True, True, True
    # Vectorized check of the ordering through diffusion_coefficients on a subsample, and the
    # closed-form on all draws (the function is per-theta; the closed form is what it computes).
    d_a = phys[:, key_index(mob.diffusivity_key, condition)]
    r_b = phys[:, key_index(mob.dimer_ratio_key, condition)]
    mode_factors = [np.ones_like(d_a)] + [phys[:, key_index(k, condition)] for k in mob.mode_ratio_keys[1:]]
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
        d = rds.diffusion_coefficients(phys[idx], condition)
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
    out = {c: _d3_condition(c) for c in RDS.condition_tokens}
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def _d3_condition(condition: str) -> dict:
    rng = np.random.default_rng(SEED + 1)
    table = table_of(condition)
    low, high = prior_box(condition)
    flow = rng.uniform(low, high, size=(N_PRIOR_DRAWS, len(low)))
    flow = np.vstack([flow, low[None], high[None]])
    phys = par.to_physical(flow, table)
    back = par.to_flow(phys, table)
    max_err = float(np.max(np.abs(back - flow)))
    linear_keys = [e["KEY"] for e in table if not par.is_log_row(e)]
    expected_linear = [F_KEY] if par.declared_row_values(condition) else [F_KEY, R_KEY]
    linear_rows_ok = linear_keys == expected_linear
    is_log = np.array([par.is_log_row(e) for e in table])
    blanket = np.power(10.0, flow)
    # the blanket differs from the rule on every linear column (10**u = u has no solution on (0, 1)
    # except u = 10**u, which no draw hits) and equals it on every log column
    blanket_differs_on_linear = bool(np.all(np.abs(blanket[:, ~is_log] - phys[:, ~is_log]) > 1e-9))
    blanket_equals_on_log = bool(np.allclose(blanket[:, is_log], phys[:, is_log], rtol=1e-12))
    linear_pass_through = bool(np.array_equal(phys[:, ~is_log], flow[:, ~is_log]))
    no_table_refused = 0
    for call in (lambda: par.to_physical(flow, None), lambda: par.to_flow(phys, None),
                 lambda: par.fraction_flags(phys, None)):
        try:
            call()
        except ValueError:
            no_table_refused += 1
    wrong_width_refused = False
    try:
        par.to_physical(flow[:, :-1], table)
    except ValueError:
        wrong_width_refused = True
    per_entry_ok = all(np.isclose(par.entry_to_flow(e, par.entry_to_physical(e, 0.3)), 0.3) for e in table)
    center_ok = all(np.isclose(par.prior_center(e), e["VALUE"], rtol=1e-6) for e in table)
    flags = par.fraction_flags(phys, table)
    flags_ok = set(flags) == set(linear_keys) and not any(f.any() for f in flags.values())
    return dict(condition=condition, n_rows=len(table), max_roundtrip_error=max_err, linear_rows=linear_keys,
                linear_rows_as_decided=linear_rows_ok, blanket_power_differs_on_linear_rows=blanket_differs_on_linear,
                blanket_power_equals_to_physical_on_log_rows=blanket_equals_on_log,
                linear_rows_pass_through=linear_pass_through, conversions_without_table_refused=no_table_refused == 3,
                wrong_width_refused=wrong_width_refused, per_entry_roundtrip=per_entry_ok,
                table_values_are_prior_centers=center_ok, no_prior_draw_flagged=bool(flags_ok),
                ok=(max_err < TOL_ROUNDTRIP and linear_rows_ok and blanket_differs_on_linear and blanket_equals_on_log
                    and linear_pass_through and no_table_refused == 3 and wrong_width_refused and per_entry_ok
                    and center_ok and flags_ok))


def d4_composition() -> dict:
    """The reference probe-bound population against the audit-local ratio-form literal, per condition."""
    problems, n_cases = [], 0
    n_grid = 10.0 ** np.linspace(3.0, 5.0, 41)
    f_grid = np.linspace(0.0, 1.0, 101)                  # both endpoints
    ceilings, monotone = {}, {}
    rng = np.random.default_rng(SEED + 4)
    for condition in RDS.condition_tokens:
        p = par.occupancy_of(condition)
        classes = bool(RDS.condition_setting(condition).ligand_classes)
        declared = par.declared_row_values(condition).get(R_KEY, 0.0)
        nn, ff = np.meshgrid(n_grid, f_grid, indexing="ij")
        pop = realize(condition, nn, ff, declared)
        n_cases += int(nn.size)
        mismatches = 0
        for i in range(nn.shape[0]):
            for j in range(nn.shape[1]):
                lit = reference_population_ratio_form(nn[i, j], ff[i, j], p, classes)
                got = (int(pop.n_monomers_reference[i, j]), int(pop.n_dimers_reference[i, j]),
                       int(pop.n_dimers_one_probe[i, j]), int(pop.n_dimers_two_probe_reference[i, j]),
                       int(pop.n_subunits[i, j]))
                if got != lit:
                    mismatches += 1
                    if mismatches <= 3:
                        problems.append(f"{condition} N={nn[i, j]:.0f}, f={ff[i, j]:.3f}: helper {got} != literal {lit}")
        if mismatches:
            problems.append(f"{condition}: {mismatches} grid mismatches against the ratio-form literal")
        # 1e5 random draws over the box, zero allocation
        n_rand = 10.0 ** rng.uniform(3.0, 5.0, N_D4_RANDOM)
        f_rand = rng.uniform(0.0, 1.0, N_D4_RANDOM)
        pop_r = realize(condition, n_rand, f_rand, declared)
        n_cases += N_D4_RANDOM
        rand_mismatch = 0
        for k in range(N_D4_RANDOM):
            lit = reference_population_ratio_form(n_rand[k], f_rand[k], p, classes)
            got = (int(pop_r.n_monomers_reference[k]), int(pop_r.n_dimers_reference[k]), int(pop_r.n_dimers_one_probe[k]),
                   int(pop_r.n_dimers_two_probe_reference[k]), int(pop_r.n_subunits[k]))
            if got != lit:
                rand_mismatch += 1
        if rand_mismatch:
            problems.append(f"{condition}: {rand_mismatch} of {N_D4_RANDOM} random draws differ from the ratio-form literal")
        # N_R within rounding of its expectation (grid and random draws)
        for name, pp in (("grid", pop), ("random", pop_r)):
            dev = np.abs(pp.n_subunits - pp.expected_subunits)
            if float(dev.max()) > TOL_SUBUNITS + 1e-9:
                problems.append(f"{condition} ({name}): N_R deviates {float(dev.max()):.2f} from its expectation")
        # D_sim monotone in f at every N of the grid; no dimer at f = 0; no monomer at f = 1
        monotone[condition] = bool(np.all(np.diff(pop.n_dimers_reference, axis=1) >= 0))
        if not monotone[condition]:
            problems.append(f"{condition}: D_sim is not monotone in f")
        if np.any(pop.n_dimers_reference[:, 0] != 0) or np.any(pop.n_monomers_reference[:, -1] != 0):
            problems.append(f"{condition}: dimers at f = 0 or monomers at f = 1")
        # the reference rule itself: N_R = A + 2 D_sim (FAB) or A + B1 + 2 B2 (INLB); classes split the dimers
        n_r_rule = (pop.n_monomers_reference + pop.n_dimers_one_probe + 2 * pop.n_dimers_two_probe_reference if classes
                    else pop.n_monomers_reference + 2 * pop.n_dimers_reference)
        if not np.array_equal(pop.n_subunits, n_r_rule) or not np.array_equal(
                pop.n_dimers_one_probe + pop.n_dimers_two_probe_reference, pop.n_dimers_reference):
            problems.append(f"{condition}: the represented-receptor rule or the class split is broken")
        if not np.isclose(pop.two_probe_share, p / (2 - p)):
            problems.append(f"{condition}: two-probe share is not p / (2 - p)")
        # the ceilings, reported: represented receptors and (under FAB) probe-bound receptors apart
        ceilings[condition] = {}
        for f in (0.0, 0.25, 1.0):
            c = par.realize_initial_composition(count_total=1e5, fraction_basal_dimers_reference=f,
                                                relative_count_active_dimers_initial=declared, condition=condition)
            ceilings[condition][f"f={f:g}"] = dict(
                represented_receptors=int(c.n_subunits), initial_particles=int(c.n_complexes),
                monomers=int(c.n_monomers), dimers=int(c.n_dimers),
                probe_bound_receptors_expected=float(c.n_probe_bound_receptors_expected))
    # refusals: nothing is returned on an invalid input, on either path
    refusals = []
    cases = [dict(count_total=10, fraction_basal_dimers_reference=-0.5, relative_count_active_dimers_initial=0.0, condition="FAB"),
             dict(count_total=10, fraction_basal_dimers_reference=1.5, relative_count_active_dimers_initial=0.0, condition="FAB"),
             dict(count_total=0.0, fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=0.0, condition="FAB"),
             dict(count_total=100, fraction_basal_dimers_reference=float("nan"), relative_count_active_dimers_initial=0.0, condition="INLB"),
             dict(count_total=100, fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=0.3, condition="FAB"),
             dict(count_total=100, fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=-0.1, condition="INLB"),
             dict(count_total=100, fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=1.2, condition="INLB"),
             dict(count_total=100, fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=0.0, condition="XYZ")]
    for bad in cases:
        for path in (par.realize_initial_composition, par.realize_population):
            try:
                path(**bad)
            except (ValueError, KeyError):
                continue
            refusals.append(f"{path.__name__} accepted {bad}")
    try:                                                  # the keyword-only initializer refuses a positional call
        par.realize_initial_composition(100, 0.1, "FAB")
        refusals.append("positional call accepted")
    except TypeError:
        pass
    valid = par.population_valid(count_total=[100, 100, 0], fraction_basal_dimers_reference=[0.1, 1.5, 0.1],
                                 relative_count_active_dimers_initial=[0.0, 0.0, 0.0], condition="FAB")
    if not np.array_equal(valid, [True, False, False]):
        refusals.append(f"population_valid returned {valid.tolist()} for [valid, f = 1.5, N = 0]")
    problems += refusals
    # association reference
    d_a, r_nm = 0.2, PARAMETERS.simulation.stem.particle_diameter_nm
    r = r_nm * 1e-3
    smol = 4 * np.pi * (2 * d_a) * r / ((4 / 3) * np.pi * r ** 3)
    ref_ok = bool(np.isclose(rds.association_reference_rate(d_a, r_nm), smol, rtol=1e-12)
                  and np.isclose(rds.association_reference_rate(d_a, r_nm), 6 * d_a / r ** 2, rtol=1e-12))
    theta = prior_center_theta("INLB")
    lamb_ref = rds.association_reference_rate(theta[key_index(RDS.mobility.diffusivity_key, "INLB")], r_nm)
    lamb_inlb = {c.rate for c in rds.reaction_channels(theta, "INLB") if c.kind == "fusion"}
    lamb_ok = bool(len(lamb_inlb) == 1 and np.isclose(lamb_inlb.pop(), RDS.association_ratio_of("INLB") * lamb_ref)
                   and RDS.association_ratio_of("INLB") == 1.0)
    fab_fusions = [c for c in rds.reaction_channels(prior_center_theta("FAB"), "FAB") if c.kind == "fusion"]
    fab_ok = len(fab_fusions) == 0          # structural: no channel, not a tiny rate
    return dict(n_cases=n_cases, grid=dict(n_total=[float(n_grid[0]), float(n_grid[-1]), int(n_grid.size)],
                                           fraction=[0.0, 1.0, int(f_grid.size)]),
                n_random_draws_per_condition=N_D4_RANDOM, subunit_tolerance=TOL_SUBUNITS, problems=problems,
                dimers_monotone_in_fraction=monotone, ceilings_at_count_ceiling=ceilings, refusals_ok=not refusals,
                association_reference_is_smoluchowski=ref_ok, lambda_on_equals_reference_under_INLB=lamb_ok,
                no_fusion_channel_under_FAB=fab_ok, ok=not problems and ref_ok and lamb_ok and fab_ok)


def d5_stationary() -> dict:
    rng = np.random.default_rng(SEED + 2)
    mob = RDS.mobility
    ok_sum, ok_balance = True, True
    for condition in RDS.condition_tokens:
        low, high = prior_box(condition)
        table = table_of(condition)
        for _ in range(200):
            theta = par.to_physical(rng.uniform(low, high), table)
            pi = rds.stationary_mode_law(theta, condition)
            if not np.isclose(pi.sum(), 1.0) or np.any(pi <= 0):
                ok_sum = False
            values = rds.theta_by_key(theta, condition)
            fwd = {(a, b): values[k] for a, b, k in mob.switching}
            for i in range(len(mob.modes) - 1):
                a, b = mob.modes[i], mob.modes[i + 1]
                if not np.isclose(pi[i] * fwd[(a, b)], pi[i + 1] * fwd[(b, a)], rtol=1e-10):
                    ok_balance = False
    return dict(n_draws_per_condition=200, sums_to_one=ok_sum, detailed_balance=ok_balance, ok=ok_sum and ok_balance)


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


ACCEPTED_PLACEMENT_RESIDUAL = 0.04   # the lead accepted a 2-3% residual (2026-10-13): 3.2% at 4 x the reaction distance


def in_plane_within_fraction(reaction_nm: float, product_nm: float) -> float:
    """Fraction of daughter pairs within the reaction distance in the plane for an isotropic
    three-dimensional daughter axis in a slab thinner than the reaction distance:
    q = 1 - sqrt(1 - (r/d)^2) (the in-plane separation is d sin(theta), cos(theta) uniform)."""
    return float(1.0 - np.sqrt(max(0.0, 1.0 - (reaction_nm / product_nm) ** 2)))


def d7_fission_placement() -> dict:
    import inspect
    geom = par.PARAMETERS.simulation.stem
    r_nm = float(geom.particle_diameter_nm)
    d_prod = float(geom.fission_product_distance_nm)
    outside = d_prod > r_nm
    declared_factor = int(geom.fission_product_distance_factor) == 4 and abs(d_prod - 4.0 * r_nm) < 1e-9
    src = inspect.getsource(rds.build_system)
    passes_field = ("fission_product_distance_nm" in src
                    and "product_distance=product_distance_nm" in src
                    and "product_distance=reaction_distance_nm" not in src)
    q = in_plane_within_fraction(r_nm, d_prod)
    q_former = in_plane_within_fraction(r_nm, 2.0 * r_nm)
    # Next-step re-fusion under the declared placement (geometry plus one diffusive step of each
    # daughter: relative displacement variance 2 * (2 D) dt per axis, 2D Brownian) and the one-step
    # rebinding under placement AT the radius, both at the prior center, per daughter mode.
    theta = prior_center_theta("INLB")                 # the mobility rows are shared; either condition's center
    coeff = rds.diffusion_coefficients(theta, "INLB")
    dt_s = par.PARAMETERS.simulation.timing.frame_time_seconds / par.PARAMETERS.simulation.timing.steps_per_frame
    mono = RDS.stoichiometry.monomer.name
    rng = np.random.default_rng(0)
    n = 200_000
    cos = rng.uniform(-1.0, 1.0, n)
    s = d_prod * np.sqrt(1.0 - cos ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    pair = np.stack([s * np.cos(phi), s * np.sin(phi)], axis=1)
    refusion, rebind_at_radius = {}, {}
    for mode in RDS.mobility.modes:
        d_um2_s = coeff[RDS.type_name(mono, mode)]
        sigma_nm = np.sqrt(4.0 * d_um2_s * dt_s) * 1e3
        after = pair + rng.normal(0.0, sigma_nm, (n, 2))
        refusion[mode] = float(np.mean(np.hypot(after[:, 0], after[:, 1]) <= r_nm))
        xi = rng.normal(0.0, sigma_nm, (n, 2)); xi[:, 0] += r_nm
        rebind_at_radius[mode] = float(np.mean(np.hypot(xi[:, 0], xi[:, 1]) <= r_nm))
    return dict(reaction_distance_nm=r_nm, product_distance_nm=d_prod, outside_fusion_radius=outside,
                declared_factor_four=declared_factor, build_system_uses_field=passes_field,
                in_plane_within_reaction_distance_fraction=q, in_plane_within_fraction_at_former_20_nm=q_former,
                accepted_residual=ACCEPTED_PLACEMENT_RESIDUAL, sub_step_s=dt_s,
                next_step_refusion_fraction_by_mode_prior_center=refusion,
                one_step_rebind_probability_if_placed_at_radius=rebind_at_radius,
                ok=bool(outside and declared_factor and passes_field and q <= ACCEPTED_PLACEMENT_RESIDUAL))


def _removed_symbol_names() -> tuple:
    """The names the window-start revision removed from ``parameterization``; D8 checks that none
    of them is defined any more."""
    return ("PARAMETERIZATION", "PARAMETER_KEYS", "PARAMETER_FIND", "COMPOSITION_BAND",
            "COMPOSITION_RATIO_BOX", "ratio_to_receptor_fraction", "ratio_to_complex_fraction",
            "receptor_fraction_to_ratio", "retained_ratio_of", "composition_ratio_key")


def d8_conditions() -> dict:
    """The condition registry, the retired rows, the per-condition tables and the detector mirror."""
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp.experiment_support import CONDITION_DISPLAY
    tokens = RDS.condition_tokens
    tokens_agree = (set(tokens) == set(lab.LABELING_CONDITIONS) == set(CONDITION_DISPLAY)
                    and len(set(tokens)) == len(tokens))
    fab_zero = RDS.association_ratio_of("FAB") == 0.0
    inlb_one = RDS.association_ratio_of("INLB") == 1.0
    master_keys = list(par.MASTER_PARAMETER_KEYS)
    mirror_keys = [e["KEY"] for e in det.DETECTOR_NUISANCE]
    tables = {c: par.parameter_keys_of(c) for c in tokens}
    row_absent = (not any(k in tables[c] for k in RETIRED_KEYS for c in tokens)
                  and not any(k in master_keys or k in mirror_keys for k in RETIRED_KEYS)
                  and not hasattr(RDS.stoichiometry, "association_ratio_key"))
    mirror_ok = mirror_keys == master_keys and len(master_keys) == 12
    fab_prefix_ok = (tables["FAB"] == tables["INLB"][:11] and len(tables["FAB"]) == 11 and len(tables["INLB"]) == 12
                     and tables["INLB"][-1] == R_KEY and tables["INLB"][1] == F_KEY)
    declared_ok = (par.declared_row_values("FAB") == {R_KEY: 0.0} and par.declared_row_values("INLB") == {}
                   and RDS.condition_setting("FAB").relative_count_active_dimers_initial == 0.0
                   and RDS.condition_setting("INLB").relative_count_active_dimers_initial is None)
    removed_absent = not any(hasattr(par, name) for name in _removed_symbol_names())
    removed_absent &= not hasattr(RDS, _removed_symbol_names()[-2]) and not hasattr(STO, _removed_symbol_names()[-1])
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
    # the allocation setting: a declared value outside [0, 1] refused; an inferred (None) or nonzero
    # allocation without ligand classes refused; the declared 0 without classes and a declared
    # sensitivity value under classes accepted (the latter removes the row)
    allocation_refusals = 0
    for kw in (dict(ligand_classes=True, dissociation_constant_nm=5.0, relative_count_active_dimers_initial=1.5),
               dict(ligand_classes=True, dissociation_constant_nm=5.0, relative_count_active_dimers_initial=float("nan")),
               dict(occupancy=0.5),                                              # inferred, no classes
               dict(occupancy=0.5, relative_count_active_dimers_initial=0.3)):   # nonzero, no classes
        try:
            par.ConditionSetting("X", 0.0 if not kw.get("ligand_classes") else 1.0, **kw)
        except ValueError:
            allocation_refusals += 1
    try:
        fab_like = par.ConditionSetting("X", 0.0, occupancy=0.5, relative_count_active_dimers_initial=0.0)
        variant = par.ConditionSetting("Y", 1.0, ligand_classes=True, dissociation_constant_nm=5.0,
                                       relative_count_active_dimers_initial=0.3)
        allocation_accepted = fab_like.declared_rows == {R_KEY: 0.0} and variant.declared_rows == {R_KEY: 0.3}
    except ValueError:
        allocation_accepted = False
    allocation_setting_ok = allocation_refusals == 4 and allocation_accepted
    return dict(tokens=list(tokens), tokens_agree_across_registries=tokens_agree, fab_ratio_is_zero=fab_zero,
                inlb_ratio_is_one=inlb_one, retired_keys=list(RETIRED_KEYS), retired_rows_absent=row_absent,
                parameter_keys=tables, master_keys=master_keys, detector_mirror_equals_master=mirror_ok,
                fab_keys_equal_inlb_prefix=fab_prefix_ok, declared_rows_as_decided=declared_ok,
                removed_symbols_absent=bool(removed_absent), disguised_zero_refused=epsilon_refused,
                bare_rds_alias_refused=bare_refused, tier_aliases=aliases, detector_reads_same_tier=det_alias_ok,
                occupancy={c: round(v, 5) for c, v in occ.items()}, visibility=vis, occupancy_source=src,
                two_probe_share={c: round(v, 5) for c, v in s2.items()},
                ligand_classes={c: RDS.condition_setting(c).ligand_classes for c in tokens},
                fab_occupancy_derived_ok=fab_derived_ok, inlb_occupancy_equilibrium_ok=inlb_declared_ok,
                both_and_neither_refused=both_refused and neither_refused,
                association_without_classes_refused=no_classes_refused,
                allocation_setting_refusals=allocation_refusals, allocation_setting_ok=allocation_setting_ok,
                ok=tokens_agree and fab_zero and inlb_one and row_absent and mirror_ok and fab_prefix_ok and declared_ok
                and removed_absent and epsilon_refused and bare_refused and alias_ok and det_alias_ok and fab_derived_ok
                and inlb_declared_ok and both_refused and neither_refused and no_classes_refused and allocation_setting_ok)


# The decided prior ranges and scales (the mobility and dissociation rows decided 2026-09-14, the
# count row 2026-10-05, the two fraction rows 2026-10-15), as a second, literal copy in MASTER order:
# a table edit that is not a decision fails D9. Each value is (lower, upper, log): the estimator
# coordinate is log10 of the physical value for the log rows and the value itself for the two LINEAR
# fraction rows, both uniform on (0, 1). The allocation row exists only under MET-INLB (MET-FAB
# declares it 0).
DECIDED_RANGES = {
    "count_total": (3.0, 5.0, True), "fraction_basal_dimers_reference": (0.0, 1.0, False),
    "rate_dissociation": (-3.0, 1.0, True),
    "diffusivity_alp": (-1.25, -0.25, True), "relative_diffusivity_dimer": (-1.0, 0.0, True),
    "relative_diffusivity_slow": (-1.0, 0.0, True), "relative_diffusivity_immobile": (-3.0, -2.0, True),
    "rate_fast_slow": (-1.0, 1.0, True), "rate_slow_fast": (-1.0, 1.0, True), "rate_slow_immobile": (-1.0, 1.0, True),
    "rate_immobile_slow": (-1.0, 1.0, True),
    "relative_count_active_dimers_initial": (0.0, 1.0, False),
}
DECIDED_KEYS_OF_CONDITION = {"FAB": [k for k in DECIDED_RANGES if k != "relative_count_active_dimers_initial"],
                             "INLB": list(DECIDED_RANGES)}
IMMOBILE_THRESHOLDS_UM2_S = (0.0028, 0.0065)     # the tracking pipelines' immobility thresholds


def d9_ranges() -> dict:
    """The decided ranges and scales per condition, immobility by construction, the dissociation floor."""
    per_condition = {}
    for condition in RDS.condition_tokens:
        table = table_of(condition)
        keys = [e["KEY"] for e in table]
        got = {e["KEY"]: (float(e["PRIOR_RANGE"][0]), float(e["PRIOR_RANGE"][1]), bool(par.is_log_row(e))) for e in table}
        expected = {k: DECIDED_RANGES[k] for k in DECIDED_KEYS_OF_CONDITION[condition]}
        order_ok = keys == DECIDED_KEYS_OF_CONDITION[condition]
        values_ok = order_ok and all(np.allclose(got[k][:2], expected[k][:2], atol=1e-12) and got[k][2] == expected[k][2]
                                     for k in expected)
        linear = [k for k in keys if not got[k][2]]
        per_condition[condition] = dict(keys=keys, order_as_decided=order_ok, ranges_and_scales_as_decided=values_ok,
                                        linear_rows=linear, ok=order_ok and values_ok)
    ranges_match = all(v["ok"] for v in per_condition.values())
    master_ok = list(par.MASTER_PARAMETER_KEYS) == list(DECIDED_RANGES)
    table = table_of("INLB")                            # the shared rows read from the twelve-row table
    e_da = table[key_index(RDS.mobility.diffusivity_key, "INLB")]
    e_ri = table[key_index("relative_diffusivity_immobile", "INLB")]
    e_rs = table[key_index("relative_diffusivity_slow", "INLB")]
    d_a_center, d_a_max = par.prior_center(e_da), float(par.entry_to_physical(e_da, e_da["PRIOR_RANGE"][1]))
    r_i_max = float(par.entry_to_physical(e_ri, e_ri["PRIOR_RANGE"][1]))
    r_s_min = float(par.entry_to_physical(e_rs, e_rs["PRIOR_RANGE"][0]))
    immobile_at_center = r_i_max * d_a_center <= IMMOBILE_THRESHOLDS_UM2_S[1]
    immobile_at_ceiling_strict = r_i_max * d_a_max <= IMMOBILE_THRESHOLDS_UM2_S[0]   # reported, not required
    decade_gap = r_s_min / r_i_max >= 10.0 - 1e-9
    e_off = table[key_index(RDS.stoichiometry.dissociation_rate_key, "INLB")]
    k_off_min = float(par.entry_to_physical(e_off, e_off["PRIOR_RANGE"][0]))
    loss_20s = 1.0 - np.exp(-k_off_min * 20.0)
    count_box_ok = tuple(par.COUNT_TOTAL_BOX) == (3.0, 5.0) and tuple(par.FRACTION_BOX) == (0.0, 1.0)
    return dict(per_condition=per_condition, ranges_match_decided_table=ranges_match, master_order_as_decided=master_ok,
                d_i_max_at_d_a_center=r_i_max * d_a_center,
                immobile_below_looser_threshold_at_center=bool(immobile_at_center),
                immobile_below_stricter_threshold_at_d_a_ceiling=bool(immobile_at_ceiling_strict),
                slow_over_immobile_gap=r_s_min / r_i_max, decade_gap=bool(decade_gap),
                dissociation_floor_loss_over_20s=float(loss_20s), count_and_fraction_boxes_ok=count_box_ok,
                ok=ranges_match and master_ok and immobile_at_center and decade_gap and loss_20s <= 0.02 and count_box_ok)


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


def frame0_class_counts(tray, condition: str) -> dict:
    """Frame-0 particle counts per molecular species (A, B under FAB; A, B1, B2 under INLB) from the
    particles observable, summed over the mobility modes."""
    _, types, _, _ = tray.read_observable_particles()
    name_of_id = {int(v): k for k, v in tray.particle_types.items()}
    counts = {sp.name: 0 for sp in RDS.species_for(condition)}
    for t in types[0]:
        counts[RDS.molecule_of_type[name_of_id[int(t)]]] += 1
    return counts


def starting_class_counts(comp) -> dict:
    """The STARTING counts the helper places, keyed like ``frame0_class_counts``."""
    return dict(rds.initial_species_counts(comp))


def r1_r5_reactive(workdir: str, condition: str) -> tuple:
    theta = prior_center_theta(condition)
    tray, timing = _run(theta, condition, R1_SECONDS, workdir, f"structure_audit_center_{condition}.h5")
    poses = rds.extract_trajectory_poses(tray)
    lineage = rds.extract_subunit_lineage(tray)
    comp = rds.initial_composition_of(theta, condition)
    species_of_rank = rds.rank_to_species(tray)
    # per-frame represented subunit total from the particles observable (independent of the lineage)
    _, types, ids, _ = tray.read_observable_particles()
    subunits_of_rank = {}
    for name, n_sub in zip(RDS.particle_type_names, RDS.subunit_counts_per_type):
        if name in tray.particle_types:
            subunits_of_rank[int(tray.particle_types[name])] = n_sub
    totals = np.array([sum(subunits_of_rank[int(t)] for t in types[f]) for f in range(lineage.n_frames)])
    frame0 = frame0_class_counts(tray, condition)
    starting = starting_class_counts(comp)
    box = PARAMETERS.simulation.stem.box_size
    xy = rds.collapse_species_axis(poses)[..., :2]
    present = np.isfinite(xy).all(axis=-1)
    outside = ((xy[..., 0] < 0) | (xy[..., 0] > box[0]) | (xy[..., 1] < 0) | (xy[..., 1] > box[1])) & present
    r1 = dict(condition=condition, n_subunits=lineage.n_subunits, n_total_realized=comp.n_total,
              frame0_counts=frame0, starting_counts=starting, frame0_equals_starting=frame0 == starting,
              pairs_allocated=int(comp.n_pairs_allocated), reference_counts=dict(
                  monomers=int(comp.n_monomers_reference), dimers=int(comp.n_dimers_reference),
                  two_probe=int(comp.n_dimers_two_probe_reference)),
              subunit_total_constant=bool(np.all(totals == totals[0])), first_total=int(totals[0]),
              ok=(lineage.n_subunits == comp.n_total and bool(np.all(totals == comp.n_total)) and frame0 == starting))
    r5 = dict(condition=condition, frames=int(present.shape[0]), particles_outside_last_frame=int(outside[-1].sum()),
              fraction_outside_mean=float(outside.sum() / max(present.sum(), 1)),
              subunit_total_constant=r1["subunit_total_constant"],
              note="the open lateral boundary keeps particles simulated outside the imaged field; "
                   "the total is conserved regardless (in-field count != N_R).",
              ok=r1["subunit_total_constant"])
    return r1, r5, lineage, tray, species_of_rank


def r2_stationarity(workdir: str) -> dict:
    theta = prior_center_theta("FAB")
    theta[key_index(F_KEY, "FAB")] = 0.0                 # f = 0 (inside the prior): monomers only
    # about 600 represented monomers: the receptor total that the FAB occupancy thins to 600 (a
    # monomer-only chain needs the particles, not the box)
    theta[key_index(N_KEY, "FAB")] = 600.0 / par.occupancy_of("FAB")
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
    pi = rds.stationary_mode_law(theta, "FAB")
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
    """Wall time of one 2 s run per condition at the simulated-population ceiling
    (N_total = 100,000, D_A at its ceiling; MET-FAB at f = 1 with kappa_OFF at its ceiling, 10 per
    s: 1,466 initial dimers that split within the run toward 2,932 particles, one per represented
    receptor; MET-INLB at f = 0 and R_active = 0: 4,762 monomers, kappa_OFF at the prior center),
    plus a 20-frame render of each (descriptive). The particle counts at the first and last frame
    and their peak are read from the run, so the load reached is reported rather than assumed. A
    high-load benchmark, not a runtime bound. The memory figures are the process's peak resident size
    (ru_maxrss) before and after the run: cumulative over the whole audit, so an unchanged value was
    reached earlier and bounds the run without measuring it; never an isolated measurement."""
    import resource
    import time
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video
    imaging = np.array([10 ** ((e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2) for e in det.DETECTOR_IMAGING])
    out = {}
    for condition in RDS.condition_tokens:
        theta = prior_center_theta(condition)
        table = table_of(condition)
        e_n = table[key_index(N_KEY, condition)]
        e_d = table[key_index(RDS.mobility.diffusivity_key, condition)]
        theta[key_index(N_KEY, condition)] = float(par.entry_to_physical(e_n, e_n["PRIOR_RANGE"][1]))   # 100,000 receptors
        theta[key_index(e_d["KEY"], condition)] = float(par.entry_to_physical(e_d, e_d["PRIOR_RANGE"][1]))   # fastest diffusion
        if RDS.condition_setting(condition).ligand_classes:
            theta[key_index(F_KEY, condition)] = 0.0       # all monomers, no allocation: the most particles
            theta[key_index(R_KEY, condition)] = 0.0
        else:
            theta[key_index(F_KEY, condition)] = 1.0       # all dimers: the most represented receptors
            # fastest dissociation, so the dimers split within the 2 s run (at the prior center,
            # 0.1 per s, only about 18 % would, ending near 1,730 particles)
            e_k = table[key_index(RDS.stoichiometry.dissociation_rate_key, condition)]
            theta[key_index(e_k["KEY"], condition)] = float(par.entry_to_physical(e_k, e_k["PRIOR_RANGE"][1]))
        comp = rds.initial_composition_of(theta, condition)
        koff = float(theta[key_index(RDS.stoichiometry.dissociation_rate_key, condition)])
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        tray, timing = _run(theta, condition, 2.0, workdir, f"structure_audit_timing_{condition}.h5")
        wall = time.perf_counter() - t0
        _, frame_types, _, _ = tray.read_observable_particles()
        particles = np.array([len(t) for t in frame_types])          # the load reached, per frame
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
                              fraction_basal_dimers_reference=float(comp.fraction_basal_dimers_reference),
                              pairs_allocated=int(comp.n_pairs_allocated),
                              n_subunits=comp.n_subunits, n_particles=comp.n_complexes, n_monomers=comp.n_monomers,
                              n_dimers=comp.n_dimers, rate_dissociation=koff,
                              particles_first_frame=int(particles[0]), particles_last_frame=int(particles[-1]),
                              particles_peak=int(particles.max()), wall_seconds=float(wall),
                              wall_seconds_per_simulated_second=float(wall / 2.0),
                              lineage_and_render_20_frames_seconds=float(render_wall),
                              peak_rss_mib_before=rss_before / 1024.0, peak_rss_mib_after=rss_after / 1024.0,
                              ok=bool(np.isfinite(wall)))
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


R7_MOVEMENT_ALLOWANCE_NM = 1.0   # R7: residual diffusion of the near-frozen run (about 0.3 nm per frame)


def r7_conversion_placement(workdir: str) -> dict:
    """Measured placement of the `B1 -> A` conversion product and of the fission daughters.

    One 0.2 s MET-INLB run with diffusion switched almost off (D_A = 1e-6 um^2/s: 0.3 nm per frame),
    kappa_OFF at the prior ceiling (events within the run), N_total at the box ceiling and f = 1
    (most B1; no probe-bound monomer pair exists, so the allocation is void). For every `B1 -> A`
    record and every fission record the product's position at the first frame at or after the step
    is compared with the educt's at the last frame before it (in-plane distance, nm). Events whose
    educt or product is not present at those frames (a mode switch within the same frame) are
    skipped and counted.
    """
    condition = "INLB"
    theta = prior_center_theta(condition)
    table = table_of(condition)
    e_n = table[key_index(N_KEY, condition)]
    e_k = table[key_index(RDS.stoichiometry.dissociation_rate_key, condition)]
    theta[key_index(N_KEY, condition)] = float(par.entry_to_physical(e_n, e_n["PRIOR_RANGE"][1]))
    theta[key_index(F_KEY, condition)] = 1.0
    theta[key_index(e_k["KEY"], condition)] = float(par.entry_to_physical(e_k, e_k["PRIOR_RANGE"][1]))
    theta[key_index(RDS.mobility.diffusivity_key, condition)] = 1e-6
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
                                      and f["max_nm"] <= kick + R7_MOVEMENT_ALLOWANCE_NM)
    out.update(enough_events=enough, conversion_in_place=in_place, fission_as_declared=fission_as_declared,
               fission_max_allowed_nm=kick + R7_MOVEMENT_ALLOWANCE_NM, movement_allowance_nm=R7_MOVEMENT_ALLOWANCE_NM,
               ok=bool(enough and in_place and fission_as_declared))
    return out


R8_DIMERS = 20_000


def geminate_refusions(fission_records, fusion_records) -> tuple:
    """Count the fusions whose two educts are the two products of one earlier fission (the same
    daughter pair re-fusing: geminate), apart from the fusions of daughters of different dimers
    (chance encounters). ``fission_records`` and ``fusion_records`` are sequences of ReaDDy reaction
    records (``.type``, ``.educts``, ``.products``); returns ``(geminate, chance)``."""
    pairs = {frozenset(int(x) for x in np.atleast_1d(rec.products))
             for rec in fission_records if str(rec.type) == "fission"}
    geminate = chance = 0
    for rec in fusion_records:
        if str(rec.type) != "fusion":
            continue
        if frozenset(int(x) for x in np.atleast_1d(rec.educts)) in pairs:
            geminate += 1
        else:
            chance += 1
    return geminate, chance


def r8_placement_geometry(workdir: str) -> dict:
    """Measured daughter-pair geometry of the fission placement in the production box and slab.

    A two-type system (one dimer type that fissions into two of one monomer type, both at
    D = 1e-6 um^2/s, immobile on the step) in the production box with the production reaction
    distance and fission distance: 20,000 dimers, fission at 5000/s (every dimer fissions in the
    first step), fusion of two monomers within the reaction distance at 1e5/s, four 2 ms steps.
    From the reaction records and the frames: the in-plane separation of every daughter pair at the
    frame after the fission and the fraction within the reaction distance, against the geometric
    fraction q of D7; and the fusions at the following step whose educts are the two daughters of
    one fission (``geminate_refusions``; the fusions of daughters of different dimers are counted
    apart as chance encounters), over the fissions of the first step: the next-step re-fusion of an
    immobile daughter pair, also expected q. The tolerance is four sampling standard deviations of
    q at this count (about 0.5 %), so a PASS places both measured fractions within about 0.5 % of
    q, below the 4 % the geometric criterion of D7 allows; the measured values are reported.
    """
    import readdy
    geom = PARAMETERS.simulation.stem
    r_nm, d_nm = float(geom.particle_diameter_nm), float(geom.fission_product_distance_nm)
    box = [float(x) for x in geom.box_size]
    system = readdy.ReactionDiffusionSystem(box_size=box, unit_system=geom.unit_dict)
    system.periodic_boundary_conditions = [False, False, True]
    tiny = 1e-6 * readdy.units.micrometer ** 2 / readdy.units.second
    system.add_species("DIMER", tiny)
    system.add_species("MONO", tiny)
    system.reactions.add_fission(name="fis", type_from="DIMER", type_to1="MONO", type_to2="MONO",
                                 rate=5000.0 / readdy.units.second, product_distance=d_nm, weight1=0.5, weight2=0.5)
    system.reactions.add_fusion(name="fus", type_from1="MONO", type_from2="MONO", type_to="DIMER",
                                rate=1e5 / readdy.units.second, educt_distance=r_nm, weight1=0.5, weight2=0.5)
    sim = system.simulation(kernel="CPU")
    rng = np.random.default_rng(SEED)
    lo, hi = [-b / 2.0 for b in box], [b / 2.0 for b in box]
    sim.add_particles("DIMER", rng.uniform(lo, hi, size=(R8_DIMERS, 3)))
    sim.observe.particles(stride=1)
    sim.observe.reaction_counts(stride=1)
    sim.observe.reactions(stride=1)
    path = os.path.join(workdir, "structure_audit_placement_geometry.h5")
    sim.output_file = path
    sim.run(n_steps=4, timestep=PARAMETERS.simulation.timing.delta_time_nanoseconds * readdy.units.nanosecond,
            show_summary=False)
    tray = readdy.Trajectory(path)
    _, recs = tray.read_observable_reaction_counts()
    fis = [int(x) for x in recs["reactions"]["fis"]]
    fus = [int(x) for x in recs["reactions"]["fus"]]
    times, rlists = tray.read_observable_reactions()
    by_step = {int(t): list(rl) for t, rl in zip(times, rlists)}
    _spans, _ranks, ids, poses = tray.read_observable_particles()
    pos = {int(s): np.asarray(p, dtype=float) for s, p in zip(ids[1], poses[1])}       # the frame after step 1
    sep = []
    for rec in by_step.get(1, []):
        if str(rec.type) != "fission":
            continue
        p1, p2 = (pos.get(int(x)) for x in np.atleast_1d(rec.products))
        if p1 is None or p2 is None:
            continue
        delta = p2 - p1
        sep.append(float(np.hypot(delta[0], delta[1])))
    geminate, chance = geminate_refusions(by_step.get(1, []), by_step.get(2, []))
    del tray, sim, system
    sep = np.asarray(sep, dtype=float)
    q = in_plane_within_fraction(r_nm, d_nm)
    n = int(sep.size)
    n_fissions_1 = fis[1] if len(fis) > 1 else 0
    within = float(np.mean(sep <= r_nm)) if n else float("nan")
    refusion = float(geminate / n_fissions_1) if n_fissions_1 else float("nan")
    tol = float(4.0 * np.sqrt(q * (1.0 - q) / max(n, 1)))
    mean_ok = n > 0 and abs(sep.mean() - np.pi / 4.0 * d_nm) <= 5.0 * sep.std() / np.sqrt(n)
    ok = bool(n >= 0.9 * R8_DIMERS and abs(within - q) <= tol and abs(refusion - q) <= tol and mean_ok)
    return dict(n_dimers=R8_DIMERS, fissions_per_step=fis, fusions_per_step=fus, n_pairs_measured=n,
                pair_in_plane_separation_mean_nm=float(sep.mean()) if n else float("nan"),
                expected_mean_nm=float(np.pi / 4.0 * d_nm),
                pair_in_plane_separation_max_nm=float(sep.max()) if n else float("nan"),
                fraction_within_reaction_distance=within,
                geminate_refusions_next_step=int(geminate), chance_encounter_fusions_next_step=int(chance),
                next_step_refusion_fraction=refusion, expected_fraction=q, tolerance=tol, ok=ok)

# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

def write_report(det_out: dict, run_out: dict | None) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    L = ["# Model-structure audit: separated stoichiometry-mobility generator", "",
         f"Repository `srm-and-sbi-monomer-dimer-alp`; particle types {RDS.particle_type_names}; "
         f"parameters per condition { {c: par.parameter_keys_of(c) for c in RDS.condition_tokens} } "
         f"(declared rows { {c: par.declared_row_values(c) for c in RDS.condition_tokens} }); condition association ratios "
         f"{ {c: RDS.association_ratio_of(c) for c in RDS.condition_tokens} }.", "",
         "## Deterministic tier", "", "| check | verdict | detail |", "|---|---|---|"]
    d = det_out
    for c in RDS.condition_tokens:
        dc = d["d1"][c]
        L.append(f"| D1 channels under {c} (R_ON = {dc['association_ratio']:g}) | {passed(dc['ok'])} | {dc['n_channels']} channels "
                 f"(expected {dc['expected']['total']}), {dc['n_unique_names']} unique names, kinds {dc['kinds']}, "
                 f"association rate shared/absent as declared: {dc['one_shared_association_rate']}, "
                 f"ReaDDy names match: {dc['registered_matches']} |")
    for c in RDS.condition_tokens:
        dc = d["d2"][c]
        L.append(f"| D2 diffusion ordering under {c} | {passed(dc['ok'])} | {dc['n_draws']} prior draws incl. box corners; "
                 f"immobile < slow <= fast within species: {dc['within_species_ordering']}; dimer <= monomer within "
                 f"mode: {dc['dimer_not_faster_than_monomer']}; closed form: {dc['closed_form_matches']} |")
    for c in RDS.condition_tokens:
        dc = d["d3"][c]
        L.append(f"| D3 transforms under {c} ({dc['n_rows']} rows) | {passed(dc['ok'])} | max round-trip error "
                 f"{dc['max_roundtrip_error']:.2e}; linear rows {dc['linear_rows']} as decided: {dc['linear_rows_as_decided']}; "
                 f"blanket 10**u differs from to_physical on them: {dc['blanket_power_differs_on_linear_rows']} and equals it on "
                 f"the log rows: {dc['blanket_power_equals_to_physical_on_log_rows']}; linear rows pass through: "
                 f"{dc['linear_rows_pass_through']}; conversions without a table refused: {dc['conversions_without_table_refused']}; "
                 f"wrong width refused: {dc['wrong_width_refused']}; table values are prior centers: "
                 f"{dc['table_values_are_prior_centers']}; no prior draw flagged: {dc['no_prior_draw_flagged']} |")
    ceil = "; ".join(f"{c} " + ", ".join(f"{k}: N_R {v['represented_receptors']}, {v['initial_particles']} particles "
                                          f"({v['monomers']} monomers + {v['dimers']} dimers), probe-bound "
                                          f"{v['probe_bound_receptors_expected']:.0f}" for k, v in x.items())
                     for c, x in d["d4"]["ceilings_at_count_ceiling"].items())
    L.append(f"| D4 reference population and association reference | {passed(d['d4']['ok'])} | {d['d4']['n_cases']} "
             f"(N_total, f) cases per condition against the ratio-form literal at zero allocation (grid "
             f"{d['d4']['grid']['n_total'][2]} x {d['d4']['grid']['fraction'][2]} over the box x [0, 1] plus "
             f"{d['d4']['n_random_draws_per_condition']} random draws); N_R within {d['d4']['subunit_tolerance']} of its "
             f"expectation; D_sim monotone in f: {d['d4']['dimers_monotone_in_fraction']}; refusals: {d['d4']['refusals_ok']}; "
             f"problems: {d['d4']['problems'] or 'none'}; ceilings at N_total = 100,000 (reported): {ceil}; "
             f"lambda_ref = 6 D_A / r^2 = Smoluchowski/volume: {d['d4']['association_reference_is_smoluchowski']}; "
             f"INLB fusions at exactly lambda_ref: {d['d4']['lambda_on_equals_reference_under_INLB']}; "
             f"no fusion channel under FAB: {d['d4']['no_fusion_channel_under_FAB']} |")
    L.append(f"| D5 stationary mode law | {passed(d['d5']['ok'])} | sums to one: {d['d5']['sums_to_one']}; "
             f"detailed balance on every link: {d['d5']['detailed_balance']} |")
    L.append(f"| D6 occupancy by molecular species | {passed(d['d6']['ok'])} | identical within species: "
             f"{d['d6']['occupancy_identical_within_species']}; keyed by particle type rejected: "
             f"{d['d6']['occupancy_keyed_by_particle_type_rejected']}; legacy A/B/C trajectory rejected: "
             f"{d['d6']['legacy_species_trajectory_rejected']} |")
    L.append(f"| D7 fission placement | {passed(d['d7']['ok'])} | daughters {d['d7']['product_distance_nm']:.0f} nm apart = 4 x reaction distance "
             f"{d['d7']['reaction_distance_nm']:.0f} nm: {d['d7']['declared_factor_four']}; build_system passes the field: {d['d7']['build_system_uses_field']}; "
             f"isotropic daughter axis in the 2 nm slab: fraction of pairs within the reaction distance in the plane "
             f"{d['d7']['in_plane_within_reaction_distance_fraction']:.3f} (accepted <= {d['d7']['accepted_residual']:.2f}; "
             f"{d['d7']['in_plane_within_fraction_at_former_20_nm']:.3f} at the former 20 nm); next-step re-fusion by daughter mode at the prior center: "
             f"{ {k: round(v, 3) for k, v in d['d7']['next_step_refusion_fraction_by_mode_prior_center'].items()} }; one-step rebinding if placed AT the "
             f"radius (sub-step {d['d7']['sub_step_s']*1e3:.0f} ms): { {k: round(v, 3) for k, v in d['d7']['one_step_rebind_probability_if_placed_at_radius'].items()} } |")
    L.append(f"| D8 condition registry | {passed(d['d8']['ok'])} | tokens {d['d8']['tokens']} agree across registries: "
             f"{d['d8']['tokens_agree_across_registries']}; FAB ratio 0.0: {d['d8']['fab_ratio_is_zero']}; INLB ratio 1.0: "
             f"{d['d8']['inlb_ratio_is_one']}; retired rows {d['d8']['retired_keys']} absent from both tables, the master and "
             f"the detector mirror: {d['d8']['retired_rows_absent']}; detector mirror = the twelve master keys: "
             f"{d['d8']['detector_mirror_equals_master']}; Fab keys = InlB keys[:11]: {d['d8']['fab_keys_equal_inlb_prefix']}; "
             f"declared rows as decided (FAB allocation 0, INLB inferred): {d['d8']['declared_rows_as_decided']}; removed "
             f"symbols absent: {d['d8']['removed_symbols_absent']}; allocation-setting refusals "
             f"{d['d8']['allocation_setting_refusals']}/4 and the declared variants accepted: {d['d8']['allocation_setting_ok']}; "
             f"disguised zero refused: {d['d8']['disguised_zero_refused']}; bare rds_alias refused: "
             f"{d['d8']['bare_rds_alias_refused']}; tier aliases {d['d8']['tier_aliases']}; detector reads the same tier: "
             f"{d['d8']['detector_reads_same_tier']}; occupancies {d['d8']['occupancy']} ({d['d8']['occupancy_source']}), "
             f"visibilities { {c: round(v, 5) for c, v in d['d8']['visibility'].items()} }, two-probe shares "
             f"{d['d8']['two_probe_share']}, ligand classes {d['d8']['ligand_classes']}; FAB derived from the "
             f"declared ratio and the INLB anchor: {d['d8']['fab_occupancy_derived_ok']}; INLB equilibrium at 0.25 nM / K_D 5 nM: "
             f"{d['d8']['inlb_occupancy_equilibrium_ok']}; both/neither refused: {d['d8']['both_and_neither_refused']}; "
             f"association without classes refused: {d['d8']['association_without_classes_refused']} |")
    per = "; ".join(f"{c}: {len(v['keys'])} rows, linear {v['linear_rows']}, as decided {v['ok']}"
                    for c, v in d["d9"]["per_condition"].items())
    L.append(f"| D9 decided ranges and scales | {passed(d['d9']['ok'])} | tables equal the decided literal ({per}); master order "
             f"as decided: {d['d9']['master_order_as_decided']}; count box [3, 5] and fraction box (0, 1): "
             f"{d['d9']['count_and_fraction_boxes_ok']}; "
             f"D_i max at the D_A center {d['d9']['d_i_max_at_d_a_center']:.4f} um^2/s below the looser immobility threshold: "
             f"{d['d9']['immobile_below_looser_threshold_at_center']} (below the stricter one at the D_A ceiling: "
             f"{d['d9']['immobile_below_stricter_threshold_at_d_a_ceiling']}, reported); slow/immobile gap "
             f"{d['d9']['slow_over_immobile_gap']:.1f}x (>= one decade: {d['d9']['decade_gap']}); dissociation floor loses "
             f"{100 * d['d9']['dissociation_floor_loss_over_20s']:.1f}% over 20 s |")
    for c in RDS.condition_tokens:
        dc = d["d10"][c]
        L.append(f"| D10 theta schema under {c} ({dc['schema_width']} columns) | {passed(dc['ok'])} | round trip .zarr/.npy: "
                 f"{dc['roundtrip.zarr']}/{dc['roundtrip.npy']}; schema requires the condition: {dc['schema_requires_condition']}; "
                 f"refuses renamed key: {dc['refuses_renamed_key']}, shifted bound: {dc['refuses_shifted_bound']}, "
                 f"other condition: {dc['refuses_other_condition']}, other condition's table: {dc['refuses_other_conditions_table']}, "
                 f"the retired ratio table: {dc['refuses_retired_ratio_table']} and a record drawn under it: "
                 f"{dc['refuses_record_of_retired_ratio_table']}, schema-less .zarr: {dc['refuses_schemaless_zarr']}, "
                 f".npy without sidecar: {dc['refuses_npy_without_sidecar']}; status names the reason: "
                 f"{dc['status_names_reason']}; package {dc['schema_package_version']} |")
    L.append(f"| D11 conversion displacement | {passed(d['d11']['ok'])} | B1 -> A is a conversion for every mode "
             f"(product in place): {d['d11']['conversion_in_place']}; no channel re-forms B1 (at most one release per B1 "
             f"lineage): {d['d11']['b1_never_formed']}; a fission places each daughter {d['d11']['fission_daughter_kick_nm']:.0f} nm "
             f"from the parent (fixed 3D distance, random direction; {d['d11']['in_plane_mean_kick_nm']:.1f} nm in-plane on average) "
             f"-- the per-event difference between the two placements; against the per-frame diffusive RMS of a monomer at the "
             f"prior floor: " + ", ".join(f"{m} {v:.1f} nm ({d['d11']['kick_over_rms'][m]:.2f}x)" for m, v in d['d11']['per_frame_rms_at_prior_floor_nm'].items())
             + f"; against the localization precision {d['d11']['localization_precision_nm']:.0f} nm: {d['d11']['kick_over_precision']:.2f}x; "
             f"against the pixel {d['d11']['pixel_nm']:.0f} nm: {d['d11']['kick_over_pixel']:.3f}x |")
    w, agree = d["d12"]["worked_literals"], d["d12"]["helper_vs_initializer"]
    L.append(f"| D12 allocation | {passed(d['d12']['ok'])} | {d['d12']['n_cases']} grid cases over N_total {d['d12']['grid']['n_total']} "
             f"x f {d['d12']['grid']['fraction']} x R_active {d['d12']['grid']['allocation']} ({d['d12']['half_integer_products']} "
             f"half-integer products, rounded half to even); capacity A_ref // 2, c = round(R_active cap), conservation, B1 and the "
             f"reference unchanged, odd A_ref leaves one monomer at R_active = 1, zero capacity allocates nothing, the derived "
             f"underlying quantities, c monotone in R_active, Fab start = reference; worked literals (N_total = 1e4, f = 0.2: A_ref "
             f"{w['n1e4_f02']['A_ref']}, B1 {w['n1e4_f02']['B1']}, B2_ref {w['n1e4_f02']['B2_ref']}, capacity {w['n1e4_f02']['capacity']}, "
             f"c {w['n1e4_f02']['c_at_half']} at 0.5 and {w['n1e4_f02']['c_at_one']} at 1, {w['n1e4_f02']['monomers_left_at_one']} monomer "
             f"left; the InlB prior center {w['prior_center']['A']} A + {w['prior_center']['B1']} B1 + {w['prior_center']['B2']} B2, "
             f"c {w['prior_center']['c']}, f_start {w['prior_center']['f_start']:.3f}, active share {w['prior_center']['active_share']:.3f}): "
             f"{d['d12']['worked_literals_ok']}; vectorized helper vs one-draw initializer on prior draws: "
             + ", ".join(f"{c} {v['mismatches']}/{v['n_draws']} mismatches ({v['draws_with_allocation']} draws with c > 0)"
                         for c, v in agree.items())
             + f"; problems: {d['d12']['problems'] or 'none'} |")
    for c in RDS.condition_tokens:
        L += ["", f"Channels under {c}:", ""] + [f"- `{n}`" for n in d["d1"][c]["names"]] + [""]
    if run_out is None:
        L += ["## Run tier", "", "Not executed (`--run` not given). The run tier needs explicit approval.", ""]
    else:
        r = run_out
        L += ["## Run tier (tiny simulations)", "", "| check | verdict | detail |", "|---|---|---|"]
        for c in RDS.condition_tokens:
            rc = r["r1"][c]
            L.append(f"| R1 conservation {c} | {passed(rc['ok'])} | frame-0 counts {rc['frame0_counts']} = starting counts "
                     f"{rc['starting_counts']}: {rc['frame0_equals_starting']} (c = {rc['pairs_allocated']} pairs allocated from "
                     f"the reference {rc['reference_counts']}); lineage {rc['n_subunits']} subunits = realized "
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
                 + "; ".join(f"{c}: N_total {x['n_total']:.0f}, f = {x['fraction_basal_dimers_reference']:g}, c = "
                             f"{x['pairs_allocated']} -> {x['n_subunits']} represented receptors in {x['n_particles']} initial particles "
                             f"({x['n_monomers']} monomers + {x['n_dimers']} dimers), D_A at its ceiling, kappa_OFF "
                             f"{x['rate_dissociation']:g} per s; particles {x['particles_first_frame']} at the first frame -> "
                             f"{x['particles_last_frame']} at the last (peak {x['particles_peak']}); {x['seconds_simulated']:.0f} s "
                             f"simulated in {x['wall_seconds']:.1f} s wall ({x['wall_seconds_per_simulated_second']:.1f} s per simulated "
                             f"second), lineage + 20-frame render {x['lineage_and_render_20_frames_seconds']:.1f} s, process peak RSS "
                             f"{x['peak_rss_mib_before']:.0f} -> {x['peak_rss_mib_after']:.0f} MiB (cumulative over "
                             f"the audit, not isolated to this run)"
                             for c, x in r['r6'].items() if isinstance(x, dict))
                 + "; descriptive: a high-load benchmark, not a runtime bound |")
        p7 = r["r7"]
        L.append(f"| R7 conversion and fission placement (measured) | {passed(p7['ok'])} | MET-INLB, {p7['seconds_simulated']:g} s, "
                 f"D_A = 1e-6 um^2/s, kappa_OFF at the ceiling, {p7['n_b1_initial']} B1 + {p7['n_b2_initial']} B2 at onset; "
                 f"B1 -> A products: n = {p7['conversion']['n']} ({p7['conversion']['skipped']} skipped), in-plane displacement "
                 f"mean {p7['conversion']['mean_nm']:.2f} nm, max {p7['conversion']['max_nm']:.2f} nm (in place: {p7['conversion_in_place']}); "
                 f"fission daughters: n = {p7['fission']['n']} ({p7['fission']['skipped']} skipped), mean {p7['fission']['mean_nm']:.2f} nm "
                 f"(expected pi/4 x {p7['expected_fission_3d_nm']:.0f} = {p7['expected_fission_in_plane_mean_nm']:.2f}), sd "
                 f"{p7['fission']['sd_nm']:.2f}, max {p7['fission']['max_nm']:.2f} nm (<= {p7['fission_max_allowed_nm']:.0f} nm, including "
                 f"the {p7['movement_allowance_nm']:.0f} nm movement allowance: {p7['fission_as_declared']}) |")
        p8 = r["r8"]
        L.append(f"| R8 placement geometry (measured) | {passed(p8['ok'])} | {p8['n_dimers']} immobile dimers in the production box and slab; "
                 f"fissions per step {p8['fissions_per_step']}, fusions per step {p8['fusions_per_step']}; daughter pairs measured "
                 f"{p8['n_pairs_measured']}: in-plane separation mean {p8['pair_in_plane_separation_mean_nm']:.2f} nm (expected pi/4 x d = "
                 f"{p8['expected_mean_nm']:.2f}), max {p8['pair_in_plane_separation_max_nm']:.2f}; fraction within the reaction distance "
                 f"{p8['fraction_within_reaction_distance']:.4f}; same-pair re-fusions at the next step {p8['geminate_refusions_next_step']} "
                 f"(fraction {p8['next_step_refusion_fraction']:.4f}), chance encounters of daughters of different dimers "
                 f"{p8['chance_encounter_fusions_next_step']}; expected {p8['expected_fraction']:.4f} +/- {p8['tolerance']:.4f} |")
        L.append("")
    with open(REPORT, "w") as handle:
        handle.write("\n".join(L) + "\n")
    with open(os.path.join(OUT_DIR, "audit_summary.json"), "w") as handle:
        json.dump(dict(deterministic=det_out, run=run_out), handle, indent=2, default=str)


def d10_theta_schema() -> dict:
    """Theta_Set schema round trip and refusals per condition (io.write_theta_set / load_theta_set /
    theta_set_status), the audit-local table carrying the retired ratio row included."""
    out = {c: _d10_condition(c) for c in RDS.condition_tokens}
    out["ok"] = all(v["ok"] for v in out.values())
    return out


def _d10_condition(condition: str) -> dict:
    from srm_and_sbi_monomer_dimer_alp import io as sio
    import zarr
    table = table_of(condition)
    other = next(c for c in RDS.condition_tokens if c != condition)
    rng = np.random.default_rng(20260914)
    low, high = prior_box(condition)
    theta = par.to_physical(rng.uniform(low, high, size=(5, len(low))), table)
    schema = sio.theta_set_schema(table, condition=condition, timing_label="2S_50FPS", generator="audit")
    out = dict(condition=condition, schema_keys=list(schema["parameter_keys"]) == par.parameter_keys_of(condition),
               schema_width=len(schema["parameter_keys"]), schema_package_version=schema["package_version"])

    def refused(path, tbl, cond=condition):
        try:
            sio.load_theta_set(path, tbl, condition=cond)
            return False
        except sio.ThetaSetSchemaError:
            return True

    condition_required = False
    try:
        sio.theta_set_schema(table, condition=None, timing_label="2S_50FPS", generator="audit")
    except ValueError:
        condition_required = True
    out["schema_requires_condition"] = condition_required
    with tempfile.TemporaryDirectory() as d:
        for ext in (".zarr", ".npy"):
            p = os.path.join(d, "Theta_Set" + ext)
            sio.write_theta_set(p, theta, schema)
            back = np.asarray(sio.load_theta_set(p, table, condition=condition)[:])
            out["roundtrip" + ext] = bool(np.allclose(back, theta))
            out["status_ok" + ext] = sio.theta_set_status(p, table, condition=condition) == "OK"
        p = os.path.join(d, "Theta_Set.zarr")
        renamed = [dict(e) for e in table]; renamed[1]["KEY"] = "fraction_dimer_initial"
        shifted = [dict(e) for e in table]; shifted[0]["PRIOR_RANGE"] = (2.0, 3.5)
        # the superseded table: the condition's table with the basal fraction row replaced by the retired
        # log10 ratio row (an audit-local literal); a Theta_Set drawn under it must be refused, and so
        # must a current record read under it
        ratio_table = [dict(e) for e in table]; ratio_table[1] = dict(RETIRED_RATIO_ROW)
        out["refuses_renamed_key"] = refused(p, renamed)
        out["refuses_shifted_bound"] = refused(p, shifted)
        out["refuses_other_condition"] = refused(p, table, other)
        out["refuses_other_conditions_table"] = refused(p, table_of(other))
        out["refuses_retired_ratio_table"] = refused(p, ratio_table)
        out["status_names_reason"] = sio.theta_set_status(p, renamed, condition=condition).startswith("SCHEMA MISMATCH")
        ratio_set = os.path.join(d, "Ratio_Theta_Set.zarr")
        sio.write_theta_set(ratio_set, theta, sio.theta_set_schema(ratio_table, condition=condition,
                                                                   timing_label="2S_50FPS", generator="audit"))
        out["refuses_record_of_retired_ratio_table"] = refused(ratio_set, table)
        bare = os.path.join(d, "Bare_Theta_Set.zarr")
        z = zarr.open(store=bare, mode="w", shape=theta.shape, chunks=(1, theta.shape[1]), dtype=np.float64)
        z[:, :] = theta
        out["refuses_schemaless_zarr"] = refused(bare, table)
        bare_npy = os.path.join(d, "Bare_Theta_Set.npy")
        np.save(bare_npy, theta)
        out["refuses_npy_without_sidecar"] = refused(bare_npy, table)
        out["status_missing_file"] = sio.theta_set_status(os.path.join(d, "none.zarr"), table, condition=condition) == "MISSING"
    out["ok"] = all(v for k, v in out.items() if isinstance(v, bool))
    return out


LOCALIZATION_PRECISION_NM = 29.0   # MET recordings (Rahm et al. 2021, Front. Comput. Sci. 3, 757653)


def d11_conversion_displacement() -> dict:
    """The `B1 -> A` release as an in-place conversion against the fission placement and the
    observable scales (arithmetic from the parameterization; R7 measures the two placements)."""
    geom = par.PARAMETERS.simulation.stem
    timing = par.PARAMETERS.simulation.timing
    channels = rds.reaction_channels(prior_center_theta("INLB"), "INLB")
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
    theta_floor = prior_center_theta("INLB")
    table = table_of("INLB")
    for key in (RDS.mobility.diffusivity_key,) + tuple(k for k in RDS.mobility.mode_ratio_keys if k):
        entry = table[key_index(key, "INLB")]
        theta_floor[key_index(key, "INLB")] = float(par.entry_to_physical(entry, entry["PRIOR_RANGE"][0]))
    coeff = rds.diffusion_coefficients(theta_floor, "INLB")
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
                        and kick_nm <= LOCALIZATION_PRECISION_NM and kick_nm <= pixel_nm / 5.0))


def d12_allocation() -> dict:
    """The window-start allocation (MET-INLB) and its absence (MET-FAB), against the specification."""
    problems = []
    n_grid = np.array([25.0, 1e3, 3.16e3, 1e4, 3.16e4, 1e5])
    f_grid = np.array([0.0, 0.05, 0.2, 0.5, 0.95, 1.0])
    r_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    nn, ff, rr = np.meshgrid(n_grid, f_grid, r_grid, indexing="ij")
    inlb = realize("INLB", nn, ff, rr)
    zero = realize("INLB", nn, ff, 0.0)                  # the same reference population, no allocation
    n_cases = int(nn.size)
    # capacity and c; conservation; B1 and the reference unchanged by the allocation
    if not np.array_equal(inlb.capacity, inlb.n_monomers_reference // 2):
        problems.append("INLB: capacity is not A_ref // 2")
    if not np.array_equal(inlb.n_pairs_allocated, np.round(rr * inlb.capacity).astype(np.int64)):
        problems.append("INLB: c is not round(R_active x capacity)")
    if np.any(inlb.n_pairs_allocated < 0) or np.any(inlb.n_pairs_allocated > inlb.capacity):
        problems.append("INLB: c outside [0, capacity]")
    if not np.array_equal(inlb.n_monomers, inlb.n_monomers_reference - 2 * inlb.n_pairs_allocated) or \
            not np.array_equal(inlb.n_dimers_two_probe, inlb.n_dimers_two_probe_reference + inlb.n_pairs_allocated) or \
            not np.array_equal(inlb.n_dimers, inlb.n_dimers_one_probe + inlb.n_dimers_two_probe):
        problems.append("INLB: the starting counts are not A_ref - 2c, B2_ref + c, B1 + B2_start")
    if not np.array_equal(inlb.n_subunits, inlb.n_monomers + inlb.n_dimers_one_probe + 2 * inlb.n_dimers_two_probe) or \
            not np.array_equal(inlb.n_subunits, zero.n_subunits):
        problems.append("INLB: N_R is not conserved through the allocation")
    for name in ("n_monomers_reference", "n_dimers_reference", "n_dimers_one_probe", "n_dimers_two_probe_reference",
                 "capacity", "n_monomers_underlying_reference", "n_dimers_underlying_reference"):
        if not np.array_equal(getattr(inlb, name), getattr(zero, name)):
            problems.append(f"INLB: {name} changes with R_active")
    if np.any(inlb.n_monomers < 0):
        problems.append("INLB: negative monomers after the allocation")
    # half-integer products R_active x capacity = x.5 at R_active = 0.5 and an odd capacity: rounded half to even
    odd = (inlb.capacity % 2 == 1) & (rr == 0.5)
    n_half = int(odd.sum())
    if n_half == 0:
        problems.append("INLB: the grid holds no half-integer product")
    else:
        c_odd, cap_odd = inlb.n_pairs_allocated[odd], inlb.capacity[odd]
        if not np.all(c_odd % 2 == 0) or not np.all(np.abs(c_odd - cap_odd / 2.0) == 0.5) or \
                not all(int(c) == int(round(0.5 * int(k))) for c, k in zip(c_odd, cap_odd)):
            problems.append("INLB: a half-integer product is not rounded half to even")
    # an odd A_ref leaves one monomer at R_active = 1; a zero capacity allocates nothing
    full = rr == 1.0
    odd_a = full & (inlb.n_monomers_reference % 2 == 1)
    if not np.all(inlb.n_monomers[odd_a] == 1) or not np.all(inlb.n_monomers[full & ~odd_a] == 0):
        problems.append("INLB: R_active = 1 does not leave A_ref mod 2 monomers")
    cap0 = inlb.capacity == 0
    if not np.any(cap0 & (ff == 1.0)) or not np.any(cap0 & (nn == 25.0)) or np.any(inlb.n_pairs_allocated[cap0] != 0):
        problems.append("INLB: a zero capacity (f = 1, or N = 25) does not allocate nothing")
    # the derived underlying quantities
    m_start = inlb.n_monomers_underlying_reference - 2.0 * inlb.n_pairs_allocated
    d_start = inlb.n_dimers_underlying_reference + inlb.n_pairs_allocated
    if not np.allclose(inlb.n_monomers_underlying_start, m_start) or not np.allclose(inlb.n_dimers_underlying_start, d_start):
        problems.append("INLB: M_start or D_start differ from M_ref - 2c, D_ref + c")
    if not np.allclose(inlb.n_monomers_underlying_start + 2.0 * inlb.n_dimers_underlying_start, nn, rtol=1e-9):
        problems.append("INLB: M_start + 2 D_start != N_total")
    with np.errstate(invalid="ignore", divide="ignore"):
        f_start = d_start / (m_start + d_start)
        share = inlb.n_pairs_allocated / d_start
    pos = d_start > 0
    if not np.allclose(inlb.fraction_basal_dimers_start[pos], f_start[pos]) or not np.allclose(inlb.active_share[pos], share[pos]) \
            or not np.all(np.isnan(inlb.active_share[~pos])) or not np.any(~pos):
        problems.append("INLB: f_start or the active share differ from their definitions (NaN at D_start = 0)")
    if np.any(inlb.fraction_basal_dimers_start[pos & (rr > 0) & (inlb.n_pairs_allocated > 0)] <= ff[pos & (rr > 0) & (inlb.n_pairs_allocated > 0)]):
        problems.append("INLB: f_start does not exceed f where pairs were allocated")
    # c monotone in R_active at every (N_total, f)
    if not np.all(np.diff(inlb.n_pairs_allocated, axis=2) >= 0):
        problems.append("INLB: c is not monotone in R_active")
    # the worked literals
    w = par.realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=0.2,
                                        relative_count_active_dimers_initial=0.5, condition="INLB")
    w1 = par.realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=0.2,
                                         relative_count_active_dimers_initial=1.0, condition="INLB")
    center = par.realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=0.5,
                                             relative_count_active_dimers_initial=0.5, condition="INLB")
    literals_ok = ((w.n_monomers_reference, w.n_dimers_one_probe, w.n_dimers_two_probe_reference, w.capacity,
                    w.n_pairs_allocated) == (317, 151, 4, 158, 79)
                   and (w1.n_pairs_allocated, w1.n_monomers, w1.n_dimers_two_probe) == (158, 1, 162)
                   and (center.n_monomers, center.n_dimers_one_probe, center.n_dimers_two_probe) == (79, 302, 48))
    if not literals_ok:
        problems.append(f"INLB: the worked literals do not hold ({w}; {w1}; {center})")
    # MET-FAB: the starting population IS the reference, the capacity and c are zero
    fab = realize("FAB", nn[..., 0], ff[..., 0], 0.0)
    n_cases += int(nn[..., 0].size)
    if np.any(fab.capacity != 0) or np.any(fab.n_pairs_allocated != 0) or not np.array_equal(fab.n_monomers, fab.n_monomers_reference) \
            or not np.array_equal(fab.n_dimers_two_probe, fab.n_dimers_two_probe_reference) \
            or not np.array_equal(fab.n_dimers, fab.n_dimers_reference) or not np.all(np.isnan(fab.active_share) | (fab.active_share == 0.0)) \
            or not np.allclose(fab.fraction_basal_dimers_start[ff[..., 0] < 1.0], ff[..., 0][ff[..., 0] < 1.0]):
        problems.append("FAB: the starting population differs from the reference, or the allocation is not zero")
    # the vectorized helper equals the one-draw initializer (by key, through the condition's theta) on prior draws
    rng = np.random.default_rng(SEED + 12)
    agreement = {}
    for condition in RDS.condition_tokens:
        low, high = prior_box(condition)
        table = table_of(condition)
        phys = par.to_physical(rng.uniform(low, high, size=(N_D12_DRAWS, len(low))), table)
        vec = par.population_from_theta(phys, condition)
        mismatches = 0
        for i in range(N_D12_DRAWS):
            one = rds.initial_composition_of(phys[i], condition)
            item = vec.item(i)
            for name in par.POPULATION_FIELD_DOCS:
                a, b = getattr(one, name), getattr(item, name)
                same = (a == b) or (isinstance(a, float) and np.isnan(a) and np.isnan(b))
                if not same:
                    mismatches += 1
                    break
        agreement[condition] = dict(n_draws=N_D12_DRAWS, mismatches=mismatches,
                                    draws_with_allocation=int((vec.n_pairs_allocated > 0).sum()),
                                    median_capacity=float(np.median(vec.capacity)), median_c=float(np.median(vec.n_pairs_allocated)))
        if mismatches:
            problems.append(f"{condition}: {mismatches} of {N_D12_DRAWS} prior draws differ between the vectorized helper "
                            f"and the one-draw initializer")
        if condition == "INLB" and agreement[condition]["draws_with_allocation"] == 0:
            problems.append("INLB: no prior draw allocates a pair")
        if condition == "FAB" and agreement[condition]["draws_with_allocation"] != 0:
            problems.append("FAB: a prior draw allocates a pair")
    return dict(n_cases=n_cases, grid=dict(n_total=n_grid.tolist(), fraction=f_grid.tolist(), allocation=r_grid.tolist()),
                half_integer_products=n_half, worked_literals=dict(
                    n1e4_f02=dict(A_ref=w.n_monomers_reference, B1=w.n_dimers_one_probe, B2_ref=w.n_dimers_two_probe_reference,
                                  capacity=w.capacity, c_at_half=w.n_pairs_allocated, c_at_one=w1.n_pairs_allocated,
                                  monomers_left_at_one=w1.n_monomers),
                    prior_center=dict(A=center.n_monomers, B1=center.n_dimers_one_probe, B2=center.n_dimers_two_probe,
                                      c=center.n_pairs_allocated, f_start=center.fraction_basal_dimers_start,
                                      active_share=center.active_share)),
                worked_literals_ok=literals_ok, helper_vs_initializer=agreement, problems=problems, ok=not problems)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true",
                        help="also execute the run tier (tiny simulations and renders); needs approval")
    args = parser.parse_args()
    det_out = dict(d1=d1_channels(), d2=d2_diffusion(), d3=d3_transforms(),
                   d4=d4_composition(), d5=d5_stationary(), d6=d6_occupancy(),
                   d7=d7_fission_placement(), d8=d8_conditions(), d9=d9_ranges(),
                   d10=d10_theta_schema(), d11=d11_conversion_displacement(), d12=d12_allocation())
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
            r8 = r8_placement_geometry(workdir)
            run_out = dict(r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6, r7=r7, r8=r8)
            for k in ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"):
                print(f"  [{passed(run_out[k]['ok'])}] {k}")
    write_report(det_out, run_out)
    all_ok = all(v["ok"] for v in det_out.values()) and (run_out is None or all(v["ok"] for v in run_out.values()))
    print("report:", REPORT)
    print("OVERALL:", passed(all_ok))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
