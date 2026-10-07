"""The retained-population model: species inventory, channels, initialization, lineage, and labeling.

The generator simulates the probe-associated complexes (PROJECT_CONTEXT.md sec. 2, *Modeling assumptions
of the MET model*). These tests hold the seven checks of the implementation increment without ReaDDy:

1. Inventory and channels: MET-FAB six particle types and eleven channels (three fissions, eight mode
   switches); MET-INLB nine types and twenty-four (six fusions A + A -> B2, three fissions B2 -> A + A,
   three conversions B1 -> A at kappa_OFF, twelve mode switches). Nothing associates under FAB.
2. Nonfluorescent bound receptors still associate: every simulated INLB monomer is InlB-bound whatever
   its dye draw, so the fusion educts are the monomer types and no labeling quantity enters the network.
3. B1 releases one eligible daughter (a conversion that preserves the one retained subunit); B2 releases
   two (a fission that hands each daughter one subunit): the lineage replay rules on synthetic records.
4. The retained count is conserved by every channel: n_A + 2 n_B under FAB, n_A + n_B1 + 2 n_B2 under
   INLB, through fusion, fission, conversion and mode switches.
5. The integer initialization realizes the expectations from (N_total, f_B, p): dimers and classes first,
   monomers last; at the ceiling about 4,762 (INLB) and 1,600-2,100 (FAB) retained subunits; at the
   floor every draw places at least one monomer and one dimer; the realized ratio is recorded beside the
   requested true one and differs by the selection factor (2 - p), not by rounding.
6. The two prior rows carry the decided boxes: count_total log10 [3.0, 5.0]; ratio_dimer_monomer_initial
   log10 [log10(0.05/0.95), log10(0.25/0.75)], the 5-25 % band of the true basal complex fraction; the
   conditions' occupancies are 0.0476 (equilibrium) and 0.0148 (derived), with the two-probe shares.
7. The labeling follows the probe rule: every retained monomer bound; FAB basal dimers one- or two-Fab
   at the declared share; INLB classes by species; the record carries the probe classes.

Runnable with ``python -m pytest`` or directly:
``MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_retained_population.py``.
"""
from types import SimpleNamespace

import numpy as np

from srm_and_sbi_monomer_dimer_alp import labeling as lab
from srm_and_sbi_monomer_dimer_alp import parameterization as par
from srm_and_sbi_monomer_dimer_alp import simulation_rds_support as rds
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERIZATION, PARAMETERS

RDS = PARAMETERS.simulation.rds
STO, MOB = RDS.stoichiometry, RDS.mobility


def _theta(**overrides):
    theta = np.array([par.prior_center(e) for e in PARAMETERIZATION], dtype=float)
    for key, value in overrides.items():
        theta[par.parameter_find(key)] = value
    return theta


# ---- 1. inventory and channels ------------------------------------------------------------------

def test_inventory_and_channels_per_condition():
    assert RDS.particle_type_names_for("FAB") == ("A_f", "A_s", "A_i", "B_f", "B_s", "B_i")
    assert RDS.particle_type_names_for("INLB") == ("A_f", "A_s", "A_i", "B1_f", "B1_s", "B1_i", "B2_f", "B2_s", "B2_i")
    assert len(RDS.particle_type_names) == 12 and RDS.monomer_type_names == ("A_f", "A_s", "A_i")
    assert all(RDS.species_of_type[n] == "B" for n in RDS.particle_type_names if not n.startswith("A"))
    assert RDS.molecule_of_type["B1_s"] == "B1" and RDS.molecule_of_type["B2_i"] == "B2"
    assert dict(zip(RDS.particle_type_names, RDS.subunit_counts_per_type)) == {
        "A_f": 1, "A_s": 1, "A_i": 1, "B_f": 2, "B_s": 2, "B_i": 2,
        "B1_f": 1, "B1_s": 1, "B1_i": 1, "B2_f": 2, "B2_s": 2, "B2_i": 2}
    theta = _theta()
    fab = rds.reaction_channels(theta, "FAB")
    inlb = rds.reaction_channels(theta, "INLB")
    kinds = lambda chans: {k: sum(c.kind == k for c in chans) for k in ("fusion", "fission", "conversion")}
    assert len(fab) == 11 and kinds(fab) == {"fusion": 0, "fission": 3, "conversion": 8}
    assert len(inlb) == 24 and kinds(inlb) == {"fusion": 6, "fission": 3, "conversion": 15}
    assert len({c.name for c in inlb}) == 24 and len({c.name for c in fab}) == 11
    kappa = theta[par.parameter_find("rate_dissociation")]
    dissociations = [c for c in inlb if c.rate_key == "rate_dissociation"]
    assert len(dissociations) == 6 and all(np.isclose(c.rate, kappa) for c in dissociations)
    assert sorted(c.name for c in dissociations) == sorted(
        [f"B1_{m} => A_{m}" for m in MOB.modes] + [f"B2_{m} => A_{m} + A_{m}" for m in MOB.modes])
    for c in fab:
        assert not any(n.startswith("B1") or n.startswith("B2") for n in c.educts + c.products)


# ---- 2. nonfluorescent bound receptors associate ------------------------------------------------

def test_association_is_by_bound_monomer_type_not_by_dye():
    theta = _theta()
    fusions = [c for c in rds.reaction_channels(theta, "INLB") if c.kind == "fusion"]
    for c in fusions:
        assert all(RDS.species_of_type[e] == "A" for e in c.educts), c
        assert RDS.molecule_of_type[c.products[0]] == "B2", c
        m1, m2 = (RDS.mode_of_type[e] for e in c.educts)
        assert RDS.mode_of_type[c.products[0]] == MOB.slower(m1, m2)
    # the plan labels every retained INLB subunit as bound: a dark InlB keeps its receptor eligible
    plan = lab.resolve_labeling("INLB")
    assert plan.occupancy_pair == (1.0, 1.0)
    n_a, n_b1, n_b2 = 50, 30, 10
    host_index = np.concatenate([np.arange(n_a), n_a + np.arange(n_b1), n_a + n_b1 + np.repeat(np.arange(n_b2), 2)])
    ranks = {n: i for i, n in enumerate(RDS.particle_type_names)}
    host_rank = np.concatenate([np.full(n_a, ranks["A_f"]), np.full(n_b1, ranks["B1_s"]), np.full(2 * n_b2, ranks["B2_i"])])
    species_of_rank = {i: RDS.species_of_type[n] for n, i in ranks.items()}
    bound = lab.assign_probes(plan, host_index, host_rank, species_of_rank, np.random.default_rng(0))
    assert bound.all()
    dyes = lab.draw_dye_counts_bound(plan.law, bound, np.random.default_rng(0))
    assert 0 < (dyes == 0).sum() < dyes.size            # dark probes exist, and they are still bound


# ---- 3./4. lineage rules and conservation ---------------------------------------------------------

def _record(kind, educts, products):
    return SimpleNamespace(type=kind, educts=np.array(educts), products=np.array(products), reaction_label=kind)


def test_b1_releases_one_eligible_daughter_and_b2_two_and_the_retained_count_is_conserved():
    # INLB souls: 1 = A, 2 = B1 (one retained subunit), 3 = B2 (two)
    souls = {1: (0,), 2: (1,), 3: (2, 3)}
    counts = {}
    rds._apply_reaction_record(_record("conversion", [2], [4]), souls, counts)      # B1 -> A
    assert souls[4] == (1,) and 2 not in souls                                       # one eligible daughter, same identity
    rds._apply_reaction_record(_record("fission", [3], [5, 6]), souls, counts)      # B2 -> A + A
    assert souls[5] == (2,) and souls[6] == (3,)                                     # two eligible daughters
    rds._apply_reaction_record(_record("fusion", [1, 4], [7]), souls, counts)        # A + A -> B2
    assert souls[7] == (0, 1)
    rds._apply_reaction_record(_record("conversion", [7], [8]), souls, counts)      # a mode switch keeps both
    assert souls[8] == (0, 1)
    covered = sorted(s for subs in souls.values() for s in subs)
    assert covered == [0, 1, 2, 3], "every retained subunit is hosted exactly once"
    assert counts == {"conversion": 2, "fission": 1, "fusion": 1}
    # a one-subunit particle can never fission (a B1 is written as a conversion, never a fission)
    try:
        rds._apply_reaction_record(_record("fission", [5], [9, 10]), souls, counts)
    except ValueError:
        pass
    else:
        raise AssertionError("a fission of a one-subunit particle must be refused")
    # the retained count per channel: subunits in == subunits out
    sub = dict(zip(RDS.particle_type_names, RDS.subunit_counts_per_type))
    for condition in RDS.condition_tokens:
        for c in rds.reaction_channels(_theta(), condition):
            assert sum(sub[e] for e in c.educts) == sum(sub[p] for p in c.products), c.name


# ---- 5. initialization ----------------------------------------------------------------------------

def test_initialization_realizes_the_retained_expectations_dimers_first():
    for condition in ("FAB", "INLB"):
        p = par.occupancy_of(condition)
        s2 = par.two_probe_share_of(condition)
        for n_total, f_b in ((1e3, 0.05), (1e4, 0.117), (1e5, 0.25), (3.2e4, 0.075)):
            r = f_b / (1 - f_b)
            c = par.realize_initial_composition(n_total, r, condition)
            n_a_true, n_b_true = n_total / (1 + 2 * r), r * n_total / (1 + 2 * r)
            assert c.n_dimers == round(p * (2 - p) * n_b_true)
            assert c.n_dimers_two_probe == min(round(p * p * n_b_true), c.n_dimers)
            assert c.n_dimers_one_probe == c.n_dimers - c.n_dimers_two_probe
            assert c.n_monomers == round(p * n_a_true)
            if c.ligand_classes:
                assert c.n_subunits == c.n_monomers + c.n_dimers_one_probe + 2 * c.n_dimers_two_probe
            else:
                assert c.n_subunits == c.n_monomers + 2 * c.n_dimers
            assert abs(c.n_subunits - c.expected_subunits) <= 1.5 + 0.5 * (c.ligand_classes and 2 or 1)
            assert np.isclose(c.two_probe_share, s2) and np.isclose(c.ratio_retained_expected, (2 - p) * r)
            assert np.isclose(c.complex_fraction_true, f_b)
            assert c.n_monomers >= 1 and c.n_dimers >= 1
        # the ceiling: about 4,762 (INLB) and 1,600-2,100 (FAB) retained subunits
        lo = par.realize_initial_composition(1e5, 0.05 / 0.95, condition).n_subunits
        hi = par.realize_initial_composition(1e5, 0.25 / 0.75, condition).n_subunits
        if condition == "INLB":
            assert 4755 <= lo <= 4765 and 4755 <= hi <= 4765
        else:
            assert 1600 <= lo <= 1650 and 2040 <= hi <= 2100
    # the retained composition differs from the true one by selection: 25 % true -> about 39-40 % retained
    c = par.realize_initial_composition(1e5, 0.25 / 0.75, "INLB")
    assert 0.385 <= c.complex_fraction_retained <= 0.40 and np.isclose(c.complex_fraction_true, 0.25)
    # the requested true total is a float; a non-positive total or a negative ratio is refused
    for bad in ((0.0, 0.1), (100.0, -0.1), (float("nan"), 0.1)):
        try:
            par.realize_initial_composition(bad[0], bad[1], "FAB")
        except ValueError:
            continue
        raise AssertionError(f"{bad} must be refused")


# ---- 6. the prior rows and the occupancies ----------------------------------------------------------

def test_prior_rows_and_occupancies_carry_the_decided_values():
    lo, hi = par.theta_lower_bound(), par.theta_upper_bound()
    i_n, i_r = par.parameter_find("count_total"), par.parameter_find("ratio_dimer_monomer_initial")
    assert (lo[i_n], hi[i_n]) == (3.0, 5.0)
    assert np.isclose(lo[i_r], np.log10(0.05 / 0.95)) and np.isclose(hi[i_r], np.log10(0.25 / 0.75))
    assert np.isclose(lo[i_r], -1.27875360, atol=1e-8) and np.isclose(hi[i_r], -0.47712125, atol=1e-8)
    assert par.COMPOSITION_BAND == (0.05, 0.25) and par.COUNT_TOTAL_BOX == (3.0, 5.0)
    assert len(par.PARAMETER_KEYS) == 11                       # the composition stays an inference coordinate
    assert np.isclose(par.occupancy_of("INLB"), 0.25 / 5.25) and par.occupancy_source_of("INLB") == "equilibrium"
    assert np.isclose(par.occupancy_of("FAB"), 0.5 * (0.25 / 5.25 * 0.5) / (1 - np.exp(-1.64)))
    assert par.occupancy_source_of("FAB") == "derived"
    for c in ("FAB", "INLB"):
        p = par.occupancy_of(c)
        assert np.isclose(par.two_probe_share_of(c), p / (2 - p))
        assert np.isclose(par.retained_ratio_of(c, 0.1), (2 - p) * 0.1)
    assert RDS.condition_setting("INLB").ligand_classes and not RDS.condition_setting("FAB").ligand_classes
    # a condition with association but without ligand classes is refused; so is a mixed occupancy declaration
    for kwargs in (dict(association_ratio=1.0), dict(association_ratio=0.0, dissociation_constant_nm=5.0, occupancy=0.3)):
        try:
            par.ConditionSetting("X", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} must be refused")


# ---- 7. the labeling record carries the probe classes -----------------------------------------------

def test_labeling_record_carries_the_probe_classes():
    cols = lab.LABELING_SET_COLUMNS
    assert cols[-3:] == ("dimers_one_probe_0", "dimers_two_probe_0", "two_probe_share")
    plan = lab.resolve_labeling("FAB")
    n_a, n_b = 2000, 4000
    host_index = np.concatenate([np.arange(n_a), n_a + np.repeat(np.arange(n_b), 2)])
    ranks = {n: i for i, n in enumerate(RDS.particle_type_names)}
    host_rank = np.concatenate([np.full(n_a, ranks["A_s"]), np.full(2 * n_b, ranks["B_f"])])
    species_of_rank = {i: RDS.species_of_type[n] for n, i in ranks.items()}
    dyes, row = lab.label_subunits(plan, host_index, host_rank, species_of_rank, [ranks["A_f"], ranks["A_s"], ranks["A_i"]],
                                   np.random.default_rng(7))
    col = dict(zip(cols, row))
    assert col["dimers_one_probe_0"] + col["dimers_two_probe_0"] == n_b
    assert abs(col["dimers_two_probe_0"] / n_b - plan.two_probe_share) < 0.005
    assert col["two_probe_share"] == plan.two_probe_share and col["occupancy_monomer"] == 1.0
    # a one-Fab dimer has exactly one bound subunit, so at most one labeled subunit
    bound = lab.assign_probes(plan, host_index, host_rank, species_of_rank, np.random.default_rng(7))
    per_host = np.bincount(host_index[n_a:], weights=bound[n_a:].astype(float))[n_a:]
    assert set(np.unique(per_host).tolist()) <= {1.0, 2.0}


def test_the_horizon_audit_refuses_every_phase_under_the_retained_population():
    """Its composition metrics compare the inferred TRUE composition with the trajectory's RETAINED
    census; until they are restated, every phase (the dry run included) is refused before any spec
    is built or any path is touched."""
    from srm_and_sbi_monomer_dimer_alp import horizon_audit_runner as ha
    from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow
    parser = ha.build_parser("test")
    for phase in ("prepare", "generate", "analyze", "selftest"):
        for extra in ([], ["--dry-run"]):
            args = parser.parse_args(["--total-time-seconds", "2", "--condition", "INLB", "--phase", phase] + extra)
            try:
                ha.run_horizon_audit(biology_workflow(), args)
            except SystemExit as exc:
                assert "disabled" in str(exc) and "RETAINED" in str(exc), str(exc)
            else:
                raise AssertionError(f"phase {phase} {extra} must be refused")
    assert "retained" in ha.HORIZON_AUDIT_DISABLED.lower()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
