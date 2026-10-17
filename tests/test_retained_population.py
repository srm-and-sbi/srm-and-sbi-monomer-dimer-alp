"""The simulated-population model: species inventory, channels, initialization, lineage, and labeling.

The generator simulates the probe-associated complexes (PROJECT_CONTEXT.md sec. 2, *Modeling assumptions
of the MET model*). These tests hold the seven checks of the implementation without ReaDDy:

1. Inventory and channels: MET-FAB six particle types and eleven channels (three fissions, eight mode
   switches); MET-INLB nine types and twenty-four (six fusions A + A -> B2, three fissions B2 -> A + A,
   three conversions B1 -> A at kappa_OFF, twelve mode switches). Nothing associates under FAB.
2. Nonfluorescent bound receptors still associate: every simulated INLB monomer is InlB-bound whatever
   its dye draw, so the fusion educts are the monomer types and no labeling quantity enters the network.
3. B1 releases one eligible daughter (a conversion that preserves the one represented subunit); B2
   releases two (a fission that hands each daughter one subunit): the lineage replay rules on synthetic
   records.
4. The represented count is conserved by every channel: n_A + 2 n_B under FAB, n_A + n_B1 + 2 n_B2 under
   INLB, through fusion, fission, conversion and mode switches.
5. The integer initialization realizes the REFERENCE population from (N_total, f, p) -- dimers and
   classes first, monomers last -- and then the ALLOCATION over a grid of N x f x R_active: c =
   round(R_active x capacity) probe-bound monomer pairs become two-InlB dimers (B1 unchanged, N_R
   conserved); under MET-FAB the allocation is declared 0 and the starting population is the
   reference. Invalid inputs (a non-positive or non-finite total, f or R_active outside [0, 1], a
   Fab allocation of 0.3) are refused, never clipped.
6. The prior rows carry the decided boxes per condition: count_total log10 [3.0, 5.0], the basal
   reference fraction f linear on [0, 1] (both conditions), the allocation R_active linear on [0, 1]
   under MET-INLB only (eleven rows under MET-FAB, twelve under MET-INLB); the conditions' occupancies
   are 0.0476 (equilibrium) and 0.0148 (derived), with the two-probe shares.
7. The labeling follows the probe rule: every represented monomer bound; FAB basal dimers one- or
   two-Fab at the declared share; INLB classes by species; the record carries the probe classes.

Plus: the horizon audit refuses every phase under both conditions.

Runnable with ``python -m pytest`` or directly:
``MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_retained_population.py``.
"""
from types import SimpleNamespace

import numpy as np

from srm_and_sbi_monomer_dimer_alp import labeling as lab
from srm_and_sbi_monomer_dimer_alp import parameterization as par
from srm_and_sbi_monomer_dimer_alp import simulation_rds_support as rds
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, parameterization_of

RDS = PARAMETERS.simulation.rds
STO, MOB = RDS.stoichiometry, RDS.mobility
F_KEY, R_KEY = STO.basal_fraction_key, STO.active_allocation_key


def _theta(condition, **overrides):
    """The condition's theta at the prior center (physical), with overrides by key; eleven entries
    under MET-FAB, twelve under MET-INLB."""
    table = parameterization_of(condition)
    theta = np.array([par.prior_center(e) for e in table], dtype=float)
    assert theta.shape == ((11,) if condition == "FAB" else (12,)), (condition, theta.shape)
    for key, value in overrides.items():
        theta[par.parameter_find(key, condition)] = value
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
    fab = rds.reaction_channels(_theta("FAB"), "FAB")
    inlb = rds.reaction_channels(_theta("INLB"), "INLB")
    kinds = lambda chans: {k: sum(c.kind == k for c in chans) for k in ("fusion", "fission", "conversion")}
    assert len(fab) == 11 and kinds(fab) == {"fusion": 0, "fission": 3, "conversion": 8}
    assert len(inlb) == 24 and kinds(inlb) == {"fusion": 6, "fission": 3, "conversion": 15}
    assert len({c.name for c in inlb}) == 24 and len({c.name for c in fab}) == 11
    kappa = _theta("INLB")[par.parameter_find("rate_dissociation", "INLB")]
    dissociations = [c for c in inlb if c.rate_key == "rate_dissociation"]
    assert len(dissociations) == 6 and all(np.isclose(c.rate, kappa) for c in dissociations)
    assert sorted(c.name for c in dissociations) == sorted(
        [f"B1_{m} => A_{m}" for m in MOB.modes] + [f"B2_{m} => A_{m} + A_{m}" for m in MOB.modes])
    for c in fab:
        assert not any(n.startswith("B1") or n.startswith("B2") for n in c.educts + c.products)


# ---- 2. nonfluorescent bound receptors associate ------------------------------------------------

def test_association_is_by_bound_monomer_type_not_by_dye():
    fusions = [c for c in rds.reaction_channels(_theta("INLB"), "INLB") if c.kind == "fusion"]
    for c in fusions:
        assert all(RDS.species_of_type[e] == "A" for e in c.educts), c
        assert RDS.molecule_of_type[c.products[0]] == "B2", c
        m1, m2 = (RDS.mode_of_type[e] for e in c.educts)
        assert RDS.mode_of_type[c.products[0]] == MOB.slower(m1, m2)
    # the plan labels every represented INLB subunit as bound: a dark InlB keeps its receptor eligible
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


def test_b1_releases_one_eligible_daughter_and_b2_two_and_the_represented_count_is_conserved():
    # INLB souls: 1 = A, 2 = B1 (one represented subunit), 3 = B2 (two)
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
    assert covered == [0, 1, 2, 3], "every represented subunit is hosted exactly once"
    assert counts == {"conversion": 2, "fission": 1, "fusion": 1}
    # a one-subunit particle can never fission (a B1 is written as a conversion, never a fission)
    try:
        rds._apply_reaction_record(_record("fission", [5], [9, 10]), souls, counts)
    except ValueError:
        pass
    else:
        raise AssertionError("a fission of a one-subunit particle must be refused")
    # the represented count per channel: subunits in == subunits out
    sub = dict(zip(RDS.particle_type_names, RDS.subunit_counts_per_type))
    for condition in RDS.condition_tokens:
        for c in rds.reaction_channels(_theta(condition), condition):
            assert sum(sub[e] for e in c.educts) == sum(sub[p] for p in c.products), c.name


# ---- 5. initialization: the reference population, then the allocation -----------------------------

def test_initialization_realizes_the_reference_population_then_the_allocation():
    grid_n = (1e3, 1e4, 1e5, 3.2e4)
    grid_f = (0.0, 0.05, 0.117, 0.25, 0.5, 1.0)
    for condition in ("FAB", "INLB"):
        p = par.occupancy_of(condition)
        s2 = par.two_probe_share_of(condition)
        ligand = RDS.condition_setting(condition).ligand_classes
        grid_r = (0.0, 0.5, 1.0) if ligand else (0.0,)
        for n_total in grid_n:
            for f in grid_f:
                m_ref, d_ref = n_total * (1 - f) / (1 + f), n_total * f / (1 + f)
                for r in grid_r:
                    c = par.realize_initial_composition(count_total=n_total, fraction_basal_dimers_reference=f,
                                                        relative_count_active_dimers_initial=r, condition=condition)
                    # the reference population: dimers and classes first, monomers last
                    assert c.n_dimers_reference == round(p * (2 - p) * d_ref)
                    assert c.n_dimers_two_probe_reference == min(round(p * p * d_ref), c.n_dimers_reference)
                    assert c.n_dimers_one_probe == c.n_dimers_reference - c.n_dimers_two_probe_reference
                    assert c.n_monomers_reference == round(p * m_ref)
                    assert np.isclose(c.n_monomers_underlying_reference, m_ref) and np.isclose(c.n_dimers_underlying_reference, d_ref)
                    # the allocation: c = round(R cap) pairs of the cap = A_ref // 2 available ones
                    cap = c.n_monomers_reference // 2 if ligand else 0
                    pairs = round(r * cap)
                    assert (c.capacity, c.n_pairs_allocated) == (cap, pairs)
                    assert c.n_monomers == c.n_monomers_reference - 2 * pairs
                    assert c.n_dimers_two_probe == c.n_dimers_two_probe_reference + pairs
                    assert c.n_dimers == c.n_dimers_one_probe + c.n_dimers_two_probe
                    if ligand:
                        assert c.n_subunits == c.n_monomers_reference + c.n_dimers_one_probe + 2 * c.n_dimers_two_probe_reference
                        assert c.n_subunits == c.n_monomers + c.n_dimers_one_probe + 2 * c.n_dimers_two_probe
                    else:
                        assert c.n_subunits == c.n_monomers + 2 * c.n_dimers and pairs == 0
                    assert abs(c.n_subunits - c.expected_subunits) <= 1.5 + 0.5 * (ligand and 2 or 1)
                    assert np.isclose(c.two_probe_share, s2) and c.fraction_basal_dimers_reference == f
                    assert np.isclose(c.n_monomers_underlying_start + 2 * c.n_dimers_underlying_start, n_total)
                    if c.n_monomers_underlying_start + c.n_dimers_underlying_start > 0:
                        assert c.fraction_basal_dimers_start >= f - 1e-12
                    assert c.n_total_true == n_total and c.n_total == c.n_subunits
        # the ceilings are reported, not asserted as windows: N_R at N = 1e5 is about 4,762 under
        # MET-INLB for every f and R, and rises with f under MET-FAB
        top = {f: par.realize_initial_composition(count_total=1e5, fraction_basal_dimers_reference=f,
                                                  relative_count_active_dimers_initial=0.0, condition=condition).n_subunits
               for f in (0.0, 0.25, 1.0)}
        if ligand:
            assert all(abs(v - 4762) <= 2 for v in top.values()), top
        else:
            assert top[0.0] < top[0.25] < top[1.0], top
    # under MET-FAB the starting population IS the reference, whatever f
    for f in (0.0, 0.075, 1.0):
        c = par.realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=f,
                                            relative_count_active_dimers_initial=0.0, condition="FAB")
        assert (c.n_monomers, c.n_dimers_two_probe, c.n_pairs_allocated) == (c.n_monomers_reference, c.n_dimers_two_probe_reference, 0)
        assert np.isclose(c.fraction_basal_dimers_start, f)
    # refusals: a non-positive or non-finite total, f or R outside [0, 1], a Fab allocation of 0.3
    for condition, bad in (("FAB", dict(count_total=0.0, fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=0.0)),
                           ("FAB", dict(count_total=float("nan"), fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=0.0)),
                           ("FAB", dict(count_total=100.0, fraction_basal_dimers_reference=-0.1, relative_count_active_dimers_initial=0.0)),
                           ("FAB", dict(count_total=100.0, fraction_basal_dimers_reference=1.2, relative_count_active_dimers_initial=0.0)),
                           ("FAB", dict(count_total=100.0, fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=0.3)),
                           ("INLB", dict(count_total=100.0, fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=1.02)),
                           ("INLB", dict(count_total=100.0, fraction_basal_dimers_reference=0.1, relative_count_active_dimers_initial=-0.01)),
                           ("INLB", dict(count_total=100.0, fraction_basal_dimers_reference=float("inf"), relative_count_active_dimers_initial=0.5))):
        try:
            par.realize_initial_composition(condition=condition, **bad)
        except ValueError:
            continue
        raise AssertionError(f"{condition} {bad} must be refused")


# ---- 6. the prior rows and the occupancies ----------------------------------------------------------

def test_prior_rows_and_occupancies_carry_the_decided_values():
    for condition, width in (("FAB", 11), ("INLB", 12)):
        lo, hi = par.theta_lower_bound(condition), par.theta_upper_bound(condition)
        keys = par.parameter_keys_of(condition)
        assert len(keys) == len(lo) == len(hi) == width
        i_n, i_f = par.parameter_find("count_total", condition), par.parameter_find(F_KEY, condition)
        assert (lo[i_n], hi[i_n]) == (3.0, 5.0) and (lo[i_f], hi[i_f]) == (0.0, 1.0)
        table = parameterization_of(condition)
        assert table[i_n]["LOG_FLAG"] and not table[i_f]["LOG_FLAG"]
        if condition == "INLB":
            i_r = par.parameter_find(R_KEY, condition)
            assert (lo[i_r], hi[i_r]) == (0.0, 1.0) and not table[i_r]["LOG_FLAG"] and i_r == 11
            assert par.declared_row_values("INLB") == {}
        else:
            assert R_KEY not in keys and par.declared_row_values("FAB") == {R_KEY: 0.0}
            try:
                par.parameter_find(R_KEY, "FAB")
            except KeyError as exc:
                assert "declared" in str(exc)
            else:
                raise AssertionError("the declared allocation is not a Fab theta coordinate")
    assert par.FRACTION_BOX == (0.0, 1.0) and par.COUNT_TOTAL_BOX == (3.0, 5.0)
    assert par.parameter_keys_of("INLB")[:11] == par.parameter_keys_of("FAB")
    assert np.isclose(par.occupancy_of("INLB"), 0.25 / 5.25) and par.occupancy_source_of("INLB") == "equilibrium"
    assert np.isclose(par.occupancy_of("FAB"), 0.5 * (0.25 / 5.25 * 0.5) / (1 - np.exp(-1.64)))
    assert par.occupancy_source_of("FAB") == "derived"
    for c in ("FAB", "INLB"):
        p = par.occupancy_of(c)
        assert np.isclose(par.two_probe_share_of(c), p / (2 - p))
    assert RDS.condition_setting("INLB").ligand_classes and not RDS.condition_setting("FAB").ligand_classes
    assert RDS.condition_setting("FAB").relative_count_active_dimers_initial == 0.0
    assert RDS.condition_setting("INLB").relative_count_active_dimers_initial is None
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


def test_the_horizon_audit_refuses_every_phase_under_both_conditions():
    """Its composition metrics compare the inferred reference composition with the trajectory's
    represented census, and under MET-INLB the window-start allocation makes the two differ further;
    until the estimands are restated, every phase (the dry run included) is refused before any spec
    is built or any path is touched, under both conditions."""
    from srm_and_sbi_monomer_dimer_alp import horizon_audit_runner as ha
    from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow
    parser = ha.build_parser("test")
    for condition in ("FAB", "INLB"):
        for phase in ("prepare", "generate", "analyze", "selftest"):
            for extra in ([], ["--dry-run"]):
                args = parser.parse_args(["--total-time-seconds", "2", "--condition", condition, "--phase", phase] + extra)
                try:
                    ha.run_horizon_audit(biology_workflow(), args)
                except SystemExit as exc:
                    assert "disabled" in str(exc) and "RETAINED" in str(exc), str(exc)
                else:
                    raise AssertionError(f"{condition} phase {phase} {extra} must be refused")
    message = ha.HORIZON_AUDIT_DISABLED.lower()
    assert "retained" in message and "reference" in message and "allocation" in message
    # no reference to a removed name survives in the audit's modules (names, attributes, imports)
    import ast
    from pathlib import Path
    removed = {"PARAMETERIZATION", "PARAMETER_KEYS", "PARAMETER_FIND", "COMPOSITION_BAND", "COMPOSITION_RATIO_BOX",
               "ratio_to_receptor_fraction", "ratio_to_complex_fraction", "receptor_fraction_to_ratio",
               "retained_ratio_of", "composition_ratio_key", "_COUNT_KEYS"}
    package = Path(ha.__file__).resolve().parent
    for module in ("horizon_audit.py", "horizon_audit_runner.py"):
        source = (package / module).read_text()
        tree = ast.parse(source)
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                used.update(alias.name for alias in node.names)
        assert not used & removed, (module, sorted(used & removed))
        assert "ratio_dimer_monomer_initial" not in source, module
    # the spec, were it built, binds the condition: keys, box, prior and table
    for condition, width in (("FAB", 11), ("INLB", 12)):
        args = parser.parse_args(["--total-time-seconds", "2", "--condition", condition, "--phase", "prepare"])
        spec = ha._horizon_audit_spec(biology_workflow(), args)
        assert spec["condition"] == condition and len(spec["keys"]) == width == len(spec["lower"]) == len(spec["table"])
        assert spec["binding"].count_index == ((0, 1) if condition == "FAB" else (0, 1, 11))
        assert spec["build_prior"]().event_shape == (width,)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
