"""Every renderer of simulated trajectories labels them the way the training data were labeled.

The DLI stage of both workflows, the posterior-predictive video and the horizon audit resolve a
run's static labeling with `labeling.resolve_labeling` and draw it with `labeling.label_trajectory`
(the condition's law and the retained-population probe rule: every monomer bound, dimer probe
classes by species under MET-INLB and drawn with the two-probe share under MET-FAB; or the
`--occupancy` override's independent coins). These tests hold the contract:

1. The shared path reproduces, bit for bit, the reference labeling block written out below from
   the primitives (`assign_probes` + `draw_dye_counts_bound`, or the coins of an override), for
   both conditions, several seeds, tasks and simulations, and the supported overrides (an
   alternative law, a scalar and a per-species occupancy).
2. The three command lines accept the same labeling flags and resolve the same plan from them:
   by default the model's probe rule with the condition's equilibrium (MET-INLB) or derived
   (MET-FAB) occupancy setting the two-probe share, an override recorded as such.
3. For the same condition, lineage and seed, the production stage and the posterior-predictive
   video obtain the same dye counts and the same labeling record (the render's stream is the
   stage's stream for task 0, simulation 0); the horizon audit's spawned labeling seed goes through
   the same call. The stage's stream for task 0, simulation 0 coincides with the bare-seed stream
   that also seeds placement and render: the production convention, retained and pinned here, so
   that no document can claim the streams are separated.
4. Each runner reaches the draw only through the shared path, and no module or script outside it
   draws dye counts except the diagnostics that test the bare law on purpose or re-derive the
   declared composition from the primitives, each named here with its reason.

Runnable with ``python -m pytest`` or directly:
``MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_labeling_compliance.py``.
"""
import ast
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import labeling as lab
from srm_and_sbi_monomer_dimer_alp import parameterization as par
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS
from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import SubunitLineage

REPO = Path(__file__).resolve().parents[1]
RDS = PARAMETERS.simulation.rds
TYPE_NAMES = RDS.particle_type_names                     # A_*, B_*, B1_*, B2_* over the three modes

# (labeling_law, occupancy) as a command line would pass them: the baseline, an alternative law,
# a scalar occupancy override and a per-species override.
OVERRIDES = [(None, None), ("FAB_BINOMIAL", None), (None, "0.3"), (None, "A=0.2,B=0.9")]
SEEDS = [None, 0, 7, 20260928]


# ---- a trajectory stub: the two things the labeling reads from a ReaDDy trajectory ----------

class _TrajectoryStub:
    """``particle_types`` (type name -> rank), as a ``readdy.Trajectory`` exposes it."""
    def __init__(self):
        self.particle_types = {name: rank for rank, name in enumerate(TYPE_NAMES)}


def _lineage(n_monomers, n_dimers, rng, condition="FAB", n_dimers_two_probe=None):
    """A one-frame subunit lineage of the retained population: monomers are one-subunit hosts; under
    FAB every dimer is a two-subunit ``B`` host; under INLB the dimers are ``B1`` hosts (one subunit)
    except ``n_dimers_two_probe`` of them, which are two-subunit ``B2`` hosts. Each host is in a
    random mobility mode of its species."""
    ranks = _TrajectoryStub().particle_types
    modes = ["f", "s", "i"]
    mono = [ranks[f"A_{m}"] for m in rng.choice(modes, size=n_monomers)]
    host_index = [np.arange(n_monomers)]
    host_rank = [np.asarray(mono, dtype=np.int64)]
    if condition == "FAB":
        dim = [ranks[f"B_{m}"] for m in rng.choice(modes, size=n_dimers)]
        host_index.append(n_monomers + np.repeat(np.arange(n_dimers), 2))
        host_rank.append(np.repeat(np.asarray(dim, dtype=np.int64), 2))
    else:
        n_two = n_dimers // 10 if n_dimers_two_probe is None else n_dimers_two_probe
        n_one = n_dimers - n_two
        one = [ranks[f"B1_{m}"] for m in rng.choice(modes, size=n_one)]
        two = [ranks[f"B2_{m}"] for m in rng.choice(modes, size=n_two)]
        host_index += [n_monomers + np.arange(n_one), n_monomers + n_one + np.repeat(np.arange(n_two), 2)]
        host_rank += [np.asarray(one, dtype=np.int64), np.repeat(np.asarray(two, dtype=np.int64), 2)]
    host_index = np.concatenate(host_index)
    host_rank = np.concatenate(host_rank)
    return SubunitLineage(host_index=host_index[None, :].astype(np.int64),
                          host_rank=host_rank[None, :].astype(np.int64),
                          soul_ids=np.arange(n_monomers + n_dimers, dtype=np.int64))


def _reference_production_labeling(condition, labeling_law, occupancy_arg, tray, lineage, seed, task, sim):
    """The labeling block written out from the primitives: the probe rule, then one dye draw per
    bound probe; the record with the applied rule."""
    from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import monomer_ranks, rank_to_species
    law_name, law = lab.resolve_labeling_law(condition, labeling_law)
    ligand_classes = RDS.condition_setting(condition).ligand_classes
    species_of_rank = rank_to_species(tray)
    host_index_0, host_rank_0 = lineage.host_index[0], lineage.host_rank[0]
    initial_species = np.array([species_of_rank[int(rank)] for rank in host_rank_0])
    labeling_rng = np.random.default_rng(None if seed is None else [seed, task, sim])
    if occupancy_arg is None:
        occupancy = par.occupancy_of(condition)
        occupancy_source = par.occupancy_source_of(condition)
        s2 = par.two_probe_share_of(condition)
        occ_pair = (1.0, 1.0 if ligand_classes else 0.5 * (1.0 + s2))
        bound = np.ones(lineage.n_subunits, dtype=bool)
        is_dimer = initial_species == "B"
        n_hosts = int(host_index_0.max()) + 1
        per_host = np.bincount(host_index_0[is_dimer], minlength=n_hosts)
        two_hosts = np.flatnonzero(per_host == 2)
        if two_hosts.size and not ligand_classes:
            two_probe = labeling_rng.random(two_hosts.size) < s2
            one_hosts = two_hosts[~two_probe]
            if one_hosts.size:
                members = np.flatnonzero(is_dimer)
                order = members[np.argsort(host_index_0[members], kind="stable")]
                first = np.searchsorted(host_index_0[order], one_hosts)
                dark = labeling_rng.integers(0, 2, size=one_hosts.size)
                bound[order[first + dark]] = False
        kappa = law.draw(lineage.n_subunits, labeling_rng)
        dye_counts = np.where(bound, kappa, 0).astype(np.int64)
    else:
        occupancy = lab.parse_occupancy(occupancy_arg)
        occupancy_source = "override"
        s2 = float("nan")
        occ_pair = lab.occupancy_by_species(occupancy, RDS.molecular_species_names)
        p = lab.occupancy_per_subunit(occupancy, initial_species)
        bound = labeling_rng.random(lineage.n_subunits) < p
        kappa = law.draw(lineage.n_subunits, labeling_rng)
        dye_counts = np.where(bound, kappa, 0).astype(np.int64)
    row = lab.labeling_summary(dye_counts, host_index_0, host_rank_0, monomer_ranks(tray),
                               occupancy_by_species_values=occ_pair, probe_bound=bound, two_probe_share=s2)
    return law_name, occupancy, occupancy_source, dye_counts, row


# ---- 1. the shared path is the production block, bit for bit --------------------------------

def test_shared_path_reproduces_the_production_block():
    tray = _TrajectoryStub()
    for condition in lab.LABELING_CONDITIONS:
        lineage = _lineage(700, 150, np.random.default_rng(1), condition)
        for law_spec, occ_spec in OVERRIDES:
            if law_spec is not None and not law_spec.startswith(condition + "_"):
                continue
            plan = lab.resolve_labeling(condition, law_spec, occ_spec)
            for seed in SEEDS:
                for task, sim in ((0, 0), (3, 2)):
                    name, occ, source, ref_dyes, ref_row = _reference_production_labeling(
                        condition, law_spec, occ_spec, tray, lineage, seed, task, sim)
                    assert (plan.law_name, plan.occupancy, plan.occupancy_source) == (name, occ, source)
                    dyes, row = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(seed, task, sim))
                    if seed is None:          # unseeded: nondeterministic, so compare the record's fixed part
                        fixed = [0, 3, 5, 8, 9]
                        assert dyes.shape == ref_dyes.shape and np.array_equal(row[fixed], ref_row[fixed])
                    else:
                        assert np.array_equal(dyes, ref_dyes), (condition, law_spec, occ_spec, seed, task, sim)
                        assert np.array_equal(row, ref_row, equal_nan=True)


def test_declared_occupancy_is_the_default_and_overrides_are_recorded():
    for condition in lab.LABELING_CONDITIONS:
        plan = lab.resolve_labeling(condition)
        assert plan.probe_rule == "classes"
        assert plan.occupancy == par.occupancy_of(condition)
        assert plan.occupancy_source == par.occupancy_source_of(condition) != "override"
        assert plan.two_probe_share == par.two_probe_share_of(condition)
        assert plan.occupancy_pair[0] == 1.0                       # every retained monomer carries a probe
        assert np.isclose(plan.visible_per_subunit[0], plan.law.visible_probability)
    fab, inlb = lab.resolve_labeling("FAB"), lab.resolve_labeling("INLB")
    assert fab.occupancy_source == "derived" and inlb.occupancy_source == "equilibrium"
    assert np.isclose(inlb.occupancy, 0.25 / 5.25) and np.isclose(fab.occupancy, 0.0148, atol=5e-5)
    assert inlb.ligand_classes and not fab.ligand_classes
    assert inlb.occupancy_pair == (1.0, 1.0)                        # B1 one bound subunit, B2 two
    assert np.isclose(fab.occupancy_pair[1], 0.5 * (1.0 + fab.two_probe_share))
    over = lab.resolve_labeling("FAB", None, "A=0.2,B=0.9")
    assert over.probe_rule == "coins" and over.two_probe_share is None
    assert over.occupancy_source == "override" and over.occupancy_pair == (0.2, 0.9)
    try:
        lab.resolve_labeling("FAB", None, "1.5")
    except ValueError:
        pass
    else:
        raise AssertionError("an occupancy outside [0, 1] must be refused")


# ---- 2. the three command lines resolve the same plan ----------------------------------------

def _parsers():
    from srm_and_sbi_monomer_dimer_alp import horizon_audit_runner as ha
    from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv
    from srm_and_sbi_monomer_dimer_alp import simulation_dli_runner as dli
    base = {
        "dli": (dli.build_dli_parser(), ["--total-time-seconds", "2", "--condition", "{cond}"]),
        "ppv": (ppv.build_parser("test"), ["--total-time-seconds", "2", "--kind", "{kind}", "--cell", "0"]),
        "horizon": (ha.build_parser("test"), ["--total-time-seconds", "2", "--condition", "{cond}",
                                             "--phase", "generate"]),
    }
    return base


def test_command_lines_resolve_the_same_plan():
    from srm_and_sbi_monomer_dimer_alp.experiment_support import CONDITION_DISPLAY
    for condition in lab.LABELING_CONDITIONS:
        for law_spec, occ_spec in OVERRIDES:
            if law_spec is not None and not law_spec.startswith(condition + "_"):
                continue
            extra = ([] if law_spec is None else ["--labeling-law", law_spec]) + \
                    ([] if occ_spec is None else ["--occupancy", occ_spec])
            plans = {}
            for name, (parser, argv) in _parsers().items():
                argv = [a.format(cond=condition, kind=CONDITION_DISPLAY[condition]) for a in argv] + extra
                ns = parser.parse_args(argv)
                plans[name] = lab.resolve_labeling(condition, ns.labeling_law, ns.occupancy)
            first = next(iter(plans.values()))
            for name, plan in plans.items():
                assert plan == first, (name, condition, law_spec, occ_spec)


# ---- 3. same condition, lineage and seed -> same labeling -------------------------------------

def test_render_stream_is_the_stage_stream_of_task0_sim0():
    tray = _TrajectoryStub()
    for condition in lab.LABELING_CONDITIONS:
        lineage = _lineage(400, 90, np.random.default_rng(2), condition)
        for occ_spec in (None, "0.4"):
            plan = lab.resolve_labeling(condition, None, occ_spec)
            for seed in (0, 11):
                stage = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(seed, 0, 0))
                render = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(seed))
                assert np.array_equal(stage[0], render[0]) and np.array_equal(stage[1], render[1], equal_nan=True)
                # The horizon audit passes its spawned labeling seed through the same call.
                spawned = int(np.random.SeedSequence([seed, 5]).generate_state(1)[0])
                a = lab.label_trajectory(plan, tray, lineage, np.random.default_rng(spawned))
                b = lab.label_subunits(plan, lineage.host_index[0], lineage.host_rank[0],
                                       {r: TYPE_NAMES[r][0] for r in range(len(TYPE_NAMES))},
                                       [0, 1, 2], np.random.default_rng(spawned))
                assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1], equal_nan=True)


def test_task0_sim0_labeling_stream_coincides_with_the_bare_seed_stream():
    """The retained production convention: numpy's SeedSequence pads missing entropy words with
    zeros, so ``[seed, 0, 0]`` is the bare seed's stream. Separation from the placement and render
    streams is therefore NOT established for task 0, simulation 0 (other tasks and simulations
    differ)."""
    for seed in (0, 7, 20260928):
        bare = np.random.default_rng(seed).random(8)
        assert np.array_equal(lab.labeling_rng(seed, 0, 0).random(8), bare)
        assert not np.array_equal(lab.labeling_rng(seed, 0, 1).random(8), bare)
        assert not np.array_equal(lab.labeling_rng(seed, 3, 2).random(8), bare)


def test_probe_classes_follow_the_retained_population_rule():
    """Every retained monomer carries a probe (visible with the law's probability); under FAB the
    basal dimers split into one- and two-Fab classes at the declared two-probe share, a one-Fab
    dimer keeping exactly one bound subunit; under INLB a B1 host is one bound subunit and a B2
    host two, and every retained subunit is bound."""
    tray = _TrajectoryStub()
    # FAB: 20000 monomers -> visible share = P(dye >= 1) of the law, not thinned by an occupancy coin
    plan = lab.resolve_labeling("FAB")
    lineage = _lineage(20000, 0, np.random.default_rng(3), "FAB")
    dyes, row = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(0))
    assert abs((dyes >= 1).mean() - plan.law.visible_probability) < 0.01
    col = dict(zip(lab.LABELING_SET_COLUMNS, row))
    assert col["occupancy_monomer"] == plan.occupancy_pair[0] == 1.0
    # FAB: 20000 basal dimers -> two-Fab share s_2, one-Fab dimers with exactly one bound subunit
    lineage = _lineage(0, 20000, np.random.default_rng(4), "FAB")
    species_of_rank = {r: TYPE_NAMES[r][0] for r in range(len(TYPE_NAMES))}
    rng = np.random.default_rng(1)
    bound = lab.assign_probes(plan, lineage.host_index[0], lineage.host_rank[0], species_of_rank, rng)
    per_host = np.bincount(lineage.host_index[0], weights=bound.astype(float))
    assert set(np.unique(per_host).tolist()) <= {1.0, 2.0}, "every retained basal dimer carries one or two Fab"
    assert abs((per_host == 2).mean() - plan.two_probe_share) < 0.003
    dyes, row = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(1))
    col = dict(zip(lab.LABELING_SET_COLUMNS, row))
    assert col["dimers_one_probe_0"] + col["dimers_two_probe_0"] == 20000
    assert col["two_probe_share"] == plan.two_probe_share
    # INLB: every retained subunit is bound; B1 hosts are one-probe, B2 hosts two-probe dimers
    plan = lab.resolve_labeling("INLB")
    lineage = _lineage(300, 200, np.random.default_rng(5), "INLB", n_dimers_two_probe=40)
    bound = lab.assign_probes(plan, lineage.host_index[0], lineage.host_rank[0], species_of_rank, rng)
    assert bound.all()
    dyes, row = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(2))
    col = dict(zip(lab.LABELING_SET_COLUMNS, row))
    assert (col["dimers_0"], col["dimers_one_probe_0"], col["dimers_two_probe_0"]) == (200, 160, 40)
    assert col["dimers_two_labeled_0"] <= 40 and col["occupancy_dimer"] == 1.0
    # an override replaces the classes by coins and records no two-probe share
    over = lab.resolve_labeling("INLB", None, "0.4")
    dyes, row = lab.label_trajectory(over, tray, lineage, lab.labeling_rng(3))
    col = dict(zip(lab.LABELING_SET_COLUMNS, row))
    assert np.isnan(col["two_probe_share"]) and abs((dyes >= 1).mean() - 0.4 * over.law.visible_probability) < 0.05


# ---- 4. the draw is reached only through the shared path -------------------------------------

# Where `draw_dye_counts` may be called directly, and why.
ALLOWED_DIRECT_DRAWS = {
    "srm_and_sbi_monomer_dimer_alp/labeling.py": "the shared path itself (label_subunits)",
    "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit.py":
        "Level 3 tests the bare law at occupancy 1 by design; Level 3b uses the shared path",
    "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_PSF_Width.py":
        "synthetic self-test scenes at the bare law, occupancy 1, stated in the script and its note",
    "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Mismatch.py":
        "the flicker harness scenes at the bare law, occupancy 1, stated in the script and its note",
    "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.py":
        "re-derives the declared composition from the primitives (R3, R4) at the declared occupancy",
    "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py":
        "re-derives the visibility chain from the primitives (P6) at the declared occupancy",
}
RUNNERS = ("srm_and_sbi_monomer_dimer_alp/simulation_dli_runner.py",
           "srm_and_sbi_monomer_dimer_alp/posterior_predictive_video_runner.py",
           "srm_and_sbi_monomer_dimer_alp/horizon_audit_runner.py")


def _called_names(path):
    tree = ast.parse((REPO / path).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            names.append(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None)
    return names


def test_runners_label_only_through_the_shared_path():
    for path in RUNNERS:
        called = _called_names(path)
        assert "resolve_labeling" in called and "label_trajectory" in called, path
        for primitive in ("draw_dye_counts", "draw_dye_counts_bound", "assign_probes", "resolve_labeling_law",
                          "occupancy_per_subunit", "labeling_summary"):
            assert primitive not in called, f"{path} calls {primitive} directly"


def test_no_direct_draw_outside_the_allowed_diagnostics():
    offenders = []
    for folder in ("srm_and_sbi_monomer_dimer_alp", "Script_Bank", "tests"):
        for path in sorted((REPO / folder).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel in ("tests/test_labeling_compliance.py", "tests/test_retained_population.py"):
                continue
            called = _called_names(rel)
            if ("draw_dye_counts" in called or "draw_dye_counts_bound" in called) and rel not in ALLOWED_DIRECT_DRAWS:
                offenders.append(rel)
    assert not offenders, f"direct dye draws outside the shared path: {offenders}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
