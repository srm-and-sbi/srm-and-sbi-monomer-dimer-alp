"""Every renderer of simulated trajectories labels them the way the training data were labeled.

The DLI stage of both workflows, the posterior-predictive video and the horizon audit resolve a
run's static labeling with `labeling.resolve_labeling` and draw it with `labeling.label_trajectory`
(the condition's law and its declared or derived probe occupancy, applied by initial molecular
species). Before this path existed the two runners drew the bare law at occupancy 1, so their
renders did not carry the observation model of the training data. These tests hold the contract:

1. The shared path reproduces, bit for bit, the labeling block the DLI stage carried inline before
   it moved onto the shared path (kept verbatim below as the reference), for both conditions,
   several seeds, tasks and simulations, and the supported overrides (an alternative law, a scalar
   and a per-species occupancy).
2. The three command lines accept the same labeling flags and resolve the same plan from them:
   by default the condition's declared (MET-INLB) or derived (MET-FAB) occupancy, an override
   recorded as such.
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
TYPE_NAMES = RDS.particle_type_names                     # ('A_f', 'A_s', 'A_i', 'B_f', 'B_s', 'B_i')

# (labeling_law, occupancy) as a command line would pass them: the baseline, an alternative law,
# a scalar occupancy override and a per-species override.
OVERRIDES = [(None, None), ("FAB_BINOMIAL", None), (None, "0.3"), (None, "A=0.2,B=0.9")]
SEEDS = [None, 0, 7, 20260928]


# ---- a trajectory stub: the two things the labeling reads from a ReaDDy trajectory ----------

class _TrajectoryStub:
    """``particle_types`` (type name -> rank), as a ``readdy.Trajectory`` exposes it."""
    def __init__(self):
        self.particle_types = {name: rank for rank, name in enumerate(TYPE_NAMES)}


def _lineage(n_monomers, n_dimers, rng):
    """A one-frame subunit lineage: monomers are one-subunit hosts, dimers two-subunit hosts, each
    host in a random mobility mode of its species."""
    ranks = _TrajectoryStub().particle_types
    mono = [ranks[f"A_{m}"] for m in rng.choice(["f", "s", "i"], size=n_monomers)]
    dim = [ranks[f"B_{m}"] for m in rng.choice(["f", "s", "i"], size=n_dimers)]
    host_index = np.concatenate([np.arange(n_monomers), n_monomers + np.repeat(np.arange(n_dimers), 2)])
    host_rank = np.concatenate([np.asarray(mono, dtype=np.int64), np.repeat(np.asarray(dim, dtype=np.int64), 2)])
    return SubunitLineage(host_index=host_index[None, :].astype(np.int64),
                          host_rank=host_rank[None, :].astype(np.int64),
                          soul_ids=np.arange(n_monomers + n_dimers, dtype=np.int64))


def _reference_production_labeling(condition, labeling_law, occupancy_arg, tray, lineage, seed, task, sim):
    """The DLI stage's labeling block before it moved onto the shared path, kept verbatim."""
    from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import monomer_ranks, rank_to_species
    law_name, law = lab.resolve_labeling_law(condition, labeling_law)
    if occupancy_arg is None:
        occupancy = par.occupancy_of(condition)
        occupancy_source = par.occupancy_source_of(condition)
    else:
        occupancy = lab.parse_occupancy(occupancy_arg)
        occupancy_source = "override"
    occ_pair = lab.occupancy_by_species(occupancy, RDS.molecular_species_names)
    species_of_rank = rank_to_species(tray)
    initial_species = [species_of_rank[int(rank)] for rank in lineage.host_rank[0]]
    labeling_rng = np.random.default_rng(None if seed is None else [seed, task, sim])
    dye_counts = lab.draw_dye_counts(law, lineage.n_subunits, labeling_rng,
                                     occupancy=lab.occupancy_per_subunit(occupancy, initial_species))
    row = lab.labeling_summary(dye_counts, lineage.host_index[0], lineage.host_rank[0], monomer_ranks(tray),
                               occupancy_by_species_values=occ_pair)
    return law_name, occupancy, occupancy_source, dye_counts, row


# ---- 1. the shared path is the production block, bit for bit --------------------------------

def test_shared_path_reproduces_the_production_block():
    tray = _TrajectoryStub()
    lineage = _lineage(700, 150, np.random.default_rng(1))
    for condition in lab.LABELING_CONDITIONS:
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
                        assert dyes.shape == ref_dyes.shape and np.array_equal(row[[0, 3, 5, 8, 9]], ref_row[[0, 3, 5, 8, 9]])
                    else:
                        assert np.array_equal(dyes, ref_dyes), (condition, law_spec, occ_spec, seed, task, sim)
                        assert np.array_equal(row, ref_row)


def test_declared_occupancy_is_the_default_and_overrides_are_recorded():
    for condition in lab.LABELING_CONDITIONS:
        plan = lab.resolve_labeling(condition)
        assert plan.occupancy == par.occupancy_of(condition)
        assert plan.occupancy_source == par.occupancy_source_of(condition) != "override"
        assert np.allclose(plan.visible_per_subunit, par.visibility_of(condition))
    assert lab.resolve_labeling("FAB").occupancy_source == "derived"
    assert lab.resolve_labeling("INLB").occupancy_source == "declared"
    over = lab.resolve_labeling("FAB", None, "A=0.2,B=0.9")
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
    lineage = _lineage(400, 90, np.random.default_rng(2))
    for condition in lab.LABELING_CONDITIONS:
        for occ_spec in (None, "0.4"):
            plan = lab.resolve_labeling(condition, None, occ_spec)
            for seed in (0, 11):
                stage = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(seed, 0, 0))
                render = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(seed))
                assert np.array_equal(stage[0], render[0]) and np.array_equal(stage[1], render[1])
                # The horizon audit passes its spawned labeling seed through the same call.
                spawned = int(np.random.SeedSequence([seed, 5]).generate_state(1)[0])
                a = lab.label_trajectory(plan, tray, lineage, np.random.default_rng(spawned))
                b = lab.label_subunits(plan, lineage.host_index[0], lineage.host_rank[0],
                                       {r: TYPE_NAMES[r][0] for r in range(len(TYPE_NAMES))},
                                       [0, 1, 2], np.random.default_rng(spawned))
                assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


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


def test_declared_occupancy_thins_the_visible_population():
    """The defect this path closes: at the declared MET-FAB occupancy about 12.5 % of subunits are
    visible, against about 81 % under the bare law."""
    tray = _TrajectoryStub()
    lineage = _lineage(20000, 0, np.random.default_rng(3))
    plan = lab.resolve_labeling("FAB")
    dyes, row = lab.label_trajectory(plan, tray, lineage, lab.labeling_rng(0))
    visible = (dyes >= 1).mean()
    assert abs(visible - 0.125) < 0.01, visible
    bare = lab.draw_dye_counts(plan.law, lineage.n_subunits, np.random.default_rng(0))
    assert abs((bare >= 1).mean() - plan.law.visible_probability) < 0.01
    col = dict(zip(lab.LABELING_SET_COLUMNS, row))
    assert col["occupancy_monomer"] == plan.occupancy_pair[0]


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
        for primitive in ("draw_dye_counts", "resolve_labeling_law", "occupancy_per_subunit", "labeling_summary"):
            assert primitive not in called, f"{path} calls {primitive} directly"


def test_no_direct_draw_outside_the_allowed_diagnostics():
    offenders = []
    for folder in ("srm_and_sbi_monomer_dimer_alp", "Script_Bank", "tests"):
        for path in sorted((REPO / folder).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel == "tests/test_labeling_compliance.py":
                continue
            if "draw_dye_counts" in _called_names(rel) and rel not in ALLOWED_DIRECT_DRAWS:
                offenders.append(rel)
    assert not offenders, f"direct dye draws outside the shared path: {offenders}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
