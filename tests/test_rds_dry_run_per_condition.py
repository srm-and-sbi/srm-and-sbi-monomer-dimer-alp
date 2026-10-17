"""The trajectory stage per condition: the dry run states the condition's coordinates and declared
constants; ``run_rds`` with the simulation stubbed writes stores of the condition's width whose
columns are the draws in physical units; the mixed-schema guard classifies every other record of
the tier; the dataset orchestrator reports tier status by schema.

Run directly from the repository root (no ReaDDy simulation is stepped)::

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/test_rds_dry_run_per_condition.py
"""
import dataclasses
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MACHINE_PROFILE", "mars_pc")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srm_and_sbi_monomer_dimer_alp import io as pio                                  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import parameterization as par                   # noqa: E402
from srm_and_sbi_monomer_dimer_alp import simulation_rds_runner as rds_runner       # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS                # noqa: E402
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow                 # noqa: E402

ENTRY = REPO / "Script_Bank" / "Prime" / "SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py"
PY = sys.executable


def _dry_run(condition):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    out = subprocess.run([PY, str(ENTRY), "--condition", condition, "--total-time-seconds", "2.0", "--tasks", "1",
                          "--task-simulations", "1", "--dry-run"], capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_the_dry_run_states_the_coordinates_and_the_declared_constants():
    fab = _dry_run("FAB")
    assert "Parameter table of FAB: 11 inferred coordinates (1 linear, the rest log10)" in fab
    assert "declared constants: {'relative_count_active_dimers_initial': 0.0}" in fab
    assert "[DRY RUN] no trajectories generated." in fab
    inlb = _dry_run("INLB")
    assert "Parameter table of INLB: 12 inferred coordinates (2 linear, the rest log10)" in inlb
    assert "declared constants: none" in inlb
    assert "'relative_count_active_dimers_initial']" in inlb


class _StubSimulation:
    """What the engine loop needs of a ReaDDy simulation: it never steps."""
    def __init__(self):
        self.output_file = None
        self.progress_output_stride = None

    def run(self, n_steps, timestep, show_summary=False):
        return None


def _args(condition, tasks, sims, seed, root, extra=()):
    argv = ["--condition", condition, "--total-time-seconds", "2.0", "--tasks", str(tasks),
            "--task-simulations", str(sims), "--seed", str(seed), "--split", "train", *extra]
    return rds_runner.build_rds_parser().parse_args(argv)


def _with_temp_bank(fn):
    original_machine, original_build = PARAMETERS.machine, rds_runner._build_sim
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "Data_Bank"
        root.mkdir()
        object.__setattr__(PARAMETERS, "machine",
                           dataclasses.replace(original_machine, data_bank_root=root, scratch_data_bank_root=None))
        rds_runner._build_sim = lambda theta, condition, seed, skin_factor, verbose: _StubSimulation()
        try:
            fn(root)
        finally:
            rds_runner._build_sim = original_build
            object.__setattr__(PARAMETERS, "machine", original_machine)


def test_run_rds_writes_stores_of_the_condition_width_in_physical_units():
    def body(root):
        cfg = biology_workflow()
        for cond, width in (("FAB", 11), ("INLB", 12)):
            paths = cfg.paths.with_condition(cond)
            table = par.parameterization_of(cond)
            rds_runner.run_rds(cfg, _args(cond, 2, 3, 11, root))
            low, high = np.array(par.theta_lower_bound(cond)), np.array(par.theta_upper_bound(cond))
            draws = np.random.default_rng(11).uniform(low=low, high=high, size=(2, 3, width))
            for task in range(2):
                path = paths.theta_set_path(task, root, "2S_50FPS", True, "TRAIN")
                stored = np.asarray(pio.load_theta_set(path, table, condition=cond))
                assert stored.shape == (3, width)
                for j, entry in enumerate(table):
                    if par.is_log_row(entry):
                        assert np.allclose(stored[:, j], 10.0 ** draws[task, :, j])
                    else:
                        assert np.array_equal(stored[:, j], draws[task, :, j]) and np.all((stored[:, j] >= 0) & (stored[:, j] <= 1))
                other = "INLB" if cond == "FAB" else "FAB"
                status = pio.theta_set_status(path, par.parameterization_of(other), condition=other)
                assert status.startswith("SCHEMA MISMATCH"), status
    _with_temp_bank(body)


def test_the_mixed_schema_guard_on_the_six_fixtures():
    def body(root):
        cfg = biology_workflow()
        paths = cfg.paths.with_condition("FAB")
        table = par.parameterization_of("FAB")
        schema = pio.theta_set_schema(table, condition="FAB", timing_label="2S_50FPS", generator="rds")
        old = dict(schema, package_version="0.1.39")
        old["parameter_keys"] = ["count_total", "ratio_dimer_monomer_initial"] + schema["parameter_keys"][2:]
        theta = np.tile([par.prior_center(e) for e in table], (2, 1))
        rec = lambda task: paths.theta_set_path(task, root, "2S_50FPS", True, "TRAIN")  # noqa: E731
        # no records -> proceeds (task 5 of an otherwise empty tier)
        rds_runner.run_rds(cfg, _args("FAB", 1, 2, 1, root, ["--task-id", "5"]))
        assert rec(5).exists()
        # current records only -> proceeds
        rds_runner.run_rds(cfg, _args("FAB", 1, 2, 1, root, ["--task-id", "6"]))
        assert rec(6).exists()
        # a .partial name beside current records -> skipped, proceeds
        (rec(7).parent / (rec(7).name + ".partial")).mkdir()
        rds_runner.run_rds(cfg, _args("FAB", 1, 2, 1, root, ["--task-id", "8"]))
        assert rec(8).exists()
        # a task's own record is never a reason to refuse: re-running task 6 replaces it
        rds_runner.run_rds(cfg, _args("FAB", 1, 2, 2, root, ["--task-id", "6"]))
        assert rec(6).exists()
        # current plus superseded -> refused, the superseded task named, nothing written
        pio.write_theta_set(rec(1), theta, old)
        try:
            rds_runner.run_rds(cfg, _args("FAB", 1, 2, 1, root, ["--task-id", "9"]))
        except SystemExit as exc:
            assert "TASK_1" in str(exc) and "another schema" in str(exc), str(exc)
        else:
            raise AssertionError("a superseded record did not refuse")
        assert not rec(9).exists()
        # --supersede-tier tolerates the superseded record (it is being replaced) and proceeds
        rds_runner.run_rds(cfg, _args("FAB", 1, 2, 1, root, ["--task-id", "9", "--supersede-tier"]))
        assert rec(9).exists()
        # an unreadable record beside the superseded one -> refused, both named, no compatibility verdict
        rec(2).mkdir(); (rec(2) / "junk").write_text("x")
        try:
            rds_runner.run_rds(cfg, _args("FAB", 1, 2, 1, root, ["--task-id", "10"]))
        except SystemExit as exc:
            text = str(exc)
            assert "TASK_1" in text and "TASK_2" in text and "unknown" in text and "verified" not in text, text
        else:
            raise AssertionError("an unreadable record did not refuse")
        # unknown records refuse even under --supersede-tier
        try:
            rds_runner.run_rds(cfg, _args("FAB", 1, 2, 1, root, ["--task-id", "10", "--supersede-tier"]))
        except SystemExit as exc:
            assert "TASK_2" in str(exc) and "unreadable" in str(exc)
        else:
            raise AssertionError("an unreadable record passed under --supersede-tier")
        assert not rec(10).exists()
        # an unreadable record alone (fresh tier of the other condition) -> refused as unknown
        ipaths = cfg.paths.with_condition("INLB")
        bad = ipaths.theta_set_path(0, root, "2S_50FPS", True, "TRAIN")
        bad.mkdir(); (bad / "junk").write_text("x")
        try:
            rds_runner.run_rds(cfg, _args("INLB", 1, 2, 1, root, ["--task-id", "1"]))
        except SystemExit as exc:
            assert "TASK_0" in str(exc) and "unknown" in str(exc)
        else:
            raise AssertionError("an unreadable record alone did not refuse")
        # the INLB tier is a separate namespace: the FAB records are not scanned there
        assert pio.tier_record_paths(bad.parent, ipaths.theta_set_pattern, project_alias=ipaths.rds_alias,
                                     timing_label="2S_50FPS", split="TRAIN") == [bad]
    _with_temp_bank(body)


def test_generate_datasets_reports_tier_status_by_schema():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_datasets", REPO / "Script_Bank" / "Prime" / "SRM_AND_SBI_MONOMER_DIMER_ALP_Generate_Datasets.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    def body(root):
        argv = ["--total-time-seconds", "2.0", "--core-tasks", "5", "--task-simulations", "2",
                "--eval-floor", "2", "--conditions", "FAB", "--workflows", "detector", "--dry-run", "--reuse-rds"]
        args = gen.parse_args(argv)
        args.conditions, args.workflows = ["FAB"], ["detector"]      # as main() parses them
        plan = gen._plan(args)
        paths = PARAMETERS.paths.with_condition("FAB")
        table = par.parameterization_of("FAB")
        schema = pio.theta_set_schema(table, condition="FAB", timing_label="2S_50FPS", generator="rds")
        old = dict(schema, package_version="0.1.39")
        old["parameter_keys"] = ["count_total", "ratio_dimer_monomer_initial"] + schema["parameter_keys"][2:]
        theta = np.tile([par.prior_center(e) for e in table], (2, 1))
        splits = {split: tasks for split, tasks, _ in plan["splits"]}
        # TRAIN: every planned task current; TEST: one superseded; EVAL: nothing
        for task in range(splits["train"]):
            pio.write_theta_set(paths.theta_set_path(task, root, "2S_50FPS", True, "TRAIN"), theta, schema)
        pio.write_theta_set(paths.theta_set_path(0, root, "2S_50FPS", True, "TEST"), theta, old)
        status = gen._tier_status(plan, args)
        assert status[("train", "FAB")] == (splits["train"], splits["train"], splits["train"])
        assert status[("test", "FAB")] == (0, 1, splits["test"])
        assert status[("eval", "FAB")] == (0, 0, splits["eval"])
        import contextlib
        import io as _io

        def main_text(arg_list):
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                gen.main(gen.parse_args(arg_list))     # under --dry-run a refusal is printed, not raised
            return buf.getvalue()

        text = main_text(argv)
        assert "!! --reuse-rds" in text and "superseded tier, SCHEMA MISMATCH" in text and "TEST" in text, text
        assert "would be REFUSED" in text
        # a plan whose every tier is current and planned passes the prerequisite checks under --dry-run
        for task in range(splits["test"]):
            pio.write_theta_set(paths.theta_set_path(task, root, "2S_50FPS", True, "TEST"), theta, schema)
        for task in range(splits["eval"]):
            pio.write_theta_set(paths.theta_set_path(task, root, "2S_50FPS", True, "EVAL"), theta, schema)
        text = main_text(argv)
        assert "would be REFUSED" not in text and "Dry run complete" in text, text
        # without --reuse-rds an existing tier is refused and its schema state is named
        text = main_text([a for a in argv if a != "--reuse-rds"])
        assert "the current schema" in text and "--overwrite-rds" in text and "would be REFUSED" in text, text
    _with_temp_bank(body)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)")
