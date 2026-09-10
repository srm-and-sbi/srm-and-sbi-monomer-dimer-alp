"""Entry-point: generate the TRAIN / TEST / EVAL datasets in one command.

Runs the generation flow three times -- once per split -- with the task counts derived
from the dataset-sizing rule: one RDS trajectory tier per requested condition, then one DLI
pass per requested workflow over each condition's tier. The three splits are independent
draws that never mix: by default no seed is passed, so each split draws fresh entropy
(non-deterministic).

One RDS tier per condition, shared by both workflows. The association ratio of the
reaction-diffusion model is a declared per-condition constant (MET-FAB 0: no association
channel; MET-INLB 1: the reference convention), so the two conditions' trajectories differ
and the RDS stage runs once per condition per split. Within a condition the trajectories
and the eleven-parameter ``Theta_Set`` carry the sibling alias plus the condition token and
serve both workflows (the biology's learnable labels are the detector's marginalized
reaction-diffusion nuisance); the DLI passes -- condition-specific (the static labeling law
of MET-FAB or MET-INLB re-images that condition's trajectories) and workflow-specific (the
detector draws the six imaging labels from its prior box; the biology draws them from the
condition's calibrated ``Nuisance_DLI`` artifact) -- fan out over ``--workflows`` for each
condition.

Two guards protect each condition's tier. Without ``--reuse-rds`` the RDS stage generates
it and REFUSES to run over an existing tier: the RDS stage would replace the trajectories
and the ``Theta_Set`` with a fresh draw, silently mislabeling every video already rendered
from the old tier; ``--overwrite-rds`` lifts that refusal deliberately. With ``--reuse-rds``
the RDS stage is skipped and every requested condition's tier must be present for every
planned task. The biology DLI needs
the condition's ``Nuisance_DLI`` artifact (minted by the detector chain), so a biology pass
is refused up front, before anything runs, when a requested condition has none.

Sizing rule (CORE = TRAIN + TEST):
    test_tasks  = round(test_fraction * core_tasks)              # default 0.2
    train_tasks = core_tasks - test_tasks                        # 0.8 of CORE
    eval_tasks  = max(ceil(eval_floor / sims), round(eval_fraction * core_tasks))
                                                                 # EVAL = max(floor, 0.1*CORE)
So EVAL is floored at `eval_floor` samples (for a stable recovery number) and
only the proportional `0.1 * CORE` exceeds it once CORE >= eval_floor / eval_fraction.

Seeds: none by default (non-deterministic; the global task index is the
provenance handle and theta is persisted per task). An optional --seed restores
a deterministic distinct-seed-per-split run: train = seed, test = seed + 1,
eval = seed + 2 (forwarded to the RDS and every DLI pass of that split).

Usage:
    # first campaign: both conditions' tiers + the detector videos of both conditions
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_MONOMER_DIMER_ALP_Generate_Datasets.py \\
        --workflows detector --conditions FAB,INLB --core-tasks 10 --task-simulations 10 \\
        --total-time-seconds 2.0 --seed None
    # once the detector chain has minted the per-condition Nuisance_DLI artifacts:
    ... --workflows biology --conditions FAB,INLB --reuse-rds --core-tasks 10 --task-simulations 10 --total-time-seconds 2.0
    # preview the plan and the prerequisite checks without generating anything:
    ... --workflows detector,biology --conditions FAB,INLB --core-tasks 10 --task-simulations 10 --total-time-seconds 2.0 --dry-run
"""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
from srm_and_sbi_monomer_dimer_alp.detector_nuisance_dli import artifact_path as nuisance_dli_artifact_path
from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming

_PRIME = Path(__file__).resolve().parent
_RDS = _PRIME / "SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py"
# One DLI entry point per workflow, both over the condition's trajectory tier.
_DLI = {
    "detector": _PRIME / "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Simulation_DLI.py",
    "biology": _PRIME / "SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py",
}
WORKFLOWS = tuple(_DLI)


def _parse_list(text: str, choices, flag: str) -> List[str]:
    """Comma-separated list restricted to ``choices``, order preserved, no repeats."""
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    if not items:
        raise SystemExit(f"{flag} needs at least one value from {list(choices)}.")
    unknown = [item for item in items if item not in choices]
    if unknown:
        raise SystemExit(f"{flag}: unknown value(s) {unknown}; choose from {list(choices)}.")
    if len(set(items)) != len(items):
        raise SystemExit(f"{flag}: a value repeats in {items}.")
    return items


def _timing_label(args: argparse.Namespace) -> str:
    return RunTiming(total_time_seconds=args.total_time_seconds,
                     frames=PARAMETERS.simulation.timing).label


def _run(script: Path, label: str, split: str, tasks: int, seed: Optional[int],
         args: argparse.Namespace, condition: Optional[str] = None) -> None:
    """Run one stage pass for one split, or print it under --dry-run."""
    cmd = [
        sys.executable, str(script),
        "--split", split,
        "--tasks", str(tasks),
        "--task-simulations", str(args.task_simulations),
        "--total-time-seconds", str(args.total_time_seconds),
    ]
    if seed is not None:                  # default: omit --seed -> non-deterministic
        cmd += ["--seed", str(seed)]
    if args.no_compress:
        cmd.append("--no-compress")
    # --skin-factor is an RDS-only knob (a ReaDDy neighbor-list performance setting;
    # the DLI entry points have no such flag). Omit when None -> the code default.
    if script == _RDS and args.skin_factor is not None:
        cmd += ["--skin-factor", str(args.skin_factor)]
    # --condition selects the tier (RDS: the association setting is per condition) and the
    # labeling law (DLI); every stage pass takes it.
    if condition is not None:
        cmd += ["--condition", condition]

    line = f"{label:<20} {split:<5} (--tasks {tasks}, --seed {seed})"
    if args.dry_run:
        print(f"  [dry-run] {line}")
        return

    print(f"  - {line} ...", end="", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        print(" FAILED")
        raise SystemExit(
            f"FAILED: {script.name} --split {split}"
            f"{' --condition ' + condition if condition else ''} (exit code {result.returncode}). "
            f"Re-run that command directly to see the error.")
    print(f" done ({time.time() - t0:.1f}s)")


def _plan(args: argparse.Namespace) -> dict:
    """Compute per-split task counts and seeds from the sizing rule."""
    cfg = PARAMETERS.inference.training
    test_fraction = args.test_fraction if args.test_fraction is not None else cfg.test_fraction
    eval_fraction = args.eval_fraction if args.eval_fraction is not None else cfg.eval_fraction
    eval_floor = args.eval_floor if args.eval_floor is not None else cfg.eval_floor

    sims = args.task_simulations
    core_tasks = args.core_tasks
    core_samples = core_tasks * sims

    if core_samples < cfg.core_min:
        raise SystemExit(
            f"CORE = {core_samples} samples ({core_tasks} tasks x {sims} sims) is "
            f"below the minimum of {cfg.core_min}. Increase --core-tasks or "
            f"--task-simulations.")

    test_tasks = max(1, round(test_fraction * core_tasks))
    train_tasks = core_tasks - test_tasks
    if train_tasks < 1:
        raise SystemExit(
            f"train_tasks = {train_tasks} < 1 (core_tasks={core_tasks}, "
            f"test_tasks={test_tasks}); increase --core-tasks.")
    eval_tasks = max(math.ceil(eval_floor / sims), round(eval_fraction * core_tasks))

    # Default: no seed -> each split draws fresh entropy (non-deterministic).
    # An explicit --seed restores a distinct-seed-per-split run for optional determinism.
    base_seeds = (None, None, None) if args.seed is None else (
        args.seed, args.seed + 1, args.seed + 2)

    return {
        "sims": sims,
        "core_tasks": core_tasks,
        "core_samples": core_samples,
        "test_fraction": test_fraction,
        "eval_fraction": eval_fraction,
        "eval_floor": eval_floor,
        "splits": [
            ("train", train_tasks, base_seeds[0]),
            ("test", test_tasks, base_seeds[1]),
            ("eval", eval_tasks, base_seeds[2]),
        ],
    }


def _tier_status(plan: dict, args: argparse.Namespace) -> Dict[tuple, tuple]:
    """Per (split, condition): how many of the planned tasks already have that condition's
    tier ``Theta_Set`` (the tier's per-task marker; the trajectories sit beside it)."""
    timing_label = _timing_label(args)
    compress = not args.no_compress
    status = {}
    for split, tasks, _seed in plan["splits"]:
        root = PARAMETERS.machine.root_for(split.upper())
        for condition in args.conditions:
            paths = PARAMETERS.paths.with_condition(condition)
            present = sum(
                paths.theta_set_path(task, root, timing_label, compress, split.upper()).exists()
                for task in range(tasks))
            status[(split, condition)] = (present, tasks)
    return status


def _biology_artifacts(args: argparse.Namespace) -> Dict[str, Path]:
    """Per condition: the calibrated ``Nuisance_DLI`` artifact the biology DLI pass requires
    (the detector chain mints it; the biology stage refuses to run without it)."""
    timing_label = _timing_label(args)
    posit_dir = PARAMETERS.machine.root_for("EVAL") / PARAMETERS.paths.posit_subdir
    return {
        condition: nuisance_dli_artifact_path(
            posit_dir, det.detector_paths(PARAMETERS.paths).with_condition(condition).project_alias,
            timing_label)
        for condition in args.conditions
    }


def main(args: argparse.Namespace) -> None:
    args.workflows = _parse_list(args.workflows, WORKFLOWS, "--workflows")
    args.conditions = _parse_list(args.conditions, LABELING_CONDITIONS, "--conditions")
    if args.reuse_rds and args.overwrite_rds:
        raise SystemExit("--reuse-rds and --overwrite-rds are mutually exclusive.")
    plan = _plan(args)
    sims = plan["sims"]
    div = "=" * 72

    print(div)
    print(" SRM_AND_SBI_MONOMER_DIMER_ALP — Generate_Datasets"
          + ("   [DRY RUN]" if args.dry_run else ""))
    print(div)
    print(f"  task_simulations : {sims}")
    print(f"  CORE             : {plan['core_tasks']} tasks  "
          f"({plan['core_samples']} samples)")
    print(f"  rule             : test_fraction={plan['test_fraction']}, "
          f"eval_fraction={plan['eval_fraction']}, eval_floor={plan['eval_floor']}")
    print(f"  total-time-secs  : {args.total_time_seconds}   (timing label {_timing_label(args)})")
    print(f"  RDS tiers        : one per condition; "
          f"{'REUSE the existing tiers (RDS skipped)' if args.reuse_rds else 'generate them' + (' (overwriting existing ones)' if args.overwrite_rds else '')}")
    print(f"  workflows        : {', '.join(args.workflows)}   (one DLI pass each, over each condition's tier)")
    print(f"  conditions       : {', '.join(args.conditions)}   (one RDS tier each -- the association setting is per condition -- and its DLI labeling law)")
    print()
    print(f"  {'split':<7}{'cond':<6}{'tasks':>7}{'samples':>9}{'seed':>7}{'tier present':>14}")
    tier = _tier_status(plan, args)
    for split, tasks, seed in plan["splits"]:
        for condition in args.conditions:
            present, planned = tier[(split, condition)]
            print(f"  {split.upper():<7}{condition:<6}{tasks:>7}{tasks * sims:>9}{str(seed):>7}{f'{present}/{planned}':>14}")
    print(div)

    # ---- Prerequisite checks (refuse before anything runs) --------------
    problems = []
    for split, tasks, _seed in plan["splits"]:
        for condition in args.conditions:
            present, planned = tier[(split, condition)]
            if args.reuse_rds and present < planned:
                problems.append(
                    f"--reuse-rds: split {split.upper()} has the {condition} tier for {present}/{planned} "
                    f"planned tasks; generate it first (drop --reuse-rds) or shrink the plan.")
            if not args.reuse_rds and not args.overwrite_rds and present > 0:
                problems.append(
                    f"split {split.upper()} already holds the {condition} tier for {present}/{planned} planned "
                    f"tasks; the RDS stage would replace those trajectories and Theta_Set with a fresh "
                    f"draw and silently mislabel every video already rendered from them. Pass "
                    f"--reuse-rds to re-image the existing tier, or --overwrite-rds to regenerate it.")
    if "biology" in args.workflows:
        for condition, artifact in _biology_artifacts(args).items():
            ok = artifact.exists()
            print(f"  biology {condition:<4} needs Nuisance_DLI : {artifact}  [{'OK' if ok else 'MISSING'}]")
            if not ok:
                problems.append(
                    f"biology DLI for {condition} needs the calibrated Nuisance_DLI artifact above; "
                    f"run the detector chain for {condition} (--workflows detector, then detector "
                    f"Inference/Experiment and the Nuisance_DLI analysis) before the biology pass.")
    if problems:
        print()
        for problem in problems:
            print(f"  !! {problem}")
        if args.dry_run:
            print("\nDry run complete -- the plan above would be REFUSED for the reason(s) listed.")
            return
        raise SystemExit("Refusing to generate: fix the listed prerequisite(s) and re-run.")

    run_start = time.time()
    for split, tasks, seed in plan["splits"]:
        print(f"\n[{split.upper()}]")
        for condition in args.conditions:
            if not args.reuse_rds:
                _run(_RDS, f"RDS tier {condition}", split, tasks, seed, args, condition=condition)
            for workflow in args.workflows:
                _run(_DLI[workflow], f"DLI {workflow} {condition}", split, tasks, seed, args,
                     condition=condition)

    if args.dry_run:
        print("\nDry run complete -- nothing was generated.")
    else:
        print(f"\nAll three datasets generated in {time.time() - run_start:.1f}s.")


def parse_args(argv=None) -> argparse.Namespace:
    """Construct the CLI parser and parse argv."""
    parser = argparse.ArgumentParser(
        description="Generate TRAIN/TEST/EVAL datasets per the dataset-sizing rule: one RDS tier "
                    "per condition per split, then one DLI pass per workflow over each tier.",
    )
    parser.add_argument(
        "--workflows", default="detector,biology",
        help="Comma-separated DLI passes to render over each condition's tier, in run order: "
             "detector (imaging drawn from the detector prior; needs no artifact) and/or biology "
             "(imaging drawn from the condition's Nuisance_DLI artifact, which the detector chain "
             "must have minted). Default: detector,biology.",
    )
    parser.add_argument(
        "--conditions", default=",".join(LABELING_CONDITIONS),
        help="Comma-separated experimental conditions; each gets its own RDS tier (the association "
             "setting is a per-condition constant) and its labeling law re-images it "
             f"(from {list(LABELING_CONDITIONS)}; FAB = MET-FAB, INLB = MET-INLB). Forwarded to "
             "the RDS and the DLI passes. Default: all.",
    )
    parser.add_argument(
        "--reuse-rds", action="store_true",
        help="Skip the RDS stage and re-image the condition tiers already on disk (every requested "
             "condition's tier must be present for every planned task of every split).",
    )
    parser.add_argument(
        "--overwrite-rds", action="store_true",
        help="Allow the RDS stage to regenerate a tier that already exists (default: refuse, "
             "because the fresh draw would mislabel every video rendered from the old tier).",
    )
    parser.add_argument(
        "--core-tasks", type=int, required=True,
        help="Number of TRAIN+TEST (CORE) task files; split 0.8/0.2 into train/test.",
    )
    parser.add_argument(
        "--task-simulations", type=int, default=10,
        help="Simulations per task (default: 10).",
    )
    parser.add_argument(
        "--total-time-seconds", type=float,
        required=True,
        help="Simulation duration per video in seconds (required; e.g. 2.0, 5.0).",
    )
    parser.add_argument(
        "--seed", type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v), default=None,
        help="Optional base seed; if set, train=seed, test=seed+1, eval=seed+2. "
             "Default None -> non-deterministic, each split a fresh draw.",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=None,
        help="TEST as a fraction of CORE (default from config: 0.2).",
    )
    parser.add_argument(
        "--eval-fraction", type=float, default=None,
        help="EVAL as a fraction of CORE (default from config: 0.1).",
    )
    parser.add_argument(
        "--eval-floor", type=int, default=None,
        help="Minimum EVAL samples for a stable recovery number "
             "(default from config: 10).",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Save sets as .npy instead of compressed .zarr (passed to RDS/DLI).",
    )
    parser.add_argument(
        "--skin-factor", type=float, default=None,
        help="ReaDDy neighbor-list skin as a MULTIPLE of the particle diameter "
             "(performance-only; RDS stage only -- not passed to DLI). Default None "
             "-> the code default (PARAMETERS.simulation.rds.neighbor_list_skin_factor, "
             "10x = 100 nm).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the computed sizing plan, the prerequisite checks, and the per-split "
             "commands without generating anything.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
