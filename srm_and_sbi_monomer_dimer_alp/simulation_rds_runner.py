"""Shared RDS-stage engine: the per-condition reaction-diffusion trajectory tier both workflows read.

``run_rds(cfg, args)`` holds the entire RDS orchestration -- pre-run banner,
task-index resolution, dry-run probe, log10 prior sampling + exponentiation, the
per-task theta write, the sim-0 diagnostics block, the per-simulation ReaDDy
build+run with the kernel-leak mitigation, and the end-of-task report. The single
entry-point script (``SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py``) shrinks to:
build the biology ``WorkflowConfig``, parse args, and call ``run_rds``.

One tier per CONDITION, shared by both workflows. The association ratio of the model is a
declared per-condition constant (``parameterization.ConditionSetting``: MET-FAB 0, no
association channels; MET-INLB 1, the reference convention), so the trajectories of the two
conditions differ and ``--condition`` is required here. The eleven reaction-diffusion
parameters -- the same table for both conditions -- are drawn from the biology prior once
per simulation and persisted as the ``Theta_Set``; the trajectories and that ``Theta_Set``
carry the sibling alias plus the condition token (``Paths.rds_alias``) and no workflow
qualifier, because both workflows re-image the same trajectories at the DLI stage under
the condition's labeling law. To the biology workflow the ``Theta_Set`` is the learnable
label; to the detector workflow the same file is the record of the reaction-diffusion
nuisance it marginalizes (``detector_parameterization.DETECTOR_NUISANCE``,
nuisance-from-object: supplied by this tier). There is therefore no detector RDS stage
and no separate RDS-nuisance draw: the detector marginalizes the biology prior by
construction, not by a checked copy of its ranges.
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime, timezone

import numpy as np
import readdy
import zarr

from srm_and_sbi_monomer_dimer_alp.diagnostics import (
    DiagnosticReporter,
    fixed_parameters_table,
    prior_sampling_table,
)
from srm_and_sbi_monomer_dimer_alp.experiment_support import CONDITION_DISPLAY
from srm_and_sbi_monomer_dimer_alp.io import (
    theta_set_schema, tier_record_paths, tier_schema_scan, write_theta_set,
)
from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp.parameterization import (
    PARAMETERS,
    RunTiming,
    declared_row_values,
    is_log_row,
    parameterization_of,
    parameterization_raw_of,
    prior_center,
    theta_lower_bound,
    theta_upper_bound,
    to_physical,
)
from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import (
    build_simulation, build_system, initial_composition_of, rank_to_molecule, reaction_channels,
)
from srm_and_sbi_monomer_dimer_alp.utils import (
    SINK, SOCK, log_memory_state, log_resource_limits, probe_resources,
)
from srm_and_sbi_monomer_dimer_alp.workflow import WorkflowConfig


# Short-form unit labels for terminal display (the canonical parameterization
# UNIT field uses long-form English for documentation clarity).
_UNIT_DISPLAY = {
    "Count": "Count",
    "Square Micrometer Per Second": "μm²/s",
    "Square Micrometer Per (Count*Second)": "μm²/(count·s)",
    "Count Per Second": "count/s",
    "Dimensionless": "dimensionless",
    "Fraction": "fraction",
}


def _build_sim(theta, condition, seed, skin_factor, verbose):
    """Build the ReaDDy simulation for one theta under the run's condition: register the
    condition's reaction channels (`build_system`) and place the initial particles
    (`build_simulation`)."""
    stem = build_system(theta, condition, verbose=verbose)
    # `stem` is reachable only through the returned Simulation; deleting the
    # Simulation (+ gc) in the engine loop releases the ReaDDy kernel and the system.
    return build_simulation(stem, theta, condition, seed=seed, skin_factor=skin_factor, verbose=verbose)


def _require_biology_config(cfg: WorkflowConfig) -> None:
    """A condition's RDS tier is generated once, through the biology config, whose ``Paths``
    carry the unqualified sibling alias and whose parameter table is the condition's prior
    (eleven coordinates under MET-FAB, twelve under MET-INLB) the tier samples. A detector
    config here would only re-generate the same tier under a qualified name, so it is refused."""
    if cfg.tag != "biology":
        raise ValueError(
            f"run_rds: a condition's RDS trajectory tier is shared by both workflows and is "
            f"generated once through the unqualified entry point (biology config); got workflow "
            f"{cfg.tag!r}. The detector re-images this tier at its DLI stage.")


def run_rds(cfg: WorkflowConfig, args: argparse.Namespace) -> None:
    """Run the full RDS generation pipeline for the given workflow + CLI args."""
    _require_biology_config(cfg)

    # Per-run timing from the required --total-time-seconds + the fixed frame cadence.
    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    split = args.split.upper()   # "TRAIN" / "TEST" / "EVAL" namespace suffix
    data_bank_root = PARAMETERS.machine.root_for(split)  # TRAIN/TEST -> scratch tier; EVAL -> permanent (single-tier machines: always data_bank_root)
    compress = not args.no_compress
    output_fmt = "npy" if args.no_compress else "zarr"

    # ---- Pre-run banner -------------------------------------------------
    machine = PARAMETERS.machine
    geom = PARAMETERS.simulation.stem
    rds_cfg = PARAMETERS.simulation.rds
    # The tier is per condition (the association setting differs), so the run's Paths carry
    # the condition token from here on; rds_alias composes the tier alias from it.
    condition = args.condition
    paths = cfg.paths.with_condition(condition)
    r_on = rds_cfg.association_ratio_of(condition)
    # The condition's table, its declared constants and its full specification, bound once:
    # every draw, schema, banner and diagnostic of this run reads them.
    table = parameterization_of(condition)
    declared = declared_row_values(condition)
    table_raw = parameterization_raw_of(condition)
    div = "=" * 72

    print(div)
    print(f" {paths.rds_alias} — Simulation_RDS   (the {CONDITION_DISPLAY[condition]} trajectory tier: both workflows)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)

    print("\nMachine profile:")
    print(f"  name              : {machine.name}")
    print(f"  running_mode      : {machine.running_mode}")
    print(f"  compute_backend   : {machine.compute_backend}")
    if machine.gpu_device_index is not None:
        print(f"  gpu_device_index  : {machine.gpu_device_index}")
    print(f"  num_workers       : {machine.num_workers}")

    print("\nRun configuration (CLI args):")
    print(f"  --condition          : {condition}   ({CONDITION_DISPLAY[condition]}; association ratio R_ON = {r_on:g}"
          + (" -> no association channel, pre-existing dimers may dissociate)" if r_on == 0.0
             else " -> lambda_on = R_ON x 6 D_A / r^2 for every association channel)"))
    print(f"  --total-time-seconds : {args.total_time_seconds}")
    if args.task_id is not None:
        print(f"  --task-id            : {args.task_id}        (this run generates exactly one task)")
    else:
        print(f"  --tasks              : {args.tasks}        (this run generates tasks 0..{args.tasks - 1})")
    print(f"  --task-simulations   : {args.task_simulations}")
    print(f"  --split              : {args.split}   (namespace suffix: _{split})")
    print(f"  --seed               : {args.seed}")
    _skin_factor = (args.skin_factor if args.skin_factor is not None
                    else PARAMETERS.simulation.rds.neighbor_list_skin_factor)
    _skin_nm = _skin_factor * PARAMETERS.simulation.stem.particle_diameter_nm
    print(f"  --skin-factor        : {args.skin_factor}   "
          f"(effective neighbor-list skin {_skin_factor:g}x diameter = {_skin_nm:g} nm; "
          f"performance only, no physics change)")
    print(f"  --no-compress        : {args.no_compress}  (output format: .{output_fmt})")
    print(f"  --verbose            : {args.verbose}")
    print(f"  --show               : {args.show}")

    print("\nSimulation timing (derived):")
    print(f"  frame_time_seconds       : {timing.frame_time_seconds}       "
          f"({1 / timing.frame_time_seconds:.0f} Hz)")
    print(f"  frame_count              : {timing.frame_count}        "
          f"(= {timing.total_time_seconds} / {timing.frame_time_seconds})")
    print(f"  steps_per_frame          : {timing.steps_per_frame}")
    print(f"  total_steps              : {timing.total_steps}")
    print(f"  delta_time_nanoseconds   : {timing.delta_time_nanoseconds}")

    print("\nSystem geometry:")
    print(f"  pixel_size_nm        : {geom.pixel_size_nm}")
    print(f"  root_size_px         : {geom.root_size_px}        "
          f"(image: {geom.root_size_px} × {geom.root_size_px})")
    print(f"  box_size_nm          : {geom.box_size}")
    print(f"  particle_diameter_nm : {geom.particle_diameter_nm}")

    print(f"\nParticle types of the condition (molecular species x mobility mode, {condition}):")
    print(f"  {rds_cfg.particle_type_names_for(condition)}")
    print(f"  molecular species of the condition: {tuple(sp.name for sp in rds_cfg.species_for(condition))}; "
          f"monomer/dimer classification {rds_cfg.molecular_species_names} (the census axis); "
          f"modes {rds_cfg.mobility.modes} = (fast, slow, immobile); "
          f"inheritance: {rds_cfg.mobility.inheritance}")
    _n_channels = len(reaction_channels(np.array([prior_center(para) for para in table], dtype=float), condition))
    print(f"  reaction channels under {condition}: {_n_channels} "
          f"({'3 fissions + 8 mobility switches, no association' if r_on == 0.0 else '6 fusions + 3 fissions + 3 conversions B1 -> A + 12 mobility switches'})")

    timing_label = timing.label
    print("\nOutput destinations:")
    print(f"  data_bank_root  : {data_bank_root}")
    print(f"  trajectories    : <data_bank>/{paths.video_subdir}/"
          f"{paths.trajectory_repo}/{paths.rds_alias}_{timing_label}_TASK_{{n}}/"
          f"{paths.rds_alias}_{timing_label}_TASK_{{n}}_SIM_{{m}}.h5   ({condition} tier, both workflows)")
    print(f"  theta sets      : <data_bank>/{paths.theta_subdir}/"
          f"{paths.rds_alias}_{timing_label}_Theta_Set_TASK_{{n}}.{output_fmt}   "
          f"(biology labels = detector RDS nuisance)")
    print(f"\nParameter table of {condition}: {len(table)} inferred coordinates "
          f"({sum(1 for p in table if not is_log_row(p))} linear, the rest log10)")
    print(f"  keys              : {[p['KEY'] for p in table]}")
    print(f"  declared constants: {declared if declared else 'none'}"
          + ("   (the window-start allocation is declared, not inferred, under this condition)" if declared else ""))

    if args.verbose:
        print(f"\nLearnable theta prior spec ({len(table)} parameters, "
              f"sampled in estimator space, mapped to physical values by to_physical; "
              f"decided prior ranges of 2026-09-14, composition rows revised 2026-10-15):")
        for para in table:
            lo, hi = para["PRIOR_RANGE"]
            unit = _UNIT_DISPLAY.get(para["UNIT"], para["UNIT"])
            derived = para.get("DERIVED_UNIT")
            derived_str = (f"   derived: {_UNIT_DISPLAY.get(derived, derived)}"
                           if derived else "")
            scale = "log10" if is_log_row(para) else "linear"
            print(f"  {para['KEY']:<32}  {scale:<6} ∈ [{lo:+6.2f}, {hi:+6.2f}]   "
                  f"units: {unit}{derived_str}")

    print(f"\n{div}\n")

    if args.probe:
        log_resource_limits()

    # ---- Determine which task indices to run ----------------------------
    # --task-id K runs exactly one task (HPC array fan-out: one task per job).
    # It reproduces the same per-task theta as task K of a full --tasks run:
    # rng.uniform fills rows in stream order, so row K occupies the same stream
    # positions regardless of the total row count, as long as we draw >= K+1 rows.
    if args.task_id is not None:
        task_indices = [args.task_id]
    else:
        task_indices = list(range(args.tasks))
    n_rows = max(task_indices) + 1

    # ---- Dry run: resolve the planned task/sim workload + output destinations
    # and exit before sampling, the ReaDDy loop, and any directory creation -- so
    # it computes nothing and writes nothing. RDS is the first stage, so it has no
    # inputs to probe -- only the writes it would produce.
    if args.dry_run:
        n_tasks = len(task_indices)
        planned = n_tasks * args.task_simulations
        task_span = (f"{task_indices[0]}" if n_tasks == 1
                     else f"{task_indices[0]}..{task_indices[-1]}")
        print(f"[DRY RUN] plans {n_tasks} task(s) (index {task_span}) x "
              f"{args.task_simulations} sim(s) = {planned} trajectory(ies), "
              f"split _{split}.")
        for task in task_indices:
            theta_set_path = paths.theta_set_path(
                task, data_bank_root, timing_label, compress, split)
            traj_dir = paths.trajectory_dir(
                task, data_bank_root, timing_label, split)
            print(f"  writes theta set    (task {task}): {theta_set_path}")
            print(f"  writes trajectories (task {task}): {traj_dir}/  "
                  f"({args.task_simulations} .h5 file(s))")
        print("\n[DRY RUN] configuration validated; all outputs resolved.")
        print("[DRY RUN] no trajectories generated.")
        return

    # ---- Sample the parameter sets in estimator space, map to physical values ----
    # (the one shared rule per row: log rows exponentiated, the two linear fraction rows pass
    # through; the box is the condition's table, eleven or twelve coordinates)
    low = np.array(theta_lower_bound(condition))
    high = np.array(theta_upper_bound(condition))
    rng = np.random.default_rng(args.seed)
    theta_flow = rng.uniform(
        low=low, high=high,
        size=(n_rows, args.task_simulations, len(low)),
    )
    theta_sets = to_physical(theta_flow, table)

    run_start = time.time()

    # ---- Per-task loop --------------------------------------------------
    for loop_i, task in enumerate(task_indices):
        task_alias = task
        print(f"{SOCK} Task {task_alias}  ({loop_i + 1}|{len(task_indices)}) {SOCK}")

        if args.verbose:
            log_memory_state()

        # Ensure the per-task trajectory directory exists.
        traj_dir = paths.trajectory_dir(task_alias, data_bank_root, timing_label, split)
        traj_dir.mkdir(parents=True, exist_ok=True)

        theta_set_path = paths.theta_set_path(
            task_alias, data_bank_root, timing_label, compress, split)

        # ---- Mixed-schema guard: a tier accepts new tasks only under its own schema -----
        # Every OTHER record of the tier (final names only; a `.partial` name is an in-progress
        # write of a concurrent task and is skipped, and so is a record that a concurrent task is
        # replacing at the moment it is read) is classified against this run's table and
        # condition, reading schemas only. A mismatched record (a superseded tier that was not
        # moved aside) or an unknown record (no readable schema: unknown is never a compatibility
        # verdict) refuses the task before anything is written. --supersede-tier, which
        # Generate_Datasets passes under --overwrite-rds, tolerates mismatched records of the
        # tier being replaced deliberately; unknown records still refuse. This is a mixed-schema
        # check, not proof of dataset identity.
        others = [p for p in tier_record_paths(theta_set_path.parent, paths.theta_set_pattern,
                                               project_alias=paths.rds_alias, timing_label=timing_label,
                                               split=split)
                  if p.resolve() != theta_set_path.resolve()]
        scan = tier_schema_scan(others, table, condition=condition)
        print(f"  Tier schema scan ({len(others)} other record(s)): {scan.verdict}")
        if scan.unknown or (scan.mismatched and not args.supersede_tier):
            raise SystemExit(
                f"run_rds [{condition}, task {task_alias}]: refusing to write into a tier that holds records "
                f"of another schema or without a readable schema -- {scan.verdict}. Move the superseded "
                f"tier aside, or pass --supersede-tier (Generate_Datasets --overwrite-rds) to replace a "
                f"tier generated under another table deliberately; unreadable records always refuse.")
        if scan.mismatched:
            print(f"  --supersede-tier: {len(scan.mismatched)} superseded record(s) are being replaced.")

        # Write the sampled set WITH its schema (io.write_theta_set: .zarr attrs, or a JSON
        # sidecar bound by digest to a plain .npy; built under .partial names and published so
        # that the final name only ever holds a complete record paired with its own schema). The schema
        # names the condition's keys in order, their prior bounds and scales, the condition and
        # timing, and the package version, so no reader can consume this tier under a different
        # table or condition (io.load_theta_set refuses it).
        theta_set_data = theta_sets[task]  # shape (task_simulations, n_params)
        theta_schema = theta_set_schema(table, condition=condition,
                                        timing_label=timing_label, generator="rds")
        write_theta_set(theta_set_path, theta_set_data, theta_schema)
        print(f"  Theta_Set schema: {len(theta_schema['parameter_keys'])} keys, condition "
              f"{theta_schema['condition']}, package {theta_schema['package_version']}")

        # ---- Diagnostics reporter (debug mode) ------------------------
        reporter = DiagnosticReporter(
            stage="RDS",
            enabled=args.debug or args.debug_dump,
            dump=args.debug_dump,
            dump_dir=(paths.debug_run_dir(data_bank_root, timing_label, "RDS", split)
                      / f"TASK_{task_alias}"),
            run_label=f"{paths.rds_alias}_{timing_label}_TASK_{task_alias}_{split}",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
        theta_flow_task = theta_flow[task]

        # ---- Per-simulation loop ----------------------------------------
        for sim in range(args.task_simulations):
            sim_start = time.time()
            theta = theta_set_data[sim]

            if args.verbose:
                print(f"\n  Sampled theta (sim {sim + 1}|{args.task_simulations}):")
                for i, para in enumerate(table):
                    val = theta[i]
                    scale = "log10" if is_log_row(para) else "linear"
                    print(f"    {para['KEY']:<36}  =  {val:10.4g}   "
                          f"({scale} coordinate = {theta_flow_task[sim][i]:+.3f})")
                for key, value in declared.items():
                    print(f"    {key:<36}  =  {value:10.4g}   (declared constant of {condition})")

            # Placement RNG follows --seed (default None -> non-deterministic).
            # Generation is non-deterministic by design: ReaDDy's stepper is
            # unseeded, so identical-posterior reproducibility is unreachable
            # regardless; the sampled set is persisted per task and the global
            # task index is the provenance handle.
            smut = _build_sim(theta, condition, args.seed, args.skin_factor, args.verbose)

            traj_path = paths.trajectory_path(
                task_alias, sim, data_bank_root, timing_label, split)
            if traj_path.exists():
                traj_path.unlink()
            smut.output_file = str(traj_path)
            smut.progress_output_stride = timing.total_steps
            print(f"  Writing trajectory: {traj_path}")
            smut.run(
                n_steps=timing.total_steps,
                timestep=timing.delta_time_nanoseconds * readdy.units.nanosecond,
                show_summary=False,
            )

            sim_elapsed = time.time() - sim_start
            print(f"{SINK} Simulation {sim + 1}|{args.task_simulations} done  "
                  f"(elapsed: {sim_elapsed:.1f}s) {SINK}")

            # ---- Sim-0 diagnostics (debug mode) -----------------------
            if reporter.enabled and sim == 0:
                composition = initial_composition_of(theta, condition)
                total_initial = composition.n_monomers + composition.n_dimers
                theta_flow_sim = theta_flow_task[sim]
                in_bounds = bool(np.all(theta_flow_sim >= low)
                                 and np.all(theta_flow_sim <= high))

                reporter.checkpoint(
                    "theta sample (sim 0)",
                    n_params=len(theta),
                    initial_particles=total_initial,
                )
                reporter.check(
                    "theta_in_prior_bounds", in_bounds,
                    "all estimator-space values within the prior box",
                    note="the sampled parameters lie inside the prior box they were drawn from "
                         "(log10 for log rows, the value itself for the two linear fraction rows).",
                )
                c = composition
                allocation = (f"allocation R_active={c.relative_count_active_dimers_initial:.4f} of capacity "
                              f"{c.capacity} -> c={c.n_pairs_allocated} pairs" if c.ligand_classes
                              else "allocation declared 0 (reference = start)")
                reporter.stat(
                    "initial_composition",
                    f"reference: {c.n_monomers_reference} probe-bound monomers + {c.n_dimers_reference} dimers "
                    f"({c.n_dimers_one_probe} one-probe + {c.n_dimers_two_probe_reference} two-probe); "
                    f"start (placed): {c.n_monomers} monomers + {c.n_dimers} dimers "
                    f"({c.n_dimers_one_probe} one-probe + {c.n_dimers_two_probe} two-probe); N_R={c.n_subunits}",
                    note=f"window-start composition realized from N_total={c.n_total_true:.0f} at occupancy "
                         f"{c.occupancy:.4f} with basal reference fraction f={c.fraction_basal_dimers_reference:.4f}; "
                         f"{allocation}; underlying M {c.n_monomers_underlying_reference:.1f} -> "
                         f"{c.n_monomers_underlying_start:.1f}, D {c.n_dimers_underlying_reference:.1f} -> "
                         f"{c.n_dimers_underlying_start:.1f}, f_start={c.fraction_basal_dimers_start:.4f}, derived "
                         f"active share={c.active_share:.4f}; the represented count "
                         f"({'n_A + n_B1 + 2 n_B2' if c.ligand_classes else 'n_A + 2 n_B'}) is conserved by the "
                         f"allocation and by every channel.",
                )
                reporter.check(
                    "initial_particles_positive", total_initial > 0,
                    f"total={total_initial}",
                    note="the simulation starts with at least one particle.",
                )
                reporter.check_file("trajectory", traj_path)

                # Full prior-range vs sampled-value summary for this simulation.
                headers, prior_rows = prior_sampling_table(table, theta)
                reporter.table(
                    "Prior sampling (sim 0)", headers, prior_rows,
                    note="Prior box (estimator space) and the value drawn for this simulation. "
                         "count_total, the basal reference fraction and, under INLB, the allocation fix "
                         "the starting monomer/dimer numbers the run was seeded with (see "
                         "initial_composition) -- cross-check against the rendered video.",
                )
                fixed_headers, fixed_rows = fixed_parameters_table(table_raw)
                reporter.table(
                    "Fixed parameters", fixed_headers, fixed_rows,
                    note="Non-learnable parameters held constant across all simulations (Known "
                         "scientific constants + tuning Hyperparameters, and the condition's declared "
                         "constants): camera, PSF, photophysics, capture radius, brightness quantiles"
                         + (", the declared window-start allocation" if declared else "") + ".",
                )
                reporter.stat(
                    "total_initial_particles", total_initial,
                    note="monomer + dimer particles of the starting population placed at t=0 (the "
                         "represented subunit total N_R counts a two-subunit dimer twice and a one-probe "
                         "INLB dimer once).",
                )

                # Read the trajectory back for reaction-event diagnostics.
                tray = readdy.Trajectory(filename=str(traj_path))
                _, recs = tray.read_observable_reaction_counts()
                reaction_counts = recs["reactions"]

                # Frame-0 census: the particles the trajectory holds at its first frame are the
                # STARTING counts of the window-start composition, per molecular species, and their
                # subunits are N_R. Fatal: a disagreement means the placement and the record differ.
                _, ranks_per_frame, _, _ = tray.read_observable_particles()
                molecule_of_rank = rank_to_molecule(tray)
                frame0 = np.asarray(ranks_per_frame[0])
                census = {}
                for rank in np.unique(frame0):
                    census[molecule_of_rank[int(rank)]] = int(np.sum(frame0 == rank)) + census.get(molecule_of_rank[int(rank)], 0)
                sto = rds_cfg.stoichiometry
                expected = ({sto.monomer.name: c.n_monomers, sto.dimer_one_probe.name: c.n_dimers_one_probe,
                             sto.dimer_two_probe.name: c.n_dimers_two_probe} if c.ligand_classes
                            else {sto.monomer.name: c.n_monomers, sto.dimer.name: c.n_dimers})
                expected = {k: v for k, v in expected.items() if v}
                subunits_of_molecule = {sp.name: sp.subunits for sp in sto.species}
                census_subunits = sum(n * subunits_of_molecule[m] for m, n in census.items())
                reporter.check(
                    "frame0_census_equals_starting_counts",
                    census == expected and census_subunits == c.n_subunits,
                    f"frame 0 {census} ({census_subunits} subunits) vs starting counts {expected} (N_R {c.n_subunits})",
                    note="the trajectory's first frame holds exactly the starting counts of the window-start "
                         "composition (reference population plus the allocation under INLB).",
                )
                if census != expected or census_subunits != c.n_subunits:
                    raise AssertionError(f"run_rds debug [{condition}]: frame-0 census {census} ({census_subunits} "
                                         f"subunits) differs from the starting counts {expected} (N_R {c.n_subunits}).")
                for rec_name, rec_counts in reaction_counts.items():
                    reporter.stat(
                        f"events[{rec_name}]", int(np.sum(rec_counts)),
                        note="total firings of this reaction channel over the run.")

                if reporter.dump:
                    from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import (
                        collapse_species_axis,
                        extract_subunit_lineage,
                        extract_trajectory_poses,
                    )
                    from srm_and_sbi_monomer_dimer_alp.visualization_rds import (
                        figure_position_heatmap,
                        figure_reaction_events,
                    )
                    reporter.save_figure(
                        "reaction_events", figure_reaction_events(reaction_counts),
                        caption="Total firings of each reaction channel of the condition "
                                "over the trajectory (association fusions, dissociation "
                                "fissions and B1 -> A conversions, mobility switches).",
                    )
                    poses = collapse_species_axis(extract_trajectory_poses(tray))
                    # Replay the reaction records into the subunit lineage the DLI stage
                    # consumes; the extractor fails loud on any conservation violation, so
                    # reaching the stat below IS the check.
                    lineage = extract_subunit_lineage(tray)
                    reporter.check(
                        "subunit_lineage_replays", True,
                        f"{lineage.n_subunits} subunits over {lineage.soul_ids.shape[0]} "
                        f"particle ids",
                        note="the reaction records replay into a per-frame subunit -> "
                             "particle table covering every subunit exactly once (the "
                             "RDS/DLI handoff of the labeling model).",
                    )
                    xy = poses[..., :2].reshape(-1, 2)
                    xy = xy[~np.isnan(xy).any(axis=1)]
                    reporter.save_figure(
                        "position_heatmap", figure_position_heatmap(xy),
                        caption="2D spatial occupancy of all particle positions "
                                "across all frames; should fill the simulation box.",
                    )

            if args.show:
                tray = readdy.Trajectory(filename=str(traj_path))
                _, recs = tray.read_observable_reaction_counts()
                for rec_name, counts in recs["reactions"].items():
                    print(f"  reaction {rec_name}: {np.count_nonzero(counts)} events")

            # ---- Optional debug probe ------------------------------------
            if args.probe:
                _th, _fd, _rss = probe_resources()
                print(f"[probe] sim {sim + 1}: threads={_th} fds={_fd} "
                      f"rss_mb={_rss}", flush=True)
            # Release the ReaDDy CPU kernel (its worker-thread pool and the
            # observable/output handles) and reclaim memory before the next
            # simulation. Without this each iteration leaks ~320 threads and
            # ~350 MB; a task's RSS then reaches the per-task memory cap after
            # ~47 sims and the cgroup stalls (manifesting as a hang). ``smut`` is
            # the sole reference to the ReaDDy system, so deleting it releases both.
            del smut
            gc.collect()

        # ---- Diagnostics: end-of-task summary + report ----------------
        reporter.summary()
        reporter.write_report()

    total_elapsed = time.time() - run_start
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")


def build_rds_parser() -> argparse.ArgumentParser:
    """Construct the RDS CLI parser (identical for both workflows)."""
    parser = argparse.ArgumentParser(
        description="Generate ReaDDy reaction-diffusion trajectories for the DIMER model.",
    )
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
        help="Simulation duration per trajectory in seconds (required; e.g. 2.0, 5.0).",
    )
    parser.add_argument(
        "--condition", required=True, choices=LABELING_CONDITIONS,
        help="Experimental condition whose trajectory tier this run generates: FAB (MET-FAB; "
             "association ratio 0 -> no association channel, pre-existing dimers may dissociate) "
             "or INLB (MET-INLB; association ratio 1, the reference convention). Required: the "
             "association setting is per condition, so each condition has its own tier, shared "
             "by both workflows and named <sibling alias>_<CONDITION>_<timing>_....",
    )
    parser.add_argument(
        "--tasks", type=int, default=2,
        help="Number of theta-set batches to generate (default: 2). Ignored when "
             "--task-id is given.",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Generate exactly one task with this index (for HPC array fan-out: "
             "one task per job). Reproduces the same per-task theta as task <id> of "
             "the equivalent full --tasks run. Overrides --tasks.",
    )
    parser.add_argument(
        "--task-simulations", type=int, default=5,
        help="Number of simulations per task (default: 5).",
    )
    parser.add_argument(
        "--split", choices=["train", "test", "eval"], default="train",
        help="Dataset role this run generates, written to its own namespace "
             "(filename suffix _TRAIN / _TEST / _EVAL). Default: train.",
    )
    parser.add_argument(
        "--seed",
        type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v),
        default=None,
        help="RNG seed for theta sampling and initial particle placement. "
             "Default: None (non-deterministic).",
    )
    parser.add_argument(
        "--skin-factor", type=float, default=None,
        help="ReaDDy neighbor-list (Verlet) skin as a MULTIPLE of the particle "
             "diameter (skin = skin-factor x particle_diameter_nm). PERFORMANCE-ONLY "
             "knob: it coarsens the cell-linked-list grid in the large, dilute box; it "
             "does NOT change the physics (reactions still fire at the true reaction "
             "radius, so the output is statistically identical). Default: None -> the "
             "configured PARAMETERS.simulation.rds.neighbor_list_skin_factor (10x = "
             "100 nm), which sits on the fast plateau. See that field for the U-shaped "
             "cost rationale.",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Save theta sets as .npy instead of compressed .zarr.",
    )
    parser.add_argument(
        "--supersede-tier", action="store_true",
        help="Replace a tier generated under another parameter table deliberately: the mixed-schema "
             "guard then tolerates the tier's mismatched records (they are being regenerated) instead "
             "of refusing the task. Records without a readable schema always refuse. Generate_Datasets "
             "passes this under --overwrite-rds; the default refuses to write into a tier that holds "
             "records of another schema.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and inputs, print what would be read/written, then exit "
             "without running the stage (no GPU, no compute). Use before a queue submission or a long local run.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print diagnostic info during setup (rates, particle counts).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="After each simulation, print non-zero reaction counts from the trajectory.",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Debug instrumentation: log RLIMIT_NPROC/NOFILE at startup and "
             "threads/open-fds/RSS after each simulation (logging only; no behavior "
             "change).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug diagnostics: per-step checkpoints, fail-loud invariant "
             "checks, and an end-of-stage PASS/FAIL summary (console only).",
    )
    parser.add_argument(
        "--debug-dump", action="store_true",
        help="Implies --debug; additionally writes a Markdown diagnostic report "
             "and PNG figures under <data_bank>/Debug/. Skipped if disk space is low.",
    )
    return parser
