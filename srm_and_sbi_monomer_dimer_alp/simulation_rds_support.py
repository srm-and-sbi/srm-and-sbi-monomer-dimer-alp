"""ReaDDy primitives for the reaction-diffusion simulation (RDS) stage.

This module wraps the ReaDDy 2 particle-based reaction-diffusion solver
(Hoffmann et al., 2019, "ReaDDy 2: Fast and flexible software framework
for interacting particle reaction dynamics", PLoS Comp Bio.
https://doi.org/10.1371/journal.pcbi.1006830) for the three-species DIMER
model used by this sibling.

Three molecular species:
    A = Monomer
    B = Mobile Dimer
    C = Immobile Dimer

Two reactions:
    A + A <-> B    (dimerization / dissociation; the forward rate is
                    parameterized relative to the diffusion-limited cap)
    B    <-> C    (immobilization / mobilization)

Functions:
    build_system(theta, ...)
        Builds the ReaDDy ReactionDiffusionSystem: registers species with
        their diffusion constants and adds the four reactions. Returns the
        configured system, ready to be wrapped in a Simulation.

    build_simulation(stem, theta, ...)
        Wraps the system in a Simulation, registers observables (per-frame
        particle positions, per-step reaction counts, per-step reaction
        records), and places initial particles uniformly in the box. Returns
        the runnable Simulation.

    extract_trajectory_poses(tray, ...)
        Reads a saved .h5 trajectory file and produces a dense
        (frame, particle, spatial-dim, species-rank) tensor of particle
        positions.

    collapse_species_axis(tray_poses)
        Collapses the species-rank axis of that tensor to
        (frame, particle, spatial-dim).

    extract_subunit_lineage(tray, ...)
        Replays the reaction records and produces, per frame, the particle
        hosting each receptor subunit -- the RDS/DLI handoff the DOL-explicit
        observation layer needs, since ReaDDy assigns a new particle id to
        every reaction product (including plain species conversions).

The reaction records are the only RDS-side requirement of the labeling model: they
carry no labeling logic, they only make the subunit lineage recoverable. All
labeling statistics live in the DLI stage (`labeling`, `simulation_dli_support`).
"""

import warnings
from typing import NamedTuple, Optional, Tuple

import numpy as np
import readdy

from .parameterization import (
    PARAMETERS,
    parameter_find,
)


# =============================================================================
# Reaction-diffusion system builder
# =============================================================================

def build_system(theta: np.ndarray,
                 particle_species_names: Optional[Tuple[str, ...]] = None,
                 verbose: bool = False) -> "readdy.ReactionDiffusionSystem":
    """Build a ReaDDy ReactionDiffusionSystem for the three-species DIMER model.

    Args:
        theta: Learnable-parameter values (physical units, exp-transformed from
            the log-space prior). 1D numpy array of length `len(PARAMETERIZATION)`
            (currently 7), in the order defined by `parameter_find`.
        particle_species_names: Optional override for the species names.
            Defaults to `PARAMETERS.simulation.rds.particle_species_names` —
            the standard ('A', 'B', 'C').
        verbose: If True, print diffusion constants and reaction rates to stdout.

    Returns:
        A configured `readdy.ReactionDiffusionSystem`, ready to be passed to
        `build_simulation`.

    Diffusion model:
        - The monomer diffusion `D_A` is the learnable parameter `diffusivity_alp`.
        - Dimer diffusions are specified RELATIVE to `D_A`:
              `D_B = R_B * D_A`,  `D_C = R_C * D_A`,
          where `R_B` and `R_C` are the learnable parameters
          `relative_diffusivity_bet` and `relative_diffusivity_chi`.
          The prior samples these ratios directly. Storing relative values
          guarantees `D_B` and `D_C` track the magnitude of `D_A` without
          requiring the prior to encode their joint dependence.

    Reaction model — dimerization:
        Dimerization (A + A -> B) is parameterized by `R_ON`, a dimensionless
        ratio in (0, 1] of the macroscopic rate to the diffusion-limited cap:

              kappa_ON = R_ON * kappa_ON_CAP
              kappa_ON_CAP = 4 * pi * D_R * rho_CAP      (Smoluchowski)

        with `D_R = 2 * D_A` (relative diffusion coefficient of two A particles)
        and `rho_CAP = capture_radius` (in micrometres, converted from nm).
        The microscopic ReaDDy rate is then:

              lamb_ON = kappa_ON / V_CAP
              V_CAP = (4/3) * pi * rho_CAP^3            (effective capture volume)

        Parameterizing via R_ON instead of an absolute kappa_ON guarantees
        the macroscopic rate never exceeds the physically attainable
        diffusion-limited rate, regardless of how the prior is sampled.

    Reaction model — other reactions:
        Dissociation (B -> A + A), immobilization (B -> C), and mobilization
        (C -> B) are parameterized by absolute macroscopic rates in counts
        per second. The corresponding microscopic rates are just the
        macroscopic rates with ReaDDy's per-second unit attached.

    Boundary conditions:
        [False, False, True] — the simulation box is open in xy (the
        observation plane) and periodic in z (the thin direction
        perpendicular to the membrane).
    """
    if particle_species_names is None:
        particle_species_names = PARAMETERS.simulation.rds.particle_species_names

    # --- Extract parameter values from theta -------------------------------
    diffusion_keys = ("diffusivity_alp", "relative_diffusivity_bet", "relative_diffusivity_chi")
    theta_diffusion = theta[[parameter_find(k) for k in diffusion_keys]]
    diffusion_rates = np.empty(3)
    diffusion_rates[0] = theta_diffusion[0]                       # D_A (monomer)
    diffusion_rates[1] = theta_diffusion[1] * theta_diffusion[0]  # D_B = R_B * D_A
    diffusion_rates[2] = theta_diffusion[2] * theta_diffusion[0]  # D_C = R_C * D_A

    # Smoluchowski reaction radius = the center-to-center contact distance of two
    # monomers (2 * monomer_radius = 1 * diameter), derived from the physical
    # particle diameter so it tracks the per-dataset geometry. particle_diameter_nm
    # is the single physical input; the table's capture_radius entry is display-only.
    capture_radius = PARAMETERS.simulation.stem.particle_diameter_nm  # nm

    reaction_keys = ("relative_rate_dimerization", "rate_dissociation",
                     "rate_immobility", "rate_mobility")
    theta_reaction = theta[[parameter_find(k) for k in reaction_keys]]
    R_ON = theta_reaction[0]               # dimerization ratio (0, 1]; relative to diffusion-limited cap
    kappa_OFF = theta_reaction[1]          # dissociation rate (1/s)
    kappa_IMMOBILITY = theta_reaction[2]   # B -> C rate (1/s)
    kappa_MOBILITY = theta_reaction[3]     # C -> B rate (1/s)

    # --- Build the system --------------------------------------------------
    stem_geometry = PARAMETERS.simulation.stem
    stem = readdy.ReactionDiffusionSystem(
        box_size=stem_geometry.box_size,
        unit_system=stem_geometry.unit_dict,
    )
    stem.periodic_boundary_conditions = [False, False, True]

    # Add species with diffusion constants (in um^2/s, attached to ReaDDy units).
    for idx, species_name in enumerate(particle_species_names):
        diffusion_constant = (
            diffusion_rates[idx]
            * pow(readdy.units.micrometer, 2)
            / readdy.units.second
        )
        stem.add_species(name=species_name, diffusion_constant=diffusion_constant)

    # --- Diffusion-limited dimerization rate ---------------------------
    D_R = 2 * diffusion_rates[0]                          # um^2 / s
    rho_CAP_um = capture_radius * 1e-3                    # nm -> um
    kappa_ON_CAP = 4 * np.pi * D_R * rho_CAP_um           # um^3 / s
    kappa_ON = R_ON * kappa_ON_CAP                        # um^3 / s
    V_CAP = (4 / 3) * np.pi * pow(rho_CAP_um, 3)          # um^3
    lamb_ON = (kappa_ON / V_CAP) / readdy.units.second    # 1/s
    lamb_OFF = kappa_OFF / readdy.units.second
    lamb_IMMOBILITY = kappa_IMMOBILITY / readdy.units.second
    lamb_MOBILITY = kappa_MOBILITY / readdy.units.second

    # --- Add the four reactions ----------------------------------------
    # The four channels implement the reversible dimerization / mobilization
    # scheme: A + A <-> B (dimerization / dissociation) and B <-> C
    # (immobilization / mobilization). A=monomer, B=mobile dimer, C=immobile
    # dimer. See the DIMER reaction model in PROJECT_CONTEXT.md.

    # A + A -> B : forward dimerization (microscopic rate lamb_ON, derived from
    # R_ON relative to the diffusion-limited Smoluchowski cap).
    stem.reactions.add_fusion(
        name="A + A => B", type_from1="A", type_from2="A", type_to="B",
        rate=lamb_ON, educt_distance=capture_radius, weight1=0.5, weight2=0.5,
    )
    # B -> A + A : reverse dissociation (macroscopic rate kappa_OFF).
    stem.reactions.add_fission(
        name="B => A + A", type_from="B", type_to1="A", type_to2="A",
        rate=lamb_OFF, product_distance=capture_radius, weight1=0.5, weight2=0.5,
    )
    # B -> C : immobilization, mobile dimer becomes immobile (rate kappa_IMMOBILITY).
    stem.reactions.add_conversion(
        name="B => C", type_from="B", type_to="C", rate=lamb_IMMOBILITY,
    )
    # C -> B : mobilization, immobile dimer becomes mobile again (rate kappa_MOBILITY).
    stem.reactions.add_conversion(
        name="C => B", type_from="C", type_to="B", rate=lamb_MOBILITY,
    )

    if verbose:
        rates_by_species = dict(zip(particle_species_names, diffusion_rates))
        print(f"  Diffusion rates per species (um^2/s): {rates_by_species}")
        print("  Reaction rates:")
        print(f"    A+A -> B (dimerization):     macroscopic={kappa_ON:.6g} um^3/s "
              f"(cap={kappa_ON_CAP:.6g}, R_ON={R_ON:.6g})")
        print(f"    B   -> A+A (dissociation):   macroscopic={kappa_OFF:.6g} 1/s")
        print(f"    B   -> C (immobilization):   macroscopic={kappa_IMMOBILITY:.6g} 1/s")
        print(f"    C   -> B (mobilization):     macroscopic={kappa_MOBILITY:.6g} 1/s")

    return stem


# =============================================================================
# Simulation builder
# =============================================================================

def build_simulation(stem: "readdy.ReactionDiffusionSystem",
                     theta: np.ndarray,
                     particle_species_names: Optional[Tuple[str, ...]] = None,
                     seed: Optional[int] = None,
                     skin_factor: Optional[float] = None,
                     verbose: bool = False) -> "readdy.Simulation":
    """Wrap the ReactionDiffusionSystem in a Simulation, register observables,
    and place initial particles uniformly in the box.

    Args:
        stem: ReactionDiffusionSystem from `build_system`.
        theta: Learnable-parameter values (same vector as passed to `build_system`).
            Used to extract per-species initial particle counts.
        particle_species_names: Optional override; defaults to
            `PARAMETERS.simulation.rds.particle_species_names`.
        seed: RNG seed for initial particle placement. If None, the placement
            is non-deterministic (different on every call); pass an integer
            for reproducible runs. Note: this seed only controls the NumPy
            RNG for particle positions; ReaDDy's internal RNG (used for
            reaction events and diffusion steps during `simulation.run`)
            has its own seeding mechanism.
        skin_factor: ReaDDy neighbor-list (Verlet) skin as a MULTIPLE of the
            particle diameter (skin = skin_factor * particle_diameter_nm, in nm).
            A PURE PERFORMANCE knob -- it coarsens the cell-linked-list grid in the
            large, dilute imaging box without changing the physics (reactions still
            fire only at the true reaction radius; the skin only widens which
            particles are considered as candidates). If None (default), the
            configured `PARAMETERS.simulation.rds.neighbor_list_skin_factor` is used.
            See that field for the full rationale and the U-shaped cost curve.
        verbose: If True, print the per-species and total initial particle counts.

    Returns:
        A `readdy.Simulation` with:
            - 'particles' observable at the per-frame stride
              (PARAMETERS.simulation.timing.steps_per_frame),
            - 'reaction_counts' observable at every step (stride=1),
            - 'reactions' observable at every step (stride=1): one record per
              reaction event with its educt and product particle ids, read back
              by `extract_subunit_lineage` (the DLI stage's subunit bookkeeping),
            - initial particles placed uniformly in the box for each species.

    Implementation notes:
        - Particle counts are rounded from the continuous theta sample
          (which lives in log-space and was exp-transformed before being
          passed here) to int32.
        - Particles are placed uniformly in [-box/2, +box/2] in each
          spatial dimension, ignoring the periodic z boundary for initial
          placement.
        - The simulation runs on the CPU kernel.
    """
    if particle_species_names is None:
        particle_species_names = PARAMETERS.simulation.rds.particle_species_names

    count_keys = ("count_alp", "count_bet", "count_chi")
    theta_counts = np.round(theta[[parameter_find(k) for k in count_keys]]).astype(np.int32)

    smut = stem.simulation(kernel="CPU")

    # Neighbor-list (Verlet) skin, as a MULTIPLE of the particle diameter. Pure
    # performance knob: it enlarges ReaDDy's cell-linked-list cells (cell edge =
    # reaction_radius + skin) so the huge, dilute imaging box is not partitioned into
    # ~16 million mostly-empty cells whose per-step management dominates the runtime.
    # It never changes the physics -- reactions still fire only at the true reaction
    # radius; the skin only widens which particles are considered as CANDIDATES. See
    # PARAMETERS.simulation.rds.neighbor_list_skin_factor for the rationale and the
    # U-shaped cost curve. None -> the configured default.
    if skin_factor is None:
        skin_factor = PARAMETERS.simulation.rds.neighbor_list_skin_factor
    if skin_factor < 0:
        raise ValueError(
            f"skin_factor must be non-negative (got {skin_factor}); it is a multiple "
            f"of the particle diameter and sets the neighbor-list skin distance.")
    smut.skin = (skin_factor * PARAMETERS.simulation.stem.particle_diameter_nm
                 * readdy.units.nanometer)

    smut.observe.particles(stride=PARAMETERS.simulation.timing.steps_per_frame)
    smut.observe.reaction_counts(stride=1)
    # Reaction RECORDS (educt ids -> product ids, per event). ReaDDy gives every reaction
    # product a fresh particle id -- fusion, fission, and plain conversion alike -- so
    # without these records the DLI stage could not follow a receptor subunit (and the
    # static dye count it carries) through the reactions. Sparse: one record per event.
    smut.observe.reactions(stride=1)

    box_size = PARAMETERS.simulation.stem.box_size
    lo = [-box_size[0] / 2, -box_size[1] / 2, -box_size[2] / 2]
    hi = [box_size[0] / 2, box_size[1] / 2, box_size[2] / 2]

    # Single RNG controls placement across all species (reproducible if seed is set).
    rng = np.random.default_rng(seed)
    for idx, species_name in enumerate(particle_species_names):
        n_particles = int(theta_counts[idx])
        positions = rng.uniform(low=lo, high=hi, size=(n_particles, 3))
        smut.add_particles(type=species_name, positions=positions)

    if verbose:
        counts_by_species = dict(zip(particle_species_names, theta_counts.tolist()))
        total = int(np.sum(theta_counts))
        print(f"  Initial particle counts: {counts_by_species} (total = {total})")

    return smut


# =============================================================================
# Trajectory extraction: particle poses, subunit lineage
# =============================================================================

def _soul_index(souls) -> Tuple[np.ndarray, dict]:
    """Particle-index order shared by every extractor in this module.

    ``souls`` is the per-frame list of ReaDDy particle-id arrays. The particle index of
    the pose tensor and of the lineage table is the position of the id in the sorted
    array of every id that ever appears, so the two extractors agree on it by
    construction.
    """
    all_souls = np.unique(
        np.concatenate([np.asarray(frame_souls).reshape(-1) for frame_souls in souls])
    )
    return all_souls, {int(soul): idx for idx, soul in enumerate(all_souls)}


def extract_trajectory_poses(tray, verbose: bool = False) -> np.ndarray:
    """Read a saved ReaDDy .h5 trajectory and produce a dense pose tensor.

    Args:
        tray: A `readdy.Trajectory` object opened from a .h5 file.
        verbose: If True, print per-reaction counts and the output tensor shape.

    Returns:
        tray_poses: shape `(n_frames, n_particles, 3, n_species)`.
            Each entry is the (x, y, z) coordinate of a given particle in
            a given frame for a given species rank. Entries are `np.nan`
            when the particle does not exist as that species in that frame.
            Coordinates are shifted from box-centered (ReaDDy default) to
            box-anchored (origin at the corner): each component has
            `box_size[dim] / 2` added so all coordinates are non-negative.

    Implementation notes:
        - The "particle" axis enumerates ReaDDy particle IDS in the order of
          `_soul_index` (sorted unique ids). ReaDDy gives every reaction product a
          new id, so one receptor subunit visits several particle indices over a
          recording; `extract_subunit_lineage` maps subunits onto these indices.
        - The "rank" axis of `tray_poses` is the species index assigned by
          ReaDDy (from `tray.particle_types`). For the standard species
          order ('A', 'B', 'C'), rank 0 is A, rank 1 is B, rank 2 is C.
        - The last frame in the ReaDDy trajectory is dropped (the
          `n_frames - 1` slice) because ReaDDy writes an extra final
          observable that doesn't correspond to a fully evolved frame.
    """
    if verbose:
        _, recs = tray.read_observable_reaction_counts()
        for rec_name, counts in recs["reactions"].items():
            print(f"    reaction {rec_name}: {np.count_nonzero(counts)} events")

    part_spans, ranks, souls, poses = tray.read_observable_particles()

    species_to_rank = tray.particle_types  # {species_name: rank_index}
    all_souls, soul_to_index = _soul_index(souls)

    poses_shape = (part_spans.shape[0] - 1, all_souls.shape[0], 3, len(species_to_rank))
    poses_centred = np.full(shape=poses_shape, fill_value=np.nan)

    # Drive the loop from the frames actually present in this trajectory
    # (poses_centred is sized to part_spans.shape[0] - 1). Using the data's own
    # length -- not the global PARAMETERS frame_count -- keeps extraction correct
    # for any simulation duration. The 2 s global default (100 frames) previously
    # truncated 5 s/10 s trajectories to their first 100 frames and overran sub-2 s ones.
    for span_index in range(poses_centred.shape[0]):
        frame_ranks = ranks[span_index]
        frame_souls = souls[span_index]
        frame_poses = poses[span_index]
        for part_index in range(frame_poses.shape[0]):
            rank = frame_ranks[part_index]
            soul_index = soul_to_index[int(frame_souls[part_index])]
            poses_centred[span_index, soul_index, :, rank] = frame_poses[part_index]

    # Sanity guard: every frame must hold at least one real position. An entirely
    # NaN frame means the fill loop skipped it (a trajectory/frame-count mismatch),
    # which would silently emit a noise-only video. Fail loudly rather than ship
    # truncated poses.
    empty_frames = np.isnan(poses_centred).all(axis=(1, 2, 3))
    if empty_frames.any():
        raise ValueError(
            f"extract_trajectory_poses: {int(empty_frames.sum())} of "
            f"{poses_centred.shape[0]} frames are entirely NaN (first at index "
            f"{int(np.argmax(empty_frames))}); the per-frame loop did not populate "
            f"every frame. Refusing to emit truncated poses."
        )

    # Shift coordinates from box-centered to box-anchored (origin at corner).
    box_size = PARAMETERS.simulation.stem.box_size
    half_box = np.array([dim / 2 for dim in box_size]).reshape(1, 1, 3, 1)
    tray_poses = half_box + poses_centred

    if verbose:
        print(f"  tray_poses shape: {tray_poses.shape}")

    return tray_poses


def collapse_species_axis(tray_poses: np.ndarray) -> np.ndarray:
    """Collapse the species-rank axis: `(n_frames, n_particles, 3, n_species)` ->
    `(n_frames, n_particles, 3)`.

    A particle is exactly one species per frame, so at most one rank holds a
    non-NaN coordinate and the max over the rank axis selects it. The result is
    NaN wherever the particle is absent from the frame. The all-NaN-slice
    RuntimeWarning that `nanmax` emits for absent particles is benign and
    suppressed here, once, instead of at every call site.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning,
        )
        return np.nanmax(tray_poses, axis=3)


class SubunitLineage(NamedTuple):
    """Which particle hosts each receptor subunit at each frame.

    The receptor SUBUNIT is the persistent physical object of the model: the reaction
    network (A + A <-> B, B <-> C) conserves the number of subunits, while ReaDDy
    assigns a NEW particle id to every reaction product -- including the product of a
    plain species conversion (B -> C keeps the molecule, not the id). Static
    per-subunit quantities of the DLI stage (dye counts, PSF widths) therefore attach
    to subunits, and this table maps them onto the particle that carries them in each
    frame. It is built by `extract_subunit_lineage` from the reaction records with the
    three bookkeeping rules of the model specification: fusion concatenates the two
    educts' subunits, fission hands each daughter its own subunit, conversion
    preserves the subunits.

    Attributes:
        host_index: int array `(n_frames, n_subunits)`; the particle index (the
            `extract_trajectory_poses` particle order) of the particle hosting the
            subunit at that frame. Every entry is valid: subunits are conserved.
        host_rank: int array `(n_frames, n_subunits)`; the species rank
            (`tray.particle_types` order) of that host, i.e. the subunit's
            molecular state at that frame.
        soul_ids: int array `(n_particles,)`; the ReaDDy particle ids in
            particle-index order (provenance only).
    """
    host_index: np.ndarray
    host_rank: np.ndarray
    soul_ids: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.host_index.shape[0])

    @property
    def n_subunits(self) -> int:
        return int(self.host_index.shape[1])


def _apply_reaction_record(record, subunits_of_soul: dict, counts: dict) -> None:
    """Apply one ReaDDy reaction record to the soul -> subunits map, in place.

    The three rules are the model specification's reaction bookkeeping for static
    per-subunit quantities. Fission assigns the dimer's first subunit to the first
    listed product and the second to the second: ReaDDy places the two daughters
    symmetrically about the educt, so the two are exchangeable and a deterministic
    assignment is statistically equivalent to a randomized one while keeping the
    lineage reproducible from the trajectory file alone.
    """
    kind = str(record.type)
    educts = [int(x) for x in np.atleast_1d(record.educts)]
    products = [int(x) for x in np.atleast_1d(record.products)]
    try:
        if kind == "fusion":
            if len(educts) != 2 or len(products) != 1:
                raise ValueError(f"fusion record with educts {educts} and products {products}.")
            subunits_of_soul[products[0]] = (subunits_of_soul.pop(educts[0])
                                             + subunits_of_soul.pop(educts[1]))
        elif kind == "fission":
            if len(educts) != 1 or len(products) != 2:
                raise ValueError(f"fission record with educts {educts} and products {products}.")
            subunits = subunits_of_soul.pop(educts[0])
            if len(subunits) != 2:
                raise ValueError(
                    f"fission of particle {educts[0]} carrying {len(subunits)} subunit(s); "
                    f"a dissociating particle must carry exactly two.")
            subunits_of_soul[products[0]] = (subunits[0],)
            subunits_of_soul[products[1]] = (subunits[1],)
        elif kind == "conversion":
            if len(educts) != 1 or len(products) != 1:
                raise ValueError(f"conversion record with educts {educts} and products {products}.")
            subunits_of_soul[products[0]] = subunits_of_soul.pop(educts[0])
        else:
            raise NotImplementedError(
                f"reaction record type {kind!r} ({record.reaction_label!r}) has no subunit "
                f"bookkeeping rule; the DIMER network has fusion, fission, and conversion only.")
    except KeyError as exc:
        raise ValueError(
            f"{kind} record {record.reaction_label!r}: educt particle {exc.args[0]} has no "
            f"lineage (a reaction record referenced a particle the replay never saw).") from exc
    counts[kind] = counts.get(kind, 0) + 1


def extract_subunit_lineage(tray, verbose: bool = False) -> SubunitLineage:
    """Replay the reaction records of a saved trajectory into a per-frame
    subunit -> host-particle table (see `SubunitLineage`).

    Args:
        tray: A `readdy.Trajectory` opened from a .h5 file written by a simulation
            that registered the 'reactions' observable (`build_simulation` does).
        verbose: If True, print the subunit count, the particle-id count, and the
            number of records replayed per reaction type.

    Returns:
        A `SubunitLineage` whose frame axis matches `extract_trajectory_poses`
        (the trailing ReaDDy observable is dropped the same way) and whose
        particle indices are the same particle order.

    Raises:
        ValueError: if the trajectory carries no reaction records (generated
            before the observable was registered), if a particle appears without a
            record explaining it, or if any frame does not cover every subunit
            exactly once (the conservation check that makes silent bookkeeping
            errors impossible).

    Implementation notes:
        - Frame `f` of the particles observable is the state at step
          `part_spans[f]`, evaluated after that step's reactions, so a record at
          step `t` is applied before assigning hosts for the first frame with
          `part_spans[f] >= t`. Several records inside one frame interval are
          replayed in file order; each is a single dictionary update.
        - Frame 0 seeds the lineage: every initial particle receives fresh
          subunit ids, one per subunit of its species
          (`PARAMETERS.simulation.rds.subunit_counts_per_species`).
    """
    part_spans, ranks, souls, _ = tray.read_observable_particles()
    n_frames = int(part_spans.shape[0]) - 1
    all_souls, soul_to_index = _soul_index(souls)

    rds = PARAMETERS.simulation.rds
    species_to_rank = tray.particle_types
    subunits_of_rank = {}
    for name, n_sub in zip(rds.particle_species_names, rds.subunit_counts_per_species):
        if name in species_to_rank:
            subunits_of_rank[int(species_to_rank[name])] = int(n_sub)

    try:
        record_times, record_lists = tray.read_observable_reactions()
    except Exception as exc:  # h5py/ReaDDy raise their own types when the group is absent
        raise ValueError(
            "extract_subunit_lineage: the trajectory carries no reaction records "
            "(the 'reactions' observable is absent). Trajectories must be generated by "
            "this version's RDS stage, whose build_simulation registers it; the "
            "DOL-explicit DLI stage cannot render older trajectories."
        ) from exc
    record_times = np.asarray(record_times).reshape(-1)

    # Seed: fresh subunit ids for the initial particles, per species.
    subunits_of_soul: dict = {}
    next_subunit = 0
    for soul, rank in zip(souls[0], ranks[0]):
        try:
            n_sub = subunits_of_rank[int(rank)]
        except KeyError as exc:
            raise ValueError(
                f"extract_subunit_lineage: species rank {int(rank)} has no subunit count "
                f"(configured species {rds.particle_species_names}).") from exc
        subunits_of_soul[int(soul)] = tuple(range(next_subunit, next_subunit + n_sub))
        next_subunit += n_sub
    n_subunits = next_subunit

    host_index = np.full((n_frames, n_subunits), -1, dtype=np.int64)
    host_rank = np.full((n_frames, n_subunits), -1, dtype=np.int64)
    counts: dict = {}
    record_ptr = 0
    for frame in range(n_frames):
        step = int(part_spans[frame])
        while record_ptr < record_times.shape[0] and int(record_times[record_ptr]) <= step:
            for record in record_lists[record_ptr]:
                _apply_reaction_record(record, subunits_of_soul, counts)
            record_ptr += 1
        row = host_index[frame]
        for soul, rank in zip(souls[frame], ranks[frame]):
            subunits = subunits_of_soul.get(int(soul))
            if subunits is None:
                raise ValueError(
                    f"extract_subunit_lineage: frame {frame} (step {step}) holds particle "
                    f"{int(soul)} with no lineage -- it appeared without a reaction record.")
            if len(subunits) != subunits_of_rank.get(int(rank), -1):
                raise ValueError(
                    f"extract_subunit_lineage: frame {frame} (step {step}) particle {int(soul)} "
                    f"of species rank {int(rank)} carries {len(subunits)} subunit(s); its species "
                    f"has {subunits_of_rank.get(int(rank))}. The reaction replay is inconsistent.")
            for subunit in subunits:
                if row[subunit] != -1:
                    raise ValueError(
                        f"extract_subunit_lineage: frame {frame} assigns subunit {subunit} "
                        f"to two particles; the reaction replay is inconsistent.")
                row[subunit] = soul_to_index[int(soul)]
                host_rank[frame, subunit] = int(rank)
        missing = int((row < 0).sum())
        if missing:
            raise ValueError(
                f"extract_subunit_lineage: frame {frame} leaves {missing} of {n_subunits} "
                f"subunit(s) without a host particle; subunit conservation is violated "
                f"(a reaction record is missing or misordered).")

    if verbose:
        replayed = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"
        print(f"  subunit lineage: {n_subunits} subunits over {all_souls.shape[0]} particle ids "
              f"in {n_frames} frames; records replayed: {replayed}")

    return SubunitLineage(host_index=host_index, host_rank=host_rank,
                          soul_ids=all_souls.astype(np.int64))
