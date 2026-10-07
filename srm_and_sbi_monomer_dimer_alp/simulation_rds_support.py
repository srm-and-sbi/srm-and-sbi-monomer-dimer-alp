"""ReaDDy primitives for the reaction-diffusion simulation (RDS) stage.

This module wraps the ReaDDy 2 particle-based reaction-diffusion solver
(Hoffmann et al., 2019, "ReaDDy 2: Fast and flexible software framework
for interacting particle reaction dynamics", PLoS Comp Bio.
https://doi.org/10.1371/journal.pcbi.1006830) for the separated
stoichiometry-mobility model of this sibling.

The simulated population is the RETAINED one: the probe-associated complexes (PROJECT_CONTEXT.md
sec. 2, *Modeling assumptions of the MET model*). Three mobility modes; the molecular species
depend on the condition's ligand classes (``PARAMETERS.simulation.rds.species_for``):
    MET-FAB  (no ligand classes)   A_f, A_s, A_i = monomer;  B_f, B_s, B_i = dimer (two retained
                                   subunits; its one-/two-Fab class is assigned at labeling)
    MET-INLB (ligand classes)      A_f, A_s, A_i = InlB-bound monomer;
                                   B1_f, B1_s, B1_i = one-InlB dimer (ONE retained subunit);
                                   B2_f, B2_s, B2_i = two-InlB dimer (two retained subunits)
so six particle types under MET-FAB and nine under MET-INLB
(``particle_types_for(condition)``; ``particle_types`` is the union of the model family).

The reaction channels are GENERATED from the two model blocks and the run's CONDITION by
``reaction_channels(theta, condition)``:
    association    A_m + A_m' -> B2_{slower(m, m')}  six fusions, one per unordered pair of
                                                    monomer modes, all at the same
                                                    lambda_on = R_ON[condition] * lambda_ref;
                                                    NONE when the condition's ratio is zero
    dissociation   B_m  -> A_m + A_m  (FAB)           three fissions at kappa_OFF (mode conserved)
                   B2_m -> A_m + A_m  (INLB)          three fissions at kappa_OFF
                   B1_m -> A_m       (INLB)          three CONVERSIONS at kappa_OFF: the probe-free
                                                    partner of a one-InlB dimer is not represented,
                                                    so no lineage entry is created or removed
    switching      X_f <-> X_s <-> X_i               the four shared rates, once per species
MET-INLB R_ON = 1 -> twenty-four channels (6 + 3 + 3 + 12); MET-FAB R_ON = 0 -> eleven (3 + 8;
no association, the retained basal dimers may dissociate and both daughters stay retained).
The association ratio is a declared per-condition constant (``parameterization.ConditionSetting``),
not a learnable row in either condition.

Diffusion: D[X, m] = D_A * (R_B if X is a dimer class else 1) * (1, R_s, R_i)[m].

Functions:
    reaction_channels(theta, condition)
        The declarative list of the condition's channels (kind, educts, products, rate)
        derived from the model blocks and the condition's association setting;
        ``build_system`` registers exactly these, and the structure audit checks them
        without ReaDDy.

    build_system(theta, condition, ...)
        Builds the ReaDDy ReactionDiffusionSystem: registers the condition's particle types
        with their diffusion constants and adds the condition's channels. Returns the
        configured system, ready to be wrapped in a Simulation.

    build_simulation(stem, theta, condition, ...)
        Wraps the system in a Simulation, registers observables (per-frame
        particle positions, per-step reaction counts, per-step reaction
        records), and places the initial RETAINED particles uniformly in the box: the
        integer composition realized from (N_total, r) through the condition's occupancy
        (``realize_initial_composition``) and each particle's mode from the stationary law of
        the switching chain. Returns the runnable Simulation.

    extract_trajectory_poses(tray, ...)
        Reads a saved .h5 trajectory file and produces a dense
        (frame, particle, spatial-dim, species-rank) tensor of particle
        positions.

    collapse_species_axis(tray_poses)
        Collapses the species-rank axis of that tensor to
        (frame, particle, spatial-dim).

    extract_subunit_positions(tray, lineage, ...)
        Reads the same trajectory and produces the (frame, subunit, spatial-dim)
        positions of each subunit's host particle directly, without the dense
        tensor, whose particle axis grows with every reaction of a long recording;
        the renderer's lean input, identical value for value to the gather of the
        dense tensor through the lineage.

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
    PARAMETERIZATION,
    PARAMETERS,
    parameter_find,
    realize_initial_composition,
)


# =============================================================================
# The reaction network, generated from the model blocks
# =============================================================================

class ReactionChannel(NamedTuple):
    """One channel of the generated network.

    ``kind`` is the ReaDDy primitive (``fusion`` / ``fission`` / ``conversion``); ``educts``
    and ``products`` are particle-type names; ``rate`` is the rate handed to ReaDDy in 1/s
    (the microscopic association rate for fusions); ``rate_key`` names the parameter it came
    from. ``name`` is the ReaDDy reaction label (also the key of the reaction-count records).
    """
    kind: str
    name: str
    educts: Tuple[str, ...]
    products: Tuple[str, ...]
    rate: float
    rate_key: str


def theta_by_key(theta: np.ndarray) -> dict:
    """Physical parameter values keyed by parameter KEY (``theta`` in canonical order)."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(PARAMETERIZATION),):
        raise ValueError(f"theta has shape {theta.shape}; expected ({len(PARAMETERIZATION)},).")
    return {para["KEY"]: float(theta[i]) for i, para in enumerate(PARAMETERIZATION)}


def diffusion_coefficients(theta: np.ndarray) -> dict:
    """Diffusion coefficient of every particle type, um^2/s: D[X, m] = D_A * species * mode factor."""
    rds = PARAMETERS.simulation.rds
    values = theta_by_key(theta)
    d_a = values[rds.mobility.diffusivity_key]
    mode_factor = {mode: (1.0 if key is None else values[key])
                   for mode, key in zip(rds.mobility.modes, rds.mobility.mode_ratio_keys)}
    # the dimer factor applies to every dimer class (B, B1, B2): pt.species is the stoichiometric class
    species_factor = {rds.stoichiometry.monomer.name: 1.0,
                      rds.stoichiometry.dimer.name: values[rds.mobility.dimer_ratio_key]}
    return {pt.name: d_a * species_factor[pt.species] * mode_factor[pt.mode]
            for pt in rds.particle_types}


def association_reference_rate(diffusivity_um2_s: float, reaction_distance_nm: float) -> float:
    """The compatibility normalization lambda_ref = 6 D_A / r^2, in 1/s.

    Equal to the Smoluchowski encounter rate of two monomers, 4 pi (2 D_A) r, divided by the
    reaction volume (4/3) pi r^3. It is a declared reference with units of inverse time that
    depends on D_A and on the reaction distance; it is NOT a physical upper bound on the
    association rate (the diffusion-limited regime is the large-intensity limit of the spatial
    rule), so R_ON = lambda_on / lambda_ref is a dimensionless parameterization convenience.
    """
    r_um = float(reaction_distance_nm) * 1e-3
    return 6.0 * float(diffusivity_um2_s) / (r_um * r_um)


def reaction_channels(theta: np.ndarray, condition: str) -> Tuple[ReactionChannel, ...]:
    """The channels of the network for one theta under one condition, derived from the blocks.

    Order: the association fusions (unordered pairs of monomer modes, fastest first; NONE when
    the condition's association ratio is zero; the product is the two-probe dimer class of the
    condition), the dissociations (one per dimer class and mode: a fission into two monomers
    for ``B`` and ``B2``, a conversion into one monomer for ``B1``), then the switching
    conversions (per species of the condition, in the block's switching order). Every channel
    is unique by construction; the structure audit asserts the count per condition (MET-INLB
    24, MET-FAB 11), the products, and the inherited modes. The association ratio is the
    condition's declared constant, never a theta entry.
    """
    rds = PARAMETERS.simulation.rds
    sto, mob = rds.stoichiometry, rds.mobility
    values = theta_by_key(theta)
    species = rds.species_for(condition)
    dimers = [s for s in species if s.stoichiometry == "dimer"]
    mono = sto.monomer.name
    # association produces the dimer class whose subunits are all ligand-bound (B2), or B under FAB
    product_dimer = next(s for s in dimers if s.dissociation == "fission").name
    d_a = values[mob.diffusivity_key]
    reaction_distance_nm = PARAMETERS.simulation.stem.particle_diameter_nm
    r_on = rds.association_ratio_of(condition)
    lamb_on = r_on * association_reference_rate(d_a, reaction_distance_nm)
    kappa_off = values[sto.dissociation_rate_key]

    channels = []
    if r_on > 0.0:                      # exactly zero declares NO association channel (structural)
        for i, m1 in enumerate(mob.modes):
            for m2 in mob.modes[i:]:
                product_mode = mob.inherited_mode(m1, m2)
                e1, e2 = rds.type_name(mono, m1), rds.type_name(mono, m2)
                pr = rds.type_name(product_dimer, product_mode)
                channels.append(ReactionChannel(
                    "fusion", f"{e1} + {e2} => {pr}", (e1, e2), (pr,), lamb_on, f"R_ON[{condition}]"))
    for dimer in dimers:
        for m in mob.modes:
            ed, pr = rds.type_name(dimer.name, m), rds.type_name(mono, m)
            if dimer.dissociation == "fission":
                channels.append(ReactionChannel(
                    "fission", f"{ed} => {pr} + {pr}", (ed,), (pr, pr), kappa_off, sto.dissociation_rate_key))
            else:                        # B1 -> A: one eligible daughter; the probe-free partner is not represented
                channels.append(ReactionChannel(
                    "conversion", f"{ed} => {pr}", (ed,), (pr,), kappa_off, sto.dissociation_rate_key))
    for sp in species:
        for frm, to, key in mob.switching:
            ed, pr = rds.type_name(sp.name, frm), rds.type_name(sp.name, to)
            channels.append(ReactionChannel("conversion", f"{ed} => {pr}", (ed,), (pr,), values[key], key))
    return tuple(channels)


def stationary_mode_law(theta: np.ndarray) -> np.ndarray:
    """Stationary occupancies of the isolated switching chain, one probability per mode.

    For the sequential chain the detailed-balance ratio pi[m+1] / pi[m] equals the forward
    over the backward rate of the link between them, so pi_f : pi_s : pi_i =
    1 : k_fs/k_sf : (k_fs/k_sf)(k_si/k_is). This is the initial-mode law for every particle;
    it is NOT the steady state of the reactive system (reactions with inheritance disturb it).
    """
    mob = PARAMETERS.simulation.rds.mobility
    values = theta_by_key(theta)
    forward = {(frm, to): values[key] for frm, to, key in mob.switching}
    weights = [1.0]
    for i in range(len(mob.modes) - 1):
        a, b = mob.modes[i], mob.modes[i + 1]
        weights.append(weights[-1] * forward[(a, b)] / forward[(b, a)])
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()


def build_system(theta: np.ndarray,
                 condition: str,
                 verbose: bool = False) -> "readdy.ReactionDiffusionSystem":
    """Build a ReaDDy ReactionDiffusionSystem for the separated stoichiometry-mobility model.

    Args:
        theta: Learnable-parameter values in PHYSICAL units (``parameterization.to_physical``
            of an estimator-space sample), 1D of length ``len(PARAMETERIZATION)`` in the
            canonical order.
        condition: The stored condition token (``FAB`` or ``INLB``); selects the declared
            association ratio and therefore which channels exist (``reaction_channels``).
        verbose: If True, print the per-type diffusion constants and every channel's rate.

    Returns:
        A configured ``readdy.ReactionDiffusionSystem`` ready for ``build_simulation``.

    The particle types, their diffusion constants, and the condition's channels come from
    ``PARAMETERS.simulation.rds`` through ``diffusion_coefficients`` and
    ``reaction_channels`` (module docstring). Association fires within the reaction distance
    (one particle diameter, the Smoluchowski contact distance, derived from
    ``SimulationStem.particle_diameter_nm``); fission products are placed OUTSIDE that distance,
    at ``SimulationStem.fission_product_distance_nm`` (2 x the reaction distance), so that a
    freshly dissociated pair is not re-fused deterministically at the next sub-step (see the
    field's comment in ``parameterization.py``).

    Boundary conditions: ``[False, False, True]`` -- open in x and y (the observation plane)
    and periodic in z (the thin membrane normal). No potential confines the particles, so a
    receptor that diffuses beyond the imaged field stays simulated, keeps reacting and
    switching, and may return; it is simply not rendered while outside. The conserved
    receptor total N_R therefore refers to the simulated patch, and the in-field count is a
    distinct, time-dependent quantity.
    """
    rds = PARAMETERS.simulation.rds
    stem_geometry = PARAMETERS.simulation.stem
    reaction_distance_nm = stem_geometry.particle_diameter_nm
    product_distance_nm = stem_geometry.fission_product_distance_nm
    coefficients = diffusion_coefficients(theta)
    channels = reaction_channels(theta, condition)

    stem = readdy.ReactionDiffusionSystem(
        box_size=stem_geometry.box_size,
        unit_system=stem_geometry.unit_dict,
    )
    stem.periodic_boundary_conditions = [False, False, True]

    for type_name in rds.particle_type_names_for(condition):
        stem.add_species(
            name=type_name,
            diffusion_constant=coefficients[type_name] * pow(readdy.units.micrometer, 2) / readdy.units.second,
        )

    for ch in channels:
        rate = ch.rate / readdy.units.second
        if ch.kind == "fusion":
            stem.reactions.add_fusion(
                name=ch.name, type_from1=ch.educts[0], type_from2=ch.educts[1], type_to=ch.products[0],
                rate=rate, educt_distance=reaction_distance_nm, weight1=0.5, weight2=0.5)
        elif ch.kind == "fission":
            stem.reactions.add_fission(
                name=ch.name, type_from=ch.educts[0], type_to1=ch.products[0], type_to2=ch.products[1],
                rate=rate, product_distance=product_distance_nm, weight1=0.5, weight2=0.5)
        elif ch.kind == "conversion":
            stem.reactions.add_conversion(
                name=ch.name, type_from=ch.educts[0], type_to=ch.products[0], rate=rate)
        else:  # pragma: no cover -- reaction_channels emits the three kinds only
            raise ValueError(f"unknown channel kind {ch.kind!r}")

    if verbose:
        values = theta_by_key(theta)
        d_a = values[rds.mobility.diffusivity_key]
        print("  Diffusion coefficients per particle type (um^2/s): "
              + ", ".join(f"{k}={v:.4g}" for k, v in coefficients.items()))
        r_on = rds.association_ratio_of(condition)
        print(f"  Association reference lambda_ref = 6 D_A / r^2 = "
              f"{association_reference_rate(d_a, reaction_distance_nm):.6g} 1/s "
              f"(compatibility normalization, not a bound); condition {condition}: R_ON = {r_on:.6g}"
              + (" -> no association channel" if r_on == 0.0 else f" -> lambda_on = {r_on * association_reference_rate(d_a, reaction_distance_nm):.6g} 1/s"))
        print(f"  Reaction channels ({len(channels)}):")
        for ch in channels:
            print(f"    {ch.kind:<10} {ch.name:<22} rate={ch.rate:.6g} 1/s   [{ch.rate_key}]")

    return stem


# =============================================================================
# Simulation builder
# =============================================================================

def build_simulation(stem: "readdy.ReactionDiffusionSystem",
                     theta: np.ndarray,
                     condition: str,
                     seed: Optional[int] = None,
                     skin_factor: Optional[float] = None,
                     verbose: bool = False) -> "readdy.Simulation":
    """Wrap the ReactionDiffusionSystem in a Simulation, register observables, and place
    the initial RETAINED particles.

    Args:
        stem: ReactionDiffusionSystem from ``build_system``.
        theta: The same physical parameter vector passed to ``build_system``; supplies the
            true receptor total N_total, the requested initial dimer-to-monomer ratio r, and the
            switching rates whose stationary law draws each particle's initial mode.
        condition: The same condition token passed to ``build_system``; its occupancy realizes
            the retained population from (N_total, r) and its ligand classes decide which dimer
            species are placed (``parameterization.realize_initial_composition``).
        seed: RNG seed for the initial composition's mode draw and the placement. None ->
            non-deterministic. Note: this seed only controls the NumPy RNG; ReaDDy's own
            RNG for reactions and diffusion has its own mechanism.
        skin_factor: ReaDDy neighbor-list (Verlet) skin as a MULTIPLE of the particle
            diameter -- a PURE PERFORMANCE knob (see ``SimulationRDS.neighbor_list_skin_factor``).
            None -> the configured default.
        verbose: If True, print the realized composition and the per-type initial counts.

    Returns:
        A ``readdy.Simulation`` with:
            - 'particles' observable at the per-frame stride,
            - 'reaction_counts' observable at every step (stride=1),
            - 'reactions' observable at every step (stride=1): one record per event with
              its educt and product particle ids, read back by ``extract_subunit_lineage``,
            - the initial particles: ``realize_initial_composition(N_total, r, condition)`` gives
              the integer retained monomer and dimer counts, the dimers split into their probe
              classes (B1 / B2 under INLB; one species B under FAB, whose classes are assigned
              at labeling); each particle's mode is drawn from ``stationary_mode_law``, positions
              are uniform in the box.
    """
    rds = PARAMETERS.simulation.rds
    values = theta_by_key(theta)
    composition = realize_initial_composition(
        values[rds.stoichiometry.count_total_key], values[rds.stoichiometry.composition_ratio_key], condition)
    mode_law = stationary_mode_law(theta)

    smut = stem.simulation(kernel="CPU")

    # Neighbor-list (Verlet) skin, as a MULTIPLE of the particle diameter. Pure performance
    # knob: it enlarges ReaDDy's cell-linked-list cells (cell edge = reaction_radius + skin) so
    # the huge, dilute imaging box is not partitioned into ~16 million mostly-empty cells whose
    # per-step management dominates the runtime. It never changes the physics -- reactions
    # still fire only at the true reaction radius; the skin only widens which particles are
    # considered as CANDIDATES. See PARAMETERS.simulation.rds.neighbor_list_skin_factor.
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

    rng = np.random.default_rng(seed)
    initial_counts = {}
    for species, n_particles in initial_species_counts(composition):
        modes = rng.choice(len(rds.mobility.modes), size=n_particles, p=mode_law)
        for mode_index, mode in enumerate(rds.mobility.modes):
            n_type = int(np.sum(modes == mode_index))
            type_name = rds.type_name(species, mode)
            initial_counts[type_name] = n_type
            if n_type:
                positions = rng.uniform(low=lo, high=hi, size=(n_type, 3))
                smut.add_particles(type=type_name, positions=positions)

    if verbose:
        classes = (f" (B1 {composition.n_dimers_one_probe} + B2 {composition.n_dimers_two_probe})"
                   if composition.ligand_classes else
                   f" (expected two-Fab share {composition.two_probe_share:.4f}, assigned at labeling)")
        print(f"  Initial retained composition [{condition}]: N_total={composition.n_total_true:.0f} true subunits "
              f"at occupancy {composition.occupancy:.4f} -> N_R={composition.n_subunits} retained subunits: "
              f"{composition.n_monomers} monomers + {composition.n_dimers} dimers{classes}; "
              f"r requested {composition.ratio_requested:.4g} (true f_B {composition.complex_fraction_true:.4f}), "
              f"retained ratio expected {composition.ratio_retained_expected:.4g}, realized "
              f"{composition.ratio_realized:.4g} (retained f_B {composition.complex_fraction_retained:.4f}, "
              f"x_B {composition.receptor_fraction_retained:.4f})")
        print("  Stationary mode law (f, s, i): " + ", ".join(f"{p:.4f}" for p in mode_law))
        print(f"  Initial particle counts per type: {initial_counts}")

    return smut


def initial_species_counts(composition) -> Tuple[Tuple[str, int], ...]:
    """The (molecular species, particle count) pairs a composition places: (A, B) under FAB, where
    the one-/two-probe classes are assigned at labeling, and (A, B1, B2) under INLB."""
    sto = PARAMETERS.simulation.rds.stoichiometry
    if composition.ligand_classes:
        return ((sto.monomer.name, composition.n_monomers),
                (sto.dimer_one_probe.name, composition.n_dimers_one_probe),
                (sto.dimer_two_probe.name, composition.n_dimers_two_probe))
    return ((sto.monomer.name, composition.n_monomers), (sto.dimer.name, composition.n_dimers))


def initial_composition_of(theta: np.ndarray, condition: str):
    """The ``InitialComposition`` a simulation of ``theta`` (physical) under ``condition`` is seeded with."""
    rds = PARAMETERS.simulation.rds
    values = theta_by_key(theta)
    return realize_initial_composition(
        values[rds.stoichiometry.count_total_key], values[rds.stoichiometry.composition_ratio_key], condition)


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
        - The "rank" axis of `tray_poses` is the PARTICLE-TYPE index assigned by
          ReaDDy (from `tray.particle_types`, one rank per (species, mode) type);
          `rank_to_species` maps it back to the molecular species.
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


def rank_to_species(tray) -> dict:
    """Particle-type rank (``tray.particle_types`` order) -> STOICHIOMETRIC class (``A`` or ``B``).

    The observation layer selects and counts by stoichiometric class (monomer versus dimer),
    never by mobility mode and never by ligand class, so every consumer of ``host_rank`` maps
    through this table rather than reading the type name; ``rank_to_molecule`` gives the
    ligand class where a consumer needs it.
    """
    species_of_type = PARAMETERS.simulation.rds.species_of_type
    out = {}
    for name, rank in tray.particle_types.items():
        if name not in species_of_type:
            raise ValueError(f"trajectory particle type {name!r} is not a configured particle type "
                             f"{tuple(species_of_type)}; the trajectory was generated by another model.")
        out[int(rank)] = species_of_type[name]
    return out


def rank_to_molecule(tray) -> dict:
    """Particle-type rank -> molecular species token (``A``, ``B``, ``B1`` or ``B2``)."""
    molecule_of_type = PARAMETERS.simulation.rds.molecule_of_type
    out = {}
    for name, rank in tray.particle_types.items():
        if name not in molecule_of_type:
            raise ValueError(f"trajectory particle type {name!r} is not a configured particle type "
                             f"{tuple(molecule_of_type)}; the trajectory was generated by another model.")
        out[int(rank)] = molecule_of_type[name]
    return out


def monomer_ranks(tray) -> list:
    """Particle-type ranks of the monomer types (stoichiometric class ``A``, every mode)."""
    monomer_types = set(PARAMETERS.simulation.rds.monomer_type_names)
    return sorted(int(rank) for name, rank in tray.particle_types.items() if name in monomer_types)


def collapse_species_axis(tray_poses: np.ndarray) -> np.ndarray:
    """Collapse the particle-type rank axis: `(n_frames, n_particles, 3, n_types)` ->
    `(n_frames, n_particles, 3)`.

    A particle is exactly one type per frame, so at most one rank holds a
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
    network (association, dissociation, mobility switching) conserves the number of
    subunits, while ReaDDy assigns a NEW particle id to every reaction product -- including
    the product of a plain type conversion (a mobility switch keeps the molecule, not the id). Static
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
        host_rank: int array `(n_frames, n_subunits)`; the PARTICLE-TYPE rank
            (`tray.particle_types` order) of that host, i.e. the subunit's molecular
            species AND mobility mode at that frame; map ranks to molecular species with
            `rank_to_species`.
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
                f"bookkeeping rule; the network has fusion, fission, and conversion only.")
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
          subunit ids, one per subunit of its particle type
          (`PARAMETERS.simulation.rds.subunit_counts_per_type`; a monomer type
          carries one subunit, a dimer type two, whatever its mobility mode).
    """
    part_spans, ranks, souls, _ = tray.read_observable_particles()
    n_frames = int(part_spans.shape[0]) - 1
    all_souls, soul_to_index = _soul_index(souls)

    rds = PARAMETERS.simulation.rds
    species_to_rank = tray.particle_types
    subunits_of_rank = {}
    for name, n_sub in zip(rds.particle_type_names, rds.subunit_counts_per_type):
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
                f"(configured particle types {rds.particle_type_names}).") from exc
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


def extract_subunit_positions(tray, lineage: SubunitLineage, verbose: bool = False) -> np.ndarray:
    """Read a saved ReaDDy .h5 trajectory and produce, per frame, the position of each receptor
    SUBUNIT: the coordinates of the particle hosting it, ``(n_frames, n_subunits, 3)`` in nm,
    box-anchored (origin at the corner) like `extract_trajectory_poses`.

    The lean counterpart of `extract_trajectory_poses` followed by `collapse_species_axis` for the
    renderer, which only ever gathers each dye's host coordinates per subunit
    (`simulation_dli_support.build_dye_tracks`). It never allocates the dense
    (frame, particle id, spatial-dim, rank) tensor, whose particle axis holds every id that ever
    appears -- ReaDDy gives each reaction product and each mode switch a fresh id, so that axis
    grows with duration, receptor count and reaction rates, quadratically in the duration --
    and its output scales with frames x subunits instead. The trajectory's own per-frame
    observables, id lookup and reaction records are read as before.

    The values are identical to the gather
    ``collapse_species_axis(extract_trajectory_poses(tray))[f, lineage.host_index[f]]``: the same
    per-frame coordinates, converted to float64 and shifted by half the box in the same order of
    operations, and the collapse of the rank axis selects the one non-NaN entry a present particle
    has. The trailing ReaDDy observable is dropped exactly as the dense reader drops it. The lineage
    is checked for consistency with the trajectory, frame count and hosts: its frame count must equal
    the trajectory's and every frame must hold the host particle of every subunit, or the read stops.
    That is not a proof of origin: a lineage of another trajectory with the same frame count and
    reused ids passes it, so pass the lineage extracted from this very trajectory, as the runner does.

    Args:
        tray: A `readdy.Trajectory` object opened from a .h5 file.
        lineage: its `SubunitLineage`.
        verbose: If True, print the output shape.

    Returns:
        subunit_positions: float64 array ``(n_frames, n_subunits, 3)``; every entry is a real
            coordinate, since every subunit has a host in every frame.
    """
    part_spans, _ranks, souls, poses = tray.read_observable_particles()
    n_frames = int(part_spans.shape[0]) - 1          # the trailing observable, dropped as the dense reader drops it
    if lineage.n_frames != n_frames:
        raise ValueError(
            f"extract_subunit_positions: the lineage holds {lineage.n_frames} frames but the "
            f"trajectory {n_frames}; the lineage must come from this trajectory.")
    box_size = PARAMETERS.simulation.stem.box_size
    half_box = np.array([dim / 2 for dim in box_size]).reshape(1, 3)
    soul_ids = np.asarray(lineage.soul_ids, dtype=np.int64)
    out = np.empty((n_frames, lineage.n_subunits, 3), dtype=float)
    for frame in range(n_frames):
        frame_souls = np.asarray(souls[frame]).reshape(-1).astype(np.int64)
        frame_poses = np.asarray(poses[frame], dtype=float).reshape(frame_souls.shape[0], 3)
        wanted = soul_ids[lineage.host_index[frame]]                # the ReaDDy id hosting each subunit
        order = np.argsort(frame_souls, kind="stable")
        found = np.searchsorted(frame_souls, wanted, sorter=order)
        inside = found < frame_souls.shape[0]
        if not inside.all() or not np.array_equal(frame_souls[order[found[inside]]], wanted[inside]):
            raise ValueError(
                f"extract_subunit_positions: frame {frame} does not hold the host particle of every "
                f"subunit; the lineage does not belong to this trajectory.")
        out[frame] = half_box + frame_poses[order[found]]
    if verbose:
        print(f"  subunit_positions shape: {out.shape}")
    return out
