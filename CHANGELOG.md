# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.2 - 2026-09-09

The separated stoichiometry-mobility model, its twelve learnable parameters, and the one
parameter-conversion rule.

### Added

- **Two declarative model blocks** on `PARAMETERS.simulation.rds` (`parameterization.py`):
  `StoichiometryBlock` (molecular species A monomer with one subunit and B dimer with two; the
  association and dissociation channels and their parameter keys) and `MobilityBlock` (the
  mobility modes f fast / s slow / i immobile, their diffusion-ratio keys, the sequential
  switching chain f <-> s <-> i with its four rate keys, the slower-parent inheritance rule).
  `SimulationRDS` derives the PARTICLE TYPES = species x mode (`A_f`, `A_s`, `A_i`, `B_f`,
  `B_s`, `B_i`), their subunit counts, and the maps type -> species / mode; nothing downstream
  lists them by hand. Import-time validation (`_validate_model_blocks`) checks that every block
  key is a learnable row, that `R_i < R_s <= 1` holds by the DISJOINT declared ranges, that
  `0 < R_B <= 1`, and that the initial dimer fraction is a linear row on `[0, 1]`.
- **Generated reaction network** (`simulation_rds_support.reaction_channels`): seventeen
  channels derived from the two blocks and registered verbatim by `build_system` -- six
  association fusions `A_m + A_m' -> B_slower(m, m')` (one per unordered pair of monomer modes,
  all at the same microscopic rate), three dissociation fissions `B_m -> A_m + A_m` at
  `kappa_OFF` (mode conserved), eight switching conversions (the four rates shared by both
  species; no direct f <-> i). Companion helpers: `diffusion_coefficients`
  (`D[X, m] = D_A x (R_B if X is the dimer else 1) x (1, R_s, R_i)[m]`),
  `association_reference_rate` (the compatibility normalization
  `lambda_ref = 6 D_A / r^2`, `r` = one particle diameter = 10 nm, explicitly NOT a physical
  bound; `lambda_on = R_ON x lambda_ref`), `stationary_mode_law` (the initial-mode law
  `pi_f : pi_s : pi_i = 1 : k_fs/k_sf : (k_fs/k_sf)(k_si/k_is)`), `rank_to_species` and
  `monomer_ranks` (particle-type rank -> molecular species, for the visibility layer).
- **Initial composition** (`parameterization.realize_initial_composition(N_R, x_B)`):
  `n_total = max(1, round(N_R))`, `n_B = min(round(n_total x_B / 2), floor(n_total / 2))`,
  `n_A = n_total - 2 n_B`; the requested and the realized `x_B` are distinguished and the
  realized composition is recorded in the `Labeling_Set`; each particle's initial mode is drawn
  from the stationary law of the isolated switching chain.
- **The one parameter-conversion rule** (`parameterization.to_physical` / `to_flow`, with
  `entry_to_physical` / `entry_to_flow` / `prior_center` per row): every ranged row declares its
  scale through `LOG_FLAG` (True = the estimator coordinate is log10 of the physical value;
  False = linear), and these are the only sanctioned conversions between the estimator's space
  and physical values. The RDS sampling, the training dataset (`inference_support.VideoDataset`
  takes the workflow's table), evaluation, calibration, the diagnostics tables, and the analyses
  all use them; no consumer writes `10**theta` by hand. `Theta_Set` files still store PHYSICAL
  values.
- **Structure audit** (`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.py`
  with its companion `.md`): a deterministic tier (D1-D6) checking the seventeen channels, the
  transform round trips, the initial compositions at `x_B = 0` and `1` with odd totals, and
  occupancy by species -- PASSED 6/6 on 2026-09-09 (profile `mars_pc`); and, behind the
  explicit `--run` flag, small-scale run checks -- subunit conservation, stationary mode
  occupancies, visible fractions, both-condition generation, and a lateral-boundary check --
  not yet executed, awaiting approval. Dry runs of the RDS, biology DLI, detector DLI,
  Inference, Evaluation, and Experiment entry points pass on the twelve-parameter table.

### Changed

- **Fission daughters are placed at twice the reaction distance** (20 nm,
  `SimulationStem.fission_product_distance_nm`, factor field
  `fission_product_distance_factor = 2`), outside the fusion radius, instead of at the
  reaction distance. At the 2 ms sub-step an eligible pair reacts with probability ~1, so
  daughters placed at the radius re-fused deterministically unless they diffused apart within
  one step (~50% per step for immobile daughters), which made the effective unbinding rate
  mode dependent. Declared convention; documented in PROJECT_CONTEXT.md and checked by the
  structure audit (D7).
- **The reaction-diffusion model** is the separated stoichiometry-mobility model: two molecular
  species x three mobility modes = six particle types, replacing the three-species
  monomer / mobile-dimer / immobile-dimer system. Immobility is a mode available to monomers
  and dimers alike, so the biology and the detector read the same model through the same
  blocks. Boundary behavior is unchanged and now stated: open laterally (x, y), periodic
  axially (z), no confining potential -- a receptor beyond the imaged field stays simulated,
  may return, and is not rendered while outside; the conserved total `N_R` is the simulated
  patch's total, and the in-field count is a distinct, time-dependent quantity.
- **Twelve learnable parameters replace the ten**, in the table groups `stoichiometry` and
  `mobility`: `count_total` (`N_R`, log10 [0.5, 3.0]), `fraction_dimer_initial` (`x_B`, LINEAR
  on [0, 1]), `relative_rate_dimerization` (`R_ON`, log10 [-2, 0]), `rate_dissociation`
  (`kappa_OFF`, log10 [-1, 1]), `diffusivity_alp` (`D_A`, log10 [-1.25, -0.25]),
  `relative_diffusivity_dimer` (`R_B`, log10 [-1, 0]), `relative_diffusivity_slow` (`R_s`,
  log10 [-1, 0]), `relative_diffusivity_immobile` (`R_i`, log10 [-3.0, -1.3]),
  `rate_fast_slow`, `rate_slow_fast`, `rate_slow_immobile`, `rate_immobile_slow` (each log10
  [-1, 1]). ALL ranges are DEVELOPMENT SETTINGS for building and checking the generator, not
  scientifically approved training priors.
- **The detector table's RDS nuisance rows** (`detector_parameterization.py`) are the same
  twelve keys (nuisance-from-object, supplied by the shared trajectory tier), grouped
  `stoichiometry` / `mobility`.
- **The visibility layer** acts by MOLECULAR species: occupancy and labeling select through
  `rank_to_species`, the `--occupancy` per-species syntax is `A=1.0,B=0.8`, and the lineage
  extractor reads subunit counts per particle type.
- **Old artifacts are rejected, never overwritten.** Trajectories and `Theta_Set`s generated by
  the 0.1.1 three-species model are rejected at the DLI stage by `rank_to_species` (unknown
  particle types `A`/`B`/`C`) and by the estimator schema guard
  (`artifacts.assert_schema_compatible` / the theta-width guard); estimators carrying the
  ten-parameter schema fail the same guard.
- **Documentation**: `PROJECT_CONTEXT.md` sec. 2 describes the model (species, modes, particle
  types, the seventeen channels, the diffusion parameterization, the association
  normalization, the boundary behavior, the twelve-parameter table with its development
  ranges, the initial state, the conversion rule); `DETECTOR_WORKFLOW.md` sec. 6.1 tabulates
  the twelve rows the detector marginalizes; `CLAUDE.md`, `README.md`, the Prime entry-point
  docstrings, and the analysis companion notes follow. Companion notes that record results
  computed under the 0.1.1 model keep those results under a dated note stating that the
  readout definitions changed (composition from `N_R` and `x_B`; `f_B = x_B / (2 - x_B)`;
  `f_R = x_B`).

### Removed

- The immobile-dimer species C and its parameters (`count_alp`, `count_bet`, `count_chi`,
  `relative_diffusivity_bet`, `relative_diffusivity_chi`, `rate_immobility`, `rate_mobility`),
  the hand-written four-channel network (`A + A <-> B`, `B <-> C`), and the per-species initial
  counts as learnable parameters.
- The blanket `10**theta` convention: `LOG_FLAG` is no longer a documentation field, and no
  consumer exponentiates or log-transforms a theta vector outside `to_physical` / `to_flow`.
- The deferred `C -> 0` internalization item of the scope statement, restated as receptor
  synthesis, degradation, and internalization (deferred; `N_R` conserved within a recording).

## 0.1.1 - 2026-09-07

The DOL-explicit observation layer, the reactive detector, and the condition slot.

### Added

- **Static labeling stoichiometry** (`labeling.py`): every receptor subunit draws an integer dye
  count once per recording from the condition's measured law -- MET-INLB `Bernoulli(q = 0.5)`
  (one engineered attachment site; the labeling probability measured for this preparation),
  MET-FAB `Poisson(1.64)` (the measured ensemble mean; matched-mean binomial and
  negative-binomial alternatives registered for sensitivity runs) -- composed with an optional
  static probe-occupancy multiplier (`--occupancy`, default 1, optionally per initial species).
  The laws are fixed and never inferred. A `Labeling_Set` record per task (one row per
  simulation: true and visible initial composition) is written beside the theta sets.
- **Subunit lineage** (`simulation_rds_support.extract_subunit_lineage`): the RDS stage registers
  ReaDDy's reaction-record observable and the DLI stage replays the records into a per-frame
  subunit-to-particle table -- fusion concatenates, fission distributes, conversion preserves --
  with a conservation check that fails loud. ReaDDy mints a new particle id at every reaction,
  plain species conversions included, so the lineage is what lets a static per-subunit quantity
  follow its subunit. `collapse_species_axis` replaces the repeated species-axis collapse.
- **Dye-centric renderer**: `render_dli_video(soul_poses, host_index, dye_counts, imaging)`
  renders every dye as an emitter at its subunit's host particle (`build_dye_tracks`), with one
  PSF width per subunit and one stationary OU brightness and bleaching process per dye. A
  subunit without a dye never renders; a dimer's dyes render at one position, so photons add
  with no multiplier. The renderer no longer restarts a dimer's brightness, PSF width, and
  bleach clock at every mobility conversion (the previous per-particle emitters did, once per
  new ReaDDy id).
- **The condition slot** of the runtime grammar: `Paths.with_condition` appends the condition
  token (`FAB`, `INLB`) to the alias -- `..._ALP_FAB_2S_50FPS_Video_Set_...`,
  `..._ALP_DETECTOR_INLB_2S_50FPS_Estimator.npz` -- while `rds_alias` / `theta_set_alias` keep
  the RDS products (the trajectories and their ten-parameter `Theta_Set`) under the bare alias;
  `Paths.record_set_path` builds the DLI-stage record sets. Every stage past RDS and every
  analysis take a required `--condition`; the HPC stage scripts, dispatchers, and generation
  controllers take `CONDITION` (validated, forwarded, and placed in the job name; an RDS-only
  simulation is exempt). The DLI stage additionally takes `--labeling-law` and `--occupancy`.
- **One shared RDS trajectory tier.** The trajectories and the ten-parameter `Theta_Set` are
  generated once, under the bare sibling alias (`Paths.sibling_alias`, exposed as `rds_alias`),
  and re-imaged by both workflows and both conditions at the DLI stage: the biology reads the
  `Theta_Set` as its label, the detector as the record of the reaction-diffusion nuisance it
  marginalizes (the ten RDS rows of `detector_parameterization` are nuisance-from-object,
  without ranges of their own). `Generate_Datasets.py` runs the tier once per split, fans the
  DLI passes out over `--workflows` x `--conditions`, refuses to regenerate an existing tier
  unless `--overwrite-rds` (a fresh draw would mislabel the videos rendered from the old one),
  re-images an existing tier with `--reuse-rds`, and checks the biology's per-condition
  `Nuisance_DLI` artifacts before anything runs. The biology HPC controller forwards
  `SIM_STAGE`, so `SIM_STAGE=rds` generates the tier alone.

### Changed

- **The detector calibrates against the reactive biology.** The detector re-images the shared
  reactive trajectory tier, so the ten reaction-diffusion parameters it marginalizes are the
  biology prior by construction. Under the labeling model a dissociating one-dye dimer leaves
  one visible and one invisible daughter, a kinetic track disappearance a static-composition
  detector would absorb into photobleaching; and a system with no registered reactions writes no
  reaction records, so a diffusion-only trajectory cannot be rendered. The detector
  posterior-predictive scene is rendered reactively at an RDS nuisance drawn from the biology
  prior (or pinned). The detector's HPC Simulation wrapper, dispatcher, and generation
  controller are DLI-only passes over the tier; an RDS-only job carries neither qualifier nor
  condition.
- The horizon audit spawns a (placement, render, labeling) seed triple per arm and records the
  condition and labeling law in every generated file; the posterior-predictive runner derives
  the condition from `--kind` and reads the condition's `Nuisance_DLI`; the Experiment,
  embedding-distance, and Nuisance_DLI analyses default their recording kinds to the run's
  condition (naming the other condition is a deliberate cross-condition application).
- The Sample-Geometric-Median analyses take the dataset condition as `--condition` and restrict
  the collection to that condition's rows; the `Nuisance_DLI` spec's `[block].condition`
  defaults to the artifact's own condition.
- `DETECTOR_WORKFLOW.md`, `PROJECT_CONTEXT.md`, `README.md`, `VALIDATION.md`, the HPC runbook,
  the analysis companion notes, and this repository's `CLAUDE.md` describe the implemented
  observation layer, the reactive detector, the condition slot, and the shared trajectory
  tier, and state the probe-kinetics assumption: ligand binding and unbinding within a
  recording are not modeled; first-order unbinding is absorbed by the per-condition
  calibrated bleach parameter; appearance from free labeled ligand and state-dependent
  affinity are declared residuals; partial occupancy is the static occupancy knob.

### Removed

- The dimer brightness multiplier and its two combination modes (`dimer_mule`, `dimer_model`,
  the dimer mask, the second-label draw, and the posterior-predictive `--dimer-model` flag):
  the dye-centric renderer needs no dimer-specific code path.
- The diffusion-only simulator mode (`build_system(pure_diffusion=...)`).
- The detector's own RDS tier: the `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Simulation_RDS.py`
  entry point, `detector_simulation_rds_support.py`, `build_nuisance_prior` and the RDS nuisance
  bounds of `detector_parameterization`, the import-time range-equality check, and the
  `Nuisance_RDS_Theta_Set` record.
- The `pooled` selection of the geometric-median analyses and the `CONDITION_CHOICES` constant:
  estimators, artifacts, and analyses are per condition.

## 0.1.0 - 2026-09-02

Founding release of the MONOMER_DIMER model family.

### Added

- **Codebase founded as a copy** of the tracked tree of `srm-and-sbi/srm-and-sbi-dimer-alp` at
  its frozen release `v0.4.23` (commit `d78b2f2`) — the reference implementation of the
  three-species DIMER model with the stationary OU brightness photo-physics. That repository's
  changelog records the copied functionality; this changelog records what this repository
  changes on top of it.

### Changed

- **Namespace**: package `srm_and_sbi_monomer_dimer_alp`; runtime prefix
  `SRM_AND_SBI_MONOMER_DIMER_ALP` on every entry-point script, output-file prefix, and
  project alias.
- **Condition tokens**: the experimental conditions are `FAB` (MET-FAB) and `INLB` (MET-INLB)
  in every filename pattern, schema field, CLI argument, and document; display surfaces prepend
  the receptor. The repo-iteration suffixes (`alp`, `bet`, ...) are a separate namespace and
  never name a condition. Experimental recordings enter this repository's data bank under the
  `FAB`/`INLB` names, with a provenance mapping to their public accession recorded at staging
  time.
- **Repository status** sections of `README.md` and `PROJECT_CONTEXT.md` describe this
  repository's charter: the DOL-explicit observation layer (measured degree of labeling,
  per-subunit label draws, probe occupancy), the reparameterized counts (true receptor
  abundance and composition), the condition axis (MET-FAB and MET-INLB as two frozen
  configurations of one codebase), and the `Nuisance_DLI` recalibration under the DOL-explicit
  model. Scientific sections describe the copied state where not yet rewritten.
