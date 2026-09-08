# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
