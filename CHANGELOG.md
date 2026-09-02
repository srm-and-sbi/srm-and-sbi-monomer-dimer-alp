# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
