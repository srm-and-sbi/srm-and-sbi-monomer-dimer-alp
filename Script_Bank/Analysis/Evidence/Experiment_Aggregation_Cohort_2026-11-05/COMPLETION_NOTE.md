# Full-cohort Experiment aggregation and the clearer report (2026-11-05): completion note

Plan: `Special_Analyses/EXPERIMENT_AGGREGATION_REPORT_IMPROVEMENT_2026-11-05/CLAUDE_REPORT_IMPROVEMENT_PLAN.md`.
Baseline: the accepted 0.1.45 checkpoint (`bafe2a5`) and the accepted pilot review copies `PILOT2CELLSR1`. Package
version after this work: 0.1.46 (committed after Codex's review; see §10). Nothing here selects a production
aggregation, window set, statistic, weight or nuisance mixture (§9).

## 1. The two complete-cohort main reports and their figures

| condition | review copy (data bank, beside the adopted product) | report | figures (six) |
|---|---|---|---|
| MET-FAB | `Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment_COHORT60PLOTS/` | `report.md` (7 checks passed) | `figures/aggregation_FAB_{all_marginals,all_joint,all_temporal,opening_marginals,opening_joint,window_selection}.png` |
| MET-INLB | `Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_MAP_Experiment_COHORT60PLOTS/` | `report.md` (7 checks passed) | `figures/aggregation_INLB_{...}.png` |

Copies of both folders (reports, exports, all figures) are under `review_copies/` here. Each report carries the early
**Data scope** table (original Experiment summaries: 60 cells, 600 windows; retained-cloud aggregation: 60 of 60 cells,
600 of 600 windows, 10,000 joint draws per window, "complete cohort"), the section "Experimental aggregation and nuisance
candidates" with the coverage statement "complete cohort", and 793 groups per condition: under `all` 60 per cell, 10 per
window position, 1 for the included dataset, 600 per cell–window pair; under `opening` 60, 1, 1, 60 (per-cell equal to
per-cell-window, per-position equal to global, listed as equivalences). The global pool is 6,000,000 draws; pooled groups
above 10,000 rows took the `weiszfeld_snap` route (73 groups per condition), single-window groups reuse the stored exact
medoid (720); every SGM is the retained draw its identity names (`aggregation_sgm_members` PASS). The adopted products,
their reports and arrays are untouched (`--regenerate-report` writes a review copy only).

The pilot review copies regenerated under the new presentation are `..._MAP_Experiment_PILOT2CELLSR2/` (both conditions);
the accepted `PILOT2CELLSR1` folders stay as the numerical baseline.

## 2. Retained-cloud index, identity and staging evidence, checksums

- **Catalog mapping (closed).** `catalog_verification/`: the sixty-cell catalog `Catalog_Note_Experiment.tsv` (from the
  JUPITER Experiment folder the adopted products ran on; sha256 `55d6ec65…`), SHA-256 lists of the 120 JUPITER working
  files, the 120 raw movies of the PC's read-only complete MET bank, the 15 PC-staged copies and rcl01's 120 staged
  files, and `verify_catalog_mapping.py` (+ `.log`): 120 of 120 JUPITER working files equal the bank movies the catalog
  names (all 120 hashes distinct); 15 of 15 PC copies and 120 of 120 rcl01 files equal the same sources. The historical
  InlB catalog cross-check is thereby resolved: the index rule of the 2026-11-03 staging note (CS3 01–20 → 0–19,
  CS3_160205 07/08/16 → 20–22, CS4 → 23–42, CS5 → 43–59) is the catalog's.
- **Staging on the PC.** The 105 recordings not copied before are symbolic links into the read-only bank (no copy, no
  disk use; the bank is only read), listed with source and hash in the data-bank note
  `Experiment/SPT_Data_MET_FAB_INLB_S-BSST712/COHORT_STAGED_FROM_PC_BANK_2026-11-05.md`; the catalog TSV now sits beside the
  recordings as the documentation names it. The collection itself read rcl01's own staged copies (RCL_FIAS_C, verified).
- **Collections.** 116 folders `Posit/..._<COND>_2S_50FPS_<TAG>_Window_Posterior_Draws_COHORTCELL<n>/` on the PC data bank
  (pulled from rcl01 with `--ignore-existing`; 478 MB with their per-collection reports and figures; the 116 `.npz` clouds
  are listed in `collections_SHA256SUMS`). Seeds: Fab 2026110500 + cell, InlB 2026110600 + cell (distinct per cell, so no
  two cells share one base-noise sequence in the across-cell pools; the four pilot collections keep their seed 20261103 and
  are reused, not resampled). `validate_union_{FAB,INLB}.log`: every collection valid against its product (shape, finite,
  identities, contract, SGM rows), union = 600 unique windows of 600, 60 cells of 60, no duplicates, no missing. rcl01
  keeps its copy of the 116 collections on RCL_FIAS_C (a second copy on another machine).
- **Helpers (here).** `collect_cohort.sh` (per-cell loop, restartable, never overwrites), `missing_cells.py` (missing list
  derived from product identities minus validated collections), `validate_collection.py` (per-collection and `--union`
  checks), `pull_and_regenerate.sh` (pull, union validation, CPU regeneration); `rcl01_logs/` (per-cell companion logs and
  `/usr/bin/time -v` records, stream logs).
- **Unresolved limitation.** None on identities. The InlB PSF width above its prior ceiling remains the documented
  extrapolation (visible in the outside-prior columns and the joint figures).

## 3. Measured cost

| item | value |
|---|---|
| GPU collection, rcl01 (RTX 6000 Ada, env SRM_AND_SBI_ENVY_V0, torch 2.9.1+cu126), four concurrent streams | 18 min wall (16:26:01 to 16:43:57 UTC) for 116 cells; per cell 32.7–38.6 s (median 34.6 s) including estimator load, recording read, figures and report; sampling 1.60 s per window median (2.30 s max); peak GPU memory 1.68 GiB per process; peak RSS 5.49 GB per process |
| CPU regeneration, PC (`--regenerate-report`, 60 collections) | Fab 56 s wall, 2.29 GB peak RSS (aggregation 36 s); InlB 50 s, 2.30 GB (aggregation 30 s) |
| Disk | 116 collection folders 478 MB on the PC data bank (the clouds alone 2.1 MB each); review copies 16 MB here; PC staging 0 bytes (links) |

The PC GPU was not used for the collection (the lead's direction of 2026-11-05: rcl01 or JUPITER for GPU work).

## 4. Pilot numerical invariance and tests

- `pilot_r2/pilot_invariance_R1_R2.py` (+ `.log`): PILOT2CELLSR1 against PILOT2CELLSR2 (the same two collections per
  condition, the new presentation): every array of `aggregation_candidates.npz` except the manifest (27 arrays per
  condition), both CSV exports and every manifest field outside presentation/provenance/timing identical, INVARIANCE PASS.
  The first and the second cohort regeneration (before and after the compact source description) have identical archive
  arrays as well.
- Tests: `tests/test_experiment_aggregation.py` 17 of 17 (the new `test_presentation_labels_scope_lines_and_the_early_scope_table`:
  labels, window-set phrases, scope lines, the Data scope table for a complete product and a two-cell pilot, a named set,
  the opening equivalences, the median-statistic label, the figures' legends and subtitles, the compact description of a
  sixty-source set); the full suite 311 of 311 over 38 files (`MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/<file>.py`,
  PASS lines counted against `def test_` definitions). The illustrative Nuisance DLI build (the 0.1.45 example driver)
  rebuilt from the PILOT2CELLSR2 archive: 3 vectors, probabilities 0.375/0.25/0.375, figures with the new labels.

## 4a. Review correction (Codex, `Special_Analyses/EXPERIMENT_AGGREGATION_COHORT_REVIEW_2026-11-05/REVIEW.md` R1)

`scope_table` now tells three states apart: aggregation explicitly disabled ("not computed (aggregation disabled)",
whether or not the product stores clouds), enabled without any clouds ("not computed (no retained clouds)"), enabled
with clouds (the coverage). The test asserts all three. Neither complete-cohort report is affected (both take the
enabled branch); nothing was resampled or regenerated. Aggregation tests after the correction: 17 of 17.

## 5. What changed in the labels, scope table and captions

- One formatter (`experiment_aggregation.display_name`, `pooled_samples`, `window_set_phrase`, `scope_line`,
  `scope_table`, `sources_description`): "One SGM vector per cell / per window position / for the included dataset / per
  cell–window pair", "Existing fixed nuisance vector: SEL"; legends "SGM per cell (60 vectors)" etc.; a marginal-median
  component is "marginal-median vector per …"; window sets "all ten 2 s windows", "opening 0–2 s window only (60 selected
  windows of the 600)", or a named set with positions and window starts. The constructions table gains "visible
  description" and "samples pooled before one SGM is selected"; the section states the two independent choices once.
- Every aggregation figure: title per the plan, a scope subtitle (condition, covered cells and windows, windows included,
  draws per window, representative), legends outside the panels, "training prior bounds" labeled; the original diagnostic
  figures carry "Source: the original Experiment summaries (…), not the retained clouds of the aggregation."
- Captions name what each curve, point, line and band is; the temporal caption says the bold line is pooled joint samples
  (not an average of the thin lines), the gray bands summarize the input posteriors (not a confidence band for the bold
  line, not a range containing 90 % of the cells) and no physical cause is assigned; the opening captions name the exact
  equivalences; the window-selection caption gives two data-derived cell examples and states that the effect is
  recording-dependent. Screenshots for review: `review_copies/*/figures/aggregation_*.png` (both conditions).

## 6. Statement

No production aggregation, window set, construction, statistic, weight, bleaching treatment or nuisance mixture has been
selected. The SEL artifacts are unchanged and shown as the fixed reference only. The adopted Experiment products, the
estimators, the R1 pilot review copies, the original movies and every retained draw collection are preserved.

## 7. Commands (PC unless stated)

    # rcl01 (MACHINE_PROFILE=rcl01), four streams, two per condition:
    bash <Evidence>/collect_cohort.sh FAB C128V42A 2026110500 <0|1> 2 <Evidence>/rcl01_logs
    bash <Evidence>/collect_cohort.sh INLB C128V42AP3LR32 2026110600 <0|1> 2 <Evidence>/rcl01_logs
    # PC: pull, validate the union, regenerate both reports (label COHORT60PLOTS)
    bash Script_Bank/Analysis/Evidence/Experiment_Aggregation_Cohort_2026-11-05/pull_and_regenerate.sh

## 8. Files here

`catalog_verification/`, `collect_cohort.sh`, `missing_cells.py`, `validate_collection.py`, `pull_and_regenerate.sh`,
`rcl01_logs/`, `validate_union_{FAB,INLB}.log`, `collections_SHA256SUMS`, `regenerate_{FAB,INLB}_COHORT60PLOTS.{log,time}`,
`pilot_r2/` (the R2 regeneration logs and the invariance check), `review_copies/`, `SHA256SUMS`.

## 9. Open for the lead

The scientific selection (acquisition window set, construction or mixture, weights, statistic, bleaching treatment); the
commit of 0.1.46 (code, tests, documentation, this evidence) and the push; whether rcl01's copy of the 116 collections is
kept as the backup copy.

## 10. Changed files (the 0.1.46 commit)

`srm_and_sbi_monomer_dimer_alp/{visualization_aggregation,experiment_aggregation,experiment_runner,__init__}.py`,
`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py`, `tests/test_experiment_aggregation.py`,
`pyproject.toml`, `CHANGELOG.md` (0.1.46), `VALIDATION.md`, `DETECTOR_WORKFLOW.md` (§7.2, §7.3),
`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.md`, and this Evidence folder (new).
