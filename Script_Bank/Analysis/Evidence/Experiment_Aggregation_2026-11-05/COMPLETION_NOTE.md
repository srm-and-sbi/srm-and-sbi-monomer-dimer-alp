# Completion note: Experiment aggregation and the empirical Nuisance DLI (2026-11-05)

Implementation of `Special_Analyses/EXPERIMENT_AGGREGATION_AND_NUISANCE_DLI_2026-11-05/CLAUDE_IMPLEMENTATION_PLAN.md` for review.
Package version 0.1.45 (uncommitted; PATCH bump with its changelog entry). No neural architecture, training, prior,
Evaluation definition, production nuisance artifact, HPC submission, commit, push or fleet sync. The adopted products,
estimators and SEL artifacts are unchanged (checksums below).

## 1. Index

| item | location |
|---|---|
| Shared calculation (clouds, contract, coverage, the four constructions, exports, the report section) | `srm_and_sbi_monomer_dimer_alp/experiment_aggregation.py` (new, 1139 lines) |
| Figures (marginal CDFs, joint corner, temporal, window selection; shared with the Nuisance build) | `srm_and_sbi_monomer_dimer_alp/visualization_aggregation.py` (new) |
| Experiment integration (settings, dry-run, retention, merge pass-through, CPU regeneration, reference loading, parser) | `srm_and_sbi_monomer_dimer_alp/experiment_runner.py` |
| Weighted `empirical_representatives` construction, spec template/validation/build, round trip | `srm_and_sbi_monomer_dimer_alp/detector_nuisance_dli.py` |
| Analysis script: `--emit-representatives-template --candidate-archive`, CPU `--build` route, report with weighted figures | `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py` (+ usage block in its `.md`) |
| Kernel capacity control; reporter prose hook; provenance file list | `sample_geometric_median.py`, `diagnostics.py`, `provenance.py` |
| Launchers and submit scripts (both workflows): `AGGREGATION_WINDOW_SETS`, `NO_AGGREGATION`, `REFERENCE_NUISANCE_TAG` | `Script_Bank/HPC/*_HPC_Experiment.sh`, `*_HPC_Submit.sh`, `Script_Bank/HPC/README.md` |
| Tests | `tests/test_experiment_aggregation.py` (new, 16 tests), `tests/test_hpc_submit_forwarding.py` (new, 2 tests) |
| Changelog and version | `CHANGELOG.md` (0.1.45), `pyproject.toml`, `__init__.py` |
| Extended Experiment review report, Fab (two of sixty cells) | data bank `Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment_PILOT2CELLSR1/report.md` (copy: `review_copies/`; the earlier `_PILOT2CELLS` folders are superseded, see section 9) |
| Extended Experiment review report, InlB (two of sixty cells) | data bank `Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_MAP_Experiment_PILOT2CELLSR1/report.md` (copy: `review_copies/`; the earlier `_PILOT2CELLS` folders are superseded, see section 9) |
| Supporting exports beside each review report | `aggregation_candidates.npz` (Fab sha256 47c6155fb77f…, InlB 28db6fe0eb6a…; the group arrays equal the earlier archives d353d4bffed1… / bb7a7d8e225f… exactly, the manifest gained the target binding), `aggregation_groups.csv`, `aggregation_comparison.csv`, `aggregation_manifest.json`, `figures/aggregation_<COND>_{all,opening}_{marginals,joint}.png`, `..._all_temporal.png`, `..._window_selection.png` |
| Illustrative representatives build | `example_build/` (see §6) |

Scoped source diffs (`git diff --stat` plus the three new files): experiment_runner.py +205/−; detector_nuisance_dli.py +276/−; the
analysis script +175/−; diagnostics.py +14; provenance.py +4; sample_geometric_median.py +18/−; the four launcher/submit scripts and the
HPC README; CHANGELOG, pyproject, `__init__`. The unrelated uncommitted files of 2026-11-03 (the Review/Draws companions and notes,
the two Evidence folders, the InlB declared-RDS TOML, the predictive-video note edit) are untouched and separate.

## 2. What the implementation does

**One reusable calculation (plan §4).** `experiment_aggregation.aggregate(collection, table, window_sets=...)` takes retained clouds
with identities (`CloudCollection`: from an Experiment product that stored `posterior_samples_cloud`, or from retained draw
collections combined under one contract), the parameter table (keys, prior box, log/linear conversion; nothing hardcodes six
dimensions, sixty cells, ten windows or one geometry) and the requested named window sets, and returns the four constructions per
condition. Each group stores the complete SGM with its source identity (cell, chunk, draw index, collection, flat index), its
method (`window_stored`: the window's stored exact medoid reused when its row is located in the cloud; `exact_medoid` up to the
pooled capacity of 10,000 rows; `weiszfeld_snap` above it, through the collection kernel with the new `exact_capacity` control, so
no pooled group allocates a quadratic matrix and the 20,000-row position pools of the two-cell pilot take the Weiszfeld route), the
marginal-median vector, the mixture quantiles of the pooled draws, the outside-prior mass per group and per coordinate, and the
members. Canonical identity order (kind, cell, chunk) rules every pool; equal weights are realized by the equal draw counts
(checked). Coverage must be complete and identities unique; a window set naming an absent position is refused with the
identities; contract disagreements (checkpoint, keys, transform, scale, geometry, draw count, sampling mode) are refused naming
the source and the field; labels, seeds, implementation hashes and write times are disclosed. Opening-only and single-cell
equivalences are detected and reuse the computed group (recorded as `equivalent_to`).

**Experiment report (plan §5).** `write_experiment_outputs(..., aggregation=AggregationRequest)` adds the section "Experimental
aggregation and nuisance candidates" after the existing tables, for a live run (single process after the complete product; sharded
run once, inside `--merge` after the validated merge, never per rank) and for a CPU regeneration alike, through one function
(`report_aggregation`). Contents: the prose header (purpose, coverage per condition as "C of N cells, W of M windows" with the cell
ids, window sets, retained-cloud sources with checksums and disclosed differences, methods, the reference, the supporting-file
index, reading rules), checks (`aggregation_coverage_complete`, `aggregation_equal_window_weights`,
`aggregation_contract_matches_product`, `aggregation_sgm_members`), the canonical constructions table with the reference row, the
equivalent-groups table, per (condition, window set) the SGM-representatives table (n, median, IQR, central 90 % range by the
inverse weighted empirical CDF, median versus reference in percent and dex, outside-prior mass) and the global-representative table
(global SGM as a draw, global median vector as a composite, pooled 90 % interval, reference, pooled draws outside prior), the
source-uncertainty summary (within-window and pooled posterior intervals against the dispersion of representatives) and the
SGM-versus-median-vector table; figures per condition: marginal CDFs and joint corner per window set, temporal comparison for a
multi-position set, window-selection comparison across sets. Exports beside the report: `aggregation_candidates.npz` (every
group's SGM and median vectors, quantiles, identities, methods, equivalences, window-set definitions, keys, scale, prior box,
reference vector, manifest), `aggregation_groups.csv`, `aggregation_comparison.csv`, `aggregation_manifest.json` (constructions
and definitions, window sets, coverage and note, contract, sources with checksums, disclosed differences, methods, timing,
reference identity, the product manifest's identity block, the archive checksum). The existing `--aggregation` point-summary view
is untouched and distinct. A disabled request or a product without clouds writes the statement that the calculations are
unavailable; nothing is substituted.

**Stage controls (plan §5).** `--aggregation-window-sets` (default `all,opening`; items `all`, `opening`, `NAME=a-b`, `NAME=a+b+c`),
`--no-aggregation` (legacy report; no retention unless `--dump-posterior-samples`), `--reference-nuisance-tag` (default `SEL`;
detector only; absent artifact = no reference shown). Enabling aggregation sets the existing cloud retention (the manifest's
`stored_optional_fields` carries `posterior_samples_cloud`; no second cloud is drawn). The dry-run prints the effective retention
(with the size), the aggregation state and sets, the reference tag, and for a regeneration the product, the cloud sources and
the review destination. Both launchers forward the three knobs in `EXP_ARGS`, so every rank and the merge see the same setting;
both Submit scripts export them; the HPC README's recompute-a-rank paragraph names them.

**Saved results (plan §6).** `--regenerate-report --review-label <TOKEN> [--cloud-sources a.npz,b.npz]`: CPU only, loads the saved
product through the schema, builds the reporter at `<product dir>_<TOKEN>/` (refused if it exists), renders with
`persist_arrays=False` (the adopted arrays and report are never touched), combines the named retained collections under the
product's contract (or uses the product's own clouds), and states for a subset: "This is a pilot subset, not the complete
condition: MET-FAB 2 of 60 cells (cells [4, 26]), 20 of 600 windows. The original MAP and summary references above come from the
adopted product; the aggregation summaries come from the retained collections named below, fresh draws of the same estimator
that do not reconstruct the original sampling attempt." The label is output-only; the estimator tag and the product read are
unchanged. With all compatible clouds the same path yields the complete condition without new sampling.

**Nuisance DLI (plan §7).** New construction `empirical_representatives` (`POOL_CHOICES`, `_EMPIRICAL`, `_WEIGHTED`): a sample
matrix with per-row `probabilities` (validated finite, non-negative, positive mass, normalized within 1e-6), the selection record
(archive path/checksum/product/coverage note, components with identities and member probabilities, consolidation, overrides,
rationale, outside-prior mass, whether the rows are sampled draws). The spec: `[block] candidate_archive`,
`candidate_archive_sha256` (verified at build when given), `rationale` (required), `[[component]]` tables (`source_experiment`
must equal the archive's product; `condition`; `window_set`; `grouping` in the four constructions; `statistic` `sgm` (default)
or `median`; `weight`), optional `[override.<KEY>]` (`value` physical, `reason`). Member probability = weight / member count;
weights must sum to one within 1e-6 (never repaired); exact duplicates consolidated by adding probabilities with memberships
kept; an override replaces the coordinate in every vector, is recorded with the replaced range and marks the rows as transformed.
`sample(n, seed=None)` draws whole rows at the recorded probabilities; legacy artifacts stay uniform; an unfamiliar construction is
refused by the constructor, so an old reader never samples a weighted artifact uniformly. `pool_mode` records `unrestricted` when
any probability mass lies outside the prior box (the `selection_user` convention). The analysis script's
`--emit-representatives-template --candidate-archive <archive>` prints the archive's comparison (medians of the SGM
representatives per collection beside SEL) and writes the skeleton with every component commented out; `--build` (CPU) with the
spec's choice routes to the representatives build and writes the report (components, consolidation, overrides, weighted marginal
summary, the weighted marginal-CDF and joint figures of the mixture beside its components and SEL, drawn by the Experiment's
plotting functions). SEL is never mixed in automatically; raw posterior draws and an SGM of SGMs are not constructions of this
route; the bleaching reference stays a separate selection question (an override is the explicit mechanism, shown in the report).

## 3. Validation

Tests: `tests/test_experiment_aggregation.py` 16/16 PASS and `tests/test_hpc_submit_forwarding.py` 2/2 PASS; the whole suite after the review corrections: 38 files, 310 PASS of 310 test definitions, every file exit code zero (run file by file after the corrections) (0.1.45; the first-pass figure was 37 files, 303 of 303).

| plan §10 item | where |
|---|---|
| 1 four constructions, equal weights, coverage/duplicate/contract refusals | `test_four_constructions_on_identifiable_clouds_with_equal_weights`, `test_incomplete_coverage_duplicates_and_incompatible_metadata_are_refused`, `test_window_sets_parse_and_refuse_malformed_items` |
| 2 opening equivalences; membership follows the declared sets | `test_opening_equivalences_and_membership_follow_the_declared_sets` |
| 3 pilot representatives reproduce (Fab 26 chunk 1/draw 3422; Fab 4 chunk 3/7806; InlB 11 chunk 7/2186; InlB 42 chunk 8/1582; by identity and vector) | `test_the_saved_four_cell_pilot_representatives_reproduce` (the retained collections of the Evidence folders), and the review archives: the same four identities, vectors and pooled quantiles equal the saved `result.json` files of the four aggregation folders (checked 2026-11-05) |
| 4 pooled quantiles recompute; SGM membership; medians not claimed as rows | `test_exports_round_trip_and_the_archive_marks_medians_as_composites`; the `aggregation_sgm_members` check of every report |
| 5 single process = merged shards; incomplete-shard and mixed-storage guards | `test_single_process_and_merged_shards_give_the_same_grouping` (+ the existing `test_contract_enforcement.py`) |
| 6 the extended report: every construction, set, table, linked figure; live and saved use one implementation; pilot coverage stated | `test_extended_report_contains_the_section_for_live_and_saved_rendering`; the two review copies |
| 7 probabilities weight/n, singleton, overlapping components, invalid selections, weighted quantiles, outside-box mass | `test_mixture_probabilities_consolidation_overrides_and_invalid_selections`, `test_weighted_quantile_is_the_generalized_inverse` |
| 8 weighted round trip; sampler probabilities (seeded frequency check); legacy uniform | `test_weighted_artifact_round_trip_sampler_loader_and_legacy_uniformity` |
| 9 loader on an isolated test artifact | the same test (`NuisanceDLI.load` on a temporary artifact) and the illustrative build in `example_build/` (Evidence folder, not the data bank) |
| 10 source checksums unchanged | Fab product `addc6fc1395aed08…`, InlB product `bc764157da595fe0…`, Fab estimator `598e3b0c0c0d67c4…`, InlB estimator `98b22fdc33a1af79…`, Fab SEL `fe9cb5bae47eb0fc…`, InlB SEL `2b2d98f99e4fd0c5…`; the adopted report files keep their dates (2026-10-23, 2026-10-27) and sizes; the four retained collections verify against their Evidence `SHA256SUMS` |

Report checks of each review copy: 6 passed (the original two plus the four aggregation checks).

## 4. Effective configuration and manifest fields

Regeneration commands (CPU; Fab shown, InlB analogous with `--condition INLB --artifact-tag C128V42AP3LR32` and its two collections):

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment.py \
        --condition FAB --total-time-seconds 2.0 --artifact-tag C128V42A --pool-mode unrestricted \
        --regenerate-report --review-label PILOT2CELLSR1 \
        --cloud-sources <Evidence>/Window_Posterior_Pilot_2026-11-03/..._FAB_..._Window_Posterior_Draws.npz,<Evidence>/Window_Posterior_Additional_Cells_2026-11-03/..._FAB_..._Window_Posterior_Draws_CELL4.npz

Effective settings printed by the stage: retention `posterior_samples_cloud` on (required by the aggregation); aggregation on with
window sets `all` = chunks 0..9 and `opening` = chunk 0; reference tag SEL (artifact present for both conditions); exact capacity
10,000. `aggregation_manifest.json` fields: `archive_format_version`, `stage`, `product_label`, `condition_labels`,
`constructions` + `construction_definitions`, `statistics`, `default_statistic`, `window_sets`, `coverage`, `coverage_note`,
`n_product_cells`, `n_product_windows`, `parameter_keys`, `quantile_levels`, `metric`, `sgm_scale`, `prior_low`/`prior_high`,
`methods`, `contract`, `sources`, `disclosed_source_differences`, `reference` (+ `reference_vector`), `equivalences`,
`timing_seconds`, `aggregation_seconds`, `source_product_manifest`, `source_clouds_note`, `exports`, `archive_sha256`, `written_at`.

## 5. Time, memory, storage (PC, CPU; measured)

| run | wall | peak RSS | notes |
|---|---|---|---|
| Fab regeneration (20 windows, 2 collections, 2 window sets, 39 groups); after the corrections 17.0 s / 0.95 GB | 14.5 s | 1.00 GB | aggregation itself 1.05 s (200,000-row global pool, 100,000-row per-cell pools, 20,000-row position pools, all `weiszfeld_snap`; 24 groups reuse stored window SGMs) |
| InlB regeneration; after the corrections 16.9 s / 0.87 GB | 14.6 s | 0.98 GB | aggregation 0.9 s |
| Review folder on disk | 5.2 MB (Fab), 5.6 MB (InlB) | | archive 23 KB each; the rest are the report and 19 figures |
| Illustrative representatives build | about 6 s | | artifact 3 vectors |

For planning only (not a guarantee): a complete condition pools 6,000,000 rows for the global group; the Weiszfeld route holds that
pool in float64 (288 MB) plus temporaries, and no quadratic allocation occurs.

## 6. Nuisance DLI interface and the illustrative example

`example_build/EXAMPLE_Nuisance_DLI_Spec_template.toml` is the emitted skeleton for the Fab PILOT2CELLSR1 archive;
`EXAMPLE_Nuisance_DLI_Spec.toml` is the filled example (two components: `all/per_cell/sgm` at weight 0.75 and `all/global/sgm` at
weight 0.25; the weights are an interface example, NOT a scientific decision), built by `example_representatives_build.py`
through the production build route into `EXAMPLE_Nuisance_DLI.npz` (3 vectors, probabilities 0.375 / 0.25 / 0.375, pool_mode bounded)
with its report `EXAMPLE_Nuisance_DLI_Analysis/report.md` (6 checks passed; components, weighted marginal summary, the weighted
marginal-CDF and joint figures beside SEL). Nothing was written to the data bank; no production spec exists.

## 7. Remaining scientific choices (not made here)

- The represented acquisition period: window set `all`, `opening`, or another named set.
- The construction or mixture (`per_cell`, `per_window_position`, `global`, `per_cell_window`, or weighted components) and its weights.
- The representative statistic (SGM default; the marginal-median vector as an explicitly named alternative).
- Any parameter-specific treatment (the bleaching reference 0.05 against the pooled centers near 0.09; an override is the explicit mechanism).
- Full-cohort sampling (sixty cells per condition) before a production selection; the pending InlB catalog cross-check.

## 8. Textual deviations and notes for the documentation integration

- Named window sets use the grammar `NAME=a-b` (range) and `NAME=a+b+c` (list) on the command line and in the launcher knob.
- Component weights must be written normalized (sum to one within 1e-6); the build refuses other sums rather than normalizing.
- The pooled exact-medoid capacity of the aggregation is 10,000 rows (the collection kernel's default of 20,000 is unchanged for other callers), so every multi-window pool at 10,000 draws per window takes the Weiszfeld route, as the completed four-cell comparisons did.
- The source-uncertainty table is written for the first window set spanning several positions (`all` in the standard request).
- The distance-rank diagnostic of the pilots is not part of the Experiment section (the draft does not require it); the section's reading rules state that membership of the reference inside intervals from the same draws is a compatibility diagnostic.
- The reference row appears in the constructions table as `reference` with the artifact's tag; the weighted artifact's `pool_mode` is `unrestricted` when any probability mass lies outside the prior box.

## 9. Review corrections (Codex, 2026-11-05; `Special_Analyses/EXPERIMENT_AGGREGATION_REVIEW_2026-11-05/IMPLEMENTATION_REVIEW.md`)

All seven corrections and the four presentation points are implemented; the probes of the review
(`review_probes.py`, rerun unchanged on the corrected state) now record every negative case as refused and the
sampler consistent: `review_probes_after_corrections.json` in this folder.

| item | correction | where | regression |
|---|---|---|---|
| R1 bind the archive to its target | every archive manifest records `target` (project alias, condition, timing label, window length, estimator tag) from the Experiment run; `representatives_from_spec`, `build_nuisance_dli` and `emit_representatives_template` take the explicit `target=` of the artifact under construction and refuse another condition, duration or window length, an unbound archive, and a component whose recording condition is not the target's (a cross-condition application stays an inspection result); the analysis script passes its `--condition` and timing through template emission, dry-run and build, before any file is written | `experiment_aggregation.export_manifest/AggregationRequest`, `experiment_runner` (target binding), `detector_nuisance_dli.check_archive_target`, the analysis script `_target` | `test_r1_...` ; probe `inlb_archive_under_fab_target` refused |
| R2 HPC forwarding | `AGGREGATION_WINDOW_SETS` rides the exported environment via `ALL` in both Submit scripts (like `KINDS`), never the explicit comma-split `--export`; the dry-run names it; both launchers forward it as one argument to every worker and the merge | both `*_HPC_Submit.sh`, `Script_Bank/HPC/README.md` | `tests/test_hpc_submit_forwarding.py` (fake `sbatch`, fake interpreter; no job); probe `comma_bearing_slurm_exports` clean |
| R3 complete coverage against the declared geometry | `all` = every position the declared geometry admits; every included cell must carry every requested position (a position missing from every cell is refused for `all`; an explicit subset keeps its scope and weights); named positions beyond the geometry are refused at preflight before neural work; supplied identities must be windows of the source product (`aggregation_identities_in_product` check) | `expected_positions`, `validate_window_sets_against_geometry`, `validate_coverage`, `verify_identities_in_product`; the runner's preflight | `test_r3_...`; probe `same_window_missing_from_every_cell` refused |
| R4 preprocessing contract between collections | conversions compared pairwise between the collections and against the product; a conflict is refused even when the product records none; the agreed conversion is kept in the merged contract with a status (verified / unverified / unavailable) stated in the report | `combine_clouds`, `cloud_from_product`, the contract check note | `test_r4_...`; probe `different_input_conversions_with_product_reference` refused |
| R5 physical domain of overrides | one shared rule `physical_domain_error` (finite; positive for a log row; a probability below one) for `selection_user`, `[override.<KEY>]` and every final vector of an explicitly selected artifact; the constructor refuses a stored probability parameter at or above one in those constructions; acknowledged extrapolations of physically valid values stay available; diagnostic clouds are not clipped | `detector_nuisance_dli` | `test_r5_r6_...`; probe `bleaching_override_above_one` refused |
| R6 probabilities accepted = probabilities sampled | a total within 1e-6 of one is accepted and divided by its sum before storage and sampling (an already normalized vector stays bit-identical); the manifest records the convention and the removed residual; materially different totals are refused | `NuisanceDLI.__init__`, `sample`, `manifest` | `test_r5_r6_...`; probe `probability_tolerance_and_sampler` samples |
| R7 one median convention | the selection console uses `archive_summary_rows` (the report's `representative_summary`, inverse empirical CDF); console, comparison export, report table and mixture report agree (Fab `all/per_cell/sgm` mu_r 1.6486 px everywhere; the earlier console value 1.7345 px was `np.median`) | the analysis script, `experiment_aggregation.archive_summary_rows` | `test_r7_...`; console/CSV/report check in `regenerate_*_R1.log` and `emit_template_INLB_dryrun.log` |
| presentation | physical tick labels from the plotted range (four to five values, never only the decades) and unit labels on every axis (PSF width in px as sqrt(2) x sigma, dimensionless log-spreads, photons per dye and frame, bleaching probability over the 100-frame reference interval, flicker rate per second); the reference row states the current method with the per-parameter decision record in a table; the opener distinguishes input posterior mixtures from the exported collections of point representatives; supporting files and figures are links | `visualization_aggregation`, `report_aggregation` | the two review copies |

Reruns: both review copies regenerated on the CPU under the label `PILOT2CELLSR1` (`regenerate_FAB_R1.log/.time`,
`regenerate_INLB_R1.log/.time`: 17.0 s / 0.95 GB and 16.9 s / 0.87 GB; 7 report checks passed each). Their group arrays
(constructions, identities, SGM and median vectors, quantiles, methods) equal the first-pass archives exactly; the four
pilot representatives reproduce (Fab 26 chunk 1/3422, Fab 4 chunk 3/7806, InlB 11 chunk 7/2186, InlB 42 chunk 8/1582).
The illustrative build and the InlB template dry-run were rerun on the new archives (`example_build/`, `example_build.log`,
`emit_template_INLB_dryrun.log`). The first-pass review copies, example build and logs are kept under
`_superseded_before_review/`; the first-pass data-bank folders `..._MAP_Experiment_PILOT2CELLS/` (both conditions) are
superseded by `..._PILOT2CELLSR1/` and await the lead's word before deletion (nothing was deleted). Adopted products,
estimators and SEL artifacts: checksums unchanged (section 3, item 10).

Not changed by the corrections: the scientific definitions, the constructions, the saved results, the pilot collections,
the adopted artifacts; no sampling, simulation, training or HPC job; no commit, push or sync.
