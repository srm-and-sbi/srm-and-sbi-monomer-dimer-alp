# Completion note: posterior compatibility of SGM representatives and `posterior_rank_median` member weighting

Implementation of `Special_Analyses/EXPERIMENT_COMPATIBILITY_AND_NUISANCE_WEIGHTS_2026-11-06/CLAUDE_IMPLEMENTATION_PLAN.md`
on the PC (profile `mars_pc`, CPU only), 2026-11-06, and the corrections requested by Codex's first review
(`Special_Analyses/EXPERIMENT_COMPATIBILITY_REVIEW_2026-11-06/REVIEW.md`, §9 below). Baseline 0.1.46 at `f0228498bfda180ad000c3a175bc6e91f361957a`, clean
working tree; the brief's source snapshot hashes were verified before the first edit. Nothing is committed: the result is
returned for Codex review before the normal release checkpoint. No HPC computation, no new posterior sampling, no
production nuisance construction, no biology generation and no deletion of any data product (see §7 for the one
self-created intermediate copy I replaced).

## 1. Changed files

| File | Change |
|---|---|
| `srm_and_sbi_monomer_dimer_alp/experiment_aggregation.py` | shared rank primitive `reference_distance_ranks`; `CompatibilityBlock`/`CompatibilityResult`, `compute_compatibility` (per_cell block per condition and set; per_cell_window block per cell; statuses `computed` / `insufficient_peers` / `not_applicable`); `compatibility_protocol`, `compatibility_manifest`; archive format 2 (`ARCHIVE_FORMAT_VERSION = 2`, `SUPPORTED_ARCHIVE_FORMATS = (1, 2)`), `export_manifest(..., compatibility=)`, `write_aggregation_exports(..., compatibility=)` adding the score/status/reference/outside-prior-key arrays and columns, the rank blocks and `aggregation_compatibility.csv`; `load_candidate_archive` accepting formats 1 and 2 with `validate_archive_compatibility`; `archive_component` exposing each identity's compatibility and group index; the report subsection `_report_compatibility` (definition, scope and availability table with zero counts, inline figures, availability statements) and the atlas writer `_write_atlas`; the read-back check `aggregation_archive_reads_back` |
| `srm_and_sbi_monomer_dimer_alp/visualization_aggregation.py` | `figure_compatibility_cells` (matrix with aligned median-score strip), `figure_compatibility_windows`, `figure_compatibility_atlas_page`, `figure_member_probabilities`; one 0–1 scale, gray self comparisons |
| `srm_and_sbi_monomer_dimer_alp/detector_nuisance_dli.py` | `MEMBER_WEIGHTINGS`, `RANK_WEIGHTED_GROUPINGS`; spec validation of `member_weighting`; `member_probabilities` (the hierarchical normalization and every refusal); `representatives_from_spec` with per-member scores/contributions, component normalizers, zero counts, effective numbers with their scopes, `force_uniform` and the recorded `equal_member_baseline`; `component_weighting_availability`; template text and per-component availability lines. Artifact format unchanged |
| `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py` | truthful dry-run per component (`_policy_line`); template console availability; `_report_representatives` rewritten: selection table, consolidation with contributions, overrides note, `representatives_probabilities`, `representatives_marginals` (selected against the equal-member baseline), `representatives_components`, `representatives_joint`, `representatives_members.csv`, `representatives_rows.csv` |
| `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Draws.py` | the joint-depth table calls the shared primitive; the `joint_depth.csv` contract and values are unchanged (the primitive reproduces `mean(d_draws > d_ref)` exactly; tested) |
| `tests/test_experiment_compatibility.py` (new, 13 tests) | the brief's §9 table and the review's regressions (§9): primitive on hand-countable clouds and against the companion's direct count; self exclusion by identity with an identical peer; two mutually supporting distant windows (nonzero mean rank, zero median, zero probability, never sampled); missing against zero, constructor refusals; reference scopes (same cell, same condition and set, named sets, no cross-condition contamination); opening aliases; pool invariance; archive round trip, legacy format 1, malformed data (reordered block, stale score, unmasked diagonal, inconsistent status, other protocol, unknown version); hierarchy (equal cell mass despite different score totals, demonstrated against a flat normalization); mixtures, duplicates, overrides, the baseline, the artifact record and legacy records; the report subsection, figures, atlas and links for live and saved rendering, disabled products untouched |
| `tests/test_experiment_aggregation.py` | the three export call sites pass the compatibility result (no assertion changed) |
| `DETECTOR_WORKFLOW.md` §7.2–§7.3 | the documentation draft integrated: member weighting equations and rules, the compatibility definition, reference-population table, report contents, archive format 2; §7.3 steps 2–3 |
| `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.md` | files list, compatibility paragraph, the `member_weighting` field and its rules, template and dry-run interpretation, report contents, project state, evidence links |
| `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Draws.md` | the shared rank definition and the distinction from the official aggregation reporting |
| `PROJECT_CONTEXT.md`, `VALIDATION.md`, `CHANGELOG.md`, `pyproject.toml`, `__init__.py` | module descriptions, the test inventory (321 tests over 39 files), the 0.1.47 entry, version 0.1.47 |

Not touched: Prime entry points, HPC scripts, `experiment_runner.py` (its existing `report_aggregation` call serves live,
merged and regenerated reports), the neural estimator, training, Evaluation, SGM optimization, data generation, camera
settings, labeling, the artifact sampler and the artifact format.

**Deviations from the brief, explicitly.**
- The constructor's `representatives_marginals` figure now shows the selected probabilities against the equal-member
  baseline (and SEL), as §8.3 asks; the earlier components-beside-mixture view moved to a new figure
  `representatives_components` so that information is not lost.
- No caching of reference contexts was implemented: the full computation takes 0.65 s per condition, so a cache would
  add a correctness surface without a measurable benefit.
- `compute_compatibility` returns NaN for a missing score inside the engine; the archive stores NaN with status
  `insufficient_peers`, the CSV stores an empty field, and `archive_component` exposes `None`. Zero is stored as 0.0.
- The illustrative build (§6) was produced through the companion's own functions into this evidence folder rather than
  through the CLI, because the CLI writes its spec and artifact into the production `Posit` directory; the CLI's dry-run
  console path and its build/report path were called unchanged.

## 2. The two regenerated reports and the archive format

The corrected reports carry the review label `COHORT60RANKSR2` (the first checkpoint's `COHORT60RANKS` copies stay on
the PC until their cleanup is authorized; their scores, ranks and candidate arrays are identical to R2, only the
zero-median wording and the manifest timestamps differ):

| Condition | Review copy (PC `Posit`) | Archive sha256 | Wall | Peak RSS | Compatibility |
|---|---|---|---|---|---|
| MET-FAB | `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment_COHORT60RANKSR2/` | `a24f973dfb891458dab80474f0de110f231de65af8c24ceff65417d00d704ef1` | 59.9 s | 2.41 GB | 0.665 s |
| MET-INLB | `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_MAP_Experiment_COHORT60RANKSR2/` | `ccec2817ce20a2c86800e56192e6591cd95b0b58fd4ec9d158b3f51f0b0a4803` | 53.9 s | 2.41 GB | 0.815 s |

First checkpoint (`COHORT60RANKS`): FAB `a5b138dce4d7…`, 58.9 s, 2.40 GB; INLB `b7b1e06c7006…`, 53.6 s, 2.41 GB.

Both cover 60/60 cells and 600/600 windows, all four constructions, window sets `all` and `opening`, 793 groups, 62 rank
blocks (one per_cell block for `all`, one for `opening`, sixty per_cell_window blocks for `all`), statuses 720 computed /
60 `insufficient_peers` (the opening per_cell_window groups) / 13 `not_applicable` per condition, 8 report checks PASS
including `aggregation_archive_reads_back`. The wall time is dominated by the existing aggregation (33.6 s and 27.9 s)
and the loading of the 60 collections; the compatibility computation is the 0.65 s. The review's 37.9 s included the
raw-distance baselines that are outside production scope. Atlas indexes:
`<review copy>/compatibility/FAB/all/windows/index.md` and `<review copy>/compatibility/INLB/all/windows/index.md`
(five pages each, twelve cells per page). Complete copies of both R2 review folders are in `review_copies/`.

Candidate archive format 2: the v1 arrays, row order and identities unchanged (verified byte-identical to COHORT60PLOTS
for every candidate array, §4); added `compatibility_score` (NaN = missing), `compatibility_n_references`,
`compatibility_status`, `sgm_outside_prior_keys` (one entry per group) and, per block b,
`compatibility_block_<b>_group_index`, `_ranks` (NaN diagonal), `_reference_draws`; the manifest section
`compatibility` carries the protocol `sgm_distance_rank_v1` (strict comparison, metric and scale, ordinary median
reduction, availability rules, precision float64), the block map with condition, window set, construction and cell
context, the status counts and the zero count, bound to the manifest's target, contract, coverage and sources.
`aggregation_groups.csv` gains `compatibility_score`, `compatibility_status`, `compatibility_n_references`,
`sgm_outside_prior_keys`; `aggregation_compatibility.csv` holds the 12,480 off-diagonal pairs per condition. A 0.1.46
reader refuses a format-2 archive; the 0.1.47 reader accepts format 1 for uniform selections.

Spec example (illustrative, not a decision; the skeleton emitted by `--emit-representatives-template` writes every
component commented out with a `member_weighting = "uniform"` line and its availability):

```toml
[block]
posterior_sample_pool_choice = "empirical_representatives"
candidate_archive = "<reviewed format-2 aggregation_candidates.npz>"
candidate_archive_sha256 = "<its SHA-256>"
rationale = "<why this acquisition period, construction, weighting and any treatments were selected>"

[[component]]
source_experiment = "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A"
condition = "FAB"
window_set = "all"
grouping = "per_cell"
statistic = "sgm"
weight = 1.0
member_weighting = "posterior_rank_median"     # or "uniform" (the default when omitted)
```

## 3. Tests

Targeted first, then the complete suite once stable.

| Run | Result |
|---|---|
| `tests/test_experiment_compatibility.py` (new) | 13/13 (10 at the first checkpoint, 3 review regressions added) |
| `tests/test_experiment_aggregation.py` | 17/17 |
| `tests/test_selection_user.py` | 8/8 |
| complete repository suite (`full_suite.log`, after the corrections and the identity-guard follow-up) | 39 files, 324 PASS, 0 FAIL (324 test definitions; VALIDATION.md states 324 over 39) |

## 4. Full-data numerical oracle (`oracle_comparison.py`, `oracle_comparison.json`)

Against `Special_Analyses/POSTERIOR_RANK_WEIGHTING_REVIEW_2026-11-06` (`comparison_matrices.npz`, `candidate_weights.csv`,
method `median_rank`), from the regenerated `COHORT60RANKSR2` archives (and, identically, the first checkpoint's), in
explicit identity order; 24,960 off-diagonal ranks in all:

| Comparison | Entries | Result |
|---|---|---|
| FAB cells, `all` and `opening` (60 x 60 each) | 2 x 3,540 | exactly equal, diagonals masked on both sides |
| INLB cells, `all` and `opening` | 2 x 3,540 | exactly equal |
| FAB and INLB within-cell window matrices (60 x 10 x 10 each) | 2 x 5,400 | exactly equal |
| median scores, per_cell `all`/`opening` and per_cell_window `all`, both conditions | 6 vectors | exactly equal |
| normalized probabilities (per_cell: constructor against review; per_cell_window: constructor x 60 against the review's within-cell weights) | | max difference 5.6e-17; every cell carries exactly 1/60 under window weighting |
| effective representatives, median weighting | | FAB 58.487 / 56.645, INLB 49.084 / 35.493 (brief: 58.49 / 56.64, 49.08 / 35.49) |
| candidate arrays against COHORT60PLOTS | 14 arrays per condition | byte-identical |

No floating-point boundary moved any rank by a draw, so no affected pair is listed. The named facts of the brief hold:
INLB cell 34 windows 8 and 9 have zero median scores with mutual ranks 0.9995 and 0.9999 (their row means are 0.111,
which is what the mean rule would retain); cell 54 windows 0 and 1 have zero median scores; all-window cell weighting
keeps all 60 cells positive in both conditions; InlB opening cell weighting gives zero to cells 24, 49 and 58. The known
incomplete recording (INLB cell 34) was included unchanged; nothing was fixed, excluded or curated.

## 5. Source preservation

`source_hashes_before.txt` and `source_hashes_after.txt`: 136 files (the two adopted Experiment products, both
COHORT60PLOTS archives and manifests, the 120 retained draw collections, both SEL artifacts, the estimators) have
identical SHA-256 before and after the work. The independent review outputs were only read (their hashes were verified
against the brief's snapshot before starting).

## 6. Illustrative constructor report (not a production selection)

`illustrative_build/` (tag `ILLUSTRATIVERANK`, kept outside production names and outside `Posit`): MET-FAB, all ten
windows, one SGM per cell, `member_weighting = "posterior_rank_median"`, read from the regenerated R2 FAB archive through
the companion's dry-run and build paths (`illustrative_build.log`; rebuilt after the corrections, numbers unchanged). Result: 60 stored rows, 0 at zero probability,
probabilities 0.005738 to 0.019489 (equal-member 0.016667), effective rows 58.49 against the baseline 60.00,
outside-prior mass 8.9 % (the existing extrapolation treatment; `pool_mode = unrestricted`). Lowest: cell 49 (score
0.280, p = 0.00574), cell 52 (0.493, 0.01012), cell 41 (0.550, 0.01129); highest: cell 47 (0.950, 0.01949), cell 54
(0.943, 0.01934), cell 9 (0.941, 0.01932). Exact probabilities and scores per member in
`illustrative_build/*_Analysis/representatives_members.csv` and per stored row in `representatives_rows.csv`; the
report carries the selection table, the probability figure (labeled by source cell), the CDF comparison against the
equal-member baseline, the components view and the joint projections.

## 7. Intermediate copies replaced

The first regeneration pass produced both `COHORT60RANKS` copies with the cell-matrix figure's colorbar tick labels
printed over the score strip. I removed those two folders, which I had created minutes earlier in this task, fixed the
figure layout (matrix, strip, colorbar) and regenerated under the same label; all numbers are identical between the two
passes (`first_pass_logs/`, both oracle runs). After the review corrections the reports were regenerated under the fresh
label `COHORT60RANKSR2`; the `COHORT60RANKS` copies on the PC were kept. In this evidence folder, my own first-checkpoint
`review_copies/` and `illustrative_build/` were replaced by the R2 versions. No data product was deleted.

## 8. Review corrections (Codex, 2026-11-06) and their verification

| Finding | Correction | Verification |
|---|---|---|
| R1 zero-mass components crashed the ranked dry-run/report; the uniform case reported its members as positive | `_member_counts` reads the recorded positive/zero counts and renders an undefined effective count explicitly ("undefined (component mass 0: no member receives probability)"); zero masses stay valid and in provenance | `test_zero_mass_components_report_truthfully_in_the_dry_run_and_the_report` (both policies, dry-run console and report); Codex's probes: both cases complete, "60 / 0 / 60" |
| R2 override collisions emptied the row memberships | every member records its `stored_row` at consolidation (before overrides); the report maps memberships by that identity, checks that contributions add up to each row's probability (`representative_row_provenance`), and falls back to the exact-vector match only for a legacy record (disclosed as a failed non-fatal check) | `test_override_collisions_keep_every_stored_row_s_provenance`; Codex's probe: 60 stored rows, 0 missing memberships |
| R3 the reader accepted a split peer population, wrong reference draw counts and a conflicting protocol scale | `validate_archive_compatibility` derives every eligible context from the candidate groups and requires one block over exactly its peer population, draw counts equal to the referenced groups' pooled draws, the protocol scale equal to the archive's and the contract's, integer identities and counts (never coerced), `insufficient_peers` only for singleton contexts | `test_the_reader_requires_the_complete_reference_context_of_every_block` (split, draws, protocol and contract scale, fractional index and counts, missing block, extra block, foreign group, false insufficient); Codex's three probe archives refused; both valid cohorts read back unchanged |
| R4 the zero-median explanation was wrong | the subsection now says a zero median means a majority of the reference ranks are zero while some references may still rank the candidate high; DETECTOR_WORKFLOW §7.2 and the companion note say the same | regenerated reports (`COHORT60RANKSR2`) carry the wording; matrix values unchanged |
| R5 computed all-zero scores were advertised as selectable | `component_weighting_availability` applies the constructor's own normalization (`member_probabilities`) and reports "unavailable: … cannot be normalized" while the zero scores stay valid diagnostics | `test_missing_scores_are_not_zeros_and_the_constructor_refuses_them` (per_cell and per_cell_window levels); Codex's probe shows the new text |
| R3 follow-up (re-review, `REREVIEW_R2/REVIEW.md`): the rewritten validator accepted duplicated cell or window identities | the scientific identities are validated before the contexts are derived from them: distinct cells per per_cell context (pooled groups carry chunk -1), distinct window positions per per_cell_window context | `test_the_reader_requires_the_complete_reference_context_of_every_block` gained the two mutations and the chunk convention; Codex's identity probe, copied unchanged (`codex_identity_regression_probes.py`), refuses both archives (`identity_regression_reproduction.json`); the valid R2 cohorts, the first-label copies and the format-1 COHORT60PLOTS archives still read; oracle exact; no report regeneration (validator only) |
| presentation | selection-table header `statistic`; `representatives_rows.csv` carries `<key>_stored_log10` and `<key>_physical`; the probability plot of a single per-cell selection is drawn in numeric cell order (view only); this note's oracle count corrected to 24,960 | illustrative build and tests |

Codex's probe script was copied unchanged into this folder (`codex_review_probes.py`) and rerun here after the
corrections (`review_probes.json`, `codex_review_probes.log`, `probe_archives/`, the zero-component and
override-collision CSV folders), so nothing was written into Codex's review folder.

## 9. Scientific choices still open (for the lead)

- the acquisition period (window set: `all`, `opening`, or a named set);
- the construction or mixture of constructions and the component masses;
- whether to enable consensus weighting (`posterior_rank_median`) for a component, in the knowledge that it changes the
  representation of acquisition batches (the review's finding) and gives zero mass to some valid recordings under the
  opening set;
- any parameter-specific treatment (the bleaching override remains a separate recorded decision);
- the statistic (SGM by default; a marginal-median component is uniform only).

Nothing in this checkpoint selects any of them. Codex reviews this checkpoint; commit, push, fleet sync and the verified
workspace backup follow the lead's word.
