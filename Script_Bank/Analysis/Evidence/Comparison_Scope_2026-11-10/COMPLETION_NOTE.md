# Completion note: aggregation populations, nuisance weights and exact marginal plots

Written for Codex as reviewer and for the lead. Date: 2026-11-10. It answers the handoff
`Special_Analyses/NUISANCE_DLI_PRODUCTION_PROPOSAL_2026-11-09/CLAUDE_HANDOFF_WEIGHTING_REVISION_2026-11-10.md`.
Repository: `srm-and-sbi-monomer-dimer-alp`, baseline `532c943` (0.1.48); the changes below are in the working tree as
0.1.49 and are **not committed**.

## 0. Status in five lines

1. Biology generation is on hold. Nothing was ever submitted on Goethe (the queue was empty when checked today and no
   `sbatch` was issued in this work); no job was cancelled and no product deleted.
2. The construction-wide comparison scope is implemented in Experiment, the archive, the constructor and the reports;
   the hierarchical scope is kept under the explicit name `within_cell`.
3. Both complete-cohort Experiment reports are regenerated under the fresh review label `COHORT60RANKSR4`, and they
   reproduce the exploratory calculation exactly (every rank, every score; probabilities to 2e-18).
4. Revised nuisance artifacts exist as isolated review products (tag `PROPOSALCW4`, outside the Posit directory).
   `RANK4`, `SEL`, the `COHORT60RANKSR3` archives and every earlier report are byte-identical to their recorded
   checksums.
5. The existing trajectory tiers are inventoried (section 8). Their staging route is undecided and needs the lead.

## 1. What changed (files)

| File | Change |
|---|---|
| `srm_and_sbi_monomer_dimer_alp/experiment_aggregation.py` | comparison scopes `construction` / `within_cell` (`SCOPES_BY_CONSTRUCTION`, `REFERENCE_POPULATIONS`, `rank_scored_constructions`); `compute_compatibility(..., scope=)` and `compute_rank_spaces(..., scope=)`; protocol `sgm_distance_rank_v2`; archive format 4; `archive_comparison_scope`; scope-aware context derivation and validation; `archive_component` returns the scope of the component's scores; `weighted_histogram`, `marginal_histograms`; the report subsection and atlas rewritten for the construction-wide scope |
| `srm_and_sbi_monomer_dimer_alp/detector_nuisance_dli.py` | `[[component]].rank_scope` (required with `posterior_rank_median`), validated against the archive; complete-component normalization; record fields; availability statements name the scope; template text |
| `srm_and_sbi_monomer_dimer_alp/visualization_aggregation.py` | `figure_compatibility_population` (new), `figure_compatibility_cells` generalized to window positions, `figure_compatibility_windows` relabeled, `figure_member_probability_grid` shows stored row probabilities and cell totals, `figure_weighted_histograms` (new) |
| `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py` | exact marginal summary and histograms for weighted empirical artifacts, sampler check separated, scope in the console line and the selection table, cell-total table and CSV, grid caption |
| `srm_and_sbi_monomer_dimer_alp/experiment_runner.py` | console header and help text name the scope (no interface change, so no launcher or submitter change) |
| `tests/test_experiment_comparison_scope.py` (new), `tests/test_experiment_compatibility.py` | section 5 |
| `DETECTOR_WORKFLOW.md` (7.2, 7.3, a status paragraph under the `RANK4` record in 7.6), `PROJECT_CONTEXT.md`, the constructor companion note, `VALIDATION.md`, `CHANGELOG.md`, `pyproject.toml`, `__init__.py` | documentation and the PATCH bump to 0.1.49 |

Not changed: the SGM calculation, the pooling and its weights, the candidate vectors and identities, the rank
primitive (`reference_distance_ranks`: strict `>`, float64), the ordinary off-diagonal median, the rank coordinate
spaces, the sampler, the artifact format, any estimator, prior, camera, labeling law or reaction-diffusion model.

## 2. The interface chosen (the smallest one I found consistent with the handoff)

- **One comparison scope per archive, declared.** Format 4 records `compatibility.comparison_scope` and the reference
  scope of each rank-scored aggregation (`comparison_scopes`); every protocol record repeats them and every block
  names its `scope`. The Experiment stage always computes `construction`; it has no option for the other scope.
- **Reference populations under `construction`,** per condition and window set, derived from the groups:
  one block over the included cells (unchanged population), one over the selected window positions (newly scored;
  each reference is the position's equal-cell cloud mixture), one over every cell–window group (the other windows of
  the candidate's own cell included; only the candidate's own group identity excluded). The dataset representative
  stays `not_applicable`. A population of one group is `insufficient_peers`.
- **Legacy archives keep their recorded meaning.** Formats 2 and 3 carry protocol `sgm_distance_rank_v1` and no scope
  field; the reader interprets them as `within_cell` and refuses a scope field in them. A construction-wide archive
  relabeled as a legacy format is refused (its blocks are not that format's contexts). The engine can still compute
  `within_cell` by name (`compute_compatibility(..., scope="within_cell")`); the existing tests use it.
- **Constructor.** `rank_scope` must be `"construction"` or `"within_cell"` and must equal the scope the archive
  computed that grouping's scores under; otherwise the build is refused with the archive's format, protocol and scope
  named. `"construction"`: `p_i = weight * s_i / sum of the component's scores` for `per_cell`,
  `per_window_position` and `per_cell_window`, with the additional check that every member's score was computed
  against exactly the other `n - 1` members. `"within_cell"`: the hierarchical rule, `per_cell_window` only. Uniform is
  unchanged and is the only policy for the dataset representative. Mixtures normalize inside each component and
  multiply by the explicit mass, as before.
- **Zero handling.** A zero score is zero probability. Under `construction` a cell whose members all score zero is
  valid (`zero_mass_cells` in the record, a red marker in the grid, a row in the cell-total table); an all-zero
  component is refused; nothing is floored, clipped or replaced by uniform.

Two consequences to judge (section 9, items 1 and 2): a specification written before the field existed is refused
until it names its scope, and one SGM per cell is accepted as `"construction"` from a legacy archive as well, because
that archive's per-cell scores are construction-wide by its own record (the scores are identical in both formats).

## 3. Complete-cohort Experiment reports (review label `COHORT60RANKSR4`)

Paths (PC data bank, `Posit`):

- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment_COHORT60RANKSR4/`
  (archive sha256 `aa5a668bc86030dfcaa0eb67efcab038855a7293ea8d1b285e3315cf9a95ecd8`)
- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_MAP_Experiment_COHORT60RANKSR4/`
  (archive sha256 `179617f2665504a9e0c1662bc6c135279833bb6d056d841bc00cdfedffb91b6f`)

Copies: `review_copies/` in this folder. Regeneration: `REGENERATION_RECORD.md` (CPU, about 70 s each).

What the subsection "posterior compatibility of SGM representatives" now contains, per condition, window set and rank
space (`all` and `spatial_brightness`):

- the statement of aggregation, comparison scope, coordinate space and where the normalization happens;
- the table "aggregation, comparison scope and availability" (60 candidates / 59 references; 10 / 9; 600 / 599; the
  dataset representative not applicable; under `opening`: 60 / 59 for cells and for cell–window groups, one position
  unavailable);
- `aggregation_<COND>_<set>_compatibility_cells[_<space>]`: the cell matrix (unchanged);
- `..._compatibility_positions[_<space>]`: the 10 x 10 window-position matrix (new);
- `..._compatibility_population[_<space>]`: the cell–window population as ONE 600 x 600 matrix, cells delimited and
  labeled, with the strip of row medians over the other 599 groups (new);
- `..._compatibility_windows[_<space>]`: the same population-wide scores arranged by cell and window start;
- `compatibility/<COND>/<set>/windows[_<space>]/`: the atlas, now titled and captioned as the same-cell blocks of
  the population matrix, with the statement that these blocks are 9 of each candidate's 599 references and are not
  the candidate's score.

## 4. Reproduction of the exploratory calculation and independent checks

`verify_construction_wide.py` -> `verification_construction_wide.json` (all assertions passed, both conditions):

| Check | FAB | INLB |
|---|---|---|
| Groups, SGM and median vectors, identities, memberships, methods and sources identical to the `COHORT60RANKSR3` archive | yes (24 arrays, 60 source checksums, contract, target) | yes |
| Rank matrices identical to the exploratory ones (per cell 60 x 60, per position 10 x 10, per cell–window 600 x 600) | identical, all three | identical, all three |
| Scores identical to the exploratory ones; constructor probabilities against the exploratory ones | identical; max difference 0 | identical; max difference 2e-18 |
| Effective representatives / zero rows / opening-window mass (cell–window, four coordinates) | 427.30 / 1 / 3.18 % | 317.97 / 64 / 4.88 % |
| Per cell: effective / zeros | 52.23 / 0 | 45.53 / 0 |
| Per window position: effective / zeros | 9.07 / 0 | 9.38 / 0 |
| The 5,400 earlier within-cell entries equal the same-cell blocks of the new matrix, in both rank spaces | yes | yes |
| Per-cell scores equal those of the previous archive, in both rank spaces | yes | yes |
| Opening set: per-cell scores equal cell–window scores; the single position is `insufficient_peers` | yes | yes |
| Plain-counting oracle from the retained clouds (no sort, no lookup): 300 + 90 + 1,500 pairs per rank space | 0 mismatches in 3,780 | 0 mismatches in 3,780 |

The oracle rebuilds each reference pool from the group's recorded members and the source collections, so it also
confirms the equal-window and equal-cell mixtures and the pooled draw counts.

## 5. Tests

- New: `tests/test_experiment_comparison_scope.py`, 8 tests (direct-counting ranks over the complete populations of
  the three aggregations with identity exclusion; same-cell peers inside the population, complete-component
  normalization, scope explicit and validated both ways, normalization alone shown insufficient; window-position
  ranks, singleton and dataset handling, the opening equivalence; a zero-mass cell valid and never sampled against an
  all-zero component refused; the format-4 scope records, thirteen malformed-scope refusals, legacy read-back, the
  earlier guards; explicit masses, consolidation, overrides, baseline, artifact round trip, whole-vector sampling, a
  record written before the scope field; exact histograms; the companion report with seed-independent exact output).
- `tests/test_experiment_compatibility.py` (18): the earlier tests describe the hierarchical policy, so they now
  request it by name (`scope="within_cell"`, `rank_scope`), and their legacy fixtures are produced by rewriting the
  manifest into formats 2 and 3 with the `v1` protocol. Assertions I changed: the archive format number (3 -> 4, the
  unknown version 4 -> 5), the report test (new table rows, figures, captions, atlas wording, block list of the
  written archive), the availability texts (they now name the scope), one message ("every score of this component is
  zero"), the grid caption and the member CSV columns. No assertion was weakened to pass.
- Forwarding tests: unchanged, because no command-line or launcher interface changed.
- Full suite on the final code: 337 PASS lines over 40 files, no failure, every file's PASS count equal to its number of tests, every exit status 0 (2026-11-10 11:53 to 12:04; `full_suite.log`; `VALIDATION.md` inventory 337 over 40 files).

## 6. Exact marginal plots and the probability grid

- `agg.weighted_histogram(values, weights, edges)`: `H(B) = sum_i p_i 1[v_i in B]`, bins half-open with the last
  closed; a value outside the bins is refused. `agg.marginal_histograms(...)`: per parameter 30 equal-width bins in
  the stored coordinate (uniform in log10 for a log-scaled parameter) from the smallest to the largest stored value
  over all stored vectors, zero-probability vectors included; the same edges for the selected distribution and the
  equal-member baseline; a single-valued coordinate is a point mass.
- The constructor report's `nuisance_marginals` figure and its "Per-parameter marginal summary" are computed from the
  stored vectors and probabilities for weighted empirical artifacts. The 10,000 draws remain only as a sampler check
  ("every draw is a stored row of positive probability"). The test runs the report twice with different global random
  states and draw counts and requires identical bin masses and identical summary rows.
- The figure: physical units, logarithmic axis stated in the axis label, y = probability mass per bin, blue bars =
  selected probabilities, dark outline = equal-member baseline in the same bins, bleaching and flicker drawn and
  labeled as point masses fixed by explicit overrides, prior bounds and the `SEL` reference as annotations. The bin
  masses are exported (`nuisance_marginal_histograms.csv`). The exact weighted CDFs (`representatives_marginals`) are
  unchanged.
- The cell–window grid shows the stored global row probabilities (one color scale for the grid, which sums to one),
  with each cell's total against the equal-member value, a table of the totals and `representatives_cell_totals.csv`.
  The earlier "share of the cell's mass" display is gone; the text no longer states that cell totals must be equal.

Scope of the exact path: `empirical_representatives` artifacts. The continuous constructions (box, Gaussian) and the
unweighted sample pools keep the sampled histogram (section 9, item 5).

## 7. Revised nuisance artifacts (isolated review products, tag `PROPOSALCW4`)

Folder: `Special_Analyses/NUISANCE_DLI_PRODUCTION_PROPOSAL_2026-11-09/WEIGHTING_REVISION_2026-11-10/` (copy in
`review_builds/`). Not in `Posit`; the biology loader cannot select them by tag from the data bank.

- `revision_build.py`: derives the review spec from the adopted `RANK4` spec text (archive -> `COHORT60RANKSR4`,
  `rank_scope = "construction"`, a new rationale; the two override tables verbatim, asserted), runs the companion's
  own dry-run and build, loads the artifact through `require_nuisance_dli` pointed at the review folder, draws 20,000
  vectors, and compares with `RANK4`, the equal-member baseline and the exploratory probabilities.
- `<COND>/..._PROPOSALCW4_Nuisance_DLI{.npz,_Spec.toml,_Analysis/}`; `comparison_<COND>.json`; `revision_build_<COND>.log`.
- Artifact sha256: FAB `47201f303ee1b0b1…`, INLB `3eb66dc85238edeb…` (full values in `comparison_<COND>.json` and
  `SHA256SUMS`).

Verified by the script: the 600 stored vectors and their cell and window identities equal those of `RANK4` row by
row; the overrides are identical (bleaching 0.05; flicker 5.1 per second FAB, 4.7 INLB); the ranked coordinates
exclude every overridden coordinate; every member has 599 references; the probabilities sum to one and equal the
exploratory ones; no zero row is ever drawn and the drawn frequencies follow the probabilities (largest deviation
0.0013); the `RANK4` artifact checksum is unchanged after the build.

| | FAB: revised | FAB: `RANK4` | FAB: equal members | INLB: revised | INLB: `RANK4` | INLB: equal members |
|---|---:|---:|---:|---:|---:|---:|
| Effective representatives (of 600) | 427.30 | 483.55 | 600 | 317.97 | 380.05 | 600 |
| Zero-probability rows | 1 | 1 | 0 | 64 | 24 | 0 |
| Opening-window mass | 3.18 % | 1.27 % | 10 % | 4.88 % | 1.16 % | 10 % |
| Smallest / largest cell total | 0.035 % / 3.04 % | 1.667 % each | 1.667 % each | 0.067 % / 4.54 % | 1.667 % each | 1.667 % each |
| Cells with zero total | 0 | 0 | 0 | 0 | 0 | 0 |
| Effective cells (1 / sum of squared cell totals) | 50.4 | 60 | 60 | 43.4 | 60 | 60 |
| Mass outside the training prior (any coordinate) | 12.3 % | 17.3 % | 19.5 % | 96.9 % | 93.9 % | 94.7 % |
| ... of which `mu_r` / `mu_pc` / `sigma_pc` | 0.2 / 12.0 / 0 % | 3.7 / 13.7 / 0 % | 7.7 / 12.0 / 0 % | 96.9 / 3.5 / 2.2 % | 93.9 / 18.2 / 6.6 % | 94.7 / 20.5 / 7.8 % |
| PSF width `mu_r`, median [5 %, 95 %] (px) | 1.709 [1.619, 1.855] | 1.721 [1.612, 1.957] | 1.738 [1.614, 2.016] | 2.223 [2.019, 2.334] | 2.254 [1.963, 2.337] | 2.254 [1.973, 2.338] |
| Brightness `mu_pc`, median [5 %, 95 %] (photons) | 104.2 [98.4, 112.6] | 104.3 [98.1, 113.8] | 104.9 [98.1, 118.6] | 147.8 [103.1, 212.4] | 143.2 [67.1, 381.0] | 147.8 [65.1, 468.7] |
| Total-variation distance to `RANK4` | 0.247 | - | - | 0.408 | - | - |

Descriptive remarks, not recommendations:

- The revised weights narrow the marginal ranges, most visibly the InlB brightness range (5 % to 95 %: 103 to 212
  photons against 65 to 469 at equal weights). The medians move little.
- Mass by window start (percent, ten windows in order). FAB revised: 3.2, 8.5, 10.2, 11.0, 10.6, 11.0, 10.6, 11.4,
  12.0, 11.5. INLB revised: 4.9, 7.3, 7.1, 8.2, 9.4, 10.9, 12.9, 13.0, 12.2, 14.1. The opening window stays
  down-weighted under the population-wide scope, less than under `RANK4`; for InlB the mass rises along the
  recording.
- InlB zero rows: 17 are zero under both policies, 47 only under the revised one, 7 only under `RANK4`.
- Lowest cell totals: FAB cells 49 (0.035 %), 52, 39, 14; InlB cells 53 (0.067 %), 52, 59, 54. No cell is at zero.
- InlB: the mass outside the training prior rises from 93.9 % to 96.9 % on `mu_r` and falls on brightness; the
  outside-prior values are kept and sampled as before, with no clipping.

## 8. Existing trajectory tiers

`Special_Analyses/NUISANCE_DLI_PRODUCTION_PROPOSAL_2026-11-09/TIER_INVENTORY_2026-11-10/INVENTORY.md`.

- The current-model 2 s tiers exist on the JSC filesystems, readable from the JUPITER login nodes: per condition 200
  TRAIN + 50 TEST (scratch) + 25 EVAL (project) trajectory folders and the same 275 `Theta_Set` records, task
  identities matching and contiguous. Eight records read: `generator = rds`, `package_version = 0.1.42`,
  `schema_version = 1`, written 2026-10-21. These are the tiers of the completed campaign of section 9.10, not the
  superseded 2026-10-14 tiers.
- The Goethe preflight note named 2026-10-14; that was my error and is corrected in the note.
- Not done: the per-folder file count for all 550 folders, the DLI dry-run against staged tiers, and the staging
  itself. The documented Goethe-initiated pull is refused by JUPITER (Goethe's key is registered for JUWELS only); a
  JUPITER-initiated push needs a new key authorized on Goethe. I created no credential. At the documented rate the
  copy of 1.28 TiB takes about eight hours, which the lead found too long; the tier's own timestamps give 0.8 to 0.9 h
  (FAB) and 1.7 to 1.8 h (INLB) per regenerated task. Reuse or regeneration is the lead's decision.
- Goethe: the successor package is not installed in the environment there (the launcher would fail at import);
  `pip install -e . --no-deps` has not been run.

## 9. Unresolved items and decisions needed

1. **Old specifications.** A `posterior_rank_median` component without `rank_scope` is refused. The adopted `RANK4`
   spec files therefore need the line `rank_scope = "within_cell"` to be rebuilt with 0.1.49 (their artifacts load
   unchanged). The alternative, an implied scope for legacy archives, would keep old specs working at the price of an
   implicit rule. I chose the explicit refusal.
2. **Per-cell scope from a legacy archive.** Accepted as `"construction"`, because the legacy per-cell population is
   the complete construction and the scores are identical. If the reviewer prefers that construction-wide weighting
   always require a format-4 archive, it is a one-line restriction.
3. **No Experiment option for `within_cell`.** The hierarchical scope is computable through the engine and readable
   from formats 2 and 3 only. If it should remain producible from the command line, that needs a new option and its
   forwarding through four launcher and submitter scripts.
4. **Pairwise CSV size.** `aggregation_compatibility*.csv` now hold 359,400 rows for the cell–window population, about
   25 MB per rank space and condition. They are written as before; the evidence copies leave them in the data bank.
5. **Unweighted sample pools.** Their report keeps the sampled histogram in log10 units. Extending the exact path to
   them is straightforward if wanted.
6. **Histogram axis range.** Bins span the stored support, but the prior-bound annotations extend the axis, which
   compresses a narrow distribution (FAB brightness). The bins themselves are not affected.
7. **Lost console logs.** The logs of the two successful regenerations were overwritten by a refused second
   invocation (`REGENERATION_RECORD.md`). The products are intact and re-validated.
8. **Housekeeping for the lead.** Nothing is committed, pushed, synced or backed up for this revision. The
   `COHORT60RANKSR4` copies (65 MB each) and the `PROPOSALCW4` builds are review products whose retention or deletion is
   the lead's decision. The proposal folder's top-level `SHA256SUMS` predates today's additions; today's subfolders
   carry their own.
9. **Adoption is a separate step.** A production tag, a build into `Posit`, the decision record in section 7.6 and
   the resumption of the biology campaign all wait for the review and the lead's decision. This note does not
   request them.

## 10. Refinements after the review (2026-11-10)

Codex's review (`WEIGHTING_REVISION_2026-11-10/CODEX_REVIEW/REVIEW.md`, copy in `review_builds/CODEX_REVIEW/`) passed
the numerical implementation and asked for two refinements. Neither changes a score, a probability, a bin edge, a bin
mass or a stored vector.

- **R1, histogram visibility.** `figure_weighted_histograms`: every binned panel now shows the full support of the
  stored vectors (first to last bin edge, plus 4 % of the span on each side in the stored coordinate;
  `HISTOGRAM_VIEW_MARGIN`); no tail is cropped. A prior bound or the reference that lies inside the view is a line; one
  that lies outside is named with its value by an arrow label at that edge of the panel and no longer widens the
  axis. A point-mass panel keeps the prior range as its display range, and its annotation is placed in axes
  coordinates, inside the panel. The legend is built from fixed entries that explain both forms. The test checks the
  view limits against the exported edges, the arrow labels for off-view annotations, lines for in-view ones, the
  point-mass text placement, and that the drawn bars are the exported masses.
- **R2, legacy documentation.** The constructor companion note has a table "Comparison scope and archive format"
  (the scope is a property of the recorded comparison population: per-cell scores are `construction` in an archive of
  either scope; cell–window and window-position construction-wide weights need an archive of the `construction`
  scope) and a paragraph on specifications written before `rank_scope`, with the one-line migration shown for a COPY
  of an adopted `RANK4` specification; the stored specifications are not edited. `DETECTOR_WORKFLOW.md` 7.2 and the
  changelog carry the same statements in short form.
- **Rebuilt review reports.** The `PROPOSALCW4` builds were repeated so that their reports carry the refined figure.
  The artifact files are byte-identical to the reviewed ones (FAB `47201f30…`, INLB `3eb66dc8…`), the comparison
  results are unchanged and the exported bin masses are byte-identical to the reviewed export. The reviewed state of
  the comparison files, logs, checksums and the two reviewed marginal figures is kept in
  `WEIGHTING_REVISION_2026-11-10/first_review_state/`.
- **Inventory wording.** The tier inventory now says "complete task-name inventory; content completeness still to be
  checked", as asked.
- **Tests after the refinements.** The complete suite was run again on the refined code: 337 PASS lines over 40 files, no failure, every file's PASS count equal to its number of tests, every exit status 0 (2026-11-10 12:35 to 12:46; `full_suite.log`; the run before the refinements is kept as `full_suite_before_review_refinements.log`).
- Not added, as the review agreed: an Experiment option for `within_cell`, an exact path for unweighted pools, a new
  report regeneration for the lost logs.

Pilot staging of the existing tiers and the Goethe preflight are recorded separately
(`Special_Analyses/NUISANCE_DLI_PRODUCTION_PROPOSAL_2026-11-09/goethe_transfer/PILOT_STAGING_2026-11-10.md`).
