# Rank coordinate spaces in Experiment and the constructor's `rank_coordinates` (0.1.48, 2026-11-09)

Implementation record of the lead's decision of 2026-11-09 to rank the SGM representatives over the four retained
coordinates (mu_r, sigma_r, mu_pc, sigma_pc) beside the all-coordinate ranks, to document the accepted outside-prior
mass of the proposed InlB selection, and to make the probability figure readable for 600 rows. The scientific
comparison of the two weightings is reviewed in `Special_Analyses/NUISANCE_DLI_PRODUCTION_PROPOSAL_2026-11-09/round2/`
(outside the repository); this folder holds the numerical evidence of the code change.

## 1. What changed

- Experiment: `--compatibility-coordinates LABEL=key,key[;...]` computes additional rank coordinate spaces over the same
  groups, reference SGMs and reference clouds projected onto the named coordinates (scale = their prior widths), with
  their own figures, atlas, CSV and archive arrays; candidate archive format 3 (format-2 arrays unchanged); the reader
  validates every space (formats 1, 2, 3 accepted).
- Constructor: `[[component]].rank_coordinates` selects the space by its keys (default every coordinate); the record and
  the report state whether the ranked coordinates are disjoint from the overridden ones.
- Figure: a cell-by-window grid of the stored probabilities for a selection of one SGM per cell and window.
- Full description: CHANGELOG 0.1.48; DETECTOR_WORKFLOW.md sec. 7.2/7.3; the constructor's companion document.

## 2. Regenerations (CPU, PC)

Both full-cohort reports regenerated under the review label `COHORT60RANKSR3` with the space
`spatial_brightness = mu_r, sigma_r, mu_pc, sigma_pc` (`regenerate_<COND>_COHORT60RANKSR3.{log,time}`; complete copies in
`review_copies/`): FAB 68.4 s wall / 2.40 GB RSS, INLB 60.6 s / 2.41 GB; 8 checks PASS each; 62 blocks per space;
statuses per space 720 computed / 60 insufficient_peers / 13 not_applicable.

## 3. Independent oracle

`independent_rank_oracle.py` recomputes every rank of every space by plain distance counting from the 60 retained draw
collections per condition (no call into the package's rank primitive) and compares exactly
(`independent_rank_oracle_<COND>.json`): all 12,480 off-diagonal ranks per space and condition equal, every stored score
equal to its row median, and the primary (`all`) arrays byte-identical to the `COHORT60RANKSR2` archives. Zero scores:
FAB `all` 0 and `spatial_brightness` 1; INLB `all` 7 and `spatial_brightness` 36.

## 4. The two weightings compared

`compare_spaces.py` on the review builds of the proposed selection (`comparison_<COND>.{json,md}`; the builds themselves
are in the Special_Analyses folder): effective rows 572.5 -> 483.6 (FAB) and 507.7 -> 380.1 (INLB) from the six- to the
four-coordinate space; zero rows 0 -> 1 and 4 -> 24 (13 of them InlB's opening window); weighted marginal medians within
a percent of each other; Spearman correlation of the row probabilities 0.80 / 0.83. The four-coordinate ranks are
stricter because the two excluded coordinates are the wide, poorly recovered ones that inflate every reference radius in
the primary space. Not a selection.

## 5. Tests and preservation

`tests/test_experiment_compatibility.py` 17/17; full suite in `full_suite.log` (39 files, 329 PASS, 0 FAIL, after the review corrections); the 136
numerical source products unchanged (`source_hashes_check.log`, SHA-256 against the 0.1.47 list).

## 6. Review corrections (Codex, 2026-11-09)

- R1: `component_weighting_availability` evaluates every rank space separately through the constructor's own
  normalization with that space's coordinates (one clause per space); regression added (usable primary space with an
  unusable subset, the reverse, missing scores).
- R2: `COMPATIBILITY_COORDINATES` is forwarded by both HPC Experiment launchers (one quoted argument to worker and merge)
  and carried by both submitters through the exported environment via `ALL`; the fake-scheduler forwarding tests cover
  two named spaces and the knob's absence. No HPC job was run.
- Presentation: the probability figure's subtitle names the rank space and its coordinates.
- Codex's probes rerun after the corrections (`codex_review_probes_rerun/`): the subset space's refusal is reported with
  its reason, and both launchers forward the option.
- R3 (the flicker override of the proposed selection) is a scientific decision recorded in the Special_Analyses folder
  (`round3/`), not a repository change.
