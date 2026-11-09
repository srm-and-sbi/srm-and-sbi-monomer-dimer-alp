# Production selection RANK4: the empirical_representatives Nuisance_DLI of both conditions (2026-11-09)

Decision record: DETECTOR_WORKFLOW.md sec. 7.6, "The production `Nuisance_DLI` decision record (2026-11-09): `RANK4`". The
review that led to it: `Special_Analyses/NUISANCE_DLI_PRODUCTION_PROPOSAL_2026-11-09/` (outside the repository; three
rounds, the final-wording review builds in `round3/final/`). This folder is the construction record.

## Selection

One SGM representative per cell and 2 s window over all ten windows (600 vectors per condition), equal total mass per
cell, within-cell `posterior_rank_median` weights over `mu_r`, `sigma_r`, `mu_pc`, `sigma_pc` (`rank_coordinates`; archive
rank space `spatial_brightness`), `prob_photo_bleach` overridden to 0.05, `lambda_rate` overridden to 5.1 (FAB) and 4.7
(INLB) per second as empirically anchored working overrides from the localization-table derivation, with their
qualifications in the spec reasons. Candidate archives: the `COHORT60RANKSR3` review copies (format 3) of the adopted
detector estimators' complete-cohort Experiment products.

## Construction (CPU, PC)

1. Production specs written into the data bank from the final-wording proposal specs (header replaced, rationale labeled
   as the production selection; everything else identical): `production_copies/*_RANK4_Nuisance_DLI_Spec.toml`.
2. Companion CLI dry-runs (`--build --dry-run`): `dry_run_<COND>.log`.
3. Companion CLI builds (`--build`): `build_<COND>.log`; 6 checks PASS each; artifacts
   `..._<COND>_2S_50FPS_RANK4_Nuisance_DLI.npz` in the data bank (`production_artifact_sha256.txt`); reports and figures in
   `production_copies/..._RANK4_Nuisance_DLI_Analysis/`.
4. Loader validation through the biology require-gate (`require_nuisance_dli(..., nuisance_tag="RANK4")`) against the
   final-wording proposal artifacts: `verification_RANK4.json`.

| condition | rows | zero rows | effective rows | bleaching | flicker (per second) | outside-prior mass | artifact sha256 |
|---|---|---|---|---|---|---|---|
| FAB | 600 | 1 | 483.55 | 0.05 | 5.1 | 17.3 % | 3b89edab3bb4… |
| INLB | 600 | 24 | 380.05 | 0.05 | 4.7 | 93.9 % | 50a7678ea858… |

Verified for both conditions: the 600 row probabilities and the stored vectors are identical to the reviewed final
proposal; the overrides, the archive checksum and the rank coordinates match; the probabilities sum to one; every sampled
row is a stored row and no zero-probability row is ever drawn; the ranked coordinates are disjoint from the overridden
ones (`all_ok`: True).

## Unchanged

The `SEL` artifacts, their specs and their decision record; the candidate archives; the estimators; the priors. The
biology generation selects this artifact with `--nuisance-tag RANK4` (dispatcher `NUISANCE_TAG=RANK4`) and remains a
separate step.
