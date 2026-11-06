# Retained joint posterior draws of experimental windows — usage and interpretation

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Draws.py`. The adopted Experiment
products keep three point summaries and the marginal quantiles of 10,000 draws per window; the draws were not stored
and cannot be reconstructed from their quantiles. This companion draws a fresh cloud for every window of selected
recordings with the same trained estimator, input conversion, normalization, window geometry and sampling mode as the
source Experiment, and preserves every complete draw. It is a collection step and a joint-structure diagnostic. It
selects nothing and changes no product or artifact.

## Route

A collection companion that calls the engine's own functions, not a rerun of the Experiment stage (whose
`--artifact-tag` selects both the estimator and the output directory, so a rerun would write into the adopted
product's directory). Reused as is: `experiment_support.read_cell_chunks` (TIFF layout, the fixed 16-bit to 8-bit
conversion, windowing), `artifacts.load_estimator` with the parameter-key guard, `evaluation.posterior_summary` with
`return_samples=True` and `return_sgm=True`, `evaluation.sample_geometric_median` with the prior-width scale
(`evaluation.prior_scale`). No MAP is re-optimized: the adopted MAP exists and is joined as a reference.

## What it stores

`<alias>_<COND>_<timing>[_<TAG>]_Window_Posterior_Draws[_<LABEL>].npz`, a collection with its own manifest
(`collection_manifest_json`, stage `window_posterior_draws`), not an Experiment-stage product:

| field | shape | meaning |
|---|---|---|
| `posterior_samples_cloud` | windows x draws x 6, float32 | every complete joint draw, estimator coordinates (log10); rows are never shuffled, rank-matched or resampled per column |
| `kind_index`, `cell`, `chunk`, `kinds` | windows | the window identities (chunk w = frames 100w to 100w+99 at the 2 s window) |
| `fresh_quantiles`, `fresh_sgm`, `fresh_sgm_index` | windows x 6 x 5, windows x 6, windows | summaries computed FROM THE RETAINED CLOUD; the SGM is the cloud row at its index |
| `draws_outside_prior` | windows | count of draws with any coordinate outside the prior box (retained, never rejected) |
| `source_map_estimate`, `source_posterior_quantiles`, `source_posterior_sgm`, `source_scores`, `source_row_index` | joined | the source product's references, matched by `(kind_index, cell, chunk)` |
| `prior_low`, `prior_high`, `sgm_scale`, `quantile_levels`, `parameter_keys` | | the box, the SGM scale, the levels, the order |

The manifest records the sampling mode and draw label, the loaded weights checksum (verified against the one the
source product recorded, before any sampling), the source product's path and checksum, the reference artifact, the
seed policy, the device, library versions, seconds per window, peak GPU memory, the command line and the code
provenance (implementation hash at startup and at write).

## What it examines

Per window: Pearson and Spearman correlation matrices in estimator coordinates and Pearson in physical coordinates
(`correlations.csv`); for chunks 0, 4 and 9, joint plots of `mu_r` x `mu_pc`, `mu_pc` x `sigma_pc`, `mu_r` x
`sigma_r`, `sigma_pc` x `lambda_rate`, `prob_photo_bleach` x `lambda_rate` and `mu_pc` x `prob_photo_bleach` on
physical axes (contours of the full cloud; the scatter is a fixed-seed subset, display only); the marginal densities of
the ten windows; the within-window correlation heat maps. `joint_depth.csv` places the reference vector, the source
MAP, the source SGM and the fresh median vector relative to each cloud: the prior-scaled distance to the fresh SGM and
the share of draws strictly farther from the SGM than the reference (a distance rank relative to the SGM: near 1 when most
draws lie farther than the reference, near 0 when almost all lie closer; not a posterior-density or credible-region
statement), beside the marginal membership (inside every 90 % interval). The rank is the shared definition
`experiment_aggregation.reference_distance_ranks` (strict comparison, prior-width scale, float64), the same rank the
official Experiment report uses in its posterior-compatibility subsection; the two differ in what is compared: this
companion places fixed reference vectors (SEL, the source MAP, the source SGM, the fresh median vector) in one window's
cloud, while the Experiment report compares the SGM representatives of groups with the clouds of their peer groups and
reduces each row to its median score. This companion's `joint_depth.csv` is a diagnostic of one collection, not the
official aggregation reporting. `window_summaries.csv` carries the fresh and the
source summaries side by side with their differences in dex, which measure Monte Carlo reproducibility, not a change
of the posterior.

## Reading rules

- Three objects stay apart: marginal uncertainty within a window; joint relationships within that same posterior;
  changes between windows or cells. The three point summaries are summaries of one conditional distribution.
- A complete alternative imaging vector supported by these data names a recording and a window and is a joint draw
  (for example that window's SGM) or an explicitly defined selection procedure; it is never assembled from marginal
  percentile ranges coordinate by coordinate. The SGM is a co-occurring draw but does not carry the joint uncertainty;
  the median of SGM coordinates across windows is not itself a sampled configuration.
- Outside-prior draws are a property of unrestricted flow draws, counted and labeled; a window whose every draw lies
  outside on one coordinate (the InlB PSF width) is the documented limitation of that condition's prior, not a defect
  of the collection.
- A seed documents the attempt; bitwise reproducibility across devices or library versions is not promised.

## Usage

```
MACHINE_PROFILE=<p> python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Draws.py \
    --total-time-seconds 2.0 --condition FAB --artifact-tag C128V42A --cells 26 --seed 20261103 \
    [--posterior-samples 10000] [--nuisance-tag SEL] [--scatter-subset 2000] [--scatter-seed 0] \
    [--collection-label CELL4] [--dry-run]
```

Local GPU (the PC profile) or rcl01; no HPC submission. Writes a new folder
`<data_bank>/Posit/<alias>_<COND>_<timing>[_<TAG>]_Window_Posterior_Draws[_<LABEL>]/` (the `.npz`, `window_summaries.csv`,
`correlations.csv`, `joint_depth.csv`, `figures/`, `report.md`, `provenance.json`); an existing folder is refused.
`--collection-label` (alphanumeric, optional) is output-only: it appends `_<LABEL>` to the folder and file names so that
several collections of ONE estimator and ONE source Experiment can coexist (one per cell, say) without touching the
unlabeled collection; it changes no input, and `--artifact-tag` keeps selecting the estimator and the source product.
The label is recorded in the collection manifest (`collection_label`), the report and `provenance.json`. One
cell of ten windows at 10,000 draws takes under a minute on a 4 GB Quadro T2000 (about 2.3 s per window after the
first, peak 0.9 GiB) and stores 2 MiB compressed. The data-bank file is outside the workspace backup: copy it into the
analysis evidence folder (or another recorded location) so the draws have a backup of their own.
