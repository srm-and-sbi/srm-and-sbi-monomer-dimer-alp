# Window-posterior review of an Experiment product — usage and interpretation

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Review.py`. The detector Experiment
stage stores, for every model-length window of every experimental recording, three point summaries of one posterior
(the numerical MAP, the sample geometric median of the draws, the vector of marginal medians) and the marginal
quantiles of the draws at the levels 0.05, 0.25, 0.50, 0.75 and 0.95. A condition-wide selection such as the median
over the per-window medians conceals the differences between windows and between recordings, and composes
coordinates that were never jointly drawn. This analysis reads one saved product and lays its windows open.

**This is a read-only diagnostic.** It loads no estimator, draws nothing, and changes no product or artifact. The
joint draws of selected windows are collected by the companion
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Draws.py`, documented beside it.

## What it computes

- **Validation of the product** through the artifact schema: parameter order, quantile levels, a single condition,
  unique `(kind_index, cell, chunk)` identifiers, exactly ten chunks numbered 0 to 9 per cell, finite estimates,
  monotone quantiles, the estimator artifact present, the window geometry.
- **Per-window physical-unit tables** for every cell: the complete MAP, SGM and median vectors, the 50 % and 90 %
  marginal intervals, the interval widths computed inside each window (`10^q95 - 10^q05`), the stored log10 values,
  and flags (the reference vector inside the 90 % interval, the MAP inside it, outside-prior coordinates).
- **The condition-wide interval summary**: per parameter, the medians over windows of the interval endpoints, of the
  posterior medians, of the widths (a median of widths, which need not equal the difference of separately summarized
  endpoints), of the SGM and of the MAP; the window-median percentiles; the counts of windows whose 90 % interval
  contains the reference. With `--verified-csv` it reproduces an externally prepared summary and records the maximum
  relative deviation as a check.
- **A cell inventory** (one row per cell and parameter): cell medians of the three statistics, the range of the window
  medians, the median widths, the first-to-last change of the median in dex, the reference-inside count.
- **Figures for the pilot cells** (`--pilot-cells`): ten-value empirical distributions of each statistic per parameter;
  time series over the ten windows with each window's own 50 % and 90 % bands; pairwise plots of the complete vectors
  colored by window time. **Figures for the condition**: the between-cell overview (one marker and range per cell) and
  the pooled histogram, labeled as a descriptive mixture of cells and times.

## Conventions kept

Stored coordinates are the estimator's (log10 of the physical value for every detector row); a physical value is ten
to the power of the stored value, applied to each stored quantile (`parameterization.to_physical`); the finite-sample
quantile convention of the product (numpy linear interpolation over the window's draws) is never recomputed. The SGM
is the product's own: the exact sample medoid in estimator coordinates, each coordinate divided by its prior width.
Units: `mu_r` in pixels and equal to sqrt(2) times the Gaussian PSF standard deviation (the model's parameter, not a
spot-fit output); `sigma_r` and `sigma_pc` dimensionless natural-log spreads; `mu_pc` photons per active dye per
frame; `prob_photo_bleach` a probability over the fixed 100-frame reference interval; `lambda_rate` per second.

## Reading rules

- Each window's interval is marginal and per window. The time series shows changes between windows. Neither shows the
  joint structure inside a window; the draws companion does.
- The three point summaries describe the same conditional distribution; they are never pooled as thirty observations,
  and ten values per statistic are drawn as ten points, not fitted with a smooth density.
- The pooled histogram and the condition-wide summary are descriptive summaries over cells and times, not a posterior
  for a shared parameter.
- A proposal for a complete alternative imaging vector is never read off these marginal tables coordinate by
  coordinate; it identifies a window and a joint draw (see the draws companion).

## Usage

```
MACHINE_PROFILE=<p> python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Review.py \
    --total-time-seconds 2.0 --condition FAB --artifact-tag C128V42A --pilot-cells 26 [--nuisance-tag SEL] \
    [--verified-csv <summary.csv>] [--experiment-span-seconds 20] [--dry-run]
```

Reads `<data_bank>/Posit/<alias>_<COND>_<timing>[_<TAG>]_MAP_Experiment/<same>.npz` and the reference
`selection_user` Nuisance_DLI artifact of `--nuisance-tag`. Writes a new folder
`<data_bank>/Posit/<alias>_<COND>_<timing>[_<TAG>]_Window_Posterior_Review/` (`window_estimates.csv`,
`window_vectors.csv`, `cell_inventory.csv`, `condition_interval_summary.csv`, `figures/`, `report.md`,
`provenance.json` with source checksums); an existing folder is refused. CPU only, a few seconds.
