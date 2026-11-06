# Experiment diagnostics - SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A
Run: 2026-11-06 15:35:49 UTC
Note: Review copy (COHORT60RANKSR2) regenerated on the CPU from the saved product SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment.npz; no estimation was repeated. Adopted report: /media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment/report.md.

## Checks - 8 passed, 0 failed

| check | result | detail |
|---|---|---|
| estimates_nonempty | PASS | 600 windows estimated over experimental recordings |
| three_point_estimates_present | PASS | map_estimate, posterior_quantiles (median at level 0.50) and posterior_sgm stored for every window |
| aggregation_coverage_complete | PASS | FAB: 60 cells x 10 positions |
| aggregation_equal_window_weights | PASS | 10000 retained draws in every window, so concatenation realizes equal window and cell weights |
| aggregation_contract_matches_product | PASS | checkpoint 01a2d0c17b99…, 6 keys, scale [0.3, 0.75, 0.75, 0.75, 1.5, 1.0], unrestricted, 10000 draws, geometry {'n_frames': 100, 'span_frames': 1000, 'step_frames': 100}; input conversion verified: every source (60) records the same input conversion; the reference product does not record one |
| aggregation_identities_in_product | PASS | 600 supplied windows, all windows of this product |
| aggregation_sgm_members | PASS | 793 groups; every SGM is the retained draw its source identity names |
| aggregation_archive_reads_back | PASS | format 2, 62 rank blocks, 793 groups re-validated |

## Quantitative

| metric | value | expected | meaning |
|---|---|---|---|
| regenerated_from | /media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment.npz | - | the saved Experiment product re-rendered; its arrays are not modified. |
| conditions | 1 | - | - |
| total_estimates | 600 | - | number of (cell, chunk) windows estimated across all conditions. |
| aggregation | pooled | - | report distribution view: pooled over (cell x chunk). |
| artifact_schema | v2 / estimate definitions v2 | - | the stored computation contract; the manifest inside the .npz records the optimizer settings, draw count, quantile levels, SGM scaling, code and checkpoint identity this product was made under. |
| summary_draws | 10000 | - | flow-draws per window that the median and the SGM summarize (pool mode unrestricted). |
| n[MET-FAB] | 600 | - | estimates for condition MET-FAB (stored as 'FAB'). |
| estimator_condition | MET-FAB | - | the condition the estimator was trained for; its parameter table (6 learnable rows) is the table of every value in this report. |
| mean_log_prob | 13.44 | - | mean log-density at the returned MAP candidate (optimization diagnostic; not a calibration/quality metric). |

## Point estimates: definitions

| key | stored field | definition |
|---|---|---|
| MAP | map_estimate | numerical MAP candidate: the highest-scoring point retained by the configured gradient-ascent optimizer of the flow's log-density, initialized from the top-scoring candidate draws and stepping each coordinate in units of its interquartile range over all the candidate draws, in estimator coordinates (log10 for a log row, the value itself for a linear row). The optimization is unconstrained: prior support and convergence to a mode are not guaranteed, so a value outside the prior box is a flow optimum, not a MAP of the prior-supported posterior. |
| median | posterior_quantiles (the quantile at level 0.50, index 2) | marginal median of draws: the 0.50 quantile of each coordinate (linear interpolation between order statistics), taken independently per coordinate over the observation's flow-draws, in estimator coordinates (log10 for a log row, the value itself for a linear row). A coordinate-wise composite: the resulting vector is not necessarily a sampled vector and need not be a probable point of the joint. A physical value is the transform of this quantile. |
| SGM | posterior_sgm | SGM of draws: the sample geometric median: the complete draw minimizing the summed Euclidean distance to all other draws (an exact sample medoid) over the same flow-draws, each coordinate divided by its prior width before the distance is taken, in estimator coordinates (log10 for a log row, the value itself for a linear row). A realized draw, so its coordinates co-occurred; selecting a member avoids a coordinate-wise composite but does not by itself preserve the distribution's correlations. Distinct from an SGM of MAP vectors, from a pooled SGM of per-window SGMs, and from an SGM in physical coordinates -- each of those names its own population and scaling where it is reported. |

*one row per point estimate (evaluation.POINT_ESTIMATES); every table and figure below uses these three tokens. The three are read together; no one replaces the others.*

## Data scope

| data | condition | cells | windows | draws | coverage |
|---|---|---|---|---|---|
| original Experiment summaries (the tables and diagnostic figures) | MET-FAB | 60 | 600 | 10000 flow-draws per window summarized when the product was written; the draws themselves were not retained | every cell of the product |
| retained-cloud aggregation | MET-FAB | 60 of 60 | 600 of 600 | 10,000 joint draws retained per window | complete cohort |

*the two scopes are stated separately: the original diagnostics summarize the product's own windows; the aggregation groups retained joint draws (the product's own, or compatible collections of the same estimator drawn after the run). A pilot subset never relabels the original figures.*

## MAP theta by condition (estimator coordinates)

| parameter | label | units | kind | n | median (estimator) | IQR (estimator) | median value | outside prior |
|---|---|---|---|---|---|---|---|---|
| mu_r | $\mu_{r}$ | log10 | MET-FAB | 600 | +0.240 | 0.039 | 1.737 | 7% |
| sigma_r | $\sigma_{r}$ | log10 | MET-FAB | 600 | -0.750 | 0.132 | 0.1778 | 0% |
| mu_pc | $\mu_{pc}$ | log10 | MET-FAB | 600 | +2.023 | 0.028 | 105.3 | 10% |
| sigma_pc | $\sigma_{pc}$ | log10 | MET-FAB | 600 | -0.238 | 0.055 | 0.5779 | 0% |
| prob_photo_bleach | $\rho_{pb}$ | log10 | MET-FAB | 600 | -0.831 | 0.104 | 0.1476 | 0% |
| lambda_rate | $\lambda$ | log10 | MET-FAB | 600 | +0.914 | 0.056 | 8.204 | 0% |

*no ground truth for experimental recordings; values are the distribution of the MAP estimate per condition (pooled over (cell x chunk)), in the row's estimator coordinate ('units': log10 for a log row, the fraction itself for a linear row). 'outside prior' is the share beyond the row's prior box. The gradient ascent is unconstrained, so an outside-prior value happens under either pool mode -- --pool-mode bounds the candidate pool, not the optimizer's steps -- and is a flow optimum rather than a MAP of the prior-supported posterior. The three tables are read together.*

## median theta by condition (estimator coordinates)

| parameter | label | units | kind | n | median (estimator) | IQR (estimator) | median value | outside prior |
|---|---|---|---|---|---|---|---|---|
| mu_r | $\mu_{r}$ | log10 | MET-FAB | 600 | +0.239 | 0.040 | 1.734 | 7% |
| sigma_r | $\sigma_{r}$ | log10 | MET-FAB | 600 | -0.752 | 0.142 | 0.177 | 0% |
| mu_pc | $\mu_{pc}$ | log10 | MET-FAB | 600 | +2.021 | 0.024 | 105 | 8% |
| sigma_pc | $\sigma_{pc}$ | log10 | MET-FAB | 600 | -0.233 | 0.055 | 0.5848 | 0% |
| prob_photo_bleach | $\rho_{pb}$ | log10 | MET-FAB | 600 | -1.030 | 0.166 | 0.09327 | 0% |
| lambda_rate | $\lambda$ | log10 | MET-FAB | 600 | +0.908 | 0.073 | 8.096 | 0% |

*no ground truth for experimental recordings; values are the distribution of the median estimate per condition (pooled over (cell x chunk)), in the row's estimator coordinate ('units': log10 for a log row, the fraction itself for a linear row). 'outside prior' is the share beyond the row's prior box. The three tables are read together.*

## SGM theta by condition (estimator coordinates)

| parameter | label | units | kind | n | median (estimator) | IQR (estimator) | median value | outside prior |
|---|---|---|---|---|---|---|---|---|
| mu_r | $\mu_{r}$ | log10 | MET-FAB | 600 | +0.240 | 0.040 | 1.738 | 7% |
| sigma_r | $\sigma_{r}$ | log10 | MET-FAB | 600 | -0.756 | 0.143 | 0.1755 | 0% |
| mu_pc | $\mu_{pc}$ | log10 | MET-FAB | 600 | +2.020 | 0.028 | 104.8 | 14% |
| sigma_pc | $\sigma_{pc}$ | log10 | MET-FAB | 600 | -0.232 | 0.054 | 0.5861 | 0% |
| prob_photo_bleach | $\rho_{pb}$ | log10 | MET-FAB | 600 | -1.045 | 0.158 | 0.09014 | 0% |
| lambda_rate | $\lambda$ | log10 | MET-FAB | 600 | +0.897 | 0.076 | 7.884 | 0% |

*no ground truth for experimental recordings; values are the distribution of the SGM estimate per condition (pooled over (cell x chunk)), in the row's estimator coordinate ('units': log10 for a log row, the fraction itself for a linear row). 'outside prior' is the share beyond the row's prior box. The three tables are read together.*

## Point estimates compared (per parameter, estimator coordinates)

| parameter | label | units | kind | n | median MAP | median median | median SGM | IQR MAP | IQR median | IQR SGM | outside prior MAP | outside prior median | outside prior SGM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mu_r | $\mu_{r}$ | log10 | MET-FAB | 600 | +0.240 | +0.239 | +0.240 | 0.039 | 0.040 | 0.040 | 7% | 7% | 7% |
| sigma_r | $\sigma_{r}$ | log10 | MET-FAB | 600 | -0.750 | -0.752 | -0.756 | 0.132 | 0.142 | 0.143 | 0% | 0% | 0% |
| mu_pc | $\mu_{pc}$ | log10 | MET-FAB | 600 | +2.023 | +2.021 | +2.020 | 0.028 | 0.024 | 0.028 | 10% | 8% | 14% |
| sigma_pc | $\sigma_{pc}$ | log10 | MET-FAB | 600 | -0.238 | -0.233 | -0.232 | 0.055 | 0.055 | 0.054 | 0% | 0% | 0% |
| prob_photo_bleach | $\rho_{pb}$ | log10 | MET-FAB | 600 | -0.831 | -1.030 | -1.045 | 0.104 | 0.166 | 0.158 | 0% | 0% | 0% |
| lambda_rate | $\lambda$ | log10 | MET-FAB | 600 | +0.914 | +0.908 | +0.897 | 0.056 | 0.073 | 0.076 | 0% | 0% | 0% |

*the three point estimates on the same windows (pooled over (cell x chunk)), columns grouped by statistic with the estimates consecutive: the median over windows, the IQR over windows (both in the row's estimator coordinate) and the share outside the prior box, each for MAP, median and SGM (definitions above). A statement about a parameter is read from the row as a whole, never from one column.*

## Point-estimate agreement (per parameter, estimator coordinates)

| parameter | label | units | kind | n | MAP vs median | MAP vs SGM | SGM vs median | MAP outside 90% |
|---|---|---|---|---|---|---|---|---|
| mu_r | $\mu_{r}$ | dex | MET-FAB | 600 | 0.001 | 0.002 | 0.002 | 0% |
| sigma_r | $\sigma_{r}$ | dex | MET-FAB | 600 | 0.006 | 0.007 | 0.006 | 0% |
| mu_pc | $\mu_{pc}$ | dex | MET-FAB | 600 | 0.002 | 0.006 | 0.005 | 0% |
| sigma_pc | $\sigma_{pc}$ | dex | MET-FAB | 600 | 0.002 | 0.007 | 0.005 | 0% |
| prob_photo_bleach | $\rho_{pb}$ | dex | MET-FAB | 600 | 0.198 | 0.212 | 0.019 | 0% |
| lambda_rate | $\lambda$ | dex | MET-FAB | 600 | 0.008 | 0.018 | 0.010 | 0% |

*each 'X vs Y' column is the median over windows of the absolute difference between the two point estimates, in the row's unit (dex for a log row, percentage points for a linear row). 'MAP outside 90%' is the share of windows whose MAP falls outside the central 90% interval of the draws. Large gaps with a high outside share establish that the optimized candidate and the draw summaries disagree; the cause is not decided by this table and needs separate checks.*

## Within-recording drift across windows (per point estimate, stored coordinates)

| condition | parameter | estimate | cells | change first->last (median) | IQR across cells | sign consistency | cells over 0.3 dex | signed-rank p |
|---|---|---|---|---|---|---|---|---|
| MET-FAB | mu_r | MAP | 60 | -0.031 dex | 0.035 | 90% | 0% | 5.5e-09 |
| MET-FAB | mu_r | posterior median | 60 | -0.031 dex | 0.032 | 90% | 0% | 9.8e-09 |
| MET-FAB | mu_r | SGM | 60 | -0.030 dex | 0.035 | 88% | 0% | 1.6e-08 |
| MET-FAB | sigma_r | MAP | 60 | -0.146 dex | 0.088 | 92% | 3% | 2.8e-10 |
| MET-FAB | sigma_r | posterior median | 60 | -0.155 dex | 0.094 | 92% | 8% | 2.7e-10 |
| MET-FAB | sigma_r | SGM | 60 | -0.152 dex | 0.103 | 92% | 7% | 2.4e-10 |
| MET-FAB | mu_pc | MAP | 60 | -0.037 dex | 0.033 | 100% | 0% | 1.6e-11 |
| MET-FAB | mu_pc | posterior median | 60 | -0.030 dex | 0.028 | 98% | 0% | 1.7e-11 |
| MET-FAB | mu_pc | SGM | 60 | -0.031 dex | 0.031 | 100% | 0% | 1.6e-11 |
| MET-FAB | sigma_pc | MAP | 60 | -0.036 dex | 0.027 | 95% | 0% | 1.4e-09 |
| MET-FAB | sigma_pc | posterior median | 60 | -0.043 dex | 0.031 | 95% | 0% | 3.7e-10 |
| MET-FAB | sigma_pc | SGM | 60 | -0.044 dex | 0.033 | 90% | 0% | 2.7e-10 |
| MET-FAB | prob_photo_bleach | MAP | 60 | +0.087 dex | 0.075 | 97% | 0% | 2.0e-11 |
| MET-FAB | prob_photo_bleach | posterior median | 60 | +0.133 dex | 0.103 | 98% | 3% | 1.7e-11 |
| MET-FAB | prob_photo_bleach | SGM | 60 | +0.130 dex | 0.099 | 97% | 2% | 1.9e-11 |
| MET-FAB | lambda_rate | MAP | 60 | +0.045 dex | 0.057 | 95% | 0% | 1.3e-10 |
| MET-FAB | lambda_rate | posterior median | 60 | +0.062 dex | 0.075 | 95% | 0% | 3.5e-11 |
| MET-FAB | lambda_rate | SGM | 60 | +0.065 dex | 0.079 | 95% | 0% | 5.1e-11 |

*per recording, a line is fitted to each estimate against the window index and the fitted first-to-last change is taken; the row aggregates those changes across recordings. One row per point estimate (MAP, posterior median, SGM), read together: the three come from the same draws, so a difference between them describes the shape of the density along the recording. The change is in dex for a log row and in percentage points ('pp', an absolute difference of a fraction) for a linear row; 'cells over 0.3 dex' is the share whose change exceeds a factor of two and exists for log rows only ('n/a' for a linear row, which has no multiplicative tolerance). The signed-rank p states detectability, not magnitude, and no row attributes a cause: the block not inferred is marginalized here, so an imaging trend and a biological one are not separable in this stage alone.*

## Experimental aggregation and nuisance candidates

Collections of representative parameter vectors obtained by grouping the retained window posteriors (the summary draws of the per-window estimates above) in explicitly defined ways: the four constructions of the canonical table below, each for every requested window set. The INPUTS are mixtures of window posteriors (the pooled retained draws of a group, equal weight per window); the OUTPUTS exported below are collections of point representatives, one complete vector per group, which do not themselves retain the complete input mixtures (the mixture quantiles stored beside each representative are summaries of the pooled draws, not the draws). These are descriptive groupings, not a posterior conditioned on a whole recording and not products of posteriors; the aggregation performs no neural inference on an input longer than one window and optimizes no MAP. The collections are candidates for the Nuisance DLI selection (the analysis `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py` reads the archive below); no construction, window set, statistic or mixture is selected here.

**Coverage.** MET-FAB: complete cohort, 60 of 60 cells of this product (identities in aggregation_groups.csv), 600 of 600 windows, positions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10000 retained draws per window. Complete cohort: MET-FAB 60 of 60 cells, 600 of 600 windows. The original MAP and summary references above come from the adopted product; the aggregation summaries come from the retained collections named below, fresh draws of the same estimator that do not reconstruct the original sampling attempt.

**Window sets.** all = chunks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; opening = chunks [0]. `all` is every position of the declared window geometry (100-frame windows, step 100 frames, recording span 1000 frames: positions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), and every included cell must carry every position a set requests; `opening` is the first window. A deliberate subset is named explicitly; nothing is dropped silently.

**Two independent choices.** (1) Windows included: `all` = all ten 2 s windows, `opening` = opening 0–2 s window only (this is all a window set selects). (2) Samples pooled before one representative is selected: within a cell (one vector per cell), across the included cells at one window position (one vector per position), across the included dataset (one vector), or separately for every cell–window pair (one vector per window). `all` describes the window selection only: under `all` the per-cell construction still returns one vector per cell. With `opening`, per-cell equals per-cell-window and per-position equals global, so overlapping curves and points in the opening figures are expected and are listed in the equivalent-groups table, not missing output.

**Retained clouds.** 60 retained draw collections, each named with its path, collection label and sha256 in aggregation_manifest.json (`sources`): first `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws.npz`, last `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws_COHORTCELL9.npz`; 59 carry a collection label (CELL4, COHORTCELL0, COHORTCELL1, …). Retained draw collections (fresh clouds of the adopted estimator, collected after the Experiment run). Every supplied identity (condition, cell, chunk) is a window of this product. Input conversion (preprocessing): verified: every source (60) records the same input conversion; the reference product does not record one; recorded as `read_cell_chunks(domain='stored'): 16-bit raw frames to 8-bit, normalize_video in the sampler`. Disclosed differences between the sources (not part of the computation contract; listed per source in aggregation_manifest.json under `disclosed_source_differences`): 60 distinct collection label values, 59 distinct seed policy values, 2 distinct implementation sha256 values, written between 2026-11-03 15:19:28 UTC and 2026-11-05 16:43:43 UTC.

**Methods.** Distances in estimator coordinates divided by the prior widths [0.3, 0.75, 0.75, 0.75, 1.5, 1.0] (the per-window SGM convention). A single-window group reuses the window's stored exact sample medoid; a pooled group above 10000 rows takes the collection kernel's Weiszfeld geometric median snapped to the nearest retained draw (`weiszfeld_snap`, an approximation to the exact medoid, recorded per group). Pooled marginal quantiles are mixture quantiles of the pooled draws in estimator coordinates (the stage's interpolation), endpoints transformed to physical units; representative-distribution quantiles (median, IQR, central 90 % range over a collection's representatives) use the inverse weighted empirical CDF and are empirical dispersion, not credible or confidence intervals. Unrestricted draws stay unrestricted: outside-prior mass is counted, never clipped. SGM is the default candidate statistic; the marginal-median vector of each group is stored beside it, labeled as a coordinate-wise composite. Aggregation took 34.3 s on the CPU.

**Reference.** SEL: mu_r 1.734, sigma_r 0.177, mu_pc 105, sigma_pc 0.5848, prob_photo_bleach 0.05, lambda_rate 8 (physical), loaded from /media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_SEL_Nuisance_DLI.npz. Its current method, per the artifact's own decision record (the per-parameter table below): the median over the per-window posterior medians of the adopted Experiment product, transformed to physical units, followed by the documented parameter-specific selections. A historical fixed scenario for comparison, neither reconstructed from the draws nor mixed into any collection.

**Supporting files (this directory).** [aggregation_candidates.npz](aggregation_candidates.npz) (every group's SGM and median vectors, mixture quantiles, identities, methods, contract, sources and the target binding), [aggregation_groups.csv](aggregation_groups.csv), [aggregation_comparison.csv](aggregation_comparison.csv), [aggregation_compatibility.csv](aggregation_compatibility.csv) (every pairwise distance rank), [aggregation_manifest.json](aggregation_manifest.json). Figures: [aggregation_FAB_all_marginals.png](figures/aggregation_FAB_all_marginals.png), [aggregation_FAB_all_joint.png](figures/aggregation_FAB_all_joint.png), [aggregation_FAB_opening_marginals.png](figures/aggregation_FAB_opening_marginals.png), [aggregation_FAB_opening_joint.png](figures/aggregation_FAB_opening_joint.png), [aggregation_FAB_all_temporal.png](figures/aggregation_FAB_all_temporal.png), [aggregation_FAB_window_selection.png](figures/aggregation_FAB_window_selection.png); posterior compatibility (the subsection at the end of this section): [aggregation_FAB_all_compatibility_cells.png](figures/aggregation_FAB_all_compatibility_cells.png), [aggregation_FAB_opening_compatibility_cells.png](figures/aggregation_FAB_opening_compatibility_cells.png), [aggregation_FAB_all_compatibility_windows.png](figures/aggregation_FAB_all_compatibility_windows.png), [within-cell matrix atlas FAB/all](compatibility/FAB/all/windows/index.md) (captions at the end of the report).

Reading rules: an acknowledged training-bound extrapolation stays visible in the outside-prior columns (the InlB PSF width lies above its prior ceiling); pairwise correlations in the joint figures are descriptive, not selection constraints; nothing here declares a winner, and membership of a reference inside intervals computed from the same draws is a compatibility diagnostic, not a validation.

## Experimental aggregation and nuisance candidates: constructions

| ID | visible description | samples pooled before one SGM is selected | calculation | representatives per window set | interpretation |
|---|---|---|---|---|---|
| `reference` (SEL) | Existing fixed nuisance vector: SEL | loaded from the reference artifact and its selection record (nothing pooled) | median over the per-window posterior medians of the adopted Experiment product (physical units), followed by the documented parameter-specific selections (per-parameter record below); not an SGM of draws | 1 | one fixed scenario; comparison only |
| `per_cell` | One SGM vector per cell | all selected window clouds from one cell | pool all retained draws of the cell's selected windows, equal weight per window, and select the SGM | all: 60 (MET-FAB); opening: 60 (MET-FAB) | variation between cell representatives over the declared acquisition period |
| `per_window_position` | One SGM vector per window position | the clouds at the same position across the included cells | at each selected position, pool the retained draws across the included cells, equal weight per cell, and select the SGM | all: 10 (MET-FAB); opening: 1 (MET-FAB) | variation between acquisition-period representatives, cell differences compressed at each position |
| `global` | One SGM vector for the included dataset | all selected cell-window clouds | pool the retained draws with equal cell weight and equal selected-window weight within each cell, and select the SGM | all: 1 (MET-FAB); opening: 1 (MET-FAB) | an overall joint center of the selected collection |
| `per_cell_window` | One SGM vector per cell–window pair | one window's own cloud | the SGM of each included cell-window posterior | all: 600 (MET-FAB); opening: 60 (MET-FAB) | variation between window representatives, cell and window identities retained |

*the canonical options; counts are per condition. Legends shorten the descriptions to 'SGM per cell', 'SGM per window position', 'SGM for included dataset', 'SGM per cell–window' and 'Fixed reference: SEL', with the actual number of vectors. An SGM of window SGMs is not an active construction; no aggregate MAP exists.*

## Experimental aggregation and nuisance candidates: the reference SEL per parameter (its decision record)

| parameter | value (physical) | in prior box | source (current method) | limitation |
|---|---|---|---|---|
| mu_r | 1.734 | yes | adopted Fab detector estimator C128V42A, record ..._FAB_2S_50FPS_C128V42A_MAP_Experiment, median over the 600 per-window posterior medians (ten 2 s windows of the sixty recordings), transformed to physical units; MAP and SGM aggregates read beside it (MAP 1.737, SGM 1.738). Neural estimate supported by synthetic recovery (MAE 0.011 dex, r 0.98 on the 25,000 EVAL videos of the current model, DETECTOR_WORKFLOW.md sec. 9.10); replaces the direct PSF-width estimate 1.585 selected before, which becomes the sensitivity scenario SEL_MU_R | a demonstrated synthetic accuracy, no demonstrated experimental one; 7 % of windows above the 2.0 px prior edge; the direct estimator disagrees by 0.04 dex with no ground truth to decide; the fitted width falls 0.03 dex along a recording, which no fixed value represents; point estimate retained as produced, precision not established |
| sigma_r | 0.177 | yes | adopted Fab detector estimator C128V42A, same record and aggregation (MAP 0.178, SGM 0.176); supported by the direct PSF-width cross-check 0.21 and the fit-corrected localization reference about 0.15 (sec. 6.7). Neural estimate supported by synthetic recovery (MAE 0.034 dex, r 0.98) | falls 0.15 dex along a recording in 92 % of recordings, a within-recording decline no fixed value represents; experimental accuracy model-dependent |
| mu_pc | 105 | yes | adopted Fab detector estimator C128V42A, same record and aggregation (MAP 105.3, SGM 104.8). Neural estimate supported by synthetic recovery (MAE 0.028 dex, r 0.98) | no independent experimental cross-check (the localization per-detection brightness 386 is a spot-level sum, sec. 6.7 caveat 3); 8 to 14 % of windows below the 100-photon prior edge; falls 0.03 dex along a recording; a per-dye window median under the labeling and camera assumptions |
| sigma_pc | 0.5848 | yes | adopted Fab detector estimator C128V42A, same record and aggregation (MAP 0.5779, SGM 0.5861); fit-corrected localization reference about 0.5. Neural estimate supported by synthetic recovery (MAE 0.079 dex, r 0.90, the weakest of the four spatial and brightness parameters) | experimental accuracy model-dependent |
| prob_photo_bleach | 0.05 | yes | empirically anchored: a declared per-dye bleaching probability, anchored to the measured whole-recording fluorescence decline of the sixty MET-FAB recordings over 20 s (0.054 to 0.057 per 100-frame interval on the 16-bit frames; record ..._20S_50FPS_Direct_Fluorescence_Loss_Experiment_RAW), rounded as a working value, selected on 2026-10-01 from the overlay of single-rate declines on the measured curves (figures/field_flux_curves_FAB.png). The adopted estimator's own value (median 0.093, MAP 0.148) plays no part: it does not recover the parameter (r 0.06, MAE comparable to a constant guess). Probability over the fixed 100-frame reference interval | the analysis measures the field-fluorescence decline, not per-dye bleaching; the mapping to the simulator's per-dye probability rests on the assumptions recorded in DETECTOR_WORKFLOW.md sec. 7.6 (ideal independent dyes of stationary mean brightness); the recorded decline is not single-exponential and is stronger in brighter recordings; the background itself falls 9 %; emitters leaving the field and label exchange are not separable from bleaching; the late-phase rate 0.03 and the faster-loss value 0.10 are the declared sensitivity scenarios SEL_PB03 and SEL_PB10 |
| lambda_rate | 8 | yes | declared, rounded from the adopted Fab detector estimator C128V42A (median 8.096 per second, MAP 8.204, SGM 7.884; window spread 0.07 dex), the neural value being the default source under the hierarchy; partially recovered (MAE 0.217 dex, r 0.49), so its digits are not retained; replaces the nominal 5 chosen before. The localization-autocorrelation anchor about 5 (sec. 6.5) sits 0.21 dex below, comparable in magnitude to the synthetic mean absolute error | not a uniquely measured rate; nothing on the recordings decides between 8 and the anchor 5; rises 0.06 dex along a recording; correlation time 0.12 s |

*copied from the reference artifact's selection record; the reference is a fixed scenario and its limitations travel with it.*

## Experimental aggregation and nuisance candidates: equivalent groups

| group | equals | condition | cell | position |
|---|---|---|---|---|
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 0 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 1 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 2 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 3 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 4 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 5 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 6 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 7 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 8 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 9 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 10 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 11 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 12 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 13 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 14 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 15 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 16 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 17 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 18 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 19 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 20 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 21 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 22 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 23 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 24 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 25 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 26 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 27 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 28 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 29 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 30 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 31 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 32 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 33 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 34 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 35 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 36 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 37 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 38 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 39 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 40 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 41 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 42 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 43 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 44 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 45 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 46 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 47 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 48 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 49 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 50 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 51 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 52 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 53 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 54 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 55 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 56 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 57 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 58 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 59 |  |

*a single-window set makes per_cell equal per_cell_window and per_window_position equal global; a single cell makes per_window_position equal per_cell_window and global equal per_cell; the computed group is reused. 1 further rows in aggregation_groups.csv.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set all]: SGM representatives (physical units)

| construction | parameter | n | median | IQR (q25 – q75) | central 90 % range | median vs reference | outside-prior mass |
|---|---|---|---|---|---|---|---|
| SGM per cell (`per_cell`) | mu_r | 60 | 1.732 | 1.677 – 1.821 | 1.642 – 1.949 | -0.1 % (-0.000 dex) | 1.7 % |
| SGM per cell (`per_cell`) | sigma_r | 60 | 0.1746 | 0.1616 – 0.2046 | 0.1473 – 0.2414 | -1.4 % (-0.006 dex) | 0.0 % |
| SGM per cell (`per_cell`) | mu_pc | 60 | 104.7 | 102.6 – 107.6 | 98.19 – 114.2 | -0.3 % (-0.001 dex) | 8.3 % |
| SGM per cell (`per_cell`) | sigma_pc | 60 | 0.5805 | 0.5509 – 0.6111 | 0.5253 – 0.6633 | -0.7 % (-0.003 dex) | 0.0 % |
| SGM per cell (`per_cell`) | prob_photo_bleach | 60 | 0.0911 | 0.07562 – 0.1033 | 0.06343 – 0.1193 | +82.2 % (+0.261 dex) | 0.0 % |
| SGM per cell (`per_cell`) | lambda_rate | 60 | 7.876 | 7.144 – 8.299 | 5.788 – 8.742 | -1.5 % (-0.007 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | mu_r | 10 | 1.74 | 1.727 – 1.751 | 1.713 – 1.888 | +0.3 % (+0.001 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | sigma_r | 10 | 0.1727 | 0.172 – 0.1878 | 0.166 – 0.2689 | -2.4 % (-0.011 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | mu_pc | 10 | 106 | 104.1 – 110.8 | 103.3 – 115.4 | +1.0 % (+0.004 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | sigma_pc | 10 | 0.5824 | 0.5726 – 0.5991 | 0.5698 – 0.6293 | -0.4 % (-0.002 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | prob_photo_bleach | 10 | 0.09131 | 0.08001 – 0.09992 | 0.07504 – 0.1053 | +82.6 % (+0.262 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | lambda_rate | 10 | 7.662 | 7.252 – 7.839 | 6.943 – 8.284 | -4.2 % (-0.019 dex) | 0.0 % |
| SGM for included dataset (`global`) | mu_r | 1 | 1.751 | 1.751 – 1.751 | 1.751 – 1.751 | +1.0 % (+0.004 dex) | 0.0 % |
| SGM for included dataset (`global`) | sigma_r | 1 | 0.1819 | 0.1819 – 0.1819 | 0.1819 – 0.1819 | +2.8 % (+0.012 dex) | 0.0 % |
| SGM for included dataset (`global`) | mu_pc | 1 | 106 | 106 – 106 | 106 – 106 | +0.9 % (+0.004 dex) | 0.0 % |
| SGM for included dataset (`global`) | sigma_pc | 1 | 0.5824 | 0.5824 – 0.5824 | 0.5824 – 0.5824 | -0.4 % (-0.002 dex) | 0.0 % |
| SGM for included dataset (`global`) | prob_photo_bleach | 1 | 0.09131 | 0.09131 – 0.09131 | 0.09131 – 0.09131 | +82.6 % (+0.262 dex) | 0.0 % |
| SGM for included dataset (`global`) | lambda_rate | 1 | 7.662 | 7.662 – 7.662 | 7.662 – 7.662 | -4.2 % (-0.019 dex) | 0.0 % |
| SGM per cell–window (`per_cell_window`) | mu_r | 600 | 1.738 | 1.674 – 1.844 | 1.614 – 2.016 | +0.3 % (+0.001 dex) | 7.7 % |
| SGM per cell–window (`per_cell_window`) | sigma_r | 600 | 0.1766 | 0.1537 – 0.2135 | 0.136 – 0.2967 | -0.2 % (-0.001 dex) | 0.0 % |
| SGM per cell–window (`per_cell_window`) | mu_pc | 600 | 104.9 | 101.8 – 108.3 | 98.08 – 118.6 | -0.1 % (-0.001 dex) | 12.0 % |
| SGM per cell–window (`per_cell_window`) | sigma_pc | 600 | 0.5844 | 0.5495 – 0.6243 | 0.5125 – 0.6832 | -0.1 % (-0.000 dex) | 0.0 % |
| SGM per cell–window (`per_cell_window`) | prob_photo_bleach | 600 | 0.08944 | 0.0745 – 0.1074 | 0.06099 – 0.1369 | +78.9 % (+0.253 dex) | 0.0 % |
| SGM per cell–window (`per_cell_window`) | lambda_rate | 600 | 7.863 | 7.026 – 8.411 | 5.471 – 8.852 | -1.7 % (-0.007 dex) | 0.0 % |

*window set all = chunks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; quantiles over the collection's representatives by the inverse weighted empirical CDF (empirical dispersion across representatives, not posterior credible intervals); for one representative every column is that vector. 'outside-prior mass' is the share of representatives beyond the row's prior box.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set all]: global representative and reference (physical units)

| parameter | global SGM (a draw) | global median vector (composite) | pooled 90 % interval | reference | global SGM vs reference | pooled draws outside prior |
|---|---|---|---|---|---|---|
| mu_r | 1.751 | 1.743 | 1.588 – 2.015 | 1.734 | +1.0 % | 6.7 % |
| sigma_r | 0.1819 | 0.1799 | 0.1192 – 0.3132 | 0.177 | +2.8 % | 0.9 % |
| mu_pc | 106 | 105 | 92.42 – 127 | 105 | +0.9 % | 27.2 % |
| sigma_pc | 0.5824 | 0.5883 | 0.4853 – 0.733 | 0.5848 | -0.4 % | 0.0 % |
| prob_photo_bleach | 0.09131 | 0.09514 | 0.01438 – 0.3211 | 0.05 | +82.6 % | 6.2 % |
| lambda_rate | 7.662 | 7.944 | 3.952 – 11.03 | 8 | -4.2 % | 14.7 % |

*the global SGM is retained draw cell 40, chunk 3, index 6234 (weiszfeld_snap; 6000000 pooled draws); the pooled 90 % interval is a mixture quantile of the pooled draws, a different quantity from the dispersion of representatives.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set opening]: SGM representatives (physical units)

| construction | parameter | n | median | IQR (q25 – q75) | central 90 % range | median vs reference | outside-prior mass |
|---|---|---|---|---|---|---|---|
| SGM per cell (`per_cell`) | mu_r | 60 | 1.885 | 1.793 – 2.018 | 1.693 – 2.045 | +8.7 % (+0.036 dex) | 33.3 % |
| SGM per cell (`per_cell`) | sigma_r | 60 | 0.2684 | 0.221 – 0.308 | 0.1659 – 0.3593 | +51.6 % (+0.181 dex) | 0.0 % |
| SGM per cell (`per_cell`) | mu_pc | 60 | 109.9 | 106.5 – 117.5 | 100.5 – 143.5 | +4.7 % (+0.020 dex) | 1.7 % |
| SGM per cell (`per_cell`) | sigma_pc | 60 | 0.6264 | 0.5812 – 0.6623 | 0.5501 – 0.7165 | +7.1 % (+0.030 dex) | 0.0 % |
| SGM per cell (`per_cell`) | prob_photo_bleach | 60 | 0.07131 | 0.06408 – 0.08194 | 0.0583 – 0.107 | +42.6 % (+0.154 dex) | 0.0 % |
| SGM per cell (`per_cell`) | lambda_rate | 60 | 6.968 | 6.268 – 7.715 | 4.815 – 8.473 | -12.9 % (-0.060 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | mu_r | 1 | 1.888 | 1.888 – 1.888 | 1.888 – 1.888 | +8.9 % (+0.037 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | sigma_r | 1 | 0.2689 | 0.2689 – 0.2689 | 0.2689 – 0.2689 | +51.9 % (+0.182 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | mu_pc | 1 | 115.4 | 115.4 – 115.4 | 115.4 – 115.4 | +9.9 % (+0.041 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | sigma_pc | 1 | 0.6293 | 0.6293 – 0.6293 | 0.6293 – 0.6293 | +7.6 % (+0.032 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | prob_photo_bleach | 1 | 0.07504 | 0.07504 – 0.07504 | 0.07504 – 0.07504 | +50.1 % (+0.176 dex) | 0.0 % |
| SGM per window position (`per_window_position`) | lambda_rate | 1 | 6.943 | 6.943 – 6.943 | 6.943 – 6.943 | -13.2 % (-0.062 dex) | 0.0 % |
| SGM for included dataset (`global`) | mu_r | 1 | 1.888 | 1.888 – 1.888 | 1.888 – 1.888 | +8.9 % (+0.037 dex) | 0.0 % |
| SGM for included dataset (`global`) | sigma_r | 1 | 0.2689 | 0.2689 – 0.2689 | 0.2689 – 0.2689 | +51.9 % (+0.182 dex) | 0.0 % |
| SGM for included dataset (`global`) | mu_pc | 1 | 115.4 | 115.4 – 115.4 | 115.4 – 115.4 | +9.9 % (+0.041 dex) | 0.0 % |
| SGM for included dataset (`global`) | sigma_pc | 1 | 0.6293 | 0.6293 – 0.6293 | 0.6293 – 0.6293 | +7.6 % (+0.032 dex) | 0.0 % |
| SGM for included dataset (`global`) | prob_photo_bleach | 1 | 0.07504 | 0.07504 – 0.07504 | 0.07504 – 0.07504 | +50.1 % (+0.176 dex) | 0.0 % |
| SGM for included dataset (`global`) | lambda_rate | 1 | 6.943 | 6.943 – 6.943 | 6.943 – 6.943 | -13.2 % (-0.062 dex) | 0.0 % |
| SGM per cell–window (`per_cell_window`) | mu_r | 60 | 1.885 | 1.793 – 2.018 | 1.693 – 2.045 | +8.7 % (+0.036 dex) | 33.3 % |
| SGM per cell–window (`per_cell_window`) | sigma_r | 60 | 0.2684 | 0.221 – 0.308 | 0.1659 – 0.3593 | +51.6 % (+0.181 dex) | 0.0 % |
| SGM per cell–window (`per_cell_window`) | mu_pc | 60 | 109.9 | 106.5 – 117.5 | 100.5 – 143.5 | +4.7 % (+0.020 dex) | 1.7 % |
| SGM per cell–window (`per_cell_window`) | sigma_pc | 60 | 0.6264 | 0.5812 – 0.6623 | 0.5501 – 0.7165 | +7.1 % (+0.030 dex) | 0.0 % |
| SGM per cell–window (`per_cell_window`) | prob_photo_bleach | 60 | 0.07131 | 0.06408 – 0.08194 | 0.0583 – 0.107 | +42.6 % (+0.154 dex) | 0.0 % |
| SGM per cell–window (`per_cell_window`) | lambda_rate | 60 | 6.968 | 6.268 – 7.715 | 4.815 – 8.473 | -12.9 % (-0.060 dex) | 0.0 % |

*window set opening = chunks [0]; quantiles over the collection's representatives by the inverse weighted empirical CDF (empirical dispersion across representatives, not posterior credible intervals); for one representative every column is that vector. 'outside-prior mass' is the share of representatives beyond the row's prior box.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set opening]: global representative and reference (physical units)

| parameter | global SGM (a draw) | global median vector (composite) | pooled 90 % interval | reference | global SGM vs reference | pooled draws outside prior |
|---|---|---|---|---|---|---|
| mu_r | 1.888 | 1.898 | 1.657 – 2.086 | 1.734 | +8.9 % | 26.6 % |
| sigma_r | 0.2689 | 0.2697 | 0.1565 – 0.3928 | 0.177 | +51.9 % | 0.2 % |
| mu_pc | 115.4 | 111.5 | 94.25 – 146 | 105 | +9.9 % | 15.5 % |
| sigma_pc | 0.6293 | 0.6287 | 0.5059 – 0.7986 | 0.5848 | +7.6 % | 0.0 % |
| prob_photo_bleach | 0.07504 | 0.07791 | 0.01309 – 0.2851 | 0.05 | +50.1 % | 4.0 % |
| lambda_rate | 6.943 | 7.277 | 3.206 – 10.53 | 8 | -13.2 % | 9.2 % |

*the global SGM is retained draw cell 27, chunk 0, index 1555 (weiszfeld_snap; 600000 pooled draws); the pooled 90 % interval is a mixture quantile of the pooled draws, a different quantity from the dispersion of representatives.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set all]: source uncertainty against representative dispersion (physical units)

| parameter | within-window 90 % width (median over windows) | per-cell pooled 90 % width (median over cells) | global pooled 90 % interval | central 90 % range of per_cell SGMs | central 90 % range of per_cell_window SGMs |
|---|---|---|---|---|---|
| mu_r | 0.1809 | 0.2726 | 1.588 – 2.015 | 1.642 – 1.949 (n=60) | 1.614 – 2.016 (n=600) |
| sigma_r | 0.1025 | 0.1478 | 0.1192 – 0.3132 | 0.1473 – 0.2414 (n=60) | 0.136 – 0.2967 (n=600) |
| mu_pc | 25.06 | 28.71 | 92.42 – 127 | 98.19 – 114.2 (n=60) | 98.08 – 118.6 (n=600) |
| sigma_pc | 0.148 | 0.1834 | 0.4853 – 0.733 | 0.5253 – 0.6633 (n=60) | 0.5125 – 0.6832 (n=600) |
| prob_photo_bleach | 0.2992 | 0.3023 | 0.01438 – 0.3211 | 0.06343 – 0.1193 (n=60) | 0.06099 – 0.1369 (n=600) |
| lambda_rate | 6.147 | 6.339 | 3.952 – 11.03 | 5.788 – 8.742 (n=60) | 5.471 – 8.852 (n=600) |

*the first three columns are intervals of input or pooled posteriors (retained posterior uncertainty; mixture quantiles for the pooled ones); the last two are the dispersion of selected representatives (compression). They are different quantities and are never read as calibrated uncertainty.*

## Experimental aggregation and nuisance candidates [MET-FAB]: SGM versus marginal-median vector of the per_cell groups (median / max |difference|, stored coordinates)

| window set | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| all | 0.002 / 0.008 | 0.005 / 0.027 | 0.005 / 0.015 | 0.004 / 0.020 | 0.016 / 0.057 | 0.008 / 0.032 |
| opening | 0.002 / 0.006 | 0.006 / 0.017 | 0.004 / 0.021 | 0.005 / 0.017 | 0.018 / 0.070 | 0.010 / 0.041 |

*the marginal-median vector is a coordinate-wise composite whose coordinates need not have occurred together; it is stored beside the SGM and must be named explicitly by a selection that uses it.*

## Experimental aggregation and nuisance candidates: posterior compatibility of SGM representatives

For every rank-scored representative, each other group of its block is a reference posterior: that group's retained cloud (for one SGM per cell, the equal-window mixture of the cell's selected windows; for one SGM per cell–window, the window's own cloud) with the group's own existing SGM. The distance rank R_ij of candidate SGM v_i against reference j is the share of reference j's draws that lie strictly farther from v_j than v_i does, with distances in estimator coordinates divided by the prior widths [0.3, 0.75, 0.75, 0.75, 1.5, 1.0] (log10 coordinates are not transformed again). Rows are candidates, columns references; the self comparison is excluded by identity; the matrix is generally asymmetric because every reference has its own spread. The candidate's score is the ordinary median of its row over the other references (two middle ranks averaged when their number is even), a raw score that is not normalized across candidates. A rank near one means the candidate sits closer to the reference's SGM than most of that reference's draws; near zero, almost every reference draw is closer. The rank describes radial centrality, not posterior density, not membership in a joint credible region and not the validity of a recording.

**Reference populations.** One SGM per cell: the other included cells of the same condition and window set; each reference is its own equal-window cloud mixture. One SGM per cell–window: the other selected windows of the same cell and condition; each reference is one window's own cloud. One SGM per window position: no rank policy in this scope (uniform selection remains available). One SGM for the included dataset: no same-level peer (uniform selection remains available). Under a single-window set such as `opening`, the per-cell score is computed across cells while the within-cell window score is unavailable (no other selected window): the identical vectors of the two constructions keep their own availability, and no score is copied between equivalent groups.

**Zeros and availability.** A median compatibility score of exactly zero means that a majority of the candidate's reference ranks are zero (at least half of the other references place the candidate beyond all of their draws); some references may still assign it a high rank. It is a valid centrality outcome of a valid recording, distinct from an unavailable score (`insufficient_peers`: a singleton reference set, no score, zero references), and it is not a data-quality classification: nothing is removed, excluded or curated on its account. Experiment reports raw median ranks only; turning scores into sampling probabilities is the Nuisance DLI constructor's explicit choice (`member_weighting`). Training-prior flags stay separate (`sgm_outside_prior_keys` in aggregation_groups.csv: whether the SGM itself lies outside the training prior, a different quantity from the outside-box fractions of the pooled draws).

**Files.** Scores, statuses and reference counts per group: [aggregation_groups.csv](aggregation_groups.csv); every pairwise rank: [aggregation_compatibility.csv](aggregation_compatibility.csv); the rank-matrix blocks and the protocol `sgm_distance_rank_v1`: [aggregation_candidates.npz](aggregation_candidates.npz) (archive format 2, manifest section `compatibility`). Computation took 0.7 s on the CPU. The figures (captions at the end of the report) show raw median ranks; the complete within-cell matrices of every cell are in the linked atlas pages, twelve cells per page.

## Experimental aggregation and nuisance candidates: posterior compatibility of SGM representatives: scope and availability

| condition | window set | grouping | matrices | candidates | references per candidate | status | zero scores (centrality outcomes, not defective recordings) |
|---|---|---|---|---|---|---|---|
| MET-FAB | all | SGM per cell (`per_cell`) | 1 | 60 | 59 | computed | 0 |
| MET-FAB | all | SGM per window position (`per_window_position`) | 0 | 10 | - | not applicable (no rank policy) | - |
| MET-FAB | all | SGM for included dataset (`global`) | 0 | 1 | - | not applicable (no rank policy) | - |
| MET-FAB | all | SGM per cell–window (`per_cell_window`) | 60 | 600 | 9 | computed | 0 |
| MET-FAB | opening | SGM per cell (`per_cell`) | 1 | 60 | 59 | computed | 0 |
| MET-FAB | opening | SGM per window position (`per_window_position`) | 0 | 1 | - | not applicable (no rank policy) | - |
| MET-FAB | opening | SGM for included dataset (`global`) | 0 | 1 | - | not applicable (no rank policy) | - |
| MET-FAB | opening | SGM per cell–window (`per_cell_window`) | 0 | 60 | - | unavailable: no other selected window within a cell (insufficient_peers) | - |

*a matrix is one block of candidates against references (per_cell: one block over the included cells; per_cell_window: one block per cell over its selected windows); 'unavailable' keeps the groups with no score and zero references, never a fabricated 0 or 1.*

## Experimental aggregation and nuisance candidates: posterior compatibility of SGM representatives [MET-FAB, window set opening]: window scores

Not computed: no other selected window within a cell (opening 0–2 s window only); the within-cell window scores are unavailable (insufficient_peers) and no heat map of artificial zeros is drawn. The per-cell scores across cells remain available above.

## What these mean

A check **PASS** means the invariant held; a **FAIL** aborts the run immediately with a located error message.

- **estimates_nonempty** -- at least one experimental window was estimated.
- **three_point_estimates_present** -- the product contract: no estimate is reported without the other two.
- **aggregation_coverage_complete** -- every included cell carries every requested window position; conditions are never pooled together.
- **aggregation_equal_window_weights** -- the equal weights of the mixtures are realized by equal draw counts.
- **aggregation_contract_matches_product** -- the retained clouds and this product share one computation contract; the input conversion is compared between the supplied collections as well, and 'unavailable' means no source records it (not verified).
- **aggregation_identities_in_product** -- every supplied (condition, cell, chunk) identity belongs to the source Experiment product.
- **aggregation_sgm_members** -- each group's SGM equals the retained draw (cell, chunk, draw index) it records; a median vector is never claimed as a draw.
- **aggregation_archive_reads_back** -- the written candidate archive is read back through the constructor's reader, which re-validates the compatibility blocks (dimensions, identities, masked diagonals, rank bounds, stored scores against the row medians).

## Figures

![window_drift_FAB](figures/window_drift_FAB.png)

*Within-recording drift, condition MET-FAB: the MAP, posterior median and SGM as the median across recordings at each window, over the per-window central 50 % and 90 % intervals of the draws (median across recordings). Dotted red lines are the prior bounds. The three lines share one draw set, so the band is drawn once. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![experiment_mu_r](figures/experiment_mu_r.png)

*mu_r ($\mu_{r}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![experiment_sigma_r](figures/experiment_sigma_r.png)

*sigma_r ($\sigma_{r}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![experiment_mu_pc](figures/experiment_mu_pc.png)

*mu_pc ($\mu_{pc}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![experiment_sigma_pc](figures/experiment_sigma_pc.png)

*sigma_pc ($\sigma_{pc}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![experiment_prob_photo_bleach](figures/experiment_prob_photo_bleach.png)

*prob_photo_bleach ($\rho_{pb}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![experiment_lambda_rate](figures/experiment_lambda_rate.png)

*lambda_rate ($\lambda$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![point_estimates_FAB_mu_r](figures/point_estimates_FAB_mu_r.png)

*mu_r ($\mu_{r}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![point_estimates_FAB_sigma_r](figures/point_estimates_FAB_sigma_r.png)

*sigma_r ($\sigma_{r}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![point_estimates_FAB_mu_pc](figures/point_estimates_FAB_mu_pc.png)

*mu_pc ($\mu_{pc}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![point_estimates_FAB_sigma_pc](figures/point_estimates_FAB_sigma_pc.png)

*sigma_pc ($\sigma_{pc}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![point_estimates_FAB_prob_photo_bleach](figures/point_estimates_FAB_prob_photo_bleach.png)

*prob_photo_bleach ($\rho_{pb}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![point_estimates_FAB_lambda_rate](figures/point_estimates_FAB_lambda_rate.png)

*lambda_rate ($\lambda$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn. Source: the original Experiment summaries (MET-FAB: 60 cells, 600 windows), not the retained clouds of the aggregation.*

![aggregation_FAB_all_marginals](figures/aggregation_FAB_all_marginals.png)

*MET-FAB · 60/60 cells, 600/600 windows · windows included: all ten 2 s windows · 10,000 joint draws/window · representative: SGM. Each panel is one parameter in physical units; each curve is the empirical cumulative distribution of one construction's SGM vectors (the fraction of its representatives at or below the x value), every construction normalized separately to total mass one, so the curves assign no weights of a selected nuisance mixture. The dataset vector and the fixed reference are vertical lines: a point representation, whose lack of width shows nothing about parameter recovery. Dotted red lines are the training-prior bounds of the estimator: training bounds, not uncertainty intervals and not clipping thresholds. Reading example: the 'SGM per cell' curve of mu_pc rises in 60 equal steps, one at the mu_pc coordinate of each complete cell-level SGM vector (the smallest 98.04 for cell 35, the largest 116.1 for cell 31). Dispersion across representatives, not posterior uncertainty.*

![aggregation_FAB_all_joint](figures/aggregation_FAB_all_joint.png)

*MET-FAB · 60/60 cells, 600/600 windows · windows included: all ten 2 s windows · 10,000 joint draws/window · representative: SGM. Each point projects one complete representative vector onto a parameter pair (physical axes); the dataset vector and the fixed reference are single points; the empty half of the matrix avoids showing every pair twice. A trend across representatives is variation BETWEEN representatives, not the correlation inside an individual window posterior and not a within-window compensation law; nothing is fitted and no relation is imposed. Dotted red lines are the training-prior bounds of the estimator: training bounds, not uncertainty intervals and not clipping thresholds.*

![aggregation_FAB_all_temporal](figures/aggregation_FAB_all_temporal.png)

*MET-FAB · 60/60 cells, 600/600 windows · windows included: all ten 2 s windows · 10,000 joint draws/window · representative: SGM. x = start of each 2 s window within the recording. The last point at 18 s represents the frames from 18 to 20 s. Thin lines: 'SGM per cell–window', one line per cell (60 cells; identities in the report inventory). Thick line: 'SGM per window position', calculated from the joint samples pooled across cells at each position: it is not an average of the thin lines and need not lie between their values in every panel. Horizontal lines: 'SGM for included dataset' and 'Fixed reference: SEL'. Gray shading: the source within-window 50 % and 90 % marginal intervals, summarized by the median endpoint across cells: a summary of the input posteriors, not a confidence band for the thick line and not a range containing 90 % of the cells. Tightly grouped SGM centers can coexist with broad source intervals (bleaching shows this); representative dispersion is never posterior precision, and a change of a fitted imaging coordinate along the recording is assigned no physical cause here. Dotted red lines are the training-prior bounds of the estimator: training bounds, not uncertainty intervals and not clipping thresholds.*

![aggregation_FAB_opening_marginals](figures/aggregation_FAB_opening_marginals.png)

*MET-FAB · 60/60 cells, 600/600 windows · windows included: opening 0–2 s window only (60 selected windows of the 600) · 10,000 joint draws/window · representative: SGM. Each panel is one parameter in physical units; each curve is the empirical cumulative distribution of one construction's SGM vectors (the fraction of its representatives at or below the x value), every construction normalized separately to total mass one, so the curves assign no weights of a selected nuisance mixture. The dataset vector and the fixed reference are vertical lines: a point representation, whose lack of width shows nothing about parameter recovery. Dotted red lines are the training-prior bounds of the estimator: training bounds, not uncertainty intervals and not clipping thresholds. Reading example: the 'SGM per cell' curve of mu_pc rises in 60 equal steps, one at the mu_pc coordinate of each complete cell-level SGM vector (the smallest 96.97 for cell 21, the largest 147.6 for cell 50). Exact equivalences under this window selection: SGM for included dataset = SGM per window position; SGM per cell = SGM per cell–window (the same vectors drawn twice, so their curves coincide; expected, not missing output). Dispersion across representatives, not posterior uncertainty.*

![aggregation_FAB_opening_joint](figures/aggregation_FAB_opening_joint.png)

*MET-FAB · 60/60 cells, 600/600 windows · windows included: opening 0–2 s window only (60 selected windows of the 600) · 10,000 joint draws/window · representative: SGM. Each point projects one complete representative vector onto a parameter pair (physical axes); the dataset vector and the fixed reference are single points; the empty half of the matrix avoids showing every pair twice. A trend across representatives is variation BETWEEN representatives, not the correlation inside an individual window posterior and not a within-window compensation law; nothing is fitted and no relation is imposed. Dotted red lines are the training-prior bounds of the estimator: training bounds, not uncertainty intervals and not clipping thresholds. Exact equivalences under this window selection: SGM for included dataset = SGM per window position; SGM per cell = SGM per cell–window (the same vectors drawn twice, so their curves coincide; expected, not missing output).*

![aggregation_FAB_window_selection](figures/aggregation_FAB_window_selection.png)

*MET-FAB · 60/60 cells, 600/600 windows · windows included: all ten 2 s windows versus opening 0–2 s window only · 10,000 joint draws/window · representative: SGM. x = cell identity. Each cell carries one marker per window selection (the per-cell construction's blue: filled for 'all ten 2 s windows', hollow for 'opening 0–2 s window only'), joined by a thin connector that represents a change of window selection, not elapsed time and not an uncertainty interval. Orange lines: 'SGM for included dataset' of each selection (one line style per selection); purple dashed: 'Fixed reference: SEL'. Blue never means the dataset vector and orange never means a per-cell vector. Examples (descriptive, not a rule): cell 50: mu_pc 147.6 (opening 0–2 s window only) against 111.9 (all ten 2 s windows); cell 3: mu_pc 103.8 (opening 0–2 s window only) against 103.6 (all ten 2 s windows). The effect of the window selection is recording-dependent. Dotted red lines are the training-prior bounds of the estimator: training bounds, not uncertainty intervals and not clipping thresholds.*

![aggregation_FAB_all_compatibility_cells](figures/aggregation_FAB_all_compatibility_cells.png)

*MET-FAB · 60/60 cells, 600/600 windows · windows included: all ten 2 s windows · 10,000 joint draws/window · representative: SGM. Rows: candidate cell; columns: reference cell (stable numeric order, every cell retained). Each reference is the other cell's retained cloud, the equal-window mixture of its selected windows (10 windows x 10,000 draws), with that cell's own SGM; color = the share of the reference's draws strictly farther from its SGM than the candidate SGM, distances in estimator coordinates divided by the prior widths; the self comparison is gray (excluded). The matrix is asymmetric because each reference has its own spread. Right strip: the row median over the other cells, the raw score on the same 0–1 scale, not a sampling probability.*

![aggregation_FAB_all_compatibility_windows](figures/aggregation_FAB_all_compatibility_windows.png)

*MET-FAB · 60/60 cells, 600/600 windows · windows included: all ten 2 s windows · 10,000 joint draws/window · representative: SGM. Rows: cells in stable order; columns: the selected window start times. Color = the within-cell median compatibility rank of the cell–window SGM: its references are the other 9 selected windows of that same cell, each with its own cloud (10,000 draws) and SGM, and nothing outside the cell. Zeros stay visible as the darkest color and are centrality outcomes, not data failures. Each cell's full window-to-window matrix is in the linked atlas (compatibility/FAB/all/windows/index.md).*

![aggregation_FAB_opening_compatibility_cells](figures/aggregation_FAB_opening_compatibility_cells.png)

*MET-FAB · 60/60 cells, 600/600 windows · windows included: opening 0–2 s window only (60 selected windows of the 600) · 10,000 joint draws/window · representative: SGM. Rows: candidate cell; columns: reference cell (stable numeric order, every cell retained). Each reference is the other cell's retained cloud, the equal-window mixture of its selected windows (1 window x 10,000 draws), with that cell's own SGM; color = the share of the reference's draws strictly farther from its SGM than the candidate SGM, distances in estimator coordinates divided by the prior widths; the self comparison is gray (excluded). The matrix is asymmetric because each reference has its own spread. Right strip: the row median over the other cells, the raw score on the same 0–1 scale, not a sampling probability.*
