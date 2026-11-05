# Experiment diagnostics - SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A
Run: 2026-11-05 14:14:22 UTC
Note: Review copy (PILOT2CELLSR1) regenerated on the CPU from the saved product SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment.npz; no estimation was repeated. Adopted report: /media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment/report.md.

## Checks - 7 passed, 0 failed

| check | result | detail |
|---|---|---|
| estimates_nonempty | PASS | 600 windows estimated over experimental recordings |
| three_point_estimates_present | PASS | map_estimate, posterior_quantiles (median at level 0.50) and posterior_sgm stored for every window |
| aggregation_coverage_complete | PASS | FAB: 2 cells x 10 positions |
| aggregation_equal_window_weights | PASS | 10000 retained draws in every window, so concatenation realizes equal window and cell weights |
| aggregation_contract_matches_product | PASS | checkpoint 01a2d0c17b99…, 6 keys, scale [0.3, 0.75, 0.75, 0.75, 1.5, 1.0], unrestricted, 10000 draws, geometry {'n_frames': 100, 'span_frames': 1000, 'step_frames': 100}; input conversion verified: every source (2) records the same input conversion; the reference product does not record one |
| aggregation_identities_in_product | PASS | 20 supplied windows, all windows of this product |
| aggregation_sgm_members | PASS | 39 groups; every SGM is the retained draw its source identity names |

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

**Coverage.** MET-FAB: 2 of 60 cells of this product (cells [4, 26]), 20 of 600 windows, positions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10000 retained draws per window. This is a pilot subset, not the complete condition: MET-FAB 2 of 60 cells (cells [4, 26]), 20 of 600 windows. The original MAP and summary references above come from the adopted product; the aggregation summaries come from the retained collections named below, fresh draws of the same estimator that do not reconstruct the original sampling attempt.

**Window sets.** all = chunks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; opening = chunks [0]. `all` is every position of the declared window geometry (100-frame windows, step 100 frames, recording span 1000 frames: positions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), and every included cell must carry every position a set requests; `opening` is the first window. A deliberate subset is named explicitly; nothing is dropped silently.

**Retained clouds.** `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws.npz`, sha256 8a0fc325338f…; `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws_CELL4.npz` (collection label CELL4), sha256 5d517eb42cbf…. Retained draw collections (fresh clouds of the adopted estimator, collected after the Experiment run). Every supplied identity (condition, cell, chunk) is a window of this product. Input conversion (preprocessing): verified: every source (2) records the same input conversion; the reference product does not record one; recorded as `read_cell_chunks(domain='stored'): 16-bit raw frames to 8-bit, normalize_video in the sampler`. Disclosed differences between the sources (not part of the computation contract): collection_label = [None, 'CELL4'], written_at = ['2026-11-03 15:19:28 UTC', '2026-11-03 17:16:10 UTC'].

**Methods.** Distances in estimator coordinates divided by the prior widths [0.3, 0.75, 0.75, 0.75, 1.5, 1.0] (the per-window SGM convention). A single-window group reuses the window's stored exact sample medoid; a pooled group above 10000 rows takes the collection kernel's Weiszfeld geometric median snapped to the nearest retained draw (`weiszfeld_snap`, an approximation to the exact medoid, recorded per group). Pooled marginal quantiles are mixture quantiles of the pooled draws in estimator coordinates (the stage's interpolation), endpoints transformed to physical units; representative-distribution quantiles (median, IQR, central 90 % range over a collection's representatives) use the inverse weighted empirical CDF and are empirical dispersion, not credible or confidence intervals. Unrestricted draws stay unrestricted: outside-prior mass is counted, never clipped. SGM is the default candidate statistic; the marginal-median vector of each group is stored beside it, labeled as a coordinate-wise composite. Aggregation took 1.1 s on the CPU.

**Reference.** SEL: mu_r 1.734, sigma_r 0.177, mu_pc 105, sigma_pc 0.5848, prob_photo_bleach 0.05, lambda_rate 8 (physical), loaded from /media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_SEL_Nuisance_DLI.npz. Its current method, per the artifact's own decision record (the per-parameter table below): the median over the per-window posterior medians of the adopted Experiment product, transformed to physical units, followed by the documented parameter-specific selections. A historical fixed scenario for comparison, neither reconstructed from the draws nor mixed into any collection.

**Supporting files (this directory).** [aggregation_candidates.npz](aggregation_candidates.npz) (every group's SGM and median vectors, mixture quantiles, identities, methods, contract, sources and the target binding), [aggregation_groups.csv](aggregation_groups.csv), [aggregation_comparison.csv](aggregation_comparison.csv), [aggregation_manifest.json](aggregation_manifest.json). Figures: [aggregation_FAB_all_marginals.png](figures/aggregation_FAB_all_marginals.png), [aggregation_FAB_all_joint.png](figures/aggregation_FAB_all_joint.png), [aggregation_FAB_opening_marginals.png](figures/aggregation_FAB_opening_marginals.png), [aggregation_FAB_opening_joint.png](figures/aggregation_FAB_opening_joint.png), [aggregation_FAB_all_temporal.png](figures/aggregation_FAB_all_temporal.png), [aggregation_FAB_window_selection.png](figures/aggregation_FAB_window_selection.png) (captions at the end of the report).

Reading rules: an acknowledged training-bound extrapolation stays visible in the outside-prior columns (the InlB PSF width lies above its prior ceiling); pairwise correlations in the joint figures are descriptive, not selection constraints; nothing here declares a winner, and membership of a reference inside intervals computed from the same draws is a compatibility diagnostic, not a validation.

## Experimental aggregation and nuisance candidates: constructions

| ID | construction | calculation | representatives per window set | interpretation |
|---|---|---|---|---|
| `reference` (SEL) | Historical fixed scenario | median over the per-window posterior medians of the adopted Experiment product (physical units), followed by the documented parameter-specific selections (per-parameter record below); not an SGM of draws | 1 | one fixed scenario; comparison only |
| `per_cell` | One representative per cell | pool all retained draws of the cell's selected windows, equal weight per window, and select the SGM | all: 2 (MET-FAB); opening: 2 (MET-FAB) | variation between cell representatives over the declared acquisition period |
| `per_window_position` | One representative per window position | at each selected position, pool the retained draws across the included cells, equal weight per cell, and select the SGM | all: 10 (MET-FAB); opening: 1 (MET-FAB) | variation between acquisition-period representatives, cell differences compressed at each position |
| `global` | One representative of the selected collection | pool the retained draws with equal cell weight and equal selected-window weight within each cell, and select the SGM | all: 1 (MET-FAB); opening: 1 (MET-FAB) | an overall joint center of the selected collection |
| `per_cell_window` | One representative per cell-window pair | the SGM of each included cell-window posterior | all: 20 (MET-FAB); opening: 2 (MET-FAB) | variation between window representatives, cell and window identities retained |

*the canonical options; counts are per condition. An SGM of window SGMs is not an active construction; no aggregate MAP exists.*

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
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 4 |  |
| per_cell[opening] | per_cell_window[opening] | MET-FAB | 26 |  |
| global[opening] | per_window_position[opening] | MET-FAB |  |  |

*a single-window set makes per_cell equal per_cell_window and per_window_position equal global; a single cell makes per_window_position equal per_cell_window and global equal per_cell; the computed group is reused.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set all]: SGM representatives (physical units)

| construction | parameter | n | median | IQR (q25 – q75) | central 90 % range | median vs reference | outside-prior mass |
|---|---|---|---|---|---|---|---|
| per_cell | mu_r | 2 | 1.649 | 1.649 – 1.821 | 1.649 – 1.821 | -4.9 % (-0.022 dex) | 0.0 % |
| per_cell | sigma_r | 2 | 0.1529 | 0.1529 – 0.1746 | 0.1529 – 0.1746 | -13.6 % (-0.064 dex) | 0.0 % |
| per_cell | mu_pc | 2 | 101.7 | 101.7 – 104.9 | 101.7 – 104.9 | -3.1 % (-0.014 dex) | 0.0 % |
| per_cell | sigma_pc | 2 | 0.5614 | 0.5614 – 0.5944 | 0.5614 – 0.5944 | -4.0 % (-0.018 dex) | 0.0 % |
| per_cell | prob_photo_bleach | 2 | 0.08756 | 0.08756 – 0.09389 | 0.08756 – 0.09389 | +75.1 % (+0.243 dex) | 0.0 % |
| per_cell | lambda_rate | 2 | 7.81 | 7.81 – 8.015 | 7.81 – 8.015 | -2.4 % (-0.010 dex) | 0.0 % |
| per_window_position | mu_r | 10 | 1.738 | 1.724 – 1.763 | 1.706 – 1.902 | +0.2 % (+0.001 dex) | 0.0 % |
| per_window_position | sigma_r | 10 | 0.1514 | 0.1503 – 0.1718 | 0.1448 – 0.275 | -14.5 % (-0.068 dex) | 0.0 % |
| per_window_position | mu_pc | 10 | 104.3 | 102.1 – 106.4 | 100.5 – 108 | -0.7 % (-0.003 dex) | 0.0 % |
| per_window_position | sigma_pc | 10 | 0.5611 | 0.5576 – 0.5738 | 0.5523 – 0.6257 | -4.0 % (-0.018 dex) | 0.0 % |
| per_window_position | prob_photo_bleach | 10 | 0.081 | 0.07989 – 0.09427 | 0.07717 – 0.1065 | +62.0 % (+0.210 dex) | 0.0 % |
| per_window_position | lambda_rate | 10 | 7.974 | 7.37 – 8.27 | 7.265 – 8.323 | -0.3 % (-0.001 dex) | 0.0 % |
| global | mu_r | 1 | 1.736 | 1.736 – 1.736 | 1.736 – 1.736 | +0.1 % (+0.000 dex) | 0.0 % |
| global | sigma_r | 1 | 0.1639 | 0.1639 – 0.1639 | 0.1639 – 0.1639 | -7.4 % (-0.033 dex) | 0.0 % |
| global | mu_pc | 1 | 103.9 | 103.9 – 103.9 | 103.9 – 103.9 | -1.0 % (-0.004 dex) | 0.0 % |
| global | sigma_pc | 1 | 0.5696 | 0.5696 – 0.5696 | 0.5696 – 0.5696 | -2.6 % (-0.011 dex) | 0.0 % |
| global | prob_photo_bleach | 1 | 0.09264 | 0.09264 – 0.09264 | 0.09264 – 0.09264 | +85.3 % (+0.268 dex) | 0.0 % |
| global | lambda_rate | 1 | 7.595 | 7.595 – 7.595 | 7.595 – 7.595 | -5.1 % (-0.023 dex) | 0.0 % |
| per_cell_window | mu_r | 20 | 1.752 | 1.663 – 1.81 | 1.632 – 1.926 | +1.0 % (+0.004 dex) | 5.0 % |
| per_cell_window | sigma_r | 20 | 0.1519 | 0.1439 – 0.1711 | 0.1314 – 0.2125 | -14.2 % (-0.066 dex) | 0.0 % |
| per_cell_window | mu_pc | 20 | 104.6 | 102.3 – 105.4 | 101.1 – 108.9 | -0.3 % (-0.001 dex) | 0.0 % |
| per_cell_window | sigma_pc | 20 | 0.5638 | 0.5534 – 0.5906 | 0.5306 – 0.6018 | -3.6 % (-0.016 dex) | 0.0 % |
| per_cell_window | prob_photo_bleach | 20 | 0.09198 | 0.0841 – 0.09519 | 0.06674 – 0.1093 | +84.0 % (+0.265 dex) | 0.0 % |
| per_cell_window | lambda_rate | 20 | 7.802 | 7.542 – 8.068 | 6.997 – 8.32 | -2.5 % (-0.011 dex) | 0.0 % |

*window set all = chunks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; quantiles over the collection's representatives by the inverse weighted empirical CDF (empirical dispersion across representatives, not posterior credible intervals); for one representative every column is that vector. 'outside-prior mass' is the share of representatives beyond the row's prior box.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set all]: global representative and reference (physical units)

| parameter | global SGM (a draw) | global median vector (composite) | pooled 90 % interval | reference | global SGM vs reference | pooled draws outside prior |
|---|---|---|---|---|---|---|
| mu_r | 1.736 | 1.745 | 1.599 – 1.975 | 1.734 | +0.1 % | 3.9 % |
| sigma_r | 0.1639 | 0.1604 | 0.1146 – 0.2748 | 0.177 | -7.4 % | 1.2 % |
| mu_pc | 103.9 | 104.5 | 93.2 – 119.2 | 105 | -1.0 % | 26.4 % |
| sigma_pc | 0.5696 | 0.5713 | 0.5054 – 0.6774 | 0.5848 | -2.6 % | 0.0 % |
| prob_photo_bleach | 0.09264 | 0.09555 | 0.01469 – 0.3213 | 0.05 | +85.3 % | 6.2 % |
| lambda_rate | 7.595 | 8.087 | 4.79 – 11.09 | 8 | -5.1 % | 15.6 % |

*the global SGM is retained draw cell 26, chunk 4, index 9416 (weiszfeld_snap; 200000 pooled draws); the pooled 90 % interval is a mixture quantile of the pooled draws, a different quantity from the dispersion of representatives.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set opening]: SGM representatives (physical units)

| construction | parameter | n | median | IQR (q25 – q75) | central 90 % range | median vs reference | outside-prior mass |
|---|---|---|---|---|---|---|---|
| per_cell | mu_r | 2 | 1.752 | 1.752 – 2 | 1.752 – 2 | +1.0 % (+0.004 dex) | 50.0 % |
| per_cell | sigma_r | 2 | 0.2025 | 0.2025 – 0.3088 | 0.2025 – 0.3088 | +14.4 % (+0.058 dex) | 0.0 % |
| per_cell | mu_pc | 2 | 105 | 105 – 110.8 | 105 – 110.8 | +0.0 % (+0.000 dex) | 0.0 % |
| per_cell | sigma_pc | 2 | 0.5906 | 0.5906 – 0.6277 | 0.5906 – 0.6277 | +1.0 % (+0.004 dex) | 0.0 % |
| per_cell | prob_photo_bleach | 2 | 0.06674 | 0.06674 – 0.09488 | 0.06674 – 0.09488 | +33.5 % (+0.125 dex) | 0.0 % |
| per_cell | lambda_rate | 2 | 6.997 | 6.997 – 8.004 | 6.997 – 8.004 | -12.5 % (-0.058 dex) | 0.0 % |
| per_window_position | mu_r | 1 | 1.902 | 1.902 – 1.902 | 1.902 – 1.902 | +9.7 % (+0.040 dex) | 0.0 % |
| per_window_position | sigma_r | 1 | 0.275 | 0.275 – 0.275 | 0.275 – 0.275 | +55.3 % (+0.191 dex) | 0.0 % |
| per_window_position | mu_pc | 1 | 106.4 | 106.4 – 106.4 | 106.4 – 106.4 | +1.3 % (+0.006 dex) | 0.0 % |
| per_window_position | sigma_pc | 1 | 0.6257 | 0.6257 – 0.6257 | 0.6257 – 0.6257 | +7.0 % (+0.029 dex) | 0.0 % |
| per_window_position | prob_photo_bleach | 1 | 0.08009 | 0.08009 – 0.08009 | 0.08009 – 0.08009 | +60.2 % (+0.205 dex) | 0.0 % |
| per_window_position | lambda_rate | 1 | 8.028 | 8.028 – 8.028 | 8.028 – 8.028 | +0.3 % (+0.002 dex) | 0.0 % |
| global | mu_r | 1 | 1.902 | 1.902 – 1.902 | 1.902 – 1.902 | +9.7 % (+0.040 dex) | 0.0 % |
| global | sigma_r | 1 | 0.275 | 0.275 – 0.275 | 0.275 – 0.275 | +55.3 % (+0.191 dex) | 0.0 % |
| global | mu_pc | 1 | 106.4 | 106.4 – 106.4 | 106.4 – 106.4 | +1.3 % (+0.006 dex) | 0.0 % |
| global | sigma_pc | 1 | 0.6257 | 0.6257 – 0.6257 | 0.6257 – 0.6257 | +7.0 % (+0.029 dex) | 0.0 % |
| global | prob_photo_bleach | 1 | 0.08009 | 0.08009 – 0.08009 | 0.08009 – 0.08009 | +60.2 % (+0.205 dex) | 0.0 % |
| global | lambda_rate | 1 | 8.028 | 8.028 – 8.028 | 8.028 – 8.028 | +0.3 % (+0.002 dex) | 0.0 % |
| per_cell_window | mu_r | 2 | 1.752 | 1.752 – 2 | 1.752 – 2 | +1.0 % (+0.004 dex) | 50.0 % |
| per_cell_window | sigma_r | 2 | 0.2025 | 0.2025 – 0.3088 | 0.2025 – 0.3088 | +14.4 % (+0.058 dex) | 0.0 % |
| per_cell_window | mu_pc | 2 | 105 | 105 – 110.8 | 105 – 110.8 | +0.0 % (+0.000 dex) | 0.0 % |
| per_cell_window | sigma_pc | 2 | 0.5906 | 0.5906 – 0.6277 | 0.5906 – 0.6277 | +1.0 % (+0.004 dex) | 0.0 % |
| per_cell_window | prob_photo_bleach | 2 | 0.06674 | 0.06674 – 0.09488 | 0.06674 – 0.09488 | +33.5 % (+0.125 dex) | 0.0 % |
| per_cell_window | lambda_rate | 2 | 6.997 | 6.997 – 8.004 | 6.997 – 8.004 | -12.5 % (-0.058 dex) | 0.0 % |

*window set opening = chunks [0]; quantiles over the collection's representatives by the inverse weighted empirical CDF (empirical dispersion across representatives, not posterior credible intervals); for one representative every column is that vector. 'outside-prior mass' is the share of representatives beyond the row's prior box.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set opening]: global representative and reference (physical units)

| parameter | global SGM (a draw) | global median vector (composite) | pooled 90 % interval | reference | global SGM vs reference | pooled draws outside prior |
|---|---|---|---|---|---|---|
| mu_r | 1.902 | 1.883 | 1.668 – 2.083 | 1.734 | +9.7 % | 33.8 % |
| sigma_r | 0.275 | 0.2591 | 0.1724 – 0.3643 | 0.177 | +55.3 % | 0.0 % |
| mu_pc | 106.4 | 107.5 | 93.54 – 125.5 | 105 | +1.3 % | 19.5 % |
| sigma_pc | 0.6257 | 0.6061 | 0.5438 – 0.7487 | 0.5848 | +7.0 % | 0.0 % |
| prob_photo_bleach | 0.08009 | 0.08498 | 0.01374 – 0.2921 | 0.05 | +60.2 % | 4.0 % |
| lambda_rate | 8.028 | 7.769 | 4.18 – 10.76 | 8 | +0.3 % | 12.1 % |

*the global SGM is retained draw cell 26, chunk 0, index 4691 (weiszfeld_snap; 20000 pooled draws); the pooled 90 % interval is a mixture quantile of the pooled draws, a different quantity from the dispersion of representatives.*

## Experimental aggregation and nuisance candidates [MET-FAB, window set all]: source uncertainty against representative dispersion (physical units)

| parameter | within-window 90 % width (median over windows) | per-cell pooled 90 % width (median over cells) | global pooled 90 % interval | central 90 % range of per_cell SGMs | central 90 % range of per_cell_window SGMs |
|---|---|---|---|---|---|
| mu_r | 0.1711 | 0.2463 | 1.599 – 1.975 | 1.649 – 1.821 (n=2) | 1.632 – 1.926 (n=20) |
| sigma_r | 0.08982 | 0.1496 | 0.1146 – 0.2748 | 0.1529 – 0.1746 (n=2) | 0.1314 – 0.2125 (n=20) |
| mu_pc | 23.79 | 26 | 93.2 – 119.2 | 101.7 – 104.9 (n=2) | 101.1 – 108.9 (n=20) |
| sigma_pc | 0.1404 | 0.1677 | 0.5054 – 0.6774 | 0.5614 – 0.5944 (n=2) | 0.5306 – 0.6018 (n=20) |
| prob_photo_bleach | 0.3082 | 0.3065 | 0.01469 – 0.3213 | 0.08756 – 0.09389 (n=2) | 0.06674 – 0.1093 (n=20) |
| lambda_rate | 6.105 | 6.296 | 4.79 – 11.09 | 7.81 – 8.015 (n=2) | 6.997 – 8.32 (n=20) |

*the first three columns are intervals of input or pooled posteriors (retained posterior uncertainty; mixture quantiles for the pooled ones); the last two are the dispersion of selected representatives (compression). They are different quantities and are never read as calibrated uncertainty.*

## Experimental aggregation and nuisance candidates [MET-FAB]: SGM versus marginal-median vector of the per_cell groups (median / max |difference|, stored coordinates)

| window set | mu_r | sigma_r | mu_pc | sigma_pc | prob_photo_bleach | lambda_rate |
|---|---|---|---|---|---|---|
| all | 0.003 / 0.004 | 0.006 / 0.006 | 0.005 / 0.009 | 0.005 / 0.010 | 0.022 / 0.022 | 0.010 / 0.015 |
| opening | 0.003 / 0.004 | 0.007 / 0.013 | 0.003 / 0.003 | 0.002 / 0.004 | 0.022 / 0.043 | 0.013 / 0.024 |

*the marginal-median vector is a coordinate-wise composite whose coordinates need not have occurred together; it is stored beside the SGM and must be named explicitly by a selection that uses it.*

## What these mean

A check **PASS** means the invariant held; a **FAIL** aborts the run immediately with a located error message.

- **estimates_nonempty** -- at least one experimental window was estimated.
- **three_point_estimates_present** -- the product contract: no estimate is reported without the other two.
- **aggregation_coverage_complete** -- every included cell carries every requested window position; conditions are never pooled together.
- **aggregation_equal_window_weights** -- the equal weights of the mixtures are realized by equal draw counts.
- **aggregation_contract_matches_product** -- the retained clouds and this product share one computation contract; the input conversion is compared between the supplied collections as well, and 'unavailable' means no source records it (not verified).
- **aggregation_identities_in_product** -- every supplied (condition, cell, chunk) identity belongs to the source Experiment product.
- **aggregation_sgm_members** -- each group's SGM equals the retained draw (cell, chunk, draw index) it records; a median vector is never claimed as a draw.

## Figures

![window_drift_FAB](figures/window_drift_FAB.png)

*Within-recording drift, condition MET-FAB: the MAP, posterior median and SGM as the median across recordings at each window, over the per-window central 50 % and 90 % intervals of the draws (median across recordings). Dotted red lines are the prior bounds. The three lines share one draw set, so the band is drawn once.*

![experiment_mu_r](figures/experiment_mu_r.png)

*mu_r ($\mu_{r}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table.*

![experiment_sigma_r](figures/experiment_sigma_r.png)

*sigma_r ($\sigma_{r}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table.*

![experiment_mu_pc](figures/experiment_mu_pc.png)

*mu_pc ($\mu_{pc}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table.*

![experiment_sigma_pc](figures/experiment_sigma_pc.png)

*sigma_pc ($\sigma_{pc}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table.*

![experiment_prob_photo_bleach](figures/experiment_prob_photo_bleach.png)

*prob_photo_bleach ($\rho_{pb}$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table.*

![experiment_lambda_rate](figures/experiment_lambda_rate.png)

*lambda_rate ($\lambda$). Left: per-condition distribution of the MAP (pooled over (cell x chunk)). Right: each window's median +/- IQR of its draws per condition (within-window spread), the MAP overlaid. The SGM is tabulated above and its gap to the median is in the agreement table.*

![point_estimates_FAB_mu_r](figures/point_estimates_FAB_mu_r.png)

*mu_r ($\mu_{r}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn.*

![point_estimates_FAB_sigma_r](figures/point_estimates_FAB_sigma_r.png)

*sigma_r ($\sigma_{r}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn.*

![point_estimates_FAB_mu_pc](figures/point_estimates_FAB_mu_pc.png)

*mu_pc ($\mu_{pc}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn.*

![point_estimates_FAB_sigma_pc](figures/point_estimates_FAB_sigma_pc.png)

*sigma_pc ($\sigma_{pc}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn.*

![point_estimates_FAB_prob_photo_bleach](figures/point_estimates_FAB_prob_photo_bleach.png)

*prob_photo_bleach ($\rho_{pb}$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn.*

![point_estimates_FAB_lambda_rate](figures/point_estimates_FAB_lambda_rate.png)

*lambda_rate ($\lambda$), MET-FAB. The three point estimates per window, one panel each (MAP, median, SGM): recordings along the x axis ordered by their median value, one point per window, the recording's median of that estimate as the line, and the IQR of the draws (median over the recording's windows of the per-window Q25 and Q75) as the band, drawn once and identical on every panel. The estimate axis is clipped to the prior box widened by 40 %; values beyond it are counted in the tables, not drawn.*

![aggregation_FAB_all_marginals](figures/aggregation_FAB_all_marginals.png)

*MET-FAB, window set all = chunks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]: normalized empirical CDFs of the SGM representatives of every construction, physical units; the global representative and the reference are vertical lines. Dispersion across representatives, not posterior uncertainty.*

![aggregation_FAB_all_joint](figures/aggregation_FAB_all_joint.png)

*MET-FAB, window set all = chunks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]: pairwise projections of the representatives (co-occurring coordinates of retained draws); correlations are descriptive, no relation is imposed.*

![aggregation_FAB_all_temporal](figures/aggregation_FAB_all_temporal.png)

*MET-FAB, window set all = chunks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]: per_cell_window SGMs (thin lines, one per cell: across-cell dispersion), per_window_position SGMs (bold), the global SGM and the reference (horizontal lines), over the within-window 50 % and 90 % intervals of the draws (median across cells: retained posterior uncertainty). x = window start in seconds (step 2 s).*

![aggregation_FAB_opening_marginals](figures/aggregation_FAB_opening_marginals.png)

*MET-FAB, window set opening = chunks [0]: normalized empirical CDFs of the SGM representatives of every construction, physical units; the global representative and the reference are vertical lines. Dispersion across representatives, not posterior uncertainty.*

![aggregation_FAB_opening_joint](figures/aggregation_FAB_opening_joint.png)

*MET-FAB, window set opening = chunks [0]: pairwise projections of the representatives (co-occurring coordinates of retained draws); correlations are descriptive, no relation is imposed.*

![aggregation_FAB_window_selection](figures/aggregation_FAB_window_selection.png)

*MET-FAB: the per_cell SGM of every cell under each requested window set (all = chunks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; opening = chunks [0]), cell identity preserved (a thin line joins the same cell), with the global SGM of each set and the reference.*
