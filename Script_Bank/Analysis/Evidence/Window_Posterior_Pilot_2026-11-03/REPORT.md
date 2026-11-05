# Detector window posteriors: existing-results review and two-cell joint-draw pilot (2026-11-03)

Response to `Special_Analyses/DETECTOR_WINDOW_POSTERIOR_HANDOFF_2026-11-03/CLAUDE_HANDOFF.md`. Both authorized parts
were run on the PC (profile `mars_pc`, package 0.1.44, repository HEAD `3824fd1` plus the uncommitted analysis files
listed in section 1). Nothing was selected; the SEL artifacts, the adopted Experiment products and the estimators are
unchanged (checksums verified before and after, section 6).

## 1. Code, inputs and outputs

New repository companions (Script_Bank/Analysis convention, each with a `.md` note):

- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Review.py` (CPU): validates an Experiment
  product, exports every window's MAP, SGM, median and intervals in physical units, draws the pilot-cell
  distributions, time series and pairwise plots, inventories all cells, reproduces the condition-wide summary.
- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Draws.py` (GPU): retains fresh joint draws
  per window through the engine's own `read_cell_chunks`, `load_estimator`, `posterior_summary` (`return_samples`,
  `return_sgm`) and `sample_geometric_median` with the prior-width scale; joins the source MAP, quantiles and SGM by
  `(kind_index, cell, chunk)`; examines the joint structure. Route: collection companion, no Experiment rerun, no
  MAP re-optimization, no engine change.

Inputs (data bank `/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/`):

| condition | Experiment product (`Posit/`) | sha256 | estimator weights sha256 (loaded = recorded) | recording |
|---|---|---|---|---|
| FAB | `..._DETECTOR_FAB_2S_50FPS_C128V42A_MAP_Experiment/<same>.npz` | `addc6fc1…9737` | `01a2d0c1…6d7` | `Experiment/SPT_Data_MET_FAB_INLB_S-BSST712/Experiment_FAB_Cell_26_20S_RAW.tif` (sha256 `953a30b1…0c34`, matches the handoff inventory) |
| INLB | `..._DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_MAP_Experiment/<same>.npz` | `bc764157…f1ef` | `178e7dd9…92e8` | `Experiment_INLB_Cell_11_20S_RAW.tif` (staged from the PC complete bank on 2026-11-03 under the validated index rule; the cross-check against the JUWELS `Catalog_Note_Experiment.tsv` is still pending and is not claimed) |

Reference displayed: `SEL` `selection_user` Nuisance_DLI artifacts (FAB `fe9cb5ba…59ad`, INLB `2b2d98f9…ac6f`).

Outputs (new folders under `Posit/`, existing folders refused):

| folder | contents |
|---|---|
| `..._DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Review/` | `window_estimates.csv` (600 x 6 rows), `window_vectors.csv`, `cell_inventory.csv`, `condition_interval_summary.csv`, `figures/{ecdf,timeseries,pairwise}_cell_26.png`, `figures/cells_overview.png`, `figures/pooled_mixture.png`, `report.md`, `provenance.json` |
| `..._DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Review/` | the same for cell 11 |
| `..._DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws/` | `<same>.npz` (10 x 10000 x 6 float32 cloud + fresh summaries + joined source references + collection manifest), `window_summaries.csv`, `correlations.csv`, `joint_depth.csv`, `figures/joint_cell_26_chunk_{0,4,9}.png`, `figures/marginals_cell_26.png`, `figures/correlations_cell_26.png`, `report.md`, `provenance.json` |
| `..._DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws/` | the same for cell 11 |

Commands, logs and the backup copies of the two draw collections (sha256 in `SHA256SUMS`) are in this folder
(`Script_Bank/Analysis/Evidence/Window_Posterior_Pilot_2026-11-03/`), which the workspace backup covers.

## 2. Validation

Review (both conditions, 10/10 checks PASS): 600 unique `(kind_index, cell, chunk)`; 60 cells with chunks 0..9
exactly; one kind; parameter order `mu_r, sigma_r, mu_pc, sigma_pc, prob_photo_bleach, lambda_rate`; window
geometry 100/100/1000 frames; finite estimates; monotone quantiles; estimator artifact present; the handoff's
`VERIFIED_INTERVAL_SUMMARIES.csv` reproduced with zero relative deviation on every column and equal SEL-inside counts.

Draws (both cells, 13/13 checks PASS): loaded weights checksum equals the one the source product recorded (checked
before sampling); cloud shape (10, 10000, 6) float32, finite; ten window identities per cell, no duplicates;
quantiles recomputed from the saved cloud equal the saved ones (max 3.4e-13 in log10); the saved SGM is the cloud row
at its index and the medoid recomputed with the prior-width scale returns the same row; physical endpoints are ten to
the power of the stored quantiles; source references joined by identity with recorded row indices; source product,
reference artifact and estimator checksums unchanged; implementation hash unchanged during the run and identical to
the one the source Experiment recorded (`c8641626…`).

Runtime and storage (PC, Quadro T2000 4 GB, torch 2.7.1+cu118, seed 20261103): 2.3 s per window after a 3 to 5 s
first window, 25 s of sampling per cell, 47 s wall; peak GPU memory 914 MiB; 2.1 MiB compressed per cell (2.3 MiB of
float32 draws). The 60-cell extension would be about 25 minutes and 130 MiB per condition at this size.

Monte Carlo reproducibility (fresh versus source summaries, same estimator, different draws): medians agree within
0.004 dex for the four spatial and brightness parameters and lambda, within 0.017 dex for bleaching; SGMs within
0.03 dex except bleaching (0.06 dex), as expected for the medoid of a nearly flat marginal.

## 3. Existing results: per-window distributions and time courses

Condition-wide (600 windows each; a summary of summaries): the per-window medians and 90 % widths of the handoff CSV
reproduce. SEL lies inside the window's own 90 % interval in FAB 356/600 (`mu_r`), 469 (`sigma_r`), 593 (`mu_pc`),
505 (`sigma_pc`), 600 (`prob_photo_bleach`), 600 (`lambda_rate`) windows; in INLB 426, 554, 257, 518, 600, 422.
The MAP lies inside its own window's 90 % interval in every window of both conditions.

Between cells (cell medians of the window medians; `cell_inventory.csv`): FAB `mu_r` 1.60 to 2.03 px, `mu_pc` 97 to
113 photons, `lambda_rate` 5.4 to 9.1 /s, `prob_photo_bleach` 0.06 to 0.16; INLB `mu_pc` 57 to 385 photons (the
widest), `lambda_rate` 1.3 to 7.8 /s, `mu_r` 1.85 to 2.33 px, `sigma_pc` 0.63 to 1.02. Within a cell the ratio of
the largest to the smallest window median is typically 1.1 to 1.2 (FAB) but 2.5 for INLB `mu_pc` (up to 12).

Fab cell 26 (`timeseries_cell_26.png`): the opening window stands apart on `mu_r` (1.74 then 1.64 to 1.67),
`sigma_r` (0.21 then 0.14 to 0.16), `sigma_pc` (0.63 then 0.55 to 0.60) and `lambda_rate` (6.9 then 7.7 to 8.5); the
other nine windows are mutually consistent within their 50 % bands. `prob_photo_bleach` has a 90 % band of 0.013 to
0.33 in every window (the prior is 0.01 to 0.32) with medians 0.074 to 0.107 rising along the recording; the MAP sits
at 0.13 to 0.16, above every median, as the known optimizer behavior on a flat marginal. The three statistics agree
within 0.01 dex on the other five parameters.

InlB cell 11 (`timeseries_cell_11.png`): `mu_pc` moves 281 (0 to 2 s), 168, 91, 117, 81, 100, 153, 153, 158, 204
photons, with 90 % widths of 75 to 165 photons, so the windows exclude each other's values; `sigma_pc` moves 0.80 to
0.97 and down to 0.66, anticorrelated with `mu_pc` across windows; `mu_r` falls 2.33 to 2.19 px (every window above the
2.0 px prior edge, the documented InlB limitation); `lambda_rate` 1.35 to 1.96 /s with 90 % bands 0.8 to 4.4;
`prob_photo_bleach` as in Fab (flat, medians 0.09 to 0.11).

## 4. Joint structure inside the windows (fresh clouds)

Correlations in estimator coordinates (Pearson, Spearman beside it in the folders; physical-coordinate Pearson in
`correlations.csv`):

| pair | FAB cell 26, chunks 0..9 | INLB cell 11, chunks 0..9 |
|---|---|---|
| `sigma_pc` x `lambda_rate` | −0.94 to −0.83 (every window) | −0.98 to −0.92 (every window) |
| `mu_pc` x `sigma_pc` | −0.58 (chunk 0) decaying to −0.23 (chunk 9) | −0.08 to −0.69 (strongest mid-recording) |
| `mu_pc` x `lambda_rate` | +0.38 (chunk 0) to about 0 | −0.03 to +0.61 (mid-recording) |
| `mu_pc` x `prob_photo_bleach` | about +0.1 | −0.35 to −0.43 |
| `sigma_pc` x `prob_photo_bleach` | about 0 | +0.13 to +0.37 |
| `mu_r` x `sigma_r`, `mu_r` x `mu_pc` | within ±0.15 | +0.05 to +0.34 |

The `sigma_pc` x `lambda_rate` ridge is the one strong, universal joint feature: inside every window the posterior
trades a wider per-dye brightness spread against a slower flicker rate (log scales: nearly a straight ridge, see the
`joint_cell_*_chunk_*.png` panels). Ranks and Pearson agree, and the contours show one region of mass per window; no
window shows two modes. In InlB the brightness block (`mu_pc`, `sigma_pc`, `lambda_rate`, and weakly bleaching)
is jointly confounded in the middle of the recording where `mu_pc` is lowest.

Joint depth of the references (`joint_depth.csv`; share of draws farther from the window's SGM than the reference, in
prior-scaled estimator coordinates):

| reference | FAB cell 26 | INLB cell 11 |
|---|---|---|
| SEL | 0.55 to 0.92; inside every marginal 90 % in 8/10 windows (not chunks 6, 7: `mu_r` and `sigma_r` below SEL) | 0.21 to 0.79; inside every marginal 90 % in 5/10 windows (`mu_pc` excludes 152.9 in chunks 0, 2, 4, 5, 9) |
| source MAP | 0.81 to 0.91; 10/10 | 0.93 to 1.00; 10/10 |

Outside-prior draws (unrestricted flow draws, retained): FAB 25 to 52 % of complete vectors per window, driven by
`mu_pc` below 100 photons (15 to 32 % of draws), `lambda_rate` above 10 /s (7 to 21 %), bleaching above 0.32 (4 to
8 %), `sigma_r` below 0.1 (0 to 4 %); INLB 100 % of vectors to the stated precision (99,997 of 100,000 draws have `mu_r` above the 1.995 px prior ceiling), and on the other
coordinates `mu_pc` below 100 photons in 0 to 77 % of a window's draws, `sigma_pc` above 1.0 in 0 to 42 %,
`lambda_rate` below 1 /s in 1 to 25 %.

## 5. Reading, and the three review questions

**Does SEL represent these pilot recordings and windows?** Fab cell 26: yes for `mu_pc` (105 against 103 to 110),
`sigma_pc` (0.585 against 0.55 to 0.63) and `lambda_rate` (8 inside every window, near the medians after 4 s), with
two qualifications: the cell's `mu_r` sits at 1.64 to 1.67 px after the first window (SEL 1.734 is the window-0
value and the condition median; it leaves the 90 % interval in chunks 6 and 7), and `sigma_r` falls from 0.21 to 0.14
(SEL 0.177 crosses the cell's trajectory). Bleaching 0.05 lies inside every window's 90 % interval because that
interval spans the prior; the posterior neither supports nor contradicts it. InlB cell 11: no single vector
represents this recording's windows. `mu_pc` changes by a factor 3.5 between windows whose own 90 % intervals are a
factor 1.5 to 2 wide, and SEL's 152.9 is excluded by half of them; SEL's `sigma_pc`/`lambda_rate` pair (0.86, 2) sits
near but above the ridge the windows define (their SGMs pair 0.77 with 1.46, 0.91 with 1.98, 0.63 with 1.42).

**Which parameters vary mainly between windows, which have broad within-window uncertainty, which vary together?**
Between windows (shift larger than the within-window width): InlB `mu_pc` and `sigma_pc`, InlB `mu_r` (a steady
decline of 0.03 dex against a 90 % width of 0.03 dex), Fab `sigma_r` and the opening window of Fab `mu_r`,
`sigma_pc`, `lambda_rate`. Broad within a window: `prob_photo_bleach` (90 % width 0.27 to 0.32, the prior's width;
unidentified in a 2 s window), `lambda_rate` (Fab 3.7 to 11 /s; InlB 0.8 to 4.4 /s), InlB `mu_pc` and `sigma_pc`.
Vary together inside a posterior: `sigma_pc` with `lambda_rate` (strong, every window, both conditions), `mu_pc`
with `sigma_pc` (moderate, strongest early in Fab and mid-recording in InlB), and in InlB `mu_pc` with
`lambda_rate` and with bleaching. Consequence for the rendering trials run so far: raising `mu_pc` alone or
`sigma_pc` alone (with `lambda_rate` fixed) steps off the ridge the posterior supports; a wider brightness spread is
supported only together with a slower flicker rate.

**Which complete, data-supported imaging alternatives merit a later predictive comparison?** Candidates are joint
draws of named windows (physical units; `window_summaries.csv` column `fresh_sgm`):

| candidate | `mu_r` | `sigma_r` | `mu_pc` | `sigma_pc` | `prob_photo_bleach` | `lambda_rate` |
|---|---|---|---|---|---|---|
| FAB cell 26, chunk 0 SGM (the opening 2 s, the window the histogram comparisons start with) | 1.752 | 0.2025 | 110.8 | 0.628 | 0.067 | 7.00 |
| FAB cell 26, chunk 4 SGM (a typical later window) | 1.642 | 0.155 | 107.6 | 0.559 | 0.085 | 7.76 |
| INLB cell 11, chunk 0 SGM (the opening 2 s) | 2.333 | 0.394 | 284.3 | 0.773 | 0.093 | 1.46 |
| INLB cell 11, chunk 4 SGM (the dim middle) | 2.303 | 0.361 | 81.0 | 0.912 | 0.108 | 1.98 |
| INLB cell 11, chunk 9 SGM (the closing 2 s) | 2.177 | 0.437 | 197.7 | 0.629 | 0.085 | 1.42 |

Each is a realized draw of its window, so its coordinates co-occurred; none carries the window's joint uncertainty,
and a window-specific vector applies to that window's 2 s, not to the 20 s render. The InlB opening-window candidate
(284 photons per dye, spread 0.77, 1.46 /s) is the posterior's own account of the bright opening band the renders
lacked; it was never among the one-coordinate arms. For bleaching, the posterior offers no information in 2 s windows
and the declared empirical value stands on its separate rationale. These are candidates for a comparison the lead may
authorize, not selections.

## 6. What this does not claim

No catalog cross-check against the JUWELS TSV was run (unavailable from the PC); the InlB recording identity rests on
the validated index rule. The draws are fresh Monte Carlo samples under the same estimator; they are not the source
run's draws. No new rendering, no artifact, no training, no HPC submission; the stronger `sigma_pc` arms remain unrun.
