# Window-posterior extension: Fab cell 4 and InlB cell 42 (2026-11-03)

Indexed report for the extension handoff (`Special_Analyses/DETECTOR_WINDOW_POSTERIOR_HANDOFF_2026-11-03/ADDITIONAL_CELLS_HANDOFF.md`).
Two additional local draw collections were made with the adopted estimators, the same aggregation comparison was repeated for
each, and the four cells now holding retained joint draws are tabulated side by side. Everything here is descriptive: no
aggregation rule, no nuisance selection, no artifact, no rendering, no training, no HPC submission. The first-pilot report
(`../Window_Posterior_Pilot_2026-11-03/REPORT.md`) carries the definitions this report reuses.

## 1. Index

| item | location |
|---|---|
| Fab 4 collection (clouds, fresh summaries, joined source references, per-window tables, correlations, joint depth, figures, report, provenance) | data bank `Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws_CELL4/` |
| InlB 42 collection (same contents) | data bank `Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws_CELL42/` |
| Backup copies of both `.npz` collections, their `report.md` and `provenance.json`, the run logs, `SHA256SUMS` | this folder |
| Fab 4 aggregation comparison (comparison.csv, window_sgms.csv, result.json, checks.json, comparison.png, README.md) | `Special_Analyses/FAB4_AGGREGATION_COMPARISON_2026-11-03/` |
| InlB 42 aggregation comparison (same contents) | `Special_Analyses/INLB42_AGGREGATION_COMPARISON_2026-11-03/` |
| Four-cell comparison (four_cell_comparison.csv, per_cell_depth.csv, result.json, four_cell_comparison.png, README.md) | `Special_Analyses/FOUR_CELL_AGGREGATION_COMPARISON_2026-11-03/` |
| Helpers | `aggregation_comparison.py`, `four_cell_comparison.py` (this folder); the draws companion and its note in `Script_Bank/Analysis/` |

Per-window tables (fresh median, SGM, 90 % interval and width, source MAP and median, SEL membership, outside-prior share) for
every parameter: each collection's `report.md` (section "Cell c: fresh summaries per window") and `window_summaries.csv`.
Joint diagnostics: `correlations.csv`, `joint_depth.csv`, `figures/joint_cell_<c>_chunk_{0,4,9}.png`, `figures/marginals_cell_<c>.png`,
`figures/correlations_cell_<c>.png`.

## 2. What was done

1. The draws companion gained an output-only `--collection-label` (alphanumeric; appends `_<LABEL>` to the collection folder and
   file names; recorded in the collection manifest, the report and `provenance.json`; the unlabeled default is unchanged;
   `--artifact-tag` still selects the estimator and the source Experiment; an existing destination is still refused). Dry-runs
   showed the adopted inputs (Fab C128V42A, weights sha256 `01a2d0c1…a6d7`, source product `addc6fc1…9737`; InlB
   C128V42AP3LR32, weights `178e7dd9…92b8`, source product `bc764157…f1ef`) and the two distinct new destinations; the
   unlabeled default against the existing pilot folder was refused, and a repeat against the written `CELL42` folder was refused.
2. Two local collections with the matched settings: ten non-overlapping 2 s windows (100 frames at 50 FPS), 10,000 complete
   six-parameter draws per window, unrestricted sampling with outside-prior draws retained, seed 20261103, the same input
   conversion, normalization, estimator loader, `posterior_summary`, parameter order and per-window SGM scale (the prior widths
   [0.3, 0.75, 0.75, 0.75, 1.5, 1.0] in log10 coordinates); source MAP, quantiles and SGM joined by (kind_index, cell, chunk); no
   MAP re-optimized; fresh quantiles, medians and SGMs computed from the retained clouds.
3. The aggregation comparison of the two completed reviews repeated by a cell-agnostic helper that reads identities, parameter
   order, prior box, SGM scale, reference vector and checkpoint from the collection's own metadata. Validation: pointed at the
   Fab-26 and InlB-11 collections, it reproduces `Special_Analyses/FAB26_…/comparison.csv` and `INLB11_…/comparison.csv` with
   zero numerical difference in every shared column, the same selected flat index, source window and draw index, the same method
   (`weiszfeld_snap`) and the same alternate-metric index.
4. A four-cell tabulation and figure (Fab 26, Fab 4, InlB 11, InlB 42) from the four `result.json` files and the collections'
   `joint_depth.csv` / `window_summaries.csv`; nothing recomputed from the draws.

## 3. Checks

Draw collections (13 checks each, all PASS for both cells; `draws_FAB4.log`, `draws_INLB42.log`): the loaded weights equal the
checksum the source Experiment recorded (verified before sampling); cloud shape (10, 10000, 6) float32; every draw finite; each
cell carries chunks 0..9 once; parameter order and log10 transform as in the source; quantiles recomputed from the saved cloud
agree with the saved ones (max |diff| 0); the saved SGM is the cloud row at its index and the medoid recomputed with the
prior-width scale returns the same row; physical endpoints equal 10 to the stored log10 quantiles; source references joined by
identity with recorded row indices; checkpoint hash recorded; source product unchanged (sha256 above); SEL artifact unchanged
(Fab `fe9cb5ba…9fad`, InlB `2b2d98f9…ac6f`); implementation hash unchanged during the run.

Aggregation (12 checks each, all PASS; `checks.json` in each folder): one cell and one condition; shape; window identities;
finite; pool mode unrestricted; parameter order; SGM scale equals the prior widths and the manifest's scale; quantile levels;
equal window weights (10 x 10,000 = 100,000 vectors, 0.1 each); pooled quantiles identical after a row permutation; the
representative is a complete saved vector (flat row = cloud[window, draw]); the opening-window SGM is a cloud row.

Methods recorded: primary metric = log10 estimator coordinates divided by the prior widths; representative = the repository
collection kernel's `weiszfeld_snap` member (nearest saved draw to the continuous Weiszfeld geometric median; an approximation
to the exact sample medoid at 100,000 members); pooled quantiles = mixture quantiles of the full pooled log10 sample, endpoints
transformed to physical units; the physical-coordinate normalization recorded separately in `result.json` as the sensitivity
check (`alternate_metric`), not adopted.

## 4. Runtime and storage (PC, Quadro T2000 4 GB, torch 2.7.1+cu118)

| cell | wall (companion) | sampling | peak GPU | `/usr/bin/time` wall, max RSS | collection folder | `.npz` |
|---|---|---|---|---|---|---|
| Fab 4 | 44 s | 24 s (3.2 s first window, 2.2–2.3 s after) | 914 MiB | 46.8 s, 5.2 GiB | 4.0 MiB | 2.1 MiB |
| InlB 42 | 44 s | 24 s | 914 MiB | 47.4 s, 5.2 GiB | 4.2 MiB | 2.1 MiB |

Aggregation helper: about 10 s per cell on the CPU. Evidence copies here: 4.4 MiB. Each aggregation folder: about 0.3 MiB.

## 5. Fab cell 4 (reference SEL 1.734 / 0.177 / 105 / 0.5848 / 0.05 / 8)

Per-window SGMs (chunks 0..9): mu_r 2.00, 1.93, 1.84, 1.82, 1.80, 1.79, 1.81, 1.83, 1.79, 1.77 px; sigma_r 0.309, 0.213, 0.191,
0.159, 0.149, 0.152, 0.151, 0.180, 0.171, 0.149; mu_pc 101–109 photons per dye per frame; sigma_pc 0.531–0.602; bleaching
0.077–0.131; lambda 7.3–8.7 per second.

- The opening window is a strong PSF transient: the width SGM sits at the 2.0 px prior edge (67 % of the window's draws above it
  on mu_r) and sigma_r at 0.31; both fall over the first three windows and then hold at 1.77–1.84 px and 0.15–0.19. The
  settled width lies ABOVE SEL (1.734), whereas Fab 26 settled BELOW it (1.63–1.70 px).
- Brightness, spread and flicker rate stay near SEL in every window: the pooled representative is 101.7 / 0.561 / 7.81 (−3 %,
  −4 %, −2 %), the pooled medians 104.0 / 0.561 / 8.08. Brightness closeness is partly by construction (the cell was chosen for it).
- Bleaching: pooled representative 0.094, pooled 90 % interval 0.015–0.323, broad and comparable with the prior range
  (0.010–0.316; per-window 90 % widths are 0.84–0.90 of the prior width in log10); weakly identified within a window, as before;
  0.05 remains the separately documented reference.
- SEL inside every marginal 90 % interval in 7 of 10 windows (excluded in chunks 0–2 by mu_r and sigma_r); the cell's pooled
  vector in 8 of 10. Distance rank of SEL relative to the window SGM 0.15 (chunk 0) to 0.84 (chunks 3–4); the source MAP ranks
  0.82–0.94 in every window. The ten windows' 90 % intervals share a non-empty intersection for four parameters (none for
  mu_r and sigma_r, because of the opening transient).
- Within-window correlations: sigma_pc–lambda_rate −0.78 to −0.92 (Spearman −0.75 to −0.89) in every window; mu_pc–sigma_pc
  −0.11 to −0.46; every other pair within ±0.25.
- Pooled representative: chunk 3, draw 7806: 1.8205 / 0.1746 / 101.74 / 0.5614 / 0.0939 / 7.81.
- Fresh-versus-source medians differ by at most 0.01 dex (bleaching) and 0.00 dex elsewhere: Monte Carlo reproducibility.

## 6. InlB cell 42 (reference SEL 2.256 / 0.4029 / 152.9 / 0.8588 / 0.05 / 2)

Per-window SGMs (chunks 0..9): mu_pc 150, 157, 182, 231, 246, 190, 192, 212, 176, 162 photons per dye per frame; lambda 4.10,
4.48, 4.12, 3.57, 3.21, 3.58, 3.09, 2.56, 3.29, 3.19 per second; sigma_pc 0.98 falling to 0.89–0.92; mu_r 2.27 falling to
2.15 px; sigma_r 0.43–0.46; bleaching 0.091–0.106.

- Brightness is bright in the MIDDLE of the recording (231–246 at 6–10 s) and near SEL at both ends, the mirror of InlB 11's
  bright opening band (284 at 0–2 s, then 81–204). Within-window 90 % widths are 84–106 photons, so the movement across windows
  (factor 1.6) is resolved; SEL 152.9 lies inside the brightness 90 % interval in 7 of 10 windows.
- Flicker rate sits at 2.6–4.5 per second in every window, above SEL's 2 (inside its 90 % interval in 4 windows only); InlB 11
  sat at 1.4–2.0. Spread sigma_pc 0.89–0.98 (pooled 0.92) is above SEL; InlB 11 ran 0.63–0.97 (pooled 0.79). Between these two
  cells, higher spread comes with a higher flicker rate; inside any one window the two remain strongly anti-correlated
  (−0.93 to −0.98), so the between-cell difference is not a move along the within-window ridge.
- PSF: 99,983 of the 100,000 width draws lie above the exact prior ceiling (10^0.3 = 1.995 px; 99,976 above 2.0 px), the
  documented InlB prior limitation; width declines steadily 2.27 → 2.15 px; sigma_r 0.43–0.46 is above SEL (pooled +12 %).
- SEL inside every marginal 90 % interval in 1 of 10 windows (chunk 6); the cell's pooled vector in 10 of 10. SEL distance rank
  0.14–0.56; the source MAP ranks 0.88–0.99. The ten windows' 90 % intervals share a non-empty intersection for all six
  parameters; for brightness it is 183–195 photons and contains the pooled representative's 186 (SEL's 152.9 lies outside
  it, as does SEL's flicker rate 2 against 3.2–6.5). This does not establish a common joint configuration, but it leaves a
  cell-level representative as a reasonable option for this recording.
- Correlations as in InlB 11: sigma_pc–lambda_rate −0.93 to −0.98, mu_pc–sigma_pc −0.42 to −0.68, mu_pc–lambda_rate +0.34 to
  +0.61, mu_pc–bleaching −0.27 to −0.38.
- Pooled representative: chunk 8, draw 1582: 2.2219 / 0.4502 / 185.9 / 0.9215 / 0.0994 / 3.577. Pooled 90 % intervals:
  mu_pc 129–259, sigma_pc 0.70–1.10, lambda 1.94–7.12, bleaching 0.016–0.313 (broad, comparable with the prior range 0.010–0.316).
- Fresh-versus-source medians differ by at most 0.02 dex (bleaching) and 0.01 dex elsewhere.

## 7. Four-cell comparison (pooled representative versus SEL; full tables in `FOUR_CELL_AGGREGATION_COMPARISON_2026-11-03/README.md`)

| parameter | Fab 26 | Fab 4 | InlB 11 | InlB 42 |
|---|---:|---:|---:|---:|
| mu_r | 1.649 (−4.9 %) | 1.821 (+5.0 %) | 2.268 (+0.5 %) | 2.222 (−1.5 %) |
| sigma_r | 0.153 (−13.6 %) | 0.175 (−1.4 %) | 0.390 (−3.3 %) | 0.450 (+11.8 %) |
| mu_pc | 104.9 (−0.1 %) | 101.7 (−3.1 %) | 140.7 (−8.0 %) | 185.9 (+21.6 %) |
| sigma_pc | 0.594 (+1.6 %) | 0.561 (−4.0 %) | 0.789 (−8.2 %) | 0.922 (+7.3 %) |
| prob_photo_bleach | 0.088 (+75 %) | 0.094 (+88 %) | 0.095 (+91 %) | 0.099 (+99 %) |
| lambda_rate | 8.01 (+0.2 %) | 7.81 (−2.4 %) | 1.65 (−17 %) | 3.58 (+79 %) |
| SEL inside every marginal 90 % (windows of 10) | 8 | 7 | 5 | 1 |
| the cell's pooled vector inside every marginal 90 % (windows of 10) | 9 | 8 | 6 | 10 |
| parameters whose ten 90 % intervals share a non-empty intersection (of 6) | 6 | 4 | 5 | 6 |
| SEL distance rank, min–max | 0.55–0.92 | 0.15–0.84 | 0.21–0.79 | 0.14–0.56 |

The membership counts and the intersections (per parameter in `FOUR_CELL_AGGREGATION_COMPARISON_2026-11-03/README.md`) are
marginal-compatibility diagnostics, not joint credible-region tests and not an independent validation: the pooled vector is
selected from the same draws it is compared with.

What differs between these examples (no claim about every cell):

- **Fab.** The two cells have similar pooled brightness, spread and flicker-rate terms, each within 4 % of SEL; brightness
  closeness influenced the cell selection, so this similarity is not evidence that the terms are shared across Fab cells. Their
  PSF summaries differ and lie on either side of SEL: the settled width is 5 % below SEL in cell 26 and 5 % above in cell 4,
  and both cells open with a larger-width, larger-sigma_r transient (cell 4's at the prior edge).
- **InlB.** Brightness, spread and flicker rate differ between the two cells and lie on either side of SEL: pooled brightness
  141 versus 186 (SEL 152.9), spread 0.79 versus 0.92 (0.86), flicker 1.65 versus 3.58 per second (2). Each cell also varies
  within its recording (brightness by a factor 1.6 to 3.5 across windows, with a different time course in each cell). That
  within-recording variation does not by itself rule out one representative per cell: a representative summarizes the chosen
  distribution and need not coincide with every window's center. In cell 42 the ten windows' 90 % intervals overlap for all six
  parameters and the pooled vector lies inside every window's marginals; in cell 11 the brightness intervals share no common
  value (opening 284 versus later windows), so a cell-level representative there depends on which windows are chosen. The
  two cells' opening-window draws (284 versus 150 photons) differ as much as their pooled representatives.
- **Bleaching** pools to 0.088–0.099 in all four cells, with per-window 90 % intervals that are broad and comparable with the
  prior range (widths 0.77–0.91 of the prior width in log10): a stable pooled center of a weakly identified marginal. It gives no
  new empirical reason to replace 0.05, which stays the separate reference.
- **Reference distance.** SEL's distance rank relative to the window SGM is high in most Fab windows (0.55–0.92 in cell 26,
  0.59–0.84 after cell 4's opening transient) and lower in the InlB windows (at most 0.79 in cell 11, 0.56 in cell 42). The
  rank is a distance rank relative to each window's SGM; it does not establish where SEL sits in posterior density.
- **Window position acts differently per recording.** Fab 4 has a pronounced opening shift of the inferred PSF parameters;
  InlB 42 has its highest inferred brightness in the middle of the recording; InlB 11 its highest at the opening. Which
  acquisition period the nuisance distribution should represent is a definition to make explicitly, not an assumption that the
  opening window is preferable.

The selection rule chose these cells for brightness closeness to SEL, so the Fab brightness agreement is partly by construction;
InlB 42 was the closest available cell (existing median 192) and still pools 22 % above SEL.

## 8. Not claimed, not done, still pending

- The external catalog cross-check of the staged InlB recordings (cell 42 among them) against the JUWELS TSV remains pending and
  is not claimed here; the collection used the recording as staged in the PC Experiment tier.
- No aggregation rule or nuisance selection is adopted; neither coordinate convention is a policy; the pooled representative and
  the opening-window SGM are candidates named by window and draw, not selections. The evidence supports comparing one complete
  representative per cell over alternative sets of 2 s windows, once the acquisition period to represent is defined.
- No rendering, artifact, training, full-cohort sampling, or HPC submission; the stronger sigma_pc arms stay unrun.
- No commit, push or fleet sync. Uncommitted: the companion's `--collection-label` change and its note, this Evidence folder,
  and the earlier uncommitted files (the Review and Draws companions and notes, the pilot Evidence folder, the InlB declared
  RDS TOML, the predictive-video note edit).
