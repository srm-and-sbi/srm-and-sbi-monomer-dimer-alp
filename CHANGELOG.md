# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.12 - 2026-09-21

Adds three direct (non-neural) imaging estimators -- PSF width, fluorescence loss and flicker rate --
and the information budget that grades them against what the recordings can support. No canonical
stage, parameter role, or preprocessing step changes; the detector continues to infer all six imaging
parameters, and `DETECTOR_WORKFLOW.md` §9.4 remains a proposal that is not in force.

### Added

- `DETECTOR_WORKFLOW.md` §6 restructured into eleven subsections: the camera ranges (§6.3) and the provenance table
  (§6.4) leave §6.2; the calibration outcome becomes run identity and metric definitions (§6.8), the multiple-dye
  outcome (§6.9), the one-dye comparison with side-by-side tables (§6.10), and both estimators on the experimental
  recordings (§6.11); the superseded predecessor-repository comparison is removed. §7 rewritten as five subsections
  around the marginalized blocks and their media (§7.1, now holding the block table formerly in §9.3), the
  `Nuisance_DLI` artifact with its build status and the analyst-owned choice of minting estimator (§7.2), the
  construction step (§7.3), the persisted records (§7.4), and the implemented estimator artifact format (§7.5).
  Cross-references renumbered in the workflow doc and ten companion/project documents.
- One-dye comparison results recorded in `DETECTOR_WORKFLOW.md` §6.10: the three pre-specified
  comparisons all resolve in the one-dye direction (`sigma_r` correlation 0.98 vs 0.17, `lambda_rate`
  MAE 0.083 vs 0.201 dex, `mu_r` offset +0.005 vs +0.018 dex); joint coverage 0.89 at nominal 0.90
  vs 0.62; `prob_photo_bleach` unchanged (MAE 0.205 vs 0.206), the parameter with by far the largest
  information-budget benchmark relative to its prior at 2 s. On the MET-FAB recordings the estimators
  disagree on `lambda_rate` (2.30 vs 5.23), left open; the direct flicker estimator is a cross-check
  pending validation on multiple-dye synthetic recordings. §6.10 closes with a block stating what the
  comparison establishes and what it does not: the labeling law strongly affects inference
  performance and the multiple-dye estimator undercovers, but the comparison does not isolate an
  estimator-only failure, does not show bleaching unrecoverable at 2 s, does not make the one-dye
  estimator calibrated per parameter, and does not validate either model on the experimental
  recordings. The multiple-dye model stays the working baseline.
- `srm_and_sbi_monomer_dimer_alp/direct_imaging_estimates.py` — pure measurement kernels that read
  imaging parameters off the renderer's own forward model: the 8-bit-to-ADU domain conversion, the
  EMCCD mean and variance laws, matched-filter spot detection, the pixel-integrated Gaussian fit,
  greedy cross-frame linking, and the errors-in-variables population summary. No file access, no
  machine profile, no printing.
- `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_PSF_Width.py` and its
  companion note — a direct estimate of `mu_r` and `sigma_r`, with `--selftest` (nine in-memory
  scenes rendered at known widths through the production renderer), `--dry-run`, `--workers`, and
  acceptance thresholds fixed before the first run. Never wired into the stage dispatcher; outputs
  go to the Data_Bank `Posit` tier. A tier-size check names the case of a stale development tier
  carrying production filenames, which would otherwise score a handful of videos silently.
- `srm_and_sbi_monomer_dimer_alp/information_budget.py` and
  `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Information_Budget.py` with its
  companion note — Cramer-Rao lower bounds on each imaging parameter from one recording, so that a
  weak estimator can be told apart from uninformative data. Covers the per-spot width and amplitude
  information, the log-normal population bounds, the decay-rate bound and its cubic dependence on
  duration, and the Ornstein-Uhlenbeck correlation term that reduces the effective frame count.
- `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Fluorescence_Loss.py` and its
  companion note — a direct estimate of `prob_photo_bleach` from the decay of total fluorescence,
  with the same `--selftest`/`--dry-run`/`--workers` structure. Its acceptance is stated at the full
  recording length and applied only over the part of the prior where the budget's benchmark is itself
  inside the threshold; outside it the estimate is recorded without a verdict.
- `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Rate.py` and its
  companion note, plus the flicker kernels in `direct_imaging_estimates.py` — `lambda_rate` from the
  autocorrelation of per-spot ln-brightness, measured from VIDEO rather than from localization
  tables. The method deliberately mirrors `Flicker_Rate_Derivation` (per-trace log, per-trace linear
  detrend, gap-aware pooled autocorrelation, normalization at lag ONE, shape match against a
  matched-length Ornstein-Uhlenbeck arm) so that a video-derived value and the recorded 5.1/4.7
  table-derived values are directly comparable. That utility is left untouched so its result keeps
  its provenance; the kernels are reimplemented rather than imported.

  The model arm is SINGLE-DYE by choice. In that case the method is exactly free of `mu_pc` and
  `sigma_pc` -- the first shifts ln-brightness additively, the second scales it linearly, and both
  vanish under the detrend and the normalization. Summing several dyes before the logarithm breaks
  the exactness, so modeling the multiplicity would remove a small bias at the price of making the
  estimator depend on `sigma_pc`, a parameter that is itself under inference. In a campaign that
  exists to decide which parameters can be pinned INDEPENDENTLY that is circular, so the bias is
  left in and bounded instead: measured against the renderer it is 0.0077 in shape at
  `sigma_pc = 0.42` (about 0.009 dex) and at most 0.04 at the top corner (about 0.05 dex), against
  roughly 0.035 in shape for a ten percent change in `lambda_rate`. The report states that band
  across the whole `sigma_pc` prior without needing a `sigma_pc` value.

  Recording length is the binding constraint: the per-track detrend removes low-frequency power, so
  the total shape travel from `lambda_rate` 1 to 10 falls from 0.584 on MET-length tracks to 0.337
  on a 2 s tier, with the low-rate end hardest (the factor-of-two step from 1 to 2 separated by only
  0.042 in shape). The model arm is vectorized by trace length, costing about 1.5 s per grid point.

- `DETECTOR_WORKFLOW.md` §9.5 — the information budget as a diagnostic: the bound/direct/neural
  comparison, the population-spread crossover that linking moves, the duration, correlation and
  shallow-decay terms of the photobleaching budget with the duration table, and an explicit
  statement that the bounds are optimistic because they omit emitter turnover, dye-multiplicity
  changes, spot overlap and detection truncation.

### Fixed after adversarial review

An adversarial review of the new code (four independent lenses over the kernels, the scripts and
their companion notes) raised 29 findings. Those that changed a number:

- `psf_width_population` trimmed 2% of every sample and then used the trimmed variance as the
  population variance. A symmetric trim removes variance from a roughly normal sample by the
  truncated-normal factor -- 0.8735 at the 1%/99% default -- so `sigma_r` came back **6.5% short at
  every value tested**, against a pure lognormal draw with no measurement error. The trim is kept,
  because a few failed fits in a tail would otherwise dominate a variance, but it is now divided
  out analytically; the corrected estimator recovers 0.999-1.000 of truth.
- The photobleaching bound divided the per-spot brightness variance by the track length, copying
  the treatment that is correct for the PSF width. It is not correct for brightness: the width is
  drawn once per subunit and held, but brightness is a stationary Ornstein-Uhlenbeck process, so
  every frame is a fresh draw from the very population whose spread `sigma_pc` describes. Verified
  against the renderer, the within-track variance of ln brightness is 0.145 at `sigma_pc = 0.42`
  against a population variance of 0.176 -- most of the spread lives within a track, not between
  tracks. The bound now counts spot-frames discounted by the flicker correlation.
- The budget utility and the fluorescence-loss estimator independently quoted 0.018 and 0.028 for
  the same whole-field relative noise, which made the published duration table disagree with
  the tool that produced it. The constant now lives once, in the kernel, with its derivation.
- `Information_Budget` wrote to a path with no condition token, so a FAB run and an INLB run would
  have overwritten one another; the emitter density and dyes per spot differ between them.

Claims corrected against the code that produces them: the effective sample size at the prior center
is 3.2 frames per hundred, not seven, and the resulting inflation of a decay-rate standard deviation
is a constant 5.6x independent of recording length; the flicker is NOT the largest term in the
bleaching budget once the amplitude and offset are profiled out -- the shallow-decay degeneracy costs
2.7x at the top of the prior and 150x at the bottom, so it dominates everywhere but the top; the
background is about thirty times the emitter signal over the field at MET-FAB density, not a hundred;
the SCOPE camera boxes are +/-1.15% about their centers for three of the five, not "better than 1%";
and `sigma_r`'s linear-unit threshold was being compared against a dex-unit prior width.

Also fixed: a second check that could never fail (`check_shape` comparing two arrays built in the
same loop); `make_figure` raising on an empty reduction and discarding the report that would have
explained why every estimate was NaN; both estimators exiting 0 on a failed acceptance gate;
`--condition` choices hardcoded rather than taken from `labeling.LABELING_CONDITIONS`;
`Direct_Fluorescence_Loss` hard-requiring a camera record its default observable never reads;
`SELFTEST` occupying the alias slot reserved for the condition; and `--workers` advertised for a
selftest that renders serially.

One finding was refuted rather than fixed: the flicker correction applied to the whole-field curve
is correct, because a sum of independent Ornstein-Uhlenbeck processes keeps the same normalized
autocorrelation (measured lag-1 of the summed field 0.922-0.935 against a per-dye 0.9387). Summing
reduces the fluctuation amplitude, not its correlation time.

### Result of record

The budget's sharpest output concerns `prob_photo_bleach`. With the amplitude and offset of the
decay curve profiled out -- they are fitted, not known, and where the decay is shallow the
exponential is nearly a straight line, so only their product with the rate is determined -- the
benchmark standard deviation of an unbiased estimate **exceeds the 1.5 dex prior width everywhere at
2 s**, and falls inside the 0.10 dex threshold only in the upper half of the prior at 20 s. This is an
approximate precision benchmark for an unbiased decay-rate estimator using total fluorescence, not a
fundamental recovery limit for inference from the full video (the one-dye estimator tracks the
parameter with correlation 0.80 at 2 s); it supports constraining bleaching preferentially from longer
recordings, subject to validating that measurement. A benchmark computed with the amplitude and
offset treated as known is optimistic by an order of magnitude and would have missed this entirely.

### Notes on the measurements behind the estimator design

Three choices were adopted because the alternative was measured against the renderer and found
biased: the local background is supplied from the SCOPE block rather than fitted (a free background
is near-degenerate with a broad faint Gaussian and pulled the fitted width down by 6% at the bottom
of the `mu_r` box and 25% at the top); the fit is flat-weighted with the EMCCD variance law entering
only the standard error through a sandwich covariance (model-variance weighting down-weights the
bright core and biased `mu_r` high by 0.06 dex); and neighboring detections are masked out of each
patch (their flux inflated `mu_r` by 0.018 dex at realistic emitter density). Linking before
summarizing moves `sigma_r` out of the regime where the measurement variance dominates its bound.

For the fluorescence-loss estimator the corresponding correction was the observable itself. Apertures
pinned to the spots of the opening frames -- the natural first design -- measure diffusion rather than
bleaching: emitters wander roughly thirteen pixels over a 20 s recording and leave a four-pixel
aperture, and on rendered recordings that returned a bleaching probability near 0.5 for every true
value from 0.01 to 0.316. The whole-field sum is immune to motion within the frame by construction and
is now the default, with the aperture path retained behind a flag for diagnosis.

## 0.1.11 - 2026-09-18

Documentation and release bookkeeping only: no parameter role, preprocessing step, or executable
model changes. Version 0.1.10 is the `one-dye-sensitivity` branch and is not part of `main`.

### Fixed

- Corrected descriptions of existing calculations, without new claims. The camera block is described
  as externally constrained rather than jointly inferred in this workflow, with the structural
  degeneracies (`gamma = g/C`; the products `gamma·kappa_q` and `gamma·kappa_q·kappa_o`) kept
  separate from the empirical recovery failure observed when the predecessor detector inferred the
  camera jointly; controlled-illumination photon transfer constrains `gamma` on its own
  (`DETECTOR_WORKFLOW.md` §5, §6.2, §9.3; `REFERENCE_EMCCD_NOISE_MODEL.md` §6, §9;
  `PROJECT_CONTEXT.md` §2; `detector_parameterization.py` comments). The renderer is described as
  sampling positions and brightness at the frame interval without integrating motion during exposure;
  the experimental exposure duration is separate acquisition metadata. The flicker-rate derivation is
  described as a single-emitter log-brightness match that corrects finite-track detrending only and
  does not model multi-dye intensity sums, so its result is a single-dye-equivalent reference
  (`DETECTOR_WORKFLOW.md` §6.5; the derivation's companion note). The within-recording fall of the
  inferred bleaching probability is described as an observation that motivates investigation, not as
  evidence that a single-rate bleaching model is misspecified (`DETECTOR_WORKFLOW.md` §6.2).
- Provenance of every externally supplied value is recorded as two separate properties, evidence
  (acquisition setting, measured quantity, datasheet value, convention, assumption) and source
  (ThunderSTORM protocol or output column, publication, code definition), in a new table in
  `DETECTOR_WORKFLOW.md` §6.2 and in the camera table of `REFERENCE_EMCCD_NOISE_MODEL.md` §6. The
  table names two conventions the code fixed silently: no exposure integration, and the fixed global
  16-bit to 8-bit video map.

### Added

- `DETECTOR_WORKFLOW.md` §6.9, the calibration outcome of the MET-FAB detector under the Poisson
  labeling law: run identity, metric definitions, per-parameter and joint results from the Evaluation
  and Posterior_Calibration reports, the separate calculations on the saved draws (including the
  stratification by realized dye multiplicity), the limits of what the results support, and the
  status and pre-specified comparisons of the one-dye sensitivity run. Cross-referenced from
  `PROJECT_CONTEXT.md` §7.

### Changed

- `DETECTOR_WORKFLOW.md` §9.4 records a proposal, explicitly not implemented: the acquisition-
  information contract (what the pipeline needs from outside and from where), a reduced inferred block
  of `mu_pc`, `sigma_pc`, and provisionally `lambda_rate`, the treatment of quantities that leave the
  block as nuisances with explicit uncertainty, the adoption gates (external inputs, replacement
  measurements, reduced neural estimator, experimental adequacy), the validation-data tiers, and the decision
  statement. The implemented six-parameter detector remains in force. `PROJECT_CONTEXT.md` §8 gains
  open question S6 pointing to it. Terminology fixed throughout: the amortized flow is the neural
  posterior estimator and the replacement methods are direct estimators; both are validated on
  simulated data, so neither is opposed to simulation-based inference.

## 0.1.9 - 2026-09-17

### Fixed

- The sharded GPU stages (Evaluation, Experiment, the `Posterior_Calibration` and DETECTOR
  `Nuisance_DLI` wrappers, both workflows) launch as plain Slurm tasks -- one per GPU on
  every allocated node, `srun --ntasks-per-node=$GPUS` -- instead of through `torchrun`.
  torchrun's elastic agent enforces a 300 s exit barrier that no launcher setting changes
  (torch 2.9 never reads `TORCHELASTIC_EXIT_BARRIER_TIMEOUT`, which the wrappers exported as
  a safety net); when ranks finished more than five minutes apart, the first node's agent
  tore down the rendezvous, the other nodes' agents died with a connection error and killed
  any rank still working, and its shard was lost (Evaluation of the FAB detector estimator
  on JUPITER: 16 ranks spread over 18 minutes, 15 shards saved, no report). The runner's
  `resolve_topology()` already reads `SLURM_NTASKS / SLURM_PROCID / SLURM_LOCALID`, so the
  sharding is unchanged; there is simply no rendezvous and no barrier any more. Inference
  keeps `torchrun` (DistributedDataParallel needs it, and its ranks finish together).
- `resolve_topology()` binds the GPU robustly under per-task GPU binding: when Slurm hides
  the other GPUs from a task, the local rank is folded onto the visible devices.
- Every `--merge` refuses an incomplete shard set (`experiment_support.assert_complete_shard_set()`),
  naming the missing ranks, instead of silently concatenating whatever exists and deleting
  the shards; `--allow-partial` merges what is present and records the fraction in the report
  (`shards_merged`).

## 0.1.8 - 2026-09-17

### Added

- Evaluation and Experiment report three point estimates of every posterior side by side:
  the MAP (the optimizer's mode, as before), the 1-D posterior median (Q50 of each
  marginal) and the sample geometric median (SGM: the posterior sample closest, in
  prior-width-scaled log10 distance, to all other samples -- a joint point estimate that
  is itself a probable point; `evaluation.sample_geometric_median()`, `prior_scale()`).
  `posterior_summary()` returns the SGM of the same draws on request (`return_sgm`), both
  runners store it as `posterior_sgm` beside `posterior_quantiles` (shards and merged
  arrays), Evaluation repeats the recovery table for the median and the SGM, Experiment
  repeats the per-condition table for both, and a new point-estimate agreement table
  (`point_estimate_agreement_table()`) gives the median |MAP - median|, |MAP - SGM| and
  |SGM - median| gaps plus the share of observations whose MAP lies outside the
  posterior's central 90% interval. Motivation: on the MET-FAB recordings the
  unrestricted-pool MAP of the detector estimator landed in flow density spikes outside
  the prior box (mu_r bands 0.125 dex above and below the mode, with a higher
  log-density than the mode) while the posterior medians stayed put; the medians beside
  the MAP make that diagnosis immediate instead of a side computation.

## 0.1.7 - 2026-09-16

### Fixed

- Non-array HPC stage logs are named by the job id (`--output="$MON_OUT/%x_%j.out"`) in both
  dispatchers, in the stage scripts' baked fallbacks and in the runbook. The previous `%x_%A.out`
  used the array master id, which JUPITER's Slurm resolves to 0 for a non-array job, so every
  Inference, Evaluation or Experiment submission with the same job name (a resurrect follow-up,
  for instance) would have shared and truncated one log file. Array stages keep
  `%x_%A_Node_%a.out`.

## 0.1.6 - 2026-09-16

HPC housekeeping for the FAB production campaign on JUWELS and JUPITER.

### Added

- **`Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Bulk_Delete.sh`** and the runbook recipe
  "Freeing a scratch tier under an inode quota" (HPC README §6): parallel deletion of a data
  tier's contents (16 `rm -rf` streams over the top-level entries, `READY_TRACT` expanded one
  level), dry-run by default, refusing paths outside a `Data_Bank` tree or inside the legacy
  read-only trees, keeping the tier directory the machine profiles require. Measured on a JUWELS
  login node: ~400 K files/min against ~30 K files/min for a single `find -delete`; the retired
  dimer-alp and dimer-bet TRAIN/TEST scratch tiers (3.75 M files) went in 11 minutes.

### Changed

- Generation controller: the 2 s arrays run with `--time=24:00:00` (the JUWELS `batch` maximum),
  matching the 5 s arrays and the detector controller; they carried 18 h.
- `hpc_local.env.example` documents the `TIME` override; JUPITER's machine env pins
  `export TIME=12:00:00`, the only wall time its booster partition allows.

## 0.1.5 - 2026-09-16

Report fixes found while reading the smoke campaign's reports on rcl01, and two report columns
that make an estimator's failure modes visible without opening the arrays.

### Added

- **MAP recovery report: `outside prior` and `corr(inf, true)` columns** (`evaluation.recovery_table`,
  `fraction_outside_prior`, `correlation_with_truth`). The first is the share of MAP estimates that
  left the row's prior box, which only `--pool-mode unrestricted` permits; the second is the
  correlation between inferred and true values, `n/a` below three pairs or at zero spread. On the
  smoke tiers the biology FAB estimator placed every N_R estimate above the box ceiling and returned
  a near-constant vector for the other ten rows; neither was visible in the error columns.
- **Experiment report: `outside prior` column** (`evaluation.experiment_table`), per parameter and
  condition.
- **Nuisance_DLI report: a checks table** (artifact written, parameter keys in the detector table's
  order, draw shape, finite draws, and, for the bounded and box constructions, every draw inside the
  imaging prior box; under `pool_mode = "unrestricted"` the outside share is reported as a statistic).
  The report carried an empty checks table that could not fail.

### Changed

- **The experimental set is the whole accession**: sixty 20 s recordings per condition (BioStudies
  S-BSST712) staged as `Experiment_{FAB,INLB}_Cell_{0..59}_20S_RAW.tif` in the archive's
  coverslip-and-cell order, with `Catalog_Note_Experiment.tsv` (index, kind, working filename, source
  archive, source member) beside them. The earlier set of 25 curated recordings per condition, a
  computational-capacity cap, and its spreadsheet note are retired. No code change: cell discovery
  globs the directory, `--max-cells` defaults to all, and the HPC Experiment scripts default
  `MAX_CELLS=0`. The Experiment stage now estimates 600 windows per condition.

### Fixed

- Prior-realization audit P5: the share of deposited recordings inside the simulated range printed
  `None` wherever the Special_Analyses tree is absent (every rcl01 report). The 60 + 60 first-2 s
  per-recording spot counts of A9 are embedded in the script (`A9_FIRST2S_SPOTS`), so the share is
  computed on every machine; the CSV, when present, still takes precedence.
- Structure audit R4 drew dye counts for every subunit and skipped the occupancy step the DLI stage
  applies, so its dye totals (1688 Fab, 486 InlB for 1000 subunits) were not the pipeline's labeled
  counts. R4 now labels through `occupancy_per_subunit` at the condition's declared occupancy, as R3
  and the DLI stage do, and reports labeled subunits beside the expectation and the dye total.
- VALIDATION §2.4: the resurrect run trains the requested `--epochs` more (five in the recipe, numbered
  globally 6 to 10), not "one more epoch".

## 0.1.4 - 2026-09-14

The decided prior ranges and the declared visibility inputs: every learnable row leaves its
development range for a decided box with a named source, the initial composition becomes the
log dimer-to-monomer ratio, and probe occupancy becomes a declared per-condition input.

### Added

- **The Theta_Set schema** (`io.theta_set_schema` / `write_theta_set` / `load_theta_set` /
  `theta_set_status`, `ThetaSetSchemaError`). Every `Theta_Set` now carries, beside the numbers, its
  ordered parameter keys, prior bounds and per-row scales, condition, timing label, generating stage,
  and package version (`.zarr` attributes; a JSON sidecar beside a plain `.npy`). The RDS stage writes
  it for the eleven-row tier; the DLI stage writes the six-row imaging schema on the detector's
  `Theta_Set` (and on the biology `Nuisance_DLI` record, provenance only). Every reader -- the
  training dataset, the DLI stage's read of the RDS labels, evaluation, calibration, the
  embedding-distance analysis, the prior-realization audit (new P0) -- refuses a `Theta_Set` whose
  schema is absent or differs in keys, bounds, scales, or condition, naming the difference; the
  dry runs print the schema status per file. Structure-audit D10 exercises the round trip and the
  refusals. Motivation: two tables with the same row count are indistinguishable by shape, so a tier
  generated under an earlier table could otherwise be consumed silently.
- **Decided prior ranges for all eleven learnable rows** (`parameterization.py`,
  `_PARAMETERIZATION_RAW_NESTED`; rationale in `PROJECT_CONTEXT.md` sec. 2, "How the prior ranges
  and the declared inputs are set"), log10 estimator coordinate: `count_total` [2.5, 3.5]
  (316-3162 subunits; from the first-2 s spot counts of the 120 deposited recordings divided by
  the declared visibility per subunit, Special_Analyses A9: log10 N_R peaked at 3.0-3.1, sd 0.21
  InlB / 0.30 Fab, 94% inside the box; localizations undercount the visible receptors, which is the
  reason for not extending the box downward; the box is at least as wide as the empirical shape
  on purpose), `ratio_dimer_monomer_initial`
  [-2, 2], `rate_dissociation` [-3, 1] (the lower bound lets MET-FAB dimers persist through a
  20 s recording: 0.001 per s loses 2% in 20 s), `diffusivity_alp` [-1.25, -0.25],
  `relative_diffusivity_dimer` [-1, 0], `relative_diffusivity_slow` [-1, 0],
  `relative_diffusivity_immobile` [-3, -2], and the four switching rates [-1, 1] each. Every
  row's `DOC` names its source (the dimer-alp 0.4.23 baseline, Special_Analyses A9, the model
  specification's sec. 9 anchors, the tracking pipelines' thresholds); both conditions share the
  one table, and no range encodes a condition's expected answer.
- **Probe occupancy as a declared per-condition input** (`ConditionSetting.occupancy` /
  `visibility_ratio` / `visibility_ratio_to`; `SimulationRDS.occupancy_of`, `visibility_of`,
  `occupancy_source_of` and the module-level `occupancy_of`, `visibility_of`): MET-INLB 0.5
  declared (the collaborators' statement; the published uPAINT protocol reports 0.25 nM in the
  imaging medium, unreconciled; questions sent 2026-09-11; provisional). MET-FAB DERIVED from a
  declared Fab/InlB VISIBILITY RATIO of 0.5 (a rounded convention over the measured Fab/InlB
  spot-density ratios 0.38-0.48 of the deposited recordings, Special_Analyses A9) and the INLB
  anchor: p_FAB = 0.5 x (0.5 x 0.5) / 0.806 = 0.155 (0.806 = 1 - exp(-1.64), the Poisson
  probability a Fab carries at least one dye). Visibility per subunit INLB 0.25, FAB 0.125;
  both-labeled share among visible dimers 14% and 6.7%. Exactly one of `occupancy` and
  `visibility_ratio` per condition is enforced at import; derivations are one step deep.
- **`Labeling_Set` columns `occupancy_monomer` and `occupancy_dimer`** (`labeling.LABELING_SET_COLUMNS`,
  ten columns now): the occupancy actually applied to each species -- declared, derived, or override.
- **Structure audit D9** (`..._Model_Structure_Audit.py`): the table equals the decided ranges,
  immobility by construction (D_i below the pipelines' thresholds 0.0028 / 0.0065 um^2/s), the
  dissociation floor, and the symmetric composition box; D8 extended with the declared
  occupancies; D3/D4 rewritten for the ratio row; R3 at the declared occupancies with the both-labeled
  share; R6 timing at the count ceiling. Deterministic tier PASSED 9/9 on 2026-09-14 (profile
  `mars_pc`).
- **Prior-realization audit** (`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py`
  with its companion `.md`): read-only checks of GENERATED products -- the `Theta_Set` against the
  prior box (P1) and the composition rule (P2), trajectories against the realized composition and
  the stationary mode law at frame 0 (P3, `--trajectories K`), the `Labeling_Set` against the
  declared occupancies (P4: per-subunit visibility, visible fractions, both-labeled share, emitters per
  subunit), a descriptive comparison of the simulated visible counts against the deposited spot
  counts (P5, Special_Analyses A9), and the theoretical visibility chain corroborated through the
  DLI stage's own labeling functions on a synthetic lineage for both conditions, with the
  FAB/INLB visibility ratio against the declared one (P6; `--visibility` runs it alone). The report
  ends with theory, code path, and products side by side. `--selftest` PASSED for both conditions
  on 2026-09-14. Listed in `PROJECT_CONTEXT.md` sec. 2 and 5, `README.md`, and `VALIDATION.md`
  (run after any tier or DLI pass).

### Changed

- **The initial composition row**: `ratio_dimer_monomer_initial` (r = n_B / n_A, log10 on [-2, 2],
  label r_{B/A}; complex fraction f_B = r / (1 + r) from 1% to 99%, symmetric about an even split)
  replaces the linear `fraction_dimer_initial` (x_B on [0, 1]). `realize_initial_composition(N_R, r)`:
  `n_B = min(round(N_R r / (1 + 2 r)), floor(N_R / 2))`, `n_A = N_R - 2 n_B`; x_B = 2 r / (1 + 2 r)
  and f_B are DERIVED. The decided table has no linear row; the per-row `LOG_FLAG` rule stays the
  contract. The sample-geometric-median plane is the log ratio r against `kappa_OFF`; the
  population-composition kernel forms x_B from (N_R, r) inside each draw.
- **The immobile mode is immobile by construction**: R_i in [0.001, 0.01] keeps D_i = R_i D_A below
  the tracking pipelines' immobility thresholds over nearly the whole D_A range; the lower half is
  below the 2 s resolution floor (~0.0005-0.001 um^2/s), so the posterior of R_i is flat there
  (accepted). The baseline [-2, -1] was rejected: at 0.1 the mode is classified confined and
  touches R_s.
- **The receptor count is conditional on the declared occupancies**: the data constrain the visible
  subset; an absolute count inherits the occupancy convention, within-visible ratios and
  compositions do not. A revised occupancy anchor shifts the count range by a constant in log10.
- **`--occupancy` on the DLI stage** (`simulation_dli_runner`) is an explicit OVERRIDE for
  sensitivity runs, recorded as `override`; by default the stage reads the condition's declared or
  derived value and prints it with its source. Full occupancy is retired as a baseline (uPAINT
  labels a sparse subset by design).
- **Documentation**: `PROJECT_CONTEXT.md` sec. 2 carries the decided table and the new subsection
  "How the prior ranges and the declared inputs are set" (rules, the eleven rows with sources, the
  declared inputs, the structural statements, what would change them), the neighbor-list skin
  statement names what the search guarantees at sub-step boundaries, and sec. 3-5 follow (the
  declared occupancy at the DLI stage, the ten `Labeling_Set` columns, the prior-realization audit
  in the script map); `DETECTOR_WORKFLOW.md` sec. 6.1 carries the decided ranges with a pointer to
  the rationale, sec. 6.4 and 6.5 the declared occupancies, and sec. 8 states that the detector
  infers no biological rate because the reaction-diffusion side is marginalized from the
  condition's tier; `README.md`, `VALIDATION.md` (the occupancy default and its override, the
  prior-realization audit recipe), and the analysis companion notes follow, with a dated note in
  the companions whose text involved the linear fraction or the occupancy default. Wording
  reviewed the same day: the share `a / (2 - a)` is named the share of visible dimers with BOTH
  SUBUNITS LABELED (it equals a two-dye share only under the one-dye-per-ligand InlB law); the
  count, immobile-mode, and switching-rate rationales state expectations, not demonstrated
  posterior behavior; the visible composition is stated to depend on the visibility `a`; the
  Special_Analyses A9 all-dimer receptor bound gained its missing factor of two.

### Removed

- The learnable row `fraction_dimer_initial`. Estimator artifacts and resurrect states of the
  earlier table are rejected by the schema guard (`parameter_keys`); `Theta_Set` files of the
  earlier table carry no schema and are refused by every reader (see the Theta_Set schema under
  Added). A fresh per-condition tier under the decided ranges is required, and none exists yet.
- The full-occupancy default of the DLI stage.

## 0.1.3 - 2026-09-10

The per-condition association setting: the association ratio leaves the learnable table and
becomes a declared constant of each experimental condition, and the RDS trajectory tier follows
the condition.

### Added

- **`ConditionSetting` and `CONDITION_SETTINGS`** (`parameterization.py`): the per-condition
  setting of the reaction-diffusion model, carried on `SimulationRDS.conditions` and read through
  `SimulationRDS.association_ratio_of` (module-level `association_ratio_of(condition)`). MET-FAB
  `R_ON = 0` -- no association channel at all, a structural setting (a ratio between 0 and 1e-3 is
  refused at import as a disguised zero); pre-existing dimers may still dissociate, and the readout
  is dimers PRESENT at the recording start, not dimers formed. MET-INLB `R_ON = 1` --
  `lambda_on = lambda_ref = 6 D_A / r^2` for every association channel, a diffusion-scaled
  reference convention, not a measured association rate or a verified diffusion-limited regime;
  composition and unbinding estimates are conditional on it. Import-time validation keeps the
  condition tokens equal to the labeling and experiment registries and requires association to be
  switched on in at least one condition.
- **`--condition` on the RDS stage** (`simulation_rds_runner.build_rds_parser`, the Prime
  `..._Simulation_RDS.py`): required, `FAB` or `INLB`; it selects the reaction network and the
  tier the run writes.
- **Structure audit D8 and per-condition checks** (`..._Model_Structure_Audit.py`): D8 checks the
  condition registry (the tokens, the exact 0.0 / 1.0 ratios, the retired row absent from both
  tables, the disguised-zero refusal, `Paths.rds_alias` carrying the condition token and refusing
  to resolve without one); D1 and D4 run per condition (17 channels under MET-INLB, 11 under
  MET-FAB; every MET-INLB fusion at exactly `lambda_ref`, no fusion under MET-FAB); the run tier
  simulates each condition from its own network and renders each condition from its own
  trajectory. Deterministic tier PASSED 8/8 on 2026-09-10 (profile `mars_pc`); the run tier awaits
  approval.

### Changed

- **Eleven learnable parameters, identical for both conditions**: `count_total`,
  `fraction_dimer_initial`, `rate_dissociation`, `diffusivity_alp`, `relative_diffusivity_dimer`,
  `relative_diffusivity_slow`, `relative_diffusivity_immobile`, `rate_fast_slow`,
  `rate_slow_fast`, `rate_slow_immobile`, `rate_immobile_slow` -- one table, one estimator layout
  (`event_shape == (11,)`). Unbinding stays inferred in both conditions. ALL ranges remain
  DEVELOPMENT SETTINGS; the training priors -- including the MET-FAB unbinding lower bound needed
  for dimers that persist over a 20 s recording -- are a later, dedicated decision.
- **The reaction network is generated per condition**: `reaction_channels(theta, condition)` and
  `build_system(theta, condition, ...)` -- 17 channels under MET-INLB (6 fusions, 3 fissions,
  8 conversions), 11 under MET-FAB (3 fissions, 8 conversions). Fission placement (2 x the reaction
  distance) and the 2 ms sub-step are unchanged.
- **One RDS trajectory tier per condition, shared by both workflows.** `Paths.rds_alias` is the
  sibling alias plus the condition token (`SRM_AND_SBI_MONOMER_DIMER_ALP_FAB`), never the workflow
  qualifier, and refuses to resolve without a condition; the trajectories and the eleven-parameter
  `Theta_Set` carry `SRM_AND_SBI_MONOMER_DIMER_ALP_<CONDITION>_<timing>_...`. Every stage takes
  `--condition`. Held-out recordings are matched across conditions by parameter draw, not by
  trajectory.
- **`Generate_Datasets.py` runs the RDS stage once per condition per split**, then the DLI passes
  per workflow over each condition's tier; the refusal to regenerate an existing tier and the
  `--reuse-rds` presence check are per condition.
- **HPC: `CONDITION` is required for RDS-only jobs too** (`SIM_STAGE=rds`), and the job name always
  carries the condition slot: `SRM_AND_SBI_MONOMER_DIMER_ALP_<CONDITION>_<timing>_Simulation_<SPLIT>`.
- **The detector's RDS nuisance** (`detector_parameterization.DETECTOR_NUISANCE`) has eleven rows
  (nuisance-from-object, supplied by the condition's tier); the association ratio is a constant of
  the generator, not a row.
- **The sample-geometric-median plane** (`sample_geometric_median_runner`) shows the initial dimer
  fraction `x_B` against the dissociation rate `kappa_OFF`.
- **The horizon audit** reports coverage and the exploratory table without the former
  association-ratio exclusions, and its generate phase simulates the cohort under the run's
  condition.
- **The structure audit's stationarity check (R2)** runs the isolated switching chain under the
  MET-FAB configuration (no association channel by construction) instead of at a tiny ratio.
- **Documentation**: `PROJECT_CONTEXT.md` sec. 2 states the per-condition association setting and
  the eleven-row table; sec. 3-5 the per-condition tier and the naming grammar
  (`..._ALP_FAB_2S_50FPS_TASK_0_SIM_0_TRAIN.h5`, `..._ALP_FAB_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr`);
  `DETECTOR_WORKFLOW.md` sec. 4-7 and 9 follow; `CLAUDE.md`, `README.md`, `VALIDATION.md` (every
  RDS smoke command takes `--condition`), and the analysis companion notes follow, with a dated
  note in the companions whose earlier results or figures involved the association ratio.

### Removed

- The learnable row `relative_rate_dimerization` (`R_ON`) from the biology table, the
  corresponding nuisance row from the detector table, and
  `StoichiometryBlock.association_ratio_key`.
- Resolution of the 0.1.2 trajectory tiers under the bare sibling alias: `rds_alias` requires a
  condition, so those tiers are no longer resolvable by name, and a fresh per-condition tier must
  be generated before any DLI run.

## 0.1.2 - 2026-09-09

The separated stoichiometry-mobility model, its twelve learnable parameters, and the one
parameter-conversion rule.

### Added

- **Two declarative model blocks** on `PARAMETERS.simulation.rds` (`parameterization.py`):
  `StoichiometryBlock` (molecular species A monomer with one subunit and B dimer with two; the
  association and dissociation channels and their parameter keys) and `MobilityBlock` (the
  mobility modes f fast / s slow / i immobile, their diffusion-ratio keys, the sequential
  switching chain f <-> s <-> i with its four rate keys, the slower-parent inheritance rule).
  `SimulationRDS` derives the PARTICLE TYPES = species x mode (`A_f`, `A_s`, `A_i`, `B_f`,
  `B_s`, `B_i`), their subunit counts, and the maps type -> species / mode; nothing downstream
  lists them by hand. Import-time validation (`_validate_model_blocks`) checks that every block
  key is a learnable row, that `R_i < R_s <= 1` holds by the DISJOINT declared ranges, that
  `0 < R_B <= 1`, and that the initial dimer fraction is a linear row on `[0, 1]`.
- **Generated reaction network** (`simulation_rds_support.reaction_channels`): seventeen
  channels derived from the two blocks and registered verbatim by `build_system` -- six
  association fusions `A_m + A_m' -> B_slower(m, m')` (one per unordered pair of monomer modes,
  all at the same microscopic rate), three dissociation fissions `B_m -> A_m + A_m` at
  `kappa_OFF` (mode conserved), eight switching conversions (the four rates shared by both
  species; no direct f <-> i). Companion helpers: `diffusion_coefficients`
  (`D[X, m] = D_A x (R_B if X is the dimer else 1) x (1, R_s, R_i)[m]`),
  `association_reference_rate` (the compatibility normalization
  `lambda_ref = 6 D_A / r^2`, `r` = one particle diameter = 10 nm, explicitly NOT a physical
  bound; `lambda_on = R_ON x lambda_ref`), `stationary_mode_law` (the initial-mode law
  `pi_f : pi_s : pi_i = 1 : k_fs/k_sf : (k_fs/k_sf)(k_si/k_is)`), `rank_to_species` and
  `monomer_ranks` (particle-type rank -> molecular species, for the visibility layer).
- **Initial composition** (`parameterization.realize_initial_composition(N_R, x_B)`):
  `n_total = max(1, round(N_R))`, `n_B = min(round(n_total x_B / 2), floor(n_total / 2))`,
  `n_A = n_total - 2 n_B`; the requested and the realized `x_B` are distinguished and the
  realized composition is recorded in the `Labeling_Set`; each particle's initial mode is drawn
  from the stationary law of the isolated switching chain.
- **The one parameter-conversion rule** (`parameterization.to_physical` / `to_flow`, with
  `entry_to_physical` / `entry_to_flow` / `prior_center` per row): every ranged row declares its
  scale through `LOG_FLAG` (True = the estimator coordinate is log10 of the physical value;
  False = linear), and these are the only sanctioned conversions between the estimator's space
  and physical values. The RDS sampling, the training dataset (`inference_support.VideoDataset`
  takes the workflow's table), evaluation, calibration, the diagnostics tables, and the analyses
  all use them; no consumer writes `10**theta` by hand. `Theta_Set` files still store PHYSICAL
  values.
- **Structure audit** (`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.py`
  with its companion `.md`): a deterministic tier (D1-D6) checking the seventeen channels, the
  transform round trips, the initial compositions at `x_B = 0` and `1` with odd totals, and
  occupancy by species -- PASSED 6/6 on 2026-09-09 (profile `mars_pc`); and, behind the
  explicit `--run` flag, small-scale run checks -- subunit conservation, stationary mode
  occupancies, visible fractions, both-condition generation, and a lateral-boundary check --
  not yet executed, awaiting approval. Dry runs of the RDS, biology DLI, detector DLI,
  Inference, Evaluation, and Experiment entry points pass on the twelve-parameter table.

### Changed

- **Fission daughters are placed at twice the reaction distance** (20 nm,
  `SimulationStem.fission_product_distance_nm`, factor field
  `fission_product_distance_factor = 2`), outside the fusion radius, instead of at the
  reaction distance. At the 2 ms sub-step an eligible pair reacts with probability ~1, so
  daughters placed at the radius re-fused deterministically unless they diffused apart within
  one step (~50% per step for immobile daughters), which made the effective unbinding rate
  mode dependent. Declared convention; documented in PROJECT_CONTEXT.md and checked by the
  structure audit (D7).
- **The reaction-diffusion model** is the separated stoichiometry-mobility model: two molecular
  species x three mobility modes = six particle types, replacing the three-species
  monomer / mobile-dimer / immobile-dimer system. Immobility is a mode available to monomers
  and dimers alike, so the biology and the detector read the same model through the same
  blocks. Boundary behavior is unchanged and now stated: open laterally (x, y), periodic
  axially (z), no confining potential -- a receptor beyond the imaged field stays simulated,
  may return, and is not rendered while outside; the conserved total `N_R` is the simulated
  patch's total, and the in-field count is a distinct, time-dependent quantity.
- **Twelve learnable parameters replace the ten**, in the table groups `stoichiometry` and
  `mobility`: `count_total` (`N_R`, log10 [0.5, 3.0]), `fraction_dimer_initial` (`x_B`, LINEAR
  on [0, 1]), `relative_rate_dimerization` (`R_ON`, log10 [-2, 0]), `rate_dissociation`
  (`kappa_OFF`, log10 [-1, 1]), `diffusivity_alp` (`D_A`, log10 [-1.25, -0.25]),
  `relative_diffusivity_dimer` (`R_B`, log10 [-1, 0]), `relative_diffusivity_slow` (`R_s`,
  log10 [-1, 0]), `relative_diffusivity_immobile` (`R_i`, log10 [-3.0, -1.3]),
  `rate_fast_slow`, `rate_slow_fast`, `rate_slow_immobile`, `rate_immobile_slow` (each log10
  [-1, 1]). ALL ranges are DEVELOPMENT SETTINGS for building and checking the generator, not
  scientifically approved training priors.
- **The detector table's RDS nuisance rows** (`detector_parameterization.py`) are the same
  twelve keys (nuisance-from-object, supplied by the shared trajectory tier), grouped
  `stoichiometry` / `mobility`.
- **The visibility layer** acts by MOLECULAR species: occupancy and labeling select through
  `rank_to_species`, the `--occupancy` per-species syntax is `A=1.0,B=0.8`, and the lineage
  extractor reads subunit counts per particle type.
- **Old artifacts are rejected, never overwritten.** Trajectories and `Theta_Set`s generated by
  the 0.1.1 three-species model are rejected at the DLI stage by `rank_to_species` (unknown
  particle types `A`/`B`/`C`) and by the estimator schema guard
  (`artifacts.assert_schema_compatible` / the theta-width guard); estimators carrying the
  ten-parameter schema fail the same guard.
- **Documentation**: `PROJECT_CONTEXT.md` sec. 2 describes the model (species, modes, particle
  types, the seventeen channels, the diffusion parameterization, the association
  normalization, the boundary behavior, the twelve-parameter table with its development
  ranges, the initial state, the conversion rule); `DETECTOR_WORKFLOW.md` sec. 6.1 tabulates
  the twelve rows the detector marginalizes; `CLAUDE.md`, `README.md`, the Prime entry-point
  docstrings, and the analysis companion notes follow. Companion notes that record results
  computed under the 0.1.1 model keep those results under a dated note stating that the
  readout definitions changed (composition from `N_R` and `x_B`; `f_B = x_B / (2 - x_B)`;
  `f_R = x_B`).

### Removed

- The immobile-dimer species C and its parameters (`count_alp`, `count_bet`, `count_chi`,
  `relative_diffusivity_bet`, `relative_diffusivity_chi`, `rate_immobility`, `rate_mobility`),
  the hand-written four-channel network (`A + A <-> B`, `B <-> C`), and the per-species initial
  counts as learnable parameters.
- The blanket `10**theta` convention: `LOG_FLAG` is no longer a documentation field, and no
  consumer exponentiates or log-transforms a theta vector outside `to_physical` / `to_flow`.
- The deferred `C -> 0` internalization item of the scope statement, restated as receptor
  synthesis, degradation, and internalization (deferred; `N_R` conserved within a recording).

## 0.1.1 - 2026-09-07

The DOL-explicit observation layer, the reactive detector, and the condition slot.

### Added

- **Static labeling stoichiometry** (`labeling.py`): every receptor subunit draws an integer dye
  count once per recording from the condition's measured law -- MET-INLB `Bernoulli(q = 0.5)`
  (one engineered attachment site; the labeling probability measured for this preparation),
  MET-FAB `Poisson(1.64)` (the measured ensemble mean; matched-mean binomial and
  negative-binomial alternatives registered for sensitivity runs) -- composed with an optional
  static probe-occupancy multiplier (`--occupancy`, default 1, optionally per initial species).
  The laws are fixed and never inferred. A `Labeling_Set` record per task (one row per
  simulation: true and visible initial composition) is written beside the theta sets.
- **Subunit lineage** (`simulation_rds_support.extract_subunit_lineage`): the RDS stage registers
  ReaDDy's reaction-record observable and the DLI stage replays the records into a per-frame
  subunit-to-particle table -- fusion concatenates, fission distributes, conversion preserves --
  with a conservation check that fails loud. ReaDDy mints a new particle id at every reaction,
  plain species conversions included, so the lineage is what lets a static per-subunit quantity
  follow its subunit. `collapse_species_axis` replaces the repeated species-axis collapse.
- **Dye-centric renderer**: `render_dli_video(soul_poses, host_index, dye_counts, imaging)`
  renders every dye as an emitter at its subunit's host particle (`build_dye_tracks`), with one
  PSF width per subunit and one stationary OU brightness and bleaching process per dye. A
  subunit without a dye never renders; a dimer's dyes render at one position, so photons add
  with no multiplier. The renderer no longer restarts a dimer's brightness, PSF width, and
  bleach clock at every mobility conversion (the previous per-particle emitters did, once per
  new ReaDDy id).
- **The condition slot** of the runtime grammar: `Paths.with_condition` appends the condition
  token (`FAB`, `INLB`) to the alias -- `..._ALP_FAB_2S_50FPS_Video_Set_...`,
  `..._ALP_DETECTOR_INLB_2S_50FPS_Estimator.npz` -- while `rds_alias` / `theta_set_alias` keep
  the RDS products (the trajectories and their ten-parameter `Theta_Set`) under the bare alias;
  `Paths.record_set_path` builds the DLI-stage record sets. Every stage past RDS and every
  analysis take a required `--condition`; the HPC stage scripts, dispatchers, and generation
  controllers take `CONDITION` (validated, forwarded, and placed in the job name; an RDS-only
  simulation is exempt). The DLI stage additionally takes `--labeling-law` and `--occupancy`.
- **One shared RDS trajectory tier.** The trajectories and the ten-parameter `Theta_Set` are
  generated once, under the bare sibling alias (`Paths.sibling_alias`, exposed as `rds_alias`),
  and re-imaged by both workflows and both conditions at the DLI stage: the biology reads the
  `Theta_Set` as its label, the detector as the record of the reaction-diffusion nuisance it
  marginalizes (the ten RDS rows of `detector_parameterization` are nuisance-from-object,
  without ranges of their own). `Generate_Datasets.py` runs the tier once per split, fans the
  DLI passes out over `--workflows` x `--conditions`, refuses to regenerate an existing tier
  unless `--overwrite-rds` (a fresh draw would mislabel the videos rendered from the old one),
  re-images an existing tier with `--reuse-rds`, and checks the biology's per-condition
  `Nuisance_DLI` artifacts before anything runs. The biology HPC controller forwards
  `SIM_STAGE`, so `SIM_STAGE=rds` generates the tier alone.

### Changed

- **The detector calibrates against the reactive biology.** The detector re-images the shared
  reactive trajectory tier, so the ten reaction-diffusion parameters it marginalizes are the
  biology prior by construction. Under the labeling model a dissociating one-dye dimer leaves
  one visible and one invisible daughter, a kinetic track disappearance a static-composition
  detector would absorb into photobleaching; and a system with no registered reactions writes no
  reaction records, so a diffusion-only trajectory cannot be rendered. The detector
  posterior-predictive scene is rendered reactively at an RDS nuisance drawn from the biology
  prior (or pinned). The detector's HPC Simulation wrapper, dispatcher, and generation
  controller are DLI-only passes over the tier; an RDS-only job carries neither qualifier nor
  condition.
- The horizon audit spawns a (placement, render, labeling) seed triple per arm and records the
  condition and labeling law in every generated file; the posterior-predictive runner derives
  the condition from `--kind` and reads the condition's `Nuisance_DLI`; the Experiment,
  embedding-distance, and Nuisance_DLI analyses default their recording kinds to the run's
  condition (naming the other condition is a deliberate cross-condition application).
- The Sample-Geometric-Median analyses take the dataset condition as `--condition` and restrict
  the collection to that condition's rows; the `Nuisance_DLI` spec's `[block].condition`
  defaults to the artifact's own condition.
- `DETECTOR_WORKFLOW.md`, `PROJECT_CONTEXT.md`, `README.md`, `VALIDATION.md`, the HPC runbook,
  the analysis companion notes, and this repository's `CLAUDE.md` describe the implemented
  observation layer, the reactive detector, the condition slot, and the shared trajectory
  tier, and state the probe-kinetics assumption: ligand binding and unbinding within a
  recording are not modeled; first-order unbinding is absorbed by the per-condition
  calibrated bleach parameter; appearance from free labeled ligand and state-dependent
  affinity are declared residuals; partial occupancy is the static occupancy knob.

### Removed

- The dimer brightness multiplier and its two combination modes (`dimer_mule`, `dimer_model`,
  the dimer mask, the second-label draw, and the posterior-predictive `--dimer-model` flag):
  the dye-centric renderer needs no dimer-specific code path.
- The diffusion-only simulator mode (`build_system(pure_diffusion=...)`).
- The detector's own RDS tier: the `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Simulation_RDS.py`
  entry point, `detector_simulation_rds_support.py`, `build_nuisance_prior` and the RDS nuisance
  bounds of `detector_parameterization`, the import-time range-equality check, and the
  `Nuisance_RDS_Theta_Set` record.
- The `pooled` selection of the geometric-median analyses and the `CONDITION_CHOICES` constant:
  estimators, artifacts, and analyses are per condition.

## 0.1.0 - 2026-09-02

Founding release of the MONOMER_DIMER model family.

### Added

- **Codebase founded as a copy** of the tracked tree of `srm-and-sbi/srm-and-sbi-dimer-alp` at
  its frozen release `v0.4.23` (commit `d78b2f2`) — the reference implementation of the
  three-species DIMER model with the stationary OU brightness photo-physics. That repository's
  changelog records the copied functionality; this changelog records what this repository
  changes on top of it.

### Changed

- **Namespace**: package `srm_and_sbi_monomer_dimer_alp`; runtime prefix
  `SRM_AND_SBI_MONOMER_DIMER_ALP` on every entry-point script, output-file prefix, and
  project alias.
- **Condition tokens**: the experimental conditions are `FAB` (MET-FAB) and `INLB` (MET-INLB)
  in every filename pattern, schema field, CLI argument, and document; display surfaces prepend
  the receptor. The repo-iteration suffixes (`alp`, `bet`, ...) are a separate namespace and
  never name a condition. Experimental recordings enter this repository's data bank under the
  `FAB`/`INLB` names, with a provenance mapping to their public accession recorded at staging
  time.
- **Repository status** sections of `README.md` and `PROJECT_CONTEXT.md` describe this
  repository's charter: the DOL-explicit observation layer (measured degree of labeling,
  per-subunit label draws, probe occupancy), the reparameterized counts (true receptor
  abundance and composition), the condition axis (MET-FAB and MET-INLB as two frozen
  configurations of one codebase), and the `Nuisance_DLI` recalibration under the DOL-explicit
  model. Scientific sections describe the copied state where not yet rewritten.
