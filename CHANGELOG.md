# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.19 - 2026-09-24

Development tooling for the direct imaging estimators, the declared split of the EVAL tiers into
development and reserved tasks, and the guards that hold every run to it. No canonical stage, parameter
role, preprocessing step, or estimate changes; the direct estimators return the same numbers as before,
with more recorded beside them.

### Added

- `DETECTOR_WORKFLOW.md` §9.6 — the declared split, fixed before any further direct-estimator run. Two-second
  MET-FAB tier: tasks 0 to 19 are development data, tasks 20 to 24 (5,000 recordings) the reserved set for
  the verdicts of the two-second estimators. Twenty-second tier: tasks 0 to 9 development, tasks 10 to 19
  (1,000 attempted recordings) reserved for the fluorescence-loss verdict at 1000 frames; its usable-recording
  minima are judged separately. A reserved task scored by a direct estimator outside its verdict run, or used
  in tuning, leaves the reserved set. The direct-estimator runs on record had read two-second EVAL tasks 0
  and 1 and no other EVAL task (their run folders on the PC, rcl01 and JUWELS).
- **The split is enforced before any recording is read.** A tier run must declare `--purpose development` or
  `--purpose verdict`; there is no default (`direct_acceptance.check_run_purpose` over `DECLARED_SPLIT`). A development
  run on a declared tier refuses every task outside its development set. A verdict run requires exactly the
  reserved tasks of its tier, every recording of them, an estimator the set is kept for, and no earlier
  verdict folder of that estimator on the tier. TRAIN and TEST tasks and undeclared tiers hold no reserved
  task. The purpose record goes into `summary.json`, `provenance.json` and the report, and a dry run applies
  the same refusals.
- **Run folders are never reused.** A tier run's folder carries its purpose (`_DEV` or `_VERDICT`), then
  `--run-suffix`, for example `_DEV_<commit>`; a suffix that repeats a purpose token is refused. An existing
  folder is refused before anything is read, and the folder is created in one atomic step
  (`provenance.reserve_run_folder`). The flicker harness names its labeling and bleaching setting in the
  folder name, so its documented variants never share a folder.
- **A changed implementation invalidates the run.** Every utility compares the implementation hash at startup
  with the one taken after the computation, before it writes. When they differ, the verdict table marks the
  run `INVALID` in its step 0, the arrays are kept as diagnostics, and the exit status is 3
  (`direct_acceptance.apply_code_provenance`), whatever the other verdicts were.
- **Traceable runs** in all three direct estimators: `task` and `index` arrays identifying every recording,
  and a `provenance.json` in every run folder, the PSF experiment mode's included
  (`provenance.analysis_run_record` over `provenance.DIRECT_ESTIMATOR_FILES`: command, host, Slurm job,
  versions, declared commit, purpose, implementation hash at startup and at write).
- **`Direct_PSF_Width` observable per-recording quantities**, candidate predictors for a correction of the
  estimates or their ranges (§9.6 permits only quantities an experimental recording provides as well):
  median spot signal in photons, median spot signal over the background noise, spot fits per frame, the
  median length of the tracks the population estimate keeps (counted in fits that pass its relative-error
  filter; `direct_imaging_estimates.psf_width_population` now also returns those lengths), and the tracks
  kept over the mean number of fits per frame. The last one is not a fragmentation measure: partial
  visibility, bleaching and turnover raise it as much as broken tracks do. The planned bootstrap over
  tracks is not implemented: tracks of one subunit need not be independent units, and resampling them
  cannot see a brightness-dependent offset.
- **`Direct_PSF_Width` selftest diagnostics**: `--selftest-grid brightness` (`mu_pc` on the five edges of its
  prior quarters at three spreads), `--selftest-labeling FAB` (the bare FAB dye-count law at probe occupancy
  1, not the tier's occupancy model), and an accounting of every scene's `mu_r` error against the truth in
  five terms that sum to the total, all five shown: finite population, detection selection, duplication of
  subunits whose detections formed several tracks, fitting, and a remainder. The accounting is conditional
  on nearest-subunit matching and on the order of the terms; it is not a causal decomposition. The true
  widths come from the renderer's own seeded draw.
- `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Mismatch.py` and companion note:
  renders scenes with known truth, reproduces every dye's photons from the renderer's seed, and applies the
  flicker estimator's shape match to seven versions of the traces: multiplicity, span selection, detection
  gaps, photometry noise, linking alone (the same matched detections grouped by the production linker), and
  then the detections outside the matched set. Every level records why it produced no estimate, and every
  mean and paired contrast carries its scene count. A contrast suggests a contribution under those scenes
  and that ordering. Scenes use the bare dye-count law at probe occupancy 1. It reads no EVAL task.
- `Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Direct_Estimator.sh`: the direct estimators and
  the mismatch study on one whole CPU node, submitted directly with `sbatch` (HPC README §6). `PURPOSE=`
  declares a tier run's purpose and is required; the log states that exit statuses 1 and 2 also cover an uncaught exception
  and an argument error or refusal.
- `tests/test_direct_estimator_guards.py`: the required purpose and its checks against the declared split, the
  folder refusal in every utility, the harness's variant folders, linking contrast and failure records, the
  retained-track observable, the decomposition table, and the invalidation of a changed implementation.

### Changed

- The section headings of the three direct-estimator companion notes no longer carry package versions
  (`Acceptance mechanics`, not `Acceptance mechanics (0.1.13)`): the workspace document rules keep version
  labels out of body text, and this log records when each section arrived. Headings that identify the code
  behind a recorded result, such as `Development outcome (2026-09-21, code 2b9c32e, EVAL tasks 0-1)`, stay.

### Verification

- The PSF estimates are bit-identical to 0.1.18: the committed 0.1.18 tree and this one, run in separate
  processes on three rendered FAB scenes (dim and broad, dim, bright and broad), agree in every returned
  number (`mu_r`, `sigma_r`, the uncorrected spread, both standard errors, the track and spot counts).
  The default selftest returns exactly the estimates of its earlier run on the same machine, and the
  recorded selftest estimates to 1.4e-7, the size of cross-machine differences in the spot fits; the five
  terms of its accounting table sum to the total in every scene.
- The flicker harness reproduces the renderer's photons exactly. A one-scene run completes all seven levels,
  and the six levels it shares with the earlier one-scene run reproduce that run's estimates.
- The tier path of all three estimators, checked on the local development task 0 (two recordings on the
  PC), writes the identifiers, the purpose record and `provenance.json`, and exits 2 (PSF width, flicker
  rate: insufficient evidence) and 0 (fluorescence loss at 2 s: not applicable), as the rules require. With
  the write-time hash forced to differ, a loss selftest exits 3 with step 0 `INVALID` in its report and
  summary.
- All 72 existing tests and the 19 new ones pass; the wrapper passes `bash -n`.

### Development diagnostics

- PSF brightness selftest under the FAB law (15 scenes, about 120 visible subunits each, 2 s; informational,
  no verdict). The fitting term of the `mu_r` error rises with brightness, from −0.009 dex at 100 photons
  per dye to +0.009 dex at 562 (mean over three spreads), the direction of the brightness dependence on the
  development tier: a preliminary lead. Detection selection matters only for broad spreads when dim,
  −0.038 dex at 100 photons and a spread of 0.5. The duplication term ranges up to 0.03 dex across the
  scenes; fifteen different scenes do not establish it as a variance component the standard error omits.
  These scenes calibrate nothing; the development tasks are to carry any correction.

## 0.1.18 - 2026-09-24

Regenerates the first analyses downstream of the 0.1.17 optimizer and records what the corrected MAP
shows. Bounds the PSF-integration working memory using 2 s frame blocks; total renderer memory still
grows with recording length. The one code change leaves every rendered video bit-identical; no stage,
parameter role, or preprocessing step changes.

### Changed

- **Experiment stage regenerated for the multiple-dye baseline and `CAP256`** (JUPITER jobs 1972300
  and 1972333, four nodes each, unrestricted pool, 60 MET-FAB recordings in ten windows). All 1,200
  ascents stopped on patience (median 276 and 378 steps); the products validate under artifact schema 2
  and estimate definitions 2 with the frozen optimizer settings. The pre-0.1.17 baseline and `CAP256`
  MAP products on JUPITER were deleted; the `CAP256` originals are kept on the PC beside the new
  products as `_RUN_<jobid>` copies for now. The regenerated comparison record
  `..._CAP256_vs_Baseline_Experiment` reads both products through the artifact schema, matches windows
  on their stored identifiers, totals the stop reasons from the job logs and stores the scripts, the
  logs and the source checksums beside its statistics and figures.
- **The renderer integrates the PSF in 2 s frame blocks** (`simulation_dli_support.add_pixel_counts`;
  the block, `PSF_FRAME_BLOCK`, is the frame count of 2 s at the fixed cadence, 100 frames, and
  `psf_frame_blocks` gives the layout). The blocks are computational only: trajectories, labeling,
  flicker and bleaching are generated once for the whole recording and stay continuous across block
  boundaries; only the PSF accumulation is split. The integration's temporaries scale with pixels ×
  emitters × frames, so one isolated 20 s render with 900 dyes, about the largest emitter count recorded
  in the completed 20 s tasks (880), peaked at 7.88 GiB. The 20 s FAB EVAL rendering of 2026-09-22
  (JUWELS job 14266249, ten tasks per node) ended with 3 of 20 tasks complete (625 of 2,000 videos): the
  other rendering processes were killed with no Python error while the batch step's sampled `MaxRSS`
  reached 78.5 and 82.0 GiB of the 91.8 GiB configured per node. `MaxRSS` is the maximum across a step's
  accounting tasks; it covers a whole node here only because the launcher starts the ten renderers as
  background children of the batch script, one accounting task. That is consistent with memory
  exhaustion but does not establish it: Slurm recorded the job as failed, not out of memory, and no log
  holds an explicit out-of-memory record. Its reaction-diffusion stage had completed. Blocking bounds
  the integration's working memory at the 2 s size. On one isolated render with 900 dyes the peak falls
  from 7.88 to 3.50 GiB at 20 s, from 4.16 to 1.98 GiB at 10 s and from 2.30 to 1.44 GiB at 5 s; at 2 s
  (1.18 GiB) and 1 s (0.81 GiB) the recording is one block and the integration runs in one pass, as
  before. Render time is unchanged. Total renderer memory still grows with recording length: the
  full-length intensity, emitter tracks and brightness, the noise model's arrays and the output frames
  are not blocked. A production task holds more than one render call (it loads trajectories, converts
  and compresses each video, and keeps the previous video's frames), so ten 20 s tasks per node are
  expected to fit, pending a full-task memory confirmation on JUWELS from a completed ten-task run and
  its final accounting. Frames are independent throughout the integration, so the output does not
  change: `tests/test_render_frame_blocking.py` holds the intensity and the rendered video under a fixed
  seed to the single-pass computation, bit for bit, at 1, 2, 5, 10 and 20 s, on scenes with monomers,
  dissociating dimers, zero to three dyes per subunit, absent particles and bleached dyes, and at twelve
  block sizes from 1 to 4,096 frames. On the PC its 5 tests and the 68 existing tests pass.

### Documentation

- `DETECTOR_WORKFLOW.md` §9.8 — the regenerated Experiment stage and the finding that the remaining MAP
  behavior follows from the shape of each estimator's learned density and the seed-dependence of the
  ascent on it: `CAP256`'s density holds two competing joint solutions for bleaching of nearly equal
  height (0.004 to 0.20 nats apart on the probed windows), between which the ascent's seeds select from
  window to window while the median and the SGM stay at the prior center of a weakly constrained
  parameter; its `sigma_r` and `mu_r` densities are skewed, placing the MAP a stable half-IQR below the
  median; the baseline's ascents converge consistently and its three estimates coincide. A probe on the
  real checkpoints (six windows, nine independent candidate pools each, density profiles, one ascent
  seeded in each branch) reproduces every MAP within its region to 0.04 dex, so the instability is not
  the corrected bookkeeping. The brightness gap to the per-detection reference of §6.7 is qualified as an
  arithmetic comparison. The §9.7 and §6.11 banners name what is regenerated and what is pending.
- `VALIDATION.md` §3.4 — step 4 (regenerated analyses) with its status, and the reporting rule extended
  with the empirical case for reading the three point estimates together.
- `PROJECT_CONTEXT.md` — one-sentence pointer to that case.

## 0.1.17 - 2026-09-23

Resolves the MAP optimizer before the analyses downstream of training and testing are regenerated,
and records the validation protocol for the three point estimates (`VALIDATION.md` §3.4). The
median and SGM calculations are unchanged; they now summarize 10,000 draws per observation.
**Products made before this version are refused on
read** (estimate-definitions version 2): their MAP estimates come from the unscaled optimizer and are
recomputed, not converted. **Artifact schema version 2**: the optimizer block requires
`step_coordinates`, so schema-1 products are refused as well.

### Fixed

- **The MAP ascent overshot and usually returned its best starting candidate.** The initial Adam
  step of 0.128 dex (`learning_rate_minimum` 1e-3 x `learning_rate_maximum_factor` 128) is several
  times the posterior IQR of the well-identified parameters (`mu_r` about 0.02 dex), so the first
  steps carried a seed about eight IQRs away and dropped the density by a median 55 nats; with a
  patience of 100 most ascents stopped before recovering. On 2,000 EVAL recordings with two
  independent candidate pools each (the MAP benchmark, JUPITER job 1970791), the optimizer came
  within 1e-3 nats of the best optimum found in 21 % of pools, and the two pools of one recording
  gave MAP vectors a median 0.25 IQR apart (maximum 0.94).
- **Best-point retention was gated by the tolerance.** A point that beat the retained pair by less
  than the tolerance was not kept, so the returned pair could sit up to one tolerance (1e-3 nats)
  below the best point visited. Every strictly better finite (score, vector) pair is now retained;
  the tolerance gates only the stopping patience and the plateau scheduler.
- **Unrestricted sampling kept every call's autograd graph.** `collect_theta_prex` sampled the flow
  with gradients enabled, so each call's draws held the embedding's forward graph (about 1.5 GiB for
  a 2 s video) for as long as they lived; memory grew with the number of calls, and 300 draws in
  batches of 100 ran out of memory on a 4 GB GPU. Sampling now runs under `torch.no_grad()`, whose
  scope ends before the MAP ascent; peak GPU memory was then approximately constant, about
  0.43 GiB, over the tested range up to 50,000 draws, on this checkpoint and a 4 GB T2000.
- `detector_nuisance_dli.build_map_estimate_pool` read `eval_cfg.learning_rate`, a field the
  evaluation config did not have, so the MapEstimate pool path raised on first use; it now reads the
  configured learning rate and tolerance.
- `collect_theta_prex` looped forever on a zero batch size; a size or batch below one is now refused
  before sampling.

### Changed

- **The ascent steps in units of each observation's candidate-pool IQR.** Adam moves
  `u = (theta - m) / IQR` per coordinate (`evaluation.pool_scale`: the pool's median and
  interquartile range), so one learning rate is the same fraction of every parameter's posterior
  spread; the density is still evaluated and maximized in theta. A zero or non-finite IQR skips the
  ascent, returns the best candidate (its score is the density at it) and is reported, never
  divided through. The loop structure is unchanged: all seeds in one Adam tensor, the scheduler on
  their mean score.
- **Settings** (`InferenceEvaluation`): `learning_rate` 0.05 and `learning_rate_minimum` 5e-4, both
  in pool-IQR units; `numb_steps` 2000, `optimizer_patience` 200, `scheduler_patience` 20,
  `learning_rate_factor` 0.5; `tolerance` 1e-3 nats as its own field. `learning_rate_maximum_factor`
  and `tolerance_factor` are removed from the evaluation config, since a learning rate in IQR units
  and a tolerance in nats are no longer tied to one floor; the training config keeps its own.
  `--learning-rate` is read in pool-IQR units. A learning-rate reduction does not reset the patience.
- **One outer sampler call per draw set, and 10,000 summary draws.** `theta_prex_batch_size`
  (configuration only; no command-line flag) goes from 100 to 10,000, so the MAP candidate pool
  (`theta_prex_size`, still 1,000) and the summary draws are each drawn in one outer call. Batches
  of 100 re-ran the embedding network for every 100 draws: 10,000 draws took 16.9 s in batches of
  100 and 0.20 s in one outer call on a Quadro T2000, with about the same peak GPU memory; the
  embedding still runs a few times per outer call (twice in the measured bounded case). Draws from
  one call and from batches are consistent with the same distribution (ten recordings, both pool
  modes, per-coordinate two-sample KS, none rejected after correcting for 60 tests). That supports
  consistency; it does not prove equality, and a fixed seed no longer reproduces the same draws, or
  the same returned MAP vectors, once the batching changes.
- **`posterior_samples` goes from 1,000 to 10,000** (`--posterior-samples` overrides it). This raises
  the numerical resolution of the quantiles, the median and the SGM; it does not correct posterior
  miscalibration. The exact SGM's CPU memory is quadratic, about 1.5 GiB per concurrent worker at
  10,000 draws, within the stage scripts' `--mem=480G` per node. Posterior calibration, the horizon
  audit and the Nuisance_DLI posterior-sample pools inherit the new count unless overridden on their
  command lines; their scoring, storage and pooling costs do not necessarily share the sampler's
  speedup.
- **Stop reasons are reported.** `optimize_elite` returns `(score, theta, info)` with the stop
  reason (`early`, `budget`, `non-finite`), the steps taken, the best seed's score and the final
  learning rate; a non-finite score ends the ascent with the best finite pair kept.
  `map_estimate(..., return_info=True)` adds the pool median and IQR and a per-coordinate validity
  flag (`invalid-scale` when the ascent was skipped). The Evaluation, Experiment and CD86/CTLA-4
  controls runners log each observation's stop reason and steps and a closing count of stop
  reasons; `--debug` adds each observation's pool IQR.
- **Effective settings are printed and recorded from one object.** The runners build
  `evaluation.optimizer_contract` once, print it (`evaluation.optimizer_summary_lines`) and store
  the same dict in the manifest; the block gains `step_coordinates`, which
  `artifact_schema.OPTIMIZER_KEYS` now requires (artifact schema version 2), and names the
  retention rule.
- **Estimate-definitions version 2.** The MAP definition names the IQR-unit steps;
  `SUPPORTED_ESTIMATE_DEFINITIONS_VERSIONS` is `(2,)`, and the refusal message says why a version-1
  product is recomputed. The Nuisance_DLI MapEstimate pool contract carries the version and the new
  settings, so an old MAP pool is not reused.
- **Definitions text aligned with the protocol**: the median names its linear interpolation and
  transform order, the SGM is named an exact sample medoid.
- The MAP benchmark's configuration `B` runs the production optimizer as configured and records its
  stop reasons; configurations `C` to `H` and their scaling formula are labeled a benchmark proposal,
  not adopted.

### Added

- **Point-estimate validation protocol**, `VALIDATION.md` §3.4: median correctness, SGM
  correctness and sampling stability, MAP optimization, then regeneration and interpretation, each
  with its completion criterion and status, and the reporting rule separating implementation
  correctness, stability and accuracy against synthetic truth. Status summary in
  `DETECTOR_WORKFLOW.md` §9.8; pointers in `PROJECT_CONTEXT.md` and the README.
- **Validation utilities** (Analysis family, outside the dispatcher): the point-estimate validation
  utility (`..._DETECTOR_Point_Estimate_Validation`, kernel `point_estimate_validation`), which ran the
  median check and supplied the SGM checks' draws, and the MAP benchmark
  (`..._DETECTOR_MAP_Benchmark`, kernel `map_benchmark`, with the JUPITER script
  `..._DETECTOR_HPC_MAP_Benchmark.sh`), each with a companion note recording its results.
- **Tests**: `test_median_reference.py` (interpolation, parameter order, transform order),
  `test_sgm_reference.py` (summed distances, prior-width scaling, membership, duplicates, ties),
  `test_point_estimate_validation.py`, `test_map_benchmark.py` (the batched engine against a serial
  `torch.optim` reference), and `test_map_optimizer_invariant.py` rewritten for the adopted optimizer
  (the two invariants, strict retention below the tolerance, patience counted from the last
  meaningful improvement, IQR units, non-finite stop, the zero-IQR path, the configured settings,
  the embedding and condition shape restored after a failure inside `map_estimate`, learning-rate
  reductions that reach the floor without resetting the patience while the best visited point is
  returned, and the sampler's boundaries in both pool modes: one draw, an exact batch multiple, a
  partial final batch, no autograd graph, invalid sizes refused); a row-order test for the SGM; and
  refusal cases for a version-1 product and a missing `step_coordinates`. Each of the new failure
  tests was checked to fail when its fault is injected.

### Verification

On the ten pilot recordings, two independently seeded candidate pools each, the production entry
point returned a score equal to the re-evaluated density at the returned vector in all 20 pools and
above the best initial candidate by 0.02 to 0.59 nats; every ascent stopped on patience, after 219 to
312 steps. Evaluated on the same machine, the benchmark's reference optimum lies within 4.5e-4 nats
of the returned score, an L-BFGS polish from the returned vector gains at most 5.6e-4 nats, and the
two pools agree to a median 0.0009 IQR (maximum 0.03). That run drew its candidate pools in
batches of 100; the optimizer and the candidate count are unchanged, so it remains relevant, though
the same seeds no longer reproduce those exact vectors.

### Documentation

- Docstrings distinguish the exact medoid from a Weiszfeld geometric median snapped to its nearest
  member (the collection-level kernel above 20,000 members) and name each summary's population and
  coordinate scaling: the per-observation posterior-draw SGM (estimator coordinates over prior
  widths), the SGMs of window MAPs in temporal dynamics and the SGM analysis (physical coordinates
  over the physical prior range), the posterior-predictive `cell-sgm` (a cell's chunk MAPs, physical
  coordinates over their range) and the Nuisance_DLI per-window SGM (posterior draws, physical
  coordinates over the physical prior range); none claims that selecting one member preserves
  correlations. `temporal_dynamics.pooled_summary` and `sample_geometric_median.summary_vectors`
  state their interpolation convention.
- `DETECTOR_WORKFLOW.md` §9.8 no longer says the Nuisance_DLI pool cache ignores the optimizer
  implementation; it has keyed on the MAP computation contract since 0.1.16.

## 0.1.16 - 2026-09-23

Makes the three point estimates a contract of every Evaluation and Experiment product, renames the
stored MAP field, and records the computation provenance each product was made under. No optimizer
setting, density objective, prior-bound handling or SGM distance convention changes, and no
downstream analysis expands its scientific output. This is a **breaking artifact-schema change**.

### Changed

- **Every product carries all three point estimates for every observation.** `map_estimate` (the
  numerical MAP candidate), `posterior_quantiles` (five levels; the marginal median is the 0.50
  level) and `posterior_sgm` (the SGM of the same draws) are computed unconditionally and validated
  before anything is written; a missing, malformed or non-finite estimate stops publication and
  names the observation, estimate and stage. The definitions live in one place,
  `evaluation.POINT_ESTIMATES`, each stating operator, source population, grouping, coordinate
  space, distance scaling and stored field; draws are labeled `posterior-draw` under the bounded
  pool and `flow-draw` under the unrestricted one. Every report opens with a definitions table
  and shows the three side by side.
- **Breaking rename: `inferred_log10` -> `map_estimate`**, through every active producer and
  consumer (Evaluation, Experiment, the CD86/CTLA-4 controls runner and its temporal companion,
  temporal dynamics, population composition, the standalone SGM stage, Nuisance-DLI construction
  and its SGM analysis, posterior-predictive video generation). No fallback: an old product is
  refused by `artifact_schema.reject_obsolete` with the message that schema conversion does not
  repair affected MAP estimates. Existing products on disk are untouched.
- **Persisted computation contract.** Each shard and merged product stores a `manifest_json`
  (`artifact_schema.build_manifest`): schema and estimate-definition versions, parameter keys and
  coordinate transform, pool mode and draw label, draw count, quantile levels, SGM scaling and
  coordinates, optimizer settings (`evaluation.optimizer_contract`, including the 0.1.15
  bookkeeping marker), code provenance (`provenance.code_provenance`: HEAD, dirty flag and a
  deterministic hash over the implementation files by path and content, missing files marked),
  checkpoint identity (the estimator's `weights_sha256`), run identity, seed policy and observation ids.
  Evaluation shards now store `task_index` / `sim_index`.
- **Merging requires one computation.** `experiment_support.merge_validated_shards` validates
  every shard, requires the shard manifests to agree on every contract key, and requires the
  merged observation set to EQUAL the expected inventory (Evaluation: every `(task, sim)` of the
  EVAL tasks; Experiment: every `(kind, cell, chunk)` a recording on disk yields), not merely to
  have the right count. Rank coverage stays a separate check.
- **Nuisance-DLI MapEstimate pool cache** keys on the MAP computation contract
  (`detector_nuisance_dli.live_map_pool_contract`), so an implementation change cannot reuse an
  old MAP pool; the draw-only PosteriorSample pools carry no such entry and are unaffected.
  `POOL_FORMAT_VERSION` is unchanged for that reason.
- **Rendering separated from computation.** The "View A / View B" framing is gone; both combined
  figures always draw every panel; the marginal median is located by level from the manifest
  (`artifact_schema.median_level_index`), never by an assumed index, and the five canonical levels
  are validated on load. Report-only rendering (`persist_arrays=False`) leaves arrays untouched
  and reads the recorded definitions.

### Removed

- `--summary {map,posterior,both}` (all three runners) and `SUMMARY=` (four stage scripts, both
  dispatchers): supplying either is an explicit error, not a silent no-op.
- `--allow-partial` on Evaluation and Experiment `--merge`: a merge that skips ranks can never
  satisfy the exact-inventory check. (It remains on the posterior
  calibration stage and the Nuisance-DLI pool merge, which are not products of this contract.)

### Unchanged, deliberately

- All optimizer settings and the learning-rate ladder; the SGM distance conventions of every
  existing construction (posterior-draw SGM in estimator coordinates scaled by prior width;
  window-SGM and the standalone stage in physical coordinates); the experimental population
  composition (posterior draws) and its synthetic validation arm (MAP-based); the standalone
  Experiment SGM (an SGM of window MAPs, now named as such).
- Production cost: unchanged relative to the runs since 0.1.8 that already drew the posterior and
  computed quantiles and SGM at the same draw count (`SUMMARY=both`); the baseline 2 s FAB
  Experiment product predates the SGM output and is among the products to be regenerated. A
  local `--summary map` shortcut that skipped the draws no longer exists, by design.

### Enforcement completed after review (same release, 2026-09-23)

Two rounds of independent review reproduced seven, then three further, gaps between the contract
as stated and what the code enforced. Each is closed here with a regression test; none changes an estimate's definition, an
optimizer setting or any product on disk.

- **The manifest is validated, not only recorded** (`artifact_schema.validate_manifest`, called
  by `validate_product`): every contract key present and typed; the estimate-definitions version
  among the supported ones; `pool_mode` known and `draw_label` equal to its entry in
  `artifact_schema.DRAW_LABELS` (which `evaluation.draw_label` now reads); `sgm_scale` finite,
  positive, one entry per parameter; the optimizer, code, checkpoint, run-identity and
  seed-policy blocks complete; `window_geometry` and `condition_labels` required for the
  experiment stage (the labels must equal the stored `kinds`, and `kind_index` must index them)
  and null for evaluation. Arrays: `scores` and `true_log10` finite; identifier arrays
  integer-valued; `posterior_quantiles` nondecreasing along the level axis (equal neighbors
  valid); the optional `posterior_samples_cloud` of shape `(N, n_summary_draws, D)` and finite.
- **Writers validate on entry.** The Evaluation, Experiment and controls writers call
  `validate_product` first and decode the manifest from the arrays; the separate `manifest`
  argument is gone, so no caller can persist or render an invalid product. Report-only rendering
  (`persist_arrays=False`) exists on all three.
- **Every consumer reads through the schema with exact parameter keys.** The population
  composition's experimental loader now uses `load_product` (a manifest-less legacy file is
  refused); every consumer calls `artifact_schema.assert_parameter_keys`, which requires the
  product's keys to equal the workflow's key for key and in order, instead of comparing counts.
- **Shards of one computation are identified fully.** The contract gains `seed_policy` (base
  seed and the explicit per-rank rule: same seed on every rank, no offset), `window_geometry`
  (frames, stride, span) and `condition_labels`. `run_identity` is the logical invocation: the
  product label and an `invocation_id` created by the launcher (the stage scripts export
  `SRM_AND_SBI_INVOCATION_ID` before `srun`; a distributed rank without it stops, a
  single-process run generates its own); it must agree across shards. The execution attempt is
  recorded separately in each product's `execution` block (its Slurm job id, null outside
  Slurm) and is never compared, so a missing rank recomputed in a new job, or locally, merges
  with the original shards; the merged manifest keeps each shard's rank, world size,
  observation count, write time and execution under `shards` (validated: the counts must sum
  to the product's) and records the merge step's own execution. A replacement rank must run
  from the same HEAD, dirty state and implementation files, because the whole code block is part
  of the contract. Per-run fields (`kinds`) must be equal on every shard, not taken from the
  first.
- **Optional storage is a run-level setting.** Each manifest declares the optional arrays its
  run stores in `stored_optional_fields`, a contract key (today only the Experiment stage's
  `posterior_samples_cloud`, under `--dump-posterior-samples`). A product must carry exactly the
  declared arrays; the merge refuses shards whose presence differs, naming every shard on each
  side, and refuses any stored array it would not carry. Before this, a merge in which one shard
  lacked the raw draws succeeded and silently dropped them.
- **Empty ranks write valid empty shards.** A rank with no work writes correctly shaped
  `(0, D)` / `(0, D, Q)` / `(0,)` arrays (`artifact_schema.empty_product_arrays`) and a full
  manifest; the merge admits them and refuses a final product with no observations. The
  incomplete-shard-set message no longer recommends the removed `--allow-partial`.
- **One TIFF layout rule.** `experiment_support.inspect_recording` reads the series metadata
  (one 3-D series, axes `TYX` / `IYX` / `QYX`, frames first) and is used by the window inventory,
  by `read_cell_chunks` (which now also checks the loaded array against it) and by a preflight
  over every selected recording before estimation; a single page holding a 3-D image, an extra
  axis or several series is refused with every offending file named. The controls runner reads
  through the same function instead of its own `imread` loop. The MET recordings on disk are
  1000-page `TYX` series, so the earlier page-count inventory happened to agree on them; the
  rule now makes that agreement a check rather than a coincidence.
- **Provenance captured at startup.** Code provenance is taken once the estimator is loaded and
  finalized at write time (`provenance.finalize_code_provenance`: startup hash, at-write hash,
  `changed_during_run`); the validator compares the startup and at-write records itself
  (aggregate hash and per-file entries) and requires the flag to agree, so neither a changed
  implementation nor a code block that contradicts itself passes. The refusal is fail-closed at
  publication, not fail-fast: the computation has already finished, and the product is not
  written. The checkpoint identity is the checksum of the weights actually loaded
  (`load_estimator` attaches `weights_sha256`), never a re-read of the path. The hashed file list
  gains the controls runner, the Nuisance-DLI module and `provenance.py` itself.
- Smaller: the stage scripts and dispatchers refuse `SUMMARY` when it is supplied at all, even
  empty (`${SUMMARY+x}`); four stale comments naming `--allow-partial` are corrected; the retired
  `--summary` option is hidden from `--help` and still an explicit error.
- **Schema 1 is frozen at this release.** It was refined in place during the release while no
  product of it existed on disk (checked before release: no retained `.npz` in the Posit or Labor
  tiers carries `manifest_json`); any later incompatible change takes a new
  `ARTIFACT_SCHEMA_VERSION`.

### Tests

New: `tests/test_point_estimates_contract.py` (definitions and stored-field mapping; canonical
levels; draw labels; median and SGM from one draw set, SGM membership and expected medoid under the
declared scaling; retired options rejected by both parsers), `tests/test_artifact_schema.py`
(valid products; rejection of obsolete fields, missing companions, non-finite values, bad shapes,
duplicate ids, missing manifest; membership-not-count; merge contract, exact inventory and order;
MAP-pool cache contract invalidating only MAP pools; deterministic implementation hash with
missing-file marking; report-only rendering leaving arrays byte-identical while producing every
table and figure) and `tests/test_contract_enforcement.py` (one negative test per review finding:
manifest keys, versions, types and cross-field rules; experiment geometry and labels; scores,
truth, identifiers, quantile order and the optional cloud; the three writers refusing an invalid
product before persisting or rendering, and report-only rendering for Experiment and controls; the
composition loader refusing a legacy product and a wrong key order; exact parameter keys; merges
refusing a different seed, invocation, geometry, label set or `kinds` array and retaining shard
provenance; empty shards valid and mergeable, an empty final product refused, the coverage message
without the removed option; the TIFF rule shared by inventory, reader and preflight and rejecting a
planar single page and a 4-D stack; startup-versus-write provenance and the hashed file list; the
launcher-created invocation id and the stage scripts exporting it before `srun`;
`posterior_summary` obtaining ONE collection and deriving quantiles and SGM from it; the retired
option hidden from help; a replacement shard recomputed in another Slurm job, or locally, merging
with every execution retained while the logical identity is still compared; optional draw storage
as a run-level setting, with all-present, all-absent and mixed shards, empty ones included; the
validator comparing the startup and at-write implementation records itself and refusing a
contradictory flag). Shared fixtures in `tests/_product_fixtures.py`. Executed by direct
invocation (pytest is not installed in the PC `SRM_AND_SBI_ENVY_V0` environment): 7 + 4 + 18 = 29
tests pass; the pre-existing `test_map_optimizer_invariant.py` (3) and
`test_temporal_dynamics_drift.py` (2) still pass. The HPC scripts pass `bash -n`; the detector
dispatcher exits 1 with `FATAL` on `SUMMARY=` (empty) and proceeds to its dry-run preview without
it; Evaluation, Experiment and the controls runner `--dry-run` cleanly on the PC for both workflows.
These tests establish the artifact contract; they do not establish MAP convergence or scientific
calibration.

### Deferred

Expanding the population-composition validation arm and the standalone SGM stage to report all
three estimates; conversion of the one corrected old-schema product (`..._CAP256_MAP_Experiment_RUN_1953816`,
schema-only, with original and conversion provenance kept distinct); production re-runs; the
optimizer benchmark; attempt-specific progress-log names, so that a recomputed rank 0 no longer
truncates the original run's log (until then the recovery recipe in `Script_Bank/HPC/README.md`
saves the log and `figures/` first). None is implied by this release.

## 0.1.15 - 2026-09-22

Corrects the MAP seed-then-optimize step, which returned a score and a parameter vector taken
from two different points. No canonical stage, parameter role, preprocessing step, prior, or
trained estimator changes, and no retraining is implied.

### Fixed

- **The MAP optimizer returned coordinates one step past the point it scored.** In
  `evaluation.optimize_elite` the bookkeeping ran after `optimizer.step()`, which updates the
  parameter tensor in place, so the recorded pair combined step t's score with step t+1's
  coordinates. The returned vector therefore sat about one learning rate away from the scored
  point in every coordinate whose gradient pushed consistently, while the reported score belonged
  to the point before the move. The bookkeeping now runs before the update, which keeps two
  invariants: the returned score is the density at the returned vector, and, because step 1 scores
  the elite seeds themselves, it is never worse than the best seed's. An analytic single-peak
  density reproduces the old behavior without any trained flow (returned 0.97739 or 1.02261 for a
  mode at 1.0, depending on which side the seeds approached from, reporting the mode's score in
  both cases); regression tests in `tests/test_map_optimizer_invariant.py`.

  Measured effect on the stored 2 s FAB products: `|MAP - posterior median|` carries a sharp spike
  at exactly 0.128 dex, the initial Adam learning rate (`learning_rate_minimum` 1e-3 x
  `learning_rate_maximum_factor` 128, printed by the stage as `learning_rate: 1.280e-01`), standing
  3.4x above the neighboring background in the baseline; `mu_r` lands within 10 % of exactly that
  value for 51 % of baseline and 66 % of `CAP256` recordings, which is what the three `mu_r` MAP
  bands at -0.13, 0 and +0.13 dex are. Every MAP column, figure and conclusion in the Evaluation
  and Experiment reports, in the MAP drift rows, in the direct-versus-neural comparison and in
  `DETECTOR_WORKFLOW.md` sections 6.9 to 6.11, 9.6 and 9.7 is therefore under re-measurement. The
  posterior median, the SGM and every calibration result come from posterior draws and never from
  this routine, so they are untouched; `capacity256`'s bleaching collapse is in the median and the
  SGM and is not explained by this defect. The biology workflow shares the routine, so its
  MAP-based products are in the same scope.

- **The reports claimed an outside-prior estimate was possible only under the unrestricted pool.**
  The gradient ascent is unconstrained under either mode: `--pool-mode` bounds the candidate pool,
  not the optimizer's steps. The `CAP256` Evaluation ran `pool_mode: bounded` and still placed 35 %
  of its `mu_r` MAP estimates outside the prior box. The note now says so and states that such an
  estimate is a flow optimum rather than a MAP of the prior-supported posterior. Whether to
  constrain the ascent is a separate decision, not taken here.

### Added

- `VERBOSE` and `SHOW_PROGRESS` knobs on the Detector Experiment stage script, forwarding
  `--verbose` and `--show-progress-steps` so a run can record the per-window optimizer trace: the
  per-step line carrying the current learning rate and the running optimum, and the stop line
  naming `early` or `full-run` and the step reached. Off by default, because they multiply the log
  volume by the window count.

## 0.1.14 - 2026-09-22

Within-recording drift becomes a standard output, computed for every stored point estimate and
read against the posterior's own per-window interval; the direct PSF-width estimator gains an
experimental mode. No canonical stage, parameter role, or preprocessing step changes.

### Added

- **Window drift in the Experiment report.** The Experiment stage now writes a "Within-recording
  drift across windows" table and one `window_drift_<condition>` figure per condition. For every
  point estimate the run stored (MAP; posterior median and SGM under `--summary posterior|both`) the
  table gives the per-recording first-to-last change fitted against the window index, aggregated
  across recordings as median, interquartile range, sign consistency, share of recordings over
  0.3 dex, and the signed-rank p. The figure draws the three estimates as the median across
  recordings at each window over one shared band, the median across recordings of the per-window
  posterior 50 % and 90 % intervals -- the estimates come from the same posterior draws, so the
  band is the posterior's and is drawn once; there are no per-estimate error bars and no standard
  errors. Kernel functions `point_estimate_grids`, `window_medians`, `shared_posterior_bands`,
  `window_drift_rows`, `format_drift_rows` in `temporal_dynamics.py`; figure
  `figure_window_drift` in `visualization_inference.py`.
- **"Point estimates compared" table and `point_estimates_<parameter>` figures.** Both stage
  reports write one table with the three point estimates side by side, columns grouped by
  statistic with MAP, median and SGM consecutive (Evaluation: correlation, MAE, signed bias,
  outside-prior share against the truth; Experiment: per condition, median over windows, IQR
  over windows, outside-prior share). The agreement table's columns are now "MAP vs median",
  "MAP vs SGM", "SGM vs median" with the note stating that each is the median over videos of the
  absolute gap. The Evaluation report gains one figure per parameter with the three estimates
  against the truth in three panels: the estimate's median over equal-count bins of the truth as
  the line and the posterior IQR as one shared band, the construction of the window-drift
  figure, with the truth drawn as a dashed diagonal on top of the density
  (`figure_point_estimates_vs_truth` in `visualization_inference.py`). The Experiment report gains
  the counterpart without truth, `point_estimates_<condition>_<parameter>`: recordings along the
  x axis ordered by their posterior-median value, one point per window, the recording's median of
  each estimate as the line, the posterior IQR as the shared band
  (`figure_point_estimates_by_cell`). Helpers `point_estimates_compared_table`,
  `experiment_estimates_compared_table` in `evaluation.py`.
  The stored 2 s FAB Evaluation and Experiment reports (baseline and `CAP256`) on the Posit tier
  were re-rendered from their arrays with this code; each keeps the report as produced by its
  job as `report_as_produced.md`.
- **Temporal-dynamics utility reads tagged products and all three estimates.** `--artifact-tag`
  selects a tagged estimator's Experiment products (e.g. `CAP256`); the report gains the
  three-estimate drift table and `window_drift_overview_<condition>.png` with the shared bands. The
  per-cell figures and the original drift table remain MAP-based and say so.
- **Direct PSF-width estimator on experimental recordings** (`--experiment`, with
  `--experiment-span-seconds`, `--chunk-step-seconds`, `--cells`, `--max-cells`,
  `--experiment-dir`). Windows every recording exactly as the Experiment stage does
  (`read_cell_chunks`), supplies the section 6.3 acquisition camera values, and writes per-window
  estimates with their nominal 90 % ranges, the pooled distribution table, the same drift table,
  and `window_drift_direct_<condition>` with the estimator's own range as the band, drawn as
  reported (it under-covers on synthetic recordings). No ground truth, so no acceptance verdict.
  Validated end to end on two rendered 4 s recordings.

### Fixed

- **Drift fractions over contributing recordings only.** `temporal_dynamics.drift_statistics` took
  the sign-consistency and material-drift fractions over every cell slot of the grid, so an unused
  cell index, a deselected recording or a failed estimate entered the denominator as a "no drift"
  vote (two rising recordings in slots 1 and 2 of a three-slot grid read 67 % instead of 100 %).
  Both fractions are now taken over the finite fitted changes, the same recordings the table's
  cell count reports. The stored neural products have complete cell indices, so their figures are
  unchanged; the case matters for `--cells`, missing recordings and failed direct estimates.
  Regression test `tests/test_temporal_dynamics_drift.py`, runnable with `python -m pytest` or by
  invoking its functions directly; executed here by direct invocation (pytest is not installed in
  the PC `SRM_AND_SBI_ENVY_V0` environment), both tests passed.

### Changed

- **Report-only rendering leaves the arrays alone.** `write_recovery_outputs` and
  `write_experiment_outputs` take `persist_arrays` (default `True`, the stage behavior); a
  re-render of an existing product passes `False`, so the stored npz is never re-saved (a re-save
  would also drop any field the writer does not know). The re-rendered 2 s FAB reports record the
  source npz md5 and the rendering date in their run note.
- **Agreement notes describe, they do not diagnose.** The Evaluation and Experiment agreement
  notes now state that a large MAP-to-summary gap establishes disagreement and that a density
  spike, an optimizer that stopped short, or another feature of the posterior's shape are
  separate checks. The Experiment figure caption mentions the SGM only when the run stored one.
- **Compared tables count valid videos.** The side-by-side tables use one shared finite-row mask
  per row across the estimates and report that count as `n`.

### Documentation

- `DETECTOR_WORKFLOW.md` §9.7 — results of the capacity test's comparison steps 3 and 4 on the
  shared 25,000-recording EVAL set (JUPITER jobs 1951212, 1951221, 1951236; record
  `..._CAP256_vs_Baseline_Evaluation` on the Posit tier): `capacity256` substantially improves joint
  coverage and removes the baseline's two location biases (joint coverage 0.87 against 0.62 at nominal
  0.90, L-C2ST rejection 0.000 against 0.998), with residual marginal and joint miscalibration
  remaining (largest joint gap 0.065 against the 0.05 reference, marginal 90 % coverage 83 to 86 %,
  SBC still flagging `mu_r`, `mu_pc` and `sigma_pc`, bleaching coverage 87 % to 85 %), recovers
  `mu_r` and `mu_pc` with smaller error and no offset,
  leaves `sigma_pc` and `lambda_rate` unchanged, moves `sigma_r` little, and does not recover
  `prob_photo_bleach` (posterior close to the prior); its MAP separates further from the posterior on
  every parameter. Three-way: the direct estimator remains the source of `sigma_r`; for `mu_r` the
  capacity run's posterior summaries reach the direct estimator's accuracy. Neither estimator is
  adopted; next measurements named (repeat training, 20 s tier). The operating subgroup is described
  as the lower half of the brightness prior in step 3, matching §9.6.
- `..._PSF_Direct_vs_Neural` record — the direct estimator's outside-prior share corrected from 0 %
  to 4 % (`mu_r`) and 2 % (`sigma_r`): edge excursions of at most 0.07 and 0.12 dex.

## 0.1.13 - 2026-09-21

Freezes the acceptance rules for the direct estimators before their corrected evaluation, and
begins the acceptance-mechanics fixes an external review of 0.1.12 required. No canonical stage,
parameter role, or preprocessing step changes.

### Added

- `DETECTOR_WORKFLOW.md` §9.6 — frozen acceptance rules for the direct estimators: four evaluation
  steps in a fixed order (evidence adequacy of the run, operational success, evidence adequacy for
  accuracy and coverage, accuracy and uncertainty) with separate verdicts; the operating subgroup
  (`log10 mu_pc` in [2.00, 2.375)) in which accuracy and coverage must pass; prior-fixed quartile
  boundaries; the accuracy thresholds of 0.1.12 restated with units (`sigma_r` MAE 0.08 in LINEAR
  units, preserving the code's criterion and correcting an inventory that stated dex); three
  bleaching outcomes (failed, valid-but-uninformative, usable) with an observable eligibility
  diagnostic that never uses the true value; a nominal-90 % range per recording whose coverage must
  reach 85 % overall and in the operating subgroup, with width reported separately as informativeness;
  EVAL tasks 0 and 1 declared development data and unscored EVAL tasks the reserved validation set;
  development outputs preserved under `_DEV_<commit>` with full provenance. The three companion notes
  point to §9.6 from their result sections.

- `srm_and_sbi_monomer_dimer_alp/direct_acceptance.py` — the frozen rules of §9.6 as one shared kernel:
  prior-fixed subgroups (the operating subgroup and the quarters of each prior), the four evaluation
  steps with separate verdicts, Wilson intervals on the success fraction and on the measured coverage,
  named per-quantity ranges whose coverage and relative width are reported separately, the bleaching
  usable/uninformative split, and a reporter rendering. Exit status 0 / 1 (any FAIL) / 2 (insufficient
  evidence only). Tested on synthetic cases including the review's three-of-a-hundred case
  (INSUFFICIENT EVIDENCE at every step) and a sixty-percent-success run (FAIL operational).
- Per-recording nominal 90 % ranges from all three direct estimators, each constructed as its
  companion note specifies and validated by coverage: `mu_r` from the standard error of the mean log
  width, `sigma_r` from its own delta-method standard error (`psf_width_population` now returns
  `sigma_r_se`), `lambda_rate` from a bootstrap over the recording's traces against the fixed model
  shapes (`flicker_model_shapes`, `match_shapes`, `flicker_bootstrap_range`), `prob_photo_bleach` from
  the flicker-corrected fit standard error in log10.
- Reason codes on every dropped recording (`no_spots`, `too_few_tracks`, `too_few_traces`,
  `too_few_pairs`, `no_apertures`, `fit_failed`, `nonpositive_estimate`), the full true parameter row,
  the validity mask and the range bounds in every saved array set.
- `DETECTOR_WORKFLOW.md` §9.7 — the capacity test on the multiple-dye baseline: one larger estimator
  (256-dimensional embedding via `start_channels` 16; flow 128 hidden / 8 transforms / 2 blocks /
  dropout 0.1) trained under otherwise identical data, targets, splits, preprocessing, and protocol,
  compared with the baseline on recovery, marginal and joint calibration, widths, and failed-estimate
  rates, overall and in the operating subgroup; parameter counts (0.755 M → 3.308 M) and the activation
  budget (0.85 → 1.70 GiB per video) that motivate 16 videos per rank on 64 ranks (16 nodes × 4 GPUs;
  global batch 1024 unchanged, synchronized batch statistics unchanged). Not yet run.
- `parameterization.InferenceFlow` — the MAF settings (`hidden_features`, `num_transforms`,
  `num_blocks`, `dropout_probability`, `use_batch_norm`, `z_score_x/y`) as an explicit configuration
  whose defaults are the library values the earlier estimators used; the Inference stage passes them
  explicitly and persists them in the estimator's rebuild specification (`maf_args`) for every run,
  preset or not. `NETWORK_PRESETS` (`baseline`, `capacity256`) and `--network-preset` select the
  embedding and flow fields together; the resolved settings are printed and stored in the manifest.
- `Paths.product_label` and `--artifact-tag` (dispatcher knob `ARTIFACT_TAG`, wrapper knob of the
  Inference, Evaluation, Experiment, and Posterior_Calibration stages of both workflows): an optional
  SCREAMING_SNAKE token after the timing label of every PRODUCT (checkpoint, resurrect state,
  estimator, test-loss distribution, backups, debug directory, MAP recovery, posterior calibration,
  experiment report) and of the GPU stages' job names, so a named experiment lives beside the canonical
  run instead of overwriting it; inputs are always read under the plain timing label. The estimator
  manifest records the tag and the preset; the resurrect guard keys on the tagged label.
- `NETWORK_PRESET` dispatcher and Inference-wrapper knob (forwarded as `--network-preset`).
- `FAIL (protocol)` as its own §9.6 verdict for a drop without a reason code, separate from the
  measured success fraction (the operational verdict), after the development flicker run showed the
  two being conflated.
- A progress line every five percent of the queued recordings in the three direct-estimator scripts;
  the development runs were silent for 35 and 80 minutes.
- Per-epoch peak device memory (allocated / reserved, rank 0) on the training log line next to the
  epoch time.
- An explicit statement of the priority between the two workflows (`DETECTOR_WORKFLOW.md` §2,
  `PROJECT_CONTEXT.md` §3): the biology inference is the scientific task with no direct replacement;
  the detector workflow and the direct estimators are supporting tools that constrain imaging well
  enough and propagate what remains uncertain; estimator developments tested on the detector problem
  (§9.7) are controlled benchmarks, not evidence of better biological inference.

### Fixed after external review of 0.1.12

- **One point-estimate convention across the documentation and the reports.** `DETECTOR_WORKFLOW.md`
  §6.8 had made the posterior median "the point estimate throughout" and §6.9 had quoted only its view,
  while §6.11 tabulated only the MAP. All three point estimates (MAP, posterior median, sample geometric
  median) are now tabulated side by side in §6.9, §6.10 and §6.11 and in `PROJECT_CONTEXT.md` §7, with
  conclusions drawn from the set; the Experiment report's MAP table is titled as such, and the agreement
  notes in `experiment_runner.py` and `evaluation_runner.py` no longer tell the reader to substitute one
  estimate for another. The multiple-dye Experiment run predates the SGM output and stored no draw cloud,
  so its SGM does not exist; its posterior median is computed from its stored quantiles.

- **Flicker refinement on an uneven grid.** The parabolic refinement in log-rate used the
  equal-spacing formula on a grid that is not equally spaced in log-rate; it returned 4.3506 for an
  exactly quadratic objective with its minimum at 4.3. The parabola is now fitted on the real
  log-grid coordinates, applied only when it curves upward with its vertex inside the bracket, and
  the report counts unrefined interior estimates and grid-edge minima. Estimates from the
  2026-09-21 development run (`_DEV_2b9c32e`) carry the old bias, of order 0.005 dex near 4.
- **A test could pass while failing on most recordings.** The scorers masked invalid estimates and
  reported the count with no minimum success fraction; three exact estimates and 97 missing ones
  passed every criterion. Replaced by the §9.6 evidence and operational steps.
- **Discarded optimizer flag.** The fluorescence fit's `success` flag is now honored; a finite result
  from an unsuccessful optimization is a `fit_failed` drop, not an accuracy sample.
- **Truth-based bleaching eligibility.** The "identifiable range" was selected from the benchmark at
  the true bleaching value, which no experimental recording has. Replaced by the observable diagnostic
  of the companion note (fitted decay at least 3x the residual scatter; fit standard error on log10 p
  at most 0.25 dex, calibrated on the four self-test scenes), with the rejected recordings' recovery
  reported beside the usable ones.
- **Decay fit collapsed on a fast decay.** Started from the opening level with a slow initial rate,
  the least-squares fit of the p = 0.316 self-test scene converged to a flat line (amplitude 9.5 on an
  offset of -900,000) and reported success; the new decay-visibility diagnostic caught it. The
  amplitude is now initialized to the total drop over the recording and the fit is multi-started over
  five initial rates, keeping the lowest cost; the scene now recovers to -0.03 dex with a
  signal-to-noise of 16.
- **True flicker rate leaked into the bleaching error.** The fluorescence utility passed each
  recording's true `lambda_rate` to the standard-error correction. It now uses the prior center, or a
  measured value via `--lambda-rate`, for every recording alike; the self-test never sees the scene's
  true rate. The kernel also returns the fit's residual scale (`resid_sd`).
- **`sigma_r` units.** The threshold is 0.08 in LINEAR units, as the code has always applied; the
  companion note's comparison against the 0.75 dex log-prior width is withdrawn and the inventory that
  stated dex is corrected. No threshold changed.
- **Physical `Theta_Set` rows.** Stratification converts the stored physical values to log10 before
  applying the prior-fixed boundaries; a first development check that read them as log10 found empty
  strata.

### Self-test findings under the corrected mechanics (development evidence)

- Flicker: the four 6 s scenes now recover with MAE 0.0596 dex (0.0637 under the old refinement),
  correlation 0.9997, every error still positive; the bootstrap 90 % ranges (about 0.11 dex wide)
  cover 1 of 4 truths because the systematic offset is as large as the scatter they represent.
- PSF: `mu_r` ranges cover 8 of 9 scenes, `sigma_r` ranges 4 of 9 — the delta-method sampling error
  is of the size of the estimator's systematic error. Neither range is corrected here: widening
  would be a tuning decision needing justification and validation on the reserved set.
- Fluorescence loss: the observable eligibility classes the two scenes recovered within 0.03 dex
  as usable and the two with errors of 0.3 and 0.6 dex as uninformative; both usable ranges cover.

### Regression run of record (PSF, 0.1.13 mechanics)

- `..._DETECTOR_FAB_2S_50FPS_Direct_PSF_Width` on EVAL tasks 0-1 (2000 recordings, rcl01, 36 min):
  steps 1-3 PASS with reason codes; 4a `FAIL (accuracy)` (operating `mu_r` bias −0.0118 dex, unchanged);
  4b `FAIL (uncertainty)` — nominal 90 % ranges cover 65.9 / 63.3 % overall and 60.0 / 55.7 % in the
  operating subgroup; the sampling-only standard error is ~2.3x too small at every track count; details in
  the folder's `COVERAGE_DIAGNOSIS.md` and `DETECTOR_WORKFLOW.md` §9.6.
- The unchanged two-second flicker rerun is paused (its development run: 1909/2000, bias +0.11 dex,
  `FAIL (accuracy)`); next steps recorded in the flicker note.
- Added a matched comparison on 2,000 multiple-dye synthetic recordings
  (`..._DETECTOR_FAB_2S_50FPS_PSF_Direct_vs_Neural`: arrays, statistics, two-panel figures per parameter)
  showing lower point-estimation error for the direct PSF estimator than for the current neural posterior
  median — substantially for `sigma_r`, modestly for `mu_r`. Recorded the remaining dim-subgroup bias and
  uncertainty undercoverage separately. Clarified that fixed photophysics is an optional biology-input
  construction (§7.2), not the universal workflow contract. No parameter-role change follows from this
  comparison alone.

### Development outputs preserved

- `…_DETECTOR_FAB_2S_50FPS_Direct_PSF_Width_DEV_2b9c32e` and `…_Direct_Flicker_Rate_DEV_2b9c32e` on
  rcl01 hold the 2000-video runs made before the mechanics fixes, each with `PROVENANCE.md` (commit,
  file hashes, commands, settings, recording identifiers, environment) and `DEV_CHECK.md` (the
  stratified development read). They are development evidence, not adoption verdicts; EVAL tasks 0
  and 1 are development data from these runs onward.

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
benchmark standard deviation of an unbiased estimate **exceeds the 0.10 dex threshold by more than an
order of magnitude everywhere at 2 s** (1.26 dex at the top of the prior, thousands of dex at the bottom,
against a 1.5 dex prior width), and falls inside that threshold only in the upper half of the prior at 20 s. This is an
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
