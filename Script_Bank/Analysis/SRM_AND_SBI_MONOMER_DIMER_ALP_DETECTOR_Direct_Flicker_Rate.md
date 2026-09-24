# Direct flicker-rate estimator — method and usage

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Rate.py`. This utility
estimates the brightness-flicker rate `lambda_rate` from video alone, without a network. This
note explains what it computes, how to run it, and how to read the result, without reading the
code.

It completes the set of direct estimators alongside the PSF-width and fluorescence-loss ones.
Like them it is simulation-based inference — validated against known ground truth on simulated
recordings, under an acceptance criterion fixed before the run — and **direct** in that nothing
is learned. It is a special-situation utility, never wired into the stage dispatcher, and reads
only. No trained estimator, no GPU.

## Relation to the derivation of record

`Flicker_Rate_Derivation` fixed the `lambda_rate` **prior** from ThunderSTORM localization
tables, giving 5.1 (MET-Fab) and 4.7 (MET-InlB) against a locked log-uniform prior of [1, 10].
That utility consumes ThunderSTORM's fitted per-localization photon counts, so it is
unavailable under the acquisition-information premise of `DETECTOR_WORKFLOW.md` §9.4, which
assumes those outputs are not to hand.

This utility measures the same quantity from the frames themselves. The method is
**deliberately identical** — per-trace log, per-trace linear detrend, gap-aware pooled
autocorrelation, normalization at lag one, and a shape match against a matched-length
Ornstein–Uhlenbeck model arm. Keeping the technique the same is the point: it makes the
video-derived number and the table-derived number directly comparable, so a disagreement
between them carries information instead of being an artifact of doing it differently.

The functions are reimplemented in the package kernel rather than imported from that script,
which is left untouched so the recorded 5.1/4.7 result keeps its provenance.

## What it computes

1. **Measure.** The same detection-and-fit pass the PSF-width estimator runs. Its fitted
   per-spot amplitude is the total spot signal with the background already accounted for, so
   it is the photometric series the autocorrelation needs, obtained for free.
2. **Link.** Detections are joined into per-subunit tracks, giving one intensity series per
   emitter. The frame stride is pinned to 1, because the autocorrelation is indexed in frames
   and a stride would silently rescale every lag.
3. **Detrend.** Each series is logged and linearly detrended against its own frame index,
   which removes photobleaching and the per-emitter mean together — both are multiplicative in
   intensity and hence additive in the log. Non-positive samples become **gaps** rather than
   discarding the whole track: background-subtracted video photometry goes negative in dim
   frames, and dropping those tracks would select against faint emitters.
4. **Match.** The pooled autocorrelation is normalized at **lag one**, not lag zero — white
   measurement noise is uncorrelated between frames and so lives entirely at lag zero, and
   discarding it is what makes the statistic insensitive to per-frame photometry error. The
   resulting shape is matched against model shapes at a grid of `lambda_rate`, each simulated
   with traces cut to the *observed* span distribution and detrended identically. That
   matched-length arm is what nulls the finite-track detrending bias, which is why the bare
   `1/tau_corr` reads high.

## Why the model arm is single-dye, and what that costs

In the single-dye case the method is **exactly free of the brightness parameters**: `mu_pc`
shifts ln-brightness additively and `sigma_pc` scales it linearly, so both vanish under the
detrend and the normalization. That is a property worth protecting.

Under `FAB_POISSON` a labeled subunit carries about two dyes whose photons sum *before* the
logarithm, which breaks the exactness — `ln(sum of k lognormals)` is not an Ornstein–Uhlenbeck
process, and its normalized autocorrelation acquires a dependence on `sigma_pc`.

Modeling that multiplicity would remove a small bias at the price of making this estimator
depend on `sigma_pc`, which is itself one of the six parameters under inference. In a campaign
whose purpose is to decide which parameters can be pinned **independently**, that is circular,
and the small bias is the better trade. So it is left in and **bounded** instead:

| σ_pc | shape shift, 1 dye → 2 | in λ terms |
|---|---|---|
| 0.178 | 0.0040 | ~0.005 dex |
| 0.42 (center) | 0.0077 | ~0.009 dex |
| 1.0 (top), 3 dyes | ~0.040 | ~0.05 dex |

against roughly 0.035 in shape for a ten percent change in `lambda_rate`. The report states
that band, computed across the whole `sigma_pc` prior, **without ever needing a `sigma_pc`
value**. That is the difference between an estimator that reports its own systematic and one
that depends on an answer it does not have.

## Recording length matters, and 2 s is the hard case

The per-track detrend removes low-frequency power, so short tracks retain less of the
autocorrelation's shape. Measured across the model grid:

| regime | total shape travel, λ=1 → 10 | hardest neighboring gap |
|---|---|---|
| MET tables (spans 40–400) | 0.584 | 0.108 |
| spans 40–300 | 0.546 | 0.100 |
| **2 s tier (spans 40–100)** | **0.337** | **0.042** (λ=1→2) |

A 2 s recording keeps about 58% of the discrimination the MET tables had, and the low-λ end is
hardest: the step from λ=1 to λ=2 is a factor of two — 0.3 dex — separated by only 0.042 in
shape. Expect the estimator to do better at the top of the prior than the bottom, and read the
reported error with the recording length in mind.

## What the acceptance run measured

The four scenes are single-dye, 6 s, with the other imaging parameters at their prior centers and
bleaching disabled. This is a self-test of the measurement chain, not a validation across the
intended operating conditions: the estimator has not been validated on multiple-dye recordings
spanning brightness, multiplicity, bleaching, and track length, and until it is, its result on
experimental recordings is a cross-check, not an arbiter (`DETECTOR_WORKFLOW.md` §6.11).

On four 6 s scenes spanning the prior, the estimator passed both criteria — correlation 0.9997
against the 0.80 threshold, mean absolute error 0.0596 dex against 0.08 — with these points
(re-run 2026-09-21 under the exact log-grid refinement; the 2026-09-18 run under the
equal-spacing formula gave 1.839, 3.614, 5.310, 9.173 and a mean absolute error of 0.0637 dex):

| true λ | estimate | error (dex) | bootstrap 90 % range | usable traces |
|---|---|---|---|---|
| 1.5 | 1.776 | +0.0734 | [1.51, 2.03] | 257 |
| 3.0 | 3.495 | +0.0664 | [3.06, 4.16] | 292 |
| 5.0 | 5.533 | +0.0440 | [4.96, 6.23] | 290 |
| 8.0 | 9.069 | +0.0545 | [8.15, 10.17] | 294 |

Three features are worth stating rather than leaving to be read off the table. **The error is
almost entirely systematic**: the mean absolute error and the mean signed error are the same
number to four decimals, because every point runs fast, and the correlation of 0.9997 says the
ordering is essentially perfect. **The bootstrap range covers only one of the four truths**: its
width, about 0.11 dex, is the trace-to-trace scatter of the measurement, and the systematic offset
of about 0.06 dex is of the same size, so a range built from scatter alone sits above the truth.
Correcting or widening it would be a tuning decision that needs a justification and a
validation on the reserved set; until then the range is reported as it is and its coverage on
the full-scale runs is the measurement of record. The estimator ranks flicker rates far better than it places
them. **Dye multiplicity does not explain it** — the multiplicity band is 0.009 dex at the
center of the `sigma_pc` prior, about a sixth of the observed offset — so the remainder is a
property of the measurement chain. The mismatch study below, on two-second multiple-dye scenes,
places most of the fast bias in detection gaps and photometry noise, with a small share in linking.

The predicted low-λ-worst pattern holds at the ends (0.0734 at the bottom against 0.0545 at the
top) but not monotonically — the best point is λ=5, not λ=8. Four scenes is too few to resolve
the shape of the bias, and nothing here licenses correcting for it; it is reported so that a
user reads the estimate as running fast by roughly 0.06 dex rather than as scatter.

## Requirements

- The `SRM_AND_SBI_ENVY_V0` environment. No GPU and no trained estimator.
- For `--selftest`, nothing else; recordings are rendered in memory.
- For a tier run, read access to a `Video_Set`, its `Theta_Set` and its
  `Nuisance_SCOPE_Theta_Set` for the split and tasks requested.

## How to run

```
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Rate.py \
    --condition FAB --total-time-seconds 2.0 --tasks 0 --max-videos 100 --workers 16 \
    --purpose development --run-suffix <commit>
```

`--min-track-length` sets the shortest usable trace, defaulting to the 40 frames the derivation
of record required. `--model-traces` sizes the Ornstein–Uhlenbeck arm per grid point; the arm is
vectorized by trace length, costing about 1.5 s per grid point, so roughly 16 s per recording
for the ten-point grid. `--dry-run` resolves settings and prints what it would read and write.

## Result and interpretation

**Acceptance is governed by `DETECTOR_WORKFLOW.md` §9.6 (frozen 2026-09-21).** The thresholds below are
the accuracy step of those rules; §9.6 adds the evidence-adequacy, operational-success, operating-subgroup,
and uncertainty-coverage requirements and the order in which they are evaluated, and defines the verdicts
`PASS`, `FAIL (operational | accuracy | uncertainty)` and `INSUFFICIENT EVIDENCE`. This utility evaluates
every step through the shared kernel and reports all verdicts side by side.

The report gives the Pearson correlation and mean absolute log10 error against the prespecified
thresholds — correlation ≥ 0.80 and MAE ≤ 0.08 dex, the latter being 8% of the 1.0 dex prior
width — together with the multiplicity systematic as a band. That band sits inside the threshold
at the center of the `sigma_pc` prior and consumes most of it at the top corner, which the
report says explicitly rather than leaving it to be inferred.

## Acceptance mechanics

The utility evaluates the frozen rules of `DETECTOR_WORKFLOW.md` §9.6 through the shared kernel
`srm_and_sbi_monomer_dimer_alp.direct_acceptance`, which every direct estimator uses so that a rule
cannot drift between them. The report carries the five verdicts side by side (evidence adequacy of
the run, operational success, evidence adequacy for accuracy, accuracy, uncertainty coverage), the
prior-fixed quartile table, and the dropped recordings by reason code. The exit status is 0 when
nothing failed, 1 on any `FAIL` verdict, 2 when the only shortfall is insufficient evidence, and 3 when
the implementation changed during the run, which makes the run invalid for acceptance. Python also
exits 1 on an uncaught exception and the argument parser 2 on an argument error or a refusal, so 1 and
2 are not verdicts alone; the log says which. A `--selftest` reaches no verdict: its few scenes are
reported as `SELFTEST (informational)`.

**Reason codes.** Every attempted recording that returns no valid estimate carries one of the codes
listed below; a dropped recording without a code fails the run itself. The saved arrays hold the
full true parameter row (`theta`, physical units, six columns in the detector's order), the
validity mask, the reason codes, and the per-recording range bounds, so any stratum can be
recomputed from the arrays without rerunning the estimator.

| reason code | meaning |
|---|---|
| `no_spots` | nothing detected above threshold |
| `too_few_traces` | fewer than five intensity traces reach the minimum track length |
| `too_few_pairs` | the pooled autocorrelation has too few lag-one pairs |

**Refinement on the real grid.** The grid `[1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 14]` is uneven in
log-rate, so the parabolic refinement between the bracketing points is fitted on the actual
log-grid coordinates. An equal-spacing formula used earlier returned 4.3506 for an exactly
quadratic objective with its minimum at 4.3; the corrected fit returns 4.3. The refinement is applied
only when the parabola curves upward and its vertex lies inside the bracket; otherwise the grid
minimum is kept and counted under "interior estimates left unrefined". Grid minima at 1 or 14 are
counted as "estimates at a grid edge" and are never refined.

**Range construction (validated by coverage, not assumed).** The model-arm shapes depend on the
recording only through its span distribution, so they are computed once per recording and reused.
The traces are then resampled with replacement (`--n-boot`, default 40), the pooled data shape is
recomputed for each resample and matched against those same shapes, and the 5th to 95th percentile
of the resampled rates is the nominal 90 % range. It carries the trace-to-trace scatter of the
measurement, detection, photometry, linking and gaps included, since those shaped the traces. It
does not carry the single-dye model approximation, the detrending approximation, or the fixed-span
approximation, and the fixed multiplicity band of 0.009 to 0.05 dex is not an uncertainty for the
detection-and-tracking chain either. Whether the range is nevertheless reliable is what its coverage
against the truth, overall and in the operating subgroup, measures.

## Development outcome (2026-09-21, code 2b9c32e, EVAL tasks 0-1)

1909 of 2000 two-second MET-FAB recordings estimated; the 91 drops all had fewer than 29 usable traces
and 73 % of them lie in the dimmest brightness quarter (success 95.5 % overall, 91.6 % in the operating
subgroup, both within the rules; the drops carried no reason code, a reporting failure, `FAIL (protocol)`
under §9.6, not a measurement failure). Correlation 0.847 meets its threshold. MAE 0.144 dex and bias
+0.110 dex fail the 0.08 dex bound; operating-subgroup bias +0.142 dex, about 39 % overestimation. The
bias depends on the true rate (+0.26 dex near 1 per second, +0.04 dex above 5 per second) and on
brightness (+0.16 dex dim, +0.06 dex bright). The signed errors are the evidence; within-quarter
correlations shrink with the truth range by construction. The 6 s single-dye self-test at the prior
center (0.06 dex) probed none of these conditions, and the exact-parabola correction accounts for about
0.005 dex. Outputs preserved under `..._Direct_Flicker_Rate_DEV_2b9c32e` with `PROVENANCE.md` and
`DEV_CHECK.md`.

Decisions: the unchanged two-second rerun is paused. The model arm matches track spans and detrends but
does not represent gaps, detection and linking selection, measurement noise, dye multiplicity, or
bleaching in the observed traces; which omission drives the bias was not isolated by this run. The
mismatch study that followed renders its own scenes with known truth instead of reading a development
subset, and rebuilds the observed traces one step at a time; its outcome is recorded below. Still open: a
comparison against estimation over the full experimental recording with one rate and its uncertainty
propagated to the windows, contingent on validating that the rate is constant over a recording at that
duration. The comparator is the multiple-dye neural posterior estimator on the same recordings (bias, MAE,
uncertainty, and correlation together), not the one-dye estimator.

## Development runs and the mismatch study

**Every run is kept, and a tier run is held to its purpose.** A tier run must declare why it reads its
tasks: `--purpose development` or `--purpose verdict`; there is no default. Before it reads a recording,
a development run on a tier with a declared split refuses every task outside the development set, and a
verdict run requires exactly the reserved tasks of the tier, in full, and no earlier verdict folder of
this estimator on that tier (`DETECTOR_WORKFLOW.md` §9.6, declared split). The run folder carries the
purpose, `_DEV` or `_VERDICT`, and `--run-suffix` appends to it, for example the commit:
`..._Direct_Flicker_Rate_DEV_<commit>`. A run never reuses a folder; an existing one is refused before
anything is read, so an earlier run is never overwritten. Every run folder holds `provenance.json`: the
command line, host, Slurm job, package and library versions, the declared commit, the purpose record,
and the implementation hash of the direct-estimator files at startup and at write. When the two hashes
differ, the run is invalid for acceptance: its arrays are kept as diagnostics, the verdict table marks
it `INVALID`, and it exits with status 3. The arrays carry `task` and `index` for every recording. On
JUWELS, `Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Direct_Estimator.sh` runs the
utility on one whole CPU node (`ESTIMATOR=`, `TASKS=`, `PURPOSE=`, `RUN_SUFFIX=`, `CODE_COMMIT=`).

**Where along the observation chain the bias arises.**
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Mismatch.py` (companion note beside it) renders
scenes with known truth and applies this estimator's shape match to seven versions of the traces: each
dye's true photons over the whole recording, each spot's summed true photons, those over the detected
span, those at the detected frames only, the fitted amplitudes of the matched detections grouped by
true identity, the same detections grouped by the production linker, and the production traces with
every accepted detection. Each version changes one thing against the one before it; the two fitted
groupings read the same detections, so their contrast isolates linking, and the last contrast isolates
the detections outside the matched set. A contrast suggests a contribution of that step under these
scenes and this ordering, not a causal decomposition; every level reports how many scenes produced an
estimate and why the others did not. The harness reads no EVAL task; the development tasks confirm what
it finds.

**Outcome of the default grid (2026-09-24, code 9a9076a, rcl01).** Twenty-four two-second scenes under the
bare FAB dye-count law, true rates 1.25, 2, 4 and 8 per second at 120, 240 and 480 photons per dye, every
level producing an estimate in every scene. The production level reproduces the development tier's pattern:
+0.094 dex over all scenes, +0.237 at 1.25 per second against +0.035 at 8, and +0.113 at 120 photons against
+0.088 at 480. The paired contrasts between adjacent levels, mean and median in dex, are: dye multiplicity
−0.004 and +0.002; span selection +0.015 and +0.007; detection gaps +0.025 and +0.032; photometry +0.047 and
+0.010; linking alone +0.008 and +0.014; detections outside the matched set +0.006 and +0.005. The photometry
contrast is concentrated at the slowest rate (+0.17 at 1.25 per second) and the dimmest brightness (+0.077
at 120 photons) and is near zero from 4 per second up; the detection-gap contrast is spread over the rates
(−0.003, +0.045, +0.035 and +0.024 from 1.25 to 8 per second). Under these scenes and this ordering, the
fast bias enters where the true photons give way to the measurement, in the detection gaps and the
photometry, and not in the dye count or the linker. Both steps act on the observed traces and have
observable anchors (next section); whether a correction built on them closes the bias is for the harness
to show. The single-dye and bleaching variants have not been run, so nothing here speaks to bleaching, and
the multiplicity contrast holds for these scenes; the harness note carries the full tables.

**Standing of this estimator (decision of 2026-09-24).** The flicker-rate estimator is a biased cross-check
of the working value of `lambda_rate`, not its source. The development tier placed its bias at +0.11 dex
overall and up to +0.26 dex at the slowest rates, the mismatch study located it in the detection gaps and
the photometry noise of the observation chain, and further refinement is unlikely to change the immediate
decision about the imaging values. Development is closed here: the harness keeps its per-level shapes and
pooled sums for the record, and no correction-and-validation cycle follows. The working value of
`lambda_rate` comes from the calibration results and the derivation of record, with this estimator's
readings and their known bias recorded beside it. The classification is specific to this estimator: the
direct PSF recovery, `sigma_r` above all, remains a candidate source for the working imaging vector. An
imaging parameter is reopened only if its plausible
uncertainty changes a biological conclusion materially, not because another diagnostic can be imagined.

## Every input is observable on an experimental recording

The estimator exists to run on experimental recordings, which have no ground truth. So every quantity it
uses is one of three kinds: a setting of the detector, a camera value known from the acquisition, or a
number computed from the recording itself. The seven levels of the mismatch study use truth (each dye's
photons, each detection's subunit); they are development scaffolding, and only the production chain,
detect, fit, link, detrend, pool and match, ever runs on an experimental recording. A correction of the
fast bias is held to the rule §9.6 sets for the PSF estimator's correction: it may use only quantities
available on experimental recordings, and it is validated once on the reserved tasks.

| input | kind | where it comes from | kernel function |
|---|---|---|---|
| the frames, stored 8-bit and converted to ADU by the fixed global map | recording | the video | `levels_to_adu` |
| detection threshold of four filtered background standard deviations, matched-filter width, isolation radius, border margin, fit patch | detector settings | `--n-sigma`, `--half-px` and the kernel defaults | `detect_spots`, `measure_spot_widths` |
| background level and background noise, from the five SCOPE camera values | camera values | the acquisition values of `DETECTOR_WORKFLOW.md` §6.3, on which the SCOPE box is centered: gain ratio, baseline, read noise and quantum efficiency are acquisition settings and a datasheet value; the optical background is a measured quantity, obtained from the frames conditional on those constants (§6.4). A tier run reads each recording's drawn values from its `Nuisance_SCOPE` record; an experimental run uses the acquisition values. The box is a tight anchor around them | `background_mean_adu`, `background_sigma_adu` |
| per-detection fitted amplitude, the background-subtracted total spot signal, and its fit covariance | recording | the least-squares fit of each detection | `fit_spot_width` |
| traces: the frames each spot was fitted in, their spans and gaps | recording | linking of the detections | `link_spot_tracks`, `spot_intensity_traces` |
| pooled autocorrelation, lag 0 included | recording | the detrended log-amplitudes; lag 0 is accumulated and then discarded by the lag-1 normalization | `flicker_pooled_acf`, `flicker_data_shape` |
| the model arm | nothing from the recording beyond the spans | simulated single-dye traces at the grid rates, cut to the observed spans | `flicker_model_shape` |

Not needed, and not used: the true rate; `mu_pc` and `sigma_pc`, which cancel under the detrend and the
normalization in the single-dye model; the dye count of a spot, whose effect is bounded above and was
near zero in the mismatch study's scenes; the bleaching rate, which the per-trace detrend is designed to
remove (the harness's bleaching variant has not been run); and which detections belong to which subunit,
which only the harness knows.

**What a correction may use.** The two steps the mismatch study points to, detection gaps and photometry
noise, act on the traces after the true photons are formed. Each has observable anchors, with the
qualifications that make the correction a modeling step rather than a lookup:

- *Detection*: the detector thresholds the matched-filter **peak** of a spot, while the fit returns its
  **integrated** signal; the two are related through the spot's width and the filter, both known (the
  width from the fit). A censoring model therefore applies the detector's own rule to a simulated spot
  rather than a ratio of two numbers, and the synthetic chain loses frames through more than the
  threshold: the isolation rule that drops the dimmer of two close spots, the border margin, fits
  rejected on their width or their error, bleaching, and the linker's gap limit. The detected amplitudes
  are themselves selected by detection, so the missing part of the brightness distribution is inferred
  through the model, not read off.
- *Photometry noise*: the amplitude's standard error from the fit covariance, which the fit computes
  (with the known camera values) and returns today only for the width, gives the size of the noise, not
  its correlation in time. The lag-0 excess of the pooled autocorrelation, which the accumulator already
  holds, mixes photometry noise with brightness fluctuations faster than a frame and separates them only
  under an assumption.
- *Frame sets*: the spans and gaps of the traces, already used.

A correction in the data arm, normalizing at a later lag or matching from lag 3 on, needs no new input at
all; its success cannot be predicted from a white-noise argument, because the per-trace detrend makes even
white photometry noise correlated across the lags, so it is tested on the harness, not assumed. A
correction in the model arm, simulated traces passed through the detector's loss rules and carrying noise
of the measured size, is a modeling extension whose adequacy the harness has to show; its inputs are
observable, which is necessary and not sufficient. None of the inputs is an inferred parameter, so the
circularity that rules out modeling multiplicity through `sigma_pc` does not arise.

**Why gaps are modeled, not counted.** Frames go missing from a trace for several reasons: in the
synthetic chain a dim frame falls below the threshold, a spot loses the isolation rule to a brighter
neighbor, sits inside the border margin, has its fit rejected, bleaches, or exceeds the linker's gap
limit; on an experimental recording a dye may also enter a dark state and a probe may unbind. The pooled
accumulator skips every pair that touches a gap, which avoids treating gaps as zeros but does not make
the shape unbiased: that depends on why the frames are missing, how the traces are selected, and how each
incomplete trace is detrended. Brightness-driven losses are the ones that remove dim frames selectively,
and a model arm has to reproduce them through the detector's rules; a raw gap fraction cannot tell them
apart from the others. An excess of observed gaps over what the modeled observation process predicts then
indicates a mismatch with that process on an experimental recording; it does not by itself identify dark
states or probe residence.

**Consistency checks on an experimental recording.** Without truth, four checks remain, each with its
limit. The corrected model shape should fit the pooled shape over all twelve lags rather than pass through
one point; a good fit is consistent with the rate, it does not establish it. Bright and dim spots should
give the same rate, since the bias on the tier depends on brightness; a disagreement can be residual bias
or physical heterogeneity between the spots. Estimating from every second frame of the same traces should
give the same rate in seconds; the subsampled traces are detrended and matched against a model arm at the
doubled frame interval, and detection and linking are not redone, so this tests the shape match and the
detrend rather than the whole chain. The value is compared with the derivation of record from the
localization tables (5.1 and 4.7 per second) and with the neural posterior's rate marginal on the same
windows; both are references that share assumptions with this estimator, not ground truth. Passing all
four bounds the failure modes that matter; it proves no value. An experiment mode of this utility does not
exist yet; when it is written, its report carries these checks in place of a verdict.

## Essential notes

- **A passing estimator is a candidate, not a decision.** §9.4 gates removal from the inferred
  block on more than accuracy; a quantity that leaves must enter the downstream stage as a
  nuisance with an explicit range, never a point value.
- **`lambda_rate` is photophysical** and should not depend on the biological condition. The
  derivation of record found 5.1 versus 4.7 across conditions, agreeing within cell-to-cell
  scatter. A video-derived estimate that differs markedly between conditions would be evidence
  about the observation model, not about biology.
- **Works in the stored 8-bit domain**, the same pixels the neural estimator reads.
- **The lag-one normalization is not optional.** Normalizing at lag zero instead would leave the
  per-frame photometry noise in the statistic and make the result incomparable with the 5.1/4.7
  values of record.

## References

- `DETECTOR_WORKFLOW.md` §6.5 (the flicker model and the derivation), §6.2 (the prior), §9.4
  (the acquisition-information contract) and §9.5 (the information budget).
- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Flicker_Rate_Derivation.md` — the localization-table
  derivation this mirrors, and the source of the 5.1/4.7 values of record.
- Uhlenbeck, G.E., Ornstein, L.S. (1930). On the theory of the Brownian motion. *Physical
  Review* 36(5):823–841.
- Dempsey, G.T., Vaughan, J.C., Chen, K.H., Bates, M., Zhuang, X. (2011). Evaluation of
  fluorophores for optimal performance in localization-based super-resolution imaging. *Nature
  Methods* 8(12):1027–1036.
- Ha, T., Tinnefeld, P. (2012). Photophysics of Fluorescent Probes for Single-Molecule
  Biophysics and Super-Resolution Imaging. *Annual Review of Physical Chemistry* 63:595–617.
