# Direct fluorescence-loss estimator — method and usage

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Fluorescence_Loss.py`. This
utility estimates the photobleaching probability `prob_photo_bleach` without a network, by
measuring how the total background-subtracted fluorescence of a recording decays over its
length. This note explains what it computes, how to run it, and how to read the result,
without reading the code.

Like the direct PSF-width estimator it is simulation-based inference — validated against known
ground truth on simulated recordings, under an acceptance criterion fixed before the run — and
**direct** in that nothing is learned. It is a special-situation utility, never wired into the
stage dispatcher, and reads only; it writes its report and arrays to the Data_Bank `Posit`
tier. It needs no trained estimator and no GPU.

## What it computes

The model bleaches each dye independently, with a per-frame probability defined so that the
loss accrues to `prob_photo_bleach` over a **fixed** hundred-frame reference window, whatever
the clip length. The surviving fraction is therefore exactly

```
S(t) / S(0) = (1 - prob_photo_bleach) ^ (t / numb_photo_bleach)
```

and the estimator recovers `prob_photo_bleach` from that decay in two steps.

1. **Measure total flux per frame over the whole field.** The frame is summed and its
   background removed by a quantile of the frame's own pixels. Emitters occupy a small
   fraction of a 256 × 256 field, so the median pixel is background, and a per-frame estimate
   also absorbs any drift.

2. **Fit the decay.** `A·(1−p)^(t/N) + B` in linear space, with a free offset `B`, and the
   probability read off the fitted rate.

## Three choices the measurements forced

- **The whole field, not fixed apertures.** Pinning apertures to the spots found in the
  opening frames is the natural first design, and it is wrong here. Emitters diffuse: at a
  typical receptor diffusion coefficient a spot wanders √(4Dt) ≈ 13 px over a 20 s recording,
  far outside a 4 px aperture. The flux inside fixed apertures then decays because the spots
  **walk out of them**, which is a much larger effect than photobleaching and enters the
  fitted rate identically. Measured on rendered 20 s recordings, fixed apertures returned a
  bleaching probability near 0.5 for *every* true value from 0.01 to 0.316 — the parameter was
  not being measured at all. Summing the whole field removes the failure by construction: an
  emitter moving within the field does not change the total. The aperture path is retained
  behind `--observable apertures` for diagnosis only.

- **A per-frame background quantile, not an assumed floor.** The absolute background cannot be
  trusted at the level required: the optical floor contributes about thirty times the
  emitter signal at the MET-FAB density, and the stored 8-bit quantization alone shifts the frame sum by more than
  the entire emitter signal — rounding the background to the nearest of the 257 ADU steps is
  enough. Taking the floor from each frame's own pixels removes it without ever needing its
  absolute value, and a naive estimator using the nominal floor was measured 0.8 dex out at
  the bottom of the prior.

- **A linear-space fit with a free offset, not least squares on the logarithm.** As the curve
  decays into the noise, the log transform pulls the late points down and steepens the slope,
  which overestimates `p` — measured at +0.20 dex at the top of the prior on 1000-frame
  curves. And the offset must be free: the background estimate carries a small residual, and
  forcing the curve through zero converts that residual straight into slope.

**Total fluorescence, not a spot count.** A count of visible spots is not a substitute: a
two-dye spot stays visible when one of its dyes bleaches, so counting spots understates the
loss and does so in a way that depends on the labeling law. Total flux is linear in the number
of surviving dyes, which is what the parameter controls. `DETECTOR_WORKFLOW.md` §9.4 makes the
same point.

## Why the recording must be long

This is the parameter for which recording length is decisive, and the reason is not only the
frame count.

Decay-rate information grows as the **cube** of the duration, so ten times the frames is about
thirty-two times the precision on the rate. On top of that, the brightness is a stationary
Ornstein–Uhlenbeck process with a correlation time of roughly fifteen frames, so consecutive
frames of a fluorescence curve are **not** independent samples of the decay. The effective
count is `n(1−rho)/(1+rho)`: a 100-frame recording carries about **three** effectively
independent samples, not a hundred.

A third effect matters more than either, and is easy to miss. The amplitude and the offset of
the curve are unknown and must be fitted alongside the rate. Where the decay is shallow the
exponential is nearly a straight line over the observed window, so the three parameters are
nearly degenerate and only the **product** of amplitude and rate — the slope — is determined.
The rate alone is then barely constrained, however many frames are collected. A bound that
treats the amplitude and offset as known misses this entirely and is optimistic by an order of
magnitude precisely where the answer matters.

With all three folded in, the bound in dex at the MET-FAB emitter density is:

| recording | p = 0.01 | p = 0.032 | p = 0.10 | p = 0.32 |
|---|---|---|---|---|
| 2 s (100 frames) | 1817 | 178 | 16.5 | 1.26 |
| 20 s (1000 frames) | 6.01 | 0.65 | **0.083** | **0.019** |
| 60 s (3000 frames) | 0.43 | **0.057** | **0.014** | **0.011** |

Bold entries are inside the 0.10 dex acceptance threshold. The prior is 1.5 dex wide, so any
bound above that is no constraint at all. Read plainly: under this reduced model the benchmark
standard deviation exceeds the threshold by more than an order of magnitude everywhere at 2 s
(1.26 dex at the top of the prior against a 1.5 dex prior width, thousands of dex at the bottom),
and falls inside the threshold only in the upper half of the prior at 20 s. The benchmark is approximate — an unbiased decay-rate
estimator using total fluorescence, with an effective-sample-size adjustment for the flicker
correlation — and does not establish a fundamental recovery limit for inference from the full
video; it is the reason this estimator is held to its threshold at 1000 frames and not at 100.

The acceptance threshold is therefore stated at 1000 frames **and** restricted to the range
where the bound permits it. Outside that range the report records the estimate and states that
no threshold applies, rather than passing or failing it.

## Requirements

- The `SRM_AND_SBI_ENVY_V0` environment. No GPU and no trained estimator.
- For `--selftest`, nothing else; the recordings are rendered in memory. Full-length renders
  are memory-hungry (roughly a gigabyte per scene at 150 subunits × 1000 frames).
- For a tier run, read access to a `Video_Set`, its `Theta_Set` and its
  `Nuisance_SCOPE_Theta_Set` for the split and tasks requested.

## How to run

```
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Fluorescence_Loss.py \
    --selftest --selftest-frames 1000
```

`--observable` selects the flux curve: `field` (the default) sums the whole frame and is
immune to emitter motion; `apertures` pins apertures to the opening frames and is retained for
diagnosis only, with `--detect-frames` setting how many frames seed that set. `--dry-run`
resolves the settings and prints what it would read and write. `--workers` sets the process
pool for a tier run. `--expect-videos-per-task` guards against a stale development tier that
carries production filenames while holding only a couple of videos.

## Result and interpretation

**Acceptance is governed by `DETECTOR_WORKFLOW.md` §9.6 (frozen 2026-09-21).** The thresholds below are
the accuracy step of those rules; §9.6 adds the evidence-adequacy, operational-success, operating-subgroup,
and uncertainty-coverage requirements and the order in which they are evaluated, and defines the verdicts
`PASS`, `FAIL (operational | accuracy | uncertainty)` and `INSUFFICIENT EVIDENCE`. This utility evaluates
every step through the shared kernel and reports all verdicts side by side.

The report gives the mean absolute log10 error against the prespecified threshold of **0.10
dex at 1000 frames** (6.7% of the 1.5 dex prior width), together with the bias, the
correlation, and — for context — the information-budget bound at the recording length actually
used.

The report also carries a check that the measured error is not far *below* that bound. A
measurement well under the benchmark is a prompt to look for a defect — ground truth leaking into
the estimate, or a mis-stated bound — before it is read as an unusually good estimator; the
benchmark constrains an unbiased estimator, and a fit with bounded parameters that shrinks toward
the middle of its range can legitimately beat it, so the check prompts rather than fails.

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
| `no_apertures` | aperture observable only: no spots found in the opening frames |
| `fit_failed` | the least-squares optimizer did not report success (previously such fits entered the accuracy calculation if finite) |
| `nonpositive_estimate` | the fit converged to a non-positive probability |

**Three outcomes, and the observable eligibility diagnostic.** A recording that returns a valid
estimate is classed **usable** or **valid but uninformative** by two observable quantities that never
read the true value: the fitted total decay over the recording,
`amplitude · (1 − exp(−rate · n_frames))`, must be at least 3 times the standard deviation of the fit
residuals, and the fit's own standard error on log10 p must be at most 0.25 dex. The accuracy
threshold applies to usable recordings, with the frozen minima of 100 usable overall and 50 in the
operating subgroup; usable and rejected fractions are reported against all attempted recordings, and
the recovery of the rejected recordings is reported beside that of the usable ones so that selection
cannot hide a failure. The two eligibility values were frozen before the full-length validation run and calibrated
only on the four self-test scenes: a 0.15 dex cap rejected the p = 0.1 scene that was recovered to
0.025 dex (its flicker-inflated standard error is 0.195 dex); 0.25 dex accepts the two scenes
recovered within 0.03 dex and rejects the two with errors of 0.3 and 0.6 dex, whose standard errors
are 5 and 6 dex. The visibility floor alone would have accepted the p = 0.0316 scene
(signal-to-noise 4.2, error −0.61 dex), so both criteria are needed. Whether "usable" recordings
then meet the accuracy and coverage requirements is what the validation run establishes. The information budget is reported
as context at the prior center and takes no part in eligibility; the truth-based "identifiable range"
selection used before 0.1.13 is gone.

**The flicker correction uses a supplied rate.** The effective-sample-size correction of the fit's
standard error needs a flicker rate. Before 0.1.13 the utility passed each recording's true rate,
which is unavailable on an experimental recording. It now uses the prior center by default and a
measured value when `--lambda-rate` is given, for every recording alike; the self-test never sees the
scene's true rate.

**Range construction (validated by coverage, not assumed).** The nominal 90 % range is the estimate
times `10^(± 1.645 · se_log10)`, with `se_log10 = prob_se / (p · ln 10)` and `prob_se` the
flicker-corrected standard error propagated from the rate. Its coverage against the truth, overall
and in the operating subgroup, is what validates it; its median width is reported against the
1.5 dex prior width.

## Development runs on the 20 s tier

**Every run is kept, and a tier run is held to its purpose.** A tier run must declare why it reads its
tasks: `--purpose development` or `--purpose verdict`; there is no default. Before it reads a recording,
a development run on a tier with a declared split refuses every task outside the development set, and a
verdict run requires exactly the reserved tasks of the tier, in full, and no earlier verdict folder of
this estimator on that tier (`DETECTOR_WORKFLOW.md` §9.6, declared split). The run folder carries the
purpose, `_DEV` or `_VERDICT`, and `--run-suffix` appends to it, for example the commit:
`..._Direct_Fluorescence_Loss_DEV_<commit>`. A run never reuses a folder; an existing one is refused
before anything is read, so an earlier run is never overwritten. Every run folder holds
`provenance.json`: the command line, host, Slurm job, package and library versions, the declared commit,
the purpose record, and the implementation hash of the direct-estimator files at startup and at write.
When the two hashes differ, the run is invalid for acceptance: its arrays are kept as diagnostics, the
verdict table marks it `INVALID`, and it exits with status 3. The arrays carry `task` and `index` for
every recording. On JUWELS,
`Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Direct_Estimator.sh` runs the utility on one
whole CPU node (`ESTIMATOR=`, `TASKS=`, `PURPOSE=`, `RUN_SUFFIX=`, `CODE_COMMIT=`).

**The 20 s tier and its split.** The MET-FAB 20 s EVAL tier holds 20 tasks of 100 recordings at 1000
frames. Tasks 0 to 9 are development data; tasks 10 to 19 are reserved for this estimator's verdict at
1000 frames (§9.6, declared split). The tier holds 100 recordings per task, so a run passes
`--expect-videos-per-task 100` (the default of 1000 matches the 2 s tier):

```
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Fluorescence_Loss.py \
    --condition FAB --total-time-seconds 20.0 --tasks 0 1 2 3 4 5 6 7 8 9 \
    --expect-videos-per-task 100 --workers 48 --purpose development --run-suffix <commit>
```

The threshold and the eligibility diagnostic stay frozen. A development run shows how the estimator
behaves on full-length multiple-dye recordings; any change it motivates is made on the development
tasks and judged once on the reserved ones.

## Essential notes

- **A passing estimator is a candidate, not a decision.** `DETECTOR_WORKFLOW.md` §9.4 gates
  removal from the inferred block on more than accuracy; in particular a quantity that leaves
  must enter the downstream stage as a nuisance with an explicit range, never as a point value.
- **The camera block is supplied, never fitted**, from the recording's own `Nuisance_SCOPE`
  record.
- **Works in the stored 8-bit domain**, the same pixels the neural estimator reads.
- **Emitters leaving the FIELD are not separable from bleaching.** The field observable is
  immune to motion within the frame, but an emitter crossing the boundary is a genuine loss of
  flux that no per-frame background estimate can distinguish from a bleaching event. On a
  trajectory tier that turnover is part of the measured error and is reported as such; the
  information budget, which omits it, is correspondingly optimistic.

## References

- `DETECTOR_WORKFLOW.md` §6.2 (the inferred imaging block and its priors), §6.5 (the flicker
  model whose correlation dominates this budget), §9.4 (the acquisition-information contract,
  which requires this estimator to be tested on full-length simulations) and §9.5 (the
  information budget).
- Hirsch, M., Wareham, R.J., Martin-Fernandez, M.L., Hobson, M.P., Rolfe, D.J. (2013). A
  stochastic model for electron multiplication charge-coupled devices — from theory to
  practice. *PLoS ONE* 8(1):e53671.
- Ha, T., Tinnefeld, P. (2012). Photophysics of Fluorescent Probes for Single-Molecule
  Biophysics and Super-Resolution Imaging. *Annual Review of Physical Chemistry* 63:595–617.
