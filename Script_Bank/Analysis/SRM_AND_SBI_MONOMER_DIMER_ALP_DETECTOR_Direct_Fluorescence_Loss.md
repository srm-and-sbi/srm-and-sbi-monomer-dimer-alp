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

In the short-window, shallow-decay limit with known amplitude and offset and independent
constant-variance noise, rate information grows approximately as the **cube** of the duration,
ten times the frames giving about thirty-two times the precision on the rate; with
both fitted, the dependence is steeper where the decay is shallow and shallower where it
completes within the window, so the table below, not the law, carries it. On top of that, the
log-brightness is a stationary Ornstein–Uhlenbeck process with a correlation time of roughly
fifteen frames, so consecutive frames of a fluorescence curve are **not** independent samples of
the decay. The effective count `n(1−rho)/(1+rho)`, a mean-estimation approximation applied here to
a fitted rate, gives about **three** effectively independent samples in a 100-frame recording,
not a hundred.

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

The acceptance threshold is therefore stated at 1000 frames, and which recordings it applies to is
decided by an **observable** eligibility rule, never by the benchmark or the true value
(`DETECTOR_WORKFLOW.md` §9.6): a recording is usable when its fit converged, its fitted total
decay is at least three times the residual scatter, and the fit's own standard error on log10 p is
at most 0.25 dex. The accuracy threshold applies to the usable recordings; the recovery of the
rejected ones is reported beside them, so that selection cannot hide a failure. The report states
the benchmark at the prior center as explanatory context only.

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

Every tier run also writes the recovery figures `figures/recovery_prob_photo_bleach_log10.png` and `..._linear.png` and lists them at the end of
`report.md`: the true value against the inferred one for every attempted recording, by outcome, in the
parameter's log10 prior coordinates and in absolute values, with the identity line, the prior bounds, the
accuracy rule's band, the nominal 90 % ranges of the scored recordings and the error against the true value
with binned medians. `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Estimator_Figures.py <run folder>` redraws
them for a run folder written before the figures existed.

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

**State of the tier and why it is parked (2026-09-25).** Nine of the twenty tasks are rendered (5, 8, 10,
12, 13, 14, 15, 17, 19; 900 recordings, two of them in the development set); the render of tasks 9 to 18 lost
four tasks when their processes were killed, each while rendering one of the tier's largest trajectories.
The cause was located on the PC by replaying the runner's loop on two of those trajectories, without running
any stage: the trajectory reader (`extract_trajectory_poses`) allocates one dense slot per *distinct particle
id* over the whole recording, frames × ids × 3 × species ranks, filled with NaN where a particle is absent,
and ReaDDy assigns a fresh id at every reaction and mode switch, so the id count grows with duration,
receptor count and reaction rates. The tier's largest scene (task 9, simulation 5; about 3,000 particles per
frame) holds 491,645 distinct ids over 1,001 frames, a 66 GiB tensor before the 11 GiB collapsed copy and the
render's own temporaries; a small scene (491 subunits, 6,961 ids) peaks at 4.6 GiB. Reading the file itself
costs under 0.2 GiB. Nothing in the rendered videos is affected; the limitation is memory alone, and it is
quadratic in the duration. Three ways to complete the tier were weighed: one task per 180 GB node with no code
change (about 80 GiB per worst-case render); a lean per-frame gather of subunit positions in place of the
dense tensor, a change to one function that can be tested for exact equality against the current one; and
segmenting the simulation with ReaDDy checkpoints, which would still need the reader, the id-keyed lineage
and the photophysics state to be carried across segments and is the larger change. The tier is parked: at the
working bleaching value of 0.034 per 100-frame interval the benchmark above gives about 0.5 dex even at 20 s,
so the completed tier could confirm the order of magnitude of that value and little more, and the working
imaging vector (`DETECTOR_WORKFLOW.md` §7.6) carries the value as a provisional anchor. If the biology step
turns out to depend on the value, the large-memory-node completion needs no code change; the development
tasks already rendered (5 and 8) allow a development run of this estimator on 200 full-length recordings at
any time. The partial stores of tasks 9, 11, 16 and 18 stayed on JUWELS until the tier was completed, and
were then archived there (below).

**Completed (2026-09-30 to 2026-10-01): the lean reader is implemented, the missing tasks were rendered on rcl01, and the whole tier is on JUWELS.**
The second of the three ways above is in the package: `simulation_rds_support.extract_subunit_positions`
gathers each subunit's host coordinates per frame directly from the trajectory's per-frame observables
through the lineage, `(frames, subunits, 3)`, never building the dense tensor, and the shared DLI runner
of both workflows renders from it through the same rendering calculation as before, the dense reader
staying in place for its other consumers. On a stubbed trajectory with mode switches, association and
dissociation the lean positions equal the dense gather value for value at every documented duration, the
rendered frames are identical under a fixed seed, and the runner's stored videos and labeling records
equal the dense path's for both workflows. On real trajectories pulled to the PC the same holds: the
three 2 s EVAL recordings of task 0 and a 20 s recording of task 9 (459 subunits) read identically through
both readers and render identical frames; the released 0.1.30 and this version write byte-identical video,
theta, SCOPE, nuisance and labeling stores for both workflows on the 2 s recordings under one seed; and on
the largest scene (task 9, simulation 5: 3,011 subunits, 491,645 particle ids, the 65.9 GiB dense tensor)
the lean read peaks at 0.9 GiB and the complete detector runner at 3.8 GiB for one recording in 60 s, where
the dense path was killed at 59 GiB on 2026-09-25. The identity checks were repeated on rcl01 on the same 20 s recordings, the lean read against the dense
gather and identical frames from either source, with the same result, and the eleven missing tasks (0, 1,
2, 3, 4, 6, 7, 9, 11, 16 and 18) were rendered there with the released reader as their trajectories arrived
from JUWELS, task 9 alone first: 100 recordings of 1,000 frames per task, exit status 0 for every task, and
for the first four (9, 0, 1 and 2) 28 to 33 minutes per task and a kernel peak resident set of 4,788,048 to
4,837,656 kB (4.57 to 4.61 GiB) per task process; the per-task numbers of all eleven, a read-only check of
every store and the sha256 identity of the source trajectories with the JUWELS originals are in the data
bank record `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_20S_50FPS_Renderer_Verification/` under `Posit`, with
the scripts, logs and checksums of every check. On 2026-10-01 the record's archive-and-transfer manifest was
executed: the four partial stores (952 files) were moved on JUWELS to `Data_Bank/Superseded_0.1.18/`, their
checksums identical before and after; the eleven tasks' 44 stores (4,455 files) were pushed directly from
rcl01 to JUWELS and read back identical by sha256 through a second login node; the nine complete stores were
not touched. The tier's 80 stores are whole on JUWELS, the tier's single home. Three rendering versions made
them (tasks 5, 8 and 19 by 0.1.13; tasks 10, 12 to 15 and 17 by 0.1.18; the eleven by 0.1.31), and the
record's README lists every change between those versions: all lie outside the detector rendering
calculation, and the stores' layout equality and the rendered-value equality of the reader are recorded as
separate evidence. The frozen estimator then ran on the development tasks for characterization (below); any
modification of it is a separate decision before the reserved verdict is opened. Completing the tier closes
the synthetic validation question; it does not by itself establish a better experimental bleaching value.

**Development characterization on the rendered tasks 5 and 8 (2026-09-25; JUWELS job 14273151, run folder
`..._20S_50FPS_Direct_Fluorescence_Loss_DEV_45177cb`, 200 multiple-dye recordings at 1000 frames, field
observable, 45 s of wall time).** By the frozen rules the run is INSUFFICIENT EVIDENCE (200 attempted against
the 1000 the rules require), so it is a characterization and not a verdict. Outcomes: 108 usable (54 %), 63
valid but uninformative, 29 failed fits. Among the usable recordings the error is MAE 0.121 dex with a bias of
+0.089 dex (the estimator reads high) and a correlation of 0.74 with the truth; in the operating brightness
subgroup MAE 0.143 and bias +0.101. The usability and the bias depend strongly on the true value: for true
log10 values in [−0.875, −0.5) all 57 recordings are usable with bias +0.03 dex and MAE 0.05; in
[−1.25, −0.875) 35 of 50 valid, bias +0.07, MAE 0.11; in [−1.625, −1.25) 15 of 39 valid, bias +0.30, MAE 0.34;
below −1.625 one recording is usable, with an error of +1.1 dex. Around the working value (true log10 in
[−1.75, −1.25)) 15 of 72 recordings are usable and those carry a bias of +0.30 dex with MAE 0.34. The reading for
the experimental measurement is therefore: where the true loss is near the working value, most recordings come
out uninformative, and the recordings the eligibility rule does admit overstate the value by about a factor of
two, a selection effect of the rule that admits only recordings whose decay is visible above the residual
scatter. An experimental result in which most recordings are uninformative is itself consistent with a low
loss rate, and a usable subset near 0.03 to 0.06 per interval is to be read against this +0.3 dex bias rather
than at face value. No correction cycle follows from this; it is the characterization the experimental reading
is judged against.

**Development run on the complete development set (2026-10-01; JUWELS job 14281932, code c909496, run
folder `..._20S_50FPS_Direct_Fluorescence_Loss_DEV_c909496`, tasks 0 to 9, 1,000 multiple-dye recordings at
1000 frames, field observable, 64 s of wall time on one CPU node).** The run meets the evidence requirements of
the frozen rules (1,000 attempted, 475 in the operating brightness subgroup; 523 usable overall and 210 in the
subgroup) and fails the operational, accuracy and uncertainty steps. Valid estimates: 863 (86.3 %, against the
95 % required; 86.5 % in the operating subgroup against 90 %), all 137 drops carrying `fit_failed`. Among the
usable recordings MAE 0.176 dex against the 0.10 dex threshold, bias +0.141 dex and correlation 0.55 (operating
subgroup: 0.152 dex, +0.117 dex, 0.63); among the 340 valid but uninformative recordings MAE 0.51 dex. Coverage
of the nominal 90 % range: 63 % overall and 73 % in the subgroup against the 85 % required, with a median range
width of 0.17 of the prior width. The characterization on 200 recordings above is reproduced with five times
the recordings and localized by prior quarter of the true value: in [−0.875, −0.5) (0.13 to 0.32 per interval)
242 of 244 recordings are usable with MAE 0.053 dex and bias +0.016; in [−1.25, −0.875) (0.056 to 0.13) 186 of
261, MAE 0.122, bias +0.075; in [−1.625, −1.25) (0.024 to 0.056) 65 of 241, MAE 0.38, bias +0.37; in [−2,
−1.625) (0.010 to 0.024) 30 of 254, bias +1.06. In absolute values the usable recordings carry MAE 0.040 and
bias +0.029 per interval, the error concentrated below 0.05, where the inferred values gather near 0.1 to 0.2
while the failed fits sit at zero; the run folder's `figures/recovery_prob_photo_bleach_log10.png` and
`_linear.png` show every recording by outcome. The reading stands: above about 0.1 per interval the estimator
measures the parameter within the threshold; around the working value of 0.03 most recordings are
uninformative and the admitted ones overstate it by a factor of two or more, a selection effect of the
eligibility rule; and its range is too narrow at every level. Under the frozen rules this is a development
FAIL on accuracy and uncertainty, with the operational shortfall beside it. It motivates no correction cycle
here, and the reserved tasks 10 to 19 stay unread.

**Planned experimental measurement (agreed 2026-09-25; each step separately approved).** The working
imaging vector's bleaching row (`DETECTOR_WORKFLOW.md` §7.6) is to be anchored on a measurement made on
the raw recordings rather than on either neural estimator, as a bounded measurement task and not an
estimator-development programme. (1) Observable: background-subtracted total fluorescence over time in a
fixed region of each recording, with no per-frame normalization, reported as the early-to-late fractional
loss between averaging-window centers and as this estimator's fitted effective loss parameter with its fit
diagnostics; the full curve is shown, no initial transient is assumed or discarded beforehand, and any
excluded interval is disclosed. The parameter so chosen approximates the observed fluorescence decline
under the renderer; it is not an independently identified molecular photobleaching probability, and
bleaching is not assumed to be the only contributor to a rendered field-flux change (finite sampling,
flicker, motion and the observation region also enter), so the match is checked by rendering, never
assumed. (2) One development characterization on the rendered development tasks 5 and 8 (200 recordings,
no new rendering): bias, scatter, usable fraction and rejection reasons, including the dim subgroup; a
major failure is reported as a limitation and starts no correction cycle. (3) The experimental input path,
with unchanged estimator arithmetic: the sixty FAB recordings, the externally anchored camera values, an
explicitly recorded background treatment, saved curves, early/late measurements, fit results and
eligibility reasons, and no experimental accuracy verdict; detected-spot counts accompany the result only
if already available, as an optional diagnostic. (4) Per-recording results and their distribution are
reported without silently pooling successful fits or assuming one shared rate; a representative effective
value is chosen only if the curves and diagnostics support one, with the aggregation rule and the plausible
variation recorded; the current 0.034 is a comparison value, not an expected answer or an acceptance
target; if background uncertainty prevents a useful measurement, that is the result. (5) `selection_user`
is built before or alongside this work, the bleaching row stays provisional until the measurement is read,
and the rendered vector is then checked for field-fluorescence decline and apparent spot persistence before
the reference vector is frozen. No new 20 s tier, reserved-set campaign or estimator refinement is part of
this scope.

**Experimental measurement (2026-09-25; rcl01, code 2116c90; record
`..._20S_50FPS_Direct_Fluorescence_Loss_Experiment` on the PC Posit tier).** The sixty MET-FAB recordings, one
1000-frame window each, the field observable in the stored 8-bit domain as designed. Outcomes: 2 failed fits, 18
valid but uninformative, 40 usable; fitted effective loss parameter median 0.28 per interval among the usable
fits; early-to-late fractional loss median 1.21 with a NEGATIVE closing-window flux in 60 % of recordings. Those
numbers do not describe emitter flux. In the stored domain the raw range is compressed 257-fold, the background
sits at five or six levels, the emitter excess is a few levels above it, and the raw background declines by about
9 % over a recording; the per-frame median therefore steps between integer levels during the recording, and
each step moves the whole-field sum by 65,536 levels, more than the entire emitter signal. The design assumption
of the field observable, that a per-frame median absorbs the floor, fails when the median itself is quantized this
coarsely, and the "usable" fits are fits to the stepping floor. As the plan foresaw for this case, the result is
stated as such: **the estimator as designed yields no usable value on these recordings**, and no correction cycle
follows.

**Raw-domain diagnostic (same day; `raw_domain_diagnostic/` in the record, with the script that produced it,
`raw_domain_diag.py`, kept beside its outputs).** Read on the 16-bit raw frames of all sixty recordings, the
opening against the closing 50 frames, median over recordings with the IQR. The background treatment is fixed
and recorded so that the decline stays auditable: the background of a 50-frame window is the median over all of
its pixels, and the emitter excess is the window's mean minus that median (per pixel; multiplied by the pixel
count it is a field sum, and the factor cancels in every ratio); no camera model, gain or offset enters. The
background level falls to 0.91 of its opening value [0.89, 0.93]; the emitter excess to 0.57 [0.53, 0.67]; the
99.9th percentile to 0.71 [0.68, 0.75]; the bright-pixel area (pixels above the window median by five robust
standard deviations, 1.4826 times the median absolute deviation) to 0.54 [0.47, 0.60]. For the time course the
same subtraction is applied frame by frame (per-frame mean minus per-frame median), averaged over 50-frame
windows centered at each time and divided by the opening window: 0.91 of the opening value at 2 s, 0.80 at 5 s,
0.69 at 10 s, 0.62 at 15 s and 0.57 in the closing window (centered at 19.5 s), medians over recordings, so the
fall is fast in the first seconds and slower later. Applying this note's fit to
the raw excess curves (`raw_domain_fits.json`; a diagnostic use of the arithmetic on a different domain, not the
estimator's observable) gives an effective loss parameter of 0.21 per interval (56 of 60 usable; 0.18 with the
first 2 s excluded; 0.17 to 0.19 after dividing the excess by the background level to remove the illumination
decline): the fitted rate describes the fast initial component, which the free offset separates from a slower
remainder. A single-rate renderer at 0.2 per interval would lose 89 % of its dyes in 20 s where the recordings lose
43 % of their excess signal (36 % after dividing the excess by the background level); the model-free rate over
the second half of the recordings, computed per recording between the windows centered at 10 to 11 s and at
19.5 s and aggregated as the median over recordings, is 0.034 to 0.036 per interval (IQR about 0.02 to 0.05;
0.029 to 0.032 after the background division), and the single rate that reproduces the whole decline is 0.054
to 0.057 (over 20 s, or over the 19 s between the opening and closing window centers; 0.044 to 0.046 after the
background division). Three qualifications stay attached: the 9 % background decline shows that the
illumination or the floor is not constant, so part of the excess decline is not emitter loss; emitters leaving
the field and label exchange are not separable from bleaching in a field measure; and the excess is a crude flux
measure with no uncertainty attached beyond the spread across recordings. Across recordings the decline is
coupled to brightness: the fraction of the excess remaining falls as the opening excess rises (Spearman −0.85
over the sixty; 0.69 in the dimmest third, 0.50 in the brightest) and as the opening 99.9th percentile rises
(−0.87), and it tracks each recording's own background decline (+0.89). That is consistent with excitation
intensity differing between recordings, or with a bleachable diffuse component, which a field measure cannot
separate; either way one effective rate is a population compromise.

**What this settles for the working vector.** The decline of fluorescence over a recording is not
single-exponential: a fast component in the first seconds, then a near-constant rate of about 0.035 per interval
that matches the neural baseline's plateau over windows 2 to 9. The bleaching row of `DETECTOR_WORKFLOW.md`
§7.6 selects 0.03 over the fixed 100-frame reference interval (2 s at 50 fps), the baseline's 0.034 rounded
because the parameter is poorly constrained (neural MAE 0.21 dex, a non-exponential decline), with an
estimator-independent leg (the late-phase raw-domain rate), as a late-time effective value rather than a
whole-recording decay match: at 0.03 a single-rate renderer loses 26 % of its dyes over 20 s, against the 43 %
signal loss measured. The variant 0.05 (the whole-recording rate, rounded) tests stronger loss. The two are
working scenarios, not an uncertainty bracket: the brightest recordings decline beyond both, and the initial
transient and the brightness coupling are the features one per-interval probability does not represent.

**What the raw-domain measurement is, and what remains to be checked.** The raw-recording decline measurements
above stand as measured. Their reading as a bleaching proxy rests on the ideal relation for identical,
independent dyes of stationary mean brightness, E[F(t)]/E[F(0)] = (1 − p)^(t/100) with t in frames, which the
grouping of the dyes into monomers and dimers does not change. The measured quantity is instead a finite-field,
camera-rendered, background-subtracted statistic. Particles entering or leaving the field, composition-dependent
motion and overlap, the median background estimate, quantization and clipping, and finite-sample fluctuation can
all move it. The renders of 2026-09-25 that compared it with synthetic recordings used occupancy 1 and
prior-center reaction-diffusion settings and are withdrawn as a validation (`DETECTOR_WORKFLOW.md` §7.6). The
synthetic validation of this statistic, and its interpretation as a bleaching proxy, are checked again under the
corrected rendering configuration.

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
