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
standard deviation exceeds the prior width everywhere at 2 s, and falls inside the threshold only
in the upper half of the prior at 20 s. The benchmark is approximate — an unbiased decay-rate
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

The report gives the mean absolute log10 error against the prespecified threshold of **0.10
dex at 1000 frames** (6.7% of the 1.5 dex prior width), together with the bias, the
correlation, and — for context — the information-budget bound at the recording length actually
used.

The report also carries a check that the measured error is not far *below* that bound. A
measurement well under the benchmark is a prompt to look for a defect — ground truth leaking into
the estimate, or a mis-stated bound — before it is read as an unusually good estimator; the
benchmark constrains an unbiased estimator, and a fit with bounded parameters that shrinks toward
the middle of its range can legitimately beat it, so the check prompts rather than fails.

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
