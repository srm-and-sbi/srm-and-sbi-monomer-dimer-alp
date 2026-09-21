# Direct PSF-width estimator — method and usage

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_PSF_Width.py`. The detector
workflow infers six imaging parameters with an amortized neural posterior. This utility
estimates two of them — the PSF-width population parameters `mu_r` and `sigma_r` — without a
network, by fitting the renderer's own spot shape to individual emitters and summarizing the
fitted widths. This note explains what it computes, how to run it, and how to read the
result, without reading the code.

Both approaches are simulation-based inference: each is validated against known ground truth
on simulated recordings, under acceptance criteria fixed before the run. The difference is
that this one is **direct** — nothing is learned, and every number is read off a closed-form
property of the forward model. The comparison it supports is *neural versus direct*, never
"SBI versus not".

This is a special-situation utility, not one of the canonical pipeline stages. It lives in
`Script_Bank/Analysis`, is never wired into the stage dispatcher, and **reads only** — it
writes a report and its arrays to the Data_Bank `Posit` tier and touches no pipeline input.
It needs no trained estimator and no GPU.

## Why this exists

`DETECTOR_WORKFLOW.md` §9.4 proposes reducing the inferred block, on the argument that a
parameter a cheap measurement can pin does not need to spend the inference's capacity. That
section is a **proposal and is not in force**: the implemented detector still infers all six.
This utility produces the evidence the proposal is gated on — for each parameter, how
accurately a direct estimator recovers it from the same pixels the network sees. A parameter
that clears its thresholds becomes a candidate to leave the block; one that does not, stays.

## What it computes

`sample_psf_width` draws `sqrt(2)·sigma` in pixels from a log-normal with scale `mu_r` and
shape `sigma_r` — **one draw per subunit**, carried by every dye of that subunit for the
whole recording and across its reactions. So `mu_r` is the *median* of `sqrt(2)·sigma` and
`sigma_r` the standard deviation of its logarithm, and both are population parameters of the
spots in one video. The estimator recovers that population in four steps.

1. **Detect.** Each frame is matched-filtered with a Gaussian of the expected spot width and
   thresholded at a multiple of the filtered background noise. Detections closer together
   than the isolation radius are dropped, because an overlapping pair cannot be fitted as one
   Gaussian. Detection runs **per frame**, not on a time average: the emitters diffuse, so a
   time-averaged image is smeared by the motion and its fitted width is not the PSF width.

2. **Fit.** Each detected spot is fitted with the renderer's own shape — the exact integral
   of a Gaussian over the pixel square, the same `erf` difference `add_pixel_counts` uses —
   so there is no shape mismatch between the model that generated the pixel and the model
   that reads it. Pixels within the exclusion radius of another detection are masked out.

3. **Link.** Detections are linked across frames into per-subunit tracks by greedy
   nearest-neighbor. Because a subunit's width is constant for the whole recording, a track
   is repeat measurement of *one* number, and averaging within a track before taking the
   population spread cuts the per-fit noise by the square root of the track length.

4. **Summarize.** `mu_r` is `exp(mean(ln w))` over track means; `sigma_r` is the spread of
   those track means with the measurement variance subtracted,
   `sigma_r² = max(var(ln w) − mean(se²), 0)`. That subtraction is the same
   errors-in-variables correction that makes the ThunderSTORM fitted spreads upper-biased
   (`DETECTOR_WORKFLOW.md` §6.7, caveat 1) — here the per-fit variance is available from the
   fit covariance, so it is subtracted rather than assumed.

## Three design choices the measurements forced

Each of these was adopted because the alternative was measured against the renderer and
found biased. They are recorded here because in every case the biased choice is the one that
looks more careful.

- **The background is supplied, not fitted.** A wide faint Gaussian and a slightly raised
  flat floor are nearly degenerate over a finite patch, so a free background absorbs the
  wings of a broad spot and the fitted width collapses. Measured: the fitted width ran 6%
  low at the bottom of the `mu_r` box and 25% low at the top — a *width-dependent* bias, which
  compresses the very spread `sigma_r` measures. The camera block is the SCOPE nuisance and is
  known to about 1% (three of the five boxes are ±1.15% about their centers, the other two ±0.58%), so `gamma·kappa_q·kappa_o + kappa_b` is supplied.

- **The fit is flat-weighted; the noise model enters only the uncertainty.** EMCCD variance
  rises with signal, so weighting each pixel by the model variance is the textbook efficient
  choice — and it down-weights the bright core that carries the width information, leaning
  instead on the noisy outskirts. Measured: `mu_r` biased **+0.06 dex** high. The point
  estimate therefore uses flat weights, and the variance law
  `Var = 2·gamma·(mean − kappa_b) + kappa_s²` is applied afterwards, through a sandwich
  covariance, to the standard error — which is where it matters, because that standard error
  is what the errors-in-variables correction subtracts.

- **Neighbors are masked.** At the emitter density of a real recording, one or two further
  emitters fall inside each fit patch on average, and their flux inflates the fitted width.
  Measured: **+0.018 dex** on `mu_r`. Pixels near another detection are excluded from the fit.

## Requirements

- The `SRM_AND_SBI_ENVY_V0` environment. No GPU and no trained estimator.
- For `--selftest`, nothing else: the scenes are rendered in memory.
- For a tier run, read access to a `Video_Set`, its `Theta_Set` (the ground truth) and its
  `Nuisance_SCOPE_Theta_Set` (the camera values the video was rendered with) for the split
  and tasks requested.

## How to run

```
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_PSF_Width.py \
    --condition FAB --total-time-seconds 2.0 --tasks 0 1 --max-videos 200 --workers 16
```

`--selftest` renders nine in-memory scenes on a 3 × 3 grid of `mu_r` and `sigma_r` and needs
no data tier; it is the check to run first on a new machine, and after any change to the
kernel. `--dry-run` resolves the profile, the paths and the settings and prints what it would
read and write without reading or computing anything. `--workers` sets the process pool;
the work is one independent measurement per video and scales nearly linearly.

`--expect-videos-per-task` guards a real hazard: a stale development tier can carry
production filenames while holding only a couple of videos, and scoring it would produce a
confident-looking but meaningless error. The check names that case in the report rather than
hiding it.

## Result and interpretation

The report states, for each parameter, the accuracy against the prespecified thresholds:

| parameter | criterion | threshold | basis |
|---|---|---|---|
| `mu_r` | mean absolute log10 error | ≤ 0.02 dex | 6.7% of its 0.3 dex prior width |
| `mu_r` | absolute mean signed error | ≤ 0.01 dex | a systematic offset propagates into every downstream use, so it is held tighter than the scatter |
| `sigma_r` | Pearson correlation with truth | ≥ 0.80 | whether the quantity is measured at all |
| `sigma_r` | mean absolute error | ≤ 0.08 | 10.7% of its prior width, in linear units because `sigma_r` is itself a spread |

A parameter meeting its thresholds is a **candidate** to leave the inferred block — not an
instruction to remove it. §9.4 requires more than accuracy: the quantity must also enter the
downstream stage as a nuisance with an explicit range rather than a point value, so that
uncertainty is retained rather than discarded.

## Essential notes

- **Not a calibration of the camera.** The five SCOPE camera values are read from the
  recording's own `Nuisance_SCOPE` record and never fitted. Gain and quantum efficiency enter
  the pixel mean only through the product `gamma·kappa_q`, so neither is separately
  identifiable from a video in any case.
- **Works in the stored 8-bit domain.** Videos are read as stored and converted to ADU by the
  fixed global map, the same domain the neural estimator sees, so a direct estimate and a
  posterior are read off the very same pixels. The quantization step (257 ADU) is smaller
  than the background standard deviation (≈302 ADU), which dithers it; float and 8-bit
  widths agree to four decimal places. This matters for deployment too: experimental
  recordings are 16-bit but every stage converts them with the same map before reading them
  (`io.convert_video_dtype`), so no separate experimental path is needed.
- **The observable population is truncated by detectability.** A spot wide enough spreads a
  fixed photon budget below the noise floor and is detected by nothing — neither this
  estimator nor the network. At the top of the `sigma_r` box a few percent of drawn widths
  are lost this way. That is a statement about the information in the data, not about the
  estimator, and it is quantified separately in the information-budget analysis.
- **Dye multiplicity is not modeled here.** Under `FAB_POISSON` a labeled subunit carries
  about two dyes, which share one width, so a single-subunit spot is unaffected. A dimer with
  both subunits labeled carries two *different* widths at one position, and its fitted width
  falls between them; under the MET-FAB occupancy that is a small share of visible dimers.
- **Reads only.** No pipeline input is modified; the utility writes its report, arrays and
  figures to the Data_Bank `Posit` tier.

## References

- `DETECTOR_WORKFLOW.md` §6.2 (the inferred imaging block and its priors), §6.7 (the
  ThunderSTORM reference values and the errors-in-variables caveat), §9.3 (the SCOPE camera
  marginalization) and §9.4 (the acquisition-information contract this utility serves).
- Thompson, R.E., Larson, D.R., Webb, W.W. (2002). Precise nanometer localization analysis
  for individual fluorescent probes. *Biophysical Journal* 82(5):2775–2783.
- Ober, R.J., Ram, S., Ward, E.S. (2004). Localization accuracy in single-molecule
  microscopy. *Biophysical Journal* 86(2):1185–1200.
- Mortensen, K.I., Churchman, L.S., Spudich, J.A., Flyvbjerg, H. (2010). Optimized
  localization analysis for single-molecule tracking and super-resolution microscopy.
  *Nature Methods* 7(5):377–381.
- Hirsch, M., Wareham, R.J., Martin-Fernandez, M.L., Hobson, M.P., Rolfe, D.J. (2013). A
  stochastic model for electron multiplication charge-coupled devices — from theory to
  practice. *PLoS ONE* 8(1):e53671.
