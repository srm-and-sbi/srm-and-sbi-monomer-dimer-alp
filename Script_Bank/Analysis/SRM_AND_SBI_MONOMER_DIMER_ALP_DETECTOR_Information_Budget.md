# Information budget — method and usage

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Information_Budget.py`. This utility
computes, for each learnable imaging parameter, an approximate lower benchmark on the standard
deviation of an unbiased estimator from one recording — what the data allow under a reduced
model, independently of which estimator is used. This note explains what it computes, how to run it, and how to read the
result, without reading the code.

This is a special-situation utility, not one of the canonical pipeline stages. It lives in
`Script_Bank/Analysis`, is never wired into the stage dispatcher, and reads nothing: the
bounds are analytic functions of the parameter values, the camera block and the size of the
recording. It needs no data tier, no trained estimator and no GPU.

## Why this exists

When an estimator recovers a parameter poorly, two explanations look identical in the
results: the estimator may be weak, or the recordings may not carry the information. They
call for opposite responses — improve the estimator, or stop trying to infer the parameter —
so a campaign that cannot distinguish them will spend effort in the wrong place. The
posterior-calibration outcome of `DETECTOR_WORKFLOW.md` §6.9 raised exactly that question,
and §9.4 proposes reducing the inferred block partly in response. That section is a
**proposal and is not in force**. This utility turns the question into a measurement by
supplying the missing third number:

| | what it is | where it comes from |
|---|---|---|
| **bound** | what the data allow | computed here |
| **direct** | what a direct, non-neural estimator achieves | the direct-estimator reports |
| **neural** | the posterior width of the amortized flow | the calibration report, §6.9 |

Reading them together grades each parameter:

- **neural near the bound** — unlikely to gain much from a different estimator or a wider
  inferred block.
- **neural far from the bound** — probable headroom; the estimator or its training is the first
  place to look.
- **bound wider than the prior** — the reduced model expects one recording to constrain the
  parameter poorly, which makes it a candidate to leave the inferred block. It does not prove
  non-identifiability: the bound is approximate, treats a reduced observable rather than the
  full video, and constrains an unbiased estimator's standard deviation, not the error of a
  biased or Bayesian one.

## What it computes

The likelihood is treated as Gaussian with the exact EMCCD mean and variance of the
Poisson–Gamma–Normal chain, both verified numerically against the renderer:

```
mean(ADU) = gamma * kappa_q * I + kappa_b
var(ADU)  = 2 * gamma^2 * kappa_q * I + kappa_s^2
```

The factor 2 is the excess-noise factor of the electron-multiplication stage: multiplication
doubles the variance per detected photoelectron, so an EMCCD pixel carries the noise of half
as many photons as its count suggests. The approximation is standard in localization
microscopy and is accurate wherever a pixel collects more than a few photoelectrons, which
holds across the whole prior box here.

**Per spot.** The Fisher information for a spot's `(amplitude, x, y, sigma)` is built by
numerically differentiating the same pixel-integrated Gaussian the renderer uses, so the
pixelation is exact rather than approximated by a continuum integral. The full matrix is
inverted before the width entry is taken, so the bound accounts for the amplitude and the
center being unknown too — which they are, and treating them as known would understate the
bound substantially, because amplitude and width trade off directly.

**Population parameters.** Each spot contributes `ln w_i ~ Normal(ln mu, sigma_r² + v)`, where
`v` is the per-unit measurement variance. The bounds are the standard normal-sample results:

```
sd(ln mu)    >=  sqrt((sigma_r^2 + v) / n)
sd(sigma_r)  >=  (sigma_r^2 + v) / (sigma_r * sqrt(2 * (n - 1)))
```

The second expression behaves in two regimes, and the crossover is the decision-relevant
fact. For `sigma_r` well above `sqrt(v)` it reduces to `sigma_r / sqrt(2(n-1))` — a *constant
relative* precision, no degradation anywhere. Only below `sigma_r ≈ sqrt(v)` does the
measurement term take over and the bound grow. Linking a spot across frames replaces `v` by
`v/L`, which moves that crossover down. At the operating point of a MET-FAB recording,
linking puts `sqrt(v)` at about 0.012, well under the prior floor of 0.10, so the whole prior
sits in the good regime; without linking `sqrt(v)` is about 0.086, comparable to the floor,
and the bottom of the prior is genuinely information-starved. That is what averaging within a
track buys, and it matches what the direct estimator measures.

**Photobleaching.** The curve is `S(t) = A exp(-k t) + B` with the amplitude and offset
unknown, so the bound is the `(k, k)` entry of the **inverse** of the 3 × 3 information matrix,
not the reciprocal of its own diagonal. That distinction decides the answer. Where the decay is
shallow, the exponential is nearly a straight line over the observed window, the three
parameters become nearly degenerate, and only the product `A k` is determined — the rate alone
is barely constrained however many frames are collected. Treating `A` and `B` as known
understates the bound by an order of magnitude exactly in that regime. With them known the
bound would reduce to the familiar `sd(k) >= eps sqrt(3 / T^3)`; the cubic dependence on
duration survives profiling. The model's parameter is a probability over a **fixed**
hundred-frame reference window, not over the clip, so the rate bound is propagated through
`p = 1 - exp(-100 k)`.

**The flicker correlation, which dominates.** The brightness is an Ornstein–Uhlenbeck process
with correlation time `1 / lambda_rate`, so consecutive frames of a total-fluorescence curve
are *not* independent samples of the decay. The effective count is
`n_eff = n (1 - rho) / (1 + rho)` with `rho = exp(-lambda_rate * frame_time)`. At the center
of the `lambda_rate` prior the correlation time is about fifteen frames, so a hundred-frame
recording carries roughly **three** effectively independent samples of the decay, not a
hundred. It is easy to omit, and omitting it makes the recordings look far more informative
than they are. Combined with the shallow-decay degeneracy above, it puts the benchmark standard
deviation for `prob_photo_bleach` more than an order of magnitude above the 0.10 dex threshold
anywhere in its prior for a 2 s recording (1.26 dex at the top against a 1.5 dex prior width),
and inside that threshold only in the upper half of the prior at 20 s — see
`DETECTOR_WORKFLOW.md` §9.5 for the table. The effective-sample-size adjustment is an approximation
to the correlated decay likelihood, not an exact treatment of it, so the result is an approximate
precision benchmark for an unbiased decay-rate estimator using total fluorescence and not a
fundamental recovery limit for inference from the full video.

**Flicker rate.** From the lag-one correlation of a stationary AR(1) series,
`sd(rho) >= sqrt((1 - rho^2) / (n L))`, propagated through `lambda_rate = -ln(rho)/dt`.

## Requirements

The `SRM_AND_SBI_ENVY_V0` environment. Nothing else: no data, no estimator, no GPU.

## How to run

```
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Information_Budget.py \
    --total-time-seconds 2.0 --n-emitters 154
```

`--n-emitters` is the visible spot count of one recording; the default is the MET-FAB
operating point (about 1233 subunits at a visible fraction of 0.125). `--compare-seconds`
tabulates the duration scaling, defaulting to the 2 s training clips and the 20 s
experimental recordings. `--measured` takes a small JSON of measured direct and neural
spreads and overlays them into the comparison table. `--dry-run` resolves the settings and
prints what it would compute.

## Result and interpretation

The report gives the bound for each parameter against its own prior width, the `sigma_r`
crossover table, and the duration/correlation table for `prob_photo_bleach`. A bound
approaching the prior width means one recording carries almost no information about that
parameter — the posterior can then only return the prior, and a calibration run will show
exactly that.

**Every bound here is optimistic.** They omit emitters entering and leaving the field,
reactions changing a spot's dye multiplicity mid-recording, overlapping spots, and the
truncation of the observable population by detectability — a spot wide enough spreads a fixed
photon budget below the noise floor and is detected by nothing. A real estimator faces all
four. These are not predictions of achievable error. A measured scatter well *below* one of
these bounds is a prompt to check the measurement — ground truth leaking into the estimate, or a
mis-stated bound — rather than an automatic failure: the bounds constrain an unbiased estimator,
and a biased or Bayesian estimator that shrinks toward the prior can legitimately do better.

## Essential notes

- **A bound is not a verdict.** A parameter whose bound is comfortably inside its prior has
  favorable information *in principle*, under the reduced model; whether any implemented
  estimator attains that precision is a separate, measured question, and whether it should leave the inferred block is a third question that
  §9.4 gates on more than accuracy — in particular, a quantity that leaves must enter the
  downstream stage as a nuisance with an explicit range, never as a point value.
- **The camera block is assumed known.** Gain and quantum efficiency enter the pixel mean only
  through the product `gamma * kappa_q`, so neither is separately identifiable from a video;
  both are supplied from the SCOPE box, whose width is a deliberate anchor rather than a
  measured uncertainty (§9.3).
- **Bounds assume unbiasedness.** A biased estimator can have a smaller mean squared error
  than the Cramér–Rao bound on the variance. Comparisons in the reports are therefore stated
  against estimator *scatter*, with bias reported separately.

## References

- Rao, C.R. (1945). Information and the accuracy attainable in the estimation of statistical
  parameters. *Bulletin of the Calcutta Mathematical Society* 37:81–89.
- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
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
- Uhlenbeck, G.E., Ornstein, L.S. (1930). On the theory of the Brownian motion. *Physical
  Review* 36(5):823–841.
- `DETECTOR_WORKFLOW.md` §6.2 (the inferred imaging block and its priors), §6.5 (the flicker
  model), §6.9 (the calibration outcome these bounds contextualize), §9.3 (the SCOPE camera
  marginalization) and §9.4 (the acquisition-information contract).
