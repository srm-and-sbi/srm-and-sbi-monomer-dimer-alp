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

On four 6 s scenes spanning the prior, the estimator passed both criteria — correlation 0.9975
against the 0.80 threshold, mean absolute error 0.0637 dex against 0.08 — with these points:

| true λ | estimate | error (dex) | usable traces |
|---|---|---|---|
| 1.5 | 1.839 | +0.0885 | 257 |
| 3.0 | 3.614 | +0.0808 | 292 |
| 5.0 | 5.310 | +0.0261 | 290 |
| 8.0 | 9.173 | +0.0594 | 294 |

Two features are worth stating rather than leaving to be read off the table. **The error is
almost entirely systematic**: the mean absolute error and the mean signed error are the same
number to four decimals, because every point runs fast, and the correlation of 0.9975 says the
ordering is essentially perfect. The estimator ranks flicker rates far better than it places
them. **Dye multiplicity does not explain it** — the multiplicity band is 0.009 dex at the
center of the `sigma_pc` prior, about a seventh of the observed offset — so the remainder is a
property of the measurement chain, most plausibly track fragmentation: a linker that splits one
emitter into two traces decorrelates the series faster than the model arm, which assumes intact
traces, and a faster decay reads as a higher rate.

The predicted low-λ-worst pattern holds at the ends (0.0885 at the bottom against 0.0594 at the
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
    --condition FAB --total-time-seconds 2.0 --tasks 0 --max-videos 100 --workers 16
```

`--min-track-length` sets the shortest usable trace, defaulting to the 40 frames the derivation
of record required. `--model-traces` sizes the Ornstein–Uhlenbeck arm per grid point; the arm is
vectorized by trace length, costing about 1.5 s per grid point, so roughly 16 s per recording
for the ten-point grid. `--dry-run` resolves settings and prints what it would read and write.

## Result and interpretation

The report gives the Pearson correlation and mean absolute log10 error against the prespecified
thresholds — correlation ≥ 0.80 and MAE ≤ 0.08 dex, the latter being 8% of the 1.0 dex prior
width — together with the multiplicity systematic as a band. That band sits inside the threshold
at the center of the `sigma_pc` prior and consumes most of it at the top corner, which the
report says explicitly rather than leaving it to be inferred.

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
