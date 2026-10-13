# Raw-domain decline cross-check (detector)

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Raw_Decline_Cross_Check.py`. It places one detector
estimator's Experiment estimates on the experimental recordings beside the raw-domain fluorescence-decline
record of the same recordings (`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Fluorescence_Loss.md`, mode
`experiment`, domain `raw`), recording by recording and window by window. It reads finished products and
computes; it loads no estimator, renders nothing and fits nothing. The reading belongs to
`DETECTOR_WORKFLOW.md`.

## What it compares, and what it does not claim

The quantities are different and are kept apart:

- **The estimator's `prob_photo_bleach`** is the per-dye probability of entering the absorbing dark state
  within the 100-frame reference interval, as the renderer defines it (`DETECTOR_WORKFLOW.md` §6.2),
  inferred from each 2 s window of a recording. Its per-recording value here is the median over the windows.
- **The raw rates** are readings of the median-subtracted total fluorescence of the field, a proxy for the
  per-dye process and not a count of dyes, probes or spots: the free-offset fitted rate (the direct
  estimator's value, a free-offset fit that also returns 0.19 and 0.23 on single-rate renders at 0.05, so
  it does not by itself establish a fast component; usable fits only), the whole-recording rate implied by
  the closing-to-opening window ratio, the late-phase rate implied by the mid-to-closing ratio, and the
  local rate inside each window of the estimator's duration, computed by the raw-domain companion from the
  opening and closing quarters of that window. Each is expressed per 100 frames: the same units as the
  estimator's parameter, not the same measurement scale.
- **The analytical dye retention** at a per-dye value `p` over the `d` frames between the opening and
  closing windows is `(1 − p)^(d/100)` under independent per-dye bleaching and stationary mean brightness;
  grouping the dyes onto probes and complexes does not change this expectation. It is an expectation of dye
  fluorescence, not of the median-subtracted field statistic or of spot retention, and not a render. It is
  computed at the estimator's per-recording value and at the comparison values the raw record carries, and
  placed beside the experimental field retention without a difference or a test: a field retention is
  compared with synthetic field retention measured in the same way.
- **Brightness**: the estimator's `mu_pc` (the median photon contribution of one active dye per frame) is
  placed beside the opening field excess of the recording (the opening window's mean pixel level minus its
  median, per pixel; a median-subtracted field statistic that includes a camera-noise contribution). These
  are different quantities; their relationship is diagnostic, never an equality.

Agreement is not correctness and disagreement does not say which side is right. The cross-check assumes
neither the estimator's values nor any selected simulation vector to be correct, and establishes no cause
for a discrepancy: limited information in experimental 2 s windows, a mismatch between simulated and
experimental recordings, the inadequacy of one constant rate and the median subtraction of the field
statistic, which keeps part of the camera noise, remain open readings.

## Inputs (read-only)

| record | read as |
|---|---|
| `<alias>_<timing>[_<TAG>]_MAP_Experiment/` | through the artifact schema: the posterior median, the MAP and the SGM per window, the `(kind, cell, chunk)` identifiers, the window geometry and the manifest identity |
| `<alias>_<span>_Direct_Fluorescence_Loss_Experiment_RAW/direct_fluorescence_loss_experiment.npz` and its `summary.json` | per recording: the free-offset fitted rate and its usability, the whole-recording and late-phase rates and ratios, the opening-to-closing separation in frames, the opening field excess and the opening and closing background levels; per window of the estimator's duration: the local rate and the window starts; the comparison values |

## Checks before anything is computed

- the Experiment product passes the schema, holds the workflow's parameter keys and unique windows, and
  carries a window geometry;
- the raw record is the raw domain of the same condition and carries local rates at the estimator's window
  duration;
- the recording sets are equal, and every window start of the estimator (window index times step) is a
  window start of the raw record; otherwise the two were not read from the same recordings in the same way
  and the cross-check is refused;
- an existing output folder is never overwritten.

## Computed

For each point-estimate view (posterior median in the tables; MAP and SGM in the JSON as well):

- **Per recording**: the estimator's bleaching against the free-offset fitted rate, the whole-recording rate
  and the late-phase rate: over every recording with both values finite, the count, Spearman and Pearson
  correlation, the medians, the count of recordings where the estimator's value is the larger and the count
  where the raw rate is zero or below; over the recordings where both are positive, with their own count,
  the median of `log10(estimator / raw)` in dex.
- **Retention**: the analytical dye retention at the estimator's per-recording value beside the
  experimental field retention: their medians, their correlation, the correlation of the estimator's
  bleaching with the field retention, and the analytical retention at each comparison value; no difference
  and no count of one below the other.
- **Window by window**: the estimator's bleaching against the local direct rate of the same window of the
  same recording: over every window with both values finite (correlation, medians, count where the
  estimator is the larger), with the windows whose field statistic did not fall (local rate zero or below)
  counted apart, since any positive estimate exceeds them by construction; the median shift in dex over the
  windows with a positive local rate, with their count; the same per window position; and within each
  recording across its positions (the median Spearman over recordings and the count of recordings where it
  is positive).
- **Brightness**: the correlation of the estimator's `mu_pc` (window 0 and recording median) with the
  opening field excess; beside it the correlation of the estimator's bleaching, the raw late-phase rate and
  the raw whole-recording rate with the opening field excess.

A correlation is undefined, and reported as such, below three pairs or when a series is constant.

## Output

`<data_bank>/<posit>/<alias>_<timing>_Raw_Decline_Cross_Check_<ESTIMATOR>/` (never overwritten; `--out-dir`
writes elsewhere): `cross_check.md` (the tables of the posterior-median view), `cross_check.json` (every
number, every view), `cross_check.npz` (the aligned arrays), `figures/rates_per_recording.png`,
`figures/retention.png`, `figures/window_positions.png`, `figures/brightness.png` (lossless PNG),
`PROVENANCE.md` (inputs with checksums, identities, alignment, the diagnostic purpose and the recorded
authorization) and `README.md`.

## Usage

```bash
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Raw_Decline_Cross_Check.py \
    --total-time-seconds 2 --estimator CAP256KERNEL7EARLYCONVSTATSGB128 \
    --authorization "<who, when>" [--raw-span-seconds 20] [--raw-record <npz>] [--dry-run]
```

CPU only, seconds. Tests: `tests/test_raw_decline_cross_check.py` (run directly).
