# Point-estimate validation

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Point_Estimate_Validation.py`, over the shared
engine `srm_and_sbi_monomer_dimer_alp.point_estimate_validation_runner` and the kernel
`srm_and_sbi_monomer_dimer_alp.point_estimate_validation`.

The authoritative protocol for validating the three point estimates is `VALIDATION.md` §3.4:
median correctness, then SGM correctness and sampling stability, then MAP optimization, then the
regenerated analyses, then scientific interpretation. This utility served the median check and
supplied the draws of the SGM checks; this note records what it does and what was run with it.
Posterior calibration stays separate throughout: it evaluates the distribution, not an individual
point estimate.

## Decision of record (2026-09-23)

The median and the exact per-observation SGM need correctness checks, not a sampling study; the
dedicated run over 2,000 recordings is cancelled. Regenerating the analyses is the production-scale
test for both summaries. What was run:

- **Median.** `tests/test_median_reference.py` establishes the calculation, the parameter order and
  the quantile-then-transform convention. The ten-recording pilot below checked the real
  sampler-to-output path: stored medians equal the independent recomputation exactly, and every
  per-recording ratio was at most 0.058.
- **SGM.** `tests/test_sgm_reference.py` checks hand-computed summed distances, the declared prior-width
  scaling (including a cloud where it changes the answer), membership, and deterministic handling of
  duplicates and ties. On the same ten recordings and the median's draws, the production SGM equals a
  brute-force medoid on all 50 clouds and on the sampler-to-output path.
- **SGM convergence check.** On the same ten recordings, on which the repeat-to-repeat SGM
  variability was first seen: five repeats at 1,000, 2,000 and 4,000 draws per recording. Variability
  is the standard deviation across repeats divided by the posterior IQR of the pilot's 10,000-draw
  reference. The median's shrinks at about the rate sample size predicts (median 0.025, 0.020, 0.015).
  The SGM's shrinks slowly (0.13, 0.11, 0.09; worst case about 0.4 throughout), while every selected
  SGM is nearly as central as the most central draw of the common reference cloud (at most 6 % of the
  gap to a typical draw at 1,000 draws, 2.5 % at 4,000). The summed-distance objective is flat near its
  minimum: several quite different draws are almost equally central, and more draws improve the score
  without pinning the vector. In dex, one recording's SGM carries a median 0.005 (`mu_r`) to 0.03
  (`prob_photo_bleach`) of Monte Carlo spread at 1,000 draws. Averages over many recordings wash this
  out; it matters where one recording's SGM vector is used directly. From 0.1.17 the summaries use
  10,000 draws per observation.

The approximate large-collection method of the collection-level SGM kernel is separate and does not
affect the per-observation SGM.

## Phase 1: the median as the reference (the protocol as first drafted)

**The quantity.** For one observation, the 0.50 level of `posterior_quantiles`: numpy's default
linear interpolation (Hyndman & Fan type 7) of each coordinate's draws, in estimator coordinates
(log10 for a log row). A physical value is the transform of that quantile. With finite-sample
interpolation the reverse order, transforming the draws first, need not give the same number, so it
is never used for this quantity; the implementation checks in `tests/test_median_reference.py` pin
that convention.

**Frozen for the whole program.**

| item | value |
|---|---|
| checkpoint | the estimator of record, `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Estimator.npz`, identified by its full weights and file checksums, recorded at run start |
| observations | every recording of EVAL tasks 0 and 1 of the detector FAB 2 s tier: 2,000 recordings, the set of the direct-estimator record |
| dim subgroup | true log10 `mu_pc` in [2.00, 2.375), as frozen in `DETECTOR_WORKFLOW.md` §9.6 |
| sampling | bounded, recorded in the artifact |

This validates that checkpoint's summaries; it does not endorse its calibration. The observations
are development data, not untouched adoption evidence.

**The draws.** Per recording, five repeats at the production 1,000 draws and one reference run of
10,000, each from its own random stream: a seed derived from the base seed and `(task, sim,
stream)`, so a recording gets the same streams whichever rank processes it. All draws come from the
production `posterior_summary` with `return_samples=True` and `return_sgm=False`; no medoid is
computed in this phase.

**Acceptance, fixed before any comparison.** The smallest subset-level difference in MAE or bias the
program interprets is 0.005 dex: an analysis-resolution choice, not a biological tolerance or a
significance threshold. Per parameter, overall and in the dim subgroup:

1. *Recomputation.* Every stored quantile equals a sort-and-interpolate recomputation from its own
   stored draws to an absolute 1e-6 dex.
2. *Subset stability.* MAE and signed bias are computed separately for each repeat; their sample
   standard deviation across the five repeats is at most 0.001 dex. Minimum and maximum are reported.
   Five repeats estimate the Monte Carlo variability; they do not establish an upper bound.
3. *Resolution check.* The repeat mean of MAE and of bias lies within 0.001 dex of the reference run.
   The reference is a finite sample, not exact truth: a narrow failure is assessed against the
   reference's own sampling variability before 1,000 draws are called inadequate.
4. *Per recording.* The standard deviation of the five repeat medians, divided by the posterior IQR of
   the reference run, is at most 0.1 for at least 95 % of recordings. A normal posterior gives about
   0.03 at 1,000 draws. Every exceedance and the maximum are reported; a zero IQR carries an explicit
   status, counted as meeting the criterion when the repeat spread is also zero and as an exceedance
   otherwise.

A large error is `|median - truth|` above 0.3 dex, a factor of two; its share is reported. These
checks establish sampling stability. They cannot remove posterior bias or miscalibration.

## The validation artifact

A separate format, never a schema-1 stage product: `kind = "point_estimate_validation"`, version 1.
Arrays: `task_index`, `sim_index`, `true_log10`, `dim_subgroup`, `repeat_draws`
`(N, 5, 1000, D)`, `repeat_quantiles`, `reference_draws` `(N, 10000, D)`, `reference_quantiles`,
`stream_seeds`, `stream_seconds`, `stream_peak_bytes`. The manifest records the parameter order,
the full checkpoint checksums, the code provenance at startup and at write, the seed rule and base
seed, the sampling mode and draw counts, the interpolation convention, the frozen observation list,
the dim rule and the thresholds. The float32 draws of the full list take about 720 MB before
metadata. A multi-GPU run writes one shard per rank; `--merge` requires every rank and the exact
frozen observation list.

## How to run

Dry run first: it resolves the inputs, lists the frozen observations and, with `--pilot`, the
selected recordings.

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Point_Estimate_Validation.py \
    --condition FAB --total-time-seconds 2.0 --dry-run
```

**Pilot.** `--pilot 10` selects ten recordings deterministically, five dim and five bright, at evenly
spaced ranks of the true `mu_pc` within each subgroup. It validates the utility and measures
throughput and memory; its report issues no acceptance verdict. On a machine without the EVAL tier,
prepare a scratch data bank with the same layout (`Posit/`, `Theta/`, `Video/`) holding the checkpoint
(checksum compared with its source), the two full theta stores, and only the selected video chunks,
then pass `--data-bank-root` and `--output-dir`. A video that reads as all zeros, the fill value of a
chunk absent from a partial store, stops the run.

**Full run.** The complete list on the machine that holds the tier, with a separately approved job
and a dry run of its launch script; then `--merge`.

## What it does not establish

It validates the median of one checkpoint on development data. Accuracy differences between the
median, the SGM and the MAP are judged in the later phases on the same observations; agreement
between two point estimates is descriptive, and synthetic truth decides which has lower error.
