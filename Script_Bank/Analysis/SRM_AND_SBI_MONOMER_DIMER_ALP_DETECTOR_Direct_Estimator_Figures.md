# Recovery figures of a direct-estimator run

`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Estimator_Figures.py` redraws, for one run folder of a
direct estimator (PSF width, flicker rate or fluorescence loss), the recovery figures the utilities write
at the end of every tier run: the true value of the estimated imaging parameter against the inferred one,
per attempted recording.

## What it draws

Two figures per estimated parameter, each of three panels, through the shared builder
`direct_acceptance.recovery_figures`, so that a redrawn figure and a run's own figure are the same figure:

1. **Every attempted recording by outcome**: scored (the recordings the accuracy rule judges; for the
   fluorescence-loss estimator those its eligibility diagnostic accepts), valid but not scored, and
   failed; the identity line; the prior bounds of the parameter (dotted); the accuracy rule's band.
2. **The scored recordings with their nominal 90 % ranges**, and in the panel title the mean absolute
   error, the mean signed error and the measured coverage in the figure's coordinates.
3. **The error against the true value**, inferred minus true, with the binned median and interquartile
   band of the scored recordings and the binned median of the rejected ones over eight equal bins of the
   log10 prior.

The first figure, `figures/recovery_<key>_log10.png`, is in the parameter's log10 prior coordinates, the
coordinates of the detector prior and of the dex accuracy rules; the second,
`figures/recovery_<key>_linear.png`, is in absolute values, where the lower end of a log-uniform prior
is a narrow sliver. Both say the same thing; a reader uses the one whose scale matches the question.

For a fluorescence-loss **experiment** run folder (`direct_fluorescence_loss_experiment.npz`) the utility
redraws the field-decline figure instead, through `direct_acceptance.field_decline_figure`: every recording's
background-subtracted field flux divided by its opening-window mean with the median over recordings, and over
it the ideal single-rate decline `(1 − p)^(t/100)` for the comparison values the run recorded (by default
the working bleaching value 0.05 and its late-time variant 0.03 of `DETECTOR_WORKFLOW.md` §7.6); the same for the
dimmest and the brightest third of the recordings by opening brightness; the per-frame background level; and
the fitted whole-window values with the scenarios marked. The overlay is what shows at a glance where a
one-rate renderer at those values runs against the recordings. It also redraws `figures/window_rates_<condition>.png`
(`direct_acceptance.window_rate_figure`, from `direct_imaging_estimates.field_decline_by_window` on the saved
curves): the local single rate on non-overlapping windows of every documented duration against the window's
position along the recording, one series per duration with the median over recordings and the interquartile
band, each series' pooled median as a thin horizontal line, and the same scenarios as reference lines.

## How to run

```
PYTHONPATH=$PWD python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Estimator_Figures.py \
    <run folder> [--out-dir DIR] [--dpi 200]
```

The run folder is a tier run's output (`summary.json` naming the estimator and `direct_<estimator>.npz`
with the per-recording arrays). The utility writes only PNG files, into the run's `figures/` unless
`--out-dir` names another place, and leaves `report.md` and every other file of the run as the run
wrote them. A folder without the arrays (an experiment-mode run, a selftest, or anything else) is
refused with exit status 2. Runs written before the figures existed get them this way; runs from this
release on carry them already and list them at the end of `report.md`.

## Scope

A reading aid for runs that already exist. It judges nothing: the verdicts of a run are those of its
`report.md` and `summary.json`, under the frozen rules of `DETECTOR_WORKFLOW.md` §9.6, and the figures
show the recordings those verdicts were computed from.
