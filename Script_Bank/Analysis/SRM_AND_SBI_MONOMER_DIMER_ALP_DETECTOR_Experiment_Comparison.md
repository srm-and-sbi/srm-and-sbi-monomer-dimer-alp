# Experiment comparison (detector)

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Comparison.py`. It takes any set of trained
detector estimators, named by their artifact tags, and compares what their Experiment stages returned on the
same windows of the experimental recordings, with the estimators as columns. It reads finished products and
computes; it loads no estimator and draws no posterior. The reading belongs to `DETECTOR_WORKFLOW.md`.

## What it can and cannot establish

The experimental recordings carry no ground truth. The comparison measures where each estimator places its
estimates, how wide its posteriors are, how often its estimates leave the prior box, how the three point
estimates of one posterior agree, how an estimate moves between the windows of one recording, how it drifts
with window position, how it is structured between recordings, and how the estimators agree window by window
and recording by recording. It measures no recovery and no accuracy. Agreement between two estimators is not
correctness, and a disagreement locates where they differ, not which is right; synthetic recovery and
calibration (the Evaluation and Posterior_Calibration stages) are the evidence on accuracy.

The prior box is the interval of parameter values that generated the training recordings. The Experiment
stage samples the posterior without restriction to the box, so a per-window estimate outside it is an
extrapolation of the learned density beyond the training recordings, not a posterior estimate under the box
prior; the count and share of windows below and above the box are reported for every point estimate, with the
farthest excess beyond it and the windows whose whole stored 90 % interval lies beyond it.

## Reading rules

- **Crossing the box is not evidence that the box is too narrow.** Estimates clustering near a boundary and
  frequently crossing it, by a small amount relative to their own interval, leave two readings open: the
  recordings sit near that value, or they pull beyond it and the learned density stops near its edge. The
  comparison cannot separate them; a whole interval beyond the box is the stronger statement, and neither
  decides the prior for a regeneration.
- **Agreement of the three point estimates is a typical discrepancy.** The tables report the median over
  windows of |MAP − median|, |SGM − median| and |MAP − SGM| beside the maximum over windows; close values do
  not establish that every posterior is single-peaked.
- **A statement about every window position is not a statement about every window.** The drift figures and
  the window-position medians describe the median across recordings at each window index; the range of the
  window medians and of the per-recording medians is reported separately.
- **Agreement with a selected vector is compatibility, not confirmation.** The shift from the vector is
  reported in dex and as a percentage with the exact count of windows above it; the vector has no privileged
  correctness, and a departure from it is not an error of the estimator.
- **Narrow and consistent is not correct.** Posterior width, the agreement of point estimates, boundary
  clustering and temporal trends each answer a different question; none of them measures accuracy.

## Inputs, per column (read-only)

| record | required | read as |
|---|---|---|
| `<alias>_<timing>[_<TAG>]_MAP_Experiment/` | yes | through the artifact schema (a superseded product is refused); the MAP, the posterior quantiles at 0.05 / 0.25 / 0.50 / 0.75 / 0.95 and the SGM per window, the `(kind, cell, chunk)` identifiers, the manifest (checkpoint, pool mode, draw count, optimizer, window geometry, job, invocation) |
| `<alias>_<timing>[_<TAG>]_Estimator.npz` | no | its weights checksum, which must equal the product's checkpoint |
| the Experiment job log (`--log NAME=PATH`) | no | the per-rank `MAP stops` lines, totaled |

Columns are `--control` (optional), `--base` (the order of the windows) and `--candidates` (any number,
possibly none). `baseline` names the untagged estimator. A candidate without an Experiment product is listed as
missing and not compared; a missing base or control stops the run.

`--vector-tag` (optional) names a `selection_user` Nuisance_DLI of the same condition and window, for example
`REF`; its one vector is printed beside the estimates as the selected simulation values, with the median shift
of every estimator from it. An artifact of another kind is refused. The localization-table reference values of
`DETECTOR_WORKFLOW.md` §6.7 (the ThunderSTORM fits on the public recordings) are printed in the same way, as
the temporal-dynamics stage carries them; they are reference methods, not ground truth, and
`prob_photo_bleach` has none.

## Checks before anything is computed

- every product passes the schema, holds the workflow's parameter keys in order and the canonical quantile
  levels, and has unique window identifiers;
- the window sets are equal across columns, and the condition labels, the window geometry and the pool mode
  agree; otherwise the estimators were not applied to the same windows in the same way, and the comparison is
  refused. The draw count is recorded, not required to agree;
- where the estimator artifact is present beside the product, its weights are the product's checkpoint;
- an existing output folder is never overwritten.

## Computed

Per column, per point estimate (MAP, posterior median, SGM) and per parameter, all in log10:

- **Location**: the median over windows, its IQR, the mean and SD, the physical value of the median, the
  count and share of windows below and above the prior box, the range of the window medians (log10 and
  physical) and the farthest excess below the floor and above the top (dex).
- **Widths and boundary**: the median over windows of the central 50 % and 90 % interval widths from the
  stored quantiles, the 90 % width as a share of the prior width, and the count and share of windows whose
  whole stored 90 % (and 50 %) interval lies below the floor or above the top.
- **The three point estimates of one posterior**: the signed median MAP − median, the same in units of each
  window's posterior IQR, the share of windows whose MAP lies inside its own stored 50 % and 90 % intervals,
  and |MAP − median|, |SGM − median| and |MAP − SGM| as the median over windows (typical) and the maximum over
  windows (largest). A disagreement among the three is a lead about the shape of the posterior before it is
  a statement about the recordings; agreement says nothing about the number of modes.
- **Movement between the windows of one recording**: the SD across a recording's windows, median over
  recordings, and the number of adjacent-window changes larger than 0.3 dex (the material-drift bar of the
  Experiment stage, a factor of two) with the number of recordings in which they occur.
- **Drift with window position**: the Experiment stage's own kernel (`temporal_dynamics.window_drift_rows`):
  each recording's estimate fitted against the window index, the first-to-last change aggregated across
  recordings (median, IQR, sign consistency, share over 0.3 dex, signed-rank p).
- **Structure between recordings**: the SD of the per-recording means against the median SD within a
  recording, and the range of the per-recording medians (log10 and physical).

Against the selected vector, per point estimate and parameter: the median over windows minus the vector value
in dex and as a percentage of the vector value, and the count and share of windows above the vector.

For every pair of columns, `later − earlier` in the column order, per point estimate and parameter:

- **window by window**: Pearson and Spearman correlation, the median and mean shift, the median |difference|
  and the share of windows where the later column is higher;
- **recording by recording**: the Pearson correlation of the per-recording means and their median shift.

A correlation is undefined, and reported as such, when one series is constant.

## Output

`<data_bank>/<posit>/<alias>_<timing>_Experiment_Comparison_<COLUMNS>/` (the columns in order, upper case;
never overwritten; `--out-dir` writes elsewhere):

- `comparison.md`: the columns with their provenance and identity, the location tables of the three views
  (counts beside shares), the range-and-boundary table, the physical values beside the selected vector and the
  references, the shift from the selected vector, the agreement of the three point estimates (typical and
  largest), the widths, the movement within recordings, the agreement between estimators, the drift and the
  structure between recordings;
- `comparison.json`: every number; `comparison.npz`: the aligned arrays (identifiers, prior bounds, per column
  the three point estimates, the quantiles and the scores, and the selected vector);
- `figures/distribution_<parameter>.png` (per-window estimates of every column, three views, prior bounds,
  selected vector and references), `figures/pairs_<parameter>.png` (every pair, three views) and
  `figures/window_drift_<column>.png` (the three estimates against window position over the posterior's 50 %
  and 90 % bands), all lossless PNG;
- `PROVENANCE.md` (inputs with checksums, jobs, invocations, checkpoints, optimizer and window geometry, the
  selected vector's identity, and the code identity: package version, git revision, working-tree state and the
  script's own sha256) and `README.md`.

## Usage

```bash
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Comparison.py \
    --total-time-seconds 2 --base CAP256 --control baseline \
    --candidates CAP256KERNEL7EARLYCONVSTATSGB128 --vector-tag REF \
    --log baseline=<job log> --log CAP256=<job log> --log CAP256KERNEL7EARLYCONVSTATSGB128=<job log> \
    [--dry-run]
```

CPU only, seconds for 600 windows. Tests: `tests/test_experiment_comparison.py` (run directly).
