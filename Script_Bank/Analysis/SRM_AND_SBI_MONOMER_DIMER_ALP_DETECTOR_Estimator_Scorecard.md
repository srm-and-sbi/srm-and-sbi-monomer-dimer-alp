# Estimator scorecard (detector)

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Estimator_Scorecard.py`, the compiler of the scorecard
that the encoder screening of `DETECTOR_WORKFLOW.md` judges its candidates by. It takes any set of trained
detector estimators, named by their artifact tags, and writes one record with the estimators as columns and
the scorecard's aspects as rows, from the records the pipeline already produces. It computes and marks; the
reading and the selection are made in the workflow document, under the selection rule written there.

## What is scored

The whole neural posterior estimator that a training produced: its encoder, the flow trained jointly with
it, and its training configuration (per-GPU batch, steps). A difference between two columns is a difference
between two trained estimators, not an isolated architecture effect, for the reasons the workflow document
records (joint training, the flexible-batch rule, one training run per arm). The encoder row is the one
encoder-level view, and it selects nothing.

## Inputs, per column (read-only)

| record | required | read as |
|---|---|---|
| `<alias>_<timing>[_<TAG>]_MAP_Recovery/` (Evaluation) | yes | through the artifact schema (a superseded product is refused); MAP, posterior quantiles at 0.05 / 0.25 / 0.50 / 0.75 / 0.95, SGM, truth, `(task, sim)` identifiers, the checkpoint of the weights evaluated |
| `<alias>_<timing>[_<TAG>]_Posterior_Calibration/` (`.npz` + `report.md`) | no | the truth log-density per recording from the arrays; the joint tests, the SBC statistic and the location-versus-width diagnosis lifted verbatim from the report |
| `<alias>_<timing>[_<TAG>]_Embedding_Probe/` (`.json`) | no | the held-out MAE, null MAE, slope and correlation per parameter, and the weights probed |
| `<alias>_<timing>[_<TAG>]_Estimator.npz` | no | the encoder and flow settings of the rebuild specification and the weights checksum, for the column description |

Columns are `--control` (optional), `--base` (the reference of every difference) and `--candidates`
(any number, possibly none). `baseline` names the untagged estimator. A candidate whose Evaluation
product does not exist is listed as missing and not compiled; a missing base or control stops the run.

## Identity of a column

The records under one tag must come from one checkpoint. The Evaluation's checkpoint hash, the estimator
artifact's weights hash and the probe's weights hash, those present, are compared, and a mismatch stops the
run, so recovery from one checkpoint cannot be set beside architecture metadata or probe numbers of
another. The calibration product carries no checkpoint provenance; its identity is recorded as
**unverified** in the record, not implied to have been checked.

## Alignment

Every column is reordered to the base's `(task, sim)` order and the truths must agree exactly, or the run
stops. Each calibration cloud carries no identifiers and is aligned to the same recordings by the six true
parameters **rounded to nine decimal places**; the truths must be unique at that rounding (an ambiguity
stops the run), and the cloud must hold **exactly the Evaluation's set** of recordings, neither fewer nor
more. The calibration report is lifted only when the arrays are present and the report's recorded number
of calibration videos equals their rows; a report without arrays is recorded as present but not lifted,
since its coverage of the Evaluation's recordings cannot be verified. The products of record satisfy all
of this, 25,000 of 25,000.

## Computed

- **Recovery**, per parameter, column, view (MAP, posterior median, SGM) and regime: MAE, bias, the slope of
  the estimate on the truth, their correlation, RMSE, median error and the outside-prior fraction, in log10
  units. A constant estimate is scored with slope and correlation zero by convention (its correlation is
  undefined), so the flags read it as no recovery.
- **Marginal coverage and widths**, per parameter, column and regime, from the stored quantiles: coverage of
  the central 50 % and 90 % intervals, their gaps to the nominal level, the median interval widths in dex and
  the 90 % width as a share of the prior width.
- **Joint posterior quality**: mean and median truth log-density per column and regime, and the paired
  difference of a column against the base and against the control on the same recordings, with a percentile
  bootstrap interval and the share of recordings on which the column scores higher. The resample count
  (`--bootstrap`, default 2,000) and the seed (`--seed`) are recorded in the output; the interval measures
  the uncertainty of the paired mean over the evaluated recordings, not the variation between training runs.
- **Regimes**: every recording; the low and high halves of the `mu_r` prior and of the `sigma_r` prior,
  split at the prior midpoint in log10; the operating subgroup (the lower half of the `mu_pc` prior, as the
  acceptance rules of the direct estimators define it). The definitions are computed from the workflow's
  prior bounds and written to the record.
- **Failed estimates**: non-finite rows per view, non-finite scores, MAP outside the prior on any parameter.

## Lifted, not recomputed

From each calibration report: expected coverage (largest gap, the points at 0.50 and 0.90), TARP
area-to-curve, the L-C2ST rejection fraction and median p-value, the worst one- and two-dimensional
marginals, the dependence excess, the SBC KS statistic per parameter, and the diagnosis of location versus
width (bias z, spread z). L-C2ST needs a trained classifier, and the other tests come from the sample
clouds; copying the report keeps the scorecard consistent with the calibration record of each estimator.
The lifted values are the individual diagnostic results; the scorecard does not rank estimators by them.

## Flags (marks, not decisions)

From the median view over every recording, a column against the base and against the control:

| flag | condition | default threshold |
|---|---|---|
| below floor | correlation of the posterior median with the truth under `--collapse-corr` (unrecovered) | 0.3 |
| large correlation drop | correlation more than `--collapse-drop` below the reference column's | 0.3 |
| collapse | a large correlation drop that leaves the parameter below the floor | both |
| improved / worsened | MAE better / worse by at least `--min-effect` dex, with the 90 % coverage gap not worse by more than `--coverage-tolerance` for "improved" | 0.01 dex |
| points better, uncertainty worse | MAE better by at least `--min-effect` dex while the 90 % coverage gap worsens by more than `--coverage-tolerance` | 0.05 |
| under threshold | MAE difference under `--min-effect` dex | 0.01 dex |

The thresholds are recorded in the output. "Below floor" is the state of a parameter no estimator of record
recovers yet (`sigma_r`). A drop from 0.90 to 0.55 is a large correlation drop with useful recovery
remaining; "collapse" is reserved for a drop that leaves the parameter below the floor (the `capacity256`
bleaching precedent, 0.79 to 0.16). "Under threshold" is a statement about MAE alone: it says nothing about
calibration, width or joint density. The selection rule of the workflow document decides what the flags mean.

## Output

`<data_bank>/<posit>/<alias>_<timing>_Estimator_Scorecard_<COLUMNS>/` (the columns in order, upper case;
never overwritten; `--out-dir` writes elsewhere):

- `scorecard.md`: the columns with their settings and identity status, the recovery tables of the three
  views, marginal coverage, widths, the lifted joint tests, SBC and diagnosis, the truth log-density and
  paired differences, one table per regime, the encoder row, the flags, the failed estimates and a summary
  line per column;
- `scorecard.json`: every number, keyed by aspect, parameter, column, view and regime, plus the column
  descriptions (identity, lifted status), the regime definitions, the thresholds and the bootstrap settings;
- `scorecard.npz`: the aligned arrays (identifiers, truths, regime masks, per-column MAP / median / SGM /
  quantiles / truth log-density, prior bounds);
- `figures/recovery_and_coverage.png` and `figures/truth_log_density_paired.png` (lossless PNG);
- `PROVENANCE.md` (inputs with job identifiers and checkpoints, identity and lifted status per column, the
  estimator settings, alignment, regimes, what is computed and what is lifted, bootstrap settings,
  thresholds, code identity) and `README.md`.

## Usage

```bash
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Estimator_Scorecard.py \
    --total-time-seconds 2 --base CAP256 --control baseline \
    --candidates CAP256KERNEL7STATS CAP256EARLYCONVSTATS [--dry-run] [--out-dir DIR] [--posit DIR]
```

`--dry-run` resolves the inputs, reports which records exist and compiles nothing. `--posit` reads the
records from another folder (a copied data bank). The control compilation on the two estimators of record
(`--base CAP256 --control baseline`, no candidates) reproduces every recovery, coverage and width number of
the comparison record of the capacity test exactly; reproducing them is a check of the computation, and the
identity and coverage checks above are the checks of the inputs.

## Tests

`tests/test_estimator_scorecard.py`: synthetic, schema-valid products stored in different row orders are
aligned by identifier and the clouds by truth match; the recovery, coverage and paired log-density numbers
reproduce what the synthetic estimators were built to have; the flags fire on a collapsed parameter, on a
large drop that is not a collapse, on better points with worse uncertainty and on a difference under the
threshold; a probe or estimator artifact of other weights under a column's tag is refused; a calibration
product with a recording the Evaluation lacks, and a report whose count does not match its arrays, are
refused; absent records read as "not run" and a missing candidate is listed; the bootstrap settings are
recorded; the dry run writes nothing; an altered truth or a foreign recording is refused; an existing record
is not overwritten.
