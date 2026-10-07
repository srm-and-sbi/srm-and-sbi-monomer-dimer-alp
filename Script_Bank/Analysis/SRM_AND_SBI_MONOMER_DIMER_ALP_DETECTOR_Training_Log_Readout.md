# Training-log readout (detector)

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Training_Log_Readout.py`, the reader of the Inference
stage's logs for the encoder screening of `DETECTOR_WORKFLOW.md`. It takes any set of training logs and
writes the tables a screening is read from: one row per run, one per chain of legs, one per configuration.
It computes and lists; the reading and the selection are made in the workflow document, under the
screening and selection guidance written there. A screening is then read from its evidence, the logs kept
in a record folder of the data bank, and not from memory.

## What it reads

An Inference log is the HPC job's output, named `<alias>[_<TAG>]_Inference_<jobid>.out` by the dispatcher.
The tool reads, per log:

| line | read as |
|---|---|
| the header `=== Inference \| train_tasks=… test_tasks=… epochs=… batch=… global_batch=… resurrect=… preset=… tag=… nodes=… gpus_per_node=… world_size=…` | the run's settings |
| `Epoch k\|K (global g)  train=…  test=…  lr=…  epoch=…s  elapsed=…  peak_mem=a/bGiB` | one epoch: the TRAIN and TEST losses, the learning rate, the epoch time, the peak memory of rank 0; `(global g)` is present in a `--resurrect` leg and continues the first leg's count |
| `[new best] committed live artifacts at epoch k (test loss v)` | the epochs at which the live artifacts were replaced |
| `WARM RESTART: …` | one restart of the learning-rate schedule from the best checkpoint |
| `Total elapsed: …s` | the wall time of the training (the largest value over the ranks) |
| `=== Inference complete ===`, `Traceback (most recent call last)` | completion, and failures |

Lines repeated by several ranks are counted once (epochs by their global number). A log without the
header line, from a version that did not print it, is read from its settings block (`--network-preset`,
`--artifact-tag`, `--global-batch`, `--resurrect`, `--batch-size`, `--epochs`). A global batch the log
does not state is derived as the per-rank batch times the rank count, the value of a training without
gradient accumulation, and the derivation is recorded in the run's row; when neither is known the field
stays empty and the run forms its own configuration.

## What it writes

Under `--out-dir`, refused when a `readout.md` already exists there unless `--overwrite` is given:

| file | content |
|---|---|
| `readout_runs.csv` | one row per log: settings, epochs run, first and last global epoch, best TEST loss and the global epoch that reached it, last TEST and TRAIN losses, new-best events, warm restarts, learning-rate floor, wall time, peak memory, completion, tracebacks, smoke flag |
| `readout_chains.csv` | one row per artifact tag: the legs (a first leg and its `--resurrect` continuations) read together: jobs, epochs covered, the best TEST loss over the legs and where it was reached, each leg's best, warm restarts, completion, and whether the legs agree on preset and global batch |
| `readout_configurations.csv` | one row per preset and global batch (within one condition and timing): the number of chains N, the best chain and its value, every chain's best, their spread (largest minus smallest), the epochs each chain covered |
| `readout.json` | all of the above with every parsed epoch, for machines |
| `readout.md` | the three tables with the inputs, the exclusions and the reading notes |

The TEST loss is the mean negative log-probability of the parameters on the TEST videos, as the Inference
stage logs it; lower is better; the best is the minimum over the epoch lines. Smoke runs (tags beginning
with `SMOKE`) and logs without an epoch line are listed among the runs and excluded from the chains and
the configurations; `--include-smokes` keeps the smokes. A job id that appears twice is refused.

## How to read it

- **A configuration is read by its best chain, with N beside it.** Training is stochastic: replicate
  chains of one configuration, same data, priors, protocol and global batch, ended 0.9 to 2.6 nats apart
  in the screening of 2026-10-02 to 2026-10-07. The spread column is that context; it is not a significance
  test, and a difference between two configurations smaller than their spreads is not a result. Best-of-2
  against best-of-1 is not like for like; the N column says which comparison is being made.
- **The best TEST loss of a chain is the loss of a selected checkpoint.** The same split chose it and
  scores it, so the number is optimistic by construction. The clean number for a chosen estimator comes
  from the Evaluation and Posterior_Calibration stages on the held-out EVAL videos (the scorecard); this
  readout screens, it does not select.
- **"Plateaued under this schedule", not "converged".** The learning rate reaching its floor and the warm
  restarts are properties of the schedule (a per-epoch plateau rule on the noisy TEST loss, then restarts
  from the best checkpoint), not evidence that the optimum was reached; `PROJECT_CONTEXT.md` §8 records
  the schedule's contribution to the replicate spread as an open question.
- **What carried the screening, and holds only under its data, prior, loss and schedule:** one run cannot
  separate an architecture from the run-to-run variation; the global batch defines the optimization, the
  node count only its speed, so configurations are compared at one global batch (the workflow document,
  "Nodes for speed, a fixed global batch for comparability"); a smoke run says nothing about a loss.
- **What it is not.** Not the scorecard: it reads logs, not products, and it says nothing about recovery,
  calibration or posterior width. Not a comparison across conditions, timings, data tiers or priors: a
  configuration key holds the condition and the timing, and the data and priors behind a log are the
  tier's, which the log does not restate.

## Usage

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Training_Log_Readout.py \
        --logs <directory or log files> --out-dir <directory> [--include-smokes] [--overwrite] [--dry-run]

The logs of a screening are kept, copied once from the HPC system and never edited, under
`<data_bank>/<posit>/<alias>_<condition>_<timing>_Encoder_Screening_Record/evidence/logs/`, with the job
accounting beside them and the readout written at the record's root, so that the tables are regenerated
from the evidence. `--dry-run` lists the logs and their settings and writes nothing. The test is
`tests/test_training_log_readout.py` (synthetic logs; text only).
