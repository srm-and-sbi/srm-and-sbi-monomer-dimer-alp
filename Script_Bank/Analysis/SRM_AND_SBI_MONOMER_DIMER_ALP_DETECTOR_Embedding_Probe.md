# Embedding probe of a frozen detector encoder

Companion note of `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Embedding_Probe.py`, the diagnostic that opens
the encoder screening (`DETECTOR_WORKFLOW.md`, the section on the encoder screening). It answers one
question about an estimator that already exists: how much of each imaging parameter is linearly
accessible in the embedding its trained `Complex3DCNN` produces, before the flow sees it.

## What it does

1. Loads the persisted detector estimator of the label given (the canonical `..._2S_50FPS_Estimator.npz`,
   or a tagged one such as `CAP256` through `--artifact-tag`) with the artifact loader the stages use, so
   the encoder, its weights and the video preprocessing are exactly the trained ones.
2. Embeds the synthetic EVAL videos of the tasks named (`embedding_space_distance.embed_videos`, the same
   pass the embedding-space distance analysis uses) and reads their generating parameters from the
   matching theta sets. Targets are the log10 of the physical values, as the Evaluation stage's recovery
   metrics.
3. Fits one ridge regression per parameter from the standardized embedding to the target on the fit set,
   selects its penalty on a development set, refits on fit + development and scores the held-out set.
   The split is by task (`--fit-tasks`, `--dev-tasks`, `--held-out-tasks`); with no development tasks, the
   last `--dev-fraction` of each fit task's videos, by simulation index, is the development set: held-out
   scoring is then task-disjoint and fit versus development simulation-disjoint. Each set must hold at
   least 20 videos (`MIN_VIDEOS`), or the probe refuses before fitting; a regime half of fewer than two
   videos reports its count with NaN statistics.
4. Reports, per parameter, the held-out mean absolute error in dex, the bias, the slope of predicted on
   true, the correlation, the error of predicting the refit set's mean (`null`), the span of the true
   values, the penalty chosen, and the error in the low and high halves of the true values (median
   split, same predictions), so an improvement can be read across the prior rather than in one region.

Outputs go to `<data_bank>/<posit>/<alias>_<timing>[_<TAG>]_Embedding_Probe/`: the report (`.md`), the
numbers (`.json`), the embeddings with their theta and split (`.npz`) and a predicted-versus-true figure.
An existing folder is never overwritten. `--dry-run` resolves the estimator, the tasks and the output and
embeds nothing.

## How to read it

- A parameter the probe predicts well (an error clearly below `null`, a slope near 1, a correlation
  near 1) is linearly accessible in the embedding. If the estimator of record recovers it poorly, the
  loss lies between the embedding and the posterior (the flow, its training), not in the encoder.
- A parameter the probe predicts poorly is **not** shown to be absent from the embedding: the
  information may be present but not linearly accessible. The probe is a diagnostic, not a gate.
- The probe fits a point predictor. A successful probe establishes accessible predictive information,
  not a complete posterior representation; calibration and joint quality are read from the Evaluation
  and calibration stages of a trained estimator.
- Videos previously inspected in an Evaluation are development evidence. A verdict on an estimator
  comes from its own Evaluation on the reserved split, not from this probe.

Nothing is adopted by running it: the estimator, the flow and the training data are untouched.

## Usage

```bash
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Embedding_Probe.py \
    --total-time-seconds 2 --fit-tasks 0 --held-out-tasks 1 --dry-run
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Embedding_Probe.py \
    --total-time-seconds 2 --fit-tasks 0 --held-out-tasks 1 --artifact-tag CAP256
```

A GPU pass. In inference mode no activations are kept for a backward pass, so the device memory is set by
the batch (`--batch-size`, default 16). Each task of 1,000 videos is embedded once and kept in the `.npz`.
The machine must hold full EVAL tasks: the PC's sets are smoke-sized, rcl01 holds tasks 0 and 1, and the
25 tasks of the split live on JUWELS and JUPITER.
