# MAP optimization benchmark

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_MAP_Benchmark.py`, over the engine
`srm_and_sbi_monomer_dimer_alp.map_benchmark_runner` and the kernel
`srm_and_sbi_monomer_dimer_alp.map_benchmark`.

Resolving the MAP optimizer is a prerequisite for regenerating the analyses; the authoritative
protocol is `VALIDATION.md` §3.4. This benchmark compares optimizer settings on the real trained
flow, from identical starting candidates, so the production configuration is chosen from evidence.

**Status.** Run on ten recordings on the PC and on 2,000 on JUPITER (job 1970791). Configurations
`C` to `H` below, their loop and their scaling formula are a **benchmark proposal — not adopted**.
Production adopted a separate configuration in 0.1.17, defined in `VALIDATION.md` §3.4: steps in
units of the plain candidate-pool IQR, the existing joint loop, a 2,000-step budget with stopping
patience 200 and scheduler patience 20, and a 1e-3-nat tolerance. `B` always runs the production
optimizer as configured, so the runs recorded here measured the 0.1.16 absolute-step optimizer and
a later run measures the adopted one.

## What is compared

The objective is fixed: the unconstrained flow log-density in estimator coordinates, the production
objective. A returned point is a numerical MAP candidate; a prior-constrained objective is a
separate scientific decision.

Per recording, two independent bounded candidate pools of the production size are drawn, and the
production number of top candidates are the seeds. Within a pool every configuration starts from
the same seeds.

| config | setting |
|---|---|
| A | best candidate, no optimization |
| B | production `evaluation.optimize_elite` as configured (the recorded runs: 0.1.16, absolute step 0.128) |
| C | benchmark loop, absolute learning rate 0.128, 0.1.16 budget |
| D | benchmark loop, scale-aware learning rate 0.05, 0.1.16 budget |
| E | benchmark loop, scale-aware learning rate 0.05, extended budget |
| F | benchmark loop, absolute learning rate 0.128, extended budget |
| G | benchmark loop, scale-aware learning rate 0.02, extended budget |
| H | benchmark loop, scale-aware learning rate 0.2, extended budget |

The 0.1.16 budget is 1,000 steps, stopping patience 100 and scheduler patience 10. The extended
budget is 3,000 steps, stopping patience 300, scheduler patience 30, and no early stop within 50
steps after a learning-rate reduction (benchmark proposal — not adopted).

The benchmark loop (benchmark proposal — not adopted) runs each seed as its own chain, with its own
Adam state, learning rate, plateau schedule and stop. Every strictly better finite score–vector
pair is retained before the update, so the returned score is the density at the returned vector.
Patience counts steps since the last improvement larger than 1e-4 nats. Scale-aware chains move
`u = (theta - c) / s` with `s_i = min(W_i, max(IQR_i, 0.001 W_i))`, `c` the pool median, `IQR` the
pool's interquartile range and `W` the prior width. A learning rate of 0.05 then means about 5 % of
each parameter's posterior spread per step. The density is still maximized in theta.

All chains of many recordings run as one batch: one flow evaluation per step over every active
chain, each with its own conditioning embedding. Chains never interact, and the batched engine
reproduces a serial loop built on `torch.optim.Adam` and `ReduceLROnPlateau` chain by chain
(`tests/test_map_benchmark.py`).

The reference optimum per recording is the best vector found by any configuration on either pool,
polished with L-BFGS.

## What is reported

Density gain over the best starting candidate; gap to the reference; score–vector consistency;
stopping reasons and step counts; agreement between the two pools; path diagnostics (how far the
density drops and how far the chain travels before converging); scale safeguard activations; the
production optimizer's time; and, descriptively, recovery against truth overall and in the dim
subgroup.

Settings are chosen for optimization reliability and cost: gap to the reference, agreement between
pools, no budget hits, steps. They are not chosen for moving the MAP towards the median, and not
from the recovery table: a higher density does not imply a lower parameter error.

## Results

### Ten recordings on the PC

The ten deterministic pilot recordings, five dim and five bright, estimator of record:

- The 0.1.16 production optimizer improved on the best starting candidate in 35 % of pools and was
  within 1e-3 nats of the reference in 20 %.
- The mechanism is the absolute step. At 0.128 dex it is several times the posterior IQR of the
  well-identified parameters, so the first Adam steps carry a chain about eight IQRs away and the
  density drops by a median 55 nats. With a patience of 100, most chains do not recover before
  stopping, and the starting candidate is returned.
- Scale-aware steps remove the overshoot. At the 0.1.16 budget they reached the reference on every
  pool, within 4e-4 nats, in a median of about 150 steps. With the extended budget every scale-aware
  learning rate from 0.02 to 0.2 converges, as does the absolute step, and the two pools agree to
  within 0.004 IQR.

### 2,000 recordings on JUPITER

EVAL tasks 0 and 1 of the detector FAB 2 s tier (1,026 dim recordings), two pools each; 8 nodes,
32 GPUs, 10.5 minutes.

| | A, best candidate | B, 0.1.16 production | D, scale-aware, 0.1.16 budget | E, scale-aware, extended |
|---|---|---|---|---|
| pools within 1e-3 nats of the reference | 0 % | 21 % | 99.4 % | 100 % |
| gap to the reference, median / max (nats) | 0.165 / 2.10 | 0.058 / 0.40 | 0.0000 / 0.003 | 0.0000 / 0.0000 |
| two pools' largest coordinate difference, median / max (IQR) | 0.39 / 0.96 | 0.25 / 0.94 | 0.003 / 0.06 | 0.001 / 0.009 |
| steps per seed, median / 90th percentile | — | not recorded | 164 / 226 | 371 / 490 |

- Dim and bright recordings behave alike: `B` reached the reference in 24 % of dim and 17 % of
  bright pools, `D` in 99.6 % and 99.2 %. `D`'s few misses stopped 0.002 to 0.003 nats short.
- The absolute step also converges under the extended budget (`F`), but only after the density drops
  by a median 62 nats and the chain travels about seven IQRs. Every scale-aware rate from 0.02 to 0.2
  (`G`, `E`, `H`) converges; the density drop grows with the rate (median 0, 0.18 and 4.8 nats).
- No configuration hit its budget or produced a non-finite score. The pool IQR never came near zero,
  the smallest being 2.6 % of the prior width, so no scale safeguard activated in 24,000 scales.
- Distance of the 0.1.16 MAP from the converged point (`E`), per parameter: median 0.02 to 0.06 IQR,
  90th percentile about 0.25 IQR, maximum about one IQR. In dex the 90th percentile runs from 0.004
  (`mu_r`) to 0.08 (`prob_photo_bleach`, `lambda_rate`), the maximum reaches 0.29.
- Subset recovery error against truth changes by at most 0.004 dex between configurations
  (descriptive; not a selection criterion).

## How to run

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_MAP_Benchmark.py \
    --condition FAB --total-time-seconds 2.0 --tasks 0 1 --dry-run
```

On JUPITER the job script `Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_MAP_Benchmark.sh`
runs one rank per GPU and then `--merge`. A `--pilot N` run on another machine takes
`--data-bank-root` and `--output-dir`, as in the point-estimate validation.
