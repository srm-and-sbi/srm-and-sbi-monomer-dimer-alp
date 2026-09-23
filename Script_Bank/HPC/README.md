# HPC Operations Runbook — srm-and-sbi-monomer-dimer-alp

The single authoritative reference for running the DIMER pipeline on a Slurm
cluster. The batch scripts in this directory are generic and committed; each
machine supplies its own values through a gitignored `hpc_local.env`, so the
same scripts run unchanged on any cluster.

This runbook describes what the scripts in `Script_Bank/HPC/` actually do.
Treat the `#SBATCH` blocks and header examples inside those scripts as the
ground truth and replicate them; this document organizes and explains them.

---

## 1. Stages and partitions

The pipeline runs as a sequence of stages, each driven by one batch script
here. Each script activates the `SRM_AND_SBI_ENVY_V0` conda environment and
runs the matching entry point under `Script_Bank/Prime/`.

| Stage | Script | Compute | Production partition | Check partition |
|-------|--------|---------|----------------------|-----------------|
| Simulation (RDS → DLI; `SIM_STAGE=rds` generates the condition's trajectory tier alone) | `SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Simulation.sh` | CPU | `general1` | `test` |
| Inference (train + select) | `SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Inference.sh` | GPU | `gpu` | `gpu_test` |
| Evaluation (MAP recovery) | `SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Evaluation.sh` | GPU | `gpu` | `gpu_test` |
| Experiment (real videos) | `SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Experiment.sh` | GPU | `gpu` | `gpu_test` |

- **Simulation** packs many generation tasks per node (RDS reaction-diffusion
  trajectories — the condition's tier, one per condition because the association setting
  is per condition, shared by both workflows — then the biology's DLI diffraction-limited
  videos; `SIM_STAGE=rds` generates the tier alone, `SIM_STAGE=dli` re-images the
  condition's existing tier) and is CPU-bound.
- **Inference** trains the posterior on the TRAIN tasks and selects on the
  TEST tasks. It adapts to the allocation: more than one **node** trains
  data-parallel *across nodes* (`srun` + `torchrun`, one process per GPU, all ranks
  bound by a c10d rendezvous); one node with more than one GPU trains data-parallel
  on that node (`torchrun --standalone`); one GPU uses the single-GPU path.
- **Evaluation** estimates the maximum-a-posteriori parameter vector on the
  held-out EVAL set and reports per-parameter recovery accuracy and posterior
  calibration. It shards EVAL across one Slurm task per GPU on every allocated node
  (`srun --ntasks-per-node=$GPUS`, no torchrun), each writing its own shard,
  then merges the per-shard results into one report; one GPU uses the single-GPU path.
- **Experiment** applies the trained estimator to real microscopy `.tif`
  recordings (no ground truth) and reports the inferred-parameter distribution
  per condition. It shards its `(kind, cell)` work across one Slurm task per GPU on
  every allocated node (`srun --ntasks-per-node=$GPUS`, no torchrun), then
  merges the per-shard results into one report. It is the scientific end use, not
  a correctness check.

The normal ordering is **Simulation → Inference → Evaluation**. Experiment runs
once a trained posterior exists. Generation runs on the CPU partition; training,
evaluation, and the real-data application run on the GPU partition.

**Durations are general.** The codebase is duration-parameterized: the recording
length is supplied per run via `--total-time-seconds`, and the frame count
follows as `frame_count = total_time_seconds / 0.020 s` (50 frames per second).
Specific durations below (2 s, 5 s, 1 s, 10 s) are concrete examples, not a
fixed pairing; substitute the duration the campaign calls for.

---

## 2. Submission recipe

Slurm spools a copy of each batch script into `/var/spool` before running it, so
the script's own path is unreliable for a directly-submitted job. Every script
resolves the repository root with a `_find_repo` helper that tries, in order:

1. an explicit `REPO` (e.g. `--export=ALL,REPO=$PWD,...`),
2. `SLURM_SUBMIT_DIR` (and its grandparent),
3. the script's own location (for a non-Slurm `bash <script>` invocation),

and accepts the first candidate that actually contains `pyproject.toml` and the
`srm_and_sbi_monomer_dimer_alp/` package. If none match, the script **fails loud** with
guidance rather than crashing on a `/var/spool` path.

**The rule:** submit from the repository root, or pass `REPO` explicitly via
`--export`. Either makes `REPO` resolvable.

```bash
cd /path/to/srm-and-sbi-monomer-dimer-alp        # so SLURM_SUBMIT_DIR resolves the repo
# or, from anywhere, add REPO=$PWD (run from the root) to --export
```

The same resolved `REPO` is used to source the per-machine `hpc_local.env` (see
§5), so it is found even under spooling.

### Dry-run first: the submit helper

Always preview a submission before it reaches the queue. The unified
`SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Submit.sh` builds the exact `sbatch` command for any
stage — the resolved `REPO`, the data-pattern `--job-name` (with the rendered
`timing_label`), and a comma-split-safe `--export` — and **prints it without
submitting** unless you set `DRYRUN=0`. Because the recipe, the naming, and the
config are built by the tool, they cannot be mistyped at submit time.

```bash
# Dry run (the default): print the exact sbatch line, submit nothing
bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Submit.sh inference CONDITION=FAB TOTAL_TIME=5.0 TRAIN_TASKS=400 TEST_TASKS=100 EPOCHS=25
# Submit it (only after the printed command checks out)
DRYRUN=0 GPU_PART=gpu bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Submit.sh inference CONDITION=FAB TOTAL_TIME=5.0 TRAIN_TASKS=400 TEST_TASKS=100 EPOCHS=25
```

`<stage>` is `simulation | inference | evaluation | experiment`; the `KEY=VALUE`
pairs are that stage's `--export` knobs (anything omitted falls back to the
stage script's own default). sbatch-level overrides go in the environment:
`PART` (CPU partition — required for a live `simulation`, whose baked
`--partition` is a placeholder), `GPU_PART`, `ACCT`, `TIME`,
`ARRAY`/`NTPN`/`CPT` (simulation only), `GRES`, `NODES` (GPU stages — multi-node,
`--gres` is per node so `NODES=2 GRES=gpu:4` → `world_size` 8), `MON_OUT`. `CONDITION` (`FAB`|`INLB`; required for every stage, the RDS stage included, because the
trajectory tier is per condition) is validated and
forwarded by the dispatcher and enters the job name right after the alias. A multi-value `KINDS`
(e.g. `KINDS=FAB,INLB`, a deliberate cross-condition application) is carried safely through the exported environment via
`ALL` rather than the comma-split `--export`. The helper is to a single job what
the generation controller (§5) is to the full generation campaign — both default
to dry-run, and you opt in to submitting with `DRYRUN=0`.

For a **local** run — or as a final configuration check before any submission —
pass `--dry-run` to the Prime entry point itself: it resolves the machine
profile and the input paths, prints what it would read and write (flagging
anything MISSING), and exits before any compute (no GPU, no data load).

```bash
MACHINE_PROFILE=<profile> python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Inference.py \
    --total-time-seconds 2.0 --tasks 8 --test-tasks 2 --epochs 50 --dry-run
```

The raw `sbatch` invocations below are the ground truth the helper generates;
use them directly when you want full manual control.

### Runnable examples (one per stage)

Submit from the repository root. Set `--partition` to the partition for the run
(production vs check, per §1); the baked `#SBATCH --partition` is a placeholder
on the Simulation script and `gpu` on the GPU scripts.

```bash
# Simulation — the FAB trajectory tier alone (SIM_STAGE=rds CONDITION=FAB; one tier per condition, shared by
# both workflows; the INLB tier is a second submission with CONDITION=INLB and the _INLB_ job-name token),
# TRAIN split, one node, 8 packed tasks (always submit with --array)
cd /path/to/srm-and-sbi-monomer-dimer-alp
sbatch --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Simulation_TRAIN \
       --partition=general1 --array=0-0 --ntasks-per-node=8 \
       --output="$MON_OUT/%x_%A_Node_%a.out" \
       --export=ALL,REPO=$PWD,SIM_STAGE=rds,CONDITION=FAB,SPLIT=train,TASK_OFFSET=0,TASK_COUNT=8,TASK_SIMS=1000,TOTAL_TIME=2.0 \
       Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Simulation.sh
# Simulation — RDS then the biology DLI of one condition in one job (SIM_STAGE=both, the default;
# the biology DLI needs that condition's Nuisance_DLI artifact)
sbatch --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Simulation_TRAIN \
       --partition=general1 --array=0-0 --ntasks-per-node=8 \
       --output="$MON_OUT/%x_%A_Node_%a.out" \
       --export=ALL,REPO=$PWD,CONDITION=FAB,SPLIT=train,TASK_OFFSET=0,TASK_COUNT=8,TASK_SIMS=1000,TOTAL_TIME=2.0 \
       Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Simulation.sh
```

```bash
# Inference — train on 8 TRAIN tasks, select on 2 TEST tasks, 50 epochs
cd /path/to/srm-and-sbi-monomer-dimer-alp
sbatch --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Inference \
       --partition=gpu \
       --output="$MON_OUT/%x_%j.out" \
       --export=ALL,REPO=$PWD,TRAIN_TASKS=8,TEST_TASKS=2,EPOCHS=50,TOTAL_TIME=2.0 \
       Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Inference.sh
# Continue a wall-stopped run from its checkpoint: add RESURRECT=1 to the --export
# (or pass RESURRECT=1 to the Submit.sh helper). The first job runs fresh.
```

```bash
# Evaluation — MAP recovery on the held-out EVAL set
cd /path/to/srm-and-sbi-monomer-dimer-alp
sbatch --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Evaluation \
       --partition=gpu \
       --output="$MON_OUT/%x_%j.out" \
       --export=ALL,REPO=$PWD,EVAL_TASKS=1,POOL_MODE=bounded,TOTAL_TIME=2.0 \
       Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Evaluation.sh
```

```bash
# Experiment — apply the trained posterior to real videos.
# KINDS defaults to the run's CONDITION (baked in the script). Do NOT place a multi-value
# KINDS inside --export: Slurm splits --export on commas, so KINDS=FAB,INLB would
# parse as KINDS=FAB plus a stray, value-less INLB. To override KINDS with multiple
# values, pre-export it in the submitting shell and let --export=ALL carry it
# (export KINDS=FAB,INLB) — or just use the Submit.sh helper, which does this for you.
cd /path/to/srm-and-sbi-monomer-dimer-alp
sbatch --job-name=SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Experiment \
       --partition=gpu \
       --output="$MON_OUT/%x_%j.out" \
       --export=ALL,REPO=$PWD,TOTAL_TIME=2.0 \
       Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Experiment.sh
```

`MON_OUT` is your monitoring/batch-log directory and **must already exist**
(see §5).

---

## 3. Job and log naming

Job names mirror the theta/video data files so a batch log and the artifacts it
produces share one provenance string. The data files are named, for example,
`SRM_AND_SBI_MONOMER_DIMER_ALP_2S_50FPS_Video_Set_TASK_0_TRAIN.zarr` and
`SRM_AND_SBI_MONOMER_DIMER_ALP_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr`: the pattern is
`{project_alias}_{timing_label}_<descriptor>`.

- **`project_alias`** = `SRM_AND_SBI_MONOMER_DIMER_ALP`
- **`timing_label`** = `<duration>S_<FPS>FPS`, placed **immediately after the
  alias** (e.g. `1S_50FPS`, `2S_50FPS`, `5S_50FPS`). No `HPC` token.

Job names follow the same shape, with the condition slot (`FAB` or `INLB`) right after the
alias — every stage carries it, the RDS-only simulation included, because the trajectory
tier is per condition:

```
SRM_AND_SBI_MONOMER_DIMER_ALP_<CONDITION>_<timing_label>_<Stage>[_<SPLIT>]
```

| Stage | Job name |
|-------|----------|
| Simulation | `SRM_AND_SBI_MONOMER_DIMER_ALP_<CONDITION>_<timing_label>_Simulation_<SPLIT>` (SPLIT = `TRAIN`/`TEST`/`EVAL`) |
| Inference | `SRM_AND_SBI_MONOMER_DIMER_ALP_<CONDITION>_<timing_label>_Inference` |
| Evaluation | `SRM_AND_SBI_MONOMER_DIMER_ALP_<CONDITION>_<timing_label>_Evaluation` |
| Experiment | `SRM_AND_SBI_MONOMER_DIMER_ALP_<CONDITION>_<timing_label>_Experiment` |

Examples: `SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_5S_50FPS_Inference`,
`SRM_AND_SBI_MONOMER_DIMER_ALP_INLB_1S_50FPS_Simulation_TRAIN`.

Set the job name with `--job-name` and direct the batch log into the monitoring
directory:

- Simulation (an array, one element per node): `--output="$MON_OUT/%x_%A_Node_%a.out"`
- All other stages: `--output="$MON_OUT/%x_%j.out"` (`%j` = the job id. `%A`, the array master id, resolved to 0 for a non-array job on JUPITER's Slurm, so two same-name jobs would have shared and truncated one log.)

`%x` is the job name, `%A` the array job id, `%a` the array element (the node
number). The Simulation array element is always a clean node number because the
job is always submitted with `--array` (`--array=0-0` for a single node);
without `--array`, Slurm sets `%a` to its not-an-array sentinel.

Inside the Simulation node, each packed task additionally writes its own per-task
log: `${MON}/${job_name}_${job_tag}_Node_${array_id}_Task_${tid}.out`.

---

## 4. Hardware configurations to replicate

Use the layouts encoded in the scripts' `#SBATCH` blocks and header examples.
Do not recompute node geometry — replicate these.

### Simulation (CPU)

The baked layout packs tasks per node and pins the core geometry:

- `--nodes=1`, `--ntasks-per-node=8`, `--cpus-per-task=5`, `--mem-per-cpu=4400`
- `--extra-node-info=2:20:1` (40-core nodes: 2 sockets × 20 cores × 1 thread)
- `--time=08:00:00`

Always submit with `--array` (one element per node; `--array=0-0` for a single
node). Multi-node scaling is an `--array` of single-node jobs; `TASK_OFFSET`
shifts the global task index so a later submission appends tasks rather than
regenerating existing ones. The per-task global index is
`tid = TASK_OFFSET + SLURM_ARRAY_TASK_ID * ntasks_per_node + k`.

**Production (`general1`), 1000 sims/task, one node per split:**

```bash
# TRAIN 8 / TEST 2 / EVAL 1 tasks (per CORE=100), submit from the repo root:
sbatch --partition=general1 --array=0-0 --ntasks-per-node=8 --export=ALL,REPO=$PWD,CONDITION=FAB,SPLIT=train,TASK_OFFSET=0,TASK_COUNT=8 ...Simulation.sh
sbatch --partition=general1 --array=0-0 --ntasks-per-node=2 --export=ALL,REPO=$PWD,CONDITION=FAB,SPLIT=test,TASK_OFFSET=0,TASK_COUNT=2  ...Simulation.sh
sbatch --partition=general1 --array=0-0 --ntasks-per-node=1 --export=ALL,REPO=$PWD,CONDITION=FAB,SPLIT=eval,TASK_OFFSET=0,TASK_COUNT=1  ...Simulation.sh
```

Larger campaigns pack 10 tasks/node with `--cpus-per-task=4` (10 × 4 = 40 cores,
matching `--extra-node-info=2:20:1`) and fan out over `--array`; the generation
controller (§5) drives this. Its per-case wall time is 24 h for every 2 s and 5 s array
(the JUWELS `batch` maximum). On JUPITER every submission is 12 h, pinned by
`export TIME=12:00:00` in that machine's `hpc_local.env` (the booster partition allows
nothing else).

**Check (`test`), 1 s smoke — TRAIN 16 / TEST 4 / EVAL 2 tasks, 10 sims/task:**

```bash
sbatch --partition=test --array=0-0 --ntasks-per-node=16 --export=ALL,REPO=$PWD,CONDITION=FAB,SPLIT=train,TASK_COUNT=16,TASK_SIMS=10,TOTAL_TIME=1.0 ...Simulation.sh
sbatch --partition=test --array=0-0 --ntasks-per-node=4  --export=ALL,REPO=$PWD,CONDITION=FAB,SPLIT=test,TASK_COUNT=4,TASK_SIMS=10,TOTAL_TIME=1.0  ...Simulation.sh
sbatch --partition=test --array=0-0 --ntasks-per-node=2  --export=ALL,REPO=$PWD,CONDITION=FAB,SPLIT=eval,TASK_COUNT=2,TASK_SIMS=10,TOTAL_TIME=1.0  ...Simulation.sh
```

### Inference / Evaluation / Experiment (GPU)

The GPU scripts bake a single-node default that scales out with `NODES`:

- `--nodes=1` (default; override with `NODES=N` via `Submit.sh`, or `--nodes=N` on
  a raw `sbatch`), `--gres=gpu:8` (per node), `--cpus-per-task=64`, `--mem=480G`
- `--time=1-00:00:00` (Inference, Experiment); `--time=12:00:00` (Evaluation)

Inference, Evaluation, and Experiment adapt to the allocation. `--gres` is per
node, so `--nodes=N --gres=gpu:G` gives `world_size = N*G` ranks:

- **Inference, more than one node**: `srun` places one `torchrun` launcher per node,
  all ranks bound by a c10d rendezvous, and training runs data-parallel across every
  rank on every node (SyncBatchNorm + the loss all-reduced across all ranks, so
  `--batch-size` stays per-rank and the effective batch is `batch*world_size`).
  **One node, more than one GPU** uses `torchrun --standalone`.
- **Evaluation and Experiment, more than one GPU** (one node or many): plain Slurm
  tasks, one per GPU on every allocated node (`srun --ntasks-per-node=$GPUS`), each
  writing its own shard to the shared filesystem, then a single `--merge` pass
  combines the shards. No torchrun and no rendezvous: `resolve_topology()` reads the
  rank from `SLURM_PROCID`, so nothing waits on anything and a slow rank can never
  be killed by a fast one (see the straggler note below).
- **One GPU** runs the original single-GPU path.

The GPU count per node is `SLURM_GPUS_ON_NODE`, capped by the optional
`SRM_AND_SBI_GPUS`; the node count is `SLURM_NNODES` (from `--nodes`). Use the full
`gpu` partition (whole node, `gpu:8`) for validation unless a run is a genuinely
tiny throwaway. Multi-node Evaluation/Experiment need the output directory on a
shared filesystem so the merge sees every node's shards (the case on the HPC data
banks).

**Wall-limited training continues with `RESURRECT`.** Training checkpoints its
optimum each epoch. A run that will not reach its target epochs within the
partition wall is continued by relaunching the same submission with `RESURRECT=1`
(a `Submit.sh` knob, or `RESURRECT=1` in the raw `--export`): it loads that
checkpoint and resumes. The first job runs fresh; every continuation passes
`RESURRECT=1`. Repeat until the target epochs are reached. This works on the
single-GPU, single-node data-parallel, and multi-node paths alike.

To pre-submit the whole chain so each link starts **even if the previous job hit
the wall** (a timeout is recorded as a failure), gate each continuation on the
previous job id with `DEP=afterany:<jobid>` — `afterany`, not `afterok`, because an
`afterok` successor would never start after a wall-stopped predecessor:

```bash
id=$(DRYRUN=0 GPU_PART=gpu_test GRES=gpu:4 bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Submit.sh \
       inference CONDITION=FAB TOTAL_TIME=5.0 TRAIN_TASKS=100 EPOCHS=10 HEARTBEAT=20 | grep -oP 'Submitted batch job \K\d+')
for _ in 1 2; do   # two continuations
  id=$(DRYRUN=0 GPU_PART=gpu_test GRES=gpu:4 DEP="afterany:$id" bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Submit.sh \
         inference CONDITION=FAB TOTAL_TIME=5.0 TRAIN_TASKS=100 EPOCHS=10 HEARTBEAT=20 RESURRECT=1 | grep -oP 'Submitted batch job \K\d+')
done
```

**Check (`gpu_test`), single-GPU smoke — Inference:**

```bash
sbatch --partition=gpu_test --gres=gpu:1 --time=01:00:00 \
       --export=ALL,REPO=$PWD,CONDITION=FAB,TRAIN_TASKS=8,TEST_TASKS=2,EPOCHS=1 ...Inference.sh
```

**Long durations need a smaller batch.** Leaving `BATCH` unset uses the script
default (`PARAMETERS` batch size, 32). One early conv3d activation is
batch × ~1 GiB at 500 frames, so batch 32 OOMs a GPU with ~64 GB of VRAM at long durations
(e.g. a 10 s, 500-frame run); reduce `BATCH` accordingly.

**Starting learning rate: `LR`.** Both Inference wrappers (biology and detector)
forward a non-empty `LR` to the entry point as `--learning-rate`, the run's
starting (peak) learning rate; left unset, the entry point's default peak applies
(`learning_rate_minimum × max_factor` = 1.28e-03). Useful when continuing a
wall-stopped run (`RESURRECT=1`) at a chosen rate.

**Stragglers and the merge guard.** The sharded stages (Evaluation, Experiment, the
`Posterior_Calibration` and DETECTOR `Nuisance_DLI` wrappers, §6) are embarrassingly
parallel and routinely skewed: the FAB detector Evaluation's 16 ranks finished 18
minutes apart. They therefore run as plain Slurm tasks, never through `torchrun`,
whose elastic agent enforces a 300 s exit barrier that no launcher setting changes
(torch 2.9 ignores `TORCHELASTIC_EXIT_BARRIER_TIMEOUT`); under torchrun the first
node's agent tore the rendezvous down after five minutes and the other nodes' agents
killed their still-working ranks, losing their shards. With Slurm tasks nothing waits
on anything. Should a rank still die (node failure, wall time), `--merge` refuses the
incomplete shard set and names the missing ranks; recompute just that rank with
`RANK=<r> WORLD_SIZE=<n> LOCAL_RANK=0 SRM_AND_SBI_INVOCATION_ID=<the launch's id> python
<stage>.py <same args>` on one GPU (the video-level sharding is deterministic; the id is printed
by the stage script at launch and stored in every shard's manifest) and merge again. The
replacement may run in a new Slurm job or locally: each shard records the execution attempt that
wrote it (its own job id, or none), the merged manifest keeps every shard's, and the merge never
compares them. "The same args" includes `--dump-posterior-samples` exactly when the original run
used it: raw-draw storage is a run-level setting, and shards that differ in it are refused. The
replacement must also run from the same repository state: the same HEAD commit, the same
dirty-state flag and byte-identical implementation files, because the manifest's whole code block
is part of the contract. A commit made between the original run and the replacement makes the
merge refuse even when no implementation file changed. Before recomputing rank 0, copy
`progress.log` and `figures/` out of the stage's output directory: when rank 0 starts it
truncates the shared progress log, which holds the original run's per-window progress and, under
`--debug`, its optimizer trace, and it clears `figures/`. The other ranks overwrite neither. For
Evaluation and Experiment there is no partial merge: every shard is validated against the
artifact schema, the shard manifests must describe one computation (same settings, stored optional
arrays, seed policy, window geometry, condition labels, checkpoint, implementation hash, product
label AND invocation id, so a stale shard from an earlier launch is refused even inside the same
job), and the merged observation
set must equal the expected inventory exactly (every `(task, sim)` of the EVAL tasks; every
`(kind, cell, chunk)` a recording on disk yields), so a missing rank is always a hard stop. A rank
that drew no work (more GPUs than cells, or only absent recordings) writes a valid empty shard,
so it never reads as a dead rank. The Evaluation and Experiment stage scripts create the
invocation id (`SRM_AND_SBI_INVOCATION_ID`) before `srun` starts the ranks; a rank launched
without it stops rather than inventing its own. `--allow-partial` survives only on the
`Posterior_Calibration` and `Nuisance_DLI` wrappers, whose shards are not products of that
contract (their reports then record `shards_merged`).

**Smoke / check evaluation uses `--pool-mode unrestricted`.** An undertrained
posterior's probability mass can fall outside the prior box, and the default
`bounded` rejection sampling stalls on it. Pass `POOL_MODE=unrestricted` for any
smoke or undertrained-posterior evaluation; `bounded` is only for a fully
trained posterior. (See the MAP-recovery validation in `VALIDATION.md`.)

```bash
sbatch --partition=gpu_test --gres=gpu:1 --time=01:00:00 \
       --export=ALL,REPO=$PWD,CONDITION=FAB,EVAL_TASKS=1,POOL_MODE=unrestricted,TOTAL_TIME=1.0 ...Evaluation.sh
```

---

## 5. Generation controller and per-machine config

### Generation controller

`SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Generate_Controller.sh` is a rolling submit-and-gate
controller for a full generation campaign. It submits the per-split generation
arrays (train first), keeps within the QOS caps (≤40 running, ≤50 in-system),
then **hard-gates** the EVAL splits until every TRAIN + TEST job has reached
`COMPLETED`. If any train/test job ends in a non-`COMPLETED` state it stops
before submitting EVAL, so EVAL is never generated against broken data.

- **Dry run is the default.** With no override it prints the exact `sbatch`
  lines and submits nothing:

  ```bash
  CONDITION=FAB bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Generate_Controller.sh
  ```

- **Live submission** requires `DRYRUN=0`. It polls for hours to days, so run it
  on the login node inside `tmux`/`screen`:

  ```bash
  DRYRUN=0 CONDITION=FAB bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Generate_Controller.sh 2>&1 | tee ~/dimer_gen_controller.log
  ```

- `CASES` selects which dataset(s) to drive (`5s` | `2s` | `both`, default
  `both`), so one controller can run separate campaigns on separate clusters.
- Re-run a failed node/task under its **original global label** (same `SPLIT` /
  `TASK_SIMS` / `TOTAL_TIME`, with `TASK_OFFSET`/`TASK_COUNT` set to the gap):
  this fills the gap via the incremental-append mechanism and regenerates
  nothing good. Confirm label completeness before training.

### Per-machine config: `hpc_local.env`

Every script in this directory sources `Script_Bank/HPC/hpc_local.env` at
startup (via the resolved `REPO`, so it is found even under spooling). The file
is **gitignored** and never committed; each machine supplies its own values,
which keeps the committed scripts generic. Create it from the template:

```bash
cp Script_Bank/HPC/hpc_local.env.example Script_Bank/HPC/hpc_local.env
# then edit for this machine
```

Settings it provides:

| Variable | Purpose |
|----------|---------|
| `MACHINE_PROFILE` | profile in `machine_profiles.toml` (selects this machine's data/compute paths) — **required**; the scripts fail loud if unset |
| `CONDA_SETUP` | path to the conda `profile.d/conda.sh` that defines `conda` in a non-interactive shell |
| `MON` / `MON_OUT` | monitoring / batch-log output directory (**must already exist**) |
| `PART` | CPU partition for Simulation submissions — consumed by the `Submit.sh` dispatchers and the generation controllers alike (passed only when set) |
| `ACCT` | Slurm account, if the cluster requires one — likewise consumed by the dispatchers and the controllers (passed only when set) |
| `USER_ME` | queue-owner username for the controller's polling (defaults to `$USER`) |
| `REPO` | auto-derived from the script location; override only if the repo is reached by a different path |

Anything left unset falls back to the script defaults (e.g.
`MON` → `$HOME/process_monitoring`), and an unset `PART`/`ACCT` leaves the
submit line at the script's baked defaults.

### Fleet sync: propagating the repo

`Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_Fleet_Sync.sh` is the **single supported
way to propagate this repository** to the other machines; hand-rolled `rsync`
invocations have failed in the documented ways its header records, so do not
improvise one. It reconciles every remote's repo to the reference machine exactly:
the file list is always relative to the repo root (nothing flattens into the wrong
directory), deletions on the reference propagate (per-directory recursion with
`--delete`), the secrets file never leaves the reference machine, and each
remote's own machine-local files (`machine_profiles.toml`,
`Script_Bank/HPC/hpc_local.env`) are never touched. Dry run is the default
(`DRYRUN=1` prints what would change and transfers nothing; set `DRYRUN=0` only
after reading the printed plan), and an optional machine argument restricts the
sync to one remote.

---

## 6. Special-situation entry points (run ad hoc)

The four stages in §1 form the standard, dispatcher-driven pipeline. A handful of
ad-hoc utilities sit outside it — the post-hoc analyses and calibration builders in
`Script_Bank/Analysis/` (for example the pooled `Nuisance_DLI` construction, the
embedding-space distance, and the flicker-rate derivation). They stay out of the
`Submit.sh` dispatchers by design, so the standard dispatcher surface stays exactly
the four stages — but three of them have dedicated standalone wrappers in this
directory, submitted directly with `sbatch`:

- `SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Posterior_Calibration.sh` — the posterior-calibration
  diagnostic (SBC / coverage / TARP / L-C2ST) for either workflow
  (`WORKFLOW=biology|detector`). It shares the Evaluation stage's shard-then-merge
  execution: one worker per GPU on every allocated node writes its own shard, then
  a single `--merge` pass concatenates them and runs the global statistics.
- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Nuisance_DLI.sh` — the pooled `Nuisance_DLI`
  spec-template build (`--emit-template`): shards the `(kind, cell)` pool build
  across one worker per GPU on every allocated node, then a single-process, no-GPU
  `--merge` step assembles the cached pool and the spec.
- `SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Embedding_Space_Distance.sh` — the
  experimental-versus-synthetic embedding-space distance for either workflow
  (`WORKFLOW=biology|detector`). Its engine is single-GPU by design (no sharding,
  no merge) on a whole-node allocation; do not read the allocated GPUs as data
  parallelism.

- `SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Bulk_Delete.sh` — parallel bulk deletion that empties a
  data tier (its contents, never the directory) on a GPFS/Lustre filesystem; dry-run by
  default, `DRYRUN=0` deletes, `P=16` streams. The recipe below says when and how.

### Freeing a scratch tier under an inode quota

A parallel filesystem refuses writes with "Disk quota exceeded" when the **file-count**
quota is hit, however many bytes are free: on JUWELS `$SCRATCH` (2026-09-16) 4.26 M files
against a 4.0 M soft / 4.4 M hard limit blocked a `mkdir` with 80 TB free. Diagnose with
`jutil project dataquota -p <project>` (both byte and inode columns) and `df -i <tier>`.
What fills a scratch tier is the regenerable TRAIN/TEST data of retired campaigns — video
and theta zarr stores and, above all, the `READY_TRACT/` trajectory folder (one `.h5`
per simulation). EVAL, Posit, Labor and Experiment live on the permanent tier and are
never deletion targets. Confirm what a tier holds before deleting (count `_TRAIN` /
`_TEST` / `_EVAL` entries and list any `Posit`, `Labor`, `Experiment` subdirectory).

Delete with the utility above, detached, never inside a timeout:

```bash
# dry-run (default) prints the top-level entry counts and deletes nothing
bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Bulk_Delete.sh \
    /p/scratch/<project>/.../<retired-repo>/Data_Bank/Video /p/scratch/<project>/.../<retired-repo>/Data_Bank/Theta
DRYRUN=0 nohup bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Bulk_Delete.sh <same dirs> \
    > ~/bulk_delete_$(date +%F).log 2>&1 < /dev/null &
```

Rates measured on a JUWELS login node: a single `find -delete` ~30 K files/min; 16 parallel
`rm -rf` streams over the top-level entries ~400 K files/min (3.75 M files in 11 min, load
~20). The `Data_Bank` directories themselves are kept because the machine profiles require
`data_bank_root` and `scratch_data_bank_root` to exist. Two gotchas: `df -i` reads low for
the first minute while the quota accounting catches up, so measure the rate over a full
minute before judging it; and when stopping a deletion from an ssh command, anchor the
pattern (`pkill -f "^find /p/scratch/..."`) — an unanchored `pkill -f <name>` also matches
the remote shell running that very command and kills the session.

The rest are run by hand — single-process with plain `python`, not `torchrun` —
and each is documented in its own companion `.md`. When one of those needs a GPU,
launch it with a one-off `sbatch --wrap` on the check partition that sources
`hpc_local.env` (§5) and activates `SRM_AND_SBI_ENVY_V0`, following the §3
job-name pattern with the utility's own descriptor. The DETECTOR calibration
workflow has its own submission machinery (§8).

---

## 7. Artifact backups

The trained artifacts — an estimator (`Posit/…_Estimator.npz`) and its checkpoint
(`Labor/…_Optimum_ANN.pth`) — are overwritten in place whenever the stage that
produces them re-runs (a fresh Inference run). The canonical names never change:
they are the live objects every downstream stage (Evaluation, Experiment) loads.
To keep a superseded model
identifiable and recoverable after it is overwritten, a copy is set aside under a
distinct name — automatically by a finished run, or by hand for an ad-hoc keep.

### Automatic provenance backup

A finished Inference run that loaded a TEST set (`--test-tasks > 0`, so a
model-selection loss exists) writes, alongside the canonical checkpoint and
estimator, a provenance-named copy of each. The suffix records the run's training
scale and result in the filename itself — legible at a glance, without opening the
artifact — inserted before the extension:

    <original-stem>_TRAIN+TEST_<train>+<test>_Epoch_<n>_TEST_LOSS_<loss>.<ext>

- **`<train>` / `<test>`** — the number of TRAIN and TEST videos the run used, as
  thousands-tokens (`50000` → `50K`, `47500` → `47.5K`). Sizes are in the name
  because a test loss is only comparable at equal test-set size: a −17.00 measured
  on 50K selection videos is not the same result as −17.00 on 10K.
- **`Epoch_<n>`** — the number of epochs this job ran (`--epochs`), counted for the
  current job (a warm-started `--resurrect` run counts its own epochs, not the
  history of the weights it loaded).
- **`TEST_LOSS_<loss>`** — the checkpoint's best TEST loss, always exactly two
  decimals. A negative loss keeps its native `-`; a positive loss gets an explicit
  `+`; a value that rounds to zero is written `0.00`, with no sign.

For a 200K-train / 50K-test run over 25 epochs that reached a best test loss of
−17.05, the pair is:

    Labor/SRM_AND_SBI_MONOMER_DIMER_ALP_2S_50FPS_Optimum_ANN_TRAIN+TEST_200K+50K_Epoch_25_TEST_LOSS_-17.05.pth
    Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_2S_50FPS_Estimator_TRAIN+TEST_200K+50K_Epoch_25_TEST_LOSS_-17.05.npz

The backup is a copy, so the canonical `…_Optimum_ANN.pth` / `…_Estimator.npz`
stay the active artifacts and a backup is never picked up as the live model. Across
warm-started runs the canonical checkpoint is additionally protected by
save-on-improvement — it is overwritten only when the current run beats the loaded
best on the TEST set — so each accepted optimum is preserved under its own backup
name. To make a backup the active model again, copy it onto the canonical name.

A run with no TEST set (`--test-tasks 0`, which trains on all of TRAIN and keeps
the last-epoch checkpoint) has no selection loss to name a backup by, so it writes
the canonical pair only.

### Manual ad-hoc keep

The automatic backup covers finished training runs. To preserve an artifact the
automatic scheme does not — for instance the current canonical *before* you
deliberately launch a run that will overwrite it, or a milestone worth tagging —
copy it aside by hand under a tag plus a day-month-year date, inserted before the
extension:

    <original-stem>_<TAG>_<DD.MM.YYYY>.<ext>

```bash
# preserve the current best 2S estimator + its checkpoint before an overwriting run
cp Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_2S_50FPS_Estimator.npz \
   Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_2S_50FPS_Estimator_PREPROD_01.07.2026.npz
cp Labor/SRM_AND_SBI_MONOMER_DIMER_ALP_2S_50FPS_Optimum_ANN.pth \
   Labor/SRM_AND_SBI_MONOMER_DIMER_ALP_2S_50FPS_Optimum_ANN_PREPROD_01.07.2026.pth
```

`<TAG>` is a short label for why the copy was kept (e.g. `PREPROD` before a
production run); `<DD.MM.YYYY>` is the date (e.g. `01.07.2026`).

### Both kinds land beside the original

Whether automatic or manual, the suffix sits before the extension, so a backup
never matches the `…_Estimator.npz` / `…_Optimum_ANN.pth` names the pipeline loads —
it is kept, but never picked up as the active artifact. Keep each backup in the
same `Posit/` or `Labor/` directory as its original; if that storage tier is not
itself backed up (for example a scratch `data_bank_root`), also copy the backup to
a backed-up location.

---

## 8. Detector calibration workflow (own submission machinery)

The **Detector calibration workflow** is a complete workflow parallel to the
biology pipeline: it runs the same four-step process (simulate → infer →
evaluate → experiment) but infers the imaging (diffraction-limited-imaging)
parameters with the reaction-diffusion biology marginalized over its prior, so those parameters are
calibrated for production rather than hand-tuned. It has its **own committed
submission machinery**, filename-namespaced (`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_*`)
and coexisting with the biology wrappers in this directory — the same
filename-alias scheme as the `_DETECTOR` data and entry scripts. It is a separate,
parallel workflow: it is **never wired into the biology `Submit.sh` dispatcher
or the four biology stage wrappers**, and they are never wired into it. The one
shared input is the condition's RDS trajectory tier (one per condition, because the
association setting is per condition): generated by the biology dispatcher's RDS-only
simulation with the same `CONDITION` (`simulation SIM_STAGE=rds CONDITION=<FAB|INLB>`)
and re-imaged by the Detector's DLI-only Simulation wrapper, so a Detector campaign
needs no RDS submission of its own.

| Detector script | Role | Compute |
|---|---|---|
| `..._DETECTOR_HPC_Simulation.sh` | generation: a DLI-only pass over the condition's trajectory tier (B2) — imaging drawn from the detector prior, per condition; the tier itself comes from the biology dispatcher's RDS-only simulation with the same `CONDITION` (`SIM_STAGE=rds CONDITION=<FAB|INLB>`), one tier per condition, shared by both workflows; packed per node, `--array` fan-out, seedless | CPU |
| `..._DETECTOR_HPC_Inference.sh` | train the imaging posterior (B3); >1 GPU → DDP via `torchrun`, >1 node → DDP across nodes (`srun`+torchrun, c10d); saves the version-portable A5 estimator | GPU |
| `..._DETECTOR_HPC_Evaluation.sh` | imaging MAP recovery on the held-out EVAL set (B5); shards across all ranks (>1 GPU/node) + a separate `--merge` step | GPU |
| `..._DETECTOR_HPC_Experiment.sh` | imaging MAP estimation on real microscopy videos (B4); shards across all ranks (>1 GPU/node) + a separate `--merge` step | GPU |
| `..._DETECTOR_HPC_Submit.sh` | the Detector dispatcher — dry-run-first single-job submit builder | — |
| `..._DETECTOR_HPC_Nuisance_DLI.sh` | pooled `Nuisance_DLI` spec-template build (`--emit-template`): posterior-sample pool over the real recordings, sharded across all ranks (>1 GPU/node) + a separate no-GPU `--merge` step; submitted directly with `sbatch`, not via `Submit.sh` | GPU |
| `..._DETECTOR_HPC_Generate_Controller.sh` | rolling submit-and-gate controller for a full Detector generation campaign — QOS caps, EVAL hard-gated on TRAIN+TEST completion; dry-run by default | — |

(Posterior calibration and the embedding-space distance are covered for the
Detector by the standalone analysis wrappers in §6 —
`SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Posterior_Calibration.sh` and
`SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Embedding_Space_Distance.sh`, each with
`WORKFLOW=detector`.)

**Dispatcher.** `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Submit.sh` mirrors the
biology `Submit.sh` — dry-run first (`DRYRUN=1` prints the exact `sbatch` line;
`DRYRUN=0` submits) — for the Detector stages, and renders the `_DETECTOR`
job-name `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_<CONDITION>_<timing_label>_<Stage>[_<SPLIT>]`.

**Two GPU modes (Goethe), set explicitly at submit time.** Neither dispatcher pins
a GPU mode: both `Submit.sh` scripts only forward what the submitter sets
(a non-empty `GPU_PART`/`GRES`/`NODES` becomes `--partition`/`--gres`/`--nodes`;
anything unset leaves the stage script's baked `#SBATCH` defaults, `gpu` +
`gpu:8`). The two modes in use are therefore a manual recipe: checks set
`GPU_PART=gpu_test GRES=gpu:4 TIME=08:00:00` explicitly (the `gpu_test` nodes
carry 4 GPUs); production uses the real `gpu` partition — a whole node, `gpu:8` —
which is the baked default, so `GPU_PART=gpu` alone suffices.

**Seedless generation.** Detector generation passes no seed by design: each task
draws fresh entropy, and provenance is carried by the global task index in the
file names. `SEED` is an off-by-default reproducibility-debug knob only, and it
must be left unset for the smoke and for normal campaigns.

**8-bit video.** The DLI step of the generation stage renders 8-bit video via
the `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Simulation.sh` `VIDEO_DTYPE_BITS` knob
(default 8), matching `VALIDATION.md` section 2.5.

**Chaining** (with `DEP=afterok:<jobid>[:...]`) — the check-run sequence
(2 s, 16/4/2 tasks × 10 sims, 5 epochs), run seedless, all DRY-RUN by default;
capture each printed job id and feed it to the next `DEP`. The authoritative
recipe is `VALIDATION.md` section 2.5 ("Detector calibration smoke test"):

    cd /path/to/srm-and-sbi-monomer-dimer-alp
    S=Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Submit.sh
    # generation (CPU test partition), one job per split:
    # (the FAB tier must exist first: the biology dispatcher's `simulation SIM_STAGE=rds CONDITION=FAB`, one job per split)
    PART=test NTPN=16 bash $S simulation CONDITION=FAB SPLIT=train TASK_COUNT=16 TASK_SIMS=10 TOTAL_TIME=2.0
    PART=test NTPN=4  bash $S simulation CONDITION=FAB SPLIT=test  TASK_COUNT=4  TASK_SIMS=10 TOTAL_TIME=2.0
    PART=test NTPN=2  bash $S simulation CONDITION=FAB SPLIT=eval  TASK_COUNT=2  TASK_SIMS=10 TOTAL_TIME=2.0
    # train (afterok gen train+test), then evaluate (afterok train + gen eval):
    GPU_PART=gpu_test DEP=afterok:<gen-train>:<gen-test> bash $S inference  CONDITION=FAB TRAIN_TASKS=16 TEST_TASKS=4 EPOCHS=5 BATCH=8 TOTAL_TIME=2.0
    GPU_PART=gpu_test DEP=afterok:<inference>:<gen-eval> bash $S evaluation CONDITION=FAB EVAL_TASKS=2 POOL_MODE=unrestricted TOTAL_TIME=2.0
    # experiment (afterok inference): apply the undertrained posterior to real videos.
    # KINDS defaults to the run's CONDITION; because Slurm --export splits on commas, leave KINDS
    # at its default or pre-export it. POOL_MODE=unrestricted is required for the
    # undertrained smoke posterior.
    GPU_PART=gpu_test DEP=afterok:<inference> bash $S experiment MAX_CELLS=2 POOL_MODE=unrestricted TOTAL_TIME=2.0

Respect the check-partition QOS (e.g. Goethe `test`: at most 3 submitted / 2
running / 2 nodes per user) — consolidate generation accordingly.

---

## 9. Do not

- **Do not invent job or log names.** Use exactly
  `SRM_AND_SBI_MONOMER_DIMER_ALP_<CONDITION>_<timing_label>_<Stage>[_<SPLIT>]` (§3). No invented
  tokens such as `5S_PROD`, `smoke`, `mgpuval`, or an `HPC` segment.
- **Do not recompute node geometry.** Replicate the `#SBATCH` layouts and header
  examples in the scripts (§4) — the core counts, `--extra-node-info`, GPU
  count, and memory are deliberate.
- **Do not cancel and resubmit a correctly-running job for cosmetics.** A job
  whose name or log path is merely not your preference is still producing valid
  output; leave it running.
- **Do not edit the committed scripts for per-machine values.** Put machine
  differences in `hpc_local.env` (§5), not in the scripts.
- **Do not submit Simulation without `--array`.** Always pass `--array`
  (`--array=0-0` for a single node) so the log's `%a` is a clean node number.
- **Do not submit without a dry-run.** Preview every submission first — the
  `Submit.sh` helper or the generation controller with their default `DRYRUN=1`,
  or the entry point's own `--dry-run` — and submit only once the printed command
  and the resolved inputs check out (§2).
