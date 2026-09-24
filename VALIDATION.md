# Validation and Reproducibility — srm-and-sbi-monomer-dimer-alp

This guide describes how to set up the environment, confirm that each pipeline
stage runs, and validate a trained posterior. It covers three things:

1. **Environment setup** — one-time per machine.
2. **Smoke tests** — quick "does it run at all" checks for each entry point.
3. **Validation methodology** — how the pipeline establishes that the
   simulation and inference code is behaving correctly, including the
   reproducibility guarantees and the dual-duration checks.

Read through the whole document first so the dependencies between steps are
clear: the inference smoke test needs simulated data, and the validation checks
build on the smoke tests.

---

## 1. Environment setup (one-time per machine)

### 1.1 Activate or build the Python environment

The project environment is **`SRM_AND_SBI_ENVY_V0`** — a Python 3.13 scientific
stack (ReaDDy, NumPy 2, zarr, sbi) with a hardware-specific PyTorch build
(ROCm, CUDA, or CPU).

The complete specification and step-by-step install — from scratch or from a
per-machine snapshot — live in **[`env_snapshots/README.md`](env_snapshots/README.md)**,
which is the canonical install guide. Rather than repeating the version pins
here (they would drift out of sync), follow that guide.

There are two first-class ways to get a working environment:

- **Reuse an existing compatible environment.** If you already have an
  environment with the required stack (a shared lab environment, another
  machine's `SRM_AND_SBI_ENVY_V0`, or any environment that matches the
  specification), simply activate it:

  ```bash
  conda activate SRM_AND_SBI_ENVY_V0
  ```

- **Build a fresh environment.** Follow the from-scratch recipe in
  [`env_snapshots/README.md`](env_snapshots/README.md). It installs the
  conda-forge scientific layer, then the hardware-matched PyTorch wheel plus
  the sbi ecosystem, and finally the project package. The guide also documents
  every install gotcha (channel priority, the `psutil`/`ipython` requirement,
  the PyTorch backend table, and the `--no-deps` rule for the editable install).

**Verify** the core stack imports and reports the expected versions:

```bash
python -c "import readdy, torch, sbi, zarr, numpy, psutil; \
print('readdy', readdy.__version__, '| torch', torch.__version__, \
'| sbi', sbi.__version__, '| cuda/hip avail:', torch.cuda.is_available())"
```

(`cuda/hip avail` is `True` only when a GPU is actually visible — on an HPC node
that means inside a GPU allocation, not on the login node; on a CPU-only machine
`False` is expected.)

### 1.2 Install the package in editable mode

From inside the repository, with the environment active:

```bash
pip install -e . --no-deps
```

The `-e` (editable) flag means subsequent edits to package modules take effect
immediately without re-installation. Use `--no-deps`: the runtime dependencies
are already provided by the environment, and a plain `pip install -e .` would
re-resolve them — downgrading sbi and overwriting the hardware-specific PyTorch
build (see the install guide's gotchas).

**Verify**:

```bash
python -c "import srm_and_sbi_monomer_dimer_alp; print(srm_and_sbi_monomer_dimer_alp.__version__)"
```

### 1.3 Configure `machine_profiles.toml`

Copy the template and edit it for the current machine:

```bash
cp machine_profiles.example.toml machine_profiles.toml
$EDITOR machine_profiles.toml
```

Define at least one profile. Required keys per profile:

```toml
[my_local_profile]
running_mode      = "LOCAL"
script_bank_root  = "/full/path/to/srm-and-sbi-monomer-dimer-alp/Script_Bank"
data_bank_root    = "/full/path/to/Data_Bank"
compute_backend   = "GPU"
gpu_device_index  = 0
num_workers       = 0          # 0 = derive from available cores; positive = pin
```

`data_bank_root` must exist as a directory; create it now if it does not. The
output subdirectories (`Theta/`, `Video/`, `Posit/`, `Labor/`) are created
automatically by the entry-point scripts as needed. A machine with a large but
impermanent scratch filesystem may also set `scratch_data_bank_root` to route
the regenerable TRAIN and TEST bulk onto scratch while keeping everything that
must persist on backed-up storage; omit it for a single-tier machine.

### 1.4 Set the `MACHINE_PROFILE` environment variable

```bash
export MACHINE_PROFILE=my_local_profile
```

For persistence, add this to `~/.bashrc` on a local machine or to the HPC job
submission script.

**Verify** the configuration loads cleanly:

```bash
python -c "from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS; print(PARAMETERS.machine.name)"
```

This should print your profile name. The configuration validates at import time:
if the environment variable is unset, the profile is missing, a required key is
absent, or a root directory does not exist, it raises a clear `ValueError`
pointing to the misconfiguration. There is no silent fallback.

---

## 2. Smoke tests

These are quick checks that each entry point runs end-to-end with minimal inputs.
The point is to surface obvious problems (missing imports, wrong shapes, path
misconfiguration) before any longer run. The same structure applies to **both**
workflows in this repository — the biology reaction-diffusion pipeline
(sections 2.1–2.4b) and the Detector imaging-calibration pipeline (section 2.5,
`DETECTOR_WORKFLOW.md`) — which share the stage sequence generate → infer →
evaluate → (experiment). Every smoke passes `--total-time-seconds` (always
required), runs seedless (`--seed None`), keeps the task and simulation counts
small, and passes `--condition FAB` on every stage, the RDS stage included (the
condition selects the trajectory tier — the association setting is per condition —
and the labeling law, and is the condition of the recordings the estimator is for;
the INLB configuration is the same sequence with `--condition INLB`, generating its
own tier). The **Detector calibration smoke (section 2.5) is the reference
configuration** the whole repository follows: one shared duration and the
three-split sizing `--tasks 25 / 5 / 2 --task-simulations 10` (TRAIN / TEST / EVAL
→ 250 / 50 / 20 videos). The biology sections below replicate that same sizing with
the biology entry points.

**Run order (both workflows).** The smoke is a small, fast replica of the full
production run, so it reproduces the production dependency chain rather than reordering
it for convenience. Run it in this order: first the **detector smoke (§2.5)** — the
detector workflow calibrates the imaging model; then **build the `Nuisance_DLI` (§2.5b)**
— the calibrated imaging turned into the samplable artifact the biology renderer draws
from; then the **biology smoke (§2.1–2.4b)** — it consumes that artifact at its DLI
stage. The section numbering is a reading convenience; this is the required execution
order. (The detector DLI itself needs no artifact: it draws imaging from the prior box,
because imaging is the detector's inference target.) Both workflows re-image the same per-condition RDS tier;
what is drawn once and what each workflow adds is summarized in `PROJECT_CONTEXT.md` §4,
*One draw of the biology per condition, two imagings per draw*.

Two rules apply to every smoke and to every production run:

- **Seedless.** Pass `--seed None` explicitly on each stage. Every stage defaults
  to `None` (non-deterministic); a fixed seed freezes per-video particle
  placement, PSF, brightness, and detector noise across a split, collapsing the
  per-video variability the estimator must learn. The one deliberate exception is
  the theta-only regression test (section 3.2), which fixes `--seed` on purpose
  to check theta reproducibility.
- **Approval required.** No smoke or production run is launched without the
  project owner's explicit approval. Local checks (imports, stochastic-matrix
  diagnostics, dry-run prints) and code synchronization are fine; submitting any
  compute job — single-GPU or HPC, smoke or production — requires sign-off first.

On HPC, run smoke and check submissions on the short-lived `test` (CPU) and
`gpu_test` (GPU) partitions, leaving the production `general1` (CPU generation)
and `gpu` (train/eval) partitions for full runs. Take the partition,
node-geometry, and resource layout from each stage script's `#SBATCH` block and
header examples and replicate them verbatim — change only what the check
requires (typically the duration and the task counts). Do not recompute node
counts, core-per-node geometry, or GPU counts; the scripts already pin them.

### 2.1 RDS (reaction-diffusion simulation — the condition's trajectory tier)

```bash
# the MET-FAB tier (no association channel); the INLB arm generates its own tier with --condition INLB
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py --condition FAB --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 --seed None
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py --condition FAB --total-time-seconds 2.0 --split test  --tasks 5  --task-simulations 10 --seed None
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py --condition FAB --total-time-seconds 2.0 --split eval  --tasks 2  --task-simulations 10 --seed None
```

**Expected**: per split, one `.h5` trajectory per simulation under the RDS
trajectory directory `READY_TRACT/`, namespaced by split — for TRAIN,
`<data_bank>/Video/READY_TRACT/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_TASK_0_TRAIN/`
(`..._TASK_0_SIM_0_TRAIN.h5`, …), and one `.zarr` theta set per task at
`<data_bank>/Theta/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr`
(the condition token right after the sibling alias, no workflow qualifier; with the
`_TEST` / `_EVAL` namespaces for the other two splits). Each theta set carries its schema in
the store's attributes — the eleven keys in order, their prior bounds and scales, the condition,
the timing label, and the package version — and the stage prints a `Theta_Set schema:` line per
task; every later stage refuses a theta set whose schema differs from its table. This generates
250 / 50 / 20 (train / test / eval) trajectories. Add `--verbose` to print the
sampled diffusion and reaction rates per simulation. This is the MET-FAB tier, shared
by both workflows: the detector smoke (§2.5) re-images these same trajectories under
the FAB labeling law. The second condition has its own tier, because the association
setting differs (MET-INLB associates at the reference intensity, MET-FAB has no
association channel): generate it with the same three commands and `--condition INLB`.
Never re-run the RDS stage over a condition's existing tier — it would replace the tier
and mislabel the videos already rendered from it.

### 2.2 DLI (diffraction-limited imaging)

```bash
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py --condition FAB --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 --video-dtype-bits 8 --seed None
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py --condition FAB --total-time-seconds 2.0 --split test  --tasks 5  --task-simulations 10 --video-dtype-bits 8 --seed None
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py --condition FAB --total-time-seconds 2.0 --split eval  --tasks 2  --task-simulations 10 --video-dtype-bits 8 --seed None
```

**Expected**: one `.zarr` video set per task at
`<data_bank>/Video/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Video_Set_TASK_0_TRAIN.zarr`
(the condition slot follows the alias on every DLI product; and the `_TEST` / `_EVAL`
namespaces), 250 / 50 / 20 videos in all. At 2 s each
video is `(100, 256, 256)` — 100 frames at 50 frames per second over a 256×256
detector grid — and every value is a non-negative integer pixel count. Because the
biology renderer marginalizes imaging, each task additionally writes the per-task draws
it used, all under `<data_bank>/Theta/` with the condition slot: a `Nuisance_DLI_Theta_Set`
(the imaging vectors drawn from the artifact), a `Nuisance_SCOPE_Theta_Set` (the EMCCD
camera vectors drawn from the SCOPE box), and a `Labeling_Set` (the per-simulation labeling
record: true and visible initial composition under the MET-FAB labeling law, and the probe
occupancy applied, in `occupancy_monomer` / `occupancy_dimer`). `--video-dtype-bits 8` matches the bit depth the
estimator trains on and is also the DLI default.

**Occupancy.** The DLI stage applies the condition's declared probe occupancy by default —
MET-FAB 0.155 (derived from the declared Fab/InlB visibility ratio 0.5 and the INLB anchor),
MET-INLB 0.5 (declared) — and prints it with its source (`derived` / `declared`); no flag is
passed, and there is no full-occupancy default (`PROJECT_CONTEXT.md` §2, *How the prior ranges
and the declared inputs are set*). `--occupancy` is an explicit OVERRIDE for a sensitivity run
only, recorded as `override` in the `Labeling_Set`; an override writes into the same product
names as the default pass, so run it on a throwaway smoke tier and never over a tier that feeds
training or calibration:

```bash
# SENSITIVITY RUN, not a training tier: re-image a throwaway INLB smoke tier under the within-dimer alternative (every InlB dimer carries two ligands: monomer subunits 0.5, dimer subunits 1.0; both-subunits-labeled share among visible dimers 1/3 instead of 1/7 (equal to the two-dye share under the one-dye-per-ligand InlB law))
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py --condition INLB --total-time-seconds 2.0 --split eval --tasks 2 --task-simulations 10 --video-dtype-bits 8 --seed None --occupancy A=0.5,B=1.0
```

**Requires**: two prerequisites. First, the RDS smoke (§2.1) must have run with the
same duration, splits, and `--task-simulations`; DLI reads the `.h5` trajectories and
theta set RDS wrote, so the two stages share `--tasks`, `--split`, and
`--task-simulations`. Second — the cross-workflow dependency — the condition's `Nuisance_DLI`
artifact (`..._DETECTOR_FAB_2S_50FPS_Nuisance_DLI.npz`) must already exist: the biology
renderer **marginalizes imaging** by drawing the photophysics from that artifact (and the
camera from the SCOPE box), and fails loud if it is absent. The artifact is a detector-side product, so the detector smoke (§2.5)
and the `Nuisance_DLI` build (§2.5b) must run first — see **Run order** in the section
intro.

### 2.2b Prior-realization audit (after any tier or DLI pass)

Run the prior-realization audit (it reads the products and writes only its report) over every
generated tier and every DLI pass before
the products are used, and again whenever a range or a declared input changes. It reads the
condition's `Theta_Set` (P1: every draw inside the prior box, per-row uniformity; P2: the
composition rule from `(N_R, r)`), optionally a few trajectories (P3: frame-0 counts against
the realized composition, subunit conservation, the stationary mode law), and the workflow's
`Labeling_Set` (P4: the recorded occupancy equals the condition's declared or derived value,
the per-subunit visibility, the visible fractions, the both-labeled share, and the emitters per
subunit against the declared visibility), compares the simulated visible counts with the
deposited recordings' spot counts descriptively (P5, Special_Analyses A9), and always
corroborates the theoretical visibility chain through the DLI stage's own labeling functions
on a synthetic lineage for both conditions, including the FAB/INLB visibility ratio against
the declared one (P6). The report ends with theory, code path, and products side by side.
Nothing is simulated or rendered; it takes seconds. `--workflow detector` reads the
detector's `Labeling_Set` instead of the biology's.

```bash
# the FAB TRAIN tier after the RDS + DLI smokes (biology Labeling_Set); <profile> is the machine profile, as everywhere in this file
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py --condition FAB --split train --total-time-seconds 2
# the same with P3 over eight evenly spaced trajectory files
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py --condition FAB --split train --total-time-seconds 2 --trajectories 8
# the checks themselves, on in-memory draws from the prior and the declared occupancies (no tier needed)
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py --selftest
# the visibility chain alone (P6, both conditions): theory vs the DLI code path, no tier needed
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py --visibility
```

**Expected**: a report `.md` and `prior_realization_summary.json` under
`<data_bank_root>/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Prior_Realization_Audit_TRAIN/`,
every check PASS; the companion
`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.md` explains each
verdict. Products generated before the decided ranges cannot be reused: a `Theta_Set` written
before the schema existed carries none, and one drawn under another table carries different keys
or bounds, so the audit (P0) and every stage refuse it by name, with the reason. Generate a fresh
per-condition tier. The audit reads TRAIN/TEST products from the split-aware root (the scratch
tier on two-tier machines) and writes its report on the permanent tier.

### 2.3 Inference (posterior training)

```bash
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Inference.py --condition FAB --total-time-seconds 2.0 --tasks 25 --test-tasks 5 --epochs 5 --batch-size 8 --seed None
```

**Expected**: a network checkpoint at
`<data_bank>/Labor/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Optimum_ANN.pth` and a
version-portable estimator artifact at
`<data_bank>/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Estimator.npz`. The
training loop also writes a full-state resume file beside the checkpoint,
`<data_bank>/Labor/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Resurrect_State_ANN.pth`, updated
every epoch — its presence is what lets a later `--resurrect` hot-restart (see §2.4).
Five epochs on the small smoke dataset (250 train / 50 test videos) will not
produce a useful posterior, but they exercise the full training and save path,
including the network construction and the data loader. Because it passes
`--test-tasks 5`, the run loads the held-out TEST set, selects the checkpoint on
best test loss, and additionally writes a provenance-named backup of the
checkpoint and estimator (see PROJECT_CONTEXT §3 and the HPC runbook §7).

**Requires**: the RDS and DLI smoke tests (sections 2.1 and 2.2) must have run
first with the same `--task-simulations 10` and matching task counts. With too few
simulations the dataset is too small to form even a single training batch and the
run fails early.

**Small-GPU batch-size workaround**: the `Complex3DCNN` forward pass over 2 s videos
is memory-heavy (one 3D convolution alone needs roughly 6 GB), so even the smoke's
`--batch-size 8` can exceed a small GPU (for example a 4 GB card) and raise
`CUDA out of memory` partway into the first epoch. Two options:

- **Reduce the batch size** to `1` or `2` for a memory-light smoke check on a
  small GPU. This verifies the pipeline without producing a useful posterior:

  ```bash
  python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Inference.py --condition FAB --total-time-seconds 2.0 --tasks 25 --test-tasks 5 --epochs 5 --batch-size 1 --seed None
  ```

- **Train on a larger GPU**, or on CPU when no adequate GPU is available
  (select the CPU backend in `env_snapshots/README.md` and the CPU compute
  backend in the machine profile).

### 2.4 Resurrect (continue from an existing run)

```bash
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Inference.py --condition FAB --total-time-seconds 2.0 --tasks 25 --test-tasks 5 --epochs 5 --seed None --resurrect
```

**Expected**: because the inference smoke test (§2.3) left a full-state resume file
beside the checkpoint, this run **hot-restarts** — it prints a
`HOT RESTART: resumed full state ...` line reporting the resumed global epoch and
learning rate, then trains the requested `--epochs` more (five here, numbered globally
6–10 in the epoch lines) continuing the exact optimizer + learning-rate schedule (no
re-converging, no LR reset). Because it passes `--test-tasks 5`, it loads
the held-out TEST set and selects the checkpoint on best test loss (overwritten only on
a new best), continuing the best-on-test bookkeeping §2.3 began. If the resume file is
absent (for example deleted, or a run predating this feature), the same command falls
back to a **cold** restart — loading the best checkpoint weights into a fresh optimizer
at the peak LR (a `RESURRECT (cold: ...)` line) — and then writes a resume file so the
next `--resurrect` hot-restarts. This makes incremental training across separate,
wall-time-limited invocations behave like one continuous run.

**Continuity check** (optional): run §2.3 with `--epochs 4` and note the epoch-4
learning rate; then run this `--resurrect` command — the printed global epoch resumes
where §2.3 left off and the learning rate continues from the saved schedule rather than
jumping back to the peak.

### 2.4b Evaluate and Experiment (biology)

The two remaining stages complete the biology smoke's `generate → infer → evaluate →
experiment` sequence, mirroring the Detector reference (§2.5, steps 4–5) with the
biology entry points and the same flags:

```bash
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Evaluation.py --condition FAB --total-time-seconds 2.0 --eval-tasks 2 --pool-mode unrestricted --seed None
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment.py --condition FAB --total-time-seconds 2.0 --max-cells 2 --pool-mode unrestricted --seed None
```

`--pool-mode unrestricted` is mandatory here for the same reason as §2.5: the smoke
posterior is undertrained, so its mass falls outside the prior box and the default
`bounded` rejection pool stalls. Evaluation reports MAP recovery on the 20 EVAL videos
(requires the Inference smoke's estimator); Experiment applies the posterior to the real
MET recordings of the run's condition and writes a report for the FAB cells (requires
those recordings present under the experiment data directory).

**Acceptance** (biology): all five stages exit zero; the Inference test loss descends
across the epochs on fresh per-video data (a flat curve signals a frozen-seed
regression); Evaluation reports MAP recovery on the 20 EVAL videos; Experiment writes a
report for the FAB cells.

### 2.5 Detector calibration smoke test

The Detector calibration workflow (imaging-parameter inference with the
reaction-diffusion biology marginalized by re-imaging the condition's trajectory tier; see
`DETECTOR_WORKFLOW.md`) has its own smoke — the condition's tier, then its four stages — run in
order on a single GPU with plain `python`.
It is seedless and requires approval (both rules above). Use one duration for all
five stages — 2.0 s here; the pipeline is duration-general, but the DLI stage
checks its frame count against the RDS trajectories, so a single run must share
one duration. The inferred imaging vector is 6-dimensional — the five EMCCD camera parameters are marginalized as the SCOPE nuisance (drawn at the DLI stage, recorded separately as `Nuisance_SCOPE`), so the DLI stage writes a `Theta_Set` (6 learnable), a `Nuisance_SCOPE_Theta_Set` (5 camera), and a `Labeling_Set` (the per-simulation labeling record, the applied probe occupancy included — the detector DLI applies the condition's declared occupancy by default, exactly as the biology DLI does, §2.2) per task, all under the condition-qualified alias (`..._DETECTOR_FAB_2S_50FPS_...`).

```bash
# 1. The MET-FAB trajectory tier, per split (seedless; the sibling alias plus the condition token). The
#    detector has no RDS stage of its own; if the biology smoke (section 2.1) already generated this tier
#    at these sizes, skip this step. The INLB arm generates its own tier with --condition INLB.
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py --condition FAB --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 --seed None
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py --condition FAB --total-time-seconds 2.0 --split test  --tasks 5  --task-simulations 10 --seed None
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py --condition FAB --total-time-seconds 2.0 --split eval  --tasks 2  --task-simulations 10 --seed None
# 2. Render the detector videos over that tier, per split (seedless; 8-bit, matching what the estimator trains on)
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Simulation_DLI.py --condition FAB --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 --video-dtype-bits 8 --seed None
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Simulation_DLI.py --condition FAB --total-time-seconds 2.0 --split test  --tasks 5  --task-simulations 10 --video-dtype-bits 8 --seed None
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Simulation_DLI.py --condition FAB --total-time-seconds 2.0 --split eval  --tasks 2  --task-simulations 10 --video-dtype-bits 8 --seed None
# 3. Train the imaging posterior (single GPU)
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Inference.py --condition FAB --total-time-seconds 2.0 --epochs 5 --tasks 25 --test-tasks 5 --batch-size 8 --seed None
# 4. MAP recovery on the held-out EVAL set
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Evaluation.py --condition FAB --total-time-seconds 2.0 --eval-tasks 2 --pool-mode unrestricted --seed None
# 5. Real-data application (the run's condition; --kinds names another condition only for a deliberate cross-condition application)
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment.py --condition FAB --total-time-seconds 2.0 --max-cells 2 --pool-mode unrestricted --seed None
```

This generates 250 / 50 / 20 (train / test / eval) videos. `--pool-mode
unrestricted` is mandatory on Evaluation and Experiment: the smoke posterior is
undertrained, so its mass falls outside the prior box and the default `bounded`
rejection pool stalls; `bounded` is for a well-trained (production) posterior.
`--video-dtype-bits 8` matches the synthetic-video bit depth the estimator trains
on (raw experimental frames are 16-bit, converted to 8-bit for inference) and is
also the DLI default. Unlike biology, the detector DLI needs no `Nuisance_DLI` artifact
— it draws imaging from the prior box, because imaging is what the detector infers. As
with biology, Evaluation and Experiment require the estimator trained in step 3, and
Experiment additionally requires the real FAB/INLB recordings staged under the experiment
data directory.

**Acceptance**: all five stages exit zero; the Inference test loss descends across
the five epochs on fresh per-video data (a flat curve signals a frozen-seed
regression); Evaluation reports MAP recovery on the 20 EVAL videos; Experiment
writes a per-condition report for the FAB and INLB cells. After step 2, run the
prior-realization audit over the detector pass (§2.2b, `--workflow detector`).

### 2.5b Build the Nuisance_DLI (detector → biology bridge)

The biology DLI stage draws its imaging from the `Nuisance_DLI` artifact — the detector
calibration turned into a samplable imaging distribution. It is therefore built **after**
the detector workflow (§2.5) and **before** the biology smoke (see **Run order** in the
section intro). The artifact is Detector-namespaced and per condition
(`<data_bank>/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Nuisance_DLI.npz`): the
FAB detector calibration produces the FAB artifact, which the FAB biology DLI consumes.

The construction (`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py`)
reads a value-based spec whose `posterior_sample_pool_choice` sets how the calibration
becomes the samplable artifact. **Production** uses `raw` — the faithful calibration,
resampled from the detector estimator's posterior over the real recordings; it is a GPU
step and needs the estimator trained in the detector Inference stage. For a **smoke** the
imaging values need only be valid, so use `box_user` with each parameter's range set to its
imaging prior box — the six learnable imaging parameters' `PRIOR_RANGE` values defined in
`srm_and_sbi_monomer_dimer_alp/detector_parameterization.py` (the `DETECTOR_PARAMETERIZATION` table).
This is a per-parameter uniform over the prior that needs no estimator, no GPU, and no
recordings — the smoke's fast stand-in for the calibration, occupying the same recipe slot. Write the spec to
`<data_bank>/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Nuisance_DLI_Spec.toml`:

```toml
[block]
posterior_sample_pool_choice = "box_user"   # per-parameter uniform over the ranges below; no pool / estimator / GPU
pool_mode = "bounded"
# Each [low, high] is that parameter's imaging prior PRIOR_RANGE (log10), defined in
# srm_and_sbi_monomer_dimer_alp/detector_parameterization.py (the DETECTOR_PARAMETERIZATION table;
# also det.theta_lower_bound() / det.theta_upper_bound()). The numbers below are that prior
# at the current version — if the imaging prior changes there, re-derive these to match.
# box_user validates every range against the prior box and fails loud otherwise, so a stale
# copy cannot silently pass; a uniform over the full prior is trivially valid smoke imaging.
[imaging.mu_r]
low = 0.0
high = 0.3
[imaging.sigma_r]
low = -1.0
high = -0.25
[imaging.mu_pc]
low = 2.0
high = 2.75
[imaging.sigma_pc]
low = -0.75
high = 0.0
[imaging.prob_photo_bleach]
low = -2.0
high = -0.5
[imaging.lambda_rate]
low = 0.0
high = 1.0
```

Then build it (CPU, seconds):

```bash
python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py --condition FAB --total-time-seconds 2.0 --build
```

**Expected**: `<data_bank>/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Nuisance_DLI.npz`
plus a `..._Nuisance_DLI_Analysis/` directory (a `report.md` and a 1-D marginals figure).
The biology DLI (§2.2) then loads it; with `box_user` over the prior box every drawn imaging
vector lies inside the prior (the report's "frac outside prior" is 0). The full set of
representations and the calibration rationale are in the companion note
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.md`.

### 2.6 Running smokes on HPC

The same smoke runs on a cluster through the committed wrappers — the biology
`SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_*` set and the Detector `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_*`
set — which exercise the paths a single workstation cannot: **multi-CPU
generation** (many tasks packed per node, fanned out with a Slurm `--array`) and
**multi-GPU training, evaluation, and experiment** (data-parallel training;
sharded evaluation/experiment with a separate `--merge` step). Each GPU stage also
scales **across nodes** with `NODES=N` — `--gres` is per node, so
`world_size = NODES * GPUs-per-node`; add `NODES=2` to a GPU smoke to exercise the
multi-node path (inference binds all ranks in a c10d rendezvous; the sharded stages
just need the shared output dir for the merge). Running the smoke on HPC therefore
validates the multi-node and multi-GPU machinery, not only the stage logic.

Submit on the short-lived check partitions (`test` for CPU generation, `gpu_test`
for GPU stages), leaving the production partitions for full runs. Take the
partition, node geometry, and GPU counts from each wrapper's `#SBATCH` block and
header, and replicate them — changing only the duration and the small smoke
counts (section 2.5). Submit through the dispatchers — `..._HPC_Submit.sh` and
`..._HPC_Generate_Controller.sh` (biology and Detector alike) — which default to
dry-run (`DRYRUN=1` prints the resolved `sbatch` line and submits nothing; set
`DRYRUN=0` only after the printed command is verified); the fleet-sync utility
follows the same convention. Every submission, an RDS-only simulation included, also
takes `CONDITION=FAB|INLB`, which the dispatchers validate, forward, and place in the
job name (`SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Inference`;
`SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Simulation_TRAIN` for the FAB tier). The per-stage wrappers themselves are plain
`sbatch` scripts with no dry-run mode, which is why submission goes through the
dispatchers. The end-to-end dependency-chained sequence is in
the HPC runbook (`Script_Bank/HPC/README.md`).

### 2.7 Production runs

Production runs share the same stages, the seedless rule, and the approval
requirement, but are sized to the scientific goal and the machine; many
configurations are valid, and every value below is a recommended reference point,
not a mandate. As a reproducible reference, a production campaign holds the dataset
size fixed — 200K TRAIN / 50K TEST / 25K EVAL videos — for every recording
duration. What changes with duration is how those videos are packed into
generation tasks, and how large the training batch can be: longer videos are
larger, so the simulations per task fall (and the task count rises to hold the
video totals constant), and the recommended training batch size falls to fit GPU
memory.

The RDS trajectory tier is generated once per condition (`SIM_STAGE=rds`,
`CONDITION=FAB|INLB`, no qualifier; the association setting is per condition) and
shared by both workflows: the detector re-images the condition's tier through its
DLI-only Simulation wrapper, the biology re-images it with a DLI-only submission
(`SIM_STAGE=dli`, `CONDITION=...`) once that condition's `Nuisance_DLI` exists, and
inference, evaluation, and experiment run per workflow and condition.

| duration | sims/task | CORE (TRAIN+TEST) | TRAIN | TEST | EVAL | batch | videos (TRAIN/TEST/EVAL) |
|---|---|---|---|---|---|---|---|
| 1 s  | 1000 |  250 |  200 |  50 |  25 | 64 | 200K / 50K / 25K |
| 2 s  | 1000 |  250 |  200 |  50 |  25 | 32 | 200K / 50K / 25K |
| 5 s  |  500 |  500 |  400 | 100 |  50 | 16 | 200K / 50K / 25K |
| 10 s |  250 | 1000 |  800 | 200 | 100 |  8 | 200K / 50K / 25K |
| 20 s |  125 | 2000 | 1600 | 400 | 200 |  4 | 200K / 50K / 25K |

The split follows the dataset-sizing rule in `Generate_Datasets.py`: TRAIN, TEST,
and EVAL are 0.8, 0.2, and 0.1 of CORE (CORE = TRAIN + TEST), with EVAL floored at
a minimum recovery count.

The epoch budget and the batch size are both flexible; the batch column is a
recommendation that scales down with duration to fit GPU memory — adjust it to the
available hardware. The epoch budget is per invocation. Splitting it across
`--resurrect` rounds (for example, **25 epochs run twice**) is the crash-safe path for
wall-time-limited queues and costs nothing extra: each round hot-restarts from the full
resume file, so the two rounds continue one uninterrupted optimizer + learning-rate
schedule — equivalent to a single **50-epoch** run, without the re-convergence a cold
restart spends at each requeue. The in-run LR warm restart (see `train_loop`) is
a within-run plateau-escape at the LR floor, and its state persists across requeues too,
so the sawtooth is continuous across rounds.

**The sharded stages balance at video granularity, so any task count divides evenly over
any worker count.** Evaluation and the calibration diagnostic enumerate the individual
`(task, sim)` videos and deal *those* round-robin, rather than splitting whole tasks. This
matters because every video costs about the same, so a whole-task split would set the wall
clock by the heaviest rank: ten tasks over eight workers would give two of them twice the
work of the other six, and the run would cost what sixteen tasks should. Video-level
sharding balances any combination to within a single video, so the task count needs no
relation to `world_size` (`world_size = NODES × GPUs-per-node`) and no configuration is
penalized.

Skew is also why the sharded stages are not launched through `torchrun`. Its elastic
agent enforces a 300 s exit barrier that no launcher setting changes (torch 2.9 never reads
`TORCHELASTIC_EXIT_BARRIER_TIMEOUT`): ranks that finish early wait five minutes for the rest
and then tear down the rendezvous, killing any rank still working and discarding its shard,
after which the wrapper's `set -e` aborts before the merge and no report is written. Even
with video-level sharding, sixteen ranks on four nodes finished eighteen minutes apart on
JUPITER and one shard was lost that way. The wrappers therefore launch one plain Slurm task
per GPU (`srun --ntasks-per-node=$GPUS`); `resolve_topology()` reads the rank from Slurm and
nothing waits on anything. The `--merge` step additionally refuses an incomplete shard set
unless `--allow-partial` is passed.

Evaluation uses `--pool-mode bounded` — the well-trained-posterior default, in
contrast to the smoke's `unrestricted` — because the EVAL parameters *are* prior
draws, so rejection sampling inside the prior box is both correct and efficient.
**Experiment uses `--pool-mode unrestricted`**, even with a well-trained posterior:
the real recordings are not prior draws, so the posterior conditioned on one can sit
largely or wholly outside the prior box, and bounded rejection sampling then accepts
essentially no proposals and the stage hangs (sbi reports `Only 0.000% proposal
samples are accepted`). The pool mode therefore follows the *data*, not the training
quality: bounded for synthetic parameters drawn from the prior, unrestricted for real
recordings. Generation is seedless.

Submit on HPC: generation on the CPU partition (tasks packed per node, fanned out
with `--array` per split), training and evaluation on the GPU partition
(multi-GPU, optionally multi-node via `NODES=N` — `world_size = NODES *
GPUs-per-node`). Partitions and node geometry are per-machine, supplied by each
cluster's `hpc_local.env`; the submission pattern and commands are in the HPC
runbook (`Script_Bank/HPC/README.md`). The biology DLI stage runs with the
imaging block marginalized (§3.1); a full biology production run, like the detector
production re-run, is a separate, approval-gated step. As
with smokes, no production run is submitted without the project owner's explicit
approval.

---

## 3. Validation methodology

Validation rests on three ideas: the simulation and imaging code is checked for
**semantic equivalence** against the reference scientific behavior; the
pipeline's reproducibility is pinned down by a **theta-only regression test**;
and the duration-parameterized code is exercised at **two durations** to confirm
the frame-count arithmetic is correct throughout.

### 3.1 Semantic equivalence (three pillars)

The simulation and imaging stages thread explicit, per-function random-number
generators rather than relying on a single shared process-wide stream. As a
result the code is validated **semantically** — confirming it produces the
correct scientific behavior — rather than by per-element numerical matching
against any particular reference run. Equivalence rests on three pillars:

1. **Theta-sampling determinism (mathematically exact).** The theta sampler
   draws from the prior with `np.random.default_rng(seed).uniform(low, high, size)`
   over fixed bounds. The same seed produces bit-identical theta vectors. This
   is directly verifiable — extract the theta set written by a seeded RDS run
   and compare it across runs:

   ```bash
   python -c "
   import zarr
   z = zarr.open('<data_bank>/Theta/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr', mode='r')
   print(dict(z.attrs))      # the schema: parameter_keys, prior bounds, condition, timing, package version
   print(z[:].tolist())
   "
   ```

2. **Reaction-diffusion primitive equivalence.** ReaDDy is a stochastic
   simulator — its stepper draws its own random numbers for diffusive motion
   and reaction-event timing, so trajectories differ run-to-run even at a fixed
   system specification (species, reactions, rates, simulation box, particle
   complement). What is deterministic, and what this pillar verifies, is the
   construction of that specification: the system builder and the simulation
   builder produce a ReaDDy system with the declared rates, geometry, and
   observables expected for a given theta. The verbose RDS banner (`--verbose`)
   prints the diffusion and reaction rates, so the constructed system can be
   inspected directly. The stepper's own internal randomness is the intended
   source of run-to-run variability (see the theta-only regression test,
   section 3.2).

3. **Imaging-pipeline functional equivalence.** The DLI stage applies a Gaussian
   point-spread function (erf-based pixel integration), an EMCCD detector model
   (Poisson photoelectrons, stochastic Gamma electron multiplication, and gain-independent Gaussian read noise), the stationary
   OU brightness flicker, and the duration-independent photobleaching model. The verbose DLI
   banner (`--verbose`) prints the detector parameters, and a rendered video
   (via `--show`) shows sparse fluorescent spots on a near-zero background with
   plausible pixel-value ranges. The pipeline produces videos of the correct
   shape, dtype, and value distribution. The **biology** DLI stage renders through
   the shared, source-agnostic
   `render_dli_video` with the imaging block marginalized — the six photophysics drawn
   per simulation from the `Nuisance_DLI` artifact and the five camera parameters from
   the SCOPE box, both recorded beside the reaction-diffusion labels. The biology DLI
   smoke (section 2.2) and the long-duration DLI leg (section 3.3) exercise this path.

**Acceptance**: pillars (1)–(3) hold — theta is bit-reproducible at a fixed
seed, the constructed reaction-diffusion system matches the declared model, and
the imaging output is correctly shaped and physically plausible.

### 3.2 Reproducibility — theta-only regression test

The `--seed` flag defaults to `None`, which means **non-deterministic by
design**: each run draws fresh entropy for prior sampling, particle placement,
imaging noise, and network initialization. This matches the inherent
stochasticity of experimental fluorescence data and is the intended production
behavior. To exercise the reproducibility guarantee, pass `--seed` explicitly. This is the
one deliberately seeded check; every smoke in section 2 is run seedless
(`--seed None`).

The acceptance bar is **theta-only** reproducibility — the prior-sampling stage
is fully seeded, the downstream stages are not. Run the pipeline three times at
the same explicit seed:

```bash
DB=<data_bank>            # the data_bank_root from your machine profile
for i in 1 2 3; do
    rm -rf "$DB/Theta" "$DB/Video"
    python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py \
        --condition FAB --total-time-seconds 2.0 --tasks 1 --task-simulations 5 --seed 42
    find "$DB/Theta/SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr" \
        -type f | sort | xargs md5sum
done
```

**Acceptance**:

- **Theta sets are bit-identical** across all three runs (the per-file md5
  listings match line for line). This proves the prior-sampling RNG is seeded and correctly propagated
  through the entry point; if a future code change ever drops that seed
  handling, this check catches the regression immediately.
- **Trajectories (`.h5`), videos (`.zarr`), and inference outputs
  (`.pth`/`.npz`) differ across runs**, and that is expected, not a regression.
  The reaction-diffusion stepper draws from an internal random source that is
  not seedable, so trajectories vary; the videos inherit that variability
  through their trajectory input (even though the imaging pipeline's own noise
  RNGs are fully seeded); and the trained network inherits it through its
  training data. Posterior summaries should converge across runs within
  statistical uncertainty given enough training data, but bit-identity past the
  theta stage is neither expected nor required.

In short: **theta is reproducible at a fixed seed; trajectories, videos, and
trained networks vary by design.**

**Pre-launch fan-out label check.** Because generation is seedless by design,
dataset integrity does not rest on value reproducibility past the theta stage; it
rests on the output file labels being unique across the fan-out, so a silent label
collision cannot let one task overwrite another's output. An ad-hoc diagnostic
asserts exactly that before a large generation fan-out — especially an incremental
grow that appends tasks with a task offset — and also confirms the theta sampler's
seeding behaves as designed (two default draws differ, so no seed is silently
forced; an explicit seed reproduces the draw bit-for-bit). Run it by hand on any
machine with the package installed and a valid `MACHINE_PROFILE`, once per recording
duration (the duration sets the timing label whose labels are checked):

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Seeding_Validation.py \
    --total-time-seconds 2.0
```

It builds path strings and samples small in-memory arrays only — it never reads or
writes the data bank and needs no GPU. It writes no files: it prints one
`[PASS]`/`[FAIL]` line per check and a final `RESULT`, exiting nonzero on any
failure so it can gate a generation launch from a script. This check is a
standalone reproducibility diagnostic, not one of the biology pipeline stages,
and is kept out of the stage dispatcher. See the companion note
`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Seeding_Validation.md` for what each
check covers, how to read a failure, and the precise scope of what it does and does
not guarantee.

### 3.3 Dual-duration checks (2 s and 10 s)

The codebase is duration-parameterized: the recording length is supplied per run
via the required `--total-time-seconds`, and the frame count follows from it.
Validation runs both a short and a long duration end-to-end to confirm the
arithmetic is correct throughout the pipeline — not only at the short duration
where a stale length default would happen to be right.

The frame count is **`frame_count = total_time_seconds / frame_time_seconds`**,
with the frame time fixed at 0.020 s (50 frames per second). So:

- **2 s → 100 frames**: video shape `(100, 256, 256)`.
- **10 s → 500 frames**: video shape `(500, 256, 256)`; trajectory length,
  video frame count, and file sizes all scale five-fold; outputs are namespaced
  by their timing label (`10S_50FPS`) so they never collide with the 2 s
  outputs.

Repeat the smoke tests with the long duration (seedless). The DLI and Inference
legs run on the biology DLI path (section 3.1); the duration arithmetic is also
confirmed through the RDS leg and the derived-frame-count check below:

```bash
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py --condition FAB \
    --total-time-seconds 10.0 --tasks 1 --task-simulations 5 --seed None --verbose
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py --condition FAB \
    --total-time-seconds 10.0 --tasks 1 --task-simulations 5 --seed None --verbose
python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Inference.py --condition FAB \
    --total-time-seconds 10.0 --tasks 1 --epochs 1 --seed None --verbose
```

The inference network's temporal depth is derived from the duration: the video
encoder is constructed with `n_frames = total_time_seconds / frame_time_seconds`
(100 for 2 s, 500 for 10 s), and it asserts that the input frame count matches,
so a duration/data mismatch fails loudly rather than silently truncating.
Confirm the derived frame count:

```bash
python -c "
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming
t = RunTiming(total_time_seconds=10.0, frames=PARAMETERS.simulation.timing)
print('frame_count =', t.frame_count)
"
```

**Acceptance**: both durations run end-to-end without errors; video shapes are
`(100, 256, 256)` and `(500, 256, 256)`; the encoder accepts the matching frame
count at each duration.

### 3.4 Validating a trained posterior and its three point estimates

Beyond confirming the code runs, a trained posterior is validated on data it has
never seen, using the two MAP-recovery stages.

- **Simulated recovery** (`Evaluation.py`): for each held-out EVAL video the
  maximum-a-posteriori parameter vector is estimated and compared to the known
  ground truth. This reports per-parameter recovery accuracy and posterior
  calibration (whether the credible intervals contain the truth at their nominal
  rate — roughly 50% for the interquartile range, roughly 90% for the 5–95%
  interval; under-coverage signals an overconfident posterior, over-coverage an
  underconfident one).

  ```bash
  python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Evaluation.py \
      --total-time-seconds 2.0 --eval-tasks 1 --pool-mode unrestricted
  ```

  Use `--pool-mode unrestricted` for a smoke or check run. The smoke-tested
  posterior is undertrained, so much of its probability mass lies outside the
  prior box; the default `bounded` pool draws candidates by rejection sampling
  within the prior and stalls when almost every draw is rejected. The
  `unrestricted` pool samples the flow directly and does not stall. Switch back
  to the default `bounded` pool only once the posterior is well trained (see
  the pool-mode note below).

- **Real-data application** (`Experiment.py`): the same estimator is applied to
  experimental microscopy videos, which have no ground truth. Each recording is
  split into model-length windows and the inferred-parameter distribution is
  reported per experimental condition. This is the scientific end use of the
  posterior, not a correctness check.

  ```bash
  python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment.py \
      --total-time-seconds 2.0 --pool-mode unrestricted
  ```

Both stages compute and store **three point estimates for every observation**,
always; nothing selects among them (the former `--summary` option is retired
and rejected). They are defined once, in `evaluation.POINT_ESTIMATES`, and each
product records the contract it was made under in a `manifest_json`. That
manifest is validated in full before a product is written or a report rendered
(`artifact_schema.validate_product`: every contract entry present, typed and
consistent; every estimate, score and truth finite; quantiles nondecreasing;
identifiers integer-valued; optional arrays exactly as declared; shards of one
computation only), and downstream
analyses read products only through the schema, with an exact parameter-key
match. The three estimates:

- **MAP** (`map_estimate`): the numerical MAP candidate — the highest-scoring
  point retained by the gradient-ascent optimizer of the flow's log-density,
  initialized from the observation's candidate draws and stepping each
  coordinate in units of its interquartile range over those draws. The
  optimization is unconstrained, so prior support and convergence to a mode are
  not guaranteed; a value outside the prior box is a flow optimum, not a MAP of
  the prior-supported posterior.
- **median** (`posterior_quantiles` at the 0.50 level, located from the
  manifest): the marginal median of the observation's draws, each coordinate
  independently, with linear interpolation between order statistics, in
  estimator coordinates; a physical value is the transform of that quantile. A
  composite that need not be a sampled vector.
- **SGM** (`posterior_sgm`): the sample geometric median of the same draws —
  the complete draw minimizing the summed Euclidean distance to all others after
  dividing each coordinate by its prior width, found exactly (an exact sample
  medoid); a realized draw. This is the *posterior-draw SGM*; an SGM of window
  MAP vectors, one taken in physical coordinates, or the collection-level
  approximation that snaps a Weiszfeld geometric median to its nearest member
  above 20,000 members, is a different quantity and is named as such where it
  appears.

Under the `bounded` pool the draws are posterior draws; under `unrestricted`
they are flow draws that may fall outside the prior's support — the manifest
carries the label. Each stage repeats its recovery (or per-condition) table for
all three, places them side by side in a "Point estimates compared" table, and
reports their pairwise agreement together with the share of observations whose
MAP lies outside the central 90% interval of the draws. A large gap with a high
outside share establishes that the optimized candidate and the draw summaries
disagree; the table does not decide why, and a conclusion about a parameter is
read from all three together. Each stage writes a self-contained report
(figures, tables, arrays, and a live, tail-able `progress.log`) under `Posit/`.

For a small or undertrained posterior whose probability mass can fall outside
the prior box, use the `unrestricted` candidate pool (`--pool-mode unrestricted`)
so candidate sampling does not stall; for a well-trained posterior the default
`bounded` pool (rejection sampling within the prior) is correct **on Evaluation
only** — its EVAL parameters are synthetic prior draws, so the posterior mass
lies inside the box. Experiment stays `unrestricted` even with a well-trained
posterior: the pool mode follows the *data*, not the training quality, and the
experimental recordings are not prior draws (see the pool-mode rule in
section 2.7). Run any stage with `--help` for the full flag list.

#### Validating the point-estimate calculations

The calculations behind the three point estimates are validated before the
analyses are regenerated. The checks are small and focused; the regenerated
analyses are the production-scale test. The order is:

**median correctness → SGM correctness and sampling stability → MAP
optimization → regenerate analyses → scientific interpretation.**

These checks do not establish parameter recoverability or posterior calibration.

**1. Marginal median.** The coordinate-wise 50th percentile of an observation's
draws, calculated in estimator coordinates with linear interpolation, then
transformed into physical units.

- Implementation tests cover the interpolation, the parameter order and the
  order of the transform (`tests/test_median_reference.py`).
- The sampler-to-output path is checked on 5–10 existing recordings.
- No dedicated large-scale median validation is required.

*Completion:* an independent recomputation agrees with the production output.

*Status: complete.* On the ten pilot recordings of the point-estimate
validation utility (five dim, five bright; its companion note records the run),
every stored median equals the recomputation from the stored draws exactly.

**2. Posterior-draw SGM.** The complete draw minimizing the summed Euclidean
distance to the other draws, in estimator coordinates scaled by the prior
widths. It is an exact sample medoid; selecting a complete draw does not
preserve a distribution's correlations.

- Verify summed distances, sample membership, scaling, duplicates and
  deterministic tie handling.
- On 5–10 recordings, including previously unstable cases, compare **N, 2N and
  4N draws**, where N is the production count.
- Use a small, explicitly recorded number of independent repeats at each count.
- Report the per-coordinate standard deviation across repeats, normalized by an
  IQR estimated from a common reference cloud, and the median's variability on
  the same draws.
- Evaluate the selected candidates' average distances against that common
  reference cloud, to distinguish unstable locations from meaningfully worse
  objective values.

*Interpretation:* decreasing variability indicates a sample-count limitation.
Different vectors with nearly equal scores indicate a weakly determined
representative location. Persistent variability alone does not establish an
implementation defect.

*Completion:* characterize the stability and decide whether the production draw
count is adequate. Do not increase it automatically. The large-cloud
approximation of the collection-level SGM kernel (a Weiszfeld geometric median
snapped to its nearest member, used above 20,000 members) requires a separate
check wherever it is used.

*Status: correctness complete; stability characterized; draw count raised.* The
reference tests pass (`tests/test_sgm_reference.py`), and the production SGM
equals a brute-force medoid on 50 real clouds. Stability was measured on the ten
pilot recordings, on which the repeat-to-repeat variability was first seen:
five repeats at 1,000, 2,000 and 4,000 draws against a common 10,000-draw
reference cloud. The SGM's variability falls slowly (median 0.13, 0.11 and 0.09
of the reference IQR), the median's as sample size predicts (0.025, 0.020 and
0.015), and every selected SGM is nearly as central as the most central
reference draw: a weakly determined location with a partial sample-count
effect, not a defect. From 0.1.17 the summaries use 10,000 draws per
observation, drawn in one outer sampler call (about 0.2 s on the measured GPU);
the exact SGM takes about 1 s and 1.5 GiB of RAM per concurrent worker, its
memory growing quadratically with the draw count. More draws raise the numerical
resolution; they do not correct posterior miscalibration. The large-cloud
approximation has not been checked.

**3. Numerical MAP candidate.** MAP extraction optimizes the flow density in
estimator coordinates. The unconstrained procedure does not guarantee prior
support or convergence to a mode.

- Verify score–vector consistency and a returned score no worse than the best
  initial candidate.
- Benchmark on the real trained flow, using identical observations and initial
  candidates across configurations.
- Compare the current settings with scale-aware learning rates and increased
  step and patience budgets.
- Record density gains, stopping reasons, initialization sensitivity and
  runtime.
- Keep the benchmark's own scaling formula and numerical settings labeled
  **benchmark proposal — not adopted**.

*Completion:* resolve the optimizer's behavior and freeze the selected
configuration **before regenerating the analyses**. Agreement with the median
or the SGM is not a convergence criterion.

*Status: configuration adopted in the code and verified on the trained flow; the
freeze follows its review.*

- *Benchmark* (the MAP benchmark companion note; 2,000 EVAL recordings, two
  independent candidate pools each, identical seeds across configurations). The
  0.1.16 optimizer, whose initial step of 0.128 dex is several times the
  posterior IQR of the well-identified parameters, came within 1e-3 nats of the
  best optimum found in 21 % of pools, and its two pools' MAP vectors differed
  by a median 0.25 IQR. Scale-aware steps converged in essentially every pool.
  The benchmark's configurations — per-seed chains, a floor and cap on the step
  scale, a no-stop window after a learning-rate reduction, a 1e-4-nat threshold
  and their budgets — are a **benchmark proposal — not adopted**.
- *Adopted configuration* (`evaluation.optimize_elite`, `InferenceEvaluation`).
  Adam moves `u = (θ − m) / IQR` per coordinate, `m` and `IQR` being the median
  and interquartile range of the observation's candidate pool. Initial learning
  rate 0.05 and floor 0.0005, both in `u`; factor 0.5 after 20 steps without a
  meaningful improvement; stop after 200 such steps; at most 2,000 steps. A
  meaningful improvement is a rise of the best score by more than 1e-3 nats
  since the last one. Every strictly better finite (score, vector) pair is
  retained whatever its size; the tolerance gates only the patience and the
  scheduler, and a learning-rate reduction does not reset the patience. A zero
  or non-finite IQR skips the ascent, returns the best candidate and is
  reported. The patience settings, the floor, the factor and the tolerance come
  from configuration, not the command line; each stage prints the effective
  values, records them in its manifest's `optimizer` block, and logs each
  observation's stop reason.
- *Verification on the trained flow* (the production entry point on the ten pilot
  recordings, two independently seeded pools each, 2026-09-23). The returned
  score equals the re-evaluated density at the returned vector in all 20 pools
  and exceeds the best initial candidate by 0.02 to 0.59 nats. Every pool
  stopped on patience, after 219 to 312 steps. Evaluated on the same machine,
  the JUPITER benchmark's reference optimum lies within 4.5e-4 nats of the
  returned score, and an L-BFGS polish from the returned vector gains at most
  5.6e-4 nats. The two pools agree to a median 0.0009 IQR (maximum 0.03). The
  effective settings equal the requested ones.

**4. Regenerated analyses.** Every Evaluation and Experiment product of the
affected estimators, and every analysis reading their MAP arrays, is recomputed
under the frozen configuration; products made earlier are refused on read.

*Status: the Experiment stage of the multiple-dye baseline and of `CAP256` is
regenerated (2026-09-24, `DETECTOR_WORKFLOW.md` §9.8); their Evaluation stages,
the one-dye stages and the derived analyses are pending.* All 1,200 ascents
stopped on patience; the baseline's MAP lies within 0.07 dex of its posterior
median on every parameter and inside its own central 50 % interval in every
window. What remains of the MAP's behavior is the shape of each estimator's
learned density, which the regenerated products and a probe on the real
checkpoints make explicit (§9.8): `CAP256`'s learned density has two competing
joint solutions of nearly equal height for bleaching, near the two ends of the
prior, and the seed-dependent ascent selects between them from window to window
while the median and the SGM stay at the prior center; its `sigma_r` and `mu_r`
densities are skewed, so the MAP sits a stable half-IQR below the median; the
baseline's ascents converge consistently on the probed windows and its three
estimates coincide, moving together along a recording. Independent candidate
pools reproduce every MAP within its region to 0.04 dex: the instability is the
density's shape together with the ascent's initialization, not the corrected
bookkeeping.

**Reporting rule.** For each method, distinguish **implementation
correctness**, **sampling or optimization stability**, and **accuracy against
synthetic truth**. Close agreement among the point estimates does not establish
that a parameter is recoverable. The regenerated Experiment products give the
empirical case for reading the three estimates together: a MAP is one point of
the density, displaced by a skewed marginal and switching between competing
high-density regions, while a median and an SGM that stay at the prior center
with an interval spanning most of it agree on a center without establishing a
measurement. Each of these readings needs the other estimates and the interval
width to be told apart from a well-determined center; no single estimate
carries it.

---

## 4. Troubleshooting

### Import-time configuration errors

If `python -c "import srm_and_sbi_monomer_dimer_alp.parameterization"` raises:

- **"MACHINE_PROFILE environment variable is not set"** — set it (selecting the
  active profile, section 1.4).
- **"machine_profiles.toml not found"** — create it from the template
  (configuring `machine_profiles.toml`, section 1.3).
- **"Profile '…' not found"** — the profile name in the environment variable
  does not match a section in the TOML file.
- **"missing required keys"** — the profile is missing one of the required keys
  listed under configuring `machine_profiles.toml` (section 1.3).
- **"… is not a directory"** — `script_bank_root` or `data_bank_root` does not
  exist on disk; create it or fix the path.

### `ImportError: No module named 'psutil'` (or `ipython`)

These packages are not pulled in transitively and must be installed explicitly;
without `psutil` the simulation stage fails immediately. Reinstall the
environment per `env_snapshots/README.md` (the gotchas section), or add the
missing package to the active environment.

### ReaDDy import error

ReaDDy is conda-distributed. If `python -c "import readdy"` fails, the
environment was not built correctly; rebuild it per `env_snapshots/README.md`
and verify with `conda list readdy`.

### `CUDA out of memory` during inference

The default batch size over full-resolution videos can exceed a small GPU's
memory. Reduce `--batch-size` to 1 or 2, train on a larger GPU, or use the CPU
backend (the inference smoke test's small-GPU workaround, section 2.3).

### Wrong PyTorch backend or CUDA/ROCm mismatch

If PyTorch reports a backend mismatch at runtime, the installed wheel does not
match the hardware. Reinstall the matching build using the PyTorch backend table
in `env_snapshots/README.md` (AMD → ROCm; NVIDIA → a `cuXXX` wheel whose CUDA
version does not exceed the driver's maximum; no/small GPU → the CPU build).

### `--resurrect` fails with "file not found"

`--resurrect` requires a prior checkpoint at the expected path. Run inference
once without `--resurrect` first to produce the initial checkpoint.

### DLI fails to load the theta set

DLI reads the theta set written by RDS. If it reports a missing file, ensure the
matching RDS run completed first and produced both the trajectories and the
theta set, with the same `--tasks` and `--task-simulations`.

### A long generation run hangs without finishing

If a long, memory-tight generation run stalls (rather than erroring), confirm
resource usage stays flat with the simulation stages' `--probe` flag, which logs
per-simulation thread, file-descriptor, and memory counts. Per-simulation
reaction-diffusion kernels are released inside the generation loop so that
threads and resident memory stay flat across arbitrarily many simulations; the
`--probe` instrumentation is how that is confirmed.

---

## 5. Success criteria

A correctly set up and validated installation has:

- A working `SRM_AND_SBI_ENVY_V0` environment (or a reused compatible one) with
  the package installed editable, and a `machine_profiles.toml` configured for
  the machine.
- All six biology smoke-test invocations (RDS, DLI, Inference, Inference
  `--resurrect`, Evaluation, and Experiment — sections 2.1–2.4b) running
  end-to-end on minimal inputs, meeting the biology acceptance in section 2.4b.
  (The biology DLI stage renders
  through the shared `render_dli_video` with the imaging block marginalized — see the
  imaging-pipeline pillar in section 3.1 — so it requires the `Nuisance_DLI` artifact to
  be built first, per the DLI smoke prerequisites in section 2.2.)
- The Detector calibration five-stage smoke test (section 2.5) running end-to-end,
  seedless, with its small overrides.
- The three-pillar semantic-equivalence checks holding: theta bit-reproducible
  at a fixed seed, the constructed reaction-diffusion system matching the
  declared model, and imaging output correctly shaped and physically plausible.
- The theta-only reproducibility regression passing: theta sets bit-identical
  across three same-seed runs, with trajectories, videos, and trained networks
  varying by design.
- Both the 2 s and 10 s configurations running end-to-end with the correct
  frame counts (100 and 500).
- A trained posterior validated by MAP recovery on the held-out EVAL set, with
  recovery accuracy and posterior calibration reported.

---

## 6. Assurance roadmap (not yet implemented)

Everything above describes validation the repository performs today. The items
below are engineering-assurance work the project has deliberately deferred
behind the scientific program. They are listed here so a reader can distinguish
completed validation from planned assurance — none of them exists yet.

- **Automated pytest suite.** The checks in section 3 are run by hand; an
  automated suite would pin them as regression tests. The natural targets:
  parameter-table schemas and ordering (the theta layout the estimator depends
  on), estimator artifact save/load round-trips with checksum and schema
  rejection of a corrupted or mismatched file, shard/merge coverage and balance
  invariants for the sharded stages (every video assigned exactly once, ranks
  balanced to within one video), the deterministic renderer components (PSF
  pixel integration at a fixed input), the EMCCD noise model's mean/variance
  behavior against its analytic form, and a CLI `--dry-run` smoke driven by a
  temporary machine profile so the entry points are exercised without a data
  bank.

- **CPU-only continuous integration.** A CI job running compilation, the unit
  tests above, and packaging checks on every push would catch import breakage,
  schema drift, and dependency-resolution regressions before they reach a
  compute run. CPU-only keeps it runnable on any hosted runner; the GPU paths
  remain covered by the smoke tests.

- **Tagged releases plus a `CITATION.cff`.** Versions are currently identified
  by the package version string; tagged releases would make each version
  immutable and retrievable, and a `CITATION.cff` would make them citable, so a
  result can name the exact code that produced it.

- **A minimal runnable example.** A small reference dataset with expected
  outputs would let a new user (or a reviewer) confirm an installation
  end-to-end in minutes, without generating data first — the smoke tests verify
  that the code runs, but only against data the user must produce.

- **A per-release machine-readable validation manifest.** Each release would
  ship a manifest recording the commit, the environment specification, the
  dataset checksums, the per-parameter recovery and calibration metrics, and a
  pass/fail verdict per criterion in section 5 — turning "validated" from a
  narrative claim into a checkable record.

- **Decoupling module import from machine configuration.** The configuration
  currently validates at import time (section 1.4), which is a deliberate
  fail-loud choice but couples every import — including a test collector's — to
  a valid `MACHINE_PROFILE`. Loading the profile at the CLI boundary instead
  would keep the fail-loud behavior for runs while letting the package import
  cleanly on an unconfigured machine.

- **An executable derivation script for the localization-derived imaging
  priors.** The imaging prior ranges trace to localization analyses of the
  public reference recordings; an executable script reproducing that chain —
  from the public accession to the fitted values — would make the prior
  derivation itself reproducible rather than only documented.
