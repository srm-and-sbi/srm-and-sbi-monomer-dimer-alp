# Receptor total for a declared-configuration render — companion note

Part of the posterior-predictive video check. The files of the check, what each is needed for, and the order
of a check at a declared configuration are listed in `SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md`
(*The files of the posterior-predictive video check*).

`SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Count_Match.py` chooses the receptor total `N_R` of a
posterior-predictive render at a declared reaction-diffusion configuration
(`SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md`, *A declared reaction-diffusion
configuration*). The receptor total is the only per-recording adjustment of such a render, and the value
chosen is a **rendering setting** that approximates the recording's opening spot density. It is not a
biological estimate. It is a post-hoc analysis, never wired into the stage dispatcher.

## Method

Short clips (the first 2 s by default) are rendered by the production chain the posterior-predictive engine
uses (`simulate_and_render`): the RDS stage's system, the DLI stage's labeling at the condition's declared
occupancy, and its renderer with the tagged `Nuisance_DLI` imaging vector. Spots are counted on the
recording and on every render with the direct estimators' detection (8-bit stored levels, 4 sigma, fit
half-width 14 px, SCOPE box center). The detected count of a render is written `c = eta * V`, with `V` its
visible hosts at frame 0 and `eta` the detected-per-visible-host ratio, which carries the missed detections,
overlap, saturation, in-window bleaching and field exits at that density. With the expected visible hosts
per subunit `v` under the declared visibility and initial composition, each iteration proposes
`N_R = c_exp / (eta * v)`, moving at most a factor of two per step, until the rendered and proposed values
agree within the tolerance.

Every render passes one count test before it is made: the first, each update and the check renders. No
render exceeds `--max-count` (10000), and none lies outside the prior box of `N_R` (316-3162, the training
support) without the user's decision, which `--allow-outside-prior` records. Without it, a count outside
the box stops the recording's match before anything is rendered there.

Normalizing by each render's realized initial visibility reduces the labeling-count variability of a single
render. The remaining simulation variability, and the dependence of `eta` on density, arrangement, overlap,
dye multiplicity and motion, are not removed (`E[eta V]` is not `E[eta] E[V]` in general). The 3 %
tolerance is an iteration criterion, not an uncertainty of `N_R`. One independent check render at the
accepted value (`--check-renders`, default 1) reports the count it produces beside the recording's.

## Statuses

Only `matched` (inside the prior box) and `matched_outside_prior` carry a count. `matched_outside_prior`
occurs only under `--allow-outside-prior`, is flagged, and is entered only by the user's decision. Every
other status keeps its diagnostics, carries no count, and makes the run exit with status 2:

| status | meaning |
|---|---|
| `undefined_no_experimental_detections` | the recording's opening window has no accepted spot |
| `undefined_no_visibility` | the declared labeling and composition give no visible host |
| `undefined_no_visible_hosts` | a render has no visible host at frame 0, so `eta` is undefined |
| `undefined_no_synthetic_detections` | a render has no accepted spot |
| `undefined_nonfinite_update` | the proposal is not a finite positive count |
| `unresolved_saturation` | count matching unresolved under detector saturation: across a step of at least 25 % the detected count responds with an elasticity below `--min-elasticity` (0.3) while a larger count is proposed; no value is reported |
| `unresolved_max_count` | the next count (the first included) or the converged one exceeds `--max-count` (10000); nothing is rendered above it |
| `user_decision_outside_prior` | the next count (the first included) or the converged one lies outside the prior box: nothing is rendered there, and the user decides whether to render it (`--allow-outside-prior`) |
| `not_converged` | the iteration budget is exhausted |

An `--initial-count` beyond either limit stops the run before any recording is read.

## What it does and does not establish

A render at the accepted value has the recording's opening density by construction. That density is not
evidence of realism. Later-window densities remain checks, since only the opening window is used. Nothing is
clipped. The accepted value is entered by hand in the declared configuration's `per_cell` table.

## Usage and outputs

```bash
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Count_Match.py \
    --total-time-seconds 2 --kind MET-FAB --cell 0 5 26 --nuisance-tag REF --seed 20260928 \
    --declared-rds Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Declared_RDS_FAB.toml
```

Results are written to
`<data_bank>/<posit>/<alias>_<timing>_Posterior_Predictive_Video/Count_Match/<SCENARIO>_<TAG>[_<attempt>]/`,
one `.json` and one `.md` per recording. A run refuses, before any computation, to start when a requested
recording's result already exists in its attempt folder; `--attempt-label` names a distinct folder. Each
result records:

- the status and reason;
- the recording's count;
- every iteration and the check render;
- the labeling plan and the imaging artifact's identity;
- the declared file's path and SHA-256;
- the resolved values of all eleven parameters;
- the settings and the package version.
