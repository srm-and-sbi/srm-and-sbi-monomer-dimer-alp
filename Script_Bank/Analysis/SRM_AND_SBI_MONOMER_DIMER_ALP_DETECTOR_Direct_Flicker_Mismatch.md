# Flicker mismatch study — method and usage

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Mismatch.py`. This utility looks
for where along the observation chain the direct flicker-rate estimator's bias arises. It explains
what the harness computes, how to run it, and how to read the result, without reading the code.

It is a development diagnostic. It renders its own scenes and reads no EVAL task, so it consumes none
of the development or reserved data of `DETECTOR_WORKFLOW.md` §9.6. It tunes nothing and reaches no
verdict. Like the estimators it serves, it is never wired into the stage dispatcher.

## Why this exists

The direct flicker estimator reads `lambda_rate` off the pooled, detrended, lag-1-normalized
ln-intensity autocorrelation of linked spot traces, matched against a single-dye Ornstein–Uhlenbeck
model arm cut to the observed spans and detrended identically. On the two-second multiple-dye
development recordings it ran fast: +0.11 dex overall, +0.26 dex at the slowest rates and +0.16 dex in
the dim half of the brightness prior (§9.6). The model arm reproduces the spans and the detrend. It does
not reproduce the rest of the path a trace takes: a spot is the sum of its dyes, a dim frame may go
undetected, a detected frame carries photometry noise, the linker may split or merge emitters, and
detections that belong to no single emitter enter the traces. The development record left open which of
these contributes the bias. This harness compares them under known truth.

## What it computes

Every scene is rendered through the production renderer with known imaging values. Subunits diffuse
at 0.05 µm²/s, and bleaching is off by default. Dye counts follow the bare FAB dye-count law at probe
occupancy 1 by default: about 81 % of subunits carry a dye. A production tier also applies the
condition's probe occupancy (0.155 for MET-FAB), which thins the visible subunits without changing a
visible one's dye count; here the density of visible spots is set by `--n-subunits` instead. The
renderer draws every dye's brightness with a seeded call, and calling it again with the same arguments
and seed returns the very photons the video was rendered from. So the true photon series of every dye,
and of every spot, is known without an instrumented renderer. The production measurement then runs on
the rendered video: detection, spot fits and linking at frame stride 1, exactly as the estimator runs
them.

The same shape match is applied to seven versions of the traces. Each changes one thing against the one
before it:

| level | traces | changes |
|---|---|---|
| `dye_truth_full` | every dye's true photons over the whole recording | nothing: the model arm's own assumption, so its error is the method's floor |
| `spot_truth_full` | every visible subunit's summed true photons, whole recording | dye multiplicity |
| `spot_truth_span` | that series from the subunit's first to its last detection, without gaps | the selection of spans |
| `spot_truth_detected` | that series at the detected frames only | detection gaps: the censoring of dim frames |
| `fitted_oracle` | the fitted amplitudes of the matched detections, grouped by true identity | photometry noise |
| `fitted_linked` | the same matched detections, grouped by the production linker | linking only |
| `production` | every accepted detection, grouped by the production linker: the estimator as it runs on a tier | the detections outside the matched set |

A detection is matched to the nearest visible subunit of its frame within 1.5 px. Within one frame a
subunit keeps one detection at most, the nearest, so a trace never holds two values in one frame. The
detections outside the matched set are the fits with no visible subunit within 1.5 px and the second
fits near a subunit already matched in that frame. `fitted_oracle` and `fitted_linked` read exactly the
same detections, so their contrast isolates the grouping; `production` then adds the detections outside
the matched set, and its contrast with `fitted_linked` isolates their inclusion. Every level that uses
detections applies the estimator's own trace rule (at least three points and a span of at least 40
frames) and its model arm of 4000 traces per grid point, and refuses as the estimator does: fewer than
five traces, or fewer pooled lag-one pairs than the estimator requires.

## How to read the result

The report tabulates the mean signed log10 error of each level by true rate and by brightness, each
cell with the number of scenes that produced an estimate. Cells of different levels can therefore
average different scenes, and one column averages only the scenes where every level produced an
estimate. A second table gives the contrast between adjacent levels, paired within scenes so that
scene-to-scene scatter cancels, with the number of scenes that entered each pair. A third counts the
scenes without an estimate at each level by reason, and the estimates that fell on a grid edge. The
figure draws the errors against the true rate over the scenes where every level produced an estimate,
one line per level and one panel per brightness.

A contrast suggests a contribution of that step under these scenes and this ordering. The steps are
added in one fixed order, and an effect measured after one step can differ in size, even in sign, when
taken in another; the contrasts are diagnostic accounting, not a causal decomposition. A step whose
contrast is near zero in these scenes is not shown to be negligible elsewhere.

The default grid puts the true rate at 1.25, 2, 4 and 8 per second, inside the model grid of 1 to 14, so
an estimate can fall on either side of the truth. Brightness runs at 120, 240 and 480 photons per dye;
120 lies in the dim operating subgroup of §9.6. Each of the 12 combinations is rendered twice.

What the harness shows holds for its scenes: one diffusion coefficient, subunit counts near 200, the
other imaging values at their prior centers. A finding is confirmed on the development tasks of the tier
before it changes the estimator, and the changed estimator is judged once on the reserved tasks.

## How to run

```
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Mismatch.py \
    --workers 16
```

`--lambda-grid`, `--mu-pc-grid` and `--replicates` set the scene grid. `--labeling single` gives every
subunit one dye, which removes multiplicity from every level. `--prob-photo-bleach 0.056` turns
bleaching on at the center of its prior. `--dry-run` prints the plan and applies the refusals without
rendering anything. One scene takes about two minutes on one core, most of it in the seven model arms;
the default 24 scenes take about five minutes on 16 workers. On JUWELS the wrapper
`Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Direct_Estimator.sh` runs it with
`ESTIMATOR=Flicker_Mismatch`.

## Outputs

`<data_bank_root>/Posit/<alias>_<timing>_Direct_Flicker_Mismatch_<labeling>[_BLEACH_<p>][_<suffix>]/`
holds `report.md`, `direct_flicker_mismatch.npz` (per scene and level: the true values, the estimate,
the signed error, the reason for a missing estimate, the grid-edge and refinement flags, the trace count
and median span; per scene, the matched detections and the two kinds outside the matched set),
`summary.json` (per level: estimates produced, reasons, means over the scenes with an estimate and over
the scenes where every level produced one; per step: the paired contrast and its count),
`provenance.json` (command, host, versions, implementation hash at startup and at write) and the figure.
The folder name carries the labeling (`FAB_LAW`, `INLB_LAW` or `SINGLE_DYE`) and, when bleaching is on,
its probability (`BLEACH_0p056`), so the documented variants never share a folder. A run never reuses a
folder: an existing one is refused before anything is rendered, and any other change of settings needs
its own `--run-suffix`. A run whose implementation changed while it executed is marked `INVALID` and
exits with status 3.

## References

- `DETECTOR_WORKFLOW.md` §6.5 (the flicker model), §9.6 (the frozen rules, the development outcome of
  the flicker estimator, and the declared split of the EVAL tiers).
- `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Flicker_Rate.md` — the estimator this harness
  diagnoses.
