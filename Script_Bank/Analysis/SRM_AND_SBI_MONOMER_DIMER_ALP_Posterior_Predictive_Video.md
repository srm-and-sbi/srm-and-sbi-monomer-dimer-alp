# Posterior-predictive video — usage and interpretation

Authoritative companion for both workflows' posterior-predictive video analyses and its viewer
notebook `notebooks/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb`. The
script renders a synthetic video from the MAP imaging estimate of one experimental recording and
persists it beside a static comparison figure; the notebook views the persisted clip. This
note explains how to run the script and how to read its outputs, so the check can be used
and understood without reverse-engineering the code.

## The files of the posterior-predictive video check

Every file the check needs lives under the name `SRM_AND_SBI_MONOMER_DIMER_ALP_[DETECTOR_]Posterior_Predictive_Video*`,
beside this note in `Script_Bank/Analysis/`, except the engine module, the viewer notebook and the tests:

| file | role | needed for |
|---|---|---|
| `srm_and_sbi_monomer_dimer_alp/posterior_predictive_video_runner.py` | the engine both workflows share: resolves the imaging, the reaction-diffusion block and the labeling, simulates and renders one clip through the production chain (`simulate_and_render`), writes the clip and the figure | every render |
| `SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.py` | biology entry point (thin shim) | biology renders, including renders at a declared configuration |
| `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py` | detector entry point (thin shim) | detector renders |
| `SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md` | this note: how to run and read the check (authoritative for both workflows and the notebook) | reading |
| `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.md` | the detector shim's pointer to this note | reading |
| `SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Declared_RDS_FAB.toml` | the declared reaction-diffusion configuration for the MET-FAB check renders: each of the eleven values with its basis and source, and the receptor total per recording. It is an input of this analysis, specific to the MET-FAB recordings compared, confirmed by the user before rendering; it is not a generic model setting (section *A declared reaction-diffusion configuration*) | MET-FAB renders at a declared configuration (`--declared-rds`) |
| `SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Count_Match.py` | chooses each recording's receptor total for the declared configuration: the rendering setting that approximates the recording's opening spot density | filling the configuration's `per_cell` table |
| `SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Count_Match.md` | that script's method, statuses and outputs | reading |
| `notebooks/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb` | the viewer: experimental and synthetic clip side by side, scrubber and real-time player at the recording's frame rate (50 fps) | viewing a clip |
| `tests/test_posterior_predictive_video.py` | regression tests of the engine's output refusal and figure labels and of every path of the count match (no simulation) | after a change to the engine or the count-match script |

The order of a check at a declared configuration: confirm the configuration with the user; choose the
receptor total of each recording (the count-match script); render each clip (the biology entry point with
`--declared-rds` and `--nuisance-tag REF`); compare the pixel histograms in the static figure; view the clip
in the notebook.

## One engine, two workflows — which block the MAP supplies

Both entry points are thin shims over `srm_and_sbi_monomer_dimer_alp.posterior_predictive_video_runner`.
The mechanics are identical; what inverts is **which half of the model the MAP supplies and which
half is held fixed**:

| | MAP supplies | held fixed | system built |
|---|---|---|---|
| **biology** (`..._Posterior_Predictive_Video.py`) | the **11 reaction-diffusion** parameters, or none when they come from a declared configuration (`--declared-rds`) | imaging, at the calibrated `Nuisance_DLI` vector + MET SCOPE camera | **full reactive** system |
| **detector** (`..._DETECTOR_Posterior_Predictive_Video.py`) | the **6 imaging** parameters | the reaction-diffusion block, drawn from the biology prior, pinned as a nuisance, or declared | **full reactive** system |

This is not cosmetic: the comparison figure labels each block by the role it plays in the run that
produced it — labeling a fixed block "INFERRED" (or an inferred one "NUISANCE") would invert the
reader's conclusion about what any visible mismatch implies. Both workflows build the same reactive
system; only the source of its eleven reaction-diffusion parameters differs. The recording's condition (`--kind`) selects
the reaction network the scene is simulated with (the condition's declared association setting: on under
MET-INLB, off under MET-FAB), the condition-specific estimator namespace, the condition's calibrated `Nuisance_DLI` on the biology
path, and the static labeling law the render uses (`--labeling-law` overrides it for a sensitivity
render).

In both workflows the five SCOPE camera parameters are pinned to their MET values rather than drawn:
the comparison is against one specific real acquisition, so a random camera draw would inject
variation unrelated to the question.

## The render follows the production observation layer

Every render labels its trajectory exactly as the training videos were labeled. The labeling goes through
the one path the DLI stage of both workflows and the horizon audit share (`labeling.resolve_labeling`,
`labeling.label_trajectory`): the condition's law (MET-FAB Poisson(1.64), MET-INLB Bernoulli(0.5)) and its
declared (MET-INLB 0.5) or derived (MET-FAB 0.155) probe occupancy, applied by each subunit's initial
molecular species, so about 12.5 % of MET-FAB subunits and 25 % of MET-INLB subunits carry a dye.
`--labeling-law` and `--occupancy` override them for a sensitivity render with the DLI stage's grammar and
are recorded as overrides. The reaction-diffusion system, the timing, the geometry (the whole 256 x 256 px
field, open lateral boundary), the trajectory handling, the renderer and the eleven-key imaging order are
the production ones (`simulate_and_render`).

What differs from a production draw is declared: the five camera values are pinned to the MET acquisition
values, which lie inside the SCOPE box the DLI stage draws from; the reaction-diffusion block is the MAP,
a prior draw, a pinned vector, or a declared configuration; and the clip is stored at 16 bits after a
clip to [0, 65535] (the recording's domain), where the training videos store the 8-bit conversion
`io.convert_video_dtype`, which the direct estimators and the count match apply to both videos.

`--seed` is used as production uses it: it seeds the placement and the render directly and the labeling
through `labeling.labeling_rng` (the DLI stage's stream for task 0, simulation 0). numpy's `SeedSequence`
pads missing entropy words with zeros, so that labeling stream coincides with the bare-seed stream: the
production convention, retained; the separation of those streams is not established. ReaDDy's dynamics
stay OS-seeded, so a seed does not reproduce a trajectory.

## A declared reaction-diffusion configuration (`--declared-rds`)

Where no estimate is to be used -- a prediction check before a biology estimator exists, or any render
whose biology must be an explicit assumption -- the reaction-diffusion block comes from a TOML file
(`load_declared_rds`). It states each of the eleven parameters in physical units under
`[parameters.<KEY>]` with a non-empty `basis` and `source`, and exactly one of `value` or `per_cell` (a
table by cell index, for the receptor total). `[scenario]` names it and its condition, which must be the
recording's. `--set-rds KEY=VALUE` overrides one value and is recorded as an override. A value outside the
biology prior box is flagged in the console, the clip and the figure, never clipped.

**On the biology path, the configuration is presented to the user and confirmed before any render.**
Prior centers, literature values and inferred values are not interchangeable, and none is selected
silently. The receptor total is the only per-recording adjustment: a rendering setting that approximates
the recording's opening spot density, chosen through the forward model by
`SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Count_Match.py` (companion note beside it), not a
biological estimate. The match renders nothing outside the prior box without the user's decision, and a
value outside it, or no value under detector saturation, goes back to the user. The MET-FAB configuration is
`SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Declared_RDS_FAB.toml`.

The imaging of a declared-configuration render comes from the tagged `Nuisance_DLI` artifact
(`--nuisance-tag REF`), read as the biology DLI stage reads it: the same loader and tag resolution, the same
key-order check, and the same accessor for a fixed vector.

**Biology's imaging vector is read at run time, never hardcoded.** Its six values exist only inside
the `Nuisance_DLI` artifact — they appear in no source file — so hardcoding them would drift silently
from whatever the training videos were actually generated with. If that artifact ever holds more than
one vector, the engine takes its Sample Geometric Median rather than an arbitrary row, and the report
says so.

## Selecting the MAP vector: `--map-source`

- `chunk` — one specific window's estimate.
- `cell-sgm` (**default**) — an **SGM of that cell's window MAPs**: the Sample Geometric Median over
  the cell's per-window MAP candidates, so the render uses an actual window's vector whose
  coordinates co-occurred (it avoids composing coordinates, it does not preserve the collection's
  correlations). It is a different quantity from the posterior-draw SGM the Experiment product
  stores per window (`posterior_sgm`).
- `cell-median` — the per-dimension median. Faster to explain, but it composes coordinates that need
  never have co-occurred, and for a *render* that matters: the simulator is then asked to realize a
  combination no chunk supported. Retained as an option; not the default.

This is a post-hoc, ad-hoc analysis — a visual posterior-predictive check of a workflow's MAP
estimates against one experimental recording. It is not one of the canonical pipeline stages and
is kept out of the stage dispatcher.

## What it does

Each workflow's Experiment stage stores three point estimates of its inferred parameter block for
each experimental window, keyed by `(kind, cell, chunk)`; this script renders from the MAP candidate
(`map_estimate`, read through the artifact schema, which refuses an obsolete product). It takes one
such estimate, renders a
synthetic recording under it — with the other block held fixed or drawn as the table above
specifies — at the experimental recording's own length, and places the two side by side. A close
match is evidence that the estimated model reproduces how the experimental recording looks; a poor
match points to a parameter the estimate missed.

The synthetic's **motion** is a fresh stochastic draw, not the experimental track — no MAP pins the
specific trajectory — so the comparison reads statistical **appearance** (point-spread size,
brightness, noise, flicker), not the specific molecular motion.

## How to run it

Run on a machine that holds the MAP database and the experimental recordings, under the render
environment (the project package plus ReaDDy). Preview first with `--dry-run`, which resolves
the inputs and the output names without simulating:

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py \
      --total-time-seconds 2.0 --kind MET-FAB --cell 3 [--map-source chunk --chunk 5] \
      [--seed 0] [--dry-run]

A declared-configuration render on the biology path (no MAP read), imaging from the tagged artifact:

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.py \
      --total-time-seconds 2.0 --kind MET-FAB --cell 0 --nuisance-tag REF \
      --declared-rds Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Declared_RDS_FAB.toml \
      [--seed 0] [--run-label REF_DECL] [--dry-run]

Arguments:

- `--total-time-seconds` — the trained estimator's model window (e.g. `2.0` → `2S_50FPS`). It
  locates the MAP database; it does **not** set the render length.
- `--kind`, `--cell` — the experimental condition (`MET-FAB`, the monomer control and the
  default, or `MET-INLB`, the dimer condition) and the cell index, validated against the
  database.
- `--chunk` — selects one MAP entry for `(kind, cell)`; required only for
  `--map-source chunk`.
- `--map-source` — `chunk` (one specific window's MAP, at the selected `(kind, cell, chunk)`),
  `cell-sgm` (the **default**; an SGM of that cell's window MAPs — a real window's `map_estimate`
  whose coordinates co-occurred), or `cell-median` (the per-dimension median
  over that cell's chunk MAPs). `--chunk` is ignored in both cell modes. See *Selecting the MAP
  vector* above.
- `--experiment-span-seconds` — recording length used only to locate the `.tif` (default
  `20`); the render length is read from the `.tif`'s own frame count.
- `--display-norm` — color scaling for the comparison figure's frame panels: `full`
  (default; a shared full-range `[min, max]` window over both the experimental and synthetic
  pixels, so identical intensities map to identical colors and nothing is clipped), `autoscale`
  (each displayed frame stretched to its own min/max, per-frame), or `percentile` (a fixed
  whole-clip `[min, p99.99]` window). The notebook uses the identical convention, so a given
  frame renders the same in the static figure and the notebook. It does not change the stored
  pixels.
- `--seed` — RNG seed, used as production uses it (above); each run otherwise draws a fresh motion
  realization (the check reads statistical appearance, not the specific track).
- `--labeling-law`, `--occupancy` — sensitivity overrides of the condition's labeling, with the DLI
  stage's grammar; the default is the production labeling.
- `--declared-rds`, `--set-rds` — a declared reaction-diffusion configuration and single-value
  overrides of it (above).
- `--run-label` — a token appended to the output stem. A run never overwrites (*Outputs*), so another
  attempt or a variant of an existing render needs its own label.

## Outputs

Written to `<data_bank>/<posit>/<alias>_<model_window>_Posterior_Predictive_Video/`, with
`<stem> = <alias>_<model_window>_<descriptor>[_Fixed_Nuisance]_<KIND>_Cell_<cell>[_Chunk_<chunk>]_<clip_span>[_<run_label>]`.
The `<descriptor>` names the imaging source: `MAP_Estimate` for `--map-source chunk` (the only
mode carrying the `_Chunk_<chunk>` token), `MAP_Estimate_SGM` for the default `cell-sgm`,
`MAP_Estimate_Median` for `cell-median`, or `Fixed_Imaging` for a `--fixed-imaging-parameters`
render (no MAP database read), or `Declared_RDS` for a biology render at a declared configuration (no MAP
database read). `_Fixed_Nuisance` appears when the RDS nuisance is pinned with `--fixed-nuisance-RDS`,
`_Declared_RDS` when a detector render takes a declared configuration, and `--run-label` appends a sanitized token at the end.

A run never overwrites. It names its three files before any work and refuses to start when one of them
exists, and `simulate_and_render` refuses an existing trajectory instead of deleting it. The stem carries
neither the nuisance tag nor the declared configuration's identity, so another attempt or a variant of the
same recording (another `--nuisance-tag`, `--declared-rds` file or `--set-rds` override, or a repeat)
needs its own `--run-label`. The dry run reports files that already exist. Files:

- `<stem>_Synthetic_Video.npz` — experimental and synthetic frames (16-bit, non-negative) plus
  provenance: the imaging vector (`imaging_physical`, `imaging_keys`) and its source
  (`imaging_record_json`: the artifact identity with its tag and selection record, or the MAP source);
  the reaction-diffusion vector (`rds_provenance`, `rds_keys`), its source (`rds_source`: map, declared,
  pinned or drawn) and record (`rds_record_json`: for a declared configuration the file path, SHA-256 and
  text, every value's basis, source and origin, and the overrides), and the keys outside the prior box
  (`rds_outside_prior`); the labeling (`labeling_law`, `dye_counts`, `labeling_row` with
  `labeling_columns`, `occupancy_source`, `occupancy_monomer`, `occupancy_dimer`, `labeling_record_json`);
  the selection, the seed, `synth_label` and `package_version`.
- `<stem>_Comparison.png` — the static side-by-side figure (below).
- `<stem>_Trajectory.h5` — the drawn reaction-diffusion trajectory (provenance; regenerable
  from `--seed`).

The `<model_window>` token (e.g. `2S_50FPS`) names the trained estimator; the `<clip_span>`
token (e.g. `20S`) is the rendered clip's own length — a 20 s clip from a 2 s-window estimate
is not the same as one from a 10 s window, so both appear in the name.

## How to read the comparison figure

A 2×4 grid. Column 0 is **EXPERIMENTAL** and column 1 is **SYNTH**, each showing a mid frame
over its max projection, so a frame and its projection sit in the same column. Column 2 holds
the shared **pixel-intensity histograms** — log-y over the full range (top) and linear-y
through ~p99.99 (bottom, essentially the whole range bar the top-0.01% hot-pixel sliver). Column 3 holds the syn/exp **ratio-per-quantile match plot**
(top) over a **quantile table and provenance text box** (bottom). The provenance lists the eleven
imaging values and the eleven reaction-diffusion values in absolute units, each block under the role it
plays in the run (INFERRED, FIXED, NUISANCE or DECLARED), with out-of-prior values flagged. The title
and the synthetic panel name the synthetic source in the words the clip file stores (`synth_label`,
e.g. *SYNTH (declared reaction-diffusion)*), which the notebook also shows. The synthetic panel names a
MAP selection only when a MAP was read.

- The **frame panels** show single-frame appearance — point-spread size, brightness, and
  per-frame noise. Read them for whether a synthetic frame looks like an experimental one.
- The **max projections** summarize the whole clip; they differ by design, because the
  synthetic motion is a fresh draw, not the experimental track.
- The **histograms** (ADU) show the pixel-intensity distributions; a close overlap is the
  evidence the imaging model matches. Both series are the stored, non-negative frames.
- The **ratio-per-quantile plot** is the direct "do they match?" read — the synthetic/experimental
  ratio across quantiles (min, p0.01, median, p90, p99, p99.9, p99.99, max), a flat line on 1.0
  being a perfect match. It exposes tail mismatches the log histogram can hide.

## Viewing interactively — the notebook

The notebook is a portable, pure viewer: it needs only `numpy`, `matplotlib`, and
`ipywidgets` — no project package, no `MACHINE_PROFILE`, and no ReaDDy — so it opens on any
machine, not just the one that rendered the clip. Step by step:

1. **Get a clip.** Render one with the script above on the data machine, then copy its
   `*_Synthetic_Video.npz` to the machine where you will view it. The `.npz` is
   self-contained (experimental + synthetic + provenance), so any location works.
2. **Launch Jupyter** in any environment that has `jupyter` and `ipywidgets` (a viewing
   environment, not the render environment). Run `jupyter lab` and open
   `notebooks/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb`.
3. **Run the cells top to bottom.** The first code cell imports the viewer; the second is
   the only one you normally edit.
4. **Point it at your clip.** In the second code cell, set `CLIP_PATH` to the absolute path
   of your `.npz`. `NORM_MODE` there defaults to `full` (a shared full-range `[min, max]` window
   over both panels); optionally set it to `autoscale` (each displayed frame to its own min/max,
   per-frame) or `percentile` (a fixed whole-clip `[min, p99.99]` window). Run the cell; it
   prints the frame count, frame rate, selection, and the display window.
5. **Scrub and zoom.** Run the scrubber cell. Drag `frame` to step through the recording;
   use `center x`, `center y`, and `zoom` to zoom the same region-of-interest into both the
   experimental and synthetic panels at once.
6. **Play.** Run the playback cell for a real-time, side-by-side player, using the same
   per-frame color scaling as the scrubber. Set `PLAY_ZOOM` (and the center) to play a cropped
   region and check whether experimental and synthetic coincide locally. If a long clip builds
   a heavy player, raise `PLAY_EVERY` (2, 5, …) to subsample frames; it stays real-time.

To change the display, edit `NORM_MODE` and re-run from the second code cell down. To view a
different clip, change `CLIP_PATH` and re-run.

## What it shows — and what it does not

- It reads **imaging appearance**, not motion. The synthetic track is a fresh
  reaction-diffusion draw, so the max projections and any track-level feature will differ.
- The stored synthetic is clipped to the non-negative range (a physical camera cannot record
  negative counts); the clip's negative-excursion count (`n_under`) is printed on every run.
  See `REFERENCE_EMCCD_NOISE_MODEL.md` for the corrected noise model and what the clip removes.
- If the MAP for a parameter lies outside the training prior, the run warns that it
  **extrapolates** there, so a poor match may reflect an out-of-prior estimate rather than
  the calibration itself.
- The estimator was calibrated on the fixed 8-bit rescale of the recordings; the clip is
  shown at full 16-bit, so the sub-8-bit detail displayed here lies just outside the
  calibrated domain.

## Reference

Real recordings: MET single-particle-tracking data, BioImage Archive accession S-BSST712. The
imaging model and the read-noise units discrepancy are documented under DLI Imaging in
`PROJECT_CONTEXT.md`; the Detector calibration workflow and its deferred items are in
`DETECTOR_WORKFLOW.md`.
