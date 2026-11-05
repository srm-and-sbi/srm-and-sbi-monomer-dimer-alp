# Nuisance DLI construction — usage and interpretation

The detector Experiment stage reports collections of imaging-parameter representatives. This analysis
selects a distribution from those collections and builds the `Nuisance_DLI` that the biology DLI stage
samples. Candidate calculation belongs in the **existing Experiment report**; selection and construction
are a separate, user-driven CPU analysis.

The scientific definitions, full options table, and weighting equations are in
[DETECTOR_WORKFLOW.md §7.2–§7.3](../../DETECTOR_WORKFLOW.md#72-the-nuisance_dli-artifact).
The current creation route is `empirical_representatives`. It reads the Experiment candidate archive and
performs no neural sampling. The older pool-based routes remain available for compatibility and are
summarized at the end of this note.

## 1. What Experiment provides

Each recording contains 1,000 frames at 50 FPS (20 s). The adopted detector receives ten non-overlapping
2 s inputs, each with 10,000 joint summary draws in this setup. Aggregation does not change the neural
input length. For each named window set, Experiment calculates:

| Grouping ID | Representative population | Count with 60 cells and all ten windows |
|---|---|---:|
| `per_cell` | One SGM of the pooled window draws per cell | 60 |
| `per_window_position` | One SGM of the pooled cell draws per position | 10 |
| `global` | One SGM of the pooled draws from all selected cell-window pairs | 1 |
| `per_cell_window` | The SGM of each selected cell-window posterior | 600 |
| `reference` (SEL, comparison only) | The historical coordinate-wise median of window posterior medians followed by documented parameter-specific selections | 1 |

Input clouds are pooled with equal cell weights and equal selected-window weights within cells. The
SGM is a complete retained draw. Small groups can use an exact sample medoid; pooled groups above
10,000 rows use the logged `weiszfeld_snap` approximation. Distances use estimator coordinates divided
by prior widths; displayed values use physical units. A single point retains its co-occurring coordinates,
not the entire posterior's correlations or uncertainty.

Every group also stores the vector of marginal medians (`statistic = "median"`) and pooled marginal
quantiles. The median vector is a coordinate-wise composite, not necessarily a sampled vector. An aggregate
MAP and an SGM of SGMs are not constructions of this route. Raw posterior draws are retained as calculation
inputs, rather than offered as the nuisance distribution in this strategy.

The standard sets are `all` and `opening`. `all` contains every position implied by the validated geometry;
`opening` is chunk 0. Additional sets use zero-based positions, for example `later=5-9` or `selected=1+3+5`.
Every included cell must contain every requested position. Missing positions are refused, rather than
silently dropped from `all`. Fab and InlB remain separate.

The main Experiment report contains **Experimental aggregation and nuisance candidates**, with coverage,
construction tables, physical-unit summaries, marginal CDFs, joint projections, temporal comparisons, and
matched window-selection plots. Its visible language is the project's own: each construction is described as
what it yields ("One SGM vector per cell", "One SGM vector per window position", "One SGM vector for the
included dataset", "One SGM vector per cell–window pair"; the reference as "Existing fixed nuisance vector:
SEL"), with the samples pooled before one representative is selected and, in legends, the actual number of
vectors; a component selected with the marginal-median statistic is labeled "marginal-median vector per …",
never an SGM. The report opens its data with a **Data scope** table (the original Experiment summaries and the
retained-cloud aggregation, stated apart: cells, windows, draws, "complete cohort" or "pilot subset: cells
[...]"), and every aggregation figure carries a subtitle with the covered cells and windows, the windows
included ("all ten 2 s windows", "opening 0–2 s window only" with the number of selected windows, or a named
set with its positions), the draws per window and the representative. The section states the two independent
choices once: which windows are included, and which samples are pooled; `opening` makes per-cell equal
per-cell-window and per-position equal global, and the opening figures name these equivalences. It links
these supporting files in its own output directory:

- `aggregation_candidates.npz`: group vectors, quantiles, identities, methods, source contract, and target binding.
- `aggregation_groups.csv`: each group's physical values and source draw identity.
- `aggregation_comparison.csv`: representative-distribution summaries by grouping, set, statistic, and parameter.
- `aggregation_manifest.json`: source identities and checksums, coverage, methods, and archive checksum.
- `figures/aggregation_*`: the figures embedded in the report.

The source-uncertainty table names the first requested multi-position set it summarizes. The archive retains
quantiles for every other group. Ranges over representatives describe empirical dispersion, not posterior
credible intervals. Representative-distribution summaries use the inverse empirical CDF; raw posterior
quantiles retain their interpolated convention. The selection console and final mixture report use the
same representative-summary definition as Experiment.

## 2. Run or regenerate the Experiment report

Aggregation is enabled by default, with `--aggregation-window-sets all,opening`, retaining the same clouds
already drawn for per-window summaries. It runs after the complete single-process product or validated shard
merge. `--no-aggregation` selects the legacy reporting mode; `--dump-posterior-samples` can still retain clouds
in that mode. The older `--aggregation pooled|cell-median` controls the existing point-summary display and
is a different option.

The detector reference is selected by `--reference-nuisance-tag` (default `SEL`); an absent reference is
reported. No reference artifact is rebuilt or automatically included in a mixture.

To regenerate a saved report on CPU, use an unused output-only review label. This example loads Fab's
adopted product and uses its own stored clouds if present:

```bash
MACHINE_PROFILE=mars_pc python Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment.py \
    --condition FAB --total-time-seconds 2.0 --artifact-tag C128V42A \
    --regenerate-report --review-label AGGREGATIONREVIEW
```

When the original product did not retain clouds, add
`--cloud-sources /path/to/first_collection.npz,/path/to/second_collection.npz`.
These must be compatible retained window-draw collections. Their identities must be windows of the source
product; checkpoint, parameter order, scale, coordinates, geometry, draw count, and sampling mode must match.
Recorded preprocessing descriptions are compared between collections even when the original product lacks
that field. Missing preprocessing metadata are disclosed. Collection labels and seeds can differ and remain
in provenance. A partial review explicitly states its cell and window coverage and identifies fresh clouds
as a separate sampling attempt.

The report is written beside the source product under the review label; an existing destination is refused.
The label does not change the estimator tag. InlB uses `--condition INLB --artifact-tag C128V42AP3LR32`.
The biology Experiment entry point shares the reporting implementation, but only detector candidates feed
this imaging artifact.

On HPC, `AGGREGATION_WINDOW_SETS` carries the same comma-separated grammar. Both submitters export it via
the environment and `ALL`, following the same comma-safe rule as `KINDS`; they do not insert the value into
Slurm's comma-delimited explicit export list. `NO_AGGREGATION=1` selects the legacy report and
`REFERENCE_NUISANCE_TAG` selects a displayed fixed reference. Workers and merge receive the same settings.

## 3. Emit a selection specification

Use the archive whose report was reviewed. It must record the source Experiment's target condition,
timing label, and window geometry; old archives without that binding require CPU regeneration. Template
emission, build dry-run, and actual build refuse a mismatched target. A cross-condition Experiment application
can be inspected, but its recording-condition component cannot become another condition's native nuisance.

The following commands illustrate the interface. Replace the archive path with the reviewed source.
`STUDY` is an illustrative nuisance tag, not a selected scientific treatment.

```bash
MACHINE_PROFILE=mars_pc python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py \
    --condition FAB --total-time-seconds 2.0 --nuisance-tag STUDY \
    --emit-representatives-template --candidate-archive /path/to/aggregation_candidates.npz --dry-run
```

Remove `--dry-run` to emit the skeleton. It lists every available component as a commented table and leaves
`rationale` empty. It cannot be built until the selection is filled. An existing spec is refused, so use a
separate tag for another construction. The generated provenance and source checksum identify the archive.

## 4. Fill the scientific selection

The spec uses the following fields:

| Field | Meaning |
|---|---|
| `[block].posterior_sample_pool_choice` | `"empirical_representatives"` |
| `[block].candidate_archive` | Absolute path, or path relative to the spec |
| `[block].candidate_archive_sha256` | Checksum of the reviewed archive |
| `[block].rationale` | Why these acquisition windows, representatives, statistics, and weights were chosen |
| `[[component]].source_experiment` | Product label recorded in the archive |
| `[[component]].condition` | Native target condition, `FAB` or `INLB` |
| `[[component]].window_set` | A named set present in the archive |
| `[[component]].grouping` | One of the four grouping IDs above |
| `[[component]].statistic` | `"sgm"` by default, or explicitly `"median"` |
| `[[component]].weight` | Non-negative component mass; all component weights sum to one |
| `[override.<KEY>].value` and `.reason` | Optional explicit physical value and the recorded justification for applying it to every vector |

A single component with weight one is a valid choice. If component k has weight alpha_k and n_k members,
each member receives probability alpha_k / n_k. Uniform concatenation of differently sized collections
would give larger collections more mass solely because of their row counts; this is not the rule used here.

Components reuse observations. Combining them changes the probability assigned to representative states,
not the amount of independent experimental evidence. Adding a global representative places a point mass at
the overall center. No mixture is automatically better or chosen by the template. Exact duplicate vectors
are consolidated by adding probabilities, with memberships retained; displayed rounded values are not used
for deduplication.

Weights must be finite, non-negative, have positive total mass, and sum to one within 1e-6. Larger discrepancies
are refused. Accepted roundoff residuals are removed by dividing probabilities by their sum before storage
and sampling. The manifest records this numerical convention separately from the scientific component weights.

An override is an explicit transformation and is shown in the final report, including the replaced range.
It does not create another joint posterior draw. Physical domains are validated: values must be finite and
positive for the log-ranged imaging parameters, and bleaching probability must be strictly between zero and
one. A valid value outside the detector training range remains a labeled extrapolation, rather than being
clipped. A broad bleaching posterior or a pooled center near 0.1 does not automatically replace the separately
anchored 0.05 reference. That treatment remains a recorded scientific choice.

The selected representative distribution does not carry the full within-window posterior uncertainty. That
uncertainty remains available for inspection in Experiment. No pairwise correlation imposes a separate
sigma_pc–lambda selection rule.

## 5. Dry-run and build

After filling the spec, use the same condition, duration, and nuisance tag:

```bash
MACHINE_PROFILE=mars_pc python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py \
    --condition FAB --total-time-seconds 2.0 --nuisance-tag STUDY --build --dry-run
```

The preview validates the selected archive, target, components, weights, and physical values, and lists the
vectors and output paths. It performs CPU calculations but neither neural sampling nor an artifact write.
Remove `--dry-run` for the reviewed build. This step also uses CPU only.

The output paths follow the active machine profile, under `Posit`:

- `<alias>_<timing>_<TAG>_Nuisance_DLI_Spec.toml`: the selection and rationale.
- `<alias>_<timing>_<TAG>_Nuisance_DLI.npz`: vectors, probabilities, parameter keys, and construction record.
- `<alias>_<timing>_<TAG>_Nuisance_DLI_Analysis/report.md`: components, normalization, memberships, overrides,
  weighted marginal summaries, and final marginal/joint comparisons with the components and SEL.

The artifact stores the six imaging coordinates in log10. `pool_mode` records `unrestricted` when positive
probability mass lies outside the detector training box; support is not clipped. The NumPy-only sampler
draws complete rows at their stored probabilities. The camera remains a separate SCOPE draw.

The tag identifies the nuisance artifact, not the source estimator. The biology DLI stage selects it with
`--nuisance-tag` (dispatcher `NUISANCE_TAG`) and records its identity in each `Nuisance_DLI_Theta_Set`.
The posterior-predictive renderer and other consumers use the same identity mechanism. Consumers load an
existing artifact through `require_nuisance_dli`; they do not construct one implicitly.

## 6. Compatibility with historical constructions

These routes remain implemented for historical reproduction, fixed scenarios, and tests. They are not
alternative defaults of the representative-based procedure above. Older unweighted artifacts keep uniform
row sampling, and an old reader refuses the unfamiliar weighted construction rather than ignoring probabilities.

| Stored construction | Meaning and compatibility |
|---|---|
| `selection_user` | One fixed vector with physical values, sources, and limitations in `[selection.<KEY>]`; the existing SEL references use this route. `--emit-selection-user` emits its skeleton on CPU. Values outside the training prior require its explicit acknowledgment fields |
| `sgm_percentiles` | Signed-distance selection from stored window MAPs (`selection_source = "experiment"`) or window SGMs (`"window-sgm"`); a different calculation from pooling full window posterior clouds |
| `map_estimate_pool` | Uniform sampling of the stored per-window MAP candidates |
| `raw` | Uniform sampling of a retained posterior-draw pool, including within-window uncertainty; not the selected representation strategy |
| `gaussian` | A full-covariance Gaussian fitted to a posterior-draw pool |
| `box` | Independent coordinate-wise ranges derived from pool quantiles |
| `box_user` | Independent user-declared coordinate ranges; used as a fast stand-in for smoke tests, not a calibrated empirical distribution |

The older `--emit-template`, `--repool`, and pool-cache machinery belongs to those historical pool-based
routes. Drawing a missing pool can require the trained estimator, recordings, and GPU; these are not inputs
to `empirical_representatives`. `--migrate-pool-labels` adds labels to an old posterior-sample cache only
after its alignment check, keeping a backup of the original. Per-coordinate box constructions do not retain
joint parameter relationships. The six imaging coordinates are distinct from the five camera parameters;
the current detector does not infer camera gain or conversion as part of this six-vector artifact.

## 7. Project state and sources

The Fab and InlB SEL artifacts are built fixed references. The empirical construction was validated on saved
draws for Fab cells 26 and 4 and InlB cells 11 and 42; its illustrative build is an interface example, not a
production selection. The acquisition period, construction or mixture, weights, statistic, and bleaching
treatment remain open. The InlB external catalog cross-check and full-cohort draw collection also remain pending.

The corrected review reports carry the label `PILOT2CELLSR1`; earlier `PILOT2CELLS` reports are historical.
Their scientific candidate arrays agree. Both sets remain on disk unless a deletion is explicitly authorized.

Scientific and execution records:

- [DETECTOR_WORKFLOW.md §7](../../DETECTOR_WORKFLOW.md#7-nuisance-and-artifact-design): definitions, source decisions, and artifact design.
- [PROJECT_CONTEXT.md §3](../../PROJECT_CONTEXT.md#3-inference-pipeline-two-stage-architecture): the two-workflow scientific context.
- [VALIDATION.md](../../VALIDATION.md): pipeline checks, including the artifact smoke-test stand-in.
- [Implementation evidence](Evidence/Experiment_Aggregation_2026-11-05/COMPLETION_NOTE.md): measured costs, validation, review corrections, and output index.
