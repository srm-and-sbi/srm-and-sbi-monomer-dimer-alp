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

- `aggregation_candidates.npz` (format 4): group vectors, quantiles, identities, methods, source contract, target
  binding, and the posterior-compatibility data (per-group median score, status and reference count; the
  rank-matrix blocks; the protocol `sgm_distance_rank_v2` with the archive's comparison scope, `construction`) of
  the primary rank space `all` (every coordinate) and of every additional rank coordinate space Experiment was
  asked for (`compatibility_<label>_*`, manifest section `compatibility.rank_spaces` with each space's coordinates
  and restricted scale).
- `aggregation_groups.csv`: each group's physical values, source draw identity, compatibility score, status,
  reference count and the keys on which the SGM itself lies outside the training prior; one further
  score/status/reference-count triple per additional rank space.
- `aggregation_comparison.csv`: representative-distribution summaries by grouping, set, statistic, and parameter.
- `aggregation_compatibility.csv`: every pairwise distance rank (candidate, reference, reference draws, rank,
  reference scope) of the primary space; `aggregation_compatibility_<label>.csv` the same for an additional space.
- `aggregation_manifest.json`: source identities and checksums, coverage, methods, the compatibility block map,
  and archive checksum.
- `figures/aggregation_*`: the figures embedded in the report, including, per condition and window set, the
  compatibility cell matrix and the window-position matrix with their score strips, the complete cell–window
  population matrix (one matrix over all cell–window groups) and the cell-by-window overview of its scores.
- `compatibility/<condition>/<set>/windows/index.md`: the atlas that enlarges the same-cell blocks of the
  cell–window population matrix, twelve cells per page, linked once from the report.

**Posterior compatibility.** The subsection "Posterior compatibility of SGM representatives" reports, for one SGM
per cell, one SGM per window position and one SGM per cell–window, how central each SGM is relative to the
posterior clouds of its peers: the distance rank R_ij (the share of reference j's retained draws strictly farther
from reference j's own SGM than the candidate SGM is, distances in estimator coordinates divided by the prior
widths; rows candidates, columns references, self comparison excluded by identity, generally asymmetric) and the
candidate's score, the ordinary median of its row over the other references. Experiment computes the comparison
scope `construction`: the references of a representative are ALL the other representatives of its own
aggregation in the same condition and window set (the other included cells, each its equal-window cloud mixture;
the other selected window positions, each its equal-cell cloud mixture; every other cell–window group, each one
window's own cloud, the other windows of the candidate's own cell included). The cell–window matrix is therefore
one population matrix (600 x 600 for 60 cells and ten windows), not a set of per-cell matrices; its small
diagonal blocks, enlarged in the atlas, are the same-cell comparisons and are a small part of each candidate's
references. A median score of zero means that a majority of the candidate's reference ranks are zero; some
references may still assign it a high rank. A zero score is a valid centrality outcome of a valid recording and
differs from an unavailable score (a population of one group, `insufficient_peers`); it is not a data-quality
classification. The archive reader determines the comparison scope from the archive, checks the complete
reference context of every block (one block of that scope per eligible context over exactly its peer population,
reference draw counts equal to the referenced groups' pooled draws, the protocol scale equal to the archive's)
and refuses an archive whose blocks were split, trimmed, rescaled or relabeled. Archives of formats 2 and 3 hold
the scope `within_cell` (each cell–window SGM against the other windows of its own cell; window positions
unscored) and are read under that meaning. Under `opening` the per-cell and the cell–window groups are the same
windows, so both are scored and agree, while the single window position has no peer. Experiment reports raw
median ranks only; the definitions and equations are in DETECTOR_WORKFLOW.md §7.2.

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

On HPC, `AGGREGATION_WINDOW_SETS` carries the same comma-separated grammar and `COMPATIBILITY_COORDINATES` the
rank coordinate spaces (`LABEL=key,key[;LABEL2=...]`). Both submitters export them via the environment and `ALL`,
following the same comma-safe rule as `KINDS`; they do not insert the values into Slurm's comma-delimited explicit
export list, and each launcher appends each value as one quoted argument. `NO_AGGREGATION=1` selects the legacy
report and `REFERENCE_NUISANCE_TAG` selects a displayed fixed reference. Workers and merge receive the same settings.

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

Remove `--dry-run` to emit the skeleton. It lists every available component as a commented table, with a
commented `member_weighting = "uniform"` line stating for each component whether `posterior_rank_median` is
available, evaluated separately for every rank coordinate space the archive holds (per space: compatibility scores
computed for all members and normalizable under the constructor's own rules with that space's coordinates;
unavailable with the reason, including a computed but all-zero normalizer; a space's availability never stands in
for another's; or not applicable to the grouping), and leaves `rationale` empty. The console prints the same availability. It cannot be built until the
selection is filled. An existing spec is refused, so use a separate tag for another construction. The generated
provenance and source checksum identify the archive.

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
| `[[component]].member_weighting` | `"uniform"` (default: every member receives alpha_k / n_k) or `"posterior_rank_median"` (consensus weighting from Experiment's compatibility scores; requires `statistic = "sgm"`, a grouping with peers (`per_cell`, `per_window_position`, `per_cell_window`), an explicit `rank_scope` and an archive that holds that grouping's scores under that scope) |
| `[[component]].rank_scope` | Required with `posterior_rank_median`, not allowed otherwise: `"construction"` (scores against all the other representatives of the grouping, condition and window set; normalized over the whole component) or `"within_cell"` (`per_cell_window` only: scores against the other windows of the same cell, equal cell mass). The scope is a property of the comparison population the archive recorded for that grouping's scores, and the field must name it; it is not tied to a format number as such (see "Comparison scope and archive format" below) |
| `[[component]].rank_coordinates` | Optional, with `posterior_rank_median` only: the parameter keys of the archive rank space that supplies the scores (default: every coordinate, the space `all`); the archive must hold a space over exactly those keys |
| `[override.<KEY>].value` and `.reason` | Optional explicit physical value and the recorded justification for applying it to every vector |

A single component with weight one is a valid choice. If component k has weight alpha_k and n_k members,
uniform weighting gives each member probability alpha_k / n_k. Uniform concatenation of differently sized
collections would give larger collections more mass solely because of their row counts; this is not the rule
used here.

Under `posterior_rank_median`, a member's probability is proportional to the median compatibility rank s of its
SGM, and the component names the comparison scope of those scores (`rank_scope`). The aggregation does not imply
the scope. Under `"construction"` the scores compare every representative with all the other representatives of
its grouping in the same condition and window set, and they are normalized over the complete component: p_i =
alpha_k s_i / sum of the component's scores. For one SGM per cell–window over 60 cells and ten windows each
representative has 599 references, the other nine windows of its own cell among them, and no cell keeps a
reserved mass: a cell's total is the sum of its representatives' probabilities and may be small or exactly zero.
Under `"within_cell"` (one SGM per cell–window only) the scores compare each representative with the other
selected windows of its own cell, and every included cell keeps the equal mass alpha_k / |C_k|: p = (alpha_k /
|C_k|) s_cw / sum of that cell's selected-window scores. The two are different selection targets with different
reference populations; the scope must be the one the archive computed the grouping's scores under, a mismatch is
refused, and one scope's scores are never renormalized into the other's. A zero score is exactly zero probability: the member stays in the record and the
exports, nothing is floored, clipped or removed, and the report counts the zero rows as sampling outcomes of the
selected policy, not as defective recordings. A component whose every score is zero (under `"within_cell"`: any
included cell whose every selected-window score is zero) is refused rather than dropped or redistributed; under
`"construction"` a single cell whose representatives all score zero is valid and carries zero total mass. An
unknown policy, a missing or mismatched scope, a missing or unavailable score (a population of one group), a
marginal-median statistic or a grouping without peers (the dataset representative, selectable with uniform
probability) are explicit errors with no fallback to uniform. A format-1 archive (written before the compatibility data) still
supplies a uniform selection; rank weighting needs the CPU regeneration of the Experiment report. The scores
describe the original SGM candidates; explicit overrides transform the stored rows afterwards and are shown. The
consensus-weighted distribution need not preserve the original proportions of acquisition batches; that is a
consequence to describe, not an automatic reason to reject a selection.

**Comparison scope and archive format.** What a specification names with `rank_scope` is the reference population
the archive's scores of that grouping were computed against, as the archive records it:

| Archive | `per_cell` scores | `per_window_position` scores | `per_cell_window` scores |
|---|---|---|---|
| Comparison scope `construction` (format 4) | `"construction"`: against all the other cells | `"construction"`: against all the other positions | `"construction"`: against all the other cell–window groups |
| Comparison scope `within_cell` (formats 2 and 3) | `"construction"`: against all the other cells | none (uniform only) | `"within_cell"`: against the other windows of the same cell |

One SGM per cell is therefore selected with `rank_scope = "construction"` from either kind of archive: its scores
already compare every cell with all the other cells in both, with the same reference population and the same
values. The construction-wide weighting of cell–window representatives and of window positions needs scores that
were computed against those complete populations, which only an archive of the `construction` scope holds; an
archive of the other scope is never relabeled or renormalized to supply them, and the build is refused with the
archive's format, protocol and scope named.

**Specifications written before `rank_scope` existed.** The field is required and has no default, so that the scope
is never implied. A built artifact is unaffected: it loads and samples exactly as before, whatever its age. A
rank-weighted specification file written earlier is refused until it names its scope. To rebuild or dry-run such a
selection, add the field to a COPY of the specification, under its `[[component]]` table, and leave the stored
specification as the historical record it is:

```toml
[[component]]
# ... source_experiment, condition, window_set, grouping = "per_cell_window", statistic, weight as stored ...
member_weighting = "posterior_rank_median"
rank_scope = "within_cell"          # the scope of a cell–window selection built from a format-2 or format-3 archive
rank_coordinates = ["mu_r", "sigma_r", "mu_pc", "sigma_pc"]
```

This is the migration of the adopted `RANK4` specifications (one SGM per cell and window, format-3 archives): with
that one line a copy reproduces the stored `RANK4` probabilities exactly. A per-cell selection written earlier takes
`rank_scope = "construction"`.

The scores come from one rank coordinate space of the archive. By default they are the ranks over every
coordinate (the space `all`); `rank_coordinates` names a list of parameter keys and selects the archive space over
exactly those keys (Experiment's `--compatibility-coordinates`), and a space the archive does not hold is refused
with the available spaces named. The relation to the overrides is recorded and reported: when the ranked
coordinates include an overridden coordinate, disagreement in that coordinate entered the member weights although
the final vectors no longer carry it (stated, not corrected); when the ranked coordinates are exactly the retained
ones, the weights rest on the coordinates the final vectors carry. The equal-member baseline does not depend on
the space.

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

The preview validates the selected archive, target, components, weights, member policies, and physical values,
and lists the vectors and output paths. For every component it states how its mass reaches its members: under
uniform weighting the probability each member receives; under rank weighting the positive and zero member counts,
the probability range and the effective number of members. A component of mass zero is valid under either policy:
it stays in the provenance with no positive member and an explicitly undefined effective count. It performs CPU
calculations but neither neural sampling nor an artifact write. Remove `--dry-run` for the reviewed build. This step also uses CPU only.

The output paths follow the active machine profile, under `Posit`:

- `<alias>_<timing>_<TAG>_Nuisance_DLI_Spec.toml`: the selection and rationale.
- `<alias>_<timing>_<TAG>_Nuisance_DLI.npz`: vectors, probabilities, parameter keys, and construction record.
- `<alias>_<timing>_<TAG>_Nuisance_DLI_Analysis/report.md`: the selection table (source archive and checksum,
  condition, window set, grouping, SGM statistic, component mass, member policy, members / positive / zero, the
  member probability range and the effective number of representatives 1/sum(p_i^2) with its normalization scope,
  a concentration measure and not a count of independent experiments; for a rank-weighted component the comparison
  scope, its reference population, the rank space and the normalization rule), consolidation, overrides, the exact
  marginal summary and the exact marginal histograms (`nuisance_marginals`, with the bin masses in
  `nuisance_marginal_histograms.csv`: probability mass per bin in physical units, computed from the stored vectors
  and their probabilities with deterministic bins shared by the selected distribution and the equal-member
  baseline, an overridden coordinate drawn as a point mass, nothing resampled; the sampled draws of the report
  serve its sampler check only), the stored sampling probabilities beside the equal-member baseline
  (`representatives_probabilities`; a cell-by-window grid of the stored row probabilities themselves with every
  cell's total beside it for a single non-duplicated selection of one SGM per cell and window, whose totals are
  also tabulated and exported as `representatives_cell_totals.csv`; labeled by source cell for a single
  non-duplicated per-cell selection, otherwise by stored row), the six-parameter
  CDF comparison of the selected probabilities against that baseline (`representatives_marginals`; the baseline uses
  the same components, component masses, duplicate consolidation and overrides with uniform members, so the
  comparison isolates the member weighting), the components view (`representatives_components`) and the joint
  projections (`representatives_joint`), beside `representatives_members.csv` (per member: identities, the
  primary-space score, status and references, the rank space, the rank scope and the score the policy used,
  contribution, stored row) and `representatives_rows.csv` (per stored row: probability,
  baseline probability, memberships by the recorded stored-row identity, the stored estimator coordinates
  `<key>_stored_log10` and the physical values `<key>_physical`). Every stored row keeps its memberships even when
  an override makes distinct rows identical; the report checks that the contributions add up to each row's
  probability. Experiment's strips and heat maps display raw median ranks; only this
  report displays adopted probabilities.

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
draws for Fab cells 26 and 4 and InlB cells 11 and 42 and on the complete cohorts (60 cells, 600 windows per
condition; review copies `COHORT60PLOTS` without and `COHORT60RANKS` with the compatibility data); its
illustrative builds are interface examples, not production selections. The acquisition period, construction or
mixture, weights, statistic, whether to enable consensus weighting, and the bleaching treatment remain open.

The corrected review reports carry the label `PILOT2CELLSR1`; earlier `PILOT2CELLS` reports are historical.
Their scientific candidate arrays agree. Both sets remain on disk unless a deletion is explicitly authorized.

Scientific and execution records:

- [DETECTOR_WORKFLOW.md §7](../../DETECTOR_WORKFLOW.md#7-nuisance-and-artifact-design): definitions, source decisions, and artifact design.
- [PROJECT_CONTEXT.md §3](../../PROJECT_CONTEXT.md#3-inference-pipeline-two-stage-architecture): the two-workflow scientific context.
- [VALIDATION.md](../../VALIDATION.md): pipeline checks, including the artifact smoke-test stand-in.
- [Implementation evidence](Evidence/Experiment_Aggregation_2026-11-05/COMPLETION_NOTE.md): measured costs, validation, review corrections, and output index.
- [Full-cohort collection evidence](Evidence/Experiment_Aggregation_Cohort_2026-11-05/COMPLETION_NOTE.md): the 116 cohort collections and the complete-cohort reports.
- [Compatibility and member-weighting evidence](Evidence/Experiment_Compatibility_2026-11-06/COMPLETION_NOTE.md): the regenerated `COHORT60RANKS` reports, the full-data oracle comparison, the illustrative rank-weighted build.
- [Rank coordinate spaces evidence](Evidence/Rank_Coordinate_Spaces_2026-11-09/COMPLETION_NOTE.md): the `COHORT60RANKSR3` regenerations with the four-coordinate space, the independent plain-counting oracle, the comparison of the two weightings.
- [Production Nuisance DLI of the biology workflow](Evidence/Nuisance_DLI_Production_2026-11-10/COMPLETION_NOTE.md): the
  canonical artifacts adopted on 2026-11-10 (construction-wide rank weights), with the scope implementation in
  [Comparison scope](Evidence/Comparison_Scope_2026-11-10/COMPLETION_NOTE.md).
- [Selection RANK4 (2026-11-09; replaced as the production selection, kept as a record)](Evidence/Nuisance_DLI_RANK4_2026-11-09/COMPLETION_NOTE.md): the production `empirical_representatives` artifacts of both conditions (decision record in DETECTOR_WORKFLOW.md sec. 7.6): specs, CLI dry-runs and builds, loader validation against the reviewed proposal, report copies.
