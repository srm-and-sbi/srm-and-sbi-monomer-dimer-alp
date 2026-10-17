# Experiment Population Composition

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_Population_Composition.py`. It reports the
monomer–dimer composition across the experimental MET recordings — the share of complexes that are
dimers, the share of receptors sitting in dimers, the totals and, under MET-INLB, the window-start
allocation with the counts it places — and reports beside it the error of that same readout measured
on held-out synthetic videos whose parameters are known.

The two halves belong in one document. An experimental estimate is only as readable as the measured
accuracy of the instrument that produced it, and here the instrument can be measured exactly: the
identical function of the identical estimator, applied where the answer is known.

## Why proportions and not counts

The estimator infers the stoichiometry of the model directly: the receptor-subunit total `N_total`
of the underlying population (a log row), the basal reference fraction `f` of dimers among its
complexes (a linear row on [0, 1]) and, under MET-INLB, the initial active-dimer allocation
`R_active` relative to its capacity (a linear row on [0, 1]; MET-FAB declares it 0). The absolute
total is the worst-identified coordinate the model has, and the reason is information-theoretic
rather than a defect of training. Counting few emitters in a diffraction-limited scene is a
square-root-of-n problem, so the relative precision of a count degrades as the count falls; and the
total trades off against the fraction inside the posterior, because the same number of detected
spots can be few dimers or many monomers.

The run recorded at the end of this note was made with the three-species model that preceded the
separated stoichiometry–mobility model, whose estimator inferred three species counts directly — A
(monomer), B (mobile dimer), C (immobile dimer). On its held-out synthetic set the effect showed up
directly:

| quantity | error |
|---|---|
| A (monomer) alone | 0.117 dex |
| B (mobile dimer) alone | 0.158 dex |
| C (immobile dimer) alone | 0.130 dex |
| **T = A + B + C** | **0.0117 dex** (r = 0.9995) |

The sum is recovered an order of magnitude better than any of its parts. That is only possible if the
parts' errors are strongly anti-correlated — the posterior moves abundance between species while
holding the total nearly fixed — and it means any quantity dividing one part by the total inherits
that cancellation. The composition is therefore a well-identified function of a badly-identified
vector, which is the whole reason this analysis exists. The report carries the same parts-versus-whole
comparison for the coordinates above (*What it reports*, below), so the claim is measured for every
estimator it is run on rather than carried over from that run.

## Why the fractions must be formed inside each draw

A function of correlated coordinates is not a function of their marginals. The starting fraction, the
realized counts and the totals combine the receptor total, the basal reference fraction and, under
MET-INLB, the allocation; building them from per-parameter medians asserts a combination the
posterior never drew, and it discards exactly the cancellation that makes the ratio identifiable.

Every derived quantity here is therefore computed **within** a single posterior draw, through the one
population helper the generator itself uses (`parameterization.realize_population`), and only then
averaged.
This is the same principle the Sample Geometric Median analysis applies to a single summary vector,
applied here to a derived quantity: a per-coordinate composite is not a configuration the system was
ever inferred to be in.

The consequence is a hard input requirement. The analysis needs the Experiment stage's raw per-window
posterior draws (`--dump-posterior-samples`); the stored marginal quantiles cannot substitute,
because a fraction of marginals is a *different quantity*, not a coarser version of the same one. The
script fails loudly and says which re-run is needed rather than silently reporting the wrong thing.

## The quantities

**Two populations.** The reference population is what the declared occupancy defines from the
underlying population of `N_total` receptor subunits with the basal reference fraction `f`:
`M_ref = N_total (1 − f) / (1 + f)` monomers and `D_ref = N_total f / (1 + f)` dimers. The starting
(window-start) population is what the generator places: under MET-INLB the allocation turns
`c = round(R_active · cap)` of the `cap = A_ref // 2` available probe-bound monomer pairs of the
reference into additional two-InlB dimers, so `M_start = M_ref − 2c` and `D_start = D_ref + c`; under
MET-FAB the allocation is declared 0 and the starting population is the reference, so one table
describes both (`PROJECT_CONTEXT.md` §2, *Modeling assumptions of the MET model*, item 9). Both are
reported, the starting one first under MET-INLB.

For either population, with `n_A` and `n_B` its monomers and dimers (`M_ref`, `D_ref` or `M_start`,
`D_start`) and `T = n_A + n_B`:

| symbol | definition | reads as |
|---|---|---|
| `f_A` | `n_A / T` | share of complexes that are monomers |
| `f_B` | `n_B / T` | share of complexes that are dimers: the basal reference fraction `f` itself for the reference population, the derived starting fraction `f_start` for the starting population |
| `f_R` | `2 n_B / (n_A + 2 n_B) = x_B` | share of individual **receptors** sitting in a dimer, `2 f_B / (1 + f_B)` |
| `T` | `n_A + n_B` | total complexes (a count, not a share) |
| `N_total` | `n_A + 2 n_B` | total receptors (a count; the inferred coordinate, which the allocation conserves) |

`f_B` and `f_R` differ because a dimer holds two receptors: a population half dimeric by complex is
more than half dimeric by receptor. Mobility (fast, slow, immobile) is a mode of either species, not
a species, so it does not enter the composition. `f_A + f_B = 1` for every draw by construction, and
the report checks it (`composition_closure`, the largest `|f_A + f_B − 1|` over every valid draw) as
the cheapest possible detection of a coordinate mix-up. Because `f_B = 1 − f_A`, their recovery errors
are equal in magnitude by identity; the two rows agreeing in the validation table is an internal
consistency check, not two independent results.

**The realization per draw.** Beside the shares, every draw is realized by the population helper
under the estimator's condition into the quantities the generator works with: `N_total`, `f`,
`R_active`, the capacity, the allocated pairs `c`, `M_ref`, `D_ref`, `M_start`, `D_start`, the
derived starting fraction `f_start` and the derived active share `c / D_start` (NaN where `D_start`
is 0), the probe-bound counts of the reference (`A_ref`, the dimers `D_sim` carrying at least one
probe, `B1`, `B2_ref`) and of the start (`A_start`, `B2_start`, the dimers and complexes placed), the
represented receptors `N_R`, the expected probe-bound receptors, and the expected fluorescent
complexes at frame 0 (`labeling.expected_fluorescent_complexes` under the condition's production
labeling). `f_start` and the active share stand beside `f` and `R_active`, never in their place, and
nothing reconstructs underlying monomers as `A_start / p`.

**Flagged draws.** The posterior sampler is unrestricted, so a draw can carry a fraction outside
[0, 1]. The one validity rule (`parameterization.population_valid`: `N_total` finite and positive,
both fractions finite and inside [0, 1], the allocation equal to its declared value where the
condition declares one) flags such a draw: it is counted (per condition: draws flagged, windows with
any and with every draw flagged, and per row), carries NaN in every derived quantity, and is never
clipped. A window's mean is taken over its remaining draws; a window with every draw flagged drops out
of its recording's mean. When no draw of the product is valid, the run still writes its report: it
opens with *no valid composition draws*, keeps the flagged-draw counts and the synthetic validation,
marks the closure and replicate checks UNAVAILABLE (neither a pass nor a failure), and leaves out every
summary table and figure built from the Experiment draws rather than fill them with values computed
from nothing; the stored arrays hold NaN.

**Units.** Shares are reported in percent and their errors in percentage points; the totals `T` and
`N_total` are counts whose errors are multiplicative and reported in dex; the realized integer
counts (the capacity, `c`, `A`, `B1`, `B2`, `N_R`) are reported as counts (a mean over draws need not
be an integer) with errors in counts, never in dex, because 0 is a legitimate value; the allocation
`R_active` is a fraction whose errors are in percentage points, with no dex band and no
multiplicative factor.

## The aggregation ladder, and the replicate unit

Four levels, each an unweighted mean over the level below:

```
draws   -> window       posterior mean of the fraction for that window
windows -> recording    the recording's composition over its analyzed span
cells   -> condition    the reported value; the standard error is taken HERE
```

**The replicate unit is the recording, never the window.** Ten windows of one cell are ten views of
one biological object, not ten independent measurements; treating them as independent would divide
the standard error by the square root of the window count and manufacture significance out of
pseudo-replication. Every interval and every test in the report is computed over recordings, and the
report states the count it used.

## What it reports

**One estimator, the recordings of both kinds.** The Experiment product holds one estimator, the
`--condition` of the run, applied to the recordings of every kind it was run on. Every derived
quantity is computed under the estimator's condition — its table, its occupancy and its declared
values — never under the recording's kind. Recordings of the other condition are therefore a
misspecification test, and the report labels them so beside every derived table and figure. The
comparison of kinds in one product is a comparison under one estimator's assumptions; it does not
replace a comparison of independently inferred populations, each under its own condition's
estimator, which this analysis does not perform.

**Experimental half**

- **Span-averaged composition** per condition, mean ± SEM over recordings, plus the totals: under
  MET-INLB one table for the starting population (`f_B = f_start`) and one for the reference
  population (`f_B = f`); under MET-FAB one table, the starting population being the reference.
- **First-window composition** — the earliest state the recordings resolve, for each population,
  reported separately because a span average over a changing population describes no instant of it.
- **Window-start realization per draw** — the realized quantities listed under *The quantities*,
  span-averaged, fractions in percent and counts as counts.
- **Flagged draws** — per condition, the draws and windows flagged by the validity rule, counted and
  left out of every realized quantity, never clipped.
- **Bootstrap check** — a percentile bootstrap resampling recordings, so the reported error bars can
  be seen not to depend on the normal-theory assumption behind the SEM (reference population).
- **Sensitivity** — the headline (reference population) recomputed under two independent challenges:
  restricting posterior draws to the trained prior box (the stoichiometry coordinates only — the
  receptor total, the basal reference fraction and, under MET-INLB, the allocation — then every
  coordinate), and replacing the arithmetic mean with the compositional (closed geometric) center,
  which is the natural center on the simplex. A draw outside a fraction's box is flagged in every
  variant. The table reports what each variant costs in discarded draws and in windows left with
  none. The spread across variants is the model-conditional bracket on the value.
- **Per-recording spread** of the monomer share (reference population) — whether the condition mean
  describes a typical cell or falls between two groups of them.
- **Within-recording time course** — per-window means, the median per-recording Spearman correlation
  against window index, and a first-versus-last signed-rank test paired within recording (reference
  population; under MET-INLB a figure of the starting fraction as well).
- **Condition contrast** at recording level, two-sided rank-sum, per quantity (reference population).

**Synthetic half**

- **Recovery of the same quantities** on the held-out evaluation set (reference population): mean
  absolute error, its 95th percentile, bias, and correlation with truth. Fractions in percentage
  points, the totals in dex. A flagged MAP estimate is excluded from every row and counted.
- **Parts versus whole** — the errors of the species counts `n_A = N_total (1 − x_B)` and
  `n_B = N_total x_B / 2` beside the error of their sum, the comparison the composition argument rests
  on.
- **Recovery in the dimer-rich region** — the same statistics restricted to videos whose *true* `f_B`
  exceeds a threshold (default 0.7), the corner the activated condition occupies. Selection is on
  truth, never on the estimate, so it is independent of the estimator's own error.
- **Starting-composition recovery** — the realized counts from the MAP against the truth, in counts
  (zero-safe) and, under MET-INLB, per true-capacity stratum ({0}, 1–9, 10–99, 100–999, ≥ 1000), with
  the allocation `R_active` (MAP) per stratum in percentage points. At capacity 0 the allocation has no
  effect on the window, so a prior-like posterior is the expected outcome there. Under MET-FAB the
  allocation is declared 0, `c` is 0 for every truth and every estimate, and no strata are formed.

Figures: `composition_conditions` (both conditions per quantity, with the synthetic error drawn on
each bar; the reference population, which is the starting population under MET-FAB),
`composition_conditions_start` (MET-INLB: the starting population), `composition_recovery` (true
versus inferred share on held-out videos), `composition_time_course` (`f_B` of the reference
population across the recording per condition), `composition_time_course_start` (MET-INLB: the
starting fraction `f_start` across the recording), `composition_per_cell` (every recording's monomer
share against the condition mean).

The derived arrays — per-window, per-recording, and per-condition — are written alongside as `.npz`,
so the tables can be re-tabulated or carried downstream without re-running.

## How to run

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_Population_Composition.py \
    --condition FAB|INLB --total-time-seconds 2.0 [--bootstrap 10000] [--stratum-threshold 0.7] [--dry-run]
```

`--condition` is the estimator's condition (`FAB` = MET-FAB, `INLB` = MET-INLB): it selects the
condition-specific estimator namespace, the estimator's parameter table (eleven parameters under
MET-FAB, twelve under MET-INLB), and the occupancy and declared values every derived composition is
realized under.

CPU only, seconds to run — it reads two completed outputs and neither loads the estimator nor renders
videos. It is a post-hoc analysis, never wired into the stage dispatcher.

It requires a biology Experiment output produced with `--dump-posterior-samples`. The MAP_Recovery
output is optional: without it the experimental half still runs, and the report states that the
validation column is unavailable rather than presenting experimental numbers with no measured error.

The tokens `ALP` and `BET` appear nowhere in the interface, the report, or the figures; they survive
only inside the Experiment output's stored `kinds` field and are translated at the boundary.

## Shared engine, and why this one is biology only

The analysis runs on the engine `srm_and_sbi_monomer_dimer_alp.population_composition_runner` over the
workflow-agnostic kernel `srm_and_sbi_monomer_dimer_alp.population_composition`. The kernel knows nothing
about receptors: it takes draws, the indices of the stoichiometry coordinates, a prior box, and the
conversion the runner binds to its parameter table (the runner realizes every draw through the
population helper, so the composition rules live in one place). It reuses the temporal-dynamics
kernel's grid builder, so "a (condition, recording, window) grid" means one thing across both
analyses and cannot drift between them.

Unlike the mirrored stage runners there is **no detector counterpart**, and the asymmetry is
scientific rather than incidental: the composition is a function of the inferred stoichiometry
coordinates (`N_total`, `f` and, under MET-INLB, `R_active`), and the detector workflow infers imaging
parameters and no stoichiometry at all — it treats the population implicitly, as part of what it
marginalizes. There is nothing there to compose, in the same way the detector's `Nuisance_DLI` pool
has no biology counterpart. The spec resolver checks for the stoichiometry parameters and fails loudly
for a workflow that lacks them, rather than composing six imaging coordinates into a well-formatted
meaningless number.

## What it found on the 2 s MET data

This run was made with the three-species model that preceded the separated stoichiometry–mobility
model: ten parameters, per-species initial counts of A (monomer), B (mobile dimer) and C (immobile
dimer), with `f_D` the share of dimers of either kind. Its readouts are kept as the record of that run
and are not re-derived; the split into mobile and immobile dimers has no counterpart in the separated
stoichiometry–mobility model, where mobility is a mode of either species.

Run on the 500-window Experiment output (250 windows per condition, 1,000 draws each) with the
10,000-video held-out recovery:

| condition | monomers | mobile dimers | immobile dimers | all dimers | receptors in dimers |
|---|---|---|---|---|---|
| MET-FAB (resting) | 68.6 ± 2.0 % | 12.9 ± 1.7 % | 18.4 ± 0.7 % | 31.4 ± 2.0 % | 46.4 ± 2.2 % |
| MET-INLB (activated) | 21.8 ± 3.7 % | 15.6 ± 1.7 % | 62.6 ± 2.7 % | 78.2 ± 3.7 % | 86.2 ± 2.5 % |

Against a synthetic in-model error of **2.8 pp** on `f_D` (2.6 pp with +0.3 pp bias in the dimer-rich
region the activated condition occupies), the 46.8 pp difference between conditions is roughly
seventeen times the estimator's own error. At recording level the contrast is `f_A` ×0.31,
`f_C` ×3.42, `f_D` ×2.57 (all p ≈ 2e-9, 25 versus 25 recordings), and total abundance ×1.88
(p = 0.014). The mobile-dimer subdivision does not separate (p = 0.19) — and it is also the least
identified quantity of the set (5.0 pp), which is the honest reading of that null.

Three qualifications the report carries with those numbers:

- **The absolute value has a bracket.** Restricting to draws inside the full ten-parameter prior box
  keeps only 3.4 % of MET-INLB's draws (123 of its 250 windows keep none) and gives `f_D` = 73.1 %;
  the compositional center gives 86.4 %. The arithmetic mean, 78.2 %, is the conservative headline,
  and the honest model-conditional bracket is **73–86 %**. Under every variant the contrast against
  MET-FAB (29–36 %) survives — the *comparison* is far more robust than the absolute level.
- **The composition changes within the recording.** MET-INLB's dimer share falls from 87.0 % in the
  first window to 70.2 % in the last (median per-recording Spearman −0.71; first-versus-last
  p = 0.0015), while MET-FAB is flat over the same span (32.5 % to 31.7 %, p = 0.52). A condition that
  moves while its control does not cannot be explained by anything shared between the two
  acquisitions. The composition is therefore a trajectory, and the first window is the initial
  post-activation state rather than a noisier version of the average.
- **The condition mean is not a typical cell.** The MET-INLB per-recording monomer share spans 1.2 %
  to 52.8 %, with 9 of 25 recordings below 10 % and 6 above 40 %. Downstream use should carry the
  per-recording values, which are in the saved arrays.

## Reading the result against published measurements

Two published measurements of this receptor system are direction checks, and neither is
unit-compatible with a complex census, so magnitudes are not directly comparable:

- Harwardt et al., *FEBS Open Bio* **7**(9):1422–1440 (2017), doi:10.1002/2211-5463.12285 — the study
  whose recordings these are (BioStudies accession S-BSST712). It classifies *trajectories* by
  motion, reporting immobile/confined/free occurrences of 12 ± 1 % / 31 ± 3 % / 57 ± 3 % for
  Fab-bound MET and 23 ± 1 % / 17 ± 2 % / 60 ± 1 % for InlB321-bound MET (mean ± SEM over N = 60
  cells, at 23 °C). It reports **no** receptor density and **no** dimer fractions. Activation roughly
  doubling the immobile share is the same direction this analysis finds; the magnitudes belong to
  different estimands (a detection-weighted trajectory classification versus a model-based complex
  census) and should not be equated.
- Dietz et al., *BMC Biophysics* **6**:6 (2013), doi:10.1186/2046-1682-6-6 — single-molecule
  photobleaching in chemically fixed cells with sub-stoichiometric labeling, giving 18 % dimeric
  spots unstimulated rising to 29 % after InlB321. The publication itself states its dimer fraction
  "is clearly underestimated" for that reason, so these are **protocol lower bounds**: they confirm
  the direction and bound the value from below, and a lower bound of 29 % neither corroborates nor
  contradicts a live-cell estimate near 78 %.

## What this analysis does not establish

- **It is not a molecular census.** The composition describes the monomer–dimer model as fitted to
  these recordings, conditional on the declared occupancies and, under MET-INLB, on the declared
  starting-state approximation of the allocation. Where the model has no state for something slow,
  bright, and persistent, one of its states absorbs it.
- **It does not validate the posterior's interval.** The synthetic half measures point-estimate
  error. The coverage of the inferred coordinates themselves (`f` and, under MET-INLB, `R_active`) is
  reported by the Evaluation stage and the `Posterior_Calibration` diagnostic. Whether a credible
  interval on a derived quantity (`f_start`, `f_R`, the counts) covers truth at its nominal rate needs
  joint posterior draws on the held-out synthetic set, which a recovery artifact storing per-parameter
  marginal quantiles does not carry. The kernel's composition function takes any draws array, so that
  check becomes available as soon as such draws exist.
- **It does not compare independently inferred populations.** Recordings of the other condition are
  estimated under this run's estimator, a misspecification test; a comparison of each condition's
  recordings under its own condition's estimator is a separate analysis.
- **It does not place the recordings inside the trained domain.** That is the embedding-space distance
  analysis's question, and its answer conditions everything here.
- **It does not attribute the within-recording change** to a mechanism. It establishes that one
  condition changes and its control does not.
