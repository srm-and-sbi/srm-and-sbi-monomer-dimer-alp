## Project Context — srm-and-sbi-monomer-dimer-alp

This document is self-contained. It describes the scientific context for the
model family and its reaction-diffusion-parameter inference: the research
question, the molecular system, the two-stage inference architecture, the data
and computational flow, the inference network, the validation methodology, and
the design rationale that shapes the implementation.

**Repository status.** In development (0.1.11). The codebase began as a copy of
the tracked tree of `srm-and-sbi/srm-and-sbi-dimer-alp` at its frozen release
`v0.4.23` and implements the MONOMER_DIMER model family on top of it. Landed:
the DOL-explicit observation layer (the measured degree of labeling as a static
per-subunit dye draw carried through the reactions, emitters that are dyes, the
declared per-condition probe occupancy), the condition axis (MET-FAB and MET-INLB
as two frozen configurations of one codebase, entering at the RDS stage through
the condition's declared association setting and at the DLI stage through its
labeling law, and carried by a condition slot in every product name), a detector
calibration that marginalizes the full reactive biology prior, and the separated
stoichiometry–mobility model of §2 (two molecular species × three mobility
modes; the reaction channels generated from the model blocks and the condition's
association setting — seventeen under MET-INLB, eleven under MET-FAB; eleven
learnable parameters, identical for both conditions, led by the conserved
receptor total `N_R` and the initial dimer-to-monomer ratio `r = n_B / n_A`, from
which the receptor fraction `x_B` is derived; one shared parameter-conversion
rule), and the decided prior range of every learnable row together with the
declared per-condition probe occupancies (2026-09-14; §2, *How the prior ranges
and the declared inputs are set*; the occupancies are provisional until the
collaborators answer the questions sent on 2026-09-11). Pending: the
`Nuisance_DLI` imaging recalibration per condition under the DOL-explicit
observation model, and the first per-condition trajectory tier under the decided
ranges. The
scientific sections below describe the implemented state.

**Condition tokens.** The experimental conditions are named `FAB` (MET-FAB, the
Fab-labeled monomer control) and `INLB` (MET-INLB, the InlB-labeled dimer
condition) in every filename, schema field, and CLI argument; display surfaces
prepend the receptor. The repo-iteration suffixes (`alp`, `bet`, ...) are a
separate namespace and never name a condition. Experimental recordings enter
this repository's data bank under the `FAB`/`INLB` names: all sixty 20 s
recordings per condition of BioStudies S-BSST712, indexed `Cell_0` to `Cell_59`
in the archive's coverslip-and-cell order, with the mapping to the archive
members in `Catalog_Note_Experiment.tsv` beside them.

---

## §1. Research Question

**Objective:** Infer posterior distributions `p(θ | x)` of reaction-diffusion
(RDS) parameters from time-resolved fluorescence-microscopy videos of
membrane-receptor dimerization.

**Why posterior distributions, not point estimates?**
Molecular-dynamics parameters are often partially confounded (for example,
diffusion coefficient and dwell time co-determine observed intensities).
Uncertainty quantification is part of the scientific result. Downstream
applications (drug-sensitivity analysis, population heterogeneity) require the
full posterior, not a single best-fit value. The inference target is therefore
a full conditional density `p(θ | x)` over the parameter vector, and the MAP
point estimate and the posterior credible interval are always reported together
because they answer different questions — a sharp posterior at the wrong
location and a broad posterior at the right one are distinguishable only when
both are shown.

**Why simulation-based inference (SBI)?**
The forward model is complex: a ReaDDy reaction-diffusion simulation feeds an
optical-imaging step (diffraction-limited, photon noise) to produce a video. The
likelihood is intractable. SBI sidesteps explicit likelihood evaluation by
learning a neural density estimator (neural posterior estimation with a masked
autoregressive flow) from simulated (RDS, video) pairs.

---

## §2. System: MET-Receptor Dimerization (the separated stoichiometry–mobility model)

**Molecular system:** A simplified, receptor-agnostic model of reversible
receptor dimerization on the plasma membrane, in which the *stoichiometric*
state of a receptor (monomer or dimer) and its *mobility* (fast, slow, or
immobile) are declared as two separate layers that interact only at named
points. The pipeline is applied to single-particle-tracking microscopy of the
**MET receptor** (c-Met / hepatocyte growth factor receptor); the Experiment
stage consumes the real recordings under
`Experiment/SPT_Data_MET_FAB_INLB_S-BSST712` (BioStudies accession S-BSST712).

**Two molecular species (the stoichiometry layer):**
- **A (monomer):** a single receptor subunit.
- **B (dimer):** two receptor subunits bound.

The receptor-subunit total `N_R = n_A + 2 n_B` is conserved within a recording:
no receptor synthesis, degradation, or internalization occurs during the
recorded window (a deferred biological extension). There is no immobile-dimer
species; immobility is a mobility mode available to monomers and dimers alike.

**Three mobility modes (the mobility layer):** **f** (fast), **s** (slow), and
**i** (immobile). A mode is a value on the diffusion-coefficient axis — Brownian
motion within the mode, with no mechanism of its own (a slow or immobile mode
represents reduced motion, not the membrane structure that may cause it). The
modes are shared by both species.

**Six particle types.** The reaction-diffusion stage simulates *particle types*
= molecular species × mobility mode: `A_f`, `A_s`, `A_i`, `B_f`, `B_s`, `B_i`.
They are derived by `PARAMETERS.simulation.rds` from two declarative model
blocks in `parameterization.py` — `StoichiometryBlock` (the species, their
subunit counts, the association and dissociation channels and their parameter
keys) and `MobilityBlock` (the modes, their diffusion-ratio keys, the sequential
switching chain and its rate keys, the inheritance rule) — together with each
type's subunit count and the maps type → species / mode. Nothing downstream
lists the types by hand.

**Diffusion.** The diffusion coefficient of particle type `(X, m)` is

    D[X, m] = D_A × (R_B if X = B else 1) × (1, R_s, R_i)[m]

with `D_A` the monomer scale coefficient (`D[A, f]`), `R_B` the dimer factor
within a mode (`0 < R_B ≤ 1`; at `R_B = 1` dimerization adds no slowdown and
population-level differences come from mode occupancy and inheritance alone),
and `R_s`, `R_i` the slow and immobile mode factors. The ordering
`R_i < R_s ≤ 1` is enforced by the *disjoint declared ranges* of the two keys,
checked at import (`_validate_model_blocks`), so it holds for every draw; `R_i`
is a practical resolution floor, not zero.

**Up to seventeen reaction channels**, generated from the two blocks and the
run's condition by `simulation_rds_support.reaction_channels(theta, condition)`
and registered verbatim by `build_system(theta, condition)` — never written by
hand. MET-INLB has all seventeen; MET-FAB has eleven, because its association
ratio is zero and no association channel is generated (*Association setting per
condition*, below):
- **Association (six fusions; MET-INLB only):** `A_m + A_m' → B_slower(m, m')`,
  one channel per unordered pair of monomer modes, all at the same microscopic
  rate `λ_on`. The
  dimer inherits the *slower parent's* mode (the declared working hypothesis;
  alternatives are comparators, not built here).
- **Dissociation (three fissions):** `B_m → A_m + A_m` at `κ_OFF`, one per dimer
  mode; the mode is conserved (both daughters keep the dimer's mode).
  Dissociation is not governed by contact.
- **Switching (eight conversions):** `X_f ↔ X_s ↔ X_i` for `X ∈ {A, B}` — the
  four rates `k_fs`, `k_sf`, `k_si`, `k_is` are shared by both species (the
  working hypothesis that switching is a membrane-environment process
  independent of stoichiometric state). The chain is sequential: there is no
  direct `f ↔ i` channel.

**Association normalization.** The microscopic association rate is
`λ_on = R_ON × λ_ref`, with `R_ON` the association ratio of the run's condition
(a declared constant, next paragraph), `λ_ref = 6 D_A / r²` (1/s), and `r` the
reaction distance = one particle diameter (10 nm,
`SimulationStem.particle_diameter_nm`). `λ_ref` equals the Smoluchowski
encounter rate of two monomers, `4π (2 D_A) r`, divided by the reaction volume
`(4/3) π r³`. It is a *compatibility normalization* that keeps `R_ON`
dimensionless — a declared reference with units of inverse time that depends on
`D_A` and on `r` — and explicitly **not** a physical upper bound on association
(the diffusion-limited regime is the large-intensity limit of the spatial rule),
so a ratio above one would be admissible in principle. Association requires an
encounter within `r` followed by the reaction.

**Association setting per condition.** The association ratio `R_ON` is not
inferred in either condition. It is a declared per-condition constant of the
generative model (`parameterization.ConditionSetting`, registered in
`CONDITION_SETTINGS` and read through `SimulationRDS.association_ratio_of`): the
ratio was not recoverable in the earlier workflow, there are no grounds to make
its estimation a requirement of the model, and a free ratio let the initial
composition, the association, and the unbinding trade off at a constant dimer
fraction. **MET-FAB: `R_ON = 0`** — a structural setting that generates no
association channel at all, never a small positive stand-in for a logarithmic
scale (a ratio between 0 and 10⁻³ is refused at import as a disguised zero). No
activating ligand is present, which removes induced association; basal
association is what is omitted, as a working approximation of an unresolved
within-recording process. MET-FAB is therefore modeled as a population of
pre-existing monomers and dimers that may dissociate during observation, and its
composition readout is the dimers *present at the recording start*, not dimers
formed. **MET-INLB: `R_ON = 1`**, so `λ_on = λ_ref = 6 D_A / r²` for every
association channel — a diffusion-scaled reference convention, not a measured
association rate or a verified diffusion-limited regime; the MET-INLB
composition and unbinding estimates are conditional on this choice. Unbinding
(`κ_OFF`) stays inferred in both conditions: under MET-FAB the expected dimer
count decays as `n_B(0) · e^(−κ_OFF t)`, and the lower bound of its range reaches
rates at which pre-existing dimers persist over a 20 s recording (0.001 per s
loses 2 % of the dimers in 20 s; decided 2026-09-14, *How the prior ranges and
the declared inputs are set*, below). Because the
two settings produce different trajectories, each condition has its own RDS
trajectory tier (§4), shared by both workflows; the eleven learnable parameters
and the estimator layout are the same for both conditions.

**Fission placement.** The two daughters of a dissociation are placed at
`SimulationStem.fission_product_distance_nm` = 2 × `r` = 20 nm, *outside* the
fusion radius. Declared convention (2026-09-09): at the 2 ms sub-step the
per-step reaction probability of an eligible pair, `1 − exp(−λ_on δt)`, is ≈ 1
for every `D_A` in range under MET-INLB (`λ_on δt` ≈ 7–67), so daughters
placed *at* `r`
would re-fuse at the next step unless they diffused apart within one step —
≈ 3% for fast daughters but ≈ 50% per step for immobile ones — which would make
the effective unbinding rate mode dependent (immobile dimers effectively never
dissociating) and couple the mobility layer to the stoichiometry layer through
a numerical artifact. Starting the daughters outside the radius removes that
deterministic rebinding; diffusive re-encounter remains possible, largest for
slow daughters. The reaction distance and the placement factor are both
declared conventions, not measurements. The 2 ms sub-step itself is kept as an
efficiency choice: at this resolution association is contact-detection
limited ("react when a pair is detected within `r`"), encounters that begin
and end between two sub-steps are not seen, and the MET-INLB association ratio
is not lowered to satisfy a numerical criterion (that would change the
biological assumption without recovering missed encounters).

**Boundary behavior.** The simulation box is open laterally (`x`, `y`, the
observation plane) and periodic axially (`z`, the thin membrane normal), with
no confining potential. A receptor that diffuses beyond the imaged field stays
simulated — it keeps reacting and switching and may return — and is simply not
rendered while outside. The conserved total `N_R` therefore refers to the
*simulated patch*; the in-field count is a distinct, time-dependent quantity.

**Parameters to infer (θ) — the eleven learnables, identical for both
conditions.** The table below reproduces the ranged rows of `parameterization.py`
(groups `stoichiometry` and `mobility`); the association ratio is not among them
— it is the condition's declared constant (above) — and both conditions share
this table and the estimator layout. Every range is a **decided prior**
(2026-09-14): a box-uniform prior in the estimator coordinate that covers the
plausible support of the quantity with a margin and never encodes the answer an
experiment is expected to give; the source and the reasoning of every row follow
the table (*How the prior ranges and the declared inputs are set*). The *scale*
column is the row's `LOG_FLAG`: a log row's estimator coordinate is `log10` of
the physical value (a log-uniform prior on the physical value). Every row of the
decided table is a log row; the per-row rule stays the contract, so a linear row
(whose coordinate *is* the value, a uniform prior) can be declared without
touching any consumer.

| key | symbol | meaning | scale | prior range (estimator coordinate, log10) | physical |
|---|---|---|---|---|---|
| `count_total` | `N_R` | conserved receptor-subunit total of the simulated patch, `n_A + 2 n_B`; conditional on the declared occupancies | log10 | [2.5, 3.5] | 316–3162 subunits |
| `ratio_dimer_monomer_initial` | `r_{B/A}` | *requested* initial dimer-to-monomer ratio, `n_B(0) / n_A(0)`; the receptor fraction `x_B = 2r / (1 + 2r)` and the complex fraction `f_B = r / (1 + r)` are derived | log10 | [−2, 2] | 1:100 to 100:1 (complex fraction 1 %–99 %) |
| `rate_dissociation` | `κ_OFF` | dimer unbinding rate, `B_m → A_m + A_m` (1/s); inferred in both conditions | log10 | [−3, 1] | 0.001–10 /s |
| `diffusivity_alp` | `D_A` | monomer scale coefficient `D[A, f]` (µm²/s) | log10 | [−1.25, −0.25] | 0.056–0.562 µm²/s |
| `relative_diffusivity_dimer` | `R_B` | dimer factor within a mode, `D[B, m] = R_B · D[A, m]` | log10 | [−1, 0] | 0.1–1 |
| `relative_diffusivity_slow` | `R_s` | slow-mode factor, `D[X, s] = R_s · D[X, f]` | log10 | [−1, 0] | 0.1–1 |
| `relative_diffusivity_immobile` | `R_i` | immobile-mode factor, `D[X, i] = R_i · D[X, f]`; immobile by construction, a full decade below `R_s` | log10 | [−3, −2] | 0.001–0.01 |
| `rate_fast_slow` | `k_fs` | switching f → s, both species (1/s) | log10 | [−1, 1] | 0.1–10 /s |
| `rate_slow_fast` | `k_sf` | switching s → f, both species (1/s) | log10 | [−1, 1] | 0.1–10 /s |
| `rate_slow_immobile` | `k_si` | switching s → i, both species (1/s) | log10 | [−1, 1] | 0.1–10 /s |
| `rate_immobile_slow` | `k_is` | switching i → s, both species (1/s) | log10 | [−1, 1] | 0.1–10 /s |

The fixed `capture_radius` row (10 nm, display-only; `build_system` derives the
active reaction distance from `particle_diameter_nm`) is the Smoluchowski
contact distance of two monomers.

### How the prior ranges and the declared inputs are set

**Rules.** Every learnable row has a box-uniform prior in the estimator
coordinate, `log10` of the physical value. A range covers the plausible support
of the quantity with a margin; it never encodes the answer the experiment is
expected to give, so no range is narrowed toward a condition's expected outcome
and both conditions share the one table. Bounds come from four sources, named
per row: the frozen reference ranges of the earlier model
(`srm-and-sbi-dimer-alp` at `v0.4.23`, "baseline" below); the per-recording
analysis of the deposited MET localization tables (Special_Analyses A9 —
`srm-and-sbi/Special_Analyses/MET_NEXT_MODEL_DESIGN_PLAN/scripts/A9_spot_density_receptor_totals.py`
with `results/A9_spot_density.md` and `results/A9_per_recording.csv`, outside
this repository; the recipe of record for the count range and the Fab visibility
ratio); the literature anchors of the model specification (its §9); and the
resolution or classification thresholds of the tracking pipelines. Declared
inputs are conventions, not measurements: each carries its source, is editable
in one place (`parameterization.ConditionSetting`), and is labeled provisional
where a collaborator answer is pending. Decided 2026-09-14.

**The learnable rows (eleven).** The table records the decision and its source; the paragraphs below give the reasoning per row, kept apart from the numbers.

| key | symbol | range (estimator, log10) | physical | source |
|---|---|---|---|---|
| `count_total` | `N_R` | [2.5, 3.5] | 316–3162 receptor subunits in the simulated patch | A9 per-recording spot counts under the declared visibility; baseline: three per-species counts, each [0, 2.5] |
| `ratio_dimer_monomer_initial` | `r = n_B / n_A` | [−2, 2] | 1:100 to 100:1 (complex fraction 1–99 %) | symmetric log box replacing the linear receptor fraction `x_B`; lead's decision |
| `rate_dissociation` | `κ_OFF` | [−3, 1] | 0.001–10 per s | baseline [−1, 1]; floor widened for dimer persistence under MET-FAB |
| `diffusivity_alp` | `D_A` | [−1.25, −0.25] | 0.056–0.56 µm²/s | baseline unchanged; specification §9.3 anchors |
| `relative_diffusivity_dimer` | `R_B` | [−1, 0] | 0.1–1 | baseline [−0.625, −0.125] (0.24–0.75) widened at both ends; lead's decision |
| `relative_diffusivity_slow` | `R_s` | [−1, 0] | 0.1–1 | new row; specification §9.3 anchors |
| `relative_diffusivity_immobile` | `R_i` | [−3, −2] | 0.001–0.01 | baseline [−2, −1] rejected; tracking-pipeline immobility thresholds |
| `rate_fast_slow`, `rate_slow_fast`, `rate_slow_immobile`, `rate_immobile_slow` | `k_fs`, `k_sf`, `k_si`, `k_is` | [−1, 1] each | 0.1–10 per s | baseline switching band, kept for all four shared rates |

*Receptor count.* A9 divides the spots per frame in the first 2 s of each of the
120 deposited recordings by the declared visibility per subunit `a`. The
all-monomer reading `N_R = spots / a` and the all-dimer reading
`N_R = 2 spots / (1 − (1 − a)²)` stand in the ratio `2 / (2 − a)`, a relative
increase of `a / (2 − a)`: 6.7 % for MET-FAB and 14 % for MET-INLB. Within this
simplified count calculation the implied total therefore depends only weakly on
the composition; the uncertainty from occupancy and detection is a separate matter.
The all-monomer reading gives `log10 N_R` peaked at 3.0–3.1 (sd 0.21 InlB, 0.30
Fab) with 94 % of recordings inside the box. Localizations undercount the visible
receptors (bleached bound probes, missed detections, dimers with both subunits
labeled seen as one spot), so the implied totals are lower readings of the
visible population; this is the reason for not extending the box downward, not
evidence that no recording has fewer than 316 subunits. Uniform on [2.5, 3.5]
has sd 0.289 against the empirical 0.2–0.3: the box is deliberately at least as
wide as the empirical shape. The prior sets where the training budget goes and,
where the data are weak, contributes to the posterior width; a narrower or shaped
count prior is a later decision. The count is conditional on the declared
occupancy (below).

*Initial composition.* Uniform in `log10 r` on [−2, 2]: symmetric in the complex
fraction `f_B = r / (1 + r)` (median 1/2; quartiles 0.09 and 0.91), so the prior
is even-handed between mostly-monomer and mostly-dimer populations but is not
uniform in `f_B` — half its mass lies below `f_B = 0.09` or above 0.91. The
receptor fraction `x_B = 2r / (1 + 2r)` has median 2/3. Rejected: `log10 x_B` on
[−2, 0], which puts 65 % of the prior mass on `f_B < 0.1` and 5 % on `f_B > 0.6`
and would force the resting answer. Realization: `n_B = round(N_R r / (1 + 2r))`
capped at `floor(N_R / 2)`, `n_A = N_R − 2 n_B`; at the box floor (`N_R = 316`,
`r = 0.01`) three dimers are placed, so every permitted draw contains dimers and
an exactly monomeric initial population lies outside the prior — a scope
statement: the prior represents a small initial dimer population, not its
absence. The box covers 1–99 % complexes without being tuned to any reported
figure; the literature values for resting dimers measure different quantities
under different conditions and are not a reference interval for this row
(specification §5.8).

*Dissociation.* Baseline [−1, 1]. The floor is widened to 0.001 per s so that
under MET-FAB, where no association exists, dimers can persist through a 20 s
recording (0.001 per s loses 2 % in 20 s). The ceiling, a 0.1 s lifetime (five
frames), is the baseline's.

*Diffusivities.* `D_A` keeps the baseline range; anchors are the fast class at
0.13 µm²/s by segment analysis and 0.25 by hidden Markov analysis (specification
§9.3). `R_B` widens the baseline 0.24–0.75 at both ends to 0.1–1: the ceiling so
that dimerization may add no slowdown of its own, population-level slowing then
arising from mode occupancy and inheritance, and the floor so that a strongly
slowed dimer is not excluded by the earlier model's band. `R_s` is a new row (the baseline had
no slow mode); anchors are confined/free 0.7 by segments and 0.28 by hidden
Markov; at 1 the slow mode coincides with the fast one, a degeneracy that is
reported, not removed. `R_i` on [−3, −2] replaces the baseline [−2, −1], under
which the immobile mode reached 0.018–0.056 µm²/s, a range the tracking
pipelines classify as confined and one that touches `R_s`. On the decided range
`D_i = R_i D_A` lies between 0.00006 and 0.0056 µm²/s, below the pipelines'
immobility thresholds (0.0028 and 0.0065 µm²/s) except at the top of the range
near the `D_A` ceiling, and a full decade below `R_s`: the range represents
strongly reduced motion, mostly below the cited thresholds. Its lower half lies
below the 2 s localization-resolution floor (about 0.0005–0.001 µm²/s), so
sensitivity to those values is expected to be weak over two seconds; whether the
classification pipelines assign the mode to their immobile class, and how well
`R_i` is recovered, remain to be evaluated.

*Switching rates.* The baseline band of the earlier mobile/immobile switching is
kept for all four shared rates. Its timescales, 0.1–10 s, include transitions
well inside a 2 s window and slower ones, near the top of the range, that a 2 s
recording may constrain only weakly; the identifiability of four rates from the
videos is not established by the range and is part of the recovery validation. The observed class fractions (67/22/11 Fab,
43/29/28 InlB) imply stationary-law rate ratios between 0.3 and 1, inside the
band.

**Declared inputs of the visibility layer (per condition, editable, provisional).**

| input | MET-FAB | MET-INLB | source and status |
|---|---|---|---|
| probe occupancy `p_occ` | 0.155, **derived** = (0.5 × 0.25) / 0.806 | 0.5, **declared** | InlB: the collaborators' statement (5 nM against `K_D` 5 nM; the published uPAINT protocol reports 0.25 nM in the imaging medium, unreconciled; questions sent 2026-09-11). Fab: no affinity is published; the code stores a declared Fab/InlB **visibility ratio** of 0.5 (`ConditionSetting.visibility_ratio`, anchored on INLB) and derives the Fab occupancy from the InlB anchor as `p_FAB = ratio × a_INLB / P_FAB(dye ≥ 1)`, so a revised anchor propagates. The ratio is a rounded convention over the measured Fab/InlB spot-density ratios (A9): 0.38 first 2 s per µm², 0.41 first 2 s per recording, 0.48 whole-recording means. The derived Fab value is algebraically exact CONDITIONAL on the declared ratio; the measured Fab/InlB spot-count ratio (0.38–0.48, A9) does not by itself establish that visibility ratio, since receptor abundance, composition, detection, and fluorescence loss may also differ between the conditions — 0.155 is a provisional convention, not an estimated occupancy |
| dye probability per bound probe `P(dye ≥ 1)` | 0.806 = 1 − e^(−1.64) (Poisson, DOL 1.64) | 0.5 (Bernoulli, labeling probability) | the measured labeling laws (unchanged) |
| visibility per subunit `a = p_occ × P(dye ≥ 1)` | 0.125 | 0.25 | products of the two rows above (`parameterization.visibility_of`) |
| share of visible dimers with **both subunits labeled**, `a / (2 − a)`, independent occupancy | 6.7 % | 14 % | this fraction helps determine the brightness mixture expected from dimers; it does not by itself determine how accurately composition can be inferred. For InlB (one dye per bound ligand) this is also the share carrying two dyes; for Fab a labeled subunit carries a Poisson number of dyes, so this share is not a two-dye share and brightness classes do not map onto it. If InlB dimers require two bound ligands the InlB share is 1/3 (question sent) |

The DLI stage reads the condition's occupancy by default (`--occupancy` is an
explicit override for sensitivity runs, recorded as such), and the `Labeling_Set`
records the value actually applied in its `occupancy_monomer` and
`occupancy_dimer` columns. Full occupancy is retired as a baseline: uPAINT labels
a sparse subset by design.

**Structural statements that accompany the ranges.** Three mobility modes per
species, sequential switching, four shared rates (decided 2026-09-10). The
immobile mode is immobile by construction (the range of `R_i`). Association is a
per-condition constant, `R_ON = 0` under MET-FAB and 1 under MET-INLB (decided
2026-09-09). The initial composition prior is symmetric in the complex fraction.
Absolute receptor counts and the inferred molecular composition are conditional
on the labeling assumptions: the data constrain the visible subset, and under
independent labeling with per-subunit visibility `a` the expected dimer share
among visible complexes is `n_B (2 − a) / (n_A + n_B (2 − a))`, so even the
visible composition depends on `a`. Descriptive measurements of the detected
population are distinct from their molecular interpretation.

**What would change these.** The collaborators' answers on the probe
concentration, the within-dimer occupancy, and probe residence (questions of
2026-09-11) re-anchor the occupancies and shift the count range by a constant in
`log10`; a lower anchor than 0.5 would push the count above the ceiling and
require regeneration. A narrower or shaped count prior is a later efficiency
decision if the training budget binds. Condition-specific ranges remain
permissible but unused. The structure audit's run tier (`--run`) times one 2 s
simulation at the count ceiling before any tier is generated, and the
prior-realization audit
(`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py`)
checks every generated tier and `Labeling_Set` against this table and these
inputs.

### Initial state

`parameterization.realize_initial_composition(N_R, r)`
turns the sampled pair into integers: `n_total = max(1, round(N_R))`,
`n_B = min(round(n_total · r / (1 + 2r)), floor(n_total / 2))`, `n_A = n_total − 2 n_B`,
so conservation holds exactly for the realized integers (the cap binds only for
ratios beyond the box; at the box floor, `N_R = 316` and `r = 0.01`, three dimers
are placed, so no draw realizes zero dimers). The *requested* ratio `r` and the
*realized* receptor fraction `2 n_B / n_total` are distinguished, and the
realized composition is recorded per simulation in the `Labeling_Set`. Derived
readouts follow from the realized counts: the receptor-level dimer fraction
`x_B = f_R = 2 n_B / N_R` (`= 2r / (1 + 2r)` in the continuum) and the complex
fraction `f_B = n_B / (n_A + n_B)` (`= r / (1 + r) = x_B / (2 − x_B)`). Each
particle's initial mode is drawn from the stationary law of the *isolated*
switching chain, `π_f : π_s : π_i = 1 : k_fs/k_sf : (k_fs/k_sf)(k_si/k_is)`
(`simulation_rds_support.stationary_mode_law`; this is not the steady state of
the reactive system, which inheritance disturbs), and positions are uniform in
the box.

### The one parameter-conversion rule

Each ranged row declares its scale
(`LOG_FLAG` True = log10 coordinate; False = linear). `parameterization.to_physical`
and `to_flow` are the only sanctioned conversions between the estimator's space
and physical values: the RDS sampling, the training dataset
(`inference_support.VideoDataset` takes the workflow's table), evaluation,
calibration, the diagnostics tables, and every analysis use them, and no
consumer writes `10**theta` by hand — a blanket exponentiation would silently
corrupt any linear row, and the rule is what keeps declaring one a free choice.
`Theta_Set` files store **physical** values and carry their **schema** beside
the numbers (`io.theta_set_schema`): the ordered parameter keys, the prior
bounds in the estimator coordinate and the per-row scale flags, the condition
and timing label, the generating stage, and the package version — as `.zarr`
attributes, or as a JSON sidecar beside a plain `.npy`. Every reader of a
`Theta_Set` (the training dataset, the DLI stage's read of the RDS labels,
evaluation, calibration, the embedding-distance analysis, the prior-realization
audit) goes through `io.load_theta_set` with its own table and refuses a file
whose schema is absent or differs in keys, bounds, scales, or condition, naming
what differs. Two tables with the same number of rows are indistinguishable by
shape, which is why the schema exists: a tier generated before the decided
ranges carries no schema and is refused by every reader, so a fresh
per-condition tier under the decided ranges is required (none exists yet).
Prior bounds are part of the schema on purpose — a tier drawn from a different
box is a different training distribution even when the keys agree. Estimator
artifacts and resurrect states carry the same `parameter_keys` guard
(`artifacts.assert_schema_compatible`). The structure audit's D10 exercises the
round trip and the refusals. Trajectories or `Theta_Set`s generated by the earlier three-species
model are rejected at the DLI stage by `rank_to_species` (unknown particle types
`A`/`B`/`C`) — never overwritten or reinterpreted. A deterministic structure
audit (`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.py`,
with its companion note) checks each condition's generated channels (seventeen
under MET-INLB, eleven under MET-FAB), the condition registry and its
association ratios and declared occupancies, the transform round trips, the
integer compositions from the dimer-to-monomer ratio at the box floor and
ceiling, the decided-range table itself (the ranges, immobility by construction,
the dissociation floor, the symmetric composition box), and occupancy by species
without ReaDDy; its run tier (`--run`, small-scale generation checks: subunit
conservation under each condition's network, stationary mode occupancies under
the MET-FAB configuration, visible fractions at the declared occupancies with the
both-labeled share, both-condition generation from each condition's own trajectory,
the lateral boundary, timing at the count ceiling) is gated behind explicit
approval. Its companion, the prior-realization audit
(`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py`,
with its companion note), reads generated products and writes only its report: the tier's
`Theta_Set` against the prior box and the composition rule, its trajectories
against the realized composition and the stationary mode law at frame 0, the
`Labeling_Set` against the declared occupancies, and a descriptive comparison of
the simulated visible counts against the deposited spot counts (Special_Analyses
A9), and — with or without products — the theoretical visibility chain realized
through the DLI stage's own labeling functions for both conditions, the FAB/INLB
visibility ratio against the declared one included; it runs after any tier
generation or DLI pass.

### Detector parameters: calibrated, not free constants

Inferred in the Stage-1 detector workflow, then marginalized as the
calibrated-imaging nuisance for molecular inference (see §3):
- Point-spread-function (PSF) width: **inferred** as a lognormal distribution
  over the Gaussian width (median `mu_r`, log-spread `sigma_r`) — *not* a fixed σ.
- Brightness and photophysics: **inferred** — emitter brightness (median `mu_pc`,
  log-spread `sigma_pc`), photobleaching probability `prob_photo_bleach`, and flicker
  rate `lambda_rate`.
- The five EMCCD camera parameters — gain-conversion ratio `gamma = g/C`, optical
  background `kappa_o`, read noise `kappa_s`, baseline `kappa_b`, and quantum
  efficiency `kappa_q`: **marginalized as the SCOPE camera nuisance**, not inferred.
  They are externally constrained rather than jointly inferred in this workflow: the
  products `gamma·kappa_q` and `gamma·kappa_q·kappa_o` set the amplitude and the floor, so
  the individual quantities are confounded in the biological recordings, while camera
  calibration constrains them one by one; each is drawn from a tight a-priori box and integrated over
  (`DETECTOR_WORKFLOW.md` §9.3); `g` and `C` are fixed nominal spec metadata for the
  `gamma` drift check.
- Labeling (the degree of labeling): **measured, fixed, never inferred** — the
  per-subunit dye-count law of the condition (MET-INLB: Bernoulli with labeling
  probability 0.5; MET-FAB: Poisson at the measured mean of 1.64 dyes per probe),
  drawn once per recording at the DLI stage (§4, *DLI Imaging*).
- Video frame rate: 50 Hz (20 ms per frame) — a fixed sampling cadence. The renderer
  samples positions and brightness at the frame interval and does not integrate motion
  during exposure; the experimental exposure duration is separate acquisition metadata.
- Recording length: supplied per run via `total_time_seconds` (commonly 2 s,
  5 s, or 10 s)

---

## §3. Inference Pipeline: Two-Stage Architecture

The full program separates detector inference from molecular-parameter
inference. The detector is characterized first, with the biology marginalized, and then
marginalized in turn — drawn per simulation from its calibrated nuisance — while
the molecular parameters are inferred. This disentangles
optical and sensor effects from the biological reaction-diffusion parameters,
reduces the dimensionality of each inference problem, improves posterior
geometry, and speeds convergence.

### Stage 1: Detector Parameters (this repository — the Detector calibration workflow)

**Input:** Synthetic videos rendered from the SAME reactive trajectory tiers the
biology workflow uses — one tier per condition, generated under the sibling
alias plus the condition token, whose eleven-parameter `Theta_Set` is the
biology's learnable label and, to the detector, the record of the
reaction-diffusion nuisance it marginalizes — re-imaged through the same imaging
and labeling model under the condition's labeling law, with the imaging drawn
from the detector prior. The detector has no RDS stage of its own.
The reactions are kept because the labeling model makes them observable: a
dissociating one-dye dimer leaves one visible and one invisible daughter, a track
disappearance that a detector trained on static composition could only explain as
photobleaching.

**Objective:** Infer the six learnable imaging parameters (`β`) — the PSF-width
lognormal (`mu_r`, `sigma_r`), the emitter-brightness lognormal (`mu_pc`,
`sigma_pc`), the photobleaching probability `prob_photo_bleach`, and the flicker
rate `lambda_rate` — with the biology marginalized over its full prior, so the
imaging estimate is conditioned on no particular kinetics. The EMCCD camera
chain (`gamma`, `kappa_o`, `kappa_b`, `kappa_s`, `kappa_q`) is **not** inferred:
it is marginalized as the SCOPE camera nuisance, each value drawn per simulation
from its a-priori box (§2). The calibration is per condition, because the
labeling law that shapes the videos is.

**Output:** A posterior over `β` and a versioned, provenanced imaging-parameter
artifact. This calibration is a complete workflow parallel to the biology
pipeline, run in this repository with its own committed submission machinery,
separate from — never wired into — the biology `Submit.sh` dispatcher and its
stage wrappers; its Simulation stage is a DLI-only pass over the condition's
trajectory tier that the RDS-only biology simulation generates.
The calibrated values are the basis for the detector parameters Stage 2 applies;
the mechanism that seeds them into production is developed alongside this
workflow.

### Stage 2: RDS Parameters (this repository)

**Input:** Synthetic videos generated in two steps:
1. **RDS simulation:** ReaDDy solves the stoichiometry–mobility model's
   reactions and diffusion (§2) for the configured recording length.
2. **DLI imaging:** A Gaussian PSF, Poisson photon noise, and EMCCD readout
   noise are applied. The imaging block is **marginalized** per simulation: the
   six calibrated photophysics parameters are drawn from the persisted
   `Nuisance_DLI` artifact (which may be collapsed to a single representative
   vector, such as the sample geometric median), and the five SCOPE camera
   parameters are drawn from their a-priori boxes.

**Objective:** Infer the RDS parameters (`θ`) with the imaging block
marginalized — `p(θ | video)`, integrating over the calibrated-imaging and
SCOPE camera nuisances rather than conditioning on a single fixed `β`.

**Output:** A version-portable estimator artifact and a trained neural-network
checkpoint. Both are written under canonical, duration-stamped names that every
downstream stage loads (`Posit/…_Estimator.npz`, `Labor/…_Optimum_ANN.pth`), and a
re-run on the same duration overwrites them. During training the loop also writes a transient full-state
resume file beside the checkpoint (`Labor/…_Resurrect_State_ANN.pth`) every epoch, from
which a `--resurrect` requeue hot-restarts; it is overwritten continuously and is not a
downstream deliverable. So that superseded models stay identifiable and recoverable,
every finished run also writes a provenance-named backup of both — encoding the
train/test set sizes, epochs, and test loss. See the HPC operations runbook
(*Artifact backups*) for the naming and restore conventions.

---

## §4. Data and Computational Flow

### One draw of the biology per condition, two imagings per draw

The pipeline generates data in two stages, and the two workflows differ only in
the second.

**Stage 1, RDS: the receptors.** One simulation draws the eleven
reaction-diffusion parameters from the prior table and simulates the receptors:
monomers and dimers diffusing in three mobility modes, associating,
dissociating, and switching mode. Its output is a trajectory, and the set of
trajectories of one condition, split, and duration is that condition's **tier**.
The tier is generated once per condition because the association intensity is a
declared per-condition constant of the generator, off under MET-FAB and at the
reference value under MET-INLB, so the two conditions have different reaction
networks (eleven and seventeen channels) and different trajectories. Nothing
else about the tier is condition-specific: both conditions share the one prior
table, and neither workflow asks for a different draw.

**Stage 2, DLI: the imaging.** Each workflow re-images the same tier in its own
way.

| | biology workflow | detector workflow |
|---|---|---|
| what it learns | the eleven reaction-diffusion parameters | the six imaging parameters |
| its training label | the tier's `Theta_Set`, read as is | its own imaging draw from the prior box |
| where its imaging comes from | the condition's `Nuisance_DLI` bridge, built from the detector's calibration | the imaging prior box |
| what it marginalizes | the imaging, through the bridge | the biology, through the tier's `Theta_Set` |
| condition-specific inputs | dye-count law, occupancy, calibrated imaging | dye-count law, occupancy |

The detector marginalizes the biology prior by construction: the tier is a
sample from that prior, already simulated, so re-imaging it needs no second
reaction-diffusion draw and no second copy of the biology ranges. Within a
condition, the two workflows' videos therefore sit on the same trajectories and
the same parameter draws. That is intended: the detector never uses the biology
parameters as a target, and the biology's imaging comes from the bridge rather
than from the detector's training videos, so the two estimators learn different
labels over the same dynamics.

**Naming follows the two stages.** The tier and its `Theta_Set` carry the sibling
alias and the condition token with no workflow qualifier (`Paths.rds_alias`,
e.g. `SRM_AND_SBI_MONOMER_DIMER_ALP_FAB`), because both workflows read them.
Every DLI product carries the alias of the workflow that wrote it. Worked
example, task 3, simulation 7, MET-FAB, TRAIN, 2 s:

- RDS writes `SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_TASK_3_SIM_7_TRAIN.h5`
  and row 7 of `SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Theta_Set_TASK_3_TRAIN.zarr`.
- Detector DLI writes video 7 of
  `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_Video_Set_TASK_3_TRAIN.zarr`
  and row 7 of its `Theta_Set` (the imaging labels), `Nuisance_SCOPE_Theta_Set`,
  and `Labeling_Set`.
- Biology DLI writes video 7 of
  `SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Video_Set_TASK_3_TRAIN.zarr` and
  row 7 of its `Nuisance_DLI_Theta_Set`, `Nuisance_SCOPE_Theta_Set`, and
  `Labeling_Set`; its label is row 7 of the RDS `Theta_Set` above.

One trajectory, two videos, and each video's label is in the `Theta_Set` that
carries the same alias as the video.

### RDS Simulation (this repository, step 1)

**Script:** `Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py`

**Process:**
1. Sample RDS parameters `θ` from the box-uniform prior in the estimator space
   (the log10 coordinate of every row of the decided table; the per-row
   `LOG_FLAG` rule would carry a linear row's value itself) over the prior ranges
   of §2, and map them to physical values
   with `parameterization.to_physical` — the one conversion rule.
2. Initialize a ReaDDy system: the six particle types (`A_f`, `A_s`, `A_i`,
   `B_f`, `B_s`, `B_i`) with their diffusion coefficients, the reaction channels
   generated for the run's condition (`--condition`, required: seventeen under
   MET-INLB, eleven under MET-FAB, whose association ratio is zero), the
   realized initial composition and stationary initial modes, simulation box,
   and observables.
3. Evolve the system for the recording length (`total_time_seconds`).
4. Record particle trajectories (positions, species, time) together with the
   reaction records (educt and product particle ids per event — the RDS/DLI
   handoff the labeling model needs) to an `.h5` file (HDF5, the ReaDDy
   convention), and the sampled theta set to a compressed `.zarr` array.

**One tier per condition, shared by both workflows.** This is the only RDS entry
point, run once per condition; why the tier is per condition and not per
workflow, and how each workflow reads it, is stated at the top of this section
(*One draw of the biology per condition, two imagings per draw*). Because a tier
serves both workflows, regenerating it silently mislabels every video already
rendered from it, so the dataset orchestrator refuses to run the RDS stage over
a condition's existing tier unless told to overwrite, and re-images an existing
tier with `--reuse-rds`; both checks are per condition. Held-out recordings are
matched across conditions by parameter draw, not by trajectory.

**Example quantities:** Particle count per particle type and per molecular
species over time, mode occupancies, mean inter-particle distances,
reaction-event counts per channel.

**Per-simulation kernel release.** The generation loop builds a fresh ReaDDy
system and simulation for each draw. The CPU compute kernel allocates a
worker-thread pool and observable/output handles per simulation; without an
explicit release these accumulate across a long run, growing the task's thread
count and resident memory until a memory-tight node thrashes and the task hangs.
The loop therefore releases the simulation and system objects and forces a
garbage collection at the end of each iteration, so threads and resident memory
stay flat across arbitrarily many simulations while every simulation's output is
unchanged. An opt-in `--probe` flag logs per-simulation thread, file-descriptor,
and memory counts for diagnosing resource behavior, and is off by default.

**Neighbor-list skin (performance, not physics).** ReaDDy finds reaction partners
with a cell-linked list whose cell edge is `reaction_radius + skin`. In the large,
dilute imaging box (~40 µm across, ~1000 particles) the reaction radius alone
(= one particle diameter = 10 nm) partitions the box into a ~16-million-cell grid
that is >99.99% empty, and per-step management of that grid — not the physics —
dominates runtime. The **skin** (Verlet skin) decouples the cell size from the
reaction radius: enlarging it coarsens the grid and recovers roughly a 13× RDS
speedup, while the physics is untouched (reactions still fire only at the true
reaction radius; the skin only widens which particles are considered as
candidates). It is exposed as `SimulationRDS.neighbor_list_skin_factor` — a
**multiple of the particle diameter** (default `10×` = 100 nm) — overridable per run
via `--skin-factor` (the RDS entry point, `Generate_Datasets.py`) or the `SKIN_FACTOR`
batch knob. The cost is U-shaped: too small leaves the empty-cell sweep, too large
collapses the box toward one cell and degrades the candidate search to O(N²); the
default sits on the broad fast plateau and clears the worst-case per-step
displacement (~47 nm at max diffusivity) with margin, so no eligible pair within
the reaction distance at a sub-step boundary is missed by the neighbor search;
encounters between sub-step boundaries are the separately documented
approximation of the 2 ms sub-step.

### DLI Imaging (this repository, step 2)

**Script:** `Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py`

**Process:**
1. Read the RDS trajectory from `.h5`: extract per-frame poses and replay the
   reaction records into the **subunit lineage** — which particle hosts each
   receptor subunit at each frame (fusion concatenates, fission distributes,
   conversion preserves). ReaDDy assigns a new particle id to every reaction
   product, including plain species conversions, so the lineage is what lets a
   static per-subunit quantity follow its subunit through the reactions.
2. Draw the **static dye count** of every subunit once, from the condition's
   labeling law composed with the condition's declared probe occupancy
   (MET-INLB 0.5 declared, MET-FAB 0.155 derived; §2, *How the prior ranges and
   the declared inputs are set*; `--occupancy` overrides it for a sensitivity run
   and the override is recorded as such).
3. Render every **dye** as an emitter at the position of the particle hosting its
   subunit: Gaussian PSF (one width per subunit, carried by its dyes), per-dye
   stationary OU brightness with absorbing photobleaching, and PSF integrals
   accumulated over the pixel grid — co-located dyes' photons add.
4. Apply the corrected EMCCD chain (Poisson photoelectrons, stochastic Gamma
   multiplication, gain-independent read noise, bias).
5. Discretize to an integer pixel count (8- or 16-bit) and save a compressed
   `.zarr` video set, plus the per-simulation `Labeling_Set` record.

**Emitters are dyes (the DOL-explicit observation layer).** Every receptor subunit
carries an integer dye count `kappa` drawn once per recording from the condition's
labeling law and held fixed: dye conjugation happened during sample preparation.
A subunit with no dye has no emitter and never renders; a dimer renders the dyes
of both subunits at one position, so its brightness is the sum of independent
per-dye processes — an *n*-dye spot's brightness distribution is the *n*-fold
convolution of the single-dye law (Mutch et al. 2007; Digman & Gratton 2008
number-and-brightness) — with no brightness multiplier anywhere. The laws are
measured or preparation-level inputs, fixed and never inferred: from the video
alone the labeling probability is nearly degenerate with the receptor counts.

| condition | law per subunit | invisible monomers | invisible dimers |
|---|---|---|---|
| MET-INLB | Bernoulli, `q = 0.5` — one engineered attachment site; the labeling probability measured for this preparation (reported by the collaborating laboratory, 2026; not in the source publications) | 50% | 25% |
| MET-FAB | Poisson at the measured ensemble mean `DOL = 1.64` (Harwardt et al. 2017) — the working preparation-level model, with matched-mean binomial and negative-binomial alternatives registered for sensitivity runs | 19.4% | 3.8% |

The table states the laws alone. In the rendered videos each law is composed
with the condition's declared probe occupancy — the probability that a subunit
carries a probe at all (MET-INLB 0.5 declared; MET-FAB 0.155, derived from the
declared Fab/InlB visibility ratio 0.5 and the InlB anchor; §2, *How the prior
ranges and the declared inputs are set*) — so a subunit is visible with
`a = p_occ × P(dye ≥ 1)` = 0.25 (MET-INLB) or 0.125 (MET-FAB), a dimer with
`1 − (1 − a)²`, and dimers with both subunits labeled are `a / (2 − a)` = 14 % (MET-INLB)
or 6.7 % (MET-FAB) of the visible dimers (for MET-INLB, one dye per bound ligand, these are
also the two-dye dimers; a labeled Fab subunit carries a Poisson number of dyes).

Three consequences follow from the draw, none an extra assumption. The visible
fraction differs by species (a dimer is visible when either subunit is), so the
visible population is dimer-enriched relative to the true composition, and the
receptor total `N_R` is a **true receptor abundance**: the RDS stage simulates every
receptor, labeled or not, because the reacting population sets the encounter
rates, and the observation layer decides which of them render. Among visible
MET-INLB dimers under the bare law two thirds carry one dye and one third two
(86 % and 14 % under the declared occupancy), so the visible-dimer brightness is
a mixture of one-dye and two-dye emission rather than a doubled monomer. And because the dye count travels with its subunit, a one-dye
dimer that dissociates leaves one visible daughter and one permanently invisible
one — an observable signature of the labeling statistics, reproduced by the
lineage bookkeeping and erased by any renderer that redrew labels per frame or
per particle. The `labeling` module holds the laws (`LABELING_LAWS`), the draw
composed with the probe occupancy (the condition's declared or derived value by
default, from `parameterization.occupancy_of`; `--occupancy` is an explicit
override for sensitivity runs, given as one scalar or per molecular species as
`A=0.5,B=0.4`), and the ten `Labeling_Set` columns recorded per simulation (the
true and visible initial composition, the dimers with both subunits labeled, and
`occupancy_monomer` / `occupancy_dimer`, the occupancy actually applied —
declared, derived, or override).
Occupancy and labeling act by *molecular species*, never by mobility mode:
`simulation_rds_support.rank_to_species` maps the trajectory's particle-type
ranks to species, and the lineage extractor reads subunit counts per particle
type.

**The condition axis.** The condition (MET-FAB or MET-INLB) enters the imaging
here through its labeling law and its own detector calibration, having already
selected the trajectory tier at the RDS stage (the association setting is per
condition, §2): the DLI stage re-images the condition's own trajectories, and
every product — the RDS products included — carries the condition token
(`--condition`, required on every stage; *Condition slot in the naming grammar*,
below). MET-FAB and MET-INLB are two frozen configurations of one codebase,
generated, trained, and validated separately.

**Probe kinetics: the stated assumption.** The dye count of a subunit is fixed for
the whole recording, so the simulator removes a dye only by photobleaching and
never adds one. Ligand binding and unbinding within a recording are therefore not
in the model, and for MET-INLB the probe is the ligand. The unbinding channel is
absorbed by construction: the detector calibrates the bleach parameter on the INLB
recordings, so the calibrated value is photobleaching plus first-order unbinding,
and the biology DLI draws that value — a first-order, state-independent unbinding
is statistically indistinguishable from the modeled bleaching. Two residuals are
not absorbed and are declared here rather than modeled: a labeled InlB binding
from solution during a recording would create a spot the simulator never creates,
which matters only if free labeled ligand was present during imaging (a property
of the source acquisition); and a ligand affinity that differs between monomeric
and dimeric MET would make disappearances stoichiometry-dependent, which a
state-independent bleach cannot absorb. Partial ligand occupancy, the static part
of the same question, is the declared per-condition probe occupancy (MET-INLB
0.5, a provisional convention until the collaborators' answers; §2), applied by
default at the DLI stage and overridable with `--occupancy` for a sensitivity
run. The posterior-predictive video check is where a violated assumption shows:
appearances in the experimental video with none in the synthetic one.

**Photobleaching model:** Each dye can irreversibly transition to a dark
(bleached) state, applied per frame through a two-state transition matrix. The
per-frame bleach probability is
`prob_1 = 1 − (1 − prob_photo_bleach)^(1 / numb_photo_bleach)`, where
`numb_photo_bleach = 100` is a fixed reference-frame count (a calibration
convention, *not* the clip or video frame count) and `prob_photo_bleach` is the
cumulative bleach fraction over that 100-frame (2 s at 50 Hz) reference window —
a detector-inferred photophysics parameter, marginalized per simulation from the
`Nuisance_DLI` artifact (production calibration ≈ 0.107), not a fixed constant.

The bleach rate is parameterized by this pair — a cumulative bleach probability
together with the reference-frame count over which it accrues — rather than by a
single hardcoded 100-frame probability, which makes the model correct for any
recording length. Because `numb_photo_bleach` is pinned to 100 regardless of
clip duration, the per-frame rate `prob_1` is constant across video lengths; the
cumulative bleach over an `n`-frame clip is
`p_video = 1 − (1 − prob_photo_bleach)^(n / numb_photo_bleach)`, so a longer clip
accumulates more bleaching through repeated application of the same per-frame
transition matrix (at `prob_photo_bleach ≈ 0.1`: about 10% over 100 frames /
2 s, about 41% over 500 frames / 10 s).

The reference count is pinned to 100 by design rather than tied to the clip
length. The value of `prob_photo_bleach` over the 100-frame reference was
calibrated by detector-parameter SBI inference on the experimental raw videos,
with the reaction-diffusion parameters held out — the production `Nuisance_DLI`
calibration puts it at ≈ 0.107; `numb_photo_bleach = 100` is the
convention that inference was run under, so it is preserved to keep the
calibrated value meaningful. Making `numb_photo_bleach` track the clip length
would render the per-frame rate duration-dependent — unphysical, since bleaching
is a property of the fluorophore and illumination, not of how long a recording
happens to be — and would contradict the detector-inferred value. (The aggregate
`p_bleach` reported by swift / SPTAnalyser is a track-disappearance rate that
lumps together bleaching, diffusion out of the field, blinking and gaps, and
unbinding; it describes a different process and is not substituted for
`prob_photo_bleach`.)

**Detector noise model (EMCCD Poisson–Gamma–Normal).** The imaging stage renders
camera counts in a single function, `add_noise` (`simulation_dli_support.py`),
following the physically grounded EMCCD chain specified in
`REFERENCE_EMCCD_NOISE_MODEL.md`: Poisson photoelectrons, stochastic `Gamma(N, g)`
electron multiplication (excess-noise factor `F² = 2`), conversion to ADU by the
factor `C`, a gain-independent Gaussian read noise of standard deviation `σ` added
*after* the register, and a constant camera baseline `b`. The gain `g` and
conversion `C` enter the image likelihood only through the ratio `γ = g/C` (the
ADU-per-photoelectron), so `γ` is inferred directly and `g`/`C` are fixed nominal
metadata; the optical background `kappa_o`, read noise `σ`, and baseline `b` are
identified directly from the background and dark pixels.

The read-noise term produces a small negative excursion (pixels below the baseline, and
rarely values above the sensor range); the non-negative storage clip in
`convert_video_dtype` removes it, aligning the stored synthetic with the recordable
camera domain, and its floor is re-examined against the read-noise scale after
re-calibration. Parameter recovery is validated on the held-out synthetic EVAL
namespace; real recordings have no ground truth. See `REFERENCE_EMCCD_NOISE_MODEL.md`
for the full specification, moments, prior ranges, and sources.

**Output:** A `.zarr` video set (chunked array, efficient I/O). Shape
`(frame_count, height, width)` — for example `(100, 256, 256)` at 2 s and 50 Hz.

### Fixed-cadence, per-run timing model

The frame count is always derived from the recording length and the fixed frame
rate: **`frame_count = total_time_seconds / frame_time_seconds`** (recording
length times frame rate). The frame rate is fixed configuration; the recording
length is per-run. These two roles are kept structurally separate so the frame
count cannot be read from a stale global default:

- A global `FrameConfig` holds only the fixed sampling cadence
  (`frame_time_seconds`, `steps_per_frame`, and the derived frames-per-second
  and time step). It carries no per-run duration at all.
- A per-run `RunTiming` is constructed at each entry point from a **required**
  `--total-time-seconds` argument. It derives the per-run `frame_count`, total
  step count, and timing label, and passes through the fixed-cadence values.

Because `frame_count`, `total_time_seconds`, and the timing label exist only on
the per-run object and never on the global, no code can silently inherit a
default duration. Fail-loud guards back this up: trajectory extraction iterates
the trajectory's own recorded frame count and raises on an entirely empty frame,
and the imaging stage raises if the extracted frame count differs from the run's
declared `frame_count`. This is why `--total-time-seconds` is required on every
stage and why every duration produces full-length, fully-populated videos.

### Inference (this repository, step 3)

**Script:** `Script_Bank/Prime/SRM_AND_SBI_MONOMER_DIMER_ALP_Inference.py` (with an
optional `--resurrect` flag to continue from an existing checkpoint)

**Process:**
1. **Training data:** Use the (θ, video) pairs produced by RDS and DLI, where
   `θ ~ prior` and the video is the imaging of the trajectory that `θ` produced.
2. **Neural density estimation:** Train a neural posterior estimator (a masked
   autoregressive flow over a learned video embedding) to approximate
   `p(θ | x)` from the simulated pairs. The embedding is a 3D convolutional
   video encoder followed by a temporal transformer (§6).
3. **Resurrect mode (optional, runtime flag):** With `--resurrect`, the script
   resumes training. When a full-state resume file is present it **hot-restarts**
   from the exact latest state — model weights, optimizer moments, learning-rate
   schedule, global epoch, and warm-restart counters — so the schedule continues
   seamlessly and no epochs are spent re-converging. When it is absent (the first
   resumed run, or the file was deleted) it falls back to loading the best-on-test
   checkpoint weights into a fresh optimizer at the peak learning rate, then writes a
   resume file so the next requeue hot-restarts. A new optimum overwrites the
   checkpoint. This makes incremental training across separate, wall-time-limited
   invocations behave like one continuous run.
4. **In-run warm restart (automatic):** The training loop monitors the learning
   rate; once it has decayed to its floor and stalled there without improving, the
   loop reloads the best checkpoint and restarts the rate at a decaying peak — a
   plateau-escape that periodically re-raises the rate to find a better optimum. Each
   restart peak is `warm_restart_factor` (default 0.25) times the previous, a dedicated
   amplitude knob separate from the per-epoch anneal `scheduler_factor`, so the restart
   stays a gentle probe (a quarter of the peak) rather than a jump halfway back up. Its
   state is carried in the resume file, so the sawtooth continues mid-stride across a
   `--resurrect` requeue rather than resetting. Governed by `warm_restart_dwell`
   (epochs of stalled floor before a restart; `0` disables it); composes with
   `--resurrect`.
5. **Output:** Posterior samples are obtained by passing a real experimental
   video (or a synthetic holdout) through the trained network.

**Output:**
- A version-portable estimator artifact (`Estimator.npz`), loaded downstream as a
  `DirectPosterior`.
- A trained network checkpoint (encoder, transformer, and posterior-parameterizer
  weights), saved whenever a new optimum is reached.

**Training-loop efficiency.** The data loaders keep their worker pool alive
across epochs (`persistent_workers`) so that a long training run does not pay the
cost of re-spawning and re-importing the full stack in every worker each epoch —
which otherwise dominates the per-epoch wall time. Because each data-parallel rank
builds its own loaders and the train and validation loaders' workers stay alive
together, the live worker-process count is (workers per loader) × (ranks) ×
(concurrent loaders). The per-loader count is therefore derived from a node-wide
TOTAL budget — the machine profile's `num_workers`, or the CPU core count when unset
— divided across the ranks and concurrent loaders, so the live total stays near one
worker per core at any GPU count (and reduces to half the cores per loader on a
single GPU). This keeps a multi-GPU run from exhausting host memory through worker
multiplication. This budget is the data-loading workers only; the GPU/shard-worker
count is bounded separately (see Multi-GPU scaling).

**Multi-GPU / multi-node scaling.** Training, MAP-recovery evaluation, and the
real-data application adapt to the allocation — across GPUs on one node and across
nodes. `--gres` is per node, so `--nodes=N --gres=gpu:G` gives `world_size = N*G`
ranks. Launched with one worker per GPU (via `torchrun` on one node, or `srun` +
`torchrun` with a c10d rendezvous across nodes), training runs data-parallel
through `DistributedDataParallel` — each worker holds a replica, processes its own
shard of every batch, and synchronizes gradients each step across every rank on
every node, with `SyncBatchNorm` sharing batch statistics across all ranks by
default (so the batch size is per-rank and the effective batch is
`batch*world_size`) — while MAP-recovery evaluation partitions the held-out videos
across the ranks, and the experiment stage partitions its `(condition, cell)` work
the same way, each writing its own shard to the shared filesystem and a single
`--merge` step combining them into one report. The single-GPU run is the collapse
case of the same code: with one worker the distributed wrappers reduce to no-ops
and the loop is exactly the original single-GPU path, so behavior is unchanged
where only one GPU is present. The per-node GPU count is read from the allocation,
capped by an optional `SRM_AND_SBI_GPUS` override (set it to 1 to force the
single-GPU path), and the node count from `SLURM_NNODES`; `SRM_AND_SBI_NO_SYNC_BN=1`
opts each worker into its own local batch statistics for speed at the cost of
re-validating recovery.

### Leak-proof data split (TRAIN / TEST / EVAL)

Training data, model-selection data, and final-validation data are physically
separated into three on-disk namespaces, distinguished by an explicit suffix at
the end of every output name (`_TRAIN`, `_TEST`, `_EVAL`):

- **TRAIN** supplies the gradient updates.
- **TEST** supplies the per-epoch model-selection signal (the best-on-TEST
  checkpoint is kept; the network never trains on TEST). With no TEST set, the
  last-epoch checkpoint is kept instead.
- **EVAL** is held out entirely — never touched by gradients or model selection,
  only by the final MAP-recovery report.

Each split is an **independent draw with its own seed**, not a shuffled reuse of
a single pool. Because the splits never share samples, validation leakage is
impossible by construction: a reported recovery can never reflect data the
posterior already optimized against.

A single orchestrator script generates a complete, correctly proportioned set,
running the full RDS → DLI flow for all three splits with independent seeds and
enforcing the sizing rule:

- CORE = TRAIN + TEST, with a minimum of 10 samples,
- TRAIN = 0.8 · CORE,
- TEST = 0.2 · CORE,
- EVAL = max(10, 0.1 · CORE).

A `--dry-run` flag previews the sizing plan without generating anything.

### Condition slot in the naming grammar

The runtime grammar is `[program]_[sibling]_[iter][_qualifier]_[condition]_[timing]_[stage]`:
the workflow qualifier (`_DETECTOR`) and the experimental-condition token (`FAB` or
`INLB`) sit between the iteration and the timing label. The condition enters at the
RDS stage — the association setting of the reaction-diffusion model is per condition
— so every product carries the token. The RDS products carry it without the workflow
qualifier, because each condition's tier is shared by both workflows, which re-image
it at the DLI stage:

| product | alias | example |
|---|---|---|
| trajectories (one tier per condition, shared by both workflows) | sibling alias + condition token, no qualifier (`Paths.rds_alias`) | `SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_TASK_0_SIM_0_TRAIN.h5` |
| the tier's eleven-parameter `Theta_Set` (the biology labels; the detector's RDS-nuisance record) | sibling alias + condition token, no qualifier | `SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr` |
| `Video_Set`, `Labeling_Set`, `Nuisance_SCOPE_Theta_Set`, biology `Nuisance_DLI_Theta_Set`, detector `Theta_Set` (the imaging labels) | conditioned, qualified for the detector | `SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Video_Set_TASK_0_TRAIN.zarr` |
| estimator, checkpoints, recovery and experiment reports, analyses, the `Nuisance_DLI` artifact | conditioned, qualified for the detector | `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_Estimator.npz` |

Every stage takes `--condition`, the RDS stage included; `Paths.with_condition`
appends the token to the alias, and the trajectory and theta-set builders resolve the
condition's tier alias (`Paths.rds_alias` — the sibling alias plus the condition token,
which refuses to resolve without a condition) where a product belongs to the tier. HPC
job and log names follow the same slot
(`SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Inference`,
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_Experiment`); an RDS-only
simulation job carries the condition and no qualifier
(`SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Simulation_TRAIN`), and the detector's
Simulation job — a DLI-only pass over the condition's tier — carries both.

### Non-deterministic generation and task-index provenance

Generation passes no random seed by default: prior sampling, particle placement,
PSF and brightness draws, and camera noise all draw fresh entropy, so every
simulation is an independent random draw. This matches the reality that the
reaction-diffusion stepper itself is not seedable — identical-output
reproducibility past the prior-sampling stage is unreachable regardless, so a
deterministic per-simulation seeding scheme would add bookkeeping without a
payoff and could introduce subtle cross-split correlations. No scientific
information is lost, because the sampled theta is persisted to disk per task, so
every (theta, video) pair is recorded.

Dataset integrity instead rests on a **global task index encoded in the file
names** (`..._TASK_<tid>_<split>`). When generation is fanned out across many
parallel tasks, each task is assigned a distinct global index, so parallel
packing, array overflow, and incremental appends (via a task-offset base) can
never collide on a filename or mix split namespaces. The index scheme is
injective by construction, and an analysis script asserts file-label uniqueness
across an entire fan-out before training. A `--seed` flag remains available on
every stage for an optional deterministic run, but the default — and the
batch-generation scripts — pass nothing.

### Two-tier data-bank storage, routed by data role

The permanence of a file follows its scientific role rather than the machine. A
machine whose fast scratch filesystem is large but impermanent (auto-purged, not
backed up) can route the *regenerable* bulk — TRAIN and TEST — onto scratch while
keeping everything that must persist — EVAL, posteriors, training checkpoints,
and experimental data — on permanent, backed-up storage. The machine profile
gains an optional scratch root; a single resolver returns the scratch root for
TRAIN/TEST when it is configured and the permanent root otherwise. The on-disk
layout under each root is identical (a sparse mirror), and the split suffix
already in every filename identifies which tier a file belongs to. A machine that
configures only the permanent root is single-tier and behaves identically, so the
feature is invisible where it is not used. Experimental microscopy data is
external and irreplaceable, so it always lives on permanent storage and never on
scratch.

### Support functions (Python package: `srm_and_sbi_monomer_dimer_alp/`)

The support code is organized into a flat Python package of focused modules
rather than a single monolithic support file. The modules and their roles:

- **`workflow.py`** — the shared-engine control layer. It defines the frozen
  `WorkflowConfig` and the two factories `biology_workflow()` and
  `detector_workflow()` that build it. The config carries the genuine
  per-workflow differences — the workflow tag, the alias-qualified paths, the
  parameterization module, and the console-log paths — so one engine serves
  both the biology workflow (infers the eleven reaction-diffusion parameters and
  marginalizes the imaging block) and the detector workflow (infers the six
  imaging parameters and marginalizes the reaction-diffusion domain and the
  camera).
- **`simulation_rds_runner.py`, `simulation_dli_runner.py`,
  `inference_runner.py`, `evaluation_runner.py`, `experiment_runner.py`** — one
  shared runner per stage, each exposing `run_<stage>(cfg, args)`. This is the
  single orchestration engine both workflows execute for that stage, so neither
  can silently drift; the per-workflow differences are localized in a
  `_<stage>_spec(cfg)` resolver rather than duplicated across entry points. In
  particular, `simulation_dli_runner`'s per-stage spec resolver resolves the DLI
  imaging source — the `Nuisance_DLI` artifact for biology, the imaging prior
  box for detector. `simulation_rds_runner` is the one runner without a workflow
  fork: it generates one condition's trajectory tier (`--condition`, required;
  the sibling alias plus the condition token) through the biology config and
  refuses a detector config, since the detector re-images that tier rather than
  simulating its own.
- **`parameterization.py`** — the single source of truth for configuration. It
  loads the per-machine profile, holds the sibling-wide defaults as frozen
  dataclasses (path conventions, simulation geometry and timing, RDS/DLI
  defaults, the model blocks and the per-condition association settings,
  inference training and network architecture, plotting), and carries
  the rich parameter specification (parameter ranges, log flags, units, and
  labels). It exposes a typed `PARAMETERS` singleton and prior/bounds helpers,
  and validates the configuration at import time.
- **`detector_parameterization.py`** — the detector workflow's parameter
  contract (value-based roles): the six learnable imaging parameters' priors,
  with the reaction-diffusion block marginalized as a nuisance supplied by the
  condition's trajectory tier (nuisance-from-object: the eleven rows declare the
  role and carry no ranges of their own; the association ratio is a constant of
  the generator, not a row) and the SCOPE camera as a nuisance drawn from its
  box; deliberately decoupled from `parameterization.py`, since the two
  workflows' roles differ by design.
- **`simulation_rds_support.py`** — the ReaDDy primitives: system builder,
  simulation builder (which registers the reaction-record observable),
  trajectory-pose extraction, and the subunit-lineage extraction that replays
  the reaction records into a per-frame subunit-to-particle table with a
  conservation check (the RDS/DLI handoff of the labeling model).
- **`labeling.py`** — the static labeling stoichiometry: the per-condition
  dye-count laws and their registry, the once-per-recording draw composed with
  the condition's declared probe occupancy (or its `--occupancy` override), and
  the `Labeling_Set` provenance columns, the applied occupancy included.
- **`simulation_dli_support.py`** — the imaging pipeline: Gaussian PSF, EMCCD
  detector, intensity accumulation, the brightness photo-physics (stationary OU
  ln-brightness flicker with absorbing photobleaching), the dye-track builder,
  and the top-level dye-centric renderer.
- **`detector_simulation_dli_support.py`** — the detector DLI forward model:
  re-exports the shared, source-agnostic renderer `render_dli_video` under the
  detector-facing name `render_detector_video`, so both DLI stages render
  through one implementation.
- **`detector_nuisance_dli.py`** — the `Nuisance_DLI` calibrated-imaging
  nuisance: the artifact format, its construction from the detector posterior
  (including the pool modes and the single-vector sample-geometric-median
  collapse), and the require-gate through which the biology DLI stage loads it.
- **`experiment_support.py`** — the workflow-agnostic real-recording machinery:
  loading, windowing, and preparing experimental microscopy videos. It is shared
  by both the Experiment stage and the `Nuisance_DLI` analysis, so it carries no
  workflow-specific assumptions.
- **`inference_network.py`** — the network architecture (§6): positional
  encoding, attention block, temporal transformer, and the 3D-CNN video encoder.
- **`inference_support.py`** — the training pipeline: the video dataset (with
  augmentation), normalization, training/validation set-up, the training loop
  (with the resurrect branch), and posterior save/load.
- **`artifacts.py`** — the self-describing, version-portable estimator artifact:
  persists a trained estimator as separable components (compile-stripped
  `state_dict`, rebuild spec, parameter schema) in one `.npz`, so it
  reconstructs under whatever torch version loads it; the sole persisted
  estimator format for both workflows.
- **`evaluation.py`** — the MAP-recovery core shared by the validation stages:
  the seed-then-optimize estimator, posterior-quantile summaries, and report
  tables.
- **`posterior_calibration.py`, `posterior_calibration_runner.py`** — the
  workflow-agnostic posterior-calibration diagnostic (an Analysis tool, not a
  pipeline stage). The kernel scores a trained posterior's calibration with
  simulation-based calibration, expected coverage, TARP, and local C2ST (§7),
  overall and stratified by target-theta dimension, operating only on pre-drawn
  theta-space arrays and embeddings (it wraps `sbi.diagnostics` and imports nothing
  from `parameterization`/`artifacts`). The runner streams the EVAL set, draws each
  video's posterior samples, sample and truth log-densities, and embedding, and
  writes the report; its two namespaced Analysis shims share it exactly as the stage
  shims share their runner.
- **`estimator_comparison.py`, `estimator_comparison_runner.py`** — the
  workflow-agnostic estimator-comparison diagnostic (an Analysis tool, not a pipeline
  stage). The kernel decides whether one trained estimator generalizes better than
  another by the paired log-score on the shared `(task, sim)` subset of the held-out
  TEST set (Diebold-Mariano + Wilcoxon + paired bootstrap; §7), operating on two
  per-video loss arrays and importing nothing from `parameterization`/`artifacts`. The
  runner reads two `TestLossDistribution` artifacts (per-video loss keyed by
  `(task, sim)`), runs the statistics, and reports; its two namespaced Analysis shims
  share it over one engine. No GPU.
- **`test_loss_distribution.py`** — the per-example best-epoch TEST-loss
  artifact: holds the held-out per-example loss keyed by `(task_index,
  sim_index)` with a self-describing manifest; written by the Inference stage
  and read by the comparison and test-loss analyses.
- **`test_loss_analysis.py`, `test_loss_analysis_runner.py`** — the workflow-agnostic
  test-loss-distribution analysis (an Analysis tool, not a pipeline stage). The kernel
  reads a best-epoch per-example NLL artifact and produces the distribution shape, the
  uniform-prior NLL reference (the no-information baseline), and the tail-vs-parameter
  identifiability read — which learnable parameters, and which end of their range, mark
  the hardest examples (for biology the receptor total `N_R`, whose low end yields
  uninformative videos). Everything is read from the artifact manifest, so it is
  workflow-agnostic; the runner resolves the artifact through `cfg.paths` (or an ad-hoc
  `--tld-path`) and reports; its two namespaced Analysis shims share it. No GPU.
- **`embedding_space_distance.py`** — the workflow-agnostic embedding-space
  distance kernel: the maximum-mean-discrepancy (MMD, permutation null) and
  classifier two-sample (C2ST) statistics over trained-embedding vectors,
  blocked by recording so within-recording correlation cannot masquerade as a
  real difference.
- **`embedding_space_distance_runner.py`** — the shared engine for the
  experimental-versus-synthetic embedding-distance analysis (does the trained
  network place the real recordings where it places its own synthetic
  distribution?); its two namespaced Analysis shims share it.
- **`sample_geometric_median.py`** — the workflow-agnostic sample-geometric-median
  kernel: the correlation-preserving single-vector summary of a cloud of
  parameter vectors (the median vector snapped to a realized sample, never the
  vector of per-dimension medians).
- **`sample_geometric_median_runner.py`** — the shared engine for the
  sample-geometric-median analysis over the Experiment MAP cloud (and, for
  detector, the `Nuisance_DLI` pool); its two namespaced Analysis shims share it.
- **`posterior_predictive_video_runner.py`** — the shared engine for the
  posterior-predictive video comparison: render a synthetic video at the
  parameters inferred from one real recording and put the two side by side; its
  two namespaced Analysis shims share it.
- **`temporal_dynamics.py`** — the workflow-agnostic temporal-dynamics kernel:
  scatters the per-window Experiment estimates into a (condition, recording,
  window) grid; forms the two Sample-Geometric-Median central estimates (the
  trajectory-level medoid, one real recording across the whole time course, and
  the per-time-point medoid, whose selected recording may change between time
  points); fits the per-recording drift in dex with its sign-consistency and
  signed-rank test; and separates the within-window posterior spread from the
  between-cell spread so the two are never conflated.
- **`temporal_dynamics_runner.py`** — the shared engine for the temporal-dynamics
  analysis (does an inferred value hold still across a recording?); its two
  namespaced Analysis shims share it. Both workflows must be run to interpret
  either: biology holds imaging fixed and so is blind to imaging drift, the
  detector marginalizes the reaction-diffusion block and so is blind to
  biological drift, and the two read the same recordings — each is the other's
  control, and neither attributes a cause alone.
- **`population_composition.py`** — the workflow-agnostic population-composition
  kernel: the monomer–dimer composition derived from JOINT posterior draws of the
  stoichiometry coordinates — the receptor total `N_R` and the initial
  dimer-to-monomer ratio `r`, from which `x_B = f_R = 2r / (1 + 2r)`,
  `f_B = r / (1 + r)`, and the realized integer counts follow (§2). It forms each readout inside a draw and
  only then averages, because a function of correlated coordinates is not a
  function of their marginals. It carries the aggregation ladder (draws to window, windows to
  recording, recordings to condition, with the recording as the replicate unit),
  the prior-support and compositional-center sensitivity variants, the
  recording-level rank tests, and the same readout's recovery on held-out
  synthetic videos.
- **`population_composition_runner.py`** — the shared engine for the
  population-composition analysis, reporting the experimental composition and its
  measured in-model error in one document. Biology only, and the asymmetry is
  scientific: the composition is a function of the inferred stoichiometry
  coordinates (`N_R`, `r`), and the detector workflow infers imaging parameters
  and treats the population implicitly, so it has nothing to compose — as the
  detector's `Nuisance_DLI` pool has no biology counterpart. The spec resolver
  fails loudly for a workflow without stoichiometry parameters rather than
  composing unrelated coordinates. It
  requires the Experiment stage's raw per-window draws
  (`--dump-posterior-samples`); the stored marginal quantiles cannot substitute,
  since a fraction of marginals is a different quantity rather than a coarser one.
- **`io.py`** — file I/O: transparent loading of `.zarr`/`.npy`/`.npz`, video
  and theta-set writing, and bit-depth conversion. All paths come from the
  configuration helpers, never hardcoded. The `Theta_Set` schema lives here too: `theta_set_schema()` builds it from a parameter table, `write_theta_set()` stores a theta set with it (`.zarr` attributes or a JSON sidecar beside a `.npy`), `load_theta_set()` / `check_theta_set_schema()` refuse a theta set whose schema is absent or differs (`ThetaSetSchemaError`), and `theta_set_status()` renders the verdict for dry runs.
- **`visualization_rds.py`, `visualization_dli.py`, `visualization_inference.py`,
  `visualization_calibration.py`** — stage-specific diagnostic and figure builders
  (matplotlib imported lazily so headless runs do not pay its import cost).
- **`diagnostics.py`** — the shared diagnostics engine behind the `--debug` /
  `--debug-dump` flags: per-step checkpoints, fail-loud invariant checks,
  quantitative stats, and a self-contained dumped report.
- **`utils.py`** — small cross-cutting helpers (terminal separators, memory-state
  logging, and the resource-probe helpers).

Each Prime entry point is a thin shim: it parses CLI arguments, builds a
`workflow.WorkflowConfig` (via `biology_workflow()` or `detector_workflow()`),
and calls the stage's shared `run_<stage>(cfg, args)` runner, which loads the
configuration, calls the package functions, and writes outputs to the
configuration-defined paths. Each stage has two such shims over one shared
runner — the unqualified biology entry point and its `_DETECTOR`-qualified
detector counterpart. The package is installed editable, so edits to the package
take effect without reinstallation.

`Script_Bank/Analysis/` collects post-hoc analyses that run on completed outputs
rather than producing pipeline artifacts:
`SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_Temporal_Dynamics.py` tracks each inferred
parameter's MAP estimate over the real recordings per condition (non-overlapping
chunk → time), overlays the experimental range for the parameters the source paper
constrains (Li et al. 2026, doi:10.1002/smll.202507115), annotates each figure with
its held-out recovery quality, and writes figures plus a self-contained `report.md`;
its companion `Experiment_Temporal_Dynamics.md` gives the full interpretation.
`SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_Population_Composition.py` reports the
monomer–dimer composition across the experimental recordings — the share of receptors in
dimers `f_R = x_B`, the share of complexes that are dimers `f_B`, and the receptor total
`N_R` — formed inside each posterior draw so the correlations between the stoichiometry
coordinates are carried through, aggregated with the
recording as the replicate unit, and reported beside the same readout's error on held-out
synthetic videos with known truth, so an experimental value never appears without the
measured accuracy of the instrument that produced it. It reports the span-averaged and
first-window compositions, the within-recording time course, the per-recording spread, a
bootstrap check of the error bars, the sensitivity of the headline to prior-support
restriction and to the choice of compositional center, and the recording-level condition
contrast. Biology only: the detector workflow infers no stoichiometry parameters and so
has no composition to report. Its companion
`SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_Population_Composition.md` documents the derivation, what
the result does and does not establish, and how it relates to the published
trajectory-classification and photobleaching-stoichiometry measurements of the same receptor
system.
`SRM_AND_SBI_MONOMER_DIMER_ALP_Seeding_Validation.py` checks the RNG / non-determinism
behavior of the generation stack.
`SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.py` is the deterministic structure
audit of the stoichiometry–mobility model (§2): the channels generated per condition
(seventeen under MET-INLB, eleven under MET-FAB), the condition registry with its
association ratios and declared occupancies, the `to_physical` / `to_flow` round trips,
the integer initial compositions from the dimer-to-monomer ratio, the decided-range table
itself, and occupancy by molecular species, with an approval-gated `--run` tier of
small-scale generation checks; its companion `.md` documents the checks and the verdict.
`SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py` is its read-only companion over
generated products: it checks a condition's `Theta_Set` against the prior box and the
composition rule, its trajectories against the realized composition and the stationary
mode law at frame 0, and its `Labeling_Set` against the declared occupancies, and compares
the simulated visible counts descriptively against the deposited spot counts
(Special_Analyses A9), and corroborates the theoretical visibility chain (per-subunit
visibility, visible fractions, both-labeled and dark-partner shares, emitters per subunit, the
dye-count law among visible subunits, the FAB/INLB visibility ratio) through the DLI stage's
own labeling functions on a synthetic lineage for both conditions; `--selftest` verifies the
checks on in-memory draws without a tier, `--visibility` runs the visibility check alone.
It runs after any tier generation or DLI pass; its companion `.md` documents the checks.

`SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Calibration.py` and its
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Calibration.py` twin score how
well-calibrated a trained posterior is on the held-out EVAL set — simulation-based
calibration, expected coverage, TARP, and local C2ST (§7), overall and stratified by
each target parameter — over one shared engine
(`posterior_calibration_runner.run_posterior_calibration`), the same two-shim structure
the pipeline stages use, so one implementation serves both workflows and the entry-point
name carries the namespace. It reads only the estimator and the EVAL set, writes its
report to `Posit/`, is multi-GPU sharded with a `--merge` combine step, and is kept out
of the stage dispatcher; the companion `SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Calibration.md`
documents both workflows.

`SRM_AND_SBI_MONOMER_DIMER_ALP_Estimator_Comparison.py` and its
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Estimator_Comparison.py` twin decide whether one trained
estimator generalizes better than another by the paired log-score on the shared
`(task, sim)` TEST subset — pairing cancels each video's intrinsic entropy floor, so the
difference isolates the two estimators' KL gap (§7) — over one shared engine
(`estimator_comparison_runner.run_estimator_comparison`), the same two-shim structure.
It reads two `TestLossDistribution` artifacts, needs no GPU, writes its report to
`Posit/`, is kept out of the dispatcher, and is documented for both workflows in
`SRM_AND_SBI_MONOMER_DIMER_ALP_Estimator_Comparison.md`.

`SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py` reuses the trained DIMER-ALP
posterior — with no retraining — to MAP-estimate parameters from real recordings of two
oligomeric-state control receptors, the constitutive monomer CD86 and the constitutive dimer
CTLA-4 (BioImage Archive accession S-BIAD1369). A special-scope, ad-hoc reuse of the posterior
on a different study's data, it clones the
Experiment stage bar the dataset folder, output directory, and default conditions, and is kept
out of the stage dispatcher. Run it under the inference environment (single-process or
multi-GPU sharded, with a `--merge` pass to concatenate shards; `--dry-run` first); it writes a
per-condition inferred-parameter `report.md`, per-parameter figures, and the reusable
per-(cell, chunk) `.npz` arrays. Real data carry no ground truth, so the deliverable is a
per-condition distribution rather than a recovery check, with the diffusion scale as the
transferable quantitative read-out. Usage and interpretation are in the companions
`SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.md` and
`SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.md`.

### Configuration architecture

Configuration is split into two complementary parts:

- **Per-machine** — `machine_profiles.toml` carries the absolute paths and
  compute resources for each machine (script-bank root, data-bank root, optional
  scratch root, compute backend and device, worker count, running mode). It is
  kept out of version control; a committed `machine_profiles.example.toml`
  documents the schema. The active profile is selected by the `MACHINE_PROFILE`
  environment variable, and the configuration refuses to load — with a clear,
  remediating error — if the variable is unset, the profile is missing, a
  required key is absent, or a root directory does not exist. There is no silent
  fallback.
- **Project-wide** — the committed defaults and the parameter specification live
  in `parameterization.py` as frozen dataclasses. Freezing them makes the
  scientific defaults part of the reproducibility contract: they cannot be
  mutated at runtime. A user override (such as `--total-time-seconds 10.0`) is
  applied at the call site, not by mutating the global.

This separation decouples the code from machine-specific paths: a single
codebase runs on a local prototyping machine and on an HPC system with one
environment-variable switch, and absolute filesystem paths stay out of the
committed record.

---

## §5. Implementation Map (Science → Code)

Each scientific concept and pipeline stage maps to a specific module and
function in the package, driven by a thin entry-point shim over the stage's
shared runner, and produces a defined on-disk artifact. Module paths are relative to the package
`srm_and_sbi_monomer_dimer_alp/`; entry-point scripts live under `Script_Bank/Prime/`.

| Scientific concept / stage | Code (module → function/class) | On-disk artifact |
| --- | --- | --- |
| The separated stoichiometry–mobility model (§2): the two model blocks `StoichiometryBlock` / `MobilityBlock` and the derived six particle types; the channels generated per condition (seventeen under MET-INLB, eleven under MET-FAB); per-type diffusion; the association normalization and the per-condition association setting (`ConditionSetting` / `CONDITION_SETTINGS`); the realized initial composition and stationary initial modes | `parameterization.py` → `PARAMETERS.simulation.rds` (blocks, `conditions`, `particle_types`, `association_ratio_of()`), `realize_initial_composition()`; `simulation_rds_support.py` → `reaction_channels(theta, condition)`, `diffusion_coefficients()`, `association_reference_rate()`, `stationary_mode_law()`, `build_system(theta, condition)` (registers exactly the condition's generated channels); the ReaDDy simulation is then assembled by `build_simulation()` | (in-memory ReaDDy system/simulation; trajectory written below) |
| RDS trajectory recording (particle positions, particle type, time over the recording length) — one tier per condition (`--condition`; the sibling alias plus the condition token, no qualifier), shared by both workflows | entry point `SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py`, the only RDS entry point, run once per condition (drives `build_system()` → `build_simulation()`) | trajectory `.h5` (HDF5, ReaDDy convention); sampled theta set `.zarr` (via `io.py` → `save_theta_set()`) |
| Trajectory extraction (per-frame poses), the type-rank → molecular-species map the visibility layer selects by, and the subunit lineage (per-frame subunit-to-particle table replayed from the reaction records, subunit counts read per particle type) | `simulation_rds_support.py` → `extract_trajectory_poses()`, `collapse_species_axis()`, `rank_to_species()`, `monomer_ranks()`, `extract_subunit_lineage()` | (pose arrays and the lineage table passed to imaging) |
| Static labeling stoichiometry: the condition's dye-count law, the once-per-recording draw composed with the condition's declared probe occupancy (`--occupancy` overrides for a sensitivity run), the applied occupancy recorded | `labeling.py` → `LABELING_LAWS`, `resolve_labeling_law()`, `draw_dye_counts()`, `labeling_summary()`; `parameterization.py` → `occupancy_of()`, `occupancy_source_of()`, `visibility_of()` | `Labeling_Set` `.zarr` per task (one `LABELING_SET_COLUMNS` row per simulation; ten columns, `occupancy_monomer` / `occupancy_dimer` among them) |
| Diffraction-limited imaging forward model: dyes as emitters following their subunit's host particle, Gaussian PSF, per-dye brightness photo-physics with photobleaching, Poisson + EMCCD readout noise | `simulation_dli_support.py` → `render_dli_video()` (dye-centric, source-agnostic renderer of the poses, the lineage, the dye counts, and an assembled 11-key imaging vector; shared by both DLI stages), with `build_dye_tracks()` (per-dye emitter tracks), `Gaussian` / `sample_psf_width()` (PSF), `compute_intensity()` + `add_pixel_counts()` (intensity accumulation), `generate_brightness_photons()` (brightness photo-physics: stationary OU ln-brightness flicker + absorbing photobleaching), `EMCCD` / `add_noise()` / `generate_frames()` (detector noise) | (noised video array passed to writer below) |
| DLI video output (chunked, bit-depth-converted) | entry point `SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py` (per `--condition`; drives `extract_trajectory_poses()` + `extract_subunit_lineage()` → `draw_dye_counts()` → `render_dli_video()`, with the imaging block marginalized from the condition's `Nuisance_DLI` + the SCOPE box) | `.zarr` video set, shape `(frame_count, height, width)` (via `io.py` → `convert_video_dtype()`, `save_video_set()`) |
| Parameter prior and specification (the decided prior ranges, the per-condition settings — association ratio, occupancy — the per-row scale `LOG_FLAG`, units, labels; box-uniform prior in the estimator space) and the one conversion rule between estimator space and physical values | `parameterization.py` → `PARAMETERS` (a `Parameters` singleton) with `build_prior()`, `theta_lower_bound()`, `theta_upper_bound()`, `parameter_find()`, `to_physical()` / `to_flow()`, `prior_center()`; `ConditionSetting` / `CONDITION_SETTINGS` with `association_ratio_of()`, `occupancy_of()` | (configuration in code; sampled theta persisted in the RDS theta-set `.zarr`) |
| Prior-realization audit of generated products (read-only): the tier's `Theta_Set` against the prior box and the composition rule, its trajectories against the realized composition and the stationary law at frame 0, the `Labeling_Set` against the declared occupancies, the simulated visible counts against the deposited spot counts (Special_Analyses A9), and the theoretical visibility chain corroborated through the DLI stage's own labeling functions for both conditions | `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py` (checks P1–P6; `--selftest` on in-memory draws; `--visibility` for the visibility check alone) | report `.md` + `prior_realization_summary.json` under the `Posit` tier, per condition, split, and recording length |
| NPE + MAF estimator with 3D-CNN + temporal-transformer embedding | `inference_network.py` → `Complex3DCNN` (video encoder), `TemporalTransformer` (with `AttentionBlock`, `PositionalEncoding`); training wired in `inference_support.py` → `setup_training()`, `train_loop()` (with the resurrect branch) | (in-memory network; checkpoint + posterior written below) |
| Leak-proof TRAIN / TEST / EVAL split, sizing rule, and dataset construction | entry point `SRM_AND_SBI_MONOMER_DIMER_ALP_Generate_Datasets.py` (runs the RDS stage once per `--conditions` entry per split, then one DLI pass per `--workflows` entry over each condition's tier, with the `CORE = TRAIN + TEST`, `EVAL = max(floor, 0.1·CORE)` sizing; refuses to regenerate a condition's existing tier and checks the biology's `Nuisance_DLI` artifacts up front); dataset assembly in `inference_support.py` → `build_datasets()` (with `VideoDataset`, `normalize_video()`) | `_TRAIN` / `_TEST` / `_EVAL`-suffixed trajectory `.h5` and video `.zarr` namespaces |
| Posterior training run (gradient updates on TRAIN, selection on TEST) | entry point `SRM_AND_SBI_MONOMER_DIMER_ALP_Inference.py` (drives `build_datasets()` → `setup_training()` → `train_loop()`, then `artifacts.save_estimator()`) | version-portable estimator artifact (`Estimator.npz`, via `artifacts.py` → `save_estimator()`), loaded downstream as a `DirectPosterior`; network checkpoint at each new optimum |
| MAP recovery and calibration on held-out EVAL | `evaluation.py` → `map_estimate()` (seed-then-optimize: `collect_theta_prex()`, `collect_score_prex()`, `extract_elite_prex()`, `optimize_elite()`), `posterior_summary()` (quantiles + `sample_geometric_median()`), `point_estimate_agreement_table()`, `recovery_stats()`, `recovery_table()`, `posterior_coverage_table()`; driven by entry point `SRM_AND_SBI_MONOMER_DIMER_ALP_Evaluation.py` | recovery report (figures + tables + arrays + a live `progress.log`) under the validation output directory |
| Real-data application (no ground truth) | same `evaluation.py` estimator (`map_estimate()`, `experiment_table()`); driven by entry point `SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment.py` | per-condition inferred-parameter report under the validation output directory |
| Configuration, paths, storage routing, and file I/O | `parameterization.py` → `Paths`, `MachineProfile` / `load_machine_profile()`, `FrameConfig`, `RunTiming`; `io.py` → `load_data()`, `save_video_set()`, `save_theta_set()`, `convert_video_dtype()`, `theta_set_schema()`, `write_theta_set()`, `load_theta_set()`, `theta_set_status()` | resolved absolute paths (per-machine `machine_profiles.toml`); all artifacts above land under the configured roots |

The five pipeline stages are RDS, DLI, Inference, Evaluation, and Experiment.
Each stage has one shared runner (`<stage>_runner.py` → `run_<stage>()`). Every stage
past RDS has two thin Prime entry-point shims over it: the unqualified biology entry
point (`SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_DLI.py`,
`SRM_AND_SBI_MONOMER_DIMER_ALP_Inference.py`, `SRM_AND_SBI_MONOMER_DIMER_ALP_Evaluation.py`,
`SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment.py`) and its `_DETECTOR`-qualified detector
counterpart. The RDS stage has one shim only, `SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py`,
run once per condition, because each condition's trajectory tier is shared by both
workflows. Each shim parses arguments,
builds a `WorkflowConfig`, and calls the shared runner, which drives the package
functions above and writes outputs to the configuration-defined paths.
`SRM_AND_SBI_MONOMER_DIMER_ALP_Generate_Datasets.py` orchestrates the per-condition tiers
and the DLI passes of every requested workflow over each of them across all three splits
in one command.

---

## §6. Inference Network Architecture

The network learns an embedding of the video, then parameterizes a flexible
posterior over θ from that embedding.

**Video encoder (a 3D convolutional network):**
- Input: a `(batch, 1, n_frames, height, width)` video tensor.
- 3D convolutions over space and time extract hierarchical features (edges,
  textures, motion patterns).
- Output: a flattened feature map. The encoder accepts any duration — the input
  temporal depth is `n_frames`, derived from the recording length, and it asserts
  that the input frame count matches `n_frames` so a duration/data mismatch fails
  loudly.
- **Long videos are reduced in time before the transformer.** The first conv
  block strides time by `s = n_frames // temporal_target_frames` (default target
  **100 frames**), with its temporal kernel widened to `max(3, s)` so consecutive
  windows leave no gap — learnable pooling, not decimation. So the transformer
  sequence stays bounded no matter how long the recording is: 10 s (500 frames)
  is summarized over 100 positions, not 500. The target is a factor rather than an
  exact length, so 5 s (250 frames) reduces to **124**. The kernel is widened so
  the last window ends exactly on the last frame, so every input frame is read at
  every documented duration (asserted at construction). The exact arithmetic and
  a per-duration table are on the `temporal_target_frames` field in
  `parameterization.py`. At or below the
  target (1 s, 2 s at 50 FPS) the network is bit-identical to the un-reduced one,
  which is what keeps the 2 s baseline and its checkpoints comparable.

**Sequence model (a temporal transformer):**
- Input: temporal features from the encoder.
- Self-attention learns temporal dependencies (for example, particles moving
  consistently across early frames correlating with a high diffusion coefficient).
- A learnable CLS token is prepended; its output embedding is the summary used
  for posterior parameterization.
- Sinusoidal positional encoding accommodates the full frame range.

**Posterior parameterizer:**
- Input: the summary embedding from the transformer.
- Dense layers map the embedding to the parameters of a masked autoregressive
  flow, yielding an expressive, multimodal posterior rather than a single point
  estimate.

**Sampling:** Draw samples from the posterior parameterized by the embedding.

**Training:** Minimize the flow's negative log-likelihood on the simulated
(video, θ) pairs, with rotation and flip augmentation on the videos.

---

## §7. Validation and Diagnostics

### Semantic equivalence

The simulation and imaging stages thread explicit, per-function random-number
generators, so correctness is established **semantically** — confirming the code
produces the right scientific behavior — rather than by matching any particular
reference run element-by-element. Three pillars carry this:

1. **Theta-sampling determinism.** The prior sampler draws from the same fixed
   bounds with a seeded generator, so the same seed produces bit-identical theta
   vectors — directly verifiable.
2. **Reaction-diffusion primitive equivalence.** ReaDDy is a stochastic
   simulator (its stepper draws random numbers for diffusion and reaction
   events, so trajectories vary run-to-run), but the construction of its system
   specification (species, reactions, rates, box) is deterministic: the system
   and simulation builders construct exactly the declared model for a given
   theta.
3. **Imaging-pipeline functional equivalence.** The Gaussian PSF (erf-based
   pixel integration), the EMCCD model (Poisson photoelectrons, stochastic Gamma
   multiplication, Gaussian readout), the stationary brightness photo-physics,
   and the duration-independent photobleaching model
   produce videos of the correct shape, dtype, and value distribution.

### Reproducibility characteristics

Prior sampling is fully seeded, so theta sets are bit-reproducible at a fixed
seed. The reaction-diffusion stepper is not seedable, so trajectories — and the
videos and trained networks that depend on them — vary run-to-run by design,
matching the inherent stochasticity of experimental fluorescence data. The
default is non-deterministic (no seed); an explicit seed is available for
reproducible debugging and for the theta-only reproducibility regression test.

### Leak-proof split and MAP-recovery validation

A trained posterior is validated on data it has never seen, using the
three-namespace split (TRAIN / TEST / EVAL) described in §4. Because each split
is an independent draw with its own seed, the validation cannot be contaminated
by data the network optimized against.

**Simulated recovery (`Evaluation.py`).** For each held-out EVAL video the
maximum-a-posteriori parameter vector is estimated (seed-then-optimize: draw a
candidate pool, score it by the flow's log-probability, keep the top-K elite
seeds, then gradient-ascend the log-probability with Adam, a plateau
learning-rate schedule, and early stopping) and compared to the known ground
truth, giving two complementary read-outs:
- *Recovery accuracy* — how close the inferred parameter is to truth
  (per-parameter error; fraction within a tolerance band).
- *Posterior calibration* — whether the per-video credible intervals contain the
  truth at their nominal rate (roughly 50% for the interquartile range, roughly
  90% for the 5–95% interval). Under-coverage signals an overconfident
  posterior; over-coverage, an underconfident one.

**Real-data application (`Experiment.py`).** The same estimator is applied to
experimental microscopy videos, for which there is no ground truth. Each
recording is split into model-length windows and the inferred-parameter
distribution is reported per experimental condition, so treatment groups can be
compared. This is the scientific end use of the trained posterior, not a
correctness check. A two-mode candidate pool supports both regimes: a `bounded`
pool (rejection sampling within the prior; correct for a well-trained posterior)
and an `unrestricted` pool (sampling the flow directly, for smoke tests and
undertrained posteriors whose mass can lie outside the prior box).

Both stages report the **MAP point estimate** (the posterior mode) and the
**posterior credible summary** (median plus interquartile range) side by side,
because the two answer different questions: a sharp posterior at the wrong
location and a broad posterior at the right one are distinguishable only when
both are shown.

**Outcome for the MET-FAB detector (2 s, Poisson labeling).** The first production
calibration of the detector under the DOL-explicit observation layer is recorded, as
measurements with their definitions and limits, in `DETECTOR_WORKFLOW.md` §6.6: on
25,000 held-out synthetic videos the posterior median tracks the truth for the PSF
median and the brightness pair (correlation 0.89–0.96) but not for the PSF spread
(0.17), the joint 90 % credible region covers 62 % of truths, and the brightness
error grows monotonically with the realized number of dyes per labeled subunit. A
one-dye sensitivity run and the proposal that follows from these results are in
`DETECTOR_WORKFLOW.md` §6.6 and §9.4.

### Estimator generalization and test-loss interpretation

The Inference stage reports one per-epoch scalar, the **mean test loss** (the
flow's negative log-density on the held-out TEST set). That mean is a
*consistency* check, not a generalization metric, and reading it correctly
matters:

- **It confounds two quantities.** The logarithmic score is a strictly proper
  scoring rule whose associated divergence is the Kullback–Leibler divergence, so
  the expected loss decomposes into an intrinsic **posterior-entropy floor** (a
  property of the problem, not the estimator) plus the estimator's **KL error**
  (Gneiting & Raftery 2007). The absolute mean is therefore not "the estimator's
  error," and two runs reproducing the same mean at different TEST-set sizes
  confirm only that the mean is well estimated (a 1/√N consistency result), not
  that the estimator generalizes.
- **A lower loss does not certify a better posterior.** Closeness in KL bounds
  neither moments nor calibration (Deshpande et al. 2022); posterior faithfulness
  is a separate measurement (below).
- **The central-limit reading is contingent.** The per-example negative
  log-density is unbounded above — one example where the flow assigns near-zero
  density at the truth contributes an arbitrarily large value — so if that upper
  tail is heavy the variance is large or undefined and the Gaussian standard error
  breaks down. Finite variance is *testable*, not assumed.

**Best-epoch test-loss distribution (instrumented).** Rather than the mean alone,
the stage captures the best epoch's **per-example** TEST loss, keyed by the
`(task_index, sim_index)` pair — authoritative against the on-disk task/sim
layout and extension-stable as the TEST set grows — with a self-describing
manifest (parameter table, prior bounds, θ-space, best epoch and loss). It is
committed alongside the checkpoint and posterior at each new best; because the
per-example loss is already computed to form the mean, capturing it is a
no-reduction store and essentially free (`--test-loss-distribution`, on by
default when a TEST set is present). Reporting is three-tier: the per-epoch mean
and standard deviation (a spread band on the loss curve); at each new best an
extended card (quantiles, skew, tail mass, train−test gap, bootstrap confidence
interval); and heavier analyses post-hoc. **Model selection stays by the mean** —
the extended statistics are diagnostic, since re-ranking checkpoints on an
outcome-selected tail subset would be an improper score (Gneiting & Ranjan 2011).

**Rigorous cross-run comparison (post-hoc).** To decide whether estimator A
generalizes better than B, form the per-example log-score difference on the
**shared** `(task_index, sim_index)` subset and test that its mean is zero.
Pairing cancels the common entropy term, so the statistic isolates the difference
of the two estimators' KL divergences to the truth (Amisano & Giacomini 2007) —
the Diebold–Mariano test of equal predictive accuracy (Diebold & Mariano 1995),
with the Wilcoxon signed-rank test and the paired bootstrap as heavy-tail-robust
alternatives. Comparing the two *means* of two different-sized sets is invalid: it
confounds the sets' intrinsic-entropy content. Whether the mean itself is
trustworthy is checkable from the stored scores (a subsampling-rate log-log slope
near −0.5; tail-index estimation; bootstrap skew); a heavy tail is itself the
generalization red flag the mean concealed. The `Estimator_Comparison` Analysis
diagnostic implements this over one shared, workflow-agnostic engine: it reads two
`TestLossDistribution` artifacts, pairs them on the shared `(task, sim)` subset, and
reports the Diebold-Mariano statistic with the Wilcoxon signed-rank test and the
paired-bootstrap interval as its heavy-tail-robust companions, for both the biology and
the detector estimators.

**Calibration and coverage.** A low loss cannot detect an overconfident
posterior, so faithfulness — the coverage read-out introduced above — is measured
by the field-standard amortized-inference diagnostics: simulation-based
calibration (rank uniformity of the true θ within posterior samples; Talts et al.
2018, with the necessary-not-sufficient caveat of Modrák et al. 2023), expected
coverage of credible regions (Hermans et al. 2022), TARP (necessary and
sufficient, in the population limit, for matching the true posterior; Lemos et al.
2023), and local C2ST for per-observation fidelity on the few real observations
(Linhart et al. 2023). These require posterior sampling and are therefore
periodic/post-hoc rather than per-epoch. The `Posterior_Calibration` Analysis
diagnostic implements all four over one shared, workflow-agnostic engine — reported
overall and stratified along each target parameter by the posterior's **inferred**
value (conditioning on the observation, since the rank is uniform conditional on it;
binning on the latent truth would confound Bayesian shrinkage with miscalibration). A
subregion where calibration degrades — the low-count regime for biology, an imaging
setting for detector — is thereby localized rather than averaged away, for both the
biology and the detector posteriors.

**Scope.** All of the above measure **in-distribution** generalization — to new
draws from the same prior-predictive. Out-of-distribution / real-data
generalization is a different question, answered by coverage under model
misspecification, the embedding-space experimental-versus-synthetic distance (MMD / C2ST,
implemented by the workflow-agnostic `embedding_space_distance` kernel and its shared
runner; the biology companion note is
`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Embedding_Space_Distance.md`), and
posterior-predictive checks; misspecification-robust simulation-based inference
(Ward et al. 2022; Kelly et al. 2023 — pending independent verification) is the
relevant literature.

**References.** Gneiting & Raftery (2007), *Strictly Proper Scoring Rules,
Prediction, and Estimation*, JASA; Deshpande et al. (2022), *Are you using test
log-likelihood correctly?*, TMLR; Amisano & Giacomini (2007), *Comparing Density
Forecasts via Weighted Likelihood Ratio Tests*, JBES; Diebold & Mariano (1995),
*Comparing Predictive Accuracy*, JBES; Gneiting & Ranjan (2011), on the
impropriety of non-constant score weighting, JBES; Talts et al. (2018),
*Validating Bayesian Inference Algorithms with Simulation-Based Calibration* (with
Modrák et al. 2023); Hermans et al. (2022), *A Trust Crisis in Simulation-Based
Inference?*, TMLR; Lemos et al. (2023), *Sampling-Based Accuracy Testing of
Posterior Estimators (TARP)*, ICML; Linhart et al. (2023), *L-C2ST*, NeurIPS (on
the classifier two-sample test of Lopez-Paz & Oquab 2017); and, pending
independent verification, Ward et al. (2022) and Kelly et al. (2023) on
misspecification-robust neural posterior estimation.

### Matched-imaging embedding validation (planned)

The embedding-space experimental-versus-synthetic distance introduced above
(detailed in `DETECTOR_WORKFLOW.md`) compares the experimental recordings against
the held-out EVAL set, which spans the entire imaging prior. Because the
experimental recordings sit at a single imaging setting while the synthetic
reference is broad, the experimental embeddings form a tight cluster nested inside
the diffuse synthetic cloud, and a two-sample classifier separates the two by
concentration alone — even where their supports overlap. That separation is a
breadth artifact of a prior-spanning reference, not evidence that the experimental
recordings lie off the synthetic manifold. Isolating imaging realism at the
calibrated operating point requires a synthetic reference generated at the
inferred imaging, not across the prior.

A planned analysis supplies it: render synthetic videos with the imaging fixed to
the inferred values and the receptor counts and diffusion pinned to the
experimental data, then measure the same embedding distance against the
experimental recordings. The obstacle is that the Detector marginalizes the
reaction-diffusion block — particle counts and diffusion coefficients — so it
never estimates them for any single recording; they are learned only implicitly. A
matched render must therefore obtain them from outside the Detector.

Until the RDS estimator exists, that external source is the single-molecule
localization data (accession S-BSST712). Per-condition receptor counts come from
the localization density, corrected upward for the emitter on-fraction: the
localizations per frame undercount the receptors, both because dim or overlapping
emitters are missed and because a fraction of emitters are dark or bleached at any
instant. The bleached fraction follows from the calibrated photobleaching
probability through the fixed hundred-frame survival law; the blinking rate enters
only to second order, since at the calibrated brightness law only of order one
percent of emitter-frames falls below the localizer's photon-acceptance floor,
so flicker barely manufactures apparent darkness. Diffusion coefficients come from the tracked trajectories by
mean-squared displacement (`MSD(τ) = 4·D·τ` for two-dimensional free diffusion,
reported in µm²/s). Once the RDS estimator is trained it supplies the counts and
diffusion directly, closing the loop without the localization step.

The analysis reuses the existing machinery end to end — the reaction-diffusion
simulation at the pinned counts and diffusion, the imaging renderer driven by one
fixed inferred-imaging vector, and the embedding together with the
maximum-mean-discrepancy and classifier two-sample statistics of the
embedding-distance analysis — replacing only the synthetic reference, from
prior-spanning to a single operating point. The interpretation shifts accordingly:
a residual distance then measures model-versus-data mismatch at the calibrated
imaging, rather than the prior-averaged realism the broad reference reports. Like
the flicker-rate derivation drawn from the same localization data, it is a
read-only post-hoc utility rather than a pipeline stage, and gains its own
companion note when it ships.

### Observability and diagnostics

Every stage shares one diagnostics scheme. `--debug` prints per-step
checkpoints, fail-loud invariant checks, and an end-of-stage summary; a dump mode
additionally persists a self-contained report (figures plus a console
transcript) under a diagnostics workbench directory, kept separate from the
scientific deliverables. The validation stages also write a live, tail-able
progress log so a long run can be monitored. The diagnostics are off by default,
and each check is a cheap no-op when off.

### Posterior geometry

Useful diagnostics to compute when analyzing a trained posterior:
- Posterior mean and covariance.
- Marginal distributions (1D histograms per parameter).
- Bivariate correlations (2D scatter plots, especially for confounded
  parameters).
- Posterior-predictive check — implemented as the `Posterior_Predictive_Video`
  analysis (biology and detector Analysis shims over
  `posterior_predictive_video_runner`): render a synthetic video at the
  parameters inferred from one real recording and compare the two side by side.

---

## §8. Open Scientific Questions

*(Scientific and methodological questions only.)*

**S1. Identifiability of RDS parameters.** Are all parameters in θ uniquely
identifiable from video data? Which parameters are confounded (co-determined)?
This calls for theoretical or empirical analysis.

**S2. Prior sensitivity.** How sensitive is the posterior to the prior choice
(uniform versus log-normal)? Should informative priors from the biophysical
literature be used?

**S3. Posterior convergence across restart cycles.** The training loop now performs
warm-restart cycles automatically within a single run (and `--resurrect` chains them
across runs); how many cycles are needed for posterior stability, and is there a
principled stopping criterion beyond an empirical plateau (for example, a test-loss
delta below tolerance, or a restart that fails to improve the best)?

**S4. Out-of-distribution robustness.** If the network is trained at one
recording length and tested at a different one (for example, trained on 2 s
videos and tested on 10 s), how does posterior accuracy degrade? Can it be
trained jointly on multiple durations?

**S5. Multi-cell heterogeneity.** Real microscopy data contains cells with
varying expression levels, spatial organization, and cell-cycle phase. Can the
single-cell, per-video model extend to population posteriors?

**S6. Which imaging parameters the detector should infer.** Under the Poisson labeling
law two of the six inferred imaging parameters are not recovered from 2 s recordings
and the joint posterior is too narrow (`DETECTOR_WORKFLOW.md` §6.6). Which of the six
are better supplied from acquisition metadata or from direct, non-neural estimators on
the raw frames, with explicit uncertainty, and which must remain coupled inference
targets? The proposal and its adoption gates are `DETECTOR_WORKFLOW.md` §9.4.

---

**End of Project Context — srm-and-sbi-monomer-dimer-alp**
