# Control-receptor temporal-dynamics analysis — interpretation

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.py`.
The script applies the DIMER-ALP posterior — trained on the MET single-particle-tracking
regime — to two control receptors and tracks each inferred parameter over their
recordings; this note explains what the figures mean, how to read them, and why only
the diffusion readout is interpreted quantitatively. It is written so the analysis can
be understood and reused without reverse-engineering the code.

This is a special-scope, ad-hoc reuse of a trained posterior on data from a different
study. It is not one of
the canonical pipeline stages and is kept out of the stage dispatcher.

## The parameter table

`--condition` names the MET estimator the control Experiment run applied (MET-FAB or
MET-INLB), and that condition binds the parameter table: eleven learnable rows under
MET-FAB, twelve under MET-INLB, whose twelfth row is the window-start allocation
R_active. The run's product is read only if its stored `parameter_keys` equal the
condition's, key for key and in order; the control kinds CD86 / CTLA-4 never select a
table, so every value here is read under the estimator's model. Values are converted to
physical units by the one conversion rule (`parameterization.to_physical`): log rows are
exponentiated, the linear rows — the basal reference fraction f and, under MET-INLB,
R_active — are the fractions themselves.

## The controls, and why diffusion is the readout

CD86 is a constitutive monomer and CTLA-4 a constitutive dimer. They are
oligomeric-state controls: a known-monomer and a known-dimer whose mobile diffusion
coefficients have been measured independently, so they **bracket the mobile-diffusion
scale** (a monomer diffuses faster than a dimer). Applying the posterior to them tests
whether its transferable observable — the diffusion coefficient — lands on the measured
values.

Only the diffusion coefficient is read quantitatively, because of a deliberate
**label / model mismatch**: the posterior was trained on always-visible permanent-label
emitters, whereas the control recordings use an exchangeable, blinking SiR-S5 HaloTag
probe. The diffusion coefficient is a per-track property, read from the displacement
statistics of a molecule *while it is visible*, so it is insensitive to how many
molecules are lit at once and transfers across the mismatch. Counts and rates depend on
the number of co-visible emitters and on track continuity — both corrupted by blinking —
so they are not interpreted here. The controls are also constitutive states, not the
dynamic A + A ⇌ B dimerization mechanism with mobility switching the posterior encodes; they bracket the
diffusion scale and stress-test transferability, they do not exercise the kinetic model.

## The headline figure — mobile mixture diffusivity

`dmix_mobile_temporal.png`. The posterior cannot reliably separate monomer from dimer
for these constitutive controls, so the headline observable is the number-weighted mean
diffusivity of the **mobile** population: both molecular species (monomer A, dimer B) in
the mobile modes (fast and slow), excluding the immobile mode, which is a resolution
floor rather than motion:

    D_mix_mobile = (n_A · d_A + n_B · d_B) / (n_A + n_B)     [µm²/s]

Here n_A and n_B are the underlying monomers and dimers at the window start, computed per
window by the population helper the generator places with
(`parameterization.population_from_theta`) under the estimator's condition: the inferred
receptor total N and basal reference fraction f give the reference population of
M_ref = N (1 − f) / (1 + f) monomers and D_ref = N f / (1 + f) dimers, and the window-start
allocation (the inferred R_active under a MET-INLB estimator, the declared 0 under
MET-FAB) turns c probe-bound monomer pairs into dimers, so n_A = M_ref − 2c and
n_B = D_ref + c. d_A and d_B are the monomer and dimer diffusivities averaged over the
mobile modes with the stationary occupancies of the inferred switching chain (the
initial-mode law every particle is seeded with), from the generator's own coefficients
D[X, m] = D_A · (R_B if X = B) · (1, R_s)[m], so d_B = R_B · d_A; because the switching
rates are shared by both species, the mode weights cancel in the species split. The
figure is a clean per-receptor comparison — every line is solid, and the four series are
told apart by color alone:

- **inferred D_mix_mobile**, in the condition color (CD86 blue, CTLA-4 orange) — a
  trajectory (mean over cells) with a mean ± 1 SD between-cell band;
- **experimental D_mobile**, in a separate per-condition color (CD86 green, CTLA-4 red)
  — a flat reference line with a value ± SD band. Its own colors keep experiment and
  inference from ever sharing a color, while still telling the two references apart
  (there are two here, one per receptor, unlike the single MET reference).

d_A, d_B, the reference fraction f and the starting dimer-complex split
f_start = n_B / (n_A + n_B) are reported in the run's report table, not drawn on the
figure, so the headline stays an uncluttered inferred-vs-experiment comparison;
f_start = f whenever no pair is allocated, which is every window under a MET-FAB
estimator.

D_mix_mobile lies between d_A and d_B, in either order. On these out-of-distribution
controls the model's monomer / dimer (A / B) assignment is not physically constrained —
the relative dimer diffusivity R_B can exceed 1 (so d_B > d_A), and under unrestricted
pooling the mode can even leave the training prior (the report flags any condition where
D_A or R_B does so). D_mix is therefore read as a point between d_A and d_B, **not** as a
resolved monomer / dimer mixture, unless the split f_start has an informative (non-edge)
posterior — the split comes from the composition rows (N, f and the allocation), which
are the label-fragile quantities, so a collapse of f_start toward 0 or 1 pushes D_mix
toward one endpoint. Reporting d_A, d_B, f and f_start alongside D_mix keeps that
dependence in view, and this robustness to the A / B mis-assignment is exactly why D_mix
(not d_A or d_B alone) is the readout. The per-run values are tabulated in `report.md`.

**Flagged windows.** Unrestricted pooling also lets a window's composition rows leave
their box. A window that fails the helper's validity rule (f, and R_active under a
MET-INLB estimator, finite and in [0, 1]; N finite and positive) is flagged: it is left
out of every quantity of the mixture readout, so each number of a report row comes from
the same windows, it is counted per receptor in the report ("flagged windows: k of n"),
and its values are never clipped — the per-parameter figures show them as estimated.

## Units and why the comparison is quantitative

D_A is stored directly in µm²/s (the posterior samples log10(D_A); the absolute value
is 10^θ), and D_B inherits those units through the dimensionless ratio R_B — so
D_mix_mobile is in µm²/s and is directly comparable to the experimental D_mobile with
**no unit conversion**. The comparison is quantitative because the acquisition geometry
matches the training regime: 50 FPS, 256 × 256 pixels, ~157 nm/px. The comparison is
between **ensemble means**: D_mix_mobile is the abundance-weighted mean of a
two-component mobile distribution, and the experimental D_mobile is a single
mobile-fraction coefficient pooled over the mobile tracks; a bimodal mobile population
has a mean between its modes, matching neither individually.

## The per-parameter figures — two readings at once

The script also emits one `<key>_temporal.png` per learnable parameter of the
estimator's table (the receptor total, the composition fractions, rates, ratios), with the
same solid / band / faint-line grammar. A linear row's axis follows its band below zero,
so a fraction estimated outside [0, 1] stays in view. These are diagnostic, not
quantitative comparisons (no experimental band is drawn on them):

**1. Temporal dynamics.** Estimating a parameter in each short window and plotting it
against the window's position in the recording shows how the inferred value behaves over
time — something a single whole-recording estimate cannot reveal.

**2. Robustness / stationarity.** A parameter that is a constant property of the system
should be flat over time; a flat trajectory is positive evidence that the estimator is
time-invariant and self-consistent. A systematic trend means either genuine dynamics or
an acquisition confound. For the receptor total in particular, a downward drift over the
recording is the expected signature of blinking / photobleaching (fewer fluorescent
emitters in later windows) — a real non-stationarity in a confounded parameter, which is
why the composition is not read quantitatively.

## Why this exceeds a single whole-recording readout

A whole-recording measurement yields one estimate plus a distribution for the entire
recording; it cannot resolve a parameter in time or per cell. This inference resolves it
per short window **and** per cell. Averaging the per-window MAP estimates over time and
cells recovers the comparable whole-recording point estimate — the D_mix_mobile
time-average — now backed by the demonstrated (or refuted) stationarity of the
underlying diffusivities rather than assumed.

## Reliability: recovery × stationarity

Recovery quality is a property of the posterior, independent of which real data it is
applied to, so it carries over to this reuse. Each figure is annotated (when the
Evaluation MAP-recovery arrays of the same condition are present; their `parameter_keys`
must equal the table's) with the fraction of held-out EVAL videos recovered within
±0.3 dex. The band is a multiplicative tolerance and exists for log rows only; a linear
row is annotated as not applicable. The monomer diffusivity D_A recovers well and is the
backbone of the headline readout; a parameter that recovers poorly carries no signal,
and its trajectory must not be over-read. Trust a
parameter when it both recovers well on ground truth and is stationary where it should
be.

## Caveats

- **Label / model mismatch.** Exchangeable, blinking SiR-S5 vs the always-visible
  permanent-label training regime. Diffusion transfers; the composition and the rates
  (N, f, the allocation under MET-INLB, κ_OFF, the switching rates) do not and are not
  read quantitatively.
- **Not the trained mechanism.** CD86 and CTLA-4 are constitutive monomer / dimer
  controls, not the dynamic A + A ⇌ B mechanism with mobility switching the posterior encodes. They bracket
  the diffusion scale and stress-test transferability; they do not exercise the kinetics.
- **D_mix weights are the fragile quantity.** D_mix is number-weighted, and the
  composition (N, f and the allocation) is label-fragile; read D_mix between d_A and d_B
  (in either order), not as a resolved mixture, unless f_start is informative.
- **Prior leakage under unrestricted pooling.** On these out-of-distribution controls
  the MAP can leave the training prior — R_B above 1, or D_A past its ceiling — because
  unrestricted pooling does not enforce the prior box. The report flags any such
  condition. D_A / D_B are then not physically constrained, but the count-weighted D_mix,
  which does not depend on the A / B ordering, stays the robust readout.
- **Blinking / photobleaching** drives the downward drift of the receptor total over
  the recording — a real non-stationarity in a confounded parameter, not receptor loss.
- **First-pass posteriors.** The current 2 s / 5 s posteriors come from interrupted
  training; the absolute values will sharpen with the full production posteriors. Re-run
  this analysis on those for the definitive numbers.
- **Relative parameters** (R_B, R_s, R_i) are plotted as dimensionless ratios. The
  composition rows are linear fractions on [0, 1]: the basal reference fraction f (the
  share of dimers among the complexes of the reference population) and, under a MET-INLB
  estimator, the window-start allocation R_active (the share of the possible probe-bound
  monomer pairs turned into dimers at the window start). The association ratio is a
  declared per-condition constant, not a parameter, so no figure is emitted for it.

## Not yet implemented: aggregated posterior distributions

The historical figures also included, per parameter, a pooled **posterior distribution**
panel: one histogram per condition of the full posterior sample cloud pooled across every
cell and every chunk. Reproducing it faithfully needs the per-window posterior
**samples**, aggregated across all windows into a single distribution per (condition,
parameter). The current Experiment stage persists only five quantiles per window, not the
sample pool, so the quantiles alone cannot be re-pooled into that experiment-wide
distribution. To add it: draw posterior samples per (cell, chunk) window from the trained
posterior, concatenate across all windows of a condition, and histogram the pooled samples
in estimator coordinates. This requires either persisting the per-window sample pool in the Experiment
stage or a dedicated resampling pass, and is left as a documented extension.

## Reference

Catapano et al., "Long-Term Single-Molecule Tracking in Living Cells using Weak-Affinity
Protein Labeling," *Angew. Chem. Int. Ed.* **2025**, 64, e202413117.
doi:10.1002/anie.202413117. Data: BioImage Archive accession S-BIAD1369 (CD86 and
CTLA-4, SiR-S5 / HaloTag single-color tracking; the experimental mobile-fraction
diffusion coefficients are D_mobile = 0.319 ± 0.010 µm²/s for CD86 and 0.279 ± 0.005
µm²/s for CTLA-4).
