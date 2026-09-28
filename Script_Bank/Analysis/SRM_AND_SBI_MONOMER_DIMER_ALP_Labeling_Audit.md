# Labeling audit — companion note

`SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit.py` verifies that the DOL-explicit observation
layer does what the documents say it does. It is a post-hoc analysis, never wired into the
stage dispatcher, and it needs no trained estimator and no recordings: it simulates one small
reactive trajectory at the biology prior center, draws labelings on it, and renders static
scenes through the production renderer.

Probe occupancy is a declared per-condition input of the visibility layer (`parameterization.ConditionSetting`:
MET-INLB 0.5 declared; MET-FAB 0.155, derived from the declared Fab/InlB visibility ratio 0.5 and the InlB
anchor). Every renderer of simulated trajectories applies it through one labeling path
(`labeling.resolve_labeling`, `labeling.label_trajectory`). The audit keeps the two apart. Level 3 draws the
bare labeling laws at occupancy 1, so the visible fractions and the two-thirds one-dye share among visible
MET-INLB dimers it checks are properties of the laws themselves. Level 3b draws the declared occupancy through
the shared path, the visibility chain of the training data: visibility per subunit 0.25 (MET-INLB) and 0.125
(MET-FAB), and a share of visible dimers with both subunits labeled of 14 % and 6.7 %; for MET-INLB that share
is also the two-dye share.

## What it checks, and why each check exists

| level | object | what could be wrong without it |
|---|---|---|
| 1 | the registered labeling laws (`labeling.LABELING_LAWS`) | a law whose sampler and analytic moments disagree would make the documented visible fractions false in the data while true on paper |
| 2 | the subunit lineage replayed from the reaction records | a replay that skipped a record or mis-assigned a daughter would attach a dye count to the wrong receptor for the rest of the recording; the conservation check inside the extractor is exercised here on a real reactive trajectory and cross-checked against the particles observable |
| 3 | repeated static draws of the bare law (occupancy 1) on that lineage | the derived consequences of the law — the visible fractions per species and the two-thirds one-dye share among visible MET-INLB dimers — must emerge from the draw with no further assumption |
| 3b | repeated draws at each condition's declared occupancy, through the production labeling path | the visibility chain every training video and every check render carries — visible monomers a, visible dimers 1 − (1 − a)², the both-labeled share a / (2 − a), dyes per subunit p_occ × E[dye], and the recorded occupancy — must emerge from the call the renderers make, not from a re-derivation of it |
| 4 | the renderer on static scenes | a zero-dye subunit must render nothing, two dyes must carry twice the photons of one, and at a dissociation the dye must follow its own subunit; each is a way a renderer could silently reintroduce complete labeling or a brightness multiplier |

The acceptance tolerances are prespecified in the script header and are practical, not
significance thresholds: the render-level checks read a lognormal OU brightness through
9x9 apertures over a few hundred frames, so the two-dye ratio is accepted within a band
around 2 rather than at 2.

## Reading the report

The one-dye dimer of the first render scene carries the same signal as a one-dye monomer.
That is the labeling model's first derived consequence: a dimer whose partner subunit is
unlabeled is not brighter than a monomer, so the visible-dimer brightness is a mixture of
one-dye and two-dye emission (two thirds to one third under the MET-INLB law), not a doubled
monomer. The lineage's particle-id churn — one subunit visiting many ReaDDy ids over two
seconds, because every reaction product, conversions included, receives a fresh id — is the
fact that makes the lineage necessary and is the reason a per-particle emitter would restart
its brightness process at every mobility conversion.

## Usage

```bash
MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit.py
```

Outputs are data and land in the Data_Bank Posit tier, never in the codebase:
`<data_bank_root>/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit/` holds the report
(`SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit.md`) and `audit_summary.json` with every number
the report quotes. The audit is not tied to one condition: the lineage is simulated under the
MET-INLB reaction network (association switched on, so every channel is exercised), and both
conditions' baseline laws are drawn on that same lineage.
