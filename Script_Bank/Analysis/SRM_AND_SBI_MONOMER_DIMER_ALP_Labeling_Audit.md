# Labeling audit — companion note

`SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit.py` verifies that the DOL-explicit observation
layer does what the documents say it does. It is a post-hoc analysis, never wired into the
stage dispatcher, and it needs no trained estimator and no recordings: it simulates one small
reactive trajectory at the biology prior center, draws labelings on it, and renders static
scenes through the production renderer.

## What it checks, and why each check exists

| level | object | what could be wrong without it |
|---|---|---|
| 1 | the registered labeling laws (`labeling.LABELING_LAWS`) | a law whose sampler and analytic moments disagree would make the documented visible fractions false in the data while true on paper |
| 2 | the subunit lineage replayed from the reaction records | a replay that skipped a record or mis-assigned a daughter would attach a dye count to the wrong receptor for the rest of the recording; the conservation check inside the extractor is exercised here on a real reactive trajectory and cross-checked against the particles observable |
| 3 | repeated static draws on that lineage | the derived consequences the design rests on — the visible fractions per species and the two-thirds one-dye share among visible MET-INLB dimers — must emerge from the draw with no further assumption |
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
