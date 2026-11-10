# Completion note: the production Nuisance DLI of the biology workflow (2026-11-10)

Written for the lead and the reviewer. The lead adopted the reviewed population-wide selection on 2026-11-10 as the
production `Nuisance_DLI` of the biology workflow, without a tag: it is the canonical artifact of each condition, and
the biology generation is launched without `--nuisance-tag`. Decision record: `DETECTOR_WORKFLOW.md` section 7.6.

## Built

With the committed release 0.1.49 (`7c977a8`), on the PC, CPU only, through the constructor's command line:

1. `write_production_specs.py` wrote `<alias>_2S_50FPS_Nuisance_DLI_Spec.toml` into `Posit` from the reviewed proposal
   specifications (`PROPOSALCW4`): components, comparison scope, rank coordinates, candidate archive and checksum, and
   both override tables verbatim (asserted); only the header and the status sentences of the rationale differ.
2. `--condition <COND> --total-time-seconds 2.0 --build --dry-run` (`cli_dry_run_<COND>.out`), then `--build`
   (`cli_build_<COND>.out`).
3. `verify_production.py` (`verification_production.json`): the artifacts load through the biology gate
   `require_nuisance_dli(<Posit>, <alias>, "2S_50FPS")` with no tag; vectors, probabilities, member identities, scores,
   archive and overrides are identical to the reviewed proposal; 20,000 sampled vectors are all stored rows and no
   zero-probability row is drawn.

| Condition | Artifact sha256 | Spec | Archive | Rows | Effective rows | Zero rows | Bleaching | Flicker (1/s) | Outside prior | Largest sampling deviation |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| FAB | `2a239febdab93cbbac3a2975f4f556e482e25cc480e02471d92d40d49d134e7e` | `ae1598eea1c310a3…` | `aa5a668bc86030df…` | 600 | 427.30 | 1 | 0.05 | 5.1 | 12.3 % | 0.0012 |
| INLB | `667b3b2b6b58b4031b91754d527eb5a2123eba10f906bb14f00ea0f126d2f2aa` | `4d66ce0e9b45cd29…` | `179617f2665504a9…` | 600 | 317.97 | 64 | 0.05 | 4.7 | 96.9 % | 0.0013 |

Selection: one SGM per cell and 2 s window over all ten windows; `posterior_rank_median`, `rank_scope = "construction"`
(every representative against the other 599 of its condition, normalized over all 600, no reserved cell mass), rank
space `spatial_brightness` (`mu_r`, `sigma_r`, `mu_pc`, `sigma_pc`); `prob_photo_bleach` 0.05; `lambda_rate` 5.1 (FAB)
and 4.7 (INLB).

## Unchanged

`RANK4` and `SEL` artifacts, specifications and reports; the `COHORT60RANKSR3` and `COHORT60RANKSR4` archives.

## Files

`production_copies/` (the two specifications, artifacts and report folders as built), the command-line outputs, the
two scripts, `verification_production.json`, `SHA256SUMS`. The implementation and its review are in
`../Comparison_Scope_2026-11-10/`.
