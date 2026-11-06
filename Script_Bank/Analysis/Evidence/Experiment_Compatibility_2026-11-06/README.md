# Evidence: posterior compatibility of SGM representatives and the `posterior_rank_median` member weighting (0.1.47)

Implementation record of the brief `Special_Analyses/EXPERIMENT_COMPATIBILITY_AND_NUISANCE_WEIGHTS_2026-11-06/CLAUDE_IMPLEMENTATION_PLAN.md`
(2026-11-06) and of the corrections requested by Codex's review (`Special_Analyses/EXPERIMENT_COMPATIBILITY_REVIEW_2026-11-06`). Start with [COMPLETION_NOTE.md](COMPLETION_NOTE.md), §8 for the corrections.

| File | Purpose |
|---|---|
| `COMPLETION_NOTE.md` | the indexed completion note: changed files, deviations, report paths, tests, oracle comparison, source preservation, costs, the illustrative build, open scientific choices |
| `source_hashes_before.txt`, `source_hashes_after.txt` | SHA-256 of the 136 numerical source products (both adopted Experiment products, both COHORT60PLOTS archives and manifests, the 120 retained draw collections, both SEL artifacts, the estimators) before and after the work; identical |
| `cloud_sources_FAB.txt`, `cloud_sources_INLB.txt` | the comma-joined 60 collection paths per condition, taken from the COHORT60PLOTS manifests |
| `regenerate_<COND>_COHORT60RANKS.{log,time}`, `regenerate_<COND>_COHORT60RANKSR2.{log,time}` | the CPU regenerations of both complete-cohort reports: the first checkpoint (`COHORT60RANKS`) and, after Codex's review corrections, `COHORT60RANKSR2` (console logs; `/usr/bin/time -v` wall time and peak RSS) |
| `codex_review_probes.py`, `codex_review_probes.log`, `review_probes.json`, `probe_archives/`, `zero_component_*/`, `override_collisions/`, `all_zero_scores/` | Codex's review probes (`EXPERIMENT_COMPATIBILITY_REVIEW_2026-11-06/review_probes.py`, copied unchanged) rerun from this folder after the corrections: the three inconsistent archives refused, both zero-mass cases reported, every override-collision row with its provenance, the all-zero policy described as not normalizable |
| `codex_identity_regression_probes.py`, `codex_identity_regression_probes.log`, `identity_regression_reproduction.json`, `duplicate_cell_identity.npz`, `duplicate_chunk_identity.npz` | Codex's re-review probe (`REREVIEW_R2/identity_regression_probes.py`, copied unchanged) rerun from this folder after the identity-guard follow-up: both mutated archives refused |
| `first_pass_logs/` | the first regeneration pass (identical numbers); its review copies were replaced after a figure-layout fix (the colorbar labels crossed the score strip) |
| `oracle_comparison.py`, `oracle_comparison.json`, `oracle_comparison.log` | the full-data oracle: the regenerated rank matrices, scores and normalized probabilities against the independent review (`Special_Analyses/POSTERIOR_RANK_WEIGHTING_REVIEW_2026-11-06`), the named facts of the brief, and the identity of the candidate arrays with COHORT60PLOTS |
| `illustrative_build.py`, `illustrative_build.log`, `illustrative_build/` | the illustrative rank-weighted constructor build (MET-FAB, all ten windows, one SGM per cell, `posterior_rank_median`) against the R2 archive, written into this folder under the tag `ILLUSTRATIVERANK`: spec, artifact, report, figures, member and row CSV exports; not a production selection |
| `review_copies/` | complete copies of the two corrected review reports (`..._MAP_Experiment_COHORT60RANKSR2/`: report, archive, CSV exports, manifest, figures, atlas) |
| `full_suite.log` | the complete repository test suite at this checkpoint |
| `SHA256SUMS` | integrity of this package |

Nothing here is a production selection: the acquisition period, the construction or mixture, the component masses,
whether to enable consensus weighting and any parameter-specific treatment remain open scientific choices.
