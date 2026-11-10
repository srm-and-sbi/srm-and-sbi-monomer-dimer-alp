# Evidence: construction-wide comparison scope and exact marginals (2026-11-10, release 0.1.49)

| Item | Content |
|---|---|
| `COMPLETION_NOTE.md` | the indexed completion note: changes, tests, report paths, scope provenance, reproduction, unresolved items |
| `REGENERATION_RECORD.md`, `regenerate_<COND>_COHORT60RANKSR4.command` | the CPU regenerations of both complete-cohort Experiment reports |
| `review_copies/` | the two regenerated reports (report, format-4 archive, group and comparison CSV, manifest, figures, atlases; the two pairwise CSV files of 25 MB each stay in the data bank) |
| `data_bank_checksums.sha256` | SHA-256 of the archives and of the pairwise CSV files in the data bank |
| `verify_construction_wide.py`, `verification_construction_wide.json`, `verify_construction_wide.out` | reproduction of the exploratory calculation entry by entry, the plain-counting oracle, identity with the source archives, the earlier within-cell matrices as same-cell blocks, the unchanged earlier artifacts |
| `review_builds/` | the isolated review nuisance artifacts (tag `PROPOSALCW4`: specs, artifacts, reports, exports), the build script and logs, `comparison_<COND>.json` |
| `full_suite.log` | the complete test suite on the final code (local; `*.log` files are not tracked) |
| `SHA256SUMS` | checksums of this folder |

Nothing here is a production artifact. `RANK4` and `SEL` are unchanged.
