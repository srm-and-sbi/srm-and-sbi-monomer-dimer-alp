# Evidence: rank coordinate spaces (0.1.48, 2026-11-09)

Start with [COMPLETION_NOTE.md](COMPLETION_NOTE.md).

| File | Purpose |
|---|---|
| `regenerate_<COND>_COHORT60RANKSR3.{log,time}` | the CPU regenerations of both complete-cohort reports with the four-coordinate space `spatial_brightness` |
| `review_copies/` | complete copies of the two regenerated reports (report, format-3 archive, CSV exports incl. `aggregation_compatibility_spatial_brightness.csv`, manifest, figures, both atlases) |
| `independent_rank_oracle.py`, `independent_rank_oracle_<COND>.{json,log}` | plain-counting recomputation of every rank of every space from the draw collections; exact agreement; primary arrays identical to R2 |
| `compare_spaces.py`, `comparison_<COND>.{json,md}` | the member weights of the proposed selection under the two spaces |
| `source_hashes_check.log` | SHA-256 check of the 136 numerical source products against the 0.1.47 list |
| `full_suite.log` | the complete repository test suite at this checkpoint |
| `codex_review_probes_rerun/` | Codex's round-2 review probes rerun after the corrections (availability per space, HPC forwarding) |
| `SHA256SUMS` | integrity of this package |
