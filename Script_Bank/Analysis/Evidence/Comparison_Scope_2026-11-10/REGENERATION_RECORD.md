# Regeneration record: complete-cohort Experiment reports, review label COHORT60RANKSR4

CPU regeneration on the PC (`mars_pc`), 2026-11-10, of both complete-cohort reports from the retained clouds (sixty
window-posterior collections per condition, the sources of the previous review label), with the construction-wide
comparison scope and the additional rank space `spatial_brightness = mu_r, sigma_r, mu_pc, sigma_pc`. No neural
sampling, no estimator load, no HPC job.

| Condition | Command | Written | Wall time | Exit | Report checks |
|---|---|---|---|---|---|
| FAB | `regenerate_FAB_COHORT60RANKSR4.command` | 11:39 to 11:40 | 1 min 12 s | 0 | 8 passed, 0 failed |
| INLB | `regenerate_INLB_COHORT60RANKSR4.command` | 11:40 to 11:41 | 1 min 05 s | 0 | 8 passed, 0 failed |

The commands are those of the previous regeneration (`Rank_Coordinate_Spaces_2026-11-09/regenerate_<COND>_COHORT60RANKSR3.time`)
with the review label replaced. Products: `<Posit>/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_<COND>_2S_50FPS_<TAG>_MAP_Experiment_COHORT60RANKSR4/`
(report, format-4 archive, CSV exports, manifest, figures, atlases); copies without the two large pairwise CSV files
in `review_copies/`; the checksums of the archives and of the pairwise CSV files left in the data bank are in
`data_bank_checksums.sha256`.

**Console logs.** The console logs and timing files of the two successful runs were overwritten by a second
invocation of the same commands, made to confirm the final console header; that invocation was refused by the
stage's own guard ("review copy exists already; refusing to overwrite it") and changed nothing in the data bank. Its
logs are kept as `regenerate_<COND>_COHORT60RANKSR4_second_attempt_refused.{log,time}`. The table above records the
successful runs as they were observed (wall time and exit status from the timing output, the check line from the
console); the reports themselves carry the complete check tables.

**Code state.** `srm_and_sbi_monomer_dimer_alp/experiment_aggregation.py`, which computes the ranks and writes the
archive and the subsection, was last modified at 11:25, before the regeneration, and is unchanged since. After the
regeneration only the constructor-side figure functions of `visualization_aggregation.py` (histogram axis labels, grid
margins) and two console/help strings of `experiment_runner.py` changed. `verify_construction_wide.py` was run again
after those changes: the archives read back and validate under the final code
(`verification_construction_wide.json`, `verify_construction_wide.out`).
