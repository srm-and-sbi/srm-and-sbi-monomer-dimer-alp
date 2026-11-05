# Window-posterior additional cells (2026-11-03)

Evidence of the one-cell-per-condition extension of the window-posterior pilot (handoff
`Special_Analyses/DETECTOR_WINDOW_POSTERIOR_HANDOFF_2026-11-03/ADDITIONAL_CELLS_HANDOFF.md`): Fab cell 4 and InlB cell 42,
chosen by the handoff's rule (closest existing cell-level brightness median to the condition's SEL brightness among unused
local recordings). The findings are in `REPORT.md`. The first pilot (Fab 26, InlB 11) is in `../Window_Posterior_Pilot_2026-11-03/`.

Scripts: the repository companion `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Draws.py`
(GPU; fresh joint draws; this extension added its output-only `--collection-label`), and two helpers kept here:
`aggregation_comparison.py` (cell-agnostic equal-window aggregation of one collection; reproduces the Fab-26 and InlB-11
`reproduce.py` results exactly) and `four_cell_comparison.py` (side-by-side tabulation and figure; recomputes nothing).

Runs (PC profile `mars_pc`, package 0.1.44, HEAD 3824fd1 plus uncommitted analysis files; seed 20261103; CUDA Quadro T2000):

| log | command | product |
|---|---|---|
| `draws_FAB4.log` (+ `.time`) | Draws `--condition FAB --artifact-tag C128V42A --cells 4 --posterior-samples 10000 --seed 20261103 --collection-label CELL4` | data bank `Posit/..._DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws_CELL4/` |
| `draws_INLB42.log` (+ `.time`) | Draws `--condition INLB --artifact-tag C128V42AP3LR32 --cells 42 ... --collection-label CELL42` | data bank `Posit/..._DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws_CELL42/` |
| `aggregation_FAB4.log` | `aggregation_comparison.py --source <CELL4 npz copy here> --out-dir Special_Analyses/FAB4_AGGREGATION_COMPARISON_2026-11-03` | that folder |
| `aggregation_INLB42.log` | `aggregation_comparison.py --source <CELL42 npz copy here> --out-dir Special_Analyses/INLB42_AGGREGATION_COMPARISON_2026-11-03` | that folder |
| `four_cell.log` | `four_cell_comparison.py --out-dir Special_Analyses/FOUR_CELL_AGGREGATION_COMPARISON_2026-11-03 --cell ... (four cells)` | that folder |

Dry-runs preceded the two collections (adopted inputs, distinct new destinations); the unlabeled default against the existing
pilot folder and a second run against the written `CELL42` folder were both refused, as required.

Backup of the draws: the two `.npz` collections (2.1 MiB each), their `report.md` and `provenance.json` are copied here, so the
workspace backup (`Backup_RCL_Agent_Projects.sh`) covers them; the data-bank originals are the records. SHA-256: `SHA256SUMS`.

Revision after review (2026-11-03, later): the four-cell comparison gained two marginal-compatibility diagnostics (the pooled
vector's per-window membership and the intersection of the ten windows' 90 % intervals) and its folder was regenerated
(deterministic tabulation; nothing resampled); `REPORT.md` wording was corrected (distance rank, bleaching comparable with the
prior range, exact draw counts, cell-level representatives kept open, acquisition period to be defined). The companion's
report sentence on the distance rank was reworded for future runs; the four data-bank `report.md` files written earlier keep
the earlier sentence (they were not regenerated, which would resample).
