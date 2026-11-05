# Window-posterior pilot (2026-11-03)

Evidence of the detector window-posterior review and the two-cell joint-sampling pilot (handoff
`Special_Analyses/DETECTOR_WINDOW_POSTERIOR_HANDOFF_2026-11-03/CLAUDE_HANDOFF.md`). The findings are in `REPORT.md`.

Scripts (repository companions): `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Window_Posterior_Review.py`
(CPU; existing products) and `..._Experiment_Window_Posterior_Draws.py` (GPU; fresh joint draws), each with a `.md` note.

Runs (PC profile `mars_pc`, package 0.1.44, HEAD 3824fd1 plus uncommitted analysis files):

| log | command | product (data bank, `Posit/`) |
|---|---|---|
| `review_FAB.log` | Review `--condition FAB --artifact-tag C128V42A --pilot-cells 26 --verified-csv <handoff>/VERIFIED_INTERVAL_SUMMARIES.csv` | `..._DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Review/` |
| `review_INLB.log` | Review `--condition INLB --artifact-tag C128V42AP3LR32 --pilot-cells 11 ...` | `..._DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Review/` |
| `draws_FAB26.log` | Draws `--condition FAB --artifact-tag C128V42A --cells 26 --seed 20261103` | `..._DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws/` |
| `draws_INLB11.log` | Draws `--condition INLB --artifact-tag C128V42AP3LR32 --cells 11 --seed 20261103` | `..._DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws/` |

The review folders were regenerated once after a figure-layout fix (identical numbers; CPU, deterministic). The draw
collections were sampled once; `rerender_correlation_heatmaps.py` re-rendered their `figures/correlations_cell_<c>.png`
from the saved clouds (same numbers; tick labels overlapped in the first rendering).

Backup of the draws: the two `.npz` collections (2.1 MiB each), their `report.md` and `provenance.json` are copied here,
so the workspace backup (`Backup_RCL_Agent_Projects.sh`) covers them; the data-bank originals are the records.
SHA-256 of the copies: see `SHA256SUMS`.
