# Evidence: the SEL predictive pilot (Fab cell 26 and InlB cell 11), 2026-11-02 to 2026-11-03

Scripts, logs and measurement summaries of the predictive assessment of the two `SEL` imaging vectors under declared
reaction-diffusion settings (DETECTOR_WORKFLOW.md §7.6, order of the remaining steps). The clips, trajectories, figures,
spot-count and tail-structure folders and the viewing players live in the PC data bank's `Posit` tier
(`..._{FAB,INLB}_2S_50FPS_Posterior_Predictive_Video/` and its `Viewing_SEL_Pilot_2026-11-03/` subfolder, the
`..._Spot_Count_SEL_R<k>/` and `..._Tail_Structure_SEL_R<k>/` folders); this folder holds what reproduces and summarizes them.

- `render_fab26.sh`, `render_fab26.log`: the three Fab cell-26 renders (seeds 20261102/3/4, labels SEL_R1..3).
- `analyze_fab26.sh`, `analyze_fab26.log`: their spot counts and tail structure; `pilot_fab26_spot_tail_summary.txt`.
- `inlb11_pipeline.sh`, `inlb11_pipeline.log`: the InlB staging from the PC's complete movie bank (catalog rule validated
  on the ten Fab working files), the three InlB cell-11 renders, their analyses; `pilot_inlb11_spot_tail_summary.txt`.
- `pilot_width_background.py`, `pilot_width_background_{FAB26,INLB11}.out`: PSF width (direct per-frame fit), background,
  signal and quantiles, opening (frames 0-99) and closing (900-999) windows, experimental vs synthetic.
- `fab_candidates_experimental_summary.md`: the ten Fab recordings' direct opening/closing summaries (REF-era records).
- `export_common_window_players.py` (+ logs, `players_manifest_{FAB26,INLB11}.json`): the lossless common-window players.
- `stage_inlb_recordings.sh`: the JUWELS staging alternative, not used (no master); superseded by the PC-bank staging.

The repository's `*.log` rule ignores the logs here: add them explicitly (`git add -f`) if this folder is committed.

## sigma_pc-only arms (2026-11-03, reviewer brief via the lead)

`sigma_arms.sh` / `sigma_arms.log`: mu_pc restored to the SEL values (Fab 105, InlB 152.9); `sigma_pc` raised alone on the
saved higher-abundance renders (Fab cell 26 `SEL_N30000_SINGLE` 0.5848 -> 0.8, arm `SIGPC08`; InlB cell 11
`SEL_N20000_SINGLE` 0.8588 -> 1.0, arm `SIGPC10`), trajectory, dye counts, seed and lambda kept, via `brightness_arm.py`;
histograms via `brightness_histograms.py` (series labels generalized to the overridden key). Outputs beside the clips:
`<source stem>_SIGPC08|SIGPC10_{Synthetic_Video.npz,Comparison.png,Brightness_Histograms.{png,md,json}}`.
Pooled synth/exp: Fab p99.9 0.65 -> 0.76, p99.99 0.61 -> 0.79, max 0.66 -> 1.34; InlB p99 0.42 -> 0.43, p99.9 0.39 -> 0.46,
p99.99 0.45 -> 0.60, max 1.03 -> 1.51; medians and p90 unchanged in both. The stronger values of the brief (Fab 1.0,
InlB 1.2 = above the detector's training ceiling, exploratory) were not run: the moderate arms already overshoot the maximum.
