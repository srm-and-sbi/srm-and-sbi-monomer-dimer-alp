# Evidence: the `SEL` `Nuisance_DLI` artifacts of both conditions, built and validated on 2026-11-02

Copies of the persisted objects of the Fab and InlB `SEL` imaging treatment (DETECTOR_WORKFLOW.md §7.6, the
decision record of 2026-10-27, built state of 2026-11-02), kept inside the workspace so that the workspace backup
covers them; the originals live in the PC data bank's `Posit` tier, which that backup does not cover.

- `..._DETECTOR_{FAB,INLB}_2S_50FPS_SEL_Nuisance_DLI_Spec.toml`: the filled `selection_user` specs (unchanged since
  2026-10-27), the inputs of the build.
- `..._DETECTOR_{FAB,INLB}_2S_50FPS_SEL_Nuisance_DLI.npz`: the artifacts built from them on the PC with
  `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py --condition <C> --total-time-seconds 2.0 --nuisance-tag SEL --build`
  (CPU only; package 0.1.44, tree at commit 3824fd1 plus this folder).
- `..._SEL_Nuisance_DLI_Analysis/`: the build's report and marginals figure per condition.
- `verify_sel.py`, `verify_sel.out`: the validation through the production loader (`require_nuisance_dli` with the
  tag): key order, physical values against the approved tables, repeated draws identical to the fixed vector, the
  selection record's sources, limitations and the InlB `mu_r` acknowledgement; result ALL PASS. Run from the
  repository root with `MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python <this file>`.
- `SHA256SUMS`: checksums of the copies; those of the specs and artifacts equal the ones recorded in §7.6.

Known reporting inconsistency, documented and left: the build's `--dry-run` prints `pool_mode=bounded` for InlB
before the build, while the persisted InlB artifact records `pool_mode=unrestricted`, which marks the acknowledged
out-of-box `mu_r` and governs no sampling.
