# Posterior-predictive check in the embedding space (detector)

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Embedding.py`, the detector shim
of the posterior-predictive embedding check. For every experimental window it simulates videos at joint
draws of the window's **imaging** posterior, embeds them through the trained detector network, and
ranks the window among them.

The steps, the statistic, the options, the outputs and the caveats are **identical for both workflows**
and are documented once in the authoritative companion:

→ **`SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Embedding.md`**

Read that note's section *"What a predictive video is"* before interpreting a result. In this workflow
the camera parameters and the whole reaction-diffusion block (receptor number, composition, mobility)
are nuisances drawn from their training distribution for every video, so a window's predictive cloud is
broad. A window outside it is a warning about the combination of imaging model and inferred values,
without saying which is at fault; a window inside it is weaker evidence than the same reading in the
biology workflow.

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Embedding.py \
    --step plan|simulate|analyze --condition FAB --total-time-seconds 2.0 [--artifact-tag TAG] [--dry-run]
```
