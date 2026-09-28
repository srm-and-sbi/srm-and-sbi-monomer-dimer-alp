# Posterior-predictive video (detector)

Companion to `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py`, the detector shim of the
posterior-predictive video comparison. It renders a synthetic video from the **imaging** parameters
inferred for one real MET recording and places it beside that recording.

The mechanics, the options, the outputs, and the interpretation are documented once in the
authoritative companion:

→ **`SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md`**, whose section *The files of the
posterior-predictive video check* lists every file the check needs, including the declared configuration
and the receptor-total script beside it.

Read that note's table *"One engine, two workflows"* first: for this workflow the MAP supplies the
six imaging parameters, the reaction-diffusion block is a marginalized nuisance (drawn per render from the
biology prior as the RDS stage draws a tier, pinned with `--fixed-nuisance-RDS`, or declared with
`--declared-rds`), and the system is built with its **full reaction network** — the same simulator the
detector was calibrated against. Biology inverts the first two. A detector render follows the training
generation's conventions: the same reaction-diffusion implementation, timing, geometry and trajectory
handling, the condition's labeling law and declared occupancy through the production labeling path, and the
same renderer and parameter order; an override applies only when requested and is recorded (the companion's
*The render follows the production observation layer*). MET camera provenance:
`REFERENCE_EMCCD_NOISE_MODEL.md` Sec. 6 and `DETECTOR_WORKFLOW.md` Sec. 6.5.

Outputs land under `<data_bank>/<posit>/<alias>_<timing_label>_Posterior_Predictive_Video/`, where
`<alias>` is `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR`.

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py \
    --total-time-seconds 2.0 --kind MET-FAB --cell 0 [--fixed-imaging-parameters] [--dry-run]
```
