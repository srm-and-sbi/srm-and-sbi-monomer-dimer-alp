# Detector Calibration Workflow — design and justification

*The Detector workflow calibrates the diffraction-limited-imaging model — camera, point-spread function, emitter brightness, and brightness flicker — by inferring it with the reaction-diffusion biology marginalized over its full prior, so that the reaction-diffusion inference downstream rests on a justified, reproducible imaging model. "DLI" (diffraction-limited imaging) and "Detector" denote the same model throughout.*

**Scope.** This document records the Detector-workflow design, the parameter ranges and their justification, and the implementation plan. It is self-contained: it justifies choices on physics, the in-repository numerical study, publicly available data and methods, and the current codebase (package `srm_and_sbi_monomer_dimer_alp`). American spelling; ranges in log10 space where noted.

**Why this workflow exists.** The imaging parameters (camera gain, offset, read-noise, photon conversion; PSF widths; emitter brightness distribution; the brightness-flicker generator) shape every synthetic training video the reaction-diffusion (biology) production model learns from, and the biology workflow marginalizes over them. For peer review the imaging model behind them must be *justified and reproducible* rather than set by visual matching. The Detector workflow calibrates them by simulation-based inference against experimental videos, producing a provenanced parameter artifact with quantified justification, and exposes an explicit route from that artifact into the production runs.

---

## 1. Motivation and problem

The production model infers biology (diffusion coefficients, reaction rates, populations) from single-particle-tracking microscopy videos via an amortized neural posterior estimator with a normalizing-flow density. Every synthetic training video is rendered through an imaging model that must match the experimental acquisitions. If it is misspecified relative to them, experimental videos fall off the synthetic manifold the estimator was trained on, and the posterior extrapolates — the mechanism behind the observed out-of-prior estimates on experimental data.

Two deliverables follow:

1. **Justification.** Each imaging parameter must have a defensible value with a stated basis, not a hand-tuned constant.
2. **Reproducibility.** The route from experimental videos → calibrated parameters → production priors must be auditable and re-runnable.

The Detector workflow provides both by treating the imaging model itself as the inference target.

---

## 2. Role in the pipeline — how it feeds production

The Detector workflow is a **complete, first-class calibration workflow**, permanent and fundamental to the methodology. It mirrors the biology workflow stage for stage — simulate → infer → evaluate → experiment — differing only in its inference target (below), and it calibrates the diffraction-limited-imaging model the production pipeline depends on, so the imaging parameters are justified and reproducible rather than hand-tuned and the experimental-versus-synthetic domain gap is measured rather than assumed away. The two workflows run on **one shared engine per stage**: each stage's orchestration lives in a `srm_and_sbi_monomer_dimer_alp.<stage>_runner.run_<stage>(cfg, args)`, and the two Prime entry points per stage past RDS — the biology one and its `_DETECTOR`-qualified twin — are thin shims that build a `srm_and_sbi_monomer_dimer_alp.workflow.WorkflowConfig` (`biology_workflow()` / `detector_workflow()`) and call the shared runner (the RDS stage has one entry point and no twin, run once per condition: each condition's trajectory tier is shared by both workflows, §4). The Detector carries its **own committed HPC submission machinery** — the `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_*` stage wrappers (Simulation — a DLI-only pass over the condition's trajectory tier —, Inference, Evaluation, and Experiment; the coverage and gap analyses run through the workflow-general analysis wrappers `SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Posterior_Calibration.sh` and `SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Embedding_Space_Distance.sh`, each serving either workflow via its `WORKFLOW=biology|detector` knob; the workflow-general post-hoc analyses — posterior calibration, embedding-space distance, posterior-predictive video, the sample geometric median, and temporal dynamics — each run from a `_DETECTOR`-namespaced Analysis shim over the same shared engine as its biology twin) plus a `_DETECTOR_HPC_Submit` dispatcher, mirroring the biology stage-wrappers-plus-`Submit.sh` pattern (generic, `hpc_local.env`-driven, dry-run default), filename-namespaced alongside the biology wrappers in `Script_Bank/HPC/` — the same filename-alias scheme (Option F) as the Prime entry scripts, not a subfolder. At the HPC level the two dispatchers stay distinct: the biology `Submit.sh` and the Detector `_DETECTOR_HPC_Submit` each drive only their own stage wrappers. The one shared input is the condition's RDS trajectory tier, generated per condition by the biology dispatcher's RDS-only simulation and re-imaged by the Detector's DLI-only Simulation wrapper.

Its output is a **versioned, provenanced DLI-parameter artifact**: for each imaging parameter, the calibrated estimate with a credible interval, an in/out-of-prior flag, the source video conditions and public accessions, the estimator checksum, and the training configuration.

**Priority between the two workflows.** The biology workflow carries the scientific question: it infers
composition, diffusion, switching, and reaction parameters from videos, and nothing replaces that
inference. The Detector workflow and the direct imaging estimators are supporting tools whose purpose is
to keep imaging error out of those biological estimates: an incorrect or overly narrow imaging assumption
is absorbed into the inferred biology, which is why the imaging block must be constrained and its
remaining uncertainty propagated rather than assumed away. The imaging work therefore aims at defensible
values with uncertainty ranges, obtained through acquisition information (§9.4), direct measurements
(§9.5, §9.6), and neural inference where those do not reach, not at perfect recovery of every imaging
parameter. The practical objective is to constrain imaging well enough, propagate what remains uncertain,
and assess which biological quantities the videos actually support. Estimator developments tested on the
Detector problem (§9.7) are controlled benchmarks: success there does not establish better biological
inference, which requires its own recovery and calibration tests against known biological truth.

That artifact **feeds the production model's imaging parameters**. The production treatment of those parameters — holding them fixed at the calibrated values, inferring them jointly under priors *centered on the calibrated values*, or marginalizing them as a nuisance (§7) — is decided when the artifact is ported into production; the calibration itself is agnostic to that choice, and the guess → refinement → prior chain stays fully auditable under any of them.

**Workflow architecture.** The biology and Detector workflows are two mirrored passes over the same stages — simulate → infer → evaluate → experiment — differing only in the target: the biology workflow infers the reaction-diffusion parameters and marginalizes the imaging block, while the Detector infers the imaging parameters and marginalizes the reaction-diffusion domain and the camera. Both run on one shared engine per stage — `run_<stage>(cfg, args)` in the stage-runner modules (`simulation_rds_runner.py`, `simulation_dli_runner.py`, `inference_runner.py`, `evaluation_runner.py`, `experiment_runner.py`) — selected by the `WorkflowConfig` a thin Prime shim builds (`biology_workflow()` / `detector_workflow()`). Because the learnable target is what every stage samples, places priors over, trains on, and recovers, the genuine per-workflow differences — the parameterization module, the alias-qualified paths, the workflow tag, and the DLI imaging source (the `Nuisance_DLI` artifact for biology, the imaging prior box for the Detector) — are carried by the config and localized in a `_<stage>_spec(cfg)` resolver, so the two workflows mirror each other and neither can silently drift. The production treatment of the calibrated imaging is decided at the shared parameterization (§9.2 Phase D); §5 identifies the exact hazard.

---

## 3. Method — two-stage factorized calibration

Inferring biology and imaging jointly is a high-dimensional problem (roughly twenty-one parameters) that a single amortized estimator calibrates poorly. The workflow factorizes it into a cascade in which each stage marginalizes the other's parameters as a nuisance drawn from a restricted distribution, approximating the intractable joint.

**Stage 1 — simulation-based inference over the imaging model.** The biology is marginalized over its full reactive prior (§5), so the imaging estimate is conditioned on no particular kinetics while the videos still carry the labeling signatures the kinetics produce. An amortized neural posterior estimator whose parameter vector *is* the imaging model is trained on synthetic videos, then maximum-a-posteriori estimated on experimental videos, per condition. This yields a principled initial estimate for every imaging parameter, with a posterior (not only a point) so that uncertainty and identifiability are visible.

**Stage 2 — classical calibration and quantitative matching.** The Stage-1 estimate is refined with closed-form estimators grounded in detector physics:

- **Offset** from the mode of the pixel-intensity histogram.
- **Read-noise** from the variance of background pixels.
- **Overall system gain** `γ = g/C` (ADU per photoelectron) from the photon-transfer slope; a single photon-transfer curve identifies only this ratio, so `γ` is inferred directly and the electron-multiplication gain `g` and conversion factor `C` are fixed to their nominal spec values (metadata for the `γ`-vs-`g_spec/C_spec` drift check), not decomposed from the videos (see `REFERENCE_EMCCD_NOISE_MODEL.md` §6, §9).
- **Brightness** from a log-normal fit to the experimental photon histogram.
- **PSF width** from a direct fit to isolated emitters.

Stage 2 adds a **quantitative experimental-versus-synthetic distance** (§8) so that "the videos match" becomes a measured scalar with a significance test, not a visual impression. The stage-1 → stage-2 refinement rule is recorded explicitly as part of the provenance.

---

## 4. The reactive physics model and the per-condition trajectory tier

Stage 1 renders the SAME reactive trajectories the biology workflow renders. There is one RDS trajectory tier per condition, generated by the single RDS entry point (`SRM_AND_SBI_MONOMER_DIMER_ALP_Simulation_RDS.py --condition FAB|INLB`; `build_system` and `build_simulation` in `simulation_rds_support.py`, the condition's generated reaction channels registered — seventeen under MET-INLB, eleven under MET-FAB, whose association ratio is zero) from the biology prior, under the sibling alias plus the condition token and no workflow qualifier (`Paths.rds_alias`). The condition enters at the RDS stage because the association ratio of the model is a declared per-condition constant — MET-FAB `R_ON = 0`, no association channel, pre-existing dimers may still dissociate; MET-INLB `R_ON = 1`, the reference convention `λ_on = 6 D_A / r²` — and not an inferred parameter (`PROJECT_CONTEXT.md` §2). Each tier's eleven-parameter `Theta_Set` (physical values) is the biology's learnable label and, to the Detector, the record of the reaction-diffusion nuisance it marginalizes: the Detector has no RDS stage of its own, draws no separate nuisance, and keeps no copy of the biology ranges to check for equality — it marginalizes the biology prior by construction. In the value-based scheme (§5) the eleven RDS rows of the Detector table — the same eleven keys as the biology table, grouped `stoichiometry` / `mobility` — are therefore nuisance-from-object, the supplied distribution being the condition's tier. The Detector's Simulation stage is its DLI pass over that tier, per condition, with the imaging drawn from the Detector prior. The one-page account of what is drawn once per condition and what each workflow adds at imaging, with a worked example of the file names, is `PROJECT_CONTEXT.md` §4, *One draw of the biology per condition, two imagings per draw*.

**Why reactive, not diffusion-only.** A diffusion-only variant — the six particle types registered with their diffusion constants and the reaction registrations skipped — would be congruent with the reactive system in species set, diffusion constants, initial placement, and the entire imaging pipeline, and it would still be misspecified for the calibration in two ways that the labeling model (§6.6) makes decisive. First, under incomplete labeling a dissociating one-dye dimer leaves one visible and one permanently invisible daughter, so track disappearance is partly a kinetic event; a detector shown only static composition could explain such disappearances as photobleaching alone and would bias `prob_photo_bleach` upward on the experimental recordings, where dissociation happens. Second, the DOL-explicit rendering follows every receptor subunit through the reactions via the reaction records (`extract_subunit_lineage`), and ReaDDy writes no reaction-record dataset for a system with no registered reactions — a diffusion-only trajectory cannot be rendered at all.

**What the calibration marginalizes.** Eleven parameters — the whole reaction-diffusion model of the separated stoichiometry–mobility family (`PROJECT_CONTEXT.md` §2): the stoichiometry block (the conserved receptor total `N_R`, the requested initial dimer-to-monomer ratio `r`, the dissociation rate `κ_OFF`; the association ratio is the condition's declared constant, not a row) and the mobility block (the monomer scale `D_A`, the dimer factor `R_B`, the slow and immobile mode factors `R_s`, `R_i`, and the four switching rates shared by both species; §6.1), all drawn per simulation from the biology's decided prior ranges, identical for both conditions. Population dynamics are therefore realistic rather than frozen: the monomer–dimer composition and the mode occupancies evolve through the condition's generated channels — under MET-INLB dimers form and dissociate, under MET-FAB pre-existing dimers dissociate and none form — and the visible population carries the labeling statistics of §6.6. The imaging parameters remain species-agnostic and are applied per dye by identical code in both workflows: both workflows use the same observation model, which establishes implementation consistency. The condition-specific calibration supplies imaging-parameter distributions for downstream marginalization; their adequacy under the revised model remains to be assessed.

**Cost.** The Detector adds no RDS compute at all: each condition's tier is generated once and both workflows re-image it, so a two-condition, two-workflow campaign is two RDS tiers and four DLI passes. Sharing also yields matched pairs within a condition for free — a condition's held-out trajectories are identical under both workflows, so its biology and detector estimators are compared on the same latent motion; across conditions, held-out recordings are matched by parameter draw, not by trajectory.

---

## 5. Parameterization mechanism — value-based roles

The Detector parameterization lives in its **own module** (`detector_parameterization.py`), decoupled from the biology `parameterization.py`, because the detector-calibration system is similar to but distinct from the production system and its ranges differ by design.

**Biology scheme (for context).** The biology `parameterization.py` uses the same value-based role dispatch described below (`role_of`, ported from this workflow): a concrete `VALUE` with a `(low, high)` range is learnable (included in the inference prior and the θ vector, which `build_prior` turns into a `BoxUniform` over the estimator space); a concrete `VALUE` with `PRIOR_RANGE = None` is fixed; the sentinels mark the nuisance and posterior roles. Its learnable subset is the eleven reaction-diffusion rows of the two model blocks (the association ratio is a per-condition constant, not a row). `LOG_FLAG` is **not** documentation: it is each ranged row's declared scale — `True` means the prior box and the estimator coordinate are `log10` of the physical value, `False` means the coordinate *is* the physical value (a linear row; the decided biology table has none, and the rule is what keeps declaring one a free choice) — and `parameterization.to_physical` / `to_flow` are the only sanctioned conversions between the estimator's space and physical values, applied row by row by every consumer (sampling, the training dataset, evaluation, calibration, diagnostics, analyses). No consumer writes `10**theta` by hand, because a blanket exponentiation would corrupt any linear row. `Theta_Set` files store physical values.

**The value-based scheme (Detector).** To let one specification express learnable, fixed, nuisance, and (future) posterior-drawn parameters, the role selector moves to the `VALUE` field via two reserved string sentinels. The complete role matrix over `VALUE × PRIOR_RANGE`:

| `VALUE` | `PRIOR_RANGE` | role |
|---|---|---|
| concrete value (scalar or list) | `(low, high)` | **Learnable** — inferred; `VALUE` is the prior center |
| concrete value | `None` | **Fixed** — constant read directly |
| `"NUISANCE"` | `(low, high)` | **Nuisance from spec** — drawn from an inline `BoxUniform` over the range |
| `"NUISANCE"` | `None` | **Nuisance from object** — supplied from outside the table: the `Nuisance_DLI` artifact with its parameter-key manifest, or the condition's RDS trajectory tier (§4) |
| `"POSTERIOR"` | `None` | **Posterior draw** — future multiround inference |
| `"POSTERIOR"` | `(low, high)` | **undefined** — rejected by a build-time assertion |

**Design constraints (verified against the current code — these are the port hazards).**

1. **Role dispatch must be sentinel-based**, testing `VALUE in {"NUISANCE", "POSTERIOR"}`, and must **never** test whether `VALUE` is numeric — a list-valued fixed parameter already exists in the specification and would be misclassified by a numeric test.
2. **The learnable-subset selector must be `VALUE-is-not-a-sentinel AND PRIOR_RANGE-is-not-None`.** The biology filter selects learnable rows by `PRIOR_RANGE is not None` alone; under the value-based scheme a nuisance-from-spec row *also* carries a range, so reusing the biology filter verbatim would pull nuisance parameters into the inference prior and the θ vector. This is the single most important incompatibility to handle at port time: nuisance rows carry a range that feeds only their own draw and must be excluded from the inference prior.
3. **Log semantics must be explicit.** The scheme honors `LOG_BASE` consistently for both the center invariant and the sample-to-physical mapping (or enforces log10 for all ranged rows), and it specifies, for the supplied-distribution nuisance and the posterior cases, whether draws arrive already in physical space or in log space requiring exponentiation.

**Parameter categorization (the concrete instantiation).** Applying the role matrix above, every parameter of the Detector model falls into one of five categories:

| category | role | members | rationale |
|---|---|---|---|
| **Inferred imaging** — the calibration targets | learnable | the 6 identifiable emitter parameters of §6.2: PSF (`mu_r`, `sigma_r`), brightness (`mu_pc`, `sigma_pc`), photophysics (`prob_photo_bleach`, `lambda_rate`) | the imaging model the workflow calibrates, inferred within the data-anchored priors of §6.2 |
| **SCOPE camera nuisance** — the camera | nuisance | the 5 EMCCD camera parameters (`gamma`, `kappa_o`, `kappa_b`, `kappa_s`, `kappa_q`) of §6.2 | externally constrained rather than jointly inferred in this workflow: the gain and conversion enter the image likelihood only through their ratio, and the amplitude and floor depend on the products `gamma·kappa_q` and `gamma·kappa_q·kappa_o`, so the individual camera quantities are confounded in the biological recordings, while camera-calibration measurements constrain them individually (`REFERENCE_EMCCD_NOISE_MODEL.md` §8); marginalized over a-priori boxes, drawn at the DLI stage, and shared with the production workflow (§9.3) |
| **RDS nuisance** — the biology | nuisance | the stoichiometry block (`count_total`, `ratio_dimer_monomer_initial`, `rate_dissociation`) and the mobility block (`diffusivity_alp`, `relative_diffusivity_dimer`, `relative_diffusivity_slow`, `relative_diffusivity_immobile`, `rate_fast_slow`, `rate_slow_fast`, `rate_slow_immobile`, `rate_immobile_slow`) of §6.1 — eleven rows, the biology prior | marginalized while the imaging is calibrated, by re-imaging the condition's trajectory tier (nuisance-from-object, §4); the reaction-diffusion biology is not the target here |
| **Conditioned by the coordinate frame / acquisition** | fixed | pixel size, field size (`root_size_px`), and frame time (`delta_frame`) | a coordinate frame and cadence, not physics — degenerate with the emitter dynamics and geometry, hence supplied per dataset as acquisition metadata rather than inferred. The renderer samples positions and brightness at the frame interval and does not integrate motion during exposure; the experimental exposure duration is separate acquisition metadata |
| **Fixed modeling hyperparameters and spec metadata** | fixed | the bleaching reference window (`numb_photo_bleach`) and the nominal EM gain / conversion (`kappa_g`, `kappa_c`) | modeling choices and spec constants held fixed by design; `kappa_g`/`kappa_c` are retained only as metadata for the `γ = g/C` drift check (§8) |

The gain/conversion pair is the one subtlety: `kappa_g` (`g`) and `kappa_c` (`C`) are individually fixed spec metadata, and their ratio `gamma = g/C` — the only gain quantity identifiable from the videos — is marginalized as part of the SCOPE camera nuisance (§6.2; §9.3; `REFERENCE_EMCCD_NOISE_MODEL.md` §9). The counterpart categorization for the production (biology) workflow makes the biology the inference target and marginalizes the whole imaging block — which is the value-based scheme's reason for existing (§2).

---

## 6. Parameter ranges and their justification

Two groups: the **RDS nuisance** (biology marginalized during detector calibration) and the **learnable imaging parameters** (the calibration targets). Ranges are given in the estimator coordinate declared by each row's `LOG_FLAG`: log10 for every imaging row and for every one of the eleven RDS rows (the decided biology table has no linear row). Absolute ranges are shown for reference, and each prior center (a log row's geometric midpoint) is a representative value the calibration is free to move away from as the data require.

### 6.1 RDS nuisance (the biology prior, supplied by the condition's trajectory tier)

These are the biology table's ranges, reproduced here for reading. The Detector table carries these eleven rows without ranges (nuisance-from-object, §5; grouped `stoichiometry` / `mobility` like the biology table), and the condition's tier's `Theta_Set` is the draw; the authoritative ranges are the biology table in `parameterization.py`, and changing one there changes the tiers both workflows read. The association ratio has no row: it is the condition's declared constant (MET-FAB 0, MET-INLB 1; §4), so the MET-FAB calibration videos contain no association event and the MET-INLB videos associate at the reference intensity. **Every range below is a decided prior** (2026-09-14; rationale: `PROJECT_CONTEXT.md` §2, *How the prior ranges and the declared inputs are set*): box-uniform in the estimator coordinate, covering the plausible support with a margin and never encoding a condition's expected answer, and shared by both conditions (condition-specific ranges remain permissible but unused). The receptor count is conditional on the declared per-condition probe occupancies (MET-INLB 0.5 declared, MET-FAB 0.155 derived; §6.6), which the DLI stages of both workflows apply by default. The model itself (two molecular species × three mobility modes, the channels generated per condition, the association normalization and the per-condition setting, the boundary behavior) is described in `PROJECT_CONTEXT.md` §2.

| parameter | scale | prior range (estimator coordinate, log10) | absolute | role in the calibration videos |
|---|---|---|---|---|
| `count_total` (N_R) | log10 | [2.5, 3.5] | 316–3162 subunits | The conserved receptor-subunit total of the simulated patch (the in-field count is a distinct, time-dependent quantity under the open lateral boundary), realized as integers by `realize_initial_composition`. Set from the deposited recordings' first-2 s spot counts divided by the declared visibility per subunit (Special_Analyses A9), so it is conditional on the declared occupancies; the labeling law and the occupancy then thin the population per condition (§6.6). |
| `ratio_dimer_monomer_initial` (r_{B/A}) | log10 | [−2, 2] | 1:100 to 100:1 | The requested initial dimer-to-monomer ratio `n_B(0) / n_A(0)`, symmetric about an even split (complex fraction `f_B = r / (1 + r)` from 1 % to 99 %); realized as `n_B(0) = min(round(N_R r / (1 + 2r)), floor(N_R / 2))`, the realized composition recorded beside it. The receptor fraction `x_B = 2r / (1 + 2r)` is derived. |
| `rate_dissociation` (κ_OFF) | log10 | [−3, 1] | 0.001–10 /s | Dimer unbinding, `B_m → A_m + A_m` for every mode (mode conserved), inferred in both conditions; the lower bound lets pre-existing MET-FAB dimers persist through a 20 s recording. Under incomplete labeling a dissociating one-dye dimer leaves one visible and one invisible daughter (§5, §6.6), so this rate shapes track disappearance in the calibration videos exactly as it does in the recordings. |
| `diffusivity_alp` (D_A) | log10 | [−1.25, −0.25] | 0.056–0.562 µm²/s | The monomer scale coefficient `D[A, f]`; every other coefficient is a declared ratio of it. Brackets a representative measured monomer diffusion of ~0.10 µm²/s (public accessions S-BSST712, S-BIAD1369). |
| `relative_diffusivity_dimer` (R_B) | log10 | [−1, 0] | 0.1–1 | The dimer factor within a mode, `D[B, m] = R_B · D[A, m]`, `0 < R_B ≤ 1` (at 1 dimerization adds no slowdown; population-level differences then come from mode occupancy and inheritance). |
| `relative_diffusivity_slow` (R_s) | log10 | [−1, 0] | 0.1–1 | The slow-mode factor, `D[X, s] = R_s · D[X, f]`, for monomers and dimers alike. |
| `relative_diffusivity_immobile` (R_i) | log10 | [−3, −2] | 0.001–0.01 | The immobile-mode factor, `D[X, i] = R_i · D[X, f]` — immobile by construction: `D_i` stays below the tracking pipelines' immobility thresholds (0.0028 and 0.0065 µm²/s) over nearly the whole `D_A` range, and the lower half of the range lies below the 2 s resolution floor. A full decade below the slow range, so `R_i < R_s` for every draw (checked at import). |
| `rate_fast_slow` (k_fs) | log10 | [−1, 1] | 0.1–10 /s | Switching f → s, shared by both species. |
| `rate_slow_fast` (k_sf) | log10 | [−1, 1] | 0.1–10 /s | Switching s → f, shared by both species. |
| `rate_slow_immobile` (k_si) | log10 | [−1, 1] | 0.1–10 /s | Switching s → i, shared by both species. |
| `rate_immobile_slow` (k_is) | log10 | [−1, 1] | 0.1–10 /s | Switching i → s, shared by both species. There is no direct f ↔ i channel; each particle's initial mode is drawn from the stationary law of the isolated chain. |

No equality with the biology prior is checked any more, because none needs to be: the Detector re-images the biology's own trajectories, so the marginalized biology is the biology prior by construction.

### 6.2 Inferred imaging parameters (calibration targets)

Log-uniform priors; the prior center shown is the range's geometric midpoint (a representative value, not a fixed setting). The six inferred imaging parameters are the identifiable emitter model — PSF, brightness, and photophysics. The **reference** column gives the value that guides each prior; where the two experimental conditions differ, both are shown, and the adopted value is the monomer control (Fab) — see the placement note below and §6.7. The five EMCCD camera parameters are not inferred; they are marginalized as the SCOPE camera nuisance, whose ranges are §6.3 and whose treatment is §9.3.

| parameter | log10 range | absolute | reference (source) | forward-model role |
|---|---|---|---|---|
| `mu_r` | (0.0, 0.3) | 1.0–2.0 | 1.36 (Fab) — InlB 1.47 is dimer-broadened (§6.7) | median PSF spread (log-normal scale) |
| `sigma_r` | (−1.0, −0.25) | 0.10–0.56 | 0.37 fitted (Fab; InlB 0.42) → ≈0.15 fit-corrected (§6.7) | emitter-to-emitter PSF variability (log-normal shape) |
| `mu_pc` | (2.0, 2.75) | 100–562 | 386 (Fab, monomer) — InlB 690 is the dimer sum (§6.6/§6.7) | median monomer brightness (log-normal scale) |
| `sigma_pc` | (−0.75, 0) | 0.18–1.0 | 0.61 fitted (Fab; InlB 0.55) → ≈0.5 fit-corrected (§6.7) | brightness spread (log-normal shape); the flicker autocorrelation is independent of it (§6.5) |
| `prob_photo_bleach` | (−2, −0.5) | 0.01–0.316 | — (no public anchor; see the within-recording evidence below) | photobleaching probability over the reference frame count; sets the rate into the dark state |
| `lambda_rate` | (0.0, 1.0) | 1.0–10.0 | ≈5 — flicker correlation-time of track `intensity[photon]` (§6.5) | brightness-switching rate — the inferred flicker parameter (§6.5) |

**No external anchor.** No public measurement pins `prob_photo_bleach`, so its prior is placed on range
alone. One within-recording observation bears on how a calibrated value should be read: the temporal analysis of
the Experiment estimates shows the inferred value falling by roughly half a decade *within* a single recording,
coherently and in nearly every recording of both conditions, while the PSF median and the brightness spread stay
flat over the same windows. Heterogeneous bleaching would produce this signature in a one-rate model, but so
would an estimator bias that tracks the observation conditions changing across a recording's windows, a
declining emitter density among them; the analysis that produces the observation does not identify the cause
(`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_Temporal_Dynamics.md`). A calibrated value is
therefore an average over the windows of a recording, not a demonstrated constant of the acquisition. Whether a
recording carries the information to constrain this parameter at all is answered separately by the information
budget (§9.5): from a 2 s recording it does not, under either labeling law (§6.10).

**Prior placement.** The two population *spreads* `sigma_r` and `sigma_pc` are kept broad-but-capped: their ThunderSTORM-fitted values are upper-biased (§6.7 caveat 1), so the prior is broad enough to contain the fitted spread yet capped below the runaway-tail regime, letting the calibration likelihood pull the inferred spread down toward its fit-corrected value. Every reference is the monomer-control (Fab) statistic where the two conditions differ, because InlB's higher brightness and wider PSF are dimer artifacts the model reproduces by other means — the brightness sum (§6.6), not the priors (§6.7 caveats).

### 6.3 Camera block ranges — the SCOPE nuisance

**SCOPE camera nuisance (marginalized).** The five EMCCD camera parameters are externally constrained rather than jointly inferred in this workflow (§9.3) and are not inference targets: each is drawn per simulation from the a-priori box in the table, rendered into every video, and recorded as `Nuisance_SCOPE`. The boxes mirror the authoritative camera-parameter specification in `REFERENCE_EMCCD_NOISE_MODEL.md` §6, the source of record for the EMCCD parameterization. `kappa_g` and `kappa_c` (nominal EM gain and conversion) are held fixed as spec metadata for the `γ = g/C` drift check.

| parameter | log10 range | absolute | reference (source) | forward-model role |
|---|---|---|---|---|
| `gamma` | (1.62, 1.625) | 41.7–42.2 | 41.84 — config `g/C` (identical Fab & InlB) | gain-conversion ratio γ = g/C (ADU/e⁻); the only gain quantity the videos identify (`g`, `C` fixed metadata) |
| `kappa_o` | (1.455, 1.465) | 28.5–29.2 | 28.7 — `offset[photon]` median (Fab 28.9 / InlB 28.6) | optical background offset (incident photons), one scalar per video, added before QE |
| `kappa_b` | (2.24, 2.25) | 173.8–177.8 | 175 — config baseline (identical Fab & InlB) | camera baseline b (ADU), added after the register |
| `kappa_s` | (1.02, 1.025) | 10.5–10.6 | 10.5 — camera datasheet | read-noise standard deviation σ (ADU), gain-independent, added after the register |
| `kappa_q` | (−0.05, −0.04) | 0.89–0.91 | 0.90 — config QE (identical Fab & InlB) | quantum efficiency (only γ·κ_q is identifiable) |

The camera boxes are *tight anchors* around the acquisition-protocol values rather than broad decade brackets: the ADU floor is `gamma·kappa_q·kappa_o`, so a broad camera range would let the floor dominate the video-to-video variation and collapse the detector embedding onto that single axis, keeping the marginalization within the realistic camera manifold. The read noise `kappa_s` — a datasheet quantity, weakly identifiable — is likewise pinned to a tight band at its datasheet value (its wide band would be harmless, but the tight band keeps every camera axis minimal).

The camera boxes were sized for the computational reason recorded above, not from a measured uncertainty. A measured per-session uncertainty would be narrower than a decade in any case, so the two justifications are compatible once the values are measured (§9.4).

### 6.4 Provenance of every externally supplied value

**Evidence and source of every externally supplied value.** Two properties are recorded separately for each value the workflow does not infer: the *evidence* it rests on (an acquisition setting, a measured quantity, a datasheet value, a convention, or an assumption) and the *source* it was read from (an acquisition record, a ThunderSTORM protocol file or output column, a publication, or a code definition). The two are not competing categories: the camera baseline is an acquisition setting that was read from a ThunderSTORM protocol file. Removing the dependence on ThunderSTORM's files removes a source, not the need for the evidence; every acquisition setting below must then come from the microscope's own records, and every measured quantity from an analysis of the raw frames (§9.4).

| value | evidence | source used here |
|---|---|---|
| pixel size 158 nm, frame interval 20 ms | acquisition settings | per-cell ThunderSTORM camera protocol of `S-BSST712`; the frame interval is the fixed cadence of `PROJECT_CONTEXT.md` §2 |
| field 256 × 256 px | convention (the crop the model consumes) | code definition (`root_size_px`) |
| exposure duration | acquisition setting that the model does not represent: the renderer samples positions and brightness at the frame interval and does not integrate motion during exposure | none (a modeling convention) |
| `kappa_g` = 200, `kappa_c` = 4.78, `kappa_q` = 0.90, `kappa_b` ≈ 175 | acquisition settings | per-cell ThunderSTORM camera protocol of `S-BSST712` |
| `kappa_o` ≈ 28.7 photons | measured quantity, conditional on the camera settings above | median of the ThunderSTORM `offset[photon]` output column |
| `kappa_s` ≈ 10.5 ADU | datasheet value | public camera specification (`S-BIAD1369`) |
| the six imaging reference values above | measured medians, and derived quantities for the fit-corrected spreads and the matched flicker rate | ThunderSTORM output columns (`intensity [photon]`, `sigma [nm]`) and the derivations of §6.5 and §6.7 |
| labeling mean 1.64 dyes per Fab | measured quantity (the preparation's degree of labeling) | `PROJECT_CONTEXT.md` §2 |
| Poisson form of the labeling law | assumption | code definition (`labeling.py`), with matched-mean dispersion alternatives registered for sensitivity runs |
| occupancy 0.155 | derived from a declared visibility ratio (an assumption) | `PROJECT_CONTEXT.md` §2 |
| `numb_photo_bleach` = 100 frames | convention (how the bleaching probability is expressed) | code definition |
| 16-bit to 8-bit video conversion | convention: a fixed global map of 0–65535 onto 0–255 with clipping and no per-video normalization, applied identically to synthetic and experimental frames | code definition (`io.convert_video_dtype`) |

### 6.5 The brightness-flicker model: a stationary rate process

Per-dye ln-brightness follows a stationary Ornstein-Uhlenbeck (OU) process, discretized per frame as an AR(1) (`generate_brightness_photons`, `simulation_dli_support.py`):

`z_0 ~ N(0, sigma_pc²)`, `z_{t+1} = rho·z_t + sigma_pc·sqrt(1 − rho²)·eps_t` with `rho = exp(−lambda_rate·delta_frame)`, and `photons_t = mu_pc·exp(z_t)`.

By induction the per-frame marginal is exactly `LogNormal(ln mu_pc, sigma_pc²)` at every frame and for every clip duration: the documented brightness law holds without an initialization transient or a brightness ceiling, and `sigma_pc` carries its marginal meaning directly. `lambda_rate` is the correlation-decay rate of ln-brightness — `ACF(lag) = exp(−lambda_rate·lag)`, so the flicker correlation time is `tau_corr = 1/lambda_rate` — not a jump-event rate. Photobleaching is a state-independent absorbing process applied as an independent per-frame Bernoulli (`prob_1 = 1 − (1 − prob_photo_bleach)^(1/numb_photo_bleach)`); because it is independent of brightness, the brightness law among still-active dyes is unchanged by it. For a single emitter the flicker dynamics carry a single free parameter: `mu_pc` shifts ln-brightness additively and `sigma_pc` scales it linearly, so both cancel exactly in the normalized ln-autocorrelation, leaving `lambda_rate` alone to set the tempo — there is no locality/rate degeneracy to break. For a spot carrying several dyes (§6.6) the recorded intensity is a sum of exponentiated processes; the logarithm of that sum is not a single-dye OU process, and its normalized autocorrelation depends on `sigma_pc` as well as on `lambda_rate`. The stationarity of the law and the exactness of the correlation-decay semantics are verified mechanically by the brightness stationarity audit (`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Brightness_Stationarity_Audit.py`), whose acceptance suite also demonstrates, as its positive control, the occupancy drift of the finite-state jump-chain alternative it supersedes.

**Setting the rate from the data.** Organic-dye emission flickers: the per-frame brightness of a single label fluctuates on a photophysical timescale set by the dye and its environment (Dempsey et al. 2011, *Nat. Methods* 8:1027–1036; Ha & Tinnefeld 2012, *Annu. Rev. Phys. Chem.* 63:595–617). The `lambda_rate` prior is anchored to reproduce that timescale, measured directly from the experimental recordings. For each MET track the log of the ThunderSTORM `intensity[photon]` series is linearly detrended — removing bleaching and the per-emitter mean, both multiplicative and hence additive in the log — and a temporal autocorrelation is formed and pooled over tracks. Its lag-0→1 drop is per-localization fit noise (white, hence confined to lag 0), and the decay of the remainder is the physical flicker correlation time: `tau_corr(1/e) ≈ 0.135 s` (Fab) / `0.145 s` (InlB). The bare closed form `lambda_rate = 1/tau_corr ≈ 7` overestimates the rate, because the per-track linear detrend removes low-frequency power and shortens the apparent correlation time on finite tracks; the derivation therefore simulates OU trajectories cut to the empirical track-length distribution and detrended identically, and matches the early autocorrelation shape, giving `lambda_rate = 5.1` (Fab) / `4.7` (InlB) — condition-independent within the cell-to-cell scatter, as a photophysical quantity should be (the video-level calibration on the experimental recordings, reported in §8, instead resolves a condition-dependent rate). The condition-independence claim also admits a finer check than the pooled fit: the temporal analysis resolves the inferred rate per window of each recording, so a systematic within-recording trend would show there even where the pooled per-condition values agree. The matched model arm is a single-emitter process: each simulated trace is one dye's ln-brightness, cut and detrended like a track, so the derivation corrects the finite-track detrending bias and nothing else. The measured tracks are spots, and under the FAB law a spot may carry several dyes (§6.6), whose summed intensity has the autocorrelation dependence stated above; the derived `≈ 5` is therefore a single-dye-equivalent reference under the single-emitter assumption, not a measurement that accounts for dye multiplicity.

`lambda_rate` therefore carries a log-uniform prior `(0.0, 1.0)` (linear `1–10`), bracketing the measured `≈5`. The derivation is reproducible from the public localization tables alone by the committed utility `Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Flicker_Rate_Derivation.py`.

**Imaging priors from the data.** The learnable imaging priors bracket representative values obtained by single-molecule localization on the experimental recordings (§6.7), placed so the calibration is free to move within them: the median brightness `mu_pc` prior `(2.0, 2.75)`, the brightness shape `sigma_pc` prior `(−0.75, 0)`, the PSF median `mu_r` prior `(0, 0.3)`, and the PSF shape `sigma_r` prior `(−1, −0.25)`. Both log-spread (shape) bands exclude the high-spread regime, where the localization fits' overestimate of the true emitter-to-emitter spread would dominate (§6.7); each stays broad enough to contain its fitted spread, leaving the calibration likelihood to locate the true value within the prior. Recovery of the imaging parameters is quantified only on the held-out synthetic EVAL namespace — the experimental recordings have no ground truth. The gain enters as the single identifiable ratio `gamma` (prior `(1.62, 1.625)`, tightly anchoring the nominal `g/C = 200/4.78 ≈ 41.84`).

### 6.6 Emitters are dyes: labeling and brightness summation

The renderer's emitters are dyes, not particles. Every receptor subunit carries an integer dye count `kappa` drawn once per recording from the condition's measured labeling law (`labeling.py`: MET-INLB Bernoulli with labeling probability 0.5, one engineered attachment site per InlB probe; MET-FAB Poisson at the measured mean of 1.64 dyes per Fab, with matched-mean dispersion alternatives for sensitivity runs) and held fixed, because dye conjugation happened during sample preparation. The dye count travels with its subunit through fusion, fission, and conversion via the subunit lineage the DLI stage replays from the reaction records. A subunit with no dye has no emitter and never renders; a dimer's dyes render at one position, so a visible spot's per-frame brightness is the **sum of independent per-dye brightness processes** — each drawn from the same log-normal single-dye law (`mu_pc`, `sigma_pc`) with its own flicker trajectory (§6.5). For independent emitters, brightness distributions combine by convolution: the intensity distribution of an *n*-dye spot is the *n*-fold convolution of the single-dye distribution (Mutch et al. 2007, *Biophys. J.* 92(8):2926–2943), the additivity that underlies number-and-brightness analysis (Digman, Dalal, Horwitz, Gratton 2008, *Biophys. J.* 94(6):2320–2332).

Three consequences follow from the law, none an assumption of its own. Under the bare MET-INLB law a dimer carries zero, one, or two dyes with probabilities 25/50/25%, so among visible dimers two thirds carry one dye and one third two: the visible-dimer brightness is a mixture of one-dye and two-dye emission, not a doubled monomer. A monomer is visible with probability 0.50 and a dimer with 0.75 under the bare law, so the visible population is dimer-enriched relative to the true composition and every count parameter is a true receptor abundance. In the rendered videos the law is composed with the condition's declared probe occupancy (MET-INLB 0.5 declared; MET-FAB 0.155, derived from the declared Fab/InlB visibility ratio 0.5 and the InlB anchor; `PROJECT_CONTEXT.md` §2, *How the prior ranges and the declared inputs are set*), so a subunit is visible with `a = p_occ × P(dye ≥ 1)` = 0.25 (MET-INLB) or 0.125 (MET-FAB), a dimer with `1 − (1 − a)²`, and two-dye dimers are `a / (2 − a)` = 14 % (MET-INLB) or 6.7 % (MET-FAB) of the visible dimers; both DLI stages apply the condition's value by default, `--occupancy` is an explicit override for sensitivity runs, and the `Labeling_Set` records the value applied (`occupancy_monomer`, `occupancy_dimer`). And a one-dye dimer that dissociates leaves one visible and one permanently invisible daughter — the kinetic track disappearance of §5. A fixed brightness multiplier for dimers would misstate both the mean (a half-labeled dimer is as bright as a monomer) and the tail (doubling a single draw quadruples its variance where summing two independent draws doubles it); the summation needs no multiplier and no dimer-specific code path. Perfect independence of co-located dyes is an idealization — quenching or energy transfer between nearby dyes would add correlation the sum omits — so the sum is the first-order model, with a bounded within-probe sensitivity reserved for the model-comparison gates.

### 6.7 Imaging values from single-molecule localization data, and their caveats

The imaging parameters are calibration targets — the Detector infers them within the priors of §6.2. Independently of that inference, representative values for the emitter-brightness and PSF-width populations and the emitter density were obtained by single-molecule localization on the experimental recordings, to (i) confirm the forward model can reproduce a experimental recording at plausible settings — a fixed-imaging posterior-predictive check on the pixel-intensity histogram and its quantiles — and (ii) place the priors sensibly. These are validation and prior-placement inputs, **not** deployed parameters.

**Derivation.** The values come from the single-molecule localization tables distributed with the public accession `S-BSST712` (BioImage Archive / EBI BioStudies; `Fab.zip`, `InlB.zip`; the MET single-molecule-tracking study of Harwardt et al. 2017, *FEBS Open Bio* 7(9):1422–1440, in which Fab labels the resting receptor and InlB the internalin-B-activated one), which package — per coverslip and cell — a ThunderSTORM (Ovesný et al. 2014, *Bioinformatics* 30(16):2389–2390) localization table (`…/tracks/<cell>.csv`) and its processing protocol (`<cell>-protocol.txt`). The protocol fixes the localization: wavelet (B-spline) detection at a `std(Wave.F1)` threshold, an Integrated-Gaussian point-spread-function fit by maximum likelihood, multi-emitter fitting (up to three emitters per region), and a photon-count acceptance window of 73–1225 photons — a floor and ceiling that censor the photon distribution in both conditions. Pooling each condition over all its cells, a log-normal fit (maximum likelihood, location fixed at zero) is taken of the `intensity [photon]` column → brightness (`mu_pc`, `sigma_pc`), and of the `sigma [nm]` column → PSF width, after `√2·σ[nm] / 158 nm` converts the fitted Gaussian width to the model's `√2·σ` pixel convention (`mu_r`, `sigma_r`); localizations per `frame` over the ROI area (`<cell>_size.csv`) give an apparent density, and the per-localization `uncertainty_xy [nm]` column is the fitting error caveat 1 rests on. The two conditions fit to:

| condition | brightness median | `sigma_pc` | PSF σ median | `mu_r` | `sigma_r` |
|---|---|---|---|---|---|
| Fab (monomer control) | ≈ 386 photons | 0.61 | 152 nm | 1.36 | 0.37 |
| InlB (activating dimer) | ≈ 690 photons | 0.55 | 164 nm | 1.47 | 0.42 |

The monomer-control (Fab) medians lie within their §6.2 priors and are the ones the model adopts; the log-spreads are treated under caveat 1, and the doubled InlB brightness under caveat 3. No scripts or bundled data are needed to reproduce this — the tables are public and the fit is a location-zero log-normal on the two named columns.

Three caveats govern how these values are used.

**Caveat 1 — the fitted log-spreads overestimate the true dispersion.** Each per-spot photon count and PSF width is itself a noisy fit: the localizer estimates it from a single shot-noise-limited spot and reports a per-localization fitting uncertainty. The variance of a set of noisy estimates is the true variance plus the fitting-error variance (standard deviations add in quadrature), so a log-spread read off the fits overstates the true emitter-to-emitter spread — the classical additive (errors-in-variables) measurement-error inflation of an estimated variance. The fitted spreads (`sigma_r ≈ 0.37`, `sigma_pc ≈ 0.61`) are therefore upper-biased. In the forward model an inflated `sigma_r` manufactures an excess of anomalously narrow — and hence disproportionately bright — spots, the direct cause of an unphysically heavy pixel-intensity tail. The §6.2 priors respond by bounding both shape bands away from the high-spread regime where the tail runs away — the `sigma_r` band capped at 0.56 and the `sigma_pc` band at 1.0 — while keeping each broad enough to contain its fitted spread (`0.37`, `0.61`), so the calibration likelihood, which fits the experimental videos, pulls the inferred spread down toward its true, fit-noise-corrected value. The fixed-imaging posterior-predictive histogram check (validation input i above) locates that fit-corrected value directly: the experimental pixel-intensity histogram is reproduced at `sigma_r ≈ 0.15` and `sigma_pc ≈ 0.5` — both well below the upper-biased fits (`0.37`, `0.61`) and comfortably inside their priors — so these are the operating spreads a full calibration is expected to recover, not the raw fitted values. (The fits are pooled per condition, and the two are consistent under this correction: Fab `sigma_r` `0.37` / InlB `0.42`, Fab `sigma_pc` `0.61` / InlB `0.55`.)

**Caveat 2 — localization undercounts the true emitter density.** Dim and spatially overlapping emitters are missed, so localizations per frame are a lower bound on the true number present. Where a synthetic recording is matched to an experimental one, the apparent density is scaled up to approximate the true count; taken at face value it under-populates the field.

**Caveat 3 — the dimer condition's photon values are doubled by co-located labels.** InlB's brightness median (≈ 690 photons) is about 1.8× Fab's (≈ 386), but this is a localization artifact, not a per-emitter property: an activated receptor dimer carries two labels within one diffraction-limited spot, which the single-emitter fit reports as one detection whose photons sum. The ratio is in fact a lower bound — the 73–1225 photon acceptance window is a shared setting, but it bites the dimer condition far harder: 23.7% of InlB localizations pile up at the 1225 ceiling (against 3.0% for Fab), truncating the very tail the doubling produces, so the true per-spot ratio is ≈ 2×. The emitter-brightness population (`mu_pc`, `sigma_pc`) is therefore taken from the monomer control, where a spot is one label; the dimer's brightness is then built by the sum model (§6.6) — two independent monomer draws — rather than read from the doubled, clamped InlB localizations. Under the measured MET-INLB labeling only one third of the visible dimers carry two dyes, so co-location alone predicts a smaller per-spot ratio than complete labeling would; the remainder is the empirical tension between the summation law and the single visible-dimer brightness population that the model-comparison gates carry as an acceptance constraint.

Across all three, the reliable quantities are the **medians** (`mu_pc`, `mu_r`) from the monomer control — a median is far more robust to per-spot fitting noise than a spread — while the **spreads** and the **density** are the biased quantities the caveats correct for. This is the emitter-signal analog of the background-side caution in `REFERENCE_EMCCD_NOISE_MODEL.md` §5, that post-detection localization summaries are not latent model parameters.

**A stated assumption beside the caveats: probe kinetics.** The observation layer holds every dye count fixed for the recording, so it removes a dye only by photobleaching and never adds one; ligand binding and unbinding within a recording are not modeled, and for MET-INLB the probe *is* the ligand. The unbinding channel is absorbed by construction: this workflow calibrates `prob_photo_bleach` per condition on the condition's own recordings, so the INLB value is photobleaching plus first-order unbinding, and the biology DLI draws that value — a first-order, state-independent unbinding is statistically indistinguishable from the modeled bleaching. Two residuals are declared rather than modeled: a labeled InlB binding from solution during a recording would create a spot the simulator never creates (relevant only if free labeled ligand was present during imaging, a property of the source acquisition), and a ligand affinity that differs between monomeric and dimeric MET would make disappearances stoichiometry-dependent, which a state-independent bleach cannot absorb. Partial ligand occupancy is the declared per-condition probe occupancy of §6.6 (MET-INLB 0.5, provisional until the collaborators' answers), applied by default and overridable with `--occupancy` for a sensitivity run. The posterior-predictive video comparison is where a violated assumption shows, as appearances in the experimental clip with none in the synthetic one.

### 6.8 Calibration runs — identity and metric definitions

Subsections 6.8–6.11 record measurements, not conclusions. Every number is either copied from a report the
pipeline wrote or computed separately from the draws those reports saved; each table says which. The report
folders live in the Data_Bank `Posit/` tier under the names given; the reports' own threshold labels ("ok",
"check", "calibrated") are not reproduced here. Two estimators are compared throughout: the **multiple-dye**
estimator trained under the Poisson labeling law of §6.6, and the **one-dye** estimator of §6.10, trained on the
same trajectories under a Bernoulli law that puts exactly one dye on every visible subunit.

| | multiple-dye | one-dye |
|---|---|---|
| condition, labeling law | MET-FAB (`FAB`); `FAB_POISSON`, mean 1.64 dyes per Fab; occupancy 0.155 | MET-FAB (`FAB`); `FAB_BERNOULLI`, `q = 1 − e^{−1.64} = 0.806`; occupancy 0.155 |
| recordings | 2 s at 50 fps (100 frames), 256 × 256 px, 8-bit; imaging from the §6.2 prior, camera from the §6.3 box, biology from the FAB trajectory tier | identical, file for file |
| estimator artifact | `…_DETECTOR_FAB_2S_50FPS_Estimator.npz`, weights SHA-256 `363a2614…`, torch 2.9.1 | `…_ONEDYE_DETECTOR_FAB_2S_50FPS_Estimator.npz`, weights SHA-256 `87b77f7d…`, torch 2.9.1 |
| training | 200,000 TRAIN / 50,000 TEST videos; 100 epochs over two chained runs (50 + a 50-epoch resurrect); best mean test loss −5.693 at global epoch 87 | same data and architecture; 50 epochs + a resurrect stopped at global epoch 93 after 38 epochs without improvement; best mean test loss −10.959 at global epoch 55 |
| Evaluation (`…_MAP_Recovery`) | 25 EVAL tasks, 25,000 videos, bounded pool, 1,000 draws per video; report 2026-09-17 18:04 UTC | same settings on the one-dye EVAL tier; report 2026-09-18 17:01 UTC |
| Posterior_Calibration | the same 25,000 videos and draws; report 2026-09-17 12:23 UTC | report 2026-09-18 16:17 UTC |
| Experiment (`…_MAP_Experiment`) | 60 MET-FAB recordings, 600 windows, unrestricted pool; report 2026-09-17 12:02 UTC | the same recordings and windows; report 2026-09-18 16:18 UTC |
| dye-multiplicity stratification | separate calculation on the saved draws and the EVAL `Labeling_Set`; folder `…_DETECTOR_FAB_2S_50FPS_Dye_Multiplicity_Stratification` | not applicable (one dye per visible subunit) |

**Metric definitions.** Three point estimates are reported for every per-video posterior and read
together, here and in every Evaluation and Experiment report: the MAP, the optimizer's mode; the
posterior median, the 50 % quantile of each marginal over the 1,000 draws; and the sample geometric
median (SGM), the draw closest in prior-scaled `log10` distance to all other draws. No single one of
the three is the estimate; every recovery statistic below is given for all three, and a conclusion
rests on the set. Where the three disagree, the disagreement is reported as a disagreement between
the optimized mode and the posterior summaries; what produces it is a separate question, settled by
its own checks rather than by the tables (§9.8). *corr* is the Pearson correlation
across videos between the estimate's `log10` value and the true `log10` value. *MAE* is the mean
absolute error of the estimate in `log10` units; *median error* is the median signed error (estimate
minus truth, `log10`), so its sign is the direction of the offset; *within ±0.15* is the share of
videos whose absolute error is at most 0.15 dex (a factor 1.41); *outside prior* is the share of
estimates beyond the prior box. *Marginal coverage at
c* is the share of videos whose true value lies inside the central `c` interval of that marginal,
read from quantiles. *Joint coverage at c* is the share of videos whose true parameter vector has a
flow log-density above the `1 − c` quantile of the log-densities of that video's own draws. The
*standardized error* is `z = (truth − median) / posterior sd`; *bias* is its mean and *spread* its
standard deviation over videos, so a well-calibrated marginal has bias 0 and spread 1; *sharpness* is
the posterior standard deviation as a fraction of the prior width. *SBC KS D* is the largest deviation
of the rank CDF of the truth among the draws from uniform; *TARP ATC* is the area between the expected
coverage curve and the diagonal, negative when the posterior is too narrow; the *L-C2ST reject
fraction* is the share of 1,000 observations at which a local classifier rejects calibration at
α = 0.05.

### 6.9 Multiple-dye calibration outcome (synthetic)

> **MAP numbers here are under recomputation (§9.8).** The MAP routine returned coordinates one
> optimizer step away from the point whose density it reported; the defect is fixed in 0.1.15 and the
> affected stages are being re-run. The posterior-median and SGM columns, and every sampling-based
> calibration result, are unaffected.


**Per parameter (from the Evaluation report), MAP view.**

| parameter | corr | MAE (log10) | median error (log10) | within ±0.15 | outside prior | marginal coverage 50 % / 90 % |
|---|---|---|---|---|---|---|
| `mu_r` | 0.67 | 0.079 | +0.024 | 90 % | 26 % | 23 % / 59 % |
| `sigma_r` | 0.08 | 0.195 | −0.051 | 41 % | 0 % | 34 % / 76 % |
| `mu_pc` | 0.87 | 0.095 | +0.037 | 80 % | 8 % | 40 % / 80 % |
| `sigma_pc` | 0.79 | 0.112 | −0.009 | 71 % | 6 % | 42 % / 85 % |
| `prob_photo_bleach` | 0.77 | 0.214 | +0.033 | 47 % | 1 % | 48 % / 87 % |
| `lambda_rate` | 0.46 | 0.214 | +0.020 | 41 % | 1 % | 42 % / 83 % |

Marginal coverage is a property of the posterior, not of the point estimate, and is the same in the
three views.

**Posterior-median view**, the same statistics for the 50 % quantile of each marginal.

| parameter | corr | MAE (log10) | median error (log10) | within ±0.15 | outside prior |
|---|---|---|---|---|---|
| `mu_r` | 0.96 | 0.025 | +0.018 | 100 % | 0 % |
| `sigma_r` | 0.17 | 0.186 | −0.020 | 41 % | 0 % |
| `mu_pc` | 0.95 | 0.054 | +0.021 | 94 % | 0 % |
| `sigma_pc` | 0.89 | 0.082 | +0.006 | 86 % | 0 % |
| `prob_photo_bleach` | 0.79 | 0.206 | −0.006 | 48 % | 0 % |
| `lambda_rate` | 0.54 | 0.201 | −0.002 | 43 % | 0 % |

**Sample-geometric-median view**, the same statistics for the SGM of the same draws.

| parameter | corr | MAE (log10) | median error (log10) | within ±0.15 | outside prior |
|---|---|---|---|---|---|
| `mu_r` | 0.96 | 0.025 | +0.018 | 100 % | 0 % |
| `sigma_r` | 0.15 | 0.186 | −0.021 | 41 % | 0 % |
| `mu_pc` | 0.95 | 0.056 | +0.021 | 94 % | 0 % |
| `sigma_pc` | 0.89 | 0.083 | +0.008 | 86 % | 0 % |
| `prob_photo_bleach` | 0.78 | 0.210 | −0.010 | 46 % | 0 % |
| `lambda_rate` | 0.53 | 0.202 | +0.001 | 42 % | 0 % |

**Point-estimate agreement.** The MAP falls outside the central 90 % interval of its own posterior in
56 % of videos for `mu_r`, 44 % for `mu_pc`, and 22 % for `sigma_pc` (2 % or less for the other
three), with median |MAP − median| gaps of 0.119, 0.092 and 0.071 dex on those three; the sample
geometric median and the per-dimension median agree within 0.004–0.028 dex on every parameter. On
those three parameters the MAP recovers the truth worse than the two posterior summaries on the same
videos (`mu_r` correlation 0.67 against 0.96, 26 % of MAP estimates outside the prior box under the
bounded pool against none). That gap was read as the optimizer landing in flow density spikes away from
the posterior mass. It is not: §9.8 traces it to a defect in the MAP routine itself, corrected in
0.1.15, and the numbers in this paragraph are under recomputation.

**Joint and standardized (from the Posterior_Calibration report).** Joint coverage 0.24 at nominal 0.50 and
0.62 at nominal 0.90, largest gap 0.31 (at nominal 0.75); TARP ATC −0.05; L-C2ST reject fraction 0.998.

| parameter | SBC KS D | standardized bias | standardized spread | sharpness (% of prior width) |
|---|---|---|---|---|
| `mu_r` | 0.447 | +1.22 | 1.33 | 5.3 |
| `sigma_r` | 0.121 | −0.11 | 1.29 | 22.3 |
| `mu_pc` | 0.178 | +0.53 | 1.17 | 6.7 |
| `sigma_pc` | 0.061 | +0.02 | 1.09 | 11.6 |
| `prob_photo_bleach` | 0.036 | +0.08 | 1.02 | 16.2 |
| `lambda_rate` | 0.053 | −0.03 | 1.12 | 20.8 |

**Results from separate calculations on the saved draws.** The `mu_r` truth lies below the marginal
90 % interval in 39 % of videos and above it in 2 %. Among the 14,737 videos whose `mu_r` truth lies
inside its marginal 90 % interval the joint 90 % coverage is 0.83; among the remaining 10,263 it is
0.31. Stratifying the 25,000 videos into ten equal-count bins of the realized number of dyes per
labeled subunit (10th to 90th percentile 1.90 to 2.17; the join to the `Labeling_Set` inverts the
Evaluation stage's round-robin sharding and reproduces the recorded truths exactly): the
posterior-median error of `mu_pc` rises monotonically from +0.014 dex in the lowest bin to +0.087 dex
in the highest (difference 0.073 ± 0.002 dex; Pearson correlation of the signed error with the dye
count +0.31); `lambda_rate` changes by −0.038 ± 0.007 dex from lowest to highest bin; `sigma_r`
changes by +0.016 ± 0.006 dex with a truth-tracking correlation of 0.15 in the lowest and 0.13 in the
highest bin; `mu_r` is non-monotonic with a −0.005 dex end-to-end difference.

**Limits.** All results are on synthetic data drawn from the same generator the estimator was trained
on; nothing here measures transfer to experimental recordings, which have no ground truth. The
causes of the two clearest features are unresolved: the constant positive `mu_r` offset in a very
narrow posterior, and the flat `sigma_r` estimate that does not follow the truth while its interval
stays narrow. The dye-multiplicity stratification varies only the realized *mean* dye count per spot
between videos; the within-video *variance* of the dye count is fixed by the law and identical in
every video, so that analysis carries no leverage on any mechanism acting through that variance, and
its null result for `sigma_r` is uninformative rather than exculpatory. The `mu_pc` trend is a
measured association within the sampled range; extrapolating it to one dye per spot is not
supported by these data.

### 6.10 One-dye comparison

> **MAP numbers here are under recomputation (§9.8).** The MAP routine returned coordinates one
> optimizer step away from the point whose density it reported; the defect is fixed in 0.1.15 and the
> affected stages are being re-run. The posterior-median and SGM columns, and every sampling-based
> calibration result, are unaffected.


**Design.** A sensitivity branch (`one-dye-sensitivity`, version 0.1.10,
product namespace `…_ALP_ONEDYE_…`) regenerates the DLI products with the labeling law
`FAB_BERNOULLI`, Bernoulli with `q = 1 − e^{−1.64} = 0.806`, so that a labeled Fab is visible with the
same probability as under the Poisson law (derived occupancy 0.1551 and visible fraction 0.125
unchanged) and every visible subunit carries exactly one dye. Everything else is identical: the same
FAB trajectory tier file for file, 200/50/25 tasks, architecture, epochs, batch, and analysis
settings. The one-dye EVAL labeling records verify the intent (dyes equal labeled subunits in every
one of 3,000 checked recordings, against a mean ratio of 2.03 under the Poisson law). The comparisons
were specified on 2026-09-17, before any one-dye data existed: the `sigma_r` truth-tracking
correlation against 0.17, the `lambda_rate` MAE against 0.201, and whether the +0.02 dex `mu_r` offset
persists. Their thresholds indicate recovery improvements under a changed labeling law; they do not
establish identifiability in general, a mechanism, or the validity of any replacement measurement
(§9.4).

**Synthetic results (from the one-dye Evaluation and Posterior_Calibration reports; multiple-dye values from §6.9
beside them).** All three pre-specified comparisons resolve in the one-dye direction. `prob_photo_bleach` is the
exception in both tables — unchanged — and it is the parameter for which the information budget's approximate
precision benchmark (§9.5), an estimated standard deviation in log10 units set against the prior width, is by far
the largest at 2 s.

*MAP view.*

| parameter | corr | MAE (log10) | median error (log10) | within ±0.15 | outside prior | marginal coverage 50 % / 90 % |
|---|---|---|---|---|---|---|
| `mu_r` | 0.67 → **0.95** | 0.079 → **0.016** | +0.024 → +0.005 | 90 % → 100 % | 26 % → 2 % | 23/59 → 48/89 % |
| `sigma_r` | 0.08 → **0.97** | 0.195 → **0.040** | −0.051 → −0.008 | 41 % → 98 % | 0 % → 1 % | 34/76 → 52/91 % |
| `mu_pc` | 0.87 → 0.99 | 0.095 → 0.020 | +0.037 → −0.003 | 80 % → 100 % | 8 % → 1 % | 40/80 → 58/94 % |
| `sigma_pc` | 0.79 → 0.98 | 0.112 → 0.036 | −0.009 → −0.020 | 71 % → 99 % | 6 % → 1 % | 42/85 → 43/85 % |
| `prob_photo_bleach` | 0.77 → 0.79 | 0.214 → 0.209 | +0.033 → +0.064 | 47 % → 50 % | 1 % → 0 % | 48/87 → 45/85 % |
| `lambda_rate` | 0.46 → 0.94 | 0.214 → **0.088** | +0.020 → +0.052 | 41 % → 82 % | 1 % → 2 % | 42/83 → 41/84 % |

*Posterior-median view.* Marginal coverage is a property of the posterior and repeats the MAP table.

| parameter | corr | MAE (log10) | median error (log10) | within ±0.15 |
|---|---|---|---|---|
| `mu_r` | 0.96 → **0.99** | 0.025 → **0.011** | +0.018 → +0.005 | 100 % → 100 % |
| `sigma_r` | 0.17 → **0.98** | 0.186 → **0.036** | −0.020 → −0.006 | 41 % → 99 % |
| `mu_pc` | 0.95 → 1.00 | 0.054 → 0.016 | +0.021 → −0.003 | 94 % → 100 % |
| `sigma_pc` | 0.89 → 0.99 | 0.082 → 0.032 | +0.006 → −0.018 | 86 % → 100 % |
| `prob_photo_bleach` | 0.79 → 0.80 | 0.206 → 0.205 | −0.006 → +0.052 | 48 % → 50 % |
| `lambda_rate` | 0.54 → 0.94 | 0.201 → **0.083** | −0.002 → +0.045 | 43 % → 84 % |

*Sample-geometric-median view.*

| parameter | corr | MAE (log10) | median error (log10) | within ±0.15 |
|---|---|---|---|---|
| `mu_r` | 0.96 → **0.98** | 0.025 → **0.012** | +0.018 → +0.005 | 100 % → 100 % |
| `sigma_r` | 0.15 → **0.97** | 0.186 → **0.038** | −0.021 → −0.006 | 41 % → 99 % |
| `mu_pc` | 0.95 → 0.99 | 0.056 → 0.017 | +0.021 → −0.003 | 94 % → 100 % |
| `sigma_pc` | 0.89 → 0.99 | 0.083 → 0.033 | +0.008 → −0.018 | 86 % → 100 % |
| `prob_photo_bleach` | 0.78 → 0.80 | 0.210 → 0.205 | −0.010 → +0.050 | 46 % → 50 % |
| `lambda_rate` | 0.53 → 0.94 | 0.202 → 0.084 | +0.001 → +0.045 | 42 % → 84 % |

*Point-estimate agreement, one-dye.* The MAP falls outside its own central 90 % interval in 5 % or fewer
of videos on every parameter (multiple-dye: 56 %, 44 % and 22 % on `mu_r`, `mu_pc` and `sigma_pc`), and
the three estimates agree within 0.031 dex; under the one-dye law the three views coincide.

Each cell reads multiple-dye → one-dye; bold marks the three pre-specified comparisons. Joint coverage 0.51 at
nominal 0.50 and 0.89 at nominal 0.90 with largest gap 0.0095 (multiple-dye 0.24, 0.62, 0.31); TARP ATC −0.02
(−0.05); L-C2ST reject fraction 1.000 (0.998) — at 25,000 observations this test detects any deviation and does
not size it.

| parameter | SBC KS D | standardized bias | standardized spread | sharpness (% of prior width) | physical bias |
|---|---|---|---|---|---|
| `mu_r` | 0.447 → 0.191 | +1.22 → +0.42 | 1.33 → 0.91 | 5.3 → 4.6 | 1.013× |
| `sigma_r` | 0.121 → 0.059 | −0.11 → −0.16 | 1.29 → 0.95 | 22.3 → 6.2 | 1.018× |
| `mu_pc` | 0.178 → 0.081 | +0.53 → −0.08 | 1.17 → 0.84 | 6.7 → 3.1 | 1.004× |
| `sigma_pc` | 0.061 → 0.267 | +0.02 → −0.66 | 1.09 → 0.94 | 11.6 → 4.3 | 1.050× |
| `prob_photo_bleach` | 0.036 → 0.118 | +0.08 → +0.36 | 1.02 → 1.12 | 16.2 → 14.4 | 1.196× |
| `lambda_rate` | 0.053 → 0.250 | −0.03 → +0.60 | 1.12 → 0.98 | 20.8 → 8.3 | 1.121× |

The one-dye standardized spreads sit near 1 and the multiple-dye spreads above it. The larger one-dye KS
discrepancies for `sigma_pc`, `prob_photo_bleach` and `lambda_rate` coexist with systematic standardized offsets.
Posterior sharpness also changes, substantially for the brightness spread and the flicker rate but much less for
bleaching. The closing block below states what these summaries do and do not show.

**Test-loss distributions (from the two `Test_Loss_Distribution_Analysis` reports).** Over 50,000 TEST videos
each — independent draws of θ from the same prior, so an unpaired comparison — the mean NLL is −5.69
[−5.73, −5.66] for the multiple-dye estimator and −10.96 [−10.98, −10.94] for the one-dye estimator, against a
uniform-prior baseline of −1.66; the 95th percentile is +1.84 against −6.56; the share of videos with NLL above
zero is 8.3 % against 0.15 %. The hardest 5 % of both estimators is marked by low `mu_pc` (KS D 0.47 against 0.35).

**What the comparison establishes, and what it does not.** It establishes two things. First, the labeling law
strongly affects inference performance: with the trajectory tier, architecture, training budget, and analysis
settings held identical, five of the six parameters recover markedly better under the one-dye law and the joint
credible regions are near nominal in aggregate. Second, the multiple-dye estimator undercovers: its nominal joint
90 % region contains the truth in only 62 % of the evaluation recordings. Its standardized-error summaries also
indicate dispersion and location discrepancies. That is a defect whatever the recordings contain, because less
informative recordings should yield broader posteriors, not confident ones that miss.

It does not establish that the multiple-dye failures are estimator-side only. Changing the dye-count law changes
both the information the recordings carry and the difficulty of learning the posterior from them, and this design
cannot separate the two. There is an inference problem to fix; better training need not reach the one-dye precision.

It does not establish that `prob_photo_bleach` cannot be recovered from a 2 s recording. Both estimators return a
mean absolute error near 0.205 dex, but the one-dye estimate tracks the truth with correlation 0.80: limited
precision, not absence of information. The information budget (§9.5) provides an approximate precision benchmark
for an unbiased decay-rate estimator using total fluorescence; it does not establish a fundamental recovery limit
for inference from the full video. The conclusion retained is the practical one: bleaching is recovered
substantially less precisely than the other parameters in 2 s windows and should preferentially be constrained
from longer recordings, subject to validating that measurement.

It does not establish that the one-dye estimator is calibrated per parameter. Near-nominal aggregate joint coverage
coexists with systematic marginal biases (standardized offsets of 0.36 to 0.66 on four parameters) and
parameter-dependent undercoverage (marginal 90 % coverage of 85 % for `sigma_pc`, 85 % for `prob_photo_bleach`,
and 84 % for `lambda_rate` in the recovery report). Reading the second table as a location error in one estimator
and a width error in the other is a plausible account, not a demonstrated decomposition: marginal rank
uniformity and joint coverage both respond to bias, dispersion, and shape, and near-nominal pooled coverage does
not guarantee accurate uncertainty within individual recordings or parameter regimes (Modrák et al. 2023). The
physical-bias factors are conversions from standardized summaries, diagnostic approximations rather than measured
mean offsets.

It does not validate the one-dye model for the experimental recordings, nor decide either experimental estimate
(§6.11). `mu_pc` keeps its definition, the per-dye brightness, under both laws; the laws map spot brightness onto
it differently, so the disagreement between 139 and 186 photons is model-dependent inference, not a difference in
definitions. Agreement on `sigma_r` and `prob_photo_bleach` does not make the labeling law irrelevant to them:
their observations still depend on multiplicity, and two estimators can agree while both are inaccurate. The
measured mean of 1.64 dyes per probe is data; the Poisson form of the dye-count law is a modeling assumption.

**Standing decision.** The multiple-dye model remains the working baseline. Before a direct estimator replaces a
neural inference target, it must be validated on multiple-dye synthetic recordings spanning the relevant
brightness, multiplicity, bleaching, and track-length conditions; the few-scene self-tests of the companion notes
are not that validation. Those validated measurements then decide which imaging parameters can leave the neural
inference block (§9.4). Bleaching is the strongest candidate for longer-recording analysis. The direct flicker
estimator is a cross-check to be run after its validation, and its result informs, but does not alone decide, the
choice between an improved neural estimator and a hybrid.

### 6.11 Both estimators on the experimental recordings

> **MAP numbers here are under recomputation (§9.8).** The MAP routine returned coordinates one
> optimizer step away from the point whose density it reported; the defect is fixed in 0.1.15 and the
> affected stages are being re-run. The posterior-median and SGM columns, and every sampling-based
> calibration result, are unaffected.


From the two Experiment reports (60 MET-FAB recordings, 600 windows, `--pool-mode unrestricted`); the three
per-window point estimates pooled over recordings and windows, medians and interquartile ranges in log10, the
median in absolute units, and the share of estimates outside the prior box. The multiple-dye run predates the
SGM output (0.1.8) and stored no draw cloud, so its SGM does not exist; its posterior median is computed from
its stored quantiles.

*MAP view.*

| parameter | multiple-dye: median (IQR) | absolute | outside prior | one-dye: median (IQR) | absolute | outside prior |
|---|---|---|---|---|---|---|
| `mu_r` | +0.316 (0.145) | 2.07 | 64 % | +0.224 (0.035) | 1.68 | 4 % |
| `sigma_r` | −0.803 (0.110) | 0.157 | 0 % | −0.831 (0.137) | 0.148 | 0 % |
| `mu_pc` | +2.143 (0.171) | 139 photons | 10 % | +2.269 (0.158) | 186 photons | 1 % |
| `sigma_pc` | −0.194 (0.173) | 0.64 | 0 % | −0.139 (0.112) | 0.73 | 0 % |
| `prob_photo_bleach` | −1.380 (0.547) | 0.042 | 1 % | −1.374 (0.599) | 0.042 | 1 % |
| `lambda_rate` | +0.719 (0.175) | 5.23 | 2 % | +0.361 (0.219) | 2.30 | 0 % |

*Posterior-median view.*

| parameter | multiple-dye: median (IQR) | absolute | outside prior | one-dye: median (IQR) | absolute | outside prior |
|---|---|---|---|---|---|---|
| `mu_r` | +0.302 (0.030) | 2.01 | 52 % | +0.224 (0.035) | 1.68 | 3 % |
| `sigma_r` | −0.782 (0.053) | 0.165 | 0 % | −0.833 (0.137) | 0.147 | 1 % |
| `mu_pc` | +2.129 (0.093) | 135 photons | 0 % | +2.270 (0.157) | 186 photons | 1 % |
| `sigma_pc` | −0.192 (0.040) | 0.64 | 0 % | −0.139 (0.113) | 0.73 | 0 % |
| `prob_photo_bleach` | −1.471 (0.430) | 0.034 | 1 % | −1.420 (0.539) | 0.038 | 0 % |
| `lambda_rate` | +0.705 (0.096) | 5.07 | 1 % | +0.357 (0.222) | 2.28 | 0 % |

*Sample-geometric-median view (one-dye only).*

| parameter | one-dye: median (IQR) | absolute | outside prior |
|---|---|---|---|
| `mu_r` | +0.224 (0.036) | 1.67 | 3 % |
| `sigma_r` | −0.833 (0.140) | 0.147 | 1 % |
| `mu_pc` | +2.271 (0.163) | 187 photons | 1 % |
| `sigma_pc` | −0.140 (0.111) | 0.72 | 0 % |
| `prob_photo_bleach` | −1.417 (0.540) | 0.038 | 0 % |
| `lambda_rate` | +0.359 (0.226) | 2.29 | 0 % |

*Point-estimate agreement on the recordings.* Multiple-dye: the MAP falls outside its own central 90 % interval
in 55 % of windows for `mu_r`, 46 % for `mu_pc` and 28 % for `sigma_pc`, with median |MAP − median| gaps of
0.12, 0.10 and 0.10 dex, and the `mu_r` MAP leaves the prior box in 64 % of windows against 52 % for the median.
One-dye: 1 % or less on every parameter, and the three estimates agree within 0.029 dex. The two runs differ on
every view, not only on the MAP.

The two estimators agree on `sigma_r` and `prob_photo_bleach` and disagree where the labeling law enters the
observation: `mu_r`, `mu_pc` (the same per-dye brightness, reached through two different dye-count models), `sigma_pc`, and
`lambda_rate` — **2.30 against 5.23**, a 0.36 dex disagreement. That last one is the open question this
comparison raises. The multiple-dye value matches the localization-table derivation of §6.5 (5.1); the one-dye
estimator recovers `lambda_rate` on its own simulator to 0.083 dex (§6.10); neither fact decides which describes
the recordings. The direct flicker estimator (§9.5) is the planned cross-check. Its model arm needs no fitted
brightness parameters but retains a single-dye approximation, and its self-test used four single-dye 6 s scenes with
the other imaging parameters at their prior centers and bleaching disabled; it is to be validated on multiple-dye
synthetic recordings spanning the relevant operating conditions before it is applied here. A value near either
estimate would inform, not decide, the question, and agreement with the ThunderSTORM-derived value would not
provide fully independent methodological confirmation, because both estimates rely on single-dye matching, though a
measurement from the raw frames still adds evidence. No result here is a measurement of transfer to the recordings,
which have no ground truth: agreement between two estimators is not correctness, and disagreement locates where the
labeling law matters.

---

## 7. Nuisance and artifact design

### 7.1 The marginalized blocks and their media

Each workflow infers one block and marginalizes the others, and every marginalized block is recorded per
simulation as a self-labeling record. The blocks differ in *where their truth comes from*, and that decides
what has to be built:

| block | source of truth | record token | detector | production (biology) |
|---|---|---|---|---|
| biology (eleven rows, §6.1) | the biology prior, realized once per condition as the trajectory tier (§4) | the tier's own `Theta_Set` | marginalized — re-imaged | inferred — the same file is the label |
| photophysics (six rows, §6.2) | the detector calibration on the experimental recordings | `Nuisance_DLI` artifact, recorded as `Nuisance_DLI_Theta_Set` | inferred — its posterior mints the artifact | marginalized — drawn from the artifact |
| camera (five rows, §6.3) | acquisition-protocol values, known a priori | transient `BoxUniform`, recorded as `Nuisance_SCOPE_Theta_Set` (§9.3) | marginalized | marginalized — the one block shared by both |
| a photophysics row measured directly (§9.4, proposal) | a direct estimator's measured range (§9.5) | nuisance-from-spec with that range (§5), same record token | — | marginalized over the measured range, never a point value |

The biology needs no construction: the detector re-images the condition's trajectory tier, so its draws *are*
that tier's `Theta_Set`, the biology's learnable labels read as a nuisance record. The camera needs none either:
its five ranges are the whole specification, so it is declared on the fly and only its draws are recorded. The
photophysics is the one block whose content *is* a calibration result and cannot be declared in advance, so it
is the only nuisance that exists as a persisted, samplable object. The fourth row is not a new mechanism — it is
the nuisance-from-spec role of §5 with a measured range in place of an a-priori one — and it is what §9.4
proposes for any parameter that leaves the inferred block.

### 7.2 The `Nuisance_DLI` artifact

**What it is.** A trained detector estimator is a prerequisite. The construction reads every experimental
recording, cuts each into model-length windows, draws the posterior conditioned on each window, and pools the
draws across every window of every recording into the **`posterior_sample_pool`**. That pool is a *mixture*: the
recordings genuinely differ in imaging, so the across-recording spread *is* the nuisance, and a single posterior
conditioned on all recordings at once — which would assume one shared parameter vector — is the wrong object.
One user choice, **`posterior_sample_pool_choice`**, turns the pool into the artifact (`detector_nuisance_dli.py`):

| choice | representation | cross-parameter correlations |
|---|---|---|
| `raw` (default) | resample the pool per whole vector | preserved exactly |
| `map_estimate_pool` | resample the pooled per-window MAP estimates per whole vector | preserved (point estimates) |
| `gaussian` | a full-covariance multivariate normal fit to the pool | linear only (via the covariance) |
| `box` | a per-parameter uniform over pool quantiles | none (independent per dimension) |
| `box_user` | a per-parameter uniform over user-set ranges | none |
| `sgm_percentiles` | whole actual vectors at signed distance-to-SGM percentiles -- an SGM of window MAPs under `selection_source = "experiment"`, of window posterior SGMs under `"window-sgm"`; one frozen vector (`[50]`) or a small pool | whole vectors: a multi-member pool keeps its members' co-occurring coordinates; a single frozen vector carries none |

`raw` is the faithful form: resampling whole vectors preserves the joint structure exactly, including the ridges
the calibration constrains — the ADU floor depends on the product `gamma·kappa_o`, so those two ride a joint
ridge that independent per-parameter sampling would break. `gaussian` keeps such a ridge as a linear
correlation but not multimodality or curvature; `box`/`box_user` discard correlations. `sgm_percentiles` selects
a few *whole* actual vectors at percentiles of a signed distance to the Sample Geometric Median — the
correlation-preserving median vector, an actual member of the cloud, never the vector of per-dimension medians,
which can be an impossible combination — so `[50]` freezes the imaging to one actual vector and several
percentiles give a small pool of actual acquisitions; it reuses the Experiment stage's MAP estimates and needs no
GPU. A single selected vector fixes the six photophysics parameters across generated recordings. It does not fix
the separately sampled SCOPE camera parameters (§9.3). Selecting fixed photophysics is an explicit construction
choice; it should not be described as the general biology-workflow behavior. The companion `…_Sample_Geometric_Median` analysis reports that median for inspection and mints nothing; the
construction is the authoritative source, and both call one `sample_geometric_median` implementation. `pool_mode`
(`bounded` default, `unrestricted`) follows the Evaluation and Experiment convention: bounded rejection-samples
the pool within the imaging prior box, unrestricted keeps the flow's draws wherever they fall. The
calibration-faithful forms (`raw`, `map_estimate_pool`, `gaussian`, `sgm_percentiles`) are never clipped; only
`box`/`box_user` are clamped to the prior box, at build, logged and counted. The artifact stores the numeric
parameters of its representation — a `[low, high]` box, a sample matrix, or a Gaussian mean and covariance — so
generation samples it standalone, needing neither the estimator nor the recordings. Per-parameter
`[imaging.<KEY>]` ranges are supplied if and only if the choice is `box_user`.

**Status and the decision it waits on.** No `Nuisance_DLI` has been built for this project; the biology
production run of `VALIDATION.md` §2.5b intends `raw`, and smokes use `box_user` over the full prior box as a
stand-in that needs no estimator. Two detector estimators now exist — multiple-dye and one-dye (§6.10) — and
they disagree on `lambda_rate` on the very recordings the pool would be drawn from (§6.11). **Which estimator
mints the artifact is the analyst's decision**, made explicitly at construction: the construction resolves the
estimator from the product namespace it is run under (`…_DETECTOR_…` or `…_ONEDYE_DETECTOR_…`) and writes that
estimator's weights SHA-256 into the spec's provenance comment and the cached pool (`pool_provenance`), so the
choice is visible in the artifact and never implicit in a default. The direct flicker estimator on the recordings (§6.11) is the
evidence that decision waits on.

**An empirical-Bayes choice, stated.** The pool is drawn from *all* experimental recordings, and the resulting
imaging distribution then conditions the biology pipeline applied to those same recordings. The recordings are
the only available anchor for the imaging manifold, so this is deliberate — but the downstream fit to them is
not an independent evaluation on experimental data, and no reader should take it for one. A calibration/analysis
split of the cells, cross-fitting across cells, or a leave-one-recording-out sensitivity check would restore that
independence; none is performed here.

### 7.3 The construction step

The `Nuisance_DLI` is a **user-authored, analysis-emitted artifact**, not a hardcoded table, because its source
of truth is the calibration plus a human judgment that varies per experiment. It is a post-hoc analysis
(`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py`), never a canonical stage, and it
is where the two-stage calibration's human refinement (§3) becomes explicit: the posterior is the initial guess,
the analyst decides how it feeds production. Three properties govern it.

1. **A gate, not a rebuild.** The analysis emits the spec (`…_Nuisance_DLI_Spec.toml`, values in log10; a spec
   cannot exist without having run it), validates it at build — structure, a valid choice and `pool_mode`, and
   for `box_user` ranges inside the imaging prior box of §6.2 — and builds the artifact (`…_Nuisance_DLI.npz`).
   Consumers (the production import; the matched-synthetic generation of §8) call `require_nuisance_dli`, which
   loads the built artifact and fails loud, naming the analysis to run, if it is absent. It never rebuilds:
   building needs the estimator and a GPU.
2. **The analysis suggests; the user decides.** Emit-template mode draws a light pool and presents the calibrated
   imaging (per-parameter 5th/95th percentiles) as suggestions, pre-filling `box_user`; the user sets the choice.
   Those percentiles are estimates on experimental recordings, which have no ground truth — decision support, not
   a demonstrated recovery (recovery is quantified only on held-out synthetic data, §6.9–§6.10 and §9.2 B5).
3. **The build is self-contained.** Build mode loads the estimator, reads and windows the recordings itself
   (`--total-time-seconds`, `--experiment-span-seconds`, `--chunk-step-seconds`), builds the pool, materializes
   the chosen representation, and writes the artifact with its manifest; provenance goes into the spec's
   provenance comment. The per-simulation `Nuisance_DLI_Theta_Set` record is written later, at generation, when
   production marginalizes the imaging (§9.2 Phase D) — not by this build.

Arguments, the six choices, and how to choose among them are in the companion
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.md`; this section records the design, not the usage.

### 7.4 Persisted nuisance records

Per-simulation nuisance draws are recorded as a `Theta_Set` variant — one file per task in the `Theta/`
subdirectory beside the learnable `Theta_Set`, built by `Paths.record_set_path` from the theta-set pattern
`{project_alias}_{condition}_{timing_label}_Nuisance_<DOMAIN>_Theta_Set_TASK_{n}_{split}.{ext}`. The token names
the marginalized domain: `Nuisance_DLI_Theta_Set` when production marginalizes the imaging, `Nuisance_SCOPE_Theta_Set`
for the camera in both workflows. The detector's biology nuisance has no token of its own — its record is the
condition's tier's `Theta_Set` (§4). Every nuisance record is a DLI-stage product and carries the condition token,
as do `Labeling_Set` and every other DLI-side product. The record is distinct from the sampler that produced it
(the transient camera box, or the persisted `Nuisance_DLI` artifact) and is its self-labeling trace. Every
learnable `Theta_Set` carries its schema in the store's attributes — ordered keys, prior bounds and scales,
condition, timing label, generating stage, package version (`io.theta_set_schema`) — and every reader refuses a
`Theta_Set` whose schema differs from its table (`io.load_theta_set`); the `Nuisance_DLI_Theta_Set` record carries
the six-row schema for provenance only, and the `Nuisance_SCOPE_Theta_Set` record carries none.

### 7.5 The estimator artifact format

Estimators are persisted by `artifacts.py` as three separable components in one `.npz`: **(a)** a
compile-stripped `state_dict` — tensor weights only, the `_orig_mod.` prefix that `torch.compile` adds during
the Inference stage removed; **(b)** a rebuild spec — the architecture hyperparameters and normalizing-flow
configuration sufficient to reconstruct the estimator under whatever torch version loads it; **(c)** a
manifest — `artifact_format_version`, the ordered `parameter_keys`, `torch_version`, `weights_sha256`, and a
metadata block with timing label and provenance — stored beside the `prior_low`/`prior_high` bound arrays. `load_estimator` rebuilds the module
from the spec under the running torch, applies `load_state_dict`, reattaches a freshly built device-aware prior,
and refuses an artifact whose `parameter_keys` differ from the table it is asked to serve. Nothing torch-internal
is ever deserialized, so the artifact is self-describing and version-portable — the reason for the format: a
pickled posterior stores its prior bounds positionally without parameter names and, after `torch.compile`,
embeds `torch._dynamo` internals whose private layouts change between releases. This is the sole persisted
estimator format; both workflows write and read it, and the run-identity table of §6.8 quotes its
`weights_sha256`.

---

## 8. Quantitative experimental-versus-synthetic distance

Embedding-space distance between experimental and synthetic videos quantifies the domain gap that arises when the imaging model is misspecified. The gap must be measured, not read off a low-dimensional projection: non-metric projections are seed-dependent and do not preserve inter-point distances, so proximity in a projection is not a metric statement.

**Target of the analysis.** A standalone analysis, complementary to the Detector Experiment stage, that consumes the trained detector estimator and the generated videos; it is not part of the biology workflow. Because the detector workflow marginalizes the biology as a nuisance, the estimator's embedding is a representation of the imaging parameters, and a distance measured in that embedding is a statement about imaging realism — attributable to the imaging model and, because the embedding encodes those parameters, reducible by adjusting it. The biology, integrated out, introduces no imaging-versus-biology ambiguity into the measured distance.

**Statistics.** Experimental recordings and the held-out synthetic EVAL set are embedded through the trained `Complex3DCNN` (the estimator's embedding network), and two two-sample statistics are computed on the raw embeddings. The Maximum Mean Discrepancy (MMD; Gretton et al. 2012) is a kernel discrepancy, here an RBF kernel at the median-heuristic bandwidth. The Classifier Two-Sample Test (C2ST; Lopez-Paz & Oquab 2017) is the cross-validated accuracy of a classifier trained to separate the two embedding sets. The EVAL set is the synthetic reference because it is held out from training and drawn from the same imaging prior as the training set — an unbiased in-distribution synthetic sample.

**A realism gap the distance cannot localize: within-recording non-stationarity.** The forward
model renders a recording with **time-invariant** imaging parameters — one PSF, one brightness
population, one bleaching rate, one flicker rate for the whole video. The temporal analysis of the
Experiment estimates shows the experimental recordings do not honor that: the inferred bleaching
probability and the inferred brightness median both fall coherently across a recording, in nearly
every recording, while the PSF median stays flat. This is a distinct realism gap from the static
mismatches discussed above, and the embedding distance cannot separate the two — a single distance
between two clouds of windows is blind to whether the discrepancy is a constant offset or a drift
within each recording, because windows enter the statistic unordered. Reducing the static gap would
therefore leave this one untouched, and a candidate imaging model that matches the pooled
distribution can still misrepresent every recording's time course. Quantifying it is the temporal
analysis's job (`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_Temporal_Dynamics.md`); what
belongs here is the consequence for this section's statistics — they bound the pooled discrepancy
only.

**Cell-level significance.** Each recording is partitioned into fixed-length windows matching the estimator input, so windows from one recording share its acquisition and are statistically dependent. Treating windows as independent would inflate the effective sample size and render the significance anticonservative. The MMD permutation therefore resamples whole recordings (block permutation over recordings), and the C2ST uses recording-grouped cross-validation (`GroupKFold`), with significance taken from the across-fold accuracies rather than pooled per-window predictions. Synthetic videos are mutually independent, each its own block.

**Interpretation.** A C2ST accuracy statistically indistinguishable from 0.5 indicates the embedding sets are not separable — the experimental and synthetic distributions overlap, indicating realistic imaging; an accuracy approaching 1.0 indicates separability, hence a gap. Correspondingly, an MMD consistent with zero under its permutation null indicates no detectable discrepancy, and a significant MMD scales with the gap. Overlap is concluded only when both statistics concur; the decision thresholds are specified in the companion note.

**Visualization.** Figures are deterministic and tied to the statistics; stochastic embeddings (UMAP, t-SNE) are excluded because their layouts are seed-dependent and need not co-vary with the computed distance. Two summaries are reported: the out-of-fold C2ST score distributions of the two sets, whose overlap is the graphical form of the classifier accuracy; and the within-experimental, within-synthetic, and cross-set embedding-distance distributions, the graphical form of the MMD.

**Scope.** The analysis is a measurement and a runtime out-of-distribution indicator; it does not modify the model. A measured gap is reduced by one of two interventions — recalibrating the imaging or widening the imaging prior so training spans the experimental acquisitions (a model change), or adding an embedding-alignment term to the estimator objective (a training change) — each assessed against the distance reported here, neither performed by this analysis.

**The mirror analysis in the biology workflow.** The same construction applies to the biology workflow with the two parameter blocks' roles exchanged: there the imaging is the nuisance and the embedding represents the biology, so the measured distance is a statement about biological rather than imaging discrepancy. That companion analysis is where imaging-versus-biology hypotheses are posed and tested — for instance by re-measuring against synthetics whose imaging is fixed to the calibrated experimental values (§7), so that a distance that collapses attributes the gap to imaging while a residual distance is candidate biological signal — and where gap reduction requires the analogous interventions plus additional biology-specific steps. It is implemented on the same shared engine as this analysis — one measure module (`embedding_space_distance.py`) and one runner in the package, each workflow driving them through its own thin shim (`SRM_AND_SBI_MONOMER_DIMER_ALP_Embedding_Space_Distance.py` for the biology workflow, its `DETECTOR`-qualified twin for this one), with the biology companion note beside its script — so the two analyses share one measurement and cannot drift.

**A complementary reading from the Experiment stage.** The two-sample distance above is the formal gap measure; the Detector Experiment stage (§9.2, B4) already supplies a coarser experimental-versus-synthetic reading, from applying the calibrated six-parameter estimator to the experimental MET recordings of the two conditions of Harwardt et al. (2017; §6.7) — the resting-receptor monomer control (Fab, MET-FAB) and the internalin-B-activated, dimerization-competent variant (InlB, MET-INLB). Because the biology is marginalized (§5), each per-condition maximum-a-posteriori estimate is a statement about the imaging alone, read here against the independent single-molecule-localization reference of §6.7. Values are per-condition medians pooled over 250 cell-and-chunk windows each, back-transformed to physical units from the log10 posterior.

| parameter | MET-FAB inferred | MET-INLB inferred | MET-FAB reference (§6.7) | MET-INLB reference (§6.7) |
|---|---|---|---|---|
| `mu_r` — PSF (√2·σ; Gaussian σ in nm) | 1.60 (σ ≈ 179 nm) | 1.83 (σ ≈ 205 nm), at the prior ceiling 2.0 (σ ≈ 223 nm) | 1.36 (152 nm) | 1.47 (164 nm) |
| `sigma_r` — PSF spread | 0.15 | 0.22 | 0.37 fitted → ≈0.15 corrected | 0.42 fitted |
| `mu_pc` — monomer brightness (photons) | 287 | 365 (ratio 1.27×) | 386 (per-detection) | 690 (per-detection; ratio 1.8×, true ≈2×) |
| `sigma_pc` — brightness spread | 0.56 | 0.55 | 0.61 fitted → ≈0.5 corrected | 0.55 fitted |
| `prob_photo_bleach` | 0.061 | 0.093 | — | — |
| `lambda_rate` — flicker (s⁻¹) | 3.46 | 2.34 | ≈5 (localization-autocorrelation anchor, §6.5) | ≈5 |

*In the reference columns, **fitted** is the spread read directly off the ThunderSTORM log-normal fit; **corrected** is that same spread after the per-localization fitting-error variance inflating it is removed — the errors-in-variables bias of §6.7, caveat 1 — and is the value a full calibration is expected to recover. The correction applies only to the two population spreads (`sigma_r`, `sigma_pc`); the medians (`mu_r`, `mu_pc`) are robust to fitting noise and are quoted as fitted. The brightness reference is labeled **per-detection** because a localization records one spot's summed photons, so the dimer condition's value is the two-label sum, not the monomer-parent `mu_pc` the model infers (§6.6, §6.7 caveat 3).*

**The monomer control fixes the clean manifold.** In MET-FAB a monovalent Fab labels the resting receptor, so a diffraction-limited spot is a single label. The estimator recovers the emitter model at its uncontaminated values: `sigma_r` = 0.15, exactly the fit-corrected PSF spread §6.7 predicts once the localizer's errors-in-variables inflation is removed (caveat 1); `sigma_pc` ≈ 0.56, at the fit-corrected brightness spread ≈0.5; and a monomer brightness and PSF width inside their §6.2 priors. The control behaves as a control should — it lands the identifiable emitter parameters at the reference manifold, corroborating the calibration.

**The dimerization condition's differences are imaging signatures the model resolves.** In MET-INLB, internalin-B activation brings receptor pairs together, so a spot can carry two co-located labels. The estimate differs from the control in exactly the parameters a dimer perturbs, each in the direction the dimer physics dictates:

- *PSF.* The effective PSF is wider (σ ≈ 205 vs 179 nm) and more variable (`sigma_r` 0.22 vs 0.15). Real dimers assemble and separate across a range of sub-diffraction distances, producing effective spots broader and more dispersed than a single label; the forward model represents a dimer as two *co-located* labels with no association or dissociation dynamics (§6.6) and so does not span that range, and the estimator absorbs the excess width by raising `mu_r`. This is why MET-INLB, alone, presses the upper `mu_r` prior boundary.
- *Brightness.* `mu_pc` is the **monomer-parent** brightness — a dimer's brightness is built by summing two independent monomer draws (§6.6), not by a larger `mu_pc` — so the between-condition `mu_pc` ratio should fall well below the per-detection doubling, and it does: 1.27×, against the localization per-detection ratio of ≈1.8× (itself a lower bound on the true ≈2×, the 73–1225-photon acceptance window clipping the dimer tail harder; §6.7 caveat 3). The two ratios measure different quantities — monomer parent versus summed per-detection — and their separation is the quantitative signature that the sum model (§6.6), not the brightness prior, carries the dimer doubling. The residual above unity (1.27× rather than 1.0×) is the fraction of the dimer brightness the estimator attributes to `mu_pc` where the trained dimer statistics under-represent the true population.
- *Flicker.* The video-level inference returns `lambda_rate` ≈ 3.5 (MET-FAB) and ≈ 2.3 (MET-INLB) — below the localization-autocorrelation anchor of §6.5 (≈5) and, unlike that anchor which is condition-independent by construction, split between the conditions. Summing two independent flicker trajectories reduces the coefficient of variation by √2 (§6.6), damping the apparent fluctuation, so a dimer-bearing condition reads as a slower effective switching rate; the split is consistent with that damping. The §6.5 anchor is a prior-placement input from a distinct signal — the autocorrelation of the localization `intensity[photon]` series — and the calibrated posterior is the operative estimate.

**`sigma_pc` is the internal control.** The one emitter quantity a dimer should not change is the brightness spread of the underlying monomer population, a property of the dye. The inference returns it condition-independent (0.56 vs 0.55). That the estimator moves `mu_r`, `mu_pc`, and `lambda_rate` between the conditions while holding `sigma_pc` fixed is direct evidence that it does not inflate the emitter model indiscriminately for the harder condition, but resolves the specific quantities a dimer perturbs.

**Marginalizable variables, by design.** The detector infers no biological rate. Its forward model contains the full reaction-diffusion side — association at the condition's declared constant, dissociation, mode switching, and the initial composition — because it re-images the condition's trajectory tier (§4), and every one of those quantities is marginalized from that tier rather than calibrated, so no biological unknown enters the imaging characterization as a target and the separation the two-workflow design maintains (§5, §7) is kept: making a reaction-diffusion quantity a calibration target would not yield a better imaging fit, it would only import a biological unknown into the imaging estimate. The detector's purpose is to characterize the imaging manifold as it appears in the videos — dimer-induced effects included — and to deliver those effects to production as calibrated, marginalizable variables. Because it captures rather than discards them, the production (biology) workflow marginalizes the imaging over a realistic distribution — the `Nuisance_DLI` photophysics pool and the shared SCOPE camera box (§7, §9.3) — so the imaging's confounding effect on the biology inference is propagated and accounted for, neither neglected nor understated. The MET-FAB / MET-INLB read-out is the check that the device responds to the imaging differences between two biologically distinct conditions and returns condition-specific imaging distributions for downstream marginalization. Whether those distributions separate the imaging parameters from the biological ones the detector marginalizes is not established by the read-out itself; it is assessed by the posterior-predictive and embedding-gap analyses under the revised model.

**In-distribution on the experimental recordings.** The calibrated six-parameter model places the experimental recordings inside its training support. Prior-bounded pooling — rejection sampling confined to the training prior box — completes for all 500 windows of both conditions, whereas treating the camera as an inference target (the eleven-parameter alternative) stalls at ≈0% acceptance because the experimental-data posterior mass falls outside the box. Marginalizing the camera removes that out-of-distribution driver: the six-parameter emitter posterior lies within the training prior, which is the operational meaning of in-distribution for this workflow. The maximum-a-posteriori log-density is comparable on the experimental recordings and the held-out synthetic EVAL set (≈14.0 versus ≈14.2), consistent with the same conclusion, though as an optimizer diagnostic in the estimator's z-scored space it corroborates rather than measures — the quantitative gap statement is the two-sample distance above, and calibration is quantified by the coverage diagnostics of §9. The single place the support is tight is the MET-INLB `mu_r` boundary.

**The `mu_r` boundary is a diagnostic, not a call to widen the prior.** The MET-INLB ceiling marks where the dimer condition's effective PSF meets the edge of the trained imaging manifold, and it is tempting to treat it as the widen-the-prior intervention above — but that is the wrong instrument here. The excess width is a dimer-geometry effect — two labels across a range of sub-diffraction separations — not a broader single-emitter point-spread function; letting the single-Gaussian `mu_r` prior inflate to absorb it would fit a geometric effect with the wrong degree of freedom, pushing `mu_r` past the physical optical width (σ ≈ 152–164 nm, §6.7, already exceeded by the inferred effective 179–205 nm) and laundering a model-mismatch into an unphysical parameter. The bounded behavior is the correct one: production draws the imaging from the same bounded manifold (§7, §9.3), so the marginalization stays internally consistent, and the boundary is a quantified, accepted consequence of the deliberate no-association/dissociation scope rather than a defect to patch. The faithful account of dimer-separation variability, should it ever be required, is a biology-side model extension in a later iteration — not a wider emitter-PSF prior.

---

## 9. Peer-review gaps and the execution plan

### 9.1 Gaps this workflow closes, and those it defers

**Closed:** justification and reproducibility of every imaging parameter; a posterior summary (estimate plus credible interval) rather than a bare point; a prior-bounded (or explicitly logged out-of-bound) maximum-a-posteriori estimate; simulation-based calibration and coverage diagnostics (SBC, Talts et al. 2018; expected coverage and the trust crisis, Hermans et al. 2022; TARP, Lemos et al. 2023; local C2ST, Linhart et al. 2023); a quantitative experimental-versus-synthetic distance (the two-sample measure implemented, with its analysis script and companion note in place, and run on the calibrated θ=6 estimator — 500 experimental windows against 10,000 synthetic EVAL windows, with a gap verdict on every comparison; a complementary in-distribution read-out of the calibrated imaging on the experimental MET recordings is given in §8); disclosed training configuration; the brightness-flicker identifiability, resolved by inferring only the switching rate and deriving the locality from the brightness scale (§6.5); the corrected EMCCD noise model — a stochastic Gamma electron-multiplication register (excess-noise factor `F² = 2`), a gain-independent read noise added after the register, and a unit-explicit `(quantum efficiency, EM gain, conversion factor, read noise, bias)` parameterization — implemented in the shared `add_noise` / `EMCCD` forward model and specified, with its derivations, background-illumination model, and validation protocol, in `REFERENCE_EMCCD_NOISE_MODEL.md`; and the gain–conversion degeneracy, reduced to the single identifiable ratio `gamma = g/C` (marginalized as part of the SCOPE camera nuisance, with `g` and `C` held at spec values as drift-check metadata, §6.2).

**Realized in the biology stage.** The biology `parameterization.py` and its Simulation_DLI stage share the corrected EMCCD forward model and the source-agnostic renderer `render_dli_video`: the value-based roles marginalize the whole imaging block (the six photophysics from the `Nuisance_DLI` artifact, the five camera from the shared SCOPE box; §9.3, Phase D), so the imaging block carries no separate `(offset, gain, variance)` camera construction and no brightness-transition penalty. The residual imaging identifiability question — the brightness-scale / PSF-width coupling (a spot's peak scales as brightness / width²) — is constrained jointly by the data, not removed by a reparameterization. The detector's own camera block is marginalized as a shared, independent-uniform SCOPE nuisance, so the inference target is the six identifiable emitter parameters (§9.3).

The estimator-evaluation methods that operationalize the calibration/coverage and quantitative-distance gaps above — the per-example loss distribution, the paired cross-run comparison, the central-limit-assumption checks, and the diagnostics cited here — are detailed with full references in the Validation and Diagnostics section of `PROJECT_CONTEXT.md`.

### 9.2 Implementation plan

Both workflows are fully implemented; the phases below record the module structure and the dependency order in which the shared engine and the Detector stages were assembled, so this section reads as the realized build record, not outstanding work.

**Guardrails.** Both workflows share one engine per stage, so a change to a stage's engine lands in both and neither can silently drift; the genuine per-workflow differences (parameterization module, alias-qualified paths, workflow tag, and the DLI imaging source) are carried by the `WorkflowConfig` and localized in a `_<stage>_spec(cfg)` resolver, not duplicated across entry points. Shared machinery is parameterized rather than branched: both workflows read the same per-condition reactive trajectory tiers (§4) and differ in what their DLI stage draws; the dataset and training builders (`VideoDataset` / `build_datasets` / `setup_training`) and `console_log_context` take the alias-qualified `paths=` / `data_bank_root=` from the config, so each workflow reads and writes its own filename-namespaced data and its debug transcript carries the correct tag. The production treatment of the calibrated imaging is set at the shared parameterization (Phase D); §5 identifies the exact hazard. Everything here is code and documentation; no simulation or training compute runs — on rcl01 or the HPC machines — without explicit approval (rcl01 is used to prove the workflow; production runs on the HPC machines). The runnable end-to-end proof of this workflow is the Detector calibration smoke test in VALIDATION.md (section 2.5) — the five detector stages run seedless with small overrides on a single GPU, or on HPC via the DETECTOR_HPC_* wrappers.

**Phase A — schema, parameter machinery, and adapted forward models (new modules).**
- **A1 `detector_parameterization.py`** — the value-based parameter table (imaging parameters Learnable §6.2; the eleven reaction-diffusion rows Nuisance-from-object §6.1, supplied by the condition's trajectory tier; the camera Nuisance-from-spec §9.3; `capture_radius`, `delta_frame`, and `numb_photo_bleach` Fixed), together with the **role resolver and θ constructor**: the value-based dispatch (§5), the learnable-subset selector (`VALUE`-not-a-sentinel AND `PRIOR_RANGE`-not-`None`), the SCOPE nuisance draw from its inline `BoxUniform`, and the physical-space mapping the forward models consume.
- **A2 — no Detector RDS forward model.** The Detector reads the condition's trajectory tier (§4); the reactive simulator lives only in `simulation_rds_support.py` and runs once per condition, through the single RDS entry point.
- **A3 `detector_simulation_dli_support.py`** — the Detector-facing DLI forward model. The source-agnostic renderer that reuses the shared imaging building blocks (`build_dye_tracks`, `generate_brightness_photons`, `EMCCD`, `Gaussian`, `sample_psf_width`, `compute_intensity`, `add_noise`, `generate_frames`) and sources every imaging value from an assembled eleven-key vector — reading each by key, so a value renders the same whether it arrived as an inference target or a marginalized nuisance — lives in the shared `simulation_dli_support.py` as `render_dli_video` and reads its fixed hyperparameters from the biology parameter table; this module re-exports it as `render_detector_video` for the Detector DLI stage and the Detector posterior-predictive analysis. Both DLI stages therefore share one renderer.
- **A4 `detector_nuisance_dli.py`** — the `NuisanceDLI` artifact (§7): a samplable object with a `parameter_keys` manifest and the two knobs, plus the on-the-fly pool builders, the six-choice construction (including the CPU `sgm_percentiles` selection — the shared `sample_geometric_median` + `select_signed_percentile_vectors`), and the loading gate. (There is no separate `nuisance.py`; the RDS nuisance is supplied by the condition's tier and the camera nuisance by its inline box.)
- **A5 `artifacts.py`** — the self-describing estimator format (§7): compile-stripped `state_dict` + rebuild spec + metadata, with a loader that rebuilds eagerly. It is the sole persisted estimator format for both the Detector and the biology workflow.

**Phase B — Detector entry scripts (new; C28-named `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_*`; each drives the adapted forward-model support of Phase A).**
- **B1 — no Detector RDS entry point.** Each condition's tier is generated by the single RDS entry point (`..._Simulation_RDS.py --condition`, §4); the Detector's Simulation stage is B2.
- **B2 Detector DLI simulation** (`..._DETECTOR_Simulation_DLI.py`) — drives `detector_simulation_dli_support.py`: renders videos per `--condition` (the condition's labeling law) with the imaging θ drawn per simulation from the Detector prior.
- **B3 Detector inference** (`..._DETECTOR_Inference.py`) — train the imaging posterior on (video, imaging-θ) pairs, reusing the embedding and flow machinery; save via A5. Multi-GPU / multi-node data-parallel like the biology Inference (`torchrun` on one node, or `srun` + `torchrun` with a c10d rendezvous across nodes, one process per GPU under `DistributedDataParallel`; the single-GPU path is used when one GPU is allocated).
- **B4 Detector experiment** (`..._DETECTOR_Experiment.py`) — maximum-a-posteriori estimate on experimental videos per condition. The MMD / C2ST gap check (`embedding_space_distance.py`, §8 -- the workflow-agnostic measure module, shared with the biology analysis) is provided as the standalone measure module and run as a complementary analysis alongside this stage, not inside it; its analysis script and companion note complete it. The `Nuisance_DLI` is *not* exported here: it is constructed by the separate, user-driven analysis step (§7, "Constructing the `Nuisance_DLI`"), which the production import requires via a validating gate. Multi-GPU / multi-node sharded like the biology Experiment (per-worker shard across every rank on every allocated node, then a combine step).
- **B5 Detector evaluation** (`..._DETECTOR_Evaluation.py`) — coverage and held-out recovery for the imaging posterior. Multi-GPU / multi-node sharded like the biology Evaluation: the held-out tasks split round-robin across one worker per GPU on every allocated node (`torchrun`, or `srun` + `torchrun` across nodes), each worker writes its partial recovery arrays as a shard to the shared filesystem, and a separate `--merge` step (single process, no GPU) concatenates the shards into one report; the single-worker path writes the report directly.

The Detector downstream stages (B3–B5) therefore match the parallelism of the biology stages they mirror — multi-node data-parallel training and sharded recovery — rather than running single-process.

**Phase C — validation and gap closure.** Coverage (B5); per-condition gap quantification (B4); prior-bounded or logged-out-of-bound estimation; and one versioned, provenanced imaging-parameter file.

**Phase D — the production (biology) parameterization.** The biology `parameterization.py` carries the value-based roles, its learnable-subset selector being `VALUE`-not-a-sentinel AND `PRIOR_RANGE`-not-`None` so nuisance rows are not pulled into the inference prior (§5, constraint 2); the biology Inference uses the self-describing artifact format; and the production treatment of the imaging parameters is the whole-imaging marginalization — the photophysics drawn from the supplied `Nuisance_DLI` artifact and the camera from the shared SCOPE box (§9.3). The alternatives the value-based scheme equally expresses — holding the imaging fixed at the calibrated values, or inferring it jointly under calibrated-centered priors — are other settings of the same mechanism, not separate code paths.

The chosen treatment is the whole-imaging marginalization, realized in the biology Simulation_DLI stage. The six photophysics rows and the five camera rows in `parameterization.py` carry `VALUE = 'NUISANCE'` — the photophysics with `PRIOR_RANGE = None` (nuisance-from-object, drawn from the persisted `Nuisance_DLI` artifact) and the camera with their boxes retained (nuisance-from-spec), so the learnable subset stays exactly the eleven reaction-diffusion parameters and the inference prior is `event_shape == (11,)`. Per task the stage draws the photophysics from the artifact (resolved from the durable tier under the Detector alias, with its `parameter_keys` schema guarded) and the camera from the shared SCOPE box, records each as a self-labeling `Theta_Set` variant — `Nuisance_DLI_Theta_Set` and `Nuisance_SCOPE_Theta_Set` — beside the learnable eleven-parameter `Theta_Set` (read for diagnostics only, never re-written), assembles the eleven-key imaging vector, and renders every video through the shared source-agnostic `render_dli_video`. The Inference, Evaluation, Experiment, and dataset-generation stages are unaffected: they read no imaging value and train on `(video, eleven-parameter Theta_Set)` only, so the `Nuisance_*` records are additive provenance they ignore.

**Ordering.** A1–A5 → B1–B5 → C → (validate) → D.

**Realized in the biology stage.** The biology `parameterization.py` and its DLI stage carry the reworked forward model: the stationary OU brightness flicker (§6.5), the revised diffusion and count ranges (§6.1), and the value-based-role parameterization that lets each parameter block be held fixed, inferred, or marginalized as a nuisance rather than fixed as a vector only. The camera block constructs the detector with the corrected EMCCD parameterization (§6.2; `REFERENCE_EMCCD_NOISE_MODEL.md`) through the shared renderer, and the whole imaging block is marginalized in production (§9.3, Phase D). The alternative noise model remains a distinct, later iteration (out of scope here).

**A general capability.** The value-based role scheme (§5) is more than the Detector's parameterization: the biology codebase carries it as a general capability — the flexible treatment of any parameter block as a **posterior, a nuisance, or a fixed vector**, the same three modes by which the Detector's imaging output feeds production (§2), so a block's role switches without structural change. Holding the imaging fixed is one setting of that mechanism rather than a hardcoded default. The mechanism, the self-describing artifact format (§7), and the learnable-subset selector hazard (§5, constraint 2) are what let a block's role change safely across the two workflows.

### 9.3 Marginalizing the camera block — the SCOPE nuisance

The imaging inference target is the six identifiable emitter parameters — the PSF pair (`mu_r`, `sigma_r`), the brightness pair (`mu_pc`, `sigma_pc`), and the photophysics pair (`prob_photo_bleach`, `lambda_rate`). The five EMCCD camera parameters (`gamma`, `kappa_o`, `kappa_b`, `kappa_s`, `kappa_q`) are marginalized as the SCOPE camera nuisance rather than inferred. This section gives the rationale and the implementation.

**Why the camera is a nuisance, not a target.** The five camera parameters are externally constrained rather than jointly inferred in this workflow, for two facts that are kept distinct. The first is structural: the algebra below shows which camera combinations the image likelihood confounds. The second is empirical: when the predecessor detector inferred the camera block jointly with the emitter parameters, its held-out recovery, read as a per-parameter error against the width of each prior, exceeded the prior width for `gamma`, `kappa_o`, and `kappa_q`, sat near it for `kappa_b`, and left `kappa_s` only weakly constrained, while the emitter parameters were recovered well within their priors. Neither fact says the camera cannot be measured: under controlled illumination the mean–variance relation constrains `gamma` on its own (`REFERENCE_EMCCD_NOISE_MODEL.md` §8). The algebra is explicit. A spot's peak signal scales as `mu_pc·kappa_q·gamma / sigma_r²`, so inferring `gamma` splits the brightness amplitude with `mu_pc` and degrades the brightness calibration, which is a target; the data pin only the products `gamma·kappa_q` and the optical floor `gamma·kappa_q·kappa_o`, not the camera factors separately (§6.2; `REFERENCE_EMCCD_NOISE_MODEL.md` §9). The read noise `kappa_s` does not dominate the signal-to-noise ratio: the electron-multiplication register amplifies the signal above the read-noise floor, so its standard deviation stays subordinate to shot noise even for dim spots (`REFERENCE_EMCCD_NOISE_MODEL.md`). Carrying five externally constrained camera axes in the inference prior inflates the dimension of a low-dimensional problem and lets those axes absorb variation that belongs to the brightness and PSF targets. Integrating them out over their a-priori range is the Bayesian treatment of a nuisance block, and it is a treatment the value-based role scheme already expresses (§5): a camera row set to nuisance-from-spec carries its §6.2 range as the box it is drawn from and is excluded from the inference prior.

The imaging categorization of §5 is:

| category | role | members | rationale |
|---|---|---|---|
| **Inferred imaging** — the calibration targets | learnable | PSF (`mu_r`, `sigma_r`), brightness (`mu_pc`, `sigma_pc`), photophysics (`prob_photo_bleach`, `lambda_rate`) — six | the identifiable emitter model the workflow calibrates |
| **SCOPE nuisance** — the camera | nuisance | `gamma`, `kappa_o`, `kappa_b`, `kappa_s`, `kappa_q` — five | externally constrained rather than jointly inferred: the EM-gain degeneracies confound the individual quantities in the image likelihood; marginalized over a fixed a-priori box; the one block **both** workflows marginalize |
| **RDS nuisance** — the biology | nuisance | the stoichiometry and mobility blocks (§6.1) — eleven rows, the biology prior | reaction-diffusion biology marginalized while the imaging is calibrated; detector-only |
| **Conditioned by the coordinate frame** | fixed | pixel size, field size, `delta_frame` | a coordinate frame and cadence, not physics — supplied as acquisition metadata, not inferred; positions and brightness are sampled at the frame interval without integrating motion during exposure |
| **Fixed hyperparameters and spec metadata** | fixed | `numb_photo_bleach`; `kappa_g`, `kappa_c` | modeling choices; `kappa_g`/`kappa_c` retained as `γ = g/C` drift-check metadata (§8) |

**Three nuisance blocks, and the first shared one.** The camera joins the biology and the photophysics as a
marginalized block, distinguished by being marginalized in *both* workflows — the block common to both
marginalizations, recorded under a single shared token. The blocks, their sources of truth, and their records
are tabulated in §7.1.

**The camera nuisance needs no artifact.** Its range is known a priori — the acquisition-protocol values that anchor the §6.3 camera ranges — so it is a transient `BoxUniform` declared on the fly at generation, unlike `Nuisance_DLI`. `Nuisance_DLI` must exist as a persisted, samplable object because its content is the calibration result — the pooled detector posterior — which cannot be declared in advance (§7); it remains the only nuisance that exists as an object. The camera nuisance has no such content to store: it is fully specified by its five ranges in the parameter table, so there is nothing to build. Only its per-simulation draws are recorded, as the self-labeling `Nuisance_SCOPE_Theta_Set`.

**Shared means one specification, not one sample set.** Both workflows draw the camera from the same a-priori box — one specification, the same five rows — and record their draws under the same token. The draws are independent and per-run: each generation draws one camera vector per simulation, sized to that run, and neither workflow reuses the other's. A production generation draws as many camera vectors as it has simulations, a count decided by that run alone and independent of the detector's.

**Where each workflow draws it.** The detector draws the camera at the DLI stage, beside the learnable draw, and writes `Nuisance_SCOPE_Theta_Set` — placing the draw where the camera is consumed and avoiding a read of the earlier reaction-diffusion record. The production (biology) Simulation_DLI stage marginalizes the whole imaging block: per task it draws the six photophysics from the `Nuisance_DLI` artifact and the five camera from the shared SCOPE box, records each under its token beside the learnable eleven-parameter reaction-diffusion `Theta_Set` (which it reads for diagnostics only, never re-writing), and concatenates them per simulation into the eleven-key imaging vector the renderer consumes. The renderer is source-agnostic — it requires only the eleven-key vector — so the imaging-key list (`_IMAGING_KEYS`) stays the full eleven and the forward model is unchanged whether a value arrives as a target or a nuisance.

**Distributional invariant.** The camera is drawn from its independent uniform box in both workflows and is never drawn from the `Nuisance_DLI` pool. The camera and the photophysics are independent in the generative model; their apparent coupling — the `mu_pc·gamma` amplitude, the `gamma·kappa_q·kappa_o` floor — is a property of the likelihood, not of the prior, so the pool, which preserves the joint structure of whatever it holds, holds only the photophysics. The calibrated `Nuisance_DLI` therefore covers the six photophysics parameters, and the camera degeneracies it would otherwise carry are represented by the a-priori SCOPE box instead — consistent with the camera's role as a nuisance rather than a calibration target.

**Implementation.** In `detector_parameterization.py` the five camera rows carry `VALUE = 'NUISANCE'` with their `PRIOR_RANGE` and `LOG_FLAG`/`LOG_BASE` retained, so each resolves to a nuisance-from-spec (§5): the learnable subset (`DETECTOR_PARAMETERIZATION`) is the six emitter parameters, and the five camera form the SCOPE block (`DETECTOR_NUISANCE_SCOPE`), the selectors keying on the sentinel (§5, constraint 2). The RDS biology block (`DETECTOR_NUISANCE`) is nuisance-from-object, supplied by the condition's trajectory tier whose `Theta_Set` records the eleven biology parameters (§4). The full imaging vector (`DETECTOR_IMAGING`, eleven keys = the six learnable then the five SCOPE) is the render contract: the source-agnostic renderer `render_dli_video` (in `simulation_dli_support`, re-exported by `detector_simulation_dli_support` as `render_detector_video` for the Detector callers) reads every value by key and is unchanged, agnostic to whether a value arrives as a target or a nuisance; its fixed hyperparameter (`numb_photo_bleach`) comes from the biology parameter table, whose value matches the Detector table so the Detector's rendered output is unchanged. The Detector DLI stage draws the six learnable (→ `Theta_Set`, the inference target) and the five SCOPE camera (→ `Nuisance_SCOPE_Theta_Set`, columns in `DETECTOR_SCOPE_KEYS` order, minted by the `Theta_Set`-token swap of `Paths.record_set_path`), assembles the eleven-key vector, and renders every video. On the production (biology) side the Simulation_DLI stage marginalizes the whole imaging block: it draws the six photophysics from the `Nuisance_DLI` artifact and records them as `Nuisance_DLI_Theta_Set` (columns in `DETECTOR_PARAMETER_KEYS` order), draws the five camera from the shared SCOPE box and records them as `Nuisance_SCOPE_Theta_Set`, and concatenates them into the eleven-key vector — realizing the whole-imaging marginalization. In `parameterization.py` the six photophysics rows carry `VALUE = 'NUISANCE'` with `PRIOR_RANGE = None`, so each resolves to a nuisance-from-object (drawn from the supplied `Nuisance_DLI` artifact), while the five camera rows carry `VALUE = 'NUISANCE'` with their box retained (nuisance-from-spec); the learnable subset stays exactly the eleven reaction-diffusion parameters.


### 9.4 The acquisition-information contract and the reduced inferred block (proposal)

Two statements are recorded separately, because they are true at once. **The implemented detector
infers six parameters** — the design of §5, §6.2, and §9.3 is in force and unchanged by this
subsection. **Its current calibration results (§6.9) do not establish reliable posterior
uncertainty**; retaining the implementation is not a validation of it. What follows is a proposal
and the conditions under which it would be adopted. Nothing in it changes a parameter role, a
preprocessing step, or the executable model.

**Terminology.** Two kinds of estimator appear below. The *neural posterior estimator* is the amortized flow trained on simulated videos. The *direct estimators* fit a quantity from the frames without a trained network: spot fits, a decay fit, a simulation-matched autocorrelation. Both are validated against the simulator's truth, and the autocorrelation match is itself simulation-based inference in the classical sense, so the distinction is neural versus direct, not simulation-based versus not.

**What the pipeline needs from outside.** The pipeline can drop its dependence on ThunderSTORM's
file format; it cannot drop the need for acquisition information and calibration assumptions.
Supplying some quantities externally is a defensible way to make the inference tractable, provided
their sources, uncertainties, and consequences are recorded (the evidence-and-source table of §6.2).

| required information | why it is needed | acceptable source |
|---|---|---|
| pixel size, frame interval, exposure duration, image dimensions | set the spatial and temporal scales; assess whether neglecting motion blur is reasonable, since the exposure need not equal the frame interval | acquisition metadata; image dimensions from the frames |
| effective camera gain, baseline, read noise | map photoelectrons to camera values and describe the measurement noise | acquisition-matched calibration (dark stacks and uniform-illumination stacks, `REFERENCE_EMCCD_NOISE_MODEL.md` §8), acquisition settings, or qualified specifications |
| quantum efficiency | convert detected photoelectrons into incident photons; otherwise brightness must keep a detected-signal interpretation | camera characterization or an explicit assumption; gain calibration alone does not determine it |
| optical background | separate emitter signal from background | analysis of the original raw frames, conditional on the camera calibration: background photons ≈ (mean background ADU − baseline) / (`gamma` · QE) |
| labeling properties and probe occupancy | relate visible spots and their brightness to receptors and dyes; distinguish the measured mean labeling from the assumed dye-count distribution and occupancy | preparation measurements, collaborator information, and declared assumptions, each labeled as such |
| intensity encoding and preprocessing | ensure synthetic and experimental pixels undergo compatible scaling, clipping, and quantization | file metadata and the verified preprocessing code, including the fixed 16-bit to 8-bit conversion (`io.convert_video_dtype`: 0–65535 onto 0–255, clipped, no per-video normalization) |

The renderer's own conventions belong in this contract as well: it samples positions and brightness
at the frame interval and does not integrate motion during exposure, and the stored videos use the
fixed 8-bit map of the last row, applied identically to synthetic and experimental frames. Both must
hold for a recording the estimator is applied to.

**The proposed inferred block.** Three parameters: `mu_pc`, `sigma_pc`, and `lambda_rate`. The
brightness pair stays inferred because a spot's brightness is the sum over an unknown number of dyes
(§6.6), so its per-dye interpretation needs the labeling and imaging model that the simulator
provides and a brightness histogram does not. The fluctuation rate stays inferred provisionally: the
direct correlation-based estimator is plausible but must fit the model's own multi-dye intensity
autocorrelation (§6.5), account for motion and noise, and be validated before it replaces inference;
if that validation passes, the block reduces to two. The quantities that leave the block are the PSF
median and spread, constrained by noise-aware fits to isolated spots, and the fluorescence-loss
rate, constrained from full-length recordings. For the latter the observable is background-corrected
total fluorescence, which in the simulator is linear in the number of surviving dyes and therefore
decays at the per-dye rate whatever the dye count per spot, whereas a visible-spot count is not (a
two-dye spot survives the loss of one dye). Probe replenishment cannot in general be represented by
lowering an irreversible loss rate: total intensity may match while appearances, lifetimes, and
fluctuations do not, so a rate measured in its presence is an effective quantity and is labeled as
such.

**Uncertainty is retained, not removed.** A quantity that leaves the inferred block enters the
biology stage as a nuisance with an explicit range set by its measurement and its assumptions
(§7, §9.3), never as a point value. Where a replacement measurement fails or is weak, the quantity
keeps a justified nuisance range or the proposed split is revisited; it is not forced into a narrow
band. Dependencies between quantities that the measurements reveal are preserved in how they are
drawn, rather than everything being drawn independently.

**Adoption gates.**

| gate | required evidence |
|---|---|
| external inputs | acquisition settings and camera assumptions have documented sources and defensible uncertainties (the evidence-and-source table of §6.2, extended to the acquisition in hand) |
| replacement measurements | each direct estimator recovers the corresponding simulator quantity adequately over the intended operating range, on synthetic recordings with known truth; the full-recording loss estimator is tested on full-length simulations, not on 2 s clips |
| reduced neural estimator | the three-target estimator receives its own recovery and calibration assessment (§6.9's measures), including conditional failures and interval widths; a smaller inferred block is not assumed to fix coverage or the MAP density spikes |
| experimental adequacy | recordings generated with the measured inputs reproduce relevant raw-image and temporal statistics of the experimental recordings, not only the learned embedding |

Synthetic validation establishes performance under the tested generator, not experimental
correctness. For it, three tiers of data are kept apart: development data used to build a method;
data reserved from further method tuning; and, where a claim requires it, a genuinely fresh final
evaluation. Of the 25 existing EVAL tasks, five can be reserved from future tuning, but they cannot
be described as untouched, since the full set has already entered the calibration analyses of §6.9.

**What the one-dye comparison (§6.10) can and cannot settle for this proposal.** It tests the
sensitivity of the six-parameter estimator to the labeling model. It cannot by itself establish
fundamental non-identifiability of `sigma_r` or `lambda_rate`, nor the validity of the replacement
measurements; those are the gates above.

**Decision statement.** We propose reducing the detector's neural posterior estimator to brightness, brightness variation, and
fluctuation rate. Acquisition and camera information remain external inputs. Dedicated analyses will
constrain PSF properties and fluorescence loss where validated; unresolved quantities will retain
explicit nuisance uncertainty. Adoption depends on validating both the replacement measurements and
the reduced estimator.

### 9.5 The information budget — separating a weak estimator from uninformative data

An estimator that recovers a parameter poorly admits two explanations that look identical in
the results: the estimator may be weak, or the recordings may not carry the information.
They call for opposite responses — improve the estimator, or stop inferring the parameter —
so the calibration outcome of §6.9 cannot on its own decide the reduced block §9.4 proposes.
The missing quantity is an estimator-independent benchmark for what the data allow, approximate
but computed from the forward model alone. With it, each parameter is graded by three numbers:

| quantity | meaning | source |
|---|---|---|
| bound | an approximate benchmark: the estimated standard deviation, in log10 units, below which an unbiased estimator of the parameter from one recording is not expected to go under the reduced model | `Information_Budget` utility |
| direct | the scatter a direct, non-neural estimator achieves | the direct-estimator utilities |
| neural | the posterior width of the amortized flow | §6.9 |

A neural posterior near the benchmark is unlikely to gain much from a different estimator or a
change to the inferred block. One far from it has probable headroom, and the estimator or its
training is the first place to look. A benchmark wider than the parameter's own prior says the
reduced model expects one recording to constrain the parameter poorly, which makes the parameter a
candidate to leave the inferred block; it does not prove non-identifiability, because the benchmark
is approximate, treats a reduced observable rather than the full video, and constrains an unbiased
estimator's standard deviation rather than the error of a biased or Bayesian one. Conversely, a
small benchmark shows favorable information in principle, not that any implemented estimator
achieves that precision.

**Basis.** The likelihood is Gaussian with the exact EMCCD mean and variance of the
Poisson–Gamma–Normal chain, `mean = gamma·kappa_q·I + kappa_b` and
`var = 2·gamma²·kappa_q·I + kappa_s²`, both verified numerically against the renderer. The
factor 2 is the excess-noise factor of the multiplication register: an EMCCD pixel carries
the noise of half as many photons as its count suggests. Spot-level information is obtained
by differentiating the same pixel-integrated Gaussian the renderer uses, and the full
`(amplitude, x, y, sigma)` matrix is inverted before the width entry is read, because
amplitude and width trade off directly and treating either as known would understate the bound.

**Population spreads and the crossover that matters.** With `n` units each measured with
variance `v`, `sd(ln mu) ≥ sqrt((sigma² + v)/n)` and
`sd(sigma) ≥ (sigma² + v)/(sigma·sqrt(2(n−1)))`. The second has two regimes: above
`sigma ≈ sqrt(v)` it is `sigma/sqrt(2(n−1))`, a constant *relative* precision with no
degradation anywhere; below it, the measurement term takes over and the bound grows. Linking
a spot across frames replaces `v` by `v/L` and moves the crossover down. At the MET-FAB
operating point linking places `sqrt(v)` near 0.012, well below the `sigma_r` prior floor of
0.10, so the whole prior sits in the good regime; unlinked, `sqrt(v)` is near 0.086 — comparable
to that floor — and the bottom of the prior is genuinely information-starved. The direct PSF-width
estimator shows exactly that transition, which is why it links before summarizing.

**Duration, correlation, and the shallow-decay degeneracy.** Three effects set what a
recording can say about `prob_photo_bleach`, and they compound.

Decay-rate information grows as the cube of the recording length, so ten times the frames is
about thirty-two times the precision on a rate. Against that, the brightness is a stationary
Ornstein–Uhlenbeck process, so consecutive frames of a total-fluorescence curve are not
independent samples of the decay; the effective count is `n(1−rho)/(1+rho)` with
`rho = exp(−lambda_rate·delta_frame)`, and at the center of the `lambda_rate` prior a 100-frame
recording carries roughly three effectively independent samples rather than a hundred.

The third effect dominates and is the one most easily missed. The amplitude and offset of the
curve are unknown and are fitted alongside the rate. Where the decay is shallow, the
exponential is close to a straight line over the observed window, so the three parameters are
nearly degenerate and only the **product** of amplitude and rate is determined; the rate alone
is barely constrained however many frames are collected. A bound that treats the amplitude and
offset as known misses this and is optimistic by an order of magnitude exactly where the answer
decides a design question. The bound below profiles them out.

Bound on `prob_photo_bleach` in dex, at the MET-FAB emitter density, with all three effects
included (bold = inside the 0.10 dex threshold the direct estimator is held to; the prior is
1.5 dex wide, so a bound above that is no constraint at all):

| recording | p = 0.01 | p = 0.032 | p = 0.10 | p = 0.32 |
|---|---|---|---|---|
| 2 s (100 frames) | 1817 | 178 | 16.5 | 1.26 |
| 20 s (1000 frames) | 6.01 | 0.65 | **0.083** | **0.019** |
| 60 s (3000 frames) | 0.43 | **0.057** | **0.014** | **0.011** |

Read plainly: under this reduced model the estimated standard deviation of an unbiased
`prob_photo_bleach` estimate exceeds the 0.10 dex threshold by more than an order of magnitude
everywhere at 2 s (1.26 dex at the top of the prior, thousands of dex at the bottom, against a 1.5 dex
prior width), and falls inside that threshold only in the upper half of the prior at 20 s. The budget provides an approximate
precision benchmark for an unbiased decay-rate estimator using total fluorescence; the
effective-sample-size adjustment approximates the correlated decay likelihood rather than treating it
exactly, and the benchmark does not establish a fundamental recovery limit for inference from the
full video. The one-dye estimator's correlation of 0.80 on this parameter (§6.10) shows limited
precision rather than absence of information. What the benchmark supports is the direction §9.4
takes: constrain bleaching preferentially from longer recordings, and validate the fluorescence-loss
estimator on full-length simulations.

**These bounds are optimistic, deliberately.** They omit emitters entering and leaving the
field, reactions changing a spot's dye multiplicity mid-recording, overlapping spots, and the
truncation of the observable population by detectability — a spot wide enough spreads a fixed
photon budget below the noise floor and is detected by nothing, neither a direct estimator nor
the network. Every implemented estimator faces all four. The benchmarks are not predictions of
achievable error. A measured scatter well below one of them is a prompt to check the measurement,
for ground truth leaking into the estimate or a mis-stated benchmark, rather than an automatic
failure: they constrain an unbiased estimator, and a biased or Bayesian estimator that shrinks toward
the prior can legitimately do better. Comparisons are therefore stated against estimator scatter,
with bias reported separately.

**Standing of this section.** The budget is a diagnostic, not a decision. It constrains which
of §9.4's gates can be met and where effort is worth spending; it does not by itself move any
parameter out of the inferred block, and the implemented detector continues to infer all six.

### 9.6 Frozen acceptance rules for the direct estimators

> **MAP numbers here are under recomputation (§9.8).** The MAP routine returned coordinates one
> optimizer step away from the point whose density it reported; the defect is fixed in 0.1.15 and the
> affected stages are being re-run. The posterior-median and SGM columns, and every sampling-based
> calibration result, are unaffected.


These rules were fixed on 2026-09-21, after the development runs of the direct estimators on EVAL
tasks 0 and 1 had started and before any corrected evaluation ran. They govern how a direct estimator
of a detector parameter is judged from here on. They validate **direct estimates of the detector
parameters**; their later use in the biology workflow is a separate integration check and carries no
threshold here. The numerical accuracy thresholds are the ones each estimator's companion note has
carried since 0.1.12, restated with their units; what is new is the evidence, operational, subgroup,
and uncertainty requirements around them, and the order in which they are evaluated.

**Order of evaluation.** The four steps are evaluated in sequence and their verdicts are kept
separate. A large run with many failed estimates receives an operational FAIL even when its accuracy
step is also INSUFFICIENT EVIDENCE.

1. **Evidence adequacy of the run** — the number of recordings attempted, overall and in the
   operating subgroup.
2. **Operational success** — the fraction of attempted recordings that returned a valid estimate.
3. **Evidence adequacy for accuracy and coverage** — the number of successful estimates remaining,
   overall, in the operating subgroup, and in each reported quartile.
4. **Accuracy and uncertainty** — the thresholds below.

**The operating subgroup** contains the synthetic recordings whose true `mu_pc` lies in the lower half
of its log10 prior, `log10 mu_pc` in [2.00, 2.375), about 100 to 237 photons per dye. The experimental
estimates of §6.11 (139 to 186 photons per dye from the two neural estimators) motivate the emphasis on
this range; they do not establish the experimental truth. Quartile boundaries for every stratification
are the prior's own quarters in log10 coordinates, fixed here, never recomputed from the successful
recordings: `mu_pc` [2.00, 2.1875, 2.375, 2.5625, 2.75]; `mu_r` [0.00, 0.075, 0.15, 0.225, 0.30];
`sigma_r` [−1.00, −0.8125, −0.625, −0.4375, −0.25]; `lambda_rate` [0.00, 0.25, 0.50, 0.75, 1.00];
`prob_photo_bleach` [−2.00, −1.625, −1.25, −0.875, −0.50].

**Step 1 — evidence adequacy of the run.**

| requirement | value | verdict if unmet |
|---|---|---|
| recordings attempted | ≥ 1000 | INSUFFICIENT EVIDENCE (run) |
| recordings attempted in the operating subgroup | ≥ 400 | INSUFFICIENT EVIDENCE (run) |

**Step 2 — operational success.** A valid estimate is finite, and, where the estimator fits a model,
comes from a fit whose optimizer reported success. Every recording that returns no valid estimate
carries a reason code (`too_few_tracks`, `too_few_traces`, `fit_failed`, `no_apertures`, or another
named reason); a dropped recording without a reason fails the run itself, not the estimator.

| requirement | value | verdict if unmet |
|---|---|---|
| success fraction, overall | ≥ 95 % | FAIL (operational) |
| success fraction, operating subgroup | ≥ 90 % | FAIL (operational) |

**Step 3 — evidence adequacy for accuracy and coverage.**

| requirement | value | verdict if unmet |
|---|---|---|
| successful estimates, overall | ≥ 900 | INSUFFICIENT EVIDENCE (accuracy) |
| successful estimates, operating subgroup | ≥ 250 | INSUFFICIENT EVIDENCE (accuracy) |
| successful estimates in a reported quartile | ≥ 100 | that quartile is reported without a verdict |

**Step 4a — accuracy.** Required both overall and in the operating subgroup; reported, without a
verdict, in every other quartile. Correlation is computed in log10 coordinates for `mu_r` and
`lambda_rate` and in linear coordinates for `sigma_r`, matching the implementation.

| parameter | criterion | threshold | units |
|---|---|---|---|
| `mu_r` | mean absolute log10 error | ≤ 0.02 | dex |
| `mu_r` | absolute mean signed log10 error | ≤ 0.01 | dex |
| `sigma_r` | Pearson correlation with truth | ≥ 0.80 | linear coordinates |
| `sigma_r` | mean absolute error | ≤ 0.08 | **linear** units (17 % of the 0.10 to 0.56 linear range). This preserves the criterion the code has applied since 0.1.12 and corrects an inventory that had stated it in dex. |
| `lambda_rate` | Pearson correlation with truth | ≥ 0.80 | log10 coordinates |
| `lambda_rate` | mean absolute log10 error | ≤ 0.08 | dex |
| `prob_photo_bleach` | mean absolute log10 error at 1000 frames | ≤ 0.10 | dex, among **usable** recordings (below) |

**Step 4a, bleaching — three outcomes, not two.** Each attempted recording is classed as one of:
a **failed measurement** (invalid output, optimizer failure, or insufficient data; reason code
recorded); a **valid but uninformative measurement** (the fit converged but the observable
eligibility diagnostic does not support a useful estimate); or a **usable measurement** (the
recording meets the predefined observable criteria). The accuracy requirement applies to usable
recordings only, with a frozen minimum of **100 usable overall and 50 usable in the operating
subgroup**, else INSUFFICIENT EVIDENCE (accuracy). Usable and rejected fractions are reported
against **all attempted** recordings, and the recovery of rejected recordings is reported beside
that of usable ones, so that selection cannot hide a failure. The eligibility diagnostic is fixed
before the validation run, may be calibrated only on self-test scenes, and **never uses the true
bleaching value**; the information budget of §9.5 remains explanatory and takes no part in
eligibility.

**Step 4b — uncertainty.** Each estimator emits, per recording, a nominal 90 % range whose
construction is specified in its companion note; `sigma_r` receives its own range, not the mean-width
standard error. Reliability and informativeness are assessed separately.

| requirement | value | verdict if unmet |
|---|---|---|
| empirical coverage of the nominal 90 % range, overall | ≥ 85 %, with its confidence interval reported | FAIL (uncertainty) |
| empirical coverage, operating subgroup | ≥ 85 %, with its confidence interval reported | FAIL (uncertainty) |
| median range width relative to the parameter's prior width, same coordinates | reported | none — descriptive |

A broad range can have adequate coverage without providing a precise measurement. Coverage
establishes uncertainty reliability; interval width describes informativeness. Returning nearly the
whole prior is not described as successful, precise recovery, however well it covers.

**Data tiers and provenance.** EVAL tasks 0 and 1 are development data from the moment the first
direct-estimator run read them; the corrected reruns on them are regression checks, never the
adoption verdict. EVAL tasks that no direct estimator has scored form the **reserved validation set**
for that verdict, provided they stay excluded from further tuning; they are not described as globally
untouched, because the neural calibration of §6.9 has already examined every EVAL task. Development
outputs are preserved rather than overwritten, under the run folder suffixed `_DEV_<commit>`, together
with the exact commands, settings, recording identifiers, script and kernel hashes, and any
uncommitted change in the executing tree — the commit identifier alone does not capture the executed
version.

**Verdict vocabulary.** `PASS`, `FAIL (operational)`, `FAIL (protocol)`, `FAIL (accuracy)`,
`FAIL (uncertainty)`, and `INSUFFICIENT EVIDENCE (run | accuracy)`. A report states all applicable
verdicts; one does not absorb another. `FAIL (protocol)` names a drop recorded without a reason code:
a reporting failure of the run, stated beside the measured success fraction and never in place of
it, so an estimator's operational record is judged on what it measured.

**Development outcome on EVAL tasks 0 and 1 (2026-09-21, code 2b9c32e, 2000 MET-FAB 2 s recordings
each; outputs preserved under `_DEV_2b9c32e`).** The two-second estimators were run once under the
mechanics that preceded these rules, and the rules were then applied to their arrays after the fact.

*PSF width.* Every recording produced an estimate (2000 of 2000; usable tracks per recording 22 to 548,
median 114). Overall: `mu_r` MAE 0.0155 dex and bias −0.003 dex, `sigma_r` MAE 0.027 with correlation
0.96, all within threshold. In the operating subgroup (1026 recordings): `mu_r` MAE 0.017 dex meets the
threshold, `mu_r` bias −0.0118 dex misses the 0.01 dex bound, about 2.7 % underestimation; `sigma_r` MAE
0.031 and correlation 0.96 meet theirs. Verdict `FAIL (accuracy)` on that one criterion. The bias runs
monotonically with true brightness, from −0.020 dex at the dim end of the prior to +0.009 dex at the
bright end, and `sigma_r` is underestimated most where the true spread is broadest (−0.057 in the broad
quarter of the operating subgroup). The brightness dependence alone does not establish the mechanism:
detection selection, fitting bias, and track selection are all candidates. Decision: continue with the PSF
estimator; rerun it under the corrected mechanics to obtain ranges and reason codes; keep the thresholds
and the operating range unchanged, since the dim recordings are the ones the experiments contain; assess
`mu_r` and `sigma_r` separately; investigate a correction on development data that uses only quantities
available on experimental recordings, and validate it on the reserved EVAL tasks.

*PSF width, regression run of record under the corrected mechanics (2026-09-21, 0.1.13 working tree,
same 2000 recordings, 36 minutes on 28 workers; output `..._Direct_PSF_Width` with `PROVENANCE.md` and
`COVERAGE_DIAGNOSIS.md`).* The point estimates are identical to the development run, so steps 1 to 3 pass
with every drop now carrying a reason code (none occurred), and step 4a repeats `FAIL (accuracy)` on the
operating-subgroup `mu_r` bias of −0.0118 dex. Step 4b, evaluated for the first time, is
`FAIL (uncertainty)`: the nominal 90 % ranges cover 65.9 % (`mu_r`) and 63.3 % (`sigma_r`) overall and
60.0 % and 55.7 % in the operating subgroup, against the 85 % rule; the ranges are informative, with
median widths of 0.105 and 0.113 of the prior widths. The diagnosis is specific. The standard error is
about 2.3 times too small at every track count, for both quantities, so the sampling-only construction
misses a variance component beyond track-to-track sampling, and an inflation of 1.85 to 1.95 would be
needed for 90 % coverage whether or not the mean offset is removed. On top of that the dim operating
subgroup carries a negative offset (mean standardized error −1.06 for `mu_r`) and `sigma_r` shrinks toward
the middle of its prior (+1.05 in the narrowest quarter, −1.42 in the broadest). The `mu_r` error
correlates with the true brightness (+0.46) far more than with the observable spot count (+0.25), so the
spot count is a weak experimental proxy; a per-recording measured spot brightness or signal-to-noise
ratio, which the script does not yet emit, is the candidate proxy for any correction calibrated on
development data and validated on the reserved EVAL tasks.

*Flicker rate.* 1909 of 2000 recordings produced an estimate; the 91 drops all had fewer than 29 usable
traces, and 73 % of them lie in the dimmest brightness quarter. The success fractions (95.5 % overall,
91.6 % operating) meet the rules; the drops carried no reason code, which is the `FAIL (protocol)`
above and not a measurement failure. Correlation 0.847 meets its threshold. MAE 0.144 dex and bias
+0.110 dex fail the 0.08 dex bound; in the operating subgroup the bias is +0.142 dex, about 39 %
overestimation. The bias depends on the true rate, +0.26 dex for the slowest flicker near 1 per second
falling to +0.04 dex above 5 per second, and on brightness, +0.16 dex dim against +0.06 dex bright. The
within-quarter correlations are low but shrink with the truth range by construction; the signed errors
are the evidence. The four single-dye 6 s self-test scenes at the prior center, which passed at 0.06 dex,
probed none of these conditions. The exact-parabola correction accounts for about 0.005 dex of this and is
not the explanation. The model arm already matches track spans and detrends; what it lacks is a faithful
treatment of the observed traces: gaps, detection and linking selection, measurement noise, dye
multiplicity, and bleaching. The result shows poor recovery under production conditions; it does not
isolate which omission causes it. Decision: pause the unchanged two-second rerun; study the mismatch on a
small representative development subset, adding the omitted effects to the model arm one at a time and
recording which closes the bias; and compare against estimation over the full experimental recording,
propagating one rate and its uncertainty to the windows, if the rate can be treated as constant over a
recording and the assumption is validated at that duration. The direct estimators are also to be compared
with the neural posterior estimator on the same multiple-dye recordings by bias, MAE, and uncertainty,
not correlation alone. All three targets (`mu_r`, `sigma_r`, and `lambda_rate`) remain in the
implemented neural block pending validated replacements; this retains the implementation and does not
validate the current neural estimates or their uncertainty.

**Point estimates and uncertainty answer different questions.** `mu_r` and `sigma_r` describe the
per-subunit PSF-width distribution: its median and log-space spread. Comparing point estimates establishes
which method recovers these parameters more accurately. Comparing uncertainty ranges establishes how reliably
each method quantifies its remaining error. Failure of the latter does not reverse a demonstrated advantage in
point accuracy.

The biology workflow samples imaging vectors from `Nuisance_DLI`; a fixed photophysics vector is an available
special case (§7.2), not the universal implementation contract. If that option is selected, biological
inference is conditional on the chosen photophysics values. Emitter-to-emitter PSF variation remains
represented by `sigma_r`, but uncertainty about `mu_r` and `sigma_r` is not propagated by fixing them.

**Head-to-head on point values, PSF parameters (2026-09-21; stored under
`..._DETECTOR_FAB_2S_50FPS_PSF_Direct_vs_Neural` with arrays, statistics, and one four-panel figure per
parameter, each method on its own evaluated sample).** The direct estimator on its 2000 EVAL recordings
against the three point estimates of the baseline multiple-dye neural posterior estimator, MAP, posterior
median and sample geometric median, read together. The 2,000 direct recordings have unique matching
neural rows with identical six-parameter ground truths. The matched-subset and full-set neural statistics are
similar, but not identical. The numerical comparison below uses the matched subset; the figures retain
separate panels showing each method's full evaluated sample (25,000 recordings for the neural estimator).

| parameter | method | n | slope | correlation | MAE (dex) | signed bias (dex) | outside prior |
|---|---|---:|---:|---:|---:|---:|---:|
| `mu_r` | direct | 2,000 | 0.97 | 0.967 | 0.0155 | −0.0031 | 4 % |
| `mu_r` | neural MAP | 2,000 | 1.03 | 0.687 | 0.0795 | +0.0294 | 26 % |
| `mu_r` | neural posterior median | 2,000 | 0.93 | 0.955 | 0.0254 | +0.0218 | 0 % |
| `mu_r` | neural SGM | 2,000 | 0.93 | 0.952 | 0.0259 | +0.0220 | 0 % |
| `sigma_r` | direct | 2,000 | 0.89 | 0.963 | 0.0460 | −0.0101 | 2 % |
| `sigma_r` | neural MAP | 2,000 | 0.03 | 0.065 | 0.1978 | −0.0499 | 0 % |
| `sigma_r` | neural posterior median | 2,000 | 0.04 | 0.152 | 0.1870 | −0.0162 | 0 % |
| `sigma_r` | neural SGM | 2,000 | 0.03 | 0.129 | 0.1882 | −0.0169 | 0 % |

Errors are expressed in log10 units here for comparison. The direct estimator is not bounded by the prior
box; its excursions (85 and 42 recordings) sit at the prior edges and reach at most 0.07 and 0.12 dex. The
frozen `sigma_r` acceptance criterion remains in linear units. Direct estimation uses the supplied simulated camera settings; neural inference marginalizes
over the camera nuisance distribution.

For `mu_r` the direct estimate and the two neural posterior summaries track the truth strongly (fitted
slopes 0.97 direct, 0.93 median and SGM); the neural MAP tracks it far less well (correlation 0.69, MAE
0.080 dex, 26 % of estimates outside the prior box). That was attributed to density spikes; §9.8 traces it
instead to the MAP routine's own defect, so the MAP row here is under recomputation.
The neural summaries carry an average positive offset of approximately 0.02 dex (MAP 0.03 dex), the direct
estimate a small overall bias (−0.0031 dex), with a brightness-dependent residual bias of −0.0118 dex in the
dim subgroup. For `sigma_r` all three neural estimates return nearly the same value whatever the truth
(fitted slopes 0.03 to 0.04, correlations 0.07 to 0.15) and do not recover the parameter; the direct
estimator shows compressed recovery, with a fitted slope below one (0.89). The direct estimator substantially improves recovery of `sigma_r` compared with the
current neural estimator. For `mu_r`, both methods recover the parameter well; direct estimation gives lower
average error and bias on the matched evaluation subset, but the advantage is more modest. These results
strongly support direct estimation of the PSF spread, while leaving both methods viable candidates for the
typical PSF width.

**Conclusion of record (2026-09-21).** On the matched multiple-dye synthetic recordings, the direct
estimator provides better point estimates than any of the three neural point estimates for both PSF parameters,
including in the dim operating subgroup. It is therefore the preferred candidate for supplying PSF point
values in experimental imaging calibration. This comparative result does not change the frozen acceptance
outcomes: the dim-subgroup `mu_r` bias exceeds its threshold, and both reported uncertainty ranges under-cover
(65.9 % and 63.3 % overall, about 60 % and 56 % in the dim operating subgroup; a bounded correction is to be
tested). Experimental deployment and the choice between fixed values and sampled nuisance inputs are separate
decisions. The experimental cross-check compares direct experimental PSF estimates with the ThunderSTORM
references (§6.11) after aligning width conventions, units, and population summaries; ThunderSTORM is a
reference method, not ground truth, and agreement or disagreement must account for fitting uncertainty and
detection selection, particularly for width spreads. The direct flicker estimator is not adopted as a
replacement: its bias of +0.11 dex overall and +0.14 dex in the dim subgroup, rate-dependent, calls for an
investigation of the measurement model and of the recording duration before further tuning, and that
diagnosis guarantees no outcome. The frozen rules produced this information as intended: both estimators had
passed their own self-tests and earlier acceptance reports.

**What this means for the biology.** The diagnostics identify imaging quantities and regimes that need
particular care, but do not yet establish reliable nuisance distributions. Fixing these quantities to
unsupported point values is to be avoided, and the ranges used to represent their uncertainty must be
validated before they are propagated; passing biased estimates or under-covered neural posteriors
downstream would not solve the problem. The detector capacity test (§9.7) can identify a promising
estimator configuration; any transfer to biology still requires biological-parameter recovery and
calibration tests. The central conclusion: keep developing the direct measurements, retain the current
implementation provisionally, and do not confuse either choice with validated imaging inputs for biology.

### 9.7 The capacity test on the multiple-dye baseline

The multiple-dye estimator of §6.10 recovers the parameters but under-covers: 62 % of the true
values fall inside its nominal 90 % joint region, and its marginal ranks are far from uniform. §6.10
lists the candidate causes and states that the comparison with the one-dye law does not isolate an
estimator-only failure. One of those causes is testable without touching the data, the targets, or the
priors: the estimator may lack capacity for the multiple-dye videos, whose brightness distribution is a
mixture over the dye count. A sibling repository has met this failure mode before, where a 64-dimensional
embedding produced impossible estimates that a 128-dimensional one did not. The capacity test trains one
larger estimator under otherwise identical conditions and compares it with the baseline. It is a
diagnosis of the estimator, not a change to it: the baseline stays the working estimator of record until
the comparison is read.

**What changes and what does not.** The embedding is widened from 128 to 256 dimensions by doubling
every convolutional block's channels (`start_channels` 8 → 16 with the same five blocks, so the deepest
block carries 16 · 2⁴ = 256 features per temporal token, 64 per attention head), and the flow is
enlarged to `hidden_features` 128, `num_transforms` 8, `num_blocks` 2, `dropout_probability` 0.1. The flow
settings are now an explicit configuration (`InferenceFlow`, whose defaults are the library values the
earlier estimators used, so a run without the preset reproduces them exactly) and are persisted in the
saved estimator's rebuild specification together with the embedding arguments. Both changes are one named
preset, `capacity256` (`NETWORK_PRESETS`), selected with `--network-preset`. Everything else is held fixed:
the multiple-dye datasets and splits (TRAIN 200 tasks, TEST 50, EVAL 25), the six detector targets and
their priors, the preprocessing and standardization, batch normalization, the optimizer, the learning-rate
schedule, and the number of epochs. The test is not combined with any parameter removal (§9.4) or prior
change.

| | baseline | `capacity256` |
|---|---|---|
| Embedding parameters (conv stack + temporal transformer + head) | 0.691 M | 2.757 M |
| Flow parameters | 0.064 M | 0.551 M |
| Total | 0.755 M | 3.308 M |
| Conv-stack activations kept for the backward pass, per 2 s video (fp32) | 0.85 GiB | 1.70 GiB |
| First block's output tensor, per video | 8 × 100 × 256 × 256 (0.20 GiB) | 16 × 100 × 256 × 256 (0.39 GiB) |

**Where the memory goes.** The additional weights are small in memory terms (3.3 M parameters with their
gradients and optimizer moments occupy well under 100 MiB). The memory growth is in the embedding's
activations, which the backward pass must keep: every convolutional block's output doubles with its
channel count, and the first block, which works at full spatial resolution, dominates. The flow adds
negligible activation memory, because it operates on the six-dimensional parameter vector conditioned on
one 256-dimensional embedding per video. The activation estimate is a lower bound on device memory: the
compiled graph's workspaces, the batch-normalization statistics, the gradient buffers, and the
data-loader staging add to it, so the smoke test below measures the actual peak.

**Batch geometry.** The baseline trained with 32 videos per rank on 8 nodes × 4 GPUs (32 ranks), a global
batch of 1024. The activations per video double, so 32 videos per rank would bring the conv-stack
activations alone from about 27 GiB to about 54 GiB per GPU before workspaces, too close to the device
for comfort. Halving the per-rank batch restores the baseline's activation footprint exactly, so the
capacity run uses 16 videos per rank on 16 nodes × 4 GPUs (64 ranks): the global batch stays 1024, the
number of optimizer steps per epoch is unchanged, and because batch normalization is synchronized across
ranks under data-parallel training, the batch statistics are still computed over the same 1024 videos
per step. The per-rank batch is the one fixed setting this test relaxes, and it relaxes it for device
memory alone, which is the documented condition for changing it. The training log now prints each
rank-0 epoch's peak allocated and reserved device memory next to the epoch time, so both quantities are
recorded for the two configurations.

**Naming.** The capacity run's products carry the artifact tag `CAP256` right after the timing label
(`Paths.product_label`, `--artifact-tag`, dispatcher knob `ARTIFACT_TAG`):
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_CAP256_Estimator.npz`, the matching checkpoint and
resurrect state, `..._CAP256_MAP_Recovery`, `..._CAP256_Posterior_Calibration`, `..._CAP256_MAP_Experiment`,
and the job names `..._2S_50FPS_CAP256_Inference` / `_Evaluation` / `_Experiment`. The shared inputs (video,
theta, and record sets, the experimental recordings) are read under the plain timing label. The baseline
products keep their names and are not overwritten. The tag is a SCREAMING_SNAKE token without underscores
so the runtime grammar stays unambiguous; the estimator manifest records the tag and the preset.

**Protocol.**

1. *Smoke* (code and memory check, deleted afterwards): one node × 4 GPUs, 30 minutes, TRAIN 4 tasks /
   TEST 1 task, one epoch, batch 8, for both presets in turn, under the tags `SMOKEBASE` and
   `SMOKECAP256` (distinct, so the two jobs never share a checkpoint path). It records the peak device
   memory and the epoch time of each configuration, from which the batch-16 footprint follows, and
   confirms that the tagged products, the rebuild specification, and the downstream loaders work. Its
   estimates carry no scientific meaning.
2. *Training*: 16 nodes × 4 GPUs, batch 16 per rank, TRAIN 200 / TEST 50, 50 epochs within a 12 h wall,
   then a second 50-epoch run continued with `--resurrect`, matching the baseline's 100 epochs.
3. *Comparison* on the same EVAL tasks the baseline used: MAP recovery (bias, MAE, and correlation per
   parameter), marginal and joint calibration (§6.9's tests: rank uniformity, expected coverage, TARP,
   L-C2ST), posterior widths against the prior widths, and the failed-estimate rates, each overall and in
   the operating subgroup of §9.6 (true `log10 mu_pc` in [2.00, 2.375), the lower half of the brightness
   prior), where the baseline's difficulties concentrate. Training time and peak memory per configuration
   are reported alongside.
4. *Three-way comparison with the direct estimators.* The comparison is not only against the baseline. For
   `mu_r` and `sigma_r` the direct PSF-width estimator's point values on EVAL tasks 0 and 1 (§9.6, the
   `..._PSF_Direct_vs_Neural` record) are the third column: slope, intercept, correlation, MAE, and signed
   bias of the capacity estimator's three point estimates (MAP, posterior median, SGM) on the same 2000
   recordings, beside the baseline's three and
   the direct estimator's, each method plotted on its own evaluated sample. For `lambda_rate` the direct
   flicker estimator's development result (bias +0.11 dex, correlation 0.85) is the reference the capacity
   estimator's recovery is read against. A capacity estimator that recovers `sigma_r` no better than the
   baseline leaves the direct estimator as the source of that value; one that matches or exceeds the direct
   estimator changes which measurement supplies it.

**Reading the result.** The larger embedding and flow are relevant primarily as candidates for the
biology estimator; the Detector problem is a controlled benchmark with known imaging truth, cheaper to run
and to read. A gain here motivates the same test on the biology workflow, with its own recovery and
calibration against known biological truth, and does not by itself establish better biological inference
(§2). If the larger estimator recovers the parameters with calibrated marginals and a
joint coverage near nominal, the baseline's under-coverage was a capacity limit, and the choice between
the two estimators is a cost question. If calibration does not improve while recovery does, or neither
improves, capacity is not the binding constraint, and the remaining candidates of §6.10 (the mixture
structure of the multiple-dye brightness law, the data budget, the identifiability of individual blocks)
move forward. Either outcome is informative; neither replaces the validation on experimental recordings
that §6.10 lists as open.

**Smoke result (2026-09-21, JUPITER booster, one node × 4 GH200 with 96 GiB each, batch 8, TRAIN 4 /
TEST 1, one epoch; products deleted afterwards, job logs kept).** Both presets ran to completion in about
six minutes each, saved a tagged estimator whose rebuild specification carried the preset's embedding and
flow arguments, and loaded back.

| per GPU, batch 8 | baseline | `capacity256` | ratio |
|---|---|---|---|
| peak allocated | 11.9 GiB | 23.5 GiB | 1.97 |
| peak reserved | 19.4 GiB | 38.6 GiB | 1.99 |
| epoch over 4000 videos on 4 GPUs | 53.6 s | 53.6 s | 1.00 |

The memory doubles exactly as the activation count predicts, and at this small scale the epoch time is
set by input throughput, not by the network, so the production epoch time has to be read from the
production run itself. Scaling the allocation linearly, `capacity256` at 16 videos per rank needs about
47 GiB allocated and 77 GiB reserved per GPU, the same footprint the baseline had at 32 videos per rank
on the same devices, which the baseline production run already sustained.

**Training (2026-09-21, JUPITER jobs 1929130 and 1929732).** Two legs of 50 epochs on 16 nodes × 4 GH200,
16 videos per rank, TRAIN 200 / TEST 50 tasks, the second leg continued with `--resurrect` from the first
leg's state (global epochs 51 to 100). Each leg took 1 h 46 min of wall time, 120 s per epoch after the
first, at a steady 46.8 GiB allocated / 77.1 GiB reserved per GPU, against the baseline's two legs of 2 h 47
min at 32 videos per rank on 8 nodes × 4 GPUs. Best TEST loss (mean negative log-probability of the TEST
set): −7.50 after the first leg and −7.81 after the second, against the baseline's −5.35 and −5.69 at the
same epochs. A lower TEST loss is the model-selection criterion, not a ranking of estimators; the comparison
that decides this test is the recovery and calibration of the two estimators on the same EVAL recordings
(protocol step 3), which follows. Products live under the `CAP256` tag; the baseline products are untouched.

**Comparison (2026-09-22, JUPITER jobs 1951212 Evaluation, 1951221 Posterior_Calibration, 1951236
Experiment; record `..._2S_50FPS_CAP256_vs_Baseline_Evaluation` on the PC Posit tier, with the Experiment
comparison in `..._CAP256_vs_Baseline_Experiment`).** Same 25,000 EVAL recordings, same tests, same
production code; rows aligned on the true parameters. The three neural point estimates are read together.

| | baseline | `capacity256` |
|---|---:|---:|
| joint expected coverage, nominal 0.50 / 0.90 | 0.24 / 0.62 | 0.44 / 0.87 |
| largest joint coverage gap | 0.308 | 0.065 |
| TARP area-to-curve | −0.053 | −0.025 |
| L-C2ST rejection fraction | 0.998 | 0.000 |
| joint coverage at 0.90, operating subgroup | 0.49 | 0.85 |
| SBC KS D: `mu_r` / `sigma_r` / `mu_pc` / `sigma_pc` / `prob_photo_bleach` / `lambda_rate` | 0.447 / 0.121 / 0.178 / 0.061 / 0.036 / 0.053 | 0.103 / 0.048 / 0.116 / 0.061 / 0.031 / 0.044 |
| location errors (bias in units of posterior sd, over 0.5) | `mu_r` −1.22, `mu_pc` −0.53 | none |
| MAP outside the prior box on at least one parameter | 34 % | 49 % |
| GPU-hours for 100 epochs | about 178 | about 226 |

Recovery on the full EVAL set, the three point estimates side by side (baseline → `capacity256`; correlation
with the truth, MAE in dex, signed bias in dex):

| parameter | MAP corr | MAP MAE | MAP bias | median corr | median MAE | median bias | SGM corr | SGM MAE | SGM bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mu_r` | 0.67 → 0.56 | 0.079 → 0.109 | +0.026 → −0.006 | 0.96 → 0.96 | 0.025 → 0.018 | +0.021 → −0.001 | 0.96 → 0.96 | 0.025 → 0.019 | +0.021 → +0.001 |
| `sigma_r` | 0.08 → 0.07 | 0.195 → 0.247 | −0.048 → −0.185 | 0.17 → 0.27 | 0.186 → 0.180 | −0.016 → +0.012 | 0.15 → 0.25 | 0.186 → 0.181 | −0.017 → +0.006 |
| `mu_pc` | 0.87 → 0.84 | 0.095 → 0.114 | +0.035 → −0.014 | 0.95 → 0.97 | 0.054 → 0.044 | +0.031 → −0.009 | 0.95 → 0.96 | 0.056 → 0.046 | +0.031 → −0.011 |
| `sigma_pc` | 0.79 → 0.75 | 0.112 → 0.129 | −0.013 → +0.007 | 0.89 → 0.89 | 0.082 → 0.082 | +0.002 → +0.008 | 0.89 → 0.89 | 0.083 → 0.083 | +0.004 → +0.011 |
| `prob_photo_bleach` | 0.77 → 0.09 | 0.214 → 0.471 | +0.059 → +0.070 | 0.79 → 0.16 | 0.206 → 0.369 | +0.028 → +0.004 | 0.78 → 0.13 | 0.210 → 0.371 | +0.025 → +0.004 |
| `lambda_rate` | 0.46 → 0.41 | 0.214 → 0.226 | +0.018 → −0.015 | 0.54 → 0.54 | 0.201 → 0.199 | −0.008 → −0.005 | 0.53 → 0.54 | 0.202 → 0.200 | −0.005 → +0.002 |

The MAP's share of recordings outside the prior box rises on every parameter (`mu_r` 26 % → 35 %, `mu_pc`
8 % → 12 %, `sigma_pc` 6 % → 9 %, `sigma_r` 0 % → 4 %); median and SGM stay inside it in both runs. The MAP
is worse in the capacity run on every parameter; the median and the SGM, which agree within 0.005 to
0.04 dex in both runs, improve on `mu_r` and `mu_pc`, hold on `sigma_pc` and `lambda_rate`, move `sigma_r`
a little and collapse on `prob_photo_bleach`.

The capacity run improves the joint coverage substantially and removes the baseline's two location errors, with
residual miscalibration remaining (largest joint gap 0.065 against the 0.05 reference, marginal 90 % coverage
83 to 86 %, SBC still flagging `mu_r`, `mu_pc` and `sigma_pc`, bleaching coverage 87 % → 85 %); its posterior summaries recover `mu_r` and `mu_pc` with smaller error and no
offset (also in the operating subgroup, where the baseline's offsets were 0.03 to 0.06 dex), leave `sigma_pc`
and `lambda_rate` unchanged, and move `sigma_r` little (slope 0.10 against 0.04; not recovered). It does not
recover `prob_photo_bleach`: its posterior standard deviation on that parameter is 26.8 % of the prior range
(median over recordings) against 28.9 % for the uniform prior itself (baseline 17.0 %), its median sits at the prior center whatever
the truth, and it correlates −0.46 with the true `mu_pc` and +0.31 with the true `lambda_rate` against +0.16
with the true bleaching probability; the same width appears in the stage's own report, and on the recordings
the run's bleaching MAP drifts across windows while its median and SGM stay flat (§6.11 companion record). The
MAP separates further from the posterior in the capacity run on every parameter (outside the posterior's own
90 % interval on 85 % of recordings for `mu_r`, 74 % for `mu_pc`; `sigma_r` MAP offset −0.185 dex), so the MAP
recovery columns are worse while the posterior is better calibrated.

*Three-way (step 4).* On the matched 2,000 recordings the direct PSF-width estimator recovers `sigma_r` with
slope 0.89 / MAE 0.046 dex against at most 0.09 / 0.182 for any neural point estimate of either run; the
direct estimator stays the source of that value. For `mu_r` the capacity run's median and SGM reach the
direct estimator's accuracy (MAE 0.018 against 0.016 dex, bias −0.001 against −0.003) and remove the
baseline's +0.02 dex offset; in the operating subgroup the capacity median has the smaller bias (+0.001
against −0.012) and the direct estimate the smaller MAE (0.017 against 0.020). Both are candidates for `mu_r`.
For `lambda_rate` the neural medians of both runs give correlation 0.55, MAE 0.20 dex and no average bias on
EVAL tasks 0–1 against the direct flicker estimator's 0.85 / 0.144 / +0.110 dex, a reference comparison rather
than a matched one (the direct figures are over its 1,909 successful recordings, whose 91 failures concentrate in
dim recordings; the neural figures over all 2,000); the capacity change left the rate where the baseline had it.

*Reading.* By the protocol's own criterion, calibration improved with recovery for `mu_r` and `mu_pc`, so the
baseline's under-coverage on those and on the joint was within reach of capacity, without being fully closed.
This capacity increase did not resolve `sigma_r` recovery; one architecture change and one training run do not
exclude other capacity or optimization explanations. The loss of `prob_photo_bleach` was not anticipated by the protocol and is measured
from one training run of one configuration; the next measurements are a repeat training of `capacity256`
under identical settings, to test whether the bleaching collapse repeats, and the 20 s tier, where the
bleaching information budget is far larger (§9.5), to test whether either estimator's bleaching posterior
narrows with duration. Neither estimator is adopted by this comparison: the baseline stays the estimator of
record; `capacity256` is the preferred candidate on calibration and on `mu_r`/`mu_pc` recovery and is not
adoptable while bleaching is unrecovered. The result concerns the detector benchmark; the biology estimator
needs its own capacity test (§2).

**Status.** Training, Evaluation, Posterior_Calibration and Experiment complete under the `CAP256` tag;
comparison recorded above. Open: repeat training of `capacity256`; the 20 s tier (in generation on JUWELS).

> **The MAP columns of this section are under recomputation (§9.8).** The MAP routine returned
> coordinates one optimizer step away from the point whose density it reported, so every MAP number
> above, and the reading that `capacity256`'s MAP "separates further from the posterior", is
> provisional until the affected stages are re-run. The calibration comparison, the posterior-median
> and SGM recovery, and the `prob_photo_bleach` collapse are unaffected: they come from posterior
> draws and never from that routine.

### 9.8 A defect in the MAP routine, and what it puts under recomputation

The seed-then-optimize step of §6.8 draws a candidate pool, keeps the best `K` seeds, and gradient-ascends
the flow's log-density from them, returning the best `(score, theta)` it saw. Until 0.1.15 it did not: the
bookkeeping ran after `optimizer.step()`, which updates the parameter tensor in place, so the recorded pair
combined one step's score with the next step's coordinates. The returned vector therefore sat about one
learning rate away from the point whose density was reported, in every coordinate whose gradient pushed
consistently, and the reported score belonged to the point before the move.

**Confirmation.** An analytic density with one smooth peak and no secondary structure reproduces it without
any trained flow: driving the unchanged routine at a mode of 1.0 returned 0.97739 when the seeds approached
from below and 1.02261 when they approached from above, reporting the mode's score in both cases. On the
stored products the fingerprint is explicit: `|MAP − posterior median|` carries a sharp spike at exactly
0.128 dex, the initial Adam learning rate (`learning_rate_minimum` 1e-3 × `learning_rate_maximum_factor`
128, which the stage prints as `learning_rate: 1.280e-01`), standing 3.4× above the neighboring background
in the baseline; `mu_r` lands within 10 % of exactly that value for 51 % of baseline and 66 % of `CAP256`
recordings. The three `mu_r` MAP bands at −0.13, 0 and +0.13 dex are that displacement, not posterior
geometry.

**The fix and its invariants.** The bookkeeping now runs before the update, which makes two statements true
by construction: the returned score is the density at the returned vector, and, because the first step scores
the elite seeds themselves, it is never worse than the best seed's. Both are covered by
`tests/test_map_optimizer_invariant.py`. No estimator is retrained and no prior, role or preprocessing step
changes.

**A second statement was wrong.** The reports said an outside-prior estimate was possible only under
`--pool-mode unrestricted`. The ascent is unconstrained under either mode: the pool mode bounds the candidate
pool, not the steps. The `CAP256` Evaluation ran `pool_mode: bounded` and still placed 35 % of its `mu_r`
estimates outside the prior box. Such an estimate is a flow optimum, not a MAP of the prior-supported
posterior. Whether to constrain the ascent is a separate decision, open.

**Scope.** Affected, and therefore under recomputation: every MAP array, statistic and figure of the
Evaluation and Experiment stages for this model — the multiple-dye baseline, `CAP256`, and the one-dye run —
together with what is derived from them, namely the MAP rows of the within-recording drift and
point-estimate agreement tables, the MAP columns of §§6.9 to 6.11, §9.6 and §9.7, the two `CAP256`-versus-baseline
comparison records, the neural MAP rows of the direct-versus-neural PSF record, the MAP-vector geometric-median
utility, and any posterior-predictive render whose parameter vector came from a MAP. The synthetic validation
arm of the population-composition analysis reads MAP estimates and is affected; its experimental readout reads
posterior draws and is not.

Not affected, because they never pass through this routine: the trained weights and their training and test
losses, the posterior draws, the per-dimension posterior medians and the posterior-draw SGM, every sampling
based calibration diagnostic (coverage, SBC, TARP, L-C2ST), and the direct estimators. The `capacity256`
calibration gain and its `prob_photo_bleach` collapse both stand, the latter being measured in the median and
the SGM.

A caution the artifact format carries: the Nuisance_DLI pool cache records the checkpoint checksum and the
sampling settings but not the optimizer implementation, so correcting the code does not invalidate an existing
MAP-derived pool. Any such pool must be rebuilt explicitly, and a construction named "SGM" is not by itself
safe: `sgm_percentiles` with `selection_source = "experiment"` (the default) summarizes stored MAP vectors,
while `window-sgm` summarizes posterior draws.

**Status.** Fixed in 0.1.15 and validated on the analytic case. The `CAP256` Experiment is the first stage
re-run (JUPITER 1953816, with the per-window optimizer trace recorded); the previous product is preserved
beside it as `..._CAP256_MAP_Experiment_RUN_1951236` so the two can be compared on the same 600 windows.

From 0.1.16 every Evaluation and Experiment product carries all three point estimates for every
observation -- `map_estimate`, the marginal median of the draws (the 0.50 level of `posterior_quantiles`)
and `posterior_sgm` -- with a `manifest_json` recording the computation contract (optimizer settings,
draw count and label, quantile levels, SGM scaling, the optional arrays the run stores, seed policy,
window geometry and condition labels for the experiment stage, code identity at startup and at write,
the checksum of the checkpoint actually loaded, the launcher's invocation identity, and the execution
attempt that wrote each product). The manifest is validated in full, not
merely stored, and the writers validate before they persist or render; every consumer reads a
product through the schema and requires its parameter keys to match the workflow's exactly. The
estimate-selection option is retired, the stored MAP field is renamed from `inferred_log10` to
`map_estimate` with no fallback, and the definitions live in one place (`evaluation.POINT_ESTIMATES`).
Renaming a field certifies no numerical correction: the products listed above are recomputed, not
converted.
