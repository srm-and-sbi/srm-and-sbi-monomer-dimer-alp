# Posterior-predictive check in the embedding space — usage and interpretation

Authoritative companion for **both** workflows' posterior-predictive embedding checks:
`SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Embedding.py` (biology) and
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Embedding.py` (detector). Both are thin shims
over one shared engine, `srm_and_sbi_monomer_dimer_alp.posterior_predictive_embedding_runner`.

The embedding-distance analysis (`SRM_AND_SBI_MONOMER_DIMER_ALP_Embedding_Space_Distance.md`) asks whether
the experimental windows lie inside the cloud of synthetic EVAL videos. Those videos are drawn from
the whole training prior, so that is a statement about the prior: the recordings are, or are not,
something the training distribution contains. This analysis asks the sharper question, one window at a
time: **do videos simulated at the parameters the estimator assigns to a window look, to the network,
like that window?** A window can lie inside the EVAL cloud and still be far from every video its own
posterior produces; the estimate for that window then points to the wrong region of the prior.

It is a post-hoc analysis, not a pipeline stage: it lives in `Script_Bank/Analysis`, is never wired
into the stage dispatcher, changes no product and selects nothing.

## What a predictive video is

For every experimental window the analysis takes complete joint draws from the window's stored
posterior (whole rows of its retained cloud, never recombined per coordinate) and simulates one video
per draw through the production chain: the reaction-diffusion simulation, the condition's labeling,
the renderer and the training data's storage conversion (16 to 8 bit). Each video is a fresh window of
the estimator's length, as every training video is.

Which parameters come from the posterior depends on the workflow:

| | from the window's posterior | drawn as the training data draw them |
|---|---|---|
| **detector** | the six imaging parameters | the five camera parameters (the acquisition record's box) and the reaction-diffusion parameters (the condition's biology prior) |
| **biology** | the reaction-diffusion parameters | the six emitter parameters (the condition's `Nuisance_DLI` artifact) and the five camera parameters (the acquisition record's box) |

So a window's predictive cloud is the model's distribution of videos given the window's *marginal*
posterior of the inferred block, **averaged over the training distribution of the nuisances**. That is
what the check tests: it does not reproduce the unknown nuisance settings of that window. For the detector the nuisances include the receptor number and the mobility,
which shape a video strongly, so a detector predictive cloud is broad and "inside its own cloud" is a
weaker statement there than in the biology workflow, where every parameter that shapes the
trajectories is inferred.

**Which draws are admitted (`--pool`).** `bounded` (the default) admits only draws inside the prior box
in every coordinate: the simulator's training tiers hold no system outside the box, and posterior mass
outside it is the flow leaking past the prior's support. `unrestricted` admits draws outside the box
on the logarithmic rows (those videos extrapolate beyond the training support) but never on a linear
row, where a fraction outside [0, 1] is not a realizable composition. The plan records the share of
each window's cloud that was admitted, and the report states the excluded share of every window and
its largest and median value: the predictive videos describe the admitted part of a posterior only,
and an excluded share of several percent is part of the reading, not a detail.

**The window layout is the one of the posterior draws.** The number of frames per window, the step
between window starts and the recording span are read from the source of the draws; chunk `w` of a
recording must be the window whose posterior is stored as chunk `w`. `--chunk-step-seconds` and
`--experiment-span-seconds` are cross-checks: omitted, they follow the stored layout (recordings of
any stored length are found without naming it); given, a value that contradicts it stops the plan. A window whose admitted draws are fewer than
`--draws-per-window` stops the plan.

## The statistic

The videos of a window and the window itself are embedded through the trained network. The window's
distance to its `k`-th nearest predictive video (`--knn-k-cloud`, default 5) is ranked against the
same distance among the predictive videos. The **tail share** is the share of predictive videos at
least as isolated as the window. With `S` predictive videos the smallest
value is 1 / (S + 1), so 50 videos resolve the tail to about 0.02. The reading is given in the raw
embedding and in the standardized one (every coordinate divided by its spread over the window's
predictive videos and the window together, so that the window is treated like the videos).

**The tail share is not a calibrated p-value.** The window determined the posterior its comparison
videos were drawn at, so the window and those videos are not exchangeable, and no flag rate follows:
there is no number of windows "expected outside" under a correct model. Ranks of this kind tend to
avoid the tails when the model is right. A low tail share is a warning about the *combination* of
forward model and inference procedure. It does not say which of the two is at fault: the posterior can
be wrong although the simulator is right, and the reverse. The measure is
`embedding_space_distance.knn_typicality`, the same one the embedding-distance analysis applies to the
EVAL cloud.

When the embedding-distance analysis has stored the EVAL embeddings of the same estimator
(`embeddings.npz` in its output folder), the report adds for every window its tail share in the EVAL
cloud, and the two are read together:

| in the EVAL cloud | in its own predictive cloud | reading |
|---|---|---|
| inside | inside | the window is in range, and videos simulated at the inferred parameters look like it to the network |
| inside | outside | such videos occur in the training distribution, but not at the parameters inferred for this window: the forward model or the posterior of this window is off |
| outside | outside | neither the training distribution nor the predictive videos contain such a window |
| outside | inside | unusual: the window is isolated among the EVAL videos yet close to its own predictive videos; the posterior sits in a sparsely sampled corner of the prior |

A stored reference of another estimator is refused: embeddings of two networks are not comparable.

**Where the predictive videos lie in the EVAL cloud.** For every window the report gives the share of
its predictive videos that are themselves in the thin outskirts of the EVAL cloud (tail share below
0.05 there). A large share does not by itself indicate a fault: a posterior can concentrate, with valid
parameters, in a region the prior populated thinly. It is a reason to look at the videos, at the
storage and labeling steps and at where the posterior sits in the prior box, without identifying a
cause.

**Pixel-level comparison.** The embedding need not retain every difference between videos, so the
report also places every window among its predictive videos by six pixel statistics, computed on the
videos as the network receives them (stored 8-bit values): the median, the 99th and the 99.9th
percentile and the maximum of the pixel values, the share of pixels at the top of the range (the
clipped fraction), and the frame-to-frame variation of the mean pixel value. For every statistic the
report counts the windows that lie outside the 5-95 % range of their predictive videos. It needs no
further simulation and is descriptive, like the tail shares and for the same reasons.

## How to run it

Three invocations, because the steps need different machines and the middle one is the expensive one.
Preview any of them with `--dry-run`, which resolves the paths and reports what exists.

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Embedding.py \   # or the biology script
      --step plan     --condition FAB --total-time-seconds 2.0 --artifact-tag <TAG> [--draws-per-window 50]
      --step simulate --condition FAB --total-time-seconds 2.0 --artifact-tag <TAG> [--shard 0/8]
      --step analyze  --condition FAB --total-time-seconds 2.0 --artifact-tag <TAG>

- **`plan`** (CPU, seconds) selects the draws and fixes every parameter of every video in `plan.npz`:
  the posterior draws, the nuisance draws, the window identities, the checksum of the draws' source
  and of the estimator that produced them. A plan is never replaced; a second selection needs its own
  `--run-label`. The draws come from the Experiment product of the estimator when it retained its
  clouds, or from a retained draw collection (`--draws-source <file>`,
  `*_Window_Posterior_Draws*.npz`).
- **`simulate`** (CPU) renders the plan's videos, one file per window under `videos/`. It is
  resumable: a window file appears only when all its videos are written, and an existing one is never
  simulated again. Every file records the checksum of the plan it was simulated from; a file of another
  plan (other draws, or the same draws with other nuisance values) stops the step. `--shard i/n` takes the windows whose index is `i` modulo `n`, so `n` jobs share the
  work. Videos are simulated unseeded, as the training data are; the plan holds every parameter.
- **`analyze`** (GPU for the embedding) embeds the predictive videos and the experimental windows,
  ranks, and writes the report. It refuses an estimator other than the one the draws came from, a window file
  that was not simulated from this plan (by the plan's checksum), and an incomplete set of windows
  unless `--allow-partial` is given.

Arguments beyond the three above: `--kinds` and `--max-cells` (plan: which recordings),
`--pool` and `--seed` (plan: which draws; the seed fixes the selection and the nuisance draws),
`--nuisance-tag` (plan, biology: a tagged `Nuisance_DLI` artifact), `--experiment-span-seconds` and `--chunk-step-seconds` (cross-checks of the recording length and of the
window layout, both read from the source of the draws), `--reference-embeddings` (analyze: the stored EVAL embeddings; default: the
embedding-distance analysis' output folder for the same estimator), `--knn-k-cloud`, `--knn-k` and
`--alpha` (analyze).

**Cost.** The cost is the simulation: windows times draws videos, each one reaction-diffusion run and
one render. Twenty-five recordings of ten windows at 50 draws are 12,500 videos, the size of twelve
and a half generation tasks of the training data; `--max-cells` and `--draws-per-window` scale it.

## Outputs

Written under the workflow's `Posit/` subdirectory, in
`<alias>_<timing_label>[_<TAG>]_Posterior_Predictive_Embedding[_<LABEL>]/`:

- `plan.npz` — every video's parameters and the plan's manifest.
- `videos/Window_<GROUP>_Cell_<n>_Chunk_<w>.npz` — the predictive videos of one window (8 bit, as
  stored training videos) with their parameters.
- `report.md` — the readings by group, the excluded share of the posterior draws, the table by window
  (up to 80 windows), the pixel-level comparison by group, the provenance.
- `window_pixel_statistics.csv` — one row per window and pixel statistic: the window's value, the
  median and the 5-95 % range of its predictive videos, the share of predictive videos below it.
- `window_predictive_typicality.csv` — one row per window: the share of its posterior draws the pool
  excluded, tail share in its predictive cloud (raw and standardized), tail share in the EVAL cloud,
  the share of its predictive videos in the outskirts of the EVAL cloud, the reading.
- `figures/predictive_projection` — per recording group, the predictive videos (dots) and the
  experimental windows (diamonds) of every recording on two principal axes of the EVAL cloud, the EVAL
  videos in gray.
- `embeddings.npz` — the embeddings of the predictive videos and of the windows, with the readings, the
  pixel statistics, the excluded shares and the plan's checksum.

## Caveats

- **Nuisances at their training distribution.** Stated above; it is the main limit of the detector reading. A
  conditioned version would draw the reaction-diffusion parameters from the biology posterior of the
  same window, which needs both estimators and is not done here.
- **The embedding sees what informs the inferred parameters.** A window inside its predictive cloud is
  reproduced *as far as the network can tell*. A difference the embedding discards is invisible in
  the tail shares; the pixel-level table of the report and the posterior-predictive videos with their
  intensity histograms cover that side.
- **No calibrated rate.** The tail shares are not p-values (the window informed its own posterior), and
  windows of one recording are not independent. The count of windows outside their clouds has no
  expected value under a correct model; it is read window by window, as a list of warnings.
- **A fresh window, not a continued recording.** Each predictive video starts unbleached at the
  window-start composition the parameters describe, as the training videos do. A late window of a
  recording has a history the simulation does not replay.
- **The posterior is the estimator's.** Where the estimator returns the prior for a parameter, the
  predictive videos span the prior in that parameter, and the check says nothing about it.

## Reference

The tail share is a conformal p-value with the k-th-nearest-neighbor distance as its score: Vovk,
Gammerman, Shafer, *Algorithmic Learning in a Random World*, Springer, 2005. Posterior predictive
checks: Gelman, Meng, Stern, "Posterior predictive assessment of model fitness via realized
discrepancies," *Statistica Sinica* 6, 1996. Experimental recordings: MET single-particle-tracking
data, BioImage Archive accession S-BSST712.
