# Experimental-versus-synthetic embedding distance — usage and interpretation

Authoritative companion for **both** workflows' embedding-distance analyses:
`SRM_AND_SBI_MONOMER_DIMER_ALP_Embedding_Space_Distance.py` (biology) and
`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Embedding_Space_Distance.py` (detector). Both are thin shims over
one shared engine, `srm_and_sbi_monomer_dimer_alp.embedding_space_distance_runner`.

The analysis measures how far the experimental recordings sit from synthetic videos in the trained
network's embedding — the only representation the posterior ever sees. If the two occupy different
regions of it, every estimate produced from experimental recordings is an extrapolation: confidently
reported, but outside the region the network learned. It gives three readings: two for the sets as a
whole (a kernel distance and a classifier test) and one for every experimental window on its own. This
note explains what it measures, how to run it, and how to read the result, so it can be used and
understood without reading the code.

The companion analysis `SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Embedding.md` asks the next
question, window by window: whether videos simulated at the parameters inferred for a window look, to
the network, like that window. It reads the synthetic embeddings this analysis stores.

## The measurement is shared; the reading inverts

The numbers are computed identically for both workflows. What a difference *means* needs care in
both, and differs between them.

**A detected difference is not by itself a failure of the forward model.** The synthetic EVAL set
covers the whole training prior. A set of recordings can occupy a narrow part of it and still be
described by the simulator, window by window. The two-sample tests then detect a difference between
the two *populations*, correctly, although no window lies outside the synthetic cloud. Whether windows
lie outside the cloud is the question of the per-window typicality; whether the model reproduces a
window at the parameters inferred for it is the question of the posterior-predictive embedding check.

What a window **outside** the cloud points at depends on the workflow, because the two workflows infer
opposite halves of the model:

- **Detector** — the imaging parameters are the inference target, so the synthetic videos are
  supposed to reproduce the experimental ones' *appearance*. A window outside the cloud points at the
  imaging forward model or at the prior ranges of its parameters.
- **Biology** — the imaging is held **fixed** and the reaction-diffusion parameters are inferred, so
  a window outside the cloud is **not** primarily an imaging defect. It carries structure the
  synthetic prior never generated: biological variation outside the prior's reach, or an imaging
  mismatch the frozen imaging vector cannot express. Neither is fixed by training longer — the
  remedies are a wider prior or a recalibrated imaging vector.

Reading a biology finding as an imaging defect (or a detector finding as biological variation) would
send the work in the wrong direction, which is why the two shims exist under separate names
rather than one script with a flag.

This is a post-hoc, user-driven analysis, not one of the canonical pipeline stages. It lives in
`Script_Bank/Analysis`, is never wired into the stage dispatcher, and is complementary to the
Experiment stage. Because it embeds videos through the trained estimator, it is a GPU step. Its design
and justification are in `DETECTOR_WORKFLOW.md` (section "Quantitative experimental-versus-synthetic
distance").

## What it measures

The detector workflow marginalizes the biology as a nuisance, so the estimator's embedding is a
representation of the imaging parameters; a distance measured in that embedding is a statement about
the imaging, not the biology. The analysis embeds two sets of videos through the trained
`Complex3DCNN` — the experimental recordings and the held-out synthetic EVAL set — and reads the raw
embeddings three ways.

**Two-sample statistics (the sets as a whole).**

- **Maximum Mean Discrepancy (MMD)** — a kernel discrepancy (RBF kernel, median-heuristic bandwidth)
  with a permutation p-value. Zero means identical distributions; larger means a bigger difference.
- **Classifier Two-Sample Test (C2ST)** — a classifier is trained to separate the two embedding sets
  and scored on held-out windows by the **area under the ROC curve (AUC)**: the probability that it
  ranks a synthetic video above an experimental window. An AUC near 0.5 means the classifier cannot tell the two
  sets apart; approaching 1.0 means separable. The AUC is a rank statistic, so 0.5 is its
  no-information value whatever the sizes of the two sets; an accuracy would instead sit at the share
  of the larger set. The classifier weighs the two sets equally (class weights) and is a regularized
  logistic regression by default: a linear classifier, whose AUC measures *linear* separability. A
  nonlinear classifier (`--c2st-classifier rf` or `mlp`) can use structure the linear one misses, but
  a fitted one is not guaranteed to score higher, so the linear AUC is not a bound on it.

Both resample at the **recording (cell) level**. Each experimental recording is cut into model-length
windows, and windows from one recording share its acquisition and are statistically dependent.
Treating them as independent would inflate the effective sample size and make the significance
anticonservative. A recording is therefore never split between the training and the test side of a
cross-validation fold, and the null distributions exchange whole units. Synthetic videos are
independent draws with no unit structure, so they are grouped into **blocks of the recordings' window
counts**: exchanging the labels of blocks then keeps both the number of units and the number of windows
on each side. The MMD permutes the block labels; the C2ST repeats the whole cross-validated fit for
every relabeling and reports where the observed AUC falls in that null (its 95 % range and a one-sided
p-value). The classifier test reads at most `--c2st-reference-cap` synthetic videos (2000 by default,
whole blocks drawn at random), because every relabeling refits the classifier and a classifier that
weighs the sets equally gains little from a reference many times the experimental set. The MMD reads
all of them.

**One window per recording.** Matching the block sizes does not make a recording's correlated windows
exchangeable with a block of independent synthetic videos, so the p-values of the comparisons on all
windows are approximate. Every comparison is therefore repeated on one window of every recording: the
window whose start index is `--single-window-chunk` (default 0, the opening window), a position
declared before the run and not chosen by its result. Each unit is then a single sample, the
synthetic videos enter singly, and if the recordings are independent of each other the permutation
p-values are valid. These rows carry the test; the rows with all windows describe the same comparison
with more data. Every window stays in the descriptive readings (the per-window typicality, the
figures).

**Per-window typicality (every window on its own).** For each experimental window the analysis takes
the distance to its `k`-th nearest synthetic video (`--knn-k`, default 10) and compares it with the
same quantity for the synthetic videos among themselves: each synthetic video's distance to its `k`-th
nearest other synthetic video. The **tail share** of a window is the share of synthetic videos that
are at least as isolated as the window is. If the window were one more independent draw from the distribution
of the synthetic videos, a tail share at or below 0.05 would occur for at most 5 % of such windows
(exactly 5 % when no two scores are equal; equal scores count against flagging); a window far outside the cloud gets the smallest possible value,
1 / (number of synthetic videos + 1). The reading is given in two geometries: the raw embedding, whose
distances follow the coordinates with the largest spread, and the standardized embedding, in which
every coordinate is divided by its spread over the synthetic set. The standardized reading takes its
scale from the synthetic set alone, so the window is not treated like the synthetic videos and the
5 % statement does not carry over to it: it is a descriptive rank. This reading needs no classifier,
works with a handful of windows, and says *which* windows and recordings lie outside the cloud — which
a single number for the whole set cannot.

## Comparisons

The imaging is a property of the microscope, not the biological condition, so the **primary**
comparison pools all experimental recordings across conditions against the synthetic set. The by-condition comparisons are **diagnostics**: whether each
condition individually differs from the synthetic set, and whether the two conditions separate from each
other in the detector embedding — which they should not, since the biology is marginalized, so a
separation would indicate residual biological signal the embedding has not fully integrated out.

| comparison | role |
|---|---|
| experimental (MET-FAB + MET-INLB pooled) vs. synthetic | **primary** — all recordings against the synthetic set |
| experimental MET-FAB vs. synthetic | diagnostic — one condition against the synthetic set |
| experimental MET-INLB vs. synthetic | diagnostic — one condition against the synthetic set |
| experimental MET-FAB vs. MET-INLB | diagnostic — condition separation in the detector embedding |

The conditions are named `MET-FAB` (monomer control) and `MET-INLB` (dimer) on the command line, in
the figures, and in the tables. The tokens `ALP` and `BET` survive only in the on-disk recording
filenames and the stored schemas; the analysis maps each condition name to its filename token
internally (`MET-FAB` → `ALP`: Fab, the non-activating monomer control; `MET-INLB` → `BET`: InlB,
the activating dimer).

The pooled group is formed by **natural composition** by default: every window from both conditions,
in the proportions the dataset has, each recording kept as its own cell so the resampling is
unchanged. A **balanced** alternative (`--mix balanced`) instead draws an equal number of windows from
each condition, so neither dominates the aggregate; it downsamples the larger condition and is the
deliberate choice when equal weighting is wanted. The per-condition counts are always reported, so the
composition is transparent under either setting.

## How to run it

Preview first with `--dry-run`, which resolves the input and output paths and reports what would be
read or written without loading the estimator or computing.

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Embedding_Space_Distance.py \        # or ..._DETECTOR_... \
      --condition FAB --total-time-seconds 2.0 [--artifact-tag TAG] [--mix balanced] [--dry-run]

Arguments:

- `--condition` (required) — the condition of the run, the estimator's condition (`FAB` or `INLB`).
- `--artifact-tag` — the tag the estimator's training run was given (for example `C128V49A`); the
  estimator `<alias>_<timing_label>_<TAG>_Estimator.npz` is read and the outputs carry the same
  product label. Without it the untagged estimator is read.
- `--total-time-seconds` (required) — the model window / recording duration that sets the
  `timing_label` (for example `2.0` → `2S_50FPS`), used to locate the estimator and the EVAL set and
  to name the outputs.
- `--experiment-span-seconds` — the duration of the experimental recordings to read (files named
  `Experiment_<KIND>_Cell_<n>_<span>S_RAW.tif`, where `<KIND>` is the on-disk filename token `ALP` or
  `BET`); default `20`.
- `--kinds` — comma-separated experimental condition names (default `MET-FAB,MET-INLB`); each
  contributes a diagnostic row and all are pooled for the primary row.
- `--eval-tasks` — number of synthetic EVAL tasks to embed as the synthetic reference (default: all).
- `--chunk-step-seconds` — the sliding-window step used to dice the recordings; default the model
  window (non-overlapping chunks).
- `--mix` (`natural` default, or `balanced`) — how the pooled experimental group is composed (above).
- `--n-permutations` — MMD permutation count (default 1000).
- `--c2st-permutations` — relabelings of the units for the classifier test's null (default 200; `0`
  reports the AUC alone).
- `--c2st-reference-cap` — the most synthetic videos the classifier test reads (default 2000; `0` = all).
- `--c2st-classifier` — `logreg` (default), `rf` or `mlp`.
- `--single-window-chunk` — the window of every recording read by the one-window-per-recording rows, by
  the index of its start (default 0, the opening window). Declare it before the run. The HPC launcher
  runs the opening window: it has no variable for another position (pass one through `EXTRA` when needed).
- `--knn-k` — neighbor rank of the per-window typicality (default 10).
- `--alpha` — the level below which a p-value counts as a detected difference and a tail share as far
  (default 0.05).
- `--max-cells` — cap the cells per kind (0 = all).
- `--dry-run` — resolve the paths and report what would be read and written; load nothing, compute
  nothing.

## Reading the result

**The two-sample table.** For each comparison the report gives the MMD with its permutation p-value,
the C2ST AUC with the 95 % range it takes when the unit labels are exchanged and its p-value, and a
verdict. The MMD and the AUC are the quantities to read: the MMD against 0, the AUC against 0.5 and
against its null range. Every comparison has two rows: all windows, and one window per recording. The verdict
is **difference detected** when either p-value is below α and **no difference detected** otherwise. The
latter says that this sample did not show a difference, not that the two distributions are equal. The
pooled rows are the headline and the diagnostics qualify them. A detected difference is one between
two populations and not by itself a failure of the forward model (see the section on the reading
above).

What the p-values state, and what they do not. In the rows with one window per recording the units are
single samples; if the recordings are independent of each other the permutation p-values are valid,
and these rows carry the test. With few recordings they have little power, so "no difference detected"
is weak evidence there. In the rows with all windows the null hypothesis is that a recording's windows
are exchangeable with a block of independent synthetic videos. Windows of one recording are correlated and
a block of synthetic videos is not, so the null can be rejected by that correlation alone, with no
difference between the distributions of single windows. The p-values are therefore approximate guides,
most informative when they do NOT reject. With thousands of synthetic videos a small real difference is
always detectable, so a rejection by itself says little about its size: the AUC and the MMD say how
large the difference is. When the recordings hold different numbers of windows, the synthetic blocks
repeat those sizes and a relabeling keeps the window counts only on average. A classifier-test p-value
is `n/a` when no unit can be exchanged or no relabeling was requested; the verdict then rests on the MMD
and says so.

**The per-window tables.** By group: the number of windows that are far in each geometry, beside the
count that windows drawn like the synthetic videos would give on average (at most α times the number of
windows, in the raw geometry; the standardized count is descriptive), and the number of recordings with at least one far window. By recording: the smallest and the
median tail share. `window_typicality.csv` holds every window. The expected count is a bound on the *average* count under the
hypothesis that every window is exchangeable with the synthetic videos (for 40 windows at 5 %: two). It
is not a limit on the observed count, which varies around its average, and more so because windows of one
recording are correlated; the count is not compared with a binomial distribution. It is a hypothetical
reference, not a criterion for accepting the model.

**How the readings relate.** A classifier can separate two sets whose members are all individually
typical of the other (a consistent small shift), and a few far windows can leave the set statistics
almost unchanged. Read the per-window tables for *where* the experimental data lie relative to the
training distribution, and the two-sample statistics for *whether the sets differ as a whole*. Being
inside the synthetic cloud is necessary for trusting an estimate, not sufficient: the cloud spans the
whole training prior, and the embedding keeps what informs the inferred parameters, so it can be blind
to a difference that does not.

## Visualization

The figures are deterministic and tied to the statistics; stochastic embeddings (UMAP, t-SNE) are not
used, because their layouts are seed-dependent and need not track the computed distance. Every figure
shows all groups together — experimental MET-FAB, experimental MET-INLB, and synthetic — so overlap
and clustering read directly:

- **Embedding-distance distributions** — a four-panel grid. Columns are within-group distances
  (within MET-FAB, within MET-INLB, within synthetic) and cross-group distances (each condition to
  synthetic, and MET-FAB to MET-INLB); rows are the distance geometry — the raw embedding (the
  primary geometry, arbitrary scale) on top, the L2-normalized embedding (unit sphere, cosine
  distance bounded in [0, 2]) below. In either row, a cross-group curve sitting on the within-group
  curves is overlap; a shift to larger distances is separation. The raw row holds the distances the
  MMD is built from, so its picture and the headline number agree; the L2 row is a scale check on
  whether the raw picture is a magnitude artifact, with the pooled MMD in that geometry reported as a
  secondary diagnostic.
- **C2ST score distributions** — the classifier's out-of-fold predicted probabilities for each group
  on one axis; overlapping score distributions are the graphical form of an AUC near 0.5.
- **Window typicality** — the synthetic videos' nearest-neighbor distances among themselves as a
  histogram, with one tick per experimental window at its own distance to the synthetic set, colored
  by group, in the raw and in the standardized geometry. Ticks inside the histogram are windows as
  close to the synthetic cloud as its own members are to each other; ticks beyond its right edge are
  windows the network has seen nothing like.
- **Inter-group mean-distance matrix** — a compact MET-FAB/MET-INLB/synthetic square panel of mean
  embedding distances, the at-a-glance summary of which groups sit close and which apart.
- **PCA / classical-MDS panels (projection, PC1 density, Shepard)** — the raw embeddings projected
  onto their top two principal components (the optimal linear distance-preserving projection), in
  three panels: (a) the projection colored by group, each axis labeled with its explained-variance
  fraction — when one component dominates, the points fall on a single curved arc (the horseshoe
  signature of one-dimensional data), not separate blobs; (b) the distribution of each group's PC1
  coordinate, the honest one-dimensional reading of where each group sits along the single real axis
  and how far it spreads; (c) a Shepard diagram of projected 2-D distance against true
  high-dimensional distance (points on or below the identity line, with an annotated correlation)
  stating how faithful the projection is. Deterministic, unlike UMAP/t-SNE, and self-diagnosing:
  trust the projection to the degree the explained variance and correlation are high, with the
  distance-distribution and matrix figures as the exact high-dimensional summary. The accompanying
  variance-spectrum table quantifies the embedding's intrinsic dimensionality.
- **PC1 parameter tracking** — when the synthetic EVAL imaging parameters are available: (a) a bar
  chart of the absolute Spearman correlation between the dominant embedding axis (PC1) and each
  imaging parameter, sorted, so the parameter driving the single dominant direction stands out; (b)
  PC1 against that top parameter. This names the imaging parameter the embedding varies along when it
  collapses toward one dimension. Rank correlation, so it is unaffected by the log/linear storage of
  the parameters.

## Outputs

Written to the workflow's `Posit/` subdirectory under the data bank, in a
`<alias>_<timing_label>[_<TAG>]_Embedding_Space_Distance/` directory. `<alias>` is
`SRM_AND_SBI_MONOMER_DIMER_ALP` for biology and `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR` for detector, so the two
workflows' results never collide, and the tag keeps the results of two estimators apart:

- `report.md` — the per-comparison table (MMD, C2ST AUC with its null range, p-values, verdict), the
  per-window typicality by group and by recording, the pooled-group composition,
  the PCA variance spectrum (intrinsic dimensionality), the PC1-versus-imaging-parameter ranking with
  each parameter's meaning (when the EVAL imaging parameters are present), the pooled MMD in the L2 (cosine) geometry as a secondary
  diagnostic, and the run provenance (estimator checksum, the EVAL and experimental sources, the chunk
  count, the permutation count, α).
- `figures/` — the figures above (the PC1 parameter-tracking figure is included when the EVAL imaging
  parameters are present).
- `window_typicality.csv` — one row per experimental window: group, recording, window, its distance
  to the synthetic set, its percentile and tail share (raw), its tail share (standardized), and the
  reading in each geometry.
- `embeddings.npz` — the embeddings of this run: every experimental window with its group, recording
  and window index, and every synthetic EVAL video, with the checksum of the estimator's weights. The
  posterior-predictive embedding check reads the synthetic cloud from this file, and only for the same
  estimator.

## Caveats

- **These are comparisons on experimental recordings, not ground-truth recovery.** The recordings
  have no ground truth; the analysis measures distributional overlap in the embedding, not a
  demonstrated recovery of true parameters. Recovery is quantified only on held-out synthetic data in
  the workflow's Evaluation stage.
- **A detected difference is not a verdict on the model.** The synthetic set covers the whole training
  prior and the recordings may occupy a narrow part of it, so the two populations can differ while every
  window lies inside the synthetic cloud.
- **The measure diagnoses; it does not repair.** Windows outside the cloud are brought in only by changing the
  model (recalibrating the imaging, or widening the imaging prior) or the training (an
  embedding-alignment term), neither performed here (see `DETECTOR_WORKFLOW.md`, section "Quantitative
  experimental-versus-synthetic distance").
- **Few recordings weaken the set-level tests.** With few recordings there are few units to exchange
  and few cross-validation folds (at most one fold per recording), so the null ranges are wide; the
  MMD, the AUC and the per-window tail shares remain informative.
- **Typicality is relative to the synthetic set embedded.** The smallest tail share is one over the
  number of synthetic videos plus one, and a small EVAL subset (`--eval-tasks`) gives a coarse grid.
- **The interpretation is the detector question only.** A distance here concerns the imaging; the
  analogous distance under the biology workflow (imaging marginalized) is a
  biological question and a separate companion analysis.

## Reference

Maximum Mean Discrepancy: Gretton, Borgwardt, Rasch, Schölkopf, Smola, "A Kernel Two-Sample Test,"
*Journal of Machine Learning Research*, 2012. Classifier Two-Sample Test: Lopez-Paz and Oquab,
"Revisiting Classifier Two-Sample Tests," *ICLR*, 2017. Cell-grouped cross-validation controls the
within-recording correlation of the video windows. The per-window tail share is a conformal p-value
with the k-th-nearest-neighbor distance as its score: Vovk, Gammerman, Shafer, *Algorithmic Learning in
a Random World*, Springer, 2005. Experimental recordings: MET single-particle-tracking
data, BioImage Archive accession S-BSST712. The embedding, the imaging prior box, and the
detector-versus-biology framing are in `DETECTOR_WORKFLOW.md`.
