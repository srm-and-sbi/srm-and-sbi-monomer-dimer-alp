"""Nuisance_DLI: the calibrated-imaging nuisance — artifact, construction, and gate.

Design reference: DETECTOR_WORKFLOW.md, "Nuisance and artifact design".

The Detector calibrates the imaging model; the imaging block is then marginalized during
the production run by drawing from a `Nuisance_DLI`. Unlike the biology nuisance, a
`Nuisance_DLI` cannot be declared a priori — its content *is* the calibration result — so
it is built once, by a user-driven analysis step, and persisted as a self-contained
artifact that generation samples standalone.

Construction. The detector estimator is a REQUIRED prerequisite. The analysis step reads
all experimental recordings, draws the posterior per model-window chunk on the fly, and
pools the draws across every chunk of every recording into the `posterior_sample_pool` — a
mixture that represents how the calibrated imaging varies across the real acquisitions. One
user choice, `posterior_sample_pool_choice`, turns that pool into the samplable artifact:

  - ``raw``               — resample the pool per whole vector (rows). Preserves the joint
                            correlation structure exactly; the faithful default.
  - ``map_estimate_pool`` — resample, per whole vector, the pooled per-chunk MAP estimates
                            (point estimates rather than full posterior draws).
  - ``gaussian``          — a full-covariance multivariate normal fit to the pool. Keeps the
                            linear correlations (via the covariance); loses multimodality
                            and curvature.
  - ``box``               — a per-parameter uniform over quantiles of the pool, clamped to
                            the imaging prior box. Independent per dimension: no correlation.
  - ``box_user``          — a per-parameter uniform over user-set ranges, clamped to the
                            prior box. No correlation.
  - ``selection_user``    — ONE explicitly chosen, fixed imaging vector assembled from documented
                            sources (six named physical values, each with its source and
                            limitation), returned unchanged to every simulation. Not a posterior
                            sample, not an automatically selected SGM; needs no estimator, no
                            Experiment product, no GPU and no pool. It carries no calibrated joint
                            uncertainty. A value outside the detector prior box is refused unless
                            the spec acknowledges it explicitly with a recorded justification; the
                            acknowledgement flags extrapolation beyond the tested imaging domain and
                            certifies nothing; nothing is clipped.
  - ``empirical_representatives`` — a WEIGHTED empirical distribution over complete representative
                            vectors selected from the Experiment stage's candidate archive
                            (``aggregation_candidates.npz``: the per_cell / per_window_position /
                            global / per_cell_window SGMs, or the marginal-median vectors, of named
                            window sets). The user-authored spec names K components (source
                            Experiment, condition, window set, grouping, statistic) with explicit
                            weights alpha_k; every member of component k receives probability
                            alpha_k / n_k, exact duplicates are consolidated by adding their
                            probabilities (memberships kept), and the artifact stores the vectors
                            with their probabilities and provenance. CPU only; no neural sampling.
                            SEL is a comparison, never an automatically mixed component; raw
                            posterior draws and an SGM of SGMs are not constructions of this route.

Selecting an artifact. The canonical artifact carries the plain timing label. A ``nuisance_tag``
(a SCREAMING_SNAKE token, the grammar of the estimator's artifact tag) names a further artifact
beside it -- a reference vector, a sensitivity variant -- and every consumer (the biology
generation, the posterior-predictive render, the horizon audit, the SGM analysis) selects it by
that tag and records the selected artifact's identity in its outputs, so a reference run and a
variant cannot be confused. The tag names the nuisance artifact, never the estimator that informed
it: for ``selection_user`` the estimators behind the values belong to the spec's provenance.

`pool_mode` (``bounded`` default / ``unrestricted``) is identical to the Evaluation and
Experiment convention: bounded rejection-samples the pool within the imaging prior box;
unrestricted takes the raw-flow draws, whose mass may lie outside it. The empirical and
gaussian forms are faithful to the calibration and are NOT clipped — their support is
governed by `pool_mode`. Only `box`/`box_user` are constrained to the prior box, clamped
(never silently) at build.

Distinction from the RDS nuisance. The reaction-diffusion biology marginalized during
detector calibration is never drawn or built anywhere: the detector re-images the condition's
RDS trajectory tier, whose `Theta_Set` (the biology's own labels: eleven under MET-FAB, twelve
under MET-INLB) is the record of that nuisance. The two share the marginalization *role* but
not the artifact:
the `Nuisance_DLI` is a persisted calibration result, so it — and only it — is the object
defined here.

Enforcement. Downstream consumers call `require_nuisance_dli(...)`, which loads the built
artifact and fails loud (naming the analysis to run) if it is absent — the `Nuisance_DLI`
is a user decision, never an automatic fabrication.
"""
import json
import logging
import re
from pathlib import Path

import numpy as np

from . import artifact_schema as schema

try:
    import tomllib                       # stdlib on Python 3.11+ (this env is 3.13)
except ModuleNotFoundError:              # pragma: no cover
    import tomli as tomllib

logger = logging.getLogger(__name__)

SPEC_SUFFIX = "_Nuisance_DLI_Spec.toml"      # the user-authored spec (analysis-emitted)
ARTIFACT_SUFFIX = "_Nuisance_DLI.npz"        # the built, samplable Nuisance_DLI artifact
ARTIFACT_FORMAT_VERSION = 2                  # v2: posterior_sample_pool_choice scheme
POOL_FORMAT_VERSION = 1                      # v1: pool cache carries per-row kind_index/cell/chunk labels

POOL_CHOICES = ("raw", "map_estimate_pool", "gaussian", "box", "box_user", "sgm_percentiles",
                "empirical_representatives",
                "selection_user")
POOL_MODES = ("bounded", "unrestricted")
DEFAULT_BOX_QUANTILES = (0.05, 0.95)
_EMPIRICAL = ("raw", "map_estimate_pool", "sgm_percentiles",
              "selection_user", "empirical_representatives")   # stored as a sample matrix, resampled per-vector (one row: fixed)
_WEIGHTED = ("empirical_representatives",)   # a sample matrix with per-row sampling probabilities (stored, never implied)
REPRESENTATIVE_STATISTICS = ("sgm", "median")
WEIGHT_NORMALIZATION_TOLERANCE = 1e-6
# Member-weighting policies of an empirical_representatives component. ``uniform`` (the backward-compatible default:
# every member of component k receives alpha_k / n_k; available for every grouping, the single dataset representative
# included) and ``posterior_rank_median`` (consensus weighting: the member's probability is proportional to the median
# cross-posterior distance rank Experiment exported for its SGM). The rank policy names its COMPARISON SCOPE explicitly
# (``rank_scope``), and the scope must be the one the archive's scores of that grouping were computed under:
#   ``construction``: the scores compare every representative with ALL the other representatives of its grouping,
#       condition and window set (per_cell: the other cells; per_window_position: the other positions; per_cell_window:
#       every other cell-window group, same-cell windows included); p_i = alpha_k * s_i / sum_j s_j over the component's
#       complete population. No cell receives a reserved mass under per_cell_window: a cell's total is the sum of its
#       members' probabilities and may be small or exactly zero.
#   ``within_cell`` (per_cell_window only; the hierarchical policy of archive formats 2 and 3): the scores compare each
#       cell-window SGM with the other selected windows of its own cell; equal total cell mass alpha_k / |C_k|, then
#       s_cw / sum_u s_cu WITHIN each cell.
# A zero score gives exactly zero probability (kept, never floored, clipped or dropped); an all-zero normalizer is
# refused (the whole component under ``construction``; any included cell under ``within_cell``). An unknown policy, a
# missing or mismatched scope, a missing or unavailable score, a non-SGM statistic, a grouping without scores in the
# archive's scope or a format-1 archive is an explicit error, never a fallback to uniform.
MEMBER_WEIGHTINGS = ("uniform", "posterior_rank_median")
RANK_WEIGHTED_GROUPINGS = ("per_cell", "per_window_position", "per_cell_window")
RANK_SCOPES = ("construction", "within_cell")     # experiment_aggregation.COMPARISON_SCOPES
RANK_SPACE_ALL = "all"      # the archive's primary rank coordinate space (every parameter coordinate); experiment_aggregation.RANK_SPACE_ALL
# Parameters whose physical domain is a probability (0, 1): the per-dye bleaching probability. An explicit
# selection or override of such a parameter, and every final vector of an explicitly selected artifact
# (selection_user, empirical_representatives), must respect it; a physically valid value outside the TRAINING
# prior (an acknowledged extrapolation) is a different case and stays available.
PROBABILITY_KEYS = ("prob_photo_bleach",)


def physical_domain_error(key, value, entry=None) -> str:
    """The reason a physical ``value`` of parameter ``key`` is outside its physical domain, or ``""`` when it
    is valid: finite; positive for a log-ranged row (``entry`` with ``PRIOR_RANGE``/``LOG_FLAG``); strictly
    below one for a probability parameter. Shared by the selection_user, the override and the final-vector checks."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{key} = {value!r} is not a number"
    if not np.isfinite(v):
        return f"{key} = {v!r} must be finite"
    log_row = entry is not None and (entry.get("PRIOR_RANGE") is not None or bool(entry.get("LOG_FLAG")))
    if (log_row or key in PROBABILITY_KEYS) and v <= 0:
        return f"{key} = {v} must be positive" + (" (a log10-ranged row)" if log_row else "")
    if key in PROBABILITY_KEYS and v >= 1.0:
        return f"{key} = {v} must be a probability below 1"
    return ""
_BOX = ("box", "box_user")                   # stored as low/high, drawn as a per-param uniform

# The sgm_percentiles selection (whole vectors at signed distance-to-SGM percentiles; p50 = the SGM).
SGM_DEFAULT_PERCENTILES = [50.0]             # one percentile 50 -> the SGM alone (a frozen vector)
SGM_SELECTION_SOURCES = ("experiment", "window-sgm")
SGM_CONDITIONS = ("FAB", "INLB")

# Which cached pool each choice draws on. raw/gaussian/box share the posterior-sample pool;
# map_estimate_pool uses the MAP pool; box_user and sgm_percentiles need none (sgm_percentiles REUSES
# already-computed data -- the Experiment product's map_estimate (window MAPs), or the labeled
# posterior pool's per-window SGMs -- with no GPU).
POOL_KINDS = {"raw": "PosteriorSample", "gaussian": "PosteriorSample", "box": "PosteriorSample",
              "map_estimate_pool": "MapEstimate", "box_user": None, "sgm_percentiles": None,
              "selection_user": None, "empirical_representatives": None}

_ANALYSIS = "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py"


def nuisance_label(timing_label, nuisance_tag=None):
    """The label a Nuisance_DLI artifact and its spec are named under: the timing label,
    optionally followed by the nuisance tag (``2S_50FPS`` -> ``2S_50FPS_REF``).

    The tag is a SCREAMING_SNAKE token (``[A-Z0-9]+``, the grammar of ``Paths.product_label``),
    and it names the NUISANCE artifact, never an estimator: a reference vector or a sensitivity
    variant lives beside the canonical artifact instead of overwriting it, and every consumer
    selects an artifact by this tag and records it. No tag returns the timing label unchanged.
    """
    if nuisance_tag is None or nuisance_tag == "":
        return timing_label
    if not re.fullmatch(r"[A-Z0-9]+", str(nuisance_tag)):
        raise ValueError(f"nuisance_tag {nuisance_tag!r} must be a SCREAMING_SNAKE token without "
                         f"underscores ([A-Z0-9]+), e.g. 'REF' or 'BLEACHLOW'.")
    return f"{timing_label}_{nuisance_tag}"


def spec_path(posit_dir, project_alias, timing_label, nuisance_tag=None):
    """Path of the user-authored Nuisance_DLI spec (in the Posit subdir)."""
    return Path(posit_dir) / f"{project_alias}_{nuisance_label(timing_label, nuisance_tag)}{SPEC_SUFFIX}"


def artifact_path(posit_dir, project_alias, timing_label, nuisance_tag=None):
    """Path of the built, samplable Nuisance_DLI artifact."""
    return Path(posit_dir) / f"{project_alias}_{nuisance_label(timing_label, nuisance_tag)}{ARTIFACT_SUFFIX}"


# =============================================================================
# The artifact (numpy-only: generation samples/loads it without torch)
# =============================================================================
class NuisanceDLI:
    """Samplable, self-describing calibrated-imaging nuisance (see module docstring).

    One of three underlying representations, selected by ``posterior_sample_pool_choice``:
    ``box``/``box_user`` draw a per-parameter uniform over ``[low, high]`` (independent per
    dimension — no cross-parameter correlation); ``raw``/``map_estimate_pool`` resample a
    stored matrix **per whole vector** (rows), preserving the joint correlation structure
    exactly; ``gaussian`` draws ``N(mean, cov)`` (a full covariance keeps the *linear*
    correlations, but not multimodality or curvature). ``prior_low``/``prior_high`` (the
    imaging prior box) are recorded for provenance; only ``box``/``box_user`` are clamped to
    it. Sampling is numpy-only, so generation needs neither torch nor the estimator.
    """

    def __init__(self, parameter_keys, posterior_sample_pool_choice, pool_mode,
                 prior_low, prior_high, *, low=None, high=None, samples=None,
                 mean=None, cov=None, space="log10", selection=None, probabilities=None):
        self.parameter_keys = list(parameter_keys)
        self.posterior_sample_pool_choice = posterior_sample_pool_choice
        self.pool_mode = pool_mode
        # selection_user only: the values, their sources and limitations, and any acknowledged
        # outside-prior coordinates -- the decision record travels with the artifact.
        self.selection = None if selection is None else dict(selection)
        self.prior_low = np.asarray(prior_low, dtype=float)
        self.prior_high = np.asarray(prior_high, dtype=float)
        self.space = space
        self.low = None if low is None else np.asarray(low, dtype=float)
        self.high = None if high is None else np.asarray(high, dtype=float)
        self.samples = None if samples is None else np.asarray(samples, dtype=float)
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.cov = None if cov is None else np.asarray(cov, dtype=float)
        # empirical_representatives only: one sampling probability per stored row (normalized, stored).
        self.probabilities = None if probabilities is None else np.asarray(probabilities, dtype=float)

        d = len(self.parameter_keys)
        if self.posterior_sample_pool_choice not in POOL_CHOICES:
            raise ValueError(f"posterior_sample_pool_choice "
                             f"{self.posterior_sample_pool_choice!r} must be one of {POOL_CHOICES}.")
        if self.pool_mode not in POOL_MODES:
            raise ValueError(f"pool_mode {self.pool_mode!r} must be one of {POOL_MODES}.")
        if self.prior_low.shape != (d,) or self.prior_high.shape != (d,):
            raise ValueError(f"prior_low/prior_high must have shape ({d},) to match "
                             f"parameter_keys; got {self.prior_low.shape}, {self.prior_high.shape}.")
        choice = self.posterior_sample_pool_choice
        if choice in _BOX:
            if (self.low is None or self.high is None
                    or self.low.shape != (d,) or self.high.shape != (d,)):
                raise ValueError(f"choice {choice!r} needs low/high of shape ({d},).")
        elif choice in _EMPIRICAL:
            if (self.samples is None or self.samples.ndim != 2
                    or self.samples.shape[1] != d):
                raise ValueError(f"choice {choice!r} needs a (n, {d}) sample matrix; got "
                                 f"{None if self.samples is None else self.samples.shape}.")
            if choice in _WEIGHTED:
                pr = self.probabilities
                if pr is None or pr.shape != (self.samples.shape[0],):
                    raise ValueError(f"choice {choice!r} needs one sampling probability per stored row "
                                     f"({self.samples.shape[0]}); got {None if pr is None else pr.shape}.")
                if not np.all(np.isfinite(pr)) or np.any(pr < 0) or pr.sum() <= 0:
                    raise ValueError(f"choice {choice!r}: probabilities must be finite, non-negative with positive total mass.")
                if abs(pr.sum() - 1.0) > WEIGHT_NORMALIZATION_TOLERANCE:
                    raise ValueError(f"choice {choice!r}: probabilities sum to {pr.sum():.9f}, not 1 within "
                                     f"{WEIGHT_NORMALIZATION_TOLERANCE}; a weighted artifact stores normalized probabilities.")
                # The accepted roundoff residual is removed here, so every accepted vector is sampled and
                # round-tripped identically: the stored probabilities are the given ones divided by their sum
                # (a numerical convention; the scientific component weights live in the selection record).
                self.probability_residual = float(pr.sum() - 1.0)
                if abs(self.probability_residual) > 1e-12:          # an already normalized vector is kept bit-identical
                    self.probabilities = pr / pr.sum()
                if not self.selection or self.selection.get("kind") != "representatives":
                    raise ValueError(f"choice {choice!r} needs its selection record (components, weights, sources); "
                                     "build it through load_spec + build_nuisance_dli.")
            elif self.probabilities is not None:
                raise ValueError(f"choice {choice!r} carries no per-row probabilities; only {_WEIGHTED} do.")
            if choice == "selection_user":
                if self.samples.shape[0] != 1:
                    raise ValueError("choice 'selection_user' holds exactly one fixed vector; got "
                                     f"{self.samples.shape[0]} rows.")
                if not self.selection or self.selection.get("kind") != "user":
                    raise ValueError("choice 'selection_user' needs its selection record (values, "
                                     "sources, limitations); build it through load_spec + "
                                     "build_nuisance_dli.")
        else:  # gaussian
            if (self.mean is None or self.cov is None
                    or self.mean.shape != (d,) or self.cov.shape != (d, d)):
                raise ValueError(f"choice 'gaussian' needs mean ({d},) and cov ({d}, {d}).")
        # Final-vector physical domain of the explicitly selected constructions (never a clipping; the
        # calibration pools of raw/gaussian/box are diagnostic clouds and are not checked here).
        if choice in ("selection_user",) + _WEIGHTED and self.space == "log10":
            for j, k in enumerate(self.parameter_keys):
                if k in PROBABILITY_KEYS:
                    bad = np.flatnonzero(self.samples[:, j] >= 0.0)      # log10(p) >= 0  <=>  p >= 1
                    if bad.size:
                        raise ValueError(f"choice {choice!r}: {k} is a probability; {bad.size} stored vector(s) carry "
                                         f"{k} >= 1 (rows {bad[:5].tolist()}).")

    # -- builders ----------------------------------------------------------
    @classmethod
    def from_box(cls, parameter_keys, low, high, *, choice="box", pool_mode="bounded",
                 prior_low=None, prior_high=None, space="log10"):
        """A per-parameter uniform over ``[low, high]`` (``box`` from pool quantiles, or
        ``box_user`` from user ranges); ``low``/``high`` are already clamped to the prior box."""
        return cls(parameter_keys, choice, pool_mode, prior_low, prior_high,
                   low=low, high=high, space=space)

    @classmethod
    def from_samples(cls, parameter_keys, samples, *, choice="raw", pool_mode="bounded",
                     prior_low=None, prior_high=None, space="log10", selection=None):
        """A stored sample matrix (``raw`` posterior draws or ``map_estimate_pool`` MAP
        vectors), resampled per whole vector at draw time to preserve correlations."""
        return cls(parameter_keys, choice, pool_mode, prior_low, prior_high,
                   samples=samples, space=space, selection=selection)

    @classmethod
    def from_selection(cls, parameter_keys, vector_log10, selection, *, prior_low, prior_high,
                       pool_mode="bounded"):
        """ONE fixed, user-selected vector (log10) with its decision record; every draw returns it."""
        vec = np.asarray(vector_log10, dtype=float).reshape(1, -1)
        return cls(parameter_keys, "selection_user", pool_mode, prior_low, prior_high,
                   samples=vec, selection=selection)

    @classmethod
    def from_representatives(cls, parameter_keys, vectors_log10, probabilities, selection, *, prior_low, prior_high,
                             pool_mode="bounded"):
        """A weighted empirical distribution over complete representative vectors (log10) with their recorded
        sampling probabilities and the selection record (components, weights, sources, consolidation, overrides)."""
        return cls(parameter_keys, "empirical_representatives", pool_mode, prior_low, prior_high,
                   samples=np.asarray(vectors_log10, dtype=float), probabilities=probabilities, selection=selection)

    @classmethod
    def from_gaussian(cls, parameter_keys, mean, cov, *, pool_mode="bounded",
                      prior_low=None, prior_high=None, space="log10"):
        """A full-covariance multivariate normal fit to the posterior_sample_pool."""
        return cls(parameter_keys, "gaussian", pool_mode, prior_low, prior_high,
                   mean=mean, cov=cov, space=space)

    # -- sampling (numpy only; no clipping — see module docstring) ---------
    def sample(self, n, seed=None):
        """Draw ``n`` nuisance vectors, shape ``(n, D)`` in the declared ``space`` (log10). ``seed`` (optional)
        fixes the generator for a controlled check; production draws unseeded, as before.

        ``box``/``box_user`` draw each parameter independently within ``[low, high]``;
        ``raw``/``map_estimate_pool`` resample **whole rows** of the stored matrix (so the
        joint correlations are preserved — never per-column); ``gaussian`` draws from
        ``N(mean, cov)``. No clipping: the empirical and gaussian forms are faithful to the
        calibration, and ``box``/``box_user`` are bounded by construction.
        """
        n = int(n)
        rng = np.random.default_rng(seed)
        choice = self.posterior_sample_pool_choice
        if choice in _BOX:
            return rng.uniform(self.low, self.high, size=(n, self.low.size))
        if choice in _WEIGHTED:
            pr = self.probabilities / self.probabilities.sum()        # stored normalized; the sampler's own tolerance is tighter
            idx = rng.choice(self.samples.shape[0], size=n, p=pr)
            return self.samples[idx]                          # whole-vector resample at the recorded probabilities
        if choice in _EMPIRICAL:
            idx = rng.integers(0, self.samples.shape[0], size=n)
            return self.samples[idx]                          # whole-vector resample (uniform)
        return rng.multivariate_normal(self.mean, self.cov, size=n)   # gaussian

    # -- persistence -------------------------------------------------------
    def manifest(self):
        return {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "parameter_keys": self.parameter_keys,
            "posterior_sample_pool_choice": self.posterior_sample_pool_choice,
            "pool_mode": self.pool_mode,
            "space": self.space,
            "n_samples": (int(self.samples.shape[0]) if self.posterior_sample_pool_choice
                          in _EMPIRICAL else None),
            "selection": self.selection,
            "weighted": self.posterior_sample_pool_choice in _WEIGHTED,
            "n_components": (len(self.selection.get("components", [])) if self.selection
                             and self.posterior_sample_pool_choice in _WEIGHTED else None),
            "probability_normalization": (f"the stored probabilities are the given ones divided by their sum (accepted total within "
                                          f"{WEIGHT_NORMALIZATION_TOLERANCE} of one; a roundoff convention, not a weighting); residual "
                                          f"removed {getattr(self, 'probability_residual', 0.0):.3e}"
                                          if self.posterior_sample_pool_choice in _WEIGHTED else None),
        }

    @property
    def fixed_vector_log10(self):
        """The one vector a ``selection_user`` artifact returns (log10), else ``None``."""
        if self.posterior_sample_pool_choice != "selection_user":
            return None
        return np.asarray(self.samples[0], dtype=float)

    def identity(self, path=None):
        """A compact, JSON-ready identity for consumers to record: the artifact file, the choice,
        the pool mode, the sample count and (for selection_user) the selection record."""
        m = self.manifest()
        out = {"artifact": None if path is None else str(Path(path).name),
               "posterior_sample_pool_choice": m["posterior_sample_pool_choice"],
               "pool_mode": m["pool_mode"], "n_samples": m["n_samples"],
               "fixed": self.posterior_sample_pool_choice == "selection_user"}
        if self.selection is not None:
            out["selection"] = self.selection
        if self.posterior_sample_pool_choice in _WEIGHTED:
            out["weighted"] = True
            out["n_components"] = m["n_components"]
        return out

    def flush(self, path):
        """Write the self-contained artifact to ``path`` as a compressed ``.npz`` (the
        payload arrays for its choice, the prior box, and the manifest)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"prior_low": self.prior_low, "prior_high": self.prior_high,
                  "manifest": np.asarray(json.dumps(self.manifest()))}
        choice = self.posterior_sample_pool_choice
        if choice in _BOX:
            arrays["low"] = self.low
            arrays["high"] = self.high
        elif choice in _EMPIRICAL:
            arrays["samples"] = self.samples
            if choice in _WEIGHTED:
                arrays["probabilities"] = self.probabilities
        else:                                                 # gaussian
            arrays["mean"] = self.mean
            arrays["cov"] = self.cov
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def load(cls, path):
        """Reload a flushed artifact for sampling (numpy only)."""
        with np.load(str(path), allow_pickle=False) as data:
            m = json.loads(str(data["manifest"]))
            payload = {k: data[k] for k in ("low", "high", "samples", "mean", "cov", "probabilities")
                       if k in data.files}
            return cls(m["parameter_keys"], m["posterior_sample_pool_choice"],
                       m["pool_mode"], data["prior_low"], data["prior_high"],
                       space=m.get("space", "log10"), selection=m.get("selection"), **payload)


# =============================================================================
# Pool construction (analysis --build; needs the estimator + GPU)
# =============================================================================
def build_posterior_sample_pool(posterior, chunks, device, vista_device, *,
                                 n_per_chunk, theta_prex_batch_size, pool_mode,
                                 chunk_labels=None):
    """Draw ``n_per_chunk`` posterior samples per chunk and pool them, shape ``(N, D)``.

    Uses the Evaluation/Experiment candidate sampler (`collect_theta_prex`) conditioned on
    each chunk in turn, so ``bounded``/``unrestricted`` behave exactly as in those stages,
    and concatenates the draws over every chunk of every recording. The result is a mixture
    over the real acquisitions — the empirical calibrated-imaging distribution. Each draw is
    a whole parameter vector, so the joint correlation structure is retained in the pool.

    Returns ``(pool, labels)``. ``labels`` is ``None`` unless ``chunk_labels`` (one
    ``(kind_index, cell, chunk)`` tuple per chunk, aligned to ``chunks``) is given, in which
    case it is the per-row ``{"kind_index", "cell", "chunk"}`` dict expanded to each chunk's
    ACTUAL draw count — so it stays row-aligned even if bounded rejection yields a variable
    number of accepted draws per chunk."""
    import torch
    from .evaluation import collect_theta_prex
    from .inference_support import normalize_video

    pieces, ki, ce, ch = [], [], [], []
    for i, chunk in enumerate(chunks):
        omega = torch.tensor(normalize_video(chunk), dtype=torch.float32, device=device)
        cond = omega.unsqueeze(0)
        posterior.set_default_x(cond)
        flow = posterior.posterior_estimator
        drawn = collect_theta_prex(posterior, flow, vista_device, cond,
                                   n_per_chunk, theta_prex_batch_size, pool_mode)
        arr = drawn.detach().cpu().numpy()
        pieces.append(arr)
        if chunk_labels is not None:
            k, c, h = chunk_labels[i]
            ki.append(np.full(arr.shape[0], k, dtype=np.int64))
            ce.append(np.full(arr.shape[0], c, dtype=np.int64))
            ch.append(np.full(arr.shape[0], h, dtype=np.int64))
    if not pieces:
        raise ValueError("no experimental chunks were provided; the posterior_sample_pool "
                         "is empty (check the experimental recordings and windowing).")
    pool = np.concatenate(pieces, axis=0)
    labels = (None if chunk_labels is None
              else {"kind_index": np.concatenate(ki), "cell": np.concatenate(ce),
                    "chunk": np.concatenate(ch)})
    return pool, labels


def build_map_estimate_pool(posterior, chunks, device, vista_device, eval_cfg, *,
                            pool_mode, log_fn=None, chunk_labels=None):
    """MAP-estimate each chunk (seed-then-optimize) and pool the per-chunk MAP vectors,
    shape ``(n_chunks, D)``. Uses `evaluation.map_estimate` with the shared Evaluation
    hyperparameters, so the point estimates match the Detector Experiment stage.

    Returns ``(pool, labels)`` -- one MAP vector per chunk; ``labels`` (when ``chunk_labels``,
    one ``(kind_index, cell, chunk)`` per chunk, is given) is the per-window
    ``{"kind_index", "cell", "chunk"}`` dict, row-aligned to the ``(n_chunks, D)`` pool."""
    from .evaluation import map_estimate

    lr = eval_cfg.learning_rate
    tolerance = eval_cfg.tolerance
    pieces = []
    for chunk in chunks:
        _score, theta_log = map_estimate(
            posterior, chunk, device, vista_device,
            eval_cfg.theta_prex_size, eval_cfg.theta_prex_batch_size,
            eval_cfg.score_prex_batch_size, eval_cfg.elite_prex_size,
            eval_cfg.numb_steps, eval_cfg.optimizer_patience,
            eval_cfg.scheduler_patience, eval_cfg.show_progress_steps,
            eval_cfg.learning_rate_minimum, eval_cfg.learning_rate_factor,
            lr, tolerance, pool_mode=pool_mode, log_fn=log_fn,
        )
        pieces.append(np.asarray(theta_log, dtype=float))
    if not pieces:
        raise ValueError("no experimental chunks were provided; the map_estimate_pool is "
                         "empty (check the experimental recordings and windowing).")
    pool = np.stack(pieces, axis=0)
    labels = (None if chunk_labels is None
              else {"kind_index": np.asarray([lbl[0] for lbl in chunk_labels], dtype=np.int64),
                    "cell": np.asarray([lbl[1] for lbl in chunk_labels], dtype=np.int64),
                    "chunk": np.asarray([lbl[2] for lbl in chunk_labels], dtype=np.int64)})
    return pool, labels


# =============================================================================
# Fit helpers (pool -> a representation)
# =============================================================================
def fit_gaussian(pool):
    """Full-covariance Gaussian fit: ``(mean (D,), cov (D, D))`` from the pool rows.

    ``cov`` is the joint sample covariance over whole vectors, so its off-diagonals carry
    the linear cross-parameter correlations (e.g. the ``kappa_g``/``kappa_c`` degeneracy).
    A tiny diagonal jitter keeps it positive-definite for sampling.
    """
    pool = np.asarray(pool, dtype=float)
    mean = pool.mean(axis=0)
    cov = np.atleast_2d(np.cov(pool, rowvar=False))
    cov = cov + 1e-9 * np.eye(cov.shape[0])
    return mean, cov


def quantile_box(pool, quantiles, prior_low, prior_high):
    """Per-parameter ``[low, high]`` from pool quantiles, clamped to the prior box.

    Returns clamped ``(low, high)`` and logs — never silently — how many parameter ranges
    the prior-box clamp reduced.
    """
    pool = np.asarray(pool, dtype=float)
    ql, qh = float(quantiles[0]), float(quantiles[1])
    if not 0.0 <= ql < qh <= 1.0:
        raise ValueError(f"box_quantiles must satisfy 0 <= low < high <= 1; got {quantiles}.")
    low = np.quantile(pool, ql, axis=0)
    high = np.quantile(pool, qh, axis=0)
    pl = np.asarray(prior_low, dtype=float)
    ph = np.asarray(prior_high, dtype=float)
    clamped_low = np.maximum(low, pl)
    clamped_high = np.minimum(high, ph)
    n_clamped = int(np.sum((clamped_low != low) | (clamped_high != high)))
    if n_clamped:
        logger.warning("Nuisance_DLI[box]: clamped %d/%d parameter range(s) to the imaging "
                       "prior box.", n_clamped, int(low.size))
    return clamped_low, clamped_high


# =============================================================================
# Sample Geometric Median + signed-percentile selection (shared: analysis tool + build)
# =============================================================================
_SGM_EXACT_MEDOID_CAPACITY = 20000   # above this, Weiszfeld's iteration + snap, not the exact medoid
_SGM_WEISZFELD_ITERS = 2000


def sample_geometric_median(vecs_abs, range_abs):
    """``(index, method)`` of the collection's SGM member, in prior-range-normalized absolute space:
    the exact medoid up to the kernel's capacity, else the member nearest the Weiszfeld geometric
    median (an approximation); ``method`` says which. An actual member, so its coordinates
    co-occurred. (Ramirez Sierra & Sokolowski 2025; see DETECTOR_WORKFLOW.md.)

    Delegates to the workflow-agnostic kernel so the imaging nuisance construction, the detector
    analysis, and the biology Experiment analysis all share ONE implementation; a divergence between
    them would make two reports' "median vector" mean different things under the same name.
    """
    from .sample_geometric_median import sample_geometric_median as _sgm
    return _sgm(vecs_abs, range_abs)


def _per_window_sgm_labeled(flat_log, labels, range_abs):
    """Per-window Sample Geometric Median from a labeled pool: the exact medoid of each window's
    posterior draws in PHYSICAL coordinates divided by the physical prior range -- not the stage
    products' ``posterior_sgm``, which divides estimator coordinates by the log10 prior widths.
    Groups the window-major flat pool into windows by its (kind_index, cell, chunk) labels.
    Returns (vectors_log (n_win, D), kind_index (n_win,), kinds list)."""
    keys = np.stack([np.asarray(labels["kind_index"]), np.asarray(labels["cell"]),
                     np.asarray(labels["chunk"])], axis=1)
    bounds = np.concatenate([[0], np.where(np.any(keys[1:] != keys[:-1], axis=1))[0] + 1,
                             [keys.shape[0]]])
    n_win = bounds.shape[0] - 1
    out = np.empty((n_win, flat_log.shape[1]))
    ki = np.empty(n_win, np.int64)
    for w in range(n_win):
        rows = flat_log[bounds[w]:bounds[w + 1]]
        idx, _ = sample_geometric_median(10.0 ** rows, range_abs)
        out[w] = rows[idx]
        ki[w] = keys[bounds[w]][0]
    return out, ki, list(labels["kinds"])


def select_signed_percentile_vectors(vecs_log, percentiles, prior_low, prior_high, *,
                                     brightness_index=2):
    """Select whole vectors at the given percentiles along the SIGNED distance-to-SGM coordinate.

    Construction (documented in DETECTOR_WORKFLOW.md and the Nuisance_DLI note): in prior-range-
    normalized absolute space, the magnitude is the distance to the Sample Geometric Median ``g`` of
    the source vectors (a member: the exact medoid up to the kernel's capacity; the method is in the
    provenance), and the sign is the side of ``g`` along the cloud's
    main axis of variation (PC1 of the ``g``-centered points, oriented toward increasing brightness
    ``mu_pc`` -- dim/narrow -> bright/wide). So ``p50`` is exactly ``g``; ``p < 50`` walks the negative
    (dim/narrow) side to the extreme at ``p=0``; ``p > 50`` walks the positive (bright/wide) side to
    the extreme at ``p=100``. Every returned vector is a real, whole member whose coordinates
    co-occurred (a multi-member pool keeps the collection's realized joint structure; a single frozen
    vector carries none);
    repeated percentiles (or a percentile on an empty side, which falls back to ``g``) collapse.
    Returns ``(members_log (M, D), provenance)``."""
    vecs_log = np.asarray(vecs_log, dtype=float)
    lo, hi = np.asarray(prior_low, float), np.asarray(prior_high, float)
    range_abs = 10.0 ** hi - 10.0 ** lo
    x = (10.0 ** vecs_log) / range_abs                        # normalized absolute space
    g_idx, method = sample_geometric_median(10.0 ** vecs_log, range_abs)
    d = x - x[g_idx]
    if x.shape[0] > 1:                                         # main axis: PC1 of the SGM-centered cloud
        evals, evecs = np.linalg.eigh(np.atleast_2d(np.cov(d, rowvar=False)))
        u = evecs[:, int(np.argmax(evals))]
        if u[brightness_index] < 0:                           # orient toward increasing brightness
            u = -u
    else:
        u = np.zeros(x.shape[1])
    s = np.sign(d @ u) * np.linalg.norm(d, axis=1)            # signed distance-to-SGM (g -> 0)
    neg_idx = np.where(s < 0)[0][np.argsort(s[s < 0])]        # most-negative -> nearest-below
    pos_idx = np.where(s > 0)[0][np.argsort(s[s > 0])]        # nearest-above -> most-positive
    picks, seen, notes = [], set(), []
    for p in percentiles:
        p = float(p)
        if abs(p - 50.0) < 1e-9:
            j = g_idx
        elif p < 50.0:
            if neg_idx.size == 0:
                j = g_idx; notes.append(f"p{p:g}: no vectors below the SGM -> SGM")
            else:
                j = int(neg_idx[int(round((p / 50.0) * (neg_idx.size - 1)))])
        else:
            if pos_idx.size == 0:
                j = g_idx; notes.append(f"p{p:g}: no vectors above the SGM -> SGM")
            else:
                j = int(pos_idx[int(round(((p - 50.0) / 50.0) * (pos_idx.size - 1)))])
        if j not in seen:
            seen.add(j)
            picks.append((p, j))
    members = np.stack([vecs_log[j] for _, j in picks], axis=0)
    provenance = {
        "selection": "sgm_percentiles",
        "percentiles": [float(p) for p in percentiles],
        "percentile_of_member": [float(p) for p, _ in picks],
        "member_rows": [int(j) for _, j in picks],
        "sgm_row": int(g_idx), "sgm_method": method,
        "axis": f"PC1 oriented toward increasing parameter index {brightness_index} (mu_pc)",
        "n_source": int(x.shape[0]), "notes": notes,
    }
    return members, provenance


def load_map_vectors(experiment_map_path, posterior_pool_path, *, source, condition,
                     prior_low, prior_high, parameter_keys):
    """Load condition-labeled per-window MAP vectors for the selection, REUSING already-computed
    data (no GPU). ``source='experiment'`` reuses the Detector Experiment product
    (``map_estimate`` + ``kind_index``/``kinds``; a selection built on it is an SGM of window MAPs),
    whose parameter keys must equal ``parameter_keys`` exactly (key and order);
    ``source='window-sgm'`` computes the per-window
    Sample Geometric Median from the labeled posterior-sample pool. Then restricts to ``condition``
    (``FAB`` or ``INLB``; None keeps every row) via the labels. Returns ``(vectors_log (n_win, D),
    source_label)``."""
    range_abs = 10.0 ** np.asarray(prior_high, float) - 10.0 ** np.asarray(prior_low, float)
    if source == "experiment":
        p = Path(experiment_map_path)
        if not p.exists():
            raise FileNotFoundError(
                f"sgm_percentiles selection_source='experiment' needs the Detector Experiment MAP "
                f"output:\n    {p}\nRun the Detector Experiment stage first (it is the reused source).")
        arrays, manifest = schema.load_product(p, stage="experiment")
        schema.assert_parameter_keys(manifest, parameter_keys, source=str(p))
        vecs = np.asarray(arrays["map_estimate"], dtype=float)
        kind_index = np.asarray(arrays["kind_index"])
        kinds = list(np.asarray(arrays["kinds"]))
        label = f"window MAPs (Experiment map_estimate, {p.name})"
    elif source == "window-sgm":
        p = Path(posterior_pool_path)
        labels = load_pool_labels(p)
        if labels is None:
            raise FileNotFoundError(
                f"sgm_percentiles selection_source='window-sgm' needs the labeled posterior-sample "
                f"pool:\n    {p}\nBuild it (--emit-template) or migrate it (--migrate-pool-labels).")
        flat = np.asarray(np.load(str(p), allow_pickle=False)["pool"], dtype=float)
        vecs, kind_index, kinds = _per_window_sgm_labeled(flat, labels, range_abs)
        label = f"window-SGM of the posterior pool ({p.name})"
    else:
        raise ValueError(f"unknown selection_source {source!r} (use 'experiment' or 'window-sgm').")
    if condition is not None:
        if condition not in kinds:
            raise ValueError(f"condition {condition!r} is not among the source kinds {kinds}.")
        mask = kind_index == kinds.index(condition)
        if not mask.any():
            raise ValueError(f"condition {condition!r}: the source has no MAP vectors for it.")
        vecs = vecs[mask]
        label += f", condition {condition}"
    return vecs, label


# =============================================================================
# Pool cache — separate the expensive GPU pool from the cheap per-choice representation
# =============================================================================
# A pool is expensive (embedding + sampling, or 1 MAP optimization per chunk); the choice
# applied to it is cheap. So each pool is persisted once, stamped with the exact inputs that
# produced it, and reused across `--build` invocations (and across raw/gaussian/box, which
# share the posterior-sample pool). It is recomputed only when an input changes.

def pool_cache_path(posit_dir, project_alias, timing_label, pool_kind):
    """Path of a cached pool (``pool_kind`` in ``{"PosteriorSample", "MapEstimate"}``)."""
    return Path(posit_dir) / f"{project_alias}_{timing_label}_Nuisance_DLI_{pool_kind}Pool.npz"


def pool_provenance(*, pool_mode, n_per_chunk, span_seconds, chunk_step_seconds, kinds,
                    max_cells, estimator_sha256, map_contract=None):
    """The cache key: every input that determines the pool's contents. A cached pool is
    reused only when this matches exactly, so any change (including a different estimator,
    identified by its weights checksum) forces a recompute.

    ``map_contract`` is given for the **MapEstimate** pool only (:func:`live_map_pool_contract`):
    the MAP computation contract -- schema and estimate-definition versions, the implementation
    hash of the code that runs the optimizer, its settings and the checkpoint. A MAP pool cached
    under a different contract is stale and is recomputed; the draw-only PosteriorSample pools,
    whose provenance carries no such entry, are untouched by an optimizer-only change."""
    prov = {
        "pool_mode": pool_mode,
        "n_per_chunk": int(n_per_chunk),
        "span_seconds": int(span_seconds),
        "chunk_step_seconds": (None if chunk_step_seconds is None else int(chunk_step_seconds)),
        "kinds": list(kinds),
        "max_cells": int(max_cells),
        "estimator_sha256": estimator_sha256,
    }
    if map_contract is not None:
        prov["map_contract"] = dict(map_contract)
    return prov


def live_map_pool_contract(eval_cfg, *, pool_mode, checkpoint_sha256) -> dict:
    """The MAP computation contract of a MapEstimate pool built NOW by this code: the pool runs the
    optimizer itself (it is not read from an Experiment product), so its identity is the live
    implementation hash plus the config-default optimizer settings the build uses."""
    from .evaluation import ESTIMATE_DEFINITIONS_VERSION, optimizer_contract
    from .provenance import code_provenance
    return {
        "artifact_schema_version": schema.ARTIFACT_SCHEMA_VERSION,
        "estimate_definitions_version": ESTIMATE_DEFINITIONS_VERSION,
        "implementation_sha256": code_provenance()["implementation"]["sha256"],
        "optimizer": optimizer_contract(
            eval_cfg,
            learning_rate=eval_cfg.learning_rate,
            tolerance=eval_cfg.tolerance,
            theta_prex_size=eval_cfg.theta_prex_size, elite_prex_size=eval_cfg.elite_prex_size,
            numb_steps=eval_cfg.numb_steps, pool_mode=pool_mode),
        "checkpoint_sha256": checkpoint_sha256,
    }


def map_pool_contract(experiment_manifest: dict) -> dict:
    """The same contract read from an Experiment product's manifest -- for comparing a cached
    MAP pool against the product whose MAP vectors it should agree with."""
    m = experiment_manifest
    return {
        "artifact_schema_version": m["artifact_schema_version"],
        "estimate_definitions_version": m["estimate_definitions_version"],
        "implementation_sha256": m["code"]["implementation"]["sha256"],
        "optimizer": dict(m["optimizer"]),
        "checkpoint_sha256": m["checkpoint_sha256"],
    }


def save_pool(path, pool, provenance, *, labels=None):
    """Persist a pool matrix + its provenance to ``path`` (compressed ``.npz``).

    When ``labels`` is given -- a ``{"kind_index", "cell", "chunk", "kinds"}`` dict of the
    per-row condition and time-window (cell, sliding-window chunk) fields row-aligned to
    ``pool`` (``kinds`` is the shared name array ``kind_index`` indexes into) -- the pool
    becomes self-describing and is stamped with ``pool_format_version``. Omitting ``labels``
    writes the legacy label-less layout, so an un-upgraded caller is unaffected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"pool": np.asarray(pool, dtype=float),
              "provenance": np.asarray(json.dumps(provenance, sort_keys=True))}
    if labels is not None:
        n = arrays["pool"].shape[0]
        for key in ("kind_index", "cell", "chunk"):
            col = np.asarray(labels[key])
            if col.shape[0] != n:
                raise ValueError(
                    f"save_pool: labels['{key}'] length {col.shape[0]} != pool rows {n}.")
            arrays[key] = col.astype(np.int64)
        arrays["kinds"] = np.asarray(labels["kinds"])
        arrays["pool_format_version"] = np.asarray(POOL_FORMAT_VERSION)
    np.savez_compressed(str(path), **arrays)


def load_pool_if_fresh(path, provenance):
    """Return the cached pool iff ``path`` exists and its stored provenance matches
    ``provenance`` exactly; otherwise ``None`` (missing or stale -> caller recomputes).

    Returns the pool matrix only; per-row labels (when present) are read separately via
    :func:`load_pool_labels`, since a representation fit does not need them."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(str(path), allow_pickle=False) as data:
        stored = json.loads(str(data["provenance"]))
        if stored == provenance:
            return data["pool"]
    return None


def load_pool_labels(path):
    """Return the pool cache's per-row labels dict (``kind_index``, ``cell``, ``chunk``,
    ``kinds``, ``pool_format_version``) for a self-describing (labeled) pool, else ``None``
    for a legacy label-less pool. Reads only the small label arrays, not the pool matrix."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(str(path), allow_pickle=False) as data:
        if "pool_format_version" not in data.files:
            return None
        return {"kind_index": data["kind_index"], "cell": data["cell"],
                "chunk": data["chunk"], "kinds": data["kinds"],
                "pool_format_version": int(data["pool_format_version"])}


# =============================================================================
# Emit the spec template (pre-filled with pool-derived suggestions)
# =============================================================================
def emit_spec_template(path, imaging_keys, suggestions, *, pool_mode="bounded", provenance=None):
    """Write a commented value-based Nuisance_DLI spec pre-filled with SUGGESTIONS.

    ``suggestions[key]`` may carry ``ci_low`` / ``ci_high`` (log10, the sampling space),
    used to pre-fill the per-parameter ranges that ``box_user`` reads. ``pool_mode`` is the
    mode the suggestion pool was drawn under and becomes the spec's default (so the finalized
    spec matches how emit sampled). Every number is a suggestion; the user edits the file to
    set the ultimate choice and values. Nothing here is authoritative until the user saves it.
    """
    def _num(v):
        return "null" if v is None else repr(round(float(v), 4))

    ql, qh = DEFAULT_BOX_QUANTILES
    out = [
        "# ============================================================================",
        "# Nuisance_DLI spec -- EDIT THIS FILE, then build with --build.",
        "# ============================================================================",
        "# The detector estimator is REQUIRED. --build reads ALL experimental recordings,",
        "# draws the posterior per model-window chunk on the fly, and pools the draws into",
        "# the posterior_sample_pool. `posterior_sample_pool_choice` decides how that pool",
        "# becomes the samplable Nuisance_DLI. All values are in LOG10 (the sampling space).",
        "#",
        "#   raw               -> resample the pool per whole vector (keeps the pool's realized joint structure; DEFAULT).",
        "#   map_estimate_pool -> resample the pooled per-chunk MAP estimates (point estimates).",
        "#   gaussian          -> full-covariance Gaussian fit to the pool (linear correlations).",
        "#   box               -> per-parameter uniform over `box_quantiles` of the pool, clamped.",
        "#   box_user          -> per-parameter uniform over the [imaging.<KEY>] ranges below, clamped.",
        "#   selection_user    -> ONE fixed, user-selected vector from documented sources; write its spec",
        "#                        with --emit-selection-user (no estimator, no GPU), not from this template.",
        "#   sgm_percentiles   -> whole real vectors at SIGNED distance-to-SGM percentiles. REUSES the",
        "#                        Detector Experiment MAPs (or the labeled posterior pool); NO GPU. In",
        "#                        prior-range-normalized absolute space: magnitude = distance to the",
        "#                        Sample Geometric Median g (the median vector); sign = side of g along",
        "#                        the cloud's main variation axis (PC1, toward brighter mu_pc). p50 = g",
        "#                        EXACTLY (a frozen vector); p<50 walks the dim/narrow side to p=0, p>50",
        "#                        the bright/wide side to p=100. Several percentiles -> a small,",
        "#                        correlation-preserving marginalization pool. Set `percentiles`,",
        "#                        `condition`, `selection_source` below.",
        "#",
        "# pool_mode matches Evaluation/Experiment: bounded (rejection within the prior) | unrestricted.",
        "# raw / map_estimate_pool / gaussian are NOT clipped; box / box_user clamp to the prior box.",
        "# sgm_percentiles reuses already-computed data and ignores pool_mode.",
        f"# provenance (auto-filled, do not edit): {json.dumps(provenance or {}, sort_keys=True)}",
        "",
        "[block]",
        'posterior_sample_pool_choice = "raw"   # raw|map_estimate_pool|gaussian|box|box_user|sgm_percentiles',
        f'pool_mode = "{pool_mode}"                  # bounded | unrestricted',
        f"box_quantiles = [{ql}, {qh}]                # used only by \"box\"",
        "",
        '# Read ONLY by "sgm_percentiles":',
        'percentiles = [50]                       # 50 = the SGM (a single frozen vector); e.g. [5,25,50,75,95] = a 5-vector pool',
        '# condition = "FAB"                      # optional: FAB (MET-FAB, monomer) | INLB (MET-INLB, dimer); default = the artifact\'s own condition',
        'selection_source = "experiment"          # experiment (reuse the Experiment MAPs) | window-sgm (posterior pool)',
        "",
        "# Per-parameter ranges are read ONLY by \"box_user\"; pre-filled with the pool's",
        f"# {int(ql * 100)}th/{int(qh * 100)}th percentiles as suggestions. Ignored by every other choice.",
    ]
    for k in imaging_keys:
        s = suggestions.get(k, {}) if suggestions else {}
        out += [
            f"[imaging.{k}]",
            f"low  = {_num(s.get('ci_low'))}",
            f"high = {_num(s.get('ci_high'))}",
            "",
        ]
    Path(path).write_text("\n".join(out))
    return Path(path)


def emit_selection_user_template(path, table, *, provenance=None, comparison=None):
    """Write the spec skeleton of a ``selection_user`` Nuisance_DLI (CPU; no estimator).

    ``table`` is the six-row detector parameter table (``KEY``, ``PRIOR_RANGE`` in log10). Every
    value line is written COMMENTED OUT, so the file cannot be built before the user has typed each
    value, its source and its limitation; ``comparison`` (``{key: physical}``) is printed beside
    each row as a reference only. Physical units: the values are typed in physical units and stored
    in log10 by the build.
    """
    out = [
        "# ============================================================================",
        "# Nuisance_DLI spec -- selection_user: ONE explicitly chosen, fixed imaging vector.",
        "# ============================================================================",
        "# Fill every [selection.<KEY>] table: `value` in PHYSICAL units, `source` (where the value",
        "# comes from: which estimator, product and aggregation, or which measurement) and",
        "# `limitation` (what the value does not establish). The build converts to log10, refuses a",
        "# value outside the detector prior box unless its key is listed in allow_outside_prior with",
        "# a non-empty outside_prior_justification (an acknowledgement of extrapolation beyond the",
        "# tested imaging domain, not a certification), never clips, needs no estimator and no GPU,",
        "# and records the vector as user-selected and fixed, with no calibrated joint uncertainty.",
        f"# provenance (auto-filled, do not edit): {json.dumps(provenance or {}, sort_keys=True)}",
        "",
        "[block]",
        'posterior_sample_pool_choice = "selection_user"',
        "allow_outside_prior = []                 # keys whose value may sit outside the prior box",
        'outside_prior_justification = ""         # required (non-empty) when allow_outside_prior is not empty',
        "",
    ]
    for e in table:
        k = e["KEY"]
        lo, hi = e["PRIOR_RANGE"]
        cmp = "" if not comparison or k not in comparison else f"   (comparison value: {comparison[k]:.6g})"
        out += [
            f"[selection.{k}]",
            f"# prior box (physical): [{10 ** lo:.6g}, {10 ** hi:.6g}]{cmp}",
            "# value = <physical value>",
            '# source = "<estimator or measurement, product, aggregation rule>"',
            '# limitation = "<what this value does not establish>"',
            "",
        ]
    Path(path).write_text("\n".join(out))
    return Path(path)


def selection_from_spec(spec, table, prior_low, prior_high):
    """Validate a ``selection_user`` spec against the table; return ``(vector_log10, record)``.

    Every key needs ``value`` (physical, finite, inside its physical domain), ``source`` and
    ``limitation`` (non-empty strings). The value is converted to the storage space of its row
    (log10 for a ranged row). A coordinate outside the prior box is refused unless its key appears
    in ``[block].allow_outside_prior`` and ``[block].outside_prior_justification`` is non-empty;
    nothing is clipped. Raises ``ValueError`` with the offending key, value and box.
    """
    block = spec.get("block", {})
    allow = block.get("allow_outside_prior", []) or []
    just = str(block.get("outside_prior_justification", "") or "").strip()
    if not isinstance(allow, (list, tuple)) or not all(isinstance(a, str) for a in allow):
        raise ValueError("[block].allow_outside_prior must be a list of parameter keys.")
    keys = [e["KEY"] for e in table]
    unknown = [a for a in allow if a not in keys]
    if unknown:
        raise ValueError(f"[block].allow_outside_prior names unknown keys {unknown}; keys are {keys}.")
    if allow and not just:
        raise ValueError("[block].outside_prior_justification must be non-empty when "
                         "allow_outside_prior lists a key.")
    sel = spec.get("selection", {})
    missing = [k for k in keys if k not in sel]
    if missing:
        raise ValueError(f"choice 'selection_user' needs a [selection.<KEY>] table for every "
                         f"parameter; missing {missing}.")
    vec, physical, sources, outside = [], {}, {}, []
    tol = 1e-9
    for i, e in enumerate(table):
        k = e["KEY"]
        row = sel[k]
        if not isinstance(row, dict) or "value" not in row:
            raise ValueError(f"[selection.{k}] needs `value` (physical units).")
        try:
            v = float(row["value"])
        except (TypeError, ValueError):
            raise ValueError(f"[selection.{k}].value = {row['value']!r} is not a number.")
        if not np.isfinite(v):
            raise ValueError(f"[selection.{k}].value must be finite; got {v!r}.")
        src = str(row.get("source", "") or "").strip()
        lim = str(row.get("limitation", "") or "").strip()
        if not src or not lim:
            raise ValueError(f"[selection.{k}] needs non-empty `source` and `limitation` strings.")
        err = physical_domain_error(k, v, e)
        if err:
            raise ValueError(f"[selection.{k}].value: {err}.")
        if e.get("PRIOR_RANGE") is None:
            stored = v
        else:
            stored = float(np.log10(v)) if e.get("LOG_BASE", 10) == 10 else float(
                np.log(v) / np.log(e["LOG_BASE"]))
        lo, hi = float(prior_low[i]), float(prior_high[i])
        if stored < lo - tol or stored > hi + tol:
            if k not in allow:
                raise ValueError(
                    f"[selection.{k}].value = {v} (stored {stored:+.4f}) lies outside the detector "
                    f"prior box [{lo:+.4f}, {hi:+.4f}] (physical [{10 ** lo:.6g}, {10 ** hi:.6g}]). "
                    f"A value outside the tested imaging domain is accepted only when the spec "
                    f"acknowledges it: list '{k}' in [block].allow_outside_prior and give a "
                    f"non-empty outside_prior_justification. Nothing is clipped.")
            outside.append(k)
            logger.warning("Nuisance_DLI[selection_user]: %s = %s lies outside the prior box "
                           "[%s, %s]; accepted by explicit acknowledgement (extrapolation beyond the "
                           "tested imaging domain, not certified).", k, v, lo, hi)
        vec.append(stored)
        physical[k] = v
        sources[k] = {"source": src, "limitation": lim}
    record = {"kind": "user", "fixed": True, "input_units": "physical",
              "values_physical": physical,
              "values_log10": {k: float(x) for k, x in zip(keys, vec)},
              "sources": sources, "outside_prior": outside,
              "outside_prior_justification": just if outside else "",
              "joint_uncertainty": "none (a user-selected, fixed vector; no calibrated joint uncertainty)"}
    return np.asarray(vec, dtype=float), record


REPRESENTATIVE_GROUPINGS = ("per_cell", "per_window_position", "global", "per_cell_window")


def emit_representatives_template(path, archive, *, provenance=None, reference=None, target=None):
    """Write the spec skeleton of an ``empirical_representatives`` Nuisance_DLI from an Experiment candidate
    archive (``experiment_aggregation.load_candidate_archive``): every available component (condition, window
    set, grouping) is listed COMMENTED OUT with its member count, so the file cannot be built before the user
    has named the components, their weights and the rationale. ``reference`` (``{key: physical}``) is printed
    beside the block as a comparison only. ``target`` (condition, timing label; required) is checked against the
    archive's recorded target before anything is written (:func:`check_archive_target`)."""
    a, m = archive["arrays"], archive["manifest"]
    check_archive_target(archive, target, where="template emission")
    combos = {}
    for k, w, c in zip(a["kind"], a["window_set"], a["construction"]):
        combos[(str(k), str(w), str(c))] = combos.get((str(k), str(w), str(c)), 0) + 1
    out = [
        "# ============================================================================",
        "# Nuisance_DLI spec -- empirical_representatives: a WEIGHTED distribution over complete",
        "# representative vectors selected from the Experiment stage's candidate archive.",
        "# ============================================================================",
        "# Name K components ([[component]] tables): each is one (source Experiment, condition, window",
        "# set, grouping, statistic) collection of the archive with an explicit weight alpha_k. The",
        "# weights must sum to 1 (tolerance 1e-6). `member_weighting` selects how alpha_k is spread over",
        "# the component's members: \"uniform\" (the default: every member receives alpha_k / n_k, so a",
        "# 600-member collection never outweighs a 60-member one by its row count) or",
        "# \"posterior_rank_median\" (consensus weighting: a member's probability is proportional to the",
        "# median cross-posterior distance rank Experiment exported for its SGM; a zero score is exactly zero",
        "# probability, kept and shown; an all-zero normalizer is refused; requires statistic \"sgm\" and an",
        "# archive that carries scores for the grouping).",
        "# `rank_scope` (REQUIRED with posterior_rank_median) names the comparison scope of the scores and must",
        "# be the scope this archive computed them under: \"construction\" = every representative against ALL the",
        "# other representatives of its grouping, condition and window set, probabilities normalized over the",
        "# whole component (p_i = alpha * s_i / sum of the component's scores; under per_cell_window no cell",
        "# keeps a reserved mass, and a cell's total may be small or zero); \"within_cell\" (per_cell_window in an",
        "# archive of that comparison scope: formats 2 and 3) = each cell-window SGM against the other windows of",
        "# its own cell, equal cell mass.",
        "# `rank_coordinates` (optional, with posterior_rank_median only) names the parameter keys whose",
        "# rank space of the archive supplies the scores (default: every coordinate, the space \"all\"); the",
        "# archive must hold a space over exactly those keys (Experiment --compatibility-coordinates).",
        "# Exact duplicate vectors are consolidated by adding their probabilities (memberships kept).",
        "# `statistic` is \"sgm\" (a retained joint draw; the default) or \"median\" (the marginal-median",
        "# vector of the group, a coordinate-wise composite, never a sampled row). `rationale` records",
        "# the scientific reason for the selection. An [override.<KEY>] table (value in physical units,",
        "# reason) is an explicit parameter-specific treatment applied to EVERY selected vector and shown",
        "# in the report; a transformed vector is no longer the original joint draw. The reference",
        "# scenario (SEL) is a comparison only and is never mixed in automatically. No component is",
        "# selected here: this file is a skeleton, and the example weights below are NOT a decision.",
        f"# provenance (auto-filled, do not edit): {json.dumps(provenance or {}, sort_keys=True)}",
        "",
        "[block]",
        'posterior_sample_pool_choice = "empirical_representatives"',
        f'candidate_archive = "{archive["path"]}"',
        f'candidate_archive_sha256 = "{archive["sha256"]}"',
        'rationale = ""                            # REQUIRED: why these components, this window set and this statistic',
        "",
        f"# archive target (binding): {json.dumps(m.get('target'), sort_keys=True, default=str)}",
        f"# archive: product {m.get('product_label')}, conditions {m.get('condition_labels')}, window sets "
        f"{ {k: v for k, v in m.get('window_sets', {}).items()} }, written {m.get('written_at')}",
        f"# coverage: {m.get('coverage')}" + (f"; NOTE {m.get('coverage_note')}" if m.get("coverage_note") else ""),
    ]
    if reference:
        out.append("# reference (comparison only): " + ", ".join(f"{k} {v:.6g}" for k, v in reference.items()))
    avail = component_weighting_availability(archive)
    out += ["", f"# member_weighting policies: {list(MEMBER_WEIGHTINGS)}; archive format {archive.get('format_version', 1)}"
            + (f" (compatibility scores present; comparison scope {archive['compatibility'].get('comparison_scope')!r}: per grouping "
               f"{archive['compatibility'].get('comparison_scopes')})" if archive.get("compatibility") else
               " (no compatibility scores: posterior_rank_median needs a format-2 or later archive; regenerate the Experiment report on the CPU)"),
            "# Available components (uncomment, set the weights so they sum to 1):"]
    for (k, w, c), n in sorted(combos.items()):
        out += ["# [[component]]", f'# source_experiment = "{m.get("product_label")}"', f'# condition = "{k}"', f'# window_set = "{w}"',
                f'# grouping = "{c}"                       # {n} representative(s)', '# statistic = "sgm"', "# weight = 1.0",
                f'# member_weighting = "uniform"           # posterior_rank_median: {avail[(k, w, c)]}',
                '# rank_scope = "construction"            # with posterior_rank_median only; must match the archive (see above)', ""]
    out += ["# Optional explicit parameter-specific treatment (shown in the report; NOT applied unless uncommented):",
            "# [override.prob_photo_bleach]", "# value = 0.05", '# reason = "<the recorded decision this value rests on>"', ""]
    Path(path).write_text("\n".join(out))
    return Path(path)


def component_weighting_availability(archive) -> dict:
    """Per (condition, window set, grouping) of an archive, one statement on whether ``posterior_rank_median`` can be
    selected, under the comparison scope the archive computed that grouping's scores in (the ``rank_scope`` a spec must
    name), evaluated SEPARATELY for every rank coordinate space the archive holds (``rank space <label> (<keys>): ...``
    clauses joined by '; '): per space the computed-score count over the members and, when every score is computed,
    whether the constructor's own normalization of that space's scores succeeds (``member_probabilities`` with that
    space's coordinates and the archive's scope); the unavailable reason otherwise (missing scores, or computed scores
    that cannot be normalized, the computed zeros remaining valid diagnostics). A space's availability never stands in
    for another's. Groupings without scores in the archive's scope and format-1 archives keep their single statements."""
    a = archive["arrays"]; compat = archive.get("compatibility")
    out = {}
    for k, w, c, i in zip(a["kind"], a["window_set"], a["construction"], range(a["kind"].shape[0])):
        key = (str(k), str(w), str(c))
        out.setdefault(key, []).append(i)
    text = {}
    for key, idx in out.items():
        c = key[2]
        if c not in RANK_WEIGHTED_GROUPINGS:
            text[key] = "not applicable to this grouping (uniform only)"
            continue
        if compat is None:
            text[key] = "unavailable: archive format 1 without compatibility scores (regenerate the Experiment report on the CPU)"
            continue
        from .experiment_aggregation import archive_component
        comp = archive_component(archive, kind=key[0], window_set=key[1], construction=c)
        scope = comp.get("comparison_scope")
        if scope is None:
            text[key] = (f"not applicable to this grouping in this archive (its comparison scope {comp.get('archive_comparison_scope')!r} "
                         f"scores no {c} representatives; uniform only)")
            continue
        spaces = compat.get("spaces") or {RANK_SPACE_ALL: {**compat, "coordinates": list(archive["keys"])}}
        clauses = []
        for lab, sp in spaces.items():
            head = f"rank space {lab} ({', '.join(sp['coordinates'])}): "
            st = [sp["by_group"][int(i)]["status"] for i in idx]
            n_ok = sum(1 for s in st if s == "computed")
            if n_ok != len(st):
                reasons = sorted(set(s for s in st if s != "computed"))
                clauses.append(head + f"unavailable under rank_scope \"{scope}\": {len(st) - n_ok}/{len(st)} members without a computed score "
                                      f"({', '.join(reasons)}; a population of one group has no score)")
                continue
            # scores computed is not the same as normalizable: apply the constructor's own normalization rules to THIS space
            try:
                member_probabilities("posterior_rank_median", 1.0, comp, where="availability", rank_scope=scope,
                                     rank_coordinates=None if lab == RANK_SPACE_ALL else list(sp["coordinates"]))
            except ValueError as exc:
                reason = str(exc).split(": ", 1)[-1].split(";")[0]
                clauses.append(head + f"unavailable under rank_scope \"{scope}\": compatibility scores computed for {n_ok}/{len(st)} members, "
                                      f"but the policy cannot be normalized ({reason}); the computed zero scores remain valid diagnostics")
                continue
            clauses.append(head + f"available under rank_scope \"{scope}\", compatibility scores computed for {n_ok}/{len(st)} members and normalizable")
        text[key] = "; ".join(clauses)
    return text


def _validate_representatives_block(spec, imaging_keys, where, table=None):
    block = spec.get("block", {})
    entries = {e["KEY"]: e for e in (table or [])}
    arc = str(block.get("candidate_archive", "") or "").strip()
    if not arc:
        raise ValueError(f"{where}: [block].candidate_archive must name the Experiment aggregation_candidates.npz.")
    if not str(block.get("rationale", "") or "").strip():
        raise ValueError(f"{where}: [block].rationale must be a non-empty statement of the selection's reason.")
    comps = spec.get("component", [])
    if not isinstance(comps, list) or not comps:
        raise ValueError(f"{where}: at least one [[component]] table is required.")
    weights = []
    for i, c in enumerate(comps):
        if not isinstance(c, dict):
            raise ValueError(f"{where}: [[component]] {i} is not a table.")
        for k in ("source_experiment", "condition", "window_set", "grouping"):
            if not str(c.get(k, "") or "").strip():
                raise ValueError(f"{where}: [[component]] {i} needs a non-empty `{k}`.")
        if c.get("grouping") not in REPRESENTATIVE_GROUPINGS:
            raise ValueError(f"{where}: [[component]] {i}.grouping={c.get('grouping')!r} must be one of {REPRESENTATIVE_GROUPINGS}.")
        if c.get("statistic", "sgm") not in REPRESENTATIVE_STATISTICS:
            raise ValueError(f"{where}: [[component]] {i}.statistic={c.get('statistic')!r} must be one of {REPRESENTATIVE_STATISTICS}.")
        mw = c.get("member_weighting", "uniform")
        if mw not in MEMBER_WEIGHTINGS:
            raise ValueError(f"{where}: [[component]] {i}.member_weighting={mw!r} must be one of {MEMBER_WEIGHTINGS}; no other policy "
                             f"exists and nothing falls back to uniform.")
        if mw == "posterior_rank_median":
            if c.get("statistic", "sgm") != "sgm":
                raise ValueError(f"{where}: [[component]] {i}: member_weighting 'posterior_rank_median' requires statistic = \"sgm\": the "
                                 f"compatibility scores describe the SGM representatives and are never reused for marginal-median vectors.")
            if c.get("grouping") not in RANK_WEIGHTED_GROUPINGS:
                raise ValueError(f"{where}: [[component]] {i}: member_weighting 'posterior_rank_median' is available for groupings "
                                 f"{RANK_WEIGHTED_GROUPINGS} only; {c.get('grouping')!r} has no peer and carries no rank score (select "
                                 f"'uniform': its single representative then receives the component's whole mass).")
            rs = c.get("rank_scope")
            if rs not in RANK_SCOPES:
                raise ValueError(f"{where}: [[component]] {i}: member_weighting 'posterior_rank_median' needs an explicit rank_scope, one of "
                                 f"{RANK_SCOPES} (\"construction\": every representative against all the other representatives of its "
                                 f"grouping, condition and window set, normalized over the whole component; \"within_cell\": the hierarchical "
                                 f"policy of an archive of that comparison scope (formats 2 and 3), per_cell_window only); got {rs!r}. The scope is "
                                 f"never implied by the grouping.")
            if rs == "within_cell" and c.get("grouping") != "per_cell_window":
                raise ValueError(f"{where}: [[component]] {i}: rank_scope \"within_cell\" applies to grouping per_cell_window only; "
                                 f"{c.get('grouping')!r} is compared across its whole construction (rank_scope \"construction\").")
        elif c.get("rank_scope") is not None:
            raise ValueError(f"{where}: [[component]] {i}: rank_scope applies to member_weighting 'posterior_rank_median' only (uniform members "
                             f"carry no rank); remove it or select the rank policy.")
        rc = c.get("rank_coordinates")
        if rc is not None:
            if mw != "posterior_rank_median":
                raise ValueError(f"{where}: [[component]] {i}: rank_coordinates applies to member_weighting 'posterior_rank_median' only "
                                 f"(uniform members carry no rank); remove it or select the rank policy.")
            if not isinstance(rc, list) or not rc or any(str(k) not in imaging_keys for k in rc) or len({str(k) for k in rc}) != len(rc):
                raise ValueError(f"{where}: [[component]] {i}: rank_coordinates must be a non-empty list of distinct parameter keys from "
                                 f"{list(imaging_keys)}; got {rc!r}.")
        try:
            w = float(c.get("weight"))
        except (TypeError, ValueError):
            raise ValueError(f"{where}: [[component]] {i}.weight = {c.get('weight')!r} is not a number.")
        if not np.isfinite(w) or w < 0:
            raise ValueError(f"{where}: [[component]] {i}.weight = {w} must be finite and non-negative.")
        weights.append(w)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError(f"{where}: the component weights have no positive total mass ({weights}).")
    if abs(total - 1.0) > WEIGHT_NORMALIZATION_TOLERANCE:
        raise ValueError(f"{where}: the component weights sum to {total:.9f}, not 1 within {WEIGHT_NORMALIZATION_TOLERANCE}: "
                         f"{weights}. Write the intended normalized weights; nothing is repaired silently.")
    over = spec.get("override", {}) or {}
    if not isinstance(over, dict):
        raise ValueError(f"{where}: [override.<KEY>] tables expected.")
    for k, row in over.items():
        if k not in imaging_keys:
            raise ValueError(f"{where}: [override.{k}] names an unknown parameter; keys are {list(imaging_keys)}.")
        if not isinstance(row, dict) or "value" not in row or not str(row.get("reason", "") or "").strip():
            raise ValueError(f"{where}: [override.{k}] needs `value` (physical units) and a non-empty `reason`.")
        err = physical_domain_error(k, row["value"], entries.get(k, {"PRIOR_RANGE": (0, 1)}))
        if err:
            raise ValueError(f"{where}: [override.{k}].value: {err} (physical units).")
    return comps, weights


TARGET_FIELDS = ("condition", "timing_label")


def check_archive_target(archive, target, *, where="candidate archive"):
    """The archive's recorded target (the Experiment run's condition, timing label, geometry) must equal the
    ``target`` the artifact is built for; a native nuisance artifact of one condition and duration is never
    built from another's Experiment, and an archive without a target binding is refused. Returns the binding."""
    if not target or any(not target.get(k) for k in TARGET_FIELDS):
        raise ValueError(f"{where}: the target identity (condition, timing_label) of the artifact under construction "
                         f"must be passed explicitly (target=); got {target!r}.")
    m = archive["manifest"]
    bound = m.get("target")
    if not bound:
        raise ValueError(f"{where} ({archive['path']}) records no target binding (condition, timing label, geometry); it was "
                         f"written before target binding. Regenerate the Experiment report with the current implementation.")
    for k in TARGET_FIELDS:
        if str(target[k]) != str(bound.get(k)):
            raise ValueError(f"{where}: target {k} {target[k]!r} != the archive's {bound.get(k)!r} (source Experiment "
                             f"{m.get('product_label')}). A native Nuisance DLI of {target['condition']} {target['timing_label']} "
                             f"is built only from that condition's and duration's Experiment; another Experiment's candidates "
                             f"are inspection results, not nuisance sources.")
    if target.get("project_alias") and bound.get("project_alias") and str(target["project_alias"]) != str(bound["project_alias"]):
        raise ValueError(f"{where}: project alias {target['project_alias']!r} != the archive's {bound['project_alias']!r}.")
    geom = (m.get("contract") or {}).get("window_geometry") or {}
    if target.get("n_frames") is not None and bound.get("n_frames") is not None and int(target["n_frames"]) != int(bound["n_frames"]):
        raise ValueError(f"{where}: window length {target['n_frames']} frames != the archive's {bound['n_frames']} (geometry {geom}).")
    if target.get("n_frames") is not None and geom.get("n_frames") is not None and int(target["n_frames"]) != int(geom["n_frames"]):
        raise ValueError(f"{where}: window length {target['n_frames']} frames != the archive contract's geometry {geom}.")
    return dict(bound)


def resolve_rank_space(comp, rank_coordinates, *, where="component"):
    """The archive rank space a component weights with: ``(label, coordinates)``. ``rank_coordinates`` None, or every
    parameter key, selects the primary space ``all``; otherwise the archive must hold a space over exactly those keys (in
    table order), else the selection is refused and the available spaces are named."""
    keys = [str(k) for k in (comp.get("parameter_keys") or [])]
    spaces = {str(lab): [str(k) for k in ks] for lab, ks in (comp.get("rank_spaces") or {}).items()}
    if not spaces:
        spaces = {RANK_SPACE_ALL: keys}
    if rank_coordinates is None or (keys and sorted(str(k) for k in rank_coordinates) == sorted(keys)):
        return RANK_SPACE_ALL, spaces.get(RANK_SPACE_ALL, keys)
    wanted = [k for k in keys if k in {str(c) for c in rank_coordinates}]
    for lab, ks in spaces.items():
        if ks == wanted:
            return lab, ks
    raise ValueError(f"{where}: no rank coordinate space of the archive runs over {wanted}; the archive holds {spaces}. Regenerate the "
                     f"Experiment report with --compatibility-coordinates <label>={','.join(wanted)} (CPU) and name that archive; nothing "
                     f"is approximated from another space.")


def member_probabilities(policy, weight, comp, *, where="component", rank_coordinates=None, rank_scope=None):
    """The member contributions of ONE component under ``policy`` (see ``MEMBER_WEIGHTINGS``): ``(probabilities (n,),
    normalizers, scores (n,) or None)``. ``comp`` is :func:`experiment_aggregation.archive_component`'s result; the
    scores are read through each member's identity record (``compatibility_by_space`` for the selected rank space,
    :func:`resolve_rank_space`; None = the primary space ``all``), never by archive position. ``rank_scope`` (required
    with the rank policy) is the comparison scope the selection asks for; it must equal the scope the archive computed
    this grouping's scores under (``comp["comparison_scope"]``), and it fixes the normalization: ``construction`` =
    ``weight * score / sum of the component's scores`` over the complete component, with no mass reserved per cell;
    ``within_cell`` (per_cell_window) = equal cell mass, then the score share within the cell. Refusals: a format-1
    archive or missing compatibility data; a missing, unknown or mismatched scope; a grouping without scores in the
    archive's scope; a rank space the archive does not hold; any member whose score is not ``computed``; an all-zero
    normalizer (the whole component under ``construction``; any included cell under ``within_cell``). A zero score
    keeps the member with exactly zero probability; under ``construction`` a whole cell may sum to zero."""
    idents = comp["identities"]; n = len(idents)
    if policy == "uniform":
        return np.full(n, float(weight) / n), {"scope": "component", "rule": "weight / n_members", "n_members": n}, None
    if policy != "posterior_rank_median":
        raise ValueError(f"{where}: unknown member_weighting {policy!r}; policies are {MEMBER_WEIGHTINGS}.")
    if not comp.get("compatibility_available"):
        raise ValueError(f"{where}: member_weighting 'posterior_rank_median' needs the compatibility scores of a format-2 (or later) candidate "
                         f"archive; this archive is format {comp.get('archive_format_version')} without them. Regenerate the Experiment report "
                         f"on the CPU (--regenerate-report --review-label <TOKEN>, with --cloud-sources when the product stores no clouds) and "
                         f"name the regenerated archive.")
    if rank_scope not in RANK_SCOPES:
        raise ValueError(f"{where}: member_weighting 'posterior_rank_median' needs an explicit rank_scope, one of {RANK_SCOPES}; got "
                         f"{rank_scope!r}. The scope is never implied by the grouping.")
    have = comp.get("comparison_scope")
    arc = (f"archive format {comp.get('archive_format_version')}, protocol {comp.get('compatibility_protocol')}, comparison scope "
           f"{comp.get('archive_comparison_scope')!r}")
    if have is None:
        raise ValueError(f"{where}: the archive carries no rank score for grouping {comp['construction']!r} ({arc}); select 'uniform', or name "
                         f"an archive whose comparison scope scores this grouping. Nothing falls back silently.")
    if have != rank_scope:
        raise ValueError(f"{where}: rank_scope {rank_scope!r} is not the scope this archive's {comp['construction']} scores were computed under "
                         f"({have!r}; {arc}). " + ("Construction-wide scores need a format-4 archive: regenerate the Experiment report on the CPU "
                         "(--regenerate-report --review-label <TOKEN>) and name that archive. " if rank_scope == "construction" else
                         "The hierarchical within-cell scores live in archives of the within_cell comparison scope (formats 2 and 3). ")
                         + "One scope's scores are never renormalized into another's.")
    from .experiment_aggregation import REFERENCE_POPULATIONS
    label, coords = resolve_rank_space(comp, rank_coordinates, where=where)
    def score_of(d):
        by = d.get("compatibility_by_space")
        return (by or {}).get(label) if by is not None else (d.get("compatibility") if label == RANK_SPACE_ALL else None)
    bad = [(d["cell"], d["chunk"], (score_of(d) or {}).get("status")) for d in idents if (score_of(d) or {}).get("status") != "computed"]
    if bad:
        raise ValueError(f"{where}: {len(bad)} member(s) carry no computed compatibility score in rank space {label!r} (cell, chunk, status): "
                         f"{bad[:6]}{' …' if len(bad) > 6 else ''}. A population of one group has no score; select 'uniform' for this component "
                         f"or a selection with peers. Nothing falls back silently.")
    s = np.asarray([float(score_of(d)["score"]) for d in idents], float)
    if not np.all(np.isfinite(s)) or np.any(s < 0):
        raise ValueError(f"{where}: the compatibility scores must be finite and non-negative; got {s.tolist()[:6]}.")
    refs = sorted({int(score_of(d)["n_references"]) for d in idents})
    space = {"rank_space": label, "rank_coordinates": list(coords), "comparison_scope": rank_scope,
             "reference_population": REFERENCE_POPULATIONS[rank_scope][comp["construction"]], "references_per_member": refs}
    if rank_scope == "construction":
        if refs != [n - 1]:
            raise ValueError(f"{where}: the members' scores were computed against {refs} references, not the other {n - 1} members of this "
                             f"component; the archive's scores do not cover the component's complete population.")
        tot = float(s.sum())
        if tot <= 0:
            raise ValueError(f"{where}: every score of this component is zero (sum {tot}) in rank space {label!r}; the selection cannot "
                             f"be normalized and is refused (no uniform fallback, no floor).")
        p = float(weight) * s / tot
        norm = {"scope": "component", "rule": "weight * score / sum of the component's scores", "sum_of_scores": tot, "n_members": n, **space}
        if comp["construction"] == "per_cell_window":
            cells = sorted({int(d["cell"]) for d in idents})
            mass = {str(c): float(sum(p[j] for j, d in enumerate(idents) if int(d["cell"]) == c)) for c in cells}
            norm.update({"n_cells": len(cells), "cell_mass_by_cell": mass, "zero_mass_cells": [int(c) for c, m in mass.items() if m == 0.0],
                         "cell_mass_rule": "the sum of the cell's member probabilities; no mass is reserved per cell"})
        return p, norm, s
    if comp["construction"] != "per_cell_window":
        raise ValueError(f"{where}: rank_scope \"within_cell\" applies to grouping per_cell_window only.")
    cells = sorted({int(d["cell"]) for d in idents}); C = len(cells)
    p = np.empty(n); sums = {}
    for c in cells:
        idx = [j for j, d in enumerate(idents) if int(d["cell"]) == c]
        tot = float(s[idx].sum()); sums[str(c)] = tot
        if tot <= 0:
            raise ValueError(f"{where}: cell {c}: every selected-window score is zero (sum {tot}) in rank space {label!r}; the selection is "
                             f"refused rather than dropping the cell or redistributing its mass (no uniform fallback, no floor).")
        p[idx] = (float(weight) / C) * s[idx] / tot
    return p, {"scope": "within each cell", "rule": "(weight / n_cells) * score / sum of that cell's selected-window scores",
               "n_cells": C, "cell_mass": float(weight) / C, "sum_of_scores_by_cell": sums, "n_members": n, **space}, s


def representatives_from_spec(spec, archive, table, prior_low, prior_high, *, spec_path=None, target=None, force_uniform=False):
    """Build the weighted representative distribution a finalized ``empirical_representatives`` spec declares
    from the loaded candidate ``archive``. Returns ``(vectors_log10, probabilities, record)``:

    - every component's members get their contributions under its ``member_weighting`` (:func:`member_probabilities`:
      ``uniform`` = ``weight / n_members``; ``posterior_rank_median`` = the normalization of Experiment's compatibility
      scores under the component's explicit ``rank_scope``, read through each member's identity); the contributions of a
      component sum to its weight;
    - exact duplicate vectors are consolidated by ADDING their probabilities; every membership is retained
      in the record (component index and member identity); rows are never deduplicated by displayed values; a zero
      contribution stays a zero contribution of that membership (nothing is transferred between memberships);
    - ``[override.<KEY>]`` tables replace that coordinate in every vector, recorded as a transformation with
      the replaced values' range; the result is then no longer the original joint draws and the record says so;
      the scores describe the ORIGINAL SGMs and are not recomputed for transformed points;
    - the record names the archive (path, checksum, product, target binding), each component's identities, scores and
      probabilities, the rationale, the zero counts, the effective number of rows, the outside-prior mass and whether
      the rows are sampled draws; when a component is rank-weighted, the record also carries the equal-member baseline
      (the same components, masses and overrides with uniform members, aligned to the same stored rows);
    - ``target`` (``condition``, ``timing_label``, optional ``project_alias``, ``n_frames``) is the identity of the
      artifact under construction: it must equal the archive's recorded target, and every component's recording
      condition must equal it (a cross-condition Experiment application stays an inspection result);
    - ``force_uniform`` builds the equal-member baseline of a spec (every component uniform; nothing else changes).
    """
    from .experiment_aggregation import archive_component
    keys = [e["KEY"] for e in table]
    comps, weights = _validate_representatives_block(spec, keys, "spec", table=table)
    block = spec["block"]
    m = archive["manifest"]
    bound = check_archive_target(archive, target)
    if archive["keys"] != keys:
        raise ValueError(f"the archive's parameter keys {archive['keys']} differ from the table's {keys}.")
    declared = str(block.get("candidate_archive_sha256", "") or "").strip()
    if declared and declared != archive["sha256"]:
        raise ValueError(f"candidate archive checksum {archive['sha256']} differs from the spec's {declared}; the spec was "
                         f"written against another archive.")
    vectors, probs, memberships, comp_records = [], [], [], []
    for i, (c, w) in enumerate(zip(comps, weights)):
        if str(c["source_experiment"]) != str(m.get("product_label")):
            raise ValueError(f"[[component]] {i}: source_experiment {c['source_experiment']!r} is not the archive's product "
                             f"{m.get('product_label')!r}.")
        if str(c["condition"]) not in [str(k) for k in m.get("condition_labels", [])]:
            raise ValueError(f"[[component]] {i}: condition {c['condition']!r} not in the archive's {m.get('condition_labels')}.")
        if str(c["condition"]) != str(target["condition"]):
            raise ValueError(f"[[component]] {i}: recordings of condition {c['condition']!r} cannot enter the native Nuisance DLI of "
                             f"{target['condition']!r}; the archive's application of this estimator to {m.get('condition_labels')} "
                             f"is a labeled inspection result, not a nuisance source of another condition.")
        stat = c.get("statistic", "sgm")
        policy = "uniform" if force_uniform else str(c.get("member_weighting", "uniform"))
        comp = archive_component(archive, kind=str(c["condition"]), window_set=str(c["window_set"]),
                                 construction=str(c["grouping"]), statistic=stat)
        n = comp["vectors"].shape[0]
        p_members, normalizers, scores = member_probabilities(policy, w, comp, where=f"[[component]] {i}",
                                                              rank_coordinates=None if force_uniform else c.get("rank_coordinates"),
                                                              rank_scope=None if force_uniform else c.get("rank_scope"))
        if abs(float(p_members.sum()) - float(w)) > 1e-9:
            raise ValueError(f"[[component]] {i}: member contributions sum to {p_members.sum()!r}, not the component mass {w!r}.")
        members = []
        for j in range(n):
            vectors.append(comp["vectors"][j]); probs.append(float(p_members[j])); memberships.append((i, j))
            members.append({**comp["identities"][j], "probability": float(p_members[j]),
                            "score_used": None if scores is None else float(scores[j]), "stored_row": None})   # stored_row set below
        within = p_members / float(w) if w > 0 else p_members
        comp_records.append({"index": i, "source_experiment": str(c["source_experiment"]), "condition": str(c["condition"]),
                             "window_set": str(c["window_set"]), "grouping": str(c["grouping"]), "statistic": stat,
                             "weight": float(w), "n_members": int(n), "member_weighting": policy,
                             "member_probability": float(w / n) if policy == "uniform" else None,
                             "normalizers": normalizers, "n_positive_members": int(np.sum(p_members > 0)), "n_zero_members": int(np.sum(p_members == 0)),
                             "probability_range": [float(p_members.min()), float(p_members.max())],
                             "effective_members": float(1.0 / np.sum(within ** 2)) if w > 0 and np.sum(within ** 2) > 0 else None,
                             "effective_members_scope": "1 / sum(p^2) over this component's members with their contributions normalized to the "
                                                        "component mass; a concentration measure, not a count of independent experiments",
                             "rank_scope": None if scores is None else normalizers.get("comparison_scope"),
                             "reference_population": None if scores is None else normalizers.get("reference_population"),
                             "score_source": None if scores is None else {"protocol": ((archive.get("compatibility") or {}).get("protocol") or {}).get("name"),
                                                                           "archive_format_version": int(archive.get("format_version", 1)),
                                                                           "archive_comparison_scope": comp.get("archive_comparison_scope"),
                                                                           "statistic_scored": "sgm", "note": "the scores describe the original SGM "
                                                                           "candidates; explicit overrides transform the stored rows afterwards"},
                             "rank_space": None if scores is None else normalizers.get("rank_space"),
                             "rank_coordinates": None if scores is None else list(normalizers.get("rank_coordinates") or []),
                             "rank_coordinates_overridden": None,       # filled once the overrides are read
                             "rank_coordinates_declared": None if scores is None else c.get("rank_coordinates"),
                             "window_set_chunks": m.get("window_sets", {}).get(str(c["window_set"])),
                             "members": members, "is_sampled_rows": stat == "sgm"})
    V = np.asarray(vectors, dtype=float); P = np.asarray(probs, dtype=float)
    uniq, inverse = np.unique(V, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    probabilities = np.zeros(uniq.shape[0]); groups = [[] for _ in range(uniq.shape[0])]
    for r, (p_, mem) in enumerate(zip(P, memberships)):
        probabilities[inverse[r]] += p_; groups[inverse[r]].append({"component": mem[0], "member": mem[1], "contribution": float(p_)})
        # the member's stored row, carried by identity through consolidation (an override may later make distinct rows equal)
        comp_records[mem[0]]["members"][mem[1]]["stored_row"] = int(inverse[r])
    probabilities = probabilities / probabilities.sum()
    consolidated = [{"row": int(u), "memberships": g, "probability": float(probabilities[u])} for u, g in enumerate(groups) if len(g) > 1]
    rank_weighted = any(cr["member_weighting"] != "uniform" for cr in comp_records)
    overrides, over = {}, (spec.get("override", {}) or {})
    for cr in comp_records:
        if cr["member_weighting"] != "uniform":
            cr["rank_coordinates_overridden"] = [k for k in (cr["rank_coordinates"] or []) if k in over]
    final = uniq.copy()
    for k, row in over.items():
        j = keys.index(k); e = table[j]
        v = float(row["value"])
        stored = float(np.log10(v)) if bool(e.get("LOG_FLAG")) else v      # the row's own coordinate (log10 or linear)
        overrides[k] = {"value_physical": v, "value_stored": stored, "reason": str(row["reason"]).strip(),
                        "replaced_range_stored": [float(final[:, j].min()), float(final[:, j].max())],
                        "replaced_rows": int(final.shape[0])}
        final[:, j] = stored
    # physical domain of every FINAL vector (archive rows and overrides alike; never a clipping)
    for j, e in enumerate(table):
        phys = np.power(10.0, final[:, j]) if bool(e.get("LOG_FLAG")) else final[:, j]
        for r in range(final.shape[0]):
            err = physical_domain_error(e["KEY"], phys[r], e)
            if err:
                raise ValueError(f"final vector {r}: {err} (physical units); the selection cannot be stored.")
    lo = np.asarray(prior_low, float); hi = np.asarray(prior_high, float)
    outside_rows = np.any((final < lo) | (final > hi), axis=1)
    record = {"kind": "representatives", "fixed": False, "input_units": "archive (estimator coordinates, log10)",
              "target": {k: target.get(k) for k in ("project_alias", "condition", "timing_label", "n_frames")},
              "archive": {"path": archive["path"], "sha256": archive["sha256"], "product_label": m.get("product_label"),
                          "target": bound, "condition_labels": m.get("condition_labels"), "written_at": m.get("written_at"),
                          "coverage": m.get("coverage"), "coverage_note": m.get("coverage_note")},
              "spec_path": None if spec_path is None else str(spec_path),
              "rationale": str(block.get("rationale")).strip(), "components": comp_records,
              "member_weightings": [cr["member_weighting"] for cr in comp_records],
              "normalization": ("member contribution = component weight / component member count (uniform) or the normalization of "
                                "Experiment's median compatibility ranks under the component's rank_scope (posterior_rank_median; "
                                "construction: weight * score / sum of the component's scores over its complete population, no mass reserved "
                                "per cell; within_cell, per_cell_window only: equal cell mass weight / n_cells, then score / sum of that cell's "
                                "selected-window scores); exact duplicates consolidated by adding contributions; a zero score is exactly zero "
                                "probability"),
              "n_member_rows": int(V.shape[0]), "n_unique_rows": int(uniq.shape[0]), "consolidated": consolidated,
              "n_zero_member_rows": int(np.sum(P == 0)), "n_zero_probability_rows": int(np.sum(probabilities == 0)),
              "effective_rows": float(1.0 / np.sum(probabilities ** 2)),
              "effective_rows_scope": "1 / sum(p^2) over the stored (consolidated) rows of the whole artifact; a concentration measure of the "
                                      "sampling probabilities, not a count of independent experiments and not a calibrated uncertainty",
              "overrides": overrides,
              "rank_spaces_used": sorted({cr["rank_space"] for cr in comp_records if cr["rank_space"]}),
              "rank_scopes_used": sorted({cr["rank_scope"] for cr in comp_records if cr["rank_scope"]}),
              # True when every rank-weighted component's coordinates exclude every overridden key (the weights then rest only on
              # coordinates the final vectors carry); False when an overridden coordinate entered the ranks; None without rank weighting
              "rank_coordinates_disjoint_from_overrides": (None if not rank_weighted else
                                                           all(not cr["rank_coordinates_overridden"] for cr in comp_records if cr["member_weighting"] != "uniform")),
              "is_sampled_rows": bool(all(c["is_sampled_rows"] for c in comp_records) and not overrides),
              "outside_prior_mass": float(probabilities[outside_rows].sum()),
              "outside_prior_by_key": {k: float(probabilities[(final[:, j] < lo[j]) | (final[:, j] > hi[j])].sum()) for j, k in enumerate(keys)},
              "joint_uncertainty": "none beyond the dispersion of the selected representatives; the input posteriors' uncertainty is not carried"}
    if rank_weighted and not force_uniform:
        # the equal-member baseline: the same selection, consolidation and overrides with uniform members, aligned row by row
        base_final, base_p, base_rec = representatives_from_spec(spec, archive, table, prior_low, prior_high, spec_path=spec_path,
                                                                 target=target, force_uniform=True)
        if base_final.shape != final.shape or not np.array_equal(base_final, final):
            raise ValueError("internal: the equal-member baseline does not align with the selected rows.")
        record["equal_member_baseline"] = {"probabilities": [float(v) for v in base_p], "effective_rows": base_rec["effective_rows"],
                                           "n_zero_probability_rows": base_rec["n_zero_probability_rows"],
                                           "definition": "the same components, component masses, duplicate consolidation and overrides with "
                                                         "member_weighting = uniform in every component; aligned to the stored rows"}
    return final, probabilities, record


# =============================================================================
# Load + validate the (user-finalized) spec
# =============================================================================
def load_spec(path, imaging_keys, prior_low, prior_high, table=None):
    """Parse and FULLY validate the user-authored spec.

    Validates: ``posterior_sample_pool_choice`` in ``POOL_CHOICES``; ``pool_mode`` in
    ``POOL_MODES``; ``box_quantiles`` (when ``box``); and — for ``box_user`` only — that
    every imaging key has a ``[low, high]`` with ``low < high`` lying inside the imaging
    prior box (``prior_low``/``prior_high``, log10, in ``imaging_keys`` order). The other
    choices read no per-parameter ranges. Raises ``ValueError`` with a precise message.
    """
    with open(path, "rb") as fh:
        spec = tomllib.load(fh)
    block = spec.get("block", {})
    choice = block.get("posterior_sample_pool_choice")
    if choice not in POOL_CHOICES:
        raise ValueError(f"{path}: [block].posterior_sample_pool_choice={choice!r} must be "
                         f"one of {POOL_CHOICES}.")
    pool_mode = block.get("pool_mode", "bounded")
    if pool_mode not in POOL_MODES:
        raise ValueError(f"{path}: [block].pool_mode={pool_mode!r} must be one of {POOL_MODES}.")
    if choice == "box":
        q = block.get("box_quantiles", list(DEFAULT_BOX_QUANTILES))
        if (not isinstance(q, (list, tuple)) or len(q) != 2
                or not 0.0 <= float(q[0]) < float(q[1]) <= 1.0):
            raise ValueError(f"{path}: [block].box_quantiles must be [low, high] with "
                             f"0 <= low < high <= 1; got {q!r}.")
    if choice == "box_user":
        imaging = spec.get("imaging", {})
        missing = [k for k in imaging_keys if k not in imaging]
        if missing:
            raise ValueError(f"{path}: choice 'box_user' needs an [imaging.<KEY>] range for "
                             f"every parameter; missing {missing}.")
        bl = {k: float(prior_low[i]) for i, k in enumerate(imaging_keys)}
        bh = {k: float(prior_high[i]) for i, k in enumerate(imaging_keys)}
        tol = 1e-9
        for k in imaging_keys:
            row = imaging[k]
            lo, hi = row.get("low"), row.get("high")
            if lo is None or hi is None:
                raise ValueError(f"{path}: [imaging.{k}] needs `low` and `high` (log10).")
            lo, hi = float(lo), float(hi)
            if not lo < hi:
                raise ValueError(f"{path}: [imaging.{k}] low {lo} must be < high {hi}.")
            if lo < bl[k] - tol or hi > bh[k] + tol:
                raise ValueError(
                    f"{path}: [imaging.{k}] range [{lo}, {hi}] exceeds the imaging prior box "
                    f"[{bl[k]}, {bh[k]}] (log10); a Nuisance_DLI range must lie within it.")
    if choice == "selection_user":
        if table is None:
            raise ValueError(f"{path}: choice 'selection_user' needs the parameter table (pass table=).")
        try:
            selection_from_spec(spec, table, prior_low, prior_high)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if choice == "empirical_representatives":
        try:
            _validate_representatives_block(spec, imaging_keys, str(path), table=table)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
    if choice == "sgm_percentiles":
        pct = block.get("percentiles", list(SGM_DEFAULT_PERCENTILES))
        if (not isinstance(pct, (list, tuple)) or len(pct) == 0
                or not all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 100.0 for v in pct)):
            raise ValueError(f"{path}: [block].percentiles must be a non-empty list of numbers in "
                             f"[0, 100]; got {pct!r}.")
        cond = block.get("condition", None)      # None = the artifact's own condition
        if cond is not None and cond not in SGM_CONDITIONS:
            raise ValueError(f"{path}: [block].condition={cond!r} must be one of {SGM_CONDITIONS}.")
        src = block.get("selection_source", "experiment")
        if src not in SGM_SELECTION_SOURCES:
            raise ValueError(f"{path}: [block].selection_source={src!r} must be one of "
                             f"{SGM_SELECTION_SOURCES}.")
    return spec


# =============================================================================
# Build the Nuisance_DLI from the finalized spec (+ pool, for the derived choices)
# =============================================================================
def build_nuisance_dli(spec, imaging_keys, prior_low, prior_high, *, pool=None, table=None, archive=None,
                       spec_path=None, target=None):
    """Construct the `NuisanceDLI` for the spec's choice, from an already-computed pool.

    CPU-only: the expensive pool is built and cached by the caller (see the pool cache and
    `build_posterior_sample_pool` / `build_map_estimate_pool`), so switching among choices
    that share a pool costs nothing here. ``box_user`` and ``selection_user`` need no pool;
    ``empirical_representatives`` needs the loaded candidate ``archive`` (and ``table``) and the explicit
    ``target`` identity (condition, timing label) the artifact is built for, checked against the archive;
    every other choice requires the matching ``pool`` (the posterior-sample pool for
    raw/gaussian/box, the MAP pool for map_estimate_pool).
    """
    block = spec["block"]
    choice = block["posterior_sample_pool_choice"]
    pool_mode = block.get("pool_mode", "bounded")

    if choice == "selection_user":
        if table is None:
            raise ValueError("choice 'selection_user' needs the parameter table (pass table=).")
        vec, record = selection_from_spec(spec, table, prior_low, prior_high)
        # pool_mode records whether the vector sits inside the prior box (bounded) or was
        # acknowledged outside it (unrestricted); it governs no sampling here.
        mode = "unrestricted" if record["outside_prior"] else "bounded"
        return NuisanceDLI.from_selection(imaging_keys, vec, record, prior_low=prior_low,
                                          prior_high=prior_high, pool_mode=mode)

    if choice == "empirical_representatives":
        if table is None or archive is None:
            raise ValueError("choice 'empirical_representatives' needs the parameter table (table=) and the loaded "
                             "candidate archive (archive=, from experiment_aggregation.load_candidate_archive).")
        vectors, probabilities, record = representatives_from_spec(spec, archive, table, prior_low, prior_high,
                                                                   spec_path=spec_path, target=target)
        mode = "unrestricted" if record["outside_prior_mass"] > 0 else "bounded"
        return NuisanceDLI.from_representatives(imaging_keys, vectors, probabilities, record, prior_low=prior_low,
                                                prior_high=prior_high, pool_mode=mode)

    if choice == "box_user":
        low = [float(spec["imaging"][k]["low"]) for k in imaging_keys]
        high = [float(spec["imaging"][k]["high"]) for k in imaging_keys]
        return NuisanceDLI.from_box(imaging_keys, low, high, choice="box_user",
                                    pool_mode=pool_mode, prior_low=prior_low,
                                    prior_high=prior_high)

    if pool is None:
        raise ValueError(f"choice {choice!r} needs its pool (pass pool=...); only 'box_user' "
                         "builds without one.")

    if choice in ("raw", "map_estimate_pool", "sgm_percentiles"):
        # sgm_percentiles: `pool` is the already-selected whole vectors (the SGM at p50, plus any
        # signed-distance percentile members) -- the caller ran the selection; here it is just stored.
        return NuisanceDLI.from_samples(imaging_keys, pool, choice=choice, pool_mode=pool_mode,
                                        prior_low=prior_low, prior_high=prior_high)
    if choice == "gaussian":
        mean, cov = fit_gaussian(pool)
        return NuisanceDLI.from_gaussian(imaging_keys, mean, cov, pool_mode=pool_mode,
                                         prior_low=prior_low, prior_high=prior_high)
    # choice == "box"
    q = block.get("box_quantiles", list(DEFAULT_BOX_QUANTILES))
    low, high = quantile_box(pool, q, prior_low, prior_high)
    return NuisanceDLI.from_box(imaging_keys, low, high, choice="box", pool_mode=pool_mode,
                                prior_low=prior_low, prior_high=prior_high)


# =============================================================================
# The enforcement gate — downstream generation LOADS the built artifact
# =============================================================================
def require_nuisance_dli(posit_dir, project_alias, timing_label, nuisance_tag=None):
    """LOAD the built Nuisance_DLI artifact, or FAIL CLEARLY (naming the analysis).

    ``nuisance_tag`` selects a tagged artifact (a reference vector or a sensitivity variant)
    beside the canonical one; the consumer records the selected artifact's identity.

    Generation consumes the persisted, self-contained artifact; it does not rebuild
    (rebuilding needs the estimator and a GPU — the analysis step's job). If the artifact is
    absent, the analysis has not been run: this raises a clear, actionable error rather than
    fabricating a nuisance. The `Nuisance_DLI` is a user decision, never an automatic output.
    """
    ap = artifact_path(posit_dir, project_alias, timing_label, nuisance_tag)
    if not ap.exists():
        sp = spec_path(posit_dir, project_alias, timing_label, nuisance_tag)
        raise FileNotFoundError(
            f"Nuisance_DLI artifact not found:\n    {ap}\n\n"
            f"The Nuisance_DLI is a user decision built by the analysis step, not an "
            f"automatic output. Run it to inspect the calibrated imaging, emit the spec, "
            f"edit it, and build:\n"
            f"    python {_ANALYSIS} --emit-template --total-time-seconds <T>\n"
            f"    (edit {sp.name}: set posterior_sample_pool_choice and, for box_user, the ranges)\n"
            f"    python {_ANALYSIS} --build --total-time-seconds <T>\n\n"
            f"See DETECTOR_WORKFLOW.md, \"Nuisance and artifact design\".")
    return NuisanceDLI.load(ap)
