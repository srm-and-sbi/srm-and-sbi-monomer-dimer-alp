"""Point-estimate machinery for the Evaluation and Experiment stages.

Every product of those stages carries THREE point estimates for each observation, defined once in
:data:`POINT_ESTIMATES` below: the numerical MAP candidate (``map_estimate``), the marginal median
of the observation's draws (the 0.50 level of ``posterior_quantiles``) and the SGM of the same draws
(``posterior_sgm``). Evaluation compares each against the known ground-truth theta of the held-out
EVAL videos; Experiment reports each per experimental condition. No one of them is produced or read
without the other two.

The MAP candidate is a **seed-then-optimize** procedure (one per observation):

    1. collect_theta_prex  -- draw a pool of candidate theta from the posterior
                              conditioned on the video (respects prior bounds).
    2. collect_score_prex  -- score every candidate by the flow's log-probability.
    3. extract_elite_prex  -- keep the top-K candidates as optimization seeds.
    4. pool_scale          -- the pool's per-coordinate median and interquartile
                              range: the units the ascent steps in.
    5. optimize_elite      -- gradient-ascent the log-probability from those seeds
                              (Adam + ReduceLROnPlateau + early stopping, in
                              pool-IQR units); return the best (score, theta) seen
                              and the reason the ascent stopped.

All theta live in ESTIMATOR coordinates (the flow's space and the prior's space): log10 of the
physical value for a log row, the value itself for a linear row (the two fractions of the
biology table). The ground-truth theta sets are stored physical, so the report compares
``to_flow(theta_true)`` against the inferred coordinates. Every per-row statistic carries its
unit: dex for a log row (an error of 0.3 dex is a factor of two), percentage points (pp) for a
linear row (an absolute error of 0.01 is one pp); a linear row has no multiplicative band and no
bias factor. The pool sampler uses the bounded ``DirectPosterior`` (rejection sampling between
prior ranges); the gradient steps use ``posterior.posterior_estimator`` (the flow, which exposes
``log_prob`` and tracks gradients but does not enforce bounds).

Module contents:
    POINT_ESTIMATES, QUANTILE_LEVELS, draw_label(...), point_estimate_rows(...), optimizer_contract(...),
    optimizer_summary_lines(...) -- the three estimates' definitions and the recorded contract.
    map_estimate(...)         -- full seed-then-optimize MAP candidate for one observation.
    posterior_summary(...)    -- the quantiles (median at 0.50) and the SGM of one observation's draws.
    sample_geometric_median(...) -- the medoid under a per-coordinate scaling.
    recovery_stats(...)       -- per-parameter recovery error statistics of any point estimate.
    recovery_table(...)       -- (headers, rows) recovery summary for the report.
    point_estimates_compared_table(...), point_estimate_agreement_table(...)
                              -- the three estimates side by side.
    fraction_flag_rows(...)   -- the linear rows' values outside [0, 1], counted, never changed.
    starting_composition_recovery(...), starting_composition_tables(...)
                              -- recovery of the window-start allocation and of c per true-capacity stratum.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .artifact_schema import DRAW_LABELS
from .inference_support import normalize_video
from .parameterization import (PARAMETERS, entry_to_physical, is_log_row, parameter_keys_of,
                               population_from_theta, population_valid, to_physical)
# band_label lives in the temporal-dynamics kernel (pure numpy, no machine profile) so the
# recovery tolerances render identically wherever they are reported -- one definition, not two.
from .temporal_dynamics import band_label


# =============================================================================
# Per-row units: dex for a log row, percentage points for a linear row
# =============================================================================

#: Multiplier from an estimator-coordinate error to the reported unit: 1 for a log row (dex),
#: 100 for a linear row (percentage points).
_PP = 100.0


def row_error_unit(entry: dict) -> str:
    """The unit of an error on this row: ``'dex'`` for a log row, ``'pp'`` (percentage points)
    for a linear row."""
    return "dex" if is_log_row(entry) else "pp"


def row_coordinate_unit(entry: dict) -> str:
    """The unit of this row's estimator coordinate: ``'log10'`` for a log row, ``'fraction'``
    for a linear row (the value itself)."""
    return "log10" if is_log_row(entry) else "fraction"


def _error_scale(entry: dict) -> float:
    """Factor turning an estimator-coordinate difference into the row's reported error unit."""
    return 1.0 if is_log_row(entry) else _PP


# =============================================================================
# The three point estimates: one vocabulary, one definition each
# =============================================================================

#: The five quantile levels every product stores along the last axis of ``posterior_quantiles``.
#: The marginal median is the level 0.50; readers locate it by level from the product's manifest
#: (``artifact_schema.median_level_index``), never by an assumed position.
QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)

#: Bumped when a definition below changes meaning (not wording). Distinct from the package
#: version and from the artifact schema version. 2 (0.1.17): the MAP ascent steps in units of the
#: candidate pool's interquartile range and retains every strictly better finite pair; products
#: made under 1 carry MAP estimates from the unscaled optimizer and are refused on read.
ESTIMATE_DEFINITIONS_VERSION = 2

#: The three point estimates, keyed by the stable machine name used in code, table notes,
#: captions and documentation. Each definition states the operator, the source population, the
#: observation-level grouping, the coordinate space and the distance scaling (where one applies),
#: and the stored field that carries it. Configurable quantities -- candidate-pool size, elite
#: count, step budget, draw count -- are deliberately absent from the text: they live in the
#: product's manifest, where the values actually used are recorded.
POINT_ESTIMATES = {
    "map": {
        "label": "numerical MAP candidate",
        "short": "MAP",
        "stored_field": "map_estimate",
        "operator": "the highest-scoring point retained by the configured gradient-ascent "
                    "optimizer of the flow's log-density, initialized from the top-scoring "
                    "candidate draws and stepping each coordinate in units of its interquartile "
                    "range over all the candidate draws",
        "population": "the flow's conditional density for one observation; the candidate draws "
                      "that seed the ascent are that observation's own draws",
        "grouping": "one vector per observation",
        "coordinates": "estimator coordinates (log10 for a log row, the value itself for a linear row)",
        "scaling": None,
        "caveats": "the optimization is unconstrained: prior support and convergence to a mode "
                   "are not guaranteed, so a value outside the prior box is a flow optimum, not "
                   "a MAP of the prior-supported posterior",
    },
    "median": {
        "label": "marginal median of draws",
        "short": "median",
        "stored_field": "posterior_quantiles",
        "stored_slice": "the quantile at level 0.50",
        "operator": "the 0.50 quantile of each coordinate (linear interpolation between order "
                    "statistics), taken independently per coordinate",
        "population": "one observation's summary draws (posterior draws under the bounded pool, "
                      "flow draws under the unrestricted one)",
        "grouping": "one vector per observation",
        "coordinates": "estimator coordinates (log10 for a log row, the value itself for a linear row)",
        "scaling": None,
        "caveats": "a coordinate-wise composite: the resulting vector is not necessarily a "
                   "sampled vector and need not be a probable point of the joint. A physical "
                   "value is the transform of this quantile",
    },
    "sgm": {
        "label": "SGM of draws",
        "short": "SGM",
        "stored_field": "posterior_sgm",
        "operator": "the sample geometric median: the complete draw minimizing the summed "
                    "Euclidean distance to all other draws (an exact sample medoid)",
        "population": "the same summary draws as the marginal median, for the same observation",
        "grouping": "one vector per observation",
        "coordinates": "estimator coordinates (log10 for a log row, the value itself for a linear row)",
        "scaling": "each coordinate divided by its prior width before the distance is taken",
        "caveats": "a realized draw, so its coordinates co-occurred; selecting a member avoids a "
                   "coordinate-wise composite but does not by itself preserve the "
                   "distribution's correlations. Distinct from an SGM of MAP vectors, from a "
                   "pooled SGM of per-window SGMs, and from an SGM in physical coordinates -- "
                   "each of those names its own population and scaling where it is reported",
    },
}

POINT_ESTIMATE_KEYS = tuple(POINT_ESTIMATES)          # ("map", "median", "sgm")


def short_labels() -> dict:
    """Machine key -> the short display token used in table headers, figure legends and dict
    keys handed to the figure helpers (``{"map": "MAP", "median": "median", "sgm": "SGM"}``).
    Display tokens never travel back into stored field names."""
    return {k: v["short"] for k, v in POINT_ESTIMATES.items()}


def draw_label(pool_mode: str) -> str:
    """What the summary draws are under each candidate-pool mode: rejection sampling inside the
    prior yields posterior draws; direct flow sampling yields flow draws that may lie outside the
    prior's support. The stored field names are the same in both cases; the manifest and this
    label carry the meaning. The table is :data:`artifact_schema.DRAW_LABELS`, which the schema
    validator also enforces (a product's label must be the entry for its pool mode)."""
    try:
        return DRAW_LABELS[pool_mode]
    except KeyError:
        raise ValueError(f"unknown pool_mode {pool_mode!r} (bounded|unrestricted)") from None


def point_estimate_note(pool_mode: str) -> str:
    """One sentence per estimate for a report note, with the draw label resolved for this run."""
    d = draw_label(pool_mode)
    m, q, s = POINT_ESTIMATES["map"], POINT_ESTIMATES["median"], POINT_ESTIMATES["sgm"]
    return (f"'{m['short']}' = {m['label']}: {m['operator']}, in {m['coordinates']}; {m['caveats']}. "
            f"'{q['short']}' = {q['label']}: {q['operator']} over the observation's {d}s, in "
            f"{q['coordinates']}; {q['caveats']}. "
            f"'{s['short']}' = {s['label']}: {s['operator']} over the same {d}s, {s['scaling']}, in "
            f"{s['coordinates']}; a realized draw. The three are read together; no one of them "
            f"replaces the others.")


def point_estimate_rows(pool_mode: str, median_index: int) -> list:
    """Rows of the "Point estimates: definitions" table every report opens with -- built from
    :data:`POINT_ESTIMATES` directly (key token, stored field, definition), with the draw label
    resolved for this run and the median's validated level index shown."""
    d = draw_label(pool_mode)
    m, q, s = POINT_ESTIMATES["map"], POINT_ESTIMATES["median"], POINT_ESTIMATES["sgm"]
    cap = lambda text: text[0].upper() + text[1:]
    return [
        [m["short"], m["stored_field"],
         f"{m['label']}: {m['operator']}, in {m['coordinates']}. {cap(m['caveats'])}."],
        [q["short"], f"{q['stored_field']} ({q['stored_slice']}, index {median_index})",
         f"{q['label']}: {q['operator']} over the observation's {d}s, in {q['coordinates']}. "
         f"{cap(q['caveats'])}."],
        [s["short"], s["stored_field"],
         f"{s['label']}: {s['operator']} over the same {d}s, {s['scaling']}, in "
         f"{s['coordinates']}. {cap(s['caveats'])}."],
    ]


#: How the MAP ascent measures a step, as every product manifest records it.
STEP_COORDINATES = ("pool-IQR units: Adam moves u = (theta - m) / IQR per coordinate, m and IQR "
                    "the median and interquartile range of the observation's candidate pool; the "
                    "learning rates are in u; a zero or non-finite IQR skips the ascent and returns "
                    "the best candidate, reported")

#: Why an ascent ended: patience exhausted, step budget spent, a non-finite score, or a pool whose
#: IQR could not set the step (no ascent; the best candidate is returned).
STOP_REASONS = ("early", "budget", "non-finite", "invalid-scale")


def optimizer_contract(eval_cfg, *, learning_rate, tolerance, theta_prex_size, elite_prex_size,
                       numb_steps, pool_mode) -> dict:
    """The MAP optimizer settings a product manifest (and the MapEstimate pool cache) records: the
    values actually used, the units the ascent steps in, and the bookkeeping rule -- kept distinct
    from the artifact schema version, since either may change without the other."""
    return {
        "pool_mode": str(pool_mode), "theta_prex_size": int(theta_prex_size),
        "elite_prex_size": int(elite_prex_size), "numb_steps": int(numb_steps),
        "optimizer_patience": int(eval_cfg.optimizer_patience),
        "scheduler_patience": int(eval_cfg.scheduler_patience),
        "learning_rate": float(learning_rate),
        "learning_rate_minimum": float(eval_cfg.learning_rate_minimum),
        "learning_rate_factor": float(eval_cfg.learning_rate_factor),
        "tolerance": float(tolerance),
        "step_coordinates": STEP_COORDINATES,
        # The returned score is the density AT the returned vector: the pair is recorded before
        # the update (0.1.15), and every strictly better finite pair is kept (0.1.17).
        "bookkeeping": ("every strictly better finite (score, vector) pair retained before "
                        "optimizer.step; tolerance gates only the stopping patience and the "
                        "plateau scheduler"),
    }


def optimizer_summary_lines(contract: dict) -> list:
    """The effective MAP-optimizer settings of ``contract`` (:func:`optimizer_contract`), one
    banner line each, so a stage prints exactly the values its manifest records."""
    c = contract
    return [
        f"theta_prex_size      : {c['theta_prex_size']}   (elite seeds {c['elite_prex_size']})",
        f"numb_steps           : {c['numb_steps']}   (optimizer_patience {c['optimizer_patience']}, "
        f"scheduler_patience {c['scheduler_patience']})",
        f"learning_rate        : {c['learning_rate']:.3e}   (minimum "
        f"{c['learning_rate_minimum']:.3e}, factor {c['learning_rate_factor']}; pool-IQR units)",
        f"tolerance            : {c['tolerance']:.3e} nats   (a meaningful improvement: stopping "
        f"patience and scheduler only)",
    ]


# =============================================================================
# Seed-then-optimize MAP estimate (one video)
# =============================================================================

def _empty_cache(device: torch.device) -> None:
    """Release cached CUDA memory between batches (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _theta_repr(theta_log10: np.ndarray, arc: int = 3) -> str:
    """Compact ``[v0, v1, ...]`` repr of a theta vector, rounded to ``arc`` places."""
    return str([round(float(v), arc) for v in np.asarray(theta_log10).ravel()])


def collect_theta_prex(posterior, flow, vista_device: torch.device,
                       cond: torch.Tensor, theta_prex_size: int,
                       theta_prex_batch_size: int, pool_mode: str = "bounded",
                       show: bool = False, verbose: bool = False) -> torch.Tensor:
    """Draw a pool of candidate theta conditioned on ``cond``, in one of two modes.

    ``pool_mode``:
        ``"bounded"``      -- ``DirectPosterior.sample`` rejection-samples within
                              the prior ranges, so every candidate is a valid
                              prior draw. Correct for a well-trained posterior;
                              can stall if the posterior's mass lies mostly
                              outside the prior box (rejection never accepts).
        ``"unrestricted"`` -- sample the flow (``posterior_estimator``) directly,
                              with no prior-range rejection. Never stalls, so it
                              suits smoke tests and landscape exploration on an
                              undertrained posterior (candidates may fall outside
                              the prior box; the gradient ascent explores freely).

    Each outer sampler call draws up to ``theta_prex_batch_size`` candidates, offloaded to
    ``vista_device`` (typically CPU) until ``theta_prex_size`` are collected. The embedding network
    runs on the one video a few times per outer call, not once per draw: twice in the measured
    bounded case, more when rejection sampling needs further internal batches. On the detector
    checkpoint of record and a 4 GB Quadro T2000, peak GPU memory was approximately constant over
    the tested range, up to 50,000 draws. Sampling runs under ``torch.no_grad()``, whose scope ends
    with this function, so the MAP ascent keeps its gradients: the draws are data, and without it
    each unrestricted call kept its autograd graph (about 1.5 GiB for a 2 s video) alive for as
    long as the draws lived.

    Returns:
        Tensor of shape ``(theta_prex_size, D)`` on ``vista_device``.
    """
    if pool_mode not in ("bounded", "unrestricted"):
        raise ValueError(
            f"pool_mode={pool_mode!r}; must be 'bounded' or 'unrestricted'.")
    if theta_prex_size < 1 or theta_prex_batch_size < 1:
        raise ValueError(
            f"theta_prex_size={theta_prex_size} and theta_prex_batch_size={theta_prex_batch_size} "
            f"must both be at least 1 (a zero batch would never finish).")
    theta_set = []
    quota = 0
    t0 = time.time()
    with torch.no_grad():
        while quota < theta_prex_size:
            batch_size = min(theta_prex_batch_size, theta_prex_size - quota)
            if pool_mode == "unrestricted":
                # flow.sample -> (batch_size, 1, D); drop the singleton condition dim.
                theta = flow.sample((batch_size,), condition=cond).squeeze(1)
            else:
                theta = posterior.sample(sample_shape=(batch_size,), x=cond,
                                         show_progress_bars=verbose)
            theta_set.append(theta.to(vista_device))
            quota += batch_size
            _empty_cache(cond.device)
    theta_prex = torch.cat(theta_set, dim=0)
    if show:
        print(f"  [pool ] ({pool_mode}) candidate theta predictive samples: "
              f"shape={tuple(theta_prex.shape)}  ({time.time() - t0:.3f}s)",
              flush=True)
    return theta_prex


def collect_score_prex(flow, train_device: torch.device, vista_device: torch.device,
                       cond: torch.Tensor, theta_prex: torch.Tensor,
                       score_prex_batch_size: int, show: bool = False) -> torch.Tensor:
    """Score each candidate theta by the flow's log-probability given ``cond``.

    Returns:
        1D tensor of shape ``(theta_prex_size,)`` on ``vista_device``: the
        log-probability of each candidate under the posterior flow.
    """
    score_set = []
    quota = theta_prex.size(0)
    t0 = time.time()
    for index in range(0, quota, score_prex_batch_size):
        theta = theta_prex[index:index + score_prex_batch_size].to(train_device)
        with torch.no_grad():
            theta_batch = theta.unsqueeze(0)                       # (1, B, D)
            cond_batch = cond.squeeze(0).expand(theta.size(0), *cond.shape[1:])
            score_batch = flow.log_prob(input=theta_batch, condition=cond_batch)
            score = score_batch.squeeze(0)                         # (B,)
        score_set.append(score.to(vista_device))
        del theta, theta_batch, cond_batch, score_batch, score
        _empty_cache(train_device)
    score_prex = torch.cat(score_set, dim=0)
    if show:
        print(f"  [score] flow log-probability of candidates: "
              f"shape={tuple(score_prex.shape)}  ({time.time() - t0:.3f}s)",
              flush=True)
    return score_prex


def extract_elite_prex(theta_prex: torch.Tensor, score_prex: torch.Tensor,
                       elite_prex_size: int, show: bool = False) -> torch.Tensor:
    """Keep the top-``elite_prex_size`` candidates by log-probability as seeds."""
    indices = torch.topk(score_prex, elite_prex_size).indices
    elite_prex = theta_prex[indices]                               # (K, D)
    if show:
        print(f"  [elite] optimization seeds (top-{elite_prex_size}): "
              f"shape={tuple(elite_prex.shape)}", flush=True)
    return elite_prex


def pool_scale(theta_prex) -> tuple:
    """``(center, iqr)`` of one observation's candidate pool ``(n, D)``, per coordinate, in float64:
    the median and the interquartile range (numpy's default linear interpolation). The MAP ascent
    moves ``u = (theta - center) / iqr``, so one learning rate is the same fraction of every
    parameter's spread in that pool. Returned as computed: a zero or non-finite IQR is the caller's
    to report, never to divide through (:func:`map_estimate`)."""
    pool = theta_prex.detach().cpu().numpy() if torch.is_tensor(theta_prex) else theta_prex
    q25, q50, q75 = np.quantile(np.asarray(pool, dtype=np.float64), (0.25, 0.50, 0.75), axis=0)
    return q50, q75 - q25


def _best_candidate(theta_prex: torch.Tensor, score_prex: torch.Tensor) -> tuple:
    """The highest-scoring finite candidate as ``(score, theta, info)``, returned unoptimized for a
    pool whose IQR cannot set the ascent's units. Its score is the density at it."""
    finite = torch.isfinite(score_prex)
    if not bool(finite.any()):
        raise RuntimeError("no candidate of the pool has a finite score; the candidate pool or the "
                           "flow is broken for this observation.")
    best = int(torch.where(finite, score_prex, torch.full_like(score_prex, float("-inf"))).argmax())
    score = float(score_prex[best])
    info = {"stop": "invalid-scale", "steps": 0, "seed_score": score,
            "learning_rate_final": float("nan")}
    return score, theta_prex[best].detach().cpu().numpy(), info


def optimize_elite(flow, train_device: torch.device, vista_device: torch.device,
                   cond: torch.Tensor, elite_prex: torch.Tensor,
                   numb_steps: int, optimizer_patience: int,
                   scheduler_patience: int, show_progress_steps: int,
                   learning_rate_minimum: float, learning_rate_factor: float,
                   learning_rate: float, tolerance: float,
                   show: bool = False, verbose: bool = False, log_fn=None, *,
                   center, scale) -> tuple:
    """Gradient-ascent the flow log-probability from the elite seeds, in pool-IQR units.

    Adam moves ``u = (theta - center) / scale`` for all ``K`` seeds at once, ``center`` and
    ``scale`` being the candidate pool's per-coordinate median and interquartile range
    (:func:`pool_scale`; ``scale`` must be finite and positive). ``learning_rate`` and
    ``learning_rate_minimum`` are therefore fractions of each parameter's spread, while the
    density is still evaluated, and maximized, in theta. Adam is elementwise and the loss is the
    seeds' mean, so each seed follows its own gradient; the learning rate is reduced when the mean
    score plateaus. The ascent stops early after ``optimizer_patience`` steps without a meaningful
    improvement -- the best score rising more than ``tolerance`` (nats) above its value at the last
    such improvement -- and so can halt before ``numb_steps``.

    Bookkeeping runs each step BEFORE the parameters move. Every strictly better finite
    ``(score, theta)`` is retained, however small the gain; ``tolerance`` gates only the patience
    count and the scheduler. So the returned score is the density at the returned vector and,
    because step 1 scores the seeds themselves, never worse than the best seed's. A non-finite
    score at any seed ends the ascent, keeping the best finite pair.

    The optimization itself is unconstrained: ``pool_mode`` bounds the candidate
    pool, not the gradient steps, so a returned vector may lie outside the prior
    box under either pool mode. It is then a flow optimum, not a MAP of the
    prior-supported posterior, and the callers' 'outside prior' columns count it.

    ``log_fn``, if given, is called with each per-step progress line (at the
    ``show_progress_steps`` cadence) and the final stop line, so a caller can
    stream them to a file live (e.g. the debug progress log). Console printing is
    independent and gated by ``show``.

    Returns:
        ``(optimal_score, optimal_theta, info)``: the float log-probability, the 1D numpy vector
        (estimator coordinates) on ``vista_device``, and a dict with ``stop`` (``"early"``,
        ``"budget"`` or ``"non-finite"``), ``steps`` (density evaluations made), ``seed_score``
        (the best seed's score) and ``learning_rate_final``.
    """
    dtype = elite_prex.dtype
    c = torch.as_tensor(np.asarray(center), dtype=dtype, device=train_device)
    s = torch.as_tensor(np.asarray(scale), dtype=dtype, device=train_device)
    if not bool(torch.all(torch.isfinite(s)) and torch.all(s > 0)):
        raise ValueError(f"scale must be finite and positive; got {np.asarray(scale).tolist()}.")
    u = ((elite_prex.to(train_device) - c) / s).clone().detach()
    u.requires_grad_(True)
    optimizer = torch.optim.Adam(params=[u], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=learning_rate_factor,
        patience=scheduler_patience, threshold=tolerance,
        threshold_mode="abs", min_lr=learning_rate_minimum,
    )

    optimal_score = float("-inf")
    optimal_theta = None
    seed_score = float("nan")
    reference = float("-inf")      # the best score at the last meaningful improvement
    steps_without_improve = 0
    stop = "budget"                # "budget" (ran every step) | "early" (patience) | "non-finite"

    if show:
        print(f"  [optim] gradient-ascent: up to {numb_steps} steps, "
              f"optimizer-patience {optimizer_patience}", flush=True)
    if verbose:
        print(f"  [optim] scheduler-patience={scheduler_patience}, "
              f"lr={learning_rate:.3e} (min {learning_rate_minimum:.3e}, "
              f"factor {learning_rate_factor}; pool-IQR units), tolerance={tolerance:.3e} nats",
              flush=True)
        print(f"          pool IQR [estimator space] {_theta_repr(np.asarray(scale), 4)}", flush=True)
    t0 = time.time()
    step = 0
    for step in range(1, numb_steps + 1):
        optimizer.zero_grad()
        theta = c + s * u                                          # (K, D), in the graph
        theta_batch = theta.unsqueeze(0)                           # (1, K, D)
        cond_batch = cond.squeeze(0).expand(theta.size(0), *cond.shape[1:])
        score_batch = flow.log_prob(input=theta_batch, condition=cond_batch)
        score = score_batch.squeeze(0)                             # (K,)
        # Bookkeeping BEFORE the parameters move: `optimizer.step()` updates `u` in place, and a
        # theta read after it would pair this step's score with the next step's coordinates. The
        # retention is strict (any finite gain counts); the patience count and the scheduler use
        # `tolerance`, so a long run of tiny gains is kept without resetting the patience.
        with torch.no_grad():
            finite = torch.isfinite(score)
            if bool(finite.any()):
                masked = torch.where(finite, score, torch.full_like(score, float("-inf")))
                value, idx = masked.max(dim=0)
                value = value.item()
                if step == 1:
                    seed_score = value
                if value > optimal_score:
                    optimal_score = value
                    optimal_theta = theta[idx].detach().clone()
                if value > reference + tolerance:
                    reference = value
                    steps_without_improve = 0
                else:
                    steps_without_improve += 1
        if not bool(finite.all()):
            stop = "non-finite"
            break
        if steps_without_improve >= optimizer_patience:
            stop = "early"
            break
        loss = -score.mean()        # mean keeps the lr consistent across K seeds
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())   # scheduler only reads the value; detach avoids the requires_grad->scalar warning
        del theta, theta_batch, cond_batch, score_batch, score, loss
        _empty_cache(train_device)
        if (show or log_fn is not None) and (step % show_progress_steps == 0 or step == 1):
            dynamic_threshold = scheduler.best - scheduler.threshold
            progress_msg = (f"progress [{step}/{numb_steps}]  "
                            f"lr={scheduler.get_last_lr()[0]:.2e}  "
                            f"dynamic_threshold={dynamic_threshold:.3f}  "
                            f"optimal_log_prob={optimal_score:.3f}")
            if show:
                print(f"    {progress_msg}", flush=True)
            if log_fn is not None:
                log_fn(progress_msg)

    if optimal_theta is None:
        raise RuntimeError("the MAP ascent found no finite score at its seeds; the candidate pool "
                           "or the flow is broken for this observation.")
    optimal_np = optimal_theta.to(vista_device).numpy()
    info = {"stop": stop, "steps": step, "seed_score": seed_score,
            "learning_rate_final": float(optimizer.param_groups[0]["lr"])}
    stop_msg = (f"{stop} stop at step {step} ({time.time() - t0:.3f}s)  "
                f"optimal_log_prob={optimal_score:.3f}  "
                f"gain over the best seed={optimal_score - seed_score:.4f}")
    if show:
        print(f"  [optim] {stop_msg}", flush=True)
        # Estimator-space coordinates only: this function has no parameter table, and the
        # per-row scale (a linear initial dimer fraction beside log rows) forbids a blanket
        # 10**theta; callers print physical values through parameterization.to_physical.
        print(f"          optimal theta [estimator space] {_theta_repr(optimal_np)}", flush=True)
    if log_fn is not None:
        log_fn(stop_msg)
    return optimal_score, optimal_np, info


def map_estimate(posterior, video_chunk: np.ndarray,
                 train_device: torch.device, vista_device: torch.device,
                 theta_prex_size: int, theta_prex_batch_size: int,
                 score_prex_batch_size: int, elite_prex_size: int,
                 numb_steps: int, optimizer_patience: int,
                 scheduler_patience: int, show_progress_steps: int,
                 learning_rate_minimum: float, learning_rate_factor: float,
                 learning_rate: float, tolerance: float,
                 pool_mode: str = "bounded", show: bool = False,
                 verbose: bool = False, log_fn=None, return_info: bool = False) -> tuple:
    """Estimate the MAP theta for one video via seed-then-optimize.

    Args:
        posterior: a trained ``DirectPosterior`` (its ``posterior_estimator`` is
            the gradient-enabled flow used for scoring and optimization).
        video_chunk: a single raw video array ``(n_frames, H, W)``; normalized to
            ``[0, 1]`` internally to match the training-time input.
        train_device / vista_device: compute device and CPU offload device.
        pool_mode: ``"bounded"`` (rejection-sample the candidate pool within the
            prior) or ``"unrestricted"`` (sample the flow directly); see
            ``collect_theta_prex``.
        show: when True, print the per-stage diagnostics (pool/score/
            elite shapes and timings, per-step optimization progress, the stopping
            reason, and the optimal theta in estimator coordinates).
        verbose: deeper level (implies show at the call site): enables sbi sampling
            progress bars (bounded pool) and the optimizer-configuration line.
        log_fn: optional sink called with each per-step optimization-progress line
            and the stop line, for live streaming to a file (e.g. the debug log).
        return_info: also return the ascent's diagnostics (see Returns).
        (remaining args): seed-then-optimize hyperparameters; see
            ``InferenceEvaluation`` for meanings and defaults.

    The ascent steps in units of the candidate pool's per-coordinate interquartile range
    (:func:`pool_scale`). A zero or non-finite IQR cannot set a step: the ascent is then skipped,
    the best candidate is returned (its score is the density at it), and a WARNING line is printed
    and passed to ``log_fn``, with ``stop = "invalid-scale"``.

    Returns:
        ``(optimal_score, optimal_theta)`` -- the log-probability and the inferred
        MAP theta (1D numpy array, estimator coordinates). With ``return_info``, a third item: a dict with
        ``stop`` (one of :data:`STOP_REASONS`), ``steps``, ``seed_score`` (the best seed's score),
        ``learning_rate_final``, ``center`` and ``scale`` (the pool's median and IQR per
        coordinate) and ``scale_valid`` (per coordinate).
    """
    omega = torch.tensor(normalize_video(video_chunk), dtype=torch.float32,
                         device=train_device)
    cond = omega.unsqueeze(0)                                      # mount batch dim
    posterior.set_default_x(cond)
    flow = posterior.posterior_estimator

    t0 = time.time()
    # Candidate pool: drawn via the bounded posterior, which embeds the video
    # itself; leave it on the raw conditioning.
    theta_prex = collect_theta_prex(posterior, flow, vista_device, cond,
                                    theta_prex_size, theta_prex_batch_size,
                                    pool_mode, show, verbose)
    # Embed the conditioning video ONCE, then swap the flow's embedding net for an
    # identity so scoring + optimization reuse the cached latent instead of re-running
    # the (expensive) Complex3DCNN + TemporalTransformer forward per candidate / per
    # gradient step. Behavior-preserving: the log-prob values are identical -- only the
    # embedding is hoisted out of the inner loops. The cached latent stands in for
    # `cond`; collect_score_prex / optimize_elite expand it exactly as before.
    with torch.no_grad():
        emb_x = flow.embedding_net(cond).detach()
    embed_holder = flow.net if (hasattr(flow, "net")
                                and hasattr(flow.net, "_embedding_net")) else flow
    original_embedding_net = embed_holder._embedding_net
    original_condition_shape = flow.condition_shape
    embed_holder._embedding_net = torch.nn.Identity()
    flow._condition_shape = tuple(emb_x.shape[1:])   # condition is now the cached latent (backing attr; the public one is read-only)
    try:
        score_prex = collect_score_prex(flow, train_device, vista_device, emb_x,
                                        theta_prex, score_prex_batch_size, show)
        elite_prex = extract_elite_prex(theta_prex, score_prex, elite_prex_size, show)
        center, iqr = pool_scale(theta_prex)
        scale_valid = np.isfinite(iqr) & (iqr > 0)
        if scale_valid.all():
            optimal_score, optimal_theta, info = optimize_elite(
                flow, train_device, vista_device, emb_x, elite_prex,
                numb_steps, optimizer_patience, scheduler_patience, show_progress_steps,
                learning_rate_minimum, learning_rate_factor, learning_rate, tolerance,
                show, verbose, log_fn, center=center, scale=iqr,
            )
        else:
            # A zero or non-finite IQR cannot set a step size: report it and return the best
            # candidate unoptimized, rather than divide through.
            optimal_score, optimal_theta, info = _best_candidate(theta_prex, score_prex)
            bad = np.flatnonzero(~scale_valid)
            msg = (f"WARNING: the candidate pool's IQR is zero or non-finite at coordinate(s) "
                   f"{bad.tolist()} (IQR {iqr[bad].tolist()}); the MAP ascent is skipped and the "
                   f"best candidate returned.")
            print(f"  [optim] {msg}", flush=True)
            if log_fn is not None:
                log_fn(msg)
    finally:
        embed_holder._embedding_net = original_embedding_net
        flow._condition_shape = original_condition_shape
    info.update(center=center, scale=iqr, scale_valid=scale_valid)
    if show:
        print(f"  [MAP  ] process time {time.time() - t0:.3f}s", flush=True)
    return (optimal_score, optimal_theta, info) if return_info else (optimal_score, optimal_theta)


def posterior_summary(posterior, video_chunk: np.ndarray,
                      train_device: torch.device, vista_device: torch.device,
                      n_samples: int, theta_prex_batch_size: int,
                      pool_mode: str = "bounded",
                      quantiles=QUANTILE_LEVELS,
                      return_samples: bool = False,
                      return_sgm: bool = False, sgm_scale=None):
    """Per-parameter quantile summary of one observation's summary draws, and the draw-derived
    point estimates ``median`` and ``sgm`` of :data:`POINT_ESTIMATES`.

    Draws ``n_samples`` conditioned on the video through the same two-mode sampler as the
    candidate pool (``bounded``: rejection sampling inside the prior, so posterior draws;
    ``unrestricted``: direct flow sampling, so flow draws that may fall outside the prior's
    support -- see :func:`draw_label`) and returns the quantiles of each parameter at the
    ``quantiles`` levels (default :data:`QUANTILE_LEVELS`; the 0.50 level is the marginal
    median), each coordinate independently, in estimator coordinates, with numpy's default
    linear interpolation between order statistics. A physical value is the transform of the
    quantile; the quantile of transformed draws can differ in a finite sample and is not used.
    This captures the *within-observation* spread that a single point estimate discards.

    ``return_samples`` additionally hands back the draws the quantiles were computed
    from. Quantiles describe a marginal per parameter and so discard the joint
    structure -- the correlations between parameters, and any multimodality -- which is
    exactly what a pooled sample cloud is needed for downstream. The draws are the
    same ones the summary used, not a second independent set, so the returned cloud and
    quantiles are guaranteed consistent with each other.

    ``return_sgm`` additionally returns the sample geometric median of the same draws: the
    exact sample medoid (:func:`sample_geometric_median`), distances measured in estimator
    coordinates after dividing each dimension by ``sgm_scale`` -- pass :func:`prior_scale` of
    the learnable table so no parameter dominates the norm. The SGM is a joint point estimate
    that is itself one of the draws: it summarizes the cloud without leaving it, which the
    per-marginal medians (a vector of 1-D medians need not be a probable point) and the MAP (an
    optimizer climbing the flow's density, possibly into a spike outside the training support)
    do not guarantee.

    Returns:
        ``(D, len(quantiles))`` numpy array (estimator coordinates): per parameter, the
        sampled quantiles in the order given by ``quantiles``. With
        ``return_samples``, the tuple ``(summary, draws)`` where ``draws`` is
        ``(n_samples, D)`` float32 in estimator coordinates. With ``return_sgm`` the SGM
        vector ``(D,)`` is appended as the last element of the returned tuple.
    """
    omega = torch.tensor(normalize_video(video_chunk), dtype=torch.float32,
                         device=train_device)
    cond = omega.unsqueeze(0)
    posterior.set_default_x(cond)
    flow = posterior.posterior_estimator
    samples = collect_theta_prex(posterior, flow, vista_device, cond,
                                 n_samples, theta_prex_batch_size, pool_mode)
    arr = samples.detach().cpu().numpy()                       # (n_samples, D)
    summary = np.quantile(arr, list(quantiles), axis=0).T      # (D, len(quantiles))
    out = [summary]
    if return_samples:
        out.append(arr.astype(np.float32))
    if return_sgm:
        out.append(sample_geometric_median(arr, scale=sgm_scale)[0])
    return out[0] if len(out) == 1 else tuple(out)


def prior_scale(parameterization) -> np.ndarray:
    """Per-parameter prior-box width ``(D,)`` in the table's coordinate (log10 for a log
    row) -- the natural scale for a joint distance between parameter vectors, so that a
    parameter with a wide prior does not dominate one with a narrow prior."""
    return np.asarray([float(p["PRIOR_RANGE"][1]) - float(p["PRIOR_RANGE"][0])
                       for p in parameterization], dtype=float)


def sample_geometric_median(samples: np.ndarray, scale=None):
    """The sample geometric median (SGM) of one cloud: the EXACT sample medoid, the sample
    minimizing the summed Euclidean distance to all samples, found by evaluating that sum for
    every sample. It holds the N x N distance matrix: about 1.5 GiB and 1 s at 10,000 draws.
    It is not the continuous geometric median snapped to its nearest sample, the approximation the
    collection-level kernel (:mod:`.sample_geometric_median`) switches to above its capacity; the
    two can select different members. Ties resolve to the lowest row index, so the selection is
    deterministic. (SGM discussion: Mach. Learn.: Sci. Technol. 2025,
    doi 10.1088/2632-2153/ada0a3.)

    ``samples`` is ``(N, D)`` in the caller's coordinates; ``scale`` (optional, ``(D,)``) divides
    each dimension before the distance is taken. The stages pass one observation's summary draws
    in estimator coordinates with the prior widths (:func:`prior_scale`): the posterior-draw SGM of
    :data:`POINT_ESTIMATES`. The result is a realized draw, so its coordinates co-occurred, unlike
    a vector of per-dimension medians, which can land in a gap between modes; selecting one draw
    does not by itself preserve the cloud's correlations. Returns ``(vector, index)``: the SGM in
    the ORIGINAL coordinates ``(D,)`` and its row index.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("sample_geometric_median: expected a non-empty (N, D) array.")
    z = arr / np.asarray(scale, dtype=float) if scale is not None else arr
    sq = np.einsum("ij,ij->i", z, z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)
    np.maximum(d2, 0.0, out=d2)
    idx = int(np.argmin(np.sqrt(d2).sum(axis=1)))
    return arr[idx].copy(), idx


def point_estimate_agreement_table(parameterization, map_log10: np.ndarray,
                                   post_q: np.ndarray, sgm_log10=None,
                                   groups=None, group_header: str = "kind") -> tuple:
    """Build ``(headers, rows)`` comparing the three point estimates of one posterior.

    Per parameter (and per group when ``groups`` -- a label per observation -- is
    given): the median absolute gap between the MAP and the 1-D posterior median
    (``post_q[:, :, 2]``), between the MAP and the SGM, and between the SGM and the 1-D
    median, in the row's error unit (dex for a log row, percentage points for a linear row;
    the ``units`` column); and the share of observations whose MAP lies outside the
    posterior's central 90% interval ``[Q05, Q95]``. A MAP far from both medians and
    outside the 90% interval marks an optimizer that climbed into a density spike the
    posterior samples do not visit, or an optimizer that stopped short, or another feature
    of the posterior's shape; the table establishes the disagreement, and separate checks
    decide between those readings. ``sgm_log10`` may be ``None`` (columns show ``n/a``).
    """
    map_log10 = np.asarray(map_log10, dtype=float)
    post_q = np.asarray(post_q, dtype=float)
    sgm = None if sgm_log10 is None else np.asarray(sgm_log10, dtype=float)
    n_obs = map_log10.shape[0]
    labels = np.asarray(["all"] * n_obs) if groups is None else np.asarray(groups)
    headers = ["parameter", "label", "units"] + ([group_header] if groups is not None else []) + [
        "n", "MAP vs median", "MAP vs SGM", "SGM vs median", "MAP outside 90%"]
    rows = []
    for i, para in enumerate(parameterization):
        scale = _error_scale(para)
        for g in (dict.fromkeys(labels) if groups is not None else ["all"]):
            m = labels == g
            mp, q = map_log10[m, i], post_q[m, i, :]
            ok = np.isfinite(mp) & np.all(np.isfinite(q), axis=1)
            mp, q = mp[ok], q[ok]
            cells = ([para["KEY"], para.get("LABEL") or "-", row_error_unit(para)]
                     + ([str(g)] if groups is not None else []))
            if not mp.size:
                rows.append(cells + ["0", "-", "-", "-", "-"])
                continue
            med = q[:, 2]
            outside = float(np.mean((mp < q[:, 0]) | (mp > q[:, 4])))
            if sgm is not None:
                sg = sgm[m, i][ok]
                map_sgm = f"{scale * np.median(np.abs(mp - sg)):.3f}"
                sgm_med = f"{scale * np.median(np.abs(sg - med)):.3f}"
            else:
                map_sgm = sgm_med = "n/a"
            rows.append(cells + [str(mp.size), f"{scale * np.median(np.abs(mp - med)):.3f}",
                                 map_sgm, sgm_med, f"{outside * 100:.0f}%"])
    return headers, rows


def point_estimates_compared_table(parameterization, true_log10: np.ndarray,
                                   estimates: dict) -> tuple:
    """Build ``(headers, rows)`` placing the point estimates side by side per parameter.

    ``estimates`` maps a short estimate name (``"MAP"``, ``"median"``, ``"SGM"``) to its
    ``(N, D)`` array in estimator coordinates; insertion order is the column order within each
    statistic. One row per learnable parameter. The columns are grouped by statistic, with the
    estimates consecutive inside each group: correlation with the truth for MAP, median, SGM;
    then MAE; then signed bias (mean of inferred - true), both in the row's error unit (dex for
    a log row, percentage points for a linear row; the ``units`` column); then the share
    outside the prior box. The single-estimate recovery tables carry each estimate's full
    statistics; this table is the view in which the three are read against each other. A
    ``None`` entry renders as ``n/a``.
    """
    true_log10 = np.asarray(true_log10, dtype=float)
    names = [n for n in estimates]
    stats = ("corr", "MAE", "bias", "outside prior")
    headers = ["parameter", "label", "units", "n"] + [f"{s} {n}" for s in stats for n in names]
    present = {n: np.asarray(estimates[n], dtype=float) for n in names
               if estimates[n] is not None and np.asarray(estimates[n]).size}
    rows = []
    for i, para in enumerate(parameterization):
        scale = _error_scale(para)
        cells = {s: [] for s in stats}
        # One shared mask per row: the videos finite in the truth and in every present estimate,
        # so the three columns of a statistic describe the same videos and 'n' counts them.
        shared = np.isfinite(true_log10[:, i])
        for arr in present.values():
            shared &= np.isfinite(arr[:, i])
        for name in names:
            if name not in present:
                for s in stats:
                    cells[s].append("n/a")
                continue
            est, tru = present[name][shared, i], true_log10[shared, i]
            if est.size < 3:
                for s in stats:
                    cells[s].append("n/a")
                continue
            corr = correlation_with_truth(tru, est)
            cells["corr"].append("n/a" if corr is None else f"{corr:+.2f}")
            cells["MAE"].append(f"{scale * np.mean(np.abs(est - tru)):.3f}")
            cells["bias"].append(f"{scale * np.mean(est - tru):+.3f}")
            cells["outside prior"].append(f"{fraction_outside_prior(para, est) * 100:.0f}%")
        rows.append([para["KEY"], para.get("LABEL") or "-", row_error_unit(para), str(int(shared.sum()))]
                    + [c for s in stats for c in cells[s]])
    return headers, rows


def experiment_estimates_compared_table(parameterization, estimates_by_kind: dict,
                                        kinds) -> tuple:
    """Build ``(headers, rows)`` placing the point estimates side by side per condition.

    ``estimates_by_kind`` maps a short estimate name to the ``{kind: (N, D) array}`` dict (in
    estimator coordinates) that ``experiment_table`` consumes; insertion order is the column
    order within each statistic. One row per (parameter, kind); columns grouped by statistic
    with the estimates consecutive: the median over windows for MAP, median, SGM; then the IQR
    over windows, both in the row's estimator coordinate (log10 for a log row, the fraction
    itself for a linear row; the ``units`` column); then the share outside the prior box. No
    ground truth exists for experimental recordings, so this is the distribution view, read
    across the three estimates on the same windows.
    """
    names = [n for n in estimates_by_kind]
    stats = ("median", "IQR", "outside prior")
    headers = ["parameter", "label", "units", "kind", "n"] + [f"{s} {n}" for s in stats for n in names]
    rows = []
    for i, para in enumerate(parameterization):
        for kind in kinds:
            cells = {s: [] for s in stats}
            cols = {}
            for name in names:
                arr = np.asarray(estimates_by_kind[name].get(kind, []), dtype=float)
                cols[name] = arr[:, i] if arr.ndim == 2 and arr.size else None
            # One shared mask per row: windows finite in every present estimate, so the columns
            # of a statistic describe the same windows and 'n' counts them.
            lengths = {c.size for c in cols.values() if c is not None}
            shared = None
            if len(lengths) == 1:
                shared = np.ones(lengths.pop(), dtype=bool)
                for c in cols.values():
                    if c is not None:
                        shared &= np.isfinite(c)
            for name in names:
                col = cols[name]
                if col is None or shared is None:
                    for s in stats:
                        cells[s].append("n/a")
                    continue
                col = col[shared]
                if not col.size:
                    for s in stats:
                        cells[s].append("n/a")
                    continue
                q1, med, q3 = np.quantile(col, [0.25, 0.5, 0.75])
                cells["median"].append(f"{med:+.3f}")
                cells["IQR"].append(f"{q3 - q1:.3f}")
                cells["outside prior"].append(f"{fraction_outside_prior(para, col) * 100:.0f}%")
            rows.append([para["KEY"], para.get("LABEL") or "-", row_coordinate_unit(para), kind,
                         str(int(shared.sum())) if shared is not None else "0"]
                        + [c for s in stats for c in cells[s]])
    return headers, rows


# =============================================================================
# Recovery statistics (true vs. inferred, estimator coordinates)
# =============================================================================

def recovery_stats(true_log10: np.ndarray, estimate_log10: np.ndarray,
                   guide: float = 0.3, guide_tight: float = 0.15, *, log_rows=None) -> list:
    """Per-parameter recovery error statistics, in estimator coordinates.

    Args:
        true_log10: ground-truth theta in estimator coordinates, shape ``(N, D)``.
        estimate_log10: the point estimate (any of MAP, median, SGM), shape ``(N, D)``.
        guide: half-width of the "within guide" band (dex). The default
            0.3 ~= log10(2), so it counts recoveries within a factor of 2 of the truth.
        guide_tight: half-width of a tighter, nested band. The default 0.15
            ~= log10(sqrt(2)) = guide / 2, counting recoveries within a factor of
            sqrt(2) ~= 1.41 -- a stricter concentration measure.
        log_rows: one boolean per column, True for a log row. ``None`` means every column
            is a log row (an all-log table). The bands are multiplicative tolerances and so
            exist only for log rows: a linear row's ``frac_within_guide`` and
            ``frac_within_guide_tight`` are NaN.

    Returns:
        A list of ``D`` dicts, one per parameter, each with keys: ``n``,
        ``median_error``, ``mae``, ``rmse``, ``q95_abs_error`` (all in the coordinate's own
        unit: dex for a log row, the fraction itself for a linear row -- callers scale the
        latter to percentage points), ``frac_within_guide`` (within +/-guide) and
        ``frac_within_guide_tight`` (within +/-guide_tight) -- computed over the finite
        (true, inferred) pairs.
    """
    true_log10 = np.asarray(true_log10, dtype=float)
    estimate_log10 = np.asarray(estimate_log10, dtype=float)
    d = true_log10.shape[1]
    if log_rows is None:
        log_rows = [True] * d
    log_rows = [bool(v) for v in log_rows]
    if len(log_rows) != d:
        raise ValueError(f"recovery_stats: log_rows has {len(log_rows)} entries; the arrays have {d} columns.")
    stats = []
    for i in range(d):
        x = true_log10[:, i]
        y = estimate_log10[:, i]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        error = y - x
        if error.size:
            abs_error = np.abs(error)
            stats.append({
                "n": int(error.size),
                "median_error": float(np.median(error)),
                "mae": float(np.mean(abs_error)),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "q95_abs_error": float(np.quantile(abs_error, 0.95)),
                "frac_within_guide": float(np.mean(abs_error <= guide)) if log_rows[i] else float("nan"),
                "frac_within_guide_tight": (float(np.mean(abs_error <= guide_tight)) if log_rows[i]
                                            else float("nan")),
            })
        else:
            stats.append({"n": 0, "median_error": float("nan"),
                          "mae": float("nan"), "rmse": float("nan"),
                          "q95_abs_error": float("nan"),
                          "frac_within_guide": float("nan"),
                          "frac_within_guide_tight": float("nan")})
    return stats


def experiment_table(parameterization, inferred_by_kind: dict, kinds) -> tuple:
    """Build a ``(headers, rows)`` summary of inferred theta per condition.

    For real microscopy data there is no ground truth, so this reports, per
    learnable parameter and per condition (kind), the distribution of the
    inferred MAP theta: count, median and IQR in the row's estimator coordinate (log10 for
    a log row, the fraction itself for a linear row; the ``units`` column), the median in
    physical units, and the share of estimates outside the row's prior box
    (``fraction_outside_prior``; possible under either pool mode, because the pool mode
    bounds the candidate pool and not the unconstrained gradient ascent).
    ``inferred_by_kind`` maps each kind to an ``(N, D)`` array of inferred theta in estimator
    coordinates (N = cells x chunks for that kind).
    """
    headers = ["parameter", "label", "units", "kind", "n", "median (estimator)",
               "IQR (estimator)", "median value", "outside prior"]
    rows = []
    for i, para in enumerate(parameterization):
        for kind in kinds:
            arr = np.asarray(inferred_by_kind.get(kind, []), dtype=float)
            col = arr[:, i] if arr.ndim == 2 and arr.size else np.array([])
            col = col[np.isfinite(col)]
            if col.size:
                med = float(np.median(col))
                q1, q3 = np.quantile(col, [0.25, 0.75])
                rows.append([para["KEY"], para.get("LABEL") or "-", row_coordinate_unit(para), kind,
                             str(col.size), f"{med:+.3f}", f"{q3 - q1:.3f}",
                             f"{float(entry_to_physical(para, med)):.4g}",
                             f"{fraction_outside_prior(para, col) * 100:.0f}%"])
            else:
                rows.append([para["KEY"], para.get("LABEL") or "-", row_coordinate_unit(para), kind,
                             "0", "nan", "nan", "nan", "n/a"])
    return headers, rows


def posterior_coverage_table(parameterization, true_log10: np.ndarray,
                             post_q: np.ndarray) -> tuple:
    """Posterior calibration summary per parameter (recovery stage).

    ``post_q`` is an ``(N, D, 5)`` array of per-observation posterior quantiles
    ``[Q05, Q25, Q50, Q75, Q95]`` in estimator coordinates. Reports, per parameter, the
    fraction of ground-truths falling inside the 50% (IQR) and 90% (Q05-Q95) posterior
    credible intervals -- a well-calibrated posterior covers ~50% and ~90%
    respectively -- plus the median absolute bias of the posterior median, in the row's
    error unit (dex or percentage points; the ``units`` column).
    """
    headers = ["parameter", "label", "units", "n", "cover@50% (IQR)", "cover@90%",
               "median |median-true|"]
    true_log10 = np.asarray(true_log10, dtype=float)
    rows = []
    for i, para in enumerate(parameterization):
        t = true_log10[:, i]
        q = post_q[:, i, :]
        mask = np.isfinite(t) & np.all(np.isfinite(q), axis=1)
        t, q = t[mask], q[mask]
        if t.size:
            c50 = float(np.mean((t >= q[:, 1]) & (t <= q[:, 3])))
            c90 = float(np.mean((t >= q[:, 0]) & (t <= q[:, 4])))
            bias = float(np.median(np.abs(q[:, 2] - t))) * _error_scale(para)
            rows.append([para["KEY"], para.get("LABEL") or "-", row_error_unit(para), str(t.size),
                         f"{c50 * 100:.0f}%", f"{c90 * 100:.0f}%", f"{bias:.3f}"])
        else:
            rows.append([para["KEY"], para.get("LABEL") or "-", row_error_unit(para), "0", "-", "-", "-"])
    return headers, rows


def fraction_outside_prior(para, values) -> float:
    """Share of ``values`` (in the table's coordinate: log10 for a log row, the value itself
    for a linear row) outside the row's ``PRIOR_RANGE``. NaN when no finite value is present.
    Non-zero only when the estimator placed a MAP estimate beyond the prior box, which the
    unrestricted pool permits. Counts only; the values are never changed."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan")
    lo, hi = (float(b) for b in para["PRIOR_RANGE"])
    return float(np.mean((v < lo) | (v > hi)))


def fraction_flag_rows(parameterization, estimates: dict, post_q=None) -> tuple:
    """The flagged-fractions table: ``(headers, rows, n_flagged)`` over the LINEAR rows of
    ``parameterization`` (the two fractions of the biology table; an all-log table yields no
    rows). Per linear row: the number of MAP, median and SGM values (``estimates``, a short
    name -> ``(N, D)`` array) that are non-finite or outside the row's box, and, when
    ``post_q`` ``(N, D, 5)`` is given, the share of observations whose Q05 lies below the box
    or whose Q95 lies above it. The values are counted and left unchanged: nothing is clipped;
    a flagged vector carries NaN in every derived quantity downstream. ``n_flagged`` is the
    total count over the estimates, for a non-fatal report check.
    """
    headers = ["parameter", "label", "box", "n"] + [f"{name} outside" for name in estimates] + [
        "Q05 < lo or Q95 > hi"]
    rows, n_flagged = [], 0
    for i, para in enumerate(parameterization):
        if is_log_row(para):
            continue
        lo, hi = (float(b) for b in para["PRIOR_RANGE"])
        counts, n_obs = [], 0
        for name, arr in estimates.items():
            if arr is None or not np.asarray(arr).size:
                counts.append("n/a")
                continue
            col = np.asarray(arr, dtype=float)[:, i]
            n_obs = max(n_obs, int(col.size))
            bad = int(np.sum(~(np.isfinite(col) & (col >= lo) & (col <= hi))))
            n_flagged += bad
            counts.append(str(bad))
        if post_q is not None and np.asarray(post_q).size:
            q = np.asarray(post_q, dtype=float)[:, i, :]
            share = float(np.mean((q[:, 0] < lo) | (q[:, 4] > hi)))
            share_cell = f"{share * 100:.1f}%"
        else:
            share_cell = "n/a"
        rows.append([para["KEY"], para.get("LABEL") or "-", f"[{lo:g}, {hi:g}]", str(n_obs)]
                    + counts + [share_cell])
    return headers, rows, n_flagged


# =============================================================================
# Recovery of the window-start allocation (the Evaluation stage, biology)
# =============================================================================

#: The strata of the TRUE capacity ``cap = A_ref // 2`` (the probe-bound monomer pairs the
#: reference can turn into two-InlB dimers): ``(label, low, high)`` with ``high`` inclusive.
CAPACITY_STRATA = (
    ("0 (no effect, prior-like posterior expected)", 0, 0),
    ("1-9", 1, 9),
    ("10-99", 10, 99),
    ("100-999", 100, 999),
    (">= 1000", 1000, None),
)


def _capacity_stratum(capacity: np.ndarray) -> np.ndarray:
    """Index into :data:`CAPACITY_STRATA` for every true capacity (``-1`` for a NaN)."""
    cap = np.asarray(capacity, dtype=float)
    out = np.full(cap.shape, -1, dtype=int)
    for s, (_, lo, hi) in enumerate(CAPACITY_STRATA):
        m = np.isfinite(cap) & (cap >= lo) & ((cap <= hi) if hi is not None else True)
        out[m] = s
    return out


def starting_composition_recovery(condition: str, parameterization, true_log10: np.ndarray,
                                  estimates: dict, post_q: np.ndarray, median_index: int) -> dict:
    """Recovery of the window-start composition from a biology Evaluation product.

    ``parameterization`` is the condition's learnable table (``parameterization_of(condition)``:
    the keys must equal ``parameter_keys_of(condition)``); ``true_log10``, the estimates (a
    short name -> ``(N, D)`` array; ``"MAP"``, ``"median"``, ``"SGM"``) and ``post_q`` are in
    estimator coordinates. The true capacity and the true number of allocated pairs c come
    from the physical truth through ``population_from_theta``; every estimate vector is
    realized the same way AFTER masking with ``population_valid`` (a vector whose fraction or
    allocation falls outside [0, 1], or whose count is not positive, is excluded from this
    section and counted under ``flagged``; nothing is clipped).

    Under MET-INLB the result holds, per TRUE capacity stratum (:data:`CAPACITY_STRATA`):
    the allocation's MAE and signed bias in percentage points for every estimate, the 50 %
    and 90 % coverage of the true allocation by the per-video quantiles, and the recovery of
    c in counts from the JOINT vectors MAP and SGM (never the coordinate-wise median, which
    is not a vector of the posterior). Under MET-FAB the allocation is the declared 0: the
    result carries no strata and states that c is 0 for every video and every valid estimate.

    Returns a dict: ``has_allocation``, ``n_videos``, ``flagged`` (estimate -> count excluded),
    ``allocation_key``, ``strata`` (a list of per-stratum dicts, empty under MET-FAB),
    ``c_true_all_zero`` and ``c_estimates_all_zero`` (MET-FAB), ``true_capacity`` and
    ``true_pairs`` (arrays over the videos).
    """
    keys = [e["KEY"] for e in parameterization]
    if keys != parameter_keys_of(condition):
        raise ValueError(f"starting_composition_recovery [{condition}]: the table's keys {keys} are not the "
                         f"condition's {parameter_keys_of(condition)}.")
    true_log10 = np.asarray(true_log10, dtype=float)
    post_q = np.asarray(post_q, dtype=float)
    n = true_log10.shape[0]
    true_phys = to_physical(true_log10, parameterization)
    pop_true = population_from_theta(true_phys, condition)
    capacity = np.asarray(pop_true.capacity, dtype=float)
    pairs_true = np.asarray(pop_true.n_pairs_allocated, dtype=float)
    stratum = _capacity_stratum(capacity)

    index = {k: i for i, k in enumerate(keys)}
    sto = PARAMETERS.simulation.rds.stoichiometry
    r_key = sto.active_allocation_key
    has_allocation = r_key in index

    # Every estimate: the valid subset (population_valid on the physical vector), realized.
    valid, pairs_est, flagged = {}, {}, {}
    for name, arr in estimates.items():
        if arr is None or not np.asarray(arr).size:
            continue
        est = np.asarray(arr, dtype=float)
        phys = to_physical(est, parameterization)
        r = (phys[:, index[r_key]] if has_allocation
             else np.zeros(n, dtype=float))
        ok = population_valid(count_total=phys[:, index[sto.count_total_key]],
                              fraction_basal_dimers_reference=phys[:, index[sto.basal_fraction_key]],
                              relative_count_active_dimers_initial=r, condition=condition)
        valid[name] = ok
        flagged[name] = int(n - np.count_nonzero(ok))
        pe = np.full(n, np.nan)
        if np.any(ok):
            pe[ok] = np.asarray(population_from_theta(phys[ok], condition).n_pairs_allocated, dtype=float)
        pairs_est[name] = pe

    result = {"condition": str(condition), "has_allocation": bool(has_allocation), "n_videos": int(n),
              "flagged": flagged, "allocation_key": r_key, "strata": [],
              "true_capacity": capacity, "true_pairs": pairs_true, "estimated_pairs": pairs_est}
    if not has_allocation:
        result["c_true_all_zero"] = bool(np.all(pairs_true == 0))
        result["c_estimates_all_zero"] = {name: bool(np.all(pe[valid[name]] == 0)) if np.any(valid[name]) else True
                                          for name, pe in pairs_est.items()}
        return result

    j = index[r_key]
    r_true = true_log10[:, j]
    for s, (label, lo, hi) in enumerate(CAPACITY_STRATA):
        m = stratum == s
        row = {"label": label, "low": lo, "high": hi, "n": int(m.sum()), "allocation": {}, "pairs": {}}
        if m.any():
            q = post_q[m, j, :]
            t = r_true[m]
            q25, q75 = q[:, median_index - 1], q[:, median_index + 1]
            row["cover50"] = float(np.mean((t >= q25) & (t <= q75)))
            row["cover90"] = float(np.mean((t >= q[:, 0]) & (t <= q[:, 4])))
            row["true_pairs_median"] = float(np.median(pairs_true[m]))
        else:
            row["cover50"] = row["cover90"] = row["true_pairs_median"] = float("nan")
        for name, arr in estimates.items():
            if name not in valid:
                continue
            sel = m & valid[name]
            if sel.any():
                err = np.asarray(arr, dtype=float)[sel, j] - r_true[sel]
                row["allocation"][name] = {"n": int(sel.sum()), "mae_pp": float(_PP * np.mean(np.abs(err))),
                                           "bias_pp": float(_PP * np.mean(err))}
            else:
                row["allocation"][name] = {"n": 0, "mae_pp": float("nan"), "bias_pp": float("nan")}
            if name == "median":
                continue        # c is read from the joint vectors only
            if sel.any():
                derr = pairs_est[name][sel] - pairs_true[sel]
                row["pairs"][name] = {"n": int(sel.sum()), "mae": float(np.mean(np.abs(derr))),
                                      "bias": float(np.mean(derr))}
            else:
                row["pairs"][name] = {"n": 0, "mae": float("nan"), "bias": float("nan")}
        result["strata"].append(row)
    return result


def starting_composition_tables(result: dict) -> list:
    """Render :func:`starting_composition_recovery` as report tables: a list of
    ``(title, headers, rows, note)``. Under MET-FAB a single one-row table states the declared
    allocation and that c is 0; under MET-INLB two tables, the allocation's recovery (pp) with
    coverage per true-capacity stratum and the recovery of c in counts from MAP and SGM."""
    fmt = lambda v, f="{:.2f}": "n/a" if not np.isfinite(v) else f.format(v)   # noqa: E731
    flagged = ", ".join(f"{k} {v}" for k, v in result["flagged"].items()) or "none"
    if not result["has_allocation"]:
        rows = [[result["condition"], "declared 0", "0 for every video" if result["c_true_all_zero"] else "NOT 0",
                 ", ".join(f"{k}: {'0' if v else 'NOT 0'}" for k, v in result["c_estimates_all_zero"].items()),
                 flagged]]
        return [("Starting composition (window start)",
                 ["condition", "allocation R_active", "c from the truth", "c from the estimates (valid vectors)",
                  "flagged vectors excluded"], rows,
                 "the condition declares the allocation as a constant, so no allocation is inferred, no "
                 "true-capacity strata are formed and the number of allocated pairs c is 0 for every video; "
                 "the estimates are checked for the same. A flagged vector (fraction outside [0, 1] or a "
                 "non-positive count) is excluded and counted, never clipped.")]
    names = [n for n in next(iter(result["strata"]))["allocation"]] if result["strata"] else []
    joint = [n for n in names if n != "median"]
    a_headers = ["true capacity", "n"] + [f"{s} {n}" for n in names for s in ("MAE (pp)", "bias (pp)")] + [
        "cover@50%", "cover@90%"]
    a_rows, c_rows = [], []
    for row in result["strata"]:
        cells = [row["label"], str(row["n"])]
        for n in names:
            a = row["allocation"].get(n, {"mae_pp": np.nan, "bias_pp": np.nan})
            cells += [fmt(a["mae_pp"]), fmt(a["bias_pp"], "{:+.2f}")]
        cells += [fmt(100 * row["cover50"], "{:.0f}%"), fmt(100 * row["cover90"], "{:.0f}%")]
        a_rows.append(cells)
        c_cells = [row["label"], str(row["n"]), fmt(row["true_pairs_median"], "{:.0f}")]
        for n in joint:
            p = row["pairs"].get(n, {"mae": np.nan, "bias": np.nan})
            c_cells += [fmt(p["mae"], "{:.1f}"), fmt(p["bias"], "{:+.1f}")]
        c_rows.append(c_cells)
    c_headers = ["true capacity", "n", "c true (median)"] + [f"{s} {n}" for n in joint
                                                               for s in ("MAE (counts)", "bias (counts)")]
    return [
        (f"Starting composition: recovery of the allocation {result['allocation_key']} per true-capacity stratum",
         a_headers, a_rows,
         f"the true capacity cap = A_ref // 2 (the probe-bound monomer pairs the reference can pair) is "
         f"computed from the physical truth; each stratum holds the videos whose true capacity falls in it. "
         f"MAE and signed bias (inferred - true) of the allocation are in percentage points for every "
         f"estimate; the coverage columns are the share of truths inside the per-video 50 % and 90 % "
         f"intervals of the draws (one posterior, so they do not depend on the estimate). At capacity 0 "
         f"the allocation has no effect on the video and a prior-like posterior is expected there. "
         f"Flagged vectors excluded and counted: {flagged}."),
        ("Starting composition: recovery of the allocated pairs c (counts)", c_headers, c_rows,
         "c = round(R_active x capacity) realized from the JOINT vectors (MAP and SGM, which are "
         "vectors of the posterior; the coordinate-wise median is not) against c from the truth, in "
         "counts, per true-capacity stratum. Zero-safe: at capacity 0 every c is 0."),
    ]


def correlation_with_truth(true_values, inferred_values, min_pairs: int = 3):
    """Pearson correlation between inferred and true values over the finite pairs, or
    ``None`` below ``min_pairs`` pairs or when either side has zero spread. Near zero for
    an estimator whose output does not depend on its input (an undertrained network
    returns a near-constant vector), which no error statistic exposes on its own."""
    x = np.asarray(true_values, dtype=float)
    y = np.asarray(inferred_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_pairs or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def recovery_table(parameterization, true_log10: np.ndarray,
                   estimate_log10: np.ndarray, guide: float = 0.3,
                   guide_tight: float = 0.15) -> tuple:
    """Build a ``(headers, rows)`` recovery summary for the diagnostic report.

    One row per learnable parameter, with the error statistics from
    ``recovery_stats`` in the row's error unit (the ``units`` column: dex for a log row,
    percentage points for a linear row, where the estimator-coordinate error is multiplied by
    100). Two "within" columns report the fraction recovered inside the +/-guide band and
    the tighter, nested +/-guide_tight band. Both columns are headed by the multiplicative
    range the tolerance permits rather than its log10 half-width -- ``[0.50x, 2.00x]`` for
    0.3 dex and ``[0.71x, 1.41x]`` for 0.15 -- because that is the form in which a reader can
    judge whether a parameter is usable, without doing the arithmetic first. A linear row has
    no multiplicative band: its two cells read ``n/a``. Two further columns expose failure
    modes the error statistics hide: ``outside prior`` is the share of MAP estimates
    beyond the row's prior box (``fraction_outside_prior``), and ``corr(inf, true)`` is
    the correlation between inferred and true values (``correlation_with_truth``; ``n/a``
    below three pairs or at zero spread). ``parameterization`` is the learnable table of the
    condition (its order matches the theta columns).
    """
    headers = ["parameter", "label", "units", "n", "median err", "MAE", "RMSE",
               "q95|err|", f"within {band_label(guide)}",
               f"within {band_label(guide_tight)}", "outside prior", "corr(inf, true)"]
    log_rows = [is_log_row(p) for p in parameterization]
    stats = recovery_stats(true_log10, estimate_log10, guide, guide_tight, log_rows=log_rows)
    true_log10 = np.asarray(true_log10, dtype=float)
    estimate_log10 = np.asarray(estimate_log10, dtype=float)
    rows = []
    for i, (para, st) in enumerate(zip(parameterization, stats)):
        scale = _error_scale(para)
        outside = fraction_outside_prior(para, estimate_log10[:, i]) if estimate_log10.size else float("nan")
        corr = correlation_with_truth(true_log10[:, i], estimate_log10[:, i]) if estimate_log10.size else None
        band = lambda v: "n/a" if not np.isfinite(v) else f"{v * 100:.0f}%"   # noqa: E731
        rows.append([
            para["KEY"],
            para.get("LABEL") or "-",
            row_error_unit(para),
            str(st["n"]),
            f"{scale * st['median_error']:+.3f}",
            f"{scale * st['mae']:.3f}",
            f"{scale * st['rmse']:.3f}",
            f"{scale * st['q95_abs_error']:.3f}",
            band(st["frac_within_guide"]),
            band(st["frac_within_guide_tight"]),
            "n/a" if not np.isfinite(outside) else f"{outside * 100:.0f}%",
            "n/a" if corr is None else f"{corr:+.2f}",
        ])
    return headers, rows
