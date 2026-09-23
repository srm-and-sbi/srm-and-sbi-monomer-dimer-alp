"""Temporal-dynamics kernel: how inferred parameters behave across one recording.

Workflow-agnostic numerics shared by the biology and detector temporal analyses. Nothing here knows
which parameters it is describing: every function takes arrays, a prior box, a time axis, and --
wherever estimator-space values must become physical ones -- the ``to_physical`` callable the runner
binds to its parameter table (``parameterization.to_physical``), plus a per-row ``log_rows`` mask
where a statistic is only defined for log rows. Pure numpy (plus the shared geometric-median kernel
and a lazy scipy import for the sign test), so it imports and unit-tests without a machine profile.

WHAT THE ANALYSIS ASKS. The Experiment stage estimates the parameters independently in every
non-overlapping window of every recording, reporting one MAP estimate per window. Stacking those
windows along time asks a question the stage cannot: does the inferred value hold still across the
recording? A parameter that is a constant property of the system should be flat. A trend is either
real dynamics or an acquisition confound, and one workflow's estimates cannot distinguish them.

THE ONE ARRAY EVERYTHING STARTS FROM. All functions here operate on the MAP grid

    G[k, c, t, p]   MAP estimate in estimator space (log10 for log rows; the value itself for
                    the linear initial dimer fraction)
      k = condition        c = cell (recording)
      t = chunk (window)   p = parameter

built by :func:`reshape_to_grid`. Every entry is the MAP point estimate the Experiment stage
optimized for one (condition, cell, chunk) window -- never a posterior draw and never an average.

THE FOUR CENTRAL ESTIMATES. A timeseries needs one vector per chunk, which means aggregating the
cell axis; a single summary line needs one vector overall, which means aggregating the cell AND
chunk axes. Crossing that choice of axis with the choice of estimator gives exactly four, and each
function below is named for what it aggregates:

    mean-window       mean value vector, aggregated across cells for a given chunk
    sgm-window        realized value vector, aggregated across cells for a given chunk
    mean-trajectory   mean value vector, aggregated across chunks and cells
    sgm-trajectory    realized value vector, aggregated across chunks and cells

"Mean" aggregates each parameter independently, so its coordinates need not have co-occurred in any
recording. "Realized" selects an actual member of the set -- the exact medoid, the member minimizing
the summed distance to every other member -- so every coordinate co-occurred in one real window;
selecting one member does not by itself preserve the set's correlations (Ramirez Sierra &
Sokolowski, Mach. Learn.: Sci. Technol. 6, 015004, 2025). The set is the stored window MAP vectors,
so the realized estimates are SGMs of window MAPs, not the posterior-draw SGM each window also
stores. The two `*-window` functions produce a timeseries; the two `*-trajectory` functions produce
a single vector, and they pair with their window counterpart so a figure never mixes estimators.

DISTANCES. Both realized estimates use the same metric: physical values ``to_physical(G)``, each
parameter divided by its physical prior width ``to_physical(high) - to_physical(low)`` so no
parameter dominates, Euclidean, exact medoid (the shared kernel switches to a snapped Weiszfeld
approximation only above its ``EXACT_MEDOID_CAPACITY``, far beyond a condition's window count).
Selection is on ALL parameters jointly, so a selected
vector is internally coherent -- and consequently the value it reports for one parameter is that
jointly-central window's value, not that parameter's own median.
"""
from __future__ import annotations

import warnings

import numpy as np

from .sample_geometric_median import sample_geometric_median

# A change of this many dex over the recording is called material: 0.3 dex is a factor of two, the
# same practical bar the recovery tables use, so exceeding it moves the estimate by more than the
# tolerance the recovery is judged against.
MATERIAL_DRIFT_DEX = 0.3

# Held-out recovery tolerances, as log10 half-widths, matching the two nested bands the Evaluation
# stage reports. Kept separate from MATERIAL_DRIFT_DEX even though the wider one coincides
# numerically: one is a tolerance on recovery against known truth, the other a threshold on drift
# across a recording, and conflating them would tie two unrelated decisions to one constant.
# Both are DEX quantities and therefore apply to LOG rows only: for a linear row (none in the decided
# biology table; the rule stays general) a dex band is meaningless, so every band-based statistic below reports NaN for it and
# the runner says so instead of silently applying log arithmetic to a linear coordinate.
RECOVERY_BANDS_DEX = (0.3, 0.15)


def band_label(dex):
    """Render a log10 half-width as the multiplicative range it means, e.g. ``[0.50x, 2.0x]``.

    A tolerance stated in dex is unreadable without mental arithmetic, and the arithmetic is the
    interesting part: +/-0.3 dex is "between half and double the truth", +/-0.15 dex is "within
    roughly a third either way". The range is asymmetric in absolute terms and symmetric in log,
    which the two multipliers show directly.
    """
    lo, hi = 10.0 ** -float(dex), 10.0 ** float(dex)
    return f"[{lo:.2f}x, {hi:.2f}x]"


def recovery_fractions(true_flow, inferred_flow, log_rows, bands=RECOVERY_BANDS_DEX):
    """Fraction of held-out videos recovered inside each nested tolerance band, per parameter.

    ``true_flow`` / ``inferred_flow`` are ``(N, D)`` in estimator space; ``log_rows`` is the
    ``(D,)`` boolean mask of log rows. The bands are dex half-widths, so a LINEAR row gets NaN:
    its estimator-space error is an absolute difference, not a dex, and a dex tolerance does not
    apply to it. Returns a list of ``(dex, fractions)`` pairs, one per band, ``fractions`` ``(D,)``.
    """
    err = np.abs(np.asarray(inferred_flow, dtype=float) - np.asarray(true_flow, dtype=float))
    log_rows = np.asarray(log_rows, dtype=bool)
    out = []
    for b in bands:
        frac = np.mean(err <= float(b), axis=0)
        out.append((float(b), np.where(log_rows, frac, np.nan)))
    return out

CENTRAL_FAMILIES = ("sgm", "mean")


def reshape_to_grid(values, kind_index, cell, chunk, n_kinds):
    """Scatter flat per-window rows into a dense ``(kind, cell, chunk, ...)`` grid.

    The Experiment output stores one flat row per analyzed window. Each row is placed at
    ``grid[kind_index, cell, chunk]``. Windows never estimated stay NaN and every statistic here is
    nan-aware, so a missing window narrows nothing silently.

    Returns ``(grid, n_cells, n_chunks)``.
    """
    values = np.asarray(values, dtype=float)
    kind_index = np.asarray(kind_index, dtype=int)
    cell = np.asarray(cell, dtype=int)
    chunk = np.asarray(chunk, dtype=int)
    n_cells = int(cell.max()) + 1
    n_chunks = int(chunk.max()) + 1
    grid = np.full((n_kinds, n_cells, n_chunks) + values.shape[1:], np.nan, dtype=float)
    grid[kind_index, cell, chunk] = values
    return grid, n_cells, n_chunks


def _range_abs(prior_low, prior_high, to_physical):
    """Physical prior width per parameter -- the normalizer for every distance here.

    The bounds are in estimator space; ``to_physical`` (the runner's table-bound conversion) maps
    them row by row, so a log row's width is ``10**high - 10**low`` and a linear row's is
    ``high - low``.
    """
    lo = np.asarray(prior_low, dtype=float)
    hi = np.asarray(prior_high, dtype=float)
    span = np.asarray(to_physical(hi), dtype=float) - np.asarray(to_physical(lo), dtype=float)
    span[span <= 0] = 1.0
    return span


# =============================================================================
# The four central estimates
# =============================================================================

def mean_window(grid_flow, to_physical):
    """**mean-window**: mean value vector, aggregated across cells for a given chunk.

    For each condition, chunk, and parameter independently, the arithmetic mean over cells of the
    physical MAP values (``to_physical`` of the estimator-space grid). Each parameter is averaged
    on its own, so the resulting vector's coordinates need not have co-occurred in any recording.

    Returns ``(n_kinds, n_chunks, D)`` in PHYSICAL units.
    """
    return np.nanmean(to_physical(np.asarray(grid_flow, dtype=float)), axis=1)


def sgm_window(grid_flow, prior_low, prior_high, to_physical):
    """**sgm-window**: realized value vector, aggregated across cells for a given chunk.

    At each chunk independently, the exact medoid among that chunk's cell vectors: the cell whose
    D-dimensional MAP vector minimizes the summed normalized distance to the other cells' vectors at
    the same chunk. The returned vector is that cell's stored values verbatim.

    The selected cell is returned per chunk because **it may differ between chunks**: a step in the
    resulting series can be a change of cell rather than a change in time, and the caller must show
    the selections wherever it shows the curve.

    Returns ``(series_abs, cells)`` -- ``(n_kinds, n_chunks, D)`` in PHYSICAL units and
    ``(n_kinds, n_chunks)`` selected cell indices (-1 where a chunk has no complete cell).
    """
    grid = np.asarray(grid_flow, dtype=float)
    n_kinds, n_cells, n_chunks, dim = grid.shape
    span = _range_abs(prior_low, prior_high, to_physical)
    out = np.full((n_kinds, n_chunks, dim), np.nan)
    picked = np.full((n_kinds, n_chunks), -1, dtype=int)
    for k in range(n_kinds):
        for t in range(n_chunks):
            rows = [c for c in range(n_cells) if np.isfinite(grid[k, c, t]).all()]
            if not rows:
                continue
            block = to_physical(np.stack([grid[k, c, t] for c in rows], axis=0))
            idx, _method = sample_geometric_median(block, span)
            picked[k, t] = rows[idx]
            out[k, t] = block[idx]
    return out, picked


def mean_trajectory(grid_flow, to_physical):
    """**mean-trajectory**: mean value vector, aggregated across chunks and cells.

    For each condition and parameter independently, the arithmetic mean over every (cell, chunk)
    window of the physical MAP values -- the grand mean over both axes. This is the single-vector
    counterpart of :func:`mean_window`.

    Returns ``(n_kinds, D)`` in PHYSICAL units.
    """
    g = to_physical(np.asarray(grid_flow, dtype=float))
    return np.nanmean(g.reshape(g.shape[0], -1, g.shape[3]), axis=1)


def sgm_trajectory(grid_flow, prior_low, prior_high, to_physical):
    """**sgm-trajectory**: realized value vector, aggregated across chunks and cells.

    The exact medoid among ALL (cell, chunk) window vectors of a condition: the single window whose
    D-dimensional MAP vector minimizes the summed normalized distance to every other window vector.
    The result is one genuinely realized vector, drawn from one specific cell at one specific chunk,
    and it is the single-vector counterpart of :func:`sgm_window`.

    This is the same quantity the standalone sample-geometric-median analysis reports over the same
    pooled windows, so the two analyses agree by construction.

    Returns ``(vector_abs, picks)`` -- ``(n_kinds, D)`` in PHYSICAL units and a list of
    ``(cell, chunk)`` tuples naming the selected window per condition (``(-1, -1)`` if none).
    """
    grid = np.asarray(grid_flow, dtype=float)
    n_kinds, n_cells, n_chunks, dim = grid.shape
    span = _range_abs(prior_low, prior_high, to_physical)
    out = np.full((n_kinds, dim), np.nan)
    picks = []
    for k in range(n_kinds):
        rows, labels = [], []
        for c in range(n_cells):
            for t in range(n_chunks):
                if np.isfinite(grid[k, c, t]).all():
                    rows.append(to_physical(grid[k, c, t]))
                    labels.append((c, t))
        if not rows:
            picks.append((-1, -1))
            continue
        block = np.stack(rows, axis=0)
        idx, _method = sample_geometric_median(block, span)
        out[k] = block[idx]
        picks.append(labels[idx])
    return out, picks


# =============================================================================
# Drift statistics -- fit per cell, so independent of the central-estimate choice
# =============================================================================

def pooled_cloud(cloud_flow, kind_index, n_kinds):
    """Pool every window's posterior draws within each condition, collapsing the time axis.

    ``cloud_flow`` is ``(N, S, D)`` in estimator space: for each of the ``N`` (recording, window)
    pairs, the ``S`` draws the Experiment stage took from that window's posterior. Selecting one
    condition and flattening the leading two axes gives ``(n_windows * S, D)`` draws.

    WHAT THIS IS. The pooled set is the equal-weight MIXTURE of the per-window posteriors, so its
    density answers: for a 2 s window drawn at random from this condition, what values are
    consistent with it? It is emphatically NOT a joint posterior for the condition. Combining
    independent observations under Bayes multiplies their likelihoods; pooling draws adds their
    densities. The mixture is therefore as wide as the spread BETWEEN windows plus the uncertainty
    WITHIN each one, whereas a genuine joint posterior over 250 windows would be far narrower than
    any single one of them. Read it as a description of the window-to-window population, never as
    evidence accumulated across the recording.

    Because the mixture is a population and not an estimate, its mode and median are not the
    analysis's central estimate: that remains the trajectory-level medoid, which is one jointly
    realized vector rather than a per-parameter summary of pooled marginals.

    Returns a list of ``n_kinds`` arrays, each ``(n_windows_k * S, D)`` in estimator space.
    """
    cloud_flow = np.asarray(cloud_flow, dtype=float)
    kind_index = np.asarray(kind_index, dtype=int)
    out = []
    for e in range(n_kinds):
        sel = cloud_flow[kind_index == e]                      # (n_windows_k, S, D)
        out.append(sel.reshape(-1, sel.shape[-1]) if sel.size else sel.reshape(0, 0))
    return out


def cloud_interval(pooled, quantiles=(0.05, 0.50, 0.95)):
    """Per-parameter quantiles of a pooled mixture, for the report table.

    Marginal quantiles of a mixture, reported per parameter: they describe each coordinate's spread
    across the condition, and -- being per-coordinate -- they carry no joint information, which is
    exactly why they belong in a table beside the medoid rather than replacing it.
    """
    if pooled.size == 0:
        return np.full((3, 0), np.nan)
    return np.quantile(pooled, list(quantiles), axis=0)         # (len(quantiles), D)


def pooled_summary(pooled, to_physical, statistic="median"):
    """Marginal central value of a pooled mixture, per parameter, in PHYSICAL units.

    The 2x2 of central estimates -- ``{mean, sgm} x {window, trajectory}`` -- all summarize the set
    of per-window MAP vectors. This summarizes the pooled posterior DRAWS instead, which is a
    different object: the mixture is where the probability mass is, while the MAP set is where the
    per-window modes are, and the two need not agree. Naming follows the same scheme:
    ``median-pooled`` and ``mean-pooled``.

    WHY THIS EXISTS. A vertical line drawn on a one-dimensional marginal histogram is read as the
    center of THAT histogram. The trajectory-level medoid is not: it is one jointly realized vector
    chosen to minimize distance in the full parameter space, so any single coordinate of it can sit
    far from that coordinate's marginal center -- correctly, but not in a way a marginal plot
    communicates. These statistics are the marginal centers of the plotted distribution, and are the
    honest thing to mark on it.

    WHAT IS LOST. Being per-coordinate, they are composites: the returned vector's coordinates come
    from different draws and need never have co-occurred, which is precisely the defect the geometric
    median exists to avoid. Use them to describe a marginal, never as the condition's parameter
    vector.

    ``statistic``:
        ``"median"``  the marginal median, taken here AFTER the transform (the median of the
                      physical draws). For an odd count it is the same draw as
                      ``to_physical(median(estimator))``. For an even count numpy interpolates
                      between the two middle values, and interpolating physical values (their
                      arithmetic mean) differs from transforming the interpolated estimator value
                      (their geometric mean, for a log row), so the two conventions need not agree
                      in a finite sample. The per-observation median stored in the stage products
                      is taken the other way (in estimator coordinates, then transformed); it
                      summarizes a different population and is not compared with this one number
                      for number. The median stays the default because it depends on the basis
                      only through that interpolation, whereas the mean differs by a factor that
                      grows with the spread.
        ``"mean"``    the arithmetic mean in physical units. NOT equivariant for a log row: it
                      differs from the geometric mean ``10 ** mean(log10)`` by a factor that grows
                      with the spread, and for a log-uniform prior that factor is large. Provided
                      because it is what "the mean of the posterior" usually names, but it reports
                      a property of the chosen basis as much as of the posterior.
        ``"geometric-mean"`` ``to_physical(mean(estimator))``: the mean in the space the prior is
                      uniform in -- the geometric mean for a log row, and simply the arithmetic mean
                      for a linear row, where the two coincide.

    Returns an ``(n_kinds, D)`` array in physical units.
    """
    if statistic not in ("median", "mean", "geometric-mean"):
        raise ValueError(f"statistic={statistic!r}; expected 'median', 'mean' or 'geometric-mean'.")
    out = []
    for draws_flow in pooled:
        if draws_flow.size == 0:
            out.append(np.full(draws_flow.shape[-1:], np.nan))
            continue
        if statistic == "median":
            out.append(np.median(to_physical(draws_flow), axis=0))
        elif statistic == "mean":
            out.append(np.mean(to_physical(draws_flow), axis=0))
        else:
            out.append(to_physical(np.mean(draws_flow, axis=0)))
    return np.asarray(out)


def drift_statistics(grid_flow, times, to_physical, log_rows, threshold=MATERIAL_DRIFT_DEX):
    """Per-cell linear drift of each parameter across the recording, aggregated per condition.

    For every (condition, cell, parameter) an ordinary least-squares line is fit to the stored
    estimator-space MAP estimate against time -- log10 for log rows, because their drift is
    multiplicative; the value itself for the linear initial dimer fraction -- giving a slope and
    hence a fitted start and end value:

        change     = slope * (t_last - t_first)           (dex for a log row; absolute for a linear row)
        start      = to_physical(fitted value at t_first)
        end        = to_physical(fitted value at t_last)

    Every reported statistic aggregates those per-cell fits, so **none of them depends on which
    central estimate the figures display**: swapping a mean for a medoid changes what is drawn, not
    what is measured. Named results, all shaped ``(n_kinds, D)`` unless noted:

        drift_absolute          median across cells of (end - start), in the parameter's own units
        drift_sign_consistency  fraction of cells whose change shares the median's sign
        drift_fold              median across cells of (end / start), a multiplicative factor
        drift_dex               median across cells of change, in log10 units -- LOG ROWS ONLY
                                (NaN for a linear row, whose change is drift_absolute already)
        drift_material_fraction fraction of cells whose |change| exceeds ``threshold`` dex -- LOG
                                ROWS ONLY (NaN for a linear row: a dex threshold does not apply)
        drift_wilcoxon_p        two-sided signed-rank p that the per-cell change is centered at
                                zero -- a DETECTABILITY statement, not a magnitude; NaN when scipy
                                is unavailable or fewer than six cells contribute
        start_median, end_median  median across cells of the fitted endpoints, physical units
        change_dex_per_cell     ``(n_kinds, n_cells, D)`` the underlying per-cell changes, in
                                estimator-space units (dex for log rows, absolute for linear rows)
        log_rows                the ``(D,)`` mask that says which rows the dex statistics cover
    """
    grid = np.asarray(grid_flow, dtype=float)
    t = np.asarray(times, dtype=float)
    log_rows = np.asarray(log_rows, dtype=bool)
    n_kinds, n_cells, n_chunks, dim = grid.shape
    span = float(t[-1] - t[0]) if n_chunks > 1 else 0.0
    change = np.full((n_kinds, n_cells, dim), np.nan)
    fit_start = np.full((n_kinds, n_cells, dim), np.nan)     # fitted endpoints, estimator space
    fit_end = np.full((n_kinds, n_cells, dim), np.nan)
    for k in range(n_kinds):
        for c in range(n_cells):
            for p in range(dim):
                y = grid[k, c, :, p]
                ok = np.isfinite(y)
                if ok.sum() < 2:
                    continue
                slope, intercept = np.polyfit(t[ok], y[ok], 1)
                d = slope * span
                change[k, c, p] = d
                fit_start[k, c, p] = intercept + slope * t[0]
                fit_end[k, c, p] = intercept + slope * t[0] + d
    # The endpoints are converted as whole D-vectors so the per-row rule comes from the table.
    start = to_physical(fit_start)
    end = to_physical(fit_end)
    median_dex = np.nanmedian(change, axis=1)
    # Sign consistency and the material share are fractions of the recordings that contributed a
    # fitted change (the same cells the table counts), never of the grid's cell slots: a missing
    # recording, a deselected cell or a failed estimate must not enter the denominator as a
    # "no drift" vote. A comparison with NaN evaluates to False, so the mask is applied first.
    finite = np.isfinite(change)
    with np.errstate(invalid="ignore", divide="ignore"):
        agree = np.where(finite, np.sign(change) == np.sign(median_dex)[:, None, :], np.nan)
        over = np.where(finite, np.abs(change) > threshold, np.nan)
        sign = np.nanmean(agree, axis=1)
        material = np.nanmean(over, axis=1)
        fold = np.nanmedian(end / start, axis=1)
    # Dex statistics are undefined for a linear row: flag with NaN rather than report a number that
    # only looks like a dex.
    median_dex = np.where(log_rows, median_dex, np.nan)
    material = np.where(log_rows, material, np.nan)
    pvals = np.full((n_kinds, dim), np.nan)
    try:
        from scipy.stats import wilcoxon
        for k in range(n_kinds):
            for p in range(dim):
                v = change[k, :, p]
                v = v[np.isfinite(v)]
                if v.size >= 6 and np.any(v != 0):
                    pvals[k, p] = float(wilcoxon(v)[1])
    except ImportError:
        pass
    return {
        "drift_absolute": np.nanmedian(end - start, axis=1),
        "drift_sign_consistency": sign,
        "drift_fold": fold,
        "drift_dex": median_dex,
        "drift_material_fraction": material,
        "drift_wilcoxon_p": pvals,
        "start_median": np.nanmedian(start, axis=1),
        "end_median": np.nanmedian(end, axis=1),
        "change_dex_per_cell": change,
        "threshold": float(threshold),
        "log_rows": log_rows,
    }


def within_window_interval(quant_grid_flow):
    """Median across cells, per chunk, of the stored per-window posterior quantile levels.

    Summarizes how uncertain a TYPICAL SINGLE window's estimate is: for each condition, chunk, and
    quantile level, the median over cells of that level. Taking the median of a level across cells
    (rather than pooling) keeps the reported interval the interval of one typical window instead of
    an envelope over recordings.

    This is an interval WIDTH summary of the stored five-quantile record, not a posterior density:
    a density pooled over windows would require the per-window sample clouds.

    Returns ``(n_kinds, n_chunks, D, Q)`` in estimator space (the caller converts).
    """
    with warnings.catch_warnings():
        # A level the run did not store (a direct estimator's single nominal range has no
        # 25/75 % levels) is an all-NaN slice; NaN is the intended answer, not a warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.asarray(quant_grid_flow, dtype=float), axis=1)


# --------------------------------------------------------------------------------------------
# Window drift for every stored point estimate, read together
# --------------------------------------------------------------------------------------------
POINT_ESTIMATE_ORDER = ("MAP", "posterior median", "SGM")


def point_estimate_grids(map_estimate, kind_index, cell, chunk, n_kinds,
                         post_quantiles, post_sgm, *, median_index):
    """Dense ``(kind, cell, chunk, D)`` grids for the three point estimates of a product.

    All three are required: ``map_estimate``; the marginal median, the ``median_index`` level
    of ``post_quantiles`` (located from the product manifest, never assumed); and ``post_sgm``.
    Returns ``(grids, n_cells, n_chunks)`` with ``grids`` an ordered dict keyed by the display
    tokens MAP, posterior median, SGM.
    """
    grids = {}
    g, n_cells, n_chunks = reshape_to_grid(map_estimate, kind_index, cell, chunk, n_kinds)
    grids["MAP"] = g
    q = np.asarray(post_quantiles, dtype=float)
    if q.size:
        grids["posterior median"] = reshape_to_grid(q[:, :, median_index], kind_index, cell,
                                                    chunk, n_kinds)[0]
    s = None if post_sgm is None else np.asarray(post_sgm, dtype=float)
    if s is not None and s.size:
        grids["SGM"] = reshape_to_grid(s, kind_index, cell, chunk, n_kinds)[0]
    return grids, n_cells, n_chunks


def window_medians(grid):
    """Median across cells of one estimate, per (kind, chunk, D): the line a drift figure draws."""
    return np.nanmedian(np.asarray(grid, dtype=float), axis=1)


def shared_posterior_bands(post_quantiles, kind_index, cell, chunk, n_kinds):
    """Median across cells, per (kind, chunk, D, Q), of the stored per-window posterior quantiles.

    All point estimates come from the same posterior draws, so the interval that accompanies them
    is the posterior's own and is drawn ONCE, not once per estimate (``within_window_interval``).
    """
    qgrid, _, _ = reshape_to_grid(post_quantiles, kind_index, cell, chunk, n_kinds)
    return within_window_interval(qgrid)


def window_drift_rows(grids, kinds, keys, to_physical, log_rows, threshold=MATERIAL_DRIFT_DEX):
    """Drift statistics for every stored point estimate, one row per (kind, parameter, estimate).

    Each row aggregates the per-cell first-to-last change of that estimate across the windows of
    a recording (``drift_statistics`` fitted against the window index, so the change equals the
    fitted difference between the last and the first window):

        change_median     median across cells of the change (dex for a log row, absolute otherwise)
        change_iqr        interquartile range of the per-cell changes across cells
        sign_consistency  fraction of cells whose change shares the median's sign
        material_fraction fraction of cells with |change| > ``threshold`` dex (log rows; NaN otherwise)
        wilcoxon_p        two-sided signed-rank p that the per-cell changes are centered at zero

    The three estimates come from the same posterior, so a difference between their rows is a
    statement about the posterior's shape along the recording, not about three measurements.
    """
    rows = []
    first = next(iter(grids.values()))
    n_chunks = first.shape[2]
    times = np.arange(n_chunks, dtype=float)
    stats = {name: drift_statistics(g, times, to_physical, log_rows, threshold)
             for name, g in grids.items()}
    # Neural estimates in their fixed order, then any other estimate a caller supplies (a
    # direct estimator's single point value is one such grid).
    order = [n for n in POINT_ESTIMATE_ORDER if n in stats] + \
            [n for n in stats if n not in POINT_ESTIMATE_ORDER]
    for k, kind in enumerate(kinds):
        for p, key in enumerate(keys):
            for name in order:
                d = stats[name]
                change = d["change_dex_per_cell"][k, :, p]
                ok = np.isfinite(change)
                q75, q25 = (np.nanpercentile(change, [75, 25]) if ok.sum() > 1
                            else (np.nan, np.nan))
                rows.append({
                    "kind": kind, "parameter": key, "estimate": name,
                    "n_cells": int(ok.sum()),
                    "change_median": float(np.nanmedian(change)) if ok.any() else np.nan,
                    "change_iqr": float(q75 - q25),
                    "sign_consistency": float(d["drift_sign_consistency"][k, p]),
                    "material_fraction": float(d["drift_material_fraction"][k, p]),
                    "wilcoxon_p": float(d["drift_wilcoxon_p"][k, p]),
                    "is_log": bool(log_rows[p]),
                })
    return rows


def format_drift_rows(rows):
    """Render ``window_drift_rows`` output as report-table headers and string rows."""
    headers = ["condition", "parameter", "estimate", "cells", "change first->last (median)",
               "IQR across cells", "sign consistency", "cells over 0.3 dex", "signed-rank p"]
    out = []
    for r in rows:
        unit = "dex" if r["is_log"] else "abs"
        out.append([r["kind"], r["parameter"], r["estimate"], str(r["n_cells"]),
                    f"{r['change_median']:+.3f} {unit}", f"{r['change_iqr']:.3f}",
                    f"{100 * r['sign_consistency']:.0f}%",
                    ("n/a" if not np.isfinite(r["material_fraction"])
                     else f"{100 * r['material_fraction']:.0f}%"),
                    ("n/a" if not np.isfinite(r["wilcoxon_p"]) else f"{r['wilcoxon_p']:.1e}")])
    return headers, out
