"""Inference-stage diagnostics and interactive plotting.

Functions:
    plot_loss_curves(losses_train, losses_test, losses_replay)
        Plot training, validation, and replay loss curves over epochs.

Matplotlib is imported lazily inside the function so HPC headless runs
that never plot don't pay the import cost.

Posterior-level diagnostics (corner plots, posterior-predictive video
overlays, etc.) can be added here as further analyses are developed.
"""

import numpy as np

# Shared with the Evaluation report so a tolerance is written the same way in the table and on the
# figure; derived from the guide value passed in, never hardcoded, so a reconfigured band stays true.
from .temporal_dynamics import band_label


def plot_loss_curves(losses_train: np.ndarray,
                     losses_test: np.ndarray,
                     losses_replay: np.ndarray) -> None:
    """Plot training, validation, and replay loss curves on a single axis.

    "Replay" loss is the training loss recomputed with data augmentation
    disabled — a proxy for the loss on the augmentation-free distribution
    that the validation loss measures, but computed on the training set.
    Useful for diagnosing whether the train-vs-validation gap is
    augmentation-driven or genuine overfitting.

    Args:
        losses_train: 1D array of per-epoch training loss.
        losses_test: 1D array of per-epoch validation loss (same length).
        losses_replay: 1D array of per-epoch replay loss (same length).
    """
    import matplotlib.pyplot as plt

    epochs = np.arange(len(losses_train))
    plt.figure()
    plt.plot(epochs, losses_train, label="Train loss")
    plt.plot(epochs, losses_test, label="Test loss")
    plt.plot(epochs, losses_replay, label="[Replay] Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, test, and replay losses over epochs")
    plt.legend()
    plt.show()


# =============================================================================
# Headless figure builder (for --debug-dump reports)
# =============================================================================


def figure_loss_curves(losses_train: np.ndarray,
                       losses_test: np.ndarray,
                       losses_replay: np.ndarray):
    """Headless counterpart to ``plot_loss_curves``: build and return a Figure.

    Builds the train / test / replay loss-curve plot via
    ``matplotlib.figure.Figure`` (no display needed) and returns it for the
    DiagnosticReporter to save as PNG.

    Args:
        losses_train: 1D array of per-epoch training loss.
        losses_test: 1D array of per-epoch validation loss (same length).
        losses_replay: 1D array of per-epoch replay loss (same length).

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    epochs = np.arange(len(losses_train))
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(epochs, losses_train, label="Train loss")
    ax.plot(epochs, losses_test, label="Test loss")
    ax.plot(epochs, losses_replay, label="[Replay] Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training, test, and replay losses over epochs")
    ax.legend()
    return fig


# =============================================================================
# MAP-recovery figure builders (Evaluation stage)
# =============================================================================


def _conditional_quantiles(x: np.ndarray, value: np.ndarray, n_bins: int,
                           min_count: int, bin_mode: str = "quantile",
                           prior_range=None):
    """Conditional quantiles of ``value`` over bins of ``x``.

    Splits ``x`` into ``n_bins`` bins and, for each bin holding at least
    ``min_count`` points, computes the Q05/Q25/Q50/Q75/Q95 of ``value``. Bins
    below the threshold are left as NaN, so a band is drawn only where the sample
    is dense enough (graceful degradation for small EVAL).

    ``bin_mode``:
        ``"quantile"`` -- bin edges at data quantiles of ``x`` (equal-count bins;
                          the default).
        ``"prior"``    -- bin edges uniform across ``prior_range`` (equal-width
                          bins over the prior bounds); falls back to quantile bins
                          if ``prior_range`` is None.

    Returns:
        ``(centers, q05, q25, q50, q75, q95)`` — each a length-``n_bins`` array;
        sparse bins are NaN.
    """
    if bin_mode == "prior" and prior_range is not None:
        edges = np.linspace(prior_range[0], prior_range[1], n_bins + 1)
    else:
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    quantiles = {q: np.full(n_bins, np.nan) for q in (5, 25, 50, 75, 95)}
    bin_id = np.clip(np.digitize(x, edges, right=True) - 1, 0, n_bins - 1)
    for k in range(n_bins):
        vk = value[bin_id == k]
        if vk.size >= min_count:
            q05, q25, q50, q75, q95 = np.quantile(vk, [0.05, 0.25, 0.50, 0.75, 0.95])
            quantiles[5][k], quantiles[25][k], quantiles[50][k] = q05, q25, q50
            quantiles[75][k], quantiles[95][k] = q75, q95
    return (centers, quantiles[5], quantiles[25], quantiles[50],
            quantiles[75], quantiles[95])


def _overlay_quantile_bands(ax, centers, q05, q25, q50, q75, q95):
    """Overlay the Q05-Q95 / Q25-Q75 bands and the Q50 median, where present."""
    ok = np.isfinite(q50)
    if not np.any(ok):
        return False
    ax.fill_between(centers[ok], q05[ok], q95[ok], alpha=0.50, linewidth=0,
                    color="tab:red", label="Q05-Q95")
    ax.fill_between(centers[ok], q25[ok], q75[ok], alpha=0.75, linewidth=0,
                    color="tab:green", label="Q25-Q75")
    ax.plot(centers[ok], q50[ok], alpha=0.95, linewidth=1,
            color="tab:orange", label="Q50")
    return True




def _draw_recovery_axis(ax, true_log10, map_estimate, prior_range=None,
                        n_bins=20, min_count=50, bin_mode="quantile"):
    """Draw the recovery scatter (true vs. inferred, log10) + bands on ``ax``.

    Points on the dashed identity line are perfectly recovered; conditional
    quantile bands of inferred-given-true overlay where a bin is dense enough.
    ``bin_mode`` selects data-quantile vs prior-range bin edges.
    """
    x = np.asarray(true_log10, dtype=float).ravel()
    y = np.asarray(map_estimate, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    limits = (tuple(prior_range) if prior_range is not None
              else (float(np.floor(np.min(x))), float(np.ceil(np.max(x)))))
    ax.axline((limits[0], limits[0]), (limits[1], limits[1]),
              color="k", linestyle="--", alpha=0.75)
    ax.scatter(x, y, s=7, color="tab:blue", alpha=0.25)
    centers, q05, q25, q50, q75, q95 = _conditional_quantiles(
        x, y, n_bins, min_count, bin_mode, prior_range)
    bands = _overlay_quantile_bands(ax, centers, q05, q25, q50, q75, q95)
    ax.set_xlabel(r"true [$\log_{10}$]")
    ax.set_ylabel(r"inferred [$\log_{10}$]")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    if bands:
        ax.legend(fontsize=9, frameon=False)
    else:
        ax.text(0.5, 0.02, f"bands sparse (n<{min_count}/bin) — scatter only",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color="tab:red", alpha=0.8)
    ax.set_title("MAP: inferred against true", fontsize=11)


def _draw_error_axis(ax, true_log10, map_estimate, prior_range=None,
                     n_bins=20, min_count=50, error_guide=0.3,
                     error_guide_tight=0.15, error_ylim_floor=0.5,
                     error_ylim_quantile=0.95, bin_mode="quantile"):
    """Draw the residual-error view (inferred - true, log10) + bands on ``ax``.

    A zero line marks perfect recovery. Two nested tolerance bands are drawn as
    +/- guide lines, each the log10 of a linear accuracy factor:
    ``+/- error_guide`` (0.3 ~= log10(2): within a factor of 2) and the tighter
    ``+/- error_guide_tight`` (0.15 ~= log10(sqrt(2)): within a factor of ~1.41).
    The y-axis spans ``+/- max(error_ylim_floor, quantile(|error|,
    error_ylim_quantile))``.
    """
    x = np.asarray(true_log10, dtype=float).ravel()
    y = np.asarray(map_estimate, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    error = y - x
    x_limits = (tuple(prior_range) if prior_range is not None
                else (float(np.floor(np.min(x))), float(np.ceil(np.max(x)))))
    half = max(error_ylim_floor, float(np.quantile(np.abs(error), error_ylim_quantile)))
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.75)
    # Factor-2 band (0.3 = log10 2) and the tighter factor-sqrt(2) band (0.15 = log10 sqrt 2).
    ax.axhline(+error_guide, color="k", linestyle=":", alpha=0.60,
               label=rf"$\pm${error_guide:g} = {band_label(error_guide)}")
    ax.axhline(-error_guide, color="k", linestyle=":", alpha=0.60)
    ax.axhline(+error_guide_tight, color="k", linestyle=(0, (1, 3)), alpha=0.45,
               label=rf"$\pm${error_guide_tight:g} = {band_label(error_guide_tight)}")
    ax.axhline(-error_guide_tight, color="k", linestyle=(0, (1, 3)), alpha=0.45)
    ax.scatter(x, error, s=7, color="tab:blue", alpha=0.25)
    centers, q05, q25, q50, q75, q95 = _conditional_quantiles(
        x, error, n_bins, min_count, bin_mode, prior_range)
    bands = _overlay_quantile_bands(ax, centers, q05, q25, q50, q75, q95)
    ax.set_xlabel(r"true [$\log_{10}$]")
    ax.set_ylabel(r"error = inferred - true [$\log_{10}$]")
    ax.set_xlim(x_limits)
    ax.set_ylim((-half, half))
    if bands:
        ax.legend(fontsize=9, frameon=False)
    else:
        ax.text(0.5, 0.02, f"bands sparse (n<{min_count}/bin) — scatter only",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color="tab:red", alpha=0.8)
    ax.set_title("MAP: error (inferred - true)", fontsize=11)


def _draw_posterior_recovery_axis(ax, true_log10, map_inferred, post_median,
                                  post_q25, post_q75, prior_range=None):
    """Draw true against the marginal median +/- IQR of the draws, with the MAP overlaid, on ``ax``.

    Both estimators are shown so the posterior median is not mistaken for the
    point estimate: blue circles + bars are the posterior median and IQR
    (credible width); orange crosses are the MAP mode.
    """
    x = np.asarray(true_log10, dtype=float).ravel()
    mp = np.asarray(map_inferred, dtype=float).ravel()
    med = np.asarray(post_median, dtype=float).ravel()
    q25 = np.asarray(post_q25, dtype=float).ravel()
    q75 = np.asarray(post_q75, dtype=float).ravel()
    mask = (np.isfinite(x) & np.isfinite(med) & np.isfinite(q25)
            & np.isfinite(q75) & np.isfinite(mp))
    x, mp, med, q25, q75 = x[mask], mp[mask], med[mask], q25[mask], q75[mask]
    limits = (tuple(prior_range) if prior_range is not None
              else (float(np.floor(np.min(x))), float(np.ceil(np.max(x)))))
    yerr = np.vstack([np.clip(med - q25, 0, None), np.clip(q75 - med, 0, None)])
    ax.axline((limits[0], limits[0]), (limits[1], limits[1]),
              color="k", linestyle="--", alpha=0.75)
    ax.errorbar(x, med, yerr=yerr, fmt="o", ms=4, color="tab:blue",
                ecolor="tab:blue", elinewidth=1, alpha=0.5, capsize=2,
                label="posterior median $\\pm$ IQR")
    ax.scatter(x, mp, marker="x", s=28, color="tab:orange", alpha=0.8,
               zorder=3, label="MAP")
    ax.set_xlabel(r"true [$\log_{10}$]")
    ax.set_ylabel(r"inferred [$\log_{10}$]")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("median +/- IQR of draws, MAP overlaid", fontsize=11)


def figure_recovery_combined(true_log10, map_estimate, post_q, median_index,
                             prior_range=None, label="", n_bins=20, min_count=50,
                             error_guide=0.3, error_guide_tight=0.15,
                             error_ylim_floor=0.5,
                             error_ylim_quantile=0.95, bin_mode="quantile"):
    """Recovery figure for one parameter, three panels: MAP against truth, MAP error, and the
    marginal median +/- IQR of the draws against truth with the MAP overlaid.

    ``post_q`` is the per-observation ``(N, Q)`` quantile array; ``median_index`` locates the
    0.50 level along its last axis (from the product manifest -- never assumed), and the IQR is
    the two levels adjacent to it. Every panel is always drawn: a product carries all three
    estimates by contract, so there is no absent view to stamp.

    Returns:
        A headless ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 5.2))
    ax_rec, ax_err, ax_post = (fig.add_subplot(1, 3, k) for k in (1, 2, 3))
    _draw_recovery_axis(ax_rec, true_log10, map_estimate, prior_range,
                        n_bins, min_count, bin_mode)
    _draw_error_axis(ax_err, true_log10, map_estimate, prior_range, n_bins,
                     min_count, error_guide=error_guide,
                     error_guide_tight=error_guide_tight,
                     error_ylim_floor=error_ylim_floor,
                     error_ylim_quantile=error_ylim_quantile, bin_mode=bin_mode)
    q = np.asarray(post_q, dtype=float)
    _draw_posterior_recovery_axis(ax_post, true_log10, map_estimate,
                                  q[:, median_index], q[:, median_index - 1],
                                  q[:, median_index + 1], prior_range)
    if label:
        fig.suptitle(label, fontsize=13)
    return fig


def _draw_experiment_distribution_axis(ax, values_by_kind, prior_range=None, seed=0):
    """Draw the per-condition distribution of the MAP for one parameter on ``ax``.

    No ground truth for real data, so this shows the *distribution* of the inferred
    MAP value across all (cell, chunk) estimates -- one box per condition (kind),
    with jittered points overlaid; dashed lines mark the prior bounds. Comparing
    conditions (e.g. ALP vs BET) is the scientific read-out.
    """
    kinds = list(values_by_kind.keys())
    data = [np.asarray(values_by_kind[k], dtype=float).ravel() for k in kinds]
    data = [d[np.isfinite(d)] for d in data]

    if prior_range is not None:
        ax.axhline(prior_range[0], color="k", linestyle=":", alpha=0.5)
        ax.axhline(prior_range[1], color="k", linestyle=":", alpha=0.5)
        ax.set_ylim(prior_range[0] - 0.2, prior_range[1] + 0.2)

    positions = list(range(1, len(kinds) + 1))
    # Boxplot needs non-empty sequences; substitute a NaN so empty kinds keep a slot.
    ax.boxplot([d if d.size else np.array([np.nan]) for d in data],
               positions=positions, showfliers=False, widths=0.5)
    rng = np.random.RandomState(seed)
    for pos, d in zip(positions, data):
        if d.size:
            jitter = (rng.rand(d.size) - 0.5) * 0.18
            ax.scatter(np.full(d.size, pos) + jitter, d, s=6, alpha=0.3,
                       color="tab:blue")
    ax.set_xticks(positions)
    ax.set_xticklabels(kinds)
    ax.set_xlabel("experimental condition")
    ax.set_ylabel(r"inferred [$\log_{10}$]")
    ax.set_title("MAP: distribution per condition", fontsize=11)


def _draw_experiment_posterior_axis(ax, by_kind, prior_range=None, seed=0):
    """Draw each window's marginal median +/- IQR of the draws, with its MAP, per condition.

    Each chunk is drawn as its **posterior median** with **IQR (Q25-Q75)** error
    bars (blue) and its **MAP** (orange cross), jittered within its condition --
    showing within-chunk posterior uncertainty, the point estimate, and cross-chunk
    spread, one group per condition. ``by_kind`` maps each kind to an ``(n, 4)``
    array ``[MAP, median, q25, q75]`` (log10). Dashed lines mark the prior bounds.
    """
    kinds = list(by_kind.keys())
    if prior_range is not None:
        ax.axhline(prior_range[0], color="k", linestyle=":", alpha=0.5)
        ax.axhline(prior_range[1], color="k", linestyle=":", alpha=0.5)
        ax.set_ylim(prior_range[0] - 0.2, prior_range[1] + 0.2)
    rng = np.random.RandomState(seed)
    for pos, kind in enumerate(kinds, start=1):
        arr = np.asarray(by_kind[kind], dtype=float)
        if arr.size == 0:
            continue
        mp, med, q25, q75 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        m = np.isfinite(med) & np.isfinite(q25) & np.isfinite(q75) & np.isfinite(mp)
        mp, med, q25, q75 = mp[m], med[m], q25[m], q75[m]
        if med.size == 0:
            continue
        x = np.full(med.size, pos) + (rng.rand(med.size) - 0.5) * 0.3
        yerr = np.vstack([np.clip(med - q25, 0, None), np.clip(q75 - med, 0, None)])
        labels = ({"med": "posterior median $\\pm$ IQR", "map": "MAP"}
                  if pos == 1 else {"med": None, "map": None})
        ax.errorbar(x, med, yerr=yerr, fmt="o", ms=4, color="tab:blue",
                    ecolor="tab:gray", elinewidth=1, alpha=0.5, capsize=2,
                    linestyle="none", label=labels["med"])
        ax.scatter(x, mp, marker="x", s=24, color="tab:orange", alpha=0.8,
                   zorder=3, label=labels["map"])
    ax.set_xticks(range(1, len(kinds) + 1))
    ax.set_xticklabels(kinds)
    ax.set_xlabel("experimental condition")
    ax.set_ylabel(r"inferred [$\log_{10}$]")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("median +/- IQR of draws, MAP overlaid", fontsize=11)


def figure_experiment_combined(values_by_kind, by_kind_post, prior_range=None,
                               label="", seed=0):
    """Experiment figure for one parameter, two panels: the per-condition distribution of the
    MAP, and each window's marginal median +/- IQR of the draws with its MAP overlaid.

    ``values_by_kind`` is ``{kind: 1D MAP log10}``; ``by_kind_post`` is ``{kind: (n,4)
    [MAP, median, q25, q75]}``. Both panels are always drawn: a product carries all three
    estimates by contract.

    Returns:
        A headless ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 5.5))
    ax_map, ax_post = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    _draw_experiment_distribution_axis(ax_map, values_by_kind, prior_range, seed)
    _draw_experiment_posterior_axis(ax_post, by_kind_post, prior_range, seed)
    if label:
        fig.suptitle(label, fontsize=13)
    return fig


# --------------------------------------------------------------------------------------------
# Within-recording drift: every point estimate against window position, shared posterior bands
# --------------------------------------------------------------------------------------------
_DRIFT_STYLE = {
    "MAP": dict(color="tab:orange", marker="o"),
    "posterior median": dict(color="tab:blue", marker="s"),
    "SGM": dict(color="tab:green", marker="^"),
    "direct": dict(color="tab:purple", marker="D"),
}


def figure_point_estimates_vs_truth(true_log10, estimates, post_q=None, prior_range=None,
                                    label="", n_bins=20, min_count=50, pad_fraction=0.4):
    """One panel per point estimate: the estimate against the truth, read as a summary.

    ``estimates`` maps a short name (``"MAP"``, ``"median"``, ``"SGM"``) to a length-N log10
    array; a ``None`` value is skipped. Each panel shows the videos as a density (hexbin), the
    binned median of that estimate over equal-count bins of the truth as a line, and one shared
    band: the posterior's own interquartile range, taken as the median across the videos of a
    bin of the per-video Q25 and Q75 (``post_q``, ``(N, 5)`` at Q05/Q25/Q50/Q75/Q95). The band
    is the same on every panel because the three estimates come from the same posterior; it is
    the posterior's width, not an error bar on the estimate. Axes are clipped to the prior box
    widened by ``pad_fraction`` of its width on the estimate axis, so MAP values far outside
    the box are off-panel (the tables count them). The panel text gives correlation, MAE and
    signed bias over all videos.
    """
    from matplotlib.figure import Figure
    x = np.asarray(true_log10, dtype=float)
    names = [n for n, a in estimates.items() if a is not None and np.asarray(a).size]
    if not names:
        return None
    style_of = {"MAP": _DRIFT_STYLE["MAP"], "median": _DRIFT_STYLE["posterior median"],
                "posterior median": _DRIFT_STYLE["posterior median"], "SGM": _DRIFT_STYLE["SGM"]}
    fig = Figure(figsize=(4.8 * len(names), 4.6))
    axes = fig.subplots(1, len(names), sharex=True, sharey=True, squeeze=False)[0]
    lo, hi = (float(prior_range[0]), float(prior_range[1])) if prior_range is not None else (
        float(np.nanmin(x)), float(np.nanmax(x)))
    pad = pad_fraction * (hi - lo)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.clip(np.digitize(x, edges, right=True) - 1, 0, n_bins - 1)

    def binned(values, q=0.5):
        out = np.full(n_bins, np.nan)
        for k in range(n_bins):
            vk = values[bin_id == k]
            vk = vk[np.isfinite(vk)]
            if vk.size >= min_count:
                out[k] = np.quantile(vk, q)
        return out

    band = None
    if post_q is not None and np.asarray(post_q).size:
        pq = np.asarray(post_q, dtype=float)
        band = (binned(pq[:, 1]), binned(pq[:, 3]))
    for ax, name in zip(axes, names):
        y = np.asarray(estimates[name], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.hexbin(x[ok], y[ok], gridsize=60, bins="log", cmap="Greys", mincnt=1,
                  extent=(lo, hi, lo - pad, hi + pad))
        if band is not None and np.any(np.isfinite(band[0])):
            ax.fill_between(centers, band[0], band[1], color="tab:blue", alpha=0.18, lw=0,
                            label="posterior IQR (median over the bin)")
        st = style_of.get(name, dict(color="k", marker="o"))
        ax.plot(centers, binned(y), color=st["color"], marker=st["marker"], ms=4, lw=1.5,
                label=f"{name}, median over the bin")
        ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.8, zorder=6,
                label="truth (estimate = true value)")
        for v in (lo, hi):
            ax.axhline(v, color="r", ls=":", lw=0.8)
        err = y[ok] - x[ok]
        corr = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 and np.std(y[ok]) > 0 else np.nan
        ax.text(0.03, 0.97, f"n={int(ok.sum())}\ncorr {corr:+.2f}\nMAE {np.mean(np.abs(err)):.3f}\n"
                            f"bias {np.mean(err):+.3f}", transform=ax.transAxes, va="top",
                fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_title(name, color=st["color"], fontsize=11)
        ax.set_xlabel(f"true log10 {label}".strip())
        ax.set_xlim(lo, hi); ax.set_ylim(lo - pad, hi + pad)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    axes[0].set_ylabel(f"estimate log10 {label}".strip())
    fig.suptitle(f"{label}: point estimates against the truth; band = posterior IQR; "
                 f"dotted red = prior box", fontsize=11)
    fig.tight_layout()
    return fig


def figure_point_estimates_by_cell(estimates, post_q, cell_ids, prior_range=None, label="",
                                   pad_fraction=0.4):
    """One panel per point estimate on experimental recordings, which carry no truth.

    Recordings (cells) are placed along the x axis in the order of their posterior-median value
    (the median over the recording's windows of the per-window 1-D posterior median), so the
    median panel reads as a monotone staircase and the other two panels show how far the MAP
    and the SGM fall from it recording by recording. In each panel every window is one point,
    the recording's median of that estimate is the line, and the posterior's own interquartile
    range (median over the recording's windows of the per-window Q25 and Q75) is one shared
    band, identical on every panel because the three estimates come from the same posterior.
    ``estimates`` maps a short name (``"MAP"``, ``"median"``, ``"SGM"``) to a length-N array over
    windows; ``post_q`` is ``(N, 5)``; ``cell_ids`` labels each window's recording. The estimate
    axis is clipped to the prior box widened by ``pad_fraction`` of its width.
    """
    from matplotlib.figure import Figure
    names = [n for n, a in estimates.items() if a is not None and np.asarray(a).size]
    if not names:
        return None
    cell_ids = np.asarray(cell_ids)
    pq = np.asarray(post_q, dtype=float)
    cells = np.unique(cell_ids)
    med_ref = np.array([np.nanmedian(pq[cell_ids == c, 2]) for c in cells])
    order = np.argsort(med_ref)
    cells = cells[order]
    rank = {c: i for i, c in enumerate(cells)}
    xs = np.array([rank[c] for c in cell_ids], dtype=float)
    band_lo = np.array([np.nanmedian(pq[cell_ids == c, 1]) for c in cells])
    band_hi = np.array([np.nanmedian(pq[cell_ids == c, 3]) for c in cells])
    style_of = {"MAP": _DRIFT_STYLE["MAP"], "median": _DRIFT_STYLE["posterior median"],
                "posterior median": _DRIFT_STYLE["posterior median"], "SGM": _DRIFT_STYLE["SGM"]}
    if prior_range is not None:
        lo, hi = float(prior_range[0]), float(prior_range[1])
    else:
        allv = np.concatenate([np.asarray(estimates[n], dtype=float) for n in names])
        lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
    pad = pad_fraction * (hi - lo)
    fig = Figure(figsize=(4.8 * len(names), 4.6))
    axes = fig.subplots(1, len(names), sharex=True, sharey=True, squeeze=False)[0]
    xc = np.arange(len(cells))
    for ax, name in zip(axes, names):
        y = np.asarray(estimates[name], dtype=float)
        st = style_of.get(name, dict(color="k", marker="o"))
        ax.fill_between(xc, band_lo, band_hi, color="tab:blue", alpha=0.18, lw=0,
                        label="posterior IQR (median over the recording's windows)")
        jitter = (np.random.default_rng(0).uniform(-0.3, 0.3, size=y.size))
        ax.scatter(xs + jitter, y, s=6, color=st["color"], alpha=0.35, lw=0,
                   label=f"{name}, one point per window")
        per_cell = np.array([np.nanmedian(y[cell_ids == c]) for c in cells])
        ax.plot(xc, per_cell, color=st["color"], lw=1.4, zorder=5,
                label=f"{name}, median over the recording")
        for v in (lo, hi):
            ax.axhline(v, color="r", ls=":", lw=0.8)
        outside = float(np.mean((y < lo) | (y > hi))) if y.size else float("nan")
        ax.text(0.03, 0.97, f"windows={int(np.isfinite(y).sum())}  recordings={len(cells)}\n"
                            f"median over windows {np.nanmedian(y):+.3f}\n"
                            f"IQR over windows {np.nanquantile(y, .75) - np.nanquantile(y, .25):.3f}\n"
                            f"outside prior {100 * outside:.0f} %",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_title(name, color=st["color"], fontsize=11)
        ax.set_xlabel("recording, ordered by its posterior-median value")
        ax.set_ylim(lo - pad, hi + pad)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    axes[0].set_ylabel(f"estimate log10 {label}".strip())
    fig.suptitle(f"{label}: point estimates per window and recording; band = posterior IQR; "
                 f"dotted red = prior box", fontsize=11)
    fig.tight_layout()
    return fig


def figure_window_drift(keys, labels, medians_by_estimate, bands=None, prior_ranges=None,
                        title="", x_label="window index within the recording",
                        y_label="estimate (stored coordinates)", band_label="posterior"):
    """Six-panel (or ``ceil(D/3) x 3``) figure of every point estimate against window position.

    ``medians_by_estimate`` maps an estimate name to a ``(T, D)`` array: the median across cells
    of that estimate at each window. All lines are drawn against ONE shared band per window,
    ``bands`` of shape ``(T, D, 5)`` holding the median across cells of the per-window posterior
    quantiles at levels 5/25/50/75/95 %: the central 50 % as the darker band and the central 90 %
    as the lighter one. A missing inner or outer level (NaN) is skipped, which is how a direct
    estimator's single nominal range is drawn. Prior bounds are dotted red lines.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = len(keys)
    n_rows = max(1, math.ceil(D / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(13.5, 3.7 * n_rows), sharex=True, squeeze=False)
    axes = axes.ravel()
    for i in range(D):
        ax = axes[i]
        T = next(iter(medians_by_estimate.values())).shape[0]
        x = np.arange(T)
        if bands is not None:
            # Q axis = artifact_schema.CANONICAL_QUANTILE_LEVELS, validated when the product was
            # loaded: levels (0, 4) bound the 90 % interval, (1, 3) the 50 % interval.
            b = np.asarray(bands, dtype=float)[:, i, :]
            outer = np.isfinite(b[:, 0]) & np.isfinite(b[:, 4])
            inner = np.isfinite(b[:, 1]) & np.isfinite(b[:, 3])
            if outer.any():
                ax.fill_between(x, b[:, 0], b[:, 4], where=outer, color="0.55", alpha=0.14,
                                lw=0, label=f"{band_label} 90 %")
            if inner.any():
                ax.fill_between(x, b[:, 1], b[:, 3], where=inner, color="0.45", alpha=0.28,
                                lw=0, label=f"{band_label} 50 %")
        for name, arr in medians_by_estimate.items():
            st = _DRIFT_STYLE.get(name, dict(color="0.2", marker="."))
            ax.plot(x, np.asarray(arr)[:, i], lw=1.5, ms=4.5, label=name, **st)
        if prior_ranges is not None and prior_ranges[i] is not None:
            lo, hi = prior_ranges[i]
            for v in (lo, hi):
                ax.axhline(v, color="firebrick", ls=":", lw=1.0)
        ax.set_title(f"{labels[i]}  {keys[i]}", fontsize=10.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if i == 0:
            ax.legend(fontsize=7.5, framealpha=0.9)
    for j in range(D, len(axes)):
        axes[j].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
