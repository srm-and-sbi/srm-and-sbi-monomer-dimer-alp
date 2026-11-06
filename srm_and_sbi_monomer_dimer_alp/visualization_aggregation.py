"""Figures of the experimental aggregation and of a selected representative distribution.

One plotting implementation serves the Experiment report (equal weights inside each construction) and the
Nuisance DLI build (recorded mixture probabilities). Every figure takes NAMED SERIES, each a dict with
``name`` (the complete legend text the caller composes, see ``experiment_aggregation.display_name``: the
visible description of the construction, its statistic and its vector count), ``vectors`` ((n, D) estimator
coordinates), optional ``weights`` ((n,), normalized here; ``None`` = equal), an optional ``role`` (the
construction ID, which selects the color and marker) and an optional ``style``; a one-member series is drawn
as a vertical or horizontal reference line, never as a smoothed distribution, and every series is normalized
to unit mass independently, so a 600-member collection is not drawn as more massive than a 60-member one.
Axes are physical units; a log row takes a logarithmic axis. The training-prior bounds are dotted red lines,
labeled so: training bounds, not uncertainty intervals or clipping thresholds. Legends sit outside the data
panels. Every figure carries a ``subtitle`` line for the data scope (which cells and windows contributed, which
samples were pooled, which representative), written by the caller. Matplotlib is imported lazily (headless
Agg figures).
"""
from __future__ import annotations

import math

import numpy as np

from .parameterization import is_log_row, to_physical


def _phys1(values, entry):
    """Physical values of ONE parameter's column (any shape) under its table row."""
    arr = np.asarray(values, float)
    return to_physical(arr[..., None], [entry])[..., 0]

STYLES = {
    "per_cell": dict(color="tab:blue", marker="o", lw=1.6),
    "per_window_position": dict(color="tab:green", marker="s", lw=1.6),
    "per_cell_window": dict(color="0.55", marker=".", lw=1.0),
    "global": dict(color="tab:orange", marker="D", lw=1.8),
    "reference": dict(color="tab:purple", marker="*", lw=1.6),
    "mixture": dict(color="k", marker="x", lw=2.0),
}
_FALLBACK = [dict(color=c, marker="o", lw=1.4) for c in ("tab:red", "tab:brown", "tab:pink", "tab:olive", "tab:cyan")]
PRIOR_LABEL = "training prior bounds"
BAND_COLOR = "0.6"


def _style(series, i):
    st = series.get("style") or STYLES.get(series.get("role", series["name"]))
    return dict(st) if st else dict(_FALLBACK[i % len(_FALLBACK)])


def _normalized_weights(series):
    v = np.asarray(series["vectors"], float)
    w = series.get("weights")
    w = np.ones(v.shape[0]) if w is None else np.asarray(w, float)
    return v, w / w.sum()


def _grid(n):
    cols = 3 if n > 4 else max(1, n)
    rows = max(1, math.ceil(n / cols))
    return rows, cols


def _titles(fig, title, subtitle):
    """Title and the scope subtitle under it (the subtitle is the figure's data-scope line)."""
    fig.suptitle(title, fontsize=12, y=0.995)
    if subtitle:
        fig.text(0.5, 0.962, subtitle, ha="center", va="top", fontsize=8.5, color="0.25")


# Physical axis labels of the detector imaging parameters (presentation only; the conventions of
# DETECTOR_WORKFLOW.md sec. 7.6 and the parameterization notes): short form for corner axes, long form for panels.
PHYSICAL_AXIS_LABELS = {
    "mu_r": ("PSF width (px)", "mu_r: PSF width in pixels (sqrt(2) x sigma of the Gaussian PSF; median over receptors)"),
    "sigma_r": ("PSF-width log-spread", "sigma_r: log-spread of the PSF-width population across receptors (dimensionless)"),
    "mu_pc": ("brightness (photons)", "mu_pc: median photons per active dye and frame, before the camera"),
    "sigma_pc": ("ln-brightness spread", "sigma_pc: spread of the ln-brightness of one dye (dimensionless)"),
    "prob_photo_bleach": ("bleaching prob. / 100 frames", "prob_photo_bleach: per-dye bleaching probability over the fixed 100-frame reference interval"),
    "lambda_rate": ("flicker rate (1/s)", "lambda_rate: ln-brightness correlation-decay rate (per second; correlation time 1/lambda)"),
}


def axis_label(e, *, short=False) -> str:
    """The physical axis label of a table row: the documented convention and unit when known, else key and UNIT."""
    key = e["KEY"]
    if key in PHYSICAL_AXIS_LABELS:
        return PHYSICAL_AXIS_LABELS[key][0 if short else 1]
    unit = e.get("UNIT")
    return f"{key}" + (f" ({unit})" if unit else "")


def _physical_axis(ax, e, axis="x"):
    """A log axis for a log row. Call AFTER plotting: the major ticks are chosen from the plotted range so a
    narrow span (PSF widths 1 to 2.3 px, brightness 100 to 300 photons) still carries informative labels
    (about four to five '%g' values), never only the decades; minor ticks are unlabeled."""
    if is_log_row(e):
        from matplotlib.ticker import FixedLocator, FormatStrFormatter, LogLocator, MaxNLocator, NullFormatter
        ax_obj = ax.xaxis if axis == "x" else ax.yaxis
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
        lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
        lo = max(float(lo), 1e-300); hi = max(float(hi), lo * 1.0001)
        if np.log10(hi / lo) < 1.3:
            ticks = [t for t in MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi) if lo <= t <= hi and t > 0]
        else:
            ticks = [t for t in LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)).tick_values(lo, hi) if lo <= t <= hi]
        if len(ticks) >= 2:
            ax_obj.set_major_locator(FixedLocator(ticks))
        ax_obj.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0, numticks=20))
        ax_obj.set_minor_formatter(NullFormatter())
        ax_obj.set_major_formatter(FormatStrFormatter("%g"))


def _panel_title(ax, e):
    ax.set_title(f"{e['KEY']}  {e.get('LABEL') or ''}", fontsize=10)


def figure_marginal_cdfs(series, table, *, title="", subtitle=""):
    """Six (``ceil(D/3) x 3``) panels of normalized weighted empirical CDFs (the fraction of a series'
    representatives at or below each physical value), one series per construction; a one-member series is a
    vertical line (a point representation, which has no width to show); the training-prior bounds dotted; one
    shared legend below the panels."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    keys = [e["KEY"] for e in table]; D = len(keys)
    rows, cols = _grid(D)
    fig = Figure(figsize=(4.6 * cols, 3.6 * rows + 1.6))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    for j, e in enumerate(table):
        ax = axes[j]
        lo, hi = _phys1(float(e["PRIOR_RANGE"][0]), e), _phys1(float(e["PRIOR_RANGE"][1]), e)
        for i, s in enumerate(series):
            v, w = _normalized_weights(s); st = _style(s, i)
            x = _phys1(v[:, j], e)
            if v.shape[0] == 1:
                ax.axvline(x[0], color=st["color"], ls="--" if s.get("role") == "reference" else "-", lw=st["lw"], label=s["name"])
                continue
            order = np.argsort(x, kind="stable")
            xs, cw = x[order], np.cumsum(w[order])
            ax.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0.0], cw]), where="post", color=st["color"], lw=st["lw"], label=s["name"])
        for n, b in enumerate((lo, hi)):
            ax.axvline(b, color="r", ls=":", lw=0.8, label=PRIOR_LABEL if n == 0 else None)
        _physical_axis(ax, e, "x")
        ax.set_ylim(0, 1.02); _panel_title(ax, e)
        ax.set_xlabel(axis_label(e), fontsize=7.5, wrap=True)
        ax.set_ylabel("fraction of representatives\nat or below this value", fontsize=8); ax.grid(alpha=0.2)
    for ax in axes[D:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2 if len(labels) > 4 else len(labels) or 1, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    _titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    return fig


def figure_joint_corner(series, table, *, title="", subtitle="", max_points=4000, seed=0):
    """Lower-triangle corner figure of every coordinate pair: each point projects one complete representative
    vector onto the pair (a subset of ``max_points`` per series for display; one-member series as stars). The
    upper half is left empty so no pair is shown twice. Physical axes; the legend sits in the empty half."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    keys = [e["KEY"] for e in table]; D = len(keys)
    fig = Figure(figsize=(2.6 * D + 1, 2.6 * D + 1.4))
    axes = fig.subplots(D, D, squeeze=False)
    rng = np.random.default_rng(seed)
    for a in range(D):
        for b in range(D):
            ax = axes[a, b]
            if b >= a:
                ax.set_visible(False)
                continue
            for i, s in enumerate(series):
                v, w = _normalized_weights(s); st = _style(s, i)
                x = _phys1(v[:, b], table[b]); y = _phys1(v[:, a], table[a])
                if v.shape[0] > max_points:
                    sub = rng.choice(v.shape[0], max_points, replace=False); x, y, w = x[sub], y[sub], w[sub]
                if v.shape[0] == 1:
                    ax.scatter(x, y, marker="*", s=160, color=st["color"], edgecolors="k", zorder=6, label=s["name"] if (a, b) == (1, 0) else None)
                else:
                    size = 6 if v.shape[0] > 50 else 24
                    ax.scatter(x, y, s=size * (0.3 + 0.7 * w / w.max()), marker=st["marker"], color=st["color"], alpha=0.55, lw=0,
                               label=s["name"] if (a, b) == (1, 0) else None)
            for n, v_ in enumerate(table[b]["PRIOR_RANGE"]):
                ax.axvline(_phys1(float(v_), table[b]), color="r", ls=":", lw=0.6, label=PRIOR_LABEL if (a, b, n) == (1, 0, 0) else None)
            for v_ in table[a]["PRIOR_RANGE"]:
                ax.axhline(_phys1(float(v_), table[a]), color="r", ls=":", lw=0.6)
            _physical_axis(ax, table[b], "x"); _physical_axis(ax, table[a], "y")
            if a == D - 1:
                ax.set_xlabel(f"{keys[b]}\n{axis_label(table[b], short=True)}", fontsize=8)
            else:
                ax.set_xticklabels([])
            if b == 0:
                ax.set_ylabel(f"{keys[a]}\n{axis_label(table[a], short=True)}", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7); ax.grid(alpha=0.15)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9, frameon=False, bbox_to_anchor=(0.98, 0.93),
                   title="each point = one complete representative vector", title_fontsize=9)
    _titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_temporal(per_cell_window, per_position, global_vec, reference, table, *, times, window_bands=None,
                    title="", subtitle="", window_seconds=None, labels=None):
    """Representatives over acquisition time: one thin line per cell (``per_cell_window``: cell -> (T, D)
    over ``times``, the window starts in seconds), the position-level vectors (bold; pooled joint samples
    across cells, not an average of the thin lines), the dataset vector and the fixed reference (horizontal
    lines), over the SOURCE within-window 50 % and 90 % marginal intervals (``window_bands``, (T, D, 5)
    estimator coordinates, the median endpoint across cells) drawn in neutral gray: a summary of the input
    posteriors, not a confidence band for the bold line and not a range containing 90 % of the cells. Up to six
    cells are identified by color in the legend; more are drawn faint and anonymous (identities in the report).
    ``labels`` overrides the legend texts (keys: cell_window, position, global, band90, band50)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    keys = [e["KEY"] for e in table]; D = len(keys)
    rows, cols = _grid(D)
    fig = Figure(figsize=(4.8 * cols, 3.6 * rows + 1.8))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    t = np.asarray(times, float)
    lab = {"cell_window": "One SGM per cell–window pair (thin, one line per cell)",
           "position": "One SGM per window position (pooled joint samples across cells)",
           "global": "One SGM for the included dataset", "band90": "source within-window 90 % marginal interval (median endpoint across cells)",
           "band50": "source within-window 50 % marginal interval (median endpoint across cells)"}
    lab.update(labels or {})
    cells = list(per_cell_window)
    few = len(cells) <= 6
    palette = ["tab:blue", "tab:red", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    for j, e in enumerate(table):
        ax = axes[j]
        if window_bands is not None:
            b = np.asarray(window_bands, float)[:, j, :]
            ax.fill_between(t, _phys1(b[:, 0], e), _phys1(b[:, 4], e), color=BAND_COLOR, alpha=0.18, lw=0, label=lab["band90"])
            ax.fill_between(t, _phys1(b[:, 1], e), _phys1(b[:, 3], e), color=BAND_COLOR, alpha=0.32, lw=0, label=lab["band50"])
        for n, c in enumerate(cells):
            arr = np.asarray(per_cell_window[c], float)[:, j]
            if few:
                ax.plot(t, _phys1(arr, e), color=palette[n % len(palette)], lw=1.1, alpha=0.9, marker=".", ms=4,
                        label=f"{lab['cell_window']}: cell {c}")
            else:
                ax.plot(t, _phys1(arr, e), color=STYLES["per_cell_window"]["color"], lw=0.6, alpha=0.35, label=lab["cell_window"] if n == 0 else None)
        if per_position is not None:
            ax.plot(t, _phys1(np.asarray(per_position, float)[:, j], e), color=STYLES["per_window_position"]["color"],
                    lw=2.4, marker="s", ms=5, label=lab["position"], zorder=5)
        if global_vec is not None:
            ax.axhline(_phys1(np.asarray(global_vec, float)[j], e), color=STYLES["global"]["color"], lw=1.8, label=lab["global"])
        if reference is not None:
            ax.axhline(_phys1(np.asarray(reference["log10"], float)[j], e), color=STYLES["reference"]["color"], ls="--",
                       lw=1.5, label=reference.get("name", "reference"))
        for n, v_ in enumerate(e["PRIOR_RANGE"]):
            ax.axhline(_phys1(float(v_), e), color="r", ls=":", lw=0.7, label=PRIOR_LABEL if n == 0 else None)
        _physical_axis(ax, e, "y")
        _panel_title(ax, e); ax.grid(alpha=0.2)
        ax.set_ylabel(axis_label(e, short=True), fontsize=8)
        ax.set_xlabel(f"start of the {window_seconds:g} s window (s)" if window_seconds else "start of the window (s)")
        ax.set_xticks(t)
    for ax in axes[D:]:
        ax.set_visible(False)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    _titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    return fig


def figure_window_selection(per_cell_by_set, table, *, reference=None, global_by_set=None, title="", subtitle="",
                            set_labels=None):
    """One SGM per cell under every requested window set, cell by cell: cells along x (ordered by their first
    set's value of the first parameter), one marker per set (the per-cell construction's blue: filled for
    the first set, hollow for the second, further shapes after), a thin connector joining the same cell
    across sets (a change of window selection, not elapsed time and not an uncertainty interval), the
    dataset vector of each set as an orange line (solid, dash-dot, dotted) and the fixed reference dashed
    purple. ``set_labels`` maps a set name to its visible phrase (e.g. 'all ten 2 s windows')."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    keys = [e["KEY"] for e in table]; D = len(keys)
    sets = list(per_cell_by_set)
    set_labels = dict(set_labels or {})
    cells = sorted(set().union(*[set(per_cell_by_set[s]) for s in sets]))
    first = sets[0]
    order_key = [float(np.asarray(per_cell_by_set[first][c], float)[0]) if c in per_cell_by_set[first] else np.inf for c in cells]
    cells = [c for _, c in sorted(zip(order_key, cells))]
    rows, cols = _grid(D)
    fig = Figure(figsize=(4.8 * cols, 3.6 * rows + 1.8))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    markers = ["o", "o", "^", "D", "v", "P", "X"]
    blue = STYLES["per_cell"]["color"]
    x = np.arange(len(cells))
    for j, e in enumerate(table):
        ax = axes[j]
        ys = {}
        for k, s in enumerate(sets):
            y = np.array([_phys1(np.asarray(per_cell_by_set[s][c], float)[j], e) if c in per_cell_by_set[s] else np.nan for c in cells])
            ys[s] = y
            phrase = set_labels.get(s, f"window set '{s}'")
            if k == 1:
                ax.scatter(x, y, marker=markers[1], s=34, facecolors="none", edgecolors=blue, lw=1.2, label=f"One SGM per cell, windows: {phrase}", zorder=4)
            else:
                ax.scatter(x, y, marker=markers[k % len(markers)], s=30, color=blue, label=f"One SGM per cell, windows: {phrase}", zorder=4)
            if global_by_set and s in global_by_set:
                ax.axhline(_phys1(np.asarray(global_by_set[s], float)[j], e), color=STYLES["global"]["color"],
                           lw=1.4, ls=["-", "-.", ":"][k % 3], label=f"One SGM for the included dataset, windows: {phrase}")
        if len(sets) > 1:
            for i in range(len(cells)):
                ax.plot([i] * len(sets), [ys[s][i] for s in sets], color="0.6", lw=0.7, zorder=2,
                        label="same cell under the two window selections (a change of selection, not time)" if (i == 0) else None)
        if reference is not None:
            ax.axhline(_phys1(np.asarray(reference["log10"], float)[j], e), color=STYLES["reference"]["color"], ls="--",
                       lw=1.5, label=reference.get("name", "reference"))
        for n, v_ in enumerate(e["PRIOR_RANGE"]):
            ax.axhline(_phys1(float(v_), e), color="r", ls=":", lw=0.7, label=PRIOR_LABEL if n == 0 else None)
        _physical_axis(ax, e, "y")
        ax.set_ylabel(axis_label(e, short=True), fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([str(c) for c in cells], fontsize=6 if len(cells) > 24 else 7, rotation=90 if len(cells) > 12 else 0)
        ax.set_xlabel(f"cell identity (ordered by {keys[0]} under '{first}')", fontsize=8); _panel_title(ax, e); ax.grid(alpha=0.2, axis="y")
    for ax in axes[D:]:
        ax.set_visible(False)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    _titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    return fig


# ---- posterior compatibility (Experiment) and member probabilities (Nuisance DLI constructor) ----------------------
# Every compatibility figure shares one 0–1 color scale for the distance rank, gray for the excluded self comparison,
# stable identity/time ordering on both axes and the candidate/reference orientation in its axis labels. Experiment's
# strips and heat maps show RAW median ranks, never sampling probabilities; the constructor's probability figure shows
# the actual stored row probabilities beside the equal-member baseline.
RANK_LABEL = "Distance rank (higher = closer to reference SGM)"
SCORE_LABEL = "Median compatibility rank"
SELF_COLOR = "0.78"


def _rank_cmap():
    import matplotlib
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(SELF_COLOR)
    return cmap


def _id_fontsize(n):
    return 5 if n > 40 else (6.5 if n > 20 else 8)


def figure_compatibility_cells(ranks, cell_ids, scores, *, title="", subtitle=""):
    """The cell-to-cell distance-rank matrix (rows: candidate cell; columns: reference cell; stable numeric cell order
    on both axes, every cell retained) with a narrow aligned strip of each row's median compatibility rank on the same
    0–1 scale (a raw score, not a sampling probability). Self comparisons are gray (NaN)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    R = np.asarray(ranks, float); n = R.shape[0]
    labels = [str(c) for c in cell_ids]; fs = _id_fontsize(n)
    fig = Figure(figsize=(11.6, 8.4))
    # matrix, the aligned score strip, then the colorbar at the far right (its tick labels never cross the strip)
    gs = fig.add_gridspec(1, 3, width_ratios=[20, 5.5, 0.8], wspace=0.12, left=0.07, right=0.9, top=0.9, bottom=0.1)
    ax = fig.add_subplot(gs[0, 0]); sx = fig.add_subplot(gs[0, 1], sharey=ax); cax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(R, vmin=0, vmax=1, cmap=_rank_cmap(), aspect="auto", origin="upper", interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=fs, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xlabel("Reference cell"); ax.set_ylabel("Candidate cell")
    fig.colorbar(im, cax=cax, label=RANK_LABEL)
    s = np.asarray(scores, float)
    sx.barh(np.arange(n), np.where(np.isnan(s), 0.0, s), color="0.3", height=0.8)
    for i in np.flatnonzero(s == 0):
        sx.plot([0.01], [i], marker="|", color="tab:red", ms=6, lw=0)
    sx.set_xlim(0, 1); sx.set_xlabel(f"{SCORE_LABEL}\n(0–1; a raw score, not a sampling probability)", fontsize=8)
    sx.tick_params(axis="y", labelleft=False); sx.grid(alpha=0.25, axis="x")
    sx.set_title("row median over the other cells\n(red tick = exactly zero)", fontsize=8.5)
    _titles(fig, title, subtitle)
    return fig


def figure_compatibility_windows(scores, cell_ids, times, *, title="", subtitle="", window_seconds=None):
    """Cells (rows, stable order) by selected window start times (columns): the within-cell median compatibility rank
    of each cell–window SGM against the other selected windows of the same cell, on the shared 0–1 scale. Zeros stay
    visible (dark); an unavailable score would be gray."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    S = np.asarray(scores, float); C, W = S.shape
    fig = Figure(figsize=(8.0, max(4.8, 0.115 * C + 2.6)))
    ax = fig.add_subplot(111)
    im = ax.imshow(S, vmin=0, vmax=1, cmap=_rank_cmap(), aspect="auto", interpolation="nearest")
    ax.set_xticks(range(W)); ax.set_xticklabels([f"{float(t):g}" for t in times], fontsize=8)
    ax.set_yticks(range(C)); ax.set_yticklabels([str(c) for c in cell_ids], fontsize=_id_fontsize(C))
    ax.set_xlabel(f"start of the selected {float(window_seconds):g} s window (s)" if window_seconds else "start of the selected window (s)")
    ax.set_ylabel("cell (candidate and reference windows belong to this cell)")
    fig.colorbar(im, ax=ax, label=f"{SCORE_LABEL} (within the cell; 0–1)")
    _titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_compatibility_atlas_page(panels, times, *, title="", subtitle="", ncols=3, nrows=4):
    """One contact sheet of within-cell window-to-window matrices (``panels``: ``[(cell_id, (W, W) ranks), ...]``, at most
    ``ncols * nrows``), each titled by its cell; candidate and reference axes are window start times; shared 0–1 color
    scale; gray self comparisons."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    fig = Figure(figsize=(3.7 * ncols + 1.4, 3.5 * nrows + 1.3))
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    fig.subplots_adjust(left=0.07, right=0.88, top=0.9, bottom=0.07, hspace=0.55, wspace=0.4)
    lab = [f"{float(t):g}" for t in times]; W = len(lab); im = None
    for ax, (cell, R) in zip(axes, panels):
        im = ax.imshow(np.asarray(R, float), vmin=0, vmax=1, cmap=_rank_cmap(), aspect="equal", interpolation="nearest")
        ax.set_title(f"cell {cell}", fontsize=10)
        ax.set_xticks(range(W)); ax.set_xticklabels(lab, fontsize=7, rotation=90)
        ax.set_yticks(range(W)); ax.set_yticklabels(lab, fontsize=7)
        ax.set_xlabel("reference window start (s)", fontsize=8); ax.set_ylabel("candidate window start (s)", fontsize=8)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    if im is not None:
        cax = fig.add_axes([0.905, 0.25, 0.015, 0.5])
        fig.colorbar(im, cax=cax, label=RANK_LABEL)
    _titles(fig, title, subtitle)
    return fig


def figure_member_probabilities(labels, selected, baseline, *, title="", subtitle="", xlabel="member",
                                selected_name="stored sampling probability (selected policy)",
                                baseline_name="equal-member baseline (same components, masses and overrides)"):
    """The constructor's probability figure: the actual stored row probabilities (black dots) beside the equal-member
    baseline (gray bars) for every row, zero probabilities marked (a sampling outcome, not a data-quality verdict)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    sel = np.asarray(selected, float); base = np.asarray(baseline, float); n = sel.size; x = np.arange(n)
    fig = Figure(figsize=(max(7.5, 0.17 * n + 2.5), 5.2))
    ax = fig.add_subplot(111)
    ax.bar(x, base, width=0.8, color="0.8", label=baseline_name)
    ax.plot(x, sel, "o", color="k", ms=4, label=selected_name)
    zero = np.flatnonzero(sel == 0)
    if zero.size:
        ax.plot(zero, np.zeros(zero.size), "x", color="tab:red", ms=7, label=f"zero probability ({zero.size}; a sampling outcome, not a data-quality verdict)")
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in labels], rotation=90, fontsize=_id_fontsize(n))
    ax.set_xlabel(xlabel); ax.set_ylabel("sampling probability of the stored row"); ax.set_ylim(bottom=0); ax.grid(alpha=0.2, axis="y")
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=1, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    _titles(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    return fig
