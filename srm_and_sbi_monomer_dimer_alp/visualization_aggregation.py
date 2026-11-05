"""Figures of the experimental aggregation and of a selected representative distribution.

One plotting implementation serves the Experiment report (equal weights inside each construction) and the
Nuisance DLI build (recorded mixture probabilities). Every figure takes NAMED SERIES, each a dict with
``name``, ``vectors`` ((n, D) estimator coordinates), optional ``weights`` ((n,), normalized here; ``None``
= equal), and an optional ``style`` (color, marker); a one-member series is drawn as a step or reference
line, never as a smoothed distribution, and every series is normalized to unit mass independently, so a
600-member collection is not drawn as more massive than a 60-member one. Axes are physical units; a log row
takes a logarithmic axis. The prior box is dotted. Matplotlib is imported lazily (headless Agg figures).
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


def figure_marginal_cdfs(series, table, *, title="", subtitle=""):
    """Six (``ceil(D/3) x 3``) panels of normalized weighted empirical CDFs, physical units; singletons as
    vertical reference lines; the prior box dotted."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    keys = [e["KEY"] for e in table]; D = len(keys)
    rows, cols = _grid(D)
    fig = Figure(figsize=(4.6 * cols, 3.6 * rows + 0.9))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    for j, e in enumerate(table):
        ax = axes[j]
        lo, hi = _phys1(float(e["PRIOR_RANGE"][0]), e), _phys1(float(e["PRIOR_RANGE"][1]), e)
        for i, s in enumerate(series):
            v, w = _normalized_weights(s); st = _style(s, i)
            x = _phys1(v[:, j], e)
            if v.shape[0] == 1:
                ax.axvline(x[0], color=st["color"], ls="--" if s.get("role") == "reference" else "-", lw=st["lw"],
                           label=f"{s['name']} (one vector)")
                continue
            order = np.argsort(x, kind="stable")
            xs, cw = x[order], np.cumsum(w[order])
            ax.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0.0], cw]), where="post", color=st["color"], lw=st["lw"],
                    label=f"{s['name']} (n={v.shape[0]})")
        for b in (lo, hi):
            ax.axvline(b, color="r", ls=":", lw=0.8)
        _physical_axis(ax, e, "x")
        ax.set_ylim(0, 1.02); ax.set_title(f"{e['KEY']}  {e.get('LABEL') or ''}", fontsize=10)
        ax.set_xlabel(axis_label(e), fontsize=7.5, wrap=True)
        ax.set_ylabel("cumulative mass"); ax.grid(alpha=0.2)
        if j == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axes[D:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12)
    if subtitle:
        fig.text(0.5, 0.005, subtitle, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


def figure_joint_corner(series, table, *, title="", max_points=4000, seed=0):
    """Lower-triangle corner figure of every coordinate pair: co-occurring coordinates of the representatives
    (a subset of ``max_points`` per series for display; singletons as stars). Physical axes."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    keys = [e["KEY"] for e in table]; D = len(keys)
    fig = Figure(figsize=(2.6 * D + 1, 2.6 * D + 1))
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
                               label=f"{s['name']} (n={v.shape[0]})" if (a, b) == (1, 0) else None)
            for v_ in table[b]["PRIOR_RANGE"]:
                ax.axvline(_phys1(float(v_), table[b]), color="r", ls=":", lw=0.6)
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
        fig.legend(handles, labels, loc="upper right", fontsize=9, frameon=False, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def figure_temporal(per_cell_window, per_position, global_vec, reference, table, *, times, window_bands=None,
                    title="", cell_label="cell"):
    """Per-cell-window SGMs over acquisition time (one thin line per cell), the position-level SGMs (bold),
    the global and reference vectors (horizontal lines), and the within-window marginal intervals (median
    across cells of the per-window 50 % and 90 % intervals of the draws) as inspectable bands, labeled apart
    from the across-cell dispersion. ``per_cell_window`` maps cell -> (T, D) over ``times``; ``per_position``
    is (T, D); ``window_bands`` is (T, D, 5) in estimator coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    keys = [e["KEY"] for e in table]; D = len(keys)
    rows, cols = _grid(D)
    fig = Figure(figsize=(4.8 * cols, 3.6 * rows + 1.0))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    t = np.asarray(times, float)
    for j, e in enumerate(table):
        ax = axes[j]
        if window_bands is not None:
            b = np.asarray(window_bands, float)[:, j, :]
            ax.fill_between(t, _phys1(b[:, 0], e), _phys1(b[:, 4], e), color="tab:blue", alpha=0.10, lw=0,
                            label="within-window 90 % interval of the draws (median across cells)")
            ax.fill_between(t, _phys1(b[:, 1], e), _phys1(b[:, 3], e), color="tab:blue", alpha=0.18, lw=0,
                            label="within-window 50 % interval of the draws (median across cells)")
        for n, (c, arr) in enumerate(per_cell_window.items()):
            ax.plot(t, _phys1(np.asarray(arr, float)[:, j], e), color=STYLES["per_cell_window"]["color"], lw=0.8, alpha=0.6,
                    marker=".", ms=3, label="per_cell_window SGM, one line per cell (across-cell dispersion)" if n == 0 else None)
        if per_position is not None:
            ax.plot(t, _phys1(np.asarray(per_position, float)[:, j], e), color=STYLES["per_window_position"]["color"],
                    lw=2.0, marker="s", ms=5, label="per_window_position SGM")
        if global_vec is not None:
            ax.axhline(_phys1(np.asarray(global_vec, float)[j], e), color=STYLES["global"]["color"], lw=1.8, label="global SGM")
        if reference is not None:
            ax.axhline(_phys1(np.asarray(reference["log10"], float)[j], e), color=STYLES["reference"]["color"], ls="--",
                       lw=1.5, label=reference.get("name", "reference"))
        for v_ in e["PRIOR_RANGE"]:
            ax.axhline(_phys1(float(v_), e), color="r", ls=":", lw=0.7)
        _physical_axis(ax, e, "y")
        ax.set_title(f"{e['KEY']}  {e.get('LABEL') or ''}", fontsize=10); ax.grid(alpha=0.2)
        ax.set_ylabel(axis_label(e, short=True), fontsize=8)
        ax.set_xlabel("start of the window (s)")
    for ax in axes[D:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    return fig


def figure_window_selection(per_cell_by_set, table, *, reference=None, global_by_set=None, title=""):
    """Matched per-cell representatives under every requested window set: cells along x (ordered by their
    first set's value of the first parameter), one marker per set, a thin line joining the same cell across
    sets, so the change caused by the represented period is read cell by cell."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    keys = [e["KEY"] for e in table]; D = len(keys)
    sets = list(per_cell_by_set)
    cells = sorted(set().union(*[set(per_cell_by_set[s]) for s in sets]))
    rows, cols = _grid(D)
    fig = Figure(figsize=(4.8 * cols, 3.6 * rows + 1.0))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    x = np.arange(len(cells))
    for j, e in enumerate(table):
        ax = axes[j]
        ys = {}
        for k, s in enumerate(sets):
            y = np.array([_phys1(np.asarray(per_cell_by_set[s][c], float)[j], e) if c in per_cell_by_set[s] else np.nan
                          for c in cells])
            ys[s] = y
            ax.scatter(x, y, marker=markers[k % len(markers)], s=28, label=f"per_cell SGM, window set '{s}'", zorder=4)
            if global_by_set and s in global_by_set:
                ax.axhline(_phys1(np.asarray(global_by_set[s], float)[j], e), color=STYLES["global"]["color"],
                           lw=1.2, ls=["-", "-.", ":"][k % 3], label=f"global SGM, '{s}'")
        if len(sets) > 1:
            for i in range(len(cells)):
                ax.plot([i] * len(sets), [ys[s][i] for s in sets], color="0.6", lw=0.7, zorder=2)
        if reference is not None:
            ax.axhline(_phys1(np.asarray(reference["log10"], float)[j], e), color=STYLES["reference"]["color"], ls="--",
                       lw=1.5, label=reference.get("name", "reference"))
        for v_ in e["PRIOR_RANGE"]:
            ax.axhline(_phys1(float(v_), e), color="r", ls=":", lw=0.7)
        _physical_axis(ax, e, "y")
        ax.set_ylabel(axis_label(e, short=True), fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([str(c) for c in cells], fontsize=7, rotation=90 if len(cells) > 12 else 0)
        ax.set_xlabel("cell"); ax.set_title(f"{e['KEY']}  {e.get('LABEL') or ''}", fontsize=10); ax.grid(alpha=0.2, axis="y")
    for ax in axes[D:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    return fig
