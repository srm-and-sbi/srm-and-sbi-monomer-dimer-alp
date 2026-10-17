"""Shared engine for the population-composition analysis.

WHAT IT PRODUCES. One report holding both halves of a single comparison:

  * the **experimental** composition -- relative species abundance inferred from the real recordings,
    per condition, with cell-to-cell error bars, the within-recording time course, per-recording
    spread, and the sensitivity of every headline number to the choices that could have driven it;
  * the **synthetic** validation -- the identical readout computed on held-out videos whose truth is
    known, which is what licenses reading the experimental numbers at all.

Keeping them in one document is the point: the experimental value and the measured error of the
instrument that produced it belong on the same page, and the two are otherwise reported by different
tools with different run stamps.

TWO POPULATIONS. The estimator's stoichiometry coordinates are the receptor-subunit total N_total,
the basal REFERENCE fraction of dimers f and, under MET-INLB, the initial active-dimer allocation
R_active (PROJECT_CONTEXT.md sec. 2, *Modeling assumptions of the MET model*, items 4, 9 and 11).
Two populations follow from them and both are reported: the REFERENCE population (f: what the
declared occupancy defines from the underlying population) and the STARTING, window-start
population the generator places (the allocation turns c of the available probe-bound monomer pairs
into active dimers, so the starting fraction f_start >= f; under MET-FAB the allocation is declared
0 and the starting population IS the reference). Every derived quantity is realized INSIDE each
posterior draw through the one population helper (``parameterization.realize_population``) and
only then averaged; a draw with a fraction outside [0, 1] is FLAGGED and counted, carries NaN, and
is never clipped.

ONE WORKFLOW. Unlike the mirrored stage runners, this analysis exists for the **biology** workflow
only, and the asymmetry is scientific rather than incidental: the composition is a function of the
inferred stoichiometry coordinates, and the detector workflow infers imaging parameters and no
stoichiometry at all -- it treats the population implicitly, as part of what it marginalizes. There
is therefore no detector composition to compute, in the same way the detector's ``Nuisance_DLI`` pool
has no biology counterpart. The engine is nonetheless written in the shared shape -- a
``CompositionSpec`` resolved once in :func:`_population_composition_spec` -- so that everything
workflow-specific stays in one place and a future workflow carrying a stoichiometry plugs in without
touching the body.

ONE ESTIMATOR, THE RECORDINGS OF BOTH KINDS. The Experiment product holds one estimator (the
``--condition`` of this run) applied to the recordings of every kind it was run on. The analysis
compares the kinds held in that product, which is a comparison under ONE estimator's assumptions:
every derived composition is computed under the estimator's condition (its occupancy and declared
values), never the recording's kind. An estimator applied to the other condition's recordings is a
MISSPECIFICATION TEST, and the report says so beside every derived table and figure of those
recordings. It does not replace a comparison of independently inferred populations, each under its
own condition's estimator.

INPUTS, AND WHAT EACH ONE COSTS IF ABSENT.
  * the Experiment product **with per-window posterior draws** (``--dump-posterior-samples``):
    required. The composition must be formed inside each draw, so the stored marginal quantiles
    cannot substitute -- a fraction built from marginals is a different quantity, not a coarser one.
  * the MAP_Recovery output: optional. Without it the experimental half still runs and the report
    states that the validation column is unavailable, rather than quietly reporting experimental
    numbers with no measured error.

NO GPU, no estimator, no video. It reads two completed outputs and writes a report; it is a post-hoc
analysis in ``Script_Bank/Analysis``, never a pipeline stage and never wired into the dispatcher.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from matplotlib.figure import Figure

from .labeling import LABELING_CONDITIONS, expected_fluorescent_complexes, resolve_labeling
from . import population_composition as pc
from .diagnostics import DiagnosticReporter
from .evaluation import CAPACITY_STRATA
from .experiment_support import condition_display, kind_label   # kind_label: the misspecification label
from .parameterization import (POPULATION_FIELD_DOCS, declared_row_values, population_valid,
                               realize_population, to_physical)

# Colors: one per condition, held fixed across every figure so a condition is the same color
# everywhere in the report, and one neutral for the synthetic reference.
CONDITION_COLORS = ("tab:blue", "tab:red")
SYNTHETIC_COLOR = "0.45"

# The per-draw realization ladder: every quantity the population helper realizes from one draw
# (N_total, f, R_active), in report order, with its table label and its unit. Counts are the integer
# probe-bound counts the generator places (realized per draw, then averaged, so a mean count need
# not be an integer); the derived fraction f_start and the active share c / D_start are reported
# beside f and R_active, never in their place.
REALIZATION_LADDER = (
    ("count_total", "N_total (receptor-subunit total, underlying population)", "count"),
    ("fraction_basal_dimers_reference", "f (basal reference fraction of dimers)", "fraction"),
    ("relative_count_active_dimers_initial", "R_active (allocation, relative to capacity)", "fraction"),
    ("capacity", "capacity (probe-bound monomer pairs available)", "count"),
    ("n_pairs_allocated", "c (pairs allocated to active dimers)", "count"),
    ("n_monomers_underlying_reference", "M_ref (underlying monomers, reference)", "count"),
    ("n_dimers_underlying_reference", "D_ref (underlying dimers, reference)", "count"),
    ("n_monomers_underlying_start", "M_start (underlying monomers, window start)", "count"),
    ("n_dimers_underlying_start", "D_start (underlying dimers, window start)", "count"),
    ("fraction_basal_dimers_start", "f_start (starting fraction, derived)", "fraction"),
    ("active_share", "active share c / D_start (derived; NaN where D_start = 0)", "fraction"),
    ("n_monomers_reference", "A_ref (probe-bound monomers, reference)", "count"),
    ("n_dimers_reference", "D_sim (dimers with at least one probe, reference)", "count"),
    ("n_dimers_one_probe", "B1 (one-probe dimers; unchanged by the allocation)", "count"),
    ("n_dimers_two_probe_reference", "B2_ref (two-probe dimers, reference)", "count"),
    ("n_monomers", "A_start (monomers placed)", "count"),
    ("n_dimers_two_probe", "B2_start (two-probe dimers placed)", "count"),
    ("n_dimers", "dimers placed (B1 + B2_start)", "count"),
    ("n_complexes", "complexes placed (A_start + B1 + B2_start)", "count"),
    ("n_subunits", "N_R (represented receptors)", "count"),
    ("n_probe_bound_receptors_expected", "probe-bound receptors (expected)", "count"),
    ("expected_fluorescent_complexes", "expected fluorescent complexes at frame 0", "count"),
)
REALIZATION_KEYS = tuple(name for name, _label, _unit in REALIZATION_LADDER)
_REALIZATION_LABEL = {name: label for name, label, _unit in REALIZATION_LADDER}
_REALIZATION_UNIT = {name: unit for name, _label, unit in REALIZATION_LADDER}
_DERIVED_FIELDS = ("n_complexes", "n_probe_bound_receptors_expected", "expected_fluorescent_complexes")
assert all(k in POPULATION_FIELD_DOCS or k in _DERIVED_FIELDS for k in REALIZATION_KEYS)


# ---- the stoichiometry binding: estimator coordinates -> the physical pair, per draw ------------

@dataclass(frozen=True)
class StoichiometryBinding:
    """The ESTIMATOR's condition bound to its stoichiometry rows: the one place the k estimator
    coordinates (N_total, f and, under MET-INLB, R_active) become the kernel's physical pair and the
    realized window-start population.

    Every method takes the ``(..., k)`` estimator-space block ``select`` cuts from a theta array
    (``k = len(count_index)``). The draws that fail ``parameterization.population_valid`` are
    FLAGGED: they map to a NaN pair and to NaN realization fields, are counted by the caller, and
    are never clipped. Nothing here reconstructs underlying monomers as A_start / p: the underlying
    and the placed counts both come from the population helper.
    """
    condition: str
    keys: tuple                 # the condition's learnable keys, in theta order
    count_keys: tuple           # (N_total, f) under MET-FAB; (N_total, f, R_active) under MET-INLB
    count_index: tuple          # their positions in theta
    rows: tuple                 # their table rows (the conversion rule of each)
    declared_allocation: object  # the declared R_active (MET-FAB 0.0) or None when inferred

    def select(self, theta_flow):
        """The ``(..., k)`` stoichiometry block of a ``(..., D)`` estimator-space array."""
        theta_flow = np.asarray(theta_flow, dtype=float)
        if theta_flow.shape[-1] != len(self.keys):
            raise ValueError(f"stoichiometry binding [{self.condition}]: theta has {theta_flow.shape[-1]} "
                             f"columns; the condition's table has {len(self.keys)}.")
        return theta_flow[..., list(self.count_index)]

    def components(self, block_flow):
        """Physical ``(N_total, f, R_active)`` arrays of one common shape from the ``(..., k)`` block;
        the declared allocation is filled in under MET-FAB."""
        block = np.asarray(block_flow, dtype=float)
        if block.shape[-1] != len(self.count_index):
            raise ValueError(f"stoichiometry binding [{self.condition}]: expected the (..., "
                             f"{len(self.count_index)}) block {self.count_keys}, got shape {block.shape}.")
        phys = np.asarray(to_physical(block, list(self.rows)), dtype=float)
        n_total, fraction = phys[..., 0], phys[..., 1]
        if len(self.count_index) == 3:
            allocation = phys[..., 2]
        else:
            allocation = np.full(n_total.shape, float(self.declared_allocation))
        return n_total, fraction, allocation

    def valid(self, block_flow):
        """The ONE validity rule, per draw (``parameterization.population_valid``)."""
        n_total, fraction, allocation = self.components(block_flow)
        return population_valid(count_total=n_total, fraction_basal_dimers_reference=fraction,
                                relative_count_active_dimers_initial=allocation, condition=self.condition)

    def reference_pair(self, block_flow):
        """The kernel's ``(..., 2)`` pair of the REFERENCE population: ``(N_total, 2 f / (1 + f))``;
        NaN for a flagged draw."""
        n_total, fraction, _allocation = self.components(block_flow)
        ok = self.valid(block_flow)
        with np.errstate(invalid="ignore", divide="ignore"):
            x_b = np.where(ok, 2.0 * fraction / (1.0 + fraction), np.nan)
        return np.stack([np.where(ok, n_total, np.nan), x_b], axis=-1)

    def realize(self, block_flow, plan=None):
        """The window-start realization of every draw: ``(fields, valid)`` with ``fields`` a dict
        ``name -> float array`` over :data:`REALIZATION_KEYS` (NaN for a flagged draw) and ``valid``
        the boolean mask. The expected fluorescent complexes need the condition's labeling ``plan``
        (``labeling.expected_fluorescent_complexes``); without one that field stays NaN."""
        n_total, fraction, allocation = self.components(block_flow)
        ok = population_valid(count_total=n_total, fraction_basal_dimers_reference=fraction,
                              relative_count_active_dimers_initial=allocation, condition=self.condition)
        fields = {name: np.full(ok.shape, np.nan) for name in REALIZATION_KEYS}
        if ok.any():
            pop = realize_population(count_total=n_total[ok], fraction_basal_dimers_reference=fraction[ok],
                                     relative_count_active_dimers_initial=allocation[ok],
                                     condition=self.condition)
            for name in REALIZATION_KEYS:
                if name in POPULATION_FIELD_DOCS:
                    fields[name][ok] = np.asarray(getattr(pop, name), dtype=float)
            fields["n_complexes"][ok] = np.asarray(pop.n_complexes, dtype=float)
            fields["n_probe_bound_receptors_expected"][ok] = np.asarray(pop.n_probe_bound_receptors_expected,
                                                                         dtype=float)
            if plan is not None:
                fields["expected_fluorescent_complexes"][ok] = np.asarray(
                    expected_fluorescent_complexes(pop, plan), dtype=float)
        return fields, ok

    def start_pair(self, block_flow):
        """The kernel's ``(..., 2)`` pair of the STARTING population: ``(N_total, 2 f_start / (1 +
        f_start))`` -- the total receptors stay N_total (M_start + 2 D_start = N_total); NaN for a
        flagged draw."""
        fields, _ok = self.realize(block_flow)
        return start_pair_of(fields)


def capacity_strata(capacity):
    """``(label, mask)`` pairs over the observations by TRUE capacity, one per stratum of
    ``evaluation.CAPACITY_STRATA`` (the strata the Evaluation stage reports, so both reports use
    ONE definition): ``{0}`` (no effect, a prior-like posterior expected), ``1-9``, ``10-99``,
    ``100-999``, ``>= 1000``. Selection is on the truth, never on the estimate; a NaN capacity
    belongs to no stratum."""
    cap = np.asarray(capacity, dtype=float)
    out = []
    for label, lo, hi in CAPACITY_STRATA:
        mask = np.isfinite(cap) & (cap >= lo)
        if hi is not None:
            mask &= cap <= hi
        out.append((label, mask))
    return out


def start_pair_of(fields):
    """The starting pair ``(N_total, 2 f_start / (1 + f_start))`` from realized fields (NaN where
    the draw was flagged)."""
    n_total = np.asarray(fields["count_total"], dtype=float)
    f_start = np.asarray(fields["fraction_basal_dimers_start"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_b = 2.0 * f_start / (1.0 + f_start)
    return np.stack([n_total, x_b], axis=-1)


def stoichiometry_binding(table, keys, condition) -> StoichiometryBinding:
    """Bind the ESTIMATOR's ``condition`` to the stoichiometry rows of its ``table`` and ``keys``
    (``workflow.parameter_table`` / ``parameter_keys`` under that condition). The count keys are
    N_total and f, plus R_active where the condition infers it; where the condition declares the
    allocation (MET-FAB: 0) the declared value is read from ``declared_row_values``. Refuses a table
    that carries no stoichiometry (the detector's)."""
    from .parameterization import PARAMETERS
    sto = PARAMETERS.simulation.rds.stoichiometry
    keys = tuple(str(k) for k in keys)
    table = list(table)
    if [e["KEY"] for e in table] != list(keys):
        raise ValueError(f"stoichiometry_binding [{condition}]: the table's keys "
                         f"{[e['KEY'] for e in table]} differ from the keys {list(keys)}.")
    declared = declared_row_values(condition)
    count_keys = [sto.count_total_key, sto.basal_fraction_key]
    if sto.active_allocation_key in keys:
        count_keys.append(sto.active_allocation_key)
    elif sto.active_allocation_key not in declared:
        raise ValueError(f"stoichiometry_binding [{condition}]: {sto.active_allocation_key!r} is neither "
                         f"inferred nor declared under {condition}.")
    missing = [k for k in count_keys if k not in keys]
    if missing:
        raise ValueError(f"stoichiometry_binding [{condition}]: the table lacks the stoichiometry keys "
                         f"{missing}; this analysis is biology-only by construction.")
    count_index = tuple(keys.index(k) for k in count_keys)
    return StoichiometryBinding(
        condition=str(condition), keys=keys, count_keys=tuple(count_keys), count_index=count_index,
        rows=tuple(dict(table[i]) for i in count_index),
        declared_allocation=(None if sto.active_allocation_key in keys
                             else float(declared[sto.active_allocation_key])))


@dataclass(frozen=True)
class CompositionSpec:
    """Everything about this analysis that a workflow supplies."""

    condition: str                        # the ESTIMATOR's condition (--condition), never a recording kind
    parameter_keys: list
    binding: StoichiometryBinding
    count_index: tuple                    # theta indices of the stoichiometry coordinates (k = 2 or 3)
    to_physical: object                   # (..., k) estimator-space block -> the REFERENCE pair (N_total, x_B)
    plan: object                          # the condition's production labeling plan (expected fluorescence)
    prior_low: np.ndarray                 # prior lower bounds, estimator space (all parameters)
    prior_high: np.ndarray                # prior upper bounds, estimator space (all parameters)
    experiment_npz: object
    recovery_npz: object
    report_dir: object
    alias: str
    timing_label: str
    stage: str
    window_seconds: float


# ---- loading -----------------------------------------------------------------------------------

def load_experiment_draws(path, parameter_keys):
    """Per-window posterior draws and their labels from a completed Experiment stage.

    The product is read through the artifact schema (:func:`artifact_schema.load_product`), so an
    obsolete or malformed product is refused rather than read, and its parameter keys must equal
    this workflow's key for key and in order. Fails loud in two further, distinguishable ways,
    because the remedies differ: a missing output means the stage was never run, while an output
    without ``posterior_samples_cloud`` means it ran without ``--dump-posterior-samples`` and must
    be re-run. Substituting the stored quantiles would produce a confident report about a
    different quantity.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"the population composition needs the Experiment stage's per-window posterior draws, "
            f"and the Experiment output is absent:\n    {path}\nRun the biology Experiment stage for "
            f"this timing with --dump-posterior-samples first.")
    arrays, manifest = schema.load_product(path, stage="experiment")
    schema.assert_parameter_keys(manifest, parameter_keys, source=str(path))
    if "posterior_samples_cloud" not in arrays:
        raise SystemExit(
            f"the Experiment output\n    {path}\nholds no 'posterior_samples_cloud'. The "
            f"composition is formed inside each posterior draw, so the stored marginal quantiles "
            f"cannot substitute: a fraction of marginals is a different quantity, not a coarser "
            f"one. Re-run the Experiment stage with --dump-posterior-samples.")
    cloud = np.asarray(arrays["posterior_samples_cloud"], dtype=float)
    labels = dict(kind_index=arrays["kind_index"].astype(int), cell=arrays["cell"].astype(int),
                  chunk=arrays["chunk"].astype(int), kinds=[str(k) for k in arrays["kinds"]])
    return cloud, labels


def load_recovery(path, parameter_keys):
    """Held-out truth and the MAP estimate for the synthetic validation half, or ``None`` if the
    Recovery product is absent. The product is validated against the artifact schema (an
    obsolete ``inferred_log10`` product is refused, not silently read) and its parameter keys
    must equal this workflow's exactly. This arm is MAP-based by design; extending it to the
    marginal median and the SGM is a separate task. The stored ``true_log10`` field holds
    estimator coordinates (its name is kept by the schema)."""
    if not path.exists():
        return None
    arrays, manifest = schema.load_product(path, stage="evaluation")
    schema.assert_parameter_keys(manifest, parameter_keys, source=str(path))
    true_log10 = np.asarray(arrays["true_log10"], dtype=float)
    map_estimate = np.asarray(arrays["map_estimate"], dtype=float)
    return true_log10, map_estimate


# ---- the misspecification label ------------------------------------------------------------------

def misspecification_note(kinds, condition):
    """The sentence that stands beside every derived table and figure holding recordings of the
    other condition; empty when every kind is the estimator's own."""
    other = [k for k in kinds if str(k) != str(condition)]
    if not other:
        return ""
    return (f" MISSPECIFICATION TEST: the {condition_display(condition)} estimator is applied to "
            f"{', '.join(condition_display(k) for k in other)} recordings; every derived composition is "
            f"computed under the {condition_display(condition)} estimator's assumptions (its occupancy and "
            f"declared values), never the recording's kind, so those rows test the estimator outside its "
            f"condition rather than describe that condition's population.")


# ---- figures (object API; each returns a Figure, never saves) -----------------------------------

def _fraction_rows():
    """The fraction quantities, in report order, as ``(index, Quantity)`` pairs."""
    return [(i, pc.COMPOSITION[i]) for i in pc.FRACTION_INDICES]


def _figure_conditions(mean, sem, labels, recovery_rows, title):
    """Grouped bars: every fraction, both conditions, with the estimator's in-model error beside."""
    rows = _fraction_rows()
    fig = Figure(figsize=(1.55 * len(rows) + 3.2, 4.8), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(rows))
    width = 0.34
    for k, label in enumerate(labels):
        vals = np.array([100.0 * mean[k, i] for i, _q in rows])
        errs = np.array([100.0 * sem[k, i] for i, _q in rows])
        ax.bar(x + (k - 0.5) * width, vals, width=width, yerr=errs, capsize=3,
               color=CONDITION_COLORS[k % len(CONDITION_COLORS)],
               label=f"{label} (± SEM over recordings)")
    if recovery_rows is not None:
        # The estimator's own error on held-out truth, drawn as a symmetric whisker centered on each
        # experimental bar. It is a DIFFERENT kind of error from the SEM beside it -- in-model
        # measurement error against known truth, versus biological spread across recordings -- and the
        # caption says so. Both belong here: the contrast is only meaningful if the second is small
        # against the first.
        by_key = {r["key"]: r for r in recovery_rows}
        for k in range(len(labels)):
            for j, (i, q) in enumerate(rows):
                mae = by_key.get(q.key, {}).get("mae")
                if mae is None:
                    continue
                center = 100.0 * mean[k, i]
                ax.plot([x[j] + (k - 0.5) * width] * 2, [center - mae, center + mae],
                        color=SYNTHETIC_COLOR, lw=2.4, solid_capstyle="butt", zorder=5,
                        label="synthetic MAE (held-out truth)" if (k == 0 and j == 0) else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q.symbol}\n{q.formula}" for _i, q in rows], fontsize=8)
    ax.set_ylabel("share of the population (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncols=3)
    ax.set_title(title)
    return fig


def _figure_recovery(true_comp, inferred_comp, rows, rng, stratum=None, max_points=6000):
    """Held-out truth versus estimate for the monomer and dimer-complex shares."""
    by_key = {r["key"]: r for r in rows}
    show = [pc.COMPOSITION_KEYS.index("monomer_fraction"), pc.DIMER_INDEX]
    n = true_comp.shape[0]
    idx = np.arange(n) if n <= max_points else rng.choice(n, max_points, replace=False)
    fig = Figure(figsize=(9.4, 4.6), layout="constrained")
    axes = fig.subplots(1, len(show), squeeze=False)[0]
    for ax, i in zip(axes, show):
        q = pc.COMPOSITION[i]
        ax.scatter(100.0 * true_comp[idx, i], 100.0 * inferred_comp[idx, i], s=5, c=SYNTHETIC_COLOR,
                   alpha=0.25, edgecolors="none")
        ax.plot([0, 100], [0, 100], color="k", lw=1.2, ls="--", label="exact recovery")
        if stratum is not None and i == pc.DIMER_INDEX and stratum.get("n"):
            ax.axvspan(100.0 * stratum["threshold"], 100.0, color="tab:orange", alpha=0.12,
                       label=f"stratum n={stratum['n']:,}: MAE {stratum['mae']:.1f} pp, "
                             f"bias {stratum['bias']:+.1f} pp")
        r = by_key.get(q.key, {})
        ax.set_title(f"{q.symbol} = {q.formula}\nMAE {r.get('mae', float('nan')):.1f} pp, "
                     f"p95 {r.get('p95', float('nan')):.1f} pp, r = {r.get('r', float('nan')):.3f}",
                     fontsize=10)
        ax.set_xlabel("true share (%)")
        ax.set_ylabel("inferred share (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    return fig


def _figure_time_course(mean, sem, labels, window_seconds, quantity_index, title):
    """The dimer-complex share against window index, per condition."""
    q = pc.COMPOSITION[quantity_index]
    n_chunks = mean.shape[1]
    t = window_seconds * (np.arange(n_chunks) + 0.5)
    fig = Figure(figsize=(7.4, 4.6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    for k, label in enumerate(labels):
        y = 100.0 * mean[k, :, quantity_index]
        e = 100.0 * sem[k, :, quantity_index]
        color = CONDITION_COLORS[k % len(CONDITION_COLORS)]
        ax.plot(t, y, marker="o", ms=5, lw=2, color=color, label=label)
        ax.fill_between(t, y - e, y + e, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel(f"window center (s), {window_seconds:g} s windows")
    ax.set_ylabel(f"{q.symbol} = {q.formula}  (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(title)
    return fig


def _figure_per_cell(cell_values, labels, quantity_index, bands, rng):
    """Every recording's value for one quantity, so the condition mean is read against its spread."""
    q = pc.COMPOSITION[quantity_index]
    fig = Figure(figsize=(7.0, 4.6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    for k, label in enumerate(labels):
        row = 100.0 * cell_values[k, :, quantity_index]
        row = row[np.isfinite(row)]
        jitter = (rng.random(row.size) - 0.5) * 0.28
        color = CONDITION_COLORS[k % len(CONDITION_COLORS)]
        ax.scatter(np.full(row.size, k) + jitter, row, s=34, color=color, alpha=0.75,
                   edgecolors="k", linewidths=0.4, label=f"{label} (n={row.size} recordings)")
        if row.size:
            ax.plot([k - 0.3, k + 0.3], [row.mean()] * 2, color="k", lw=2.2, zorder=5)
    for band in bands:
        ax.axhline(100.0 * band, color="0.75", lw=1, ls=":")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"{q.symbol} = {q.formula}  (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Per-recording values (black bar = condition mean; dotted lines = spread bands)")
    return fig


# ---- report ------------------------------------------------------------------------------------

def _pct(value):
    return "-" if not np.isfinite(value) else f"{100.0 * value:.1f}"


_COUNT_NAMES = {"total_complexes": "total complexes", "total_receptors": "total receptors"}


def _quantity_name(index):
    """Table label for one quantity: the symbol for a share, spelled out for the totals."""
    q = pc.COMPOSITION[int(index)]
    return q.symbol if q.is_fraction else _COUNT_NAMES.get(q.key, q.key)


def _quantity_value(index, value):
    """Format a value in the units of its quantity.

    One formatter for every table, because the alternative is what this replaced: a percent
    formatter applied to a total (a count), which multiplied it by a hundred and printed a
    plausible-looking number two orders of magnitude wrong. A share and a count cannot share a
    formatter, so the quantity decides.
    """
    if not np.isfinite(value):
        return "-"
    return _pct(value) if pc.COMPOSITION[int(index)].is_fraction else f"{value:.0f}"


def _num(value, fmt):
    """A statistic in ``fmt``, or ``-`` where it is undefined (an empty stratum, a flagged draw)."""
    return fmt.format(value) if np.isfinite(value) else "-"


def _ladder_value(name, value):
    """A realization-ladder value in its unit: fractions in percent, counts as counts."""
    if not np.isfinite(value):
        return "-"
    return _pct(value) if _REALIZATION_UNIT[name] == "fraction" else f"{value:.1f}"


def _composition_ladder(pair_per_draw, valid, labels_, n_kinds):
    """The full aggregation ladder of one population from its per-draw pair ``(N, S, 2)``:
    window means over the valid draws, the grid, the per-recording values, the condition
    summary, the first window and the time course."""
    comp = pc.window_draw_mean(pc.composition(pair_per_draw), valid)                # (N, 5)
    grid, n_cells, n_chunks = pc.composition_grid(comp, labels_["kind_index"], labels_["cell"],
                                                  labels_["chunk"], n_kinds)
    cell_values = pc.per_cell(grid)
    mean, sem, n_cells_used = pc.condition_summary(cell_values)
    first_mean, _first_sem = pc.window_slice_summary(grid, 0)
    tc_mean, tc_sem = pc.time_course(grid)
    return dict(window=comp, grid=grid, cell_values=cell_values, mean=mean, sem=sem, n_cells=n_cells_used,
                first_mean=first_mean, tc_mean=tc_mean, tc_sem=tc_sem, n_cells_grid=int(n_cells),
                n_chunks=int(n_chunks))


def _write_report(args, spec, data, rng):
    # No valid draw anywhere: the report still states the exclusion counts and the synthetic
    # validation, and marks every composition check and summary UNAVAILABLE -- nothing is clipped
    # and no population is manufactured to fill the tables.
    available = data["n_valid_draws"] > 0
    flagged = data["flagged"]
    reporter = DiagnosticReporter(
        stage=spec.stage, enabled=True, dump=True, dump_dir=spec.report_dir,
        run_label=f"{spec.alias}_{spec.timing_label}",
        run_note="" if available else (
            f"NO VALID COMPOSITION DRAWS. Every one of the {flagged['n_total']:,} posterior draws is "
            f"flagged (a fraction outside [0, 1] or a non-finite value), so the composition checks and "
            f"summaries are unavailable. The exclusion counts are reported below; nothing is clipped and "
            f"no population is manufactured."))
    kinds = data["kinds"]
    labels = [kind_label(k, spec.condition) for k in kinds]
    misspec = misspecification_note(kinds, spec.condition)
    rows = _fraction_rows()
    ref, start = data["reference"], data["start"]
    mean, sem, n_cells_used = ref["mean"], ref["sem"], ref["n_cells"]
    fab_like = spec.binding.declared_allocation is not None      # the allocation is declared: start = reference
    no_valid = "no valid composition draw: every draw is flagged"

    reporter.check("posterior_draws_present", True,
                   f"{data['n_windows']} windows x {data['n_draws']} draws x "
                   f"{len(spec.parameter_keys)} parameters")
    if available:
        reporter.check("composition_closure", data["closure"] < 1e-9,
                       f"max |f_A + f_B - 1| = {data['closure']:.2e} over every valid draw")
    else:
        reporter.check_unavailable("composition_closure", no_valid + ", so f_A + f_B has no value to test")
    reporter.check("no_nan_inf(posterior draws)", data["draws_finite"],
                   "clean" if data["draws_finite"] else "NaN/Inf present in the stored draws")
    reporter.check("flagged_draws_counted", True,
                   f"{flagged['total']:,} of {flagged['n_total']:,} draws flagged "
                   f"({100.0 * flagged['share']:.3f} %): a fraction outside [0, 1] or a non-finite value; "
                   f"they carry NaN and are never clipped")
    if available:
        reporter.check("replicate_unit_is_recording", True,
                       f"every interval and test below is over recordings "
                       f"({', '.join(f'{lab} n={int(n)}' for lab, n in zip(labels, n_cells_used[:, 0]))})"
                       f", never over the {data['n_windows']} windows")
    else:
        reporter.check_unavailable("replicate_unit_is_recording",
                                   no_valid + ", so no recording carries a value and no interval or test "
                                   "is reported")
    reporter.stat("estimator condition", condition_display(spec.condition),
                  note=f"the --condition of this run: the estimator's table ({len(spec.parameter_keys)} "
                       f"parameters), occupancy and declared values bind every derived quantity."
                       + (misspec or " Every recording kind in the product is the estimator's own condition."))

    if not available:
        reporter.stat("composition summaries", "unavailable",
                      note="no valid composition draw: every table and figure built from the "
                           "Experiment draws is left out rather than filled with values computed "
                           "from nothing; the flagged-draw counts below and the synthetic validation "
                           "(held-out truths, independent of these draws) are reported as usual."
                           + misspec)
    else:
        # -- the two populations ---------------------------------------------------------------------
        def composition_table(title, pop, note):
            reporter.table(
                title,
                ["condition"] + [f"{q.symbol} ({q.formula})" for _i, q in rows]
                + [_quantity_name(i) for i in pc.COUNT_INDICES],
                [[labels[k]]
                 + [f"{_pct(pop['mean'][k, i])} ± {_pct(pop['sem'][k, i])}" for i, _q in rows]
                 + [f"{pop['mean'][k, i]:.0f} ± {pop['sem'][k, i]:.0f}" for i in pc.COUNT_INDICES]
                 for k in range(len(kinds))],
                note=note + misspec)

        ladder_note = ("Each value is the mean over recordings of the mean over that recording's windows of "
                       "the mean over that window's posterior draws of the quantity -- formed inside each "
                       "draw, so the correlation between the receptor total and the fractions is carried "
                       "through; flagged draws (a fraction outside [0, 1]) carry NaN and are counted below. "
                       "The error is the standard error across RECORDINGS, which is cell-to-cell biological "
                       "spread; it is not the within-window posterior width, and it is deliberately not "
                       "divided by the number of windows, since ten windows of one recording are not ten "
                       "independent measurements. The shares f_A + f_B close to 100% exactly; f_B = 1 - f_A "
                       "by definition, and f_R (the share of RECEPTORS in dimers) is x_B = 2 f_B / (1 + f_B).")
        if fab_like:
            composition_table(
                "Population composition, span-averaged (percent, mean ± SEM over recordings) -- "
                "start = reference", ref,
                "Under this condition the allocation is declared "
                f"{spec.binding.declared_allocation:g}, so the STARTING (window-start) population IS the "
                "REFERENCE population: f_B is the inferred basal reference fraction f, and one table "
                "describes both. " + ladder_note)
        else:
            composition_table(
                "STARTING (window-start) population composition, span-averaged (percent, mean ± SEM over "
                "recordings)", start,
                "The population the generator places at the window start: the inferred allocation R_active "
                "turns c = round(R_active x capacity) of the available probe-bound monomer pairs into "
                "active dimers, so f_B here is the DERIVED starting fraction f_start = D_start / (M_start "
                "+ D_start) >= f, realized inside every draw by the population helper. " + ladder_note)
            composition_table(
                "REFERENCE population composition, span-averaged (percent, mean ± SEM over recordings)", ref,
                "The reference population: f_B is the inferred basal reference fraction f itself, before the "
                "allocation (what the declared occupancy defines from the underlying population). Read it "
                "beside the starting table above; the difference between the two is the allocation. "
                + ladder_note)

        reporter.table(
            f"Population composition, first window only (0-{spec.window_seconds:g} s)"
            + (" -- start = reference" if fab_like else ""),
            ["condition", "population"] + [f"{q.symbol}" for _i, q in rows],
            [[labels[k], name] + [_pct(pop["first_mean"][k, i]) for i, _q in rows]
             for k in range(len(kinds))
             for name, pop in ((("reference = start", ref),) if fab_like
                               else (("starting", start), ("reference", ref)))],
            note="The earliest state the recordings resolve. Reported separately because a span average "
                 "over a population that changes during the recording describes no instant of it -- see "
                 "the time-course table below for whether that is the case here." + misspec)

        # -- the per-draw realization ladder --------------------------------------------------------
        lad = data["ladder"]
        reporter.table(
            "Window-start realization per draw, span-averaged (mean ± SEM over recordings; fractions in "
            "percent, counts as counts)",
            ["quantity"] + labels,
            [[_REALIZATION_LABEL[name]]
             + [(f"{_ladder_value(name, lad['mean'][k, j])} ± {_ladder_value(name, lad['sem'][k, j])}"
                 if np.isfinite(lad["mean"][k, j]) else "-")
                for k in range(len(kinds))]
             for j, name in enumerate(REALIZATION_KEYS)],
            note="Every row is realized INSIDE each posterior draw by the one population helper "
                 "(parameterization.realize_population: the reference population from (N_total, f) under the "
                 "estimator's occupancy, then the allocation c = round(R_active x capacity) with capacity = "
                 "A_ref // 2 under ligand classes and 0 otherwise), then averaged draws -> window -> "
                 "recording -> condition. Counts are integers per draw; a mean count need not be one. "
                 "f_start and the active share are DERIVED and stand beside f and R_active, never in their "
                 "place; the active share is NaN where D_start = 0 and such draws drop out of its mean. "
                 "The expected fluorescent complexes follow labeling.expected_fluorescent_complexes under the "
                 f"condition's production labeling plan ({spec.plan.describe()}). Flagged draws carry NaN "
                 "in every row." + misspec)

    reporter.table(
        "Flagged draws (a fraction outside [0, 1] or a non-finite value): counted, never clipped",
        ["condition", "windows", "draws flagged / total", "share %", "windows with any flagged draw",
         "windows with every draw flagged"]
        + [f"{key} outside [0, 1]" for key in flagged["per_key"]],
        [[labels[k], f"{flagged['windows'][k]}", f"{flagged['count'][k]:,} / {flagged['draws'][k]:,}",
          f"{100.0 * flagged['share_kind'][k]:.3f}", f"{flagged['windows_any'][k]}",
          f"{flagged['windows_all'][k]}"]
         + [f"{flagged['per_key'][key][k]:,}" for key in flagged["per_key"]]
         for k in range(len(kinds))],
        note="The one validity rule (parameterization.population_valid): N_total finite and positive, "
             "the fractions finite and inside [0, 1], the allocation equal to its declared value where "
             "the condition declares one. The posterior sampler is unrestricted, so draws beyond a "
             "fraction's box occur; they are excluded from every realized quantity and the window "
             "mean is taken over the remaining draws, which the first columns make visible. A window "
             "with every draw flagged contributes NaN and drops out of its recording's mean." + misspec)

    if available:
        if args.bootstrap:
            reporter.table(
                f"Bootstrap check on the reported means ({args.bootstrap:,} resamples of recordings; "
                f"reference population)",
                ["condition", "quantity", "mean %", "normal-theory ± SEM", "bootstrap 95% interval",
                 "half-width difference (pp)"],
                [[labels[k], q.symbol, _pct(mean[k, i]),
                  f"[{_pct(mean[k, i] - 1.96 * sem[k, i])}, {_pct(mean[k, i] + 1.96 * sem[k, i])}]",
                  f"[{_pct(ref['boot_lo'][k, i])}, {_pct(ref['boot_hi'][k, i])}]",
                  f"{abs(100.0 * (0.5 * (ref['boot_hi'][k, i] - ref['boot_lo'][k, i]) - 1.96 * sem[k, i])):.2f}"]
                 for k in range(len(kinds)) for i, q in rows],
                note="A percentile bootstrap over the replicate unit, free of the normal-theory "
                     "assumption behind the SEM. Close agreement means the reported error bars do not "
                     "depend on that assumption; a large difference would say the cell-to-cell "
                     "distribution is skewed enough that the SEM misstates it." + misspec)

        reporter.table(
            "Sensitivity: does the headline survive the choices that could have driven it? (reference "
            "population)",
            ["variant", "condition", "f_B %", "shift vs headline (pp)", "draws retained %",
             "windows left empty"],
            [[v["name"], labels[k], _pct(v["mean"][k, pc.DIMER_INDEX]),
              f"{100.0 * (v['mean'][k, pc.DIMER_INDEX] - mean[k, pc.DIMER_INDEX]):+.1f}",
              f"{100.0 * v['retained'][k]:.1f}" if v["retained"] is not None else "-",
              f"{v['empty'][k]} / {v['windows'][k]}" if v["empty"] is not None else "-"]
             for v in ref["variants"] for k in range(len(kinds))],
            note="Two independent challenges to the headline. (1) Support: the posterior places mass "
                 "beyond the box the estimator was trained on, most of all in the receptor total, so the "
                 "composition is recomputed from progressively more restricted draws (a draw outside a "
                 "fraction's box is flagged in every variant). Because the composition is a ratio, "
                 "out-of-support mass in the total largely cancels -- the table measures whether it does, "
                 "and at what cost in discarded windows. (2) Center: the compositional (closed geometric) "
                 "center replaces the arithmetic mean, which is the natural center on the simplex. Read the "
                 "spread of f_B across these variants as the model-conditional bracket on the value, and "
                 "lead with the most conservative one." + misspec)

        reporter.table(
            "Per-recording spread of the monomer share (reference population)",
            ["condition", "recordings", "min %", "median %", "max %",
             f"below {100 * pc.HETEROGENEITY_BANDS[0]:.0f}%", f"above {100 * pc.HETEROGENEITY_BANDS[1]:.0f}%"],
            [[labels[k], h.get("n", 0),
              _pct(h.get("minimum", float("nan"))), _pct(h.get("median", float("nan"))),
              _pct(h.get("maximum", float("nan"))), h.get("n_below", "-"), h.get("n_above", "-")]
             for k, h in enumerate(ref["heterogeneity"])],
            note="Whether the condition mean describes a typical recording. Mass at both extremes means "
                 "the mean sits between two groups of cells rather than inside one, and any downstream "
                 "use should carry the per-recording values (saved in the arrays) rather than the mean "
                 "alone." + misspec)

        reporter.table(
            "Within-recording time course (reference population)",
            ["condition", "quantity", "first window", "last window",
             "median per-recording Spearman rho", "first-vs-last signed-rank p", "recordings"],
            [[labels[k], _quantity_name(qi),
              _quantity_value(qi, ref["tc_mean"][k, 0, qi]),
              _quantity_value(qi, ref["tc_mean"][k, -1, qi]),
              f"{m[k]['spearman_median']:+.2f}", f"{m[k]['wilcoxon_p']:.3g}", m[k]["n_cells"]]
             for qi, m in ref["monotonicity"] for k in range(len(kinds))],
            note="The two endpoint columns are condition MEANS across recordings, matching every other "
                 "table here (shares in percent, the totals as counts). "
                 "Spearman rho is computed per recording against the window index and then median-ed, so "
                 "it is a direction and strength robust to a single outlying window; the signed-rank p "
                 "pairs each recording's last window against its own first, so between-cell spread "
                 "cannot mask a shared trend. A significant p in one condition and not the other is the "
                 "informative pattern -- it makes the flat condition a control for whatever is common to "
                 "both acquisitions." + misspec)

        if len(kinds) >= 2:
            reporter.table(
                "Condition contrast at the recording level (two-sided rank-sum; reference population)",
                ["quantity", f"{labels[0]} median", f"{labels[1]} median", "ratio", "p", "n"],
                [[_quantity_name(j), _quantity_value(j, r["median_a"]), _quantity_value(j, r["median_b"]),
                  f"{r['ratio']:.2f}x", f"{r['p']:.2g}", f"{r['n_a']} vs {r['n_b']}"]
                 for j, r in enumerate(ref["contrast"])],
                note="Recording-level, so the p-values carry the replicate count the design actually has. "
                     "Read each p beside the synthetic validation table: a contrast is only interpretable "
                     "where the same quantity is recovered accurately on held-out truth, and a quantity that "
                     "fails to separate here may simply be the least identified one. This is a contrast "
                     "between recording kinds under ONE estimator's assumptions; it does not replace a "
                     "comparison of independently inferred populations, each under its own condition's "
                     "estimator." + misspec)

    # -- synthetic half --------------------------------------------------------------------------
    if data["recovery"] is None:
        reporter.stat("synthetic validation", "unavailable",
                      note=f"no MAP_Recovery output at {spec.recovery_npz}; the experimental values "
                           f"above therefore carry no measured in-model error. Run the Evaluation "
                           f"stage for this timing to supply it.")
    else:
        rec = data["recovery"]
        reporter.table(
            f"Synthetic validation: the same readout where the truth is known "
            f"(n = {rec['rows'][0]['n']:,} held-out videos; reference population)",
            ["quantity", "definition", "MAE", "p95 abs. error", "bias", "r"],
            [[q.symbol if q.is_fraction else _COUNT_NAMES.get(q.key, q.key), q.formula,
              f"{r['mae']:.3g} {r['unit']}", f"{r['p95']:.3g} {r['unit']}",
              f"{r['bias']:+.3g} {r['unit']}", f"{r['r']:.4f}"]
             for q, r in zip(pc.COMPOSITION, rec["rows"])],
            note="The identical function of the identical estimator, applied to videos whose "
                 "parameters are known. Fractions are compared in percentage points, the totals in "
                 "dex (a count's error is multiplicative). This is POINT-ESTIMATE accuracy: whether "
                 "the posterior INTERVAL of a fraction covers truth at its nominal rate is a "
                 "different question, and it is not computable from a recovery artifact that stores "
                 "only per-parameter marginal quantiles -- it needs joint posterior draws on the "
                 "held-out set. The f_A and f_B rows are necessarily identical, since f_B = 1 - f_A "
                 "makes their errors equal in magnitude; the agreement is an internal check, not two "
                 f"independent results. Flagged MAP estimates (a fraction outside [0, 1]): "
                 f"{rec['n_flagged_estimates']:,} of {rec['n_total']:,}, excluded from every row.")
        reporter.table(
            "Why the composition is better identified than the counts it is built from",
            ["quantity", "MAE (dex)", "r", "n"],
            [[r["key"], f"{r['mae']:.4f}", f"{r['r']:.4f}", f"{r['n']:,}"] for r in rec["parts"]],
            note="The species counts n_A = N_total (1 - x_B) and n_B = N_total x_B / 2 against their sum, "
                 "all from the same posterior and the same videos. The sum being recovered far "
                 "better than either part means the parts' errors are strongly anti-correlated -- "
                 "they trade off inside the posterior -- and any quantity that divides one part by "
                 "the total inherits that cancellation. This is the whole reason a composition can "
                 "be reported from counts that individually cannot be.")
        if rec["stratum"].get("n"):
            reporter.stat(
                "recovery in the dimer-rich region", f"MAE {rec['stratum']['mae']:.1f} pp, "
                f"bias {rec['stratum']['bias']:+.1f} pp (n = {rec['stratum']['n']:,})",
                note=f"restricted to held-out videos whose TRUE f_B exceeds "
                     f"{100 * rec['stratum']['threshold']:.0f}% -- the corner the activated condition "
                     f"occupies. Selection is on truth, never on the estimate, so it is independent "
                     f"of the estimator's own error. A mean over the whole prior could hide a weak "
                     f"corner; this says whether the relevant one is weak.")
        # The starting-composition recovery: the allocated pairs c and the capacity in COUNTS, and
        # the allocation R_active per true-capacity stratum where the condition infers it.
        reporter.table(
            "Starting-composition recovery: realized counts from the MAP against the truth (counts; "
            "zero-safe)",
            ["quantity", "true-capacity stratum", "n", "MAE", "p95 abs. error", "bias", "Spearman rho"],
            [[_REALIZATION_LABEL[r["key"]], r["stratum"], f"{r['n']:,}", _num(r["mae"], "{:.2f}"),
              _num(r["p95"], "{:.2f}"), _num(r["bias"], "{:+.2f}"), _num(r["rho"], "{:.3f}")]
             for r in rec["counts"]],
            note="Both sides are realized by the same population helper from the physical values of "
                 "the truth and of the MAP; errors are in COUNTS, never in dex, because c = 0 is a "
                 "legitimate value (a capacity of 0, or an allocation that rounds to no pair). "
                 + ("Under this condition the allocation is declared "
                    f"{spec.binding.declared_allocation:g}, so c = 0 for every truth and every "
                    "estimate and no capacity strata are reported for R_active."
                    if fab_like else
                    "Strata are by the TRUE capacity, never by the estimate; at capacity 0 the "
                    "allocation has no effect on the window, so a prior-like posterior is the "
                    "expected outcome there."))
        if rec["allocation"] is not None:
            reporter.table(
                "Starting-composition recovery: the allocation R_active (MAP) per true-capacity stratum "
                "(percentage points)",
                ["true-capacity stratum", "n", "MAE (pp)", "p95 abs. error (pp)", "bias (pp)"],
                [[r["stratum"], f"{r['n']:,}", _num(r["mae"], "{:.2f}"), _num(r["p95"], "{:.2f}"),
                  _num(r["bias"], "{:+.2f}")]
                 for r in rec["allocation"]],
                note="R_active is a linear coordinate on [0, 1]: errors are absolute, in percentage "
                     "points, with no dex band and no multiplicative factor. The zero stratum is the "
                     "control: there the allocation has no effect and the estimate is expected to "
                     "return the prior.")

    # -- provenance ------------------------------------------------------------------------------
    reporter.stat("experimental source", str(spec.experiment_npz))
    reporter.stat("synthetic source", str(spec.recovery_npz) if data["recovery"] is not None
                  else "absent")
    reporter.stat("windows analyzed", f"{data['n_windows']} "
                  f"({len(kinds)} recording kinds x {ref['n_cells_grid']} recordings x "
                  f"{ref['n_chunks']} windows)")
    reporter.stat("posterior draws per window", f"{data['n_draws']:,}")
    reporter.stat("species mapping", "A = monomer (one receptor), B = dimer (two receptors)",
                  note="the monomer/dimer classification of the reaction-diffusion model (B1 and B2 are both dimers under MET-INLB), each class present in "
                       "the fast / slow / immobile mobility modes; the modes are particle TYPES, "
                       "not species, and do not enter the composition. A dimer holds two "
                       "receptors, which is what separates f_B (a share of complexes) from f_R (a "
                       "share of receptors)")
    reporter.stat("space", f"physical stoichiometry via parameterization.to_physical on the estimator's "
                           f"stoichiometry rows {list(spec.binding.count_keys)} (the receptor total a log "
                           f"row, the fractions linear), then the population helper per draw: the "
                           f"reference pair (N_total, 2 f / (1 + f)) and the starting pair (N_total, "
                           f"2 f_start / (1 + f_start)); fractions and counts formed per draw")
    reporter.stat("labeling plan (expected fluorescence)", spec.plan.describe())

    # -- figures ---------------------------------------------------------------------------------
    rec_rows = None if data["recovery"] is None else data["recovery"]["rows"]
    conditions_caption = (
        "Colored bars are the condition means with the standard error across recordings; the grey "
        "whisker on each bar is the estimator's mean absolute error for that quantity on held-out "
        "synthetic videos with known truth. The two are different kinds of error on purpose -- "
        "biological spread across cells against in-model measurement error -- and the comparison "
        "worth reading is the size of the between-condition difference against the grey whisker."
        if rec_rows is not None else
        "Condition means with the standard error across recordings. No synthetic validation was "
        "available for this run, so no in-model error is drawn.")
    if available:
        if fab_like:
            reporter.save_figure(
                "composition_conditions",
                _figure_conditions(mean, sem, labels, rec_rows,
                                   "Inferred population composition per condition (start = reference)"),
                caption="Inferred composition per condition; the starting population is the reference "
                        "population under this condition (allocation declared). " + conditions_caption + misspec)
        else:
            reporter.save_figure(
                "composition_conditions_start",
                _figure_conditions(start["mean"], start["sem"], labels, None,
                                   "STARTING (window-start) population composition per condition"),
                caption="The starting population the generator places (f_B = the derived f_start, after the "
                        "allocation), per condition: colored bars are the condition means with the standard "
                        "error across recordings. No in-model error is drawn on this figure; the synthetic "
                        "validation is reported for the reference population (the next figure) and for the "
                        "realized counts (the starting-composition recovery tables)." + misspec)
            reporter.save_figure(
                "composition_conditions",
                _figure_conditions(mean, sem, labels, rec_rows,
                                   "REFERENCE population composition per condition"),
                caption="The reference population (f_B = the inferred basal reference fraction f, before "
                        "the allocation), per condition. " + conditions_caption + misspec)
    if data["recovery"] is not None:
        reporter.save_figure(
            "composition_recovery",
            _figure_recovery(data["recovery"]["true_comp"], data["recovery"]["inferred_comp"],
                             data["recovery"]["rows"], rng, data["recovery"]["stratum"]),
            caption="Held-out synthetic videos: the true share against the inferred share, for the "
                    "monomer and the dimer-complex fraction of the reference population. Scatter about "
                    "the dashed line is the in-model error of the readout; the shaded band on the right "
                    "panel is the dimer-rich region the activated condition occupies, with its own error "
                    "restricted to that region.")
    if available:
        reporter.save_figure(
            "composition_time_course",
            _figure_time_course(ref["tc_mean"], ref["tc_sem"], labels, spec.window_seconds, pc.DIMER_INDEX,
                                "Composition across the recording (reference population; mean ± SEM over "
                                "recordings)"),
            caption="The dimer-complex share of the reference population across the recording, per "
                    "condition (mean over recordings, band = SEM). A condition that changes while the other "
                    "stays flat cannot be explained by anything shared between the two acquisitions, which "
                    "is what makes the flat condition a control here." + misspec)
        if not fab_like:
            reporter.save_figure(
                "composition_time_course_start",
                _figure_time_course(start["tc_mean"], start["tc_sem"], labels, spec.window_seconds,
                                    pc.DIMER_INDEX,
                                    "Composition across the recording (starting population; mean ± SEM "
                                    "over recordings)"),
                caption="The dimer-complex share of the STARTING population (f_start, after the allocation) "
                        "across the recording, per condition (mean over recordings, band = SEM)." + misspec)
        reporter.save_figure(
            "composition_per_cell",
            _figure_per_cell(ref["cell_values"], labels, pc.COMPOSITION_KEYS.index("monomer_fraction"),
                             pc.HETEROGENEITY_BANDS, rng),
            caption="Every recording's monomer share (reference population), so the condition mean is read "
                    "against its spread. Mass at both extremes means the mean falls between two groups of "
                    "cells rather than describing a typical one." + misspec)

    reporter.summary()
    reporter.write_report()
    return spec.report_dir


# ---- engine ------------------------------------------------------------------------------------

def _flag_summary(valid, labels_, kinds, per_key_flags):
    """Flagged-draw counts per recording kind and per window from the ``(N, S)`` validity mask."""
    n_windows, n_draws = valid.shape
    flagged_per_window = n_draws - valid.sum(axis=1)
    count, draws, windows, windows_any, windows_all, share_kind = [], [], [], [], [], []
    per_key = {key: [] for key in per_key_flags}
    for e in range(len(kinds)):
        sel = labels_["kind_index"] == e
        n_sel = int(sel.sum())
        c = int(flagged_per_window[sel].sum())
        count.append(c)
        draws.append(n_sel * n_draws)
        windows.append(n_sel)
        windows_any.append(int((flagged_per_window[sel] > 0).sum()))
        windows_all.append(int((flagged_per_window[sel] == n_draws).sum()))
        share_kind.append(c / max(n_sel * n_draws, 1))
        for key, flags in per_key_flags.items():
            per_key[key].append(int(flags[sel].sum()))
    total = int(flagged_per_window.sum())
    return dict(per_window=flagged_per_window.astype(np.int64), total=total, n_total=int(valid.size),
                share=total / max(valid.size, 1), count=count, draws=draws, windows=windows,
                windows_any=windows_any, windows_all=windows_all, share_kind=share_kind, per_key=per_key)


def run_population_composition(cfg, args):
    """Shared entry point. ``cfg`` is a WorkflowConfig; ``args`` the parsed CLI namespace."""
    spec = _population_composition_spec(cfg, args)
    binding = spec.binding

    div = "=" * 78
    print(div)
    print(f" {spec.alias} — Experiment Population Composition ({cfg.tag})")
    print(f" condition    : {condition_display(spec.condition)} (the estimator's; {len(spec.parameter_keys)} parameters)")
    print(f" timing_label : {spec.timing_label}   window : {spec.window_seconds:g} s")
    print(f" reads        : {spec.experiment_npz}")
    print(f"                {spec.recovery_npz}  (synthetic validation)")
    print(f" writes       : {spec.report_dir}")
    print(div)

    if args.dry_run:
        print("\n[DRY RUN] input validation only:")
        print(f"  experiment .npz : {spec.experiment_npz}  "
              f"[{'OK' if spec.experiment_npz.exists() else 'MISSING'}]")
        print(f"  recovery .npz   : {spec.recovery_npz}  "
              f"[{'OK' if spec.recovery_npz.exists() else 'absent (validation half skipped)'}]")
        print(f"  stoichiometry   : {', '.join(binding.count_keys)}  (N_total, f"
              + (", R_active" if len(binding.count_keys) == 3 else
                 f"; R_active declared {binding.declared_allocation:g}") + " -> the reference pair "
              "(N_total, 2f/(1+f)) and the realized starting population per draw)")
        print(f"  quantities      : "
              f"{', '.join(q.symbol + ' = ' + q.formula for q in pc.COMPOSITION)}")
        print(f"  realization     : {len(REALIZATION_KEYS)} per-draw quantities (N_total ... expected "
              f"fluorescent complexes)")
        print(f"  labeling plan   : {spec.plan.describe()}")
        print(f"  bootstrap       : {args.bootstrap:,} resamples of recordings"
              if args.bootstrap else "  bootstrap       : off")
        print("[DRY RUN] nothing read, nothing written.\n")
        return 0

    rng = np.random.default_rng(args.seed)
    cloud, labels_ = load_experiment_draws(spec.experiment_npz, spec.parameter_keys)
    kinds = labels_["kinds"]
    n_kinds = len(kinds)
    misspec = misspecification_note(kinds, spec.condition)
    if misspec:
        print("NOTE:" + misspec)

    # ---- realize the cloud ONCE: every per-draw quantity, the validity mask, both pairs ----------
    block = binding.select(cloud)                                             # (N, S, k)
    fields, valid = binding.realize(block, plan=spec.plan)
    ref_pair = binding.reference_pair(block)                                  # (N, S, 2), NaN where flagged
    start_pair = start_pair_of(fields)
    per_key_flags = {}
    n_total_phys, f_phys, r_phys = binding.components(block)
    per_key_flags[binding.count_keys[1]] = ~(np.isfinite(f_phys) & (f_phys >= 0.0) & (f_phys <= 1.0))
    if len(binding.count_keys) == 3:
        per_key_flags[binding.count_keys[2]] = ~(np.isfinite(r_phys) & (r_phys >= 0.0) & (r_phys <= 1.0))
    flagged = _flag_summary(valid, labels_, kinds, per_key_flags)

    reference = _composition_ladder(ref_pair, valid, labels_, n_kinds)
    start = _composition_ladder(start_pair, valid, labels_, n_kinds)
    if binding.declared_allocation is not None:
        # start = reference under a declared allocation of 0: a defect in the helper would show here.
        same = np.allclose(reference["window"], start["window"], equal_nan=True)
        if not same:
            raise RuntimeError(f"[{spec.condition}] the starting and the reference composition differ "
                               f"although the allocation is declared {binding.declared_allocation:g}.")
    reference["boot_lo"], reference["boot_hi"] = pc.bootstrap_interval(reference["cell_values"],
                                                                       args.bootstrap, rng)
    # The per-draw realization ladder: window means over the valid draws, then the same ladder.
    ladder_window = np.stack([pc.window_draw_mean(fields[name], valid) for name in REALIZATION_KEYS],
                             axis=-1)                                           # (N, Q)
    ladder_grid, _c, _t = pc.composition_grid(ladder_window, labels_["kind_index"], labels_["cell"],
                                              labels_["chunk"], n_kinds)
    ladder_cells = pc.per_cell(ladder_grid)
    ladder_mean, ladder_sem, ladder_n = pc.condition_summary(ladder_cells)
    ladder = dict(window=ladder_window, cell_values=ladder_cells, mean=ladder_mean, sem=ladder_sem,
                  n_cells=ladder_n)
    del fields

    # Sensitivity variants: the support restrictions recompute the whole ladder from a restricted
    # draw set (always intersected with the validity mask); the compositional center re-summarizes
    # the same per-recording values a different way.
    variants = []
    for name, mask, note in pc.support_masks(cloud, spec.prior_low, spec.prior_high, spec.count_index):
        mask = np.asarray(mask, dtype=bool) & valid
        wc = pc.window_draw_mean(pc.composition(ref_pair), mask)
        g, _c, _t = pc.composition_grid(wc, labels_["kind_index"], labels_["cell"], labels_["chunk"],
                                        n_kinds)
        cv = pc.per_cell(g)
        m, _s, _n = pc.condition_summary(cv)
        retained, empty, windows = [], [], []
        for e in range(n_kinds):
            sel = labels_["kind_index"] == e
            retained.append(float(mask[sel].mean()) if sel.any() else float("nan"))
            empty.append(int(np.isnan(wc[sel, pc.DIMER_INDEX]).sum()))
            windows.append(int(sel.sum()))
        variants.append(dict(name=name, note=note, mean=m, retained=np.array(retained),
                             empty=empty, windows=windows))
    variants.append(dict(name="compositional center (closed geometric mean)",
                         note="the natural center on the simplex, over the same recordings",
                         mean=pc.compositional_center(reference["cell_values"]), retained=None, empty=None,
                         windows=None))
    reference["variants"] = variants
    reference["monotonicity"] = [(pc.DIMER_INDEX, pc.monotonicity(reference["grid"], pc.DIMER_INDEX)),
                                 (pc.TOTAL_INDEX, pc.monotonicity(reference["grid"], pc.TOTAL_INDEX))]
    reference["contrast"] = pc.condition_contrast(reference["cell_values"]) if n_kinds >= 2 else []
    reference["heterogeneity"] = pc.heterogeneity(reference["cell_values"],
                                                  pc.COMPOSITION_KEYS.index("monomer_fraction"))

    recovery = None
    loaded = load_recovery(spec.recovery_npz, spec.parameter_keys)
    if loaded is not None:
        true_flow, map_estimate = loaded
        rows, true_comp, inf_comp = pc.recovery_statistics(true_flow, map_estimate, spec.count_index,
                                                           spec.to_physical)
        t_fields, t_valid = binding.realize(binding.select(true_flow))
        e_fields, e_valid = binding.realize(binding.select(map_estimate))
        if not t_valid.all():
            raise RuntimeError(f"[{spec.condition}] {int((~t_valid).sum())} held-out TRUTHS fail the "
                               f"validity rule; a Theta_Set is drawn inside the prior, so the product "
                               f"does not belong to this condition's table.")
        count_keys = ["n_pairs_allocated", "capacity", "n_subunits", "n_monomers", "n_dimers_two_probe"]
        strata = None if binding.declared_allocation is not None else capacity_strata(t_fields["capacity"])
        counts = pc.count_recovery_statistics({k: t_fields[k] for k in count_keys},
                                              {k: e_fields[k] for k in count_keys}, count_keys, strata)
        allocation = None
        if strata is not None:
            t_r, e_r = t_fields["relative_count_active_dimers_initial"], e_fields["relative_count_active_dimers_initial"]
            allocation = []
            for label, sel in [("all", np.ones(t_r.shape, dtype=bool))] + list(strata):
                ok = sel & np.isfinite(t_r) & np.isfinite(e_r)
                err = 100.0 * (e_r[ok] - t_r[ok])
                allocation.append(dict(stratum=label, n=int(ok.sum()),
                                       mae=float(np.mean(np.abs(err))) if ok.any() else float("nan"),
                                       p95=float(np.quantile(np.abs(err), 0.95)) if ok.any() else float("nan"),
                                       bias=float(np.mean(err)) if ok.any() else float("nan")))
        recovery = dict(
            rows=rows, true_comp=true_comp, inferred_comp=inf_comp,
            parts=pc.parts_versus_whole(true_flow, map_estimate, spec.count_index, spec.to_physical),
            stratum=pc.recovery_stratum(true_comp, inf_comp, pc.DIMER_INDEX, args.stratum_threshold),
            counts=counts, allocation=allocation,
            n_flagged_estimates=int((~e_valid).sum()), n_total=int(e_valid.size))

    # With no valid draw there is no composition to close or summarize: the closure is NaN by
    # construction (not a defect), and the report says so instead of failing on it.
    n_valid = int(valid.sum())
    data = dict(
        kinds=kinds, reference=reference, start=start, ladder=ladder, flagged=flagged, recovery=recovery,
        closure=pc.closure_residual(pc.composition(ref_pair)) if n_valid else float("nan"),
        n_valid_draws=n_valid,
        draws_finite=bool(np.isfinite(cloud).all()),
        n_windows=int(cloud.shape[0]), n_draws=int(cloud.shape[1]))

    print(f"\nLoaded {data['n_windows']} windows x {data['n_draws']:,} draws: {n_kinds} recording kinds "
          f"x {reference['n_cells_grid']} recordings x {reference['n_chunks']} windows.")
    print(f"  flagged draws        : {flagged['total']:,} of {flagged['n_total']:,} "
          f"({100.0 * flagged['share']:.3f} %), counted, never clipped")
    if not n_valid:
        print("  NO VALID COMPOSITION DRAWS: every draw is flagged, so every composition summary is "
              "unavailable (NaN in the arrays); the exclusion counts are reported, nothing is clipped.")
    for k, kind in enumerate(kinds):
        for name, pop in ((("start = reference", reference),) if binding.declared_allocation is not None
                          else (("starting", start), ("reference", reference))):
            print(f"  {kind_label(kind, spec.condition):34s} {name:18s}: "
                  + ", ".join(f"{pc.COMPOSITION[i].symbol} {_pct(pop['mean'][k, i])}%" for i in pc.FRACTION_INDICES)
                  + f", T {pop['mean'][k, pc.TOTAL_INDEX]:.0f}, N_total {pop['mean'][k, pc.RECEPTORS_INDEX]:.0f}")
    print("  synthetic validation : "
          + ("off (no MAP_Recovery)" if recovery is None else
             f"on — f_B MAE {recovery['rows'][pc.DIMER_INDEX]['mae']:.1f} pp over "
             f"{recovery['rows'][0]['n']:,} held-out videos"))
    print(f"  sensitivity variants : {', '.join(v['name'] for v in variants)}\n")

    spec.report_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(spec.report_dir / f"{spec.report_dir.name}.npz"),
        window_composition=reference["window"].astype(np.float32),
        cell_composition=reference["cell_values"].astype(np.float32),
        condition_mean=reference["mean"].astype(np.float32), condition_sem=reference["sem"].astype(np.float32),
        time_course_mean=reference["tc_mean"].astype(np.float32),
        time_course_sem=reference["tc_sem"].astype(np.float32),
        window_composition_start=start["window"].astype(np.float32),
        cell_composition_start=start["cell_values"].astype(np.float32),
        condition_mean_start=start["mean"].astype(np.float32),
        condition_sem_start=start["sem"].astype(np.float32),
        time_course_mean_start=start["tc_mean"].astype(np.float32),
        time_course_sem_start=start["tc_sem"].astype(np.float32),
        realization_window_mean=ladder["window"].astype(np.float32),
        realization_cell_mean=ladder["cell_values"].astype(np.float32),
        realization_condition_mean=ladder["mean"].astype(np.float32),
        realization_condition_sem=ladder["sem"].astype(np.float32),
        realization_keys=np.array(REALIZATION_KEYS),
        flagged_draws_per_window=flagged["per_window"],
        flagged_share=np.array(flagged["share_kind"], dtype=np.float32),
        flagged_share_total=np.float32(flagged["share"]),
        kind_index=labels_["kind_index"], cell=labels_["cell"], chunk=labels_["chunk"],
        kinds=np.array(kinds), quantities=np.array(pc.COMPOSITION_KEYS),
        condition=str(spec.condition), parameter_keys=np.array(spec.parameter_keys),
        count_keys=np.array(binding.count_keys))
    report_dir = _write_report(args, spec, data, rng)
    print(f"Report: {report_dir}")
    return 0


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--condition", required=True, choices=LABELING_CONDITIONS,
        help="the ESTIMATOR's condition (FAB = MET-FAB, INLB = MET-INLB): selects the "
             "condition-specific data and estimator namespace (the condition slot of the runtime "
             "grammar), the estimator's parameter table (eleven parameters under FAB, twelve under "
             "INLB) and the occupancy and declared values every derived composition is realized "
             "under. Recordings of the other kind in the Experiment product are a misspecification "
             "test and are labeled so.",
    )
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window / recording duration; sets the timing label locating the "
                        "inputs and naming the outputs.")
    p.add_argument("--bootstrap", type=int, default=10000,
                   help="resamples for the percentile bootstrap over recordings, which checks the "
                        "reported standard errors against a distribution-free interval "
                        "(0 = skip).")
    p.add_argument("--stratum-threshold", type=float, default=0.7,
                   help="true dimer-complex fraction above which the synthetic recovery is "
                        "additionally reported, so the accuracy of the region the activated "
                        "condition occupies is stated separately from the prior-wide mean.")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the bootstrap resampling and the figure jitter/subsampling "
                        "(every reported estimate is deterministic without it).")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; load nothing, "
                        "compute nothing.")
    return p


# ---- the one place workflow specifics live -----------------------------------------------------

def _population_composition_spec(cfg, args) -> CompositionSpec:
    """Resolve the workflow-specific half of the analysis under the ESTIMATOR's condition.

    Raises for a workflow that infers no stoichiometry, which is the honest outcome: the detector
    workflow treats the population implicitly and has nothing to compose, so a composition computed
    from its six imaging coordinates would be a well-formatted meaningless number.
    """
    from .parameterization import PARAMETERS, RunTiming
    from .workflow import parameter_keys as _wf_keys, parameter_table, prior_bounds

    condition = str(args.condition)
    keys = _wf_keys(cfg, condition)
    table = parameter_table(cfg, condition)
    try:
        binding = stoichiometry_binding(table, keys, condition)
    except ValueError as exc:
        raise SystemExit(
            f"the population composition needs the stoichiometry parameters (the receptor total, the "
            f"basal reference fraction and, under MET-INLB, the allocation), and the {cfg.tag} workflow "
            f"does not infer them ({exc}). This analysis is biology-only by construction: the detector "
            f"workflow infers imaging parameters and treats the population implicitly, so it has no "
            f"composition to report.") from None
    low, high = prior_bounds(cfg, condition)
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = cfg.paths.with_condition(condition)   # condition-specific namespace
    exp_dir = paths.experiment_recovery_dir(data_bank_root, timing.label)
    rec_dir = paths.map_recovery_dir(data_bank_root, timing.label)
    posit_dir = data_bank_root / paths.posit_subdir
    alias = paths.project_alias
    return CompositionSpec(
        condition=condition,
        parameter_keys=list(keys),
        binding=binding,
        count_index=binding.count_index,
        to_physical=binding.reference_pair,
        plan=resolve_labeling(condition),
        prior_low=np.array(low, dtype=float),
        prior_high=np.array(high, dtype=float),
        experiment_npz=exp_dir / (exp_dir.name + ".npz"),
        recovery_npz=rec_dir / (rec_dir.name + ".npz"),
        report_dir=posit_dir / f"{alias}_{timing.label}_Experiment_Population_Composition",
        alias=alias, timing_label=timing.label,
        stage="Experiment_Population_Composition",
        window_seconds=float(args.total_time_seconds),
    )
