#!/usr/bin/env python
"""Matched comparison of detector estimators on the experimental recordings, from their Experiment products.

Companion note ``SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Comparison.md``. For any set of trained
detector estimators, named by their artifact tags (the canonical one is ``baseline``), the comparison reads
each one's Experiment product (the MAP, the posterior quantiles and the sample geometric median of every
window of the experimental recordings) through the artifact schema, checks that the products hold the same
windows of the same recordings under the same window geometry and pool mode, and that each product's
checkpoint is the estimator artifact's weights when that artifact is present, and writes one record with the
estimators as columns.

The experimental recordings carry no ground truth. Nothing here measures recovery or accuracy. Per estimator
and per point estimate it measures where the estimates lie, how wide the posteriors are, how often the
estimates leave the prior box (the interval of parameter values that generated the training recordings; under
the Experiment stage's unrestricted sampling a value outside it is an extrapolation of the learned density, not
a posterior estimate under the box prior), how far they leave it and whether whole intervals do, how the three
point estimates of one posterior agree (typical and largest discrepancies), how much an
estimate moves between the windows of one recording, how it drifts with window position, how it is
structured between recordings, and how the estimators agree window by window and recording by recording.
Agreement between estimators is not correctness; a disagreement locates where they differ, not which is
right. The localization-table reference values the temporal-dynamics stage carries (DETECTOR_WORKFLOW.md
6.7) and, optionally, a user-selected imaging vector (a ``selection_user`` Nuisance_DLI, e.g. ``REF``) are
printed beside the estimates as references, not as truth.

Inputs are read-only; the output folder
``<data_bank>/<posit>/<alias>_<timing>_Experiment_Comparison_<COLUMNS>/`` (the columns in order, e.g.
``_BASELINE_CAP256``) is never overwritten (``--out-dir`` writes elsewhere). Usage::

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Experiment_Comparison.py \\
        --total-time-seconds 2 --base CAP256 --control baseline \\
        [--candidates CAP256KERNEL7EARLYCONVSTATSGB128] [--vector-tag REF] \\
        [--log CAP256=<job log> ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import __version__ as _PACKAGE_VERSION
from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from srm_and_sbi_monomer_dimer_alp import artifacts
from srm_and_sbi_monomer_dimer_alp import temporal_dynamics as tdk
from srm_and_sbi_monomer_dimer_alp.detector_nuisance_dli import NuisanceDLI, artifact_path
from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp.parameterization import (PARAMETERS, RunTiming, entry_to_physical,
                                                            is_log_row, to_physical)
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow, parameter_keys, parameter_table

VIEWS = ("MAP", "median", "SGM")
VIEW_NAMES = {"MAP": "MAP", "median": "posterior median", "SGM": "SGM"}
BASELINE = "baseline"
"""The column name of the canonical (untagged) estimator."""
JUMP_DEX = tdk.MATERIAL_DRIFT_DEX
"""An estimate that changes by more than this between two adjacent windows of one recording has jumped: the
material-drift bar of the Experiment stage, a factor of two."""


# --- inputs -----------------------------------------------------------------------------------------

def parse_logs(items):
    """``NAME=PATH`` pairs (``--log``) to a dict; a name may appear once."""
    logs = {}
    for item in items or []:
        name, sep, path = str(item).partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--log expects NAME=PATH, got {item!r}")
        if name in logs:
            raise SystemExit(f"--log names {name!r} twice")
        logs[name] = Path(path)
    return logs


def resolve(cfg, args):
    timing = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing)
    paths = cfg.paths.with_condition(args.condition)
    posit = Path(args.posit) if args.posit else PARAMETERS.machine.data_bank_root / paths.posit_subdir
    logs = parse_logs(args.log)
    columns = []
    for role, names in (("control", [args.control] if args.control else []), ("base", [args.base]),
                        ("candidate", list(args.candidates))):
        for name in names:
            if any(c["name"] == name for c in columns):
                raise SystemExit(f"estimator {name!r} named twice")
            tag = None if name == BASELINE else name
            label = paths.product_label(timing.label, tag)
            stem = f"{paths.project_alias}_{label}"
            columns.append(dict(name=name, role=role, tag=tag, label=label,
                                experiment=posit / f"{stem}_MAP_Experiment" / f"{stem}_MAP_Experiment.npz",
                                estimator=posit / f"{stem}_Estimator.npz", log=logs.pop(name, None)))
    if logs:
        raise SystemExit(f"--log names estimators that are not compared: {sorted(logs)}")
    plain = paths.product_label(timing.label, None)
    suffix = "_".join(c["name"].upper() for c in columns)          # the columns, in order: never a collision
    out_dir = Path(args.out_dir) if args.out_dir else \
        posit / f"{paths.project_alias}_{plain}_Experiment_Comparison_{suffix}"
    vector = artifact_path(posit, paths.project_alias, plain, args.vector_tag) if args.vector_tag else None
    return dict(timing=timing, paths=paths, posit=posit, columns=columns, out_dir=out_dir, vector=vector,
                vector_tag=args.vector_tag, condition=args.condition,
                keys=parameter_keys(cfg), table=parameter_table(cfg),
                lo=np.array(cfg.param_module.theta_lower_bound(), dtype=float),
                hi=np.array(cfg.param_module.theta_upper_bound(), dtype=float))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stops_from_log(path):
    """Total the per-rank ``MAP stops: <reason> <count>, ...`` lines of an Experiment job log."""
    if path is None:
        return None
    total = Counter()
    pattern = re.compile(r"MAP stops: (.*)$")
    for line in Path(path).read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if m and m.group(1).strip() != "none":
            for part in m.group(1).split(","):
                reason, count = part.strip().rsplit(" ", 1)
                total[reason] += int(count)
    return dict(sorted(total.items()))


def load_column(column, keys):
    """Read one estimator's Experiment product through the schema (a superseded product is refused); the
    parameter table must equal the workflow's key for key and the quantile levels must be the canonical
    ones. When the estimator artifact is present, its weights must be the checkpoint the product records."""
    src = str(column["experiment"])
    arrays, manifest = schema.load_product(column["experiment"], stage="experiment")
    schema.assert_parameter_keys(manifest, keys, source=src)
    ids = schema.observation_ids(arrays, "experiment")
    schema.assert_unique_observations(ids, source=src)
    levels = [round(float(v), 6) for v in manifest["quantile_levels"]]
    if levels != [round(q, 6) for q in schema.CANONICAL_QUANTILE_LEVELS]:
        raise SystemExit(f"{src}: quantile levels {levels} are not the canonical ones")
    q = np.asarray(arrays["posterior_quantiles"], dtype=float)
    identity = {"checkpoint": manifest.get("checkpoint_sha256"), "estimator_weights": None}
    if column["estimator"].exists():
        identity["estimator_weights"] = artifacts.load_estimator_manifest(column["estimator"]).get("weights_sha256")
        if identity["estimator_weights"] != identity["checkpoint"]:
            raise SystemExit(f"{column['name']}: the Experiment product records checkpoint "
                             f"{str(identity['checkpoint'])[:12]} but the estimator artifact holds weights "
                             f"{str(identity['estimator_weights'])[:12]}")
        identity["status"] = "the product's checkpoint is the estimator artifact's weights"
    else:
        identity["status"] = "estimator artifact not present beside the product; checkpoint recorded, not compared"
    if column["log"] is not None and not Path(column["log"]).exists():
        raise SystemExit(f"{column['name']}: job log not found: {column['log']}")
    return dict(column, ids=ids, quantiles=q, manifest=manifest,
                estimates={"MAP": np.asarray(arrays["map_estimate"], dtype=float),
                           "median": q[:, :, schema.median_level_index(manifest)],
                           "SGM": np.asarray(arrays["posterior_sgm"], dtype=float)},
                scores=np.asarray(arrays["scores"], dtype=float),
                kinds=[str(k) for k in np.asarray(arrays["kinds"]).ravel()],
                identity=identity, stops=stops_from_log(column["log"]), product_sha256=sha256_of(column["experiment"]))


def align(columns):
    """Reorder every column's windows to the base's ``(kind, cell, chunk)`` order. The window sets must be
    equal, and the condition labels, the window geometry and the pool mode must agree: otherwise the
    estimators were not applied to the same windows in the same way."""
    base = next(c for c in columns if c["role"] == "base")
    order = {tuple(int(v) for v in row): j for j, row in enumerate(base["ids"])}
    for col in columns:
        rows = [tuple(int(v) for v in row) for row in col["ids"]]
        if set(rows) != set(order):
            missing, extra = sorted(set(order) - set(rows))[:5], sorted(set(rows) - set(order))[:5]
            raise SystemExit(f"{col['name']}: windows differ from the base's (missing {missing}, extra {extra})")
        for field in ("kinds",):
            if col[field] != base[field]:
                raise SystemExit(f"{col['name']}: condition labels {col[field]} differ from the base's {base[field]}")
        for key in ("window_geometry", "pool_mode"):
            if col["manifest"].get(key) != base["manifest"].get(key):
                raise SystemExit(f"{col['name']}: {key} {col['manifest'].get(key)} differs from the base's "
                                 f"{base['manifest'].get(key)}")
        perm = np.empty(len(rows), dtype=np.int64)
        for j, r in enumerate(rows):
            perm[order[r]] = j                         # row of this column that holds the base's j-th window
        col["ids"] = col["ids"][perm]
        col["quantiles"] = col["quantiles"][perm]
        col["scores"] = col["scores"][perm]
        col["estimates"] = {v: a[perm] for v, a in col["estimates"].items()}
    return base["ids"]


def load_vector(path, keys):
    """The one log10 vector of a ``selection_user`` Nuisance_DLI, with its identity; ``None`` if not asked."""
    if path is None:
        return None
    if not Path(path).exists():
        raise SystemExit(f"imaging vector artifact not found: {path}")
    nd = NuisanceDLI.load(path)
    if nd.posterior_sample_pool_choice != "selection_user":
        raise SystemExit(f"{path}: a {nd.posterior_sample_pool_choice!r} artifact holds no single selected vector")
    if list(nd.parameter_keys) != list(keys):
        raise SystemExit(f"{path}: parameter keys {nd.parameter_keys} differ from {list(keys)}")
    return dict(path=str(path), log10=np.asarray(nd.fixed_vector_log10, dtype=float), identity=nd.identity(path),
                sha256=sha256_of(path))


def references_for(condition, keys):
    """The localization-table references of DETECTOR_WORKFLOW.md 6.7 that apply to the condition, as the
    temporal-dynamics stage carries them (``prob_photo_bleach`` has none)."""
    from srm_and_sbi_monomer_dimer_alp.temporal_dynamics_runner import _REFERENCE_DETECTOR
    out = {}
    for k in keys:
        ref = _REFERENCE_DETECTOR.get(k)
        if ref is None or (ref.get("applies_to") and f"MET-{condition}" not in ref["applies_to"]):
            continue
        out[k] = {"unit": ref["unit"], "values": [{"value": float(v), "label": lab} for v, _, lab in ref["estimates"]],
                  "upper_biased": (None if "upper_biased" not in ref else
                                   {"value": float(ref["upper_biased"][0]), "label": ref["upper_biased"][1]})}
    return out


# --- statistics -------------------------------------------------------------------------------------

def _iqr(v):
    q75, q25 = np.percentile(v, [75, 25])
    return float(q75 - q25)


def _pearson(a, b):
    """Pearson correlation; ``None`` when either series is constant (undefined)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _ranks(v):
    try:
        from scipy.stats import rankdata
        return rankdata(v)
    except ImportError:                                # continuous values: ties are not expected
        return np.argsort(np.argsort(v)).astype(float)


def location(vals, lo, hi, entry):
    """Location over windows, the prior-box shares as counts and fractions, and how far the estimates reach
    beyond the box (the farthest excess below the floor and above the top, dex; 0 when none)."""
    below, above = vals < lo, vals > hi
    return {"n": int(vals.size), "median_log10": float(np.median(vals)), "iqr_log10": _iqr(vals),
            "mean_log10": float(np.mean(vals)), "sd_log10": float(np.std(vals, ddof=1)),
            "median_physical": float(entry_to_physical(entry, float(np.median(vals)))),
            "below_prior_frac": float(np.mean(below)), "above_prior_frac": float(np.mean(above)),
            "outside_prior_frac": float(np.mean(below | above)),
            "n_below_prior": int(below.sum()), "n_above_prior": int(above.sum()), "n_outside_prior": int((below | above).sum()),
            "min_log10": float(vals.min()), "max_log10": float(vals.max()),
            "min_physical": float(entry_to_physical(entry, float(vals.min()))),
            "max_physical": float(entry_to_physical(entry, float(vals.max()))),
            "below_extent_log10": float(max(0.0, lo - vals.min())), "above_extent_log10": float(max(0.0, vals.max() - hi))}


def widths(q, i, lo, hi):
    """Interval widths from the stored quantiles, whose levels ``load_column`` checked are the canonical
    0.05 / 0.25 / 0.50 / 0.75 / 0.95."""
    w90, w50 = q[:, i, 4] - q[:, i, 0], q[:, i, 3] - q[:, i, 1]
    pw = float(hi - lo)
    return {"median_width_50_log10": float(np.median(w50)), "median_width_90_log10": float(np.median(w90)),
            "prior_width_log10": pw, "median_width_90_over_prior": float(np.median(w90) / pw)}


def boundary(q, i, lo, hi):
    """Whole stored intervals beyond the prior box: the windows whose central 90 % interval lies entirely below
    the floor (its 0.95 quantile below it) or entirely above the top (its 0.05 quantile above it), and the same
    for the central 50 % interval; counts and fractions. A view-independent property of the posteriors."""
    n = q.shape[0]
    b90, a90 = q[:, i, 4] < lo, q[:, i, 0] > hi
    b50, a50 = q[:, i, 3] < lo, q[:, i, 1] > hi
    return {"n_interval90_below": int(b90.sum()), "frac_interval90_below": float(b90.sum() / n),
            "n_interval90_above": int(a90.sum()), "frac_interval90_above": float(a90.sum() / n),
            "n_interval50_below": int(b50.sum()), "frac_interval50_below": float(b50.sum() / n),
            "n_interval50_above": int(a50.sum()), "frac_interval50_above": float(a50.sum() / n)}


def within_run(est, q, i):
    """The three point estimates of one posterior, against each other and against its stored intervals: the
    typical (median over windows) and the largest (max over windows) absolute discrepancies. Agreement of the
    three says nothing about the number of modes of a posterior."""
    mp, md, sg = est["MAP"][:, i], est["median"][:, i], est["SGM"][:, i]
    w50 = q[:, i, 3] - q[:, i, 1]
    inside = lambda v, a, b: (v >= q[:, i, a]) & (v <= q[:, i, b])   # noqa: E731
    return {"median_signed_MAP_minus_median_log10": float(np.median(mp - md)),
            "frac_MAP_above_median": float(np.mean(mp > md)),
            "median_abs_MAP_minus_median_log10": float(np.median(np.abs(mp - md))),
            "median_abs_MAP_minus_median_over_iqr": float(np.median(np.abs(mp - md) / np.maximum(w50, 1e-12))),
            "median_signed_MAP_minus_median_over_iqr": float(np.median((mp - md) / np.maximum(w50, 1e-12))),
            "MAP_inside_50_frac": float(np.mean(inside(mp, 1, 3))), "MAP_inside_90_frac": float(np.mean(inside(mp, 0, 4))),
            "SGM_inside_90_frac": float(np.mean(inside(sg, 0, 4))),
            "median_abs_SGM_minus_median_log10": float(np.median(np.abs(sg - md))),
            "median_abs_MAP_minus_SGM_log10": float(np.median(np.abs(mp - sg))),
            "max_abs_MAP_minus_median_log10": float(np.max(np.abs(mp - md))),
            "max_abs_SGM_minus_median_log10": float(np.max(np.abs(sg - md))),
            "max_abs_MAP_minus_SGM_log10": float(np.max(np.abs(mp - sg)))}


def per_recording(values, cells):
    """Per-recording means and within-recording SDs (and the cell ids, in order)."""
    uniq = np.unique(cells)
    means = np.array([values[cells == c].mean() for c in uniq])
    sds = np.array([values[cells == c].std(ddof=1) if (cells == c).sum() > 1 else np.nan for c in uniq])
    return uniq, means, sds


def recording_medians(values, cells):
    """The median over the windows of each recording, in cell order."""
    return np.array([np.median(values[cells == c]) for c in np.unique(cells)])


def fluctuation(grid_p):
    """``grid_p`` is ``(cells, chunks)`` of one estimate of one parameter. Median over recordings of the SD
    across a recording's windows; adjacent-window jumps larger than ``JUMP_DEX``."""
    sd = np.nanstd(grid_p, axis=1, ddof=1)
    step = np.abs(np.diff(grid_p, axis=1))
    jumps = np.nan_to_num(step, nan=0.0) > JUMP_DEX
    return {"median_within_recording_sd_log10": float(np.nanmedian(sd)),
            "jumps_over_threshold": int(jumps.sum()), "recordings_with_a_jump": int(jumps.any(axis=1).sum()),
            "jump_threshold_dex": float(JUMP_DEX)}


def pair_agreement(a, b, cells):
    """Window by window and recording by recording, ``b - a`` for the later column ``b`` against ``a``."""
    d = b - a
    _, ma, _ = per_recording(a, cells)
    _, mb, _ = per_recording(b, cells)
    sp = _pearson(_ranks(a), _ranks(b))
    return {"pearson_r": _pearson(a, b), "spearman_r": sp, "median_shift_log10": float(np.median(d)),
            "mean_shift_log10": float(np.mean(d)), "median_abs_difference_log10": float(np.median(np.abs(d))),
            "frac_later_higher": float(np.mean(b > a)),
            "recording_pearson_r": _pearson(ma, mb), "recording_median_shift_log10": float(np.median(mb - ma))}


def compile_comparison(columns, keys, table, lo, hi, vector=None, references=None):
    order = [c["name"] for c in columns]
    by = {c["name"]: c for c in columns}
    base = next(c for c in columns if c["role"] == "base")
    ids = base["ids"]
    kind, cells, chunks = ids[:, 0], ids[:, 1], ids[:, 2]
    n_kinds = int(kind.max()) + 1
    D = len(keys)
    log_rows = np.array([is_log_row(e) for e in table], dtype=bool)
    to_phys = lambda u: to_physical(u, table)   # noqa: E731
    result = {"order": order, "base": base["name"],
              "control": next((c["name"] for c in columns if c["role"] == "control"), None),
              "n_windows": int(ids.shape[0]), "n_recordings": int(np.unique(cells).size),
              "chunks": sorted(int(v) for v in np.unique(chunks)), "kinds": base["kinds"],
              "parameter_keys": list(keys), "prior_low": lo.tolist(), "prior_high": hi.tolist(),
              "columns": {}, "location": {}, "widths": {}, "boundary": {}, "within_run": {}, "fluctuation": {},
              "pairs": {}, "drift": {}, "between_recordings": {}, "references": references or {},
              "vector": None, "vector_shift": {}}
    for c in columns:
        m = c["manifest"]
        result["columns"][c["name"]] = {
            "role": c["role"], "tag": c["tag"], "experiment": str(c["experiment"]), "product_sha256": c["product_sha256"],
            "estimator": str(c["estimator"]) if c["estimator"].exists() else None, "identity": c["identity"],
            "job": (m.get("execution") or {}).get("job_id"), "invocation": (m.get("run_identity") or {}).get("invocation_id"),
            "written_at": m.get("written_at"), "pool_mode": m.get("pool_mode"), "n_summary_draws": m.get("n_summary_draws"),
            "optimizer": m.get("optimizer"), "window_geometry": m.get("window_geometry"),
            "code": {k: v for k, v in (m.get("code") or {}).items() if not isinstance(v, (dict, list))},
            "log": None if c["log"] is None else str(c["log"]), "map_stops": c["stops"]}
    grids = {}
    for c in columns:
        g, _, _ = tdk.point_estimate_grids(c["estimates"]["MAP"], kind, cells, chunks, n_kinds, c["quantiles"],
                                           c["estimates"]["SGM"],
                                           median_index=schema.CANONICAL_QUANTILE_LEVELS.index(0.50))
        grids[c["name"]] = {"MAP": g["MAP"], "median": g["posterior median"], "SGM": g["SGM"]}
    for v in VIEWS:
        result["location"][v] = {n: {keys[i]: location(by[n]["estimates"][v][:, i], lo[i], hi[i], table[i])
                                     for i in range(D)} for n in order}
        result["fluctuation"][v] = {n: {keys[i]: fluctuation(grids[n][v][:, :, :, i].reshape(-1, grids[n][v].shape[2]))
                                        for i in range(D)} for n in order}
        result["between_recordings"][v] = {}
        for n in order:
            per = {}
            for i in range(D):
                _, means, sds = per_recording(by[n]["estimates"][v][:, i], cells)
                rm = recording_medians(by[n]["estimates"][v][:, i], cells)
                per[keys[i]] = {"between_recording_sd_log10": float(np.std(means, ddof=1)),
                                "median_within_recording_sd_log10": float(np.nanmedian(sds)),
                                "recording_median_min_log10": float(rm.min()), "recording_median_max_log10": float(rm.max()),
                                "recording_median_min_physical": float(entry_to_physical(table[i], float(rm.min()))),
                                "recording_median_max_physical": float(entry_to_physical(table[i], float(rm.max())))}
            result["between_recordings"][v][n] = per
        result["pairs"][v] = {}
        for a, b in combinations(order, 2):
            result["pairs"][v][f"{b} - {a}"] = {keys[i]: pair_agreement(by[a]["estimates"][v][:, i],
                                                                        by[b]["estimates"][v][:, i], cells)
                                                for i in range(D)}
    for n in order:
        c = by[n]
        result["widths"][n] = {keys[i]: widths(c["quantiles"], i, lo[i], hi[i]) for i in range(D)}
        result["boundary"][n] = {keys[i]: boundary(c["quantiles"], i, lo[i], hi[i]) for i in range(D)}
        result["within_run"][n] = {keys[i]: within_run(c["estimates"], c["quantiles"], i) for i in range(D)}
        drift_grids = {"MAP": grids[n]["MAP"], "posterior median": grids[n]["median"], "SGM": grids[n]["SGM"]}
        result["drift"][n] = tdk.window_drift_rows(drift_grids, base["kinds"], keys, to_phys, log_rows)
    if vector is not None:
        vec = vector["log10"]
        result["vector"] = {"path": vector["path"], "sha256": vector["sha256"], "identity": vector["identity"],
                            "log10": vec.tolist(),
                            "physical": [float(entry_to_physical(table[i], vec[i])) for i in range(D)]}
        for v in VIEWS:
            result["vector_shift"][v] = {n: {keys[i]: {
                "median_minus_vector_log10": float(np.median(by[n]["estimates"][v][:, i]) - vec[i]),
                "median_minus_vector_percent": float(100.0 * (10.0 ** (np.median(by[n]["estimates"][v][:, i]) - vec[i]) - 1.0)),
                "frac_windows_above_vector": float(np.mean(by[n]["estimates"][v][:, i] > vec[i])),
                "n_windows_above_vector": int(np.sum(by[n]["estimates"][v][:, i] > vec[i]))}
                for i in range(D)} for n in order}
    arrays = {"kind_index": kind, "cell": cells, "chunk": chunks, "parameter_keys": np.array(keys),
              "quantile_levels": np.array(schema.CANONICAL_QUANTILE_LEVELS), "prior_low": lo, "prior_high": hi}
    for n in order:
        c = by[n]
        arrays.update({f"{n}_MAP": c["estimates"]["MAP"], f"{n}_median": c["estimates"]["median"],
                       f"{n}_SGM": c["estimates"]["SGM"], f"{n}_quantiles": c["quantiles"], f"{n}_scores": c["scores"]})
    if vector is not None:
        arrays["vector_log10"] = vector["log10"]
    return result, arrays, grids


# --- outputs ----------------------------------------------------------------------------------------

def _pct(x):
    """A share as a percentage; one decimal where rounding to an integer would read 0 % or 100 % for a share
    that is not exactly 0 or 1 (597 of 600 reads 99.5 %, not 100 %)."""
    if x is None:
        return "–"
    v = 100.0 * x
    if 0.0 < x < 1.0 and round(v) in (0, 100):
        return f"{v:.1f} %"
    return f"{v:.0f} %"


def _r(x):
    return "–" if x is None else f"{x:+.2f}"


def write_tables(result, keys, table, path):
    order, R = result["order"], result
    lo, hi = np.array(R["prior_low"]), np.array(R["prior_high"])
    L = [f"# Experiment comparison: {', '.join(order)}", "",
         f"{R['n_windows']} windows of {R['n_recordings']} {'/'.join(R['kinds'])} recordings (windows {R['chunks'][0]} "
         f"to {R['chunks'][-1]} of each), the same windows for every estimator. The experimental recordings carry no "
         f"ground truth: these tables measure where each estimator places its estimates, how wide its posteriors are, "
         f"how the three point estimates of one posterior agree, how estimates move within and between recordings, "
         f"and how the estimators agree; never recovery or accuracy. Values in log10 unless stated; the prior box is "
         f"the interval of parameter values that generated the training recordings, and a value outside it is an "
         f"extrapolation beyond them.", "", "## Columns", "",
         "| column | role | Experiment product | job | written | pool | draws | checkpoint | identity | MAP stops |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n in order:
        c = R["columns"][n]
        stops = "not read (no log given)" if c["map_stops"] is None else ", ".join(f"{k} {v}" for k, v in c["map_stops"].items())
        L.append(f"| {n} | {c['role']} | `{Path(c['experiment']).parent.name}` | {c['job'] or '-'} | {c['written_at'] or '-'} | "
                 f"{c['pool_mode']} | {c['n_summary_draws']:,} | `{(c['identity']['checkpoint'] or '-')[:12]}` | "
                 f"{c['identity']['status']} | {stops} |")
    for v in VIEWS:
        L += ["", f"## {VIEW_NAMES[v]}: median over windows (IQR); windows below / above the prior box, count (share)", "",
              "| parameter | prior box | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
        for i, k in enumerate(keys):
            cells = []
            for n in order:
                s = R["location"][v][n][k]
                cells.append(f"{s['median_log10']:+.3f} ({s['iqr_log10']:.3f}); {s['n_below_prior']} ({_pct(s['below_prior_frac'])}) / "
                             f"{s['n_above_prior']} ({_pct(s['above_prior_frac'])})")
            L.append(f"| `{k}` | [{lo[i]:+.3f}, {hi[i]:+.3f}] | " + " | ".join(cells) + " |")
    L += ["", "## Range and boundary (posterior median view)", "",
          "The range of the window medians and of the per-recording medians (physical units); how far the window "
          "medians reach beyond the prior box (the farthest excess below the floor / above the top, dex; 0 when none); "
          "and the windows whose whole stored central 90 % interval lies below the floor / above the top, count (share). "
          f"The Experiment stage samples the posterior without restriction to the box, so an estimate outside it is an "
          f"extrapolation of the learned density and crossing the box does not by itself show that the box is too "
          f"narrow; a whole interval beyond it is the stronger statement.", "",
          "| column | parameter | window medians | per-recording medians | excess below / above (dex) | whole 90 % interval below / above |",
          "|---|---|---|---|---|---|"]
    for n in order:
        for k in keys:
            s, b, br = R["location"]["median"][n][k], R["boundary"][n][k], R["between_recordings"]["median"][n][k]
            L.append(f"| {n} | `{k}` | {s['min_physical']:.4g} to {s['max_physical']:.4g} | "
                     f"{br['recording_median_min_physical']:.4g} to {br['recording_median_max_physical']:.4g} | "
                     f"{s['below_extent_log10']:.3f} / {s['above_extent_log10']:.3f} | "
                     f"{b['n_interval90_below']} ({_pct(b['frac_interval90_below'])}) / {b['n_interval90_above']} ({_pct(b['frac_interval90_above'])}) |")
    head = "| parameter | " + " | ".join(f"{n} MAP / median / SGM" for n in order)
    sep = "|---|" + "---|" * len(order)
    if R["vector"] is not None:
        head += f" | selected vector `{Path(R['vector']['path']).name}`"
        sep += "---|"
    head += " | localization-table reference (DETECTOR_WORKFLOW.md 6.7) |"
    sep += "---|"
    L += ["", "## In physical units: the transform of the median over windows", "", head, sep]
    for i, k in enumerate(keys):
        row = [" / ".join(f"{R['location'][v][n][k]['median_physical']:.4g}" for v in VIEWS) for n in order]
        if R["vector"] is not None:
            row.append(f"{R['vector']['physical'][i]:.4g}")
        ref = R["references"].get(k)
        if ref is None:
            row.append("none")
        else:
            txt = "; ".join(f"{e['value']:.4g} ({e['label']})" for e in ref["values"])
            if ref["upper_biased"]:
                txt += f"; {ref['upper_biased']['value']:.4g} ({ref['upper_biased']['label']}, upper-biased)"
            row.append(txt)
        L.append(f"| `{k}` | " + " | ".join(row) + " |")
    if R["vector"] is not None:
        L += ["", "## Against the selected vector: median over windows minus the vector value (dex, and as a percentage "
              "of the vector value); windows above the vector, count (share)", "",
              "| parameter | view | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
        for k in keys:
            for v in VIEWS:
                L.append(f"| `{k}` | {VIEW_NAMES[v]} | " + " | ".join(
                    f"{R['vector_shift'][v][n][k]['median_minus_vector_log10']:+.3f} "
                    f"({R['vector_shift'][v][n][k]['median_minus_vector_percent']:+.1f} %); "
                    f"{R['vector_shift'][v][n][k]['n_windows_above_vector']} ({_pct(R['vector_shift'][v][n][k]['frac_windows_above_vector'])})"
                    for n in order) + " |")
    L += ["", "## Within each run: the three point estimates of one posterior", "",
          "Signed median MAP − median (dex), the same in units of each window's posterior IQR, the share of windows "
          "whose MAP lies inside its own stored central 50 % and 90 % intervals, and the absolute discrepancies "
          "|MAP − median|, |SGM − median| and |MAP − SGM| (dex): the median over windows (typical) and the maximum "
          "over windows (largest). Agreement of the three says nothing about the number of modes of a posterior.", "",
          "| column | parameter | MAP − median | in IQR | MAP inside 50 % / 90 % | \\|MAP − median\\| typical / max | "
          "\\|SGM − median\\| typical / max | \\|MAP − SGM\\| typical / max |",
          "|---|---|---|---|---|---|---|---|"]
    for n in order:
        for k in keys:
            a = R["within_run"][n][k]
            L.append(f"| {n} | `{k}` | {a['median_signed_MAP_minus_median_log10']:+.3f} | "
                     f"{a['median_signed_MAP_minus_median_over_iqr']:+.2f} | {_pct(a['MAP_inside_50_frac'])} / "
                     f"{_pct(a['MAP_inside_90_frac'])} | {a['median_abs_MAP_minus_median_log10']:.3f} / {a['max_abs_MAP_minus_median_log10']:.3f} | "
                     f"{a['median_abs_SGM_minus_median_log10']:.3f} / {a['max_abs_SGM_minus_median_log10']:.3f} | "
                     f"{a['median_abs_MAP_minus_SGM_log10']:.3f} / {a['max_abs_MAP_minus_SGM_log10']:.3f} |")
    L += ["", "## Posterior widths: median over windows of the 50 % / 90 % interval width (dex), 90 % width as a share "
          "of the prior width", "", "| parameter | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for k in keys:
        L.append(f"| `{k}` | " + " | ".join(
            f"{R['widths'][n][k]['median_width_50_log10']:.3f} / {R['widths'][n][k]['median_width_90_log10']:.3f} "
            f"({_pct(R['widths'][n][k]['median_width_90_over_prior'])})" for n in order) + " |")
    L += ["", f"## Movement between the windows of one recording: median over recordings of the SD across its "
          f"windows (dex); jumps larger than {JUMP_DEX} dex between adjacent windows (count, recordings)", "",
          "| column | estimate | " + " | ".join(f"`{k}`" for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
    for n in order:
        for v in VIEWS:
            L.append(f"| {n} | {VIEW_NAMES[v]} | " + " | ".join(
                f"{R['fluctuation'][v][n][k]['median_within_recording_sd_log10']:.3f}; "
                f"{R['fluctuation'][v][n][k]['jumps_over_threshold']} ({R['fluctuation'][v][n][k]['recordings_with_a_jump']})"
                for k in keys) + " |")
    L += ["", "## Agreement between estimators", "",
          "Window by window: Pearson r over the matched windows, Spearman r, the median shift (the later column minus the "
          "earlier, dex) and the median |difference|; recording by recording: Pearson r of the per-recording means and "
          "their median shift. A correlation is undefined (–) when one series is constant.", ""]
    for v in VIEWS:
        L += [f"### {VIEW_NAMES[v]}", "",
              "| pair | parameter | r | Spearman | median shift | median \\|difference\\| | recording r | recording shift |",
              "|---|---|---|---|---|---|---|---|"]
        for pair, per in R["pairs"][v].items():
            for k in keys:
                p = per[k]
                L.append(f"| {pair} | `{k}` | {_r(p['pearson_r'])} | {_r(p['spearman_r'])} | {p['median_shift_log10']:+.3f} | "
                         f"{p['median_abs_difference_log10']:.3f} | {_r(p['recording_pearson_r'])} | "
                         f"{p['recording_median_shift_log10']:+.3f} |")
        L.append("")
    L += ["## Drift with window position", "",
          "Each recording's estimate is fitted against the window index and the first-to-last change taken; the rows "
          "aggregate those changes across the recordings (median change, IQR across recordings, share sharing the "
          "majority sign, share of recordings over 0.3 dex, two-sided signed-rank p). The same table the Experiment "
          "stage writes.", ""]
    for n in order:
        L += [f"### {n}", "", "| parameter | estimate | change (dex) | IQR across recordings | sign | over 0.3 dex | p |",
              "|---|---|---|---|---|---|---|"]
        for row in R["drift"][n]:
            L.append(f"| `{row['parameter']}` | {row['estimate']} | {row['change_median']:+.3f} | {row['change_iqr']:.3f} | "
                     f"{_pct(row['sign_consistency'])} | {_pct(row['material_fraction'])} | {row['wilcoxon_p']:.1e} |")
        L.append("")
    L += ["## Structure between recordings: SD of the per-recording means / median SD within a recording (dex)", "",
          "A statement about every window position (the median across recordings at each window index) is not a "
          "statement about every window; the window and per-recording medians range in the range-and-boundary table.", "",
          "| column | estimate | " + " | ".join(f"`{k}`" for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
    for n in order:
        for v in VIEWS:
            L.append(f"| {n} | {VIEW_NAMES[v]} | " + " | ".join(
                f"{R['between_recordings'][v][n][k]['between_recording_sd_log10']:.3f} / "
                f"{R['between_recordings'][v][n][k]['median_within_recording_sd_log10']:.3f}" for k in keys) + " |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def write_figures(result, arrays, grids, keys, table, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from srm_and_sbi_monomer_dimer_alp.visualization_inference import figure_window_drift

    order = result["order"]
    lo, hi = np.array(result["prior_low"]), np.array(result["prior_high"])
    colors = {n: plt.get_cmap("tab10")(j) for j, n in enumerate(order)}
    labels = [e["LABEL"] for e in table]
    fig_dir = Path(out_dir) / "figures"
    vec = None if result["vector"] is None else np.array(result["vector"]["log10"])
    for i, k in enumerate(keys):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
        vals = [arrays[f"{n}_{v}"][:, i] for n in order for v in VIEWS]
        span = (min(min(x.min() for x in vals), lo[i]), max(max(x.max() for x in vals), hi[i]))
        bins = np.linspace(span[0], span[1], 61)
        for ax, v in zip(axes, VIEWS):
            for n in order:
                ax.hist(arrays[f"{n}_{v}"][:, i], bins=bins, histtype="step", lw=1.6, color=colors[n], label=n)
            for b in (lo[i], hi[i]):
                ax.axvline(b, color="firebrick", lw=1.0, ls=":")
            if vec is not None:
                ax.axvline(vec[i], color="black", lw=1.2, ls="--", label=f"selected vector ({vec[i]:+.3f})")
            for ref in (result["references"].get(k) or {}).get("values", []):
                ax.axvline(np.log10(ref["value"]), color="0.45", lw=1.2, ls="-.", label=f"reference: {ref['label']}")
            ax.set_title(VIEW_NAMES[v])
            ax.set_xlabel(f"{k} (log10)")
        axes[0].set_ylabel("windows")
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"{labels[i]}  {k}: per-window estimates, {result['n_windows']} windows "
                     f"(dotted red = prior bounds)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"distribution_{k}.png", dpi=150)
        plt.close(fig)
    pairs = list(combinations(order, 2))
    for i, k in enumerate(keys):
        fig, axes = plt.subplots(len(pairs), 3, figsize=(15, 4.4 * len(pairs)), squeeze=False)
        for r, (a, b) in enumerate(pairs):
            for cidx, v in enumerate(VIEWS):
                ax = axes[r, cidx]
                x, y = arrays[f"{a}_{v}"][:, i], arrays[f"{b}_{v}"][:, i]
                span = [min(x.min(), y.min(), lo[i]) - 0.01, max(x.max(), y.max(), hi[i]) + 0.01]
                ax.plot(span, span, ls="--", lw=1.0, color="0.55", zorder=1)
                for bound in (lo[i], hi[i]):
                    ax.axvline(bound, color="firebrick", lw=1.0, ls=":", zorder=1)
                    ax.axhline(bound, color="firebrick", lw=1.0, ls=":", zorder=1)
                ax.scatter(x, y, s=9, alpha=0.45, edgecolor="none", color=colors[b], zorder=2)
                p = result["pairs"][v][f"{b} - {a}"][k]
                ax.set_title(f"{VIEW_NAMES[v]}: r = {_r(p['pearson_r'])}, median shift {p['median_shift_log10']:+.3f} dex",
                             fontsize=10)
                ax.set_xlabel(f"{a} (log10)")
                ax.set_ylabel(f"{b} (log10)")
                ax.set_xlim(span)
                ax.set_ylim(span)
        fig.suptitle(f"{labels[i]}  {k}: matched windows (dotted red = prior bounds, dashed grey = equality)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"pairs_{k}.png", dpi=150)
        plt.close(fig)
    for n in order:
        g = {VIEW_NAMES[v]: grids[n][v] for v in VIEWS}
        med = {name: tdk.window_medians(grid)[0] for name, grid in g.items()}
        bands = None
        if f"{n}_quantiles" in arrays:
            kind, cells, chunks = arrays["kind_index"], arrays["cell"], arrays["chunk"]
            bands = tdk.shared_posterior_bands(arrays[f"{n}_quantiles"], kind, cells, chunks, int(kind.max()) + 1)[0]
        fig = figure_window_drift(keys, labels, med, bands=bands, prior_ranges=list(zip(lo, hi)),
                                  title=f"{n}: the three point estimates against window position (lines = median across "
                                        f"the {result['n_recordings']} recordings; bands = posterior 50 % / 90 %, median "
                                        f"across recordings)", y_label="estimate (log10)")
        fig.savefig(fig_dir / f"window_drift_{n}.png", dpi=150)
        plt.close(fig)


def code_identity(repo):
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    head = git("rev-parse", "HEAD")
    return dict(package_version=_PACKAGE_VERSION, git_head=(head or None),
                git_dirty=(bool(git("status", "--porcelain")) if head else None),
                script_sha256=sha256_of(Path(__file__).resolve()))


def write_provenance(result, R, path):
    ident = result["code_identity"]
    L = ["# Provenance", "",
         f"Record: `{R['out_dir'].name}`. Compiled by `Script_Bank/Analysis/{Path(__file__).name}` "
         f"(package {ident['package_version']}, git {ident['git_head'] or 'not a checkout'}"
         f"{', working tree modified' if ident['git_dirty'] else ''}; script sha256 `{ident['script_sha256']}`). "
         f"Method: the companion note of the same name.", "", "## Inputs (read-only)", "",
         "| column | role | Experiment product | sha256 | job | invocation | written | checkpoint | identity | pool | draws | job log |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in result["order"]:
        c = result["columns"][n]
        L.append(f"| {n} | {c['role']} | `{Path(c['experiment']).parent.name}` | `{c['product_sha256'][:16]}` | {c['job'] or '-'} | "
                 f"`{c['invocation'] or '-'}` | {c['written_at'] or '-'} | `{(c['identity']['checkpoint'] or '-')[:16]}` | "
                 f"{c['identity']['status']} | {c['pool_mode']} | {c['n_summary_draws']:,} | "
                 f"{'`' + Path(c['log']).name + '`' if c['log'] else 'not given'} |")
    L += ["", "Optimizer and window geometry per column:", ""]
    for n in result["order"]:
        c = result["columns"][n]
        L.append(f"- **{n}**: optimizer `{json.dumps(c['optimizer'], default=str)}`; window geometry "
                 f"`{json.dumps(c['window_geometry'], default=str)}`; product code `{json.dumps(c['code'], default=str)}`")
    if result["vector"] is not None:
        L += ["", "## Selected vector", "",
              f"`{Path(result['vector']['path']).name}` (sha256 `{result['vector']['sha256'][:16]}`), the one vector of a "
              f"`selection_user` Nuisance_DLI, printed beside the estimates as the selected simulation values, not as truth: "
              f"`{json.dumps(result['vector']['identity'], default=str)}`"]
    L += ["", "## Alignment", "",
          f"Every column is reordered to the base's (kind, cell, chunk) order; the window sets must be equal and the "
          f"condition labels, the window geometry and the pool mode must agree, or the comparison is refused. "
          f"{result['n_windows']:,} windows of {result['n_recordings']} recordings.", "",
          "## Computed", "",
          "Everything is computed from the stored per-window estimates and quantiles: location, prior-box counts and shares "
          "and the reach beyond the box per point estimate, posterior widths and whole intervals beyond the box from the "
          "stored quantiles, the typical and largest discrepancies of the three point estimates of one posterior, the movement between adjacent windows and the SD within a recording, window-by-window and "
          "recording-by-recording agreement between the estimators, the drift with window position (the Experiment "
          "stage's kernel) and the structure between recordings. The MAP stop totals are read from the job logs when "
          "given. The localization-table references are those the temporal-dynamics stage carries "
          "(DETECTOR_WORKFLOW.md 6.7). No posterior is redrawn and no estimator is loaded."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def write_outputs(result, arrays, grids, R):
    out = R["out_dir"]
    (out / "figures").mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "comparison.npz", **arrays)
    (out / "comparison.json").write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    write_tables(result, R["keys"], R["table"], out / "comparison.md")
    write_figures(result, arrays, grids, R["keys"], R["table"], out)
    write_provenance(result, R, out / "PROVENANCE.md")
    (out / "README.md").write_text(
        f"# {out.name}\n\nThe detector estimators named in `comparison.md` (columns), compared on the same windows of "
        f"the experimental recordings from their Experiment products. No ground truth: the record measures location, "
        f"posterior width, prior-box counts and reach, whole intervals beyond the box, the typical and largest "
        f"discrepancies of the three point estimates, the ranges of window and per-recording medians, movement within and between "
        f"recordings, drift with window position and agreement between the estimators, never accuracy. "
        f"`comparison.json` holds every number, `comparison.npz` the aligned arrays, `figures/` the distributions, the "
        f"pairwise scatter and the window drift, `PROVENANCE.md` the inputs and the code identity. The reading belongs "
        f"to DETECTOR_WORKFLOW.md.\n", encoding="utf-8")


# --- entry point ------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True, help="model window of the estimators (s).")
    ap.add_argument("--condition", default="FAB", choices=LABELING_CONDITIONS)
    ap.add_argument("--base", required=True, help="the reference column (an artifact tag, or 'baseline').")
    ap.add_argument("--control", default=None, help="a second reference column (tag or 'baseline'); optional.")
    ap.add_argument("--candidates", nargs="*", default=[], help="further columns (artifact tags); may be empty.")
    ap.add_argument("--vector-tag", default=None,
                    help="tag of a selection_user Nuisance_DLI (e.g. REF) printed beside the estimates; optional.")
    ap.add_argument("--log", action="append", default=[], metavar="NAME=PATH",
                    help="an Experiment job log of column NAME, for its MAP stop totals; repeatable, optional.")
    ap.add_argument("--posit", default=None, help="read the products from this folder instead of the machine's Posit tier.")
    ap.add_argument("--out-dir", default=None, help="write here instead of the Posit tier's comparison folder "
                                                    "(named after the columns).")
    ap.add_argument("--dry-run", action="store_true", help="resolve the inputs and outputs; compare nothing.")
    args = ap.parse_args(argv)

    cfg = detector_workflow()
    R = resolve(cfg, args)
    assert all(is_log_row(e) for e in R["table"] if e["KEY"] in R["keys"]), "imaging rows are expected in log10"
    print(f"=== Experiment comparison | {R['paths'].project_alias}_{R['timing'].label} | package {_PACKAGE_VERSION}")
    present, missing = [], []
    for c in R["columns"]:
        ok = c["experiment"].exists()
        (present if ok else missing).append(c)
        print(f"  {c['role']:9s} {c['name']:34s} Experiment {'present' if ok else 'MISSING'}"
              f" | estimator {'present' if c['estimator'].exists() else 'absent'}"
              f" | job log {'given' if c['log'] else 'not given'}")
    if R["vector"] is not None:
        print(f"  selected vector {R['vector_tag']}: {'present' if R['vector'].exists() else 'MISSING'} ({R['vector'].name})")
    print(f"  writes: {R['out_dir']}/")
    if any(c["role"] == "base" for c in missing):
        raise SystemExit("the base column has no Experiment product; nothing to compare against")
    if any(c["role"] == "control" for c in missing):
        raise SystemExit("the control column has no Experiment product; drop --control or supply it")
    if R["vector"] is not None and not R["vector"].exists():
        raise SystemExit(f"imaging vector artifact not found: {R['vector']}")
    if R["out_dir"].exists():
        raise SystemExit(f"refusing to overwrite an existing comparison: {R['out_dir']}")
    if args.dry_run:
        print("dry run: nothing compared.")
        return 0
    columns = [load_column(c, R["keys"]) for c in present]
    align(columns)
    vector = load_vector(R["vector"], R["keys"])
    result, arrays, grids = compile_comparison(columns, R["keys"], R["table"], R["lo"], R["hi"], vector=vector,
                                               references=references_for(R["condition"], R["keys"]))
    result["missing_columns"] = [c["name"] for c in missing]
    result["code_identity"] = code_identity(Path(__file__).resolve().parents[2])
    write_outputs(result, arrays, grids, R)
    for n in result["order"]:
        loc = result["location"]["median"][n]
        print(f"  {n}: posterior median over windows " + ", ".join(
            f"{k} {loc[k]['median_log10']:+.3f} ({_pct(loc[k]['outside_prior_frac'])} outside)" for k in R["keys"]))
    if missing:
        print(f"  columns without an Experiment product, not compared: {[c['name'] for c in missing]}")
    print(f"written: {R['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
