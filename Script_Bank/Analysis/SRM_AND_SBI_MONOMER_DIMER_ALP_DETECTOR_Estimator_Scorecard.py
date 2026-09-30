#!/usr/bin/env python
"""The scorecard of the detector's neural posterior estimators, compiled from their records.

Part of the encoder screening (DETECTOR_WORKFLOW.md, the section on the encoder screening; companion note
``SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Estimator_Scorecard.md``). For any set of trained detector
estimators, named by their artifact tags (the canonical one is ``baseline``), the compiler reads each
one's Evaluation product (MAP, posterior quantiles and sample geometric median per EVAL recording, with
the truth) and, when present, its Posterior_Calibration product (the sample clouds, the truth
log-densities and the report's joint tests) and its embedding-probe record, checks that the records of a
column come from one checkpoint, aligns every estimator to the same recordings, and writes one table of the
scorecard's aspects with the estimators as columns:
parameter recovery in the three point-estimate views, marginal coverage and posterior widths, the joint
tests lifted from the calibration reports, the truth log-density with its paired difference against the
base and the control, the same recovery and coverage in predefined regimes of the prior, and, as a
separate row that selects nothing, what the frozen encoder carries linearly (the probe).

The scorecard scores the whole posterior estimator that a training produced: its encoder, the flow
trained jointly with it, and its training configuration. Nothing here isolates the encoder. Flags mark
the conditions the selection rule names (a parameter below the recovery floor, a large correlation drop, a
collapse, better point estimates with worse uncertainty, MAE differences under a practical threshold); the
thresholds are options recorded in the output, and the selection rule, not the compiler, decides.

Inputs are read-only; the output folder ``<data_bank>/<posit>/<alias>_<timing>_Estimator_Scorecard_<COLUMNS>/``
(the columns in order, e.g. ``_BASELINE_CAP256``) is never overwritten (``--out-dir`` writes elsewhere). Usage::

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Estimator_Scorecard.py \\
        --total-time-seconds 2 --base CAP256 --control baseline \\
        [--candidates CAP256KERNEL7STATS CAP256EARLYCONVSTATS] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import __version__ as _PACKAGE_VERSION
from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from srm_and_sbi_monomer_dimer_alp import artifacts
from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming, is_log_row
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow, parameter_keys, parameter_table

VIEWS = ("MAP", "median", "SGM")
BASELINE = "baseline"
"""The column name of the canonical (untagged) estimator."""
LIFTED_QUANTITIES = ("coverage_max_gap", "coverage_points", "tarp_ATC", "lc2st_reject_fraction",
                     "lc2st_median_pvalue", "marginal_1d_worst", "marginal_2d_worst", "dependence_excess",
                     "calibration_videos", "posterior_samples")
"""Joint tests copied verbatim from a Posterior_Calibration report: they come from the sample clouds and,
for L-C2ST, from a classifier the compiler does not retrain."""
DEFAULT_THRESHOLDS = dict(collapse_corr=0.3, collapse_drop=0.3, coverage_tolerance=0.05, min_effect=0.01)
"""The flag thresholds. A parameter is below the floor when the correlation of its posterior median with
the truth is under ``collapse_corr`` (unrecovered, as ``sigma_r`` in every estimator of record so far); it
has a large correlation drop when that correlation is more than ``collapse_drop`` below the reference
column's; it is collapsed when the drop leaves it below the floor (the ``capacity256`` bleaching precedent;
a drop from 0.90 to 0.55 is a large drop, not a collapse). A column has better points with worse
uncertainty when its MAE improves by at least ``min_effect`` dex while its 90 % coverage gap worsens by more
than ``coverage_tolerance``; a MAE difference under ``min_effect`` dex is under the practical threshold, a
statement about MAE alone. A constant estimate is scored with zero correlation by convention (its
correlation is undefined)."""
CALIBRATION_UNVERIFIED = "unverified: the calibration product carries no checkpoint provenance"


# --- inputs -----------------------------------------------------------------------------------------

def resolve(cfg, args):
    timing = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing)
    paths = cfg.paths.with_condition(args.condition)
    posit = Path(args.posit) if args.posit else PARAMETERS.machine.data_bank_root / paths.posit_subdir
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
                                evaluation=posit / f"{stem}_MAP_Recovery" / f"{stem}_MAP_Recovery.npz",
                                calibration=posit / f"{stem}_Posterior_Calibration" / f"{stem}_Posterior_Calibration.npz",
                                calibration_report=posit / f"{stem}_Posterior_Calibration" / "report.md",
                                probe=posit / f"{stem}_Embedding_Probe" / f"{stem}_Embedding_Probe.json",
                                estimator=posit / f"{stem}_Estimator.npz"))
    suffix = "_".join(c["name"].upper() for c in columns)          # the columns, in order: never a collision
    out_dir = Path(args.out_dir) if args.out_dir else \
        posit / f"{paths.project_alias}_{paths.product_label(timing.label, None)}_Estimator_Scorecard_{suffix}"
    return dict(timing=timing, paths=paths, posit=posit, columns=columns, out_dir=out_dir,
                keys=parameter_keys(cfg), table=parameter_table(cfg),
                lo=np.array(cfg.param_module.theta_lower_bound(), dtype=float),
                hi=np.array(cfg.param_module.theta_upper_bound(), dtype=float))


def load_column(column, keys):
    """Read one estimator's records. The Evaluation product is required (through the schema, so a
    superseded product is refused); the calibration product, its report, the probe record and the
    estimator manifest are optional and recorded as absent. The records of one column must come from one
    checkpoint: the Evaluation's checkpoint, the estimator's weights and the probe's weights are compared
    and a mismatch is refused; the calibration product carries no checkpoint provenance, so its identity is
    recorded as unverified. Its report is lifted only when the calibration arrays are present and the
    report's recorded video count equals their rows."""
    arrays, manifest = schema.load_product(column["evaluation"], stage="evaluation")
    schema.assert_parameter_keys(manifest, keys, source=str(column["evaluation"]))
    ids = schema.observation_ids(arrays, "evaluation")
    schema.assert_unique_observations(ids, source=str(column["evaluation"]))
    levels = [round(float(v), 6) for v in manifest["quantile_levels"]]
    if levels != [round(q, 6) for q in schema.CANONICAL_QUANTILE_LEVELS]:
        raise SystemExit(f"{column['evaluation']}: quantile levels {levels} are not the canonical ones")
    q = np.asarray(arrays["posterior_quantiles"], dtype=float)
    col = dict(column, ids=ids, truth=np.asarray(arrays["true_log10"], dtype=float), quantiles=q,
               estimates={"MAP": np.asarray(arrays["map_estimate"], dtype=float),
                          "median": q[:, :, schema.median_level_index(manifest)],
                          "SGM": np.asarray(arrays["posterior_sgm"], dtype=float)},
               scores=np.asarray(arrays["scores"], dtype=float), evaluation_manifest=manifest,
               calibration_truth=None, truth_log_density=None, lifted=None, probe_result=None,
               estimator_manifest=None)
    identity = {"evaluation_checkpoint": manifest.get("checkpoint_sha256"), "estimator_weights": None,
                "probe_weights": None, "calibration": CALIBRATION_UNVERIFIED}
    col["lifted_status"] = "no calibration report"
    if column["calibration"].exists():
        cal = np.load(column["calibration"])
        cal_keys = [str(k) for k in np.asarray(cal["parameter_keys"]).ravel()]
        if cal_keys != list(keys):
            raise SystemExit(f"{column['calibration']}: parameter keys {cal_keys} differ from {list(keys)}")
        col["calibration_truth"] = np.asarray(cal["truths"], dtype=float)
        col["truth_log_density"] = np.asarray(cal["truth_log_probs"], dtype=float)
    if column["calibration_report"].exists():
        if col["calibration_truth"] is None:
            col["lifted_status"] = ("report present, not lifted: without the calibration arrays its coverage of "
                                    "the Evaluation's recordings cannot be verified")
        else:
            lifted = lift_calibration_report(column["calibration_report"], keys)
            counted = lifted["quantitative"].get("calibration_videos")
            n_cal = int(col["calibration_truth"].shape[0])
            if counted is None or not np.isfinite(_float(counted)) or int(_float(counted)) != n_cal:
                raise SystemExit(f"{column['calibration_report']}: the report counts {counted!r} calibration videos; "
                                 f"the calibration arrays hold {n_cal}; the report does not describe these arrays")
            col["lifted"], col["lifted_status"] = lifted, f"lifted; the report's {n_cal} videos are the arrays' rows"
    if column["probe"].exists():
        col["probe_result"] = json.loads(column["probe"].read_text(encoding="utf-8"))
        identity["probe_weights"] = (col["probe_result"].get("meta") or {}).get("weights_sha256")
    if column["estimator"].exists():
        col["estimator_manifest"] = artifacts.load_estimator_manifest(column["estimator"])
        identity["estimator_weights"] = col["estimator_manifest"].get("weights_sha256")
    hashes = {k: v for k, v in identity.items() if k != "calibration" and v}
    if len(set(hashes.values())) > 1:
        raise SystemExit(f"{column['name']}: the records under this tag come from different weights: "
                         + ", ".join(f"{k} {str(v)[:12]}" for k, v in hashes.items()))
    identity["status"] = ((f"one checkpoint across {', '.join(hashes)}" if len(hashes) > 1
                           else "a single record carries a checkpoint; nothing to compare") + "; calibration unverified")
    col["identity"] = identity
    return col


def parse_md_tables(text):
    """Every pipe table of a Markdown text as a list of row dicts keyed by the header cells."""
    tables, block = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
        elif block:
            header = [c.strip() for c in block[0].strip().strip("|").split("|")]
            rows = []
            for ln in block[2:]:
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if len(cells) > len(header):                 # a pipe inside the last cell (e.g. "worst |ATC|")
                    cells = cells[:len(header) - 1] + ["|".join(cells[len(header) - 1:]).strip()]
                if len(cells) == len(header):
                    rows.append(dict(zip(header, cells)))
            tables.append(rows)
            block = []
    return tables


def lift_calibration_report(path, keys):
    """The joint tests, the per-parameter SBC statistic and the location-versus-width diagnosis, copied
    from a Posterior_Calibration report as written."""
    out = {"quantitative": {}, "sbc_ks_d": {}, "diagnosis": {}}
    for rows in parse_md_tables(Path(path).read_text(encoding="utf-8")):
        if rows and "metric" in rows[0] and "value" in rows[0]:
            for r in rows:
                if r["metric"] in LIFTED_QUANTITIES:
                    out["quantitative"][r["metric"]] = r["value"]
        if rows and "KS D (effect size)" in rows[0]:
            for r in rows:
                if r["parameter"] in keys:
                    out["sbc_ks_d"][r["parameter"]] = _float(r["KS D (effect size)"])
        if rows and "bias (z)" in rows[0] and "spread (z)" in rows[0]:
            for r in rows:
                if r["parameter"] in keys:
                    out["diagnosis"][r["parameter"]] = {"bias_z": _float(r["bias (z)"]),
                                                        "spread_z": _float(r["spread (z)"]),
                                                        "sharpness": r.get("sharpness"),
                                                        "defect": r.get("defect")}
    return out


def _float(text):
    try:
        return float(str(text).replace("+", "").replace("%", "").strip())
    except ValueError:
        return float("nan")


# --- alignment and regimes ---------------------------------------------------------------------------

def align(columns):
    """Reorder every column's rows to the base's ``(task, sim)`` order and check that the truths agree
    exactly; align each calibration cloud to the same rows by the six true parameters rounded to nine
    decimals, requiring unique truths and the same set of recordings as the Evaluation. Returns the base
    truths."""
    base = next(c for c in columns if c["role"] == "base")
    order = {tuple(int(v) for v in row): j for j, row in enumerate(base["ids"])}
    for col in columns:
        rows = [tuple(int(v) for v in row) for row in col["ids"]]
        if set(rows) != set(order):
            missing, extra = sorted(set(order) - set(rows))[:5], sorted(set(rows) - set(order))[:5]
            raise SystemExit(f"{col['name']}: recordings differ from the base's (missing {missing}, extra {extra})")
        perm = np.empty(len(rows), dtype=np.int64)
        for j, r in enumerate(rows):
            perm[order[r]] = j                         # row of this column that holds the base's j-th recording
        col["ids"] = col["ids"][perm]
        col["truth"] = col["truth"][perm]
        col["quantiles"] = col["quantiles"][perm]
        col["scores"] = col["scores"][perm]
        col["estimates"] = {v: a[perm] for v, a in col["estimates"].items()}
        if not np.array_equal(col["truth"], base["truth"]):
            raise SystemExit(f"{col['name']}: truths differ from the base's after identifier alignment")
        if col["calibration_truth"] is not None:
            n_cal, n_eval = col["calibration_truth"].shape[0], base["truth"].shape[0]
            if n_cal != n_eval:
                raise SystemExit(f"{col['name']}: the calibration product holds {n_cal} recordings, the Evaluation "
                                 f"{n_eval}; the same set is required before its lifted tests stand beside the "
                                 f"Evaluation's numbers")
            lut = {tuple(np.round(t, 9)): j for j, t in enumerate(col["calibration_truth"])}
            if len(lut) != n_cal:
                raise SystemExit(f"{col['name']}: calibration truths are not unique at nine decimals; cannot align by truth")
            try:
                p = np.array([lut[tuple(np.round(t, 9))] for t in base["truth"]])
            except KeyError as e:
                raise SystemExit(f"{col['name']}: a calibration row for truth {e} is missing") from None
            col["truth_log_density"] = col["truth_log_density"][p]
            col["calibration_truth"] = col["calibration_truth"][p]
    return base["truth"]


def regime_masks(truth, keys, lo, hi):
    """The predefined regimes: every recording; the low and high halves of the prior of ``mu_r`` and of
    ``sigma_r`` (split at the prior's midpoint in log10); the operating subgroup (the lower half of the
    brightness prior, DETECTOR_WORKFLOW.md, the acceptance rules of the direct estimators)."""
    mid = (lo + hi) / 2
    masks, definitions = {"all": np.ones(truth.shape[0], bool)}, {"all": "every recording"}
    for key in ("mu_r", "sigma_r"):
        d = keys.index(key)
        masks[f"{key}_low"] = truth[:, d] < mid[d]
        masks[f"{key}_high"] = truth[:, d] >= mid[d]
        definitions[f"{key}_low"] = f"true log10 {key} in [{lo[d]:g}, {mid[d]:g})"
        definitions[f"{key}_high"] = f"true log10 {key} in [{mid[d]:g}, {hi[d]:g}]"
    d = keys.index("mu_pc")
    masks["operating_subgroup"] = truth[:, d] < mid[d]
    definitions["operating_subgroup"] = f"true log10 mu_pc in [{lo[d]:g}, {mid[d]:g})"
    return masks, definitions


# --- statistics -------------------------------------------------------------------------------------

def recovery_stats(est, tru, lo, hi):
    """Recovery of one parameter (log10): MAE, bias, the slope of the estimate on the truth, their
    correlation, and the outside-prior fraction. A constant estimate carries nothing about the truth: its slope
    and correlation are reported as zero, not left undefined, so the flags read it as no recovery."""
    est, tru = np.asarray(est, float), np.asarray(tru, float)
    n = int(est.size)
    if n < 3:
        return dict(n=n, mae_dex=float("nan"), bias_dex=float("nan"), slope=float("nan"), corr=float("nan"),
                    rmse_dex=float("nan"), median_err_dex=float("nan"), outside_prior=float("nan"))
    err = est - tru
    if np.ptp(tru) == 0:
        slope = corr = float("nan")                      # a degenerate set of truths: nothing to regress on
    elif np.ptp(est) == 0:
        slope = corr = 0.0
    else:
        slope, corr = float(np.polyfit(tru, est, 1)[0]), float(np.corrcoef(est, tru)[0, 1])
    return dict(n=n, mae_dex=float(np.mean(np.abs(err))), bias_dex=float(np.mean(err)), slope=slope, corr=corr,
                rmse_dex=float(np.sqrt(np.mean(err ** 2))), median_err_dex=float(np.median(err)),
                outside_prior=float(np.mean((est < lo) | (est > hi))))


def coverage_stats(q, tru, prior_width):
    """``q``: (n, 5) quantiles at 0.05 / 0.25 / 0.50 / 0.75 / 0.95; coverage of the central 50 % and 90 %
    intervals, their gaps to the nominal level, and the widths (dex, and as a share of the prior width)."""
    if q.shape[0] == 0:
        nan = float("nan")
        return dict(n=0, cover50=nan, cover90=nan, gap50=nan, gap90=nan, width50_dex=nan, width90_dex=nan,
                    width90_over_prior=nan)
    c50 = float(np.mean((tru >= q[:, 1]) & (tru <= q[:, 3])))
    c90 = float(np.mean((tru >= q[:, 0]) & (tru <= q[:, 4])))
    w90 = float(np.median(q[:, 4] - q[:, 0]))
    return dict(n=int(q.shape[0]), cover50=c50, cover90=c90, gap50=abs(c50 - 0.5), gap90=abs(c90 - 0.9),
                width50_dex=float(np.median(q[:, 3] - q[:, 1])), width90_dex=w90,
                width90_over_prior=w90 / prior_width)


def paired_difference(a, b, rng, n_boot):
    """Mean of ``a - b`` over the same recordings with a percentile bootstrap interval, and the share of
    recordings on which ``a`` exceeds ``b``."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    if n == 0:
        return dict(n=0, mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"), share_higher=float("nan"))
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = d[rng.integers(0, n, n)].mean()
    return dict(n=int(n), mean=float(d.mean()), ci_low=float(np.quantile(means, 0.025)),
                ci_high=float(np.quantile(means, 0.975)), share_higher=float(np.mean(d > 0)))


def flag(candidate, reference, thresholds):
    """The flags of one column against a reference column, per parameter, from the median view over every
    recording: the category of the MAE change (a difference under the threshold is "under threshold", a
    statement about MAE alone), whether better points come with worse uncertainty, whether the parameter is
    below the correlation floor, whether its correlation dropped by more than the threshold against the
    reference (a large correlation drop), and whether that drop leaves it below the floor (a collapse)."""
    t = thresholds
    out = {}
    for key in candidate["recovery"]:
        r, ref = candidate["recovery"][key]["median"]["all"], reference["recovery"][key]["median"]["all"]
        c, cref = candidate["coverage"][key]["all"], reference["coverage"][key]["all"]
        d_mae = ref["mae_dex"] - r["mae_dex"]                # positive = the candidate is better
        d_gap = c["gap90"] - cref["gap90"]                   # positive = the candidate's coverage is further off
        if abs(d_mae) < t["min_effect"]:
            category = "under threshold"
        elif d_mae > 0:
            category = "improved" if d_gap <= t["coverage_tolerance"] else "points better, uncertainty worse"
        else:
            category = "worsened"
        out[key] = dict(mae_change_dex=float(d_mae), coverage90_gap_change=float(d_gap), category=category,
                        corr=r["corr"], corr_reference=ref["corr"],
                        below_floor=bool(r["corr"] < t["collapse_corr"]),
                        large_drop=bool(ref["corr"] - r["corr"] > t["collapse_drop"]))
        out[key]["collapse"] = out[key]["below_floor"] and out[key]["large_drop"]
    return out


# --- compilation ------------------------------------------------------------------------------------

def compile_scorecard(columns, keys, lo, hi, thresholds, rng, n_boot=2000, seed=None):
    """Every number of the scorecard, as nested dictionaries keyed by aspect, parameter, column name,
    view and regime. ``columns`` are loaded and aligned. ``n_boot`` and ``seed`` are recorded with the
    result: the bootstrap interval measures the uncertainty of a paired mean over the evaluated recordings,
    not the variation between training runs."""
    truth = align(columns)
    masks, definitions = regime_masks(truth, keys, lo, hi)
    names = [c["name"] for c in columns]
    base = next(c for c in columns if c["role"] == "base")
    control = next((c for c in columns if c["role"] == "control"), None)
    for col in columns:
        col["recovery"], col["coverage"] = {}, {}
        for d, key in enumerate(keys):
            col["recovery"][key] = {v: {m: recovery_stats(col["estimates"][v][mk, d], truth[mk, d], lo[d], hi[d])
                                        for m, mk in masks.items()} for v in VIEWS}
            col["coverage"][key] = {m: coverage_stats(col["quantiles"][mk, d, :], truth[mk, d], hi[d] - lo[d])
                                    for m, mk in masks.items()}
        col["failed"] = {f"non_finite_{v}_rows": int((~np.isfinite(col["estimates"][v])).any(axis=1).sum()) for v in VIEWS}
        col["failed"]["non_finite_scores"] = int((~np.isfinite(col["scores"])).sum())
        col["failed"]["MAP_outside_prior_any_parameter"] = float(np.mean(np.any(
            (col["estimates"]["MAP"] < lo) | (col["estimates"]["MAP"] > hi), axis=1)))
        col["joint"] = {}
        if col["truth_log_density"] is not None:
            for m, mk in masks.items():
                v = col["truth_log_density"][mk]
                col["joint"][m] = dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)))
    joint_paired = {}
    for col in columns:
        joint_paired[col["name"]] = {}
        for ref in (base, control):
            if ref is None or ref is col or col["truth_log_density"] is None or ref["truth_log_density"] is None:
                continue
            joint_paired[col["name"]][ref["name"]] = {
                m: paired_difference(col["truth_log_density"][mk], ref["truth_log_density"][mk], rng, n_boot)
                for m, mk in masks.items()}
    flags = {}
    for col in columns:
        flags[col["name"]] = {}
        for ref in (base, control):
            if ref is not None and ref is not col:
                flags[col["name"]][ref["name"]] = flag(col, ref, thresholds)
    summary = {}
    for col in columns:
        s = {"below_floor_parameters": [k for k in keys if col["recovery"][k]["median"]["all"]["corr"] < thresholds["collapse_corr"]]}
        if base is not col:
            f = flags[col["name"]][base["name"]]
            s["versus_base"] = {"improved": [k for k in keys if f[k]["category"] == "improved"],
                                "worsened": [k for k in keys if f[k]["category"] == "worsened"],
                                "under_threshold": [k for k in keys if f[k]["category"] == "under threshold"],
                                "points_better_uncertainty_worse": [k for k in keys if f[k]["category"] == "points better, uncertainty worse"],
                                "large_drop": [k for k in keys if f[k]["large_drop"]],
                                "collapse": [k for k in keys if f[k]["collapse"]]}
        summary[col["name"]] = s
    result = dict(
        columns={c["name"]: describe_column(c) for c in columns}, order=names, base=base["name"],
        control=(control["name"] if control else None), thresholds=dict(thresholds), n_recordings=int(truth.shape[0]),
        bootstrap=dict(resamples=int(n_boot), seed=(None if seed is None else int(seed)),
                       measures="uncertainty of the paired mean over the evaluated recordings; not variation between training runs"),
        regimes={m: dict(definition=definitions[m], n=int(mk.sum())) for m, mk in masks.items()},
        recovery={k: {c["name"]: c["recovery"][k] for c in columns} for k in keys},
        coverage={k: {c["name"]: c["coverage"][k] for c in columns} for k in keys},
        joint={c["name"]: c["joint"] for c in columns}, joint_paired=joint_paired,
        lifted={c["name"]: c["lifted"] for c in columns},
        probe={c["name"]: probe_row(c["probe_result"], keys) for c in columns},
        failed={c["name"]: c["failed"] for c in columns}, flags=flags, summary=summary)
    arrays = dict(parameter_keys=np.array(keys), task_index=base["ids"][:, 0], sim_index=base["ids"][:, 1],
                  true_log10=truth, quantile_levels=np.array(schema.CANONICAL_QUANTILE_LEVELS),
                  prior_low=lo, prior_high=hi, **{f"mask_{m}": mk for m, mk in masks.items()})
    for c in columns:
        for v in VIEWS:
            arrays[f"{c['name']}_{v}"] = c["estimates"][v]
        arrays[f"{c['name']}_quantiles"] = c["quantiles"]
        if c["truth_log_density"] is not None:
            arrays[f"{c['name']}_truth_log_density"] = c["truth_log_density"]
    return result, arrays


def describe_column(col):
    em, m = col["evaluation_manifest"], col["estimator_manifest"]
    d = dict(role=col["role"], tag=col["tag"], evaluation=str(col["evaluation"]),
             evaluation_job=(em.get("execution") or {}).get("job_id"), evaluation_written=em.get("written_at"),
             n_observations=int(em.get("n_observations", 0)), checkpoint_sha256=em.get("checkpoint_sha256"),
             pool_mode=em.get("pool_mode"), n_summary_draws=em.get("n_summary_draws"),
             calibration=(str(col["calibration"]) if col["truth_log_density"] is not None else None),
             calibration_report_lifted=bool(col["lifted"]), lifted_status=col["lifted_status"],
             identity=col["identity"], probe=(str(col["probe"]) if col["probe_result"] else None),
             estimator=(str(col["estimator"]) if m else None))
    if m:
        spec = m.get("rebuild_spec", {})
        emb, maf = spec.get("embedding_args", {}), spec.get("maf_args", {})
        d["encoder"] = {k: emb.get(k, default) for k, default in (("start_channels", 8), ("first_spatial_kernel", 3),
                                                                  ("extra_spatial_convs", 0),
                                                                  ("extra_spatial_conv_blocks", 2),
                                                                  ("spatial_pooling", "mean"))}
        d["flow"] = {k: maf.get(k, default) for k, default in (("hidden_features", 50), ("num_transforms", 5),
                                                               ("num_blocks", 2), ("dropout_probability", None))}
        d["weights_sha256"] = m.get("weights_sha256")
        d["estimator_metadata"] = m.get("metadata", {})
    return d


def probe_row(probe_json, keys):
    if not probe_json:
        return None
    res = probe_json.get("result", {})
    row = {}
    for key in keys:
        r = res.get(key)
        if r:
            row[key] = dict(held_out_mae=r["held_out"]["mae"], null_mae=r["null_mae"], slope=r["held_out"]["slope"],
                            corr=r["held_out"]["corr"], n=r["held_out"]["n"])
    meta = probe_json.get("meta", {})
    return dict(parameters=row, fit_tasks=meta.get("fit_tasks"), held_out_tasks=meta.get("held_out_tasks"),
                weights_sha256=meta.get("weights_sha256"))


# --- outputs ----------------------------------------------------------------------------------------

def _cell_recovery(s):
    return f"{s['mae_dex']:.3f} · {s['bias_dex']:+.3f} · {s['slope']:.2f} · {s['corr']:.2f}"


def write_tables(result, keys, path):
    names, base, control = result["order"], result["base"], result["control"]
    head = "| parameter | " + " | ".join(names) + " |"
    rule = "|---|" + "---|" * len(names)
    L = ["# Estimator scorecard (machine-written)", "",
         f"Columns: {', '.join(f'{n} ({result['columns'][n]['role']})' for n in names)}. "
         f"Base = {base}; control = {control or 'none'}. {result['n_recordings']:,} EVAL recordings, aligned by "
         f"(task, simulation). Log10 units throughout. Cells of the recovery tables read MAE · bias · slope · "
         f"correlation. The scorecard scores each trained posterior estimator as a whole; it isolates nothing.", ""]
    L += ["## Columns", "", "| column | role | tag | encoder | flow | Evaluation job | calibration | probe | identity |", "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        c = result["columns"][n]
        enc = c.get("encoder")
        enc_s = (f"widths from {enc['start_channels']}, first kernel {enc['first_spatial_kernel']}, extra convs "
                 f"{enc['extra_spatial_convs']} × {enc['extra_spatial_conv_blocks']} blocks, pooling {enc['spatial_pooling']}"
                 if enc else "estimator artifact not read")
        flow = c.get("flow")
        flow_s = f"hidden {flow['hidden_features']}, {flow['num_transforms']} transforms, {flow['num_blocks']} blocks" if flow else "-"
        L.append(f"| {n} | {c['role']} | {c['tag'] or '-'} | {enc_s} | {flow_s} | {c['evaluation_job'] or '-'} | "
                 f"{'yes' if c['calibration'] else 'not run'} | {'yes' if c['probe'] else 'not run'} | {c['identity']['status']} |")
    for view in VIEWS:
        L += ["", f"## Recovery, {view} view, every recording (MAE · bias · slope · correlation)", "", head, rule]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(_cell_recovery(result["recovery"][k][n][view]["all"]) for n in names) + " |")
    L += ["", "## Marginal coverage, every recording (empirical coverage of the central 50 % / 90 % intervals)", "", head, rule]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{100 * result['coverage'][k][n]['all']['cover50']:.0f} / "
                                          f"{100 * result['coverage'][k][n]['all']['cover90']:.0f} %" for n in names) + " |")
    L += ["", "## Posterior widths, every recording (median 50 % / 90 % interval width, dex; 90 % width as share of the prior)", "", head, rule]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{result['coverage'][k][n]['all']['width50_dex']:.3f} / "
                                          f"{result['coverage'][k][n]['all']['width90_dex']:.3f} "
                                          f"({100 * result['coverage'][k][n]['all']['width90_over_prior']:.0f} %)" for n in names) + " |")
    L += ["", "## Joint tests, lifted from the Posterior_Calibration reports", "", "| test | " + " | ".join(names) + " |", rule]
    for q in LIFTED_QUANTITIES:
        L.append(f"| {q} | " + " | ".join(((result["lifted"][n] or {}).get("quantitative", {}).get(q, "not run")) for n in names) + " |")
    L += ["", "## SBC rank uniformity (KS D per parameter, lifted) and location-versus-width diagnosis (bias z, spread z)", "", head, rule]
    for k in keys:
        cells = []
        for n in names:
            lf = result["lifted"][n] or {}
            ks = lf.get("sbc_ks_d", {}).get(k)
            dg = lf.get("diagnosis", {}).get(k)
            cells.append(("not run" if ks is None else f"{ks:.3f}") + (f"; {dg['bias_z']:+.2f}, {dg['spread_z']:.2f}" if dg else ""))
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    b = result["bootstrap"]
    L += ["", "## Joint posterior quality: truth log-density on the EVAL recordings", "",
          f"Paired differences with a percentile bootstrap interval ({b['resamples']:,} resamples, seed {b['seed']}): "
          f"the interval measures the uncertainty of the paired mean over the evaluated recordings, not the variation "
          f"between training runs.", "",
          "| quantity | " + " | ".join(names) + " |", rule]
    for lab, fn in (("mean", lambda j: f"{j['mean']:.3f}"), ("median", lambda j: f"{j['median']:.3f}")):
        L.append(f"| {lab}, every recording | " + " | ".join(fn(result["joint"][n]["all"]) if result["joint"][n] else "not run" for n in names) + " |")
    for ref in (base, control):
        if ref is None:
            continue
        cells = []
        for n in names:
            p = result["joint_paired"][n].get(ref)
            cells.append("-" if n == ref else ("not run" if not p else
                         f"{p['all']['mean']:+.3f} [{p['all']['ci_low']:+.3f}, {p['all']['ci_high']:+.3f}], "
                         f"higher on {100 * p['all']['share_higher']:.0f} %"))
        L.append(f"| paired difference against {ref} (mean, 95 % bootstrap interval, share of recordings higher) | " + " | ".join(cells) + " |")
    for m, info in result["regimes"].items():
        if m == "all":
            continue
        L += ["", f"## Regime: {m} ({info['definition']}; {info['n']:,} recordings). Median view: MAE · slope · correlation; coverage 90 %", "", head, rule]
        for k in keys:
            cells = []
            for n in names:
                r, c = result["recovery"][k][n]["median"][m], result["coverage"][k][n][m]
                cells.append(f"{r['mae_dex']:.3f} · {r['slope']:.2f} · {r['corr']:.2f}; {100 * c['cover90']:.0f} %")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "## Encoder row (frozen-encoder linear probe; selects nothing): held-out MAE (null MAE) · slope · correlation", "", head, rule]
    for k in keys:
        cells = []
        for n in names:
            p = result["probe"][n]
            r = (p or {}).get("parameters", {}).get(k)
            cells.append("not run" if not r else f"{r['held_out_mae']:.3f} ({r['null_mae']:.3f}) · {r['slope']:.2f} · {r['corr']:.2f}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    t = result["thresholds"]
    L += ["", "## Flags (median view, every recording; marks, not decisions)", "",
          f"Thresholds: below the floor when the correlation of the posterior median with the truth is under "
          f"{t['collapse_corr']} (unrecovered); large correlation drop when that correlation is more than "
          f"{t['collapse_drop']} below the reference's; collapse when the drop leaves it below the floor; better points "
          f"with worse uncertainty when MAE improves by at least {t['min_effect']} dex while the 90 % coverage gap "
          f"worsens by more than {t['coverage_tolerance']}; under threshold when the MAE difference is under "
          f"{t['min_effect']} dex, a statement about MAE alone that says nothing about calibration, width or joint "
          f"density.", ""]
    for n in names:
        for ref, f in result["flags"][n].items():
            L += [f"### {n} against {ref}", "", "| parameter | MAE change (dex, + = better) | 90 % coverage gap change (+ = worse) | category | correlation (reference) | floor / drop / collapse |", "|---|---|---|---|---|---|"]
            for k in keys:
                x = f[k]
                mark = ("COLLAPSE (large drop, below floor)" if x["collapse"] else
                        "; ".join(w for w, on in (("below floor", x["below_floor"]), ("large correlation drop", x["large_drop"])) if on))
                L.append(f"| {k} | {x['mae_change_dex']:+.4f} | {x['coverage90_gap_change']:+.3f} | {x['category']} | "
                         f"{x['corr']:.2f} ({x['corr_reference']:.2f}) | {mark or 'no'} |")
            L.append("")
    L += ["## Failed estimates", "", "| quantity | " + " | ".join(names) + " |", rule]
    for q in ("non_finite_MAP_rows", "non_finite_median_rows", "non_finite_SGM_rows", "non_finite_scores"):
        L.append(f"| {q} | " + " | ".join(str(result["failed"][n][q]) for n in names) + " |")
    L.append("| MAP outside the prior on any parameter | " + " | ".join(f"{100 * result['failed'][n]['MAP_outside_prior_any_parameter']:.1f} %" for n in names) + " |")
    L += ["", "## Summary", ""]
    for n in names:
        s = result["summary"][n]
        line = f"- **{n}**: parameters below the correlation floor: {', '.join(s['below_floor_parameters']) or 'none'}"
        if "versus_base" in s:
            vb = s["versus_base"]
            line += (f"; against the base, MAE improved: {', '.join(vb['improved']) or 'none'}; MAE worsened: "
                     f"{', '.join(vb['worsened']) or 'none'}; MAE difference under the threshold: "
                     f"{', '.join(vb['under_threshold']) or 'none'}; points better with uncertainty worse: "
                     f"{', '.join(vb['points_better_uncertainty_worse']) or 'none'}; large correlation drop: "
                     f"{', '.join(vb['large_drop']) or 'none'}; collapse: {', '.join(vb['collapse']) or 'none'}")
        L.append(line + ".")
    L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def write_figures(result, arrays, keys, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = result["order"]
    x = np.arange(len(keys))
    w = 0.8 / len(names)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric, title) in zip(axes.ravel(), (("mae_dex", "MAE (dex), median view"), ("corr", "correlation with the truth, median view"))):
        for i, n in enumerate(names):
            vals = [result["recovery"][k][n]["median"]["all"][metric] for k in keys]
            ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, label=n)
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20); ax.set_title(f"{title}, every recording", fontsize=10)
    ax = axes[1, 0]
    for i, n in enumerate(names):
        vals = [result["coverage"][k][n]["all"]["cover90"] for k in keys]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, label=n)
    ax.axhline(0.9, color="k", lw=0.8, ls=":"); ax.set_ylim(0, 1)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20); ax.set_title("empirical coverage of the 90 % interval (dotted = nominal)", fontsize=10)
    ax = axes[1, 1]
    for i, n in enumerate(names):
        vals = [result["recovery"][k][n]["median"]["operating_subgroup"]["mae_dex"] for k in keys]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w, label=n)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20)
    ax.set_title(f"MAE (dex), median view, operating subgroup ({result['regimes']['operating_subgroup']['n']:,} recordings)", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Estimator scorecard: recovery and marginal coverage", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "recovery_and_coverage.png", dpi=130)
    plt.close(fig)
    paired = [(n, result["joint_paired"][n].get(result["base"])) for n in names if n != result["base"]]
    paired = [(n, p) for n, p in paired if p]
    if paired:
        fig, axes = plt.subplots(1, len(paired), figsize=(4.6 * len(paired), 3.8), squeeze=False)
        base_ld = arrays.get(f"{result['base']}_truth_log_density")
        for ax, (n, p) in zip(axes.ravel(), paired):
            d = arrays[f"{n}_truth_log_density"] - base_ld
            lim = max(float(np.quantile(np.abs(d), 0.99)), 1e-6)
            ax.hist(np.clip(d, -lim, lim), bins=120, range=(-lim, lim), color="0.4")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{n} minus {result['base']}: mean {p['all']['mean']:+.3f} "
                         f"[{p['all']['ci_low']:+.3f}, {p['all']['ci_high']:+.3f}]", fontsize=9.5)
            ax.set_xlabel("truth log-density difference per recording (clipped at the 99th percentile of |difference|)")
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / "truth_log_density_paired.png", dpi=130)
        plt.close(fig)


def code_identity(repo):
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    head = git("rev-parse", "HEAD")
    return dict(package_version=_PACKAGE_VERSION, git_head=(head or None),
                git_dirty=(bool(git("status", "--porcelain")) if head else None))


def write_provenance(result, R, path):
    ident = code_identity(Path(__file__).resolve().parents[2])
    L = ["# Provenance", "",
         f"Record: `{R['out_dir'].name}`. Compiled by `Script_Bank/Analysis/{Path(__file__).name}` "
         f"(package {ident['package_version']}, git {ident['git_head'] or 'not a checkout'}"
         f"{', dirty' if ident['git_dirty'] else ''}). Protocol: DETECTOR_WORKFLOW.md, the encoder screening "
         f"(scorecard and selection rule).", "", "## Inputs (read-only)", "",
         "| column | role | Evaluation product | job | written | n | checkpoint | calibration | probe | identity |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in result["order"]:
        c = result["columns"][n]
        L.append(f"| {n} | {c['role']} | `{Path(c['evaluation']).parent.name}` | {c['evaluation_job'] or '-'} | "
                 f"{c['evaluation_written'] or '-'} | {c['n_observations']:,} | `{(c['checkpoint_sha256'] or '-')[:12]}` | "
                 f"{'`' + Path(c['calibration']).parent.name + '`' if c['calibration'] else 'not run'} | "
                 f"{'`' + Path(c['probe']).parent.name + '`' if c['probe'] else 'not run'} | {c['identity']['status']} |")
    L += ["", "Identity: within each column the Evaluation's checkpoint, the estimator artifact's weights and the "
          "probe's weights (those present) must be one hash, or the compilation is refused; the calibration product "
          "carries no checkpoint provenance, so its identity is unverified. Calibration report: " +
          "; ".join(f"{n}: {result['columns'][n]['lifted_status']}" for n in result["order"]) + "."]
    L += ["", "## Estimator artifacts", ""]
    for n in result["order"]:
        c = result["columns"][n]
        if c.get("encoder"):
            L.append(f"- **{n}**: encoder {json.dumps(c['encoder'])}; flow {json.dumps(c['flow'])}; weights "
                     f"`{(c['weights_sha256'] or '-')[:12]}`; metadata {json.dumps(c['estimator_metadata'], default=str)}")
        else:
            L.append(f"- **{n}**: estimator artifact not present beside the records; architecture not read.")
    L += ["", "## Alignment", "",
          f"Every column is reordered to the base's (task_index, sim_index) order and the truths must agree exactly; "
          f"each calibration cloud is aligned to the same recordings by the six true parameters rounded to nine "
          f"decimals, unique, and must hold exactly the Evaluation's set. {result['n_recordings']:,} recordings.",
          "", "## Regimes", ""]
    for m, info in result["regimes"].items():
        L.append(f"- `{m}`: {info['definition']} ({info['n']:,} recordings)")
    L += ["", "## Computed and lifted", "",
          f"Computed from the products: recovery of the three point estimates, marginal coverage and widths from the "
          f"stored quantiles, the truth log-density statistics and their paired differences (percentile bootstrap, "
          f"{result['bootstrap']['resamples']:,} resamples, seed {result['bootstrap']['seed']}; the interval measures "
          f"the uncertainty of the paired mean over the evaluated recordings, not the variation between training "
          f"runs), the regimes, the flags. Lifted verbatim from each Posterior_Calibration report: the "
          "joint tests (expected coverage, TARP, L-C2ST, the marginal worst cases), the SBC statistic and the "
          "location-versus-width diagnosis. The encoder row copies the held-out numbers of each embedding-probe "
          "record and selects nothing.", "", "## Thresholds", "", "`" + json.dumps(result["thresholds"]) + "`", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


def write_outputs(result, arrays, R, keys):
    out = R["out_dir"]
    (out / "figures").mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "scorecard.npz", **arrays)
    (out / "scorecard.json").write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    write_tables(result, keys, out / "scorecard.md")
    write_figures(result, arrays, keys, out)
    write_provenance(result, R, out / "PROVENANCE.md")
    (out / "README.md").write_text(
        f"# {out.name}\n\nThe scorecard of the detector's neural posterior estimators named in `scorecard.md` "
        f"(columns), compiled from their Evaluation, Posterior_Calibration and embedding-probe records on the same "
        f"EVAL recordings. `scorecard.json` holds every number, `scorecard.npz` the aligned arrays, `figures/` the "
        f"overview, `PROVENANCE.md` the inputs, alignment, regimes and thresholds. The reading and the selection "
        f"belong to DETECTOR_WORKFLOW.md, the encoder screening.\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


# --- entry point ------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True, help="model window of the estimators (s).")
    ap.add_argument("--condition", default="FAB", choices=LABELING_CONDITIONS)
    ap.add_argument("--base", required=True, help="the reference column (an artifact tag, or 'baseline').")
    ap.add_argument("--control", default=None, help="the secondary reference column (tag or 'baseline'); optional.")
    ap.add_argument("--candidates", nargs="*", default=[], help="candidate columns (artifact tags); may be empty.")
    for key, value in DEFAULT_THRESHOLDS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=float, default=value, help=f"flag threshold (default {value}).")
    ap.add_argument("--bootstrap", type=int, default=2000, help="resamples of the paired log-density difference.")
    ap.add_argument("--seed", type=int, default=0, help="seed of the bootstrap.")
    ap.add_argument("--posit", default=None, help="read the records from this folder instead of the machine's Posit tier.")
    ap.add_argument("--out-dir", default=None, help="write here instead of the Posit tier's scorecard folder "
                                                    "(named after the columns).")
    ap.add_argument("--dry-run", action="store_true", help="resolve the inputs and outputs; compile nothing.")
    args = ap.parse_args(argv)
    thresholds = {key: float(getattr(args, key)) for key in DEFAULT_THRESHOLDS}

    cfg = detector_workflow()
    R = resolve(cfg, args)
    assert all(is_log_row(e) for e in R["table"] if e["KEY"] in R["keys"]), "imaging rows are expected in log10"
    print(f"=== Estimator scorecard | {R['paths'].project_alias}_{R['timing'].label} | package {_PACKAGE_VERSION}")
    present, missing = [], []
    for c in R["columns"]:
        ok = c["evaluation"].exists()
        (present if ok else missing).append(c)
        print(f"  {c['role']:9s} {c['name']:24s} Evaluation {'present' if ok else 'MISSING'}"
              f" | calibration {'present' if c['calibration'].exists() else 'absent'}"
              f" | probe {'present' if c['probe'].exists() else 'absent'}"
              f" | estimator {'present' if c['estimator'].exists() else 'absent'}")
    print(f"  thresholds: {thresholds}")
    print(f"  writes: {R['out_dir']}/")
    if any(c["role"] == "base" for c in missing):
        raise SystemExit("the base column has no Evaluation product; nothing to compare against")
    if any(c["role"] == "control" for c in missing):
        raise SystemExit("the control column has no Evaluation product; drop --control or supply it")
    if R["out_dir"].exists():
        raise SystemExit(f"refusing to overwrite an existing scorecard: {R['out_dir']}")
    if args.dry_run:
        print("dry run: nothing compiled.")
        return 0
    columns = [load_column(c, R["keys"]) for c in present]
    rng = np.random.default_rng(args.seed)
    result, arrays = compile_scorecard(columns, R["keys"], R["lo"], R["hi"], thresholds, rng, n_boot=args.bootstrap,
                                       seed=args.seed)
    result["missing_candidates"] = [c["name"] for c in missing]
    write_outputs(result, arrays, R, R["keys"])
    for n in result["order"]:
        print(f"  {n}: {json.dumps(result['summary'][n])}")
    if missing:
        print(f"  candidates without an Evaluation product, not compiled: {[c['name'] for c in missing]}")
    print(f"written: {R['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
