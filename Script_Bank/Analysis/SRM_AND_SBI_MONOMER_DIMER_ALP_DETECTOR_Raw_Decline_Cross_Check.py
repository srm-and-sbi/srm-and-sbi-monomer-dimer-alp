#!/usr/bin/env python
"""Per-recording cross-check of a detector estimator's Experiment estimates against the raw-domain
fluorescence-decline record of the same recordings.

Companion note ``SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Raw_Decline_Cross_Check.md``. One detector estimator,
named by its artifact tag (``baseline`` for the untagged one), read from its Experiment product (the posterior
median, MAP and SGM of every window of the experimental recordings); the raw-domain record of the direct
fluorescence-loss companion read from its arrays. The two are aligned recording by recording and, for the
local rates, window by window at the same frame starts, and compared as DIFFERENT quantities placed side by
side:

- the estimator's ``prob_photo_bleach``: the per-dye probability of entering the absorbing dark state within
  the 100-frame reference interval, as the renderer defines it, inferred from each 2 s window;
- the measured whole-recording decline of the field statistic (the closing-to-opening ratio and the single
  per-interval rate it implies), the late-phase rate, the free-offset fitted rate, and the local rate inside
  each 2 s window, all from the median-subtracted total fluorescence of the field, which is a proxy for the
  per-dye process and not a count of dyes, probes or spots;
- the estimator's ``mu_pc`` (the median photon contribution of one active dye per frame) against the opening
  field excess of the recording (the opening window's mean pixel level minus its median, which includes a
  camera-noise contribution), a diagnostic relation between different quantities, never an equality.

Counts, medians and correlations use every recording or window where both values are finite; a logarithmic
shift is defined only where both are positive and is reported over that subset with its own count. A raw rate
of zero or below (a field statistic that did not fall) is counted apart: any positive estimate exceeds it by
construction.

The analytical dye retention a per-dye value implies under the renderer's independent bleaching and
stationary mean brightness, ``(1 - p) ** (d / 100)`` over the ``d`` frames between the opening and closing
windows, is computed from the estimator's per-recording value and placed beside the experimental field
retention, with the same expectation for the comparison values the raw record carries. It is an expectation
of dye fluorescence, not of the median-subtracted field statistic or of spot retention, and not a render, so
the two are placed side by side and not tested against each other.

Nothing here assumes that either the estimator's values or any selected simulation vector is correct; the
record is diagnostic. Inputs are read-only; the output folder is never overwritten. Usage::

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Raw_Decline_Cross_Check.py \\
        --total-time-seconds 2 --estimator CAP256KERNEL7EARLYCONVSTATSGB128 [--raw-span-seconds 20] \\
        [--raw-record <npz>] [--authorization "<who, when>"] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import __version__ as _PACKAGE_VERSION
from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from srm_and_sbi_monomer_dimer_alp.labeling import LABELING_CONDITIONS
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming, entry_to_physical, is_log_row
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow, parameter_keys, parameter_table

VIEWS = ("median", "MAP", "SGM")
VIEW_NAMES = {"median": "posterior median", "MAP": "MAP", "SGM": "SGM"}
BASELINE = "baseline"
RAW_RECORD_SUFFIX = "Direct_Fluorescence_Loss_Experiment_RAW"
RAW_ARRAYS = "direct_fluorescence_loss_experiment.npz"
NUMB_PHOTO_BLEACH = 100


# --- inputs -----------------------------------------------------------------------------------------

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(cfg, args):
    timing = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing)
    paths = cfg.paths.with_condition(args.condition)
    posit = Path(args.posit) if args.posit else PARAMETERS.machine.data_bank_root / paths.posit_subdir
    tag = None if args.estimator == BASELINE else args.estimator
    label = paths.product_label(timing.label, tag)
    stem = f"{paths.project_alias}_{label}"
    raw_timing = RunTiming(total_time_seconds=args.raw_span_seconds, frames=PARAMETERS.simulation.timing)
    raw_stem = f"{paths.project_alias}_{paths.product_label(raw_timing.label, None)}_{RAW_RECORD_SUFFIX}"
    raw = Path(args.raw_record) if args.raw_record else posit / raw_stem / RAW_ARRAYS
    plain = paths.product_label(timing.label, None)
    out_dir = Path(args.out_dir) if args.out_dir else \
        posit / f"{paths.project_alias}_{plain}_Raw_Decline_Cross_Check_{args.estimator.upper()}"
    return dict(timing=timing, paths=paths, posit=posit, estimator=args.estimator, tag=tag,
                experiment=posit / f"{stem}_MAP_Experiment" / f"{stem}_MAP_Experiment.npz", raw=raw,
                out_dir=out_dir, condition=args.condition, keys=parameter_keys(cfg, args.condition), table=parameter_table(cfg, args.condition),
                authorization=args.authorization)


def load_experiment(path, keys):
    """The Experiment product through the schema; the parameter keys must be the workflow's, the windows
    unique, and the window geometry present (it places every window at a frame start)."""
    src = str(path)
    arrays, manifest = schema.load_product(path, stage="experiment")
    schema.assert_parameter_keys(manifest, keys, source=src)
    ids = schema.observation_ids(arrays, "experiment")
    schema.assert_unique_observations(ids, source=src)
    geometry = manifest.get("window_geometry") or {}
    if not all(k in geometry for k in ("n_frames", "step_frames")):
        raise SystemExit(f"{src}: the manifest carries no window geometry; the windows cannot be placed in time")
    q = np.asarray(arrays["posterior_quantiles"], dtype=float)
    return dict(path=src, sha256=sha256_of(path), manifest=manifest, ids=ids, geometry=geometry,
                kinds=[str(k) for k in np.asarray(arrays["kinds"]).ravel()],
                estimates={"median": q[:, :, schema.median_level_index(manifest)],
                           "MAP": np.asarray(arrays["map_estimate"], dtype=float),
                           "SGM": np.asarray(arrays["posterior_sgm"], dtype=float)})


def load_raw(path, condition, window_seconds):
    """The raw-domain record's arrays and summary. The record must be the raw domain of the same condition
    and must carry the local rates at the estimator's window duration."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"raw-domain record not found: {path}")
    z = np.load(path, allow_pickle=False)
    summary_path = path.parent / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    domain = str(np.asarray(z["domain"]).ravel()[0]) if "domain" in z.files else summary.get("domain")
    if domain != "raw":
        raise SystemExit(f"{path}: domain {domain!r} is not the raw domain")
    kinds = [str(k) for k in np.asarray(z["kinds"]).ravel()] if "kinds" in z.files else [summary.get("condition")]
    if kinds != [condition]:
        raise SystemExit(f"{path}: condition {kinds} is not {[condition]}")
    key = f"window_rate_{window_seconds:g}s"
    if key not in z.files or f"window_starts_{window_seconds:g}s" not in z.files:
        raise SystemExit(f"{path}: no local rates at {window_seconds:g} s windows ({key})")
    needed = ("cell", "estimate", "usable", "decline_p_whole", "decline_p_late", "decline_ratio_whole",
              "decline_ratio_late", "center_separation_frames", "opening_median", "closing_median", "opening_excess")
    missing = [k for k in needed if k not in z.files]
    if missing:
        raise SystemExit(f"{path}: arrays missing: {missing}")
    cells = np.asarray(z["cell"]).astype(np.int64)
    if np.unique(cells).size != cells.size:
        raise SystemExit(f"{path}: a recording appears more than once")
    return dict(path=str(path), sha256=sha256_of(path), summary=summary, cells=cells,
                estimate=np.asarray(z["estimate"], float), usable=np.asarray(z["usable"], bool),
                p_whole=np.asarray(z["decline_p_whole"], float), p_late=np.asarray(z["decline_p_late"], float),
                ratio_whole=np.asarray(z["decline_ratio_whole"], float), ratio_late=np.asarray(z["decline_ratio_late"], float),
                separation=np.asarray(z["center_separation_frames"], float),
                opening_background=np.asarray(z["opening_median"], float),
                closing_background=np.asarray(z["closing_median"], float),
                opening_excess=np.asarray(z["opening_excess"], float),
                window_rate=np.asarray(z[key], float), window_starts=np.asarray(z[f"window_starts_{window_seconds:g}s"]).astype(np.int64),
                compare_values=[float(v) for v in (summary.get("compare_values") or [])])


def align(exp, raw, keys, table):
    """Recording by recording and, for the local rates, window by window at the same frame start. The
    recording sets must be equal and every window start of the estimator must be a window start of the raw
    record; otherwise the two were not read from the same recordings in the same way."""
    kind, cells, chunks = exp["ids"][:, 0], exp["ids"][:, 1], exp["ids"][:, 2]
    if np.unique(kind).size != 1:
        raise SystemExit("the Experiment product holds more than one condition")
    exp_cells = np.unique(cells)
    if set(exp_cells.tolist()) != set(raw["cells"].tolist()):
        missing = sorted(set(exp_cells.tolist()) - set(raw["cells"].tolist()))[:5]
        extra = sorted(set(raw["cells"].tolist()) - set(exp_cells.tolist()))[:5]
        raise SystemExit(f"recordings differ: the raw record lacks {missing} and adds {extra}")
    starts = chunks * int(exp["geometry"]["step_frames"])
    raw_pos = {int(s): j for j, s in enumerate(raw["window_starts"])}
    if any(int(s) not in raw_pos for s in np.unique(starts)):
        raise SystemExit(f"window starts {sorted(set(int(s) for s in np.unique(starts)) - set(raw_pos))[:5]} of the "
                         f"estimator are not window starts of the raw record {sorted(raw_pos)}")
    raw_index = {int(c): j for j, c in enumerate(raw["cells"])}
    i_pb, i_pc = keys.index("prob_photo_bleach"), keys.index("mu_pc")
    phys = {k: (lambda x, i=i: float(entry_to_physical(table[i], float(x)))) for k, i in (("pb", i_pb), ("pc", i_pc))}
    n_cells = exp_cells.size
    uniq_chunks = np.unique(chunks)
    rows = {}
    for v in VIEWS:
        est = exp["estimates"][v]
        p_win = np.full((n_cells, uniq_chunks.size), np.nan)
        pc_win = np.full_like(p_win, np.nan)
        for r, c in enumerate(exp_cells):
            for w, ch in enumerate(uniq_chunks):
                m = (cells == c) & (chunks == ch)
                if m.any():
                    p_win[r, w] = phys["pb"](est[m][0, i_pb])
                    pc_win[r, w] = phys["pc"](est[m][0, i_pc])
        rows[v] = dict(p_window=p_win, pc_window=pc_win,
                       p_recording=np.array([phys["pb"](np.median(est[cells == c, i_pb])) for c in exp_cells]),
                       pc_recording=np.array([phys["pc"](np.median(est[cells == c, i_pc])) for c in exp_cells]))
    direct = np.full((n_cells, uniq_chunks.size), np.nan)
    for r, c in enumerate(exp_cells):
        for w, ch in enumerate(uniq_chunks):
            direct[r, w] = raw["window_rate"][raw_index[int(c)], raw_pos[int(ch) * int(exp["geometry"]["step_frames"])]]
    order = np.array([raw_index[int(c)] for c in exp_cells])
    aligned = {k: raw[k][order] for k in ("estimate", "usable", "p_whole", "p_late", "ratio_whole", "ratio_late",
                                           "separation", "opening_background", "closing_background",
                                           "opening_excess")}
    aligned.update(cells=exp_cells, chunks=uniq_chunks, direct_window=direct, views=rows)
    return aligned


# --- statistics -------------------------------------------------------------------------------------

def _finite(*arrays):
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def correlate(a, b):
    """Spearman and Pearson over the finite pairs; ``None`` when fewer than three or a series is constant."""
    from scipy.stats import pearsonr, spearmanr
    m = _finite(a, b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return {"n": int(m.sum()), "spearman_r": None, "pearson_r": None}
    return {"n": int(m.sum()), "spearman_r": float(spearmanr(a[m], b[m]).correlation),
            "pearson_r": float(pearsonr(a[m], b[m])[0])}


def pairing(nn, raw, mask=None):
    """The estimator's value against a raw quantity. Over every pair where both are finite: the count ``n``,
    correlation, medians and the count where the estimator's value is the larger. Over the pairs where both are
    positive (``n_positive``): the median of ``log10(nn / raw)`` (dex) and the count where the estimator is the
    larger there. A raw value of zero or below is counted in ``n_raw_nonpositive``; any positive estimate
    exceeds it by construction."""
    nn, raw = np.asarray(nn, float), np.asarray(raw, float)
    m = _finite(nn, raw)
    if mask is not None:
        m &= np.asarray(mask, bool)
    pos = m & (raw > 0) & (nn > 0)
    out = correlate(nn[m], raw[m])
    out.update({"median_nn": float(np.median(nn[m])) if m.any() else None,
                "median_raw": float(np.median(raw[m])) if m.any() else None,
                "frac_nn_higher": float(np.mean(nn[m] > raw[m])) if m.any() else None,
                "n_nn_higher": int(np.sum(nn[m] > raw[m])),
                "n_raw_nonpositive": int(np.sum(raw[m] <= 0)),
                "n_positive": int(pos.sum()),
                "median_log10_ratio_nn_over_raw": float(np.median(np.log10(nn[pos] / raw[pos]))) if pos.any() else None,
                "n_nn_higher_positive": int(np.sum(nn[pos] > raw[pos]))})
    return out


def retention(p, separation):
    """Expected fluorescence retention over ``separation`` frames at the per-dye value ``p``."""
    return (1.0 - np.asarray(p, float)) ** (np.asarray(separation, float) / NUMB_PHOTO_BLEACH)


def compile_cross_check(al, raw, keys):
    sep = al["separation"]
    result = {"n_recordings": int(al["cells"].size), "n_windows_per_recording": int(al["chunks"].size),
              "n_usable_fits": int(al["usable"].sum()), "separation_frames_median": float(np.median(sep)),
              "compare_values": raw["compare_values"], "per_view": {}}
    for v in VIEWS:
        row = al["views"][v]
        p_rec, pc_rec = row["p_recording"], row["pc_recording"]
        pred = retention(p_rec, sep)
        per_pos = []
        for w in range(al["chunks"].size):
            s = pairing(row["p_window"][:, w], al["direct_window"][:, w])
            per_pos.append({"chunk": int(al["chunks"][w]), "n": s["n"], "median_nn": s["median_nn"],
                            "median_direct": s["median_raw"], "n_nn_higher": s["n_nn_higher"],
                            "frac_nn_higher": s["frac_nn_higher"], "n_direct_nonpositive": s["n_raw_nonpositive"],
                            "n_positive": s["n_positive"],
                            "median_log10_ratio_nn_over_direct": s["median_log10_ratio_nn_over_raw"],
                            "n_nn_higher_positive": s["n_nn_higher_positive"]})
        within = []
        for r in range(al["cells"].size):
            c = correlate(row["p_window"][r], al["direct_window"][r])
            within.append(c["spearman_r"])
        within = np.array([np.nan if x is None else x for x in within], float)
        m_ret = _finite(pred, al["ratio_whole"])
        excess = al["opening_excess"]
        result["per_view"][v] = {
            "rates_per_recording": {
                "free_offset_fit": pairing(p_rec, al["estimate"], mask=al["usable"]),
                "whole_recording": pairing(p_rec, al["p_whole"]),
                "late_phase": pairing(p_rec, al["p_late"])},
            "retention": {
                "analytical_dye_retention_median": float(np.median(pred[m_ret])),
                "field_retention_median": float(np.median(al["ratio_whole"][m_ret])), "n": int(m_ret.sum()),
                **{k: val for k, val in correlate(pred, al["ratio_whole"]).items() if k != "n"},
                "compare_values_analytical": {f"{cv:g}": float(np.median(retention(cv, sep))) for cv in raw["compare_values"]},
                "nn_bleaching_vs_field_retention": correlate(p_rec, al["ratio_whole"])},
            "window_positions": {"all_windows": pairing(row["p_window"].ravel(), al["direct_window"].ravel()),
                                 "per_position": per_pos,
                                 "within_recording_spearman_median": float(np.nanmedian(within)) if np.isfinite(within).any() else None,
                                 "within_recording_spearman_n_positive": int(np.nansum(within > 0)),
                                 "within_recording_n": int(np.isfinite(within).sum())},
            "brightness": {"mu_pc_window0_vs_opening_excess": correlate(row["pc_window"][:, 0], excess),
                           "mu_pc_recording_vs_opening_excess": correlate(pc_rec, excess),
                           "nn_bleaching_vs_opening_excess": correlate(p_rec, excess),
                           "late_rate_vs_opening_excess": correlate(al["p_late"], excess),
                           "whole_rate_vs_opening_excess": correlate(al["p_whole"], excess),
                           "mu_pc_window0_median": float(np.nanmedian(row["pc_window"][:, 0])),
                           "opening_excess_median": float(np.nanmedian(excess))}}
    arrays = {"cell": al["cells"], "chunk": al["chunks"], "direct_window_rate": al["direct_window"],
              "free_offset_fit_rate": al["estimate"], "usable": al["usable"], "whole_rate": al["p_whole"],
              "late_rate": al["p_late"], "field_retention": al["ratio_whole"], "late_retention": al["ratio_late"],
              "separation_frames": sep, "opening_excess": al["opening_excess"],
              "opening_background": al["opening_background"], "closing_background": al["closing_background"]}
    for v in VIEWS:
        row = al["views"][v]
        arrays.update({f"{v}_bleaching_window": row["p_window"], f"{v}_bleaching_recording": row["p_recording"],
                       f"{v}_mu_pc_window": row["pc_window"], f"{v}_mu_pc_recording": row["pc_recording"],
                       f"{v}_analytical_dye_retention": retention(row["p_recording"], sep)})
    return result, arrays


# --- outputs ----------------------------------------------------------------------------------------

def _r(x):
    return "–" if x is None else f"{x:+.2f}"


def _f(x, fmt=".3f"):
    return "–" if x is None else format(x, fmt)


def write_tables(result, R, path):
    v = "median"
    P = result["per_view"][v]
    L = [f"# Raw-domain decline cross-check: {R['estimator']}", "",
         f"{result['n_recordings']} recordings, {result['n_windows_per_recording']} windows each; the estimator's values "
         f"({VIEW_NAMES[v]} view in this file; every view in `cross_check.json`) beside the raw-domain decline quantities "
         f"of the same recordings. Different quantities placed side by side, never an equality: the estimator's "
         f"`prob_photo_bleach` is a per-dye probability over 100 frames inferred from 2 s windows; the raw quantities are "
         f"readings of the median-subtracted total field fluorescence, a proxy for the per-dye process that keeps part "
         f"of the camera noise. No ground truth, no assumption that either the estimator or a selected vector is "
         f"correct.", "",
         "## Per recording: the estimator's bleaching (median over windows) against the raw rates", "",
         "Each raw rate is expressed per 100 frames: the same units as the estimator's parameter, not the same "
         "measurement scale. Correlations, medians and the count where the estimator's value is the larger use every "
         "recording with both values finite (n); the shift, the median of log10(estimator / raw) in dex, uses only the "
         "recordings where both are positive (its own n). A raw rate of zero or below is exceeded by any positive "
         "estimate by construction.", "",
         "| raw quantity | n | Spearman | Pearson | median estimator | median raw | estimator larger | raw rate ≤ 0 | "
         "n (both positive) | shift (dex, both positive) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    names = {"free_offset_fit": "free-offset fitted rate (usable fits only)",
             "whole_recording": "whole-recording rate (opening to closing window)",
             "late_phase": "late-phase rate (mid to closing window)"}
    for k, s in P["rates_per_recording"].items():
        L.append(f"| {names[k]} | {s['n']} | {_r(s['spearman_r'])} | {_r(s['pearson_r'])} | {_f(s['median_nn'])} | "
                 f"{_f(s['median_raw'])} | {s['n_nn_higher']} of {s['n']} | {s['n_raw_nonpositive']} | {s['n_positive']} | "
                 f"{_f(s['median_log10_ratio_nn_over_raw'], '+.3f')} |")
    r = P["retention"]
    L += ["", "## Analytical dye retention beside experimental field retention (median-subtracted field statistic; "
              "not a test)", "",
          f"The analytical dye retention over the {result['separation_frames_median']:.0f} frames between the opening and "
          f"closing windows at the estimator's per-recording value, `(1 − p)^(d/100)`, under independent per-dye "
          f"bleaching and stationary mean brightness (an expectation of dye fluorescence, not of the field statistic or "
          f"of spot retention, and not a render), beside the experimental closing-to-opening ratio of the "
          f"median-subtracted field statistic. The two are different quantities and are not tested against each "
          f"other; a field retention is compared with synthetic field retention measured in the same way.", "",
          "| quantity | value |", "|---|---|",
          f"| analytical dye retention at the estimator's value, median over recordings | "
          f"{r['analytical_dye_retention_median']:.3f} |",
          f"| experimental field retention, median over recordings | {r['field_retention_median']:.3f} |",
          f"| recordings | {r['n']} |",
          f"| Spearman / Pearson, analytical against field retention | {_r(r['spearman_r'])} / {_r(r['pearson_r'])} |",
          f"| Spearman, estimator's bleaching against field retention | "
          f"{_r(r['nn_bleaching_vs_field_retention']['spearman_r'])} |"]
    for cv, val in r["compare_values_analytical"].items():
        L.append(f"| analytical dye retention at the comparison value {cv} | {val:.3f} |")
    W = P["window_positions"]
    a = W["all_windows"]
    L += ["", "## Window by window: the estimator's bleaching against the local direct rate of the same window", "",
          f"Over all {a['n']} windows with both values finite: Spearman {_r(a['spearman_r'])}, Pearson "
          f"{_r(a['pearson_r'])}. The estimator's per-dye bleaching probability exceeds the effective loss probability "
          f"calculated from the field statistic in {a['n_nn_higher']} of {a['n']} matched windows. In {a['n_positive']} "
          f"windows the local rate is positive: median shift {_f(a['median_log10_ratio_nn_over_raw'], '+.3f')} dex, "
          f"estimator larger in {a['n_nn_higher_positive']} of {a['n_positive']}; in the other {a['n_raw_nonpositive']} the "
          f"field statistic did not fall within the window (local rate zero or below), and any positive estimate is "
          f"larger by construction. Within a recording, across its window positions: median Spearman "
          f"{_f(W['within_recording_spearman_median'], '+.2f')} (positive in {W['within_recording_spearman_n_positive']} "
          f"of {W['within_recording_n']} recordings).", "",
          "| window | n | median estimator | median direct | estimator larger | direct ≤ 0 | n (direct positive) | "
          "shift (dex, direct positive) |", "|---|---|---|---|---|---|---|---|"]
    for p in W["per_position"]:
        L.append(f"| {p['chunk']} | {p['n']} | {_f(p['median_nn'])} | {_f(p['median_direct'])} | "
                 f"{p['n_nn_higher']} of {p['n']} | {p['n_direct_nonpositive']} | {p['n_positive']} | "
                 f"{_f(p['median_log10_ratio_nn_over_direct'], '+.3f')} |")
    B = P["brightness"]
    L += ["", "## Brightness beside the decline", "",
          f"The estimator's `mu_pc` (median photon contribution of one active dye per frame; window 0 median "
          f"{B['mu_pc_window0_median']:.1f}) beside the opening field excess of the recording (the opening window's mean "
          f"pixel level minus its median, per pixel, median {B['opening_excess_median']:.1f}; a median-subtracted field "
          f"statistic that includes a camera-noise contribution): different quantities, a diagnostic relation. Beside "
          f"it, how the estimator's bleaching and the raw rates rank with the opening field excess.", "",
          "| pairing | n | Spearman | Pearson |", "|---|---|---|---|"]
    for k, lab in (("mu_pc_window0_vs_opening_excess", "estimator `mu_pc` (window 0) vs opening field excess"),
                   ("mu_pc_recording_vs_opening_excess", "estimator `mu_pc` (recording median) vs opening field excess"),
                   ("nn_bleaching_vs_opening_excess", "estimator bleaching (recording median) vs opening field excess"),
                   ("late_rate_vs_opening_excess", "raw late-phase rate vs opening field excess"),
                   ("whole_rate_vs_opening_excess", "raw whole-recording rate vs opening field excess")):
        c = B[k]
        L.append(f"| {lab} | {c['n']} | {_r(c['spearman_r'])} | {_r(c['pearson_r'])} |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def write_figures(result, arrays, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = Path(out_dir) / "figures"
    v = "median"
    p_rec, p_win = arrays[f"{v}_bleaching_recording"], arrays[f"{v}_bleaching_window"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, name, lab) in zip(axes, (("free_offset_fit_rate", "free_offset_fit", "free-offset fitted rate"),
                                           ("whole_rate", "whole_recording", "whole-recording rate"),
                                           ("late_rate", "late_phase", "late-phase rate"))):
        raw = arrays[key]
        m = np.isfinite(raw) & (raw > 0) & (p_rec > 0)
        if key == "free_offset_fit_rate":
            m &= arrays["usable"]
        ax.loglog(raw[m], p_rec[m], "o", ms=5, alpha=0.7)
        ax.set_xlabel(f"raw: {lab} (per 100 frames; positive values)")
        ax.set_ylabel("estimator: prob_photo_bleach (per 100 frames)")
        s = result["per_view"][v]["rates_per_recording"][name]
        ax.set_title(f"n {s['n_positive']} of {s['n']}: Spearman {_r(s['spearman_r'])}, "
                     f"shift {_f(s['median_log10_ratio_nn_over_raw'], '+.2f')} dex", fontsize=10)
    fig.suptitle("Per recording: the estimator's bleaching beside the raw-domain rates (different quantities)")
    fig.tight_layout()
    fig.savefig(fig_dir / "rates_per_recording.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    pred, meas = arrays[f"{v}_analytical_dye_retention"], arrays["field_retention"]
    ax.plot(meas, pred, "o", ms=5, alpha=0.7, label="estimator's value")
    for cv, val in result["per_view"][v]["retention"]["compare_values_analytical"].items():
        ax.axhline(val, color="0.3", lw=1, ls=":", label=f"analytical at {cv}")
    ax.set_xlabel("experimental field retention (closing / opening, median-subtracted field statistic)")
    ax.set_ylabel("analytical dye retention at the estimator's value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Analytical dye retention beside experimental field retention (not a test)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "retention.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = arrays["chunk"]
    for arr, lab, col in ((p_win, "estimator (median over recordings, IQR)", "C0"),
                          (arrays["direct_window_rate"], "direct local rate (median, IQR)", "C1")):
        med = np.nanmedian(arr, axis=0)
        lo, hi = np.nanpercentile(arr, 25, axis=0), np.nanpercentile(arr, 75, axis=0)
        ax.plot(x, med, "-o", color=col, label=lab)
        ax.fill_between(x, lo, hi, color=col, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("window position")
    ax.set_ylabel("bleaching / local loss rate (per 100 frames; values ≤ 0 off the log axis)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("By window position: the estimator's bleaching and the local direct rate", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "window_positions.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    excess_label = "opening field excess (ADU per pixel)\nmean minus median pixel level; includes a camera-noise contribution"
    axes[0].plot(arrays["opening_excess"], arrays[f"{v}_mu_pc_window"][:, 0], "o", ms=5, alpha=0.7)
    axes[0].set_xlabel(excess_label, fontsize=8)
    axes[0].set_ylabel("estimator mu_pc, window 0 (photons per dye per frame)")
    c = result["per_view"][v]["brightness"]["mu_pc_window0_vs_opening_excess"]
    axes[0].set_title(f"Spearman {_r(c['spearman_r'])}", fontsize=10)
    axes[1].plot(arrays["opening_excess"], p_rec, "o", ms=5, alpha=0.7, label="estimator bleaching")
    axes[1].plot(arrays["opening_excess"], arrays["late_rate"], "s", ms=4, alpha=0.7,
                 label="raw late-phase rate (values ≤ 0 off the log axis)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(excess_label, fontsize=8)
    axes[1].set_ylabel("rate (per 100 frames)")
    b = result["per_view"][v]["brightness"]
    axes[1].set_title(f"Spearman: estimator {_r(b['nn_bleaching_vs_opening_excess']['spearman_r'])}, "
                      f"late rate {_r(b['late_rate_vs_opening_excess']['spearman_r'])}", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Brightness: different quantities placed side by side")
    fig.tight_layout()
    fig.savefig(fig_dir / "brightness.png", dpi=150)
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


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def write_outputs(result, arrays, exp, raw, R):
    out = R["out_dir"]
    (out / "figures").mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "cross_check.npz", **arrays)
    (out / "cross_check.json").write_text(json.dumps(result, indent=1, default=_json_default), encoding="utf-8")
    write_tables(result, R, out / "cross_check.md")
    write_figures(result, arrays, out)
    ident = result["code_identity"]
    m = exp["manifest"]
    prov = ["# Provenance", "",
            f"Record: `{out.name}`. Compiled by `Script_Bank/Analysis/{Path(__file__).name}` (package {ident['package_version']}, "
            f"git {ident['git_head'] or 'not a checkout'}{', working tree modified' if ident['git_dirty'] else ''}; script "
            f"sha256 `{ident['script_sha256']}`). Method: the companion note of the same name.", "",
            "## Inputs (read-only)", "",
            f"- Experiment product `{Path(exp['path']).parent.name}` (sha256 `{exp['sha256'][:16]}`): checkpoint "
            f"`{str(m.get('checkpoint_sha256') or '-')[:16]}`, job {(m.get('execution') or {}).get('job_id') or '-'}, invocation "
            f"`{(m.get('run_identity') or {}).get('invocation_id') or '-'}`, pool {m.get('pool_mode')}, draws {m.get('n_summary_draws')}, "
            f"window geometry `{json.dumps(exp['geometry'], default=str)}`.",
            f"- Raw-domain record `{Path(raw['path']).parent.name}/{Path(raw['path']).name}` (sha256 `{raw['sha256'][:16]}`): "
            f"domain raw, {result['n_recordings']} recordings, {result['n_usable_fits']} usable fits, comparison values "
            f"{raw['compare_values']}.", "",
            "## Alignment", "",
            "Recording by recording on the recording index; the local rates window by window at the same frame start "
            "(the estimator's window index times its step against the raw record's window starts). Equal recording sets "
            "and matching starts are required.", "",
            "## Purpose and authorization", "",
            f"A diagnostic of the estimator's inference against the raw-domain decline of the same recordings: it assumes "
            f"neither the estimator's values nor any selected simulation vector to be correct. "
            f"{('Authorization: ' + R['authorization'] + '.') if R['authorization'] else 'Authorization: not recorded.'}"]
    (out / "PROVENANCE.md").write_text("\n".join(prov) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        f"# {out.name}\n\nThe detector estimator `{R['estimator']}` on the experimental recordings (its Experiment product) "
        f"against the raw-domain fluorescence-decline record of the same recordings, recording by recording and window by "
        f"window: the estimator's per-dye bleaching probability beside the free-offset fitted, whole-recording, "
        f"late-phase and local direct rates of the median-subtracted field statistic; the analytical dye retention at "
        f"the estimator's value beside the experimental field retention (not a test); the estimator's per-dye brightness "
        f"beside the opening field excess. Counts, medians and correlations over every finite pair, logarithmic shifts "
        f"over the positive pairs with their own counts. Different quantities side by side, never an "
        f"equality; no ground truth; a diagnostic that assumes neither the estimator nor a selected vector correct. "
        f"`cross_check.json` holds every number, `cross_check.npz` the aligned arrays, `figures/` the scatter, retention, "
        f"window-position and brightness figures, `PROVENANCE.md` the inputs, the code identity and the authorization. "
        f"The reading belongs to DETECTOR_WORKFLOW.md.\n", encoding="utf-8")


# --- entry point ------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True, help="model window of the estimator (s).")
    ap.add_argument("--condition", default="FAB", choices=LABELING_CONDITIONS)
    ap.add_argument("--estimator", required=True, help="the estimator's artifact tag, or 'baseline'.")
    ap.add_argument("--raw-span-seconds", type=float, default=20.0, help="recording span of the raw-domain record (s).")
    ap.add_argument("--raw-record", default=None, help="the raw-domain record's arrays (.npz) instead of the Posit default.")
    ap.add_argument("--authorization", default=None, help="who authorized this diagnostic and when; recorded in the provenance.")
    ap.add_argument("--posit", default=None, help="read the products from this folder instead of the machine's Posit tier.")
    ap.add_argument("--out-dir", default=None, help="write here instead of the Posit tier's cross-check folder.")
    ap.add_argument("--dry-run", action="store_true", help="resolve the inputs and outputs; compute nothing.")
    args = ap.parse_args(argv)

    cfg = detector_workflow()
    R = resolve(cfg, args)
    assert all(is_log_row(e) for e in R["table"] if e["KEY"] in R["keys"]), "imaging rows are expected in log10"
    print(f"=== Raw-domain decline cross-check | {R['paths'].project_alias}_{R['timing'].label} | {R['estimator']} | "
          f"package {_PACKAGE_VERSION}")
    print(f"  Experiment product: {'present' if R['experiment'].exists() else 'MISSING'} ({R['experiment'].parent.name})")
    print(f"  raw-domain record:  {'present' if R['raw'].exists() else 'MISSING'} ({R['raw'].parent.name})")
    print(f"  writes: {R['out_dir']}/")
    if not R["experiment"].exists():
        raise SystemExit("the estimator has no Experiment product; nothing to cross-check")
    if not R["raw"].exists():
        raise SystemExit(f"raw-domain record not found: {R['raw']}")
    if R["out_dir"].exists():
        raise SystemExit(f"refusing to overwrite an existing cross-check: {R['out_dir']}")
    if args.dry_run:
        print("dry run: nothing computed.")
        return 0
    exp = load_experiment(R["experiment"], R["keys"])
    raw = load_raw(R["raw"], R["condition"], args.total_time_seconds)
    al = align(exp, raw, R["keys"], R["table"])
    result, arrays = compile_cross_check(al, raw, R["keys"])
    result["estimator"] = R["estimator"]
    result["code_identity"] = code_identity(Path(__file__).resolve().parents[2])
    write_outputs(result, arrays, exp, raw, R)
    P = result["per_view"]["median"]
    w, a = P["rates_per_recording"]["whole_recording"], P["window_positions"]["all_windows"]
    print(f"  bleaching (median view) vs whole-recording rate: Spearman {_r(w['spearman_r'])}, "
          f"shift {_f(w['median_log10_ratio_nn_over_raw'], '+.3f')} dex over {w['n_positive']} of {w['n']}")
    print(f"  windows: estimator larger in {a['n_nn_higher']} of {a['n']} ({a['n_nn_higher_positive']} of {a['n_positive']} "
          f"with a positive local rate; {a['n_raw_nonpositive']} with a local rate of zero or below)")
    print(f"  analytical dye retention {P['retention']['analytical_dye_retention_median']:.3f} beside experimental field "
          f"retention {P['retention']['field_retention_median']:.3f} (not a test)")
    print(f"written: {R['out_dir']}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
