#!/usr/bin/env python
"""Extreme-tail structure of posterior-predictive renders: experimental versus synthetic arms, per recording.

Part of the posterior-predictive video check (``SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md``).
For every recording given, the persisted render under ``--source-label`` (the production labeling law) and
every labeling arm under ``--arm-label`` (``..._Posterior_Predictive_Video_Labeling_Arm.py``) are compared
with the experimental recording they share, on the bright tail where the observation model shows: pixel
quantiles, the per-frame maximum, the pixels above a threshold and their spatial and temporal structure.
Genuinely bright emitters are PSF-shaped (several hot pixels in one frame) and revisit the same place;
camera gain fluctuations are single-pixel, single-frame events. The statistics are computed over the
opening window (``--opening-frames``) and over the whole clip.

The ``4 x 4-px bin`` statistics count the distinct frames in which a fixed bin holds a hot pixel: a
REPEATED HOT-PIXEL OCCUPANCY. Frames need not be consecutive and different emitters can visit one bin, so
it is not a spot lifetime and not a dye survival time.

The clips compared are the declared-configuration biology render under ``--source-label`` and the labeling
arms named with ``--arm-label``, matched exactly as the engine and the labeling arm name their files. Before
anything is computed or written, the script checks every recording's clips: they exist, share the
experimental frames, carry one and the same imaging vector (printed in the report), and every arm labeled
exactly the source's subunits. A violation stops the run with nothing written.

Pure render consumer: no simulation. It writes a report, a summary JSON and survival figures to
``<data_bank>/Posit/<alias>_<timing>_Posterior_Predictive_Video_Tail_Structure_<SOURCE_LABEL>/``, one folder per
source label; a rerun under the same source label regenerates that folder's files (they are derived from the
persisted clips and carry no unique data), so another set of arms for the same source needs another source
label to keep both reports.

Usage::

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Brightness_Tail_Structure.py \
        --total-time-seconds 2 --kind MET-FAB --cell 0 5 16 --source-label REF --arm-label BINOMIAL4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from srm_and_sbi_monomer_dimer_alp import __version__ as _PACKAGE_VERSION
from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv
from srm_and_sbi_monomer_dimer_alp.experiment_support import KIND_OF_CONDITION
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow

QUANTILES = ((50, "median"), (99, "p99"), (99.9, "p99.9"), (99.99, "p99.99"), (100, "max"))
STRUCTURE_ROWS = (
    ("hot pixels", "npix", "{:,d}"),
    ("per-frame 8-connected components", "blobs", "{:,d}"),
    ("PSF-shaped components (3+ px)", "blobs_3plus", "{:,d}"),
    ("fraction with a hot 8-neighbor, same frame", "frac_neighbor", "{:.2f}"),
    ("median local ground / stack median", "local_over_background", "{:.2f}"),
    ("distinct 4x4-px bins with a hot pixel", "sites", "{:,d}"),
    ("repeated hot-pixel occupancy per bin, median (frames)", "frames_per_site_median", "{:.0f}"),
    ("repeated hot-pixel occupancy per bin, max (frames)", "frames_per_site_max", "{:,d}"),
)


def label_token(label: str) -> str:
    """A run or arm label as the engine (``_build_stem``) and the labeling arm write it in file names."""
    token = "".join(c if c.isalnum() else "_" for c in str(label)).strip("_")
    if not token:
        raise SystemExit(f"a label must contain at least one alphanumeric character, not {label!r}.")
    return token


def output_dir(posit, project_alias: str, map_label: str, source_label: str) -> Path:
    """The folder of one comparison, keyed by the source renders' run label, so comparisons of renders
    under different source labels never overwrite each other."""
    return Path(posit) / f"{project_alias}_{map_label}_Posterior_Predictive_Video_Tail_Structure_{label_token(source_label)}"


def structure(stack: np.ndarray, threshold: int) -> dict:
    """Spatial and temporal structure of the pixels above ``threshold``."""
    mask = stack > threshold
    npix = int(mask.sum())
    if npix == 0:
        return dict(npix=0, blobs=0, blobs_3plus=0, frac_neighbor=0.0, local_over_background=float("nan"),
                    sites=0, frames_per_site_median=0.0, frames_per_site_max=0)
    blobs = big = 0
    for t in np.unique(np.argwhere(mask)[:, 0]):
        labels, n = ndimage.label(mask[t], structure=np.ones((3, 3)))
        blobs += n
        sizes = np.bincount(labels.ravel())[1:]
        big += int((sizes >= 3).sum())
    hot = mask.astype(np.uint8)
    neighbors = np.zeros_like(hot)
    h, w = hot.shape[1], hot.shape[2]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            neighbors[:, max(0, dx):h + min(0, dx), max(0, dy):w + min(0, dy)] += \
                hot[:, max(0, -dx):h + min(0, -dx), max(0, -dy):w + min(0, -dy)]
    frac_neighbor = float((neighbors[mask] > 0).mean())
    background = float(np.median(stack))
    idx = np.argwhere(mask)
    ratios = []
    for t, x, y in idx[:: max(1, len(idx) // 400)]:
        patch = stack[t, max(0, x - 5):x + 6, max(0, y - 5):y + 6].astype(float)
        ratios.append(float(np.median(patch)) / background)
    sites: dict = {}
    for t, x, y in idx:
        sites.setdefault((int(x) // 4, int(y) // 4), set()).add(int(t))
    per_site = np.array([len(v) for v in sites.values()])
    return dict(npix=npix, blobs=blobs, blobs_3plus=big, frac_neighbor=frac_neighbor,
                local_over_background=float(np.median(ratios)), sites=len(sites),
                frames_per_site_median=float(np.median(per_site)), frames_per_site_max=int(per_site.max()))


def window_stats(stack: np.ndarray, threshold: int, exp_max: int | None) -> dict:
    fm = stack.reshape(stack.shape[0], -1).max(1)
    out = {"quantiles": {name: float(np.percentile(stack, q)) for q, name in QUANTILES},
           "per_frame_max_median": float(np.median(fm)), "per_frame_max_p90": float(np.percentile(fm, 90)),
           "structure": structure(stack, threshold)}
    if exp_max is not None:
        out["above_experimental_max"] = int((stack > exp_max).sum())
    return out


def survival_figure(path: Path, title: str, stacks: dict, threshold: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    hi = max(float(s.max()) for s in stacks.values())
    grid = np.linspace(2000, hi, 400)
    for label, stack in stacks.items():
        values = np.sort(stack.ravel())
        surv = 1.0 - np.searchsorted(values, grid, side="right") / values.size
        ax.semilogy(grid, np.clip(surv, 1e-9, None), linewidth=1.7, label=label)
    ax.axvline(threshold, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("pixel value (ADU)"); ax.set_ylabel("survival fraction P(pixel > x)")
    ax.set_title(title, fontsize=10, loc="left"); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fmt_ratio(value: float, reference: float) -> str:
    if reference and np.isfinite(reference):
        return f"{value:,.0f} ({value / reference:.2f})"
    return f"{value:,.0f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True, help="the renders' model window (timing label).")
    ap.add_argument("--kind", default="MET-FAB", choices=tuple(KIND_OF_CONDITION))
    ap.add_argument("--cell", type=int, nargs="+", required=True, help="recording indices.")
    ap.add_argument("--experiment-span-seconds", type=int, default=20, help="the renders' clip span (s).")
    ap.add_argument("--source-label", required=True, help="run label of the production-law renders (e.g. REF).")
    ap.add_argument("--arm-label", action="append", default=[], help="labeling arm token(s) to compare; repeatable.")
    ap.add_argument("--threshold-adu", type=int, default=10_000, help="hot-pixel threshold (ADU).")
    ap.add_argument("--opening-frames", type=int, default=100, help="length of the opening window (frames).")
    args = ap.parse_args(argv)

    cfg = biology_workflow()
    kind_token = KIND_OF_CONDITION[args.kind]
    paths = cfg.paths.with_condition(kind_token)
    map_label = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing).label
    posit = PARAMETERS.machine.data_bank_root / paths.posit_subdir
    ppv_dir = posit / f"{paths.project_alias}_{map_label}_Posterior_Predictive_Video"
    out_dir = output_dir(posit, paths.project_alias, map_label, args.source_label)
    span = f"{args.experiment_span_seconds}S"
    arm_tokens = [label_token(a) for a in args.arm_label]

    # ---- resolve and check every recording's clips before anything is computed or written ----------------
    imaging_vector = None
    column_names = None
    resolved = {}
    for cell in args.cell:
        source_stem = ppv._build_stem(paths.project_alias, map_label, args.kind, cell, None, "cell-sgm", span,
                                      run_label=args.source_label, declared_rds=True, map_block="rds")
        clips = {}
        for title, stem in [(f"{args.source_label} ({{law}})", source_stem)] + \
                           [(f"{a} ({{law}})", f"{source_stem}_{tok}") for a, tok in zip(args.arm_label, arm_tokens)]:
            path = ppv.render_output_paths(ppv_dir, stem)["clip"]
            if not path.exists():
                raise SystemExit(f"cell {cell}: missing {path.name}")
            d = np.load(path, allow_pickle=False)
            clips[title.format(law=str(d["labeling_law"]))] = d
        names = list(clips)
        if column_names is None:
            column_names = names
        elif names != column_names:
            raise SystemExit(f"cell {cell}: arm laws {names} differ from {column_names}")
        first = clips[names[0]]
        experimental = first["experimental"]
        for name, d in clips.items():
            if not np.array_equal(d["experimental"], experimental):
                raise SystemExit(f"cell {cell}: {name} references different experimental frames")
            vec = np.asarray(d["imaging_physical"], dtype=float)
            if imaging_vector is None:
                imaging_vector = vec
            elif not np.allclose(vec, imaging_vector, rtol=0, atol=1e-9):
                raise SystemExit(f"cell {cell}: {name} was rendered with a different imaging vector {vec} != {imaging_vector}")
            if not np.array_equal(d["dye_counts"] > 0, first["dye_counts"] > 0):
                raise SystemExit(f"cell {cell}: {name} labels different subunits than {names[0]}")
        resolved[cell] = clips
        del experimental

    # ---- compute and write ---------------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"package_version": _PACKAGE_VERSION, "threshold_adu": args.threshold_adu,
               "opening_frames": args.opening_frames, "source_label": args.source_label, "arm_labels": args.arm_label,
               "cells": {}}
    for cell, clips in resolved.items():
        names = list(clips)
        first = clips[names[0]]
        experimental = first["experimental"]
        block = {"experimental_tif": str(first["experimental_tif"]), "seed": int(first["seed"]),
                 "labeled_subunits": int((first["dye_counts"] > 0).sum()),
                 "dyes": {name: int(d["dye_counts"].sum()) for name, d in clips.items()},
                 "dye_count_histogram": {name: {int(k): int(v) for k, v in zip(*np.unique(d["dye_counts"][d["dye_counts"] > 0], return_counts=True))}
                                         for name, d in clips.items()},
                 "windows": {}}
        stacks = {"experimental": experimental, **{name: d["synth"] for name, d in clips.items()}}
        for window, sl in ((f"opening {args.opening_frames} frames", slice(0, args.opening_frames)),
                           (f"whole clip ({experimental.shape[0]} frames)", slice(None))):
            exp_max = int(experimental[sl].max())
            block["windows"][window] = {name: window_stats(s[sl], args.threshold_adu, None if name == "experimental" else exp_max)
                                        for name, s in stacks.items()}
        survival_figure(out_dir / f"T_survival_{args.kind}_Cell_{cell}.png",
                        f"{args.kind} cell {cell}: pixel-value survival (whole clip)", stacks, args.threshold_adu)
        results["cells"][str(cell)] = block
        print(f"cell {cell}: done")
    results["imaging_vector"] = dict(zip([str(k) for k in first["imaging_keys"]], [float(v) for v in imaging_vector]))
    (out_dir / "tail_structure.json").write_text(json.dumps(results, indent=1))

    # ---- report -------------------------------------------------------------------------------------
    L: list[str] = []
    add = L.append
    add(f"# Extreme-tail structure: {args.kind}, {len(args.cell)} recordings, experimental versus {', '.join(column_names)}")
    add("")
    add(f"Renders under `{args.source_label}` (the production labeling law) and the labeling arm(s) "
        f"{', '.join(args.arm_label) or '(none)'} of the same trajectory and the same labeled subunits, compared with "
        f"the experimental recording each shares. Hot pixels are those above {args.threshold_adu:,} ADU. The "
        f"`4x4-px bin` statistics count distinct frames with a hot pixel in a fixed bin (repeated hot-pixel "
        "occupancy), not a spot lifetime. Ratios in parentheses are synthetic / experimental. Regenerate with "
        "`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Brightness_Tail_Structure.py`; package version "
        f"{_PACKAGE_VERSION}.")
    add("")
    add("**Imaging vector shared by every synthetic clip (verified):** "
        + ", ".join(f"{k} {v:.4g}" for k, v in results["imaging_vector"].items()) + ".")
    add("")
    for window in next(iter(results["cells"].values()))["windows"]:
        add(f"## Summary, {window}")
        add("")
        for key, label, kind in (("npix", "hot pixels", "structure"), ("max", "maximum (ADU)", "quantile"),
                                 ("per_frame_max_median", "per-frame maximum, median (ADU)", "top"),
                                 ("p99.99", "p99.99 (ADU)", "quantile"), ("blobs_3plus", "PSF-shaped components", "structure"),
                                 ("frames_per_site_max", "repeated occupancy per bin, max (frames)", "structure"),
                                 ("above_experimental_max", "pixels above the experimental maximum", "top")):
            add(f"**{label}**")
            add("")
            add("| cell | experimental | " + " | ".join(column_names) + " |")
            add("|---|---|" + "---|" * len(column_names))
            for cell, block in results["cells"].items():
                w = block["windows"][window]

                def get(name):
                    s = w[name]
                    if kind == "structure":
                        return s["structure"][key]
                    if kind == "quantile":
                        return s["quantiles"][key]
                    return s.get(key, float("nan"))
                e = get("experimental")
                cells = [f"{e:,.0f}" if np.isfinite(e) else "-"]
                for name in column_names:
                    v = get(name)
                    cells.append(fmt_ratio(v, e) if key != "above_experimental_max" else f"{v:,.0f}")
                add(f"| {cell} | " + " | ".join(cells) + " |")
            add("")
    for cell, block in results["cells"].items():
        add(f"## Cell {cell}")
        add("")
        add(f"Recording `{Path(block['experimental_tif']).name}`; seed {block['seed']}; {block['labeled_subunits']} labeled subunits; dyes "
            + "; ".join(f"{name} {n} {block['dye_count_histogram'][name]}" for name, n in block["dyes"].items()) + ".")
        add("")
        for window, w in block["windows"].items():
            add(f"*{window}*")
            add("")
            add("| quantile (ADU) | experimental | " + " | ".join(column_names) + " |")
            add("|---|---|" + "---|" * len(column_names))
            for _, name in QUANTILES:
                e = w["experimental"]["quantiles"][name]
                add(f"| {name} | {e:,.0f} | " + " | ".join(fmt_ratio(w[c]["quantiles"][name], e) for c in column_names) + " |")
            e = w["experimental"]["per_frame_max_median"]
            add(f"| per-frame max, median | {e:,.0f} | " + " | ".join(fmt_ratio(w[c]["per_frame_max_median"], e) for c in column_names) + " |")
            add("")
            add(f"| structure of pixels above {args.threshold_adu:,} ADU | experimental | " + " | ".join(column_names) + " |")
            add("|---|---|" + "---|" * len(column_names))
            for title, key, fmt in STRUCTURE_ROWS:
                add(f"| {title} | " + " | ".join(fmt.format(w[c]["structure"][key]) if w[c]["structure"]["npix"] else "-"
                                                 for c in ["experimental"] + column_names) + " |")
            add(f"| pixels above the experimental maximum | - | " + " | ".join(f"{w[c]['above_experimental_max']:,d}" for c in column_names) + " |")
            add("")
        add(f"![cell {cell} survival](T_survival_{args.kind}_Cell_{cell}.png)")
        add("")
    report = out_dir / f"{paths.project_alias}_Brightness_Tail_Structure.md"
    report.write_text("\n".join(L))
    print("wrote", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
