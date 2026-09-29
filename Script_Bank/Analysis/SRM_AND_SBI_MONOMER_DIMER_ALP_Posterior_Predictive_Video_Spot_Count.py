#!/usr/bin/env python
"""Detected spots per frame of persisted posterior-predictive clips: experimental versus synthetic, per recording.

Part of the posterior-predictive video check (``SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md``).
A declared configuration sets the renders' receptor total: one value for every recording, or one per
recording scaled from the recording's opening spot count. This companion measures the resulting density
difference on the persisted clips, without any new render (the recordings' opening counts are also the ones
that scale the per-recording totals): the detection rule of the count match (the direct estimators' detection on the stored 8-bit levels,
4 sigma, fit half-width 14 px, SCOPE box center; ``..._Posterior_Predictive_Video_Count_Match.py``, whose
``detected_per_frame`` is reused so that there is one rule) is applied to the experimental frames and to
the synthetic clips of a recording (the declared-configuration biology render under ``--source-label`` and
the labeling arms named with ``--arm-label``), over the opening window and the closing window of the clip.
Labels are matched exactly as the engine and the labeling arm name their files.

A count is a DETECTED count: the isolated spots the direct estimators accept in a frame. At high density
the overlap and the isolation radius lower it, so the ratio synthetic / experimental is read as a density
comparison under one rule, not as a census of emitters. Beside the counts, each synthetic clip reports the
visible spots it started with (its labeling record, frame 0) and the detected-per-visible ratio ``eta`` of
the opening window, the quantity the count match iterates on.

Pure render consumer: no simulation, no render, nothing adopted. It verifies that the clips of a
recording share the experimental frames and one imaging vector, and writes a report and a JSON (the
per-frame counts) to ``<data_bank>/Posit/<alias>_<timing>_Posterior_Predictive_Video_Spot_Count_<SOURCE_LABEL>/``,
one folder per source label. Existing counts are never overwritten: the run stops before any work when the
report or the JSON exists, so another set of arms under the same source label needs its own count folder
(another source label). ``--rewrite-report`` rebuilds the report from the counts already saved in the JSON,
counting nothing, and replaces only the report.

Usage::

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Spot_Count.py \
        --total-time-seconds 2 --kind MET-FAB --cell 0 5 16 --source-label REF --arm-label BINOMIAL4 --workers 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import __version__ as _PACKAGE_VERSION
from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv
from srm_and_sbi_monomer_dimer_alp.experiment_support import KIND_OF_CONDITION
from srm_and_sbi_monomer_dimer_alp.io import convert_video_dtype
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow

COUNT_MATCH = Path(__file__).with_name("SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Count_Match.py")
EXPERIMENTAL = "experimental"


def count_match_module():
    """The count-match companion, loaded from its file: the one detection rule of the check."""
    spec = importlib.util.spec_from_file_location("count_match", COUNT_MATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detected_per_frame(video16: np.ndarray) -> np.ndarray:
    """Detected spots in every frame of a 16-bit stack, under the count match's rule (8-bit stored levels)."""
    return count_match_module().detected_per_frame(convert_video_dtype(video16, bits_from=16, bits_to=8))


def label_token(label: str) -> str:
    """A run or arm label as the engine (``_build_stem``) and the labeling arm write it in file names."""
    token = "".join(c if c.isalnum() else "_" for c in str(label)).strip("_")
    if not token:
        raise SystemExit(f"a label must contain at least one alphanumeric character, not {label!r}.")
    return token


def output_dir(posit, project_alias: str, map_label: str, source_label: str) -> Path:
    """The folder of one count, keyed by the source renders' run label, so counts of renders under
    different source labels never collide."""
    return Path(posit) / f"{project_alias}_{map_label}_Posterior_Predictive_Video_Spot_Count_{label_token(source_label)}"


def windows(n_frames: int, opening: int, closing: int) -> dict:
    """The two windows counted: the first ``opening`` frames and the last ``closing`` frames."""
    if opening < 1 or closing < 1 or opening > n_frames or closing > n_frames:
        raise ValueError(f"windows of {opening} and {closing} frames do not fit a clip of {n_frames} frames")
    return {f"opening {opening} frames": (0, opening), f"closing {closing} frames": (n_frames - closing, n_frames)}


def summarize(counts: np.ndarray) -> dict:
    counts = np.asarray(counts, dtype=float)
    return {"mean": float(counts.mean()), "median": float(np.median(counts)),
            "min": float(counts.min()), "max": float(counts.max()), "per_frame": [float(c) for c in counts]}


def _count_job(job: tuple) -> tuple:
    cell, name, window, path, key, start, stop = job
    d = np.load(path, allow_pickle=False)
    t0 = time.time()
    counts = detected_per_frame(d[key][start:stop])
    return cell, name, window, summarize(counts), time.time() - t0


def fmt_ratio(value: float, reference: float) -> str:
    if reference and np.isfinite(reference):
        return f"{value:.1f} ({value / reference:.2f})"
    return f"{value:.1f}"


def write_report(results: dict, report: Path) -> None:
    """The report of one count, written from the saved results alone (``spot_count.json``)."""
    cells, column_names = results["cells"], results["columns"]
    totals = sorted({float(block["count_total_declared"]) for block in cells.values()})
    counted_with = results["package_version"]
    L: list[str] = []
    add = L.append
    add(f"# Detected spots per frame: {results.get('kind', 'MET-FAB')}, {len(cells)} recordings, experimental versus "
        f"{', '.join(column_names)}")
    add("")
    add(f"Persisted renders under `{results['source_label']}` (the production labeling law) and the labeling arm(s) "
        f"{', '.join(results['arm_labels']) or '(none)'}, compared with the experimental recording each shares, on the "
        "number of spots the direct estimators' detection accepts per frame (stored 8-bit levels, 4 sigma, fit half-width "
        "14 px, SCOPE box center: the count match's rule). A detected count is lowered by overlap and by the isolation "
        "radius at high density, so the ratios in parentheses (synthetic / experimental) compare densities under one rule; "
        "they are not a census of emitters. Values are means over the window. Nothing was matched. Regenerate with "
        "`Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Spot_Count.py`; counted with "
        f"package version {counted_with}" + ("" if counted_with == _PACKAGE_VERSION else
                                             f", report written by package version {_PACKAGE_VERSION}") + ".")
    add("")
    add("**Imaging vector shared by every synthetic clip (verified):** "
        + ", ".join(f"{k} {v:.4g}" for k, v in results["imaging_vector"].items()) + ".")
    add("")
    for window in next(iter(cells.values()))["windows"]:
        add(f"## Detected spots per frame, {window}")
        add("")
        add("| cell | experimental | " + " | ".join(column_names) + " |")
        add("|---|---|" + "---|" * len(column_names))
        for cell, block in cells.items():
            w = block["windows"][window]
            e = w[EXPERIMENTAL]["mean"]
            add(f"| {cell} | {e:.1f} | " + " | ".join(fmt_ratio(w[c]["mean"], e) for c in column_names) + " |")
        add("")
    add("## The renders' receptor totals, visible spots at frame 0 and detected-per-visible ratio (opening window)")
    add("")
    add((f"Every render declares the receptor total N_R = {totals[0]:.0f}. " if len(totals) == 1 else
         "The receptor total is declared per recording (column N_R); a recording's arms share its render's trajectory, "
         "so they share its N_R. ")
        + "Visible spots are the labeled monomers plus the dimers with at least one labeled subunit at frame 0 (the "
        "labeling record). `eta` = detected per frame / visible spots, the count match's ratio.")
    add("")
    add("| cell | N_R | " + " | ".join(f"{c} visible" for c in column_names) + " | "
        + " | ".join(f"{c} eta" for c in column_names) + " |")
    add("|---|---|" + "---|" * (2 * len(column_names)))
    for cell, block in cells.items():
        add(f"| {cell} | {float(block['count_total_declared']):.0f} | "
            + " | ".join(f"{block['visible_spots_0'][c]}" for c in column_names) + " | "
            + " | ".join("-" if block["eta_opening"][c] is None else f"{block['eta_opening'][c]:.2f}" for c in column_names) + " |")
    add("")
    report.write_text("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True, help="the renders' model window (timing label).")
    ap.add_argument("--kind", default="MET-FAB", choices=tuple(KIND_OF_CONDITION))
    ap.add_argument("--cell", type=int, nargs="+", required=True, help="recording indices.")
    ap.add_argument("--experiment-span-seconds", type=int, default=20, help="the renders' clip span (s).")
    ap.add_argument("--source-label", required=True, help="run label of the production-law renders (e.g. REF).")
    ap.add_argument("--arm-label", action="append", default=[], help="labeling arm token(s) to include; repeatable.")
    ap.add_argument("--opening-frames", type=int, default=100, help="length of the opening window (frames).")
    ap.add_argument("--closing-frames", type=int, default=100, help="length of the closing window (frames).")
    ap.add_argument("--workers", type=int, default=1, help="parallel counting processes (one clip window each).")
    ap.add_argument("--dry-run", action="store_true", help="resolve the clips and outputs, count nothing.")
    ap.add_argument("--rewrite-report", action="store_true",
                    help="rebuild the report from the counts saved in the folder's spot_count.json (nothing is "
                         "counted) and replace only the report.")
    args = ap.parse_args(argv)

    cfg = biology_workflow()
    kind_token = KIND_OF_CONDITION[args.kind]
    paths = cfg.paths.with_condition(kind_token)
    map_label = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing).label
    posit = PARAMETERS.machine.data_bank_root / paths.posit_subdir
    ppv_dir = posit / f"{paths.project_alias}_{map_label}_Posterior_Predictive_Video"
    out_dir = output_dir(posit, paths.project_alias, map_label, args.source_label)
    report = out_dir / f"{paths.project_alias}_Spot_Count.md"
    table = out_dir / "spot_count.json"
    if args.rewrite_report:
        if not table.exists():
            raise SystemExit(f"no saved counts to rebuild the report from: {table}")
        results = json.loads(table.read_text())
        results.setdefault("kind", args.kind)
        write_report(results, report)
        print("rewrote", report, "from", table.name)
        return 0
    for existing in (report, table):
        if existing.exists():
            raise SystemExit(f"refusing to overwrite an existing result: {existing}")
    span = f"{args.experiment_span_seconds}S"
    arm_tokens = [label_token(a) for a in args.arm_label]

    # ---- resolve and verify the clips, build the jobs ------------------------------------------------
    column_names = None
    imaging_vector = None
    cells: dict = {}
    jobs = []
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
            clips[title.format(law=str(d["labeling_law"]))] = (path, d)
        names = list(clips)
        if column_names is None:
            column_names = names
        elif names != column_names:
            raise SystemExit(f"cell {cell}: arm laws {names} differ from {column_names}")
        first_path, first = clips[names[0]]
        experimental = first["experimental"]
        n_frames = int(experimental.shape[0])
        wins = windows(n_frames, args.opening_frames, args.closing_frames)
        def count_total(d):
            return float(dict(zip([str(k) for k in d["rds_keys"]], [float(v) for v in d["rds_provenance"]]))["count_total"])
        block = {"experimental_tif": str(first["experimental_tif"]), "seed": int(first["seed"]), "n_frames": n_frames,
                 "count_total_declared": count_total(first), "visible_spots_0": {}, "windows": {w: {} for w in wins}}
        for name, (path, d) in clips.items():
            if not np.array_equal(d["experimental"], experimental):
                raise SystemExit(f"cell {cell}: {name} references different experimental frames")
            if count_total(d) != block["count_total_declared"]:
                raise SystemExit(f"cell {cell}: {name} declares N_R {count_total(d):g}, not its source's "
                                 f"{block['count_total_declared']:g}")
            vec = np.asarray(d["imaging_physical"], dtype=float)
            if imaging_vector is None:
                imaging_vector = vec
            elif not np.allclose(vec, imaging_vector, rtol=0, atol=1e-9):
                raise SystemExit(f"cell {cell}: {name} was rendered with a different imaging vector {vec} != {imaging_vector}")
            row = dict(zip([str(k) for k in d["labeling_columns"]], [float(v) for v in d["labeling_row"]]))
            block["visible_spots_0"][name] = int(round(row["monomers_visible_0"] + row["dimers_visible_0"]))
            for window, (start, stop) in wins.items():
                jobs.append((cell, name, window, path, "synth", start, stop))
        for window, (start, stop) in wins.items():
            jobs.append((cell, EXPERIMENTAL, window, first_path, "experimental", start, stop))
        cells[str(cell)] = block
    keys = [str(k) for k in first["imaging_keys"]]
    print(f"{len(jobs)} clip windows to count over {len(args.cell)} recording(s); columns {EXPERIMENTAL}, "
          f"{', '.join(column_names)}; imaging vector shared by every synthetic clip: "
          + ", ".join(f"{k} {v:.4g}" for k, v in zip(keys, imaging_vector)))
    if args.dry_run:
        for cell, name, window, path, key, start, stop in jobs:
            print(f"[DRY RUN] cell {cell} {name} {window}: {path.name}[{key}][{start}:{stop}]")
        print(f"[DRY RUN] -> {report}")
        return 0

    # ---- count ------------------------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for cell, name, window, summary, seconds in pool.map(_count_job, jobs):
            cells[str(cell)]["windows"][window][name] = summary
            print(f"cell {cell} {name} {window}: {summary['mean']:.1f} detected spots per frame "
                  f"(median {summary['median']:.0f}, {summary['min']:.0f}-{summary['max']:.0f}); {seconds:.0f} s", flush=True)
    opening = next(iter(next(iter(cells.values()))["windows"]))
    for block in cells.values():
        block["eta_opening"] = {name: block["windows"][opening][name]["mean"] / v if v > 0 else None
                                for name, v in block["visible_spots_0"].items()}
    results = {"package_version": _PACKAGE_VERSION, "kind": args.kind, "source_label": args.source_label,
               "arm_labels": args.arm_label,
               "columns": column_names, "opening_frames": args.opening_frames, "closing_frames": args.closing_frames,
               "detection": {"rule": "the count match's detected_per_frame: direct_imaging_estimates.measure_spot_widths "
                                     "on stored 8-bit levels", "n_sigma": 4.0, "half_px": 14, "scope": "SCOPE box center"},
               "imaging_vector": dict(zip(keys, [float(v) for v in imaging_vector])), "cells": cells,
               "seconds": time.time() - t0}
    table.write_text(json.dumps(results, indent=1))

    write_report(results, report)
    print("wrote", report, f"({results['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
