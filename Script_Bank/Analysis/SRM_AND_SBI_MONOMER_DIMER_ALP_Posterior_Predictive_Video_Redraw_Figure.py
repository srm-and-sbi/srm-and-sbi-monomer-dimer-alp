#!/usr/bin/env python
"""Redraw the comparison figure of persisted posterior-predictive clips, from the clips alone.

Part of the posterior-predictive video check (``SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md``).
For each ``*_Synthetic_Video.npz`` given, the engine's ``draw_comparison_figure`` draws the figure again
from the stored clip: the same two videos, and the provenance and labels the render itself states
(``comparison_figure_inputs``, which the render uses to draw its own figure), under the display window
chosen (``--display-norm``, by default the engine's default). Nothing is simulated or rendered; only the
clip is read (its trajectory is not needed). A labeling arm's clip (``..._Labeling_Arm.py``) is redrawn the same
way, its labeling line and motion following the arm record it carries.

The figure goes to ``<stem>_Comparison.png`` beside the clip, or under ``--out-dir``. An existing figure
is replaced only under ``--replace``: without it, the run stops before drawing anything, and a figure that
appears while drawing is not replaced either. Two clips that would draw the same figure are refused. Every
figure is drawn to a temporary file first and moved into place only once complete; a failed or interrupted
draw removes the temporary file and leaves any old figure in place.

Usage::

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Redraw_Figure.py \
        <data_bank>/Posit/..._Posterior_Predictive_Video/<stem>_Synthetic_Video.npz [more clips ...] --replace
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

CLIP_SUFFIX = "_Synthetic_Video.npz"
FIGURE_SUFFIX = "_Comparison.png"


def figure_path(clip, out_dir=None) -> Path:
    """``<stem>_Comparison.png`` beside the clip, or under ``out_dir``: the name the engine gives it."""
    clip = Path(clip)
    if not clip.name.endswith(CLIP_SUFFIX):
        raise SystemExit(f"not a posterior-predictive clip (expected *{CLIP_SUFFIX}): {clip}")
    return (Path(out_dir) if out_dir else clip.parent) / (clip.name[: -len(CLIP_SUFFIX)] + FIGURE_SUFFIX)


def plan_redraws(clips, out_dir=None, replace=False) -> list:
    """``[(clip, figure), ...]``, checked before any drawing: every clip exists, and an existing figure
    is replaced only when ``replace`` says so."""
    jobs, targets = [], {}
    for clip in clips:
        clip = Path(clip)
        if not clip.is_file():
            raise SystemExit(f"clip not found: {clip}")
        figure = figure_path(clip, out_dir)
        key = figure.resolve()
        if key in targets:
            raise SystemExit(f"two clips would draw the same figure {figure}: {targets[key]} and {clip}")
        targets[key] = clip
        if figure.exists() and not replace:
            raise SystemExit(f"refusing to replace an existing figure without --replace: {figure}")
        jobs.append((clip, figure))
    return jobs


def workflow_config(tag: str):
    """The workflow configuration a clip names in its ``workflow`` field."""
    from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow, detector_workflow
    if tag == "biology":
        return biology_workflow()
    if tag == "detector":
        return detector_workflow()
    raise SystemExit(f"the clip names an unknown workflow {tag!r}")


def redraw(clip: Path, figure: Path, display_norm: str, display_percentiles, replace: bool = False) -> None:
    """Draw ``clip``'s figure to ``figure`` through a temporary file, moved into place when complete. An
    existing figure is replaced only when ``replace`` says so; the temporary file never outlives the call."""
    from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv
    d = np.load(clip, allow_pickle=False)
    figure.parent.mkdir(parents=True, exist_ok=True)
    partial = figure.with_name(figure.name[: -len(".png")] + ".partial.png")
    try:
        ppv.draw_comparison_figure(d, partial, workflow_config(str(d["workflow"])), display_norm,
                                   tuple(display_percentiles))
        if figure.exists() and not replace:
            raise SystemExit(f"refusing to replace a figure that appeared while drawing, without --replace: {figure}")
        os.replace(partial, figure)
    finally:
        if partial.exists():
            partial.unlink()


def main(argv=None):
    from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="+", help=f"*{CLIP_SUFFIX} files written by the engine or a labeling arm")
    ap.add_argument("--display-norm", default=ppv.DISPLAY_NORM_DEFAULT, choices=ppv.DISPLAY_NORMS,
                    help=f"the figure's display window (default '{ppv.DISPLAY_NORM_DEFAULT}'; see the engine's "
                         "--display-norm)")
    ap.add_argument("--display-percentiles", type=float, nargs=2, default=list(ppv.DISPLAY_PERCENTILES),
                    metavar=("LOWER", "UPPER"), help="the 'percentile' window's pair (used only by 'percentile')")
    ap.add_argument("--out-dir", default=None, help="write the figures here instead of beside the clips")
    ap.add_argument("--replace", action="store_true", help="replace an existing figure (refused otherwise)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be drawn and exit")
    args = ap.parse_args(argv)
    ppv.check_display_percentiles(args.display_percentiles)
    jobs = plan_redraws(args.clips, args.out_dir, args.replace)
    for clip, figure in jobs:
        if args.dry_run:
            print(f"[DRY RUN] {clip.name} -> {figure}" + ("  (replaces the existing figure)" if figure.exists() else ""))
            continue
        t0 = time.time()
        redraw(clip, figure, args.display_norm, args.display_percentiles, replace=args.replace)
        print(f"{figure}  ({args.display_norm}; {time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
