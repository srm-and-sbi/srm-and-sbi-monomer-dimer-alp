#!/usr/bin/env python
"""Render the viewer notebook's real-time player for posterior-predictive clips, headlessly, to HTML.

Part of the posterior-predictive video check (``SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md``).
For each ``*_Synthetic_Video.npz`` given, the viewer notebook
(``notebooks/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb``) is executed on a
copy with ``CLIP_PATH`` set to the clip, and the executed copy is exported to one self-contained HTML
file, ``<stem>_Player.html`` beside the clip (or under ``--out-dir``): the clip's provenance as the
notebook prints it, and the side-by-side player at the recording's frame rate with every frame embedded
losslessly (PNG frames). The HTML plays in any browser without a kernel.

Every player shows its display window, by default the notebook's ``percentile`` window at (0, 99.99), in its
title. ``--play-zoom Z`` plays a cropped region, zoomed ``Z`` times around the frame center (or
``--play-center CX CY``), under the same whole-clip window; its file is ``<stem>_Player_Zoom<Z>.html``, so it
never collides with the full-frame player. The copy differs from the notebook in two ways only:
``CLIP_PATH`` (and, if given, ``NORM_MODE``, ``NORM_PERCENTILES``, ``PLAY_EVERY``, ``PLAY_ZOOM`` and the
center) is set, and the scrubber's ``interact(...)`` widget call is dropped. The widget needs a
live kernel; a headless execution never returns from the cell that displays it. The scrubber's helper
definitions stay, since the player uses them.

The player must hold every frame: the run fails (exit 2) when the notebook reports a truncation at
the embed limit. An existing output is never overwritten (exit 1 before any work).

Pure viewer tooling: it needs ``nbformat``, ``nbclient`` and ``nbconvert`` (the notebook environment)
plus the notebook's own imports; no project package and no ``MACHINE_PROFILE``.

Usage::

    python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Player.py \
        <data_bank>/Posit/..._Posterior_Predictive_Video/<stem>_Synthetic_Video.npz [more clips ...]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks" / "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb"
CLIP_SUFFIX = "_Synthetic_Video.npz"
CLIP_PLACEHOLDER = 'CLIP_PATH = "<absolute path to ..._Synthetic_Video.npz>"'
PLAYER_LINE = re.compile(r"player: ([0-9.]+) MB, (\d+) of (\d+) frames embedded(.*)")
NORM_MODES = ("percentile", "experimental", "synthetic", "full")
"""The notebook's display windows (the engine's ``DISPLAY_NORMS``, repeated because this script imports
no project package; a test keeps the lists equal). Without ``--norm-mode`` a player takes the notebook's
own default, the first: ``percentile`` at the notebook's pair (0, 99.99), the minimum to the p99.99."""


def zoom_token(play_zoom=None) -> str:
    """The file-name token of a zoomed player: '' at zoom 1 (or none given), ``_Zoom2``, ``_Zoom1p5``, ..."""
    if play_zoom is None or float(play_zoom) == 1.0:
        return ""
    return "_Zoom" + f"{float(play_zoom):g}".replace(".", "p")


def check_play_zoom(play_zoom):
    """A zoom of at least 1 (the notebook crops, it never shrinks)."""
    zoom = float(play_zoom)
    if not zoom >= 1.0:
        raise SystemExit(f"--play-zoom must be at least 1, not {play_zoom!r}")
    return zoom


def player_path(clip, out_dir=None, play_zoom=None):
    """``<stem>_Player.html`` (``<stem>_Player_Zoom<Z>.html`` for a zoomed player) beside the clip, or
    under ``out_dir``."""
    clip = Path(clip)
    if not clip.name.endswith(CLIP_SUFFIX):
        raise SystemExit(f"not a posterior-predictive clip (expected *{CLIP_SUFFIX}): {clip}")
    stem = clip.name[: -len(CLIP_SUFFIX)]
    return (Path(out_dir) if out_dir else clip.parent) / f"{stem}_Player{zoom_token(play_zoom)}.html"


def _source(cell):
    return "".join(cell.source) if isinstance(cell.source, list) else cell.source


def prepare_notebook(nb, clip, norm_mode=None, play_every=None, norm_percentiles=None, play_zoom=None,
                     play_center=None):
    """The headless copy: ``CLIP_PATH`` (and the optional settings) set, the scrubber widget call dropped."""
    code = [c for c in nb.cells if c.cell_type == "code"]
    if len(code) != 4:
        raise SystemExit(f"the notebook has {len(code)} code cells; expected 4 (imports, configuration, scrubber, player)")
    cfg, scrub, player = code[1], code[2], code[3]
    text = _source(cfg)
    if text.count(CLIP_PLACEHOLDER) != 1:
        raise SystemExit(f"the notebook's configuration cell has no single {CLIP_PLACEHOLDER!r} line")
    text = text.replace(CLIP_PLACEHOLDER, f'CLIP_PATH = "{Path(clip).resolve()}"')
    if norm_mode is not None:
        if norm_mode not in NORM_MODES:
            raise SystemExit(f"--norm-mode must be one of {', '.join(NORM_MODES)}, not {norm_mode!r}")
        text, n = re.subn(r'^NORM_MODE = "[a-z]+"$', f'NORM_MODE = "{norm_mode}"', text, flags=re.M)
        if n != 1:
            raise SystemExit("the notebook's configuration cell has no single NORM_MODE line")
    if norm_percentiles is not None:
        lo, hi = (float(v) for v in norm_percentiles)
        if not (0.0 <= lo < hi <= 100.0):
            raise SystemExit(f"--norm-percentiles must satisfy 0 <= lower < upper <= 100, not ({lo:g}, {hi:g})")
        text, n = re.subn(r"^NORM_PERCENTILES = \([^)]*\)", f"NORM_PERCENTILES = ({lo:g}, {hi:g})", text, flags=re.M)
        if n != 1:
            raise SystemExit("the notebook's configuration cell has no single NORM_PERCENTILES line")
    cfg.source = text
    head, sep, _ = _source(scrub).partition("\ninteract(")
    if not sep:
        raise SystemExit("the notebook's scrubber cell has no interact(...) call to drop")
    scrub.source = head + "\n# Headless render: the scrubber widget (interact) is omitted; it needs a live kernel.\n"
    text = _source(player)
    if play_every is not None:
        text, n = re.subn(r"^PLAY_EVERY = \d+", f"PLAY_EVERY = {int(play_every)}", text, flags=re.M)
        if n != 1:
            raise SystemExit("the notebook's player cell has no single PLAY_EVERY line")
    if play_zoom is not None:
        text, n = re.subn(r"^PLAY_ZOOM = [0-9.]+$", f"PLAY_ZOOM = {check_play_zoom(play_zoom)!r}", text, flags=re.M)
        if n != 1:
            raise SystemExit("the notebook's player cell has no single PLAY_ZOOM line")
    if play_center is not None:
        cx, cy = (int(v) for v in play_center)
        text, n = re.subn(r"^PLAY_CX, PLAY_CY = .*$", f"PLAY_CX, PLAY_CY = {cx}, {cy}", text, flags=re.M)
        if n != 1:
            raise SystemExit("the notebook's player cell has no single PLAY_CX, PLAY_CY line")
    player.source = text
    return nb


def player_report(nb):
    """The player cell's ``player: ...`` line (size and embedded frames), or None."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                m = PLAYER_LINE.search("".join(out.get("text", "")))
                if m:
                    return float(m[1]), int(m[2]), int(m[3]), m[4].strip()
    return None


def render(clip, out, *, norm_mode=None, play_every=None, norm_percentiles=None, play_zoom=None,
           play_center=None, timeout=2400):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    nb = prepare_notebook(nbformat.read(NOTEBOOK, as_version=4), clip, norm_mode, play_every, norm_percentiles,
                          play_zoom, play_center)
    start = time.time()
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    report = player_report(nb)
    if report is None:
        raise SystemExit("the executed notebook did not report its player (no 'player: ...' line)")
    mb, embedded, total, note = report
    if embedded != total:
        print(f"player truncated: {embedded} of {total} frames embedded ({mb:.1f} MB) {note}", file=sys.stderr)
        raise SystemExit(2)
    body, _ = HTMLExporter().from_notebook_node(nb)
    out.write_text(body, encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1e6:.0f} MB on disk; player {mb:.1f} MB, {embedded} of {total} frames; "
          f"{time.time() - start:.0f} s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="+", help=f"*{CLIP_SUFFIX} files written by the engine")
    ap.add_argument("--out-dir", default=None, help="write the players here instead of beside the clips")
    ap.add_argument("--norm-mode", default=None, choices=NORM_MODES,
                    help="override the notebook's NORM_MODE (default: the notebook's own, "
                         f"'{NORM_MODES[0]}')")
    ap.add_argument("--norm-percentiles", type=float, nargs=2, default=None, metavar=("LOWER", "UPPER"),
                    help="override the notebook's NORM_PERCENTILES pair of the 'percentile' window "
                         "(0 <= LOWER < UPPER <= 100; the notebook's own default is 0 99.99)")
    ap.add_argument("--play-every", type=int, default=None, help="override the notebook's PLAY_EVERY stride")
    ap.add_argument("--play-zoom", type=float, default=None,
                    help="play a cropped region zoomed this many times (at least 1; default the notebook's 1, the "
                         "full frame); the file is <stem>_Player_Zoom<Z>.html")
    ap.add_argument("--play-center", type=int, nargs=2, default=None, metavar=("CX", "CY"),
                    help="center of the zoomed region in pixels (default the notebook's frame center)")
    ap.add_argument("--timeout", type=int, default=2400, help="per-cell execution timeout in seconds")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written and exit")
    args = ap.parse_args(argv)
    if args.play_zoom is not None:
        check_play_zoom(args.play_zoom)
    jobs = []
    for clip in args.clips:
        clip = Path(clip)
        if not clip.is_file():
            raise SystemExit(f"clip not found: {clip}")
        out = player_path(clip, args.out_dir, args.play_zoom)
        if out.exists():
            raise SystemExit(f"refusing to overwrite an existing player: {out}")
        jobs.append((clip, out))
    for clip, out in jobs:
        if args.dry_run:
            print(f"[DRY RUN] {clip} -> {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        render(clip, out, norm_mode=args.norm_mode, play_every=args.play_every,
               norm_percentiles=args.norm_percentiles, play_zoom=args.play_zoom, play_center=args.play_center,
               timeout=args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
