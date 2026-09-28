#!/usr/bin/env python
"""Render the viewer notebook's real-time player for posterior-predictive clips, headlessly, to HTML.

Part of the posterior-predictive video check (``SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md``).
For each ``*_Synthetic_Video.npz`` given, the viewer notebook
(``notebooks/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb``) is executed on a
copy with ``CLIP_PATH`` set to the clip, and the executed copy is exported to one self-contained HTML
file, ``<stem>_Player.html`` beside the clip (or under ``--out-dir``): the clip's provenance as the
notebook prints it, and the side-by-side player at the recording's frame rate with every frame embedded
losslessly (PNG frames). The HTML plays in any browser without a kernel.

The copy differs from the notebook in two ways only: ``CLIP_PATH`` (and, if given, ``NORM_MODE``,
``NORM_PERCENTILES`` and ``PLAY_EVERY``) is set, and the scrubber's ``interact(...)`` widget call is dropped. The widget needs a
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


def player_path(clip, out_dir=None):
    """``<stem>_Player.html`` beside the clip, or under ``out_dir``."""
    clip = Path(clip)
    if not clip.name.endswith(CLIP_SUFFIX):
        raise SystemExit(f"not a posterior-predictive clip (expected *{CLIP_SUFFIX}): {clip}")
    stem = clip.name[: -len(CLIP_SUFFIX)]
    return (Path(out_dir) if out_dir else clip.parent) / f"{stem}_Player.html"


def _source(cell):
    return "".join(cell.source) if isinstance(cell.source, list) else cell.source


def prepare_notebook(nb, clip, norm_mode=None, play_every=None, norm_percentiles=None):
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
    if play_every is not None:
        text, n = re.subn(r"^PLAY_EVERY = \d+", f"PLAY_EVERY = {int(play_every)}", _source(player), flags=re.M)
        if n != 1:
            raise SystemExit("the notebook's player cell has no single PLAY_EVERY line")
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


def render(clip, out, *, norm_mode=None, play_every=None, norm_percentiles=None, timeout=2400):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    nb = prepare_notebook(nbformat.read(NOTEBOOK, as_version=4), clip, norm_mode, play_every, norm_percentiles)
    start = time.time()
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    report = player_report(nb)
    if report is None:
        raise SystemExit("the executed notebook did not report its player (no 'player: ...' line)")
    mb, embedded, total, note = report
    if embedded != total:
        raise SystemExit(f"player truncated: {embedded} of {total} frames embedded ({mb:.1f} MB) {note}")
    body, _ = HTMLExporter().from_notebook_node(nb)
    out.write_text(body, encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1e6:.0f} MB on disk; player {mb:.1f} MB, {embedded} of {total} frames; "
          f"{time.time() - start:.0f} s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="+", help=f"*{CLIP_SUFFIX} files written by the engine")
    ap.add_argument("--out-dir", default=None, help="write the players here instead of beside the clips")
    ap.add_argument("--norm-mode", default=None, choices=("full", "percentile"),
                    help="override the notebook's NORM_MODE (default: the notebook's own)")
    ap.add_argument("--norm-percentiles", type=float, nargs=2, default=None, metavar=("LOWER", "UPPER"),
                    help="override the notebook's NORM_PERCENTILES pair of the 'percentile' window "
                         "(0 <= LOWER < UPPER <= 100; the notebook's own default is 0 99.99)")
    ap.add_argument("--play-every", type=int, default=None, help="override the notebook's PLAY_EVERY stride")
    ap.add_argument("--timeout", type=int, default=2400, help="per-cell execution timeout in seconds")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written and exit")
    args = ap.parse_args(argv)
    jobs = []
    for clip in args.clips:
        clip = Path(clip)
        if not clip.is_file():
            raise SystemExit(f"clip not found: {clip}")
        out = player_path(clip, args.out_dir)
        if out.exists():
            raise SystemExit(f"refusing to overwrite an existing player: {out}")
        jobs.append((clip, out))
    for clip, out in jobs:
        if args.dry_run:
            print(f"[DRY RUN] {clip} -> {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        render(clip, out, norm_mode=args.norm_mode, play_every=args.play_every,
               norm_percentiles=args.norm_percentiles, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
