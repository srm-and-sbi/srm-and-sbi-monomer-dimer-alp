#!/usr/bin/env python
"""Redraw the recovery figures of a direct-estimator run folder from its saved arrays.

The three direct-estimator utilities (PSF width, flicker rate, fluorescence loss) write their recovery
figures -- the true value against the inferred one, per attempted recording, in the parameter's log10
prior coordinates and in absolute values -- into ``figures/`` of every tier run and list them in
``report.md``. This utility draws the same figures, through the same shared builder
(``direct_acceptance.recovery_figures``), for a run folder written before the figures existed, or
again after a change to the builder. It reads the run's ``direct_*.npz`` and ``summary.json``, writes
only into the run's ``figures/`` (or ``--out-dir``), and leaves ``report.md`` as the run wrote it.

Usage (from the repo root):
    PYTHONPATH=$PWD python Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Estimator_Figures.py \\
        <run folder> [--out-dir DIR] [--dpi 200]

The estimator is read from ``summary.json`` (``purpose.estimator``) or, failing that, from the name of
the arrays file. Exit status 0 when every figure was written, 2 when the folder holds no readable run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from srm_and_sbi_monomer_dimer_alp import direct_acceptance as da  # noqa: E402

# Per estimator: the arrays file, and per figure the parameter key, how to read truth/estimate and
# the nominal range from the arrays, and the accuracy rule's band (from the utilities' ACCEPTANCE).
SPECS = {
    "Direct_Fluorescence_Loss": dict(
        npz="direct_fluorescence_loss.npz",
        figures=[dict(key="prob_photo_bleach", truth=lambda z: z["truth"], estimate=lambda z: z["estimate"],
                      usable=lambda z: z["usable"] if "usable" in z.files else None,
                      low=lambda z: z["range_low"], high=lambda z: z["range_high"], ranges_are_log10=False,
                      tolerance=lambda s: ("dex", float(s.get("acceptance", {}).get("prob_bleach_mae_dex", 0.10))))]),
    "Direct_Flicker_Rate": dict(
        npz="direct_flicker_rate.npz",
        figures=[dict(key="lambda_rate", truth=lambda z: z["truth"], estimate=lambda z: z["estimate"],
                      usable=lambda z: None,
                      low=lambda z: z["range_low"], high=lambda z: z["range_high"], ranges_are_log10=False,
                      tolerance=lambda s: ("dex", float(s.get("acceptance", {}).get("lambda_mae_dex", 0.08))))]),
    "Direct_PSF_Width": dict(
        npz="direct_psf_width.npz",
        figures=[dict(key="mu_r", truth=lambda z: z["truth"][:, 0], estimate=lambda z: z["estimate"][:, 0],
                      usable=lambda z: None,
                      low=lambda z: z["range_mu_r_log10"][:, 0], high=lambda z: z["range_mu_r_log10"][:, 1],
                      ranges_are_log10=True,
                      tolerance=lambda s: ("dex", float(s.get("acceptance", {}).get("mu_r_mae_dex", 0.02)))),
                 dict(key="sigma_r", truth=lambda z: z["truth"][:, 1], estimate=lambda z: z["estimate"][:, 1],
                      usable=lambda z: None,
                      low=lambda z: z["range_sigma_r"][:, 0], high=lambda z: z["range_sigma_r"][:, 1],
                      ranges_are_log10=False,
                      tolerance=lambda s: ("linear", float(s.get("acceptance", {}).get("sigma_r_mae", 0.08))))]),
}


def identify(run_dir: str):
    """The estimator name and its arrays file for a run folder, or ``(None, reason)``."""
    summary_path = os.path.join(run_dir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.isfile(summary_path) else {}
    estimator = (summary.get("purpose") or {}).get("estimator")
    if estimator not in SPECS:
        for name, spec in SPECS.items():
            if os.path.isfile(os.path.join(run_dir, spec["npz"])):
                estimator = name
                break
    if estimator not in SPECS:
        return None, None, f"{run_dir}: no summary.json naming a direct estimator and no direct_*.npz of a tier run"
    npz = os.path.join(run_dir, SPECS[estimator]["npz"])
    if not os.path.isfile(npz):
        return None, None, f"{run_dir}: {SPECS[estimator]['npz']} is missing (an experiment-mode or selftest run has no tier arrays)"
    return estimator, summary, npz


def redraw(run_dir: str, out_dir: str | None = None, dpi: int = 200) -> list:
    """Write the figures of one run folder; returns the paths written. Raises ValueError for a folder without a run."""
    estimator, summary, npz = identify(run_dir)
    if estimator is None:
        raise ValueError(npz)
    z = np.load(npz, allow_pickle=True)
    if "truth" not in z.files or "estimate" not in z.files:
        raise ValueError(f"{npz}: holds no truth/estimate arrays (not a tier run)")
    target = out_dir or os.path.join(run_dir, "figures")
    os.makedirs(target, exist_ok=True)
    title = f"{estimator}, {os.path.basename(os.path.normpath(run_dir))}"
    written = []
    for spec in SPECS[estimator]["figures"]:
        valid = z["valid"] if "valid" in z.files else np.isfinite(spec["estimate"](z))
        try:
            low, high = spec["low"](z), spec["high"](z)
        except KeyError:
            low = high = None
        for name, fig, _caption in da.recovery_figures(
                key=spec["key"], truth=spec["truth"](z), estimate=spec["estimate"](z), valid=valid,
                usable=spec["usable"](z), range_low=low, range_high=high,
                ranges_are_log10=spec["ranges_are_log10"], tolerance=spec["tolerance"](summary), title=title):
            path = os.path.join(target, f"{name}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a direct-estimator run folder (holds summary.json and direct_*.npz)")
    ap.add_argument("--out-dir", default=None, help="where to write the PNGs (default: <run folder>/figures)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)
    try:
        for path in redraw(args.run_dir, args.out_dir, args.dpi):
            print(f"wrote {path}")
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
