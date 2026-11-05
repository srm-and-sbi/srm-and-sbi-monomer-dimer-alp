"""Cells of an adopted Experiment product that have no validated retained draw collection yet (2026-11-05).

Derived from the data, never from a hand-kept list: the product's cell identities (``cell`` field of
``<alias>_<COND>_<timing>_<TAG>_MAP_Experiment.npz`` under the active machine profile) minus the cells carried by
every ``<same prefix>_Window_Posterior_Draws[_<LABEL>]/<same>.npz`` collection beside it that passes
``validate_collection.py``'s checks against that product. Prints the missing cells, comma-separated, on stdout
(nothing else there); everything else goes to stderr.

Usage: MACHINE_PROFILE=<p> python missing_cells.py --condition FAB --artifact-tag C128V42A [--total-time-seconds 2.0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_collection import validate_collection, product_paths  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=("FAB", "INLB"))
    ap.add_argument("--artifact-tag", required=True)
    ap.add_argument("--total-time-seconds", type=float, default=2.0)
    args = ap.parse_args(argv)
    product, prefix = product_paths(args.condition, args.artifact_tag, args.total_time_seconds)
    z = np.load(product, allow_pickle=True)
    cells = sorted(set(int(c) for c in z["cell"]))
    covered = set()
    for npz in sorted(prefix.parent.glob(f"{prefix.name}_Window_Posterior_Draws*/*.npz")):
        ok, detail = validate_collection(npz, product)
        print(f"{npz.parent.name}: {'valid' if ok else 'INVALID'} ({detail})", file=sys.stderr)
        if ok:
            covered.update(int(c) for c in np.load(npz, allow_pickle=True)["cell"])
    missing = [c for c in cells if c not in covered]
    print(f"product cells {len(cells)}; covered {sorted(covered)}; missing {len(missing)}", file=sys.stderr)
    print(",".join(str(c) for c in missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
