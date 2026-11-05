"""Validate retained draw collections against their source Experiment product (2026-11-05).

A collection ``<prefix>_Window_Posterior_Draws[_<LABEL>]/<same>.npz`` is valid when: it loads; ``posterior_samples_cloud``
is (W, n_draws, D) float32 and finite; its (cell, chunk) identities are unique, every chunk of the declared geometry is present
for every cell it carries, and every identity is a window of the product; its manifest's checkpoint, estimator, draw count,
pool mode, parameter keys and window geometry equal the product's; and every ``fresh_sgm`` row equals the retained draw its
``fresh_sgm_index`` names. ``--union`` additionally requires that the collections given together cover every product cell
exactly once (sixty cells, six hundred unique windows for the 2 s products).

Usage: MACHINE_PROFILE=<p> python validate_collection.py --condition FAB --artifact-tag C128V42A [--total-time-seconds 2.0]
           [--union] <collection npz> [<collection npz> ...]
       (with no npz given, every collection beside the product is checked).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def product_paths(condition, tag, total_time_seconds):
    """(product npz path, the product prefix path ``<posit>/<alias>_<COND>_<timing>_<TAG>``) under the machine profile."""
    sys.path.insert(0, os.getcwd())
    from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming
    from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow
    paths = detector_workflow().paths.with_condition(condition)
    timing = RunTiming(total_time_seconds=float(total_time_seconds), frames=PARAMETERS.simulation.timing)
    product_label = paths.product_label(timing.label, tag)
    root = Path(PARAMETERS.machine.data_bank_root)
    exp_dir = paths.experiment_recovery_dir(root, product_label)
    prefix = root / paths.posit_subdir / f"{paths.project_alias}_{product_label}"
    return exp_dir / (exp_dir.name + ".npz"), prefix


def validate_collection(npz_path, product_path):
    """(ok, detail) for one collection against the product."""
    try:
        c = np.load(npz_path, allow_pickle=True)
        p = np.load(product_path, allow_pickle=True)
        man = json.loads(str(c["collection_manifest_json"]))
        pman = json.loads(str(p["manifest_json"]))
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {exc}"
    cloud = c["posterior_samples_cloud"]
    problems = []
    if cloud.dtype != np.float32 or cloud.ndim != 3:
        problems.append(f"cloud dtype/ndim {cloud.dtype}/{cloud.ndim}")
    if not np.isfinite(cloud).all():
        problems.append("non-finite draws")
    geom = pman["window_geometry"]
    n_pos = (int(geom["span_frames"]) - int(geom["n_frames"])) // int(geom["step_frames"]) + 1
    cells = c["cell"].astype(int); chunks = c["chunk"].astype(int)
    ids = list(zip(cells.tolist(), chunks.tolist()))
    if len(set(ids)) != len(ids):
        problems.append("duplicate (cell, chunk)")
    for cell in sorted(set(cells.tolist())):
        have = sorted(chunks[cells == cell].tolist())
        if have != list(range(n_pos)):
            problems.append(f"cell {cell} chunks {have} != {list(range(n_pos))}")
    pids = set(zip(p["cell"].astype(int).tolist(), p["chunk"].astype(int).tolist()))
    foreign = [i for i in ids if i not in pids]
    if foreign:
        problems.append(f"identities not in the product: {foreign[:5]}")
    if cloud.shape[0] != len(ids) or cloud.shape[1] != int(pman["n_summary_draws"]) or cloud.shape[2] != len(pman["parameter_keys"]):
        problems.append(f"cloud shape {cloud.shape} != ({len(ids)}, {pman['n_summary_draws']}, {len(pman['parameter_keys'])})")
    for k in ("checkpoint_sha256", "n_summary_draws", "pool_mode", "parameter_keys", "window_geometry", "coordinate_transform"):
        if json.dumps(man.get(k), sort_keys=True) != json.dumps(pman.get(k), sort_keys=True):
            problems.append(f"{k}: {man.get(k)!r} != product {pman.get(k)!r}")
    if not np.allclose(man["sgm_scale"], pman["sgm_scale"]):
        problems.append("sgm_scale differs")
    if man.get("source_product", {}).get("checkpoint_sha256") != pman["checkpoint_sha256"]:
        problems.append("source_product checkpoint differs")
    fsgm, fidx = c["fresh_sgm"], c["fresh_sgm_index"].astype(int)
    picked = cloud[np.arange(cloud.shape[0]), fidx].astype(np.float64)
    if not np.array_equal(picked, fsgm.astype(np.float64)) and not np.allclose(picked, fsgm, rtol=0, atol=1e-6):
        problems.append("fresh_sgm rows are not the retained draws their indices name")
    label = man.get("collection_label")
    expected_label = npz_path.parent.name.split("_Window_Posterior_Draws", 1)[1].lstrip("_") or None
    if (label or None) != expected_label:
        problems.append(f"collection_label {label!r} != folder label {expected_label!r}")
    detail = (f"cells {sorted(set(cells.tolist()))}, {cloud.shape[0]} windows x {cloud.shape[1]} draws, seed "
              f"{man.get('seed_policy', {}).get('base_seed')}, device {man.get('execution', {}).get('device')}")
    return (not problems), (detail if not problems else detail + "; " + "; ".join(problems))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=("FAB", "INLB"))
    ap.add_argument("--artifact-tag", required=True)
    ap.add_argument("--total-time-seconds", type=float, default=2.0)
    ap.add_argument("--union", action="store_true", help="the given collections together must cover every product cell exactly once")
    ap.add_argument("npz", nargs="*")
    args = ap.parse_args(argv)
    product, prefix = product_paths(args.condition, args.artifact_tag, args.total_time_seconds)
    paths = [Path(x) for x in args.npz] or sorted(prefix.parent.glob(f"{prefix.name}_Window_Posterior_Draws*/*.npz"))
    bad = 0; union = []
    for npz in paths:
        ok, detail = validate_collection(npz, product)
        print(f"{'PASS' if ok else 'FAIL'} {npz.parent.name}: {detail}")
        bad += (not ok)
        if ok:
            c = np.load(npz, allow_pickle=True)
            union.extend(zip(c["cell"].astype(int).tolist(), c["chunk"].astype(int).tolist()))
    if args.union:
        p = np.load(product, allow_pickle=True)
        pids = sorted(zip(p["cell"].astype(int).tolist(), p["chunk"].astype(int).tolist()))
        dup = len(union) - len(set(union)); missing = sorted(set(pids) - set(union))
        ok = bad == 0 and dup == 0 and not missing and sorted(union) == pids
        print(f"{'PASS' if ok else 'FAIL'} union: {len(set(union))} unique windows of {len(pids)}; "
              f"{len(set(c for c, _ in union))} cells of {len(set(c for c, _ in pids))}; duplicates {dup}; missing {len(missing)}")
        bad += (not ok)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
