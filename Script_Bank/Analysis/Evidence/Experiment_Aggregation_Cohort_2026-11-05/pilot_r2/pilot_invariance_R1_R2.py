"""Numerical invariance of the accepted pilot review copies (PILOT2CELLSR1) against their regeneration under the
presentation changes (PILOT2CELLSR2): every scientific array of aggregation_candidates.npz (vectors, quantiles,
identities, members, methods, outside-prior masses), the group and comparison CSV exports, and the manifest
except its presentation/provenance/timing fields must be identical. Usage: python pilot_invariance_R1_R2.py"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

P = Path("/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit")
IGNORE_MANIFEST = {"written_at", "timing_seconds", "aggregation_seconds", "archive_sha256", "exports", "coverage_note", "source_clouds_note"}
bad = 0
for cond, tag in (("FAB", "C128V42A"), ("INLB", "C128V42AP3LR32")):
    base = f"SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{cond}_2S_50FPS_{tag}_MAP_Experiment"
    a, b = P / f"{base}_PILOT2CELLSR1", P / f"{base}_PILOT2CELLSR2"
    za, zb = np.load(a / "aggregation_candidates.npz", allow_pickle=True), np.load(b / "aggregation_candidates.npz", allow_pickle=True)
    keys_a, keys_b = sorted(za.files), sorted(zb.files)
    print(f"{cond}: archive keys {'equal' if keys_a == keys_b else 'DIFFER'} ({len(keys_a)})")
    for k in keys_a:
        if k in ("manifest_json",):
            continue
        x, y = za[k], zb[k]
        same = x.shape == y.shape and (np.array_equal(x, y) if x.dtype.kind in "iufb" else np.array_equal(x.astype(str), y.astype(str)))
        print(f"  {k:28s} {str(x.shape):18s} {'equal' if same else 'DIFFERENT'}")
        bad += (not same)
    ma, mb = json.loads(str(za["manifest_json"])), json.loads(str(zb["manifest_json"]))
    for k in sorted(set(ma) | set(mb)):
        if k in IGNORE_MANIFEST:
            continue
        if json.dumps(ma.get(k), sort_keys=True, default=str) != json.dumps(mb.get(k), sort_keys=True, default=str):
            print(f"  manifest[{k}] DIFFERS"); bad += 1
    for name in ("aggregation_groups.csv", "aggregation_comparison.csv"):
        ra, rb = list(csv.reader(open(a / name))), list(csv.reader(open(b / name)))
        same = ra == rb
        print(f"  {name:28s} {len(ra)} rows {'equal' if same else 'DIFFERENT'}")
        if not same:
            diff = [(i, x, y) for i, (x, y) in enumerate(zip(ra, rb)) if x != y][:3]
            print("   first differences:", diff); bad += 1
print("INVARIANCE", "PASS" if bad == 0 else f"FAIL ({bad} differences)")
sys.exit(1 if bad else 0)
