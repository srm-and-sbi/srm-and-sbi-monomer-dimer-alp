"""Catalog mapping check (2026-11-05): every working recording the adopted Experiment products ran on (staged on
JUPITER beside the sixty-cell catalog `Catalog_Note_Experiment.tsv`) is byte-identical to the raw movie the
catalog names in the PC's read-only complete MET movie bank, and every recording staged on the PC equals the same
source. Inputs: the three SHA-256 lists beside this script (hashed on JUPITER and on the PC) and the catalog TSV.
Run: python verify_catalog_mapping.py  (reads the files in its own directory; computes nothing else)."""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(p):
    out = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            h, f = line.split(None, 1)
            out[f.strip().lstrip("*")] = h
    return out


jup = load(HERE / "jupiter_staged_sha256.txt")
bank = load(HERE / "pc_bank_sha256.txt")
staged = load(HERE / "pc_staged_sha256.txt")
rows = list(csv.DictReader(open(HERE / "Catalog_Note_Experiment.tsv"), delimiter="\t"))
ok = bad = staged_ok = staged_bad = 0
for r in rows:
    w, m = r["working_filename"], r["source_member"]
    if jup.get(w) is not None and jup.get(w) == bank.get(m):
        ok += 1
    else:
        bad += 1
        print("MISMATCH", w, m, jup.get(w), bank.get(m))
    if w in staged:
        if staged[w] == bank.get(m):
            staged_ok += 1
        else:
            staged_bad += 1
            print("STAGED MISMATCH", w)
print(f"catalog rows {len(rows)}; JUPITER staged == PC bank under the catalog: {ok} ok, {bad} mismatches")
print(f"PC staged files == bank: {staged_ok} ok, {staged_bad} mismatches (of {len(staged)})")
print(f"distinct bank hashes {len(set(bank.values()))} of {len(bank)}")
sys.exit(0 if bad == 0 and staged_bad == 0 and len(rows) == 120 else 1)
