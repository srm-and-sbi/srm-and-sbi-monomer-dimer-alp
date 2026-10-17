"""The detector DLI stage's dry-run banner under the probe-class rule and under the --occupancy override.

The banner's initial-fluorescence lines once divided by the two-probe share, which the override
leaves undefined (a TypeError before any rendering, found in review on 2026-10-13). Both branches
must print and exit 0. Run directly (MACHINE_PROFILE set, as every test):

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_dli_banner.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "Script_Bank" / "Prime" / "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Simulation_DLI.py"
BASE = ["--condition", "FAB", "--total-time-seconds", "2.0", "--split", "train", "--task-id", "0",
        "--task-simulations", "2", "--seed", "None", "--dry-run"]


def _run(extra):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, str(ENTRY), *BASE, *extra], capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_banner_under_the_probe_class_rule():
    text = _run([])
    assert "initial fluorescence probabilities (before bleaching and detection" in text
    assert "simulated dimer 0.807" in text and "per represented receptor within a dimer: 0.4060" in text
    assert "both-probes-fluorescent share among fluorescent two-probe dimers: 0.675" in text
    assert "two-dye share among fluorescent two-probe dimers" not in text      # the old label
    print("PASS: DLI banner, probe-class rule")


def test_banner_under_the_occupancy_override():
    text = _run(["--occupancy", "0.5"])
    assert "per receptor (coins)  : monomer 0.4030, dimer receptor 0.4030" in text
    assert "per complex           : monomer 0.403, dimer 0.644" in text
    assert "[DRY RUN]" in text
    print("PASS: DLI banner, occupancy override")


def test_banner_under_inlb_reports_b1_and_b2_apart():
    text = _run(["--condition", "INLB"])
    assert "every represented InlB receptor carries a probe; B1 represents one receptor, B2 two" in text
    assert "probe-free partner" not in text
    # the starting B2 share carries the window-start allocation and is recorded per simulation; the
    # banner no longer presents the declared two-probe share as the starting share
    assert "the starting B2 share depends on each simulation's window-start allocation" in text
    assert "averaged over its classes" not in text
    assert "12 RDS labels of INLB" in text
    text = _run(["--condition", "INLB", "--occupancy", "0.5"])
    assert "per complex           : monomer 0.250, B1 0.250 (one represented receptor), B2 0.438 (two independent coins)" in text
    print("PASS: DLI banner, InlB B1 and B2 apart")


if __name__ == "__main__":
    test_banner_under_the_probe_class_rule()
    test_banner_under_the_occupancy_override()
    test_banner_under_inlb_reports_b1_and_b2_apart()
