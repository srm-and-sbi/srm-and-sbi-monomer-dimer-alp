"""P7 of the prior-realization audit: campaign membership against a frozen specification.

The schema guard compares keys, bounds, scales and condition only (``io.schema_mismatch``), so a
detector dataset of an earlier generator with the same imaging box passes it. P7 is the inventory
that establishes membership instead (DETECTOR_WORKFLOW.md section 9.10). Run directly:

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_prior_realization_audit_campaign.py
"""
from __future__ import annotations

import datetime
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det          # noqa: E402
from srm_and_sbi_monomer_dimer_alp import labeling as lab                            # noqa: E402
from srm_and_sbi_monomer_dimer_alp import parameterization as par                    # noqa: E402
from srm_and_sbi_monomer_dimer_alp.io import theta_set_schema, write_theta_set       # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS                # noqa: E402


def _audit():
    path = REPO / "Script_Bank" / "Analysis" / "SRM_AND_SBI_MONOMER_DIMER_ALP_Prior_Realization_Audit.py"
    spec = importlib.util.spec_from_file_location("prior_realization_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _audit()
CONDITION, TIMING, SPLIT, VERSION = "FAB", "2S_50FPS", "TRAIN", "9.9.9"
WINDOW = A.parse_campaign_window("2026-10-14T00:00:00+00:00..2026-10-21T00:00:00+00:00")
INSIDE, OUTSIDE = "2026-10-15T12:00:00+00:00", "2026-09-16T14:48:05+00:00"
PLAN = lab.resolve_labeling(CONDITION)
COL = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}


def schema(generator: str, *, version: str = VERSION, written: str = INSIDE, condition: str = CONDITION) -> dict:
    # a tier record of a condition carries that condition's table (eleven rows FAB, twelve INLB)
    table = par.parameterization_of(condition) if generator == "rds" else det.DETECTOR_PARAMETERIZATION
    sc = theta_set_schema(table, condition=condition, timing_label=TIMING, generator=generator)
    sc["package_version"] = version
    sc["written_utc"] = written
    return sc


def labeling_rows(n: int, *, share=None, occupancy=None) -> np.ndarray:
    L = np.zeros((n, len(lab.LABELING_SET_COLUMNS)))
    L[:, COL["occupancy_monomer"]] = PLAN.occupancy_pair[0] if occupancy is None else occupancy[0]
    L[:, COL["occupancy_dimer"]] = PLAN.occupancy_pair[1] if occupancy is None else occupancy[1]
    L[:, COL["two_probe_share"]] = PLAN.two_probe_share if share is None else share
    return L


def entry(task: int, n: int = 10, **kw) -> dict:
    e = dict(task=task, rds_schema=schema("rds"), rds_rows=n, det_schema=schema("dli_detector"), det_rows=n,
             labeling=labeling_rows(n))
    e.update(kw)
    return e


def run(provenance, workflow="detector", expected=3, sims=10, window=WINDOW, version=VERSION):
    return A.p7_campaign(provenance, condition=CONDITION, timing_label=TIMING, workflow=workflow, version=version,
                         expected_tasks=expected, task_simulations=sims, window=window)


def test_p7_passes_and_fails_for_the_right_reasons():
    ok = run([entry(0), entry(1), entry(2)])
    assert ok["ok"] and not ok["missing_tasks"] and not ok["failures"], ok

    missing = run([entry(0), entry(2)])
    assert not missing["ok"] and missing["missing_tasks"] == [1], missing

    beyond = run([entry(0), entry(1), entry(2), entry(7)])
    assert beyond["ok"] and beyond["unexpected_tasks"] == [7], beyond     # reported, never selected for training

    old = run([entry(0), entry(1), entry(2, det_schema=schema("dli_detector", version="0.1.6", written=OUTSIDE))])
    reasons = " ".join(old["failures"][2])
    assert not old["ok"] and "package_version '0.1.6'" in reasons and "outside the campaign window" in reasons, old

    rows = run([entry(0), entry(1), entry(2, rds_rows=9, labeling=labeling_rows(9))])
    assert not rows["ok"] and "9 rows != 10" in " ".join(rows["failures"][2]), rows

    share = run([entry(0), entry(1), entry(2, labeling=labeling_rows(10, share=0.5))])
    assert not share["ok"] and "two_probe_share" in " ".join(share["failures"][2]), share

    coins = run([entry(0), entry(1), entry(2, labeling=labeling_rows(10, occupancy=(0.5, 0.5)))])
    assert not coins["ok"] and "declared rule" in " ".join(coins["failures"][2]), coins

    absent = run([entry(0), entry(1), entry(2, labeling=None)])
    assert not absent["ok"] and absent["failures"][2] == ["Labeling_Set: absent"], absent

    wrong_gen = run([entry(0, rds_schema=schema("dli_detector")), entry(1), entry(2)])
    assert not wrong_gen["ok"] and "generator 'dli_detector' != 'rds'" in " ".join(wrong_gen["failures"][0]), wrong_gen

    other = run([entry(0, rds_schema=schema("rds", condition="INLB")), entry(1), entry(2)])
    assert not other["ok"] and "condition 'INLB'" in " ".join(other["failures"][0]), other

    biology = run([entry(0, det_schema=None, det_rows=None), entry(1), entry(2)], workflow="biology")
    assert biology["ok"], biology                       # the detector's own set is not read by the biology workflow

    no_window = run([entry(0, rds_schema=schema("rds", written=OUTSIDE)), entry(1), entry(2)], window=(None, None))
    assert no_window["ok"], no_window                   # no window given: the written time is not judged

    unbounded = run([entry(0), entry(1), entry(2)], expected=None, sims=None)
    assert unbounded["ok"] and unbounded["missing_tasks"] == [] and unbounded["unexpected_tasks"] == [], unbounded
    print("PASS: p7 verdicts and reasons")


def test_window_parsing_refuses_naive_or_reversed_bounds():
    for bad in ("2026-10-14T00:00:00..2026-10-21T00:00:00+00:00",
                "2026-10-21T00:00:00+00:00..2026-10-14T00:00:00+00:00"):
        try:
            A.parse_campaign_window(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    assert A.parse_campaign_window(None) == (None, None)
    assert A._in_window(INSIDE, WINDOW) is True and A._in_window(OUTSIDE, WINDOW) is False
    assert A._in_window(None, WINDOW) is False and A._in_window(INSIDE, (None, None)) is None
    assert A._in_window("2026-10-15T12:00:00", WINDOW) is True      # a naive time is read as UTC
    print("PASS: campaign window parsing")


def test_collect_provenance_reads_real_products():
    base = PARAMETERS.paths.with_condition(CONDITION)
    paths = det.detector_paths(PARAMETERS.paths).with_condition(CONDITION)
    n = 4
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        theta_by_task = {}
        table = par.parameterization_of(CONDITION)
        for t in (0, 1):
            rds_path = base.theta_set_path(t, root, TIMING, True, SPLIT)
            os.makedirs(os.path.dirname(rds_path), exist_ok=True)
            low, high = np.array(par.theta_lower_bound(CONDITION)), np.array(par.theta_upper_bound(CONDITION))
            theta = par.to_physical(np.random.default_rng(t).uniform(low, high, size=(n, len(low))), table)
            sc = theta_set_schema(table, condition=CONDITION, timing_label=TIMING, generator="rds")
            if t == 1:
                sc["package_version"] = "0.1.6"      # an earlier generator's tier under the current table
            write_theta_set(rds_path, theta, sc)
            theta_by_task[t] = theta
            det_path = paths.theta_set_path(t, root, TIMING, True, SPLIT)
            os.makedirs(os.path.dirname(det_path), exist_ok=True)
            dlow, dhigh = np.array(det.theta_lower_bound()), np.array(det.theta_upper_bound())
            write_theta_set(det_path, np.power(10, np.random.default_rng(10 + t).uniform(dlow, dhigh, size=(n, len(dlow)))),
                            theta_set_schema(det.DETECTOR_PARAMETERIZATION, condition=CONDITION, timing_label=TIMING,
                                             generator="dli_detector"))
            lp = paths.record_set_path("Labeling_Set", t, root, TIMING, True, SPLIT)
            rows = labeling_rows(n)
            store = zarr.open(store=str(lp), mode="w", shape=rows.shape, chunks=(1, rows.shape[1]), dtype=np.float64)
            store[:, :] = rows
        prov = A.collect_provenance(base, paths, root, TIMING, SPLIT, [0, 1], "detector", theta_by_task)
        assert [e["task"] for e in prov] == [0, 1]
        assert prov[0]["rds_schema"]["generator"] == "rds" and prov[0]["det_schema"]["generator"] == "dli_detector"
        assert prov[0]["rds_rows"] == n and prov[0]["det_rows"] == n and prov[0]["labeling"].shape == (n, len(lab.LABELING_SET_COLUMNS))
        current = prov[0]["rds_schema"]["package_version"]
        now = datetime.datetime.now(datetime.timezone.utc)
        window = (now - datetime.timedelta(hours=1), now + datetime.timedelta(hours=1))
        res = A.p7_campaign(prov, condition=CONDITION, timing_label=TIMING, workflow="detector", version=current,
                            expected_tasks=2, task_simulations=n, window=window)
        assert not res["ok"] and list(res["failures"]) == [1], res
        assert "package_version '0.1.6'" in " ".join(res["failures"][1]), res
        res0 = A.p7_campaign(prov[:1], condition=CONDITION, timing_label=TIMING, workflow="detector", version=current,
                             expected_tasks=1, task_simulations=n, window=window)
        assert res0["ok"], res0
    print("PASS: collect_provenance on written products")


if __name__ == "__main__":
    test_p7_passes_and_fails_for_the_right_reasons()
    test_window_parsing_refuses_naive_or_reversed_bounds()
    test_collect_provenance_reads_real_products()
