"""The condition-aware parameter resolver: one master table, a per-condition view (eleven
learnable rows under MET-FAB, twelve under MET-INLB), alignment by key, the per-condition
Theta_Set schema and its refusals, and the stage resolvers of both workflows.

Run directly from the repository root::

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/test_condition_resolver.py
"""
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("MACHINE_PROFILE", "mars_pc")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srm_and_sbi_monomer_dimer_alp import __version__                              # noqa: E402
from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det          # noqa: E402
from srm_and_sbi_monomer_dimer_alp import io as pio                                  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import parameterization as par                   # noqa: E402
from srm_and_sbi_monomer_dimer_alp import workflow as wf                             # noqa: E402

CONDITIONS = ("FAB", "INLB")
F_KEY, R_KEY, N_KEY = "fraction_basal_dimers_reference", "relative_count_active_dimers_initial", "count_total"
MASTER_KEYS = [N_KEY, F_KEY, "rate_dissociation", "diffusivity_alp", "relative_diffusivity_dimer",
               "relative_diffusivity_slow", "relative_diffusivity_immobile", "rate_fast_slow", "rate_slow_fast",
               "rate_slow_immobile", "rate_immobile_slow", R_KEY]
LOG_BOXES = {N_KEY: (3.0, 5.0), "rate_dissociation": (-3.0, 1.0), "diffusivity_alp": (-1.25, -0.25),
             "relative_diffusivity_dimer": (-1.0, 0.0), "relative_diffusivity_slow": (-1.0, 0.0),
             "relative_diffusivity_immobile": (-3.0, -2.0), "rate_fast_slow": (-1.0, 1.0), "rate_slow_fast": (-1.0, 1.0),
             "rate_slow_immobile": (-1.0, 1.0), "rate_immobile_slow": (-1.0, 1.0)}


def _expect(exc_type, fn, *needles):
    try:
        fn()
    except exc_type as exc:
        for needle in needles:
            assert needle in str(exc), (needle, str(exc))
        return exc
    raise AssertionError(f"expected {exc_type.__name__} with {needles}")


# ---- phase 1: the master table and its per-condition views ---------------------------------

def test_master_table_and_the_two_views():
    assert list(par.MASTER_PARAMETER_KEYS) == MASTER_KEYS
    assert [e["KEY"] for e in par.MASTER_PARAMETERIZATION] == MASTER_KEYS
    for e in par.MASTER_PARAMETERIZATION:
        if e["KEY"] in (F_KEY, R_KEY):
            assert e["LOG_FLAG"] is False and tuple(e["PRIOR_RANGE"]) == (0.0, 1.0) and e["UNIT"] == "Fraction"
            assert e["VALUE"] == 0.5 and not par.is_log_row(e)
        else:
            assert e["LOG_FLAG"] is True and tuple(map(float, e["PRIOR_RANGE"])) == LOG_BOXES[e["KEY"]], e["KEY"]
    fab, inlb = par.parameterization_of("FAB"), par.parameterization_of("INLB")
    assert [e["KEY"] for e in fab] == MASTER_KEYS[:11] and [e["KEY"] for e in inlb] == MASTER_KEYS
    assert par.parameter_keys_of("FAB") == MASTER_KEYS[:11] and par.parameter_keys_of("INLB") == MASTER_KEYS
    assert par.parameter_index_of("INLB")[R_KEY] == 11 and par.parameter_index_of("FAB")[F_KEY] == 1
    assert par.theta_lower_bound("FAB") == [e["PRIOR_RANGE"][0] for e in fab]
    assert par.theta_upper_bound("INLB")[-1] == 1.0 and len(par.theta_upper_bound("INLB")) == 12
    assert par.build_prior("FAB").event_shape == (11,) and par.build_prior("INLB").event_shape == (12,)
    assert par.parameter_find(F_KEY, "FAB") == 1 and par.parameter_find(R_KEY, "INLB") == 11
    _expect(KeyError, lambda: par.parameter_find(R_KEY, "FAB"), "declared constant")
    _expect(KeyError, lambda: par.parameter_find("nope", "INLB"), "not a learnable parameter")
    _expect(KeyError, lambda: par.parameterization_of("XYZ"))


def test_resolving_never_mutates_the_master():
    before = copy.deepcopy(par.MASTER_PARAMETERIZATION)
    before_raw = copy.deepcopy(par.PARAMETERIZATION_RAW)
    widths = [len(par.parameterization_of(c)) for c in ("FAB", "INLB", "FAB")]
    assert widths == [11, 12, 11]
    for c in ("FAB", "INLB", "FAB"):
        rows = par.parameterization_raw_of(c)
        rows[0]["VALUE"] = -1.0                      # a caller editing its copy
        par.parameterization_of(c)[0]["PRIOR_RANGE"] = (0.0, 0.0)
    assert copy.deepcopy(par.MASTER_PARAMETERIZATION) == before
    assert copy.deepcopy(par.PARAMETERIZATION_RAW) == before_raw
    assert all(a is not b for a, b in zip(par.parameterization_of("INLB"), par.MASTER_PARAMETERIZATION))


def test_declared_rows_and_the_raw_view():
    assert par.declared_row_values("FAB") == {R_KEY: 0.0} and par.declared_row_values("INLB") == {}
    raw_fab = par.parameterization_raw_of("FAB")
    row = next(e for e in raw_fab if e["KEY"] == R_KEY)
    assert row["VALUE"] == 0.0 and row["PRIOR_RANGE"] is None and row["NOTE"] == "Declared Constant"
    assert par.role_of(row) == "fixed"
    assert [e["KEY"] for e in raw_fab if par.role_of(e) == "learnable"] == MASTER_KEYS[:11]
    raw_inlb = par.parameterization_raw_of("INLB")
    assert [e["KEY"] for e in raw_inlb if par.role_of(e) == "learnable"] == MASTER_KEYS
    assert len(raw_fab) == len(raw_inlb) == len(par.PARAMETERIZATION_RAW)
    # the condition-setting rules
    CS = par.ConditionSetting
    _expect(ValueError, lambda: CS("FAB", 0.0, fluorescence_ratio=0.5, fluorescence_ratio_to="INLB"),
            "requires ligand classes")                                        # inferred without ligand classes
    _expect(ValueError, lambda: CS("FAB", 0.0, fluorescence_ratio=0.5, fluorescence_ratio_to="INLB",
                                   relative_count_active_dimers_initial=0.3), "requires ligand classes")
    _expect(ValueError, lambda: CS("INLB", 1.0, ligand_classes=True, dissociation_constant_nm=5.0,
                                   relative_count_active_dimers_initial=1.2), "[0, 1]")
    variant = CS("INLB", 1.0, ligand_classes=True, dissociation_constant_nm=5.0, relative_count_active_dimers_initial=0.3)
    assert variant.declared_rows == {R_KEY: 0.3}
    rds_variant = par.SimulationRDS(conditions=(par.CONDITION_SETTINGS[0], variant))
    assert rds_variant.declared_row_values("INLB") == {R_KEY: 0.3} and rds_variant.declared_row_values("FAB") == {R_KEY: 0.0}
    assert par.PARAMETERS.simulation.rds.condition_setting("INLB").declared_rows == {}


def test_removed_symbols_are_absent():
    for name in ("PARAMETERIZATION", "PARAMETER_KEYS", "PARAMETER_FIND", "COMPOSITION_BAND", "COMPOSITION_RATIO_BOX",
                 "ratio_to_receptor_fraction", "ratio_to_complex_fraction", "receptor_fraction_to_ratio",
                 "retained_ratio_of"):
        assert not hasattr(par, name), name
    assert not hasattr(par.PARAMETERS.simulation.rds, "retained_ratio_of")
    assert not hasattr(par.PARAMETERS.simulation.rds.stoichiometry, "composition_ratio_key")
    assert par.PARAMETERS.simulation.rds.stoichiometry.parameter_keys == (N_KEY, F_KEY, "rate_dissociation", R_KEY)


def test_conversions_require_the_table_and_round_trip_mixed_scales():
    rng = np.random.default_rng(5)
    for cond in CONDITIONS:
        table = par.parameterization_of(cond)
        flow = rng.uniform(par.theta_lower_bound(cond), par.theta_upper_bound(cond), size=(200, len(table)))
        phys = par.to_physical(flow, table)
        back = par.to_flow(phys, table)
        for j, e in enumerate(table):
            if par.is_log_row(e):
                assert np.allclose(back[:, j], flow[:, j])
            else:
                assert np.array_equal(back[:, j], flow[:, j]) and np.array_equal(phys[:, j], flow[:, j]), "linear columns pass through bitwise"
        flags = par.fraction_flags(phys, table)
        assert set(flags) == ({F_KEY} if cond == "FAB" else {F_KEY, R_KEY})
        assert not any(v.any() for v in flags.values())
        bad = phys.copy(); bad[0, 1] = 1.02; bad[1, 1] = -0.01; bad[2, 1] = np.nan
        assert par.fraction_flags(bad, table)[F_KEY].tolist()[:4] == [True, True, True, False]
        assert bad[0, 1] == 1.02, "flagging never modifies the values"
    _expect(ValueError, lambda: par.to_physical(np.zeros(11), None), "table is required")
    _expect(ValueError, lambda: par.to_flow(np.ones(12), None), "table is required")
    _expect(ValueError, lambda: par.to_physical(np.zeros(12), par.parameterization_of("FAB")), "last axis has 12")


def test_workflow_resolvers_for_both_workflows():
    b, d = wf.biology_workflow(), wf.detector_workflow()
    assert wf.parameter_keys(b, "FAB") == MASTER_KEYS[:11] and wf.parameter_keys(b, "INLB") == MASTER_KEYS
    assert wf.parameter_keys(d, "FAB") == wf.parameter_keys(d, "INLB") == list(det.DETECTOR_PARAMETER_KEYS)
    assert [e["KEY"] for e in wf.parameter_table(b, "INLB")] == MASTER_KEYS
    assert wf.parameter_table(d, "FAB") == list(det.DETECTOR_PARAMETERIZATION)
    assert [e["KEY"] for e in wf.parameter_table_raw(b, "FAB") if par.role_of(e) == "learnable"] == MASTER_KEYS[:11]
    assert wf.parameter_table_raw(d, "INLB") == list(det.DETECTOR_PARAMETERIZATION_RAW)
    lo, hi = wf.prior_bounds(b, "INLB")
    assert len(lo) == len(hi) == 12 and lo[-1] == 0.0 and hi[-1] == 1.0
    assert wf.prior_bounds(d, "FAB") == (list(det.theta_lower_bound()), list(det.theta_upper_bound()))
    assert wf.prior_builder(b, "FAB")().event_shape == (11,) and wf.prior_builder(b, "INLB")().event_shape == (12,)
    assert wf.prior_builder(d, "INLB")().event_shape == (6,)
    for cfg in (b, d):
        _expect(ValueError, lambda: wf.parameter_keys(cfg, None), "condition is required")
        _expect(ValueError, lambda: wf.parameter_table(cfg, "ALP"), "unknown condition")
        _expect(ValueError, lambda: wf.prior_bounds(cfg, "fab"), "unknown condition")


def test_detector_mirror_and_version_strings():
    keys = [e["KEY"] for e in det.DETECTOR_NUISANCE]
    assert keys == MASTER_KEYS, keys
    assert all(det.role_of(e) == "nuisance_object" for e in det.DETECTOR_NUISANCE)
    assert det.DETECTOR_PARAMETER_KEYS == ["mu_r", "sigma_r", "mu_pc", "sigma_pc", "prob_photo_bleach", "lambda_rate"]
    pyproject = (REPO / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == __version__
    schema = pio.theta_set_schema(par.parameterization_of("FAB"), condition="FAB", timing_label="2S_50FPS", generator="rds")
    assert schema["package_version"] == __version__ and schema["condition"] == "FAB"
    _expect(ValueError, lambda: pio.theta_set_schema(par.parameterization_of("FAB"), condition=None,
                                                     timing_label="2S_50FPS", generator="rds"), "condition is required")


def test_theta_set_round_trips_and_refusals():
    P = par.PARAMETERS.paths
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for cond in CONDITIONS:
            table = par.parameterization_of(cond)
            paths = P.with_condition(cond)
            n = len(table)
            theta = np.tile(np.array([par.prior_center(e) for e in table]), (4, 1))
            theta[0, 1], theta[1, 1] = 0.0, 1.0
            if cond == "INLB":
                theta[2, 11], theta[3, 11] = 0.0, 1.0
            schema = pio.theta_set_schema(table, condition=cond, timing_label="2S_50FPS", generator="rds")
            for compress in (True, False):
                path = paths.theta_set_path(0, root, "2S_50FPS", compress, "TRAIN")
                pio.write_theta_set(path, theta, schema)
                assert pio.theta_set_status(path, table, condition=cond) == "OK"
                loaded = np.asarray(pio.load_theta_set(path, table, condition=cond))
                assert loaded.shape == (4, n) and np.array_equal(loaded, theta)
                other = "INLB" if cond == "FAB" else "FAB"
                status = pio.theta_set_status(path, par.parameterization_of(other), condition=other)
                assert status.startswith("SCHEMA MISMATCH: parameter keys differ"), status
                _expect(pio.ThetaSetSchemaError, lambda: pio.load_theta_set(path, table, condition=other), "condition differs")
            # a stored None condition is refused; a 0.1.39-shaped schema names the old key and version
            none_schema = dict(schema, condition=None)
            reason = pio.schema_mismatch(none_schema, table, condition=cond)
            assert reason is not None and "condition differs" in reason and "None" in reason
            old = dict(schema, package_version="0.1.39", condition=cond)
            old["parameter_keys"] = ["count_total", "ratio_dimer_monomer_initial"] + schema["parameter_keys"][2:]
            old["prior_low"], old["prior_high"] = list(schema["prior_low"]), list(schema["prior_high"])
            old["prior_low"][1], old["prior_high"][1] = -1.27875, -0.47712
            old["log_flag"] = list(schema["log_flag"]); old["log_flag"][1] = True
            path = paths.theta_set_path(7, root, "2S_50FPS", True, "TRAIN")
            pio.write_theta_set(path, theta if cond == "INLB" else theta, dict(old, parameter_keys=old["parameter_keys"][:n]))
            status = pio.theta_set_status(path, table, condition=cond)
            assert "ratio_dimer_monomer_initial" in status and "0.1.39" in status, status
            _expect(ValueError, lambda: pio.schema_mismatch(schema, table, condition=None), "condition is required")
        # a six-row detector store stamped 0.1.39 keeps loading (the detector datasets of that release)
        dpaths = det.detector_paths(P).with_condition("FAB")
        dschema = pio.theta_set_schema(det.DETECTOR_PARAMETERIZATION, condition="FAB", timing_label="2S_50FPS",
                                       generator="dli_detector")
        dschema["package_version"] = "0.1.39"
        dpath = dpaths.theta_set_path(0, root, "2S_50FPS", True, "TRAIN")
        pio.write_theta_set(dpath, np.zeros((3, 6)), dschema)
        assert pio.theta_set_status(dpath, det.DETECTOR_PARAMETERIZATION, condition="FAB") == "OK"
        assert np.asarray(pio.load_theta_set(dpath, det.DETECTOR_PARAMETERIZATION, condition="FAB")).shape == (3, 6)
        assert dpath.name != P.with_condition("FAB").theta_set_path(0, root, "2S_50FPS", True, "TRAIN").name
        assert dpaths.rds_theta_set_path(0, root, "2S_50FPS", True, "TRAIN") == \
            P.with_condition("FAB").theta_set_path(0, root, "2S_50FPS", True, "TRAIN"), "the upstream record of the detector is the biology tier's"


def test_atomic_writes_and_the_tier_scan():
    P = par.PARAMETERS.paths.with_condition("FAB")
    table = par.parameterization_of("FAB")
    schema = pio.theta_set_schema(table, condition="FAB", timing_label="2S_50FPS", generator="rds")
    theta = np.tile(np.array([par.prior_center(e) for e in table]), (2, 1))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        theta_dir = root / P.theta_subdir
        records = lambda: pio.tier_record_paths(theta_dir, P.theta_set_pattern, project_alias=P.rds_alias,  # noqa: E731
                                                timing_label="2S_50FPS", split="TRAIN")
        assert records() == [] and pio.tier_schema_scan([], table, condition="FAB").verdict == "no records"
        # a failure after the array write (zarr: in the attrs update) leaves only a .partial name
        p0 = P.theta_set_path(0, root, "2S_50FPS", True, "TRAIN")
        original = pio.zarr.open
        class _Boom(Exception):
            pass
        def failing_open(*a, **k):
            store = original(*a, **k)
            class _Attrs:
                def update(self, *_a, **_k):
                    raise _Boom("injected")
            class _Wrapper:
                attrs = _Attrs()
                def __setitem__(self, key, value):
                    store[key] = value
            return _Wrapper()
        pio.zarr.open = failing_open
        try:
            _expect(_Boom, lambda: pio.write_theta_set(p0, theta, schema))
        finally:
            pio.zarr.open = original
        assert not p0.exists() and (theta_dir / (p0.name + ".partial")).exists()
        assert records() == [], "a .partial name is never a record"
        pio.write_theta_set(p0, theta, schema)                      # the next write succeeds
        assert p0.exists() and not (theta_dir / (p0.name + ".partial")).exists()
        assert [r.name for r in records()] == [p0.name]
        # a failure in the sidecar dump (plain .npy) comes before anything takes a final name: only
        # .partial names remain, no record, and the next write succeeds
        p1 = P.theta_set_path(1, root, "2S_50FPS", False, "TRAIN")
        original_dump = pio.json.dump
        pio.json.dump = lambda *a, **k: (_ for _ in ()).throw(_Boom("injected"))
        try:
            _expect(_Boom, lambda: pio.write_theta_set(p1, theta, schema))
        finally:
            pio.json.dump = original_dump
        assert not p1.exists() and not pio._sidecar_path(p1).exists()
        assert [r.name for r in records()] == [p0.name], "an interrupted write leaves no record"
        # a final-name array without a sidecar (not written by this package): unknown, refused, and
        # repaired by the next write
        np.save(p1, theta)
        scan = pio.tier_schema_scan(records(), table, condition="FAB")
        assert [r.name for r in scan.compatible] == [p0.name] and [r.name for r, _ in scan.unknown] == [p1.name]
        assert scan.refused and "unknown" in scan.verdict and "not a compatibility verdict" in scan.verdict
        pio.write_theta_set(p1, theta, schema)
        scan = pio.tier_schema_scan(records(), table, condition="FAB")
        assert not scan.refused and scan.verdict == "verified compatible (2 records)"
        # a listed record whose final name is gone when it is read is skipped, never unknown
        gone = P.theta_set_path(9, root, "2S_50FPS", True, "TRAIN")
        scan = pio.tier_schema_scan(records() + [gone], table, condition="FAB")
        assert not scan.refused and [p.name for p in scan.vanished] == [gone.name] and scan.n_records == 2
        assert scan.verdict == ("verified compatible (2 records); skipped 1 record(s) removed or being "
                                "replaced during the scan"), scan.verdict
        # current plus superseded: refused, the superseded task named
        old = dict(schema, package_version="0.1.39")
        old["parameter_keys"] = ["count_total", "ratio_dimer_monomer_initial"] + schema["parameter_keys"][2:]
        p2 = P.theta_set_path(2, root, "2S_50FPS", True, "TRAIN")
        pio.write_theta_set(p2, theta, old)
        scan = pio.tier_schema_scan(records(), table, condition="FAB")
        assert [r.name for r, _ in scan.mismatched] == [p2.name] and scan.refused and "TASK_2" in scan.verdict
        # an unreadable record beside the superseded one: both named, no compatibility verdict
        p3 = P.theta_set_path(3, root, "2S_50FPS", True, "TRAIN")
        p3.mkdir(); (p3 / "junk").write_text("x")
        scan = pio.tier_schema_scan(records(), table, condition="FAB")
        assert [r.name for r, _ in scan.mismatched] == [p2.name] and [r.name for r, _ in scan.unknown] == [p3.name]
        assert "TASK_2" in scan.verdict and "TASK_3" in scan.verdict and "verified" not in scan.verdict
        assert scan.n_records == 4
        # the scan continues past the unreadable record whatever the order
        scan_rev = pio.tier_schema_scan(list(reversed(records())), table, condition="FAB")
        assert len(scan_rev.mismatched) == 1 and len(scan_rev.unknown) == 1 and len(scan_rev.compatible) == 2
        # the other condition's tier is a different namespace
        assert pio.tier_record_paths(theta_dir, P.theta_set_pattern, project_alias="SRM_AND_SBI_MONOMER_DIMER_ALP_INLB",
                                     timing_label="2S_50FPS", split="TRAIN") == []
        assert pio.tier_record_paths(theta_dir, P.theta_set_pattern, project_alias=P.rds_alias,
                                     timing_label="2S_50FPS", split="TEST") == []


def _probe_states(writer, probe):
    """Run ``writer`` with ``probe`` called after every rename, unlink and tree removal on its way
    (``os.replace``, ``os.unlink`` -- which ``Path.unlink`` and ``shutil.rmtree`` call -- and
    ``shutil.rmtree`` are wrapped for the duration): every on-disk state a concurrent reader or
    tier scan can observe. Returns ``[(step, probe result)]``."""
    real = {"replace": os.replace, "unlink": os.unlink, "rmtree": shutil.rmtree}
    seen = []

    def wrapped(name):
        def run(target, *a, **k):
            out = real[name](target, *a, **k)
            seen.append((f"{name} {Path(str(target)).name}", probe()))
            return out
        return run
    os.replace, os.unlink, shutil.rmtree = wrapped("replace"), wrapped("unlink"), wrapped("rmtree")
    try:
        writer()
    finally:
        os.replace, os.unlink, shutil.rmtree = real["replace"], real["unlink"], real["rmtree"]
    return seen


def test_publication_never_pairs_values_with_another_writes_schema():
    """Every on-disk state of a fresh write and of a replacement, in both formats, probed by a
    reader holding the superseded ratio table, a reader holding the current table, and a tier scan
    (over a listing taken at that state and over one taken before the write began): no reader
    ever gets values under another write's schema, and no scan ever calls a record unknown."""
    P = par.PARAMETERS.paths.with_condition("FAB")
    table = par.parameterization_of("FAB")
    ratio_table = [dict(e) for e in table]
    ratio_table[1] = dict(ratio_table[1], KEY="ratio_dimer_monomer_initial", PRIOR_RANGE=(-1.27875, -0.47712),
                          LOG_FLAG=True)
    current = pio.theta_set_schema(table, condition="FAB", timing_label="2S_50FPS", generator="rds")
    superseded = pio.theta_set_schema(ratio_table, condition="FAB", timing_label="2S_50FPS", generator="rds")
    ratio_values = np.full((3, len(table)), 0.5); ratio_values[:, 1] = -1.0           # a log10 ratio
    fraction_values = np.full((3, len(table)), 0.5); fraction_values[:, 1] = 0.3      # a basal fraction
    for compress in (True, False):
        for replacing in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                path = P.theta_set_path(0, Path(tmp), "2S_50FPS", compress, "TRAIN")

                def listing():
                    return pio.tier_record_paths(path.parent, P.theta_set_pattern, project_alias=P.rds_alias,
                                                 timing_label="2S_50FPS", split="TRAIN")

                def reads(tbl):
                    try:
                        return np.array(pio.load_theta_set(path, tbl, condition="FAB"), dtype=float)
                    except Exception:  # noqa: BLE001 -- refused, absent or unreadable: never a mixed read
                        return None

                if replacing:
                    pio.write_theta_set(path, ratio_values, superseded)
                listed_before = listing()

                def probe():
                    scans = [pio.tier_schema_scan(lst, table, condition="FAB") for lst in (listing(), listed_before)]
                    return reads(ratio_table), reads(table), scans

                states = _probe_states(lambda: pio.write_theta_set(path, fraction_values, current), probe)
                case = f"{'replacement' if replacing else 'fresh write'} of {path.suffix}"
                assert states, case
                for step, (old, new, scans) in states:
                    assert old is None or np.array_equal(old, ratio_values), (case, step, "superseded table, other values")
                    assert new is None or np.array_equal(new, fraction_values), (case, step, "current table, other values")
                    for scan in scans:
                        assert not scan.unknown, (case, step, scan.verdict)
                assert np.array_equal(reads(table), fraction_values) and reads(ratio_table) is None, case
                assert [p.name for p in listing()] == [path.name], case
                assert not [p.name for p in path.parent.iterdir() if pio.PARTIAL_SUFFIX in p.name], case


def test_a_plain_record_is_read_only_with_the_values_its_sidecar_was_written_with():
    """A plain array's sidecar records the SHA-256 of its value file. Values copied over a record
    apart from their sidecar, a sidecar without the digest, and a reader whose schema read and
    value read straddle a replacement are all refused; the reader never gets the new values under
    the old schema."""
    table = par.parameterization_of("FAB")
    ratio_table = [dict(e) for e in table]
    ratio_table[1] = dict(ratio_table[1], KEY="ratio_dimer_monomer_initial", PRIOR_RANGE=(-1.27875, -0.47712),
                          LOG_FLAG=True)
    schema = pio.theta_set_schema(table, condition="FAB", timing_label="2S_50FPS", generator="rds")
    superseded = pio.theta_set_schema(ratio_table, condition="FAB", timing_label="2S_50FPS", generator="rds")
    a = np.full((3, len(table)), 0.5); a[:, 1] = 0.2
    b = np.full((3, len(table)), 0.5); b[:, 1] = 0.7
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for ext in ("npy", "npz"):
            pa, pb = tmp / f"A.{ext}", tmp / f"B.{ext}"
            pio.write_theta_set(pa, a, schema)
            pio.write_theta_set(pb, b, schema)
            stored = json.loads(pio._sidecar_path(pa).read_text())
            assert stored["values_sha256"] == pio.values_digest(pa.read_bytes())
            assert np.array_equal(pio.load_theta_set(pa, table, condition="FAB"), a)
            # B's values copied over A's, A's sidecar kept: refused by every reader and the dry-run status
            shutil.copyfile(pb, pa)
            _expect(pio.ThetaSetSchemaError, lambda: pio.load_theta_set(pa, table, condition="FAB"),
                    "not the ones the sidecar was written with")
            _expect(pio.ThetaSetSchemaError, lambda: pio.check_theta_set_schema(pa, table, condition="FAB"),
                    "not the ones the sidecar was written with")
            status = pio.theta_set_status(pa, table, condition="FAB")
            assert status.startswith("SCHEMA MISMATCH: the values are not the ones"), status
            # a sidecar without the digest: the values cannot be verified as the ones it describes
            pio.write_theta_set(pa, a, schema)
            legacy = {k: v for k, v in json.loads(pio._sidecar_path(pa).read_text()).items() if k != "values_sha256"}
            pio._sidecar_path(pa).write_text(json.dumps(legacy))
            _expect(pio.ThetaSetSchemaError, lambda: pio.load_theta_set(pa, table, condition="FAB"), "no digest")
            # a reader of the superseded ratio record whose value read comes after a complete
            # replacement by fraction values: refused, never the fractions read as ratios
            pio.write_theta_set(pa, a, superseded)
            real_read = pio.read_theta_set_schema

            def read_then_replace(path, _real=real_read, _pa=pa):
                out = _real(path)
                pio.read_theta_set_schema = _real
                pio.write_theta_set(_pa, b, schema)
                return out
            pio.read_theta_set_schema = read_then_replace
            try:
                _expect(pio.ThetaSetSchemaError, lambda: pio.load_theta_set(pa, ratio_table, condition="FAB"),
                        "rewritten while being read")
            finally:
                pio.read_theta_set_schema = real_read
            assert np.array_equal(pio.load_theta_set(pa, table, condition="FAB"), b)
            _expect(pio.ThetaSetSchemaError, lambda: pio.load_theta_set(pa, ratio_table, condition="FAB"),
                    "parameter keys differ")


# ---- phase 2 and 3: the stage resolvers bind the ESTIMATOR's condition ----------------------

def test_every_stage_spec_binds_the_condition():
    from srm_and_sbi_monomer_dimer_alp.evaluation_runner import _evaluation_spec
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import _experiment_spec
    from srm_and_sbi_monomer_dimer_alp.inference_runner import _inference_spec
    from srm_and_sbi_monomer_dimer_alp.posterior_calibration_runner import _posterior_calibration_spec
    from srm_and_sbi_monomer_dimer_alp.point_estimate_validation_runner import _validation_spec
    from srm_and_sbi_monomer_dimer_alp.simulation_dli_runner import _dli_spec
    b, d = wf.biology_workflow(), wf.detector_workflow()
    for cond, width in (("FAB", 11), ("INLB", 12)):
        spec = _inference_spec(b, cond)
        assert spec.parameter_keys == par.parameter_keys_of(cond) and spec.theta_dim == width
        assert spec.theta_standardization == par.THETA_STANDARDIZATION
        assert len(spec.lower_bound) == width and spec.condition == cond
        assert [e["KEY"] for e in spec.raw_spec if spec.role_of(e) == "learnable"] == par.parameter_keys_of(cond)
        ev = _evaluation_spec(b, cond)
        assert ev.parameter_keys == par.parameter_keys_of(cond) and ev.build_prior().event_shape == (width,)
        assert ev.log_rows == [par.is_log_row(e) for e in par.parameterization_of(cond)]
        ex = _experiment_spec(b, cond)
        assert ex.parameter_keys == par.parameter_keys_of(cond) and ex.build_prior().event_shape == (width,)
        pc = _posterior_calibration_spec(b, cond)
        assert pc.parameter_keys == par.parameter_keys_of(cond) and pc.build_prior().event_shape == (width,)
        dli = _dli_spec(b, cond)
        assert [e["KEY"] for e in dli.tier_table] == par.parameter_keys_of(cond) and dli.diag_spec is dli.tier_table
    dspec = _inference_spec(d, "FAB")
    assert dspec.theta_dim == 6 and dspec.theta_standardization is None
    ddli = _dli_spec(d, "INLB")
    assert [e["KEY"] for e in ddli.tier_table] == par.parameter_keys_of("INLB")
    assert [e["KEY"] for e in ddli.diag_spec] == list(det.DETECTOR_PARAMETER_KEYS)
    assert _validation_spec(d, "FAB").parameter_keys == list(det.DETECTOR_PARAMETER_KEYS)
    _expect(ValueError, lambda: _validation_spec(b, "FAB"), "detector")
    for fn in (_inference_spec, _evaluation_spec, _experiment_spec, _posterior_calibration_spec, _dli_spec):
        _expect(ValueError, lambda: fn(b, None), "condition is required")


def test_video_dataset_reads_each_condition_at_its_width():
    from srm_and_sbi_monomer_dimer_alp.inference_support import VideoDataset
    P = par.PARAMETERS.paths
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for cond, width in (("FAB", 11), ("INLB", 12)):
            table = par.parameterization_of(cond)
            paths = P.with_condition(cond)
            rng = np.random.default_rng(1)
            flow = rng.uniform(par.theta_lower_bound(cond), par.theta_upper_bound(cond), size=(3, width))
            phys = par.to_physical(flow, table)
            pio.write_theta_set(paths.theta_set_path(0, root, "1S_50FPS", True, "TRAIN"), phys,
                                pio.theta_set_schema(table, condition=cond, timing_label="1S_50FPS", generator="rds"))
            video_path = paths.video_set_path(0, root, "1S_50FPS", True, "TRAIN")
            import zarr
            zarr.open(str(video_path), mode="w", shape=(3, 2, 4, 4), chunks=(1, 2, 4, 4), dtype=np.uint8)[:] = 1
            ds = VideoDataset(tasks=1, data_bank_root=root, timing_label="1S_50FPS", compress=True, augment=False,
                              split="TRAIN", paths=paths, parameterization=table)
            _, theta = ds[1]
            assert theta.shape == (width,)
            assert np.allclose(theta.numpy(), flow[1], atol=1e-6)
            assert np.allclose(theta.numpy()[1], phys[1, 1], atol=1e-6), "the linear column is stored as is"
            other = "INLB" if cond == "FAB" else "FAB"
            _expect(pio.ThetaSetSchemaError, lambda: VideoDataset(tasks=1, data_bank_root=root, timing_label="1S_50FPS",
                                                                  compress=True, augment=False, split="TRAIN", paths=paths,
                                                                  parameterization=par.parameterization_of(other)))
            _expect(ValueError, lambda: VideoDataset(tasks=1, data_bank_root=root, timing_label="1S_50FPS", compress=True,
                                                     augment=False, split="TRAIN", paths=paths, parameterization=None), "parameterization")
            _expect(ValueError, lambda: VideoDataset(tasks=1, data_bank_root=root, timing_label="1S_50FPS", compress=True,
                                                     augment=False, split="TRAIN", paths=P, parameterization=table), "condition")


def test_diagnostics_tables_per_condition():
    from srm_and_sbi_monomer_dimer_alp.diagnostics import fixed_parameters_table, prior_sampling_table
    table = par.parameterization_of("INLB")
    theta = np.array([par.prior_center(e) for e in table])
    theta[1], theta[11] = 0.0, 1.0
    headers, rows = prior_sampling_table(table, theta)
    assert len(rows) == 12
    row_f = rows[1]
    assert "linear" in " ".join(map(str, row_f)) and "[+0.00, +1.00]" in " ".join(map(str, row_f))
    _expect(ValueError, lambda: prior_sampling_table(table, theta[:11]), "11")
    _expect(ValueError, lambda: prior_sampling_table(par.parameterization_of("FAB"), theta), "12")
    headers, fixed = fixed_parameters_table(par.parameterization_raw_of("FAB"))
    alloc = [r for r in fixed if r[0] == R_KEY]
    assert len(alloc) == 1 and float(alloc[0][2]) == 0.0, alloc
    headers, fixed = fixed_parameters_table(par.parameterization_raw_of("INLB"))
    assert not [r for r in fixed if r[0] == R_KEY]


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
            except Exception:  # noqa: BLE001 -- report every test, then exit non-zero
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
                continue
            print(f"PASS {name} ({time.time() - t0:.1f} s)")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
