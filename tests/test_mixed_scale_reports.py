"""Mixed-scale reporting (a log row beside the two linear fractions): the evaluation tables,
recovery statistics and figures carry per-row units (dex versus percentage points), the linear
rows get no multiplicative band, no bias factor and no log10 axis; the Evaluation and Experiment
writers refuse a product of other keys; the flagged-fractions tables count values outside [0, 1]
and leave them unchanged; the starting-composition recovery section (the allocation and c per
true-capacity stratum under MET-INLB, the declared 0 under MET-FAB); the calibration kernel's
factors and the shard identity at merge; the test-loss analysis on a mixed manifest; the
estimator-comparison guard; the embedding-distance spec; the estimator artifact's standardization
consistency. Synthetic products only; no GPU, no trained flow.

Run directly from the repository root::

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/test_mixed_scale_reports.py
"""
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("MACHINE_PROFILE", "mars_pc")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from srm_and_sbi_monomer_dimer_alp import artifact_schema as S                       # noqa: E402
from srm_and_sbi_monomer_dimer_alp import evaluation as ev                           # noqa: E402
from srm_and_sbi_monomer_dimer_alp import parameterization as par                    # noqa: E402
from srm_and_sbi_monomer_dimer_alp import posterior_calibration as pcal              # noqa: E402
from srm_and_sbi_monomer_dimer_alp import test_loss_analysis as tla                  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter             # noqa: E402
from srm_and_sbi_monomer_dimer_alp.evaluation import QUANTILE_LEVELS                 # noqa: E402
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow, detector_workflow  # noqa: E402
import _product_fixtures as fx                                                       # noqa: E402

F_KEY, R_KEY, N_KEY = "fraction_basal_dimers_reference", "relative_count_active_dimers_initial", "count_total"
# A three-row mixed table: one log row between two linear fractions.
MIXED = [{"KEY": "a", "LABEL": "$a$", "PRIOR_RANGE": (-1.0, 1.0), "LOG_FLAG": True},
         {"KEY": "f", "LABEL": "$f$", "PRIOR_RANGE": (0.0, 1.0), "LOG_FLAG": False},
         {"KEY": "r", "LABEL": "$r$", "PRIOR_RANGE": (0.0, 1.0), "LOG_FLAG": False}]
MIXED_KEYS = [e["KEY"] for e in MIXED]


def _expect(exc_type, fn, *needles):
    try:
        fn()
    except exc_type as exc:
        for needle in needles:
            assert needle in str(exc), (needle, str(exc))
        return exc
    raise AssertionError(f"expected {exc_type.__name__} with {needles}")


def _reporter(tmp, stage="Evaluation"):
    return DiagnosticReporter(stage=stage, enabled=True, dump=True, dump_dir=Path(tmp), run_label="T",
                              timestamp="now")


def _table(reporter, title_start):
    for title, headers, rows, note in reporter._tables:
        if title.startswith(title_start):
            return headers, rows, note
    raise AssertionError(f"no table starting with {title_start!r}; have {[t[0] for t in reporter._tables]}")


# ---- 1. evaluation tables, statistics and figures on a mixed table ----------------------------

def test_recovery_statistics_and_tables_carry_per_row_units():
    rng = np.random.default_rng(0)
    n = 400
    truth = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    est = truth + np.column_stack([rng.normal(0, 0.1, n), np.full(n, 0.02), rng.normal(0, 0.01, n)])
    stats = ev.recovery_stats(truth, est, 0.3, 0.15, log_rows=[True, False, False])
    assert np.isfinite(stats[0]["frac_within_guide"]) and np.isfinite(stats[0]["frac_within_guide_tight"])
    assert math.isnan(stats[1]["frac_within_guide"]) and math.isnan(stats[2]["frac_within_guide_tight"])
    assert abs(stats[1]["mae"] - 0.02) < 1e-12, "the linear MAE is the absolute error in the coordinate"
    default = ev.recovery_stats(truth, est, 0.3, 0.15)
    assert all(np.isfinite(d["frac_within_guide"]) for d in default), "log_rows=None reads every row as log"
    _expect(ValueError, lambda: ev.recovery_stats(truth, est, log_rows=[True, False]), "log_rows")
    headers, rows = ev.recovery_table(MIXED, truth, est, 0.3, 0.15)
    assert headers[2] == "units" and [r[2] for r in rows] == ["dex", "pp", "pp"]
    assert rows[1][5] == f"{100 * 0.02:.3f}", rows[1]           # MAE in pp = 100 x mean absolute error
    assert rows[1][8] == "n/a" and rows[1][9] == "n/a" and rows[0][8].endswith("%")
    assert rows[2][8] == "n/a", "no dex band on a linear row"
    # the compared and coverage tables carry the same units; the experiment headers no longer say log10
    post_q = np.sort(np.stack([est + d for d in (-0.2, -0.05, 0.0, 0.05, 0.2)], axis=-1), axis=-1)
    h, r = ev.point_estimates_compared_table(MIXED, truth, {"MAP": est, "median": est, "SGM": None})
    assert h[2] == "units" and [x[2] for x in r] == ["dex", "pp", "pp"]
    assert r[1][h.index("MAE MAP")] == f"{100 * 0.02:.3f}" and r[1][h.index("bias MAP")] == f"{+100 * 0.02:+.3f}"
    h, r = ev.posterior_coverage_table(MIXED, truth, post_q)
    assert h[2] == "units" and r[1][2] == "pp"
    h, r = ev.point_estimate_agreement_table(MIXED, est, post_q, est)
    assert h[2] == "units" and r[2][2] == "pp" and r[2][4] == "0.000"
    h, r = ev.experiment_table(MIXED, {"K": est}, ["K"])
    assert "log10" not in " ".join(h) and "median (estimator)" in h and "IQR (estimator)" in h
    assert [x[2] for x in r] == ["log10", "fraction", "fraction"]
    h, r = ev.experiment_estimates_compared_table(MIXED, {"MAP": {"K": est}}, ["K"])
    assert "log10" not in " ".join(h) and h[2] == "units"
    # fraction_outside_prior counts and leaves the input unchanged
    values = np.array([0.5, -0.01, 1.02, 0.3])
    before = values.copy()
    assert abs(ev.fraction_outside_prior(MIXED[1], values) - 0.5) < 1e-12
    assert np.array_equal(values, before)
    hh, rr, n_flagged = ev.fraction_flag_rows(MIXED, {"MAP": np.column_stack([values, values, values])})
    assert [x[0] for x in rr] == ["f", "r"] and n_flagged == 4 and rr[0][4] == "2"
    assert ev.fraction_flag_rows([MIXED[0]], {"MAP": est[:, :1]})[1] == [], "an all-log table has no flagged rows"


def test_linear_row_figures_carry_no_log10_labels_or_dex_guides():
    from srm_and_sbi_monomer_dimer_alp import visualization_inference as vi
    rng = np.random.default_rng(1)
    n = 300
    truth = rng.uniform(0, 1, n)
    est = np.clip(truth + rng.normal(0, 0.05, n), -0.2, 1.2)
    post_q = np.sort(np.stack([est + d for d in (-0.1, -0.03, 0.0, 0.03, 0.1)], axis=-1), axis=-1)

    def texts(fig):
        out = []
        for ax in fig.get_axes():
            out += [ax.get_xlabel(), ax.get_ylabel(), ax.get_title()]
            leg = ax.get_legend()
            if leg is not None:
                out += [t.get_text() for t in leg.get_texts()]
        if fig._suptitle is not None:
            out.append(fig._suptitle.get_text())
        return " | ".join(out)

    lin = vi.figure_recovery_combined(truth, est, post_q, 2, (0.0, 1.0), "f", n_bins=5, min_count=5, log_row=False)
    assert "log" not in texts(lin) and "[0.50x" not in texts(lin) and "0.71x" not in texts(lin), texts(lin)
    log = vi.figure_recovery_combined(truth, est, post_q, 2, (0.0, 1.0), "a", n_bins=5, min_count=5, log_row=True)
    assert "log_{10}" in texts(log) and "[0.50x" in texts(log)
    default = vi.figure_recovery_combined(truth, est, post_q, 2, (0.0, 1.0), "a", n_bins=5, min_count=5)
    assert texts(default) == texts(log), "the default stays the log-row figure"
    fig = vi.figure_point_estimates_vs_truth(truth, {"MAP": est, "median": est}, post_q, (0.0, 1.0), "f",
                                             n_bins=5, min_count=5, log_row=False)
    assert "log10" not in texts(fig)
    fig = vi.figure_experiment_combined({"K": est}, {"K": np.column_stack([est, est, est - 0.03, est + 0.03])},
                                        (0.0, 1.0), "f", log_row=False)
    assert "log" not in texts(fig)
    fig = vi.figure_point_estimates_by_cell({"MAP": est, "SGM": est}, post_q, np.repeat(np.arange(10), 30),
                                            (0.0, 1.0), "f", log_row=False)
    assert "log10" not in texts(fig)


# ---- 2. the writers' key guards and the flagged-fractions tables ------------------------------

def _evaluation_product(rng, keys, n, spec, truth=None, estimates=None):
    d = len(keys)
    lo = np.array([e["PRIOR_RANGE"][0] for e in spec], float)
    hi = np.array([e["PRIOR_RANGE"][1] for e in spec], float)
    if truth is None:
        truth = rng.uniform(lo, hi, size=(n, d))
    est = truth + rng.normal(0, 0.02, size=(n, d)) if estimates is None else estimates
    q = np.sort(np.stack([est + s for s in (-0.1, -0.03, 0.0, 0.03, 0.1)], axis=-1), axis=-1)
    m = fx.valid_manifest("evaluation", n, parameter_keys=list(keys), sgm_scale=list(hi - lo))
    return dict(map_estimate=est.copy(), posterior_quantiles=q, posterior_sgm=est.copy(), scores=rng.normal(size=n),
                true_log10=truth, task_index=np.zeros(n, np.int64), sim_index=np.arange(n, dtype=np.int64),
                **{S.MANIFEST_KEY: S.encode_manifest(m)})


def test_evaluation_writer_refuses_other_keys_and_counts_flagged_fractions():
    from srm_and_sbi_monomer_dimer_alp.evaluation_runner import write_recovery_outputs
    eval_cfg = par.PARAMETERS.inference.evaluation
    rng = np.random.default_rng(2)
    args = SimpleNamespace(bin_mode="quantile", n_bins=4, min_count=2, eval_tasks=1)
    fab = par.parameterization_of("FAB")
    old_keys = [N_KEY, "ratio_dimer_monomer_initial"] + par.parameter_keys_of("FAB")[2:]
    with tempfile.TemporaryDirectory() as tmp:
        # a product of the superseded ratio key, eleven columns like the Fab table, is refused by key
        a = _evaluation_product(rng, old_keys, 20, fab)
        _expect(S.SchemaError, lambda: write_recovery_outputs(_reporter(tmp), args, eval_cfg, fab,
                                                              Path(tmp) / "p.npz", a, run_start=0.0),
                "ratio_dimer_monomer_initial", "exact key and order")
        assert not (Path(tmp) / "p.npz").exists() and not (Path(tmp) / "report.md").exists()
        # a mixed product with values outside [0, 1]: counted, left unchanged, non-fatal
        b = _evaluation_product(rng, MIXED_KEYS, 60, MIXED)
        b["map_estimate"][0, 1], b["map_estimate"][1, 1] = -0.01, 1.02
        stored = b["map_estimate"].copy()
        rep = _reporter(tmp / "b" if False else Path(tmp) / "b")
        man = write_recovery_outputs(rep, args, eval_cfg, MIXED, Path(tmp) / "b" / "p.npz", b, run_start=0.0,
                                     persist_arrays=False)
        assert man["parameter_keys"] == MIXED_KEYS
        assert np.array_equal(b["map_estimate"], stored), "flagging never modifies the stored values"
        h, r, _ = _table(rep, "Flagged fractions")
        assert [x[0] for x in r] == ["f", "r"] and r[0][h.index("MAP outside")] == "2"
        assert any(name == "fractions_within_box" and not ok for name, ok, _ in rep._checks)
        h, r, _ = _table(rep, "MAP recovery")
        assert [x[2] for x in r] == ["dex", "pp", "pp"] and r[1][8] == "n/a"
        assert not any(t[0].startswith("Starting composition") for t in rep._tables), "no condition, no section"
        assert (Path(tmp) / "b" / "report.md").exists()


def _experiment_product(rng, keys, spec, n_cells=3, n_chunks=5, kinds=("FAB",)):
    ids = [(k, c, t) for k in range(len(kinds)) for c in range(n_cells) for t in range(n_chunks)]
    n, d = len(ids), len(keys)
    lo = np.array([e["PRIOR_RANGE"][0] for e in spec], float)
    hi = np.array([e["PRIOR_RANGE"][1] for e in spec], float)
    est = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), size=(n, d))
    q = np.sort(np.stack([est + s for s in (-0.1, -0.03, 0.0, 0.03, 0.1)], axis=-1), axis=-1)
    m = fx.valid_manifest("experiment", n, kinds=list(kinds), parameter_keys=list(keys), sgm_scale=list(hi - lo))
    return dict(map_estimate=est, posterior_quantiles=q, posterior_sgm=est.copy(), scores=rng.normal(size=n),
                kind_index=np.array([i[0] for i in ids], np.int64), cell=np.array([i[1] for i in ids], np.int64),
                chunk=np.array([i[2] for i in ids], np.int64), kinds=np.asarray(list(kinds)),
                **{S.MANIFEST_KEY: S.encode_manifest(m)})


def test_experiment_writer_key_guard_flags_and_misspecification_label():
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import write_experiment_outputs
    eval_cfg = par.PARAMETERS.inference.evaluation
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        a = _experiment_product(rng, MIXED_KEYS, MIXED)
        a["map_estimate"][0, 1], a["map_estimate"][1, 1] = -0.02, 1.03
        stored = a["map_estimate"].copy()
        other = [dict(e, KEY=k) for e, k in zip(MIXED, ["a", "r", "f"])]      # the same keys, another order
        _expect(S.SchemaError, lambda: write_experiment_outputs(_reporter(tmp, "Experiment"),
                                                                SimpleNamespace(aggregation="pooled", seed=None),
                                                                eval_cfg, other, Path(tmp) / "x.npz", a, run_start=0.0),
                "exact key and order")
        assert not (Path(tmp) / "report.md").exists(), "refused before rendering"
        rep = _reporter(Path(tmp) / "e", "Experiment")
        xargs = SimpleNamespace(aggregation="pooled", seed=None, condition="INLB")
        man = write_experiment_outputs(rep, xargs, eval_cfg, MIXED, Path(tmp) / "e" / "p.npz", a, run_start=0.0,
                                       persist_arrays=False)
        assert man["parameter_keys"] == MIXED_KEYS and np.array_equal(a["map_estimate"], stored)
        h, r, note = _table(rep, "Flagged fractions")
        assert r[0][h.index("MAP outside")] == "2" and "MISSPECIFICATION TEST" in note, note
        assert any(name == "fractions_within_box" and not ok for name, ok, _ in rep._checks)
        h, r, _ = _table(rep, "Within-recording drift")
        drift_f = [x for x in r if x[1] == "f"]
        assert drift_f and all(x[4].endswith("pp") and x[7] == "n/a" for x in drift_f), drift_f
        drift_a = [x for x in r if x[1] == "a"]
        assert all(x[4].endswith("dex") and x[7].endswith("%") for x in drift_a)
        # the misspecification label stands beside every table of the other condition's recordings
        h, r, note = _table(rep, "MAP theta by condition")
        assert "MISSPECIFICATION TEST" in note and "log10 units" not in note
        assert "log10" not in " ".join(h)
        # the estimator's own kind carries no such label
        rep2 = _reporter(Path(tmp) / "own", "Experiment")
        b = _experiment_product(rng, MIXED_KEYS, MIXED)
        write_experiment_outputs(rep2, SimpleNamespace(aggregation="pooled", seed=None, condition="FAB"),
                                 eval_cfg, MIXED, Path(tmp) / "own" / "p.npz", b, run_start=0.0, persist_arrays=False)
        assert not any("MISSPECIFICATION" in t[3] for t in rep2._tables)


# ---- 3. the starting-composition recovery section --------------------------------------------

def _inlb_truth_grid():
    """Physical INLB truths over N in {1e3, 1e5}, f in {0, 0.05, 0.5, 0.95, 1} and R in {0, 0.5, 1}."""
    table = par.parameterization_of("INLB")
    center = np.array([par.prior_center(e) for e in table])
    idx = par.parameter_index_of("INLB")
    rows = []
    for n_total in (1e3, 1e5):
        for f in (0.0, 0.05, 0.5, 0.95, 1.0):
            for r in (0.0, 0.5, 1.0):
                row = center.copy()
                row[idx[N_KEY]], row[idx[F_KEY]], row[idx[R_KEY]] = n_total, f, r
                rows.append(row)
    return table, np.asarray(rows)


def test_starting_composition_recovery_strata_pairs_flags_and_fab():
    table, phys = _inlb_truth_grid()
    truth = par.to_flow(phys, table)
    n = truth.shape[0]
    pop = par.population_from_theta(phys, "INLB")
    assert np.any(pop.n_monomers_reference % 2 == 1), "the grid holds an odd A_ref"
    assert np.any(pop.capacity == 0) and np.any(pop.capacity >= 1000)
    est = {"MAP": truth.copy(), "median": truth.copy(), "SGM": truth.copy()}
    q = np.stack([truth + s for s in (-0.01, -0.003, 0.0, 0.003, 0.01)], axis=-1)
    res = ev.starting_composition_recovery("INLB", table, truth, est, q, 2)
    assert res["has_allocation"] and res["n_videos"] == n and res["flagged"] == {"MAP": 0, "median": 0, "SGM": 0}
    assert [s["label"] for s in res["strata"]] == [s[0] for s in ev.CAPACITY_STRATA]
    assert "no effect, prior-like posterior expected" in res["strata"][0]["label"]
    assert all(s["n"] > 0 for s in res["strata"]), [s["n"] for s in res["strata"]]
    assert sum(s["n"] for s in res["strata"]) == n
    for s in res["strata"]:
        for name in ("MAP", "median", "SGM"):
            assert s["allocation"][name]["mae_pp"] == 0.0 and s["allocation"][name]["bias_pp"] == 0.0
        for name in ("MAP", "SGM"):
            assert s["pairs"][name]["mae"] == 0.0 and s["pairs"][name]["bias"] == 0.0
        assert "median" not in s["pairs"], "c is never read from the coordinate-wise median"
        assert s["cover50"] == 1.0 and s["cover90"] == 1.0
    assert np.array_equal(res["true_pairs"], pop.n_pairs_allocated), "c from the truths equals the helper"
    # an estimate with f = -0.01 and R = 1.02 is flagged: NaN c, excluded, counted, the input unchanged
    bad = truth.copy()
    i_f, i_r = par.parameter_index_of("INLB")[F_KEY], par.parameter_index_of("INLB")[R_KEY]
    bad[0, i_f], bad[1, i_r] = -0.01, 1.02
    kept = bad.copy()
    res2 = ev.starting_composition_recovery("INLB", table, truth, {"MAP": bad, "median": truth, "SGM": truth}, q, 2)
    assert res2["flagged"] == {"MAP": 2, "median": 0, "SGM": 0}
    assert np.isnan(res2["estimated_pairs"]["MAP"][:2]).all() and np.isfinite(res2["estimated_pairs"]["MAP"][2:]).all()
    assert np.array_equal(bad, kept) and bad[0, i_f] == -0.01
    assert sum(s["allocation"]["MAP"]["n"] for s in res2["strata"]) == n - 2
    tables = ev.starting_composition_tables(res2)
    assert len(tables) == 2 and "MAP 2" in tables[0][3]
    # a biased allocation: MAE in pp = 100 x the absolute error, bias signed
    shifted = truth.copy(); shifted[:, i_r] += 0.01
    valid = par.population_valid(count_total=phys[:, 0], fraction_basal_dimers_reference=phys[:, i_f],
                                 relative_count_active_dimers_initial=phys[:, i_r] + 0.01, condition="INLB")
    res3 = ev.starting_composition_recovery("INLB", table, truth, {"MAP": shifted}, q, 2)
    assert res3["flagged"]["MAP"] == int((~valid).sum())
    for s in res3["strata"]:
        if s["allocation"]["MAP"]["n"]:
            assert abs(s["allocation"]["MAP"]["mae_pp"] - 1.0) < 1e-9 and abs(s["allocation"]["MAP"]["bias_pp"] - 1.0) < 1e-9
    # MET-FAB: no strata, the allocation declared, c = 0 for the truths and the valid estimates
    fab = par.parameterization_of("FAB")
    fab_phys = phys[:, :11]
    fab_truth = par.to_flow(fab_phys, fab)
    fab_q = np.stack([fab_truth + s for s in (-0.01, -0.003, 0.0, 0.003, 0.01)], axis=-1)
    resf = ev.starting_composition_recovery("FAB", fab, fab_truth, {"MAP": fab_truth, "median": fab_truth,
                                                                     "SGM": fab_truth}, fab_q, 2)
    assert not resf["has_allocation"] and resf["strata"] == [] and resf["c_true_all_zero"]
    assert all(resf["c_estimates_all_zero"].values()) and np.all(resf["true_capacity"] == 0)
    tf = ev.starting_composition_tables(resf)
    assert len(tf) == 1 and tf[0][2][0][1] == "declared 0" and "0 for every video" in tf[0][2][0][2]
    _expect(ValueError, lambda: ev.starting_composition_recovery("FAB", table, truth, est, q, 2), "keys")


def test_evaluation_writer_renders_the_composition_section_under_inlb():
    from srm_and_sbi_monomer_dimer_alp.evaluation_runner import write_recovery_outputs
    eval_cfg = par.PARAMETERS.inference.evaluation
    rng = np.random.default_rng(4)
    table, phys = _inlb_truth_grid()
    truth = par.to_flow(phys, table)
    keys = par.parameter_keys_of("INLB")
    a = _evaluation_product(rng, keys, truth.shape[0], table, truth=truth, estimates=truth.copy())
    args = SimpleNamespace(bin_mode="quantile", n_bins=3, min_count=2, eval_tasks=1, condition="INLB")
    with tempfile.TemporaryDirectory() as tmp:
        rep = _reporter(tmp)
        write_recovery_outputs(rep, args, eval_cfg, table, Path(tmp) / "p.npz", a, run_start=0.0, persist_arrays=False)
        h, r, note = _table(rep, "Starting composition: recovery of the allocation")
        assert len(r) == 5 and r[0][0].startswith("0 (no effect") and "percentage points" in note
        h, r, _ = _table(rep, "Starting composition: recovery of the allocated pairs")
        assert all(x[3] == "0.0" for x in r), r
        assert (Path(tmp) / "figures").exists()
        # the same product under FAB is refused by key, before any rendering
        fab = par.parameterization_of("FAB")
        _expect(S.SchemaError, lambda: write_recovery_outputs(_reporter(Path(tmp) / "f"), SimpleNamespace(
            bin_mode="quantile", n_bins=3, min_count=2, eval_tasks=1, condition="FAB"), eval_cfg, fab,
            Path(tmp) / "f" / "p.npz", a, run_start=0.0), "exact key and order")


# ---- 4. the calibration kernel's factors and the shard identity -------------------------------

def test_calibration_bias_factor_only_on_log_rows_and_shards_carry_keys():
    rng = np.random.default_rng(5)
    n, L = 2000, 200
    truths = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(0, 1, n)])
    samples = truths[:, None, :] + rng.normal(0, 0.05, size=(n, L, 2)) - 0.1      # the posterior sits 0.1 BELOW the truth
    prior = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(0, 1, n)])
    mixed = pcal.CalibrationInputs(truths=truths, samples=samples, theta_keys=["a", "f"], prior_samples=prior,
                                   log_rows=[True, False])
    d_log, d_lin = pcal.diagnose(mixed)
    assert d_log.log_row and not d_lin.log_row
    assert abs(d_log.bias_theta + 0.1) < 0.01 and abs(d_log.bias_factor - 10 ** 0.1) < 0.03
    assert math.isnan(d_lin.bias_factor) and abs(d_lin.bias_theta + 0.1) < 0.01
    assert d_lin.defect.startswith("location (low)")
    today = pcal.diagnose(pcal.CalibrationInputs(truths=truths, samples=samples, theta_keys=["a", "f"],
                                                 prior_samples=prior))
    assert all(d.log_row for d in today) and all(np.isfinite(d.bias_factor) for d in today)
    assert today[0].bias_factor == d_log.bias_factor and today[0].bias_z == d_log.bias_z
    _expect(ValueError, lambda: pcal.CalibrationInputs(truths=truths, samples=samples, theta_keys=["a", "f"],
                                                       prior_samples=prior, log_rows=[True]), "log_rows")
    sub = pcal._subset(mixed, np.arange(10))
    assert list(sub.log_rows) == [True, False]
    # the runner's physical-bias cell and the worst factor over log rows only
    from srm_and_sbi_monomer_dimer_alp.posterior_calibration_runner import _physical_bias_cell, merge_shard_arrays
    assert _physical_bias_cell(d_log).endswith("x") and _physical_bias_cell(d_lin).endswith(" pp")
    assert _physical_bias_cell(d_lin).startswith("-10.")
    # shards: three with the keys, one of a foreign key set, one keyless
    with tempfile.TemporaryDirectory() as tmp:
        def shard(name, keys, cond="INLB", omit_keys=False):
            arrays = dict(truths=truths[:5], samples=samples[:5], truth_log_probs=np.zeros(5),
                          sample_log_probs=np.zeros((5, L)), embeddings=np.zeros((5, 3)))
            if not omit_keys:
                arrays["parameter_keys"] = np.asarray(keys); arrays["condition"] = np.asarray(cond)
            p = Path(tmp) / name
            np.savez_compressed(p, **arrays)
            return p
        good = [shard(f"_shard_0{i}_of_03.npz", ["a", "f"]) for i in range(3)]
        out = merge_shard_arrays(good, ["a", "f"], "INLB")
        assert out[0].shape == (15, 2)
        foreign = good[:2] + [shard("_shard_02_of_03.npz", ["a", "r"])]
        _expect(ValueError, lambda: merge_shard_arrays(foreign, ["a", "f"], "INLB"), "parameter_keys", "['a', 'r']")
        keyless = good[:2] + [shard("_shard_02_of_03.npz", None, omit_keys=True)]
        _expect(ValueError, lambda: merge_shard_arrays(keyless, ["a", "f"], "INLB"), "records no parameter_keys")
        other = good[:2] + [shard("_shard_02_of_03.npz", ["a", "f"], cond="FAB")]
        _expect(ValueError, lambda: merge_shard_arrays(other, ["a", "f"], "INLB"), "condition")
    # --stratify by the allocation under FAB stops: not a target parameter of the Fab estimator
    from srm_and_sbi_monomer_dimer_alp.posterior_calibration_runner import _parse_stratify, _posterior_calibration_spec
    fab = _posterior_calibration_spec(biology_workflow(), "FAB")
    _expect(SystemExit, lambda: _parse_stratify(R_KEY, list(fab.parameter_keys)), "not a target parameter")
    inlb = _posterior_calibration_spec(biology_workflow(), "INLB")
    assert _parse_stratify(R_KEY, list(inlb.parameter_keys)) == (11,)
    assert inlb.log_rows[11] is False and fab.log_rows[1] is False and inlb.condition == "INLB"


# ---- 5. the test-loss analysis on a mixed manifest ----------------------------------------------

def _tld_manifest(keys, log_flags, boxes, condition="INLB", fixed=(), space=None):
    table = [{"key": k, "role": "learnable", "value": 0.0, "prior_range": list(b), "log_flag": f, "log_base": 10 if f else None}
             for k, f, b in zip(keys, log_flags, boxes)]
    table += [{"key": k, "role": "fixed", "value": 0.0, "prior_range": None, "log_flag": False, "log_base": None}
              for k in fixed]
    return {"theta_keys": list(keys), "theta_log_flag": list(log_flags), "parameter_table": table,
            "theta_space": space or ("log10" if all(log_flags) else "estimator"), "condition": condition,
            "test_set_id": "TEST/2S_50FPS"}


def test_test_loss_analysis_reads_mixed_manifests():
    mixed = _tld_manifest(["a", "f", "r"], [True, False, False], [(-1, 1), (0, 1), (0, 1)])
    entries = tla.learnable_entries(mixed)
    nll, widths = tla.prior_reference_nll(entries)
    assert abs(nll - math.log(2.0)) < 1e-12 and widths == [2.0, 1.0, 1.0], "a [0, 1] row adds ln 1 = 0"
    flags = tla.theta_log_flags(mixed, entries)
    assert flags == [True, False, False] and tla.width_units(flags) == ["dex", "abs", "abs"]
    ok, detail = tla.theta_space_check(mixed, flags)
    assert ok and "estimator" in detail
    assert not tla.theta_space_check(dict(mixed, theta_space="log10"), flags)[0]
    # a Fab manifest with the allocation as a fixed row passes the key check (the test-loss runner's rule)
    fab_keys = par.parameter_keys_of("FAB")
    fab = _tld_manifest(fab_keys, [True, False] + [True] * 9, [(-1, 1)] * 11, condition="FAB", fixed=[R_KEY])
    assert [e["key"] for e in tla.learnable_entries(fab)] == fab_keys
    # an all-log manifest passes unchanged under its log10 label
    alllog = _tld_manifest(["x", "y"], [True, True], [(0, 1), (0, 2)])
    assert alllog["theta_space"] == "log10"
    assert tla.theta_space_check(alllog, tla.theta_log_flags(alllog))[0]
    assert abs(tla.prior_reference_nll(tla.learnable_entries(alllog))[0] - math.log(2.0)) < 1e-12
    # a manifest without the flag array falls back to the table's log_flag
    legacy = dict(alllog); legacy.pop("theta_log_flag")
    assert tla.theta_log_flags(legacy) == [True, True]
    fig = tla.figure_loss_vs_theta(np.random.default_rng(0).normal(size=50), np.random.default_rng(1).uniform(size=(50, 3)),
                                   ["a", "f", "r"], nll, [True, False, False])
    labels = [ax.get_xlabel() for ax in fig.get_axes()[:3]]
    assert labels == ["value (log10)", "value (linear)", "value (linear)"], labels


# ---- 6. the estimator-comparison guard ----------------------------------------------------------

def test_estimator_comparison_refuses_different_parameterizations():
    from srm_and_sbi_monomer_dimer_alp.estimator_comparison_runner import _assert_same_parameterization as same
    inlb_keys = par.parameter_keys_of("INLB")
    flags = [par.is_log_row(e) for e in par.parameterization_of("INLB")]
    boxes = [tuple(e["PRIOR_RANGE"]) for e in par.parameterization_of("INLB")]
    a = _tld_manifest(inlb_keys, flags, boxes)
    same(a, json.loads(json.dumps(a)))                                     # identical manifests compare
    b = _tld_manifest([inlb_keys[0], "ratio_dimer_monomer_initial"] + inlb_keys[2:], flags, boxes)
    _expect(ValueError, lambda: same(a, b), "index 1", "ratio_dimer_monomer_initial")
    c = _tld_manifest(inlb_keys, [True] * 12, boxes)
    _expect(ValueError, lambda: same(a, c), "scale", F_KEY)
    d = _tld_manifest(inlb_keys[:11], flags[:11], boxes[:11], condition="FAB")
    _expect(ValueError, lambda: same(a, d), "12 learnable keys", "11")
    e = _tld_manifest(inlb_keys, flags, [boxes[0]] + [(0.0, 0.5)] + boxes[2:])
    _expect(ValueError, lambda: same(a, e), "prior box", F_KEY)
    f = _tld_manifest(inlb_keys, flags, boxes, condition="FAB")
    _expect(ValueError, lambda: same(a, f), "condition")


# ---- 7. the embedding-distance spec and the estimator artifact's standardization ---------------

def test_esd_spec_binds_the_condition_with_meanings():
    from srm_and_sbi_monomer_dimer_alp.embedding_space_distance_runner import _esd_spec
    for cond, width in (("FAB", 11), ("INLB", 12)):
        R = _esd_spec(biology_workflow(), SimpleNamespace(total_time_seconds=2.0, condition=cond))
        assert len(R["parameter_keys"]) == width and R["condition"] == cond
        assert all(R["param_meaning"].get(k) for k in R["parameter_keys"]), [k for k in R["parameter_keys"]
                                                                               if not R["param_meaning"].get(k)]
        assert "ratio_dimer_monomer_initial" not in R["param_meaning"]
    D = _esd_spec(detector_workflow(), SimpleNamespace(total_time_seconds=2.0, condition="INLB"))
    assert len(D["parameter_keys"]) == 6 and all(D["param_meaning"].get(k) for k in D["parameter_keys"])
    _expect(ValueError, lambda: _esd_spec(biology_workflow(), SimpleNamespace(total_time_seconds=2.0, condition=None)),
            "condition is required")


def test_estimator_artifact_refuses_a_standardization_mode_that_disagrees_with_its_buffers():
    import torch
    from srm_and_sbi_monomer_dimer_alp import artifacts
    sd_structured = {"net._transform._transforms.0._shift": torch.zeros(()), "net._transform._transforms.0._scale": torch.ones(()),
                     "net._embedding_net.features.0.weight": torch.zeros(3)}
    sd_independent = {"net._transform._transforms.0._shift": torch.zeros(12), "net._transform._transforms.0._scale": torch.ones(12)}
    artifacts.assert_standardization_consistent(sd_structured, {"z_score_x": "structured"}, 12)
    artifacts.assert_standardization_consistent(sd_independent, {"z_score_x": "independent"}, 12)
    _expect(ValueError, lambda: artifacts.assert_standardization_consistent(sd_structured, {"z_score_x": "independent"}, 12),
            "standardization", "(12,)", "()")
    _expect(ValueError, lambda: artifacts.assert_standardization_consistent(sd_independent, {"z_score_x": "structured"}, 12),
            "standardization", "(12,)")
    _expect(ValueError, lambda: artifacts.assert_standardization_consistent(sd_independent, {"z_score_x": "independent"}, 11),
            "standardization", "(11,)")
    _expect(ValueError, lambda: artifacts.assert_standardization_consistent(sd_independent, {"z_score_x": "none"}, 12),
            "standardization")
    artifacts.assert_standardization_consistent({"net.w": torch.zeros(2)}, {"z_score_x": "none"}, 12)


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
