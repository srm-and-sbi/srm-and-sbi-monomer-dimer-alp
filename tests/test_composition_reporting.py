"""Composition reporting under the window-start composition: the stoichiometry binding, the per-draw
summaries, the flagged draws, the starting-composition recovery (strata, c, the declared Fab
allocation), the expected-fluorescence function, and an end-to-end population-composition run on a
schema-valid fixture product. Also the condition binding of the two analyses that summarize the same
window MAPs: the Sample Geometric Median (its plane, the linear rows, an out-of-box member kept
selectable) and the temporal dynamics (keys, display names and units of the fraction rows).

Run directly from the repository root::

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/test_composition_reporting.py
"""
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("MACHINE_PROFILE", "mars_pc")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _product_fixtures import valid_manifest                                        # noqa: E402
from srm_and_sbi_monomer_dimer_alp import artifact_schema as S                       # noqa: E402
from srm_and_sbi_monomer_dimer_alp import population_composition as pc               # noqa: E402
from srm_and_sbi_monomer_dimer_alp import population_composition_runner as pcr       # noqa: E402
from srm_and_sbi_monomer_dimer_alp.labeling import expected_fluorescent_complexes, resolve_labeling  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import (                         # noqa: E402
    parameter_keys_of, parameterization_of, realize_initial_composition, realize_population,
    theta_lower_bound, theta_upper_bound, to_flow,
)
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow                  # noqa: E402

CONDITIONS = ("FAB", "INLB")
F_KEY = "fraction_basal_dimers_reference"
R_KEY = "relative_count_active_dimers_initial"


def _binding(condition):
    return pcr.stoichiometry_binding(parameterization_of(condition), parameter_keys_of(condition), condition)


def _theta_flow(condition, n_total, fraction, allocation=None):
    """Estimator-space theta rows at the prior center, with the stoichiometry set (broadcast)."""
    table = parameterization_of(condition)
    keys = [e["KEY"] for e in table]
    n_total = np.asarray(n_total, dtype=float)
    fraction = np.broadcast_to(np.asarray(fraction, dtype=float), n_total.shape)
    center = np.array([(e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2.0 for e in table])
    theta = np.broadcast_to(center, n_total.shape + (len(keys),)).copy()
    theta[..., keys.index("count_total")] = np.log10(n_total)
    theta[..., keys.index(F_KEY)] = fraction
    if R_KEY in keys:
        theta[..., keys.index(R_KEY)] = 0.5 if allocation is None else allocation
    return theta


# ---- the binding and the kernel pair ----------------------------------------------------------

def test_composition_of_the_reference_pair_returns_f_and_the_start_pair_returns_f_start():
    for condition in CONDITIONS:
        b = _binding(condition)
        assert b.count_keys[:2] == ("count_total", F_KEY)
        assert (len(b.count_keys) == 3) == (condition == "INLB")
        f = np.array([0.0, 0.05, 0.075, 0.25, 0.5, 1.0])
        theta = _theta_flow(condition, np.full(f.shape, 1e4), f, allocation=0.5)
        block = b.select(theta)
        pair = b.reference_pair(block)
        assert np.allclose(pair[:, 0], 1e4) and np.allclose(pair[:, 1], 2 * f / (1 + f))
        comp = pc.composition(pair)
        assert np.allclose(comp[:, pc.DIMER_INDEX], f)                      # composition of (N, 2f/(1+f)) is f
        assert np.allclose(comp[:, pc.RECEPTORS_INDEX], 1e4)
        start = b.start_pair(block)
        fields, ok = b.realize(block)
        assert ok.all()
        f_start = fields["fraction_basal_dimers_start"]
        assert np.allclose(start[:, 0], 1e4), "the starting pair keeps the total receptors N"
        assert np.allclose(pc.composition(start)[:, pc.DIMER_INDEX], f_start)
        if condition == "FAB":
            assert np.allclose(f_start, f) and np.all(fields["n_pairs_allocated"] == 0)
        else:
            assert np.all(f_start[1:-1] > f[1:-1]) and np.all(fields["n_pairs_allocated"][1:-1] > 0)
            assert f_start[-1] == 1.0 and fields["capacity"][-1] == 0            # f = 1: no monomer, no pair
    print("PASS test_composition_of_the_reference_pair_returns_f_and_the_start_pair_returns_f_start")


def test_window_mean_over_valid_draws_and_a_flagged_draw_is_nan_never_clamped():
    b = _binding("INLB")
    f = np.array([0.1, 0.2, 0.3, 1.2, -0.05])                     # three valid, two flagged
    theta = _theta_flow("INLB", np.full(5, 1e4), f, allocation=0.5)[None]       # one window, five draws
    block = b.select(theta)
    ok = b.valid(block)
    assert ok.tolist() == [[True, True, True, False, False]]
    pair = b.reference_pair(block)
    assert np.isnan(pair[0, 3]).all() and np.isnan(pair[0, 4]).all()
    comp = pc.composition(pair)
    assert np.isnan(comp[0, 3]).all(), "a draw at f = 1.2 yields NaN, never a clamped value"
    window = pc.window_draw_mean(comp, ok)
    assert np.isclose(window[0, pc.DIMER_INDEX], np.mean([0.1, 0.2, 0.3]))
    # window_draw_mean on the raw composition without the mask drops the NaN draws too
    assert np.isclose(pc.window_draw_mean(comp)[0, pc.DIMER_INDEX], np.mean([0.1, 0.2, 0.3]))
    fields, valid = b.realize(block)
    assert np.isnan(fields["n_monomers"][0, 3]) and np.isnan(fields["capacity"][0, 4])
    assert np.isfinite(fields["n_monomers"][0, :3]).all()
    # a window with every draw flagged gives NaN
    theta_bad = _theta_flow("INLB", np.full(2, 1e4), [1.5, 2.0], allocation=0.5)[None]
    assert np.isnan(pc.window_draw_mean(pc.composition(b.reference_pair(b.select(theta_bad))))).all()
    # the shape of the mask is checked
    try:
        pc.window_draw_mean(comp, np.ones((2, 5), bool))
    except ValueError:
        pass
    else:
        raise AssertionError("a mask of the wrong shape must be refused")
    print("PASS test_window_mean_over_valid_draws_and_a_flagged_draw_is_nan_never_clamped")


def test_per_draw_binding_equals_the_one_draw_initializer_element_wise():
    n_grid = np.array([1e3, 1e4, 1e5, 25.0, 3.3e4])
    f_grid = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
    r_grid = np.array([0.0, 0.5, 1.0])
    for condition in CONDITIONS:
        b = _binding(condition)
        n_all, f_all, r_all = [np.asarray(a).ravel() for a in np.meshgrid(n_grid, f_grid, r_grid, indexing="ij")]
        if condition == "FAB":
            r_all = np.zeros_like(r_all)
        theta = _theta_flow(condition, n_all, f_all, allocation=r_all)
        fields, ok = b.realize(theta[..., list(b.count_index)])
        assert ok.all()
        saw_odd = saw_no_pair = False
        for i in range(n_all.size):
            one = realize_initial_composition(count_total=float(n_all[i]), fraction_basal_dimers_reference=float(f_all[i]),
                                              relative_count_active_dimers_initial=float(r_all[i]), condition=condition)
            for name in pcr.REALIZATION_KEYS:
                if name == "expected_fluorescent_complexes":
                    continue
                expect = getattr(one, name)
                got = fields[name][i]
                assert (np.isnan(got) and np.isnan(expect)) or np.isclose(got, expect), (condition, name, i, got, expect)
            saw_odd |= one.n_monomers_reference % 2 == 1
            saw_no_pair |= one.capacity == 0
        assert saw_odd and saw_no_pair, "the grid covers an odd A_ref and a no-pair case"
        # the InlB worked literals: N = 1e4, f = 0.2 -> A_ref 317, B1 151, B2 4, cap 158; c 79 and 158
        if condition == "INLB":
            one = _theta_flow("INLB", np.array([1e4, 1e4, 1e4]), 0.2, allocation=np.array([0.0, 0.5, 1.0]))
            got, _ok = b.realize(b.select(one))
            assert got["n_monomers_reference"].tolist() == [317, 317, 317] and got["capacity"].tolist() == [158] * 3
            assert got["n_pairs_allocated"].tolist() == [0, 79, 158] and got["n_monomers"].tolist() == [317, 159, 1]
    print("PASS test_per_draw_binding_equals_the_one_draw_initializer_element_wise")


def test_per_draw_summary_differs_from_the_summary_at_the_coordinate_medians():
    """A correlated cloud: the total and the fraction move together, so the mean of the per-draw
    starting counts is not the count at the coordinate-wise medians."""
    rng = np.random.default_rng(3)
    b = _binding("INLB")
    z = rng.normal(size=4000)
    n_total = 10 ** (4.0 + 0.4 * z)
    fraction = 1.0 / (1.0 + np.exp(-(z - 0.85)))                  # in (0, 1), rising with the total
    theta = _theta_flow("INLB", n_total, fraction, allocation=0.5)[None]           # one window
    fields, ok = b.realize(b.select(theta))
    per_draw = pc.window_draw_mean(fields["n_dimers_two_probe"], ok)[0]
    med = np.median(theta[0], axis=0)
    at_median = b.realize(b.select(med[None]))[0]["n_dimers_two_probe"][0]
    assert abs(per_draw - at_median) > 0.05 * at_median, (per_draw, at_median)
    print("PASS test_per_draw_summary_differs_from_the_summary_at_the_coordinate_medians")


# ---- the fluorescence function ---------------------------------------------------------------------

def test_expected_fluorescent_complexes_matches_hand_values():
    # MET-FAB: A q + B [(1 - s2) q + s2 (1 - (1 - q)^2)] on the placed counts
    fab = resolve_labeling("FAB")
    q, s2 = fab.law.fluorescent_probability, fab.two_probe_share
    pop = types.SimpleNamespace(n_monomers=np.array([860, 0, 100]), n_dimers=np.array([70, 20, 0]),
                                n_dimers_one_probe=None, n_dimers_two_probe=None)
    expect = pop.n_monomers * q + pop.n_dimers * ((1 - s2) * q + s2 * (1 - (1 - q) ** 2))
    assert np.allclose(expected_fluorescent_complexes(pop, fab), expect)
    assert np.isclose(expected_fluorescent_complexes(pop, fab)[0], 860 * q + 70 * ((1 - s2) * q + s2 * (1 - (1 - q) ** 2)))
    # MET-INLB: A q + B1 q + B2 (1 - (1 - q)^2); q = 0.5 -> 0.5 A + 0.5 B1 + 0.75 B2
    inlb = resolve_labeling("INLB")
    assert np.isclose(inlb.law.fluorescent_probability, 0.5)
    pop = types.SimpleNamespace(n_monomers=np.array([200, 0]), n_dimers_one_probe=np.array([100, 40]),
                                n_dimers_two_probe=np.array([40, 0]), n_dimers=np.array([140, 40]))
    assert np.allclose(expected_fluorescent_complexes(pop, inlb), [0.5 * 200 + 0.5 * 100 + 0.75 * 40, 20.0])
    # on a realized population (the same formula on the starting counts), and refused for coins
    real = realize_population(count_total=1e4, fraction_basal_dimers_reference=0.2,
                              relative_count_active_dimers_initial=0.5, condition="INLB")
    assert np.isclose(expected_fluorescent_complexes(real, inlb), 0.5 * 159 + 0.5 * 151 + 0.75 * 83)
    try:
        expected_fluorescent_complexes(real, resolve_labeling("INLB", None, "0.4"))
    except ValueError:
        pass
    else:
        raise AssertionError("the coins override has no probe classes and must be refused")
    print("PASS test_expected_fluorescent_complexes_matches_hand_values")


# ---- the starting-composition recovery: strata, c, the declared Fab allocation --------------------

def test_count_recovery_statistics_strata_c_and_the_declared_fab_allocation():
    # InlB truths covering capacity 0 (N = 25 at f = 1), an odd A_ref, R at 0 and 1, N at 1e3 and 1e5
    n = np.array([25.0, 1e3, 1e3, 1e4, 1e4, 1e5, 1e5, 1e5, 200.0])
    f = np.array([1.0, 0.05, 0.2, 0.2, 0.2, 0.05, 0.5, 0.0, 0.2])
    r = np.array([0.5, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0, 0.5, 0.5])
    b = _binding("INLB")
    truth = _theta_flow("INLB", n, f, allocation=r)
    t_fields, t_ok = b.realize(b.select(truth))
    assert t_ok.all()
    assert t_fields["capacity"][0] == 0 and t_fields["n_pairs_allocated"][0] == 0
    strata = pcr.capacity_strata(t_fields["capacity"])
    labels = [lab for lab, _m in strata]
    assert labels[0].startswith("0 (no effect") and labels[1:] == ["1-9", "10-99", "100-999", ">= 1000"]
    from srm_and_sbi_monomer_dimer_alp.evaluation import CAPACITY_STRATA
    assert labels == [s[0] for s in CAPACITY_STRATA], "one definition of the strata for both reports"
    assert sum(int(m.sum()) for _l, m in strata) == n.size            # every truth in exactly one stratum
    assert all(int(m.sum()) >= 1 for _l, m in strata), [int(m.sum()) for _l, m in strata]
    # estimates equal to the truths: c MAE 0 in every stratum
    rows = pc.count_recovery_statistics({"c": t_fields["n_pairs_allocated"]}, {"c": t_fields["n_pairs_allocated"]},
                                        ["c"], strata)
    assert [row["stratum"] for row in rows] == ["all"] + labels
    assert all(row["mae"] == 0.0 and row["bias"] == 0.0 for row in rows)
    assert rows[0]["unit"] == "count"
    # estimates with f = -0.01 and R = 1.02 are flagged: NaN c, the stored estimates unchanged
    est = truth.copy()
    est[1, b.count_index[1]] = -0.01
    est[2, b.count_index[2]] = 1.02
    before = est.copy()
    e_fields, e_ok = b.realize(b.select(est))
    assert e_ok.tolist() == [True, False, False, True, True, True, True, True, True]
    assert np.isnan(e_fields["n_pairs_allocated"][1]) and np.isnan(e_fields["n_pairs_allocated"][2])
    assert np.array_equal(est, before), "nothing is clipped or rewritten"
    rows = pc.count_recovery_statistics({"c": t_fields["n_pairs_allocated"]}, {"c": e_fields["n_pairs_allocated"]},
                                        ["c"], strata)
    assert rows[0]["n"] == 7 and rows[0]["mae"] == 0.0
    # c from the truths equals the helper on the physical values
    for i in range(n.size):
        one = realize_initial_composition(count_total=n[i], fraction_basal_dimers_reference=f[i],
                                          relative_count_active_dimers_initial=r[i], condition="INLB")
        assert one.n_pairs_allocated == t_fields["n_pairs_allocated"][i]
    # zero-safe: a zero count is a legitimate value, errors in counts, no logarithm anywhere
    rows = pc.count_recovery_statistics(np.array([[0.0], [0.0], [3.0]]), np.array([[0.0], [2.0], [3.0]]), ["c"])
    assert rows[0]["n"] == 3 and np.isclose(rows[0]["mae"], 2 / 3) and np.isclose(rows[0]["bias"], 2 / 3)
    # Fab: no R strata, c = 0 for every draw (the allocation is declared), the binding has two keys
    fb = _binding("FAB")
    fab_truth = _theta_flow("FAB", n, f)
    ff, f_ok = fb.realize(fb.select(fab_truth))
    assert f_ok.all() and np.all(ff["n_pairs_allocated"] == 0) and np.all(ff["capacity"] == 0)
    assert fb.declared_allocation == 0.0 and len(fb.count_keys) == 2
    assert np.allclose(ff["fraction_basal_dimers_start"], f)
    print("PASS test_count_recovery_statistics_strata_c_and_the_declared_fab_allocation")


# ---- end to end on a fixture product ---------------------------------------------------------------

def _experiment_product(condition, rng, kinds=("FAB", "INLB"), n_cells=3, n_chunks=4, n_draws=50):
    keys = parameter_keys_of(condition)
    D = len(keys)
    lo, hi = np.array(theta_lower_bound(condition)), np.array(theta_upper_bound(condition))
    ids = [(k, c, t) for k in range(len(kinds)) for c in range(n_cells) for t in range(n_chunks)]
    n = len(ids)
    m = valid_manifest("experiment", n, kinds=list(kinds), parameter_keys=keys, sgm_scale=[1.0] * D,
                       stored_optional_fields=["posterior_samples_cloud"], coordinate_transform="estimator")
    cloud = rng.uniform(lo, hi, size=(n, n_draws, D))
    # the second kind's windows sit at a higher basal fraction, a few draws leave the box
    cloud[n // 2:, :, keys.index(F_KEY)] = 0.3 + 0.7 * cloud[n // 2:, :, keys.index(F_KEY)]
    cloud[0, :3, keys.index(F_KEY)] = 1.3                           # three flagged draws in window 0
    a = dict(map_estimate=rng.uniform(lo, hi, size=(n, D)),
             posterior_quantiles=np.sort(rng.uniform(lo[:, None], hi[:, None], size=(n, D, 5)), axis=-1),
             posterior_sgm=rng.uniform(lo, hi, size=(n, D)), scores=rng.normal(size=n),
             kind_index=np.array([i[0] for i in ids], dtype=np.int64), cell=np.array([i[1] for i in ids], dtype=np.int64),
             chunk=np.array([i[2] for i in ids], dtype=np.int64), kinds=np.asarray(list(kinds)),
             posterior_samples_cloud=cloud)
    a[S.MANIFEST_KEY] = S.encode_manifest(m)
    return a


def _evaluation_product(condition, rng, n=60):
    keys = parameter_keys_of(condition)
    D = len(keys)
    lo, hi = np.array(theta_lower_bound(condition)), np.array(theta_upper_bound(condition))
    m = valid_manifest("evaluation", n, parameter_keys=keys, sgm_scale=[1.0] * D, coordinate_transform="estimator")
    truth = rng.uniform(lo, hi, size=(n, D))
    est = truth + rng.normal(scale=0.02, size=(n, D))
    est[0, keys.index(F_KEY)] = 1.05                                  # one flagged MAP
    a = dict(map_estimate=est, posterior_quantiles=np.sort(rng.uniform(lo[:, None], hi[:, None], size=(n, D, 5)), axis=-1),
             posterior_sgm=truth.copy(), scores=rng.normal(size=n), true_log10=truth,
             task_index=np.zeros(n, dtype=np.int64), sim_index=np.arange(n, dtype=np.int64))
    a[S.MANIFEST_KEY] = S.encode_manifest(m)
    return a


def test_end_to_end_run_on_a_fixture_product_writes_both_tables_and_the_new_arrays():
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(11)
    real_spec = pcr._population_composition_spec
    for condition in CONDITIONS:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            exp = tmp / "Experiment.npz"
            rec = tmp / "Recovery.npz"
            np.savez_compressed(exp, **_experiment_product(condition, rng))
            np.savez_compressed(rec, **_evaluation_product(condition, rng))
            report_dir = tmp / f"X_{condition}_2S_50FPS_Experiment_Population_Composition"

            def isolated_spec(cfg, args, _real=real_spec, _exp=exp, _rec=rec, _dir=report_dir):
                spec = _real(cfg, args)
                return pcr.CompositionSpec(**{**spec.__dict__, "experiment_npz": _exp, "recovery_npz": _rec,
                                              "report_dir": _dir})

            pcr._population_composition_spec = isolated_spec
            try:
                args = pcr.build_parser("t").parse_args(["--condition", condition, "--total-time-seconds", "2",
                                                        "--bootstrap", "50"])
                assert pcr.run_population_composition(biology_workflow(), args) == 0
            finally:
                pcr._population_composition_spec = real_spec
            npz = report_dir / f"{report_dir.name}.npz"
            assert npz.exists()
            with np.load(npz, allow_pickle=False) as d:
                files = set(d.files)
                for name in ("window_composition", "window_composition_start", "cell_composition_start",
                             "realization_window_mean", "realization_keys", "flagged_draws_per_window",
                             "flagged_share", "condition", "parameter_keys", "count_keys"):
                    assert name in files, name
                assert str(d["condition"]) == condition
                assert [str(k) for k in d["parameter_keys"]] == parameter_keys_of(condition)
                assert [str(k) for k in d["realization_keys"]] == list(pcr.REALIZATION_KEYS)
                flagged = d["flagged_draws_per_window"]
                assert flagged[0] == 3 and flagged[1:].sum() == 0
                assert d["realization_window_mean"].shape == (24, len(pcr.REALIZATION_KEYS))
                if condition == "FAB":
                    assert np.allclose(d["window_composition"], d["window_composition_start"], equal_nan=True)
                else:
                    assert np.nanmean(d["window_composition_start"][:, pc.DIMER_INDEX]) > np.nanmean(d["window_composition"][:, pc.DIMER_INDEX])
            report = next(report_dir.glob("*.md")).read_text()
            if condition == "FAB":
                assert "start = reference" in report
                assert "STARTING (window-start) population composition" not in report
            else:
                assert "STARTING (window-start) population composition" in report
                assert report.index("STARTING (window-start) population composition") < report.index("REFERENCE population composition")
                assert "per true-capacity stratum" in report
            assert "Window-start realization per draw" in report and "Flagged draws" in report
            assert "MISSPECIFICATION TEST" in report                   # the other kind's recordings are labeled
            assert "misspecification test" in report
            assert "expected fluorescent complexes at frame 0" in report
            assert "Starting-composition recovery" in report
    print("PASS test_end_to_end_run_on_a_fixture_product_writes_both_tables_and_the_new_arrays")


def test_a_cloud_without_a_valid_draw_reports_its_exclusions_and_marks_the_composition_unavailable():
    """Every draw flagged (every basal fraction at 1.2 in a schema-valid product): the run still
    writes its report. The closure and replicate checks are UNAVAILABLE, never a fatal failure; the
    composition summaries and their figures are marked unavailable and left out; the flagged-draw
    counts and the synthetic validation stay; the stored compositions are NaN, never clipped."""
    import matplotlib
    matplotlib.use("Agg")
    real_spec = pcr._population_composition_spec
    absent = ("composition, span-averaged", "first window only", "Window-start realization per draw",
              "Bootstrap check", "Sensitivity:", "Per-recording spread", "Within-recording time course",
              "Condition contrast")
    for condition in CONDITIONS:
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            exp, rec = tmp / "Experiment.npz", tmp / "Recovery.npz"
            product = _experiment_product(condition, rng)
            product["posterior_samples_cloud"][..., parameter_keys_of(condition).index(F_KEY)] = 1.2
            n_windows, n_draws = product["posterior_samples_cloud"].shape[:2]
            np.savez_compressed(exp, **product)
            np.savez_compressed(rec, **_evaluation_product(condition, rng))
            report_dir = tmp / f"X_{condition}_2S_50FPS_Experiment_Population_Composition"

            def isolated_spec(cfg, args, _real=real_spec, _exp=exp, _rec=rec, _dir=report_dir):
                spec = _real(cfg, args)
                return pcr.CompositionSpec(**{**spec.__dict__, "experiment_npz": _exp, "recovery_npz": _rec,
                                              "report_dir": _dir})

            pcr._population_composition_spec = isolated_spec
            try:
                args = pcr.build_parser("t").parse_args(["--condition", condition, "--total-time-seconds", "2",
                                                        "--bootstrap", "50"])
                assert pcr.run_population_composition(biology_workflow(), args) == 0
            finally:
                pcr._population_composition_spec = real_spec
            with np.load(report_dir / f"{report_dir.name}.npz", allow_pickle=False) as d:
                for name in ("window_composition", "window_composition_start", "realization_window_mean"):
                    assert np.isnan(d[name]).all(), name                 # nothing clipped, nothing manufactured
                assert (d["flagged_draws_per_window"] == n_draws).all() and float(d["flagged_share_total"]) == 1.0
            report = (report_dir / "report.md").read_text()
            assert "Note: NO VALID COMPOSITION DRAWS" in report
            assert "## Checks - 3 passed, 0 failed, 2 unavailable" in report
            assert "| composition_closure | UNAVAILABLE |" in report
            assert "| replicate_unit_is_recording | UNAVAILABLE |" in report
            assert "| composition summaries | unavailable |" in report
            per_kind_windows = n_windows // 2
            per_kind_draws = per_kind_windows * n_draws
            row = (f"| {per_kind_windows} | {per_kind_draws:,} / {per_kind_draws:,} | 100.000 | {per_kind_windows} | "
                   f"{per_kind_windows} | {per_kind_draws:,} |")
            assert report.count(row) == 2, "the exclusion counts of both recording kinds"
            for title in absent:
                assert title not in report, title
            assert "Synthetic validation" in report and "Starting-composition recovery" in report
            assert sorted(p.name for p in (report_dir / "figures").iterdir()) == ["composition_recovery.png"]
    print("PASS test_a_cloud_without_a_valid_draw_reports_its_exclusions_and_marks_the_composition_unavailable")


def test_the_spec_binds_the_estimator_condition_and_refuses_the_detector():
    from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow
    for condition in CONDITIONS:
        args = pcr.build_parser("t").parse_args(["--condition", condition, "--total-time-seconds", "2"])
        spec = pcr._population_composition_spec(biology_workflow(), args)
        assert spec.condition == condition and spec.parameter_keys == parameter_keys_of(condition)
        assert spec.binding.count_index == ((0, 1) if condition == "FAB" else (0, 1, 11))
        assert len(spec.prior_low) == len(spec.parameter_keys)
        assert pcr.kind_label("FAB", condition) == ("MET-FAB" if condition == "FAB" else "MET-FAB [misspecification test]")
        assert pcr.misspecification_note([condition], condition) == ""
        assert "MISSPECIFICATION TEST" in pcr.misspecification_note(["FAB", "INLB"], condition)
        try:
            pcr._population_composition_spec(detector_workflow(), args)
        except SystemExit as exc:
            assert "biology-only" in str(exc)
        else:
            raise AssertionError("the detector workflow must be refused")
    # the binding refuses a table whose keys differ from the keys handed in
    try:
        pcr.stoichiometry_binding(parameterization_of("INLB"), parameter_keys_of("FAB"), "INLB")
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched keys must be refused")
    # to_flow / to_physical round trip through the binding's rows: the fraction stays bitwise
    b = _binding("INLB")
    u = _theta_flow("INLB", np.array([1e4]), 0.3, allocation=0.7)
    n_total, f, r = b.components(b.select(u))
    assert f[0] == 0.3 and r[0] == 0.7 and np.isclose(n_total[0], 1e4)
    assert to_flow(np.array([[1e4, 0.3, 0.7]]), list(b.rows))[0, 1] == 0.3
    print("PASS test_the_spec_binds_the_estimator_condition_and_refuses_the_detector")


# ---- the other analyses of the same window MAPs: their condition binding ---------------------------

def test_sgm_spec_binds_the_condition_and_keeps_an_out_of_box_member_selectable():
    """The SGM analysis binds the estimator's condition: eleven or twelve keys, the plane pairs the
    basal reference fraction (linear axis) with the dissociation rate (log axis) in both conditions,
    and log_rows is False exactly at the fraction rows; the detector plane is unchanged. On a
    synthetic 12-D pool the linear rows round-trip bitwise, and a member at f = 1.05 is selected as
    the SGM with sgm_in_box False: flagged, never clipped."""
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp import sample_geometric_median as sgm
    from srm_and_sbi_monomer_dimer_alp import sample_geometric_median_runner as sgr
    from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow
    specs = {}
    for condition in CONDITIONS:
        args = sgr.build_parser("t").parse_args(["--total-time-seconds", "2", "--condition", condition])
        spec = specs[condition] = sgr._sgm_spec(biology_workflow(), args)
        keys = parameter_keys_of(condition)
        assert spec.parameter_keys == keys and len(keys) == (11 if condition == "FAB" else 12)
        xi, yi, caption = spec.plane
        assert (keys[xi], keys[yi]) == (F_KEY, "rate_dissociation")
        assert not spec.log_rows[xi] and spec.log_rows[yi]
        assert [k for k, lg in zip(keys, spec.log_rows) if not lg] == ([F_KEY] if condition == "FAB" else [F_KEY, R_KEY])
        assert np.array_equal(spec.theta_low, theta_lower_bound(condition))
        assert np.array_equal(spec.theta_high, theta_upper_bound(condition))
        assert "REFERENCE" in caption and "ratio" not in caption.split("dissociation")[0]
    dargs = sgr.build_parser("t").parse_args(["--total-time-seconds", "2", "--condition", "INLB"])
    dspec = sgr._sgm_spec(detector_workflow(), dargs)
    dkeys = list(det.DETECTOR_PARAMETER_KEYS)
    assert dspec.parameter_keys == dkeys and dspec.log_rows.all()
    assert dspec.plane[:2] == (dkeys.index("mu_r"), dkeys.index("mu_pc"))
    # a synthetic 12-D pool: 70 members at f = 1.05 (outside the box), 10 inside it
    spec, keys = specs["INLB"], parameter_keys_of("INLB")
    rng = np.random.default_rng(4)
    lo, hi = spec.theta_low, spec.theta_high
    pool = (lo + hi) / 2.0 + rng.normal(scale=0.02, size=(80, 12)) * (hi - lo)
    i_f = keys.index(F_KEY)
    pool[:70, i_f] = 1.05
    pool[70:, i_f] = 0.5 + rng.normal(scale=0.02, size=10)
    back = spec.to_flow(spec.to_physical(pool))
    linear = ~spec.log_rows
    assert np.array_equal(back[:, linear], pool[:, linear]) and np.allclose(back, pool)
    results, in_box = sgm.summary_vectors(pool, lo, hi, rng, spec.to_physical, spec.to_flow)
    assert int(in_box.sum()) == 10
    unrestricted = next(r for r in results if r["variant"] == "unrestricted")
    assert unrestricted["sgm_log"][i_f] == 1.05 and unrestricted["sgm_abs"][i_f] == 1.05
    assert unrestricted["sgm_in_box"] is False, "the member outside the box is selectable and flagged"
    bounded = next(r for r in results if r["variant"] == "bounded_in_box")
    assert bounded["n"] == 10 and bounded["sgm_in_box"] is True and bounded["sgm_log"][i_f] < 1.0
    assert np.array_equal(pool[:70, i_f], np.full(70, 1.05)), "nothing is clipped"
    print("PASS test_sgm_spec_binds_the_condition_and_keeps_an_out_of_box_member_selectable")


def test_temporal_spec_binds_the_condition_with_names_and_units_of_the_fraction_rows():
    """The temporal-dynamics analysis binds the estimator's condition (keys, table and box: eleven
    or twelve rows); the two fraction rows carry their display names and a unit that is never
    'ratio'; every key has a display name; the detector spec is unchanged (all log, condition-free)."""
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp import temporal_dynamics_runner as tdr
    from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow
    for condition in CONDITIONS:
        args = tdr.build_parser("t").parse_args(["--condition", condition, "--total-time-seconds", "2"])
        spec = tdr._temporal_dynamics_spec(biology_workflow(), args)
        keys = parameter_keys_of(condition)
        assert spec.condition == condition and spec.keys == keys and [e["KEY"] for e in spec.table] == keys
        assert np.array_equal(spec.prior_low, theta_lower_bound(condition))
        assert np.array_equal(spec.prior_high, theta_upper_bound(condition))
        assert [k for k, lg in zip(keys, spec.log_rows) if not lg] == ([F_KEY] if condition == "FAB" else [F_KEY, R_KEY])
        assert all(k in spec.display for k in keys), [k for k in keys if k not in spec.display]
        assert spec.display[F_KEY] == "Basal dimer fraction (reference)"
        assert spec.display[R_KEY] == "Initial active-dimer count, relative to capacity"
        units = {e["KEY"]: tdr._unit_of(e) for e in spec.table}
        assert units[F_KEY] == "fraction (reference)" and units["count_total"] == "count"
        if condition == "INLB":
            assert units[R_KEY] == "fraction of capacity"
        assert units["relative_diffusivity_dimer"] == "ratio"
        assert not any("ratio" in units[k] for k in (F_KEY, R_KEY) if k in units)
        # the conversion bound to the table passes the fraction rows through unchanged
        u = np.array([(e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2.0 for e in spec.table])
        u[keys.index(F_KEY)] = 1.04
        assert spec.to_physical(u)[keys.index(F_KEY)] == 1.04
    dargs = tdr.build_parser("t").parse_args(["--condition", "FAB", "--total-time-seconds", "2"])
    dspec = tdr._temporal_dynamics_spec(detector_workflow(), dargs)
    assert dspec.keys == list(det.DETECTOR_PARAMETER_KEYS) and dspec.log_rows.all()
    assert [tdr._unit_of(e) for e in dspec.table] == ["px", "log-spread", "photons", "log-spread", "probability", "1/s"]
    print("PASS test_temporal_spec_binds_the_condition_with_names_and_units_of_the_fraction_rows")


def _temporal_run(condition, kinds, tmp, rng):
    """Run the temporal-dynamics engine on a fixture Experiment product (one log row and the basal
    fraction plotted) and return the report text and, per saved figure, every text on it."""
    import dataclasses
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from srm_and_sbi_monomer_dimer_alp import temporal_dynamics_runner as tdr
    exp = tmp / "Experiment.npz"
    np.savez_compressed(exp, **_experiment_product(condition, rng, kinds=kinds))
    real_spec, real_save = tdr._temporal_dynamics_spec, Figure.savefig
    figures = {}

    def spy_save(fig, fname, *a, **k):
        figures[Path(str(fname)).name] = [t.get_text() for t in fig.findobj(Text)]
        return real_save(fig, fname, *a, **k)

    def isolated(cfg, args, _real=real_spec):
        return dataclasses.replace(_real(cfg, args), npz_path=exp, recovery_npz=tmp / "absent.npz",
                                   fig_dir=tmp / "temporal_dynamics")

    tdr._temporal_dynamics_spec, Figure.savefig = isolated, spy_save
    try:
        args = tdr.build_parser("t").parse_args(["--condition", condition, "--total-time-seconds", "2",
                                                "--params", f"rate_dissociation,{F_KEY}"])
        assert tdr.run_temporal_dynamics(biology_workflow(), args) == 0
    finally:
        tdr._temporal_dynamics_spec, Figure.savefig = real_spec, real_save
    return (tmp / "temporal_dynamics" / "report.md").read_text(), figures


def test_temporal_figures_and_tables_name_the_estimator_and_mark_the_misspecification_test():
    """Recordings of the other condition in a product: every standalone temporal figure names the
    estimator's condition, carries the MISSPECIFICATION TEST line when it holds those recordings,
    and marks their curves `[misspecification test]`; the report opens with the warning and marks
    every table row and column of those recordings. With the estimator's own recordings only, the
    figures still name the estimator and nothing reads as a misspecification test."""
    import matplotlib
    matplotlib.use("Agg")
    for condition in CONDITIONS:
        other = "INLB" if condition == "FAB" else "FAB"
        own, foreign = f"MET-{condition}", f"MET-{other}"
        with tempfile.TemporaryDirectory() as tmp:
            report, figures = _temporal_run(condition, ("FAB", "INLB"), Path(tmp), np.random.default_rng(5))
        assert {"rate_dissociation_temporal.png", f"{F_KEY}_temporal_posterior_pooled.png",
                "window_drift_overview_FAB.png", "window_drift_overview_INLB.png"} <= set(figures), sorted(figures)
        for name, texts in figures.items():
            joined = "\n".join(texts)
            assert f"{own} estimator" in joined, (condition, name)
            holds_foreign = name != f"window_drift_overview_{condition}.png"
            assert ("MISSPECIFICATION TEST" in joined) == holds_foreign, (condition, name)
            curves = [t for t in texts if t.startswith(foreign)]       # legend entries and titles of that kind
            assert all(t.startswith(f"{foreign} [misspecification test]") for t in curves), (condition, name, curves)
            assert bool(curves) == holds_foreign, (condition, name)
            assert f"{own} [misspecification test]" not in joined, (condition, name)
        assert f"**MISSPECIFICATION TEST.** The {own} estimator is applied to {foreign} recordings." in report
        assert f"Estimator: **{own}**" in report
        table_lines = [ln for ln in report.splitlines() if ln.startswith("|") and foreign in ln]
        assert table_lines and all("[misspecification test]" in ln for ln in table_lines), condition
        assert f"| {foreign} [misspecification test] |" in report       # the drift rows of every point estimate
        # the estimator's own recordings only: named, never a misspecification test
        with tempfile.TemporaryDirectory() as tmp:
            report, figures = _temporal_run(condition, (condition,), Path(tmp), np.random.default_rng(5))
        assert figures and all(f"{own} estimator" in "\n".join(texts) for texts in figures.values())
        everything = report + "\n".join("\n".join(texts) for texts in figures.values())
        assert "misspecification" not in everything.lower(), condition
    print("PASS test_temporal_figures_and_tables_name_the_estimator_and_mark_the_misspecification_test")


def test_the_controls_mobile_mixture_weights_by_the_starting_composition_and_flags_windows():
    """The CD86 / CTLA-4 temporal companion's mobile-diffusion readout on a synthetic MAP grid:
    D_mix weights the mode-averaged diffusivities by the window-start composition of the helper
    (n_A = M_start, n_B = D_start), f_start equals f under MET-FAB and differs from it in some
    MET-INLB windows, and one window with f outside [0, 1] is flagged, counted and NaN in every
    mixture quantity, while a never-estimated window is NaN and not flagged."""
    import importlib.util
    from srm_and_sbi_monomer_dimer_alp.parameterization import population_from_theta
    from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import diffusion_coefficients, stationary_mode_law
    script = (Path(__file__).resolve().parents[1] / "Script_Bank" / "Analysis"
              / "SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.py")
    spec = importlib.util.spec_from_file_location("controls_temporal_dynamics", script)
    td = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(td)
    for condition in CONDITIONS:
        table, keys = parameterization_of(condition), parameter_keys_of(condition)
        rng = np.random.default_rng(5)
        flow = rng.uniform(theta_lower_bound(condition), theta_upper_bound(condition), size=(2, 3, 4, len(keys)))
        flow[1, 2, 3, keys.index(F_KEY)] = 1.2                  # one flagged window
        flow[0, 1, 0, :] = np.nan                               # one window never estimated
        grid = td._abs(flow, table)
        mix = td._mobile_mixture(grid, condition)
        assert int(mix["flagged"].sum()) == 1 and mix["flagged"][1, 2, 3] and not mix["flagged"][0, 1, 0]
        for name in ("d_mix", "d_a", "d_b", "f", "f_start"):
            assert np.isnan(mix[name][1, 2, 3]) and np.isnan(mix[name][0, 1, 0]), name
            assert int(np.isfinite(mix[name]).sum()) == 2 * 3 * 4 - 2, name
        assert grid[1, 2, 3, keys.index(F_KEY)] == 1.2, "the flagged value is never clipped"
        theta = grid[0, 0, 0]                                   # one window by hand
        pop = population_from_theta(theta, condition)
        coeff, pi = diffusion_coefficients(theta, condition), stationary_mode_law(theta, condition)
        d_a = (pi[0] * coeff["A_f"] + pi[1] * coeff["A_s"]) / (pi[0] + pi[1])
        d_b = (pi[0] * coeff["B_f"] + pi[1] * coeff["B_s"]) / (pi[0] + pi[1])
        n_a, n_b = float(pop.n_monomers_underlying_start), float(pop.n_dimers_underlying_start)
        assert np.isclose(mix["d_mix"][0, 0, 0], (n_a * d_a + n_b * d_b) / (n_a + n_b))
        assert np.isclose(mix["f_start"][0, 0, 0], n_b / (n_a + n_b))
        finite = np.isfinite(mix["f"])
        if condition == "FAB":
            assert np.allclose(mix["f_start"][finite], mix["f"][finite]), "no pair is allocated under MET-FAB"
        else:
            assert np.max(np.abs(mix["f_start"][finite] - mix["f"][finite])) > 0, "some InlB windows allocate pairs"
        _, summary = td._dmix_figure(mix, 2.0 * np.arange(4), ["CD86", "CTLA-4"])
        assert summary["CTLA-4 [Dimer]"]["n_flagged"] == 1 and summary["CD86 [Monomer]"]["n_flagged"] == 0
        assert summary["CD86 [Monomer]"]["n_windows"] == 11 and summary["CTLA-4 [Dimer]"]["n_windows"] == 12
    print("PASS test_the_controls_mobile_mixture_weights_by_the_starting_composition_and_flags_windows")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
