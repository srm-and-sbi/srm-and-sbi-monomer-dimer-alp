"""Median reference: implementation checks (phase 1 of the point-estimate validation).

The per-observation marginal median is the 0.50 level of ``posterior_quantiles``: numpy's default
linear interpolation (Hyndman & Fan type 7) of each coordinate's draws, taken in ESTIMATOR
coordinates; a physical value is ``to_physical`` of that quantile (quantile, then transform).
These tests check the production calculation (``evaluation.posterior_summary``) against an
independent calculation written from the definition, check that each row summarizes its own
parameter, and pin the coordinate convention -- including the finite-sample case in which
"transform, then quantile" gives a different number. No trained flow is needed: the sampler is
replaced by a fixed draw collection.
"""
from types import SimpleNamespace

import numpy as np
import torch

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
from srm_and_sbi_monomer_dimer_alp import evaluation as ev
from srm_and_sbi_monomer_dimer_alp.artifacts import assert_schema_compatible
from srm_and_sbi_monomer_dimer_alp.parameterization import to_physical

MEDIAN = list(ev.QUANTILE_LEVELS).index(0.50)


def _type7(sorted_col, p):
    """Independent linear-interpolation quantile (Hyndman & Fan type 7), from its definition:
    position h = (n - 1) p, then interpolate between the order statistics around h."""
    n = sorted_col.shape[0]
    h = (n - 1) * p
    lo = int(np.floor(h))
    hi = min(lo + 1, n - 1)
    return float(sorted_col[lo] + (h - lo) * (sorted_col[hi] - sorted_col[lo]))


def _summary_from(draws):
    """The production ``posterior_summary`` run on a fixed draw collection (sampler stubbed; the
    draws are float32, as the flow returns them)."""
    original = ev.collect_theta_prex
    ev.collect_theta_prex = lambda *a, **k: torch.as_tensor(np.asarray(draws, dtype=np.float32))
    try:
        posterior = SimpleNamespace(set_default_x=lambda x: None, posterior_estimator=object())
        return np.asarray(ev.posterior_summary(
            posterior, np.zeros((4, 8, 8), dtype=np.uint8), torch.device("cpu"),
            torch.device("cpu"), int(np.asarray(draws).shape[0]), 100,
            quantiles=ev.QUANTILE_LEVELS), dtype=np.float64)
    finally:
        ev.collect_theta_prex = original


def test_quantiles_match_an_independent_type7_calculation_for_odd_and_even_counts():
    rng = np.random.default_rng(31)
    center = np.array([0.15, -0.6, 2.4, -0.4, -1.3, 0.5])
    spread = np.array([0.02, 0.1, 0.3, 0.6, 1.2, 0.05])          # unequal coordinate spreads
    for n in (7, 8, 999, 1000, 1001):
        draws = (rng.normal(size=(n, 6)) * spread + center).astype(np.float32)
        draws[: max(2, n // 10), 3] = draws[0, 3]                 # heavy ties in one coordinate
        summary = _summary_from(draws)
        assert summary.shape == (6, len(ev.QUANTILE_LEVELS))
        for j in range(6):
            col = np.sort(draws[:, j].astype(np.float64))
            for k, p in enumerate(ev.QUANTILE_LEVELS):
                expect = _type7(col, p)
                assert abs(summary[j, k] - expect) <= 1e-6 * max(1.0, abs(expect)), (n, j, p)
            # the 0.50 level is the sample median: the middle draw for an odd count, the midpoint
            # of the two middle draws for an even count
            mid = col[n // 2] if n % 2 else 0.5 * (col[n // 2 - 1] + col[n // 2])
            assert abs(summary[j, MEDIAN] - mid) <= 1e-6 * max(1.0, abs(mid)), (n, j)


def test_each_row_summarizes_its_own_parameter_and_a_permuted_schema_is_refused():
    rng = np.random.default_rng(32)
    n = 101
    draws = np.tile(np.arange(6, dtype=np.float64), (n, 1)) + 1e-3 * rng.normal(size=(n, 6))
    summary = _summary_from(draws)
    assert np.allclose(summary[:, MEDIAN], np.arange(6), atol=5e-3)   # row i <- column i
    # the draws' column order is the estimator's stored parameter order, which load_estimator
    # checks against the workflow's keys (content AND order) before anything is sampled
    keys = list(det.DETECTOR_PARAMETER_KEYS)
    assert_schema_compatible({"parameter_keys": keys}, expected_parameter_keys=keys)
    try:
        assert_schema_compatible({"parameter_keys": keys[::-1]}, expected_parameter_keys=keys)
    except ValueError:
        pass
    else:
        raise AssertionError("a permuted parameter order was accepted")


def test_quantiles_are_taken_in_estimator_coordinates_then_transformed():
    table = det.DETECTOR_PARAMETERIZATION                  # every learnable detector row is a log row
    assert "estimator" in ev.POINT_ESTIMATES["median"]["coordinates"]
    # Even count: the two middle physical values are 1 and 9 in every coordinate.
    log_draws = np.log10(np.array([[0.5] * 6, [1.0] * 6, [9.0] * 6, [20.0] * 6]))
    summary = _summary_from(log_draws)
    quantile_then_transform = to_physical(summary[:, MEDIAN], table)      # the production convention
    transform_then_quantile = np.median(to_physical(log_draws, table), axis=0)
    assert np.allclose(quantile_then_transform, 3.0, rtol=1e-5)          # sqrt(1 x 9)
    assert np.allclose(transform_then_quantile, 5.0)                     # (1 + 9) / 2
    # Odd count: both conventions select the same draw, so they agree.
    odd = log_draws[:3]
    s_odd = _summary_from(odd)
    assert np.allclose(to_physical(s_odd[:, MEDIAN], table),
                       np.median(to_physical(odd, table), axis=0), rtol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
