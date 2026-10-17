"""Regression tests for the within-recording drift kernel (``temporal_dynamics.drift_statistics``).

The fractions it reports (sign consistency, share of recordings over the material threshold)
must be taken over the recordings that contributed a fitted change, never over the grid's cell
slots: an unused cell index, a deselected recording or a failed estimate is not a "no drift" vote.
"""
import numpy as np

from srm_and_sbi_monomer_dimer_alp import temporal_dynamics as tdk


def _grid_two_increasing_recordings_in_slots_1_and_2():
    # (n_kinds=1, n_cells=3, n_chunks=4, dim=1); slot 0 never observed.
    grid = np.full((1, 3, 4, 1), np.nan)
    grid[0, 1, :, 0] = np.array([0.0, 0.2, 0.4, 0.6])       # rises 0.6 dex first to last
    grid[0, 2, :, 0] = np.array([1.0, 1.1, 1.2, 1.3])       # rises 0.3 dex (not over 0.3)
    return grid


def test_fractions_ignore_unobserved_cell_slots():
    grid = _grid_two_increasing_recordings_in_slots_1_and_2()
    out = tdk.drift_statistics(grid, np.arange(4, dtype=float), to_physical=lambda u: u, log_rows=np.array([True]),
                               threshold=0.3)
    # Both contributing recordings rise: sign consistency is 100 %, not 2/3.
    assert np.isclose(out["drift_sign_consistency"][0, 0], 1.0)
    # Exactly one of the two contributing recordings exceeds 0.3 dex: 50 %, not 1/3.
    assert np.isclose(out["drift_material_fraction"][0, 0], 0.5)
    # The fitted first-to-last change is the median over the two contributing recordings.
    assert np.isclose(out["drift_dex"][0, 0], np.median([0.6, 0.3]))


def test_linear_rows_report_absolute_change_and_no_dex_fold_or_material_fraction():
    """A log row rising +0.6 dex beside a linear row (a fraction) rising from 0 to 0.15, one cell
    starting at exactly 0: the linear row's drift_dex, material fraction and fold are NaN (a ratio
    of two values near 0 is unstable and a fraction has no multiplicative scale), its absolute
    change is 0.15, and nothing raises on the division by zero at the start."""
    grid = np.full((1, 2, 4, 2), np.nan)
    grid[0, 0, :, 0] = np.array([1.0, 1.2, 1.4, 1.6])          # log row: +0.6 dex first to last
    grid[0, 1, :, 0] = np.array([2.0, 2.2, 2.4, 2.6])
    grid[0, 0, :, 1] = np.array([0.0, 0.05, 0.10, 0.15])       # linear row: 0 -> 0.15, starts at exactly 0
    grid[0, 1, :, 1] = np.array([0.10, 0.15, 0.20, 0.25])
    log_rows = np.array([True, False])

    def to_physical(u):
        out = np.array(u, dtype=float, copy=True)
        out[..., 0] = 10.0 ** out[..., 0]
        return out

    out = tdk.drift_statistics(grid, np.arange(4, dtype=float), to_physical=to_physical, log_rows=log_rows,
                               threshold=0.3)
    assert np.isclose(out["drift_dex"][0, 0], 0.6) and np.isclose(out["drift_material_fraction"][0, 0], 1.0)
    assert np.isclose(out["drift_fold"][0, 0], 10 ** 0.6)
    assert np.isnan(out["drift_dex"][0, 1]) and np.isnan(out["drift_material_fraction"][0, 1])
    assert np.isnan(out["drift_fold"][0, 1]), "the fold is NaN for a linear row"
    assert np.isclose(out["drift_absolute"][0, 1], 0.15)
    assert np.isclose(out["drift_sign_consistency"][0, 1], 1.0)
    # the table formatter renders a fraction row in percentage points when the caller says so (the
    # Experiment and temporal-dynamics stages do) and the log row in dex, as before; by default a
    # linear row keeps its own units ("abs"), right for a linear row that is not a fraction
    rows = tdk.window_drift_rows({"MAP": grid}, ["MET-FAB"], ["log_key", "fraction_key"], to_physical, log_rows)
    headers, cells = tdk.format_drift_rows(rows, linear_unit="pp")
    by_key = {r[1]: r for r in cells}
    assert by_key["fraction_key"][4] == "+15.0 pp" and by_key["fraction_key"][5] == "0.0"
    assert by_key["fraction_key"][7] == "n/a"
    assert by_key["log_key"][4] == "+0.600 dex" and by_key["log_key"][5] == "0.000"
    _, cells_abs = tdk.format_drift_rows(rows)
    by_key_abs = {r[1]: r for r in cells_abs}
    assert by_key_abs["fraction_key"][4] == "+0.150 abs" and by_key_abs["log_key"] == by_key["log_key"]
    assert tdk.format_drift_rows(rows, linear_unit="abs")[1] == cells_abs
    try:
        tdk.format_drift_rows(rows, linear_unit="dex")
    except ValueError:
        pass
    else:
        raise AssertionError("a dex unit for a linear row must be refused")


def test_single_window_recordings_do_not_contribute():
    grid = _grid_two_increasing_recordings_in_slots_1_and_2()
    grid[0, 0, 0, 0] = 5.0                                     # one window only: no fit possible
    out = tdk.drift_statistics(grid, np.arange(4, dtype=float), to_physical=lambda u: u, log_rows=np.array([True]),
                               threshold=0.3)
    assert np.isclose(out["drift_sign_consistency"][0, 0], 1.0)
    assert np.isclose(out["drift_material_fraction"][0, 0], 0.5)


if __name__ == "__main__":
    import sys
    import time
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as exc:                      # noqa: BLE001 -- report every failure
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
