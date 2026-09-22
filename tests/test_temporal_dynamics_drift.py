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


def test_single_window_recordings_do_not_contribute():
    grid = _grid_two_increasing_recordings_in_slots_1_and_2()
    grid[0, 0, 0, 0] = 5.0                                     # one window only: no fit possible
    out = tdk.drift_statistics(grid, np.arange(4, dtype=float), to_physical=lambda u: u, log_rows=np.array([True]),
                               threshold=0.3)
    assert np.isclose(out["drift_sign_consistency"][0, 0], 1.0)
    assert np.isclose(out["drift_material_fraction"][0, 0], 0.5)
