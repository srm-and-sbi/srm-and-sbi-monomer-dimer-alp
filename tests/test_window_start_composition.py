"""Window-start composition: the reference probe-bound population, the initial allocation and
the placement (PROJECT_CONTEXT.md sec. 2, *Modeling assumptions of the MET model*, items 4, 9, 11).

Run directly from the repository root::

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/test_window_start_composition.py

The reference population at zero allocation reproduces the integers of the initializer that
preceded the allocation (release 0.1.39, commit 76050ac) EXACTLY: a frozen copy of that
initializer (never edited) and the SHA-256 digests of its output on a fixed grid, recorded
before the revision, are the golden values. The allocation and the derived quantities are
checked against an independent scalar oracle transcribed from the specification.
"""
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MACHINE_PROFILE", "mars_pc")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srm_and_sbi_monomer_dimer_alp import parameterization as par                 # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import (                       # noqa: E402
    FRACTION_BOX, InitialComposition, PopulationRealization, declared_row_values, occupancy_of,
    population_from_theta, population_valid, realize_initial_composition, realize_population,
    two_probe_share_of,
)

CONDITIONS = ("FAB", "INLB")
F_KEY = "fraction_basal_dimers_reference"
R_KEY = "relative_count_active_dimers_initial"
N_KEY = "count_total"


# ---- The 0.1.39 initializer, frozen (commit 76050ac; never edited) --------------------------

def _realize_0139(count_total, ratio_dimer_monomer, p, ligand_classes):
    """The reference population of release 0.1.39 from (N_total, r = n_B / n_A): the
    integers the revised helper must reproduce at zero allocation, with f = r / (1 + r)."""
    r = float(ratio_dimer_monomer)
    n_total = float(count_total)
    n_monomers_true = n_total / (1.0 + 2.0 * r)
    n_dimers_true = r * n_total / (1.0 + 2.0 * r)
    n_dimers = int(round(p * (2.0 - p) * n_dimers_true))
    n_two = int(min(round(p * p * n_dimers_true), n_dimers))
    n_one = n_dimers - n_two
    n_monomers = int(round(p * n_monomers_true))
    if ligand_classes:
        n_subunits = n_monomers + n_one + 2 * n_two
    else:
        n_subunits = n_monomers + 2 * n_dimers
    return n_monomers, n_dimers, n_one, n_two, n_subunits


# Recorded at 76050ac (scratch script golden_0139.py): the occupancies, and the SHA-256 of the
# int64 array (201 x 98 x 5) of (monomers, dimers, one-probe, two-probe, subunits) over
# N = 10**linspace(3, 5, 201) and f = linspace(0, 0.95, 96) + [0.05, 0.25], per condition.
GOLDEN_OCCUPANCY = {"FAB": 0.014769810338937442, "INLB": 0.047619047619047616}
GOLDEN_DIGEST = {
    "FAB": "57f9b39badeccdc53ae9f3c21d01d317b6356c28f26c60850352ec10b674356c",
    "INLB": "b355e04e62e992000426f85141fe6adbc7801898ce5e220ad54c3f2b9dec2bb9",
}
GOLDEN_SUM = {"FAB": 19862704, "INLB": 51929246}
# (condition, N, f, monomers, dimers, one-probe, two-probe, subunits) at zero allocation
GOLDEN_ROWS = [
    ("FAB", 1e3, 0.0, 15, 0, 0, 0, 15), ("FAB", 1e3, 0.05, 13, 1, 1, 0, 15), ("FAB", 1e3, 0.25, 9, 6, 6, 0, 21),
    ("FAB", 1e3, 0.5, 5, 10, 10, 0, 25), ("FAB", 1e3, 0.9, 1, 14, 14, 0, 29),
    ("FAB", 1e4, 0.0, 148, 0, 0, 0, 148), ("FAB", 1e4, 0.05, 134, 14, 14, 0, 162), ("FAB", 1e4, 0.25, 89, 59, 59, 0, 207),
    ("FAB", 1e4, 0.5, 49, 98, 97, 1, 245), ("FAB", 1e4, 0.9, 8, 139, 138, 1, 286),
    ("FAB", 1e5, 0.0, 1477, 0, 0, 0, 1477), ("FAB", 1e5, 0.05, 1336, 140, 139, 1, 1616),
    ("FAB", 1e5, 0.25, 886, 586, 582, 4, 2058), ("FAB", 1e5, 0.5, 492, 977, 970, 7, 2446),
    ("FAB", 1e5, 0.9, 78, 1389, 1379, 10, 2856),
    ("INLB", 1e3, 0.0, 48, 0, 0, 0, 48), ("INLB", 1e3, 0.05, 43, 4, 4, 0, 47), ("INLB", 1e3, 0.25, 29, 19, 19, 0, 48),
    ("INLB", 1e3, 0.5, 16, 31, 30, 1, 48), ("INLB", 1e3, 0.9, 3, 44, 43, 1, 48),
    ("INLB", 1e4, 0.0, 476, 0, 0, 0, 476), ("INLB", 1e4, 0.05, 431, 44, 43, 1, 476),
    ("INLB", 1e4, 0.25, 286, 186, 181, 5, 477), ("INLB", 1e4, 0.5, 159, 310, 302, 8, 477),
    ("INLB", 1e4, 0.9, 25, 440, 429, 11, 476),
    ("INLB", 1e5, 0.0, 4762, 0, 0, 0, 4762), ("INLB", 1e5, 0.05, 4308, 443, 432, 11, 4762),
    ("INLB", 1e5, 0.25, 2857, 1859, 1814, 45, 4761), ("INLB", 1e5, 0.5, 1587, 3099, 3023, 76, 4762),
    ("INLB", 1e5, 0.9, 251, 4404, 4297, 107, 4762),
]
N_GRID = 10.0 ** np.linspace(3.0, 5.0, 201)
F_GRID = np.concatenate([np.linspace(0.0, 0.95, 96), [0.05, 0.25]])


def _oracle(N, f, R, p, ligand_classes):
    """The specification, transcribed with Python scalars: reference population, then the
    allocation; returns a dict of the integers and the derived quantities."""
    m_ref = N * (1.0 - f) / (1.0 + f)
    d_ref = N * f / (1.0 + f)
    d_sim = int(round(p * (2.0 - p) * d_ref))
    b2_ref = int(min(round(p * p * d_ref), d_sim))
    b1 = d_sim - b2_ref
    a_ref = int(round(p * m_ref))
    cap = a_ref // 2 if ligand_classes else 0
    c = int(round(R * cap))
    a_start = a_ref - 2 * c
    b2_start = b2_ref + c
    n_r = a_ref + b1 + 2 * b2_ref if ligand_classes else a_ref + 2 * d_sim
    m_start, d_start = m_ref - 2.0 * c, d_ref + c
    return dict(m_ref=m_ref, d_ref=d_ref, d_sim=d_sim, b2_ref=b2_ref, b1=b1, a_ref=a_ref, cap=cap, c=c,
                a_start=a_start, b2_start=b2_start, n_dimers=b1 + b2_start, n_r=n_r,
                m_start=m_start, d_start=d_start,
                f_start=(d_start / (m_start + d_start)) if (m_start + d_start) > 0 else float("nan"),
                active_share=(c / d_start) if d_start > 0 else float("nan"))


def _ligand(condition):
    return bool(par.PARAMETERS.simulation.rds.condition_setting(condition).ligand_classes)


def _declared_r(condition):
    return declared_row_values(condition).get(R_KEY)


def _columns(pop):
    """(monomers, dimers, one-probe, two-probe, subunits) of a realization, int64."""
    return np.stack([pop.n_monomers, pop.n_dimers, pop.n_dimers_one_probe, pop.n_dimers_two_probe,
                     pop.n_subunits], axis=-1).astype(np.int64)


# ---- tests --------------------------------------------------------------------------------

def test_occupancies_and_declared_rows_are_the_decided_values():
    for cond in CONDITIONS:
        assert occupancy_of(cond) == GOLDEN_OCCUPANCY[cond], (cond, occupancy_of(cond))
        p = occupancy_of(cond)
        assert two_probe_share_of(cond) == p / (2.0 - p)
    assert declared_row_values("FAB") == {R_KEY: 0.0}
    assert declared_row_values("INLB") == {}
    assert FRACTION_BOX == (0.0, 1.0)
    assert par.parameter_keys_of("FAB") == [k for k in par.parameter_keys_of("INLB") if k != R_KEY]
    assert par.parameter_keys_of("INLB")[1] == F_KEY and par.parameter_keys_of("INLB")[-1] == R_KEY


def test_reference_population_reproduces_the_0139_integers_on_the_grid():
    for cond in CONDITIONS:
        p, lig = occupancy_of(cond), _ligand(cond)
        NN, FF = np.meshgrid(N_GRID, F_GRID, indexing="ij")
        pop = realize_population(count_total=NN, fraction_basal_dimers_reference=FF,
                                 relative_count_active_dimers_initial=np.zeros_like(NN), condition=cond)
        arr = _columns(pop)
        assert arr.shape == (201, 98, 5)
        assert hashlib.sha256(arr.tobytes()).hexdigest() == GOLDEN_DIGEST[cond], cond
        assert int(arr.sum()) == GOLDEN_SUM[cond]
        frozen = np.array([[_realize_0139(N, f / (1.0 - f), p, lig) for f in F_GRID] for N in N_GRID],
                          dtype=np.int64)
        assert np.array_equal(arr, frozen), cond
        # the frozen copy itself hashes to the recorded digest: the golden values are self-consistent
        assert hashlib.sha256(frozen.tobytes()).hexdigest() == GOLDEN_DIGEST[cond]


def test_reference_population_on_random_draws_and_the_literal_rows():
    rng = np.random.default_rng(20261015)
    for cond in CONDITIONS:
        p, lig = occupancy_of(cond), _ligand(cond)
        N = 10.0 ** rng.uniform(3.0, 5.0, size=100_000)
        f = rng.uniform(0.0, 1.0, size=100_000)
        f[f >= 1.0] = 0.999
        pop = realize_population(count_total=N, fraction_basal_dimers_reference=f,
                                 relative_count_active_dimers_initial=0.0, condition=cond)
        arr = _columns(pop)
        frozen = np.array([_realize_0139(n, x / (1.0 - x), p, lig) for n, x in zip(N, f)], dtype=np.int64)
        assert np.array_equal(arr, frozen), f"{cond}: {int((arr != frozen).any(axis=1).sum())} mismatches"
    for cond, N, f, *expected in GOLDEN_ROWS:
        c = realize_initial_composition(count_total=N, fraction_basal_dimers_reference=f,
                                        relative_count_active_dimers_initial=0.0, condition=cond)
        got = (c.n_monomers, c.n_dimers, c.n_dimers_one_probe, c.n_dimers_two_probe, c.n_subunits)
        assert got == tuple(expected), (cond, N, f, got, expected)
        assert all(isinstance(v, int) for v in got)


def test_both_fraction_endpoints_are_inside_the_prior():
    for cond in CONDITIONS:
        R = 1.0 if cond == "INLB" else 0.0
        at0 = realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=0.0,
                                          relative_count_active_dimers_initial=R, condition=cond)
        assert at0.n_dimers_reference == 0 and at0.n_dimers_one_probe == 0 and at0.n_dimers_two_probe_reference == 0
        at1 = realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=1.0,
                                          relative_count_active_dimers_initial=R, condition=cond)
        assert at1.n_monomers_reference == 0 and at1.capacity == 0 and at1.n_pairs_allocated == 0
        assert at1.n_monomers == 0
    fab1 = realize_initial_composition(count_total=1e5, fraction_basal_dimers_reference=1.0,
                                       relative_count_active_dimers_initial=0.0, condition="FAB")
    assert (fab1.n_monomers, fab1.n_dimers, fab1.n_subunits) == (0, 1466, 2932)
    fab0 = realize_initial_composition(count_total=1e5, fraction_basal_dimers_reference=0.0,
                                       relative_count_active_dimers_initial=0.0, condition="FAB")
    assert (fab0.n_monomers, fab0.n_dimers, fab0.n_subunits) == (1477, 0, 1477)


def test_allocation_worked_numbers():
    def inlb(N, f, R):
        return realize_initial_composition(count_total=N, fraction_basal_dimers_reference=f,
                                           relative_count_active_dimers_initial=R, condition="INLB")
    c = inlb(1e4, 0.2, 0.0)
    assert (c.n_monomers_reference, c.n_dimers_one_probe, c.n_dimers_two_probe_reference, c.capacity) == (317, 151, 4, 158)
    assert (c.n_pairs_allocated, c.n_monomers, c.n_dimers_two_probe) == (0, 317, 4)
    c = inlb(1e4, 0.2, 0.5)
    assert (c.n_pairs_allocated, c.n_monomers, c.n_dimers_one_probe, c.n_dimers_two_probe, c.n_subunits) == (79, 159, 151, 83, 476)
    c = inlb(1e4, 0.2, 1.0)
    assert (c.n_pairs_allocated, c.n_monomers, c.n_dimers_two_probe) == (158, 1, 162), "an odd A_ref leaves one monomer"
    c = inlb(1e4, 0.5, 0.5)
    assert (c.n_monomers, c.n_dimers_one_probe, c.n_dimers_two_probe, c.capacity, c.n_pairs_allocated) == (79, 302, 48, 79, 40)
    c = inlb(1e5, 0.25, 0.5)
    assert (c.n_monomers_reference, c.capacity, c.n_pairs_allocated, c.n_monomers, c.n_dimers_two_probe) == (2857, 1428, 714, 1429, 759)
    assert inlb(1e5, 0.25, 1.0).n_monomers == 1
    c = inlb(1030, 0.0, 1.0)
    assert (c.n_monomers_reference, c.n_monomers, c.n_dimers_two_probe, c.n_dimers_one_probe) == (49, 1, 24, 0)
    c = inlb(1000, 0.0, 1.0)
    assert (c.n_monomers, c.n_dimers_two_probe) == (0, 24), "a starting state without monomers is allowed"
    c = inlb(1e4, 0.25, 0.5)
    assert (c.n_monomers, c.n_dimers_one_probe, c.n_dimers_two_probe, c.n_pairs_allocated) == (142, 181, 77, 72)
    c = inlb(1e4, 0.25, 1.0)
    assert (c.n_monomers, c.n_dimers_one_probe, c.n_dimers_two_probe) == (0, 181, 148)
    assert abs(c.fraction_basal_dimers_start - 0.273) < 5e-4 and abs(c.active_share - 0.067) < 5e-4
    c = inlb(25, 1.0, 1.0)
    assert (c.capacity, c.n_pairs_allocated, c.n_monomers, c.n_dimers) == (0, 0, 0, 1), "no pair available"
    # ordinary rounding, no B2-share floor: 0 two-probe dimers of 4
    c = inlb(1e3, 0.05, 0.0)
    assert (c.n_dimers, c.n_dimers_two_probe) == (4, 0)


def test_conservation_b1_unchanged_and_the_derived_underlying_quantities():
    NN, FF, RR = np.meshgrid(10.0 ** np.linspace(3, 5, 41), np.linspace(0.0, 1.0, 21),
                             np.array([0.0, 0.25, 0.5, 0.75, 1.0]), indexing="ij")
    pop = realize_population(count_total=NN, fraction_basal_dimers_reference=FF,
                             relative_count_active_dimers_initial=RR, condition="INLB")
    base = realize_population(count_total=NN, fraction_basal_dimers_reference=FF,
                              relative_count_active_dimers_initial=np.zeros_like(RR), condition="INLB")
    assert np.array_equal(pop.n_subunits, base.n_subunits), "the allocation conserves N_R exactly"
    assert np.array_equal(pop.n_dimers_one_probe, base.n_dimers_one_probe), "B1 is unchanged"
    assert np.array_equal(pop.n_monomers + 2 * pop.n_pairs_allocated, pop.n_monomers_reference)
    assert np.array_equal(pop.n_dimers_two_probe - pop.n_pairs_allocated, pop.n_dimers_two_probe_reference)
    assert np.all(pop.n_pairs_allocated >= 0) and np.all(pop.n_pairs_allocated <= pop.capacity)
    assert np.all(pop.n_monomers >= 0)
    assert np.array_equal(pop.n_dimers, pop.n_dimers_one_probe + pop.n_dimers_two_probe)
    assert np.allclose(pop.n_monomers_underlying_start + 2.0 * pop.n_dimers_underlying_start, NN, rtol=1e-12)
    assert np.allclose(pop.n_monomers_underlying_start, pop.n_monomers_underlying_reference - 2.0 * pop.n_pairs_allocated)
    assert np.allclose(pop.n_dimers_underlying_start, pop.n_dimers_underlying_reference + pop.n_pairs_allocated)
    d_start, m_start = pop.n_dimers_underlying_start, pop.n_monomers_underlying_start
    with np.errstate(invalid="ignore", divide="ignore"):
        expect = d_start / (m_start + d_start)
    assert np.allclose(pop.fraction_basal_dimers_start, expect, equal_nan=True)
    share = np.where(d_start > 0, pop.n_pairs_allocated / np.where(d_start > 0, d_start, 1.0), np.nan)
    assert np.allclose(pop.active_share, share, equal_nan=True)
    # c is monotone in R at fixed (N, f)
    assert np.all(np.diff(pop.n_pairs_allocated, axis=-1) >= 0)
    # the underlying monomers at the start are never A_start / p where pairs were allocated
    p = occupancy_of("INLB")
    with_pairs = pop.n_pairs_allocated > 0
    assert with_pairs.any()
    assert not np.allclose(m_start[with_pairs], pop.n_monomers[with_pairs] / p, rtol=1e-3)
    # the Fab reference IS the starting population
    fab = realize_population(count_total=NN[..., 0], fraction_basal_dimers_reference=FF[..., 0],
                             relative_count_active_dimers_initial=0.0, condition="FAB")
    assert np.all(fab.capacity == 0) and np.all(fab.n_pairs_allocated == 0)
    assert np.array_equal(fab.n_monomers, fab.n_monomers_reference)
    assert np.array_equal(fab.n_dimers_two_probe, fab.n_dimers_two_probe_reference)
    assert np.allclose(fab.fraction_basal_dimers_start, FF[..., 0])


def test_fab_declares_zero_and_other_values_are_refused():
    ok = realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=0.3,
                                     relative_count_active_dimers_initial=0.0, condition="FAB")
    assert ok.relative_count_active_dimers_initial == 0.0
    for bad in (0.3, 1.0, -0.0001):
        try:
            realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=0.3,
                                        relative_count_active_dimers_initial=bad, condition="FAB")
        except ValueError as exc:
            assert "FAB" in str(exc) and R_KEY in str(exc)
        else:
            raise AssertionError(f"FAB accepted an allocation of {bad}")
    valid = population_valid(count_total=np.array([1e4, 1e4, 1e4]), fraction_basal_dimers_reference=np.array([0.3, 0.3, 0.3]),
                             relative_count_active_dimers_initial=np.array([0.0, 0.3, 1e-13]), condition="FAB")
    assert valid.tolist() == [True, False, True]


def test_invalid_inputs_are_refused_and_never_clipped():
    bad_cases = [
        dict(count_total=float("nan"), fraction_basal_dimers_reference=0.2, relative_count_active_dimers_initial=0.5),
        dict(count_total=0.0, fraction_basal_dimers_reference=0.2, relative_count_active_dimers_initial=0.5),
        dict(count_total=-10.0, fraction_basal_dimers_reference=0.2, relative_count_active_dimers_initial=0.5),
        dict(count_total=1e4, fraction_basal_dimers_reference=-0.01, relative_count_active_dimers_initial=0.5),
        dict(count_total=1e4, fraction_basal_dimers_reference=1.02, relative_count_active_dimers_initial=0.5),
        dict(count_total=1e4, fraction_basal_dimers_reference=float("inf"), relative_count_active_dimers_initial=0.5),
        dict(count_total=1e4, fraction_basal_dimers_reference=0.2, relative_count_active_dimers_initial=1.2),
        dict(count_total=1e4, fraction_basal_dimers_reference=0.2, relative_count_active_dimers_initial=-0.2),
        dict(count_total=1e4, fraction_basal_dimers_reference=0.2, relative_count_active_dimers_initial=float("nan")),
    ]
    for kw in bad_cases:
        for fn in (realize_initial_composition, realize_population):
            try:
                fn(condition="INLB", **kw)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{fn.__name__} accepted {kw}")
        assert not population_valid(condition="INLB", **kw)
    # one invalid element inside an array refuses the whole vectorized call (callers mask first)
    try:
        realize_population(count_total=np.array([1e4, 1e4]), fraction_basal_dimers_reference=np.array([0.2, 1.5]),
                           relative_count_active_dimers_initial=0.5, condition="INLB")
    except ValueError as exc:
        assert "1 of 2" in str(exc) or "invalid" in str(exc)
    else:
        raise AssertionError("an out-of-range fraction was accepted")
    mask = population_valid(count_total=np.array([1e4, 1e4]), fraction_basal_dimers_reference=np.array([0.2, 1.5]),
                            relative_count_active_dimers_initial=0.5, condition="INLB")
    assert mask.tolist() == [True, False]
    try:
        realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=0.2,
                                    relative_count_active_dimers_initial=0.5, condition="XYZ")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown condition was accepted")


def test_vectorized_helper_equals_the_one_draw_initializer_and_the_oracle_including_ties():
    rng = np.random.default_rng(7)
    for cond in CONDITIONS:
        p, lig = occupancy_of(cond), _ligand(cond)
        N = 10.0 ** rng.uniform(3.0, 5.0, size=3000)
        f = rng.uniform(0.0, 1.0, size=3000)
        R = rng.uniform(0.0, 1.0, size=3000) if lig else np.zeros(3000)
        pop = realize_population(count_total=N, fraction_basal_dimers_reference=f,
                                 relative_count_active_dimers_initial=R, condition=cond)
        for i in range(3000):
            one = realize_initial_composition(count_total=N[i], fraction_basal_dimers_reference=f[i],
                                              relative_count_active_dimers_initial=R[i], condition=cond)
            o = _oracle(float(N[i]), float(f[i]), float(R[i]), p, lig)
            for attr, key in (("n_monomers", "a_start"), ("n_dimers", "n_dimers"), ("n_dimers_one_probe", "b1"),
                              ("n_dimers_two_probe", "b2_start"), ("n_subunits", "n_r"), ("capacity", "cap"),
                              ("n_pairs_allocated", "c"), ("n_monomers_reference", "a_ref"),
                              ("n_dimers_two_probe_reference", "b2_ref"), ("n_dimers_reference", "d_sim")):
                assert int(getattr(pop, attr)[i]) == o[key] == getattr(one, attr), (cond, i, attr)
            assert np.isclose(one.fraction_basal_dimers_start, o["f_start"], equal_nan=True)
            assert np.isclose(one.active_share, o["active_share"], equal_nan=True)
            assert isinstance(one, InitialComposition) and isinstance(pop, PopulationRealization)
    # exact .5 ties through the private core at p = 0.5 (both roundings are half to even)
    core = par._realize_population_arrays
    for lig in (True, False):
        for N in range(1, 401):
            for f in (1.0 / 3.0, 0.6, 0.0, 1.0):
                for R in (0.0, 0.5, 1.0):
                    if not lig and R != 0.0:
                        continue
                    got = core(np.float64(N), np.float64(f), np.float64(R), 0.5, lig)
                    o = _oracle(float(N), float(f), float(R), 0.5, lig)
                    assert (int(got["n_monomers"]), int(got["n_dimers_one_probe"]), int(got["n_dimers_two_probe"]),
                            int(got["n_subunits"]), int(got["n_pairs_allocated"])) == (o["a_start"], o["b1"], o["b2_start"], o["n_r"], o["c"]), (lig, N, f, R)
    assert int(np.round(0.5 * 21)) == 10 and int(np.round(0.5 * 19)) == 10 and int(np.round(0.5 * 2381)) == 1190


def test_population_from_theta_aligns_by_key_for_both_conditions():
    from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import theta_by_key
    rng = np.random.default_rng(3)
    for cond in CONDITIONS:
        table = par.parameterization_of(cond)
        keys = par.parameter_keys_of(cond)
        theta_flow = rng.uniform(par.theta_lower_bound(cond), par.theta_upper_bound(cond), size=(50, len(table)))
        theta = par.to_physical(theta_flow, table)
        pop = population_from_theta(theta, cond)
        i_n, i_f = keys.index(N_KEY), keys.index(F_KEY)
        R = theta[:, keys.index(R_KEY)] if R_KEY in keys else np.zeros(50)
        direct = realize_population(count_total=theta[:, i_n], fraction_basal_dimers_reference=theta[:, i_f],
                                    relative_count_active_dimers_initial=R, condition=cond)
        assert np.array_equal(_columns(pop), _columns(direct))
        by_key = theta_by_key(theta[0], cond)
        assert list(by_key) == [e["KEY"] for e in par.MASTER_PARAMETERIZATION], "both conditions map to the master's keys"
        assert by_key[R_KEY] == (0.0 if cond == "FAB" else theta[0, keys.index(R_KEY)])
        try:
            theta_by_key(theta[0][:-1] if cond == "INLB" else np.append(theta[0], 0.5), cond)
        except ValueError:
            pass
        else:
            raise AssertionError("a wrong width was accepted")
    fab_keys, inlb_keys = par.parameter_keys_of("FAB"), par.parameter_keys_of("INLB")
    assert fab_keys == inlb_keys[:11] and len(fab_keys) == 11 and len(inlb_keys) == 12


def test_placement_census_matches_the_starting_counts():
    """build_system + build_simulation place the STARTING counts (no time stepping)."""
    from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import build_simulation, build_system
    rds = par.PARAMETERS.simulation.rds

    def census(cond, N, f, R):
        keys = par.parameter_keys_of(cond)
        table = par.parameterization_of(cond)
        theta = np.array([par.prior_center(e) for e in table], dtype=float)
        theta[keys.index(N_KEY)] = N
        theta[keys.index(F_KEY)] = f
        if R_KEY in keys:
            theta[keys.index(R_KEY)] = R
        smut = build_simulation(build_system(theta, cond, verbose=False), theta, cond, seed=1, verbose=False)
        counts = {}
        for q in smut.current_particles:
            counts[q.type] = counts.get(q.type, 0) + 1
        molecule = rds.molecule_of_type
        per_species = {}
        for name, n in counts.items():
            per_species[molecule[name]] = per_species.get(molecule[name], 0) + n
        subunits = sum(n * rds.subunit_counts_per_type[rds.particle_type_names.index(name)] for name, n in counts.items())
        return per_species, subunits

    t0 = time.time()
    cases = [("INLB", 1e5, 0.25, 0.5), ("INLB", 1030, 0.0, 1.0), ("INLB", 1000, 1.0, 1.0),
             ("FAB", 1e4, 0.0, 0.0), ("FAB", 1e4, 0.5, 0.0), ("FAB", 1e4, 1.0, 0.0)]
    for cond, N, f, R in cases:
        expect = realize_initial_composition(count_total=N, fraction_basal_dimers_reference=f,
                                             relative_count_active_dimers_initial=R, condition=cond)
        per_species, subunits = census(cond, N, f, R)
        if cond == "INLB":
            assert per_species.get("A", 0) == expect.n_monomers, (cond, N, f, R, per_species)
            assert per_species.get("B1", 0) == expect.n_dimers_one_probe
            assert per_species.get("B2", 0) == expect.n_dimers_two_probe
        else:
            assert per_species.get("A", 0) == expect.n_monomers and per_species.get("B", 0) == expect.n_dimers
        assert subunits == expect.n_subunits
    first = realize_initial_composition(count_total=1e5, fraction_basal_dimers_reference=0.25,
                                        relative_count_active_dimers_initial=0.5, condition="INLB")
    assert (first.n_monomers_reference, first.capacity, first.n_pairs_allocated) == (2857, 1428, 714)
    print(f"  (placement census of {len(cases)} systems in {time.time() - t0:.1f} s)")


def test_labeling_record_of_a_synthetic_frame0_lineage_carries_the_starting_counts():
    from srm_and_sbi_monomer_dimer_alp import labeling as lab
    from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import SubunitLineage
    rds = par.PARAMETERS.simulation.rds
    type_names = rds.particle_type_names
    ranks = {name: rank for rank, name in enumerate(type_names)}

    class _Tray:
        particle_types = ranks

    comp = realize_initial_composition(count_total=1e4, fraction_basal_dimers_reference=0.25,
                                       relative_count_active_dimers_initial=1.0, condition="INLB")
    assert comp.n_pairs_allocated > 0 and comp.n_dimers_two_probe > comp.n_dimers_two_probe_reference
    rng = np.random.default_rng(11)
    modes = ["f", "s", "i"]
    n_a, n_one, n_two = comp.n_monomers, comp.n_dimers_one_probe, comp.n_dimers_two_probe
    host_index = np.concatenate([np.arange(n_a), n_a + np.arange(n_one), n_a + n_one + np.repeat(np.arange(n_two), 2)])
    host_rank = np.concatenate([
        np.array([ranks[f"A_{m}"] for m in rng.choice(modes, size=n_a)], dtype=np.int64),
        np.array([ranks[f"B1_{m}"] for m in rng.choice(modes, size=n_one)], dtype=np.int64),
        np.repeat(np.array([ranks[f"B2_{m}"] for m in rng.choice(modes, size=n_two)], dtype=np.int64), 2)])
    lineage = SubunitLineage(host_index=host_index[None, :].astype(np.int64), host_rank=host_rank[None, :].astype(np.int64),
                             soul_ids=np.arange(n_a + n_one + n_two, dtype=np.int64))
    assert lineage.host_index.shape[1] == comp.n_subunits
    plan = lab.resolve_labeling("INLB")
    dyes, row = lab.label_trajectory(plan, _Tray(), lineage, lab.labeling_rng(0))
    col = dict(zip(lab.LABELING_SET_COLUMNS, row))
    assert col["n_subunits"] == comp.n_subunits
    assert col["monomers_0"] == n_a and col["dimers_0"] == n_one + n_two
    assert col["dimers_one_probe_0"] == n_one and col["dimers_two_probe_0"] == n_two
    assert col["two_probe_share"] == plan.two_probe_share == two_probe_share_of("INLB"), "the declared share, not the starting share"
    assert col["occupancy_dimer"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)")
