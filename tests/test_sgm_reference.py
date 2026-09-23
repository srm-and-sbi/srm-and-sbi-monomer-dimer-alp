"""SGM correctness checks on small, hand-checkable clouds.

The per-observation SGM (``posterior_sgm``) is the sample medoid: the complete draw minimizing the
summed Euclidean distance to all draws, after dividing each estimator coordinate by its prior width.
These tests check the production ``evaluation.sample_geometric_median`` against sums worked out by
hand, check that the declared scaling is applied (and can change the answer), that the result is a
complete member of the cloud, and that duplicates and exact ties resolve deterministically to the
lowest index. The approximate large-collection method of the separate collection-level kernel is
not tested here.
"""
import numpy as np

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
from srm_and_sbi_monomer_dimer_alp.evaluation import prior_scale, sample_geometric_median


def test_a_line_with_a_tie_resolves_to_the_lowest_index():
    # summed distances: 0 -> 1+2+10 = 13; 1 -> 1+1+9 = 11; 2 -> 2+1+8 = 11; 10 -> 27
    cloud = np.array([[0.0], [1.0], [2.0], [10.0]])
    vec, idx = sample_geometric_median(cloud)
    assert idx == 1 and vec.tolist() == [1.0]


def test_the_declared_scaling_is_applied_and_changes_the_answer():
    # p0..p4 = (0,0) (4,0) (5,0) (0,1) (0,2)
    cloud = np.array([[0.0, 0.0], [4.0, 0.0], [5.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    # unscaled sums: p0 12.000, p1 13.595, p2 16.484, p3 11.222, p4 12.857 -> p3
    vec, idx = sample_geometric_median(cloud)
    assert idx == 3 and vec.tolist() == [0.0, 1.0]
    # x divided by 10: p0 3.900, p1 3.617, p2 3.780, p3 4.195, p4 7.102 -> p1
    vec, idx = sample_geometric_median(cloud, scale=np.array([10.0, 1.0]))
    assert idx == 1 and vec.tolist() == [4.0, 0.0]       # returned in the ORIGINAL coordinates


def test_the_result_is_a_complete_member_and_duplicates_are_deterministic():
    rng = np.random.default_rng(6)
    cloud = rng.normal(size=(40, 6)) * np.array([0.01, 0.3, 0.05, 0.6, 1.0, 0.02])
    vec, idx = sample_geometric_median(cloud, scale=prior_scale(det.DETECTOR_PARAMETERIZATION))
    assert np.array_equal(vec, cloud[idx])                # every coordinate from one draw
    # an exact duplicate of the medoid, placed before and after it: the lowest index wins, and
    # the same answer comes back on every call
    for j in (0, len(cloud) - 1):
        if j == idx:
            continue
        dup = cloud.copy(); dup[j] = dup[idx]
        results = {sample_geometric_median(dup, scale=prior_scale(det.DETECTOR_PARAMETERIZATION))[1]
                   for _ in range(5)}
        assert results == {min(idx, j)}


def test_production_scaling_is_the_prior_width_in_estimator_coordinates():
    widths = [float(e["PRIOR_RANGE"][1]) - float(e["PRIOR_RANGE"][0]) for e in det.DETECTOR_PARAMETERIZATION]
    assert np.allclose(prior_scale(det.DETECTOR_PARAMETERIZATION), widths)
    assert np.allclose(widths, [0.30, 0.75, 0.75, 0.75, 1.50, 1.00])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")


def test_the_selection_follows_the_geometry_not_the_row_order():
    # A cloud with a unique medoid selects the same vector under any row order, in the per-observation
    # SGM and in the collection-level kernel's exact path; with tied minima (the four corners of a
    # square are equally central) the documented rule takes the lowest index of the order supplied.
    from srm_and_sbi_monomer_dimer_alp.sample_geometric_median import sample_geometric_median as kernel
    rng = np.random.default_rng(11)
    cloud, width = rng.normal(size=(60, 4)), np.array([1.0, 2.0, 0.5, 1.5])
    z = cloud / width
    sums = np.sort(np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)).sum(1))
    assert sums[1] - sums[0] > 1e-6                                    # the medoid is unique
    vec, _ = sample_geometric_median(cloud, scale=width)
    k_idx, method = kernel(cloud, width)
    assert method == "exact_medoid" and np.array_equal(cloud[k_idx], vec)
    for _ in range(5):
        perm = rng.permutation(len(cloud))
        assert np.array_equal(sample_geometric_median(cloud[perm], scale=width)[0], vec)
        assert np.array_equal(cloud[perm][kernel(cloud[perm], width)[0]], vec)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        assert sample_geometric_median(square[order])[1] == 0
        assert kernel(square[order], np.ones(2))[0] == 0

