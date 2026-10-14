"""The placement geometry of the model-structure audit: the geometric fraction of daughter pairs within
the reaction distance for an isotropic fission axis in the thin slab, and the identity-checked count of
same-pair re-fusions R8 reports. No simulation. Run directly:

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_model_structure_audit_placement.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS      # noqa: E402


def _audit():
    path = REPO / "Script_Bank" / "Analysis" / "SRM_AND_SBI_MONOMER_DIMER_ALP_Model_Structure_Audit.py"
    spec = importlib.util.spec_from_file_location("model_structure_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _audit()


def test_geometric_fraction_matches_an_isotropic_axis():
    geom = PARAMETERS.simulation.stem
    r, d = float(geom.particle_diameter_nm), float(geom.fission_product_distance_nm)
    assert d == 4.0 * r, (r, d)                                        # the declared convention, 40 nm
    q = A.in_plane_within_fraction(r, d)
    assert abs(q - (1.0 - np.sqrt(1.0 - 0.0625))) < 1e-12 and q < A.ACCEPTED_PLACEMENT_RESIDUAL
    assert abs(A.in_plane_within_fraction(r, 2.0 * r) - 0.1339746) < 1e-6
    rng = np.random.default_rng(0)
    cos = rng.uniform(-1.0, 1.0, 400_000)                               # isotropic axis: cos(theta) uniform
    assert abs(np.mean(d * np.sqrt(1.0 - cos ** 2) <= r) - q) < 0.003
    print("PASS: geometric fraction of daughter pairs within the reaction distance")


def test_geminate_refusions_are_counted_by_identity():
    rec = lambda kind, educts, products: SimpleNamespace(type=kind, educts=np.array(educts), products=np.array(products))  # noqa: E731
    fissions = [rec("fission", [1], [10, 11]), rec("fission", [2], [20, 21]), rec("fission", [3], [30, 31]),
                rec("conversion", [4], [40])]
    fusions = [rec("fusion", [11, 10], [50]),          # the same pair, educts in either order: geminate
               rec("fusion", [20, 30], [51]),          # daughters of different dimers: chance encounter
               rec("fusion", [31, 21], [52]),          # chance encounter
               rec("conversion", [40], [41])]          # not a fusion
    assert A.geminate_refusions(fissions, fusions) == (1, 2)
    assert A.geminate_refusions([], fusions) == (0, 3)
    assert A.geminate_refusions(fissions, []) == (0, 0)
    print("PASS: same-pair re-fusions counted by particle identity")


if __name__ == "__main__":
    test_geometric_fraction_matches_an_isotropic_axis()
    test_geminate_refusions_are_counted_by_identity()
