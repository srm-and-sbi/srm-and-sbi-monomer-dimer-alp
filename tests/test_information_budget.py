"""The information-budget kernel's operating point and field noise come from the model, never from a typed
constant: the derivation reproduces the hand calculation, the operating point of each condition is the retained
population's fluorescent complexes at the prior center, the relative noise scales as one over the spot count, the
two readers obtain it through the kernel, and the decay bound scales with it (why the documented table moves)."""
import os
import pathlib
import re

import numpy as np

from srm_and_sbi_monomer_dimer_alp import information_budget as ib

REPO = pathlib.Path(__file__).resolve().parents[1]
BUDGET = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Information_Budget.py"
LOSS = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Direct_Fluorescence_Loss.py"


def test_field_relative_noise_reproduces_the_hand_derivation():
    # 65,536 pixels of background at the SCOPE center against 154 spots of 2.03 dyes at 237 photons
    out = ib.field_relative_noise(154, 2.0347, 237.1, 41.93, 0.9016, 28.84, 10.53, 256 * 256)
    assert abs(out["field_noise_adu"] - 7.745e4) < 0.01e4, out["field_noise_adu"]
    assert abs(out["emitter_signal_adu"] - 2.809e6) < 0.005e6, out["emitter_signal_adu"]
    assert abs(out["relative_noise"] - 0.0276) < 0.0002, out["relative_noise"]
    assert out["n_pixels"] == 65536 and out["n_emitters"] == 154.0


def test_the_operating_point_is_the_retained_population_at_the_prior_center():
    fab = ib.operating_point("FAB")
    assert (fab["n_monomers"], fab["n_dimers"], fab["n_subunits"]) == (49, 98, 245), fab
    assert abs(fab["n_probe_bound"] - 147.7) < 0.1 and abs(fab["n_fluorescent"] - 118.6) < 0.1, fab
    assert fab["n_emitters"] == 119 and abs(fab["dyes_per_spot"] - 2.035) < 0.001 and fab["law"] == "FAB_POISSON"
    inlb = ib.operating_point("INLB")
    assert (inlb["n_monomers"], inlb["n_dimers"], inlb["n_subunits"]) == (79, 350, 477), inlb
    assert inlb["n_emitters"] == 226 and abs(inlb["dyes_per_spot"] - 1.0) < 1e-9, inlb


def test_the_relative_noise_follows_the_operating_point_and_the_scope_center():
    out = ib.operating_relative_noise("FAB")
    assert out["operating_point"]["n_emitters"] == 119 and out["n_pixels"] == 65536
    assert abs(out["relative_noise"] - 0.0357) < 0.0003, out["relative_noise"]
    # the same inputs by hand: SCOPE center, the mu_pc prior center, the operating point's spots and dyes
    by_hand = ib.field_relative_noise(119, out["dyes_per_spot"], out["mu_pc"], out["gamma"], out["kappa_q"],
                                      out["kappa_o"], out["kappa_s"], out["n_pixels"])
    assert abs(by_hand["relative_noise"] - out["relative_noise"]) < 1e-12
    # a synthetic scene passes its own count and dyes and needs no condition
    scene = ib.operating_relative_noise(None, n_emitters=150, dyes_per_spot=1.0)
    assert scene["operating_point"] is None and abs(scene["relative_noise"] - 0.0576) < 0.0003, scene["relative_noise"]
    try:
        ib.operating_relative_noise(None)
    except ValueError as exc:
        assert "condition" in str(exc)
    else:
        raise AssertionError("a scene without a condition and without its own count must be refused")


def test_the_relative_noise_scales_as_one_over_the_spot_count():
    one = ib.operating_relative_noise("FAB", n_emitters=119, dyes_per_spot=2.0)["relative_noise"]
    two = ib.operating_relative_noise("FAB", n_emitters=238, dyes_per_spot=2.0)["relative_noise"]
    assert abs(two * 2 - one) < 1e-12 and two < one


def test_no_typed_operating_point_constant_remains():
    assert not hasattr(ib, "FIELD_RELATIVE_NOISE_MET_FAB")
    for path in (BUDGET, LOSS):
        text = path.read_text()
        assert "operating_relative_noise(" in text, path.name
        assert not re.search(r"^(DEFAULT_N_EMITTERS|DEFAULT_RELATIVE_NOISE|RELATIVE_NOISE)\s*=", text, re.M), path.name
        assert "FIELD_RELATIVE_NOISE" not in text, path.name


def test_the_decay_bound_scales_with_the_relative_noise():
    # why the documented bleaching table moves with the operating point: the bound is linear in eps
    a = ib.crb_prob_bleach_dex(0.10, 1000, 0.028, lambda_rate=3.162, frame_time_seconds=0.02)["sd_dex"]
    b = ib.crb_prob_bleach_dex(0.10, 1000, 0.0357, lambda_rate=3.162, frame_time_seconds=0.02)["sd_dex"]
    assert abs(b / a - 0.0357 / 0.028) < 1e-9, (a, b)
    assert abs(a - 0.083) < 0.001 and abs(b - 0.106) < 0.001, (a, b)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
