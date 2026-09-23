"""The three-point-estimate contract: central definitions, stored-field mapping, retired options.

Every Evaluation / Experiment product carries ``map_estimate``, ``posterior_quantiles`` (the
marginal median at the 0.50 level) and ``posterior_sgm`` for every observation. These tests pin the
definitions (``evaluation.POINT_ESTIMATES``), that the two draw-derived estimates come from one draw
set, the SGM's membership and scaling, and that the estimate-selection option is an error. They
need no trained flow.
"""
import numpy as np

from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from srm_and_sbi_monomer_dimer_alp.evaluation import (
    ESTIMATE_DEFINITIONS_VERSION, POINT_ESTIMATES, POINT_ESTIMATE_KEYS, QUANTILE_LEVELS,
    draw_label, point_estimate_note, sample_geometric_median, short_labels)


def test_definitions_are_complete_and_map_to_stored_fields():
    assert POINT_ESTIMATE_KEYS == ("map", "median", "sgm")
    for key, d in POINT_ESTIMATES.items():
        for field in ("label", "short", "stored_field", "operator", "population", "grouping",
                      "coordinates", "caveats"):
            assert d.get(field), (key, field)
        assert d["stored_field"] in schema.STAGE_FIELDS["evaluation"]
        assert d["stored_field"] in schema.STAGE_FIELDS["experiment"]
    assert POINT_ESTIMATES["median"]["stored_slice"] == "the quantile at level 0.50"
    assert POINT_ESTIMATES["sgm"]["scaling"]                 # the scaling is part of the identity
    assert isinstance(ESTIMATE_DEFINITIONS_VERSION, int)
    # Configurable quantities are not baked into the definition text.
    text = " ".join(str(v) for d in POINT_ESTIMATES.values() for v in d.values())
    for banned in ("1000", "top-2", "128", "0.128"):
        assert banned not in text, banned
    assert short_labels() == {"map": "MAP", "median": "median", "sgm": "SGM"}


def test_quantile_levels_carry_the_median_and_draw_labels_follow_pool_mode():
    assert 0.50 in QUANTILE_LEVELS and len(QUANTILE_LEVELS) == 5
    assert tuple(QUANTILE_LEVELS) == schema.CANONICAL_QUANTILE_LEVELS
    assert draw_label("bounded") == "posterior-draw"
    assert draw_label("unrestricted") == "flow-draw"
    note = point_estimate_note("unrestricted")
    assert "flow-draw" in note and "'MAP'" in note and "'median'" in note and "'SGM'" in note


def test_median_and_sgm_come_from_the_same_draws_and_sgm_is_a_member():
    rng = np.random.default_rng(3)
    draws = rng.normal(size=(400, 3)) * np.array([0.05, 1.0, 0.3]) + np.array([1.0, -2.0, 0.5])
    scale = np.array([0.3, 1.0, 1.0])
    q = np.quantile(draws, list(QUANTILE_LEVELS), axis=0).T           # (D, Q) as posterior_summary
    med = q[:, list(QUANTILE_LEVELS).index(0.50)]
    vec, idx = sample_geometric_median(draws, scale=scale)
    assert np.allclose(med, np.median(draws, axis=0))
    assert np.array_equal(vec, draws[idx])                              # a realized draw
    # Expected medoid under the declared scaling: brute force in scaled coordinates.
    z = draws / scale
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    assert idx == int(np.argmin(np.sqrt(d2).sum(1)))
    # The scaling matters: a different scaling can select a different member.
    _, idx_unscaled = sample_geometric_median(draws, scale=None)
    assert isinstance(idx_unscaled, int)


def test_retired_summary_option_is_an_error_in_both_runners():
    from srm_and_sbi_monomer_dimer_alp.evaluation_runner import build_evaluation_parser
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import build_experiment_parser
    for build, base in ((build_evaluation_parser,
                         ["--condition", "FAB", "--total-time-seconds", "2.0", "--eval-tasks", "1"]),
                        (build_experiment_parser, ["--condition", "FAB", "--total-time-seconds", "2.0"])):
        parser = build()
        parser.parse_args(base)
        for extra in (["--summary", "both"], ["--summary", "map"], ["--allow-partial"]):
            try:
                parser.parse_args(base + extra)
            except SystemExit:
                continue
            raise AssertionError(f"{extra} was accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
