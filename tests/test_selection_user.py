"""selection_user Nuisance_DLI: one explicitly chosen, fixed imaging vector from documented sources,
selected by a tag of its own. Run directly: python tests/test_selection_user.py (no pytest needed)."""
import json
import tempfile
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli
from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det

TABLE = det.DETECTOR_PARAMETERIZATION
KEYS = [e["KEY"] for e in TABLE]
PLO, PHI = det.theta_lower_bound(), det.theta_upper_bound()
INSIDE = {"mu_r": 1.74, "sigma_r": 0.22, "mu_pc": 139.0, "sigma_pc": 0.64,
          "prob_photo_bleach": 0.034, "lambda_rate": 4.3}


def _spec(values, allow=(), just="", drop_source=False):
    spec = {"block": {"posterior_sample_pool_choice": "selection_user",
                      "allow_outside_prior": list(allow), "outside_prior_justification": just},
            "selection": {}}
    for k, v in values.items():
        row = {"value": v, "source": f"src of {k}", "limitation": f"lim of {k}"}
        if drop_source:
            row.pop("source")
        spec["selection"][k] = row
    return spec


def _toml(spec):
    lines = ["[block]", 'posterior_sample_pool_choice = "selection_user"',
             "allow_outside_prior = [" + ", ".join(f'"{a}"' for a in spec["block"]["allow_outside_prior"]) + "]",
             f'outside_prior_justification = "{spec["block"]["outside_prior_justification"]}"', ""]
    for k, row in spec["selection"].items():
        lines += [f"[selection.{k}]", f"value = {row['value']}",
                  f'source = "{row["source"]}"', f'limitation = "{row["limitation"]}"', ""]
    return "\n".join(lines)


def test_nuisance_label_grammar_and_tagged_paths():
    assert ndli.nuisance_label("2S_50FPS") == "2S_50FPS"
    assert ndli.nuisance_label("2S_50FPS", None) == "2S_50FPS"
    assert ndli.nuisance_label("2S_50FPS", "REF") == "2S_50FPS_REF"
    for bad in ("ref", "R_EF", "REF-1", "r3f"):
        try:
            ndli.nuisance_label("2S_50FPS", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    a = ndli.artifact_path("/p", "ALIAS", "2S_50FPS", "REF")
    s = ndli.spec_path("/p", "ALIAS", "2S_50FPS", "REF")
    assert a.name == "ALIAS_2S_50FPS_REF_Nuisance_DLI.npz" and s.name == "ALIAS_2S_50FPS_REF_Nuisance_DLI_Spec.toml"
    assert ndli.artifact_path("/p", "ALIAS", "2S_50FPS").name == "ALIAS_2S_50FPS_Nuisance_DLI.npz"


def test_selection_converts_physical_values_to_log10_in_table_order():
    vec, rec = ndli.selection_from_spec(_spec(INSIDE), TABLE, PLO, PHI)
    assert vec.shape == (len(KEYS),)
    for i, k in enumerate(KEYS):
        assert abs(vec[i] - np.log10(INSIDE[k])) < 1e-12
        assert abs(rec["values_log10"][k] - vec[i]) < 1e-12
        assert rec["values_physical"][k] == INSIDE[k]
        assert rec["sources"][k] == {"source": f"src of {k}", "limitation": f"lim of {k}"}
    assert rec["kind"] == "user" and rec["fixed"] is True and rec["outside_prior"] == []
    assert np.all(vec >= np.asarray(PLO) - 1e-9) and np.all(vec <= np.asarray(PHI) + 1e-9)


def test_outside_prior_is_refused_unless_acknowledged_and_never_clipped():
    outside = dict(INSIDE, sigma_r=0.08)           # below the prior floor of 0.10
    try:
        ndli.selection_from_spec(_spec(outside), TABLE, PLO, PHI)
    except ValueError as exc:
        assert "sigma_r" in str(exc) and "allow_outside_prior" in str(exc)
    else:
        raise AssertionError("outside-prior value accepted without acknowledgement")
    try:                                            # acknowledged but no justification
        ndli.selection_from_spec(_spec(outside, allow=("sigma_r",)), TABLE, PLO, PHI)
    except ValueError as exc:
        assert "justification" in str(exc)
    else:
        raise AssertionError("acknowledgement without justification accepted")
    vec, rec = ndli.selection_from_spec(_spec(outside, allow=("sigma_r",), just="direct estimate below the box"),
                                        TABLE, PLO, PHI)
    i = KEYS.index("sigma_r")
    assert abs(vec[i] - np.log10(0.08)) < 1e-12            # stored as given: nothing clipped
    assert rec["outside_prior"] == ["sigma_r"] and rec["outside_prior_justification"]
    try:                                            # unknown key in the allow list
        ndli.selection_from_spec(_spec(INSIDE, allow=("kappa_q",), just="x"), TABLE, PLO, PHI)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_missing_source_or_invalid_value_is_refused():
    for bad in (_spec(INSIDE, drop_source=True), _spec(dict(INSIDE, mu_pc=-3.0)),
                _spec(dict(INSIDE, prob_photo_bleach=1.5)), _spec({k: v for k, v in INSIDE.items() if k != "mu_r"})):
        try:
            ndli.selection_from_spec(bad, TABLE, PLO, PHI)
        except ValueError:
            pass
        else:
            raise AssertionError(json.dumps(bad)[:80])


def test_artifact_is_one_fixed_vector_that_every_draw_returns_and_survives_a_round_trip():
    spec = _spec(INSIDE)
    nu = ndli.build_nuisance_dli(spec, KEYS, PLO, PHI, table=TABLE)
    assert nu.posterior_sample_pool_choice == "selection_user" and nu.pool_mode == "bounded"
    draws = nu.sample(50)
    assert draws.shape == (50, len(KEYS)) and np.all(draws == draws[0])
    assert np.allclose(10 ** draws[0], [INSIDE[k] for k in KEYS])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "X_2S_50FPS_REF_Nuisance_DLI.npz"
        nu.flush(str(path))
        back = ndli.NuisanceDLI.load(path)
        assert back.posterior_sample_pool_choice == "selection_user"
        assert np.array_equal(back.samples, nu.samples)
        assert back.selection == nu.selection and back.selection["sources"]["mu_r"]["source"] == "src of mu_r"
        assert np.array_equal(back.fixed_vector_log10, nu.samples[0])
        ident = back.identity(path)
        assert ident["artifact"] == path.name and ident["fixed"] is True and ident["n_samples"] == 1
        assert ident["selection"]["values_physical"]["lambda_rate"] == 4.3
        json.dumps(ident)                                   # JSON-ready for the consumers' records
        got = ndli.require_nuisance_dli(tmp, "X", "2S_50FPS", "REF")
        assert np.array_equal(got.samples, nu.samples)
        try:
            ndli.require_nuisance_dli(tmp, "X", "2S_50FPS")     # the canonical one does not exist
        except FileNotFoundError as exc:
            assert "X_2S_50FPS_Nuisance_DLI.npz" in str(exc)
        else:
            raise AssertionError


def test_acknowledged_outside_vector_records_unrestricted_and_the_justification():
    spec = _spec(dict(INSIDE, sigma_r=0.08), allow=("sigma_r",), just="direct estimate below the box")
    nu = ndli.build_nuisance_dli(spec, KEYS, PLO, PHI, table=TABLE)
    assert nu.pool_mode == "unrestricted" and nu.selection["outside_prior"] == ["sigma_r"]
    i = KEYS.index("sigma_r")
    assert nu.sample(3)[0, i] < PLO[i]


def test_spec_file_round_trip_through_load_spec_and_the_template_cannot_build_unfilled():
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = Path(tmp) / "X_2S_50FPS_REF_Nuisance_DLI_Spec.toml"
        spec_path.write_text(_toml(_spec(INSIDE)))
        loaded = ndli.load_spec(spec_path, KEYS, PLO, PHI, table=TABLE)
        nu = ndli.build_nuisance_dli(loaded, KEYS, PLO, PHI, table=TABLE)
        assert np.allclose(10 ** nu.samples[0], [INSIDE[k] for k in KEYS])
        try:
            ndli.load_spec(spec_path, KEYS, PLO, PHI)          # the table is required
        except ValueError:
            pass
        else:
            raise AssertionError
        tpl = Path(tmp) / "T_Nuisance_DLI_Spec.toml"
        ndli.emit_selection_user_template(tpl, TABLE, provenance={"nuisance_tag": "REF"},
                                          comparison={"prob_photo_bleach": 0.034})
        text = tpl.read_text()
        assert 'posterior_sample_pool_choice = "selection_user"' in text and "# value =" in text
        assert "comparison value: 0.034" in text
        try:
            ndli.load_spec(tpl, KEYS, PLO, PHI, table=TABLE)   # values commented out -> refused
        except ValueError as exc:
            assert "missing" in str(exc) or "needs `value`" in str(exc)
        else:
            raise AssertionError("an unfilled template built")


def test_other_choices_are_untouched():
    spec = {"block": {"posterior_sample_pool_choice": "box_user"},
            "imaging": {k: {"low": PLO[i] + 0.01, "high": PHI[i] - 0.01} for i, k in enumerate(KEYS)}}
    nu = ndli.build_nuisance_dli(spec, KEYS, PLO, PHI)
    assert nu.posterior_sample_pool_choice == "box_user" and nu.selection is None
    assert nu.identity()["fixed"] is False and "selection" not in nu.identity()
    assert ndli.POOL_KINDS["selection_user"] is None and "selection_user" in ndli.POOL_CHOICES


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
