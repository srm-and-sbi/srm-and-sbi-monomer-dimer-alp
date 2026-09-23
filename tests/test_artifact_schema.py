"""The numerical-product schema: validation, obsolete-field rejection, manifest contract, merging,
cache contract and report-only rendering. Synthetic products only; no GPU, no trained flow.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import artifact_schema as S
from srm_and_sbi_monomer_dimer_alp import provenance
from srm_and_sbi_monomer_dimer_alp.experiment_support import merge_validated_shards

from _product_fixtures import D, KEYS, product as _product, valid_manifest as _manifest


def test_valid_products_pass_and_median_is_located_by_level():
    rng = np.random.default_rng(0)
    m = S.validate_product(_product("evaluation", [(0, 0), (0, 1), (1, 0)], rng), stage="evaluation")
    assert S.median_level_index(m) == 2 and m["quantile_levels"][2] == 0.5
    S.validate_product(_product("experiment", [(0, 3, 0), (0, 3, 1)], rng), stage="experiment")


def test_rejections_name_the_problem():
    rng = np.random.default_rng(1)
    good = _product("evaluation", [(0, 0), (0, 1)], rng)
    cases = {
        "obsolete": lambda a: a.update(inferred_log10=a["map_estimate"]),
        "missing companion": lambda a: a.pop("posterior_sgm"),
        "non-finite": lambda a: a["posterior_quantiles"].__setitem__((1, 0, 2), np.inf),
        "bad shape": lambda a: a.update(posterior_sgm=a["posterior_sgm"][:, :2]),
        "duplicate id": lambda a: a.update(sim_index=np.array([0, 0])),
        "no manifest": lambda a: a.pop(S.MANIFEST_KEY),
    }
    for name, mutate in cases.items():
        a = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in good.items()}
        mutate(a)
        try:
            S.validate_product(a, stage="evaluation", source="t")
        except S.SchemaError as e:
            assert "t" in str(e); continue
        raise AssertionError(f"{name} accepted")
    try:
        S.reject_obsolete({"inferred_log10": 1}, source="old.npz")
    except S.SchemaError as e:
        assert "not a numerical correction" in str(e) or "does not repair" in str(e)
    else:
        raise AssertionError


def test_membership_not_just_count():
    ids = np.array([[0, 0], [0, 1], [1, 0]])
    S.assert_unique_observations(ids, expected=[(0, 0), (0, 1), (1, 0)])
    try:
        S.assert_unique_observations(ids, expected=[(0, 0), (0, 1), (1, 1)])   # same count, wrong member
    except S.SchemaError as e:
        assert "missing [(1, 1)]" in str(e) and "unexpected [(1, 0)]" in str(e)
    else:
        raise AssertionError("count-only check")


def test_merge_requires_one_contract_and_exact_inventory_and_keeps_order():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as tmp:
        sh = []
        for r, ids in enumerate(([(0, 0), (0, 2)], [(0, 1), (1, 0)])):
            a = _product("evaluation", ids, rng)
            p = Path(tmp) / f"_shard_{r:02d}_of_02.npz"; np.savez_compressed(p, **a); sh.append(p)
        keys = ["scores", "map_estimate", "true_log10", "posterior_quantiles", "posterior_sgm",
                "task_index", "sim_index"]
        merged, man, n = merge_validated_shards(sh, stage="evaluation", concat_keys=keys,
                                                expected_ids=[(0, 0), (0, 2), (0, 1), (1, 0)])
        assert n == 2 and man["n_observations"] == 4 and man["merged_from_shards"] == 2
        assert merged["task_index"].tolist() == [0, 0, 0, 1] and merged["sim_index"].tolist() == [0, 2, 1, 0]
        # wrong inventory
        try:
            merge_validated_shards(sh, stage="evaluation", concat_keys=keys, expected_ids=[(0, 0), (0, 2), (0, 1), (1, 5)])
        except S.SchemaError:
            pass
        else:
            raise AssertionError
        # incompatible contract on one shard
        b = _product("evaluation", [(2, 0)], rng, pool_mode="unrestricted")
        p3 = Path(tmp) / "_shard_02_of_03.npz"; np.savez_compressed(p3, **b)
        try:
            merge_validated_shards(sh + [p3], stage="evaluation", concat_keys=keys)
        except S.SchemaError as e:
            assert "contract differs" in str(e)
        else:
            raise AssertionError


def test_map_pool_cache_contract_invalidates_on_implementation_change_only_for_map_pools():
    from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli
    base = dict(pool_mode="bounded", n_per_chunk=10, span_seconds=20, chunk_step_seconds=None,
                kinds=["FAB"], max_cells=60, estimator_sha256="e" * 64)
    m1 = _manifest("experiment", 1); m2 = json.loads(json.dumps(m1)); m2["code"]["implementation"]["sha256"] = "j" * 64
    assert ndli.pool_provenance(**base) == ndli.pool_provenance(**base)          # draw-only pools: no contract, unchanged
    c1, c2 = ndli.map_pool_contract(m1), ndli.map_pool_contract(m2)
    assert ndli.pool_provenance(**base, map_contract=c1) != ndli.pool_provenance(**base, map_contract=c2)
    assert "map_contract" not in ndli.pool_provenance(**base)


def test_implementation_hash_is_deterministic_and_marks_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root / "x.py").write_text("a = 1\n")
        h1 = provenance.implementation_hash(root, files=("x.py", "gone.py"))
        h2 = provenance.implementation_hash(root, files=("x.py", "gone.py"))
        assert h1 == h2 and h1["files"][1] == {"path": "gone.py", "missing": True}
        (root / "x.py").write_text("a = 2\n")
        assert provenance.implementation_hash(root, files=("x.py", "gone.py"))["sha256"] != h1["sha256"]


def test_report_only_rendering_leaves_arrays_untouched():
    from types import SimpleNamespace
    from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter
    from srm_and_sbi_monomer_dimer_alp.evaluation_runner import write_recovery_outputs
    from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS
    rng = np.random.default_rng(4)
    ids = [(0, s) for s in range(60)]
    a = _product("evaluation", ids, rng)
    spec = [{"KEY": k, "LABEL": k, "PRIOR_RANGE": (-3.0, 3.0), "LOG_FLAG": True} for k in KEYS]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prod.npz"; np.savez_compressed(path, **a)
        before = path.read_bytes()
        rep = DiagnosticReporter(stage="Evaluation", enabled=True, dump=True, dump_dir=Path(tmp),
                                 run_label="T", timestamp="now")
        args = SimpleNamespace(bin_mode="quantile", n_bins=4, min_count=2, eval_tasks=1)
        man = write_recovery_outputs(rep, args, PARAMETERS.inference.evaluation, spec, path, a,
                                     run_start=0.0, persist_arrays=False)
        assert man["n_observations"] == 60                     # the writer decoded and validated it
        assert path.read_bytes() == before
        assert (Path(tmp) / "report.md").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
