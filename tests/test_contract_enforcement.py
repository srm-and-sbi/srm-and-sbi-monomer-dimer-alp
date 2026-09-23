"""Enforcement of the three-estimate contract (0.1.16 completion): every guarantee the contract
states has a negative test here that reproduces the violation and shows it refused.

Covers: manifest validation (required keys, versions, types, cross-field rules), array validation
(finiteness of scores and truth, integer identifiers, nondecreasing quantiles, the optional cloud),
writers validating on entry, the composition loader refusing a legacy product, exact
parameter-key checks, shard merging (seed policy, invocation identity, window geometry, condition
labels, per-run field equality, retained shard provenance), empty shards, the TIFF layout rule,
startup-versus-write implementation provenance, the launcher-created invocation id, one shared
draw set in ``posterior_summary``, and the retired option hidden from help. The second review adds:
a replacement shard from another Slurm job (or a local run) merging, since only the logical
invocation is compared; optional draw storage as a run-level setting; and the validator comparing
the recorded implementation hashes itself rather than trusting the flag.
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from srm_and_sbi_monomer_dimer_alp import artifact_schema as S
from srm_and_sbi_monomer_dimer_alp import provenance
from srm_and_sbi_monomer_dimer_alp import experiment_support as es
from srm_and_sbi_monomer_dimer_alp.evaluation import QUANTILE_LEVELS

from _product_fixtures import (D, KEYS, KINDS, code_block, copy_product, expect_schema_error,
                               product, valid_manifest, with_manifest)

REPO = Path(__file__).resolve().parent.parent
CONTROLS = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py"
SPEC = [{"KEY": k, "LABEL": k, "PRIOR_RANGE": (-3.0, 3.0), "LOG_FLAG": True} for k in KEYS]


# ---- 1. the manifest is validated, not just recorded -------------------------------------------

def test_manifest_required_keys_versions_types_and_cross_field_rules():
    rng = np.random.default_rng(10)
    good = product("evaluation", [(0, 0), (0, 1)], rng)
    S.validate_product(good, stage="evaluation")
    cases = {
        "missing optimizer": (lambda m: m.pop("optimizer"), "lacks required key"),
        "missing code": (lambda m: m.pop("code"), "lacks required key"),
        "missing checkpoint": (lambda m: m.pop("checkpoint_sha256"), "lacks required key"),
        "missing n_summary_draws": (lambda m: m.pop("n_summary_draws"), "lacks required key"),
        "missing run_identity": (lambda m: m.pop("run_identity"), "lacks required key"),
        "missing seed_policy": (lambda m: m.pop("seed_policy"), "lacks required key"),
        "missing execution": (lambda m: m.pop("execution"), "lacks required key"),
        "missing stored_optional_fields": (lambda m: m.pop("stored_optional_fields"),
                                           "lacks required key"),
        "optional field foreign to the stage": (
            lambda m: m.update(stored_optional_fields=["posterior_samples_cloud"]),
            "must list distinct optional fields"),
        "unknown definitions version": (lambda m: m.update(estimate_definitions_version=99),
                                        "not supported"),
        "sgm_scale zero": (lambda m: m.update(sgm_scale=[1.0, 0.0, 1.0]), "finite and positive"),
        "sgm_scale negative": (lambda m: m.update(sgm_scale=[1.0, -1.0, 1.0]), "finite and positive"),
        "sgm_scale wrong length": (lambda m: m.update(sgm_scale=[1.0, 1.0]), "one per parameter"),
        "sgm_scale non-numeric": (lambda m: m.update(sgm_scale=["x", 1.0, 1.0]), "finite and positive"),
        "pool/draw mismatch": (lambda m: m.update(pool_mode="unrestricted"), "does not match pool_mode"),
        "unknown pool_mode": (lambda m: m.update(pool_mode="free", draw_label="flow-draw"),
                              "is not one of"),
        "wrong stage": (lambda m: m.update(stage="experiment"), "manifest stage"),
        "non-positive draws": (lambda m: m.update(n_summary_draws=0), "positive integer"),
        "reversed levels": (lambda m: m.update(quantile_levels=[0.95, 0.75, 0.5, 0.25, 0.05]),
                            "canonical"),
        "optimizer key missing": (lambda m: m["optimizer"].pop("tolerance"), "optimizer block lacks"),
        "optimizer pool differs": (lambda m: m["optimizer"].update(pool_mode="unrestricted"),
                                   "optimizer.pool_mode"),
        "optimizer lr negative": (lambda m: m["optimizer"].update(learning_rate=-1.0),
                                  "finite and positive"),
        "code changed during run": (lambda m: m.update(code=code_block(changed=True)),
                                    "changed while the stage ran"),
        "code block incomplete": (lambda m: m["code"].pop("implementation_at_write"),
                                  "code block lacks"),
        "code hash malformed": (lambda m: m["code"]["implementation"].update(sha256="abc"),
                                "sha256"),
        "checkpoint malformed": (lambda m: m.update(checkpoint_sha256="notahash"), "SHA-256"),
        "checkpoint null": (lambda m: m.update(checkpoint_sha256=None), "SHA-256"),
        "run_identity without invocation": (lambda m: m["run_identity"].pop("invocation_id"),
                                            "invocation_id"),
        "seed_policy bad seed": (lambda m: m["seed_policy"].update(base_seed="7"), "base_seed"),
        "geometry on evaluation": (lambda m: m.update(window_geometry={"n_frames": 1, "step_frames": 1,
                                                                        "span_frames": None}),
                                   "must be null for the evaluation stage"),
        "labels on evaluation": (lambda m: m.update(condition_labels=["FAB"]),
                                 "must be null for the evaluation stage"),
        "id fields wrong": (lambda m: m.update(observation_id_fields=["sim_index", "task_index"]),
                            "observation_id_fields"),
        "n_observations wrong": (lambda m: m.update(n_observations=5), "n_observations"),
        "schema version wrong": (lambda m: m.update(artifact_schema_version=2), "artifact_schema_version"),
    }
    for name, (mutate, needle) in cases.items():
        bad = with_manifest(good, mutate)
        msg = expect_schema_error(lambda: S.validate_product(bad, stage="evaluation", source="t"), "t")
        assert needle in msg, (name, msg)
    # "validated computation contract" also covers the manifest alone
    m = S.decode_manifest(good[S.MANIFEST_KEY]); S.validate_manifest(m, stage="evaluation")


def test_experiment_manifest_requires_geometry_and_labels_matching_kinds():
    rng = np.random.default_rng(11)
    good = product("experiment", [(0, 1, 0), (0, 1, 1)], rng)
    S.validate_product(good, stage="experiment")
    cases = {
        "no geometry": (lambda m: m.update(window_geometry=None), "window_geometry is required"),
        "bad step": (lambda m: m["window_geometry"].update(step_frames=0), "positive integer"),
        "no labels": (lambda m: m.update(condition_labels=None), "condition_labels must be"),
        "labels differ from kinds": (lambda m: m.update(condition_labels=["INLB"]),
                                     "differ from the manifest's condition_labels"),
        "duplicate labels": (lambda m: m.update(condition_labels=["FAB", "FAB"]), "distinct strings"),
    }
    for name, (mutate, needle) in cases.items():
        bad = with_manifest(good, mutate)
        msg = expect_schema_error(lambda: S.validate_product(bad, stage="experiment", source="t"))
        assert needle in msg, (name, msg)
    # kind_index must index the stored kinds
    bad = copy_product(good); bad["kind_index"] = np.array([0, 3])
    expect_schema_error(lambda: S.validate_product(bad, stage="experiment"), "kind_index out of range")


def test_arrays_scores_truth_ids_quantiles_and_cloud_are_validated():
    rng = np.random.default_rng(12)
    good = product("evaluation", [(0, 0), (0, 1), (0, 2)], rng)
    bad = copy_product(good); bad["scores"][1] = np.nan
    expect_schema_error(lambda: S.validate_product(bad, stage="evaluation"), "non-finite scores", "[1]")
    bad = copy_product(good); bad["true_log10"][2, 0] = np.inf
    expect_schema_error(lambda: S.validate_product(bad, stage="evaluation"), "non-finite true_log10", "[2]")
    bad = copy_product(good); bad["sim_index"] = np.array([0.0, 1.5, 2.0])
    expect_schema_error(lambda: S.validate_product(bad, stage="evaluation"), "integer-valued")
    # integer-valued floats are admitted (they are converted exactly), non-integer ones are not
    ok = copy_product(good); ok["sim_index"] = np.array([0.0, 1.0, 2.0]); S.validate_product(ok, stage="evaluation")
    # quantiles: a decrease is refused, equal neighbors are valid
    bad = copy_product(good); bad["posterior_quantiles"][0, 1, :] = [0.5, 0.4, 0.6, 0.7, 0.8]
    expect_schema_error(lambda: S.validate_product(bad, stage="evaluation"), "decrease along the level axis", "[0]")
    ok = copy_product(good); ok["posterior_quantiles"][0, 1, :] = [0.5, 0.5, 0.5, 0.7, 0.7]
    S.validate_product(ok, stage="evaluation")
    # the optional cloud (declared in stored_optional_fields): shape (N, S, D), S = n_summary_draws, finite
    exp = product("experiment", [(0, 1, 0), (0, 1, 1)], rng,
                  stored_optional_fields=["posterior_samples_cloud"])
    ok = copy_product(exp); ok["posterior_samples_cloud"] = rng.normal(size=(2, 50, D)).astype(np.float32)
    S.validate_product(ok, stage="experiment")
    bad = copy_product(exp); bad["posterior_samples_cloud"] = rng.normal(size=(2, 40, D))
    expect_schema_error(lambda: S.validate_product(bad, stage="experiment"), "posterior_samples_cloud has shape")
    bad = copy_product(exp); bad["posterior_samples_cloud"] = rng.normal(size=(2, 50, D)); bad["posterior_samples_cloud"][1, 3, 0] = np.nan
    expect_schema_error(lambda: S.validate_product(bad, stage="experiment"), "non-finite posterior_samples_cloud", "[1]")


# ---- 2. writers validate on entry ---------------------------------------------------------------

def _controls_module():
    spec = importlib.util.spec_from_file_location("controls_runner", CONTROLS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _experiment_product(rng, n_cells=3, n_chunks=5):
    ids = [(0, c, t) for c in range(n_cells) for t in range(n_chunks)]
    return product("experiment", ids, rng)


def test_writers_refuse_invalid_products_before_persisting_or_rendering():
    from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter
    from srm_and_sbi_monomer_dimer_alp.evaluation_runner import write_recovery_outputs
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import write_experiment_outputs
    from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS
    eval_cfg = PARAMETERS.inference.evaluation
    rng = np.random.default_rng(13)
    controls = _controls_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Evaluation writer: a non-finite SGM never reaches savez, and no report is rendered.
        a = product("evaluation", [(0, s) for s in range(20)], rng); a["posterior_sgm"][3, 1] = np.nan
        rep = DiagnosticReporter(stage="Evaluation", enabled=True, dump=True, dump_dir=tmp / "e",
                                 run_label="T", timestamp="now")
        args = SimpleNamespace(bin_mode="quantile", n_bins=4, min_count=2, eval_tasks=1)
        expect_schema_error(lambda: write_recovery_outputs(rep, args, eval_cfg, SPEC, tmp / "e" / "p.npz",
                                                           a, run_start=0.0), "non-finite posterior_sgm", "[3]")
        assert not (tmp / "e" / "p.npz").exists() and not (tmp / "e" / "report.md").exists()
        # Experiment writer: a manifest that disagrees with the arrays is refused (no separate
        # manifest argument exists any more; the writer decodes and validates the stored one).
        b = with_manifest(_experiment_product(rng), lambda m: m.update(n_observations=1))
        rep = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "x",
                                 run_label="T", timestamp="now")
        xargs = SimpleNamespace(aggregation="pooled", seed=None)
        expect_schema_error(lambda: write_experiment_outputs(rep, xargs, eval_cfg, SPEC, tmp / "x" / "p.npz",
                                                             b, run_start=0.0), "n_observations")
        assert not (tmp / "x" / "p.npz").exists()
        # Controls writer: same boundary.
        c = _experiment_product(rng); c["map_estimate"][0, 0] = np.inf
        rep = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "c",
                                 run_label="T", timestamp="now")
        expect_schema_error(lambda: controls.write_experiment_outputs(rep, xargs, eval_cfg, tmp / "c" / "p.npz",
                                                                      c, run_start=0.0), "non-finite map_estimate")
        assert not (tmp / "c" / "p.npz").exists()


def test_report_only_rendering_for_experiment_and_controls_leaves_arrays_untouched():
    from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import write_experiment_outputs
    from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, PARAMETER_KEYS
    eval_cfg = PARAMETERS.inference.evaluation
    rng = np.random.default_rng(14)
    xargs = SimpleNamespace(aggregation="pooled", seed=None)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = _experiment_product(rng)
        path = tmp / "x" / "p.npz"; path.parent.mkdir(); np.savez_compressed(path, **a)
        before = path.read_bytes()
        rep = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "x",
                                 run_label="T", timestamp="now")
        man = write_experiment_outputs(rep, xargs, eval_cfg, SPEC, path, a, run_start=0.0,
                                       persist_arrays=False)
        assert man["n_observations"] == 15 and path.read_bytes() == before
        assert (tmp / "x" / "report.md").exists() and any((tmp / "x" / "figures").iterdir())
        # Controls writer renders with the biology table; build a product of that width.
        controls = _controls_module()
        n_keys = len(PARAMETER_KEYS)
        ids = [(0, c, t) for c in range(3) for t in range(5)]
        m = valid_manifest("experiment", len(ids), parameter_keys=list(PARAMETER_KEYS),
                           sgm_scale=[1.0] * n_keys)
        c = product("experiment", ids, rng, manifest=m)
        for key in ("map_estimate", "posterior_sgm"):
            c[key] = rng.normal(size=(len(ids), n_keys)) * 0.1
        c["posterior_quantiles"] = np.sort(rng.normal(size=(len(ids), n_keys, 5)) * 0.1, axis=-1)
        cpath = tmp / "c" / "p.npz"; cpath.parent.mkdir(); np.savez_compressed(cpath, **c)
        before = cpath.read_bytes()
        rep = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "c",
                                 run_label="T", timestamp="now")
        man = controls.write_experiment_outputs(rep, xargs, eval_cfg, cpath, c, run_start=0.0,
                                                persist_arrays=False)
        assert man["n_observations"] == 15 and cpath.read_bytes() == before
        assert (tmp / "c" / "report.md").exists()


# ---- 3. consumers: schema loader everywhere, exact parameter keys ------------------------------

def test_composition_loader_refuses_legacy_product_and_wrong_keys():
    from srm_and_sbi_monomer_dimer_alp.population_composition_runner import load_experiment_draws
    rng = np.random.default_rng(15)
    with tempfile.TemporaryDirectory() as tmp:
        legacy = Path(tmp) / "old.npz"
        np.savez_compressed(legacy, inferred_log10=rng.normal(size=(4, D)),
                            posterior_samples_cloud=rng.normal(size=(4, 10, D)),
                            kind_index=np.zeros(4, np.int64), cell=np.arange(4), chunk=np.zeros(4, np.int64),
                            kinds=np.asarray(["FAB"]))
        expect_schema_error(lambda: load_experiment_draws(legacy, KEYS), "Obsolete artifact schema")
        # valid schema, wrong key order -> refused (count alone would pass)
        a = product("experiment", [(0, 0, 0), (0, 0, 1)], rng,
                    stored_optional_fields=["posterior_samples_cloud"])
        a["posterior_samples_cloud"] = rng.normal(size=(2, 50, D)).astype(np.float32)
        good = Path(tmp) / "good.npz"; np.savez_compressed(good, **a)
        expect_schema_error(lambda: load_experiment_draws(good, ["b", "a", "c"]), "exact key and order")
        cloud, labels = load_experiment_draws(good, KEYS)
        assert cloud.shape == (2, 50, D) and labels["kinds"] == ["FAB"]


def test_assert_parameter_keys_is_exact():
    m = valid_manifest("evaluation", 1)
    S.assert_parameter_keys(m, KEYS)
    expect_schema_error(lambda: S.assert_parameter_keys(m, ["a", "c", "b"]), "exact key and order")
    expect_schema_error(lambda: S.assert_parameter_keys(m, ["a", "b"]), "differ")


# ---- 4. shard compatibility ---------------------------------------------------------------------

_EVAL_KEYS = ["scores", "map_estimate", "true_log10", "posterior_quantiles", "posterior_sgm",
              "task_index", "sim_index"]
_EXP_KEYS = ["scores", "map_estimate", "kind_index", "cell", "chunk", "posterior_quantiles",
             "posterior_sgm"]


def _write(tmp, name, a):
    p = Path(tmp) / name; np.savez_compressed(p, **a); return p


def test_merge_refuses_different_seed_invocation_geometry_labels_and_kinds():
    from srm_and_sbi_monomer_dimer_alp.experiment_support import merge_validated_shards
    rng = np.random.default_rng(16)
    with tempfile.TemporaryDirectory() as tmp:
        base = _write(tmp, "_shard_00_of_02.npz", product("evaluation", [(0, 0)], rng))
        for name, over, needle in (
                ("seed", dict(seed=7), "seed_policy"),
                ("invocation", dict(invocation="inv-2"), "run_identity")):
            other = _write(tmp, f"_shard_01_of_02_{name}.npz", product("evaluation", [(0, 1)], rng, **over))
            expect_schema_error(lambda: merge_validated_shards([base, other], stage="evaluation",
                                                               concat_keys=_EVAL_KEYS), "contract differs", needle)
        # experiment: window geometry and condition labels are part of the contract
        e0 = _write(tmp, "_e_00.npz", product("experiment", [(0, 0, 0)], rng))
        geom = with_manifest(product("experiment", [(0, 1, 0)], rng),
                             lambda m: m["window_geometry"].update(step_frames=50))
        e1 = _write(tmp, "_e_01.npz", geom)
        expect_schema_error(lambda: merge_validated_shards([e0, e1], stage="experiment", concat_keys=_EXP_KEYS,
                                                           first_keys=["kinds"]), "window_geometry")
        # conflicting kinds mapping: the manifest labels AND the stored array must agree across shards
        other_kinds = product("experiment", [(0, 1, 0)], rng, kinds=["INLB"])
        e2 = _write(tmp, "_e_02.npz", other_kinds)
        expect_schema_error(lambda: merge_validated_shards([e0, e2], stage="experiment", concat_keys=_EXP_KEYS,
                                                           first_keys=["kinds"]), "contract differs", "condition_labels")
        # per-run field equality is enforced on its own too (manifest agreeing, array differing)
        tampered = copy_product(product("experiment", [(0, 1, 0)], rng)); tampered["kinds"] = np.asarray(["INLB"])
        e3 = _write(tmp, "_e_03.npz", tampered)
        try:
            merge_validated_shards([e0, e3], stage="experiment", concat_keys=_EXP_KEYS, first_keys=["kinds"])
        except S.SchemaError as exc:
            assert "kinds" in str(exc)
        else:
            raise AssertionError("conflicting kinds merged")


def test_merged_manifest_retains_per_shard_provenance():
    from srm_and_sbi_monomer_dimer_alp.experiment_support import merge_validated_shards
    rng = np.random.default_rng(17)
    with tempfile.TemporaryDirectory() as tmp:
        sh = [_write(tmp, f"_shard_{r:02d}_of_02.npz",
                     product("evaluation", ids, rng, rank=r, world_size=2))
              for r, ids in enumerate(([(0, 0)], [(0, 1), (0, 2)]))]
        _, man, n = merge_validated_shards(sh, stage="evaluation", concat_keys=_EVAL_KEYS)
        assert n == 2 and [s["rank"] for s in man["shards"]] == [0, 1]
        assert [s["n_observations"] for s in man["shards"]] == [1, 2]
        assert man["rank"] is None and man["merged_from_shards"] == 2


# ---- 5. empty ranks -------------------------------------------------------------------------------

def test_empty_shard_is_valid_mergeable_and_an_empty_final_product_is_refused():
    from srm_and_sbi_monomer_dimer_alp.experiment_support import merge_validated_shards
    rng = np.random.default_rng(18)
    empty = S.empty_product_arrays("evaluation", n_parameters=D, n_levels=5)
    assert empty["map_estimate"].shape == (0, D) and empty["posterior_quantiles"].shape == (0, D, 5)
    assert empty["scores"].shape == (0,) and empty["task_index"].dtype == np.int64
    empty[S.MANIFEST_KEY] = S.encode_manifest(valid_manifest("evaluation", 0))
    S.validate_product(empty, stage="evaluation", allow_empty=True)
    expect_schema_error(lambda: S.validate_product(empty, stage="evaluation"), "no observations")
    with tempfile.TemporaryDirectory() as tmp:
        e = _write(tmp, "_shard_00_of_02.npz", empty)
        full = _write(tmp, "_shard_01_of_02.npz", product("evaluation", [(0, 0), (0, 1)], rng))
        merged, man, n = merge_validated_shards([e, full], stage="evaluation", concat_keys=_EVAL_KEYS,
                                                expected_ids=[(0, 0), (0, 1)])
        assert n == 2 and man["n_observations"] == 2 and merged["map_estimate"].shape == (2, D)
        e2 = _write(tmp, "_shard_01_of_02_empty.npz", empty)
        expect_schema_error(lambda: merge_validated_shards([e, e2], stage="evaluation", concat_keys=_EVAL_KEYS),
                            "no observations")
    # experiment flavor carries the run field
    ex = S.empty_product_arrays("experiment", n_parameters=D, run_fields={"kinds": np.asarray(KINDS)})
    ex[S.MANIFEST_KEY] = S.encode_manifest(valid_manifest("experiment", 0))
    S.validate_product(ex, stage="experiment", allow_empty=True)
    # the stage-side helper writes an empty shard only when asked
    with tempfile.TemporaryDirectory() as tmp:
        topo = SimpleNamespace(rank=1, world_size=2)
        assert es.save_shard(tmp, topo, empty, count=0) is None
        p = es.save_shard(tmp, topo, empty, count=0, write_empty=True)
        assert p is not None and p.exists()
    # the coverage error no longer recommends a removed option
    with tempfile.TemporaryDirectory() as tmp:
        only = _write(tmp, "_shard_00_of_02.npz", product("evaluation", [(0, 0)], rng))
        try:
            es.assert_complete_shard_set([only], partial_option=None)
        except ValueError as exc:
            assert "--allow-partial" not in str(exc) and "missing rank(s) [1]" in str(exc)
        else:
            raise AssertionError


# ---- 6. one TIFF layout rule ----------------------------------------------------------------------

def test_tiff_layout_rule_is_shared_and_rejects_unsupported_layouts():
    import tifffile
    rng = np.random.default_rng(19)
    frames = rng.integers(0, 65535, size=(10, 8, 8), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as tmp:
        stack = Path(tmp) / "stack.tif"; tifffile.imwrite(stack, frames)                    # QYX
        imagej = Path(tmp) / "ij.tif"; tifffile.imwrite(imagej, frames, imagej=True, metadata={"axes": "TYX"})
        planar = Path(tmp) / "planar.tif"                                                   # one page, SYX
        tifffile.imwrite(planar, frames, photometric="minisblack", planarconfig="separate")
        hyper = Path(tmp) / "hyper.tif"; tifffile.imwrite(hyper, frames.reshape(2, 5, 8, 8))  # 4-D
        for good in (stack, imagej):
            assert es.inspect_recording(good) == (10, 8, 8)
            chunks = es.read_cell_chunks(good, 4, 2)
            assert len(chunks) == es.chunk_count(10, 4, 2) == 4 and chunks[0].shape == (4, 8, 8)
        for bad in (planar, hyper):
            try:
                es.inspect_recording(bad)
            except es.RecordingLayoutError as exc:
                assert bad.name in str(exc)
            else:
                raise AssertionError(f"{bad.name} accepted")
            # the reader refuses the same file the inventory refuses
            try:
                es.read_cell_chunks(bad, 4, 2)
            except es.RecordingLayoutError:
                pass
            else:
                raise AssertionError
        # preflight reports every offending file at once, before any estimation
        try:
            es.preflight_recordings([stack, planar, hyper])
        except es.RecordingLayoutError as exc:
            assert "2 recording(s)" in str(exc) and "planar.tif" in str(exc) and "hyper.tif" in str(exc)
        else:
            raise AssertionError
        assert es.preflight_recordings([stack, imagej]) == {stack: (10, 8, 8), imagej: (10, 8, 8)}


# ---- 7. provenance --------------------------------------------------------------------------------

def test_startup_provenance_detects_changes_and_the_file_list_covers_every_estimate_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root / "x.py").write_text("a = 1\n")
        start = provenance.code_provenance(root, files=("x.py",))
        same = provenance.finalize_code_provenance(start, root, files=("x.py",))
        assert same["changed_during_run"] is False and same["implementation"] == start["implementation"]
        (root / "x.py").write_text("a = 2\n")
        changed = provenance.finalize_code_provenance(start, root, files=("x.py",))
        assert changed["changed_during_run"] is True
        assert changed["implementation"]["sha256"] == start["implementation"]["sha256"]   # startup kept
        assert changed["implementation_at_write"]["sha256"] != start["implementation"]["sha256"]
    # a product carrying a changed flag is refused by the validator
    rng = np.random.default_rng(20)
    bad = product("evaluation", [(0, 0)], rng, code=code_block(changed=True))
    expect_schema_error(lambda: S.validate_product(bad, stage="evaluation"), "changed while the stage ran")
    # the hashed file list covers the estimate-producing paths and every listed file exists
    listed = set(provenance.IMPLEMENTATION_FILES)
    assert "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py" in listed
    assert "srm_and_sbi_monomer_dimer_alp/detector_nuisance_dli.py" in listed
    assert all((REPO / f).is_file() for f in listed), [f for f in listed if not (REPO / f).is_file()]
    assert not any(e.get("missing") for e in provenance.implementation_hash()["files"])


def test_invocation_id_comes_from_the_launcher():
    saved = os.environ.pop(S.INVOCATION_ID_ENV, None)
    try:
        expect_schema_error(lambda: S.run_identity("P", distributed=True), S.INVOCATION_ID_ENV)
        solo = S.run_identity("P", distributed=False)
        assert solo["invocation_id"] and solo["product_label"] == "P"
        os.environ[S.INVOCATION_ID_ENV] = "launcher-made"
        assert S.run_identity("P", distributed=True)["invocation_id"] == "launcher-made"
    finally:
        os.environ.pop(S.INVOCATION_ID_ENV, None)
        if saved is not None:
            os.environ[S.INVOCATION_ID_ENV] = saved
    # the stage scripts export it before the ranks start
    for name in ("SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Evaluation.sh", "SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Experiment.sh",
                 "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Evaluation.sh",
                 "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Experiment.sh"):
        text = (REPO / "Script_Bank" / "HPC" / name).read_text()
        assert f"export {S.INVOCATION_ID_ENV}=" in text
        assert text.index(f"export {S.INVOCATION_ID_ENV}=") < text.index("srun ")
        assert '${SUMMARY+x}' in text            # supplying SUMMARY at all (even empty) is refused


# ---- 8. one shared draw set in posterior_summary -------------------------------------------------

def test_posterior_summary_obtains_one_collection_and_derives_both_estimates_from_it():
    import torch
    from srm_and_sbi_monomer_dimer_alp import evaluation as ev
    rng = np.random.default_rng(21)
    draws = rng.normal(size=(64, D)) * np.array([0.1, 1.0, 0.3])
    calls = []

    def fake_collect(posterior, flow, vista_device, cond, n_samples, batch, pool_mode="bounded",
                     show=False, verbose=False):
        calls.append(n_samples)
        return torch.as_tensor(draws, dtype=torch.float64)

    original = ev.collect_theta_prex
    ev.collect_theta_prex = fake_collect
    try:
        posterior = SimpleNamespace(set_default_x=lambda x: None, posterior_estimator=object())
        video = np.zeros((4, 8, 8), dtype=np.uint8)
        scale = np.array([0.3, 1.0, 1.0])
        summary, cloud, sgm = ev.posterior_summary(posterior, video, torch.device("cpu"), torch.device("cpu"),
                                                   64, 16, pool_mode="bounded", quantiles=QUANTILE_LEVELS,
                                                   return_samples=True, return_sgm=True, sgm_scale=scale)
    finally:
        ev.collect_theta_prex = original
    assert calls == [64], calls                                  # ONE collection, whatever its batching
    assert np.allclose(cloud, draws.astype(np.float32))         # the returned cloud IS that collection
    assert np.allclose(summary, np.quantile(draws, list(QUANTILE_LEVELS), axis=0).T)
    vec, idx = ev.sample_geometric_median(draws, scale=scale)
    assert np.array_equal(sgm, vec) and np.array_equal(sgm, draws[idx])   # the SGM is a member of it


# ---- 9. retired option: explicit error, hidden from help ---------------------------------------

def test_retired_summary_option_is_hidden_from_help_but_still_an_error():
    from srm_and_sbi_monomer_dimer_alp.evaluation_runner import build_evaluation_parser
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import build_experiment_parser
    for build in (build_evaluation_parser, build_experiment_parser):
        parser = build()
        assert "--summary" not in parser.format_help()
        try:
            parser.parse_args(["--condition", "FAB", "--total-time-seconds", "2.0", "--summary", "both"])
        except SystemExit:
            continue
        raise AssertionError("--summary accepted")


# ---- 10. second review: execution attempts, run-level optional storage, provenance hashes ----

def test_replacement_shard_from_another_job_or_local_merges_and_keeps_each_execution():
    from srm_and_sbi_monomer_dimer_alp.experiment_support import merge_validated_shards
    rng = np.random.default_rng(22)
    saved = os.environ.pop("SLURM_JOB_ID", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            s0 = _write(tmp, "_shard_00_of_02.npz",
                        product("evaluation", [(0, 0)], rng, rank=0, world_size=2, job_id="100"))
            merged = None
            for replacement_job in ("200", None):       # recomputed in a new Slurm job, then locally
                s1 = _write(tmp, "_shard_01_of_02.npz",
                            product("evaluation", [(0, 1)], rng, rank=1, world_size=2,
                                    job_id=replacement_job))
                os.environ["SLURM_JOB_ID"] = "300"       # the merge step's own job
                merged, man, n = merge_validated_shards([s0, s1], stage="evaluation",
                                                        concat_keys=_EVAL_KEYS,
                                                        expected_ids=[(0, 0), (0, 1)])
                os.environ.pop("SLURM_JOB_ID")
                assert n == 2
                assert [s["execution"]["job_id"] for s in man["shards"]] == ["100", replacement_job]
                assert man["execution"] == {"job_id": "300"}          # the merge's attempt, not a shard's
                assert man["run_identity"] == {"product_label": "P", "invocation_id": "inv-1"}
            # the retained shard provenance is itself validated: counts must add up
            tampered = with_manifest(merged, lambda m: m["shards"][0].update(n_observations=5))
            expect_schema_error(lambda: S.validate_product(tampered, stage="evaluation"), "sum to")
            # the LOGICAL identity is still compared: another invocation, or another product label
            other_inv = _write(tmp, "_shard_01_of_02_inv.npz",
                               product("evaluation", [(0, 1)], rng, invocation="inv-2", job_id="200"))
            expect_schema_error(lambda: merge_validated_shards([s0, other_inv], stage="evaluation",
                                                               concat_keys=_EVAL_KEYS),
                                "contract differs", "run_identity")
            other_label = _write(tmp, "_shard_01_of_02_label.npz",
                                 product("evaluation", [(0, 1)], rng,
                                         run_identity={"product_label": "Q", "invocation_id": "inv-1"}))
            expect_schema_error(lambda: merge_validated_shards([s0, other_label], stage="evaluation",
                                                               concat_keys=_EVAL_KEYS),
                                "contract differs", "run_identity")
        # the job id cannot re-enter the compared identity, and the execution block is exact
        leaked = with_manifest(product("evaluation", [(0, 0)], rng),
                               lambda m: m["run_identity"].update(job_id="100"))
        expect_schema_error(lambda: S.validate_product(leaked, stage="evaluation"),
                            "exactly product_label and invocation_id")
        bad_exec = with_manifest(product("evaluation", [(0, 0)], rng),
                                 lambda m: m.update(execution={"job": "100"}))
        expect_schema_error(lambda: S.validate_product(bad_exec, stage="evaluation"),
                            "execution must carry exactly")
        # the stage-side helpers: logical identity without a job id; the attempt reads the scheduler
        os.environ["SLURM_JOB_ID"] = "400"
        assert S.execution_identity() == {"job_id": "400"}
        assert "job_id" not in S.run_identity("P", distributed=False)
        os.environ.pop("SLURM_JOB_ID")
        assert S.execution_identity() == {"job_id": None}
    finally:
        os.environ.pop("SLURM_JOB_ID", None)
        if saved is not None:
            os.environ["SLURM_JOB_ID"] = saved


def test_optional_draw_storage_is_a_run_level_setting():
    from srm_and_sbi_monomer_dimer_alp.experiment_support import merge_validated_shards
    rng = np.random.default_rng(23)
    cloud = "posterior_samples_cloud"

    def shard(ids, stored):
        a = product("experiment", ids, rng, stored_optional_fields=[cloud] if stored else [])
        if stored:
            a[cloud] = rng.normal(size=(len(ids), 50, D)).astype(np.float32)
        return a

    def empty_shard(stored):
        a = S.empty_product_arrays("experiment", n_parameters=D, run_fields={"kinds": np.asarray(KINDS)})
        a[S.MANIFEST_KEY] = S.encode_manifest(
            valid_manifest("experiment", 0, stored_optional_fields=[cloud] if stored else []))
        if stored:
            a[cloud] = np.empty((0, 50, D), dtype=np.float32)
        return a

    # the declaration and the arrays agree, in both directions
    undeclared = shard([(0, 0, 0)], stored=False); undeclared[cloud] = rng.normal(size=(1, 50, D))
    expect_schema_error(lambda: S.validate_product(undeclared, stage="experiment"), "does not declare")
    missing = shard([(0, 0, 0)], stored=True); missing.pop(cloud)
    expect_schema_error(lambda: S.validate_product(missing, stage="experiment"), "lacks it")

    def merge(paths, optional=(cloud,)):
        return merge_validated_shards(paths, stage="experiment", concat_keys=_EXP_KEYS,
                                      first_keys=["kinds"], optional_concat_keys=list(optional))

    with tempfile.TemporaryDirectory() as tmp:
        # all present, including an empty shard carrying a (0, S, D) cloud
        paths = [_write(tmp, "a0.npz", shard([(0, 0, 0)], True)), _write(tmp, "a1.npz", empty_shard(True)),
                 _write(tmp, "a2.npz", shard([(0, 1, 0), (0, 1, 1)], True))]
        merged, man, _ = merge(paths)
        assert merged[cloud].shape == (3, 50, D) and man["stored_optional_fields"] == [cloud]
        # all absent, including an empty shard
        paths = [_write(tmp, "b0.npz", shard([(0, 0, 0)], False)), _write(tmp, "b1.npz", empty_shard(False))]
        merged, man, _ = merge(paths)
        assert cloud not in merged and man["stored_optional_fields"] == []
        # mixed: refused, and the lacking shard is named
        paths = [_write(tmp, "c0.npz", shard([(0, 0, 0)], True)), _write(tmp, "c1.npz", shard([(0, 1, 0)], False))]
        expect_schema_error(lambda: merge(paths), cloud, "c1.npz", "would silently drop")
        # mixed through an empty shard written without the cloud
        paths = [_write(tmp, "d0.npz", shard([(0, 0, 0)], True)), _write(tmp, "d1.npz", empty_shard(False))]
        expect_schema_error(lambda: merge(paths), cloud, "d1.npz")
        # a merge that does not carry a stored optional array refuses instead of dropping it
        paths = [_write(tmp, "e0.npz", shard([(0, 0, 0)], True))]
        expect_schema_error(lambda: merge(paths, optional=()), "this merge does not carry")


def test_provenance_validation_checks_the_recorded_hashes_not_only_the_flag():
    rng = np.random.default_rng(24)
    good = product("evaluation", [(0, 0)], rng)
    S.validate_product(good, stage="evaluation")
    # the reviewed case: the two aggregate hashes differ while the flag claims no change
    differ = with_manifest(good, lambda m: m["code"]["implementation_at_write"].update(sha256="2" * 64))
    expect_schema_error(lambda: S.validate_product(differ, stage="evaluation"),
                        "changed while the stage ran", "aggregate hashes differ")
    # the aggregates agree but the per-file entries differ
    entries = with_manifest(good, lambda m: m["code"]["implementation_at_write"].update(
        files=[{"path": "x.py", "sha256": "e" * 64}]))
    expect_schema_error(lambda: S.validate_product(entries, stage="evaluation"), "per-file entries differ")
    # the flag says changed although the records agree: the block contradicts itself
    flag = with_manifest(good, lambda m: m["code"].update(changed_during_run=True))
    expect_schema_error(lambda: S.validate_product(flag, stage="evaluation"), "contradicts itself")
    # what the real finalizer writes is accepted, both when nothing changed and after a change
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); (root / "x.py").write_text("a = 1\n")
        start = provenance.code_provenance(root, files=("x.py",))
        ok = with_manifest(good, lambda m: m.update(code=provenance.finalize_code_provenance(
            start, root, files=("x.py",))))
        S.validate_product(ok, stage="evaluation")
        (root / "x.py").write_text("a = 2\n")
        changed = with_manifest(good, lambda m: m.update(code=provenance.finalize_code_provenance(
            start, root, files=("x.py",))))
        expect_schema_error(lambda: S.validate_product(changed, stage="evaluation"),
                            "changed while the stage ran")


if __name__ == "__main__":
    import io, contextlib
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
