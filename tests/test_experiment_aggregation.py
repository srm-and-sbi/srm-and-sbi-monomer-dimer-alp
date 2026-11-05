"""Experimental aggregation (the four constructions, window sets, coverage and contract guards, shard-merge
equality, the extended Experiment report) and the weighted ``empirical_representatives`` Nuisance_DLI
(probabilities, consolidation, overrides, round trip, sampler, loader). Synthetic clouds only, plus one
reproduction of the saved four-cell pilot representatives when the retained collections are present.

Run directly from the repository root::

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/test_experiment_aggregation.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("MACHINE_PROFILE", "mars_pc")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from srm_and_sbi_monomer_dimer_alp import artifact_schema as S                       # noqa: E402
from srm_and_sbi_monomer_dimer_alp import detector_nuisance_dli as ndli              # noqa: E402
from srm_and_sbi_monomer_dimer_alp import experiment_aggregation as agg              # noqa: E402
from srm_and_sbi_monomer_dimer_alp.diagnostics import DiagnosticReporter             # noqa: E402
from srm_and_sbi_monomer_dimer_alp.evaluation import QUANTILE_LEVELS                 # noqa: E402
from srm_and_sbi_monomer_dimer_alp.experiment_support import merge_validated_shards  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS                # noqa: E402
import _product_fixtures as fx                                                       # noqa: E402

# a three-row table of unit prior widths (one log row, two linear rows) so the fixtures' sgm_scale [1, 1, 1] is its widths
TABLE = [{"KEY": "a", "LABEL": "$a$", "PRIOR_RANGE": (-1.0, 0.0), "LOG_FLAG": True, "LOG_BASE": 10},
         {"KEY": "b", "LABEL": "$b$", "PRIOR_RANGE": (0.0, 1.0), "LOG_FLAG": False},
         {"KEY": "c", "LABEL": "$c$", "PRIOR_RANGE": (0.0, 1.0), "LOG_FLAG": False}]
KEYS = [e["KEY"] for e in TABLE]
SDRAWS = 50


def _expect(exc_type, fn, *needles):
    try:
        fn()
    except exc_type as exc:
        for needle in needles:
            assert needle in str(exc), (needle, str(exc))
        return exc
    raise AssertionError(f"expected {exc_type.__name__} with {needles}")


def _cloud(rng, ids):
    """Identifiable clouds: window (kind, cell, chunk) is a tight cluster around (-0.5 - 0.02*cell, 0.1*chunk + 0.3, 0.5)."""
    out = np.empty((len(ids), SDRAWS, 3), np.float32)
    for i, (k, c, w) in enumerate(ids):
        center = np.array([-0.5 - 0.02 * c, 0.1 * w + 0.3, 0.5 + 0.01 * k])
        out[i] = (center + rng.normal(scale=0.004, size=(SDRAWS, 3))).astype(np.float32)
    return out


def _product_with_clouds(rng, ids, kinds=("FAB",)):
    """A valid experiment product whose stored SGM is the exact medoid row of each window's cloud."""
    from srm_and_sbi_monomer_dimer_alp.evaluation import sample_geometric_median
    cloud = _cloud(rng, ids)
    m = fx.valid_manifest("experiment", len(ids), kinds=list(kinds), stored_optional_fields=["posterior_samples_cloud"],
                          window_geometry=dict(GEOMETRY))
    a = fx.product("experiment", ids, rng, manifest=m, kinds=list(kinds))
    a["posterior_samples_cloud"] = cloud
    a["posterior_quantiles"] = np.quantile(cloud.astype(float), list(QUANTILE_LEVELS), axis=1).transpose(1, 2, 0)
    a["posterior_sgm"] = np.asarray([sample_geometric_median(cloud[i].astype(float), scale=np.ones(3))[0] for i in range(len(ids))])
    a["map_estimate"] = a["posterior_sgm"].copy()
    return a


def _collection_file(path, arrays, manifest, *, label=None, seed=1, checkpoint=None):
    """Write a retained draw collection (the Window_Posterior_Draws format) from a product's clouds."""
    from srm_and_sbi_monomer_dimer_alp.evaluation import sample_geometric_median
    cloud = arrays["posterior_samples_cloud"]
    idx = np.asarray([sample_geometric_median(cloud[i].astype(float), scale=np.ones(3))[1] for i in range(cloud.shape[0])])
    man = {"stage": "window_posterior_draws", "parameter_keys": KEYS, "coordinate_transform": manifest["coordinate_transform"],
           "pool_mode": manifest["pool_mode"], "n_summary_draws": cloud.shape[1], "sgm_scale": [1.0, 1.0, 1.0],
           "checkpoint_sha256": checkpoint or manifest["checkpoint_sha256"], "window_geometry": manifest["window_geometry"],
           "input_conversion": "test", "collection_label": label, "seed_policy": S.seed_policy(seed), "written_at": "t",
           "code": {"implementation": {"sha256": "x" * 64}}, "source_product": {"path": "p", "sha256": "y" * 64}}
    np.savez_compressed(path, posterior_samples_cloud=cloud, kind_index=arrays["kind_index"], cell=arrays["cell"], chunk=arrays["chunk"],
                        kinds=arrays["kinds"], fresh_sgm=cloud[np.arange(cloud.shape[0]), idx].astype(float), fresh_sgm_index=idx,
                        parameter_keys=np.asarray(KEYS), sgm_scale=np.ones(3), prior_low=np.array([-1.0, 0.0, 0.0]),
                        prior_high=np.array([0.0, 1.0, 1.0]), collection_manifest_json=np.asarray(json.dumps(man)))
    return path


IDS = [(0, c, w) for c in (3, 7, 11) for w in range(4)]
GEOMETRY = S.window_geometry(n_frames=100, step_frames=100, span_frames=400)     # four positions, 0..3
TARGET = {"project_alias": "T", "condition": "FAB", "timing_label": "2S_50FPS", "n_frames": 100}


def test_window_sets_parse_and_refuse_malformed_items():
    sets = agg.parse_window_sets("all,opening,late=2-3,odd=1+3")
    assert sets == {"all": None, "opening": (0,), "late": (2, 3), "odd": (1, 3)}
    for bad, needle in (("", "no window set"), ("all,all", "named twice"), ("late", "NAME=a-b"), ("opening=1-2", "shadows"),
                        ("x=1+1", "distinct"), ("y=a-b", "integers")):
        _expect(agg.AggregationError, lambda b=bad: agg.parse_window_sets(b), needle)


def test_four_constructions_on_identifiable_clouds_with_equal_weights():
    rng = np.random.default_rng(0)
    a = _product_with_clouds(rng, IDS)
    coll = agg.cloud_from_product(a, S.decode_manifest(a[S.MANIFEST_KEY]))
    assert np.all(coll.window_sgm_index >= 0), "the stored SGM rows are located in the clouds"
    res = agg.aggregate(coll, TABLE, window_sets=agg.parse_window_sets("all,opening,late=2-3"))
    assert res.window_sets == {"all": (0, 1, 2, 3), "opening": (0,), "late": (2, 3)}
    counts = {(c, s): len(res.select(construction=c, window_set=s)) for c in agg.CONSTRUCTIONS for s in res.window_sets}
    assert counts[("per_cell", "all")] == 3 and counts[("per_window_position", "all")] == 4
    assert counts[("global", "all")] == 1 and counts[("per_cell_window", "all")] == 12
    assert counts[("per_cell_window", "late")] == 6 and counts[("per_window_position", "late")] == 2
    cloud = coll.cloud.astype(float)
    for g in res.groups:
        # the SGM is a retained draw of a member window, named by its identity
        rows = [i for i, (k, c, w) in enumerate(IDS) if (c, w) in g.members]
        assert (g.sgm_source["cell"], g.sgm_source["chunk"]) in g.members
        r = IDS.index((0, g.sgm_source["cell"], g.sgm_source["chunk"]))
        assert np.array_equal(cloud[r, g.sgm_source["draw_index"]], g.sgm_log10)
        # equal weights realized by equal draw counts: the pooled quantiles are those of the plain concatenation
        flat = cloud[rows].reshape(-1, 3)
        assert g.n_draws == len(rows) * SDRAWS
        assert np.allclose(g.quantiles_log10, np.quantile(flat, list(QUANTILE_LEVELS), axis=0).T)
        assert np.allclose(g.median_log10, np.median(flat, axis=0))
        if g.construction == "per_cell":
            assert all(c == g.cell for c, _ in g.members) and g.chunk == -1
        if g.construction == "per_window_position":
            assert all(w == g.chunk for _, w in g.members) and g.cell == -1
    # identifiability: a per_cell SGM over positions 0..3 sits in a middle window (the clusters are ordered along b)
    for g in res.select(construction="per_cell", window_set="all"):
        assert g.sgm_source["chunk"] in (1, 2), g.sgm_source
        assert g.sgm_method == "exact_medoid"            # 200 rows: below the exact capacity
    for g in res.select(construction="per_cell_window", window_set="all"):
        assert g.sgm_method == "window_stored"
    assert not agg.verify_members(res, coll)


def test_opening_equivalences_and_membership_follow_the_declared_sets():
    rng = np.random.default_rng(1)
    a = _product_with_clouds(rng, IDS)
    coll = agg.cloud_from_product(a, S.decode_manifest(a[S.MANIFEST_KEY]))
    res = agg.aggregate(coll, TABLE, window_sets=agg.parse_window_sets("all,opening"))
    for g in res.select(construction="per_cell", window_set="opening"):
        twin = [h for h in res.select(construction="per_cell_window", window_set="opening") if h.cell == g.cell][0]
        assert np.array_equal(g.sgm_log10, twin.sgm_log10) and g.equivalent_to == "per_cell_window[opening]"
    glob = res.select(construction="global", window_set="opening")[0]
    pos = res.select(construction="per_window_position", window_set="opening")[0]
    assert np.array_equal(glob.sgm_log10, pos.sgm_log10) and glob.equivalent_to == "per_window_position[opening]"
    assert all(g.members == [(g.cell, 0)] for g in res.select(construction="per_cell", window_set="opening"))
    assert all(len(g.members) == 4 for g in res.select(construction="per_cell", window_set="all"))
    # a single-cell collection: per_window_position equals per_cell_window, global equals per_cell
    one = [(0, 7, w) for w in range(4)]
    b = _product_with_clouds(rng, one)
    r1 = agg.aggregate(agg.cloud_from_product(b, S.decode_manifest(b[S.MANIFEST_KEY])), TABLE, window_sets={"all": None})
    assert r1.select(construction="global", window_set="all")[0].equivalent_to == "per_cell[all]"
    assert all(g.equivalent_to == "per_cell_window[all]" for g in r1.select(construction="per_window_position", window_set="all"))


def test_incomplete_coverage_duplicates_and_incompatible_metadata_are_refused():
    rng = np.random.default_rng(2)
    a = _product_with_clouds(rng, IDS)
    m = S.decode_manifest(a[S.MANIFEST_KEY])
    coll = agg.cloud_from_product(a, m)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # incomplete coverage: drop one window of one cell
        keep = [i for i, (k, c, w) in enumerate(IDS) if not (c == 7 and w == 2)]
        sub = {k: (v[keep] if hasattr(v, "shape") and v.shape[:1] == (len(IDS),) else v) for k, v in a.items()}
        c2 = agg.cloud_from_product(sub, m)
        _expect(agg.AggregationError, lambda: agg.aggregate(c2, TABLE, window_sets={"all": None}), "window set 'all'", "(7, 2)")
        # a window set naming a position beyond the declared geometry
        _expect(agg.AggregationError, lambda: agg.aggregate(coll, TABLE, window_sets={"x": (9,)}), "positions [9] beyond the declared window geometry")
        # duplicate identities across collections
        p1 = _collection_file(tmp / "c1.npz", a, m, label="A")
        p2 = _collection_file(tmp / "c2.npz", a, m, label="B")
        _expect(agg.AggregationError, lambda: agg.combine_clouds([agg.cloud_from_collection(p1), agg.cloud_from_collection(p2)]),
                "duplicate window identities")
        # a different checkpoint is a different computation; a different label or seed is disclosed, not refused
        p3 = _collection_file(tmp / "c3.npz", a, m, label="C", checkpoint="d" * 64)
        _expect(agg.AggregationError, lambda: agg.combine_clouds([agg.cloud_from_collection(p3)],
                                                                 reference_contract={k: m[k] for k in agg.CONTRACT_FIELDS}),
                "checkpoint_sha256")
        other_ids = [(0, 20, w) for w in range(4)]
        b = _product_with_clouds(rng, other_ids)
        p4 = _collection_file(tmp / "c4.npz", b, m, label="D", seed=99)
        merged = agg.combine_clouds([agg.cloud_from_collection(p1), agg.cloud_from_collection(p4)])
        assert merged.n_windows == 16 and len(merged.sources) == 2
        disclosed = agg._disclosed_differences(merged.sources)
        assert set(disclosed) >= {"collection_label", "seed_policy"}
        # the parameter table must match the collection's keys and its prior widths
        _expect(agg.AggregationError, lambda: agg.aggregate(coll, [dict(e, KEY=k) for e, k in zip(TABLE, ["a", "c", "b"])], window_sets={"all": None}),
                "exact order")


def test_single_process_and_merged_shards_give_the_same_grouping():
    rng = np.random.default_rng(3)
    a = _product_with_clouds(rng, IDS)
    m = S.decode_manifest(a[S.MANIFEST_KEY])
    single = agg.aggregate(agg.cloud_from_product(a, m), TABLE, window_sets={"all": None, "opening": (0,)})
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        # two shards in a scrambled window order (the merge concatenates in shard order; the canonical order rules)
        order = [1, 0, 2, 3, 5, 4, 7, 6, 11, 10, 9, 8]
        for r, rows in enumerate((order[:5], order[5:])):
            mm = dict(m, rank=r, world_size=2, n_observations=len(rows))
            sh = {k: (v[rows] if hasattr(v, "shape") and v.shape[:1] == (len(IDS),) else v) for k, v in a.items()}
            sh[S.MANIFEST_KEY] = S.encode_manifest(mm)
            p = tmp / f"_shard_{r:02d}_of_02.npz"; np.savez_compressed(p, **sh); paths.append(p)
        merged, man, n = merge_validated_shards(paths, stage="experiment",
                                                concat_keys=["scores", "map_estimate", "kind_index", "cell", "chunk", "posterior_quantiles", "posterior_sgm"],
                                                first_keys=["kinds"], optional_concat_keys=["posterior_samples_cloud"], expected_ids=IDS)
        assert n == 2
        both = agg.aggregate(agg.cloud_from_product(merged, man), TABLE, window_sets={"all": None, "opening": (0,)})
        assert len(single.groups) == len(both.groups)
        for g, h in zip(single.groups, both.groups):
            assert (g.construction, g.window_set, g.cell, g.chunk, g.members) == (h.construction, h.window_set, h.cell, h.chunk, h.members)
            assert np.array_equal(g.sgm_log10, h.sgm_log10) and g.sgm_source["draw_index"] == h.sgm_source["draw_index"]
            assert np.allclose(g.quantiles_log10, h.quantiles_log10)
        # the mixed-storage guard still holds: a shard without the clouds is refused
        sh = {k: v for k, v in np.load(paths[1]).items() if k != "posterior_samples_cloud"}
        mm = dict(m, rank=1, world_size=2, n_observations=7, stored_optional_fields=[])
        sh[S.MANIFEST_KEY] = S.encode_manifest(mm); np.savez_compressed(paths[1], **sh)
        _expect(S.SchemaError, lambda: merge_validated_shards(paths, stage="experiment",
                concat_keys=["scores", "map_estimate", "kind_index", "cell", "chunk", "posterior_quantiles", "posterior_sgm"],
                first_keys=["kinds"], optional_concat_keys=["posterior_samples_cloud"], expected_ids=IDS), "posterior_samples_cloud")


def test_exports_round_trip_and_the_archive_marks_medians_as_composites():
    rng = np.random.default_rng(4)
    a = _product_with_clouds(rng, IDS)
    m = S.decode_manifest(a[S.MANIFEST_KEY])
    coll = agg.cloud_from_product(a, m)
    res = agg.aggregate(coll, TABLE, window_sets={"all": None, "opening": (0,)})
    with tempfile.TemporaryDirectory() as tmp:
        man = agg.export_manifest(res, product_label="P", condition_labels=["FAB"], target=TARGET)
        paths = agg.write_aggregation_exports(res, tmp, TABLE, manifest=man)
        arc = agg.load_candidate_archive(paths["archive"])
        assert arc["manifest"]["product_label"] == "P" and arc["keys"] == KEYS and arc["manifest"]["target"] == TARGET
        comp = agg.archive_component(arc, kind="FAB", window_set="all", construction="per_cell")
        assert comp["vectors"].shape == (3, 3) and all(i["is_sampled_rows"] for i in comp["identities"])
        med = agg.archive_component(arc, kind="FAB", window_set="all", construction="per_cell", statistic="median")
        assert not any(i["is_sampled_rows"] for i in med["identities"])
        for g, v in zip(res.select(construction="per_cell", window_set="all"), comp["vectors"]):
            assert np.array_equal(g.sgm_log10, v)
        # the stored pooled quantiles recompute from the clouds
        g = res.select(construction="global", window_set="all")[0]
        flat = coll.cloud.astype(float).reshape(-1, 3)
        idx = [i for i, (k, c, w) in enumerate(IDS)]
        assert np.allclose(arc["arrays"]["quantiles_log10"][comp["group_indices"][0]],
                           np.quantile(coll.cloud.astype(float)[[i for i, (k, c, w) in enumerate(IDS) if c == 3]].reshape(-1, 3), list(QUANTILE_LEVELS), axis=0).T)
        assert np.allclose(g.quantiles_log10, np.quantile(flat, list(QUANTILE_LEVELS), axis=0).T)
        _expect(agg.AggregationError, lambda: agg.archive_component(arc, kind="FAB", window_set="late", construction="per_cell"), "no component")
        assert (Path(tmp) / "aggregation_comparison.csv").read_text().count("\n") > 10
        assert json.loads((Path(tmp) / "aggregation_manifest.json").read_text())["archive_sha256"] == arc["sha256"]


def test_weighted_quantile_is_the_generalized_inverse():
    v, w = [1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]
    assert [agg.weighted_quantile(v, w, p) for p in (0.05, 0.3, 0.5, 0.95)] == [1.0, 2.0, 3.0, 4.0]
    assert agg.weighted_quantile([5.0], [1.0], 0.5) == 5.0 == agg.weighted_quantile([5.0], [1.0], 0.99)
    s = agg.representative_summary(np.array([[0.0, 0.0], [2.0, 2.0]]), [0.75, 0.25], np.array([-1, -1]), np.array([1, 1]))
    assert s["quantiles"][0].tolist() == [0.0, 0.0, 0.0, 0.0, 2.0] and s["outside_any"] == 0.25 and s["outside_by_key"].tolist() == [0.25, 0.25]


def test_extended_report_contains_the_section_for_live_and_saved_rendering():
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import write_experiment_outputs
    eval_cfg = PARAMETERS.inference.evaluation
    rng = np.random.default_rng(5)
    a = _product_with_clouds(rng, IDS)
    m = S.decode_manifest(a[S.MANIFEST_KEY])
    xargs = SimpleNamespace(aggregation="pooled", seed=None, condition="FAB")
    ref = {"name": "REFX", "log10": np.array([-0.5, 0.4, 0.5]), "physical": np.array([10 ** -0.5, 0.4, 0.5]), "identity": {"artifact": "x"}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # live path: the product's own clouds
        rep = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "live", run_label="T", timestamp="now")
        req = agg.AggregationRequest(window_sets=agg.parse_window_sets("all,opening"), reference=ref, window_seconds=2.0, product_label="P", target=TARGET)
        write_experiment_outputs(rep, xargs, eval_cfg, TABLE, tmp / "live" / "p.npz", a, run_start=0.0, aggregation=req)
        text = (tmp / "live" / "report.md").read_text()
        assert "## Experimental aggregation and nuisance candidates" in text
        for c in agg.CONSTRUCTIONS:
            assert f"| `{c}` |" in text
        for s in ("window set all]", "window set opening]"):
            assert s in text
        for f in ("aggregation_FAB_all_marginals", "aggregation_FAB_all_joint", "aggregation_FAB_opening_marginals", "aggregation_FAB_all_temporal",
                  "aggregation_FAB_window_selection"):
            assert (tmp / "live" / "figures" / f"{f}.png").exists() and f"figures/{f}.png" in text
        for f in ("aggregation_candidates.npz", "aggregation_groups.csv", "aggregation_comparison.csv", "aggregation_manifest.json"):
            assert (tmp / "live" / f).exists() and f in text
        assert "REFX" in text and "source uncertainty against representative dispersion" in text
        assert "[aggregation_candidates.npz](aggregation_candidates.npz)" in text and "aggregation_identities_in_product" in text
        assert "every position of the declared window geometry" in text and "positions [0, 1, 2, 3]" in text
        live = agg.load_candidate_archive(tmp / "live" / "aggregation_candidates.npz")
        # saved path: the same writer, the clouds from a retained collection, a coverage statement
        p = _collection_file(tmp / "coll.npz", a, m, label="X")
        clouds = agg.combine_clouds([agg.cloud_from_collection(p)], reference_contract={k: m[k] for k in agg.CONTRACT_FIELDS})
        rep2 = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "saved", run_label="T", timestamp="now")
        req2 = agg.AggregationRequest(window_sets=agg.parse_window_sets("all,opening"), clouds=clouds, reference=ref, window_seconds=2.0,
                                      product_label="P", coverage_note="PILOT SUBSET NOTE", source_clouds_note="fresh collections")
        write_experiment_outputs(rep2, xargs, eval_cfg, TABLE, tmp / "saved" / "p.npz", a, run_start=0.0, persist_arrays=False, aggregation=req2)
        text2 = (tmp / "saved" / "report.md").read_text()
        assert "PILOT SUBSET NOTE" in text2 and "coll.npz" in text2 and "collection label X" in text2
        saved = agg.load_candidate_archive(tmp / "saved" / "aggregation_candidates.npz")
        assert np.array_equal(live["arrays"]["sgm_log10"], saved["arrays"]["sgm_log10"])
        # disabled: the statement, no substitute, no exports
        rep3 = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "off", run_label="T", timestamp="now")
        write_experiment_outputs(rep3, xargs, eval_cfg, TABLE, tmp / "off" / "p.npz", a, run_start=0.0,
                                 aggregation=agg.AggregationRequest(window_sets={}, enabled=False))
        text3 = (tmp / "off" / "report.md").read_text()
        assert "Not computed: aggregation was disabled" in text3 and not (tmp / "off" / "aggregation_candidates.npz").exists()
        # a product without clouds and no request: the statement, nothing invented
        b = fx.product("experiment", IDS, rng)
        rep4 = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "none", run_label="T", timestamp="now")
        write_experiment_outputs(rep4, xargs, eval_cfg, TABLE, tmp / "none" / "p.npz", b, run_start=0.0,
                                 aggregation=agg.AggregationRequest(window_sets={"all": None}))
        assert "Not computed: experiment product: no retained clouds" in (tmp / "none" / "report.md").read_text()


def _archive(tmp, rng):
    a = _product_with_clouds(rng, IDS)
    res = agg.aggregate(agg.cloud_from_product(a, S.decode_manifest(a[S.MANIFEST_KEY])), TABLE, window_sets={"all": None, "opening": (0,)})
    man = agg.export_manifest(res, product_label="P", condition_labels=["FAB"], target=TARGET)
    paths = agg.write_aggregation_exports(res, tmp, TABLE, manifest=man)
    return agg.load_candidate_archive(paths["archive"]), res


def _comp(grouping, weight, window_set="all", statistic="sgm"):
    return {"source_experiment": "P", "condition": "FAB", "window_set": window_set, "grouping": grouping, "statistic": statistic, "weight": weight}


def test_mixture_probabilities_consolidation_overrides_and_invalid_selections():
    rng = np.random.default_rng(6)
    PLO, PHI = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])
    with tempfile.TemporaryDirectory() as tmp:
        arc, res = _archive(tmp, rng)
        spec = {"block": {"posterior_sample_pool_choice": "empirical_representatives", "candidate_archive": arc["path"],
                          "candidate_archive_sha256": arc["sha256"], "rationale": "a test selection"},
                "component": [_comp("per_cell", 0.6), _comp("global", 0.4)]}
        vec, pr, rec = ndli.representatives_from_spec(spec, arc, TABLE, PLO, PHI, target=TARGET)
        assert vec.shape == (4, 3) and np.isclose(pr.sum(), 1.0)
        assert np.allclose([c["member_probability"] for c in rec["components"]], [0.2, 0.4]) and rec["is_sampled_rows"]
        # the global SGM is one of the per_cell SGMs? (not necessarily) -- check the probabilities per row from memberships
        glob = res.select(construction="global", window_set="all")[0].sgm_log10
        i = [k for k in range(4) if np.array_equal(vec[k], glob)]
        assert len(i) == 1 and (np.isclose(pr[i[0]], 0.4) or np.isclose(pr[i[0]], 0.6))
        # a singleton component with weight one
        v1, p1, r1 = ndli.representatives_from_spec({**spec, "component": [_comp("global", 1.0)]}, arc, TABLE, PLO, PHI, target=TARGET)
        assert v1.shape == (1, 3) and p1.tolist() == [1.0] and r1["n_unique_rows"] == 1
        # overlapping components: per_cell[opening] equals per_cell_window[opening]; duplicates consolidate, probabilities add
        v2, p2, r2 = ndli.representatives_from_spec({**spec, "component": [_comp("per_cell", 0.5, "opening"), _comp("per_cell_window", 0.5, "opening")]},
                                                     arc, TABLE, PLO, PHI, target=TARGET)
        assert v2.shape == (3, 3) and np.allclose(p2, 1 / 3) and len(r2["consolidated"]) == 3 and r2["n_member_rows"] == 6
        assert all(len(c["memberships"]) == 2 for c in r2["consolidated"])
        # a median statistic is a composite: the record says the rows are not all sampled
        v3, p3, r3 = ndli.representatives_from_spec({**spec, "component": [_comp("per_cell", 1.0, statistic="median")]}, arc, TABLE, PLO, PHI, target=TARGET)
        assert not r3["is_sampled_rows"]
        # an explicit override is applied to every vector and recorded with the replaced range
        v4, p4, r4 = ndli.representatives_from_spec({**spec, "override": {"b": {"value": 0.25, "reason": "declared"}}}, arc, TABLE, PLO, PHI, target=TARGET)
        assert np.all(v4[:, 1] == 0.25) and r4["overrides"]["b"]["replaced_rows"] == 4 and not r4["is_sampled_rows"]
        # invalid selections fail clearly
        for bad, needle in ((dict(spec, component=[_comp("per_cell", 0.5), _comp("global", 0.4)]), "sum to 0.9"),
                            (dict(spec, component=[_comp("per_cell", -0.2), _comp("global", 1.2)]), "non-negative"),
                            (dict(spec, component=[_comp("per_cell", 0.0), _comp("global", 0.0)]), "positive total mass"),
                            (dict(spec, component=[_comp("sgm_of_sgms", 1.0)]), "grouping"),
                            (dict(spec, component=[_comp("per_cell", 1.0, statistic="map")]), "statistic"),
                            (dict(spec, component=[_comp("per_cell", 1.0, window_set="late")]), "no component"),
                            (dict(spec, component=[dict(_comp("per_cell", 1.0), source_experiment="Q")]), "not the archive's product"),
                            (dict(spec, block=dict(spec["block"], rationale="")), "rationale"),
                            (dict(spec, block=dict(spec["block"], candidate_archive_sha256="0" * 64)), "checksum"),
                            (dict(spec, override={"zz": {"value": 1, "reason": "r"}}), "unknown parameter"),
                            (dict(spec, override={"b": {"value": 0.25}}), "reason")):
            _expect(ValueError, lambda b=bad: ndli.representatives_from_spec(b, arc, TABLE, PLO, PHI, target=TARGET), needle)
        # the spec loader validates the same block from a TOML file
        p = Path(tmp) / "s.toml"
        p.write_text('[block]\nposterior_sample_pool_choice = "empirical_representatives"\ncandidate_archive = "x.npz"\nrationale = "r"\n'
                     '[[component]]\nsource_experiment = "P"\ncondition = "FAB"\nwindow_set = "all"\ngrouping = "per_cell"\nweight = 1.0\n')
        ndli.load_spec(p, KEYS, PLO, PHI, table=TABLE)
        p.write_text(p.read_text().replace("weight = 1.0", "weight = 0.5"))
        _expect(ValueError, lambda: ndli.load_spec(p, KEYS, PLO, PHI, table=TABLE), "sum to 0.5")


def test_weighted_artifact_round_trip_sampler_loader_and_legacy_uniformity():
    rng = np.random.default_rng(7)
    PLO, PHI = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])
    with tempfile.TemporaryDirectory() as tmp:
        arc, res = _archive(tmp, rng)
        spec = {"block": {"posterior_sample_pool_choice": "empirical_representatives", "candidate_archive": arc["path"], "rationale": "t"},
                "component": [_comp("per_cell", 0.9), _comp("global", 0.1)]}
        nu = ndli.build_nuisance_dli(spec, KEYS, PLO, PHI, table=TABLE, archive=arc, target=TARGET)
        assert nu.posterior_sample_pool_choice == "empirical_representatives" and nu.probabilities.shape == (nu.samples.shape[0],)
        assert nu.manifest()["weighted"] and nu.manifest()["n_components"] == 2 and nu.identity()["n_components"] == 2
        art = Path(tmp) / "T_Nuisance_DLI.npz"
        nu.flush(art)
        back = ndli.NuisanceDLI.load(art)
        assert np.array_equal(back.samples, nu.samples) and np.array_equal(back.probabilities, nu.probabilities)
        assert back.parameter_keys == KEYS and back.selection["components"][0]["grouping"] == "per_cell"
        assert back.selection["archive"]["sha256"] == arc["sha256"] and back.selection["rationale"] == "t"
        # the sampler returns complete stored rows at the recorded probabilities (seeded frequency check)
        draws = back.sample(40000, seed=0)
        assert all(any(np.array_equal(d, r) for r in back.samples) for d in draws[:200])
        freq = np.array([np.mean(np.all(draws == r, axis=1)) for r in back.samples])
        assert np.allclose(freq, back.probabilities, atol=0.01), (freq, back.probabilities)
        # the production loader accepts it from an isolated test destination
        loaded = ndli.require_nuisance_dli(Path(tmp), "T", "2S", nuisance_tag=None) if False else ndli.NuisanceDLI.load(art)
        assert loaded.identity(art)["weighted"] is True
        # an unfamiliar weighted construction is refused, never sampled uniformly
        _expect(ValueError, lambda: ndli.NuisanceDLI(KEYS, "empirical_representatives_v9", "bounded", PLO, PHI,
                                                     samples=nu.samples, probabilities=nu.probabilities), "must be one of")
        _expect(ValueError, lambda: ndli.NuisanceDLI(KEYS, "empirical_representatives", "bounded", PLO, PHI, samples=nu.samples,
                                                     selection=nu.selection), "one sampling probability per stored row")
        _expect(ValueError, lambda: ndli.NuisanceDLI(KEYS, "raw", "bounded", PLO, PHI, samples=nu.samples, probabilities=nu.probabilities),
                "carries no per-row probabilities")
        # a legacy unweighted artifact keeps uniform sampling
        raw = ndli.NuisanceDLI.from_samples(KEYS, nu.samples, choice="raw", prior_low=PLO, prior_high=PHI)
        raw.flush(Path(tmp) / "R.npz"); r2 = ndli.NuisanceDLI.load(Path(tmp) / "R.npz")
        assert r2.probabilities is None and not r2.manifest()["weighted"]
        d = r2.sample(20000, seed=1)
        freq = np.array([np.mean(np.all(d == r, axis=1)) for r in r2.samples])
        assert np.allclose(freq, 1 / r2.samples.shape[0], atol=0.02)


def test_r3_a_position_missing_from_every_cell_is_refused_for_all_and_named_subsets_keep_their_scope():
    rng = np.random.default_rng(31)
    a = _product_with_clouds(rng, IDS)
    m = S.decode_manifest(a[S.MANIFEST_KEY])
    # drop position 2 from EVERY cell: the declared geometry still has four positions
    keep = [i for i, (k, c, w) in enumerate(IDS) if w != 2]
    sub = {k: (v[keep] if hasattr(v, "shape") and v.shape[:1] == (len(IDS),) else v) for k, v in a.items()}
    coll = agg.cloud_from_product(sub, m)
    _expect(agg.AggregationError, lambda: agg.aggregate(coll, TABLE, window_sets={"all": None}),
            "window set 'all' = positions [0, 1, 2, 3]", "(3, 2)", "named explicitly")
    # an explicit set excluding it has the declared scope and the equal weights; coverage records what is present
    res = agg.aggregate(coll, TABLE, window_sets=agg.parse_window_sets("opening,rest=0+1+3"))
    assert res.window_sets == {"opening": (0,), "rest": (0, 1, 3)} and res.coverage["FAB"]["chunks"] == [0, 1, 3]
    assert res.coverage["FAB"]["declared_positions"] == [0, 1, 2, 3]
    for g in res.select(construction="per_cell", window_set="rest"):
        assert [w for _, w in g.members] == [0, 1, 3] and g.n_draws == 3 * SDRAWS
    # a set naming the missing position is refused for the same reason; one beyond the geometry at preflight
    _expect(agg.AggregationError, lambda: agg.aggregate(coll, TABLE, window_sets={"mid": (1, 2)}), "window set 'mid'", "(3, 2)")
    _expect(agg.AggregationError, lambda: agg.validate_window_sets_against_geometry({"late": (3, 4)}, GEOMETRY), "positions [4] beyond")
    assert agg.validate_window_sets_against_geometry({"all": None, "late": (2, 3)}, GEOMETRY) == {"all": (0, 1, 2, 3), "late": (2, 3)}
    assert agg.expected_positions(S.window_geometry(n_frames=100, step_frames=100, span_frames=1000)) == tuple(range(10))
    assert agg.expected_positions(S.window_geometry(n_frames=100, step_frames=50, span_frames=1000)) == tuple(range(19))
    assert agg.expected_positions(S.window_geometry(n_frames=100, step_frames=100, span_frames=None), observed=[0, 1, 5]) == (0, 1, 5)
    # identities that are not windows of the source product are refused by the report writer
    other = _product_with_clouds(rng, [(0, 99, w) for w in range(4)])
    foreign = agg.cloud_from_product(other, m)
    assert agg.verify_identities_in_product(foreign, a) == [("FAB", 99, w) for w in range(4)]
    assert not agg.verify_identities_in_product(agg.cloud_from_product(a, m), a)
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import write_experiment_outputs
    with tempfile.TemporaryDirectory() as tmp:
        rep = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=Path(tmp), run_label="T", timestamp="now")
        req = agg.AggregationRequest(window_sets={"all": None}, clouds=foreign, product_label="P", target=TARGET)
        _expect(SystemExit, lambda: write_experiment_outputs(rep, SimpleNamespace(aggregation="pooled", seed=None, condition="FAB"),
                                                             PARAMETERS.inference.evaluation, TABLE, Path(tmp) / "p.npz", a, run_start=0.0,
                                                             persist_arrays=False, aggregation=req), "not windows of this")


def test_r4_conflicting_input_conversions_are_refused_even_when_the_reference_product_lacks_the_field():
    rng = np.random.default_rng(41)
    a = _product_with_clouds(rng, IDS)
    m = S.decode_manifest(a[S.MANIFEST_KEY])
    assert m.get("input_conversion") is None, "the fixture product records no input conversion (the legacy case)"
    reference = {k: m[k] for k in agg.CONTRACT_FIELDS}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p1 = _collection_file(tmp / "c1.npz", a, m, label="A")
        b = _product_with_clouds(rng, [(0, 20, w) for w in range(4)])
        p2 = _collection_file(tmp / "c2.npz", b, m, label="B")
        c1, c2 = agg.cloud_from_collection(p1), agg.cloud_from_collection(p2)
        # identical recorded conversions: accepted, verified and kept in the merged contract
        merged = agg.combine_clouds([c1, c2], reference_contract=reference)
        assert merged.contract["input_conversion"] == "test" and merged.contract["input_conversion_status"].startswith("verified")
        assert "does not record one" in merged.contract["input_conversion_status"]
        # a conflicting conversion on one collection is refused, with or without the reference field
        from dataclasses import replace
        bad = replace(c2, contract={**c2.contract, "input_conversion": "INCOMPATIBLE"})
        _expect(agg.AggregationError, lambda: agg.combine_clouds([c1, bad], reference_contract=reference), "input_conversion differs", "INCOMPATIBLE")
        _expect(agg.AggregationError, lambda: agg.combine_clouds([c1, bad]), "input_conversion differs")
        # a reference that records a different conversion is refused as before
        _expect(agg.AggregationError, lambda: agg.combine_clouds([c1, c2], reference_contract={**reference, "input_conversion": "other"}), "input_conversion")
        # unavailable metadata is stated, not called verified
        none = replace(c2, contract={**c2.contract, "input_conversion": None})
        m2 = agg.combine_clouds([c1, none], reference_contract=reference)
        assert m2.contract["input_conversion"] == "test" and m2.contract["input_conversion_status"].startswith("unverified")
        # the product's own clouds state that the conversion is unavailable
        assert agg.cloud_from_product(a, m).contract["input_conversion_status"].startswith("unavailable")


def test_r1_the_archive_is_bound_to_its_target_condition_and_timing():
    rng = np.random.default_rng(11)
    PLO, PHI = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])
    with tempfile.TemporaryDirectory() as tmp:
        arc, res = _archive(tmp, rng)
        spec = {"block": {"posterior_sample_pool_choice": "empirical_representatives", "candidate_archive": arc["path"], "rationale": "t"},
                "component": [_comp("per_cell", 1.0)]}
        ok = ndli.representatives_from_spec(spec, arc, TABLE, PLO, PHI, target=TARGET)
        assert ok[2]["target"]["condition"] == "FAB" and ok[2]["archive"]["target"] == TARGET
        # another condition, another duration, another window length, no target: refused before anything is built
        for target, needle in ((dict(TARGET, condition="INLB"), "target condition 'INLB' != the archive's 'FAB'"),
                               (dict(TARGET, timing_label="5S_50FPS"), "timing_label '5S_50FPS'"),
                               (dict(TARGET, n_frames=250), "window length 250"),
                               (None, "passed explicitly"),
                               ({"condition": "FAB"}, "passed explicitly")):
            _expect(ValueError, lambda t=target: ndli.representatives_from_spec(spec, arc, TABLE, PLO, PHI, target=t), needle)
            _expect(ValueError, lambda t=target: ndli.build_nuisance_dli(spec, KEYS, PLO, PHI, table=TABLE, archive=arc, target=t), needle)
            _expect(ValueError, lambda t=target: ndli.emit_representatives_template(Path(tmp) / "never.toml", arc, target=t), needle)
            assert not (Path(tmp) / "never.toml").exists()
        # an archive without the binding is refused
        unbound = {**arc, "manifest": {k: v for k, v in arc["manifest"].items() if k != "target"}}
        _expect(ValueError, lambda: ndli.representatives_from_spec(spec, unbound, TABLE, PLO, PHI, target=TARGET), "records no target binding")
        # a cross-condition application (the estimator applied to INLB recordings) stays an inspection result
        cross = _product_with_clouds(rng, [(0, c, w) for c in (3, 7) for w in range(4)] + [(1, 5, w) for w in range(4)], kinds=("FAB", "INLB"))
        r2 = agg.aggregate(agg.cloud_from_product(cross, S.decode_manifest(cross[S.MANIFEST_KEY])), TABLE, window_sets={"all": None})
        man = agg.export_manifest(r2, product_label="P", condition_labels=["FAB", "INLB"], target=TARGET)
        paths = agg.write_aggregation_exports(r2, Path(tmp) / "cross", TABLE, manifest=man)
        arc2 = agg.load_candidate_archive(paths["archive"])
        inlb = {**spec, "component": [dict(_comp("per_cell", 1.0), condition="INLB")]}
        _expect(ValueError, lambda: ndli.representatives_from_spec(inlb, arc2, TABLE, PLO, PHI, target=TARGET), "cannot enter the native Nuisance DLI of 'FAB'")
        _expect(ValueError, lambda: ndli.representatives_from_spec(inlb, arc2, TABLE, PLO, PHI, target=dict(TARGET, condition="INLB")),
                "target condition 'INLB' != the archive's 'FAB'")
        assert ndli.representatives_from_spec(spec, arc2, TABLE, PLO, PHI, target=TARGET)[0].shape == (2, 3)   # the FAB cells, native


def test_r5_r6_override_domain_and_probability_roundoff_are_handled_consistently():
    rng = np.random.default_rng(56)
    PLO, PHI = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])
    # the shared physical-domain rule
    bleach = {"KEY": "prob_photo_bleach", "PRIOR_RANGE": (-2.0, -0.5), "LOG_FLAG": True}
    assert ndli.physical_domain_error("prob_photo_bleach", 1.2, bleach) == "prob_photo_bleach = 1.2 must be a probability below 1"
    assert ndli.physical_domain_error("prob_photo_bleach", 1.0, bleach) and ndli.physical_domain_error("prob_photo_bleach", 0.0, bleach)
    assert ndli.physical_domain_error("prob_photo_bleach", 0.05, bleach) == "" and ndli.physical_domain_error("mu_r", 2.256, {"PRIOR_RANGE": (0, 0.3), "LOG_FLAG": True}) == ""
    assert "positive" in ndli.physical_domain_error("mu_r", -1.0, {"PRIOR_RANGE": (0, 0.3), "LOG_FLAG": True}) and "finite" in ndli.physical_domain_error("mu_r", float("nan"))
    with tempfile.TemporaryDirectory() as tmp:
        arc, res = _archive(tmp, rng)
        table = [dict(TABLE[0], KEY="prob_photo_bleach"), TABLE[1], TABLE[2]]      # the log row a is a probability here
        arc_p = {**arc, "keys": ["prob_photo_bleach", "b", "c"]}
        spec = {"block": {"posterior_sample_pool_choice": "empirical_representatives", "candidate_archive": arc["path"], "rationale": "t"},
                "component": [_comp("per_cell", 1.0)]}
        # an invalid bleaching override is refused; the documented 0.05 treatment stays available
        bad = {**spec, "override": {"prob_photo_bleach": {"value": 1.2, "reason": "negative test"}}}
        _expect(ValueError, lambda: ndli.representatives_from_spec(bad, arc_p, table, PLO, PHI, target=TARGET), "probability below 1")
        _expect(ValueError, lambda: ndli.load_spec(_toml(Path(tmp) / "bad.toml", bad), ["prob_photo_bleach", "b", "c"], PLO, PHI, table=table), "probability below 1")
        good = {**spec, "override": {"prob_photo_bleach": {"value": 0.05, "reason": "the documented reference interval treatment"}}}
        v, p, r = ndli.representatives_from_spec(good, arc_p, table, PLO, PHI, target=TARGET)
        assert np.allclose(v[:, 0], np.log10(0.05)) and r["overrides"]["prob_photo_bleach"]["value_physical"] == 0.05
        # an acknowledged extrapolation of a physically valid value is a different case and is accepted (outside mass recorded)
        wide = {**spec, "override": {"b": {"value": 1.5, "reason": "beyond the training prior, physically valid"}}}
        v2, p2, r2 = ndli.representatives_from_spec(wide, arc, TABLE, PLO, PHI, target=TARGET)
        assert r2["outside_prior_mass"] == 1.0 and ndli.build_nuisance_dli(wide, KEYS, PLO, PHI, table=TABLE, archive=arc, target=TARGET).pool_mode == "unrestricted"
        # the constructor refuses a stored probability parameter at or above one in an explicitly selected artifact
        _expect(ValueError, lambda: ndli.NuisanceDLI.from_representatives(["prob_photo_bleach", "b", "c"], np.array([[0.1, 0.5, 0.5]]), [1.0],
                                                                           {"kind": "representatives", "components": []}, prior_low=PLO, prior_high=PHI), "probability")
        _expect(ValueError, lambda: ndli.selection_from_spec({"selection": {"prob_photo_bleach": {"value": 1.2, "source": "s", "limitation": "l"},
                                                                            "b": {"value": 0.5, "source": "s", "limitation": "l"}, "c": {"value": 0.5, "source": "s", "limitation": "l"}}},
                                                             table, PLO, PHI), "probability below 1")
        # R6: a total within the tolerance is accepted, normalized before storage, sampled and round-tripped
        vecs = res.select(construction="per_cell", window_set="all")
        V = np.asarray([g.sgm_log10 for g in vecs[:2]])
        nu = ndli.NuisanceDLI.from_representatives(KEYS, V, [0.6, 0.4000005], {"kind": "representatives", "components": []}, prior_low=PLO, prior_high=PHI)
        assert abs(nu.probabilities.sum() - 1.0) < 1e-15 and abs(nu.probability_residual - 5e-7) < 1e-9 and np.allclose(nu.probabilities, [0.6, 0.4], atol=1e-6)
        draws = nu.sample(2000, seed=0)
        assert abs(np.mean(np.all(draws == V[0], axis=1)) - 0.6) < 0.05
        nu.flush(Path(tmp) / "W.npz"); back = ndli.NuisanceDLI.load(Path(tmp) / "W.npz")
        assert np.array_equal(back.probabilities, nu.probabilities) and back.sample(5, seed=1).shape == (5, 3)
        assert "divided by their sum" in back.manifest()["probability_normalization"]
        # a materially different total is still refused
        _expect(ValueError, lambda: ndli.NuisanceDLI.from_representatives(KEYS, V, [0.6, 0.41], {"kind": "representatives", "components": []},
                                                                           prior_low=PLO, prior_high=PHI), "not 1 within")


def _toml(path, spec):
    lines = ["[block]"] + [f'{k} = "{v}"' for k, v in spec["block"].items()]
    for c in spec["component"]:
        lines += ["", "[[component]]"] + [f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}" for k, v in c.items()]
    for k, o in (spec.get("override") or {}).items():
        lines += ["", f"[override.{k}]", f"value = {o['value']}", f'reason = "{o["reason"]}"']
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def test_r7_the_selection_console_and_the_exports_share_one_median_convention():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp:
        arc, res = _archive(tmp, rng)
        rows = agg.archive_summary_rows(arc, TABLE, statistic="sgm")
        by = {(r["kind"], r["window_set"], r["construction"]): r for r in rows}
        comp = {(r[0], r[1], r[2], r[3], r[4]): r for r in agg.comparison_rows(res, TABLE)}
        for (kind, ws, c), r in by.items():
            for j, key in enumerate(KEYS):
                med = r["quantiles_physical"][j, r["levels"].index(0.5)]
                assert float(comp[(kind, ws, c, "sgm", key)][6]) == med        # the exported comparison's median_physical
        # an even member count: the inverse-CDF median is a member value, never a midpoint average
        pc = by[("FAB", "all", "per_cell")]
        assert pc["n"] == 3
        two = agg.representative_summary(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), None, np.zeros(3), np.ones(3))
        assert two["quantiles"][0].tolist()[2] == 0.0 and np.median([0.0, 1.0]) == 0.5


def test_the_saved_four_cell_pilot_representatives_reproduce():
    """Reads the retained collections of the pilots (present on the PC); skipped elsewhere."""
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    ev = REPO / "Script_Bank/Analysis/Evidence"
    files = {("FAB", 26): ev / "Window_Posterior_Pilot_2026-11-03/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws.npz",
             ("FAB", 4): ev / "Window_Posterior_Additional_Cells_2026-11-03/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_C128V42A_Window_Posterior_Draws_CELL4.npz",
             ("INLB", 11): ev / "Window_Posterior_Pilot_2026-11-03/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws.npz",
             ("INLB", 42): ev / "Window_Posterior_Additional_Cells_2026-11-03/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_INLB_2S_50FPS_C128V42AP3LR32_Window_Posterior_Draws_CELL42.npz"}
    if not all(p.exists() for p in files.values()):
        print("  (skipped: the retained pilot collections are not present)")
        return
    expected = {("FAB", 26): (1, 3422), ("FAB", 4): (3, 7806), ("INLB", 11): (7, 2186), ("INLB", 42): (8, 1582)}
    table = det.DETECTOR_PARAMETERIZATION
    for cond in ("FAB", "INLB"):
        colls = [agg.cloud_from_collection(p) for (k, c), p in files.items() if k == cond]
        merged = agg.combine_clouds(colls)
        res = agg.aggregate(merged, table, window_sets={"all": None})
        for g in res.select(construction="per_cell", window_set="all"):
            chunk, draw = expected[(cond, g.cell)]
            assert (g.sgm_source["chunk"], g.sgm_source["draw_index"]) == (chunk, draw), (cond, g.cell, g.sgm_source)
            assert g.sgm_method == "weiszfeld_snap" and g.n_draws == 100000
            src = [c for c in colls if int(c.cell[0]) == g.cell][0]
            row = src.cloud[list(src.chunk).index(chunk), draw].astype(float)
            assert np.array_equal(row, g.sgm_log10)
        assert not agg.verify_members(res, merged)



def test_presentation_labels_scope_lines_and_the_early_scope_table():
    """The visible language (plan 2026-11-05): construction descriptions with the statistic and the vector count,
    window-set phrases from the declared geometry, scope lines that count the actual coverage (a subset is a pilot,
    the complete product is not) and name the selected windows of a subset, the early Data scope table with the
    two scopes apart, the opening equivalences and the median-statistic label; figures carry the scope subtitle."""
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import write_experiment_outputs
    # labels
    assert agg.display_name("per_cell") == "One SGM vector per cell"
    assert agg.display_name("per_cell", short=True, n=60) == "SGM per cell (60 vectors)"
    assert agg.display_name("global", short=True, n=1) == "SGM for included dataset (one vector)"
    assert agg.display_name("per_cell_window", short=True) == "SGM per cell–window"
    assert agg.display_name("reference", short=True, reference_name="SEL") == "Fixed reference: SEL"
    med = agg.display_name("per_window_position", statistic="median", short=True, n=10)
    assert med == "marginal-median vector per window position (10 vectors)" and "SGM" not in med
    _expect(agg.AggregationError, lambda: agg.display_name("sgm_of_sgms"), "unknown construction")
    _expect(agg.AggregationError, lambda: agg.display_name("per_cell", statistic="mean"), "unknown statistic")
    # window-set phrases from the declared geometry (four positions here)
    assert agg.window_set_phrase("all", (0, 1, 2, 3), GEOMETRY, window_seconds=2.0) == "all four 2 s windows"
    assert agg.window_set_phrase("opening", (0,), GEOMETRY, window_seconds=2.0) == "opening 0–2 s window only"
    assert agg.window_set_phrase("late", (2, 3), GEOMETRY, window_seconds=2.0) == "named set 'late' = positions [2, 3] (window starts 4, 6 s)"
    # scope lines: pilot versus complete, the selected-window count of a subset
    cov = {"cells": [3, 7], "n_windows": 8}
    line = agg.scope_line("MET-FAB", cov, 3, 12, set_name="opening", chunks=(0,), geometry=GEOMETRY, n_draws=50, window_seconds=2.0)
    assert line == ("MET-FAB · aggregation pilot: 2/3 cells (3, 7), 8/12 windows · windows included: opening 0–2 s window only "
                    "(2 selected windows of the 8) · 50 joint draws/window · representative: SGM"), line
    full = agg.scope_line("MET-FAB", {"cells": [3, 7, 11], "n_windows": 12}, 3, 12, set_name="all", chunks=(0, 1, 2, 3), geometry=GEOMETRY,
                          n_draws=10000, window_seconds=2.0)
    assert full == "MET-FAB · 3/3 cells, 12/12 windows · windows included: all four 2 s windows · 10,000 joint draws/window · representative: SGM", full
    assert "pilot" not in full
    # the report: complete product clouds -> complete cohort; a two-cell collection -> pilot subset; named set in the subtitle
    eval_cfg = PARAMETERS.inference.evaluation
    rng = np.random.default_rng(11)
    a = _product_with_clouds(rng, IDS)
    m = S.decode_manifest(a[S.MANIFEST_KEY])
    xargs = SimpleNamespace(aggregation="pooled", seed=None, condition="FAB")
    ref = {"name": "REFX", "log10": np.array([-0.5, 0.4, 0.5]), "physical": np.array([10 ** -0.5, 0.4, 0.5]), "identity": {"artifact": "x"}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rep = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "full", run_label="T", timestamp="now")
        req = agg.AggregationRequest(window_sets=agg.parse_window_sets("all,opening,late=2+3"), reference=ref, window_seconds=2.0, product_label="P", target=TARGET)
        write_experiment_outputs(rep, xargs, eval_cfg, TABLE, tmp / "full" / "p.npz", a, run_start=0.0, aggregation=req)
        text = (tmp / "full" / "report.md").read_text()
        assert "## Data scope" in text and "| retained-cloud aggregation | MET-FAB | 3 of 3 | 12 of 12 |" in text and "complete cohort" in text
        assert "Source: the original Experiment summaries (MET-FAB: 3 cells, 12 windows), not the retained clouds" in text
        assert "MET-FAB: complete cohort, 3 of 3 cells of this product" in text and "pilot" not in text.split("## Experimental aggregation")[1].split("**Window sets.**")[0]
        assert "**Two independent choices.**" in text and "`late` = named set 'late' = positions [2, 3] (window starts 4, 6 s)" in text
        assert "| `per_cell` | One SGM vector per cell | all selected window clouds from one cell |" in text
        assert "SGM per cell (`per_cell`)" in text and "Fixed reference: REFX" in text
        assert "Exact equivalences under this window selection: SGM for included dataset = SGM per window position; SGM per cell = SGM per cell–window" in text
        assert "windows included: opening 0–2 s window only (3 selected windows of the 12)" in text
        assert "named set 'late' = positions [2, 3]" in text and (tmp / "full" / "figures" / "aggregation_FAB_late_temporal.png").exists()
        assert "windows included: all four 2 s windows versus opening 0–2 s window only versus named set 'late'" in text
        assert "Reading example: the 'SGM per cell' curve of a rises in 3 equal steps" in text
        assert "Examples (descriptive, not a rule): cell " in text
        # the data-scope table counts the actual cells of a two-cell collection: a pilot, with its identities
        sub_ids = [i for i in IDS if i[1] in (3, 7)]
        b = _product_with_clouds(rng, sub_ids)
        p = _collection_file(tmp / "two.npz", b, S.decode_manifest(b[S.MANIFEST_KEY]), label="TWO")
        clouds = agg.combine_clouds([agg.cloud_from_collection(p)], reference_contract={k: m[k] for k in agg.CONTRACT_FIELDS})
        rep2 = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "pilot", run_label="T", timestamp="now")
        req2 = agg.AggregationRequest(window_sets=agg.parse_window_sets("all,opening"), clouds=clouds, reference=ref, window_seconds=2.0,
                                      product_label="P", target=TARGET)
        write_experiment_outputs(rep2, xargs, eval_cfg, TABLE, tmp / "pilot" / "p.npz", a, run_start=0.0, persist_arrays=False, aggregation=req2)
        text2 = (tmp / "pilot" / "report.md").read_text()
        assert "| retained-cloud aggregation | MET-FAB | 2 of 3 | 8 of 12 |" in text2 and "pilot subset: cells [3, 7]" in text2
        assert "MET-FAB · aggregation pilot: 2/3 cells (3, 7), 8/12 windows · windows included: all four 2 s windows" in text2
        assert "| original Experiment summaries (the tables and diagnostic figures) | MET-FAB | 3 | 12 |" in text2
        # the aggregation disabled: the scope table still states the original scope and the reason 'aggregation disabled'
        # (the product's stored clouds do not change that reason); enabled without any clouds: 'no retained clouds'
        rep3 = DiagnosticReporter(stage="Experiment", enabled=True, dump=False, run_label="T", timestamp="now")
        write_experiment_outputs(rep3, xargs, eval_cfg, TABLE, tmp / "p3.npz", a, run_start=0.0, persist_arrays=False,
                                 aggregation=agg.AggregationRequest(window_sets={}, enabled=False))
        headers, rows, note, tag = agg.scope_table(a, m, agg.AggregationRequest(window_sets={}, enabled=False), lambda k: f"MET-{k}")
        assert rows[1][4] == "not computed (aggregation disabled)" and rows[0][2] == "3" and "MET-FAB: 3 cells, 12 windows" in tag
        assert "the draws themselves are stored" in rows[0][4]
        b0 = fx.product("experiment", IDS, rng)
        _, rows0, _, _ = agg.scope_table(b0, S.decode_manifest(b0[S.MANIFEST_KEY]), agg.AggregationRequest(window_sets={"all": None}), lambda k: f"MET-{k}")
        assert rows0[1][4] == "not computed (no retained clouds)" and "were not retained" in rows0[0][4]
        _, rows_off, _, _ = agg.scope_table(b0, S.decode_manifest(b0[S.MANIFEST_KEY]), None, lambda k: f"MET-{k}")
        assert rows_off[1][4] == "not computed (aggregation disabled)"
    # a large source set (one collection per cell) is counted inline, never listed file by file
    few = [{"path": f"/x/c{i}.npz", "label": f"c{i}", "collection_label": f"L{i}", "sha256": "ab" * 32, "seed_policy": {"base_seed": i},
            "written_at": f"2026-11-05 16:{i:02d}:00 UTC"} for i in range(3)]
    many = [dict(f, path=f"/x/c{i}.npz", collection_label=f"COHORTCELL{i}", seed_policy={"base_seed": i}, written_at=f"2026-11-05 16:{i:02d}:00 UTC")
            for i, f in enumerate([few[0]] * 60)]
    assert agg.sources_description(few).count("sha256") == 3 and "c2.npz" in agg.sources_description(few)
    big = agg.sources_description(many)
    assert big.startswith("60 retained draw collections") and "first `c0.npz`, last `c59.npz`" in big and big.count("sha256") == 1 and len(big) < 400
    summ = agg.disclosed_differences_summary(many)
    assert "60 distinct collection label values" in summ and "60 distinct seed policy values" in summ
    assert "written between 2026-11-05 16:00:00 UTC and 2026-11-05 16:59:00 UTC" in summ
    # the shared figures accept the composed names and a median-statistic series (the Nuisance DLI caller's use)
    from srm_and_sbi_monomer_dimer_alp.visualization_aggregation import figure_joint_corner, figure_marginal_cdfs
    vec = np.array([[-0.5, 0.4, 0.5], [-0.6, 0.5, 0.4], [-0.4, 0.3, 0.6]])
    series = [{"name": agg.display_name("per_cell", statistic="median", short=True, n=3), "role": "per_cell", "vectors": vec},
              {"name": "selected mixture at its recorded probabilities (3 vectors)", "role": "mixture", "vectors": vec, "weights": np.array([0.5, 0.25, 0.25])},
              {"name": agg.display_name("reference", short=True, reference_name="REFX"), "role": "reference", "vectors": vec[:1]}]
    fig = figure_marginal_cdfs(series, TABLE, title="t", subtitle="scope")
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "marginal-median vector per cell (3 vectors)" in labels and "Fixed reference: REFX" in labels and "training prior bounds" in labels
    assert fig.texts and any(t.get_text() == "scope" for t in fig.texts)
    fig2 = figure_joint_corner(series, TABLE, title="t", subtitle="scope")
    assert any("REFX" in t.get_text() for t in fig2.legends[0].get_texts())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as exc:                      # noqa: BLE001 -- report every failure
            import traceback
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
