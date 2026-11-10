"""Posterior compatibility of the SGM representatives (the shared distance-rank primitive, the median scores, the
reference scopes, zeros against missing scores, the format-2 candidate archive) and the ``posterior_rank_median``
member weighting of the Nuisance DLI constructor (the hierarchical normalization, refusals, mixtures, sampling, the
equal-member baseline, the report). Synthetic clouds only.

Run directly from the repository root::

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/test_experiment_compatibility.py
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
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS                # noqa: E402
import test_experiment_aggregation as T                                              # noqa: E402  (fixtures: TABLE, KEYS, IDS, GEOMETRY, TARGET)

TABLE, KEYS, IDS, GEOMETRY, TARGET = T.TABLE, T.KEYS, T.IDS, T.GEOMETRY, T.TARGET
PLO, PHI = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])
_expect = T._expect
WC = agg.SCOPE_WITHIN_CELL


def _compat(res, coll, **kw):
    """The hierarchical ``within_cell`` comparison scope the tests of this file pin (per_cell across cells, per_cell_window
    within its own cell; still computable by name). The construction-wide scope Experiment writes has its own test file."""
    return agg.compute_compatibility(res, coll, scope=WC, **kw)


def _as_legacy(m, version=3):
    """Rewrite a within_cell format-4 manifest into the legacy format ``version`` (2 or 3) it is equivalent to: the
    sgm_distance_rank_v1 protocol, which carries no scope field anywhere."""
    m["archive_format_version"] = version
    sec = m["compatibility"]
    sec.pop("comparison_scope", None); sec.pop("comparison_scopes", None)
    sections = [sec] + [d for lab, d in (sec.get("rank_spaces") or {}).items() if "protocol" in d]
    for d in sections:
        d["protocol"]["name"] = agg.LEGACY_COMPATIBILITY_PROTOCOL
        d["protocol"].pop("comparison_scope", None); d["protocol"].pop("comparison_scopes", None)
        for b in d.get("blocks", []):
            b.pop("scope", None)
    if version == 2:
        sec.pop("rank_spaces", None)
    return m


def _collection(cloud, ids, *, kinds=("FAB",), geometry=GEOMETRY, scale=(1.0, 1.0, 1.0), keys=KEYS):
    """A CloudCollection straight from arrays (float32 clouds, exact medoid SGMs), for hand-built fixtures."""
    from srm_and_sbi_monomer_dimer_alp.evaluation import sample_geometric_median
    cloud = np.asarray(cloud, np.float32)
    sgm = np.asarray([sample_geometric_median(cloud[i].astype(float), scale=np.asarray(scale, float))[0] for i in range(cloud.shape[0])])
    contract = {"parameter_keys": list(keys), "coordinate_transform": "log10", "sgm_scale": list(scale), "pool_mode": "unrestricted",
                "n_summary_draws": int(cloud.shape[1]), "checkpoint_sha256": "c" * 64, "window_geometry": dict(geometry), "input_conversion": "test"}
    return agg.CloudCollection(cloud=cloud, kind_index=np.asarray([i[0] for i in ids], np.int64), cell=np.asarray([i[1] for i in ids], np.int64),
                               chunk=np.asarray([i[2] for i in ids], np.int64), kinds=list(kinds), contract=contract,
                               sources=[{"path": None, "sha256": None, "label": "fixture", "kind": "test", "collection_label": None}],
                               source_index=np.zeros(len(ids), np.int64), window_sgm=sgm,
                               window_sgm_index=agg._locate_window_sgms(cloud, sgm))


def _archive(tmp, coll, window_sets, *, kinds=("FAB",), table=TABLE, target=TARGET, product_label="P"):
    res = agg.aggregate(coll, table, window_sets=window_sets)
    compat = _compat(res, coll)
    man = agg.export_manifest(res, product_label=product_label, condition_labels=list(kinds), target=target, compatibility=compat)
    paths = agg.write_aggregation_exports(res, tmp, table, manifest=man, compatibility=compat)
    return agg.load_candidate_archive(paths["archive"]), res, compat, paths


def _comp(grouping, weight, window_set="all", statistic="sgm", member_weighting=None, rank_scope="auto"):
    """``rank_scope`` "auto" names the scope a within_cell archive computed the grouping's scores under (the rank policy
    requires an explicit scope); None omits the field."""
    c = {"source_experiment": "P", "condition": "FAB", "window_set": window_set, "grouping": grouping, "statistic": statistic, "weight": weight}
    if member_weighting:
        c["member_weighting"] = member_weighting
    if rank_scope == "auto":
        rank_scope = None if member_weighting != "posterior_rank_median" else ("within_cell" if grouping == "per_cell_window" else "construction")
    if rank_scope is not None:
        c["rank_scope"] = rank_scope
    return c


def _spec(arc, *comps, overrides=None):
    s = {"block": {"posterior_sample_pool_choice": "empirical_representatives", "candidate_archive": arc["path"],
                   "candidate_archive_sha256": arc["sha256"], "rationale": "a test selection"}, "component": list(comps)}
    if overrides:
        s["override"] = overrides
    return s


def test_rank_primitive_counts_strictly_farther_draws_on_hand_countable_clouds():
    # one-dimensional reference: draws at radii 0, 1, 2, 3, 4 from the SGM at 0, unit scale
    cloud = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]); sgm = np.array([0.0])
    r = agg.reference_distance_ranks(cloud, sgm, np.array([[2.5], [2.0], [0.0], [10.0], [-3.5]]), [1.0])
    assert r.dtype == np.float64 and r.tolist() == [2 / 5, 2 / 5, 4 / 5, 0.0, 1 / 5]      # 2.0 ties the draw at 2 (strict: not farther)
    # the supplied scale divides every coordinate: a (3, 4) offset with scale (3, 4) has unit distance
    cloud2 = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 8.0], [6.0, 8.0]]); sgm2 = np.zeros(2)
    r2 = agg.reference_distance_ranks(cloud2, sgm2, np.array([[3.0, 4.0], [0.0, 0.0]]), [3.0, 4.0])
    assert r2.tolist() == [2 / 4, 3 / 4]                       # radii 0, 1, 2, sqrt(8): farther than sqrt(2) are 2 and sqrt(8)
    # dimension-generic, mixed coordinates: agrees exactly with the draws companion's direct count
    rng = np.random.default_rng(0)
    for D in (3, 6):
        z = rng.normal(size=(5000, D)).astype(np.float32).astype(np.float64); v = z[17]; scale = rng.uniform(0.3, 1.5, size=D)
        cands = rng.normal(size=(9, D))
        zs = (z - v) / scale; d_draws = np.sqrt((zs ** 2).sum(axis=1))
        direct = [float(np.mean(d_draws > float(np.sqrt((((c - v) / scale) ** 2).sum())))) for c in cands]
        got = agg.reference_distance_ranks(z, v, cands, scale)
        assert got.tolist() == direct, (D, got, direct)
        assert np.all((got >= 0) & (got <= 1))
    _expect(agg.AggregationError, lambda: agg.reference_distance_ranks(np.zeros((0, 2)), np.zeros(2), np.zeros((1, 2)), [1, 1]), "non-empty")
    _expect(agg.AggregationError, lambda: agg.reference_distance_ranks(np.zeros((3, 2)), np.zeros(2), np.zeros((1, 2)), [1, 0]), "positive scale")


def _two_cells_identical_plus_one(rng):
    """Cells 3 and 7 carry IDENTICAL clouds (same draws) in every window; cell 11 is distinct. Four windows."""
    base = rng.normal(scale=0.01, size=(4, 40, 3)) + np.array([-0.5, 0.4, 0.5])
    far = rng.normal(scale=0.01, size=(4, 40, 3)) + np.array([-0.8, 0.9, 0.1])
    ids = [(0, c, w) for c in (3, 7, 11) for w in range(4)]
    cloud = np.concatenate([base, base, far], axis=0)
    return _collection(cloud, ids), ids


def test_self_exclusion_is_by_identity_and_identical_peers_stay_in_the_matrix():
    rng = np.random.default_rng(1)
    coll, ids = _two_cells_identical_plus_one(rng)
    res = agg.aggregate(coll, TABLE, window_sets={"all": None})
    compat = _compat(res, coll)
    blk = [b for b in compat.blocks if b.construction == "per_cell"][0]
    cells = [res.groups[int(i)].cell for i in blk.group_index]
    a, b = cells.index(3), cells.index(7)
    assert np.array_equal(res.groups[int(blk.group_index[a])].sgm_log10, res.groups[int(blk.group_index[b])].sgm_log10)
    assert np.isnan(blk.ranks[a, a]) and np.isnan(blk.ranks[b, b])                # only the self reference is masked
    assert np.isfinite(blk.ranks[a, b]) and blk.ranks[a, b] == blk.ranks[b, a] and blk.ranks[a, b] > 0.9   # the identical peer scores, high
    c = cells.index(11)
    assert blk.ranks[c, a] == 0.0 and blk.ranks[a, c] == 0.0                      # the distant cell is beyond every draw
    # scores: the identical pair has one high and one zero reference -> median = average of the two; the distant cell zero
    for i in (a, b):
        row = np.delete(blk.ranks[i], i)
        assert compat.score[blk.group_index[i]] == float(np.median(row)) and compat.n_references[blk.group_index[i]] == 2
    assert compat.score[blk.group_index[c]] == 0.0 and compat.status[blk.group_index[c]] == "computed"


def _ten_windows_two_distant(rng, n=60):
    """One cell, ten windows: eight tight central windows, two distant windows that support each other."""
    ids = [(0, 5, w) for w in range(10)]
    cloud = np.empty((10, n, 3))
    for w in range(10):
        center = np.array([-0.5, 0.4, 0.5]) if w < 8 else np.array([-0.95, 0.95, 0.05])
        cloud[w] = center + rng.normal(scale=0.01, size=(n, 3))
    cloud[9] = cloud[8] + rng.normal(scale=1e-4, size=(n, 3))       # near-identical posteriors, as two blank-input windows give
    geom = S.window_geometry(n_frames=100, step_frames=100, span_frames=1000)
    return _collection(cloud, ids, geometry=geom), ids, geom


def test_two_mutually_supporting_distant_windows_get_zero_median_but_nonzero_mean_and_zero_probability():
    rng = np.random.default_rng(2)
    coll, ids, geom = _ten_windows_two_distant(rng)
    res = agg.aggregate(coll, TABLE, window_sets={"all": None})
    compat = _compat(res, coll)
    blk = [b for b in compat.blocks if b.construction == "per_cell_window"][0]
    chunks = [res.groups[int(i)].chunk for i in blk.group_index]
    assert chunks == list(range(10))
    for w in (8, 9):
        row = np.delete(blk.ranks[w], w)
        assert np.mean(row) > 0.05 and np.median(row) == 0.0, (w, row)          # the mean keeps them; the median does not
        assert blk.ranks[8, 9] > 0.9 and blk.ranks[9, 8] > 0.9                    # they support each other
        assert compat.score[blk.group_index[w]] == 0.0 and compat.status[blk.group_index[w]] == "computed"
    for w in range(8):
        assert compat.score[blk.group_index[w]] > 0.0
    # the constructor: a valid zero becomes exactly zero probability; the central windows share the cell's mass
    with tempfile.TemporaryDirectory() as tmp:
        arc, res2, compat2, paths = _archive(Path(tmp), coll, {"all": None})
        spec = _spec(arc, _comp("per_cell_window", 1.0, member_weighting="posterior_rank_median"))
        vec, pr, rec = ndli.representatives_from_spec(spec, arc, TABLE, PLO, PHI, target=TARGET)
        assert vec.shape == (10, 3) and np.isclose(pr.sum(), 1.0) and int(np.sum(pr == 0)) == 2
        comp = rec["components"][0]
        assert comp["member_weighting"] == "posterior_rank_median" and comp["n_zero_members"] == 2 and comp["n_positive_members"] == 8
        zero_members = [m for m in comp["members"] if m["probability"] == 0]
        assert sorted(m["chunk"] for m in zero_members) == [8, 9]                 # identities retained, probability zero
        assert rec["n_zero_probability_rows"] == 2 and rec["equal_member_baseline"]["n_zero_probability_rows"] == 0
        assert np.allclose(rec["equal_member_baseline"]["probabilities"], 0.1)
        # the artifact never samples a zero row
        nu = ndli.build_nuisance_dli(spec, KEYS, PLO, PHI, table=TABLE, archive=arc, target=TARGET)
        draws = nu.sample(20000, seed=3)
        zero_rows = nu.samples[nu.probabilities == 0]
        assert not any(np.array_equal(d, z) for z in zero_rows for d in draws[:5000])
        freq = np.array([np.mean(np.all(draws == r, axis=1)) for r in nu.samples])
        assert np.allclose(freq, nu.probabilities, atol=0.012)
        art = Path(tmp) / "A.npz"; nu.flush(art); back = ndli.NuisanceDLI.load(art)
        assert np.array_equal(back.probabilities, nu.probabilities) and back.selection["components"][0]["member_weighting"] == "posterior_rank_median"
        assert np.isclose(back.selection["equal_member_baseline"]["effective_rows"], 10.0)


def test_missing_scores_are_not_zeros_and_the_constructor_refuses_them():
    rng = np.random.default_rng(3)
    # one cell, one selected window: both scored constructions have no peer
    ids = [(0, 4, 0)]
    coll = _collection(np.array([[-0.5, 0.4, 0.5]]) + rng.normal(scale=0.01, size=(1, 30, 3)), ids,
                       geometry=S.window_geometry(n_frames=100, step_frames=100, span_frames=100))
    res = agg.aggregate(coll, TABLE, window_sets={"all": None, "opening": (0,)})
    compat = _compat(res, coll)
    assert not compat.blocks
    for g, st, sc, nr in zip(res.groups, compat.status, compat.score, compat.n_references):
        assert np.isnan(sc) and nr == 0
        assert st == ("insufficient_peers" if g.construction in agg.rank_scored_constructions(WC) else "not_applicable"), (g.construction, st)
    with tempfile.TemporaryDirectory() as tmp:
        arc, _, _, _ = _archive(Path(tmp), coll, {"all": None, "opening": (0,)})
        avail = ndli.component_weighting_availability(arc)
        assert avail[("FAB", "all", "per_cell")].startswith("rank space all (a, b, c): unavailable") and "insufficient_peers" in avail[("FAB", "all", "per_cell")]
        assert avail[("FAB", "all", "global")].startswith("not applicable")
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, _comp("per_cell", 1.0, member_weighting="posterior_rank_median")),
                                                                   arc, TABLE, PLO, PHI, target=TARGET), "no computed compatibility score", "insufficient_peers")
        # uniform still works on the same archive
        vec, pr, rec = ndli.representatives_from_spec(_spec(arc, _comp("per_cell", 1.0)), arc, TABLE, PLO, PHI, target=TARGET)
        assert pr.tolist() == [1.0] and rec["components"][0]["member_weighting"] == "uniform"
    # an all-zero computed block: three cells pairwise far apart -> every score zero -> refused, never uniform
    ids = [(0, c, w) for c in (3, 7, 11) for w in range(2)]
    centers = {3: [-0.9, 0.1, 0.1], 7: [-0.5, 0.5, 0.5], 11: [-0.1, 0.9, 0.9]}
    cloud = np.stack([np.array(centers[c]) + rng.normal(scale=0.005, size=(30, 3)) for (_, c, _) in ids])
    coll = _collection(cloud, ids, geometry=S.window_geometry(n_frames=100, step_frames=100, span_frames=200))
    with tempfile.TemporaryDirectory() as tmp:
        arc, res, compat, _ = _archive(Path(tmp), coll, {"all": None})
        pc = [i for i, g in enumerate(res.groups) if g.construction == "per_cell"]
        assert all(compat.status[i] == "computed" and compat.score[i] == 0.0 for i in pc)
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, _comp("per_cell", 1.0, member_weighting="posterior_rank_median")),
                                                                   arc, TABLE, PLO, PHI, target=TARGET), "every score of this component is zero")
        # the advertised availability applies the same normalization rule: computed, but not normalizable
        text = ndli.component_weighting_availability(arc)[("FAB", "all", "per_cell")]
        assert text.startswith("rank space all (a, b, c): unavailable") and "computed for 3/3" in text and "cannot be normalized" in text and "zero" in text, text
        # within-cell level: one cell whose two selected windows score zero against each other (far apart), the other cells fine
        ids = [(0, c, w) for c in (3, 7, 11) for w in range(2)]
        cloud = np.stack([np.array([-0.5, 0.5, 0.5]) + (np.array([0.3, -0.4, 0.4]) if (c == 11 and w == 1) else 0.0) + rng.normal(scale=0.005, size=(30, 3))
                          for (_, c, w) in ids])
        coll = _collection(cloud, ids, geometry=S.window_geometry(n_frames=100, step_frames=100, span_frames=200))
        arc2, res2, compat2, _ = _archive(Path(tmp) / "w", coll, {"all": None})
        text = ndli.component_weighting_availability(arc2)[("FAB", "all", "per_cell_window")]
        assert text.startswith("rank space all (a, b, c): unavailable") and "computed for 6/6" in text and "cell 11" in text, text
        assert ndli.component_weighting_availability(arc2)[("FAB", "all", "per_cell")].startswith("rank space all (a, b, c): available")
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc2, _comp("per_cell_window", 1.0, member_weighting="posterior_rank_median")),
                                                                   arc2, TABLE, PLO, PHI, target=TARGET), "cell 11: every selected-window score is zero")


def test_reference_scopes_other_windows_of_the_same_cell_other_cells_of_the_same_condition_and_set():
    rng = np.random.default_rng(4)
    a = T._product_with_clouds(rng, IDS)
    coll = agg.cloud_from_product(a, S.decode_manifest(a[S.MANIFEST_KEY]))
    res = agg.aggregate(coll, TABLE, window_sets={"all": None, "late": (2, 3)})
    compat = _compat(res, coll)
    # a named set scores only within its declared positions: window 2 against window 3 only
    for g, st, nr in zip(res.groups, compat.status, compat.n_references):
        if g.construction == "per_cell_window":
            assert st == "computed" and nr == ({"all": 3, "late": 1}[g.window_set]), (g.window_set, nr)
        if g.construction == "per_cell":
            assert st == "computed" and nr == 2
    late = [b for b in compat.blocks if b.window_set == "late" and b.construction == "per_cell_window"]
    assert len(late) == 3 and all(sorted(res.groups[int(i)].chunk for i in b.group_index) == [2, 3] for b in late)
    # direct recomputation of one entry from the clouds: window 3 of cell 7 as the reference for window 2 of cell 7
    blk = [b for b in late if b.cell == 7][0]
    order = [res.groups[int(i)].chunk for i in blk.group_index]
    r2 = coll.cloud[[k for k, (ki, c, w) in enumerate(IDS) if (c, w) == (7, 3)][0]]
    v3 = res.groups[int(blk.group_index[order.index(3)])].sgm_log10; v2 = res.groups[int(blk.group_index[order.index(2)])].sgm_log10
    assert blk.ranks[order.index(2), order.index(3)] == agg.reference_distance_ranks(r2, v3, v2[None, :], res.scale)[0]
    # two conditions never contaminate each other: the FAB scores equal those of a FAB-only run
    inlb_ids = [(1, 5, w) for w in range(4)] + [(1, 9, w) for w in range(4)]
    coll2 = _collection(np.concatenate([coll.cloud, T._cloud(rng, inlb_ids)]), IDS + inlb_ids, kinds=("FAB", "INLB"))
    res2 = agg.aggregate(coll2, TABLE, window_sets={"all": None})
    compat2 = _compat(res2, coll2)
    res1 = agg.aggregate(coll, TABLE, window_sets={"all": None}); compat1 = _compat(res1, coll)
    key = lambda g: (g.construction, g.window_set, g.kind, g.cell, g.chunk)   # noqa: E731
    s1 = {key(g): float(compat1.score[i]) for i, g in enumerate(res1.groups) if g.kind == "FAB"}
    s2 = {key(g): float(compat2.score[i]) for i, g in enumerate(res2.groups) if g.kind == "FAB"}
    assert s1.keys() == s2.keys()
    for k in s1:
        assert (np.isnan(s1[k]) and np.isnan(s2[k])) or s1[k] == s2[k], k
    assert all(b.kind in ("FAB", "INLB") and len(set(res2.groups[int(i)].kind for i in b.group_index)) == 1 for b in compat2.blocks)
    inlb_pc = [b for b in compat2.blocks if b.kind == "INLB" and b.construction == "per_cell"]
    assert len(inlb_pc) == 1 and inlb_pc[0].group_index.size == 2


def test_opening_aliases_keep_their_own_availability_and_pooled_arrays_are_unchanged():
    rng = np.random.default_rng(5)
    a = T._product_with_clouds(rng, IDS)
    coll = agg.cloud_from_product(a, S.decode_manifest(a[S.MANIFEST_KEY]))
    res = agg.aggregate(coll, TABLE, window_sets={"all": None, "opening": (0,)})
    compat = _compat(res, coll)
    pc = {g.cell: i for i, g in enumerate(res.groups) if g.construction == "per_cell" and g.window_set == "opening"}
    pcw = {g.cell: i for i, g in enumerate(res.groups) if g.construction == "per_cell_window" and g.window_set == "opening"}
    for c in pc:
        assert np.array_equal(res.groups[pc[c]].sgm_log10, res.groups[pcw[c]].sgm_log10) and res.groups[pc[c]].equivalent_to == "per_cell_window[opening]"
        assert compat.status[pc[c]] == "computed" and compat.n_references[pc[c]] == 2
        assert compat.status[pcw[c]] == "insufficient_peers" and np.isnan(compat.score[pcw[c]]) and compat.n_references[pcw[c]] == 0
    # the opening per-cell references are the other cells' opening clouds (10 draws here... SDRAWS), not their pooled ones
    blk = [b for b in compat.blocks if b.window_set == "opening" and b.construction == "per_cell"][0]
    assert blk.reference_draws.tolist() == [T.SDRAWS] * 3
    blk_all = [b for b in compat.blocks if b.window_set == "all" and b.construction == "per_cell"][0]
    assert blk_all.reference_draws.tolist() == [4 * T.SDRAWS] * 3
    # pool invariance: the aggregation's groups are untouched by the compatibility step, and the archive keeps v1's arrays
    before = [(g.sgm_log10.copy(), g.median_log10.copy(), g.quantiles_log10.copy(), dict(g.sgm_source), list(g.members)) for g in res.groups]
    _compat(res, coll)
    for g, (s, m, q, src, mem) in zip(res.groups, before):
        assert np.array_equal(g.sgm_log10, s) and np.array_equal(g.median_log10, m) and np.array_equal(g.quantiles_log10, q)
        assert g.sgm_source == src and g.members == mem
    with tempfile.TemporaryDirectory() as tmp:
        arc, _, _, paths = _archive(Path(tmp), coll, {"all": None, "opening": (0,)})
        A = arc["arrays"]
        assert np.array_equal(A["sgm_log10"], np.asarray([g.sgm_log10 for g in res.groups]))
        assert [str(c) for c in A["construction"]] == [g.construction for g in res.groups]
        assert arc["format_version"] == agg.ARCHIVE_FORMAT_VERSION == 4 and arc["manifest"]["archive_format_version"] == 4
        text = paths["groups_csv"].read_text().splitlines()
        assert text[0].endswith("compatibility_score,compatibility_status,compatibility_n_references,sgm_outside_prior_keys")
        comp_csv = paths["compatibility_csv"].read_text().splitlines()
        assert comp_csv[0].startswith("kind,window_set,construction,cell_context,candidate_group") and len(comp_csv) - 1 == sum(
            b.group_index.size * (b.group_index.size - 1) for b in _compat(res, coll).blocks)


def test_archive_format_two_round_trip_legacy_format_one_and_malformed_data_are_handled():
    rng = np.random.default_rng(6)
    a = T._product_with_clouds(rng, IDS)
    coll = agg.cloud_from_product(a, S.decode_manifest(a[S.MANIFEST_KEY]))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, res, compat, paths = _archive(tmp, coll, {"all": None, "opening": (0,)})
        # round trip: blocks, scores and statuses come back identical and in order
        back = arc["compatibility"]
        assert len(back["blocks"]) == len(compat.blocks)
        for b1, b2 in zip(compat.blocks, back["blocks"]):
            assert np.array_equal(b1.group_index, b2["group_index"]) and np.array_equal(b1.ranks, b2["ranks"], equal_nan=True)
            assert (b1.kind, b1.window_set, b1.construction, b1.cell) == (b2["kind"], b2["window_set"], b2["construction"], b2["cell"])
        for i in range(len(res.groups)):
            g = back["by_group"][i]
            assert g["status"] == compat.status[i] and g["n_references"] == compat.n_references[i]
            assert (g["score"] is None and np.isnan(compat.score[i])) or g["score"] == compat.score[i]
        comp = agg.archive_component(arc, kind="FAB", window_set="all", construction="per_cell")
        assert comp["compatibility_available"] and all(d["compatibility"]["status"] == "computed" for d in comp["identities"])
        # a legacy format-1 archive (no compatibility data): accepted for a uniform selection, refused for rank weighting
        with np.load(paths["archive"], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if not k.startswith("compatibility_") and k != "sgm_outside_prior_keys"}
        man = json.loads(str(arrays.pop("manifest_json"))); man.pop("compatibility"); man["archive_format_version"] = 1
        np.savez_compressed(tmp / "v1.npz", **arrays, manifest_json=np.asarray(json.dumps(man, sort_keys=True, default=str)))
        v1 = agg.load_candidate_archive(tmp / "v1.npz")
        assert v1["format_version"] == 1 and v1["compatibility"] is None
        c1 = agg.archive_component(v1, kind="FAB", window_set="all", construction="per_cell")
        assert not c1["compatibility_available"] and all(d["compatibility"] is None for d in c1["identities"])
        v, p, r = ndli.representatives_from_spec(_spec(v1, _comp("per_cell", 1.0)), v1, TABLE, PLO, PHI, target=TARGET)
        assert np.allclose(p, 1 / 3) and r["member_weightings"] == ["uniform"] and "equal_member_baseline" not in r
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(v1, _comp("per_cell", 1.0, member_weighting="posterior_rank_median")),
                                                                   v1, TABLE, PLO, PHI, target=TARGET), "format-2", "Regenerate the Experiment report")
        assert "archive format 1" in ndli.component_weighting_availability(v1)[("FAB", "all", "per_cell")]
        assert ndli.component_weighting_availability(arc)[("FAB", "all", "per_cell")].startswith("rank space all (a, b, c): available") and \
            "normalizable" in ndli.component_weighting_availability(arc)[("FAB", "all", "per_cell")]
        # malformed format-2 data are refused by the reader: a reordered block, a stale score, another protocol, an unknown version
        def rewrite(name, mutate_arrays=None, mutate_manifest=None):
            with np.load(paths["archive"], allow_pickle=False) as z:
                arrs = {k: z[k] for k in z.files}
            m = json.loads(str(arrs.pop("manifest_json")))
            if mutate_arrays:
                mutate_arrays(arrs)
            if mutate_manifest:
                mutate_manifest(m)
            np.savez_compressed(tmp / name, **arrs, manifest_json=np.asarray(json.dumps(m, sort_keys=True, default=str)))
            return tmp / name
        def reorder(arrs):
            arrs["compatibility_block_0_group_index"] = arrs["compatibility_block_0_group_index"][::-1].copy()
        def stale(arrs):
            s = arrs["compatibility_score"].copy(); i = int(np.flatnonzero(np.isfinite(s))[0]); s[i] = s[i] + 0.25 if s[i] < 0.5 else s[i] - 0.25
            arrs["compatibility_score"] = s
        def unmask(arrs):
            R = arrs["compatibility_block_0_ranks"].copy(); R[0, 0] = 0.0; arrs["compatibility_block_0_ranks"] = R
        def status_zero(arrs):
            st = arrs["compatibility_status"].astype(object); i = int(np.flatnonzero(st == "insufficient_peers")[0]); st[i] = "computed"
            arrs["compatibility_status"] = st.astype(str)
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("reorder.npz", reorder)), "disagree with its block")
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("stale.npz", stale)), "stored score")
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("unmask.npz", unmask)), "diagonal must be masked")
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("status.npz", status_zero)), "only group of its context")
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("proto.npz", None, lambda m: m["compatibility"]["protocol"].update(name="other"))),
                "protocol 'other'")
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("v5.npz", None, lambda m: m.update(archive_format_version=5))), "not one of")
        # the constructor validates the policy field itself
        for bad, needle in ((_comp("per_cell", 1.0, member_weighting="mean_rank"), "must be one of"),
                            (_comp("per_cell", 1.0, statistic="median", member_weighting="posterior_rank_median"), 'requires statistic = "sgm"'),
                            (_comp("global", 1.0, member_weighting="posterior_rank_median"), "per_cell"),
                            (_comp("per_window_position", 1.0, member_weighting="posterior_rank_median"), "carries no rank score")):
            _expect(ValueError, lambda b=bad: ndli.representatives_from_spec(_spec(arc, b), arc, TABLE, PLO, PHI, target=TARGET), needle)
        # the TOML loader validates the same field
        p = tmp / "s.toml"
        p.write_text('[block]\nposterior_sample_pool_choice = "empirical_representatives"\ncandidate_archive = "x.npz"\nrationale = "r"\n'
                     '[[component]]\nsource_experiment = "P"\ncondition = "FAB"\nwindow_set = "all"\ngrouping = "per_cell"\nweight = 1.0\n'
                     'member_weighting = "posterior_rank_median"\nrank_scope = "construction"\n')
        ndli.load_spec(p, KEYS, PLO, PHI, table=TABLE)
        p.write_text(p.read_text().replace('"posterior_rank_median"', '"inverse_distance"'))
        _expect(ValueError, lambda: ndli.load_spec(p, KEYS, PLO, PHI, table=TABLE), "must be one of")


def _hierarchy_fixture(rng):
    """Three cells x four windows with DIFFERENT within-cell score totals: cell 3 tight (four coincident windows), cell 7
    three coincident windows and one distant window, cell 11 three mutually shifted windows and one distant window (the
    distant windows score zero; the shifted ones score between zero and one)."""
    ids = [(0, c, w) for c in (3, 7, 11) for w in range(4)]
    cloud = np.empty((12, 60, 3))
    for i, (_, c, w) in enumerate(ids):
        center = np.array([-0.5 - 0.02 * (c % 5), 0.4, 0.5])
        if (c in (7, 11)) and w == 3:
            center = center + np.array([-0.4, 0.5, -0.4])
        elif c == 11:
            center = center + np.array([0.004 * w, 0.0, 0.0])
        cloud[i] = center + rng.normal(scale=0.006, size=(60, 3))
    return _collection(cloud, ids), ids


def test_hierarchical_normalization_gives_every_cell_its_equal_mass_and_per_cell_its_component_mass():
    rng = np.random.default_rng(7)
    coll, ids = _hierarchy_fixture(rng)
    with tempfile.TemporaryDirectory() as tmp:
        arc, res, compat, _ = _archive(Path(tmp), coll, {"all": None, "opening": (0,)})
        # per_cell_window: exactly alpha / C per cell although the cells' score totals differ
        spec = _spec(arc, _comp("per_cell_window", 0.7, member_weighting="posterior_rank_median"), _comp("per_cell", 0.3, member_weighting="posterior_rank_median"))
        vec, pr, rec = ndli.representatives_from_spec(spec, arc, TABLE, PLO, PHI, target=TARGET)
        cw = rec["components"][0]
        sums = cw["normalizers"]["sum_of_scores_by_cell"]
        assert len(set(round(v, 9) for v in sums.values())) == 3, sums                 # three different totals
        by_cell = {}
        for m in cw["members"]:
            by_cell[m["cell"]] = by_cell.get(m["cell"], 0.0) + m["probability"]
        assert all(abs(v - 0.7 / 3) < 1e-12 for v in by_cell.values()), by_cell
        assert cw["n_zero_members"] == 2 and all(m["probability"] == 0 for m in cw["members"] if (m["cell"], m["chunk"]) in ((7, 3), (11, 3)))
        assert all(0 < m["score_used"] < 1 for m in cw["members"] if m["cell"] == 11 and m["chunk"] < 3)
        # per_cell: contributions proportional to the scores, summing to the component mass
        pc = rec["components"][1]
        s = np.array([m["score_used"] for m in pc["members"]]); p = np.array([m["probability"] for m in pc["members"]])
        assert np.allclose(p, 0.3 * s / s.sum()) and abs(p.sum() - 0.3) < 1e-12 and pc["normalizers"]["scope"] == "component"
        assert np.isclose(pr.sum(), 1.0) and rec["n_member_rows"] == 15
        # a global normalization over all windows would NOT give equal cell mass: demonstrate the difference explicitly
        s_all = np.array([m["score_used"] for m in cw["members"]]); c_all = np.array([m["cell"] for m in cw["members"]])
        flat = 0.7 * s_all / s_all.sum()
        flat_by_cell = {c: float(flat[c_all == c].sum()) for c in (3, 7, 11)}
        assert max(flat_by_cell.values()) - min(flat_by_cell.values()) > 0.02
        # effective numbers are stated with their scope
        assert "not a count of independent experiments" in rec["effective_rows_scope"] and cw["effective_members"] < 12
        # an archive checksum change is refused
        _expect(ValueError, lambda: ndli.representatives_from_spec(dict(spec, block=dict(spec["block"], candidate_archive_sha256="0" * 64)),
                                                                   arc, TABLE, PLO, PHI, target=TARGET), "checksum")


def test_mixtures_duplicates_overrides_baseline_and_the_artifact_record():
    rng = np.random.default_rng(8)
    coll, ids = _hierarchy_fixture(rng)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, res, compat, _ = _archive(tmp, coll, {"all": None, "opening": (0,)})
        # per_cell[opening] (rank-weighted, scores across cells) + per_cell_window[opening] (uniform: its scores are unavailable):
        # the same vectors -> consolidated, contributions added, a zero contribution never transferred
        spec = _spec(arc, _comp("per_cell", 0.5, "opening", member_weighting="posterior_rank_median"), _comp("per_cell_window", 0.5, "opening"))
        vec, pr, rec = ndli.representatives_from_spec(spec, arc, TABLE, PLO, PHI, target=TARGET)
        assert vec.shape == (3, 3) and len(rec["consolidated"]) == 3 and rec["n_member_rows"] == 6
        rank_p = {m["cell"]: m["probability"] for m in rec["components"][0]["members"]}
        for cns in rec["consolidated"]:
            contribs = {m["component"]: m["contribution"] for m in cns["memberships"]}
            assert abs(contribs[1] - 0.5 / 3) < 1e-12 and abs(cns["probability"] - (contribs[0] + contribs[1])) < 1e-12
        assert np.isclose(sum(rank_p.values()), 0.5)
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, _comp("per_cell_window", 1.0, "opening", member_weighting="posterior_rank_median")),
                                                                   arc, TABLE, PLO, PHI, target=TARGET), "insufficient_peers")
        # an override transforms the rows; the record says the scores describe the original SGMs; the baseline aligns
        spec2 = _spec(arc, _comp("per_cell", 1.0, member_weighting="posterior_rank_median"), overrides={"b": {"value": 0.25, "reason": "declared"}})
        v2, p2, r2 = ndli.representatives_from_spec(spec2, arc, TABLE, PLO, PHI, target=TARGET)
        assert np.all(v2[:, 1] == 0.25) and not r2["is_sampled_rows"] and "original SGM" in r2["components"][0]["score_source"]["note"]
        base = r2["equal_member_baseline"]
        assert np.allclose(base["probabilities"], 1 / 3) and base["effective_rows"] == 3.0
        # the baseline equals an independent uniform build of the same spec (same consolidation and overrides)
        v3, p3, r3 = ndli.representatives_from_spec({**spec2, "component": [_comp("per_cell", 1.0)]}, arc, TABLE, PLO, PHI, target=TARGET)
        assert np.array_equal(v3, v2) and np.allclose(p3, base["probabilities"])
        # physical-domain checks still apply to the final vectors
        _expect(ValueError, lambda: ndli.representatives_from_spec(dict(spec2, override={"a": {"value": -1.0, "reason": "r"}}), arc, TABLE, PLO, PHI, target=TARGET),
                "must be positive")
        # the artifact: format unchanged, record travels, legacy uniform artifacts unchanged
        nu = ndli.build_nuisance_dli(spec2, KEYS, PLO, PHI, table=TABLE, archive=arc, target=TARGET)
        assert nu.manifest()["artifact_format_version"] == ndli.ARTIFACT_FORMAT_VERSION and nu.manifest()["weighted"]
        art = tmp / "R.npz"; nu.flush(art); back = ndli.NuisanceDLI.load(art)
        assert back.selection["member_weightings"] == ["posterior_rank_median"] and np.array_equal(back.probabilities, nu.probabilities)
        uni = ndli.build_nuisance_dli({**spec2, "component": [_comp("per_cell", 1.0)]}, KEYS, PLO, PHI, table=TABLE, archive=arc, target=TARGET)
        assert np.allclose(uni.probabilities, 1 / 3) and "equal_member_baseline" not in uni.selection
        assert uni.selection["components"][0]["member_probability"] == 1 / 3 and uni.selection["components"][0]["member_weighting"] == "uniform"
        d = uni.sample(9000, seed=1); freq = np.array([np.mean(np.all(d == r, axis=1)) for r in uni.samples])
        assert np.allclose(freq, 1 / 3, atol=0.02)
        # a legacy record without the new fields still loads (scalar member_probability only)
        legacy = dict(uni.selection); legacy["components"] = [{k: v for k, v in c.items() if k not in ("member_weighting", "normalizers", "members")}
                                                              for c in legacy["components"]]
        old = ndli.NuisanceDLI.from_representatives(KEYS, uni.samples, uni.probabilities, legacy, prior_low=PLO, prior_high=PHI)
        old.flush(tmp / "L.npz"); assert np.array_equal(ndli.NuisanceDLI.load(tmp / "L.npz").probabilities, uni.probabilities)


def test_the_experiment_report_carries_the_subsection_figures_and_atlas_for_live_and_saved_rendering():
    from srm_and_sbi_monomer_dimer_alp.experiment_runner import write_experiment_outputs
    eval_cfg = PARAMETERS.inference.evaluation
    rng = np.random.default_rng(9)
    ids = [(0, c, w) for c in range(14) for w in range(4)]          # fourteen cells: two atlas pages
    a = T._product_with_clouds(rng, ids)
    m = S.decode_manifest(a[S.MANIFEST_KEY])
    xargs = SimpleNamespace(aggregation="pooled", seed=None, condition="FAB")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rep = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "live", run_label="T", timestamp="now")
        req = agg.AggregationRequest(window_sets=agg.parse_window_sets("all,opening"), window_seconds=2.0, product_label="P", target=TARGET)
        write_experiment_outputs(rep, xargs, eval_cfg, TABLE, tmp / "live" / "p.npz", a, run_start=0.0, aggregation=req)
        text = (tmp / "live" / "report.md").read_text()
        assert "posterior compatibility of SGM representatives" in text and "ordinary median" in text
        assert "| MET-FAB | all | SGM per cell (`per_cell`) | construction | 1 | 14 | 13 | computed |" in text
        # the complete populations: 56 cell–window groups against the other 55, four positions against the other three
        assert "| MET-FAB | all | SGM per cell–window (`per_cell_window`) | construction | 1 | 56 | 55 | computed |" in text
        assert "| MET-FAB | all | SGM per window position (`per_window_position`) | construction | 1 | 4 | 3 | computed |" in text
        # a single-window set: the cell–window population is the 14 opening windows (13 peers); one position has no peer
        assert "| MET-FAB | opening | SGM per cell–window (`per_cell_window`) | construction | 1 | 14 | 13 | computed |" in text
        assert ("| MET-FAB | opening | SGM per window position (`per_window_position`) | construction | 0 | 1 | - | unavailable: a population of 1 "
                "group has no peer (insufficient_peers) |") in text
        assert ("| MET-FAB | all | SGM for included dataset (`global`) | - | 0 | 1 | - | not applicable (no same-level peer; uniform selection "
                "remains available) |") in text
        assert "Comparison scope: `construction`" in text and "p_i = alpha * s_i / (s_1 + ... + s_N)" in text and "No cell receives a reserved mass" in text
        assert "zero scores (centrality outcomes, not defective recordings)" in text and "aggregation_archive_reads_back | PASS" in text
        for f in ("aggregation_FAB_all_compatibility_cells", "aggregation_FAB_opening_compatibility_cells", "aggregation_FAB_all_compatibility_windows",
                  "aggregation_FAB_all_compatibility_positions", "aggregation_FAB_all_compatibility_population"):
            assert (tmp / "live" / "figures" / f"{f}.png").exists() and f"figures/{f}.png" in text, f
        for f in ("windows", "positions", "population"):
            assert not (tmp / "live" / "figures" / f"aggregation_FAB_opening_compatibility_{f}.png").exists()
        assert "Computed without a separate figure: with one selected window (opening 0–2 s window only)" in text
        assert "Not computed: one selected window position (opening 0–2 s window only) has no peer position" in text
        # the population matrix is labeled as ONE 56 x 56 matrix, its same-cell blocks as part of it
        assert "ONE matrix over the complete cell–window population" in text and "56 candidates (rows) against the same 56 groups" in text
        assert "part of this one population and not separate matrices" in text
        assert "Median compatibility rank" not in text.split("## Figures")[0].split("posterior compatibility")[0]   # raw ranks live in the subsection
        atlas = tmp / "live" / "compatibility" / "FAB" / "all" / "windows"
        assert (atlas / "index.md").exists() and (atlas / "page_01.png").exists() and (atlas / "page_02.png").exists() and not (atlas / "page_03.png").exists()
        index = (atlas / "index.md").read_text()
        assert "| 0 | [page_01.png](page_01.png) |" in index and "| 13 | [page_02.png](page_02.png) |" in index
        assert "[same-cell blocks of the cell–window matrix, FAB/all](compatibility/FAB/all/windows/index.md)" in text
        assert "diagonal block of the complete 56 x 56 cell–window matrix" in index and "are NOT the candidate's score" in index
        for link in ("../../../../aggregation_groups.csv", "../../../../aggregation_compatibility.csv", "../../../../report.md"):
            assert (atlas / link).resolve().exists(), link
        assert (tmp / "live" / "aggregation_compatibility.csv").exists() and "[aggregation_compatibility.csv](aggregation_compatibility.csv)" in text
        live = agg.load_candidate_archive(tmp / "live" / "aggregation_candidates.npz")
        assert live["format_version"] == 4 and live["compatibility"]["comparison_scope"] == agg.SCOPE_CONSTRUCTION
        assert [(b["construction"], b["scope"], b["cell"], len(b["group_index"])) for b in live["compatibility"]["blocks"] if b["window_set"] == "all"] == [
            ("per_cell", "construction", -1, 14), ("per_window_position", "construction", -1, 4), ("per_cell_window", "construction", -1, 56)]
        # the saved path (a retained collection through the same writer) computes the same matrices
        p = T._collection_file(tmp / "coll.npz", a, m, label="X")
        clouds = agg.combine_clouds([agg.cloud_from_collection(p)], reference_contract={k: m[k] for k in agg.CONTRACT_FIELDS})
        rep2 = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "saved", run_label="T", timestamp="now")
        req2 = agg.AggregationRequest(window_sets=agg.parse_window_sets("all,opening"), clouds=clouds, window_seconds=2.0, product_label="P", target=TARGET)
        write_experiment_outputs(rep2, xargs, eval_cfg, TABLE, tmp / "saved" / "p.npz", a, run_start=0.0, persist_arrays=False, aggregation=req2)
        saved = agg.load_candidate_archive(tmp / "saved" / "aggregation_candidates.npz")
        assert np.array_equal(live["arrays"]["compatibility_score"], saved["arrays"]["compatibility_score"], equal_nan=True)
        assert all(np.array_equal(b1["ranks"], b2["ranks"], equal_nan=True) for b1, b2 in zip(live["compatibility"]["blocks"], saved["compatibility"]["blocks"]))
        # disabled and cloudless products keep their statements, no compatibility output
        rep3 = DiagnosticReporter(stage="Experiment", enabled=True, dump=True, dump_dir=tmp / "off", run_label="T", timestamp="now")
        write_experiment_outputs(rep3, xargs, eval_cfg, TABLE, tmp / "off" / "p.npz", a, run_start=0.0, aggregation=agg.AggregationRequest(window_sets={}, enabled=False))
        t3 = (tmp / "off" / "report.md").read_text()
        assert "Not computed: aggregation was disabled" in t3 and "posterior compatibility" not in t3 and not (tmp / "off" / "compatibility").exists()


def test_the_reader_requires_the_complete_reference_context_of_every_block():
    """R3 of the 2026-11-06 review: a peer population split into two blocks, reference draw counts that are not the
    referenced groups' pooled draws, a protocol scale that is not the archive's, fractional identities, a missing block,
    an unexpected block and a false insufficient_peers are refused; the valid archive passes unchanged."""
    rng = np.random.default_rng(31)
    ids = [(0, c, w) for c in range(6) for w in range(4)]            # the four positions of the fixture geometry
    a = T._product_with_clouds(rng, ids)
    coll = agg.cloud_from_product(a, S.decode_manifest(a[S.MANIFEST_KEY]))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, res, compat, paths = _archive(tmp, coll, {"all": None, "opening": (0,)})
        assert agg.load_candidate_archive(paths["archive"])["format_version"] == 4
        def rewrite(name, mutate):
            with np.load(paths["archive"], allow_pickle=False) as z:
                arrs = {k: z[k] for k in z.files}
            m = json.loads(str(arrs.pop("manifest_json")))
            mutate(arrs, m)
            np.savez_compressed(tmp / name, **arrs, manifest_json=np.asarray(json.dumps(m, sort_keys=True, default=str)))
            return tmp / name
        cell_block = next(b for b in arc["manifest"]["compatibility"]["blocks"] if b["construction"] == "per_cell" and b["window_set"] == "all")
        names = cell_block["arrays"]
        def split(arrs, m):
            import copy
            gi = arrs[names["group_index"]].copy(); R = arrs[names["ranks"]].copy(); d = arrs[names["reference_draws"]].copy()
            for part, sel in enumerate((np.arange(3), np.arange(3, 6))):
                b = cell_block if part == 0 else copy.deepcopy(cell_block)
                blocks = m["compatibility"]["blocks"]
                if part:
                    b["index"] = len(blocks); b["arrays"] = {k: "split_" + k for k in names}; blocks.append(b)
                else:
                    b = next(x for x in blocks if x["index"] == cell_block["index"])
                b["n_groups"] = int(sel.size)
                arrs[b["arrays"]["group_index"]] = gi[sel]; arrs[b["arrays"]["ranks"]] = R[np.ix_(sel, sel)]; arrs[b["arrays"]["reference_draws"]] = d[sel]
                arrs["compatibility_n_references"][gi[sel]] = sel.size - 1
                arrs["compatibility_score"][gi[sel]] = np.nanmedian(R[np.ix_(sel, sel)], axis=1)
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("split.npz", split)), "complete peer population")
        def draws(arrs, m):
            arrs[names["reference_draws"]] = np.ones_like(arrs[names["reference_draws"]])
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("draws.npz", draws)), "pooled draw counts")
        def scale(arrs, m):
            m["compatibility"]["protocol"]["sgm_scale"][0] *= 2
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("scale.npz", scale)), "protocol sgm_scale")
        def contract_scale(arrs, m):
            m["contract"]["sgm_scale"][1] = 0.5
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("cscale.npz", contract_scale)), "contract sgm_scale")
        def fractional(arrs, m):
            arrs[names["group_index"]] = arrs[names["group_index"]].astype(float) + 0.5
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("frac.npz", fractional)), "must be an integer array")
        def fractional_counts(arrs, m):
            arrs["compatibility_n_references"] = arrs["compatibility_n_references"].astype(float)
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("fraccount.npz", fractional_counts)), "must be an integer array")
        def missing_block(arrs, m):
            m["compatibility"]["blocks"] = [b for b in m["compatibility"]["blocks"] if b["index"] != cell_block["index"]]
            for k in names.values():
                arrs.pop(k)
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("missing.npz", missing_block)), "no block")
        def extra_block(arrs, m):
            # an unexpected block on the opening per_cell_window context (a singleton reference set): refused
            import copy
            b = copy.deepcopy(cell_block); b["index"] = 99; b["construction"] = "per_cell_window"; b["cell"] = 0; b["window_set"] = "opening"
            b["scope"] = WC
            b["arrays"] = {k: "extra_" + k for k in names}; m["compatibility"]["blocks"].append(b)
            gi = np.asarray([i for i, (c, w, s) in enumerate(zip(arrs["construction"], arrs["window_set"], arrs["cell"]))
                             if c == "per_cell_window" and w == "opening" and s == 0] + [0], np.int64)
            arrs["extra_group_index"] = gi; arrs["extra_ranks"] = np.full((2, 2), np.nan); arrs["extra_reference_draws"] = np.ones(2, np.int64)
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("extra.npz", extra_block)), "complete peer population")
        def foreign_group(arrs, m):
            gi = arrs[names["group_index"]].copy(); gi[0] = int(np.flatnonzero(arrs["construction"] == "global")[0]); arrs[names["group_index"]] = gi
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("foreign.npz", foreign_group)), "complete peer population")
        def false_insufficient(arrs, m):
            # a scored group of a multi-cell context relabeled as insufficient_peers while its block is dropped: refused
            m["compatibility"]["blocks"] = [b for b in m["compatibility"]["blocks"] if b["index"] != cell_block["index"]]
            gi = arrs[names["group_index"]]
            st = arrs["compatibility_status"].astype(object); st[gi] = "insufficient_peers"; arrs["compatibility_status"] = st.astype(str)
            arrs["compatibility_score"][gi] = np.nan; arrs["compatibility_n_references"][gi] = 0
            for k in names.values():
                arrs.pop(k)
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("false.npz", false_insufficient)), "no block")
        # distinct scientific identities (the re-review's two mutations): a duplicated cell id inside a per_cell context and a
        # duplicated window position inside a per_cell_window context are refused although every group index is unique
        def duplicate_cell(arrs, m):
            gi = arrs[names["group_index"]]; arrs["cell"][gi[1]] = arrs["cell"][gi[0]]
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("dupcell.npz", duplicate_cell)), "distinct cells", "duplicated cell identities")
        win_block = next(b for b in arc["manifest"]["compatibility"]["blocks"] if b["construction"] == "per_cell_window" and b["window_set"] == "all")
        def duplicate_chunk(arrs, m):
            gi = arrs[win_block["arrays"]["group_index"]]; arrs["chunk"][gi[1]] = arrs["chunk"][gi[0]]
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("dupchunk.npz", duplicate_chunk)), "distinct window positions")
        def pooled_with_chunk(arrs, m):
            gi = arrs[names["group_index"]]; arrs["chunk"][gi[0]] = 0            # a pooled cell group must carry chunk -1
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(rewrite("pooledchunk.npz", pooled_with_chunk)), "carries chunk 0")
        # the constructor cannot be fed a split population: the reader refuses before any selection
        assert agg.load_candidate_archive(paths["archive"])["compatibility"]["blocks"]


def _detector_fixture(rng, tmp, n_cells=4, n_windows=3):
    """A six-parameter fixture on the detector table (prior widths as the scale, centers inside the box, cells slightly
    apart), exported with the detector target binding so the companion's dry-run and report paths accept it."""
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    if str(REPO / "Script_Bank" / "Analysis") not in sys.path:
        sys.path.insert(0, str(REPO / "Script_Bank" / "Analysis"))
    import SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI as companion
    table = det.DETECTOR_PARAMETERIZATION
    keys = [e["KEY"] for e in table]
    lo = np.array([e["PRIOR_RANGE"][0] for e in table], float); hi = np.array([e["PRIOR_RANGE"][1] for e in table], float)
    ids = [(0, c, w) for c in range(n_cells) for w in range(n_windows)]
    cloud = np.stack([(lo + hi) / 2 + 0.004 * (hi - lo) * c + 0.002 * (hi - lo) * w + rng.normal(scale=0.01, size=(40, 6)) * (hi - lo)
                      for (_, c, w) in ids])                          # cells and windows overlap, so every score is positive
    geom = S.window_geometry(n_frames=100, step_frames=100, span_frames=100 * n_windows)
    coll = _collection(cloud, ids, geometry=geom, scale=tuple(hi - lo), keys=keys)
    R = companion._resolve(2.0, "FAB")
    target = companion._target(SimpleNamespace(condition="FAB"), R)
    arc, res, compat, paths = _archive(tmp, coll, {"all": None, "opening": (0,)}, table=table, target=target,
                                       product_label=f"{R['paths'].project_alias}_2S_50FPS_TEST")
    return arc, R, target, table, companion


def test_zero_mass_components_report_truthfully_in_the_dry_run_and_the_report():
    """R1 of the 2026-11-06 review: a component of mass zero is valid under both policies; the dry-run and the report
    complete, state no positive members and an explicitly undefined effective count, and keep the component in provenance."""
    import contextlib, io
    rng = np.random.default_rng(32)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, R, target, table, companion = _detector_fixture(rng, tmp / "arc")
        label = arc["manifest"]["product_label"]
        def comp(weight, policy):
            return {"source_experiment": label, "condition": "FAB", "window_set": "all", "grouping": "per_cell", "statistic": "sgm",
                    "weight": weight, "member_weighting": policy, **({"rank_scope": "construction"} if policy != "uniform" else {})}
        for policy in ("uniform", "posterior_rank_median"):
            spec = {"block": {"posterior_sample_pool_choice": "empirical_representatives", "candidate_archive": arc["path"],
                              "candidate_archive_sha256": arc["sha256"], "rationale": "zero-mass regression"},
                    "component": [comp(1.0, "uniform"), comp(0.0, policy)]}
            vec, pr, rec = ndli.representatives_from_spec(spec, arc, table, R["plo"], R["phi"], target=target)
            zero = rec["components"][1]
            assert zero["weight"] == 0.0 and zero["n_positive_members"] == 0 and zero["n_zero_members"] == 4 and zero["effective_members"] is None
            assert all(m["probability"] == 0.0 for m in zero["members"]) and len(rec["components"]) == 2
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                companion._build_representatives(SimpleNamespace(condition="FAB", dry_run=True), R, spec["block"], spec, tmp / f"{policy}.npz", tmp / "s.toml")
            text = out.getvalue()
            assert "component 1:" in text and "0 positive" in text and "undefined (component mass 0" in text, text
            nu = ndli.build_nuisance_dli(spec, R["imaging_keys"], R["plo"], R["phi"], table=table, archive=arc, target=target)
            rep = DiagnosticReporter(stage="Nuisance_DLI", enabled=True, dump=True, dump_dir=tmp / f"rep_{policy}", run_label="T", timestamp="now")
            companion._report_representatives(rep, nu, R, arc)
            rep.write_report()
            report = (tmp / f"rep_{policy}" / "report.md").read_text()
            assert "| 4 / 0 / 4 |" in report and "undefined (component mass 0" in report and "| statistic |" in report, report[:3000]
            assert "representative_row_provenance | PASS" in report
            rows = (tmp / f"rep_{policy}" / "representatives_rows.csv").read_text().splitlines()
            assert rows[0].endswith(",".join(f"{k}_physical" for k in R["imaging_keys"])) and "_stored_log10" in rows[0]
            assert all(line.split(",")[3] for line in rows[1:])


def test_override_collisions_keep_every_stored_row_s_provenance():
    """R2 of the 2026-11-06 review: overrides that make distinct rows identical keep each row's memberships, by the
    recorded stored-row identity, with contributions that add up to the row's probability; the baseline aligns."""
    rng = np.random.default_rng(33)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, R, target, table, companion = _detector_fixture(rng, tmp / "arc")
        label = arc["manifest"]["product_label"]
        values = {"mu_r": 1.7, "sigma_r": 0.2, "mu_pc": 110.0, "sigma_pc": 0.5, "prob_photo_bleach": 0.05, "lambda_rate": 5.0}
        spec = {"block": {"posterior_sample_pool_choice": "empirical_representatives", "candidate_archive": arc["path"],
                          "candidate_archive_sha256": arc["sha256"], "rationale": "collision regression"},
                "component": [{"source_experiment": label, "condition": "FAB", "window_set": "all", "grouping": "per_cell", "statistic": "sgm",
                               "weight": 1.0, "member_weighting": "posterior_rank_median", "rank_scope": "construction"}],
                "override": {k: {"value": v, "reason": "collision probe"} for k, v in values.items()}}
        vec, pr, rec = ndli.representatives_from_spec(spec, arc, table, R["plo"], R["phi"], target=target)
        assert vec.shape == (4, 6) and len(np.unique(vec, axis=0)) == 1              # four stored rows, one final value
        rows_of = [m["stored_row"] for m in rec["components"][0]["members"]]
        assert sorted(rows_of) == [0, 1, 2, 3] and np.allclose([rec["components"][0]["members"][j]["probability"] for j in range(4)], pr[rows_of])
        assert len(rec["equal_member_baseline"]["probabilities"]) == 4
        nu = ndli.build_nuisance_dli(spec, R["imaging_keys"], R["plo"], R["phi"], table=table, archive=arc, target=target)
        rep = DiagnosticReporter(stage="Nuisance_DLI", enabled=True, dump=True, dump_dir=tmp / "rep", run_label="T", timestamp="now")
        companion._report_representatives(rep, nu, R, arc)
        rep.write_report()
        import csv
        with (tmp / "rep" / "representatives_rows.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 4 and all(r["memberships (component, member, cell, chunk, contribution)"] for r in rows)
        for r in rows:
            contribs = [float(x.strip(" ()").split(", ")[4]) for x in r["memberships (component, member, cell, chunk, contribution)"].split("; ")]
            assert np.isclose(sum(contribs), float(r["probability"]), rtol=1e-5)
        assert "representative_row_provenance | PASS" in (tmp / "rep" / "report.md").read_text()
        with (tmp / "rep" / "representatives_members.csv").open() as fh:
            members = list(csv.DictReader(fh))
        assert sorted(int(m["stored_row"]) for m in members) == [0, 1, 2, 3]
        # a legacy record without stored rows still reports (the exact-match fallback) and loads
        legacy = dict(nu.selection); legacy["components"] = [dict(c, members=[{k: v for k, v in m.items() if k != "stored_row"} for m in c["members"]])
                                                             for c in legacy["components"]]
        old = ndli.NuisanceDLI.from_representatives(R["imaging_keys"], nu.samples, nu.probabilities, legacy, prior_low=R["plo"], prior_high=R["phi"])
        rep2 = DiagnosticReporter(stage="Nuisance_DLI", enabled=True, dump=True, dump_dir=tmp / "rep2", run_label="T", timestamp="now")
        companion._report_representatives(rep2, old, R, arc)
        rep2.write_report()
        assert "representative_row_provenance | FAIL" in (tmp / "rep2" / "report.md").read_text()     # disclosed, never silent


# ---------------------------------------------------------------------------------------------------------------------
# Rank coordinate spaces (2026-11-09): the same groups and reference populations ranked over a declared subset of the
# coordinates; archive format 3; the constructor's rank_coordinates selection; the cell-by-window probability grid
# ---------------------------------------------------------------------------------------------------------------------
def _archive_with_spaces(tmp, coll, window_sets, spaces, *, kinds=("FAB",), table=TABLE, target=TARGET, product_label="P"):
    res = agg.aggregate(coll, table, window_sets=window_sets)
    compat = _compat(res, coll)
    extra = agg.compute_rank_spaces(res, coll, spaces, scope=WC)
    man = agg.export_manifest(res, product_label=product_label, condition_labels=list(kinds), target=target, compatibility=compat, rank_spaces=extra)
    paths = agg.write_aggregation_exports(res, tmp, table, manifest=man, compatibility=compat, rank_spaces=extra)
    return agg.load_candidate_archive(paths["archive"]), res, compat, extra, paths


def _brute_force_rank(reference_cloud, reference_sgm, candidate, scale, idx):
    """Plain counting over the projected coordinates: the share of reference draws strictly farther from the SGM than the candidate."""
    z = np.asarray(reference_cloud, float)[:, idx]; v = np.asarray(reference_sgm, float)[idx]; c = np.asarray(candidate, float)[idx]
    s = np.asarray(scale, float)[idx]
    dz = np.sqrt((((z - v) / s) ** 2).sum(axis=1)); dc = float(np.sqrt((((c - v) / s) ** 2).sum()))
    return float(np.sum(dz > dc)) / z.shape[0]


def test_a_rank_space_projects_the_same_groups_and_references_onto_its_coordinates():
    """A subset space equals plain counting over the projected coordinates; the full-key subset equals the primary space;
    the groups, statuses and reference populations are those of the primary space; the parser refuses malformed specs."""
    rng = np.random.default_rng(41)
    ids = [(0, c, w) for c in range(4) for w in range(4)]
    a = T._product_with_clouds(rng, ids)
    coll = agg.cloud_from_product(a, S.decode_manifest(a[S.MANIFEST_KEY]))
    res = agg.aggregate(coll, TABLE, window_sets={"all": None, "opening": (0,)})
    full = _compat(res, coll)
    sub = _compat(res, coll, coordinates=["c", "a"], label="ac")          # any order in; table order out
    assert sub.label == "ac" and sub.coordinates == ("a", "c") and sub.key_indices == (0, 2)
    assert full.coordinates == ("a", "b", "c") and full.key_indices == (0, 1, 2) and full.label == agg.RANK_SPACE_ALL
    assert np.array_equal(sub.status, full.status) and np.array_equal(sub.n_references, full.n_references)
    assert len(sub.blocks) == len(full.blocks)
    rows_of = {(int(coll.kind_index[r]), int(coll.cell[r]), int(coll.chunk[r])): int(r) for r in range(coll.n_windows)}
    scale = np.asarray(res.scale, float); idx = np.asarray([0, 2])
    checked = 0
    for bs, bf in zip(sub.blocks, full.blocks):
        assert np.array_equal(bs.group_index, bf.group_index) and np.array_equal(bs.reference_draws, bf.reference_draws)
        assert (bs.kind, bs.window_set, bs.construction, bs.cell) == (bf.kind, bf.window_set, bf.construction, bf.cell)
        for j, gj in enumerate(bs.group_index):
            ref = res.groups[int(gj)]
            pool = np.concatenate([coll.cloud[rows_of[(0, c, w)]] for c, w in ref.members], axis=0)
            for i, gi in enumerate(bs.group_index):
                if i == j:
                    assert np.isnan(bs.ranks[i, j]); continue
                expect = _brute_force_rank(pool, ref.sgm_log10, res.groups[int(gi)].sgm_log10, scale, idx)
                assert abs(bs.ranks[i, j] - expect) < 1e-12, (bs.ranks[i, j], expect)
                checked += 1
    assert checked > 50
    # the scores are the row medians in that space, and generally differ from the primary space
    for i in range(len(res.groups)):
        if sub.status[i] == "computed":
            blk = next(b for b in sub.blocks if i in b.group_index.tolist()); a_ = blk.group_index.tolist().index(i)
            assert abs(sub.score[i] - float(np.median(np.delete(blk.ranks[a_], a_)))) < 1e-12
        else:
            assert np.isnan(sub.score[i])
    assert not np.allclose(np.nan_to_num(sub.score), np.nan_to_num(full.score))
    # every key as a declared subset is the primary space again (equal values); a labeled 'all' subset is refused
    same = _compat(res, coll, coordinates=["a", "b", "c"], label="everything")
    assert np.array_equal(np.nan_to_num(same.score), np.nan_to_num(full.score)) and same.coordinates == full.coordinates
    _expect(agg.AggregationError, lambda: _compat(res, coll, coordinates=["a"], label=agg.RANK_SPACE_ALL), "own label")
    _expect(agg.AggregationError, lambda: _compat(res, coll, coordinates=["zz"], label="x"), "not parameter keys")
    # the parser
    assert agg.parse_rank_spaces("ac=c,a;b_only=b", KEYS) == {"ac": ("a", "c"), "b_only": ("b",)}
    assert agg.parse_rank_spaces("", KEYS) == {} and agg.parse_rank_spaces(None, KEYS) == {}
    for bad, needle in (("all=a,b", "differ from"), ("AC=a,c", "must match"), ("ac=a,c;ac=b", "declared twice"), ("ac=a,zz", "unknown parameter keys"),
                        ("ac=a,a", "named twice"), ("ac=a,b,c", "fewer than all"), ("ac", "expected LABEL=")):
        _expect(agg.AggregationError, lambda bad=bad: agg.parse_rank_spaces(bad, KEYS), needle)
    print("  PASS subset ranks equal plain counting over the projected coordinates; same groups, statuses and references; parser refusals")


def test_format_three_archives_carry_every_space_and_malformed_spaces_are_refused():
    """Round trip: the primary arrays and CSV are unchanged in content, the additional space has its own arrays, CSV and
    groups-CSV columns, the reader exposes every space with its coordinates; a format-2 archive reads as the single space
    'all'; a space with wrong coordinates, a wrong scale, borrowed matrices, a stale score or a missing block is refused."""
    rng = np.random.default_rng(42)
    ids = [(0, c, w) for c in range(5) for w in range(4)]
    a = T._product_with_clouds(rng, ids)
    coll = agg.cloud_from_product(a, S.decode_manifest(a[S.MANIFEST_KEY]))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, res, compat, extra, paths = _archive_with_spaces(tmp / "v3", coll, {"all": None, "opening": (0,)}, {"ac": ("a", "c")})
        arc2, _, compat2, paths2 = _archive(tmp / "v2like", coll, {"all": None, "opening": (0,)})
        A, A2 = arc["arrays"], arc2["arrays"]
        assert arc["format_version"] == 4 and arc2["format_version"] == 4
        for k in ("compatibility_score", "compatibility_status", "compatibility_n_references", "sgm_log10", "compatibility_block_0_ranks"):
            assert np.array_equal(np.nan_to_num(A[k]) if A[k].dtype.kind == "f" else A[k], np.nan_to_num(A2[k]) if A2[k].dtype.kind == "f" else A2[k]), k
        assert paths["compatibility_csv"].read_text() == paths2["compatibility_csv"].read_text()
        assert "compatibility_ac_score" in A and "compatibility_ac_block_0_ranks" in A and paths["rank_space_csvs"]["ac"].exists()
        head = paths["groups_csv"].read_text().splitlines()[0]
        assert head.endswith("sgm_outside_prior_keys,compatibility_score_ac,compatibility_status_ac,compatibility_n_references_ac")
        assert head.replace(",compatibility_score_ac,compatibility_status_ac,compatibility_n_references_ac", "") == paths2["groups_csv"].read_text().splitlines()[0]
        spaces = arc["compatibility"]["spaces"]
        assert list(spaces) == ["all", "ac"] and spaces["all"]["coordinates"] == KEYS and spaces["ac"]["coordinates"] == ["a", "c"]
        assert arc["manifest"]["compatibility"]["rank_spaces"]["ac"]["protocol"]["sgm_scale"] == [float(res.scale[0]), float(res.scale[2])]
        assert arc["manifest"]["compatibility"]["rank_spaces"]["ac"]["protocol"]["coordinates"] == ["a", "c"]
        for i in range(len(res.groups)):
            got = spaces["ac"]["by_group"][i]["score"]; want = extra[0].score[i]
            assert (got is None and np.isnan(want)) or abs(got - want) < 1e-15
        comp = agg.archive_component(arc, kind="FAB", window_set="all", construction="per_cell_window")
        assert comp["rank_spaces"] == {"all": KEYS, "ac": ["a", "c"]} and comp["parameter_keys"] == KEYS
        assert all(set(d["compatibility_by_space"]) == {"all", "ac"} and d["compatibility_by_space"]["all"] == d["compatibility"] for d in comp["identities"])
        # an archive without additional spaces exposes the single space 'all'
        comp2 = agg.archive_component(arc2, kind="FAB", window_set="all", construction="per_cell_window")
        assert comp2["rank_spaces"] == {"all": KEYS} and all(list(d["compatibility_by_space"]) == ["all"] for d in comp2["identities"])
        # a genuine format-2 manifest (no rank_spaces section) still reads, as the single space 'all'
        def rewrite(name, mutate):
            with np.load(paths["archive"], allow_pickle=False) as z:
                arrs = {k: z[k] for k in z.files}
            m = json.loads(str(arrs.pop("manifest_json")))
            mutate(arrs, m)
            np.savez_compressed(tmp / name, **arrs, manifest_json=np.asarray(json.dumps(m, sort_keys=True, default=str)))
            return tmp / name
        def as_v2(arrs, m):
            _as_legacy(m, 2)
            for k in [k for k in arrs if k.startswith("compatibility_ac_")]:
                arrs.pop(k)
        v2 = agg.load_candidate_archive(rewrite("v2.npz", as_v2))
        assert v2["format_version"] == 2 and list(v2["compatibility"]["spaces"]) == ["all"] and v2["compatibility"]["comparison_scope"] == WC
        # the same archive as legacy format 3 (protocol v1, no scope field): read under its recorded within_cell meaning
        v3 = agg.load_candidate_archive(rewrite("v3.npz", lambda arrs, m: _as_legacy(m, 3)))
        assert v3["format_version"] == 3 and v3["compatibility"]["comparison_scope"] == WC and list(v3["compatibility"]["spaces"]) == ["all", "ac"]
        assert v3["compatibility"]["protocol"]["name"] == agg.LEGACY_COMPATIBILITY_PROTOCOL
        assert np.array_equal(v3["arrays"]["compatibility_score"], arc["arrays"]["compatibility_score"], equal_nan=True)
        # malformed spaces
        def wrong_coords(arrs, m):
            m["compatibility"]["rank_spaces"]["ac"]["coordinates"] = ["a", "b"]
        def wrong_proto_coords(arrs, m):
            m["compatibility"]["rank_spaces"]["ac"]["protocol"]["coordinates"] = ["a", "b"]
        def wrong_scale(arrs, m):
            m["compatibility"]["rank_spaces"]["ac"]["protocol"]["sgm_scale"] = [2.0 * float(res.scale[0]), float(res.scale[2])]
        def borrowed(arrs, m):
            blk = m["compatibility"]["rank_spaces"]["ac"]["blocks"][0]
            blk["arrays"]["ranks"] = "compatibility_block_0_ranks"       # the primary space's matrix
        def stale(arrs, m):
            s = arrs["compatibility_ac_score"].copy(); i = int(np.flatnonzero(np.isfinite(s))[0]); s[i] = s[i] + 0.25 if s[i] < 0.5 else s[i] - 0.25
            arrs["compatibility_ac_score"] = s
        def missing_block(arrs, m):
            m["compatibility"]["rank_spaces"]["ac"]["blocks"].pop()
        def not_subset(arrs, m):
            m["compatibility"]["rank_spaces"]["ac"]["coordinates"] = ["a", "b", "c"]
        def primary_not_all(arrs, m):
            m["compatibility"]["rank_spaces"]["all"]["coordinates"] = ["a", "c"]
        for name, mut, needle in (("coords", wrong_coords, "protocol's coordinates"), ("pcoords", wrong_proto_coords, "protocol's coordinates"),
                                  ("scale", wrong_scale, "another metric"), ("borrowed", borrowed, "does not belong to this"),
                                  ("stale", stale, "stored score"), ("missing", missing_block, "no block"), ("notsub", not_subset, "proper non-empty subset"),
                                  ("primary", primary_not_all, "every parameter key")):
            _expect(agg.AggregationError, lambda name=name, mut=mut: agg.load_candidate_archive(rewrite(f"{name}.npz", mut)), needle)
        print("  PASS format-3 round trip, format-2 compatibility, eight malformed-space refusals")


def test_the_constructor_selects_a_rank_space_by_its_coordinates_and_records_the_override_overlap():
    """`rank_coordinates` selects the archive space over exactly those keys (default and the full key list = 'all'); a
    space the archive lacks, or rank_coordinates on a uniform component, is refused; the record names the space, the
    coordinates and the overlap with the overrides, and the equal-member baseline is unchanged by the space."""
    rng = np.random.default_rng(43)
    ids = [(0, c, w) for c in range(4) for w in range(4)]
    # overlapping windows within every cell, shifted differently along each coordinate, so every within-cell score is
    # positive and the spaces 'all' and 'ac' weight the windows differently
    cloud = np.stack([np.array([-0.5 - 0.02 * c + 0.003 * w, 0.4 + 0.002 * w, 0.5 - 0.004 * w]) + rng.normal(scale=0.006, size=(60, 3))
                      for (_, c, w) in ids])
    coll = _collection(cloud, ids)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, res, compat, extra, paths = _archive_with_spaces(tmp, coll, {"all": None}, {"ac": ("a", "c")})
        comp_all = _comp("per_cell_window", 1.0, member_weighting="posterior_rank_median")
        comp_ac = {**comp_all, "rank_coordinates": ["c", "a"]}
        comp_full = {**comp_all, "rank_coordinates": ["a", "b", "c"]}
        v0, p0, r0 = ndli.representatives_from_spec(_spec(arc, comp_all), arc, TABLE, PLO, PHI, target=TARGET)
        v1, p1, r1 = ndli.representatives_from_spec(_spec(arc, comp_ac), arc, TABLE, PLO, PHI, target=TARGET)
        v2, p2, r2 = ndli.representatives_from_spec(_spec(arc, comp_full), arc, TABLE, PLO, PHI, target=TARGET)
        assert np.array_equal(v0, v1) and np.array_equal(v0, v2) and np.allclose(p0, p2) and not np.allclose(p0, p1)
        assert r0["components"][0]["rank_space"] == "all" and r2["components"][0]["rank_space"] == "all"
        assert r1["components"][0]["rank_space"] == "ac" and r1["components"][0]["rank_coordinates"] == ["a", "c"]
        assert r1["components"][0]["rank_coordinates_declared"] == ["c", "a"] and r1["rank_spaces_used"] == ["ac"]
        # the within-cell weights follow the selected space's scores exactly
        comp = agg.archive_component(arc, kind="FAB", window_set="all", construction="per_cell_window")
        s_ac = np.asarray([d["compatibility_by_space"]["ac"]["score"] for d in comp["identities"]], float)
        cells = np.asarray([d["cell"] for d in comp["identities"]])
        want = np.zeros_like(s_ac)
        for c in np.unique(cells):
            m = cells == c; want[m] = (1.0 / np.unique(cells).size) * s_ac[m] / s_ac[m].sum()
        assert np.allclose(np.asarray([mm["probability"] for mm in r1["components"][0]["members"]]), want)
        assert np.allclose([mm["score_used"] for mm in r1["components"][0]["members"]], s_ac)
        # the equal-member baseline does not depend on the space
        assert np.allclose(r0["equal_member_baseline"]["probabilities"], r1["equal_member_baseline"]["probabilities"])
        # the override overlap: ranking over a, c while overriding b is disjoint; ranking over all while overriding b is not
        over = {"b": {"value": 0.5, "reason": "test"}}
        _, _, rd = ndli.representatives_from_spec(_spec(arc, comp_ac, overrides=over), arc, TABLE, PLO, PHI, target=TARGET)
        _, _, rn = ndli.representatives_from_spec(_spec(arc, comp_all, overrides=over), arc, TABLE, PLO, PHI, target=TARGET)
        _, _, ru = ndli.representatives_from_spec(_spec(arc, _comp("per_cell_window", 1.0), overrides=over), arc, TABLE, PLO, PHI, target=TARGET)
        assert rd["rank_coordinates_disjoint_from_overrides"] is True and rd["components"][0]["rank_coordinates_overridden"] == []
        assert rn["rank_coordinates_disjoint_from_overrides"] is False and rn["components"][0]["rank_coordinates_overridden"] == ["b"]
        assert ru["rank_coordinates_disjoint_from_overrides"] is None
        # refusals: a space the archive lacks; rank_coordinates on a uniform component; malformed lists
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, {**comp_all, "rank_coordinates": ["a", "b"]}), arc, TABLE, PLO, PHI, target=TARGET),
                "no rank coordinate space", "--compatibility-coordinates")
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, {**_comp("per_cell_window", 1.0), "rank_coordinates": ["a", "c"]}), arc, TABLE, PLO, PHI, target=TARGET),
                "posterior_rank_median' only")
        for bad in ([], ["a", "a"], ["zz"], "a"):
            _expect(ValueError, lambda bad=bad: ndli.representatives_from_spec(_spec(arc, {**comp_all, "rank_coordinates": bad}), arc, TABLE, PLO, PHI, target=TARGET),
                    "rank_coordinates must be")
        # an archive without the space (format 2 style): the default works, the subset is refused with the archive's spaces named
        arc2, *_ = _archive(tmp / "plain", coll, {"all": None})
        ndli.representatives_from_spec(_spec(arc2, comp_all), arc2, TABLE, PLO, PHI, target=TARGET)
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc2, comp_ac), arc2, TABLE, PLO, PHI, target=TARGET), "'all': ['a', 'b', 'c']")
        avail = ndli.component_weighting_availability(arc)[("FAB", "all", "per_cell_window")]
        assert avail.startswith("rank space all (a, b, c): available") and "; rank space ac (a, c): available" in avail
        print("  PASS space selection by coordinates, exact within-cell weights, override overlap flags, refusals, availability text")


def test_availability_is_evaluated_separately_for_every_rank_space():
    """R1 of the 2026-11-09 review: a usable primary space with an unusable subset (a cell whose every window scores zero
    over the subset's coordinates) is reported per space, never summarized by the primary space; the reverse case (the
    primary space unusable, a subset usable) is advertised as well; missing scores keep their own wording."""
    rng = np.random.default_rng(45)
    ids = [(0, c, w) for c in range(3) for w in range(4)]
    # the reviewer's fixture: windows separated in `a`, broad and overlapping in `b`, identical in `c`
    clouds = [np.column_stack([np.full(51, -0.8 + 0.05 * w + 0.001 * c), np.linspace(0.0, 1.0, 51), np.full(51, 0.5)]) for _, c, w in ids]
    coll = _collection(np.asarray(clouds), ids)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, *_ = _archive_with_spaces(tmp / "fwd", coll, {"all": None}, {"a_only": ("a",)})
        comp = agg.archive_component(arc, kind="FAB", window_set="all", construction="per_cell_window")
        ndli.member_probabilities("posterior_rank_median", 1.0, comp, rank_scope=WC)                    # the primary space normalizes
        _expect(ValueError, lambda: ndli.member_probabilities("posterior_rank_median", 1.0, comp, rank_coordinates=["a"], rank_scope=WC),
                "every selected-window score is zero")
        text = ndli.component_weighting_availability(arc)[("FAB", "all", "per_cell_window")]
        assert text.startswith('rank space all (a, b, c): available under rank_scope "within_cell", compatibility scores computed for 12/12 members and normalizable; '), text
        assert 'rank space a_only (a): unavailable under rank_scope "within_cell": compatibility scores computed for 12/12 members, but the policy cannot be normalized' in text, text
        assert "every selected-window score is zero" in text and text.count("): available") + text.count("): unavailable") == 2, text
        # the reverse: identical windows in `a` (every `a`-rank is 1, normalizable) with `b` and `c` making the all-space zero within a cell
        clouds2 = [np.column_stack([-0.8 + 0.001 * c + np.linspace(-0.01, 0.01, 51), np.linspace(0.0, 1.0, 51) if w < 2 else np.linspace(0.0, 1.0, 51) + 3.0,
                                     np.full(51, 0.5) if w < 2 else np.full(51, 0.9)]) for _, c, w in ids]
        coll2 = _collection(np.asarray(clouds2), ids)
        arc2, *_ = _archive_with_spaces(tmp / "rev", coll2, {"all": None}, {"a_only": ("a",)})
        comp2 = agg.archive_component(arc2, kind="FAB", window_set="all", construction="per_cell_window")
        _expect(ValueError, lambda: ndli.member_probabilities("posterior_rank_median", 1.0, comp2, rank_scope=WC), "every selected-window score is zero")
        p, _, _ = ndli.member_probabilities("posterior_rank_median", 1.0, comp2, rank_coordinates=["a"], rank_scope=WC)
        assert abs(p.sum() - 1.0) < 1e-12
        text2 = ndli.component_weighting_availability(arc2)[("FAB", "all", "per_cell_window")]
        assert text2.startswith('rank space all (a, b, c): unavailable under rank_scope "within_cell": compatibility scores computed for 12/12 members, but the policy cannot be normalized'), text2
        assert '; rank space a_only (a): available under rank_scope "within_cell", compatibility scores computed for 12/12 members and normalizable' in text2, text2
        # the scope is explicit and never implied: no scope, an unknown scope and the other scope are refused
        _expect(ValueError, lambda: ndli.member_probabilities("posterior_rank_median", 1.0, comp2, rank_coordinates=["a"]), "needs an explicit rank_scope")
        _expect(ValueError, lambda: ndli.member_probabilities("posterior_rank_median", 1.0, comp2, rank_coordinates=["a"], rank_scope="cell"), "needs an explicit rank_scope")
        _expect(ValueError, lambda: ndli.member_probabilities("posterior_rank_median", 1.0, comp2, rank_coordinates=["a"], rank_scope="construction"),
                "is not the scope this archive's per_cell_window scores were computed under", "'within_cell'", "never renormalized")
        # missing scores (the opening per_cell_window singletons) keep their own wording in every space
        arc3, *_ = _archive_with_spaces(tmp / "miss", coll, {"all": None, "opening": (0,)}, {"a_only": ("a",)})
        text3 = ndli.component_weighting_availability(arc3)[("FAB", "opening", "per_cell_window")]
        assert text3.count("without a computed score (insufficient_peers") == 2, text3
        print("  PASS availability per rank space: usable primary with unusable subset, the reverse, missing scores")


def test_the_grid_figure_and_the_companion_report_for_one_sgm_per_cell_and_window():
    """The companion's report of a per_cell_window selection draws the cell-by-window grid, names the rank space and states
    the override relation; a per_cell selection keeps the per-row figure."""
    import matplotlib
    matplotlib.use("Agg")
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    from srm_and_sbi_monomer_dimer_alp.visualization_aggregation import figure_member_probability_grid
    P = np.array([[0.1, 0.0, 0.2], [0.05, 0.15, 0.1]]); B = np.full((2, 3), 0.1)
    fig = figure_member_probability_grid([3, 7], [0.0, 2.0, 4.0], P, B, title="t", subtitle="s", window_seconds=2.0)
    assert fig.axes and any("zero probability (1" in t.get_text() for t in fig.legends[0].get_texts())
    rng = np.random.default_rng(44)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, R, target, table, companion = _detector_fixture(rng, tmp / "arc", n_cells=3, n_windows=4)
        spec = {"block": {"posterior_sample_pool_choice": "empirical_representatives", "candidate_archive": arc["path"], "rationale": "test"},
                "component": [{"source_experiment": arc["manifest"]["product_label"], "condition": "FAB", "window_set": "all", "grouping": "per_cell_window",
                               "statistic": "sgm", "weight": 1.0, "member_weighting": "posterior_rank_median", "rank_scope": "within_cell"}],
                "override": {"prob_photo_bleach": {"value": 0.05, "reason": "test"}}}
        nu = ndli.build_nuisance_dli(spec, R["imaging_keys"], R["plo"], R["phi"], table=det.DETECTOR_PARAMETERIZATION, archive=arc, target=target)
        rep = DiagnosticReporter(stage="Nuisance_DLI", enabled=True, dump=True, dump_dir=tmp / "rep", run_label="T", timestamp="now")
        companion._report_representatives(rep, nu, R, arc)
        rep.write_report()
        text = (tmp / "rep" / "report.md").read_text()
        assert (tmp / "rep" / "figures" / "representatives_probabilities.png").exists()
        assert "the stored sampling probability of every row itself" in text and "share of its cell" not in text
        assert "Equal cell totals are this policy's rule (rank scope within_cell)" in text and "rank scope `within_cell`" in text
        assert "rank space `all`" in text and "include overridden coordinate(s) prob_photo_bleach" in text
        members = (tmp / "rep" / "representatives_members.csv").read_text().splitlines()
        assert "rank_space,score_used" in members[0] and members[0].endswith(",rank_scope")
        assert members[1].split(",")[-5] == "all" and members[1].split(",")[-1] == "within_cell"
        totals = (tmp / "rep" / "representatives_cell_totals.csv").read_text().splitlines()
        assert len(totals) == 4 and all(abs(float(line.split(",")[1]) - 1 / 3) < 1e-12 for line in totals[1:])     # equal cell mass by rule
        print("  PASS grid figure drawn for one SGM per cell and window; rank space and override relation stated in the report")


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
