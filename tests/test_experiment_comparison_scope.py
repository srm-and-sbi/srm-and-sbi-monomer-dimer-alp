"""The construction-wide comparison scope of the posterior compatibility ranks (every representative of an aggregation
against ALL the other representatives of that aggregation, in one condition and window set), its archive record
(format 4, protocol ``sgm_distance_rank_v2``), the Nuisance DLI constructor's explicit ``rank_scope`` with the
normalization over the complete component (no mass reserved per cell), and the exact weighted marginal histograms of an
empirical artifact. Synthetic clouds only. The hierarchical ``within_cell`` scope keeps its own tests in
``test_experiment_compatibility.py``.

Run directly from the repository root::

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/test_experiment_comparison_scope.py
"""
import copy
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
import test_experiment_aggregation as T                                              # noqa: E402  (fixtures: TABLE, KEYS, TARGET)
import test_experiment_compatibility as C                                            # noqa: E402  (fixtures: _collection, _spec, _as_legacy)

TABLE, KEYS, TARGET = T.TABLE, T.KEYS, T.TARGET
PLO, PHI = C.PLO, C.PHI
_expect, _collection, _spec, _as_legacy = T._expect, C._collection, C._spec, C._as_legacy
CW, WC = agg.SCOPE_CONSTRUCTION, agg.SCOPE_WITHIN_CELL
CELLS = (3, 7, 11, 20)                               # non-contiguous cell identities on purpose


def _geometry(n_windows):
    return S.window_geometry(n_frames=100, step_frames=100, span_frames=100 * n_windows)


def _population(rng, *, cells=CELLS, n_windows=3, n_draws=40, spread=0.05, far_cell=None, far_step=0.0, all_far=False):
    """Overlapping window clouds (cells and windows slightly apart). ``far_cell`` moves one cell's windows far from every
    other cell (and, with ``far_step``, from one another); ``all_far`` separates every window from every other."""
    ids = [(0, c, w) for c in cells for w in range(n_windows)]
    clouds = []
    for n, (_, c, w) in enumerate(ids):
        center = np.array([-0.5, 0.5, 0.5]) + 0.03 * cells.index(c) * np.array([1.0, -1.0, 0.5]) + 0.02 * w * np.array([0.5, 1.0, -1.0])
        if c == far_cell:
            center = center + 0.6 + far_step * w
        if all_far:
            center = np.array([-0.9, 0.1, 0.1]) + 0.9 * n
        clouds.append(center + rng.normal(scale=0.004 if (all_far or c == far_cell) else spread, size=(n_draws, 3)))
    return _collection(np.stack(clouds), ids, geometry=_geometry(n_windows)), ids


def _archive(tmp, coll, window_sets, *, scope=CW, spaces=None, table=TABLE, target=TARGET, product_label="P"):
    res = agg.aggregate(coll, table, window_sets=window_sets)
    compat = agg.compute_compatibility(res, coll, scope=scope)
    extra = agg.compute_rank_spaces(res, coll, spaces, scope=scope)
    man = agg.export_manifest(res, product_label=product_label, condition_labels=["FAB"], target=target, compatibility=compat, rank_spaces=extra)
    paths = agg.write_aggregation_exports(res, tmp, table, manifest=man, compatibility=compat, rank_spaces=extra)
    return agg.load_candidate_archive(paths["archive"]), res, compat, paths


def _comp(grouping, weight, window_set="all", statistic="sgm", member_weighting=None, rank_scope=None, **extra):
    c = {"source_experiment": "P", "condition": "FAB", "window_set": window_set, "grouping": grouping, "statistic": statistic, "weight": weight, **extra}
    if member_weighting:
        c["member_weighting"] = member_weighting
    if rank_scope:
        c["rank_scope"] = rank_scope
    return c


def _rank(c, w=1.0, window_set="all", **extra):
    return _comp(c, w, window_set, member_weighting="posterior_rank_median", rank_scope=CW, **extra)


def _direct_rank(reference_cloud, reference_sgm, candidate, scale):
    """Plain counting, no sorting and no lookup: the share of reference draws strictly farther from the reference's own SGM."""
    radius = np.sqrt((((np.asarray(reference_cloud, np.float64) - reference_sgm) / scale) ** 2).sum(axis=1))
    reach = np.sqrt((((np.asarray(candidate, np.float64) - reference_sgm) / scale) ** 2).sum())
    return int(np.sum(radius > reach)) / radius.size


def _pool(coll, ids, group):
    """A group's reference cloud: the draws of its member windows (its equal-window or equal-cell mixture, or one window)."""
    row = {(c, w): k for k, (_, c, w) in enumerate(ids)}
    return np.concatenate([coll.cloud[row[(c, w)]].astype(np.float64) for c, w in group.members], axis=0)


def test_every_candidate_is_ranked_against_all_the_other_groups_of_its_construction_by_direct_counting():
    rng = np.random.default_rng(101)
    coll, ids = _population(rng)
    res = agg.aggregate(coll, TABLE, window_sets={"all": None})
    compat = agg.compute_compatibility(res, coll)                                 # the default IS the construction-wide scope
    assert compat.scope == CW and agg.rank_scored_constructions(CW) == ("per_cell", "per_window_position", "per_cell_window")
    blocks = {b.construction: b for b in compat.blocks}
    assert {k: b.group_index.size for k, b in blocks.items()} == {"per_cell": 4, "per_window_position": 3, "per_cell_window": 12}   # from the geometry
    checked = 0
    for construction, blk in blocks.items():
        assert blk.cell == -1 and blk.scope == CW and blk.kind == "FAB" and blk.window_set == "all"
        gi = [int(i) for i in blk.group_index]
        assert sorted(gi) == [i for i, g in enumerate(res.groups) if g.construction == construction]      # the COMPLETE population, once
        for a, i in enumerate(gi):
            for b, j in enumerate(gi):
                if a == b:
                    assert np.isnan(blk.ranks[a, b])                              # the candidate's own group, and only it
                    continue
                want = _direct_rank(_pool(coll, ids, res.groups[j]), res.groups[j].sgm_log10, res.groups[i].sgm_log10, res.scale)
                assert blk.ranks[a, b] == want, (construction, a, b)
                checked += 1
            row = np.delete(blk.ranks[a], a)
            assert compat.status[i] == "computed" and compat.n_references[i] == len(gi) - 1 and compat.score[i] == float(np.median(row))
        assert blk.reference_draws.tolist() == [res.groups[j].n_draws for j in gi]
    assert checked == 4 * 3 + 3 * 2 + 12 * 11
    g = [i for i, x in enumerate(res.groups) if x.construction == "global"]
    assert len(g) == 1 and compat.status[g[0]] == "not_applicable" and np.isnan(compat.score[g[0]]) and compat.n_references[g[0]] == 0
    # exclusion is by group identity, never by vector equality: two windows with identical clouds (identical SGMs) stay peers
    cloud = coll.cloud.copy(); cloud[3] = cloud[0]                                # cell 7 window 0 := cell 3 window 0
    coll2 = _collection(cloud, ids, geometry=_geometry(3))
    res2 = agg.aggregate(coll2, TABLE, window_sets={"all": None}); c2 = agg.compute_compatibility(res2, coll2)
    blk = [b for b in c2.blocks if b.construction == "per_cell_window"][0]
    pos = {(res2.groups[int(i)].cell, res2.groups[int(i)].chunk): k for k, i in enumerate(blk.group_index)}
    a, b = pos[(3, 0)], pos[(7, 0)]
    assert np.array_equal(res2.groups[int(blk.group_index[a])].sgm_log10, res2.groups[int(blk.group_index[b])].sgm_log10)
    assert np.isfinite(blk.ranks[a, b]) and np.isfinite(blk.ranks[b, a]) and blk.ranks[a, b] == blk.ranks[b, a] > 0.9
    assert int(np.sum(np.isnan(blk.ranks))) == 12 and all(c2.n_references[int(i)] == 11 for i in blk.group_index)
    print(f"  PASS {checked} construction-wide entries equal plain counting; N-1 peers each; identity exclusion; identical peers kept")


def test_same_cell_windows_are_peers_among_all_the_others_and_the_scope_is_explicit_in_the_constructor():
    rng = np.random.default_rng(102)
    coll, ids = _population(rng)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, res, compat, _ = _archive(tmp / "cw", coll, {"all": None})
        arw, resw, within, _ = _archive(tmp / "wc", coll, {"all": None}, scope=WC)
        assert arc["compatibility"]["comparison_scope"] == CW and arw["compatibility"]["comparison_scope"] == WC
        pop = [b for b in compat.blocks if b.construction == "per_cell_window"][0]
        pos = {(res.groups[int(i)].cell, res.groups[int(i)].chunk): k for k, i in enumerate(pop.group_index)}
        # the within-cell matrices are the same-cell blocks of the population matrix: same-cell windows are INCLUDED among the
        # eleven peers, and they are three of twelve groups, not the reference set
        for blk in [b for b in within.blocks if b.construction == "per_cell_window"]:
            sel = [pos[(blk.cell, resw.groups[int(i)].chunk)] for i in blk.group_index]
            assert np.array_equal(pop.ranks[np.ix_(sel, sel)], blk.ranks, equal_nan=True)
        cw_idx = [i for i, g in enumerate(res.groups) if g.construction == "per_cell_window"]
        assert all(compat.n_references[i] == 11 and within.n_references[i] == 2 for i in cw_idx)
        assert not np.allclose(compat.score[cw_idx], within.score[cw_idx])
        pc = [i for i, g in enumerate(res.groups) if g.construction == "per_cell"]
        assert np.array_equal(compat.score[pc], within.score[pc])                 # one SGM per cell: the same population in both scopes
        # the constructor: scores normalized over the COMPLETE component; cell totals are whatever they sum to
        vec, p, rec = ndli.representatives_from_spec(_spec(arc, _rank("per_cell_window")), arc, TABLE, PLO, PHI, target=TARGET)
        cr = rec["components"][0]
        s = np.asarray([m["score_used"] for m in cr["members"]]); pm = np.asarray([m["probability"] for m in cr["members"]])
        assert np.array_equal(s, compat.score[cw_idx]) and np.allclose(pm, s / s.sum(), rtol=0, atol=1e-15) and abs(pm.sum() - 1.0) < 1e-12
        totals = {c: sum(m["probability"] for m in cr["members"] if m["cell"] == c) for c in CELLS}
        assert max(totals.values()) - min(totals.values()) > 1e-3                 # no equal cell mass, no factor 1 / n_cells
        norm = cr["normalizers"]
        assert norm["comparison_scope"] == CW and norm["scope"] == "component" and norm["references_per_member"] == [11]
        assert norm["cell_mass_by_cell"] == {str(c): totals[c] for c in CELLS} and norm["zero_mass_cells"] == [] and "cell_mass" not in norm
        assert cr["rank_scope"] == CW and "every other cell–window group" in cr["reference_population"] and rec["rank_scopes_used"] == [CW]
        assert cr["score_source"]["protocol"] == agg.COMPATIBILITY_PROTOCOL and cr["score_source"]["archive_comparison_scope"] == CW
        assert "no mass reserved per cell" in rec["normalization"]
        # changing only the normalization of the within-cell scores is NOT this policy: the reference population differs
        sw = within.score[cw_idx]
        assert not np.allclose(sw / sw.sum(), pm, atol=1e-6)
        # the hierarchical policy stays available under its own name from its own archive, with equal cell mass
        _, pw, recw = ndli.representatives_from_spec(_spec(arw, _comp("per_cell_window", 1.0, member_weighting="posterior_rank_median", rank_scope=WC)),
                                                    arw, TABLE, PLO, PHI, target=TARGET)
        tw = [sum(m["probability"] for m in recw["components"][0]["members"] if m["cell"] == c) for c in CELLS]
        assert np.allclose(tw, 0.25) and recw["components"][0]["rank_scope"] == WC and recw["rank_scopes_used"] == [WC]
        # the scope is validated against the archive, both ways; one scope's scores are never renormalized into another's
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arw, _rank("per_cell_window")), arw, TABLE, PLO, PHI, target=TARGET),
                "is not the scope this archive's per_cell_window scores were computed under", "'within_cell'", "Construction-wide scores need a format-4 archive")
        _expect(ValueError, lambda: ndli.representatives_from_spec(
            _spec(arc, _comp("per_cell_window", 1.0, member_weighting="posterior_rank_median", rank_scope=WC)), arc, TABLE, PLO, PHI, target=TARGET),
            "is not the scope this archive's per_cell_window scores were computed under", "'construction'", "never renormalized")
        # one SGM per cell is construction-wide in either archive: the same probabilities
        _, p1, _ = ndli.representatives_from_spec(_spec(arc, _rank("per_cell")), arc, TABLE, PLO, PHI, target=TARGET)
        _, p2, _ = ndli.representatives_from_spec(_spec(arw, _rank("per_cell")), arw, TABLE, PLO, PHI, target=TARGET)
        assert np.array_equal(p1, p2)
        # the spec states the scope; nothing is implied by the grouping
        for bad, needle in ((_comp("per_cell_window", 1.0, member_weighting="posterior_rank_median"), "needs an explicit rank_scope"),
                            (_comp("per_cell_window", 1.0, member_weighting="posterior_rank_median", rank_scope="population"), "needs an explicit rank_scope"),
                            (_comp("per_cell", 1.0, member_weighting="posterior_rank_median", rank_scope=WC), "applies to grouping per_cell_window only"),
                            (_comp("per_cell", 1.0, rank_scope=CW), "rank_scope applies to member_weighting 'posterior_rank_median' only"),
                            (_comp("global", 1.0, member_weighting="posterior_rank_median", rank_scope=CW), "has no peer")):
            _expect(ValueError, lambda b=bad: ndli.representatives_from_spec(_spec(arc, b), arc, TABLE, PLO, PHI, target=TARGET), needle)
        p = tmp / "s.toml"
        p.write_text('[block]\nposterior_sample_pool_choice = "empirical_representatives"\ncandidate_archive = "x.npz"\nrationale = "r"\n'
                     '[[component]]\nsource_experiment = "P"\ncondition = "FAB"\nwindow_set = "all"\ngrouping = "per_cell_window"\nweight = 1.0\n'
                     'member_weighting = "posterior_rank_median"\n')
        _expect(ValueError, lambda: ndli.load_spec(p, KEYS, PLO, PHI, table=TABLE), "needs an explicit rank_scope")
        p.write_text(p.read_text() + 'rank_scope = "construction"\n')
        ndli.load_spec(p, KEYS, PLO, PHI, table=TABLE)
    print("  PASS same-cell blocks inside the population; complete-component normalization; scope explicit and validated both ways")


def test_window_positions_singletons_and_the_opening_equivalence():
    rng = np.random.default_rng(103)
    coll, ids = _population(rng)
    with tempfile.TemporaryDirectory() as tmp:
        arc, res, compat, _ = _archive(Path(tmp), coll, {"all": None, "opening": (0,)})
        idx = lambda c, s: [i for i, g in enumerate(res.groups) if g.construction == c and g.window_set == s]   # noqa: E731
        # one SGM per window position: the complete population of positions, each reference its equal-cell mixture
        pw = idx("per_window_position", "all")
        assert len(pw) == 3 and all(compat.status[i] == "computed" and compat.n_references[i] == 2 for i in pw)
        assert all(res.groups[i].n_draws == 4 * 40 and res.groups[i].cell == -1 for i in pw)
        _, p, rec = ndli.representatives_from_spec(_spec(arc, _rank("per_window_position")), arc, TABLE, PLO, PHI, target=TARGET)
        s = compat.score[pw]
        assert np.allclose(p[np.argsort(np.argsort([rec["components"][0]["members"][k]["stored_row"] for k in range(3)]))].sum(), 1.0)
        assert np.allclose([m["probability"] for m in rec["components"][0]["members"]], s / s.sum(), rtol=0, atol=1e-15)
        assert "other selected window positions" in rec["components"][0]["reference_population"]
        # a single selected position has no peer: no score, never a fabricated one; rank weighting refused, uniform explicit
        po = idx("per_window_position", "opening")
        assert len(po) == 1 and compat.status[po[0]] == "insufficient_peers" and np.isnan(compat.score[po[0]]) and compat.n_references[po[0]] == 0
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, _rank("per_window_position", window_set="opening")), arc, TABLE, PLO, PHI,
                                                                   target=TARGET), "no computed compatibility score", "insufficient_peers")
        _, pu, ru = ndli.representatives_from_spec(_spec(arc, _comp("per_window_position", 1.0, "opening")), arc, TABLE, PLO, PHI, target=TARGET)
        assert pu.tolist() == [1.0] and ru["components"][0]["member_weighting"] == "uniform"
        # the dataset representative: no peers by construction; selectable with uniform probability one, no rank score
        g = idx("global", "all")
        assert len(g) == 1 and compat.status[g[0]] == "not_applicable" and np.isnan(compat.score[g[0]])
        _, pg, rg = ndli.representatives_from_spec(_spec(arc, _comp("global", 1.0)), arc, TABLE, PLO, PHI, target=TARGET)
        assert pg.tolist() == [1.0] and rg["components"][0]["members"][0]["score_used"] is None and rg["rank_scopes_used"] == []
        # opening only: the per-cell and the cell-window groups are the same windows with the same clouds -> equal scores and
        # equal probabilities under the construction-wide scope (each computed on its own)
        pc, pcw = idx("per_cell", "opening"), idx("per_cell_window", "opening")
        assert [res.groups[i].cell for i in pc] == [res.groups[i].cell for i in pcw] == list(CELLS)
        assert all(res.groups[i].equivalent_to == "per_cell_window[opening]" for i in pc)
        assert np.array_equal(compat.score[pc], compat.score[pcw]) and all(compat.n_references[i] == 3 for i in pc + pcw)
        _, _, r1 = ndli.representatives_from_spec(_spec(arc, _rank("per_cell", window_set="opening")), arc, TABLE, PLO, PHI, target=TARGET)
        _, _, r2 = ndli.representatives_from_spec(_spec(arc, _rank("per_cell_window", window_set="opening")), arc, TABLE, PLO, PHI, target=TARGET)
        assert [m["probability"] for m in r1["components"][0]["members"]] == [m["probability"] for m in r2["components"][0]["members"]]
        # availability is stated per grouping under the archive's scope
        avail = ndli.component_weighting_availability(arc)
        assert avail[("FAB", "all", "per_window_position")].startswith('rank space all (a, b, c): available under rank_scope "construction"')
        assert avail[("FAB", "all", "per_cell_window")].startswith('rank space all (a, b, c): available under rank_scope "construction", compatibility scores computed for 12/12')
        assert "unavailable under rank_scope \"construction\": 1/1 members without a computed score (insufficient_peers" in avail[("FAB", "opening", "per_window_position")]
        assert avail[("FAB", "all", "global")] == "not applicable to this grouping (uniform only)"
        arw, *_ = _archive(Path(tmp) / "wc", coll, {"all": None}, scope=WC)
        assert "scores no per_window_position representatives; uniform only" in ndli.component_weighting_availability(arw)[("FAB", "all", "per_window_position")]
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arw, _rank("per_window_position")), arw, TABLE, PLO, PHI, target=TARGET),
                "carries no rank score for grouping 'per_window_position'")
    print("  PASS position ranks over the complete population; singleton and dataset handling; opening per-cell = cell-window")


def test_a_whole_cell_may_carry_zero_mass_and_an_all_zero_component_is_refused():
    rng = np.random.default_rng(104)
    # cell 20 sits far from every other cell and its windows far from one another: every one of its SGMs lies beyond all the
    # draws of every reference, so all its scores are zero
    coll, ids = _population(rng, far_cell=20, far_step=0.3)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, res, compat, _ = _archive(tmp / "cw", coll, {"all": None})
        _, p, rec = ndli.representatives_from_spec(_spec(arc, _rank("per_cell_window")), arc, TABLE, PLO, PHI, target=TARGET)
        members = rec["components"][0]["members"]
        far = [m for m in members if m["cell"] == 20]
        assert len(far) == 3 and all(m["score_used"] == 0.0 and m["probability"] == 0.0 for m in far)
        assert all(m["probability"] > 0 for m in members if m["cell"] != 20) and abs(p.sum() - 1.0) < 1e-12
        norm = rec["components"][0]["normalizers"]
        assert norm["zero_mass_cells"] == [20] and norm["cell_mass_by_cell"]["20"] == 0.0
        assert rec["n_zero_probability_rows"] == 3 and rec["components"][0]["n_zero_members"] == 3
        nu = ndli.build_nuisance_dli(_spec(arc, _rank("per_cell_window")), KEYS, PLO, PHI, table=TABLE, archive=arc, target=TARGET)
        d = nu.sample(4000, seed=5)
        zero_rows = {tuple(r) for r in nu.samples[nu.probabilities == 0]}
        assert len(zero_rows) == 3 and not any(tuple(x) in zero_rows for x in d)   # whole stored vectors; a zero row is never drawn
        # the probability grid draws a zero-total cell without dividing by it
        from srm_and_sbi_monomer_dimer_alp.visualization_aggregation import figure_member_probability_grid
        P = np.asarray([[next(m["probability"] for m in members if m["cell"] == c and m["chunk"] == w) for w in range(3)] for c in CELLS])
        fig = figure_member_probability_grid(list(CELLS), [0.0, 2.0, 4.0], P, np.full(P.shape, 1 / 12), title="t", subtitle="s", window_seconds=2.0)
        texts = [t.get_text() for t in fig.legends[0].get_texts()]
        assert any("zero probability (3" in t for t in texts) and any("cell total exactly zero (1)" in t for t in texts)
        assert fig.axes[0].get_title().startswith("stored sampling probability of each row")
        # the same data under the hierarchical scope: the far cell cannot be normalized there (its mass is reserved), so that
        # policy refuses; under the construction-wide scope a zero-mass cell is a valid outcome
        arw, *_ = _archive(tmp / "wc", coll, {"all": None}, scope=WC)
        _expect(ValueError, lambda: ndli.representatives_from_spec(
            _spec(arw, _comp("per_cell_window", 1.0, member_weighting="posterior_rank_median", rank_scope=WC)), arw, TABLE, PLO, PHI, target=TARGET),
            "cell 20: every selected-window score is zero")
        # an all-zero required component is refused: no floor, no uniform fallback, no clipping
        coll0, _ = _population(rng, all_far=True)
        arc0, res0, compat0, _ = _archive(tmp / "zero", coll0, {"all": None})
        cw = [i for i, g in enumerate(res0.groups) if g.construction == "per_cell_window"]
        assert all(compat0.status[i] == "computed" and compat0.score[i] == 0.0 for i in cw)
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc0, _rank("per_cell_window")), arc0, TABLE, PLO, PHI, target=TARGET),
                "every score of this component is zero", "no uniform fallback, no floor")
        text = ndli.component_weighting_availability(arc0)[("FAB", "all", "per_cell_window")]
        assert 'unavailable under rank_scope "construction": compatibility scores computed for 12/12 members, but the policy cannot be normalized' in text
        # a zero-MASS component (alpha = 0) with all-zero scores is still refused as a rank component; uniform at mass zero is valid
        _, pz, rz = ndli.representatives_from_spec(_spec(arc, _rank("per_cell_window", 1.0), _comp("per_cell", 0.0)), arc, TABLE, PLO, PHI, target=TARGET)
        assert rz["components"][1]["n_positive_members"] == 0 and abs(pz.sum() - 1.0) < 1e-12
    print("  PASS a zero-mass cell is valid and never drawn; an all-zero component is refused; the grid handles a zero total")


def test_archive_scope_records_are_validated_and_legacy_archives_keep_their_recorded_meaning():
    rng = np.random.default_rng(105)
    coll, ids = _population(rng)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, res, compat, paths = _archive(tmp / "cw", coll, {"all": None, "opening": (0,)}, spaces={"ab": ("a", "b")})
        man = arc["manifest"]; sec = man["compatibility"]
        assert arc["format_version"] == agg.ARCHIVE_FORMAT_VERSION == 4 and sec["comparison_scope"] == CW
        assert sec["comparison_scopes"] == {"per_cell": CW, "per_window_position": CW, "per_cell_window": CW}
        assert sec["protocol"]["name"] == "sgm_distance_rank_v2" and sec["protocol"]["comparison_scope"] == CW
        assert sec["rank_spaces"]["ab"]["protocol"]["comparison_scope"] == CW and all(b["scope"] == CW and b["cell"] == -1 for b in sec["blocks"])
        assert "every other cell–window group" in sec["protocol"]["reference_populations"]["per_cell_window"]
        csv_head = paths["compatibility_csv"].read_text().splitlines()
        assert csv_head[0].endswith(",rank,reference_scope") and csv_head[1].endswith(",construction")
        assert len(csv_head) - 1 == sum(b.group_index.size * (b.group_index.size - 1) for b in compat.blocks)
        comp = agg.archive_component(arc, kind="FAB", window_set="all", construction="per_cell_window")
        assert comp["comparison_scope"] == CW and comp["archive_comparison_scope"] == CW and comp["rank_spaces"] == {"all": ["a", "b", "c"], "ab": ["a", "b"]}
        assert agg.archive_component(arc, kind="FAB", window_set="all", construction="global")["comparison_scope"] is None

        def rewrite(name, mutate, source=None):
            with np.load(source or paths["archive"], allow_pickle=False) as z:
                arrs = {k: z[k] for k in z.files}
            m = json.loads(str(arrs.pop("manifest_json")))
            mutate(arrs, m)
            np.savez_compressed(tmp / name, **arrs, manifest_json=np.asarray(json.dumps(m, sort_keys=True, default=str)))
            return tmp / name
        pop = next(b for b in sec["blocks"] if b["construction"] == "per_cell_window" and b["window_set"] == "all")
        pos = next(b for b in sec["blocks"] if b["construction"] == "per_window_position" and b["window_set"] == "all")
        names = pop["arrays"]
        def block_of(m, index):
            return next(b for b in m["compatibility"]["blocks"] if b["index"] == index)
        def scope_only(arrs, m):
            m["compatibility"]["comparison_scope"] = WC
        def scope_and_map(arrs, m):
            m["compatibility"]["comparison_scope"] = WC; m["compatibility"]["comparison_scopes"] = dict(agg.SCOPES_BY_CONSTRUCTION[WC])
        def no_scope(arrs, m):
            m["compatibility"].pop("comparison_scope")
        def block_scope(arrs, m):
            block_of(m, pop["index"])["scope"] = WC
        def block_cell(arrs, m):
            block_of(m, pop["index"])["cell"] = 3
        def split(arrs, m):
            # the cell-window population cut down to one cell's three windows (a consistent sub-block): refused
            gi = arrs[names["group_index"]]; keep = [k for k, i in enumerate(gi) if arrs["cell"][i] == 3]
            arrs[names["group_index"]] = gi[keep]; arrs[names["ranks"]] = arrs[names["ranks"]][np.ix_(keep, keep)]
            arrs[names["reference_draws"]] = arrs[names["reference_draws"]][keep]
        def duplicate_pair(arrs, m):
            gi = arrs[names["group_index"]]; arrs["cell"][gi[1]] = arrs["cell"][gi[0]]; arrs["chunk"][gi[1]] = arrs["chunk"][gi[0]]
        def repeated_cell_is_fine(arrs, m):
            pass
        def position_with_cell(arrs, m):
            gi = arrs[pos["arrays"]["group_index"]]; arrs["cell"][gi[0]] = 5
        def legacy_name(arrs, m):
            m["compatibility"]["protocol"]["name"] = agg.LEGACY_COMPATIBILITY_PROTOCOL
        def space_scope(arrs, m):
            m["compatibility"]["rank_spaces"]["ab"]["protocol"]["comparison_scope"] = WC
        def stale(arrs, m):
            s = arrs["compatibility_score"].copy(); i = int(arrs[names["group_index"]][0]); s[i] = s[i] + 0.25 if s[i] < 0.5 else s[i] - 0.25
            arrs["compatibility_score"] = s
        def missing_population(arrs, m):
            m["compatibility"]["blocks"] = [b for b in m["compatibility"]["blocks"] if b["index"] != pop["index"]]
        for name, mut, needle in (("scope", scope_only, "is not the map of scope"), ("scopemap", scope_and_map, "every space of an archive shares one comparison scope"),
                                  ("noscope", no_scope, "declares compatibility.comparison_scope"), ("blockscope", block_scope, "declares reference scope"),
                                  ("blockcell", block_cell, "spans its whole population (cell -1)"), ("split", split, "complete peer population"),
                                  ("duppair", duplicate_pair, "(cell, window position) pair"), ("poscell", position_with_cell, "a pooled position group carries cell -1"),
                                  ("legacyname", legacy_name, "the protocol of archive format 4"), ("spacescope", space_scope, "shares one comparison scope"),
                                  ("stale", stale, "stored score"), ("missing", missing_population, "no block"),
                                  # a construction-wide archive relabeled as a legacy format: its blocks are not that format's contexts
                                  ("aslegacy", lambda arrs, m: _as_legacy(m, 3), "carries no rank score under comparison scope 'within_cell'")):
            _expect(agg.AggregationError, lambda name=name, mut=mut: agg.load_candidate_archive(rewrite(f"{name}.npz", mut)), needle)
        # in a cell-window population the cell identity repeats legitimately; the unmodified archive reads back
        assert agg.load_candidate_archive(rewrite("same.npz", repeated_cell_is_fine))["compatibility"]["comparison_scope"] == CW
        # a legacy archive (formats 2 and 3: protocol v1, no scope field) keeps its recorded within-cell meaning: it reads back,
        # supplies the hierarchical policy under that name, and can never supply construction-wide cell-window scores
        arw, resw, within, pw = _archive(tmp / "wc", coll, {"all": None, "opening": (0,)}, scope=WC, spaces={"ab": ("a", "b")})
        v3 = agg.load_candidate_archive(rewrite("v3.npz", lambda arrs, m: _as_legacy(m, 3), source=pw["archive"]))
        assert v3["format_version"] == 3 and v3["compatibility"]["comparison_scope"] == WC and v3["compatibility"]["protocol"]["name"] == "sgm_distance_rank_v1"
        hier = _comp("per_cell_window", 1.0, member_weighting="posterior_rank_median", rank_scope=WC, rank_coordinates=["a", "b"])
        _, p3, r3 = ndli.representatives_from_spec(_spec(v3, hier), v3, TABLE, PLO, PHI, target=TARGET)
        _, p4, _ = ndli.representatives_from_spec(_spec(arw, hier), arw, TABLE, PLO, PHI, target=TARGET)
        assert np.array_equal(p3, p4) and r3["components"][0]["score_source"]["protocol"] == "sgm_distance_rank_v1" and r3["components"][0]["rank_space"] == "ab"
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(v3, _rank("per_cell_window")), v3, TABLE, PLO, PHI, target=TARGET),
                "is not the scope this archive's per_cell_window scores were computed under", "archive format 3, protocol sgm_distance_rank_v1")
        _expect(agg.AggregationError, lambda: agg.load_candidate_archive(
            rewrite("v3scope.npz", lambda arrs, m: _as_legacy(m, 3)["compatibility"].update(comparison_scope=WC), source=pw["archive"])),
            "a format-3 archive carries no comparison scope field")
        # the earlier guards hold under the construction-wide scope: coordinate space, checksum, target, statistic
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, _rank("per_cell_window", rank_coordinates=["b", "c"])), arc, TABLE, PLO, PHI,
                                                                   target=TARGET), "no rank coordinate space of the archive runs over ['b', 'c']")
        bad = _spec(arc, _rank("per_cell_window")); bad["block"]["candidate_archive_sha256"] = "0" * 64
        _expect(ValueError, lambda: ndli.representatives_from_spec(bad, arc, TABLE, PLO, PHI, target=TARGET), "checksum")
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, _rank("per_cell_window")), arc, TABLE, PLO, PHI,
                                                                   target={**TARGET, "condition": "INLB"}), "target condition")
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, _rank("per_cell_window", statistic="median")), arc, TABLE, PLO, PHI,
                                                                   target=TARGET), 'requires statistic = "sgm"')
        # the four-coordinate style subset space weights with ITS scores, over the same complete population
        _, pab, rab = ndli.representatives_from_spec(_spec(arc, _rank("per_cell_window", rank_coordinates=["a", "b"])), arc, TABLE, PLO, PHI, target=TARGET)
        sab = np.asarray([m["score_used"] for m in rab["components"][0]["members"]])
        assert rab["components"][0]["rank_space"] == "ab" and rab["components"][0]["normalizers"]["references_per_member"] == [11]
        assert np.allclose([m["probability"] for m in rab["components"][0]["members"]], sab / sab.sum(), rtol=0, atol=1e-15)
    print("  PASS format-4 scope records, thirteen malformed-scope refusals, legacy read-back under its recorded meaning")


def test_explicit_component_masses_duplicates_overrides_and_whole_vector_sampling_keep_their_behavior():
    rng = np.random.default_rng(106)
    coll, ids = _population(rng)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arc, res, compat, _ = _archive(tmp, coll, {"all": None, "opening": (0,)})
        # three constructions of different sizes mixed by EXPLICIT masses: each is normalized inside itself, then multiplied by
        # its mass; 12, 4 and 1 rows never set the mixture by their counts
        spec = _spec(arc, _rank("per_cell_window", 0.5), _rank("per_cell", 0.3), _comp("global", 0.2))
        vec, p, rec = ndli.representatives_from_spec(spec, arc, TABLE, PLO, PHI, target=TARGET)
        masses = [sum(m["probability"] for m in c["members"]) for c in rec["components"]]
        assert np.allclose(masses, [0.5, 0.3, 0.2], atol=1e-12) and abs(p.sum() - 1.0) < 1e-12
        assert [c["member_weighting"] for c in rec["components"]] == ["posterior_rank_median", "posterior_rank_median", "uniform"]
        assert [c["rank_scope"] for c in rec["components"]] == [CW, CW, None] and rec["rank_scopes_used"] == [CW]
        for c, n in zip(rec["components"][:2], (12, 4)):
            s = np.asarray([m["score_used"] for m in c["members"]])
            assert len(s) == n and np.allclose([m["probability"] for m in c["members"]], c["weight"] * s / s.sum(), rtol=0, atol=1e-15)
        assert rec["components"][2]["members"][0]["probability"] == 0.2
        _expect(ValueError, lambda: ndli.representatives_from_spec(_spec(arc, _rank("per_cell_window", 0.5), _rank("per_cell", 0.3)), arc, TABLE, PLO, PHI,
                                                                   target=TARGET), "sum to 0.8")
        # duplicate memberships: the opening per-cell and cell-window groups are the same vectors -> consolidated, contributions added
        dup = _spec(arc, _rank("per_cell", 0.5, window_set="opening"), _rank("per_cell_window", 0.5, window_set="opening"))
        v2, p2, r2 = ndli.representatives_from_spec(dup, arc, TABLE, PLO, PHI, target=TARGET)
        assert v2.shape == (4, 3) and len(r2["consolidated"]) == 4 and r2["n_member_rows"] == 8 and abs(p2.sum() - 1.0) < 1e-12
        for cns in r2["consolidated"]:
            a, b = (m["contribution"] for m in cns["memberships"])
            assert a == b and abs(cns["probability"] - 2 * a) < 1e-15           # equal scores, equal masses
        # overrides: applied to every vector, scores not recomputed, the equal-member baseline aligned to the same rows
        ov = _spec(arc, _rank("per_cell_window", rank_coordinates=None), overrides={"c": {"value": 0.25, "reason": "declared"}})
        v3, p3, r3 = ndli.representatives_from_spec(ov, arc, TABLE, PLO, PHI, target=TARGET)
        assert np.all(v3[:, 2] == 0.25) and not r3["is_sampled_rows"] and r3["rank_coordinates_disjoint_from_overrides"] is False
        base = r3["equal_member_baseline"]
        assert np.allclose(base["probabilities"], 1 / 12) and abs(base["effective_rows"] - 12.0) < 1e-9 and r3["effective_rows"] < 12.0 - 1e-6
        # the artifact carries the scope; whole stored vectors are drawn with the stored probabilities
        nu = ndli.build_nuisance_dli(ov, KEYS, PLO, PHI, table=TABLE, archive=arc, target=TARGET)
        art = tmp / "A.npz"; nu.flush(art); back = ndli.NuisanceDLI.load(art)
        assert back.selection["rank_scopes_used"] == [CW] and back.selection["components"][0]["rank_scope"] == CW
        assert np.array_equal(back.probabilities, nu.probabilities) and np.array_equal(back.samples, nu.samples)
        d = back.sample(20000, seed=3)
        stored = {tuple(r): k for k, r in enumerate(back.samples)}
        counts = np.bincount([stored[tuple(x)] for x in d], minlength=len(stored))  # KeyError if a draw is not a stored vector
        assert np.allclose(counts / d.shape[0], back.probabilities, atol=0.012)
        # a record written before the scope existed still loads and reports as the hierarchical policy of its time
        legacy = copy.deepcopy(nu.selection)
        for c in legacy["components"]:
            for k in ("rank_scope", "reference_population"):
                c.pop(k, None)
        legacy.pop("rank_scopes_used")
        old = ndli.NuisanceDLI.from_representatives(KEYS, nu.samples, nu.probabilities, legacy, prior_low=PLO, prior_high=PHI)
        old.flush(tmp / "L.npz"); assert np.array_equal(ndli.NuisanceDLI.load(tmp / "L.npz").probabilities, nu.probabilities)
        if str(REPO / "Script_Bank" / "Analysis") not in sys.path:
            sys.path.insert(0, str(REPO / "Script_Bank" / "Analysis"))
        import SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI as companion
        assert companion._scope_of(legacy["components"][0]) == WC and companion._scope_of(nu.selection["components"][0]) == CW
        assert companion._scope_of({"grouping": "per_cell"}) == CW
    print("  PASS explicit masses, consolidation, overrides, baseline, artifact round trip, whole-vector sampling, legacy record")


def test_weighted_histograms_are_exact_deterministic_and_independent_of_any_draw():
    # hand-countable: values on a grid, weights that are exact binary fractions
    v = np.array([0.0, 0.1, 0.1, 0.25, 0.5, 0.5, 1.0]); w = np.array([0.125, 0.25, 0.0, 0.125, 0.25, 0.125, 0.125])
    edges = np.array([0.0, 0.25, 0.5, 1.0])
    h = agg.weighted_histogram(v, w, edges)
    assert h.tolist() == [0.375, 0.125, 0.5] and h.sum() == 1.0                  # [0, .25): .125+.25+0; [.25, .5): .125; [.5, 1]: .25+.125+.125
    assert agg.weighted_histogram(v, np.zeros(7), edges).tolist() == [0.0, 0.0, 0.0]
    _expect(agg.AggregationError, lambda: agg.weighted_histogram(v, w, np.array([0.0, 0.5])), "outside the bins")      # never dropped silently
    _expect(agg.AggregationError, lambda: agg.weighted_histogram(v, -w, edges), "non-negative")
    _expect(agg.AggregationError, lambda: agg.weighted_histogram(v, w[:3], edges), "one weight per value")
    # complete vectors: every coordinate inherits the SAME probabilities; a fixed coordinate is a point mass; a zero-weight
    # vector contributes nothing but belongs to the support (the equal-member baseline weights it)
    rng = np.random.default_rng(107)
    V = np.column_stack([rng.normal(size=50), rng.uniform(size=50), np.full(50, -1.3)])
    p = rng.uniform(size=50); p[:5] = 0.0; p[7] = 0.0; p = p / p.sum(); b = np.full(50, 1 / 50)
    V[0, 0] = V[:, 0].min() - 1.0                                                  # a zero-probability vector sets the lower edge
    pans = agg.marginal_histograms(V, p, b, ["x", "y", "z"], notes={"z": "fixed by an explicit override"})
    assert [q["kind"] for q in pans] == ["bins", "bins", "point_mass"]
    for j, q in enumerate(pans[:2]):
        assert len(q["edges"]) == agg.MARGINAL_HISTOGRAM_BINS + 1 == 31 and q["edges"][0] == V[:, j].min() and q["edges"][-1] == V[:, j].max()
        assert abs(q["selected"].sum() - 1.0) < 1e-12 and abs(q["baseline"].sum() - 1.0) < 1e-12
        # bin by bin: the sum of the probabilities of the vectors in the bin (the last bin closed)
        for k in range(30):
            lo, hi = q["edges"][k], q["edges"][k + 1]
            inside = (V[:, j] >= lo) & ((V[:, j] < hi) if k < 29 else (V[:, j] <= hi))
            assert abs(q["selected"][k] - p[inside].sum()) < 1e-15 and abs(q["baseline"][k] - b[inside].sum()) < 1e-15
    assert pans[0]["selected"][0] == 0.0 and pans[0]["baseline"][0] == 1 / 50      # the zero-probability vector's bin: no selected mass
    z = pans[2]
    assert z["value"] == -1.3 and abs(z["selected_mass"] - 1.0) < 1e-12 and abs(z["baseline_mass"] - 1.0) < 1e-12 and z["note"] == "fixed by an explicit override"
    # no random state enters: identical results whatever the global RNG state, and no draw count is involved
    np.random.seed(1); again1 = agg.marginal_histograms(V, p, b, ["x", "y", "z"])
    np.random.seed(999); np.random.random(1000); again2 = agg.marginal_histograms(V, p, b, ["x", "y", "z"])
    for q1, q2, q0 in zip(again1, again2, pans):
        for k in ("edges", "selected", "baseline"):
            if k in q0:
                assert np.array_equal(q1[k], q2[k]) and np.array_equal(q1[k], q0[k])
    # the exact weighted CDF evaluated at the bin edges equals the cumulated bin masses (the CDF path stays exact)
    q = pans[1]
    cdf = np.asarray([p[V[:, 1] < e].sum() for e in q["edges"][1:-1]] + [p.sum()])
    assert np.allclose(np.cumsum(q["selected"]), cdf, rtol=0, atol=1e-12)
    # the figure draws from the panels alone
    import matplotlib
    matplotlib.use("Agg")
    from srm_and_sbi_monomer_dimer_alp.visualization_aggregation import figure_weighted_histograms
    from srm_and_sbi_monomer_dimer_alp.visualization_aggregation import HISTOGRAM_VIEW_MARGIN
    # a narrow distribution far inside a wide prior (key a: log10 values in [-0.52, -0.50], prior [-1, 0]), a distribution that
    # contains its prior bounds (key b), and a fixed coordinate (key c)
    Vn = np.column_stack([-0.52 + 0.02 * V[:, 1], -0.3 + 1.6 * V[:, 1], np.full(50, 0.4)])
    pans_n = agg.marginal_histograms(Vn, p, b, KEYS)
    fig = figure_weighted_histograms(pans_n, TABLE, title="t", subtitle="s", reference=np.array([-0.9, 0.5, 0.4]), reference_name="ref")
    assert all(ax.get_ylabel() == "probability mass per bin" for ax in fig.axes[:3])
    ax_a, ax_b, ax_c = fig.axes[:3]
    # the view is the full stored support plus the margin: nothing cropped, and the prior bounds do not stretch it
    ea = pans_n[0]["edges"]; pad = HISTOGRAM_VIEW_MARGIN * (ea[-1] - ea[0])
    assert np.allclose(ax_a.get_xlim(), [10 ** (ea[0] - pad), 10 ** (ea[-1] + pad)], rtol=1e-9) and ax_a.get_xscale() == "log"
    assert ax_a.get_xlim()[0] > 10 ** -1.0 * 2.5 and ax_a.get_xlim()[1] < 10 ** 0.0 / 2.5          # the prior [0.1, 1] is far outside this view
    texts_a = [t.get_text() for t in ax_a.texts]
    assert "\u2190 prior bound 0.1" in texts_a and "prior bound 1 \u2192" in texts_a and "\u2190 reference 0.1259" in texts_a
    assert len(ax_a.lines) == 0                                                                  # no line was drawn for an off-view annotation
    eb = pans_n[1]["edges"]; padb = HISTOGRAM_VIEW_MARGIN * (eb[-1] - eb[0])
    assert np.allclose(ax_b.get_xlim(), [eb[0] - padb, eb[-1] + padb]) and ax_b.get_xscale() == "linear"
    assert len(ax_b.lines) == 3 and not ax_b.texts                                               # both bounds and the reference lie inside: lines
    assert ax_c.get_xlim()[0] <= 0.0 and ax_c.get_xlim()[1] >= 1.0                                 # a point mass is shown within its prior range
    assert any("point mass 1 at 0.4" in t.get_text() for t in ax_c.texts) and all(t.get_transform() is ax_c.transAxes for t in ax_c.texts)
    # the bars drawn are the exported masses over the exported edges
    bars = [r for r in ax_a.patches if hasattr(r, "get_height")]
    assert np.allclose([r.get_height() for r in bars], pans_n[0]["selected"], rtol=0, atol=0) and len(bars) == 30
    assert len(fig.legends[0].get_texts()) == 4
    print("  PASS exact bin masses, shared deterministic edges, point masses, zero weights, no dependence on any random state")


def test_the_companion_report_plots_the_stored_probabilities_exactly_and_reports_cell_totals():
    import matplotlib
    matplotlib.use("Agg")
    from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
    if str(REPO / "Script_Bank" / "Analysis") not in sys.path:
        sys.path.insert(0, str(REPO / "Script_Bank" / "Analysis"))
    import SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI as companion
    rng = np.random.default_rng(108)
    table = det.DETECTOR_PARAMETERIZATION; keys = [e["KEY"] for e in table]
    lo = np.array([e["PRIOR_RANGE"][0] for e in table], float); hi = np.array([e["PRIOR_RANGE"][1] for e in table], float)
    n_cells, n_windows = 4, 3
    ids = [(0, c, w) for c in range(n_cells) for w in range(n_windows)]
    # cell 3 far from the others and its windows far apart (all its scores zero): a zero-mass cell in the report
    cloud = np.stack([(lo + hi) / 2 + 0.004 * (hi - lo) * c + 0.002 * (hi - lo) * w + (0.3 * (hi - lo) * (1 + w) if c == 3 else 0.0)
                      + rng.normal(scale=0.0005 if c == 3 else 0.01, size=(40, 6)) * (hi - lo) for (_, c, w) in ids])
    coll = _collection(cloud, ids, geometry=_geometry(n_windows), scale=tuple(hi - lo), keys=keys)
    R = companion._resolve(2.0, "FAB")
    target = companion._target(SimpleNamespace(condition="FAB"), R)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        label = f"{R['paths'].project_alias}_2S_50FPS_TEST"
        arc, res, compat, _ = _archive(tmp / "arc", coll, {"all": None}, spaces={"spatial_brightness": ("mu_r", "sigma_r", "mu_pc", "sigma_pc")},
                                       table=table, target=target, product_label=label)
        spec = {"block": {"posterior_sample_pool_choice": "empirical_representatives", "candidate_archive": arc["path"],
                          "candidate_archive_sha256": arc["sha256"], "rationale": "test"},
                "component": [{"source_experiment": label, "condition": "FAB", "window_set": "all", "grouping": "per_cell_window", "statistic": "sgm",
                               "weight": 1.0, "member_weighting": "posterior_rank_median", "rank_scope": "construction",
                               "rank_coordinates": ["mu_r", "sigma_r", "mu_pc", "sigma_pc"]}],
                "override": {"prob_photo_bleach": {"value": 0.05, "reason": "test"}, "lambda_rate": {"value": 5.1, "reason": "test"}}}
        nu = ndli.build_nuisance_dli(spec, R["imaging_keys"], R["plo"], R["phi"], table=table, archive=arc, target=target)
        assert int(np.sum(nu.probabilities == 0)) == 3 and nu.selection["components"][0]["normalizers"]["zero_mass_cells"] == [3]
        art = tmp / "A.npz"; nu.flush(art)

        def run(seed):
            # the global numpy state and the sampler's own draws differ between the two runs
            np.random.seed(seed); np.random.random(seed)
            out = companion._write_nuisance_report(nu, R, art, n_draws=500 + seed, archive=arc)
            text = (out / "report.md").read_text()
            hist = (out / "nuisance_marginal_histograms.csv").read_text()
            table_lines = [ln for ln in text.splitlines() if ln.startswith("| ") and any(f"| {k} |" in ln or ln.startswith(f"| {k} |") for k in keys)]
            return text, hist, table_lines
        text, hist, lines1 = run(11)
        text2, hist2, lines2 = run(700)
        assert hist == hist2 and lines1 == lines2 and len(lines1) >= 12             # the exact tables and bin masses do not move with a seed or a draw count
        # the histogram file: per parameter the selected and the equal-member masses add up to one; fixed coordinates are point masses
        rows = [ln.split(",") for ln in hist.splitlines()[1:]]
        for k in keys:
            sel = sum(float(r[7]) for r in rows if r[0] == k); base = sum(float(r[8]) for r in rows if r[0] == k)
            assert abs(sel - 1.0) < 1e-12 and abs(base - 1.0) < 1e-12, k
        kinds = {k: {r[1] for r in rows if r[0] == k} for k in keys}
        assert kinds["prob_photo_bleach"] == {"point_mass"} and kinds["lambda_rate"] == {"point_mass"} and kinds["mu_r"] == {"bin"}
        pm = next(r for r in rows if r[0] == "lambda_rate")
        assert abs(float(pm[5]) - 5.1) < 1e-12 and sum(1 for r in rows if r[0] == "mu_r") == agg.MARGINAL_HISTOGRAM_BINS
        # bin masses recomputed from the stored vectors and probabilities
        j = keys.index("mu_pc"); col = nu.samples[:, j]
        for r in [r for r in rows if r[0] == "mu_pc"]:
            a, b, k = float(r[3]), float(r[4]), int(r[2])
            inside = (col >= a) & ((col < b) if k < agg.MARGINAL_HISTOGRAM_BINS - 1 else (col <= b))
            assert abs(float(r[7]) - nu.probabilities[inside].sum()) < 1e-15
        # the report says what it shows: exact masses, physical units, the scope, global row probabilities, free cell totals
        assert "exact: 12 stored vectors with their stored probabilities (no resampling)" in text and "sampler-validation draws" in text
        assert "sampler_draws_are_stored_positive_probability_rows | PASS" in text
        assert "Per-parameter marginal summary (log10; exact)" in text and "No draw enters this table" in text
        assert "probability mass per bin" in text and "nothing is resampled" in text and "sampled from the artifact" not in text
        assert "projections of one joint distribution" in text and "fixed by explicit overrides" in text
        assert "rank scope `construction`" in text and "every other cell–window group" in text and "weight * score / sum of the component's scores" in text
        assert "the stored sampling probability of every row itself" in text and "no cell has a reserved" in text
        assert "empirical_representatives: cell totals of the stored probabilities" in text and "| 1 (3) |" in text
        out_dir = art.with_name(art.stem + "_Analysis")
        totals = [ln.split(",") for ln in (out_dir / "representatives_cell_totals.csv").read_text().splitlines()[1:]]
        assert [t[0] for t in totals] == ["0", "1", "2", "3"] and float(totals[3][1]) == 0.0 and totals[3][4] == "3"
        assert abs(sum(float(t[1]) for t in totals) - 1.0) < 1e-12 and len({t[1] for t in totals}) == 4      # unequal cell totals
        for f in ("nuisance_marginals", "representatives_probabilities", "representatives_marginals"):
            assert (out_dir / "figures" / f"{f}.png").exists(), f
        members = (out_dir / "representatives_members.csv").read_text().splitlines()
        assert members[0].endswith(",rank_scope") and all(ln.endswith(",construction") for ln in members[1:])
        # the dry-run console names the scope
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            companion._build_representatives(SimpleNamespace(condition="FAB", dry_run=True), R, spec["block"], spec, tmp / "x.npz", tmp / "s.toml")
        assert "rank scope construction; rank space spatial_brightness (mu_r, sigma_r, mu_pc, sigma_pc)" in buf.getvalue()
    print("  PASS the companion report: exact marginals, seed-independent tables and bins, global probabilities, a zero-total cell")


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
