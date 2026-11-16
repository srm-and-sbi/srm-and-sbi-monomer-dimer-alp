"""The two-sample and per-window measures of the embedding-distance analysis, on constructed data.

No estimator and no GPU: the functions under test take embedding arrays. Checked here: the AUC is the
pair-counting definition; the repaired classifier test has its no-information value at 0.5 when one set
is fifty times the other, and its relabelings keep the number of units and of samples on each side; a
real shift is found; the synthetic blocks have the recordings' sizes; the nearest-neighbor typicality is
the brute-force definition, is uniformly distributed for a query drawn like the reference, and flags a
query outside the cloud (in the standardized reading also along a low-variance coordinate). The runner's
part after the embedding pass (comparisons, per-window tables, report, stored embeddings) is run on
constructed embeddings.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from srm_and_sbi_monomer_dimer_alp import embedding_space_distance as esd
from srm_and_sbi_monomer_dimer_alp import embedding_space_distance_runner as runner
from srm_and_sbi_monomer_dimer_alp.experiment_support import CONDITION_DISPLAY


def _recordings(n_recordings, windows, dim, rng, shift=0.0, recording_sd=0.0):
    """Windows of ``n_recordings`` recordings: a recording offset (sd ``recording_sd``) plus window noise."""
    offsets = rng.normal(0.0, recording_sd, size=(n_recordings, dim)) if recording_sd else np.zeros((n_recordings, dim))
    X = np.vstack([offsets[r] + rng.normal(size=(windows, dim)) for r in range(n_recordings)]) + shift
    cells = np.repeat(np.arange(n_recordings), windows)
    return X, cells


def test_auc_is_the_pair_counting_definition():
    rng = np.random.default_rng(0)
    scores = np.round(rng.normal(size=60), 1)                 # rounding makes ties
    labels = rng.integers(0, 2, size=60)
    pos, neg = scores[labels == 1], scores[labels == 0]
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    assert abs(esd._auc(scores, labels) - wins / (len(pos) * len(neg))) < 1e-12
    assert np.isnan(esd._auc(scores, np.ones(60, dtype=int)))  # one class only: undefined, not 0.5


def test_matched_blocks_repeat_the_recording_sizes():
    labels = esd.matched_blocks([4, 4, 3], 25, np.random.default_rng(1))
    sizes = np.bincount(labels)
    assert labels.shape == (25,) and sizes.sum() == 25
    # 4, 4, 3 cycled: 4 4 3 4 4 3 = 22 samples, then the next 4 does not fit -> three single blocks
    assert sizes.tolist() == [4, 4, 3, 4, 4, 3, 1, 1, 1], sizes.tolist()
    same = esd.matched_blocks([4], 40, np.random.default_rng(2))
    assert np.bincount(same).tolist() == [4] * 10


def test_c2st_null_sits_at_half_when_one_set_is_fifty_times_the_other():
    rng = np.random.default_rng(3)
    X, cells = _recordings(10, 4, 6, rng)                     # 40 windows of 10 recordings
    Y = rng.normal(size=(2000, 6))                            # the same distribution, 50 times as many
    blocks = esd.matched_blocks([4] * 10, len(Y), np.random.default_rng(4))
    res = esd.c2st(X, Y, cells_x=cells, cells_y=blocks, seed=0, n_permutations=60)
    # The accuracy of the superseded test would sit at the share of the larger set (0.98) here.
    assert 0.35 < res["auc"] < 0.65, res["auc"]
    assert res["null_range95"][0] <= res["auc"] <= res["null_range95"][1], (res["auc"], res["null_range95"])
    assert 0.4 < res["null_mean"] < 0.6 and res["p_value"] > 0.05, res
    assert res["n_units_x"] == 10 and res["n_units_y"] == 500 and res["n_exchangeable_units_x"] == 10
    assert len(res["oof_score"]) == 2040 and np.isfinite(res["oof_score"]).all()


def test_c2st_finds_a_shift_and_reports_the_smallest_p_value():
    rng = np.random.default_rng(5)
    X, cells = _recordings(10, 4, 6, rng, shift=3.0)
    Y = rng.normal(size=(800, 6))
    blocks = esd.matched_blocks([4] * 10, len(Y), np.random.default_rng(6))
    res = esd.c2st(X, Y, cells_x=cells, cells_y=blocks, seed=0, n_permutations=40)
    assert res["auc"] > 0.95, res["auc"]
    assert abs(res["p_value"] - 1.0 / 41.0) < 1e-12, res["p_value"]


def test_c2st_without_size_matched_units_gives_no_p_value():
    rng = np.random.default_rng(7)
    X, cells = _recordings(6, 4, 4, rng)
    res = esd.c2st(X, rng.normal(size=(200, 4)), cells_x=cells, cells_y=None, seed=0, n_permutations=20)
    # Recordings of four windows against single videos: nothing to exchange them with.
    assert res["n_exchangeable_units_x"] == 0 and np.isnan(res["p_value"]) and res["n_permutations"] == 0
    assert 0.0 <= res["auc"] <= 1.0


def test_c2st_relabelings_keep_units_and_samples_on_each_side():
    # Reproduce the function's relabeling rule and count: sizes 4 and 2 on the X side, matched on the Y side.
    rng = np.random.default_rng(8)
    unit_size = np.array([4, 4, 2, 4, 4, 2, 4, 2, 1, 1])
    unit_class = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    exchangeable = np.isin(unit_size, [4, 2])
    for _ in range(200):
        permuted = unit_class.copy()
        for size in (2, 4):
            members = np.flatnonzero(exchangeable & (unit_size == size))
            permuted[members] = unit_class[rng.permutation(members)]
        assert (permuted == 0).sum() == 3
        assert unit_size[permuted == 0].sum() == 10             # the X side keeps its 10 samples
        assert permuted[-2:].tolist() == [1, 1]                 # units without a partner stay


def _typicality_by_definition(R, q, k):
    """Brute force: scores inside the set R + {q}; share of reference scores >= the query's."""
    S = np.vstack([R, q[None, :]])
    d = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    score = np.sort(d, axis=1)[:, k - 1]
    return (1 + np.count_nonzero(score[:-1] >= score[-1])) / (len(R) + 1)


def test_knn_typicality_is_the_brute_force_definition():
    rng = np.random.default_rng(9)
    R = rng.normal(size=(60, 3))
    Q = np.vstack([rng.normal(size=(12, 3)), rng.normal(size=(4, 3)) * 4.0])
    for k in (1, 2, 5):
        res = esd.knn_typicality(R, Q, k=k, block=7)            # a small block exercises the blocking
        expected = [_typicality_by_definition(R, q, k) for q in Q]
        assert np.allclose(res["tail_share"], expected), (k, res["tail_share"], expected)
        loo = np.sqrt(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(loo, np.inf)
        assert np.allclose(res["reference_distance"], np.sort(loo, axis=1)[:, k - 1])
        direct = np.sort(np.sqrt(((Q[:, None, :] - R[None, :, :]) ** 2).sum(-1)), axis=1)
        assert np.allclose(res["query_distance"], direct[:, k - 1])
        assert res["neighbors"].shape == (16, k)


def test_knn_typicality_is_uniform_for_a_query_drawn_like_the_reference():
    rng = np.random.default_rng(10)
    values = []
    for _ in range(400):                                        # a fresh reference cloud each time
        R = rng.normal(size=(99, 4))
        values.append(esd.knn_typicality(R, rng.normal(size=(1, 4)), k=5)["tail_share"][0])
    values = np.array(values)
    assert abs(values.mean() - 0.505) < 0.045, values.mean()    # uniform on {0.01, ..., 1}: mean 0.505
    share = np.mean(values <= 0.05)                             # 400 draws at 5 %: sd 1.1 %
    assert 0.015 <= share <= 0.09, share


def test_knn_typicality_flags_a_query_outside_the_cloud():
    rng = np.random.default_rng(11)
    R = rng.normal(size=(500, 5))
    inside, outside = np.zeros((1, 5)), np.full((1, 5), 6.0)
    res = esd.knn_typicality(R, np.vstack([inside, outside]), k=10)
    assert res["tail_share"][0] > 0.5 and abs(res["tail_share"][1] - 1.0 / 501.0) < 1e-12, res["tail_share"]
    assert res["percentile"][0] < 50.0 and res["percentile"][1] == 100.0


def test_standardized_reading_sees_a_low_variance_coordinate():
    rng = np.random.default_rng(12)
    R = rng.normal(size=(400, 3)) * np.array([10.0, 10.0, 0.01])     # the third coordinate barely varies
    q = np.array([[0.0, 0.0, 0.5]])                                  # 50 sd out along it, central otherwise
    raw = esd.knn_typicality(R, q, k=10)["tail_share"][0]
    std = esd.knn_typicality(R, q, k=10, standardize=True)["tail_share"][0]
    assert raw > 0.2 and std < 0.01, (raw, std)


def test_pooled_scaling_gives_every_point_the_rank_it_has_in_the_set():
    """Under pooled scaling the score does not depend on which point is called the query: each point of
    a set, ranked against the others, gets the rank it has in the whole set."""
    rng = np.random.default_rng(21)
    Z = rng.standard_t(3, size=(31, 5)) * np.array([1.0, 5.0, 0.2, 3.0, 1.0])
    s = Z / Z.std(axis=0)
    d = np.sqrt(((s[:, None] - s[None]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    for k in (1, 3, 7):
        score = np.sort(d, axis=1)[:, k - 1]
        for i in range(len(Z)):
            res = esd.knn_typicality(np.delete(Z, i, 0), Z[i], k=k, standardize="pooled")
            assert abs(res["tail_share"][0] - np.mean(score >= score[i])) < 1e-12, (k, i)
            assert res["scaling"] == "pooled" and res["neighbors"].shape == (1, k)


def test_reference_scaling_flags_too_often_and_pooled_scaling_does_not():
    """A query drawn like a small heavy-tailed reference: the raw and the pooled rank flag at the
    stated rate, the reference-scaled rank about twice as often (the query had no part in the scale)."""
    rng = np.random.default_rng(22)
    n, reps, level = 20, 1500, 2.0 / 21.0
    rate = {}
    for name, std in (("raw", False), ("reference", True), ("pooled", "pooled")):
        rate[name] = []
    for _ in range(reps):
        Z = rng.standard_t(3, size=(n + 1, 8))
        for name, std in (("raw", False), ("reference", True), ("pooled", "pooled")):
            rate[name].append(esd.knn_typicality(Z[:n], Z[n], k=3, standardize=std)["tail_share"][0] <= level + 1e-12)
    rate = {name: float(np.mean(v)) for name, v in rate.items()}
    se = np.sqrt(level * (1 - level) / reps)
    assert abs(rate["raw"] - level) < 4 * se and abs(rate["pooled"] - level) < 4 * se, rate
    assert rate["reference"] > level + 4 * se, rate


def test_knn_refuses_an_unknown_scaling():
    rng = np.random.default_rng(23)
    try:
        esd.knn_typicality(rng.normal(size=(10, 2)), rng.normal(size=(1, 2)), k=2, standardize="symmetric")
    except ValueError:
        return
    raise AssertionError("an unknown scaling name was accepted")


def test_knn_refuses_k_beyond_the_cloud():
    R = np.zeros((5, 2))
    for bad in (0, 5):                                               # leave-one-out leaves 4 neighbors
        try:
            esd.knn_typicality(R, np.zeros((1, 2)), k=bad)
        except ValueError:
            continue
        raise AssertionError(f"k={bad} accepted for 5 reference points")


def _fake_groups(rng, dim=8, shift_second=0.0):
    """Two recording groups of 5 recordings x 4 windows each, as ``_embed_experimental`` returns them."""
    kinds = list(CONDITION_DISPLAY)[:2]
    exp = {}
    for j, kind in enumerate(kinds):
        emb, cells = _recordings(5, 4, dim, rng, shift=(shift_second if j else 0.0))
        exp[kind] = {"emb": emb, "cells": np.array([f"{kind}_{c + 1}" for c in cells]), "n_cells": 5,
                     "cell_number": cells + 1, "chunk": np.tile(np.arange(4), 5)}
    return kinds, exp


def test_compare_blocks_the_synthetic_set_and_caps_the_classifier_reference():
    rng = np.random.default_rng(13)
    kinds, exp = _fake_groups(rng)
    synth = rng.normal(size=(600, 8))
    res = runner._compare(exp[kinds[0]]["emb"], exp[kinds[0]]["cells"], synth, None, alpha=0.05, n_perm=30,
                          seed=0, c2st_permutations=20, c2st_reference_cap=200)
    c = res["c2st"]
    assert c["n"] == 20 and c["m"] == 200, (c["n"], c["m"])          # 50 whole blocks of 4
    assert c["n_units_y"] == 50 and c["n_exchangeable_units_x"] == 5
    assert res["mmd"]["m"] == 600                                    # the MMD reads every synthetic video
    assert res["verdict"] in ("no difference detected", "difference detected")
    full = runner._compare(exp[kinds[0]]["emb"], exp[kinds[0]]["cells"], synth, None, alpha=0.05, n_perm=10,
                           seed=0, c2st_permutations=0, c2st_reference_cap=0)
    assert full["c2st"]["m"] == 600 and np.isnan(full["c2st"]["p_value"])
    assert full["verdict"].endswith("(C2ST inconclusive)")


def test_single_window_comparisons_use_one_window_of_every_recording():
    rng = np.random.default_rng(16)
    kinds, exp = _fake_groups(rng)
    synth = rng.normal(size=(300, 8))
    emb, cells = runner.one_window_per_recording(exp[kinds[0]], 2)
    assert emb.shape == (5, 8) and len(set(cells.tolist())) == 5
    assert np.array_equal(emb, exp[kinds[0]]["emb"][2::4])              # window 2 of each recording

    def compare(A, cells_a, B, cells_b):
        return runner._compare(A, cells_a, B, cells_b, alpha=0.05, n_perm=20, seed=0, c2st_permutations=10,
                               c2st_reference_cap=100)
    one = runner.single_window_comparisons(exp, kinds, synth, 0, compare)
    assert set(one) == {"pooled@single", f"{kinds[0]}@single", f"{kinds[1]}@single",
                        f"{kinds[0]}_vs_{kinds[1]}@single"}
    c = one["pooled@single"]["c2st"]
    assert c["n"] == 10 and c["n_units_x"] == 10 and c["n_exchangeable_units_x"] == 10     # every unit one sample
    assert c["m"] == c["n_units_y"] and not np.isnan(c["p_value"])
    assert one[f"{kinds[0]}_vs_{kinds[1]}@single"]["c2st"]["n_units_y"] == 5
    assert runner.single_window_comparisons(exp, kinds, synth, 9, compare) == {}            # no such window
    lone = {kinds[0]: {key: value[:4] for key, value in exp[kinds[0]].items() if key != "n_cells"}}
    assert runner.single_window_comparisons(lone, [kinds[0]], synth, 0, compare) == {}      # one recording


def test_window_typicality_tables_name_every_window_once():
    rng = np.random.default_rng(14)
    kinds, exp = _fake_groups(rng, shift_second=8.0)                 # the second group lies outside the cloud
    synth = rng.normal(size=(400, 8))
    typ = runner.window_typicality(synth, exp, kinds, k=10)
    windows, recordings, groups = runner.typicality_tables(typ, 0.05)
    assert len(windows) == 40 and len(recordings) == 10 and len(groups) == 2
    assert len({(w[0], w[1], w[2]) for w in windows}) == 40          # group, recording, window: unique
    assert all(len(w) == len(runner.WINDOW_TYPICALITY_HEADERS) for w in windows)
    far_first, far_second = int(groups[0][2]), int(groups[1][2])
    assert far_second == 20 and far_first <= 4, (far_first, far_second)
    assert groups[1][4] == "1.0" and groups[1][5] == "5 of 5"         # expected 0.05 x 20; every recording
    fig = runner._figure_typicality(typ)
    assert len(fig.axes) == 2


def test_report_and_stored_embeddings_are_written():
    rng = np.random.default_rng(15)
    kinds, exp = _fake_groups(rng)
    synth = rng.normal(size=(300, 8))
    with tempfile.TemporaryDirectory() as tmp:
        report_dir = Path(tmp) / "report"
        R = {"report_dir": report_dir, "product_label": "2S_40FPS_TAG", "estimator_path": Path(tmp) / "e.npz",
             "estimator_sha256": "ab" * 32, "condition": "FAB", "workflow": "detector",
             "parameter_keys": ["mu_r"], "param_meaning": {}, "paths": SimpleNamespace(project_alias="ALIAS")}
        args = SimpleNamespace(alpha=0.05, mix="natural", n_permutations=20, c2st_permutations=10,
                               c2st_reference_cap=100, c2st_classifier="logreg", knn_k=5,
                               single_window_chunk=0)
        kw = dict(alpha=0.05, n_perm=20, seed=0, c2st_permutations=10, c2st_reference_cap=100)
        pooled = np.vstack([exp[k]["emb"] for k in kinds])
        pooled_cells = np.concatenate([exp[k]["cells"] for k in kinds])
        pooled_kind = np.concatenate([[k] * 20 for k in kinds])
        comps = {"pooled": runner._compare(pooled, pooled_cells, synth, None, **kw)}
        for k in kinds:
            comps[k] = runner._compare(exp[k]["emb"], exp[k]["cells"], synth, None, **kw)
        comps[f"{kinds[0]}_vs_{kinds[1]}"] = runner._compare(exp[kinds[0]]["emb"], exp[kinds[0]]["cells"],
                                                           exp[kinds[1]]["emb"], exp[kinds[1]]["cells"], **kw)
        assert comps[f"{kinds[0]}_vs_{kinds[1]}"]["c2st"]["n_exchangeable_units_x"] == 5
        comps.update(runner.single_window_comparisons(
            exp, kinds, synth, 0, lambda A, ca, B, cb: runner._compare(A, ca, B, cb, **kw)))
        groups_emb = {k: exp[k]["emb"] for k in kinds}; groups_emb["synthetic"] = synth
        typ = runner.window_typicality(synth, exp, kinds, 5)
        out = runner._write_report(R, args, comps, exp, kinds, groups_emb, 1, pooled_kind, 40,
                                   np.random.default_rng(0), comps["pooled"]["mmd"],
                                   runner._pca_pooled(groups_emb, kinds), None, None, None, typ)
        assert out == report_dir
        text = (report_dir / "report.md").read_text()
        for needle in ("C2ST AUC", "AUC null 95%", "Per-window typicality", "by recording",
                       "one window per recording (window 0)", "not by itself a failure of the forward model",
                       "linear separability"):
            assert needle in text, needle
        assert text.count("one window per recording (window 0) |") == 4          # four rows of the table
        assert "| overlap" not in text and "| gap" not in text
        assert "C2ST acc" not in text
        lines = (report_dir / "window_typicality.csv").read_text().splitlines()
        assert len(lines) == 41 and lines[0].count(",") == len(runner.WINDOW_TYPICALITY_HEADERS) - 1
        runner.save_embeddings(report_dir / "embeddings.npz", R, exp, kinds, synth, R["estimator_sha256"])
        with np.load(report_dir / "embeddings.npz") as data:
            assert data["experimental"].shape == (40, 8) and data["synthetic"].shape == (300, 8)
            assert str(data["estimator_sha256"]) == "ab" * 32 and str(data["product_label"]) == "2S_40FPS_TAG"
            assert data["experimental_chunk"].tolist() == list(range(4)) * 10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
