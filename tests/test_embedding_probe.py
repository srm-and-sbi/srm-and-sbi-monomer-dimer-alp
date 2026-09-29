"""The embedding probe's arithmetic (`..._DETECTOR_Embedding_Probe.py`): a linear signal in the
embedding is recovered on the held-out set, a parameter absent from it scores no better than the null,
the split is disjoint and by task or by simulation index, and the report and figure are written. No
estimator, no GPU.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load():
    path = REPO / "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Embedding_Probe.py"
    spec = importlib.util.spec_from_file_location("embedding_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(seed=0, n_tasks=4, per_task=150, width=32):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n_tasks * per_task, width))
    task = np.repeat(np.arange(n_tasks), per_task); sim = np.tile(np.arange(per_task), n_tasks)
    w = rng.normal(size=width)
    signal = emb @ w / np.sqrt(width)                                  # linearly accessible target
    theta = np.stack([signal + 0.05 * rng.normal(size=signal.size),    # 'a': in the embedding
                      rng.normal(size=signal.size)], axis=1)           # 'b': independent of it
    return emb, theta, task, sim


def test_a_linear_signal_is_recovered_and_an_absent_one_scores_at_the_null():
    pr = _load()
    emb, theta, task, sim = _fixture()
    fit, dev, held = pr.split_masks(task, sim, fit_tasks=[0, 1], dev_tasks=[2], held_out_tasks=[3], dev_fraction=0.2)
    assert fit.sum() == 300 and dev.sum() == 150 and held.sum() == 150
    result = pr.probe(emb, theta, fit, dev, held, ["a", "b"])
    a, b = result["a"]["held_out"], result["b"]["held_out"]
    assert a["mae"] < 0.2 * result["a"]["null_mae"] and 0.9 < a["slope"] < 1.1 and a["corr"] > 0.95
    assert b["mae"] > 0.9 * result["b"]["null_mae"] and abs(b["corr"]) < 0.3
    assert set(result["a"]["by_regime"]) == {"low_half", "high_half"}
    assert result["a"]["by_regime"]["low_half"]["n"] + result["a"]["by_regime"]["high_half"]["n"] == 150


def test_without_development_tasks_the_last_fraction_of_each_fit_task_serves_and_sets_never_overlap():
    pr = _load()
    _, _, task, sim = _fixture()
    fit, dev, held = pr.split_masks(task, sim, fit_tasks=[0, 1], dev_tasks=[], held_out_tasks=[2], dev_fraction=0.2)
    assert fit.sum() == 240 and dev.sum() == 60 and held.sum() == 150
    assert not (fit & dev).any() and not (fit & held).any() and not (dev & held).any()
    assert sim[dev].min() == 120                                       # the last 20 % by simulation index
    emb, theta, task2, sim2 = _fixture(n_tasks=2, per_task=8)          # smoke-sized: refused before fitting
    f, d, h = pr.split_masks(task2, sim2, [0], [], [1], 0.2)
    try:
        pr.probe(emb, theta, f, d, h, ["a", "b"])
    except SystemExit as e:
        assert "too few videos" in str(e) and "development" in str(e)
    else:
        raise AssertionError("undersized sets accepted")
    assert pr.scores(np.zeros(1), np.zeros(1))["n"] == 1 and np.isnan(pr.scores(np.zeros(1), np.zeros(1))["mae"])
    try:
        pr.split_masks(task, sim, fit_tasks=[0], dev_tasks=[0], held_out_tasks=[1], dev_fraction=0.2)
    except SystemExit as e:
        assert "overlap" in str(e)
    else:
        raise AssertionError("overlapping sets accepted")


def test_the_report_and_figure_are_written_from_the_result():
    pr = _load()
    emb, theta, task, sim = _fixture()
    fit, dev, held = pr.split_masks(task, sim, [0, 1], [2], [3], 0.2)
    result = pr.probe(emb, theta, fit, dev, held, ["a", "b"])
    meta = dict(label="2S_50FPS", estimator="x.npz", weights_sha256="0" * 64, width=32, package_version="t",
                device="cpu", fit_tasks=[0, 1], dev_tasks=[2], held_out_tasks=[3], dev_fraction=0.2,
                n_fit=300, n_dev=150, n_held=150)
    with tempfile.TemporaryDirectory() as tmp:
        pr.write_report(result, ["a", "b"], ["A", "B"], meta, Path(tmp) / "r.md")
        pr.write_figure(result, ["a", "b"], ["A", "B"], Path(tmp) / "r.png")
        text = (Path(tmp) / "r.md").read_text()
        assert "| A |" in text and "| B |" in text and "null" in text and (Path(tmp) / "r.png").stat().st_size > 0
        assert "not thereby shown to be absent" in text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
