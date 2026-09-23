"""The MAP benchmark's batched optimizer engine: it must reproduce the serial reference loop
(torch.optim.Adam + ReduceLROnPlateau) chain by chain, keep chains independent, retain the best
(score, vector) pair consistently, and apply the stopping, guard and safeguard rules as specified.
Analytic objectives only; no trained flow.
"""
import numpy as np
import torch

from srm_and_sbi_monomer_dimer_alp import map_benchmark as mb

D = 6
MU = torch.tensor([0.15, -0.6, 2.4, -0.4, -1.3, 0.5], dtype=torch.float64)
SIG = torch.tensor([0.01, 0.2, 0.05, 0.1, 0.5, 0.03], dtype=torch.float64)


def _score_rows(theta, idx=None):
    z = (theta - MU) / SIG
    return -0.5 * (z * z).sum(dim=1) + 0.3 * z[:, 0] * z[:, 2]          # anisotropic, correlated


def _chains(configs, starts):
    rows = {k: [] for k in ("theta0", "center", "scale", "lr", "lr_min", "max_steps", "stop_patience",
                            "sched_patience", "guard")}
    for coords, lr, lr_min, budget in configs:
        for start in starts:
            rows["theta0"].append(start)
            rows["center"].append(MU + 0.02 * SIG if coords == "scaled" else torch.zeros(D, dtype=torch.float64))
            rows["scale"].append(SIG * 1.35 if coords == "scaled" else torch.ones(D, dtype=torch.float64))
            rows["lr"].append(lr); rows["lr_min"].append(lr_min)
            for key in ("max_steps", "stop_patience", "sched_patience", "guard"):
                rows[key].append(budget[key])
    t = lambda v, dt: torch.tensor(v, dtype=dt)
    return (torch.stack(rows["theta0"]), torch.stack(rows["center"]), torch.stack(rows["scale"]),
            t(rows["lr"], torch.float64), t(rows["lr_min"], torch.float64), t(rows["max_steps"], torch.long),
            t(rows["stop_patience"], torch.long), t(rows["sched_patience"], torch.long), t(rows["guard"], torch.long))


def test_batched_engine_reproduces_the_serial_reference_chain_by_chain():
    gen = torch.Generator().manual_seed(0)
    starts = [MU + SIG * torch.randn(D, generator=gen, dtype=torch.float64) * 0.7 for _ in range(3)]
    configs = [("absolute", 0.128, 1e-3, mb.CURRENT), ("scaled", 0.05, 5e-4, mb.CURRENT),
               ("scaled", 0.05, 5e-4, mb.EXTENDED), ("scaled", 0.2, 5e-4, mb.EXTENDED)]
    args = _chains(configs, starts)
    res = mb.run_chains(_score_rows, *args)
    m = 0
    for coords, lr, lr_min, budget in configs:
        for start in starts:
            best, theta, code, steps = mb.run_chain_serial(
                lambda th: _score_rows(th.unsqueeze(0))[0], start, args[1][m], args[2][m], lr=lr,
                lr_min=lr_min, max_steps=budget["max_steps"], stop_patience=budget["stop_patience"],
                sched_patience=budget["sched_patience"], guard=budget["guard"])
            assert abs(float(res["best"][m]) - best) < 1e-9, (coords, lr, m, float(res["best"][m]), best)
            assert int(res["stop"][m]) == code and int(res["steps"][m]) == steps, (m, code, steps)
            assert torch.allclose(res["best_theta"][m], theta, atol=1e-10)
            m += 1


def test_chains_are_independent_of_their_batch_companions():
    gen = torch.Generator().manual_seed(1)
    start = MU + 0.5 * SIG * torch.randn(D, generator=gen, dtype=torch.float64)
    alone = mb.run_chains(_score_rows, *_chains([("scaled", 0.05, 5e-4, mb.EXTENDED)], [start]))
    others = [MU + SIG * torch.randn(D, generator=gen, dtype=torch.float64) for _ in range(5)]
    batch = mb.run_chains(_score_rows, *_chains([("scaled", 0.05, 5e-4, mb.EXTENDED)], [start] + others))
    assert float(alone["best"][0]) == float(batch["best"][0])
    assert int(alone["steps"][0]) == int(batch["steps"][0])
    assert torch.equal(alone["best_theta"][0], batch["best_theta"][0])


def test_returned_score_is_the_density_at_the_returned_vector_and_never_below_the_start():
    gen = torch.Generator().manual_seed(2)
    starts = [MU + SIG * torch.randn(D, generator=gen, dtype=torch.float64) for _ in range(4)]
    args = _chains([("absolute", 0.128, 1e-3, mb.CURRENT), ("scaled", 0.05, 5e-4, mb.EXTENDED)], starts)
    res = mb.run_chains(_score_rows, *args)
    assert torch.allclose(res["best"], _score_rows(res["best_theta"]), atol=1e-12)
    assert bool((res["best"] >= res["seed_score"]).all())


def test_scale_aware_steps_reach_the_mode_where_the_absolute_step_is_too_coarse():
    start = MU + 0.8 * SIG
    args = _chains([("scaled", 0.05, 5e-4, mb.EXTENDED), ("absolute", 0.128, 1e-3, mb.CURRENT)], [start])
    res = mb.run_chains(_score_rows, *args)
    mode = MU.clone()                                          # the cross term moves the mode: solve
    a = torch.tensor([[1.0, -0.3], [-0.3, 1.0]], dtype=torch.float64)   # in z for coords 0 and 2: z = 0
    assert torch.allclose(a @ torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
    z_scaled = ((res["best_theta"][0] - mode) / SIG).abs().max()
    assert float(z_scaled) < 0.02, float(z_scaled)             # within 2 % of a posterior SD
    assert float(res["best"][0]) >= float(res["best"][1]) - 1e-12


def test_patience_guard_and_budget_rules_on_a_flat_objective():
    flat = lambda theta, idx=None: torch.zeros(theta.shape[0], dtype=theta.dtype) + 0.0 * theta.sum(dim=1)
    budget_guard = {"max_steps": 100, "stop_patience": 5, "sched_patience": 2, "guard": 3}
    budget_none = dict(budget_guard, guard=0)
    budget_short = {"max_steps": 4, "stop_patience": 50, "sched_patience": 2, "guard": 0}
    start = [MU.clone()]
    for budget, code, step in ((budget_none, mb.STOP_EARLY, 6), (budget_guard, mb.STOP_EARLY, 7),
                               (budget_short, mb.STOP_BUDGET, 4)):
        res = mb.run_chains(flat, *_chains([("absolute", 0.1, 1e-3, budget)], start))
        assert (int(res["stop"][0]), int(res["steps"][0])) == (code, step), (budget, res["stop"], res["steps"])
        s = mb.run_chain_serial(lambda th: flat(th.unsqueeze(0))[0], start[0], torch.zeros(D, dtype=torch.float64),
                                torch.ones(D, dtype=torch.float64), lr=0.1, lr_min=1e-3,
                                max_steps=budget["max_steps"], stop_patience=budget["stop_patience"],
                                sched_patience=budget["sched_patience"], guard=budget["guard"])
        assert (s[2], s[3]) == (code, step)


def test_scaled_coordinates_and_safeguard_flags():
    rng = np.random.default_rng(3)
    pool = rng.normal(size=(1000, 3)) * np.array([0.05, 1e-6, 5.0])
    w = np.array([0.3, 0.75, 1.5])
    center, iqr, scale, raised, capped = mb.scaled_coordinates(pool, w)
    assert np.allclose(center, np.median(pool, axis=0))
    assert raised.tolist() == [False, True, False] and capped.tolist() == [False, False, True]
    assert np.isclose(scale[1], mb.SCALE_EPS * w[1]) and np.isclose(scale[2], w[2])
    assert np.isclose(scale[0], iqr[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
