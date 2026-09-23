"""Regression tests for the MAP seed-then-optimize step (``evaluation.optimize_elite`` and
``evaluation.map_estimate``).

Two invariants the routine must keep: the score it returns is the density AT the vector it returns
(before 0.1.15 the bookkeeping read the parameters after ``optimizer.step()`` had moved them, so the
returned vector sat about one learning rate away from the scored point), and that score is never
worse than the best seed's. From 0.1.17 the ascent steps in units of the candidate pool's
interquartile range, retains every strictly better finite pair, and uses the tolerance only for the
stopping patience and the plateau scheduler. Analytic stand-ins exposing ``log_prob`` drive the
production routines unchanged, so no trained flow is needed.
"""
import numpy as np
import torch

from srm_and_sbi_monomer_dimer_alp import artifact_schema as schema
from srm_and_sbi_monomer_dimer_alp.evaluation import (
    STEP_COORDINATES, STOP_REASONS, collect_theta_prex, map_estimate, optimize_elite,
    optimizer_contract, pool_scale)
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS

_EV = PARAMETERS.inference.evaluation
_MODE, _WIDTH = 1.0, 0.05
_IQR = 1.349 * _WIDTH            # interquartile range of a normal with standard deviation _WIDTH
_CPU = torch.device("cpu")


class _Peak:
    """One Gaussian peak, log-density -0.5 * sum(((theta - mode) / width) ** 2): no secondary modes."""

    def __init__(self, mode=_MODE, width=_WIDTH):
        self.mode, self.width = mode, width

    def log_prob(self, input=None, condition=None):
        mode = torch.as_tensor(self.mode, dtype=input.dtype)
        return (-0.5 * ((input.squeeze(0) - mode) / self.width) ** 2).sum(-1).unsqueeze(0)


class _Ramp:
    """An unbounded ramp in the first coordinate, log-density = slope * theta_0, so Adam's step is
    exactly the learning rate; optionally non-finite above ``cut``."""

    def __init__(self, slope, cut=None):
        self.slope, self.cut = slope, cut

    def log_prob(self, input=None, condition=None):
        x = input.squeeze(0)[..., 0]
        out = self.slope * x
        if self.cut is not None:
            out = torch.where(x > self.cut, torch.full_like(out, float("nan")), out)
        return out.unsqueeze(0)


def _density(x, mode=_MODE, width=_WIDTH):
    return float(-0.5 * ((x - mode) / width) ** 2)


def _run(flow, seeds, *, center=(0.0,), scale=(_IQR,), dtype=torch.float32, **over):
    s = dict(numb_steps=_EV.numb_steps, optimizer_patience=_EV.optimizer_patience,
             scheduler_patience=_EV.scheduler_patience, learning_rate=_EV.learning_rate,
             learning_rate_minimum=_EV.learning_rate_minimum,
             learning_rate_factor=_EV.learning_rate_factor, tolerance=_EV.tolerance)
    s.update(over)
    return optimize_elite(
        flow, _CPU, _CPU, torch.zeros(1, 1), torch.tensor(seeds, dtype=dtype),
        s["numb_steps"], s["optimizer_patience"], s["scheduler_patience"], 10 ** 9,
        s["learning_rate_minimum"], s["learning_rate_factor"], s["learning_rate"], s["tolerance"],
        center=np.asarray(center, dtype=float), scale=np.asarray(scale, dtype=float))


# ---- the adopted settings, as configured and as recorded ---------------------------------------

def test_configured_settings_are_the_adopted_ones_and_the_contract_records_them():
    assert (_EV.numb_steps, _EV.optimizer_patience, _EV.scheduler_patience) == (2000, 200, 20)
    assert (_EV.learning_rate, _EV.learning_rate_minimum, _EV.learning_rate_factor) == (0.05, 5e-4, 0.5)
    assert _EV.tolerance == 1e-3
    c = optimizer_contract(_EV, learning_rate=_EV.learning_rate, tolerance=_EV.tolerance,
                           theta_prex_size=_EV.theta_prex_size, elite_prex_size=_EV.elite_prex_size,
                           numb_steps=_EV.numb_steps, pool_mode="bounded")
    assert set(schema.OPTIMIZER_KEYS) <= set(c)
    assert (c["numb_steps"], c["optimizer_patience"], c["scheduler_patience"]) == (2000, 200, 20)
    assert (c["learning_rate"], c["learning_rate_minimum"], c["tolerance"]) == (0.05, 5e-4, 1e-3)
    assert c["step_coordinates"] == STEP_COORDINATES and "IQR" in c["step_coordinates"]
    assert "strictly better" in c["bookkeeping"]


# ---- the two invariants ------------------------------------------------------------------------

def test_returned_score_is_the_density_at_the_returned_vector():
    for seeds in ([[0.97], [0.95]], [[1.03], [1.05]], [[0.97], [1.03]], [[0.5], [1.6]]):
        score, theta, info = _run(_Peak(), seeds)
        assert info["stop"] in STOP_REASONS
        assert abs(score - _density(theta[0])) < 1e-4, (seeds, score, _density(theta[0]))


def test_returned_score_is_no_worse_than_the_best_seed():
    for seeds in ([[0.97], [0.95]], [[1.03], [1.05]], [[0.5], [1.6]]):
        score, _, info = _run(_Peak(), seeds)
        assert score >= max(_density(s[0]) for s in seeds) - 1e-6
        assert abs(info["seed_score"] - max(_density(s[0]) for s in seeds)) < 1e-4


def test_the_ascent_reaches_the_mode_from_either_side():
    # The pre-0.1.15 failure put the answer about one step past the peak on whichever side the seeds
    # came from; with IQR-unit steps a step is learning_rate * IQR in theta.
    step = _EV.learning_rate * _IQR
    below, above = _run(_Peak(), [[0.97], [0.95]])[1][0], _run(_Peak(), [[1.03], [1.05]])[1][0]
    far = _run(_Peak(), [[0.5], [1.6]])[1][0]              # about 7 and 9 IQRs from the mode
    for x in (below, above, far):
        assert abs(x - _MODE) < 0.1 * step, x
    assert abs(above - below) < 0.1 * step


# ---- strict retention versus the patience threshold -------------------------------------------

def test_every_strictly_better_pair_is_kept_even_below_the_tolerance():
    # A peak so flat that every step gains far less than the tolerance: the patience runs out at
    # step optimizer_patience + 1, yet the returned pair is the best one visited, not the seed
    # (the pre-0.1.17 rule kept only gains above the tolerance and returned the seed here).
    flat = _Peak(width=100.0)
    score, theta, info = _run(flat, [[0.9], [0.8]], scale=(0.1,))
    assert info["stop"] == "early" and info["steps"] == _EV.optimizer_patience + 1
    assert score > info["seed_score"]
    assert abs(score - _density(theta[0], width=100.0)) < 1e-9


def test_patience_counts_from_the_last_meaningful_improvement():
    # Each step gains 0.4 tolerance: no single step is meaningful, but every third step the best has
    # risen more than the tolerance since the last meaningful improvement, so the ascent runs to its
    # budget. A zero-gain objective stops at patience + 1.
    tol, lr = _EV.tolerance, _EV.learning_rate
    over = dict(numb_steps=300, optimizer_patience=50, scheduler_patience=20)
    _, _, info = _run(_Ramp(slope=0.4 * tol / lr), [[0.0]], scale=(1.0,), **over)
    assert info["stop"] == "budget" and info["steps"] == 300
    _, _, info = _run(_Ramp(slope=0.0), [[0.0]], scale=(1.0,), **over)
    assert info["stop"] == "early" and info["steps"] == 51


# ---- units, non-finite scores, and the pool that cannot set a step ------------------------------

def test_steps_are_measured_in_pool_iqr_units():
    # The same problem in units a thousand times larger, with the pool's median and IQR scaled with
    # it, follows the same path: the returned vector scales, and the score, steps and stop agree.
    a = 1000.0
    s1, t1, i1 = _run(_Peak(), [[0.70], [0.75]], center=(0.9,), scale=(_IQR,), dtype=torch.float64)
    s2, t2, i2 = _run(_Peak(mode=a * _MODE, width=a * _WIDTH), [[a * 0.70], [a * 0.75]],
                      center=(a * 0.9,), scale=(a * _IQR,), dtype=torch.float64)
    assert np.allclose(t2 / a, t1, rtol=0, atol=1e-10)
    assert abs(s1 - s2) < 1e-9
    assert (i1["stop"], i1["steps"]) == (i2["stop"], i2["steps"])


def test_a_non_finite_score_ends_the_ascent_with_the_best_finite_pair():
    score, theta, info = _run(_Ramp(slope=1.0, cut=0.5), [[0.0]], scale=(1.0,))
    assert info["stop"] == "non-finite"
    assert np.isfinite(score) and abs(score - float(theta[0])) < 1e-6
    assert 0.45 < float(theta[0]) <= 0.5


def test_pool_scale_is_the_median_and_interquartile_range():
    pool = np.random.default_rng(3).normal(size=(501, 4))
    center, iqr = pool_scale(torch.tensor(pool, dtype=torch.float32))
    assert np.allclose(center, np.median(pool.astype(np.float32).astype(float), axis=0))
    q25, q75 = np.quantile(pool.astype(np.float32).astype(float), (0.25, 0.75), axis=0)
    assert np.allclose(iqr, q75 - q25)
    pool[7, 2] = np.nan
    assert not np.isfinite(pool_scale(pool)[1][2])     # reported by map_estimate, never divided


class _GradSampler(torch.nn.Module):
    """A sampler whose draws depend on a parameter, so they would carry an autograd graph. Serves
    both pool modes (``posterior.sample`` returns (n, D); ``flow.sample`` returns (n, 1, D)),
    records the size of every call, and refuses to be called endlessly."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(2))
        self.sizes = []

    def sample(self, sample_shape=(), x=None, show_progress_bars=False, condition=None):
        n = int(sample_shape[0])
        self.sizes.append(n)
        assert len(self.sizes) < 1000, "the sampler was called endlessly"
        draws = torch.randn(n, 2) * self.w
        return draws if condition is None else draws.unsqueeze(1)


def test_the_sampler_returns_exact_counts_without_a_graph_and_refuses_invalid_sizes():
    # Unrestricted sampling ran with gradients enabled, so every call's draws kept the flow's (and
    # the embedding's) autograd graph alive. In both pool modes, one draw, an exact batch multiple, a
    # final partial batch and the production case return exactly the requested count in the expected
    # calls, without a graph, and gradient tracking is on again afterwards. A size or batch below one
    # is refused before the loop: a zero batch would never finish.
    cond = torch.zeros(1, 1)
    for mode in ("bounded", "unrestricted"):
        for n, batch, calls in ((1, 100, [1]), (300, 100, [100, 100, 100]),
                                (250, 100, [100, 100, 50]), (1000, 10000, [1000])):
            sampler = _GradSampler()
            draws = collect_theta_prex(sampler, sampler, _CPU, cond, n, batch, mode)
            assert draws.shape == (n, 2) and sampler.sizes == calls, (mode, n, batch, sampler.sizes)
            assert not draws.requires_grad and draws.grad_fn is None
            assert torch.is_grad_enabled()
            (sampler.w * 2.0).sum().backward()
            assert sampler.w.grad is not None
        for n, batch in ((10, 0), (10, -5), (0, 100)):
            sampler = _GradSampler()
            try:
                collect_theta_prex(sampler, sampler, _CPU, cond, n, batch, mode)
            except ValueError:
                assert sampler.sizes == []
            else:
                raise AssertionError(f"size {n} with batch {batch} was accepted ({mode})")


# ---- the scheduler and the stopping rule together ----------------------------------------------

class _Scripted:
    """Scripted score values, one per call, with a constant gradient of +1 in the first
    coordinate: Adam then moves the seed by one learning rate per step whatever the value, so the
    best visited point and the last point are known in advance."""

    def __init__(self, values):
        self.values, self.calls = list(values), 0

    def log_prob(self, input=None, condition=None):
        x = input.squeeze(0)[..., 0]
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return (value + (x - x.detach())).unsqueeze(0)          # exact value, gradient +1


def _scripted_run(values, numb_steps):
    """Stopping patience 30, scheduler patience 5, learning rate 0.05 halved to a 0.0125 floor,
    tolerance 1e-3; one seed at 0 in unit scale. Returns the result and the per-step learning rates
    read from the progress lines."""
    lines = []
    result = optimize_elite(_Scripted(values), _CPU, _CPU, torch.zeros(1, 1), torch.zeros(1, 1),
                            numb_steps, 30, 5, 1, 0.0125, 0.5, 0.05, 1e-3, log_fn=lines.append,
                            center=np.zeros(1), scale=np.ones(1))
    rates = [float(line.split("lr=")[1].split()[0]) for line in lines if line.startswith("progress")]
    return result, rates


def test_rate_reductions_stop_at_the_floor_keep_the_patience_and_the_best_point_is_returned():
    # Four meaningful gains, then a plateau just below the best: the scheduler halves the rate twice
    # and holds it at the floor; the reductions do not reset the stopping patience, so the ascent
    # stops 30 steps after the last meaningful gain; and it returns the best point (step 5, after four
    # steps of 0.05), not the last one (about 0.91).
    (score, theta, info), rates = _scripted_run([0.0, 1.0, 2.0, 3.0, 4.0] + [3.9995] * 200, 200)
    assert rates[0] == 0.05 and all(b <= a for a, b in zip(rates, rates[1:]))
    assert sorted(set(rates)) == [0.0125, 0.025, 0.05] and info["learning_rate_final"] == 0.0125
    assert info["stop"] == "early" and info["steps"] == 5 + 30
    assert score == 4.0 and abs(float(theta[0]) - 4 * 0.05) < 1e-4
    # A gain of twice the tolerance every seventh step: the rate still falls to the floor between
    # gains, the patience never runs out, the budget ends the ascent, and the returned point is the
    # first visit of the best value (step 57), not a later step with the same value.
    values = [0.002 * ((k - 1) // 7) for k in range(1, 61)]
    (score, theta, info), rates = _scripted_run(values, 60)
    assert info["stop"] == "budget" and info["steps"] == 60
    assert min(rates) == 0.0125 and info["learning_rate_final"] == 0.0125
    assert abs(score - 0.016) < 1e-6 and abs(float(theta[0]) - (7 * 0.05 + 7 * 0.025 + 42 * 0.0125)) < 1e-4


class _StubFlow:
    """The parts of an sbi flow that ``map_estimate`` touches, over an analytic density."""

    def __init__(self, density, condition_shape=(1,), embed_dim=1):
        self.density = density
        self._embedding_net = lambda x: torch.zeros(x.shape[0], embed_dim)
        self._condition_shape = tuple(condition_shape)

    @property
    def embedding_net(self):
        return self._embedding_net

    @property
    def condition_shape(self):
        return self._condition_shape

    def log_prob(self, input=None, condition=None):
        return self.density.log_prob(input, condition)


class _StubPosterior:
    """Hands out a fixed candidate pool, batch by batch, as the bounded sampler would."""

    def __init__(self, pool, density, **flow_kw):
        self.pool = torch.as_tensor(pool, dtype=torch.float32)
        self.posterior_estimator = _StubFlow(density, **flow_kw)

    def set_default_x(self, x):
        pass

    def sample(self, sample_shape=(), x=None, show_progress_bars=False):
        n = sample_shape[0]
        out, self.pool = self.pool[:n], self.pool[n:]
        return out


def _map_on(post, log_fn=None, return_info=True):
    return map_estimate(
        post, np.ones((2, 3, 3), dtype=np.uint8), _CPU, _CPU,
        len(post.pool), 50, 20, 2, _EV.numb_steps, _EV.optimizer_patience, _EV.scheduler_patience,
        10 ** 9, _EV.learning_rate_minimum, _EV.learning_rate_factor, _EV.learning_rate,
        _EV.tolerance, log_fn=log_fn, return_info=return_info)


def _map(pool, density, log_fn=None, return_info=True):
    return _map_on(_StubPosterior(pool, density), log_fn, return_info)


class _FailOnCall:
    """Wraps a density and raises on its n-th ``log_prob`` call."""

    def __init__(self, density, n):
        self.density, self.n, self.calls = density, n, 0

    def log_prob(self, input=None, condition=None):
        self.calls += 1
        if self.calls == self.n:
            raise RuntimeError("injected failure")
        return self.density.log_prob(input, condition)


def test_a_failure_inside_map_estimate_restores_the_estimator():
    # map_estimate swaps the embedding network for an identity and the condition shape for the
    # cached latent's. A failure while scoring (call 1) or during the ascent (call 15, after the ten
    # scoring batches) must leave both restored, and the next observation must run normally.
    rng = np.random.default_rng(7)
    pool = np.stack([rng.normal(1.0, 0.05, 200), rng.normal(3.0, 0.2, 200)], axis=1)
    peak = _Peak(mode=(1.0, 3.0), width=0.05)
    for fail_at in (1, 15):
        post = _StubPosterior(pool, _FailOnCall(peak, fail_at), condition_shape=(2, 3, 3), embed_dim=4)
        flow = post.posterior_estimator
        original = flow._embedding_net
        try:
            _map_on(post)
        except RuntimeError as exc:
            assert "injected failure" in str(exc)
        else:
            raise AssertionError(f"the injected failure at call {fail_at} was not raised")
        assert flow._embedding_net is original and flow._condition_shape == (2, 3, 3)
        post.pool = torch.as_tensor(pool, dtype=torch.float32)     # the stub hands its pool out once
        flow.density = peak
        score, theta, info = _map_on(post)
        assert np.isfinite(score) and info["stop"] in ("early", "budget")
        assert flow._embedding_net is original and flow._condition_shape == (2, 3, 3)


def test_map_estimate_steps_in_the_pool_iqr_and_reports_its_scales():
    rng = np.random.default_rng(5)
    pool = np.stack([rng.normal(1.0, 0.05, 200), rng.normal(3.0, 0.2, 200)], axis=1)
    density = _Peak(mode=(1.0, 3.0), width=0.05)
    score, theta, info = _map(pool, density)
    center, iqr = pool_scale(torch.tensor(pool, dtype=torch.float32))
    assert np.allclose(info["center"], center) and np.allclose(info["scale"], iqr)
    assert info["scale_valid"].all() and info["stop"] in ("early", "budget")
    best_candidate = max(float(density.log_prob(torch.tensor(pool[None], dtype=torch.float32))[0, i])
                         for i in range(len(pool)))
    assert score >= best_candidate - 1e-6
    assert abs(score - float(density.log_prob(torch.tensor(theta[None, None]))[0, 0])) < 1e-4
    assert len(_map(pool, density, return_info=False)) == 2


def test_a_zero_iqr_skips_the_ascent_returns_the_best_candidate_and_is_reported():
    rng = np.random.default_rng(6)
    pool = np.stack([rng.normal(1.0, 0.05, 200), np.full(200, 3.0)], axis=1)   # IQR 0 in coordinate 1
    density = _Peak(mode=(1.0, 3.0), width=0.05)
    lines = []
    score, theta, info = _map(pool, density, log_fn=lines.append)
    assert info["stop"] == "invalid-scale" and info["steps"] == 0
    assert list(info["scale_valid"]) == [True, False]
    scores = density.log_prob(torch.tensor(pool[None], dtype=torch.float32))[0].numpy()
    best = int(np.argmax(scores))
    assert np.allclose(theta, pool[best].astype(np.float32)) and abs(score - scores[best]) < 1e-6
    assert any(line.startswith("WARNING") and "IQR" in line for line in lines)


if __name__ == "__main__":
    import sys
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                      # noqa: BLE001 -- report every failure
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
