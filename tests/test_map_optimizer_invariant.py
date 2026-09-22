"""Regression tests for the MAP seed-then-optimize step (``evaluation.optimize_elite``).

The invariant the routine must keep is that the score it returns is the density AT the vector it
returns. Before 0.1.15 the bookkeeping read the parameters after ``optimizer.step()`` had already
moved them, so the returned pair combined one step's score with the next step's coordinates and the
returned vector sat about one learning rate away from the scored point in every coordinate the
gradient pushed consistently. A single smooth peak was enough to expose it, so these tests need no
trained flow: a stand-in exposing ``log_prob`` drives the production routine unchanged.
"""
import numpy as np
import torch

from srm_and_sbi_monomer_dimer_alp.evaluation import optimize_elite
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS

_EV = PARAMETERS.inference.evaluation
_LR = _EV.learning_rate_minimum * _EV.learning_rate_maximum_factor
_TOL = _EV.learning_rate_minimum * _EV.tolerance_factor
_MODE, _WIDTH = 1.0, 0.05


class _SmoothPeak:
    """One Gaussian peak at ``_MODE``: no secondary modes, no density spikes."""

    def log_prob(self, input=None, condition=None):
        return (-0.5 * ((input.squeeze(0) - _MODE) / _WIDTH) ** 2).sum(-1).unsqueeze(0)


def _density(x):
    return float(-0.5 * ((x - _MODE) / _WIDTH) ** 2)


def _run(seeds):
    cpu = torch.device("cpu")
    return optimize_elite(
        _SmoothPeak(), cpu, cpu, torch.zeros(1, 1),
        torch.tensor(seeds, dtype=torch.float32),
        _EV.numb_steps, _EV.optimizer_patience, _EV.scheduler_patience, 10 ** 9,
        _EV.learning_rate_minimum, _EV.learning_rate_factor, _LR, _TOL)


def test_returned_score_is_the_density_at_the_returned_vector():
    for seeds in ([[0.97], [0.95]], [[1.03], [1.05]], [[0.97], [1.03]], [[0.5], [1.6]]):
        score, theta = _run(seeds)
        assert abs(score - _density(theta[0])) < 1e-4, (
            f"seeds {seeds}: reported {score} but the density at the returned vector is "
            f"{_density(theta[0])}")


def test_returned_score_is_no_worse_than_the_best_seed():
    for seeds in ([[0.97], [0.95]], [[1.03], [1.05]], [[0.5], [1.6]]):
        score, _ = _run(seeds)
        assert score >= max(_density(s[0]) for s in seeds) - 1e-6


def test_returned_vector_is_not_displaced_by_about_one_learning_rate():
    # The pre-0.1.15 failure put the answer roughly a step away from the peak on whichever side
    # the seeds approached from, and the two seed sets disagreed by about twice that.
    below, above = _run([[0.97], [0.95]])[1][0], _run([[1.03], [1.05]])[1][0]
    assert abs(below - _MODE) < 0.1 * _LR
    assert abs(above - _MODE) < 0.1 * _LR
    assert abs(above - below) < 0.1 * _LR
