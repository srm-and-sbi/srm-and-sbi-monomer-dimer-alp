"""Gradient accumulation to a fixed global batch (`--global-batch`; DETECTOR_WORKFLOW.md, the encoder
screening's batch rule): `resolve_accumulation_steps` turns a requested global batch into the number of
per-rank batches summed per optimizer step, refusing what cannot be realized; `train_loop` with that count
takes exactly the optimizer steps of a run whose batch is the whole group (same parameters after an epoch,
to floating-point precision), scales the last partial group the same way, and with count 1 reproduces the
original one-step-per-batch loop. CPU only, a tiny linear estimator; no data bank.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srm_and_sbi_monomer_dimer_alp.inference_support import (Topology, resolve_accumulation_steps,  # noqa: E402
                                                              train_loop)


class _LinearEstimator(nn.Module):
    """A stand-in for the posterior estimator: `loss(theta, condition)` is the per-sample squared error
    of a linear read-out of the flattened video; exactly what train_loop drives."""

    def __init__(self, n_pixels, theta_dim, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(n_pixels, theta_dim, generator=g) * 0.1)
        self.bias = nn.Parameter(torch.zeros(theta_dim))

    def loss(self, theta, condition):
        pred = condition.flatten(1) @ self.weight + self.bias
        return ((pred - theta) ** 2).sum(dim=1)

    def forward(self, theta, condition):          # the _LossModule contract: forward IS the loss
        return self.loss(theta, condition=condition)


def _data(n=12, frames=3, side=4, theta_dim=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    videos = torch.rand(n, frames, side, side, generator=g)
    theta = torch.randn(n, theta_dim, generator=g)
    return TensorDataset(videos, theta)


def _run_epoch(dataset, batch_size, accumulation_steps, lr=0.05, epochs=1, seed=0):
    """One train_loop call on a single CPU worker (no TEST set), returning the trained estimator."""
    est = _LinearEstimator(n_pixels=dataset.tensors[0][0].numel(), theta_dim=dataset.tensors[1].shape[1], seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = optim.SGD(est.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)
    topo = Topology(world_size=1, rank=0, local_rank=0, device=torch.device("cpu"), backend="CPU")
    with tempfile.TemporaryDirectory() as tmp:
        out = train_loop(estimator=est, model=est, train_loader=loader, val_loader=None, optimizer=optimizer,
                         scheduler=scheduler, device=torch.device("cpu"), topo=topo,
                         checkpoint_path=Path(tmp) / "ckpt.pth", epochs=epochs,
                         accumulation_steps=accumulation_steps)
    return est, out


def _manual_sgd(dataset, group_size, lr=0.05, seed=0):
    """Reference: plain SGD stepping once per group of `group_size` samples on the mean loss."""
    est = _LinearEstimator(n_pixels=dataset.tensors[0][0].numel(), theta_dim=dataset.tensors[1].shape[1], seed=seed)
    videos, theta = dataset.tensors
    for start in range(0, len(dataset), group_size):
        v, t = videos[start:start + group_size], theta[start:start + group_size]
        loss = est.loss(t, condition=v).mean()
        grads = torch.autograd.grad(loss, list(est.parameters()))
        with torch.no_grad():
            for p, g in zip(est.parameters(), grads):
                p -= lr * g
    return est


def _params(est):
    return [p.detach().clone() for p in est.parameters()]


def _same(a, b, atol=1e-6):
    return all(torch.allclose(x, y, atol=atol, rtol=0) for x, y in zip(a, b))


def test_resolve_accumulation_steps_realizes_the_global_batch_or_refuses():
    assert resolve_accumulation_steps(None, 16, 128) == 1          # no constraint: one step per batch
    assert resolve_accumulation_steps(1024, 16, 64) == 1           # the control's geometry needs no accumulation
    assert resolve_accumulation_steps(1024, 16, 32) == 2           # 8 nodes x 4 at batch 16
    assert resolve_accumulation_steps(1024, 32, 32) == 1           # the baseline's geometry
    assert resolve_accumulation_steps(2048, 16, 128) == 1
    assert resolve_accumulation_steps(1024, 8, 16) == 8
    for bad in ((1024, 16, 128), (512, 16, 64)):                   # smaller than one batch covers
        try:
            resolve_accumulation_steps(*bad)
        except ValueError as exc:
            assert "smaller than one batch" in str(exc)
        else:
            raise AssertionError(bad)
    for bad in ((1000, 16, 32), (1536, 16, 64)):                   # not a whole number of batches
        try:
            resolve_accumulation_steps(*bad)
        except ValueError as exc:
            assert "not a multiple" in str(exc)
        else:
            raise AssertionError(bad)


def test_accumulation_count_one_is_the_original_loop():
    ds = _data(n=12)
    est, (ltr, lte, lrep, best) = _run_epoch(ds, batch_size=4, accumulation_steps=1)
    ref = _manual_sgd(ds, group_size=4)
    assert _same(_params(est), _params(ref))
    assert ltr.shape == (1,) and np.isfinite(ltr[0]) and np.isnan(lte[0]) and best == float("inf")


def test_accumulated_steps_equal_one_step_on_the_whole_group():
    # 12 samples, batch 2, accumulate 3 -> two optimizer steps, each the mean-loss gradient over 6 samples;
    # identical (to float precision) to plain SGD with batch 6.
    ds = _data(n=12)
    est, _ = _run_epoch(ds, batch_size=2, accumulation_steps=3)
    ref = _manual_sgd(ds, group_size=6)
    assert _same(_params(est), _params(ref))
    # And NOT equal to stepping per batch of 2 (the accumulation changed the optimization, as intended).
    per_batch = _manual_sgd(ds, group_size=2)
    assert not _same(_params(est), _params(per_batch))


def test_the_last_partial_group_is_stepped_with_the_same_scaling():
    # 10 samples, batch 2, accumulate 4 -> groups of 8 and 2 samples. The loop divides every batch's mean
    # loss by 4, so the final (2-sample) group updates with the gradient of (mean over 2) / 4 * 1 batch
    # = a quarter-weight step. Reference: step 1 on the mean over the 8; step 2 on mean-over-2 scaled by 1/4.
    ds = _data(n=10)
    est, _ = _run_epoch(ds, batch_size=2, accumulation_steps=4)
    ref = _LinearEstimator(n_pixels=ds.tensors[0][0].numel(), theta_dim=2, seed=0)
    videos, theta = ds.tensors
    for sl, scale in ((slice(0, 8), 1.0), (slice(8, 10), 0.25)):
        loss = ref.loss(theta[sl], condition=videos[sl]).mean() * scale
        grads = torch.autograd.grad(loss, list(ref.parameters()))
        with torch.no_grad():
            for p, g in zip(ref.parameters(), grads):
                p -= 0.05 * g
    assert _same(_params(est), _params(ref))


def test_two_epochs_accumulate_across_the_epoch_boundary_correctly():
    # The group counter restarts every epoch: 8 samples, batch 2, accumulate 2 -> 2 steps per epoch, 4 in all.
    ds = _data(n=8)
    est, (ltr, *_r) = _run_epoch(ds, batch_size=2, accumulation_steps=2, epochs=2)
    ref = _LinearEstimator(n_pixels=ds.tensors[0][0].numel(), theta_dim=2, seed=0)
    videos, theta = ds.tensors
    for _epoch in range(2):
        for start in range(0, 8, 4):
            loss = ref.loss(theta[start:start + 4], condition=videos[start:start + 4]).mean()
            grads = torch.autograd.grad(loss, list(ref.parameters()))
            with torch.no_grad():
                for p, g in zip(ref.parameters(), grads):
                    p -= 0.05 * g
    assert _same(_params(est), _params(ref)) and ltr.shape == (2,)


def test_invalid_accumulation_count_is_rejected():
    ds = _data(n=4)
    try:
        _run_epoch(ds, batch_size=2, accumulation_steps=0)
    except ValueError as exc:
        assert "accumulation_steps" in str(exc)
    else:
        raise AssertionError("accumulation_steps=0 must be rejected")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:                              # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
