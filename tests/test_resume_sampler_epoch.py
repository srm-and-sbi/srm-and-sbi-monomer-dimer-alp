"""Continuation bookkeeping of `train_loop` and the trial knob of the schedule: a hot-restarted leg seeds the
DistributedSampler with the GLOBAL epoch (a continuation draws fresh shuffle orders, never leg 1's again), a fresh
run still counts from 0; `build_optimizer_scheduler` takes a recorded patience override (None = the configured
production value) and `train_loop` applies it to the warm-restart rebuild; the epoch line carries `lr_used`,
`lr_next`, `train_s` and `test_s`, and the readout reads `lr_next` as it read `lr`. CPU only, a tiny linear
estimator; no data bank.
"""
import io
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srm_and_sbi_monomer_dimer_alp.inference_support import (Topology, build_optimizer_scheduler,  # noqa: E402
                                                              save_resume_state, train_loop)
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS  # noqa: E402

sys.path.insert(0, str(REPO / "Script_Bank" / "Analysis"))
import SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Training_Log_Readout as readout  # noqa: E402


class _LinearEstimator(nn.Module):
    def __init__(self, n_pixels, theta_dim, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(n_pixels, theta_dim, generator=g) * 0.1)
        self.bias = nn.Parameter(torch.zeros(theta_dim))

    def loss(self, theta, condition):
        pred = condition.flatten(1) @ self.weight + self.bias
        return ((pred - theta) ** 2).sum(dim=1)

    def forward(self, theta, condition):
        return self.loss(theta, condition=condition)


class _RecordingSampler(DistributedSampler):
    """A DistributedSampler that records every epoch it is set to (what train_loop seeds the shuffle with)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_set = []

    def set_epoch(self, epoch):
        self.epochs_set.append(int(epoch))
        super().set_epoch(epoch)


def _data(n=12, frames=3, side=4, theta_dim=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.rand(n, frames, side, side, generator=g), torch.randn(n, theta_dim, generator=g))


def _run(dataset, tmp, epochs, resurrect=False, state_epoch=None, patience=None):
    """One train_loop call on a single CPU worker through a (single-replica) DistributedSampler, with the TRAIN set
    doubling as the TEST set so the schedule and the resurrect-state are exercised. Returns (sampler, printed text)."""
    est = _LinearEstimator(n_pixels=dataset.tensors[0][0].numel(), theta_dim=dataset.tensors[1].shape[1])
    sampler = _RecordingSampler(dataset, num_replicas=1, rank=0, shuffle=True, drop_last=False)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    optimizer, scheduler = build_optimizer_scheduler(est, 0.05, scheduler_patience=patience)
    topo = Topology(world_size=1, rank=0, local_rank=0, device=torch.device("cpu"), backend="CPU")
    state_path = Path(tmp) / "resume.pth"
    ckpt = Path(tmp) / "ckpt.pth"
    if state_epoch is not None:   # a prior leg's state: the loop must continue at state_epoch + 1
        torch.save(est.state_dict(), str(ckpt))
        save_resume_state(state_path, estimator=est, optimizer=optimizer, scheduler=scheduler, epoch=state_epoch,
                          optimum_loss_test=float("inf"), warm_restart_peak=0.05, epochs_at_min_lr=0)
    text = io.StringIO()
    with redirect_stdout(text):
        train_loop(estimator=est, model=est, train_loader=loader, val_loader=val_loader, optimizer=optimizer,
                   scheduler=scheduler, device=torch.device("cpu"), topo=topo, checkpoint_path=ckpt,
                   train_sampler=sampler, epochs=epochs, resurrect=resurrect, resurrect_state_path=state_path,
                   scheduler_patience=patience)
    return sampler, text.getvalue()


def _expected_order(dataset, epoch):
    s = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, drop_last=False)
    s.set_epoch(epoch)
    return list(iter(s))


def test_a_hot_restarted_leg_seeds_the_sampler_with_the_global_epochs():
    ds = _data()
    with tempfile.TemporaryDirectory() as tmp:
        sampler, text = _run(ds, tmp, epochs=2, resurrect=True, state_epoch=49)
    assert sampler.epochs_set == [50, 51], sampler.epochs_set          # the continuation, not [0, 1]
    assert "continuing at global epoch 50" in text
    # The orders the loop drew are the sampler's orders for THOSE epochs, and they differ from leg 1's first two.
    for epoch in (50, 51):
        sampler.set_epoch(epoch)
        assert list(iter(sampler)) == _expected_order(ds, epoch)
    assert _expected_order(ds, 50) != _expected_order(ds, 0) or _expected_order(ds, 51) != _expected_order(ds, 1)
    print("PASS: a hot-restarted leg seeds the sampler with the global epochs [50, 51]")


def test_a_fresh_run_still_counts_its_epochs_from_zero():
    ds = _data()
    with tempfile.TemporaryDirectory() as tmp:
        sampler, _ = _run(ds, tmp, epochs=3)
    assert sampler.epochs_set == [0, 1, 2], sampler.epochs_set
    print("PASS: a fresh run seeds the sampler with [0, 1, 2]")


def test_the_patience_override_reaches_the_scheduler_and_the_default_is_the_configured_value():
    est = _LinearEstimator(n_pixels=48, theta_dim=2)
    _, default = build_optimizer_scheduler(est, 0.05)
    _, trial = build_optimizer_scheduler(est, 0.05, scheduler_patience=3)
    assert default.patience == PARAMETERS.inference.training.scheduler_patience
    assert trial.patience == 3
    try:
        build_optimizer_scheduler(est, 0.05, scheduler_patience=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative patience must be refused")
    print("PASS: --scheduler-patience reaches the scheduler; None keeps the configured production value")


def test_the_epoch_line_carries_lr_used_lr_next_and_the_phase_times_and_the_readout_reads_them():
    ds = _data()
    with tempfile.TemporaryDirectory() as tmp:
        _, text = _run(ds, tmp, epochs=2, patience=0)
    lines = [ln for ln in text.splitlines() if ln.startswith("Epoch ")]
    assert len(lines) == 2, text
    fields = re.findall(r"(lr_used|lr_next|train_s|test_s)=([\d.e+-]+)", lines[0])
    assert [k for k, _ in fields] == ["lr_used", "lr_next", "train_s", "test_s"], lines[0]
    assert "lr=" not in lines[0].replace("lr_used=", "").replace("lr_next=", "")
    assert float(dict(fields)["lr_used"]) == 0.05                     # the initial LR was in force during epoch 1
    # epoch 2 starts with epoch 1's lr_next (the post-scheduler value), whatever the schedule decided
    next_1 = dict(re.findall(r"(lr_used|lr_next)=([\d.e+-]+)", lines[0]))["lr_next"]
    used_2 = dict(re.findall(r"(lr_used|lr_next)=([\d.e+-]+)", lines[1]))["lr_used"]
    assert float(used_2) == float(next_1), (lines[0], lines[1])
    assert readout.LR_RE.search(lines[0]).group("lr") == next_1       # the readout takes lr_next
    assert readout.LR_RE.search("... lr=1.28e-03 ...").group("lr") == "1.28e-03"   # and the old single field
    print("PASS: the epoch line carries lr_used / lr_next / train_s / test_s; the readout reads lr_next")


if __name__ == "__main__":
    test_a_hot_restarted_leg_seeds_the_sampler_with_the_global_epochs()
    test_a_fresh_run_still_counts_its_epochs_from_zero()
    test_the_patience_override_reaches_the_scheduler_and_the_default_is_the_configured_value()
    test_the_epoch_line_carries_lr_used_lr_next_and_the_phase_times_and_the_readout_reads_them()
