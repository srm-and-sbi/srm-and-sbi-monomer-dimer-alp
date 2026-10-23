"""The weights-initialized training phase (Inference ``--initialize-from-tag``): a stored estimator
artifact initializes a freshly built network whose embedding has the compiled key layout, with every
weight and buffer restored and the per-example losses of the Evaluation's rebuild reproduced; a
mismatched artifact is refused field by field; a phase starts with a fresh optimizer, its own learning
rate and patience, fresh plateau and warm-restart counters, and continues the source's global epoch
count; a resumed phase keeps its own schedule and refuses another patience while a state written
before the patience was recorded still resumes; the runner refuses an unsafe initialization before any
compute and states it in its dry run; the HPC wrappers and dispatchers forward the two settings.
CPU only, small networks, a temporary data bank; nothing is written to the machine's data bank.
"""
import dataclasses
import io
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

os.environ.setdefault("MACHINE_PROFILE", "mars_pc")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srm_and_sbi_monomer_dimer_alp import artifacts                                    # noqa: E402
from srm_and_sbi_monomer_dimer_alp import inference_runner as runner                   # noqa: E402
from srm_and_sbi_monomer_dimer_alp.inference_network import Complex3DCNN               # noqa: E402
from srm_and_sbi_monomer_dimer_alp.inference_support import (                          # noqa: E402
    ResumeStateMismatch, Topology, build_optimizer_scheduler, peek_resume_state, save_resume_state,
    source_loss_agreement, train_loop)
from srm_and_sbi_monomer_dimer_alp.parameterization import (                           # noqa: E402
    NETWORK_PRESETS, PARAMETERS, InferenceNetwork)
from srm_and_sbi_monomer_dimer_alp.workflow import detector_workflow                   # noqa: E402

VIDEO_SHAPE = (4, 256, 256)                       # a short clip: the tests are about state, not data
KEYS = ["mu_r", "sigma_r", "mu_pc", "sigma_pc", "prob_photo_bleach", "lambda_rate"]


# ---- small real estimators -----------------------------------------------------------------------

class _CompiledLayout(nn.Module):
    """Stands in for torch.compile's wrapper: the network sits under ``_orig_mod``, so the state_dict
    keys carry the ``_orig_mod.`` prefix exactly as in a training run, without compiling on CPU."""

    def __init__(self, inner):
        super().__init__()
        self._orig_mod = inner

    def forward(self, x):
        return self._orig_mod(x)


def _embedding_args():
    cfg = PARAMETERS.inference.network
    kw = dict(n_frames=VIDEO_SHAPE[0], **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(InferenceNetwork)})
    kw.update(temporal_target_frames=None, verbose=False)
    return kw


def _maf_args():
    flow = PARAMETERS.inference.flow
    return dict(z_score_x=flow.z_score_x, z_score_y=flow.z_score_y, hidden_features=flow.hidden_features,
                num_transforms=flow.num_transforms, num_blocks=flow.num_blocks,
                dropout_probability=flow.dropout_probability, use_batch_norm=flow.use_batch_norm)


def _estimator(seed, compiled_layout=False):
    """A detector-sized flow (six coordinates) on a short clip; its standardization is fitted from a
    seed-dependent dummy batch, as a fresh build fits its own."""
    from sbi.neural_nets.net_builders import build_maf
    torch.manual_seed(seed)
    embedding = Complex3DCNN(**_embedding_args())
    if compiled_layout:
        embedding = _CompiledLayout(embedding)
    return build_maf(batch_x=torch.randn(2, 6) * (1 + seed), batch_y=torch.rand(2, *VIDEO_SHAPE) * (1 + seed),
                     embedding_net=embedding, **_maf_args())


def _perturb(estimator, seed):
    """Make every floating weight and buffer, the batch-norm running statistics included, differ from
    a fresh build, so a test can tell the source's state from the initialized network's own."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in estimator.state_dict().items():
            if tensor.is_floating_point():
                noise = torch.rand(tensor.shape, generator=g)
                tensor.copy_(tensor * (1.0 + 0.5 * noise) if "running_var" in name else tensor + 0.1 * (noise - 0.5))
            elif name.endswith("num_batches_tracked"):
                tensor.fill_(1234)


def _save_source(estimator, path, *, condition="INLB", timing_label="2S_50FPS"):
    artifacts.save_estimator(
        estimator, embedding_args=_embedding_args(), maf_args=_maf_args(), theta_dim=6, video_shape=VIDEO_SHAPE,
        parameter_keys=KEYS, prior_low=np.zeros(6), prior_high=np.ones(6), path=path,
        metadata={"workflow": "detector", "condition": condition, "timing_label": timing_label,
                  "best_test_loss": -1.5, "epochs": 50})


def _compatibility_kwargs(**overrides):
    kw = dict(workflow="detector", condition="INLB", timing_label="2S_50FPS", parameter_keys=KEYS,
              lower_bound=[0.0] * 6, upper_bound=[1.0] * 6, embedding_args=_embedding_args(),
              maf_args=_maf_args(), theta_dim=6, video_shape=VIDEO_SHAPE)
    kw.update(overrides)
    return kw


def test_an_artifact_initializes_a_compiled_layout_network_with_every_weight_and_buffer():
    source = _estimator(seed=1)
    _perturb(source, seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "source.npz"
        _save_source(source, path)
        stored, manifest, low, high = artifacts.read_estimator_artifact(path)
        assert artifacts.initialization_mismatches(manifest, low, high, **_compatibility_kwargs()) == []
        fresh = _estimator(seed=2, compiled_layout=True)
        assert any("_orig_mod." in key for key in fresh.state_dict())
        # the fresh build fitted its own standardization: it differs from the stored one before loading
        before = artifacts.standardization_relative_differences(artifacts.strip_compile_prefix(fresh.state_dict()), stored)
        assert len(before) == 4 and min(before.values()) > 1e-3, before        # theta shift/scale + video mean/std
        artifacts.load_artifact_weights(fresh, stored)
        loaded = artifacts.strip_compile_prefix(fresh.state_dict())
        assert set(loaded) == set(stored)
        assert all(torch.equal(loaded[key], stored[key]) for key in stored)          # bitwise, buffers included
        assert max(artifacts.standardization_relative_differences(loaded, stored).values()) == 0.0
        running = [key for key in stored if key.endswith("running_var")]
        assert running and all(torch.equal(loaded[key], stored[key]) for key in running)
        # the initialized network reproduces the Evaluation's rebuild of the source, example by example
        g = torch.Generator().manual_seed(5)
        examples = [(torch.rand(VIDEO_SHAPE, generator=g), torch.rand(6, generator=g)) for _ in range(3)]
        reference = artifacts.load_estimator(str(path), device="cpu").posterior_estimator
        fresh.train()
        diff, ours, theirs = source_loss_agreement(fresh, reference, examples, torch.device("cpu"), n_examples=3)
        assert fresh.training, "the training mode is restored after the check"
        assert diff < 1e-4 and np.isfinite(ours).all() and len(ours) == 3, (diff, ours, theirs)
        # an uninitialized network does not
        other = _estimator(seed=3, compiled_layout=True)
        diff_other, _, _ = source_loss_agreement(other, reference, examples, torch.device("cpu"), n_examples=3)
        assert diff_other > runner.INITIALIZATION_AGREEMENT_TOLERANCE, diff_other
    print("PASS: an artifact initializes a compiled-layout network with every weight and buffer; losses reproduced")


def test_a_mismatched_artifact_is_refused_field_by_field():
    source = _estimator(seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "source.npz"
        _save_source(source, path)
        stored, manifest, low, high = artifacts.read_estimator_artifact(path)
    cases = {
        "condition": dict(condition="FAB"),
        "workflow": dict(workflow="biology"),
        "timing_label": dict(timing_label="5S_50FPS"),
        "parameter keys": dict(parameter_keys=list(reversed(KEYS))),
        "prior lower bounds": dict(lower_bound=[0.0] * 5 + [0.1]),
        "prior upper bounds": dict(upper_bound=[1.0] * 5 + [2.0]),
        "embedding arguments": dict(embedding_args={**_embedding_args(), "first_spatial_kernel": 5}),
        "flow arguments": dict(maf_args={**_maf_args(), "hidden_features": 64}),
        "theta width": dict(theta_dim=7),
        "video shape": dict(video_shape=(8, 256, 256)),
    }
    for field, override in cases.items():
        problems = artifacts.initialization_mismatches(manifest, low, high, **_compatibility_kwargs(**override))
        assert len(problems) == 1 and problems[0].startswith(field), (field, problems)
    # the console-only verbose flag is not part of the architecture
    assert artifacts.initialization_mismatches(
        manifest, low, high, **_compatibility_kwargs(embedding_args={**_embedding_args(), "verbose": True})) == []
    # a state_dict that does not fit the network is refused before anything is loaded
    net = _estimator(seed=2, compiled_layout=True)
    missing = {k: v for k, v in stored.items() if not k.endswith("running_mean")}
    for bad, word in ((missing, "missing"), ({**stored, "extra.weight": torch.zeros(1)}, "unexpected")):
        try:
            artifacts.load_artifact_weights(net, bad)
        except ValueError as exc:
            assert word in str(exc), exc
        else:
            raise AssertionError(f"accepted a state_dict with a {word} key")
    key = next(k for k in stored if k.endswith("running_mean"))
    try:
        artifacts.load_artifact_weights(net, {**stored, key: torch.zeros(stored[key].numel() + 1)})
    except ValueError as exc:
        assert "shape or dtype" in str(exc)
    else:
        raise AssertionError("accepted a state_dict with a wrong shape")
    print("PASS: a mismatched artifact is refused field by field; a non-fitting state_dict is refused")


# ---- the phase in the training loop ----------------------------------------------------------------

class _LinearEstimator(nn.Module):
    def __init__(self, n_pixels, theta_dim, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(n_pixels, theta_dim, generator=g) * 0.1)
        self.bias = nn.Parameter(torch.zeros(theta_dim))

    def loss(self, theta, condition):
        return ((condition.flatten(1) @ self.weight + self.bias - theta) ** 2).sum(dim=1)

    def forward(self, theta, condition):
        return self.loss(theta, condition=condition)


class _RecordingSampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_set = []

    def set_epoch(self, epoch):
        self.epochs_set.append(int(epoch))
        super().set_epoch(epoch)


def _data(n=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.rand(n, 3, 4, 4, generator=g), torch.randn(n, 2, generator=g))


def _loop(est, optimizer, scheduler, tmp, *, epochs, start_epoch=0, resurrect=False, meta=None, patience=None):
    ds = _data()
    sampler = _RecordingSampler(ds, num_replicas=1, rank=0, shuffle=True, drop_last=False)
    text = io.StringIO()
    with redirect_stdout(text):
        train_loop(estimator=est, model=est, train_loader=DataLoader(ds, batch_size=4, sampler=sampler),
                   val_loader=DataLoader(ds, batch_size=4, shuffle=False), optimizer=optimizer, scheduler=scheduler,
                   device=torch.device("cpu"),
                   topo=Topology(world_size=1, rank=0, local_rank=0, device=torch.device("cpu"), backend="CPU"),
                   checkpoint_path=Path(tmp) / "phase_ckpt.pth", train_sampler=sampler, epochs=epochs,
                   resurrect=resurrect, resurrect_state_path=Path(tmp) / "phase_resume.pth", resume_meta=meta,
                   scheduler_patience=patience, start_epoch=start_epoch)
    return sampler, text.getvalue()


def _source_state(tmp):
    """The source run's latest state after 100 global epochs: learning rate at the floor, an exhausted
    warm-restart sawtooth, a plateau tally and a long optimizer history -- none of it may reach the
    new phase."""
    src = _LinearEstimator(48, 2, seed=7)
    optimizer, scheduler = build_optimizer_scheduler(src, 1e-5)
    loss = src.loss(torch.zeros(4, 2), torch.rand(4, 3, 4, 4)).mean()
    loss.backward()
    optimizer.step()
    for state in optimizer.state.values():
        state["step"] = torch.tensor(9999.0)
    scheduler.num_bad_epochs, scheduler.best = 5, -42.0
    path = Path(tmp) / "source_resume.pth"
    save_resume_state(path, estimator=src, optimizer=optimizer, scheduler=scheduler, epoch=99,
                      optimum_loss_test=-42.0, warm_restart_peak=2e-5, epochs_at_min_lr=1,
                      resume_meta={"timing_label": "2S_50FPS_SRC", "parameter_keys": ["a", "b"]})
    return src, path


def _meta(patience):
    return {"timing_label": "2S_50FPS_PHASE", "parameter_keys": ["a", "b"], "scheduler_patience": patience}


def test_a_phase_starts_fresh_and_continues_the_global_epoch_count():
    with tempfile.TemporaryDirectory() as tmp:
        src, src_state = _source_state(tmp)
        peek = peek_resume_state(src_state)
        assert peek["epoch"] == 99 and peek["resume_meta"]["timing_label"] == "2S_50FPS_SRC"
        est = _LinearEstimator(48, 2, seed=8)
        est.load_state_dict(src.state_dict())                   # the source's weights, nothing else
        optimizer, scheduler = build_optimizer_scheduler(est, 3.2e-4, scheduler_patience=3)
        assert not optimizer.state and scheduler.patience == 3 and scheduler.num_bad_epochs == 0
        sampler, text = _loop(est, optimizer, scheduler, tmp, epochs=2, start_epoch=peek["epoch"] + 1,
                              meta=_meta(3), patience=3)
        assert sampler.epochs_set == [100, 101], sampler.epochs_set
        lines = [ln for ln in text.splitlines() if ln.startswith("Epoch ")]
        assert lines[0].startswith("Epoch 1|2 (global 101)") and "lr_used=3.20e-04" in lines[0], lines[0]
        state = torch.load(str(Path(tmp) / "phase_resume.pth"), weights_only=False)
        assert state["epoch"] == 101 and state["warm_restart_peak"] == 3.2e-4 and state["epochs_at_min_lr"] == 0
        assert state["resume_meta"]["scheduler_patience"] == 3
        steps = {float(s["step"]) for s in state["optimizer"]["state"].values()}
        assert steps == {6.0}, steps                            # 3 steps per epoch x 2 epochs; no 9999 imported
        assert state["scheduler"]["patience"] == 3 and state["scheduler"]["best"] != -42.0
        assert np.isfinite(state["optimum_loss_test"]) and state["optimum_loss_test"] != -42.0
    print("PASS: a phase starts with a fresh optimizer, LR, patience and counters and continues at global epoch 101")


def test_a_resumed_phase_keeps_its_schedule_and_refuses_another_patience():
    with tempfile.TemporaryDirectory() as tmp:
        src, src_state = _source_state(tmp)
        est = _LinearEstimator(48, 2, seed=8)
        est.load_state_dict(src.state_dict())
        optimizer, scheduler = build_optimizer_scheduler(est, 3.2e-4, scheduler_patience=3)
        _, text = _loop(est, optimizer, scheduler, tmp, epochs=2, start_epoch=100, meta=_meta(3), patience=3)
        last_next = re.findall(r"lr_next=([\d.e+-]+)", text)[-1]
        # resumed with the production peak passed by mistake: the phase's own schedule wins
        est2 = _LinearEstimator(48, 2, seed=9)
        opt2, sched2 = build_optimizer_scheduler(est2, 1.28e-3, scheduler_patience=3)
        sampler, text2 = _loop(est2, opt2, sched2, tmp, epochs=1, resurrect=True, meta=_meta(3), patience=3)
        assert "HOT RESTART" in text2 and sampler.epochs_set == [102], (sampler.epochs_set, text2[-400:])
        used = re.findall(r"lr_used=([\d.e+-]+)", text2)[0]
        assert float(used) == float(last_next) and float(used) != 1.28e-3, (used, last_next)
        assert sched2.patience == 3
        # another patience is refused before anything is restored
        est3 = _LinearEstimator(48, 2, seed=10)
        opt3, sched3 = build_optimizer_scheduler(est3, 3.2e-4)
        try:
            _loop(est3, opt3, sched3, tmp, epochs=1, resurrect=True, meta=_meta(1))
        except ResumeStateMismatch as exc:
            assert "patience 3" in str(exc) and "--scheduler-patience 3" in str(exc), exc
        else:
            raise AssertionError("resumed a patience-3 phase under patience 1")
        # a state written before the patience was recorded still resumes under any patience
        legacy = torch.load(str(Path(tmp) / "phase_resume.pth"), weights_only=False)
        legacy["resume_meta"].pop("scheduler_patience")
        torch.save(legacy, str(Path(tmp) / "phase_resume.pth"))
        est4 = _LinearEstimator(48, 2, seed=11)
        opt4, sched4 = build_optimizer_scheduler(est4, 3.2e-4)
        _, text4 = _loop(est4, opt4, sched4, tmp, epochs=1, resurrect=True, meta=_meta(1))
        assert "HOT RESTART" in text4
    print("PASS: a resumed phase keeps its own LR and patience; another patience is refused; legacy states resume")


# ---- the runner's refusals (dry run in a temporary data bank) ---------------------------------------

def _run(argv, bank):
    machine = dataclasses.replace(PARAMETERS.machine, data_bank_root=bank, scratch_data_bank_root=None)
    saved = runner.PARAMETERS
    runner.PARAMETERS = dataclasses.replace(PARAMETERS, machine=machine)
    try:
        args = runner.build_inference_parser().parse_args(argv)
        text = io.StringIO()
        with redirect_stdout(text):
            runner.run_inference(detector_workflow(), args)
        return text.getvalue()
    finally:
        runner.PARAMETERS = saved


def test_the_runner_refuses_an_unsafe_initialization_before_any_compute():
    cfg = detector_workflow()
    with tempfile.TemporaryDirectory() as tmp:
        bank = Path(tmp)
        paths = cfg.paths.with_condition("INLB")
        source = paths.estimator_path(bank, paths.product_label("2S_50FPS", "SRC"))
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"placeholder")                      # the dry run only probes existence
        base = ["--condition", "INLB", "--total-time-seconds", "2.0", "--tasks", "2", "--test-tasks", "1",
                "--initialize-from-tag", "SRC", "--learning-rate", "3.2e-4", "--scheduler-patience", "3"]
        ok = _run(base + ["--artifact-tag", "PHASE", "--dry-run"], bank)
        assert "--initialize-from-tag: SRC" in ok and "REFUSED" not in ok, ok[-1500:]
        assert re.search(r"source estimator : \S+_DETECTOR_INLB_2S_50FPS_SRC_Estimator\.npz\s+\[OK\]", ok)
        assert "absent: the global epoch count restarts at 0" in ok
        refusals = {
            "cannot be combined with --resurrect": base + ["--artifact-tag", "PHASE", "--resurrect"],
            "needs --artifact-tag": base,
            "must differ from --initialize-from-tag": base + ["--artifact-tag", "SRC"],
            "needs --test-tasks > 0": [a if a != "1" else "0" for a in base] + ["--artifact-tag", "PHASE"],
            "source estimator missing": [a if a != "SRC" else "NONE" for a in base] + ["--artifact-tag", "PHASE"],
        }
        for reason, argv in refusals.items():
            out = _run(argv + ["--dry-run"], bank)
            assert f"REFUSED: " in out and reason in out and "would be REFUSED" in out, (reason, out[-1500:])
            assert "all inputs present" not in out
        # the new phase already has products: resume it instead of re-initializing it
        target = paths.resurrect_state_path(bank, paths.product_label("2S_50FPS", "PHASE"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"placeholder")
        out = _run(base + ["--artifact-tag", "PHASE", "--dry-run"], bank)
        assert "already has products" in out and "resume it with --resurrect alone" in out, out[-1500:]
        # outside the dry run a refusal raises before any compute (no Labor directory is created)
        try:
            _run(base + ["--artifact-tag", "PHASE"], bank)
        except ValueError as exc:
            assert "--initialize-from-tag refused" in str(exc) and "already has products" in str(exc)
        else:
            raise AssertionError("an unsafe initialization was not refused")
        assert sorted(p.name for p in (bank / "Labor").iterdir()) == [target.name]
    print("PASS: the runner refuses an unsafe initialization in its dry run and before any compute")


def test_the_hpc_wrappers_and_dispatchers_forward_the_patience_and_the_initialization():
    hpc = REPO / "Script_Bank" / "HPC"
    for name in ("SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Inference.sh", "SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Inference.sh"):
        text = (hpc / name).read_text()
        assert re.search(r'PATIENCE_ARG=\(--scheduler-patience "\$SCHEDULER_PATIENCE"\)', text), name
        assert re.search(r'INIT_ARG=\(--initialize-from-tag "\$INIT_FROM_TAG"\)', text), name
        assert '"${PATIENCE_ARG[@]}"' in text and '"${INIT_ARG[@]}"' in text, name
    for name in ("SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Submit.sh", "SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Submit.sh"):
        text = (hpc / name).read_text()
        assert "_add SCHEDULER_PATIENCE" in text and "_add INIT_FROM_TAG" in text, name
    print("PASS: the HPC wrappers and dispatchers forward SCHEDULER_PATIENCE and INIT_FROM_TAG")


if __name__ == "__main__":
    test_an_artifact_initializes_a_compiled_layout_network_with_every_weight_and_buffer()
    test_a_mismatched_artifact_is_refused_field_by_field()
    test_a_phase_starts_fresh_and_continues_the_global_epoch_count()
    test_a_resumed_phase_keeps_its_schedule_and_refuses_another_patience()
    test_the_runner_refuses_an_unsafe_initialization_before_any_compute()
    test_the_hpc_wrappers_and_dispatchers_forward_the_patience_and_the_initialization()
