"""The theta standardization of the flow: what ``build_maf`` fixes from ``batch_x`` and never
refits, the biology's 'independent' choice fitted from all TRAIN task-0 labels, the detector's
unchanged 'structured' default, and the artifact round trip of both modes.

Run directly from the repository root::

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python tests/test_theta_standardization.py
"""
import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import zarr

os.environ.setdefault("MACHINE_PROFILE", "mars_pc")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sbi.neural_nets.net_builders import build_maf                                     # noqa: E402

from srm_and_sbi_monomer_dimer_alp import artifacts, parameterization as par            # noqa: E402
from srm_and_sbi_monomer_dimer_alp.inference_network import Complex3DCNN                 # noqa: E402
from srm_and_sbi_monomer_dimer_alp.inference_runner import _inference_spec               # noqa: E402
from srm_and_sbi_monomer_dimer_alp.inference_support import VideoDataset, theta_standardization_batch  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.io import theta_set_schema, write_theta_set           # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import InferenceNetwork, NETWORK_PRESETS, PARAMETERS  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow, detector_workflow   # noqa: E402

N_FRAMES, SIDE = 8, 32


class _Flat(torch.nn.Module):
    """A tiny embedding stand-in: flattens the video and projects it to eight features."""
    def __init__(self, n_in):
        super().__init__()
        self.lin = torch.nn.Linear(n_in, 8)

    def forward(self, x):
        return self.lin(x.reshape(x.shape[0], -1))


def _maf_args(z_score_x):
    flow = PARAMETERS.inference.flow
    return dict(z_score_x=z_score_x, z_score_y=flow.z_score_y, hidden_features=flow.hidden_features,
                num_transforms=flow.num_transforms, num_blocks=flow.num_blocks,
                dropout_probability=flow.dropout_probability, use_batch_norm=flow.use_batch_norm)


def _x_buffers(net):
    """The standardization buffers of the PARAMETER side (the flow's first transform)."""
    named = {name: buf for name, buf in net.named_buffers()}
    shift = [v for k, v in named.items() if k.endswith("_shift") and "embedding" not in k and "_transforms.0." in k]
    scale = [v for k, v in named.items() if k.endswith("_scale") and "embedding" not in k and "_transforms.0." in k]
    assert len(shift) == 1 and len(scale) == 1, sorted(named)
    return shift[0], scale[0]


def _y_buffers(net):
    named = {name: buf for name, buf in net.named_buffers()}
    return {k: v.clone() for k, v in named.items() if "embedding" in k and (k.endswith("_shift") or k.endswith("_scale"))}


def _build(theta_batch, z_score_x, video_batch=None):
    video_batch = torch.rand(2, N_FRAMES, SIDE, SIDE) if video_batch is None else video_batch
    torch.manual_seed(0)
    return build_maf(batch_x=theta_batch, batch_y=video_batch, embedding_net=_Flat(N_FRAMES * SIDE * SIDE),
                     **_maf_args(z_score_x))


def test_structured_pools_one_scalar_and_independent_fits_each_coordinate():
    """This pins the sbi semantics the stage relies on (sbi 0.26.1)."""
    rng = np.random.default_rng(1)
    theta = torch.tensor(np.column_stack([rng.uniform(3, 5, 500), rng.uniform(0, 1, 500), rng.uniform(-3, 1, 500)]),
                         dtype=torch.float32)
    # the buffers are the affine parameters of the standardizing transform: scale = 1 / std,
    # shift = -mean / std. 'structured': ONE mean over every entry and the mean of the PER-SAMPLE
    # standard deviations; 'independent': per-coordinate mean and standard deviation.
    shift, scale = _x_buffers(_build(theta, "structured"))
    assert shift.shape == () and scale.shape == ()
    std_structured = theta.std(dim=1).mean()
    assert torch.isclose(scale, 1.0 / std_structured, atol=1e-4)
    assert torch.isclose(shift, -theta.mean() / std_structured, atol=1e-4)
    assert not torch.isclose(scale, 1.0 / theta.std(dim=0).mean(), atol=1e-3), "not the per-coordinate spread"
    shift, scale = _x_buffers(_build(theta, "independent"))
    assert shift.shape == (3,) and scale.shape == (3,)
    assert torch.allclose(scale, 1.0 / theta.std(dim=0), atol=1e-4)
    assert torch.allclose(shift, -theta.mean(dim=0) / theta.std(dim=0), atol=1e-4)


def test_the_defaults_and_the_detector_stay_structured_and_the_biology_declares_independent():
    assert PARAMETERS.inference.flow.z_score_x == "structured" and PARAMETERS.inference.flow.z_score_y == "structured"
    assert par.THETA_STANDARDIZATION == "independent"
    b, d = biology_workflow(), detector_workflow()
    assert _inference_spec(d, "FAB").theta_standardization is None
    assert _inference_spec(b, "FAB").theta_standardization == "independent"
    assert _inference_spec(b, "INLB").theta_standardization == "independent"
    for preset in NETWORK_PRESETS.values():
        assert "z_score_x" not in preset["flow"] and "z_score_y" not in preset["flow"], "no preset touches the standardization"
    # the runner's override differs from the flow default only in z_score_x
    flow = PARAMETERS.inference.flow
    overridden = dataclasses.replace(flow, z_score_x=par.THETA_STANDARDIZATION)
    assert {f.name for f in dataclasses.fields(flow) if getattr(flow, f.name) != getattr(overridden, f.name)} == {"z_score_x"}


def test_the_biology_statistics_come_from_the_train_labels_never_from_two_rows():
    P = PARAMETERS.paths.with_condition("INLB")
    table = par.parameterization_of("INLB")
    rng = np.random.default_rng(2)
    n = 2000
    flow = rng.uniform(par.theta_lower_bound("INLB"), par.theta_upper_bound("INLB"), size=(n, len(table)))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_theta_set(P.theta_set_path(0, root, "1S_50FPS", True, "TRAIN"), par.to_physical(flow, table),
                        theta_set_schema(table, condition="INLB", timing_label="1S_50FPS", generator="rds"))
        video_path = P.video_set_path(0, root, "1S_50FPS", True, "TRAIN")
        zarr.open(str(video_path), mode="w", shape=(n, N_FRAMES, SIDE, SIDE), chunks=(1, N_FRAMES, SIDE, SIDE),
                  dtype=np.uint8)[:2] = 3
        ds = VideoDataset(tasks=1, data_bank_root=root, timing_label="1S_50FPS", compress=True, augment=False,
                          split="TRAIN", paths=P, parameterization=table)
        batch = theta_standardization_batch(ds)
        assert batch.shape == (n, 12) and batch.dtype == torch.float32
        assert np.allclose(batch.numpy(), flow, atol=1e-5), "estimator coordinates, by the one conversion rule"
        assert torch.equal(batch, theta_standardization_batch(ds)), "deterministic"
        widths = np.array(par.theta_upper_bound("INLB")) - np.array(par.theta_lower_bound("INLB"))
        centers = (np.array(par.theta_upper_bound("INLB")) + np.array(par.theta_lower_bound("INLB"))) / 2
        shift, scale = _x_buffers(_build(batch, "independent"))
        std_fitted = 1.0 / scale.numpy()                      # the buffers are the affine parameters
        assert np.allclose(std_fitted, widths / np.sqrt(12.0), rtol=0.05), "per-coordinate std of a uniform box"
        assert np.allclose(-shift.numpy() / scale.numpy(), centers, atol=0.05)
        # the two-row dummy would not do: its per-coordinate scales are arbitrary
        two = batch[:2]
        _, scale_two = _x_buffers(_build(two, "independent"))
        ratio = (1.0 / scale_two.numpy()) / (widths / np.sqrt(12.0))
        assert ratio.min() < 0.5 or ratio.max() > 2.0
        # the video-side (y) buffers do not depend on the theta batch
        y_full = _y_buffers(_build(batch, "independent", torch.rand(2, N_FRAMES, SIDE, SIDE).mul(0).add(0.25)))
        y_two = _y_buffers(_build(two, "independent", torch.rand(2, N_FRAMES, SIDE, SIDE).mul(0).add(0.25)))
        assert y_full.keys() == y_two.keys() and all(torch.equal(y_full[k], y_two[k]) for k in y_full)


def test_stored_estimators_rebuild_with_their_saved_mode_and_an_edited_mode_fails_loudly():
    network = PARAMETERS.inference.network
    embedding_args = dict(n_frames=N_FRAMES, **{f.name: getattr(network, f.name) for f in dataclasses.fields(InferenceNetwork)})
    video_shape = (N_FRAMES, 256, 256)
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        for cond, mode in (("INLB", "independent"), ("FAB", "structured")):
            table = par.parameterization_of(cond)
            flow = torch.tensor(rng.uniform(par.theta_lower_bound(cond), par.theta_upper_bound(cond), size=(64, len(table))),
                                dtype=torch.float32)
            torch.manual_seed(0)
            est = build_maf(batch_x=flow, batch_y=torch.rand(2, *video_shape), embedding_net=Complex3DCNN(**embedding_args),
                            **_maf_args(mode))
            shift, scale = _x_buffers(est)
            assert shift.shape == ((len(table),) if mode == "independent" else ())
            path = Path(tmp) / f"{cond}_estimator.npz"
            artifacts.save_estimator(est, embedding_args=embedding_args, maf_args=_maf_args(mode), theta_dim=len(table),
                                     video_shape=video_shape, parameter_keys=par.parameter_keys_of(cond),
                                     prior_low=par.theta_lower_bound(cond), prior_high=par.theta_upper_bound(cond),
                                     path=path, metadata={"condition": cond, "theta_standardization": mode})
            posterior = artifacts.load_estimator(path, expected_parameter_keys=par.parameter_keys_of(cond))
            shift2, scale2 = _x_buffers(posterior.posterior_estimator)
            assert torch.equal(shift.cpu(), shift2) and torch.equal(scale.cpu(), scale2), "the fitted statistics survive the round trip"
            other = "FAB" if cond == "INLB" else "INLB"
            try:
                artifacts.load_estimator(path, expected_parameter_keys=par.parameter_keys_of(other))
            except Exception as exc:  # noqa: BLE001 -- the schema guard raises before any rebuild
                assert "parameter" in str(exc).lower()
            else:
                raise AssertionError("an estimator of the other condition was loaded")
            # an artifact whose manifest claims the other mode cannot restore these buffers
            data = dict(np.load(path, allow_pickle=False))
            manifest = json.loads(str(data["manifest"]))
            manifest["rebuild_spec"]["maf_args"]["z_score_x"] = "structured" if mode == "independent" else "independent"
            data["manifest"] = np.array(json.dumps(manifest))
            edited = Path(tmp) / f"{cond}_edited.npz"
            np.savez_compressed(edited, **data)
            # torch restores the standardization buffers silently whatever their shape, so the
            # artifact loader checks the stored buffers against the manifest's mode itself
            try:
                artifacts.load_estimator(edited)
            except ValueError as exc:
                assert "standardization" in str(exc), str(exc)[:200]
            else:
                raise AssertionError("an edited standardization mode loaded silently")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)")
