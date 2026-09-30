"""The embedding network (`Complex3DCNN`) and its encoder-screening settings (DETECTOR_WORKFLOW.md, the
encoder screening): the defaults reproduce the original network module for module and value for value;
every preset builds at every documented duration and returns the embedding shape; forward and backward
passes stay finite, constant feature maps under statistics pooling included; the receptive-field
arithmetic agrees with the implemented layers; an estimator artifact restores the selected architecture;
and the persisted estimators of record, when present on this machine, load under the new class unchanged.
CPU only; no training.
"""
import dataclasses
import glob
import importlib.util
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srm_and_sbi_monomer_dimer_alp import artifacts                                   # noqa: E402
from srm_and_sbi_monomer_dimer_alp.inference_network import Complex3DCNN, SPATIAL_POOLINGS  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import (NETWORK_PRESETS, PARAMETERS,   # noqa: E402
                                                            InferenceNetwork)

SCREENING_PRESETS = ("kernel7", "earlyconv", "statspool")
COMBINED_PRESETS = ("capacity256_kernel7_stats", "capacity256_earlyconv_stats")
DOCUMENTED_DURATIONS = {1.0: 50, 2.0: 100, 5.0: 250, 10.0: 500, 20.0: 1000}     # seconds -> frames at 50 FPS


def _network_kwargs(preset="baseline", n_frames=100, **overrides):
    """The runner's embedding arguments for a preset (parameterization defaults + the preset's fields)."""
    cfg = dataclasses.replace(PARAMETERS.inference.network, **NETWORK_PRESETS[preset]["network"])
    kw = dict(n_frames=n_frames, **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(InferenceNetwork)})
    kw.update(overrides)
    return kw


def _original_network_class():
    """`Complex3DCNN` as released in 0.1.26, from the frozen fixture beside this file."""
    path = Path(__file__).with_name("_legacy_inference_network_0_1_26.py")
    spec = importlib.util.spec_from_file_location("legacy_inference_network_0_1_26", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    legacy = inspect.signature(module.Complex3DCNN.__init__).parameters
    assert "first_spatial_kernel" not in legacy and "spatial_pooling" not in legacy, \
        "the frozen fixture must predate the settings"
    return module.Complex3DCNN


def test_the_defaults_reproduce_the_original_network():
    Original = _original_network_class()
    video = torch.rand(2, 100, 256, 256)
    for preset in ("baseline", "capacity256"):
        kw = _network_kwargs(preset)
        old_kw = {k: v for k, v in kw.items() if k not in ("first_spatial_kernel", "extra_spatial_convs",
                                                           "extra_spatial_conv_blocks", "spatial_pooling")}
        torch.manual_seed(1); old = Original(**old_kw)
        torch.manual_seed(1); new = Complex3DCNN(**kw)
        assert list(old.state_dict().keys()) == list(new.state_dict().keys())
        new.load_state_dict(old.state_dict())
        old.eval(); new.eval()
        with torch.no_grad():
            assert torch.equal(old(video), new(video)), preset
    # The original network's parameter count, as DETECTOR_WORKFLOW.md's capacity-test table states it.
    assert sum(p.numel() for p in Complex3DCNN(**_network_kwargs()).parameters()) == 691_392


def test_every_preset_builds_at_the_documented_durations_and_keeps_the_embedding_width():
    for preset in NETWORK_PRESETS:
        for seconds, n_frames in DOCUMENTED_DURATIONS.items():
            net = Complex3DCNN(**_network_kwargs(preset, n_frames=n_frames))
            width = net.feature_dim
            assert width == NETWORK_PRESETS[preset]["network"].get("start_channels", 8) * 2 ** 4, (preset, seconds)
            assert net.reduced_frames == {50: 50, 100: 100, 250: 124, 500: 100, 1000: 100}[n_frames]
        net = Complex3DCNN(**_network_kwargs(preset, n_frames=100)).eval()
        with torch.no_grad():
            out = net(torch.rand(2, 100, 256, 256))
        assert out.shape == (2, net.feature_dim) and torch.isfinite(out).all(), preset


def test_the_presets_change_one_encoder_setting_each_and_keep_the_flow():
    base = PARAMETERS.inference.network
    for preset in SCREENING_PRESETS:
        spec = NETWORK_PRESETS[preset]
        assert spec["flow"] == {}, preset
        changed = {k for k, v in spec["network"].items() if getattr(base, k) != v}
        assert len(changed) == 1, (preset, changed)             # exactly one encoder setting differs
        assert "start_channels" not in spec["network"] and "n_conv_layers" not in spec["network"]
    assert NETWORK_PRESETS["kernel7"]["network"] == {"first_spatial_kernel": 7}
    assert NETWORK_PRESETS["earlyconv"]["network"] == {"extra_spatial_convs": 1, "extra_spatial_conv_blocks": 2}
    assert NETWORK_PRESETS["statspool"]["network"] == {"spatial_pooling": "stats"}
    assert SPATIAL_POOLINGS[0] == "mean" == base.spatial_pooling


def test_the_combined_presets_build_on_capacity256_with_statistics_pooling():
    cap = NETWORK_PRESETS["capacity256"]
    for preset, change in (("capacity256_kernel7_stats", {"first_spatial_kernel": 7}),
                           ("capacity256_earlyconv_stats", {"extra_spatial_convs": 1, "extra_spatial_conv_blocks": 2})):
        spec = NETWORK_PRESETS[preset]
        assert spec["flow"] == cap["flow"] == {"hidden_features": 128, "num_transforms": 8, "num_blocks": 2,
                                               "dropout_probability": 0.1}
        assert spec["network"] == {**cap["network"], "spatial_pooling": "stats", **change}, preset
        net = Complex3DCNN(**_network_kwargs(preset)).eval()
        assert net.feature_dim == 256 and net.spatial_projection.in_features == 768
        assert sum(p.numel() for p in net.spatial_projection.parameters()) == 3 * 256 * 256 + 256 == 196_864
        with torch.no_grad():
            out = net(torch.rand(1, 100, 256, 256))
        assert out.shape == (1, 256) and torch.isfinite(out).all(), preset
    a = Complex3DCNN(**_network_kwargs("capacity256_kernel7_stats"))
    b = Complex3DCNN(**_network_kwargs("capacity256_earlyconv_stats"))
    assert [m.kernel_size for m in a.features if isinstance(m, nn.Conv3d)] == [(3, 7, 7)] + [(3, 3, 3)] * 4
    assert [m.kernel_size for m in b.features if isinstance(m, nn.Conv3d)] == [(3, 3, 3), (1, 3, 3), (3, 3, 3), (1, 3, 3),
                                                                                 (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    assert sum(p.numel() for p in a.parameters()) == 2_955_520 and sum(p.numel() for p in b.parameters()) == 2_965_264


def test_forward_and_backward_stay_finite_including_constant_maps_under_statistics_pooling():
    for preset in SCREENING_PRESETS:
        net = Complex3DCNN(**_network_kwargs(preset)).train()
        x = torch.rand(2, 100, 256, 256, requires_grad=True)
        out = net(x); out.sum().backward()
        assert torch.isfinite(out).all() and torch.isfinite(x.grad).all(), preset
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None), preset
    # A constant input gives constant feature maps: zero spatial variance, whose square root must not
    # produce a NaN gradient (STATS_EPS).
    net = Complex3DCNN(**_network_kwargs("statspool")).train()
    x = torch.zeros(1, 100, 256, 256, requires_grad=True)
    net(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_statistics_pooling_projects_the_three_statistics_back_to_the_channel_width():
    net = Complex3DCNN(**_network_kwargs("statspool")).eval()
    assert net.spatial_projection.in_features == 3 * net.feature_dim
    assert net.spatial_projection.out_features == net.feature_dim
    extra = sum(p.numel() for p in net.spatial_projection.parameters())
    assert extra == 3 * 128 * 128 + 128 == 49_280
    maps = torch.rand(2, net.feature_dim, 7, 8, 8)
    with torch.no_grad():
        reduced = net._reduce_space(maps)
        flat = maps.flatten(3)
        stats = torch.cat([flat.mean(3), torch.sqrt(flat.var(3, unbiased=False) + 1e-6), flat.amax(3)], dim=1)
        expected = net.spatial_projection(stats.transpose(1, 2)).transpose(1, 2)
    assert reduced.shape == (2, net.feature_dim, 7) and torch.allclose(reduced, expected)
    # The mean network reduces exactly as before.
    mean_net = Complex3DCNN(**_network_kwargs()).eval()
    assert not hasattr(mean_net, "spatial_projection")
    assert torch.equal(mean_net._reduce_space(maps), maps.mean(dim=(3, 4)))


def _encoder_of(flow):
    """The `Complex3DCNN` inside a built flow (sbi may wrap the embedding net, e.g. with a standardizer)."""
    return next(m for m in flow.embedding_net.modules() if isinstance(m, Complex3DCNN))


def _brute_force_field(net, axis):
    """Receptive field along one axis by propagating an index interval through the layer list."""
    lo, hi, jump = 0, 0, 1
    for module in net.features:
        if isinstance(module, (nn.Conv3d, nn.MaxPool3d)):
            k = module.kernel_size[axis] if isinstance(module.kernel_size, tuple) else module.kernel_size
            s = module.stride[axis] if isinstance(module.stride, tuple) else module.stride
            hi += (k - 1) * jump
            jump *= s
    return hi - lo + 1


def test_the_receptive_field_arithmetic_agrees_with_the_implemented_layers():
    expected = {"baseline": (94, 11), "capacity256": (94, 11), "kernel7": (98, 11),
                "earlyconv": (100, 11), "statspool": (94, 11),
                "capacity256_kernel7_stats": (98, 11), "capacity256_earlyconv_stats": (100, 11)}
    assert set(expected) == set(NETWORK_PRESETS)
    for preset, (space, time) in expected.items():
        net = Complex3DCNN(**_network_kwargs(preset))
        assert (net.spatial_receptive_field(), net.temporal_receptive_field()) == (space, time), preset
        assert _brute_force_field(net, 1) == space and _brute_force_field(net, 0) == time
    # The extra convolutions are spatial-only: the temporal reach is unchanged, and they sit before pooling.
    net = Complex3DCNN(**_network_kwargs("earlyconv"))
    convs = [m for m in net.features if isinstance(m, nn.Conv3d)]
    assert [c.kernel_size for c in convs] == [(3, 3, 3), (1, 3, 3), (3, 3, 3), (1, 3, 3),
                                              (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    kinds = [type(m).__name__ for m in net.features]
    first_pool = kinds.index("MaxPool3d")
    assert kinds[:first_pool].count("Conv3d") == 2                  # block 1: main conv + one extra before its pool
    # A wider first kernel keeps H and W (padding) and touches only the first block.
    net = Complex3DCNN(**_network_kwargs("kernel7"))
    convs = [m for m in net.features if isinstance(m, nn.Conv3d)]
    assert convs[0].kernel_size == (3, 7, 7) and convs[0].padding == (1, 3, 3)
    assert all(c.kernel_size == (3, 3, 3) for c in convs[1:])


def test_bad_encoder_arguments_are_rejected_before_any_layer_is_built():
    for bad in (dict(first_spatial_kernel=4), dict(first_spatial_kernel=1), dict(extra_spatial_convs=-1),
                dict(extra_spatial_convs=1, extra_spatial_conv_blocks=6), dict(spatial_pooling="max")):
        try:
            Complex3DCNN(**_network_kwargs(**bad))
        except ValueError as e:
            assert any(key in str(e) for key in bad), (bad, str(e))
        else:
            raise AssertionError(f"accepted {bad}")
    # The block count is checked only when extra convolutions are enabled: a one-block network with the
    # default `extra_spatial_conv_blocks=2` and no extra convolutions builds.
    net = Complex3DCNN(**_network_kwargs(n_conv_layers=1))
    assert net.feature_dim == 8 and net.extra_spatial_convs == 0
    try:
        Complex3DCNN(**_network_kwargs(n_conv_layers=1, extra_spatial_convs=1))
    except ValueError as e:
        assert "extra_spatial_conv_blocks" in str(e)
    else:
        raise AssertionError("accepted two extra-conv blocks in a one-block network")


def test_an_estimator_artifact_restores_the_selected_architecture():
    from sbi.neural_nets.net_builders import build_maf
    flow = PARAMETERS.inference.flow
    maf_args = dict(z_score_x=flow.z_score_x, z_score_y=flow.z_score_y, hidden_features=flow.hidden_features,
                    num_transforms=flow.num_transforms, num_blocks=flow.num_blocks,
                    dropout_probability=flow.dropout_probability, use_batch_norm=flow.use_batch_norm)
    video_shape = (4, 256, 256)                                    # short clip: the test is about the spec, not the data
    for preset in SCREENING_PRESETS + COMBINED_PRESETS:
        kw = _network_kwargs(preset, n_frames=4, temporal_target_frames=None)
        preset_flow = dataclasses.replace(flow, **NETWORK_PRESETS[preset]["flow"])
        maf_args = dict(z_score_x=preset_flow.z_score_x, z_score_y=preset_flow.z_score_y,
                        hidden_features=preset_flow.hidden_features, num_transforms=preset_flow.num_transforms,
                        num_blocks=preset_flow.num_blocks, dropout_probability=preset_flow.dropout_probability,
                        use_batch_norm=preset_flow.use_batch_norm)
        torch.manual_seed(3)
        estimator = build_maf(batch_x=torch.randn(2, 6), batch_y=torch.rand(2, *video_shape),
                              embedding_net=Complex3DCNN(**kw), **maf_args)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "estimator.npz"
            artifacts.save_estimator(estimator, embedding_args=kw, maf_args=maf_args, theta_dim=6,
                                     video_shape=video_shape, parameter_keys=list("abcdef"),
                                     prior_low=np.zeros(6), prior_high=np.ones(6), path=path)
            spec_all = artifacts.load_estimator_manifest(path)["rebuild_spec"]
            spec = spec_all["embedding_args"]
            for key in ("first_spatial_kernel", "extra_spatial_convs", "extra_spatial_conv_blocks", "spatial_pooling",
                        "start_channels"):
                assert spec[key] == kw[key], (preset, key)
            for key, value in maf_args.items():
                assert spec_all["maf_args"][key] == value, (preset, key)
            posterior = artifacts.load_estimator(str(path), device="cpu", expected_parameter_keys=list("abcdef"))
        rebuilt = posterior.posterior_estimator.embedding_net          # the flow's (possibly wrapped) embedding
        encoder = _encoder_of(posterior.posterior_estimator)
        assert (encoder.first_spatial_kernel, encoder.extra_spatial_convs, encoder.spatial_pooling) == (
            kw["first_spatial_kernel"], kw["extra_spatial_convs"], kw["spatial_pooling"])
        estimator.eval(); posterior.posterior_estimator.eval()
        video = torch.rand(1, *video_shape)
        with torch.no_grad():
            assert torch.allclose(estimator.embedding_net(video), rebuilt(video)), preset


def test_the_persisted_estimators_of_record_load_under_the_new_class_unchanged():
    """The baseline and CAP256 detector estimators, when this machine holds them: their rebuild
    specifications predate the encoder settings, so the defaults apply, and they embed a video."""
    root = PARAMETERS.machine.data_bank_root
    found = sorted(glob.glob(str(root / "Posit" / "*_DETECTOR_FAB_2S_50FPS_Estimator.npz"))
                   + glob.glob(str(root / "Posit" / "*_DETECTOR_FAB_2S_50FPS_CAP256_Estimator.npz")))
    if not found:
        print("  (no persisted detector estimator on this machine; skipped)")
        return
    from srm_and_sbi_monomer_dimer_alp.embedding_space_distance import embed_videos
    for path in found:
        spec = artifacts.load_estimator_manifest(path)["rebuild_spec"]["embedding_args"]
        assert "spatial_pooling" not in spec and "first_spatial_kernel" not in spec, path
        posterior = artifacts.load_estimator(path, device="cpu")
        net = _encoder_of(posterior.posterior_estimator)
        assert (net.first_spatial_kernel, net.extra_spatial_convs, net.spatial_pooling) == (3, 0, "mean")
        width = 256 if "CAP256" in path else 128
        assert net.feature_dim == width
        emb = embed_videos(posterior, [np.random.default_rng(0).integers(0, 255, (100, 256, 256), dtype=np.uint8)],
                           device="cpu", expected_frames=100)
        assert emb.shape == (1, width) and np.isfinite(emb).all(), path


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
