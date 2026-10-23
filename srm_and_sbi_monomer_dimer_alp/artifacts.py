"""Self-describing, version-portable estimator artifact (see the nuisance and artifact design in DETECTOR_WORKFLOW.md).

A pickled `DirectPosterior` bakes in `torch._dynamo` internals (the embedding is
`torch.compile`d before training) whose private layouts change between torch
releases, so it is torch-version-locked. This module persists an estimator as three
**separable** components instead, so it reconstructs under whatever torch version is
loading it. It is the sole persisted estimator format for both workflows.

Three components (one `.npz`):
  (a) a compile-stripped `state_dict` — tensor weights only, the `_orig_mod.`
      compile prefix removed, serialized `weights_only`-loadably;
  (b) a rebuild spec — the `Complex3DCNN` args, the `build_maf` args, and the
      `theta_dim`/`video_shape` hints `build_maf` needs for shape inference,
      captured exactly as the caller constructed the estimator (Approach 1: the
      caller supplies them, so there is no fragile introspection and no canonical
      edit);
  (c) a metadata block — the ordered `parameter_keys` mapped to prior bounds,
      plus provenance (torch version, weights checksum, and caller metadata such
      as the timing label, best test loss, and source conditions/accessions).

`load_estimator` rebuilds the `Complex3DCNN` **uncompiled** under the current
torch, replays `build_maf` on zero-cost dummy batches of the stored shapes, loads
the stripped weights (`weights_only=True`), verifies the checksum, and attaches a
fresh device-aware prior — never deserializing any torch-internal or compiled
code. It returns a `DirectPosterior`, the same type every canonical consumer
accepts. The rebuild path is exactly Complex3DCNN + build_maf + load_state_dict,
but on an uncompiled embedding — so no re-`torch.compile` is needed.
"""

import hashlib
import io
import json

import numpy as np
import torch
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.net_builders import build_maf
from sbi.utils import BoxUniform

from .inference_network import Complex3DCNN

ARTIFACT_FORMAT_VERSION = 1
_COMPILE_PREFIX = "_orig_mod."


def strip_compile_prefix(state_dict: dict) -> dict:
    """Return a copy of ``state_dict`` with the `torch.compile` ``_orig_mod.``
    prefix removed from every key (so the weights match an uncompiled rebuild)."""
    return {key.replace(_COMPILE_PREFIX, ""): value for key, value in state_dict.items()}


def _rebuild_estimator(embedding_args, maf_args, theta_dim, video_shape, device):
    """Reconstruct the (uncompiled) MAF estimator from the rebuild spec."""
    embedding = Complex3DCNN(**embedding_args)
    # Dummy batches size the flow (build_maf infers dims from them); their values
    # are irrelevant — the trained standardizing stats are restored by
    # load_state_dict. A fixed seed keeps the reconstruction deterministic and
    # gives the standardizing net non-degenerate variance during construction.
    generator = torch.Generator().manual_seed(0)
    batch_x = torch.randn((2, theta_dim), generator=generator)
    batch_y = torch.randn((2, *video_shape), generator=generator)
    estimator = build_maf(batch_x=batch_x, batch_y=batch_y,
                          embedding_net=embedding, **maf_args)
    return estimator.to(device)


def _standardization_buffers(state_dict: dict) -> dict:
    """The flow's theta (x) standardization buffers of a compile-stripped ``state_dict``: the keys
    ending in ``_transforms.0._shift`` / ``_transforms.0._scale`` outside the embedding net
    (the first transform of the MAF is the standardizing one sbi inserts under ``z_score_x``)."""
    return {key: value for key, value in state_dict.items()
            if "embedding" not in key
            and (key.endswith("_transforms.0._shift") or key.endswith("_transforms.0._scale"))}


def assert_standardization_consistent(state_dict: dict, maf_args: dict, theta_dim: int, *,
                                      source: str = "estimator") -> None:
    """Refuse a state_dict whose theta-standardization buffers disagree with ``maf_args``.

    ``torch.nn.Module.load_state_dict`` restores the standardizing buffers SILENTLY even when the
    rebuilt mode differs (a ``(D,)`` buffer loads into a ``()`` buffer without an error), so an
    artifact whose manifest ``z_score_x`` disagrees with its stored buffers would rebuild wrongly.
    The rule: ``structured`` stores 0-d buffers (one pooled shift and scale); ``independent``
    stores ``(theta_dim,)`` buffers (one per coordinate); ``none`` / ``None`` stores no such
    buffer. Checked before an artifact is written and before a loaded state_dict is restored.
    """
    mode = maf_args.get("z_score_x", "independent")
    buffers = _standardization_buffers(state_dict)
    shapes = {key: tuple(value.shape) for key, value in buffers.items()}
    if mode in (None, "none"):
        if buffers:
            raise ValueError(f"{source}: standardization mode of the manifest does not match the stored "
                             f"buffers: z_score_x={mode!r} but the state_dict carries {shapes}.")
        return
    want = () if mode == "structured" else (int(theta_dim),)
    if mode not in ("structured", "independent"):
        raise ValueError(f"{source}: unknown z_score_x mode {mode!r}.")
    if not buffers:
        raise ValueError(f"{source}: standardization mode of the manifest does not match the stored "
                         f"buffers: z_score_x={mode!r} but no standardization buffer is stored.")
    bad = {key: shape for key, shape in shapes.items() if shape != want}
    if bad:
        raise ValueError(f"{source}: standardization mode of the manifest does not match the stored "
                         f"buffers: z_score_x={mode!r} expects shape {want} (theta_dim {theta_dim}) "
                         f"but the state_dict carries {bad}.")


def save_estimator(estimator, *, embedding_args, maf_args, theta_dim, video_shape,
                   parameter_keys, prior_low, prior_high, path, metadata=None):
    """Persist a trained estimator as a self-describing, version-portable artifact.

    Args:
        estimator: the trained MAF estimator (the `build_maf` output, whose
            embedding may be `torch.compile`d).
        embedding_args: the exact kwargs used to build the `Complex3DCNN`.
        maf_args: the exact `build_maf` kwargs (excluding `batch_x`/`batch_y`/
            `embedding_net`), e.g. `z_score_x`, `z_score_y`, `dropout_probability`,
            `use_batch_norm`.
        theta_dim: parameter-vector dimension (columns of `batch_x`).
        video_shape: per-example video shape (e.g. `(n_frames, H, W)`).
        parameter_keys: ordered learnable-parameter keys.
        prior_low, prior_high: prior bounds aligned with `parameter_keys`.
        path: output `.npz` path.
        metadata: optional provenance dict (timing label, best test loss,
            source conditions/accessions, ...).
    """
    stripped = {key: value.detach().cpu()
                for key, value in strip_compile_prefix(estimator.state_dict()).items()}
    # A mismatch between the recorded standardization mode and the buffers is never written.
    assert_standardization_consistent(stripped, dict(maf_args), int(theta_dim),
                                      source=f"save_estimator {path}")
    buffer = io.BytesIO()
    torch.save(stripped, buffer)
    weights = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    checksum = hashlib.sha256(weights.tobytes()).hexdigest()

    manifest = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "rebuild_spec": {
            "embedding_class": "Complex3DCNN",
            "embedding_args": embedding_args,
            "maf_builder": "build_maf",
            "maf_args": maf_args,
            "theta_dim": int(theta_dim),
            "video_shape": list(video_shape),
        },
        "parameter_keys": list(parameter_keys),
        "state_dict_keys": list(stripped.keys()),
        "torch_version": torch.__version__,
        "weights_sha256": checksum,
        "metadata": dict(metadata or {}),
    }
    np.savez_compressed(
        path,
        weights=weights,
        prior_low=np.asarray(prior_low, dtype=float),
        prior_high=np.asarray(prior_high, dtype=float),
        manifest=np.array(json.dumps(manifest)),
    )


def load_estimator(path, device: str = "cpu", *, expected_parameter_keys=None) -> DirectPosterior:
    """Rebuild a `DirectPosterior` from a `save_estimator` artifact, version-portably.

    Reconstructs the uncompiled estimator from the rebuild spec under the current
    torch, loads the compile-stripped weights (`weights_only=True`), verifies the
    checksum, and attaches a fresh device-aware `BoxUniform` prior. No
    torch-internal or compiled code is deserialized. The returned posterior carries
    ``weights_sha256``, the checksum of the weights it was built from.

    If ``expected_parameter_keys`` is given, the artifact's stored parameter schema
    is checked first (`assert_schema_compatible`) and a mismatch raises before any
    rebuild — so a legacy artifact is rejected loudly, never misread column-for-column.

    The stored theta-standardization buffers are checked against the manifest's
    ``maf_args["z_score_x"]`` before ``load_state_dict`` (`assert_standardization_consistent`):
    ``structured`` requires 0-d buffers, ``independent`` buffers of shape ``(theta_dim,)``,
    ``none`` no buffer. ``load_state_dict`` restores a ``(D,)`` buffer into a 0-d one silently,
    so without this check an edited or inconsistent manifest would rebuild a flow that
    standardizes theta wrongly; the mismatch raises ``ValueError`` naming the stored mode, the
    buffer shape and the theta_dim.
    """
    data = np.load(path, allow_pickle=False)
    manifest = json.loads(str(data["manifest"]))
    if expected_parameter_keys is not None:
        assert_schema_compatible(manifest, expected_parameter_keys=expected_parameter_keys)
    spec = manifest["rebuild_spec"]

    weights = data["weights"]
    if hashlib.sha256(weights.tobytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError(f"weights checksum mismatch loading estimator artifact {path}")
    state_dict = torch.load(io.BytesIO(weights.tobytes()), weights_only=True)

    estimator = _rebuild_estimator(
        spec["embedding_args"], spec["maf_args"],
        spec["theta_dim"], tuple(spec["video_shape"]), device,
    )
    # Checked on the stored state_dict BEFORE it is loaded: in one direction (a (D,) buffer into a
    # 0-d one) load_state_dict restores silently, in the other torch raises its own RuntimeError;
    # checking first gives one ValueError for both.
    assert_standardization_consistent(state_dict, dict(spec["maf_args"]), int(spec["theta_dim"]),
                                      source=f"estimator artifact {path}")
    estimator.load_state_dict(state_dict)
    estimator.eval()

    prior = BoxUniform(
        low=torch.tensor(data["prior_low"], dtype=torch.float32),
        high=torch.tensor(data["prior_high"], dtype=torch.float32),
        device=device,
    )
    posterior = DirectPosterior(estimator, prior)
    # The checksum of the weights ACTUALLY loaded (verified above against the bytes), attached so a
    # stage can record the checkpoint it ran with at startup instead of re-reading the path later
    # (a path may be replaced while a job runs; the loaded weights cannot).
    posterior.weights_sha256 = manifest["weights_sha256"]
    return posterior


def load_estimator_manifest(path) -> dict:
    """Return a `save_estimator` artifact's manifest without rebuilding the estimator.

    Reads only the stored metadata — the rebuild spec, ``parameter_keys``, torch
    version, weights checksum (``weights_sha256``), and caller metadata (timing
    label, best test loss, source conditions/accessions, ...) — plus the prior
    bounds as plain lists. Provenance is otherwise unreachable, since
    ``load_estimator`` returns only a ``DirectPosterior`` and discards the
    manifest. No torch load, no GPU, no estimator reconstruction.
    """
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        manifest["prior_low"] = data["prior_low"].tolist()
        manifest["prior_high"] = data["prior_high"].tolist()
    return manifest


def assert_schema_compatible(manifest_or_path, *, expected_parameter_keys) -> None:
    """Reject an estimator artifact whose parameter schema differs from the current one.

    A trained estimator's theta vector has a fixed meaning BY POSITION, recorded as
    ``parameter_keys`` in its manifest. When the parameterization changes while the
    vector length stays the same (e.g. the detector camera block moving from
    ``kappa_c``/``kappa_g`` to ``gamma``/``kappa_o``), a legacy artifact would rebuild
    and run silently under the WRONG semantics. This guard makes that failure loud: it
    compares the stored ``parameter_keys`` (content and order) against
    ``expected_parameter_keys`` and raises on any difference.

    Args:
        manifest_or_path: an already-parsed manifest dict, or a path to a
            `save_estimator` `.npz` (its manifest is read; no estimator rebuild).
        expected_parameter_keys: the current schema's ordered learnable-parameter keys
            (e.g. `DETECTOR_PARAMETER_KEYS`).

    Raises:
        ValueError: if the stored ``parameter_keys`` are absent or differ from
            ``expected_parameter_keys``.
    """
    manifest = (manifest_or_path if isinstance(manifest_or_path, dict)
                else load_estimator_manifest(manifest_or_path))
    stored = list(manifest.get("parameter_keys", []))
    expected = list(expected_parameter_keys)
    if stored != expected:
        raise ValueError(
            "estimator schema mismatch: this artifact was trained under a different "
            "parameterization and must not be loaded under the current schema "
            "(equal-length theta vectors would otherwise be misread column-for-column).\n"
            f"  stored   ({len(stored)}): {stored}\n"
            f"  expected ({len(expected)}): {expected}\n"
            "Regenerate and retrain under the current schema."
        )


def estimator_weights_sha256(path) -> str:
    """The weights checksum an estimator artifact records in its manifest, read without rebuilding
    the flow. This is the checkpoint identity an Evaluation / Experiment product manifest stores."""
    with np.load(path, allow_pickle=False) as data:
        return str(json.loads(str(data["manifest"]))["weights_sha256"])


# ---- Weights-initialized training phase (Inference --initialize-from-tag) ----------------------
# A new training phase can start from a stored estimator: the artifact's weights and every stored
# buffer are loaded into the freshly built training network, while the optimizer, the learning-rate
# schedule and its counters start fresh. The helpers below read the artifact without rebuilding it,
# list every way it differs from the run it would initialize, compare the standardization the run
# fits with the stored one, and load the weights into a network whose embedding is compiled.

def read_estimator_artifact(path):
    """The stored parts of a `save_estimator` artifact, without rebuilding the network.

    Returns ``(state_dict, manifest, prior_low, prior_high)``: the compile-stripped state_dict
    (checksum-verified against the manifest, loaded ``weights_only``), the manifest dict, and the
    prior bounds as lists. Raises ``ValueError`` on a checksum mismatch.
    """
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        weights = data["weights"]
        prior_low = data["prior_low"].tolist()
        prior_high = data["prior_high"].tolist()
    if hashlib.sha256(weights.tobytes()).hexdigest() != manifest["weights_sha256"]:
        raise ValueError(f"weights checksum mismatch loading estimator artifact {path}")
    state_dict = torch.load(io.BytesIO(weights.tobytes()), weights_only=True)
    return state_dict, manifest, prior_low, prior_high


def _as_stored(value):
    """``value`` in the form a manifest stores it (a JSON round trip: tuples become lists)."""
    return json.loads(json.dumps(value))


def initialization_mismatches(manifest, prior_low, prior_high, *, workflow, condition, timing_label,
                              parameter_keys, lower_bound, upper_bound, embedding_args, maf_args,
                              theta_dim, video_shape) -> list:
    """Every way a source artifact differs from the training run it would initialize.

    Compared: the workflow, condition and timing label recorded in the artifact's metadata; the
    ordered parameter keys; the prior bounds (exactly, to 1e-12); and the rebuild specification --
    the embedding arguments (the console-only ``verbose`` excepted), the flow arguments, the theta
    width and the video shape, each in the form the manifest stores. Returns a list of readable
    differences; an empty list means the artifact can initialize the run.
    """
    problems = []
    meta = manifest.get("metadata") or {}
    for key, want in (("workflow", workflow), ("condition", condition), ("timing_label", timing_label)):
        if meta.get(key) != want:
            problems.append(f"{key}: source {meta.get(key)!r}, run {want!r}")
    if list(manifest.get("parameter_keys") or []) != list(parameter_keys):
        problems.append(f"parameter keys: source {manifest.get('parameter_keys')}, run {list(parameter_keys)}")
    for name, stored, want in (("prior lower bounds", prior_low, lower_bound),
                               ("prior upper bounds", prior_high, upper_bound)):
        stored_arr = np.asarray(stored, dtype=float)
        want_arr = np.asarray(want, dtype=float)
        if stored_arr.shape != want_arr.shape or not np.allclose(stored_arr, want_arr, rtol=0.0, atol=1e-12):
            problems.append(f"{name}: source {stored_arr.tolist()}, run {want_arr.tolist()}")
    spec = manifest.get("rebuild_spec") or {}
    stored_embedding = {k: v for k, v in (spec.get("embedding_args") or {}).items() if k != "verbose"}
    run_embedding = {k: v for k, v in _as_stored(dict(embedding_args)).items() if k != "verbose"}
    if stored_embedding != run_embedding:
        keys = sorted(set(stored_embedding) | set(run_embedding))
        diff = {k: (stored_embedding.get(k), run_embedding.get(k)) for k in keys
                if stored_embedding.get(k) != run_embedding.get(k)}
        problems.append(f"embedding arguments (source, run): {diff}")
    if (spec.get("maf_args") or {}) != _as_stored(dict(maf_args)):
        problems.append(f"flow arguments: source {spec.get('maf_args')}, run {_as_stored(dict(maf_args))}")
    if spec.get("theta_dim") != int(theta_dim):
        problems.append(f"theta width: source {spec.get('theta_dim')}, run {int(theta_dim)}")
    if spec.get("video_shape") != _as_stored(list(video_shape)):
        problems.append(f"video shape: source {spec.get('video_shape')}, run {list(video_shape)}")
    return problems


def standardization_buffers(state_dict: dict) -> dict:
    """The standardization buffers of a compile-stripped state_dict: the theta shift and scale of
    the flow's first transform and, when the videos are standardized, the mean and std of the
    embedding's leading standardizing layer. Both are fitted once, when the network is built."""
    out = dict(_standardization_buffers(state_dict))
    out.update({key: value for key, value in state_dict.items()
                if key.endswith(("_embedding_net.0._mean", "_embedding_net.0._std"))})
    return out


def standardization_relative_differences(fitted: dict, stored: dict) -> dict:
    """Largest relative difference, buffer by buffer, between the standardization a run fitted and
    the one an artifact stores (both compile-stripped state_dicts). A buffer present in only one of
    them is reported as ``inf``."""
    a, b = standardization_buffers(fitted), standardization_buffers(stored)
    out = {}
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b or tuple(a[key].shape) != tuple(b[key].shape):
            out[key] = float("inf")
            continue
        x = a[key].detach().double().cpu()
        y = b[key].detach().double().cpu()
        out[key] = float(((x - y).abs() / y.abs().clamp_min(1e-12)).max())
    return out


def load_artifact_weights(estimator, state_dict: dict) -> None:
    """Load an artifact's compile-stripped ``state_dict`` into a training ``estimator`` whose
    embedding may be ``torch.compile``d (its own keys then carry the ``_orig_mod.`` prefix).

    Every key of the estimator, stripped, must name a stored tensor of the same shape and dtype,
    and no stored tensor may be left over; the load is then strict. Restores the weights and every
    buffer -- the theta and video standardization, the batch-norm running statistics, the flow's
    masks and permutations. Raises ``ValueError`` naming the first mismatches otherwise.
    """
    own = estimator.state_dict()
    stripped_of = {key: key.replace(_COMPILE_PREFIX, "") for key in own}
    wanted = set(stripped_of.values())
    missing = sorted(wanted - set(state_dict))
    unexpected = sorted(set(state_dict) - wanted)
    if missing or unexpected:
        raise ValueError(f"the stored state_dict does not match the network: {len(missing)} key(s) missing "
                         f"(first: {missing[:3]}), {len(unexpected)} unexpected (first: {unexpected[:3]}).")
    bad = [key for key, stored in stripped_of.items()
           if tuple(own[key].shape) != tuple(state_dict[stored].shape) or own[key].dtype != state_dict[stored].dtype]
    if bad:
        raise ValueError(f"the stored state_dict does not match the network in shape or dtype at "
                         f"{len(bad)} key(s) (first: {bad[:3]}).")
    estimator.load_state_dict({key: state_dict[stored] for key, stored in stripped_of.items()}, strict=True)
