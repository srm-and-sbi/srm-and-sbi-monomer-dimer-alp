"""File I/O for the entry-point scripts.

- load_data: transparently loads .zarr / .npy / .npz arrays.
- save_video_set, save_theta_set: write .npy or .npz buffers.
- convert_video_dtype: rescale pixel intensities between bit depths.
- The Theta_Set schema: ``theta_set_schema`` builds it from a parameter table,
  ``write_theta_set`` stores a theta set WITH it, ``load_theta_set`` refuses a theta set
  whose schema is absent or differs from the caller's table (``ThetaSetSchemaError``),
  ``theta_set_status`` renders the verdict for dry runs.

Note: incremental .zarr writes (slice-by-slice during simulation) are handled
inline in entry-point scripts, not by these helpers — via zarr.open at the top
of the simulation loop followed by array[i] = slice assignments.

The Theta_Set schema
--------------------
A ``Theta_Set`` is a numeric array of PHYSICAL values, one row per simulation, one column
per learnable parameter. Two tables with the same number of rows are indistinguishable by
shape, so every ``Theta_Set`` written by this package carries its schema beside the numbers:
the ordered ``parameter_keys``, the prior bounds in the estimator coordinate
(``prior_low`` / ``prior_high``) and the per-row ``log_flag``, the ``condition`` and
``timing_label`` it was generated under, the ``generator`` stage, the ``package_version``,
and a ``schema_version``. For a ``.zarr`` store the schema lives in the array attributes;
for a ``.npy`` / ``.npz`` file it lives in a JSON sidecar ``<file>.schema.json``. Every
reader of a ``Theta_Set`` goes through ``load_theta_set`` with its own parameter table, and a
file without a schema, with different keys (content or order), with different prior bounds,
or -- when the caller names one -- a different condition, is refused with a message that says
what differs and that the tier must be regenerated. Prior bounds are part of the schema on
purpose: a tier drawn from a different box is a different training distribution even when the
keys agree. The record sets beside a ``Theta_Set`` (``Nuisance_DLI_Theta_Set``,
``Nuisance_SCOPE_Theta_Set``, ``Labeling_Set``) are not schema-checked; the ``Nuisance_DLI``
record carries the imaging schema for provenance only.
"""

import datetime
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import numcodecs
import zarr
from skimage import exposure

THETA_SET_SCHEMA_VERSION = 1
_SCHEMA_KEYS = ("schema_version", "parameter_keys", "prior_low", "prior_high", "log_flag",
                "condition", "timing_label", "generator", "package_version", "written_utc")
_BOUND_TOL = 1e-9


class ThetaSetSchemaError(ValueError):
    """A Theta_Set has no schema or its schema differs from the caller's parameter table."""


# =============================================================================
# Loading
# =============================================================================

def load_data(path: Union[str, Path]):
    """Load array data from .zarr (lazy), .npy, or .npz transparently.

    Args:
        path: Path to the data file (string or pathlib.Path).

    Returns:
        For .zarr: a lazy zarr.Array (only requested chunks are decompressed on access).
        For .npy: a numpy.ndarray (memory-mapped via mmap_mode="r").
        For .npz: the first ndarray in the archive.
    """
    path_str = str(path)
    if path_str.endswith(".zarr"):
        return zarr.open(store=path_str, mode="r")
    data = np.load(file=path_str, mmap_mode="r")
    if isinstance(data, np.lib.npyio.NpzFile):
        return data[data.files[0]]  # first array in the .npz archive
    return data


# =============================================================================
# Saving (.npy and .npz only; .zarr handled inline by callers)
# =============================================================================

def save_video_set(path: Union[str, Path], video_set: np.ndarray,
                   compress: bool = True) -> None:
    """Save a video-set array to disk as .npz (compressed) or .npy.

    See module docstring for the .zarr handling note.
    """
    path_str = str(path)
    if compress:
        np.savez_compressed(file=path_str, arr=video_set, allow_pickle=False)
    else:
        np.save(file=path_str, arr=video_set, allow_pickle=False)


def save_theta_set(path: Union[str, Path], theta_set: np.ndarray,
                   compress: bool = True) -> None:
    """Save a theta-set array to disk as .npz (compressed) or .npy.

    See module docstring for the .zarr handling note.
    """
    path_str = str(path)
    if compress:
        np.savez_compressed(file=path_str, arr=theta_set, allow_pickle=False)
    else:
        np.save(file=path_str, arr=theta_set, allow_pickle=False)


# =============================================================================
# The Theta_Set schema
# =============================================================================

def theta_set_schema(table: Sequence[dict], *, condition: Optional[str], timing_label: str,
                     generator: str) -> dict:
    """The schema a Theta_Set drawn from ``table`` carries (see the module docstring).

    Args:
        table: the parameter table (dicts with ``KEY``, ``PRIOR_RANGE``, ``LOG_FLAG``) whose
            rows are the theta columns, in order.
        condition: stored condition token (``FAB`` / ``INLB``), or None for a product that
            is not condition-bound.
        timing_label: the run's timing label (``2S_50FPS``).
        generator: the stage that wrote it (``rds``, ``dli_detector``, ``dli_nuisance``).
    """
    from srm_and_sbi_monomer_dimer_alp import __version__
    return {
        "schema_version": THETA_SET_SCHEMA_VERSION,
        "parameter_keys": [str(e["KEY"]) for e in table],
        "prior_low": [float(e["PRIOR_RANGE"][0]) for e in table],
        "prior_high": [float(e["PRIOR_RANGE"][1]) for e in table],
        "log_flag": [bool(e["LOG_FLAG"]) for e in table],
        "condition": None if condition is None else str(condition),
        "timing_label": str(timing_label),
        "generator": str(generator),
        "package_version": str(__version__),
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def _sidecar_path(path: Union[str, Path]) -> Path:
    return Path(str(path) + ".schema.json")


def _theta_compressor():
    return numcodecs.Blosc(cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)


def write_theta_set(path: Union[str, Path], theta_set: np.ndarray, schema: dict) -> None:
    """Write a theta set WITH its schema: ``.zarr`` (attrs) or ``.npy`` / ``.npz`` (JSON sidecar).

    The format follows the path's extension; a ``.zarr`` store is compressed with the
    package's theta compressor (zstd, level 9, bit-shuffle), one row per chunk.
    """
    theta_set = np.asarray(theta_set, dtype=np.float64)
    if theta_set.ndim != 2 or theta_set.shape[1] != len(schema["parameter_keys"]):
        raise ValueError(f"theta set of shape {theta_set.shape} does not match the schema's "
                         f"{len(schema['parameter_keys'])} parameter keys.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".zarr"):
        store = zarr.open(store=str(path), mode="w", shape=theta_set.shape,
                          chunks=(1, theta_set.shape[1]), dtype=np.float64, compressor=_theta_compressor())
        store[:, :] = theta_set
        store.attrs.update(schema)
    else:
        save_theta_set(path, theta_set, compress=str(path).endswith(".npz"))
        with open(_sidecar_path(path), "w") as h:
            json.dump(schema, h, indent=2)


def read_theta_set_schema(path: Union[str, Path]) -> Optional[dict]:
    """The stored schema of a theta set, or None when the file carries none (a pre-schema file)."""
    path_str = str(path)
    if path_str.endswith(".zarr"):
        attrs = dict(zarr.open(store=path_str, mode="r").attrs)
        return attrs if "parameter_keys" in attrs else None
    sidecar = _sidecar_path(path)
    if not sidecar.exists():
        return None
    with open(sidecar) as h:
        return json.load(h)


def schema_mismatch(schema: Optional[dict], table: Sequence[dict], *,
                    condition: Optional[str] = None) -> Optional[str]:
    """None when ``schema`` matches ``table`` (and ``condition`` if given); otherwise the reason."""
    if schema is None:
        return "no schema (a Theta_Set written before the schema was introduced, or not by this package)"
    expected = [str(e["KEY"]) for e in table]
    if list(schema.get("parameter_keys", [])) != expected:
        return (f"parameter keys differ: stored {list(schema.get('parameter_keys', []))} "
                f"vs expected {expected}")
    low = [float(e["PRIOR_RANGE"][0]) for e in table]
    high = [float(e["PRIOR_RANGE"][1]) for e in table]
    s_low, s_high = schema.get("prior_low"), schema.get("prior_high")
    if s_low is None or s_high is None or len(s_low) != len(low) or len(s_high) != len(high):
        return "prior bounds absent or of the wrong length"
    for k, lo, hi, slo, shi in zip(expected, low, high, s_low, s_high):
        if abs(float(slo) - lo) > _BOUND_TOL or abs(float(shi) - hi) > _BOUND_TOL:
            return (f"prior bounds differ for {k!r}: stored [{float(slo):g}, {float(shi):g}] "
                    f"vs expected [{lo:g}, {hi:g}]")
    flags = [bool(e["LOG_FLAG"]) for e in table]
    if list(map(bool, schema.get("log_flag", []))) != flags:
        return "per-row scale flags differ"
    if condition is not None and schema.get("condition") not in (None, str(condition)):
        return f"condition differs: stored {schema.get('condition')!r} vs expected {condition!r}"
    return None


def check_theta_set_schema(path: Union[str, Path], table: Sequence[dict], *,
                           condition: Optional[str] = None) -> dict:
    """Raise ``ThetaSetSchemaError`` unless the theta set at ``path`` matches ``table``; return its schema."""
    schema = read_theta_set_schema(path)
    reason = schema_mismatch(schema, table, condition=condition)
    if reason is not None:
        raise ThetaSetSchemaError(
            f"Refusing Theta_Set {path}: {reason}. A Theta_Set is consumed only under the parameter "
            f"table it was drawn from; regenerate the tier under the current table.")
    return schema


def load_theta_set(path: Union[str, Path], table: Sequence[dict], *, condition: Optional[str] = None):
    """``load_data`` for a Theta_Set, after ``check_theta_set_schema``."""
    check_theta_set_schema(path, table, condition=condition)
    return load_data(path)


def theta_set_status(path: Union[str, Path], table: Sequence[dict], *,
                     condition: Optional[str] = None) -> str:
    """``OK`` / ``MISSING`` / ``SCHEMA MISMATCH: <reason>`` for dry-run listings."""
    if not Path(path).exists():
        return "MISSING"
    reason = schema_mismatch(read_theta_set_schema(path), table, condition=condition)
    return "OK" if reason is None else f"SCHEMA MISMATCH: {reason}"


# =============================================================================
# Bit-depth conversion
# =============================================================================

_DTYPE_FOR_BITS = {8: np.uint8, 16: np.uint16}


def convert_video_dtype(video: np.ndarray,
                        bits_from: int = 16,
                        bits_to: int = 8) -> np.ndarray:
    """Rescale pixel intensities between bit depths via skimage.exposure.

    Args:
        video: input array of any shape (typically 3D: (n_frames, H, W)).
        bits_from: source bit depth (e.g., 16 for uint16 input).
        bits_to: target bit depth (e.g., 8 for uint8 output).

    Returns:
        Array of shape == video.shape, dtype matching the target bit depth.

    Note: the conversion is global (in_range/out_range fixed); applies to the
    whole array at once (vectorised over all frames, no per-frame loop).

    Clipping: `rescale_intensity` clips values outside `in_range` before
    rescaling, so a negative input pixel maps to the output minimum (0) and an
    over-range pixel to the output maximum. Synthetic frames carry a small
    negative excursion from the gain-independent Gaussian read noise added after the register (`add_noise`) that a real
    camera cannot record; clipping to the non-negative range aligns the stored
    synthetic with the real camera domain. This is a deliberate, revisit-able
    choice -- a future or alternative noise model may have different negative
    behavior. See REFERENCE_EMCCD_NOISE_MODEL.md.
    """
    if bits_to not in _DTYPE_FOR_BITS:
        raise ValueError(
            f"bits_to={bits_to} not supported; must be one of {sorted(_DTYPE_FOR_BITS)}."
        )
    rescaled = exposure.rescale_intensity(
        video,
        in_range=(0, 2**bits_from - 1),
        out_range=(0, 2**bits_to - 1),
    )
    return np.round(rescaled).astype(_DTYPE_FOR_BITS[bits_to])
