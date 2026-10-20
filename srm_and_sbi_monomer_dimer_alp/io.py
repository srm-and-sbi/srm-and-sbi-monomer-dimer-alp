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

Publication. ``write_theta_set`` builds a record under ``.partial`` names and publishes it so
that the final name only ever holds a complete record paired with its own schema: a reader or
a tier scan sees the previous record, no record, or the new record, never values under another
write's schema. A ``.zarr`` store holds values and attributes together and is published by one
rename (a store being replaced is first renamed aside in one step). A plain array and its
sidecar are two files, so the sidecar also records ``values_sha256``, the SHA-256 of the value
file it was written with; the previous values leave the final name before the new sidecar takes
its name, and the new values take theirs last. ``load_theta_set`` returns plain-array values
only from bytes matching that digest, so a value file copied, restored or replaced apart from
its sidecar is refused rather than read under the wrong schema. This protects readers and tier
scans that meet a record while it is being published; it does not make concurrent replacement
and reading safe. A ``.zarr`` store read lazily across its own replacement can mix the two
writes, so a tier is never regenerated while training, rendering or an analysis reads it.
"""

import datetime
import hashlib
import json
from io import BytesIO          # the standard library's io (this module is the package's io)
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import numcodecs
import os
import shutil
import zarr
from skimage import exposure

THETA_SET_SCHEMA_VERSION = 1
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

def theta_set_schema(table: Sequence[dict], *, condition: str, timing_label: str,
                     generator: str) -> dict:
    """The schema a Theta_Set drawn from ``table`` carries (see the module docstring).

    Args:
        table: the parameter table (dicts with ``KEY``, ``PRIOR_RANGE``, ``LOG_FLAG``) whose
            rows are the theta columns, in order: one column per learnable row of the
            condition's table (eleven reaction-diffusion labels under MET-FAB, twelve under
            MET-INLB; six imaging labels for a detector product).
        condition: stored condition token (``FAB`` / ``INLB``); required, every Theta_Set is
            per condition.
        timing_label: the run's timing label (``2S_50FPS``).
        generator: the stage that wrote it (``rds``, ``dli_detector``, ``dli_nuisance``).
    """
    from srm_and_sbi_monomer_dimer_alp import __version__
    if condition is None or not str(condition):
        raise ValueError("theta_set_schema: the condition is required; every Theta_Set is per condition.")
    return {
        "schema_version": THETA_SET_SCHEMA_VERSION,
        "parameter_keys": [str(e["KEY"]) for e in table],
        "prior_low": [float(e["PRIOR_RANGE"][0]) for e in table],
        "prior_high": [float(e["PRIOR_RANGE"][1]) for e in table],
        "log_flag": [bool(e["LOG_FLAG"]) for e in table],
        "condition": str(condition),
        "timing_label": str(timing_label),
        "generator": str(generator),
        "package_version": str(__version__),
        "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def _sidecar_path(path: Union[str, Path]) -> Path:
    return Path(str(path) + ".schema.json")


def _theta_compressor():
    return numcodecs.Blosc(cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)


PARTIAL_SUFFIX = ".partial"


def _partial_path(path: Path) -> Path:
    """The temporary name a record is built under before it is renamed into place."""
    return path.with_name(path.name + PARTIAL_SUFFIX)


def _previous_path(path: Path) -> Path:
    """The name a ``.zarr`` store being replaced is moved to, in one rename, before its
    successor takes the final name (a ``.partial`` name, so no scan ever lists it)."""
    return path.with_name(path.name + ".previous" + PARTIAL_SUFFIX)


def _is_plain(path: Union[str, Path]) -> bool:
    """A ``.npy`` / ``.npz`` record (values and schema in two files), not a ``.zarr`` store."""
    return not str(path).endswith(".zarr")


def values_digest(raw: bytes) -> str:
    """SHA-256 of a plain-array record's value file, stored in its sidecar as ``values_sha256``."""
    return hashlib.sha256(raw).hexdigest()


def binding_mismatch(schema: dict, raw: bytes) -> Optional[str]:
    """None when ``raw`` is the value file a plain-array record's sidecar was written with;
    otherwise the reason. A ``.zarr`` store needs no binding: its values and attributes are one
    store, published by one rename."""
    stored = schema.get("values_sha256")
    if stored is None:
        return ("the sidecar records no digest of the values it was written with, so the value "
                "file cannot be verified as the one it describes")
    if values_digest(raw) != stored:
        return ("the values are not the ones the sidecar was written with (the value file was "
                "copied, restored or replaced apart from its sidecar, or rewritten while being read)")
    return None


def write_theta_set(path: Union[str, Path], theta_set: np.ndarray, schema: dict) -> None:
    """Write a theta set WITH its schema: ``.zarr`` (attrs) or ``.npy`` / ``.npz`` (JSON sidecar).

    The format follows the path's extension; a ``.zarr`` store is compressed with the
    package's theta compressor (zstd, level 9, bit-shuffle), one row per chunk.

    PUBLICATION (see the module docstring). The record is built completely under ``.partial``
    names, and the final name only ever holds a complete record paired with its own schema: a
    reader or a tier scan sees the previous record, no record, or the new record. An existing
    final-name record is replaced; replacing a record while a stage reads it stays unsupported
    (module docstring).

    - ``.zarr``: the store is published by one rename; a store being replaced is first renamed
      aside in one step (never deleted file by file under its final name) and removed last.
    - ``.npy`` / ``.npz``: the sidecar records ``values_sha256`` of the value file. The previous
      values leave the final name first, then the new sidecar takes its name, then the new values.
    """
    theta_set = np.asarray(theta_set, dtype=np.float64)
    if theta_set.ndim != 2 or theta_set.shape[1] != len(schema["parameter_keys"]):
        raise ValueError(f"theta set of shape {theta_set.shape} does not match the schema's "
                         f"{len(schema['parameter_keys'])} parameter keys.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(path)
    if not _is_plain(path):
        if partial.exists():
            shutil.rmtree(partial)
        store = zarr.open(store=str(partial), mode="w", shape=theta_set.shape,
                          chunks=(1, theta_set.shape[1]), dtype=np.float64, compressor=_theta_compressor())
        store[:, :] = theta_set
        store.attrs.update(schema)
        previous = _previous_path(path)
        if previous.exists():                    # left by an interrupted replacement
            shutil.rmtree(previous)
        if path.exists():
            os.replace(path, previous)
        os.replace(partial, path)
        if previous.exists():
            shutil.rmtree(previous)
    else:
        with open(partial, "wb") as fh:          # a handle: np.save would append ".npy" to a bare name
            if str(path).endswith(".npz"):
                np.savez_compressed(fh, arr=theta_set, allow_pickle=False)
            else:
                np.save(fh, theta_set, allow_pickle=False)
        sidecar = _sidecar_path(path)
        sidecar_partial = _partial_path(sidecar)
        with open(sidecar_partial, "w") as h:
            json.dump(dict(schema, values_sha256=values_digest(partial.read_bytes())), h, indent=2)
        path.unlink(missing_ok=True)             # the previous values leave the final name first,
        os.replace(sidecar_partial, sidecar)     # then the new schema takes its name,
        os.replace(partial, path)                # then the new values: complete and paired


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


def _stored_origin(schema: Optional[dict]) -> str:
    """'written by package <version> (<generator>)' from a stored schema, for refusal messages."""
    if not schema:
        return "no stored provenance"
    return (f"written by package {schema.get('package_version', '?')} "
            f"({schema.get('generator', '?')}, condition {schema.get('condition')!r})")


def schema_mismatch(schema: Optional[dict], table: Sequence[dict], *, condition: str) -> Optional[str]:
    """None when ``schema`` matches ``table`` AND ``condition``; otherwise the reason.

    The condition is required and compared strictly: a Theta_Set is per condition, so a stored
    None or another token is a mismatch (the conditions' tables differ in width, and the eleven
    shared columns would otherwise be misread across conditions)."""
    if condition is None:
        raise ValueError("schema_mismatch: the condition is required; every Theta_Set is per condition.")
    if schema is None:
        return "no schema (a Theta_Set written before the schema was introduced, or not by this package)"
    expected = [str(e["KEY"]) for e in table]
    if list(schema.get("parameter_keys", [])) != expected:
        return (f"parameter keys differ: stored {list(schema.get('parameter_keys', []))} "
                f"vs expected {expected} ({_stored_origin(schema)})")
    low = [float(e["PRIOR_RANGE"][0]) for e in table]
    high = [float(e["PRIOR_RANGE"][1]) for e in table]
    s_low, s_high = schema.get("prior_low"), schema.get("prior_high")
    if s_low is None or s_high is None or len(s_low) != len(low) or len(s_high) != len(high):
        return f"prior bounds absent or of the wrong length ({_stored_origin(schema)})"
    for k, lo, hi, slo, shi in zip(expected, low, high, s_low, s_high):
        if abs(float(slo) - lo) > _BOUND_TOL or abs(float(shi) - hi) > _BOUND_TOL:
            return (f"prior bounds differ for {k!r}: stored [{float(slo):g}, {float(shi):g}] "
                    f"vs expected [{lo:g}, {hi:g}] ({_stored_origin(schema)})")
    flags = [bool(e["LOG_FLAG"]) for e in table]
    if list(map(bool, schema.get("log_flag", []))) != flags:
        return f"per-row scale flags differ ({_stored_origin(schema)})"
    if schema.get("condition") != str(condition):
        return (f"condition differs: stored {schema.get('condition')!r} vs expected {condition!r} "
                f"({_stored_origin(schema)})")
    return None


def _checked_record(path: Union[str, Path], table: Sequence[dict], *, condition: str):
    """``(schema, raw)`` of a record that matches ``table`` and ``condition``, else
    ``ThetaSetSchemaError``. For a plain array ``raw`` is its value file, read once and verified
    against the sidecar's digest; ``None`` for a ``.zarr`` store."""
    schema = read_theta_set_schema(path)
    reason = schema_mismatch(schema, table, condition=condition)
    raw = None
    if reason is None and _is_plain(path):
        raw = Path(path).read_bytes()
        reason = binding_mismatch(schema, raw)
    if reason is not None:
        raise ThetaSetSchemaError(
            f"Refusing Theta_Set {path}: {reason}. A Theta_Set is consumed only under the parameter "
            f"table and condition it was drawn from; regenerate the tier under the current table.")
    return schema, raw


def check_theta_set_schema(path: Union[str, Path], table: Sequence[dict], *, condition: str) -> dict:
    """Raise ``ThetaSetSchemaError`` unless the theta set at ``path`` matches ``table`` and
    ``condition`` (and, for a plain array, its values are the ones its sidecar was written with);
    return its schema."""
    return _checked_record(path, table, condition=condition)[0]


def load_theta_set(path: Union[str, Path], table: Sequence[dict], *, condition: str):
    """``load_data`` for a Theta_Set, after ``check_theta_set_schema``. A plain array is returned
    in memory, parsed from the very bytes verified against its sidecar's digest, so a record
    replaced between the schema read and the value read is refused, never paired with the other
    write's schema."""
    _schema, raw = _checked_record(path, table, condition=condition)
    if raw is None:
        return load_data(path)
    data = np.load(BytesIO(raw), allow_pickle=False)
    return data[data.files[0]] if isinstance(data, np.lib.npyio.NpzFile) else data


def theta_set_status(path: Union[str, Path], table: Sequence[dict], *, condition: str) -> str:
    """``OK`` / ``MISSING`` / ``SCHEMA MISMATCH: <reason>`` for dry-run listings."""
    if not Path(path).exists():
        return "MISSING"
    try:
        schema = read_theta_set_schema(path)
        raw = Path(path).read_bytes() if _is_plain(path) else None
    except Exception as exc:  # noqa: BLE001 -- an unreadable record is reported, not raised, in a listing
        return f"SCHEMA MISMATCH: unreadable record ({type(exc).__name__}: {exc})"
    reason = schema_mismatch(schema, table, condition=condition)
    if reason is None and raw is not None:
        reason = binding_mismatch(schema, raw)
    return "OK" if reason is None else f"SCHEMA MISMATCH: {reason}"


# =============================================================================
# Tier scan: every record of a tier, classified against one table (the mixed-schema guard)
# =============================================================================

_THETA_SET_EXTENSIONS = (".zarr", ".npy", ".npz")


def tier_record_paths(theta_dir: Union[str, Path], pattern: str, *, project_alias: str,
                      timing_label: str, split: str) -> list:
    """Every final-name Theta_Set record of one tier (``pattern`` is ``Paths.theta_set_pattern``),
    sorted by task number. Names under the ``.partial`` suffix (in-progress writes) and schema
    sidecars are skipped; a record is listed whatever its schema says."""
    theta_dir = Path(theta_dir)
    if not theta_dir.exists():
        return []
    glob = pattern.format(project_alias=project_alias, timing_label=timing_label, task_alias="*",
                          split=split, ext="*")
    head, _, tail = glob.partition("*")          # '<alias>_<timing>_Theta_Set_TASK_' and '_<split>.*'
    tail = tail.rsplit(".", 1)[0]                 # '_<split>'
    found = []
    for candidate in theta_dir.glob(glob):
        name = candidate.name
        if PARTIAL_SUFFIX in name or candidate.suffix not in _THETA_SET_EXTENSIONS:
            continue
        stem = name[: -len(candidate.suffix)]
        if not (stem.startswith(head) and stem.endswith(tail)):
            continue
        task_text = stem[len(head):len(stem) - len(tail)]
        if not task_text.isdigit():
            continue
        found.append((int(task_text), candidate))
    return [path for _, path in sorted(found)]


class TierScan:
    """The classification of a tier's records against one table and condition.

    ``compatible``: paths whose schema matches; ``mismatched``: ``(path, reason)`` pairs whose
    schema differs; ``unknown``: ``(path, reason)`` pairs with no readable schema. Unknown is
    NOT a compatibility verdict: a corrupt or foreign record is refused like a mismatched one.
    ``vanished``: listed paths whose final name was gone when their schema was read (removed,
    or between the two renames that replace a ``.zarr`` store); they are not records of the
    tier at that moment and refuse nothing.
    The scan is a mixed-schema check; it proves neither dataset nor generator identity.
    """

    def __init__(self, compatible, mismatched, unknown, vanished=()):
        self.compatible = list(compatible)
        self.mismatched = list(mismatched)
        self.unknown = list(unknown)
        self.vanished = list(vanished)

    @property
    def n_records(self) -> int:
        return len(self.compatible) + len(self.mismatched) + len(self.unknown)

    @property
    def refused(self) -> bool:
        return bool(self.mismatched or self.unknown)

    @property
    def verdict(self) -> str:
        skipped = (f"; skipped {len(self.vanished)} record(s) removed or being replaced during the scan"
                   if self.vanished else "")
        if self.n_records == 0:
            return "no records" + skipped
        if not self.refused:
            return f"verified compatible ({len(self.compatible)} records)" + skipped
        parts = []
        if self.mismatched:
            parts.append("mismatched: " + "; ".join(f"{p.name} ({r})" for p, r in self.mismatched))
        if self.unknown:
            parts.append("unknown (no readable schema, not a compatibility verdict): "
                         + "; ".join(f"{p.name} ({r})" for p, r in self.unknown))
        return "refused -- " + " | ".join(parts) + skipped


def _scanned_schema(path: Path):
    """``(schema, None)``, or ``(None, why no schema was read)`` -- never raises."""
    try:
        schema = read_theta_set_schema(path)
    except Exception as exc:  # noqa: BLE001 -- classified by the scan, which goes on
        return None, f"{type(exc).__name__}: {exc}"
    return (schema, None) if schema is not None else (None, "no schema")


def tier_schema_scan(record_paths: Sequence[Union[str, Path]], table: Sequence[dict], *,
                     condition: str) -> TierScan:
    """Classify every record in ``record_paths`` against ``table`` and ``condition``, reading
    schemas only (never arrays) and continuing past an unreadable record, so that a readable
    superseded record beside it is still named.

    A record is unknown only if it is present and its schema cannot be read twice in a row. One
    whose final name is gone when its schema is not read is vanished: it was removed, or the scan
    fell between the two renames that replace a ``.zarr`` store (``write_theta_set``), and a
    record in publication is never mistaken for an unreadable one."""
    compatible, mismatched, unknown, vanished = [], [], [], []
    for raw in record_paths:
        path = Path(raw)
        schema, why = _scanned_schema(path)
        if schema is None:
            if not path.exists():                    # removed, or between the renames of a replacement
                vanished.append(path)
                continue
            schema, why = _scanned_schema(path)      # present: read once more
            if schema is None:
                if path.exists():
                    unknown.append((path, why))
                else:
                    vanished.append(path)
                continue
        reason = schema_mismatch(schema, table, condition=condition)
        if reason is None:
            compatible.append(path)
        else:
            mismatched.append((path, reason))
    return TierScan(compatible, mismatched, unknown, vanished)


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
