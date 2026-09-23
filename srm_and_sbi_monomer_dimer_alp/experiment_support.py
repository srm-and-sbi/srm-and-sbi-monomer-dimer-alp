"""Shared machinery for the Detector real-data stages (Experiment + Nuisance_DLI).

Two Detector steps read the same real recordings and run the same estimator over
them with the same overall shape:

    load the estimator  ->  discover the cells on disk  ->  read each recording and
    window it into model-length chunks  ->  run the estimator per chunk  ->  aggregate.

The Detector *Experiment* stage MAP-estimates each chunk and reports the inferred
imaging parameters per condition; the *Nuisance_DLI* analysis draws posterior samples
per chunk and pools them into the marginalized imaging distribution. Both are pooled
over whole cells, so the per-chunk work is embarrassingly parallel and the result is
order-independent.

That makes a single multi-GPU pattern serve both: launch one worker per GPU
(``torchrun``), split the flat ``(kind, cell)`` work list across workers round-robin by
rank (each worker owns whole cells), have each worker write its partial arrays as a
compressed ``_shard_*_of_*.npz`` next to the output, and then run a separate,
single-process, no-GPU ``--merge`` step that concatenates the shards into the final
product (a report for the Experiment; a cached pool + emitted spec for the Nuisance_DLI).
With one worker (``world_size == 1``) the sharding is a no-op: the single process does
all cells and produces the product directly, with no shard and no merge.

This module holds exactly the pieces both stages share -- cell discovery, per-cell
read+window, rank round-robin, the shard path, and the shard save/load/merge -- so the
two stages stay byte-for-byte identical in how they discover, window, shard, and merge.
The estimation itself (MAP vs. posterior-sample pool) stays in each stage.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile

from .io import convert_video_dtype


# =============================================================================
# Experimental conditions: one definition of the naming, for the whole codebase
# =============================================================================
# The stored condition tokens are "FAB"/"INLB": they name the labeling condition in the data-file
# names (Experiment_<KIND>_Cell_<n>_<span>S_RAW.tif) and in the `kinds` field of every Experiment
# output. The scientific display names prepend the receptor: MET-FAB (the monomer control) and
# MET-INLB (the dimer condition) appear on the command line, in reports, in figure legends, and in
# output directory names. The tokens are reserved for conditions only; the repo-iteration suffixes
# (alp, bet, ...) are a different namespace and never name a condition.
#
# One literal defines the mapping; both directions and the CLI choices derive from it, so no second
# copy can quietly acquire a different spelling.
CONDITION_DISPLAY = {"FAB": "MET-FAB", "INLB": "MET-INLB"}        # stored token -> scientific name
KIND_OF_CONDITION = {v: k for k, v in CONDITION_DISPLAY.items()}  # scientific name -> stored token


def condition_display(kind):
    """Scientific name for a stored condition token, passing unknown tokens through unchanged."""
    return CONDITION_DISPLAY.get(kind, kind)


def condition_token(condition):
    """Stored condition token for a scientific name, passing unknown names through unchanged."""
    return KIND_OF_CONDITION.get(condition, condition)


def discover_cells(experiment_dir, kind, span):
    """Return the sorted cell indices with a recording on disk for a given kind.

    Globs ``Experiment_{kind}_Cell_*_{span}S_RAW.tif`` under ``experiment_dir`` and
    parses the integer cell index out of each name. Shared by both Detector real-data
    stages so they see exactly the same recordings.
    """
    cells = []
    for path in experiment_dir.glob(f"Experiment_{kind}_Cell_*_{span}S_RAW.tif"):
        match = re.search(rf"Cell_(\d+)_{span}S_RAW", path.name)
        if match:
            cells.append(int(match.group(1)))
    return sorted(cells)


# The recording layouts this repository reads: ONE 3-D TIFF series whose first axis is the frame
# sequence (tifffile names it T for a time axis, I for a generic image sequence, Q when the writer
# declared none) followed by the two image axes. Every other layout -- a single page holding a 3-D
# image, an extra channel or sample axis, several series -- is refused BEFORE estimation, because
# the frame count could otherwise be misread (a page count is not a frame count).
ACCEPTED_RECORDING_AXES = ("TYX", "IYX", "QYX")


class RecordingLayoutError(ValueError):
    """A recording's TIFF layout is not one the reader supports; the message names the file."""


def inspect_recording(tif_path):
    """The frame count and image shape ``(n_frames, height, width)`` of one recording, read from
    its TIFF series metadata without loading pixels. This is the ONE place the layout is
    interpreted: :func:`read_cell_chunks` reads the same series and checks the loaded array
    against this shape, so the window inventory and the windows actually cut agree by
    construction. Raises :class:`RecordingLayoutError` for any layout outside
    :data:`ACCEPTED_RECORDING_AXES`."""
    with tifffile.TiffFile(str(tif_path)) as tif:
        if len(tif.series) != 1:
            raise RecordingLayoutError(
                f"{tif_path}: {len(tif.series)} TIFF series; exactly one frame stack is supported.")
        series = tif.series[0]
        axes, shape = str(series.axes), tuple(int(s) for s in series.shape)
    if axes not in ACCEPTED_RECORDING_AXES or len(shape) != 3:
        raise RecordingLayoutError(
            f"{tif_path}: TIFF series axes {axes!r} with shape {shape} is not a supported recording "
            f"layout; supported: one 3-D series with axes in {list(ACCEPTED_RECORDING_AXES)} "
            f"(frames first, then image rows and columns).")
    return shape


def chunk_count(n_available, n_frames, step_frames) -> int:
    """Number of model-length windows a recording of ``n_available`` frames yields under the
    window / stride geometry; the same arithmetic :func:`read_cell_chunks` slices with."""
    return max(0, (int(n_available) - int(n_frames)) // int(step_frames) + 1)


def preflight_recordings(tif_paths):
    """Inspect every selected recording up front and return ``{path: (n_frames, H, W)}``. An
    unsupported layout is reported for ALL offending files at once, before any estimation, so a
    long run cannot discover it near its end."""
    shapes, problems = {}, []
    for p in tif_paths:
        try:
            shapes[Path(p)] = inspect_recording(p)
        except RecordingLayoutError as exc:
            problems.append(str(exc))
    if problems:
        raise RecordingLayoutError(
            f"{len(problems)} recording(s) have an unsupported TIFF layout:\n  "
            + "\n  ".join(problems))
    return shapes


def read_cell_chunks(tif_path, n_frames, step_frames):
    """Read one recording and cut it into model-length windows.

    Interprets the layout through :func:`inspect_recording`, loads that one series, converts the
    16-bit raw frames to 8-bit (the model's input domain), and returns the list of
    ``(n_frames, H, W)`` uint8 windows stepped by ``step_frames`` (``1 s`` step -> maximal
    overlap; a step equal to the window -> non-overlapping tiling). The number of windows equals
    :func:`chunk_count` of the inspected frame count. Identical windowing in every stage.
    """
    expected_shape = inspect_recording(tif_path)
    with tifffile.TiffFile(str(tif_path)) as tif:
        raw = tif.series[0].asarray()                    # (frames, H, W) uint16
    if tuple(raw.shape) != expected_shape:
        raise RecordingLayoutError(f"{tif_path}: loaded array shape {raw.shape} differs from the "
                                   f"series metadata {expected_shape}.")
    video8 = convert_video_dtype(raw, bits_from=16, bits_to=8)
    chunks = [video8[start:start + n_frames]
              for start in range(0, video8.shape[0] - n_frames + 1, step_frames)]
    assert len(chunks) == chunk_count(video8.shape[0], n_frames, step_frames)
    return chunks


def shard_by_rank(items, topo):
    """This worker's round-robin slice of ``items`` (``world_size == 1`` -> all of them).

    Splits a flat work list across workers by ``i % world_size == rank``; each worker
    gets a disjoint subset and together they cover every item exactly once.
    """
    return [w for i, w in enumerate(items) if i % topo.world_size == topo.rank]


def shard_path(out_dir, rank, world_size):
    """Path of one worker's partial-array shard in a multi-GPU sharded run."""
    return Path(out_dir) / f"_shard_{rank:02d}_of_{world_size:02d}.npz"


def save_shard(out_dir, topo, arrays, *, count, write_empty: bool = False):
    """Write this worker's partial arrays as a shard; return the path, or ``None``.

    Creates ``out_dir`` and saves ``arrays`` (a ``{name: array}`` dict) to this rank's shard
    path as a compressed ``.npz``. When ``count == 0`` (this worker drew no work) the shard is
    written only under ``write_empty`` -- the Evaluation / Experiment stages pass it, because
    their merge requires every rank to account for itself (a valid zero-observation shard, built
    with :func:`artifact_schema.empty_product_arrays`); the pool merges of other stages keep the
    old behavior and simply see fewer files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if count == 0 and not write_empty:
        return None
    path = shard_path(out_dir, topo.rank, topo.world_size)
    np.savez_compressed(str(path), **arrays)
    return path


def load_shards(out_dir):
    """Sorted list of the ``_shard_*_of_*.npz`` files a sharded run wrote in ``out_dir``."""
    return sorted(Path(out_dir).glob("_shard_*_of_*.npz"))


def merge_shard_arrays(shard_paths, *, concat_keys, first_keys=(),
                       optional_concat_keys=()):
    """Combine per-worker shard arrays into one merged ``{name: array}`` dict.

    Reads every shard in ``shard_paths`` and, treating ``concat_keys[0]`` as the primary
    key, skips any shard whose primary array is empty (a defensive no-op -- a worker that
    drew no work writes no shard at all). For every used (non-empty) shard it:

    - appends each ``concat_keys`` array (concatenated on axis 0 in the result);
    - captures each ``first_keys`` value from the first used shard (taken as-is -- these
      are per-run constants such as the condition list, identical across shards);
    - appends each ``optional_concat_keys`` array when present, and marks the key absent
      if any used shard lacks it.

    Returns ``(merged, n_used)`` where ``merged`` has each concat key concatenated, each
    first key from the first used shard, and each optional concat key concatenated only
    if it was present in EVERY used shard (otherwise the key is omitted from ``merged``).
    Raises ``ValueError`` if ``shard_paths`` is empty or every shard was empty.
    """
    if not shard_paths:
        raise ValueError("merge_shard_arrays: no shard files were provided.")
    primary = concat_keys[0]
    collected = {key: [] for key in concat_keys}
    optional = {key: [] for key in optional_concat_keys}
    optional_present_all = {key: True for key in optional_concat_keys}
    firsts = {}
    n_used = 0
    for shard in shard_paths:
        with np.load(str(shard)) as data:
            if data[primary].shape[0] == 0:
                continue   # defensive: a zero-item shard contributes nothing
            n_used += 1
            for key in concat_keys:
                collected[key].append(data[key])
            if n_used == 1:
                for key in first_keys:
                    firsts[key] = data[key]
            for key in optional_concat_keys:
                if key in data:
                    optional[key].append(data[key])
                else:
                    optional_present_all[key] = False
    if n_used == 0:
        raise ValueError("merge_shard_arrays: every shard was empty (no items).")
    merged = {key: np.concatenate(collected[key], axis=0) for key in concat_keys}
    for key in first_keys:
        merged[key] = firsts[key]
    for key in optional_concat_keys:
        if optional_present_all[key] and optional[key]:
            merged[key] = np.concatenate(optional[key], axis=0)
    return merged, n_used


def assert_complete_shard_set(shard_paths, *, allow_partial: bool = False,
                              partial_option: str | None = "--allow-partial") -> int:
    """Guard a shard set against MISSING shards before merging (on top of the consistency
    check of :func:`assert_consistent_shard_set`).

    A sharded run writes exactly ``world_size`` shards (the Evaluation / Experiment stages write
    one even for a rank that drew no work); fewer means a rank died before saving (a killed
    worker, a node failure) and merging the rest would silently produce a report over a subset
    of the work while deleting the evidence. Raises ``ValueError`` naming the missing ranks
    unless ``allow_partial`` is set, in which case the caller is expected to record the
    incomplete coverage in its report. ``partial_option`` names the caller's opt-out flag in the
    message; stages without one pass ``None`` and the message recommends recomputing the rank
    instead. Returns the ``world_size``.
    """
    world_size = assert_consistent_shard_set(shard_paths)
    present = sorted(int(re.search(r"_shard_(\d+)_of_", Path(p).name).group(1)) for p in shard_paths)
    missing = sorted(set(range(world_size)) - set(present))
    if missing and not allow_partial:
        remedy = (f"Rerun the stage, or pass {partial_option} to merge what exists (the report then "
                  f"covers only the present shards)." if partial_option else
                  f"Recompute the missing rank(s) with RANK=<r> WORLD_SIZE={world_size} LOCAL_RANK=0, "
                  f"the same arguments, the same invocation identifier and the same repository "
                  f"state (HEAD, dirty flag, implementation files); save progress.log and figures/ "
                  f"first if rank 0 is among them. Then merge again; there is no partial merge for "
                  f"this stage (recipe: Script_Bank/HPC/README.md).")
        raise ValueError(
            f"{len(shard_paths)} of {world_size} shards present; missing rank(s) {missing}. A rank "
            f"died before saving its shard (see the job log). {remedy}")
    return world_size


def assert_consistent_shard_set(shard_paths):
    """Guard a shard set against a stale-plus-fresh mix before merging.

    Each shard is named ``_shard_{rank}_of_{world_size}.npz``; one clean sharded run
    writes at most ``world_size`` shards, all carrying the SAME ``world_size``. If a run
    crashes mid-write and a later run with a DIFFERENT worker count writes into the same
    directory, the directory holds shards from two runs and merging them would silently
    concatenate a stale partial with the fresh result. This raises ``ValueError`` when the
    shards do not all share one ``world_size``, or when there are more shards than that
    ``world_size`` -- so the merge fails loudly instead of producing a contaminated result.
    A same-``world_size`` rerun overwrites its predecessor's shards by identical filename,
    so this passes (no contamination). Returns the common ``world_size``.
    """
    world_sizes = set()
    for path in shard_paths:
        match = re.search(r"_shard_(\d+)_of_(\d+)\.npz$", Path(path).name)
        if match is None:
            raise ValueError(f"unrecognized shard filename {Path(path).name!r}.")
        world_sizes.add(int(match.group(2)))
    if len(world_sizes) != 1:
        raise ValueError(
            f"shards carry inconsistent world_size {sorted(world_sizes)} (a stale-plus-fresh mix).")
    (world_size,) = world_sizes
    if len(shard_paths) > world_size:
        raise ValueError(
            f"{len(shard_paths)} shard files but world_size={world_size} (stale shards from a prior run).")
    return world_size


def merge_validated_shards(shard_paths, *, stage, concat_keys, first_keys=(),
                           optional_concat_keys=(), expected_ids=None):
    """Load, validate and combine the shards of ONE computation into a merged product.

    Every shard is read through :func:`artifact_schema.load_product`, so an obsolete-schema shard,
    a shard missing a required estimate, a non-finite estimate or a duplicated observation is
    refused with the shard named. The shard manifests must then agree on every contract key
    (:func:`artifact_schema.assert_compatible_manifests`) -- shards of two different computations
    landing in one directory are not concatenated. ``concat_keys`` are concatenated on axis 0 in
    shard order; ``first_keys`` are per-run constants that must be EQUAL on every shard (the
    first shard's value is kept; a disagreeing shard is refused). Optional arrays are a RUN-level
    setting (``stored_optional_fields``, a contract key): every shard must store the same ones --
    mixed presence is refused with every shard on each side named -- and each stored one must be
    listed in ``optional_concat_keys``; any array a shard stores that this merge would not carry
    is refused rather than dropped. The merged observation identifiers must be unique and, when
    ``expected_ids`` is given, must equal that inventory exactly -- the right count with the
    wrong members is refused.

    The logical invocation (``run_identity``: product label and invocation id) is part of the
    contract; the execution attempt (``execution``: the Slurm job that ran a shard) is not, so a
    missing rank recomputed in a new job, or locally, merges with the original shards. The merged
    manifest keeps every shard's execution under ``shards`` and records the merge step's own.

    Returns ``(merged, manifest, n_shards)`` where ``merged`` is a ``{name: array}`` dict that
    already carries the merged manifest under :data:`artifact_schema.MANIFEST_KEY`.
    """
    from . import artifact_schema as schema

    if not shard_paths:
        raise ValueError("merge_validated_shards: no shard files were provided.")
    # A zero-observation shard (a rank that drew no work) is a valid shard here; the MERGED
    # product is validated below without that allowance, so a run whose every rank was empty
    # fails with "no observations" instead of producing a report over nothing.
    loaded = [schema.load_product(p, stage=stage, allow_empty=True) for p in shard_paths]
    # Optional storage (e.g. --dump-posterior-samples) is a run-level setting. Mixed presence --
    # say a replacement rank recomputed without the flag -- is refused here, naming every shard on
    # each side, before the generic contract comparison would name only the first disagreement.
    # Merging would otherwise drop the stored values without a trace.
    for key in schema.OPTIONAL_FIELDS[stage]:
        have = [str(p) for (a, _), p in zip(loaded, shard_paths) if key in a]
        lack = [str(p) for (a, _), p in zip(loaded, shard_paths) if key not in a]
        if have and lack:
            raise schema.SchemaError(
                f"optional field {key!r} is stored by {len(have)} shard(s) and absent from "
                f"{len(lack)}: absent from {lack}; stored by {have}. Optional storage is a "
                f"run-level setting (stored_optional_fields); merging would silently drop the "
                f"stored values, so these shards are not merged. Recompute the inconsistent "
                f"rank(s) with the run's setting.")
    contract = schema.assert_compatible_manifests([m for _, m in loaded],
                                                  sources=[str(p) for p in shard_paths])
    # Everything a shard stores is carried into the merged product, or the merge is refused.
    stored_optional = list(contract["stored_optional_fields"])
    not_carried = [k for k in stored_optional if k not in optional_concat_keys]
    if not_carried:
        raise schema.SchemaError(
            f"the shards store optional field(s) {not_carried} that this merge does not carry "
            f"(optional_concat_keys={list(optional_concat_keys)}); refusing rather than dropping "
            f"them.")
    carried = set(concat_keys) | set(first_keys) | set(stored_optional) | {schema.MANIFEST_KEY}
    for (a, _), p in zip(loaded, shard_paths):
        extra = sorted(set(a) - carried)
        if extra:
            raise schema.SchemaError(
                f"{p}: stored field(s) {extra} would not be carried into the merged product; "
                f"refusing rather than dropping them.")
    merged = {key: np.concatenate([a[key] for a, _ in loaded], axis=0) for key in concat_keys}
    for key in first_keys:
        ref = np.asarray(loaded[0][0][key])
        for (a, _), p in zip(loaded[1:], shard_paths[1:]):
            if not np.array_equal(ref, np.asarray(a[key])):
                raise schema.SchemaError(
                    f"{p}: per-run field {key!r} {np.asarray(a[key]).tolist()} differs from "
                    f"{shard_paths[0]} {ref.tolist()}; shards whose {key!r} mapping differs are "
                    f"not merged (their per-row indices would mean different things).")
        merged[key] = ref
    for key in stored_optional:
        merged[key] = np.concatenate([a[key] for a, _ in loaded], axis=0)
    n = int(merged[concat_keys[0]].shape[0])
    schema.assert_unique_observations(schema.observation_ids(merged, stage),
                                      source="merged shards", expected=expected_ids)
    manifest = schema.merged_manifest(contract, n_observations=n,
                                      shard_manifests=[m for _, m in loaded],
                                      execution=schema.execution_identity())
    merged[schema.MANIFEST_KEY] = schema.encode_manifest(manifest)
    schema.validate_product(merged, stage=stage, source="merged product")
    return merged, manifest, len(loaded)


class RetiredOption(argparse.Action):
    """A CLI option that no longer exists: naming it is an error, not a silent no-op. Used for the
    estimate-selection option retired in 0.1.16, when every run began computing and storing all
    three point estimates (map, median, sgm)."""

    def __init__(self, option_strings, dest, **kwargs):
        kwargs.setdefault("nargs", "?")
        kwargs.setdefault("help", argparse.SUPPRESS)   # a trap, not an option: absent from --help
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string} was retired in 0.1.16: every run computes and stores all "
                     f"three point estimates (map, median, sgm). Remove the option.")
