"""Experimental aggregation: representative collections from the window posteriors of the Experiment stage.

Shared by both workflows (biology and detector; the parameter table supplies keys, prior box and the
log/linear conversion of every row). The Experiment stage infers one posterior per (recording, window)
and retains its summary draws (``posterior_samples_cloud``). This module groups those posteriors into
explicitly defined collections of representative vectors and recomputes the draw-derived summaries of
every group from the retained draws. It performs no neural inference and optimizes no MAP.

CONSTRUCTIONS (the canonical options table of DETECTOR_WORKFLOW.md, "Experimental aggregation"). Let C
be the cells of one condition and S a non-empty set of window positions (zero-based chunk indices)
shared by those cells. ``q_cw`` is the retained cloud of cell c, window w (equal draw counts per window,
so concatenation realizes the equal weights directly):

    per_cell             one representative per cell: pool the cell's windows in S (weight 1/|S| each)
    per_window_position  one representative per position w in S: pool the cells at w (weight 1/C each)
    global               one representative of the selected collection: pool every (c, w), c in C, w in S
    per_cell_window      one representative per (cell, window): the window's own posterior

Each group stores the SGM (the complete retained draw nearest the collection's geometric center: the
exact sample medoid for a single-window group, reused from the window's stored SGM when its cloud,
metric and coordinates match; the collection kernel's Weiszfeld member snapped to an actual draw for a
pooled group above ``exact_capacity`` rows, recorded as ``weiszfeld_snap``), its source identity (cell,
chunk, draw index, collection), the marginal-median vector (a coordinate-wise composite, never claimed
as a draw) and the marginal quantiles of the pooled draws (mixture quantiles in estimator coordinates;
physical endpoints are the transform of the quantile). Distances use estimator coordinates divided by the
prior widths (``sgm_scale``), the per-window convention of the stage products. Unrestricted draws stay
unrestricted: outside-box mass is counted per group and per coordinate, never clipped.

WINDOW SETS. ``all`` (every position the declared window geometry admits: chunk indices
0 .. (span - n_frames) // step) and ``opening`` (chunk 0) are the standard sets; an explicitly named set is
``NAME=a-b`` (a range) or ``NAME=a+b+c`` (a list). Positions are validated against the declared geometry before
any neural work (:func:`validate_window_sets_against_geometry`), and every included cell must carry every
position a set requests (:func:`validate_coverage`): a position missing from every cell is a refusal for
``all``, never a silent redefinition of the acquisition period; a deliberate subset is named explicitly.
Every construction is computed for every requested set; equivalent cases (a single-window set makes ``per_cell`` equal ``per_cell_window``
and ``per_window_position`` equal ``global``; a single cell makes ``per_window_position`` equal
``per_cell_window`` and ``global`` equal ``per_cell``) are detected and reuse the computed group.

INPUTS. A :class:`CloudCollection` built from an Experiment product that stored its clouds
(:func:`cloud_from_product`) or from retained draw collections (:func:`cloud_from_collection`, the
``*_Window_Posterior_Draws[_<LABEL>].npz`` files) combined under one contract (:func:`combine_clouds`):
checkpoint, parameter order, coordinate transform, distance scale, window geometry, draw count and
sampling mode must agree, or the combination is refused naming the offending source; the input conversion
(preprocessing) of the collections is compared between them as well as against the reference product, and
the merged contract records whether it was verified or is unavailable; collection labels, seeds and
implementation hashes may differ and are disclosed. Coverage must be complete (every included cell carries
every requested position; duplicate identities refused with their identities listed), supplied identities
must belong to the source Experiment product (:func:`verify_identities_in_product`), and conditions are
never pooled together.

OUTPUTS. :func:`write_aggregation_exports` writes the compact candidate archive
(``aggregation_candidates.npz``: every group's vectors, quantiles, identities, methods, the contract and the
sources), ``aggregation_groups.csv``, ``aggregation_comparison.csv`` and ``aggregation_manifest.json``
beside the Experiment report; the report section itself is written by :func:`report_aggregation` with the
stage's reporter, for a live run and for a CPU regeneration alike. Representative-distribution quantiles
use the inverse weighted empirical CDF (:func:`weighted_quantile`); raw posterior quantiles keep the
stage's interpolated convention.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import artifact_schema as schema
from .evaluation import sample_geometric_median as exact_window_sgm
from .parameterization import is_log_row, to_physical
from .sample_geometric_median import EXACT_MEDOID_CAPACITY, sample_geometric_median as collection_sgm

CONSTRUCTIONS = ("per_cell", "per_window_position", "global", "per_cell_window")
CONSTRUCTION_DEFINITIONS = {
    "per_cell": ("One representative per cell", "pool all retained draws of the cell's selected windows, "
                 "equal weight per window, and select the SGM", "variation between cell representatives "
                 "over the declared acquisition period"),
    "per_window_position": ("One representative per window position", "at each selected position, pool the "
                            "retained draws across the included cells, equal weight per cell, and select the "
                            "SGM", "variation between acquisition-period representatives, cell differences "
                            "compressed at each position"),
    "global": ("One representative of the selected collection", "pool the retained draws with equal cell "
               "weight and equal selected-window weight within each cell, and select the SGM",
               "an overall joint center of the selected collection"),
    "per_cell_window": ("One representative per cell-window pair", "the SGM of each included cell-window "
                        "posterior", "variation between window representatives, cell and window identities "
                        "retained"),
}
STANDARD_WINDOW_SETS = ("all", "opening")
DEFAULT_WINDOW_SETS = "all,opening"
# A pooled group above this many rows uses the collection kernel's Weiszfeld member (snapped to an actual
# draw); at or below it the exact medoid, which allocates a quadratic distance matrix of at most this many
# rows squared. A multi-window pool at the stage's draw count always exceeds it, and a single-window group
# reuses the window's stored exact SGM, so the standard large-pool route never allocates a quadratic matrix;
# a small exact group (a pool at or below the capacity) still does, within this bound.
POOLED_EXACT_CAPACITY = 10_000
STATISTICS = ("sgm", "median")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class AggregationError(ValueError):
    """An input the aggregation refuses; the message names the offending identities."""


# =============================================================================
# Window sets
# =============================================================================
def parse_window_sets(spec) -> dict:
    """``"all,opening,late=5-9,odd=1+3+5"`` -> ``{"all": None, "opening": (0,), "late": (5,...,9), "odd": (1,3,5)}``
    in the given order (``None`` = every validated position). Refuses an empty, duplicate or malformed item."""
    out = {}
    for raw in str(spec or "").split(","):
        item = raw.strip()
        if not item:
            continue
        if item in out:
            raise AggregationError(f"window set {item!r} is named twice.")
        if item == "all":
            out[item] = None
            continue
        if item == "opening":
            out[item] = (0,)
            continue
        if "=" not in item:
            raise AggregationError(f"window set {item!r}: use 'all', 'opening', NAME=a-b or NAME=a+b+c.")
        name, members = (s.strip() for s in item.split("=", 1))
        if not _NAME.match(name) or name in STANDARD_WINDOW_SETS:
            raise AggregationError(f"window set name {name!r} is not an identifier or shadows a standard set.")
        try:
            if "-" in members:
                a, b = (int(v) for v in members.split("-", 1))
                chunks = tuple(range(a, b + 1))
            else:
                chunks = tuple(int(v) for v in members.split("+"))
        except ValueError:
            raise AggregationError(f"window set {name!r}: members {members!r} are not integers.")
        if not chunks or len(set(chunks)) != len(chunks) or any(c < 0 for c in chunks):
            raise AggregationError(f"window set {name!r}: members {chunks} must be distinct non-negative positions.")
        out[name] = tuple(sorted(chunks))
    if not out:
        raise AggregationError("no window set requested (at least 'all' or 'opening').")
    return out


def format_window_set(name, chunks) -> str:
    return f"{name} = chunks {list(chunks)}"


# =============================================================================
# Cloud collections
# =============================================================================
CONTRACT_FIELDS = ("parameter_keys", "coordinate_transform", "sgm_scale", "pool_mode", "n_summary_draws",
                   "checkpoint_sha256", "window_geometry")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class CloudCollection:
    """Retained window clouds with their identities and the computation contract they were drawn under."""
    cloud: np.ndarray            # (W, S, D) estimator coordinates
    kind_index: np.ndarray       # (W,)
    cell: np.ndarray             # (W,)
    chunk: np.ndarray            # (W,)
    kinds: list
    contract: dict               # CONTRACT_FIELDS (+ input_conversion when the source records it)
    sources: list                # one dict per source file (path, sha256, label, disclosed facts)
    source_index: np.ndarray     # (W,) -> sources
    window_sgm: np.ndarray = None        # (W, D) the windows' stored exact SGMs, when available
    window_sgm_index: np.ndarray = None  # (W,) row of the SGM in its window's cloud (-1 = unknown)

    @property
    def n_windows(self) -> int:
        return int(self.cloud.shape[0])

    @property
    def n_draws(self) -> int:
        return int(self.cloud.shape[1])

    @property
    def keys(self) -> list:
        return list(self.contract["parameter_keys"])


def _locate_window_sgms(cloud, sgm) -> np.ndarray:
    """Row index of each window's stored SGM inside its cloud (exact match), -1 where none matches."""
    idx = np.full(cloud.shape[0], -1, dtype=np.int64)
    for i in range(cloud.shape[0]):
        hit = np.flatnonzero(np.all(cloud[i].astype(np.float64) == np.asarray(sgm[i], dtype=np.float64), axis=1))
        if hit.size:
            idx[i] = int(hit[0])
    return idx


def cloud_from_product(arrays, manifest, *, path=None, label="experiment product") -> CloudCollection:
    """The clouds an Experiment product stored (``posterior_samples_cloud``) with that product's contract."""
    if "posterior_samples_cloud" not in arrays:
        raise AggregationError(f"{label}: no retained clouds (posterior_samples_cloud); the aggregation needs "
                               f"the stored draws, which this product did not retain.")
    cloud = np.asarray(arrays["posterior_samples_cloud"])
    contract = {k: manifest[k] for k in CONTRACT_FIELDS}
    contract["input_conversion"] = manifest.get("input_conversion")
    contract["input_conversion_status"] = ("recorded by the product" if manifest.get("input_conversion") is not None
                                           else "unavailable: the product does not record its input conversion")
    source = {"path": None if path is None else str(path), "sha256": None if path is None else _sha256(path),
              "label": label, "kind": "experiment_product", "collection_label": None,
              "seed_policy": manifest.get("seed_policy"), "written_at": manifest.get("written_at"),
              "implementation_sha256": (manifest.get("code") or {}).get("implementation", {}).get("sha256"),
              "run_identity": manifest.get("run_identity")}
    sgm = np.asarray(arrays["posterior_sgm"], dtype=float)
    return CloudCollection(cloud=cloud, kind_index=np.asarray(arrays["kind_index"]).astype(np.int64),
                           cell=np.asarray(arrays["cell"]).astype(np.int64), chunk=np.asarray(arrays["chunk"]).astype(np.int64),
                           kinds=[str(k) for k in np.asarray(arrays["kinds"]).ravel()], contract=contract, sources=[source],
                           source_index=np.zeros(cloud.shape[0], np.int64), window_sgm=sgm,
                           window_sgm_index=_locate_window_sgms(cloud, sgm))


def cloud_from_collection(path) -> CloudCollection:
    """A retained draw collection (``*_Window_Posterior_Draws[_<LABEL>].npz``: clouds, identities, fresh
    SGMs with their row indices, and ``collection_manifest_json``), read-only."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        needed = ("posterior_samples_cloud", "kind_index", "cell", "chunk", "kinds", "fresh_sgm", "fresh_sgm_index",
                  "parameter_keys", "sgm_scale", "collection_manifest_json")
        missing = [k for k in needed if k not in data.files]
        if missing:
            raise AggregationError(f"{path}: not a retained draw collection (missing {missing}).")
        man = json.loads(str(data["collection_manifest_json"]))
        cloud = data["posterior_samples_cloud"]
        kind_index = data["kind_index"].astype(np.int64); cell = data["cell"].astype(np.int64); chunk = data["chunk"].astype(np.int64)
        kinds = [str(k) for k in data["kinds"].ravel()]
        sgm = data["fresh_sgm"].astype(float); sgm_index = data["fresh_sgm_index"].astype(np.int64)
        keys = [str(k) for k in data["parameter_keys"]]
    if keys != [str(k) for k in man["parameter_keys"]]:
        raise AggregationError(f"{path}: parameter_keys array {keys} != manifest {man['parameter_keys']}.")
    contract = {"parameter_keys": keys, "coordinate_transform": str(man["coordinate_transform"]),
                "sgm_scale": [float(s) for s in man["sgm_scale"]], "pool_mode": str(man["pool_mode"]),
                "n_summary_draws": int(man["n_summary_draws"]), "checkpoint_sha256": man["checkpoint_sha256"],
                "window_geometry": dict(man["window_geometry"]), "input_conversion": man.get("input_conversion")}
    code = man.get("code") or {}
    source = {"path": str(path), "sha256": _sha256(path), "label": man.get("collection_label") or path.stem,
              "kind": "retained_draw_collection", "collection_label": man.get("collection_label"),
              "seed_policy": man.get("seed_policy"), "written_at": man.get("written_at"),
              "implementation_sha256": (code.get("implementation") or {}).get("sha256"),
              "source_product": (man.get("source_product") or {}).get("path"),
              "source_product_sha256": (man.get("source_product") or {}).get("sha256"),
              "cells": man.get("cells"), "execution": man.get("execution")}
    bad = np.flatnonzero((sgm_index < 0) | (sgm_index >= cloud.shape[1]))
    if bad.size or not np.allclose(cloud[np.arange(cloud.shape[0]), sgm_index].astype(float), sgm):
        raise AggregationError(f"{path}: fresh_sgm is not the cloud row at fresh_sgm_index for every window.")
    return CloudCollection(cloud=cloud, kind_index=kind_index, cell=cell, chunk=chunk, kinds=kinds, contract=contract,
                           sources=[source], source_index=np.zeros(cloud.shape[0], np.int64), window_sgm=sgm,
                           window_sgm_index=sgm_index)


def contract_differences(contract, reference) -> list:
    """Scientific-contract fields on which ``contract`` differs from ``reference`` (empty = compatible)."""
    out = []
    for k in CONTRACT_FIELDS:
        a, b = contract.get(k), reference.get(k)
        if k == "sgm_scale":
            same = np.allclose(np.asarray(a, float), np.asarray(b, float))
        else:
            same = json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
        if not same:
            out.append(f"{k}: {a!r} != {b!r}")
    a, b = contract.get("input_conversion"), reference.get("input_conversion")
    if a is not None and b is not None and a != b:
        out.append(f"input_conversion: {a!r} != {b!r}")
    return out


def ref_label_conv(label):
    return f"the reference ({label})"


def combine_clouds(collections, *, reference_contract=None, reference_label="reference") -> CloudCollection:
    """Concatenate retained collections drawn under ONE computation contract (checkpoint, keys, transform,
    scale, window geometry, draw count, sampling mode). Refuses a contract disagreement naming the source
    and the field, duplicate (kind, cell, chunk) identities and differing condition lists; collection
    labels, seeds and implementation hashes may differ and are left in ``sources`` for disclosure."""
    collections = list(collections)
    if not collections:
        raise AggregationError("no cloud collection to combine.")
    ref = dict(reference_contract) if reference_contract is not None else dict(collections[0].contract)
    # The preprocessing contract is compared BETWEEN the collections too: a reference product without the
    # field (a legacy product) does not excuse a disagreement between two known collection fields.
    convs = [(c.sources[0].get("path") or c.sources[0]["label"], c.contract.get("input_conversion")) for c in collections]
    known = sorted({v for _, v in convs if v is not None})
    if len(known) > 1:
        raise AggregationError("input_conversion differs between the collections: "
                               + "; ".join(f"{p}: {v!r}" for p, v in convs)
                               + ". Collections preprocessed differently are not pooled, whether or not the reference records a conversion.")
    ref_conv = ref.get("input_conversion")
    if known and ref_conv is not None and ref_conv != known[0]:
        raise AggregationError(f"input_conversion {known[0]!r} of the collections != {ref_label_conv(reference_label)} {ref_conv!r}.")
    ref["input_conversion"] = known[0] if known else ref_conv
    unrecorded = [p for p, v in convs if v is None]
    if known and not unrecorded:
        ref["input_conversion_status"] = (f"verified: every source ({len(convs)}) records the same input conversion"
                                          + ("" if ref_conv is not None else "; the reference product does not record one"))
    elif known:
        ref["input_conversion_status"] = f"unverified: no recorded input conversion for {unrecorded}"
    else:
        ref["input_conversion_status"] = "unavailable: no source records its input conversion"
    for c in collections:
        diff = contract_differences(c.contract, ref)
        if diff:
            raise AggregationError(f"{c.sources[0]['path']}: computation contract differs from {reference_label}: "
                                   + "; ".join(diff) + ". Collections of different computations are not pooled.")
        if c.kinds != collections[0].kinds:
            raise AggregationError(f"{c.sources[0]['path']}: condition labels {c.kinds} != {collections[0].kinds}.")
        if c.cloud.shape[1:] != collections[0].cloud.shape[1:]:
            raise AggregationError(f"{c.sources[0]['path']}: cloud shape {c.cloud.shape[1:]} != {collections[0].cloud.shape[1:]}.")
    sources, src_idx = [], []
    for c in collections:
        base = len(sources)
        sources.extend(c.sources)
        src_idx.append(c.source_index + base)
    merged = CloudCollection(
        cloud=np.concatenate([c.cloud for c in collections], axis=0),
        kind_index=np.concatenate([c.kind_index for c in collections]), cell=np.concatenate([c.cell for c in collections]),
        chunk=np.concatenate([c.chunk for c in collections]), kinds=list(collections[0].kinds), contract=ref,
        sources=sources, source_index=np.concatenate(src_idx),
        window_sgm=(None if any(c.window_sgm is None for c in collections)
                    else np.concatenate([c.window_sgm for c in collections], axis=0)),
        window_sgm_index=(None if any(c.window_sgm_index is None for c in collections)
                          else np.concatenate([c.window_sgm_index for c in collections])))
    ids = np.stack([merged.kind_index, merged.cell, merged.chunk], axis=1)
    rows = [tuple(int(v) for v in r) for r in ids]
    seen, dup = set(), []
    for r in rows:
        (dup.append(r) if r in seen else seen.add(r))
    if dup:
        raise AggregationError(f"duplicate window identities (kind_index, cell, chunk) across the collections: "
                               f"{sorted(set(dup))[:10]}{' ...' if len(set(dup)) > 10 else ''}.")
    return merged


# =============================================================================
# Groups and the aggregation
# =============================================================================
@dataclass
class Group:
    construction: str
    window_set: str
    kind: str
    cell: int            # -1 when the group spans cells
    chunk: int           # -1 when the group spans positions
    members: list        # [(cell, chunk), ...] windows pooled, canonical order
    n_draws: int
    sgm_log10: np.ndarray
    sgm_source: dict     # cell, chunk, draw_index, source_label, flat_index
    sgm_method: str      # exact_medoid | weiszfeld_snap | window_stored
    median_log10: np.ndarray
    quantiles_log10: np.ndarray    # (D, Q)
    outside_box_fraction: float
    outside_box_by_key: np.ndarray
    equivalent_to: str = ""        # "construction[window_set]" whose group this one reuses


@dataclass
class AggregationResult:
    groups: list
    window_sets: dict             # name -> tuple of chunks (resolved)
    coverage: dict                # per kind: cells, chunks, n_windows, n_draws
    contract: dict
    sources: list
    keys: list
    levels: tuple
    scale: np.ndarray
    prior_low: np.ndarray
    prior_high: np.ndarray
    exact_capacity: int
    timing: dict = field(default_factory=dict)
    equivalences: list = field(default_factory=list)

    def select(self, *, construction=None, window_set=None, kind=None):
        return [g for g in self.groups if (construction is None or g.construction == construction)
                and (window_set is None or g.window_set == window_set) and (kind is None or g.kind == kind)]


def _canonical_order(coll: CloudCollection) -> np.ndarray:
    return np.lexsort((coll.chunk, coll.cell, coll.kind_index))


def expected_positions(geometry, *, observed=None) -> tuple:
    """The window positions the declared geometry admits: chunk indices ``0 .. (span - n_frames) // step``
    (the stage's ``chunk_count`` arithmetic). A geometry without a nominal span (``span_frames`` None) falls
    back to the ``observed`` positions of the source inventory when given."""
    n, step, span = int(geometry["n_frames"]), int(geometry["step_frames"]), geometry.get("span_frames")
    if n <= 0 or step <= 0:
        raise AggregationError(f"window geometry {geometry} has no positive window length and step.")
    if span is None:
        if observed is None:
            raise AggregationError(f"window geometry {geometry} declares no recording span and no inventory was given.")
        return tuple(sorted(int(w) for w in observed))
    if int(span) < n:
        raise AggregationError(f"window geometry {geometry}: the recording span is shorter than one window.")
    return tuple(range((int(span) - n) // step + 1))


def validate_window_sets_against_geometry(window_sets: dict, geometry, *, observed=None) -> dict:
    """Preflight: resolve ``all`` to the declared positions and refuse a named position beyond them, before
    any neural work. Returns ``{name: tuple}``."""
    expected = expected_positions(geometry, observed=observed)
    resolved = {}
    for name, members in window_sets.items():
        chunks = expected if members is None else tuple(int(w) for w in members)
        absent = [w for w in chunks if w not in expected]
        if absent:
            raise AggregationError(f"window set {name!r} names positions {absent} beyond the declared window geometry "
                                   f"{geometry} (positions {list(expected)}).")
        resolved[name] = tuple(chunks)
    return resolved


def validate_coverage(coll: CloudCollection, window_sets: dict) -> tuple:
    """Resolve every requested set against the DECLARED geometry and check, per condition, that every included
    cell carries every position a set requests. ``all`` is the declared acquisition period (every position
    the geometry admits), so a position missing from every cell is refused rather than silently dropped; a
    deliberate subset must be named explicitly, and positions a named subset does not request may be absent.
    Returns ``(resolved_sets, coverage)``; raises :class:`AggregationError` naming the set and the windows."""
    geometry = coll.contract["window_geometry"]
    observed = sorted(set(int(w) for w in coll.chunk))
    expected = expected_positions(geometry, observed=observed)
    beyond = sorted(set(w for w in observed if w not in expected))
    if beyond:
        raise AggregationError(f"the clouds carry positions {beyond} beyond the declared window geometry {geometry} "
                               f"(positions {list(expected)}); the identities do not belong to this geometry.")
    resolved = validate_window_sets_against_geometry(window_sets, geometry, observed=observed)
    coverage = {}
    for ki, kind in enumerate(coll.kinds):
        m = coll.kind_index == ki
        if not np.any(m):
            continue
        cells = sorted(set(int(c) for c in coll.cell[m]))
        per_cell = {c: set(int(w) for w in coll.chunk[m & (coll.cell == c)]) for c in cells}
        for name, chunks in resolved.items():
            missing = [(c, w) for c in cells for w in chunks if w not in per_cell[c]]
            if missing:
                raise AggregationError(
                    f"{kind}: window set {name!r} = positions {list(chunks)}"
                    f"{' (every position of the declared geometry)' if window_sets.get(name) is None else ''} requires every "
                    f"included cell to carry them; missing (cell, position) {missing[:10]}{' ...' if len(missing) > 10 else ''}. "
                    f"A deliberate subset of positions must be named explicitly (NAME=a-b or NAME=a+b+c); nothing is dropped silently.")
        present = sorted(set(w for ws in per_cell.values() for w in ws))
        coverage[kind] = {"cells": cells, "chunks": present, "n_windows": int(m.sum()), "n_draws_per_window": coll.n_draws,
                          "declared_positions": list(expected)}
    if not coverage:
        raise AggregationError("no windows to aggregate.")
    return resolved, coverage


def verify_identities_in_product(coll: CloudCollection, arrays) -> list:
    """Every (kind, cell, chunk) of the supplied clouds must be a window of the source Experiment product.
    Returns the identities that are not (empty = all belong)."""
    kinds = [str(k) for k in np.asarray(arrays["kinds"]).ravel()]
    have = set(zip(np.asarray(arrays["kind_index"]).astype(int).tolist(), np.asarray(arrays["cell"]).astype(int).tolist(),
                   np.asarray(arrays["chunk"]).astype(int).tolist()))
    out = []
    for ki, c, w in zip(coll.kind_index.tolist(), coll.cell.tolist(), coll.chunk.tolist()):
        kind = coll.kinds[int(ki)]
        pk = kinds.index(kind) if kind in kinds else -1
        if (pk, int(c), int(w)) not in have:
            out.append((kind, int(c), int(w)))
    return out


def _flat_pool(coll, rows) -> np.ndarray:
    return coll.cloud[rows].reshape(-1, coll.cloud.shape[-1]).astype(np.float64, copy=False)


def _quantiles(flat, levels) -> np.ndarray:
    return np.quantile(flat, list(levels), axis=0).T


def _outside(flat, low, high):
    out = (flat < low) | (flat > high)
    return float(np.mean(np.any(out, axis=1))), out.mean(axis=0)


def aggregate(coll: CloudCollection, table, *, window_sets, levels=schema.CANONICAL_QUANTILE_LEVELS,
              exact_capacity=POOLED_EXACT_CAPACITY, reuse_window_sgm=True) -> AggregationResult:
    """The four constructions for every requested window set, per condition, from the retained clouds."""
    keys = [e["KEY"] for e in table]
    if keys != coll.keys:
        raise AggregationError(f"parameter table keys {keys} != the collection's {coll.keys} (exact order required).")
    scale = np.asarray(coll.contract["sgm_scale"], float)
    widths = np.asarray([float(e["PRIOR_RANGE"][1]) - float(e["PRIOR_RANGE"][0]) for e in table], float)
    if not np.allclose(scale, widths):
        raise AggregationError(f"the collection's SGM scale {scale.tolist()} is not this table's prior widths {widths.tolist()}.")
    low = np.asarray([float(e["PRIOR_RANGE"][0]) for e in table]); high = np.asarray([float(e["PRIOR_RANGE"][1]) for e in table])
    if not np.isfinite(coll.cloud).all():
        raise AggregationError("the retained clouds carry non-finite draws.")
    resolved, coverage = validate_coverage(coll, window_sets)
    order = _canonical_order(coll)
    groups, timing, equivalences = [], {}, []
    D = len(keys)

    def make(construction, set_name, kind, cell, chunk, rows, *, stored=None):
        """A group over the windows ``rows`` (canonical order); ``stored`` reuses a window's exact SGM."""
        t0 = time.time()
        flat = _flat_pool(coll, rows)
        if stored is not None:
            sgm_idx_flat, method = int(stored), "window_stored"
        elif flat.shape[0] <= exact_capacity:
            _, sgm_idx_flat = exact_window_sgm(flat, scale=scale); method = "exact_medoid"
        else:
            sgm_idx_flat, method = collection_sgm(flat, scale, exact_capacity=exact_capacity)
        w_local, draw = divmod(int(sgm_idx_flat), coll.n_draws)
        w = int(rows[w_local])
        q = _quantiles(flat, levels)
        frac, by_key = _outside(flat, low, high)
        g = Group(construction=construction, window_set=set_name, kind=kind, cell=int(cell), chunk=int(chunk),
                  members=[(int(coll.cell[r]), int(coll.chunk[r])) for r in rows], n_draws=int(flat.shape[0]),
                  sgm_log10=flat[sgm_idx_flat].copy(),
                  sgm_source={"cell": int(coll.cell[w]), "chunk": int(coll.chunk[w]), "draw_index": int(draw),
                              "source_label": coll.sources[int(coll.source_index[w])]["label"],
                              "source_index": int(coll.source_index[w]), "flat_index": int(sgm_idx_flat)},
                  sgm_method=method, median_log10=q[:, list(levels).index(0.5)].copy(), quantiles_log10=q,
                  outside_box_fraction=frac, outside_box_by_key=by_key)
        timing[f"{set_name}/{construction}"] = timing.get(f"{set_name}/{construction}", 0.0) + (time.time() - t0)
        return g

    def alias(g, construction, set_name, cell, chunk):
        h = Group(**{**g.__dict__})
        h.construction, h.window_set, h.cell, h.chunk = construction, set_name, int(cell), int(chunk)
        h.equivalent_to = f"{g.construction}[{g.window_set}]"
        equivalences.append((f"{construction}[{set_name}]", h.equivalent_to, g.kind, int(cell), int(chunk)))
        return h

    for ki, kind in enumerate(coll.kinds):
        if kind not in coverage:
            continue
        cells = coverage[kind]["cells"]
        rows_of = {(int(coll.cell[r]), int(coll.chunk[r])): int(r) for r in order if coll.kind_index[r] == ki}
        for set_name, chunks in resolved.items():
            # per_cell_window: the windows' own posteriors (stored exact SGM reused when its row is known)
            pcw = {}
            for c in cells:
                for w in chunks:
                    r = rows_of[(c, w)]
                    stored = None
                    if reuse_window_sgm and coll.window_sgm_index is not None and int(coll.window_sgm_index[r]) >= 0:
                        stored = int(coll.window_sgm_index[r])
                    pcw[(c, w)] = make("per_cell_window", set_name, kind, c, w, np.asarray([r]), stored=stored)
            groups.extend(pcw[(c, w)] for c in cells for w in chunks)
            # per_cell
            pc = {}
            for c in cells:
                if len(chunks) == 1:
                    pc[c] = alias(pcw[(c, chunks[0])], "per_cell", set_name, c, -1)
                else:
                    pc[c] = make("per_cell", set_name, kind, c, -1, np.asarray([rows_of[(c, w)] for w in chunks]))
            groups.extend(pc[c] for c in cells)
            # per_window_position
            pw = {}
            for w in chunks:
                if len(cells) == 1:
                    pw[w] = alias(pcw[(cells[0], w)], "per_window_position", set_name, -1, w)
                else:
                    pw[w] = make("per_window_position", set_name, kind, -1, w, np.asarray([rows_of[(c, w)] for c in cells]))
            groups.extend(pw[w] for w in chunks)
            # global
            if len(chunks) == 1 and len(cells) == 1:
                g = alias(pcw[(cells[0], chunks[0])], "global", set_name, -1, -1)
            elif len(chunks) == 1:
                g = alias(pw[chunks[0]], "global", set_name, -1, -1)
            elif len(cells) == 1:
                g = alias(pc[cells[0]], "global", set_name, -1, -1)
            else:
                g = make("global", set_name, kind, -1, -1, np.asarray([rows_of[(c, w)] for c in cells for w in chunks]))
            groups.append(g)
    return AggregationResult(groups=groups, window_sets=resolved, coverage=coverage, contract=dict(coll.contract),
                             sources=list(coll.sources), keys=keys, levels=tuple(float(v) for v in levels), scale=scale,
                             prior_low=low, prior_high=high, exact_capacity=int(exact_capacity), timing=timing,
                             equivalences=equivalences)


def verify_members(result: AggregationResult, coll: CloudCollection) -> list:
    """Every group's SGM must be the retained draw its source identity names. Returns the offending groups."""
    bad = []
    lookup = {(int(coll.kind_index[r]), int(coll.cell[r]), int(coll.chunk[r])): int(r) for r in range(coll.n_windows)}
    for g in result.groups:
        ki = coll.kinds.index(g.kind)
        r = lookup[(ki, g.sgm_source["cell"], g.sgm_source["chunk"])]
        row = coll.cloud[r, g.sgm_source["draw_index"]].astype(float)
        if not np.array_equal(row, g.sgm_log10) or (g.sgm_source["cell"], g.sgm_source["chunk"]) not in g.members:
            bad.append(f"{g.construction}[{g.window_set}] cell {g.cell} chunk {g.chunk}")
    return bad


# =============================================================================
# Representative-distribution statistics
# =============================================================================
def weighted_quantile(values, weights, p) -> float:
    """The generalized inverse of the weighted empirical CDF: the smallest value whose cumulative weight is at
    least ``p`` (weights normalized to one). For a singleton every quantile is that value."""
    v = np.asarray(values, float).ravel(); w = np.asarray(weights, float).ravel()
    if v.size == 0 or v.size != w.size or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weighted_quantile: values and non-negative weights of equal length with positive total mass.")
    order = np.argsort(v, kind="stable")
    cw = np.cumsum(w[order]) / w.sum()
    return float(v[order][int(np.searchsorted(cw, p - 1e-12, side="left"))])


def representative_summary(vectors, weights, low, high, levels=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    """Per-coordinate weighted-CDF quantiles (estimator coordinates) and outside-box mass of a representative
    collection (``vectors`` (n, D), ``weights`` (n,) or None = equal)."""
    v = np.asarray(vectors, float); n, D = v.shape
    w = np.ones(n) / n if weights is None else np.asarray(weights, float) / float(np.sum(weights))
    q = np.array([[weighted_quantile(v[:, j], w, p) for p in levels] for j in range(D)])      # (D, L)
    outside_key = np.array([float(np.sum(w[(v[:, j] < low[j]) | (v[:, j] > high[j])])) for j in range(D)])
    outside_any = float(np.sum(w[np.any((v < low) | (v > high), axis=1)]))
    return {"n": int(n), "quantiles": q, "levels": tuple(levels), "outside_by_key": outside_key, "outside_any": outside_any}


def group_vectors(groups, statistic="sgm") -> np.ndarray:
    if statistic not in STATISTICS:
        raise ValueError(f"statistic {statistic!r} must be one of {STATISTICS}.")
    return np.asarray([(g.sgm_log10 if statistic == "sgm" else g.median_log10) for g in groups], float)


# =============================================================================
# Exports
# =============================================================================
ARCHIVE_FORMAT_VERSION = 1
ARCHIVE_NAME = "aggregation_candidates.npz"


def export_manifest(result: AggregationResult, *, product_label, condition_labels, reference=None, coverage_note="",
                    product_manifest=None, n_product_cells=None, n_product_windows=None, extra=None, target=None) -> dict:
    """``target`` binds the archive to the Experiment it came from (``project_alias``, ``condition`` = the run's
    condition, ``timing_label``, ``window_seconds``, ``n_frames``, the estimator tag); a Nuisance DLI built from
    the archive must name the same target, so a cross-condition application never becomes another condition's
    native nuisance by accident."""
    return {
        "archive_format_version": ARCHIVE_FORMAT_VERSION, "stage": "experiment_aggregation", "product_label": product_label,
        "target": None if target is None else dict(target),
        "condition_labels": list(condition_labels), "constructions": list(CONSTRUCTIONS),
        "construction_definitions": {k: {"name": v[0], "calculation": v[1], "interpretation": v[2]}
                                     for k, v in CONSTRUCTION_DEFINITIONS.items()},
        "statistics": list(STATISTICS), "default_statistic": "sgm",
        "window_sets": {k: list(v) for k, v in result.window_sets.items()},
        "coverage": result.coverage, "coverage_note": coverage_note,
        "n_product_cells": n_product_cells, "n_product_windows": n_product_windows,
        "parameter_keys": list(result.keys), "quantile_levels": list(result.levels),
        "metric": "estimator coordinates divided by the prior widths (sgm_scale)", "sgm_scale": result.scale.tolist(),
        "prior_low": result.prior_low.tolist(), "prior_high": result.prior_high.tolist(),
        "methods": {"single_window": "the window's stored exact sample medoid (reused) or the exact medoid",
                    "pooled": f"exact medoid up to {result.exact_capacity} rows; above it the collection kernel's "
                              f"Weiszfeld geometric median snapped to the nearest retained draw (weiszfeld_snap), an "
                              f"approximation to the exact medoid",
                    "quantiles": "mixture quantiles of the pooled draws in estimator coordinates (numpy linear interpolation); "
                                 "physical endpoints are the transform of the quantile",
                    "representative_quantiles": "inverse weighted empirical CDF (smallest value with cumulative weight >= p)"},
        "contract": result.contract, "sources": result.sources,
        "disclosed_source_differences": _disclosed_differences(result.sources),
        "reference": reference, "equivalences": [{"group": a, "equals": b, "kind": k, "cell": c, "chunk": w}
                                                  for a, b, k, c, w in result.equivalences],
        "timing_seconds": {k: round(v, 3) for k, v in result.timing.items()},
        "source_product_manifest": product_manifest, "written_at": schema._now(), **(extra or {}),
    }


def _disclosed_differences(sources) -> dict:
    out = {}
    for k in ("collection_label", "seed_policy", "implementation_sha256", "written_at"):
        vals = {json.dumps(s.get(k), sort_keys=True, default=str) for s in sources}
        if len(vals) > 1:
            out[k] = [s.get(k) for s in sources]
    return out


def write_aggregation_exports(result: AggregationResult, out_dir, table, *, manifest: dict, reference=None) -> dict:
    """Write the candidate archive, the two CSV exports and the manifest into ``out_dir``; return their paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    G = result.groups; D = len(result.keys); Q = len(result.levels)
    arrays = dict(
        construction=np.asarray([g.construction for g in G]), window_set=np.asarray([g.window_set for g in G]),
        kind=np.asarray([g.kind for g in G]), cell=np.asarray([g.cell for g in G], np.int64),
        chunk=np.asarray([g.chunk for g in G], np.int64), n_draws=np.asarray([g.n_draws for g in G], np.int64),
        n_windows=np.asarray([len(g.members) for g in G], np.int64),
        members_json=np.asarray([json.dumps(g.members) for g in G]),
        sgm_log10=np.asarray([g.sgm_log10 for g in G], float).reshape(len(G), D),
        sgm_source_cell=np.asarray([g.sgm_source["cell"] for g in G], np.int64),
        sgm_source_chunk=np.asarray([g.sgm_source["chunk"] for g in G], np.int64),
        sgm_source_draw=np.asarray([g.sgm_source["draw_index"] for g in G], np.int64),
        sgm_source_label=np.asarray([g.sgm_source["source_label"] for g in G]),
        sgm_method=np.asarray([g.sgm_method for g in G]), equivalent_to=np.asarray([g.equivalent_to for g in G]),
        median_log10=np.asarray([g.median_log10 for g in G], float).reshape(len(G), D),
        quantiles_log10=np.asarray([g.quantiles_log10 for g in G], float).reshape(len(G), D, Q),
        outside_box_fraction=np.asarray([g.outside_box_fraction for g in G], float),
        outside_box_by_key=np.asarray([g.outside_box_by_key for g in G], float).reshape(len(G), D),
        parameter_keys=np.asarray(result.keys), quantile_levels=np.asarray(result.levels, float),
        sgm_scale=result.scale, prior_low=result.prior_low, prior_high=result.prior_high,
        window_set_names=np.asarray(list(result.window_sets)),
        window_set_chunks_json=np.asarray([json.dumps(list(v)) for v in result.window_sets.values()]),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True, default=str)))
    if reference is not None:
        arrays["reference_log10"] = np.asarray(reference["log10"], float)
    archive = out_dir / ARCHIVE_NAME
    np.savez_compressed(archive, **arrays)
    # groups.csv: physical units per key for the SGM, the median and the 90 % mixture interval
    groups_csv = out_dir / "aggregation_groups.csv"
    with groups_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "window_set", "construction", "cell", "chunk", "n_windows", "n_draws", "sgm_method", "equivalent_to",
                    "sgm_source_cell", "sgm_source_chunk", "sgm_source_draw", "sgm_source_label", "outside_box_fraction"]
                   + [f"sgm_{k}" for k in result.keys] + [f"median_{k}" for k in result.keys]
                   + [f"q05_{k}" for k in result.keys] + [f"q95_{k}" for k in result.keys])
        for g in G:
            sgm_p = to_physical(g.sgm_log10, table); med_p = to_physical(g.median_log10, table)
            q05 = to_physical(g.quantiles_log10[:, 0], table); q95 = to_physical(g.quantiles_log10[:, -1], table)
            w.writerow([g.kind, g.window_set, g.construction, g.cell, g.chunk, len(g.members), g.n_draws, g.sgm_method, g.equivalent_to,
                        g.sgm_source["cell"], g.sgm_source["chunk"], g.sgm_source["draw_index"], g.sgm_source["source_label"],
                        f"{g.outside_box_fraction:.6f}"] + [repr(float(v)) for v in sgm_p] + [repr(float(v)) for v in med_p]
                       + [repr(float(v)) for v in q05] + [repr(float(v)) for v in q95])
    comparison_csv = out_dir / "aggregation_comparison.csv"
    with comparison_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "window_set", "construction", "statistic", "parameter", "n_representatives", "median_physical",
                    "q25_physical", "q75_physical", "q05_physical", "q95_physical", "outside_box_mass",
                    "reference_physical", "median_vs_reference_percent", "median_vs_reference_dex"])
        for row in comparison_rows(result, table, reference=reference):
            w.writerow(row)
    manifest_path = out_dir / "aggregation_manifest.json"
    manifest_path.write_text(json.dumps({**manifest, "archive_sha256": _sha256(archive)}, indent=1, default=str))
    return {"archive": archive, "groups_csv": groups_csv, "comparison_csv": comparison_csv, "manifest": manifest_path}


def comparison_rows(result: AggregationResult, table, *, reference=None) -> list:
    """One row per (kind, window set, construction, statistic, parameter): weighted-CDF quantiles over the
    representatives in physical units, outside-box mass, and the difference of the representatives' median
    from the reference (percent and dex for a log row; the difference itself for a linear row)."""
    rows = []
    log_rows = [is_log_row(e) for e in table]
    for kind in result.coverage:
        for set_name in result.window_sets:
            for construction in CONSTRUCTIONS:
                groups = result.select(construction=construction, window_set=set_name, kind=kind)
                if not groups:
                    continue
                for statistic in STATISTICS:
                    vecs = group_vectors(groups, statistic)
                    summ = representative_summary(vecs, None, result.prior_low, result.prior_high)
                    q_phys = np.stack([to_physical(summ["quantiles"][:, i], table) for i in range(5)], axis=1)
                    for j, key in enumerate(result.keys):
                        ref = None if reference is None else float(reference["physical"][j])
                        med = float(q_phys[j, 2])
                        if ref is None:
                            pct, dex = "", ""
                        elif log_rows[j]:
                            pct, dex = f"{(med / ref - 1) * 100:+.4f}", f"{np.log10(med / ref):+.6f}"
                        else:
                            pct, dex = f"{(med - ref):+.6f}", ""
                        rows.append([kind, set_name, construction, statistic, key, summ["n"], repr(med), repr(float(q_phys[j, 1])),
                                     repr(float(q_phys[j, 3])), repr(float(q_phys[j, 0])), repr(float(q_phys[j, 4])),
                                     f"{summ['outside_by_key'][j]:.6f}", "" if ref is None else repr(ref), pct, dex])
    return rows


def load_candidate_archive(path) -> dict:
    """Read ``aggregation_candidates.npz`` back: arrays plus the decoded manifest; verifies the row counts."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    manifest = json.loads(str(arrays.pop("manifest_json")))
    if int(manifest.get("archive_format_version", 0)) != ARCHIVE_FORMAT_VERSION:
        raise AggregationError(f"{path}: archive format {manifest.get('archive_format_version')!r} != {ARCHIVE_FORMAT_VERSION}.")
    n = arrays["construction"].shape[0]
    for k in ("window_set", "kind", "cell", "chunk", "sgm_log10", "median_log10", "quantiles_log10", "sgm_method"):
        if arrays[k].shape[0] != n:
            raise AggregationError(f"{path}: {k} has {arrays[k].shape[0]} rows; {n} groups.")
    keys = [str(k) for k in arrays["parameter_keys"]]
    if keys != list(manifest["parameter_keys"]):
        raise AggregationError(f"{path}: parameter_keys {keys} != manifest {manifest['parameter_keys']}.")
    return {"arrays": arrays, "manifest": manifest, "path": str(path), "sha256": _sha256(path), "keys": keys}


def archive_summary_rows(archive, table, *, statistic="sgm", levels=(0.05, 0.25, 0.5, 0.75, 0.95)) -> list:
    """One row per (kind, window set, construction) of an archive: the representative-collection summary by the
    SAME convention the Experiment report and the comparison export use (:func:`representative_summary`, the
    inverse weighted empirical CDF at equal weights), in physical units: ``{kind, window_set, construction,
    n, quantiles_physical (D, L), outside_by_key}``. The selection console reads this, never ``np.median``."""
    a, m = archive["arrays"], archive["manifest"]
    low = np.asarray(a["prior_low"], float); high = np.asarray(a["prior_high"], float)
    rows = []
    for kind in m.get("condition_labels", sorted(set(a["kind"].tolist()))):
        for set_name in m.get("window_sets", {}) or sorted(set(a["window_set"].tolist())):
            for construction in CONSTRUCTIONS:
                msk = (a["kind"] == kind) & (a["window_set"] == set_name) & (a["construction"] == construction)
                if not msk.any():
                    continue
                vecs = (a["sgm_log10"] if statistic == "sgm" else a["median_log10"])[msk].astype(float)
                summ = representative_summary(vecs, None, low, high, levels=levels)
                qp = np.stack([to_physical(summ["quantiles"][:, i], table) for i in range(len(levels))], axis=1)
                rows.append({"kind": str(kind), "window_set": str(set_name), "construction": construction, "statistic": statistic,
                             "n": summ["n"], "quantiles_physical": qp, "levels": tuple(levels), "outside_by_key": summ["outside_by_key"]})
    return rows


def archive_component(archive, *, kind, window_set, construction, statistic="sgm") -> dict:
    """The representatives of one (kind, window set, construction) in an archive: vectors, identities, memberships."""
    a = archive["arrays"]
    m = (a["kind"] == kind) & (a["window_set"] == window_set) & (a["construction"] == construction)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        raise AggregationError(f"archive holds no component kind={kind!r} window_set={window_set!r} construction={construction!r}; "
                               f"available: " + ", ".join(sorted({f'{k}/{s}/{c}' for k, s, c in zip(a['kind'], a['window_set'], a['construction'])})))
    if statistic not in STATISTICS:
        raise AggregationError(f"statistic {statistic!r} must be one of {STATISTICS}.")
    vecs = (a["sgm_log10"] if statistic == "sgm" else a["median_log10"])[idx].astype(float)
    ident = [{"cell": int(a["cell"][i]), "chunk": int(a["chunk"][i]), "n_windows": int(a["n_windows"][i]),
              "sgm_source": {"cell": int(a["sgm_source_cell"][i]), "chunk": int(a["sgm_source_chunk"][i]),
                             "draw_index": int(a["sgm_source_draw"][i]), "source_label": str(a["sgm_source_label"][i])},
              "sgm_method": str(a["sgm_method"][i]), "equivalent_to": str(a["equivalent_to"][i]),
              "is_sampled_rows": statistic == "sgm"} for i in idx]
    return {"kind": kind, "window_set": window_set, "construction": construction, "statistic": statistic,
            "vectors": vecs, "identities": ident, "group_indices": idx.tolist()}


# =============================================================================
# The Experiment report section (one implementation for a live run and a CPU regeneration)
# =============================================================================
SECTION_TITLE = "Experimental aggregation and nuisance candidates"


@dataclass
class AggregationRequest:
    """What the Experiment writer aggregates: the requested window sets, the retained clouds to use (``None``
    = the product's own ``posterior_samples_cloud``), the displayed reference vector (``None`` = none; the
    detector's SEL artifact when present) and the coverage statement a regeneration over a subset must make."""
    window_sets: dict                      # name -> tuple | None (parsed)
    enabled: bool = True
    clouds: CloudCollection = None
    reference: dict = None                 # name, log10, physical, identity
    coverage_note: str = ""
    exact_capacity: int = POOLED_EXACT_CAPACITY
    window_seconds: float = None           # the model window, for the temporal axis
    product_label: str = ""
    source_clouds_note: str = ""
    target: dict = None                    # project_alias, condition, timing_label, window_seconds, n_frames, estimator_tag


def _fmt(v):
    v = float(v)
    if v == 0:
        return "0"
    return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.4e}"


def report_aggregation(reporter, request: AggregationRequest, arrays, manifest, table, *, out_dir, condition_display,
                       kind_notes="") -> dict:
    """Write the indexed section into the Experiment report (tables + figures through ``reporter``) and the
    supporting exports into ``out_dir``. Returns the aggregation manifest, or ``None`` when the section states
    that the calculations are unavailable (aggregation disabled, or no retained clouds)."""
    from .visualization_aggregation import (figure_joint_corner, figure_marginal_cdfs, figure_temporal,
                                            figure_window_selection)
    keys = [e["KEY"] for e in table]
    kinds = [str(k) for k in arrays["kinds"]]
    cell_arr = np.asarray(arrays["cell"]).astype(int); kind_index = np.asarray(arrays["kind_index"]).astype(int)
    n_product_cells = {k: int(np.unique(cell_arr[kind_index == i]).size) for i, k in enumerate(kinds)}
    n_product_windows = {k: int((kind_index == i).sum()) for i, k in enumerate(kinds)}
    if request is None or not request.enabled:
        reporter.prose(SECTION_TITLE, "Not computed: aggregation was disabled for this run (`--no-aggregation`, the legacy "
                       "report). The per-window estimates above are complete; no SGM of SGMs or other substitute is "
                       "reported. Run the stage with aggregation enabled, or regenerate this report on the CPU from "
                       "retained clouds (`--regenerate-report`).")
        return None
    try:
        coll = request.clouds if request.clouds is not None else cloud_from_product(arrays, manifest, label=request.product_label or "experiment product")
    except AggregationError as exc:
        reporter.prose(SECTION_TITLE, f"Not computed: {exc} The per-window estimates above are complete; no substitute is reported.")
        return None
    if coll.kinds != kinds:
        raise AggregationError(f"retained clouds carry condition labels {coll.kinds}; the product carries {kinds}.")
    product_contract = {k: manifest[k] for k in CONTRACT_FIELDS}
    if manifest.get("input_conversion") is not None:
        product_contract["input_conversion"] = manifest["input_conversion"]
    diff = contract_differences(coll.contract, product_contract)
    if diff:
        raise AggregationError("the retained clouds were drawn under another computation contract than this product: " + "; ".join(diff))
    foreign = verify_identities_in_product(coll, arrays)
    if foreign:
        raise AggregationError(f"{len(foreign)} supplied window identities (condition, cell, chunk) are not windows of this "
                               f"Experiment product: {foreign[:10]}{' ...' if len(foreign) > 10 else ''}.")
    t0 = time.time()
    result = aggregate(coll, table, window_sets=request.window_sets, levels=tuple(manifest["quantile_levels"]),
                       exact_capacity=request.exact_capacity)
    bad_members = verify_members(result, coll)
    elapsed = time.time() - t0
    ref = request.reference
    covered_cells = {k: result.coverage[k]["cells"] for k in result.coverage}
    coverage_lines = []
    for k in result.coverage:
        cov = result.coverage[k]
        coverage_lines.append(f"{condition_display(k)}: {len(cov['cells'])} of {n_product_cells[k]} cells of this product "
                              f"(cells {cov['cells']}), {cov['n_windows']} of {n_product_windows[k]} windows, positions "
                              f"{cov['chunks']}, {cov['n_draws_per_window']} retained draws per window.")
    agg_manifest = export_manifest(result, product_label=request.product_label, condition_labels=kinds,
                                   reference=None if ref is None else ref.get("identity"),
                                   coverage_note=request.coverage_note, product_manifest={k: manifest.get(k) for k in
                                   ("run_identity", "checkpoint_sha256", "written_at", "n_observations", "code")},
                                   n_product_cells=n_product_cells, n_product_windows=n_product_windows, target=request.target,
                                   extra={"reference_vector": None if ref is None else {"name": ref["name"], "log10": [float(v) for v in ref["log10"]],
                                                                                       "physical": [float(v) for v in ref["physical"]]},
                                          "source_clouds_note": request.source_clouds_note, "aggregation_seconds": round(elapsed, 2)})
    paths = write_aggregation_exports(result, out_dir, table, manifest=agg_manifest, reference=ref)
    # ---- the section ----------------------------------------------------------------------------------------
    sets_desc = "; ".join(format_window_set(n, c) for n, c in result.window_sets.items())
    src_desc = "; ".join(f"`{Path(s['path']).name if s.get('path') else s['label']}`"
                         + (f" (collection label {s['collection_label']})" if s.get("collection_label") else "")
                         + (f", sha256 {s['sha256'][:12]}…" if s.get("sha256") else "") for s in result.sources)
    disclosed = _disclosed_differences(result.sources)
    geom = coll.contract["window_geometry"]
    declared = next(iter(result.coverage.values()))["declared_positions"]
    conv_status = coll.contract.get("input_conversion_status") or "unavailable"
    text = [
        "Collections of representative parameter vectors obtained by grouping the retained window posteriors "
        "(the summary draws of the per-window estimates above) in explicitly defined ways: the four constructions "
        "of the canonical table below, each for every requested window set. The INPUTS are mixtures of window "
        "posteriors (the pooled retained draws of a group, equal weight per window); the OUTPUTS exported below are "
        "collections of point representatives, one complete vector per group, which do not themselves retain the "
        "complete input mixtures (the mixture quantiles stored beside each representative are summaries of the pooled "
        "draws, not the draws). These are descriptive groupings, not a posterior conditioned on a whole recording and not "
        "products of posteriors; the aggregation performs no neural inference on an input longer than one window and "
        "optimizes no MAP. The collections are candidates for the Nuisance DLI selection (the analysis "
        "`SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Nuisance_DLI.py` reads the archive below); no construction, window "
        "set, statistic or mixture is selected here.", "",
        "**Coverage.** " + " ".join(coverage_lines) + (f" {request.coverage_note}" if request.coverage_note else ""), "",
        f"**Window sets.** {sets_desc}. `all` is every position of the declared window geometry ({geom['n_frames']}-frame windows, "
        f"step {geom['step_frames']} frames, recording span {geom.get('span_frames')} frames: positions {declared}), and every "
        f"included cell must carry every position a set requests; `opening` is the first window. A deliberate subset is named "
        f"explicitly; nothing is dropped silently.", "",
        f"**Retained clouds.** {src_desc}." + (f" {request.source_clouds_note}" if request.source_clouds_note else "")
        + f" Every supplied identity (condition, cell, chunk) is a window of this product. Input conversion (preprocessing): "
          f"{conv_status}" + (f"; recorded as `{coll.contract.get('input_conversion')}`" if coll.contract.get("input_conversion") else "") + "."
        + (" Disclosed differences between the sources (not part of the computation contract): "
           + ", ".join(f"{k} = {v}" for k, v in disclosed.items()) + "." if disclosed else ""), "",
        f"**Methods.** Distances in estimator coordinates divided by the prior widths {result.scale.tolist()} (the per-window "
        f"SGM convention). A single-window group reuses the window's stored exact sample medoid; a pooled group above "
        f"{result.exact_capacity} rows takes the collection kernel's Weiszfeld geometric median snapped to the nearest retained "
        f"draw (`weiszfeld_snap`, an approximation to the exact medoid, recorded per group). Pooled marginal quantiles are "
        f"mixture quantiles of the pooled draws in estimator coordinates (the stage's interpolation), endpoints transformed to "
        f"physical units; representative-distribution quantiles (median, IQR, central 90 % range over a collection's "
        f"representatives) use the inverse weighted empirical CDF and are empirical dispersion, not credible or confidence "
        f"intervals. Unrestricted draws stay unrestricted: outside-prior mass is counted, never clipped. SGM is the default "
        f"candidate statistic; the marginal-median vector of each group is stored beside it, labeled as a coordinate-wise "
        f"composite. Aggregation took {elapsed:.1f} s on the CPU.", "",
        ("**Reference.** " + ref["name"] + ": " + ", ".join(f"{k} {_fmt(v)}" for k, v in zip(keys, ref["physical"]))
         + " (physical), loaded from " + str((ref.get("identity") or {}).get("artifact", "its artifact")) + ". Its current method, "
         "per the artifact's own decision record (the per-parameter table below): the median over the per-window posterior "
         "medians of the adopted Experiment product, transformed to physical units, followed by the documented "
         "parameter-specific selections. A historical fixed scenario for comparison, neither reconstructed from the draws "
         "nor mixed into any collection." if ref is not None else
         "**Reference.** None available for this run (no fixed reference artifact of this condition); differences from a reference are not reported."), "",
        "**Supporting files (this directory).** [aggregation_candidates.npz](aggregation_candidates.npz) (every group's SGM and "
        "median vectors, mixture quantiles, identities, methods, contract, sources and the target binding), "
        "[aggregation_groups.csv](aggregation_groups.csv), [aggregation_comparison.csv](aggregation_comparison.csv), "
        "[aggregation_manifest.json](aggregation_manifest.json). Figures: "
        + ", ".join(f"[aggregation_{kind}_{s}_{f}.png](figures/aggregation_{kind}_{s}_{f}.png)" for kind in result.coverage
                    for s in result.window_sets for f in ("marginals", "joint"))
        + "".join(f", [aggregation_{kind}_{s}_temporal.png](figures/aggregation_{kind}_{s}_temporal.png)" for kind in result.coverage
                  for s, c in result.window_sets.items() if len(c) > 1)
        + ", " + ", ".join(f"[aggregation_{kind}_window_selection.png](figures/aggregation_{kind}_window_selection.png)" for kind in result.coverage)
        + " (captions at the end of the report).", "",
        "Reading rules: an acknowledged training-bound extrapolation stays visible in the outside-prior columns (the InlB "
        "PSF width lies above its prior ceiling); pairwise correlations in the joint figures are descriptive, not selection "
        "constraints; nothing here declares a winner, and membership of a reference inside intervals computed from the same "
        "draws is a compatibility diagnostic, not a validation." + kind_notes,
    ]
    reporter.prose(SECTION_TITLE, "\n".join(text))
    reporter.check("aggregation_coverage_complete", True,
                   "; ".join(f"{k}: {len(v['cells'])} cells x {len(v['chunks'])} positions" for k, v in result.coverage.items()),
                   note="every included cell carries every requested window position; conditions are never pooled together.")
    reporter.check("aggregation_equal_window_weights", True, f"{coll.n_draws} retained draws in every window, so concatenation "
                   f"realizes equal window and cell weights", note="the equal weights of the mixtures are realized by equal draw counts.")
    reporter.check("aggregation_contract_matches_product", not diff, f"checkpoint {coll.contract['checkpoint_sha256'][:12]}…, "
                   f"{len(keys)} keys, scale {result.scale.tolist()}, {coll.contract['pool_mode']}, {coll.n_draws} draws, geometry "
                   f"{coll.contract['window_geometry']}; input conversion {conv_status}",
                   note="the retained clouds and this product share one computation contract; the input conversion is compared "
                        "between the supplied collections as well, and 'unavailable' means no source records it (not verified).")
    reporter.check("aggregation_identities_in_product", not foreign, f"{coll.n_windows} supplied windows, all windows of this product",
                   note="every supplied (condition, cell, chunk) identity belongs to the source Experiment product.")
    reporter.check("aggregation_sgm_members", not bad_members,
                   f"{len(result.groups)} groups; every SGM is the retained draw its source identity names" if not bad_members
                   else f"{len(bad_members)} group(s) fail: {bad_members[:5]}", fatal=False,
                   note="each group's SGM equals the retained draw (cell, chunk, draw index) it records; a median vector is never claimed as a draw.")
    # canonical options table
    rows = []
    for c in CONSTRUCTIONS:
        name, calc, interp = CONSTRUCTION_DEFINITIONS[c]
        counts = "; ".join(f"{s}: " + ", ".join(f"{len(result.select(construction=c, window_set=s, kind=k))} ({condition_display(k)})"
                                                 for k in result.coverage) for s in result.window_sets)
        rows.append([f"`{c}`", name, calc, counts, interp])
    if ref is not None:
        rows.insert(0, [f"`reference` ({ref['name']})", "Historical fixed scenario", "median over the per-window posterior medians of the "
                        "adopted Experiment product (physical units), followed by the documented parameter-specific selections "
                        "(per-parameter record below); not an SGM of draws", "1", "one fixed scenario; comparison only"])
    reporter.table(f"{SECTION_TITLE}: constructions", ["ID", "construction", "calculation", "representatives per window set", "interpretation"],
                   rows, note="the canonical options; counts are per condition. An SGM of window SGMs is not an active construction; no aggregate MAP exists.")
    sel = ((ref or {}).get("identity") or {}).get("selection") if ref is not None else None
    if sel and isinstance(sel.get("sources"), dict):
        reporter.table(f"{SECTION_TITLE}: the reference {ref['name']} per parameter (its decision record)",
                       ["parameter", "value (physical)", "in prior box", "source (current method)", "limitation"],
                       [[k, _fmt(sel.get("values_physical", {}).get(k, v)), "no (acknowledged)" if k in (sel.get("outside_prior") or []) else "yes",
                         str(sel["sources"].get(k, {}).get("source", "")), str(sel["sources"].get(k, {}).get("limitation", ""))]
                        for k, v in zip(keys, ref["physical"])],
                       note="copied from the reference artifact's selection record; the reference is a fixed scenario and its "
                            "limitations travel with it.")
    eq_rows = [[g, e, condition_display(k), "" if c < 0 else str(c), "" if w < 0 else str(w)] for g, e, k, c, w in result.equivalences]
    if eq_rows:
        reporter.table(f"{SECTION_TITLE}: equivalent groups", ["group", "equals", "condition", "cell", "position"], eq_rows[:60],
                       note="a single-window set makes per_cell equal per_cell_window and per_window_position equal global; a single cell "
                            "makes per_window_position equal per_cell_window and global equal per_cell; the computed group is reused."
                            + (f" {len(eq_rows) - 60} further rows in aggregation_groups.csv." if len(eq_rows) > 60 else ""))
    log_rows = [is_log_row(e) for e in table]
    for kind in result.coverage:
        kname = condition_display(kind)
        for set_name, chunks in result.window_sets.items():
            rows = []
            for c in CONSTRUCTIONS:
                groups = result.select(construction=c, window_set=set_name, kind=kind)
                summ = representative_summary(group_vectors(groups, "sgm"), None, result.prior_low, result.prior_high)
                qp = np.stack([to_physical(summ["quantiles"][:, i], table) for i in range(5)], axis=1)
                for j, key in enumerate(keys):
                    if ref is None:
                        vs = "-"
                    elif log_rows[j]:
                        vs = f"{(qp[j, 2] / ref['physical'][j] - 1) * 100:+.1f} % ({np.log10(qp[j, 2] / ref['physical'][j]):+.3f} dex)"
                    else:
                        vs = f"{qp[j, 2] - ref['physical'][j]:+.4f}"
                    rows.append([c, key, str(summ["n"]), _fmt(qp[j, 2]), f"{_fmt(qp[j, 1])} – {_fmt(qp[j, 3])}",
                                 f"{_fmt(qp[j, 0])} – {_fmt(qp[j, 4])}", vs, f"{100 * summ['outside_by_key'][j]:.1f} %"])
            reporter.table(f"{SECTION_TITLE} [{kname}, window set {set_name}]: SGM representatives (physical units)",
                           ["construction", "parameter", "n", "median", "IQR (q25 – q75)", "central 90 % range", "median vs reference", "outside-prior mass"],
                           rows, note=f"window set {format_window_set(set_name, chunks)}; quantiles over the collection's representatives by the "
                                      f"inverse weighted empirical CDF (empirical dispersion across representatives, not posterior credible "
                                      f"intervals); for one representative every column is that vector. 'outside-prior mass' is the share of "
                                      f"representatives beyond the row's prior box.{kind_notes}")
            g_glob = result.select(construction="global", window_set=set_name, kind=kind)[0]
            rows = []
            for j, key in enumerate(keys):
                gs = to_physical(g_glob.sgm_log10, table)[j]; gm = to_physical(g_glob.median_log10, table)[j]
                q = to_physical(g_glob.quantiles_log10[:, 0], table)[j], to_physical(g_glob.quantiles_log10[:, -1], table)[j]
                rv = "-" if ref is None else _fmt(ref["physical"][j])
                d = ("-" if ref is None else (f"{(gs / ref['physical'][j] - 1) * 100:+.1f} %" if log_rows[j] else f"{gs - ref['physical'][j]:+.4f}"))
                rows.append([key, _fmt(gs), _fmt(gm), f"{_fmt(q[0])} – {_fmt(q[1])}", rv, d, f"{100 * g_glob.outside_box_by_key[j]:.1f} %"])
            reporter.table(f"{SECTION_TITLE} [{kname}, window set {set_name}]: global representative and reference (physical units)",
                           ["parameter", "global SGM (a draw)", "global median vector (composite)", "pooled 90 % interval", "reference",
                            "global SGM vs reference", "pooled draws outside prior"], rows,
                           note=f"the global SGM is retained draw cell {g_glob.sgm_source['cell']}, chunk {g_glob.sgm_source['chunk']}, index "
                                f"{g_glob.sgm_source['draw_index']} ({g_glob.sgm_method}; {g_glob.n_draws} pooled draws); the pooled 90 % "
                                f"interval is a mixture quantile of the pooled draws, a different quantity from the dispersion of representatives.{kind_notes}")
        # source-uncertainty summary (the first window set that spans several positions, else the first)
        set_name = next((s for s, c in result.window_sets.items() if len(c) > 1), next(iter(result.window_sets)))
        pcw = result.select(construction="per_cell_window", window_set=set_name, kind=kind)
        pc = result.select(construction="per_cell", window_set=set_name, kind=kind)
        g_glob = result.select(construction="global", window_set=set_name, kind=kind)[0]
        rows = []
        for j, key in enumerate(keys):
            w90 = np.median([to_physical(g.quantiles_log10[:, -1], table)[j] - to_physical(g.quantiles_log10[:, 0], table)[j] for g in pcw])
            pc90 = np.median([to_physical(g.quantiles_log10[:, -1], table)[j] - to_physical(g.quantiles_log10[:, 0], table)[j] for g in pc])
            gl = to_physical(g_glob.quantiles_log10[:, 0], table)[j], to_physical(g_glob.quantiles_log10[:, -1], table)[j]
            disp_pc = representative_summary(group_vectors(pc, "sgm"), None, result.prior_low, result.prior_high)["quantiles"][j]
            disp_pcw = representative_summary(group_vectors(pcw, "sgm"), None, result.prior_low, result.prior_high)["quantiles"][j]
            dp = to_physical(disp_pc[:, None], [table[j]])[:, 0]; dw = to_physical(disp_pcw[:, None], [table[j]])[:, 0]
            rows.append([key, _fmt(w90), _fmt(pc90), f"{_fmt(gl[0])} – {_fmt(gl[1])}", f"{_fmt(dp[0])} – {_fmt(dp[4])} (n={len(pc)})",
                         f"{_fmt(dw[0])} – {_fmt(dw[4])} (n={len(pcw)})"])
        reporter.table(f"{SECTION_TITLE} [{kname}, window set {set_name}]: source uncertainty against representative dispersion (physical units)",
                       ["parameter", "within-window 90 % width (median over windows)", "per-cell pooled 90 % width (median over cells)",
                        "global pooled 90 % interval", "central 90 % range of per_cell SGMs", "central 90 % range of per_cell_window SGMs"], rows,
                       note="the first three columns are intervals of input or pooled posteriors (retained posterior uncertainty; mixture quantiles for "
                            "the pooled ones); the last two are the dispersion of selected representatives (compression). They are different "
                            f"quantities and are never read as calibrated uncertainty.{kind_notes}")
        rows = []
        for set_name in result.window_sets:
            pc = result.select(construction="per_cell", window_set=set_name, kind=kind)
            d = np.abs(group_vectors(pc, "sgm") - group_vectors(pc, "median"))
            rows.append([set_name] + [f"{np.median(d[:, j]):.3f} / {d[:, j].max():.3f}" for j in range(len(keys))])
        reporter.table(f"{SECTION_TITLE} [{kname}]: SGM versus marginal-median vector of the per_cell groups (median / max |difference|, stored coordinates)",
                       ["window set"] + keys, rows, note="the marginal-median vector is a coordinate-wise composite whose coordinates need not have "
                       "occurred together; it is stored beside the SGM and must be named explicitly by a selection that uses it.")
        # figures
        if reporter.dump:
            for set_name, chunks in result.window_sets.items():
                series = []
                for c in ("per_cell", "per_window_position", "per_cell_window", "global"):
                    gs = result.select(construction=c, window_set=set_name, kind=kind)
                    series.append({"name": c, "role": c, "vectors": group_vectors(gs, "sgm")})
                if ref is not None:
                    series.append({"name": ref["name"], "role": "reference", "vectors": np.asarray(ref["log10"], float)[None, :]})
                reporter.save_figure(f"aggregation_{kind}_{set_name}_marginals",
                                     figure_marginal_cdfs(series, table, title=f"{kname}, window set '{set_name}': marginal CDFs of the SGM representatives",
                                                          subtitle="each collection normalized to unit mass; a one-member collection is a reference line; dotted red = prior box"),
                                     caption=f"{kname}, window set {format_window_set(set_name, chunks)}: normalized empirical CDFs of the SGM representatives "
                                             f"of every construction, physical units; the global representative and the reference are vertical lines. Dispersion "
                                             f"across representatives, not posterior uncertainty.{kind_notes}")
                reporter.save_figure(f"aggregation_{kind}_{set_name}_joint",
                                     figure_joint_corner(series, table, title=f"{kname}, window set '{set_name}': joint projections of the SGM representatives"),
                                     caption=f"{kname}, window set {format_window_set(set_name, chunks)}: pairwise projections of the representatives "
                                             f"(co-occurring coordinates of retained draws); correlations are descriptive, no relation is imposed.{kind_notes}")
                if len(chunks) > 1:
                    cells = result.coverage[kind]["cells"]
                    pcw = {c: np.asarray([g.sgm_log10 for g in result.select(construction="per_cell_window", window_set=set_name, kind=kind) if g.cell == c]) for c in cells}
                    pw = np.asarray([g.sgm_log10 for g in result.select(construction="per_window_position", window_set=set_name, kind=kind)])
                    bands = np.median(np.asarray([[g.quantiles_log10 for g in result.select(construction="per_cell_window", window_set=set_name, kind=kind) if g.cell == c]
                                                  for c in cells]), axis=0)
                    geom = coll.contract["window_geometry"]
                    step_s = (float(request.window_seconds) * geom["step_frames"] / geom["n_frames"]) if request.window_seconds else 1.0
                    times = np.asarray(chunks, float) * step_s
                    reporter.save_figure(f"aggregation_{kind}_{set_name}_temporal",
                                         figure_temporal(pcw, pw, result.select(construction="global", window_set=set_name, kind=kind)[0].sgm_log10,
                                                         None if ref is None else {"name": ref["name"], "log10": ref["log10"]}, table, times=times,
                                                         window_bands=bands, title=f"{kname}, window set '{set_name}': representatives over acquisition time"),
                                         caption=f"{kname}, window set {format_window_set(set_name, chunks)}: per_cell_window SGMs (thin lines, one per cell: "
                                                 f"across-cell dispersion), per_window_position SGMs (bold), the global SGM and the reference (horizontal lines), "
                                                 f"over the within-window 50 % and 90 % intervals of the draws (median across cells: retained posterior "
                                                 f"uncertainty). x = window start in seconds{' (step ' + f'{step_s:g}' + ' s)' if request.window_seconds else ''}.{kind_notes}")
            per_cell_by_set = {s: {g.cell: g.sgm_log10 for g in result.select(construction="per_cell", window_set=s, kind=kind)} for s in result.window_sets}
            global_by_set = {s: result.select(construction="global", window_set=s, kind=kind)[0].sgm_log10 for s in result.window_sets}
            reporter.save_figure(f"aggregation_{kind}_window_selection",
                                 figure_window_selection(per_cell_by_set, table, reference=None if ref is None else {"name": ref["name"], "log10": ref["log10"]},
                                                         global_by_set=global_by_set, title=f"{kname}: per-cell representatives under each window set"),
                                 caption=f"{kname}: the per_cell SGM of every cell under each requested window set ({sets_desc}), cell identity preserved "
                                         f"(a thin line joins the same cell), with the global SGM of each set and the reference.{kind_notes}")
    agg_manifest["exports"] = {k: str(v) for k, v in paths.items()}
    return agg_manifest
