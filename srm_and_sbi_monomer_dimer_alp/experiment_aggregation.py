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

# ---- user-facing presentation ------------------------------------------------------------------------------
# The visible language of the report and the figures (the internal IDs stay unchanged in archives, specifications
# and scripts): what each construction yields, which samples are pooled before one representative is selected, the
# phrase of a window set and the data-scope line every figure carries. One formatter serves the Experiment writer
# and the Nuisance DLI report; the scientific definitions stay in CONSTRUCTION_DEFINITIONS and are not duplicated.
DISPLAY = {
    "per_cell": ("One {W} per cell", "{S} per cell", "all selected window clouds from one cell"),
    "per_window_position": ("One {W} per window position", "{S} per window position",
                            "the clouds at the same position across the included cells"),
    "global": ("One {W} for the included dataset", "{S} for included dataset", "all selected cell-window clouds"),
    "per_cell_window": ("One {W} per cell–window pair", "{S} per cell–window", "one window's own cloud"),
    "reference": ("Existing fixed nuisance vector", "Fixed reference",
                  "loaded from the reference artifact and its selection record (nothing pooled)"),
}
STATISTIC_WORDS = {"sgm": ("SGM vector", "SGM"), "median": ("marginal-median vector", "marginal-median vector")}
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 11: "eleven", 12: "twelve"}


def display_name(construction, *, statistic="sgm", short=False, n=None, reference_name=None) -> str:
    """The visible description of a construction (``short`` = the legend phrase), naming the statistic it
    selects (an SGM, or the marginal-median vector, which is never called an SGM) and, with ``n``, the actual
    number of vectors; the reference names its artifact (``reference_name``)."""
    if construction not in DISPLAY:
        raise AggregationError(f"unknown construction {construction!r}; known: {list(DISPLAY)}")
    if statistic not in STATISTIC_WORDS:
        raise AggregationError(f"unknown statistic {statistic!r}; known: {list(STATISTIC_WORDS)}")
    long, brief, _ = DISPLAY[construction]
    W, S = STATISTIC_WORDS[statistic]
    text = (brief if short else long).format(W=W, S=S)
    if construction == "reference" and reference_name:
        text += f": {reference_name}"
    if n is not None:
        text += " (one vector)" if int(n) == 1 else f" ({int(n)} vectors)"
    return text


def pooled_samples(construction) -> str:
    """Which samples are pooled before one representative is selected (the constructions table's column)."""
    return DISPLAY[construction][2]


def window_set_phrase(name, chunks, geometry, *, window_seconds=None) -> str:
    """The visible phrase of a window set: 'all ten 2 s windows' (every declared position), 'opening 0–2 s
    window only' (the first position alone), else the named set with its positions and window starts."""
    chunks = tuple(int(c) for c in chunks)
    ws = f"{float(window_seconds):g} s " if window_seconds else ""
    if chunks == tuple(expected_positions(geometry)):
        return f"all {NUMBER_WORDS.get(len(chunks), str(len(chunks)))} {ws}windows"
    if chunks == (0,):
        return f"opening 0–{float(window_seconds):g} s window only" if window_seconds else "opening window only"
    starts = ""
    if window_seconds and geometry.get("n_frames"):
        step_s = float(window_seconds) * float(geometry["step_frames"]) / float(geometry["n_frames"])
        starts = " (window starts " + ", ".join(f"{c * step_s:g}" for c in chunks) + " s)"
    return f"named set '{name}' = positions {list(chunks)}{starts}"


def scope_line(kind_display, coverage, n_product_cells, n_product_windows, *, set_name, chunks, geometry, n_draws,
               window_seconds=None, statistic="sgm") -> str:
    """The data-scope line of a figure: which cells and windows contributed (from the aggregation's coverage,
    never from an assumed cohort size; a subset is called a pilot), the window selection with the number of
    selected windows when it is a subset of the source collection, the draws per window and the representative."""
    cells = list(coverage["cells"]); nc, nw = len(cells), int(coverage["n_windows"])
    complete = nc == int(n_product_cells) and nw == int(n_product_windows)
    chunks = tuple(int(c) for c in chunks)
    n_sel = nc * len(chunks)
    if complete:
        head = f"{kind_display} · {nc}/{n_product_cells} cells, {nw}/{n_product_windows} windows"
    else:
        ids = f" ({', '.join(str(c) for c in cells)})" if nc <= 12 else ""
        head = f"{kind_display} · aggregation pilot: {nc}/{n_product_cells} cells{ids}, {nw}/{n_product_windows} windows"
    sel = f" ({n_sel} selected window{'s' if n_sel != 1 else ''} of the {nw})" if n_sel != nw else ""
    return (f"{head} · windows included: {window_set_phrase(set_name, chunks, geometry, window_seconds=window_seconds)}{sel}"
            f" · {int(n_draws):,} joint draws/window · representative: {STATISTIC_WORDS[statistic][1]}")


def scope_table(arrays, manifest, request, condition_display):
    """The early 'Data scope' table of the Experiment report: per condition, the scope of the original-summary
    diagnostics (the product's own windows and the draws summarized when it was written) and of the
    retained-cloud aggregation (the clouds the request supplies, or the product's own, or none), with the cell
    identities when the subset is small. Returns (headers, rows, note, caption_tag); ``caption_tag`` is the
    compact source statement appended to the captions of the original diagnostic figures. Computes nothing."""
    kinds = [str(k) for k in arrays["kinds"]]
    cell_arr = np.asarray(arrays["cell"]).astype(int); kind_index = np.asarray(arrays["kind_index"]).astype(int)
    # Three states, told apart: aggregation explicitly disabled; enabled but no clouds (neither the product's own nor
    # external collections); enabled with clouds (the coverage below).
    disabled = request is None or not request.enabled
    clouds = None
    if not disabled:
        clouds = request.clouds if request.clouds is not None else ("product" if "posterior_samples_cloud" in arrays else None)
    rows, tags = [], []
    for i, k in enumerate(kinds):
        m = kind_index == i
        cells = sorted(set(cell_arr[m].tolist())); nw = int(m.sum())
        rows.append(["original Experiment summaries (the tables and diagnostic figures)", condition_display(k), str(len(cells)), str(nw),
                     f"{int(manifest['n_summary_draws'])} {manifest.get('draw_label', 'draw')}s per window summarized when the product was written"
                     + ("; the draws themselves are stored" if "posterior_samples_cloud" in arrays else "; the draws themselves were not retained"),
                     "every cell of the product"])
        tags.append(f"{condition_display(k)}: {len(cells)} cells, {nw} windows")
        if clouds is None:
            rows.append(["retained-cloud aggregation", condition_display(k), "-", "-",
                         "not computed (aggregation disabled)" if disabled else "not computed (no retained clouds)", "-"])
            continue
        if clouds == "product":
            cc, cw, nd = cells, nw, int(np.asarray(arrays["posterior_samples_cloud"]).shape[1])
        else:
            cm = clouds.kind_index == i
            cc, cw, nd = sorted(set(int(c) for c in clouds.cell[cm])), int(cm.sum()), int(clouds.n_draws)
        complete = len(cc) == len(cells) and cw == nw
        rows.append(["retained-cloud aggregation", condition_display(k), f"{len(cc)} of {len(cells)}", f"{cw} of {nw}",
                     f"{nd:,} joint draws retained per window", "complete cohort" if complete else
                     (f"pilot subset: cells {cc}" if len(cc) <= 12 else f"pilot subset of {len(cc)} cells (identities in aggregation_groups.csv)")])
    note = ("the two scopes are stated separately: the original diagnostics summarize the product's own windows; the aggregation "
            "groups retained joint draws (the product's own, or compatible collections of the same estimator drawn after the run). "
            "A pilot subset never relabels the original figures.")
    tag = "Source: the original Experiment summaries (" + "; ".join(tags) + "), not the retained clouds of the aggregation."
    return ["data", "condition", "cells", "windows", "draws", "coverage"], rows, note, tag


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


MARGINAL_HISTOGRAM_BINS = 30      # bins per parameter of an exact weighted marginal histogram (deterministic edges)


def weighted_histogram(values, weights, edges) -> np.ndarray:
    """The exact bin masses of a finite weighted distribution: ``H(B) = sum_i w_i 1[v_i in B]`` for the bins of
    ``edges`` (half-open ``[a, b)``, the last bin closed, numpy's convention). Nothing is sampled and nothing is
    normalized: the masses add up to the total weight, so every value must lie inside ``[edges[0], edges[-1]]`` (a value
    outside the bins is refused, never dropped silently). A zero weight contributes exactly zero."""
    v = np.asarray(values, np.float64).ravel(); w = np.asarray(weights, np.float64).ravel()
    e = np.asarray(edges, np.float64).ravel()
    if v.shape != w.shape or e.size < 2 or not np.all(np.diff(e) > 0):
        raise AggregationError("weighted_histogram: one weight per value and at least two strictly increasing bin edges are required.")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(w)) and np.all(w >= 0)):
        raise AggregationError("weighted_histogram: values and weights must be finite and the weights non-negative.")
    if v.size and (v.min() < e[0] or v.max() > e[-1]):
        raise AggregationError(f"weighted_histogram: values span [{v.min()!r}, {v.max()!r}], outside the bins [{e[0]!r}, {e[-1]!r}].")
    return np.histogram(v, bins=e, weights=w)[0].astype(np.float64)


def marginal_histograms(vectors, probabilities, baseline, keys, *, n_bins=MARGINAL_HISTOGRAM_BINS, notes=None) -> list:
    """Per parameter, the exact marginal histogram of a finite weighted distribution over complete vectors: every
    coordinate inherits the SAME whole-vector probabilities (``probabilities``; ``baseline`` is a second weighting of the
    same vectors, e.g. the equal-member baseline, binned with the SAME edges). The edges are deterministic: ``n_bins``
    equal-width bins in the stored estimator coordinate (uniform in log10 for a log row) from the smallest to the largest
    stored value of that coordinate over ALL the vectors (zero-probability vectors included, so the two weightings share
    one support and nothing is cropped to the training box). A coordinate with a single stored value (a fixed or
    overridden parameter) is a point mass, not a histogram. Returns one dict per key: ``{"key", "kind": "bins", "edges",
    "selected", "baseline"}`` or ``{"key", "kind": "point_mass", "value", "selected_mass", "baseline_mass", "note"}``
    (``notes``: ``{key: text}`` shown beside a point mass)."""
    V = np.asarray(vectors, np.float64); p = np.asarray(probabilities, np.float64).ravel(); b = np.asarray(baseline, np.float64).ravel()
    if V.ndim != 2 or V.shape[0] != p.size or p.size != b.size or V.shape[1] != len(keys) or int(n_bins) < 1:
        raise AggregationError("marginal_histograms: (n, D) vectors, n probabilities, n baseline weights, D keys and a positive bin count are required.")
    out = []
    for j, k in enumerate(keys):
        col = V[:, j]; lo, hi = float(col.min()), float(col.max())
        if lo == hi:
            out.append({"key": str(k), "kind": "point_mass", "value": lo, "selected_mass": float(p.sum()), "baseline_mass": float(b.sum()),
                        "note": (notes or {}).get(str(k), "")})
            continue
        edges = np.linspace(lo, hi, int(n_bins) + 1)
        edges[0], edges[-1] = lo, hi                     # the exact extremes, whatever linspace rounds to
        out.append({"key": str(k), "kind": "bins", "edges": edges, "selected": weighted_histogram(col, p, edges),
                    "baseline": weighted_histogram(col, b, edges)})
    return out


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
# Posterior compatibility: cross-posterior distance ranks of the SGM representatives
# =============================================================================
# One shared rank primitive (reused by the draws companion's joint-depth table) and the assembly of the
# compatibility matrices over the groups of an aggregation. The candidate is a group's existing SGM v_i; the
# reference is another group's retained cloud (its equal-window mixture) with that group's own SGM v_j:
#
#     R_ij = (1/N_j) sum_m 1{ d(z_jm, v_j) > d(v_i, v_j) },   i != j,   d(a, b) = ||(a - b) / sgm_scale||_2
#
# a distance rank relative to the reference's own SGM and spread (strict comparison, the convention of the
# draws companion), float64, self reference excluded by group identity. The candidate's score is the ordinary
# sample median of its row over the other references (the two middle ranks averaged when their number is even;
# NOT the inverse weighted empirical CDF used for representative-distribution summaries). No peer = status
# ``insufficient_peers`` with a missing score (NaN) and zero references, never a fabricated 0 or 1; a computed
# zero is a valid outcome and stays distinct from a missing score. Scores are raw medians: Experiment reports
# them and never normalizes them into sampling probabilities (the Nuisance DLI constructor does that under an
# explicitly selected policy).
#
# COMPARISON SCOPE: the reference population of a rank-scored group, recorded in the protocol and never implied by
# the aggregation. ``construction`` (what Experiment computes and writes): every group of ONE construction in one
# condition and window set is scored against ALL the other groups of that same construction: per_cell against the
# other included cells (each reference its own equal-window cloud mixture), per_window_position against the other
# selected window positions (each reference its own equal-cell cloud mixture), per_cell_window against every other
# cell-window group (each reference one window's own cloud; the other windows of the candidate's own cell are peers
# like any other, and only the candidate's own group identity is excluded). ``within_cell`` (the hierarchical scope
# of archive formats 2 and 3, kept under its recorded meaning and still computable by name): per_cell as above,
# per_cell_window against the other selected windows of the SAME cell only, per_window_position unscored. global has
# no same-level peer in either scope (``not_applicable``). A scored construction with fewer than two groups carries
# ``insufficient_peers``. Identical vectors with an ``equivalent_to`` alias (per_cell[opening] and
# per_cell_window[opening]) are scored on their own: under ``construction`` their populations and reference clouds
# coincide, so the scores agree by computation; under ``within_cell`` their availability differs. Scores are never
# copied from one group to another.
COMPATIBILITY_PROTOCOL = "sgm_distance_rank_v2"          # written by this implementation: the comparison scope is explicit
LEGACY_COMPATIBILITY_PROTOCOL = "sgm_distance_rank_v1"   # archive formats 2 and 3: no scope field; its populations are within_cell's
COMPATIBILITY_STATUSES = ("computed", "insufficient_peers", "not_applicable")
SCOPE_CONSTRUCTION = "construction"
SCOPE_WITHIN_CELL = "within_cell"
COMPARISON_SCOPES = (SCOPE_CONSTRUCTION, SCOPE_WITHIN_CELL)
# per comparison scope of an archive: the rank-scored constructions and the reference scope of each
SCOPES_BY_CONSTRUCTION = {
    SCOPE_CONSTRUCTION: {"per_cell": SCOPE_CONSTRUCTION, "per_window_position": SCOPE_CONSTRUCTION, "per_cell_window": SCOPE_CONSTRUCTION},
    SCOPE_WITHIN_CELL: {"per_cell": SCOPE_CONSTRUCTION, "per_cell_window": SCOPE_WITHIN_CELL},
}
_REF_CELLS = "the other included cells of the same condition and window set; each reference is its own equal-window cloud mixture"
_REF_GLOBAL = "no same-level peer (uniform selection remains available)"
REFERENCE_POPULATIONS = {
    SCOPE_CONSTRUCTION: {"per_cell": _REF_CELLS,
                         "per_window_position": "the other selected window positions of the same condition and window set; each reference "
                                                "is its own equal-cell cloud mixture",
                         "per_cell_window": "every other cell–window group of the same condition and window set, the other windows of the "
                                            "candidate's own cell included; each reference is one window's own cloud",
                         "global": _REF_GLOBAL},
    SCOPE_WITHIN_CELL: {"per_cell": _REF_CELLS,
                        "per_cell_window": "the other selected windows of the same cell and condition; each reference is one window's own cloud",
                        "per_window_position": "no rank policy in this scope (uniform selection remains available)",
                        "global": _REF_GLOBAL},
}


def rank_scored_constructions(scope=SCOPE_CONSTRUCTION) -> tuple:
    """The constructions that carry a rank score under a comparison scope (construction order)."""
    return tuple(c for c in CONSTRUCTIONS if c in SCOPES_BY_CONSTRUCTION[scope])


RANK_SPACE_ALL = "all"               # the primary space: every parameter coordinate (the format-2 arrays)
_SPACE_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_rank_spaces(spec, keys) -> dict:
    """Parse ``--compatibility-coordinates``: ``LABEL=key,key[;LABEL=key,...]`` into an ordered ``{label: (keys in
    table order)}`` of ADDITIONAL rank coordinate spaces. A label is lowercase ``[a-z][a-z0-9_]*`` and never
    ``all`` (the primary space, always computed); every key must be a parameter key, named once; a space must
    hold at least one key and not every key (that is ``all``). Empty or None = no additional space."""
    spaces = {}
    if spec is None or not str(spec).strip():
        return spaces
    keys = [str(k) for k in keys]
    for item in str(spec).split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise AggregationError(f"rank space {item!r}: expected LABEL=key,key,...")
        label, names = item.split("=", 1)
        label = label.strip()
        if not _SPACE_LABEL.match(label) or label == RANK_SPACE_ALL:
            raise AggregationError(f"rank space label {label!r} must match [a-z][a-z0-9_]* and differ from {RANK_SPACE_ALL!r}.")
        if label in spaces:
            raise AggregationError(f"rank space {label!r} is declared twice.")
        chosen = [n.strip() for n in names.split(",") if n.strip()]
        unknown = [n for n in chosen if n not in keys]
        if unknown:
            raise AggregationError(f"rank space {label!r}: unknown parameter keys {unknown}; keys are {keys}.")
        if len(set(chosen)) != len(chosen):
            raise AggregationError(f"rank space {label!r}: a key is named twice ({chosen}).")
        ordered = tuple(k for k in keys if k in chosen)
        if not ordered or len(ordered) == len(keys):
            raise AggregationError(f"rank space {label!r} must hold at least one and fewer than all {len(keys)} keys "
                                   f"(every key is the primary space {RANK_SPACE_ALL!r}).")
        spaces[label] = ordered
    return spaces


def _key_indices(keys, coordinates) -> np.ndarray:
    keys = [str(k) for k in keys]
    return np.asarray([keys.index(str(c)) for c in coordinates], np.int64)


def reference_distance_ranks(reference_cloud, reference_sgm, candidates, scale) -> np.ndarray:
    """The distance rank of every candidate vector relative to ONE reference posterior: the share of the
    reference's retained draws that lie strictly FARTHER from the reference's own SGM than the candidate does,
    distances in estimator coordinates divided by ``scale``. The reference radii are sorted once and every
    candidate is located by an empirical-CDF lookup (``searchsorted(side='right')`` realizes the strict
    inequality), so one pass over the reference draws serves every candidate. Float64 throughout; returns an
    array of shape ``(n_candidates,)`` in [0, 1]. Near one: the candidate sits closer to the reference SGM than
    most reference draws; near zero: almost every reference draw is closer. A distance rank (radial centrality),
    not a density, not a credible-region membership and not a data-quality verdict."""
    v = np.asarray(reference_sgm, dtype=np.float64).ravel()
    z = np.asarray(reference_cloud, dtype=np.float64).reshape(-1, v.size)
    s = np.asarray(scale, dtype=np.float64).ravel()
    c = np.asarray(candidates, dtype=np.float64).reshape(-1, v.size)
    if z.shape[0] == 0 or s.size != v.size or not np.all(s > 0):
        raise AggregationError("reference_distance_ranks: a non-empty reference cloud and a positive scale of the "
                               "vector's dimension are required.")
    dz = (z - v) / s
    radii = np.sort(np.sqrt((dz * dz).sum(axis=1)))
    dc = (c - v) / s
    q = np.sqrt((dc * dc).sum(axis=1))
    return (radii.size - np.searchsorted(radii, q, side="right")) / float(radii.size)


@dataclass
class CompatibilityBlock:
    """One rank matrix: rows are candidates, columns references, both the SAME ordered groups (indices into
    ``AggregationResult.groups``); the diagonal (self reference) is NaN; ``reference_draws`` counts each
    reference pool's draws. ``scope`` is the block's reference scope: ``construction`` = the block holds every group of
    its construction in one condition and window set (``cell`` -1); ``within_cell`` = the selected windows of ONE cell
    (``cell`` names it)."""
    kind: str
    window_set: str
    construction: str
    cell: int
    group_index: np.ndarray
    ranks: np.ndarray
    reference_draws: np.ndarray
    scope: str = SCOPE_CONSTRUCTION


@dataclass
class CompatibilityResult:
    blocks: list
    score: np.ndarray          # (n_groups,) the median rank; NaN where unavailable
    n_references: np.ndarray   # (n_groups,) int
    status: np.ndarray         # (n_groups,) one of COMPATIBILITY_STATUSES
    seconds: float = 0.0
    label: str = RANK_SPACE_ALL          # the rank coordinate space: "all" or a declared subset label
    coordinates: tuple = None            # the parameter keys the distances were computed over (table order)
    key_indices: tuple = None            # their positions in the parameter table
    scope: str = SCOPE_CONSTRUCTION      # the comparison scope every block of this result was computed under

    def of(self, i) -> dict:
        s = float(self.score[i])
        return {"score": None if np.isnan(s) else s, "n_references": int(self.n_references[i]), "status": str(self.status[i])}


def compatibility_protocol(scale, *, coordinates=None, scope=SCOPE_CONSTRUCTION) -> dict:
    """The protocol record bound into the archive manifest (what the stored ranks and scores mean). ``coordinates``
    names the parameter keys the distances run over (``scale`` holds their prior widths, in the same order); ``scope``
    is the comparison scope (:data:`COMPARISON_SCOPES`): the reference population of every rank-scored construction."""
    if scope not in COMPARISON_SCOPES:
        raise AggregationError(f"comparison scope {scope!r} must be one of {COMPARISON_SCOPES}.")
    scored = rank_scored_constructions(scope)
    return {"name": COMPATIBILITY_PROTOCOL, "coordinates": None if coordinates is None else [str(k) for k in coordinates],
            "comparison_scope": scope, "comparison_scopes": dict(SCOPES_BY_CONSTRUCTION[scope]),
            "rank": "share of the reference's retained draws strictly farther from the reference's own SGM than the candidate "
                    "SGM (comparison '>', empirical-CDF lookup with searchsorted side='right'); rows = candidates, columns = "
                    "references, self reference excluded by group identity (NaN diagonal); generally asymmetric",
            "metric": "estimator coordinates divided by the prior widths (sgm_scale), over the listed coordinates only (every "
                      "parameter key in the primary space 'all'; a declared subset in an additional rank space); coordinates are "
                      "stored as the estimator declares them (log10 for a log row) and are not transformed again",
            "sgm_scale": [float(v) for v in scale],
            "score": "ordinary sample median of the candidate's off-diagonal row (two middle ranks averaged when even); raw, not "
                     "normalized across candidates; a computed zero is a valid outcome distinct from a missing score",
            "reference_populations": dict(REFERENCE_POPULATIONS[scope]), "statuses": list(COMPATIBILITY_STATUSES),
            "availability": (f"rank-scored constructions under scope {scope!r}: {list(scored)}; computed when the block holds at least "
                             "two groups; insufficient_peers (no score, zero references) for a scored construction whose block would "
                             "hold one group (one included cell, one selected window position, or one cell–window group"
                             + ("; within_cell: a cell with one selected window" if scope == SCOPE_WITHIN_CELL else "")
                             + "); not_applicable for every other construction (global has no same-level peer"
                             + ("; per_window_position carries no rank in this scope" if scope == SCOPE_WITHIN_CELL else "")
                             + "); equivalent groups keep their own scores, never copied from one another"),
            "precision": "float64"}


def _rank_block(result, coll, kind, set_name, construction, cell, group_indices, pools, scale, out, idx=None,
                scope=SCOPE_CONSTRUCTION) -> CompatibilityBlock:
    """Fill one block: every reference pool scores every candidate SGM of the block; writes score/references/status.
    ``idx`` (coordinate positions) projects the candidates, the reference SGMs and the clouds onto a rank space; ``scale``
    is already restricted to it."""
    score, n_ref, status = out
    gi = np.asarray(group_indices, np.int64); n = gi.size
    cand = np.asarray([result.groups[i].sgm_log10 for i in gi], np.float64)
    if idx is not None:
        cand = cand[:, idx]
    ranks = np.full((n, n), np.nan); draws = np.zeros(n, np.int64)
    for j in range(n):
        flat = _flat_pool(coll, pools[j])
        draws[j] = flat.shape[0]
        ref_sgm = np.asarray(result.groups[int(gi[j])].sgm_log10, np.float64)
        if idx is not None:
            flat = np.asarray(flat, np.float64)[:, idx]; ref_sgm = ref_sgm[idx]
        col = reference_distance_ranks(flat, ref_sgm, cand, scale)
        col[j] = np.nan                                   # the self reference, excluded by identity (never by vector equality)
        ranks[:, j] = col
    for a, i in enumerate(gi):
        row = np.delete(ranks[a], a)
        score[i] = float(np.median(row)); n_ref[i] = row.size; status[i] = "computed"
    return CompatibilityBlock(kind=kind, window_set=set_name, construction=construction, cell=int(cell), group_index=gi,
                              ranks=ranks, reference_draws=draws, scope=str(scope))


def compute_compatibility(result: AggregationResult, coll: CloudCollection, *, coordinates=None, label=RANK_SPACE_ALL,
                          scope=SCOPE_CONSTRUCTION) -> CompatibilityResult:
    """The compatibility matrices and median scores of every rank-scored group of an aggregation, from the SAME
    validated clouds the groups were built from (call after :func:`aggregate` and :func:`verify_members`). ``scope``
    is the comparison scope (see the section comment). ``construction`` (the default, what Experiment writes): per
    (condition, window set) ONE block over the complete population of each of per_cell (the included cells),
    per_window_position (the selected positions) and per_cell_window (every cell–window group), each block needing at
    least two groups. ``within_cell``: the per_cell block as above and one per_cell_window block per cell over its
    selected windows; per_window_position unscored. ``coordinates`` (parameter keys, any order; None = every key)
    restricts the distances to a rank coordinate space named ``label``: the candidates, the reference SGMs and the
    reference clouds are projected onto those keys and the scale is their prior widths; the groups, the SGMs and the
    reference populations are exactly those of the primary space."""
    t0 = time.time()
    if scope not in COMPARISON_SCOPES:
        raise AggregationError(f"comparison scope {scope!r} must be one of {COMPARISON_SCOPES}.")
    scopes = SCOPES_BY_CONSTRUCTION[scope]
    n_groups = len(result.groups)
    score = np.full(n_groups, np.nan); n_ref = np.zeros(n_groups, np.int64)
    status = np.array(["not_applicable"] * n_groups, dtype=object)
    keys = [str(k) for k in result.keys]
    if coordinates is None:
        coords, idx = tuple(keys), None
        scale = np.asarray(result.scale, np.float64)
    else:
        if label == RANK_SPACE_ALL and sorted(str(c) for c in coordinates) != sorted(keys):
            raise AggregationError(f"rank space {RANK_SPACE_ALL!r} is every coordinate; a subset needs its own label.")
        unknown = [str(c) for c in coordinates if str(c) not in keys]
        if unknown or not len(coordinates):
            raise AggregationError(f"rank space {label!r}: coordinates {list(coordinates)} are not parameter keys {keys}.")
        coords = tuple(k for k in keys if k in {str(c) for c in coordinates})
        idx = _key_indices(keys, coords)
        scale = np.asarray(result.scale, np.float64)[idx]
        if len(coords) == len(keys):
            idx = None
    index = {(g.construction, g.window_set, g.kind, int(g.cell), int(g.chunk)): i for i, g in enumerate(result.groups)}
    rows_of = {(int(coll.kind_index[r]), int(coll.cell[r]), int(coll.chunk[r])): int(r) for r in range(coll.n_windows)}
    blocks = []
    out = (score, n_ref, status)

    def block(kind, set_name, construction, cell, gi, pools):
        """One reference population: its groups are rank-scored (``insufficient_peers`` until a block scores them)."""
        for i in gi:
            status[i] = "insufficient_peers"
        if len(gi) >= 2:
            blocks.append(_rank_block(result, coll, kind, set_name, construction, cell, gi, pools, scale, out, idx,
                                      scope=scopes[construction]))

    for kind in result.coverage:
        ki = coll.kinds.index(kind)
        cells = [int(c) for c in result.coverage[kind]["cells"]]
        for set_name, chunks in result.window_sets.items():
            chunks = [int(w) for w in chunks]
            block(kind, set_name, "per_cell", -1, [index[("per_cell", set_name, kind, c, -1)] for c in cells],
                  [np.asarray([rows_of[(ki, c, w)] for w in chunks]) for c in cells])
            if "per_window_position" in scopes:
                block(kind, set_name, "per_window_position", -1, [index[("per_window_position", set_name, kind, -1, w)] for w in chunks],
                      [np.asarray([rows_of[(ki, c, w)] for c in cells]) for w in chunks])
            if scopes["per_cell_window"] == SCOPE_CONSTRUCTION:
                # the complete cell–window population: every other group is a reference, same-cell windows included
                block(kind, set_name, "per_cell_window", -1, [index[("per_cell_window", set_name, kind, c, w)] for c in cells for w in chunks],
                      [np.asarray([rows_of[(ki, c, w)]]) for c in cells for w in chunks])
            else:
                for c in cells:
                    block(kind, set_name, "per_cell_window", c, [index[("per_cell_window", set_name, kind, c, w)] for w in chunks],
                          [np.asarray([rows_of[(ki, c, w)]]) for w in chunks])
    return CompatibilityResult(blocks=blocks, score=score, n_references=n_ref, status=np.asarray(status.tolist(), dtype=str),
                               seconds=time.time() - t0, label=str(label), coordinates=coords,
                               key_indices=tuple(int(i) for i in (_key_indices(keys, coords) if idx is None else idx)), scope=str(scope))


def compute_rank_spaces(result: AggregationResult, coll: CloudCollection, spaces, *, scope=SCOPE_CONSTRUCTION) -> list:
    """The additional rank coordinate spaces (``{label: coordinates}``, see :func:`parse_rank_spaces`), each a
    :class:`CompatibilityResult` over the same groups, comparison scope and reference populations as the primary space."""
    return [compute_compatibility(result, coll, coordinates=coords, label=label, scope=scope) for label, coords in (spaces or {}).items()]


def sgm_outside_prior_keys(result: AggregationResult) -> list:
    """Per group, the keys whose SGM coordinate lies outside the training prior (';'-joined; '' = inside on every key).
    Whether the SGM itself is outside, a different quantity from the outside-box fractions of the pooled draws."""
    out = []
    for g in result.groups:
        v = np.asarray(g.sgm_log10, float)
        out.append(";".join(k for k, lo, hi, x in zip(result.keys, result.prior_low, result.prior_high, v) if x < lo or x > hi))
    return out


# =============================================================================
# Exports
# =============================================================================
ARCHIVE_FORMAT_VERSION = 4             # v2: compatibility scores, statuses, rank-matrix blocks and the protocol beside v1's arrays;
                                       # v3: additional rank coordinate spaces (a declared key subset each) beside v2's arrays, unchanged;
                                       # v4: the comparison scope is explicit (protocol sgm_distance_rank_v2, ``compatibility.comparison_scope``,
                                       #     a ``scope`` per block); the same array names now hold the scores of the DECLARED scope, so a
                                       #     format-2/3 archive (within_cell by its recorded populations) never supplies construction-wide scores
SUPPORTED_ARCHIVE_FORMATS = (1, 2, 3, 4)   # a v1 archive (no compatibility data) still supplies a uniform selection
LEGACY_SCOPE_FORMATS = (2, 3)          # formats whose protocol is sgm_distance_rank_v1: their comparison scope is within_cell, by record
ARCHIVE_NAME = "aggregation_candidates.npz"


def _space_prefix(label=RANK_SPACE_ALL) -> str:
    return "compatibility" if str(label) == RANK_SPACE_ALL else f"compatibility_{label}"


def _block_name(b, label=RANK_SPACE_ALL) -> str:
    return f"{_space_prefix(label)}_block_{b}"


def _space_arrays(compatibility: CompatibilityResult) -> dict:
    """The archive arrays of one rank space (the primary space keeps the format-2 names)."""
    p = _space_prefix(compatibility.label)
    out = {f"{p}_score": np.asarray(compatibility.score, np.float64),
           f"{p}_n_references": np.asarray(compatibility.n_references, np.int64),
           f"{p}_status": np.asarray(compatibility.status, dtype=str)}
    for b, blk in enumerate(compatibility.blocks):
        name = _block_name(b, compatibility.label)
        out[f"{name}_group_index"] = np.asarray(blk.group_index, np.int64)
        out[f"{name}_ranks"] = np.asarray(blk.ranks, np.float64)
        out[f"{name}_reference_draws"] = np.asarray(blk.reference_draws, np.int64)
    return out


def _space_section(result: AggregationResult, compatibility: CompatibilityResult) -> dict:
    """One rank space's manifest section: protocol (with its coordinates and restricted scale), array names, block map, counts."""
    label = compatibility.label; p = _space_prefix(label)
    coords = tuple(compatibility.coordinates or [str(k) for k in result.keys])
    idx = _key_indices(result.keys, coords)
    scale = np.asarray(result.scale, np.float64)[idx]
    counts = {s: int(np.sum(compatibility.status == s)) for s in COMPATIBILITY_STATUSES}
    zero = int(np.sum((compatibility.status == "computed") & (compatibility.score == 0.0)))
    blocks = []
    for b, blk in enumerate(compatibility.blocks):
        name = _block_name(b, label)
        blocks.append({"index": b, "kind": blk.kind, "window_set": blk.window_set, "construction": blk.construction,
                       "cell": int(blk.cell), "scope": str(blk.scope), "n_groups": int(blk.group_index.size),
                       "arrays": {"group_index": f"{name}_group_index", "ranks": f"{name}_ranks", "reference_draws": f"{name}_reference_draws"},
                       "orientation": "rows = candidates, columns = references, both in group_index order; diagonal NaN"})
    return {"label": str(label), "coordinates": list(coords), "key_indices": [int(i) for i in idx],
            "protocol": compatibility_protocol(scale, coordinates=coords, scope=compatibility.scope),
            "arrays": {"score": f"{p}_score", "n_references": f"{p}_n_references", "status": f"{p}_status"},
            "blocks": blocks, "status_counts": counts, "n_zero_scores": zero, "seconds": round(float(compatibility.seconds), 3)}


def compatibility_manifest(result: AggregationResult, compatibility: CompatibilityResult, rank_spaces=()) -> dict:
    """The ``compatibility`` section of the archive manifest: the primary space's protocol, block map (which arrays hold
    which matrix, with the block's condition, window set, construction, reference scope and cell context) and
    per-status counts, the archive's ``comparison_scope`` (one per archive, every space alike) with the reference scope of
    each rank-scored construction (``comparison_scopes``), plus ``rank_spaces``: every space of the archive by label (the primary space ``all``
    over every parameter key first, then each additional declared subset) with its coordinates, protocol, arrays, blocks
    and counts. The protocols are bound to the archive's own target, checkpoint, coverage, input conversion and source
    hashes through the manifest that carries this section."""
    if compatibility.label != RANK_SPACE_ALL:
        raise AggregationError(f"the primary compatibility result must be the space {RANK_SPACE_ALL!r}; got {compatibility.label!r}.")
    labels = [c.label for c in rank_spaces]
    if RANK_SPACE_ALL in labels or len(set(labels)) != len(labels):
        raise AggregationError(f"additional rank spaces need distinct labels other than {RANK_SPACE_ALL!r}; got {labels}.")
    primary = _space_section(result, compatibility)
    spaces = {RANK_SPACE_ALL: {k: v for k, v in primary.items() if k not in ("protocol", "blocks")}}
    spaces[RANK_SPACE_ALL]["protocol_and_blocks"] = "the compatibility section's own protocol and blocks (the format-2 fields)"
    for c in rank_spaces:
        if c.score.shape != compatibility.score.shape:
            raise AggregationError(f"rank space {c.label!r} does not cover the archive's groups.")
        if c.scope != compatibility.scope:
            raise AggregationError(f"rank space {c.label!r} was computed under comparison scope {c.scope!r}; the archive's is "
                                   f"{compatibility.scope!r} (one scope per archive, every space alike).")
        spaces[c.label] = _space_section(result, c)
    return {"comparison_scope": str(compatibility.scope), "comparison_scopes": dict(SCOPES_BY_CONSTRUCTION[compatibility.scope]),
            "protocol": primary["protocol"], "arrays": {**primary["arrays"], "sgm_outside_prior_keys": "sgm_outside_prior_keys"},
            "blocks": primary["blocks"], "status_counts": primary["status_counts"], "n_zero_scores": primary["n_zero_scores"],
            "seconds": primary["seconds"], "rank_spaces": spaces,
            "bound_to": "the target, contract (checkpoint, keys, scale, geometry, draws, pool mode, input conversion), coverage "
                        "and sources recorded in this manifest"}


def export_manifest(result: AggregationResult, *, product_label, condition_labels, reference=None, coverage_note="",
                    product_manifest=None, n_product_cells=None, n_product_windows=None, extra=None, target=None,
                    compatibility: CompatibilityResult = None, rank_spaces=()) -> dict:
    """``target`` binds the archive to the Experiment it came from (``project_alias``, ``condition`` = the run's
    condition, ``timing_label``, ``window_seconds``, ``n_frames``, the estimator tag); a Nuisance DLI built from
    the archive must name the same target, so a cross-condition application never becomes another condition's
    native nuisance by accident. ``compatibility`` (required) adds the protocol and the block map of the primary
    rank matrices; ``rank_spaces`` the additional rank coordinate spaces (:func:`compute_rank_spaces`)."""
    if compatibility is None:
        raise AggregationError("export_manifest: the candidate archive carries the compatibility result (compute_compatibility).")
    return {
        "compatibility": compatibility_manifest(result, compatibility, rank_spaces),
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


# Up to this many sources are named inline in the report with their labels and checksums; a larger set (the complete
# cohort's one collection per cell) is counted and summarized inline, every path, label, seed, checksum and write time
# staying in aggregation_manifest.json (`sources`, `disclosed_source_differences`).
SOURCES_LISTED_INLINE = 6


def sources_description(sources) -> str:
    """The inline description of the retained sources: each named with label and checksum up to
    ``SOURCES_LISTED_INLINE``; beyond that, the count, the labels' form and the first and last file."""
    def name(s):
        return Path(s["path"]).name if s.get("path") else s["label"]
    if len(sources) <= SOURCES_LISTED_INLINE:
        return "; ".join(f"`{name(s)}`" + (f" (collection label {s['collection_label']})" if s.get("collection_label") else "")
                         + (f", sha256 {s['sha256'][:12]}…" if s.get("sha256") else "") for s in sources)
    labels = [s.get("collection_label") for s in sources]
    n_lab = sum(1 for v in labels if v)
    return (f"{len(sources)} retained draw collections, each named with its path, collection label and sha256 in "
            f"aggregation_manifest.json (`sources`): first `{name(sources[0])}`, last `{name(sources[-1])}`; "
            f"{n_lab} carry a collection label" + (f" ({', '.join(sorted(set(v for v in labels if v))[:3])}, …)" if n_lab > 3 else ""))


def disclosed_differences_summary(sources) -> str:
    """One sentence counting the disclosed (non-contract) differences of a large source set: distinct labels,
    seeds and implementation hashes, and the range of write times; the values themselves stay in the manifest."""
    d = _disclosed_differences(sources)
    if not d:
        return ""
    parts = []
    for k, vals in d.items():
        if k == "written_at":
            times = sorted(str(v) for v in vals if v is not None)
            parts.append(f"written between {times[0]} and {times[-1]}")
        else:
            parts.append(f"{len({json.dumps(v, sort_keys=True, default=str) for v in vals})} distinct {k.replace('_', ' ')} values")
    return ("Disclosed differences between the sources (not part of the computation contract; listed per source in "
            "aggregation_manifest.json under `disclosed_source_differences`): " + ", ".join(parts) + ".")


def _disclosed_differences(sources) -> dict:
    out = {}
    for k in ("collection_label", "seed_policy", "implementation_sha256", "written_at"):
        vals = {json.dumps(s.get(k), sort_keys=True, default=str) for s in sources}
        if len(vals) > 1:
            out[k] = [s.get(k) for s in sources]
    return out


def write_aggregation_exports(result: AggregationResult, out_dir, table, *, manifest: dict, compatibility: CompatibilityResult,
                              reference=None, rank_spaces=()) -> dict:
    """Write the candidate archive (format 4: v1's arrays unchanged in content and row order; the compatibility scores,
    statuses, SGM-outside-prior keys and rank-matrix blocks of the primary space under the result's comparison scope, plus
    one array set per additional rank coordinate space, ``compatibility_<label>_*``), ``aggregation_groups.csv`` (v1's
    columns plus score, status, references and outside-prior keys, then one score/status/references triple per additional
    space), ``aggregation_comparison.csv``, ``aggregation_compatibility.csv`` (one row per off-diagonal
    candidate/reference pair of the primary space, with the pair's reference scope) and
    ``aggregation_compatibility_<label>.csv`` per additional space, and the manifest into ``out_dir``; return their paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    G = result.groups; D = len(result.keys); Q = len(result.levels)
    if compatibility is None or compatibility.score.shape != (len(G),) or compatibility.label != RANK_SPACE_ALL:
        raise AggregationError("write_aggregation_exports: the primary ('all') compatibility result over exactly the archive's groups is required.")
    for c in rank_spaces:
        if c.score.shape != (len(G),) or c.label == RANK_SPACE_ALL:
            raise AggregationError(f"write_aggregation_exports: rank space {c.label!r} must cover the archive's groups under its own label.")
    outside_keys = sgm_outside_prior_keys(result)
    compat_arrays = {**_space_arrays(compatibility), "sgm_outside_prior_keys": np.asarray(outside_keys, dtype=str)}
    for c in rank_spaces:
        compat_arrays.update(_space_arrays(c))
    arrays = dict(
        **compat_arrays,
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
                   + [f"q05_{k}" for k in result.keys] + [f"q95_{k}" for k in result.keys]
                   + ["compatibility_score", "compatibility_status", "compatibility_n_references", "sgm_outside_prior_keys"]
                   + [f"{col}_{c.label}" for c in rank_spaces for col in ("compatibility_score", "compatibility_status", "compatibility_n_references")])
        for i, g in enumerate(G):
            sgm_p = to_physical(g.sgm_log10, table); med_p = to_physical(g.median_log10, table)
            q05 = to_physical(g.quantiles_log10[:, 0], table); q95 = to_physical(g.quantiles_log10[:, -1], table)
            sc = float(compatibility.score[i])
            extra = []
            for c in rank_spaces:
                s_ = float(c.score[i]); extra += ["" if np.isnan(s_) else repr(s_), str(c.status[i]), int(c.n_references[i])]
            w.writerow([g.kind, g.window_set, g.construction, g.cell, g.chunk, len(g.members), g.n_draws, g.sgm_method, g.equivalent_to,
                        g.sgm_source["cell"], g.sgm_source["chunk"], g.sgm_source["draw_index"], g.sgm_source["source_label"],
                        f"{g.outside_box_fraction:.6f}"] + [repr(float(v)) for v in sgm_p] + [repr(float(v)) for v in med_p]
                       + [repr(float(v)) for v in q05] + [repr(float(v)) for v in q95]
                       + ["" if np.isnan(sc) else repr(sc), str(compatibility.status[i]), int(compatibility.n_references[i]), outside_keys[i]] + extra)
    comparison_csv = out_dir / "aggregation_comparison.csv"
    with comparison_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "window_set", "construction", "statistic", "parameter", "n_representatives", "median_physical",
                    "q25_physical", "q75_physical", "q05_physical", "q95_physical", "outside_box_mass",
                    "reference_physical", "median_vs_reference_percent", "median_vs_reference_dex"])
        for row in comparison_rows(result, table, reference=reference):
            w.writerow(row)
    def write_pairs(path, comp):
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kind", "window_set", "construction", "cell_context", "candidate_group", "candidate_cell", "candidate_chunk",
                        "reference_group", "reference_cell", "reference_chunk", "reference_draws", "rank", "reference_scope"])
            for blk in comp.blocks:
                ctx = "" if blk.cell < 0 else int(blk.cell)
                for a, i in enumerate(blk.group_index):
                    gi = G[int(i)]
                    for b, j in enumerate(blk.group_index):
                        if a == b:
                            continue
                        gj = G[int(j)]
                        w.writerow([blk.kind, blk.window_set, blk.construction, ctx, int(i), gi.cell, gi.chunk, int(j), gj.cell, gj.chunk,
                                    int(blk.reference_draws[b]), repr(float(blk.ranks[a, b])), blk.scope])
    compatibility_csv = out_dir / "aggregation_compatibility.csv"
    write_pairs(compatibility_csv, compatibility)
    space_csvs = {}
    for c in rank_spaces:
        space_csvs[c.label] = out_dir / f"aggregation_compatibility_{c.label}.csv"
        write_pairs(space_csvs[c.label], c)
    manifest_path = out_dir / "aggregation_manifest.json"
    manifest_path.write_text(json.dumps({**manifest, "archive_sha256": _sha256(archive)}, indent=1, default=str))
    return {"archive": archive, "groups_csv": groups_csv, "comparison_csv": comparison_csv, "compatibility_csv": compatibility_csv,
            "manifest": manifest_path, "rank_space_csvs": space_csvs}


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
    """Read ``aggregation_candidates.npz`` back: arrays plus the decoded manifest; verifies the row counts. Accepts
    format 1 (no compatibility data: ``compatibility`` is None and only a uniform selection can be built from it),
    formats 2 and 3 (the primary rank space ``all``; additional rank coordinate spaces in format 3; comparison scope
    ``within_cell`` by their recorded reference populations) and format 4 (the comparison scope declared in the
    archive), whose compatibility data are validated here, every space alike, under the archive's own scope
    (:func:`validate_archive_compatibility`): a reader never builds a weighted selection from inconsistent scores, and
    never reads one scope's scores as another's."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    manifest = json.loads(str(arrays.pop("manifest_json")))
    version = int(manifest.get("archive_format_version", 0))
    if version not in SUPPORTED_ARCHIVE_FORMATS:
        raise AggregationError(f"{path}: archive format {manifest.get('archive_format_version')!r} is not one of {SUPPORTED_ARCHIVE_FORMATS}.")
    n = arrays["construction"].shape[0]
    for k in ("window_set", "kind", "cell", "chunk", "sgm_log10", "median_log10", "quantiles_log10", "sgm_method"):
        if arrays[k].shape[0] != n:
            raise AggregationError(f"{path}: {k} has {arrays[k].shape[0]} rows; {n} groups.")
    keys = [str(k) for k in arrays["parameter_keys"]]
    if keys != list(manifest["parameter_keys"]):
        raise AggregationError(f"{path}: parameter_keys {keys} != manifest {manifest['parameter_keys']}.")
    compat = validate_archive_compatibility(arrays, manifest, where=str(path)) if version >= 2 else None
    return {"arrays": arrays, "manifest": manifest, "path": str(path), "sha256": _sha256(path), "keys": keys,
            "format_version": version, "compatibility": compat}


def _int_array(arr, name, where) -> np.ndarray:
    """An identity or count array must be stored as integers; a fractional value is refused, never coerced."""
    a = np.asarray(arr)
    if not np.issubdtype(a.dtype, np.integer):
        raise AggregationError(f"{where}: {name} must be an integer array; got dtype {a.dtype} (fractional identities or counts are "
                               f"not coerced into integers).")
    return a.astype(np.int64)


def archive_comparison_scope(manifest, *, where="archive") -> str:
    """The comparison scope an archive's stored scores were computed under. Formats 2 and 3 (protocol
    ``sgm_distance_rank_v1``) carry no scope field: their recorded reference populations are the ``within_cell`` scope,
    and a scope field in such an archive is refused. Format 4 (protocol ``sgm_distance_rank_v2``) declares it in
    ``compatibility.comparison_scope`` with the per-construction map; anything else is refused."""
    section = manifest.get("compatibility") or {}
    version = int(manifest.get("archive_format_version", 2))
    declared = section.get("comparison_scope")
    if version in LEGACY_SCOPE_FORMATS:
        if declared is not None:
            raise AggregationError(f"{where}: a format-{version} archive carries no comparison scope field (its populations are "
                                   f"{SCOPE_WITHIN_CELL!r} by record); found {declared!r}.")
        return SCOPE_WITHIN_CELL
    if declared not in COMPARISON_SCOPES:
        raise AggregationError(f"{where}: a format-{version} archive declares compatibility.comparison_scope, one of {COMPARISON_SCOPES}; "
                               f"got {declared!r}.")
    if dict(section.get("comparison_scopes") or {}) != SCOPES_BY_CONSTRUCTION[declared]:
        raise AggregationError(f"{where}: compatibility.comparison_scopes {section.get('comparison_scopes')} is not the map of scope "
                               f"{declared!r} ({SCOPES_BY_CONSTRUCTION[declared]}).")
    return str(declared)


def validate_archive_compatibility(arrays, manifest, *, where="archive") -> dict:
    """Validate an archive's compatibility data (format 2: the primary rank space ``all``; formats 3 and 4: ``all`` plus
    every additional rank coordinate space declared in ``compatibility.rank_spaces``) against its own groups, its
    comparison scope (:func:`archive_comparison_scope`) and its declared contexts. Common to all spaces, the reference
    CONTEXTS are derived from the candidate groups themselves under that scope. ``construction``: every (condition,
    window set) holds one context per rank-scored construction (per_cell, per_window_position, per_cell_window) over ALL
    its groups. ``within_cell``: one per_cell context per (condition, window set) and one per_cell_window context per
    (condition, window set, cell). A context with at least two groups must hold exactly one block over exactly its
    complete peer population (a population split into several blocks, a missing peer, an unexpected block, a block of
    another scope and a false ``insufficient_peers`` are refused); a context with one group must carry
    ``insufficient_peers`` with no score; the scientific identities of the rank-scored groups must be distinct (one
    group per cell in a per_cell context, chunk -1; one per window position in a per_window_position context, cell -1;
    one per (cell, window position) pair in a construction-wide per_cell_window context, where cell identities repeat
    legitimately; one per window position in a within-cell context). Per block: integer identities and counts (never
    coerced), unique in-range group indices, reference draw counts equal to the referenced groups' pooled draw counts, a
    NaN diagonal, finite off-diagonal ranks in [0, 1], and the stored score equal to the row median with the right
    reference count. Per space: the protocol name of the archive's format, the same comparison scope, its declared
    coordinates (every key for ``all``; a proper non-empty subset in table order otherwise, named once) and its scale,
    which must be the archive's ``sgm_scale`` restricted to those coordinates (and, for ``all``, equal to the manifest's
    and the contract's). Returns ``{"protocol", "blocks", "by_group", "coordinates", "comparison_scope",
    "comparison_scopes", "spaces": {label: {...}}}`` where the top level is the primary space ``all``; raises
    :class:`AggregationError` naming the first inconsistency."""
    section = manifest.get("compatibility") or {}
    version = int(manifest.get("archive_format_version", 2))
    keys = [str(k) for k in arrays["parameter_keys"]]
    scale_all = np.asarray(arrays["sgm_scale"], float)
    for label, other in (("manifest", manifest.get("sgm_scale")), ("contract", (manifest.get("contract") or {}).get("sgm_scale"))):
        if other is None or np.asarray(other, float).shape != scale_all.shape or not np.array_equal(np.asarray(other, float), scale_all):
            raise AggregationError(f"{where}: the {label} sgm_scale {other} is not the archive's {scale_all.tolist()}; the stored ranks were "
                                   f"computed under another metric.")
    scope = archive_comparison_scope(manifest, where=where)
    contexts = _derive_rank_contexts(arrays, where, scope)
    declared = section.get("rank_spaces")
    if version >= 3:
        if not isinstance(declared, dict) or RANK_SPACE_ALL not in declared:
            raise AggregationError(f"{where}: a format-{version} archive declares its rank spaces under compatibility.rank_spaces, with "
                                   f"{RANK_SPACE_ALL!r} first.")
        primary_decl = declared[RANK_SPACE_ALL]
        if list(primary_decl.get("coordinates") or []) != keys:
            raise AggregationError(f"{where}: the primary rank space {RANK_SPACE_ALL!r} must list every parameter key {keys}; got "
                                   f"{primary_decl.get('coordinates')}.")
    primary = _validate_rank_space(arrays, section, contexts, where=where, label=RANK_SPACE_ALL, coordinates=keys, keys=keys,
                                   scale_all=scale_all, require_coordinates=version >= 3, scope=scope, version=version)
    spaces = {RANK_SPACE_ALL: primary}
    if version >= 3:
        for label, decl in declared.items():
            if label == RANK_SPACE_ALL:
                continue
            if not _SPACE_LABEL.match(str(label)):
                raise AggregationError(f"{where}: rank space label {label!r} must match [a-z][a-z0-9_]*.")
            coords = [str(c) for c in (decl.get("coordinates") or [])]
            if not coords or len(coords) >= len(keys) or len(set(coords)) != len(coords) or any(c not in keys for c in coords) \
                    or coords != [k for k in keys if k in set(coords)]:
                raise AggregationError(f"{where}: rank space {label!r} declares coordinates {coords}; a proper non-empty subset of {keys} in "
                                       f"table order, each key once, is required.")
            spaces[str(label)] = _validate_rank_space(arrays, decl, contexts, where=where, label=str(label), coordinates=coords, keys=keys,
                                                      scale_all=scale_all, require_coordinates=True, scope=scope, version=version)
    return {**primary, "comparison_scope": scope, "comparison_scopes": dict(SCOPES_BY_CONSTRUCTION[scope]), "spaces": spaces}


def _derive_rank_contexts(arrays, where, scope) -> dict:
    """The reference contexts of an archive from its candidate groups under its comparison scope (shared by every rank
    space): ``{(kind, set, construction, cell or -1): [group indices]}`` after the distinct-identity checks. The cell
    slot names a cell only for a within-cell per_cell_window context; every construction-wide context carries -1."""
    scopes = SCOPES_BY_CONSTRUCTION[scope]
    n = arrays["construction"].shape[0]
    constr = np.asarray(arrays["construction"]).astype(str); ws = np.asarray(arrays["window_set"]).astype(str)
    kind = np.asarray(arrays["kind"]).astype(str)
    cell = _int_array(arrays["cell"], "cell", where); chunk = _int_array(arrays["chunk"], "chunk", where)
    # the scientific identities of the rank-scored groups must be distinct BEFORE their reference contexts are derived
    # from them: one group per distinct cell in a per_cell context (pooled cell groups carry chunk = -1), one per distinct
    # window position in a per_window_position context (cell = -1) or in a within-cell per_cell_window context, one per
    # distinct (cell, window position) pair in a construction-wide per_cell_window context; unique group-array indices
    # alone do not establish it
    expected = {}
    for i in range(n):
        c = constr[i]
        if c not in scopes:
            continue
        if c == "per_cell":
            if chunk[i] != -1:
                raise AggregationError(f"{where}: group {i} ({kind[i]}/{ws[i]}/per_cell cell {cell[i]}) carries chunk {chunk[i]}; a pooled "
                                       f"cell group carries chunk -1.")
            expected.setdefault((kind[i], ws[i], c, -1), []).append(i)
        elif c == "per_window_position":
            if cell[i] != -1 or chunk[i] < 0:
                raise AggregationError(f"{where}: group {i} ({kind[i]}/{ws[i]}/per_window_position) carries cell {cell[i]} and chunk {chunk[i]}; "
                                       f"a pooled position group carries cell -1 and its window position.")
            expected.setdefault((kind[i], ws[i], c, -1), []).append(i)
        else:
            if chunk[i] < 0:
                raise AggregationError(f"{where}: group {i} ({kind[i]}/{ws[i]}/per_cell_window cell {cell[i]}) carries no window position.")
            expected.setdefault((kind[i], ws[i], c, int(cell[i]) if scopes[c] == SCOPE_WITHIN_CELL else -1), []).append(i)
    for key, members in expected.items():
        if key[2] == "per_cell":
            ident, label = [int(v) for v in cell[members]], "cell"
        elif key[2] == "per_window_position" or key[3] >= 0:
            ident, label = [int(v) for v in chunk[members]], "window position"
        else:
            ident, label = [(int(a), int(b)) for a, b in zip(cell[members], chunk[members])], "(cell, window position) pair"
        if len(set(ident)) != len(members):
            dup = sorted({v for v in ident if ident.count(v) > 1})
            raise AggregationError(f"{where}: the context {_ctx_name(key)} holds {len(members)} groups over {len(set(ident))} distinct "
                                   f"{label}s (duplicated {label} identities {dup[:10]}); one representative per distinct {label} is required.")
    return expected


def _ctx_name(key):
    return f"{key[0]}/{key[1]}/{key[2]}" + (f" cell {key[3]}" if key[2] == "per_cell_window" and key[3] >= 0 else "")


def _validate_rank_space(arrays, section, expected, *, where, label, coordinates, keys, scale_all, require_coordinates,
                         scope=SCOPE_WITHIN_CELL, version=3) -> dict:
    """Validate ONE rank space's protocol, arrays and blocks against the shared contexts of the archive's comparison
    ``scope`` (see :func:`validate_archive_compatibility`); returns ``{"protocol", "blocks", "by_group", "coordinates",
    "label"}``."""
    proto = section.get("protocol") or {}
    sp = f"rank space {label!r}"
    scopes = SCOPES_BY_CONSTRUCTION[scope]
    legacy = version in LEGACY_SCOPE_FORMATS
    want_name = LEGACY_COMPATIBILITY_PROTOCOL if legacy else COMPATIBILITY_PROTOCOL
    if proto.get("name") != want_name:
        raise AggregationError(f"{where}: {sp}: compatibility protocol {proto.get('name')!r} is not {want_name!r}, the protocol of archive "
                               f"format {version}; the stored ranks are not the ones this reader interprets.")
    if legacy:
        if proto.get("comparison_scope") is not None:
            raise AggregationError(f"{where}: {sp}: a {LEGACY_COMPATIBILITY_PROTOCOL} protocol carries no comparison scope; found "
                                   f"{proto.get('comparison_scope')!r}.")
    elif proto.get("comparison_scope") != scope or dict(proto.get("comparison_scopes") or {}) != scopes:
        raise AggregationError(f"{where}: {sp}: the protocol's comparison scope {proto.get('comparison_scope')!r} "
                               f"({proto.get('comparison_scopes')}) is not the archive's {scope!r} ({scopes}); every space of an archive "
                               f"shares one comparison scope.")
    pc = proto.get("coordinates")
    if pc is not None or require_coordinates:
        if [str(c) for c in (pc or [])] != [str(c) for c in coordinates]:
            raise AggregationError(f"{where}: {sp}: the protocol's coordinates {pc} are not the space's declared {list(coordinates)}.")
    idx = _key_indices(keys, coordinates)
    scale = scale_all[idx]
    other = proto.get("sgm_scale")
    if other is None or np.asarray(other, float).shape != scale.shape or not np.array_equal(np.asarray(other, float), scale):
        raise AggregationError(f"{where}: {sp}: the protocol sgm_scale {other} is not the archive's sgm_scale over its coordinates "
                               f"{scale.tolist()}; the stored ranks were computed under another metric.")
    names = section.get("arrays") or {}
    p = _space_prefix(label)
    want = {"score": f"{p}_score", "n_references": f"{p}_n_references", "status": f"{p}_status"}
    for k, v in want.items():
        if names.get(k, v) != v:
            raise AggregationError(f"{where}: {sp}: the {k} array is named {names.get(k)!r}; {v!r} is required.")
    n = arrays["construction"].shape[0]
    required = list(want.values()) + (["sgm_outside_prior_keys"] if label == RANK_SPACE_ALL else [])
    for k in required:
        if k not in arrays or arrays[k].shape != (n,):
            raise AggregationError(f"{where}: {sp}: compatibility array {k!r} is missing or has not one entry per group ({n}).")
    score = np.asarray(arrays[want["score"]], float); nref = _int_array(arrays[want["n_references"]], want["n_references"], where)
    status = np.asarray(arrays[want["status"]]).astype(str)
    unknown = sorted(set(status.tolist()) - set(COMPATIBILITY_STATUSES))
    if unknown:
        raise AggregationError(f"{where}: {sp}: unknown compatibility statuses {unknown}.")
    constr = np.asarray(arrays["construction"]).astype(str); ws = np.asarray(arrays["window_set"]).astype(str)
    kind = np.asarray(arrays["kind"]).astype(str)
    cell = _int_array(arrays["cell"], "cell", where); chunk = _int_array(arrays["chunk"], "chunk", where)
    n_draws = _int_array(arrays["n_draws"], "n_draws", where)
    ctx_name = _ctx_name
    held = {}
    blocks = []
    for blk in section.get("blocks", []):
        bnames = blk.get("arrays") or {}
        for k in ("group_index", "ranks", "reference_draws"):
            if bnames.get(k) not in arrays:
                raise AggregationError(f"{where}: {sp}: block {blk.get('index')} names a missing array {bnames.get(k)!r}.")
        gi = _int_array(arrays[bnames["group_index"]], f"block {blk.get('index')} group_index", where)
        R = np.asarray(arrays[bnames["ranks"]], float)
        draws = _int_array(arrays[bnames["reference_draws"]], f"block {blk.get('index')} reference_draws", where)
        m = gi.size
        if m < 2 or np.unique(gi).size != m or gi.min() < 0 or gi.max() >= n:
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')}: group indices {gi.tolist()[:10]} are not at least two distinct groups of the archive.")
        if blk.get("construction") not in scopes:
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')}: construction {blk.get('construction')!r} carries no rank score "
                                   f"under comparison scope {scope!r}.")
        want_scope = scopes[str(blk["construction"])]
        if (blk.get("scope") is not None or not legacy) and blk.get("scope") != want_scope:
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')} ({blk.get('construction')}) declares reference scope "
                                   f"{blk.get('scope')!r}; the archive's comparison scope {scope!r} requires {want_scope!r}.")
        within = want_scope == SCOPE_WITHIN_CELL
        if not within and int(blk.get("cell", -1)) != -1:
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')} ({blk.get('construction')}, scope {want_scope!r}) names the cell "
                                   f"context {blk.get('cell')}; a construction-wide block spans its whole population (cell -1).")
        key = (str(blk.get("kind")), str(blk.get("window_set")), str(blk["construction"]), int(blk.get("cell", -1)) if within else -1)
        if key not in expected:
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')} names the context {ctx_name(key)}, which has no rank-scored groups in this archive.")
        if key in held:
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')}: the context {ctx_name(key)} already has block {held[key]}; a peer population "
                                   f"split into several blocks is refused (every eligible group of a context is scored against ALL its peers).")
        if sorted(gi.tolist()) != sorted(expected[key]):
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')} ({ctx_name(key)}) holds groups {sorted(gi.tolist())[:12]} but the context's complete "
                                   f"peer population is {sorted(expected[key])[:12]} ({len(expected[key])} groups): a block must hold exactly the "
                                   f"context's groups, never a subset or a foreign group.")
        held[key] = blk.get("index")
        for k in ("group_index", "ranks", "reference_draws"):
            if not str(bnames.get(k)).startswith(p + "_block_"):
                raise AggregationError(f"{where}: {sp}: block {blk.get('index')} names the array {bnames.get(k)!r}, which does not belong to this "
                                       f"space's arrays ({p}_block_*); a space never borrows another space's matrices.")
        if R.shape != (m, m) or draws.shape != (m,) or np.any(draws <= 0):
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')}: ranks {R.shape} / reference draws {draws.shape} do not match {m} groups, "
                                   f"or a reference has no draws.")
        if not np.array_equal(draws, n_draws[gi]):
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')} ({ctx_name(key)}): reference draw counts {draws.tolist()[:6]} are not the referenced "
                                   f"groups' pooled draw counts {n_draws[gi].tolist()[:6]}.")
        diag = np.diag(R); off = R[~np.eye(m, dtype=bool)]
        if not np.all(np.isnan(diag)) or not np.all(np.isfinite(off)) or off.size and (off.min() < 0 or off.max() > 1):
            raise AggregationError(f"{where}: {sp}: block {blk.get('index')}: the diagonal must be masked (NaN) and every off-diagonal rank finite in [0, 1].")
        for a, i in enumerate(gi):
            med = float(np.median(np.delete(R[a], a)))
            if status[i] != "computed" or nref[i] != m - 1 or not np.isfinite(score[i]) or abs(score[i] - med) > 1e-12:
                raise AggregationError(f"{where}: {sp}: group {int(i)} ({kind[i]}/{ws[i]}/{constr[i]} cell {cell[i]} chunk {chunk[i]}): stored score "
                                       f"{score[i]!r}/status {status[i]!r}/{nref[i]} references disagree with its block (median {med!r}, {m - 1} references).")
        blocks.append({**{k: v for k, v in blk.items() if k != "arrays"}, "group_index": gi, "ranks": R, "reference_draws": draws})
    for key, members in expected.items():
        if len(members) >= 2 and key not in held:
            raise AggregationError(f"{where}: {sp}: the context {ctx_name(key)} has {len(members)} rank-scored groups but no block; the archive is incomplete.")
        if len(members) == 1:
            i = members[0]
            if status[i] != "insufficient_peers" or np.isfinite(score[i]) or nref[i] != 0:
                raise AggregationError(f"{where}: {sp}: group {i} ({kind[i]}/{ws[i]}/{constr[i]} cell {cell[i]} chunk {chunk[i]}) is the only group of its "
                                       f"context, so its status must be 'insufficient_peers' with no score and no reference; got {status[i]!r}.")
    for i in range(n):
        if constr[i] in scopes:
            if status[i] == "not_applicable":
                raise AggregationError(f"{where}: {sp}: group {i} ({constr[i]}) cannot be 'not_applicable' under comparison scope {scope!r}.")
        else:
            if status[i] != "not_applicable" or np.isfinite(score[i]) or nref[i] != 0:
                raise AggregationError(f"{where}: {sp}: group {i} ({constr[i]}) must be 'not_applicable' with no score; got {status[i]!r}.")
    by_group = {i: {"score": None if np.isnan(score[i]) else float(score[i]), "n_references": int(nref[i]), "status": str(status[i])} for i in range(n)}
    return {"protocol": section["protocol"], "blocks": blocks, "by_group": by_group, "coordinates": [str(c) for c in coordinates], "label": str(label)}


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
    compat = archive.get("compatibility")
    spaces = {} if compat is None else (compat.get("spaces") or {RANK_SPACE_ALL: {**compat, "coordinates": [str(k) for k in archive["keys"]]}})
    ident = [{"cell": int(a["cell"][i]), "chunk": int(a["chunk"][i]), "n_windows": int(a["n_windows"][i]),
              "sgm_source": {"cell": int(a["sgm_source_cell"][i]), "chunk": int(a["sgm_source_chunk"][i]),
                             "draw_index": int(a["sgm_source_draw"][i]), "source_label": str(a["sgm_source_label"][i])},
              "sgm_method": str(a["sgm_method"][i]), "equivalent_to": str(a["equivalent_to"][i]),
              "is_sampled_rows": statistic == "sgm", "group_index": int(i),
              # the SGM's compatibility (the score describes the SGM representative, whatever statistic is selected): the
              # primary space 'all' as `compatibility`, every space of the archive by label as `compatibility_by_space`
              "compatibility": None if compat is None else dict(compat["by_group"][int(i)]),
              "compatibility_by_space": None if compat is None else {lab: dict(sp["by_group"][int(i)]) for lab, sp in spaces.items()}}
             for i in idx]
    return {"kind": kind, "window_set": window_set, "construction": construction, "statistic": statistic,
            "vectors": vecs, "identities": ident, "group_indices": idx.tolist(), "parameter_keys": [str(k) for k in archive["keys"]],
            "rank_spaces": {lab: list(sp["coordinates"]) for lab, sp in spaces.items()},
            # the reference scope the archive's scores of THIS construction were computed under (None: unscored in the archive's
            # comparison scope, or a format-1 archive), and the archive's comparison scope
            "comparison_scope": None if compat is None else (compat.get("comparison_scopes") or {}).get(construction),
            "archive_comparison_scope": None if compat is None else compat.get("comparison_scope"),
            "compatibility_protocol": None if compat is None else (compat.get("protocol") or {}).get("name"),
            "compatibility_available": compat is not None, "archive_format_version": int(archive.get("format_version", 1))}


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
    rank_spaces: dict = None               # additional rank coordinate spaces {label: (keys)} (parse_rank_spaces); None = none


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
    from .visualization_aggregation import (figure_compatibility_atlas_page, figure_compatibility_cells,
                                            figure_compatibility_population, figure_compatibility_windows, figure_joint_corner, figure_marginal_cdfs,
                                            figure_temporal, figure_window_selection)
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
    # the same validated clouds, after the groups are verified; Experiment computes the construction-wide comparison scope
    # (every group against ALL the other groups of its construction, condition and window set)
    compat = compute_compatibility(result, coll, scope=SCOPE_CONSTRUCTION)
    spaces = compute_rank_spaces(result, coll, request.rank_spaces, scope=SCOPE_CONSTRUCTION)   # additional rank coordinate spaces, same groups and references
    ref = request.reference
    covered_cells = {k: result.coverage[k]["cells"] for k in result.coverage}
    coverage_lines = []
    for k in result.coverage:
        cov = result.coverage[k]
        complete = len(cov["cells"]) == n_product_cells[k] and cov["n_windows"] == n_product_windows[k]
        coverage_lines.append(f"{condition_display(k)}: " + ("complete cohort, " if complete else "pilot subset, ")
                              + f"{len(cov['cells'])} of {n_product_cells[k]} cells of this product "
                              + (f"(cells {cov['cells']}), " if len(cov["cells"]) <= 12 else "(identities in aggregation_groups.csv), ")
                              + f"{cov['n_windows']} of {n_product_windows[k]} windows, positions {cov['chunks']}, "
                              f"{cov['n_draws_per_window']} retained draws per window.")
    agg_manifest = export_manifest(result, product_label=request.product_label, condition_labels=kinds,
                                   reference=None if ref is None else ref.get("identity"),
                                   coverage_note=request.coverage_note, product_manifest={k: manifest.get(k) for k in
                                   ("run_identity", "checkpoint_sha256", "written_at", "n_observations", "code")},
                                   n_product_cells=n_product_cells, n_product_windows=n_product_windows, target=request.target,
                                   extra={"reference_vector": None if ref is None else {"name": ref["name"], "log10": [float(v) for v in ref["log10"]],
                                                                                       "physical": [float(v) for v in ref["physical"]]},
                                          "source_clouds_note": request.source_clouds_note, "aggregation_seconds": round(elapsed, 2)},
                                   compatibility=compat, rank_spaces=spaces)
    paths = write_aggregation_exports(result, out_dir, table, manifest=agg_manifest, compatibility=compat, reference=ref, rank_spaces=spaces)
    # ---- the section ----------------------------------------------------------------------------------------
    sets_desc = "; ".join(format_window_set(n, c) for n, c in result.window_sets.items())
    src_desc = sources_description(result.sources)
    disclosed = {k: v for k, v in _disclosed_differences(result.sources).items()} if len(result.sources) <= SOURCES_LISTED_INLINE else {}
    disclosed_summary = "" if disclosed or len(result.sources) <= SOURCES_LISTED_INLINE else disclosed_differences_summary(result.sources)
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
        "**Two independent choices.** (1) Windows included: " + ", ".join(f"`{n}` = {window_set_phrase(n, c, geom, window_seconds=request.window_seconds)}"
                                                                     for n, c in result.window_sets.items())
        + " (this is all a window set selects). (2) Samples pooled before one representative is selected: within a cell (one vector "
        "per cell), across the included cells at one window position (one vector per position), across the included dataset (one "
        "vector), or separately for every cell–window pair (one vector per window). `all` describes the window selection only: under "
        "`all` the per-cell construction still returns one vector per cell. With `opening`, per-cell equals per-cell-window and "
        "per-position equals global, so overlapping curves and points in the opening figures are expected and are listed in the "
        "equivalent-groups table, not missing output.", "",
        f"**Retained clouds.** {src_desc}." + (f" {request.source_clouds_note}" if request.source_clouds_note else "")
        + f" Every supplied identity (condition, cell, chunk) is a window of this product. Input conversion (preprocessing): "
          f"{conv_status}" + (f"; recorded as `{coll.contract.get('input_conversion')}`" if coll.contract.get("input_conversion") else "") + "."
        + (" Disclosed differences between the sources (not part of the computation contract): "
           + ", ".join(f"{k} = {v}" for k, v in disclosed.items()) + "." if disclosed else "")
        + (f" {disclosed_summary}" if disclosed_summary else ""), "",
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
        "[aggregation_compatibility.csv](aggregation_compatibility.csv) (every pairwise distance rank, with its reference scope), "
        "[aggregation_manifest.json](aggregation_manifest.json). Figures: "
        + ", ".join(f"[aggregation_{kind}_{s}_{f}.png](figures/aggregation_{kind}_{s}_{f}.png)" for kind in result.coverage
                    for s in result.window_sets for f in ("marginals", "joint"))
        + "".join(f", [aggregation_{kind}_{s}_temporal.png](figures/aggregation_{kind}_{s}_temporal.png)" for kind in result.coverage
                  for s, c in result.window_sets.items() if len(c) > 1)
        + ", " + ", ".join(f"[aggregation_{kind}_window_selection.png](figures/aggregation_{kind}_window_selection.png)" for kind in result.coverage)
        + "; posterior compatibility (the subsection at the end of this section): "
        + ", ".join(f"[aggregation_{kind}_{s}_compatibility_cells.png](figures/aggregation_{kind}_{s}_compatibility_cells.png)"
                    for kind in result.coverage for s in result.window_sets if len(result.coverage[kind]["cells"]) >= 2)
        + "".join(f", [aggregation_{kind}_{s}_compatibility{f}.png](figures/aggregation_{kind}_{s}_compatibility{f}.png)"
                  for kind in result.coverage for s, c in result.window_sets.items() if len(c) > 1 for f in ("_positions", "_population", "_windows"))
        + "".join(f", [same-cell blocks of the cell–window matrix, {kind}/{s}](compatibility/{kind}/{s}/windows/index.md)"
                  for kind in result.coverage for s, c in result.window_sets.items() if len(c) > 1)
        + "".join(f"; rank space `{sp.label}` ({', '.join(sp.coordinates)}): [aggregation_compatibility_{sp.label}.csv](aggregation_compatibility_{sp.label}.csv), "
                  + ", ".join(f"[aggregation_{kind}_{s}_compatibility_cells_{sp.label}.png](figures/aggregation_{kind}_{s}_compatibility_cells_{sp.label}.png)"
                              for kind in result.coverage for s in result.window_sets if len(result.coverage[kind]["cells"]) >= 2)
                  + "".join(f", [aggregation_{kind}_{s}_compatibility{f}_{sp.label}.png](figures/aggregation_{kind}_{s}_compatibility{f}_{sp.label}.png)"
                            for kind in result.coverage for s, c in result.window_sets.items() if len(c) > 1 for f in ("_positions", "_population", "_windows"))
                  + "".join(f", [same-cell blocks of the cell–window matrix, {kind}/{s} in `{sp.label}`](compatibility/{kind}/{s}/windows_{sp.label}/index.md)"
                            for kind in result.coverage for s, c in result.window_sets.items() if len(c) > 1)
                  for sp in spaces)
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
        rows.append([f"`{c}`", display_name(c), pooled_samples(c), calc, counts, interp])
    if ref is not None:
        rows.insert(0, [f"`reference` ({ref['name']})", display_name("reference", reference_name=ref["name"]), pooled_samples("reference"),
                        "median over the per-window posterior medians of the adopted Experiment product (physical units), followed by the "
                        "documented parameter-specific selections (per-parameter record below); not an SGM of draws", "1",
                        "one fixed scenario; comparison only"])
    reporter.table(f"{SECTION_TITLE}: constructions",
                   ["ID", "visible description", "samples pooled before one SGM is selected", "calculation", "representatives per window set", "interpretation"],
                   rows, note="the canonical options; counts are per condition. Legends shorten the descriptions to "
                              + ", ".join(f"'{display_name(c, short=True)}'" for c in CONSTRUCTIONS)
                              + (f" and '{display_name('reference', short=True, reference_name=ref['name'])}'" if ref is not None else "")
                              + ", with the actual number of vectors. An SGM of window SGMs is not an active construction; no aggregate MAP exists.")
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
                    rows.append([f"{display_name(c, short=True)} (`{c}`)", key, str(summ["n"]), _fmt(qp[j, 2]), f"{_fmt(qp[j, 1])} – {_fmt(qp[j, 3])}",
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
        # figures (plan 2026-11-05: visible labels, the data-scope subtitle, captions that name what was pooled)
        if reporter.dump:
            geom = coll.contract["window_geometry"]; ws = request.window_seconds
            cov = result.coverage[kind]
            ref_label = None if ref is None else display_name("reference", short=True, reference_name=ref["name"])
            ex_key = "mu_pc" if "mu_pc" in keys else keys[0]; jx = keys.index(ex_key)
            phrases = {sn: window_set_phrase(sn, ch, geom, window_seconds=ws) for sn, ch in result.window_sets.items()}
            prior_note = ("Dotted red lines are the training-prior bounds of the estimator: training bounds, not uncertainty intervals "
                          "and not clipping thresholds.")
            for set_name, chunks in result.window_sets.items():
                phrase = phrases[set_name]
                sub = scope_line(kname, cov, n_product_cells[kind], n_product_windows[kind], set_name=set_name, chunks=chunks,
                                 geometry=geom, n_draws=coll.n_draws, window_seconds=ws)
                series = []
                for c in ("per_cell", "per_window_position", "per_cell_window", "global"):
                    gs = result.select(construction=c, window_set=set_name, kind=kind)
                    series.append({"name": display_name(c, short=True, n=len(gs)), "role": c, "vectors": group_vectors(gs, "sgm")})
                if ref is not None:
                    series.append({"name": ref_label, "role": "reference", "vectors": np.asarray(ref["log10"], float)[None, :]})
                eq = [(g, e) for g, e, k_, c_, w_ in result.equivalences if k_ == kind]
                eq_note = ""
                if eq:
                    pairs = sorted({(g.split("[")[0], e.split("[")[0]) for g, e in eq if g.endswith(f"[{set_name}]")})
                    if pairs:
                        eq_note = (" Exact equivalences under this window selection: " + "; ".join(
                            f"{display_name(a, short=True)} = {display_name(b, short=True)}" for a, b in pairs)
                            + " (the same vectors drawn twice, so their curves coincide; expected, not missing output).")
                pc = result.select(construction="per_cell", window_set=set_name, kind=kind)
                example = ""
                if len(pc) >= 2:
                    vals = sorted((float(to_physical(g.sgm_log10, table)[jx]), g.cell) for g in pc)
                    example = (f" Reading example: the '{display_name('per_cell', short=True)}' curve of {ex_key} rises in {len(pc)} equal steps, "
                               f"one at the {ex_key} coordinate of each complete cell-level SGM vector (the smallest {_fmt(vals[0][0])} for cell "
                               f"{vals[0][1]}, the largest {_fmt(vals[-1][0])} for cell {vals[-1][1]}).")
                reporter.save_figure(f"aggregation_{kind}_{set_name}_marginals",
                                     figure_marginal_cdfs(series, table, title="Aggregation methods compared: distributions of SGM parameter values", subtitle=sub),
                                     caption=f"{sub}. Each panel is one parameter in physical units; each curve is the empirical cumulative distribution "
                                             f"of one construction's SGM vectors (the fraction of its representatives at or below the x value), every "
                                             f"construction normalized separately to total mass one, so the curves assign no weights of a selected "
                                             f"nuisance mixture. The dataset vector and the fixed reference are vertical lines: a point representation, "
                                             f"whose lack of width shows nothing about parameter recovery. {prior_note}{example}{eq_note} Dispersion "
                                             f"across representatives, not posterior uncertainty.{kind_notes}")
                reporter.save_figure(f"aggregation_{kind}_{set_name}_joint",
                                     figure_joint_corner(series, table, title="Aggregation methods compared: pairs of coordinates from the same SGM vectors", subtitle=sub),
                                     caption=f"{sub}. Each point projects one complete representative vector onto a parameter pair (physical axes); the "
                                             f"dataset vector and the fixed reference are single points; the empty half of the matrix avoids showing every "
                                             f"pair twice. A trend across representatives is variation BETWEEN representatives, not the correlation "
                                             f"inside an individual window posterior and not a within-window compensation law; nothing is fitted and "
                                             f"no relation is imposed. {prior_note}{eq_note}{kind_notes}")
                if len(chunks) > 1:
                    cells = cov["cells"]
                    pcw = {c: np.asarray([g.sgm_log10 for g in result.select(construction="per_cell_window", window_set=set_name, kind=kind) if g.cell == c]) for c in cells}
                    pw = np.asarray([g.sgm_log10 for g in result.select(construction="per_window_position", window_set=set_name, kind=kind)])
                    bands = np.median(np.asarray([[g.quantiles_log10 for g in result.select(construction="per_cell_window", window_set=set_name, kind=kind) if g.cell == c]
                                                  for c in cells]), axis=0)
                    step_s = (float(ws) * geom["step_frames"] / geom["n_frames"]) if ws else 1.0
                    times = np.asarray(chunks, float) * step_s
                    last = f" The last point at {times[-1]:g} s represents the frames from {times[-1]:g} to {times[-1] + float(ws):g} s." if ws else ""
                    reporter.save_figure(f"aggregation_{kind}_{set_name}_temporal",
                                         figure_temporal(pcw, pw, result.select(construction="global", window_set=set_name, kind=kind)[0].sgm_log10,
                                                         None if ref is None else {"name": ref_label, "log10": ref["log10"]}, table, times=times,
                                                         window_bands=bands, window_seconds=ws, subtitle=sub,
                                                         title=f"SGM estimates across the {NUMBER_WORDS.get(len(chunks), len(chunks))} {f'{float(ws):g} s ' if ws else ''}windows"),
                                         caption=f"{sub}. x = start of each {f'{float(ws):g} s ' if ws else ''}window within the recording.{last} Thin lines: "
                                                 f"'{display_name('per_cell_window', short=True)}', one line per cell ({len(cells)} cells"
                                                 + ("; identified in the legend" if len(cells) <= 6 else "; identities in the report inventory") + "). "
                                                 f"Thick line: '{display_name('per_window_position', short=True)}', calculated from the joint samples pooled "
                                                 f"across cells at each position: it is not an average of the thin lines and need not lie between their "
                                                 f"values in every panel. Horizontal lines: '{display_name('global', short=True)}' and "
                                                 f"'{ref_label or 'no reference'}'. Gray shading: the source within-window 50 % and 90 % marginal intervals, "
                                                 f"summarized by the median endpoint across cells: a summary of the input posteriors, not a confidence band "
                                                 f"for the thick line and not a range containing 90 % of the cells. Tightly grouped SGM centers can coexist "
                                                 f"with broad source intervals (bleaching shows this); representative dispersion is never posterior "
                                                 f"precision, and a change of a fitted imaging coordinate along the recording is assigned no physical "
                                                 f"cause here. {prior_note}{kind_notes}")
            sets = list(result.window_sets)
            per_cell_by_set = {sn: {g.cell: g.sgm_log10 for g in result.select(construction="per_cell", window_set=sn, kind=kind)} for sn in sets}
            global_by_set = {sn: result.select(construction="global", window_set=sn, kind=kind)[0].sgm_log10 for sn in sets}
            sel_example = ""
            if len(sets) >= 2:
                a, b = sets[0], sets[1]
                common = [c for c in per_cell_by_set[a] if c in per_cell_by_set[b]]
                if len(common) >= 2:
                    diffs = sorted((abs(float(per_cell_by_set[b][c][jx]) - float(per_cell_by_set[a][c][jx])), c) for c in common)
                    pick = [diffs[-1][1], diffs[0][1]]
                    sel_example = (" Examples (descriptive, not a rule): " + "; ".join(
                        f"cell {c}: {ex_key} {_fmt(to_physical(per_cell_by_set[b][c], table)[jx])} ({phrases[b]}) against "
                        f"{_fmt(to_physical(per_cell_by_set[a][c], table)[jx])} ({phrases[a]})" for c in pick)
                        + ". The effect of the window selection is recording-dependent.")
            sub_sel = scope_line(kname, cov, n_product_cells[kind], n_product_windows[kind], set_name=sets[0], chunks=result.window_sets[sets[0]],
                                 geometry=geom, n_draws=coll.n_draws, window_seconds=ws).split(" · windows included")[0] \
                + " · windows included: " + " versus ".join(phrases[sn] for sn in sets) + f" · {int(coll.n_draws):,} joint draws/window · representative: SGM"
            reporter.save_figure(f"aggregation_{kind}_window_selection",
                                 figure_window_selection(per_cell_by_set, table, reference=None if ref is None else {"name": ref_label, "log10": ref["log10"]},
                                                         global_by_set=global_by_set, set_labels=phrases, subtitle=sub_sel,
                                                         title="One SGM per cell: " + " versus ".join(phrases[sn] for sn in sets)),
                                 caption=f"{sub_sel}. x = cell identity. Each cell carries one marker per window selection (the per-cell construction's blue: "
                                         f"filled for '{phrases[sets[0]]}'" + (f", hollow for '{phrases[sets[1]]}'" if len(sets) > 1 else "")
                                         + "), joined by a thin connector that represents a change of window selection, not elapsed time and not an "
                                         f"uncertainty interval. Orange lines: '{display_name('global', short=True)}' of each selection (one line style per "
                                         f"selection); purple dashed: '{ref_label or 'no reference'}'. Blue never means the dataset vector and orange never "
                                         f"means a per-cell vector.{sel_example} {prior_note}{kind_notes}")
    # ---- posterior compatibility of the SGM representatives (raw median ranks; no sampling probability here) ------
    _report_compatibility(reporter, result, compat, coll, request, out_dir, condition_display, n_product_cells, n_product_windows,
                          kind_notes, figure_compatibility_cells, figure_compatibility_windows, figure_compatibility_atlas_page,
                          figure_compatibility_population, rank_spaces=spaces)
    try:
        back = load_candidate_archive(paths["archive"])
        readback = (f"format {back['format_version']}, comparison scope {back['compatibility']['comparison_scope']}, "
                    f"{len(back['compatibility']['blocks'])} rank blocks, {len(result.groups)} groups re-validated")
        ok = True
    except AggregationError as exc:
        readback, ok = str(exc), False
    reporter.check("aggregation_archive_reads_back", ok, readback,
                   note="the written candidate archive is read back through the constructor's reader, which re-validates the compatibility "
                        "blocks (dimensions, identities, masked diagonals, rank bounds, stored scores against the row medians).")
    agg_manifest["exports"] = {k: str(v) for k, v in paths.items()}
    return agg_manifest


COMPATIBILITY_TITLE = f"{SECTION_TITLE}: posterior compatibility of SGM representatives"
ATLAS_CELLS_PER_PAGE = 12


def _report_compatibility(reporter, result, compat, coll, request, out_dir, condition_display, n_product_cells, n_product_windows,
                          kind_notes, fig_cells, fig_windows, fig_atlas, fig_population, rank_spaces=()):
    """The compact subsection under the construction-wide comparison scope: the definition, the aggregation / reference
    population / normalization statement, one scope and availability table (matrices, candidates, references, computed or
    unavailable, zero scores labeled as centrality outcomes; one row set per rank coordinate space), and per condition,
    window set and space: the cell matrix, the window-position matrix, the complete cell–window population matrix (one
    N x N matrix, never a set of separate within-cell matrices), the cell-by-window score overview, and the linked atlas
    of the population matrix's same-cell blocks."""
    if compat.scope != SCOPE_CONSTRUCTION or any(sp.scope != SCOPE_CONSTRUCTION for sp in rank_spaces):
        raise AggregationError(f"the Experiment report presents the {SCOPE_CONSTRUCTION!r} comparison scope; got "
                               f"{[compat.scope] + [sp.scope for sp in rank_spaces]}.")
    geom = coll.contract["window_geometry"]; ws = request.window_seconds
    step_s = (float(ws) * geom["step_frames"] / geom["n_frames"]) if ws else 1.0
    groups = result.groups
    all_spaces = [compat] + list(rank_spaces)
    refs = REFERENCE_POPULATIONS[SCOPE_CONSTRUCTION]
    space_text = ("**Rank coordinate spaces.** The primary space `all` runs the distances over every parameter coordinate "
                  f"({', '.join(str(k) for k in result.keys)}). " + (
                  "Additional spaces restrict the same candidates, reference SGMs and reference clouds to a declared subset of the "
                  "coordinates (the scale is the prior widths of those coordinates); the groups, the SGMs and the reference populations "
                  "are identical across spaces, only the metric changes, so a space whose coordinates exclude a parameter is blind to "
                  "disagreement in that parameter: " + "; ".join(f"`{sp.label}` = {', '.join(sp.coordinates)}" for sp in rank_spaces)
                  + ". Each space has its own scores, statuses, figures, atlas and CSV (`aggregation_compatibility_<space>.csv`); the "
                  "Nuisance DLI constructor names the space it weights with (`rank_coordinates`). The coordinate space and the "
                  "comparison scope are separate choices." if rank_spaces else
                  "No additional space was requested for this report (`--compatibility-coordinates`)."))
    text = [
        "For every rank-scored representative, each other group of its reference population is a reference posterior: that "
        "group's retained cloud with the group's own existing SGM. The distance rank R_ij of candidate SGM v_i against reference "
        "j is the share of reference j's draws that lie strictly farther from v_j than v_i does, with distances in estimator "
        f"coordinates divided by the prior widths {result.scale.tolist()} (log10 coordinates are not transformed again). Rows are "
        "candidates, columns references; the self comparison is excluded by group identity (never by vector equality); the matrix "
        "is generally asymmetric because every reference has its own spread. The candidate's score is the ordinary median of its "
        "row over the other references (two middle ranks averaged when their number is even), a raw score that is not normalized "
        "across candidates. A rank near one means the candidate sits closer to the reference's SGM than most of that reference's "
        "draws; near zero, almost every reference draw is closer. The rank describes radial centrality, not posterior density, "
        "not membership in a joint credible region and not the validity of a recording.", "",
        f"**Comparison scope: `{SCOPE_CONSTRUCTION}`.** The aggregation fixes which representatives exist; the comparison scope "
        "fixes whom each representative is compared with; the two are separate. In this report every representative of one "
        "aggregation is scored against ALL the other representatives of that same aggregation, within one condition and one "
        "window set, and the candidate's own group is the only one excluded. "
        f"One SGM per cell: {refs['per_cell']}. One SGM per window position: {refs['per_window_position']}. "
        f"One SGM per cell–window: {refs['per_cell_window']}. One SGM for the included dataset: {refs['global']}. "
        "For one SGM per cell–window the other windows of the candidate's own cell are therefore a small part of its references "
        "(for example nine of 599 with 60 cells and ten windows), with no separate treatment and no mass reserved for the cell. "
        "Under a single-window set such as `opening`, the cell–window groups and the per-cell groups are the same windows with "
        "the same clouds: both are scored on their own and the scores agree; a single window position has no peer position "
        "(`insufficient_peers`).", "",
        "**Normalization happens in the Nuisance DLI constructor, not here.** Experiment reports raw median ranks s_i only. Under "
        "the member policy `posterior_rank_median` with `rank_scope = \"construction\"`, the constructor assigns the N "
        "representatives of the selected aggregation, condition and window set the probabilities p_i = alpha * s_i / (s_1 + ... + "
        "s_N), where alpha is the component's explicit mass (one for a single-component selection). No cell receives a reserved "
        "mass: a cell's total is the sum of its representatives' probabilities and may be small or zero. The earlier hierarchical "
        f"policy (`rank_scope = \"{SCOPE_WITHIN_CELL}\"`: each cell–window SGM scored against the other windows of its own cell "
        "only, every cell given equal mass) is a different selection target; it is read from archive formats 2 and 3 under "
        "that name and is never substituted for the scope of this report.", "",
        "**Zeros and availability.** A median compatibility score of exactly zero means that a majority of the candidate's reference "
        "ranks are zero (at least half of the other references place the candidate beyond all of their draws); some references may "
        "still assign it a high rank. It is a valid centrality outcome of a valid recording, distinct from an unavailable score "
        "(`insufficient_peers`: a population of one group, no score, zero references), and it is not a data-quality classification: "
        "nothing is removed, excluded or curated on its account. Training-prior flags stay separate (`sgm_outside_prior_keys` "
        "in aggregation_groups.csv: whether the SGM itself lies outside the training prior, a different quantity from the "
        "outside-box fractions of the pooled draws).", "",
        f"**Files.** Scores, statuses and reference counts per group: [aggregation_groups.csv](aggregation_groups.csv); every pairwise "
        f"rank with its reference scope: [aggregation_compatibility.csv](aggregation_compatibility.csv); the rank-matrix blocks, the "
        f"protocol `{COMPATIBILITY_PROTOCOL}` and the comparison scope: [aggregation_candidates.npz](aggregation_candidates.npz) (archive "
        f"format {ARCHIVE_FORMAT_VERSION}, manifest section `compatibility`). Computation took {compat.seconds:.1f} s on the CPU. The "
        "figures (captions at the end of the report) show raw median ranks.", "",
        space_text,
    ]
    reporter.prose(COMPATIBILITY_TITLE, "\n".join(text))

    def zero_label(i, c):
        g = groups[i]
        if c == "per_cell":
            return f"cell {g.cell}"
        if c == "per_window_position":
            return f"window start {g.chunk * step_s:g} s"
        return f"cell {g.cell} window start {g.chunk * step_s:g} s"

    rows = []
    for sp in all_spaces:
      for kind in result.coverage:
        kname = condition_display(kind)
        for set_name, chunks in result.window_sets.items():
            for c in CONSTRUCTIONS:
                idx = [i for i, g in enumerate(groups) if g.kind == kind and g.window_set == set_name and g.construction == c]
                if c not in SCOPES_BY_CONSTRUCTION[SCOPE_CONSTRUCTION]:
                    rows.append([f"`{sp.label}`", kname, set_name, f"{display_name(c, short=True)} (`{c}`)", "-", "0", str(len(idx)), "-",
                                 "not applicable (no same-level peer; uniform selection remains available)", "-"])
                    continue
                st = sorted(set(sp.status[i] for i in idx))
                n_mat = 1 if len(idx) >= 2 else 0
                zero_ids = [i for i in idx if sp.status[i] == "computed" and sp.score[i] == 0.0]
                avail = "computed" if st == ["computed"] else f"unavailable: a population of {len(idx)} group has no peer (insufficient_peers)"
                if n_mat:
                    zl = ", ".join(zero_label(i, c) for i in zero_ids[:12])
                    zeros = f"{len(zero_ids)}" + (f" ({zl}{'; …' if len(zero_ids) > 12 else ''})" if zero_ids else "")
                else:
                    zeros = "-"
                rows.append([f"`{sp.label}`", kname, set_name, f"{display_name(c, short=True)} (`{c}`)", SCOPE_CONSTRUCTION, str(n_mat), str(len(idx)),
                             str(len(idx) - 1) if n_mat else "-", avail, zeros])
    reporter.table(f"{COMPATIBILITY_TITLE}: aggregation, comparison scope and availability",
                   ["rank space", "condition", "window set", "aggregation", "comparison scope", "matrices", "candidates",
                    "references per candidate", "status", "zero scores (centrality outcomes, not defective recordings)"], rows,
                   note="a matrix is one block of candidates against references: the complete population of the aggregation in that "
                        "condition and window set (the included cells; the selected window positions; every cell–window group), so "
                        "'references per candidate' is the population size minus the candidate itself; 'unavailable' keeps the groups with "
                        "no score and zero references, never a fabricated 0 or 1; rank spaces: "
                        + "; ".join(f"`{sp.label}` = {', '.join(sp.coordinates)}" for sp in all_spaces) + f".{kind_notes}")
    if not reporter.dump:
        return
    n_draws = coll.n_draws
    for sp in all_spaces:
      compat = sp
      sfx = "" if sp.label == RANK_SPACE_ALL else f"_{sp.label}"
      tag = f" [rank space {sp.label}]" if sfx else ""
      space_note = "" if sp.label == RANK_SPACE_ALL else f" Rank space `{sp.label}`: distances over {', '.join(sp.coordinates)} only."
      by = {(b.kind, b.window_set, b.construction): b for b in compat.blocks}
      for kind in result.coverage:
        kname = condition_display(kind); cells = [int(c) for c in result.coverage[kind]["cells"]]; cov = result.coverage[kind]
        for set_name, chunks in result.window_sets.items():
            chunks = [int(w) for w in chunks]
            sub = scope_line(kname, cov, n_product_cells[kind], n_product_windows[kind], set_name=set_name, chunks=chunks,
                             geometry=geom, n_draws=n_draws, window_seconds=ws)
            phrase = window_set_phrase(set_name, chunks, geom, window_seconds=ws)
            times = [w * step_s for w in chunks]
            # ---- one SGM per cell: the cell-to-cell matrix -----------------------------------------------------
            blk = by.get((kind, set_name, "per_cell"))
            if blk is not None:
                order = np.argsort([groups[int(i)].cell for i in blk.group_index], kind="stable")
                ids = [groups[int(blk.group_index[o])].cell for o in order]
                R = blk.ranks[np.ix_(order, order)]; S = compat.score[blk.group_index[order]]
                n_zero = int(np.sum(S == 0))
                reporter.save_figure(f"aggregation_{kind}_{set_name}_compatibility_cells{sfx}",
                                     fig_cells(R, ids, S, title="One SGM vector per cell: posterior compatibility" + tag, subtitle=sub),
                                     caption=f"{sub}. Rows: candidate cell; columns: reference cell (stable numeric order, every cell retained). "
                                             f"Comparison scope: every cell against the other {len(ids) - 1} included cells. "
                                             f"Each reference is the other cell's retained cloud, the equal-window mixture of its selected windows "
                                             f"({len(chunks)} window{'s' if len(chunks) != 1 else ''} x {n_draws:,} draws), with that cell's own SGM; color = the "
                                             f"share of the reference's draws strictly farther from its SGM than the candidate SGM, distances in estimator "
                                             f"coordinates divided by the prior widths; the self comparison is gray (excluded). The matrix is asymmetric because "
                                             f"each reference has its own spread. Right strip: the row median over the other cells, the raw score on the same 0–1 "
                                             f"scale, not a sampling probability" + (f"; {n_zero} cell(s) score exactly zero (a centrality outcome, not a data "
                                             f"quality verdict)" if n_zero else "") + f".{space_note}{kind_notes}")
            else:
                reporter.prose(f"{COMPATIBILITY_TITLE} [{kname}, window set {set_name}, rank space {sp.label}]: cell matrix",
                               "Not computed: fewer than two included cells, so no cell has a reference cell (insufficient_peers).")
            # ---- one SGM per window position: the position-to-position matrix --------------------------------------
            blk = by.get((kind, set_name, "per_window_position"))
            if blk is not None:
                order = np.argsort([groups[int(i)].chunk for i in blk.group_index], kind="stable")
                ids = [f"{groups[int(blk.group_index[o])].chunk * step_s:g}" for o in order]
                R = blk.ranks[np.ix_(order, order)]; S = compat.score[blk.group_index[order]]
                n_zero = int(np.sum(S == 0))
                reporter.save_figure(f"aggregation_{kind}_{set_name}_compatibility_positions{sfx}",
                                     fig_cells(R, ids, S, title="One SGM vector per window position: posterior compatibility" + tag, subtitle=sub,
                                               entity="window position (start, s)", entities="window positions"),
                                     caption=f"{sub}. Rows: candidate window position; columns: reference window position (labeled by window start in "
                                             f"seconds). Comparison scope: every position against the other {len(ids) - 1} selected positions. Each "
                                             f"reference is the other position's retained cloud, the equal-cell mixture of that window over the included "
                                             f"cells ({len(cells)} cell{'s' if len(cells) != 1 else ''} x {n_draws:,} draws), with that position's own SGM; "
                                             f"color and strip as in the cell matrix. The row median is a raw score, not a sampling probability"
                                             + (f"; {n_zero} position(s) score exactly zero (a centrality outcome)" if n_zero else "")
                                             + f".{space_note}{kind_notes}")
            else:
                reporter.prose(f"{COMPATIBILITY_TITLE} [{kname}, window set {set_name}, rank space {sp.label}]: window-position matrix",
                               f"Not computed: one selected window position ({phrase}) has no peer position (insufficient_peers); the single "
                               "representative remains selectable with uniform probability.")
            # ---- one SGM per cell–window: ONE population matrix, its score overview and its same-cell blocks -----
            blk = by.get((kind, set_name, "per_cell_window"))
            if blk is not None and len(chunks) > 1:
                pos = {(groups[int(i)].cell, groups[int(i)].chunk): k for k, i in enumerate(blk.group_index)}
                order = np.asarray([pos[(c, w)] for c in cells for w in chunks])
                R = blk.ranks[np.ix_(order, order)]; S = compat.score[blk.group_index[order]]
                N = order.size; W = len(chunks); n_zero = int(np.sum(S == 0))
                reporter.save_figure(f"aggregation_{kind}_{set_name}_compatibility_population{sfx}",
                                     fig_population(R, cells, W, S, title=f"One SGM vector per cell–window: the complete population matrix "
                                                    f"({N} x {N})" + tag, subtitle=sub),
                                     caption=f"{sub}. ONE matrix over the complete cell–window population of this condition and window set: {N} "
                                             f"candidates (rows) against the same {N} groups as references (columns), ordered by cell in stable numeric "
                                             f"order and, within a cell, by window start; thin lines mark the cell boundaries and the labels name the "
                                             f"cells. Comparison scope: every cell–window SGM against the other {N - 1} cell–window groups, each "
                                             f"reference one window's own cloud ({n_draws:,} draws) with its own SGM. The {W} x {W} blocks on the "
                                             f"diagonal are the comparisons between windows of the same cell: {W - 1} of each candidate's {N - 1} "
                                             f"references, part of this one population and not separate matrices. Color = the share of the reference's "
                                             f"draws strictly farther from its SGM than the candidate SGM; the self comparison is gray (excluded). Right "
                                             f"strip: the row median over the other {N - 1} groups, the raw score, not a sampling probability"
                                             + (f"; {n_zero} group(s) score exactly zero (a centrality outcome, not a data-quality verdict)" if n_zero else "")
                                             + f".{space_note}{kind_notes}")
                Sg = S.reshape(len(cells), W)
                reporter.save_figure(f"aggregation_{kind}_{set_name}_compatibility_windows{sfx}",
                                     fig_windows(Sg, cells, times, title="One SGM vector per cell–window: median compatibility ranks over the complete "
                                                 "population" + tag, subtitle=sub, window_seconds=ws),
                                     caption=f"{sub}. Rows: cells in stable order; columns: the selected window start times. Color = the median "
                                             f"compatibility rank of the cell–window SGM over the other {N - 1} cell–window groups of the condition and "
                                             f"window set (the same scores as the strip of the population matrix, arranged by cell and window); the other "
                                             f"windows of the same cell are {W - 1} of those references. Zeros stay visible as the darkest color and are "
                                             f"centrality outcomes, not data failures" + (f" ({n_zero} here)" if n_zero else "")
                                             + f". These are raw scores: the constructor's probabilities are these scores divided by their sum over the "
                                             f"whole grid (times the component mass), with no mass reserved per cell. The same-cell blocks of the matrix "
                                             f"are enlarged in the linked atlas (compatibility/{kind}/{set_name}/windows{sfx}/index.md)."
                                             f"{space_note}{kind_notes}")
                panels = [(c, R[a * W:(a + 1) * W, a * W:(a + 1) * W]) for a, c in enumerate(cells)]
                _write_atlas(out_dir, kind, kname, set_name, phrase, sub, panels, {c: Sg[a] for a, c in enumerate(cells)}, times, compat, fig_atlas,
                             folder_name=f"windows{sfx}", space_note=space_note, n_groups=N)
            elif blk is not None:
                reporter.prose(f"{COMPATIBILITY_TITLE} [{kname}, window set {set_name}, rank space {sp.label}]: cell–window population",
                               f"Computed without a separate figure: with one selected window ({phrase}) the cell–window groups are the same "
                               "windows, with the same clouds, as the per-cell groups, so their population matrix and scores equal the cell "
                               "matrix above (both are computed on their own; the scores are in aggregation_groups.csv).")
            else:
                reporter.prose(f"{COMPATIBILITY_TITLE} [{kname}, window set {set_name}, rank space {sp.label}]: cell–window population",
                               "Not computed: one cell–window group has no peer (insufficient_peers).")


def _write_atlas(out_dir, kind, kname, set_name, phrase, sub, panels, cell_scores, times, compat, fig_atlas,
                 folder_name="windows", space_note="", n_groups=0):
    """The atlas of the cell–window population matrix's same-cell blocks: paged contact sheets (``ATLAS_CELLS_PER_PAGE``
    cells per page, 3 x 4) under ``compatibility/<kind>/<set>/<folder_name>/`` (``windows`` for the primary rank space,
    ``windows_<label>`` otherwise) with an index mapping every cell to its page. ``panels`` are ``[(cell, (W, W) block)]``
    in cell order; ``cell_scores`` the cell's construction-wide scores by window."""
    folder = Path(out_dir) / "compatibility" / str(kind) / str(set_name) / str(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    sfx = "" if compat.label == RANK_SPACE_ALL else f"_{compat.label}"
    score_col = "compatibility_score" + sfx; pairs_csv = f"aggregation_compatibility{sfx}.csv"; block_arrays = f"{_space_prefix(compat.label)}_block_*"
    W = len(times)
    pages = [panels[i:i + ATLAS_CELLS_PER_PAGE] for i in range(0, len(panels), ATLAS_CELLS_PER_PAGE)]
    lines = [f"# Same-cell blocks of the cell–window population matrix: {kname}, windows included: {phrase}" + (f", rank space `{compat.label}`" if sfx else ""), "", sub, "",
             f"Each panel enlarges one {W} x {W} diagonal block of the complete {n_groups} x {n_groups} cell–window matrix: the "
             "distance ranks between the windows of ONE cell (rows candidate windows, columns reference windows, both labeled by "
             "window start time in seconds); color is the share of the reference window's draws strictly farther from its own SGM "
             "than the candidate window's SGM (0–1, one scale for every panel); the self comparison is gray (excluded). These "
             f"same-cell comparisons are {W - 1} of each candidate's {n_groups - 1} references under the construction-wide scope; "
             "they are shown for legibility and are NOT the candidate's score, which is the median of its whole row of the "
             f"population matrix (the `{score_col}` column of [aggregation_groups.csv](../../../../aggregation_groups.csv), listed "
             f"per cell below). Every pairwise value is in [{pairs_csv}](../../../../{pairs_csv}); the full matrix is the "
             f"`{block_arrays}` arrays of [aggregation_candidates.npz](../../../../aggregation_candidates.npz). A zero is a "
             "centrality outcome, not a data-quality classification; raw-data notes and training-prior flags stay separate." + space_note
             + " Back to the [report](../../../../report.md).", "",
             f"| cell | page | scores over the complete population ({n_groups - 1} references) by window start (s) |", "|---|---|---|"]
    for p, page in enumerate(pages, start=1):
        for c, _ in page:
            sc = ", ".join(f"{t:g}: {s:.3f}" for t, s in zip(times, cell_scores[c]))
            lines.append(f"| {c} | [page_{p:02d}.png](page_{p:02d}.png) | {sc} |")
    lines += ["", "## Pages", ""]
    for p, page in enumerate(pages, start=1):
        first, last = page[0][0], page[-1][0]
        fig = fig_atlas(page, times, title=f"Same-cell blocks of the cell–window matrix, page {p} of {len(pages)}: cells {first} to {last}"
                        + (f" [rank space {compat.label}]" if sfx else ""), subtitle=sub)
        fig.savefig(str(folder / f"page_{p:02d}.png"), dpi=150)
        lines += [f"### Page {p}: cells " + ", ".join(str(c) for c, _ in page), "", f"![page_{p:02d}](page_{p:02d}.png)", ""]
    (folder / "index.md").write_text("\n".join(lines), encoding="utf-8")
    return folder / "index.md"
