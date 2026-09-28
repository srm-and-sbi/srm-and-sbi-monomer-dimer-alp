"""Shared engine for the posterior-predictive video comparison (biology and detector workflows).

The check both workflows want is the same: take the parameters INFERRED from one real recording,
simulate a video with them, and put the two side by side. If the synthetic frames do not look like
the recording that produced the parameters, the posterior is explaining the data with a
configuration the forward model cannot actually render -- a failure no summary statistic on
held-out synthetic data can reveal, because it only appears when the model is pointed at reality.

The two workflows invert which half of the model the MAP supplies and which half is held fixed:

* **detector** -- the MAP supplies the six IMAGING parameters; the reaction-diffusion block is a
  marginalized nuisance, drawn from the biology prior (or pinned) per render, and the system is
  built with its full reaction network -- the same simulator the detector was calibrated against.
* **biology** -- the MAP supplies the eleven REACTION-DIFFUSION parameters and the system is built
  with the recording's condition's reaction network (association is a per-condition constant); the imaging block is held fixed at the calibrated vector the
  training videos were generated with, read from the ``Nuisance_DLI`` artifact at run time.

In both cases the five SCOPE camera parameters are pinned to their MET values rather than drawn:
the comparison is against one specific real acquisition, so a random camera draw would add
variation that has nothing to do with the question. The pinned values are the known acquisition
values and lie inside the SCOPE box the DLI stage draws from.

The render reproduces the production observation layer. The labeling goes through the one path
every renderer of simulated trajectories shares (``labeling.resolve_labeling`` and
``labeling.label_trajectory``): the condition's law and its declared (MET-INLB) or derived (MET-FAB)
probe occupancy, applied by initial molecular species, with ``--labeling-law`` and ``--occupancy``
overriding them exactly as at the DLI stage; the clip records the labeling row the draw produced.
Instead of the MAP, the reaction-diffusion block can come from a DECLARED configuration
(``--declared-rds``, both workflows): a file stating each of the eleven values with its basis and
source. It serves prediction checks before a biology estimator exists, and any render whose
biology must be an explicit assumption rather than an estimate or a prior draw.

The renderer itself is already common to both workflows (``simulation_dli_support.render_dli_video``,
re-exported as ``render_detector_video``), and the 11-key imaging order it consumes is a property of
that renderer rather than of either workflow -- which is why ``detector_parameterization`` is read
here for the imaging CONTRACT even on the biology path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import __version__ as _PACKAGE_VERSION
from . import artifact_schema as schema
from matplotlib.figure import Figure

from . import detector_parameterization as det
from . import parameterization as bio
from .parameterization import PARAMETERS, RunTiming
from .sample_geometric_median import sample_geometric_median
from .simulation_dli_support import render_dli_video
from .simulation_rds_support import (
    build_simulation, build_system, collapse_species_axis, extract_subunit_lineage,
    extract_trajectory_poses,
)
from .workflow import parameter_keys as _wf_keys, parameter_table

# Conditions are named scientifically wherever a reader sees them; the tokens below survive only as
# the stored ``kinds`` field of the MAP database and the recording filenames on disk.
# Condition naming (stored token <-> scientific name) has ONE definition, in experiment_support.
from .experiment_support import KIND_OF_CONDITION, inspect_recording, read_recording
from .labeling import LABELING_SET_COLUMNS, label_trajectory, labeling_rng, resolve_labeling

def _clip_span_token(n_frames, frame_time):
    """The clip's own duration as a label token (e.g. 1000 frames @ 0.02 s -> ``20S``); the
    canonical duration labeling, so the token matches the rest of the codebase."""
    return RunTiming(total_time_seconds=n_frames * frame_time,
                     frames=PARAMETERS.simulation.timing).label.split("_")[0]


def _build_stem(project_alias, map_label, kind, cell, chunk, map_source, clip_token,
                fixed_imaging=False, fixed_nuisance=False, run_label=None, declared_rds=False,
                map_block="imaging"):
    """Output basename: canonical ``{alias}_{model_window}`` + imaging descriptor + clip span.
    ``map_label`` (e.g. ``2S_50FPS``) is the estimator's model window; ``clip_token`` (e.g.
    ``20S``) is the rendered length. Chunk source names the concrete entry
    (``..._Cell_{cell}_Chunk_{chunk}_...``); cell-median drops the chunk and marks ``Median``;
    ``fixed_imaging`` marks a correct-source fixed-parameter render (no MAP database);
    ``fixed_nuisance`` marks a pinned RDS nuisance (counts/diffusivities held, not drawn); and
    ``run_label`` appends an optional caller tag at the END of the name so renders that share a
    selection (same cell, both fixed-imaging + fixed-nuisance) stay distinct instead of
    overwriting each other. ``declared_rds`` marks a reaction-diffusion block read from a declared
    configuration (``--declared-rds``): on the biology path no MAP is read and the descriptor is
    ``Declared_RDS``; on the detector path the token replaces ``_Fixed_Nuisance``."""
    if declared_rds and map_block == "imaging":
        nuis_tok = "_Declared_RDS"
    else:
        nuis_tok = "_Fixed_Nuisance" if fixed_nuisance else ""
    label_tok = ""
    if run_label:
        safe = "".join(c if c.isalnum() else "_" for c in str(run_label)).strip("_")
        label_tok = f"_{safe}" if safe else ""
    if declared_rds and map_block == "rds":
        return f"{project_alias}_{map_label}_Declared_RDS_{kind}_Cell_{cell}_{clip_token}{label_tok}"
    if fixed_imaging:
        return f"{project_alias}_{map_label}_Fixed_Imaging{nuis_tok}_{kind}_Cell_{cell}_{clip_token}{label_tok}"
    descriptor = {"cell-median": "MAP_Estimate_Median",
                  "cell-sgm": "MAP_Estimate_SGM"}.get(map_source, "MAP_Estimate")
    chunk_part = "" if map_source.startswith("cell-") else f"_Chunk_{chunk}"
    return (f"{project_alias}_{map_label}_{descriptor}{nuis_tok}_{kind}_Cell_{cell}"
            f"{chunk_part}_{clip_token}{label_tok}")


def synthetic_source_label(map_block, declared_biology=False, fixed_imaging=False):
    """What the synthetic render is made from, in the words the clip file (``synth_label``), the
    comparison figure and the viewer notebook all show."""
    if map_block == "rds":
        return ("SYNTH (declared reaction-diffusion)" if declared_biology
                else "SYNTH (MAP reaction-diffusion)")
    return "SYNTH (fixed imaging)" if fixed_imaging else "SYNTH (MAP imaging)"


def render_output_paths(out_dir, stem):
    """The three files one render writes, named before any work is done."""
    out_dir = Path(out_dir)
    return {"clip": out_dir / f"{stem}_Synthetic_Video.npz",
            "figure": out_dir / f"{stem}_Comparison.png",
            "trajectory": out_dir / f"{stem}_Trajectory.h5"}


def refuse_existing_outputs(paths):
    """Refuse, before any work, to write over an existing render. The stem carries neither the
    nuisance tag nor the declared configuration's identity, so another attempt or a variant of the
    same recording needs its own ``--run-label``."""
    existing = [str(p) for p in paths.values() if Path(p).exists()]
    if existing:
        raise SystemExit("refusing to overwrite an existing render:\n    " + "\n    ".join(existing)
                         + "\nAnother attempt or a variant (another --nuisance-tag, --declared-rds file "
                           "or --set-rds override, or a repeat) needs its own --run-label.")


def _load_map_theta(map_npz, keys, table, kind, cell, chunk, source, prior_low, prior_high):
    """Physical MAP theta for (kind, cell): a single chunk's MAP (``source='chunk'``) or the
    per-cell aggregate over all of that cell's chunk MAPs (``source='cell-sgm'`` /
    ``'cell-median'``). ``table`` is the workflow's parameter table: the stored MAP rows are in
    estimator space and are mapped to physical values through the ONE conversion rule
    (``parameterization.to_physical``), never by a blanket ``10 ** theta``. Fails with guidance
    listing the available cells/chunks."""
    arrays, manifest = schema.load_product(map_npz, stage="experiment")
    schema.assert_parameter_keys(manifest, keys, source=str(map_npz))       # keys AND order
    map_estimate = np.asarray(arrays["map_estimate"], dtype=float)
    kind_index = np.asarray(arrays["kind_index"])
    cells = np.asarray(arrays["cell"])
    chunks = np.asarray(arrays["chunk"])
    kinds = [str(k) for k in arrays["kinds"]]
    if kind not in kinds:
        raise ValueError(f"kind={kind!r} not in the MAP database; available: {kinds}.")
    ki = kinds.index(kind)
    cell_rows = np.where((kind_index == ki) & (cells == cell))[0]
    if cell_rows.size == 0:
        avail_cells = sorted({int(c) for c in cells[kind_index == ki]})
        raise ValueError(f"no MAP entries for kind={kind} cell={cell} in\n    {map_npz}\n"
                         f"available cells for {kind}: {avail_cells}")
    if source.startswith("cell-"):
        theta_log10 = _aggregate_cell(map_estimate[cell_rows], source, table)
        how = ("Sample Geometric Median (a real chunk's estimate)" if source == "cell-sgm"
               else "per-dimension median (a composite, correlations discarded)")
        print(f"MAP source: {how} over {cell_rows.size} chunk(s) of {kind} cell {cell}.")
    else:                                                            # a concrete database entry
        row = np.where((kind_index == ki) & (cells == cell) & (chunks == chunk))[0]
        if row.size == 0:
            avail_chunks = sorted({int(c) for c in chunks[cell_rows]})
            raise ValueError(f"no MAP entry for kind={kind} cell={cell} chunk={chunk} in\n"
                             f"    {map_npz}\navailable chunks for {kind} cell {cell}: {avail_chunks}")
        theta_log10 = map_estimate[int(row[0])]
        print(f"MAP source: chunk {chunk} of {kind} cell {cell}.")
    plo = np.asarray(prior_low, dtype=float)
    phi = np.asarray(prior_high, dtype=float)
    oob = [keys[i] for i in range(len(keys))
           if theta_log10[i] < plo[i] - 1e-9 or theta_log10[i] > phi[i] + 1e-9]
    if oob:
        print(f"WARNING: the MAP is outside the prior box for {oob}; the render EXTRAPOLATES "
              f"there, so a poor experimental-vs-synthetic match may reflect an out-of-prior MAP rather "
              f"than the calibration itself.")
    return bio.to_physical(theta_log10, table)


# Correct-source MET camera parameters (physical units) for --fixed-imaging-parameters.
# gamma = g/C from the MET EM gain and photons2ADU conversion; kappa_o the fitted optical
# background offset; kappa_b the configured baseline; kappa_s the datasheet read noise;
# kappa_q the configured quantum efficiency (marginalized as the SCOPE camera nuisance). See
# REFERENCE_EMCCD_NOISE_MODEL.md Sec. 6 and the MET provenance in DETECTOR_WORKFLOW.md Sec. 6.5.
MET_CAMERA_PHYSICAL = {
    "gamma":   200.0 / 4.78,   # EM gain / conversion (ADU per photoelectron) ~= 41.84
    "kappa_o": 28.7,           # optical background offset (incident photons); ThunderSTORM offset[photon] median, pooled Fab 28.9 / InlB 28.6 (REFERENCE_EMCCD_NOISE_MODEL.md sec. 6, DETECTOR_WORKFLOW.md sec. 6.2)
    "kappa_b": 175.0,          # camera baseline (ADU)
    "kappa_s": 10.5,           # read noise (ADU) at C ~= 4.78
    "kappa_q": 0.9,            # quantum efficiency (config; marginalized as SCOPE camera nuisance)
}


def _fixed_imaging_theta(overrides=None):
    """Physical imaging theta for --fixed-imaging-parameters: the 5 SCOPE camera parameters set
    to their correct-source MET values (``MET_CAMERA_PHYSICAL``), the 6 learnable emitter
    parameters held at their prior-center nominals (the table ``VALUE``), and finally any
    ``overrides`` (``{key: physical_value}``, from ``--set-imaging``) applied last -- e.g.
    emitter brightness/PSF calculated from the MET ThunderSTORM localizations, or a camera
    parameter swept for a sensitivity check. Returns the full 11-key ``DETECTOR_IMAGING`` render
    vector; overrides may target any of the 11 imaging parameters (learnable or SCOPE camera).
    No trained posterior is used -- this drives the corrected forward model directly to test
    whether it recapitulates the experimental pixel-intensity histogram."""
    unknown = [k for k in MET_CAMERA_PHYSICAL if k not in det.DETECTOR_SCOPE_KEYS]
    if unknown:
        raise ValueError(f"MET_CAMERA_PHYSICAL keys not in the SCOPE camera set: {unknown}")
    find = {k: i for i, k in enumerate(det.DETECTOR_IMAGING_KEYS)}   # 11-key render contract (6 learnable + 5 SCOPE)
    theta = np.empty(len(det.DETECTOR_IMAGING_KEYS), dtype=float)
    for element in det.DETECTOR_PARAMETERIZATION:                    # 6 learnable emitter params at prior-center nominals
        theta[find[element["KEY"]]] = element["VALUE"]
    for key, value in MET_CAMERA_PHYSICAL.items():                   # 5 SCOPE camera at correct-source MET values
        theta[find[key]] = value
    for key, value in (overrides or {}).items():                    # --set-imaging: override any of the 11 imaging params
        if key not in find:
            raise ValueError(f"--set-imaging key {key!r} is not an imaging parameter; "
                             f"valid keys are {det.DETECTOR_IMAGING_KEYS}.")
        theta[find[key]] = value
    print("Fixed imaging theta: camera parameters at correct-source MET values ("
          + ", ".join(f"{k}={v:.4g}" for k, v in MET_CAMERA_PHYSICAL.items())
          + "); non-camera parameters at prior-center nominals"
          + (f"; overrides {overrides}" if overrides else "") + ".")
    return theta


# The detector's RDS nuisance is the biology's eleven-parameter prior -- the detector re-images the
# condition's trajectory tier -- so its keys, order, and ranges are the biology table's.
_NUISANCE_KEYS = [e["KEY"] for e in bio.PARAMETERIZATION]


def _draw_nuisance_physical(rng=None):
    """One fresh RDS-nuisance draw for a posterior-predictive render: the eleven reaction-diffusion
    parameters from the biology prior (uniform in estimator space), mapped to physical values by
    the ONE conversion rule (``parameterization.to_physical``: log rows exponentiated, the linear
    initial dimer fraction passed through), in the canonical ``PARAMETERIZATION`` order -- exactly
    what a condition's RDS tier draws per simulation."""
    rng = np.random.default_rng() if rng is None else rng
    low = np.asarray(bio.theta_lower_bound(), dtype=float)
    high = np.asarray(bio.theta_upper_bound(), dtype=float)
    return bio.to_physical(rng.uniform(low, high))


def _fixed_nuisance_physical(overrides=None):
    """Physical RDS-nuisance vector for --fixed-nuisance-RDS: every reaction-diffusion parameter
    held at its prior-center nominal (``parameterization.prior_center``: the physical value at the
    midpoint of its estimator-space range), then any ``overrides`` (``{key: physical_value}``)
    applied last. This replaces the fresh ``_draw_nuisance_physical`` draw with a deterministic,
    controlled nuisance, so the stoichiometry (receptor total ``count_total``, initial
    dimer-to-monomer ratio ``ratio_dimer_monomer_initial``), diffusivities, switching rates, and reaction rates are
    pinned to condition-appropriate values rather than sampled from a flat prior. Returns a
    ``(len(PARAMETERIZATION),)`` array in canonical theta order."""
    centers = {e["KEY"]: bio.prior_center(e) for e in bio.PARAMETERIZATION}
    for key, value in (overrides or {}).items():
        if key not in centers:
            raise ValueError(
                f"--fixed-nuisance-RDS key {key!r} is not an RDS-nuisance parameter; "
                f"valid keys: {_NUISANCE_KEYS}.")
        centers[key] = value
    print("Fixed RDS nuisance: parameters at prior-center nominals"
          + (f"; overrides {overrides}" if overrides else "")
          + " (" + ", ".join(f"{k}={centers[k]:.4g}" for k in _NUISANCE_KEYS) + ").")
    return np.array([centers[k] for k in _NUISANCE_KEYS], dtype=float)


def rds_outside_prior(vector):
    """Keys of a physical reaction-diffusion vector outside the biology prior box (estimator
    coordinates): the support of the trajectory tiers both workflows are trained on. A render there
    extrapolates beyond the training data; it is reported and recorded, never clipped."""
    out = []
    for entry, value in zip(bio.PARAMETERIZATION, np.asarray(vector, dtype=float).ravel()):
        lo, hi = entry["PRIOR_RANGE"]
        coordinate = float(bio.entry_to_flow(entry, value)) if value > 0 or not bio.is_log_row(entry) \
            else float("-inf")
        if coordinate < lo - 1e-9 or coordinate > hi + 1e-9:
            out.append(entry["KEY"])
    return out


_DECLARED_ROW_FIELDS = ("value", "per_cell", "basis", "source")


def _checked_rds_value(entry, value, where):
    """A declared physical value as float: finite, and positive on a log row."""
    value = float(value)
    if not np.isfinite(value) or (bio.is_log_row(entry) and value <= 0):
        raise ValueError(f"{where}: value {value!r} must be finite"
                         + (" and positive (a log row)." if bio.is_log_row(entry) else "."))
    return value


def load_declared_rds(path, condition, cell, overrides=None):
    """Read a declared reaction-diffusion configuration (``--declared-rds``) for one recording.

    The file (TOML) states every one of the eleven reaction-diffusion parameters in physical units
    under ``[parameters.<KEY>]``, each with a non-empty ``basis`` (what kind of statement the value
    is: a declared assumption, a prior center, a value matched to the recording, ...) and
    ``source`` (where it comes from, and its scope), and exactly one of ``value`` or ``per_cell``, a
    table keyed by the cell index for a value that differs between recordings. ``[scenario]`` names
    the configuration (``name``) and its ``condition``, which must be the recording's. ``overrides``
    (``{key: physical}``, from ``--set-rds``) apply last and are recorded as such. Nothing is
    clipped: a value outside the prior box is recorded as an extrapolation (``rds_outside_prior``).

    Returns ``(vector, record)``: the physical vector in canonical ``PARAMETERIZATION`` order and a
    JSON-ready record (path, SHA-256 of the file, scenario, per-parameter value, basis, source and
    origin, the overrides, and the keys outside the prior box).
    """
    import tomllib
    path = Path(path)
    raw = path.read_bytes()
    spec = tomllib.loads(raw.decode("utf-8"))
    scenario = spec.get("scenario", {})
    if not isinstance(scenario, dict) or not str(scenario.get("name", "")).strip():
        raise ValueError(f"{path}: [scenario] needs a non-empty `name`.")
    declared_condition = str(scenario.get("condition", "")).strip()
    if declared_condition != condition:
        raise ValueError(f"{path}: [scenario].condition is {declared_condition!r}, but the recording's "
                         f"condition is {condition!r}; a declared configuration applies to one condition.")
    params = spec.get("parameters", {})
    unknown = sorted(set(params) - set(_NUISANCE_KEYS))
    missing = [k for k in _NUISANCE_KEYS if k not in params]
    if unknown or missing:
        raise ValueError(f"{path}: [parameters] must hold exactly the eleven reaction-diffusion keys "
                         f"{_NUISANCE_KEYS}; missing {missing}, unknown {unknown}.")
    overrides = dict(overrides or {})
    rows = {}
    for entry in bio.PARAMETERIZATION:
        key = entry["KEY"]
        row = params[key]
        extra = sorted(set(row) - set(_DECLARED_ROW_FIELDS))
        if extra:
            raise ValueError(f"{path}: [parameters.{key}] has unknown fields {extra}; allowed "
                             f"{list(_DECLARED_ROW_FIELDS)}.")
        basis, source = str(row.get("basis", "")).strip(), str(row.get("source", "")).strip()
        if not basis or not source:
            raise ValueError(f"{path}: [parameters.{key}] needs non-empty `basis` and `source`.")
        if ("value" in row) == ("per_cell" in row):
            raise ValueError(f"{path}: [parameters.{key}] needs exactly one of `value` or `per_cell`.")
        if "per_cell" in row:
            table = {str(k): v for k, v in dict(row["per_cell"]).items()}
            if str(cell) in table:
                value, origin = table[str(cell)], f"per_cell[{cell}]"
            elif key in overrides:                     # an override supplies the missing entry
                value, origin = None, f"per_cell[{cell}] absent"
            else:
                raise ValueError(f"{path}: [parameters.{key}].per_cell has no entry for cell {cell}; "
                                 f"cells: {sorted(table, key=lambda c: int(c))}.")
        else:
            value, origin = row["value"], "value"
        if value is not None:
            value = _checked_rds_value(entry, value, f"{path}: [parameters.{key}]")
        rows[key] = {"value": value, "basis": basis, "source": source, "origin": origin}
    for key, value in overrides.items():
        if key not in rows:
            raise ValueError(f"--set-rds key {key!r} is not a reaction-diffusion parameter; "
                             f"valid keys: {_NUISANCE_KEYS}.")
        entry = bio.PARAMETERIZATION[_NUISANCE_KEYS.index(key)]
        rows[key] = dict(rows[key], value=_checked_rds_value(entry, value, f"--set-rds {key}"),
                         origin="override (--set-rds)", declared_value=rows[key]["value"])
    vector = np.array([rows[k]["value"] for k in _NUISANCE_KEYS], dtype=float)
    record = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "scenario": scenario,
              "cell": int(cell), "parameters": rows, "overrides": dict(overrides or {}),
              "outside_prior": rds_outside_prior(vector),
              "text": raw.decode("utf-8")}                    # the file as read, so a clip is self-contained
    return vector, record


def imaging_provenance(imaging_physical):
    """``key=value`` strings for all eleven imaging parameters in ``DETECTOR_IMAGING`` order (the six
    emitter parameters and the five SCOPE camera values), ``*`` marking a value outside its box
    (the imaging prior for the six, the SCOPE box for the camera)."""
    img = np.asarray(imaging_physical, dtype=float).ravel()
    if img.size != len(det.DETECTOR_IMAGING):
        raise ValueError(f"imaging vector has {img.size} values; the render contract has "
                         f"{len(det.DETECTOR_IMAGING)} ({det.DETECTOR_IMAGING_KEYS}).")
    short = {"prob_photo_bleach": "p_bleach", "lambda_rate": "lambda"}
    out = []
    for entry, value in zip(det.DETECTOR_IMAGING, img):
        lo, hi = entry["PRIOR_RANGE"]
        coordinate = float(bio.entry_to_flow(entry, value)) if value > 0 else float("-inf")
        mark = "*" if (coordinate < lo - 1e-9 or coordinate > hi + 1e-9) else ""
        out.append(f"{short.get(entry['KEY'], entry['KEY'])}={value:.4g}{mark}")
    return out


def _save_comparison_png(path, experimental, synth, kind, cell, sel_desc, display_norm,
                         nuisance, imaging_physical, *, synth_label, fixed_imaging=False,
                         fixed_nuisance=False, imaging_label="imaging",
                         rds_label="reaction-diffusion", motion_desc=None, rds_table=None,
                         labeling_desc=None, rds_outside=()):
    """Static experimental-vs-synthetic panel. ``synth_label`` (``synthetic_source_label``) names
    the synthetic source in the synthetic panel and the title, as the clip file does; ``sel_desc``
    names the MAP selection and is None when no MAP was read.

    Col 0 = EXPERIMENTAL and col 1 = SYNTH, each
    showing the mid frame over its max projection; col 2 holds the shared pixel-intensity
    histogram in log-y (full range, top) and linear-y (zoomed to the bulk, bottom); col 3 holds
    the syn/exp RATIO-per-quantile match plot (top -- a flat line on 1.0 = perfect match, read
    across min/p0.01/median/p90/p99/p99.9/p99.99/max so no single region dominates) over the
    quantile table + provenance (bottom). The provenance lists this render's DLI/imaging
    parameters (out-of-prior ones flagged) and its RDS-nuisance draw, so a mismatch can be
    attributed to the imaging estimate or the nuisance rather than left unexplained.

    Both panels and the histogram use the STORED synthetic (``synth_u16``, clipped to the
    non-negative uint16 range), so the comparison is like-with-like against the experimental frames and
    matches the persisted clip -- not the pre-clip float. The experimental and synthetic image panels
    ALWAYS share ONE color limit, in every ``display_norm`` mode -- the max-projection row shares the
    full ``[min, max]`` range, and the mid-frame row shares a window the mode selects -- so identical
    intensities map to identical colors and the exp-vs-synth comparison is fair; the mode only sets WHAT
    the shared mid-frame window is. ``full`` (default): the full whole-clip ``[min, max]`` (nothing
    clipped). ``percentile``: the whole-clip ``[min, p99.99]``, dropping the top-0.01% hot-pixel sliver
    for contrast. ``autoscale``: the two displayed frames' shared min/max, the most contrast. The
    histogram is in ADU in every mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    imaging_header = f"{imaging_label} (absolute)"
    fig_title = ("Fixed-imaging video check (experimental vs synthetic; correct-source MET)"
                 if fixed_imaging
                 else f"Posterior-predictive video check: experimental vs {synth_label}")
    mid = experimental.shape[0] // 2
    label = {"FAB": "MET-FAB", "INLB": "MET-INLB"}.get(kind, kind)
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5), dpi=200)

    # --- pixel-intensity quantiles (ADU): drive both the frame display window and the match plot ---
    exp_r = experimental.ravel(); syn_r = synth.ravel()
    q_probs = [0, 0.01, 50, 90, 99, 99.9, 99.99, 100]
    q_names = ["min", "p0.01", "median", "p90", "p99", "p99.9", "p99.99", "max"]
    eq = np.percentile(exp_r, q_probs); sq = np.percentile(syn_r, q_probs)
    # The full quantile set drives the match plot + table (below); the frame display window
    # (below) spans the full range, so no pixels are clipped from the frame color scale.

    def _frame(a, arr, title, clim):
        a.imshow(arr, cmap="magma", origin="lower", interpolation="none",
                 vmin=clim[0], vmax=clim[1])
        a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])

    # The experimental and synthetic image panels ALWAYS share ONE color limit, in EVERY mode, so
    # identical intensities map to identical colors and the comparison is fair: the mid-frame row
    # shares `frame_clim` and the max-projection row shares `proj_clim`. `display_norm` only sets WHAT
    # the shared mid-frame window is; the max projection always shares the full [min, max] range.
    emp, smp = experimental.max(0), synth.max(0)
    proj_clim = (float(min(emp.min(), smp.min())), float(max(emp.max(), smp.max())))
    if display_norm == "full":                    # full whole-clip [min, max]; nothing clipped
        frame_clim = (float(min(eq[0], sq[0])), float(max(eq[-1], sq[-1])))
    elif display_norm == "percentile":            # whole-clip [min, p99.99]; drops the hot-pixel sliver
        frame_clim = (float(min(exp_r.min(), syn_r.min())),
                      float(max(np.percentile(exp_r, 99.99), np.percentile(syn_r, 99.99))))
    else:                                         # autoscale: the two displayed frames' shared min/max
        frame_clim = (float(min(experimental[mid].min(), synth[mid].min())),
                      float(max(experimental[mid].max(), synth[mid].max())))
    exp_clim = syn_clim = frame_clim
    exp_proj_clim = syn_proj_clim = proj_clim

    # col 0 = EXPERIMENTAL (frame over max projection), col 1 = SYNTH (same); col 2 = histograms;
    # col 3 = the syn/exp ratio-per-quantile match plot over the quantile table + provenance.
    _frame(ax[0, 0], experimental[mid], f"EXPERIMENTAL {label} cell {cell}  frame {mid}", exp_clim)
    _frame(ax[1, 0], experimental.max(0), "EXPERIMENTAL  max projection", exp_proj_clim)
    sel = f" {sel_desc}" if sel_desc else ""
    _frame(ax[0, 1], synth[mid], f"{synth_label} {kind} c{cell}{sel}  frame {mid}", syn_clim)
    _frame(ax[1, 1], synth.max(0), "SYNTH  max projection", syn_proj_clim)

    # --- histograms with SHARED bins (like-with-like): log-y over the full range (top), and
    #     linear-y through ~p99.99 (bottom) -- the whole range bar the top-0.01% hot-pixel sliver,
    #     kept off only so the linear scale stays readable; the tail is reported exactly in the
    #     quantile table at right, and the log-y panel above shows it in full. ---
    lo = float(min(exp_r.min(), syn_r.min()))
    hi_full = float(max(exp_r.max(), syn_r.max()))
    _i9999 = q_names.index("p99.99")
    hi_lin = float(max(eq[_i9999], sq[_i9999]) * 1.05)          # linear panel through ~p99.99
    bins_log = np.linspace(lo, hi_full, 200)
    bins_lin = np.linspace(lo, hi_lin, 160)

    def _hist(a, bins):
        a.hist(exp_r, bins=bins, histtype="step", density=True, color="tab:blue", label="experimental")
        a.hist(syn_r, bins=bins, histtype="step", density=True, color="tab:orange", label="synth")
        a.tick_params(labelsize=7)

    _hist(ax[0, 2], bins_log); ax[0, 2].set_yscale("log")
    ax[0, 2].set_title("pixel-intensity density (ADU) - log y, full range", fontsize=8)
    ax[0, 2].legend(fontsize=7)
    _hist(ax[1, 2], bins_lin); ax[1, 2].set_xlim(lo, hi_lin)
    ax[1, 2].set_title(f"linear y, x<={hi_lin:.0f} ADU (to p99.99)", fontsize=8)

    # --- ax[0,3]: syn/exp ratio per quantile -- the direct "do they match?" read.
    #     A flat line on 1.0 (dashed) is a perfect match; the green band is within +-10%.
    #     Log-y so a 2x over- and a 2x under-shoot read symmetrically; spans the dark end
    #     (min, p0.01) to the bright end (p99.99), so no single region dominates the judgment. ---
    axr = ax[0, 3]
    ratios = np.array([(sv / ev) if ev > 0 else np.nan for ev, sv in zip(eq, sq)])
    xq = np.arange(len(q_names))
    axr.axhspan(0.9, 1.1, color="tab:green", alpha=0.12)
    axr.axhline(1.0, color="gray", lw=1.0, ls="--")
    axr.plot(xq, ratios, "-o", color="tab:red", ms=5)
    axr.set_yscale("log"); axr.set_ylim(0.5, 3.5)
    # Fixed decimal y-labels; suppress the log MINOR ticks -- in this narrow [0.5, 3.5] range
    # they otherwise print auto sci-notation labels (e.g. 6x10^-1) that clutter the axis.
    axr.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 1.0, 1.5, 2.0, 3.0]))
    axr.yaxis.set_major_formatter(FixedFormatter(["0.5", "0.7", "1.0", "1.5", "2.0", "3.0"]))
    axr.yaxis.set_minor_locator(NullLocator())
    axr.tick_params(axis="y", labelsize=7)
    axr.set_xticks(xq); axr.set_xticklabels(q_names, rotation=45, ha="right", fontsize=7)
    axr.set_ylabel("synth / exp", fontsize=8)
    axr.set_title("MATCH: syn/exp per quantile\n(1.0 dashed = perfect; green = within 10%)", fontsize=8)
    axr.grid(True, axis="y", which="both", alpha=0.25)

    # --- ax[1,3]: quantile table + provenance (imaging theta + RDS nuisance) ---
    ax[1, 3].axis("off")
    # All eleven imaging values: the six emitter parameters and the five pinned camera values.
    dli = imaging_provenance(imaging_physical)
    dli_lines = "\n".join("  " + "  ".join(dli[j:j + 3]) for j in range(0, len(dli), 3))
    nuis = np.asarray(nuisance, dtype=float).ravel()
    npart = []
    # The label table must match the block being shown, or zip() silently truncates. Both
    # workflows' reaction-diffusion vectors are the biology's eleven parameters in canonical order
    # (the detector's RDS nuisance is that prior, re-imaged from the condition's tier), so the biology
    # table labels either; the explicit check keeps a future schema change from truncating.
    table = bio.PARAMETERIZATION if rds_table is None else rds_table
    if len(table) != nuis.size:
        raise ValueError(f"RDS label table has {len(table)} entries but the vector has "
                         f"{nuis.size}; they must correspond.")
    for ent, v in zip(table, nuis):
        lab2 = ent["LABEL"].translate({ord(c): None for c in "${}\\"})
        # Integer-format ONLY a true particle count. The prefix test this replaced also matched
        # "Count Per Second" -- biology's rate units -- and rounded continuous rates to integers,
        # printing 0.175/s and 0.127/s as "0", a value their log-uniform prior [0.1, 10] cannot
        # even represent. Exact match, so a new "Count ..." unit cannot silently re-break it.
        is_count = str(ent.get("UNIT", "")).strip().lower() == "count"
        mark = "*" if ent["KEY"] in set(rds_outside) else ""
        npart.append((f"{lab2}={v:.0f}" if is_count else f"{lab2}={v:.3g}") + mark)
    nuis_lines = "\n".join("  " + "  ".join(npart[j:j + 3]) for j in range(0, len(npart), 3))
    tbl = f"{'quantile':<8}{'exp':>7}{'synth':>7}{'ratio':>7}\n"
    for name, ev, sv in zip(q_names, eq, sq):
        tbl += f"{name:<8}{ev:7.0f}{sv:7.0f}{(sv / ev if ev > 0 else float('inf')):7.2f}\n"
    ax[1, 3].text(
        0.0, 1.0,
        "QUANTILES (ADU)  ratio=synth/exp\n" + tbl + "\n"
        + f"{imaging_header}:\n{dli_lines}\n  (* outside prior / SCOPE box)\n"
        + f"{rds_label}:\n{nuis_lines}\n"
        + ("  (* outside the biology prior)\n" if rds_outside else "")
        + (f"labeling: {labeling_desc}\n" if labeling_desc else "")
        + f"motion: {motion_desc or ('fixed nuisance (pinned)' if fixed_nuisance else 'fresh draw')}"
        f"; norm {display_norm}",
        fontsize=6.5, va="top", family="monospace")
    fig.suptitle(fig_title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def _aggregate_cell(rows_log10, source, table):
    """Reduce one cell's per-chunk MAP rows to a single vector, returned in estimator space.

    The population is the cell's window MAP vectors, so both choices summarize MAPs, not posterior
    draws. ``cell-sgm`` takes their Sample Geometric Median: the exact medoid (a cell has far fewer
    chunks than the kernel's capacity) in PHYSICAL coordinates, each divided by its range over this
    cell's chunks -- neither the prior width nor the estimator coordinates of the stage products'
    ``posterior_sgm``. The result is an actual chunk's MAP, so the rendered video corresponds to a
    configuration the optimizer returned for this cell. ``cell-median`` takes each dimension's
    median independently, in estimator coordinates, which composes coordinates that need never
    have co-occurred; for a render that matters, because the simulator is then asked to realize a
    combination no chunk supported. Both are offered, and the report names which was used.
    """
    if source == "cell-sgm":
        a = bio.to_physical(rows_log10, table)            # the medoid is defined in physical space
        rng_span = np.ptp(a, axis=0)
        rng_span[rng_span <= 0] = 1.0
        idx, _method = sample_geometric_median(a, rng_span)
        return rows_log10[idx]
    return np.median(rows_log10, axis=0)


def run_posterior_predictive_video(cfg, args):
    """Shared entry point. ``cfg`` is a WorkflowConfig; ``args`` the parsed CLI namespace."""
    S = _ppv_spec(cfg, args)
    keys = S["map_keys"]
    kind = KIND_OF_CONDITION.get(args.kind, args.kind)      # scientific name -> stored token
    frame_time = PARAMETERS.simulation.timing.frame_time_seconds
    map_npz, experimental_tif, out_dir = S["map_npz"], S["experimental_tif"], S["out_dir"]

    # ---- The observation layer and the reaction-diffusion source, resolved before any work. The
    # labeling plan is the production one: the condition's law and declared (or derived) occupancy,
    # or the --labeling-law / --occupancy override, resolved exactly as at the DLI stage.
    plan = resolve_labeling(kind, args.labeling_law, args.occupancy)
    if args.set_rds and not args.declared_rds:
        raise SystemExit("--set-rds overrides a value of the --declared-rds configuration; pass "
                         "--declared-rds as well.")
    if args.declared_rds and args.fixed_nuisance_rds:
        raise SystemExit("--declared-rds and --fixed-nuisance-RDS both set the reaction-diffusion "
                         "block; pass one of them.")
    declared = (load_declared_rds(args.declared_rds, kind, args.cell,
                                  overrides=_parse_kv(args.set_rds, "--set-rds"))
                if args.declared_rds else None)
    needs_map = not ((S["map_block"] == "imaging" and args.fixed_imaging_parameters)
                     or (S["map_block"] == "rds" and declared is not None))
    stem_kw = dict(fixed_imaging=args.fixed_imaging_parameters,
                   fixed_nuisance=bool(args.fixed_nuisance_rds), run_label=args.run_label,
                   declared_rds=declared is not None, map_block=S["map_block"])

    if args.dry_run:
        n_frames = inspect_recording(experimental_tif)[0] if experimental_tif.exists() else None
        clip_token = _clip_span_token(n_frames, frame_time) if n_frames else "<clip_span>"
        stem = _build_stem(S["paths"].project_alias, S["map_label"], args.kind, args.cell, args.chunk,
                           args.map_source, clip_token, **stem_kw)
        print(f"[DRY RUN] posterior-predictive video ({cfg.tag}) would read:")
        if needs_map:
            print(f"    MAP supplies : the {S['map_block'].upper()} block ({len(keys)} parameters)")
            print(f"    MAP database : {map_npz}  [{'OK' if map_npz.exists() else 'MISSING'}]")
            sel = f"chunk {args.chunk}" if args.map_source == "chunk" else args.map_source
            print(f"    selection    : {args.kind} cell {args.cell}  ({sel})")
        elif S["map_block"] == "imaging":
            print("    imaging theta: FIXED to MET values (no MAP read); "
                  "non-camera parameters at prior-center nominals")
        else:
            print("    MAP          : none read (the reaction-diffusion block is declared)")
        print(f"    imaging held : {S['imaging_desc']}")
        if S["map_block"] == "rds":
            try:
                img = resolve_biology_imaging(PARAMETERS.machine.root_for("EVAL"), S["map_label"],
                                              kind, nuisance_tag=args.nuisance_tag)
                print(f"    Nuisance_DLI : {img['path']}  [OK]  ({img['description']})")
            except Exception as exc:                              # report, do not stop the preview
                print(f"    Nuisance_DLI : MISSING or invalid -- {exc}")
        print(f"    RDS built    : {S['rds_desc'] if declared is None else 'the declared configuration'}")
        if declared is not None:
            _print_declared(declared[1])
        print(f"    labeling     : {plan.describe()}")
        print(f"    experimental : {experimental_tif}  "
              f"[{'OK' if experimental_tif.exists() else 'MISSING'}]")
        print(f"    display norm : {args.display_norm}")
        print(f"[DRY RUN] would write under:\n    {out_dir}/\n"
              f"        {stem}_{{Synthetic_Video.npz,Comparison.png,Trajectory.h5}}")
        existing = [p.name for p in render_output_paths(out_dir, stem).values() if p.exists()]
        if existing:
            print(f"[DRY RUN] EXISTS: {', '.join(existing)} -- the run would refuse; another attempt "
                  f"or a variant needs its own --run-label.")
        print("[DRY RUN] no simulation, no render.")
        return 0

    if needs_map and not map_npz.exists():
        raise FileNotFoundError(
            f"MAP database not found:\n    {map_npz}\nRun the {cfg.tag} Experiment stage first.")
    if not experimental_tif.exists():
        raise FileNotFoundError(
            f"Experimental recording not found:\n    {experimental_tif}\n"
            f"(check --kind/--cell and --experiment-span-seconds={args.experiment_span_seconds}).")

    # ---- the render length comes from the recording (never hardcoded; one layout rule for every
    # reader, experiment_support.inspect_recording). The three outputs are named from it before any
    # work, and an existing one is refused: a render never overwrites another. ----
    n_frames = int(inspect_recording(experimental_tif)[0])
    clip_token = _clip_span_token(n_frames, frame_time)
    stem = _build_stem(S["paths"].project_alias, S["map_label"], args.kind, args.cell, args.chunk,
                       args.map_source, clip_token, **stem_kw)
    outputs = render_output_paths(out_dir, stem)
    refuse_existing_outputs(outputs)

    map_theta = None
    if needs_map:
        map_theta = _load_map_theta(map_npz, keys, S["table"], kind, args.cell, args.chunk,
                                    args.map_source, S["prior_low"], S["prior_high"])

    imaging_physical, imaging_desc, imaging_identity = S["imaging_physical"](args, map_theta)

    experimental = np.asarray(read_recording(experimental_tif))    # (n_frames, H, W), shape checked
    declared_biology = declared is not None and S["map_block"] == "rds"
    sources = ("the reaction-diffusion block declared, no MAP" if declared_biology
               else f"MAP -> {S['map_block']}, {args.map_source}" if needs_map else "no MAP")
    print(f"Experimental recording: {n_frames} frames ({clip_token} clip); rendering a matching "
          f"synthetic for {args.kind} cell {args.cell} ({sources}).")

    # ---- the reaction-diffusion vector, then one clip at the experimental length through the
    # production chain (simulate_and_render: the RDS stage's system, the DLI stage's labeling and
    # renderer) ----
    out_dir.mkdir(parents=True, exist_ok=True)
    rds_provenance, rds_desc, rds_source, rds_record = S["rds_vector"](args, map_theta, declared)
    rds_outside = rds_outside_prior(rds_provenance)
    if rds_outside and rds_source != "map":                    # the MAP path has already warned
        print(f"WARNING: the reaction-diffusion vector lies outside the biology prior box for "
              f"{rds_outside}; the trajectory tiers the estimators are trained on contain no such "
              f"system, so this render EXTRAPOLATES beyond the training support.")
    traj_path = outputs["trajectory"]
    condition = kind
    clip = simulate_and_render(condition, rds_provenance, imaging_physical, plan, n_frames, traj_path,
                               seed=args.seed, verbose=args.verbose)
    synth, dye_counts, labeling_row = clip["synth"], clip["dye_counts"], clip["labeling_row"]
    lab = dict(zip(LABELING_SET_COLUMNS, labeling_row))
    print(f"Labeling: {plan.describe()} -> {int(lab['n_dyes'])} dyes on "
          f"{int(lab['n_labeled_subunits'])} of {int(lab['n_subunits'])} subunits; frame 0: "
          f"{int(lab['monomers_visible_0'])}/{int(lab['monomers_0'])} monomers and "
          f"{int(lab['dimers_visible_0'])}/{int(lab['dimers_0'])} dimers visible "
          f"({int(lab['dimers_two_labeled_0'])} with both subunits labeled).")

    # ---- persist (clip + provenance) and a static comparison figure ----
    clips_path = outputs["clip"]
    # Storage clip to the non-negative uint16 range. The reference EMCCD noise model emits a small
    # signed excursion -- a Gaussian read-noise tail below 0, rarely values above 65535 -- that a
    # real camera cannot record, so we clip to the domain the experimental frames live in. The
    # counts are reported on every run so an alternative noise model's behavior can be compared at
    # this exact point. See REFERENCE_EMCCD_NOISE_MODEL.md.
    n_under = int(np.count_nonzero(synth < 0.0))
    n_over = int(np.count_nonzero(synth > 65535.0))
    if n_under or n_over:
        print(f"  storage clip to [0, 65535]: {n_under} pixel(s) < 0, {n_over} pixel(s) > 65535 "
              f"(of {synth.size}); clipped to match the experimental non-negative domain.")
    synth_u16 = np.clip(np.rint(synth), 0, 65535).astype(np.uint16)
    synth_label = synthetic_source_label(S["map_block"], declared_biology, args.fixed_imaging_parameters)
    np.savez_compressed(
        str(clips_path),
        experimental=experimental.astype(np.uint16), synth=synth_u16,
        imaging_physical=imaging_physical, imaging_keys=np.array(det.DETECTOR_IMAGING_KEYS),
        map_theta=(np.array([]) if map_theta is None else map_theta),
        map_keys=np.array(keys), map_block=S["map_block"], workflow=cfg.tag,
        rds_provenance=rds_provenance, rds_keys=np.array(_NUISANCE_KEYS), rds_source=rds_source,
        rds_record_json=json.dumps(rds_record, default=float),
        rds_outside_prior=np.array(rds_outside, dtype=str),
        condition=condition, labeling_law=plan.law_name, dye_counts=dye_counts,
        n_subunits=clip["n_subunits"], n_dyes=int(dye_counts.sum()),
        labeling_columns=np.array(LABELING_SET_COLUMNS), labeling_row=labeling_row,
        occupancy_source=plan.occupancy_source, occupancy_monomer=float(plan.occupancy_pair[0]),
        occupancy_dimer=float(plan.occupancy_pair[1]), labeling_record_json=json.dumps(plan.record()),
        imaging_record_json=json.dumps(imaging_identity, default=str),
        synth_label=synth_label, package_version=_PACKAGE_VERSION,
        kind=args.kind, cell=args.cell,
        chunk=(-1 if args.chunk is None else args.chunk),
        map_source=("declared" if declared_biology else args.map_source),
        seed=(-1 if args.seed is None else args.seed),
        frame_time_seconds=frame_time, n_frames=n_frames, experimental_tif=str(experimental_tif),
        imaging_desc=str(imaging_desc), nuisance_tag=("" if args.nuisance_tag is None else args.nuisance_tag))
    declared_label = (None if declared is None else
                      f"DECLARED reaction-diffusion ({declared[1]['scenario']['name']}, absolute)")
    if S["map_block"] == "rds":
        imaging_label = "FIXED imaging (calibrated Nuisance_DLI + MET SCOPE)"
        if declared_label:
            rds_label = declared_label
            motion_desc = "from the declared reaction-diffusion configuration (not a draw)"
        else:
            rds_label = "INFERRED reaction-diffusion (MAP theta, absolute)"
            motion_desc = "from the MAP reaction-diffusion parameters (not a draw)"
        rds_table = parameter_table(cfg)                 # the eleven biology parameters
    else:
        imaging_label = ("FIXED imaging (MET values)" if args.fixed_imaging_parameters
                         else "INFERRED imaging (MAP theta, absolute)")
        if declared_label:
            rds_label = declared_label
            motion_desc = "from the declared reaction-diffusion configuration"
        else:
            rds_label = "NUISANCE reaction-diffusion (marginalized)"
            motion_desc = None
        rds_table = None                                 # detector: labeled by the biology table (its RDS nuisance)
    _save_comparison_png(outputs["figure"], experimental, synth_u16,
                         args.kind, args.cell, args.map_source if needs_map else None,
                         args.display_norm, rds_provenance, imaging_physical,
                         synth_label=synth_label,
                         imaging_label=imaging_label, rds_label=rds_label,
                         motion_desc=motion_desc, rds_table=rds_table,
                         fixed_imaging=args.fixed_imaging_parameters,
                         fixed_nuisance=bool(args.fixed_nuisance_rds),
                         labeling_desc=plan.describe(), rds_outside=rds_outside)
    print(f"Persisted clip + provenance:\n    {clips_path}")
    print(f"Static comparison figure:\n    {outputs['figure']}")
    print(f"Trajectory (provenance):\n    {traj_path}")
    print(f"  imaging : {imaging_desc}")
    print(f"  RDS     : {rds_desc}")
    print(f"  labeling: {plan.describe()}")
    return 0


def simulate_and_render(condition, rds_vector, imaging_physical, plan, n_frames, traj_path,
                        seed=None, verbose=False):
    """One clip through the production chain: a reaction-diffusion trajectory of ``n_frames`` frames
    under ``condition`` from the physical ``rds_vector`` (the RDS stage's ``build_system`` and
    ``build_simulation``, the configured timing), labeled under ``plan`` through the DLI stage's
    labeling path (``labeling.label_trajectory``; the stream of task 0, simulation 0 under ``seed``)
    and rendered with ``imaging_physical`` by the DLI stage's renderer. ``seed`` seeds the placement
    and the render directly, as the two stages do. The labeling stream of task 0, simulation 0
    coincides with that bare-seed stream (``labeling.labeling_rng``): the production convention,
    retained; stream separation is not established. ReaDDy's own dynamics stay OS-seeded, so a seed
    does not reproduce the trajectory. The trajectory is written to ``traj_path``, which must not
    exist: an existing file is refused, never deleted.

    Returns a dict: ``synth`` ``(n_frames, H, W)`` float (before the storage clip), ``dye_counts``,
    ``labeling_row`` (``labeling.LABELING_SET_COLUMNS``) and ``n_subunits``.
    """
    traj_path = Path(traj_path)
    if traj_path.exists():
        raise FileExistsError(f"{traj_path} exists; a render never replaces an existing trajectory.")
    import readdy
    timing = RunTiming(total_time_seconds=n_frames * PARAMETERS.simulation.timing.frame_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    stem = build_system(rds_vector, condition, verbose=verbose)
    smut = build_simulation(stem, rds_vector, seed=seed, verbose=verbose)
    smut.output_file = str(traj_path)
    smut.progress_output_stride = timing.total_steps
    smut.run(n_steps=timing.total_steps,
             timestep=timing.delta_time_nanoseconds * readdy.units.nanosecond,
             show_summary=False)
    tray = readdy.Trajectory(filename=str(traj_path))
    tray_poses = extract_trajectory_poses(tray, verbose=verbose)
    lineage = extract_subunit_lineage(tray, verbose=verbose)
    if tray_poses.shape[0] != timing.frame_count:
        raise RuntimeError(f"trajectory holds {tray_poses.shape[0]} frames but the render "
                           f"length is {timing.frame_count}.")
    soul_poses = collapse_species_axis(tray_poses)
    dye_counts, labeling_row = label_trajectory(plan, tray, lineage, labeling_rng(seed))
    del tray, smut, stem, tray_poses
    synth = render_dli_video(soul_poses=soul_poses, host_index=lineage.host_index,
                             dye_counts=dye_counts, imaging_physical=imaging_physical,
                             seed=seed, verbose=verbose)
    return {"synth": np.moveaxis(synth, -1, 0),                # (H, W, n_frames) -> (n_frames, H, W)
            "dye_counts": dye_counts, "labeling_row": labeling_row,
            "n_subunits": int(lineage.n_subunits)}


def _print_declared(record):
    """Dry-run listing of a declared reaction-diffusion configuration (``load_declared_rds``)."""
    print(f"    declared RDS : {record['scenario']['name']}  ({record['path']}, "
          f"sha256 {record['sha256'][:12]}), cell {record['cell']}")
    for key, row in record["parameters"].items():
        mark = "  [OUTSIDE THE PRIOR]" if key in record["outside_prior"] else ""
        print(f"        {key:<32} {row['value']:<10.4g} {row['origin']:<22} {row['basis']}{mark}")


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window of the RUN that produced the MAP database; sets the timing "
                        "label locating it. The render length comes from the recording itself.")
    p.add_argument("--kind", default="MET-FAB", choices=tuple(KIND_OF_CONDITION),
                   help="experimental condition: MET-FAB (monomer control) or MET-INLB (dimer).")
    p.add_argument("--cell", type=int, required=True, help="cell (recording) index.")
    p.add_argument("--chunk", type=int, default=None,
                   help="chunk index, for --map-source chunk.")
    p.add_argument("--map-source", choices=("chunk", "cell-sgm", "cell-median"), default="cell-sgm",
                   help="which MAP vector to render: 'chunk' one specific window; 'cell-sgm' "
                        "(default) an SGM of that cell's window MAPs -- a real window's map_estimate "
                        "whose coordinates co-occurred (not the product's per-window posterior-draw "
                        "SGM); 'cell-median' the per-dimension median, which can compose a "
                        "combination no window produced.")
    p.add_argument("--experiment-span-seconds", type=int, default=20,
                   help="duration (s) of the experimental recording to read (default 20).")
    p.add_argument("--labeling-law", type=str, default=None,
                   help="override the condition's baseline static labeling law (registry key or "
                        "'family:mean[:shape]'); default: the baseline of --kind's condition "
                        "(FAB_POISSON / INLB_BERNOULLI).")
    p.add_argument("--occupancy", type=str, default=None,
                   help="override the condition's probe occupancy for a sensitivity render, exactly as "
                        "at the DLI stage: one value, or per initial MOLECULAR species 'A=0.5,B=1.0' "
                        "(monomer A, dimer B). Default: the condition's declared (MET-INLB 0.5) or "
                        "derived (MET-FAB 0.155) value; the value used is recorded in the clip.")
    p.add_argument("--display-norm", default="full", choices=("full", "autoscale", "percentile"),
                   help="display normalization for the comparison figure: 'full' (default; "
                        "shared full-range window), 'autoscale' (per-image), or 'percentile'. "
                        "Display-only -- never enters the quantitative comparison.")
    p.add_argument("--fixed-imaging-parameters", action="store_true",
                   help="detector only: skip the MAP database and pin imaging to MET values.")
    p.add_argument("--set-imaging", action="append", default=[], metavar="KEY=VALUE",
                   help="override one imaging parameter (sensitivity check); repeatable.")
    p.add_argument("--fixed-nuisance-RDS", dest="fixed_nuisance_rds", action="append", default=None,
                   metavar="KEY=VALUE",
                   help="detector only: pin the RDS nuisance at its prior-center nominals, with KEY=VALUE "
                        "overrides, instead of drawing it; repeatable.")
    p.add_argument("--declared-rds", type=str, default=None, metavar="TOML",
                   help="both workflows: take the reaction-diffusion block from a declared configuration "
                        "file -- each of the eleven values with its basis and source (load_declared_rds) "
                        "-- instead of the MAP (biology) or a drawn or pinned nuisance (detector); the "
                        "file's SHA-256 and every value's origin are recorded in the clip.")
    p.add_argument("--set-rds", action="append", default=[], metavar="KEY=VALUE",
                   help="with --declared-rds: override one declared reaction-diffusion value (physical "
                        "units); repeatable; recorded as an override.")
    p.add_argument("--nuisance-tag", type=str, default=None,
                   help="biology only: SCREAMING_SNAKE token selecting a tagged Nuisance_DLI artifact "
                        "(a reference vector or a sensitivity variant) instead of the canonical one; "
                        "recorded in the clip file and the figure provenance.")
    p.add_argument("--run-label", default=None,
                   help="token appended to the output stem. A run refuses to overwrite an existing "
                        "render, and the stem carries neither the nuisance tag nor the declared "
                        "configuration, so another attempt or a variant needs its own label.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed, used as production uses it: it seeds the particle placement and the "
                        "render directly, the labeling draw through labeling.labeling_rng (the DLI stage's "
                        "stream for task 0, sim 0, which coincides with the bare-seed stream), and a drawn "
                        "detector nuisance as the RDS stage draws a tier's theta. ReaDDy's dynamics stay "
                        "OS-seeded, so a seed does not reproduce the trajectory.")
    p.add_argument("--verbose", action="store_true", help="verbose simulation/render output.")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; simulate nothing.")
    return p


# ---- the one place the two workflows differ ----------------------------------------------------

def _scope_met():
    """The five SCOPE camera parameters at their MET values, in DETECTOR_SCOPE order."""
    return np.array([MET_CAMERA_PHYSICAL[k] for k in det.DETECTOR_SCOPE_KEYS], dtype=float)


def resolve_biology_imaging(data_bank_root, map_label, condition, nuisance_tag=None):
    """The biology render's fixed 11-key imaging vector and its provenance, resolved from the artifacts.

    Biology holds imaging FIXED at the calibrated vector the training videos were generated
    with: the six emitter parameters come from the ``Nuisance_DLI`` artifact at run time (its
    Sample Geometric Median when it pools multiple vectors -- a member of whichever pool its
    construction recorded, so the chosen vector's coordinates co-occurred, rather than a composite
    of per-dimension medians), and the five SCOPE camera
    parameters are pinned to their correct-source MET values. The values live only in the
    artifact -- they appear in no source file -- so hardcoding them anywhere would silently
    drift from whatever the videos were actually built with.

    The artifact is read as the biology DLI stage reads it: the same loader and tag resolution
    (``require_nuisance_dli``), the same schema guard (the six emitter keys in
    ``DETECTOR_PARAMETER_KEYS`` order) and, for a fixed ``selection_user`` vector, the same accessor
    (``NuisanceDLI.sample``), so a check render exercises the biology input path rather than a copy
    of its numbers. A pool has no single vector: the render takes its Sample Geometric Median (a
    representative), where the DLI stage draws a member per simulation.

    Shared by the posterior-predictive render and the horizon audit, so both draw the SAME
    imaging vector from ONE definition. Returns a dict: ``vector`` (the physical 11-key
    ``DETECTOR_IMAGING``-order render input), ``description``, ``identity`` (the artifact identity
    the DLI stage records in every ``Nuisance_DLI_Theta_Set``, with the tag) and ``path``.
    """
    from .detector_nuisance_dli import artifact_path, require_nuisance_dli
    # One Nuisance_DLI per condition: the detector calibrated on that condition's recordings.
    det_paths = det.detector_paths(PARAMETERS.paths).with_condition(condition)
    posit_dir = data_bank_root / det_paths.posit_subdir
    art = artifact_path(posit_dir, det_paths.project_alias, map_label, nuisance_tag=nuisance_tag)
    nu = require_nuisance_dli(posit_dir, det_paths.project_alias, map_label, nuisance_tag=nuisance_tag)
    if list(nu.parameter_keys) != list(det.DETECTOR_PARAMETER_KEYS):
        raise ValueError(f"Nuisance_DLI schema mismatch: artifact parameter_keys "
                         f"{list(nu.parameter_keys)} != expected {det.DETECTOR_PARAMETER_KEYS} "
                         f"(the six photophysics in DETECTOR_PARAMETER_KEYS order).")
    identity = nu.identity(art)
    identity["nuisance_tag"] = nuisance_tag
    if nu.fixed_vector_log10 is not None:
        emitter_log10 = np.asarray(nu.sample(1), dtype=float)    # every draw is the fixed vector
    elif nu.samples is not None:
        emitter_log10 = np.asarray(nu.samples, dtype=float)
    else:
        raise ValueError(f"{art.name}: choice {nu.posterior_sample_pool_choice!r} stores no vectors; "
                         f"a render needs a fixed vector or an empirical pool.")
    # Nuisance_DLI samples are in the detector table's estimator space (all-log rows); the same
    # table-bound conversion applies.
    emitter_abs = bio.to_physical(emitter_log10, det.DETECTOR_PARAMETERIZATION)
    if emitter_log10.shape[0] != 1:
        # A multi-vector artifact has no single "the" imaging vector; take its SGM so the
        # choice is the correlation-preserving one rather than an arbitrary row.
        span = np.ptp(emitter_abs, axis=0)
        span[span <= 0] = 1.0
        idx, _m = sample_geometric_median(emitter_abs, span)
        emitter_abs = emitter_abs[idx:idx + 1]
    emitter = emitter_abs[0]
    vec = np.concatenate([emitter, _scope_met()])
    n_pool = int(emitter_log10.shape[0])
    desc = (f"{'user-selected fixed' if nu.posterior_sample_pool_choice == 'selection_user' else 'calibrated'} "
            f"Nuisance_DLI vector ({art.name}"
            + ("" if n_pool == 1 else f", SGM of {n_pool} vectors")
            + (f", tag {nuisance_tag}" if nuisance_tag else ", canonical")
            + ") + MET SCOPE camera")
    return {"vector": vec, "description": desc, "identity": identity, "path": art}


def biology_fixed_imaging(data_bank_root, map_label, condition, nuisance_tag=None):
    """``(vector, description)`` of :func:`resolve_biology_imaging` (the horizon audit's call)."""
    img = resolve_biology_imaging(data_bank_root, map_label, condition, nuisance_tag=nuisance_tag)
    return img["vector"], img["description"]


def _ppv_spec(cfg, args):
    """Resolve the workflow-specific half: which block the MAP supplies, and how the other is fixed."""
    kind_token = KIND_OF_CONDITION.get(args.kind, args.kind)
    # The recording's condition selects the condition-specific estimator namespace (MAP
    # database, outputs) and, on the biology path, that condition's calibrated imaging.
    paths = cfg.paths.with_condition(kind_token)
    data_bank_root = PARAMETERS.machine.data_bank_root
    map_label = RunTiming(total_time_seconds=args.total_time_seconds,
                          frames=PARAMETERS.simulation.timing).label
    posit_dir = data_bank_root / paths.posit_subdir
    exp_out_dir = paths.experiment_recovery_dir(data_bank_root, map_label)

    S = dict(
        paths=paths, map_label=map_label,
        map_npz=exp_out_dir / (exp_out_dir.name + ".npz"),
        experimental_tif=paths.experiment_video_path(
            kind_token, args.cell, args.experiment_span_seconds, data_bank_root),
        out_dir=posit_dir / f"{paths.project_alias}_{map_label}_Posterior_Predictive_Video",
        prior_low=np.asarray(cfg.param_module.theta_lower_bound(), dtype=float),
        prior_high=np.asarray(cfg.param_module.theta_upper_bound(), dtype=float),
        table=parameter_table(cfg),                          # the MAP block's conversion rule
    )

    if cfg.tag == "detector":
        S["map_block"] = "imaging"
        S["map_keys"] = _wf_keys(cfg)                       # the 6 learnable imaging parameters
        S["imaging_desc"] = "MAP emitter parameters + MET SCOPE camera"
        S["rds_desc"] = ("full reactive system from the RDS nuisance (drawn from the biology prior, "
                         "pinned, or declared)")

        def imaging_physical(a, map_theta):
            if a.fixed_imaging_parameters:
                overrides = _parse_kv(a.set_imaging, "--set-imaging")
                return (_fixed_imaging_theta(overrides=overrides or None), "fixed MET values",
                        {"source": "fixed: MET camera, prior-center emitter parameters",
                         "set_imaging_overrides": overrides})
            # The camera is marginalized, not inferred: pin it to MET rather than drawing, because
            # the comparison is against one specific acquisition.
            return (np.concatenate([map_theta, _scope_met()]),
                    "MAP emitter parameters + MET SCOPE camera",
                    {"source": "map", "map_source": a.map_source})

        def rds_vector(a, map_theta, declared):
            if declared is not None:
                nuisance, record = declared
                desc, source = f"declared RDS configuration {record['scenario']['name']}", "declared"
            elif a.fixed_nuisance_rds:
                overrides = _parse_kv(a.fixed_nuisance_rds, "--fixed-nuisance-RDS")
                nuisance = _fixed_nuisance_physical(overrides=overrides)
                desc, source = "pinned RDS nuisance", "pinned"
                record = {"basis": "prior-center nominals with overrides", "overrides": overrides}
            else:
                # The RDS stage draws a tier's theta from default_rng(--seed); the same stream here.
                nuisance = _draw_nuisance_physical(np.random.default_rng(a.seed))
                desc, source, record = "drawn RDS nuisance", "drawn", {"basis": "one biology-prior draw"}
            # The nuisance vector IS a canonical theta (biology order), so the ordinary builders
            # apply under the recording's condition -- the same system that condition's RDS tier
            # simulates (association on under INLB, off under FAB).
            return nuisance, desc + " (full reactive system)", source, record
    else:
        S["map_block"] = "rds"
        S["map_keys"] = _wf_keys(cfg)                       # the 11 learnable RDS parameters
        S["imaging_desc"] = "calibrated Nuisance_DLI vector + MET SCOPE camera"
        S["rds_desc"] = "full reactive system from the MAP reaction-diffusion parameters"

        def imaging_physical(a, map_theta):
            # Biology holds imaging FIXED at the calibrated Nuisance_DLI + MET SCOPE vector, read
            # from the durable tier exactly as the biology DLI stage reads it (one definition, also
            # used by the horizon audit), with any --set-imaging overrides applied on top.
            img = resolve_biology_imaging(PARAMETERS.machine.root_for("EVAL"), map_label, kind_token,
                                          nuisance_tag=a.nuisance_tag)
            vec, desc, identity = img["vector"].copy(), img["description"], dict(img["identity"])
            overrides = _parse_kv(a.set_imaging, "--set-imaging")
            if overrides:
                find = {k: i for i, k in enumerate(det.DETECTOR_IMAGING_KEYS)}
                for k, v in overrides.items():
                    if k not in find:
                        raise ValueError(f"--set-imaging {k!r} is not an imaging parameter; "
                                         f"valid keys are {det.DETECTOR_IMAGING_KEYS}.")
                    vec[find[k]] = v
                desc += " [with --set-imaging overrides]"
                identity["set_imaging_overrides"] = overrides
            return vec, desc, identity

        def rds_vector(a, map_theta, declared):
            # The reaction-diffusion block IS the inference target here: the system is built from
            # the MAP -- or from a declared configuration when no estimate is to be used -- under the
            # recording's condition (its declared association setting).
            if a.fixed_nuisance_rds:
                raise SystemExit("--fixed-nuisance-RDS applies to the detector workflow only: here "
                                 "the reaction-diffusion block is the MAP, not a nuisance; a render "
                                 "at stated values takes --declared-rds.")
            if declared is not None:
                vec, record = declared
                desc, source = f"full reactive system from the declared configuration {record['scenario']['name']}", "declared"
            else:
                vec, record, desc, source = map_theta, {"map_source": a.map_source}, "full reactive system from the MAP", "map"
            return vec, desc, source, record

    S["imaging_physical"] = imaging_physical
    S["rds_vector"] = rds_vector
    return S


def _parse_kv(items, flag):
    out = {}
    for item in (items or []):
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"{flag} expects KEY=VALUE, got {item!r}")
        out[key.strip()] = float(val)
    return out
