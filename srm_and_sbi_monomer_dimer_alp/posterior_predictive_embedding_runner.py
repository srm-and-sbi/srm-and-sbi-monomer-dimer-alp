"""Posterior-predictive check in the embedding space (biology and detector workflows).

The embedding-distance analysis asks whether the experimental windows lie inside the cloud of
synthetic EVAL videos, which are drawn from the whole training prior. This analysis asks the sharper
question, one window at a time: *do videos simulated at the parameters the estimator assigns to a
window look, to the network, like that window?* For every experimental window it takes complete
joint draws from the window's stored posterior, simulates one video per draw through the production
chain, embeds the videos and the window through the trained network, and ranks the window's distance
to its nearest predictive neighbors against the same distance among the predictive videos
(``embedding_space_distance.knn_typicality``).

Three steps, each its own invocation, because they need different machines and the middle one is the
expensive one:

* ``plan``     -- select the draws and fix every parameter of every video (CPU, seconds);
* ``simulate`` -- simulate and render the videos of the plan (CPU; resumable, split with ``--shard``);
* ``analyze``  -- embed, rank, report (GPU for the embedding).

What a video of the plan is, in each workflow:

* **detector** -- the six imaging parameters are one posterior draw of the window. Everything the
  detector marginalizes is drawn as the training data draws it: the five camera parameters from the
  acquisition record's box, the reaction-diffusion parameters from the condition's biology prior.
* **biology** -- the reaction-diffusion parameters are one posterior draw of the window. The imaging
  is what the biology training data used: the six emitter parameters from the condition's
  ``Nuisance_DLI`` artifact, the five camera parameters from the acquisition record's box.

So the predictive cloud of a window is the model's distribution of videos given the window's
*marginal* posterior of the inferred block, AVERAGED OVER THE TRAINING DISTRIBUTION OF THE NUISANCES.
That is what the check tests. It does not reproduce the unknown nuisance settings of that window
(for the detector: the receptor number and the mobility, which shape a video strongly). The
detector's predictive clouds are therefore broad, and "inside its own cloud" is a weaker statement
there than in the biology workflow, where every parameter that shapes the trajectories is inferred.

How to read the ranks. A window's tail share in its own predictive cloud is NOT a calibrated
p-value: the window determined the posterior its comparison videos were drawn at, so the window and
those videos are not exchangeable, and no flag rate (such as "5 % of the windows under a correct
model") follows. Posterior-predictive ranks of this kind tend to sit away from the tails when the
model is right. A low rank is a warning about the COMBINATION of forward model and inference
procedure (the posterior may be wrong although the simulator is right, or the reverse); it does not
say which. Likewise, predictive videos that lie in a thin region of the synthetic EVAL set need not
indicate a fault: a posterior can concentrate, with valid parameters, where the prior put few
videos.

The window layout (frames per window, step between window starts, recording span) is read from the
source of the posterior draws and cannot be overridden: chunk ``w`` of a recording must be the
window the posterior of chunk ``w`` was computed from.

Each video is a fresh window of the estimator's length, as the training videos are, stored through
the training data's conversion (``io.convert_video_dtype``, 16 to 8 bit), so the network sees it
exactly as it sees a training video. Videos are simulated unseeded, as the training data are; the
plan fixes and records every parameter, so nothing depends on reproducing a random stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from . import __version__ as _PACKAGE_VERSION
from . import artifact_schema as schema
from . import detector_parameterization as det
from . import embedding_space_distance as esd
from . import experiment_aggregation as agg
from . import parameterization as bio
from .diagnostics import DiagnosticReporter
from .embedding_space_distance_runner import (_chunk_geometry, _disp, _esd_spec, _load_posterior,
                                              build_parser as _distance_parser, resolve_kinds)
from .experiment_support import read_cell_chunks
from .io import convert_video_dtype
from .labeling import LABELING_CONDITIONS, resolve_labeling
from .parameterization import PARAMETERS
from .workflow import prior_bounds

STEPS = ("plan", "simulate", "analyze")
DEFAULT_SPAN_SECONDS = int(_distance_parser("").get_default("experiment_span_seconds"))   # the project's recordings
POOLS = ("bounded", "unrestricted")
_BOX_TOLERANCE = 1e-9


# =================================================================================================
# Pure parts: draw selection, readings
# =================================================================================================

def select_draws(cloud, n, low, high, linear, pool, rng):
    """Choose ``n`` complete joint draws of one window's cloud, without replacement.

    ``cloud`` is ``(draws, P)`` in estimator coordinates; ``low``/``high`` the prior box there;
    ``linear`` marks the rows stored on a linear scale (fractions). ``pool="bounded"`` admits only draws
    inside the box in every coordinate: the simulator's training tiers hold no system outside it, and
    posterior mass outside the box is the flow leaking past the prior's support. ``pool="unrestricted"``
    admits draws outside the box on the log rows (the render then extrapolates) but never on a linear
    row, where a value outside ``[0, 1]`` is not a realizable composition. Rows are taken whole, never
    recombined per coordinate.

    Returns ``(rows (m, P), row_index (m,), n_eligible)`` with ``m = min(n, n_eligible)``.
    """
    if pool not in POOLS:
        raise ValueError(f"pool={pool!r}; expected one of {POOLS}.")
    cloud = np.asarray(cloud, dtype=float)
    low, high, linear = np.asarray(low, float), np.asarray(high, float), np.asarray(linear, bool)
    inside = (cloud >= low - _BOX_TOLERANCE) & (cloud <= high + _BOX_TOLERANCE)
    eligible = inside.all(axis=1) if pool == "bounded" else inside[:, linear].all(axis=1)
    index = np.flatnonzero(eligible)
    take = np.sort(rng.choice(index, size=min(int(n), index.size), replace=False))
    return cloud[take], take, int(index.size)


def window_reading(posterior_tail, prior_tail, alpha):
    """Plain reading of one window from its two tail shares (``prior_tail`` None when no synthetic
    reference cloud was given)."""
    in_posterior = posterior_tail >= alpha
    if prior_tail is None or np.isnan(prior_tail):
        return "inside its predictive cloud" if in_posterior else "outside its predictive cloud"
    in_prior = prior_tail >= alpha
    if in_prior and in_posterior:
        return "inside both"
    if in_prior:
        return "inside the EVAL cloud, outside its predictive cloud"
    if in_posterior:
        return "outside the EVAL cloud, inside its predictive cloud"
    return "outside both"


def plan_geometry(contract, timing, chunk_step_seconds, experiment_span_seconds):
    """The window layout of a plan: ``(n_frames, step_frames, span_frames, span_seconds)`` as the
    source of the posterior draws records it (``contract["window_geometry"]``). The layout is not a
    choice of this analysis: chunk ``w`` of a recording has to be the window whose posterior is stored
    as chunk ``w``. A source without a recorded layout, a layout of another window length than the
    estimator's, or a ``--chunk-step-seconds`` / ``--experiment-span-seconds`` that contradicts the
    record is refused. An omitted option follows the record: ``experiment_span_seconds=None`` returns the
    stored span in seconds (the project's default span when the source records none)."""
    geom = contract.get("window_geometry") or {}
    if not geom.get("n_frames") or not geom.get("step_frames"):
        raise SystemExit("the source of the posterior draws records no window geometry (n_frames, step_frames); "
                         "the windows its posteriors belong to cannot be identified.")
    n_frames, step_frames, span = int(geom["n_frames"]), int(geom["step_frames"]), geom.get("span_frames")
    if n_frames != int(timing.frame_count):
        raise SystemExit(f"the posterior draws were computed on windows of {n_frames} frames; the estimator of this "
                         f"run reads {int(timing.frame_count)}.")
    if chunk_step_seconds is not None:
        asked = int(_chunk_geometry(timing, chunk_step_seconds)[1])
        if asked != step_frames:
            raise SystemExit(f"--chunk-step-seconds {chunk_step_seconds} is a step of {asked} frames; the posterior "
                             f"draws were computed on windows stepped by {step_frames} frames. The plan follows the "
                             f"stored layout: omit the option.")
    if span is None:
        span_seconds = DEFAULT_SPAN_SECONDS if experiment_span_seconds is None else int(experiment_span_seconds)
        return n_frames, step_frames, None, span_seconds
    stored_seconds = float(span) * float(timing.total_time_seconds) / n_frames
    if experiment_span_seconds is None:
        if abs(stored_seconds - round(stored_seconds)) > 1e-9:
            raise SystemExit(f"the posterior draws were computed on recordings of {int(span)} frames "
                             f"({stored_seconds:g} s), not a whole number of seconds; the recording files are "
                             f"named by whole seconds.")
        return n_frames, step_frames, int(span), int(round(stored_seconds))
    asked_span = int(round(float(experiment_span_seconds) * n_frames / float(timing.total_time_seconds)))
    if asked_span != int(span):
        raise SystemExit(f"--experiment-span-seconds {experiment_span_seconds} is {asked_span} frames; the "
                         f"posterior draws were computed on recordings of {int(span)} frames "
                         f"({stored_seconds:g} s). The plan follows the stored span: omit the option.")
    return n_frames, step_frames, int(span), int(experiment_span_seconds)


PIXEL_STATISTICS = ("pixel median", "pixel 99th percentile", "pixel 99.9th percentile", "pixel maximum",
                    "share of pixels at the top of the range", "frame-to-frame variation of the mean")


def pixel_statistics(videos):
    """Pixel-level summaries of stored videos ``(n, T, H, W)``, one row per video, in the order of
    ``PIXEL_STATISTICS``: three quantiles and the maximum of the pixel values, the share of pixels at
    the top of the stored range (255 for 8-bit videos: clipped), and the coefficient of variation over
    the frames of the frame's mean pixel value. They are computed on the videos as the network
    receives them, and complement the embedding, which need not retain every difference."""
    arr = np.asarray(videos)
    if arr.ndim == 3:
        arr = arr[None]
    top = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else float(arr.max())
    flat = arr.reshape(arr.shape[0], -1).astype(np.float64)
    q = np.quantile(flat, [0.5, 0.99, 0.999], axis=1)
    frame_mean = arr.reshape(arr.shape[0], arr.shape[1], -1).mean(axis=2, dtype=np.float64)
    mean = frame_mean.mean(axis=1)
    variation = np.divide(frame_mean.std(axis=1), mean, out=np.zeros_like(mean), where=mean > 0)
    return np.stack([q[0], q[1], q[2], flat.max(axis=1), (flat >= top).mean(axis=1), variation], axis=1)


def pixel_comparison(window_stats, cloud_stats):
    """Where a window's pixel statistics lie among its predictive videos' ``(S, n_statistics)``:
    ``(low, median, high, below)`` per statistic, the 5th, 50th and 95th percentile of the predictive
    values and the share of predictive videos whose value is below the window's. Descriptive."""
    cloud = np.asarray(cloud_stats, float)
    low, median, high = np.quantile(cloud, [0.05, 0.5, 0.95], axis=0)
    return low, median, high, (cloud < np.asarray(window_stats, float)[None, :]).mean(axis=0)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _window_name(kind, cell, chunk):
    return f"Window_{kind}_Cell_{int(cell)}_Chunk_{int(chunk)}"


# =================================================================================================
# Paths and sources
# =================================================================================================

def _spec(cfg, args):
    """The distance analysis' resolution of the workflow (estimator, product label, paths, keys, table)
    plus this analysis' own directory and the default sources."""
    R = _esd_spec(cfg, args)
    alias, label = R["paths"].project_alias, R["product_label"]
    name = f"{alias}_{label}_Posterior_Predictive_Embedding" + (f"_{args.run_label}" if args.run_label else "")
    R["out_dir"] = R["posit_dir"] / name
    R["plan_path"] = R["out_dir"] / "plan.npz"
    R["video_dir"] = R["out_dir"] / "videos"
    exp_dir = R["paths"].experiment_recovery_dir(R["data_bank_root"], label)
    R["draws_source"] = Path(args.draws_source) if args.draws_source else exp_dir / (exp_dir.name + ".npz")
    R["reference_embeddings"] = (Path(args.reference_embeddings) if args.reference_embeddings
                                 else R["report_dir"] / "embeddings.npz")
    low, high = prior_bounds(cfg, args.condition)
    R["prior_low"], R["prior_high"] = np.asarray(low, float), np.asarray(high, float)
    R["linear"] = np.array([not bio.is_log_row(e) for e in R["parameterization"]], dtype=bool)
    return R


def load_draws(path):
    """The retained window clouds: an Experiment product that stored them, or a retained draw
    collection (``*_Window_Posterior_Draws*.npz``). Returns an ``experiment_aggregation.CloudCollection``."""
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as data:
        is_collection = "collection_manifest_json" in data.files
    if is_collection:
        return agg.cloud_from_collection(path)
    arrays, manifest = schema.load_product(path, stage="experiment")
    return agg.cloud_from_product(arrays, manifest, path=path, label=str(path))


# =================================================================================================
# Step 1: plan
# =================================================================================================

def _nuisance_blocks(cfg, R, args, theta_physical, rng):
    """The full parameter set of every video: ``(imaging_physical (N, 11), rds_physical (N, P_rds),
    description, record)`` for the ``N`` rows of ``theta_physical`` (the posterior draws, physical)."""
    condition = R["condition"]
    n = theta_physical.shape[0]
    scope = np.power(10.0, rng.uniform(np.array(det.scope_lower_bound()), np.array(det.scope_upper_bound()),
                                       size=(n, len(det.DETECTOR_SCOPE_KEYS))))
    if list(det.DETECTOR_IMAGING_KEYS) != list(det.DETECTOR_PARAMETER_KEYS) + list(det.DETECTOR_SCOPE_KEYS):
        raise RuntimeError("the imaging render vector is no longer the six emitter keys followed by the "
                           "five camera keys; this analysis assembles it in that order.")
    rds_table = bio.parameterization_of(condition)
    if cfg.tag == "detector":
        if list(R["parameter_keys"]) != list(det.DETECTOR_PARAMETER_KEYS):
            raise RuntimeError(f"detector parameter keys {R['parameter_keys']} != {det.DETECTOR_PARAMETER_KEYS}.")
        low = np.asarray(bio.theta_lower_bound(condition), float)
        high = np.asarray(bio.theta_upper_bound(condition), float)
        rds = np.asarray(bio.to_physical(rng.uniform(low, high, size=(n, low.size)), rds_table), float)
        imaging = np.concatenate([theta_physical, scope], axis=1)
        description = ("imaging: the window's posterior draw; camera: drawn from the acquisition record's "
                       "box; reaction-diffusion: drawn from the condition's biology prior")
        record = {"emitter_source": "posterior draw", "nuisance_artifact": None}
    else:
        if list(R["parameter_keys"]) != list(bio.parameter_keys_of(condition)):
            raise RuntimeError(f"biology parameter keys {R['parameter_keys']} != "
                               f"{list(bio.parameter_keys_of(condition))}.")
        from .detector_nuisance_dli import artifact_path, require_nuisance_dli
        det_paths = det.detector_paths(PARAMETERS.paths).with_condition(condition)
        posit_dir = PARAMETERS.machine.root_for("EVAL") / det_paths.posit_subdir
        nuisance = require_nuisance_dli(posit_dir, det_paths.project_alias, R["timing_label"],
                                        nuisance_tag=args.nuisance_tag)
        art = artifact_path(posit_dir, det_paths.project_alias, R["timing_label"], nuisance_tag=args.nuisance_tag)
        if list(nuisance.parameter_keys) != list(det.DETECTOR_PARAMETER_KEYS):
            raise ValueError(f"Nuisance_DLI parameter_keys {list(nuisance.parameter_keys)} != "
                             f"{det.DETECTOR_PARAMETER_KEYS}.")
        emitter = np.power(10.0, np.asarray(nuisance.sample(n, seed=int(rng.integers(2 ** 31))), float))
        imaging = np.concatenate([emitter, scope], axis=1)
        rds = np.asarray(theta_physical, float)
        description = ("reaction-diffusion: the window's posterior draw; emitter: the condition's "
                       "Nuisance_DLI artifact, sampled as the training data sample it; camera: drawn from "
                       "the acquisition record's box")
        identity = nuisance.identity(art)
        identity["nuisance_tag"] = args.nuisance_tag
        record = {"emitter_source": "Nuisance_DLI artifact", "nuisance_artifact": identity}
    return imaging, rds, description, record


def run_plan(cfg, args, R):
    if R["plan_path"].exists():
        raise SystemExit(f"{R['plan_path']} exists; a plan is never replaced. Another selection needs its "
                         f"own --run-label.")
    if not R["draws_source"].exists():
        raise FileNotFoundError(f"source of the posterior draws not found:\n    {R['draws_source']}\n"
                                f"Run the Experiment stage with retained clouds, or pass --draws-source "
                                f"(an Experiment product or a *_Window_Posterior_Draws*.npz collection).")
    coll = load_draws(R["draws_source"])
    if list(coll.keys) != list(R["parameter_keys"]):
        raise SystemExit(f"the draws carry the keys {coll.keys}; this run's estimator has "
                         f"{R['parameter_keys']}.")
    kinds = resolve_kinds(args)
    rng = np.random.default_rng(args.seed)
    rows, row_index, identity, eligible = [], [], [], []
    for w in range(coll.n_windows):
        kind = coll.kinds[int(coll.kind_index[w])]
        if kind not in kinds:
            continue
        if args.max_cells > 0:
            cells_of_kind = sorted({int(c) for c, ki in zip(coll.cell, coll.kind_index)
                                    if coll.kinds[int(ki)] == kind})
            if int(coll.cell[w]) not in cells_of_kind[:args.max_cells]:
                continue
        take, index, n_eligible = select_draws(coll.cloud[w], args.draws_per_window, R["prior_low"],
                                               R["prior_high"], R["linear"], args.pool, rng)
        if take.shape[0] < args.draws_per_window:
            raise SystemExit(
                f"{_window_name(kind, coll.cell[w], coll.chunk[w])}: only {n_eligible} of "
                f"{coll.cloud.shape[1]} draws are admitted by pool={args.pool}; "
                f"--draws-per-window {args.draws_per_window} cannot be met. Lower it, or use "
                f"--pool unrestricted and say so in the reading.")
        rows.append(take); row_index.append(index); eligible.append(n_eligible)
        identity.append((kind, int(coll.cell[w]), int(coll.chunk[w])))
    if not rows:
        raise SystemExit(f"no window of {[_disp(k) for k in kinds]} in {R['draws_source']}.")
    theta = np.stack(rows)                                              # (W, S, P) estimator coordinates
    W, S, P = theta.shape
    physical = np.asarray(bio.to_physical(theta.reshape(W * S, P), R["parameterization"]), float)
    imaging, rds, description, record = _nuisance_blocks(cfg, R, args, physical, rng)
    n_frames, step_frames, span_frames, span_seconds = plan_geometry(
        coll.contract, R["timing"], args.chunk_step_seconds, args.experiment_span_seconds)
    manifest = {
        "analysis": "posterior_predictive_embedding", "workflow": cfg.tag, "condition": R["condition"],
        "product_label": R["product_label"], "package_version": _PACKAGE_VERSION,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "draws_source": {"path": str(R["draws_source"]), "sha256": _sha256(R["draws_source"]),
                         "pool_mode_of_source": coll.contract.get("pool_mode"),
                         "n_draws_per_window_in_source": int(coll.n_draws)},
        "checkpoint_sha256": coll.contract.get("checkpoint_sha256"),
        "pool": args.pool, "draws_per_window": int(S), "seed": args.seed,
        "n_frames": int(n_frames), "step_frames": int(step_frames), "span_frames": span_frames,
        "window_geometry_source": "the window_geometry recorded by the source of the posterior draws",
        "n_draws_in_source": int(coll.n_draws),
        "experiment_span_seconds": int(span_seconds),
        "video": "a fresh window of the estimator's length per draw, stored through io.convert_video_dtype "
                 "(16 to 8 bit), simulated unseeded",
        "parameters": description, "nuisance": record,
        "labeling": resolve_labeling(R["condition"], None, None).describe(),
    }
    R["out_dir"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(R["plan_path"]),
        theta_estimator=theta.astype(np.float64), theta_physical=physical.reshape(W, S, P),
        imaging_physical=imaging.reshape(W, S, -1), rds_physical=rds.reshape(W, S, -1),
        kind=np.array([i[0] for i in identity]), cell=np.array([i[1] for i in identity], dtype=np.int64),
        chunk=np.array([i[2] for i in identity], dtype=np.int64),
        row_index=np.stack(row_index).astype(np.int64), n_eligible=np.array(eligible, dtype=np.int64),
        parameter_keys=np.array(R["parameter_keys"]), imaging_keys=np.array(det.DETECTOR_IMAGING_KEYS),
        rds_keys=np.array(list(bio.parameter_keys_of(R["condition"]))),
        manifest_json=json.dumps(manifest, default=str))
    share = np.array(eligible) / float(coll.n_draws)
    print(f"plan: {W} window(s) x {S} draw(s) = {W * S} video(s) of {n_frames} frames, windows stepped by "
          f"{step_frames} frames (the layout of the posterior draws); pool={args.pool} (share of each window's "
          f"cloud excluded: max {1 - share.min():.3f}, median {1 - np.median(share):.3f}).")
    print(f"  {description}")
    print(f"plan written:\n    {R['plan_path']}")
    return 0


def load_plan(path):
    with np.load(str(path), allow_pickle=False) as data:
        plan = {k: data[k] for k in data.files}
    plan["manifest"] = json.loads(str(plan.pop("manifest_json")))
    return plan


# =================================================================================================
# Step 2: simulate
# =================================================================================================

def _parse_shard(text):
    if not text:
        return 0, 1
    try:
        i, n = (int(part) for part in text.split("/"))
    except ValueError:
        raise SystemExit(f"--shard {text!r}: expected i/n, e.g. 0/8.")
    if not 0 <= i < n:
        raise SystemExit(f"--shard {text!r}: needs 0 <= i < n.")
    return i, n


def window_video_path(video_dir, kind, cell, chunk):
    return Path(video_dir) / f"{_window_name(kind, cell, chunk)}.npz"


def require_plan(video_file, plan_sha256):
    """Refuse a window file that was not simulated from the plan with this checksum. The checksum
    covers the whole plan (the inferred draws AND the nuisance draws of every video), so videos of
    another selection or of other nuisance draws cannot enter under the same window name."""
    with np.load(str(video_file), allow_pickle=False) as data:
        recorded = str(data["plan_sha256"]) if "plan_sha256" in data.files else ""
    if recorded != plan_sha256:
        raise SystemExit(f"{Path(video_file).name} was simulated from another plan (plan checksum "
                         f"{recorded[:12] or 'not recorded'}, this plan {plan_sha256[:12]}). Videos are never "
                         f"mixed across plans: use the folder of that plan, or a new --run-label.")


def run_simulate(cfg, args, R, renderer=None):
    """Simulate the plan's videos, one file per window. ``renderer`` replaces the production chain in
    tests; it is called as ``renderer(condition, rds_vector, imaging_physical, labeling_plan, n_frames,
    trajectory_path)`` and returns a dict with ``synth`` ``(n_frames, H, W)``."""
    if not R["plan_path"].exists():
        raise FileNotFoundError(f"no plan at {R['plan_path']}; run --step plan first.")
    plan = load_plan(R["plan_path"])
    if renderer is None:
        from .posterior_predictive_video_runner import simulate_and_render
        renderer = simulate_and_render
    condition = R["condition"]
    labeling_plan = resolve_labeling(condition, None, None)
    n_frames = int(plan["manifest"]["n_frames"])
    shard, n_shards = _parse_shard(args.shard)
    R["video_dir"].mkdir(parents=True, exist_ok=True)
    W, S = plan["theta_estimator"].shape[:2]
    plan_sha256 = _sha256(R["plan_path"])
    done = skipped = 0
    for w in range(W):
        if w % n_shards != shard:
            continue
        out = window_video_path(R["video_dir"], plan["kind"][w], plan["cell"][w], plan["chunk"][w])
        if out.exists():                                # a finished window is kept, never simulated again
            require_plan(out, plan_sha256)              # ... if it is a window of THIS plan
            skipped += 1
            continue
        videos, n_dyes, n_subunits = [], [], []
        with tempfile.TemporaryDirectory(dir=str(R["video_dir"]), prefix="trajectories_") as scratch:
            for s in range(S):
                trajectory = Path(scratch) / f"trajectory_{w}_{s}.h5"
                clip = renderer(condition, plan["rds_physical"][w, s], plan["imaging_physical"][w, s],
                                labeling_plan, n_frames, trajectory)
                # The training data's storage step, on the render as it leaves the renderer.
                videos.append(convert_video_dtype(np.asarray(clip["synth"]), bits_from=16, bits_to=8))
                n_dyes.append(int(np.sum(clip.get("dye_counts", 0))))
                n_subunits.append(int(clip.get("n_subunits", 0)))
                if trajectory.exists():
                    trajectory.unlink()                 # this run's own scratch file
        stack = np.stack(videos)
        if stack.shape[1] != n_frames or stack.dtype != np.uint8:
            raise RuntimeError(f"{out.name}: videos of shape {stack.shape}, dtype {stack.dtype}; expected "
                               f"{n_frames} frames, uint8.")
        partial = out.with_name(out.stem + ".partial.npz")
        np.savez_compressed(str(partial), videos=stack, n_dyes=np.array(n_dyes), n_subunits=np.array(n_subunits),
                            theta_estimator=plan["theta_estimator"][w], imaging_physical=plan["imaging_physical"][w],
                            rds_physical=plan["rds_physical"][w], kind=str(plan["kind"][w]),
                            cell=int(plan["cell"][w]), chunk=int(plan["chunk"][w]),
                            plan_sha256=plan_sha256)
        os.replace(partial, out)                        # a window file appears only when it is complete
        done += 1
        print(f"  [{done}] {out.name}: {S} video(s) written.", flush=True)
    print(f"simulate (shard {shard}/{n_shards}): {done} window(s) written, {skipped} already present, "
          f"under\n    {R['video_dir']}")
    return 0


# =================================================================================================
# Step 3: analyze
# =================================================================================================

def _load_reference(R, estimator_sha256):
    """The synthetic EVAL embeddings stored by the embedding-distance analysis for the SAME estimator,
    or ``(None, reason)``."""
    path = R["reference_embeddings"]
    if not path.exists():
        return None, f"no stored embeddings at {path}"
    with np.load(str(path), allow_pickle=False) as data:
        recorded = str(data["estimator_sha256"])
        synthetic = np.asarray(data["synthetic"], dtype=float)
    if recorded != str(estimator_sha256):
        raise SystemExit(f"{path}: embeddings of another estimator (weights {recorded[:12]}, this run "
                         f"{str(estimator_sha256)[:12]}); a reference cloud is only comparable within one "
                         f"network.")
    return synthetic, str(path)


def predictive_statistics(window_emb, cloud_emb, reference_emb, *, k_cloud, k_reference):
    """Tail shares of one experimental window: against its own predictive cloud (raw, and standardized
    with the scale pooled over the cloud and the window, so that the window is treated like the
    videos) and, when a reference cloud is given, against the synthetic EVAL set; plus the share of
    the predictive videos that lie in the thin outskirts of that reference (their own tail share
    below 0.05)."""
    out = {}
    for name, standardize in (("raw", False), ("standardized", "pooled")):
        res = esd.knn_typicality(cloud_emb, window_emb[None, :], k=k_cloud, standardize=standardize)
        out[f"posterior_tail_{name}"] = float(res["tail_share"][0])
        out[f"posterior_distance_{name}"] = float(res["query_distance"][0])
    out["prior_tail_raw"] = out["cloud_far_from_reference"] = float("nan")
    if reference_emb is not None:
        ref = esd.knn_typicality(reference_emb, np.vstack([window_emb[None, :], cloud_emb]), k=k_reference)
        out["prior_tail_raw"] = float(ref["tail_share"][0])
        out["cloud_far_from_reference"] = float(np.mean(ref["tail_share"][1:] < 0.05))
    return out


WINDOW_HEADERS = ["group", "recording", "window", "posterior draws excluded by the pool",
                  "tail share in its predictive cloud (raw)",
                  "tail share in its predictive cloud (standardized)", "tail share in the EVAL cloud (raw)",
                  "predictive videos in the outskirts of the EVAL cloud", "reading"]


def predictive_tables(kinds, cells, chunks, stats, alpha, excluded=None):
    """``(window_rows, group_rows, readings)``: one row per window, and per group the count of each
    reading. ``excluded`` is each window's share of posterior draws the pool did not admit."""
    readings = [window_reading(s["posterior_tail_raw"], s["prior_tail_raw"], alpha) for s in stats]

    def _f(value, spec=".3g"):
        return "n/a" if np.isnan(value) else format(value, spec)

    excluded = [float("nan")] * len(stats) if excluded is None else excluded
    window_rows = [[_disp(kinds[i]), str(int(cells[i])), str(int(chunks[i])), _f(excluded[i], ".3f"),
                    _f(stats[i]["posterior_tail_raw"]), _f(stats[i]["posterior_tail_standardized"]),
                    _f(stats[i]["prior_tail_raw"]), _f(stats[i]["cloud_far_from_reference"], ".2f"), readings[i]]
                   for i in range(len(stats))]
    group_rows = []
    for key in dict.fromkeys(list(kinds)):
        sel = [i for i in range(len(stats)) if kinds[i] == key]
        counts = {}
        for i in sel:
            counts[readings[i]] = counts.get(readings[i], 0) + 1
        outside = sum(stats[i]["posterior_tail_raw"] < alpha for i in sel)
        outside_std = sum(stats[i]["posterior_tail_standardized"] < alpha for i in sel)
        group_rows.append([_disp(key), str(len(sel)), str(int(outside)), str(int(outside_std)),
                           "; ".join(f"{name}: {n}" for name, n in sorted(counts.items()))])
    return window_rows, group_rows, readings


def pixel_tables(kinds, cells, chunks, window_pixels, cloud_pixels):
    """``(window_rows, group_rows)`` of the pixel-level comparison: per window and statistic the
    window's value, the predictive videos' median and 5-95 % range and the share of predictive videos
    below the window; per group and statistic the medians and the number of windows outside that range."""
    window_rows, outside = [], {}
    values = {}
    for i in range(len(kinds)):
        low, median, high, below = pixel_comparison(window_pixels[i], cloud_pixels[i])
        for j, name in enumerate(PIXEL_STATISTICS):
            out = bool(window_pixels[i][j] < low[j] or window_pixels[i][j] > high[j])
            window_rows.append([_disp(kinds[i]), str(int(cells[i])), str(int(chunks[i])), name,
                                f"{window_pixels[i][j]:.4g}", f"{median[j]:.4g}", f"{low[j]:.4g}",
                                f"{high[j]:.4g}", f"{below[j]:.2f}", "outside" if out else "inside"])
            outside[(kinds[i], name)] = outside.get((kinds[i], name), 0) + int(out)
            values.setdefault((kinds[i], name), []).append((window_pixels[i][j], median[j]))
    group_rows = []
    for key in dict.fromkeys(list(kinds)):
        n = sum(1 for k in kinds if k == key)
        for name in PIXEL_STATISTICS:
            pairs = np.array(values[(key, name)])
            group_rows.append([_disp(key), name, f"{np.median(pairs[:, 0]):.4g}", f"{np.median(pairs[:, 1]):.4g}",
                               f"{outside[(key, name)]} of {n}"])
    return window_rows, group_rows


PIXEL_HEADERS = ["group", "recording", "window", "statistic", "experimental window", "predictive median",
                 "predictive 5th percentile", "predictive 95th percentile",
                 "share of predictive videos below the window", "window against the 5-95 % range"]


def _figure_projection(kinds, cells, window_emb, cloud_emb, reference_emb, rng):
    """One panel per recording group, on two principal axes of the reference cloud (of the pooled
    predictive videos when there is none): the reference in gray, each recording's predictive videos as
    small dots, its experimental windows as outlined markers of the same color."""
    basis_points = reference_emb if reference_emb is not None else np.vstack(cloud_emb)
    center = basis_points.mean(axis=0)
    _, S, Vt = np.linalg.svd(basis_points - center, full_matrices=False)
    ev = (S ** 2) / np.sum(S ** 2)

    def _project(E):
        return (np.asarray(E, float) - center) @ Vt[:2].T

    labels = list(dict.fromkeys(list(kinds)))
    fig = Figure(figsize=(6.2 * len(labels), 5.4), layout="constrained")
    axes = np.atleast_1d(fig.subplots(1, len(labels), sharex=True, sharey=True))
    for ax, key in zip(axes, labels):
        if reference_emb is not None:
            ref = reference_emb if len(reference_emb) <= 3000 else reference_emb[
                rng.choice(len(reference_emb), 3000, replace=False)]
            pts = _project(ref)
            ax.scatter(pts[:, 0], pts[:, 1], s=4, color="0.8", edgecolors="none", label="synthetic EVAL", zorder=1)
        recordings = sorted({int(cells[i]) for i in range(len(kinds)) if kinds[i] == key})
        for j, cell in enumerate(recordings):
            color = f"C{j % 10}"
            sel = [i for i in range(len(kinds)) if kinds[i] == key and int(cells[i]) == cell]
            pts = _project(np.vstack([cloud_emb[i] for i in sel]))
            ax.scatter(pts[:, 0], pts[:, 1], s=5, color=color, alpha=0.35, edgecolors="none", zorder=2)
            win = _project(np.vstack([window_emb[i] for i in sel]))
            ax.scatter(win[:, 0], win[:, 1], s=60, color=color, edgecolors="black", linewidths=1.0,
                       marker="D", zorder=3, label=f"recording {cell}" if j < 10 else None)
        ax.set_title(f"{_disp(key)}: predictive videos (dots) and experimental windows (diamonds)")
        ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% of the basis cloud's variance)")
        ax.legend(fontsize=7, markerscale=0.8)
    axes[0].set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    return fig


def run_analyze(cfg, args, R, embedder=None, window_reader=None):
    """Embed, rank and report. ``embedder(videos) -> [n, D]`` and ``window_reader(kind, cell) -> list of
    windows`` replace the trained network and the recordings in tests."""
    if not R["plan_path"].exists():
        raise FileNotFoundError(f"no plan at {R['plan_path']}; run --step plan first.")
    plan = load_plan(R["plan_path"])
    manifest = plan["manifest"]
    n_frames, step_frames = int(manifest["n_frames"]), int(manifest["step_frames"])
    W = plan["theta_estimator"].shape[0]
    files = [window_video_path(R["video_dir"], plan["kind"][w], plan["cell"][w], plan["chunk"][w]) for w in range(W)]
    missing = [f.name for f in files if not f.exists()]
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} of {W} window file(s) are missing under {R['video_dir']} (first: "
                         f"{missing[0]}). Finish --step simulate, or pass --allow-partial to report the "
                         f"windows present.")
    present = [w for w in range(W) if files[w].exists()]
    if not present:
        raise SystemExit(f"no simulated window under {R['video_dir']}.")

    estimator_sha256 = ""
    if embedder is None:
        posterior, device = _load_posterior(R)
        estimator_sha256 = getattr(posterior, "weights_sha256", "")
        if manifest.get("checkpoint_sha256") and estimator_sha256 != manifest["checkpoint_sha256"]:
            raise SystemExit(f"the draws of the plan came from the estimator with weights "
                             f"{manifest['checkpoint_sha256'][:12]}; the estimator loaded now has "
                             f"{estimator_sha256[:12]}. The check is only meaningful within one network.")

        def embedder(videos):
            return esd.embed_videos(posterior, videos, device=str(device), expected_frames=n_frames)
    if window_reader is None:
        def window_reader(kind, cell):
            tif = R["paths"].experiment_video_path(kind, cell, int(manifest["experiment_span_seconds"]),
                                                   R["data_bank_root"])
            if not tif.exists():
                raise FileNotFoundError(f"experimental recording not found:\n    {tif}")
            return read_cell_chunks(tif, n_frames, step_frames)

    plan_sha256 = _sha256(R["plan_path"])
    n_source = int(manifest.get("n_draws_in_source") or manifest["draws_source"]["n_draws_per_window_in_source"])
    reference_emb, reference_source = _load_reference(R, estimator_sha256)
    if args.reference_embeddings and reference_emb is None:
        raise SystemExit(f"--reference-embeddings: {reference_source}.")

    kinds = [str(plan["kind"][w]) for w in present]
    cells = [int(plan["cell"][w]) for w in present]
    chunks = [int(plan["chunk"][w]) for w in present]
    S = plan["theta_estimator"].shape[1]
    if not 1 <= args.knn_k_cloud <= S - 1:
        raise SystemExit(f"--knn-k-cloud {args.knn_k_cloud} is not in [1, {S - 1}] for {S} videos per window.")
    excluded = [1.0 - float(plan["n_eligible"][w]) / n_source for w in present]
    window_emb, cloud_emb, stats, recordings = [], [], [], {}
    window_pixels, cloud_pixels, recording_pixels = [], [], {}
    for i, w in enumerate(present):
        key = (kinds[i], cells[i])
        if key not in recordings:
            windows = window_reader(*key)                               # every window of the recording
            recordings[key] = embedder(windows)
            recording_pixels[key] = pixel_statistics(np.stack([np.asarray(v) for v in windows]))
        if chunks[i] >= len(recordings[key]):
            raise SystemExit(f"{_window_name(*key, chunks[i])}: the recording has {len(recordings[key])} "
                             f"window(s) at this geometry.")
        require_plan(files[w], plan_sha256)                             # the whole plan, nuisances included
        with np.load(str(files[w]), allow_pickle=False) as data:
            videos = data["videos"]
            if not np.array_equal(data["theta_estimator"], plan["theta_estimator"][w]):
                raise SystemExit(f"{files[w].name} was simulated from another plan.")
        cloud = np.asarray(embedder(videos), dtype=float)
        window = np.asarray(recordings[key][chunks[i]], dtype=float)
        window_emb.append(window); cloud_emb.append(cloud)
        window_pixels.append(recording_pixels[key][chunks[i]]); cloud_pixels.append(pixel_statistics(videos))
        stats.append(predictive_statistics(window, cloud, reference_emb, k_cloud=args.knn_k_cloud,
                                           k_reference=args.knn_k))
        print(f"  [{i + 1}/{len(present)}] {files[w].stem}: tail share in its predictive cloud "
              f"{stats[-1]['posterior_tail_raw']:.3g}", flush=True)

    window_rows, group_rows, readings = predictive_tables(kinds, cells, chunks, stats, args.alpha, excluded)
    pixel_window_rows, pixel_group_rows = pixel_tables(kinds, cells, chunks, window_pixels, cloud_pixels)
    reporter = DiagnosticReporter(stage="Posterior_Predictive_Embedding", enabled=True, dump=True,
                                  dump_dir=R["out_dir"],
                                  run_label=f"{R['paths'].project_alias}_{R['product_label']}")
    finite = bool(all(np.isfinite(c).all() for c in cloud_emb) and all(np.isfinite(v).all() for v in window_emb))
    reporter.check("windows_analyzed", len(present) > 0, f"{len(present)} of {W} planned window(s)")
    reporter.check("no_nan_inf(embeddings)", finite, "clean" if finite else "NaN/Inf present")
    reporter.table(
        "Experimental windows against their own predictive videos: by group",
        ["group", "windows", "outside their predictive cloud (raw)",
         "outside their predictive cloud (standardized)", "readings"], group_rows,
        note=f"For every window, {S} videos were simulated at joint draws of its posterior ({manifest['parameters']}). "
             f"The window's distance to its {args.knn_k_cloud}-th nearest predictive video is ranked against "
             f"the same distance among the predictive videos; the tail share is the share of predictive "
             f"videos at least as isolated as the window, and a window is called 'outside' below {args.alpha}. "
             f"With {S} videos the smallest tail share is {1.0 / (S + 1):.3g}. The standardized reading "
             f"divides every embedding coordinate by its spread over the predictive videos and the window "
             f"together. These tail shares are not calibrated p-values and no number of 'outside' windows "
             f"is expected under a correct model: the window determined the posterior its videos were "
             f"drawn at, so the window and the videos are not exchangeable (ranks of this kind tend to "
             f"avoid the tails when the model is right), and windows of one recording are not independent. "
             f"A window outside its cloud is a warning about the combination of forward model and "
             f"inference procedure; it does not say which of the two is at fault. The nuisances of the "
             f"workflow are drawn from their training distribution, not set to the window's own unknown "
             f"values: the check tests predictions averaged over that distribution, and a predictive "
             f"cloud is as broad as those nuisances make it.")
    reporter.table(
        "Posterior draws excluded by the pool",
        ["pool", "draws per window in the source", "largest excluded share", "median excluded share",
         "windows with more than 5 % excluded"],
        [[str(manifest["pool"]), str(n_source), f"{max(excluded):.3f}", f"{float(np.median(excluded)):.3f}",
          f"{sum(e > 0.05 for e in excluded)} of {len(excluded)}"]],
        note="The share of a window's stored posterior draws that the pool did not admit (bounded: any "
             "coordinate outside the prior box; unrestricted: a linear coordinate outside its range). "
             "The predictive videos describe the admitted part of the posterior only; the per-window "
             "shares are in the window table and the CSV.")
    if len(window_rows) <= 80:
        reporter.table("Experimental windows against their own predictive videos: by window",
                       WINDOW_HEADERS, window_rows,
                       note="'Predictive videos in the outskirts of the EVAL cloud' is the share of a window's "
                            "predictive videos whose own tail share in the synthetic EVAL set is below "
                            "0.05. It describes where the predictive videos lie in the prior's cloud. A "
                            "large share does not by itself indicate a fault: a posterior can concentrate, "
                            "with valid parameters, in a region the prior populated thinly. It is a reason "
                            "to look at the videos, at the storage and labeling steps and at where the "
                            "posterior sits in the prior box, without identifying a cause.")
    reporter.table(
        "Pixel-level comparison: experimental windows against their predictive videos, by group",
        ["group", "statistic", "median over the experimental windows", "median of the windows' predictive medians",
         "windows outside their predictive 5-95 % range"], pixel_group_rows,
        note="Computed on the videos as the network receives them (stored 8-bit values). For every window "
             "and statistic the experimental value is placed among the values of that window's predictive "
             "videos; 'outside' counts the windows below the 5th or above the 95th percentile. The share "
             "of pixels at the top of the range is the clipped fraction. Descriptive, like the tail "
             "shares, and for the same reasons; it complements the embedding, which need not retain "
             "every difference between videos. Per-window values are in window_pixel_statistics.csv.")
    with open(R["out_dir"] / "window_pixel_statistics.csv", "w") as handle:
        handle.write(",".join(h.replace(",", ";") for h in PIXEL_HEADERS) + "\n")
        for row in pixel_window_rows:
            handle.write(",".join(cell.replace(",", ";") for cell in row) + "\n")
    with open(R["out_dir"] / "window_predictive_typicality.csv", "w") as handle:
        handle.write(",".join(h.replace(",", ";") for h in WINDOW_HEADERS) + "\n")
        for row in window_rows:
            handle.write(",".join(cell.replace(",", ";") for cell in row) + "\n")
    reporter.save_figure(
        "predictive_projection",
        _figure_projection(kinds, cells, window_emb, cloud_emb, reference_emb, np.random.default_rng(0)),
        caption="Embeddings on two principal axes of the synthetic EVAL cloud (of the pooled predictive "
                "videos when no EVAL embeddings were given). Gray: synthetic EVAL videos. Small dots: the "
                "videos simulated at posterior draws of a recording's windows. Diamonds: the experimental "
                "windows of that recording. A diamond inside the dots of its color is a window that looks "
                "to the network like the videos simulated at the parameters inferred for it. A linear projection can hide a separation "
                "that exists in the full embedding; the tail shares are computed there.")
    reporter.stat("plan", f"{R['plan_path']} (sha256 {plan_sha256[:16]})")
    reporter.stat("window layout", f"{n_frames} frames, step {step_frames} frames (as recorded by the source of "
                                   f"the posterior draws)")
    reporter.stat("draws", f"{manifest['draws_source']['path']} (pool {manifest['pool']}, "
                           f"{S} per window, seed {manifest['seed']})")
    reporter.stat("estimator weights sha256", str(estimator_sha256))
    reporter.stat("synthetic EVAL reference", reference_source if reference_emb is not None
                  else f"none ({reference_source})")
    reporter.stat("neighbor ranks", f"k={args.knn_k_cloud} in a predictive cloud, k={args.knn_k} in the EVAL cloud")
    reporter.stat("alpha", args.alpha)
    reporter.write_report()
    np.savez_compressed(
        str(R["out_dir"] / "embeddings.npz"), experimental=np.vstack(window_emb).astype(np.float32),
        predictive=np.stack(cloud_emb).astype(np.float32), kind=np.array(kinds), cell=np.array(cells),
        chunk=np.array(chunks), estimator_sha256=str(estimator_sha256), readings=np.array(readings),
        plan_sha256=plan_sha256, excluded_share=np.array(excluded),
        experimental_pixel_statistics=np.array(window_pixels),
        predictive_pixel_statistics=np.stack(cloud_pixels), pixel_statistics=np.array(PIXEL_STATISTICS))
    n_out = sum(s["posterior_tail_raw"] < args.alpha for s in stats)
    print(f"\n{n_out} of {len(present)} window(s) outside their own predictive cloud (raw, alpha {args.alpha}).")
    print(f"report + figures:\n    {R['out_dir']}")
    return 0


# =================================================================================================
# Entry point
# =================================================================================================

def run_posterior_predictive_embedding(cfg, args):
    """Shared entry point. ``cfg`` is a WorkflowConfig; ``args`` the parsed CLI namespace."""
    R = _spec(cfg, args)
    if args.dry_run:
        n_frames, step_frames = _chunk_geometry(R["timing"], args.chunk_step_seconds)
        n_videos = len(list(R["video_dir"].glob("Window_*.npz"))) if R["video_dir"].exists() else 0
        print(f"[DRY RUN] posterior-predictive embedding check ({cfg.tag}), step '{args.step}':")
        print(f"    estimator     : {R['estimator_path']}  [{'OK' if R['estimator_path'].exists() else 'MISSING'}]")
        print(f"    draws source  : {R['draws_source']}  [{'OK' if R['draws_source'].exists() else 'MISSING'}]")
        print(f"    selection     : {args.draws_per_window} draw(s) per window, pool={args.pool}, seed={args.seed}; "
              f"groups {[_disp(k) for k in resolve_kinds(args)]}")
        print(f"    window        : {n_frames} frames; the step between windows is read from the draws source "
              f"at --step plan (this run's default would be {step_frames})")
        print(f"    plan          : {R['plan_path']}  [{'EXISTS' if R['plan_path'].exists() else 'not written'}]")
        print(f"    videos        : {R['video_dir']}/Window_<GROUP>_Cell_<n>_Chunk_<w>.npz  [{n_videos} present]")
        print(f"    EVAL reference: {R['reference_embeddings']}  "
              f"[{'OK' if R['reference_embeddings'].exists() else 'MISSING: the EVAL column is left empty'}]")
        print(f"    writes        : {R['out_dir']}/")
        print("[DRY RUN] no draw selected, no simulation, no embedding.")
        return 0
    if args.step == "plan":
        return run_plan(cfg, args, R)
    if args.step == "simulate":
        return run_simulate(cfg, args, R)
    return run_analyze(cfg, args, R)


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--step", required=True, choices=STEPS,
                   help="plan: select the draws and fix every video's parameters; simulate: render the "
                        "plan's videos (resumable; --shard splits the windows); analyze: embed, rank, report.")
    p.add_argument("--condition", required=True, choices=LABELING_CONDITIONS,
                   help="condition of the run (the estimator's condition).")
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window; sets the timing label locating the estimator and naming the outputs.")
    p.add_argument("--artifact-tag", type=str, default=None,
                   help="artifact tag of the estimator (and of its Experiment product). Default: untagged.")
    p.add_argument("--experiment-span-seconds", type=int, default=None,
                   help="plan only: a cross-check, not a choice. The duration (s) of the recordings is read "
                        "from the source of the posterior draws; a value given here that differs is refused "
                        f"(a source that records no span: default {DEFAULT_SPAN_SECONDS}).")
    p.add_argument("--kinds", type=str, default=None,
                   help="comma-separated recording groups to check (default: those of the condition).")
    p.add_argument("--chunk-step-seconds", type=int, default=None,
                   help="plan only: a cross-check, not a choice. The window layout is read from the source "
                        "of the posterior draws; a step given here that differs from it is refused.")
    p.add_argument("--max-cells", type=int, default=0,
                   help="plan only: the first N recordings of each group (0 = all).")
    p.add_argument("--draws-source", type=str, default=None,
                   help="plan only: file holding the windows' posterior draws, an Experiment product with "
                        "retained clouds or a *_Window_Posterior_Draws*.npz collection (default: the "
                        "Experiment product of the estimator).")
    p.add_argument("--draws-per-window", type=int, default=50,
                   help="plan only: videos per experimental window (default 50).")
    p.add_argument("--pool", choices=POOLS, default="bounded",
                   help="plan only: 'bounded' (default) admits only draws inside the prior box; "
                        "'unrestricted' admits draws outside it on the log rows.")
    p.add_argument("--seed", type=int, default=0,
                   help="plan only: seed of the draw selection and of the nuisance draws (default 0).")
    p.add_argument("--nuisance-tag", type=str, default=None,
                   help="plan only, biology: a tagged Nuisance_DLI artifact instead of the canonical one.")
    p.add_argument("--shard", type=str, default=None,
                   help="simulate only: i/n, the windows whose index is i modulo n (default: all).")
    p.add_argument("--reference-embeddings", type=str, default=None,
                   help="analyze only: embeddings.npz of the embedding-distance analysis for the same "
                        "estimator (default: that analysis' output folder, used when present).")
    p.add_argument("--knn-k-cloud", type=int, default=5,
                   help="analyze only: neighbor rank inside a window's predictive cloud (default 5).")
    p.add_argument("--knn-k", type=int, default=10,
                   help="analyze only: neighbor rank inside the synthetic EVAL cloud (default 10).")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="analyze only: tail share below which a window is called outside (default 0.05).")
    p.add_argument("--allow-partial", action="store_true",
                   help="analyze only: report the windows already simulated when some are missing.")
    p.add_argument("--run-label", type=str, default=None,
                   help="suffix of the output folder, for a second selection beside the first.")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the paths and report what exists; select, simulate and embed nothing.")
    return p
