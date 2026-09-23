"""Point-estimate validation, phase 1: the marginal median as the reference.

Workflow-agnostic kernel: definitions, the frozen acceptance thresholds, the independent
recomputation, the accuracy and stability summaries, the gates, and the validation-artifact format.
Nothing here loads a checkpoint or reads a video; the runner
(``point_estimate_validation_runner``) produces the draws and calls these functions.

WHAT THE MEDIAN IS HERE. For one observation, the 0.50 level of ``posterior_quantiles``: numpy's
default linear interpolation (Hyndman & Fan type 7) of each coordinate's draws, taken in ESTIMATOR
coordinates (log10 for a log row). A physical value is ``to_physical`` of that quantile -- quantile,
then transform. With finite-sample interpolation the reverse order need not give the same number,
so the reverse order is never used for this quantity.

WHAT PHASE 1 ESTABLISHES. That the production calculation (``evaluation.posterior_summary``)
returns that quantity, and that its Monte Carlo variability at the production draw count is small
relative to the smallest accuracy difference the program will interpret. It then serves as the
reference the SGM and MAP phases are compared against -- a reference, not presumed truth. It does
not establish calibration, and it cannot remove posterior bias: those are properties of the
checkpoint, measured elsewhere.

THE PROTOCOL (frozen before any comparison). Fixed checkpoint, fixed synthetic observations, bounded
sampling. Per observation: five independent repeats at the production draw count and one larger
reference run, each from its own random stream. Per parameter, overall and in the dim subgroup:

* recomputation: the stored median equals an independent recomputation from the stored draws to
  an absolute ``THRESHOLDS["recompute_abs_dex"]``;
* subset stability: MAE and signed bias are computed separately for each repeat; their sample
  standard deviation across repeats is at most ``THRESHOLDS["subset_sd_dex"]`` (minimum and maximum
  are reported too). Five repeats estimate the Monte Carlo variability; they do not bound it;
* resolution check: the repeat mean of MAE and of bias is within ``THRESHOLDS["reference_diff_dex"]``
  of the reference run. The reference is itself a finite sample, not exact truth: a narrow failure is
  assessed against the reference's own sampling variability before 1,000 draws are called inadequate;
* per recording: the standard deviation of the five repeat medians, divided by the posterior IQR
  from the reference run, is at most ``THRESHOLDS["ratio_max"]`` for at least
  ``THRESHOLDS["ratio_share_min"]`` of recordings; every exceedance and the maximum are reported,
  and a zero IQR carries an explicit status instead of being dropped.

``THRESHOLDS["resolution_dex"]`` is the smallest subset-level difference in MAE or bias the program
will interpret: an analysis-resolution choice, not a biological tolerance or a significance level.
"""
from __future__ import annotations

import json

import numpy as np

from .direct_acceptance import OPERATING_KEY, OPERATING_RANGE

VALIDATION_KIND = "point_estimate_validation"
VALIDATION_SCHEMA_VERSION = 1
#: Phases implemented so far. The SGM and MAP phases are added when their protocols are frozen.
PHASES = ("median",)
#: Posterior-sampling mode, fixed for the whole program.
SAMPLING_MODE = "bounded"
INTERPOLATION = ("numpy default linear interpolation (Hyndman & Fan type 7), per coordinate, in "
                 "estimator coordinates; a physical value is to_physical of the quantile")

#: Frozen acceptance thresholds of phase 1 (all in dex except the two dimensionless ratio entries).
THRESHOLDS = {
    "recompute_abs_dex": 1e-6,     # |stored median - recomputation from its stored draws|
    "resolution_dex": 0.005,       # smallest subset-level MAE or bias difference interpreted
    "subset_sd_dex": 0.001,        # sample SD across repeats of subset MAE and of subset bias
    "reference_diff_dex": 0.001,   # |repeat mean - reference run|, for MAE and for bias
    "ratio_max": 0.1,              # per recording: SD of repeat medians / reference IQR
    "ratio_share_min": 0.95,       # share of recordings that must meet ratio_max
    "large_error_dex": 0.3,        # |median - truth| above this is a large error (a factor of 2)
}
#: The dim operating subgroup, as frozen in DETECTOR_WORKFLOW.md section 9.6.
DIM_SUBGROUP = {"key": OPERATING_KEY, "log10_range": [float(v) for v in OPERATING_RANGE],
                "source": "direct_acceptance.OPERATING_RANGE (DETECTOR_WORKFLOW.md section 9.6)"}
SUBGROUPS = ("overall", "dim")
ID_FIELDS = ("task_index", "sim_index")
#: Manifest entries every shard of one run must share.
CONTRACT_KEYS = ("kind", "validation_schema_version", "phase", "workflow", "condition",
                 "product_label", "parameter_keys", "coordinate_transform", "sampling", "quantiles",
                 "checkpoint", "code", "seeds", "observations", "dim_subgroup", "thresholds",
                 "run_identity")


class ValidationError(ValueError):
    """A validation artifact or shard set does not satisfy its format."""


# ---- observations, seeds -----------------------------------------------------------------------

def stream_seed(base_seed: int, task: int, sim: int, stream: int) -> int:
    """The torch seed of one random stream of one observation. Derived from ``(task, sim, stream)``
    through ``numpy.random.SeedSequence``, so streams are independent, and an observation gets the
    same seeds whichever rank or order processes it. Streams ``0 .. R-1`` are the repeats; stream
    ``R`` is the reference run."""
    ss = np.random.SeedSequence(entropy=int(base_seed), spawn_key=(int(task), int(sim), int(stream)))
    hi, lo = (int(v) for v in ss.generate_state(2, dtype=np.uint32))
    return ((hi << 32) | lo) & ((1 << 63) - 1)


def dim_mask(true_log10, parameter_keys, rule=DIM_SUBGROUP) -> np.ndarray:
    """Membership of the dim subgroup from the TRUE value of the subgroup key (half-open range)."""
    j = list(parameter_keys).index(rule["key"])
    lo, hi = rule["log10_range"]
    v = np.asarray(true_log10, dtype=float)[:, j]
    return (v >= lo) & (v < hi)


def select_pilot(task_index, sim_index, true_log10, parameter_keys, n, rule=DIM_SUBGROUP):
    """Deterministic pilot selection: ``n // 2`` dim and ``n - n // 2`` bright recordings, each set
    taken at evenly spaced ranks of the true subgroup key within its subgroup (ties broken by task,
    then sim), so both ends of each subgroup are covered. Returns sorted indices into the inputs."""
    task = np.asarray(task_index)
    sim = np.asarray(sim_index)
    key = np.asarray(true_log10, dtype=float)[:, list(parameter_keys).index(rule["key"])]
    dim = dim_mask(true_log10, parameter_keys, rule)
    picks = []
    for mask, k in ((dim, n // 2), (~dim, n - n // 2)):
        idx = np.flatnonzero(mask)
        if idx.size < k:
            raise ValidationError(f"pilot needs {k} recordings in a subgroup that has {idx.size}.")
        if k == 0:
            continue
        order = idx[np.lexsort((sim[idx], task[idx], key[idx]))]
        positions = np.round(np.linspace(0, idx.size - 1, k)).astype(int)
        picks.extend(int(order[p]) for p in positions)
    return np.array(sorted(set(picks)), dtype=int)


# ---- quantiles, independent recomputation ------------------------------------------------------

def type7_quantiles(draws, levels) -> np.ndarray:
    """Quantiles of each column of ``draws`` ``(n, D)``, computed from the definition (sort, then
    interpolate at position ``(n - 1) p``) and not through ``numpy.quantile``. Returns ``(D, Q)``."""
    x = np.sort(np.asarray(draws, dtype=np.float64), axis=0)
    n = x.shape[0]
    out = np.empty((x.shape[1], len(levels)))
    for k, p in enumerate(levels):
        h = (n - 1) * float(p)
        lo = int(np.floor(h))
        hi = min(lo + 1, n - 1)
        out[:, k] = x[lo] + (h - lo) * (x[hi] - x[lo])
    return out


def recomputation_max_abs_diff(arrays, levels):
    """Largest ``|stored quantile - recomputation from its stored draws|`` over every observation,
    stream, parameter and level. Returns ``(max_abs_dex, where)`` with ``where`` naming the worst
    observation index, stream and parameter index."""
    worst, where = 0.0, None
    rd, rq = arrays["repeat_draws"], arrays["repeat_quantiles"]
    fd, fq = arrays["reference_draws"], arrays["reference_quantiles"]
    for i in range(rd.shape[0]):
        for r in range(rd.shape[1]):
            d = np.abs(type7_quantiles(rd[i, r], levels) - rq[i, r])
            if d.size and d.max() > worst:
                worst = float(d.max()); where = (i, f"repeat {r}", int(np.unravel_index(d.argmax(), d.shape)[0]))
        d = np.abs(type7_quantiles(fd[i], levels) - fq[i])
        if d.size and d.max() > worst:
            worst = float(d.max()); where = (i, "reference", int(np.unravel_index(d.argmax(), d.shape)[0]))
    return worst, where


# ---- accuracy and stability ----------------------------------------------------------------------

def accuracy(medians, truth, mask, large_dex=THRESHOLDS["large_error_dex"]):
    """MAE, signed bias and large-error share of ``medians`` against ``truth`` over ``mask``.
    ``medians`` is ``(N, D)`` or ``(N, R, D)``; returns arrays of shape ``(D,)`` or ``(R, D)``."""
    med = np.asarray(medians, dtype=float)[mask]
    tru = np.asarray(truth, dtype=float)[mask]
    err = med - (tru[:, None, :] if med.ndim == 3 else tru)
    return (np.abs(err).mean(axis=0), err.mean(axis=0),
            (np.abs(err) > float(large_dex)).mean(axis=0))


def across_repeats(values) -> dict:
    """Mean, sample standard deviation (ddof 1), minimum and maximum over the repeat axis 0."""
    v = np.asarray(values, dtype=float)
    return {"mean": v.mean(axis=0), "sd": v.std(axis=0, ddof=1), "min": v.min(axis=0),
            "max": v.max(axis=0)}


def per_recording_ratio(repeat_medians, reference_quantiles, levels):
    """Per recording and parameter: the standard deviation (ddof 1) of the repeat medians divided by
    the posterior IQR of the reference run. Returns ``(ratio, sd, iqr, status)``, each ``(N, D)``.
    ``status`` is ``ok``; ``zero_iqr_zero_sd`` (ratio 0, counted as meeting the criterion); or
    ``zero_iqr_nonzero_sd`` (ratio infinite, an exceedance). Nothing is dropped."""
    lv = [round(float(v), 12) for v in levels]
    i25, i75 = lv.index(0.25), lv.index(0.75)
    q = np.asarray(reference_quantiles, dtype=float)
    iqr = q[:, :, i75] - q[:, :, i25]
    sd = np.asarray(repeat_medians, dtype=float).std(axis=1, ddof=1)
    ratio = np.zeros_like(sd)
    status = np.full(sd.shape, "ok", dtype="<U20")
    positive = iqr > 0
    ratio[positive] = sd[positive] / iqr[positive]
    zero_zero = ~positive & (sd == 0)
    zero_nonzero = ~positive & (sd > 0)
    status[zero_zero] = "zero_iqr_zero_sd"
    status[zero_nonzero] = "zero_iqr_nonzero_sd"
    ratio[zero_nonzero] = np.inf
    return ratio, sd, iqr, status


def median_metrics(arrays, manifest) -> dict:
    """Everything the phase-1 report and gates read, computed from the artifact alone."""
    levels = manifest["quantiles"]["levels"]
    qi = [round(float(v), 12) for v in levels].index(0.50)
    truth = np.asarray(arrays["true_log10"], dtype=float)
    rep_med = np.asarray(arrays["repeat_quantiles"], dtype=float)[:, :, :, qi]      # (N, R, D)
    ref_med = np.asarray(arrays["reference_quantiles"], dtype=float)[:, :, qi]      # (N, D)
    masks = {"overall": np.ones(truth.shape[0], dtype=bool),
             "dim": np.asarray(arrays["dim_subgroup"], dtype=bool)}
    out = {"recompute": recomputation_max_abs_diff(arrays, levels), "subgroups": {}}
    ratio, sd, iqr, status = per_recording_ratio(rep_med, arrays["reference_quantiles"], levels)
    out["ratio"], out["ratio_sd"], out["ratio_iqr"], out["ratio_status"] = ratio, sd, iqr, status
    for name, m in masks.items():
        if not m.any():
            out["subgroups"][name] = None
            continue
        mae, bias, large = accuracy(rep_med, truth, m)
        mae_ref, bias_ref, large_ref = accuracy(ref_med, truth, m)
        meets = ratio[m] <= THRESHOLDS["ratio_max"]
        out["subgroups"][name] = {
            "n": int(m.sum()), "mae": across_repeats(mae), "bias": across_repeats(bias),
            "large": across_repeats(large), "mae_ref": mae_ref, "bias_ref": bias_ref,
            "large_ref": large_ref, "ratio_share": meets.mean(axis=0),
            "ratio_max": ratio[m].max(axis=0), "exceedances": (~meets).sum(axis=0)}
    return out


def evaluate_gates(metrics, thresholds=THRESHOLDS) -> dict:
    """Phase-1 gates from :func:`median_metrics`: ``recompute`` (one boolean) and, per subgroup, one
    boolean per parameter for ``subset_sd``, ``reference_diff`` and ``per_recording``."""
    gates = {"recompute": metrics["recompute"][0] <= thresholds["recompute_abs_dex"]}
    for name, s in metrics["subgroups"].items():
        if s is None:
            continue
        gates[name] = {
            "subset_sd": (s["mae"]["sd"] <= thresholds["subset_sd_dex"])
                         & (s["bias"]["sd"] <= thresholds["subset_sd_dex"]),
            "reference_diff": (np.abs(s["mae"]["mean"] - s["mae_ref"]) <= thresholds["reference_diff_dex"])
                              & (np.abs(s["bias"]["mean"] - s["bias_ref"]) <= thresholds["reference_diff_dex"]),
            "per_recording": s["ratio_share"] >= thresholds["ratio_share_min"],
        }
    return gates


# ---- artifact ------------------------------------------------------------------------------------

def encode_manifest(manifest) -> np.ndarray:
    return np.asarray(json.dumps(manifest, sort_keys=True))


def decode_manifest(value) -> dict:
    return json.loads(str(np.asarray(value).item()))


def artifact_shapes(manifest, n) -> dict:
    """Required arrays and their shapes for ``n`` observations under ``manifest``."""
    s = manifest["sampling"]
    d, q = len(manifest["parameter_keys"]), len(manifest["quantiles"]["levels"])
    r, nd, nr = s["n_repeats"], s["repeat_draws"], s["reference_draws"]
    return {"task_index": (n,), "sim_index": (n,), "true_log10": (n, d), "dim_subgroup": (n,),
            "repeat_draws": (n, r, nd, d), "repeat_quantiles": (n, r, d, q),
            "reference_draws": (n, nr, d), "reference_quantiles": (n, d, q),
            "stream_seeds": (n, r + 1), "stream_seconds": (n, r + 1), "stream_peak_bytes": (n, r + 1)}


def validate_artifact(arrays, *, source="artifact", allow_empty=False) -> dict:
    """Refuse a malformed validation artifact or shard; return its manifest. Checks the manifest
    kind and version, every array and its shape, finiteness, integer unique identifiers, the dim
    flags against the declared rule, and the code block's startup and at-write records."""
    if "manifest_json" not in arrays:
        raise ValidationError(f"{source}: no manifest_json.")
    m = decode_manifest(arrays["manifest_json"])
    if m.get("kind") != VALIDATION_KIND or m.get("validation_schema_version") != VALIDATION_SCHEMA_VERSION:
        raise ValidationError(f"{source}: not a {VALIDATION_KIND} v{VALIDATION_SCHEMA_VERSION} artifact.")
    if m.get("phase") not in PHASES:
        raise ValidationError(f"{source}: unknown phase {m.get('phase')!r}.")
    missing = [k for k in CONTRACT_KEYS if k not in m]
    if missing:
        raise ValidationError(f"{source}: manifest lacks {missing}.")
    n = int(np.asarray(arrays.get("task_index", [])).shape[0])
    if n == 0 and not allow_empty:
        raise ValidationError(f"{source}: no observations.")
    for key, shape in artifact_shapes(m, n).items():
        if key not in arrays:
            raise ValidationError(f"{source}: missing {key}.")
        if tuple(np.asarray(arrays[key]).shape) != shape:
            raise ValidationError(f"{source}: {key} has shape {np.asarray(arrays[key]).shape}; "
                                  f"expected {shape}.")
    for key in ("true_log10", "repeat_draws", "repeat_quantiles", "reference_draws",
                "reference_quantiles", "stream_seconds"):
        if not np.all(np.isfinite(np.asarray(arrays[key], dtype=float))):
            raise ValidationError(f"{source}: non-finite values in {key}.")
    for key in ID_FIELDS:
        if np.asarray(arrays[key]).dtype.kind not in "iu":
            raise ValidationError(f"{source}: {key} must be an integer array.")
    ids = list(zip(np.asarray(arrays["task_index"]).tolist(), np.asarray(arrays["sim_index"]).tolist()))
    if len(set(ids)) != len(ids):
        raise ValidationError(f"{source}: duplicate observation identifiers.")
    if n and not np.array_equal(np.asarray(arrays["dim_subgroup"], dtype=bool),
                                dim_mask(arrays["true_log10"], m["parameter_keys"], m["dim_subgroup"])):
        raise ValidationError(f"{source}: dim_subgroup disagrees with the declared rule.")
    code = m["code"]
    if (code.get("changed_during_run") is not False
            or code["implementation"]["sha256"] != code["implementation_at_write"]["sha256"]
            or code["implementation"]["files"] != code["implementation_at_write"]["files"]):
        raise ValidationError(f"{source}: implementation files changed while the run was in "
                              f"progress; the artifact describes no single implementation.")
    return m


def merge_shards(shards, expected_ids):
    """Combine the validated shards of one run. The shard manifests must agree on every
    :data:`CONTRACT_KEYS` entry; the merged observations must EQUAL ``expected_ids`` (a list of
    ``(task, sim)``) and are returned in that order. ``shards`` is a list of ``(arrays, manifest)``.
    Returns ``(arrays, manifest)`` with the merged manifest carrying each shard's rank, count and
    execution under ``shards``."""
    if not shards:
        raise ValidationError("no shards to merge.")
    ref = shards[0][1]
    for arrays, m in shards[1:]:
        for key in CONTRACT_KEYS:
            if json.dumps(m[key], sort_keys=True) != json.dumps(ref[key], sort_keys=True):
                raise ValidationError(f"shards disagree on {key!r}; they are not one run.")
    keys = list(artifact_shapes(ref, 0))
    merged = {k: np.concatenate([np.asarray(a[k]) for a, _ in shards], axis=0) for k in keys}
    ids = list(zip(merged["task_index"].tolist(), merged["sim_index"].tolist()))
    want = [tuple(int(v) for v in e) for e in expected_ids]
    if len(set(ids)) != len(ids):
        raise ValidationError("merged shards contain duplicate observations.")
    if set(ids) != set(want):
        missing, extra = sorted(set(want) - set(ids)), sorted(set(ids) - set(want))
        raise ValidationError(f"merged observations differ from the frozen list: {len(missing)} "
                              f"missing {missing[:5]}; {len(extra)} unexpected {extra[:5]}.")
    position = {e: i for i, e in enumerate(ids)}
    order = np.array([position[e] for e in want], dtype=int)
    merged = {k: v[order] for k, v in merged.items()}
    manifest = dict(ref)
    manifest.update(n_observations=len(want), rank=None, world_size=None,
                    shards=[{"rank": m.get("rank"), "n_observations": m.get("n_observations"),
                             "execution": m.get("execution"), "written_at": m.get("written_at")}
                            for _, m in shards])
    return merged, manifest
