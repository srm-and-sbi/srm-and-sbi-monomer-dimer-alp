"""Synthetic, schema-valid products for the contract tests (no GPU, no trained flow).

``valid_manifest`` builds a complete manifest through ``artifact_schema.build_manifest`` with every
block populated the way a stage populates it; ``product`` wraps it with correctly shaped arrays.
Tests mutate the returned dicts to construct each negative case, so a rejection is always tested
against an otherwise-valid product.
"""
import numpy as np

from srm_and_sbi_monomer_dimer_alp import artifact_schema as S
from srm_and_sbi_monomer_dimer_alp.evaluation import ESTIMATE_DEFINITIONS_VERSION, QUANTILE_LEVELS

D = 3
KEYS = ["a", "b", "c"]
KINDS = ["FAB"]
SHA_I = "1" * 64
SHA_C = "c" * 64


def code_block(changed=False):
    files = [{"path": "x.py", "sha256": "f" * 64}]
    return {"git_head": "h" * 40, "git_dirty": False,
            "implementation": {"sha256": SHA_I, "files": files},
            "implementation_at_write": {"sha256": ("2" * 64) if changed else SHA_I, "files": files},
            "changed_during_run": bool(changed)}


def optimizer_block(pool_mode="bounded"):
    return {"pool_mode": pool_mode, "theta_prex_size": 1000, "elite_prex_size": 2,
            "numb_steps": 1000, "optimizer_patience": 100, "scheduler_patience": 10,
            "learning_rate": 0.128, "learning_rate_minimum": 1e-3, "learning_rate_factor": 0.5,
            "tolerance": 1e-3, "bookkeeping": "best (score, vector) recorded before optimizer.step"}


def valid_manifest(stage, n, *, pool_mode="bounded", invocation="inv-1", job_id="1", seed=None,
                   kinds=KINDS, **over):
    kw = dict(stage=stage, parameter_keys=KEYS, coordinate_transform="log10", pool_mode=pool_mode,
              draw_label=S.DRAW_LABELS[pool_mode], n_summary_draws=50,
              quantile_levels=QUANTILE_LEVELS, sgm_scale=[1.0] * D, sgm_coordinates="estimator",
              optimizer=optimizer_block(pool_mode), code=code_block(), checkpoint_sha256=SHA_C,
              run_identity={"product_label": "P", "invocation_id": invocation},
              execution={"job_id": job_id}, stored_optional_fields=[],
              seed_policy=S.seed_policy(seed),
              window_geometry=(S.window_geometry(n_frames=100, step_frames=100, span_frames=1000)
                               if stage == "experiment" else None),
              condition_labels=(list(kinds) if stage == "experiment" else None),
              n_observations=n, estimate_definitions_version=ESTIMATE_DEFINITIONS_VERSION)
    kw.update(over)
    return S.build_manifest(**kw)


def product(stage, ids, rng, *, manifest=None, kinds=KINDS, **over):
    """A valid product for ``ids`` (evaluation: ``(task, sim)``; experiment: ``(kind, cell,
    chunk)``). ``manifest`` overrides the built one (already a dict); ``over`` feeds
    :func:`valid_manifest`."""
    n = len(ids)
    a = dict(map_estimate=rng.normal(size=(n, D)),
             posterior_quantiles=np.sort(rng.normal(size=(n, D, 5)), axis=-1),
             posterior_sgm=rng.normal(size=(n, D)), scores=rng.normal(size=n))
    if stage == "evaluation":
        a["true_log10"] = rng.normal(size=(n, D))
        a["task_index"] = np.array([i[0] for i in ids], dtype=np.int64)
        a["sim_index"] = np.array([i[1] for i in ids], dtype=np.int64)
    else:
        a["kind_index"] = np.array([i[0] for i in ids], dtype=np.int64)
        a["cell"] = np.array([i[1] for i in ids], dtype=np.int64)
        a["chunk"] = np.array([i[2] for i in ids], dtype=np.int64)
        a["kinds"] = np.asarray(list(kinds))
    m = manifest if manifest is not None else valid_manifest(stage, n, kinds=kinds, **over)
    a[S.MANIFEST_KEY] = S.encode_manifest(m)
    return a


def copy_product(a):
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in a.items()}


def with_manifest(a, mutate):
    """Copy of product ``a`` whose decoded manifest was passed through ``mutate(dict)``."""
    b = copy_product(a)
    m = S.decode_manifest(b[S.MANIFEST_KEY])
    mutate(m)
    b[S.MANIFEST_KEY] = S.encode_manifest(m)
    return b


def expect_schema_error(fn, *needles):
    try:
        fn()
    except S.SchemaError as exc:
        msg = str(exc)
        for needle in needles:
            assert needle in msg, (needle, msg)
        return msg
    raise AssertionError("accepted")
