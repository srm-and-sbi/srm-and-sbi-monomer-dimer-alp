"""MAP optimization benchmark: configurations, a batched optimizer engine, summaries.

STATUS. Configurations ``C``..``H``, their loop (per-seed chains, the post-reduction guard, the
1e-4-nat threshold) and their safeguarded scaling formula are a BENCHMARK PROPOSAL -- NOT ADOPTED.
Production adopted a separate configuration (0.1.17): steps in units of the plain pool IQR, the
existing joint loop, and its own budget and tolerance (``evaluation.optimize_elite``,
``InferenceEvaluation``). ``B`` always runs the production optimizer as configured, so a run made
before 0.1.17 measured the absolute-step optimizer and a later run measures the adopted one.

THE OBJECTIVE (frozen for the benchmark). The UNCONSTRAINED flow log-density ``log q(theta | x)`` in
estimator coordinates, the production objective. A point returned by a finite search on it is a
numerical MAP candidate. A prior-constrained objective is a separate scientific decision.

THE COMPARISON. For each recording, two independent bounded candidate pools of the production size
are drawn. Within a pool, every configuration starts from the SAME seeds (the production number of
top-scoring candidates), so differences between configurations are differences of optimization:

* ``A`` the best candidate, no optimization;
* ``B`` the production ``evaluation.optimize_elite``, as configured when the benchmark runs;
* ``C``..``H`` the new loop (:func:`run_chains`) under the settings of :data:`CHAIN_CONFIGS`:
  absolute versus scale-aware steps, the production versus an extended step budget and patience,
  and a bracket of scale-aware learning rates.

THE NEW LOOP. Each seed is its own chain: its own Adam state, its own learning rate reduced on its
own plateau, its own patience, its own stop. Every strictly better finite (score, vector) pair is
retained, recorded before the update, so the returned score is the density at the returned vector.
Patience counts steps since the last improvement larger than :data:`IMPROVE_TOL`; after a
learning-rate reduction the chain may not stop for ``guard`` steps. A non-finite score stops the
chain. Scale-aware chains move ``u = (theta - c) / s`` with ``s_i = min(W_i, max(IQR_i, eps W_i))``,
``c`` the pool median, ``IQR`` the pool's interquartile range and ``W`` the prior width: one global
learning rate then means a fixed fraction of each parameter's posterior spread. The objective is
unchanged by this: the density is maximized in theta, without a Jacobian term.

The engine runs all chains of many recordings in one batch: each step is one flow evaluation over
every active chain, with each chain's own conditioning embedding. Chains never interact: the loss
is a sum of per-chain terms, Adam is elementwise, and every per-chain quantity is kept separately.

SELECTION (after the run). Settings are chosen for optimization reliability -- gap to a polished
reference optimum, agreement between the two pools, stops that are not budget hits -- and for cost.
Not for moving the MAP towards the median: a higher density does not imply a lower parameter error.
"""
from __future__ import annotations

import numpy as np
import torch

CURRENT = {"max_steps": 1000, "stop_patience": 100, "sched_patience": 10, "guard": 0}
EXTENDED = {"max_steps": 3000, "stop_patience": 300, "sched_patience": 30, "guard": 50}
IMPROVE_TOL = 1e-4          # nats: a meaningful improvement, for patience and for the scheduler
LR_FACTOR = 0.5             # the production plateau factor
SCALE_EPS = 1e-3            # safeguard: a scale never below this fraction of the prior width
ADAM_BETAS, ADAM_EPS = (0.9, 0.999), 1e-8     # torch.optim.Adam defaults

#: name -> (coordinates, learning rate, learning-rate floor, budget) -- benchmark proposal, not adopted
CHAIN_CONFIGS = {
    "C": ("absolute", 0.128, 1e-3, CURRENT),
    "D": ("scaled", 0.05, 5e-4, CURRENT),
    "E": ("scaled", 0.05, 5e-4, EXTENDED),
    "F": ("absolute", 0.128, 1e-3, EXTENDED),
    "G": ("scaled", 0.02, 5e-4, EXTENDED),
    "H": ("scaled", 0.2, 5e-4, EXTENDED),
}
CONFIG_ORDER = ("A", "B") + tuple(CHAIN_CONFIGS)
CONFIG_LABELS = {
    "A": "best candidate, no optimization",
    "B": "production optimizer, current settings",
    "C": "new loop, absolute lr 0.128, current budget",
    "D": "new loop, scale-aware lr 0.05, current budget",
    "E": "new loop, scale-aware lr 0.05, extended budget",
    "F": "new loop, absolute lr 0.128, extended budget",
    "G": "new loop, scale-aware lr 0.02, extended budget",
    "H": "new loop, scale-aware lr 0.2, extended budget",
}
#: stop codes stored per seed (STOP_PRODUCTION: a pre-0.1.17 production run, stop not reported)
STOP_EARLY, STOP_BUDGET, STOP_NONFINITE, STOP_PRODUCTION, STOP_NONE, STOP_INVALID_SCALE = 1, 2, 3, 4, 5, 6
STOP_NAMES = {STOP_EARLY: "early", STOP_BUDGET: "budget", STOP_NONFINITE: "non-finite",
              STOP_PRODUCTION: "production", STOP_NONE: "no optimization",
              STOP_INVALID_SCALE: "invalid-scale"}
#: ``evaluation.STOP_REASONS`` -> the stored code, for the production configuration ``B``
PRODUCTION_STOP_CODES = {"early": STOP_EARLY, "budget": STOP_BUDGET, "non-finite": STOP_NONFINITE,
                         "invalid-scale": STOP_INVALID_SCALE}


def scaled_coordinates(pool, prior_width, eps=SCALE_EPS):
    """``(center, iqr, scale, below, above)`` of one candidate pool ``(n, D)``: the pool median, its
    interquartile range, the scale ``min(W, max(IQR, eps W))``, and where each safeguard bound was
    active (``IQR < eps W``: scale raised; ``IQR > W``: scale capped at the prior width)."""
    pool = np.asarray(pool, dtype=np.float64)
    w = np.asarray(prior_width, dtype=np.float64)
    center = np.median(pool, axis=0)
    iqr = np.quantile(pool, 0.75, axis=0) - np.quantile(pool, 0.25, axis=0)
    scale = np.minimum(w, np.maximum(iqr, eps * w))
    return center, iqr, scale, iqr < eps * w, iqr > w


def run_chains(score_fn, theta0, center, scale, lr, lr_min, max_steps, stop_patience,
               sched_patience, guard, *, deviation_scale=None, improve_tol=IMPROVE_TOL,
               lr_factor=LR_FACTOR, betas=ADAM_BETAS, eps=ADAM_EPS):
    """Independent Adam chains maximizing ``score_fn``, batched over chains.

    ``score_fn(theta, idx)`` returns the log-density ``(n,)`` of the chains ``idx`` (a long tensor)
    at ``theta`` ``(n, D)``. ``theta0``, ``center``, ``scale`` are ``(M, D)``; ``lr``, ``lr_min``,
    ``max_steps``, ``stop_patience``, ``sched_patience``, ``guard`` are ``(M,)``. Per chain, each
    step: evaluate at the current point; retain a strictly better finite pair; stop early when the
    steps since the last improvement larger than ``improve_tol`` reach ``stop_patience`` and the
    chain is outside the ``guard`` window after its last learning-rate reduction; stop at
    ``max_steps``; otherwise take one Adam step in ``u = (theta - center) / scale`` (torch's
    arithmetic, defaults ``betas``, ``eps``) and step a plateau scheduler on the chain's own score
    (``torch.optim.lr_scheduler.ReduceLROnPlateau`` semantics: mode max, absolute threshold
    ``improve_tol``, reduction when the bad-step count exceeds ``sched_patience``, floor ``lr_min``).

    Returns a dict of tensors: ``best`` ``(M,)``, ``best_theta`` ``(M, D)``, ``stop`` codes,
    ``steps``, ``seed_score`` (the score at the start), ``overshoot`` (start score minus the lowest
    score visited) and ``max_deviation`` (largest ``|theta - theta0| / deviation_scale``).
    """
    dev, fdt = theta0.device, theta0.dtype
    m_count, dim = theta0.shape
    lr = lr.clone().to(dev, fdt)
    lr_min = lr_min.to(dev, fdt)
    max_steps, stop_patience = max_steps.to(dev), stop_patience.to(dev)
    sched_patience, guard = sched_patience.to(dev), guard.to(dev)
    dscale = scale if deviation_scale is None else deviation_scale
    u = ((theta0 - center) / scale).detach().clone()
    m1, m2 = torch.zeros_like(u), torch.zeros_like(u)
    t = torch.zeros(m_count, dtype=torch.long, device=dev)
    best = torch.full((m_count,), -float("inf"), device=dev, dtype=fdt)
    best_theta = theta0.detach().clone()
    last_imp = torch.zeros(m_count, dtype=torch.long, device=dev)
    last_red = torch.full((m_count,), -10 ** 9, dtype=torch.long, device=dev)
    sched_best = torch.full((m_count,), -float("inf"), device=dev, dtype=fdt)
    bad = torch.zeros(m_count, dtype=torch.long, device=dev)
    active = torch.ones(m_count, dtype=torch.bool, device=dev)
    stop = torch.zeros(m_count, dtype=torch.long, device=dev)
    steps = torch.zeros(m_count, dtype=torch.long, device=dev)
    seed_score = torch.full((m_count,), float("nan"), device=dev, dtype=fdt)
    worst = torch.full((m_count,), float("inf"), device=dev, dtype=fdt)
    max_dev = torch.zeros(m_count, device=dev, dtype=fdt)
    b1, b2 = betas
    for step in range(1, int(max_steps.max().item()) + 1):
        idx = active.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        ua = u[idx].detach().requires_grad_(True)
        theta = center[idx] + scale[idx] * ua
        s = score_fn(theta, idx)
        (g,) = torch.autograd.grad((-s).sum(), ua)
        val = s.detach()
        th = theta.detach()
        if step == 1:
            seed_score[idx] = val
        finite = torch.isfinite(val)
        better = finite & (val > best[idx])
        meaningful = finite & (val > best[idx] + improve_tol)
        last_imp[idx[meaningful]] = step
        best[idx[better]] = val[better]
        best_theta[idx[better]] = th[better]
        worst[idx[finite]] = torch.minimum(worst[idx[finite]], val[finite])
        dev_now = ((th - theta0[idx]).abs() / dscale[idx]).amax(dim=1)
        max_dev[idx] = torch.maximum(max_dev[idx], dev_now)
        steps[idx] = step
        early = finite & ((step - last_imp[idx]) >= stop_patience[idx]) & ((step - last_red[idx]) >= guard[idx])
        nonfinite = ~finite
        budget = ~early & ~nonfinite & (step >= max_steps[idx])
        stop[idx[early]] = STOP_EARLY
        stop[idx[nonfinite]] = STOP_NONFINITE
        stop[idx[budget]] = STOP_BUDGET
        done = early | nonfinite | budget
        active[idx[done]] = False
        keep = ~done
        if not bool(keep.any()):
            continue
        k = idx[keep]
        gk = g[keep]
        t[k] += 1
        m1[k] = b1 * m1[k] + (1 - b1) * gk
        m2[k] = b2 * m2[k] + (1 - b2) * gk * gk
        tk = t[k].to(fdt)
        bc1 = 1 - b1 ** tk
        bc2_sqrt = torch.sqrt(1 - b2 ** tk)
        denom = m2[k].sqrt() / bc2_sqrt[:, None] + eps
        u[k] = u[k] - (lr[k] / bc1)[:, None] * m1[k] / denom
        vk = val[keep]
        improved = vk > sched_best[k] + improve_tol
        sched_best[k] = torch.where(improved, vk, sched_best[k])
        bad[k] = torch.where(improved, torch.zeros_like(bad[k]), bad[k] + 1)
        reduce = bad[k] > sched_patience[k]
        if bool(reduce.any()):
            r = k[reduce]
            new_lr = torch.maximum(lr[r] * lr_factor, lr_min[r])
            changed = (lr[r] - new_lr) > 1e-8
            lr[r[changed]] = new_lr[changed]
            last_red[r[changed]] = step
            bad[r] = 0
    return {"best": best, "best_theta": best_theta, "stop": stop, "steps": steps,
            "seed_score": seed_score, "overshoot": seed_score - worst, "max_deviation": max_dev}


def run_chain_serial(score, theta0, center, scale, *, lr, lr_min, max_steps, stop_patience,
                     sched_patience, guard, improve_tol=IMPROVE_TOL, lr_factor=LR_FACTOR):
    """One chain with ``torch.optim.Adam`` and ``ReduceLROnPlateau`` -- the reference the batched
    engine is tested against. ``score(theta (D,)) -> scalar``. Returns ``(best, theta, stop, steps)``."""
    u = ((theta0 - center) / scale).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=lr_factor,
                                                     patience=sched_patience, threshold=improve_tol,
                                                     threshold_mode="abs", min_lr=lr_min)
    best, best_theta, last_imp, last_red, code, step = -float("inf"), theta0.clone(), 0, -10 ** 9, 0, 0
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        theta = center + scale * u
        s = score(theta)
        val = float(s.item())
        if not np.isfinite(val):
            code = STOP_NONFINITE
            break
        if val > best:
            if val > best + improve_tol:
                last_imp = step
            best, best_theta = val, theta.detach().clone()
        if step - last_imp >= stop_patience and step - last_red >= guard:
            code = STOP_EARLY
            break
        if step >= max_steps:
            code = STOP_BUDGET
            break
        (-s).backward()
        opt.step()
        before = opt.param_groups[0]["lr"]
        sch.step(val)
        if opt.param_groups[0]["lr"] < before:
            last_red = step
    return best, best_theta, code, step


def summary_rows(arrays, levels=(0.5, 0.9)):
    """Per-configuration summary rows for the report, from the merged artifact arrays."""
    score, ref = np.asarray(arrays["score"]), np.asarray(arrays["reference_score"])
    cand = np.asarray(arrays["best_candidate"])
    gain = score - cand[:, :, None]                                     # (N, P, K)
    gap = ref[:, None, None] - score                                    # (N, P, K)
    incons = np.abs(np.asarray(arrays["reeval"]) - score)
    stop = np.asarray(arrays["stop_code"])                              # (N, P, K, S)
    steps = np.asarray(arrays["steps"])
    rows = []
    for k, name in enumerate(CONFIG_ORDER):
        g, d, c = gain[:, :, k].ravel(), gap[:, :, k].ravel(), incons[:, :, k].ravel()
        st, sp = stop[:, :, k, :].ravel(), steps[:, :, k, :].ravel()
        # stopping is reported where it was recorded: the chains, and B from 0.1.17 on
        chain = name in CHAIN_CONFIGS or (name == "B" and not np.isin(st, (0, STOP_PRODUCTION)).any())
        rows.append({
            "config": name, "label": CONFIG_LABELS[name],
            "gain_median": float(np.median(g)), "gain_mean": float(np.mean(g)),
            "improved_share": float(np.mean(g > 1e-3)),
            "gap_median": float(np.median(d)), "gap_p90": float(np.quantile(d, 0.9)),
            "gap_max": float(np.max(d)), "within_1e-3": float(np.mean(d <= 1e-3)),
            "within_1e-2": float(np.mean(d <= 1e-2)), "inconsistency_max": float(np.max(c)),
            "budget_share": float(np.mean(st == STOP_BUDGET)) if chain else float("nan"),
            "nonfinite_share": float(np.mean(st == STOP_NONFINITE)) if chain else float("nan"),
            "steps_median": float(np.median(sp)) if chain else float("nan"),
            "steps_p90": float(np.quantile(sp, 0.9)) if chain else float("nan"),
        })
    return rows


def pool_agreement_rows(arrays):
    """Sensitivity to initialization: per configuration, the two pools' score difference and the
    largest coordinate difference of their returned vectors in units of the pool IQR."""
    score, theta = np.asarray(arrays["score"]), np.asarray(arrays["theta"])
    iqr = np.asarray(arrays["iqr"]).mean(axis=1)                        # (N, D)
    rows = []
    for k, name in enumerate(CONFIG_ORDER):
        ds = np.abs(score[:, 0, k] - score[:, 1, k])
        dt = (np.abs(theta[:, 0, k, :] - theta[:, 1, k, :]) / iqr).max(axis=1)
        rows.append({"config": name, "score_diff_median": float(np.median(ds)),
                     "score_diff_p90": float(np.quantile(ds, 0.9)), "score_diff_max": float(ds.max()),
                     "theta_diff_iqr_median": float(np.median(dt)),
                     "theta_diff_iqr_p90": float(np.quantile(dt, 0.9)), "theta_diff_iqr_max": float(dt.max())})
    return rows
