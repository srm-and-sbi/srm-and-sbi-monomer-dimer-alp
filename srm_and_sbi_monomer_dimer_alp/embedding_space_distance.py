"""Experimental-versus-synthetic domain distance in the learned embedding space.

Part of the Detector calibration workflow (see the quantitative experimental-versus-synthetic
distance in DETECTOR_WORKFLOW.md). Quantifies how far experimental recordings sit
from synthetic videos (the held-out EVAL set), measured in the trained
``Complex3DCNN`` embedding space, via two complementary two-sample statistics:

  - Maximum Mean Discrepancy (MMD; Gretton et al. 2012) -- a kernel two-sample
    distance with an RBF kernel (median-heuristic bandwidth) and a permutation
    p-value. Larger means a bigger gap.
  - Classifier Two-Sample Test (C2ST; Lopez-Paz & Oquab 2017) -- how well a classifier
    trained to tell experimental from synthetic ranks held-out samples, read as the area
    under the ROC curve (AUC). 0.5 means indistinguishable (a small gap); ->1.0 means
    easily separable (a large gap). The AUC is a rank statistic, so its no-information
    value is 0.5 whatever the sizes of the two sets; its null distribution is built by
    exchanging whole units.

Both resample at the CELL level: video chunks within one recording are correlated,
not independent, so the significance test permutes / splits whole cells rather than
individual chunks. A sample with no cell label is treated as its own block (i.e.
independent), which is the correct treatment for synthetic videos; ``matched_blocks``
groups independent synthetic videos into blocks of the recordings' sizes, so that a
relabeling of blocks keeps the number of windows on each side.

A third reading works per window instead of per set: ``knn_typicality`` ranks each
query embedding's distance to its nearest reference neighbors against the same
distance for the reference points themselves. It needs no classifier and gives one
value per window, which is what a handful of experimental windows against thousands
of synthetic videos can support.

Pure analysis: numpy + torch (for the embedding forward pass); scikit-learn and scipy
are imported lazily inside ``c2st`` so importing this module is cheap and its MMD path
carries no hard sklearn dependency. It imports nothing from ``parameterization`` /
``artifacts``, so it is unit-testable in isolation.
"""
from __future__ import annotations

import numpy as np
import torch


# =============================================================================
# Embedding
# =============================================================================

def embed_videos(posterior, videos, device=None, *, batch_size=16,
                 expected_frames=None, to_numpy=True):
    """Embed videos through the trained ``Complex3DCNN`` into CLS-token vectors.

    Args:
        posterior: a trained ``DirectPosterior`` (its ``posterior_estimator`` is the
            flow whose ``embedding_net`` is the ``Complex3DCNN``), or the flow itself.
        videos: sequence/array of ``(n_frames, H, W)`` videos (raw integer or already
            normalized); each is passed through ``normalize_video`` before embedding,
            the same preprocessing the training/evaluation path uses.
        device: torch device to embed on; defaults to the flow's own device.
        batch_size: videos embedded per forward pass (bounds GPU memory).
        expected_frames: if given, assert every video's frame axis equals it (guards a
            train/eval window-length mismatch).
        to_numpy: return a numpy array (default) or a CPU tensor.

    Returns:
        ``[N, D]`` embeddings (``D`` read from the network output, never hardcoded).
    """
    from .inference_support import normalize_video   # lazy: keeps the MMD/C2ST math import-light

    flow = getattr(posterior, "posterior_estimator", posterior)
    flow.eval()
    dev = torch.device(device) if device is not None else next(flow.parameters()).device
    out = []
    n = len(videos)
    for i in range(0, n, batch_size):
        batch = videos[i:i + batch_size]
        arr = np.stack([normalize_video(np.asarray(v)) for v in batch])   # (b, T, H, W) float32 [0,1]
        if expected_frames is not None and arr.shape[1] != expected_frames:
            raise ValueError(
                f"video frame axis {arr.shape[1]} != expected_frames {expected_frames}; "
                f"experimental and synthetic videos must be diced to the estimator's window.")
        x = torch.as_tensor(arr, dtype=torch.float32, device=dev)
        with torch.no_grad():
            emb = flow.embedding_net(x)                                   # [b, D] CLS embedding
        out.append(emb.detach().to("cpu"))
    emb_all = torch.cat(out, dim=0)
    return emb_all.numpy() if to_numpy else emb_all


# =============================================================================
# Cell-level block bookkeeping
# =============================================================================

def _blocks(n_x, n_y, cells_x, cells_y):
    """Return a per-sample integer block id for the pooled [X; Y] stack.

    A sample with a cell label shares a block with the other samples of that cell
    (experimental chunks within a recording); a sample with no label becomes its own block
    (independent -- the correct treatment for synthetic videos). Experimental and synthetic
    cell-id namespaces are kept disjoint so an experimental cell never merges with a synthetic
    one.
    """
    blocks = np.empty(n_x + n_y, dtype=object)
    for off, cnt, cells, tag in ((0, n_x, cells_x, "x"), (n_x, n_y, cells_y, "y")):
        if cells is None:
            for j in range(cnt):
                blocks[off + j] = (tag, "iid", j)
        else:
            cells = np.asarray(cells)
            for j in range(cnt):
                blocks[off + j] = (tag, "cell", cells[j])
    # map to contiguous integer ids
    uniq = {b: i for i, b in enumerate(dict.fromkeys(blocks.tolist()))}
    return np.array([uniq[b] for b in blocks.tolist()], dtype=int)


# =============================================================================
# MMD
# =============================================================================

def _sq_dists(Z):
    sq = np.einsum("ij,ij->i", Z, Z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    return np.maximum(d2, 0.0)


def _mmd2_unbiased(K, n):
    """Unbiased squared-MMD from a pooled Gram matrix ``K`` whose first ``n`` rows are X."""
    m = K.shape[0] - n
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    sxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) if n > 1 else 0.0
    syy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) if m > 1 else 0.0
    sxy = Kxy.mean()
    return float(sxx + syy - 2.0 * sxy)


def mmd_rbf(X, Y, *, bandwidth="median", n_permutations=1000, seed=None,
            cells_x=None, cells_y=None):
    """Unbiased squared-MMD (RBF kernel) with a cell-level permutation p-value.

    The bandwidth defaults to the median-heuristic (median of pooled squared pairwise
    distances). The permutation test reuses the single precomputed Gram matrix and
    permutes at the BLOCK level (whole cells; see ``_blocks``), so correlated chunks
    move together and the null respects the correlation structure.

    Returns a dict: ``mmd2``, ``mmd`` (=sqrt(max(mmd2,0))), ``sigma``, ``gamma``,
    ``p_value``, ``n``, ``m``, ``n_permutations``.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, m = len(X), len(Y)
    Z = np.vstack([X, Y]); L = n + m
    D2 = _sq_dists(Z)
    if bandwidth == "median":
        med = np.median(D2[np.triu_indices(L, k=1)])
        sigma2 = float(med) if med > 0 else 1.0
    else:
        sigma2 = float(bandwidth) ** 2
    gamma = 1.0 / (2.0 * sigma2)
    K = np.exp(-gamma * D2)
    observed = _mmd2_unbiased(K, n)

    block = _blocks(n, m, cells_x, cells_y)
    uniq_blocks = np.unique(block)
    # each block belongs wholly to X or Y in the observed split (X = indices < n)
    block_is_x = np.array([np.all(np.where(block == b)[0] < n) for b in uniq_blocks])
    n_x_blocks = int(block_is_x.sum())
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(uniq_blocks))
        x_blocks = set(uniq_blocks[perm[:n_x_blocks]])
        mask_x = np.array([block[i] in x_blocks for i in range(L)])
        order = np.concatenate([np.where(mask_x)[0], np.where(~mask_x)[0]])
        Kp = K[np.ix_(order, order)]
        if _mmd2_unbiased(Kp, int(mask_x.sum())) >= observed:
            ge += 1
    p = (1 + ge) / (n_permutations + 1)
    return {"mmd2": observed, "mmd": float(np.sqrt(max(observed, 0.0))),
            "sigma": float(np.sqrt(sigma2)), "gamma": float(gamma),
            "p_value": float(p), "n": n, "m": m, "n_permutations": int(n_permutations)}


# =============================================================================
# C2ST
# =============================================================================

def matched_blocks(sizes, n, rng=None):
    """Group ``n`` independent samples into blocks whose sizes repeat ``sizes``.

    ``sizes`` are the window counts of the experimental recordings (their blocks). Independent
    synthetic videos have no block structure, so any grouping of them is as good as another; grouping
    them into blocks of the SAME sizes makes a relabeling of whole blocks keep the number of windows on
    each side, which a permutation of unequal blocks (recordings of several windows against single
    videos) does not. The samples are assigned to blocks in random order (``rng``); the sizes are cycled
    in the order given, and samples left over when the next size no longer fits become single-sample
    blocks.

    Returns an integer block label per sample, shape ``(n,)``.
    """
    sizes = [int(s) for s in sizes]
    if not sizes or min(sizes) < 1:
        raise ValueError(f"matched_blocks needs positive block sizes; got {sizes}.")
    rng = np.random.default_rng() if rng is None else rng
    order = rng.permutation(int(n))
    labels = np.empty(int(n), dtype=np.int64)
    pos, block, i = 0, 0, 0
    while pos < n:
        size = sizes[i % len(sizes)]
        if pos + size > n:                      # the next block no longer fits: single-sample blocks
            size = 1
        labels[order[pos:pos + size]] = block
        pos += size; block += 1; i += 1
    return labels


def _auc(scores, labels):
    """Area under the ROC curve of ``scores`` for binary ``labels`` (1 = positive): the Mann-Whitney
    statistic with average ranks for ties. NaN when one class is absent."""
    from scipy.stats import rankdata
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum()); n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, float))
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _unit_labels(n, m, cells_x, cells_y):
    """Integer unit id per sample of the pooled ``[X; Y]`` stack (a recording, or a single sample when
    no label is given), with the X and Y namespaces kept apart."""
    gx = [("x", c) for c in (np.asarray(cells_x).tolist() if cells_x is not None else range(n))]
    gy = [("y", c) for c in (np.asarray(cells_y).tolist() if cells_y is not None else range(m))]
    gmap = {g: i for i, g in enumerate(dict.fromkeys(gx + gy))}
    return np.array([gmap[g] for g in (gx + gy)], dtype=np.int64)


def c2st(X, Y, *, classifier="logreg", n_folds=5, standardize=True, seed=None,
         cells_x=None, cells_y=None, n_permutations=200):
    """Classifier two-sample test: cross-validated AUC with a unit-level permutation null.

    Trains a classifier to separate X (label 0) from Y (label 1) under grouped, stratified k-fold
    cross-validation (a unit -- a recording, or a block of synthetic videos -- is never split between
    training and test, and every test fold holds units of both sets) and scores it by the area under the
    ROC curve of the held-out samples, averaged over the folds. The classifier weighs the two sets
    equally whatever their sizes (``class_weight="balanced"``; the ``mlp`` option, which has no class
    weights, trains on an equal-sized random subset of the larger set).

    The AUC is a rank statistic: with no information it is 0.5 for ANY ratio of the two set sizes,
    unlike the accuracy, whose no-information value is the share of the larger set.

    Null distribution: the unit labels are exchanged and the whole procedure repeated
    (``n_permutations`` times). Units are exchanged only with units of the same number of samples, so
    every relabeling keeps the number of units AND the number of samples on each side; pass synthetic
    blocks of the recordings' sizes (``matched_blocks``) so that every recording has partners. A unit
    with no partner of its size stays on its side. The p-value is the share of relabelings whose AUC
    is at least the observed one (one-sided, with the observed labeling counted); it is NaN when no
    unit can be exchanged or ``n_permutations`` is 0. What the null states: a recording's windows are
    exchangeable with a block of independent synthetic videos. Matching the block sizes does not make
    that true: windows of one recording are correlated, so the null can be rejected by that
    correlation alone, and with several windows per recording the AUC and its null range are the
    quantities to read, the p-value an approximate guide. With ONE window per recording, and
    recordings that are independent of each other, every unit is a single sample, the units are
    exchangeable when the two distributions are equal, and the permutation p-value is valid. A
    p-value above the level then says that no difference was detected at this sample size, not that
    the two distributions are equal.

    ``classifier``: ``"logreg"`` (default: L2-regularized logistic regression, cheap enough for the
    permutations; a linear classifier, so its AUC measures LINEAR separability: a nonlinear classifier
    can use structure it misses, but a fitted one is not guaranteed to score higher, so the linear AUC
    is not a bound on it), ``"rf"`` (random forest) or ``"mlp"`` (two-layer MLP, as in the sbi package).

    Returns a dict: ``auc``, ``auc_std``, ``per_fold``, ``null`` (0.5), ``null_mean``, ``null_range95``,
    ``p_value``, ``n_permutations``, ``oof_score`` (held-out P(class Y) per sample of ``[X; Y]``),
    ``classifier``, ``n_folds``, ``n``, ``m``, ``n_units_x``, ``n_units_y``, ``n_exchangeable_units_x``.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, m = len(X), len(Y)
    Z = np.vstack([X, Y]); y = np.concatenate([np.zeros(n), np.ones(m)]).astype(int)
    D = Z.shape[1]
    units = _unit_labels(n, m, cells_x, cells_y)
    unit_ids, unit_first, unit_index, unit_size = np.unique(units, return_index=True, return_inverse=True,
                                                            return_counts=True)
    unit_class = y[unit_first]                  # a unit lies wholly in X or in Y
    n_units_x = int((unit_class == 0).sum()); n_units_y = int((unit_class == 1).sum())
    n_splits = min(int(n_folds), n_units_x, n_units_y)
    if n_splits < 2:
        raise ValueError(f"c2st needs at least 2 units on each side for grouped cross-validation; "
                         f"got {n_units_x} (X) and {n_units_y} (Y).")

    def _make():
        if classifier == "mlp":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(hidden_layer_sizes=(10 * D, 10 * D), max_iter=1000, random_state=seed)
        if classifier == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=seed, class_weight="balanced")
        if classifier == "logreg":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, class_weight="balanced")
        raise ValueError(f"classifier={classifier!r}; expected 'logreg', 'rf' or 'mlp'.")

    def _statistic(labels, sample_units):
        """Mean held-out AUC over the folds for one labeling of the samples."""
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        aucs, oof = [], np.full(len(labels), np.nan)
        sub_rng = np.random.default_rng(seed)
        for tr, te in splitter.split(Z, labels, sample_units):
            if classifier == "mlp":                       # no class weights: equal-sized training subset
                pos, neg = tr[labels[tr] == 1], tr[labels[tr] == 0]
                small = min(len(pos), len(neg))
                tr = np.concatenate([sub_rng.choice(pos, small, replace=False),
                                     sub_rng.choice(neg, small, replace=False)])
            Ztr, Zte = Z[tr], Z[te]
            if standardize:
                sc = StandardScaler().fit(Ztr)
                Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
            clf = _make().fit(Ztr, labels[tr])
            score = clf.predict_proba(Zte)[:, 1]
            oof[te] = score
            a = _auc(score, labels[te])
            if not np.isnan(a):
                aucs.append(a)
        return np.array(aucs), oof

    aucs, oof = _statistic(y, units)
    auc = float(aucs.mean())
    sd = float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0

    # Units exchangeable under the null: the units of a size that both sets hold.
    exchangeable = np.zeros(len(unit_ids), dtype=bool)
    for size in np.unique(unit_size):
        members = unit_size == size
        if len(np.unique(unit_class[members])) == 2:
            exchangeable |= members
    n_exch_x = int((exchangeable & (unit_class == 0)).sum())
    null = np.array([])
    if n_permutations > 0 and n_exch_x > 0:
        rng = np.random.default_rng(seed)
        null = np.empty(int(n_permutations))
        for b in range(int(n_permutations)):
            permuted = unit_class.copy()
            for size in np.unique(unit_size[exchangeable]):
                members = np.flatnonzero(exchangeable & (unit_size == size))
                permuted[members] = unit_class[rng.permutation(members)]
            null[b] = _statistic(permuted[unit_index], units)[0].mean()
        p = float((1 + np.count_nonzero(null >= auc)) / (len(null) + 1))
        null_mean = float(null.mean())
        null_range = (float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975)))
    else:
        p, null_mean, null_range = float("nan"), float("nan"), (float("nan"), float("nan"))
    return {"auc": auc, "auc_std": sd, "per_fold": aucs.tolist(), "null": 0.5,
            "null_mean": null_mean, "null_range95": null_range, "p_value": p,
            "n_permutations": int(len(null)), "oof_score": oof.tolist(), "classifier": classifier,
            "n_folds": int(n_splits), "n": n, "m": m, "n_units_x": n_units_x, "n_units_y": n_units_y,
            "n_exchangeable_units_x": n_exch_x}


# =============================================================================
# Nearest-neighbor typicality (one value per window)
# =============================================================================

def _sorted_knn(reference, query, k, block=1024):
    """Distances to and indices of the ``k`` nearest reference points of each query point, sorted from
    nearest; ``query=None`` takes the reference points themselves, each excluded from its own
    neighbors (leave-one-out). Euclidean; computed in blocks of queries."""
    R = np.asarray(reference, float)
    loo = query is None
    Q = R if loo else np.asarray(query, float)
    limit = len(R) - 1 if loo else len(R)
    if not 1 <= int(k) <= limit:
        raise ValueError(f"k={k} is not in [1, {limit}] for {len(R)} reference points"
                         + (" (leave-one-out)." if loo else "."))
    rr = np.einsum("ij,ij->i", R, R)
    dist = np.empty((len(Q), k)); index = np.empty((len(Q), k), dtype=np.int64)
    for start in range(0, len(Q), block):
        q = Q[start:start + block]
        d2 = np.maximum(np.einsum("ij,ij->i", q, q)[:, None] + rr[None, :] - 2.0 * (q @ R.T), 0.0)
        if loo:
            d2[np.arange(len(q)), np.arange(start, start + len(q))] = np.inf
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        part_d2 = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(part_d2, axis=1)
        index[start:start + len(q)] = np.take_along_axis(part, order, axis=1)
        dist[start:start + len(q)] = np.sqrt(np.take_along_axis(part_d2, order, axis=1))
    return dist, index


def _pooled_typicality(R, q, k):
    """One query under pooled scaling: every coordinate is centered and divided by its standard
    deviation over the ``n + 1`` points (the reference and the query together), so the score of a point
    does not depend on which of them is called the query. Returns ``(tail_share, query_distance,
    percentile, neighbors)``."""
    n = len(R)
    Z = np.vstack([R, q[None, :]])
    scale = Z.std(axis=0)
    Z = (Z - Z.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    zz = np.einsum("ij,ij->i", Z, Z)
    d2 = np.maximum(zz[:, None] + zz[None, :] - 2.0 * (Z @ Z.T), 0.0)
    np.fill_diagonal(d2, np.inf)
    score = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])
    tail = np.count_nonzero(score >= score[n]) / (n + 1.0)          # the query counts itself
    percentile = 100.0 * np.count_nonzero(score[:n] < score[n]) / n
    return tail, score[n], percentile, np.argsort(d2[n, :n], kind="stable")[:k]


def knn_typicality(reference, query, *, k=10, standardize=False, block=1024):
    """How typical each query embedding is of a reference cloud, by nearest-neighbor distance.

    The score of a point is the distance to its ``k``-th nearest neighbor among the OTHER points of
    the set it is judged in: small in the dense body of the cloud, large at its edge and beyond. For
    one query point the set is the reference cloud plus that query; the query's score is compared with
    the scores of the reference points in that same set (a reference point's score changes when the
    query lands among its ``k`` nearest neighbors, and that is accounted for). ``tail_share`` is

        (1 + number of reference points whose score is at least the query's) / (n_reference + 1).

    If the query were one more independent draw from the distribution of the reference points, the
    ``n + 1`` points would be exchangeable and the score treats them alike, so a ``tail_share`` at or
    below ``a`` would occur with probability AT MOST ``a``: exactly ``a`` on the grid of attainable
    values when no two scores are equal, less when scores tie (ties count against flagging, so the
    reading is conservative). A small value therefore says the query lies farther from the cloud than
    almost all of the cloud's own points do from each other. That statement needs the exchangeability:
    it does not hold for a query that helped to produce the reference (an observation ranked among
    simulations drawn at its own posterior). Each query is judged on its own; queries from one
    recording are not independent, so the values of several windows are not combined into one p-value
    here. A query deep inside the cloud gets a large value: only being far out is flagged.

    ``standardize`` rescales the coordinates before the distances are taken. Raw distances
    (``False``) are dominated by the coordinates with the largest spread; a standardized reading weighs
    all coordinates alike, low-variance ones included. Two scalings:

    * ``True`` -- center and scale by the REFERENCE cloud alone. Cheap, and the reference scores are
      computed once. The query is then treated differently from the reference points (it had no part
      in the scale), so the statement above is not guaranteed: a query far out in a coordinate is
      measured with a scale it did not widen. With a large reference the departure is small, but the
      value is a DESCRIPTIVE rank, not one with a stated rate.
    * ``"pooled"`` -- center and scale by the reference and the query together, anew for every query.
      The score is then symmetric in the ``n + 1`` points and the statement above holds as for raw
      distances. Costs one full distance matrix of the reference per query: meant for small reference
      sets (a window's predictive videos).

    The raw and the standardized readings are complementary and neither is the 'correct' geometry.

    Returns a dict: ``k``, ``n_reference``, ``standardized``, ``scaling`` (``"none"``,
    ``"reference"`` or ``"pooled"``), ``query_distance`` (the query's ``k``-th-neighbor distance),
    ``reference_distance`` (each reference point's leave-one-out ``k``-th-neighbor distance; under
    pooled scaling in the reference-scaled geometry, as a description of the cloud), ``tail_share``,
    ``percentile`` (share of the reference scores below the query's, in percent) and ``neighbors``
    (the indices of the query's ``k`` nearest reference points, nearest first).
    """
    R = np.asarray(reference, float); Q = np.atleast_2d(np.asarray(query, float))
    if R.ndim != 2 or Q.shape[1] != R.shape[1]:
        raise ValueError(f"reference {R.shape} and query {Q.shape} must be 2-D with one embedding width.")
    pooled = isinstance(standardize, str)
    if pooled and standardize != "pooled":
        raise ValueError(f"standardize={standardize!r}; expected False, True or 'pooled'.")
    if standardize:
        center = R.mean(axis=0)
        scale = R.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        R_scaled = (R - center) / scale
    n = len(R)
    if pooled:
        ref_k = _sorted_knn(R_scaled, None, k, block)[0][:, -1]
        out = [_pooled_typicality(R, q, k) for q in Q]
        return {"k": int(k), "n_reference": int(n), "standardized": True, "scaling": "pooled",
                "query_distance": np.array([o[1] for o in out]), "reference_distance": ref_k,
                "tail_share": np.array([o[0] for o in out]), "percentile": np.array([o[2] for o in out]),
                "neighbors": np.array([o[3] for o in out], dtype=np.int64).reshape(len(Q), k)}
    if standardize:
        R, Q = R_scaled, (Q - center) / scale
    ref_knn, _ = _sorted_knn(R, None, k, block)
    ref_k = ref_knn[:, -1]
    ref_k_minus_1 = ref_knn[:, -2] if k > 1 else np.zeros(n)
    query_knn, neighbors = _sorted_knn(R, Q, k, block)
    query_k = query_knn[:, -1]
    rr = np.einsum("ij,ij->i", R, R)
    tail = np.empty(len(Q))
    for start in range(0, len(Q), block):
        q = Q[start:start + block]
        d = np.sqrt(np.maximum(np.einsum("ij,ij->i", q, q)[:, None] + rr[None, :] - 2.0 * (q @ R.T), 0.0))
        # A reference point's score in the set that includes the query: the query replaces its k-th
        # neighbor when it lies closer than that neighbor.
        score = np.where(d < ref_k[None, :], np.maximum(ref_k_minus_1[None, :], d), ref_k[None, :])
        at_least = np.count_nonzero(score >= query_k[start:start + len(q), None], axis=1)
        tail[start:start + len(q)] = (1.0 + at_least) / (n + 1.0)
    ref_sorted = np.sort(ref_k)
    percentile = 100.0 * np.searchsorted(ref_sorted, query_k, side="left") / n
    return {"k": int(k), "n_reference": int(n), "standardized": bool(standardize),
            "scaling": "reference" if standardize else "none",
            "query_distance": query_k, "reference_distance": ref_k, "tail_share": tail,
            "percentile": percentile, "neighbors": neighbors}


# =============================================================================
# Orchestrator + report table
# =============================================================================

def compute_distance(posterior, experimental_videos, synth_videos, *, device=None,
                     batch_size=16, expected_frames=None, experimental_cells=None,
                     synth_cells=None, mmd_kwargs=None, c2st_kwargs=None, seed=None):
    """Embed both sets and run MMD + C2ST; returns a combined dict incl. the embeddings.

    ``experimental_cells`` is REQUIRED: the experimental recordings are cell-correlated (chunks within a
    recording are not independent), so the cell-level resampling needs per-video cell
    labels; omitting them would silently degrade to an iid (anti-conservative) test.
    ``synth_cells`` may be None — synthetic videos are independent draws (each its own
    block).
    """
    if experimental_cells is None:
        raise ValueError(
            "compute_distance requires experimental_cells (one cell/recording label per experimental video): "
            "experimental chunks within a recording are correlated, so the cell-level resampling that "
            "§8 specifies needs them; omitting them would silently treat correlated chunks as "
            "independent. Pass experimental_cells=<per-video cell id>. (synth_cells may be None.)")
    er = embed_videos(posterior, experimental_videos, device=device, batch_size=batch_size,
                      expected_frames=expected_frames)
    es = embed_videos(posterior, synth_videos, device=device, batch_size=batch_size,
                      expected_frames=expected_frames)
    mmd = mmd_rbf(er, es, cells_x=experimental_cells, cells_y=synth_cells, seed=seed,
                  **(mmd_kwargs or {}))
    c = c2st(er, es, cells_x=experimental_cells, cells_y=synth_cells, seed=seed,
             **(c2st_kwargs or {}))
    return {"mmd": mmd, "c2st": c, "n_experimental": len(er), "n_synth": len(es),
            "embed_dim": int(er.shape[1]), "emb_experimental": er, "emb_synth": es}


def distance_table(result):
    """Render a ``compute_distance`` result as ``(headers, rows)`` for the reporter."""
    mmd, c = result["mmd"], result["c2st"]
    headers = ["metric", "value", "null / ref", "p-value", "reading"]
    rows = [
        ["MMD (unbiased sq.)", f"{mmd['mmd2']:.4g}", "0 (identical)",
         f"{mmd['p_value']:.3g}", "larger => bigger gap"],
        ["MMD (sqrt)", f"{mmd['mmd']:.4g}", f"RBF sigma={mmd['sigma']:.3g}", "", ""],
        ["C2ST AUC", f"{c['auc']:.3f}",
         f"0.5; null 95% [{c['null_range95'][0]:.3f},{c['null_range95'][1]:.3f}]", f"{c['p_value']:.3g}",
         "closer to 0.5 => smaller gap"],
    ]
    return headers, rows
