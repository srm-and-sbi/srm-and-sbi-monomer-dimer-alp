"""Re-render figures/correlations_cell_<c>.png of an existing Window_Posterior_Draws collection from its saved
cloud (the same numbers as the collection run; tick labels were overlapping in the first rendering).
Usage: python rerender_correlation_heatmaps.py <collection folder>"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
d = Path(sys.argv[1]); S = np.load(d / (d.name + ".npz"), allow_pickle=False)
keys = [str(k) for k in S["parameter_keys"]]; D = len(keys); cl = S["posterior_samples_cloud"]; cond = str(S["kinds"][0])
pairs = [(a, b) for a in range(D) for b in range(a + 1, D)]
for c in sorted(set(S["cell"].tolist())):
    rows = [int(np.where((S["cell"] == c) & (S["chunk"] == w))[0][0]) for w in range(10)]
    M = np.array([[np.corrcoef(cl[i].astype(float), rowvar=False)[a, b] for a, b in pairs] for i in rows])
    Ms = np.array([[spearmanr(cl[i].astype(float)).statistic[a, b] for a, b in pairs] for i in rows])
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, mat, name in ((axes[0], M, "Pearson (estimator coordinates, log10)"), (axes[1], Ms, "Spearman")):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pairs))); ax.set_xticklabels([f"{keys[a]} x {keys[b]}" for a, b in pairs], fontsize=7, rotation=60, ha="right"); ax.set_yticks(range(10)); ax.set_ylabel("chunk")
        for (y, x), v in np.ndenumerate(mat): ax.text(x, y, f"{v:+.2f}", ha="center", va="center", fontsize=6)
        ax.set_title(f"{cond} cell {c}: within-window {name} correlations of the fresh draws", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8); fig.savefig(d / "figures" / f"correlations_cell_{c}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("re-rendered", d.name, "cell", c)
