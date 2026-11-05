"""Histogram comparison of a persisted render and its brightness arm against the experimental recording: same
experimental frames, same frame intervals (opening 0-99, middle 450-549, closing 900-999, pooled 0-999), same ADU bins,
density normalization; log-density view of the bright tail and a linear view; a quantile table per window.
Usage: python brightness_histograms.py <source_clip.npz> <arm_clip.npz> <out_stem>"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, arm, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
S = np.load(src, allow_pickle=False); A = np.load(arm, allow_pickle=False)
assert np.array_equal(S["experimental"], A["experimental"]) and np.array_equal(S["dye_counts"], A["dye_counts"])
keys = [str(k) for k in S["imaging_keys"]]; iS = dict(zip(keys, map(float, S["imaging_physical"]))); iA = dict(zip(keys, map(float, A["imaging_physical"])))
diff = {k: (iS[k], iA[k]) for k in keys if iS[k] != iA[k]}
exp = S["experimental"]; syn0 = S["synth"]; syn1 = A["synth"]
WIN = {"opening 0-2 s (frames 0-99)": slice(0, 100), "middle 9-11 s (frames 450-549)": slice(450, 550),
       "closing 18-20 s (frames 900-999)": slice(900, 1000), "pooled 0-20 s (frames 0-999)": slice(0, 1000)}
Q = [("min", 0), ("p0.01", 0.01), ("median", 50), ("p90", 90), ("p99", 99), ("p99.9", 99.9), ("p99.99", 99.99), ("max", 100)]
lab0 = f"synthetic, source ({', '.join(f'{k} {a:g}' for k, (a, _) in diff.items())})"; lab1 = f"synthetic, arm ({', '.join(f'{k} {b:g}' for k, (_, b) in diff.items())})"
series = [("experimental", exp, "tab:blue"), (lab0, syn0, "tab:orange"), (lab1, syn1, "tab:green")]
hi = max(float(a.max()) for _, a, _ in series); lo = min(float(a.min()) for _, a, _ in series)
bins = np.linspace(lo, hi, 400)                                   # the same ADU bins for every series and window
p9999 = max(float(np.percentile(a, 99.99)) for _, a, _ in series)
bins_lin = np.linspace(lo, p9999, 200)
fig, axes = plt.subplots(2, len(WIN), figsize=(5.2 * len(WIN), 8.6))
table = {}
for j, (wname, sl) in enumerate(WIN.items()):
    rows = {}
    for name, a, c in series:
        w = a[sl].ravel()
        axes[0, j].hist(w, bins=bins, histtype="step", density=True, color=c, label=name)
        axes[1, j].hist(w, bins=bins_lin, histtype="step", density=True, color=c, label=name)
        rows[name] = {q: float(np.percentile(w, p)) if 0 < p < 100 else float(w.min() if p == 0 else w.max()) for q, p in Q}
    axes[0, j].set_yscale("log"); axes[0, j].set_title(wname, fontsize=10); axes[0, j].set_xlabel("ADU"); axes[0, j].set_ylabel("density (log)")
    axes[1, j].set_xlabel(f"ADU (to the joint p99.99 = {p9999:.0f})"); axes[1, j].set_ylabel("density (linear)")
    table[wname] = rows
axes[0, 0].legend(fontsize=8); axes[1, 0].legend(fontsize=8)
kind = str(S["kind"]); cell = int(S["cell"]); N = int(round(dict(zip(map(str, S["rds_keys"]), map(float, S["rds_provenance"])))["count_total"]))
fig.suptitle(f"{kind} cell {cell}: experimental vs synthetic at count_total {N:,}, source vs brightness arm ({', '.join(f'{k} {a:g} -> {b:g}' for k, (a, b) in diff.items())}); "
             f"same trajectory, dye counts and seed; same bins; density per window", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out.with_suffix(".png"), dpi=130); plt.close(fig)
md = [f"# {kind} cell {cell}: brightness arm histograms", "", f"Source clip `{src.name}` (count_total {N:,}, {int((S['dye_counts']>0).sum())} labeled subunits, {int(S['dye_counts'].sum())} dyes, seed {int(S['seed'])}); "
      f"arm clip `{arm.name}`: the same trajectory, dye counts and seed, imaging override {', '.join(f'{k} {a:g} -> {b:g}' for k, (a, b) in diff.items())}; every other imaging value identical "
      f"({', '.join(f'{k} {iS[k]:g}' for k in keys if k not in diff)}). Quantiles in ADU; ratio = synthetic / experimental.", ""]
for wname, rows in table.items():
    md += [f"## {wname}", "", "| quantile | experimental | " + lab0 + " | ratio | " + lab1 + " | ratio |", "|---|---:|---:|---:|---:|---:|"]
    for q, _ in Q:
        e, s0, s1 = rows["experimental"][q], rows[lab0][q], rows[lab1][q]
        md.append(f"| {q} | {e:.0f} | {s0:.0f} | {s0/e:.2f} | {s1:.0f} | {s1/e:.2f} |")
    md.append("")
out.with_suffix(".md").write_text("\n".join(md)); json.dump({"source": src.name, "arm": arm.name, "override": diff, "windows": table}, open(out.with_suffix(".json"), "w"), indent=1)
print("\n".join(md))
