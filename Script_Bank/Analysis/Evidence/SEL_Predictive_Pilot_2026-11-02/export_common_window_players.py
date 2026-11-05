"""Viewing derivatives of the Fab cell-26 SEL pilot: lossless PNG-frame HTML players (the viewer notebook's
player mechanism, matplotlib to_jshtml) under ONE numeric ADU window shared by every panel and fixed over all
frames (the handoff manifest's common window), magma, origin lower, full field. Source arrays are read from the
persisted clips and never modified. Usage: python export_common_window_players.py <out_dir> [KIND CELL]  (default MET-FAB 26; the window is
computed by the manifest's rule when not the Fab cell-26 case: min of the four minima, max of the four p99.99)"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
mpl.rcParams["animation.embed_limit"] = 1024
KIND = sys.argv[2] if len(sys.argv) > 2 else "MET-FAB"; CELL = int(sys.argv[3]) if len(sys.argv) > 3 else 26
COND = KIND.split("-")[1]; TAG = f"{COND}{CELL}"
D = Path(f"/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_{COND}_2S_50FPS_Posterior_Predictive_Video")
STEM = f"SRM_AND_SBI_MONOMER_DIMER_ALP_{COND}_2S_50FPS_Declared_RDS_{KIND}_Cell_{CELL}_20S_{{}}_Synthetic_Video.npz"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
clips = {L: np.load(D / STEM.format(L), allow_pickle=False) for L in ("SEL_R1", "SEL_R2", "SEL_R3")}
exp = clips["SEL_R1"]["experimental"]
for L, z in clips.items():
    assert np.array_equal(z["experimental"], exp), L
n_frames = int(clips["SEL_R1"]["n_frames"]); dt = float(clips["SEL_R1"]["frame_time_seconds"]); fps = round(1 / dt)
seeds = {L: int(z["seed"]) for L, z in clips.items()}
_stacks = [exp] + [z["synth"] for z in clips.values()]
VMIN = float(min(a.min() for a in _stacks)); VMAX = float(max(np.percentile(a, 99.99) for a in _stacks))
print(f"common window by the manifest rule: [{VMIN:.0f}, {VMAX:.0f}] ADU")
assert n_frames == 1000 and fps == 50
HEAD = (f"{KIND} cell {CELL}, recording Experiment_{COND}_Cell_{CELL}_20S_RAW.tif (1000 frames at 50 fps, 20 s) beside the SEL "
        "synthetic realization(s) (declared Fab reaction-diffusion configuration, SEL imaging, MET camera). One display "
        f"window [{VMIN:.0f}, {VMAX:.0f}] ADU in every panel, fixed over all frames; colormap magma; origin lower; full field.")
def build(name, panels, frame_idx, play_fps, title_note):
    fig, ax = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.9))
    ax = np.atleast_1d(ax); ims = []
    for a, (title, stack) in zip(ax, panels):
        im = a.imshow(stack[frame_idx[0]], cmap="magma", origin="lower", interpolation="none", vmin=VMIN, vmax=VMAX)
        a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([]); ims.append(im)
    plt.close(fig)
    def _update(i):
        for im, (_, stack) in zip(ims, panels):
            im.set_data(stack[i])
        fig.suptitle(f"frame {i}/{n_frames - 1}   t = {i * dt:5.2f} s   {title_note}   window [{VMIN:.0f}, {VMAX:.0f}] ADU", fontsize=10)
        return ims
    anim = animation.FuncAnimation(fig, _update, frames=frame_idx, interval=1000.0 / play_fps, blit=False)
    player = anim.to_jshtml()
    n_emb = player.count("data:image/png;base64")
    assert n_emb == len(frame_idx), f"{name}: {n_emb} of {len(frame_idx)} frames embedded (truncated)"
    page = ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>" + name + "</title></head><body style='font-family:sans-serif'>"
            f"<h2>{name}</h2><p>{HEAD}</p><p>{title_note}. Source frames shown: {frame_idx[0]} to {frame_idx[-1]} "
            f"(every {frame_idx[1]-frame_idx[0]}th of {n_frames}); displayed at {play_fps:g} fps so that elapsed time is preserved; "
            f"{len(frame_idx)} lossless PNG frames embedded. Seeds: " + ", ".join(f"{L} {s}" for L, s in seeds.items()) + ".</p>"
            + player + "</body></html>")
    p = out / f"{name}.html"; p.write_text(page)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    rec = dict(file=p.name, bytes=p.stat().st_size, frames_embedded=n_emb, first_frame=frame_idx[0], last_frame=frame_idx[-1],
               stride=frame_idx[1] - frame_idx[0], display_fps=play_fps, duration_s=len(frame_idx) / play_fps,
               window_adu=[VMIN, VMAX], panels=[t for t, _ in panels], sha256=sha)
    print(json.dumps(rec)); return rec
records = []
idx5 = list(range(0, n_frames, 5))          # 200 frames, 10 fps -> 20 s
for L in ("SEL_R1", "SEL_R2", "SEL_R3"):
    records.append(build(f"{TAG}_experimental_vs_{L}_20s_every5", [(f"EXPERIMENTAL {KIND} cell {CELL}", exp), (f"SYNTHETIC {L} (seed {seeds[L]})", clips[L]["synth"])],
                         idx5, 10.0, "full 20 s, every 5th source frame at 10 fps"))
four = [(f"EXPERIMENTAL {KIND} cell {CELL}", exp)] + [(f"SYNTHETIC {L} (seed {seeds[L]})", clips[L]["synth"]) for L in ("SEL_R1", "SEL_R2", "SEL_R3")]
records.append(build(f"{TAG}_experimental_vs_R1_R2_R3_20s_every5", four, idx5, 10.0, "full 20 s, every 5th source frame at 10 fps, synchronized"))
records.append(build(f"{TAG}_experimental_vs_R1_R2_R3_opening4s_everyframe", four, list(range(0, 200)), 50.0, "opening 4 s, every source frame at 50 fps, synchronized"))
json.dump(dict(kind=KIND, cell=CELL, window_adu=[VMIN, VMAX], source_fps=fps, n_source_frames=n_frames, seeds=seeds, players=records), open(out / "players_manifest.json", "w"), indent=1)
print("ALL PLAYERS WRITTEN")
