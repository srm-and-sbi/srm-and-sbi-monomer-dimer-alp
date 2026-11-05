"""One side-by-side experimental-versus-synthetic video from a persisted posterior-predictive clip: all source frames
at the native frame rate (no subsampling, no retiming), magma, origin lower, one fixed ADU window shared by both
panels, labels (condition, cell, receptor total, elapsed time), nearest-neighbour 2x upscale for legibility. Lossless
encodings only (the project's rule for science images): FFV1 in .mkv (the repository's convention) and VP9 lossless in
.webm (browser-playable). Usage: python assemble_pair_video.py <clip.npz> <out_stem> <vmin> <vmax> <label>"""
import sys, json, subprocess, hashlib
from pathlib import Path
import numpy as np
import matplotlib
from PIL import Image, ImageDraw, ImageFont
FFMPEG = "/home/mars-fias/anaconda3/envs/READY_MARS/bin/ffmpeg"
clip, out_stem, vmin, vmax, label = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
z = np.load(clip, allow_pickle=False)
exp, syn = z["experimental"], z["synth"]; n = int(z["n_frames"]); dt = float(z["frame_time_seconds"]); fps = round(1 / dt)
assert exp.shape == syn.shape == (n, 256, 256) and fps == 50
rds = dict(zip([str(k) for k in z["rds_keys"]], [float(v) for v in z["rds_provenance"]]))
N = int(round(rds["count_total"])); kind = str(z["kind"]); cell = int(z["cell"])
cmap = matplotlib.colormaps["magma"]; SCALE = 2; H = 256 * SCALE; W = 256 * SCALE; TOP = 44; GAP = 8
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16); fontb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
def panel(frame):
    a = np.clip((frame.astype(np.float64) - vmin) / (vmax - vmin), 0, 1)
    rgb = (cmap(a)[:, :, :3] * 255).astype(np.uint8)
    rgb = rgb[::-1]                                    # origin="lower": row 0 at the bottom, as in the figures and players
    return np.repeat(np.repeat(rgb, SCALE, axis=0), SCALE, axis=1)
WIDTH = 2 * W + GAP; HEIGHT = H + TOP + 24
sat_exp = int(np.count_nonzero(exp > vmax)); sat_syn = int(np.count_nonzero(syn > vmax))
cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-"]
mkv = out_stem.with_suffix(".mkv"); webm = out_stem.with_suffix(".webm")
procs = [subprocess.Popen(cmd + ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24", str(mkv)], stdin=subprocess.PIPE),
         subprocess.Popen(cmd + ["-c:v", "libvpx-vp9", "-lossless", "1", "-pix_fmt", "gbrp", "-row-mt", "1", "-threads", "6", str(webm)], stdin=subprocess.PIPE)]
for i in range(n):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0)); d = ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(panel(exp[i])), (0, TOP)); canvas.paste(Image.fromarray(panel(syn[i])), (W + GAP, TOP))
    d.text((6, 4), f"EXPERIMENTAL  {kind} cell {cell}", fill=(255, 255, 255), font=fontb)
    d.text((W + GAP + 6, 4), f"SYNTHETIC  {label}  count_total {N:,}", fill=(255, 255, 255), font=fontb)
    d.text((6, 24), f"t = {i * dt:5.2f} s   frame {i}/{n - 1}   {fps} fps   window [{vmin:.0f}, {vmax:.0f}] ADU, fixed, both panels   magma, origin lower", fill=(200, 200, 200), font=font)
    d.text((6, TOP + H + 4), "all 1000 source frames, 20 ms each; lossless encoding; a viewing derivative of the persisted clip", fill=(150, 150, 150), font=font)
    buf = np.asarray(canvas).tobytes()
    for p in procs: p.stdin.write(buf)
for p in procs:
    p.stdin.close(); assert p.wait() == 0, "ffmpeg failed"
def probe(path):
    out = subprocess.run([FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height,codec_name", "-of", "json", str(path)], capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]
rec = {"clip": clip.name, "kind": kind, "cell": cell, "count_total": N, "seed": int(z["seed"]), "window_adu": [vmin, vmax], "fps": fps, "n_frames": n,
       "saturated_pixel_samples_above_window": {"experimental": sat_exp, "synthetic": sat_syn, "of": int(exp.size)},
       "labeling": {str(k): float(v) for k, v in zip(z["labeling_columns"], z["labeling_row"])},
       "files": {str(p.name): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "probe": probe(p)} for p in (mkv, webm)}}
print(json.dumps(rec, indent=1)); json.dump(rec, open(out_stem.with_name(out_stem.name + "_video_record.json"), "w"), indent=1)
