"""Pilot companion: PSF width, background and signal of persisted posterior-predictive clips, experimental vs
synthetic, opening window (frames 0-99) and closing window (frames 900-999). Widths by the direct estimator's
per-frame fit (direct_imaging_estimates.measure_spot_widths) on the 8-bit conversion the companions use, with the
MET camera values the render pins. Usage: python pilot_width_background.py <KIND> <cell> <label> [<label> ...]"""
import sys, json
from pathlib import Path
import numpy as np
from srm_and_sbi_monomer_dimer_alp import io as pio
from srm_and_sbi_monomer_dimer_alp.direct_imaging_estimates import measure_spot_widths
from srm_and_sbi_monomer_dimer_alp.posterior_predictive_video_runner import MET_CAMERA_PHYSICAL
kind, cell, labels = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
cond = kind.split("-")[1]
D = Path("/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit") / f"SRM_AND_SBI_MONOMER_DIMER_ALP_{cond}_2S_50FPS_Posterior_Predictive_Video"
WIN = {"opening": slice(0, 100), "closing": slice(900, 1000)}
def summarize(video16, name):
    out = {}
    v8 = pio.convert_video_dtype(np.asarray(video16), 16, 8)
    for wname, sl in WIN.items():
        w16 = np.asarray(video16[sl], dtype=np.float64)
        bg = np.median(w16, axis=(1, 2))                      # per-frame background median (ADU)
        signal = np.clip(w16 - bg[:, None, None], 0, None).sum(axis=(1, 2))   # per-frame signal above the median (ADU)
        m = measure_spot_widths(v8[sl], MET_CAMERA_PHYSICAL, frame_stride=5)
        wd = m["sqrt2sigma"]; good = np.isfinite(wd) & (wd > 0.3) & (wd < 6)
        out[wname] = dict(background_median_adu=float(np.median(bg)), signal_above_bg_per_frame=float(np.median(signal)),
                          p99_adu=float(np.percentile(w16, 99)), p999_adu=float(np.percentile(w16, 99.9)),
                          frame_max_median=float(np.median(w16.max(axis=(1, 2)))),
                          width_median_px=float(np.median(wd[good])) if good.any() else float("nan"),
                          width_iqr_px=[float(np.percentile(wd[good], 25)), float(np.percentile(wd[good], 75))] if good.any() else None,
                          spot_fits=int(good.sum()), frames_used=int(m["n_frames_used"]),
                          amplitude_median_adu=float(np.median(m["amplitude"][good])) if good.any() else float("nan"))
    return out
rows = {}
exp_done = None
for L in labels:
    f = D / f"SRM_AND_SBI_MONOMER_DIMER_ALP_{cond}_2S_50FPS_Declared_RDS_{kind}_Cell_{cell}_20S_{L}_Synthetic_Video.npz"
    z = np.load(f, allow_pickle=True)
    if exp_done is None:
        exp_done = summarize(z["experimental"], "experimental"); rows["experimental"] = exp_done
    rows[L] = summarize(z["synth"], L)
    rows[L]["seed"] = int(z["seed"]); rows[L]["n_dyes"] = int(z["n_dyes"]); rows[L]["n_subunits"] = int(z["n_subunits"])
print(json.dumps({"kind": kind, "cell": cell, "camera": MET_CAMERA_PHYSICAL, "windows": {k: [v.start, v.stop] for k, v in WIN.items()}, "rows": rows}, indent=1))
hdr = f"{'series':>12} {'window':>8} {'bg med':>7} {'signal/frame':>12} {'p99':>6} {'p99.9':>6} {'fmax med':>8} {'width med':>9} {'IQR':>13} {'fits':>5} {'amp med':>7}"
print("\n" + hdr)
for name, r in rows.items():
    for w in WIN:
        q = r[w]; iqr = q["width_iqr_px"]
        print(f"{name:>12} {w:>8} {q['background_median_adu']:7.0f} {q['signal_above_bg_per_frame']:12.3e} {q['p99_adu']:6.0f} {q['p999_adu']:6.0f} {q['frame_max_median']:8.0f} {q['width_median_px']:9.3f} {('%.2f-%.2f' % tuple(iqr)) if iqr else 'n/a':>13} {q['spot_fits']:5d} {q['amplitude_median_adu']:7.0f}")
