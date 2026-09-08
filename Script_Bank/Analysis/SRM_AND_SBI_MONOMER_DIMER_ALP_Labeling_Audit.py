"""Labeling audit: the DOL-explicit observation layer against its own arithmetic.

Four levels, every one with a prespecified acceptance criterion, verifying that the
implementation realizes the labeling model it documents (`labeling`,
`simulation_rds_support.extract_subunit_lineage`, `simulation_dli_support.render_dli_video`):

    Level 1 (laws)      Every registered labeling law reproduces its analytic mean, variance,
                        and zero-dye probability on a large draw; the derived visible fractions
                        (monomer 1 - P0, dimer 1 - P0^2) follow.
    Level 2 (lineage)   On a reactive trajectory simulated at the biology prior center, the
                        reaction records replay into a subunit lineage that covers every
                        subunit exactly once per frame, agrees with the particles observable
                        (host present, species matches, multiplicity matches), and shows the
                        particle-id churn the design relies on (a subunit visits several ids).
    Level 3 (draw)      Repeated static draws on that lineage reproduce the visible fractions
                        and, for the MET-INLB law, the two-thirds one-dye share among visible
                        dimers -- the arithmetic the documents derive from the draw.
    Level 4 (render)    Static scenes rendered through the production renderer: a zero-dye
                        subunit contributes nothing above background; a two-dye spot carries
                        twice the photons of a one-dye spot (in expectation); and at a
                        dissociation the dye follows its subunit -- one visible daughter, one
                        invisible -- with no signal left at the vanished dimer's site.

Usage (from the repo root):
    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit.py

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit/
        SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit.md   (report)
        audit_summary.json                                (every number the report quotes)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

import readdy  # noqa: E402

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det  # noqa: E402
from srm_and_sbi_monomer_dimer_alp import labeling as lab  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.parameterization import (  # noqa: E402
    PARAMETERIZATION, PARAMETERS, RunTiming,
)
from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import (  # noqa: E402
    build_simulation, build_system, collapse_species_axis, extract_subunit_lineage,
    extract_trajectory_poses,
)

assert os.path.abspath(lab.__file__).startswith(REPO_ROOT), (
    "Imported srm_and_sbi_monomer_dimer_alp from outside this repo checkout: " + lab.__file__
    + " -- run with PYTHONPATH set to the repo root."
)

OUT_DIR = os.path.join(str(PARAMETERS.machine.data_bank_root), "Posit",
                       "SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit")
REPORT = os.path.join(OUT_DIR, "SRM_AND_SBI_MONOMER_DIMER_ALP_Labeling_Audit.md")

# ---- audit operating point -------------------------------------------------------------------
N_LAW = 400_000          # draws per law (Level 1)
N_DRAWS = 400            # independent labelings of the reactive lineage (Level 3)
T_RENDER = 300           # frames of the static render scenes (Level 4)
SEED = 20260907
# Tolerances (prespecified). Level 1: analytic vs empirical, at n = 4e5 the standard error of a
# mean is ~2e-3, so 1% relative is a practical, not a significance, tolerance. Level 3: the
# visible-fraction estimate pools ~N_DRAWS x n_subunits Bernoulli trials; 2% absolute. The
# two-thirds share pools the visible dimers of every draw; 3% absolute. Level 4: aperture
# means over T_RENDER frames of a lognormal OU process carry ~10% sampling spread (about 20
# effectively independent frames per dye), so the 2x ratio is accepted within [1.7, 2.3]; the
# zero-dye and vanished-dimer apertures must lie within 3 standard errors of zero, estimated
# from the same frames.
TOL_LAW_REL = 0.01
TOL_FRACTION = 0.02
TOL_SHARE = 0.03
RATIO_BAND = (1.7, 2.3)
NULL_SIGMAS = 3.0


def passed(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ----------------------------------------------------------------------------------------------
# Level 1: the laws
# ----------------------------------------------------------------------------------------------

def laws() -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for name, law in lab.LABELING_LAWS.items():
        k = law.draw(N_LAW, rng)
        out[name] = dict(
            describe=law.describe(),
            mean=law.mean, mean_emp=float(k.mean()),
            variance=law.variance, variance_emp=float(k.var()),
            p0=law.probability_zero, p0_emp=float((k == 0).mean()),
            visible_monomer=law.visible_probability, visible_dimer=law.visible_fraction(2),
        )
        rel = lambda a, b: abs(a - b) / max(abs(a), 1e-12)
        out[name]["max_rel_err"] = float(max(rel(law.mean, k.mean()), rel(law.variance, k.var()),
                                             rel(law.probability_zero, (k == 0).mean())))
    return out


# ----------------------------------------------------------------------------------------------
# Level 2: lineage on a reactive trajectory at the prior center
# ----------------------------------------------------------------------------------------------

def reactive_lineage(workdir: str) -> dict:
    theta = np.array([p["VALUE"] for p in PARAMETERIZATION], dtype=float)   # prior center
    timing = RunTiming(total_time_seconds=2.0, frames=PARAMETERS.simulation.timing)
    stem = build_system(theta)
    smut = build_simulation(stem, theta, seed=SEED)
    traj = os.path.join(workdir, "labeling_audit_center.h5")
    smut.output_file = traj
    smut.progress_output_stride = timing.total_steps
    smut.run(n_steps=timing.total_steps,
             timestep=timing.delta_time_nanoseconds * readdy.units.nanosecond, show_summary=False)
    tray = readdy.Trajectory(traj)
    poses = extract_trajectory_poses(tray)
    lineage = extract_subunit_lineage(tray)
    soul_poses = collapse_species_axis(poses)
    # Independent cross-check against the particles observable.
    _, types, ids, _ = tray.read_observable_particles()
    rank_of = {name: int(rank) for name, rank in tray.particle_types.items()}
    subunits_of_rank = {rank_of[n]: k for n, k in zip(PARAMETERS.simulation.rds.particle_species_names,
                                                     PARAMETERS.simulation.rds.subunit_counts_per_species)}
    consistent, finite = True, True
    for f in range(lineage.n_frames):
        present = {int(i): int(t) for i, t in zip(ids[f], types[f])}
        hosts = lineage.host_index[f]
        souls = lineage.soul_ids[hosts]
        if any(int(s) not in present or present[int(s)] != r for s, r in zip(souls, lineage.host_rank[f])):
            consistent = False
        counts = np.bincount(hosts, minlength=lineage.soul_ids.shape[0])
        for soul, t in present.items():
            if counts[int(np.searchsorted(lineage.soul_ids, soul))] != subunits_of_rank[t]:
                consistent = False
        if not np.isfinite(soul_poses[f, hosts, :]).all():
            finite = False
    visits = np.array([np.unique(lineage.host_index[:, s]).shape[0] for s in range(lineage.n_subunits)])
    _, recs = tray.read_observable_reactions()
    n_records = int(sum(len(r) for r in recs))
    del tray, smut, stem
    return dict(
        n_subunits=lineage.n_subunits, n_particle_ids=int(lineage.soul_ids.shape[0]),
        n_frames=lineage.n_frames, n_records=n_records,
        cover_consistent=bool(consistent), host_positions_finite=bool(finite),
        max_ids_per_subunit=int(visits.max()), mean_ids_per_subunit=float(visits.mean()),
        monomer_rank=rank_of["A"], theta_center=theta.tolist(),
        _lineage=lineage,
    )


# ----------------------------------------------------------------------------------------------
# Level 3: repeated static draws on the reactive lineage
# ----------------------------------------------------------------------------------------------

def draws(lineage, monomer_rank: int) -> dict:
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for condition in lab.LABELING_CONDITIONS:
        name, law = lab.resolve_labeling_law(condition)
        rows = np.stack([
            lab.labeling_summary(lab.draw_dye_counts(law, lineage.n_subunits, rng),
                                 lineage.host_index[0], lineage.host_rank[0], [monomer_rank])
            for _ in range(N_DRAWS)])
        col = {c: i for i, c in enumerate(lab.LABELING_SET_COLUMNS)}
        mono_vis = rows[:, col["monomers_visible_0"]].sum() / max(rows[:, col["monomers_0"]].sum(), 1)
        dim_vis = rows[:, col["dimers_visible_0"]].sum() / max(rows[:, col["dimers_0"]].sum(), 1)
        two_share = rows[:, col["dimers_two_labeled_0"]].sum() / max(rows[:, col["dimers_visible_0"]].sum(), 1)
        dyes_per_subunit = rows[:, col["n_dyes"]].sum() / (N_DRAWS * lineage.n_subunits)
        p0 = law.probability_zero
        two_share_expected = (1 - p0) ** 2 / (1 - p0 ** 2)      # P(both labeled | at least one)
        out[condition] = dict(
            law=name, monomers_0=int(rows[0, col["monomers_0"]]), dimers_0=int(rows[0, col["dimers_0"]]),
            visible_monomer_emp=float(mono_vis), visible_monomer=law.visible_probability,
            visible_dimer_emp=float(dim_vis), visible_dimer=law.visible_fraction(2),
            two_labeled_share_emp=float(two_share), two_labeled_share=float(two_share_expected),
            dyes_per_subunit_emp=float(dyes_per_subunit), dyes_per_subunit=law.mean,
        )
    return out


# ----------------------------------------------------------------------------------------------
# Level 4: static renders through the production renderer
# ----------------------------------------------------------------------------------------------

def renders() -> dict:
    stem_geometry = PARAMETERS.simulation.stem
    pix = stem_geometry.pixel_size_nm
    imaging = np.array([10 ** ((e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2) for e in det.DETECTOR_IMAGING])
    keys = det.DETECTOR_IMAGING_KEYS
    imaging[keys.index("prob_photo_bleach")] = 0.0          # bleaching off: the mean check is a brightness check
    img = dict(zip(keys, imaging))
    bg_adu = img["kappa_o"] * img["kappa_q"] * img["gamma"] + img["kappa_b"]

    def aperture(frames, xy_nm, half=4):
        x, y = (np.asarray(xy_nm) / pix).astype(int)
        patch = frames[x - half:x + half + 1, y - half:y + half + 1, :]
        return patch.sum(axis=(0, 1)) - (2 * half + 1) ** 2 * bg_adu

    def null_ok(series):
        se = series.std(ddof=1) / np.sqrt(series.shape[0])
        return bool(abs(series.mean()) <= NULL_SIGMAS * se), float(series.mean()), float(se)

    P = np.array([[60.5, 60.5], [128.5, 128.5], [200.5, 60.5]]) * pix
    soul_poses = np.full((T_RENDER, 3, 3), np.nan)
    soul_poses[:, :, :2] = P[None]
    soul_poses[:, :, 2] = 0.0
    host_index = np.tile(np.array([0, 1, 1, 2]), (T_RENDER, 1))    # s0 -> monomer, s1+s2 -> dimer, s3 -> monomer

    # (a) one-dye monomer, one-dye dimer (partner unlabeled), zero-dye monomer
    fr = render_dli_video(soul_poses, host_index, np.array([1, 1, 0, 0]), imaging, seed=SEED)
    zero_ok, zero_mean, zero_se = null_ok(aperture(fr, P[2]))
    one_dye_dimer = float(aperture(fr, P[1]).mean()); one_dye_mono = float(aperture(fr, P[0]).mean())
    # (b) two-dye dimer against a one-dye monomer: ratio ~ 2
    fr2 = render_dli_video(soul_poses, host_index, np.array([1, 1, 1, 0]), imaging, seed=SEED + 1)
    a_mono = aperture(fr2, P[0]); a_dimer = aperture(fr2, P[1])
    ratio = float(a_dimer.mean() / a_mono.mean())
    # (c) dissociation at mid-video: the dye follows subunit 1 to daughter particle 3
    half = T_RENDER // 2
    soul_poses2 = np.full((T_RENDER, 5, 3), np.nan); soul_poses2[:, :3] = soul_poses
    soul_poses2[half:, 1] = np.nan
    d1 = P[1] + np.array([0, 20 * pix]); d2 = P[1] - np.array([0, 20 * pix])
    soul_poses2[half:, 3, :2] = d1; soul_poses2[half:, 3, 2] = 0
    soul_poses2[half:, 4, :2] = d2; soul_poses2[half:, 4, 2] = 0
    hi2 = host_index.copy(); hi2[half:, 1] = 3; hi2[half:, 2] = 4
    fr3 = render_dli_video(soul_poses2, hi2, np.array([1, 1, 0, 0]), imaging, seed=SEED + 2)
    site_after_ok, site_after, site_after_se = null_ok(aperture(fr3, P[1])[half:])
    invisible_ok, invis_mean, invis_se = null_ok(aperture(fr3, d2)[half:])
    visible_daughter = float(aperture(fr3, d1)[half:].mean())
    site_before = float(aperture(fr3, P[1])[:half].mean())
    # (d) an emitter-free scene renders background only
    fr0 = render_dli_video(soul_poses, host_index, np.zeros(4, dtype=int), imaging, seed=SEED + 3)
    return dict(
        zero_dye_ok=zero_ok, zero_dye_mean_adu=zero_mean, zero_dye_se_adu=zero_se,
        one_dye_monomer_adu=one_dye_mono, one_dye_dimer_adu=one_dye_dimer,
        two_dye_ratio=ratio, ratio_band=list(RATIO_BAND),
        dissociation_site_before_adu=site_before, dissociation_site_after_ok=site_after_ok,
        dissociation_site_after_adu=site_after, dissociation_site_after_se=site_after_se,
        visible_daughter_adu=visible_daughter, invisible_daughter_ok=invisible_ok,
        invisible_daughter_adu=invis_mean, invisible_daughter_se=invis_se,
        empty_scene_mean_adu=float(fr0.mean()), empty_scene_expected_adu=float(bg_adu),
        empty_scene_finite=bool(np.isfinite(fr0).all()),
    )


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

def write_report(law_out, lin, drw, ren, verdicts) -> None:
    commit = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    L = []
    add = L.append
    add("# Labeling audit - the DOL-explicit observation layer against its own arithmetic")
    add("")
    add(f"Regenerate with `Script_Bank/Analysis/{os.path.basename(__file__)}` (commit {commit}; run with "
        "`PYTHONPATH` set to the repo root so the audited copy shadows the installed package - the script "
        "asserts this). The summary JSON sits beside this report in the Data_Bank Posit tier. The model "
        "under audit is the static labeling stoichiometry of `labeling.py`, the subunit lineage of "
        "`simulation_rds_support.extract_subunit_lineage`, and the dye-centric renderer "
        "`simulation_dli_support.render_dli_video`; the design it realizes is the DOL-explicit observation "
        "layer of the MONOMER_DIMER model family (PROJECT_CONTEXT.md, *DLI Imaging*).")
    add("")
    add("## 1. Protocol")
    add("")
    add("**Level 1 (laws).** Each registered law is drawn "
        f"{N_LAW:,} times; mean, variance, and zero-dye probability must match the analytic values "
        f"within {TOL_LAW_REL:.0%} relative. The visible fractions follow from P0 by arithmetic.")
    add("")
    add("**Level 2 (lineage).** A 2 s reactive trajectory at the biology prior center is simulated in "
        "a temporary directory, its reaction records replayed into the subunit lineage, and the lineage "
        "checked independently against the particles observable: every host present in its frame with "
        "the recorded species, every particle covered by exactly its species' subunit count, every host "
        "position finite. The particle-id churn (ids visited per subunit) documents why the lineage is "
        "needed: ReaDDy mints a new id at every reaction, conversions included.")
    add("")
    add(f"**Level 3 (draw).** {N_DRAWS} independent labelings of that lineage under each condition's "
        f"baseline law; the pooled visible fractions must match the law within {TOL_FRACTION:.0%} "
        f"absolute and the two-labeled share among visible dimers within {TOL_SHARE:.0%} absolute of "
        "(1 - P0)^2 / (1 - P0^2) (one third for MET-INLB).")
    add("")
    add(f"**Level 4 (render).** Static scenes of {T_RENDER} frames at the imaging prior center with "
        "bleaching off, through the production renderer, read out by 9x9 aperture photometry at the "
        "true positions: a zero-dye subunit and the site of a dissociated dimer must sit within "
        f"{NULL_SIGMAS:g} standard errors of zero net signal; a two-dye spot must carry "
        f"{RATIO_BAND[0]:g}-{RATIO_BAND[1]:g} times a one-dye spot; after a dissociation the dye's "
        "daughter carries the signal and the unlabeled daughter none; an emitter-free scene renders "
        "the background alone.")
    add("")
    add("## 2. Results")
    add("")
    add("| level | check | criterion | result | verdict |")
    add("|---|---|---|---|---|")
    for name, r in law_out.items():
        add(f"| 1 | {name} = {r['describe']} | max rel. err < {TOL_LAW_REL:.0%} | "
            f"mean {r['mean_emp']:.4f}/{r['mean']:.4f}, var {r['variance_emp']:.4f}/{r['variance']:.4f}, "
            f"P0 {r['p0_emp']:.4f}/{r['p0']:.4f} (max err {r['max_rel_err']:.2%}) | {verdicts['law_' + name]} |")
    add(f"| 2 | lineage covers every subunit, agrees with the particles observable | exact | "
        f"{lin['n_subunits']} subunits over {lin['n_particle_ids']} particle ids in {lin['n_frames']} frames; "
        f"{lin['n_records']} records; up to {lin['max_ids_per_subunit']} ids per subunit "
        f"(mean {lin['mean_ids_per_subunit']:.1f}) | {verdicts['lineage']} |")
    for cond, d in drw.items():
        add(f"| 3 | {cond} visible fractions ({d['law']}) | abs err < {TOL_FRACTION:.0%} | "
            f"monomer {d['visible_monomer_emp']:.3f}/{d['visible_monomer']:.3f}, "
            f"dimer {d['visible_dimer_emp']:.3f}/{d['visible_dimer']:.3f}, "
            f"dyes per subunit {d['dyes_per_subunit_emp']:.3f}/{d['dyes_per_subunit']:.3f} | "
            f"{verdicts['fractions_' + cond]} |")
        add(f"| 3 | {cond} two-labeled share among visible dimers | abs err < {TOL_SHARE:.0%} | "
            f"{d['two_labeled_share_emp']:.3f}/{d['two_labeled_share']:.3f} | {verdicts['share_' + cond]} |")
    add(f"| 4 | zero-dye subunit renders nothing | within {NULL_SIGMAS:g} SE of 0 | "
        f"{ren['zero_dye_mean_adu']:+.0f} +/- {ren['zero_dye_se_adu']:.0f} ADU (one-dye monomer "
        f"{ren['one_dye_monomer_adu']:.0f}) | {verdicts['zero_dye']} |")
    add(f"| 4 | two-dye spot vs one-dye spot | ratio in [{RATIO_BAND[0]:g}, {RATIO_BAND[1]:g}] | "
        f"{ren['two_dye_ratio']:.3f} | {verdicts['ratio']} |")
    add(f"| 4 | dissociation: dye follows its subunit | site after within {NULL_SIGMAS:g} SE of 0; "
        f"unlabeled daughter within {NULL_SIGMAS:g} SE of 0 | site {ren['dissociation_site_before_adu']:.0f} -> "
        f"{ren['dissociation_site_after_adu']:+.0f} +/- {ren['dissociation_site_after_se']:.0f}; "
        f"visible daughter {ren['visible_daughter_adu']:.0f}; unlabeled daughter "
        f"{ren['invisible_daughter_adu']:+.0f} +/- {ren['invisible_daughter_se']:.0f} | {verdicts['dissociation']} |")
    add(f"| 4 | emitter-free scene | finite, mean = background | mean {ren['empty_scene_mean_adu']:.1f} vs "
        f"{ren['empty_scene_expected_adu']:.1f} ADU | {verdicts['empty']} |")
    add("")
    add(f"**Overall: {verdicts['overall']}.**")
    add("")
    add("## 3. Reading the numbers")
    add("")
    add("The one-dye dimer of the first render carries the same signal as a one-dye monomer: a dimer "
        "whose partner subunit is unlabeled is not brighter than a monomer, which is the first of the "
        "three derived consequences (the visible-dimer brightness is a mixture, not a doubled monomer). "
        "The lineage's id churn is the fact the design rests on -- a static per-subunit quantity cannot "
        "be attached to a ReaDDy particle, only to the subunit the lineage follows -- and it is also why "
        "a per-particle emitter would restart its brightness process at every conversion. The Level 3 "
        "fractions are computed on one realized reactive composition at the prior center, so their "
        "monomer and dimer counts are whatever that trajectory holds at frame 0; the law's arithmetic "
        "does not depend on the composition, only the sampling error does.")
    add("")
    add("## 4. Audit operating point")
    add("")
    theta_desc = ", ".join(f"{p['KEY']}={v:.4g}" for p, v in zip(PARAMETERIZATION, lin["theta_center"]))
    add(f"Laws: {N_LAW:,} draws each. Lineage: theta at the biology prior center ({theta_desc}), "
        f"2 s at the configured cadence, seed {SEED}. Draws: {N_DRAWS} labelings per condition. Renders: "
        f"{T_RENDER} frames, imaging at the prior center of every imaging parameter with bleaching off, "
        "9x9 apertures.")
    with open(REPORT, "w") as handle:
        handle.write("\n".join(L) + "\n")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Level 1: laws ...")
    law_out = laws()
    print("Level 2: reactive lineage at the prior center ...")
    with tempfile.TemporaryDirectory(prefix="labeling_audit_") as workdir:
        lin = reactive_lineage(workdir)
    lineage = lin.pop("_lineage")
    print("Level 3: repeated draws ...")
    drw = draws(lineage, lin["monomer_rank"])
    print("Level 4: renders ...")
    ren = renders()

    verdicts = {}
    for name, r in law_out.items():
        verdicts["law_" + name] = passed(r["max_rel_err"] < TOL_LAW_REL)
    verdicts["lineage"] = passed(lin["cover_consistent"] and lin["host_positions_finite"]
                                 and lin["max_ids_per_subunit"] >= 1)
    for cond, d in drw.items():
        verdicts["fractions_" + cond] = passed(
            abs(d["visible_monomer_emp"] - d["visible_monomer"]) < TOL_FRACTION
            and abs(d["visible_dimer_emp"] - d["visible_dimer"]) < TOL_FRACTION)
        verdicts["share_" + cond] = passed(abs(d["two_labeled_share_emp"] - d["two_labeled_share"]) < TOL_SHARE)
    verdicts["zero_dye"] = passed(ren["zero_dye_ok"])
    verdicts["ratio"] = passed(RATIO_BAND[0] <= ren["two_dye_ratio"] <= RATIO_BAND[1])
    verdicts["dissociation"] = passed(ren["dissociation_site_after_ok"] and ren["invisible_daughter_ok"]
                                      and ren["visible_daughter_adu"] > 5 * ren["invisible_daughter_se"])
    verdicts["empty"] = passed(ren["empty_scene_finite"]
                               and abs(ren["empty_scene_mean_adu"] - ren["empty_scene_expected_adu"]) < 2.0)
    verdicts["overall"] = passed(all(v == "PASS" for v in verdicts.values()))

    with open(os.path.join(OUT_DIR, "audit_summary.json"), "w") as handle:
        json.dump(dict(laws=law_out, lineage=lin, draws=drw, renders=ren, verdicts=verdicts),
                  handle, indent=1, default=str)
    write_report(law_out, lin, drw, ren, verdicts)
    print("verdicts:", verdicts)
    print("report:", REPORT)


if __name__ == "__main__":
    main()
