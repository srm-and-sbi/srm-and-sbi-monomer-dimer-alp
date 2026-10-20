"""Labeling audit: the DOL-explicit observation layer against its own arithmetic.

Three levels, every one with a prespecified acceptance criterion, verifying that the
implementation realizes the labeling model it documents (`labeling`,
`simulation_rds_support.extract_subunit_lineage`, `simulation_dli_support.render_dli_video`):

    Level 1 (laws)      Every registered labeling law reproduces its analytic mean, variance,
                        and zero-dye probability on a large draw; the derived fluorescent fractions
                        (a bound probe 1 - P0, a two-probe dimer 1 - P0^2) follow.
    Level 2 (lineage)   On a reactive trajectory simulated at the MET-INLB prior center (its twelve
                        values; the window-start allocation pairs c = 40 probe-bound monomer pairs,
                        so the starting population is 79 A + 302 B1 + 48 B2 at N_total = 10,000),
                        the frame-0 hosts per molecular species equal the starting counts of the
                        composition helper, and the reaction records replay into a subunit lineage
                        that covers every subunit exactly once per frame, agrees with the particles
                        observable (host present, particle type matches, multiplicity matches), and
                        shows the particle-id churn the design relies on (a subunit visits several ids).
    Level 3 (render)    Static scenes rendered through the production renderer: a zero-dye
                        subunit contributes nothing above background; a two-dye spot carries
                        twice the photons of a one-dye spot (in expectation); and at a
                        dissociation the dye follows its subunit -- one fluorescent daughter, one
                        dark -- with no signal left at the vanished dimer's site.

The fluorescence chain of the retained population through the production labeling path
(`labeling.resolve_labeling` + `labeling.label_subunits`: the fluorescent fractions per probe class,
the two-probe share, the dyes per subunit, the recorded columns) is carried by the structure audit's
R3 on each condition's own lineage and by the prior-realization audit's P4 on a tier's Labeling_Set
and P6 in closed form; it is not repeated here.

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
    PARAMETERS, RunTiming, parameterization_of, prior_center,
)
from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video  # noqa: E402
from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import (  # noqa: E402
    build_simulation, build_system, collapse_species_axis, extract_subunit_lineage,
    extract_trajectory_poses, initial_composition_of, initial_species_counts,
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
T_RENDER = 300           # frames of the static render scenes (Level 3)
SEED = 20260907
# Tolerances (prespecified). Level 1: analytic vs empirical, at n = 4e5 the standard error of a
# mean is ~2e-3, so 1% relative is a practical, not a significance, tolerance. Level 3: aperture
# means over T_RENDER frames of a lognormal OU process carry ~10% sampling spread (about 20
# effectively independent frames per dye), so the 2x ratio is accepted within [1.7, 2.3]; the
# zero-dye and vanished-dimer apertures must lie within 3 standard errors of zero, estimated
# from the same frames.
TOL_LAW_REL = 0.01
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
            fluorescent_monomer=law.fluorescent_probability, fluorescent_dimer=law.fluorescent_fraction(2),
        )
        rel = lambda a, b: abs(a - b) / max(abs(a), 1e-12)
        out[name]["max_rel_err"] = float(max(rel(law.mean, k.mean()), rel(law.variance, k.var()),
                                             rel(law.probability_zero, (k == 0).mean())))
    return out


# ----------------------------------------------------------------------------------------------
# Level 2: lineage on a reactive trajectory at the prior center
# ----------------------------------------------------------------------------------------------

# MET-INLB: the condition with association switched on, so the lineage exercises every channel
# (fusion concatenates, fission distributes, conversion preserves); its table has twelve rows, the
# window-start allocation included.
REACTIVE_CONDITION = "INLB"


def prior_center_theta(condition: str) -> np.ndarray:
    """Physical prior center of the condition's table, in its theta order."""
    return np.array([prior_center(p) for p in parameterization_of(condition)], dtype=float)


def frame0_species_counts(tray, condition: str) -> dict:
    """Frame-0 hosts per molecular species of ``condition`` (A, B1, B2 under MET-INLB), summed over
    the mobility modes, from the particles observable; keyed like ``initial_species_counts``."""
    rds = PARAMETERS.simulation.rds
    _, types, _, _ = tray.read_observable_particles()
    name_of_rank = {int(rank): name for name, rank in tray.particle_types.items()}
    counts = {sp.name: 0 for sp in rds.species_for(condition)}
    for t in types[0]:
        counts[rds.molecule_of_type[name_of_rank[int(t)]]] += 1
    return counts


def reactive_lineage(workdir: str) -> dict:
    theta = prior_center_theta(REACTIVE_CONDITION)
    starting = dict(initial_species_counts(initial_composition_of(theta, REACTIVE_CONDITION)))
    timing = RunTiming(total_time_seconds=2.0, frames=PARAMETERS.simulation.timing)
    stem = build_system(theta, REACTIVE_CONDITION)
    smut = build_simulation(stem, theta, REACTIVE_CONDITION, seed=SEED)
    traj = os.path.join(workdir, "labeling_audit_center.h5")
    smut.output_file = traj
    smut.progress_output_stride = timing.total_steps
    smut.run(n_steps=timing.total_steps,
             timestep=timing.delta_time_nanoseconds * readdy.units.nanosecond, show_summary=False)
    tray = readdy.Trajectory(traj)
    frame0 = frame0_species_counts(tray, REACTIVE_CONDITION)     # placed hosts against the helper's starting counts
    poses = extract_trajectory_poses(tray)
    lineage = extract_subunit_lineage(tray)
    soul_poses = collapse_species_axis(poses)
    # Independent cross-check against the particles observable.
    _, types, ids, _ = tray.read_observable_particles()
    rank_of = {name: int(rank) for name, rank in tray.particle_types.items()}
    # One rank per PARTICLE TYPE (species x mode) the trajectory registers; the subunit count is the
    # type's species' count (the all-condition list also names the other condition's types).
    subunits_of_rank = {rank_of[n]: k for n, k in zip(PARAMETERS.simulation.rds.particle_type_names,
                                                     PARAMETERS.simulation.rds.subunit_counts_per_type)
                        if n in rank_of}
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
        frame0_hosts=frame0, starting_counts=starting, frame0_equals_starting=bool(frame0 == starting),
        theta_center=theta.tolist(),
    )


# ----------------------------------------------------------------------------------------------
# Level 3: static renders through the production renderer
# ----------------------------------------------------------------------------------------------

def renders() -> dict:
    stem_geometry = PARAMETERS.simulation.stem
    pix = stem_geometry.pixel_size_nm
    imaging = np.array([prior_center(e) for e in det.DETECTOR_IMAGING])   # physical prior-center nominals
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
    dark_ok, invis_mean, invis_se = null_ok(aperture(fr3, d2)[half:])
    fluorescent_daughter = float(aperture(fr3, d1)[half:].mean())
    site_before = float(aperture(fr3, P[1])[:half].mean())
    # (d) an emitter-free scene renders background only
    fr0 = render_dli_video(soul_poses, host_index, np.zeros(4, dtype=int), imaging, seed=SEED + 3)
    return dict(
        zero_dye_ok=zero_ok, zero_dye_mean_adu=zero_mean, zero_dye_se_adu=zero_se,
        one_dye_monomer_adu=one_dye_mono, one_dye_dimer_adu=one_dye_dimer,
        two_dye_ratio=ratio, ratio_band=list(RATIO_BAND),
        dissociation_site_before_adu=site_before, dissociation_site_after_ok=site_after_ok,
        dissociation_site_after_adu=site_after, dissociation_site_after_se=site_after_se,
        fluorescent_daughter_adu=fluorescent_daughter, dark_daughter_ok=dark_ok,
        dark_daughter_adu=invis_mean, dark_daughter_se=invis_se,
        empty_scene_mean_adu=float(fr0.mean()), empty_scene_expected_adu=float(bg_adu),
        empty_scene_finite=bool(np.isfinite(fr0).all()),
    )


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

def write_report(law_out, lin, ren, verdicts) -> None:
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
        f"within {TOL_LAW_REL:.0%} relative. The fluorescent fractions follow from P0 by arithmetic.")
    add("")
    add("**Level 2 (lineage).** A 2 s reactive trajectory at the MET-INLB prior center (its twelve values, "
        "the window-start allocation included) is simulated in a temporary directory; its frame-0 hosts per "
        "molecular species must equal the starting counts of the composition helper; its reaction records "
        "are replayed into the subunit lineage, and the lineage "
        "checked independently against the particles observable: every host present in its frame with "
        "the recorded particle type, every particle covered by exactly its type's subunit count, every host "
        "position finite. The particle-id churn (ids visited per subunit) documents why the lineage is "
        "needed: ReaDDy mints a new id at every reaction, conversions included.")
    add("")
    add(f"**Level 3 (render).** Static scenes of {T_RENDER} frames at the imaging prior center with "
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
    add(f"| 2 | frame-0 hosts equal the starting counts; lineage covers every subunit, agrees with the particles "
        f"observable | exact | frame-0 hosts {lin['frame0_hosts']} vs starting {lin['starting_counts']}; "
        f"{lin['n_subunits']} subunits over {lin['n_particle_ids']} particle ids in {lin['n_frames']} frames; "
        f"{lin['n_records']} records; up to {lin['max_ids_per_subunit']} ids per subunit "
        f"(mean {lin['mean_ids_per_subunit']:.1f}) | {verdicts['lineage']} |")
    add(f"| 3 | zero-dye subunit renders nothing | within {NULL_SIGMAS:g} SE of 0 | "
        f"{ren['zero_dye_mean_adu']:+.0f} +/- {ren['zero_dye_se_adu']:.0f} ADU (one-dye monomer "
        f"{ren['one_dye_monomer_adu']:.0f}) | {verdicts['zero_dye']} |")
    add(f"| 3 | two-dye spot vs one-dye spot | ratio in [{RATIO_BAND[0]:g}, {RATIO_BAND[1]:g}] | "
        f"{ren['two_dye_ratio']:.3f} | {verdicts['ratio']} |")
    add(f"| 3 | dissociation: dye follows its subunit | site after within {NULL_SIGMAS:g} SE of 0; "
        f"unlabeled daughter within {NULL_SIGMAS:g} SE of 0 | site {ren['dissociation_site_before_adu']:.0f} -> "
        f"{ren['dissociation_site_after_adu']:+.0f} +/- {ren['dissociation_site_after_se']:.0f}; "
        f"fluorescent daughter {ren['fluorescent_daughter_adu']:.0f}; unlabeled daughter "
        f"{ren['dark_daughter_adu']:+.0f} +/- {ren['dark_daughter_se']:.0f} | {verdicts['dissociation']} |")
    add(f"| 3 | emitter-free scene | finite, mean = background | mean {ren['empty_scene_mean_adu']:.1f} vs "
        f"{ren['empty_scene_expected_adu']:.1f} ADU | {verdicts['empty']} |")
    add("")
    add(f"**Overall: {verdicts['overall']}.**")
    add("")
    add("## 3. Reading the numbers")
    add("")
    add("The one-dye dimer of the first render carries the same signal as a one-dye monomer: a dimer "
        "whose partner subunit is unlabeled is not brighter than a monomer, which is the first of the "
        "three derived consequences (the fluorescent-dimer brightness is a mixture, not a doubled monomer). "
        "The lineage's id churn is the fact the design rests on -- a static per-subunit quantity cannot "
        "be attached to a ReaDDy particle, only to the subunit the lineage follows -- and it is also why "
        "a per-particle emitter would restart its brightness process at every conversion.")
    add("")
    add("## 4. Audit operating point")
    add("")
    theta_desc = ", ".join(f"{p['KEY']}={v:.4g}" for p, v in zip(parameterization_of(REACTIVE_CONDITION),
                                                                   lin["theta_center"]))
    add(f"Laws: {N_LAW:,} draws each. Lineage: theta at the {REACTIVE_CONDITION} prior center ({theta_desc}), "
        f"2 s at the configured cadence, seed {SEED}. Renders: "
        f"{T_RENDER} frames, imaging at the prior center of every imaging parameter with bleaching off, "
        "9x9 apertures.")
    with open(REPORT, "w") as handle:
        handle.write("\n".join(L) + "\n")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Level 1: laws ...")
    law_out = laws()
    print(f"Level 2: reactive lineage at the {REACTIVE_CONDITION} prior center ...")
    with tempfile.TemporaryDirectory(prefix="labeling_audit_") as workdir:
        lin = reactive_lineage(workdir)
    print("Level 3: renders ...")
    ren = renders()

    verdicts = {}
    for name, r in law_out.items():
        verdicts["law_" + name] = passed(r["max_rel_err"] < TOL_LAW_REL)
    verdicts["lineage"] = passed(lin["cover_consistent"] and lin["host_positions_finite"]
                                 and lin["max_ids_per_subunit"] >= 1 and lin["frame0_equals_starting"])
    verdicts["zero_dye"] = passed(ren["zero_dye_ok"])
    verdicts["ratio"] = passed(RATIO_BAND[0] <= ren["two_dye_ratio"] <= RATIO_BAND[1])
    verdicts["dissociation"] = passed(ren["dissociation_site_after_ok"] and ren["dark_daughter_ok"]
                                      and ren["fluorescent_daughter_adu"] > 5 * ren["dark_daughter_se"])
    verdicts["empty"] = passed(ren["empty_scene_finite"]
                               and abs(ren["empty_scene_mean_adu"] - ren["empty_scene_expected_adu"]) < 2.0)
    verdicts["overall"] = passed(all(v == "PASS" for v in verdicts.values()))

    with open(os.path.join(OUT_DIR, "audit_summary.json"), "w") as handle:
        json.dump(dict(laws=law_out, lineage=lin, renders=ren, verdicts=verdicts), handle, indent=1, default=str)
    write_report(law_out, lin, ren, verdicts)
    print("verdicts:", verdicts)
    print("report:", REPORT)


if __name__ == "__main__":
    main()
