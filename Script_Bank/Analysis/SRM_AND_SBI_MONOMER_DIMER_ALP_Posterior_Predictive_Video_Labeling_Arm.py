#!/usr/bin/env python
"""Re-render a persisted posterior-predictive clip under another labeling law: a controlled labeling arm.

Part of the posterior-predictive video check (``SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video.md``).
For each recording given, the persisted render (``--source-label``, e.g. ``REF``) is reused whole: its
trajectory file, its imaging vector, its seed and the very subunits it labeled. Only the dye counts of
those labeled subunits are drawn again, from the law given (``--labeling-law``, a registry key such as
``FAB_BINOMIAL`` or ``family:mean[:shape]``) conditioned on at least one dye, so the visible subunits and
the motion are identical in both arms and the arms differ in the dye multiplicity alone. The clip is
rendered through the production renderer and written beside the source as
``<source stem>_<ARM>_Synthetic_Video.npz`` with the comparison figure ``..._Comparison.png``; the source's
provenance is carried over with the labeling fields replaced (``labeling_law``, ``dye_counts``,
``labeling_row``, ``labeling_record_json``, ``synth_label``). Nothing is overwritten: an existing arm file
stops the run before any work. No trajectory is simulated.

This is a test tool. The arm's law is not adopted anywhere by running it; the production labeling law and
the training data are untouched. The occupancy recorded is the source render's (the labeled subunits are
kept), not the occupancy the arm's law would derive for production.

Usage::

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Labeling_Arm.py \
        --total-time-seconds 2 --kind MET-FAB --cell 0 5 16 --source-label REF \
        --labeling-law FAB_BINOMIAL --arm-label BINOMIAL4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from srm_and_sbi_monomer_dimer_alp import __version__ as _PACKAGE_VERSION
from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv
from srm_and_sbi_monomer_dimer_alp.experiment_support import KIND_OF_CONDITION
from srm_and_sbi_monomer_dimer_alp.labeling import (LABELING_SET_COLUMNS, LabelingLaw, labeling_rng,
                                                    labeling_summary, resolve_labeling_law)
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video
from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import (collapse_species_axis, extract_subunit_lineage,
                                                                  extract_trajectory_poses, monomer_ranks)
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow


def positive_conditioned_counts(law: LabelingLaw, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` dye counts from ``law`` conditioned on at least one dye (rejection of zeros)."""
    k = law.draw(n, rng)
    while (k == 0).any():
        k[k == 0] = law.draw(int((k == 0).sum()), rng)
    return k.astype(np.int64)


def arm_dye_counts(source_counts: np.ndarray, law: LabelingLaw, rng: np.random.Generator) -> np.ndarray:
    """The arm's dye counts: the source's labeled subunits keep at least one dye, drawn from ``law``;
    unlabeled subunits stay unlabeled."""
    source_counts = np.asarray(source_counts, dtype=np.int64)
    out = np.zeros_like(source_counts)
    labeled = source_counts > 0
    out[labeled] = positive_conditioned_counts(law, int(labeled.sum()), rng)
    return out


def histogram(counts: np.ndarray) -> dict:
    values, n = np.unique(counts[counts > 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(values, n)}


def arm_fields(src, source_name: str, counts: np.ndarray, row, synth_u16: np.ndarray, law_name: str,
               law: LabelingLaw, arm_label: str) -> dict:
    """The arm clip's fields: the source clip's fields (``src``, an opened npz or a dict saved as one) with
    the labeling replaced and the arm recorded in ``labeling_record_json`` (its ``arm`` entry is what the
    engine's figure reads to say that the source's labeled subunits and trajectory were kept)."""
    source_counts = np.asarray(src["dye_counts"]).astype(np.int64)
    source_record = json.loads(str(np.asarray(src["labeling_record_json"])))
    record = dict(source_record)
    record.update({"law_name": law_name,
                   "law": {"family": law.family, "mean": law.mean, "shape": law.shape,
                           "description": law.describe()},
                   "arm": {"label": arm_label, "source_clip": source_name,
                           "source_law_name": str(np.asarray(src["labeling_law"])),
                           "labeled_subunits_kept_from_source": True,
                           "conditioned_on_at_least_one_dye": True,
                           "source_dye_counts": histogram(source_counts), "arm_dye_counts": histogram(counts)}})
    synth_label = f"{str(np.asarray(src['synth_label']))[:-1]}; labeling arm {law_name})"
    fields = {k: src[k] for k in (src.files if hasattr(src, "files") else list(src))}
    fields.update(experimental=np.asarray(src["experimental"]).astype(np.uint16), synth=synth_u16,
                  labeling_law=law_name, dye_counts=counts, n_dyes=int(counts.sum()), labeling_row=row,
                  labeling_columns=np.array(LABELING_SET_COLUMNS), labeling_record_json=json.dumps(record),
                  synth_label=synth_label, package_version=_PACKAGE_VERSION)
    return fields


def render_arm(source_clip: Path, source_traj: Path, outputs: dict, law_name: str, law: LabelingLaw,
               arm_label: str, cfg, verbose=False) -> dict:
    import readdy

    src = np.load(source_clip, allow_pickle=False)
    seed = int(src["seed"])
    if seed < 0:
        raise SystemExit(f"{source_clip.name} was rendered without a seed; the arm needs the source's seed.")
    source_counts = src["dye_counts"].astype(np.int64)
    counts = arm_dye_counts(source_counts, law, labeling_rng(seed))
    tray = readdy.Trajectory(filename=str(source_traj))
    lineage = extract_subunit_lineage(tray, verbose=verbose)
    if counts.shape[0] != lineage.n_subunits:
        raise SystemExit(f"{source_clip.name}: {counts.shape[0]} dye counts for {lineage.n_subunits} subunits.")
    soul = collapse_species_axis(extract_trajectory_poses(tray, verbose=verbose))
    occupancy_pair = (float(src["occupancy_monomer"]), float(src["occupancy_dimer"]))
    row = labeling_summary(counts, lineage.host_index[0], lineage.host_rank[0], monomer_ranks(tray),
                           occupancy_by_species_values=occupancy_pair)
    del tray
    imaging = np.asarray(src["imaging_physical"], dtype=float)
    t0 = time.time()
    synth = render_dli_video(soul_poses=soul, host_index=lineage.host_index, dye_counts=counts,
                             imaging_physical=imaging, seed=seed, verbose=verbose)
    synth = np.moveaxis(synth, -1, 0)
    synth_u16 = np.clip(np.rint(synth), 0, 65535).astype(np.uint16)
    fields = arm_fields(src, source_clip.name, counts, row, synth_u16, law_name, law, arm_label)
    np.savez_compressed(str(outputs["clip"]), **fields)
    # The figure is drawn from the arm's own fields by the engine (its labeling line and motion follow
    # the record's ``arm`` entry), under the engine's default display window, exactly as a redraw of
    # the arm clip would draw it.
    ppv.draw_comparison_figure(fields, outputs["figure"], cfg)
    return {"seconds": time.time() - t0, "source_counts": histogram(source_counts), "arm_counts": histogram(counts),
            "source_dyes": int(source_counts.sum()), "arm_dyes": int(counts.sum()), "labeled": int((counts > 0).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True,
                    help="model window of the run the source renders belong to (their timing label).")
    ap.add_argument("--kind", default="MET-FAB", choices=tuple(KIND_OF_CONDITION))
    ap.add_argument("--cell", type=int, nargs="+", required=True, help="recording indices.")
    ap.add_argument("--experiment-span-seconds", type=int, default=20, help="the source renders' clip span (s).")
    ap.add_argument("--source-label", required=True, help="run label of the persisted renders to reuse (e.g. REF).")
    ap.add_argument("--labeling-law", required=True,
                    help="the arm's law: a registry key of the recording's condition (e.g. FAB_BINOMIAL) or "
                         "'family:mean[:shape]'.")
    ap.add_argument("--arm-label", required=True, help="token appended to the source stem for the arm's files.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="resolve inputs and outputs, render nothing.")
    args = ap.parse_args(argv)

    cfg = biology_workflow()
    kind_token = KIND_OF_CONDITION[args.kind]
    paths = cfg.paths.with_condition(kind_token)
    map_label = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing).label
    out_dir = (PARAMETERS.machine.data_bank_root / paths.posit_subdir
               / f"{paths.project_alias}_{map_label}_Posterior_Predictive_Video")
    law_name, law = resolve_labeling_law(kind_token, args.labeling_law)
    arm_token = "".join(c if c.isalnum() else "_" for c in str(args.arm_label)).strip("_")   # as _build_stem does
    if not arm_token:
        raise SystemExit("--arm-label must contain at least one alphanumeric character.")
    span = f"{args.experiment_span_seconds}S"

    jobs = []
    for cell in args.cell:
        source_stem = ppv._build_stem(paths.project_alias, map_label, args.kind, cell, None, "cell-sgm", span,
                                      run_label=args.source_label, declared_rds=True, map_block="rds")
        source = ppv.render_output_paths(out_dir, source_stem)
        for key in ("clip", "trajectory"):
            if not source[key].exists():
                raise SystemExit(f"source render missing for cell {cell}: {source[key]}")
        outputs = ppv.render_output_paths(out_dir, f"{source_stem}_{arm_token}")
        del outputs["trajectory"]                                   # the arm simulates nothing
        ppv.refuse_existing_outputs(outputs, hint="Another labeling arm of this render needs its own --arm-label.")
        jobs.append((cell, source, outputs))

    print(f"labeling arm {arm_token}: law {law_name} = {law.describe()} (P(0) = {law.probability_zero:.3f}; the "
          f"source's labeled subunits are kept and every one carries at least one dye)")
    for cell, source, outputs in jobs:
        if args.dry_run:
            print(f"[DRY RUN] cell {cell}: {source['clip'].name} + {source['trajectory'].name} -> {outputs['clip'].name}")
            continue
        info = render_arm(source["clip"], source["trajectory"], outputs, law_name, law, arm_token, cfg,
                          verbose=args.verbose)
        print(f"cell {cell}: {info['labeled']} labeled subunits; dyes {info['source_dyes']} (source "
              f"{info['source_counts']}) -> {info['arm_dyes']} (arm {info['arm_counts']}); {info['seconds']:.0f} s\n"
              f"    {outputs['clip']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
