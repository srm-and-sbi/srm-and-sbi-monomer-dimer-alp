#!/usr/bin/env python
"""Brightness arm of a persisted posterior-predictive render: re-render the source clip's trajectory with its
very dye assignments, seed and imaging vector, ONE imaging coordinate overridden (``--set-imaging KEY=VALUE``,
e.g. mu_pc=150). Follows the trajectory-reuse pattern of ..._Posterior_Predictive_Video_Labeling_Arm.py; the dye
counts are kept exactly (no redraw). Writes <source stem>_<ARM>_{Synthetic_Video.npz,Comparison.png} beside the
source; never overwrites; simulates no trajectory. A diagnostic, not a selection: the SEL artifacts are untouched.
Usage: MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python brightness_arm.py --total-time-seconds 2 --kind MET-FAB
       --cell 26 --source-label SEL_N30000_SINGLE --set-imaging mu_pc=150 --arm-label MUPC150 [--dry-run]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from srm_and_sbi_monomer_dimer_alp import __version__ as _PACKAGE_VERSION
from srm_and_sbi_monomer_dimer_alp import posterior_predictive_video_runner as ppv
from srm_and_sbi_monomer_dimer_alp.experiment_support import KIND_OF_CONDITION
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_monomer_dimer_alp.simulation_dli_support import render_dli_video
from srm_and_sbi_monomer_dimer_alp.simulation_rds_support import (collapse_species_axis, extract_subunit_lineage,
                                                                  extract_trajectory_poses)
from srm_and_sbi_monomer_dimer_alp.workflow import biology_workflow
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # the Analysis folder: the labeling arm's helpers
from SRM_AND_SBI_MONOMER_DIMER_ALP_Posterior_Predictive_Video_Labeling_Arm import assert_source_keys

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--total-time-seconds", type=float, required=True)
    ap.add_argument("--kind", default="MET-FAB", choices=tuple(KIND_OF_CONDITION))
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--experiment-span-seconds", type=int, default=20)
    ap.add_argument("--source-label", required=True)
    ap.add_argument("--set-imaging", action="append", default=[], metavar="KEY=VALUE", required=True)
    ap.add_argument("--arm-label", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = biology_workflow(); kind_token = KIND_OF_CONDITION[args.kind]; paths = cfg.paths.with_condition(kind_token)
    map_label = RunTiming(total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing).label
    out_dir = (PARAMETERS.machine.data_bank_root / paths.posit_subdir / f"{paths.project_alias}_{map_label}_Posterior_Predictive_Video")
    overrides = ppv._parse_kv(args.set_imaging, "--set-imaging")
    arm_token = "".join(c if c.isalnum() else "_" for c in str(args.arm_label)).strip("_")
    span = f"{args.experiment_span_seconds}S"
    source_stem = ppv._build_stem(paths.project_alias, map_label, args.kind, args.cell, None, "cell-sgm", span,
                                  run_label=args.source_label, declared_rds=True, map_block="rds")
    source = ppv.render_output_paths(out_dir, source_stem)
    for key in ("clip", "trajectory"):
        if not source[key].exists():
            raise SystemExit(f"source render missing: {source[key]}")
    assert_source_keys(source["clip"], kind_token)
    outputs = ppv.render_output_paths(out_dir, f"{source_stem}_{arm_token}"); del outputs["trajectory"]
    ppv.refuse_existing_outputs(outputs, hint="Another arm of this render needs its own --arm-label.")
    src = np.load(source["clip"], allow_pickle=False)
    keys = [str(k) for k in src["imaging_keys"]]
    imaging = np.asarray(src["imaging_physical"], dtype=float).copy()
    for k, v in overrides.items():
        if k not in keys:
            raise SystemExit(f"--set-imaging key {k!r} is not one of {keys}")
        imaging[keys.index(k)] = float(v)
    print(f"brightness arm {arm_token}: source {source['clip'].name}\n    trajectory {source['trajectory'].name}\n"
          f"    imaging source {dict(zip(keys, np.round(src['imaging_physical'], 4)))}\n    override {overrides} -> "
          f"{dict(zip(keys, np.round(imaging, 4)))}\n    dye counts kept: {int((src['dye_counts'] > 0).sum())} labeled subunits, "
          f"{int(src['dye_counts'].sum())} dyes; seed {int(src['seed'])}\n    -> {outputs['clip'].name}")
    if args.dry_run:
        print("[DRY RUN] nothing rendered."); return 0
    import readdy
    seed = int(src["seed"]); counts = src["dye_counts"].astype(np.int64)
    tray = readdy.Trajectory(filename=str(source["trajectory"]))
    lineage = extract_subunit_lineage(tray, verbose=args.verbose)
    if counts.shape[0] != lineage.n_subunits:
        raise SystemExit(f"{counts.shape[0]} dye counts for {lineage.n_subunits} subunits.")
    soul = collapse_species_axis(extract_trajectory_poses(tray, verbose=args.verbose)); del tray
    t0 = time.time()
    synth = render_dli_video(soul_poses=soul, host_index=lineage.host_index, dye_counts=counts, imaging_physical=imaging,
                             seed=seed, verbose=args.verbose)
    synth_u16 = np.clip(np.rint(np.moveaxis(synth, -1, 0)), 0, 65535).astype(np.uint16)
    fields = {k: src[k] for k in src.files}
    rec = json.loads(str(src["imaging_record_json"])); rec["set_imaging_overrides"] = overrides
    rec["brightness_arm"] = {"label": arm_token, "source_clip": source["clip"].name, "source_imaging_physical": dict(zip(keys, map(float, src["imaging_physical"]))),
                             "trajectory_and_dye_counts_kept_from_source": True}
    lab = json.loads(str(src["labeling_record_json"]))
    lab["arm"] = {"label": arm_token, "source_clip": source["clip"].name, "kind": "brightness arm (imaging override; dye counts kept)",
                  "labeled_subunits_kept_from_source": True, "dye_counts_kept_from_source": True}
    fields.update(synth=synth_u16, imaging_physical=imaging, imaging_record_json=json.dumps(rec), labeling_record_json=json.dumps(lab),
                  synth_label=f"{str(src['synth_label'])[:-1]}; brightness arm {', '.join(f'{k}={v:g}' for k, v in overrides.items())})",
                  package_version=_PACKAGE_VERSION)
    np.savez_compressed(str(outputs["clip"]), **fields)
    ppv.draw_comparison_figure(fields, outputs["figure"], cfg)
    print(f"rendered in {time.time() - t0:.0f} s\n    {outputs['clip']}\n    {outputs['figure']}"); return 0

if __name__ == "__main__":
    sys.exit(main())
