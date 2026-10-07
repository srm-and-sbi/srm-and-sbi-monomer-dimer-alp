#!/usr/bin/env python
"""Readout of Inference-stage training logs: the best TEST loss per run, per chain and per configuration.

Part of the encoder screening (DETECTOR_WORKFLOW.md, the section on the encoder screening; companion note
``SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Training_Log_Readout.md``). The Inference stage prints one line per
epoch with the TRAIN and TEST losses, the learning rate, the epoch time and the peak memory, a ``[new best]``
line whenever the TEST loss improves, a ``WARM RESTART`` line whenever the schedule restarts, and a header
with the run's settings (preset, tag, global batch, per-rank batch, ranks, nodes, resurrect). This tool
reads any set of those logs and writes three tables from them, so that a screening is read from its
evidence and not from memory:

* **runs** -- one row per log: settings, epochs run, best TEST loss and the global epoch that reached it,
  last TEST loss, warm restarts, wall time, peak memory, completion and tracebacks;
* **chains** -- one row per artifact tag: the legs of a training (a first leg and its ``--resurrect``
  continuations) read together, the best TEST loss over all legs and the epochs covered;
* **configurations** -- one row per architecture preset and global batch: the number of chains (N), the
  best chain, every chain's best beside it, and their spread.

The TEST loss is the mean negative log-probability of the parameters on the TEST videos, as logged; lower
is better; "best" is the minimum over the epoch lines of the log. A configuration is read by its best
chain with N stated beside it; the spread between chains is context, not a significance test. The tool
computes and lists; the reading and the selection are made in the workflow document, under the screening
and selection guidance written there. Smoke runs (tags beginning with ``SMOKE``) are listed among the runs
and excluded from the chains and configurations unless ``--include-smokes`` is given; a log without epoch
lines is listed with zero epochs and excluded in the same way.

Outputs, under ``--out-dir`` (refused when ``readout.md`` already exists there, unless ``--overwrite``):
``readout_runs.csv``, ``readout_chains.csv``, ``readout_configurations.csv``, ``readout.json`` and
``readout.md`` (the three tables with the inputs and the reading notes).

Usage::

    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \\
        Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Training_Log_Readout.py \\
        --logs <directory or log files> --out-dir <directory> [--include-smokes] [--overwrite] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:  # the package version labels the readout; the tool itself reads only text
    from srm_and_sbi_monomer_dimer_alp import __version__ as _PACKAGE_VERSION
except Exception:  # pragma: no cover - the tool also runs outside the package environment
    _PACKAGE_VERSION = "unknown"

FILENAME_RE = re.compile(
    r"^(?P<alias>.*_DETECTOR_(?P<condition>[A-Z]+)_(?P<timing>\d+S_\d+FPS))(?:_(?P<tag>[A-Z0-9]+))?"
    r"_Inference_(?P<job>\d+)\.out$")
"""The HPC log name: ``<alias>[_<TAG>]_Inference_<jobid>.out`` (the dispatcher's naming)."""
HEADER_RE = re.compile(r"=== Inference \| (?P<fields>.*)$")
FIELD_RE = re.compile(r"(\w+)=(\S+)")
EPOCH_RE = re.compile(r"Epoch (?P<leg>\d+)\|(?P<planned>\d+)(?: \(global (?P<glob>\d+)\))?"
                      r"\s+train=(?P<train>-?[\d.]+)\s+test=(?P<test>-?[\d.]+)")
LR_RE = re.compile(r"lr=(?P<lr>[\d.]+e[+-]?\d+|[\d.]+)")
SECONDS_RE = re.compile(r"epoch=(?P<s>[\d.]+)s")
ELAPSED_RE = re.compile(r"elapsed=(?P<e>\d+:\d+:\d+)")
MEM_RE = re.compile(r"peak_mem=(?P<a>[\d.]+)/(?P<r>[\d.]+)GiB")
NEW_BEST_RE = re.compile(r"\[new best\] committed live artifacts at epoch (?P<epoch>\d+) \(test loss (?P<loss>-?[\d.]+)\)")
WARM_RE = re.compile(r"WARM RESTART:")
TOTAL_RE = re.compile(r"Total elapsed: (?P<s>[\d.]+)s")
COMPLETE_MARK = "=== Inference complete ==="
TRACEBACK_MARK = "Traceback (most recent call last)"
FALLBACK_RES = {  # the per-rank settings block, for logs whose first line carries no header
    "preset": re.compile(r"--network-preset\s*:\s*(\S+)"),
    "tag": re.compile(r"--artifact-tag\s*:\s*(\S+)"),
    "global_batch": re.compile(r"--global-batch\s*:\s*(\S+)"),
    "resurrect": re.compile(r"--resurrect\s*:\s*(True|False)"),
    "batch": re.compile(r"--batch-size\s*:\s*(\d+)"),
    "epochs": re.compile(r"--epochs\s*:\s*(\d+)"),
}
SMOKE_PREFIX = "SMOKE"


@dataclass
class EpochRecord:
    leg_epoch: int
    leg_epochs: int
    global_epoch: int
    train: float
    test: float
    lr: float | None
    seconds: float | None
    elapsed: str | None
    mem_alloc_gib: float | None
    mem_reserved_gib: float | None


@dataclass
class RunRecord:
    path: str
    job_id: int
    alias: str
    condition: str
    timing: str
    tag: str
    preset: str
    global_batch: int | None
    global_batch_source: str
    batch: int | None
    world_size: int | None
    nodes: int | None
    gpus_per_node: int | None
    resurrect: bool
    epochs_planned: int | None
    train_tasks: int | None
    test_tasks: int | None
    epochs: list = field(default_factory=list)
    new_best: list = field(default_factory=list)
    warm_restarts: int = 0
    wall_seconds: float | None = None
    complete: bool = False
    tracebacks: int = 0

    # --- derived -----------------------------------------------------------------------------------
    @property
    def is_smoke(self) -> bool:
        return self.tag.startswith(SMOKE_PREFIX)

    @property
    def epochs_run(self) -> int:
        return len(self.epochs)

    @property
    def first_global_epoch(self) -> int | None:
        return self.epochs[0].global_epoch if self.epochs else None

    @property
    def last_global_epoch(self) -> int | None:
        return self.epochs[-1].global_epoch if self.epochs else None

    @property
    def best(self) -> EpochRecord | None:
        return min(self.epochs, key=lambda e: e.test) if self.epochs else None

    @property
    def best_test(self) -> float | None:
        return self.best.test if self.best else None

    @property
    def best_global_epoch(self) -> int | None:
        return self.best.global_epoch if self.best else None

    @property
    def last_test(self) -> float | None:
        return self.epochs[-1].test if self.epochs else None

    @property
    def last_train(self) -> float | None:
        return self.epochs[-1].train if self.epochs else None

    @property
    def peak_mem_alloc_gib(self) -> float | None:
        vals = [e.mem_alloc_gib for e in self.epochs if e.mem_alloc_gib is not None]
        return max(vals) if vals else None

    @property
    def lr_floor(self) -> float | None:
        vals = [e.lr for e in self.epochs if e.lr is not None]
        return min(vals) if vals else None

    def row(self) -> dict:
        return {
            "job_id": self.job_id, "tag": self.tag, "condition": self.condition, "timing": self.timing,
            "preset": self.preset, "global_batch": self.global_batch, "global_batch_source": self.global_batch_source,
            "batch": self.batch, "world_size": self.world_size, "nodes": self.nodes,
            "gpus_per_node": self.gpus_per_node, "resurrect": int(self.resurrect), "smoke": int(self.is_smoke),
            "epochs_planned": self.epochs_planned, "epochs_run": self.epochs_run,
            "first_global_epoch": self.first_global_epoch, "last_global_epoch": self.last_global_epoch,
            "best_test": self.best_test, "best_global_epoch": self.best_global_epoch,
            "last_test": self.last_test, "last_train": self.last_train, "new_best_events": len(self.new_best),
            "warm_restarts": self.warm_restarts, "lr_floor": self.lr_floor,
            "wall_seconds": self.wall_seconds, "peak_mem_alloc_gib": self.peak_mem_alloc_gib,
            "complete": int(self.complete), "tracebacks": self.tracebacks, "file": Path(self.path).name,
        }


# --- parsing -------------------------------------------------------------------------------------------

def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_header(line: str) -> dict:
    match = HEADER_RE.search(line)
    if not match:
        return {}
    return dict(FIELD_RE.findall(match.group("fields")))


def parse_log(path: Path) -> RunRecord:
    """Read one Inference log into a RunRecord. Rank-concatenated lines are searched, not matched whole."""
    name_match = FILENAME_RE.match(path.name)
    if not name_match:
        raise ValueError(f"not an Inference log name: {path.name}")
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    header = _parse_header(lines[0]) if lines else {}
    fallback = {}
    if not header:
        for key, regex in FALLBACK_RES.items():
            found = regex.search(text)
            if found:
                fallback[key] = found.group(1)

    def setting(key, default=None):
        return header.get(key, fallback.get(key, default))

    tag = name_match.group("tag") or setting("tag", "") or ""
    if tag == "None":
        tag = ""
    batch = _to_int(setting("batch"))
    world = _to_int(setting("world_size"))
    gb_raw = setting("global_batch")
    if gb_raw is not None and str(gb_raw).lower() != "none" and _to_int(gb_raw) is not None:
        global_batch, gb_source = _to_int(gb_raw), "logged"
    elif batch is not None and world is not None:
        global_batch, gb_source = batch * world, "derived (batch x ranks, no accumulation)"
    else:
        global_batch, gb_source = None, "unknown"
    resurrect_raw = str(setting("resurrect", "0"))
    run = RunRecord(
        path=str(path), job_id=int(name_match.group("job")), alias=name_match.group("alias"),
        condition=name_match.group("condition"), timing=name_match.group("timing"), tag=tag,
        preset=str(setting("preset", "baseline")), global_batch=global_batch, global_batch_source=gb_source,
        batch=batch, world_size=world, nodes=_to_int(setting("nodes")),
        gpus_per_node=_to_int(setting("gpus_per_node")),
        resurrect=resurrect_raw in ("1", "True", "true"),
        epochs_planned=_to_int(setting("epochs")), train_tasks=_to_int(setting("train_tasks")),
        test_tasks=_to_int(setting("test_tasks")),
    )
    seen_global = set()
    total_max = None
    for line in lines:
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            leg_epoch = int(epoch_match.group("leg"))
            glob = int(epoch_match.group("glob")) if epoch_match.group("glob") else leg_epoch
            if glob in seen_global:  # a line repeated by another rank
                continue
            seen_global.add(glob)
            lr = LR_RE.search(line)
            sec = SECONDS_RE.search(line)
            ela = ELAPSED_RE.search(line)
            mem = MEM_RE.search(line)
            run.epochs.append(EpochRecord(
                leg_epoch=leg_epoch, leg_epochs=int(epoch_match.group("planned")), global_epoch=glob,
                train=float(epoch_match.group("train")), test=float(epoch_match.group("test")),
                lr=float(lr.group("lr")) if lr else None, seconds=float(sec.group("s")) if sec else None,
                elapsed=ela.group("e") if ela else None,
                mem_alloc_gib=float(mem.group("a")) if mem else None,
                mem_reserved_gib=float(mem.group("r")) if mem else None))
            continue
        best_match = NEW_BEST_RE.search(line)
        if best_match:
            run.new_best.append((int(best_match.group("epoch")), float(best_match.group("loss"))))
            continue
        if WARM_RE.search(line):
            run.warm_restarts += 1
            continue
        total_match = TOTAL_RE.search(line)
        if total_match:
            value = float(total_match.group("s"))
            total_max = value if total_max is None else max(total_max, value)
            continue
        if COMPLETE_MARK in line:
            run.complete = True
        if TRACEBACK_MARK in line:
            run.tracebacks += 1
    run.epochs.sort(key=lambda e: e.global_epoch)
    run.wall_seconds = total_max
    return run


def collect_logs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir() if q.is_file() and FILENAME_RE.match(q.name)))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(item)
    return paths


# --- grouping ------------------------------------------------------------------------------------------

def build_chains(runs: list[RunRecord]) -> list[dict]:
    """One chain per (alias, tag): the legs ordered by their first global epoch, then by job id."""
    groups: dict[tuple, list[RunRecord]] = {}
    for run in runs:
        groups.setdefault((run.alias, run.tag), []).append(run)
    chains = []
    for (alias, tag), legs in groups.items():
        legs = sorted(legs, key=lambda r: (r.first_global_epoch or 0, r.job_id))
        with_epochs = [r for r in legs if r.epochs]
        best_leg = min(with_epochs, key=lambda r: r.best_test) if with_epochs else None
        global_batches = [r.global_batch for r in legs if r.global_batch is not None]
        presets = {r.preset for r in legs}
        chains.append({
            "tag": tag, "alias": alias, "condition": legs[0].condition, "timing": legs[0].timing,
            "preset": legs[0].preset, "global_batch": global_batches[0] if global_batches else None,
            "settings_consistent": int(len(presets) == 1 and len(set(global_batches)) <= 1),
            "legs": len(legs), "job_ids": [r.job_id for r in legs],
            "epochs_run": sum(r.epochs_run for r in legs),
            "last_global_epoch": max((r.last_global_epoch or 0) for r in legs),
            "best_test": best_leg.best_test if best_leg else None,
            "best_global_epoch": best_leg.best_global_epoch if best_leg else None,
            "best_job_id": best_leg.job_id if best_leg else None,
            "leg_bests": [r.best_test for r in legs],
            "complete": int(all(r.complete for r in legs)),
            "tracebacks": sum(r.tracebacks for r in legs),
            "warm_restarts": sum(r.warm_restarts for r in legs),
            "wall_seconds": sum(r.wall_seconds or 0.0 for r in legs),
        })
    chains.sort(key=lambda c: (c["best_test"] is None, c["best_test"] if c["best_test"] is not None else 0.0))
    return chains


def build_configurations(chains: list[dict]) -> list[dict]:
    """One row per (condition, timing, preset, global batch): N chains, the best, every chain's best, spread."""
    groups: dict[tuple, list[dict]] = {}
    for chain in chains:
        if chain["best_test"] is None:
            continue
        groups.setdefault((chain["condition"], chain["timing"], chain["preset"], chain["global_batch"]), []).append(chain)
    rows = []
    for (condition, timing, preset, gb), members in groups.items():
        members = sorted(members, key=lambda c: c["best_test"])
        bests = [c["best_test"] for c in members]
        rows.append({
            "condition": condition, "timing": timing, "preset": preset, "global_batch": gb,
            "chains": len(members), "best_test": bests[0], "best_tag": members[0]["tag"],
            "best_global_epoch": members[0]["best_global_epoch"],
            "chain_bests": bests, "chain_tags": [c["tag"] for c in members],
            "spread": (max(bests) - min(bests)) if len(bests) > 1 else None,
            "epochs_per_chain": [c["last_global_epoch"] for c in members],
        })
    rows.sort(key=lambda r: r["best_test"])
    return rows


# --- output --------------------------------------------------------------------------------------------

def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.5f}" if abs(value) < 1000 else f"{value:.0f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _markdown_table(columns: list[tuple[str, str]], rows: list[dict]) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def write_outputs(out_dir: Path, runs: list[RunRecord], chains: list[dict], configurations: list[dict],
                  inputs: list[Path], excluded: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [r.row() for r in sorted(runs, key=lambda r: r.job_id)]
    with (out_dir / "readout_runs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(run_rows[0].keys()) if run_rows else ["job_id"])
        writer.writeheader()
        writer.writerows(run_rows)
    chain_keys = ["tag", "alias", "condition", "timing", "preset", "global_batch", "settings_consistent", "legs",
                  "job_ids", "epochs_run", "last_global_epoch", "best_test", "best_global_epoch", "best_job_id",
                  "leg_bests", "complete", "tracebacks", "warm_restarts", "wall_seconds"]
    with (out_dir / "readout_chains.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=chain_keys)
        writer.writeheader()
        for chain in chains:
            writer.writerow({k: (" ".join(map(str, v)) if isinstance(v, list) else v)
                             for k, v in chain.items() if k in chain_keys})
    conf_keys = ["condition", "timing", "preset", "global_batch", "chains", "best_test", "best_tag",
                 "best_global_epoch", "chain_bests", "chain_tags", "spread", "epochs_per_chain"]
    with (out_dir / "readout_configurations.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=conf_keys)
        writer.writeheader()
        for row in configurations:
            writer.writerow({k: (" ".join(map(str, v)) if isinstance(v, list) else v)
                             for k, v in row.items() if k in conf_keys})
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload = {
        "generated": stamp, "package_version": _PACKAGE_VERSION, "inputs": [p.name for p in inputs],
        "excluded_from_configurations": excluded,
        "runs": run_rows, "chains": chains, "configurations": configurations,
        "epochs": {str(r.job_id): [asdict(e) for e in r.epochs] for r in runs},
    }
    (out_dir / "readout.json").write_text(json.dumps(payload, indent=1))
    md = [
        "# Training-log readout",
        "",
        f"Generated {stamp} by `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Training_Log_Readout.py` "
        f"(package {_PACKAGE_VERSION}) from {len(inputs)} Inference log(s). The TEST loss is the mean negative "
        "log-probability of the parameters on the TEST videos, as logged; lower is better; the best is the minimum "
        "over the epoch lines of a log. A chain is the legs of one artifact tag read together. A configuration is "
        "one architecture preset at one global batch, under the data, priors and protocol of its logs; it is read "
        "by its best chain with N stated beside it, and the spread between chains is context, not a test. "
        "The reading and the selection are made in the workflow document.",
        "",
        "## Configurations (best chain first)",
        "",
        _markdown_table([("preset", "preset"), ("global_batch", "global batch"), ("chains", "N"),
                         ("best_test", "best TEST"), ("best_tag", "best chain"),
                         ("best_global_epoch", "at global epoch"), ("chain_bests", "chain bests"),
                         ("spread", "spread"), ("epochs_per_chain", "epochs per chain")], configurations),
        "",
        "## Chains (one per tag, best first)",
        "",
        _markdown_table([("tag", "tag"), ("preset", "preset"), ("global_batch", "global batch"), ("legs", "legs"),
                         ("job_ids", "jobs"), ("last_global_epoch", "epochs"), ("best_test", "best TEST"),
                         ("best_global_epoch", "at global epoch"), ("leg_bests", "leg bests"),
                         ("warm_restarts", "warm restarts"), ("complete", "complete"),
                         ("tracebacks", "tracebacks"), ("settings_consistent", "settings consistent")], chains),
        "",
        "## Runs (one per log)",
        "",
        _markdown_table([("job_id", "job"), ("tag", "tag"), ("preset", "preset"), ("global_batch", "global batch"),
                         ("batch", "batch"), ("world_size", "ranks"), ("nodes", "nodes"), ("resurrect", "resurrect"),
                         ("epochs_run", "epochs"), ("first_global_epoch", "first"), ("last_global_epoch", "last"),
                         ("best_test", "best TEST"), ("best_global_epoch", "at"), ("last_test", "last TEST"),
                         ("warm_restarts", "warm restarts"), ("wall_seconds", "wall s"),
                         ("peak_mem_alloc_gib", "peak GiB"), ("complete", "complete"), ("tracebacks", "tracebacks"),
                         ("smoke", "smoke")], run_rows),
        "",
        "## Excluded from the chains and configurations",
        "",
        ("\n".join(f"- {item}" for item in excluded) if excluded else "- none"),
        "",
        "## Inputs",
        "",
        "\n".join(f"- `{p.name}`" for p in inputs),
        "",
    ]
    (out_dir / "readout.md").write_text("\n".join(md))


# --- entry point ---------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", nargs="+", required=True, help="Inference log files, or directories holding them.")
    ap.add_argument("--out-dir", required=True, help="where the tables are written.")
    ap.add_argument("--include-smokes", action="store_true", help="keep SMOKE* tags in the chains and configurations.")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing readout in --out-dir.")
    ap.add_argument("--dry-run", action="store_true", help="list the logs and their settings; write nothing.")
    args = ap.parse_args(argv)

    inputs = collect_logs(args.logs)
    if not inputs:
        print("FATAL: no Inference log found", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    if not args.dry_run and (out_dir / "readout.md").exists() and not args.overwrite:
        print(f"FATAL: {out_dir / 'readout.md'} exists; pass --overwrite to replace it", file=sys.stderr)
        return 2

    runs = [parse_log(p) for p in inputs]
    seen: dict[int, str] = {}
    for run in runs:
        if run.job_id in seen:
            print(f"FATAL: job {run.job_id} appears twice ({seen[run.job_id]} and {Path(run.path).name})",
                  file=sys.stderr)
            return 2
        seen[run.job_id] = Path(run.path).name

    kept, excluded = [], []
    for run in runs:
        if run.is_smoke and not args.include_smokes:
            excluded.append(f"{Path(run.path).name}: smoke tag {run.tag}")
        elif not run.epochs:
            excluded.append(f"{Path(run.path).name}: no epoch line")
        else:
            kept.append(run)
    chains = build_chains(kept)
    configurations = build_configurations(chains)

    print(f"{len(runs)} log(s) read; {len(kept)} in {len(chains)} chain(s) and {len(configurations)} configuration(s); "
          f"{len(excluded)} excluded")
    for run in sorted(runs, key=lambda r: r.job_id):
        best = f"best {run.best_test:.5f} @ global {run.best_global_epoch}" if run.epochs else "no epochs"
        print(f"  {run.job_id}  {run.tag or '(canonical)':40s} preset={run.preset:36s} gb={_fmt(run.global_batch):5s} "
              f"resurrect={int(run.resurrect)}  epochs={run.epochs_run:3d}  {best}"
              f"{'  COMPLETE' if run.complete else ''}{'  TRACEBACK' if run.tracebacks else ''}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    write_outputs(out_dir, runs, chains, configurations, inputs, excluded)
    print(f"written: {out_dir / 'readout.md'} (+ readout_runs.csv, readout_chains.csv, readout_configurations.csv, readout.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
