"""The training-log readout (Script_Bank/Analysis/..._DETECTOR_Training_Log_Readout.py) on synthetic Inference
logs: a leg's settings, epochs, best TEST loss and its epoch, last loss, warm restarts, wall time and
completion are read from the log text; the legs of one tag form a chain whose best spans the legs and whose
global epochs continue; chains of one preset and global batch form a configuration read by its best chain
with N and the spread stated; smoke tags and logs without epoch lines are excluded unless asked for;
duplicated job ids are refused; the dry run writes nothing; an existing readout is not overwritten without
the flag; a log without the one-line header is read from its settings block. Text only; no torch."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _companion():
    path = REPO / "Script_Bank" / "Analysis" / "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_Training_Log_Readout.py"
    spec = importlib.util.spec_from_file_location("detector_training_log_readout", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses resolve the module's annotations through sys.modules
    spec.loader.exec_module(module)
    return module


RO = _companion()
ALIAS = "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS"


def _log_text(tag, preset, gb, resurrect, tests, start_global=1, planned=None, header=True,
              warm_at=(), complete=True, total=1234.5, traceback=False):
    planned = planned or len(tests)
    lines = []
    if header:
        lines.append(f"=== Inference | train_tasks=200 test_tasks=50 epochs={planned} time=2.0s batch=2 "
                     f"global_batch={gb} resurrect={int(resurrect)} preset={preset} tag={tag} nodes=16 "
                     f"gpus_per_node=4 world_size=64 seed=None")
        lines.append("    multi-node: nnodes=16 nproc_per_node=4 rdzv=node:29500")
    else:
        lines.append(f"  --epochs             : {planned}  --epochs             : {planned}")
        lines.append(f"  --global-batch       : {gb}   (None = one optimizer step per batch)")
        lines.append(f"  --resurrect          : {'True' if resurrect else 'False'}")
        lines.append(f"  --network-preset     : {preset}   -> embedding start_channels=16")
        lines.append(f"  --artifact-tag       : {tag}   -> product label 2S_50FPS_{tag}")
        lines.append("  --batch-size         : 2")
    lines.append("Training & network summary:  optimizer : AdamW   scheduler: ReduceLROnPlateau (factor=0.5, patience=1)")
    best = None
    for i, test in enumerate(tests, start=1):
        glob = start_global + i - 1
        suffix = f" (global {glob})" if resurrect else ""
        lines.append(f"Epoch {i}|{planned}{suffix}    train={test + 0.5:.5f}    test={test:.5f}    replay=off    "
                     f"lr={1.0e-3 / (2 ** i):.2e}    epoch=234.3s    elapsed=00:0{min(i, 9)}:54    ETA=03:11:20    "
                     f"peak_mem={6.0 + 0.1 * i:.1f}/9.7GiB")
        lines.append(f"Epoch {i}|{planned}{suffix}    train={test + 0.5:.5f}    test={test:.5f}    (rank repeat)")
        if best is None or test < best:
            best = test
            lines.append(f"  [new best] committed live artifacts at epoch {i} (test loss {test:.5f}); backup deferred to finish")
        if i in warm_at:
            lines.append(f"WARM RESTART: LR held at the floor for 2 epoch(s) without improving; reloaded best (test loss "
                         f"{best:.5f}) and restarted LR at 3.20e-04.")
    if traceback:
        lines.append("Traceback (most recent call last):")
    for k in range(3):
        lines.append(f"Total elapsed: {total - k:.1f}s")
    if complete:
        lines.append("=== Inference complete ===")
    return "\n".join(lines) + "\n"


def _write(folder, tag, job, **kw):
    name = f"{ALIAS}_{tag}_Inference_{job}.out" if tag else f"{ALIAS}_Inference_{job}.out"
    path = Path(folder) / name
    path.write_text(_log_text(tag or "None", **kw))
    return path


def _standard_folder(folder):
    # tag A: two legs at global batch 128, best in leg 2; tag B: one leg, same configuration; tag C: other preset
    _write(folder, "ARMA", 101, preset="capacity256", gb=128, resurrect=False, tests=[-2.0, -5.0, -9.0, -9.3],
           warm_at=(3,), total=10000.0)
    _write(folder, "ARMA", 102, preset="capacity256", gb=128, resurrect=True, tests=[-9.4, -10.1, -10.0, -9.9],
           start_global=5, total=9000.0)
    _write(folder, "ARMB", 103, preset="capacity256", gb=128, resurrect=False, tests=[-3.0, -8.0, -8.5, -8.4])
    _write(folder, "ARMC", 104, preset="capacity256_kernel7_earlyconv_stats", gb=128, resurrect=False,
           tests=[-1.0, -11.0, -12.0, -11.5])
    _write(folder, "SMOKEARMA", 105, preset="capacity256", gb=128, resurrect=False, tests=[2.0, 1.0])
    _write(folder, "ARMD", 106, preset="capacity256", gb=256, resurrect=False, tests=[], complete=False)


def test_one_leg_is_read_from_its_text():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "ARMA", 101, preset="capacity256", gb=128, resurrect=False,
                      tests=[-2.0, -5.0, -9.0, -9.3], warm_at=(3,), total=10000.0)
        run = RO.parse_log(path)
        assert (run.job_id, run.tag, run.preset, run.global_batch, run.resurrect) == (101, "ARMA", "capacity256", 128, False)
        assert run.global_batch_source == "logged"
        assert (run.batch, run.world_size, run.nodes, run.gpus_per_node) == (2, 64, 16, 4)
        assert run.epochs_run == 4, "rank-repeated epoch lines are counted once"
        assert run.best_test == -9.3 and run.best_global_epoch == 4 and run.last_test == -9.3
        assert [e.global_epoch for e in run.epochs] == [1, 2, 3, 4]
        assert run.new_best == [(1, -2.0), (2, -5.0), (3, -9.0), (4, -9.3)]
        assert run.warm_restarts == 1
        assert run.wall_seconds == 10000.0, "the largest Total elapsed of the ranks"
        assert run.complete and run.tracebacks == 0
        assert abs(run.peak_mem_alloc_gib - 6.4) < 1e-9
        assert run.epochs[0].lr is not None and run.epochs[0].elapsed == "00:01:54"


def test_a_chain_spans_its_legs_and_a_configuration_is_read_by_its_best_chain():
    with tempfile.TemporaryDirectory() as tmp:
        _standard_folder(tmp)
        runs = [RO.parse_log(p) for p in RO.collect_logs([tmp])]
        kept = [r for r in runs if r.epochs and not r.is_smoke]
        chains = {c["tag"]: c for c in RO.build_chains(kept)}
        a = chains["ARMA"]
        assert a["legs"] == 2 and a["job_ids"] == [101, 102] and a["last_global_epoch"] == 8
        assert a["best_test"] == -10.1 and a["best_global_epoch"] == 6 and a["best_job_id"] == 102
        assert a["leg_bests"] == [-9.3, -10.1] and a["warm_restarts"] == 1 and a["complete"] == 1
        assert a["wall_seconds"] == 19000.0
        confs = RO.build_configurations(list(chains.values()))
        assert [c["preset"] for c in confs] == ["capacity256_kernel7_earlyconv_stats", "capacity256"], "best first"
        cap = confs[1]
        assert cap["global_batch"] == 128 and cap["chains"] == 2 and cap["best_tag"] == "ARMA"
        assert cap["chain_bests"] == [-10.1, -8.5] and abs(cap["spread"] - 1.6) < 1e-9
        assert confs[0]["chains"] == 1 and confs[0]["spread"] is None


def test_smokes_and_empty_logs_are_excluded_unless_asked_and_the_outputs_are_written():
    with tempfile.TemporaryDirectory() as tmp:
        _standard_folder(tmp)
        out = Path(tmp) / "out"
        assert RO.main(["--logs", tmp, "--out-dir", str(out)]) == 0
        payload = json.loads((out / "readout.json").read_text())
        assert len(payload["runs"]) == 6, "every log is listed among the runs"
        assert sorted(c["tag"] for c in payload["chains"]) == ["ARMA", "ARMB", "ARMC"]
        assert len(payload["excluded_from_configurations"]) == 2
        assert any("smoke" in item for item in payload["excluded_from_configurations"])
        assert any("no epoch line" in item for item in payload["excluded_from_configurations"])
        md = (out / "readout.md").read_text()
        assert "| capacity256_kernel7_earlyconv_stats | 128 | 1 | -12.00000 |" in md
        assert "| capacity256 | 128 | 2 | -10.10000 | ARMA | 6 | -10.10000, -8.50000 | 1.60000 |" in md
        for name in ("readout_runs.csv", "readout_chains.csv", "readout_configurations.csv"):
            assert (out / name).exists()
        # included smokes
        out2 = Path(tmp) / "out2"
        assert RO.main(["--logs", tmp, "--out-dir", str(out2), "--include-smokes"]) == 0
        payload2 = json.loads((out2 / "readout.json").read_text())
        assert sorted(c["tag"] for c in payload2["chains"]) == ["ARMA", "ARMB", "ARMC", "SMOKEARMA"]


def test_duplicate_jobs_are_refused_the_dry_run_writes_nothing_and_an_existing_readout_is_kept():
    with tempfile.TemporaryDirectory() as tmp:
        _standard_folder(tmp)
        out = Path(tmp) / "out"
        assert RO.main(["--logs", tmp, "--out-dir", str(out), "--dry-run"]) == 0
        assert not out.exists(), "dry run writes nothing"
        assert RO.main(["--logs", tmp, "--out-dir", str(out)]) == 0
        assert RO.main(["--logs", tmp, "--out-dir", str(out)]) == 2, "refused without --overwrite"
        assert RO.main(["--logs", tmp, "--out-dir", str(out), "--overwrite"]) == 0
        other = Path(tmp) / "other"
        other.mkdir()
        _write(other, "ARMA", 101, preset="capacity256", gb=128, resurrect=False, tests=[-2.0])
        assert RO.main(["--logs", tmp, str(other), "--out-dir", str(Path(tmp) / "out3")]) == 2, "job 101 twice"


def test_a_log_without_the_header_line_is_read_from_its_settings_block():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "OLD", 7, preset="baseline", gb="None", resurrect=False, tests=[-1.0, -1.5], header=False)
        run = RO.parse_log(path)
        assert (run.tag, run.preset, run.resurrect, run.epochs_planned) == ("OLD", "baseline", False, 2)
        assert run.batch == 2 and run.world_size is None
        assert run.global_batch is None and run.global_batch_source == "unknown"
        assert run.best_test == -1.5 and run.best_global_epoch == 2
        path2 = _write(tmp, "", 8, preset="baseline", gb=1024, resurrect=False, tests=[-0.5])
        run2 = RO.parse_log(path2)
        assert run2.tag == "" and run2.global_batch == 1024, "the canonical (untagged) log name"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
