"""The HPC submit helpers and Experiment launchers of both workflows forward a comma-bearing window selection
as ONE value: through the exported environment (``--export=ALL``), never inside Slurm's comma-split explicit
export list, and into the stage's ``--aggregation-window-sets`` argument; the rank coordinate spaces of the
compatibility subsection (``COMPATIBILITY_COORDINATES``, comma- and semicolon-bearing) travel the same way into
``--compatibility-coordinates``. A fake ``sbatch`` and a fake ``python`` record what reaches them; no scheduler is
called and no job runs.

Run directly from the repository root::

    PYTHONPATH=$PWD python tests/test_hpc_submit_forwarding.py
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMITTERS = ("SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Submit.sh", "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Submit.sh")
LAUNCHERS = ("SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Experiment.sh", "SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_HPC_Experiment.sh")
SETS = "all,opening,late=5-9,odd=1+3"
SPACES = "spatial_brightness=mu_r,sigma_r,mu_pc,sigma_pc;widths=mu_r,sigma_r"       # two named spaces, commas and a semicolon


def _fake(path, body):
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _run(cmd, env, cwd):
    proc = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=120)
    return proc.returncode, proc.stdout + proc.stderr


def test_the_submitters_carry_the_window_sets_via_the_environment_never_the_explicit_export():
    for name in SUBMITTERS:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); bin_dir = tmp / "bin"; bin_dir.mkdir()
            record = tmp / "sbatch.json"
            # the fake scheduler records its arguments and the two environment variables of interest
            _fake(bin_dir / "sbatch", 'python3 - "$@" <<EOF\nimport json, os, sys\n'
                  'json.dump({"argv": sys.argv[1:], "env_sets": os.environ.get("AGGREGATION_WINDOW_SETS"), '
                  '"env_spaces": os.environ.get("COMPATIBILITY_COORDINATES"), '
                  '"env_kinds": os.environ.get("KINDS")}, open("' + str(record) + '", "w"))\nEOF\n')
            env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", DRYRUN="0", HPC_ENV="/dev/null")
            env.pop("AGGREGATION_WINDOW_SETS", None); env.pop("KINDS", None); env.pop("COMPATIBILITY_COORDINATES", None)
            rc, out = _run(["bash", str(REPO / "Script_Bank/HPC" / name), "experiment", "CONDITION=FAB",
                            f"AGGREGATION_WINDOW_SETS={SETS}", f"COMPATIBILITY_COORDINATES={SPACES}", "KINDS=FAB,INLB", "NO_AGGREGATION=0"], env, REPO)
            assert rc == 0, out
            rec = json.loads(record.read_text())
            export = [a for a in rec["argv"] if a.startswith("--export=")][0]
            assert "AGGREGATION_WINDOW_SETS" not in export and "KINDS" not in export and "COMPATIBILITY_COORDINATES" not in export, export
            assert "NO_AGGREGATION=0" in export and export.startswith("--export=ALL,"), export
            assert rec["env_sets"] == SETS and rec["env_kinds"] == "FAB,INLB" and rec["env_spaces"] == SPACES, rec
            # the dry-run names the comma-safe carriage of the three variables
            rc, out = _run(["bash", str(REPO / "Script_Bank/HPC" / name), "experiment", "CONDITION=FAB",
                            f"AGGREGATION_WINDOW_SETS={SETS}", f"COMPATIBILITY_COORDINATES={SPACES}", "KINDS=FAB,INLB"], dict(env, DRYRUN="1"), REPO)
            assert rc == 0 and f"AGGREGATION_WINDOW_SETS={SETS} (carried via ALL, comma-safe)" in out and "KINDS=FAB,INLB (carried via ALL" in out, out
            assert f"COMPATIBILITY_COORDINATES={SPACES} (carried via ALL, comma-safe)" in out, out
            line = [l for l in out.splitlines() if "--export     :" in l][0]
            assert "AGGREGATION_WINDOW_SETS" not in line and "late=5-9" not in line and "COMPATIBILITY_COORDINATES" not in line and "widths=" not in line, line
            # without the knob nothing is exported or forwarded for it
            rc, out = _run(["bash", str(REPO / "Script_Bank/HPC" / name), "experiment", "CONDITION=FAB", f"AGGREGATION_WINDOW_SETS={SETS}"], dict(env, DRYRUN="1"), REPO)
            assert rc == 0 and "COMPATIBILITY_COORDINATES" not in out, out


def test_the_launchers_pass_the_window_sets_to_the_worker_and_the_merge_as_one_argument():
    for name in LAUNCHERS:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); bin_dir = tmp / "bin"; bin_dir.mkdir()
            calls = tmp / "python_calls.jsonl"
            # a fake interpreter records every invocation's arguments (the launcher also asks it for a uuid)
            _fake(bin_dir / "python", 'if [ "$1" = "-c" ]; then echo fake-invocation-id; exit 0; fi\n'
                  'printf \'%s\\n\' "$(python3 -c \'import json,sys; print(json.dumps(sys.argv[1:]))\' "$@")" >> "' + str(calls) + '"\n')
            _fake(bin_dir / "srun", 'shift $(( $# - $(($# - 0)) )); while [ "${1#--}" != "$1" ]; do shift; done; "$@"\n')
            conda = _fake(tmp / "conda.sh", "conda() { :; }\n")
            env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", HPC_ENV="/dev/null", CONDA_SETUP=str(conda),
                       MACHINE_PROFILE="mars_pc", REPO=str(REPO), CONDITION="FAB", AGGREGATION_WINDOW_SETS=SETS, KINDS="FAB,INLB",
                       COMPATIBILITY_COORDINATES=SPACES, SRM_AND_SBI_GPUS="2", SLURM_NNODES="1", SLURM_CPUS_ON_NODE="8")
            rc, out = _run(["bash", str(REPO / "Script_Bank/HPC" / name)], env, REPO)
            assert rc == 0, out
            recorded = [json.loads(l) for l in calls.read_text().splitlines() if l.strip()]
            assert len(recorded) == 2, recorded                    # the sharded worker launch and the merge
            for argv in recorded:
                i = argv.index("--aggregation-window-sets")
                assert argv[i + 1] == SETS, argv
                j = argv.index("--compatibility-coordinates")
                assert argv[j + 1] == SPACES and argv.count("--compatibility-coordinates") == 1, argv     # ONE argument, both spaces
                k = argv.index("--kinds")
                assert argv[k + 1] == "FAB,INLB", argv
            assert "--merge" in recorded[1] and "--merge" not in recorded[0]
            # without the knob the option is absent from the worker and the merge
            calls.write_text("")
            env.pop("COMPATIBILITY_COORDINATES")
            rc, out = _run(["bash", str(REPO / "Script_Bank/HPC" / name)], env, REPO)
            assert rc == 0, out
            recorded = [json.loads(l) for l in calls.read_text().splitlines() if l.strip()]
            assert len(recorded) == 2 and all("--compatibility-coordinates" not in argv for argv in recorded), recorded


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as exc:                      # noqa: BLE001
            import traceback
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
