"""Fleet sync: every failure status propagates, and a live sync is reported as a success only after a
content verification.

The real ``Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_Fleet_Sync.sh`` runs against local directories that
stand in for the remote repositories. ``FLEET_FILE`` replaces the script's machine table, and a stand-in for
ssh maps each "host" to a directory and runs the remote command there. The stand-in can refuse the
connection, break the secrets check, make the repository unreadable while that check runs, or keep the names
of the files a transfer writes while losing their content, which is the state the files of the 2026-09-30
JUPITER sync were found in. No real machine is contacted: a guard ``ssh`` first on PATH refuses every
connection, and the module refuses to run a script without the ``FLEET_FILE`` hook. Run directly: it prints
one PASS line per test and exits non-zero on any failure.
"""
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "Script_Bank" / "HPC" / "SRM_AND_SBI_MONOMER_DIMER_ALP_Fleet_Sync.sh"
FAKE_REPO = "/fleet-test/repo"
REAL_HOSTS = ("login.jupiter.fz-juelich.de", "juwels-cluster.fz-juelich.de", "goethe.hhlr-gu.de", "10.83.255.103")
SAMPLE_FILES = ("pyproject.toml", "srm_and_sbi_monomer_dimer_alp/parameterization.py",
                "Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_Fleet_Sync.sh")
assert "FLEET_FILE" in SCRIPT.read_text(), "the script under test has no FLEET_FILE hook; refusing to run it live"
GUARD_SSH = "#!/bin/sh\necho 'test guard: a real ssh was attempted' >&2\nexit 255\n"

SHIM = r'''#!/bin/bash
# Stand-in for ssh (tests/test_fleet_sync.py). Argument 1 is the host; the rest is the remote command,
# joined and run by a shell as ssh would. FAKE_ROOT_<host> names the directory that plays the remote repo.
host="$1"; shift
key="${host//[^A-Za-z0-9]/_}"
root_var="FAKE_ROOT_$key"; root="${!root_var:-}"
refuse_var="FAKE_REFUSE_$key"
if [ -z "$root" ] || [ -n "${!refuse_var:-}" ]; then
    echo "ssh: connect to host $host port 22: Connection refused" >&2; exit 255
fi
cmd="$*"; cmd="${cmd//\/fleet-test\/repo/$root}"
secrets_var="FAKE_SECRETS_FAIL_$key"
if [ -n "${!secrets_var:-}" ] && [[ "$cmd" == *RECOVERY_CODES* ]]; then
    echo "ssh: connection to $host closed by remote host" >&2; exit 255
fi
unreadable_var="FAKE_SECRETS_UNREADABLE_$key"
if [ -n "${!unreadable_var:-}" ] && [[ "$cmd" == *RECOVERY_CODES* ]]; then
    # the repository cannot be read while the secrets check runs (a failed mount, a stale handle)
    chmod 000 "$root"; sh -c "$cmd"; rc=$?; chmod 775 "$root"; exit "$rc"
fi
read -ra words <<< "$cmd"
real_transfer=""
if [ "${words[0]:-}" = rsync ] && [ "${words[1]:-}" = --server ] && [[ "${words[2]:-}" != *n* ]]; then
    real_transfer=1                                  # an rsync receiver that writes (not a dry run)
fi
limit_var="FAKE_FILE_LIMIT_KIB_$key"
if [ -n "$real_transfer" ] && [ -n "${!limit_var:-}" ]; then
    ulimit -f "${!limit_var}"                        # writes beyond the limit fail: "File too large"
fi
lose_var="FAKE_LOSE_WRITES_$key"
if [ -n "${!lose_var:-}" ] && [ -n "$real_transfer" ]; then
    # a real (not dry-run) transfer: let it finish, then keep every written file's name but drop its content
    marker=$(mktemp); sleep 0.05
    sh -c "$cmd"; rc=$?
    find "$root" -type f -cnewer "$marker" -exec truncate -s 0 {} +
    rm -f "$marker"; exit "$rc"
fi
exec sh -c "$cmd"
'''


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Fleet:
    """A temporary fleet of local stand-in remotes for one test."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fleet_sync_test_"))
        self.shim = self.tmp / "fake_ssh.sh"
        self.shim.write_text(SHIM)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("FAKE_", "VERIFY_VIA_", "FLEET_"))}
        guard = self.tmp / "guard_bin"                       # a real ssh must never be reached from a test
        guard.mkdir()
        (guard / "ssh").write_text(GUARD_SSH)
        (guard / "ssh").chmod(0o755)
        self.env["PATH"] = f"{guard}{os.pathsep}{self.env.get('PATH', '')}"

    def target(self, name, with_package_dir=True):
        root = self.tmp / name
        root.mkdir()
        if with_package_dir:
            (root / "srm_and_sbi_monomer_dimer_alp").mkdir()
        return root

    def host(self, host, root):
        self.env[f"FAKE_ROOT_{host}"] = str(root)

    def run(self, machines, args=(), dryrun=False, **flags):
        fleet_file = self.tmp / "fleet.txt"
        fleet_file.write_text("".join(f"{name}|bash {self.shim}|{host}|{FAKE_REPO}\n" for name, host in machines))
        env = dict(self.env, FLEET_FILE=str(fleet_file), DRYRUN="1" if dryrun else "0")
        env.update({k: str(v) for k, v in flags.items()})
        proc = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=600)
        out = proc.stdout + proc.stderr
        for real in REAL_HOSTS:
            assert real not in out, f"a real machine appeared in the output: {real}"
        return proc.returncode, out

    def cleanup(self):
        for path in self.tmp.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
        shutil.rmtree(self.tmp, ignore_errors=True)


def _with_fleet(test):
    def wrapped():
        fleet = Fleet()
        try:
            test(fleet)
        finally:
            fleet.cleanup()
    wrapped.__name__ = test.__name__
    return wrapped


@_with_fleet
def test_a_clean_live_sync_is_verified_by_content_and_exits_zero(fleet):
    a = fleet.target("a")
    fleet.host("nodeA", a)
    rc, out = fleet.run([("t1", "nodeA")])
    assert rc == 0, out
    assert "verified: 0 content differences (checksums, read through nodeA)" in out, out
    assert "secrets check: RECOVERY_CODES.md absent  OK" in out, out
    assert "synced and verified: t1" in out and "FAILED" not in out, out
    assert "note: read back through the session that wrote the files" in out, out   # the default is said out loud
    assert "read back only through the session that wrote the files" in out and ": t1" in out, out
    for rel in SAMPLE_FILES:
        assert _sha(a / rel) == _sha(REPO / rel), rel
    assert not (a / "RECOVERY_CODES.md").exists() and not (a / "machine_profiles.toml").exists()
    rc, out = fleet.run([("t1", "nodeA")])                     # a second run finds nothing to change
    assert rc == 0 and "(already identical by size and time)" in out and "verified: 0 content" in out, out


@_with_fleet
def test_an_rsync_failure_is_reported_with_its_messages_and_exits_nonzero(fleet):
    a = fleet.target("a")
    fleet.host("nodeA", a)
    rc, out = fleet.run([("t1", "nodeA")], FAKE_FILE_LIMIT_KIB_nodeA=16)   # the receiver cannot write past 16 KiB
    assert rc != 0, out
    assert "     <f+++++++++ .gitignore" in out, out                 # the itemized lines up to the failure ...
    assert "     > rsync: write failed on" in out and "File too large" in out, out   # ... and rsync's errors
    assert "FAILED: rsync exited with status 11; the remote may be partially updated" in out, out
    assert "already identical" not in out and "verified:" not in out and "synced and verified" not in out, out
    assert "FAILED t1: rsync exited with status 11" in out, out


@_with_fleet
def test_files_whose_content_is_lost_after_the_transfer_fail_the_verification(fleet):
    a = fleet.target("a")
    fleet.host("nodeA", a)
    rc, out = fleet.run([("t1", "nodeA")], FAKE_LOSE_WRITES_nodeA=1)
    assert rc != 0, out
    assert "content difference(s) remain after the transfer (read through nodeA)" in out, out
    assert "     differs: " in out and "verified:" not in out and "synced and verified" not in out, out
    assert (a / "pyproject.toml").exists() and (a / "pyproject.toml").stat().st_size == 0   # the stand-in did its job


@_with_fleet
def test_an_unreachable_machine_fails_and_nothing_is_transferred(fleet):
    a = fleet.target("a")
    fleet.host("nodeA", a)
    rc, out = fleet.run([("t1", "nodeA")], FAKE_REFUSE_nodeA=1)
    assert rc != 0, out
    assert "FAILED: unreachable or repo missing (status 255); nothing transferred" in out, out
    assert "     > ssh: connect to host nodeA port 22: Connection refused" in out, out
    assert sorted(p.name for p in a.iterdir()) == ["srm_and_sbi_monomer_dimer_alp"]


@_with_fleet
def test_the_verification_reads_through_the_named_host(fleet):
    a = fleet.target("a")
    stale = fleet.target("stale")                             # another view of "the same" repo, without the files
    fleet.host("nodeA", a)
    fleet.host("nodeB", stale)
    fleet.host("nodeC", a)                                    # a second node that sees the same directory
    rc, out = fleet.run([("t1", "nodeA")], VERIFY_VIA_t1="nodeB")
    assert rc != 0, out
    assert "content difference(s) remain after the transfer (read through nodeB)" in out, out
    rc, out = fleet.run([("t1", "nodeA")], VERIFY_VIA_t1="nodeC")
    assert rc == 0, out
    assert "verified: 0 content differences (checksums, read through nodeC)" in out, out
    assert "read back through the session that wrote the files" not in out, out
    assert "read back only through the session" not in out, out
    rc, out = fleet.run([("t1", "nodeA")], VERIFY_VIA_t1="nodeD")   # a verification host that cannot be reached
    assert rc != 0 and "the verification through nodeD could not run" in out, out


@_with_fleet
def test_the_secrets_check_fails_closed(fleet):
    a = fleet.target("a")
    fleet.host("nodeA", a)
    rc, out = fleet.run([("t1", "nodeA")], FAKE_SECRETS_FAIL_nodeA=1)
    assert rc != 0, out
    assert "verified: 0 content differences" in out and "FAILED: the secrets check could not run (status 255)" in out, out
    assert "absent  OK" not in out, out
    (a / "RECOVERY_CODES.md").write_text("placeholder\n")      # excluded by name, so the sync leaves it in place
    rc, out = fleet.run([("t1", "nodeA")])
    assert rc != 0 and "FAILED: the secrets file RECOVERY_CODES.md is present on t1" in out, out
    (a / "RECOVERY_CODES.md").unlink()
    os.symlink("/nonexistent/RECOVERY_CODES.md", a / "RECOVERY_CODES.md")   # a dangling link is still a file by that name
    rc, out = fleet.run([("t1", "nodeA")])
    assert rc != 0 and "FAILED: the secrets file RECOVERY_CODES.md is present on t1" in out, out


@_with_fleet
def test_the_secrets_check_fails_closed_when_the_repository_cannot_be_read(fleet):
    a = fleet.target("a")
    fleet.host("nodeA", a)
    (a / "RECOVERY_CODES.md").write_text("placeholder\n")      # present, but unreadable while the check runs
    rc, out = fleet.run([("t1", "nodeA")], FAKE_SECRETS_UNREADABLE_nodeA=1)
    assert rc != 0, out
    assert "verified: 0 content differences" in out, out       # the transfer and its verification were fine ...
    assert "FAILED: the secrets check could not run (status 3)" in out, out   # ... and the check still fails closed
    assert "absent  OK" not in out and "synced and verified" not in out, out


@_with_fleet
def test_machine_names_are_checked_before_anything_is_transferred(fleet):
    a, b = fleet.target("a"), fleet.target("b")
    fleet.host("nodeA", a)
    fleet.host("nodeB", b)
    rc, out = fleet.run([("t1", "nodeA"), ("rcl-02", "nodeB")])   # a name that cannot follow VERIFY_VIA_
    assert rc != 0, out
    assert "FAILED rcl-02: the machine name is not usable in VERIFY_VIA_<name>" in out, out
    assert "synced and verified: t1" in out, out                 # the other machine is still processed
    assert sorted(p.name for p in b.iterdir()) == ["srm_and_sbi_monomer_dimer_alp"]   # nothing reached it
    rc, out = fleet.run([("t1", "nodeA")], args=("t1", "t3"))    # a requested name that is not in the table
    assert rc != 0 and "synced and verified: t1" in out and "FAILED t3: no such machine in the table" in out, out
    rc, out = fleet.run([("t.1", "nodeA")], args=("tx1",))       # names match literally, not as patterns
    assert rc != 0 and "---- t.1" not in out, out


@_with_fleet
def test_a_dry_run_transfers_nothing_and_still_reports_failures(fleet):
    a = fleet.target("a")
    fleet.host("nodeA", a)
    rc, out = fleet.run([("t1", "nodeA")], dryrun=True)
    assert rc == 0, out
    assert "DRY RUN: nothing was transferred." in out and "planned without error: t1" in out, out
    assert "<f+++++++++ pyproject.toml" in out and "verified:" not in out, out
    assert sorted(p.name for p in a.iterdir()) == ["srm_and_sbi_monomer_dimer_alp"]
    assert not any((a / "srm_and_sbi_monomer_dimer_alp").iterdir())
    rc, out = fleet.run([("t1", "nodeA")], dryrun=True, FAKE_REFUSE_nodeA=1)
    assert rc != 0 and "FAILED: unreachable" in out, out


@_with_fleet
def test_one_failing_machine_fails_the_run_and_the_others_are_still_synced(fleet):
    a, b = fleet.target("a"), fleet.target("b")
    fleet.host("nodeA", a)
    fleet.host("nodeB", b)
    rc, out = fleet.run([("t1", "nodeA"), ("t2", "nodeB")], FAKE_REFUSE_nodeB=1)
    assert rc != 0, out
    assert "synced and verified: t1" in out and "FAILED t2: unreachable" in out, out
    assert _sha(a / "pyproject.toml") == _sha(REPO / "pyproject.toml")
    rc, out = fleet.run([("t1", "nodeA"), ("t2", "nodeB")], args=("t1",), FAKE_REFUSE_nodeB=1)
    assert rc == 0 and "---- t2" not in out, out                 # only the named machine is touched
    rc, out = fleet.run([("t1", "nodeA")], args=("nosuch",))
    assert rc != 0 and "FAILED: no machine matched: nosuch" in out, out


if __name__ == "__main__":
    import time
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as exc:                      # noqa: BLE001 -- report every failure
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
