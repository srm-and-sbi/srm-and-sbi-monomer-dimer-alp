"""Working-tree provenance for numerical products.

A source revision alone does not identify the code that ran: a stage launched from a dirty tree,
or from a tree holding new untracked modules, executed code that no commit names. So every
Evaluation / Experiment product records three things separately:

- ``git_head``   -- the commit the tree was based on (``None`` outside a git checkout);
- ``git_dirty``  -- whether tracked files differed from that commit (``None`` when unknown);
- ``implementation`` -- a deterministic SHA-256 over the RELEVANT implementation files, taken from
  their relative paths and contents on disk regardless of tracking status, with any listed file
  that is absent recorded explicitly as missing (and hashed as such, so a missing file changes
  the hash rather than being silently skipped).

The implementation hash is what the artifact manifest and the Nuisance-DLI MAP-pool cache compare;
two runs agree on it exactly when the estimate-producing code was byte-identical.

A stage captures :func:`code_provenance` at STARTUP (the implementation it loaded) and, at write
time, :func:`finalize_code_provenance` recomputes the hash and records both under
``implementation`` / ``implementation_at_write`` with a ``changed_during_run`` flag. The schema
validator compares the two records itself (aggregate hash and per-file entries) and requires the
flag to agree: files edited while a job ran leave a product that describes no single
implementation, and it is refused. The refusal is fail-closed at publication, not fail-fast: the
computation has already finished, and the product is not written.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# The modules whose code determines the three point estimates and how they are written. Kept as
# repository-relative paths so the hash is the same on every machine that holds the same tree.
IMPLEMENTATION_FILES = (
    "srm_and_sbi_monomer_dimer_alp/evaluation.py",
    "srm_and_sbi_monomer_dimer_alp/evaluation_runner.py",
    "srm_and_sbi_monomer_dimer_alp/experiment_runner.py",
    "srm_and_sbi_monomer_dimer_alp/experiment_support.py",
    "srm_and_sbi_monomer_dimer_alp/artifact_schema.py",
    "srm_and_sbi_monomer_dimer_alp/provenance.py",
    "srm_and_sbi_monomer_dimer_alp/inference_support.py",
    "srm_and_sbi_monomer_dimer_alp/artifacts.py",
    "srm_and_sbi_monomer_dimer_alp/parameterization.py",
    # Estimate-producing paths outside the two stage runners: the standalone controls runner
    # (its own estimation loop and writer) and the Nuisance-DLI module (builds live MAP pools).
    "Script_Bank/Analysis/SRM_AND_SBI_MONOMER_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py",
    "srm_and_sbi_monomer_dimer_alp/detector_nuisance_dli.py",
)


def repo_root() -> Path:
    """The repository root, taken as the parent of the package directory."""
    return Path(__file__).resolve().parent.parent


def file_sha256(path) -> str:
    """Hex SHA-256 of a file's bytes (streamed; fine for multi-hundred-MB checkpoints)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def implementation_hash(root=None, files=IMPLEMENTATION_FILES) -> dict:
    """Deterministic hash of the implementation files: relative path + contents, in the listed
    order, tracking status ignored. Returns ``{"sha256", "files": [{"path", "sha256"} or
    {"path", "missing": True}]}``; a missing file is hashed as the token ``<missing>`` so the
    aggregate changes when a listed file disappears."""
    root = Path(root) if root is not None else repo_root()
    agg = hashlib.sha256()
    entries = []
    for rel in files:
        p = root / rel
        agg.update(rel.encode("utf-8") + b"\0")
        if p.is_file():
            digest = file_sha256(p)
            agg.update(digest.encode("ascii") + b"\0")
            entries.append({"path": rel, "sha256": digest})
        else:
            agg.update(b"<missing>\0")
            entries.append({"path": rel, "missing": True})
    return {"sha256": agg.hexdigest(), "files": entries}


def _git(root: Path, *args) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                             timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_head(root=None) -> str | None:
    """The HEAD commit hash, or ``None`` when the tree is not a git checkout."""
    return _git(Path(root) if root is not None else repo_root(), "rev-parse", "HEAD")


def git_dirty(root=None) -> bool | None:
    """``True`` when tracked files differ from HEAD (staged or not), ``False`` when clean,
    ``None`` when git cannot answer. Untracked files do not count here; they enter through the
    implementation hash when they are listed implementation files."""
    root = Path(root) if root is not None else repo_root()
    out = _git(root, "status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out)


def code_provenance(root=None, files=IMPLEMENTATION_FILES) -> dict:
    """The provenance block a product manifest stores: HEAD, dirty flag, implementation hash."""
    root = Path(root) if root is not None else repo_root()
    return {"git_head": git_head(root), "git_dirty": git_dirty(root),
            "implementation": implementation_hash(root, files)}


def finalize_code_provenance(startup: dict, root=None, files=IMPLEMENTATION_FILES) -> dict:
    """The code block a manifest stores: the startup provenance (HEAD, dirty flag, implementation
    hash of the files as loaded) plus the implementation hash recomputed NOW and whether the two
    differ. ``startup`` is the :func:`code_provenance` result captured before estimation began."""
    root = Path(root) if root is not None else repo_root()
    at_write = implementation_hash(root, files)
    return {
        "git_head": startup["git_head"],
        "git_dirty": startup["git_dirty"],
        "implementation": dict(startup["implementation"]),
        "implementation_at_write": at_write,
        "changed_during_run": at_write["sha256"] != startup["implementation"]["sha256"],
    }
