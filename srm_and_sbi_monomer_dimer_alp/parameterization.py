"""Sibling configuration: per-machine paths (loaded from machine_profiles.toml)
+ sibling-wide defaults + the rich parameter spec.

This module is the single source of truth for sibling configuration. Imported by
every entry-point script and most package modules. Validates the active machine
profile at import time (refuse-on-missing: import fails fast with a clear error
if the env var is absent or the config is incomplete).

Public interface:
    PARAMETERS                  -- module-level singleton; typed access:
                                   PARAMETERS.paths.*, PARAMETERS.simulation.*,
                                   PARAMETERS.inference.*, PARAMETERS.plotting.*
    PARAMETERIZATION_RAW        -- flat list of all parameter entries (full spec)
    PARAMETERIZATION            -- filtered list of learnable parameters (for the prior)
    PARAMETER_RAW_FIND          -- dict[KEY -> index in PARAMETERIZATION_RAW]
    PARAMETER_FIND              -- dict[KEY -> index in PARAMETERIZATION]
    parameter_find(key)         -- index lookup for learnable parameters
    build_prior(device)         -- construct the BoxUniform prior over the estimator space
    theta_lower_bound()         -- lower bounds of the prior box (estimator space)
    theta_upper_bound()         -- upper bounds of the prior box (estimator space)
    to_physical(theta_flow)     -- the ONE conversion estimator space -> physical values
    to_flow(theta_physical)     -- its inverse (physical -> estimator space)
    prior_center(entry)         -- physical value at the center of a ranged row
    realize_initial_composition -- integer (n_A, n_B) from the sampled (N_R, r), r = n_B/n_A
    association_ratio_of(cond)  -- the declared association ratio R_ON of a condition (FAB 0, INLB 1)
    occupancy_of(cond)          -- the declared (INLB) or derived (FAB) probe occupancy per subunit
    visibility_of(cond)         -- occupancy x P(dye >= 1) under the condition's labeling law

Model blocks (the declarative records the reaction-diffusion generator is built from):
    StoichiometryBlock          -- molecular species A (1 subunit) and B (2), the dissociation
                                   channel's parameter key and the composition keys
    MobilityBlock               -- mobility modes, their diffusion-ratio keys, the sequential
                                   switching chain and its rate keys, the inheritance rule
    ConditionSetting            -- the per-condition settings: the association ratio R_ON
                                   (0 = no association channels; a declared constant, never
                                   inferred) and the probe OCCUPANCY of the visibility layer
                                   (declared for MET-INLB; derived for MET-FAB from a declared
                                   Fab/InlB visibility ratio and the InlB anchor)
    PARAMETERS.simulation.rds   -- carries the blocks and the condition settings and derives
                                   the PARTICLE TYPES (species x mode), their subunit counts
                                   and the maps type -> species / mode

The estimator space ("flow space"). Every ranged row declares its scale: LOG_FLAG True means
the prior box and the estimator coordinate are log10 of the physical value; LOG_FLAG False
means the coordinate IS the physical value (a linear row). `to_physical` / `to_flow` apply
the per-row rule and are the only sanctioned conversion -- no consumer exponentiates or
log-transforms a theta vector by hand. The decided table (2026-09-14) has no linear row --
the initial composition is the log10 dimer-to-monomer ratio -- but the per-row rule stays
the contract, so a linear row can be declared without touching any consumer.

Prior ranges. Every range is a DECIDED box (2026-09-14; PROJECT_CONTEXT.md sec. 2, "How the
prior ranges and the declared inputs are set"): box-uniform in the estimator coordinate,
covering the plausible support with a margin and never encoding the answer an experiment is
expected to give; both conditions share the one table. Each row's DOC names its source:
the frozen baseline ranges of the earlier model (dimer-alp 0.4.23), the per-recording analysis
of the deposited MET localization tables (Special_Analyses A9), the literature anchors of the
model specification (sec. 9), or the tracking pipelines' resolution and classification
thresholds.
"""

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sbi.utils import BoxUniform

# tomllib is stdlib in Python 3.11+ (this env uses 3.13); fall back to tomli on older Pythons.
try:
    import tomllib
except ImportError:  # Python 3.9 / 3.10
    import tomli as tomllib


# =============================================================================
# Machine profile loader
# =============================================================================

@dataclass(frozen=True)
class MachineProfile:
    """Per-machine configuration; loaded from machine_profiles.toml.

    Selected via the MACHINE_PROFILE environment variable. Refuse-on-missing
    is enforced at import time: if the env var is absent, the profile name is
    unknown, required keys are missing, or the named root directories don't
    exist on disk, an error with a clear remedy is raised. This forces the
    user to fix configuration problems before any simulation or training starts,
    rather than failing mid-run.
    """
    name: str
    running_mode: str             # "LOCAL" (prototyping) | "HPC" (production)
    script_bank_root: Path        # full path to Script_Bank/ folder
    data_bank_root: Path          # full path to Data_Bank/ folder (permanent / backed-up tier)
    compute_backend: str          # "GPU" | "CPU"
    gpu_device_index: Optional[int]
    num_workers: int
    # Optional second data tier for machines whose scratch space is large/fast
    # but impermanent (e.g. an HPC scratch filesystem that is auto-purged and not
    # backed up). When configured, regenerable TRAIN/TEST data is routed here;
    # everything to keep -- EVAL, posteriors, checkpoints, experimental data --
    # stays on data_bank_root. When left unset, the machine is single-tier and
    # every split uses data_bank_root (the behavior on single-filesystem machines).
    scratch_data_bank_root: Optional[Path] = None

    def root_for(self, split: str) -> Path:
        """Data-bank root for a generation ``split``.

        Regenerable ``TRAIN``/``TEST`` data goes to ``scratch_data_bank_root``
        when that tier is configured; ``EVAL`` -- like all non-split artifacts
        (posteriors, checkpoints, experimental data) -- always lives on the
        permanent ``data_bank_root``. With no scratch tier configured, every
        split resolves to ``data_bank_root``.
        """
        if self.scratch_data_bank_root is not None and split.upper() in {"TRAIN", "TEST"}:
            return self.scratch_data_bank_root
        return self.data_bank_root


def load_machine_profile() -> MachineProfile:
    """Load and validate the active machine profile.

    Reads MACHINE_PROFILE env var, parses machine_profiles.toml at sibling repo
    root, validates required keys and types, and returns a MachineProfile.

    Raises ValueError with a clear message pointing to machine_profiles.example.toml
    on any failure.
    """
    profile_name = os.environ.get("MACHINE_PROFILE")
    if not profile_name:
        raise ValueError(
            "MACHINE_PROFILE environment variable is not set. "
            "Set it to the name of an active profile in machine_profiles.toml. "
            "See machine_profiles.example.toml at the sibling repo root for the schema."
        )

    repo_root = Path(__file__).resolve().parent.parent  # parent of the package directory
    profiles_path = repo_root / "machine_profiles.toml"
    if not profiles_path.exists():
        raise ValueError(
            f"machine_profiles.toml not found at {profiles_path}. "
            f"Copy machine_profiles.example.toml to machine_profiles.toml and edit "
            f"for your machine. machine_profiles.toml is gitignored to keep "
            f"per-machine absolute paths out of version control."
        )

    with open(profiles_path, "rb") as fh:
        profiles = tomllib.load(fh)

    if profile_name not in profiles:
        raise ValueError(
            f"Profile '{profile_name}' not found in {profiles_path}. "
            f"Available profiles: {sorted(profiles.keys())}."
        )

    profile = profiles[profile_name]
    required_keys = {
        "running_mode", "script_bank_root", "data_bank_root",
        "compute_backend", "num_workers",
    }
    missing = required_keys - profile.keys()
    if missing:
        raise ValueError(
            f"Profile '{profile_name}' is missing required keys: {sorted(missing)}. "
            f"See machine_profiles.example.toml for the schema."
        )

    if profile["running_mode"] not in {"LOCAL", "HPC"}:
        raise ValueError(
            f"Profile '{profile_name}' has running_mode={profile['running_mode']!r}; "
            f"must be exactly 'LOCAL' or 'HPC' (case-sensitive). "
            f"'LOCAL' is for prototyping (single GPU, small task counts); "
            f"'HPC' is for production (cluster, multi-task parallelism)."
        )

    if profile["compute_backend"] not in {"GPU", "CPU"}:
        raise ValueError(
            f"Profile '{profile_name}' has compute_backend={profile['compute_backend']!r}; "
            f"must be exactly 'GPU' or 'CPU'."
        )

    gpu_device_index = profile.get("gpu_device_index")
    if profile["compute_backend"] == "GPU" and gpu_device_index is None:
        raise ValueError(
            f"Profile '{profile_name}' has compute_backend='GPU' but no gpu_device_index. "
            f"Set gpu_device_index to a non-negative integer."
        )

    script_bank_root = Path(profile["script_bank_root"])
    data_bank_root = Path(profile["data_bank_root"])
    if not script_bank_root.is_dir():
        raise ValueError(
            f"Profile '{profile_name}' script_bank_root={script_bank_root} is not a directory."
        )
    if not data_bank_root.is_dir():
        raise ValueError(
            f"Profile '{profile_name}' data_bank_root={data_bank_root} is not a directory."
        )

    # Optional second (scratch) data tier. Present only on machines with split
    # storage; when absent the machine is single-tier (see MachineProfile.root_for).
    scratch_data_bank_root = None
    if profile.get("scratch_data_bank_root"):
        scratch_data_bank_root = Path(profile["scratch_data_bank_root"])
        if not scratch_data_bank_root.is_dir():
            raise ValueError(
                f"Profile '{profile_name}' scratch_data_bank_root={scratch_data_bank_root} "
                f"is not a directory."
            )

    return MachineProfile(
        name=profile_name,
        running_mode=profile["running_mode"],
        script_bank_root=script_bank_root,
        data_bank_root=data_bank_root,
        compute_backend=profile["compute_backend"],
        gpu_device_index=gpu_device_index,
        num_workers=int(profile["num_workers"]),
        scratch_data_bank_root=scratch_data_bank_root,
    )


# =============================================================================
# Sibling-wide defaults (frozen dataclasses)
# =============================================================================

@dataclass(frozen=True)
class Paths:
    """Sibling-wide path conventions and naming patterns.

    Defines the layout of output files under the configured `data_bank_root`,
    the subdirectory names for each data category (trajectories, videos, theta
    sets, checkpoints, posteriors), and Python `str.format` patterns for
    concrete filenames. Path-construction methods compose subdirectories and
    patterns into full filesystem paths.

    The `project_alias` is embedded in every output filename to preserve
    provenance: a `.h5` or `.zarr` file moved or shared outside the original
    repository still identifies the program, model, and iteration that
    produced it.

    The runtime grammar is ``[program]_[sibling]_[iter][_qualifier]_[condition]_
    [timing][_tag]_[stage]`` (``product_label`` composes the optional product tag): the optional workflow qualifier (``_DETECTOR``) and the
    experimental-condition token (``FAB`` or ``INLB``) sit between the iteration
    and the timing label, so ``SRM_AND_SBI_MONOMER_DIMER_ALP_FAB_2S_50FPS_Video_Set_...``
    is a biology product and ``SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_FAB_2S_50FPS_...``
    a detector product of the MET-FAB condition. The condition enters at the RDS
    stage: the association intensity of the reaction-diffusion model is a per-condition
    constant (``SimulationRDS.conditions``), so the trajectory tier and its
    eleven-parameter ``Theta_Set`` are generated once PER CONDITION under the sibling
    alias plus the condition token (``rds_alias``, e.g. ``SRM_AND_SBI_MONOMER_DIMER_ALP_FAB``),
    free of the workflow qualifier, and shared by both workflows, which re-image them at
    the DLI stage under the condition's labeling law. Every product therefore carries
    the condition. A stage applies its condition with ``with_condition``; the trajectory
    and theta-set builders below pick the right alias by themselves.
    """
    project_alias: str = "SRM_AND_SBI_MONOMER_DIMER_ALP"
    labor_subdir: str = "Labor"
    debug_subdir: str = "Debug"               # diagnostics dumps, nested under labor_subdir
    posit_subdir: str = "Posit"
    theta_subdir: str = "Theta"
    video_subdir: str = "Video"
    experiment_subdir: str = "Experiment/SPT_Data_MET_FAB_INLB_S-BSST712"  # real microscopy data, BioStudies S-BSST712 (nested by accession for provenance)
    trajectory_repo: str = "READY_TRACT"
    trajectory_pattern: str = "{project_alias}_{timing_label}_TASK_{task_alias}_SIM_{task_simulation}_{split}.h5"
    theta_set_pattern: str = "{project_alias}_{timing_label}_Theta_Set_TASK_{task_alias}_{split}.{ext}"
    video_set_pattern: str = "{project_alias}_{timing_label}_Video_Set_TASK_{task_alias}_{split}.{ext}"
    checkpoint_pattern: str = "{project_alias}_{timing_label}_Optimum_ANN.pth"
    resurrect_state_pattern: str = "{project_alias}_{timing_label}_Resurrect_State_ANN.pth"
    estimator_pattern: str = "{project_alias}_{timing_label}_Estimator.npz"
    test_loss_distribution_pattern: str = "{project_alias}_{timing_label}_Test_Loss_Distribution.npz"
    recovery_pattern: str = "{project_alias}_{timing_label}_MAP_Recovery"
    experiment_recovery_pattern: str = "{project_alias}_{timing_label}_MAP_Experiment"
    # Real microscopy videos are external raw data (not produced by this pipeline),
    # so they keep their native naming rather than the project_alias prefix. The
    # user copies them into <data_bank_root>/<experiment_subdir>/.
    experiment_pattern: str = "Experiment_{kind}_Cell_{cell}_{span}S_RAW.tif"
    compressed_ext: str = "zarr"
    uncompressed_ext: str = "npy"
    # The bare sibling alias. The RDS products -- one trajectory tier per condition and its
    # eleven-parameter ``Theta_Set`` -- carry this alias plus the condition token and never
    # the workflow qualifier, because both workflows re-image the same tier (``rds_alias``
    # composes it).
    sibling_alias: str = "SRM_AND_SBI_MONOMER_DIMER_ALP"
    # The condition token these Paths carry (None = no condition applied yet; see the class
    # docstring). Set only through ``with_condition``, which also appends the token to
    # ``project_alias``; ``rds_alias`` composes the tier alias from it.
    condition: Optional[str] = None
    # Whether this workflow's ``Theta_Set`` is an RDS product (biology: the eleven
    # reaction-diffusion labels -- the condition's tier's own record, under ``rds_alias``) or
    # a DLI product (detector: the six imaging labels, drawn when the videos are rendered,
    # hence qualified and per condition).
    theta_set_is_rds_product: bool = True

    # The `timing_label` token (e.g., "2S_50FPS") encodes simulation duration
    # + frame rate in every output filename, so a 2 s and 10 s run never
    # collide on disk and a file moved out of context still identifies the
    # config that produced it. The token is rendered by RunTiming.label.

    def with_condition(self, condition: str) -> "Paths":
        """A copy of these Paths carrying the experimental condition.

        ``project_alias`` gains the token (``..._ALP_FAB``, ``..._ALP_DETECTOR_INLB``), so
        every path pattern namespaces this condition's products, and ``rds_alias`` resolves
        this condition's RDS tier. Applying a condition twice is an error.
        """
        if self.condition is not None:
            raise ValueError(
                f"Paths already carry condition {self.condition!r}; cannot apply {condition!r}.")
        token = str(condition).strip()
        if not token or not token.isupper() or not token.isalnum():
            raise ValueError(
                f"condition token must be an uppercase alphanumeric word such as FAB or INLB "
                f"(got {condition!r}).")
        return replace(self, project_alias=f"{self.project_alias}_{token}", condition=token)

    @property
    def rds_alias(self) -> str:
        """The alias of the condition's RDS tier (trajectories and the eleven-parameter
        ``Theta_Set``): the bare sibling alias plus the condition token, whatever workflow
        qualifier these Paths carry. Requires a condition (``with_condition``): the tier is
        per condition because the association setting is."""
        if not self.project_alias.startswith(self.sibling_alias):
            raise ValueError(
                f"project_alias {self.project_alias!r} does not extend the sibling alias "
                f"{self.sibling_alias!r}; the RDS tier cannot be located from it.")
        if self.condition is None:
            raise ValueError(
                "the RDS trajectory tier is per condition (the association setting differs "
                "between MET-FAB and MET-INLB); apply Paths.with_condition('FAB' | 'INLB') "
                "before resolving rds_alias.")
        return f"{self.sibling_alias}_{self.condition}"

    @property
    def theta_set_alias(self) -> str:
        """Alias of this workflow's ``Theta_Set``: the condition's RDS-tier alias for an RDS
        product (biology: the eleven reaction-diffusion labels), the qualified and conditioned
        alias for a DLI product (detector: the six imaging labels)."""
        return self.rds_alias if self.theta_set_is_rds_product else self.project_alias

    def trajectory_dir(self, task_alias: int, data_bank_root: Path,
                       timing_label: str, split: str = "TRAIN") -> Path:
        """Per-task subdirectory for .h5 trajectory files (the condition's RDS tier).

        ``split`` ∈ {"TRAIN", "TEST", "EVAL"} namespaces the data by role so
        the held-out sets never collide with the training set on disk.
        """
        return (data_bank_root / self.video_subdir / self.trajectory_repo /
                f"{self.rds_alias}_{timing_label}_TASK_{task_alias}_{split}")

    def trajectory_path(self, task_alias: int, task_simulation: int,
                        data_bank_root: Path, timing_label: str,
                        split: str = "TRAIN") -> Path:
        """Full path for a single .h5 trajectory file (the condition's RDS tier)."""
        filename = self.trajectory_pattern.format(
            project_alias=self.rds_alias,
            timing_label=timing_label,
            task_alias=task_alias,
            task_simulation=task_simulation,
            split=split,
        )
        return self.trajectory_dir(task_alias, data_bank_root, timing_label, split) / filename

    def theta_set_path(self, task_alias: int, data_bank_root: Path,
                       timing_label: str, compress: bool = True,
                       split: str = "TRAIN") -> Path:
        """Full path for a theta-set file (.zarr if compress, else .npy); see
        ``theta_set_alias`` for which alias it carries."""
        ext = self.compressed_ext if compress else self.uncompressed_ext
        filename = self.theta_set_pattern.format(
            project_alias=self.theta_set_alias,
            timing_label=timing_label,
            task_alias=task_alias,
            ext=ext,
            split=split,
        )
        return data_bank_root / self.theta_subdir / filename

    def record_set_path(self, token: str, task_alias: int, data_bank_root: Path,
                        timing_label: str, compress: bool = True,
                        split: str = "TRAIN") -> Path:
        """Full path for a per-task record set written beside the theta set: the theta-set
        pattern with ``Theta_Set`` replaced by ``token`` (``Nuisance_DLI_Theta_Set``,
        ``Nuisance_SCOPE_Theta_Set``, ``Labeling_Set``), under THIS Paths' alias -- the
        qualified, conditioned alias of the DLI stage that writes them.
        """
        if "Theta_Set" not in self.theta_set_pattern:
            raise ValueError("theta_set_pattern must contain the 'Theta_Set' token.")
        ext = self.compressed_ext if compress else self.uncompressed_ext
        filename = self.theta_set_pattern.replace("Theta_Set", token).format(
            project_alias=self.project_alias,
            timing_label=timing_label,
            task_alias=task_alias,
            ext=ext,
            split=split,
        )
        return data_bank_root / self.theta_subdir / filename

    def video_set_path(self, task_alias: int, data_bank_root: Path,
                       timing_label: str, compress: bool = True,
                       split: str = "TRAIN") -> Path:
        """Full path for a video-set file (.zarr if compress, else .npy)."""
        ext = self.compressed_ext if compress else self.uncompressed_ext
        filename = self.video_set_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
            task_alias=task_alias,
            ext=ext,
            split=split,
        )
        return data_bank_root / self.video_subdir / filename

    @staticmethod
    def product_label(timing_label: str, artifact_tag: Optional[str] = None) -> str:
        """Label under which a run's PRODUCTS are written: the timing label, optionally
        followed by an artifact tag (``2S_50FPS`` -> ``2S_50FPS_CAP256``).

        The tag namespaces one training run's outputs -- checkpoint, resurrect state,
        estimator artifact, test-loss distribution and their backups, the debug run
        directory, and every downstream product derived from that estimator (MAP
        recovery, posterior calibration, experiment reports) -- so an experiment such
        as the capacity test of DETECTOR_WORKFLOW.md sec. 9.7 lives beside the
        baseline instead of overwriting it. It sits right after the timing label,
        before the stage descriptor, and is a SCREAMING_SNAKE token (``[A-Z0-9]+``,
        no underscore, so the grammar stays unambiguous). It never enters the shared
        INPUTS (video, theta, and record sets, experimental recordings): those are
        read under the plain timing label by every run. No tag (the default) returns
        the timing label unchanged, so the canonical products keep their names.
        """
        if artifact_tag is None or artifact_tag == "":
            return timing_label
        if not re.fullmatch(r"[A-Z0-9]+", artifact_tag):
            raise ValueError(
                f"artifact_tag {artifact_tag!r} must be a SCREAMING_SNAKE token without "
                f"underscores ([A-Z0-9]+), e.g. 'CAP256'.")
        return f"{timing_label}_{artifact_tag}"

    def checkpoint_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the optimum-ANN checkpoint file."""
        filename = self.checkpoint_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )
        return data_bank_root / self.labor_subdir / filename

    def resurrect_state_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the resurrect-state file -- the complete training state
        (model + optimizer + scheduler + epoch + optimum + warm-restart counters)
        written atomically every epoch so a ``--resurrect`` requeue hot-restarts from
        the exact latest state instead of a fresh optimizer at peak LR.

        A transient resume file, not a scientific deliverable: it is always
        overwritten with the latest state and carries no provenance descriptor
        (unlike the checkpoint backups), so the read logic is a single ``exists()``
        check. Lives beside the optimum checkpoint in ``labor_subdir``.
        """
        filename = self.resurrect_state_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )
        return data_bank_root / self.labor_subdir / filename

    def estimator_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the version-portable estimator artifact (.npz).

        The self-describing estimator format (compile-stripped state_dict + rebuild
        spec + metadata; see ``artifacts.save_estimator``) that supersedes the
        torch-version-locked posterior pickle. Lives beside the former posterior in
        ``posit_subdir``.
        """
        filename = self.estimator_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )
        return data_bank_root / self.posit_subdir / filename

    def test_loss_distribution_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the best-epoch test-loss distribution (.npz), a
        scientific deliverable alongside the posterior it characterizes."""
        filename = self.test_loss_distribution_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )
        return data_bank_root / self.posit_subdir / filename

    # ---- Backup / archival artifact names --------------------------------
    # A finished training run overwrites the canonical checkpoint + posterior above
    # (the names every downstream stage loads). To keep an identifiable, restorable
    # history, each finished run also writes a backup copy whose name embeds the
    # run's provenance -- train/test set sizes, epoch count, and best TEST loss --
    # since the bare state_dict and posterior pickle carry no such metadata. The
    # canonical files remain the live objects; a backup is restored by copying it
    # back onto the canonical name.
    @staticmethod
    def format_backup_loss(test_loss: float) -> str:
        """Signed TEST-loss token with exactly two decimals: ``-17.05``, ``+3.20``.

        A value that rounds to zero (either sign) renders ``0.00`` with no sign.
        Operates on the formatted string, so it never relies on float equality.
        """
        text = f"{test_loss:.2f}"
        if text in ("0.00", "-0.00"):
            return "0.00"
        return text if text.startswith("-") else f"+{text}"

    @staticmethod
    def format_backup_size(n_videos: int) -> str:
        """Video count as a compact ``K`` token: 200000 -> ``200K``, 47500 -> ``47.5K``."""
        if n_videos % 1000 == 0:
            return f"{n_videos // 1000}K"
        return f"{n_videos / 1000:g}K"

    def backup_descriptor(self, train_videos: int, test_videos: int,
                          epochs: int, test_loss: float) -> str:
        """Provenance token appended to a backup filename, e.g.
        ``TRAIN+TEST_200K+50K_Epoch_25_TEST_LOSS_-17.05``."""
        return (
            f"TRAIN+TEST_{self.format_backup_size(train_videos)}"
            f"+{self.format_backup_size(test_videos)}"
            f"_Epoch_{epochs}_TEST_LOSS_{self.format_backup_loss(test_loss)}"
        )

    def backup_checkpoint_path(self, data_bank_root: Path, timing_label: str,
                               train_videos: int, test_videos: int,
                               epochs: int, test_loss: float) -> Path:
        """Provenance-named backup of the checkpoint, alongside the canonical one."""
        base = self.checkpoint_pattern.format(
            project_alias=self.project_alias, timing_label=timing_label)
        stem, ext = base.rsplit(".", 1)
        descriptor = self.backup_descriptor(train_videos, test_videos, epochs, test_loss)
        return data_bank_root / self.labor_subdir / f"{stem}_{descriptor}.{ext}"

    def backup_estimator_path(self, data_bank_root: Path, timing_label: str,
                              train_videos: int, test_videos: int,
                              epochs: int, test_loss: float) -> Path:
        """Provenance-named backup of the estimator artifact, alongside the canonical one."""
        base = self.estimator_pattern.format(
            project_alias=self.project_alias, timing_label=timing_label)
        stem, ext = base.rsplit(".", 1)
        descriptor = self.backup_descriptor(train_videos, test_videos, epochs, test_loss)
        return data_bank_root / self.posit_subdir / f"{stem}_{descriptor}.{ext}"

    def backup_test_loss_distribution_path(self, data_bank_root: Path, timing_label: str,
                                           train_videos: int, test_videos: int,
                                           epochs: int, test_loss: float) -> Path:
        """Provenance-named backup of the test-loss distribution, alongside the
        canonical one. ``epochs`` carries the epoch this best occurred at for a
        new-best backup, or the total planned epochs for the finish backup (the
        two coexist), exactly like the estimator/checkpoint backups."""
        base = self.test_loss_distribution_pattern.format(
            project_alias=self.project_alias, timing_label=timing_label)
        stem, ext = base.rsplit(".", 1)
        descriptor = self.backup_descriptor(train_videos, test_videos, epochs, test_loss)
        return data_bank_root / self.posit_subdir / f"{stem}_{descriptor}.{ext}"

    def debug_run_dir(self, data_bank_root: Path, timing_label: str, stage: str,
                      split: Optional[str] = None) -> Path:
        """Process-level diagnostics directory for one ``--debug-dump`` run.

        Layout: ``<data_bank_root>/<labor_subdir>/<debug_subdir>/<run_label>/<stage>/``
        where ``run_label`` is ``<project_alias>_<timing_label>[_<split>]``. This one
        directory holds the whole-process ``console.log`` plus the structured
        diagnostics: ``report.md`` + ``figures/`` directly (Inference / Evaluation),
        or per-task ``TASK_<n>/`` subdirectories (RDS / DLI). Diagnostics live under
        ``Labor/`` (the workbench) so ``Posit/`` stays reserved for scientific
        deliverables (the posterior and the recovery reports).
        """
        run_label = f"{self.project_alias}_{timing_label}"
        if split:
            run_label = f"{run_label}_{split}"
        return data_bank_root / self.labor_subdir / self.debug_subdir / run_label / stage

    def map_recovery_dir(self, data_bank_root: Path, timing_label: str) -> Path:
        """Directory holding the MAP-recovery report, figures, and saved arrays.

        Sits under ``Posit/`` (alongside the posterior it evaluates) and is the
        self-contained output of the Evaluation stage: ``report.md``,
        ``figures/``, and the recovery ``.npz`` (true vs. inferred theta).
        """
        return data_bank_root / self.posit_subdir / self.recovery_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )

    def map_recovery_array_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the saved MAP-recovery array bundle (.npz)."""
        return self.map_recovery_dir(data_bank_root, timing_label) / (
            self.recovery_pattern.format(
                project_alias=self.project_alias, timing_label=timing_label,
            ) + ".npz"
        )

    def experiment_video_path(self, kind: str, cell: int, span_seconds: int,
                              data_bank_root: Path) -> Path:
        """Full path for a real microscopy video (read by the Experiment stage).

        Files live under ``<data_bank_root>/<experiment_subdir>/`` and use the
        native raw naming ``Experiment_{kind}_Cell_{cell}_{span}S_RAW.tif`` (the
        user copies them into the data bank to keep all data sources co-located).
        """
        filename = self.experiment_pattern.format(
            kind=kind, cell=cell, span=span_seconds)
        return data_bank_root / self.experiment_subdir / filename

    def experiment_recovery_dir(self, data_bank_root: Path, timing_label: str) -> Path:
        """Directory holding the Experiment-stage MAP report, figures, and arrays."""
        return data_bank_root / self.posit_subdir / self.experiment_recovery_pattern.format(
            project_alias=self.project_alias, timing_label=timing_label,
        )


@dataclass(frozen=True)
class SimulationStem:
    """ReaDDy simulation system geometry."""
    pixel_size_nm: int = 158                       # 157 + 1
    root_size_px: int = 256                        # 2^8 = 256
    crux_size_nm: float = 2.0                      # (7 + 3) / 5
    length_unit: str = "nanometer"
    time_unit: str = "nanosecond"
    particle_diameter_nm: int = 10
    # Fission products are placed at fission_product_distance_factor x the reaction distance
    # (2 x 10 nm = 20 nm), i.e. OUTSIDE the fusion radius. Declared convention (2026-09-09):
    # at the 2 ms sub-step the per-step reaction probability of an eligible pair is ~1 for
    # every D_A in range, so daughters placed AT the reaction distance would re-fuse at the
    # next step unless they diffused apart within one step -- ~3% for fast daughters but
    # ~50% per step for immobile ones, which would make the effective unbinding rate mode
    # dependent (immobile dimers effectively never dissociating). Starting the daughters
    # outside the radius removes that deterministic rebinding; diffusive re-encounter remains.
    # Both the reaction distance and this factor are declared conventions, not measurements.
    fission_product_distance_factor: int = 2

    @property
    def fission_product_distance_nm(self) -> float:
        """Distance (nm) at which the two fission daughters are placed: factor x reaction distance."""
        return float(self.fission_product_distance_factor * self.particle_diameter_nm)

    @property
    def box_size(self) -> tuple[float, float, float]:
        """3D box dimensions (x, y, z) in nanometers."""
        return (self.pixel_size_nm * self.root_size_px,
                self.pixel_size_nm * self.root_size_px,
                self.crux_size_nm)

    @property
    def unit_dict(self) -> dict[str, str]:
        """Unit dict for ReaDDy ReactionDiffusionSystem constructor."""
        return {"length_unit": self.length_unit, "time_unit": self.time_unit}


@dataclass(frozen=True)
class FrameConfig:
    """Fixed frame-rate parameters.

    These define the sampling cadence (camera frame rate and per-frame
    sub-stepping) and never vary between runs. The global
    ``PARAMETERS.simulation.timing`` holds only this fixed configuration; the
    per-run recording length lives on :class:`RunTiming`, constructed at each
    entry-point from the required ``--total-time-seconds``. Keeping the length
    off the global makes it impossible to read a per-run ``frame_count`` from a
    global default.
    """
    frame_time_seconds: float = 0.020              # 50 Hz frame rate
    steps_per_frame: int = 10

    @property
    def fps(self) -> int:
        """Integer frames per second (``1 / frame_time_seconds``)."""
        return int(round(1 / self.frame_time_seconds))

    @property
    def delta_time_nanoseconds(self) -> float:
        """Per-step time delta in nanoseconds (for ReaDDy).

        Depends only on the fixed cadence, so it lives here.
        """
        return (self.frame_time_seconds / self.steps_per_frame) * 1e9


@dataclass(frozen=True)
class RunTiming:
    """Per-run timing: the recording length plus everything derived from it.

    Built at each entry-point from the required ``--total-time-seconds`` and the
    fixed :class:`FrameConfig`; never stored on the global ``PARAMETERS``. The
    derived ``frame_count`` is ``total_time_seconds / frame_time_seconds`` (length
    times frame rate), so it is always a function of the actual run, never a
    global default.
    """
    total_time_seconds: float                      # PER-RUN recording length in seconds (no default)
    frames: FrameConfig = field(default_factory=FrameConfig)

    # --- Fixed-cadence passthroughs, so callers read ``timing.<x>`` uniformly. ---
    @property
    def frame_time_seconds(self) -> float:
        return self.frames.frame_time_seconds

    @property
    def steps_per_frame(self) -> int:
        return self.frames.steps_per_frame

    @property
    def delta_time_nanoseconds(self) -> float:
        return self.frames.delta_time_nanoseconds

    # --- Per-run derived quantities. ---
    @property
    def frame_count(self) -> int:
        """Total number of frames = ``total_time_seconds / frame_time_seconds``."""
        return int(self.total_time_seconds / self.frames.frame_time_seconds)

    @property
    def total_steps(self) -> int:
        """Total simulation steps."""
        return self.frames.steps_per_frame * self.frame_count

    @property
    def label(self) -> str:
        """Filename-safe identifier for the active run timing.

        Format: ``"{duration}S_{fps}FPS"`` where duration uses ``:g``
        formatting (no trailing zeros: ``2`` not ``2.0``, ``2.5`` not
        ``2.500``) and fps is ``int(round(1 / frame_time_seconds))``.
        Used by every Paths.* path-construction method to namespace output
        files by run timing, so a 2 s and 10 s run never collide on disk and
        any artifact viewed in isolation identifies the config that produced it.
        """
        duration = f"{self.total_time_seconds:g}"
        return f"{duration}S_{self.frames.fps}FPS"


# =============================================================================
# Model blocks: the declarative records the reaction-diffusion generator reads
# =============================================================================
#
# Three layers, separately declared, interacting at named points (the model specification's
# state-ownership table): the STOICHIOMETRY block owns association and dissociation, the
# MOBILITY block owns the diffusion modes and the switching between them, and the VISIBILITY
# block (labeling.py) owns occupancy and dye counts. The reaction-diffusion stage simulates
# PARTICLE TYPES = molecular species x mobility modes; six computational types represent two
# molecular species. Every operation the network needs -- type-specific diffusion, unimolecular
# conversion, fusion, fission -- is a ReaDDy primitive; the network itself is GENERATED from
# these records (simulation_rds_support.reaction_channels), never written by hand.

@dataclass(frozen=True)
class MolecularSpecies:
    """A molecular species of the stoichiometry layer: its name and its receptor-subunit count."""
    name: str
    subunits: int


@dataclass(frozen=True)
class ParticleType:
    """One computational particle type = (molecular species, mobility mode)."""
    name: str
    species: str
    mode: str
    subunits: int


@dataclass(frozen=True)
class StoichiometryBlock:
    """Monomer-dimer stoichiometry, A + A -> B (per-condition setting) and B -> A + A, and the
    parameter keys it reads.

    - ``count_total_key``: the conserved receptor-subunit total N_R = n_A + 2 n_B (a count).
    - ``composition_ratio_key``: the REQUESTED initial dimer-to-monomer ratio r = n_B / n_A
      (a log10 row on [-2, 2], i.e. one dimer per hundred monomers to a hundred dimers per
      monomer); realized as integers by ``realize_initial_composition``. The receptor fraction
      in dimers x_B = 2r / (1 + 2r) and the complex fraction f_B = r / (1 + r) are DERIVED
      quantities (``ratio_to_receptor_fraction``, ``ratio_to_complex_fraction``). The log ratio
      is symmetric about an even split (f_B = 1/2), so the prior is even-handed between
      mostly-monomer and mostly-dimer populations. Under a condition without association it
      is read as the composition PRESENT at the start of the recording, not one formed.
    - ``dissociation_rate_key``: kappa_OFF, the dimer unbinding rate (1/s), one for every
      dimer mode; dissociation is not governed by contact; inferred in every condition.
    The association intensity is NOT a learnable row. It is a declared per-condition constant
    (``ConditionSetting.association_ratio``, read through ``SimulationRDS.association_ratio_of``):
    lambda_on = R_ON * lambda_ref with the COMPATIBILITY NORMALIZATION lambda_ref = 6 D_A / r^2
    (units 1/s; D_A the monomer scale coefficient, r the reaction distance). lambda_ref is NOT
    a physical upper bound on association -- the diffusion-limited regime is the
    large-intensity limit of the spatial rule -- it is a declared reference that keeps the
    ratio dimensionless. The same lambda_on applies to every association channel (one per
    unordered pair of monomer modes); a ratio of exactly zero declares that the condition has
    NO association channel. Association needs an encounter within ``reaction distance`` (one
    particle diameter, the Smoluchowski contact distance) followed by the reaction; no
    degradation, internalization, or synthesis occurs within a recording, so N_R is conserved
    by construction.
    """
    monomer: MolecularSpecies = MolecularSpecies("A", 1)
    dimer: MolecularSpecies = MolecularSpecies("B", 2)
    count_total_key: str = "count_total"
    composition_ratio_key: str = "ratio_dimer_monomer_initial"
    dissociation_rate_key: str = "rate_dissociation"

    def __post_init__(self):
        if self.monomer.subunits != 1 or self.dimer.subunits != 2:
            raise ValueError("StoichiometryBlock: the monomer carries one subunit and the dimer two.")
        if self.monomer.name == self.dimer.name:
            raise ValueError("StoichiometryBlock: monomer and dimer need distinct names.")

    @property
    def species(self) -> tuple:
        """The molecular species in declaration order (monomer, dimer)."""
        return (self.monomer, self.dimer)

    @property
    def species_names(self) -> tuple:
        return tuple(s.name for s in self.species)

    @property
    def parameter_keys(self) -> tuple:
        return (self.count_total_key, self.composition_ratio_key, self.dissociation_rate_key)


@dataclass(frozen=True)
class MobilityBlock:
    """Mobility modes shared by both molecular species, and the sequential switching chain.

    Modes are values on the diffusion-coefficient axis, Brownian within a mode, with no
    mechanism of their own (a slow or immobile mode represents reduced motion, not the
    membrane structure that may cause it). The diffusion coefficient of particle type
    (species, mode) is

        D[species, mode] = D_A * species_factor(species) * mode_factor(mode)

    with ``diffusivity_key`` -> D_A (the monomer scale), ``mode_ratio_keys`` -> the per-mode
    factor (None for the reference mode, i.e. factor 1), and ``dimer_ratio_key`` -> R_B, the
    dimer's factor within a mode (0 < R_B <= 1: at R_B = 1 dimerization adds no slowdown).
    The modes are declared fastest first; the ordering R_immobile < R_slow <= 1 is enforced
    by the DISJOINT declared ranges of their keys (checked at import, see
    ``_validate_model_blocks``).

    ``switching`` lists the conversions (from_mode, to_mode, rate_key); the chain is
    sequential (only adjacent modes are connected: f <-> s <-> i, so k_fi = k_if = 0) and the
    four rates are SHARED across species -- the working hypothesis that switching is a
    membrane-environment process independent of stoichiometric state.

    ``inheritance`` names the rule for the mode of a newly formed dimer: ``"slower_parent"``
    (the declared working hypothesis; alternatives are comparators, not built here).
    Dissociation conserves the mode: both daughters keep the dimer's mode. The initial mode
    of every particle is drawn from the stationary law of the isolated switching chain.
    """
    modes: tuple = ("f", "s", "i")
    mode_ratio_keys: tuple = (None, "relative_diffusivity_slow", "relative_diffusivity_immobile")
    diffusivity_key: str = "diffusivity_alp"
    dimer_ratio_key: str = "relative_diffusivity_dimer"
    switching: tuple = (("f", "s", "rate_fast_slow"),
                        ("s", "f", "rate_slow_fast"),
                        ("s", "i", "rate_slow_immobile"),
                        ("i", "s", "rate_immobile_slow"))
    inheritance: str = "slower_parent"

    def __post_init__(self):
        if len(set(self.modes)) != len(self.modes) or not self.modes:
            raise ValueError(f"MobilityBlock: modes must be unique and non-empty (got {self.modes}).")
        if len(self.mode_ratio_keys) != len(self.modes) or self.mode_ratio_keys[0] is not None:
            raise ValueError("MobilityBlock: one ratio key per mode, None for the first (reference) mode.")
        seen = set()
        for frm, to, key in self.switching:
            if frm not in self.modes or to not in self.modes or frm == to:
                raise ValueError(f"MobilityBlock: switching {frm}->{to} names an unknown mode.")
            if abs(self.mode_index(frm) - self.mode_index(to)) != 1:
                raise ValueError(f"MobilityBlock: the chain is sequential; {frm}->{to} skips a mode.")
            if (frm, to) in seen:
                raise ValueError(f"MobilityBlock: duplicate switching channel {frm}->{to}.")
            seen.add((frm, to))
        if self.inheritance != "slower_parent":
            raise ValueError(f"MobilityBlock: inheritance rule {self.inheritance!r} is not implemented "
                             f"(the milestone builds 'slower_parent'; alternatives are comparators).")

    def mode_index(self, mode: str) -> int:
        return self.modes.index(mode)

    def slower(self, mode_1: str, mode_2: str) -> str:
        """The slower of two modes (modes are declared fastest first)."""
        return mode_1 if self.mode_index(mode_1) >= self.mode_index(mode_2) else mode_2

    def inherited_mode(self, mode_1: str, mode_2: str) -> str:
        """Mode of the dimer formed from monomers in ``mode_1`` and ``mode_2``."""
        if self.inheritance == "slower_parent":
            return self.slower(mode_1, mode_2)
        raise NotImplementedError(self.inheritance)

    @property
    def parameter_keys(self) -> tuple:
        keys = [self.diffusivity_key, self.dimer_ratio_key]
        keys += [k for k in self.mode_ratio_keys if k is not None]
        keys += [k for _, _, k in self.switching]
        return tuple(keys)


@dataclass(frozen=True)
class ConditionSetting:
    """The declared settings of one experimental condition: association ratio and probe occupancy.

    ``token`` is the stored condition token (``FAB`` = MET-FAB, ``INLB`` = MET-INLB; the one
    definition of the naming lives in ``experiment_support.CONDITION_DISPLAY`` and the
    labeling laws in ``labeling.BASELINE_LAW_OF_CONDITION``; import-time validation keeps
    the three registries on the same tokens). ``association_ratio`` is R_ON: the microscopic
    association intensity is lambda_on = R_ON * lambda_ref (lambda_ref = 6 D_A / r^2) for every
    association channel; EXACTLY zero means the condition has no association channel at all
    (a structural setting, never a small positive stand-in for a logarithmic scale).

    Decision of 2026-09-09 (model specification, decision log): the association ratio is not
    inferred in either condition -- it was not recoverable in the earlier workflow and there
    are no grounds to make its estimation a requirement of the model. MET-FAB: R_ON = 0, a
    working approximation that omits an unresolved within-recording association process
    (no activating ligand; basal association is what is omitted) and represents pre-existing
    dimers through the inferred initial composition; the readout is dimers PRESENT at the
    recording start. MET-INLB: R_ON = 1, the reference convention -- not a measured
    association rate and not a verified diffusion-limited regime; composition and unbinding
    estimates are conditional on it. Because the two settings produce different trajectories,
    each condition has its own RDS trajectory tier (``Paths.rds_alias``), shared by both
    workflows. Unbinding stays inferred in both conditions.

    ``occupancy`` is the probability that a receptor subunit carries a probe at all (the
    visibility layer's declared input; labeling.py composes it with the condition's dye-count
    law). It is a CONVENTION with a source, not a measurement, and it is provisional until the
    experimental collaborators answer the questions sent on 2026-09-11 (probe concentration in
    the imaging medium, occupancy within dimers, probe residence). MET-INLB declares 0.5 (the
    collaborators' statement: 5 nM against a 5 nM dissociation constant; the published uPAINT
    protocol reports 0.25 nM in the medium, unreconciled). MET-FAB has no published affinity,
    so its occupancy is DERIVED: the code stores a declared Fab/InlB VISIBILITY RATIO
    (``visibility_ratio`` = 0.5, a rounded convention over the measured Fab/InlB spot-density
    ratios of the deposited recordings, 0.38-0.48 depending on the window; Special_Analyses
    A9) and the anchor condition (``visibility_ratio_to`` = INLB), and
    ``SimulationRDS.occupancy_of`` resolves p_FAB = ratio x a_INLB / P_FAB(dye >= 1) =
    0.5 x 0.25 / 0.806 = 0.155, so a revised InlB anchor propagates. Exactly one of
    ``occupancy`` and ``visibility_ratio`` is set per condition. The code default of full
    occupancy is retired: uPAINT labels a sparse subset by design.
    """
    token: str
    association_ratio: float
    occupancy: Optional[float] = None
    visibility_ratio: Optional[float] = None
    visibility_ratio_to: Optional[str] = None

    def __post_init__(self):
        token = str(self.token)
        if not token or not token.isupper() or not token.isalnum():
            raise ValueError(f"ConditionSetting: token must be an uppercase alphanumeric word "
                             f"such as FAB or INLB (got {self.token!r}).")
        ratio = float(self.association_ratio)
        if not np.isfinite(ratio) or ratio < 0.0:
            raise ValueError(f"ConditionSetting {token}: association_ratio must be finite and "
                             f">= 0 (got {self.association_ratio!r}).")
        if 0.0 < ratio < 1e-3:
            raise ValueError(f"ConditionSetting {token}: association_ratio {ratio!r} is a disguised "
                             f"zero; switch association off with exactly 0.0 (no channels are "
                             f"generated), never with a small positive stand-in.")
        declared = self.occupancy is not None
        derived = self.visibility_ratio is not None
        if declared == derived:
            raise ValueError(f"ConditionSetting {token}: declare EXACTLY ONE of occupancy (a declared "
                             f"probability) or visibility_ratio (derived from an anchor condition).")
        if declared and not (np.isfinite(self.occupancy) and 0.0 < float(self.occupancy) <= 1.0):
            raise ValueError(f"ConditionSetting {token}: occupancy must lie in (0, 1] (got {self.occupancy!r}).")
        if derived:
            if not (np.isfinite(self.visibility_ratio) and float(self.visibility_ratio) > 0.0):
                raise ValueError(f"ConditionSetting {token}: visibility_ratio must be finite and > 0 "
                                 f"(got {self.visibility_ratio!r}).")
            if not self.visibility_ratio_to or self.visibility_ratio_to == token:
                raise ValueError(f"ConditionSetting {token}: a derived occupancy names a DIFFERENT anchor "
                                 f"condition in visibility_ratio_to (got {self.visibility_ratio_to!r}).")


# The declared per-condition settings (see ConditionSetting for the decisions and their sources).
CONDITION_SETTINGS: tuple = (
    # MET-FAB: no association channels; pre-existing dimers may dissociate. Occupancy DERIVED from the
    # declared Fab/InlB visibility ratio 0.5 and the InlB anchor (-> 0.155 under the baseline laws).
    ConditionSetting("FAB", 0.0, visibility_ratio=0.5, visibility_ratio_to="INLB"),
    # MET-INLB: lambda_on = lambda_ref, the reference convention. Occupancy 0.5 declared (provisional).
    ConditionSetting("INLB", 1.0, occupancy=0.5),
)


@dataclass(frozen=True)
class SimulationRDS:
    """RDS-stage runtime defaults + the model blocks and condition settings the generator reads.

    The particle types are DERIVED here (species x modes, species-major, modes fastest
    first), as are their subunit counts and the maps back to species and mode; nothing
    downstream lists them by hand. ``conditions`` declares the per-condition association
    ratio (``association_ratio_of``) and probe occupancy (``occupancy_of``, ``visibility_of``);
    the reaction network of a run is generated from the blocks AND the run's condition, and
    the DLI stage labels the tier under the condition's occupancy unless overridden.
    """
    prior_seed: Optional[int] = None               # None = OS-determined
    stoichiometry: StoichiometryBlock = field(default_factory=StoichiometryBlock)
    mobility: MobilityBlock = field(default_factory=MobilityBlock)
    conditions: tuple = CONDITION_SETTINGS

    @property
    def condition_tokens(self) -> tuple:
        """The stored condition tokens in declaration order."""
        return tuple(c.token for c in self.conditions)

    def condition_setting(self, condition: str) -> ConditionSetting:
        """The declared setting of ``condition``; KeyError names the known tokens."""
        for setting in self.conditions:
            if setting.token == condition:
                return setting
        raise KeyError(f"unknown condition {condition!r}; the model declares {self.condition_tokens}.")

    def association_ratio_of(self, condition: str) -> float:
        """R_ON of ``condition``: 0.0 = no association channels; > 0 scales lambda_ref."""
        return float(self.condition_setting(condition).association_ratio)

    @staticmethod
    def dye_probability_of(condition: str) -> float:
        """P(dye >= 1) for one BOUND probe under the condition's baseline labeling law
        (INLB Bernoulli 0.5 -> 0.5; FAB Poisson 1.64 -> 1 - exp(-1.64) = 0.806)."""
        from .labeling import resolve_labeling_law      # local import: labeling imports only experiment_support
        return float(resolve_labeling_law(condition)[1].visible_probability)

    def occupancy_of(self, condition: str) -> float:
        """Probe occupancy per subunit of ``condition``: the declared value, or the derived one
        p = visibility_ratio x visibility(anchor) / P(dye >= 1 | condition). Declared values are
        conventions with sources (ConditionSetting); the derivation is one step deep by
        construction (an anchor must itself be declared)."""
        setting = self.condition_setting(condition)
        if setting.occupancy is not None:
            return float(setting.occupancy)
        anchor = self.condition_setting(setting.visibility_ratio_to)
        if anchor.occupancy is None:
            raise ValueError(f"ConditionSetting {condition}: anchor {anchor.token} must declare its own "
                             f"occupancy (derivations are one step deep).")
        anchor_visibility = float(anchor.occupancy) * self.dye_probability_of(anchor.token)
        return float(setting.visibility_ratio) * anchor_visibility / self.dye_probability_of(condition)

    def visibility_of(self, condition: str) -> float:
        """Probability that a receptor subunit is visible: occupancy x P(dye >= 1)."""
        return self.occupancy_of(condition) * self.dye_probability_of(condition)

    def occupancy_source_of(self, condition: str) -> str:
        """'declared' or 'derived' (provenance for the labeling record)."""
        return "declared" if self.condition_setting(condition).occupancy is not None else "derived"

    @staticmethod
    def type_name(species: str, mode: str) -> str:
        return f"{species}_{mode}"

    @property
    def particle_types(self) -> tuple:
        """All particle types: one per (molecular species, mobility mode)."""
        return tuple(
            ParticleType(self.type_name(sp.name, mode), sp.name, mode, sp.subunits)
            for sp in self.stoichiometry.species for mode in self.mobility.modes)

    @property
    def particle_type_names(self) -> tuple:
        return tuple(pt.name for pt in self.particle_types)

    @property
    def subunit_counts_per_type(self) -> tuple:
        """Receptor subunits per particle type, aligned with ``particle_type_names``."""
        return tuple(pt.subunits for pt in self.particle_types)

    @property
    def species_of_type(self) -> dict:
        """Particle-type name -> molecular species name."""
        return {pt.name: pt.species for pt in self.particle_types}

    @property
    def mode_of_type(self) -> dict:
        """Particle-type name -> mobility mode."""
        return {pt.name: pt.mode for pt in self.particle_types}

    @property
    def molecular_species_names(self) -> tuple:
        return self.stoichiometry.species_names

    @property
    def monomer_type_names(self) -> tuple:
        return tuple(pt.name for pt in self.particle_types if pt.subunits == 1)

    # ReaDDy neighbor-list (Verlet) skin, expressed as a MULTIPLE of the particle
    # diameter: the actual skin distance is
    #     skin = neighbor_list_skin_factor * SimulationStem.particle_diameter_nm  (nm).
    # This is a PURE PERFORMANCE knob -- it changes only how ReaDDy searches for
    # nearby particles, never the physics. ReaDDy finds reaction partners with a
    # cell-linked list whose cell edge is (reaction_radius + skin); reactions still
    # fire at the true reaction radius regardless of the skin, so the output is
    # unchanged. In the large, dilute imaging box (~40 um across, ~1000 particles)
    # the reaction radius alone (= 1 diameter = 10 nm) forces a ~16-million-cell grid
    # that is >99.99% empty, and per-step management of that grid -- not the physics
    # -- dominates runtime. A skin of a few tens of nm coarsens the grid ~120x and
    # recovers ~13x wall-clock with statistically identical output (verified: same
    # species composition and reaction counts at the prior extremes, incl. max
    # diffusivity + fusion-on-contact). The cost is U-shaped in the skin: too small
    # -> empty-cell sweep; too large -> the box collapses toward ~1 cell and the
    # candidate search degrades to O(N^2). The optimum is a broad plateau (~10x-100x
    # the diameter). The default 10x (= 100 nm) sits on that plateau and clears the
    # worst-case per-step displacement (~47 nm at max diffusivity) with margin, so no
    # eligible pair within the reaction distance at a sub-step boundary is missed by the
    # neighbor search (encounters BETWEEN sub-step boundaries are a separate, documented
    # approximation of the 2 ms sub-step). Overridable per run via --skin-factor / SKIN_FACTOR.
    neighbor_list_skin_factor: float = 10.0


@dataclass(frozen=True)
class SimulationDLI:
    """DLI-stage runtime defaults."""
    sqrt_2sigma_dist_label: str = "lognormal"      # PSF width sampling distribution
    # darkcounts removed (W1): the pre-PSF photon floor is the SCOPE-drawn optical background kappa_o (REFERENCE_EMCCD_NOISE_MODEL.md sec. 5); dark current is handled inside EMCCD (dark_current_e_per_s=0).


@dataclass(frozen=True)
class Simulation:
    """Aggregator. Access via PARAMETERS.simulation.{stem, timing, rds, dli}."""
    stem: SimulationStem = field(default_factory=SimulationStem)
    timing: FrameConfig = field(default_factory=FrameConfig)   # global holds ONLY fixed cadence; per-run length -> RunTiming
    rds: SimulationRDS = field(default_factory=SimulationRDS)
    dli: SimulationDLI = field(default_factory=SimulationDLI)


@dataclass(frozen=True)
class InferenceTraining:
    """Inference training hyperparameters."""
    epochs: int = 5
    batch_size: int = 32
    learning_rate_minimum: float = 1.0e-5
    learning_rate_maximum_factor: int = 128        # 2^7
    scheduler_factor: float = 0.5                  # ReduceLROnPlateau gamma (per-epoch LR anneal step)
    scheduler_patience: int = 1
    scheduler_tolerance_factor: float = 10.0       # tolerance = lr_min * factor
    # Warm restart: after the LR decays to learning_rate_minimum and stays there
    # warm_restart_dwell epochs WITHOUT a new best, reload the best checkpoint and
    # restart the LR at the previous sawtooth peak * warm_restart_factor -- a decaying
    # in-run plateau-escape that self-terminates once the next peak would reach the
    # floor. warm_restart_dwell = 0 disables it. warm_restart_factor is deliberately a
    # SEPARATE knob from scheduler_factor (the per-epoch anneal step): it sets the
    # restart AMPLITUDE decay, keeping each restart a gentle probe of a converged model
    # (0.25 -> the first restart is a quarter of the peak) rather than a large jump
    # halfway back up. Persisted in the resurrect-state, so the sawtooth is continuous
    # across a --resurrect requeue.
    warm_restart_dwell: int = 2
    warm_restart_factor: float = 0.25              # restart-peak amplitude decay per warm restart (not the anneal step)
    augmentation: bool = True                      # rotation + horizontal/vertical flip
    # Dataset-sizing defaults for the three-namespace split (TRAIN / TEST / EVAL),
    # consumed by the generation orchestrator. CORE = TRAIN + TEST.
    test_fraction: float = 0.2                     # TEST as a fraction of CORE
    eval_fraction: float = 0.1                     # EVAL as a fraction of CORE
    core_min: int = 10                             # minimum CORE size (samples)
    eval_floor: int = 10                           # minimum EVAL size (samples); for a stable recovery number


@dataclass(frozen=True)
class InferenceNetwork:
    """Complex3DCNN architecture defaults.

    These values control the embedding network that maps videos to a latent
    embedding for the downstream density estimator (MAF). See
    `inference_network.py` for the architecture; brief notes on each field:

    - `n_conv_layers`: depth of the 3D-CNN backbone (channels double per layer).
    - `n_attn_layers`: number of stacked self-attention blocks in the temporal
      transformer that summarizes the conv-stack output across time.
    - `start_channels`: output channels of the first conv block (becomes
      `start_channels * 2^(n_conv_layers - 1)` at the deepest layer).
    - `temporal_target_frames`: the temporal length, in FRAMES, that a video is
      reduced toward before the transformer, to bound memory for long
      recordings. A video of `n_frames` frames is reduced by an integer factor
      `s = n_frames // temporal_target_frames`, folded into the first conv's
      temporal stride; videos with `n_frames <= temporal_target_frames` are left
      untouched (`s = 1`, bit-identical to the un-reduced network, so the 2 s
      baseline is unaffected). Because `n_frames = duration_seconds * frame_rate`,
      this frame count corresponds to a different physical duration at different
      frame rates: 100 frames is 2 s at 50 FPS, 1 s at 100 FPS, or 4 s at 25 FPS.
      The recommended value is ~100 frames -- a memory / temporal-resolution
      balance at the sequence level, independent of FPS; values far outside
      ~50-200 either over-compress the motion signal (too small) or let memory
      grow again (too large). Set to `None` to disable temporal reduction.

      The target is a FACTOR, not an exact output length. The reduced length is
      the standard conv output size, `T_out = (n_frames - kernel) // s + 1` with
      no padding once `s > 1`. The kernel starts at `max(3, s)` and is then
      widened to the smallest value congruent to `n_frames` modulo `s`, which
      makes `(n_frames - kernel) % s == 0` so the last window ends exactly on the
      last frame and every input frame is read; this is asserted at construction.
      Widening removes only the unusable remainder, so `T_out` is unaffected.
      Over the documented durations (VALIDATION.md, dataset-sizing table):

        | duration @50 FPS | n_frames |  s | kernel | T_out               |
        |------------------|----------|----|--------|---------------------|
        |  1 s             |       50 |  1 |      3 |  50  (no reduction) |
        |  2 s             |      100 |  1 |      3 | 100  (no reduction) |
        |  5 s             |      250 |  2 |      4 | 124                 |
        | 10 s             |      500 |  5 |      5 | 100                 |
        | 20 s             |     1000 | 10 |     10 | 100                 |

      5 s is the only documented duration whose remainder is non-zero, hence the
      only one whose kernel is widened (3 -> 4); the rest are unchanged by the
      rule. A consequence worth knowing if a campaign picks a non-standard
      duration: `s` is floor division, so a video shorter than `2 * target`
      frames is not reduced at all, and peak memory therefore does not increase
      monotonically with duration.

    The network returns the CLS-token embedding directly, which feeds the
    downstream MAF density estimator.
    """
    input_channels: int = 1
    n_conv_layers: int = 5
    n_attn_layers: int = 2
    start_channels: int = 8
    use_temporal_attention: bool = True
    attention_heads: int = 4
    temporal_target_frames: int = 100             # reduce longer videos toward ~this many frames
    #                                               (see class docstring); None disables reduction.


@dataclass(frozen=True)
class InferenceEvaluation:
    """MAP-recovery (Evaluation stage) defaults.

    The recovery procedure is a seed-then-optimize MAP estimate: draw a pool of
    candidate theta from the posterior, score each by the flow's log-probability,
    keep the top-`elite_prex_size` as optimization seeds, and gradient-ascent the
    log-probability with Adam + ReduceLROnPlateau and early stopping. Defaults are
    scaled for a high-memory GPU (the Evaluation stage runs on the GPU server);
    every value is overridable at the entry-point CLI.

    The `learning_rate` field is `learning_rate_minimum * learning_rate_maximum_factor`
    and the optimization `tolerance` is `learning_rate_minimum * tolerance_factor`,
    mirroring the training-stage convention.

    The `error_*` / `quantile_*` fields control the recovery-report figures:
    a per-parameter scatter of inferred-vs-true (log10) and a residual-error view,
    each with conditional-quantile bands drawn only where a bin holds at least
    `quantile_min_count` points (so the bands degrade gracefully for small EVAL).
    """
    pool_mode: str = "bounded"                     # candidate-pool sampler:
    #   "bounded"      -- DirectPosterior rejection sampling within prior ranges
    #                     (correct for a well-trained posterior).
    #   "unrestricted" -- sample the flow directly (no prior-range rejection);
    #                     never stalls, so it suits smoke tests and landscape
    #                     exploration on an undertrained posterior whose mass
    #                     lies outside the prior box.
    theta_prex_size: int = 1000                    # candidate pool size per video
    theta_prex_batch_size: int = 100               # sampling batch size
    score_prex_batch_size: int = 20                # log-prob scoring batch size
    elite_prex_size: int = 2                       # number of optimization seeds (top-K)
    numb_steps: int = 1000                         # max gradient-ascent steps
    optimizer_patience: int = 100                  # steps without improvement -> stop
    scheduler_patience: int = 10                   # steps without improvement -> reduce lr
    show_progress_steps: int = 100                 # progress-print cadence
    learning_rate_minimum: float = 1.0e-3
    learning_rate_factor: float = 0.5              # ReduceLROnPlateau gamma
    learning_rate_maximum_factor: int = 128        # 2^7; lr = lr_min * factor
    tolerance_factor: float = 1.0                  # tolerance = lr_min * factor
    # Recovery-report rendering:
    # Recovery tolerance bands, given as log10 half-widths -- each is the log10 of
    # a linear accuracy factor, so a point inside the band is recovered to within
    # that factor of the truth. Both are drawn as +/- guide lines on the error view
    # and reported as "fraction within" columns of the recovery table:
    #   error_guide       = 0.3  ~= log10(2)       -> within a factor of 2 of the truth
    #   error_guide_tight = 0.15 ~= log10(sqrt(2)) -> within a factor of sqrt(2) ~= 1.41
    #                              (0.15 = 0.3 / 2), a tighter concentration reference
    #                              nested inside the factor-2 band.
    error_guide: float = 0.3
    error_guide_tight: float = 0.15
    error_ylim_floor: float = 0.5                  # min half-range for the error y-axis (log10)
    error_ylim_quantile: float = 0.95              # |error| quantile setting the error y-axis
    quantile_bins: int = 20                        # conditional-quantile bins over true value
    quantile_min_count: int = 50                   # min points per bin to draw a band
    posterior_samples: int = 1000                  # draws/observation summarized by the quantiles, the median and the SGM


@dataclass(frozen=True)
class InferenceFlow:
    """Masked-autoregressive-flow (MAF) density-estimator settings.

    These are the `sbi.neural_nets.net_builders.build_maf` keyword arguments the Inference
    stage passes explicitly, so the flow's capacity is a recorded configuration rather than a
    library default, and they are persisted verbatim in the saved estimator's rebuild
    specification (`artifacts.save_estimator`, ``rebuild_spec.maf_args``). The defaults below
    are the library defaults the estimators trained before 0.1.13 used, so a run with no
    preset reproduces them exactly.

    - `hidden_features`: width of each autoregressive transform's hidden layers.
    - `num_transforms`: number of stacked autoregressive transforms.
    - `num_blocks`: residual blocks per transform.
    - `dropout_probability` / `use_batch_norm`: regularization inside the transforms.
    - `z_score_x` / `z_score_y`: standardization of parameters and of the embedding input
      ("structured" standardizes each dimension independently from the training batch).
    """
    hidden_features: int = 50
    num_transforms: int = 5
    num_blocks: int = 2
    dropout_probability: float = 0.1
    use_batch_norm: bool = True
    z_score_x: str = "structured"
    z_score_y: str = "structured"


# Named capacity presets for the Inference stage (`--network-preset`). A preset overrides fields
# of `InferenceNetwork` and `InferenceFlow` together; everything else (data, targets, splits,
# preprocessing, training protocol, standardization, batch normalization) is untouched. The
# baseline preset is the empty override and reproduces the pre-0.1.13 architecture.
#
#   capacity256 -- the combined capacity test of DETECTOR_WORKFLOW.md sec. 9.7: the embedding is
#   widened to 256 dimensions by doubling every convolutional block's channels
#   (start_channels 8 -> 16 with the same five blocks: 16 * 2^4 = 256 features per temporal
#   token, 64 per attention head), and the flow to hidden_features 128 / num_transforms 8 /
#   num_blocks 2 / dropout 0.1. A combined test: an improvement supports the larger
#   configuration but does not isolate the embedding from the flow.
NETWORK_PRESETS: dict = {
    "baseline": dict(network={}, flow={}),
    "capacity256": dict(network=dict(start_channels=16),
                        flow=dict(hidden_features=128, num_transforms=8, num_blocks=2,
                                  dropout_probability=0.1)),
}


@dataclass(frozen=True)
class Inference:
    """Aggregator. Access via PARAMETERS.inference.{training, network, flow, evaluation}."""
    training: InferenceTraining = field(default_factory=InferenceTraining)
    network: InferenceNetwork = field(default_factory=InferenceNetwork)
    flow: InferenceFlow = field(default_factory=InferenceFlow)
    evaluation: InferenceEvaluation = field(default_factory=InferenceEvaluation)


@dataclass(frozen=True)
class Plotting:
    """Plot defaults."""
    dpi: int = 500
    base_size: int = 10


# =============================================================================
# Top-level Parameters container
# =============================================================================

@dataclass(frozen=True)
class Parameters:
    """Top-level config aggregator. Access via the module-level PARAMETERS singleton."""
    machine: MachineProfile
    paths: Paths = field(default_factory=Paths)
    simulation: Simulation = field(default_factory=Simulation)
    inference: Inference = field(default_factory=Inference)
    plotting: Plotting = field(default_factory=Plotting)


# Module-level singleton. Instantiated at import time so configuration errors
# (missing env var, missing profile, invalid keys, missing directories) surface
# immediately on `import srm_and_sbi_monomer_dimer_alp.parameterization` rather than at
# the first read from PARAMETERS deep inside a simulation or training run.
PARAMETERS = Parameters(machine=load_machine_profile())


# =============================================================================
# Rich parameter spec
# =============================================================================
#
# Each parameter is a dict with the following fields:
#   {
#     'KEY':          unique parameter identifier (str),
#     'VALUE':        default / fixed value (scalar or list; for log-uniform
#                     priors this is the value at the center of the prior),
#     'PRIOR_RANGE':  (low, high) bounds for log-uniform priors; None for
#                     fixed (non-learnable) parameters,
#     'LOG_FLAG':     True if PRIOR_RANGE is given in log10 space,
#     'LOG_BASE':     base for the log transform (typically 10),
#     'UNIT':         human-readable unit of the parameter AS SAMPLED
#                     ('Dimensionless' for relative/ratio parameters),
#     'DERIVED_UNIT': for a dimensionless ratio that scales a physical
#                     quantity (e.g. R_B scales D_B), the unit of that derived
#                     quantity; None when the parameter is itself the physical
#                     quantity. Display-only documentation, never used in
#                     computation.
#     'LABEL':        LaTeX label for plotting,
#     'NOTE':         one of 'Learnable Parameter', 'Known Parameter',
#                     'Hyper Parameter' — distinguishes inferred parameters
#                     from fixed scientific constants and tuning hyperparameters.
#   }
#
# Top-level keys group parameters by domain:
#   RDS (Reaction-Diffusion System; molecular dynamics):
#     'count', 'diffusivity', 'dimerization_dissociation', 'immobilization_mobilization'
#   DLI (Diffraction-Limited Imaging; optical-detector model):
#     'camera', 'psf', 'transitivity'
#
# Notes on specific entries:
#   - Photobleaching is parameterized by ('prob_photo_bleach', 'numb_photo_bleach'):
#     prob_photo_bleach is the probability that an emitter enters the absorbing
#     bleached state over numb_photo_bleach camera frames. numb_photo_bleach = 100
#     is a FIXED reference window (= 2 s at 50 FPS), NOT the movie frame count --
#     do NOT set it to frame_count. prob_photo_bleach = 0.1 over this 100-frame
#     reference was inherited from detector-only inference on the raw experimental
#     videos. The per-frame rate is therefore constant across clip lengths:
#         p_1     = 1 - (1 - prob_photo_bleach) ** (1 / numb_photo_bleach)
#         p_video = 1 - (1 - prob_photo_bleach) ** (n_frames / numb_photo_bleach)
#     so a 500-frame (10 s) clip bleaches 1 - 0.9 ** 5 ~= 0.41 cumulatively, via
#     repeated application of the same per-frame transition matrix.
#   - 'delta_frame' VALUE is hardcoded to 0.020 (the default
#     RunTiming.frame_time_seconds). If a run overrides
#     `frame_time_seconds`, consumers should use the runtime value directly
#     rather than this default.
#   - 'capture_radius' is the Smoluchowski contact radius of two monomers
#     (2 * monomer_radius = 1 * diameter = particle_diameter_nm = 10 nm). The table
#     VALUE is display-only; build_system derives the active value from
#     PARAMETERS.simulation.stem.particle_diameter_nm, so it tracks per-dataset
#     geometry (particle_diameter_nm is the single physical input).

# Value-based parameter-role sentinels (DETECTOR_WORKFLOW.md sec. 5). A row whose
# VALUE is one of these strings is a role marker, not a concrete value: NUISANCE means
# "marginalized -- drawn per simulation, never inferred"; POSTERIOR means "drawn from a
# trained posterior". Defined above the parameter table so rows can carry them (the
# role dispatcher `role_of` that reads them lives further below, beside the filters).
NUISANCE_SENTINEL = "NUISANCE"
POSTERIOR_SENTINEL = "POSTERIOR"
_SENTINELS = (NUISANCE_SENTINEL, POSTERIOR_SENTINEL)


_PARAMETERIZATION_RAW_NESTED: dict[str, list[dict]] = {
    # ----- Reaction-Diffusion System: the two model blocks -----
    # DECIDED PRIOR RANGES (2026-09-14). Box-uniform in the estimator coordinate; each DOC names its
    # source (baseline = dimer-alp 0.4.23; A9 = Special_Analyses per-recording analysis of the
    # deposited MET tables; spec = model specification sec. 9 anchors; thresholds = tracking
    # pipelines). Rationale: PROJECT_CONTEXT.md sec. 2, "How the prior ranges and the declared
    # inputs are set". The receptor count is CONDITIONAL on the declared occupancies.
    'stoichiometry': [  # StoichiometryBlock: conserved total, requested initial dimer-to-monomer ratio, dissociation (the association ratio is a per-condition CONSTANT, not a row)
        {'KEY': 'count_total', 'VALUE': 10**3.0, 'PRIOR_RANGE': (2.5, 3.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'DERIVED_UNIT': None, 'LABEL': r'$N_{R}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Conserved receptor-subunit total N_R = n_A + 2 n_B of the SIMULATED patch (the in-field count is a distinct, time-dependent quantity under the open lateral boundary). Realized as an integer by realize_initial_composition. Range 316-3162 (A9): spots per frame in the first 2 s of the 120 deposited recordings, divided by the declared visibility per subunit (INLB 0.25, FAB 0.125), give log10 N_R peaked at 3.0-3.1 (sd 0.21 InlB, 0.30 Fab), 94% inside the box; localizations UNDERCOUNT receptors (bleached bound probes, missed detections, two-dye dimers as one spot), so the tail below 2.5 is empty in truth. The box is wider than the empirical shape on purpose: posterior width comes from the data, prior width steers the training budget. CONDITIONAL on the declared occupancies. Baseline 0.4.23: three per-species counts, each (0, 2.5).'},
        {'KEY': 'ratio_dimer_monomer_initial', 'VALUE': 10**0, 'PRIOR_RANGE': (-2.0, 2.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Count', 'LABEL': r'$r_{B/A}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'REQUESTED initial dimer-to-monomer ratio r = n_B(0) / n_A(0), log10 on [-2, 2]: from one dimer per hundred monomers to a hundred dimers per monomer, i.e. complex fraction f_B = r / (1 + r) from 1% to 99%, SYMMETRIC about an even split (median f_B = 1/2), so the prior is even-handed between mostly-monomer and mostly-dimer populations; the resting (5-18% complexes) and activated (63%) anchors lie inside. Realized as n_B(0) = min(round(N_R r / (1 + 2 r)), floor(N_R / 2)), n_A = N_R - 2 n_B; the receptor fraction x_B = 2 r / (1 + 2 r) and f_B are DERIVED and recorded beside the requested ratio (Labeling_Set). Rejected: log10 x_B on [-2, 0] (65% of the prior mass on f_B < 0.1 and 5% on f_B > 0.6, forcing the resting answer) and the linear x_B on [0, 1] (median f_B = 1/3). Replaces the earlier per-species counts of the baseline.'},
        {'KEY': 'capture_radius', 'VALUE': 10, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Nanometer', 'DERIVED_UNIT': None, 'LABEL': r'$\rho_{CAP}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'rate_dissociation', 'VALUE': 10**(-1), 'PRIOR_RANGE': (-3, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{OFF}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Dimer unbinding rate, B_m -> A_m + A_m for every mode m (dissociation conserves the mode); inferred in every condition. Range 0.001-10 per s: the lower bound is widened from the baseline (-1, 1) so that under MET-FAB, where no association exists, dimers can PERSIST through a 20 s recording (0.001 per s loses 2% in 20 s); the upper bound, a 0.1 s lifetime (five frames), is the baseline value. Under MET-FAB kappa_OFF is the dimer lifetime rate directly.'},
    ],
    'mobility': [  # MobilityBlock: monomer scale, dimer factor, mode factors, the four shared switching rates
        {'KEY': 'diffusivity_alp', 'VALUE': 10**(-0.75), 'PRIOR_RANGE': (-1.25, -0.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Square Micrometer Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$D_{A}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Monomer scale coefficient D_A = D[A, fast]; every other coefficient is a declared ratio of it. Range 0.056-0.56 um^2/s, the baseline 0.4.23 range unchanged; the fast-class anchors 0.13 (segment analysis) and 0.25 (hidden Markov analysis) of the specification sec. 9.3 lie inside.'},
        {'KEY': 'relative_diffusivity_dimer', 'VALUE': 10**(-0.5), 'PRIOR_RANGE': (-1, 0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Square Micrometer Per Second', 'LABEL': r'$R_{B}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Dimer factor within a mode: D[B, m] = R_B * D[A, m], 0 < R_B <= 1. Range 0.1-1: the baseline 0.4.23 range (-0.625, -0.125), i.e. 0.24-0.75 around the 1.65x-slower-dimer anchor, widened to 1 by the lead so that dimerization may add no slowdown; population-level slowing then comes from mode occupancy and inheritance.'},
        {'KEY': 'relative_diffusivity_slow', 'VALUE': 10**(-0.5), 'PRIOR_RANGE': (-1, 0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Square Micrometer Per Second', 'LABEL': r'$R_{s}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Slow-mode factor: D[X, s] = R_s * D[X, f]. Range 0.1-1 (a new row; the baseline had no slow mode): the confined/free anchors 0.7 (segment analysis) and 0.28 (hidden Markov analysis) lie inside. R_s = 1 does NOT reduce the model to two modes (the chain still passes through s); at 1 the slow mode coincides with the fast one, a degeneracy to report, not to remove. Disjoint from R_i by a full decade.'},
        {'KEY': 'relative_diffusivity_immobile', 'VALUE': 10**(-2.5), 'PRIOR_RANGE': (-3.0, -2.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Square Micrometer Per Second', 'LABEL': r'$R_{i}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Immobile-mode factor: D[X, i] = R_i * D[X, f]. Range 0.001-0.01. IMMOBILITY IS IMPOSED BY CONSTRUCTION: over nearly the whole D_A range the immobile coefficient stays below the tracking pipelines\' immobility thresholds (0.0028 and 0.0065 um^2/s; only the top of the range at the D_A ceiling exceeds the stricter one), so the third mode IS the class the pipelines call immobile, and the data decide its occupancy and exchange rates, not whether it is immobile. The lower half lies below the 2 s resolution floor (~0.0005-0.001 um^2/s), so the posterior of R_i is flat there; accepted. Baseline 0.4.23 (-2, -1) rejected: at 0.1 the mode reaches 0.018-0.056 um^2/s (classified confined) and touches R_s. A full decade below the slow range guarantees R_i < R_s.'},
        {'KEY': 'rate_fast_slow', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$k_{fs}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Switching fast -> slow, shared by both species. Range 0.1-10 per s: the baseline 0.4.23 band of the earlier mobile/immobile switching, kept for all four shared rates; resolvable between the 20 ms frame and the 2 s window; the observed class fractions (67/22/11 Fab, 43/29/28 InlB) imply rate ratios between 0.3 and 1, well inside.'},
        {'KEY': 'rate_slow_fast', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$k_{sf}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Switching slow -> fast, shared by both species. Range 0.1-10 per s: the baseline 0.4.23 band of the earlier mobile/immobile switching, kept for all four shared rates; resolvable between the 20 ms frame and the 2 s window; the observed class fractions (67/22/11 Fab, 43/29/28 InlB) imply rate ratios between 0.3 and 1, well inside.'},
        {'KEY': 'rate_slow_immobile', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$k_{si}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Switching slow -> immobile, shared by both species. Range 0.1-10 per s: the baseline 0.4.23 band of the earlier mobile/immobile switching, kept for all four shared rates; resolvable between the 20 ms frame and the 2 s window; the observed class fractions (67/22/11 Fab, 43/29/28 InlB) imply rate ratios between 0.3 and 1, well inside.'},
        {'KEY': 'rate_immobile_slow', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$k_{is}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Switching immobile -> slow, shared by both species. Range 0.1-10 per s: the baseline 0.4.23 band of the earlier mobile/immobile switching, kept for all four shared rates; resolvable between the 20 ms frame and the 2 s window; the observed class fractions (67/22/11 Fab, 43/29/28 InlB) imply rate ratios between 0.3 and 1, well inside.'},
    ],
    # ----- Diffraction-Limited Imaging -----
    'camera': [  # EMCCD camera chain (REFERENCE_EMCCD_NOISE_MODEL.md): gamma, kappa_o, kappa_b, kappa_s, kappa_q marginalized as the SCOPE camera nuisance (externally constrained, not jointly inferred; DETECTOR_WORKFLOW.md sec. 9.3, marginalized in both workflows); kappa_g, kappa_c fixed nominal spec metadata (gamma = kappa_g/kappa_c).
        {'KEY': 'gamma', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.62, 1.625), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU Per Electron', 'DERIVED_UNIT': None, 'LABEL': r'$\gamma$',
         'NOTE': 'Gain-conversion ratio gamma = kappa_g/kappa_c (ADU per photoelectron) -- the only gain quantity the videos identify. Marginalized as a SCOPE camera nuisance: inferring it splits the peak-ADU amplitude with mu_pc (only gamma*kappa_q is identifiable), so it is drawn from its a-priori box rather than treated as a calibration target (DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: kappa_g/kappa_c = 200/4.78 = 41.84, identical across all MET cells (both Fab and InlB camera protocols); box [41.7, 42.2].'},
        {'KEY': 'kappa_o', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.455, 1.465), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Photon', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{o}$',
         'NOTE': 'Optical background offset: incident photons per pixel per frame, pre-gain, one scalar per movie; amplified by gamma*kappa_q to set the ADU floor. Marginalized as a SCOPE camera nuisance (externally constrained, not jointly inferred; DETECTOR_WORKFLOW.md sec. 9.3). Reference = ThunderSTORM offset[photon] median, a condition-independent background: Fab 28.9 / InlB 28.6 (pooled 28.7). Narrow box [28.5, 29.2] around the measured value: a broad offset lets the gain*offset floor dominate the video-to-video variation (embedding collapse).'},
        {'KEY': 'kappa_b', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (2.24, 2.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{b}$',
         'NOTE': 'Camera baseline: post-gain ADU constant, added last. Marginalized as a SCOPE camera nuisance (externally constrained, not jointly inferred; DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: MET configured baseline = 175, identical across all cells (both conditions); box [173.8, 177.8].'},
        {'KEY': 'kappa_s', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.02, 1.025), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{s}$',
         'NOTE': 'Read noise: post-register Gaussian sigma (ADU). Marginalized as a SCOPE camera nuisance: it does not dominate the SNR (the EM register amplifies signal above the read-noise floor) and is only weakly identifiable (DETECTOR_WORKFLOW.md sec. 9.3). Reference = camera datasheet ~10.5 ADU; box [10.5, 10.6] (weakly identifiable but pinned tight to the datasheet value, like the other camera constants).'},
        {'KEY': 'kappa_q', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-0.05, -0.04), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{q}$',
         'NOTE': 'Quantum efficiency, applied once in the Poisson step. Marginalized as a SCOPE camera nuisance: only the product gamma*kappa_q is identifiable from the videos (DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: MET quantumEfficiency = 0.90, identical across all cells; box [0.89, 0.91].'},
        {'KEY': 'kappa_g', 'VALUE': 200, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{g}$',
         'NOTE': 'Nominal EM gain g from the MET acquisition config (ThunderSTORM camera protocol; audit-pending). Not inferred -- kept as spec metadata so the drawn gamma (SCOPE nuisance) can be checked against kappa_g/kappa_c (drift check).'},
        {'KEY': 'kappa_c', 'VALUE': 4.78, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Electron Per ADU', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{c}$',
         'NOTE': 'Nominal conversion C (e-/ADU) from the MET acquisition config (photons2ADU field; audit-pending). Not inferred -- kept as spec metadata for the gamma drift check. gamma = kappa_g / kappa_c.'},
    ],
    'psf': [  # Point Spread Function (lowercased for casing consistency)
        {'KEY': 'mu_r', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\mu_{r}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**0.15.'},
        {'KEY': 'sigma_r', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\sigma_{r}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**-0.625.'},
        {'KEY': 'mu_pc', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\mu_{pc}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**2.375.'},
        {'KEY': 'sigma_pc', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\sigma_{pc}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**-0.375.'},
    ],
    'transitivity': [  # Brightness photo-physics (stationary OU ln-brightness flicker + absorbing bleach)
        {'KEY': 'delta_frame', 'VALUE': 0.020, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Second', 'DERIVED_UNIT': None, 'LABEL': r'$\delta_{f}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'prob_photo_bleach', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\rho_{pb}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**-1.25.'},
        {'KEY': 'numb_photo_bleach', 'VALUE': 100, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\psi_{pb}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'lambda_rate', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\lambda$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Correlation-decay rate of the stationary OU ln-brightness flicker: ACF(lag) = exp(-lambda_rate * lag), so tau_corr = 1/lambda_rate (generate_brightness_photons). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**0.5.'},
    ],
}

# (Model-block validation runs below, once the learnable subset exists: _validate_model_blocks.)


# Flat list (raw): all parameters in declaration order, regardless of group
PARAMETERIZATION_RAW: list[dict] = [
    para
    for group in _PARAMETERIZATION_RAW_NESTED.values()
    for para in group
]

# Index map: KEY -> position in PARAMETERIZATION_RAW (for full-spec access)
PARAMETER_RAW_FIND: dict[str, int] = {
    para['KEY']: index for index, para in enumerate(PARAMETERIZATION_RAW)
}


# ---- Value-based parameter roles ---------------------------------------------
# The role of each parameter is read from its (VALUE, PRIOR_RANGE) cell, so a block
# can be held Fixed, inferred as a Posterior (learnable), or marginalized as a
# Nuisance without structural change. A concrete VALUE + a range is learnable; a
# concrete VALUE + no range is fixed; the sentinels (defined above the table) mark the
# nuisance/posterior roles. Ported from the Detector workflow (DETECTOR_WORKFLOW.md
# sec. 5). The whole imaging block is marginalized in production (DETECTOR_WORKFLOW.md
# sec. 9.3, Phase D): the camera five (gamma, kappa_o, kappa_b, kappa_s, kappa_q) are the
# SCOPE camera nuisance (nuisance_spec, drawn per simulation from their a-priori boxes),
# and the six photophysics (mu_r, sigma_r, mu_pc, sigma_pc, prob_photo_bleach, lambda_rate)
# are the calibrated-imaging nuisance (nuisance_object, drawn per simulation from the
# persisted Nuisance_DLI artifact). The remaining imaging rows (kappa_g, kappa_c,
# delta_frame, numb_photo_bleach) resolve to 'fixed', so
# the learnable subset is exactly the 10 RDS parameters.


def role_of(entry: dict) -> str:
    """Value-based role of one parameter entry.

    Returns 'learnable', 'fixed', 'nuisance_spec', 'nuisance_object', or 'posterior'.
    Dispatch is sentinel-based and never tests whether VALUE is numeric, so a
    list-valued fixed parameter is classified correctly.
    The learnable subset is VALUE-not-a-sentinel AND PRIOR_RANGE-not-None
    (DETECTOR_WORKFLOW.md sec. 5, constraint 2): a nuisance-from-spec row also carries
    a range, so a bare PRIOR_RANGE test would wrongly pull it into the inference prior.
    POSTERIOR with a range is undefined -- a posterior draw carries its own support.
    """
    value, prior_range = entry['VALUE'], entry['PRIOR_RANGE']
    is_sentinel = isinstance(value, str) and value in _SENTINELS
    if value == POSTERIOR_SENTINEL:
        if prior_range is not None:
            raise ValueError(
                f"parameter {entry['KEY']!r}: POSTERIOR with a PRIOR_RANGE is undefined "
                f"(a posterior draw carries its own support); set PRIOR_RANGE=None.")
        return 'posterior'
    if value == NUISANCE_SENTINEL:
        return 'nuisance_spec' if prior_range is not None else 'nuisance_object'
    if not is_sentinel and prior_range is not None:
        return 'learnable'
    return 'fixed'


# Filtered list (learnable subset): VALUE-not-a-sentinel AND PRIOR_RANGE-not-None.
PARAMETERIZATION: list[dict] = [
    para for para in PARAMETERIZATION_RAW if role_of(para) == 'learnable'
]

# Index map: KEY -> position in PARAMETERIZATION (for theta-vector indexing)
PARAMETER_FIND: dict[str, int] = {
    para['KEY']: index for index, para in enumerate(PARAMETERIZATION)
}

# Ordered learnable-parameter keys = the theta-vector schema. Passed to
# artifacts.load_estimator(expected_parameter_keys=...) / assert_schema_compatible to
# hard-reject an estimator whose parameter schema differs (equal-length theta vectors
# would otherwise be misread column-for-column). Mirrors
# detector_parameterization.DETECTOR_PARAMETER_KEYS.
PARAMETER_KEYS: list[str] = [para['KEY'] for para in PARAMETERIZATION]


# =============================================================================
# The one parameter-conversion rule (estimator space <-> physical values)
# =============================================================================

def is_log_row(entry: dict) -> bool:
    """True for a ranged row whose estimator coordinate is log10 of the physical value."""
    return bool(entry.get('LOG_FLAG'))


def to_physical(theta_flow, parameterization=None):
    """Map estimator-space coordinates to physical values, row by row.

    ``theta_flow`` has the learnable parameters on its LAST axis, in the order of
    ``parameterization`` (default: ``PARAMETERIZATION``). Log rows are exponentiated
    (``10 ** u``); linear rows pass through. Returns a float64 array of the same shape.
    This is the only sanctioned conversion; a blanket ``10 ** theta`` corrupts the linear
    initial dimer fraction.
    """
    table = PARAMETERIZATION if parameterization is None else parameterization
    arr = np.array(theta_flow, dtype=float, copy=True)
    if arr.shape[-1] != len(table):
        raise ValueError(f"to_physical: last axis has {arr.shape[-1]} entries; the table has {len(table)}.")
    for j, entry in enumerate(table):
        if is_log_row(entry):
            arr[..., j] = np.power(10.0, arr[..., j])
    return arr


def to_flow(theta_physical, parameterization=None):
    """Map physical values to estimator-space coordinates (inverse of ``to_physical``).

    Log rows become ``log10`` (a non-positive physical value on a log row is an error, not a
    NaN, because it can only come from a corrupted table or a mis-scaled input); linear rows
    pass through. Returns a float64 array of the same shape.
    """
    table = PARAMETERIZATION if parameterization is None else parameterization
    arr = np.array(theta_physical, dtype=float, copy=True)
    if arr.shape[-1] != len(table):
        raise ValueError(f"to_flow: last axis has {arr.shape[-1]} entries; the table has {len(table)}.")
    for j, entry in enumerate(table):
        if is_log_row(entry):
            col = arr[..., j]
            if np.any(col <= 0):
                raise ValueError(f"to_flow: parameter {entry['KEY']!r} is a log row but received a "
                                 f"non-positive physical value.")
            arr[..., j] = np.log10(col)
    return arr


def entry_to_physical(entry: dict, value_flow):
    """Scalar (or array) conversion of ONE row's estimator coordinate to its physical value."""
    value_flow = np.asarray(value_flow, dtype=float)
    return np.power(10.0, value_flow) if is_log_row(entry) else value_flow


def entry_to_flow(entry: dict, value_physical):
    """Scalar (or array) conversion of ONE row's physical value to its estimator coordinate."""
    value_physical = np.asarray(value_physical, dtype=float)
    return np.log10(value_physical) if is_log_row(entry) else value_physical


def prior_center(entry: dict) -> float:
    """Physical value at the center of a ranged row's prior box (its nominal)."""
    lo, hi = entry['PRIOR_RANGE']
    return float(entry_to_physical(entry, (lo + hi) / 2.0))


# =============================================================================
# Initial composition: integer realization of the sampled (N_R, r), r = n_B / n_A
# =============================================================================

@dataclass(frozen=True)
class InitialComposition:
    """The realized initial state of the stoichiometry layer for one simulation.

    ``n_total`` is the integer receptor-subunit total actually placed (N_R rounded, at least
    one); ``n_dimers`` and ``n_monomers`` the particles placed; ``ratio_requested`` the
    sampled dimer-to-monomer ratio r; ``fraction_requested`` = 2r / (1 + 2r) the receptor
    fraction it implies; ``fraction_realized`` = 2 n_dimers / n_total, which differs from it by
    rounding and by the cap n_dimers <= floor(n_total / 2). Every derived fraction is
    computed from the realized counts.
    """
    n_total: int
    n_monomers: int
    n_dimers: int
    ratio_requested: float
    fraction_requested: float
    fraction_realized: float

    @property
    def n_complexes(self) -> int:
        return self.n_monomers + self.n_dimers

    @property
    def complex_fraction(self) -> float:
        """f_B = n_B / (n_A + n_B), the fraction of complexes that are dimers."""
        return self.n_dimers / self.n_complexes if self.n_complexes else float("nan")

    @property
    def ratio_realized(self) -> float:
        """r = n_B / n_A of the realized counts (inf when no monomer was placed)."""
        return self.n_dimers / self.n_monomers if self.n_monomers else float("inf")


def ratio_to_receptor_fraction(ratio) -> np.ndarray:
    """x_B = 2r / (1 + 2r): the receptor fraction in dimers implied by the ratio r = n_B / n_A."""
    r = np.asarray(ratio, dtype=float)
    return 2.0 * r / (1.0 + 2.0 * r)


def ratio_to_complex_fraction(ratio) -> np.ndarray:
    """f_B = r / (1 + r): the fraction of complexes that are dimers implied by r = n_B / n_A."""
    r = np.asarray(ratio, dtype=float)
    return r / (1.0 + r)


def receptor_fraction_to_ratio(fraction) -> np.ndarray:
    """r = x_B / (2 (1 - x_B)): the inverse of ``ratio_to_receptor_fraction`` (inf at x_B = 1)."""
    x = np.asarray(fraction, dtype=float)
    with np.errstate(divide="ignore"):
        return x / (2.0 * (1.0 - x))


def realize_initial_composition(count_total: float, ratio_dimer_monomer: float) -> InitialComposition:
    """Integer (n_A, n_B) from the sampled receptor total and requested dimer-to-monomer ratio.

    n_total = max(1, round(N_R));  n_B = min(round(n_total * r / (1 + 2 r)), floor(n_total / 2));
    n_A = n_total - 2 n_B.  Conservation N_R = n_A + 2 n_B holds exactly for the realized
    integers. At r = 0.01 and n_total = 316 three dimers are placed (the decided box never
    realizes zero dimers at its count floor); the cap binds only for r beyond the box.
    """
    if not np.isfinite(count_total) or not np.isfinite(ratio_dimer_monomer):
        raise ValueError(f"realize_initial_composition: non-finite input ({count_total}, {ratio_dimer_monomer}).")
    if ratio_dimer_monomer < 0.0:
        raise ValueError(f"realize_initial_composition: the dimer-to-monomer ratio must be >= 0 "
                         f"(got {ratio_dimer_monomer}).")
    r = float(ratio_dimer_monomer)
    n_total = int(max(1, round(float(count_total))))
    n_dimers = int(min(round(n_total * r / (1.0 + 2.0 * r)), n_total // 2))
    n_monomers = n_total - 2 * n_dimers
    return InitialComposition(n_total, n_monomers, n_dimers, r, float(ratio_to_receptor_fraction(r)),
                              2.0 * n_dimers / n_total)


# =============================================================================
# Model-block validation (import time, fail fast)
# =============================================================================

def _validate_model_blocks() -> None:
    rds = PARAMETERS.simulation.rds
    learnable = set(PARAMETER_KEYS)
    for key in rds.stoichiometry.parameter_keys + rds.mobility.parameter_keys:
        if key not in learnable:
            raise ValueError(f"model block references parameter {key!r}, which is not a learnable row.")
    for entry in PARAMETERIZATION:
        lo, hi = entry['PRIOR_RANGE']
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"parameter {entry['KEY']!r}: invalid PRIOR_RANGE {entry['PRIOR_RANGE']}.")
        if entry['LOG_FLAG'] not in (True, False):
            raise ValueError(f"parameter {entry['KEY']!r}: a ranged row declares LOG_FLAG True or False.")
        if is_log_row(entry) and entry['LOG_BASE'] != 10:
            raise ValueError(f"parameter {entry['KEY']!r}: log rows are base 10.")
    ratio = PARAMETERIZATION[PARAMETER_FIND[rds.stoichiometry.composition_ratio_key]]
    if not is_log_row(ratio):
        raise ValueError("the initial dimer-to-monomer ratio is a LOG row (symmetric about an even split).")
    if abs(ratio['PRIOR_RANGE'][0] + ratio['PRIOR_RANGE'][1]) > 1e-12:
        raise ValueError("the initial dimer-to-monomer ratio's box must be symmetric about 0 (an even split).")
    count = PARAMETERIZATION[PARAMETER_FIND[rds.stoichiometry.count_total_key]]
    n_floor = int(round(entry_to_physical(count, count['PRIOR_RANGE'][0])))
    if realize_initial_composition(n_floor, entry_to_physical(ratio, ratio['PRIOR_RANGE'][0])).n_dimers < 1:
        raise ValueError("the composition box realizes zero dimers at the count floor; widen the count "
                         "floor or raise the ratio's lower edge.")
    # Mobility ordering R_immobile < R_slow <= 1 and 0 < R_dimer <= 1, by the declared ranges.
    def phys_range(key):
        e = PARAMETERIZATION[PARAMETER_FIND[key]]
        return float(entry_to_physical(e, e['PRIOR_RANGE'][0])), float(entry_to_physical(e, e['PRIOR_RANGE'][1]))
    ratio_keys = [k for k in rds.mobility.mode_ratio_keys if k is not None]
    previous_low = 1.0 + 1e-12
    for key in ratio_keys:                      # slow, then immobile: each range strictly below the last
        lo, hi = phys_range(key)
        if not (0.0 < lo and hi <= 1.0):
            raise ValueError(f"{key}: mode ratio range must lie in (0, 1] (got [{lo}, {hi}]).")
        if hi >= previous_low:
            raise ValueError(f"{key}: its range [{lo}, {hi}] must lie strictly below the previous mode's "
                             f"lower edge {previous_low} so the mode ordering holds for every draw.")
        previous_low = lo
    lo, hi = phys_range(rds.mobility.dimer_ratio_key)
    if not (0.0 < lo and hi <= 1.0):
        raise ValueError(f"{rds.mobility.dimer_ratio_key}: range must lie in (0, 1] (got [{lo}, {hi}]).")
    # Condition settings: one per stored condition token, on the tokens the whole codebase names
    # conditions by (experiment_support is the one definition; a local import keeps the module
    # graph lean and cycle-free). The ratio's own validity is ConditionSetting.__post_init__.
    from .experiment_support import CONDITION_DISPLAY
    tokens = rds.condition_tokens
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"SimulationRDS.conditions: a condition token repeats in {tokens}.")
    if set(tokens) != set(CONDITION_DISPLAY):
        raise ValueError(f"SimulationRDS.conditions {tokens} must declare exactly the conditions of "
                         f"experiment_support.CONDITION_DISPLAY {tuple(CONDITION_DISPLAY)}.")
    if not any(rds.association_ratio_of(t) > 0.0 for t in tokens):
        raise ValueError("SimulationRDS.conditions: no condition has association switched on; the "
                         "model family is A + A -> B in at least one condition.")
    # Occupancy: declared or derived, every value a probability in (0, 1]; the derived MET-FAB
    # value guards the anchor arithmetic (0.5 x 0.25 / 0.806 = 0.155 under the baseline laws).
    for t in tokens:
        p = rds.occupancy_of(t)
        if not (np.isfinite(p) and 0.0 < p <= 1.0):
            raise ValueError(f"SimulationRDS.conditions: occupancy of {t} resolves to {p!r}, not a probability.")
        if rds.condition_setting(t).visibility_ratio_to is not None and \
                rds.condition_setting(rds.condition_setting(t).visibility_ratio_to).occupancy is None:
            raise ValueError(f"SimulationRDS.conditions: {t} derives its occupancy from an anchor that is itself derived.")
    if abs(rds.occupancy_of("FAB") - 0.155) > 5e-4 or abs(rds.occupancy_of("INLB") - 0.5) > 1e-12:
        raise ValueError(f"SimulationRDS.conditions: occupancies resolve to FAB {rds.occupancy_of('FAB'):.4f} / "
                         f"INLB {rds.occupancy_of('INLB'):.4f}; the decided values are 0.155 (derived) / 0.5. "
                         f"Change the declared settings AND this guard together.")


_validate_model_blocks()


def association_ratio_of(condition: str) -> float:
    """The declared association ratio R_ON of ``condition`` (``SimulationRDS.association_ratio_of``):
    0.0 means the condition has no association channel; a positive value scales lambda_ref."""
    return PARAMETERS.simulation.rds.association_ratio_of(condition)


def occupancy_of(condition: str) -> float:
    """Probe occupancy per subunit of ``condition`` (``SimulationRDS.occupancy_of``): INLB 0.5
    declared; FAB 0.155 derived from the declared Fab/InlB visibility ratio and the InlB anchor."""
    return PARAMETERS.simulation.rds.occupancy_of(condition)


def occupancy_source_of(condition: str) -> str:
    """'declared' or 'derived': how the condition's occupancy is set (``SimulationRDS.occupancy_source_of``)."""
    return PARAMETERS.simulation.rds.occupancy_source_of(condition)


def visibility_of(condition: str) -> float:
    """Probability that a receptor subunit is visible under ``condition``: occupancy x P(dye >= 1)
    (INLB 0.25, FAB 0.125 under the baseline labeling laws)."""
    return PARAMETERS.simulation.rds.visibility_of(condition)


# =============================================================================
# Helpers
# =============================================================================

def parameter_find(key: str) -> int:
    """Return the index of a learnable parameter by KEY.

    Use this to index into theta vectors (which contain only learnable parameters).
    For all-parameter access (including 'Known' / 'Hyper'), use PARAMETER_RAW_FIND.

    Raises KeyError with the list of learnable parameters if the key is not found.
    """
    if key not in PARAMETER_FIND:
        raise KeyError(
            f"Parameter {key!r} is not a learnable parameter. "
            f"Learnable parameters: {list(PARAMETER_FIND.keys())}."
        )
    return PARAMETER_FIND[key]


def theta_lower_bound() -> list[float]:
    """Lower bounds of the prior box in estimator space (log10 for log rows, the value for linear rows)."""
    return [para['PRIOR_RANGE'][0] for para in PARAMETERIZATION]


def theta_upper_bound() -> list[float]:
    """Upper bounds of the prior box in estimator space (log10 for log rows, the value for linear rows)."""
    return [para['PRIOR_RANGE'][1] for para in PARAMETERIZATION]


def build_prior(device: str = "cpu") -> BoxUniform:
    """Construct the BoxUniform prior over the learnable parameters, in estimator space.

    Sampled theta live in estimator space: log10 for log rows (a log-uniform prior on the
    physical value) and the value itself for linear rows (a uniform prior). Consumers map
    to physical values with ``to_physical`` -- never with a blanket ``10 ** theta``.
    """
    return BoxUniform(
        low=torch.tensor(theta_lower_bound()),
        high=torch.tensor(theta_upper_bound()),
        device=device,
    )
