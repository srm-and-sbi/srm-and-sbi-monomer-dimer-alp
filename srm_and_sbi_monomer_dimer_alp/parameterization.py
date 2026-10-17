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
    PARAMETERIZATION_RAW        -- flat list of all parameter entries (full spec, the master)
    MASTER_PARAMETERIZATION     -- the twelve learnable row definitions (never a theta schema by itself)
    parameterization_of(cond)   -- the condition's learnable table: eleven rows under FAB, twelve under INLB
    parameterization_raw_of(cond) -- the full spec with the condition's declared rows as fixed rows
    parameter_keys_of(cond)     -- the condition's theta schema (ordered keys); parameter_index_of(cond)
    parameter_find(key, cond)   -- index lookup in the condition's theta vector
    declared_row_values(cond)   -- the master rows the condition declares as constants (FAB: allocation 0)
    PARAMETER_RAW_FIND          -- dict[KEY -> index in PARAMETERIZATION_RAW]
    build_prior(cond, device)   -- the BoxUniform prior of the condition's table (estimator space)
    theta_lower_bound(cond)     -- lower bounds of the condition's prior box (estimator space)
    theta_upper_bound(cond)     -- upper bounds of the condition's prior box (estimator space)
    to_physical(theta_flow, table) -- the ONE conversion estimator space -> physical values
    to_flow(theta_physical, table) -- its inverse (physical -> estimator space)
    fraction_flags(theta, table) -- per linear row, the draws outside [0, 1] (flagged, never clipped)
    prior_center(entry)         -- physical value at the center of a ranged row
    realize_population          -- the window-start composition of many draws (reference, allocation, start)
    population_from_theta       -- the same from physical theta vectors, aligned by key
    realize_initial_composition -- the one-draw record the generator places from
    population_valid            -- the one validity rule of a window-start input
    THETA_STANDARDIZATION       -- the biology flow's theta standardization ("independent")
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
log-transforms a theta vector by hand. The decided table (2026-09-14, revised 2026-10-15)
carries two LINEAR rows, the basal reference fraction and the InlB window-start allocation,
beside ten log rows; the per-row rule is the contract every consumer applies.

Prior ranges. Every range is a DECIDED box (2026-09-14; PROJECT_CONTEXT.md sec. 2, "How the
prior ranges and the declared inputs are set"): box-uniform in the estimator coordinate,
covering the plausible support with a margin and never encoding the answer an experiment is
expected to give; both conditions draw from the one MASTER table, each under its own view
(``parameterization_of``: MET-FAB declares the allocation 0 and infers eleven coordinates,
MET-INLB infers all twelve). Each row's DOC names its source:
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
    constant (``SimulationRDS.conditions``), so the trajectory tier and its ``Theta_Set``
    (eleven reaction-diffusion labels under MET-FAB, twelve under MET-INLB) are generated once
    PER CONDITION under the sibling alias plus the condition token (``rds_alias``, e.g.
    ``SRM_AND_SBI_MONOMER_DIMER_ALP_FAB``),
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
    # ``Theta_Set`` (eleven labels under MET-FAB, twelve under MET-INLB) -- carry this alias
    # plus the condition token and never the workflow qualifier, because both workflows
    # re-image the same tier (``rds_alias`` composes it).
    sibling_alias: str = "SRM_AND_SBI_MONOMER_DIMER_ALP"
    # The condition token these Paths carry (None = no condition applied yet; see the class
    # docstring). Set only through ``with_condition``, which also appends the token to
    # ``project_alias``; ``rds_alias`` composes the tier alias from it.
    condition: Optional[str] = None
    # Whether this workflow's ``Theta_Set`` is an RDS product (biology: the condition's
    # reaction-diffusion labels -- the tier's own record, under ``rds_alias``) or a DLI
    # product (detector: the six imaging labels, drawn when the videos are rendered, hence
    # qualified and per condition).
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
        """The alias of the condition's RDS tier (trajectories and the condition's
        ``Theta_Set``, eleven labels under MET-FAB and twelve under MET-INLB): the bare sibling
        alias plus the condition token, whatever workflow qualifier these Paths carry. Requires
        a condition (``with_condition``): the tier is per condition because the association
        setting is."""
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
        product (biology: the condition's reaction-diffusion labels), the qualified and
        conditioned alias for a DLI product (detector: the six imaging labels)."""
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

    def rds_theta_set_path(self, task_alias: int, data_bank_root: Path,
                           timing_label: str, compress: bool = True,
                           split: str = "TRAIN") -> Path:
        """Full path of the condition's RDS-tier ``Theta_Set`` (the biology labels under
        ``rds_alias``, the namespace ``trajectory_dir`` resolves), for ANY workflow: the
        biology reads its own labels here, and the detector checks the UPSTREAM record of the
        trajectories it re-images -- never its own six-column imaging ``Theta_Set``, which
        ``theta_set_path`` resolves under the DETECTOR alias."""
        ext = self.compressed_ext if compress else self.uncompressed_ext
        filename = self.theta_set_pattern.format(
            project_alias=self.rds_alias,
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
    # Fission daughters are placed fission_product_distance_factor x the reaction distance apart
    # (4 x 10 nm = 40 nm: four times the sum of the two daughters' radii). Declared convention
    # (2026-09-09; the distance revised from 2 x on 2026-10-13). Why apart at all: at the 2 ms
    # sub-step an eligible pair reacts with probability ~1, so daughters placed AT the reaction
    # distance would re-fuse at the next step unless they diffused apart (~50% per step for
    # immobile daughters), making the effective unbinding mode dependent. Why 40 nm: ReaDDy
    # orients the daughter axis uniformly in three dimensions inside the 2 nm periodic slab, so
    # the in-plane separation is d sin(theta) and a fraction 1 - sqrt(1 - (r/d)^2) of the pairs
    # still lies within the reaction distance after the z wrap -- 13% at 20 nm, 3% at 40 nm; the
    # 3% residual is accepted in exchange for a daughter displacement from the parent (d/2 =
    # 20 nm) that stays below the 29 nm localization precision of the recordings. No slab
    # thickness removes the residual: a slab thin enough for planar association wraps z, and
    # one thick enough to keep the daughters apart lets z enter the association condition. The
    # reaction distance and this factor are declared conventions, not measurements; the
    # structure audit states the arithmetic (D7) and measures the geometry (R8).
    fission_product_distance_factor: int = 4

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
    """A molecular species of the stoichiometry layer.

    ``name`` is the species token (``A``, ``B``, ``B1``, ``B2``); ``subunits`` the number of
    RETAINED receptor subunits a particle of this species carries (the subunits the lineage
    follows); ``stoichiometry`` the stoichiometric class, ``monomer`` or ``dimer``;
    ``dissociation`` the ReaDDy primitive that dissociates it (``fission`` into two monomers,
    ``conversion`` into one monomer when the other subunit is not represented, None for a
    monomer); ``bound_subunits`` the number of ligand-bound subunits the class declares (None
    where the probe classes are assigned at labeling rather than by species).
    """
    name: str
    subunits: int
    stoichiometry: str = "monomer"
    dissociation: Optional[str] = None
    bound_subunits: Optional[int] = None

    def __post_init__(self):
        if self.stoichiometry not in ("monomer", "dimer"):
            raise ValueError(f"MolecularSpecies {self.name}: stoichiometry must be monomer or dimer.")
        if self.dissociation not in (None, "fission", "conversion"):
            raise ValueError(f"MolecularSpecies {self.name}: dissociation must be fission, conversion or None.")
        if (self.stoichiometry == "monomer") != (self.dissociation is None):
            raise ValueError(f"MolecularSpecies {self.name}: exactly the dimers dissociate.")
        if self.dissociation == "fission" and self.subunits != 2:
            raise ValueError(f"MolecularSpecies {self.name}: a fission needs two retained subunits.")
        if self.dissociation == "conversion" and self.subunits != 1:
            raise ValueError(f"MolecularSpecies {self.name}: a conversion carries one retained subunit.")


@dataclass(frozen=True)
class ParticleType:
    """One computational particle type = (molecular species, mobility mode).

    ``species`` is the STOICHIOMETRIC class (``A`` monomer, ``B`` dimer), which every consumer of
    the lineage counts by; ``molecule`` the molecular species token the type is built from
    (``A``, ``B``, ``B1`` or ``B2``); ``subunits`` the retained subunits it carries.
    """
    name: str
    species: str
    mode: str
    subunits: int
    molecule: str = ""


@dataclass(frozen=True)
class StoichiometryBlock:
    """Monomer-dimer stoichiometry of the RETAINED population, and the parameter keys it reads.

    The simulated population is the probe-associated one (PROJECT_CONTEXT.md sec. 2, *Modeling
    assumptions of the MET model*): receptors without a probe neither react nor emit under the
    model's eligibility rule, so they are not simulated, except that the probe-free partner
    within a retained Fab dimer stays represented (both daughters of ``B`` are retained) while
    the probe-free partner of a one-InlB dimer is never represented. Two molecular species describe a
    condition without ligand classes (MET-FAB): the monomer ``A`` (one subunit) and the dimer
    ``B`` (two retained subunits, both daughters retained on fission; its one- or two-probe
    class is assigned at labeling). A condition with ligand classes (MET-INLB, where only
    ligand-bound monomers associate) uses the monomer ``A`` and two dimer classes: ``B1``, one
    InlB-bound subunit, whose dissociation is written as the CONVERSION B1 -> A (the probe-free
    partner is not represented, so no lineage entry is created or removed), and ``B2``, two
    InlB-bound subunits, formed by association and dissociating by fission into two eligible
    monomers. ``species_for(ligand_classes)`` selects the inventory of a condition; the
    stoichiometric class of every type is ``A`` or ``B``.

    - ``count_total_key``: N_total, the receptor-subunit total of the underlying population,
      probe-bound and unbound (the inferred count, shared by both conditions). The represented
      count N_R the simulation conserves is realized from it through the condition's occupancy
      (``realize_population``): N_R = A_ref + 2 D_sim under FAB and A_ref + B1 + 2 B2_ref, the
      InlB-bound subunits, under INLB.
    - ``basal_fraction_key``: f, the basal REFERENCE fraction of dimers among the complexes of
      the underlying population, a LINEAR row uniform on [0, 1] (``FRACTION_BOX``): M_ref =
      N_total (1 - f) / (1 + f) monomers and D_ref = N_total f / (1 + f) dimers. Under a
      condition without association it is the fraction PRESENT at the start of the recording,
      not one formed; under MET-INLB the starting fraction f_start also carries the allocation
      and is a DERIVED quantity reported beside f.
    - ``active_allocation_key``: R_active, the initial active-dimer allocation of the
      window-start composition, a LINEAR row uniform on [0, 1], inferred under MET-INLB and
      declared 0 under MET-FAB (``ConditionSetting.relative_count_active_dimers_initial``):
      c = round(R_active cap) of the cap = A_ref // 2 probe-bound monomer pairs of the reference
      become additional two-InlB dimers before the recording starts, conserving N_R. It is
      capacity-relative, not the active share c / D_start, which is derived.
    - ``dissociation_rate_key``: kappa_OFF, the dimer unbinding rate (1/s), one for every
      dimer class and mode; dissociation is not governed by contact; inferred in every condition.
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
    monomer: MolecularSpecies = MolecularSpecies("A", 1, "monomer")
    dimer: MolecularSpecies = MolecularSpecies("B", 2, "dimer", "fission")
    dimer_one_probe: MolecularSpecies = MolecularSpecies("B1", 1, "dimer", "conversion", 1)
    dimer_two_probe: MolecularSpecies = MolecularSpecies("B2", 2, "dimer", "fission", 2)
    count_total_key: str = "count_total"
    basal_fraction_key: str = "fraction_basal_dimers_reference"
    dissociation_rate_key: str = "rate_dissociation"
    active_allocation_key: str = "relative_count_active_dimers_initial"

    def __post_init__(self):
        if self.monomer.stoichiometry != "monomer" or self.monomer.subunits != 1:
            raise ValueError("StoichiometryBlock: the monomer carries one subunit.")
        for d in (self.dimer, self.dimer_one_probe, self.dimer_two_probe):
            if d.stoichiometry != "dimer":
                raise ValueError(f"StoichiometryBlock: {d.name} must be a dimer class.")
        if self.dimer.dissociation != "fission" or self.dimer_two_probe.dissociation != "fission":
            raise ValueError("StoichiometryBlock: B and B2 dissociate by fission into two monomers.")
        if self.dimer_one_probe.dissociation != "conversion" or self.dimer_one_probe.bound_subunits != 1:
            raise ValueError("StoichiometryBlock: B1 carries one bound subunit and dissociates by conversion.")
        if self.dimer_two_probe.bound_subunits != 2:
            raise ValueError("StoichiometryBlock: B2 carries two bound subunits.")
        names = [s.name for s in self.species]
        if len(set(names)) != len(names):
            raise ValueError(f"StoichiometryBlock: species names must be distinct (got {names}).")

    @property
    def species(self) -> tuple:
        """Every molecular species of the model family, in declaration order (A, B, B1, B2)."""
        return (self.monomer, self.dimer, self.dimer_one_probe, self.dimer_two_probe)

    def species_for(self, ligand_classes: bool) -> tuple:
        """The species a condition simulates: (A, B) without ligand classes, (A, B1, B2) with them."""
        if ligand_classes:
            return (self.monomer, self.dimer_one_probe, self.dimer_two_probe)
        return (self.monomer, self.dimer)

    def dimers_for(self, ligand_classes: bool) -> tuple:
        return tuple(s for s in self.species_for(ligand_classes) if s.stoichiometry == "dimer")

    @property
    def species_names(self) -> tuple:
        """The STOICHIOMETRIC class names, (monomer, dimer): the axis every census and the
        labeling record count by, whatever the condition's ligand classes."""
        return (self.monomer.name, self.dimer.name)

    @property
    def parameter_keys(self) -> tuple:
        """The block's master rows in master order: count, basal fraction, dissociation, allocation."""
        return (self.count_total_key, self.basal_fraction_key, self.dissociation_rate_key,
                self.active_allocation_key)


@dataclass(frozen=True)
class MobilityBlock:
    """Mobility modes shared by all molecular species of a condition, and the sequential switching chain.

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

    ``ligand_classes`` declares whether the condition's dimers carry a ligand class (B1 / B2,
    ``StoichiometryBlock``). A condition with association requires them: only ligand-bound
    monomers associate, a one-ligand dimer releases one eligible monomer and a two-ligand dimer
    two, so the retained network must distinguish the two (MET-INLB). Without association the
    probe classes have no dynamical consequence and are assigned at labeling (MET-FAB).

    The probe OCCUPANCY per receptor subunit, p, is a declared conversion between the inferred
    receptor total N_total of the true population and the retained population the generator
    simulates (``SimulationRDS.occupancy_of``; PROJECT_CONTEXT.md sec. 2, *Modeling assumptions
    of the MET model*, items 4 and 14): the expected retained count is p N_total under INLB and
    p N_total [1 + 2 (1 - p) f_B / (1 + f_B)] under FAB, and p also sets the one-/two-probe split
    2 (1 - p) : p of the initial dimers. It enters the generator at initialization only; every
    retained subunit of a monomer carries a probe. It is read from the independent-site
    EQUILIBRIUM at the published probe concentration (uPAINT, 0.25 nM for both probes in the
    imaging medium, acquisition about three minutes after addition when the authors report the
    bound-to-unbound ratio as balanced; Harwardt et al. 2017): p = L / (L + K_D), with the
    in-vitro K_D of InlB321, about 5 nM (``dissociation_constant_nm``), giving p_INLB = 0.0476
    (``occupancy_source_of`` = ``equilibrium``). The Fab has no published affinity, so its
    occupancy is DERIVED from a declared Fab/InlB VISIBILITY RATIO (``visibility_ratio`` = 0.5,
    a rounded convention over the measured Fab/InlB spot-density ratios of the deposited
    recordings, 0.38-0.48 depending on the window; Special_Analyses A9) and the anchor condition
    (``visibility_ratio_to`` = INLB): p_FAB = ratio x a_INLB / P_FAB(dye >= 1) =
    0.5 x 0.0238 / 0.806 = 0.0148, so a revised anchor propagates. A literal ``occupancy`` may
    be declared instead for a sensitivity scenario. Exactly one of ``dissociation_constant_nm``,
    ``visibility_ratio`` and ``occupancy`` is set per condition. The equilibrium reading is a
    declared initialization, not a measurement: the acquisition delay and the unknown rates
    are recorded as its caveats, and fixed attachment within a recording is its own assumption.

    ``relative_count_active_dimers_initial`` is the condition's setting of the window-start
    allocation row (``StoichiometryBlock.active_allocation_key``): None means the row is
    INFERRED (MET-INLB, twelve coordinates); a value in [0, 1] DECLARES it a constant, removes
    the row from the condition's table and realizes every simulation with it (MET-FAB declares
    0: no association, so the reference population is the starting population and eleven
    coordinates remain). An inferred or nonzero allocation requires ligand classes, because
    the pairs it forms are two-InlB dimers. A declared nonzero value under ligand classes is a
    sensitivity variant, not a decided setting.
    """
    token: str
    association_ratio: float
    ligand_classes: bool = False
    probe_concentration_nm: float = 0.25
    dissociation_constant_nm: Optional[float] = None
    occupancy: Optional[float] = None
    visibility_ratio: Optional[float] = None
    visibility_ratio_to: Optional[str] = None
    relative_count_active_dimers_initial: Optional[float] = None

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
        if ratio > 0.0 and not self.ligand_classes:
            raise ValueError(f"ConditionSetting {token}: a condition with association needs ligand "
                             f"classes (only ligand-bound monomers associate; B1 -> A, B2 -> A + A).")
        if not (np.isfinite(self.probe_concentration_nm) and float(self.probe_concentration_nm) > 0.0):
            raise ValueError(f"ConditionSetting {token}: probe_concentration_nm must be finite and > 0 "
                             f"(got {self.probe_concentration_nm!r}).")
        sources = [self.dissociation_constant_nm is not None, self.visibility_ratio is not None,
                   self.occupancy is not None]
        if sum(sources) != 1:
            raise ValueError(f"ConditionSetting {token}: declare EXACTLY ONE of dissociation_constant_nm "
                             f"(equilibrium occupancy), visibility_ratio (derived from an anchor "
                             f"condition) or occupancy (a literal probability).")
        if self.dissociation_constant_nm is not None and not (
                np.isfinite(self.dissociation_constant_nm) and float(self.dissociation_constant_nm) > 0.0):
            raise ValueError(f"ConditionSetting {token}: dissociation_constant_nm must be finite and > 0 "
                             f"(got {self.dissociation_constant_nm!r}).")
        if self.occupancy is not None and not (np.isfinite(self.occupancy) and 0.0 < float(self.occupancy) <= 1.0):
            raise ValueError(f"ConditionSetting {token}: occupancy must lie in (0, 1] (got {self.occupancy!r}).")
        if self.visibility_ratio is not None:
            if not (np.isfinite(self.visibility_ratio) and float(self.visibility_ratio) > 0.0):
                raise ValueError(f"ConditionSetting {token}: visibility_ratio must be finite and > 0 "
                                 f"(got {self.visibility_ratio!r}).")
            if not self.visibility_ratio_to or self.visibility_ratio_to == token:
                raise ValueError(f"ConditionSetting {token}: a derived occupancy names a DIFFERENT anchor "
                                 f"condition in visibility_ratio_to (got {self.visibility_ratio_to!r}).")
        elif self.visibility_ratio_to is not None:
            raise ValueError(f"ConditionSetting {token}: visibility_ratio_to is set without a visibility_ratio.")
        allocation = self.relative_count_active_dimers_initial
        if allocation is not None and not (np.isfinite(allocation) and 0.0 <= float(allocation) <= 1.0):
            raise ValueError(f"ConditionSetting {token}: a declared relative_count_active_dimers_initial lies in "
                             f"[0, 1] (got {allocation!r}); None leaves the row inferred.")
        if not self.ligand_classes and (allocation is None or float(allocation) != 0.0):
            raise ValueError(f"ConditionSetting {token}: an inferred or nonzero initial active-dimer allocation "
                             f"requires ligand classes (the allocated pairs are two-InlB dimers); a condition "
                             f"without them declares relative_count_active_dimers_initial=0.0.")

    @property
    def declared_rows(self) -> dict:
        """The master rows this setting declares as constants, KEY -> value."""
        if self.relative_count_active_dimers_initial is None:
            return {}
        return {StoichiometryBlock.active_allocation_key: float(self.relative_count_active_dimers_initial)}

    @property
    def equilibrium_occupancy(self) -> Optional[float]:
        """p = L / (L + K_D) when the condition declares a dissociation constant, else None."""
        if self.dissociation_constant_nm is None:
            return None
        L, kd = float(self.probe_concentration_nm), float(self.dissociation_constant_nm)
        return L / (L + kd)


# The declared per-condition settings (see ConditionSetting for the decisions and their sources).
CONDITION_SETTINGS: tuple = (
    # MET-FAB: no association channels; the retained basal dimers (at least one Fab) may dissociate
    # and both daughters stay retained. Occupancy DERIVED from the declared Fab/InlB visibility
    # ratio 0.5 and the InlB anchor (-> 0.0148 under the baseline laws); used at initialization only.
    # The window-start allocation is DECLARED 0 (no active dimers form before the window), so the
    # reference population is the starting population and the Fab table has eleven coordinates.
    ConditionSetting("FAB", 0.0, visibility_ratio=0.5, visibility_ratio_to="INLB",
                     relative_count_active_dimers_initial=0.0),
    # MET-INLB: lambda_on = lambda_ref, the reference convention; ligand classes B1 / B2. Occupancy =
    # the independent-site equilibrium at 0.25 nM with K_D 5 nM (-> 0.0476); initialization only.
    # The window-start allocation is INFERRED (None): twelve coordinates.
    ConditionSetting("INLB", 1.0, ligand_classes=True, dissociation_constant_nm=5.0),
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

    def _own_occupancy(self, setting: ConditionSetting) -> Optional[float]:
        """The occupancy a setting fixes by itself (equilibrium or literal); None when derived."""
        if setting.dissociation_constant_nm is not None:
            return float(setting.equilibrium_occupancy)
        if setting.occupancy is not None:
            return float(setting.occupancy)
        return None

    def occupancy_of(self, condition: str) -> float:
        """Probe occupancy per subunit of the TRUE population under ``condition``: the equilibrium
        value L / (L + K_D) (INLB), a literal, or the derived one p = visibility_ratio x
        visibility(anchor) / P(dye >= 1 | condition) (FAB). It converts the inferred N_total to the
        retained population and sets the probe classes of the initial dimers; the derivation is
        one step deep by construction (an anchor must fix its own occupancy)."""
        setting = self.condition_setting(condition)
        own = self._own_occupancy(setting)
        if own is not None:
            return own
        anchor = self.condition_setting(setting.visibility_ratio_to)
        anchor_occupancy = self._own_occupancy(anchor)
        if anchor_occupancy is None:
            raise ValueError(f"ConditionSetting {condition}: anchor {anchor.token} must fix its own "
                             f"occupancy (derivations are one step deep).")
        anchor_visibility = anchor_occupancy * self.dye_probability_of(anchor.token)
        return float(setting.visibility_ratio) * anchor_visibility / self.dye_probability_of(condition)

    def visibility_of(self, condition: str) -> float:
        """Probability that a subunit of the TRUE population is visible: occupancy x P(dye >= 1).
        The per-recording spot counts of the deposited recordings divided by this value are the
        implied N_total (Special_Analyses A9, re-read for the retained-population model)."""
        return self.occupancy_of(condition) * self.dye_probability_of(condition)

    def two_probe_share_of(self, condition: str) -> float:
        """The declared independent-binding share of two-probe dimers among the REFERENCE
        dimers, s_2 = p / (2 - p): the two-Fab share Fab labeling draws with (FAB) and the B2
        share of the reference dimers (INLB), before the window-start allocation."""
        p = self.occupancy_of(condition)
        return p / (2.0 - p)

    def declared_row_values(self, condition: str) -> dict:
        """The master rows ``condition`` declares as constants, KEY -> value
        (``ConditionSetting.declared_rows``): FAB {allocation: 0.0}, INLB {}."""
        return dict(self.condition_setting(condition).declared_rows)

    def occupancy_source_of(self, condition: str) -> str:
        """'equilibrium', 'declared' or 'derived' (provenance for the labeling record)."""
        setting = self.condition_setting(condition)
        if setting.dissociation_constant_nm is not None:
            return "equilibrium"
        return "declared" if setting.occupancy is not None else "derived"

    @staticmethod
    def type_name(species: str, mode: str) -> str:
        return f"{species}_{mode}"

    def _types_of(self, species: tuple) -> tuple:
        sto = self.stoichiometry
        return tuple(
            ParticleType(self.type_name(sp.name, mode),
                         sto.monomer.name if sp.stoichiometry == "monomer" else sto.dimer.name,
                         mode, sp.subunits, sp.name)
            for sp in species for mode in self.mobility.modes)

    @property
    def particle_types(self) -> tuple:
        """Every particle type of the model family: one per (molecular species, mobility mode),
        over all four species (A, B, B1, B2); a condition registers ``particle_types_for``."""
        return self._types_of(self.stoichiometry.species)

    def particle_types_for(self, condition: str) -> tuple:
        """The particle types a condition simulates: six without ligand classes, nine with them."""
        return self._types_of(self.species_for(condition))

    def species_for(self, condition: str) -> tuple:
        """The molecular species of ``condition``: (A, B), or (A, B1, B2) with ligand classes."""
        return self.stoichiometry.species_for(self.condition_setting(condition).ligand_classes)

    def particle_type_names_for(self, condition: str) -> tuple:
        return tuple(pt.name for pt in self.particle_types_for(condition))

    @property
    def particle_type_names(self) -> tuple:
        return tuple(pt.name for pt in self.particle_types)

    @property
    def subunit_counts_per_type(self) -> tuple:
        """Retained receptor subunits per particle type, aligned with ``particle_type_names``."""
        return tuple(pt.subunits for pt in self.particle_types)

    @property
    def species_of_type(self) -> dict:
        """Particle-type name -> STOICHIOMETRIC class (``A`` monomer or ``B`` dimer)."""
        return {pt.name: pt.species for pt in self.particle_types}

    @property
    def molecule_of_type(self) -> dict:
        """Particle-type name -> molecular species token (``A``, ``B``, ``B1``, ``B2``)."""
        return {pt.name: pt.molecule for pt in self.particle_types}

    @property
    def mode_of_type(self) -> dict:
        """Particle-type name -> mobility mode."""
        return {pt.name: pt.mode for pt in self.particle_types}

    @property
    def molecular_species_names(self) -> tuple:
        """The stoichiometric class names (monomer, dimer) every census counts by."""
        return self.stoichiometry.species_names

    @property
    def monomer_type_names(self) -> tuple:
        return tuple(pt.name for pt in self.particle_types if pt.species == self.stoichiometry.monomer.name)

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

    - `first_spatial_kernel`, `extra_spatial_convs`, `extra_spatial_conv_blocks`,
      `spatial_pooling`: the encoder-screening settings of DETECTOR_WORKFLOW.md
      (the section on the encoder screening). Their defaults (3, 0, 2, `mean`)
      are the original network, module for module; each screening preset in
      `NETWORK_PRESETS` changes exactly one of them. `first_spatial_kernel` is
      the first block's spatial kernel (odd, at least 3); `extra_spatial_convs`
      adds that many spatial-only (1, 3, 3) convolutions, each with batch
      normalization and Mish, before the pooling of the first
      `extra_spatial_conv_blocks` blocks; `spatial_pooling` reduces the final
      feature maps over space by their `mean` or by their `stats` (mean,
      standard deviation and maximum, projected back to the channel width).

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
    first_spatial_kernel: int = 3                 # first block's spatial kernel (odd, >= 3)
    extra_spatial_convs: int = 0                  # spatial-only convs added before early pooling
    extra_spatial_conv_blocks: int = 2            # ... in this many leading blocks
    spatial_pooling: str = "mean"                 # "mean" | "stats" (inference_network.SPATIAL_POOLINGS)


@dataclass(frozen=True)
class InferenceEvaluation:
    """MAP-recovery defaults, shared by every stage that computes the MAP.

    The recovery procedure is a seed-then-optimize MAP estimate: draw a pool of
    candidate theta from the posterior, score each by the flow's log-probability,
    keep the top-`elite_prex_size` as optimization seeds, and gradient-ascent the
    log-probability with Adam + ReduceLROnPlateau and early stopping. Defaults are
    scaled for a high-memory GPU (the Evaluation stage runs on the GPU server).
    The entry points override the pool size, the seed count, the step budget, the
    initial learning rate and the summary draw count from the CLI; the patience
    settings, the learning-rate floor and factor, the tolerance and the sampler batch
    size come from here only. A stage prints the optimizer values it used and records
    them in its product (`evaluation.optimizer_contract`).

    The ascent moves each coordinate in units of its spread over the observation's own
    candidate pool, `u = (theta - m) / IQR` with `m` and `IQR` the pool's per-coordinate
    median and interquartile range (`evaluation.pool_scale`). `learning_rate` and
    `learning_rate_minimum` are therefore dimensionless: 0.05 is a step of about 5 % of
    each parameter's posterior spread. `tolerance` is in nats of log-density and decides
    only what counts as a meaningful improvement, for the stopping patience and the
    plateau scheduler; every strictly better finite (score, vector) pair is retained
    whatever its size.

    The `error_*` / `quantile_*` fields control the recovery-report figures:
    a per-parameter scatter of inferred-vs-true (estimator coordinates: log10 for log rows,
    the value for the linear fraction rows, whose errors are absolute) and a residual-error view,
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
    theta_prex_batch_size: int = 10000             # max draws per outer sampler call (configuration only)
    score_prex_batch_size: int = 20                # log-prob scoring batch size
    elite_prex_size: int = 2                       # number of optimization seeds (top-K)
    numb_steps: int = 2000                         # max gradient-ascent steps
    optimizer_patience: int = 200                  # steps without a meaningful improvement -> stop
    scheduler_patience: int = 20                   # steps without a meaningful improvement -> reduce lr
    show_progress_steps: int = 100                 # progress-print cadence
    learning_rate: float = 0.05                    # initial Adam step, in pool-IQR units
    learning_rate_minimum: float = 5.0e-4          # plateau-schedule floor, in pool-IQR units
    learning_rate_factor: float = 0.5              # ReduceLROnPlateau gamma
    tolerance: float = 1.0e-3                      # nats: a meaningful improvement of the best score
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
    posterior_samples: int = 10000                 # draws/observation summarized by the quantiles, the median and the SGM;
    #                                                the exact SGM's CPU memory is quadratic: about 1.5 GiB per concurrent
    #                                                worker at 10,000 (the stage scripts request --mem=480G per node)


@dataclass(frozen=True)
class InferenceFlow:
    """Masked-autoregressive-flow (MAF) density-estimator settings.

    These are the `sbi.neural_nets.net_builders.build_maf` keyword arguments the Inference
    stage passes explicitly, so the flow's capacity is a recorded configuration rather than a
    library default, and they are persisted verbatim in the saved estimator's rebuild
    specification (`artifacts.save_estimator`, ``rebuild_spec.maf_args``). The defaults below
    are the settings every earlier estimator of this pipeline was trained with, so a run with
    no preset reproduces them exactly; they are NOT the library defaults (sbi 0.26.1 builds
    with ``z_score`` "independent", dropout 0 and no batch normalization).

    - `hidden_features`: width of each autoregressive transform's hidden layers.
    - `num_transforms`: number of stacked autoregressive transforms.
    - `num_blocks`: residual blocks per transform.
    - `dropout_probability` / `use_batch_norm`: regularization inside the transforms.
    - `z_score_x` / `z_score_y`: standardization of the parameters and of the embedding input,
      fixed from the batch handed to ``build_maf`` and never refitted. "structured" pools EVERY
      entry of the batch into ONE scalar mean and one scalar scale (the mean of the per-row
      standard deviations), so coordinates of different scale are not equalized;
      "independent" standardizes each dimension by its own mean and standard deviation.
    """
    hidden_features: int = 50
    num_transforms: int = 5
    num_blocks: int = 2
    dropout_probability: float = 0.1
    use_batch_norm: bool = True
    z_score_x: str = "structured"
    z_score_y: str = "structured"


# The standardization of the BIOLOGY theta (``z_score_x`` of the biology flow), applied by the
# Inference stage after the preset: the biology table mixes log10 coordinates spanning two to
# four decades with two linear fractions on [0, 1], which one shared scalar scale ("structured")
# would leave an order of magnitude apart. "independent" fits each coordinate's mean and scale,
# which needs a REPRESENTATIVE batch: the stage hands ``build_maf`` every theta label of the
# first TRAIN task (about a thousand prior draws), never the two-row dummy batch that fixes
# the video side (two draws would mis-scale at least one of twelve coordinates more than
# tenfold in most builds). The detector keeps ``InferenceFlow.z_score_x``.
THETA_STANDARDIZATION: str = "independent"


# Named capacity presets for the Inference stage (`--network-preset`). A preset overrides fields
# of `InferenceNetwork` and `InferenceFlow` together; everything else (data, targets, splits,
# preprocessing, training protocol, batch normalization) is untouched, and no preset touches the
# standardization: the biology theta standardization (THETA_STANDARDIZATION) is applied by the
# workflow after the preset. The baseline preset is the empty override and reproduces the
# pre-0.1.13 architecture.
#
#   capacity256 -- the combined capacity test of DETECTOR_WORKFLOW.md sec. 9.7: the embedding is
#   widened to 256 dimensions by doubling every convolutional block's channels
#   (start_channels 8 -> 16 with the same five blocks: 16 * 2^4 = 256 features per temporal
#   token, 64 per attention head), and the flow to hidden_features 128 / num_transforms 8 /
#   num_blocks 2 / dropout 0.1. A combined test: an improvement supports the larger
#   configuration but does not isolate the embedding from the flow.
#
#   kernel7 / earlyconv / statspool -- the three isolated encoder arms of the encoder screening
#   (DETECTOR_WORKFLOW.md, the section on the encoder screening). Each changes ONE encoder setting
#   at the baseline widths, depth, temporal transformer and flow: `kernel7` widens the first
#   block's spatial kernel from 3 to 7; `earlyconv` adds one spatial-only (1, 3, 3) convolution
#   (with batch normalization and Mish) before the pooling of blocks 1 and 2; `statspool`
#   reduces the final feature maps by their mean, standard deviation and maximum over positions,
#   projected back to the channel width, instead of the mean alone. The flow is the baseline's
#   in all three, so a difference is attributable to the encoder and its training. Smoked on
#   2026-09-30 as the technical evidence for the combined arms below; not trained.
#
#   capacity256_kernel7_stats / capacity256_earlyconv_stats -- the two combined candidates of the
#   encoder screening, both on the `capacity256` base (its widths, 16 * 2^4 = 256 features per
#   temporal token, and its flow) with statistics pooling (3 * 256 -> 256), differing in the early
#   spatial processing: the first block's spatial kernel widened to 7, or one spatial-only (1, 3, 3)
#   convolution before the pooling of blocks 1 and 2. Neither isolates one modification; each is a
#   candidate architecture for the whole posterior estimator, compared with `capacity256` (its base)
#   and with `baseline` (the secondary reference).
#
#   capacity256_kernel7_earlyconv_stats -- the third combined candidate (2026-10-02): both early
#   modifications together on the same base, the 7-wide first spatial kernel AND the spatial-only
#   (1, 3, 3) convolution before the pooling of blocks 1 and 2, with statistics pooling and the
#   `capacity256` flow. Its stored activations equal the earlyconv candidate's (the wider kernel changes
#   no tensor size); its theoretical spatial reach is 104 px.
_CAPACITY256_FLOW: dict = dict(hidden_features=128, num_transforms=8, num_blocks=2, dropout_probability=0.1)
NETWORK_PRESETS: dict = {
    "baseline": dict(network={}, flow={}),
    "capacity256": dict(network=dict(start_channels=16), flow=dict(_CAPACITY256_FLOW)),
    "kernel7": dict(network=dict(first_spatial_kernel=7), flow={}),
    "earlyconv": dict(network=dict(extra_spatial_convs=1, extra_spatial_conv_blocks=2), flow={}),
    "statspool": dict(network=dict(spatial_pooling="stats"), flow={}),
    "capacity256_kernel7_stats": dict(network=dict(start_channels=16, first_spatial_kernel=7, spatial_pooling="stats"),
                                      flow=dict(_CAPACITY256_FLOW)),
    "capacity256_earlyconv_stats": dict(network=dict(start_channels=16, extra_spatial_convs=1, extra_spatial_conv_blocks=2,
                                                     spatial_pooling="stats"),
                                        flow=dict(_CAPACITY256_FLOW)),
    "capacity256_kernel7_earlyconv_stats": dict(network=dict(start_channels=16, first_spatial_kernel=7,
                                                             extra_spatial_convs=1, extra_spatial_conv_blocks=2,
                                                             spatial_pooling="stats"),
                                                flow=dict(_CAPACITY256_FLOW)),
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

# The prior boxes of the window-start composition (decided 2026-10-05 and 2026-10-15;
# PROJECT_CONTEXT.md sec. 2, *Modeling assumptions of the MET model*, items 9 and 14).
# COUNT_TOTAL_BOX is log10 of the true receptor-subunit total N_total, one box for both
# conditions. FRACTION_BOX is the box of the two LINEAR rows: the basal reference fraction of
# dimers in the underlying population and, under MET-INLB, the initial active-dimer allocation
# relative to its capacity; both are uniform on [0, 1] in the estimator coordinate.
COUNT_TOTAL_BOX: tuple = (3.0, 5.0)
FRACTION_BOX: tuple = (0.0, 1.0)


_PARAMETERIZATION_RAW_NESTED: dict[str, list[dict]] = {
    # ----- Reaction-Diffusion System: the two model blocks -----
    # DECIDED PRIOR RANGES (2026-09-14; the count and composition rows revised 2026-10-05 for the
    # retained-population model). Box-uniform in the estimator coordinate; each DOC names its
    # source (baseline = dimer-alp 0.4.23; A9 = Special_Analyses per-recording analysis of the
    # deposited MET tables; spec = model specification sec. 9 anchors; thresholds = tracking
    # pipelines). Rationale: PROJECT_CONTEXT.md sec. 2, "Modeling assumptions of the MET model" and
    # "How the prior ranges and the declared inputs are set". The receptor count is the TRUE total,
    # conditional on the declared occupancies that realize the simulated (retained) population.
    'stoichiometry': [  # StoichiometryBlock: true receptor total, basal reference fraction of dimers, dissociation (the association ratio is a per-condition CONSTANT, not a row; the initial active-dimer allocation is the 'initialization' row below)
        {'KEY': 'count_total', 'VALUE': 10**4.0, 'PRIOR_RANGE': COUNT_TOTAL_BOX, 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'DERIVED_UNIT': None, 'LABEL': r'$N_{\mathrm{total}}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'N_total, the receptor-subunit total of the underlying population in the simulated patch, probe-bound and unbound receptors included; one prior for both conditions, log10 on [3.0, 5.0] (1,000-100,000 subunits), decided 2026-10-05. It is NOT the number of simulated particles: the generator realizes the REFERENCE probe-bound population from it through the condition\'s occupancy p (realize_population), then applies the initial allocation of the window-start composition. Sizing at N_total = 100,000, bookkeeping rather than a population change: under MET-FAB the initial particles stay at the expected bound probes (about 1,477; 1,472 at a basal fraction of 0.25, 1,466 at 1) for every basal fraction, while the represented receptors N_R rise from 1,477 (basal fraction 0) through 2,058 (0.25) to 2,932 (1), because more probe-free partners are represented inside dimers; particles can approach N_R only after dissociation, so the upper figure is a memory and timing consideration for the worst case. Under MET-INLB N_R stays about 4,762 for every basal fraction and allocation (the allocation conserves it). The two conditions share this quantity because they share the cell line and the probe concentration; their simulated populations differ through the occupancies. The deposited recordings (A9: spots per frame in the first 2 s, divided by the visibility p x P(dye >= 1)) imply totals of about 1,400-25,800 (FAB) and 3,000-33,300 (INLB), computed without missed detections and overlap, so the ceiling carries headroom for the undercount; it is a declared modeling choice, not an experimental maximum. Compatible N_total posteriors across the conditions are a consistency check, not independent validation (the Fab occupancy rests on the declared visibility ratio). In-field count and represented count are distinct quantities.'},
        {'KEY': 'fraction_basal_dimers_reference', 'VALUE': 0.5, 'PRIOR_RANGE': FRACTION_BOX, 'LOG_FLAG': False, 'LOG_BASE': None, 'UNIT': 'Fraction', 'DERIVED_UNIT': 'Count', 'LABEL': r'$f_{B}^{ref}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Basal reference fraction f of the underlying population, the fraction of its complexes that are dimers before any initial allocation: M_ref = N_total (1 - f) / (1 + f) monomers and D_ref = N_total f / (1 + f) dimers, so that M_ref + 2 D_ref = N_total. A LINEAR row, uniform on [0, 1] (decided 2026-10-15; the earlier log10 dimer-to-monomer ratio on the 5-25 % basal band is retired): the basal measurements (5 +/- 1 % of clusters dimeric, Baldering et al. 2021; 18 % of fitted spots assigned to a dimer component in fixed cells, stated as an underestimate, Dietz et al. 2013) inform the reading of the posterior, not the prior. Under MET-FAB the reference population IS the starting population (the allocation is declared 0), so f is the fraction of dimers present at the recording start, not one formed. Under MET-INLB the reference population is the state from which the initial allocation (relative_count_active_dimers_initial) pairs probe-bound monomers into two-InlB dimers, so the starting fraction f_start = (D_ref + c) / (M_ref + D_ref - c) differs from f; both are reported. The row is an inference coordinate whose calibrated, prior-like posterior is acceptable; a narrow prior is not evidence of recovery.'},
        {'KEY': 'capture_radius', 'VALUE': 10, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Nanometer', 'DERIVED_UNIT': None, 'LABEL': r'$\rho_{CAP}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'rate_dissociation', 'VALUE': 10**(-1), 'PRIOR_RANGE': (-3, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{OFF}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Dimer unbinding rate, one rate for every dimer class and mode (dissociation conserves the mode): B_m -> A_m + A_m (FAB) and B2_m -> A_m + A_m by fission, B1_m -> A_m by conversion (the probe-free partner is not represented); inferred in every condition. Range 0.001-10 per s: the lower bound is widened from the baseline (-1, 1) so that under MET-FAB, where no association exists, dimers can PERSIST through a 20 s recording (0.001 per s loses 2% in 20 s); the upper bound, a 0.1 s lifetime (five frames), is the baseline value. Under MET-FAB kappa_OFF is the dimer lifetime rate directly.'},
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
         'DOC': 'Switching fast -> slow, shared by all molecular species of the condition. Range 0.1-10 per s: the baseline 0.4.23 band of the earlier mobile/immobile switching, kept for all four shared rates; resolvable between the 20 ms frame and the 2 s window; the observed class fractions (67/22/11 Fab, 43/29/28 InlB) imply rate ratios between 0.3 and 1, well inside.'},
        {'KEY': 'rate_slow_fast', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$k_{sf}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Switching slow -> fast, shared by all molecular species of the condition. Range 0.1-10 per s: the baseline 0.4.23 band of the earlier mobile/immobile switching, kept for all four shared rates; resolvable between the 20 ms frame and the 2 s window; the observed class fractions (67/22/11 Fab, 43/29/28 InlB) imply rate ratios between 0.3 and 1, well inside.'},
        {'KEY': 'rate_slow_immobile', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$k_{si}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Switching slow -> immobile, shared by all molecular species of the condition. Range 0.1-10 per s: the baseline 0.4.23 band of the earlier mobile/immobile switching, kept for all four shared rates; resolvable between the 20 ms frame and the 2 s window; the observed class fractions (67/22/11 Fab, 43/29/28 InlB) imply rate ratios between 0.3 and 1, well inside.'},
        {'KEY': 'rate_immobile_slow', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$k_{is}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Switching immobile -> slow, shared by all molecular species of the condition. Range 0.1-10 per s: the baseline 0.4.23 band of the earlier mobile/immobile switching, kept for all four shared rates; resolvable between the 20 ms frame and the 2 s window; the observed class fractions (67/22/11 Fab, 43/29/28 InlB) imply rate ratios between 0.3 and 1, well inside.'},
    ],
    'initialization': [  # the window-start allocation: inferred under MET-INLB, a declared constant (0) under MET-FAB (ConditionSetting.relative_count_active_dimers_initial)
        {'KEY': 'relative_count_active_dimers_initial', 'VALUE': 0.5, 'PRIOR_RANGE': FRACTION_BOX, 'LOG_FLAG': False, 'LOG_BASE': None, 'UNIT': 'Fraction', 'DERIVED_UNIT': 'Count', 'LABEL': r'$R_{active}$', 'NOTE': 'Learnable Parameter',
         'DOC': 'Initial active-dimer allocation R_active of the window-start composition, a LINEAR row uniform on [0, 1] (decided 2026-10-15), inferred under MET-INLB only; MET-FAB declares it 0 on its condition setting, so the Fab table has no such row. It is a CAPACITY-RELATIVE count, not the active share: the reference probe-bound population holds A_ref probe-bound monomers, of which at most cap = A_ref // 2 pairs can be formed; c = round(R_active cap) pairs are turned into additional two-InlB dimers before the recording starts, A_start = A_ref - 2c, B2_start = B2_ref + c, B1 unchanged, and the represented receptor and probe budget N_R is conserved exactly. The declared starting-state approximation: at initialization, the declared occupancy defines a reference probe-bound population; additional initial B2 dimers are obtained by pairing already probe-bound monomers from that population, preserving its receptor and probe budget. This is not a reconstruction of probe binding or receptor reactions before the analysis window; fixed attachment remains the assumption applied during the simulated recording. The active share c / (D_ref + c) of the starting dimers is a DERIVED quantity, reported per joint posterior draw; at zero capacity (no pair available) the row has no effect and a prior-like posterior is expected.'},
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
# delta_frame, numb_photo_bleach) resolve to 'fixed', so the learnable subset of the MASTER
# table is exactly the twelve reaction-diffusion rows; a condition's own table removes the
# rows its setting declares (eleven inferred coordinates under MET-FAB, twelve under
# MET-INLB; `parameterization_of`).


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


# The MASTER learnable table: VALUE-not-a-sentinel AND PRIOR_RANGE-not-None, in declaration
# order (twelve rows). It is a set of row DEFINITIONS, never a theta schema by itself: a
# condition's theta vector holds the master rows minus the rows its condition setting declares
# as constants (``parameterization_of``), so the eleven shared coordinates sit at positions
# 0-10 under both conditions and the InlB allocation is the twelfth. Alignment across
# conditions is always by KEY (``parameter_keys_of``, ``parameter_index_of``).
MASTER_PARAMETERIZATION: tuple = tuple(
    para for para in PARAMETERIZATION_RAW if role_of(para) == 'learnable'
)
MASTER_PARAMETER_KEYS: tuple = tuple(para['KEY'] for para in MASTER_PARAMETERIZATION)


def declared_row_values(condition: str) -> dict:
    """The master rows ``condition`` declares as constants, KEY -> value (FAB: the allocation
    0.0; INLB: none); ``SimulationRDS.declared_row_values``."""
    return PARAMETERS.simulation.rds.declared_row_values(condition)


def parameterization_of(condition: str) -> list[dict]:
    """The learnable table of ``condition``: the master rows minus its declared rows, as a
    fresh list of copied row dictionaries (the master is never modified; eleven rows under
    MET-FAB, twelve under MET-INLB). This list is the theta schema of every product of the
    condition: ``Theta_Set`` columns, estimator coordinates, prior box."""
    declared = declared_row_values(condition)
    return [dict(para) for para in MASTER_PARAMETERIZATION if para['KEY'] not in declared]


def parameterization_raw_of(condition: str) -> list[dict]:
    """The full specification seen from ``condition``: every row of ``PARAMETERIZATION_RAW``,
    copied, with each declared row turned into a FIXED row (VALUE = the declared constant,
    PRIOR_RANGE None, NOTE 'Declared Constant'), so that ``role_of`` reads the condition's
    roles without a condition of its own (the test-loss manifest, the fixed-parameter
    tables). The master is never modified."""
    declared = declared_row_values(condition)
    rows = []
    for para in PARAMETERIZATION_RAW:
        row = dict(para)
        if row['KEY'] in declared:
            row['VALUE'] = float(declared[row['KEY']])
            row['PRIOR_RANGE'] = None
            row['NOTE'] = 'Declared Constant'
        rows.append(row)
    return rows


def parameter_keys_of(condition: str) -> list[str]:
    """Ordered learnable keys of ``condition`` = its theta-vector schema. Passed to
    ``artifacts.load_estimator(expected_parameter_keys=...)`` and the by-key guards of the
    stage writers, so a product of another condition or of a superseded table is refused
    (equal-length theta vectors would otherwise be misread column for column)."""
    return [para['KEY'] for para in parameterization_of(condition)]


def parameter_index_of(condition: str) -> dict[str, int]:
    """KEY -> position in the theta vector of ``condition``."""
    return {key: index for index, key in enumerate(parameter_keys_of(condition))}


# =============================================================================
# The one parameter-conversion rule (estimator space <-> physical values)
# =============================================================================

def is_log_row(entry: dict) -> bool:
    """True for a ranged row whose estimator coordinate is log10 of the physical value."""
    return bool(entry.get('LOG_FLAG'))


def _require_table(parameterization, where: str) -> list:
    if parameterization is None:
        raise ValueError(f"{where}: the parameter table is required (parameterization_of(condition) "
                         f"for the biology; the width differs between the conditions).")
    return list(parameterization)


def to_physical(theta_flow, parameterization):
    """Map estimator-space coordinates to physical values, row by row.

    ``theta_flow`` has the learnable parameters on its LAST axis, in the order of
    ``parameterization`` (the condition's table, ``parameterization_of(condition)``; required,
    because the width differs between the conditions). Log rows are exponentiated
    (``10 ** u``); linear rows pass through. Returns a float64 array of the same shape.
    This is the only sanctioned conversion; a blanket ``10 ** theta`` corrupts the linear
    rows (the basal reference fraction and the InlB allocation).
    """
    table = _require_table(parameterization, "to_physical")
    arr = np.array(theta_flow, dtype=float, copy=True)
    if arr.shape[-1] != len(table):
        raise ValueError(f"to_physical: last axis has {arr.shape[-1]} entries; the table has {len(table)}.")
    for j, entry in enumerate(table):
        if is_log_row(entry):
            arr[..., j] = np.power(10.0, arr[..., j])
    return arr


def to_flow(theta_physical, parameterization):
    """Map physical values to estimator-space coordinates (inverse of ``to_physical``).

    Log rows become ``log10`` (a non-positive physical value on a log row is an error, not a
    NaN, because it can only come from a corrupted table or a mis-scaled input); linear rows
    pass through. Returns a float64 array of the same shape. The table is required.
    """
    table = _require_table(parameterization, "to_flow")
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


def fraction_flags(theta_physical, parameterization) -> dict:
    """Per LINEAR row of ``parameterization``: a boolean array (the leading shape of
    ``theta_physical``) that is True where the value is non-finite or outside the row's box.
    Reporting code flags and counts such draws and carries NaN for their derived quantities;
    nothing is ever clipped. Returns KEY -> flags."""
    table = _require_table(parameterization, "fraction_flags")
    arr = np.asarray(theta_physical, dtype=float)
    if arr.shape[-1] != len(table):
        raise ValueError(f"fraction_flags: last axis has {arr.shape[-1]} entries; the table has {len(table)}.")
    flags = {}
    for j, entry in enumerate(table):
        if is_log_row(entry):
            continue
        lo, hi = entry['PRIOR_RANGE']
        col = arr[..., j]
        flags[entry['KEY']] = ~(np.isfinite(col) & (col >= lo) & (col <= hi))
    return flags


# =============================================================================
# Window-start composition: the reference probe-bound population and the initial allocation
# =============================================================================

_POPULATION_FIELDS = (
    # name, dtype ('f' float64 / 'i' int64), one-line meaning
    ("count_total", "f", "N_total, the receptor-subunit total of the underlying population"),
    ("fraction_basal_dimers_reference", "f", "f, the basal reference fraction (the inferred coordinate)"),
    ("relative_count_active_dimers_initial", "f", "R_active, the allocation (inferred under INLB; the declared 0 under FAB)"),
    ("n_monomers_underlying_reference", "f", "M_ref = N_total (1 - f) / (1 + f), underlying monomers of the reference"),
    ("n_dimers_underlying_reference", "f", "D_ref = N_total f / (1 + f), underlying dimers of the reference"),
    ("n_monomers_reference", "i", "A_ref = round(p M_ref), probe-bound monomers of the reference"),
    ("n_dimers_reference", "i", "D_sim = round(p (2 - p) D_ref), dimers carrying at least one probe, in the reference"),
    ("n_dimers_one_probe", "i", "B1 = D_sim - B2_ref, one-probe dimers (unchanged by the allocation)"),
    ("n_dimers_two_probe_reference", "i", "B2_ref = min(round(p^2 D_ref), D_sim), two-probe dimers of the reference"),
    ("capacity", "i", "cap = A_ref // 2 under ligand classes, else 0: the pairs the reference can form"),
    ("n_pairs_allocated", "i", "c = round(R_active cap), probe-bound monomer pairs turned into two-InlB dimers"),
    ("n_monomers", "i", "A_start = A_ref - 2c, the probe-bound monomers PLACED"),
    ("n_dimers_two_probe", "i", "B2_start = B2_ref + c, the two-probe dimers PLACED (B2 under INLB; the two-Fab class under FAB, assigned at labeling)"),
    ("n_dimers", "i", "B1 + B2_start, the dimers PLACED"),
    ("n_subunits", "i", "N_R, the represented receptors, conserved by the allocation and by every channel"),
    ("n_monomers_underlying_start", "f", "M_start = M_ref - 2c, underlying monomers at the window start"),
    ("n_dimers_underlying_start", "f", "D_start = D_ref + c, underlying dimers at the window start"),
    ("fraction_basal_dimers_start", "f", "f_start = D_start / (M_start + D_start), the starting fraction (DERIVED)"),
    ("active_share", "f", "c / D_start, the share of the starting dimers the allocation formed (DERIVED; NaN at D_start = 0)"),
)
POPULATION_FIELD_DOCS: dict = {name: doc for name, _, doc in _POPULATION_FIELDS}


@dataclass(frozen=True)
class PopulationRealization:
    """The window-start composition of one or many draws, as arrays of one common shape.

    Two populations are distinguished throughout (PROJECT_CONTEXT.md sec. 2, *Modeling
    assumptions of the MET model*, items 4, 9 and 11). The REFERENCE probe-bound population is
    what the declared occupancy p defines from the underlying population of N_total subunits
    with the basal reference fraction f: underlying M_ref = N_total (1 - f) / (1 + f) monomers
    and D_ref = N_total f / (1 + f) dimers (M_ref + 2 D_ref = N_total), realized DIMERS AND
    THEIR CLASSES FIRST, then the monomers, as rounded expectations of the independent-binding
    thinning: D_sim = round(p (2 - p) D_ref), B2_ref = min(round(p^2 D_ref), D_sim),
    B1 = D_sim - B2_ref, A_ref = round(p M_ref), rounded half to even. The STARTING population
    is what the generator places: under ligand classes (MET-INLB) the allocation pairs
    c = round(R_active cap) of the cap = A_ref // 2 available probe-bound monomer pairs into
    additional two-InlB dimers (A_start = A_ref - 2c, B2_start = B2_ref + c, B1 unchanged),
    conserving the represented receptor and probe budget N_R = A_ref + B1 + 2 B2_ref exactly;
    without ligand classes (MET-FAB, allocation declared 0) the starting population IS the
    reference, with N_R = A_ref + 2 D_sim (both subunits of a retained Fab dimer are
    represented; its one-/two-Fab class is assigned at labeling).

    The placed-count fields (``n_monomers``, ``n_dimers``, ``n_dimers_one_probe``,
    ``n_dimers_two_probe``, ``n_subunits``) are the STARTING counts; the reference counts carry
    the ``_reference`` suffix; the underlying (continuum) counts carry ``_underlying``. The
    derived starting fraction f_start and the active share c / D_start are reported beside f and
    R_active, never in their place; nothing reconstructs underlying monomers as A_start / p.
    ``POPULATION_FIELD_DOCS`` names every field.
    """
    condition: str
    ligand_classes: bool
    occupancy: float
    two_probe_share: float
    count_total: np.ndarray
    fraction_basal_dimers_reference: np.ndarray
    relative_count_active_dimers_initial: np.ndarray
    n_monomers_underlying_reference: np.ndarray
    n_dimers_underlying_reference: np.ndarray
    n_monomers_reference: np.ndarray
    n_dimers_reference: np.ndarray
    n_dimers_one_probe: np.ndarray
    n_dimers_two_probe_reference: np.ndarray
    capacity: np.ndarray
    n_pairs_allocated: np.ndarray
    n_monomers: np.ndarray
    n_dimers_two_probe: np.ndarray
    n_dimers: np.ndarray
    n_subunits: np.ndarray
    n_monomers_underlying_start: np.ndarray
    n_dimers_underlying_start: np.ndarray
    fraction_basal_dimers_start: np.ndarray
    active_share: np.ndarray

    @property
    def shape(self) -> tuple:
        return tuple(np.shape(self.n_subunits))

    @property
    def n_complexes(self) -> np.ndarray:
        """Particles placed, A_start + B1 + B2_start."""
        return self.n_monomers + self.n_dimers

    @property
    def n_probe_bound_receptors_expected(self) -> np.ndarray:
        """Probe-bound receptors among the represented ones: exact under ligand classes
        (A + B1 + 2 B2 = N_R), the expectation A + D_sim (1 + s_2) under FAB, whose one-/two-Fab
        classes are drawn at labeling."""
        if self.ligand_classes:
            return self.n_monomers + self.n_dimers_one_probe + 2 * self.n_dimers_two_probe
        return self.n_monomers + self.n_dimers * (1.0 + self.two_probe_share)

    @property
    def expected_subunits(self) -> np.ndarray:
        """Expectation of N_R before rounding: p N_total under ligand classes, else
        p N_total [1 + f (3 - 2p)] / (1 + f)."""
        p, f = self.occupancy, self.fraction_basal_dimers_reference
        if self.ligand_classes:
            return p * self.count_total
        return p * self.count_total * (1.0 + f * (3.0 - 2.0 * p)) / (1.0 + f)

    def item(self, index=()) -> "InitialComposition":
        """The one-draw record (Python scalars) at ``index`` of these arrays."""
        values = {}
        for name, kind, _ in _POPULATION_FIELDS:
            value = np.asarray(getattr(self, name))[index]
            values[name] = int(value) if kind == "i" else float(value)
        return InitialComposition(condition=self.condition, ligand_classes=self.ligand_classes,
                                  occupancy=self.occupancy, two_probe_share=self.two_probe_share, **values)


@dataclass(frozen=True)
class InitialComposition:
    """The window-start composition of ONE simulation, with Python scalars: the record the
    generator places from and the diagnostics print. Same fields and meanings as
    ``PopulationRealization`` (``POPULATION_FIELD_DOCS``); the placed-count fields are the
    STARTING counts, the reference counts carry ``_reference``."""
    condition: str
    ligand_classes: bool
    occupancy: float
    two_probe_share: float
    count_total: float
    fraction_basal_dimers_reference: float
    relative_count_active_dimers_initial: float
    n_monomers_underlying_reference: float
    n_dimers_underlying_reference: float
    n_monomers_reference: int
    n_dimers_reference: int
    n_dimers_one_probe: int
    n_dimers_two_probe_reference: int
    capacity: int
    n_pairs_allocated: int
    n_monomers: int
    n_dimers_two_probe: int
    n_dimers: int
    n_subunits: int
    n_monomers_underlying_start: float
    n_dimers_underlying_start: float
    fraction_basal_dimers_start: float
    active_share: float

    @property
    def n_complexes(self) -> int:
        """Particles placed, A_start + B1 + B2_start."""
        return self.n_monomers + self.n_dimers

    @property
    def n_total_true(self) -> float:
        """N_total, the receptor-subunit total of the underlying population (``count_total``)."""
        return self.count_total

    @property
    def n_probe_bound_receptors_expected(self) -> float:
        if self.ligand_classes:
            return float(self.n_monomers + self.n_dimers_one_probe + 2 * self.n_dimers_two_probe)
        return self.n_monomers + self.n_dimers * (1.0 + self.two_probe_share)

    @property
    def expected_subunits(self) -> float:
        p, f = self.occupancy, self.fraction_basal_dimers_reference
        if self.ligand_classes:
            return p * self.count_total
        return p * self.count_total * (1.0 + f * (3.0 - 2.0 * p)) / (1.0 + f)

    # compatibility name kept for the diagnostics that print the composition
    @property
    def n_total(self) -> int:
        """The conserved represented receptor count N_R (``n_subunits``)."""
        return self.n_subunits


def _realize_population_arrays(count_total, fraction_basal_dimers_reference,
                               relative_count_active_dimers_initial, occupancy: float,
                               ligand_classes: bool) -> dict:
    """The pure arithmetic of the window-start composition on broadcast float64 arrays, with
    the given occupancy ``p`` and ligand-class flag; no validation (``realize_population`` does
    it). Rounding is ``np.round`` (half to even, as Python's ``round``), with the operand order
    of the reference rules kept so that the zero-allocation integers are reproduced exactly.
    Exposed for the tests, which probe rounding ties at a chosen ``p``."""
    n_total, f, r_active = np.broadcast_arrays(
        np.asarray(count_total, dtype=np.float64), np.asarray(fraction_basal_dimers_reference, dtype=np.float64),
        np.asarray(relative_count_active_dimers_initial, dtype=np.float64))
    p = float(occupancy)
    m_ref = n_total * (1.0 - f) / (1.0 + f)
    d_ref = n_total * f / (1.0 + f)
    d_sim = np.round(p * (2.0 - p) * d_ref)
    b2_ref = np.minimum(np.round(p * p * d_ref), d_sim)
    b1 = d_sim - b2_ref
    a_ref = np.round(p * m_ref)
    cap = np.floor_divide(a_ref, 2.0) if ligand_classes else np.zeros_like(a_ref)
    c = np.round(r_active * cap)
    a_start = a_ref - 2.0 * c
    b2_start = b2_ref + c
    n_r = a_ref + b1 + 2.0 * b2_ref if ligand_classes else a_ref + 2.0 * d_sim
    m_start = m_ref - 2.0 * c
    d_start = d_ref + c
    with np.errstate(invalid="ignore", divide="ignore"):
        f_start = np.where(m_start + d_start > 0.0, d_start / np.where(m_start + d_start > 0.0, m_start + d_start, 1.0), np.nan)
        share = np.where(d_start > 0.0, c / np.where(d_start > 0.0, d_start, 1.0), np.nan)
    as_int = lambda a: np.asarray(a, dtype=np.int64)  # noqa: E731
    return dict(
        count_total=n_total, fraction_basal_dimers_reference=f, relative_count_active_dimers_initial=r_active,
        n_monomers_underlying_reference=m_ref, n_dimers_underlying_reference=d_ref,
        n_monomers_reference=as_int(a_ref), n_dimers_reference=as_int(d_sim), n_dimers_one_probe=as_int(b1),
        n_dimers_two_probe_reference=as_int(b2_ref), capacity=as_int(cap), n_pairs_allocated=as_int(c),
        n_monomers=as_int(a_start), n_dimers_two_probe=as_int(b2_start), n_dimers=as_int(b1 + b2_start),
        n_subunits=as_int(n_r), n_monomers_underlying_start=m_start, n_dimers_underlying_start=d_start,
        fraction_basal_dimers_start=f_start, active_share=share)


def population_valid(*, count_total, fraction_basal_dimers_reference, relative_count_active_dimers_initial,
                     condition: str) -> np.ndarray:
    """The ONE validity rule of a window-start input, elementwise over broadcast arrays: N_total
    finite and > 0; f and R_active finite and in [0, 1]; R_active equal to the declared constant
    where ``condition`` declares one (MET-FAB: 0). Reporting code masks posterior clouds and MAP
    vectors with this rule, realizes the valid subset, counts the flagged entries and carries
    NaN for them, so one out-of-range draw never aborts a report and nothing is clipped."""
    n_total, f, r_active = np.broadcast_arrays(
        np.asarray(count_total, dtype=np.float64), np.asarray(fraction_basal_dimers_reference, dtype=np.float64),
        np.asarray(relative_count_active_dimers_initial, dtype=np.float64))
    declared = declared_row_values(condition).get(PARAMETERS.simulation.rds.stoichiometry.active_allocation_key)
    lo, hi = FRACTION_BOX
    ok = np.isfinite(n_total) & (n_total > 0.0)
    ok &= np.isfinite(f) & (f >= lo) & (f <= hi)
    ok &= np.isfinite(r_active) & (r_active >= lo) & (r_active <= hi)
    if declared is not None:
        ok &= np.abs(r_active - float(declared)) <= 1e-12
    return ok


def realize_population(*, count_total, fraction_basal_dimers_reference, relative_count_active_dimers_initial,
                       condition: str) -> PopulationRealization:
    """The window-start composition of every draw in the broadcast inputs, under ``condition``
    (its occupancy, ligand classes and declared allocation). Strict: any invalid element
    (``population_valid``) raises ``ValueError`` naming the count; nothing is clipped. Callers
    holding posterior clouds mask with ``population_valid`` first."""
    rds = PARAMETERS.simulation.rds
    setting = rds.condition_setting(condition)
    p = rds.occupancy_of(condition)
    valid = population_valid(count_total=count_total, fraction_basal_dimers_reference=fraction_basal_dimers_reference,
                             relative_count_active_dimers_initial=relative_count_active_dimers_initial, condition=condition)
    if not np.all(valid):
        n_bad, n_all = int(np.size(valid) - np.count_nonzero(valid)), int(np.size(valid))
        declared = declared_row_values(condition).get(rds.stoichiometry.active_allocation_key)
        rule = (f"N_total finite and > 0, {rds.stoichiometry.basal_fraction_key} and "
                f"{rds.stoichiometry.active_allocation_key} finite in [0, 1]")
        if declared is not None:
            rule += f", and {rds.stoichiometry.active_allocation_key} equal to the declared constant {declared:g} under {condition}"
        raise ValueError(f"realize_population [{condition}]: {n_bad} of {n_all} input(s) invalid ({rule}); "
                         f"nothing is clipped. Mask with population_valid first.")
    arrays = _realize_population_arrays(count_total, fraction_basal_dimers_reference,
                                        relative_count_active_dimers_initial, p, bool(setting.ligand_classes))
    return PopulationRealization(condition=str(condition), ligand_classes=bool(setting.ligand_classes),
                                 occupancy=float(p), two_probe_share=float(p / (2.0 - p)), **arrays)


def population_from_theta(theta_physical, condition: str) -> PopulationRealization:
    """``realize_population`` on physical theta vectors (``(..., D)`` in the order of
    ``parameterization_of(condition)``): the count, fraction and allocation columns are read by
    KEY, a declared allocation is inserted, and the width is checked."""
    arr = np.asarray(theta_physical, dtype=np.float64)
    keys = parameter_keys_of(condition)
    if arr.shape[-1] != len(keys):
        raise ValueError(f"population_from_theta [{condition}]: theta has {arr.shape[-1]} columns; "
                         f"the condition's table has {len(keys)} ({keys}).")
    sto = PARAMETERS.simulation.rds.stoichiometry
    index = {key: i for i, key in enumerate(keys)}
    declared = declared_row_values(condition)
    if sto.active_allocation_key in index:
        r_active = arr[..., index[sto.active_allocation_key]]
    else:
        r_active = np.full(arr.shape[:-1], float(declared[sto.active_allocation_key]))
    return realize_population(count_total=arr[..., index[sto.count_total_key]],
                              fraction_basal_dimers_reference=arr[..., index[sto.basal_fraction_key]],
                              relative_count_active_dimers_initial=r_active, condition=condition)


def realize_initial_composition(*, count_total: float, fraction_basal_dimers_reference: float,
                                relative_count_active_dimers_initial: float,
                                condition: str) -> InitialComposition:
    """The window-start composition of ONE simulation (``PopulationRealization`` for the
    rules): the record the generator places from. Keyword-only; strict (``ValueError`` on an
    invalid input, never a clipped one). Under MET-FAB the allocation must be the declared 0."""
    pop = realize_population(count_total=float(count_total),
                             fraction_basal_dimers_reference=float(fraction_basal_dimers_reference),
                             relative_count_active_dimers_initial=float(relative_count_active_dimers_initial),
                             condition=condition)
    return pop.item(())


# =============================================================================
# Model-block validation (import time, fail fast)
# =============================================================================

def _validate_model_blocks() -> None:
    rds = PARAMETERS.simulation.rds
    master = {para['KEY']: para for para in MASTER_PARAMETERIZATION}
    for key in rds.stoichiometry.parameter_keys + rds.mobility.parameter_keys:
        if key not in master:
            raise ValueError(f"model block references parameter {key!r}, which is not a learnable master row.")
    for entry in MASTER_PARAMETERIZATION:
        lo, hi = entry['PRIOR_RANGE']
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"parameter {entry['KEY']!r}: invalid PRIOR_RANGE {entry['PRIOR_RANGE']}.")
        if entry['LOG_FLAG'] not in (True, False):
            raise ValueError(f"parameter {entry['KEY']!r}: a ranged row declares LOG_FLAG True or False.")
        if is_log_row(entry) and entry['LOG_BASE'] != 10:
            raise ValueError(f"parameter {entry['KEY']!r}: log rows are base 10.")
        if abs(float(entry['VALUE']) - prior_center(entry)) > 1e-9 * max(1.0, abs(prior_center(entry))):
            raise ValueError(f"parameter {entry['KEY']!r}: VALUE {entry['VALUE']!r} is not the prior center "
                             f"{prior_center(entry)!r} of its box.")
    for key in (rds.stoichiometry.basal_fraction_key, rds.stoichiometry.active_allocation_key):
        row = master[key]
        if is_log_row(row) or tuple(row['PRIOR_RANGE']) != FRACTION_BOX:
            raise ValueError(f"{key} is a LINEAR row on the box {FRACTION_BOX} (got LOG_FLAG {row['LOG_FLAG']}, "
                             f"PRIOR_RANGE {row['PRIOR_RANGE']}).")
    count = master[rds.stoichiometry.count_total_key]
    if tuple(count['PRIOR_RANGE']) != COUNT_TOTAL_BOX:
        raise ValueError(f"count_total must carry the declared box {COUNT_TOTAL_BOX} (got {count['PRIOR_RANGE']}).")
    # The twelve master keys in their decided order: eleven shared coordinates, then the allocation.
    if MASTER_PARAMETER_KEYS[-1] != rds.stoichiometry.active_allocation_key or len(MASTER_PARAMETER_KEYS) != 12:
        raise ValueError(f"the master table holds twelve learnable rows ending with the allocation "
                         f"(got {MASTER_PARAMETER_KEYS}).")
    # Mobility ordering R_immobile < R_slow <= 1 and 0 < R_dimer <= 1, by the declared ranges.
    def phys_range(key):
        e = master[key]
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
    # Occupancy: equilibrium, literal or derived, every value a probability in (0, 1]; the derived
    # MET-FAB value guards the anchor arithmetic (0.5 x 0.0238 / 0.806 = 0.0148 under the baseline
    # laws) and the MET-INLB value the equilibrium 0.25 / 5.25 = 0.0476.
    for t in tokens:
        p = rds.occupancy_of(t)
        if not (np.isfinite(p) and 0.0 < p <= 1.0):
            raise ValueError(f"SimulationRDS.conditions: occupancy of {t} resolves to {p!r}, not a probability.")
        anchor = rds.condition_setting(t).visibility_ratio_to
        if anchor is not None and rds.condition_setting(anchor).visibility_ratio is not None:
            raise ValueError(f"SimulationRDS.conditions: {t} derives its occupancy from an anchor that is itself derived.")
    if abs(rds.occupancy_of("FAB") - 0.0148) > 5e-5 or abs(rds.occupancy_of("INLB") - 0.25 / 5.25) > 1e-12:
        raise ValueError(f"SimulationRDS.conditions: occupancies resolve to FAB {rds.occupancy_of('FAB'):.5f} / "
                         f"INLB {rds.occupancy_of('INLB'):.5f}; the decided values are 0.0148 (derived) / "
                         f"0.0476 (equilibrium at 0.25 nM, K_D 5 nM). Change the declared settings AND this guard "
                         f"together.")
    if rds.condition_setting("INLB").ligand_classes is not True or rds.condition_setting("FAB").ligand_classes:
        raise ValueError("SimulationRDS.conditions: MET-INLB carries the ligand classes B1 / B2 and MET-FAB does not.")
    # Declared rows: every declared key is a master row and its value lies inside the row's box;
    # each condition's table is the master minus its declared rows, in master order; the raw view
    # agrees; the eleven shared coordinates keep their positions in every condition.
    for t in tokens:
        declared = rds.declared_row_values(t)
        for key, value in declared.items():
            if key not in master:
                raise ValueError(f"SimulationRDS.conditions: {t} declares {key!r}, which is not a master row.")
            lo, hi = master[key]['PRIOR_RANGE']
            if not (np.isfinite(value) and lo <= value <= hi):
                raise ValueError(f"SimulationRDS.conditions: {t} declares {key} = {value!r} outside its box [{lo}, {hi}].")
        table = parameterization_of(t)
        if [e['KEY'] for e in table] != [k for k in MASTER_PARAMETER_KEYS if k not in declared]:
            raise ValueError(f"parameterization_of({t!r}) is not the master minus the declared rows.")
        raw_learnable = [e['KEY'] for e in parameterization_raw_of(t) if role_of(e) == 'learnable']
        if raw_learnable != [e['KEY'] for e in table]:
            raise ValueError(f"parameterization_raw_of({t!r}): its learnable rows {raw_learnable} differ from the "
                             f"condition's table {[e['KEY'] for e in table]}.")
    shared = list(MASTER_PARAMETER_KEYS[:-1])
    for t in tokens:
        if parameter_keys_of(t)[:len(shared)] != shared:
            raise ValueError(f"{t}: the eleven shared coordinates must keep the master positions 0-10 "
                             f"(got {parameter_keys_of(t)}).")
    # The decided declarations: MET-FAB fixes the allocation at 0 (eleven inferred coordinates),
    # MET-INLB infers it (twelve). Change the condition settings AND this guard together.
    if parameter_keys_of("FAB") != list(MASTER_PARAMETER_KEYS[:-1]) or parameter_keys_of("INLB") != list(MASTER_PARAMETER_KEYS):
        raise ValueError(f"the decided tables are eleven coordinates under FAB (the allocation declared 0) and "
                         f"twelve under INLB (got FAB {len(parameter_keys_of('FAB'))}, INLB "
                         f"{len(parameter_keys_of('INLB'))}). Change the condition settings AND this guard together.")
    if THETA_STANDARDIZATION not in ("independent", "structured", "none"):
        raise ValueError(f"THETA_STANDARDIZATION {THETA_STANDARDIZATION!r} is not an sbi z_score option.")


_validate_model_blocks()


def association_ratio_of(condition: str) -> float:
    """The declared association ratio R_ON of ``condition`` (``SimulationRDS.association_ratio_of``):
    0.0 means the condition has no association channel; a positive value scales lambda_ref."""
    return PARAMETERS.simulation.rds.association_ratio_of(condition)


def occupancy_of(condition: str) -> float:
    """Probe occupancy per subunit of the TRUE population under ``condition``
    (``SimulationRDS.occupancy_of``): INLB 0.0476, the equilibrium at 0.25 nM with K_D 5 nM;
    FAB 0.0148, derived from the declared Fab/InlB visibility ratio and the InlB anchor. It
    converts the inferred N_total to the retained population and sets the initial probe classes."""
    return PARAMETERS.simulation.rds.occupancy_of(condition)


def occupancy_source_of(condition: str) -> str:
    """'equilibrium', 'declared' or 'derived': how the condition's occupancy is set
    (``SimulationRDS.occupancy_source_of``)."""
    return PARAMETERS.simulation.rds.occupancy_source_of(condition)


def visibility_of(condition: str) -> float:
    """Probability that a subunit of the TRUE population is visible under ``condition``:
    occupancy x P(dye >= 1) (INLB 0.0238, FAB 0.0119 under the baseline labeling laws)."""
    return PARAMETERS.simulation.rds.visibility_of(condition)


def two_probe_share_of(condition: str) -> float:
    """The declared independent-binding share of two-probe dimers among the REFERENCE dimers,
    s_2 = p / (2 - p) (``SimulationRDS.two_probe_share_of``): INLB 0.0244, FAB 0.0074. It is the
    share Fab labeling draws its one-/two-Fab classes with; under InlB the starting B2 share
    also carries the allocation and is read from the starting counts, not from s_2."""
    return PARAMETERS.simulation.rds.two_probe_share_of(condition)


# =============================================================================
# Helpers
# =============================================================================

def parameter_find(key: str, condition: str) -> int:
    """Return the index of a learnable parameter of ``condition`` by KEY.

    Use this to index into the condition's theta vectors (which contain only its learnable
    parameters). For all-parameter access (including 'Known' / 'Hyper'), use
    PARAMETER_RAW_FIND. Raises KeyError naming the condition's learnable parameters and, for a
    row the condition declares as a constant, saying so (``declared_row_values``).
    """
    index = parameter_index_of(condition)
    if key not in index:
        declared = declared_row_values(condition)
        if key in declared:
            raise KeyError(
                f"Parameter {key!r} is a declared constant under {condition} ({declared[key]!r}), not a "
                f"theta coordinate; read it from declared_row_values({condition!r}).")
        raise KeyError(
            f"Parameter {key!r} is not a learnable parameter under {condition}. "
            f"Learnable parameters: {list(index)}."
        )
    return index[key]


def theta_lower_bound(condition: str) -> list[float]:
    """Lower bounds of the condition's prior box in estimator space (log10 for log rows, the value
    for linear rows)."""
    return [para['PRIOR_RANGE'][0] for para in parameterization_of(condition)]


def theta_upper_bound(condition: str) -> list[float]:
    """Upper bounds of the condition's prior box in estimator space (log10 for log rows, the value
    for linear rows)."""
    return [para['PRIOR_RANGE'][1] for para in parameterization_of(condition)]


def build_prior(condition: str, device: str = "cpu") -> BoxUniform:
    """Construct the BoxUniform prior over the condition's learnable parameters, in estimator
    space (eleven dimensions under MET-FAB, twelve under MET-INLB).

    Sampled theta live in estimator space: log10 for log rows (a log-uniform prior on the
    physical value) and the value itself for linear rows (a uniform prior). Consumers map
    to physical values with ``to_physical`` -- never with a blanket ``10 ** theta``.
    """
    return BoxUniform(
        low=torch.tensor(theta_lower_bound(condition)),
        high=torch.tensor(theta_upper_bound(condition)),
        device=device,
    )
