"""Detector-calibration parameter spec (value-based role scheme).

This module is the parameter contract for the Detector calibration workflow — a
special-situation entry point that infers the diffraction-limited-imaging (DLI)
model with the reaction-diffusion biology marginalized over its full prior: the
detector re-images the shared reactive trajectory tier the biology prior generates
once, so the twelve reaction-diffusion parameters are a nuisance SUPPLIED by that tier,
never drawn here. It is deliberately DECOUPLED
from the canonical ``parameterization.py``: the detector-calibration system is
similar to but distinct from the production system, its parameter roles differ
(the imaging parameters are inferred here, marginalized as a nuisance there), and its ranges differ
by design. Code duplication with the canonical module is intentional.

Value-based role scheme
-----------------------
The canonical scheme keys a parameter's role solely on ``PRIOR_RANGE`` (a tuple
means learnable, ``None`` means fixed). To express learnable, fixed, nuisance,
and (future) posterior-drawn parameters from one table, the role selector here
moves to the ``VALUE`` field via two reserved string sentinels:

    VALUE                 PRIOR_RANGE     role
    -------------------   -------------   ------------------------------------
    concrete (num/list)   (low, high)     learnable   (VALUE is the prior center)
    concrete (num/list)   None            fixed       (constant, read directly)
    "NUISANCE"            (low, high)     nuisance_spec   (inline BoxUniform)
    "NUISANCE"            None            nuisance_object (supplied distribution)
    "POSTERIOR"           None            posterior       (future multiround)
    "POSTERIOR"           (low, high)     undefined -> rejected at import

Three design constraints (enforced at import by ``_validate_table``):
  1. Role dispatch is sentinel-based (``VALUE in {NUISANCE, POSTERIOR}``); it
     never tests whether ``VALUE`` is numeric, so a list-valued fixed parameter
     would be classified correctly should one exist.
  2. The learnable subset is ``VALUE-not-a-sentinel AND PRIOR_RANGE-not-None``.
     A nuisance-from-spec row also carries a range, so a ``PRIOR_RANGE is not
     None`` test alone (the canonical filter) would pull nuisance rows into the
     inference prior. That is the single most important port hazard (the production port to the canonical codebase).
  3. Log semantics are explicit: every ranged row is log10 (``LOG_FLAG=True``,
     ``LOG_BASE=10``), so a draw from a range lives in log10 space and maps to
     physical space by ``LOG_BASE ** draw`` (``to_physical``). Fixed values are
     already physical.

Camera parameters and the SCOPE nuisance
----------------------------------------
The five EMCCD camera parameters (``gamma``, ``kappa_o``, ``kappa_b``, ``kappa_s``,
``kappa_q``) are not identifiable from the videos and are marginalized as the SCOPE
camera nuisance rather than inferred (DETECTOR_WORKFLOW.md sec. 9.3): the videos
identify the gain and conversion only through their ratio ``gamma = kappa_g / kappa_c``,
and only the product ``gamma * kappa_q`` sets the amplitude, so inferring these
splits the brightness amplitude with ``mu_pc`` instead of constraining it. They are
drawn from their a-priori boxes at the DLI stage and recorded as ``Nuisance_SCOPE``;
the EM gain ``kappa_g`` and conversion ``kappa_c`` are retained as FIXED spec metadata
(from the MET acquisition configuration; audit-pending) so the drawn ``gamma`` can be
checked against the nominal ratio ``kappa_g / kappa_c`` (a drift check). Per-row
provenance is in each entry's ``NOTE``; the full noise model is in
REFERENCE_EMCCD_NOISE_MODEL.md.

Public interface
----------------
    NUISANCE_SENTINEL, POSTERIOR_SENTINEL
    DETECTOR_PARAMETERIZATION_RAW   -- flat list of all entries (full spec)
    DETECTOR_PARAMETERIZATION       -- learnable subset (the inference prior / theta)
    DETECTOR_PARAMETER_KEYS         -- ordered learnable keys (the theta schema; load-guard)
    DETECTOR_NUISANCE               -- RDS biology nuisance subset (nuisance-from-object: supplied by the shared RDS tier)
    DETECTOR_NUISANCE_SCOPE         -- SCOPE camera nuisance subset (drawn at the DLI stage)
    DETECTOR_IMAGING                -- full imaging vector: learnable + SCOPE (the render contract)
    DETECTOR_IMAGING_KEYS / DETECTOR_SCOPE_KEYS   -- ordered imaging / SCOPE keys
    DETECTOR_RAW_FIND / DETECTOR_FIND  -- KEY -> index maps
    role_of(entry)                  -- value-based role of a single entry
    detector_find(key)              -- learnable-parameter index (for theta vectors)
    to_physical(draw, entry)        -- map a (log10) draw to physical space
    build_prior(device)             -- BoxUniform over the learnable imaging params
    theta_lower_bound / theta_upper_bound             -- learnable log10 bounds
    flag_out_of_bounds(theta, low, high)              -- flag/measure learnable values outside the prior box
    scope_lower_bound / scope_upper_bound             -- SCOPE camera-nuisance log10 bounds

Assembling the eleven-key imaging vector the renderer (``render_dli_video``) consumes
is finalized against its call site in the shared DLI runner, where ``to_physical`` and
the subset/index maps here are the building blocks.
"""

import dataclasses

import numpy as np
import torch
from sbi.utils import BoxUniform

# Detector-generated data namespaces separately from canonical data by carrying
# this qualifier in the runtime prefix (a C28 stage token right after the iter),
# e.g. SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_5S_50FPS_Theta_Set_TASK_0_TRAIN.zarr.
DETECTOR_ALIAS_SUFFIX = "DETECTOR"

# Reserved VALUE sentinels that move a parameter's role off PRIOR_RANGE.
NUISANCE_SENTINEL = "NUISANCE"
POSTERIOR_SENTINEL = "POSTERIOR"
_SENTINELS = (NUISANCE_SENTINEL, POSTERIOR_SENTINEL)


# =============================================================================
# Detector parameter spec
# =============================================================================
#
# Fields per entry (a lean subset of the canonical schema; ROLE is computed from
# VALUE x PRIOR_RANGE, never stored):
#   KEY          unique identifier (str)
#   VALUE        prior center for a learnable row (VALUE = LOG_BASE**mid(range));
#                the physical constant for a fixed row (scalar or list);
#                a sentinel string for a nuisance/posterior row
#   PRIOR_RANGE  (low, high) log10 bounds for a ranged row; None otherwise
#   LOG_FLAG     True for every ranged row (log10); None for fixed rows
#   LOG_BASE     10 for every ranged row; None for fixed rows
#   UNIT         human-readable unit of the parameter as sampled
#   LABEL        LaTeX label for plotting
#   NOTE         optional free-text provenance/description (present where a
#                parameter needs explanation, e.g. the camera chain); ignored by
#                role dispatch and every builder
#
# The "op" comment on each learnable imaging row is the production operating
# point (the value used as a fixed constant in the canonical model); every
# learnable range brackets it, so calibration can only refine, never contradict
# by construction, that operating point (see the learnable imaging-parameter ranges in DETECTOR_WORKFLOW.md).

_DETECTOR_RAW_NESTED: dict[str, list[dict]] = {
    # ----- RDS nuisance: biology marginalized during detector calibration -----
    # Nuisance-from-object (VALUE = NUISANCE, PRIOR_RANGE = None): the twelve reaction-diffusion
    # parameters are SUPPLIED by the shared RDS trajectory tier, which the biology prior
    # (`parameterization.PARAMETERIZATION`) generates once and both workflows re-image at the
    # DLI stage -- the tier's `Theta_Set` is the detector's record of this nuisance. Nothing
    # is drawn here and no range is duplicated, so the detector marginalizes the biology
    # prior by construction; these rows declare the role and label the provenance tables.
    # The ranges live only in the biology table (DETECTOR_WORKFLOW.md sec. 6.1 points there).
    'stoichiometry': [
        {'KEY': 'count_total', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Count', 'LABEL': r'$N_{R}$',
         'NOTE': 'Conserved receptor-subunit total N_R = n_A + 2 n_B of the simulated patch. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'fraction_dimer_initial', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Dimensionless', 'LABEL': r'$x_{B}$',
         'NOTE': 'Requested initial fraction of receptors in dimers, x_B in [0, 1] (a LINEAR row in the biology table). RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'relative_rate_dimerization', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Dimensionless', 'LABEL': r'$R_{ON}$',
         'NOTE': 'Association ratio R_ON: lambda_on = R_ON * 6 D_A / r^2 (compatibility normalization, not a physical bound); one rate for all six association channels. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'rate_dissociation', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Count Per Second', 'LABEL': r'$\kappa_{OFF}$',
         'NOTE': 'Dimer unbinding rate kappa_OFF (B_m -> A_m + A_m, every mode), 1/s. Under the labeling model a dissociating one-dye dimer leaves one visible and one invisible daughter -- a signature the detector must see during calibration. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
    ],
    'mobility': [
        {'KEY': 'diffusivity_alp', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Square Micrometer Per Second', 'LABEL': r'$D_{A}$',
         'NOTE': 'Monomer scale coefficient D_A = D[A, fast]; every other coefficient is a ratio of it. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'relative_diffusivity_dimer', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Dimensionless', 'LABEL': r'$R_{B}$',
         'NOTE': 'Dimer factor within a mode: D[B, m] = R_B * D[A, m], 0 < R_B <= 1. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'relative_diffusivity_slow', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Dimensionless', 'LABEL': r'$R_{s}$',
         'NOTE': 'Slow-mode factor: D[X, s] = R_s * D[X, f]. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'relative_diffusivity_immobile', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Dimensionless', 'LABEL': r'$R_{i}$',
         'NOTE': 'Immobile-mode factor: D[X, i] = R_i * D[X, f]; a resolution floor, not zero. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'rate_fast_slow', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Count Per Second', 'LABEL': r'$k_{fs}$',
         'NOTE': 'Mobility switching fast -> slow, shared by both species, 1/s. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'rate_slow_fast', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Count Per Second', 'LABEL': r'$k_{sf}$',
         'NOTE': 'Mobility switching slow -> fast, shared by both species, 1/s. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'rate_slow_immobile', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Count Per Second', 'LABEL': r'$k_{si}$',
         'NOTE': 'Mobility switching slow -> immobile, shared by both species, 1/s. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
        {'KEY': 'rate_immobile_slow', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Count Per Second', 'LABEL': r'$k_{is}$',
         'NOTE': 'Mobility switching immobile -> slow, shared by both species, 1/s. RDS nuisance supplied by the shared trajectory tier (the biology prior; development ranges in parameterization.py).'},
    ],
    # ----- Fixed geometry -----
    'geometry': [
        {'KEY': 'capture_radius', 'VALUE': 10, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Nanometer', 'LABEL': r'$\rho_{CAP}$',
         'NOTE': 'Smoluchowski reaction radius rho_CAP (nm) = particle_diameter_nm = 2*monomer_radius (center-to-center contact of two monomers). Drives kappa_ON = 4*pi*D_R*rho_CAP, the capture volume V_CAP ~ rho_CAP^3, and the fusion/fission distances. Active in the reactive detector system exactly as in biology; build_system derives the value from PARAMETERS.simulation.stem.particle_diameter_nm.'},
    ],
    # ----- Learnable imaging parameters (calibration targets) -----
    # Ranges from the learnable-imaging-parameter section of DETECTOR_WORKFLOW.md; VALUE = 10**mid(range) (center).
    'camera': [  # EMCCD detector camera chain (REFERENCE_EMCCD_NOISE_MODEL.md): gamma, kappa_o, kappa_b, kappa_s, kappa_q marginalized as the SCOPE camera nuisance (non-identifiable; DETECTOR_WORKFLOW.md sec. 9.3); kappa_g, kappa_c fixed nominal spec metadata (gamma = kappa_g/kappa_c).
        {'KEY': 'gamma', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.62, 1.625), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU Per Electron', 'LABEL': r'$\gamma$',
         'NOTE': 'Gain-conversion ratio gamma = kappa_g/kappa_c (ADU per photoelectron) -- the only gain quantity the videos identify. Marginalized as a SCOPE camera nuisance: inferring it splits the peak-ADU amplitude with mu_pc (only gamma*kappa_q is identifiable), so it is drawn from its a-priori box rather than treated as a calibration target (DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: kappa_g/kappa_c = 200/4.78 = 41.84, identical across all MET cells (both Fab and InlB camera protocols); box [41.7, 42.2].'},
        {'KEY': 'kappa_o', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.455, 1.465), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Photon', 'LABEL': r'$\kappa_{o}$',
         'NOTE': 'Optical background offset: incident photons per pixel per frame, pre-gain, one scalar per movie; amplified by gamma*kappa_q to set the ADU floor. Marginalized as a SCOPE camera nuisance (non-identifiable; DETECTOR_WORKFLOW.md sec. 9.3). Reference = ThunderSTORM offset[photon] median, a condition-independent background: Fab 28.9 / InlB 28.6 (pooled 28.7). Narrow box [28.5, 29.2] around the measured value: a broad offset lets the gain*offset floor dominate the video-to-video variation (detector-embedding collapse).'},
        {'KEY': 'kappa_b', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (2.24, 2.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU', 'LABEL': r'$\kappa_{b}$',
         'NOTE': 'Camera baseline: post-gain ADU constant, added last. Marginalized as a SCOPE camera nuisance (non-identifiable; DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: MET configured baseline = 175, identical across all cells (both conditions); box [173.8, 177.8].'},
        {'KEY': 'kappa_s', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.02, 1.025), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU', 'LABEL': r'$\kappa_{s}$',
         'NOTE': 'Read noise: post-register Gaussian sigma (ADU). Marginalized as a SCOPE camera nuisance: it does not dominate the SNR (the EM register amplifies signal above the read-noise floor) and is only weakly identifiable (DETECTOR_WORKFLOW.md sec. 9.3). Reference = camera datasheet ~10.5 ADU; box [10.5, 10.6] (weakly identifiable but pinned tight to the datasheet value, like the other camera constants).'},
        {'KEY': 'kappa_q', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-0.05, -0.04), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\kappa_{q}$',
         'NOTE': 'Quantum efficiency, applied once in the Poisson step. Marginalized as a SCOPE camera nuisance: only the product gamma*kappa_q is identifiable from the videos (DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: MET quantumEfficiency = 0.90, identical across all cells; box [0.89, 0.91].'},
        {'KEY': 'kappa_g', 'VALUE': 200, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'LABEL': r'$\kappa_{g}$',
         'NOTE': 'Nominal EM gain g from the MET acquisition config (ThunderSTORM camera protocol; audit-pending). Not inferred -- kept as spec metadata so the drawn gamma (SCOPE nuisance) can be checked against kappa_g/kappa_c (drift check).'},
        {'KEY': 'kappa_c', 'VALUE': 4.78, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Electron Per ADU', 'LABEL': r'$\kappa_{c}$',
         'NOTE': 'Nominal conversion C (e-/ADU) from the MET acquisition config (photons2ADU field; audit-pending). Not inferred -- kept as spec metadata for the gamma drift check. gamma = kappa_g / kappa_c.'},
    ],
    'psf': [  # PSF widths (mu_r, sigma_r) + emitter brightness (mu_pc, sigma_pc)
        {'KEY': 'mu_r', 'VALUE': 10**0.15, 'PRIOR_RANGE': (0.0, 0.3), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\mu_{r}$',
         'NOTE': 'Median of the per-emitter Gaussian PSF-width distribution (sqrt(2)*sigma, in pixels). Learnable; the model infers a distribution over the PSF width. Reference = monomer-control Fab median 1.36 (sqrt(2)*sigma[nm]/158); InlB 1.47 is dimer-broadened -- two labels within one diffraction spot fit wider -- so Fab is adopted (DETECTOR_WORKFLOW.md sec. 6.5 caveat 3). Prior [1.0, 2.0], excluding implausibly wide PSFs while staying general.'},
        {'KEY': 'sigma_r', 'VALUE': 10**(-0.625), 'PRIOR_RANGE': (-1.0, -0.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\sigma_{r}$',
         'NOTE': 'Log-spread of the per-emitter PSF-width distribution. Learnable; with mu_r it parameterizes the lognormal PSF-width population. The ThunderSTORM fitted spread (Fab 0.37 / InlB 0.42) is upper-biased by per-localization fit error (errors-in-variables, sec. 6.5 caveat 1); the fixed-imaging histogram check indicates the fit-corrected population spread is ~0.15. Prior broad-but-capped [0.10, 0.56]: broad enough to contain the fitted 0.37, capped so the narrow-spot tail does not manufacture unphysically bright pixels. See DETECTOR_WORKFLOW.md.'},
        {'KEY': 'mu_pc', 'VALUE': 10**2.375, 'PRIOR_RANGE': (2.0, 2.75), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\mu_{pc}$',
         'NOTE': 'Median of the per-emitter (monomer) brightness (photon-count) parent distribution. Learnable. Reference = monomer-control Fab median 386 photons; InlB 690 is NOT a wider monomer brightness but the sum of two co-located labels on a dimer (localization artifact, sec. 6.5 caveat 3), which the model builds as the sum of two draws from this monomer parent (sec. 6.4) -- so mu_pc is monomer-scoped and NOT widened to the dimer value. Prior [100, 562].'},
        {'KEY': 'sigma_pc', 'VALUE': 10**(-0.375), 'PRIOR_RANGE': (-0.75, 0.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\sigma_{pc}$',
         'NOTE': 'Log-spread of the per-emitter (monomer) brightness parent distribution. Learnable; with mu_pc it sets the lognormal brightness population. The ThunderSTORM fitted spread (Fab 0.61 / InlB 0.55) is upper-biased by fit error (sec. 6.5 caveat 1); the fixed-imaging histogram check indicates ~0.5. Prior broad-but-capped [0.178, 1.0]: contains the fitted 0.61, capped to exclude the heavy-brightness-tail regime. See DETECTOR_WORKFLOW.md.'},
    ],
    'transitivity': [  # brightness photo-physics (stationary OU ln-brightness flicker + absorbing bleach)
        {'KEY': 'delta_frame', 'VALUE': 0.020, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Second', 'LABEL': r'$\delta_{f}$',
         'NOTE': 'Camera frame interval (s); 0.020 = 50 FPS. Fixed acquisition constant. Duration-general: n_frames is supplied per run, this is only the per-frame time.'},
        {'KEY': 'numb_photo_bleach', 'VALUE': 100, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'LABEL': r'$\psi_{pb}$',
         'NOTE': 'Reference frame window normalizing prob_photo_bleach (NOT the video length): p_video = 1 - (1 - prob_photo_bleach)^(n_frames/numb_photo_bleach). Fixed = 100.'},
        {'KEY': 'prob_photo_bleach', 'VALUE': 10**(-1.25), 'PRIOR_RANGE': (-2.0, -0.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\rho_{pb}$',
         'NOTE': 'Probability an emitter enters the absorbing bleached state over numb_photo_bleach (100) frames. Learnable photophysics target; drawn from [10^-2, 10^-0.5].'},
        {'KEY': 'lambda_rate', 'VALUE': 10**0.5, 'PRIOR_RANGE': (0.0, 1.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\lambda$',
         'NOTE': 'Correlation-decay rate of the stationary OU ln-brightness flicker: ACF(lag) = exp(-lambda_rate * lag), so tau_corr = 1/lambda_rate (generate_brightness_photons). Learnable; drawn from [10^0, 10^1] = [1, 10]. Reference ~5 (log10 ~0.7), shape-matched to the flicker correlation-time of the MET track intensity[photon] autocorrelation (tau_corr ~0.135 s; the bare 1/tau_corr ~7 is biased high by per-track detrending -- see DETECTOR_WORKFLOW.md sec. 6.3 and the Flicker_Rate_Derivation utility).'},
    ],
}


# =============================================================================
# Role resolution (value-based dispatch)
# =============================================================================

def role_of(entry: dict) -> str:
    """Value-based role of a single parameter entry.

    Returns one of 'learnable', 'fixed', 'nuisance_spec', 'nuisance_object',
    'posterior'. Dispatch is sentinel-based and never tests whether ``VALUE`` is
    numeric (constraint 1), so a list-valued fixed parameter is classified
    correctly. ``POSTERIOR`` with a range is undefined and raises.
    """
    value, prior_range = entry['VALUE'], entry['PRIOR_RANGE']
    is_sentinel = isinstance(value, str) and value in _SENTINELS
    if value == POSTERIOR_SENTINEL:
        if prior_range is not None:
            raise ValueError(
                f"parameter {entry['KEY']!r}: POSTERIOR with a PRIOR_RANGE is "
                f"undefined (a posterior draw carries its own support). Set "
                f"PRIOR_RANGE=None for a posterior-drawn parameter."
            )
        return 'posterior'
    if value == NUISANCE_SENTINEL:
        return 'nuisance_spec' if prior_range is not None else 'nuisance_object'
    # Concrete value: learnable iff it also carries a range (constraint 2).
    if not is_sentinel and prior_range is not None:
        return 'learnable'
    return 'fixed'


def _validate_table(raw: list[dict]) -> None:
    """Enforce the three design constraints at import time (fail-fast)."""
    seen = set()
    for entry in raw:
        key = entry['KEY']
        if key in seen:
            raise ValueError(f"duplicate parameter KEY {key!r} in the detector table.")
        seen.add(key)
        role = role_of(entry)  # raises on the undefined POSTERIOR+range cell
        prior_range = entry['PRIOR_RANGE']
        if prior_range is not None:
            # Constraint 3: every ranged row is log10.
            if not (entry['LOG_FLAG'] is True and entry['LOG_BASE'] == 10):
                raise ValueError(
                    f"parameter {key!r}: a ranged row must be log10 "
                    f"(LOG_FLAG=True, LOG_BASE=10); got LOG_FLAG={entry['LOG_FLAG']!r}, "
                    f"LOG_BASE={entry['LOG_BASE']!r}."
                )
            low, high = prior_range
            if not low < high:
                raise ValueError(f"parameter {key!r}: PRIOR_RANGE low {low} !< high {high}.")
            if role == 'learnable':
                # VALUE = LOG_BASE ** mid(range) invariant (linear-space center).
                center = entry['LOG_BASE'] ** ((low + high) / 2)
                if not abs(entry['VALUE'] - center) < 1e-9 * max(1.0, center):
                    raise ValueError(
                        f"parameter {key!r}: learnable VALUE {entry['VALUE']} is not the "
                        f"prior center {center} (= {entry['LOG_BASE']}**{(low + high) / 2})."
                    )
        else:
            # Fixed / nuisance-object / posterior rows carry no log metadata.
            if not (entry['LOG_FLAG'] is None and entry['LOG_BASE'] is None):
                raise ValueError(
                    f"parameter {key!r}: a non-ranged row must have LOG_FLAG=None, "
                    f"LOG_BASE=None."
                )


# =============================================================================
# Flat list, subsets, and index maps
# =============================================================================

DETECTOR_PARAMETERIZATION_RAW: list[dict] = [
    entry for group in _DETECTOR_RAW_NESTED.values() for entry in group
]

_validate_table(DETECTOR_PARAMETERIZATION_RAW)

# Learnable subset (the inference prior + theta columns): VALUE-not-a-sentinel
# AND PRIOR_RANGE-not-None (constraint 2).
DETECTOR_PARAMETERIZATION: list[dict] = [
    entry for entry in DETECTOR_PARAMETERIZATION_RAW if role_of(entry) == 'learnable'
]

# Ordered learnable-parameter keys = the theta-vector schema. Pass to
# `artifacts.load_estimator(expected_parameter_keys=...)` / `assert_schema_compatible`
# to hard-reject a legacy estimator whose parameter schema differs (equal-length theta
# vectors would otherwise be misread column-for-column).
DETECTOR_PARAMETER_KEYS: list[str] = [entry['KEY'] for entry in DETECTOR_PARAMETERIZATION]

# The nuisance parameters form two blocks with different media (DETECTOR_WORKFLOW.md
# sec. 7 / 9.3): the RDS biology (nuisance-from-object, supplied by the shared RDS tier,
# whose twelve-parameter Theta_Set is its record) and the SCOPE camera (nuisance-from-spec,
# drawn at the DLI stage and recorded as Nuisance_SCOPE). They are grouped by the
# nested-dict category, so flipping the camera rows to a nuisance does not pull them
# into the RDS block.
_RDS_NUISANCE_GROUPS = ('stoichiometry', 'mobility')
_SCOPE_NUISANCE_GROUPS = ('camera',)

# RDS biology nuisance subset (marginalized during calibration; supplied by the shared RDS tier).
DETECTOR_NUISANCE: list[dict] = [
    entry for group in _RDS_NUISANCE_GROUPS for entry in _DETECTOR_RAW_NESTED[group]
    if role_of(entry) in ('nuisance_spec', 'nuisance_object')
]

# SCOPE camera nuisance subset (marginalized in both workflows; drawn at the DLI stage).
DETECTOR_NUISANCE_SCOPE: list[dict] = [
    entry for group in _SCOPE_NUISANCE_GROUPS for entry in _DETECTOR_RAW_NESTED[group]
    if role_of(entry) == 'nuisance_spec'
]

# The full imaging vector the DLI renderer consumes: the learnable inference targets
# followed by the SCOPE camera nuisance. Render reads by KEY, so this fixed order is
# the persistence/assembly convention shared by the DLI stage and its support module.
DETECTOR_IMAGING: list[dict] = DETECTOR_PARAMETERIZATION + DETECTOR_NUISANCE_SCOPE
DETECTOR_IMAGING_KEYS: list[str] = [entry['KEY'] for entry in DETECTOR_IMAGING]
DETECTOR_SCOPE_KEYS: list[str] = [entry['KEY'] for entry in DETECTOR_NUISANCE_SCOPE]

DETECTOR_RAW_FIND: dict[str, int] = {
    entry['KEY']: index for index, entry in enumerate(DETECTOR_PARAMETERIZATION_RAW)
}
DETECTOR_FIND: dict[str, int] = {
    entry['KEY']: index for index, entry in enumerate(DETECTOR_PARAMETERIZATION)
}


# =============================================================================
# Helpers
# =============================================================================

def detector_find(key: str) -> int:
    """Index of a learnable imaging parameter by KEY (for theta-vector indexing)."""
    if key not in DETECTOR_FIND:
        raise KeyError(
            f"Parameter {key!r} is not a learnable detector parameter. "
            f"Learnable parameters: {list(DETECTOR_FIND.keys())}."
        )
    return DETECTOR_FIND[key]


def to_physical(draw, entry: dict):
    """Map a sampled ``draw`` for ``entry`` to physical space.

    A ranged row is sampled in log10 space, so its physical value is
    ``LOG_BASE ** draw``; a fixed row is already physical and returned as-is.
    Accepts scalars or tensors (``LOG_BASE ** draw`` broadcasts).
    """
    if entry['PRIOR_RANGE'] is None:
        return entry['VALUE']
    return entry['LOG_BASE'] ** draw


def theta_lower_bound() -> list[float]:
    """Lower bounds of the learnable imaging prior, in log10 space."""
    return [entry['PRIOR_RANGE'][0] for entry in DETECTOR_PARAMETERIZATION]


def theta_upper_bound() -> list[float]:
    """Upper bounds of the learnable imaging prior, in log10 space."""
    return [entry['PRIOR_RANGE'][1] for entry in DETECTOR_PARAMETERIZATION]


def flag_out_of_bounds(theta_log10, low=None, high=None):
    """Flag learnable-parameter values outside the prior box (log10 space).

    Args:
        theta_log10: array-like of log10 parameter values; the last axis is the
            parameter axis (length == the number of learnable imaging parameters).
        low, high: prior bounds in log10 space; default to ``theta_lower_bound()``
            / ``theta_upper_bound()``.

    Returns:
        ``(out_of_bounds, signed_margin)``, both ``numpy`` arrays shaped like
        ``theta_log10``. ``out_of_bounds`` is a boolean mask (True where a value is
        below ``low`` or above ``high``). ``signed_margin`` is the signed distance
        outside the box, in log10 units: negative below the lower bound, positive
        above the upper bound, and 0 inside — so its magnitude is how far
        out-of-prior a value sits. Used to flag — never silently clip — MAP
        estimates that drift past a prior edge (the seed-then-optimize step is
        unconstrained). Inputs are
        assumed finite (a MAP estimate always is); a NaN would be reported as
        out-of-bounds with a NaN margin.
    """
    theta = np.asarray(theta_log10, dtype=float)
    lo = np.asarray(theta_lower_bound() if low is None else low, dtype=float)
    hi = np.asarray(theta_upper_bound() if high is None else high, dtype=float)
    below = np.minimum(theta - lo, 0.0)   # < 0 only where theta < lo
    above = np.maximum(theta - hi, 0.0)   # > 0 only where theta > hi
    signed_margin = below + above          # at most one term is non-zero
    return signed_margin != 0.0, signed_margin


def scope_lower_bound() -> list[float]:
    """Lower bounds of the SCOPE camera-nuisance box, in log10 space."""
    return [entry['PRIOR_RANGE'][0] for entry in DETECTOR_NUISANCE_SCOPE]


def scope_upper_bound() -> list[float]:
    """Upper bounds of the SCOPE camera-nuisance box, in log10 space."""
    return [entry['PRIOR_RANGE'][1] for entry in DETECTOR_NUISANCE_SCOPE]


def build_prior(device: str = "cpu") -> BoxUniform:
    """BoxUniform log-uniform prior over the learnable imaging parameters.

    Sampled values are in log10 space; map to physical via ``to_physical`` (i.e.
    ``10 ** theta``), exactly as the canonical prior convention.
    """
    return BoxUniform(
        low=torch.tensor(theta_lower_bound()),
        high=torch.tensor(theta_upper_bound()),
        device=device,
    )


def detector_paths(canonical_paths):
    """Return a copy of the canonical `Paths` whose `project_alias` carries the
    Detector qualifier, so every canonical path pattern namespaces Detector data
    separately (e.g. `SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_<timing>_...`).

    Takes the canonical `Paths` as an argument (rather than importing the machine
    profile) so this module stays decoupled from `parameterization`/`PARAMETERS`.
    """
    return dataclasses.replace(
        canonical_paths,
        project_alias=f"{canonical_paths.project_alias}_{DETECTOR_ALIAS_SUFFIX}",
        # The detector's Theta_Set holds the six imaging labels drawn at the DLI stage, so
        # it is a qualified, per-condition product (unlike the shared RDS tier's Theta_Set,
        # which `rds_alias` keeps under the bare sibling alias for both workflows).
        theta_set_is_rds_product=False,
    )
