"""Information budget: what a recording can support, independently of any estimator.

Pure analytic kernels. Every function here takes numbers and returns numbers; none reads
a file, resolves a machine profile, or prints.

An estimator that misses a parameter can be failing for two very different reasons: it may
be a poor estimator, or the recordings may not carry the information. Those call for
opposite responses -- improve the estimator, or stop trying to infer the parameter -- and
measured accuracy alone cannot tell them apart. This module computes the second quantity:
a Cramer-Rao lower bound on the standard deviation of ANY unbiased estimator of each
imaging parameter, from one recording of a given size. Comparing three numbers then grades
the situation:

    bound    what the data allow, computed here
    direct   what the direct estimators achieve, measured
    neural   the posterior width of the amortized flow, measured

A parameter whose neural posterior already sits at the bound is being inferred as well as
it can be, and a wider inferred block will not help it. A parameter far from the bound has
recoverable headroom. A parameter whose bound is itself wider than its prior is not
identifiable from one recording at all, and belongs outside the inferred block whatever
estimator is used. This is the quantitative criterion behind `DETECTOR_WORKFLOW.md` sec. 9.4,
which is a proposal and not in force.

Scope and honesty about the approximations. These are bounds under a Gaussian approximation
to the EMCCD likelihood, using the exact variance law of the Poisson-Gamma-Normal chain,

    mean(ADU) = gamma * kappa_q * I + kappa_b
    var(ADU)  = 2 * gamma^2 * kappa_q * I + kappa_s^2

verified numerically against the renderer. The approximation is standard in localization
microscopy and is accurate where a pixel collects more than a few photoelectrons, which
holds throughout the prior box here. Three effects are modeled explicitly because they
dominate and are easy to get wrong:

  - the FACTOR 2 excess noise of electron multiplication, which halves the effective
    photon count relative to an ideal detector;
  - the pixel integration, handled by numerically differentiating the same pixel-integrated
    Gaussian the renderer uses, rather than by a continuum approximation;
  - the CORRELATION of the brightness flicker in time, which reduces the number of
    effectively independent frames and costs a constant factor 5.6 in any decay-rate
    standard deviation, at the center of the lambda_rate prior;
  - the DEGENERACY of a shallow decay with its own unknown amplitude and offset, which costs
    between 2.7x at the top of the prob_photo_bleach prior and 150x at the bottom, and is
    therefore the dominant term everywhere except the very top.

What is NOT modeled, and therefore makes every bound here optimistic: emitters leaving and
entering the field, reactions changing a spot's multiplicity mid-recording, spot overlap,
and the truncation of the observable population by detectability. A real estimator faces
all four. These bounds are a floor on the achievable error, not a prediction of it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "FIELD_RELATIVE_NOISE_MET_FAB",
    "pixel_variance_adu",
    "spot_fisher_information",
    "crb_log_width_one_spot",
    "crb_population_lognormal",
    "effective_sample_size_ar1",
    "crb_decay_rate",
    "crb_prob_bleach_dex",
    "crb_ar1_rate",
    "crb_lambda_rate_dex",
]


# Relative noise of the WHOLE-FIELD fluorescence curve, per frame, at the MET-FAB emitter
# density. Defined here so the budget utility and the fluorescence-loss estimator cannot
# drift apart: they quoted 0.018 and 0.028 independently at one point, which silently made
# the published identifiability table disagree with the tool that produced it.
#
# Derivation: the per-frame field noise is dominated by the optical background over the
# 256 x 256 grid, sqrt(n_px * (2 gamma^2 kappa_q kappa_o + kappa_s^2)) ~ 7.7e4 ADU, against an
# emitter signal of about 154 spots x 2.03 dyes x mu_pc x gamma x kappa_q ~ 2.8e6 ADU. The
# value is SPECIFIC to that emitter density: a single-dye scene of 150 emitters measures
# 0.058 instead, and any bound quoted for such a scene must use that figure.
FIELD_RELATIVE_NOISE_MET_FAB = 0.028


# =============================================================================
# The EMCCD variance law
# =============================================================================

def pixel_variance_adu(mean_adu: np.ndarray, gamma: float, kappa_b: float,
                       kappa_s: float) -> np.ndarray:
    """Variance of a pixel, in ADU squared, given its mean.

    ``var = 2 * gamma * (mean - kappa_b) + kappa_s^2``. The factor 2 is the excess-noise
    factor of the Gamma electron-multiplication stage: multiplication doubles the variance
    per detected photoelectron, so an EMCCD pixel carries the noise of half as many photons
    as its count suggests. Read noise adds in quadrature after the register and is
    gain-independent.

    Args:
        mean_adu: pixel mean in ADU, any shape.
        gamma: ADU per photoelectron.
        kappa_b: electronic bias in ADU.
        kappa_s: read-noise standard deviation in ADU.

    Returns:
        Variance in ADU squared, same shape as ``mean_adu``.
    """
    net = np.maximum(np.asarray(mean_adu, dtype=np.float64) - float(kappa_b), 0.0)
    return 2.0 * float(gamma) * net + float(kappa_s) ** 2


# =============================================================================
# Per-spot: how well can one spot's width be measured?
# =============================================================================

def spot_fisher_information(amplitude_photons: float, sigma_px: float, *,
                            gamma: float, kappa_q: float, kappa_o: float,
                            kappa_b: float, kappa_s: float,
                            half_px: int = 14, step: float = 1e-4) -> np.ndarray:
    """Fisher information for one spot's ``(amplitude, x, y, sigma)``.

    Built by numerically differentiating the pixel-integrated Gaussian the renderer uses,
    so the pixelation is exact rather than approximated by a continuum integral. Under the
    Gaussian approximation with a mean-dependent variance, the information is

        I_ab = sum_pixels  (d mean / d theta_a)(d mean / d theta_b) / var
                         + (1/2) (d var / d theta_a)(d var / d theta_b) / var^2

    Both terms are included; the second is small wherever the background dominates, but it
    is not zero and costs nothing to carry.

    Args:
        amplitude_photons: total incident photons of the spot in one frame.
        sigma_px: PSF standard deviation in pixels (NOT ``sqrt(2) * sigma``).
        gamma, kappa_q, kappa_o, kappa_b, kappa_s: the SCOPE camera block.
        half_px: half-width of the pixel patch summed over.
        step: relative step for the numerical derivative.

    Returns:
        ``(4, 4)`` Fisher information matrix in the order ``(amplitude, x, y, sigma)``.
    """
    from .direct_imaging_estimates import pixel_gaussian_patch

    size = 2 * int(half_px)
    center = size / 2.0
    gq = float(gamma) * float(kappa_q)
    bg_adu = gq * float(kappa_o) + float(kappa_b)

    def mean_adu(amp, x0, y0, sig):
        # pixel_gaussian_patch returns amplitude * unit_integral + background, in the units
        # of `amplitude`; here amplitude is carried in ADU so the photon amplitude is scaled.
        return pixel_gaussian_patch(gq * amp, x0, y0, sig, bg_adu, size)

    theta = np.array([float(amplitude_photons), center, center, float(sigma_px)])
    base = mean_adu(*theta)
    var = pixel_variance_adu(base, gamma, kappa_b, kappa_s)

    derivs = []
    dvar = []
    for k in range(4):
        h = max(abs(theta[k]) * step, step)
        up, dn = theta.copy(), theta.copy()
        up[k] += h
        dn[k] -= h
        d_mean = (mean_adu(*up) - mean_adu(*dn)) / (2.0 * h)
        derivs.append(d_mean)
        # var depends on theta only through the mean, so d var / d theta = 2 gamma d mean / d theta
        dvar.append(2.0 * float(gamma) * d_mean)

    fisher = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            fisher[a, b] = float(np.sum(derivs[a] * derivs[b] / var)
                                 + 0.5 * np.sum(dvar[a] * dvar[b] / var ** 2))
    return fisher


def crb_log_width_one_spot(amplitude_photons: float, sigma_px: float, **camera) -> float:
    """Lower bound on the standard deviation of ``ln sigma`` from ONE spot in ONE frame.

    The full four-parameter matrix is inverted before the width entry is taken, so the bound
    accounts for the amplitude and the two center coordinates being unknown too -- which they
    are. Treating them as known would understate the bound substantially, because amplitude
    and width trade off directly.

    Args:
        amplitude_photons: total incident photons of the spot in one frame.
        sigma_px: PSF standard deviation in pixels.
        **camera: the SCOPE block, passed through to `spot_fisher_information`.

    Returns:
        ``sd(ln sigma)``, dimensionless. Returns ``inf`` if the information matrix is singular.
    """
    fisher = spot_fisher_information(amplitude_photons, sigma_px, **camera)
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        return float("inf")
    if cov[3, 3] <= 0:
        return float("inf")
    return float(np.sqrt(cov[3, 3]) / float(sigma_px))     # delta method: sd(ln s) = sd(s)/s


# =============================================================================
# Population: from many noisy widths to (mu_r, sigma_r)
# =============================================================================

def crb_population_lognormal(n_units: int, population_sd: float,
                             measurement_variance: float) -> dict:
    """Bounds on a log-normal population's scale and shape under measurement error.

    Each unit contributes ``ln w_i ~ Normal(ln mu, population_sd^2 + measurement_variance)``.
    The scale and shape are then the mean and the spread-with-a-known-offset-removed, and
    their bounds follow from the normal-sample results:

        sd(ln mu)  >=  sqrt((S) / n)
        sd(shape)  >=  S / (shape * sqrt(2 * (n - 1))),      S = shape^2 + v

    The second expression is the one that matters for this campaign. It DIVERGES as the
    population spread approaches zero: when a population is nearly uniform, the observed
    spread is almost all measurement noise, and the small remainder attributable to the
    population is bounded away from measurable. That is a property of the data, not of any
    estimator, and it predicts that ``sigma_r`` is hardest to recover exactly where it is
    smallest -- which is what the direct estimator shows.

    It also shows why linking matters. Averaging ``L`` repeat measurements of one unit
    replaces ``v`` by ``v / L``, which shrinks ``S`` toward ``shape^2`` and moves the bound
    toward its noise-free value.

    Args:
        n_units: number of independent units (spots, or linked tracks).
        population_sd: the true population spread of ``ln w`` (this is ``sigma_r``).
        measurement_variance: variance of ``ln w_hat`` about ``ln w`` for one unit.

    Returns:
        ``dict`` with ``sd_log_scale`` (a bound on ``sd(ln mu_r)``), ``sd_log_scale_dex``
        (the same in log10 units), ``sd_shape`` (a bound on ``sd(sigma_r)``), and
        ``total_variance`` (``S``).
    """
    n = int(n_units)
    shape = float(population_sd)
    total = shape ** 2 + float(measurement_variance)
    if n < 2 or total <= 0:
        return dict(sd_log_scale=float("inf"), sd_log_scale_dex=float("inf"),
                    sd_shape=float("inf"), total_variance=total)
    sd_log_scale = float(np.sqrt(total / n))
    sd_shape = (float("inf") if shape <= 0
                else float(total / (shape * np.sqrt(2.0 * (n - 1)))))
    return dict(sd_log_scale=sd_log_scale,
                sd_log_scale_dex=sd_log_scale / np.log(10.0),
                sd_shape=sd_shape, total_variance=total)


# =============================================================================
# Temporal parameters: bleaching and flicker
# =============================================================================

def effective_sample_size_ar1(n_frames: int, rho: float) -> float:
    """Effectively independent frames in a series whose noise is AR(1)-correlated.

    ``n_eff = n * (1 - rho) / (1 + rho)``, the standard variance-inflation result for the
    mean of a correlated series. It is easy to miss: the brightness flicker is an
    Ornstein-Uhlenbeck process with correlation time ``1 / lambda_rate``, so consecutive
    frames of a total-fluorescence curve are NOT independent samples of the decay. At the
    center of the ``lambda_rate`` prior, ``rho = 0.9387``, so a hundred-frame recording
    carries about 3.2 effectively independent samples of the decay rather than a hundred.

    The resulting inflation of a decay-rate standard deviation is ``sqrt(n / n_eff)``, which
    is ``sqrt((1 + rho) / (1 - rho))`` and therefore INDEPENDENT of the recording length: a
    constant factor 5.6 at the prior center, the same at 100 frames as at 1000. It is a fixed
    tax on the measurement, not something a longer recording escapes -- a longer recording
    helps through the duration term instead.

    The correlation time of the SUMMED field is the same as one dye's: a sum of independent
    Ornstein-Uhlenbeck processes has the same normalized autocorrelation, so applying this to
    a whole-field flux curve is correct. Summing reduces the fluctuation AMPLITUDE as the
    reciprocal square root of the dye count, which the caller carries in ``relative_noise``,
    not the correlation time.

    Args:
        n_frames: number of frames.
        rho: lag-one correlation of the noise, ``exp(-lambda_rate * frame_time_seconds)``.

    Returns:
        Effective number of independent frames (at least 1).
    """
    r = float(np.clip(rho, -0.999999, 0.999999))
    return float(max(int(n_frames) * (1.0 - r) / (1.0 + r), 1.0))


def crb_decay_rate(n_frames: int, relative_noise: float, rate_per_frame: float,
                   *, n_eff: Optional[float] = None) -> float:
    """Lower bound on the standard deviation of an exponential decay rate, per frame.

    The curve is ``S(t) = A exp(-k t) + B`` with the amplitude and the offset unknown and
    fitted alongside the rate, so the bound is the ``(k, k)`` entry of the INVERSE of the
    3x3 information matrix, not the reciprocal of its own diagonal entry. The distinction is
    not cosmetic: when the decay is shallow the exponential is close to a straight line over
    the observed window, the three parameters become nearly degenerate, and only the product
    ``A k`` is determined. Measured on 1000-frame curves at the MET-FAB emitter density,
    profiling widens the bound by 2.7x at the top of the ``prob_photo_bleach`` prior, 11.7x at
    ``p = 0.1``, 44.6x at ``p = 0.032`` and 150x at the bottom -- so "an order of magnitude" is
    right only in the middle, and the effect is far larger exactly where the answer decides
    whether a recording is long enough.

    With the amplitude and offset known the bound would reduce to the familiar
    ``sd(k) >= eps sqrt(3 / T^3)``. That cubic dependence on duration survives profiling and
    is why a photobleaching probability unmeasurable in a two-second clip becomes measurable
    in a twenty-second recording.

    Args:
        n_frames: number of frames observed.
        relative_noise: standard deviation of one point divided by the signal, ``eps``.
        rate_per_frame: the true decay rate per frame, used for the exact sum.
        n_eff: effective independent frames, if the noise is correlated. When given, the
            bound is inflated by ``sqrt(n_frames / n_eff)``.

    Returns:
        Lower bound on ``sd(k)`` in units of inverse frames.
    """
    t = np.arange(int(n_frames), dtype=np.float64)
    k = float(rate_per_frame)
    decay = np.exp(-k * t)

    # The amplitude and the offset are NOT known: both are fitted alongside the rate, and
    # profiling them out is not optional. When the decay is shallow, exp(-kt) over the
    # observed window is close to a straight line, so A exp(-kt) + B is nearly degenerate
    # with a two-parameter line: only the PRODUCT A*k -- the slope -- is determined, and k
    # alone is not. Treating A and B as known therefore understates the bound by an order of
    # magnitude exactly where the parameter is hardest, which is the regime that decides
    # whether a recording length is adequate. The full 3x3 information matrix in
    # (A, k, B) is built and inverted, with A carried in units of itself (A = 1) because the
    # noise is stated relative to the amplitude.
    #     dS/dA = exp(-kt),   dS/dk = -A t exp(-kt),   dS/dB = 1
    if float(relative_noise) <= 0:
        return float("inf")
    inv_var = 1.0 / (float(relative_noise) ** 2)
    d_a = decay
    d_k = -t * decay
    d_b = np.ones_like(t)
    jac = np.stack([d_a, d_k, d_b], axis=1)
    fisher = (jac.T @ jac) * inv_var
    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        return float("inf")
    if cov[1, 1] <= 0:
        return float("inf")
    sd = float(np.sqrt(cov[1, 1]))
    if n_eff is not None and n_eff > 0:
        sd *= float(np.sqrt(int(n_frames) / float(n_eff)))
    return sd


def crb_prob_bleach_dex(prob_photo_bleach: float, n_frames: int, relative_noise: float,
                        *, numb_photo_bleach: int = 100,
                        lambda_rate: Optional[float] = None,
                        frame_time_seconds: float = 0.02) -> dict:
    """Lower bound on ``prob_photo_bleach``, in dex, for one recording.

    The model's parameter is a probability over a FIXED reference window of
    ``numb_photo_bleach`` frames, not over the clip: the per-frame rate is
    ``k = -ln(1 - p) / numb_photo_bleach``, so a clip shorter than the window measures only
    part of one decay. Propagating the rate bound through that relation gives the bound on
    ``p``, and dividing by ``p ln 10`` puts it in dex, the unit the prior is stated in.

    Args:
        prob_photo_bleach: the true value.
        n_frames: frames in the recording.
        relative_noise: relative noise on the total-fluorescence curve per frame.
        numb_photo_bleach: the fixed reference window (100).
        lambda_rate: flicker rate; when given, the brightness correlation is folded in
            through `effective_sample_size_ar1` and the bound widens accordingly.
        frame_time_seconds: frame interval, for the correlation.

    Returns:
        ``dict`` with ``sd_p``, ``sd_dex``, ``rate_per_frame``, and ``n_eff``.
    """
    p = float(prob_photo_bleach)
    npb = int(numb_photo_bleach)
    k = -np.log(max(1.0 - p, 1e-12)) / npb

    n_eff = None
    if lambda_rate is not None:
        rho = float(np.exp(-float(lambda_rate) * float(frame_time_seconds)))
        n_eff = effective_sample_size_ar1(n_frames, rho)

    sd_k = crb_decay_rate(n_frames, relative_noise, k, n_eff=n_eff)
    # p = 1 - exp(-npb * k)  =>  dp/dk = npb * exp(-npb * k) = npb * (1 - p)
    sd_p = float(npb * (1.0 - p) * sd_k)
    sd_dex = float(sd_p / (max(p, 1e-12) * np.log(10.0)))
    return dict(sd_p=sd_p, sd_dex=sd_dex, rate_per_frame=float(k),
                n_eff=(float(n_eff) if n_eff is not None else float(n_frames)))


def crb_ar1_rate(n_traces: int, trace_length: int, rho: float) -> float:
    """Lower bound on the standard deviation of an AR(1) lag-one correlation.

    For a stationary AR(1) series of length ``L``, ``sd(rho_hat) >= sqrt((1 - rho^2) / L)``;
    with ``n`` independent traces the total sample is ``n * L``.

    Args:
        n_traces: number of independent traces.
        trace_length: frames per trace.
        rho: the true lag-one correlation.

    Returns:
        Lower bound on ``sd(rho)``.
    """
    total = int(n_traces) * int(trace_length)
    if total < 2:
        return float("inf")
    r = float(np.clip(rho, -0.999999, 0.999999))
    return float(np.sqrt((1.0 - r * r) / total))


def crb_lambda_rate_dex(lambda_rate: float, n_traces: int, trace_length: int, *,
                        frame_time_seconds: float = 0.02,
                        measurement_noise_ratio: float = 0.0) -> dict:
    """Lower bound on ``lambda_rate``, in dex, from the brightness autocorrelation.

    ``lambda_rate = -ln(rho) / dt``, so ``d lambda / d rho = -1 / (rho dt)`` and the bound on
    the rate follows from the bound on the correlation.

    White measurement noise on each brightness sample attenuates the observed lag-one
    correlation toward zero by the factor ``1 / (1 + nsr)`` where ``nsr`` is the ratio of
    measurement variance to signal variance. The flicker derivation removes that attenuation
    by normalizing the autocorrelation at lag one rather than lag zero, but the noise still
    costs precision, and ``measurement_noise_ratio`` folds that cost in.

    Args:
        lambda_rate: the true rate, per second.
        n_traces: number of independent emitter traces.
        trace_length: frames per trace.
        frame_time_seconds: frame interval.
        measurement_noise_ratio: measurement variance over signal variance, per sample.

    Returns:
        ``dict`` with ``rho``, ``sd_rho``, ``sd_lambda``, ``sd_dex``.
    """
    dt = float(frame_time_seconds)
    rho = float(np.exp(-float(lambda_rate) * dt))
    sd_rho = crb_ar1_rate(n_traces, trace_length, rho)
    if measurement_noise_ratio > 0:
        sd_rho *= float(1.0 + measurement_noise_ratio)
    sd_lambda = float(sd_rho / (max(rho, 1e-12) * dt))
    sd_dex = float(sd_lambda / (max(float(lambda_rate), 1e-12) * np.log(10.0)))
    return dict(rho=rho, sd_rho=sd_rho, sd_lambda=sd_lambda, sd_dex=sd_dex)
