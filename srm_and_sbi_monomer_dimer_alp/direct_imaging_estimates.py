"""Direct (non-neural) estimators of imaging parameters from rendered videos.

Pure measurement kernels: every function here takes arrays and returns arrays or
scalars, performs no file access, resolves no machine profile, and prints nothing.
The Analysis scripts in ``Script_Bank/Analysis`` supply the data and own the report.

These estimators are simulation-based in the same sense the neural posterior
estimator is -- they are validated against known ground truth on simulated
recordings, and their acceptance criteria are stated before they are run. What
distinguishes them is that they are not learned: each one reads a parameter off a
closed-form property of the forward model (`simulation_dli_support`) rather than
off an amortized flow. See `DETECTOR_WORKFLOW.md` sec. 9.4 for the reduced-inferred-block
proposal these serve, which is a proposal and not in force.

The forward model these invert, per pixel ``(i, j)`` and frame ``t``:

    I[i,j,t] = kappa_o + sum_d b_d(t) * G(i; x_d(t), sigma_d) * G(j; y_d(t), sigma_d)
    ADU      = Gamma(shape=Poisson(kappa_q * I), scale=gamma) + N(0, kappa_s) + kappa_b

where ``G`` is the exact integral of a unit Gaussian over the pixel square (the erf
difference of `add_pixel_counts`), ``sigma_d`` is the PSF width of the dye's SUBUNIT,
and ``b_d`` its stationary OU brightness. Two consequences the estimators rest on, both
verified numerically against the renderer:

    mean(ADU)     = gamma * kappa_q * I + kappa_b
    variance(ADU) = 2 * gamma^2 * kappa_q * I + kappa_s^2          (EMCCD excess noise, F^2 = 2)

The camera block (``gamma``, ``kappa_o``, ``kappa_b``, ``kappa_s``, ``kappa_q``) is the
SCOPE nuisance: known to within a tight box, so it is supplied, never fitted.

Width convention. `sample_psf_width` draws ``sqrt(2) * sigma`` in PIXELS from a lognormal
with scale ``mu_r`` and shape ``sigma_r``; one draw per SUBUNIT, carried by every dye of
that subunit for the whole recording and across its reactions. So ``mu_r`` is the MEDIAN
of ``sqrt(2) * sigma`` and ``sigma_r`` the standard deviation of ``ln(sqrt(2) * sigma)``.
Everything here works in that convention and converts once, at the boundary.

Storage domain. Stored videos are 8-bit: `io.convert_video_dtype` maps 0-65535 ADU onto
0-255 globally, with clipping and no per-video normalization, applied identically to
synthetic and experimental frames. These kernels therefore take 8-bit levels and convert
to ADU with `levels_to_adu`, so that a direct estimate and the neural posterior are read
off the very same pixels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import least_squares
from scipy.special import erf

__all__ = [
    "ADU_PER_LEVEL",
    "levels_to_adu",
    "background_mean_adu",
    "background_sigma_adu",
    "adu_to_photons",
    "pixel_gaussian_patch",
    "detect_spots",
    "fit_spot_width",
    "measure_spot_widths",
    "link_spot_tracks",
    "psf_width_population",
    "spot_flux_curve",
    "frame_flux_curve",
    "fit_fluorescence_loss",
]


# =============================================================================
# Camera domain: 8-bit levels -> ADU -> photons
# =============================================================================

# One 8-bit level in ADU. `io.convert_video_dtype` rescales the fixed range
# 0-65535 onto 0-255, so a stored level is this many ADU wide. The quantization
# step is smaller than the EMCCD background standard deviation (~302 ADU at the
# SCOPE box center), so the noise dithers it and the step costs the width
# estimator nothing measurable -- but the step is large in absolute terms, and
# any estimator that differences large nearly-equal sums must account for it.
ADU_PER_LEVEL = (2 ** 16 - 1) / (2 ** 8 - 1)


def levels_to_adu(video_levels: np.ndarray) -> np.ndarray:
    """Convert stored 8-bit levels back to the ADU domain of the renderer.

    Args:
        video_levels: array of any shape holding stored 8-bit pixel levels.

    Returns:
        Float array of the same shape, in ADU. The inverse of the global
        0-65535 -> 0-255 map only up to the quantization step; it does not undo
        clipping at either end.
    """
    return np.asarray(video_levels, dtype=np.float64) * ADU_PER_LEVEL


def background_mean_adu(gamma: float, kappa_q: float, kappa_o: float,
                        kappa_b: float) -> float:
    """Expected ADU of an emitter-free pixel: ``gamma * kappa_q * kappa_o + kappa_b``.

    Args:
        gamma: ADU per photoelectron (the sole identifiable gain quantity).
        kappa_q: quantum efficiency.
        kappa_o: optical background, incident photons per pixel per frame.
        kappa_b: electronic bias in ADU.

    Returns:
        The mean background level in ADU.
    """
    return float(gamma) * float(kappa_q) * float(kappa_o) + float(kappa_b)


def background_sigma_adu(gamma: float, kappa_q: float, kappa_o: float,
                         kappa_s: float) -> float:
    """Standard deviation of an emitter-free pixel, in ADU.

    ``sqrt(2 * gamma^2 * kappa_q * kappa_o + kappa_s^2)``: the factor 2 is the EMCCD
    excess-noise factor of the Poisson-Gamma cascade, and the read noise adds in
    quadrature after the register.

    Args:
        gamma: ADU per photoelectron.
        kappa_q: quantum efficiency.
        kappa_o: optical background in photons per pixel per frame.
        kappa_s: read-noise standard deviation in ADU.

    Returns:
        The background standard deviation in ADU.
    """
    g, q = float(gamma), float(kappa_q)
    return float(np.sqrt(2.0 * g * g * q * float(kappa_o) + float(kappa_s) ** 2))


def adu_to_photons(adu_net: np.ndarray, gamma: float, kappa_q: float) -> np.ndarray:
    """Convert background-subtracted ADU to incident photons.

    Gain and quantum efficiency enter the mean only through the product
    ``gamma * kappa_q``, which is why neither is separately identifiable from the
    videos and both are supplied from the SCOPE box.

    Args:
        adu_net: background-subtracted ADU, any shape.
        gamma: ADU per photoelectron.
        kappa_q: quantum efficiency.

    Returns:
        Float array of incident photons, same shape as ``adu_net``.
    """
    return np.asarray(adu_net, dtype=np.float64) / (float(gamma) * float(kappa_q))


# =============================================================================
# The pixel-integrated Gaussian (the renderer's own spot shape)
# =============================================================================

def pixel_gaussian_patch(amplitude: float, x0: float, y0: float, sigma: float,
                         background: float, size: int) -> np.ndarray:
    """Render one spot on a ``size x size`` patch exactly as the renderer does.

    Each pixel is the integral of the Gaussian over the pixel square, not the
    Gaussian sampled at the pixel center. This is the same erf difference
    `add_pixel_counts` uses, so a fit of this model measures the simulator's own
    ``sigma`` with no shape mismatch. Pixel ``k`` spans ``[k, k+1]``, matching the
    renderer's ``bounds = linspace(0, n_px, n_px + 1)``.

    Args:
        amplitude: total signal of the spot over the whole plane, in the units of
            the patch (ADU). The peak pixel holds only a fraction of it.
        x0, y0: spot center in patch-local pixel units (axis 0 is x, axis 1 is y,
            matching the renderer's einsum order).
        sigma: Gaussian standard deviation in pixels (NOT ``sqrt(2) * sigma``).
        background: additive floor in the same units as ``amplitude``.
        size: patch edge length in pixels.

    Returns:
        ``(size, size)`` float array.
    """
    s2 = np.sqrt(2.0) * float(sigma)
    edges = np.arange(size + 1, dtype=np.float64)
    ex = (erf((edges[1:] - x0) / s2) - erf((edges[:-1] - x0) / s2)) / 2.0
    ey = (erf((edges[1:] - y0) / s2) - erf((edges[:-1] - y0) / s2)) / 2.0
    return float(amplitude) * np.outer(ex, ey) + float(background)


# =============================================================================
# Spot detection
# =============================================================================

def detect_spots(frame_adu: np.ndarray, background_adu: float, noise_adu: float,
                 *, match_sigma_px: float = 1.0, n_sigma: float = 5.0,
                 min_separation_px: int = 5, edge_margin_px: int = 8,
                 max_spots: int = 400) -> np.ndarray:
    """Find isolated bright spots in one frame by matched filtering.

    The frame is convolved with a Gaussian of the expected spot width (a matched
    filter, which is the maximum-SNR linear detector for a known shape), and local
    maxima that clear ``n_sigma`` times the *filtered* noise level are kept. Where two
    detections fall within ``min_separation_px``, the BRIGHTER is kept and the dimmer
    discarded -- not both -- because an overlapping neighbor biases a single-Gaussian width
    fit upward, while discarding the brighter member as well would throw away the majority
    of the usable signal. The survivor still has a neighbor nearby, so its fit masks the
    neighbor's pixels rather than relying on isolation alone (`fit_spot_width`).

    A threshold at four filtered standard deviations admits occasional noise peaks. They are
    not removed here; they are removed downstream, by the width bounds and the relative
    standard-error cut of `measure_spot_widths`, which a noise peak fails.

    Args:
        frame_adu: ``(n_x, n_y)`` frame in ADU (axis 0 is the renderer's x axis).
        background_adu: expected emitter-free level, from `background_mean_adu`.
        noise_adu: per-pixel background standard deviation, from `background_sigma_adu`.
        match_sigma_px: width of the matched filter in pixels; the middle of the
            ``mu_r`` box is a good default and the detector is not sharp in it.
        n_sigma: detection threshold in units of the filtered noise.
        min_separation_px: minimum distance between accepted spots.
        edge_margin_px: reject detections closer than this to any border, so a fit
            patch always lies inside the frame.
        max_spots: keep at most this many, brightest first (a guard against a
            pathological frame, not a scientific choice).

    Returns:
        ``(n_spots, 2)`` integer array of ``(x, y)`` pixel indices, brightest first.
        May be empty.
    """
    frame = np.asarray(frame_adu, dtype=np.float64)
    filtered = gaussian_filter(frame - float(background_adu), sigma=float(match_sigma_px),
                               mode="nearest")
    # A Gaussian filter of width s attenuates white noise by 1/sqrt(4*pi*s^2).
    noise_filtered = float(noise_adu) / np.sqrt(4.0 * np.pi * float(match_sigma_px) ** 2)
    threshold = float(n_sigma) * noise_filtered

    peaks = (filtered == maximum_filter(filtered, size=int(min_separation_px))) \
        & (filtered > threshold)
    if edge_margin_px > 0:
        m = int(edge_margin_px)
        mask = np.zeros_like(peaks)
        mask[m:-m, m:-m] = True
        peaks &= mask
    xs, ys = np.nonzero(peaks)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    order = np.argsort(filtered[xs, ys])[::-1]
    xs, ys = xs[order], ys[order]

    # Drop any detection that has a brighter neighbor within min_separation_px:
    # an overlapping pair cannot be fitted as one isolated Gaussian.
    keep = np.ones(xs.size, dtype=bool)
    for i in range(xs.size):
        if not keep[i]:
            continue
        d = np.hypot(xs[i + 1:] - xs[i], ys[i + 1:] - ys[i])
        keep[i + 1:][d < float(min_separation_px)] = False
    xs, ys = xs[keep], ys[keep]
    out = np.stack([xs, ys], axis=1)[:int(max_spots)]
    return out.astype(np.int64)


# =============================================================================
# Per-spot width fit
# =============================================================================

def fit_spot_width(frame_adu: np.ndarray, x_px: int, y_px: int, *,
                   half_px: int = 14, sigma_bounds: Tuple[float, float] = (0.3, 6.0),
                   noise_adu: float = 1.0,
                   background_adu: Optional[float] = None,
                   gamma: Optional[float] = None,
                   kappa_b: Optional[float] = None,
                   kappa_s: Optional[float] = None,
                   neighbors: Optional[np.ndarray] = None,
                   neighbor_mask_px: float = 4.0) -> Optional[dict]:
    """Fit one pixel-integrated Gaussian to one spot and return its width.

    Four free parameters -- amplitude, center (x, y) and ``sigma`` -- fitted by
    Levenberg-Marquardt least squares, with the background SUPPLIED rather than
    fitted whenever it is known.

    Why the background is not free. A wide, faint Gaussian and a slightly raised
    flat floor are nearly degenerate over a finite patch: a free background absorbs
    the wings of a broad spot and the fitted ``sigma`` collapses. Measured against the
    renderer, a free background pulls the fitted width down by 6% at the bottom of
    the ``mu_r`` box and by 25% at the top -- a width-dependent bias, which is the worst
    kind, because it compresses the very spread ``sigma_r`` is meant to measure. The
    camera block is the SCOPE nuisance and is known to about 1% -- the widest of the five boxes,
    ``kappa_o``, ``kappa_b`` and ``kappa_q``, are +/-1.15% about their centers and the other two
    +/-0.58% -- so supplying ``gamma * kappa_q * kappa_o + kappa_b`` removes the degeneracy at a
    cost far below the bias it prevents.
    Passing ``background_adu=None`` restores the free-background fit for diagnosis.

    The patch half-width matters in the same direction: a spot wider than the patch
    is truncated and its fitted width is biased low. ``half_px = 14`` holds 3 sigma
    for every width the ``mu_r`` x ``sigma_r`` prior box puts appreciable mass on.

    Args:
        frame_adu: ``(n_x, n_y)`` frame in ADU.
        x_px, y_px: integer pixel index of the detection.
        half_px: patch half-width; the fit uses ``2 * half_px`` pixels per side.
        sigma_bounds: hard bounds on the fitted ``sigma`` in pixels. The upper bound
            must exceed the widest width the prior puts mass on, or it truncates the
            population itself.
        noise_adu: per-pixel noise used to scale the residual, so the returned
            standard error is calibrated.
        background_adu: known background level in ADU; ``None`` fits it as a fifth
            free parameter (biased, see above).
        gamma, kappa_b, kappa_s: camera constants enabling SIGNAL-DEPENDENT residual
            weighting. EMCCD variance is not flat -- it rises with the signal as
            ``Var = 2 * gamma * (mean_ADU - kappa_b) + kappa_s^2`` -- so weighting every
            pixel by the background noise alone over-weights the bright core and makes
            the returned ``sigma_se`` roughly half the true fit scatter. Since that
            standard error is what the errors-in-variables correction of
            `psf_width_population` subtracts, getting it wrong leaves ``sigma_r``
            inflated (measured: +0.20 at the bottom of the ``sigma_r`` box). Supply all
            three to weight correctly; omit them to fall back on flat ``noise_adu``.
            The weight uses the MODEL intensity, not the observed pixel, so that noise
            in a pixel cannot lower its own weight.
        neighbors: optional ``(n, 2)`` array of OTHER detected spot positions in the
            frame. Pixels within ``neighbor_mask_px`` of any of them are excluded from
            the fit. At the emitter density of a real recording a patch this size
            contains one or two further emitters on average, and their flux inflates the
            fitted width -- measured as a +0.018 dex bias on ``mu_r``, which is what
            masking removes. The spot being fitted is never masked by its own entry:
            pass the neighbors only, or rely on the center-distance guard below.
        neighbor_mask_px: exclusion radius around each neighbor, in pixels.

    Returns:
        ``dict`` with ``sigma``, ``sigma_se`` (standard error of ``sigma``),
        ``amplitude``, ``x``, ``y`` (patch-local), ``background``, ``cost``, or
        ``None`` if the patch falls outside the frame or the fit fails.
    """
    frame = np.asarray(frame_adu, dtype=np.float64)
    h = int(half_px)
    x0i, y0i = int(x_px) - h, int(y_px) - h
    if x0i < 0 or y0i < 0 or x0i + 2 * h > frame.shape[0] or y0i + 2 * h > frame.shape[1]:
        return None
    patch = frame[x0i:x0i + 2 * h, y0i:y0i + 2 * h]
    size = 2 * h

    # Local coordinates of the detected pixel's center.
    xc, yc = float(x_px) - x0i + 0.5, float(y_px) - y0i + 0.5
    bg0 = float(np.median(patch)) if background_adu is None else float(background_adu)
    amp0 = max(float(patch.max()) - bg0, 1e-6) * 6.0      # a spot holds ~1/6 of its flux in the peak pixel

    have_camera = (gamma is not None and kappa_b is not None and kappa_s is not None)

    def model_of(p):
        bg = p[4] if background_adu is None else bg0
        return pixel_gaussian_patch(p[0], p[1], p[2], p[3], bg, size)

    # The fit itself is FLAT-weighted. Weighting by the model variance is the
    # efficient choice only if the model is exact; in practice it down-weights the
    # bright core -- which carries the width information -- and leans on the noisy
    # outskirts, which measured +0.06 dex HIGH on mu_r against the renderer. Flat
    # weighting leaves the point estimate very nearly unbiased in aggregate (measured bias
    # -0.0014 dex on mu_r across the prior box, against -0.06 dex under model-variance
    # weighting); a residual corner bias remains at the widest spots and is reported rather
    # than corrected. The variance law is applied afterwards, to the uncertainty, where it
    # belongs.
    # Exclude pixels dominated by a neighboring emitter, so its flux cannot be
    # absorbed into this spot's width.
    fit_mask = np.ones(patch.shape, dtype=bool)
    if neighbors is not None and len(neighbors):
        nb = np.asarray(neighbors, dtype=np.float64).reshape(-1, 2)
        gx, gy = np.mgrid[0:2 * h, 0:2 * h]
        gx = gx + x0i + 0.5
        gy = gy + y0i + 0.5
        for (nx, ny) in nb:
            if np.hypot(nx - (float(x_px) + 0.5), ny - (float(y_px) + 0.5)) < 1e-6:
                continue                       # this is the spot being fitted
            fit_mask &= np.hypot(gx - nx, gy - ny) > float(neighbor_mask_px)
        if fit_mask.sum() < 40:                # too little left to constrain four parameters
            return None

    def residual(p):
        return ((model_of(p) - patch)[fit_mask] / float(noise_adu)).ravel()

    if background_adu is None:
        x0 = [amp0, xc, yc, 1.0, bg0]
        lower = [0.0, xc - 2.0, yc - 2.0, sigma_bounds[0], -np.inf]
        upper = [np.inf, xc + 2.0, yc + 2.0, sigma_bounds[1], np.inf]
    else:
        x0 = [amp0, xc, yc, 1.0]
        lower = [0.0, xc - 2.0, yc - 2.0, sigma_bounds[0]]
        upper = [np.inf, xc + 2.0, yc + 2.0, sigma_bounds[1]]

    try:
        fit = least_squares(residual, x0=x0, bounds=(lower, upper),
                            method="trf", max_nfev=200)
    except Exception:
        return None
    if not fit.success:
        return None

    sigma = float(fit.x[3])
    # Standard error of sigma. The fit is flat-weighted but the data are not
    # homoscedastic, so the naive (J^T J)^-1 understates the true scatter by about a
    # factor of two at a bright spot -- and that standard error is exactly what the
    # errors-in-variables correction of `psf_width_population` subtracts, so
    # understating it leaves sigma_r inflated. The sandwich form
    #     cov = (J^T J)^-1 J^T D J (J^T J)^-1,   D = diag(Var_i / noise_adu^2)
    # is the covariance of a weighted least-squares estimator whose weights are not
    # the inverse true variance, with Var_i from the EMCCD law at the fitted model.
    sigma_se = np.nan
    try:
        jac = fit.jac
        jtj_inv = np.linalg.inv(jac.T @ jac)
        if have_camera:
            model_opt = model_of(fit.x)[fit_mask].ravel()
            var_i = 2.0 * float(gamma) * np.maximum(model_opt - float(kappa_b), 0.0) \
                + float(kappa_s) ** 2
            d = var_i / (float(noise_adu) ** 2)
            cov = jtj_inv @ (jac.T @ (jac * d[:, None])) @ jtj_inv
        else:
            cov = jtj_inv
        if cov[3, 3] > 0:
            sigma_se = float(np.sqrt(cov[3, 3]))
    except np.linalg.LinAlgError:
        pass

    # A width pinned on a bound is not a measurement.
    if sigma <= sigma_bounds[0] * 1.001 or sigma >= sigma_bounds[1] * 0.999:
        return None

    bg_out = float(fit.x[4]) if background_adu is None else bg0
    return dict(sigma=sigma, sigma_se=sigma_se, amplitude=float(fit.x[0]),
                x=float(fit.x[1]) + x0i, y=float(fit.x[2]) + y0i,
                background=bg_out, cost=float(fit.cost))


def measure_spot_widths(video_levels: np.ndarray, scope: dict, *,
                        frame_stride: int = 1, max_frames: Optional[int] = None,
                        half_px: int = 14, n_sigma: float = 4.0,
                        min_separation_px: int = 5,
                        match_sigma_px: float = 1.0,
                        max_spots_per_frame: int = 400) -> dict:
    """Measure every isolated spot's PSF width across one stored video.

    Detection and fitting run per FRAME, not on a time average: the emitters
    diffuse, so a time-averaged image is smeared by the motion and its fitted width
    is not the PSF width. Per-frame fits are noisier, but the noise is accounted
    for explicitly by `psf_width_population` rather than being averaged away here.

    Args:
        video_levels: ``(n_frames, n_x, n_y)`` stored 8-bit video.
        scope: mapping with the five SCOPE camera values ``gamma``, ``kappa_o``,
            ``kappa_b``, ``kappa_s``, ``kappa_q`` in physical units.
        frame_stride: use every ``frame_stride``-th frame.
        max_frames: stop after this many used frames (None = all).
        half_px: fit patch half-width, passed to `fit_spot_width`.
        n_sigma: detection threshold, passed to `detect_spots`.
        min_separation_px: isolation radius, passed to `detect_spots`.
        match_sigma_px: matched-filter width, passed to `detect_spots`.
        max_spots_per_frame: detection cap per frame.

    Returns:
        ``dict`` with ``sqrt2sigma`` (1D array, one entry per accepted spot-frame,
        in the ``mu_r`` convention), ``sqrt2sigma_se`` (its standard error),
        ``amplitude``, ``frame_index``, and ``n_frames_used``.
    """
    video = np.asarray(video_levels)
    if video.ndim != 3:
        raise ValueError(f"video_levels has shape {video.shape}; expected (n_frames, n_x, n_y).")

    bg = background_mean_adu(scope["gamma"], scope["kappa_q"], scope["kappa_o"], scope["kappa_b"])
    noise = background_sigma_adu(scope["gamma"], scope["kappa_q"], scope["kappa_o"], scope["kappa_s"])

    frames = range(0, video.shape[0], int(frame_stride))
    if max_frames is not None:
        frames = list(frames)[:int(max_frames)]

    widths, width_se, amps, frame_idx, xs_fit, ys_fit = [], [], [], [], [], []
    n_used = 0
    for t in frames:
        frame = levels_to_adu(video[t])
        spots = detect_spots(frame, bg, noise, match_sigma_px=match_sigma_px,
                             n_sigma=n_sigma, min_separation_px=min_separation_px,
                             edge_margin_px=half_px + 1, max_spots=max_spots_per_frame)
        n_used += 1
        spot_centers = spots.astype(np.float64) + 0.5
        for si, (sx, sy) in enumerate(spots):
            others = np.delete(spot_centers, si, axis=0)
            fit = fit_spot_width(frame, sx, sy, half_px=half_px, noise_adu=noise,
                                 background_adu=bg, gamma=scope["gamma"],
                                 kappa_b=scope["kappa_b"], kappa_s=scope["kappa_s"],
                                 neighbors=others)
            if fit is None or not np.isfinite(fit["sigma_se"]):
                continue
            widths.append(np.sqrt(2.0) * fit["sigma"])
            width_se.append(np.sqrt(2.0) * fit["sigma_se"])
            amps.append(fit["amplitude"])
            frame_idx.append(t)
            xs_fit.append(fit["x"])
            ys_fit.append(fit["y"])

    return dict(
        sqrt2sigma=np.asarray(widths, dtype=np.float64),
        sqrt2sigma_se=np.asarray(width_se, dtype=np.float64),
        amplitude=np.asarray(amps, dtype=np.float64),
        frame_index=np.asarray(frame_idx, dtype=np.int64),
        x=np.asarray(xs_fit, dtype=np.float64),
        y=np.asarray(ys_fit, dtype=np.float64),
        n_frames_used=int(n_used),
    )


# =============================================================================
# Linking measurements of one subunit across frames
# =============================================================================

def link_spot_tracks(frame_index: np.ndarray, x: np.ndarray, y: np.ndarray, *,
                     max_step_px: float = 3.0, max_gap: int = 2,
                     frame_stride: int = 1) -> np.ndarray:
    """Group per-frame spot measurements into per-subunit tracks.

    The PSF width is drawn once per SUBUNIT and carried by every dye of that subunit
    for the whole recording, across its reactions (`render_dli_video`). So every frame
    in which a subunit is detected is a repeat measurement of ONE width, and averaging
    a track's measurements before taking the population spread cuts the per-fit noise
    by the square root of the track length. That matters more than it may seem: the
    errors-in-variables correction of `psf_width_population` subtracts an estimate of
    that noise, and an error in the estimate propagates straight into ``sigma_r``.
    Averaging first makes the correction a small perturbation rather than the dominant
    term.

    Greedy nearest-neighbor linking, frames in order: each measurement joins the
    closest open track whose last position is within the displacement gate and whose last
    frame is no more than the gap gate behind, otherwise it starts a track. Only
    well-separated spots reach this function (see `detect_spots`), and over one sampled
    interval a receptor moves a small fraction of the isolation radius, so the
    assignment is not ambiguous in practice.

    Both gates are expressed in SAMPLED frames and scaled by ``frame_stride``, because
    ``measure_spot_widths`` records raw frame numbers while visiting every
    ``frame_stride``-th frame. Gating on raw frame numbers instead would silently break the
    moment the stride exceeded the gap: at a stride of three with a two-frame gap, no two
    detections are ever adjacent, every detection becomes its own track, and the population
    estimate returns nothing at all. The displacement gate scales as the SQUARE ROOT of the
    stride, since a diffusing emitter's displacement grows with the square root of elapsed
    time.

    Args:
        frame_index: frame of each measurement, ascending within the array.
        x, y: fitted spot centers in pixel units.
        max_step_px: largest accepted displacement between consecutive SAMPLED frames. Note
            that the scaled gate approaches ``detect_spots``'s ``min_separation_px``: at a
            stride of two the gate is 4.2 px against an isolation radius of 5, so two spots
            near that separation can in principle be swapped between tracks. A swap costs
            only the pairing of two width measurements of similar spots, which is why the
            tolerance is accepted; tightening it fragments tracks instead, which costs more.
        max_gap: largest accepted gap, in SAMPLED frames, before a track is closed.
        frame_stride: the stride the measurements were taken at, so both gates convert
            from sampled frames to the raw frame numbers carried in ``frame_index``.

    Returns:
        Integer array of track labels, one per measurement, starting at 0.
    """
    frame_index = np.asarray(frame_index)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = frame_index.size
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels

    stride = max(int(frame_stride), 1)
    gap_limit = int(max_gap) * stride
    step_limit = float(max_step_px) * np.sqrt(float(stride))

    order = np.argsort(frame_index, kind="stable")
    # Open tracks: label -> (last_frame, last_x, last_y)
    last_frame: list = []
    last_x: list = []
    last_y: list = []
    n_tracks = 0

    for idx in order:
        f, px, py = int(frame_index[idx]), x[idx], y[idx]
        best, best_d = -1, np.inf
        for k in range(n_tracks):
            gap = f - last_frame[k]
            if gap <= 0 or gap > gap_limit:
                continue                      # same frame, or the track has lapsed
            d = np.hypot(px - last_x[k], py - last_y[k])
            if d <= step_limit and d < best_d:
                best, best_d = k, d
        if best < 0:
            best = n_tracks
            last_frame.append(f); last_x.append(px); last_y.append(py)
            n_tracks += 1
        else:
            last_frame[best] = f; last_x[best] = px; last_y[best] = py
        labels[idx] = best
    return labels



def _norm_ppf(q: float) -> float:
    """Standard-normal quantile, for the trim correction. Acklam's rational approximation,
    accurate to about 1e-9 over the range this module uses."""
    q = float(min(max(q, 1e-12), 1.0 - 1e-12))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if q < plow:
        t = np.sqrt(-2.0 * np.log(q))
        return (((((c[0]*t + c[1])*t + c[2])*t + c[3])*t + c[4])*t + c[5]) / \
               ((((d[0]*t + d[1])*t + d[2])*t + d[3])*t + 1.0)
    if q > phigh:
        t = np.sqrt(-2.0 * np.log(1.0 - q))
        return -(((((c[0]*t + c[1])*t + c[2])*t + c[3])*t + c[4])*t + c[5]) / \
                ((((d[0]*t + d[1])*t + d[2])*t + d[3])*t + 1.0)
    t = q - 0.5
    r = t * t
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * t / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

# =============================================================================
# From measured widths to (mu_r, sigma_r)
# =============================================================================

def psf_width_population(sqrt2sigma: np.ndarray, sqrt2sigma_se: np.ndarray,
                         *, track_id: Optional[np.ndarray] = None,
                         min_track_length: int = 3,
                         trim_quantile: float = 0.01,
                         max_relative_se: float = 0.5) -> dict:
    """Estimate ``(mu_r, sigma_r)`` from a sample of measured spot widths.

    The measured widths carry the population spread AND the per-fit noise:

        ln w_hat = ln w_true + e,     var(ln w_hat) = sigma_r^2 + var(e)

    so the raw spread of ``ln w_hat`` overestimates ``sigma_r``. This is the same
    errors-in-variables inflation that makes the ThunderSTORM fitted spreads upper-biased
    (`DETECTOR_WORKFLOW.md` sec. 6.5, caveat 1). Here ``var(e)`` is available per spot from
    the fit covariance, so it is subtracted rather than assumed:

        sigma_r^2 = max(var_trimmed(ln w_hat) / trim_factor - mean(se(ln w_hat)^2), 0)

    where ``trim_factor`` undoes the variance a symmetric trim removes from a roughly normal
    sample (0.8735 at the 1%/99% default). Without it the estimate comes back 6.5% short at
    every value of ``sigma_r``, which was measured against a pure lognormal draw.

    ``mu_r`` is the median of the lognormal, estimated as ``exp(mean(ln w_hat))``, which
    the additive noise in the log leaves unbiased.

    Args:
        sqrt2sigma: measured ``sqrt(2) * sigma`` values, one per spot-frame.
        sqrt2sigma_se: their standard errors, same shape.
        track_id: optional per-measurement track label from `link_spot_tracks`. When
            given, measurements are averaged WITHIN a track first, because a track is
            repeat measurement of one subunit's single width; the population spread is
            then taken across tracks. This is strictly better conditioned: the noise
            term falls as the reciprocal of the track length, so the correction below
            stops dominating. When ``None``, each spot-frame is treated as an
            independent draw, which inflates ``sigma_r`` wherever the per-fit noise is
            comparable to the population spread.
        min_track_length: tracks shorter than this are dropped when ``track_id`` is given.
        trim_quantile: symmetric quantile trimmed off each tail of ``ln w_hat``
            before the moments are taken, to stop a handful of failed fits
            dominating a variance. Set 0 to disable.
        max_relative_se: drop any spot whose relative standard error exceeds this;
            such a fit carries almost no width information and inflates the
            noise-variance correction.

    Returns:
        ``dict`` with ``mu_r``, ``sigma_r``, ``sigma_r_raw`` (uncorrected),
        ``noise_variance`` (the subtracted term), ``n`` (spots used), and
        ``log_mean_se`` (standard error of ``ln mu_r``). With ``track_id``, also ``n_tracks``
        and ``track_lengths``: the number of tracks kept and each one's count of fits that
        passed the relative-error filter, both after that filter and the length floor and
        before the tail trim.
    """
    w = np.asarray(sqrt2sigma, dtype=np.float64)
    se = np.asarray(sqrt2sigma_se, dtype=np.float64)
    ok = np.isfinite(w) & np.isfinite(se) & (w > 0) & (se > 0)
    ok &= (se / np.maximum(w, 1e-12)) <= float(max_relative_se)
    w, se = w[ok], se[ok]
    if w.size < 8:
        return dict(mu_r=np.nan, sigma_r=np.nan, sigma_r_raw=np.nan, sigma_r_se=np.nan,
                    noise_variance=np.nan, n=int(w.size), log_mean_se=np.nan)

    log_w = np.log(w)
    log_se = se / w                      # delta method: sd(ln w) = sd(w) / w

    n_tracks = 0
    track_lengths = None
    if track_id is not None:
        tid = np.asarray(track_id)[ok]
        uniq, inv = np.unique(tid, return_inverse=True)
        counts = np.bincount(inv)
        # Inverse-variance mean of ln w within each track, and its standard error.
        wt = 1.0 / np.maximum(log_se ** 2, 1e-12)
        sw = np.bincount(inv, weights=wt)
        swx = np.bincount(inv, weights=wt * log_w)
        keep = counts >= int(min_track_length)
        if keep.sum() < 8:
            return dict(mu_r=np.nan, sigma_r=np.nan, sigma_r_raw=np.nan, sigma_r_se=np.nan,
                        noise_variance=np.nan, n=int(keep.sum()),
                        n_tracks=int(keep.sum()), track_lengths=counts[keep],
                        log_mean_se=np.nan)
        log_w = (swx / sw)[keep]
        log_se = np.sqrt(1.0 / sw[keep])
        n_tracks = int(keep.sum())
        track_lengths = counts[keep]

    # Trimming removes the tails of a roughly normal sample, so the retained values have a
    # SMALLER variance than the population they came from -- by the truncated-normal factor,
    # which at 1%/99% is 0.8735 in variance, i.e. 6.5% low in the standard deviation. Measured
    # against a pure lognormal draw with no measurement error, an uncorrected estimate came
    # back 6.5% short at every value of sigma_r tested. The trim is worth keeping, because a
    # handful of failed fits in a tail would otherwise dominate a variance; but it must be
    # compensated, or it silently biases the very quantity this function reports.
    trim_factor = 1.0
    if trim_quantile and trim_quantile > 0:
        lo, hi = np.quantile(log_w, [trim_quantile, 1.0 - trim_quantile])
        keep = (log_w >= lo) & (log_w <= hi)
        log_w, log_se = log_w[keep], log_se[keep]
        a = float(_norm_ppf(1.0 - trim_quantile))
        phi = float(np.exp(-0.5 * a * a) / np.sqrt(2.0 * np.pi))
        trim_factor = float(1.0 - 2.0 * a * phi / (1.0 - 2.0 * trim_quantile))

    var_raw = float(np.var(log_w, ddof=1)) / max(trim_factor, 1e-6)
    var_noise = float(np.mean(log_se ** 2))
    sigma_r = float(np.sqrt(max(var_raw - var_noise, 0.0)))

    # Standard error of sigma_r by the delta method on the sample variance: var(s^2) is about
    # 2 (sigma_r^2 + v)^2 / (n - 1) for a roughly normal sample, and d(sigma_r)/d(s^2) is
    # 1 / (2 sigma_r), so se(sigma_r) = (sigma_r^2 + v) / (sigma_r sqrt(2 (n - 1))). This is the
    # same expression as the information budget's population bound, evaluated at the estimate;
    # it is a sampling error for the summary, not an allowance for detection or linking defects,
    # which the coverage measurement of DETECTOR_WORKFLOW.md sec. 9.6 is there to expose.
    n_units = int(log_w.size)
    if sigma_r > 0 and n_units > 2:
        sigma_r_se = float(var_raw / (sigma_r * np.sqrt(2.0 * (n_units - 1))))
    else:
        sigma_r_se = float("nan")

    return dict(
        mu_r=float(np.exp(np.mean(log_w))),
        sigma_r=sigma_r,
        sigma_r_raw=float(np.sqrt(var_raw)),   # trim-corrected, BEFORE the noise subtraction
        sigma_r_se=sigma_r_se,
        noise_variance=var_noise,
        n=n_units,
        n_tracks=n_tracks,
        track_lengths=track_lengths,
        log_mean_se=float(np.sqrt(var_raw / log_w.size)),
    )


# =============================================================================
# Fluorescence loss: the photobleaching probability from the decay of spot flux
# =============================================================================

def spot_flux_curve(video_levels: np.ndarray, scope: dict, *,
                    r_inner_px: float = 4.0, r_gap_px: float = 2.0,
                    r_outer_px: float = 9.0, n_sigma: float = 4.0,
                    detect_frames: int = 5, min_separation_px: int = 10,
                    match_sigma_px: float = 1.0) -> dict:
    """Total background-subtracted flux inside spot apertures, frame by frame.

    The observable for photobleaching must be TOTAL fluorescence, which is linear in the
    number of surviving dyes. A count of visible spots is not a substitute: a two-dye spot
    stays visible when one of its dyes bleaches, so counting spots understates the loss and
    does so in a way that depends on the labeling law.

    Why apertures rather than the whole frame. The frame sum is dominated by background: at
    256 x 256 the optical floor contributes about thirty times the emitter signal at the MET-FAB density, so a
    fractional error in the assumed background swamps the decay. The stored 8-bit
    quantization alone is enough to do it -- rounding the background level to the nearest of
    257 ADU steps shifts the frame sum by more than the entire emitter signal. Summing inside
    apertures and subtracting a LOCAL annulus estimate removes the floor without ever needing
    its absolute value.

    Aperture positions are fixed once, from the union of detections over the first
    ``detect_frames`` frames, and held for the whole recording. Re-detecting every frame would
    make the aperture set itself shrink as emitters bleach, which is the very signal being
    measured, and would bias the decay toward zero.

    Args:
        video_levels: ``(n_frames, n_x, n_y)`` stored 8-bit video.
        scope: the five SCOPE camera values in physical units.
        r_inner_px: aperture radius.
        r_gap_px: dead zone between aperture and annulus, so PSF wings do not enter the
            background estimate.
        r_outer_px: outer annulus radius.
        n_sigma: detection threshold for placing apertures.
        detect_frames: frames whose detections seed the aperture set.
        min_separation_px: isolation radius for aperture placement.
        match_sigma_px: matched-filter width for detection.

    Returns:
        ``dict`` with ``flux`` (``(n_frames,)`` net ADU summed over all apertures),
        ``n_apertures``, ``aperture_pixels``, and ``positions``.
    """
    video = np.asarray(video_levels)
    if video.ndim != 3:
        raise ValueError(f"video_levels has shape {video.shape}; expected (n_frames, n_x, n_y).")
    n_frames, n_x, n_y = video.shape

    bg = background_mean_adu(scope["gamma"], scope["kappa_q"], scope["kappa_o"], scope["kappa_b"])
    noise = background_sigma_adu(scope["gamma"], scope["kappa_q"], scope["kappa_o"], scope["kappa_s"])

    # --- fix the aperture set from the earliest frames, before much bleaching -----------
    seen: list = []
    for t in range(min(int(detect_frames), n_frames)):
        frame = levels_to_adu(video[t])
        for (sx, sy) in detect_spots(frame, bg, noise, match_sigma_px=match_sigma_px,
                                     n_sigma=n_sigma, min_separation_px=min_separation_px,
                                     edge_margin_px=int(np.ceil(r_outer_px)) + 1):
            if all(np.hypot(sx - px, sy - py) >= min_separation_px for px, py in seen):
                seen.append((int(sx), int(sy)))
    if not seen:
        return dict(flux=np.zeros(n_frames), n_apertures=0, aperture_pixels=0,
                    positions=np.empty((0, 2), dtype=np.int64))

    positions = np.asarray(seen, dtype=np.int64)
    ro = int(np.ceil(r_outer_px))
    flux = np.zeros(n_frames, dtype=np.float64)
    n_ap_px = 0
    for (sx, sy) in positions:
        x0, x1 = sx - ro, sx + ro
        y0, y1 = sy - ro, sy + ro
        if x0 < 0 or y0 < 0 or x1 > n_x or y1 > n_y:
            continue
        cut = levels_to_adu(video[:, x0:x1, y0:y1])            # (n_frames, 2ro, 2ro)
        gx, gy = np.mgrid[0:2 * ro, 0:2 * ro]
        d = np.hypot(gx - (sx - x0), gy - (sy - y0))
        ap = d < float(r_inner_px)
        an = (d >= float(r_inner_px) + float(r_gap_px)) & (d < float(r_outer_px))
        if ap.sum() == 0 or an.sum() < 8:
            continue
        local_bg = np.median(cut[:, an], axis=1)               # (n_frames,) per-pixel level
        flux += cut[:, ap].sum(axis=1) - ap.sum() * local_bg
        n_ap_px += int(ap.sum())

    return dict(flux=flux, n_apertures=int(positions.shape[0]), aperture_pixels=n_ap_px,
                positions=positions)


def fit_fluorescence_loss(flux: np.ndarray, *, numb_photo_bleach: int = 100,
                          lambda_rate: Optional[float] = None,
                          frame_time_seconds: float = 0.02) -> dict:
    """Recover ``prob_photo_bleach`` from a total-flux decay curve.

    The survival fraction of an independently bleaching dye population is
    ``(1 - p)^(t / numb_photo_bleach)`` exactly -- the per-frame Bernoulli rate is defined so
    that the probability accrues to ``p`` over the FIXED reference window, whatever the clip
    length. So the flux is ``A (1-p)^(t/N) + B`` and the model is fitted in LINEAR space with
    a free offset ``B``.

    Two details matter and both were measured. Fitting ``log flux`` by ordinary least squares
    is biased: as the curve decays into the noise the log transform pulls the late points
    down and steepens the slope, which overestimates ``p`` -- measured at +0.20 dex at the top
    of the prior on 1000-frame curves. And the offset must be free: a local annulus estimate
    carries a small residual, and forcing the curve through zero converts that residual
    directly into slope.

    The returned standard error is inflated by the flicker correlation when ``lambda_rate`` is
    supplied, because consecutive frames are not independent samples of the decay.

    Args:
        flux: ``(n_frames,)`` net aperture flux, from `spot_flux_curve`.
        numb_photo_bleach: the fixed reference window (100).
        lambda_rate: flicker rate, for the effective-sample-size correction of the error.
        frame_time_seconds: frame interval.

    Returns:
        ``dict`` with ``prob_photo_bleach``, ``prob_se``, ``amplitude``, ``offset``,
        ``rate_per_frame``, ``n_eff``, ``success``, ``resid_sd`` (standard deviation of the fit
        residuals, the scale the decay's visibility is judged against), and ``n_frames``.
    """
    y = np.asarray(flux, dtype=np.float64)
    n = y.size
    t = np.arange(n, dtype=np.float64)
    if n < 20 or not np.all(np.isfinite(y)):
        return dict(prob_photo_bleach=np.nan, prob_se=np.nan, amplitude=np.nan,
                    offset=np.nan, rate_per_frame=np.nan, n_eff=np.nan, success=False,
                    resid_sd=np.nan, n_frames=int(n))

    npb = int(numb_photo_bleach)
    k = max(n // 10, 1)
    b0 = float(np.median(y[-k:]))
    # The amplitude is the total DROP over the recording, not the opening level: the curve is
    # a frame total minus a constant background estimate, so its absolute level is arbitrary
    # and can be far below zero. Started from the opening level with a slow rate, the optimizer
    # collapsed a fast decay (p = 0.316) to a flat line -- amplitude 9.5 on an offset of
    # -900,000 -- and reported success. A multi-start over the rate, keeping the lowest cost,
    # removes that dependence on the guess.
    a0 = float(max(y[:k].mean() - b0, 1.0))

    def residual(p):
        amp, rate, off = p
        return amp * np.exp(-rate * t) + off - y

    fit = None
    for r0 in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
        try:
            cand = least_squares(residual, x0=[a0, r0, b0],
                                 bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]),
                                 x_scale=[max(a0, 1.0), r0, max(abs(b0), 1.0)], max_nfev=400)
        except Exception:
            continue
        if fit is None or cand.cost < fit.cost:
            fit = cand
    if fit is None:
        return dict(prob_photo_bleach=np.nan, prob_se=np.nan, amplitude=np.nan,
                    offset=np.nan, rate_per_frame=np.nan, n_eff=np.nan, success=False,
                    resid_sd=np.nan, n_frames=int(n))

    amp, rate, off = float(fit.x[0]), float(fit.x[1]), float(fit.x[2])
    prob = float(1.0 - np.exp(-npb * rate))

    # Standard error of the rate from the Jacobian, scaled by the residual variance and,
    # when the flicker rate is known, by the loss of independent samples.
    prob_se = np.nan
    n_eff = float(n)
    dof = max(n - 3, 1)
    resid_sd = float(np.sqrt(2.0 * fit.cost / dof))
    try:
        s2 = resid_sd ** 2
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * s2
        rate_se = float(np.sqrt(max(cov[1, 1], 0.0)))
        if lambda_rate is not None:
            rho = float(np.exp(-float(lambda_rate) * float(frame_time_seconds)))
            n_eff = max(n * (1.0 - rho) / (1.0 + rho), 1.0)
            rate_se *= float(np.sqrt(n / n_eff))
        prob_se = float(npb * np.exp(-npb * rate) * rate_se)
    except np.linalg.LinAlgError:
        pass

    return dict(prob_photo_bleach=prob, prob_se=prob_se, amplitude=amp, offset=off,
                rate_per_frame=rate, n_eff=n_eff, success=bool(fit.success),
                resid_sd=resid_sd, n_frames=int(n))


def frame_flux_curve(video_levels: np.ndarray, *, background_quantile: float = 0.5) -> dict:
    """Total emitter flux per frame, estimated over the WHOLE field.

    The motion-immune alternative to `spot_flux_curve`. Emitters diffuse: over a 20 s
    recording a receptor at a typical diffusion coefficient wanders on the order of ten
    pixels, so an aperture pinned to a spot's starting position loses it within seconds. The
    flux inside fixed apertures then decays because the emitters WALK OUT of them, which is
    a far larger effect than photobleaching and is indistinguishable from it in the fitted
    rate -- measured on rendered 20 s recordings, fixed apertures returned a bleaching
    probability near 0.5 for every true value from 0.01 to 0.316. Summing the whole field
    removes that failure by construction: an emitter moving within the field does not change
    the total at all.

    The background is removed per frame by a quantile of the frame's own pixels. Emitters
    occupy a small fraction of a 256 x 256 field, so the median pixel is background, and a
    per-frame estimate also absorbs any drift. This matters because the absolute background
    cannot be trusted at the level required: the optical floor contributes roughly a hundred
    times the emitter signal, and the stored 8-bit quantization alone shifts the frame sum by
    more than the entire emitter signal.

    What this observable does NOT survive is emitters leaving the FIELD, which is a genuine
    loss of flux that no per-frame background estimate can distinguish from bleaching. On a
    trajectory tier that turnover is part of the measured error and is reported as such.

    Args:
        video_levels: ``(n_frames, n_x, n_y)`` stored 8-bit video.
        background_quantile: quantile of each frame taken as its background level.

    Returns:
        ``dict`` with ``flux`` (``(n_frames,)`` net ADU over the field), ``background``
        (``(n_frames,)`` per-pixel level in ADU) and ``n_pixels``.
    """
    video = np.asarray(video_levels)
    if video.ndim != 3:
        raise ValueError(f"video_levels has shape {video.shape}; expected (n_frames, n_x, n_y).")
    n_frames = video.shape[0]
    n_pixels = int(video.shape[1] * video.shape[2])
    flat = video.reshape(n_frames, n_pixels)
    # Work in the stored integer domain for the quantile, then convert once: the quantile of a
    # heavily quantized array is exact, and converting first would only add float noise.
    bg_levels = np.quantile(flat, float(background_quantile), axis=1)
    total = levels_to_adu(flat.sum(axis=1))
    background = levels_to_adu(bg_levels)
    return dict(flux=total - n_pixels * background, background=background,
                n_pixels=n_pixels)


# =============================================================================
# Flicker rate: lambda_rate from the autocorrelation of per-spot ln-brightness
# =============================================================================
#
# This mirrors the method of the Flicker_Rate_Derivation utility, which fixed the
# lambda_rate prior from ThunderSTORM localization tables. The steps are deliberately
# identical -- per-trace log, per-trace linear detrend, gap-aware pooled autocorrelation,
# normalization at LAG ONE, and a shape match against an Ornstein-Uhlenbeck model arm cut to
# the empirical track-span distribution -- so that a value measured here from VIDEO and the
# recorded value measured from localization tables are directly comparable, and a
# disagreement between them is informative rather than a difference of technique.
#
# The functions are reimplemented here rather than imported from that utility: it is a
# special-situation script that carries the recorded 5.1/4.7 result, and it is left untouched
# so that result keeps its provenance.
#
# Why the model arm simulates ONE dye. In the single-dye case the method is exactly free of
# the brightness parameters: mu_pc shifts ln-brightness additively and sigma_pc scales it
# linearly, so both vanish under the detrend and the normalization. Summing several dyes
# before the logarithm breaks that exactness -- ln(sum of k lognormals) is not an
# Ornstein-Uhlenbeck process and its normalized autocorrelation picks up a dependence on
# sigma_pc. Modeling the multiplicity would remove a small bias at the price of making the
# estimator depend on a parameter that is itself under inference, which is circular in a
# campaign that exists to decide which parameters can be pinned independently. The bias is
# therefore left in and BOUNDED instead: measured against the renderer it is 0.0077 in shape
# at sigma_pc = 0.42 (about 2% in lambda_rate, 0.009 dex) and at most 0.04 in shape at the
# top corner sigma_pc = 1.0 with three dyes (about 0.05 dex). `flicker_multiplicity_band`
# reports that range without needing a sigma_pc value.

_ACF_MAX_LAG = 40           # lags accumulated, matching the derivation of record
_ACF_FIT_LAGS = 12          # lags the shape match uses
_ACF_MIN_PAIRS = 2000       # pooled pairs required at lag 1 before a shape is trusted


def _accumulate_acf(series: np.ndarray, csum: np.ndarray, cnt: np.ndarray,
                    max_lag: int = _ACF_MAX_LAG) -> None:
    """Accumulate one NaN-padded series into a pooled, unnormalized lag product-sum.

    NaN marks a gap -- a frame in which the spot was not detected, or in which its
    background-subtracted flux was not positive. Pairs touching a gap are skipped, so a
    track with holes contributes its valid pairs instead of being discarded. Mutates
    ``csum`` and ``cnt`` in place.
    """
    n = series.shape[0]
    for k in range(min(max_lag, n - 1) + 1):
        a, b = (series, series) if k == 0 else (series[:n - k], series[k:])
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 5:
            csum[k] += float(np.sum(a[m] * b[m]))
            cnt[k] += int(m.sum())


def _acf_shape(csum: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """Normalize a pooled product-sum into the flicker shape, discarding lag 0.

    Dividing by lag ONE rather than lag zero is what removes additive white measurement
    noise: such noise is uncorrelated between frames, so it contributes at lag 0 only. For a
    video-derived trace that noise is the per-frame photometry error, which is exactly the
    nuisance this normalization is needed for. The returned shape has its first entry equal
    to one by construction.
    """
    rho = csum / np.maximum(cnt, 1)
    rho = rho / rho[0]
    return rho[1:] / rho[1]


def spot_intensity_traces(measurements: dict, track_id: np.ndarray, *,
                          min_length: int = 40) -> Tuple[list, np.ndarray]:
    """Group per-spot-frame fitted amplitudes into per-track intensity traces.

    The fitted amplitude of `fit_spot_width` is the spot's total signal in ADU with the
    background already accounted for, so it is the photometric observable the flicker
    autocorrelation needs, and it comes free from the measurement pass the width estimator
    already runs.

    Args:
        measurements: the dict returned by `measure_spot_widths`, which must have been run
            with ``frame_stride = 1``; the autocorrelation is indexed in frames, so a stride
            would silently rescale every lag.
        track_id: per-measurement track labels from `link_spot_tracks`.
        min_length: shortest track kept, in frames.

    Returns:
        ``(traces, spans)``: ``traces`` is a list of ``(frame_index, amplitude)`` pairs and
        ``spans`` the integer frame span of each, for the matched-length model arm.
    """
    frames = np.asarray(measurements["frame_index"])
    amps = np.asarray(measurements["amplitude"], dtype=np.float64)
    tid = np.asarray(track_id)
    traces, spans = [], []
    for t in np.unique(tid):
        sel = tid == t
        f, a = frames[sel], amps[sel]
        order = np.argsort(f)
        f, a = f[order], a[order]
        if f.size < 3:
            continue
        span = int(f.max() - f.min() + 1)
        if span < int(min_length):
            continue
        traces.append((f, a))
        spans.append(span)
    return traces, np.asarray(spans, dtype=np.int64)


def flicker_pooled_acf(traces: list) -> Tuple[np.ndarray, np.ndarray]:
    """Pooled lag product-sums and pair counts of the detrended ln-intensity traces, lags 0 to
    ``_ACF_MAX_LAG``.

    This is the raw material of `flicker_data_shape`, which normalizes it and discards lag 0.
    It is exposed for diagnostics that need lag 0, which the shape discards by normalizing at
    lag 1. The ratio of the lag-0 to the lag-1 value is a diagnostic ratio, not a measurement of
    the photometry noise: variance uncorrelated between frames raises it, but the flicker itself,
    the detrend and the gaps move it too. Each trace is logged and linearly detrended against its own frame index, and
    non-positive samples become gaps (see `flicker_data_shape`).

    Returns:
        ``(csum, cnt)``: the pooled product-sums and the number of pairs at each lag.
    """
    csum = np.zeros(_ACF_MAX_LAG + 1)
    cnt = np.zeros(_ACF_MAX_LAG + 1)
    for f, a in traces:
        length = int(f.max() - f.min() + 1)
        series = np.full(length, np.nan)
        good = a > 0
        series[(f - f.min())[good]] = np.log(a[good])
        finite = np.isfinite(series)
        if finite.sum() < 5:
            continue
        t = np.arange(length, dtype=float)
        coef = np.polyfit(t[finite], series[finite], 1)
        series = series - np.polyval(coef, t)
        _accumulate_acf(series, csum, cnt)
    return csum, cnt


def flicker_data_shape(traces: list) -> Tuple[Optional[np.ndarray], float]:
    """Pooled, detrended, lag-1-normalized ln-intensity autocorrelation of the traces.

    Each trace is logged and linearly detrended against its own frame index, which removes
    photobleaching and the per-emitter mean together -- both are multiplicative in intensity
    and so additive in the log. Non-positive samples become gaps rather than causing the
    whole track to be dropped: background-subtracted video photometry goes negative for dim
    frames, and discarding those tracks entirely would select against faint emitters.

    Returns:
        ``(shape, tau_seconds_placeholder)``: the decay shape over lags 1.., or ``None`` if
        too few pairs were pooled, and the interpolated 1/e crossing in FRAMES (``nan`` if
        the shape never crosses).
    """
    csum, cnt = flicker_pooled_acf(traces)
    if cnt[1] < _ACF_MIN_PAIRS:
        return None, float("nan")
    shape = _acf_shape(csum, cnt)
    return shape, _one_over_e_lag(shape)


def _one_over_e_lag(shape: np.ndarray) -> float:
    """Interpolated lag, in frames, at which a decay shape crosses ``1/e``."""
    target = float(np.exp(-1.0))
    for i in range(1, shape.size):
        if shape[i] <= target <= shape[i - 1]:
            lo, hi = shape[i - 1], shape[i]
            frac = 0.0 if hi == lo else (lo - target) / (lo - hi)
            return float(i + frac)          # +1 offset: shape[0] is lag 1
    return float("nan")


def flicker_model_shape(lambda_rate: float, spans: np.ndarray, *,
                        n_traces: int = 4000, frame_time_seconds: float = 0.02,
                        seed: int = 0) -> np.ndarray:
    """Model flicker shape at one ``lambda_rate``, cut to the empirical span distribution.

    Single-dye Ornstein-Uhlenbeck traces from the production generator, with bleaching off
    (the data arm detrends it away). Cutting each simulated trace to a length drawn from the
    observed spans, and detrending it identically, is what nulls the finite-track detrending
    bias -- the reason the bare ``1 / tau_corr`` reads high.

    ``mu_pc`` and ``sigma_pc`` are immaterial here and are passed at reference values: the
    first shifts ln-brightness additively and the second scales it linearly, so both vanish
    under the detrend and the lag-1 normalization.
    """
    from .simulation_dli_support import generate_brightness_photons

    nfmax = int(min(np.percentile(spans, 99), 400)) + 5
    photons = generate_brightness_photons(
        nframes=nfmax, nemitters=int(n_traces), mu_pc=386.0, sigma_pc=0.60,
        lambda_rate=float(lambda_rate), prob_photo_bleach=0.0, numb_photo_bleach=100,
        delta_frame=float(frame_time_seconds), seed=seed)
    x = np.log(photons)
    rng = np.random.default_rng(123)
    lens = np.clip(rng.choice(spans, int(n_traces)), 10, nfmax)
    starts = np.array([rng.integers(0, nfmax - int(L) + 1) for L in lens])

    csum = np.zeros(_ACF_MAX_LAG + 1)
    cnt = np.zeros(_ACF_MAX_LAG + 1)
    # Grouped by length so the detrend and the lag products run vectorized over every trace
    # of that length at once. A per-trace Python loop here is the dominant cost of the whole
    # estimator -- it runs once per grid point per recording -- and the model arm has no gaps,
    # so the general NaN-aware accumulator is not needed.
    for length in np.unique(lens):
        sel = np.nonzero(lens == length)[0]
        L = int(length)
        segs = np.empty((L, sel.size), dtype=np.float64)
        for j, e in enumerate(sel):
            s0 = int(starts[e])
            segs[:, j] = x[s0:s0 + L, e]
        t = np.arange(L, dtype=float)
        coef = np.polyfit(t, segs, 1)                       # (2, n_sel), one fit per column
        segs = segs - (np.outer(t, coef[0]) + coef[1])
        for k in range(min(_ACF_MAX_LAG, L - 1) + 1):
            a = segs if k == 0 else segs[:L - k]
            b = segs if k == 0 else segs[k:]
            csum[k] += float(np.sum(a * b))
            cnt[k] += int((L - k) * sel.size)
    return _acf_shape(csum, cnt)


FLICKER_GRID_DEFAULT = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0])


def flicker_model_shapes(spans: np.ndarray, *, grid: Optional[np.ndarray] = None,
                         frame_time_seconds: float = 0.02, n_traces: int = 4000) -> Tuple[np.ndarray, list]:
    """The model-arm shapes for every grid point, computed once per recording.

    They depend on the recording only through its span distribution, so a bootstrap over the
    recording's traces can reuse them; recomputing them is the dominant cost of the estimator.
    """
    grid = FLICKER_GRID_DEFAULT if grid is None else np.asarray(grid, dtype=float)
    shapes = [flicker_model_shape(lam, spans, n_traces=n_traces,
                                  frame_time_seconds=frame_time_seconds) for lam in grid]
    return grid, shapes


def match_shapes(data_shape: np.ndarray, grid: np.ndarray, shapes: list) -> dict:
    """Grid search plus an exact parabolic refinement in log-lambda on the real grid coordinates.

    The refinement fits the parabola through the three bracketing ``(log lambda, residual)``
    points in their actual coordinates. The grid is uneven in log-lambda, so the classical
    equal-spacing formula does not apply: it returned 4.3506 for a quadratic objective whose
    minimum was at 4.3. Two checks guard the refinement -- the parabola must curve upward and
    its vertex must fall inside the bracketing interval -- and when either fails the grid
    minimum is returned unrefined, with ``refined=False``.
    """
    k = min(_ACF_FIT_LAGS, data_shape.size, min(s.size for s in shapes))
    l2 = np.array([float(np.sum((s[:k] - data_shape[:k]) ** 2)) for s in shapes])
    i = int(np.argmin(l2))
    ll = np.log(np.asarray(grid, dtype=float))
    lam, refined = float(grid[i]), False
    if 0 < i < len(grid) - 1:
        x, y = ll[i - 1:i + 2], l2[i - 1:i + 2]
        a2, a1, _ = np.polyfit(x, y, 2)
        if a2 > 0:
            vertex = -a1 / (2.0 * a2)
            if x[0] <= vertex <= x[2]:
                lam, refined = float(np.exp(vertex)), True
    return dict(lambda_rate=lam, lambda_grid_best=float(grid[i]), residual=float(l2[i]),
                refined=refined, at_grid_edge=bool(i == 0 or i == len(grid) - 1))


def match_flicker_rate(data_shape: np.ndarray, spans: np.ndarray, *,
                       grid: Optional[np.ndarray] = None,
                       frame_time_seconds: float = 0.02,
                       n_traces: int = 4000) -> dict:
    """Recover ``lambda_rate`` by matching the early shape against the model arm.

    Sum of squares over the first ``_ACF_FIT_LAGS`` lags, then the exact parabolic refinement
    of `match_shapes` in log-lambda between the bracketing grid points.

    Returns:
        ``dict`` with ``lambda_rate``, ``lambda_grid_best``, ``residual``, ``refined``,
        ``at_grid_edge``, and ``tau_seconds`` (from the data shape's 1/e crossing, as a
        cross-check).
    """
    if data_shape is None:
        return dict(lambda_rate=np.nan, lambda_grid_best=np.nan, residual=np.nan,
                    refined=False, at_grid_edge=False, tau_seconds=np.nan)
    grid, shapes = flicker_model_shapes(spans, grid=grid, frame_time_seconds=frame_time_seconds,
                                        n_traces=n_traces)
    out = match_shapes(data_shape, grid, shapes)
    tau_lag = _one_over_e_lag(data_shape)
    out["tau_seconds"] = (float(tau_lag * frame_time_seconds) if np.isfinite(tau_lag)
                          else float("nan"))
    return out


def flicker_bootstrap_range(traces: list, grid: np.ndarray, shapes: list, *,
                            n_boot: int = 40, seed: int = 0,
                            quantiles: Tuple[float, float] = (0.05, 0.95)) -> dict:
    """Per-recording nominal 90 % range of ``lambda_rate`` by bootstrapping the traces.

    Traces are resampled with replacement, the pooled data shape is recomputed for each
    resample, and each is matched against the SAME model shapes (which depend on the span
    distribution, left at the original recording's). The range therefore carries the
    trace-to-trace scatter of the measurement -- detection, photometry, linking and gaps
    included, since those are what shaped the traces -- but not the single-dye model
    approximation, the detrending approximation, or the fixed-span approximation; those are
    what the coverage measurement of DETECTOR_WORKFLOW.md sec. 9.6 tests.

    Returns ``dict`` with ``low``, ``high`` (in lambda units), ``n_boot_valid`` and the
    bootstrap ``sd_log10``.
    """
    rng = np.random.default_rng(seed)
    n = len(traces)
    vals = []
    if n >= 5:
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            shape, _ = flicker_data_shape([traces[j] for j in idx])
            if shape is None:
                continue
            vals.append(match_shapes(shape, grid, shapes)["lambda_rate"])
    vals = np.asarray([v for v in vals if np.isfinite(v) and v > 0], dtype=float)
    if vals.size < max(8, n_boot // 4):
        return dict(low=np.nan, high=np.nan, n_boot_valid=int(vals.size), sd_log10=np.nan)
    lo, hi = np.quantile(vals, quantiles)
    return dict(low=float(lo), high=float(hi), n_boot_valid=int(vals.size),
                sd_log10=float(np.std(np.log10(vals), ddof=1)))


def flicker_multiplicity_band(lambda_rate: float) -> Tuple[float, float]:
    """Systematic band on ``lambda_rate`` from dye multiplicity, in dex.

    The single-dye model arm keeps the estimator free of ``mu_pc`` and ``sigma_pc``, at the
    cost of a small bias when a spot carries several dyes whose photons sum before the
    logarithm. Measured against the renderer across the ``sigma_pc`` prior and dye counts one
    to three, the shape displacement is 0.004 to 0.04, against roughly 0.035 for a ten
    percent change in ``lambda_rate``. The corresponding band is about 0.009 dex at the
    center of the ``sigma_pc`` prior and 0.05 dex at its top corner.

    Returned as a band rather than a correction precisely so that no ``sigma_pc`` value is
    needed: the estimator reports its own systematic instead of depending on a parameter that
    is itself under inference.

    Args:
        lambda_rate: the estimate the band applies to (the band is multiplicative in dex and
            does not depend on it; the argument is taken for call-site clarity).

    Returns:
        ``(typical_dex, worst_dex)``.
    """
    return 0.009, 0.05
