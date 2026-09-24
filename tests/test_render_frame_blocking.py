"""The renderer integrates the PSF in 2 s frame blocks; its output must not change.

`simulation_dli_support.add_pixel_counts` integrates the Gaussian PSF over the pixel grid in
consecutive blocks of `PSF_FRAME_BLOCK` frames, the frame count of a 2 s recording (100 at 50
frames per second). The integration's temporaries scale with pixels x emitters x frames, so
blocking holds them at the size of one 2 s recording whatever the recording length, and a 1 s or
2 s recording still runs in a single pass. These tests hold the blocked integration to the
single-pass computation it replaced, bit for bit, at every documented duration (1, 2, 5, 10 and
20 s): the noise-free intensity, and the rendered video under a fixed seed. They also show that
the block size never enters the result and that the default block layout is the 2 s one.

The scenes exercise what the production renderer meets: monomers and dimers that dissociate
mid-recording, subunits with zero to three dyes, particles absent for part of the recording
(NaN poses, the renderer's ghosts) and bleaching at the top of its prior (dyes at zero photons).

Runnable with ``python -m pytest`` or directly:
``MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python tests/test_render_frame_blocking.py``.
"""
import contextlib

import numpy as np

from srm_and_sbi_monomer_dimer_alp import detector_parameterization as det
from srm_and_sbi_monomer_dimer_alp import simulation_dli_support as dli
from srm_and_sbi_monomer_dimer_alp.parameterization import PARAMETERS

DURATIONS_SECONDS = (1.0, 2.0, 5.0, 10.0, 20.0)      # the documented duration set
FRAME_TIME = PARAMETERS.simulation.timing.frame_time_seconds


def _frames(seconds: float) -> int:
    return int(round(seconds / FRAME_TIME))


def _add_pixel_counts_single_pass(intensity, tracks, brightness_array, xbounds, ybounds, PSF):
    """`add_pixel_counts` before frame blocking (0.1.17), kept verbatim as the reference."""
    ghost_mask = np.isnan(tracks)
    eternity_bound = np.max([xbounds, ybounds]) * 2
    tracks[ghost_mask] = eternity_bound
    ghost_mask_zero = np.moveaxis(ghost_mask, 0, 2)[[0], :, :]
    brightness_array[ghost_mask_zero] = 0
    X = dli._erf(tracks[:, [0], :], xbounds, PSF.sqrt_2sigma)
    X *= brightness_array
    Y = dli._erf(tracks[:, [1], :], ybounds, PSF.sqrt_2sigma)
    Y = np.transpose(Y, (1, 0, 2))
    intensity += np.einsum("ijk,jlk->ilk", X, Y)
    return intensity


@contextlib.contextmanager
def _single_pass_renderer():
    """Route `compute_intensity` (and so `render_dli_video`) through the reference integration."""
    blocked = dli.add_pixel_counts
    dli.add_pixel_counts = _add_pixel_counts_single_pass
    try:
        yield
    finally:
        dli.add_pixel_counts = blocked


def _imaging_vector(prob_photo_bleach: float = 10 ** -0.5) -> np.ndarray:
    """Prior centers of the eleven imaging keys, bleaching at the top of its prior."""
    img = {e["KEY"]: 10 ** (0.5 * (e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1])) for e in det.DETECTOR_IMAGING}
    img["prob_photo_bleach"] = prob_photo_bleach
    return np.array([img[k] for k in det.DETECTOR_IMAGING_KEYS])


def _scene(n_frames: int, seed: int, n_subunits: int = 160):
    """Poses, subunit lineage and dye counts of a diffusing monomer-dimer scene."""
    rng = np.random.default_rng(seed)
    stem = PARAMETERS.simulation.stem
    box = stem.root_size_px * stem.pixel_size_nm
    step = np.sqrt(2.0 * 0.05e6 * FRAME_TIME)          # nm per frame at D = 0.05 um^2/s
    n_dimers = n_subunits // 8                          # particles that host two subunits
    n_particles = n_subunits + n_dimers
    pos = rng.uniform(0.05 * box, 0.95 * box, size=(n_particles, 2))
    poses = np.zeros((n_frames, n_particles, 3))
    for t in range(n_frames):
        poses[t, :, :2] = pos
        pos = pos + rng.normal(0.0, step, size=pos.shape)
    for p in rng.choice(n_particles, size=n_particles // 10, replace=False):
        a = int(rng.integers(0, n_frames))                  # absent for frames [a, b)
        b = int(rng.integers(a + 1, n_frames + 1))
        poses[a:b, p, :] = np.nan
    host = np.tile(np.arange(n_subunits), (n_frames, 1))
    for k in range(n_dimers):                                # subunits 2k, 2k+1 bound until `split`
        split = int(rng.integers(0, n_frames + 1))
        host[:split, 2 * k] = n_subunits + k
        host[:split, 2 * k + 1] = n_subunits + k
    dye_counts = rng.choice([0, 1, 2, 3], size=n_subunits, p=[0.3, 0.35, 0.25, 0.1])
    return poses, host, dye_counts


def _intensity_inputs(n_frames: int, seed: int, n_subunits: int = 160):
    """The inputs `render_dli_video` hands to `compute_intensity`, for one scene."""
    poses, host, dye_counts = _scene(n_frames, seed, n_subunits)
    img = dict(zip(det.DETECTOR_IMAGING_KEYS, _imaging_vector()))
    stem = PARAMETERS.simulation.stem
    dye_nm, dye_subunit = dli.build_dye_tracks(poses, host, dye_counts)
    widths = dli.sample_psf_width(host.shape[1], PARAMETERS.simulation.dli.sqrt_2sigma_dist_label,
                                  keyword_args={"mu_r": img["mu_r"], "sigma_r": img["sigma_r"]}, seed=seed)
    photons = dli.generate_brightness_photons(
        nframes=n_frames, nemitters=dye_subunit.shape[0], mu_pc=img["mu_pc"], sigma_pc=img["sigma_pc"],
        lambda_rate=img["lambda_rate"], prob_photo_bleach=img["prob_photo_bleach"],
        numb_photo_bleach=100, delta_frame=FRAME_TIME, seed=seed)
    tracks = np.transpose(dye_nm / stem.pixel_size_nm, (0, 2, 1)).astype(np.float64)
    background = np.full((stem.root_size_px, stem.root_size_px), img["kappa_o"], dtype=float)
    bounds = np.linspace(0, stem.root_size_px, stem.root_size_px + 1)
    return tracks, photons, background, bounds, dli.Gaussian(widths[dye_subunit])


def _intensity(inputs, frame_block=None, single_pass=False):
    tracks, photons, background, bounds, psf = inputs
    intensity = np.repeat(background[:, :, None], tracks.shape[0], axis=2)
    brightness = photons.T.reshape(1, photons.shape[1], photons.shape[0]).copy()
    if single_pass:
        return _add_pixel_counts_single_pass(intensity, tracks.copy(), brightness, bounds, bounds, psf)
    return dli.add_pixel_counts(intensity, tracks.copy(), brightness, bounds, bounds, psf,
                                frame_block=frame_block)


def test_default_block_is_two_seconds():
    assert dli.PSF_FRAME_BLOCK == _frames(2.0) == 100
    assert dli.psf_frame_blocks(_frames(1.0)) == [(0, 50)]
    assert dli.psf_frame_blocks(_frames(2.0)) == [(0, 100)]
    assert dli.psf_frame_blocks(_frames(5.0)) == [(0, 100), (100, 200), (200, 250)]
    assert dli.psf_frame_blocks(_frames(10.0)) == [(a, a + 100) for a in range(0, 500, 100)]
    assert dli.psf_frame_blocks(_frames(20.0)) == [(a, a + 100) for a in range(0, 1000, 100)]
    assert dli.psf_frame_blocks(0) == []
    for bad in (0, -5):
        try:
            dli.psf_frame_blocks(100, bad)
        except ValueError:
            continue
        raise AssertionError(f"frame_block={bad} was accepted")


def test_intensity_bit_identical_at_every_duration():
    for i, seconds in enumerate(DURATIONS_SECONDS):
        inputs = _intensity_inputs(_frames(seconds), seed=100 + i)
        assert np.isnan(inputs[0]).any(), "the scene must contain ghost frames"
        assert (inputs[1] == 0).any(), "the scene must contain bleached dyes"
        blocked = _intensity(inputs)
        reference = _intensity(inputs, single_pass=True)
        assert blocked.shape == (256, 256, _frames(seconds))
        assert np.array_equal(blocked, reference), f"{seconds:g} s: intensity differs from single pass"


def test_rendered_video_bit_identical_at_every_duration():
    vec = _imaging_vector()
    for i, seconds in enumerate(DURATIONS_SECONDS):
        poses, host, dye_counts = _scene(_frames(seconds), seed=200 + i)
        blocked = dli.render_dli_video(poses, host, dye_counts, vec, seed=300 + i)
        with _single_pass_renderer():
            reference = dli.render_dli_video(poses, host, dye_counts, vec, seed=300 + i)
        assert blocked.shape == (256, 256, _frames(seconds))
        assert np.ptp(blocked) > 0
        assert np.array_equal(blocked, reference), f"{seconds:g} s: rendered video differs from single pass"


def test_block_size_never_enters_the_result():
    inputs = _intensity_inputs(_frames(20.0), seed=400, n_subunits=80)
    reference = _intensity(inputs, single_pass=True)
    for block in (1, 2, 3, 7, 64, 99, 101, 128, 333, 999, 1000, 4096):
        assert np.array_equal(_intensity(inputs, frame_block=block), reference), f"frame_block={block}"
    odd = _intensity_inputs(101, seed=401, n_subunits=80)       # default block plus a one-frame tail
    assert np.array_equal(_intensity(odd), _intensity(odd, single_pass=True))


def test_scene_without_dyes_renders_background_only():
    inputs = _intensity_inputs(_frames(5.0), seed=500)
    tracks, photons, background, bounds, _ = inputs
    empty = (tracks[:, :, :0], photons[:, :0], background, bounds, dli.Gaussian(np.zeros(0)))
    blocked = _intensity(empty)
    assert np.array_equal(blocked, _intensity(empty, single_pass=True))
    assert np.array_equal(blocked, np.repeat(background[:, :, None], _frames(5.0), axis=2))


if __name__ == "__main__":
    import sys
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
