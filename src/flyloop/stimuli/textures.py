"""Procedural textures with a 1/f^alpha power spectrum, and ways to show them.

`texture_1f` makes a periodic image whose power spectrum falls as 1/f^alpha
(natural scenes: alpha ~ 2). Wrap it around a drum and sample it per receptor with
`sample_panorama`/`rotating_texture`, or turn any cartesian movie into flyvis input
with `frames_to_hex`, which uses flyvis's own BoxEye.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import StimulusSequence, render_on_lattice, to_tensor
from .lattice import HexLattice


def _spectral_noise(shape: tuple[int, int], alpha: float, rng: np.random.Generator):
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = np.inf  # zero DC
    amplitude = f ** (-alpha / 2)  # power ~ amplitude^2 ~ 1/f^alpha
    z = np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * amplitude).real
    return (z - z.mean()) / z.std()


def texture_1f(
    shape: tuple[int, int] = (360, 720),
    alpha: float = 2.0,
    contrast: float = 1.0,
    density: float = 1.0,
    mean_lum: float = 0.5,
    binarize: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Periodic texture with power spectrum ~ 1/f^alpha.

    Args:
        shape: (height, width) in pixels. The texture tiles seamlessly, so it can
            be wrapped around a drum.
        alpha: Spectral slope of the power spectrum. 0 is white noise, 2 is
            natural-scene-like, larger values give smoother blobs.
        contrast: Peak (Michelson) contrast; values span mean_lum * (1 +- contrast).
        density: Fraction of the image covered by texture, in (0, 1]. The rest is
            set to mean_lum, using a mask thresholded from a second independent
            field with the same spectrum, so texture comes in patches.
        mean_lum: Mean luminance of the untextured background.
        binarize: If True, the texture is two-level (sign of the field).
        seed: Random seed.

    Returns:
        float32 image (height, width) with values in [0, 1] for contrast <= 1 and
        mean_lum = 0.5.
    """
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    rng = np.random.default_rng(seed)
    z = _spectral_noise(shape, alpha, rng)
    z = np.sign(z) if binarize else z / np.abs(z).max()
    if density < 1:
        m = _spectral_noise(shape, alpha, rng)
        z = z * (m >= np.quantile(m, 1 - density))
    return (mean_lum * (1 + contrast * z)).astype(np.float32)


def sample_panorama(
    panorama: np.ndarray, az: np.ndarray, el: np.ndarray, el_range=(-90.0, 90.0)
) -> np.ndarray:
    """Bilinearly sample an equirectangular panorama at azimuth/elevation (deg).

    Columns span azimuth -180..180 (wrapping); rows span el_range top to bottom
    (row 0 is the highest elevation), clamped at the edges.
    """
    h, w = panorama.shape
    col = (np.asarray(az) + 180.0) / 360.0 * w - 0.5
    row = (el_range[1] - np.asarray(el)) / (el_range[1] - el_range[0]) * h - 0.5
    row = np.clip(row, 0, h - 1)
    c0 = np.floor(col).astype(int)
    r0 = np.floor(row).astype(int)
    fc, fr = col - c0, row - r0
    c0w, c1w = c0 % w, (c0 + 1) % w
    r1 = np.minimum(r0 + 1, h - 1)
    top = panorama[r0, c0w] * (1 - fc) + panorama[r0, c1w] * fc
    bot = panorama[r1, c0w] * (1 - fc) + panorama[r1, c1w] * fc
    return top * (1 - fr) + bot * fr


def rotating_texture(
    panorama: np.ndarray,
    omega_deg_s: float,
    duration_s: float = 2.0,
    t_pre: float = 0.5,
    dt: float = 1 / 200,
    eye_azimuth_deg: float = 0.0,
    eye_elevation_deg: float = 0.0,
    mirror: bool = False,
    lattice: HexLattice | None = None,
    device: str | torch.device | None = None,
) -> StimulusSequence:
    """A textured drum rotating about the vertical axis (same convention as
    `rotating_grating`: omega > 0 moves the texture toward increasing azimuth)."""
    lattice = lattice or HexLattice()
    az, el = lattice.to_sphere(
        *lattice.sample_points(), eye_azimuth_deg, eye_elevation_deg, mirror
    )
    times = np.arange(int(round((t_pre + duration_s) / dt))) * dt - t_pre

    def lum(az, el, t):
        return sample_panorama(panorama, az - omega_deg_s * max(t, 0.0), el)

    frames = render_on_lattice(lum, times, lattice, (az, el))
    params = dict(
        kind="rotating_texture",
        omega_deg_s=omega_deg_s,
        eye_azimuth_deg=eye_azimuth_deg,
        eye_elevation_deg=eye_elevation_deg,
        mirror=mirror,
    )
    return StimulusSequence(to_tensor(frames, device), dt, t_pre, params)


def frames_to_hex(
    frames: np.ndarray | torch.Tensor, extent: int = 15, kernel_size: int = 13
) -> torch.Tensor:
    """Render cartesian frames (n_frames, height, width) onto the hex lattice with
    flyvis's BoxEye (box-filter mean over each receptor's kernel_size^2 pixels).

    The frame is taken to span the eye's field of view; BoxEye rescales it if it
    is smaller than (2 * extent + 1) * kernel_size pixels. Returns
    (n_frames, 1, n_hexals).
    """
    from flyvis.datasets.rendering import BoxEye

    frames = torch.as_tensor(np.asarray(frames), dtype=torch.float32)
    return BoxEye(extent=extent, kernel_size=kernel_size)(frames[None])[0]
