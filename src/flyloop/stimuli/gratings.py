"""Square-wave gratings: drifting (planar, any direction) and full-field rotating."""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .base import StimulusSequence, render_on_lattice, square, to_tensor
from .lattice import HexLattice


def _times(duration_s: float, t_pre: float, dt: float) -> np.ndarray:
    return np.arange(int(round((t_pre + duration_s) / dt))) * dt - t_pre


def drifting_grating(
    direction_deg: float,
    spatial_period_deg: float = 30.0,
    temporal_freq_hz: float = 2.0,
    contrast: float = 1.0,
    mean_lum: float = 0.5,
    duration_s: float = 2.0,
    t_pre: float = 0.5,
    pre: str = "grey",
    dt: float = 1 / 200,
    lattice: HexLattice | None = None,
    device: str | torch.device | None = None,
) -> StimulusSequence:
    """Square-wave grating drifting in `direction_deg` across the lattice plane.

    Luminance is mean_lum * (1 + contrast * square(2 pi (s / period - f t))) with
    s = x cos(theta) + y sin(theta), so the pattern moves toward theta at
    period * f deg/s. Directions follow flyvis MovingEdge angles (0 = rightward,
    90 = upward). During t_pre the screen is grey (pre="grey") or shows the
    grating at its t=0 phase (pre="static").
    """
    lattice = lattice or HexLattice()
    th = np.radians(direction_deg)

    def lum(x, y, t):
        if t < 0 and pre == "grey":
            return np.full_like(x, mean_lum)
        s = x * np.cos(th) + y * np.sin(th)
        phase = 2 * np.pi * (s / spatial_period_deg - temporal_freq_hz * max(t, 0.0))
        return mean_lum * (1 + contrast * square(phase))

    frames = render_on_lattice(lum, _times(duration_s, t_pre, dt), lattice)
    params = dict(
        kind="drifting_grating",
        direction_deg=direction_deg,
        spatial_period_deg=spatial_period_deg,
        temporal_freq_hz=temporal_freq_hz,
        contrast=contrast,
        mean_lum=mean_lum,
    )
    return StimulusSequence(to_tensor(frames, device), dt, t_pre, params)


def rotating_grating(
    omega_deg_s: float,
    spatial_period_deg: float = 30.0,
    contrast: float = 1.0,
    mean_lum: float = 0.5,
    duration_s: float = 2.0,
    t_pre: float = 0.5,
    dt: float = 1 / 200,
    eye_azimuth_deg: float = 0.0,
    eye_elevation_deg: float = 0.0,
    mirror: bool = False,
    lattice: HexLattice | None = None,
    device: str | torch.device | None = None,
) -> StimulusSequence:
    """Vertical square-wave stripes on a drum rotating about the vertical axis.

    Luminance depends only on world azimuth: mean_lum * (1 + contrast *
    square(2 pi (az - omega t) / period)). omega > 0 moves the pattern toward
    increasing azimuth (counter-clockwise seen from above), which is the optic
    flow of a clockwise (rightward) yaw turn of the fly. Receptor directions come
    from `HexLattice.to_sphere`, so flow near the eye's poles is curved as on a
    real drum. The period is rounded so an integer number of cycles fits in 360 deg.
    The drum is static (at t=0 phase) during t_pre.
    """
    lattice = lattice or HexLattice()
    n_cycles = max(1, int(round(360.0 / spatial_period_deg)))
    period = 360.0 / n_cycles
    if not np.isclose(period, spatial_period_deg):
        warnings.warn(
            f"spatial_period_deg={spatial_period_deg} rounded to {period:.3f} "
            f"({n_cycles} cycles per 360 deg)",
            stacklevel=2,
        )
    az, el = lattice.to_sphere(
        *lattice.sample_points(), eye_azimuth_deg, eye_elevation_deg, mirror
    )

    def lum(az, el, t):
        phase = 2 * np.pi * (az - omega_deg_s * max(t, 0.0)) / period
        return mean_lum * (1 + contrast * square(phase))

    frames = render_on_lattice(lum, _times(duration_s, t_pre, dt), lattice, (az, el))
    params = dict(
        kind="rotating_grating",
        omega_deg_s=omega_deg_s,
        spatial_period_deg=period,
        contrast=contrast,
        mean_lum=mean_lum,
        eye_azimuth_deg=eye_azimuth_deg,
        eye_elevation_deg=eye_elevation_deg,
        mirror=mirror,
    )
    return StimulusSequence(to_tensor(frames, device), dt, t_pre, params)
