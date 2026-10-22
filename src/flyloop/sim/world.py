"""Visual worlds sampled by the eyes.

A world exposes `luminance(azimuth_deg, elevation_deg)` for world-frame directions
(azimuth counter-clockwise from the world's 0, seen from above).
"""

from __future__ import annotations

import numpy as np

from flyloop.stimuli.base import square
from flyloop.stimuli.textures import sample_panorama, texture_1f


class Drum:
    """A panorama on a drum that can rotate about the vertical axis.

    The panorama is equirectangular: columns span azimuth -180..180, rows span
    el_range (top row = highest elevation). `phase_deg` is the drum's rotation;
    increasing it moves the pattern toward increasing azimuth (counter-clockwise
    seen from above), as in `flyloop.stimuli.rotating_grating`.
    """

    def __init__(self, panorama: np.ndarray, el_range=(-90.0, 90.0), phase_deg: float = 0.0):
        self.panorama = np.asarray(panorama, dtype=np.float32)
        self.el_range = el_range
        self.phase_deg = phase_deg

    def luminance(self, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
        az = (np.asarray(azimuth_deg) - self.phase_deg + 180.0) % 360.0 - 180.0
        return sample_panorama(self.panorama, az, elevation_deg, self.el_range)

    def rotate(self, omega_deg_s: float, dt: float) -> None:
        self.phase_deg = (self.phase_deg + omega_deg_s * dt) % 360.0

    @classmethod
    def grating(cls, period_deg: float = 30.0, contrast: float = 1.0, mean_lum: float = 0.5,
                px_per_deg: float = 4.0) -> "Drum":
        """Vertical square-wave stripes (period rounded to divide 360)."""
        n_cycles = max(1, int(round(360.0 / period_deg)))
        w, h = int(360 * px_per_deg), int(180 * px_per_deg)
        az = (np.arange(w) + 0.5) / px_per_deg - 180.0
        row = mean_lum * (1 + contrast * square(2 * np.pi * n_cycles * az / 360.0))
        return cls(np.tile(row, (h, 1)))

    @classmethod
    def texture(cls, alpha: float = 2.0, contrast: float = 1.0, density: float = 1.0,
                mean_lum: float = 0.5, seed: int = 0, px_per_deg: float = 2.0) -> "Drum":
        """1/f^alpha texture wrapped around the drum (seamless in azimuth)."""
        w, h = int(360 * px_per_deg), int(180 * px_per_deg)
        return cls(texture_1f((h, w), alpha, contrast, density, mean_lum, seed=seed))
