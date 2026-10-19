"""The flyvis hexagonal photoreceptor lattice in visual-field coordinates.

Column order and placement follow flyvis exactly: hex coordinates come from
`flyvis.utils.hex_utils.get_hex_coords` (the order flyvis uses for its 721 input
columns), and each column (u, v) sits at

    x = v * spacing,   y = -(u + v / 2) * spacing      (degrees, x right, y up)

which is the placement used by `flyvis.datasets.rendering.HexEye` (monitor pixels
x_px = ppo * v, row_px = ppo * (u + v / 2), 5.8 deg per ommatidium) and `BoxEye`.
With this frame a motion direction theta (deg) means motion toward
(cos theta, sin theta), which is the `angle` convention of flyvis's `MovingEdge`
(0 = rightward, counter-clockwise positive).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from flyvis.utils.hex_utils import get_hex_coords, get_num_hexals

OMMATIDIUM_DEG = 5.8  # flyvis HexEye.omm_width_rad


@dataclass
class HexLattice:
    """Receptor positions of one flyvis eye.

    Args:
        extent: Hex radius in columns (15 gives the 721 columns flyvis expects).
        spacing_deg: Angular spacing between neighbouring columns.
        n_sub: Each ommatidium averages the scene over an n_sub x n_sub grid of
            points spanning +-spacing/2 in x and y (like HexEye's per-hexagon mean).
    """

    extent: int = 15
    spacing_deg: float = OMMATIDIUM_DEG
    n_sub: int = 5
    u: np.ndarray = field(init=False, repr=False)
    v: np.ndarray = field(init=False, repr=False)
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.u, self.v = get_hex_coords(self.extent)
        self.x = self.v * self.spacing_deg
        self.y = -(self.u + self.v / 2) * self.spacing_deg

    @property
    def n_hexals(self) -> int:
        return get_num_hexals(self.extent)

    @property
    def central_index(self) -> int:
        """Index of the (u, v) = (0, 0) column."""
        return int(np.flatnonzero((self.u == 0) & (self.v == 0))[0])

    def sample_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Sub-sample positions, shape (n_hexals, n_sub**2), in degrees."""
        if self.n_sub <= 1:
            return self.x[:, None], self.y[:, None]
        o = (np.arange(self.n_sub) + 0.5) / self.n_sub - 0.5
        ox, oy = np.meshgrid(o * self.spacing_deg, o * self.spacing_deg)
        return self.x[:, None] + ox.ravel()[None], self.y[:, None] + oy.ravel()[None]

    def to_sphere(
        self,
        x: np.ndarray,
        y: np.ndarray,
        eye_azimuth_deg: float = 0.0,
        eye_elevation_deg: float = 0.0,
        mirror: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map lattice positions to world azimuth/elevation (deg).

        The lattice is treated as an azimuthal-equidistant map around the eye's
        optical axis: a point at (x, y) lies at angular distance sqrt(x^2 + y^2)
        from the axis, in direction atan2(y, x) on the eye's image plane. Lattice x
        points toward increasing azimuth (counter-clockwise seen from above) and y
        toward increasing elevation. mirror=True flips x (use for the other eye).
        """
        x = -x if mirror else x
        rho = np.radians(np.hypot(x, y))
        psi = np.arctan2(y, x)
        # direction in the eye frame: optical axis along +X, image x -> +Y, y -> +Z
        d = np.stack(
            [np.cos(rho), np.sin(rho) * np.cos(psi), np.sin(rho) * np.sin(psi)], axis=-1
        )
        az0, el0 = np.radians(eye_azimuth_deg), np.radians(eye_elevation_deg)
        # rotate by elevation (about Y), then azimuth (about Z)
        ry = np.array(
            [[np.cos(el0), 0, -np.sin(el0)], [0, 1, 0], [np.sin(el0), 0, np.cos(el0)]]
        )
        rz = np.array(
            [[np.cos(az0), -np.sin(az0), 0], [np.sin(az0), np.cos(az0), 0], [0, 0, 1]]
        )
        w = d @ (rz @ ry).T
        az = np.degrees(np.arctan2(w[..., 1], w[..., 0]))
        el = np.degrees(np.arcsin(np.clip(w[..., 2], -1, 1)))
        return az, el
