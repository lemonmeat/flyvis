"""Directions and rotations in the body/world frame: x forward, y left, z up.

Azimuth is measured counter-clockwise from +x seen from above (0 = ahead,
positive to the left), elevation positive up. Rotations are right-handed.
"""

from __future__ import annotations

import numpy as np


def directions(azimuth_deg, elevation_deg) -> np.ndarray:
    """Unit vectors (..., 3) for the given azimuth and elevation."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def angles(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth and elevation (deg) of vectors (..., 3)."""
    az = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
    el = np.degrees(np.arcsin(np.clip(d[..., 2], -1.0, 1.0)))
    return az, el


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    """Right-handed rotation by angle_deg about axis (Rodrigues)."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = np.radians(angle_deg)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
