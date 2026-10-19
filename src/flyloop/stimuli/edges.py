"""Moving ON and OFF edges."""

from __future__ import annotations

import numpy as np
import torch

from .base import StimulusSequence, render_on_lattice, to_tensor
from .lattice import OMMATIDIUM_DEG, HexLattice

# flyvis tutorial 04 uses speed 19 ommatidia/s (~110 deg/s)
DEFAULT_SPEED_DEG_S = 19 * OMMATIDIUM_DEG


def moving_edge(
    direction_deg: float,
    polarity: str = "on",
    speed_deg_s: float = DEFAULT_SPEED_DEG_S,
    contrast: float = 1.0,
    mean_lum: float = 0.5,
    start_deg: float = -100.0,
    end_deg: float = 100.0,
    height_deg: float | None = None,
    t_pre: float = 1.0,
    t_post: float = 1.0,
    dt: float = 1 / 200,
    lattice: HexLattice | None = None,
    device: str | torch.device | None = None,
) -> StimulusSequence:
    """A luminance step sweeping across the lattice toward `direction_deg`.

    The edge position along s = x cos(theta) + y sin(theta) moves from start_deg
    to end_deg at speed_deg_s. Behind the edge (s < position) luminance is
    mean_lum * (1 + contrast) for ON edges or mean_lum * (1 - contrast) for OFF
    edges; ahead of it the screen stays at mean_lum. The default sweep covers the
    whole 721-column lattice, unlike flyvis's MovingEdge (+-22.5 deg around the
    centre). After the sweep the last frame is held for t_post, matching
    MovingEdge(post_pad_mode="continue"). Optionally the edge is confined to a
    strip of height_deg orthogonal to the motion.
    """
    if polarity not in ("on", "off"):
        raise ValueError(f"polarity must be 'on' or 'off', got {polarity!r}")
    lattice = lattice or HexLattice()
    th = np.radians(direction_deg)
    sign = 1.0 if polarity == "on" else -1.0
    t_sweep = (end_deg - start_deg) / speed_deg_s
    times = np.arange(int(round((t_pre + t_sweep + t_post) / dt))) * dt - t_pre

    def lum(x, y, t):
        s = x * np.cos(th) + y * np.sin(th)
        pos = start_deg + speed_deg_s * np.clip(t, 0.0, t_sweep)
        behind = s < pos
        if height_deg is not None:
            behind &= np.abs(-x * np.sin(th) + y * np.cos(th)) <= height_deg / 2
        return np.where(behind, mean_lum * (1 + sign * contrast), mean_lum)

    frames = render_on_lattice(lum, times, lattice)
    params = dict(
        kind="moving_edge",
        direction_deg=direction_deg,
        polarity=polarity,
        speed_deg_s=speed_deg_s,
        contrast=contrast,
        mean_lum=mean_lum,
        start_deg=start_deg,
        end_deg=end_deg,
        height_deg=height_deg,
        t_sweep=t_sweep,
    )
    return StimulusSequence(to_tensor(frames, device), dt, t_pre, params)
