"""Linear readout of self-rotation from HS/VS activity.

For each body axis (roll, pitch, yaw) the template is the matched-filter response
of the LPTC cells of both eyes to a world rotation about that axis
(flyloop.lptc.analysis), restricted to HS cells for yaw and VS cells for roll and
pitch, with unit norm. For yaw this is essentially HS left minus right.

    signal_a = template_a . (v - baseline)

`baseline` is the response to the static scene (set after settling), which
removes the tonic offset of the rate model. `calibrate_decoder` measures the
signal under known world rotations and rescales it so the decoder reports the
apparent world rotation rate in deg/s (positive = the world turns the right-hand
way about that body axis, i.e. the body turns the other way).
"""

from __future__ import annotations

import numpy as np

from flyloop.eyes import Eyes
from flyloop.lptc import LPTCWeights
from flyloop.lptc.analysis import matched_filter_response
from flyloop.sim.body import AXES

AXIS_VECTORS = {"roll": np.array([1.0, 0, 0]), "pitch": np.array([0, 1.0, 0]),
                "yaw": np.array([0, 0, 1.0])}
DEFAULT_GROUPS = {"roll": "VS", "pitch": "VS", "yaw": "HS"}


class RotationDecoder:
    """Templates (3 axes, 2 eyes, n_cells) applied to LPTC voltages (2, n_cells)."""

    def __init__(self, weights: LPTCWeights, eyes: Eyes, groups: dict | None = None,
                 g_exc: float = 1.0, g_inh: float = 1.0):
        groups = {**DEFAULT_GROUPS, **(groups or {})}
        self.cells = weights.cells
        t = []
        for axis in AXES:
            r = matched_filter_response(weights, eyes, AXIS_VECTORS[axis], g_exc, g_inh)
            mask = np.array([c.startswith(groups[axis]) for c in weights.cells], float)
            r = r * mask[None]
            t.append(r / np.linalg.norm(r))
        self.templates = np.stack(t)
        self.baseline = np.zeros(self.templates.shape[1:])
        self.scale = np.ones(len(AXES))  # signal units per deg/s, from calibration
        self.unmix = None  # optional (3, 3) inverse gain matrix, see calibrate_decoder

    def set_baseline(self, v: np.ndarray) -> None:
        self.baseline = np.asarray(v, float)

    def signal(self, v: np.ndarray) -> np.ndarray:
        """Raw template projections (3,) of LPTC voltages v (2, n_cells)."""
        return (self.templates * (np.asarray(v) - self.baseline)[None]).sum((1, 2))

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Apparent world rotation rate (3,) about roll, pitch, yaw, deg/s once
        calibrated."""
        if self.unmix is not None:
            return self.unmix @ self.signal(v)
        return self.signal(v) / self.scale


def calibrate_decoder(decoder: RotationDecoder, rig, make_world, omega_deg_s: float = 90.0,
                      t_move: float = 1.5, t_settle: float = 0.5,
                      unmix: bool = False) -> np.ndarray:
    """Open loop: rotate the world at +-omega about each body axis and set
    decoder.scale to the signal per deg/s.

    Args:
        rig: VisualRig (fly held fixed in the world's frame).
        make_world: Callable returning a fresh SphereWorld.
        unmix: Also set decoder.unmix to the inverse gain matrix, which removes
            linear cross-talk between axes (each estimate then mixes HS and VS).

    Returns:
        gain (3 signals, 3 rotation axes): signal per deg/s; off-diagonal terms
        are cross-talk between axes. The last half of each rotation is averaged.
    """
    n = int(round(t_move / rig.dt))
    gain = np.zeros((len(AXES), len(AXES)))
    for j, axis in enumerate(AXES):
        resp = []
        for sign in (+1, -1):
            world = make_world()
            settle = rig.reset(world, t_settle=t_settle)
            decoder.set_baseline(_settled(settle))
            sig = []
            for _ in range(n):
                world.rotate(AXIS_VECTORS[axis], sign * omega_deg_s, rig.dt)
                sig.append(decoder.signal(rig.step(world).lptc[:, -1]))
            resp.append(np.mean(sig[n // 2:], 0))
        gain[:, j] = (resp[0] - resp[1]) / (2 * omega_deg_s)
    decoder.scale = np.diag(gain).copy()
    decoder.unmix = np.linalg.inv(gain) if unmix else None
    return gain


def _settled(out, t_frames: int = 10) -> np.ndarray:
    """Mean LPTC voltage over the last frames of a settling output (2, n_cells)."""
    return out.lptc[:, -t_frames:].mean(1)
