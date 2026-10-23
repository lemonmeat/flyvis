"""The closed loop: world -> eyes -> flyvis -> LPTC -> decoder -> controller -> body.

Each dt the eyes render the world at the body's current attitude, the visual rig
advances one frame, the decoder turns the newest HS/VS voltages into an apparent
rotation rate, the controller turns that into a steering command, and the body
integrates command plus disturbance. Perturbations act on the parts (for example
`flyloop.perturb.set_t4t5_output` on rig.lptc), never on this loop.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .body import AXES, Body
from .rig import VisualRig

Disturbance = Callable[[float], np.ndarray] | np.ndarray | None


class ClosedLoop:
    def __init__(self, rig: VisualRig, body: Body, decoder, controller):
        self.rig, self.body, self.decoder, self.controller = rig, body, decoder, controller
        hs = np.array([c.startswith("HS") for c in rig.cells])
        self._hs = hs if hs.any() else None

    @property
    def dt(self) -> float:
        return self.rig.dt

    def run(self, world, t_run: float, disturbance: Disturbance = None,
            world_motion: Callable | None = None, t_settle: float = 0.5) -> pd.DataFrame:
        """Run one trial from rest.

        Args:
            world: World with luminance(); moved only by world_motion.
            t_run: Seconds of closed loop after settling.
            disturbance: (3,) roll, pitch, yaw torque in deg/s units (see
                KinematicBody), or a function of time returning one.
            world_motion: Optional f(world, t, dt) called before each frame, for
                example a drum rotating under the closed loop.
            t_settle: Seconds of static scene, body held, before the loop closes.
                The decoder baseline is the LPTC response at its end.

        Returns:
            One row per step: time, body angular velocity, integrated angle,
            command, decoder estimate and disturbance per axis, the world heading,
            and HS left minus right (mean over HS cells).
        """
        dt = self.dt
        self.body.reset()
        self.controller.reset()
        settle = self.rig.reset(world, t_settle=t_settle, orientation=self.body.orientation)
        if settle is not None:
            self.decoder.set_baseline(settle.lptc[:, -10:].mean(1))
        rows = []
        for i in range(int(round(t_run / dt))):
            t = i * dt
            if world_motion is not None:
                world_motion(world, t, dt)
            v = self.rig.step(world, orientation=self.body.orientation).lptc[:, -1]
            estimate = self.decoder(v)
            command = self.controller(estimate, dt)
            d = disturbance(t) if callable(disturbance) else disturbance
            d = np.zeros(3) if d is None else np.asarray(d, float)
            self.body.step(command, dt, d)
            row = {"t": t + dt, "heading_deg": self.body.heading_deg}
            for k, a in enumerate(AXES):
                row |= {f"omega_{a}": self.body.omega[k], f"angle_{a}": self.body.angle_deg[k],
                        f"command_{a}": command[k], f"estimate_{a}": estimate[k],
                        f"disturbance_{a}": d[k]}
            if self._hs is not None:
                row["hs_left_minus_right"] = float(v[0, self._hs].mean() - v[1, self._hs].mean())
            rows.append(row)
        return pd.DataFrame(rows)
