"""PD steering controller on visually estimated rotation.

The decoder reports r, the apparent world rotation rate about each body axis.
Integrating it gives e = -integral(r dt), the body's rotation away from the
visual world since reset (its heading error for yaw). The controller is PD on e:

    command = -kp * e - kd * de/dt = kp * integral(r dt) + kd * r

so the body turns with the world's apparent motion and back to where it started.
The D term acts directly on the HS/VS signal (an optomotor response); the P term
removes the steady drift a constant torque would otherwise leave. An optional
leak makes the integral forget with time constant leak_tau_s.
"""

from __future__ import annotations

import numpy as np


class PDController:
    """Per-axis PD on visual rotation error (roll, pitch, yaw).

    Args:
        kp: 1/s. Torque (as deg/s, see KinematicBody) per deg of estimated error.
        kd: Dimensionless. Torque (deg/s) per deg/s of estimated rotation rate.
        leak_tau_s: Forgetting time of the error integral (None = perfect).
    """

    def __init__(self, kp=5.0, kd=2.0, leak_tau_s: float | None = None):
        self.kp = np.broadcast_to(np.asarray(kp, float), (3,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, float), (3,)).copy()
        self.leak_tau_s = leak_tau_s
        self.reset()

    def reset(self) -> None:
        self.error_deg = np.zeros(3)

    def __call__(self, rate_deg_s: np.ndarray, dt: float) -> np.ndarray:
        rate = np.asarray(rate_deg_s, float)
        if self.leak_tau_s:
            self.error_deg = self.error_deg * np.exp(-dt / self.leak_tau_s)
        self.error_deg = self.error_deg - rate * dt
        return -self.kp * self.error_deg + self.kd * rate
