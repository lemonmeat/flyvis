"""Rate model of the LPTC layer driven by T4/T5 population activity.

For each eye (batch entry), with r(x) = relu(x) as in flyvis synapses:

    tau_lpi  dL/dt = -L + g_lpi * sum_{s,h} Win[k,s,h]  r(T[s,h])
    tau_lptc dV/dt = -V + g_exc * sum_{s,h} Wexc[j,s,h] r(T[s,h])
                        + g_inh * sum_k  sign_k Wout[j,k] r(L_k)

T[s, h] is the voltage of T4/T5 subtype s in column h from flyvis. Weights are
synapse counts normalised per target cell (each row sums to 1), so the gains are
dimensionless and V is in units of rectified T4/T5 activity. Inhibition is
subtractive, as in flyvis. V includes a tonic offset from T4/T5 resting activity;
compare cells (left vs right, or against a grey baseline) rather than reading V
in absolute terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .weights import LPTCWeights


def _row_normalize(w: np.ndarray) -> np.ndarray:
    flat = w.reshape(w.shape[0], -1)
    s = flat.sum(1, keepdims=True)
    return (flat / np.where(s > 0, s, 1)).reshape(w.shape)


@dataclass
class LPTCParams:
    tau_lptc: float = 0.01  # s, HS/VS membrane time constant
    tau_lpi: float = 0.01   # s
    g_exc: float = 1.0
    g_inh: float = 1.0
    g_lpi: float = 1.0


class LPTCLayer:
    """HS/VS (and LPi) dynamics for a batch of eyes.

    Args:
        weights: LPTCWeights (connectome or simple).
        dt: Integration step in seconds (match the flyvis model's dt).
        params: Time constants and gains.
        device: Torch device (default: torch's default device).
    """

    def __init__(self, weights: LPTCWeights, dt: float, params: LPTCParams | None = None,
                 device=None):
        self.weights = weights
        self.dt = dt
        self.params = params or LPTCParams()
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.w_exc = t(_row_normalize(weights.exc)).flatten(1)        # (cells, 8*h)
        self.w_lpi_in = t(_row_normalize(weights.lpi_in)).flatten(1)  # (lpi, 8*h)
        self.w_lpi_out = t(_row_normalize(weights.lpi_out) * weights.lpi_sign[None])
        # ablation masks over the 8 T4/T5 subtypes (1 = intact)
        self.subtype_gain = torch.ones(8, device=self.w_exc.device)

    @property
    def cells(self) -> list[str]:
        return self.weights.cells

    def init_state(self, batch: int) -> dict:
        z = lambda n: torch.zeros(batch, n, device=self.w_exc.device)  # noqa: E731
        return {"v": z(self.weights.n_cells), "lpi": z(self.weights.n_lpi)}

    @torch.no_grad()
    def step(self, t4t5: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        """Integrate over the frames of a T4/T5 activity chunk.

        Args:
            t4t5: (batch, n_frames, 8, n_hexals) T4/T5 voltages (FlyvisModel
                columns readout, cell types in T4T5 order).
            state: From init_state or a previous step.

        Returns:
            (v, state): v is (batch, n_frames, n_cells) HS/VS voltages.
        """
        p = self.params
        a_lpi = min(self.dt / p.tau_lpi, 1.0)
        a_v = min(self.dt / p.tau_lptc, 1.0)
        v, lpi = state["v"], state["lpi"]
        out = []
        for i in range(t4t5.shape[1]):
            r = torch.relu(t4t5[:, i]) * self.subtype_gain[None, :, None]
            r = r.flatten(1)
            lpi = lpi + a_lpi * (-lpi + p.g_lpi * r @ self.w_lpi_in.T)
            drive = p.g_exc * r @ self.w_exc.T + p.g_inh * torch.relu(lpi) @ self.w_lpi_out.T
            v = v + a_v * (-v + drive)
            out.append(v)
        return torch.stack(out, 1), {"v": v, "lpi": lpi}
