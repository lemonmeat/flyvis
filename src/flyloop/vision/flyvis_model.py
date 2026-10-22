"""Thin wrapper around a pretrained flyvis network for T4/T5 readout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import torch

import flyloop  # noqa: F401  (sets FLYVIS_ROOT_DIR before flyvis is imported)

T4T5 = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def set_flyvis_device(device: str | torch.device) -> torch.device:
    """Point flyvis (and torch's default device) at `device`.

    flyvis picks cuda or cpu at import and allocates its stimulus buffers on
    torch's default device, so this is process-global. Call it before
    constructing a FlyvisModel.
    """
    import flyvis

    device = torch.device(device)
    torch.set_default_device(device)
    flyvis.device = device
    return device


@dataclass
class Responses:
    """Readout of selected cell types.

    Attributes:
        time: (n_frames,) seconds relative to stimulus motion onset.
        cell_types: Names along the cell-type axis.
        central: (batch, n_frames, n_types) response of each type's central column.
        mean: (batch, n_frames, n_types) mean over all columns of each type.
        activity: Optional (batch, n_frames, n_nodes) full network activity.
    """

    time: np.ndarray
    cell_types: list[str]
    central: np.ndarray
    mean: np.ndarray
    activity: np.ndarray | None = field(default=None, repr=False)

    def __getitem__(self, cell_type: str) -> dict[str, np.ndarray]:
        i = self.cell_types.index(cell_type)
        return {"central": self.central[..., i], "mean": self.mean[..., i]}


class FlyvisModel:
    """Load a pretrained flyvis network and read out a list of cell types.

    Args:
        model: Path of the pretrained network relative to flyvis.results_dir.
            "flow/0000/000" is the best-ranked model of the 50-model ensemble.
        cell_types: Cell types to read out (default T4a-d, T5a-d).
        dt: Integration time step in seconds; stimuli must be sampled at dt.
        initial_state: "steady" (grey at value 0.5 for t_init seconds, as flyvis's
            moving-edge analysis does), "fade_in" (ramp the contrast of the first
            frame over t_init), or "none" (flyvis's default initial state).
        t_init: Duration used to compute the initial state.
        checkpoint: flyvis checkpoint selector ("best" or an index).
    """

    def __init__(
        self,
        model: str = "flow/0000/000",
        cell_types: Sequence[str] = T4T5,
        dt: float = 1 / 200,
        initial_state: Literal["steady", "fade_in", "none"] = "steady",
        t_init: float = 1.0,
        checkpoint: str | int = "best",
    ):
        from flyvis import NetworkView

        self.model = model
        self.dt = dt
        self.initial_state = initial_state
        self.t_init = t_init
        results_dir = flyloop.FLYVIS_ROOT_DIR / "results"
        self.view = NetworkView(results_dir / model, root_dir=results_dir)
        self.network = self.view.init_network(checkpoint=checkpoint)
        self.network.eval()
        self.device = next(self.network.parameters()).device
        self.set_cell_types(cell_types)

    # --- readout indices ---------------------------------------------------------

    def set_cell_types(self, cell_types: Sequence[str]) -> None:
        connectome = self.network.connectome
        unique = list(connectome.unique_cell_types[:].astype(str))
        missing = [t for t in cell_types if t not in unique]
        if missing:
            raise ValueError(f"cell types not in the flyvis connectome: {missing}")
        central_all = connectome.central_cells_index[:]
        self.cell_types = list(cell_types)
        self.central_index = np.array([central_all[unique.index(t)] for t in cell_types])
        self.layer_index = [connectome.nodes.layer_index[t][:] for t in cell_types]
        # torch copies for the per-frame readout in step(); one stacked (n_types,
        # n_cells) gather when all types have the same number of cells (T4/T5: 721)
        self._central_t = torch.as_tensor(self.central_index, device=self.device)
        sizes = {len(idx) for idx in self.layer_index}
        self._layer_t = (torch.as_tensor(np.stack(self.layer_index), device=self.device)
                         if len(sizes) == 1 else None)

    @property
    def n_nodes(self) -> int:
        return int(self.network.n_nodes)

    # --- simulation --------------------------------------------------------------

    def _as_movie(self, stimulus) -> tuple[torch.Tensor, float]:
        frames = getattr(stimulus, "frames", stimulus)
        t_pre = float(getattr(stimulus, "t_pre", 0.0))
        stim_dt = getattr(stimulus, "dt", None)
        if stim_dt is not None and not np.isclose(stim_dt, self.dt):
            raise ValueError(f"stimulus dt={stim_dt} differs from model dt={self.dt}")
        frames = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4:
            raise ValueError("frames must be (n_frames, 1, hexals) or (batch, n_frames, 1, hexals)")
        return frames, t_pre

    @torch.no_grad()
    def init_state(self, batch_size: int = 1, first_frame: torch.Tensor | None = None):
        """Initial network state according to `initial_state`.

        first_frame: (batch, 1, hexals), needed for "fade_in".
        """
        if self.initial_state == "steady":
            return self.network.steady_state(self.t_init, self.dt, batch_size, value=0.5)
        if self.initial_state == "fade_in":
            if first_frame is None:
                raise ValueError("fade_in needs the first frame")
            return self.network.fade_in_state(self.t_init, self.dt, first_frame)
        return None

    @torch.no_grad()
    def simulate(
        self, stimulus, return_activity: bool = False, chunk_frames: int = 100
    ) -> Responses:
        """Run a stimulus through the network.

        Args:
            stimulus: A `flyloop.stimuli.StimulusSequence` or a tensor of frames
                (n_frames, 1, 721) or (batch, n_frames, 1, 721) sampled at dt.
            return_activity: Also return the full (batch, n_frames, n_nodes)
                activity (large; for debugging or saving raw data on request).
            chunk_frames: Without return_activity, integrate in chunks of this
                many frames and keep only the readout, so memory stays bounded.
        """
        movie, t_pre = self._as_movie(stimulus)
        state = self.init_state(movie.shape[0], movie[:, 0])
        time = np.arange(movie.shape[1]) * self.dt - t_pre
        if return_activity:
            activity = self.network.simulate(movie, self.dt, initial_state=state)
            central = activity[..., self.central_index]
            mean = self._layer_means(activity)
            return Responses(time, self.cell_types, central.cpu().numpy(),
                             mean.cpu().numpy(), activity.cpu().numpy())
        centrals, means = [], []
        for k in range(0, movie.shape[1], chunk_frames):
            c, m, state = self.step(movie[:, k : k + chunk_frames], state)
            centrals.append(c.cpu())
            means.append(m.cpu())
        return Responses(time, self.cell_types, torch.cat(centrals, 1).numpy(),
                         torch.cat(means, 1).numpy())

    def _layer_means(self, activity: torch.Tensor) -> torch.Tensor:
        if self._layer_t is not None:
            return activity[..., self._layer_t].mean(-1)
        return torch.stack([activity[..., idx].mean(-1) for idx in self.layer_index], -1)

    # --- stepping (closed loop) ----------------------------------------------------

    def invalidate_params(self) -> None:
        """Drop the cached parameters; call after changing network parameters
        (e.g. a perturbation that scales time constants)."""
        self._param_cache = None

    def _params(self):
        # flyvis's Network.forward clamps and rebuilds this namespace (a gather over
        # all 1.5M edges) on every call. Parameters are fixed in eval mode, so build
        # it once. Uses flyvis-internal methods; pinned to flyvis 1.2.0.
        if getattr(self, "_param_cache", None) is None:
            self.network.clamp()
            self._param_cache = self.network._param_api()
        return self._param_cache

    @torch.no_grad()
    def step(self, frames: torch.Tensor, state):
        """Advance the network by the given frames, keeping state between calls.

        Args:
            frames: (batch, n_frames, 1, hexals) photoreceptor input for the next
                n_frames * dt seconds.
            state: State from `init_state` or a previous `step` (None starts from
                flyvis's default initial state).

        Returns:
            (central, mean, state): central and population-mean responses of the
            readout cell types as tensors (batch, n_frames, n_types) on the model
            device, and the new state.
        """
        frames = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        params = self._params()
        stim = self.network.stimulus
        stim.zero(frames.shape[0], frames.shape[1])
        stim.add_input(frames)
        x = stim()
        if state is None:
            state = self.network._initial_state(params, frames.shape[0])
        central, mean = [], []
        for i in range(x.shape[1]):
            state = self.network._next_state(params, state, x[:, i], self.dt)
            act = state.nodes.activity
            central.append(act[:, self._central_t])
            mean.append(self._layer_means(act))
        return torch.stack(central, 1), torch.stack(mean, 1), state
