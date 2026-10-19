"""Direction selectivity from responses at a set of motion directions.

Definitions follow flyvis (flyvis.analysis.moving_bar_responses):
- responses are rectified peak responses r(theta) >= 0,
- DSI = |sum_theta r(theta) e^{i theta}| / sum_theta r(theta), in [0, 1],
- preferred direction = angle of that vector sum.
flyvis normalises the denominator by the larger of the ON and OFF sums
(pass it as `norm`), which makes the DSI of the weaker polarity small.
"""

from __future__ import annotations

import numpy as np


def _vector_sum(responses: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    responses = np.asarray(responses, dtype=float)
    angles = np.radians(np.asarray(angles_deg, dtype=float))
    if responses.shape[-1] != angles.shape[-1]:
        raise ValueError("last axis of responses must match angles")
    return (responses * np.exp(1j * angles)).sum(-1)


def direction_selectivity_index(
    responses: np.ndarray, angles_deg: np.ndarray, norm: np.ndarray | float | None = None
) -> np.ndarray:
    """Vector-sum DSI over the last axis.

    Args:
        responses: Non-negative responses, shape (..., n_angles).
        angles_deg: Motion directions in degrees, shape (n_angles,).
        norm: Denominator to use instead of sum(responses), e.g. the max over
            ON and OFF stimuli as in flyvis. Broadcasts against responses[..., 0].

    Returns:
        DSI with shape responses.shape[:-1]; 0 where the denominator is 0.
    """
    responses = np.asarray(responses, dtype=float)
    if np.any(responses < 0):
        raise ValueError("responses must be non-negative (rectify first)")
    denom = responses.sum(-1) if norm is None else np.asarray(norm, dtype=float)
    length = np.abs(_vector_sum(responses, angles_deg))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, length / np.where(denom > 0, denom, 1), 0.0)


def preferred_direction(responses: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Angle of the response vector sum in degrees, in [0, 360). NaN if all zero."""
    v = _vector_sum(responses, angles_deg)
    pd = np.degrees(np.angle(v)) % 360
    return np.where(np.abs(v) > 1e-12, pd, np.nan)


def angular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed difference a - b in degrees, wrapped to (-180, 180]."""
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360
    return np.where(d > 180, d - 360, d)


def peak_response(
    trace: np.ndarray,
    time: np.ndarray,
    t_start: float | None = None,
    t_end: float | None = None,
    baseline: float | np.ndarray | str | None = None,
    rectify: bool = True,
) -> np.ndarray:
    """Peak of a response trace inside a time window.

    Args:
        trace: Responses, shape (..., n_frames).
        time: Time of each frame in seconds (n_frames,).
        t_start, t_end: Window (inclusive); None means the start/end of the trace.
        baseline: Subtracted before rectifying. "pre" uses the value at the last
            frame before t = 0 (the steady state before motion onset). None
            subtracts nothing (flyvis uses raw voltages).
        rectify: Clip at zero before taking the maximum (as flyvis does).

    Returns:
        Peak with shape trace.shape[:-1].
    """
    trace = np.asarray(trace, dtype=float)
    time = np.asarray(time, dtype=float)
    if isinstance(baseline, str):
        if baseline != "pre":
            raise ValueError(f"unknown baseline {baseline!r}")
        pre = np.flatnonzero(time < 0)
        if len(pre) == 0:
            raise ValueError("baseline='pre' needs frames before t=0")
        baseline = trace[..., pre[-1]]
    if baseline is not None:
        trace = trace - np.asarray(baseline, dtype=float)[..., None]
    lo = -np.inf if t_start is None else t_start
    hi = np.inf if t_end is None else t_end
    mask = (time >= lo) & (time <= hi)
    if not mask.any():
        raise ValueError("empty time window")
    windowed = trace[..., mask]
    if rectify:
        windowed = np.clip(windowed, 0, None)
    return windowed.max(-1)
