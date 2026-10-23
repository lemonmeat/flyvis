"""Silencing T4/T5 output onto the lobula plate.

Leonte et al. (2021) blocked T4/T5 synaptic output; here the matching hook scales
the T4/T5 -> LPTC/LPi drive per subtype (`LPTCLayer.subtype_gain`). flyvis itself
runs unchanged, so photoreceptors and T4/T5 still respond, they just reach nothing.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from flyloop.lptc import LPTCLayer
from flyloop.vision import T4T5

# named T4/T5 output conditions: subtypes left intact
VISION_CONDITIONS = {
    "intact": T4T5,
    "T4_only": tuple(t for t in T4T5 if t.startswith("T4")),
    "T5_only": tuple(t for t in T4T5 if t.startswith("T5")),
    "silenced": (),
}


def set_t4t5_output(lptc: LPTCLayer, intact=T4T5) -> None:
    """Pass the output of the `intact` subtypes (names or a VISION_CONDITIONS key)
    and block all other T4/T5 subtypes."""
    if isinstance(intact, str):
        intact = VISION_CONDITIONS[intact]
    unknown = set(intact) - set(T4T5)
    if unknown:
        raise ValueError(f"unknown T4/T5 subtypes {sorted(unknown)}")
    gain = np.array([t in intact for t in T4T5], np.float32)
    lptc.subtype_gain[:] = lptc.subtype_gain.new_tensor(gain)


@contextmanager
def t4t5_output(lptc: LPTCLayer, intact=T4T5):
    """Context manager form of set_t4t5_output; restores the previous gains."""
    saved = lptc.subtype_gain.clone()
    set_t4t5_output(lptc, intact)
    try:
        yield lptc
    finally:
        lptc.subtype_gain[:] = saved
