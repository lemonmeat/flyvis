"""Perturbation hooks (noise, time-constant scaling, ablation) applied to a single pipeline."""

from .silencing import VISION_CONDITIONS, set_t4t5_output, t4t5_output

__all__ = ["VISION_CONDITIONS", "set_t4t5_output", "t4t5_output"]
