"""Lobula plate tangential cells (HS, VS) driven by T4/T5, with LPi opponent
inhibition; weights from the optic-lobe:v1.1 connectome or a simple pooled model."""

from .model import LPTCLayer, LPTCParams
from .weights import LPTCWeights

__all__ = ["LPTCLayer", "LPTCParams", "LPTCWeights"]
