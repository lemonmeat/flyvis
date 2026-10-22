"""Compound-eye sampling of rendered scenes onto the flyvis hexagonal lattice
(721 columns per eye), for two mirror-symmetric eyes."""

from .compound_eye import EYE_ORDER, CompoundEye, Eyes

__all__ = ["EYE_ORDER", "CompoundEye", "Eyes"]
