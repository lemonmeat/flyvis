"""Visual stimuli: drifting gratings, moving edges, rotating gratings and procedural
textures, rendered onto the flyvis hexagonal lattice."""

from .base import StimulusSequence
from .edges import moving_edge
from .gratings import drifting_grating, rotating_grating
from .lattice import OMMATIDIUM_DEG, HexLattice
from .textures import frames_to_hex, rotating_texture, sample_panorama, texture_1f

__all__ = [
    "OMMATIDIUM_DEG",
    "HexLattice",
    "StimulusSequence",
    "drifting_grating",
    "frames_to_hex",
    "moving_edge",
    "rotating_grating",
    "rotating_texture",
    "sample_panorama",
    "texture_1f",
]
