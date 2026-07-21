"""Input adapters shared by the external-only demo and future simulators."""

from .carla_source import CarlaSensorSource, carla_image_to_bgr
from .external_source import ExternalFrameSource, FramePacket, OpenCVFrameSource

__all__ = [
    "CarlaSensorSource",
    "ExternalFrameSource",
    "FramePacket",
    "OpenCVFrameSource",
    "carla_image_to_bgr",
]
