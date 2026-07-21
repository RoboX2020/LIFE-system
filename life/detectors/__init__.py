"""Detector layer: each detector turns frames into Detection objects."""

from .base import Detector
from .fall import FallDetector
from .fire import FireDetector
from .weapon import WeaponDetector

__all__ = ["Detector", "FallDetector", "FireDetector", "WeaponDetector"]
