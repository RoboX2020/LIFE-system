"""Abstract detector interface.

Adding a new hazard type = subclass Detector, implement process(), and add a
fusion rule in config.yaml. Nothing else in the core needs to change.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..models import Detection


class Detector:
    name: str = "detector"

    def __init__(self, config, enabled: bool = True) -> None:
        self.config = config
        self.enabled = enabled

    def process(self, frame, timestamp: float, shared: Dict[str, Any]) -> List[Detection]:
        """Return detections for this frame.

        `shared` carries cross-detector context, e.g. shared["persons"] holds
        person detections from the shared YOLO pass.
        """
        raise NotImplementedError

    def set_enabled(self, value: bool) -> None:
        self.enabled = bool(value)
