"""Core data structures shared across the pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


class Severity(IntEnum):
    """Ordered severity levels (higher = worse). IntEnum so we can compare."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, value: Any) -> "Severity":
        if isinstance(value, Severity):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).strip().upper()]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


@dataclass
class Detection:
    """A single thing a detector found in a frame."""

    label: str                      # e.g. "person", "fire", "fall", "weapon_threat"
    confidence: float
    bbox: Optional[BBox] = None
    detector: str = ""              # which detector produced this
    signal: Optional[str] = None    # fusion signal key (defaults to label)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signal is None:
            self.signal = self.label


@dataclass
class EventCandidate:
    """A potential incident emitted by the fusion engine (pre-cooldown)."""

    event_type: str
    severity: Severity
    responder: str
    confidence: float
    rule: str
    message: str = ""
    detections: List[Detection] = field(default_factory=list)


@dataclass
class Event:
    """A confirmed, dispatched incident."""

    event_type: str
    severity: Severity
    responder: str
    responder_name: str
    responder_contact: str
    confidence: float
    message: str
    timestamp: float = field(default_factory=time.time)
    snapshot_path: Optional[str] = None
    detections: List[Detection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "severity": str(self.severity),
            "severity_level": int(self.severity),
            "responder": self.responder,
            "responder_name": self.responder_name,
            "responder_contact": self.responder_contact,
            "confidence": round(float(self.confidence), 3),
            "message": self.message,
            "timestamp": self.timestamp,
            "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "snapshot_path": self.snapshot_path,
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(float(d.confidence), 3),
                    "bbox": d.bbox,
                    "detector": d.detector,
                }
                for d in self.detections
            ],
        }
