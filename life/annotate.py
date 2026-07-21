"""Draw detections and status banner onto frames for the dashboard."""
from __future__ import annotations

from typing import List, Optional

from .models import Detection, Event, Severity

# BGR colors
_COLORS = {
    "person": (0, 200, 0),
    "fall": (0, 0, 255),
    "fire": (0, 100, 255),
    "smoke": (150, 150, 150),
    "weapon": (0, 0, 255),
    "weapon_threat": (0, 0, 255),
}
_DEFAULT_COLOR = (0, 215, 255)


def draw(frame, detections: List[Detection], active_event: Optional[Event]):
    import cv2

    for d in detections:
        if d.bbox is None:
            continue
        color = _COLORS.get(d.signal or d.label, _DEFAULT_COLOR)
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d.label} {d.confidence:.0%}"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    _banner(cv2, frame, active_event)
    return frame


def _banner(cv2, frame, active_event: Optional[Event]) -> None:
    h, w = frame.shape[:2]
    if active_event is None:
        color = (0, 150, 0)
        text = "MONITORING - normal"
    else:
        color = (0, 0, 200) if active_event.severity >= Severity.HIGH else (0, 140, 220)
        text = f"! {active_event.event_type} -> {active_event.responder_name} ({active_event.severity})"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
