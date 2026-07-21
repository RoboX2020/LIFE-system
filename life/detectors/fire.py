"""Fire / smoke detection.

Primary path: a custom YOLO model with fire/smoke classes (models/fire.pt).
Fallback: an HSV color mask for fire-like regions combined with a flicker
(frame-difference) check, so the system still demos without custom weights.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from ..models import Detection
from .base import Detector
from .yolo_engine import YoloEngine

log = logging.getLogger("life.fire")

_FIRE_LABELS = {"fire", "flame", "flames"}
_SMOKE_LABELS = {"smoke"}


class FireDetector(Detector):
    name = "fire"

    def __init__(self, config, enabled: bool = True) -> None:
        super().__init__(config, enabled)
        self.conf = float(config.get("conf", 0.40))
        heur = config.get("heuristic", {}) or {}
        self.heuristic_enabled = bool(heur.get("enabled", True))
        self.min_area_ratio = float(heur.get("min_area_ratio", 0.0015))
        self.flicker_frames = int(heur.get("flicker_frames", 4))

        self._model: Optional[YoloEngine] = None
        model_path = config.get("model")
        if model_path and os.path.exists(model_path):
            engine = YoloEngine(model_path=model_path, conf=self.conf)
            if engine.available:
                self._model = engine
                log.info("Fire/smoke YOLO model loaded from %s", model_path)

        self._prev_gray = None
        self._fire_area_hist: deque = deque(maxlen=32)

    def process(self, frame, timestamp: float, shared: Dict[str, Any]) -> List[Detection]:
        if not self.enabled:
            return []
        if self._model is not None:
            return self._detect_model(frame)
        if self.heuristic_enabled:
            return self._detect_heuristic(frame)
        return []

    def _detect_model(self, frame) -> List[Detection]:
        dets = self._model.detect(frame, conf=self.conf, detector_name=self.name)
        out: List[Detection] = []
        for d in dets:
            low = d.label.lower()
            if low in _FIRE_LABELS:
                d.signal = "fire"
                out.append(d)
            elif low in _SMOKE_LABELS:
                d.signal = "smoke"
                out.append(d)
        return out

    def _detect_heuristic(self, frame) -> List[Detection]:
        import cv2

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Fire-like: warm hues (red/orange/yellow), high saturation & value.
        lower1 = np.array([0, 100, 150], dtype=np.uint8)
        upper1 = np.array([35, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 100, 150], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2),
        )
        mask = cv2.medianBlur(mask, 5)

        area_ratio = float(np.count_nonzero(mask)) / float(h * w)
        self._fire_area_hist.append(area_ratio)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flicker = 0.0
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            flicker = float(np.mean(diff[mask > 0])) if np.count_nonzero(mask) else 0.0
        self._prev_gray = gray

        enough_area = area_ratio >= self.min_area_ratio
        recent_flicker = sum(
            1 for a in list(self._fire_area_hist)[-self.flicker_frames:]
            if a >= self.min_area_ratio
        )
        confirmed = enough_area and recent_flicker >= self.flicker_frames and flicker > 2.0

        if not confirmed:
            return []

        bbox = self._mask_bbox(mask)
        confidence = float(min(0.95, 0.45 + area_ratio * 40 + flicker / 100.0))
        return [
            Detection(
                label="fire",
                confidence=confidence,
                bbox=bbox,
                detector=self.name,
                signal="fire",
                meta={
                    "area_ratio": round(area_ratio, 4),
                    "flicker": round(flicker, 2),
                    "method": "heuristic",
                },
            )
        ]

    @staticmethod
    def _mask_bbox(mask) -> Optional[tuple]:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        return (x, y, x + bw, y + bh)
