"""Thin wrapper around Ultralytics YOLO with graceful degradation.

If ultralytics / weights are unavailable, the engine reports itself unavailable
and returns no detections, so the rest of the pipeline still runs (useful for the
demo fire source and for CI without model downloads).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..models import Detection

log = logging.getLogger("life.yolo")

# Preference order; the first that loads wins. Newer weights first.
_MODEL_FALLBACKS = ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]


class YoloEngine:
    def __init__(self, model_path: str = "yolo26n.pt", conf: float = 0.35,
                 device: str = "cpu") -> None:
        self.conf = conf
        self.device = device
        self.available = False
        self.model = None
        self.names = {}
        self._load(model_path)

    def _load(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - env dependent
            log.warning("ultralytics not available (%s); object detection disabled.", exc)
            return

        candidates = [model_path] + [m for m in _MODEL_FALLBACKS if m != model_path]
        for cand in candidates:
            try:
                self.model = YOLO(cand)
                self.names = self.model.names
                self.available = True
                log.info("Loaded YOLO weights: %s", cand)
                return
            except Exception as exc:  # try next fallback
                log.warning("Could not load YOLO weights '%s' (%s).", cand, exc)
        log.error("No YOLO weights could be loaded; object detection disabled.")

    def detect(self, frame, conf: Optional[float] = None,
               detector_name: str = "yolo") -> List[Detection]:
        if not self.available or self.model is None:
            return []
        conf = self.conf if conf is None else conf
        results = self.model.predict(
            frame, conf=conf, device=self.device, verbose=False
        )
        out: List[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            names = res.names if getattr(res, "names", None) else self.names
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                out.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        detector=detector_name,
                    )
                )
        return out
