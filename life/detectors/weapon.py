"""Weapon detection + threat reasoning.

Weapons come from (a) the shared YOLO pass (COCO includes 'knife') and/or
(b) an optional custom model (models/weapon.pt) with gun/rifle classes.

Spatial reasoning escalates a bare 'weapon' to a 'weapon_threat' when the weapon
is close to a detected person.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional

from ..models import BBox, Detection
from .base import Detector
from .yolo_engine import YoloEngine

log = logging.getLogger("life.weapon")


class WeaponDetector(Detector):
    name = "weapon"

    def __init__(self, config, enabled: bool = True) -> None:
        super().__init__(config, enabled)
        self.conf = float(config.get("conf", 0.40))
        self.labels = {str(l).lower() for l in config.get(
            "labels", ["knife", "gun", "pistol", "rifle", "firearm", "weapon"]
        )}
        self.proximity_ratio = float(config.get("proximity_ratio", 1.6))

        self._model: Optional[YoloEngine] = None
        model_path = config.get("model")
        if model_path and os.path.exists(model_path):
            engine = YoloEngine(model_path=model_path, conf=self.conf)
            if engine.available:
                self._model = engine
                log.info("Weapon YOLO model loaded from %s", model_path)

    def process(self, frame, timestamp: float, shared: Dict[str, Any]) -> List[Detection]:
        if not self.enabled:
            return []

        weapons: List[Detection] = []

        # (a) weapons that fell out of the shared YOLO pass (e.g. COCO 'knife')
        for d in shared.get("yolo", []):
            if d.label.lower() in self.labels and d.confidence >= self.conf:
                weapons.append(d)

        # (b) custom weapon model
        if self._model is not None:
            for d in self._model.detect(frame, conf=self.conf, detector_name=self.name):
                if d.label.lower() in self.labels or True:  # custom model = all weapons
                    weapons.append(d)

        if not weapons:
            return []

        persons = shared.get("persons", [])
        out: List[Detection] = []
        for wd in weapons:
            threatened = self._near_person(wd.bbox, persons)
            if threatened:
                out.append(
                    Detection(
                        label="weapon_threat",
                        confidence=wd.confidence,
                        bbox=wd.bbox,
                        detector=self.name,
                        signal="weapon_threat",
                        meta={"weapon": wd.label, "near_person": True},
                    )
                )
            else:
                out.append(
                    Detection(
                        label="weapon",
                        confidence=wd.confidence,
                        bbox=wd.bbox,
                        detector=self.name,
                        signal="weapon",
                        meta={"weapon": wd.label, "near_person": False},
                    )
                )
        return out

    def _near_person(self, weapon_bbox: Optional[BBox], persons: List[Detection]) -> bool:
        if weapon_bbox is None or not persons:
            return False
        wcx, wcy = _center(weapon_bbox)
        for p in persons:
            if p.bbox is None:
                continue
            if _boxes_overlap(weapon_bbox, p.bbox):
                return True
            pcx, pcy = _center(p.bbox)
            person_size = math.hypot(p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1])
            dist = math.hypot(wcx - pcx, wcy - pcy)
            if dist <= self.proximity_ratio * person_size:
                return True
        return False


def _center(box: BBox):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
