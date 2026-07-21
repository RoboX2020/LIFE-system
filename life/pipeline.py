"""Detection pipeline: capture -> detectors -> fusion -> events, plus shared state.

Runs in a background thread. Exposes the latest annotated JPEG and a rolling
event feed for the FastAPI server to serve.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from .alarm import AlarmManager
from .annotate import draw
from .capture import FrameSource
from .detectors import FallDetector, FireDetector, WeaponDetector
from .detectors.yolo_engine import YoloEngine
from .events import EventManager
from .fusion import FusionEngine
from .models import Detection, Event
from .notify import NotificationDispatcher

log = logging.getLogger("life.pipeline")


class SharedState:
    """Thread-safe hand-off between the pipeline thread and the web server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self.status: Dict[str, Any] = {
            "running": False,
            "fps": 0.0,
            "alarm_active": False,
            "active_event": None,
            "detectors": {},
        }
        self.events: deque = deque(maxlen=100)
        self._event_seq = 0

    def set_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def push_event(self, event: Event) -> None:
        with self._lock:
            self._event_seq += 1
            record = event.to_dict()
            record["id"] = self._event_seq
            self.events.appendleft(record)

    def snapshot_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.status)

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.events)[:limit]


class Pipeline:
    def __init__(self, config) -> None:
        self.config = config
        self.state = SharedState()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # shared YOLO pass (persons + COCO objects like 'knife')
        od_cfg = config.get("object_detector", {}) or {}
        self.object_detector: Optional[YoloEngine] = None
        if od_cfg.get("enabled", True):
            self.object_detector = YoloEngine(
                model_path=od_cfg.get("model", "yolo26n.pt"),
                conf=float(od_cfg.get("conf", 0.35)),
                device=od_cfg.get("device", "cpu"),
            )
        self.person_label = od_cfg.get("person_label", "person")

        det_cfg = config.get("detectors", {}) or {}
        self.detectors = {
            "fall": FallDetector(det_cfg.get("fall", {}), enabled=det_cfg.get("fall", {}).get("enabled", True)),
            "fire": FireDetector(det_cfg.get("fire", {}), enabled=det_cfg.get("fire", {}).get("enabled", True)),
            "weapon": WeaponDetector(det_cfg.get("weapon", {}), enabled=det_cfg.get("weapon", {}).get("enabled", True)),
        }

        self.fusion = FusionEngine(config.get("fusion", {}) or {})
        self.dispatcher = NotificationDispatcher(config.get("notifications", {}) or {})
        self.alarm = AlarmManager(config.get("alarm", {}) or {})
        self.event_manager = EventManager(config, self.dispatcher, self.alarm)
        self.event_manager.set_cooldowns(self.fusion.cooldowns)
        self.event_manager.on_event = self.state.push_event

        self.max_fps = float(config.get_path("processing.max_fps", 15) or 15)
        self._active_event: Optional[Event] = None
        self._active_event_until = 0.0

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    # ---- controls (used by the server API) --------------------------------
    def toggle_detector(self, name: str, enabled: bool) -> bool:
        det = self.detectors.get(name)
        if det is None:
            return False
        det.set_enabled(enabled)
        return True

    def acknowledge_alarm(self) -> None:
        self.alarm.acknowledge()

    # ---- main loop --------------------------------------------------------
    def _run(self) -> None:
        try:
            source = FrameSource(self.config.get("source", {}) or {})
        except Exception as exc:
            log.error("Failed to open source: %s", exc)
            self.state.status["running"] = False
            self.state.status["error"] = str(exc)
            return

        self.state.status["running"] = True
        min_dt = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        last_t = 0.0
        fps_ema = 0.0

        log.info("Pipeline started.")
        while not self._stop.is_set():
            now = time.time()
            if min_dt and (now - last_t) < min_dt:
                time.sleep(min_dt - (now - last_t))
            frame_t = time.time()
            dt = frame_t - last_t if last_t else 0.0
            last_t = frame_t

            ok, frame = source.read()
            if not ok or frame is None:
                log.warning("End of stream / read failure.")
                break

            detections = self._process_frame(frame, frame_t)

            # keep the active-event banner up for a few seconds after firing
            if self._active_event and frame_t > self._active_event_until:
                self._active_event = None

            annotated = draw(frame, detections, self._active_event)
            self._publish(annotated)

            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt
            self._update_status(fps_ema)

        source.release()
        self.state.status["running"] = False
        log.info("Pipeline stopped.")

    def _process_frame(self, frame, timestamp: float) -> List[Detection]:
        shared: Dict[str, Any] = {"yolo": [], "persons": []}

        if self.object_detector is not None:
            yolo_dets = self.object_detector.detect(frame)
            shared["yolo"] = yolo_dets
            shared["persons"] = [d for d in yolo_dets if d.label == self.person_label]

        all_dets: List[Detection] = list(shared["persons"])  # draw persons too
        signal_dets: List[Detection] = list(shared["persons"])  # persons feed 'person' signal

        for det in self.detectors.values():
            try:
                found = det.process(frame, timestamp, shared)
            except Exception as exc:  # pragma: no cover - keep loop alive
                log.warning("Detector %s failed: %s", det.name, exc)
                found = []
            all_dets.extend(found)
            signal_dets.extend(found)

        candidates = self.fusion.update(signal_dets)
        events = self.event_manager.handle(candidates, frame, timestamp)
        if events:
            top = max(events, key=lambda e: e.severity)
            self._active_event = top
            self._active_event_until = timestamp + 6.0

        return all_dets

    def _publish(self, frame) -> None:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            self.state.set_frame(buf.tobytes())

    def _update_status(self, fps: float) -> None:
        self.state.status["fps"] = round(fps, 1)
        self.state.status["alarm_active"] = self.alarm.active
        self.state.status["active_event"] = (
            self._active_event.to_dict() if self._active_event else None
        )
        self.state.status["detectors"] = {
            name: det.enabled for name, det in self.detectors.items()
        }
        self.state.status["object_detector"] = bool(
            self.object_detector and self.object_detector.available
        )
