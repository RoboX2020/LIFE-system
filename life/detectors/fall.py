"""Fall detection via MediaPipe pose + multi-criteria voting.

Landmarks come from the MediaPipe Pose Landmarker (Tasks API, 33 points). If that
build isn't available it falls back to the legacy ``mp.solutions.pose`` API. The
Tasks model bundle is downloaded automatically on first use.

Signals combined per frame:
  1. Torso angle   - shoulder->hip vector vs vertical; near-horizontal => on ground.
  2. Vertical drop - rapid downward motion of the body centroid within a short window.
  3. Aspect ratio  - pose bounding box switches from tall (standing) to wide (lying).

A weighted vote plus a per-person state machine
(STANDING -> FALLING -> ON_GROUND -> CONFIRMED_FALL) suppresses false positives
from sitting/bending.
"""
from __future__ import annotations

import logging
import math
import os
import urllib.request
from collections import deque
from typing import Any, Dict, List, Optional

from ..models import Detection
from .base import Detector

log = logging.getLogger("life.fall")

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"
)
_DEFAULT_TASK = os.path.join(_MODELS_DIR, "pose_landmarker_lite.task")
_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


class _State:
    STANDING = "STANDING"
    FALLING = "FALLING"
    ON_GROUND = "ON_GROUND"
    CONFIRMED = "CONFIRMED_FALL"


class FallDetector(Detector):
    name = "fall"

    def __init__(self, config, enabled: bool = True) -> None:
        super().__init__(config, enabled)
        self.horizontal_angle = float(config.get("horizontal_angle", 55.0))
        self.fall_velocity = float(config.get("fall_velocity", 0.30))
        self.velocity_window = float(config.get("velocity_window_sec", 0.8))
        self.aspect_wide = float(config.get("aspect_ratio_wide", 1.1))
        self.ground_frames = int(config.get("ground_persistence_frames", 6))
        self.vote_threshold = int(config.get("vote_threshold", 2))
        self.min_visibility = float(config.get("min_visibility", 0.5))
        self.task_path = config.get("model_task", _DEFAULT_TASK)

        self._backend = None            # "tasks" | "solutions" | None
        self._landmarker = None         # tasks PoseLandmarker
        self._pose = None               # solutions Pose
        self._mp = None
        self._ts_ms = 0                 # monotonically increasing timestamp for tasks API

        self._centroids: deque = deque(maxlen=64)  # (timestamp, centroid_y[0..1])
        self._state = _State.STANDING
        self._ground_count = 0
        self._init_pose()

    # ---- backend initialisation ------------------------------------------
    def _init_pose(self) -> None:
        if not self.enabled:
            return
        try:
            import mediapipe as mp

            self._mp = mp
        except Exception as exc:  # pragma: no cover - env dependent
            log.warning("MediaPipe unavailable (%s); fall detection disabled.", exc)
            return

        if self._init_tasks_backend():
            self._backend = "tasks"
            log.info("Fall detection using MediaPipe Tasks Pose Landmarker.")
            return
        if self._init_solutions_backend():
            self._backend = "solutions"
            log.info("Fall detection using legacy mp.solutions.pose.")
            return
        log.warning("No usable MediaPipe pose backend; fall detection disabled.")

    def _init_tasks_backend(self) -> bool:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception:
            return False
        if not self._ensure_task_model():
            return False
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=self.task_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            return True
        except Exception as exc:
            log.warning("Could not create PoseLandmarker (%s).", exc)
            return False

    def _init_solutions_backend(self) -> bool:
        try:
            if not hasattr(self._mp, "solutions"):
                return False
            self._pose = self._mp.solutions.pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return True
        except Exception:
            return False

    def _ensure_task_model(self) -> bool:
        if os.path.exists(self.task_path):
            return True
        try:
            os.makedirs(os.path.dirname(self.task_path), exist_ok=True)
            log.info("Downloading pose landmarker model to %s ...", self.task_path)
            urllib.request.urlretrieve(_TASK_URL, self.task_path)
            return os.path.exists(self.task_path)
        except Exception as exc:
            log.warning("Could not download pose model (%s). Fall detection off.", exc)
            return False

    # ---- per-frame processing --------------------------------------------
    def process(self, frame, timestamp: float, shared: Dict[str, Any]) -> List[Detection]:
        if not self.enabled or self._backend is None:
            return []

        pts = self._landmarks(frame)
        if pts is None:
            self._decay()
            return []

        h, w = frame.shape[:2]
        try:
            ls, rs, lh, rh = pts[11], pts[12], pts[23], pts[24]
        except IndexError:
            return []

        if _vis(ls, rs, lh, rh) < self.min_visibility:
            self._decay()
            return []

        sx, sy = (ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0
        hx, hy = (lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0

        torso_angle = self._torso_angle(sx, sy, hx, hy)
        centroid_y = (sy + hy) / 2.0
        self._centroids.append((timestamp, centroid_y))
        velocity = self._recent_down_velocity(timestamp)
        aspect = self._aspect_ratio(pts)

        votes, reasons = 0, []
        if torso_angle >= self.horizontal_angle:
            votes += 1; reasons.append("horizontal")
        if velocity >= self.fall_velocity:
            votes += 1; reasons.append("rapid_drop")
        if aspect >= self.aspect_wide:
            votes += 1; reasons.append("wide_bbox")

        self._advance_state(votes, torso_angle)

        if self._state == _State.CONFIRMED:
            confidence = min(0.99, 0.5 + 0.15 * votes)
            return [
                Detection(
                    label="fall", confidence=confidence,
                    bbox=self._pose_bbox(pts, w, h), detector=self.name, signal="fall",
                    meta={
                        "torso_angle": round(torso_angle, 1),
                        "down_velocity": round(velocity, 3),
                        "aspect": round(aspect, 2),
                        "reasons": reasons, "state": self._state,
                    },
                )
            ]
        return []

    def _landmarks(self, frame):
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._backend == "tasks":
            mp = self._mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._ts_ms += 33  # strictly increasing timestamps required
            result = self._landmarker.detect_for_video(mp_image, self._ts_ms)
            poses = getattr(result, "pose_landmarks", None)
            if not poses:
                return None
            return poses[0]
        # solutions backend
        result = self._pose.process(rgb)
        lm = getattr(result, "pose_landmarks", None)
        return lm.landmark if lm is not None else None

    # ---- geometry helpers -------------------------------------------------
    @staticmethod
    def _torso_angle(sx: float, sy: float, hx: float, hy: float) -> float:
        vx, vy = hx - sx, hy - sy
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            return 0.0
        cos_a = max(-1.0, min(1.0, vy / norm))
        return math.degrees(math.acos(cos_a))

    def _recent_down_velocity(self, now: float) -> float:
        recent = [(t, y) for (t, y) in self._centroids if now - t <= self.velocity_window]
        if len(recent) < 2:
            return 0.0
        t0, y0 = recent[0]
        t1, y1 = recent[-1]
        return (y1 - y0) / max(1e-3, t1 - t0)

    @staticmethod
    def _aspect_ratio(pts) -> float:
        xs = [p.x for p in pts if _v(p) > 0.3]
        ys = [p.y for p in pts if _v(p) > 0.3]
        if not xs or not ys:
            return 0.0
        height = max(ys) - min(ys)
        if height < 1e-6:
            return 999.0
        return (max(xs) - min(xs)) / height

    @staticmethod
    def _pose_bbox(pts, w: int, h: int):
        xs = [p.x for p in pts if _v(p) > 0.3]
        ys = [p.y for p in pts if _v(p) > 0.3]
        if not xs or not ys:
            return None
        return (
            max(0, int(min(xs) * w)), max(0, int(min(ys) * h)),
            min(w, int(max(xs) * w)), min(h, int(max(ys) * h)),
        )

    # ---- state machine ----------------------------------------------------
    def _advance_state(self, votes: int, torso_angle: float) -> None:
        on_ground_now = votes >= self.vote_threshold
        if self._state == _State.STANDING:
            if on_ground_now:
                self._state = _State.FALLING
                self._ground_count = 1
        elif self._state == _State.FALLING:
            if on_ground_now:
                self._ground_count += 1
                self._state = _State.ON_GROUND
            else:
                self._reset()
        elif self._state == _State.ON_GROUND:
            if on_ground_now:
                self._ground_count += 1
                if self._ground_count >= self.ground_frames:
                    self._state = _State.CONFIRMED
            else:
                self._reset()
        elif self._state == _State.CONFIRMED:
            if not on_ground_now and torso_angle < self.horizontal_angle * 0.7:
                self._reset()

    def _decay(self) -> None:
        if self._state in (_State.FALLING, _State.ON_GROUND):
            self._ground_count = max(0, self._ground_count - 1)
            if self._ground_count == 0:
                self._reset()

    def _reset(self) -> None:
        self._state = _State.STANDING
        self._ground_count = 0


def _v(p) -> float:
    """Visibility of a landmark (Tasks and solutions both expose .visibility)."""
    return float(getattr(p, "visibility", 1.0) or 0.0)


def _vis(*points) -> float:
    return min(_v(p) for p in points)
