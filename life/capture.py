"""Video capture: webcam, file, RTSP, or a synthetic 'demo' source.

The demo source needs no hardware and produces a flickering fire-like blob so the
whole pipeline (detection -> fusion -> alarm -> notify -> dashboard) can be seen
running on a fresh machine.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np


class FrameSource:
    """Unified frame source. Call `read()` to get (ok, frame_bgr)."""

    def __init__(self, config) -> None:
        self.type = str(config.get("type", "webcam")).lower()
        self.path = config.get("path", 0)
        self.width = int(config.get("width", 1280))
        self.height = int(config.get("height", 720))
        self.loop = bool(config.get("loop", True))
        self._cap: Optional[cv2.VideoCapture] = None
        self._demo_t0 = time.time()
        self._demo_frame = 0
        self._open()

    def _open(self) -> None:
        if self.type == "demo":
            return
        if self.type == "webcam":
            src: object = int(self.path)
        else:  # file or rtsp
            src = str(self.path)
        cap = cv2.VideoCapture(src)
        if self.type == "webcam":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video source (type={self.type}, path={self.path})."
            )
        self._cap = cap

    def read(self):
        if self.type == "demo":
            return True, self._demo_frame_img()

        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            if self.type in ("file",) and self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return False, None
        return True, frame

    def _demo_frame_img(self):
        """Synthetic scene: dark room with a flickering orange fire blob that grows."""
        h, w = self.height, self.width
        frame = np.full((h, w, 3), 24, dtype=np.uint8)  # dark background
        cv2.putText(
            frame, "LIFE DEMO source (synthetic fire)", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2,
        )

        t = self._demo_frame
        # Fire ignites after ~1.5s of frames, then flickers.
        if t > 20:
            cx, cy = int(w * 0.5), int(h * 0.7)
            base = 60 + (t - 20) * 2
            radius = min(160, base)
            flicker = int(20 * np.sin(t * 1.3) + 12 * np.sin(t * 0.7))
            radius = max(30, radius + flicker)
            # layered orange/yellow blobs (BGR)
            cv2.circle(frame, (cx, cy), radius, (20, 90, 235), -1)          # orange
            cv2.circle(frame, (cx, cy), int(radius * 0.6), (60, 170, 255), -1)  # lighter
            cv2.circle(frame, (cx, cy), int(radius * 0.3), (120, 240, 255), -1)  # near white-yellow
            cv2.putText(
                frame, "flicker=%d r=%d" % (flicker, radius), (cx - 90, cy + radius + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

        self._demo_frame += 1
        time.sleep(1 / 20.0)  # ~20 fps synthetic feed
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
