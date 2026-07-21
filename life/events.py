"""Event manager: cooldown/debounce, snapshots, JSONL logging, dispatch, alarm.

Takes EventCandidates from the fusion engine and, respecting a per-event-type
cooldown, promotes them to confirmed Events: saves a snapshot, appends a JSONL
record, fires notifications, and raises the alarm.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Callable, Dict, List, Optional

from .alarm import AlarmManager
from .models import Event, EventCandidate, Severity
from .notify import NotificationDispatcher

log = logging.getLogger("life.events")


class EventManager:
    def __init__(self, config, dispatcher: NotificationDispatcher,
                 alarm: AlarmManager) -> None:
        self.dispatcher = dispatcher
        self.alarm = alarm

        responders = config.get_path("notifications.responders", {}) or {}
        self.responders = responders

        self.cooldowns: Dict[str, float] = {}  # populated from fusion engine

        storage = config.get("storage", {}) or {}
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.snapshots_dir = os.path.join(base, storage.get("snapshots_dir", "data/snapshots"))
        self.log_file = os.path.join(base, storage.get("log_file", "data/events.jsonl"))
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        self._last_dispatch: Dict[str, float] = {}
        self.recent: deque = deque(maxlen=200)
        self.on_event: Optional[Callable[[Event], None]] = None

    def set_cooldowns(self, cooldowns: Dict[str, float]) -> None:
        self.cooldowns = dict(cooldowns)

    def handle(self, candidates: List[EventCandidate], frame, timestamp: float) -> List[Event]:
        emitted: List[Event] = []
        for cand in candidates:
            cooldown = self.cooldowns.get(cand.event_type, 30.0)
            last = self._last_dispatch.get(cand.event_type, 0.0)
            if timestamp - last < cooldown:
                continue  # debounced: already alerted for this incident recently

            self._last_dispatch[cand.event_type] = timestamp
            event = self._build_event(cand)
            event.snapshot_path = self._save_snapshot(frame, event)
            self._log(event)
            self.recent.appendleft(event)

            self.dispatcher.dispatch(event)
            self.alarm.trigger(event.severity)
            if self.on_event:
                try:
                    self.on_event(event)
                except Exception as exc:  # pragma: no cover
                    log.debug("on_event callback failed: %s", exc)

            emitted.append(event)
        return emitted

    def _build_event(self, cand: EventCandidate) -> Event:
        responder = self.responders.get(cand.responder, {}) or {}
        return Event(
            event_type=cand.event_type,
            severity=cand.severity,
            responder=cand.responder,
            responder_name=responder.get("name", cand.responder),
            responder_contact=str(responder.get("contact", "")),
            confidence=cand.confidence,
            message=cand.message,
            detections=cand.detections,
        )

    def _save_snapshot(self, frame, event: Event) -> Optional[str]:
        if frame is None:
            return None
        try:
            import cv2

            ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(event.timestamp))
            fname = f"{ts}_{event.event_type}.jpg"
            path = os.path.join(self.snapshots_dir, fname)
            cv2.imwrite(path, frame)
            return path
        except Exception as exc:  # pragma: no cover
            log.debug("Snapshot save failed: %s", exc)
            return None

    def _log(self, event: Event) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event.to_dict()) + "\n")
        except Exception as exc:  # pragma: no cover
            log.debug("Event log write failed: %s", exc)
