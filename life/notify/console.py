"""Console adapter: prints a clear, human-readable dispatch line."""
from __future__ import annotations

import logging

from ..models import Event
from .base import NotificationAdapter

log = logging.getLogger("life.notify.console")


class ConsoleAdapter(NotificationAdapter):
    name = "console"

    def send(self, event: Event) -> None:
        log.warning(
            "[DISPATCH -> %s (%s)] %s | severity=%s conf=%.0f%% | snapshot=%s",
            event.responder_name,
            event.responder_contact,
            event.message,
            event.severity,
            event.confidence * 100,
            event.snapshot_path or "-",
        )
