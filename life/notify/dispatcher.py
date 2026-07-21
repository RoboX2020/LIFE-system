"""Notification dispatcher: fans an Event out to all enabled adapters."""
from __future__ import annotations

import logging
import threading
from typing import List

from ..models import Event
from .base import NotificationAdapter
from .console import ConsoleAdapter
from .email_adapter import EmailAdapter
from .sms import SmsAdapter
from .webhook import WebhookAdapter

log = logging.getLogger("life.notify")

_ADAPTER_TYPES = {
    "console": ConsoleAdapter,
    "webhook": WebhookAdapter,
    "email": EmailAdapter,
    "sms": SmsAdapter,
}


class NotificationDispatcher:
    def __init__(self, config) -> None:
        adapters_cfg = (config or {}).get("adapters", {}) or {}
        self.adapters: List[NotificationAdapter] = []
        for key, cls in _ADAPTER_TYPES.items():
            cfg = adapters_cfg.get(key, {}) or {}
            adapter = cls(cfg)
            if adapter.enabled:
                self.adapters.append(adapter)
                log.info("Notification adapter enabled: %s", key)

    def dispatch(self, event: Event) -> None:
        """Send in a background thread so notifications never block detection."""
        threading.Thread(target=self._dispatch_sync, args=(event,), daemon=True).start()

    def _dispatch_sync(self, event: Event) -> None:
        for adapter in self.adapters:
            try:
                adapter.send(event)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Adapter %s failed: %s", adapter.name, exc)
