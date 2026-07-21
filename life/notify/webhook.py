"""Webhook adapter: POSTs the event as JSON to a configurable URL.

Best-effort: failures are logged but never crash the pipeline. Point this at any
local stub server to simulate an agency dispatch endpoint.
"""
from __future__ import annotations

import logging

from ..models import Event
from .base import NotificationAdapter

log = logging.getLogger("life.notify.webhook")


class WebhookAdapter(NotificationAdapter):
    name = "webhook"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.url = self.config.get("url", "")

    def send(self, event: Event) -> None:
        if not self.url:
            log.warning("Webhook enabled but no url configured; skipping.")
            return
        try:
            import requests

            resp = requests.post(self.url, json=event.to_dict(), timeout=4)
            log.info("Webhook -> %s (HTTP %s)", self.url, resp.status_code)
        except Exception as exc:  # never break the pipeline over a notification
            log.warning("Webhook POST to %s failed: %s", self.url, exc)
