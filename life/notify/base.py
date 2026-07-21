"""Common notification-adapter interface."""
from __future__ import annotations

from ..models import Event


class NotificationAdapter:
    name = "adapter"

    def __init__(self, config) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))

    def send(self, event: Event) -> None:
        raise NotImplementedError
