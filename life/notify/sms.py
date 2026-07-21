"""SMS adapter.

Simulated by default. If TWILIO_* env vars and the `twilio` package are present,
it sends a real SMS. Otherwise it logs the message it would have sent.
"""
from __future__ import annotations

import logging
import os

from ..models import Event
from .base import NotificationAdapter

log = logging.getLogger("life.notify.sms")


class SmsAdapter(NotificationAdapter):
    name = "sms"

    def send(self, event: Event) -> None:
        sid = os.getenv("TWILIO_SID")
        token = os.getenv("TWILIO_TOKEN")
        from_num = os.getenv("TWILIO_FROM")
        to_num = os.getenv("ALERT_SMS_TO")

        text = (
            f"LIFE {event.severity}: {event.event_type} -> "
            f"{event.responder_name} ({event.responder_contact}). "
            f"{event.message}"
        )

        if not all([sid, token, from_num, to_num]):
            log.warning("[SIMULATED SMS] To=%s | %s", to_num or "<unset ALERT_SMS_TO>", text)
            return

        try:
            from twilio.rest import Client

            client = Client(sid, token)
            client.messages.create(body=text, from_=from_num, to=to_num)
            log.info("SMS dispatched to %s", to_num)
        except Exception as exc:
            log.warning("SMS send failed (%s); message was: %s", exc, text)
