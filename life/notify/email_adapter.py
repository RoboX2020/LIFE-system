"""Email adapter.

Simulated by default: if the required SMTP_* env vars are absent it just logs the
payload it *would* have sent. If they are present, it sends a real email via SMTP.
Drop-in for real dispatch with zero code changes - only credentials.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

from ..models import Event
from .base import NotificationAdapter

log = logging.getLogger("life.notify.email")


class EmailAdapter(NotificationAdapter):
    name = "email"

    def send(self, event: Event) -> None:
        host = os.getenv("SMTP_HOST")
        port = os.getenv("SMTP_PORT")
        user = os.getenv("SMTP_USER")
        password = os.getenv("SMTP_PASS")
        sender = os.getenv("ALERT_EMAIL_FROM", user or "life@localhost")
        recipient = os.getenv("ALERT_EMAIL_TO")

        subject = f"[LIFE][{event.severity}] {event.event_type} -> {event.responder_name}"
        body = (
            f"{event.message}\n\n"
            f"Responder: {event.responder_name} ({event.responder_contact})\n"
            f"Severity : {event.severity}\n"
            f"Confidence: {event.confidence:.0%}\n"
            f"Time     : {event.to_dict()['time_iso']}\n"
            f"Snapshot : {event.snapshot_path or '-'}\n"
        )

        if not all([host, port, recipient]):
            log.warning(
                "[SIMULATED EMAIL] To=%s Subject=%s\n%s",
                recipient or "<unset ALERT_EMAIL_TO>", subject, body,
            )
            return

        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = recipient
            msg.set_content(body)
            with smtplib.SMTP(host, int(port), timeout=8) as smtp:
                smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.send_message(msg)
            log.info("Email dispatched to %s", recipient)
        except Exception as exc:
            log.warning("Email send failed (%s); payload was: %s", exc, subject)
