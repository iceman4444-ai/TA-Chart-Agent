#!/usr/bin/env python3
"""Alert when the briefing stops sending.

Deliberately standalone. It imports NOTHING from ta_briefing_v2 and uses
only the standard library, because the failure it exists to catch was an
import error in the briefing's own dependencies: a watchdog that installed
requirements.txt or imported the briefing would have died of the same
cause and stayed just as silent. Keep it that way — no third-party
imports, no shared helpers, no pip install step in the workflow.

The briefing has now stopped twice without anyone noticing: once when the
API credits ran out, and once for three weeks when mplfinance lost its
undeclared dependency on six. Both were invisible because every piece of
reporting lived inside the process that had died.

It checks the most recent weekday whose sending window has fully closed,
and emails if either slot is missing from .last-send.
"""
from __future__ import annotations

import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
MARKER = Path(".last-send")
# The last afternoon catch-up cron fires about 21:30 ET; past 22:00 ET a
# missing slot is genuinely missing rather than still pending.
WINDOW_CLOSES_HOUR = 22
ACTIONS_URL = "https://github.com/iceman4444-ai/TA-Chart-Agent/actions"


def last_closed_weekday(now: datetime) -> str:
    """The most recent weekday whose send window has fully passed, ET.

    Run at 04:45 UTC this is 23:45 ET in winter (same day) and 00:45 ET in
    summer (next day), so the hour test is what makes both land on the same
    trading day rather than the clock date doing it.
    """
    day = now.date()
    if now.hour < WINDOW_CLOSES_HOUR:
        day -= timedelta(days=1)
    while day.weekday() > 4:  # 5 = Saturday, 6 = Sunday
        day -= timedelta(days=1)
    return day.isoformat()


def owner_address() -> str:
    """The alert recipient, read from config.yaml without a YAML parser.

    A targeted line scan rather than PyYAML: the whole point of this script
    is that it has no installable dependencies. If the key moves, the
    fallback keeps the alert deliverable rather than losing it.
    """
    try:
        for line in Path("config.yaml").read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("username:"):
                return stripped.split(":", 1)[1].strip()
    except OSError:
        pass
    return "onulakjd@gmail.com"


def smtp_password() -> str | None:
    """The Gmail app password, whichever secret slot happens to hold it.

    The repo's two legacy secrets are interchangeable by name, so they are
    told apart by value: Anthropic keys start with sk-ant-.
    """
    for name in ("TA_SMTP_PASSWORD", "SECRET_A", "SECRET_B"):
        val = (os.environ.get(name) or "").strip()
        if val and not val.startswith("sk-ant-"):
            return val
    return None


def send_alert(subject: str, body: str) -> None:
    to_addr = owner_address()
    password = smtp_password()
    if not password:
        # Exit non-zero so the run goes red: an undeliverable alert is
        # itself the kind of silent failure this script exists to prevent.
        print("::error::no SMTP password available — cannot send the alert")
        sys.exit(1)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = to_addr
    msg["To"] = to_addr
    msg.set_content(body)
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as server:
        server.starttls()
        server.login(to_addr, password)
        server.send_message(msg)
    print(f"alert sent to {to_addr}")


def main() -> int:
    now = datetime.now(EASTERN)
    day = last_closed_weekday(now)

    if not MARKER.exists():
        send_alert(
            "⚠️ TA Briefing: no send history at all",
            f"{MARKER} does not exist, so no send has ever been recorded.\n"
            f"Checked at {now:%Y-%m-%d %H:%M %Z}.\n\n{ACTIONS_URL}\n",
        )
        return 0

    recorded = {ln.strip() for ln in MARKER.read_text().splitlines() if ln.strip()}
    missing = [s for s in ("morning", "afternoon") if f"{day} {s}" not in recorded]
    newest = max(recorded) if recorded else "(none)"

    if not missing:
        print(f"ok: both slots recorded for {day}")
        return 0

    send_alert(
        f"⚠️ TA Briefing missed the {' and '.join(missing)} recap on {day}",
        f"The briefing did not record its {' or '.join(missing)} send for {day}.\n\n"
        f"Most recent marker on file: {newest}\n"
        f"Checked at {now:%Y-%m-%d %H:%M %Z}.\n\n"
        "The briefing crashes before it can report anything when the failure "
        "is at import time, which is why this check runs separately. Start "
        "with the most recent failed run:\n\n"
        f"{ACTIONS_URL}\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
