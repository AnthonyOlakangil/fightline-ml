"""What the scheduled forward-test job should do now, as a pure function of its inputs.

GitHub Actions runs ``ufc-predictor forward run-due`` every few hours. Each run asks
:func:`due_actions` which events need which step:

* ``predict`` -- the card starts more than ``min_lead_hours`` and at most
  ``predict_window_hours`` from now, and has no predictions yet;
* ``missed``  -- the card has no predictions and starts within ``min_lead_hours`` (or
  has started). It is recorded and alerted, never predicted late;
* ``close``   -- predictions exist, the card starts within ``close_window_hours``, and
  no pre-start price snapshot has been taken (a fallback: closing prices are read
  from Kalshi's candlesticks at settlement);
* ``settle``  -- predictions exist, the card has started, and every bout is final, or
  the card started more than ``settle_after_hours`` ago.
"""

from __future__ import annotations

import datetime as dt

from ufc_predictor.forward.espn import Card
from ufc_predictor.forward.ledger import has


def due_actions(
    cards: list[Card],
    entries: list[dict],
    now: dt.datetime,
    *,
    min_lead_hours: float,
    predict_window_hours: float,
    close_window_hours: float,
    settle_after_hours: float,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for card in sorted(cards, key=lambda c: c.start_utc):
        ev = card.event_id
        lead = (card.start_utc - now).total_seconds() / 3600
        predicted = has(entries, ev, "predict")
        if not predicted:
            if has(entries, ev, "missed"):
                continue
            if min_lead_hours < lead <= predict_window_hours:
                out.append(("predict", ev))
            elif lead <= min_lead_hours:
                out.append(("missed", ev))
            continue
        if has(entries, ev, "settle"):
            continue
        if 0 < lead <= close_window_hours and not has(entries, ev, "close"):
            out.append(("close", ev))
        all_final = all(b.status == "STATUS_FINAL" for b in card.bouts)
        if lead < 0 and (all_final or -lead >= settle_after_hours):
            out.append(("settle", ev))
    return out
