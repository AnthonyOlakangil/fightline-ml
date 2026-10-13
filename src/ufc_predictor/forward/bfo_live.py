"""BestFightOdds' current lines: a sportsbook cross-check on the Kalshi price.

The homepage lists every upcoming event with each book's current moneyline. The
consensus here is the median over **sportsbooks** of the implied probability; Kalshi and
Polymarket also appear as columns but are exchanges, and Kalshi is the primary price
already, so both are left out of the cross-check.
"""

from __future__ import annotations

import re
import statistics

import pandas as pd
from bs4 import BeautifulSoup

HOME_URL = "https://www.bestfightodds.com/"
EXCHANGES = frozenset({"Kalshi", "Polymarket"})
_AMERICAN = re.compile(r"^([+-]\d{3,5}|EV)$")


def _implied(american: int) -> float:
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def _american(p: float) -> int:
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def _price(text: str) -> int | None:
    text = text.strip()
    if not _AMERICAN.match(text):
        return None
    return 100 if text == "EV" else int(text)


def _book_name(th) -> str:
    """A book's column title: its link text, or the cell text when it has no link."""
    for a in th.find_all("a"):
        if "bookie-bonus-badge" not in (a.get("class") or []):
            return a.get_text(strip=True)
    return next(th.stripped_strings, "")


def current_odds(fetcher) -> pd.DataFrame:
    """One row per fighter per listed bout, with the sportsbook median and best price.

    Raises:
        ValueError: if the page holds no odds tables, or a table has no book header.
    """
    soup = BeautifulSoup(fetcher.text(HOME_URL), "html.parser")
    wrappers = soup.find_all("div", class_="table-outer-wrapper")
    if not wrappers:
        raise ValueError("BestFightOdds page shape changed: no event tables on the homepage")
    rows: list[dict] = []
    for wrap in wrappers:
        header = wrap.find(class_="table-header")
        tables = wrap.find_all("table")
        if header is None or len(tables) < 2:
            raise ValueError("BestFightOdds page shape changed: an event lacks its odds table")
        table = tables[1]
        books = {
            th["data-b"]: _book_name(th)
            for th in table.find("thead").find_all("th")
            if th.get("data-b")
        }
        if not books:
            raise ValueError("BestFightOdds page shape changed: an odds table has no book header")
        event = header.find("h1").get_text(strip=True).removesuffix(" Odds")
        date_el = header.find(class_="table-header-date")
        date_text = date_el.get_text(strip=True) if date_el else ""

        fighters: list[tuple[str, dict[str, int]]] = []
        for tr in table.find("tbody").find_all("tr"):
            link = tr.find("a", href=re.compile(r"^/fighters/"))
            if link is None:
                continue  # proposition rows (over/under, method) carry no fighter link
            prices: dict[str, int] = {}
            for td in tr.find_all("td", attrs={"data-li": True}):
                book = td["data-li"].strip("[]").split(",")[0]
                span = td.find("span")
                value = _price(span.get_text()) if span else None
                if value is not None and book in books:
                    prices[books[book]] = value
            fighters.append((link.get_text(strip=True), prices))
        if len(fighters) % 2:
            raise ValueError(
                f"BestFightOdds page shape changed: {event} lists an odd fighter count"
            )
        for i in range(0, len(fighters), 2):
            for (name, prices), (opp, _) in (
                (fighters[i], fighters[i + 1]),
                (fighters[i + 1], fighters[i]),
            ):
                books_only = [v for b, v in prices.items() if b not in EXCHANGES]
                median_p = (
                    statistics.median(_implied(v) for v in books_only) if books_only else None
                )
                rows.append(
                    {
                        "event": event,
                        "event_date_text": date_text,
                        "fighter": name,
                        "opponent": opp,
                        "n_books": len(books_only),
                        "median_p": median_p,
                        "median_odds": _american(median_p) if median_p is not None else None,
                        "best_odds": max(books_only, key=lambda v: 1 / _implied(v))
                        if books_only
                        else None,
                    }
                )
    return pd.DataFrame(rows)
