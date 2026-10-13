"""BestFightOdds: historical opening and closing lines for the development window.

    uv run ufc-predictor bfo          # crawl (cached) and write the dated snapshot

The closing-line dataset used everywhere else carries no opening price, so it cannot
say whether a model's disagreement with the market at the open predicts where the line
closes. BestFightOdds fighter pages carry both: an ``Open`` price and a ``Closing
range`` (the lowest and highest close across the books it tracks) for each side of
every bout, back to 2008.

**Crawl path** (chosen in the Phase 8 discovery spike, 2026-10-13):

1. ``/sitemap-events.xml`` lists every event with its date as ``lastmod``;
2. each UFC event dated inside the window is fetched, and its odds table yields the
   fighter page of every bout (rows ``tr#mu-<matchup id>`` and the row after it);
3. each of those fighter pages is fetched and parsed. Every page lists both sides of
   every bout the fighter had, so most bouts are seen twice; the snapshot keeps one row
   per bout and raises if the two views disagree on a price.

For 2010-2015 that is 212 event pages and about 1,030 fighter pages, roughly 1,250
requests, or 1 h 45 min at one request per ``bfo.delay_s`` (5 s).

**Politeness.** ``robots.txt`` (checked 2026-10-13) reads ``User-agent: * / Allow: /``
with no crawl delay, and the terms page holds only an accuracy disclaimer, silent on
automated access. Requests carry an identifying User-Agent, every response is cached on
disk under ``data/cache/bfo/`` (git-ignored), a cached URL is never requested again,
and the delay applies before every network request.

Nothing on the graded path calls this module: the committed Parquet snapshot is the
input, and the crawl is documented so it can be repeated.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ufc_predictor.config import REPO_ROOT, Config, get_config

#: Columns of the parsed fighter-page frame, in order.
LINE_COLUMNS = [
    "fighter",
    "fighter_url",
    "opponent",
    "opponent_url",
    "event",
    "event_id",
    "event_url",
    "date",
    "open_fighter",
    "open_opponent",
    "close_lo_fighter",
    "close_hi_fighter",
    "close_lo_opponent",
    "close_hi_opponent",
]

_PRICE_COLUMNS = LINE_COLUMNS[8:]
_AMERICAN = re.compile(r"^[+-]\d{3,5}$")
_ORDINAL = re.compile(r"(\d)(st|nd|rd|th)\b")
_TRAILING_ID = re.compile(r"-(\d+)$")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def cache_path(url: str, cache_dir: Path) -> Path:
    """Where the response for ``url`` is (or will be) cached."""
    return Path(cache_dir) / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".html")


def fetch(url: str, cache_dir: Path, delay_s: float, *, user_agent: str | None = None) -> str:
    """Return the body of ``url``, from the on-disk cache when present.

    A cache hit makes no request and does not sleep. A miss sleeps ``delay_s`` first,
    so consecutive misses are spaced by at least that long, then caches the body.

    Raises:
        requests.HTTPError: on a non-2xx response (nothing is cached).
    """
    path = cache_path(url, cache_dir)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    agent = user_agent or str(get_config().bfo["user_agent"])
    time.sleep(delay_s)
    resp = requests.get(url, headers={"User-Agent": agent}, timeout=30)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text, encoding="utf-8")
    return resp.text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _trailing_id(url: str, what: str) -> int:
    m = _TRAILING_ID.search(url.rstrip("/"))
    if not m:
        raise ValueError(f"BestFightOdds page shape changed: no numeric id in {what} {url!r}")
    return int(m.group(1))


def _price(text: str) -> int | None:
    """An American price, or None when the cell is blank or not a price."""
    text = text.strip()
    return int(text) if _AMERICAN.match(text) else None


def _parse_date(text: str) -> dt.date:
    cleaned = _ORDINAL.sub(r"\1", " ".join(text.split()))
    try:
        return dt.datetime.strptime(cleaned, "%b %d %Y").date()
    except ValueError as exc:
        raise ValueError(f"cannot parse BestFightOdds date {text!r}") from exc


def _side(row) -> tuple[str, str, list[int | None]]:
    link = row.find("a", href=re.compile(r"^/fighters/"))
    cells = row.find_all("td", class_="moneyline")
    if link is None or len(cells) != 3:
        raise ValueError(
            "BestFightOdds page shape changed: a bout row lacks a fighter link or its "
            f"three price cells (open, closing low, closing high): {row.get_text(' ', strip=True)[:80]!r}"
        )
    return link.get_text(strip=True), link["href"], [_price(c.get_text()) for c in cells]


def parse_fighter_page(html: str) -> pd.DataFrame:
    """Every bout on a fighter page: both sides' opening price and closing range.

    Rows come in blocks of three: an event header (event link and date), the page
    owner's prices, then the opponent's prices. A blank or non-numeric price cell is
    returned as missing, never as zero.

    Raises:
        ValueError: if the odds table or a bout block does not have the expected shape,
            or a date cannot be parsed.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="team-stats-table")
    header = [th.get_text(" ", strip=True) for th in table.find_all("th")[:3]] if table else []
    if header != ["Matchup", "Open", "Closing range"]:
        raise ValueError(
            "BestFightOdds page shape changed: no odds table headed "
            f"'Matchup | Open | Closing range' (found {header})"
        )

    rows = table.find_all("tr")
    out: list[dict] = []
    i = 1
    while i < len(rows):
        row = rows[i]
        if "event-header" not in (row.get("class") or []):
            raise ValueError(
                "BestFightOdds page shape changed: expected an event header row, got "
                f"{row.get_text(' ', strip=True)[:80]!r}"
            )
        if i + 2 >= len(rows) or "event-header" in (rows[i + 2].get("class") or []):
            raise ValueError(
                "BestFightOdds page shape changed: an event header is not followed by two "
                f"fighter rows ({row.get_text(' ', strip=True)[:80]!r})"
            )
        event_link = row.find("a", href=re.compile(r"^/events/"))
        if event_link is None:
            raise ValueError("BestFightOdds page shape changed: event header has no event link")
        date_text = row.get_text(" ", strip=True).replace(event_link.get_text(" ", strip=True), "")
        fighter, fighter_url, own = _side(rows[i + 1])
        opponent, opponent_url, opp = _side(rows[i + 2])
        out.append(
            {
                "fighter": fighter,
                "fighter_url": fighter_url,
                "opponent": opponent,
                "opponent_url": opponent_url,
                "event": event_link.get_text(" ", strip=True),
                "event_id": _trailing_id(event_link["href"], "event link"),
                "event_url": event_link["href"],
                "date": _parse_date(date_text),
                "open_fighter": own[0],
                "open_opponent": opp[0],
                "close_lo_fighter": own[1],
                "close_hi_fighter": own[2],
                "close_lo_opponent": opp[1],
                "close_hi_opponent": opp[2],
            }
        )
        i += 3

    frame = pd.DataFrame(out, columns=LINE_COLUMNS)
    for col in _PRICE_COLUMNS:
        frame[col] = frame[col].astype("Int64")
    frame["event_id"] = frame["event_id"].astype("int64")
    return frame


def parse_event_page(html: str) -> pd.DataFrame:
    """The bouts listed on an event page, paired by matchup id.

    Each bout is a row ``tr#mu-<id>`` (first fighter) followed by a row for the second.

    Raises:
        ValueError: if there is no odds table, or a matchup row is not followed by a
            second fighter.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("BestFightOdds page shape changed: event page has no odds table")
    rows = table.find_all("tr")
    bouts: list[dict] = []
    for i, row in enumerate(rows):
        rid = row.get("id") or ""
        if not rid.startswith("mu-"):
            continue
        a = row.find("a", href=re.compile(r"^/fighters/"))
        b = rows[i + 1].find("a", href=re.compile(r"^/fighters/")) if i + 1 < len(rows) else None
        if a is None or b is None:
            raise ValueError(
                f"BestFightOdds page shape changed: matchup {rid} lacks two fighter links"
            )
        bouts.append(
            {
                "matchup_id": int(rid[3:]),
                "fighter_a": a.get_text(strip=True),
                "fighter_a_url": a["href"],
                "fighter_b": b.get_text(strip=True),
                "fighter_b_url": b["href"],
            }
        )
    if not bouts:
        raise ValueError("BestFightOdds page shape changed: event page lists no matchups")
    return pd.DataFrame(bouts)


def parse_events_sitemap(xml: str) -> pd.DataFrame:
    """Every event in ``sitemap-events.xml``: URL, slug, numeric id, date, UFC flag.

    Raises:
        ValueError: if the sitemap holds no dated event URLs.
    """
    pairs = re.findall(r"<loc>([^<]+)</loc>\s*<lastmod>([^<]+)</lastmod>", xml)
    if not pairs:
        raise ValueError("BestFightOdds page shape changed: sitemap lists no dated events")
    rows = []
    for url, lastmod in pairs:
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        rows.append(
            {
                "url": url,
                "slug": slug,
                "event_id": _trailing_id(slug, "event slug"),
                "date": dt.date.fromisoformat(lastmod[:10]),
                "is_ufc": slug.startswith("ufc-"),
            }
        )
    return pd.DataFrame(rows)


def events_in_window(events: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    """UFC events dated inside ``[start, end]``, in date order (ties by id)."""
    keep = events.loc[events["is_ufc"] & (events["date"] >= start) & (events["date"] <= end)]
    return keep.sort_values(["date", "event_id"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Crawl and snapshot
# ---------------------------------------------------------------------------


def dedupe_bouts(lines: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per bout, keeping the first view; raise if two views disagree.

    A bout appears on both fighters' pages with the sides swapped. The two views must
    agree on every price once re-oriented; a disagreement means the parser misread a
    page, which must not be resolved by picking one.

    Returns:
        The de-duplicated frame and the number of bouts that were seen twice and agreed.
    """
    seen: dict[tuple[int, frozenset[str]], dict] = {}
    keep: list[dict] = []
    confirmed = 0
    for rec in lines.to_dict("records"):
        key = (int(rec["event_id"]), frozenset((rec["fighter_url"], rec["opponent_url"])))
        if key not in seen:
            seen[key] = rec
            keep.append(rec)
            continue
        first = seen[key]
        same = rec["fighter_url"] == first["fighter_url"]
        pairs = [
            ("open_fighter", "open_fighter" if same else "open_opponent"),
            ("open_opponent", "open_opponent" if same else "open_fighter"),
            ("close_lo_fighter", "close_lo_fighter" if same else "close_lo_opponent"),
            ("close_hi_fighter", "close_hi_fighter" if same else "close_hi_opponent"),
            ("close_lo_opponent", "close_lo_opponent" if same else "close_lo_fighter"),
            ("close_hi_opponent", "close_hi_opponent" if same else "close_hi_fighter"),
        ]
        for a, b in pairs:
            va, vb = first[a], rec[b]
            if not ((pd.isna(va) and pd.isna(vb)) or va == vb):
                raise ValueError(
                    f"BestFightOdds pages disagree on {a} for {first['fighter']} vs "
                    f"{first['opponent']} ({first['event']}): {va} vs {vb}"
                )
        if first["date"] != rec["date"]:
            raise ValueError(
                f"BestFightOdds pages disagree on the date of {first['fighter']} vs "
                f"{first['opponent']}: {first['date']} vs {rec['date']}"
            )
        confirmed += 1
    out = pd.DataFrame(keep, columns=LINE_COLUMNS)
    for col in _PRICE_COLUMNS:
        out[col] = out[col].astype("Int64")
    return out.reset_index(drop=True), confirmed


def crawl(cfg: Config | None = None, *, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Crawl the configured window and return ``(lines, audit)``.

    Lines are kept only for bouts at the crawled UFC events, so a fighter page's
    non-UFC bouts and its bouts outside the window never enter the snapshot.
    """
    cfg = cfg or get_config()
    b = cfg.bfo
    base = str(b["base_url"]).rstrip("/")
    cache_dir = REPO_ROOT / b["cache_dir"]
    delay = float(b["delay_s"])
    start = dt.date.fromisoformat(str(b["window_start"]))
    end = dt.date.fromisoformat(str(b["window_end"]))

    def get(url: str) -> str:
        return fetch(url, cache_dir, delay, user_agent=str(b["user_agent"]))

    events = events_in_window(parse_events_sitemap(get(base + "/sitemap-events.xml")), start, end)
    if verbose:
        print(f"\nBestFightOdds crawl {start} .. {end}: {len(events)} UFC events")

    fighter_urls: list[str] = []
    listed_bouts = 0
    for ev in events.itertuples(index=False):
        bouts = parse_event_page(get(ev.url))
        listed_bouts += len(bouts)
        for u in (*bouts["fighter_a_url"], *bouts["fighter_b_url"]):
            if u not in fighter_urls:
                fighter_urls.append(u)
    if verbose:
        print(f"  event pages list {listed_bouts:,} bouts and {len(fighter_urls):,} fighters")

    frames = [parse_fighter_page(get(base + u)) for u in fighter_urls]
    raw = pd.concat(frames, ignore_index=True)
    in_scope = raw.loc[raw["event_id"].isin(set(events["event_id"]))]
    lines, confirmed = dedupe_bouts(in_scope)
    lines = lines.sort_values(["date", "event_id", "fighter", "opponent"], kind="stable")
    lines = lines.reset_index(drop=True)

    complete = lines[["open_fighter", "open_opponent"]].notna().all(axis=1) & lines[
        ["close_lo_fighter", "close_hi_fighter", "close_lo_opponent", "close_hi_opponent"]
    ].notna().all(axis=1)
    audit = {
        "window": [str(start), str(end)],
        "ufc_events": len(events),
        "bouts_listed_on_event_pages": int(listed_bouts),
        "fighter_pages": len(fighter_urls),
        "fighter_page_rows": len(raw),
        "rows_at_crawled_events": len(in_scope),
        "bouts": len(lines),
        "bouts_confirmed_by_both_pages": int(confirmed),
        "bouts_with_complete_open_and_close": int(complete.sum()),
        "date_min": str(lines["date"].min()),
        "date_max": str(lines["date"].max()),
    }
    if verbose:
        print(
            f"  bouts: {len(lines):,} ({confirmed:,} seen on both pages and agreeing); "
            f"complete open+close: {int(complete.sum()):,}"
        )
    return lines, audit


SOURCE_TEMPLATE = """\
# BestFightOdds historical opening and closing lines

- **Source:** https://www.bestfightodds.com (fighter pages: `Open` and `Closing range`).
- **Crawled:** {crawled} by `uv run ufc-predictor bfo` (`src/ufc_predictor/ingest/bfo.py`).
- **Window:** UFC events dated {start} to {end} (the development window only).
- **Path:** `/sitemap-events.xml` (event dates) -> {ufc_events} UFC event pages -> {fighter_pages}
  fighter pages. One request per {delay:g} s, identifying User-Agent, every response
  cached and never re-requested.
- **Rows:** {bouts:,} bouts ({confirmed:,} seen on both fighters' pages and identical);
  {complete:,} have both sides' opening price and closing range.
- **robots.txt** (fetched {crawled}):

```
{robots}
```

- **Terms of service** (`/terms`, fetched {crawled}): an "as is" accuracy disclaimer only;
  silent on automated access.

Columns: {columns}. Prices are American odds; a blank cell on the site is missing here,
never zero. `close_lo_*`/`close_hi_*` are the lowest and highest closing price across
the books BestFightOdds tracked for that bout.

The full audit is in `audit.json`.
"""


def write_snapshot(cfg: Config | None = None, *, verbose: bool = True) -> Path:
    """Crawl (from cache where possible) and write ``lines.parquet``, ``audit.json``, ``SOURCE.md``."""
    cfg = cfg or get_config()
    lines, audit = crawl(cfg, verbose=verbose)
    out_dir = REPO_ROOT / cfg.bfo["snapshot_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lines.to_parquet(out_dir / "lines.parquet", index=False)
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    b = cfg.bfo
    robots = fetch(
        str(b["base_url"]).rstrip("/") + "/robots.txt",
        REPO_ROOT / b["cache_dir"],
        float(b["delay_s"]),
        user_agent=str(b["user_agent"]),
    )
    (out_dir / "SOURCE.md").write_text(
        SOURCE_TEMPLATE.format(
            crawled=dt.date.today().isoformat(),
            start=audit["window"][0],
            end=audit["window"][1],
            ufc_events=audit["ufc_events"],
            fighter_pages=audit["fighter_pages"],
            delay=float(b["delay_s"]),
            bouts=audit["bouts"],
            confirmed=audit["bouts_confirmed_by_both_pages"],
            complete=audit["bouts_with_complete_open_and_close"],
            robots=robots.strip(),
            columns=", ".join(f"`{c}`" for c in LINE_COLUMNS),
        ),
        encoding="utf-8",
    )
    if verbose:
        print(f"  wrote {out_dir.relative_to(REPO_ROOT)}/lines.parquet ({len(lines):,} rows)\n")
    return out_dir


def load_snapshot(cfg: Config | None = None) -> pd.DataFrame:
    """The committed snapshot, with dates as ``datetime.date``."""
    cfg = cfg or get_config()
    path = REPO_ROOT / cfg.bfo["snapshot_dir"] / "lines.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"BestFightOdds snapshot not found at {path}")
    lines = pd.read_parquet(path)
    lines["date"] = pd.to_datetime(lines["date"]).dt.date
    return lines
