"""Snapshot acquisition: fetch source CSVs once and commit them as dated Parquet.

This module is the *only* part of the project that touches the network, and it is not
on the graded path. ``make data`` reads the committed Parquet snapshot; this runs by
hand when refreshing it.

Why a snapshot at all: UFCStats itself is unscrapeable from a clean environment as of
October 2026 (a JavaScript proof-of-work challenge returns a ~3 KB stub instead of
data, and the host serves HTTP only). CLAUDE.md requires that the graded path not
depend on a live site, so the canonical fight data is read from a committed snapshot of
a public mirror.

All parsing here is written for this project. No code from the mirror is reused: its
scraper is GPL-3.0, while the CSV data it publishes is a factual record of fight
results. Provenance, retrieval date, row counts, and source checksums go into
``SOURCE.md`` beside the Parquet so the snapshot is auditable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

USER_AGENT = "ufc-predictor/0.1 (portfolio research project; one-off snapshot fetch)"
REQUEST_TIMEOUT = 120


@dataclass(frozen=True)
class SourceFile:
    """One source CSV and where it lands in the snapshot."""

    table: str
    url: str
    description: str


UFCSTATS_MIRROR_BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
ODDS_BASE = "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master"

UFCSTATS_SOURCES: tuple[SourceFile, ...] = (
    SourceFile(
        "events", f"{UFCSTATS_MIRROR_BASE}/ufc_event_details.csv", "Event name, date, location"
    ),
    SourceFile(
        "fight_results",
        f"{UFCSTATS_MIRROR_BASE}/ufc_fight_results.csv",
        "One row per bout: outcome, method, round, time, referee",
    ),
    SourceFile(
        "fight_stats",
        f"{UFCSTATS_MIRROR_BASE}/ufc_fight_stats.csv",
        "One row per fighter per round: strikes, takedowns, control time",
    ),
    SourceFile(
        "fighter_tott",
        f"{UFCSTATS_MIRROR_BASE}/ufc_fighter_tott.csv",
        "Fighter tale-of-the-tape: height, weight, reach, stance, date of birth",
    ),
)

ODDS_SOURCES: tuple[SourceFile, ...] = (
    SourceFile(
        "odds_master",
        f"{ODDS_BASE}/ufc-master.csv",
        "Historical fights with closing moneylines (only 5 columns are ever used)",
    ),
)

# Columns read from the odds file. Everything else in that CSV is a precomputed career
# aggregate whose point-in-time correctness cannot be established, so it is dropped at
# the boundary rather than carried around and filtered later.
ODDS_COLUMN_WHITELIST: tuple[str, ...] = (
    "R_fighter",
    "B_fighter",
    "date",
    "R_odds",
    "B_odds",
)


def _fetch(url: str) -> bytes:
    """GET a URL, raising on any non-200 or short body."""
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    if len(resp.content) < 1024:
        raise RuntimeError(
            f"{url} returned only {len(resp.content)} bytes; refusing to snapshot a "
            "truncated or challenge-page response"
        )
    return resp.content


def _read_csv_bytes(raw: bytes, table: str) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame, keeping everything as text.

    Types are assigned in ``normalize.py``, not here: the snapshot should be a faithful
    copy of what the source published, so that re-parsing is a separate, testable step.
    """
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, na_values=[""])
    if df.empty:
        raise RuntimeError(f"{table}: parsed zero rows")
    if table == "odds_master":
        missing = [c for c in ODDS_COLUMN_WHITELIST if c not in df.columns]
        if missing:
            raise RuntimeError(f"odds_master is missing expected columns: {missing}")
        df = df.loc[:, list(ODDS_COLUMN_WHITELIST)].copy()
    return df


def fetch_snapshot(
    sources: tuple[SourceFile, ...],
    dest: Path,
    *,
    title: str,
    source_repo: str,
    source_license: str,
    notes: str = "",
) -> dict[str, int]:
    """Download ``sources``, write them as Parquet under ``dest``, and record provenance.

    Returns:
        Mapping of table name to row count.
    """
    dest.mkdir(parents=True, exist_ok=True)
    retrieved = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    counts: dict[str, int] = {}
    lines: list[str] = []

    for src in sources:
        raw = _fetch(src.url)
        digest = hashlib.sha256(raw).hexdigest()
        df = _read_csv_bytes(raw, src.table)
        out = dest / f"{src.table}.parquet"
        df.to_parquet(out, index=False, compression="zstd")
        counts[src.table] = len(df)
        lines.append(
            f"### `{src.table}.parquet`\n\n"
            f"- {src.description}\n"
            f"- Source: `{src.url}`\n"
            f"- Rows: **{len(df):,}**\n"
            f"- Columns: {len(df.columns)}\n"
            f"- SHA-256 of source CSV: `{digest}`\n"
            f"- Parquet size: {out.stat().st_size / 1024:.0f} KiB\n"
        )

    source_md = dest / "SOURCE.md"
    source_md.write_text(
        f"# {title}\n\n"
        f"- **Retrieved (UTC):** {retrieved}\n"
        f"- **Upstream repository:** {source_repo}\n"
        f"- **Upstream license:** {source_license}\n\n"
        f"{notes}\n\n"
        "No upstream code is vendored in this project. The CSV data published by the\n"
        "upstream repository is a factual record of fight outcomes; all parsing,\n"
        "normalization, and schema logic here is written for this repository.\n\n"
        "Row counts below are asserted by `tests/test_ingest.py`, so a silent change in\n"
        "a refreshed snapshot fails the suite instead of shifting results unnoticed.\n\n"
        "## Tables\n\n" + "\n".join(lines),
        encoding="utf-8",
    )
    return counts
