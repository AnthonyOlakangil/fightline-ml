"""Join BestFightOdds opening and closing lines to UFCStats fights.

    uv run ufc-predictor bfo-join   # -> data/processed/bfo_lines.parquet,
                                    #    reports/bfo_join.json, reports/bfo_unjoined.csv

The closing-line-value analysis (``analysis/clv.py``) needs, for each development-window
fight, the market's opening price and its closing price under one definition. Both come
from the committed BestFightOdds snapshot (``ingest/bfo.py``); this module attaches them
to ``fight_id`` and re-expresses them for ``fighter_1``.

**Matching is exact and same-date only.** A BestFightOdds bout joins a fight only when
the event date is identical and the two normalized names are the same pair, read
directly or through the hand-reviewed alias table (``ingest/names.py``), as in the
closing-odds join. There is no fuzzy matching and no date tolerance: a bout whose pair
UFCStats records only on another date is dropped and counted as a date mismatch, never
moved. A dropped bout costs sample size; a wrong pairing would attach one fight's prices
to another fight's result without changing any count.

**Prices.** Per bout, both sides' American prices are converted to implied
probabilities and proportionally de-vigged across the two sides:

* opening: from each side's ``open_*`` price -> ``open_p1``;
* closing: each side's closing *range* (lowest and highest close across the books
  BestFightOdds tracked) is reduced to its midpoint in implied-probability space, then
  the two midpoints are de-vigged -> ``close_p1``.

The raw per-side midpoints (``close_mid_p1_raw``/``close_mid_p2_raw``) and the oriented
American prices are kept so the arithmetic can be audited row by row. A bout missing any
of its six prices is dropped and counted.

**Fail loudly.** A pair listed twice on one date, one fight claimed by two bouts, a bout
matching two fights, and any joined fight dated in the closed evaluation window
(``windows.eval_start`` on) all raise.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ufc_predictor.config import REPO_ROOT, Config, get_config
from ufc_predictor.experiments.protocol import EvalWindowClosed
from ufc_predictor.ingest.bfo import LINE_COLUMNS, load_snapshot
from ufc_predictor.ingest.names import load_aliases, normalize_name
from ufc_predictor.ingest.odds import american_to_implied, remove_vig
from ufc_predictor.ingest.stages import Stage
from ufc_predictor.storage import db

#: The six prices a bout needs to be joined.
PRICE_COLUMNS = (
    "open_fighter",
    "open_opponent",
    "close_lo_fighter",
    "close_hi_fighter",
    "close_lo_opponent",
    "close_hi_opponent",
)
FIGHT_COLUMNS = ("fight_id", "date", "fighter_1", "fighter_2", "name_1", "name_2")
#: Oriented American prices kept on each joined row.
AMERICAN_COLUMNS = (
    "open_odds_1",
    "open_odds_2",
    "close_lo_1",
    "close_hi_1",
    "close_lo_2",
    "close_hi_2",
)

JOINED_COLUMNS = [
    "fight_id",
    "date",
    "fighter_1",
    "fighter_2",
    "bfo_event_id",
    "match_method",
    *AMERICAN_COLUMNS,
    "close_mid_p1_raw",
    "close_mid_p2_raw",
    "close_mid_odds_1",
    "close_mid_odds_2",
    "open_overround",
    "close_overround",
    "open_p1",
    "close_p1",
]
UNJOINED_COLUMNS = [
    "date",
    "event",
    "bfo_event_id",
    "fighter",
    "opponent",
    "name_fighter",
    "name_opponent",
    "reason",
    "nearest_fight_date",
    "days_apart",
    *PRICE_COLUMNS,
]

# Reasons a bout is not joined (the unjoined CSV's ``reason`` column).
NO_NAME_MATCH = "no_name_match"
DATE_MISMATCH = "date_mismatch"
INCOMPLETE_PRICES = "incomplete_prices"


# ---------------------------------------------------------------------------
# Price arithmetic
# ---------------------------------------------------------------------------


def implied_to_american(p: float) -> float:
    """The American price whose implied probability is ``p`` (no rounding).

    >>> round(implied_to_american(0.6), 6)
    -150.0
    >>> round(implied_to_american(0.4), 6)
    150.0
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"cannot express probability {p!r} as an American price")
    decimal = 1.0 / p
    return (decimal - 1.0) * 100.0 if decimal >= 2.0 else -100.0 / (decimal - 1.0)


def _implied(price) -> float:
    return american_to_implied(float(price))


def _prices(rec: dict) -> dict:
    """Both sides' implied and de-vigged probabilities, in BestFightOdds' order."""
    open_f, open_o = _implied(rec["open_fighter"]), _implied(rec["open_opponent"])
    mid_f = (_implied(rec["close_lo_fighter"]) + _implied(rec["close_hi_fighter"])) / 2.0
    mid_o = (_implied(rec["close_lo_opponent"]) + _implied(rec["close_hi_opponent"])) / 2.0
    open_p_f, open_p_o = remove_vig(open_f, open_o)
    close_p_f, close_p_o = remove_vig(mid_f, mid_o)
    return {
        "open_p": (open_p_f, open_p_o),
        "close_p": (close_p_f, close_p_o),
        "mid": (mid_f, mid_o),
        "open_overround": open_f + open_o - 1.0,
        "close_overround": mid_f + mid_o - 1.0,
    }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _as_dates(values: pd.Series, what: str) -> pd.Series:
    """Coerce to ``datetime.date``; a missing or unparseable date raises."""
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any():
        raise ValueError(f"{int(parsed.isna().sum())} {what} have no date")
    return parsed.dt.date


def _require_columns(frame: pd.DataFrame, columns, what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} lack columns {missing}")


def _join(
    lines: pd.DataFrame,
    fights: pd.DataFrame,
    aliases: dict[str, str],
    *,
    eval_start: dt.date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(joined, unjoined, audit)``; :func:`join_bfo` documents the rules."""
    _require_columns(lines, LINE_COLUMNS, "BestFightOdds lines")
    _require_columns(fights, FIGHT_COLUMNS, "fights")
    lines = lines.assign(date=_as_dates(lines["date"], "BestFightOdds bouts")).reset_index(
        drop=True
    )
    fights = fights.assign(date=_as_dates(fights["date"], "fights")).reset_index(drop=True)

    # Fights by (date, pair), and every date each pair fought on. A key can hold two
    # fights (UFC Japan 1997 ran Sakuraba vs Silveira twice in one night); a bout that
    # lands on such a key cannot be placed and raises below.
    by_key: dict[tuple[dt.date, frozenset[str]], list[int]] = defaultdict(list)
    dates_by_pair: dict[frozenset[str], set[dt.date]] = defaultdict(set)
    for i, (date, n1, n2) in enumerate(
        zip(fights["date"], fights["name_1"], fights["name_2"], strict=True)
    ):
        pair = frozenset((normalize_name(n1), normalize_name(n2)))
        by_key[(date, pair)].append(i)
        dates_by_pair[pair].add(date)

    # A bout listed twice on one date is a scrape fault, not two bouts.
    raw_names = [
        (normalize_name(a), normalize_name(b))
        for a, b in zip(lines["fighter"], lines["opponent"], strict=True)
    ]
    raw_keys = pd.Series(
        [(d, frozenset(p)) for d, p in zip(lines["date"], raw_names, strict=True)], dtype=object
    )
    if raw_keys.duplicated().any():
        first = lines.loc[raw_keys.duplicated(keep=False).idxmax()]
        raise ValueError(
            f"BestFightOdds lists {first['fighter']} vs {first['opponent']} twice on "
            f"{first['date']}; one bout cannot carry two sets of prices"
        )

    resolved: list[dict] = []
    for rec, (na, nb) in zip(lines.to_dict("records"), raw_names, strict=True):
        raw_pair = frozenset((na, nb))
        # 1. exact on the normalized pair; 2. the alias table, then exact again.
        method, fight_rows = "exact", by_key.get((rec["date"], raw_pair), [])
        if not fight_rows:
            aa, ab = aliases.get(na, na), aliases.get(nb, nb)
            if (aa, ab) != (na, nb):
                hit = by_key.get((rec["date"], frozenset((aa, ab))), [])
                if hit:
                    method, fight_rows, na, nb = "alias", hit, aa, ab
                else:
                    na, nb = aa, ab
        if len(fight_rows) > 1:
            raise ValueError(
                f"{rec['fighter']} vs {rec['opponent']} on {rec['date']} matches "
                f"{len(fight_rows)} fights; the bout cannot be placed"
            )
        out = {**rec, "name_fighter": na, "name_opponent": nb, "fight_row": None}
        if fight_rows:
            out["fight_row"] = fight_rows[0]
            out["match_method"] = method
            complete = all(pd.notna(rec[c]) for c in PRICE_COLUMNS)
            out["reason"] = None if complete else INCOMPLETE_PRICES
        else:
            # The pair (as listed, or through the alias table) fought only on other dates.
            other = dates_by_pair.get(raw_pair, set()) | dates_by_pair.get(
                frozenset((na, nb)), set()
            )
            if other:
                nearest = min(other, key=lambda d: (abs((d - rec["date"]).days), d))
                out["reason"] = DATE_MISMATCH
                out["nearest_fight_date"] = nearest
                out["days_apart"] = (nearest - rec["date"]).days
            else:
                out["reason"] = NO_NAME_MATCH
        resolved.append(out)

    matched = [r for r in resolved if r["fight_row"] is not None]
    claimed = pd.Series([r["fight_row"] for r in matched], dtype="int64")
    if claimed.duplicated().any():
        fight = fights.loc[int(claimed[claimed.duplicated()].iloc[0])]
        raise ValueError(
            f"two BestFightOdds bouts resolve to the same fight ({fight['name_1']} vs "
            f"{fight['name_2']}, {fight['date']}, fight_id {fight['fight_id']})"
        )

    rows: list[dict] = []
    for r in (r for r in matched if r["reason"] is None):
        fight = fights.loc[r["fight_row"]]
        n1, n2 = normalize_name(fight["name_1"]), normalize_name(fight["name_2"])
        if (r["name_fighter"], r["name_opponent"]) == (n1, n2):
            first_is_1 = True
        elif (r["name_fighter"], r["name_opponent"]) == (n2, n1):
            first_is_1 = False
        else:  # pragma: no cover - an exact pair match has one of the two orders
            raise AssertionError(f"cannot orient {r['fighter']} vs {r['opponent']}")
        p = _prices(r)
        f, o = (0, 1) if first_is_1 else (1, 0)
        side = ("fighter", "opponent") if first_is_1 else ("opponent", "fighter")
        rows.append(
            {
                "fight_id": fight["fight_id"],
                "date": fight["date"],
                "fighter_1": fight["fighter_1"],
                "fighter_2": fight["fighter_2"],
                "bfo_event_id": int(r["event_id"]),
                "match_method": r["match_method"],
                "open_odds_1": int(r[f"open_{side[0]}"]),
                "open_odds_2": int(r[f"open_{side[1]}"]),
                "close_lo_1": int(r[f"close_lo_{side[0]}"]),
                "close_hi_1": int(r[f"close_hi_{side[0]}"]),
                "close_lo_2": int(r[f"close_lo_{side[1]}"]),
                "close_hi_2": int(r[f"close_hi_{side[1]}"]),
                "close_mid_p1_raw": p["mid"][f],
                "close_mid_p2_raw": p["mid"][o],
                "close_mid_odds_1": implied_to_american(p["mid"][f]),
                "close_mid_odds_2": implied_to_american(p["mid"][o]),
                "open_overround": p["open_overround"],
                "close_overround": p["close_overround"],
                "open_p1": p["open_p"][f],
                "close_p1": p["close_p"][f],
            }
        )
    joined = pd.DataFrame(rows, columns=JOINED_COLUMNS)
    joined = joined.astype(dict.fromkeys(AMERICAN_COLUMNS, "int64"))
    joined = joined.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)

    late = joined.loc[joined["date"] >= eval_start]
    if not late.empty:
        raise EvalWindowClosed(
            f"{len(late)} BestFightOdds bouts join fights dated from {late['date'].min()}, "
            f"inside the closed evaluation window (from {eval_start}); the BestFightOdds "
            "join covers the development window only"
        )

    unjoined = pd.DataFrame(
        [{**r, "bfo_event_id": r["event_id"]} for r in resolved if r["reason"] is not None],
        columns=UNJOINED_COLUMNS,
    )
    unjoined["days_apart"] = unjoined["days_apart"].astype("Int64")
    unjoined = unjoined.sort_values(["date", "bfo_event_id", "fighter"], kind="stable")
    unjoined = unjoined.reset_index(drop=True)

    reasons = unjoined["reason"].value_counts().to_dict()
    drops = {
        "dropped_incomplete": int(reasons.get(INCOMPLETE_PRICES, 0)),
        "dropped_date_mismatch": int(reasons.get(DATE_MISMATCH, 0)),
        "unmatched_no_name_match": int(reasons.get(NO_NAME_MATCH, 0)),
    }
    # Every bout is joined or dropped for exactly one counted reason.
    Stage("bfo_join", len(lines), len(joined), drops).reconcile()
    audit = {
        "lines_in": len(lines),
        "joined": len(joined),
        "joined_exact": int((joined["match_method"] == "exact").sum()),
        "joined_alias": int((joined["match_method"] == "alias").sum()),
        **drops,
        "join_rate": len(joined) / len(lines) if len(lines) else 0.0,
    }
    return joined, unjoined, audit


def join_bfo(
    lines: pd.DataFrame,
    fights: pd.DataFrame,
    aliases: dict[str, str],
    *,
    eval_start: dt.date | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach BestFightOdds opening and closing prices to fights, oriented to ``fighter_1``.

    Args:
        lines: BestFightOdds bouts, one row per bout (``ingest.bfo.LINE_COLUMNS``).
        fights: the fights table; at least ``fight_id, date, fighter_1, fighter_2,
            name_1, name_2`` with ``name_1``/``name_2`` aligned to ``fighter_1``/``fighter_2``.
        aliases: the hand-reviewed alias table, odds-source name -> UFCStats name, both
            normalized (``names.load_aliases``).
        eval_start: first day of the closed evaluation window; defaults to the config's.

    Returns:
        ``(joined, audit)``: one row per joined fight with ``open_p1``/``close_p1``
        (de-vigged probabilities for ``fighter_1``) and the oriented inputs behind them;
        ``audit`` counts every bout as joined or dropped for one reason.

    Raises:
        ValueError: if a pair is listed twice on one date, two bouts resolve to one
            fight, one bout matches two fights, or a price cannot be converted.
        EvalWindowClosed: if a joined fight is dated on or after ``eval_start``.
    """
    eval_start = eval_start or get_config().windows.eval_start
    joined, _, audit = _join(lines, fights, aliases, eval_start=eval_start)
    return joined, audit


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _dev_priced_coverage(
    joined: pd.DataFrame, fights: pd.DataFrame, odds: pd.DataFrame, cfg: Config
) -> dict:
    """Join rate among the CLV population: priced, decisive, development-window fights."""
    _require_columns(fights, ("fight_id", "date", "is_decisive", "winner_id"), "fights")
    _require_columns(odds, ("fight_id",), "odds")
    dates = _as_dates(fights["date"], "fights")
    w = cfg.windows
    population = fights.loc[
        fights["fight_id"].isin(set(odds["fight_id"]))
        & fights["is_decisive"].astype(bool)
        & fights["winner_id"].notna()
        & (dates >= w.dev_start)
        & (dates <= w.dev_end),
        "fight_id",
    ]
    if population.empty:
        raise ValueError(
            f"no priced, decisive fights dated {w.dev_start}..{w.dev_end}; "
            "the fights or odds table is not the expected one"
        )
    n_joined = int(population.isin(set(joined["fight_id"])).sum())
    return {
        "dev_window": [str(w.dev_start), str(w.dev_end)],
        "dev_priced_fights": len(population),
        "dev_priced_joined": n_joined,
        "dev_priced_join_rate": n_joined / len(population),
    }


def build(cfg: Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the committed snapshot and the processed tables; return ``(joined, unjoined, audit)``.

    Writes nothing, so it is safe to call from tests.
    """
    cfg = cfg or get_config()
    lines = load_snapshot(cfg)
    fights = db.read_table(cfg.paths.processed_dir, "fights")
    odds = db.read_table(cfg.paths.processed_dir, "odds")
    aliases = load_aliases(cfg.paths.aliases)
    joined, unjoined, audit = _join(lines, fights, aliases, eval_start=cfg.windows.eval_start)
    audit = {
        "snapshot_dir": str(cfg.bfo["snapshot_dir"]),
        **audit,
        **_dev_priced_coverage(joined, fights, odds, cfg),
        "date_min": str(joined["date"].min()) if len(joined) else None,
        "date_max": str(joined["date"].max()) if len(joined) else None,
    }
    return joined, unjoined, audit


def _shown(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def run(cfg: Config | None = None, *, verbose: bool = True) -> dict:
    """Join, then write ``bfo_lines.parquet``, ``bfo_join.json`` and ``bfo_unjoined.csv``."""
    cfg = cfg or get_config()
    joined, unjoined, audit = build(cfg)

    table = db.write_table(joined, cfg.paths.processed_dir, "bfo_lines")
    reports = Path(cfg.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    audit_path, unjoined_path = reports / "bfo_join.json", reports / "bfo_unjoined.csv"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    unjoined.to_csv(unjoined_path, index=False)

    if verbose:
        print(
            f"\nBestFightOdds join: {audit['lines_in']:,} bouts in -> {audit['joined']:,} joined "
            f"(exact {audit['joined_exact']:,}, alias {audit['joined_alias']:,}; "
            f"{audit['join_rate']:.1%})\n"
            f"  dropped: no name match {audit['unmatched_no_name_match']:,}, "
            f"date mismatch {audit['dropped_date_mismatch']:,}, "
            f"incomplete prices {audit['dropped_incomplete']:,}\n"
            f"  priced decisive development-window fights: {audit['dev_priced_joined']:,} of "
            f"{audit['dev_priced_fights']:,} joined ({audit['dev_priced_join_rate']:.1%})\n"
            f"  wrote {_shown(table)} ({len(joined):,} rows), {_shown(audit_path)}, "
            f"{_shown(unjoined_path)} ({len(unjoined):,} rows)\n"
        )
    return audit
