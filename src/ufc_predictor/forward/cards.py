"""From an ESPN card to rows the point-in-time feature builder can describe.

Three steps, each conservative in the same way as the odds join (a dropped or refused
row beats a wrong one):

* **Names to fighter ids.** An ESPN name is matched to the UFCStats history by its
  normalized form (or a reordering of its words), then by the hand-reviewed table
  ``forward/espn_aliases.csv``. A name with no exact match but a near one raises,
  listing every such name, because silently treating a veteran as a debutant would hand
  them an empty history. "Near" is a character ratio of at least 0.85 on the
  word-sorted names: one-letter typos score 0.91-0.96 and "Alex"/"Alexander" 0.87,
  while the debutants on the October 2026 cards score at most 0.76 against anyone. A
  real debutant who is near a known name is confirmed in the alias table with the id
  ``DEBUT``. Only a name with no candidate at all is a debutant automatically.
* **History after the snapshot.** UFCStats is unreachable from here (its anti-bot
  challenge), so bouts after the committed snapshot enter history from ESPN results,
  saved under ``data/raw/espn_results/``: winner, method and round, but no per-round
  statistics. Elo, Glicko-2, records, streaks and layoffs stay current; per-round rate
  features use the last statistics available.
* **The card itself** is appended unlabelled, on its own date, so every feature comes
  from strictly earlier bouts by the same builder used everywhere else.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from ufc_predictor.forward.espn import Card
from ufc_predictor.ingest.names import normalize_name, similarity

NEAR_MATCH = 0.85
DEBUT = "DEBUT"
ALIASES_PATH = Path(__file__).with_name("espn_aliases.csv")
SYNTHETIC_PREFIX = "espn:"

#: Columns of the fights table, in order, with the nullable integer columns typed.
FIGHT_COLUMNS = [
    "fight_id",
    "event_id",
    "event_name",
    "date",
    "pair_key",
    "fighter_1",
    "fighter_2",
    "name_1",
    "name_2",
    "winner_id",
    "method",
    "method_class",
    "finish_round",
    "finish_time_sec",
    "weight_class",
    "title_bout",
    "scheduled_rounds",
    "is_decisive",
]
_INT_COLUMNS = ("finish_round", "finish_time_sec", "scheduled_rounds")


class NameMatchError(ValueError):
    """An upcoming fighter's name is close to, but not exactly, a known fighter's."""


def fighter_index(fights: pd.DataFrame) -> dict[str, str]:
    """Normalized name -> fighter id, from every bout in the history.

    Raises:
        ValueError: if one normalized name belongs to two fighter ids.
    """
    pairs = pd.concat(
        [
            fights[["name_1", "fighter_1"]].set_axis(["name", "id"], axis=1),
            fights[["name_2", "fighter_2"]].set_axis(["name", "id"], axis=1),
        ]
    ).drop_duplicates()
    clash = pairs.loc[pairs["name"].duplicated(keep=False)]
    if not clash.empty:
        raise ValueError(f"names shared by several fighter ids: {sorted(set(clash['name']))[:10]}")
    return dict(zip(pairs["name"], pairs["id"], strict=True))


def near_ratio(a: str, b: str) -> float:
    """Character similarity of two normalized names, ignoring word order."""
    return SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, str]:
    """ESPN name (normalized) -> fighter id (or ``DEBUT``), hand-reviewed."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {
            normalize_name(row["espn_name"]): row["fighter_id"].strip()
            for row in csv.DictReader(fh)
            if row.get("espn_name") and row.get("fighter_id")
        }


def match_names(
    names: Iterable[str], index: dict[str, str], aliases: dict[str, str]
) -> dict[str, str | None]:
    """Each ESPN name's fighter id, or None for a debutant.

    A name whose tokens are a reordering of a known name ("weili zhang" for
    "zhang weili") is the same fighter and matches.

    Raises:
        NameMatchError: listing every name that has a near match but no exact one.
    """
    out: dict[str, str | None] = {}
    near: list[str] = []
    known = list(index)
    for name in names:
        norm = normalize_name(name)
        if norm in aliases:
            out[name] = None if aliases[norm] == DEBUT else aliases[norm]
            continue
        if norm in index:
            out[name] = index[norm]
            continue
        reordered = [k for k in known if similarity(norm, k) == 1.0]
        if len(reordered) == 1:
            out[name] = index[reordered[0]]
            continue
        scored = sorted(((near_ratio(norm, k), k) for k in known), reverse=True)[:3]
        if reordered or (scored and scored[0][0] >= NEAR_MATCH):
            near.append(f"{name!r} ~ " + ", ".join(f"{k!r} ({s:.2f})" for s, k in scored))
        else:
            out[name] = None
    if near:
        raise NameMatchError(
            "upcoming fighters with a near but not exact match in the history; add the "
            f"right one to {ALIASES_PATH.name} (espn_name,fighter_id) or confirm a "
            "debutant: " + "; ".join(near)
        )
    return out


def synthetic_id(name: str) -> str:
    """A stable id for a fighter first seen after the snapshot."""
    return SYNTHETIC_PREFIX + normalize_name(name)


def _resolve(names: Iterable[str], index: dict[str, str], aliases: dict[str, str]) -> dict:
    matched = match_names(names, index, aliases)
    return {n: (fid if fid is not None else synthetic_id(n)) for n, fid in matched.items()}


def _bout_row(
    *,
    fight_id: str,
    event_id: str,
    event_name: str,
    date: dt.date,
    a: str,
    b: str,
    ids: dict[str, str],
    scheduled_rounds: int | None,
    weight_class: str,
    winner: str | None = None,
    method_class: str | None = None,
    finish_round: int | None = None,
) -> dict:
    ida, idb = ids[a], ids[b]
    if ida == idb:
        raise ValueError(f"{a!r} and {b!r} resolve to the same fighter {ida}")
    (id1, n1), (id2, n2) = sorted([(ida, normalize_name(a)), (idb, normalize_name(b))])
    winner_id = ids[winner] if winner is not None else None
    title = bool(
        scheduled_rounds == 5 and re.match(r"^UFC \d+", event_name)
    )  # ESPN carries no title flag; see title_bout_rule in the README
    return {
        "fight_id": fight_id,
        "event_id": event_id,
        "event_name": event_name,
        "date": date,
        "pair_key": "|".join(sorted([n1, n2])),
        "fighter_1": id1,
        "fighter_2": id2,
        "name_1": n1,
        "name_2": n2,
        "winner_id": winner_id,
        "method": method_class,
        "method_class": method_class,
        "finish_round": finish_round,
        "finish_time_sec": None,
        "weight_class": weight_class,
        "title_bout": title,
        "scheduled_rounds": scheduled_rounds,
        "is_decisive": winner_id is not None,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=FIGHT_COLUMNS)
    for col in _INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("Int64")
    df["title_bout"] = df["title_bout"].astype(bool)
    df["is_decisive"] = df["is_decisive"].astype(bool)
    return df


def card_rows(card: Card, index: dict[str, str], aliases: dict[str, str]) -> pd.DataFrame:
    """The card's bouts as unlabelled fights-table rows."""
    names = [n for b in card.bouts for n in (b.fighter_a, b.fighter_b)]
    ids = _resolve(names, index, aliases)
    rows = [
        _bout_row(
            fight_id=f"espn-{b.competition_id}",
            event_id=f"espn-{card.event_id}",
            event_name=card.name,
            date=card.date,
            a=b.fighter_a,
            b=b.fighter_b,
            ids=ids,
            scheduled_rounds=b.scheduled_rounds,
            weight_class=b.weight_class,
        )
        for b in card.bouts
    ]
    return _frame(rows)


def save_results(path: Path, card: Card, results: list) -> None:
    """Persist one event's ESPN results for the forward history (committed)."""
    by_id = {r.competition_id: r for r in results}
    bouts = []
    for b in card.bouts:
        r = by_id.get(b.competition_id)
        bouts.append(
            {
                "competition_id": b.competition_id,
                "fighter_a": b.fighter_a,
                "fighter_b": b.fighter_b,
                "scheduled_rounds": b.scheduled_rounds,
                "weight_class": b.weight_class,
                "completed": bool(r and r.completed),
                "winner": r.winner if r else None,
                "method_class": r.method_class if r else None,
                "finish_round": r.finish_round if r else None,
            }
        )
    payload = {
        "event_id": card.event_id,
        "name": card.name,
        "date": card.date.isoformat(),
        "start_utc": card.start_utc.isoformat(),
        "bouts": bouts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def results_rows(
    results_dir: Path, index: dict[str, str], aliases: dict[str, str], *, before: dt.date
) -> pd.DataFrame:
    """Completed bouts from every saved ESPN results file dated before ``before``.

    A bout ESPN lists as not completed (cancelled, or never held) is left out; a
    completed bout with no winner is a draw or no contest and kept as non-decisive.
    """
    rows: list[dict] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        ev = json.loads(path.read_text(encoding="utf-8"))
        date = dt.date.fromisoformat(ev["date"])
        if date >= before:
            continue
        done = [b for b in ev["bouts"] if b["completed"]]
        ids = _resolve([n for b in done for n in (b["fighter_a"], b["fighter_b"])], index, aliases)
        for b in done:
            rows.append(
                _bout_row(
                    fight_id=f"espn-{b['competition_id']}",
                    event_id=f"espn-{ev['event_id']}",
                    event_name=ev["name"],
                    date=date,
                    a=b["fighter_a"],
                    b=b["fighter_b"],
                    ids=ids,
                    scheduled_rounds=b["scheduled_rounds"],
                    weight_class=b["weight_class"],
                    winner=b["winner"],
                    method_class=b["method_class"],
                    finish_round=b["finish_round"],
                )
            )
    return _frame(rows)


def forward_history(
    fights: pd.DataFrame, results_dir: Path, aliases: dict[str, str], *, before: dt.date
) -> tuple[pd.DataFrame, dict[str, str]]:
    """The snapshot's fights plus ESPN results dated after it and before ``before``.

    Returns the combined fights table and the name index (including fighters first seen
    in the ESPN results, under their synthetic ids).

    Raises:
        ValueError: if an ESPN results file overlaps the snapshot's dates.
    """
    index = fighter_index(fights)
    extra = results_rows(results_dir, index, aliases, before=before)
    if not extra.empty and extra["date"].min() <= fights["date"].max():
        raise ValueError(
            f"ESPN results dated {extra['date'].min()} overlap the snapshot, which runs to "
            f"{fights['date'].max()}; history would count those bouts twice"
        )
    combined = pd.concat([fights.loc[:, FIGHT_COLUMNS], extra], ignore_index=True)
    combined = combined.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)
    for _, row in extra.iterrows():
        index.setdefault(row["name_1"], row["fighter_1"])
        index.setdefault(row["name_2"], row["fighter_2"])
    return combined, index
