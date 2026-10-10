"""Normalize the raw snapshot into typed tables: events, fighters, fights, fight_stats.

Two decisions here are correctness-critical rather than cosmetic.

**Canonical fight orientation.** The odds source labels fighters by corner, and the red
corner wins about 58% of the time because promotions put the favourite there. Carrying
that orientation into modelling would let a model score well by learning "pick red" and
would distort any comparison against the market. So each fight is oriented by sorted
``fighter_id`` -- a key that cannot encode the outcome -- and the label is derived from
whether ``fighter_1`` won.

**Fighter identity.** The mirror keys everything on names, not ids. Where a name
resolves to exactly one UFCStats fighter the real id is used. Where a name maps to
several fighters (there are nine such names) the bout is dropped: merging two careers
would silently corrupt every point-in-time feature for both. Where a name is absent
from the fighter table a name-derived id is minted, which at worst splits one career
into two and so understates history rather than inventing it.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd

from ufc_predictor.ingest.names import normalize_name
from ufc_predictor.ingest.stages import PipelineError, StageLog

# Method strings in UFCStats collapse to these classes. "OTHER" covers draws, no
# contests, overturned results and disqualifications: not decisive, so excluded from
# the binary label with an explicit count rather than coerced into a loss.
METHOD_KO = "KO/TKO"
METHOD_SUB = "SUB"
METHOD_DEC = "DEC"
METHOD_OTHER = "OTHER"


def _as_text(value: object) -> str:
    """Coerce a cell to stripped text, treating null-ish values as empty.

    Parquet round-trips missing cells as float NaN, so every parser below goes through
    this rather than relying on ``value or ""`` -- NaN is truthy and would reach the
    regexes as a float.
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if value is pd.NA or value is pd.NaT:
        return ""
    return str(value).strip()


def classify_method(method: str | None) -> str:
    """Map a UFCStats method string to one of four classes."""
    s = _as_text(method).lower()
    if not s:
        return METHOD_OTHER
    if s.startswith("decision"):
        return METHOD_DEC
    if "ko/tko" in s or s.startswith("tko"):
        return METHOD_KO
    if "submission" in s:
        return METHOD_SUB
    return METHOD_OTHER


def parse_event_date(value: str | None) -> dt.date:
    """Parse ``"October 03, 2026"`` to a date, raising on anything unparseable."""
    s = _as_text(value)
    if not s:
        raise PipelineError("empty event date")
    try:
        return dt.datetime.strptime(s, "%B %d, %Y").date()
    except ValueError as exc:
        raise PipelineError(f"cannot parse event date {value!r}") from exc


def parse_height_cm(value: str | None) -> float | None:
    """``5' 11"`` -> 180.34 cm. Returns None for the literal ``--`` placeholder."""
    s = _as_text(value)
    if not s or s == "--":
        return None
    m = re.match(r"^(\d+)'\s*(\d+)?\"?$", s)
    if not m:
        return None
    feet = int(m.group(1))
    inches = int(m.group(2) or 0)
    return round((feet * 12 + inches) * 2.54, 2)


def parse_reach_cm(value: str | None) -> float | None:
    """``76"`` -> 193.04 cm."""
    s = _as_text(value)
    if not s or s == "--":
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\"?$", s)
    return round(float(m.group(1)) * 2.54, 2) if m else None


def parse_weight_lbs(value: str | None) -> float | None:
    s = _as_text(value)
    if not s or s == "--":
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def parse_dob(value: str | None) -> dt.date | None:
    """``Jul 13, 1978`` -> date. Returns None for ``--``."""
    s = _as_text(value)
    if not s or s == "--":
        return None
    try:
        return dt.datetime.strptime(s, "%b %d, %Y").date()
    except ValueError:
        return None


def parse_clock(value: str | None) -> int | None:
    """``4:31`` -> 271 seconds."""
    s = _as_text(value)
    if not s or s == "--":
        return None
    m = re.match(r"^(\d+):(\d{1,2})$", s)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_landed_attempted(value: str | None) -> tuple[int | None, int | None]:
    """``16 of 40`` -> (16, 40)."""
    s = _as_text(value)
    if not s or s == "--":
        return None, None
    m = re.match(r"^(\d+)\s+of\s+(\d+)$", s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def parse_round_label(value: str | None) -> int | None:
    """``Round 1`` -> 1."""
    m = re.match(r"^\s*Round\s+(\d+)\s*$", _as_text(value))
    return int(m.group(1)) if m else None


def build_events(raw: pd.DataFrame, log: StageLog) -> pd.DataFrame:
    """events: event_id, event_name, date, location."""
    df = raw.copy()
    df["event_name"] = df["EVENT"].str.strip()
    df["event_id"] = df["URL"].str.rstrip("/").str.rsplit("/", n=1).str[-1]
    df["date"] = df["DATE"].map(parse_event_date)
    df["location"] = df["LOCATION"].str.strip()

    out = df.loc[:, ["event_id", "event_name", "date", "location"]].dropna(subset=["event_id"])
    if out["event_id"].duplicated().any():
        dupes = out.loc[out["event_id"].duplicated(), "event_id"].tolist()[:5]
        raise PipelineError(f"duplicate event_ids in snapshot: {dupes}")
    log.record("events", len(raw), len(out), {"missing_event_id": len(raw) - len(out)})
    return out.reset_index(drop=True)


def build_fighters(raw: pd.DataFrame, log: StageLog) -> pd.DataFrame:
    """fighters: fighter_id, name, name_norm, height_cm, reach_cm, weight_lbs, stance, dob."""
    df = raw.copy()
    df["fighter_id"] = df["URL"].str.rstrip("/").str.rsplit("/", n=1).str[-1]
    df["name"] = df["FIGHTER"].str.strip()
    df["name_norm"] = df["name"].map(normalize_name)
    df["height_cm"] = df["HEIGHT"].map(parse_height_cm)
    df["reach_cm"] = df["REACH"].map(parse_reach_cm)
    df["weight_lbs"] = df["WEIGHT"].map(parse_weight_lbs)
    df["stance"] = df["STANCE"].str.strip().replace({"": None})
    df["dob"] = df["DOB"].map(parse_dob)

    cols = [
        "fighter_id",
        "name",
        "name_norm",
        "height_cm",
        "reach_cm",
        "weight_lbs",
        "stance",
        "dob",
    ]
    out = df.loc[:, cols].dropna(subset=["fighter_id"])
    out = out.drop_duplicates(subset=["fighter_id"])
    log.record("fighters", len(raw), len(out), {"missing_or_dup_id": len(raw) - len(out)})
    return out.reset_index(drop=True)


def _name_index(fighters: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    """Build name_norm -> fighter_id, plus the set of names that are ambiguous.

    A name shared by several fighter ids cannot be resolved from this snapshot, so it is
    returned as ambiguous and its bouts are dropped downstream.
    """
    grouped = fighters.groupby("name_norm")["fighter_id"].apply(list)
    unique_map = {n: ids[0] for n, ids in grouped.items() if len(ids) == 1 and n}
    ambiguous = {n for n, ids in grouped.items() if len(ids) > 1 and n}
    return unique_map, ambiguous


def _synthetic_id(name_norm: str) -> str:
    """Stable id for a fighter absent from the tale-of-the-tape table."""
    return "name:" + name_norm.replace(" ", "_")


def build_fights(
    raw_results: pd.DataFrame,
    events: pd.DataFrame,
    fighters: pd.DataFrame,
    log: StageLog,
    *,
    max_ambiguous_share: float = 0.02,
) -> pd.DataFrame:
    """fights, in canonical orientation.

    Columns: fight_id, event_id, event_name, date, fighter_1, fighter_2, winner_id,
    method, method_class, finish_round, finish_time_sec, weight_class, title_bout,
    scheduled_rounds, is_decisive.
    """
    df = raw_results.copy()
    rows_in = len(df)
    df["event_name"] = df["EVENT"].str.strip()
    df["fight_id"] = df["URL"].str.rstrip("/").str.rsplit("/", n=1).str[-1]

    ev = events.loc[:, ["event_id", "event_name", "date"]]
    df = df.merge(ev, on="event_name", how="left")
    no_event = int(df["event_id"].isna().sum())
    if no_event:
        # The upstream mirror's event file does not always cover every event its fight
        # file references. A fight with no event has no date, and a fight with no date
        # cannot be placed in time, so it is dropped rather than given a guessed date.
        orphan_events = sorted(df.loc[df["event_id"].isna(), "event_name"].unique())
        print(
            f"      note: {no_event} bouts dropped - event absent from the event table, "
            f"so no date is available: {orphan_events}"
        )
    df = df.loc[df["event_id"].notna()].copy()

    # Split "A vs. B".
    parts = df["BOUT"].fillna("").str.split(" vs. ", n=1, expand=True)
    if parts.shape[1] < 2:
        raise PipelineError("no bout strings contained ' vs. '; snapshot shape changed")
    df["name_a_raw"] = parts[0].str.strip()
    df["name_b_raw"] = parts[1].fillna("").str.strip()
    bad_bout = int((df["name_b_raw"] == "").sum())
    df = df.loc[df["name_b_raw"] != ""].copy()

    df["name_a"] = df["name_a_raw"].map(normalize_name)
    df["name_b"] = df["name_b_raw"].map(normalize_name)

    name_map, ambiguous = _name_index(fighters)

    amb_mask = df["name_a"].isin(ambiguous) | df["name_b"].isin(ambiguous)
    n_ambiguous = int(amb_mask.sum())
    if rows_in and n_ambiguous / rows_in > max_ambiguous_share:
        raise PipelineError(
            f"{n_ambiguous} of {rows_in} bouts ({n_ambiguous / rows_in:.1%}) have a fighter "
            f"name shared by multiple UFCStats ids, above the {max_ambiguous_share:.1%} limit. "
            "These are dropped rather than guessed because merging two fighters' careers "
            "would corrupt point-in-time features; investigate before raising the limit."
        )
    df = df.loc[~amb_mask].copy()

    df["fighter_a"] = [name_map.get(n) or _synthetic_id(n) for n in df["name_a"]]
    df["fighter_b"] = [name_map.get(n) or _synthetic_id(n) for n in df["name_b"]]
    n_synthetic = int(
        sum(1 for x in pd.concat([df["fighter_a"], df["fighter_b"]]) if str(x).startswith("name:"))
    )

    # Outcome. "W/L" means the first-named fighter won; "L/W" the second.
    outcome = df["OUTCOME"].fillna("").str.strip()
    winner = pd.Series(pd.NA, index=df.index, dtype="object")
    winner[outcome == "W/L"] = df.loc[outcome == "W/L", "fighter_a"]
    winner[outcome == "L/W"] = df.loc[outcome == "L/W", "fighter_b"]
    df["winner_id"] = winner

    df["method"] = df["METHOD"].fillna("").str.strip()
    df["method_class"] = df["method"].map(classify_method)
    df["finish_round"] = pd.to_numeric(df["ROUND"], errors="coerce").astype("Int64")
    df["finish_time_sec"] = df["TIME"].map(parse_clock).astype("Int64")
    df["weight_class"] = df["WEIGHTCLASS"].fillna("").str.strip()
    df["title_bout"] = df["weight_class"].str.contains("Title", case=False, na=False)
    df["scheduled_rounds"] = (
        df["TIME FORMAT"]
        .fillna("")
        .str.extract(r"^(\d+)\s*Rnd", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    # Decisive = a named winner and a decisive method. Draws/NCs are kept in the table
    # (they are real history for streaks and layoff) but flagged out of the label.
    df["is_decisive"] = df["winner_id"].notna() & (df["method_class"] != METHOD_OTHER)

    # Canonical orientation: sort by fighter_id so orientation cannot encode the result.
    swap = df["fighter_a"].astype(str) > df["fighter_b"].astype(str)
    df["fighter_1"] = df["fighter_a"].where(~swap, df["fighter_b"])
    df["fighter_2"] = df["fighter_b"].where(~swap, df["fighter_a"])
    # Names aligned to the canonical id order. Downstream joins (odds, stats) arrive
    # keyed on names, so carrying the aligned names avoids re-deriving the mapping --
    # id order and name order are not the same permutation.
    df["name_1"] = df["name_a"].where(~swap, df["name_b"])
    df["name_2"] = df["name_b"].where(~swap, df["name_a"])

    self_fight = int((df["fighter_1"] == df["fighter_2"]).sum())
    df = df.loc[df["fighter_1"] != df["fighter_2"]].copy()

    # pair_key carries the sorted *normalized name* pair. fight_stats identifies bouts
    # only by event + bout text, so this is the key its rows join on to inherit fight_id.
    df["pair_key"] = [
        "|".join(sorted([a, b])) for a, b in zip(df["name_a"], df["name_b"], strict=False)
    ]

    cols = [
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
    out = df.loc[:, cols].dropna(subset=["fight_id"]).drop_duplicates(subset=["fight_id"])

    log.record(
        "fights",
        rows_in,
        len(out),
        {
            "event_not_found": no_event,
            "unparseable_bout": bad_bout,
            "ambiguous_fighter_name": n_ambiguous,
            "self_fight": self_fight,
            "missing_or_dup_fight_id": rows_in
            - no_event
            - bad_bout
            - n_ambiguous
            - self_fight
            - len(out),
        },
    )
    print(
        f"      note: {n_synthetic:,} fighter slots used a name-derived id (absent from tale-of-the-tape)"
    )
    return out.reset_index(drop=True)


def build_fight_stats(raw_stats: pd.DataFrame, fights: pd.DataFrame, log: StageLog) -> pd.DataFrame:
    """fight_stats: one row per (fight_id, fighter_id, round) with parsed counts."""
    df = raw_stats.copy()
    rows_in = len(df)
    df["event_name"] = df["EVENT"].str.strip()
    df["bout"] = df["BOUT"].fillna("").str.strip()
    df["round"] = df["ROUND"].map(parse_round_label).astype("Int64")
    df["fighter_norm"] = df["FIGHTER"].map(normalize_name)

    # Join to fights via (event_name, normalized bout pair) so stats inherit fight_id.
    bout_parts = df["bout"].str.split(" vs. ", n=1, expand=True)
    df["pair_key"] = [
        "|".join(sorted([normalize_name(a), normalize_name(b)]))
        for a, b in zip(bout_parts[0], bout_parts[1].fillna(""), strict=False)
    ]

    pair_index = raw_stats_pair_index(fights)
    df["fight_id"] = [
        pair_index.get((ev, pk)) for ev, pk in zip(df["event_name"], df["pair_key"], strict=False)
    ]
    unmatched = int(df["fight_id"].isna().sum())
    df = df.loc[df["fight_id"].notna()].copy()

    for src, landed, att in (
        ("SIG.STR.", "sig_str_landed", "sig_str_att"),
        ("TOTAL STR.", "total_str_landed", "total_str_att"),
        ("TD", "td_landed", "td_att"),
        ("HEAD", "head_landed", "head_att"),
        ("BODY", "body_landed", "body_att"),
        ("LEG", "leg_landed", "leg_att"),
        ("DISTANCE", "distance_landed", "distance_att"),
        ("CLINCH", "clinch_landed", "clinch_att"),
        ("GROUND", "ground_landed", "ground_att"),
    ):
        pairs = df[src].map(parse_landed_attempted)
        df[landed] = [p[0] for p in pairs]
        df[att] = [p[1] for p in pairs]

    df["kd"] = pd.to_numeric(df["KD"], errors="coerce").astype("Int64")
    df["sub_att"] = pd.to_numeric(df["SUB.ATT"], errors="coerce").astype("Int64")
    df["rev"] = pd.to_numeric(df["REV."], errors="coerce").astype("Int64")
    df["ctrl_sec"] = df["CTRL"].map(parse_clock).astype("Int64")

    count_cols = [
        "kd",
        "sub_att",
        "rev",
        "ctrl_sec",
        "sig_str_landed",
        "sig_str_att",
        "total_str_landed",
        "total_str_att",
        "td_landed",
        "td_att",
        "head_landed",
        "head_att",
        "body_landed",
        "body_att",
        "leg_landed",
        "leg_att",
        "distance_landed",
        "distance_att",
        "clinch_landed",
        "clinch_att",
        "ground_landed",
        "ground_att",
    ]
    for c in count_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    out = df.loc[:, ["fight_id", "fighter_norm", "round", *count_cols]].copy()
    bad_round = int(out["round"].isna().sum())
    out = out.loc[out["round"].notna()].copy()

    log.record(
        "fight_stats",
        rows_in,
        len(out),
        {"bout_not_matched_to_fight": unmatched, "unparseable_round": bad_round},
    )
    return out.reset_index(drop=True)


def raw_stats_pair_index(fights: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Index (event_name, sorted normalized fighter-pair) -> fight_id.

    ``fight_stats`` identifies a bout only by event and bout string, so this is how its
    rows inherit the ``fight_id`` assigned in ``build_fights``. ``pair_key`` is built in
    ``build_fights`` from the same normalized names, so the two sides agree by
    construction.
    """
    if "pair_key" not in fights.columns:
        raise PipelineError("fights table is missing 'pair_key'; build_fights must add it")
    return {(row.event_name, row.pair_key): row.fight_id for row in fights.itertuples(index=False)}


def load_raw(snapshot_dir: Path, table: str) -> pd.DataFrame:
    """Read one raw Parquet table from a committed snapshot."""
    path = snapshot_dir / f"{table}.parquet"
    if not path.is_file():
        raise PipelineError(
            f"snapshot table not found: {path}\n"
            "The committed snapshot is required for the offline path. "
            "Run `uv run python -m ufc_predictor.ingest.refresh` to rebuild it."
        )
    return pd.read_parquet(path)
