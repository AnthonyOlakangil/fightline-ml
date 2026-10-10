"""Tests for ingest: parsing, vig removal, name matching, and the join contract.

These run against the committed snapshot, so they are offline and deterministic.
"""

from __future__ import annotations

import datetime as dt
import json
import math

import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.ingest.names import BoutIndex, normalize_name, similarity
from ufc_predictor.ingest.normalize import (
    METHOD_DEC,
    METHOD_KO,
    METHOD_OTHER,
    METHOD_SUB,
    classify_method,
    parse_clock,
    parse_dob,
    parse_event_date,
    parse_height_cm,
    parse_landed_attempted,
    parse_reach_cm,
    parse_round_label,
)
from ufc_predictor.ingest.odds import american_to_decimal, american_to_implied, remove_vig
from ufc_predictor.ingest.stages import PipelineError, StageLog
from ufc_predictor.storage import db

# Row counts recorded in the snapshot's SOURCE.md. A refreshed snapshot that silently
# changes shape fails here rather than shifting downstream results unnoticed.
EXPECTED_RAW_ROWS = {
    "events": 791,
    "fight_results": 8950,
    "fight_stats": 42062,
    "fighter_tott": 4629,
}
EXPECTED_ODDS_RAW_ROWS = 7177


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_parse_event_date():
    assert parse_event_date("October 03, 2026") == dt.date(2026, 10, 3)


def test_parse_event_date_raises_on_garbage():
    """Fail loudly: an unparseable date must not become NaT."""
    with pytest.raises(PipelineError, match="cannot parse event date"):
        parse_event_date("not a date")


def test_parse_event_date_raises_on_empty():
    with pytest.raises(PipelineError, match="empty event date"):
        parse_event_date("")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("5' 11\"", 180.34), ("6'", 182.88), ("--", None), ("", None), (None, None)],
)
def test_parse_height(value, expected):
    assert parse_height_cm(value) == expected


@pytest.mark.parametrize(("value", "expected"), [('76"', 193.04), ("--", None), (None, None)])
def test_parse_reach(value, expected):
    assert parse_reach_cm(value) == expected


def test_parse_dob():
    assert parse_dob("Jul 13, 1978") == dt.date(1978, 7, 13)
    assert parse_dob("--") is None


@pytest.mark.parametrize(
    ("value", "expected"), [("1:28", 88), ("4:31", 271), ("0:00", 0), ("--", None), (None, None)]
)
def test_parse_clock(value, expected):
    assert parse_clock(value) == expected


def test_parse_landed_attempted():
    assert parse_landed_attempted("16 of 40") == (16, 40)
    assert parse_landed_attempted("--") == (None, None)


def test_parse_round_label():
    assert parse_round_label("Round 1") == 1
    assert parse_round_label("Round 5") == 5
    assert parse_round_label("") is None


def test_parsers_are_nan_safe():
    """Parquet round-trips missing cells as float NaN, which is truthy."""
    nan = float("nan")
    assert parse_round_label(nan) is None
    assert parse_clock(nan) is None
    assert parse_height_cm(nan) is None
    assert parse_dob(nan) is None
    assert classify_method(nan) == METHOD_OTHER
    assert normalize_name(nan) == ""


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("Decision - Unanimous", METHOD_DEC),
        ("Decision - Split", METHOD_DEC),
        ("KO/TKO", METHOD_KO),
        ("Submission", METHOD_SUB),
        ("Overturned", METHOD_OTHER),
        ("DQ", METHOD_OTHER),
        ("", METHOD_OTHER),
    ],
)
def test_classify_method(method, expected):
    assert classify_method(method) == expected


# ---------------------------------------------------------------------------
# Odds maths
# ---------------------------------------------------------------------------


def test_american_to_decimal_hand_computed():
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(-130) == pytest.approx(1 + 100 / 130)
    assert american_to_decimal(124) == pytest.approx(2.24)
    assert american_to_decimal(-2100) == pytest.approx(1 + 100 / 2100)


def test_american_to_decimal_rejects_zero():
    with pytest.raises(ValueError, match="invalid American odds"):
        american_to_decimal(0)


def test_remove_vig_sums_to_one():
    p1, p2 = remove_vig(american_to_implied(-130), american_to_implied(102))
    assert p1 + p2 == pytest.approx(1.0, abs=1e-12)
    assert p1 > p2, "the -130 side must be the favourite"


def test_remove_vig_hand_computed():
    """A known pair, computed by hand.

    -130 -> decimal 1.76923 -> implied 0.56522
    +102 -> decimal 2.02000 -> implied 0.49505
    total 1.06027 (6.03% overround); normalized: 0.53309 / 0.46691
    """
    p1, p2 = remove_vig(american_to_implied(-130), american_to_implied(102))
    assert p1 == pytest.approx(0.53309, abs=1e-5)
    assert p2 == pytest.approx(0.46691, abs=1e-5)


def test_remove_vig_rejects_degenerate_input():
    with pytest.raises(ValueError, match="cannot de-vig"):
        remove_vig(0.0, 0.0)


# ---------------------------------------------------------------------------
# Name normalization and matching
# ---------------------------------------------------------------------------


def test_normalize_name_strips_accents_punctuation_and_suffixes():
    assert normalize_name("José Aldo Jr.") == "jose aldo"
    assert normalize_name("Waldo Cortes-Acosta") == "waldo cortes acosta"
    assert normalize_name("  Jon   Jones  ") == "jon jones"
    assert normalize_name("Michael Aswell Jr.") == "michael aswell"


def test_similarity_identical_token_sets_score_one():
    """The two sources transpose many Chinese and Mongolian names."""
    assert similarity("zhang weili", "weili zhang") == 1.0
    assert similarity("batgerel danaa", "danaa batgerel") == 1.0


def test_similarity_distinguishes_different_people_sharing_a_surname():
    """Token overlap alone would score these 0.5; they must stay well below threshold."""
    assert similarity("anderson silva", "thiago silva") < 0.85
    assert similarity("mike davis", "marcus davis") < 0.85


def test_bout_index_exact_match():
    idx = BoutIndex([("f1", dt.date(2024, 1, 1), "Jon Jones", "Stipe Miocic")])
    r = idx.match(dt.date(2024, 1, 1), "Jon Jones", "Stipe Miocic")
    assert r.fight_id == "f1"
    assert r.method == "exact"


def test_bout_index_refuses_match_on_a_different_date():
    idx = BoutIndex([("f1", dt.date(2024, 1, 1), "Jon Jones", "Stipe Miocic")])
    r = idx.match(dt.date(2024, 6, 1), "Jon Jones", "Stipe Miocic")
    assert not r.is_match


def test_bout_index_applies_aliases():
    idx = BoutIndex([("f1", dt.date(2024, 1, 1), "Tecia Pennington", "Mackenzie Dern")])
    r = idx.match(
        dt.date(2024, 1, 1),
        "Tecia Torres",
        "Mackenzie Dern",
        aliases={"tecia torres": "tecia pennington"},
    )
    assert r.fight_id == "f1"
    assert r.method == "alias"


def test_bout_index_rejects_ambiguous_fuzzy_match():
    """Two equally plausible bouts on one card must not be guessed between.

    A wrong pairing attaches one fighter's closing line to another's result and would
    not change any row count, so ambiguity must drop the row. Both candidates here are
    token reorderings of the query, so both score 1.0 and the margin is zero.
    """
    idx = BoutIndex(
        [
            ("f1", dt.date(2024, 1, 1), "Antonio Rodrigo Nogueira", "Stefan Struve"),
            ("f2", dt.date(2024, 1, 1), "Rodrigo Antonio Nogueira", "Stefan Struve"),
        ]
    )
    r = idx.match(dt.date(2024, 1, 1), "Nogueira Antonio Rodrigo", "Stefan Struve")
    assert not r.is_match
    assert r.method == "unmatched_ambiguous"
    assert r.score == pytest.approx(1.0)
    assert r.runner_up == pytest.approx(1.0)


def test_bout_index_rejects_weak_fuzzy_match():
    idx = BoutIndex([("f1", dt.date(2024, 1, 1), "Jon Jones", "Stipe Miocic")])
    r = idx.match(dt.date(2024, 1, 1), "Conor McGregor", "Nate Diaz")
    assert not r.is_match
    assert r.method == "unmatched_no_candidate"


def test_bout_index_rejects_a_single_character_surname_difference():
    """The fuzzy tier is a deliberately near-inert backstop.

    A one-character surname difference ("cerqueira" / "cerquiera") scores about 0.74,
    below the 0.92 floor, so it is rejected here and must instead be resolved by a
    corroborated entry in the alias table. That ordering is intentional: an alias backed
    by several independent fights is far safer than a similarity score.
    """
    idx = BoutIndex([("f1", dt.date(2024, 1, 1), "Rafael Cerquiera", "Ibo Aslan")])
    r = idx.match(dt.date(2024, 1, 1), "Rafael Cerqueira", "Ibo Aslan")
    assert not r.is_match, "a surname typo must not be fuzzy-matched without corroboration"


# ---------------------------------------------------------------------------
# StageLog
# ---------------------------------------------------------------------------


def test_stage_log_raises_on_unaccounted_rows():
    """A join that loses rows without explaining them must raise."""
    log = StageLog(verbose=False)
    with pytest.raises(PipelineError, match="unaccounted rows"):
        log.record("lossy", rows_in=100, rows_out=90)


def test_stage_log_accepts_explained_drops():
    log = StageLog(verbose=False)
    stage = log.record("explained", rows_in=100, rows_out=90, drops={"bad_date": 10})
    assert stage.dropped == 10


# ---------------------------------------------------------------------------
# Snapshot and processed-table contracts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.mark.parametrize(("table", "expected"), sorted(EXPECTED_RAW_ROWS.items()))
def test_snapshot_row_counts_match_source_md(cfg, table, expected):
    df = pd.read_parquet(cfg.paths.ufcstats_snapshot / f"{table}.parquet")
    assert len(df) == expected, (
        f"{table} has {len(df)} rows, expected {expected}. If the snapshot was "
        "refreshed on purpose, update EXPECTED_RAW_ROWS and SOURCE.md together."
    )


def test_odds_snapshot_row_count(cfg):
    df = pd.read_parquet(cfg.paths.odds_snapshot / "odds_master.parquet")
    assert len(df) == EXPECTED_ODDS_RAW_ROWS


def test_odds_snapshot_contains_only_whitelisted_columns(cfg):
    """The upstream CSV ships 118 columns; 113 are precomputed career aggregates.

    Those cannot be shown to be point-in-time correct, so they must never enter the
    repository at all -- not merely be filtered later.
    """
    df = pd.read_parquet(cfg.paths.odds_snapshot / "odds_master.parquet")
    assert set(df.columns) == {"R_fighter", "B_fighter", "date", "R_odds", "B_odds"}


def test_source_md_exists_and_records_provenance(cfg):
    for snap in (cfg.paths.ufcstats_snapshot, cfg.paths.odds_snapshot):
        text = (snap / "SOURCE.md").read_text(encoding="utf-8")
        assert "Retrieved (UTC)" in text
        assert "SHA-256" in text
        assert "license" in text.lower()


@pytest.fixture(scope="module")
def tables(cfg):
    """Processed tables; skip if `make data` has not been run."""
    try:
        return {name: db.read_table(cfg.paths.processed_dir, name) for name in db.TABLES}
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data`")


def test_fights_have_unique_ids(tables):
    fights = tables["fights"]
    assert not fights["fight_id"].duplicated().any()


def test_fights_never_pair_a_fighter_with_themselves(tables):
    fights = tables["fights"]
    assert (fights["fighter_1"] != fights["fighter_2"]).all()


def test_fights_are_in_canonical_orientation(tables):
    """fighter_1 must be the lexicographically smaller id, independent of the result."""
    fights = tables["fights"]
    assert (fights["fighter_1"].astype(str) < fights["fighter_2"].astype(str)).all()


def test_every_date_parsed(tables):
    for name in ("events", "fights"):
        assert tables[name]["date"].notna().all(), f"{name} has an unparsed date"


def test_odds_probabilities_are_a_valid_distribution(tables):
    odds = tables["odds"]
    total = odds["p1_market"] + odds["p2_market"]
    assert (total - 1.0).abs().max() < 1e-9
    assert odds["p1_market"].between(0, 1).all()


def test_odds_overround_is_positive_and_plausible(tables):
    """A two-sided book must price above 100%.

    Impossible pairs (both sides plus-money, e.g. +176/+275) and implausible vig are
    dropped during loading, so every surviving row must sit inside the configured
    bounds. The median should land near real UFC moneyline vig, a few percent.
    """
    cfg = get_config()
    lo = float(cfg.ingest["min_overround_exclusive"])
    hi = float(cfg.ingest["max_overround"])
    odds = tables["odds"]
    assert (odds["overround"] > lo).all(), "an arithmetically impossible market survived"
    assert (odds["overround"] <= hi).all()
    assert 0.01 < odds["overround"].median() < 0.10


def test_every_odds_row_links_to_a_real_fight(tables):
    odds, fights = tables["odds"], tables["fights"]
    assert odds["fight_id"].isin(set(fights["fight_id"])).all()
    assert not odds["fight_id"].duplicated().any()


def test_fight_stats_link_to_real_fights(tables):
    stats, fights = tables["fight_stats"], tables["fights"]
    assert stats["fight_id"].isin(set(fights["fight_id"])).all()
    assert stats["round"].between(1, 5).all()


def test_market_baseline_is_sane(tables):
    """Guards against a mispaired join, which no row count would reveal.

    If closing lines were attached to the wrong fighters, favourite accuracy would
    collapse toward 50%. The published figure for UFC closing lines is around 66%.
    """
    odds, fights = tables["odds"], tables["fights"]
    # odds already carries canonical fighter_1/fighter_2, so only the outcome is joined.
    merged = odds.merge(
        fights.loc[:, ["fight_id", "winner_id", "is_decisive"]],
        on="fight_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.loc[merged["is_decisive"]]
    y = (merged["winner_id"] == merged["fighter_1"]).astype(float)
    p = merged["p1_market"].clip(1e-12, 1 - 1e-12)

    accuracy = ((p > 0.5) == (y > 0.5)).mean()
    log_loss = -(y * p.map(math.log) + (1 - y) * (1 - p).map(math.log)).mean()

    assert 0.62 <= accuracy <= 0.70, f"favourite accuracy {accuracy:.3f} implies a mispaired join"
    assert log_loss < math.log(2), f"market log loss {log_loss:.4f} is worse than a coin flip"


def test_ingest_log_was_written(cfg):
    path = cfg.paths.reports_dir / "ingest_log.json"
    if not path.is_file():
        pytest.skip("ingest_log.json absent; run `make data`")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stages"], "no stages recorded"
    assert payload["summary"]["fights_priced"] > 6000


def test_duckdb_views_are_queryable(cfg, tables):
    con = db.connect(cfg.paths.processed_dir)
    try:
        n = con.sql("SELECT count(*) FROM fights WHERE is_decisive").fetchone()[0]
        assert n > 8000
        joined = con.sql("SELECT count(*) FROM odds o JOIN fights f USING (fight_id)").fetchone()[0]
        assert joined == len(tables["odds"])
    finally:
        con.close()
