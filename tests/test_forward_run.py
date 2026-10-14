"""The forward commands end to end, offline: predict, settle, and the run-due driver."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.experiments.runner import ExperimentSpec
from ufc_predictor.forward import bfo_live, espn, http, kalshi, run

FIXTURES = Path(__file__).parent / "fixtures"
UTC = dt.UTC
CARD_START = dt.datetime(2026, 10, 17, 21, tzinfo=UTC)  # Buckley vs. Malott, first bout


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(http.requests, "get", refuse)


class FixtureFetcher:
    def __init__(self, routes: dict[str, str]):
        self.routes = routes

    def text(self, url: str) -> str:
        if url not in self.routes:
            raise AssertionError(f"unexpected URL {url}")
        return (FIXTURES / self.routes[url]).read_text(encoding="utf-8")

    def json(self, url: str):
        return json.loads(self.text(url))


ROUTES = {
    espn.SCOREBOARD_URL: "espn/scoreboard_calendar.json",
    espn.SCOREBOARD_URL + "?dates=20261017-20261018": "espn/scoreboard_20261017-20261018.json",
    espn.SCOREBOARD_URL + "?dates=20261010-20261011": "espn/scoreboard_20261010-20261011.json",
    kalshi.events_url(cursor=None): "kalshi/events_open.json",
    bfo_live.HOME_URL: "bfo/homepage.html",
}


def _git(repo, *args, when=None):
    env = dict(os.environ)
    if when is not None:
        env |= {"GIT_AUTHOR_DATE": when.isoformat(), "GIT_COMMITTER_DATE": when.isoformat()}
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    return r


@pytest.fixture(scope="module")
def cfg():
    return get_config()


def test_predict_writes_one_row_per_bout_with_model_and_market_prices(cfg, root):
    now = CARD_START - dt.timedelta(hours=50)
    path = run.predict(
        "600060773", cfg, now=now, fetcher=FixtureFetcher(ROUTES), root=root, spec=ExperimentSpec()
    )
    preds = pd.read_csv(path)
    assert len(preds) == 13
    assert preds["p_model_a"].between(0.01, 0.99).all()
    main = preds.loc[preds["fighter_a"].isin(["Mike Malott", "Joaquin Buckley"])].iloc[0]
    assert {main["fighter_a"], main["fighter_b"]} == {"Mike Malott", "Joaquin Buckley"}
    # Kalshi and the sportsbook median are oriented to fighter_a and near each other.
    assert 0.3 < main["kalshi_p_a"] < 0.7 and abs(main["kalshi_p_a"] - main["bfo_p_a"]) < 0.08
    assert preds["kalshi_ticker_a"].notna().sum() == 13
    assert preds["bfo_p_a"].notna().sum() >= 12  # every bout BestFightOdds lists on both sides
    # Orientation: Kalshi and the sportsbooks are matched independently, so agreeing on
    # every bout means both are oriented to fighter_a (a swap would put them ~|1 - 2p| apart).
    both = preds.dropna(subset=["kalshi_p_a", "bfo_p_a"])
    assert len(both) >= 12 and (both["kalshi_p_a"] - both["bfo_p_a"]).abs().max() < 0.1
    entries = json.loads((root / cfg.forward["reports_dir"] / "ledger.json").read_text())
    assert [e["action"] for e in entries] == ["predict"]
    assert entries[0]["event_id"] == "600060773" and entries[0]["n_bouts"] == 13
    meta = json.loads((path.parent / "meta.json").read_text())
    assert meta["history_through"] <= "2026-10-17" and meta["spec"]["features"] == ["base"]


def test_predict_first_saves_results_of_every_finished_card_since_the_snapshot(cfg, root):
    """The Oct 10 card was never predicted, but its bouts are history for Oct 17."""
    now = CARD_START - dt.timedelta(hours=50)
    path = run.predict(
        "600060773", cfg, now=now, fetcher=FixtureFetcher(ROUTES), root=root, spec=ExperimentSpec()
    )
    saved = root / cfg.forward["results_dir"] / "600061541.json"
    assert saved.is_file()
    assert json.loads(saved.read_text())["date"] == "2026-10-10"
    meta = json.loads((path.parent / "meta.json").read_text())
    assert meta["history_through"] == "2026-10-10"
    assert meta["espn_results_files"] == ["600061541.json"]


def test_predict_debutants_are_flagged(cfg, root):
    now = CARD_START - dt.timedelta(hours=50)
    path = run.predict(
        "600060773", cfg, now=now, fetcher=FixtureFetcher(ROUTES), root=root, spec=ExperimentSpec()
    )
    preds = pd.read_csv(path)
    nallo = preds.loc[
        (preds["fighter_a"] == "Mandel Nallo") | (preds["fighter_b"] == "Mandel Nallo")
    ]
    assert len(nallo) == 1
    deb = preds.loc[
        (preds["fighter_a"] == "Javad Mahjoub") | (preds["fighter_b"] == "Javad Mahjoub")
    ].iloc[0]
    assert deb["debut_a"] or deb["debut_b"]


@pytest.mark.parametrize("hours_before", [23.0, -1.0])
def test_predict_refuses_inside_the_lead_time_or_after_the_start(cfg, root, hours_before):
    now = CARD_START - dt.timedelta(hours=hours_before)
    with pytest.raises(run.ForwardError, match="24"):
        run.predict(
            "600060773",
            cfg,
            now=now,
            fetcher=FixtureFetcher(ROUTES),
            root=root,
            spec=ExperimentSpec(),
        )


def test_predict_never_overwrites_committed_predictions(cfg, root):
    now = CARD_START - dt.timedelta(hours=50)
    run.predict(
        "600060773", cfg, now=now, fetcher=FixtureFetcher(ROUTES), root=root, spec=ExperimentSpec()
    )
    with pytest.raises(run.ForwardError, match="already"):
        run.predict(
            "600060773",
            cfg,
            now=now,
            fetcher=FixtureFetcher(ROUTES),
            root=root,
            spec=ExperimentSpec(),
        )


def _write_predictions(root, cfg, event_id, rows, committed_at):
    d = root / cfg.forward["reports_dir"] / event_id
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "predictions.csv", index=False)
    (root / cfg.forward["reports_dir"] / "ledger.json").write_text(
        json.dumps([{"event_id": event_id, "action": "predict"}])
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "predict", when=committed_at)


def test_settle_scores_committed_bouts_and_reads_the_kalshi_close(cfg, root):
    start = dt.datetime(2026, 10, 10, 21, tzinfo=UTC)
    main_start = dt.datetime(2026, 10, 11, 0, tzinfo=UTC)
    rows = [
        {
            "competition_id": "401916276",
            "fighter_a": "Brendan Allen",
            "fighter_b": "Christian Leroy Duncan",
            "segment_start_utc": main_start.isoformat(),
            "p_model_a": 0.6,
            "kalshi_p_a": 0.52,
            "kalshi_ticker_a": "KXUFCFIGHT-26OCT10ALLDUN-ALL",
            "kalshi_ticker_b": "KXUFCFIGHT-26OCT10ALLDUN-DUN",
        },
        {
            "competition_id": "401924511",
            "fighter_a": "Melissa Gatto",
            "fighter_b": "Ernesta Kareckaite",
            "segment_start_utc": start.isoformat(),
            "p_model_a": 0.5,
            "kalshi_p_a": float("nan"),
            "kalshi_ticker_a": None,
            "kalshi_ticker_b": None,
        },
    ]
    _write_predictions(root, cfg, "600061541", rows, committed_at=start - dt.timedelta(hours=30))
    lookback = dt.timedelta(minutes=int(cfg.forward["close_lookback_minutes"]))
    routes = dict(ROUTES)
    for t in ("KXUFCFIGHT-26OCT10ALLDUN-ALL", "KXUFCFIGHT-26OCT10ALLDUN-DUN"):
        routes[kalshi.candlesticks_url(t, main_start - lookback, main_start, period_minutes=1)] = (
            "kalshi/candlesticks_ALLDUN-ALL.json"
        )
    out = run.settle(
        "600061541",
        cfg,
        now=start + dt.timedelta(hours=12),
        fetcher=FixtureFetcher(routes),
        root=root,
    )
    assert out["commitment"]["verified"] is True
    by = {b["competition_id"]: b for b in out["bouts"]}
    assert by["401916276"]["status"] == "scored" and by["401916276"]["y_a"] == 1.0
    assert by["401924511"]["status"] == "void_draw_or_nc"
    assert out["summary"]["n_scored"] == 1
    saved = root / cfg.forward["results_dir"] / "600061541.json"
    assert saved.is_file()  # results now feed later cards' history


def test_settle_flags_predictions_committed_too_late(cfg, root):
    start = dt.datetime(2026, 10, 10, 21, tzinfo=UTC)
    rows = [
        {
            "competition_id": "401924511",
            "fighter_a": "Melissa Gatto",
            "fighter_b": "Ernesta Kareckaite",
            "segment_start_utc": start.isoformat(),
            "p_model_a": 0.5,
            "kalshi_p_a": float("nan"),
            "kalshi_ticker_a": None,
            "kalshi_ticker_b": None,
        }
    ]
    _write_predictions(root, cfg, "600061541", rows, committed_at=start - dt.timedelta(hours=2))
    out = run.settle(
        "600061541",
        cfg,
        now=start + dt.timedelta(hours=12),
        fetcher=FixtureFetcher(ROUTES),
        root=root,
    )
    assert out["commitment"]["verified"] is False


def test_run_due_predicts_inside_the_window(cfg, root, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "predict", lambda ev, cfg, **kw: calls.append(("predict", ev)) or root)
    monkeypatch.setattr(
        run, "frozen_commit_time", lambda root: dt.datetime(2026, 10, 14, tzinfo=UTC)
    )
    cfg = dataclasses.replace(cfg, forward={**cfg.forward, "horizon_days": 4, "history_days": 3})
    now = CARD_START - dt.timedelta(hours=50)
    report = run.run_due(cfg, now=now, fetcher=FixtureFetcher(ROUTES), root=root)
    assert ("predict", "600060773") in calls
    assert report["missed"] == []
