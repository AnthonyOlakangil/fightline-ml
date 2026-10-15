"""The forward test's commands: predict a card before it starts, settle it afterwards.

    uv run ufc-predictor forward predict --event <espn id>
    uv run ufc-predictor forward close   --event <espn id>
    uv run ufc-predictor forward settle  --event <espn id>
    uv run ufc-predictor forward run-due        # what the GitHub Actions schedule runs

**predict** trains the frozen development configuration (``reports/dev/frozen.yaml``)
on every decisive bout known before the card -- the UFCStats snapshot plus ESPN results
since -- and writes ``reports/forward/<event>/predictions.csv`` with the model's
probability for each bout, Kalshi's mid at prediction time and the sportsbook median.
It refuses a card that starts within ``forward.min_lead_hours`` or has started, and never
overwrites predictions.

**settle** reads ESPN's results, scores only bouts held between the committed pair,
reads Kalshi's closing mid at each bout's segment start from public candlesticks, checks
that the predictions file was committed at least ``min_lead_hours`` before the card and
never changed, and writes ``settlement.json``. The results are saved under
``data/raw/espn_results/`` so later cards learn from them.

Research only: nothing here is betting advice, and no stake is ever computed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ufc_predictor.config import REPO_ROOT, Config
from ufc_predictor.experiments.runner import (
    ExperimentSpec,
    family_kwargs,
    make_model,
    training_pool,
)
from ufc_predictor.features.pit import build_features
from ufc_predictor.forward import bfo_live, cards, espn, kalshi, ledger, schedule
from ufc_predictor.forward import settle as settle_mod
from ufc_predictor.forward.http import Fetcher
from ufc_predictor.ingest.names import normalize_name
from ufc_predictor.provenance import OUTPUT_DIRS, git_state
from ufc_predictor.storage import db

FROZEN_RELPATH = "reports/dev/frozen.yaml"


class ForwardError(RuntimeError):
    """A forward-test step refused to run or failed."""


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def make_fetcher(cfg: Config) -> Fetcher:
    return Fetcher(
        REPO_ROOT / cfg.forward["cache_dir"],
        str(cfg.bfo["user_agent"]),
        delay_s=float(cfg.forward["delay_s"]),
    )


def _dirs(cfg: Config, root: Path) -> tuple[Path, Path]:
    return root / cfg.forward["reports_dir"], root / cfg.forward["results_dir"]


def _entry(fetcher, event_id: str) -> espn.CalendarEntry:
    for e in espn.calendar(fetcher):
        if e.event_id == event_id:
            return e
    raise ForwardError(f"event {event_id} is not on ESPN's UFC calendar")


def _actions_run() -> dict:
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id:
        return {}
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    return {"actions_run_id": run_id, "actions_run_url": f"{server}/{repo}/actions/runs/{run_id}"}


def frozen_spec(cfg: Config) -> ExperimentSpec:
    from ufc_predictor.experiments.stages import load_spec

    path = REPO_ROOT / FROZEN_RELPATH
    if not path.is_file():
        raise ForwardError(
            f"no frozen configuration at {FROZEN_RELPATH}; run the experiments first"
        )
    return load_spec(yaml.safe_load(path.read_text(encoding="utf-8"))["spec"])


def frozen_commit_time(root: Path = REPO_ROOT) -> dt.datetime | None:
    """When the frozen configuration was first committed; cards before then are out of scope."""
    return ledger.first_commit_time(root, FROZEN_RELPATH)


def _same_person(a: str, b: str) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    if na == nb or cards.near_ratio(na, nb) >= cards.NEAR_MATCH:
        return True
    ta, tb = na.split(), nb.split()
    return ta[-1] == tb[-1] and ta[0][0] == tb[0][0]


def _match_pair(a: str, b: str, candidates: list, names) -> tuple[object, bool] | None:
    """The single candidate whose two names are (a, b) in some order, and whether swapped."""
    hits = []
    for c in candidates:
        x, y = names(c)
        if _same_person(a, x) and _same_person(b, y):
            hits.append((c, False))
        elif _same_person(a, y) and _same_person(b, x):
            hits.append((c, True))
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def sync_results(fetcher, cfg: Config, *, before: dt.date, root: Path = REPO_ROOT) -> list[str]:
    """Save ESPN results for every finished UFC card after the snapshot and before ``before``.

    A card the forward test never predicted is still history for later cards, so it must
    not be skipped. A card whose bouts are not all final (still under way, or ESPN has not
    posted results) raises rather than entering history half-known.

    Returns the event ids saved by this call.
    """
    _, results_dir = _dirs(cfg, root)
    snapshot_end = db.read_table(cfg.paths.processed_dir, "fights")["date"].max()
    saved = []
    for entry in espn.calendar(fetcher):
        path = results_dir / f"{entry.event_id}.json"
        # The calendar's UTC start can fall a day after the card's US date.
        if path.exists() or entry.start_utc.date() <= snapshot_end:
            continue
        if entry.start_utc.date() - dt.timedelta(days=1) >= before:
            continue
        card = espn.card(fetcher, entry)
        if not snapshot_end < card.date < before:
            continue
        pending = [b for b in card.bouts if b.status not in ("STATUS_FINAL", "STATUS_CANCELED")]
        if pending:
            raise ForwardError(
                f"{card.name} ({card.date}) is history for a later card but has "
                f"{len(pending)} bouts not final on ESPN"
            )
        cards.save_results(path, card, espn.results(fetcher, entry))
        saved.append(entry.event_id)
    return saved


def _model_probabilities(
    card: espn.Card, spec: ExperimentSpec, cfg: Config, results_dir: Path
) -> tuple[pd.DataFrame, dict]:
    fights = db.read_table(cfg.paths.processed_dir, "fights")
    stats = db.read_table(cfg.paths.processed_dir, "fight_stats")
    fighters = db.read_table(cfg.paths.processed_dir, "fighters")
    aliases = cards.load_aliases()
    history, index = cards.forward_history(fights, results_dir, aliases, before=card.date)
    rows = cards.card_rows(card, index, aliases)
    feats = build_features(
        pd.concat([history, rows], ignore_index=True),
        stats,
        fighters,
        rolling_windows=tuple(cfg.features["rolling_windows"]),
        elo_params=cfg.elo,
        families=spec.features,
        **family_kwargs(cfg),
    )
    labelled = feats.loc[feats["label"].notna()]
    train, floored = training_pool(labelled, card.date, spec, int(cfg.backtest["min_train_fights"]))
    columns = spec.columns()
    if spec.top_k is not None:
        from ufc_predictor.experiments.selection import select_top_k

        columns = select_top_k(train, columns, spec.top_k, cfg, params=spec.params(cfg))
    model = make_model(spec, cfg, columns).fit(train, train["label"].astype(int))
    target = feats.set_index("fight_id").loc[rows["fight_id"]].reset_index()
    target["p1"] = model.predict_proba(target)
    target = target.merge(
        rows[["fight_id", "name_1", "name_2"]], on="fight_id", validate="one_to_one"
    )
    info = {
        "history_through": str(history["date"].max()),
        "n_training_fights": len(train),
        "training_floor_extended": bool(floored),
        "n_features": len(columns),
        "espn_results_files": sorted(p.name for p in Path(results_dir).glob("*.json")),
    }
    return target, info


def _kalshi_rows(card: espn.Card, fetcher) -> tuple[dict, list[dict]]:
    fights = kalshi.fight_markets(fetcher, card.date)
    out = {}
    for b in card.bouts:
        hit = _match_pair(b.fighter_a, b.fighter_b, fights, lambda f: f.names)
        if hit is None:
            continue
        f, swapped = hit
        side_a, side_b = (f.sides[1], f.sides[0]) if swapped else f.sides
        name_a = f.names[1] if swapped else f.names[0]
        mids = kalshi.mid_probabilities(f)
        out[b.competition_id] = {
            "kalshi_event": f.event_ticker,
            "kalshi_ticker_a": side_a.ticker,
            "kalshi_ticker_b": side_b.ticker,
            "kalshi_bid_a": side_a.bid,
            "kalshi_ask_a": side_a.ask,
            "kalshi_bid_b": side_b.bid,
            "kalshi_ask_b": side_b.ask,
            "kalshi_volume": f.volume,
            "kalshi_p_a": mids[name_a] if mids else np.nan,
        }
    raw = [f._asdict() for f in fights]
    return out, raw


def _bfo_rows(card: espn.Card, fetcher) -> tuple[dict, str]:
    try:
        odds = bfo_live.current_odds(fetcher)
    except Exception as exc:  # a cross-check only: record why it is missing, loudly
        return {}, f"unavailable: {type(exc).__name__}: {exc}"
    out = {}
    for b in card.bouts:
        # BestFightOdds lists each bout from both sides; each side must match exactly once.
        mine = odds.loc[
            [
                _same_person(b.fighter_a, f) and _same_person(b.fighter_b, o)
                for f, o in zip(odds["fighter"], odds["opponent"], strict=True)
            ]
        ]
        theirs = odds.loc[
            [
                _same_person(b.fighter_b, f) and _same_person(b.fighter_a, o)
                for f, o in zip(odds["fighter"], odds["opponent"], strict=True)
            ]
        ]
        if len(mine) != 1 or len(theirs) != 1:
            continue
        pa, pb = mine["median_p"].iloc[0], theirs["median_p"].iloc[0]
        if pd.notna(pa) and pd.notna(pb):
            out[b.competition_id] = {
                "bfo_p_a": float(pa / (pa + pb)),
                "bfo_n_books": int(min(mine["n_books"].iloc[0], theirs["n_books"].iloc[0])),
            }
    return out, "ok"


def predict(
    event_id: str,
    cfg: Config,
    *,
    now: dt.datetime,
    fetcher,
    root: Path = REPO_ROOT,
    spec: ExperimentSpec | None = None,
) -> Path:
    """Write the card's predictions. Returns the predictions file path."""
    reports_dir, results_dir = _dirs(cfg, root)
    card = espn.card(fetcher, _entry(fetcher, event_id))
    lead_h = float(cfg.forward["min_lead_hours"])
    if card.start_utc - now < dt.timedelta(hours=lead_h):
        raise ForwardError(
            f"{card.name} starts {card.start_utc:%Y-%m-%d %H:%M} UTC, less than {lead_h:g} hours "
            "from now; predictions are only made earlier than that"
        )
    out_dir = reports_dir / event_id
    path = out_dir / "predictions.csv"
    if path.exists():
        raise ForwardError(f"predictions for {event_id} already exist; they are never overwritten")
    spec = spec or frozen_spec(cfg)

    code = git_state(ignore=OUTPUT_DIRS)  # before sync_results writes new results files
    sync_results(fetcher, cfg, before=card.date, root=root)
    target, info = _model_probabilities(card, spec, cfg, results_dir)
    kalshi_by_bout, kalshi_raw = _kalshi_rows(card, fetcher)
    bfo_by_bout, bfo_status = _bfo_rows(card, fetcher)

    by_fight = target.set_index("fight_id")
    rows = []
    for b in card.bouts:
        t = by_fight.loc[f"espn-{b.competition_id}"]
        a_is_1 = normalize_name(b.fighter_a) == t["name_1"]
        rows.append(
            {
                "event_id": event_id,
                "event_name": card.name,
                "card_date": card.date.isoformat(),
                "competition_id": b.competition_id,
                "segment_start_utc": b.start_utc.isoformat(),
                "fighter_a": b.fighter_a,
                "fighter_b": b.fighter_b,
                "fighter_a_id": t["fighter_1"] if a_is_1 else t["fighter_2"],
                "fighter_b_id": t["fighter_2"] if a_is_1 else t["fighter_1"],
                "debut_a": bool(t["is_debut_1"] if a_is_1 else t["is_debut_2"]),
                "debut_b": bool(t["is_debut_2"] if a_is_1 else t["is_debut_1"]),
                "weight_class": b.weight_class,
                "scheduled_rounds": b.scheduled_rounds,
                "p_model_a": float(t["p1"] if a_is_1 else 1.0 - t["p1"]),
                **kalshi_by_bout.get(b.competition_id, {}),
                **bfo_by_bout.get(b.competition_id, {}),
            }
        )
    frame = pd.DataFrame(rows)
    for col in ("kalshi_ticker_a", "kalshi_ticker_b", "kalshi_p_a", "bfo_p_a", "bfo_n_books"):
        if col not in frame.columns:
            frame[col] = np.nan
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

    meta = {
        "event_id": event_id,
        "event_name": card.name,
        "start_utc": card.start_utc.isoformat(),
        "predicted_at_utc": now.isoformat(),
        "code_commit": code["commit"],
        "code_tree_clean": code["clean"],
        "spec": spec.to_dict(),
        "spec_sha256": hashlib.sha256(spec.key().encode()).hexdigest(),
        "kalshi_matched_bouts": len(kalshi_by_bout),
        "bfo_status": bfo_status,
        "bfo_matched_bouts": len(bfo_by_bout),
        **info,
        **_actions_run(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "kalshi_at_prediction.json").write_text(
        json.dumps(kalshi_raw, indent=1) + "\n", encoding="utf-8"
    )
    ledger.append(
        reports_dir / "ledger.json",
        {
            "event_id": event_id,
            "event_name": card.name,
            "action": "predict",
            "at_utc": now.isoformat(),
            "start_utc": card.start_utc.isoformat(),
            "n_bouts": len(frame),
            **_actions_run(),
        },
    )
    return path


# ---------------------------------------------------------------------------
# close (fallback price snapshot) and settle
# ---------------------------------------------------------------------------


def close(event_id: str, cfg: Config, *, now: dt.datetime, fetcher, root: Path = REPO_ROOT) -> Path:
    reports_dir, _ = _dirs(cfg, root)
    card = espn.card(fetcher, _entry(fetcher, event_id))
    if now >= card.start_utc:
        raise ForwardError(f"{card.name} has started; a pre-start snapshot is no longer possible")
    kalshi_by_bout, _ = _kalshi_rows(card, fetcher)
    bfo_by_bout, bfo_status = _bfo_rows(card, fetcher)
    path = reports_dir / event_id / "prices_close.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "taken_at_utc": now.isoformat(),
                "kalshi": kalshi_by_bout,
                "bfo": bfo_by_bout,
                "bfo_status": bfo_status,
            },
            indent=1,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger.append(
        reports_dir / "ledger.json",
        {"event_id": event_id, "action": "close", "at_utc": now.isoformat(), **_actions_run()},
    )
    return path


def _actions_run_check(
    fetcher, cfg: Config, meta: dict, deadline: dt.datetime, root: Path, rel: str
) -> dict:
    """Server-side evidence that the predictions came from this repository's scheduled job.

    A commit's date is whatever its committer set; a GitHub Actions run's start time is
    recorded by GitHub. The run named in ``meta.json`` must belong to this repository, be
    a scheduled or manually dispatched run, have started before the deadline, and be the
    run named in the message of the commit that added the predictions.
    """
    run_id, url = meta.get("actions_run_id"), meta.get("actions_run_url")
    if not run_id or not url:
        return {"verified": False, "reason": "no GitHub Actions run recorded for the predictions"}
    repo = str(cfg.forward["github_repository"])
    token = os.environ.get("GITHUB_TOKEN")
    data = fetcher.json(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}",
        headers={"Authorization": f"Bearer {token}"} if token else None,
    )
    started = dt.datetime.fromisoformat(data["run_started_at"].replace("Z", "+00:00"))
    message = ledger.adding_commit_message(root, rel)
    problems = []
    if data.get("repository", {}).get("full_name") != repo:
        problems.append(f"run belongs to {data.get('repository', {}).get('full_name')}, not {repo}")
    if data.get("event") not in ("schedule", "workflow_dispatch"):
        problems.append(f"run was triggered by {data.get('event')}")
    if started > deadline:
        problems.append(
            f"run started {started.isoformat()}, after the deadline {deadline.isoformat()}"
        )
    if url not in (message or ""):
        problems.append("the commit that added the predictions does not name this run")
    return {
        "verified": not problems,
        "reason": "; ".join(problems) if problems else "ok",
        "run_id": str(run_id),
        "run_url": url,
        "run_started_at": started.isoformat(),
        "event": data.get("event"),
    }


def _quotes(preds: pd.DataFrame, cfg: Config, fetcher, snapshot: dict) -> tuple[dict, dict]:
    lookback = dt.timedelta(minutes=int(cfg.forward["close_lookback_minutes"]))
    quotes: dict = {}
    source: dict = {}
    for row in preds.itertuples(index=False):
        at = dt.datetime.fromisoformat(row.segment_start_utc)
        for side, ticker in (("a", row.kalshi_ticker_a), ("b", row.kalshi_ticker_b)):
            if not isinstance(ticker, str):
                continue
            q = None
            try:
                q = kalshi.quote_at(fetcher, ticker, at, lookback=lookback, period_minutes=1)
                source[ticker] = "candlesticks"
            except Exception as exc:  # fall back to the pre-start snapshot, and say so
                source[ticker] = f"candlesticks failed ({type(exc).__name__}); snapshot"
            if q is None:
                snap = snapshot.get(row.competition_id, {})
                bid, ask = snap.get(f"kalshi_bid_{side}"), snap.get(f"kalshi_ask_{side}")
                if bid is not None and ask is not None and 0 < bid <= ask < 1:
                    q = (bid, ask)
                    source[ticker] = source.get(ticker, "") + " -> pre-start snapshot"
            quotes[ticker] = q
    return quotes, source


def settle(
    event_id: str, cfg: Config, *, now: dt.datetime, fetcher, root: Path = REPO_ROOT
) -> dict:
    reports_dir, results_dir = _dirs(cfg, root)
    entry = _entry(fetcher, event_id)
    card = espn.card(fetcher, entry)
    results = espn.results(fetcher, entry)
    path = reports_dir / event_id / "predictions.csv"
    if not path.is_file():
        raise ForwardError(f"no predictions for {event_id} to settle")
    preds = pd.read_csv(path, dtype={"competition_id": str})

    rel = str(path.relative_to(root))
    deadline = card.start_utc - dt.timedelta(hours=float(cfg.forward["min_lead_hours"]))
    added = ledger.first_commit_time(root, rel)
    commitment = {
        "predictions_file": rel,
        "first_committed_utc": added.isoformat() if added else None,
        "deadline_utc": deadline.isoformat(),
        "committed_in_time": ledger.committed_before(root, rel, deadline),
        "unchanged_since": ledger.unchanged_since_added(root, rel) if added else False,
    }
    meta_path = reports_dir / event_id / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    commitment["actions_run"] = _actions_run_check(fetcher, cfg, meta, deadline, root, rel)
    commitment["verified"] = bool(
        commitment["committed_in_time"]
        and commitment["unchanged_since"]
        and commitment["actions_run"]["verified"]
    )

    # The pre-start price snapshot is a fallback only if it was committed before the start.
    snap_path = reports_dir / event_id / "prices_close.json"
    snap_added = (
        ledger.first_commit_time(root, str(snap_path.relative_to(root)))
        if snap_path.is_file()
        else None
    )
    use_snapshot = snap_added is not None and snap_added <= card.start_utc
    snapshot = json.loads(snap_path.read_text())["kalshi"] if use_snapshot else {}
    quotes, source = _quotes(preds, cfg, fetcher, snapshot)
    settled = settle_mod.settle_bouts(preds, results, quotes)
    summary = settle_mod.summarize(settled)

    cards.save_results(results_dir / f"{event_id}.json", card, results)
    out = {
        "event_id": event_id,
        "event_name": card.name,
        "settled_at_utc": now.isoformat(),
        "commitment": commitment,
        "close_price_source": source,
        "close_snapshot_used": use_snapshot,
        "close_quotes": {k: list(v) if v else None for k, v in quotes.items()},
        "summary": summary,
        "bouts": json.loads(settled.to_json(orient="records")),
        **_actions_run(),
    }
    (reports_dir / event_id / "settlement.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    ledger.append(
        reports_dir / "ledger.json",
        {
            "event_id": event_id,
            "action": "settle",
            "at_utc": now.isoformat(),
            "verified": commitment["verified"],
            **_actions_run(),
        },
    )
    _write_summary(reports_dir)
    return out


def _write_summary(reports_dir: Path) -> None:
    """Pool every settled card whose commitment verified into ``summary.json``."""
    events, frames = [], []
    for p in sorted(reports_dir.glob("*/settlement.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        events.append(
            {
                "event_id": s["event_id"],
                "event_name": s["event_name"],
                "verified": s["commitment"]["verified"],
                **s["summary"],
            }
        )
        if s["commitment"]["verified"]:
            frames.append(pd.DataFrame(s["bouts"]))
    pooled = settle_mod.summarize(pd.concat(frames, ignore_index=True)) if frames else None
    (reports_dir / "summary.json").write_text(
        json.dumps({"events": events, "pooled_verified": pooled}, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# The scheduled driver
# ---------------------------------------------------------------------------


def run_due(cfg: Config, *, now: dt.datetime, fetcher, root: Path = REPO_ROOT) -> dict:
    """Do whatever the schedule says is due. Raises after trying everything if any step
    failed or a card was missed, so the workflow run fails and alerts."""
    reports_dir, _ = _dirs(cfg, root)
    frozen_at = frozen_commit_time(root)
    if frozen_at is None:
        return {
            "skipped": "no committed frozen configuration yet",
            "actions": [],
            "missed": [],
            "errors": [],
        }
    f = cfg.forward
    entries = ledger.load(reports_dir / "ledger.json")
    lo = now - dt.timedelta(days=float(f["history_days"]))
    hi = now + dt.timedelta(days=float(f["horizon_days"]))
    earliest = frozen_at + dt.timedelta(hours=float(f["min_lead_hours"]))
    in_scope = [
        e
        for e in espn.calendar(fetcher)
        if lo <= e.start_utc <= hi
        and e.start_utc - dt.timedelta(days=1) >= earliest
        and not ledger.has(entries, e.event_id, "settle")
        and not ledger.has(entries, e.event_id, "missed")
    ]
    cards_ = [espn.card(fetcher, e) for e in in_scope]
    cards_ = [c for c in cards_ if c.start_utc >= earliest]
    due = schedule.due_actions(
        cards_,
        entries,
        now,
        min_lead_hours=float(f["min_lead_hours"]),
        predict_window_hours=float(f["predict_window_hours"]),
        close_window_hours=float(f["close_window_hours"]),
        settle_after_hours=float(f["settle_after_hours"]),
    )
    done, missed, errors = [], [], []
    for action, event_id in due:
        try:
            if action == "predict":
                predict(event_id, cfg, now=now, fetcher=fetcher, root=root)
            elif action == "close":
                close(event_id, cfg, now=now, fetcher=fetcher, root=root)
            elif action == "settle":
                settle(event_id, cfg, now=now, fetcher=fetcher, root=root)
            elif action == "missed":
                ledger.append(
                    reports_dir / "ledger.json",
                    {
                        "event_id": event_id,
                        "action": "missed",
                        "at_utc": now.isoformat(),
                        **_actions_run(),
                    },
                )
                missed.append(event_id)
            done.append([action, event_id])
        except Exception as exc:
            errors.append(f"{action} {event_id}: {type(exc).__name__}: {exc}")
    report = {"now_utc": now.isoformat(), "actions": done, "missed": missed, "errors": errors}
    if errors or missed:
        raise ForwardError(json.dumps(report, indent=1))
    return report


def commit_message(report: dict) -> str:
    """A conventional commit subject for whatever run-due changed."""
    kinds = sorted({a for a, _ in report.get("actions", [])})
    return "docs: forward test " + (", ".join(kinds) if kinds else "update")
