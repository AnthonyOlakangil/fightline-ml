"""Prediction logs: ``predict --event`` and ``settle``.

**Research backtest only. Not betting advice.**

**Nothing upcoming is invented.** A prediction log can only be written for an event that
exists in the committed snapshot. As of October 2026 no live source of upcoming cards is
reachable from a clean environment -- UFCStats sits behind a JavaScript proof-of-work
challenge and the odds project's ``upcoming.csv`` has been stale since April 2026 -- so
every log written here is a **replay** of a completed event and is labelled as such.

**What a replay log can and cannot prove.** The *predictions* are honest: for any event
inside the walk-forward backtest the stored probabilities come from a model trained only
on fights dated strictly earlier, so they are genuinely out of sample. What a replay log
cannot establish is *advance commitment* -- the result was already known when the file
was written, so its git commit timestamp proves nothing about foresight. Each log records
``proves_advance_commitment`` accordingly, and it is False for every replay. When a
genuinely future card becomes reachable, the same command writes ``mode: live`` with the
flag True, and committing that file before the event is what makes the timestamp mean
something.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_predictor.config import Config, get_config
from ufc_predictor.models.registry import build_model
from ufc_predictor.storage import db

LOG_DIR_NAME = "predictions"

#: Columns written to every prediction log, in order.
LOG_COLUMNS = [
    "logged_at_utc",
    "mode",
    "event",
    "event_date",
    "fight_id",
    "fighter_1",
    "fighter_2",
    "p_model",
    "p_market_vigfree",
    "odds_1",
    "odds_2",
    "result",
    "settled_at_utc",
    "log_loss",
]


class PredictError(RuntimeError):
    """Raised when a prediction log cannot be written honestly."""


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def slugify(name: str) -> str:
    """``UFC 332: Silva vs. Wang`` -> ``ufc-332-silva-vs-wang``."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return re.sub(r"-+", "-", s)


def find_event(events: pd.DataFrame, query: str) -> pd.Series:
    """Resolve an event name, exactly or by unique case-insensitive substring."""
    exact = events.loc[events["event_name"].str.lower() == query.lower()]
    if len(exact) == 1:
        return exact.iloc[0]

    partial = events.loc[events["event_name"].str.contains(re.escape(query), case=False, na=False)]
    if len(partial) == 1:
        return partial.iloc[0]
    if len(partial) > 1:
        names = partial["event_name"].head(8).tolist()
        raise PredictError(
            f"{query!r} matches {len(partial)} events; be more specific. e.g. {names}"
        )
    raise PredictError(
        f"no event matches {query!r}. Events come from the committed snapshot only -- "
        "upcoming cards are never invented."
    )


def _fit_for_event(cfg: Config, panel: pd.DataFrame, event_date: dt.date):
    """Fit a model on every decisive fight strictly before ``event_date``."""
    train = panel.loc[panel["date"] < event_date].sort_values(["date", "fight_id"], kind="stable")
    if len(train) < int(cfg.backtest["min_train_fights"]):
        raise PredictError(
            f"only {len(train)} fights precede {event_date}; need at least "
            f"{cfg.backtest['min_train_fights']} to fit"
        )
    model = build_model("lightgbm", cfg)
    model.fit(train, train["label"].astype(int))
    return model, len(train)


def predict_event(
    event_query: str,
    cfg: Config | None = None,
    *,
    today: dt.date | None = None,
    verbose: bool = True,
) -> Path:
    """Write a timestamped prediction log for one event.

    Returns the path of the CSV written.
    """
    cfg = cfg or get_config()
    today = today or dt.datetime.now(dt.UTC).date()

    events = db.read_table(cfg.paths.processed_dir, "events")
    fights = db.read_table(cfg.paths.processed_dir, "fights")
    features = db.read_table(cfg.paths.processed_dir, "features")
    odds = db.read_table(cfg.paths.processed_dir, "odds")

    event = find_event(events, event_query)
    event_date = event["date"]
    mode = "live" if event_date > today else "replay"

    card = fights.loc[fights["event_id"] == event["event_id"]]
    if card.empty:
        raise PredictError(f"event {event['event_name']!r} has no fights in the snapshot")

    # Prefer the stored walk-forward prediction: it is already out-of-sample and has
    # been audited. Fall back to fitting only when the event is outside the backtest.
    stored_path = cfg.paths.reports_dir / "walkforward_predictions.csv"
    stored = None
    if stored_path.is_file():
        s = pd.read_csv(stored_path, parse_dates=["date"])
        s["date"] = s["date"].dt.date
        s = s.loc[s["event_id"] == event["event_id"]]
        if not s.empty:
            stored = s.set_index("fight_id")

    source = "stored walk-forward prediction (out-of-sample)"
    model = None
    n_train = None
    if stored is None:
        panel = features.loc[features["label"].notna()].merge(
            odds[["fight_id", "p1_market"]], on="fight_id", how="left"
        )
        model, n_train = _fit_for_event(cfg, panel, event_date)
        source = f"model fitted on {n_train:,} fights before {event_date}"

    market = odds.set_index("fight_id")
    feat = features.set_index("fight_id")
    logged_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    rows: list[dict] = []
    for fight in card.itertuples(index=False):
        fid = fight.fight_id
        if stored is not None and fid in stored.index:
            p_model = float(stored.loc[fid, "p_lightgbm"])
        elif model is not None and fid in feat.index:
            p_model = float(model.predict_proba(feat.loc[[fid]])[0])
        else:
            continue

        p_market = float(market.loc[fid, "p1_market"]) if fid in market.index else float("nan")
        o1 = float(market.loc[fid, "odds_1"]) if fid in market.index else float("nan")
        o2 = float(market.loc[fid, "odds_2"]) if fid in market.index else float("nan")

        # The result is left blank unless the fight is already decided AND this is a
        # replay. A live log must ship with an empty result column.
        result = ""
        if mode == "replay" and bool(fight.is_decisive) and pd.notna(fight.winner_id):
            result = "fighter_1" if fight.winner_id == fight.fighter_1 else "fighter_2"

        rows.append(
            {
                "logged_at_utc": logged_at,
                "mode": mode,
                "event": event["event_name"],
                "event_date": str(event_date),
                "fight_id": fid,
                "fighter_1": fight.name_1,
                "fighter_2": fight.name_2,
                "p_model": round(p_model, 6),
                "p_market_vigfree": round(p_market, 6) if np.isfinite(p_market) else "",
                "odds_1": o1 if np.isfinite(o1) else "",
                "odds_2": o2 if np.isfinite(o2) else "",
                "result": result,
                "settled_at_utc": "",
                "log_loss": "",
            }
        )

    if not rows:
        raise PredictError(f"no predictable fights on {event['event_name']!r}")

    log = pd.DataFrame(rows, columns=LOG_COLUMNS)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = cfg.paths.reports_dir / LOG_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}_{slugify(event['event_name'])}_{mode}.csv"
    log.to_csv(path, index=False)

    meta = {
        "disclaimer": "Research backtest only. Not betting advice.",
        "logged_at_utc": logged_at,
        "git_sha": _git_sha(),
        "seed": cfg.seed,
        "mode": mode,
        "event": event["event_name"],
        "event_date": str(event_date),
        "today_utc": str(today),
        "n_fights": len(log),
        "n_priced": int((log["p_market_vigfree"] != "").sum()),
        "probability_source": source,
        "proves_advance_commitment": mode == "live",
        "advance_commitment_note": (
            "This event had already taken place when the log was written, so committing "
            "the file cannot demonstrate that the predictions were made beforehand. The "
            "probabilities are still genuinely out-of-sample -- the model that produced "
            "them saw only fights dated strictly earlier -- but foresight is not what "
            "the commit timestamp establishes here."
            if mode == "replay"
            else (
                "This event has not yet taken place. Committing this file before the "
                "event is what makes the commit timestamp evidence of advance commitment."
            )
        ),
        "upcoming_data_available": False,
        "upcoming_data_note": (
            "No live source of upcoming cards is reachable: UFCStats serves a JavaScript "
            "proof-of-work challenge, and the odds project's upcoming.csv has been stale "
            "since 2026-04-04. Rather than invent fixtures or prices, completed events "
            "are replayed and labelled."
        ),
    }
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if verbose:
        print(f"\n{event['event_name']}  ({event_date}, mode={mode})")
        print(f"  probabilities : {source}")
        print(f"  fights        : {len(log)}  ({meta['n_priced']} with a closing line)")
        if mode == "replay":
            print("  note          : replay of a completed event; the commit timestamp")
            print("                  cannot prove the prediction preceded the result")
        print(f"  wrote {path}")
        print(f"        {path.with_suffix('.json')}\n")
    return path


def settle(log_path: Path | str, cfg: Config | None = None, *, verbose: bool = True) -> dict:
    """Fill in results for a prediction log and score it.

    Rewrites the CSV in place with ``result``, ``settled_at_utc`` and per-fight
    ``log_loss``, and returns a summary.
    """
    cfg = cfg or get_config()
    path = Path(log_path)
    if not path.is_file():
        raise PredictError(f"prediction log not found: {path}")

    log = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in LOG_COLUMNS if c not in log.columns]
    if missing:
        raise PredictError(f"{path} is missing columns {missing}")

    fights = db.read_table(cfg.paths.processed_dir, "fights").set_index("fight_id")
    settled_at = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")

    n_settled = 0
    losses: list[float] = []
    market_losses: list[float] = []

    for i, row in log.iterrows():
        fid = row["fight_id"]
        if fid not in fights.index:
            continue
        fight = fights.loc[fid]
        if not bool(fight["is_decisive"]) or pd.isna(fight["winner_id"]):
            # Draws and no contests have no binary result; leave them unsettled rather
            # than recording a loss for one side.
            log.loc[i, "result"] = "no_contest_or_draw"
            continue

        y = 1.0 if fight["winner_id"] == fight["fighter_1"] else 0.0
        log.loc[i, "result"] = "fighter_1" if y == 1.0 else "fighter_2"
        log.loc[i, "settled_at_utc"] = settled_at

        p = float(row["p_model"])
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        log.loc[i, "log_loss"] = f"{ll:.6f}"
        losses.append(ll)
        n_settled += 1

        if row["p_market_vigfree"]:
            pm = min(max(float(row["p_market_vigfree"]), 1e-6), 1 - 1e-6)
            market_losses.append(-(y * math.log(pm) + (1 - y) * math.log(1 - pm)))

    log.to_csv(path, index=False)

    summary = {
        "log": str(path),
        "settled_at_utc": settled_at,
        "n_rows": len(log),
        "n_settled": n_settled,
        "model_log_loss": float(np.mean(losses)) if losses else None,
        "market_log_loss": float(np.mean(market_losses)) if market_losses else None,
        "n_priced": len(market_losses),
    }
    if summary["model_log_loss"] is not None and summary["market_log_loss"] is not None:
        summary["model_minus_market"] = summary["model_log_loss"] - summary["market_log_loss"]

    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["settlement"] = summary
        sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if verbose:
        print(f"\nSettled {path.name}")
        print(f"  settled fights : {n_settled} of {len(log)}")
        if summary["model_log_loss"] is not None:
            print(f"  model log loss : {summary['model_log_loss']:.4f}")
        if summary["market_log_loss"] is not None:
            print(f"  market log loss: {summary['market_log_loss']:.4f}  (n={summary['n_priced']})")
            print(
                f"  difference     : {summary['model_minus_market']:+.4f}  "
                "(positive means the market was better)"
            )
        print()
    return summary
