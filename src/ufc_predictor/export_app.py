"""Export precomputed artefacts for the Streamlit demo.

The app must start instantly and must never fit a model, so everything it needs is
produced here and written to ``app/artifacts/``:

* ``model.joblib`` -- one LightGBM fitted on all decisive fights, for hypothetical
  matchups the app is asked about.
* ``fighters.parquet`` -- each fighter's most recent point-in-time feature snapshot,
  so a hypothetical pairing can be assembled without recomputing history.
* ``summary.json`` -- headline numbers lifted from ``metrics.json`` and
  ``parlay_metrics.json`` so the app never parses the large report files.

The matchup feature row is built from two fighter snapshots exactly as the training
rows were: per-fighter columns side by side, then their differences. A hypothetical
pairing has no closing line, and the app says so rather than inventing one.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ufc_predictor.config import Config, get_config
from ufc_predictor.features.pit import FEATURE_COLS, RATE_METRICS, _windows
from ufc_predictor.models.registry import build_model
from ufc_predictor.storage import db

ARTIFACT_DIR = "app/artifacts"

#: Per-fighter feature stems that exist as `<stem>_1` / `<stem>_2` in the model matrix.
PER_FIGHTER_STEMS = [
    "elo",
    "n_prior_fights",
    "win_rate",
    "win_streak",
    "loss_streak",
    "days_since_last",
    "finish_rate",
    "is_debut",
    "has_stats_history",
]


def _rate_stems(cfg: Config) -> list[str]:
    return [
        f"{metric}_{win}"
        for win in _windows(tuple(cfg.features["rolling_windows"]))
        for metric in RATE_METRICS
    ]


def build_fighter_snapshots(cfg: Config) -> pd.DataFrame:
    """Each fighter's latest feature state, taken from their most recent fight.

    A fighter's row in the feature table describes them *before* that fight, so the
    snapshot taken from their last bout is their state going into it. For a matchup
    today that is slightly stale -- it excludes the last fight's own contribution --
    which is noted in the app rather than silently corrected.
    """
    feats = db.read_table(cfg.paths.processed_dir, "features")
    fights = db.read_table(cfg.paths.processed_dir, "fights")
    fighters = db.read_table(cfg.paths.processed_dir, "fighters")

    stems = PER_FIGHTER_STEMS + _rate_stems(cfg)
    names = fights.set_index("fight_id")[["name_1", "name_2"]]

    rows: list[dict] = []
    for slot in ("1", "2"):
        cols = {f"{s}_{slot}": s for s in stems if f"{s}_{slot}" in feats.columns}
        sub = feats[["fight_id", "date", f"fighter_{slot}", *cols]].copy()
        sub = sub.rename(columns={f"fighter_{slot}": "fighter_id", **cols})
        sub["name"] = sub["fight_id"].map(names[f"name_{slot}"])
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long = long.sort_values(["fighter_id", "date"], kind="stable")
    latest = long.groupby("fighter_id", as_index=False).last()

    # Display name only: the table's attributes are a present-day scrape and are never a
    # model input (see features/pit.py).
    meta = fighters.set_index("fighter_id")
    latest["display_name"] = latest["fighter_id"].map(meta["name"]).fillna(latest["name"])
    latest = latest.rename(columns={"date": "last_fight_date"})
    return latest.drop(columns=["fight_id", "name"]).reset_index(drop=True)


def matchup_row(
    snap_a: pd.Series, snap_b: pd.Series, cfg: Config, *, as_of: dt.date | None = None
) -> pd.DataFrame:
    """Assemble a single model-ready feature row for a hypothetical matchup.

    Fighters are placed in canonical order (sorted ``fighter_id``) so the row matches
    how the model was trained, and the caller is told which slot each occupied.
    """
    as_of = as_of or dt.date.today()
    a, b = (
        (snap_a, snap_b)
        if str(snap_a["fighter_id"]) < str(snap_b["fighter_id"])
        else (snap_b, snap_a)
    )

    stems = PER_FIGHTER_STEMS + _rate_stems(cfg)
    row: dict[str, float] = {}
    for slot, snap in (("1", a), ("2", b)):
        for stem in stems:
            row[f"{stem}_{slot}"] = snap.get(stem, np.nan)
        # Recompute the time-dependent feature for the requested date.
        last = snap.get("last_fight_date")
        row[f"days_since_last_{slot}"] = (
            float((as_of - last).days) if last is not None and not pd.isna(last) else np.nan
        )

    for stem in stems:
        row[f"{stem}_diff"] = _safe_diff(row.get(f"{stem}_1"), row.get(f"{stem}_2"))

    # Bout-level defaults: a non-title three-round fight.
    row["title_bout"] = False
    row["scheduled_rounds"] = 3.0

    frame = pd.DataFrame([row])
    for col in FEATURE_COLS:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame.loc[:, FEATURE_COLS]


def _safe_diff(x, y):
    try:
        if x is None or y is None or pd.isna(x) or pd.isna(y):
            return np.nan
        return float(x) - float(y)
    except (TypeError, ValueError):
        return np.nan


def run(cfg: Config | None = None, *, verbose: bool = True) -> dict:
    """Write every artefact the demo app needs."""
    cfg = cfg or get_config()
    out = Path(ARTIFACT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    feats = db.read_table(cfg.paths.processed_dir, "features")
    train = feats.loc[feats["label"].notna()].sort_values(["date", "fight_id"], kind="stable")
    model = build_model("lightgbm", cfg)
    model.fit(train, train["label"].astype(int))
    joblib.dump(model, out / "model.joblib")

    snapshots = build_fighter_snapshots(cfg)
    snapshots.to_parquet(out / "fighters.parquet", index=False)

    metrics = json.loads((cfg.paths.reports_dir / "metrics.json").read_text(encoding="utf-8"))
    parlay_path = cfg.paths.reports_dir / "parlay_metrics.json"
    parlay = json.loads(parlay_path.read_text(encoding="utf-8")) if parlay_path.is_file() else {}

    # Historical predictions, trimmed to what the app displays.
    preds = pd.read_csv(cfg.paths.reports_dir / "walkforward_predictions.csv")
    fights = db.read_table(cfg.paths.processed_dir, "fights")
    preds = preds.merge(
        fights[["fight_id", "name_1", "name_2", "event_name"]], on="fight_id", how="left"
    )
    keep = [
        "fight_id",
        "event_name",
        "date",
        "name_1",
        "name_2",
        "label",
        "p_market",
        "p_lightgbm",
        "p_market_plus_model",
    ]
    preds.loc[:, keep].to_parquet(out / "predictions.parquet", index=False)

    summary = {
        "disclaimer": "Research backtest only. Not betting advice.",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "eval_run_count": metrics["eval_run_count"],
        "windows": metrics["windows"],
        "reconciliation": metrics["reconciliation"],
        "walk_forward": {
            k: metrics["walk_forward"][k]
            for k in ("retrain_every_n_events", "n_folds", "eval_events", "dev_events")
        },
        "results": metrics["results"],
        "paired_vs_market": metrics["paired_vs_market"],
        "by_year": metrics["by_year"],
        "bootstrap": metrics["bootstrap"],
        "parlay": {
            "settings": parlay.get("settings", {}),
            "by_source": {
                k: {
                    "n_candidates": v.get("n_candidates", 0),
                    "n_staked": v.get("n_staked", 0),
                    "predicted_vs_realized": v.get("predicted_vs_realized", {}),
                    "bankroll": v.get("bankroll", {}),
                    "bankroll_bootstrap": {
                        kk: vv
                        for kk, vv in v.get("bankroll_bootstrap", {}).items()
                        if kk != "path_band"
                    },
                }
                for k, v in parlay.get("by_source", {}).items()
            },
            "within_card_residual_correlation": parlay.get("within_card_residual_correlation", {}),
            "legs_priced": parlay.get("legs_priced", ""),
        },
        "n_fighters": len(snapshots),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    if verbose:
        print(f"\nExported demo artefacts to {out}/")
        print(f"  model.joblib        fitted on {len(train):,} decisive fights")
        print(f"  fighters.parquet    {len(snapshots):,} fighter snapshots")
        print(f"  predictions.parquet {len(preds):,} out-of-sample predictions")
        print("  summary.json        headline metrics (no large report parsing at runtime)\n")
    return summary


if __name__ == "__main__":  # pragma: no cover
    run()
