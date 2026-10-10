"""The walk-forward engine: train on the past, predict the next event, advance.

**Why features can be precomputed.** The feature table is built once over the whole
dataset, which looks suspicious until you note that each feature row is already
point-in-time correct by construction -- `tests/test_leakage.py` proves a fight's
features do not move when later fights are mutated. So the walk-forward here governs
*model fitting*, not feature computation. Rebuilding features per fold would be slower
and would change nothing.

**Retraining cadence and staleness.** Refitting before every one of 619 events would cost
619 LightGBM fits for no methodological gain, so models are refit every
``backtest.retrain_every_n_events`` events (default 10) and reused in between. This makes
a model *stale* -- by the ninth event after a refit it has not seen the preceding nine
cards -- but it cannot make it leak: the training cutoff is the date of the refit event,
and every event predicted with that model is on or after that date. Staleness biases
*against* the model, so it is a conservative choice rather than a flattering one. A test
asserts the per-fold cutoff holds for every event, not merely for refit events.

**One continuous pass.** The engine walks from the first scorable event through the last,
producing one out-of-sample prediction per priced decisive fight. Metrics are then split
by window. This is what lets the dev and eval figures reconcile exactly to the same
6,790 fights, and it is what gives the market-anchored variant a legitimate warm-up: at
any event, that variant is fitted only on out-of-sample predictions from strictly earlier
events.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ufc_predictor.config import Config, get_config
from ufc_predictor.models.base import validate_predictions
from ufc_predictor.models.registry import MODEL_NAMES, build_model
from ufc_predictor.storage import db

#: The market-anchored stacker, reported as a separately labelled variant because it
#: consumes the closing line and therefore cannot support a "beats the market" claim.
VARIANT_NAME = "market_plus_model"

#: Minimum out-of-sample rows before the stacker is fitted at all.
MIN_STACK_ROWS = 400


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_panel(cfg: Config) -> pd.DataFrame:
    """Features + event id + closing line, for every decisive fight, in date order.

    Unpriced decisive fights are retained: they are legitimate *training* data. Only
    scoring is restricted to priced fights.
    """
    feats = db.read_table(cfg.paths.processed_dir, "features")
    fights = db.read_table(cfg.paths.processed_dir, "fights")
    odds = db.read_table(cfg.paths.processed_dir, "odds")

    panel = feats.merge(fights[["fight_id", "event_id"]], on="fight_id", validate="one_to_one")
    panel = panel.merge(
        odds[["fight_id", "p1_market", "odds_1", "odds_2"]], on="fight_id", how="left"
    )
    panel = panel.loc[panel["label"].notna()].copy()
    panel["scorable"] = panel["p1_market"].notna()
    return panel.sort_values(["date", "event_id", "fight_id"], kind="stable").reset_index(drop=True)


def run_walk_forward(
    cfg: Config | None = None,
    *,
    through: dt.date | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk forward over every scorable event.

    Args:
        through: stop after this date. Used during development to exercise the engine
            on the dev window without touching the evaluation period.

    Returns:
        ``(predictions, folds)``. ``predictions`` has one row per scored fight with a
        column per model; ``folds`` records each refit's cutoff and training size.
    """
    cfg = cfg or get_config()
    panel = load_panel(cfg)
    if through is not None:
        panel = panel.loc[panel["date"] <= through].copy()

    cadence = int(cfg.backtest.get("retrain_every_n_events", 10))
    min_train = int(cfg.backtest.get("min_train_fights", 500))

    # Events that contain at least one scorable fight, in chronological order.
    scorable = panel.loc[panel["scorable"]]
    events = scorable.groupby("event_id")["date"].min().sort_values(kind="stable").reset_index()

    if verbose:
        print("\nWalk-forward backtest")
        print(f"  retrain cadence   : every {cadence} events")
        print(f"  scorable events   : {len(events):,}")
        print(f"  scorable fights   : {len(scorable):,}")
        print(f"  training pool     : {len(panel):,} decisive fights (priced or not)")
        print(f"  min train fights  : {min_train}")
        print()

    models: dict[str, object] = {}
    stacker: LogisticRegression | None = None
    fold_id = -1
    # Counts only events actually predicted, so the cadence is exactly "every N scored
    # events" rather than drifting with however many early events lacked training data.
    scored_events = 0
    fold_rows: list[dict] = []
    pred_rows: list[dict] = []
    # Accumulated out-of-sample predictions, used only to fit the market-anchored
    # variant, and only from events strictly before the current one.
    history: list[dict] = []

    for ev in events.itertuples(index=False):
        event_id, event_date = ev.event_id, ev.date

        train = panel.loc[panel["date"] < event_date]
        if len(train) < min_train:
            continue

        refit = (not models) or (scored_events % cadence == 0)
        scored_events += 1
        if refit:
            fold_id += 1
            # Guard the invariant directly rather than trusting the filter above.
            if train["date"].max() >= event_date:
                raise AssertionError(
                    f"fold {fold_id}: training data reaches {train['date'].max()}, "
                    f"not strictly before the predicted event on {event_date}"
                )
            train_sorted = train.sort_values(["date", "fight_id"], kind="stable")
            y_train = train_sorted["label"].astype(int)

            models = {}
            for name in MODEL_NAMES:
                model = build_model(name, cfg)
                model.fit(train_sorted, y_train)
                models[name] = model
            fold_rows.append(
                {
                    "fold": fold_id,
                    "cutoff_date": event_date,
                    "first_event_id": event_id,
                    "train_rows": len(train_sorted),
                    "train_date_min": train_sorted["date"].min(),
                    "train_date_max": train_sorted["date"].max(),
                    "trees": getattr(models["lightgbm"], "best_iteration_", None),
                    "calibrated": getattr(models["lightgbm"], "_calibrator", None) is not None,
                }
            )
            if verbose and (fold_id % 10 == 0):
                print(
                    f"  fold {fold_id:3d}  cutoff {event_date}  "
                    f"train {len(train_sorted):,}  trees "
                    f"{getattr(models['lightgbm'], 'best_iteration_', '?')}"
                )

            # Refit the stacker on out-of-sample history from earlier events only.
            stacker = _fit_stacker(history, cfg)

        test = panel.loc[(panel["event_id"] == event_id) & panel["scorable"]]
        if test.empty:
            continue

        preds = {
            name: validate_predictions(m.predict_proba(test), len(test), name)
            for name, m in models.items()
        }

        # Market-anchored variant. Before the stacker has enough history it abstains
        # (NaN) rather than silently falling back to one of its inputs.
        if stacker is not None:
            Z = np.column_stack([_logit(preds["market"]), _logit(preds["lightgbm"])])
            preds[VARIANT_NAME] = np.clip(stacker.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
        else:
            preds[VARIANT_NAME] = np.full(len(test), np.nan)

        for j, row in enumerate(test.itertuples(index=False)):
            rec = {
                "fight_id": row.fight_id,
                "event_id": row.event_id,
                "date": row.date,
                "label": float(row.label),
                "fold": fold_id,
                "train_cutoff": fold_rows[-1]["cutoff_date"],
                "odds_1": row.odds_1,
                "odds_2": row.odds_2,
            }
            for name, arr in preds.items():
                rec[f"p_{name}"] = float(arr[j])
            pred_rows.append(rec)
            history.append(
                {
                    "date": row.date,
                    "p_market": float(preds["market"][j]),
                    "p_lightgbm": float(preds["lightgbm"][j]),
                    "label": float(row.label),
                }
            )

    predictions = pd.DataFrame(pred_rows)
    folds = pd.DataFrame(fold_rows)

    if predictions.empty:
        raise RuntimeError("walk-forward produced no predictions")
    if predictions["fight_id"].duplicated().any():
        raise RuntimeError("a fight was predicted more than once")

    if verbose:
        print(
            f"\n  folds: {len(folds)}  predictions: {len(predictions):,}  "
            f"span {predictions['date'].min()} .. {predictions['date'].max()}\n"
        )
    return predictions, folds


def _fit_stacker(history: list[dict], cfg: Config) -> LogisticRegression | None:
    """Fit the market-anchored stacker on accumulated out-of-sample predictions.

    Both inputs are out-of-sample by construction: every row in ``history`` was predicted
    by a model trained strictly before that row's event. So the stacker's coefficients
    answer a real question -- given the market's opinion, does the model's opinion carry
    any additional information? -- rather than being fitted to in-sample predictions,
    which would flatter the model.
    """
    if len(history) < MIN_STACK_ROWS:
        return None
    hist = pd.DataFrame(history)
    y = hist["label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    Z = np.column_stack(
        [_logit(hist["p_market"].to_numpy()), _logit(hist["p_lightgbm"].to_numpy())]
    )
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=cfg.seed)
    clf.fit(Z, y)
    return clf
