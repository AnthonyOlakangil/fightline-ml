"""Model cards: what each model is, and how it fits on the dev window.

    uv run ufc-predictor models

Writes ``reports/model_cards.json``. Every metric here is measured on a chronological
holdout **inside the dev window** -- these are sanity figures for the fitting regime,
not results. The project's results are the walk-forward backtest, and nothing may report
an evaluation-window number before that runs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ufc_predictor.config import Config, get_config
from ufc_predictor.features.pit import FEATURE_COLS
from ufc_predictor.models.base import validate_predictions
from ufc_predictor.models.registry import MODEL_NAMES, build_model
from ufc_predictor.storage import db


def dev_frame(cfg: Config) -> pd.DataFrame:
    """Labelled dev-window fights with the closing line attached where it exists.

    Pricing is left-joined rather than inner-joined, so training can use every decisive
    fight while scoring is restricted to the priced subset.
    """
    feats = db.read_table(cfg.paths.processed_dir, "features")
    odds = db.read_table(cfg.paths.processed_dir, "odds")
    df = feats.merge(odds[["fight_id", "p1_market"]], on="fight_id", how="left")
    dev = df.loc[df["label"].notna() & (df["date"] <= cfg.windows.dev_end)].sort_values(
        ["date", "fight_id"], kind="stable"
    )
    if dev["date"].max() > cfg.windows.dev_end:
        raise RuntimeError("dev frame leaked past dev_end")
    return dev.reset_index(drop=True)


def run(cfg: Config | None = None, *, verbose: bool = True) -> dict:
    cfg = cfg or get_config()
    dev = dev_frame(cfg)

    # Train on every decisive fight, priced or not -- withholding unpriced fights would
    # discard real training signal for no benefit, since the market baseline is an input
    # and is unaffected by what the model learns from.
    #
    # Score only where a closing line exists, so all three models are compared on
    # exactly the same rows. Scoring the models on fights the market cannot be scored on
    # would make the comparison incommensurable.
    split = int(len(dev) * 0.7)
    train = dev.iloc[:split]
    test = dev.iloc[split:].loc[lambda d: d["p1_market"].notna()]
    y = test["label"].astype(int).to_numpy()

    if verbose:
        print("\nModel cards (dev window only -- these are NOT results)")
        print(
            f"  train: {len(train):,} decisive fights  "
            f"{train['date'].min()} .. {train['date'].max()}"
        )
        print(
            f"  score: {len(test):,} priced fights     "
            f"{test['date'].min()} .. {test['date'].max()}  (same rows for every model)"
        )
        print()

    cards: dict[str, dict] = {}
    for name in MODEL_NAMES:
        model = build_model(name, cfg)
        model.fit(train, train["label"].astype(int))
        p = validate_predictions(model.predict_proba(test), len(test), name)

        card = {
            "name": name,
            "requires_fit": model.requires_fit,
            "n_features": (
                len(getattr(model, "feature_cols", []))
                if name == "lightgbm"
                else (1 if name == "elo" else 0)
            ),
            "dev_holdout": {
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
                "auc": float(roc_auc_score(y, p)),
                "accuracy": float(((p > 0.5) == (y > 0.5)).mean()),
                "mean_prediction": float(np.mean(p)),
                "n": len(test),
            },
        }
        if name == "lightgbm":
            card["hyperparams"] = dict(cfg.lightgbm)
            card["fit_regime"] = dict(cfg.lightgbm_fit)
            card["trees_after_early_stopping"] = model.best_iteration_
            card["calibrated"] = model._calibrator is not None
        if name == "elo":
            card["hyperparams"] = dict(cfg.elo)
            card["inputs"] = ["elo_diff"]
        if name == "market":
            card["inputs"] = ["p1_market (de-vigged closing line)"]

        cards[name] = card
        if verbose:
            d = card["dev_holdout"]
            print(
                f"  {name:<10} log loss {d['log_loss']:.4f}  brier {d['brier']:.4f}  "
                f"auc {d['auc']:.4f}  acc {d['accuracy']:.4f}"
            )

    out = {
        "note": (
            "Dev-window chronological holdout only. Not project results: the evaluation "
            "window is untouched until the walk-forward backtest writes metrics.json."
        ),
        "dev_window": [str(cfg.windows.dev_start), str(cfg.windows.dev_end)],
        "holdout_span": [str(test["date"].min()), str(test["date"].max())],
        "train_rows_all_decisive": len(train),
        "scored_rows_priced_only": len(test),
        "seed": cfg.seed,
        "total_features_available": len(FEATURE_COLS),
        "models": cards,
    }
    path = cfg.paths.reports_dir / "model_cards.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    if verbose:
        print(f"\n  wrote {path}\n")
    return out


if __name__ == "__main__":  # pragma: no cover
    run()
