"""Hyperparameter search, confined to the dev window by construction.

    uv run python -m ufc_predictor.models.tune

Run by hand. The chosen parameters are written into ``config/config.yaml`` and then
frozen: the walk-forward backtest reads them from config and never re-tunes. That
ordering is what makes the held-out result meaningful -- hyperparameters selected
against the evaluation period would make it a second training set.

Two guards enforce the confinement rather than relying on care:

* the candidate frame is filtered to ``date <= dev_end`` and the function raises if any
  later row survives;
* validation is a chronological holdout -- the most recent slice of the dev window --
  never a random split, because random folds would let later dev fights inform the
  model that predicts earlier ones.

The search space is deliberately narrow and biased toward strong regularization. With
~2,000 usable dev fights and 117 features, the real risk is a model that memorises the
dev era; shallow trees, high ``min_child_samples`` and column subsampling cost little
and guard against that.
"""

from __future__ import annotations

import itertools
import json

import pandas as pd
from sklearn.metrics import log_loss

from ufc_predictor.config import Config, get_config
from ufc_predictor.features.pit import FEATURE_COLS
from ufc_predictor.models.lgbm import LightGBMModel
from ufc_predictor.storage import db

# Narrow grid, regularization-first. Each axis has a clear rationale:
#   num_leaves / max_depth : capacity. Small, because the dev window is small.
#   min_child_samples      : refuses to split on a handful of fights.
#   learning_rate          : lower rates with early stopping beat hand-tuned tree counts.
#   colsample_bytree       : many of the 117 features are near-duplicates across windows.
#   reg_lambda             : explicit L2 on leaf weights.
SEARCH_SPACE: dict[str, list] = {
    "num_leaves": [7, 15, 31],
    "max_depth": [3, 4, 6],
    "min_child_samples": [20, 40, 80],
    "learning_rate": [0.02, 0.05],
    "colsample_bytree": [0.5, 0.7],
    "reg_lambda": [1.0, 5.0],
}


def dev_frame(cfg: Config) -> pd.DataFrame:
    """Labelled fights inside the dev window, in date order.

    Raises:
        RuntimeError: if any row falls outside the dev window.
    """
    feats = db.read_table(cfg.paths.processed_dir, "features")
    dev = feats.loc[feats["label"].notna() & (feats["date"] <= cfg.windows.dev_end)].sort_values(
        ["date", "fight_id"], kind="stable"
    )

    if len(dev) == 0:
        raise RuntimeError("no labelled fights inside the dev window")
    if dev["date"].max() > cfg.windows.dev_end:
        raise RuntimeError(
            f"tuning frame reaches {dev['date'].max()}, past dev_end {cfg.windows.dev_end}"
        )
    return dev.reset_index(drop=True)


def chronological_holdout(
    dev: pd.DataFrame, validation_fraction: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dev window in time: earlier fights train, the recent tail validates."""
    n_val = round(len(dev) * validation_fraction)
    if n_val < 100:
        raise RuntimeError(f"validation tail of {n_val} fights is too small to tune on")
    split = len(dev) - n_val
    train, val = dev.iloc[:split].copy(), dev.iloc[split:].copy()
    if train["date"].max() > val["date"].min():
        # Ties on the boundary date are acceptable only if they cannot straddle.
        boundary = val["date"].min()
        train = train.loc[train["date"] < boundary]
        val = val.loc[val["date"] >= boundary]
    return train, val


def evaluate(params: dict, train: pd.DataFrame, val: pd.DataFrame, cfg: Config) -> dict:
    """Fit one candidate and score it on the chronological validation tail."""
    model = LightGBMModel(
        seed=cfg.seed,
        params={**params, "n_estimators": 2000},
        validation_fraction=0.2,
        early_stopping_rounds=50,
        calibrate=False,  # tuning compares raw fit quality; calibration is applied later
    )
    model.fit(train, train["label"].astype(int))
    p = model.predict_proba(val)
    ll = float(log_loss(val["label"].astype(int), p, labels=[0, 1]))
    return {"log_loss": ll, "best_iteration": model.best_iteration_, **params}


# Candidates within this log-loss distance of the best are treated as tied. With ~860
# validation fights, differences of this size are noise, and chasing them would be
# fitting the tuning split rather than the problem.
NOISE_TOLERANCE = 0.001


def capacity_rank(result: dict) -> tuple:
    """Order tied candidates from most to least regularized.

    Among statistically indistinguishable configurations, the lowest-capacity one is
    preferred: it is the choice least able to memorise the dev era. Note that
    ``num_leaves`` is inactive whenever ``max_depth`` binds first (depth 3 caps a tree at
    8 leaves), which is why 15 and 31 leaves can score identically.
    """
    effective_leaves = min(result["num_leaves"], 2 ** result["max_depth"])
    return (
        result["max_depth"],
        effective_leaves,
        -result["min_child_samples"],  # higher is more regularized
        -result["reg_lambda"],
        result["colsample_bytree"],
    )


def select_best(results: list[dict]) -> tuple[dict, list[dict]]:
    """Pick the most regularized candidate that is statistically tied with the best.

    Returns the selection and the full tied set, so the report can show the plateau
    rather than implying the winner was uniquely good.
    """
    best_ll = min(r["log_loss"] for r in results)
    tied = [r for r in results if r["log_loss"] <= best_ll + NOISE_TOLERANCE]
    tied_sorted = sorted(tied, key=capacity_rank)
    return tied_sorted[0], tied_sorted


def search(cfg: Config | None = None, *, verbose: bool = True) -> dict:
    cfg = cfg or get_config()
    dev = dev_frame(cfg)
    train, val = chronological_holdout(dev)

    if verbose:
        print("\nTuning LightGBM inside the dev window only")
        print(f"  dev window : {cfg.windows.dev_start} .. {cfg.windows.dev_end}")
        print(
            f"  train      : {len(train):,} fights  {train['date'].min()} .. {train['date'].max()}"
        )
        print(f"  validate   : {len(val):,} fights  {val['date'].min()} .. {val['date'].max()}")
        print(f"  features   : {len(FEATURE_COLS)}")

    keys = list(SEARCH_SPACE)
    combos = list(itertools.product(*(SEARCH_SPACE[k] for k in keys)))
    if verbose:
        print(f"  candidates : {len(combos)}\n")

    results: list[dict] = []
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo, strict=True))
        # Keep the fixed regularization settings from config that are not searched.
        params.setdefault("subsample", 0.8)
        params.setdefault("subsample_freq", 1)
        params.setdefault("reg_alpha", 0.1)
        res = evaluate(params, train, val, cfg)
        results.append(res)
        if verbose and (i % 10 == 0 or i == len(combos)):
            best = min(results, key=lambda r: r["log_loss"])
            print(f"  {i:3d}/{len(combos)}  best log loss so far: {best['log_loss']:.4f}")

    results.sort(key=lambda r: r["log_loss"])
    lowest = results[0]
    selected, tied = select_best(results)

    baseline = evaluate(
        {k: v for k, v in cfg.lightgbm.items() if k != "verbosity"}, train, val, cfg
    )

    out = {
        "dev_window": [str(cfg.windows.dev_start), str(cfg.windows.dev_end)],
        "train_rows": len(train),
        "validation_rows": len(val),
        "validation_span": [str(val["date"].min()), str(val["date"].max())],
        "n_candidates": len(combos),
        "noise_tolerance": NOISE_TOLERANCE,
        "lowest_log_loss_candidate": lowest,
        "n_tied_within_tolerance": len(tied),
        "selected": selected,
        "selection_rule": (
            "most regularized candidate within "
            f"{NOISE_TOLERANCE} log loss of the best; differences that small are noise "
            "at this validation size, and the lowest-capacity tied model is least able "
            "to memorise the dev era"
        ),
        "config_baseline": baseline,
        "top_10": results[:10],
    }
    path = cfg.paths.reports_dir / "tuning.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if verbose:
        print(f"\n  lowest log loss: {lowest['log_loss']:.4f}")
        print(f"  candidates within {NOISE_TOLERANCE} of it: {len(tied)} (a plateau, not a peak)")
        print("\n  SELECTED (most regularized among those tied):")
        for k in keys:
            print(f"    {k:<20} {selected[k]}")
        print(f"    {'best_iteration':<20} {selected['best_iteration']}")
        print(f"    {'log_loss':<20} {selected['log_loss']:.4f}")
        print(f"\n  current config scores {baseline['log_loss']:.4f} on the same split")
        print(f"  wrote {path}\n")

    return out


if __name__ == "__main__":  # pragma: no cover
    search()
