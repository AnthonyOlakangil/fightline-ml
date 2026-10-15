"""Dev-window leakage diagnostics: the shuffle test, its control, and attribute evidence.

    uv run ufc-predictor diagnose

Shared by ``tests/test_leakage.py`` and the README, so the figures the README reports are
produced by exactly the code the test suite asserts on. Everything here is confined to the
dev window and raises if it reaches past ``dev_end``: a performance number measured on the
evaluation window is an eval-window number whatever the intent.

Three things are measured:

* **The shuffle test.** Randomize who won every fight, rebuild every feature from the
  shuffled history, train and score. With an honest point-in-time builder there is nothing
  to learn and performance collapses to a coin flip.
* **Its control.** The same model on real outcomes must find signal, or the shuffle test
  would pass vacuously on a broken pipeline.
* **The attribute leak** that evaluation run 1 suffered: how strongly a blank in the
  present-day fighters table predicted the result. The features no longer read that table,
  so this is measured from the table directly.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ufc_predictor.config import Config, get_config
from ufc_predictor.features.pit import FEATURE_COLS, build_features
from ufc_predictor.storage import db

#: Fighters-table attributes that evaluation run 1 used as features, by feature name.
SCRAPE_ATTRIBUTES = {"reach": "reach_cm", "height": "height_cm", "age": "dob"}

#: Trees for the fixed diagnostic model. Not tuned, not early-stopped, not calibrated.
DIAGNOSTIC_TREES = 200


def build(
    fights: pd.DataFrame, tables: dict, cfg: Config, *, families: Sequence[str] = ("base",)
) -> pd.DataFrame:
    """The feature table. ``families`` defaults to the base set the README figures use;
    the leakage suite passes every family."""
    return build_features(
        fights,
        tables["fight_stats"],
        tables["fighters"],
        rolling_windows=tuple(cfg.features["rolling_windows"]),
        elo_params=cfg.elo,
        families=families,
        recency_half_life_fights=float(cfg.features["recency_half_life_fights"]),
        glicko_params=dict(cfg.glicko2),
    )


def dev_only_split(
    feats: pd.DataFrame, cfg: Config, *, through: dt.date | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 70/30 split of labelled fights up to ``through`` (default ``dev_end``).

    Phase 8's leakage tests pass ``selection_end`` so that no check built on the new
    feature families ever scores a 2015 outcome outside the once-only confirmation.

    Raises:
        RuntimeError: if the test portion reaches past the dev window.
    """
    end = min(through or cfg.windows.dev_end, cfg.windows.dev_end)
    usable = feats.loc[feats["label"].notna() & (feats["date"] <= end)]
    usable = usable.sort_values("date", kind="stable")
    split = int(len(usable) * 0.7)
    train, test = usable.iloc[:split], usable.iloc[split:]
    if test["date"].max() > cfg.windows.dev_end:
        raise RuntimeError("diagnostic reached the evaluation window")
    return train, test


def fit_predict_dev(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Config,
    *,
    feature_cols: Sequence[str] = FEATURE_COLS,
) -> np.ndarray:
    """A fixed LightGBM on the frozen hyperparameters, for diagnostics only."""
    import lightgbm as lgb

    cols = list(feature_cols)
    params = dict(cfg.lightgbm)
    params.update({"n_estimators": DIAGNOSTIC_TREES, "random_state": cfg.seed, "verbosity": -1})
    model = lgb.LGBMClassifier(**params)
    model.fit(train[cols], train["label"].astype(int))
    return model.predict_proba(test[cols])[:, 1]


def shuffle_outcomes(fights: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Reassign every decisive fight's winner by a fair coin."""
    rng = np.random.default_rng(seed)
    shuffled = fights.copy()
    decisive = shuffled["is_decisive"].to_numpy()
    pick_first = rng.random(len(shuffled)) < 0.5
    shuffled.loc[decisive, "winner_id"] = np.where(
        pick_first[decisive],
        shuffled.loc[decisive, "fighter_1"],
        shuffled.loc[decisive, "fighter_2"],
    )
    return shuffled


def _score(test: pd.DataFrame, p: np.ndarray) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    y = test["label"].astype(int)
    return {"auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def attribute_leak_evidence(tables: dict, feats: pd.DataFrame, cfg: Config) -> dict:
    """How strongly a blank fighters-table attribute predicted the result, dev window only.

    Two measurements. Per attribute: in bouts where exactly one fighter has it blank, how
    often the blank side won. And the reason blanking *both* sides would not have closed
    the leak: in pairs with any one-sided blank, how often the less-experienced fighter
    won, against complete pairs.
    """
    fighters = tables["fighters"].set_index("fighter_id")
    dates = feats["date"]
    dev = feats.loc[
        feats["label"].notna() & (dates >= cfg.windows.dev_start) & (dates <= cfg.windows.dev_end)
    ].copy()
    if dev["date"].max() > cfg.windows.dev_end:
        raise RuntimeError("attribute evidence reached the evaluation window")

    per_attribute = {}
    any_one_sided = pd.Series(False, index=dev.index)
    all_complete = pd.Series(True, index=dev.index)
    for name, col in SCRAPE_ATTRIBUTES.items():
        values = fighters[col]
        blank_1 = dev["fighter_1"].map(values).isna()
        blank_2 = dev["fighter_2"].map(values).isna()
        one_1, one_2 = blank_1 & ~blank_2, blank_2 & ~blank_1
        n = int(one_1.sum() + one_2.sum())
        wins = dev.loc[one_1, "label"].sum() + (1 - dev.loc[one_2, "label"]).sum()
        per_attribute[name] = {
            "bouts_one_side_blank": n,
            "blank_side_win_rate": float(wins / n) if n else None,
        }
        any_one_sided |= one_1 | one_2
        all_complete &= ~blank_1 & ~blank_2

    exp_diff = dev["n_prior_fights_1"] - dev["n_prior_fights_2"]
    less_won = np.where(exp_diff < 0, dev["label"], 1 - dev["label"])
    unequal = exp_diff != 0

    def rate(mask: pd.Series) -> dict:
        sel = mask & unequal
        return {"bouts": int(sel.sum()), "less_experienced_win_rate": float(less_won[sel].mean())}

    return {
        "window": [str(cfg.windows.dev_start), str(cfg.windows.dev_end)],
        "decisive_bouts": len(dev),
        "per_attribute": per_attribute,
        "pairs_with_a_one_sided_blank": rate(any_one_sided),
        "complete_pairs": rate(all_complete),
    }


def calibration_bound_evidence(cfg: Config) -> dict:
    """Dev-window evidence for LightGBM's calibration bounds, fixed a priori in config.

    Descriptive only -- the bound was not searched. It reports how extreme the vig-free
    closing line ever got in the dev window, and how often even its heaviest favourites
    lost, so the reader can see the bound sits outside anything either supports.
    """
    from ufc_predictor.backtest.walkforward import load_panel

    lo, hi = (float(b) for b in cfg.lightgbm_fit["calibration_bounds"])
    panel = load_panel(cfg)
    dev = panel.loc[
        panel["scorable"]
        & (panel["date"] >= cfg.windows.dev_start)
        & (panel["date"] <= cfg.windows.dev_end)
    ]
    if dev["date"].max() > cfg.windows.dev_end:
        raise RuntimeError("calibration-bound evidence reached the evaluation window")
    m = dev["p1_market"].to_numpy(dtype=float)
    y = dev["label"].to_numpy(dtype=float)
    fav = np.maximum(m, 1 - m)
    fav_lost = np.where(m >= 0.5, 1 - y, y)
    heavy = fav >= 0.85
    top = np.argsort(-fav, kind="stable")[: max(1, len(fav) // 100)]
    return {
        "bounds": [lo, hi],
        "window": [str(cfg.windows.dev_start), str(cfg.windows.dev_end)],
        "n": len(dev),
        "market_min": float(m.min()),
        "market_max": float(m.max()),
        "market_outside_bounds": int(((m < lo) | (m > hi)).sum()),
        "favourites_at_or_above_0_85": {
            "n": int(heavy.sum()),
            "lost": float(fav_lost[heavy].mean()),
        },
        "top_1pct_favourites": {
            "n": len(top),
            "min_probability": float(fav[top].min()),
            "lost": float(fav_lost[top].mean()),
        },
        "cost_of_a_miss_at_the_bound_nats": float(-math.log(lo)),
        "cost_of_a_miss_unbounded_nats": float(-math.log(1e-6)),
    }


def run(cfg: Config | None = None, *, n_draws: int = 8, verbose: bool = True) -> dict:
    """Compute every diagnostic and write ``reports/leakage_diagnostics.json``."""
    cfg = cfg or get_config()
    tables = {name: db.read_table(cfg.paths.processed_dir, name) for name in db.TABLES}

    real = build(tables["fights"], tables, cfg)
    train, test = dev_only_split(real, cfg)
    control = _score(test, fit_predict_dev(train, test, cfg))

    draws = []
    for offset in range(n_draws):
        feats = build(shuffle_outcomes(tables["fights"], cfg.seed + offset), tables, cfg)
        s_train, s_test = dev_only_split(feats, cfg)
        draws.append(_score(s_test, fit_predict_dev(s_train, s_test, cfg)))
        if verbose:
            print(f"  shuffle {offset + 1}/{n_draws}: AUC {draws[-1]['auc']:.4f}")

    aucs = [d["auc"] for d in draws]
    lls = [d["log_loss"] for d in draws]
    out = {
        "dev_window": [str(cfg.windows.dev_start), str(cfg.windows.dev_end)],
        "split": {
            "train_span": [str(train["date"].min()), str(train["date"].max())],
            "test_span": [str(test["date"].min()), str(test["date"].max())],
            "n_train": len(train),
            "n_test": len(test),
        },
        "model": (
            f"LightGBM on the frozen hyperparameters, a fixed {DIAGNOSTIC_TREES} trees, "
            "no early stopping or calibration"
        ),
        "seed": cfg.seed,
        "chance_log_loss": math.log(2),
        "shuffle": {
            "n_draws": n_draws,
            "seeds": [cfg.seed + i for i in range(n_draws)],
            "aucs": aucs,
            "log_losses": lls,
            "mean_auc": float(np.mean(aucs)),
            "sd_auc": float(np.std(aucs, ddof=1)),
            "mean_log_loss": float(np.mean(lls)),
        },
        "control": control,
        "attribute_leak": attribute_leak_evidence(tables, real, cfg),
        "calibration_bound": calibration_bound_evidence(cfg),
    }
    path = cfg.paths.reports_dir / "leakage_diagnostics.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if verbose:
        sh = out["shuffle"]
        print(f"  shuffled: mean AUC {sh['mean_auc']:.4f}, mean log loss {sh['mean_log_loss']:.4f}")
        print(f"  control : AUC {control['auc']:.4f}, log loss {control['log_loss']:.4f}")
        print(f"  wrote {path}")
    return out
