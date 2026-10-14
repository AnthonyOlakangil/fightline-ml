"""The development walk-forward: one configuration, scored on the selection window or 2015.

The same protocol as evaluation run 3 (``backtest/walkforward.py``), restricted to the
development window and generalized to the Phase 8 options in :class:`ExperimentSpec`:

* train only on decisive fights dated **strictly before** each predicted event (the
  pool includes pre-2010 and unpriced fights), refit every
  ``backtest.retrain_every_n_events`` scored events with at least
  ``backtest.min_train_fights`` fights;
* score only priced, decisive fights inside the requested window, and pass every scored
  date through :func:`protocol.assert_scorable` before returning, so nothing in the
  closed evaluation window can come back from here;
* every model choice -- feature families, per-fold top-k selection, training window,
  orientation symmetry, calibration, LightGBM or logistic -- is fitted inside the fold's
  training rows only.

Events sharing a date never see each other: the training filter is on the date, not the
event, so a second card on the same night is outside every model that predicts the first.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from ufc_predictor.config import Config
from ufc_predictor.experiments.protocol import Purpose, assert_scorable, window_for
from ufc_predictor.features.pit import FEATURE_FAMILIES, build_features
from ufc_predictor.models.base import Model, validate_predictions
from ufc_predictor.models.calibration import CALIBRATIONS
from ufc_predictor.models.lgbm import LightGBMModel
from ufc_predictor.models.logistic import LogisticModel
from ufc_predictor.models.symmetric import SymmetricModel
from ufc_predictor.storage import db

TRAIN_WINDOWS = ("all_history", "from_2001", "last_8_years")
MODELS = ("lightgbm", "logistic")
V2_TABLE = "features_v2"


@dataclass(frozen=True)
class ExperimentSpec:
    """One development configuration. The defaults are evaluation run 3's model."""

    features: tuple[str, ...] = ("base",)
    top_k: int | None = None
    train_window: str = "all_history"
    symmetric: bool = False
    calibration: str = "isotonic_bounded"
    model: str = "lightgbm"
    #: LightGBM hyperparameters as sorted (key, value) pairs; None means config's.
    lightgbm_params: tuple[tuple[str, Any], ...] | None = field(default=None)

    def __post_init__(self) -> None:
        unknown = [f for f in self.features if f not in FEATURE_FAMILIES]
        if unknown or "base" not in self.features:
            raise ValueError(
                f"feature family list {self.features} must include base; unknown {unknown}"
            )
        if self.train_window not in TRAIN_WINDOWS:
            raise ValueError(f"train_window {self.train_window!r} not in {TRAIN_WINDOWS}")
        if self.calibration not in CALIBRATIONS:
            raise ValueError(f"calibration {self.calibration!r} not in {CALIBRATIONS}")
        if self.model not in MODELS:
            raise ValueError(f"model {self.model!r} not in {MODELS}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive")

    def columns(self) -> list[str]:
        return [
            c for fam in FEATURE_FAMILIES if fam in self.features for c in FEATURE_FAMILIES[fam]
        ]

    def params(self, cfg: Config) -> dict:
        if self.lightgbm_params is None:
            return dict(cfg.lightgbm)
        return dict(self.lightgbm_params)

    def with_params(self, params: dict) -> ExperimentSpec:
        return _replace(self, lightgbm_params=tuple(sorted(params.items())))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = list(self.features)
        d["lightgbm_params"] = dict(self.lightgbm_params) if self.lightgbm_params else None
        return d

    def key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


def _replace(spec: ExperimentSpec, **changes) -> ExperimentSpec:
    from dataclasses import replace

    return replace(spec, **changes)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _panel(feats: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Decisive fights with event id and closing line (NaN when unpriced), date-ordered."""
    fights = db.read_table(cfg.paths.processed_dir, "fights")
    odds = db.read_table(cfg.paths.processed_dir, "odds")
    panel = feats.merge(fights[["fight_id", "event_id"]], on="fight_id", validate="one_to_one")
    panel = panel.merge(
        odds[["fight_id", "p1_market"]], on="fight_id", how="left", validate="one_to_one"
    )
    panel = panel.loc[panel["label"].notna()].copy()
    return panel.sort_values(["date", "event_id", "fight_id"], kind="stable").reset_index(drop=True)


def base_panel(cfg: Config) -> pd.DataFrame:
    """The run-3 feature table (base family only) as a panel."""
    return _panel(db.read_table(cfg.paths.processed_dir, "features"), cfg)


def family_kwargs(cfg: Config) -> dict:
    """Family parameters passed to ``build_features``; all fixed a priori in config."""
    return {
        "recency_half_life_fights": float(cfg.features["recency_half_life_fights"]),
        "glicko_params": dict(cfg.glicko2),
    }


def build_v2_features(cfg: Config) -> pd.DataFrame:
    """Every feature family, written to ``data/processed/features_v2.parquet``.

    ``features.parquet`` (the base family, read by evaluation run 3 and the demo) is
    left untouched.
    """
    tables = {
        t: db.read_table(cfg.paths.processed_dir, t) for t in ("fights", "fight_stats", "fighters")
    }
    feats = build_features(
        tables["fights"],
        tables["fight_stats"],
        tables["fighters"],
        rolling_windows=tuple(cfg.features["rolling_windows"]),
        elo_params=cfg.elo,
        families=tuple(FEATURE_FAMILIES),
        **family_kwargs(cfg),
    )
    db.write_table(feats, cfg.paths.processed_dir, V2_TABLE)
    return feats


def family_panel(cfg: Config, *, rebuild: bool = True) -> pd.DataFrame:
    feats = build_v2_features(cfg) if rebuild else db.read_table(cfg.paths.processed_dir, V2_TABLE)
    return _panel(feats, cfg)


# ---------------------------------------------------------------------------
# Models and training pools
# ---------------------------------------------------------------------------


def make_model(spec: ExperimentSpec, cfg: Config, columns: list[str]) -> Model:
    fit = cfg.lightgbm_fit
    if spec.model == "lightgbm":
        inner: Model = LightGBMModel(
            seed=cfg.seed,
            params=spec.params(cfg),
            feature_cols=columns,
            validation_fraction=float(fit["validation_fraction"]),
            early_stopping_rounds=int(fit["early_stopping_rounds"]),
            calibrate=True,
            min_validation_rows=int(fit["min_validation_rows"]),
            calibration_bounds=tuple(float(b) for b in fit["calibration_bounds"]),
            calibration=spec.calibration,
        )
    else:
        inner = LogisticModel(
            features=columns, seed=cfg.seed, validation_fraction=float(fit["validation_fraction"])
        )
    return SymmetricModel(inner) if spec.symmetric else inner


def _years_before(day: dt.date, years: int) -> dt.date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29 February
        return day.replace(year=day.year - years, day=28)


def training_pool(
    panel: pd.DataFrame, event_date: dt.date, spec: ExperimentSpec, min_train: int
) -> tuple[pd.DataFrame, bool]:
    """Decisive fights before ``event_date`` under the spec's training window.

    A restricted window holding fewer than ``min_train`` fights is extended back to the
    most recent ``min_train`` (pre-registered floor). Returns the pool, date-ordered, and
    whether the floor was applied.

    Raises:
        RuntimeError: if even the full history before the event is below ``min_train``.
    """
    pool = panel.loc[panel["date"] < event_date].sort_values(["date", "fight_id"], kind="stable")
    if len(pool) < min_train:
        raise RuntimeError(
            f"only {len(pool)} decisive fights before {event_date}; need {min_train}"
        )
    if spec.train_window == "all_history":
        return pool, False
    start = (
        dt.date(2001, 1, 1) if spec.train_window == "from_2001" else _years_before(event_date, 8)
    )
    restricted = pool.loc[pool["date"] >= start]
    if len(restricted) >= min_train:
        return restricted, False
    return pool.iloc[-min_train:], True


def assert_window_scores(frame: pd.DataFrame, cfg: Config, window: Purpose) -> None:
    assert_scorable(frame["date"], cfg, purpose=window)


# ---------------------------------------------------------------------------
# The walk-forward
# ---------------------------------------------------------------------------


def run_config(
    cfg: Config,
    spec: ExperimentSpec,
    *,
    window: Purpose,
    panel: pd.DataFrame,
    through: dt.date | None = None,
) -> pd.DataFrame:
    """Out-of-sample predictions for every priced decisive fight in ``window``.

    Returns one row per fight: ids, date, label, ``p_market``, ``p_model`` and the fold's
    training facts. ``attrs["folds"]`` lists each refit (cutoff, rows, features used).
    """
    start, end = window_for(window, cfg)
    if through is not None:
        end = min(end, through)
    cadence = int(cfg.backtest["retrain_every_n_events"])
    min_train = int(cfg.backtest["min_train_fights"])
    columns = spec.columns()
    missing = [c for c in columns if c not in panel.columns]
    if missing:
        raise KeyError(f"panel lacks {len(missing)} feature columns, e.g. {missing[:5]}")

    scorable = panel.loc[
        panel["p1_market"].notna() & (panel["date"] >= start) & (panel["date"] <= end)
    ]
    events = (
        scorable.groupby("event_id")["date"]
        .min()
        .reset_index()
        .sort_values(["date", "event_id"], kind="stable")
    )

    model: Model | None = None
    fold = -1
    folds: list[dict] = []
    rows: list[pd.DataFrame] = []
    for scored_events, ev in enumerate(events.itertuples(index=False)):
        if model is None or scored_events % cadence == 0:
            fold += 1
            train, floored = training_pool(panel, ev.date, spec, min_train)
            if train["date"].max() >= ev.date:
                raise AssertionError(f"fold {fold} trains on {train['date'].max()} for {ev.date}")
            used = columns
            if spec.top_k is not None:
                from ufc_predictor.experiments.selection import select_top_k

                used = select_top_k(train, columns, spec.top_k, cfg, params=spec.params(cfg))
            model = make_model(spec, cfg, used)
            model.fit(train, train["label"].astype(int))
            folds.append(
                {
                    "fold": fold,
                    "cutoff": ev.date,
                    "train_rows": len(train),
                    "train_min_date": train["date"].min(),
                    "train_max_date": train["date"].max(),
                    "floor_extended": floored,
                    "features": list(used),
                }
            )
        test = scorable.loc[scorable["event_id"] == ev.event_id]
        p = validate_predictions(model.predict_proba(test), len(test), spec.model)
        f = folds[-1]
        rows.append(
            pd.DataFrame(
                {
                    "fight_id": test["fight_id"].to_numpy(),
                    "event_id": test["event_id"].to_numpy(),
                    "date": test["date"].to_numpy(),
                    "label": test["label"].to_numpy(dtype=float),
                    "p_market": test["p1_market"].to_numpy(dtype=float),
                    "p_model": p,
                    "fold": f["fold"],
                    "train_rows": f["train_rows"],
                    "train_max_date": f["train_max_date"],
                    "floor_extended": f["floor_extended"],
                }
            )
        )
    if not rows:
        raise RuntimeError(f"no scorable fights in the {window} window up to {end}")
    out = pd.concat(rows, ignore_index=True)
    if out["fight_id"].duplicated().any():
        raise RuntimeError("a fight was predicted twice")
    assert_window_scores(out, cfg, window)
    out.attrs["folds"] = folds
    return out
