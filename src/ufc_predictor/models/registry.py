"""Model construction from config.

A single factory so the backtest, the CLI and the demo all build identically-configured
models. If each constructed its own, they could silently diverge -- a different seed in
the app than in the backtest would make the published numbers unreproducible.
"""

from __future__ import annotations

from ufc_predictor.config import Config, get_config
from ufc_predictor.models.base import Model
from ufc_predictor.models.elo_model import EloModel
from ufc_predictor.models.lgbm import LightGBMModel
from ufc_predictor.models.market import MarketModel

#: Order used in reports: baseline first, then increasing sophistication.
MODEL_NAMES = ("market", "elo", "lightgbm")


def build_model(name: str, cfg: Config | None = None) -> Model:
    """Construct one model by name, with every parameter and seed taken from config."""
    cfg = cfg or get_config()

    if name == "market":
        return MarketModel()
    if name == "elo":
        return EloModel(seed=cfg.seed)
    if name == "lightgbm":
        fit_cfg = cfg.lightgbm_fit
        return LightGBMModel(
            seed=cfg.seed,
            params=cfg.lightgbm,
            validation_fraction=float(fit_cfg.get("validation_fraction", 0.2)),
            early_stopping_rounds=int(fit_cfg.get("early_stopping_rounds", 50)),
            calibrate=bool(fit_cfg.get("calibrate", True)),
            min_validation_rows=int(fit_cfg.get("min_validation_rows", 300)),
            calibration_bounds=tuple(float(b) for b in fit_cfg["calibration_bounds"]),
        )
    raise KeyError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def build_all(cfg: Config | None = None) -> dict[str, Model]:
    """Every model, keyed by name, in report order."""
    cfg = cfg or get_config()
    return {name: build_model(name, cfg) for name in MODEL_NAMES}
