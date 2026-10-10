"""Typed configuration loader.

Single source of truth for seeds, paths, windows, and hyperparameters. Every module
reads its settings from here so that nothing in ``src/`` hardcodes a seed or a path
(CLAUDE.md: "Seeds live in config, not scattered literals").

The loader deliberately *fails loudly*: a missing key or an unparseable date raises
rather than silently falling back to a default, because a silent default in a
backtest is indistinguishable from a correct run until the numbers are already wrong.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root = two parents up from this file (src/ufc_predictor/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing, malformed, or internally inconsistent."""


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    """Fetch ``key`` or raise. No silent defaults."""
    if key not in mapping:
        raise ConfigError(f"missing required config key {where}.{key!r}")
    return mapping[key]


def _parse_date(value: Any, where: str) -> dt.date:
    """Parse an ISO date, raising on anything ambiguous.

    PyYAML already turns unquoted ``2016-01-01`` into a ``datetime.date``, so accept
    that too -- but never accept a bare int or a partial date.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"{where}: cannot parse {value!r} as an ISO date") from exc
    raise ConfigError(f"{where}: expected an ISO date string, got {type(value).__name__}")


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    ufcstats_snapshot: Path
    odds_snapshot: Path
    aliases: Path


@dataclass(frozen=True)
class Windows:
    """Evaluation windows.

    ``dev_*`` is the only period hyperparameters may be tuned against; ``eval_*`` is
    the held-out walk-forward period. Keeping them disjoint is enforced in __post_init__
    so an edit to config.yaml cannot quietly create an overlap -- which would leak
    tuning information into the headline result.
    """

    dev_start: dt.date
    dev_end: dt.date
    eval_start: dt.date
    eval_end: dt.date

    def __post_init__(self) -> None:
        if self.dev_start >= self.dev_end:
            raise ConfigError(
                f"windows: dev_start {self.dev_start} must precede dev_end {self.dev_end}"
            )
        if self.eval_start >= self.eval_end:
            raise ConfigError(
                f"windows: eval_start {self.eval_start} must precede eval_end {self.eval_end}"
            )
        if self.dev_end >= self.eval_start:
            raise ConfigError(
                f"windows: dev_end {self.dev_end} must precede eval_start {self.eval_start}; "
                "overlapping windows would leak tuning information into the held-out result"
            )


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    windows: Windows
    features: dict[str, Any] = field(default_factory=dict)
    elo: dict[str, Any] = field(default_factory=dict)
    lightgbm: dict[str, Any] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)
    sim: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate ``config/config.yaml``.

    Args:
        path: Override the config location (used by tests).

    Raises:
        ConfigError: if the file is absent, malformed, or internally inconsistent.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: expected a top-level mapping, got {type(raw).__name__}")

    seed = _require(raw, "seed", "config")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"config.seed must be an int, got {seed!r}")

    raw_paths = _require(raw, "paths", "config")
    paths = Paths(
        raw_dir=REPO_ROOT / _require(raw_paths, "raw_dir", "paths"),
        processed_dir=REPO_ROOT / _require(raw_paths, "processed_dir", "paths"),
        reports_dir=REPO_ROOT / _require(raw_paths, "reports_dir", "paths"),
        ufcstats_snapshot=REPO_ROOT / _require(raw_paths, "ufcstats_snapshot", "paths"),
        odds_snapshot=REPO_ROOT / _require(raw_paths, "odds_snapshot", "paths"),
        aliases=REPO_ROOT / _require(raw_paths, "aliases", "paths"),
    )

    raw_windows = _require(raw, "windows", "config")
    windows = Windows(
        dev_start=_parse_date(_require(raw_windows, "dev_start", "windows"), "windows.dev_start"),
        dev_end=_parse_date(_require(raw_windows, "dev_end", "windows"), "windows.dev_end"),
        eval_start=_parse_date(
            _require(raw_windows, "eval_start", "windows"), "windows.eval_start"
        ),
        eval_end=_parse_date(_require(raw_windows, "eval_end", "windows"), "windows.eval_end"),
    )

    return Config(
        seed=seed,
        paths=paths,
        windows=windows,
        features=raw.get("features", {}),
        elo=raw.get("elo", {}),
        lightgbm=raw.get("lightgbm", {}),
        backtest=raw.get("backtest", {}),
        sim=raw.get("sim", {}),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached accessor for the default config, for use across a single run."""
    return load_config()
