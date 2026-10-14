"""Per-fold top-k feature selection, fitted on the fold's training rows only.

Feature selection is a classic leakage path: a selector that sees the rows it is scored on
picks columns that only *look* predictive there. :func:`select_top_k` uses nothing but the
``train`` frame it is handed. The walk-forward runner calls it once per refit with that
fold's training pool (every row dated strictly before the predicted event), so the
choice of columns is part of the fitted model, made fresh on every fold.

**The fit.** A LightGBM with the run's hyperparameters, early-stopped on the chronological
tail of ``train`` exactly as the model itself is (:class:`LightGBMModel`), uncalibrated --
calibration cannot move a split, so it cannot move a gain. Importance is the total gain of
each column over the trees the early-stopped model actually uses.

**Groups.** A matchup feature comes as ``<base>_1``, ``<base>_2`` and ``<base>_diff``;
keeping one fighter's side without the other's would make the model depend on which
fighter is listed first. So columns are ranked as groups (summed gain, ties by base name),
whole groups are taken from the top of the ranking until the next one does not fit, and
only then is a group split -- ``_diff`` first, because a difference alone is antisymmetric
under swapping the fighters. A low-ranked small group never jumps ahead of a better
group's ``_diff`` just because it fits; zero-gain groups rank last and are used only when
needed to reach ``k``. The result is exactly ``k`` columns, in the order they appear in
``features``.
"""

from __future__ import annotations

import pandas as pd

from ufc_predictor.config import Config
from ufc_predictor.models.lgbm import LightGBMModel

#: Suffixes that make a column a member of a ``<base>`` group, in the order pass 3 adds a
#: partial group's remaining members. A column with none of them is its own group.
MEMBER_ORDER = ("_diff", "_1", "_2")


def _split(column: str) -> tuple[str, int]:
    """(base name, member rank) -- rank ``len(MEMBER_ORDER)`` for a bare column."""
    for rank, suffix in enumerate(MEMBER_ORDER):
        if column.endswith(suffix) and len(column) > len(suffix):
            return column[: -len(suffix)], rank
    return column, len(MEMBER_ORDER)


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    """Base name -> its present columns, each list in ``features`` order.

    ``elo_1/elo_2/elo_diff`` form the group ``elo``; a column with no siblings --
    ``title_bout``, ``glicko_rd_sum``, a lone ``glicko_expected_1`` -- is a group of one.

    Raises:
        ValueError: on a duplicate column, or a bare column whose name is also another
            group's base (``x`` beside ``x_1``), which would make the grouping ambiguous.
    """
    seen: set[str] = set()
    groups: dict[str, list[str]] = {}
    bare: set[str] = set()
    for column in features:
        if column in seen:
            raise ValueError(f"duplicate feature column {column!r}")
        seen.add(column)
        base, rank = _split(column)
        if base in groups and (rank == len(MEMBER_ORDER) or base in bare):
            raise ValueError(f"ambiguous grouping: {column!r} and {groups[base]} share {base!r}")
        if rank == len(MEMBER_ORDER):
            bare.add(base)
        groups.setdefault(base, []).append(column)
    return groups


def gain_importance(
    train: pd.DataFrame, features: list[str], cfg: Config, *, params: dict | None = None
) -> dict[str, float]:
    """Total split gain of each column, from a LightGBM fitted on ``train`` alone.

    ``params=None`` means ``cfg.lightgbm``. ``train`` must be date-ordered and carry
    ``date`` and ``label``; the model raises otherwise.
    """
    fit = cfg.lightgbm_fit
    model = LightGBMModel(
        seed=cfg.seed,
        params=dict(cfg.lightgbm) if params is None else params,
        feature_cols=list(features),
        validation_fraction=fit["validation_fraction"],
        early_stopping_rounds=fit["early_stopping_rounds"],
        min_validation_rows=fit["min_validation_rows"],
        calibrate=False,
    )
    model.fit(train, train["label"].astype(int))
    booster = model._booster.booster_
    names = booster.feature_name()
    if names != list(features):
        raise RuntimeError(f"booster features {names[:5]}... do not match the requested columns")
    gains = booster.feature_importance(importance_type="gain", iteration=model.best_iteration_)
    return {name: float(g) for name, g in zip(names, gains, strict=True)}


def pick_top_k(features: list[str], gains: dict[str, float], k: int) -> list[str]:
    """Exactly ``k`` columns of ``features`` by grouped gain (ruling K2).

    1. Whole groups in rank order, stopping at the first group that does not fit (it is
       not skipped in favour of a smaller, lower-ranked group).
    2. Then the ``_diff`` of each group not yet taken, in rank order.
    3. Then the remaining members of incomplete groups, in rank order, ``_diff`` before
       ``_1`` before ``_2`` before a bare column.

    Groups rank by summed gain, descending, ties by base name ascending. With ``k`` at or
    above ``len(features)`` the features come back unchanged.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    groups = feature_groups(features)
    if k >= len(features):
        return list(features)
    missing = [c for c in features if c not in gains]
    if missing:
        raise KeyError(f"no gain for {len(missing)} columns, e.g. {missing[:5]}")

    ranked = sorted(groups, key=lambda base: (-sum(gains[c] for c in groups[base]), base))
    chosen: set[str] = set()
    for base in ranked:  # pass 1
        if len(chosen) + len(groups[base]) > k:
            break
        chosen.update(groups[base])
    for base in ranked:  # pass 2
        if len(chosen) == k:
            break
        diff = f"{base}_diff"
        if diff in groups[base] and diff not in chosen:
            chosen.add(diff)
    for base in ranked:  # pass 3
        for column in sorted(groups[base], key=lambda c: _split(c)[1]):
            if len(chosen) == k:
                break
            chosen.add(column)

    picked = [c for c in features if c in chosen]
    if len(picked) != k:
        raise RuntimeError(f"selected {len(picked)} columns, expected exactly {k}")
    return picked


def select_top_k(
    train: pd.DataFrame, features: list[str], k: int, cfg: Config, *, params: dict | None = None
) -> list[str]:
    """The ``k`` best ``features`` by grouped LightGBM gain on ``train`` alone.

    ``params=None`` means ``cfg.lightgbm``. Deterministic under ``cfg.seed``. Returns the
    columns in ``features`` order; with ``k >= len(features)``, ``features`` unchanged and
    no model is fitted.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    feature_groups(features)  # validates before any fit
    if k >= len(features):
        return list(features)
    return pick_top_k(features, gain_importance(train, features, cfg, params=params), k)
