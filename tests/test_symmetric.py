"""Orientation symmetry: p(A beats B) + p(B beats A) = 1 exactly, by construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.features.pit import FEATURE_COLS
from ufc_predictor.models.lgbm import LightGBMModel
from ufc_predictor.models.registry import build_model
from ufc_predictor.models.symmetric import SymmetricModel, mirror
from ufc_predictor.storage import db


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def selection_frame(cfg):
    """Labelled fights through the selection window only (never 2015 or later)."""
    try:
        feats = db.read_table(cfg.paths.processed_dir, "features")
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data && make features`")
    df = feats.loc[feats["label"].notna() & (feats["date"] <= cfg.windows.selection_end)]
    return df.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)


def test_mirror_swaps_sides_negates_diffs_and_flips_the_label():
    X = pd.DataFrame(
        {
            "elo_1": [1600.0],
            "elo_2": [1500.0],
            "elo_diff": [100.0],
            "title_bout": [True],
            "glicko_expected_1": [0.7],
            "label": [1.0],
            "date": [pd.Timestamp("2012-01-01").date()],
        }
    )
    M = mirror(X)
    assert (M["elo_1"].iloc[0], M["elo_2"].iloc[0], M["elo_diff"].iloc[0]) == (
        1500.0,
        1600.0,
        -100.0,
    )
    assert bool(M["title_bout"].iloc[0]) is True
    assert M["glicko_expected_1"].iloc[0] == pytest.approx(0.3)
    assert M["label"].iloc[0] == 0.0
    pd.testing.assert_frame_equal(mirror(M), X)


def test_mirror_refuses_a_column_it_cannot_orient():
    X = pd.DataFrame({"mystery_feature": [1.0]})
    with pytest.raises(KeyError, match="orientation"):
        mirror(X)


def test_symmetric_model_is_exactly_symmetric(selection_frame, cfg):
    train = selection_frame.iloc[:-300]
    test = selection_frame.iloc[-300:]
    m = SymmetricModel(build_model("lightgbm", cfg)).fit(train, train["label"])
    p = m.predict_proba(test)
    p_swapped = m.predict_proba(mirror(test))
    np.testing.assert_allclose(p + p_swapped, 1.0, atol=1e-12)


def test_an_ordinary_model_is_not_symmetric(selection_frame, cfg):
    """The wrapper is doing work: the inner model alone breaks the identity."""
    train = selection_frame.iloc[:-300]
    test = selection_frame.iloc[-300:]
    inner = build_model("lightgbm", cfg).fit(train, train["label"])
    gap = np.abs(inner.predict_proba(test) + inner.predict_proba(mirror(test)) - 1.0)
    assert gap.max() > 1e-3


def test_symmetric_training_rows_stay_in_date_order(selection_frame, cfg):
    train = selection_frame.iloc[:2000]
    m = SymmetricModel(
        LightGBMModel(
            seed=cfg.seed, params=cfg.lightgbm, feature_cols=FEATURE_COLS, min_validation_rows=200
        )
    )
    m.fit(train, train["label"])  # LightGBMModel raises on unsorted rows
    assert m.n_training_rows_ == 2 * len(train)


@pytest.mark.slow
def test_swapping_the_fighters_rebuilds_exactly_the_mirror(cfg):
    """mirror() is only meaningful if it is what the builder produces for the swapped bout.

    Rebuild every feature family with fighter_1/fighter_2 (and their names) swapped on
    every bout, and compare with mirror() of the original build, column by column.
    """
    from ufc_predictor.experiments.runner import family_kwargs
    from ufc_predictor.features.pit import ALL_FEATURE_COLS, FEATURE_FAMILIES, build_features

    try:
        tables = {
            t: db.read_table(cfg.paths.processed_dir, t)
            for t in ("fights", "fight_stats", "fighters")
        }
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data`")
    kwargs = {
        "rolling_windows": tuple(cfg.features["rolling_windows"]),
        "elo_params": cfg.elo,
        "families": tuple(FEATURE_FAMILIES),
        **family_kwargs(cfg),
    }
    fights = tables["fights"]
    swapped = fights.copy()
    swapped[["fighter_1", "fighter_2"]] = fights[["fighter_2", "fighter_1"]].to_numpy()
    swapped[["name_1", "name_2"]] = fights[["name_2", "name_1"]].to_numpy()
    a = build_features(fights, tables["fight_stats"], tables["fighters"], **kwargs)
    b = build_features(swapped, tables["fight_stats"], tables["fighters"], **kwargs)
    a = a.set_index("fight_id").sort_index()
    b = b.set_index("fight_id").sort_index()
    expected = mirror(a[ALL_FEATURE_COLS])
    bad = []
    for col in ALL_FEATURE_COLS:
        x = pd.to_numeric(expected[col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(x, y, rtol=0, atol=1e-9, equal_nan=True):
            bad.append(col)
    assert not bad, f"columns that are not orientation-equivariant: {bad[:10]}"
    # And the label flips.
    lab_a, lab_b = a["label"], b["label"]
    both = lab_a.notna()
    assert ((lab_a[both] + lab_b[both]) == 1.0).all()
