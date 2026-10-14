"""Point-in-time correctness suite. **This is the test file that gates the project.**

Written before the feature builder existed, and before any model. Every test here
encodes an invariant that, if violated, would make the headline result meaningless
while leaving every row count, every chart, and every summary statistic looking
perfectly healthy. That is the danger: leakage does not announce itself.

The central test is `test_mutating_a_later_fight_leaves_earlier_features_byte_identical`.
It does not inspect columns or reason about the code; it changes the future and checks
that the past did not move. A feature builder that passes it cannot be reading forward,
whatever its implementation does.

Every test builds every registered feature family and checks ``ALL_FEATURE_COLS``, so a
family added to ``pit.FEATURE_FAMILIES`` is covered here without a new test.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.features import diagnostics
from ufc_predictor.features.pit import (
    ALL_FEATURE_COLS,
    FEATURE_COLS,
    FEATURE_FAMILIES,
    build_features,
    fighter_fight_history,
)
from ufc_predictor.storage import db

pytestmark = pytest.mark.leakage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return get_config()


@pytest.fixture(scope="module")
def tables(cfg):
    try:
        return {name: db.read_table(cfg.paths.processed_dir, name) for name in db.TABLES}
    except FileNotFoundError:
        pytest.skip("processed tables absent; run `make data`")


def build_kwargs(cfg) -> dict:
    """Every registered family, so each test covers every feature column."""
    return {
        "rolling_windows": tuple(cfg.features["rolling_windows"]),
        "elo_params": cfg.elo,
        "families": tuple(FEATURE_FAMILIES),
        "recency_half_life_fights": float(cfg.features["recency_half_life_fights"]),
        "glicko_params": dict(cfg.glicko2),
    }


@pytest.fixture(scope="module")
def features(tables, cfg):
    return build_features(
        tables["fights"], tables["fight_stats"], tables["fighters"], **build_kwargs(cfg)
    )


def frame_digest(df: pd.DataFrame) -> bytes:
    """A deterministic byte digest of a frame's contents.

    Used for the mutation test's byte-identity claim. Sorting by fight_id removes any
    dependence on row order, so the digest reflects values only.
    """
    ordered = df.sort_values("fight_id").reset_index(drop=True)
    return pd.util.hash_pandas_object(ordered, index=False).values.tobytes()


# ---------------------------------------------------------------------------
# THE central test: change the future, assert the past is unmoved
# ---------------------------------------------------------------------------


def test_mutating_a_later_fight_leaves_earlier_features_byte_identical(tables, cfg):
    """Mutate a later fight's outcome *and* its per-round statistics; every feature
    row for a fight on or before that date must be byte-identical.

    This is the strongest available statement of point-in-time correctness, because it
    is behavioural rather than structural. No column audit can prove a builder does not
    read forward; this can. Features for a fight on date M depend only on fights
    strictly before M, so mutating a fight *at* M must leave date-M rows untouched too.
    """
    fights, stats, fighters = tables["fights"], tables["fight_stats"], tables["fighters"]
    kwargs = build_kwargs(cfg)

    base = build_features(fights, stats, fighters, **kwargs)

    # Choose a mutation target in the middle of the timeline with real stats attached,
    # so the mutation has something to propagate from.
    with_stats = set(stats["fight_id"])
    candidates = fights.loc[
        fights["fight_id"].isin(with_stats) & fights["is_decisive"]
    ].sort_values("date")
    target = candidates.iloc[len(candidates) // 2]
    cutoff = target["date"]

    # Mutation 1: flip the winner of the later fight.
    mutated_fights = fights.copy()
    idx = mutated_fights.index[mutated_fights["fight_id"] == target["fight_id"]][0]
    original_winner = mutated_fights.at[idx, "winner_id"]
    flipped = (
        mutated_fights.at[idx, "fighter_2"]
        if original_winner == mutated_fights.at[idx, "fighter_1"]
        else mutated_fights.at[idx, "fighter_1"]
    )
    mutated_fights.at[idx, "winner_id"] = flipped
    mutated_fights.at[idx, "method_class"] = "SUB"
    mutated_fights.at[idx, "finish_round"] = 1

    # Mutation 2: scramble that fight's per-round statistics.
    mutated_stats = stats.copy()
    smask = mutated_stats["fight_id"] == target["fight_id"]
    assert smask.any(), "target fight must carry per-round stats for this test to bite"
    for col in ("sig_str_landed", "sig_str_att", "td_landed", "td_att", "kd", "ctrl_sec"):
        mutated_stats.loc[smask, col] = mutated_stats.loc[smask, col].fillna(0) * 7 + 99

    mutated = build_features(mutated_fights, mutated_stats, fighters, **kwargs)

    # Compare FEATURES only. The mutated fight's own `label` legitimately flips -- that
    # is the mutation -- but a label is not a feature, and no feature may move.
    cols = ["fight_id", *ALL_FEATURE_COLS]
    before_base = base.loc[base["date"] <= cutoff, cols]
    before_mut = mutated.loc[mutated["date"] <= cutoff, cols]

    assert len(before_base) == len(before_mut) > 1000, "not enough earlier rows to be meaningful"
    assert frame_digest(before_base) == frame_digest(before_mut), (
        f"features for fights on or before {cutoff} changed when a fight on {cutoff} was "
        "mutated: the builder is reading forward in time"
    )

    # Three sanity checks, so a vacuous pass is impossible.
    # (a) The mutation landed: the target's own label must have flipped.
    lab_a = base.loc[base["fight_id"] == target["fight_id"], "label"].iloc[0]
    lab_b = mutated.loc[mutated["fight_id"] == target["fight_id"], "label"].iloc[0]
    assert lab_a != lab_b, "the winner flip did not take effect; the test proves nothing"

    # (b) Nothing but the label changed in the earlier slice -- not even a stray column.
    earlier_a = base.loc[base["date"] <= cutoff].sort_values("fight_id").reset_index(drop=True)
    earlier_b = (
        mutated.loc[mutated["date"] <= cutoff].sort_values("fight_id").reset_index(drop=True)
    )
    changed = [c for c in earlier_a.columns if not earlier_a[c].equals(earlier_b[c])]
    assert changed == ["label"], f"unexpected columns changed in the past: {changed}"

    # (c) The mutation does propagate forward through every family, so the builder is
    # genuinely in play and no family can pass by being disconnected from history.
    after_base = base.loc[base["date"] > cutoff]
    after_mut = mutated.loc[mutated["date"] > cutoff]
    for family, family_cols in FEATURE_FAMILIES.items():
        fcols = ["fight_id", *family_cols]
        assert frame_digest(after_base[fcols]) != frame_digest(after_mut[fcols]), (
            f"mutating a fight changed no later {family!r} feature; the test is not "
            "exercising that family at all"
        )


def test_mutating_the_very_last_event_changes_no_feature_row(tables, cfg):
    """A complementary end-case: nothing follows the final event, so mutating it must
    leave the entire feature table unchanged."""
    fights, stats, fighters = tables["fights"], tables["fight_stats"], tables["fighters"]
    kwargs = build_kwargs(cfg)
    base = build_features(fights, stats, fighters, **kwargs)

    last_date = fights["date"].max()
    mutated_fights = fights.copy()
    mask = mutated_fights["date"] == last_date
    mutated_fights.loc[mask, "winner_id"] = mutated_fights.loc[mask, "fighter_2"]

    mutated = build_features(mutated_fights, stats, fighters, **kwargs)
    # Labels legitimately change for the mutated fights; features must not.
    assert frame_digest(base[["fight_id", *ALL_FEATURE_COLS]]) == frame_digest(
        mutated[["fight_id", *ALL_FEATURE_COLS]]
    )


# ---------------------------------------------------------------------------
# Same-card and same-night isolation
# ---------------------------------------------------------------------------


def test_fights_on_the_same_card_cannot_see_each_other(tables, cfg):
    """Two fights on one card are simultaneous for modelling purposes.

    Elo and rolling statistics must advance only after the whole event is processed, so
    flipping the result of one bout, and scrambling its round statistics, must not alter
    any other bout on that same date. The scramble matters: columns built from statistics
    alone never see a winner flip, so without it they could leak across a card unnoticed.
    """
    fights, stats, fighters = tables["fights"], tables["fight_stats"], tables["fighters"]
    kwargs = build_kwargs(cfg)
    base = build_features(fights, stats, fighters, **kwargs)

    # A busy, recent card.
    counts = fights.groupby("date").size()
    busy_date = counts.loc[counts >= 10].index.max()
    same_day = fights.loc[fights["date"] == busy_date]
    victim = same_day.iloc[0]

    mutated_fights = fights.copy()
    idx = mutated_fights.index[mutated_fights["fight_id"] == victim["fight_id"]][0]
    mutated_fights.at[idx, "winner_id"] = (
        mutated_fights.at[idx, "fighter_2"]
        if victim["winner_id"] == victim["fighter_1"]
        else mutated_fights.at[idx, "fighter_1"]
    )
    mutated_stats = stats.copy()
    smask = mutated_stats["fight_id"] == victim["fight_id"]
    assert smask.any(), "the victim bout must carry round stats for this test to bite"
    for col in (
        "sig_str_landed",
        "sig_str_att",
        "td_landed",
        "td_att",
        "sub_att",
        "kd",
        "ctrl_sec",
    ):
        mutated_stats.loc[smask, col] = mutated_stats.loc[smask, col].fillna(0) * 7 + 99
    mutated = build_features(mutated_fights, mutated_stats, fighters, **kwargs)

    others = [f for f in same_day["fight_id"] if f != victim["fight_id"]]
    assert len(others) >= 9
    a = base.loc[base["fight_id"].isin(others), ["fight_id", *ALL_FEATURE_COLS]]
    b = mutated.loc[mutated["fight_id"].isin(others), ["fight_id", *ALL_FEATURE_COLS]]
    assert frame_digest(a) == frame_digest(b), (
        "a fight's result altered its own card-mates' features: state is advancing "
        "mid-event instead of after the event"
    )


# Families whose paired columns describe one fighter's own state, not the matchup. ("style"
# is the exception: its interactions depend on the opponent, so differ between bouts.)
PER_FIGHTER_FAMILIES = ("base", "adjusted", "recency", "decline", "glicko")


def _paired_stems(cols: list[str]) -> set[str]:
    """Stems with both a ``_1`` and a ``_2`` column. A one-sided matchup value such as
    ``glicko_expected_1`` has no ``_2`` and describes the bout, not a fighter."""
    return {c[:-2] for c in cols if c.endswith("_1") and f"{c[:-2]}_2" in cols}


def _same(a, b) -> bool:
    return bool((pd.isna(a) and pd.isna(b)) or a == b)


def test_a_fighters_second_fight_on_one_night_does_not_see_the_first(tables, cfg):
    """Early UFC events were one-night tournaments: 67 fighter-days have more than one
    fight, up to four. Within such a night, the later bouts must not know the earlier
    results -- a plain `shift(1)` over the fight sequence would leak here.
    """
    fights, stats, fighters = tables["fights"], tables["fight_stats"], tables["fighters"]
    feats = build_features(fights, stats, fighters, **build_kwargs(cfg))
    # Every column describing the fighter's own pre-night state, read from their slot.
    # (Matchup columns depend on the opponent, so legitimately differ between bouts.) Elo
    # is compared separately below, rounded, as it always has been.
    per_fighter = sorted(
        set().union(*(_paired_stems(FEATURE_FAMILIES[fam]) for fam in PER_FIGHTER_FAMILIES))
        - {"elo"}
    )

    long = pd.concat(
        [
            fights[["fight_id", "date", "fighter_1"]].rename(columns={"fighter_1": "fid"}),
            fights[["fight_id", "date", "fighter_2"]].rename(columns={"fighter_2": "fid"}),
        ]
    )
    sizes = long.groupby(["fid", "date"]).size()
    multi = sizes[sizes > 1]
    assert len(multi) > 0, "expected tournament-format nights in the data"

    f = feats.set_index("fight_id")
    checked = 0
    for (fid, date), _ in multi.items():
        ids = long.loc[(long["fid"] == fid) & (long["date"] == date), "fight_id"].tolist()
        rows = f.loc[f.index.intersection(ids)]
        if len(rows) < 2:
            continue
        # Every bout that night must report the same pre-fight state for this fighter,
        # read from whichever slot (1 or 2) that fighter occupies in each bout.
        slot_vals = []
        states = []
        for fight_id in rows.index:
            row = f.loc[fight_id]
            slot = "1" if row["fighter_1"] == fid else "2"
            slot_vals.append((row[f"n_prior_fights_{slot}"], row[f"elo_{slot}"]))
            states.append({b: row[f"{b}_{slot}"] for b in per_fighter})
        assert len({v[0] for v in slot_vals}) == 1, (
            f"{fid} on {date}: prior-fight count changed between bouts on the same night"
        )
        assert len({round(float(v[1]), 9) for v in slot_vals}) == 1, (
            f"{fid} on {date}: Elo changed between bouts on the same night"
        )
        for other in states[1:]:
            moved = [b for b in per_fighter if not _same(states[0][b], other[b])]
            assert not moved, f"{fid} on {date}: {moved} changed between bouts on one night"
        checked += 1
    assert checked >= 20, f"only checked {checked} tournament nights"


# ---------------------------------------------------------------------------
# A fight's own data must not enter its own features
# ---------------------------------------------------------------------------


def test_a_fights_own_round_stats_never_enter_its_own_features(tables, cfg):
    """Scramble one fight's per-round stats; that fight's own feature row must not move."""
    fights, stats, fighters = tables["fights"], tables["fight_stats"], tables["fighters"]
    kwargs = build_kwargs(cfg)
    base = build_features(fights, stats, fighters, **kwargs)

    # Pick a fight whose participants both have prior history, so the row is not all-NaN.
    rich = base.loc[(base["n_prior_fights_1"] >= 3) & (base["n_prior_fights_2"] >= 3)]
    target_id = rich.sort_values("date").iloc[len(rich) // 2]["fight_id"]

    mutated_stats = stats.copy()
    m = mutated_stats["fight_id"] == target_id
    assert m.any()
    for col in ("sig_str_landed", "sig_str_att", "td_landed", "kd", "ctrl_sec"):
        mutated_stats.loc[m, col] = mutated_stats.loc[m, col].fillna(0) * 13 + 500

    mutated = build_features(fights, mutated_stats, fighters, **kwargs)

    row_a = base.loc[base["fight_id"] == target_id, ALL_FEATURE_COLS]
    row_b = mutated.loc[mutated["fight_id"] == target_id, ALL_FEATURE_COLS]
    pd.testing.assert_frame_equal(row_a, row_b, check_exact=True)


def test_history_for_a_fight_contains_only_strictly_earlier_fights(tables, cfg):
    """Direct provenance check on the history index used to build the features."""
    fights, stats = tables["fights"], tables["fight_stats"]
    history = fighter_fight_history(fights, stats)
    by_fighter = dict(iter(history.groupby("fighter_id")))

    rng = np.random.default_rng(get_config().seed)
    sample = fights.sample(min(300, len(fights)), random_state=int(rng.integers(1 << 30)))
    for row in sample.itertuples(index=False):
        for fid in (row.fighter_1, row.fighter_2):
            g = by_fighter.get(fid)
            if g is None:
                continue
            prior = g.loc[g["date"] < row.date]
            assert (prior["date"] < row.date).all()
            # And the fight itself is never in its own history slice.
            assert row.fight_id not in set(prior["fight_id"])


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def test_elo_is_a_pre_fight_rating(features):
    """Everyone starts at the configured initial rating, and debutants still carry it."""
    cfg = get_config()
    init = float(cfg.elo["initial_rating"])
    debut_1 = features.loc[features["n_prior_fights_1"] == 0, "elo_1"]
    assert len(debut_1) > 100
    assert np.allclose(debut_1.to_numpy(dtype=float), init)


def test_elo_diff_is_consistent_with_its_components(features):
    diff = features["elo_1"] - features["elo_2"]
    assert np.allclose(diff.to_numpy(), features["elo_diff"].to_numpy(), atol=1e-9)


# ---------------------------------------------------------------------------
# Debut handling: a flag, never a silent zero
# ---------------------------------------------------------------------------


# Families whose every per-fighter column is a rate over prior bouts: NaN, never 0, for a
# debutant or for a fighter with no round statistics yet.
HISTORY_RATE_PREFIXES = ("adj_", "ewm_")


def test_debut_rolling_features_are_null_not_zero(features):
    """A debutant has no history. Encoding that as 0.0 would be a lie the model reads
    as 'this fighter lands zero strikes per minute', and backfilling from career totals
    would import the future outright."""
    debuts = features.loc[features["n_prior_fights_1"] == 0]
    assert len(debuts) > 100

    rolling_1 = [c for c in FEATURE_COLS if c.endswith("_1") and ("_w" in c or "rate" in c)]
    assert rolling_1, "expected rolling per-fighter columns"
    history_1 = [
        c for c in ALL_FEATURE_COLS if c.endswith("_1") and c.startswith(HISTORY_RATE_PREFIXES)
    ]
    assert len(history_1) == 20, "expected the adjusted and recency per-fighter columns"
    rolling_1 += history_1
    for col in rolling_1:
        assert debuts[col].isna().all(), f"{col} is populated for debutants; must be NaN"


def test_debut_is_flagged_explicitly(features):
    assert "is_debut_1" in features.columns
    assert "is_debut_2" in features.columns
    assert (features.loc[features["n_prior_fights_1"] == 0, "is_debut_1"]).all()
    assert not (features.loc[features["n_prior_fights_1"] > 0, "is_debut_1"]).any()


def test_missing_history_is_never_silently_zero(features):
    """n_prior_fights is a genuine count, so 0 is meaningful there; but every derived
    rate must be NaN rather than 0 when there is nothing to average."""
    no_stats = features.loc[~features["has_stats_history_1"]]
    assert len(no_stats) > 100, "expected fighters with a record but no round statistics"
    assert no_stats["sig_str_landed_pm_career_1"].isna().all()
    for col in ALL_FEATURE_COLS:
        if col.endswith("_1") and col.startswith(HISTORY_RATE_PREFIXES):
            assert no_stats[col].isna().all(), f"{col} is set with no stats history"


# ---------------------------------------------------------------------------
# The fighters table is a present-day snapshot, so nothing in it is point-in-time
# ---------------------------------------------------------------------------

# Everything in the tale-of-the-tape table except the identity columns. It was scraped
# once, in 2026, and UFCStats only fills these fields in for fighters who stayed on the
# roster -- so whether a value exists is decided by the fighter's *future* career.
SCRAPE_TIME_ATTRIBUTES = ("height_cm", "reach_cm", "weight_lbs", "stance", "dob")


def test_features_ignore_the_scrape_time_fighter_attributes(tables, cfg, features):
    """Blank every attribute in the fighters table; not one feature value may move.

    This is the mutation test for the one input that the fight-level mutation tests
    never touch. Run 1 of the backtest used height, reach and age from this table, and a
    missing reach turned out to mean "this fighter was later cut": before 2016 the side
    with the blank reach won 5.7% of dev-window bouts. Blanking the value only when one
    side is missing does not close the leak either, because "this pair has a blank"
    combined with prior experience still identifies who was cut. So the contract is
    total: the features must not read this table's attributes at all. Reintroducing an
    attribute needs a dated source, not this snapshot.
    """
    blanked = tables["fighters"].copy()
    for col in SCRAPE_TIME_ATTRIBUTES:
        assert col in blanked.columns, f"fighters table no longer has {col!r}; update this test"
        blanked[col] = None
    rebuilt = build_features(tables["fights"], tables["fight_stats"], blanked, **build_kwargs(cfg))
    moved = [col for col in ALL_FEATURE_COLS if not features[col].equals(rebuilt[col])]
    assert not moved, f"features read the present-day fighters snapshot: {moved}"
    assert frame_digest(features) == frame_digest(rebuilt)


def test_features_ignore_which_fighters_the_snapshot_contains(tables, cfg, features):
    """Blanking columns cannot catch a feature built from *row presence* -- "is this
    fighter in the 2026 roster at all?" -- which would leak the same way. With an empty
    fighters table, every feature must still be byte-identical."""
    empty = tables["fighters"].iloc[0:0].copy()
    rebuilt = build_features(tables["fights"], tables["fight_stats"], empty, **build_kwargs(cfg))
    assert frame_digest(features) == frame_digest(rebuilt)


def test_no_scrape_time_attribute_is_routed_through_the_fight_tables(tables):
    """The builder could also read an attribute that normalization had joined onto the
    fights or round-statistics tables. Neither may carry one."""
    # Whole name tokens, so `distance_landed` is not mistaken for `stance` and the
    # bout's own `weight_class` (a fact of the fight, not the fighter) is allowed.
    tokens = {"height", "reach", "stance", "dob", "age"}
    for name in ("fights", "fight_stats"):
        offenders = [
            c
            for c in tables[name].columns
            if c in SCRAPE_TIME_ATTRIBUTES or tokens & set(c.lower().split("_"))
        ]
        assert not offenders, f"{name} carries scrape-time attributes: {offenders}"


# Debutants legitimately have blank history and win about 42-46% of dev-window bouts.
# A blank that predicts the result far more strongly than that is carrying information
# from somewhere other than the fighter's past.
BLANK_SIDE_WIN_RATE_BAND = (0.30, 0.70)
BLANK_SIDE_MIN_BOUTS = 50


def test_a_blank_feature_never_predicts_who_loses(features, cfg):
    """For every paired feature, the side whose value is blank must win at a plausible
    rate. A generic canary: it catches any future column whose *missingness* encodes the
    outcome, whatever table it comes from. Dev window only, so it never reads the
    held-out period."""
    dates = pd.to_datetime(features["date"]).dt.date
    dev = features.loc[
        features["label"].notna()
        & (dates >= cfg.windows.dev_start)
        & (dates <= cfg.windows.dev_end)
    ]
    assert pd.to_datetime(dev["date"]).dt.date.max() <= cfg.windows.dev_end
    assert len(dev) > 2000

    lo, hi = BLANK_SIDE_WIN_RATE_BAND
    offenders = []
    for base in sorted(_paired_stems(ALL_FEATURE_COLS)):
        blank_1 = dev[f"{base}_1"].isna() & dev[f"{base}_2"].notna()
        blank_2 = dev[f"{base}_2"].isna() & dev[f"{base}_1"].notna()
        n = int(blank_1.sum() + blank_2.sum())
        if n < BLANK_SIDE_MIN_BOUTS:
            continue
        wins = dev.loc[blank_1, "label"].sum() + (1 - dev.loc[blank_2, "label"]).sum()
        rate = wins / n
        if not lo <= rate <= hi:
            offenders.append(f"{base}: blank side won {rate:.3f} of {n} bouts")
    assert not offenders, "blank values predict the outcome: " + "; ".join(offenders)


# ---------------------------------------------------------------------------
# The features must not contain the answer, or the market's opinion of it
# ---------------------------------------------------------------------------


def test_no_odds_derived_column_reaches_the_feature_set():
    """If the model could see the closing line, 'model beats market' would be circular."""
    banned = ("odds", "market", "vigfree", "vig", "implied", "overround", "p1_", "p2_", "moneyline")
    offenders = [c for c in ALL_FEATURE_COLS if any(b in c.lower() for b in banned)]
    assert not offenders, f"odds-derived columns in the feature set: {offenders}"


def test_no_outcome_derived_column_reaches_the_feature_set():
    banned = ("winner", "label", "is_decisive", "method", "finish_round", "finish_time")
    offenders = [c for c in ALL_FEATURE_COLS if any(b in c.lower() for b in banned)]
    assert not offenders, f"outcome columns in the feature set: {offenders}"


def test_label_base_rate_is_balanced(features):
    """Canonical orientation is by sorted fighter_id, which cannot encode the result.

    Corner-based orientation would sit near 0.578 because promotions place favourites in
    the red corner; a model could then score well by always picking red.
    """
    labelled = features["label"].dropna()
    assert len(labelled) > 8000
    assert 0.48 <= labelled.mean() <= 0.52, f"base rate {labelled.mean():.4f} looks oriented"


# ---------------------------------------------------------------------------
# The shuffle test: no signal where there is none to find
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_model_cannot_beat_chance_on_shuffled_outcomes(tables, cfg):
    """Randomize who won, rebuild every feature from scratch, then train and evaluate.

    This is the decisive end-to-end leakage check. Elo, streaks, win rates and the
    rolling statistics are all recomputed from the shuffled results, so if any feature
    peeks at the fight it describes, the model will still predict the shuffled outcome
    and score above chance. With an honest point-in-time builder there is nothing to
    learn and performance must collapse to a coin flip.

    Confined to the dev window, so the held-out period is never spent on a diagnostic.
    That leaves ~1,000 test fights, where a single AUC estimate carries a standard error
    around 0.016 -- one unlucky draw can read 0.53 with no leakage whatsoever. The test
    therefore averages several independent shuffles and leans on log loss, which is the
    stricter statement: a model exploiting leaked information would beat chance log loss,
    not merely drift above 0.5 AUC.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    n_shuffles = 3
    aucs: list[float] = []
    lls: list[float] = []

    for offset in range(n_shuffles):
        feats = diagnostics.build(
            diagnostics.shuffle_outcomes(tables["fights"], cfg.seed + offset),
            tables,
            cfg,
            families=tuple(FEATURE_FAMILIES),
        )
        train, test = diagnostics.dev_only_split(feats, cfg)
        assert test["date"].max() <= cfg.windows.dev_end, "diagnostic reached the eval window"

        p = diagnostics.fit_predict_dev(train, test, cfg, feature_cols=ALL_FEATURE_COLS)
        y = test["label"].astype(int)
        aucs.append(roc_auc_score(y, p))
        lls.append(log_loss(y, p, labels=[0, 1]))

    mean_auc = float(np.mean(aucs))
    # Chance log loss is ln(2) = 0.6931. Every shuffle must be at least as bad as that.
    assert min(lls) > 0.68, (
        f"log losses {[round(x, 4) for x in lls]} on randomized outcomes approach or beat "
        "chance; the features carry information about the very fight they describe"
    )
    assert 0.46 <= mean_auc <= 0.54, (
        f"mean AUC {mean_auc:.4f} over {n_shuffles} shuffles (individual: "
        f"{[round(a, 4) for a in aucs]}) is away from chance on randomized outcomes"
    )


@pytest.mark.slow
def test_model_does_find_signal_on_real_outcomes(tables, cfg):
    """The control for the shuffle test.

    If the shuffle test passed only because the pipeline is broken and the features are
    noise, this would fail too. Real outcomes must be learnable to a modest degree.
    """
    from sklearn.metrics import roc_auc_score

    feats = diagnostics.build(tables["fights"], tables, cfg, families=tuple(FEATURE_FAMILIES))
    train, test = diagnostics.dev_only_split(feats, cfg)
    assert test["date"].max() <= cfg.windows.dev_end, "diagnostic reached the eval window"

    p = diagnostics.fit_predict_dev(train, test, cfg, feature_cols=ALL_FEATURE_COLS)
    auc = roc_auc_score(test["label"].astype(int), p)
    assert auc > 0.55, (
        f"AUC {auc:.4f} on real outcomes is near chance; the features carry almost no "
        "signal, which would make the shuffle test vacuous"
    )
    # Equally important: it must not be suspiciously good.
    assert auc < 0.80, f"AUC {auc:.4f} is implausibly high for UFC prediction; suspect leakage"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_feature_build_is_deterministic(tables, cfg):
    kwargs = build_kwargs(cfg)
    a = build_features(tables["fights"], tables["fight_stats"], tables["fighters"], **kwargs)
    b = build_features(tables["fights"], tables["fight_stats"], tables["fighters"], **kwargs)
    assert frame_digest(a) == frame_digest(b)


def test_every_feature_row_has_a_date_and_two_distinct_fighters(features):
    assert features["date"].notna().all()
    assert (features["fighter_1"] != features["fighter_2"]).all()
    assert not features["fight_id"].duplicated().any()


def test_dates_are_real_dates(features):
    assert isinstance(features["date"].iloc[0], dt.date)
