"""The point-in-time feature builder.

Every feature for a fight is computed from that fighter's bouts dated **strictly
before** the fight's event date. The implementation makes that structural rather than
incidental: state is snapshotted per event-date, all fights on the date are described
from the snapshot, and only afterwards is the date's history folded in.

Three properties this guards, each verified in ``tests/test_leakage.py``:

* **Fights on one card cannot see each other.** A card is simultaneous, so a bout must
  not be described using another bout's result from the same night.
* **One-night tournaments are handled.** 67 fighter-days hold more than one fight (up
  to four, in 1994-99). A plain ``shift(1)`` over a fighter's fight sequence would leak
  the first bout of the night into the second; grouping by date does not.
* **A fight's own round statistics never enter its own features.** They are folded in
  only after the date is emitted.

Missing history is always represented as ``NaN`` plus an explicit flag, never as ``0``.
A debutant encoded as 0.0 strikes-per-minute is not missing data to a model, it is a
fighter who lands nothing -- a different and false claim. LightGBM handles NaN natively.

**Not modelled.** There are no injury, camp, short-notice, weight-cut, or social-media
features here, because this project has no data for any of them. Inventing them would
mean inventing numbers. The feature set is exactly what the fight record supports.

**No tale-of-the-tape attributes.** Height, reach and age are absent on purpose. The
fighters table is a single present-day scrape, and UFCStats only records those fields
for fighters who stayed on the roster, so a blank reach means "this fighter was later
cut". Run 1 of the backtest used them, and before 2016 the side with the blank reach
won 5.7% of dev-window bouts. Blanking both sides whenever either is missing does not
close the leak, because "this pair has a blank" plus prior experience still points at
the fighter who was cut. Nothing dated exists to replace them, so they are dropped.

**Odds are never a feature.** The market is a baseline to be compared against, so
letting it into the feature set would make "the model beats the market" circular.

**Feature families.** ``FEATURE_COLS`` is the "base" family: the run-3 feature set,
always built. Phase 8 development-window candidates register further families in
``FEATURE_FAMILIES`` and are built only when ``build_features(families=...)`` asks for
them, so the default output is byte-identical to run 3's. The leakage suite builds every
family and runs every column through the same tests.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ufc_predictor.features import adjusted, decline, glicko2, styles
from ufc_predictor.features.adjusted import ADJUSTED_COLS, RECENCY_COLS
from ufc_predictor.features.decline import DECLINE_COLS
from ufc_predictor.features.elo import build_elo
from ufc_predictor.features.glicko2 import GLICKO_COLS
from ufc_predictor.features.styles import STYLE_COLS

# Seconds in a championship/standard round. Durations are reconstructed as
# (finish_round - 1) * ROUND_SECONDS + finish_time_sec, which is also correct for the
# early single-round formats, where finish_round is 1 and the clock carries the length.
ROUND_SECONDS = 300

# Per-fighter rate metrics, each computed over several history windows.
RATE_METRICS = (
    "sig_str_landed_pm",  # significant strikes landed per minute
    "sig_str_absorbed_pm",  # significant strikes absorbed per minute
    "sig_str_acc",  # landed / attempted
    "sig_str_def",  # 1 - (opponent landed / opponent attempted)
    "td_landed_p15",  # takedowns landed per 15 minutes
    "td_acc",  # landed / attempted
    "td_def",  # 1 - (opponent takedowns landed / attempted)
    "sub_att_p15",
    "kd_p15",
    "ctrl_pm",  # control-time seconds per minute of fight
)

CAREER = "career"
BOUT = "bout"


def _windows(rolling_windows: Sequence[int]) -> list[str]:
    return [f"w{w}" for w in rolling_windows] + [CAREER]


def feature_columns(rolling_windows: Sequence[int] = (3, 5)) -> list[str]:
    """The exact feature column list, in a stable order."""
    cols: list[str] = [
        "elo_1",
        "elo_2",
        "elo_diff",
        "n_prior_fights_1",
        "n_prior_fights_2",
        "n_prior_fights_diff",
        "win_rate_1",
        "win_rate_2",
        "win_rate_diff",
        "win_streak_1",
        "win_streak_2",
        "win_streak_diff",
        "loss_streak_1",
        "loss_streak_2",
        "loss_streak_diff",
        "days_since_last_1",
        "days_since_last_2",
        "days_since_last_diff",
        "finish_rate_1",
        "finish_rate_2",
        "finish_rate_diff",
        "is_debut_1",
        "is_debut_2",
        "has_stats_history_1",
        "has_stats_history_2",
        "title_bout",
        "scheduled_rounds",
    ]
    for win in _windows(rolling_windows):
        for metric in RATE_METRICS:
            cols += [f"{metric}_{win}_1", f"{metric}_{win}_2", f"{metric}_{win}_diff"]
    return cols


FEATURE_COLS: list[str] = feature_columns((3, 5))

# Every family's columns, in output order. "base" is always built; the rest are appended
# after it, in this order, when requested.
FEATURE_FAMILIES: dict[str, list[str]] = {
    "base": FEATURE_COLS,  # the 117 run-3 features, unchanged
    "adjusted": ADJUSTED_COLS,  # opponent-adjusted career rates (features/adjusted.py)
    "recency": RECENCY_COLS,  # recency-weighted rates (features/adjusted.py)
    "style": STYLE_COLS,  # style-matchup interactions (features/styles.py)
    "decline": DECLINE_COLS,  # tenure, recent finishes and form trends (features/decline.py)
    "glicko": GLICKO_COLS,  # point-in-time Glicko-2 rating (features/glicko2.py)
}
ALL_FEATURE_COLS: list[str] = [c for cols in FEATURE_FAMILIES.values() for c in cols]


# ---------------------------------------------------------------------------
# History table
# ---------------------------------------------------------------------------


def fighter_fight_history(fights: pd.DataFrame, fight_stats: pd.DataFrame) -> pd.DataFrame:
    """One row per (fight, fighter): what that fighter did, and had done to them.

    This is the raw material for every rolling statistic. Opponent columns come from the
    *other* fighter's rows in the same bout, which is what makes "absorbed" and
    "defence" computable.
    """
    long = pd.concat(
        [
            fights.assign(fighter_id=fights["fighter_1"], opponent_id=fights["fighter_2"]),
            fights.assign(fighter_id=fights["fighter_2"], opponent_id=fights["fighter_1"]),
        ],
        ignore_index=True,
    )

    long["won"] = np.where(
        long["is_decisive"] & long["winner_id"].notna(),
        (long["winner_id"] == long["fighter_id"]).astype(float),
        np.nan,
    )
    rounds = pd.to_numeric(long["finish_round"], errors="coerce")
    secs = pd.to_numeric(long["finish_time_sec"], errors="coerce")
    long["duration_sec"] = (rounds - 1) * ROUND_SECONDS + secs
    long.loc[long["duration_sec"] <= 0, "duration_sec"] = np.nan
    long["won_by_finish"] = np.where(
        long["won"] == 1.0, long["method_class"].isin(["KO/TKO", "SUB"]).astype(float), np.nan
    )

    # Per-fighter totals across rounds of that bout.
    stat_cols = [
        "sig_str_landed",
        "sig_str_att",
        "td_landed",
        "td_att",
        "sub_att",
        "kd",
        "ctrl_sec",
    ]
    present = [c for c in stat_cols if c in fight_stats.columns]

    if fight_stats.empty or not present:
        # Legitimate case, not an error: pre-2001 UFC fights carry no per-round
        # statistics at all. Those fighters still have a record, a layoff and an Elo,
        # so history is built with the statistical columns absent and every derived
        # rate left as NaN -- never as zero.
        totals = pd.DataFrame(
            {"fight_id": pd.Series(dtype="object"), "fighter_id": pd.Series(dtype="object")}
            | {c: pd.Series(dtype="float64") for c in stat_cols}
        )
        present = stat_cols
    else:
        totals = fight_stats.groupby(["fight_id", "fighter_norm"], as_index=False)[present].sum(
            min_count=1
        )

    # fight_stats keys fighters by normalized name; map that onto canonical ids using
    # the name/id alignment stored on the fights table.
    name_to_id = pd.concat(
        [
            fights[["fight_id", "name_1", "fighter_1"]].rename(
                columns={"name_1": "fighter_norm", "fighter_1": "fighter_id"}
            ),
            fights[["fight_id", "name_2", "fighter_2"]].rename(
                columns={"name_2": "fighter_norm", "fighter_2": "fighter_id"}
            ),
        ],
        ignore_index=True,
    )
    if "fighter_norm" in totals.columns:
        totals = totals.merge(name_to_id, on=["fight_id", "fighter_norm"], how="inner")
        totals = totals.drop(columns=["fighter_norm"])

    own = totals.rename(columns={c: f"own_{c}" for c in present})
    opp = totals.rename(columns={**{c: f"opp_{c}" for c in present}, "fighter_id": "opponent_id"})

    keep = [
        "fight_id",
        "date",
        "fighter_id",
        "opponent_id",
        "won",
        "won_by_finish",
        "duration_sec",
        "method_class",
    ]
    hist = long.loc[:, keep].merge(own, on=["fight_id", "fighter_id"], how="left")
    hist = hist.merge(opp, on=["fight_id", "opponent_id"], how="left")
    hist["has_stats"] = hist["own_sig_str_att"].notna()

    return hist.sort_values(["fighter_id", "date", "fight_id"], kind="stable").reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Rolling state
# ---------------------------------------------------------------------------


class _FighterState:
    """Running history for one fighter, advanced only at event-date boundaries."""

    __slots__ = (
        "bouts",
        "fights",
        "finishes",
        "first_date",
        "last_date",
        "last_finish_loss",
        "loss_streak",
        "n",
        "win_streak",
        "wins",
    )

    def __init__(self) -> None:
        # Per prior bout: (own per-bout rates, opponent's frozen pre-bout baseline).
        # Filled only when the adjusted/recency families are requested.
        self.bouts: list[tuple[dict[str, float], dict[str, float]]] = []
        self.fights: deque[dict] = deque(maxlen=64)
        # Kept outside the bounded deque, so they hold whatever the career's length.
        self.first_date: dt.date | None = None
        self.last_finish_loss: dt.date | None = None
        self.last_date: dt.date | None = None
        self.n = 0
        self.wins = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.finishes = 0.0

    def add(self, rec: dict) -> None:
        self.fights.append(rec)
        self.n += 1
        if self.first_date is None:
            self.first_date = rec["date"]
        self.last_date = rec["date"]
        won = rec["won"]
        if won == 1.0:
            self.wins += 1
            self.win_streak += 1
            self.loss_streak = 0
            if rec["won_by_finish"] == 1.0:
                self.finishes += 1
        elif won == 0.0:
            self.loss_streak += 1
            self.win_streak = 0
            if decline.is_finish_loss(rec):
                self.last_finish_loss = rec["date"]
        # Draws/NCs count as experience and reset neither streak.


def _rate_block(records: list[dict], prefix_suffix: str) -> dict[str, float]:
    """Aggregate a slice of history into the rate metrics.

    Returns NaN for any metric whose denominator is absent, so that "no data" never
    masquerades as zero.
    """
    out: dict[str, float] = {}

    with_stats = [r for r in records if r["has_stats"]]
    total_sec = sum(r["duration_sec"] for r in with_stats if pd.notna(r["duration_sec"]))
    minutes = total_sec / 60.0 if total_sec else 0.0

    def s(key: str) -> float:
        return float(sum(r[key] for r in with_stats if pd.notna(r.get(key))))

    def ratio(num: float, den: float) -> float:
        return num / den if den > 0 else np.nan

    if not with_stats or minutes <= 0:
        for metric in RATE_METRICS:
            out[f"{metric}_{prefix_suffix}"] = np.nan
        return out

    own_landed, own_att = s("own_sig_str_landed"), s("own_sig_str_att")
    opp_landed, opp_att = s("opp_sig_str_landed"), s("opp_sig_str_att")
    own_td, own_td_att = s("own_td_landed"), s("own_td_att")
    opp_td, opp_td_att = s("opp_td_landed"), s("opp_td_att")

    out[f"sig_str_landed_pm_{prefix_suffix}"] = ratio(own_landed, minutes)
    out[f"sig_str_absorbed_pm_{prefix_suffix}"] = ratio(opp_landed, minutes)
    out[f"sig_str_acc_{prefix_suffix}"] = ratio(own_landed, own_att)
    out[f"sig_str_def_{prefix_suffix}"] = 1.0 - opp_landed / opp_att if opp_att > 0 else np.nan
    out[f"td_landed_p15_{prefix_suffix}"] = ratio(own_td * 15.0, minutes)
    out[f"td_acc_{prefix_suffix}"] = ratio(own_td, own_td_att)
    out[f"td_def_{prefix_suffix}"] = 1.0 - opp_td / opp_td_att if opp_td_att > 0 else np.nan
    out[f"sub_att_p15_{prefix_suffix}"] = ratio(s("own_sub_att") * 15.0, minutes)
    out[f"kd_p15_{prefix_suffix}"] = ratio(s("own_kd") * 15.0, minutes)
    out[f"ctrl_pm_{prefix_suffix}"] = ratio(s("own_ctrl_sec"), minutes)
    return out


def _bare(block: dict[str, float], suffix: str) -> dict[str, float]:
    """``_rate_block`` output keyed by bare metric name."""
    return {metric: block[f"{metric}_{suffix}"] for metric in RATE_METRICS}


def _snapshot(state: _FighterState, event_date: dt.date, windows: Sequence[int]) -> dict:
    """Describe a fighter as of the morning of ``event_date``."""
    recs = list(state.fights)
    snap: dict[str, float | bool] = {
        "n_prior_fights": float(state.n),
        "win_rate": state.wins / state.n if state.n else np.nan,
        "win_streak": float(state.win_streak),
        "loss_streak": float(state.loss_streak),
        "finish_rate": state.finishes / state.n if state.n else np.nan,
        "days_since_last": (
            float((event_date - state.last_date).days) if state.last_date else np.nan
        ),
        "is_debut": state.n == 0,
        "has_stats_history": any(r["has_stats"] for r in recs),
    }
    for w in windows:
        snap.update(_rate_block(recs[-w:], f"w{w}"))
    snap.update(_rate_block(recs, CAREER))
    return snap


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_features(
    fights: pd.DataFrame,
    fight_stats: pd.DataFrame,
    fighters: pd.DataFrame,
    *,
    rolling_windows: Sequence[int] = (3, 5),
    elo_params: dict | None = None,
    families: Sequence[str] = ("base",),
    recency_half_life_fights: float = 3.0,
    glicko_params: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build the point-in-time feature table: one row per fight.

    ``label`` is 1.0 when ``fighter_1`` won, 0.0 when ``fighter_2`` won, and NaN for
    draws and no contests -- which stay in the table because they are real history for
    experience and layoff, but carry no binary outcome.

    ``fighters`` is accepted but its attributes are deliberately never read: see the
    module docstring. ``tests/test_leakage.py`` blanks them and asserts that no feature
    value moves.

    ``families`` names the ``FEATURE_FAMILIES`` to emit. The base columns always come
    first; each requested family's columns follow in registry order. The default emits
    the base family alone. ``recency_half_life_fights`` is the "recency" family's
    half-life, in bouts. ``glicko_params`` is the "glicko" family's ``glicko2`` config
    block; it has no default and is ignored unless that family is requested.

    Raises:
        ValueError: on an unknown family name, a non-positive half-life, or the "glicko"
            family requested without valid ``glicko_params``.
    """
    elo_params = elo_params or {}
    windows = tuple(rolling_windows)

    requested = set(families)
    unknown = sorted(requested - set(FEATURE_FAMILIES))
    if unknown:
        raise ValueError(f"unknown feature families {unknown}; known: {list(FEATURE_FAMILIES)}")
    if not recency_half_life_fights > 0:
        raise ValueError(
            f"recency_half_life_fights must be positive, got {recency_half_life_fights!r}"
        )
    if "glicko" in requested:
        glicko_params = glicko2.check_params(glicko_params)  # fail before the long loop
    family_cols = [
        c
        for name, cols in FEATURE_FAMILIES.items()
        if name != "base" and name in requested
        for c in cols
    ]
    # The adjusted and recency families share one per-bout record, kept only if needed.
    track_bouts = bool(requested & {"adjusted", "recency"})
    track_style = "style" in requested
    track_decline = "decline" in requested

    history = fighter_fight_history(fights, fight_stats)
    by_date: dict[dt.date, list[dict]] = defaultdict(list)
    for rec in history.to_dict("records"):
        by_date[rec["date"]].append(rec)

    states: dict[str, _FighterState] = defaultdict(_FighterState)
    rows: list[dict] = []

    ordered = fights.sort_values(["date", "fight_id"], kind="stable")
    for event_date, day in ordered.groupby("date", sort=True):
        # --- 1. Describe every fight on this date from the pre-date snapshot. ---
        cache: dict[str, dict] = {}
        # Each fighter's pre-date mirror quantities, frozen into their opponents' bout
        # records in step 2. Taken here, before anything on this date is folded in.
        baselines: dict[str, dict[str, float]] = {}
        # Each fighter's pre-date career rates and absorbed aggregates, for "style".
        profiles: dict[str, dict[str, float]] = {}
        for row in day.itertuples(index=False):
            feat: dict = {
                "fight_id": row.fight_id,
                "date": event_date,
                "fighter_1": row.fighter_1,
                "fighter_2": row.fighter_2,
                "is_decisive": bool(row.is_decisive),
                "label": (
                    np.nan
                    if not row.is_decisive or pd.isna(row.winner_id)
                    else float(row.winner_id == row.fighter_1)
                ),
                "title_bout": bool(row.title_bout),
                "scheduled_rounds": (
                    float(row.scheduled_rounds) if pd.notna(row.scheduled_rounds) else np.nan
                ),
            }

            for slot, fid in (("1", row.fighter_1), ("2", row.fighter_2)):
                if fid not in cache:
                    state = states[fid]
                    cache[fid] = _snapshot(state, event_date, windows)
                    career = _bare(cache[fid], CAREER)
                    # Always bound, so no branch can read the previous fighter's value.
                    absorbed = (
                        adjusted.absorbed_rates(state.fights)
                        if (track_bouts or track_style)
                        else None
                    )
                    if track_bouts:
                        cache[fid].update(
                            adjusted.family_snapshot(state.bouts, recency_half_life_fights)
                        )
                        baselines[fid] = adjusted.opponent_baseline(career, absorbed)
                    if track_style:
                        profiles[fid] = {**career, **absorbed}
                    if track_decline:
                        last = list(state.fights)[-decline.LAST_N :]
                        recent = _bare(_rate_block(last, "last"), "last")
                        cache[fid].update(
                            decline.decline_snapshot(
                                last,
                                recent,
                                career,
                                event_date,
                                state.first_date,
                                state.last_finish_loss,
                            )
                        )
                snap = cache[fid]
                for key, value in snap.items():
                    feat[f"{key}_{slot}"] = value

            if track_style:
                # Both sides' pre-date profiles: a card-mate's or tonight's other bout's
                # result cannot reach either.
                feat.update(styles.matchup(profiles[row.fighter_1], profiles[row.fighter_2]))

            rows.append(feat)

        # --- 2. Only now fold this date's fights into each fighter's history. ---
        for rec in by_date.get(event_date, ()):
            if track_bouts:
                # The opponent's baseline is the step-1, pre-date one: no card-mate's
                # result and no other bout of the opponent's tonight can reach it.
                rates = adjusted.bout_rates(_bare(_rate_block([rec], BOUT), BOUT), rec)
                states[rec["fighter_id"]].bouts.append((rates, baselines[rec["opponent_id"]]))
            states[rec["fighter_id"]].add(rec)

    feats = pd.DataFrame(rows)

    # Differences, computed after both sides exist.
    diff_bases = [
        "elo",
        "n_prior_fights",
        "win_rate",
        "win_streak",
        "loss_streak",
        "days_since_last",
        "finish_rate",
    ]
    elo = build_elo(
        fights,
        initial_rating=float(elo_params.get("initial_rating", 1500.0)),
        k_factor=float(elo_params.get("k_factor", 32.0)),
        scale=float(elo_params.get("scale", 400.0)),
    )
    feats = feats.merge(elo, on="fight_id", how="left", validate="one_to_one")
    if "glicko" in requested:
        rated = glicko2.build_glicko(fights, glicko_params)
        feats = feats.merge(rated, on="fight_id", how="left", validate="one_to_one")
        unrated = feats[rated.columns].isna().any(axis=1)
        if unrated.any():
            raise RuntimeError(f"{int(unrated.sum())} fights have no Glicko-2 rating")

    for base in diff_bases:
        if base == "elo":
            continue  # build_elo already supplies elo_diff
        feats[f"{base}_diff"] = feats[f"{base}_1"] - feats[f"{base}_2"]
    for win in _windows(windows):
        for metric in RATE_METRICS:
            feats[f"{metric}_{win}_diff"] = feats[f"{metric}_{win}_1"] - feats[f"{metric}_{win}_2"]
    if family_cols:
        stems = [c[: -len("_diff")] for c in family_cols if c.endswith("_diff")]
        diffs = {f"{s}_diff": feats[f"{s}_1"] - feats[f"{s}_2"] for s in stems}
        feats = pd.concat([feats, pd.DataFrame(diffs, index=feats.index)], axis=1)

    cols = [*feature_columns(windows), *family_cols]
    missing = [c for c in cols if c not in feats.columns]
    if missing:
        raise RuntimeError(f"feature builder did not produce expected columns: {missing}")

    ordered_cols = ["fight_id", "date", "fighter_1", "fighter_2", "label", "is_decisive", *cols]
    out = feats.loc[:, ordered_cols].sort_values(["date", "fight_id"], kind="stable")
    return out.reset_index(drop=True)
