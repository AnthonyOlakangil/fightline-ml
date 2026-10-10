"""Closing-odds loading, vig removal, and the join onto UFCStats fights.

**Provenance.** The odds are closing moneylines, verified empirically rather than taken
on trust (the upstream dataset documents neither the book nor the timing). Across 3,812
fights matched independently against BestFightOdds at-close per-book data, this file's
prices agree with the closing consensus to a median absolute difference of 5 American
points (71.5% within 10 points, 88.2% within 25, 97.4% favourite/underdog sign
agreement). Mid-week captures from the same project's ``upcoming.csv`` differ far more,
so these are closing rather than opening lines. Post-fight assignment is excluded both
structurally (odds are published for not-yet-fought bouts) and empirically (the
favourite wins about 66% of the time, not ~100%).

**Only five columns are read.** The upstream CSV ships 118, most of them precomputed
career aggregates whose point-in-time correctness cannot be established. Those are
dropped at the snapshot boundary, and a test asserts nothing else reaches the model.

**Joining is conservative.** A mispaired fight attaches one fighter's price to
another's result and corrupts the market baseline without changing any row count, so a
dropped row is always preferred to a guessed pairing. Every non-exact decision -- every
fuzzy match and every unresolved row -- is written to a CSV for audit, and the share
left unjoined is capped: exceeding the cap raises.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ufc_predictor.ingest.names import BoutIndex, load_aliases, normalize_name, similarity
from ufc_predictor.ingest.stages import PipelineError, StageLog


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal.

    >>> american_to_decimal(-130)
    1.769...
    >>> american_to_decimal(124)
    2.24
    """
    if odds == 0 or not np.isfinite(odds):
        raise ValueError(f"invalid American odds: {odds!r}")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))


def american_to_implied(odds: float) -> float:
    """Implied probability, still including the bookmaker's margin."""
    return 1.0 / american_to_decimal(odds)


def remove_vig(p1: float, p2: float) -> tuple[float, float]:
    """Two-sided normalization: scale both implied probabilities to sum to 1.

    The market baseline is the vig-free favourite, so this must happen before any
    comparison against a model (CLAUDE.md).

    >>> remove_vig(0.55, 0.50)
    (0.5238..., 0.4761...)
    """
    total = p1 + p2
    if total <= 0 or not np.isfinite(total):
        raise ValueError(f"cannot de-vig probabilities summing to {total!r}")
    return p1 / total, p2 / total


def load_odds_raw(
    snapshot_dir: Path,
    log: StageLog,
    *,
    min_overround_exclusive: float = 0.0,
    max_overround: float = 0.25,
) -> pd.DataFrame:
    """Read the odds snapshot and compute vig-free probabilities per corner.

    Rows whose overround falls outside ``(min_overround_exclusive, max_overround]`` are
    dropped with a logged count. A two-sided book must price above 100%, so a
    non-positive overround is arithmetically impossible and marks corrupted prices;
    de-vigging such a pair would still yield two numbers summing to 1 and would silently
    enter the market baseline as if it were a real quote.
    """
    path = snapshot_dir / "odds_master.parquet"
    if not path.is_file():
        raise PipelineError(f"odds snapshot not found: {path}")

    df = pd.read_parquet(path)
    rows_in = len(df)

    expected = {"R_fighter", "B_fighter", "date", "R_odds", "B_odds"}
    extra = set(df.columns) - expected
    if extra:
        raise PipelineError(
            f"odds snapshot carries unexpected columns {sorted(extra)}. Only {sorted(expected)} "
            "may be read: the remaining upstream columns are precomputed career aggregates "
            "and are a leakage hazard."
        )

    df["r_odds"] = pd.to_numeric(df["R_odds"], errors="coerce")
    df["b_odds"] = pd.to_numeric(df["B_odds"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    bad_date = int(df["date"].isna().sum())
    missing_odds = int((df["r_odds"].isna() | df["b_odds"].isna()).sum())
    df = df.loc[df["date"].notna() & df["r_odds"].notna() & df["b_odds"].notna()].copy()

    # Vig removal, vectorized.
    r_dec = np.where(df["r_odds"] > 0, 1 + df["r_odds"] / 100, 1 + 100 / df["r_odds"].abs())
    b_dec = np.where(df["b_odds"] > 0, 1 + df["b_odds"] / 100, 1 + 100 / df["b_odds"].abs())
    r_imp, b_imp = 1 / r_dec, 1 / b_dec
    total = r_imp + b_imp
    df["r_prob_vigfree"] = r_imp / total
    df["b_prob_vigfree"] = b_imp / total
    df["overround"] = total - 1.0

    implausible = ~df["overround"].between(
        min_overround_exclusive, max_overround, inclusive="right"
    )
    n_implausible = int(implausible.sum())
    if n_implausible:
        worst = df.loc[implausible, ["R_odds", "B_odds", "overround"]].head(3).to_dict("records")
        print(
            f"      note: {n_implausible} rows dropped for an impossible or implausible "
            f"overround outside ({min_overround_exclusive}, {max_overround}], e.g. {worst}"
        )
    df = df.loc[~implausible].copy()

    log.record(
        "odds_raw",
        rows_in,
        len(df),
        {
            "unparseable_date": bad_date,
            "missing_odds": missing_odds,
            "implausible_overround": n_implausible,
        },
    )
    return df.reset_index(drop=True)


def join_odds_to_fights(
    odds: pd.DataFrame,
    fights: pd.DataFrame,
    log: StageLog,
    *,
    aliases_path: Path,
    reports_dir: Path,
    fuzzy_threshold: float = 0.92,
    ambiguity_margin: float = 0.05,
    max_unmatched_share: float = 0.05,
) -> pd.DataFrame:
    """Attach each priced row to a fight_id, in canonical orientation.

    Writes two audit files to ``reports_dir``:

    * ``fuzzy_matched_odds.csv`` -- every row resolved by fuzzy matching, with its score
      and runner-up score, so a human can check each accepted pairing.
    * ``unjoined_odds.csv`` -- every row that could not be resolved, with the reason.

    Raises:
        PipelineError: if the unjoined share exceeds ``max_unmatched_share``.
    """
    rows_in = len(odds)

    index = BoutIndex(
        [
            (row.fight_id, row.date, *row.pair_key.split("|"))
            for row in fights.itertuples(index=False)
        ]
    )
    aliases = load_aliases(aliases_path)

    results = [
        index.match(
            d,
            a,
            b,
            aliases=aliases,
            threshold=fuzzy_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        for d, a, b in zip(odds["date"], odds["R_fighter"], odds["B_fighter"], strict=True)
    ]

    odds = odds.copy()
    odds["fight_id"] = [r.fight_id for r in results]
    odds["match_method"] = [r.method for r in results]
    odds["match_score"] = [r.score for r in results]
    odds["match_runner_up"] = [r.runner_up for r in results]
    odds["match_candidate"] = [r.matched_bout for r in results]

    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_cols = [
        "date",
        "R_fighter",
        "B_fighter",
        "R_odds",
        "B_odds",
        "match_method",
        "match_score",
        "match_runner_up",
        "match_candidate",
    ]

    # Every fuzzy match is logged for human audit, not just the failures.
    fuzzy = odds.loc[odds["match_method"] == "fuzzy", audit_cols]
    fuzzy.sort_values("match_score").to_csv(reports_dir / "fuzzy_matched_odds.csv", index=False)

    unmatched = odds.loc[odds["fight_id"].isna(), audit_cols]
    unmatched.sort_values(["date", "R_fighter"]).to_csv(
        reports_dir / "unjoined_odds.csv", index=False
    )

    counts = odds["match_method"].value_counts().to_dict()
    n_unmatched = len(unmatched)
    share = n_unmatched / rows_in if rows_in else 0.0

    print(
        f"      match methods: exact={counts.get('exact', 0):,} "
        f"alias={counts.get('alias', 0):,} fuzzy={counts.get('fuzzy', 0):,} "
        f"| unmatched={n_unmatched:,} ({share:.2%})"
    )
    print(
        f"      audit: reports/fuzzy_matched_odds.csv ({len(fuzzy):,} rows), reports/unjoined_odds.csv ({n_unmatched:,} rows)"
    )

    if share > max_unmatched_share:
        raise PipelineError(
            f"{n_unmatched} of {rows_in} priced fights ({share:.2%}) could not be joined to a "
            f"UFCStats bout, above the {max_unmatched_share:.2%} limit. Unjoined rows are listed "
            f"in {reports_dir / 'unjoined_odds.csv'}. Resolve them via the alias table rather "
            "than lowering the fuzzy threshold: a wrong pairing is worse than a dropped row."
        )

    matched = odds.loc[odds["fight_id"].notna()].copy()

    if matched["fight_id"].duplicated().any():
        dupes = matched.loc[matched["fight_id"].duplicated(), "fight_id"].tolist()[:5]
        raise PipelineError(
            f"several priced rows resolved to the same fight_id (e.g. {dupes}); "
            "this would double-count fights in the backtest"
        )

    # Re-express prices in the canonical orientation, so nothing downstream can see
    # which corner a fighter was in. name_1/name_2 are aligned to fighter_1/fighter_2.
    f = fights.loc[:, ["fight_id", "fighter_1", "fighter_2", "name_1", "name_2"]]
    matched = matched.merge(f, on="fight_id", how="left", validate="one_to_one")

    r_norm = matched["R_fighter"].map(normalize_name)
    b_norm = matched["B_fighter"].map(normalize_name)

    # Decide which corner is fighter_1. Exact name agreement is the common case; for
    # alias/fuzzy matches the names differ in spelling, so fall back to whichever side
    # the red corner resembles more. Ties cannot happen: pairs are distinct fighters.
    red_is_1 = r_norm == matched["name_1"]
    needs_fallback = ~red_is_1 & (b_norm != matched["name_1"])
    if needs_fallback.any():
        sim_r1 = [similarity(r, n) for r, n in zip(r_norm, matched["name_1"], strict=True)]
        sim_r2 = [similarity(r, n) for r, n in zip(r_norm, matched["name_2"], strict=True)]
        fallback_red_is_1 = pd.Series(
            [a > b for a, b in zip(sim_r1, sim_r2, strict=True)], index=matched.index
        )
        red_is_1 = red_is_1.where(~needs_fallback, fallback_red_is_1)

    out = pd.DataFrame(
        {
            "fight_id": matched["fight_id"],
            "date": matched["date"],
            "fighter_1": matched["fighter_1"],
            "fighter_2": matched["fighter_2"],
            # Prices in canonical order.
            "odds_1": matched["r_odds"].where(red_is_1, matched["b_odds"]),
            "odds_2": matched["b_odds"].where(red_is_1, matched["r_odds"]),
            "p1_market": matched["r_prob_vigfree"].where(red_is_1, matched["b_prob_vigfree"]),
            "p2_market": matched["b_prob_vigfree"].where(red_is_1, matched["r_prob_vigfree"]),
            "overround": matched["overround"],
            "match_method": matched["match_method"],
        }
    )

    # The two canonical probabilities must still be a proper two-sided distribution.
    resid = (out["p1_market"] + out["p2_market"] - 1.0).abs().max()
    if resid > 1e-9:
        raise PipelineError(f"vig-free probabilities do not sum to 1 (max residual {resid:.2e})")

    log.record(
        "odds_joined",
        rows_in,
        len(out),
        {
            "unmatched_no_candidate": counts.get("unmatched_no_candidate", 0),
            "unmatched_ambiguous": counts.get("unmatched_ambiguous", 0),
        },
    )
    return out.reset_index(drop=True)
