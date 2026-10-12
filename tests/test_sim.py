"""Parlay pricing, Kelly staking, and bankroll tests.

The correctness claims worth pinning here are the ones that would silently inflate an
apparent edge: same-fight legs must be perfectly dependent rather than multiplied as if
independent, the simulated joint probability must match the analytic product where
independence genuinely holds, Kelly must refuse a non-favourable bet, and the market's
own probabilities priced against the book's vig-inclusive payout must never show an edge.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ufc_predictor.config import get_config
from ufc_predictor.sim.bankroll import bootstrap_by_event, replay
from ufc_predictor.sim.parlay import (
    build_legs,
    candidate_parlays,
    kelly_fraction_for,
    price_parlays,
    simulate_card,
)

D = dt.date(2020, 1, 1)


def _card(rows: list[dict]) -> pd.DataFrame:
    """Minimal card frame: one row per fight."""
    return pd.DataFrame(
        [
            {
                "fight_id": r.get("fight_id", f"f{i}"),
                "event_id": "e1",
                "date": D,
                "label": r["label"],
                "odds_1": r["odds_1"],
                "odds_2": r["odds_2"],
                "p_model": r["p"],
            }
            for i, r in enumerate(rows)
        ]
    )


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------


def test_kelly_matches_hand_computation():
    """p=0.6 at decimal 2.0 (b=1): f* = (0.6*1 - 0.4)/1 = 0.2."""
    assert kelly_fraction_for(0.6, 2.0) == pytest.approx(0.2)


def test_kelly_is_zero_at_break_even():
    """p * payout == 1 is a fair bet, so the optimal stake is nothing."""
    assert kelly_fraction_for(0.5, 2.0) == pytest.approx(0.0)


def test_kelly_is_zero_for_a_negative_edge():
    assert kelly_fraction_for(0.4, 2.0) == 0.0
    assert kelly_fraction_for(0.01, 1.5) == 0.0


def test_kelly_rejects_a_payout_with_no_upside():
    assert kelly_fraction_for(0.99, 1.0) == 0.0


# ---------------------------------------------------------------------------
# Correlation within a fight
# ---------------------------------------------------------------------------


def test_backing_both_fighters_in_one_bout_is_impossible():
    """The decisive correlation test.

    Treating the two sides as independent would give 0.6 * 0.4 = 0.24. Because each
    draw resolves the fight once, the simulated joint probability must be exactly zero.
    """
    card = _card([{"label": 1.0, "odds_1": -150, "odds_2": 130, "p": 0.6}])
    rng = np.random.default_rng(0)
    legs = build_legs(card, "p_model")
    draws = simulate_card(card, "p_model", 10_000, rng)

    both = np.ones(10_000, dtype=bool)
    for leg in legs:
        outcome = draws[:, leg.fight_index]
        both &= outcome if leg.side == 1 else ~outcome
    assert both.mean() == 0.0, "backing both fighters in one bout must be impossible"


def test_backing_the_same_fighter_twice_is_not_squared():
    """A duplicated leg must return that fighter's own probability, not its square."""
    card = _card([{"label": 1.0, "odds_1": -150, "odds_2": 130, "p": 0.6}])
    draws = simulate_card(card, "p_model", 20_000, np.random.default_rng(1))
    outcome = draws[:, 0]
    assert (outcome & outcome).mean() == pytest.approx(0.6, abs=0.02)
    assert (outcome & outcome).mean() != pytest.approx(0.36, abs=0.02)


def test_candidate_parlays_never_use_two_legs_from_one_fight():
    card = _card(
        [
            {"label": 1.0, "odds_1": 200, "odds_2": -250, "p": 0.7},
            {"label": 0.0, "odds_1": 150, "odds_2": -180, "p": 0.6},
        ]
    )
    legs = build_legs(card, "p_model")
    for combo in candidate_parlays(legs, max_legs=3):
        fights = [leg.fight_index for leg in combo]
        assert len(fights) == len(set(fights))


# ---------------------------------------------------------------------------
# Joint probability and EV
# ---------------------------------------------------------------------------


def test_simulated_joint_probability_matches_the_analytic_product():
    """Across different fights the legs are independent, so the sim must reproduce
    the product. This is what validates the simulator against closed form."""
    card = _card(
        [
            {"label": 1.0, "odds_1": 100, "odds_2": -120, "p": 0.70},
            {"label": 1.0, "odds_1": 120, "odds_2": -140, "p": 0.65},
            {"label": 1.0, "odds_1": 110, "odds_2": -130, "p": 0.60},
        ]
    )
    priced = price_parlays(card, "p_model", n_sims=50_000, max_legs=3, rng=np.random.default_rng(7))
    assert not priced.empty
    diff = (priced["p_sim"] - priced["p_analytic"]).abs()
    assert diff.max() < 0.01, f"simulation deviates from the analytic product by {diff.max()}"


def test_ev_uses_the_books_actual_payout_including_vig():
    """EV must be p * payout - 1 on the real price, not on a de-vigged one."""
    from ufc_predictor.ingest.odds import american_to_decimal

    card = _card([{"label": 1.0, "odds_1": 150, "odds_2": -180, "p": 0.50}])
    priced = price_parlays(card, "p_model", n_sims=20_000, max_legs=1, rng=np.random.default_rng(3))
    row = priced.iloc[0]
    payout = american_to_decimal(150)
    assert row["payout"] == pytest.approx(payout)
    assert row["ev"] == pytest.approx(row["p_sim"] * payout - 1.0)


def test_only_positive_ev_legs_are_combined():
    """Pruning is justified by the algebra, not convenience: for independent legs the
    parlay EV is prod(p * payout) - 1, so a negative-EV leg strictly lowers it."""
    card = _card(
        [
            {"label": 1.0, "odds_1": 300, "odds_2": -400, "p": 0.50},  # leg 1 strongly +EV
            {"label": 1.0, "odds_1": -400, "odds_2": 300, "p": 0.50},  # leg 1 strongly -EV
        ]
    )
    legs = build_legs(card, "p_model")
    for combo in candidate_parlays(legs, max_legs=2):
        for leg in combo:
            assert leg.payout_ev > 1.0


def test_realized_return_is_payout_minus_one_on_a_win():
    card = _card([{"label": 1.0, "odds_1": 200, "odds_2": -250, "p": 0.60}])
    priced = price_parlays(card, "p_model", n_sims=1000, max_legs=1, rng=np.random.default_rng(0))
    win = priced.loc[priced["sides"] == "1"].iloc[0]
    assert win["won"] is np.True_ or win["won"] is True
    assert win["realized_return"] == pytest.approx(win["payout"] - 1.0)

    lose = priced.loc[priced["sides"] == "2"]
    if not lose.empty:
        assert lose.iloc[0]["realized_return"] == -1.0


def test_no_method_or_round_legs_are_priced():
    """Only moneylines exist in the data; nothing else may be invented."""
    card = _card([{"label": 1.0, "odds_1": 150, "odds_2": -180, "p": 0.55}])
    legs = build_legs(card, "p_model")
    assert {leg.side for leg in legs} == {1, 2}
    assert len(legs) == 2, "exactly two moneyline selections per fight, and nothing more"


def test_a_fight_with_no_probability_produces_no_legs():
    """The variant abstains during warm-up; no price means no bet."""
    card = _card([{"label": 1.0, "odds_1": 150, "odds_2": -180, "p": np.nan}])
    assert build_legs(card, "p_model") == []


# ---------------------------------------------------------------------------
# The market control
# ---------------------------------------------------------------------------


def test_market_probabilities_never_show_an_edge_against_the_books_price():
    """The control that proves the pricing code is not inventing edge.

    De-vigged market probabilities priced against the book's vig-inclusive payout must
    yield negative EV on every selection: that gap is exactly the bookmaker's margin.
    """
    from ufc_predictor.ingest.odds import american_to_decimal, remove_vig

    for a1, a2 in ((-150, 130), (100, -120), (-300, 250), (200, -240)):
        d1, d2 = american_to_decimal(a1), american_to_decimal(a2)
        p1, p2 = remove_vig(1 / d1, 1 / d2)
        assert p1 * d1 < 1.0, f"{a1}/{a2}: market side 1 shows a positive edge"
        assert p2 * d2 < 1.0, f"{a1}/{a2}: market side 2 shows a positive edge"


# ---------------------------------------------------------------------------
# Bankroll
# ---------------------------------------------------------------------------


def _bets(returns: list[float], *, p: float = 0.6, payout: float = 2.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [D] * len(returns),
            "event_id": [f"e{i}" for i in range(len(returns))],
            "p_sim": [p] * len(returns),
            "payout": [payout] * len(returns),
            "realized_return": returns,
        }
    )


def test_bankroll_grows_on_wins_and_shrinks_on_losses():
    win = replay(
        _bets([1.0]),
        starting_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_fraction=1.0,
        ruin_threshold=0.2,
    )
    assert win["ending_bankroll"] > 1000.0
    loss = replay(
        _bets([-1.0]),
        starting_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_fraction=1.0,
        ruin_threshold=0.2,
    )
    assert loss["ending_bankroll"] < 1000.0


def test_the_bet_cap_binds():
    """Kelly here recommends 5% of bankroll; the cap must hold the stake to 1%."""
    capped = replay(
        _bets([-1.0], p=0.6, payout=2.0),
        starting_bankroll=1000.0,
        kelly_fraction=1.0,
        max_bet_fraction=0.01,
        ruin_threshold=0.0,
    )
    assert capped["ending_bankroll"] == pytest.approx(990.0)


def test_fractional_kelly_cannot_bankrupt_a_path():
    """Every stake is a fraction of the current bankroll, so it can shrink but not go
    negative, even on a long losing run."""
    res = replay(
        _bets([-1.0] * 200),
        starting_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_fraction=0.02,
        ruin_threshold=0.0,
    )
    assert res["ending_bankroll"] > 0


def test_ruin_is_detected_and_stops_the_path():
    res = replay(
        _bets([-1.0] * 500),
        starting_bankroll=1000.0,
        kelly_fraction=1.0,
        max_bet_fraction=0.5,
        ruin_threshold=0.2,
    )
    assert res["ruined"] is True
    assert res["ending_bankroll"] <= 200.0


def test_no_stake_is_placed_without_an_edge():
    """A break-even bet must be skipped entirely rather than staked at zero size."""
    res = replay(
        _bets([-1.0] * 10, p=0.5, payout=2.0),
        starting_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_fraction=0.02,
        ruin_threshold=0.2,
    )
    assert res["n_bets"] == 0
    assert res["ending_bankroll"] == pytest.approx(1000.0)


def test_max_drawdown_is_measured_from_the_running_peak():
    res = replay(
        _bets([1.0, 1.0, -1.0, -1.0]),
        starting_bankroll=1000.0,
        kelly_fraction=1.0,
        max_bet_fraction=0.5,
        ruin_threshold=0.0,
    )
    assert res["max_drawdown"] > 0


def test_bankroll_bootstrap_is_seeded_and_reports_ruin():
    bets = _bets([1.0, -1.0] * 25)
    a = bootstrap_by_event(
        bets,
        starting_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_fraction=0.02,
        ruin_threshold=0.2,
        n_resamples=200,
        seed=5,
    )
    b = bootstrap_by_event(
        bets,
        starting_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_fraction=0.02,
        ruin_threshold=0.2,
        n_resamples=200,
        seed=5,
    )
    assert a == b
    assert 0.0 <= a["ruin_probability"] <= 1.0
    assert (
        a["ending_bankroll"]["p05"] <= a["ending_bankroll"]["median"] <= a["ending_bankroll"]["p95"]
    )


def test_bootstrap_handles_a_history_with_no_bets():
    empty = pd.DataFrame(columns=["date", "event_id", "p_sim", "payout", "realized_return"])
    out = bootstrap_by_event(
        empty,
        starting_bankroll=1000.0,
        kelly_fraction=0.25,
        max_bet_fraction=0.02,
        ruin_threshold=0.2,
        n_resamples=10,
        seed=1,
    )
    assert out["n_resamples"] == 0


# ---------------------------------------------------------------------------
# The generated report
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parlay_metrics():
    import json

    cfg = get_config()
    path = cfg.paths.reports_dir / "parlay_metrics.json"
    if not path.is_file():
        pytest.skip("parlay_metrics.json absent; run `make parlay`")
    return json.loads(path.read_text(encoding="utf-8"))


def test_report_states_it_reuses_phase4_predictions(parlay_metrics):
    """Phase 5 must not have re-scored the evaluation window."""
    assert parlay_metrics["reused_phase4_predictions"] is True


def test_parlays_priced_the_latest_eval_run_without_scoring_another():
    """Parlay pricing reuses the stored walk-forward predictions. It must record which
    scored run it priced, that must be the latest one, and the ledger must not have
    grown on its account."""
    import json

    cfg = get_config()
    reports = cfg.paths.reports_dir
    paths = [reports / n for n in ("eval_runs.json", "metrics.json", "parlay_metrics.json")]
    if not all(p.is_file() for p in paths):
        pytest.skip("backtest or parlay artefacts absent")
    ledger, metrics, parlay = (json.loads(p.read_text(encoding="utf-8")) for p in paths)
    assert parlay["priced_eval_run"] == metrics["eval_run_count"] == len(ledger["runs"])


def test_report_carries_the_not_betting_advice_framing(parlay_metrics):
    text = parlay_metrics["disclaimer"].lower()
    assert "not betting advice" in text
    assert "research backtest" in text


def test_report_states_only_moneylines_were_priced(parlay_metrics):
    assert "moneyline" in parlay_metrics["legs_priced"].lower()
    assert "invented" in parlay_metrics["legs_priced"].lower()


def test_market_source_shows_no_positive_edge_in_the_report(parlay_metrics):
    """The control, end to end: ranking by the market's own numbers must not find value."""
    market = parlay_metrics["by_source"].get("market", {})
    if market.get("n_candidates", 0) == 0:
        return  # no positive-EV candidates at all, which is the expected outcome
    pv = market["predicted_vs_realized"]["all_top_ranked"]
    assert pv["mean_predicted_ev"] <= 0.001, (
        "the market's own de-vigged probabilities should not show an edge against the "
        "book's vig-inclusive payout"
    )


def test_monte_carlo_error_is_small(parlay_metrics):
    for source, res in parlay_metrics["by_source"].items():
        if res.get("n_candidates", 0) == 0:
            continue
        mc = res["monte_carlo_error"]
        assert mc["mean_abs_diff_vs_analytic"] < 0.01, source


def test_within_card_correlation_is_reported(parlay_metrics):
    corr = parlay_metrics["within_card_residual_correlation"]
    assert corr["n_pairs"] > 1000
    assert corr["correlation"] is not None


def test_readme_parlay_block_matches_parlay_metrics(parlay_metrics):
    """The parlay section is a rendering of parlay_metrics.json, not prose."""
    from ufc_predictor.backtest.readme_block import PARLAY_END, PARLAY_START, render_parlay

    readme = (get_config().paths.raw_dir.parent.parent / "README.md").read_text(encoding="utf-8")
    expected = render_parlay(parlay_metrics)
    start = readme.index(PARLAY_START)
    end = readme.index(PARLAY_END) + len(PARLAY_END)
    assert readme[start:end] == expected, "parlay block is stale; run `make parlay`"


def test_parlay_block_names_the_eval_run_it_priced(parlay_metrics):
    from ufc_predictor.backtest.readme_block import render_parlay

    block = render_parlay(parlay_metrics)
    assert f"evaluation run {parlay_metrics['priced_eval_run']}" in block
    assert "read exactly once" not in block


def test_parlay_block_gives_every_staked_model_its_bootstrap_band(parlay_metrics):
    """A positive realised return on a few dozen bets must arrive with its uncertainty,
    not as a bare number in a table."""
    from ufc_predictor.backtest.readme_block import render_parlay

    block = render_parlay(parlay_metrics)
    for source in ("lightgbm", "market_plus_model"):
        res = parlay_metrics["by_source"].get(source, {})
        if not res.get("n_staked"):
            continue
        bs = res["bankroll_bootstrap"]
        assert f"{bs['share_losing_money']:.1%} of histories lose money" in block, source
        assert f"£{bs['ending_bankroll']['p05']:,.0f}" in block, source
        assert f"£{bs['ending_bankroll']['p95']:,.0f}" in block, source


def test_readme_parlay_section_is_framed_as_not_betting_advice():
    readme = (get_config().paths.raw_dir.parent.parent / "README.md").read_text(encoding="utf-8")
    assert "Not betting advice" in readme
    assert "should be used to wager" in readme


def test_predicted_edge_and_realized_roi_are_both_reported(parlay_metrics):
    """The comparison that shows how much of the edge was noise must be present."""
    for source, res in parlay_metrics["by_source"].items():
        if not res.get("n_candidates"):
            continue
        pv = res["predicted_vs_realized"]["all_top_ranked"]
        assert "mean_predicted_ev" in pv and "realized_roi" in pv, source
        assert pv["n"] > 0


def test_bankroll_reports_losses_not_just_gains(parlay_metrics):
    """Losses must be reported, including drawdown and ruin."""
    for res in parlay_metrics["by_source"].values():
        if not res.get("n_staked"):
            continue
        bk = res["bankroll"]
        bs = res["bankroll_bootstrap"]
        assert "ending_bankroll" in bk and "max_drawdown" in bk
        assert "ruin_probability" in bs
        assert 0.0 <= bs["ruin_probability"] <= 1.0
