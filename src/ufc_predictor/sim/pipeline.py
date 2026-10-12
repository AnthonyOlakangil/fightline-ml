"""Parlay backtest: price, rank, stake, and report what actually happened.

**Research backtest only. Not betting advice.** Nothing here should be used to wager.

Reads the out-of-sample predictions stored by Phase 4 and never re-runs the walk-forward,
so the evaluation window is not scored a second time and the eval-run ledger is untouched.

The central output is not the bankroll chart but the **predicted EV against realised
ROI** table. Ranking parlays by modelled edge and then checking what those parlays
actually returned is the cleanest way to show how much of an apparent edge was noise.
The market's own probabilities are included as a control: priced against the book's
vig-inclusive payout they must show negative EV everywhere, and anything else would mean
the pricing code is wrong.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ufc_predictor.config import Config, get_config
from ufc_predictor.sim.bankroll import bootstrap_by_event, replay
from ufc_predictor.sim.parlay import price_parlays

#: Probability sources to rank by, in report order.
SOURCES = ("lightgbm", "market_plus_model", "market")


def load_predictions(cfg: Config, *, eval_only: bool = True) -> pd.DataFrame:
    """Load Phase 4's stored out-of-sample predictions. No re-scoring."""
    path = cfg.paths.reports_dir / "walkforward_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `make backtest` first")
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    if eval_only:
        df = df.loc[(df["date"] >= cfg.windows.eval_start) & (df["date"] <= cfg.windows.eval_end)]
    return df.sort_values(["date", "event_id", "fight_id"], kind="stable").reset_index(drop=True)


def within_card_residual_correlation(preds: pd.DataFrame, prob_col: str) -> dict:
    """Measure the conditional-independence assumption instead of only asserting it.

    For each card, correlate every pair of fight residuals (outcome minus predicted
    probability). If bouts were conditionally independent given their features, the mean
    pairwise correlation should sit near zero.
    """
    resid_pairs: list[tuple[float, float]] = []
    for _, card in preds.groupby("event_id"):
        r = (card["label"] - card[prob_col]).to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                resid_pairs.append((r[i], r[j]))
    if len(resid_pairs) < 100:
        return {"n_pairs": len(resid_pairs), "correlation": None}
    a = np.array([x for x, _ in resid_pairs])
    b = np.array([y for _, y in resid_pairs])
    return {
        "n_pairs": len(a),
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "note": (
            "Pearson correlation between pairs of same-card residuals. Near zero "
            "supports treating fights on a card as conditionally independent given "
            "features, which is what the parlay joint probability assumes."
        ),
    }


COLOURS = {"lightgbm": "#2563EB", "market_plus_model": "#DC2626", "market": "#111827"}


def _plot_bankroll(results: dict, path, starting: float) -> None:
    """Bankroll paths with a bootstrap 5-95% band."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    plotted = False
    for source, res in results.items():
        boot = res.get("bankroll_bootstrap", {})
        band = boot.get("path_band")
        if not band:
            continue
        x = band["bet_index"]
        colour = COLOURS.get(source, "#666")
        ax.fill_between(x, band["p05"], band["p95"], color=colour, alpha=0.15, linewidth=0)
        ax.plot(x, band["median"], color=colour, linewidth=1.8, label=f"{source} (median)")
        plotted = True

    ax.axhline(starting, color="#111827", linestyle="--", linewidth=1, label="starting bankroll")
    ax.set_xlabel("bet number")
    ax.set_ylabel("bankroll")
    ax.set_title(
        "Simulated bankroll, 5-95% band over event-resampled histories\n"
        "Research backtest only - not betting advice"
    )
    ax.grid(alpha=0.25, linewidth=0.5)
    if plotted:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_predicted_vs_realized(results: dict, path) -> None:
    """Modelled edge against what the top-ranked parlays actually returned."""
    names, pred, real = [], [], []
    for source, res in results.items():
        pv = res.get("predicted_vs_realized", {}).get("all_top_ranked", {})
        if not pv.get("n"):
            continue
        names.append(source.replace("_", " "))
        pred.append(pv["mean_predicted_ev"])
        real.append(pv["realized_roi"])

    if not names:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(names))
    ax.barh(y - 0.2, pred, height=0.38, color="#93C5FD", label="predicted EV")
    ax.barh(y + 0.2, real, height=0.38, color="#DC2626", label="realised ROI")
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("return per unit staked")
    ax.set_title("Predicted edge vs what actually happened (top-ranked parlays)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scored_eval_run(cfg: Config) -> int:
    """Which ledgered evaluation run produced the predictions being priced."""
    path = cfg.paths.reports_dir / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError("metrics.json is missing; run `make backtest` first")
    return int(json.loads(path.read_text(encoding="utf-8"))["eval_run_count"])


def run(cfg: Config | None = None, *, verbose: bool = True) -> dict:
    """Price parlays for every eval card under each probability source."""
    cfg = cfg or get_config()
    sim_cfg = cfg.sim
    preds = load_predictions(cfg, eval_only=True)

    n_sims = int(sim_cfg["n_sims"])
    max_legs = int(sim_cfg["max_parlay_legs"])
    min_edge = float(sim_cfg["min_edge"])
    top_n = int(sim_cfg["top_n_per_card"])

    if verbose:
        print("\nParlay backtest  (research only - not betting advice)")
        print(f"  cards            : {preds['event_id'].nunique()}")
        print(f"  fights           : {len(preds):,}")
        print(f"  sims per card    : {n_sims:,}")
        print(f"  max legs         : {max_legs}")
        print(
            f"  staking          : {sim_cfg['kelly_fraction']}x Kelly, "
            f"capped at {sim_cfg['max_bet_fraction']:.1%} of bankroll"
        )
        print("  legs priced      : moneyline only (no method or round props exist in the data)")
        print()

    results: dict[str, dict] = {}
    all_top: dict[str, pd.DataFrame] = {}

    for source in SOURCES:
        col = f"p_{source}"
        priced_frames: list[pd.DataFrame] = []

        # One generator for the whole source, seeded from config, advanced per card so
        # each card's draw is independent but the whole run is reproducible.
        for i, (_, card) in enumerate(preds.groupby("event_id", sort=False)):
            rng = np.random.default_rng([cfg.seed, SOURCES.index(source), i])
            frame = price_parlays(
                card.reset_index(drop=True),
                col,
                n_sims=n_sims,
                max_legs=max_legs,
                rng=rng,
            )
            if not frame.empty:
                priced_frames.append(frame)

        if not priced_frames:
            results[source] = {"n_candidates": 0, "note": "no positive-EV candidates"}
            continue

        priced = pd.concat(priced_frames, ignore_index=True)

        # Monte-Carlo error: the simulated joint probability against the analytic
        # product. They should agree closely, which validates the simulation.
        mc_err = (priced["p_sim"] - priced["p_analytic"]).abs()

        # Top-ranked parlays per card, regardless of whether they clear the stake
        # threshold -- this is the set the predicted-vs-realised table describes.
        top = (
            priced.sort_values(["event_id", "ev"], ascending=[True, False])
            .groupby("event_id", sort=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        all_top[source] = top

        # Bets actually staked: only those clearing the modelled edge threshold.
        bets = top.loc[top["ev"] >= min_edge].copy()

        path = replay(
            bets,
            starting_bankroll=float(sim_cfg["starting_bankroll"]),
            kelly_fraction=float(sim_cfg["kelly_fraction"]),
            max_bet_fraction=float(sim_cfg["max_bet_fraction"]),
            ruin_threshold=float(sim_cfg["ruin_threshold"]),
        )
        boot = bootstrap_by_event(
            bets,
            starting_bankroll=float(sim_cfg["starting_bankroll"]),
            kelly_fraction=float(sim_cfg["kelly_fraction"]),
            max_bet_fraction=float(sim_cfg["max_bet_fraction"]),
            ruin_threshold=float(sim_cfg["ruin_threshold"]),
            n_resamples=int(sim_cfg["bankroll_bootstrap_resamples"]),
            seed=cfg.seed,
        )

        results[source] = {
            "n_candidates": len(priced),
            "n_top_ranked": len(top),
            "n_staked": len(bets),
            "monte_carlo_error": {
                "max_abs_diff_vs_analytic": float(mc_err.max()),
                "mean_abs_diff_vs_analytic": float(mc_err.mean()),
                "note": (
                    "Simulated joint probability minus the analytic product of leg "
                    "probabilities. Small values confirm the simulation reproduces the "
                    "independence structure it is given."
                ),
            },
            "predicted_vs_realized": _predicted_vs_realized(top, min_edge),
            "bankroll": {k: v for k, v in path.items() if k != "path"},
            "bankroll_bootstrap": boot,
        }

        if verbose:
            pv = results[source]["predicted_vs_realized"]["all_top_ranked"]
            print(f"  {source}")
            print(
                f"    candidates {len(priced):>7,}  top-ranked {len(top):>5,}  "
                f"staked {len(bets):>5,}"
            )
            print(
                f"    predicted EV {pv['mean_predicted_ev']:+.4f}  "
                f"realised ROI {pv['realized_roi']:+.4f}  "
                f"hit rate {pv['hit_rate']:.4f} (expected {pv['mean_predicted_prob']:.4f})"
            )
            if path["n_bets"]:
                print(
                    f"    bankroll {path['starting_bankroll']:.0f} -> "
                    f"{path['ending_bankroll']:.2f}  "
                    f"({path['total_return']:+.1%}, max DD {path['max_drawdown']:.1%}, "
                    f"ruin {boot.get('ruin_probability', 0):.1%})"
                )
            else:
                print("    no parlay cleared the staking threshold; bankroll unchanged")
            print()

    corr = within_card_residual_correlation(preds, "p_lightgbm")

    payload = {
        "disclaimer": (
            "Research backtest only. This is not betting advice and nothing here should "
            "be used to wager."
        ),
        "reused_phase4_predictions": True,
        "priced_eval_run": _scored_eval_run(cfg),
        "eval_window": [str(cfg.windows.eval_start), str(cfg.windows.eval_end)],
        "cards": int(preds["event_id"].nunique()),
        "fights": len(preds),
        "seed": cfg.seed,
        "settings": {k: sim_cfg[k] for k in sorted(sim_cfg)},
        "legs_priced": (
            "Moneyline only. The odds snapshot contains no method-of-victory or round "
            "prices, and none were invented."
        ),
        "within_card_residual_correlation": corr,
        "by_source": results,
    }

    out = cfg.paths.reports_dir / "parlay_metrics.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    for source, frame in all_top.items():
        frame.to_csv(cfg.paths.reports_dir / f"parlay_top_{source}.csv", index=False)

    _plot_bankroll(
        results, cfg.paths.reports_dir / "bankroll.png", float(sim_cfg["starting_bankroll"])
    )
    _plot_predicted_vs_realized(results, cfg.paths.reports_dir / "predicted_vs_realized.png")

    from ufc_predictor.backtest.readme_block import inject_parlay

    inject_parlay(payload, cfg)

    if verbose:
        c = corr.get("correlation")
        if c is not None:
            print(f"  within-card residual correlation: {c:+.4f} over {corr['n_pairs']:,} pairs")
        print(f"  wrote {out}\n")

    return payload


def _predicted_vs_realized(top: pd.DataFrame, min_edge: float) -> dict:
    """Compare modelled edge with what the top-ranked parlays actually returned."""

    def block(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"n": 0}
        return {
            "n": len(frame),
            "mean_predicted_ev": float(frame["ev"].mean()),
            "realized_roi": float(frame["realized_return"].mean()),
            "ev_minus_realized": float(frame["ev"].mean() - frame["realized_return"].mean()),
            "mean_predicted_prob": float(frame["p_sim"].mean()),
            "hit_rate": float(frame["won"].mean()),
            "mean_payout": float(frame["payout"].mean()),
        }

    out = {
        "all_top_ranked": block(top),
        "cleared_stake_threshold": block(top.loc[top["ev"] >= min_edge]),
        "note": (
            "realized_roi is the mean return per unit staked on the top-ranked parlays. "
            "A large positive mean_predicted_ev beside a negative realized_roi is the "
            "signature of an edge that was noise."
        ),
    }
    for k in (1, 2, 3):
        out[f"n_legs_{k}"] = block(top.loc[top["n_legs"] == k])
    return out


if __name__ == "__main__":  # pragma: no cover
    run()
