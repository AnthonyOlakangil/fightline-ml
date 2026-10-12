"""Scoring, charts, and ``reports/metrics.json``.

Every number the README publishes is produced here and read back from
``metrics.json`` -- nothing is typed by hand.

**The eval-run ledger.** Each time the evaluation window is scored, an entry is appended
to ``reports/eval_runs.json`` with a timestamp, the git SHA, and the headline figures.
A held-out period scored repeatedly while a design is adjusted stops being held out, so
the count is recorded rather than asserted, and the README states it.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: charts are generated in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ufc_predictor.backtest.metrics import (
    bootstrap_metric_ci,
    expected_calibration_error,
    paired_bootstrap_by_event,
    per_fight_log_loss,
    score,
)
from ufc_predictor.backtest.walkforward import (
    VARIANT_NAME,
    load_panel,
    run_walk_forward,
)
from ufc_predictor.config import Config, get_config

MODEL_COLUMNS = ("market", "elo", "lightgbm")
ALL_COLUMNS = (*MODEL_COLUMNS, VARIANT_NAME)

#: Charts use one consistent colour per model.
COLOURS = {
    "market": "#111827",
    "elo": "#9CA3AF",
    "lightgbm": "#2563EB",
    VARIANT_NAME: "#DC2626",
}


def _git_state() -> dict:
    """Full HEAD commit, its short form, and whether the working tree matches it.

    Untracked files count as dirty: an uncommitted module is still uncommitted code.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:  # pragma: no cover - git may be absent
        return {"commit": "unknown", "short": "unknown", "clean": False}
    return {"commit": commit, "short": commit[:7], "clean": status.strip() == ""}


def _window_slice(predictions: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    return predictions.loc[(predictions["date"] >= start) & (predictions["date"] <= end)]


def _score_window(frame: pd.DataFrame, cfg: Config, *, with_ci: bool) -> dict:
    """Score every model on one window, on identical rows."""
    out: dict[str, dict] = {}
    y = frame["label"].to_numpy(dtype=int)
    events = frame["event_id"].to_numpy()
    n_boot = int(cfg.backtest.get("bootstrap_resamples", 10000))
    level = float(cfg.backtest.get("bootstrap_ci", 0.95))

    for name in ALL_COLUMNS:
        col = f"p_{name}"
        available = frame[col].notna()
        sub = frame.loc[available]
        if sub.empty:
            continue
        s = score(sub["label"].to_numpy(dtype=int), sub[col].to_numpy(dtype=float))
        # The variant warms up, so it may cover fewer fights. Record that explicitly
        # rather than letting a different denominator pass unnoticed.
        s["covers_all_fights"] = bool(available.all())
        if not available.all():
            # The variant warms up, so it covers fewer fights than the other models.
            # Reporting the market restricted to those same fights makes the two
            # numbers comparable instead of inviting a cross-denominator read.
            s["market_on_same_subset"] = score(
                sub["label"].to_numpy(dtype=int), sub["p_market"].to_numpy(dtype=float)
            )
        if with_ci:
            s["log_loss_ci"] = bootstrap_metric_ci(
                sub["label"].to_numpy(dtype=float),
                sub[col].to_numpy(dtype=float),
                sub["event_id"].to_numpy(),
                metric="log_loss",
                n_resamples=n_boot,
                seed=cfg.seed,
                ci=level,
            )
        out[name] = s

    del y, events
    return out


def _paired_vs_market(frame: pd.DataFrame, cfg: Config) -> dict:
    """Paired, event-resampled comparison of each model against the market."""
    n_boot = int(cfg.backtest.get("bootstrap_resamples", 10000))
    level = float(cfg.backtest.get("bootstrap_ci", 0.95))
    out: dict[str, dict] = {}

    for name in ALL_COLUMNS:
        if name == "market":
            continue
        col = f"p_{name}"
        sub = frame.loc[frame[col].notna()]
        if sub.empty:
            continue
        y = sub["label"].to_numpy(dtype=float)
        loss_model = per_fight_log_loss(y, sub[col].to_numpy(dtype=float))
        loss_market = per_fight_log_loss(y, sub["p_market"].to_numpy(dtype=float))
        res = paired_bootstrap_by_event(
            loss_model,
            loss_market,
            sub["event_id"].to_numpy(),
            n_resamples=n_boot,
            seed=cfg.seed,
            ci=level,
        )
        res["n_fights_compared"] = len(sub)
        out[name] = res
    return out


def _by_year(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    years = sorted({d.year for d in frame["date"]})
    for year in years:
        sub = frame.loc[[d.year == year for d in frame["date"]]]
        rec: dict = {"year": year, "n": len(sub), "n_events": int(sub["event_id"].nunique())}
        for name in ALL_COLUMNS:
            col = f"p_{name}"
            s = sub.loc[sub[col].notna()]
            if s.empty:
                rec[f"{name}_log_loss"] = None
                continue
            rec[f"{name}_log_loss"] = float(
                per_fight_log_loss(
                    s["label"].to_numpy(dtype=float), s[col].to_numpy(dtype=float)
                ).mean()
            )
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _plot_calibration(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1, label="perfect")

    for name in ALL_COLUMNS:
        col = f"p_{name}"
        sub = frame.loc[frame[col].notna()]
        if sub.empty:
            continue
        _, table = expected_calibration_error(
            sub["label"].to_numpy(dtype=float), sub[col].to_numpy(dtype=float), n_bins=10
        )
        t = table.dropna(subset=["mean_predicted"])
        ax.plot(
            t["mean_predicted"],
            t["observed_rate"],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=COLOURS.get(name, "#000000"),
            label=name.replace("_", " "),
        )

    ax.set_xlabel("predicted probability (fighter 1 wins)")
    ax.set_ylabel("observed win rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_log_loss_by_year(rows: list[dict], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    years = [r["year"] for r in rows]
    for name in ALL_COLUMNS:
        vals = [r.get(f"{name}_log_loss") for r in rows]
        if all(v is None for v in vals):
            continue
        ax.plot(
            years,
            vals,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=COLOURS.get(name, "#000000"),
            label=name.replace("_", " "),
        )
    ax.axhline(np.log(2), linestyle=":", color="#9CA3AF", linewidth=1, label="coin flip")
    ax.set_xlabel("year")
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_paired_difference(paired: dict, path: Path, title: str) -> None:
    """Forest plot of each model's log-loss difference against the market."""
    names = [n for n in ALL_COLUMNS if n in paired]
    fig, ax = plt.subplots(figsize=(7.5, 1.3 + 0.9 * len(names)))

    for i, name in enumerate(names):
        r = paired[name]
        y = len(names) - 1 - i
        ax.plot([r["ci_lower"], r["ci_hi"]], [y, y], color=COLOURS.get(name, "#000"), linewidth=2.5)
        ax.plot([r["observed"]], [y], marker="o", markersize=7, color=COLOURS.get(name, "#000"))

    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in reversed(names)])
    ax.set_xlabel("mean log loss difference vs market  (negative = model better)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _eval_run_count(cfg: Config) -> int:
    """Current number of scored eval runs, without recording a new one."""
    path = cfg.paths.reports_dir / "eval_runs.json"
    if not path.is_file():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")).get("runs", []))


def _append_eval_ledger(cfg: Config, summary: dict, *, git: dict, reason: str) -> int:
    """Append to the eval-run ledger and return the new run count.

    ``git`` is the state captured *before* the walk-forward started, so the entry names
    the commit whose code and frozen settings produced these numbers.
    """
    path = cfg.paths.reports_dir / "eval_runs.json"
    entries: list[dict] = []
    if path.is_file():
        entries = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
    entries.append(
        {
            "run": len(entries) + 1,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "git_sha": git["short"],
            "git_commit": git["commit"],
            "worktree_clean": git["clean"],
            "reason": reason,
            "seed": cfg.seed,
            "headline": summary,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"runs": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(entries)


def run(
    cfg: Config | None = None,
    *,
    verbose: bool = True,
    through: dt.date | None = None,
    reason: str = "reproduction",
) -> dict:
    """Run the walk-forward backtest, score it, and write every artefact.

    Args:
        through: stop the walk at this date. Passing ``dev_end`` exercises this entire
            reporting path without scoring the evaluation window -- used to debug the
            report before the single scored eval run, so that bugs are not paid for
            out of the held-out period. In that mode nothing is appended to the
            eval-run ledger and no eval figure is written.
        reason: why the evaluation window is being scored, recorded in the ledger.

    Scoring the evaluation window is refused from a dirty working tree: the ledger entry
    must name a commit that contains exactly the code and frozen settings that ran.
    """
    cfg = cfg or get_config()
    dry_run = through is not None and through <= cfg.windows.dev_end
    git = _git_state()
    if not dry_run:
        if not reason or not reason.strip():
            raise ValueError("scoring the evaluation window needs a reason for the ledger")
        if not git["clean"]:
            raise RuntimeError(
                "refusing to score the evaluation window with uncommitted changes: commit "
                "the code and settings first so the eval ledger can name the exact commit"
            )
    predictions, folds = run_walk_forward(cfg, verbose=verbose, through=through)

    reports = cfg.paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    suffix = "_dryrun" if dry_run else ""
    predictions.to_csv(reports / f"walkforward_predictions{suffix}.csv", index=False)
    folds.to_csv(reports / f"walkforward_folds{suffix}.csv", index=False)

    dev = _window_slice(predictions, cfg.windows.dev_start, cfg.windows.dev_end)
    ev = _window_slice(predictions, cfg.windows.eval_start, cfg.windows.eval_end)

    total = len(predictions)
    # Scorable fights actually in range for this run, so a dry run does not report the
    # untouched evaluation window as "warm-up".
    panel_scorable = load_panel(cfg)
    panel_scorable = panel_scorable.loc[panel_scorable["scorable"]]
    if through is not None:
        panel_scorable = panel_scorable.loc[panel_scorable["date"] <= through]
    in_range = len(panel_scorable)

    reconciliation = {
        "total_priced_decisive": 6790,
        "in_range_for_this_run": in_range,
        "scored_total": total,
        "dev": len(dev),
        "gap": int(
            (
                (predictions["date"] > cfg.windows.dev_end)
                & (predictions["date"] < cfg.windows.eval_start)
            ).sum()
        ),
        "eval": len(ev),
        "unscored_warmup": in_range - total,
        "note": (
            "A fight before the walk-forward has min_train_fights of prior history "
            "cannot be scored; those are counted as warm-up. dev + gap + eval equals "
            "scored_total, and scored_total + unscored_warmup equals the fights in range."
        ),
    }
    assert total + reconciliation["unscored_warmup"] == in_range
    assert reconciliation["dev"] + reconciliation["gap"] + reconciliation["eval"] == total

    results = {"dev": _score_window(dev, cfg, with_ci=True)}
    paired = {"dev": _paired_vs_market(dev, cfg)}
    if not ev.empty:
        results["eval"] = _score_window(ev, cfg, with_ci=True)
        paired["eval"] = _paired_vs_market(ev, cfg)
    by_year = _by_year(predictions)

    _plot_calibration(dev, reports / f"calibration_dev{suffix}.png", "Calibration — dev window")
    _plot_log_loss_by_year(by_year, reports / f"log_loss_by_year{suffix}.png", "Log loss by year")
    if not ev.empty:
        _plot_calibration(ev, reports / "calibration.png", "Calibration — evaluation window")
        _plot_paired_difference(
            paired["eval"],
            reports / "paired_vs_market.png",
            "Log loss vs market (paired, 95% CI, resampled by event)",
        )

    if dry_run:
        run_count = _eval_run_count(cfg)
    else:
        headline = {name: round(results["eval"][name]["log_loss"], 4) for name in results["eval"]}
        run_count = _append_eval_ledger(cfg, headline, git=git, reason=reason)

    payload = {
        "dry_run_dev_only": dry_run,
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": git["short"],
        "git_commit": git["commit"],
        "seed": cfg.seed,
        "eval_run_count": run_count,
        "windows": {
            "dev": [str(cfg.windows.dev_start), str(cfg.windows.dev_end)],
            "eval": [str(cfg.windows.eval_start), str(cfg.windows.eval_end)],
        },
        "reconciliation": reconciliation,
        "walk_forward": {
            "retrain_every_n_events": int(cfg.backtest["retrain_every_n_events"]),
            "min_train_fights": int(cfg.backtest["min_train_fights"]),
            "n_folds": len(folds),
            "events_scored": int(predictions["event_id"].nunique()),
            "dev_events": int(dev["event_id"].nunique()),
            "eval_events": int(ev["event_id"].nunique()),
            "median_trees_after_early_stopping": float(folds["trees"].median()),
            "hyperparameters_frozen_from_dev": True,
            "lightgbm": dict(cfg.lightgbm),
            "lightgbm_fit": dict(cfg.lightgbm_fit),
        },
        "bootstrap": {
            "n_resamples": int(cfg.backtest["bootstrap_resamples"]),
            "ci_level": float(cfg.backtest["bootstrap_ci"]),
            "resampling_unit": "event",
            "why_event": (
                "Fights on one card share conditions and correlated model errors, so "
                "resampling fights would treat correlated observations as independent "
                "and understate the interval."
            ),
        },
        "results": results,
        "paired_vs_market": paired,
        "by_year": by_year,
        "variant_note": (
            f"{VARIANT_NAME} consumes the closing line as an input and therefore cannot "
            "support a 'beats the market' claim. It exists to test whether the model's "
            "features add information beyond the market, and it warms up on accumulated "
            "out-of-sample predictions, so it covers fewer fights than the other models."
        ),
    }

    path = reports / f"metrics{suffix}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

    if not dry_run:
        from ufc_predictor.backtest.readme_block import inject

        inject(payload, cfg)

    if verbose:
        _print_summary(payload)
        print(f"  wrote {path}")
        print(f"  eval run count: {run_count}\n")
    return payload


def _print_summary(payload: dict) -> None:
    r = payload["reconciliation"]
    print("  reconciliation:")
    print(
        f"    dev {r['dev']:,} + gap {r['gap']} + eval {r['eval']:,} "
        f"= {r['scored_total']:,} scored; warm-up unscored {r['unscored_warmup']} "
        f"(total priced decisive {r['total_priced_decisive']:,})"
    )
    for window, res in payload["results"].items():
        print(f"\n  {window.upper()} window  (n = {res['market']['n']:,})")
        print(f"    {'model':<20} {'log loss':>9} {'brier':>8} {'auc':>7} {'acc':>7} {'ece':>7}")
        for name, s in res.items():
            print(
                f"    {name:<20} {s['log_loss']:>9.4f} {s['brier']:>8.4f} "
                f"{s['auc']:>7.4f} {s['accuracy']:>7.4f} {s['ece']:>7.4f}"
                + ("" if s["covers_all_fights"] else f"   (n={s['n']:,})")
            )
        print("\n    paired vs market (95% CI, event-resampled):")
        for name, p in payload["paired_vs_market"][window].items():
            print(
                f"      {name:<20} {p['observed']:+.4f}  "
                f"[{p['ci_lower']:+.4f}, {p['ci_hi']:+.4f}]  {p['interpretation']}"
            )


if __name__ == "__main__":  # pragma: no cover
    run()


def regenerate(cfg: Config | None = None, *, verbose: bool = True) -> dict:
    """Redraw charts and refresh the README from artefacts already on disk.

    Deliberately does **not** re-run the walk-forward, so `make report` cannot score the
    evaluation window a second time or increment the eval-run ledger. Re-scoring is only
    ever triggered by `make backtest`.
    """
    cfg = cfg or get_config()
    reports = cfg.paths.reports_dir
    metrics_path = reports / "metrics.json"
    preds_path = reports / "walkforward_predictions.csv"
    if not metrics_path.is_file() or not preds_path.is_file():
        raise FileNotFoundError(
            "metrics.json or walkforward_predictions.csv is missing; run `make backtest` first"
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(preds_path, parse_dates=["date"])
    predictions["date"] = predictions["date"].dt.date

    dev = _window_slice(predictions, cfg.windows.dev_start, cfg.windows.dev_end)
    ev = _window_slice(predictions, cfg.windows.eval_start, cfg.windows.eval_end)

    _plot_calibration(ev, reports / "calibration.png", "Calibration — evaluation window")
    _plot_calibration(dev, reports / "calibration_dev.png", "Calibration — dev window")
    _plot_log_loss_by_year(metrics["by_year"], reports / "log_loss_by_year.png", "Log loss by year")
    _plot_paired_difference(
        metrics["paired_vs_market"]["eval"],
        reports / "paired_vs_market.png",
        "Log loss vs market (paired, 95% CI, resampled by event)",
    )

    from ufc_predictor.backtest.readme_block import inject, inject_parlay

    inject(metrics, cfg)
    if (reports / "parlay_metrics.json").is_file():
        inject_parlay(cfg=cfg)
    if verbose:
        print("  charts redrawn and README results/parlay blocks refreshed from metrics")
        print(f"  eval run count unchanged: {metrics['eval_run_count']}\n")
    return metrics
