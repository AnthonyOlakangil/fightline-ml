"""Render the README's results section from ``reports/metrics.json``.

Every figure the README publishes is generated here and written between two markers, so
no number is ever typed by hand. ``tests/test_report.py`` re-renders the block and
asserts it matches the file on disk, which means a stale README fails the suite rather
than quietly misreporting the result.
"""

from __future__ import annotations

import json

from ufc_predictor.backtest.walkforward import VARIANT_NAME
from ufc_predictor.config import Config, get_config

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

ORDER = ("market", "lightgbm", "elo", VARIANT_NAME)
LABELS = {
    "market": "Market (vig-free closing line)",
    "lightgbm": "LightGBM",
    "elo": "Elo",
    VARIANT_NAME: "Market + model *(variant)*",
}


def _fmt(x: float | None, places: int = 4) -> str:
    return "—" if x is None else f"{x:.{places}f}"


def render(metrics: dict) -> str:
    """Build the markdown block."""
    ev = metrics["results"]["eval"]
    paired = metrics["paired_vs_market"]["eval"]
    wf = metrics["walk_forward"]
    rec = metrics["reconciliation"]
    boot = metrics["bootstrap"]

    n_eval = ev["market"]["n"]
    lines: list[str] = [START, ""]

    lines += [
        "### Results — held-out evaluation window",
        "",
        f"Walk-forward over **{wf['eval_events']} events / {n_eval:,} fights** "
        f"({metrics['windows']['eval'][0]} → {metrics['windows']['eval'][1]}). "
        f"Hyperparameters frozen from the dev window; **{wf['n_folds']} refits** at a "
        f"cadence of every {wf['retrain_every_n_events']} events. "
        f"Seed {metrics['seed']}. Evaluation runs to date: **{metrics['eval_run_count']}**.",
        "",
        "| Model | n | Log loss | Brier | AUC | Accuracy | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        if name not in ev:
            continue
        s = ev[name]
        lines.append(
            f"| {LABELS[name]} | {s['n']:,} | **{_fmt(s['log_loss'])}** | "
            f"{_fmt(s['brier'])} | {_fmt(s['auc'])} | {_fmt(s['accuracy'])} | "
            f"{_fmt(s['ece'])} |"
        )

    variant = ev.get(VARIANT_NAME)
    if variant is not None and not variant["covers_all_fights"]:
        same = variant["market_on_same_subset"]
        lines += [
            "",
            f"The variant warms up on accumulated out-of-sample predictions, so it covers "
            f"{variant['n']:,} of {n_eval:,} fights. On **those same fights** the market "
            f"scores {_fmt(same['log_loss'])} and the variant {_fmt(variant['log_loss'])}, "
            "which is the like-for-like comparison.",
        ]

    lines += [
        "",
        "### Does any model beat the market?",
        "",
        f"Paired per-fight log-loss difference against the market, with a "
        f"{int(boot['ci_level'] * 100)}% confidence interval from "
        f"{boot['n_resamples']:,} bootstrap resamples **of events, not fights** "
        f"({paired[next(iter(paired))]['n_events_resampled']} events). Negative means the "
        "model is better.",
        "",
        "| Model | Δ log loss vs market | 95% CI | Verdict |",
        "|---|---:|---|---|",
    ]
    for name in ORDER:
        if name == "market" or name not in paired:
            continue
        p = paired[name]
        verdict = (
            "**beats the market**"
            if p["beats_market"]
            else ("no difference shown" if p["ci_lower"] <= 0 <= p["ci_hi"] else "market wins")
        )
        lines.append(
            f"| {LABELS[name]} | {p['observed']:+.4f} | "
            f"[{p['ci_lower']:+.4f}, {p['ci_hi']:+.4f}] | {verdict} |"
        )

    lgbm_p = paired.get("lightgbm")
    var_p = paired.get(VARIANT_NAME)
    lines += ["", "**Reading this honestly:**", ""]
    if lgbm_p is not None:
        if lgbm_p["beats_market"]:
            lines.append(
                f"- The feature-based model **beats** the closing line by "
                f"{abs(lgbm_p['observed']):.4f} nats per fight, and the interval excludes zero."
            )
        elif lgbm_p["ci_lower"] <= 0 <= lgbm_p["ci_hi"]:
            lines.append(
                "- The feature-based model is **not distinguishable** from the closing "
                "line: the interval contains zero."
            )
        else:
            lines.append(
                f"- The closing line **beats** the feature-based model by "
                f"{lgbm_p['observed']:.4f} nats per fight. The model does not beat the market, "
                "and this is the project's headline result."
            )
    if var_p is not None:
        direction = "does" if var_p["beats_market"] else "does not"
        lines.append(
            f"- The market-anchored variant {direction} improve on the market "
            f"({var_p['observed']:+.4f}, CI [{var_p['ci_lower']:+.4f}, {var_p['ci_hi']:+.4f}]). "
            "Because it takes the closing line as an input it cannot support a "
            "'beats the market' claim on its own; it is evidence about whether the "
            "features carry information the market has not already priced."
        )

    lines += [
        "",
        "### Window reconciliation",
        "",
        f"Every priced, decisive fight is accounted for: "
        f"**{rec['dev']:,} dev + {rec['gap']} gap + {rec['eval']:,} eval = "
        f"{rec['scored_total']:,} scored**, plus {rec['unscored_warmup']} unscored during "
        f"walk-forward warm-up, totalling {rec['total_priced_decisive']:,}.",
        "",
        "| Window | Span | Events | Fights | Market log loss | LightGBM log loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, label in (("dev", "Dev (tuning)"), ("eval", "Eval (held out)")):
        res = metrics["results"][key]
        span = metrics["windows"][key]
        n_events = wf[f"{key}_events"]
        lines.append(
            f"| {label} | {span[0]} → {span[1]} | {n_events} | {res['market']['n']:,} | "
            f"{_fmt(res['market']['log_loss'])} | {_fmt(res['lightgbm']['log_loss'])} |"
        )

    lines += [
        "",
        "### Log loss by year",
        "",
        "| Year | Fights | Market | LightGBM | Elo | Market + model |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["by_year"]:
        lines.append(
            f"| {row['year']} | {row['n']:,} | {_fmt(row['market_log_loss'])} | "
            f"{_fmt(row['lightgbm_log_loss'])} | {_fmt(row['elo_log_loss'])} | "
            f"{_fmt(row[f'{VARIANT_NAME}_log_loss'])} |"
        )

    lines += [
        "",
        "![Calibration, evaluation window](reports/calibration.png)",
        "",
        "![Log loss vs market, paired with 95% CI](reports/paired_vs_market.png)",
        "",
        "![Log loss by year](reports/log_loss_by_year.png)",
        "",
        f"*Generated from `reports/metrics.json` (git `{metrics['git_sha']}`, "
        f"{metrics['generated_at_utc']}Z) by `make report`. No figure above is typed by hand.*",
        "",
        END,
    ]
    return "\n".join(lines)


def inject(metrics: dict | None = None, cfg: Config | None = None) -> str:
    """Write the rendered block into README.md between the markers."""
    cfg = cfg or get_config()
    if metrics is None:
        metrics = json.loads((cfg.paths.reports_dir / "metrics.json").read_text(encoding="utf-8"))

    block = render(metrics)
    readme = cfg.paths.raw_dir.parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8")

    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        updated = head + block + tail
    else:
        raise RuntimeError(
            f"README.md is missing the {START} / {END} markers; add them where the "
            "results section should appear"
        )
    readme.write_text(updated, encoding="utf-8")
    return block
