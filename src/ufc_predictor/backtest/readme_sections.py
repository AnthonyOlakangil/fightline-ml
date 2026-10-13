"""The remaining generated README sections, each rendered from a JSON file in ``reports/``.

``readme_block`` owns the results and parlay blocks. This module owns the rest, so that
every number the README states is produced by code and checked by
``tests/test_readme_sections.py`` rather than typed by hand:

* ``HEADLINE``    -- the opening result paragraph, from ``metrics.json``;
* ``DIAGNOSTICS`` -- the shuffle test and its control, from ``leakage_diagnostics.json``;
* ``TUNING``      -- the hyperparameter search, from ``tuning.json``;
* ``DEVCARDS``    -- the dev-window sanity table, from ``model_cards.json``;
* ``LEAK``        -- evidence for the attribute leak and how every headline figure moved
  between evaluation run 1 and run 2, from ``archive/eval_run_1/`` and the current files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ufc_predictor.backtest.readme_block import LABELS, signed
from ufc_predictor.backtest.walkforward import VARIANT_NAME
from ufc_predictor.config import Config, get_config

LN2 = math.log(2)
EN_DASH = "\u2013"  # escaped so the linter does not mistake it for a hyphen
RUN_1_ARCHIVE = Path("archive") / "eval_run_1"


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}:START -->", f"<!-- {name}:END -->"


def _wrap(name: str, lines: list[str]) -> str:
    start, end = _markers(name)
    return "\n".join([start, *lines, end])


def _ci(p: dict) -> str:
    return f"{signed(p['observed'])} [{signed(p['ci_lower'])}, {signed(p['ci_hi'])}]"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_headline(metrics: dict) -> str:
    ev = metrics["results"]["eval"]
    p = metrics["paired_vs_market"]["eval"]["lightgbm"]
    if p["beats_market"]:
        verdict = "the model beats the closing line"
        gap = "an advantage"
    elif p["ci_lower"] <= 0 <= p["ci_hi"]:
        verdict = "the model is indistinguishable from the closing line"
        gap = "a difference"
    else:
        verdict = "the model does not beat the closing line"
        gap = "a gap"
    excludes = "excludes" if not p["ci_lower"] <= 0 <= p["ci_hi"] else "contains"
    return _wrap(
        "HEADLINE",
        [
            f"**Headline result: {verdict}.** Over {ev['market']['n']:,} held-out fights, the "
            f"vig-free market scores {ev['market']['log_loss']:.4f} log loss against the "
            f"model's {ev['lightgbm']['log_loss']:.4f} — {gap} of {p['observed']:+.4f} with a "
            f"95% confidence interval of [{p['ci_lower']:+.4f}, {p['ci_hi']:+.4f}] that "
            f"{excludes} zero. This is evaluation run {metrics['eval_run_count']}, recorded in "
            "`reports/eval_runs.json`.",
        ],
    )


def render_diagnostics(d: dict) -> str:
    sp, sh, ctrl = d["split"], d["shuffle"], d["control"]
    chance = d["chance_log_loss"]
    lines = [
        f"Run entirely inside the dev window (train {sp['train_span'][0][:4]} → "
        f"{sp['train_span'][1][:7]}, test {sp['test_span'][0][:7]} → {sp['test_span'][1][:7]}, "
        f"n = {sp['n_test']:,}) with {d['model']}. The held-out period is never touched by a "
        "diagnostic:",
        "",
        "| Outcomes | AUC | Log loss |",
        "|---|---|---|",
        f"| **Shuffled** (mean of {sh['n_draws']} draws) | **{sh['mean_auc']:.4f}** "
        f"(sd {sh['sd_auc']:.4f}, range {min(sh['aucs']):.3f}{EN_DASH}{max(sh['aucs']):.3f}) | "
        f"**{sh['mean_log_loss']:.3f}** (range {min(sh['log_losses']):.3f}{EN_DASH}"
        f"{max(sh['log_losses']):.3f}) |",
        f"| *Chance* | *0.5000* | *{chance:.4f}* |",
        f"| **Real (control)** | **{ctrl['auc']:.4f}** | **{ctrl['log_loss']:.4f}** |",
        "",
    ]
    z = (ctrl["auc"] - sh["mean_auc"]) / sh["sd_auc"]
    text = (
        f"Shuffled outcomes collapse to chance AUC, and the control sits {z:.1f} shuffle "
        "standard deviations above them, so real outcomes stay learnable."
    )
    if min(sh["log_losses"]) > chance:
        text += " Every shuffled draw's log loss is worse than chance."
    if ctrl["log_loss"] > chance:
        text += (
            " On real outcomes this fixed, uncalibrated model is overconfident: its log loss "
            "is worse than chance even though it ranks fights better than chance, so here the "
            "separation between shuffled and real outcomes is carried by AUC, not log loss."
        )
    lines.append(text)
    return _wrap("DIAGNOSTICS", lines)


def render_tuning(t: dict) -> str:
    s = t["selected"]
    span = t["validation_span"]
    params = ", ".join(
        f"`{k}={s[k]}`"
        for k in (
            "max_depth",
            "num_leaves",
            "min_child_samples",
            "learning_rate",
            "colsample_bytree",
            "reg_lambda",
        )
    )
    return _wrap(
        "TUNING",
        [
            f"- **{t['n_candidates']} candidates searched**, biased toward strong "
            f"regularization, on a chronological holdout inside the dev window (train "
            f"{t['train_rows']:,} fights, validate {t['validation_rows']:,} fights "
            f"{span[0]} → {span[1]}). Selected: {params}. Early stopping kept "
            f"{s['best_iteration']} trees on the tuning split.",
            f"- Its validation log loss is **{s['log_loss']:.4f}**; a coin flip scores "
            f"{LN2:.4f}, and the settings in force before this search scored "
            f"{t['config_baseline']['log_loss']:.4f} on the same split.",
            f"- **Ties are broken by a coded rule, not by the score.** {t['n_tied_within_tolerance']} "
            f"candidates scored within {t['noise_tolerance']} log loss of the best, which "
            f"is noise at {t['validation_rows']:,} validation fights, so a coded rule picks "
            "the *most regularized* model among those tied. "
            + (
                "Here the rule's pick is also the lowest-scoring candidate."
                if s == t["lowest_log_loss_candidate"]
                else "Here the rule's pick is not the lowest-scoring candidate."
            ),
        ],
    )


def render_dev_cards(c: dict) -> str:
    span = c["holdout_span"]
    lines = [
        f"For reference only, on a chronological dev-window holdout (train "
        f"{c['train_rows_all_decisive']:,} decisive fights before {span[0]}; score the "
        f"{c['scored_rows_priced_only']:,} **priced** fights {span[0]} → {span[1]}, identical "
        f"rows for every model, {c['total_features_available']} features available):",
        "",
        "| Model | Log loss | Brier | AUC | Accuracy |",
        "|---|---|---|---|---|",
    ]
    for name, label in (("market", "Market"), ("lightgbm", "LightGBM"), ("elo", "Elo")):
        h = c["models"][name]["dev_holdout"]
        lines.append(
            f"| {label} | {h['log_loss']:.4f} | {h['brier']:.4f} | {h['auc']:.4f} | "
            f"{h['accuracy']:.4f} |"
        )
    return _wrap("DEVCARDS", lines)


def render_leak(run1: dict, run2: dict, parlay1: dict, parlay2: dict, diagnostics: dict) -> str:
    ev = diagnostics["attribute_leak"]
    a, b = run1["eval_run_count"], run2["eval_run_count"]
    lines = [
        f"**Evidence, dev window {ev['window'][0]} → {ev['window'][1]}** "
        f"({ev['decisive_bouts']:,} decisive bouts):",
        "",
        "| Attribute (as run 1 used it) | Bouts where only one fighter's value is blank | "
        "Blank side's win rate |",
        "|---|---:|---:|",
    ]
    names = {"reach": "Reach", "age": "Age (from date of birth)", "height": "Height"}
    for key in ("reach", "age", "height"):
        r = ev["per_attribute"][key]
        rate = "—" if r["blank_side_win_rate"] is None else f"{r['blank_side_win_rate']:.1%}"
        lines.append(f"| {names[key]} | {r['bouts_one_side_blank']:,} | {rate} |")
    blank, full = ev["pairs_with_a_one_sided_blank"], ev["complete_pairs"]
    lines += [
        "",
        f"Blanking both sides would not have closed the leak. In pairs with a one-sided "
        f"blank the less-experienced fighter won **{blank['less_experienced_win_rate']:.1%}** "
        f"of {blank['bouts']:,} bouts, against "
        f"**{full['less_experienced_win_rate']:.1%}** of {full['bouts']:,} complete pairs, "
        'so "this pair has a blank" plus prior experience still points at the fighter who '
        "was later cut.",
        "",
        f"**How much each number changed** (run {a} → run {b}; both on the same "
        f"{run2['results']['eval']['market']['n']:,} eval fights):",
        "",
        f"| Figure | Run {a} (leaky) | Run {b} (leak-free) | Change |",
        "|---|---:|---:|---:|",
    ]

    def ll_row(window: str, name: str) -> str:
        x = run1["results"][window][name]["log_loss"]
        y = run2["results"][window][name]["log_loss"]
        return (
            f"| {window.capitalize()} log loss — {LABELS[name]} | {x:.4f} | {y:.4f} | "
            f"{y - x:+.4f} |"
        )

    def paired_row(window: str, name: str) -> str:
        x = run1["paired_vs_market"][window][name]
        y = run2["paired_vs_market"][window][name]
        return (
            f"| {window.capitalize()} Δ vs market — {LABELS[name]} | {_ci(x)} | {_ci(y)} | "
            f"{y['observed'] - x['observed']:+.4f} |"
        )

    for name in ("market", "lightgbm", "elo", VARIANT_NAME):
        lines.append(ll_row("eval", name))
    for name in ("lightgbm", "elo", VARIANT_NAME):
        lines.append(paired_row("eval", name))
    lines.append(ll_row("dev", "lightgbm"))
    lines.append(paired_row("dev", VARIANT_NAME))

    for source, label in (("lightgbm", "LightGBM"), ("market_plus_model", "market + model")):
        x, y = parlay1["by_source"][source], parlay2["by_source"][source]
        px = x["predicted_vs_realized"]["all_top_ranked"]
        py = y["predicted_vs_realized"]["all_top_ranked"]
        lines += [
            f"| Parlays ranked by {label} — top-ranked | {x['n_top_ranked']:,} | "
            f"{y['n_top_ranked']:,} | {y['n_top_ranked'] - x['n_top_ranked']:+,} |",
            f"| Parlays ranked by {label} — placed in bankroll replay | "
            f"{x['bankroll']['n_bets']:,} | {y['bankroll']['n_bets']:,} | "
            f"{y['bankroll']['n_bets'] - x['bankroll']['n_bets']:+,} |",
            f"| Parlays ranked by {label} — predicted EV, flat stakes | "
            f"{px['mean_predicted_ev']:+.1%} | "
            f"{py['mean_predicted_ev']:+.1%} | "
            f"{(py['mean_predicted_ev'] - px['mean_predicted_ev']) * 100:+.1f} pp |",
            f"| Parlays ranked by {label} — realised ROI, flat stakes | "
            f"{px['realized_roi']:+.1%} | "
            f"{py['realized_roi']:+.1%} | "
            f"{(py['realized_roi'] - px['realized_roi']) * 100:+.1f} pp |",
            f"| Parlays ranked by {label} — ending bankroll (from "
            f"£{x['bankroll']['starting_bankroll']:,.0f}) | "
            f"£{x['bankroll']['ending_bankroll']:,.2f} | £{y['bankroll']['ending_bankroll']:,.2f} "
            f"| {y['bankroll']['ending_bankroll'] - x['bankroll']['ending_bankroll']:+,.2f} |",
        ]
    return _wrap("LEAK", lines)


def render_post_hoc(ph: dict, metrics: dict) -> str:
    vd, cert = ph["variant_decomposition"], ph["lightgbm_certainties"]
    rec, add, tot = (
        vd["recalibrated_minus_market"],
        vd["variant_minus_recalibrated"],
        vd["variant_minus_market"],
    )
    coef = vd["eval_fold_coefficients"]
    lines = [
        "### What the variant's gain is made of",
        "",
        "*A post-hoc check on the stored predictions, prompted by the second review "
        "([`reports/review_run2.md`](reports/review_run2.md)). No evaluated model is refitted; "
        "the market-only control is new, and is listed in the glimpse disclosure.*",
        "",
        "The variant is a walk-forward logistic stacker on the market's and LightGBM's "
        "log-odds. Replaying that protocol on the stored predictions reproduces every variant "
        "prediction exactly; replaying it on the market's log-odds alone separates what the "
        "market contributes from what the model adds.",
        "",
        f"| Comparison on the {vd['n_fights']:,} eval fights (paired, event-resampled) | "
        "Δ log loss | 95% CI | p |",
        "|---|---:|---|---:|",
    ]
    for label, p in (
        ("Market recalibrated on its own past errors, vs the vig-free market", rec),
        ("Market + model, vs the recalibrated market", add),
        ("Market + model, vs the vig-free market (as reported above)", tot),
    ):
        lines.append(
            f"| {label} | {signed(p['observed'])} | [{signed(p['ci_lower'])}, "
            f"{signed(p['ci_hi'])}] | {p['p_value_two_sided']:.3f} |"
        )
    lg, mk = coef["variant_weight_on_logit_lightgbm"], coef["variant_weight_on_logit_market"]
    slope = coef["recalibration_slope_on_logit_market"]
    lines.append("")
    if add["ci_lower"] <= 0 <= add["ci_hi"]:
        text = (
            "Adding LightGBM to the recalibrated market changes nothing that can be told from "
            f"zero, and the stacker weights LightGBM's log-odds at {lg[0]:.3f}{EN_DASH}{lg[1]:.3f} "
            f"against {mk[0]:.2f}{EN_DASH}{mk[1]:.2f} for the market's. **The variant's gain is the "
            "closing line correcting itself, not information from the features.**"
        )
    else:
        text = (
            "Adding LightGBM to the recalibrated market moves log loss by "
            f"{signed(add['observed'])}, an interval that excludes zero."
        )
    if rec["ci_hi"] < 0:
        text += (
            f" Proportional de-vig leaves a favourite{EN_DASH}longshot bias, which a recalibration "
            f"slope of {slope[0]:.2f}{EN_DASH}{slope[1]:.2f} undoes. So the vig-free baseline used "
            "throughout this README is slightly beatable without any model: it is a "
            "marginally *easier* bar than a recalibrated reading of the same closing line, "
            "and the feature model still fails to clear it."
        )
    lines.append(text)

    cost = cert["cost_of_a_miss_nats"]
    lines += [
        "",
        "### Where LightGBM's excess log loss comes from",
        "",
        "| Window | Fights | LightGBM predictions within "
        f"{cert['tolerance']:g} of 0 or 1 | Of those, wrong | Log loss | Log loss without them "
        "| Share of all log loss from the wrong ones |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for w in ("dev", "eval"):
        c = cert[w]
        lines.append(
            f"| {w.capitalize()} | {c['n']:,} | {c['certain']} | {c['certain_and_wrong']} | "
            f"{c['log_loss']:.4f} | {c['log_loss_without_certainties']:.4f} | "
            f"{c['share_of_total_loss_from_wrong_certainties']:.1%} |"
        )
    dev, ev = cert["dev"], cert["eval"]
    market_ev = metrics["results"]["eval"]["market"]["log_loss"]
    dev_clause = (
        "still worse than a coin flip"
        if dev["log_loss_without_certainties"] >= LN2
        else "better than a coin flip"
    )
    lines += [
        "",
        "Each walk-forward fold calibrates LightGBM with isotonic regression, a step "
        "function. When its most extreme step holds only wins or only losses it outputs "
        f"exactly 1 or 0, which scoring clips to {math.exp(-cost):g}; a miss then costs "
        f"{cost:.1f} nats, {cost / LN2:.0f} times what a coin flip pays. Without those "
        f"predictions LightGBM scores {dev['log_loss_without_certainties']:.4f} in the dev "
        f"window, {dev_clause}, and {ev['log_loss_without_certainties']:.4f} on eval, still "
        f"far behind the market's {market_ev:.4f}. This is a defect in the fitting regime, "
        "not leakage, and it does not change the headline. Fixing it changes the model, so "
        "it would need a third scored run; that has not been done.",
    ]
    return _wrap("POSTHOC", lines)


# ---------------------------------------------------------------------------
# Loading and injection
# ---------------------------------------------------------------------------


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; it is committed, so the checkout is broken")
    return json.loads(path.read_text(encoding="utf-8"))


def render_all(cfg: Config | None = None) -> dict[str, str]:
    """Every section, keyed by marker name, rendered from the files on disk."""
    cfg = cfg or get_config()
    r = cfg.paths.reports_dir
    metrics = _load(r / "metrics.json")
    diagnostics = _load(r / "leakage_diagnostics.json")
    return {
        "HEADLINE": render_headline(metrics),
        "DIAGNOSTICS": render_diagnostics(diagnostics),
        "TUNING": render_tuning(_load(r / "tuning.json")),
        "DEVCARDS": render_dev_cards(_load(r / "model_cards.json")),
        "POSTHOC": render_post_hoc(_load(r / "post_hoc.json"), metrics),
        "LEAK": render_leak(
            _load(r / RUN_1_ARCHIVE / "metrics.json"),
            metrics,
            _load(r / RUN_1_ARCHIVE / "parlay_metrics.json"),
            _load(r / "parlay_metrics.json"),
            diagnostics,
        ),
    }


def extract(readme: str, name: str) -> str:
    """The README's current text for one section, markers included."""
    start, end = _markers(name)
    if start not in readme or end not in readme:
        raise RuntimeError(f"README.md is missing the {start} / {end} markers")
    _, rest = readme.split(start, 1)
    body, _ = rest.split(end, 1)
    return start + body + end


def inject_all(cfg: Config | None = None) -> None:
    """Rewrite every section in README.md between its markers."""
    cfg = cfg or get_config()
    readme = cfg.paths.raw_dir.parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8")
    for name, block in render_all(cfg).items():
        text = text.replace(extract(text, name), block)
    readme.write_text(text, encoding="utf-8")
