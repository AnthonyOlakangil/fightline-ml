"""The remaining generated README sections, each rendered from a JSON file in ``reports/``.

``readme_block`` owns the results and parlay blocks. This module owns the rest, so that
every number the README states is produced by code and checked by
``tests/test_readme_sections.py`` rather than typed by hand:

* ``HEADLINE``    -- the opening result paragraph, from ``metrics.json``;
* ``DIAGNOSTICS`` -- the shuffle test and its control, from ``leakage_diagnostics.json``;
* ``TUNING``      -- the hyperparameter search, from ``tuning.json``;
* ``DEVCARDS``    -- the dev-window sanity table, from ``model_cards.json``;
* ``LEAK``        -- evidence for the attribute leak and how every headline figure moved
  between evaluation run 1 and run 2, from ``archive/eval_run_1/`` and the current files;
* ``P8_*``        -- the Phase 8 sections, rendered by ``readme_phase8``.

    uv run python -m ufc_predictor.backtest.readme_sections   # rewrite every section

The forward-test workflow runs that after each scheduled step, so the README's forward
table follows ``reports/forward/`` without re-scoring anything.
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
#: Earlier evaluation runs, archived from git so the README can show every run.
RUN_ARCHIVES = (RUN_1_ARCHIVE, Path("archive") / "eval_run_2")


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


def render_leak(diagnostics: dict) -> str:
    ev = diagnostics["attribute_leak"]
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
    ]
    return _wrap("LEAK", lines)


def render_calibration_bound(diagnostics: dict) -> str:
    c = diagnostics["calibration_bound"]
    lo, hi = c["bounds"]
    heavy, top = c["favourites_at_or_above_0_85"], c["top_1pct_favourites"]
    return _wrap(
        "CALBOUND",
        [
            f"**Evidence for [{lo:g}, {hi:g}], dev window {c['window'][0]} → {c['window'][1]}** "
            f"({c['n']:,} priced fights; the bound was fixed before this was computed, and "
            "was not searched):",
            "",
            "| Dev-window fact | Value |",
            "|---|---:|",
            f"| Most extreme vig-free closing probabilities | {c['market_min']:.3f} and "
            f"{c['market_max']:.3f} |",
            f"| Fights where the closing line itself fell outside the bound | "
            f"{c['market_outside_bounds']:,} |",
            f"| Favourites priced at 0.85 or more: bouts, share they lost | "
            f"{heavy['n']:,}, {heavy['lost']:.1%} |",
            f"| The top 1% of favourites (priced {top['min_probability']:.3f}+): bouts, share "
            f"they lost | {top['n']:,}, {top['lost']:.1%} |",
            f"| Cost of one miss, unbounded → at the bound | "
            f"{c['cost_of_a_miss_unbounded_nats']:.1f} → "
            f"{c['cost_of_a_miss_at_the_bound_nats']:.1f} nats |",
            "",
            "The bound sits outside anything the closing line or the outcomes support, so it "
            "removes only literal certainty; it never constrains a prediction the market "
            "would consider plausible.",
        ],
    )


#: How each ledgered run is described in the side-by-side table.
RUN_LABELS = {1: "leaky features", 2: "leak fixed", 3: "calibrator bounded"}


def render_runs(runs: list[tuple[dict, dict]]) -> str:
    """Every headline figure, one column per evaluation run.

    ``runs`` is ``[(metrics, parlay_metrics), ...]`` in run order.
    """
    ms = [m for m, _ in runs]
    n_fights = {m["results"]["eval"]["market"]["n"] for m in ms}
    head = " | ".join(
        f"Run {m['eval_run_count']} ({RUN_LABELS.get(m['eval_run_count'], 'current')})" for m in ms
    )
    lines = [
        f"All on the same {min(n_fights):,} evaluation fights"
        + ("" if len(n_fights) == 1 else " (fight counts differ between runs)")
        + ". Each column is generated from that run's own `metrics.json`; earlier runs are "
        "archived under `reports/archive/`.",
        "",
        f"| Figure | {head} |",
        "|---|" + "---:|" * len(ms),
    ]

    def row(label: str, cells: list[str]) -> None:
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    for name in ("market", "lightgbm", "elo", VARIANT_NAME):
        row(
            f"Eval log loss — {LABELS[name]}",
            [f"{m['results']['eval'][name]['log_loss']:.4f}" for m in ms],
        )
    for name in ("lightgbm", "elo", VARIANT_NAME):
        row(
            f"Eval Δ vs market — {LABELS[name]}",
            [_ci(m["paired_vs_market"]["eval"][name]) for m in ms],
        )
    row(
        "Dev log loss — LightGBM",
        [f"{m['results']['dev']['lightgbm']['log_loss']:.4f}" for m in ms],
    )
    row(
        f"Dev Δ vs market — {LABELS[VARIANT_NAME]}",
        [_ci(m["paired_vs_market"]["dev"][VARIANT_NAME]) for m in ms],
    )

    for source, label in (("lightgbm", "LightGBM"), ("market_plus_model", "market + model")):
        res = [p["by_source"][source] for _, p in runs]
        pv = [r["predicted_vs_realized"]["all_top_ranked"] for r in res]
        row(f"Parlays ranked by {label} — top-ranked", [f"{r['n_top_ranked']:,}" for r in res])
        row(
            f"Parlays ranked by {label} — placed in bankroll replay",
            [f"{r['bankroll']['n_bets']:,}" for r in res],
        )
        row(
            f"Parlays ranked by {label} — predicted EV, flat stakes",
            [f"{x['mean_predicted_ev']:+.1%}" for x in pv],
        )
        row(
            f"Parlays ranked by {label} — realised ROI, flat stakes",
            [f"{x['realized_roi']:+.1%}" for x in pv],
        )
        row(
            f"Parlays ranked by {label} — ending bankroll (from "
            f"£{res[0]['bankroll']['starting_bankroll']:,.0f})",
            [f"£{r['bankroll']['ending_bankroll']:,.2f}" for r in res],
        )
    return _wrap("RUNS", lines)


def render_post_hoc(ph: dict, metrics: dict, history: list[tuple[int, dict]] = ()) -> str:
    """``history`` holds ``(run, post_hoc)`` for archived earlier runs that have one."""
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

    lines += ["", "### Where LightGBM's excess log loss comes from", ""]
    audits = [(earlier_run, earlier["lightgbm_certainties"]) for earlier_run, earlier in history]
    audits.append((metrics["eval_run_count"], cert))
    lines += [
        "| Run | Window | Fights | Exactly 0 or 1 (within "
        f"{cert['tolerance']:g}) | Of those, wrong | At the calibration bound | Of those, wrong "
        "| LightGBM log loss |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_no, audit in audits:
        for w in ("dev", "eval"):
            c = audit[w]
            at, at_wrong = (
                (str(c["at_bound"]), str(c["at_bound_and_wrong"]))
                if "at_bound" in c
                else ("— (no bound)", "—")
            )
            lines.append(
                f"| {run_no} | {w.capitalize()} | {c['n']:,} | {c['certain']} | "
                f"{c['certain_and_wrong']} | {at} | {at_wrong} | {c['log_loss']:.4f} |"
            )
    cost = cert["cost_of_a_miss_nats"]
    market_ev = metrics["results"]["eval"]["market"]["log_loss"]
    dev, ev = cert["dev"], cert["eval"]
    text = (
        "Each walk-forward fold calibrates LightGBM with isotonic regression, a step "
        "function. When its most extreme step holds only wins or only losses it outputs "
        f"exactly 1 or 0, which scoring clips to {math.exp(-cost):g}; a miss then costs "
        f"{cost:.1f} nats, {cost / LN2:.0f} times what a coin flip pays."
    )
    if ev["certain"] == 0 and dev["certain"] == 0 and "bounds" in cert:
        lo, hi = cert["bounds"]
        text += (
            f" Run {metrics['eval_run_count']} bounds the calibrator to [{lo:g}, {hi:g}], so no "
            f"prediction claims certainty; the same extremes now sit at the bound, where a miss "
            f"costs {cert['cost_of_a_miss_at_the_bound_nats']:.1f} nats. LightGBM scores "
            f"{dev['log_loss']:.4f} in the dev window, "
            + (
                "still slightly worse than a coin flip"
                if dev["log_loss"] >= LN2
                else "better than a coin flip"
            )
            + f", and {ev['log_loss']:.4f} on eval, still far behind the market's {market_ev:.4f}."
        )
    else:
        text += (
            f" Without those predictions LightGBM scores {dev['log_loss_without_certainties']:.4f}"
            f" in the dev window and {ev['log_loss_without_certainties']:.4f} on eval, still far "
            f"behind the market's {market_ev:.4f}."
        )
    lines += ["", text]
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
    from ufc_predictor.backtest import readme_phase8

    return {
        **readme_phase8.render_all(cfg),
        "HEADLINE": render_headline(metrics),
        "DIAGNOSTICS": render_diagnostics(diagnostics),
        "TUNING": render_tuning(_load(r / "tuning.json")),
        "DEVCARDS": render_dev_cards(_load(r / "model_cards.json")),
        "POSTHOC": render_post_hoc(
            _load(r / "post_hoc.json"),
            metrics,
            [
                (_load(r / a / "metrics.json")["eval_run_count"], _load(r / a / "post_hoc.json"))
                for a in RUN_ARCHIVES
                if (r / a / "post_hoc.json").is_file()
            ],
        ),
        "LEAK": render_leak(diagnostics),
        "CALBOUND": render_calibration_bound(diagnostics),
        "RUNS": render_runs(
            [
                (_load(r / a / "metrics.json"), _load(r / a / "parlay_metrics.json"))
                for a in RUN_ARCHIVES
            ]
            + [(metrics, _load(r / "parlay_metrics.json"))]
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


if __name__ == "__main__":  # pragma: no cover
    inject_all()
