"""Generated README sections for Phase 8 (development-window improvement, CLV, forward test).

Same contract as ``readme_sections``: every figure comes from a JSON file in ``reports/``
or the committed BestFightOdds snapshot, and ``tests/test_readme_sections.py`` fails if
the README drifts from a fresh rendering.

* ``P8_STAGES``  -- every pre-registered stage and candidate, adopted or not
  (``reports/dev/stages.json``);
* ``P8_CONFIRM`` -- the once-only 2015 confirmation (``reports/dev/confirmation.json``);
* ``P8_CLV``     -- closing-line value against BestFightOdds lines (``reports/clv.json``,
  ``reports/bfo_join.json``, the snapshot's ``audit.json``);
* ``P8_FORWARD`` -- the forward test so far (``reports/forward/summary.json``,
  ``reports/forward/ledger.json``).

A section whose source does not exist yet renders a one-line "not yet run" statement
rather than a number.
"""

from __future__ import annotations

import json
from pathlib import Path

from ufc_predictor.config import REPO_ROOT, Config, get_config

MINUS = "\u2212"  # escaped so the linter does not mistake it for a hyphen

FEATURE_LABELS = {
    "base": "run-3 base (117)",
    "adjusted": "opponent-adjusted",
    "recency": "recency-weighted",
    "style": "style matchups",
    "decline": "decline signals",
    "glicko": "Glicko-2",
}


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}:START -->", f"<!-- {name}:END -->"


def _wrap(name: str, lines: list[str]) -> str:
    start, end = _markers(name)
    return "\n".join([start, *lines, end])


def _s(x: float, digits: int = 4) -> str:
    return f"{x:+.{digits}f}".replace("-", MINUS)


def _ci(d: dict, digits: int = 4) -> str:
    return f"{_s(d['observed'], digits)} [{_s(d['ci_lower'], digits)}, {_s(d['ci_hi'], digits)}]"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def describe_spec(spec: dict) -> str:
    feats = " + ".join(FEATURE_LABELS.get(f, f) for f in spec["features"])
    parts = [feats]
    if spec.get("top_k"):
        parts.append(f"top {spec['top_k']} per fold")
    parts.append(
        {
            "all_history": "all history",
            "from_2001": "trained from 2001",
            "last_8_years": "last 8 years",
        }[spec["train_window"]]
    )
    if spec.get("symmetric"):
        parts.append("orientation-symmetric")
    parts.append(spec["model"] if spec["model"] != "lightgbm" else "LightGBM")
    if spec["model"] == "lightgbm":
        parts.append(
            {"isotonic_bounded": "bounded isotonic", "platt": "Platt", "none": "uncalibrated"}[
                spec["calibration"]
            ]
        )
    return ", ".join(parts)


def _candidate_label(stage: dict, cand: dict) -> str:
    name = cand["name"]
    if stage["field"] == "features":
        return " + ".join(
            FEATURE_LABELS.get(p, p)
            for p in (name.split("+") if name != "all" else ["all families"])
        )
    return name.replace("_", " ")


# ---------------------------------------------------------------------------


def render_stages(st: dict | None) -> str:
    if st is None:
        return _wrap(
            "P8_STAGES",
            ["*Not yet run: `uv run ufc-predictor experiments` writes `reports/dev/stages.json`.*"],
        )
    s0 = st["stages"][0]
    sel = s0["selected"]
    lines = [
        f"Selection window {st['selection_window'][0]} → {st['selection_window'][1]}, walk-forward, "
        f"scored on {st['stages'][1]['incumbent']['n_fights']:,} priced fights "
        f"({st['stages'][1]['incumbent']['n_events']} events). Pre-registration commit "
        f"`{st['preregistration_commit'][:7]}`; run from `{st['git_commit'][:7]}`. Rule: adopt a "
        f"candidate only if its paired log-loss difference is negative and its Holm-adjusted "
        f"one-sided p-value is below {st['alpha']:.2f}.",
        "",
        f"**Stage 0 (re-tune inside 2010-2014).** {s0['n_candidates']} candidates on a "
        f"{s0['validation_rows']:,}-fight holdout ({s0['validation_span'][0]} → {s0['validation_span'][1]}); "
        f"{s0['n_tied']} tied within 0.001 of the best ({s0['lowest_log_loss']:.4f}); the most "
        f"regularized of them was kept: depth {sel['max_depth']}, {sel['num_leaves']} leaves, "
        f"min child {sel['min_child_samples']}, learning rate {sel['learning_rate']}, column "
        f"sample {sel['colsample_bytree']}, λ {sel['reg_lambda']}.",
        "",
        "| Stage | Candidate | Log loss | Δ vs incumbent [95% CI] | p (one-sided) | Holm p | Adopted |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for stage in st["stages"][1:]:
        if stage.get("kind") == "final_retune" and "skipped" in stage:
            lines.append(f"| final re-tune | skipped ({stage['skipped']}) | | | | | |")
            continue
        label = stage["name"].replace("stage_", "").replace("_", " ")
        inc = stage["incumbent"]
        lines.append(f"| {label} | *incumbent* | {inc['log_loss']:.4f} | | | | |")
        for cand in stage["candidates"]:
            lines.append(
                f"| {label} | {_candidate_label(stage, cand) if 'field' in stage else cand['name']} | "
                f"{cand['log_loss']:.4f} | {_ci(cand)} | {_p(cand['p_one_sided'])} | "
                f"{_p(cand['p_holm'])} | {'**yes**' if cand['adopted'] else 'no'} |"
            )
    adopted = [s for s in st["stages"][1:] if s.get("adopted")]
    lines += [
        "",
        (
            "**Adopted:** "
            + "; ".join(f"{s['name'].replace('_', ' ')} → {s['adopted']}" for s in adopted)
            + "."
            if adopted
            else "**Nothing was adopted:** no candidate cleared the pre-registered rule, so the "
            "frozen configuration is the re-tuned run-3 baseline."
        ),
        f"**Frozen configuration:** {describe_spec(st['frozen_spec'])}. Selection-window log "
        f"loss {st['frozen_selection_log_loss']:.4f} against the vig-free closing line's "
        f"{st['frozen_selection_market_log_loss']:.4f} on the same fights.",
    ]
    return _wrap("P8_STAGES", lines)


def render_confirmation(c: dict | None) -> str:
    if c is None:
        return _wrap(
            "P8_CONFIRM", ["*Not yet run: `uv run ufc-predictor confirm` scores 2015 once.*"]
        )
    ll = c["log_loss"]
    fb, fm, bm = c["frozen_minus_baseline"], c["frozen_minus_market"], c["baseline_minus_market"]
    return _wrap(
        "P8_CONFIRM",
        [
            f"Scored once, from commit `{c['git_commit'][:7]}`, on {c['n_fights']:,} priced 2015 "
            f"fights ({c['n_events']} events). Log loss: frozen {ll['frozen']:.4f}, stage-0 "
            f"baseline {ll['baseline']:.4f}, vig-free closing line {ll['market']:.4f}.",
            "",
            "| Comparison (2015) | Δ log loss [95% CI] | p (one-sided) |",
            "|---|---:|---:|",
            f"| frozen - baseline | {_ci(fb)} | {_p(fb['p_one_sided'])} |",
            f"| frozen - market | {_ci(fm)} | {_p(fm['p_one_sided'])} |",
            f"| baseline - market | {_ci(bm)} | {_p(bm['p_one_sided'])} |",
            "",
            f"The interval on frozen - baseline is {fb['ci_hi'] - fb['ci_lower']:.4f} wide: wide "
            "enough to catch a large overfit to 2010-2014, too wide to certify a small gain.",
        ],
    )


def render_clv(clv: dict | None, join: dict | None, audit: dict | None) -> str:
    if clv is None:
        return _wrap(
            "P8_CLV", ["*Not yet run: `uv run ufc-predictor clv` writes `reports/clv.json`.*"]
        )
    corr = clv["correlation_edge_vs_move"]
    dmv = clv["move_toward_model_when_edge_over_5pp"]
    coef = clv["model_information_beyond_open"]["coef_logit_model"]
    sanity = clv["close_minus_open_log_loss"]
    lines = []
    if audit:
        lines.append(
            f"BestFightOdds snapshot: {audit['bouts']:,} bouts at {audit['ufc_events']} UFC events "
            f"({audit['date_min']} → {audit['date_max']}), every one seen on both fighters' pages "
            f"with identical prices ({audit['bouts_confirmed_by_both_pages']:,})."
            + (
                f" Joined exactly (same date, same pair) to {join['joined']:,} fights; "
                f"{join['dev_priced_join_rate']:.1%} of priced development-window fights."
                if join
                else ""
            )
        )
        lines.append("")
    lines += [
        f"On {clv['n_fights']:,} fights ({clv['n_events']} events, {clv['date_span'][0]} → "
        f"{clv['date_span'][1]}): open {clv['open_log_loss']:.4f}, close {clv['close_log_loss']:.4f}, "
        f"model {clv['model_log_loss']:.4f} log loss.",
        "",
        "| Question | Estimate [95% CI] |",
        "|---|---:|",
        f"| Does the close beat the open? (close - open log loss) | {_ci(sanity)} |",
        f"| Correlation of the model's edge at the open with the line's move | {_s(corr['point'], 3)} [{_s(corr['ci_lower'], 3)}, {_s(corr['ci_hi'], 3)}] |",
        f"| Mean move toward the model when it differs from the open by > 5 points ({dmv['n_fights']:,} fights) | {_s(dmv['point'], 4)} [{_s(dmv['ci_lower'], 4)}, {_s(dmv['ci_hi'], 4)}] |",
        f"| logit(p_model) coefficient for the move's direction, given logit(open) | {_s(coef['point'], 3)} [{_s(coef['ci_lower'], 3)}, {_s(coef['ci_hi'], 3)}] |",
    ]
    return _wrap("P8_CLV", lines)


def render_forward(summary: dict | None, ledger: list | None) -> str:
    if not ledger:
        return _wrap(
            "P8_FORWARD",
            [
                "*No card predicted yet. The schedule predicts each card 24-72 hours before it starts.*"
            ],
        )
    rows = [
        "| Card | Predicted | Settled | Scored bouts | Model log loss | Kalshi close log loss | Commitment verified |",
        "|---|---|---|---:|---:|---:|:---:|",
    ]
    by_event: dict[str, dict] = {}
    for e in ledger:
        by_event.setdefault(e["event_id"], {})[e["action"]] = e
    settled = {e["event_id"]: e for e in (summary or {}).get("events", [])}
    for event_id, acts in by_event.items():
        name = acts.get("predict", {}).get("event_name", event_id)
        s = settled.get(event_id)
        pred = (
            acts["predict"]["at_utc"][:16].replace("T", " ") + " UTC"
            if "predict" in acts
            else ("missed" if "missed" in acts else "")
        )
        if s:
            rows.append(
                f"| {name} | {pred} | yes | {s['n_scored']} | "
                f"{s['model_log_loss']:.4f} | "
                + (
                    f"{s['kalshi_close_log_loss']:.4f}"
                    if s.get("kalshi_close_log_loss") is not None
                    else "—"
                )
                + f" | {'yes' if s['verified'] else '**no**'} |"
            )
        else:
            rows.append(f"| {name} | {pred} | no | | | | |")
    pooled = (summary or {}).get("pooled_verified")
    tail = []
    if pooled and pooled.get("n_scored"):
        tail = [
            "",
            f"Pooled over verified cards: {pooled['n_scored']} scored bouts; model "
            f"{pooled['model_log_loss']:.4f}"
            + (
                f", Kalshi close {pooled['kalshi_close_log_loss']:.4f} on {pooled['n_with_kalshi_close']} bouts "
                f"(model - Kalshi {_s(pooled['model_minus_kalshi_close'])})."
                if pooled.get("kalshi_close_log_loss") is not None
                else "."
            ),
        ]
    return _wrap("P8_FORWARD", rows + tail)


def _maybe(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def render_all(cfg: Config | None = None) -> dict[str, str]:
    cfg = cfg or get_config()
    r = cfg.paths.reports_dir
    audit = _maybe(REPO_ROOT / cfg.bfo["snapshot_dir"] / "audit.json")
    return {
        "P8_STAGES": render_stages(_maybe(r / "dev" / "stages.json")),
        "P8_CONFIRM": render_confirmation(_maybe(r / "dev" / "confirmation.json")),
        "P8_CLV": render_clv(_maybe(r / "clv.json"), _maybe(r / "bfo_join.json"), audit),
        "P8_FORWARD": render_forward(
            _maybe(REPO_ROOT / cfg.forward["reports_dir"] / "summary.json"),
            _maybe(REPO_ROOT / cfg.forward["reports_dir"] / "ledger.json"),
        ),
    }
