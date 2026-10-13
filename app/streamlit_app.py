"""ufc-predictor demo.

Research backtest only. Not betting advice.

Loads **precomputed artefacts only** from ``app/artifacts/`` -- a fitted model, fighter
snapshots, out-of-sample predictions and a summary of the metrics. Nothing is trained,
scraped or re-scored at startup, so the app opens instantly and can never disagree with
the committed results.

Charts are drawn with Streamlit's native charting so they follow the viewer's light or
dark theme; no pre-rendered images with baked-in backgrounds are embedded.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="ufc-predictor — model vs market",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Artefact loading (cached; never trains)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    return json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(ARTIFACTS / "predictions.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


@st.cache_data(show_spinner=False)
def load_fighters() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "fighters.parquet")


@st.cache_resource(show_spinner=False)
def load_model():
    import joblib

    return joblib.load(ARTIFACTS / "model.joblib")


def artifacts_present() -> bool:
    return all(
        (ARTIFACTS / f).is_file()
        for f in ("summary.json", "predictions.parquet", "fighters.parquet", "model.joblib")
    )


# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------


def apply_theme_metadata() -> None:
    """Set `color-scheme` and `theme-color`, which Streamlit does not emit itself.

    Without `color-scheme` the browser keeps native controls and scrollbars in light
    styling on a dark page; without `theme-color` mobile browser chrome does not match
    the page background. Both are derived from the rendered background colour, so they
    follow whichever theme the viewer is actually in rather than assuming one.

    Delivered through `components.html` rather than `st.markdown`: Streamlit strips
    `<script>` from markdown, while a component renders a same-origin iframe whose
    script can reach the parent document. The iframe is zero-height and invisible.
    """
    import streamlit.components.v1 as components

    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const apply = () => {
            const bg = getComputedStyle(doc.body).backgroundColor;
            const m = (bg.match(/\\d+/g) || [255, 255, 255]).map(Number);
            const dark = (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) < 128;
            doc.documentElement.style.colorScheme = dark ? 'dark' : 'light';
            let meta = doc.querySelector('meta[name="theme-color"]');
            if (!meta) {
              meta = doc.createElement('meta');
              meta.setAttribute('name', 'theme-color');
              doc.head.appendChild(meta);
            }
            meta.setAttribute('content', bg);
          };
          apply();
          // Re-apply if the viewer flips their system theme while the page is open.
          window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply);
        })();
        </script>
        """,
        height=0,
    )


def banner() -> None:
    """Persistent, unmissable framing shown on every page."""
    st.warning(
        "**Research backtest — not betting advice.** Nothing here is a recommendation "
        "to wager. The headline finding is that this model **loses** to the closing "
        "betting line.",
        icon="⚠️",
    )


def pct(x: float | None, places: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{places}%}"


def num(x: float | None, places: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{places}f}"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview(summary: dict) -> None:
    st.title("Does a UFC model beat the betting market?")
    banner()

    ev = summary["results"]["eval"]
    paired = summary["paired_vs_market"]["eval"]["lightgbm"]
    rec = summary["reconciliation"]

    st.subheader("No. And here is the measurement.")
    st.markdown(
        f"""
Over **{ev["market"]["n"]:,} held-out fights** across
**{summary["walk_forward"]["eval_events"]} events**
({summary["windows"]["eval"][0]} → {summary["windows"]["eval"][1]}), the vig-free closing
line forecasts fight outcomes better than this model, and the gap is larger than the
uncertainty around it.

The evaluation window was scored **{summary["eval_run_count"]} time**. Hyperparameters
were fixed on an earlier development window and frozen before the held-out period was
ever touched.
"""
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Market log loss",
        num(ev["market"]["log_loss"]),
        help="Lower is better. The vig-free closing line.",
    )
    c2.metric(
        "Model log loss",
        num(ev["lightgbm"]["log_loss"]),
        delta=f"{paired['observed']:+.4f} vs market",
        delta_color="inverse",
        help="Lower is better. Positive delta means the market wins.",
    )
    c3.metric(
        "95% CI on the gap",
        f"[{paired['ci_lower']:+.4f}, {paired['ci_hi']:+.4f}]",
        help="Paired bootstrap, resampled by event. Excludes zero, so the gap is real.",
    )

    st.markdown(
        f"""
The interval **excludes zero**, so this is not a close call that more data might
overturn — on this evidence the model is genuinely worse than the market. The
confidence interval comes from resampling **whole events** rather than individual
fights, because bouts on one card share conditions and correlated errors; treating
{ev["market"]["n"]:,} fights as {ev["market"]["n"]:,} independent observations would
make the interval look narrower than it is.
"""
    )

    st.subheader("Why that is the interesting answer")
    vd = summary["post_hoc"]["variant_decomposition"]
    tot, recal, add = (
        vd["variant_minus_market"],
        vd["recalibrated_minus_market"],
        vd["variant_minus_recalibrated"],
    )
    if add["ci_lower"] <= 0 <= add["ci_hi"]:
        features_finding = (
            "- **The features add nothing the market has not already priced.** Feeding the "
            "market's and the model's probabilities into one regression improves on the "
            f"market by {abs(tot['observed']):.4f} nats per fight, but a regression on the "
            f"market alone gets {abs(recal['observed']):.4f} of that: the closing line "
            "recalibrating its own favourite–longshot bias. Adding the model to that changes "
            f"log loss by {add['observed']:+.5f} (95% CI {add['ci_lower']:+.4f} to "
            f"{add['ci_hi']:+.4f}), which cannot be told from zero."
        )
    else:
        features_finding = (
            "- **The features add a little beyond the recalibrated market**: "
            f"{add['observed']:+.5f} nats per fight (95% CI {add['ci_lower']:+.4f} to "
            f"{add['ci_hi']:+.4f})."
        )
    st.markdown(
        f"""
A closing line aggregates the opinions of people with money at risk, so beating it is
a high bar — much higher than beating a coin flip, which any reasonable model clears.
Reporting a model that loses to it, with an interval that proves the loss, is a more
useful result than a tuned number that would not survive contact with a real market.

Two findings sharpen it:

{features_finding}
- **Ranking parlays by the model's own edge is actively harmful.** It selects the
  fights the model is most wrong about. See the Parlay Lab.
"""
    )

    with st.expander("How the fights are accounted for"):
        st.markdown(
            f"""
Every priced, decisive fight is in exactly one bucket:
**{rec["dev"]:,} development + {rec["gap"]} gap + {rec["eval"]:,} evaluation
= {rec["scored_total"]:,}**, with {rec["unscored_warmup"]} unscored during walk-forward
warm-up.

{rec["total_priced_decisive"]:,} fights carry a closing line and a decisive result.
Model and market are scored on **identical fights**, so the comparison is like-for-like.
"""
        )


def page_matchup(summary: dict) -> None:
    st.title("Matchup")
    banner()

    fighters = load_fighters()
    preds = load_predictions()

    names = sorted(fighters["display_name"].dropna().unique())
    st.caption(
        f"{len(names):,} fighters. Pick two to see what the model says — and, when they "
        "actually met, what the market said."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        name_a = st.selectbox("Fighter A", names, index=names.index(names[0]), key="fa")
    with col_b:
        default_b = 1 if len(names) > 1 else 0
        name_b = st.selectbox("Fighter B", names, index=default_b, key="fb")

    if name_a == name_b:
        st.info("Pick two different fighters.")
        return

    row_a = fighters.loc[fighters["display_name"] == name_a].iloc[0]
    row_b = fighters.loc[fighters["display_name"] == name_b].iloc[0]

    # Did they actually fight? Match on the normalised names stored with predictions.
    pair = preds.loc[
        (
            (preds["name_1"].str.lower() == name_a.lower())
            & (preds["name_2"].str.lower() == name_b.lower())
        )
        | (
            (preds["name_1"].str.lower() == name_b.lower())
            & (preds["name_2"].str.lower() == name_a.lower())
        )
    ]

    if not pair.empty:
        st.subheader("They have met — these are the real out-of-sample predictions")
        for bout in pair.sort_values("date", ascending=False).itertuples(index=False):
            first, second = bout.name_1, bout.name_2
            won = first if bout.label == 1.0 else second
            st.markdown(f"**{bout.event_name}** · {bout.date.date()}")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Model: {first} wins", pct(bout.p_lightgbm))
            m2.metric(f"Market: {first} wins", pct(bout.p_market))
            m3.metric("Result", f"{won} won")
            st.divider()
        return

    st.subheader("Hypothetical matchup")
    st.info(
        "**These two have never fought, so there is no market price for this bout.** "
        "The model's number below is a forecast with nothing to compare it against — "
        "no closing line exists, and none is invented.",
        icon="🔎",
    )

    try:
        model = load_model()
        row = build_matchup_row(row_a, row_b)
        p1 = float(model.predict_proba(row)[0])
    except Exception as exc:  # pragma: no cover - surfaced in the UI
        st.error(f"Could not score this matchup: {exc}")
        return

    # fighter_1 is the canonically-first id, which may be either selection.
    a_is_first = str(row_a["fighter_id"]) < str(row_b["fighter_id"])
    p_a = p1 if a_is_first else 1.0 - p1

    c1, c2 = st.columns(2)
    c1.metric(f"{name_a} wins", pct(p_a))
    c2.metric(f"{name_b} wins", pct(1.0 - p_a))

    st.caption(
        "Two caveats worth reading. The model is **calibrated with isotonic "
        "regression**, which is a step function — it emits only a few dozen distinct "
        "probabilities, so different matchups can show identical numbers. And each "
        "fighter's statistics are taken as of their most recent recorded bout, so a "
        "long-inactive fighter is described by a stale snapshot."
    )

    with st.expander("Fighter snapshots used"):
        cols = ["display_name", "last_fight_date", "elo", "n_prior_fights", "win_rate"]
        snap = pd.DataFrame([row_a, row_b])
        st.dataframe(
            snap.loc[:, [c for c in cols if c in snap.columns]],
            hide_index=True,
            use_container_width=True,
        )


def build_matchup_row(row_a: pd.Series, row_b: pd.Series) -> pd.DataFrame:
    """Assemble the model's feature row for a hypothetical pairing.

    Mirrors ``ufc_predictor.export_app.matchup_row`` but without importing the training
    package, so the deployed app keeps a small dependency set.
    """
    model = load_model()
    feature_cols = list(model.feature_cols)
    today = dt.date.today()

    a, b = (row_a, row_b) if str(row_a["fighter_id"]) < str(row_b["fighter_id"]) else (row_b, row_a)
    stems = sorted({c[:-2] for c in feature_cols if c.endswith(("_1", "_2"))})

    row: dict[str, float] = {}
    for slot, snap in (("1", a), ("2", b)):
        for stem in stems:
            row[f"{stem}_{slot}"] = snap.get(stem, np.nan)
        last = snap.get("last_fight_date")
        if last is not None and not pd.isna(last):
            row[f"days_since_last_{slot}"] = float((today - pd.Timestamp(last).date()).days)

    for stem in stems:
        x, y = row.get(f"{stem}_1"), row.get(f"{stem}_2")
        row[f"{stem}_diff"] = (
            float(x) - float(y)
            if x is not None and y is not None and not pd.isna(x) and not pd.isna(y)
            else np.nan
        )

    row["title_bout"] = False
    row["scheduled_rounds"] = 3.0

    frame = pd.DataFrame([row])
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame.loc[:, feature_cols]


def page_backtest(summary: dict) -> None:
    st.title("Backtest")
    banner()

    ev = summary["results"]["eval"]
    paired = summary["paired_vs_market"]["eval"]

    st.subheader("Held-out evaluation window")
    labels = {
        "market": "Market (closing line)",
        "lightgbm": "LightGBM",
        "elo": "Elo",
        "market_plus_model": "Market + model (variant)",
    }
    table = pd.DataFrame(
        [
            {
                "Model": labels.get(k, k),
                "Fights": s["n"],
                "Log loss": s["log_loss"],
                "Brier": s["brier"],
                "AUC": s["auc"],
                "Accuracy": s["accuracy"],
                "ECE": s["ece"],
            }
            for k, s in ev.items()
        ]
    ).sort_values("Log loss")
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Fights": st.column_config.NumberColumn(format="%d"),
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
            "Brier": st.column_config.NumberColumn(format="%.4f"),
            "AUC": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "ECE": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    audit = summary["post_hoc"]["lightgbm_certainties"]
    cert = audit["eval"]
    if cert["certain"] == 0 and "bounds" in audit:
        lo, hi = audit["bounds"]
        lgbm_note = (
            f"LightGBM's calibrator is bounded to [{lo:g}, {hi:g}] since evaluation run 3, so "
            f"no prediction claims certainty; {cert['at_bound']} eval predictions sit at the "
            f"bound and {cert['at_bound_and_wrong']} of them missed."
        )
    else:
        lgbm_note = (
            f"LightGBM's calibrator emits literal certainties — {cert['certain']} predictions "
            f"of exactly 0 or 1 on the evaluation window, {cert['certain_and_wrong']} of them "
            f"wrong — and without them its log loss would be "
            f"{cert['log_loss_without_certainties']:.4f}, still far behind the market."
        )
    st.caption(
        "Lower is better for log loss, Brier and ECE; higher is better for AUC and "
        "accuracy. ECE averages miscalibration over probability bins, so it can look good "
        "while a few extreme predictions are badly wrong. " + lgbm_note
    )

    st.subheader("Difference from the market")
    diff = pd.DataFrame(
        [
            {
                "Model": labels.get(k, k),
                "Δ log loss": p["observed"],
                "CI low": p["ci_lower"],
                "CI high": p["ci_hi"],
                "Verdict": (
                    "no difference shown"
                    if p["ci_lower"] <= 0 <= p["ci_hi"]
                    else (
                        "market wins"
                        if not p["beats_market"]
                        # The variant takes the closing line as an input; its lower log
                        # loss is the line recalibrating itself, not an edge over it.
                        else (
                            "lower than market (the line recalibrating itself)"
                            if k == "market_plus_model"
                            else "beats market"
                        )
                    )
                ),
            }
            for k, p in paired.items()
        ]
    )
    st.dataframe(
        diff,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Δ log loss": st.column_config.NumberColumn(format="%.4f"),
            "CI low": st.column_config.NumberColumn(format="%.4f"),
            "CI high": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "Negative means the model is better. Paired per fight, with a 95% interval from "
        f"{summary['bootstrap']['n_resamples']:,} bootstrap resamples of **events**."
    )

    st.subheader("Calibration")
    preds = load_predictions()
    evw = preds.loc[preds["date"] >= pd.Timestamp(summary["windows"]["eval"][0])]
    st.altair_chart(calibration_chart(evw), use_container_width=True)
    st.caption(
        "A perfectly calibrated forecaster sits on the diagonal. The model tracks it "
        "closely in the middle but flattens at both ends, which is the shape of a "
        "forecaster that will not commit."
    )

    st.subheader("The gap, year by year")
    rows = [
        {
            "Year": r["year"],
            "Fights": r["n"],
            "Model − market": (r["lightgbm_log_loss"] - r["market_log_loss"])
            if r["lightgbm_log_loss"] is not None and r["market_log_loss"] is not None
            else None,
        }
        for r in summary["by_year"]
    ]
    gap = pd.DataFrame(rows).dropna()
    st.bar_chart(gap.set_index("Year")["Model − market"], height=260)
    st.caption(
        "Positive bars are years the market forecast better. It wins in essentially "
        "every year, which rules out the gap being an artefact of one unusual period."
    )


def calibration_chart(frame: pd.DataFrame):
    """Reliability curve, drawn natively so it follows the viewer's theme."""
    import altair as alt

    records = []
    for col, label in (("p_market", "Market"), ("p_lightgbm", "LightGBM")):
        sub = frame.loc[frame[col].notna()]
        if sub.empty:
            continue
        bins = np.clip(np.digitize(sub[col], np.linspace(0.1, 0.9, 9)), 0, 9)
        for b in range(10):
            mask = bins == b
            if mask.sum() < 10:
                continue
            records.append(
                {
                    "predicted": float(sub.loc[mask, col].mean()),
                    "observed": float(sub.loc[mask, "label"].mean()),
                    "model": label,
                    "n": int(mask.sum()),
                }
            )
    data = pd.DataFrame(records)
    ideal = pd.DataFrame({"predicted": [0.0, 1.0], "observed": [0.0, 1.0]})

    base = alt.Chart(data).encode(
        x=alt.X("predicted:Q", title="predicted probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed:Q", title="observed win rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", title=None),
        tooltip=[
            "model",
            alt.Tooltip("predicted:Q", format=".3f"),
            alt.Tooltip("observed:Q", format=".3f"),
            "n",
        ],
    )
    line = (
        alt.Chart(ideal)
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(x="predicted:Q", y="observed:Q")
    )
    return (line + base.mark_line(point=True)).properties(height=360)


def page_parlay(summary: dict) -> None:
    st.title("Parlay Lab")
    banner()

    st.error(
        "**This page is a demonstration of selection bias, not a list of bets.** It "
        "exists to show that ranking parlays by a model's own estimated edge destroys "
        "money. Nothing here is a recommendation.",
        icon="🚫",
    )

    parlay = summary.get("parlay", {})
    by_source = parlay.get("by_source", {})
    if not by_source:
        st.info("No parlay artefacts found. Run `make parlay` to generate them.")
        return

    rows = []
    for source, res in by_source.items():
        pv = res.get("predicted_vs_realized", {}).get("all_top_ranked", {})
        if not pv.get("n"):
            rows.append(
                {
                    "Ranked by": source,
                    "Top-ranked parlays": 0,
                    "Predicted EV": None,
                    "Realised ROI": None,
                    "Hit rate": None,
                    "Predicted hit rate": None,
                }
            )
            continue
        rows.append(
            {
                "Ranked by": source.replace("_", " "),
                "Top-ranked parlays": pv["n"],
                "Predicted EV": pv["mean_predicted_ev"],
                "Realised ROI": pv["realized_roi"],
                "Hit rate": pv["hit_rate"],
                "Predicted hit rate": pv["mean_predicted_prob"],
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Top-ranked parlays": st.column_config.NumberColumn(format="%d"),
            "Predicted EV": st.column_config.NumberColumn(format="%+.4f"),
            "Realised ROI": st.column_config.NumberColumn(format="%+.4f"),
            "Hit rate": st.column_config.NumberColumn(format="%.4f"),
            "Predicted hit rate": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    lg = by_source.get("lightgbm", {})
    pv = lg.get("predicted_vs_realized", {}).get("all_top_ranked", {})
    if pv.get("n"):
        st.subheader("What went wrong")
        st.markdown(
            f"""
Ranking by the model's own expected value picked parlays it priced at
**{pv["mean_predicted_ev"]:+.0%} expected return**. They returned
**{pv["realized_roi"]:+.1%}** per unit staked. The model expected them to land
**{pv["mean_predicted_prob"]:.1%}** of the time; they landed **{pv["hit_rate"]:.1%}**
of the time.

Sorting by estimated edge does not find the best bets. It finds the fights the model is
**most wrong about**, because the largest apparent edges are produced by the largest
overestimates. Combining three such legs multiplies three errors together, which is why
longer parlays were worst of all.
"""
        )
        # Deliberately NOT a shared-axis bar chart of EV against ROI: at +5.62 versus
        # -0.22 the two differ by more than twenty times, so one bar swamps the other
        # and the comparison becomes unreadable. The numbers are shown directly, and
        # the chart is reserved for hit rate, where both quantities share a scale.
        m1, m2 = st.columns(2)
        m1.metric("Predicted expected return", f"{pv['mean_predicted_ev']:+.0%}")
        m2.metric(
            "Actual return",
            f"{pv['realized_roi']:+.1%}",
            delta=f"{pv['realized_roi'] - pv['mean_predicted_ev']:+.0%} vs predicted",
            delta_color="inverse",
        )
        h1, h2 = st.columns(2)
        h1.metric("Predicted share that would land", f"{pv['mean_predicted_prob']:.1%}")
        h2.metric(
            "Share that actually landed",
            f"{pv['hit_rate']:.1%}",
            delta=f"{pv['hit_rate'] - pv['mean_predicted_prob']:+.1%}",
            delta_color="inverse",
        )

        bk = lg.get("bankroll", {})
        bs = lg.get("bankroll_bootstrap", {})
        if bk:
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Ending bankroll",
                f"{bk['ending_bankroll']:,.2f}",
                delta=f"{bk['total_return']:+.1%}",
                delta_color="normal",
            )
            c2.metric("Max drawdown", pct(bk["max_drawdown"]))
            c3.metric("Probability of ruin", pct(bs.get("ruin_probability")))
            st.caption(
                f"Starting from {bk['starting_bankroll']:,.0f} at fractional Kelly with a "
                f"per-bet cap. Across {bs.get('n_resamples', 0):,} event-resampled "
                f"histories, {pct(bs.get('share_losing_money'))} lose money."
            )

    if not by_source.get("market", {}).get("n_candidates"):
        st.success(
            "**The control behaves correctly.** Ranking by the market's own de-vigged "
            "probabilities finds **zero** positive-expected-value parlays — a de-vigged "
            "probability priced against the book's vig-inclusive payout cannot show an "
            "edge. If this had found value, the pricing code would be wrong.",
            icon="✅",
        )

    st.caption(parlay.get("legs_priced", ""))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Matchup": page_matchup,
    "Backtest": page_backtest,
    "Parlay Lab": page_parlay,
}


def main() -> None:
    if not artifacts_present():
        st.title("ufc-predictor")
        st.error(
            "Demo artefacts are missing. Generate them with:\n\n"
            "```\nmake data && make backtest && make parlay && "
            "uv run ufc-predictor export-app\n```",
            icon="📦",
        )
        return

    summary = load_summary()
    apply_theme_metadata()

    with st.sidebar:
        # Deliberately not st.header: that emits an <h2> ahead of the page's <h1>,
        # which breaks heading order for screen readers.
        st.markdown("**ufc-predictor**")
        st.caption("Model vs market, measured honestly.")
        choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption(
            f"Seed {summary['seed']} · evaluation scored "
            f"{summary['eval_run_count']}×\n\n"
            f"Artefacts generated {summary['generated_at_utc']}Z"
        )
        st.caption("Research backtest only. Not betting advice.")

    PAGES[choice](summary)


main()
