"""Streamlit demo tests, driven by Streamlit's own AppTest harness.

The claims worth pinning: the app never trains at startup, every page carries the
not-betting-advice framing, the Overview leads with the negative result rather than
burying it, a hypothetical matchup is explicitly marked as having no market price, and
the Parlay Lab is framed as a demonstration of selection bias rather than a tip sheet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ufc_predictor.config import get_config

APP = str(Path(__file__).parent.parent / "app" / "streamlit_app.py")
ARTIFACTS = Path(__file__).parent.parent / "app" / "artifacts"

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

TIMEOUT = 120


def _artifacts_ready() -> bool:
    return all(
        (ARTIFACTS / f).is_file()
        for f in ("summary.json", "predictions.parquet", "fighters.parquet", "model.joblib")
    )


requires_artifacts = pytest.mark.skipif(
    not _artifacts_ready(), reason="app artefacts absent; run `uv run ufc-predictor export-app`"
)


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    if page != "Overview":
        at.radio[0].set_value(page).run()
    return at


def all_text(at: AppTest) -> str:
    parts: list[str] = []
    for coll in (
        at.markdown,
        at.title,
        at.header,
        at.subheader,
        at.caption,
        at.warning,
        at.error,
        at.info,
        at.success,
    ):
        parts.extend(str(e.value) for e in coll)
    return "\n".join(parts).lower()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@requires_artifacts
def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, f"app raised: {[str(e.value) for e in at.exception]}"


@requires_artifacts
def test_all_four_pages_render_without_exception():
    for page in ("Overview", "Matchup", "Backtest", "Parlay Lab"):
        at = run_page(page)
        assert not at.exception, f"{page} raised: {[str(e.value) for e in at.exception]}"


def _imported_modules(path: str) -> set[str]:
    """Top-level module names the file actually imports, via AST.

    Parsed rather than grepped so that prose in a docstring or comment cannot pass or
    fail the check.
    """
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def test_app_imports_nothing_from_the_training_package():
    """The deployed app must stand alone on its minimal requirements.

    Importing the research package would drag in duckdb, matplotlib and the scrapers,
    none of which belong in a demo deployment.
    """
    assert "ufc_predictor" not in _imported_modules(APP), (
        "the app imports the training package; it must load artefacts only"
    )


def test_app_only_imports_packages_its_requirements_declare():
    declared = {
        line.split(">=")[0].split("==")[0].strip().replace("-", "_").lower()
        for line in (Path(APP).parent / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    declared |= {"scikit_learn", "sklearn"}
    stdlib = {"datetime", "json", "math", "pathlib", "__future__", "altair", "ast"}
    for mod in _imported_modules(APP):
        assert mod.lower() in declared | stdlib, (
            f"app imports {mod!r}, which app/requirements.txt does not declare"
        )


def test_app_never_trains_at_startup():
    """No fit call may appear anywhere in the app's code (comments excluded)."""
    import io
    import tokenize

    src = Path(APP).read_text(encoding="utf-8")
    code = "".join(
        tok.string if tok.type not in (tokenize.COMMENT, tokenize.STRING) else " "
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
    )
    for forbidden in (".fit(", "build_features", "run_walk_forward", "LGBMClassifier"):
        assert forbidden not in code, f"app contains {forbidden!r}; it must not train"


def test_requirements_are_minimal_and_exclude_research_only_packages():
    """Only the dependency lines count -- the explanatory comment names the excluded
    packages, so comments must be stripped before checking."""
    lines = [
        ln.strip().lower()
        for ln in (Path(APP).parent / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    req = "\n".join(lines)
    for heavy in ("duckdb", "matplotlib", "typer", "requests", "beautifulsoup4"):
        assert heavy not in req, f"{heavy} is not needed by the demo"
    for needed in ("streamlit", "pandas", "lightgbm", "scikit-learn"):
        assert needed in req
    assert len(lines) <= 10, f"{len(lines)} dependencies is not minimal"


def test_requirements_pin_exactly_what_the_lock_resolves():
    """The deployed app unpickles a model trained under uv.lock. A floating `>=` would
    let the deploy host install a newer scikit-learn or LightGBM than the pickle was
    written with, so every line must be an exact pin that agrees with the lock."""
    import tomllib

    lock = tomllib.loads((Path(APP).parent.parent / "uv.lock").read_text(encoding="utf-8"))
    locked = {pkg["name"]: pkg["version"] for pkg in lock["package"] if "version" in pkg}
    for ln in (Path(APP).parent / "requirements.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        assert "==" in ln, f"{ln!r} is not an exact pin"
        name, version = (part.strip() for part in ln.split("=="))
        assert locked.get(name) == version, (
            f"app pins {name}=={version} but uv.lock resolves {locked.get(name)}"
        )


# ---------------------------------------------------------------------------
# The framing, on every page
# ---------------------------------------------------------------------------


@requires_artifacts
@pytest.mark.parametrize("page", ["Overview", "Matchup", "Backtest", "Parlay Lab"])
def test_every_page_carries_the_not_betting_advice_banner(page):
    at = run_page(page)
    text = all_text(at)
    assert "not betting advice" in text, f"{page} is missing the banner"
    assert "wager" in text


@requires_artifacts
def test_overview_leads_with_the_negative_result():
    """The headline must be the finding, not a feature tour."""
    at = run_page("Overview")
    heading = (
        " ".join(str(e.value) for e in at.title)
        + " "
        + " ".join(str(e.value) for e in at.subheader)
    )
    assert "no." in heading.lower() or "beat the betting market" in heading.lower()

    text = all_text(at)
    assert "loses" in text or "better than this model" in text
    assert "excludes zero" in text


@requires_artifacts
def test_overview_states_the_eval_run_count():
    at = run_page("Overview")
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    assert f"{summary['eval_run_count']} time" in all_text(at)


@requires_artifacts
def test_overview_metrics_match_the_artifact():
    at = run_page("Overview")
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    shown = {m.value for m in at.metric}
    assert f"{summary['results']['eval']['market']['log_loss']:.4f}" in shown
    assert f"{summary['results']['eval']['lightgbm']['log_loss']:.4f}" in shown


# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------


@requires_artifacts
def test_matchup_offers_fighters_and_scores_a_pairing():
    at = run_page("Matchup")
    assert len(at.selectbox) >= 2, "expected a selector for each fighter"
    assert len(at.selectbox[0].options) > 100
    assert not at.exception


@requires_artifacts
def test_hypothetical_matchup_is_marked_as_having_no_market_price():
    """A pairing that never happened must say so, and must not invent a line."""
    import pandas as pd

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.radio[0].set_value("Matchup").run()

    fighters = pd.read_parquet(ARTIFACTS / "fighters.parquet")
    preds = pd.read_parquet(ARTIFACTS / "predictions.parquet")
    met = {
        frozenset((str(a).lower(), str(b).lower()))
        for a, b in zip(preds["name_1"], preds["name_2"], strict=True)
    }
    names = sorted(fighters["display_name"].dropna().unique())

    chosen = None
    for a in names[:40]:
        for b in names[-40:]:
            if a != b and frozenset((a.lower(), b.lower())) not in met:
                chosen = (a, b)
                break
        if chosen:
            break
    assert chosen, "could not construct a hypothetical pairing"

    at.selectbox[0].set_value(chosen[0]).run()
    at.selectbox[1].set_value(chosen[1]).run()
    assert not at.exception, [str(e.value) for e in at.exception]

    text = all_text(at)
    assert "never fought" in text
    assert "no market price" in text
    assert "none is invented" in text


@requires_artifacts
def test_matchup_discloses_the_isotonic_quantisation():
    """Identical probabilities across matchups are a property of the calibrator and
    must be explained rather than left to look like a bug."""
    at = run_page("Matchup")
    assert "isotonic" in all_text(at)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@requires_artifacts
def test_backtest_shows_a_metrics_table_and_charts():
    at = run_page("Backtest")
    assert len(at.dataframe) >= 2, "expected the metrics and difference tables"
    text = all_text(at)
    assert "calibration" in text
    assert "year" in text


@requires_artifacts
def test_backtest_reports_the_market_winning():
    at = run_page("Backtest")
    frames = [df.value for df in at.dataframe]
    verdicts = []
    for f in frames:
        if "Verdict" in getattr(f, "columns", []):
            verdicts = list(f["Verdict"])
    assert any("market wins" in str(v) for v in verdicts), (
        "the backtest page must show that the market beats the model"
    )


# ---------------------------------------------------------------------------
# Parlay Lab
# ---------------------------------------------------------------------------


@requires_artifacts
def test_backtest_never_labels_the_market_consuming_variant_as_beating_the_market():
    at = run_page("Backtest")
    for df in at.dataframe:
        frame = df.value
        if "Verdict" in frame.columns:
            rows = frame.loc[frame["Model"].str.contains("variant", case=False)]
            assert not rows["Verdict"].str.contains("beats").any()


@requires_artifacts
def test_app_explains_the_variant_gain_with_the_post_hoc_control():
    """The app may not say the variant's gain is indistinguishable from zero when it is
    not; it must give the decomposition the README gives."""
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    assert "post_hoc" in summary
    text = all_text(run_page("Overview"))
    assert "recalibrat" in text
    assert "not statistically distinguishable from zero" not in text


@requires_artifacts
def test_backtest_does_not_claim_lightgbm_avoids_confident_predictions():
    text = all_text(run_page("Backtest"))
    assert "refuses to commit" not in text
    assert "certaint" in text


@requires_artifacts
def test_parlay_lab_is_framed_as_selection_bias_not_recommendations():
    at = run_page("Parlay Lab")
    text = all_text(at)
    assert "selection bias" in text
    assert "not a list of bets" in text or "not a recommendation" in text
    assert "recommendation" in text


@requires_artifacts
def test_parlay_lab_shows_predicted_ev_against_realised_roi():
    at = run_page("Parlay Lab")
    frames = [df.value for df in at.dataframe]
    assert frames, "expected a results table"
    cols = set()
    for f in frames:
        cols |= set(getattr(f, "columns", []))
    assert "Predicted EV" in cols
    assert "Realised ROI" in cols


@requires_artifacts
def test_parlay_lab_reports_the_market_control():
    at = run_page("Parlay Lab")
    text = all_text(at)
    assert "control" in text
    assert "zero" in text


# ---------------------------------------------------------------------------
# Artefact integrity
# ---------------------------------------------------------------------------


@requires_artifacts
def test_summary_artifact_matches_metrics_json():
    """The app must not be able to show numbers that disagree with the backtest."""
    cfg = get_config()
    metrics = json.loads((cfg.paths.reports_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["eval_run_count"] == metrics["eval_run_count"]
    assert summary["results"]["eval"]["market"]["log_loss"] == pytest.approx(
        metrics["results"]["eval"]["market"]["log_loss"]
    )
    assert summary["reconciliation"] == metrics["reconciliation"]


@requires_artifacts
def test_streamlit_config_does_not_hardcode_a_theme():
    """Leaving the theme unset is what makes dark mode follow the viewer's system."""
    cfg_path = Path(APP).parent.parent / ".streamlit" / "config.toml"
    if not cfg_path.is_file():
        pytest.skip("no streamlit config")
    import tomllib

    parsed = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert "base" not in parsed.get("theme", {}), (
        "a hardcoded theme base would stop dark mode following the viewer's system"
    )
