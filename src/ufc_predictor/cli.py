"""Typer entrypoint for ufc-predictor.

Command surface is declared in Phase 0 so the Makefile and packaging are wired end to
end from the first commit. Each command is implemented in its own phase; until then it
exits non-zero with a message naming the phase, rather than silently succeeding.
"""

from __future__ import annotations

import typer

from ufc_predictor import __version__
from ufc_predictor.config import get_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Reproducible UFC fight-outcome modelling, evaluated against the vig-free market. "
    "Research backtest only - not betting advice.",
)


def _not_yet(command: str, phase: str) -> None:
    """Fail loudly for a command whose phase has not landed yet."""
    typer.secho(
        f"`{command}` is not implemented yet (lands in {phase}).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Print the package version and the active seed."""
    cfg = get_config()
    typer.echo(f"ufc-predictor {__version__} (seed={cfg.seed})")


@app.command()
def config() -> None:
    """Show the resolved configuration, to confirm what a run will actually use."""
    cfg = get_config()
    typer.echo(f"seed:        {cfg.seed}")
    typer.echo(f"dev window:  {cfg.windows.dev_start} .. {cfg.windows.dev_end}")
    typer.echo(f"eval window: {cfg.windows.eval_start} .. {cfg.windows.eval_end}")
    typer.echo(
        f"snapshot:    {cfg.paths.ufcstats_snapshot.relative_to(cfg.paths.raw_dir.parent.parent)}"
    )
    typer.echo(
        f"odds:        {cfg.paths.odds_snapshot.relative_to(cfg.paths.raw_dir.parent.parent)}"
    )


@app.command()
def data() -> None:
    """Build normalized Parquet tables from the committed snapshot (no network)."""
    from ufc_predictor.ingest.pipeline import run

    run()


@app.command()
def features() -> None:
    """Build the point-in-time feature table."""
    from ufc_predictor.features.pipeline import run

    run()


@app.command()
def models() -> None:
    """Fit each model on the dev window and write reports/model_cards.json."""
    from ufc_predictor.models.cards import run

    run()


@app.command()
def diagnose() -> None:
    """Shuffle test, its control and the attribute-leak evidence, dev window only."""
    from ufc_predictor.features.diagnostics import run

    run()


@app.command()
def tune() -> None:
    """Search LightGBM hyperparameters inside the dev window only."""
    from ufc_predictor.models.tune import search

    search()


@app.command()
def backtest(
    reason: str = typer.Option(
        "reproduction", "--reason", help="Why the evaluation window is being scored (ledgered)."
    ),
) -> None:
    """Run the walk-forward backtest and write reports/metrics.json.

    This is the only command that scores the evaluation window. Each invocation appends
    to reports/eval_runs.json with the full commit and the reason, and it refuses to run
    from a working tree with uncommitted changes.
    """
    from ufc_predictor.backtest.report import run

    run(reason=reason)


@app.command()
def report() -> None:
    """Redraw charts and refresh the README from metrics.json (no re-scoring)."""
    from ufc_predictor.backtest.report import regenerate

    regenerate()


@app.command()
def parlay() -> None:
    """Price parlays on the stored out-of-sample predictions (no re-scoring).

    Research backtest only - not betting advice.
    """
    from ufc_predictor.sim.pipeline import run

    run()


@app.command()
def predict(
    event: str = typer.Option(..., "--event", help="Event name, e.g. 'UFC 332: Silva vs. Wang'."),
) -> None:
    """Write a timestamped prediction log for one event.

    Events come from the committed snapshot only; upcoming cards are never invented.
    A completed event is logged as mode=replay and flagged as not proving advance
    commitment. Research backtest only - not betting advice.
    """
    from ufc_predictor.predict import PredictError, predict_event

    try:
        predict_event(event)
    except PredictError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def settle(
    log: str = typer.Argument(..., help="Path to a prediction-log CSV."),
) -> None:
    """Fill in results for a prediction log and compute its log loss."""
    from ufc_predictor.predict import PredictError
    from ufc_predictor.predict import settle as settle_log

    try:
        settle_log(log)
    except PredictError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def bfo() -> None:
    """Crawl BestFightOdds (cached, 1 request per 5 s) and write the dated snapshot.

    Not on the graded path: the committed snapshot is the input everywhere else.
    """
    from ufc_predictor.ingest.bfo import write_snapshot

    write_snapshot()


@app.command()
def experiments() -> None:
    """Run the pre-registered development stages (2010-2014 only) -> reports/dev/.

    Refuses to run from a commit that lacks or has changed config/experiments.yaml, or
    from a dirty tree. Never scores 2015 or the closed evaluation window.
    """
    from ufc_predictor.experiments.stages import run_all

    run_all()


@app.command()
def clv() -> None:
    """Closing-line value on the development window (BestFightOdds open vs close)."""
    from ufc_predictor.analysis.clv import run

    run()


forward_app = typer.Typer(
    no_args_is_help=True,
    help="Forward test: predictions committed before each card, scored after it. "
    "Research only - not betting advice.",
)
app.add_typer(forward_app, name="forward")


def _forward(step: str, event: str | None = None) -> None:
    import datetime as dt
    import json

    from ufc_predictor.forward import run

    cfg = get_config()
    now = dt.datetime.now(dt.UTC)
    fetcher = run.make_fetcher(cfg)
    try:
        if step == "run-due":
            typer.echo(json.dumps(run.run_due(cfg, now=now, fetcher=fetcher), indent=1))
        else:
            out = getattr(run, step)(event, cfg, now=now, fetcher=fetcher)
            typer.echo(out if not isinstance(out, dict) else json.dumps(out["summary"], indent=1))
    except run.ForwardError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@forward_app.command("predict")
def forward_predict(event: str = typer.Option(..., "--event", help="ESPN event id.")) -> None:
    """Predict a card at least forward.min_lead_hours before it starts."""
    _forward("predict", event)


@forward_app.command("close")
def forward_close(event: str = typer.Option(..., "--event", help="ESPN event id.")) -> None:
    """Snapshot Kalshi and sportsbook prices shortly before a card starts (fallback)."""
    _forward("close", event)


@forward_app.command("settle")
def forward_settle(event: str = typer.Option(..., "--event", help="ESPN event id.")) -> None:
    """Score a finished card against its committed predictions."""
    _forward("settle", event)


@forward_app.command("run-due")
def forward_run_due() -> None:
    """Do whatever is due now (the GitHub Actions schedule runs this)."""
    _forward("run-due")


@app.command("export-app")
def export_app_cmd() -> None:
    """Export precomputed artefacts the Streamlit demo loads at startup."""
    from ufc_predictor.export_app import run

    run()


if __name__ == "__main__":  # pragma: no cover
    app()
