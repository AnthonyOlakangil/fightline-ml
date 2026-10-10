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
    _not_yet("data", "Phase 1")


@app.command()
def features() -> None:
    """Build the point-in-time feature table."""
    _not_yet("features", "Phase 2")


@app.command()
def backtest() -> None:
    """Run the walk-forward backtest and write reports/metrics.json."""
    _not_yet("backtest", "Phase 4")


@app.command()
def report() -> None:
    """Generate charts and refresh README figures from metrics.json."""
    _not_yet("report", "Phase 4")


@app.command()
def predict(
    event: str = typer.Option(..., "--event", help="Event name, e.g. 'UFC 332: Silva vs. Wang'."),
) -> None:
    """Predict a card and write a prediction-log CSV."""
    _not_yet(f"predict --event {event!r}", "Phase 6")


if __name__ == "__main__":  # pragma: no cover
    app()
