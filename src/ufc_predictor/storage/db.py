"""DuckDB helpers.

Parquet is the storage format and DuckDB is the query engine, so tables are registered
as views over files rather than loaded into a database file. That keeps the processed
directory as the single source of truth and makes every table inspectable with a plain
SQL prompt.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

TABLES = ("events", "fighters", "fights", "fight_stats", "odds")


def write_table(df: pd.DataFrame, processed_dir: Path, name: str) -> Path:
    """Write one processed table to Parquet and return its path."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / f"{name}.parquet"
    df.to_parquet(path, index=False, compression="zstd")
    return path


def read_table(processed_dir: Path, name: str) -> pd.DataFrame:
    """Read one processed table from Parquet."""
    path = processed_dir / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"processed table {name!r} not found at {path}. Run `make data` first."
        )
    return pd.read_parquet(path)


def connect(processed_dir: Path) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB with each processed Parquet registered as a view.

    >>> con = connect(Path("data/processed"))
    >>> con.sql("SELECT count(*) FROM fights").fetchone()
    """
    con = duckdb.connect(":memory:")
    for name in TABLES:
        path = processed_dir / f"{name}.parquet"
        if path.is_file():
            # DDL cannot take prepared parameters in DuckDB, so the path is inlined as
            # a quoted literal with embedded quotes escaped. Paths come from config,
            # not user input, and `name` is from the fixed TABLES tuple.
            literal = str(path).replace("'", "''")
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{literal}')")
    return con


def table_counts(processed_dir: Path) -> dict[str, int]:
    """Row count per registered table, for the pipeline summary."""
    con = connect(processed_dir)
    counts: dict[str, int] = {}
    for name in TABLES:
        if (processed_dir / f"{name}.parquet").is_file():
            counts[name] = con.sql(f"SELECT count(*) FROM {name}").fetchone()[0]
    con.close()
    return counts
