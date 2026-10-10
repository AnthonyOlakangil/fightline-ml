"""Build the feature table from processed tables and write it to Parquet.

Reports row counts and the label base rate at every stage, and writes a summary to
``reports/feature_summary.json``.
"""

from __future__ import annotations

import json

from ufc_predictor.config import Config, get_config
from ufc_predictor.features.pit import FEATURE_COLS, build_features
from ufc_predictor.storage import db


def run(cfg: Config | None = None, *, verbose: bool = True) -> dict:
    """Build ``data/processed/features.parquet``.

    Returns:
        Summary dict (also written to ``reports/feature_summary.json``).
    """
    cfg = cfg or get_config()
    fights = db.read_table(cfg.paths.processed_dir, "fights")
    stats = db.read_table(cfg.paths.processed_dir, "fight_stats")
    fighters = db.read_table(cfg.paths.processed_dir, "fighters")
    odds = db.read_table(cfg.paths.processed_dir, "odds")

    if verbose:
        print("\nBuilding point-in-time features")
        print(f"  input: {len(fights):,} fights, {len(stats):,} round-stat rows")

    feats = build_features(
        fights,
        stats,
        fighters,
        rolling_windows=tuple(cfg.features["rolling_windows"]),
        elo_params=cfg.elo,
    )

    if len(feats) != len(fights):
        raise RuntimeError(
            f"feature builder returned {len(feats)} rows for {len(fights)} fights; "
            "every fight must produce exactly one row"
        )

    priced = set(odds["fight_id"])
    feats["is_priced"] = feats["fight_id"].isin(priced)

    labelled = feats["label"].notna()
    base_rate = float(feats.loc[labelled, "label"].mean())
    if not 0.45 <= base_rate <= 0.55:
        raise RuntimeError(
            f"label base rate is {base_rate:.4f}, which suggests the fight orientation "
            "encodes the outcome; canonical orientation must be label-independent"
        )

    db.write_table(feats, cfg.paths.processed_dir, "features")

    # Null rate per feature, for the README's honest account of coverage.
    null_rates = {c: float(feats[c].isna().mean()) for c in FEATURE_COLS}
    worst = sorted(null_rates.items(), key=lambda kv: -kv[1])[:5]

    summary = {
        "rows": len(feats),
        "feature_count": len(FEATURE_COLS),
        "labelled_rows": int(labelled.sum()),
        "unlabelled_draws_and_nc": int((~labelled).sum()),
        "priced_rows": int(feats["is_priced"].sum()),
        "priced_and_labelled": int((feats["is_priced"] & labelled).sum()),
        "label_base_rate": base_rate,
        "debut_slots": int(feats["is_debut_1"].sum() + feats["is_debut_2"].sum()),
        "date_min": str(feats["date"].min()),
        "date_max": str(feats["date"].max()),
        "highest_null_rate_features": dict(worst),
    }

    path = cfg.paths.reports_dir / "feature_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if verbose:
        print(f"  features:            {len(feats):,} rows x {len(FEATURE_COLS)} features")
        print(f"  labelled (decisive): {summary['labelled_rows']:,}")
        print(f"  draws / no contests: {summary['unlabelled_draws_and_nc']:,} (label = NaN)")
        print(f"  priced + labelled:   {summary['priced_and_labelled']:,}  <- backtest scoring set")
        print(f"  label base rate:     {base_rate:.4f}  (0.50 = orientation carries no signal)")
        print(
            f"  debut fighter slots: {summary['debut_slots']:,}  (rolling features NaN + flagged)"
        )
        print(f"  wrote {path}\n")

    return summary


if __name__ == "__main__":  # pragma: no cover
    run()
