"""The offline ingest pipeline: committed snapshot -> processed tables.

Reads only from ``data/raw`` and writes only to ``data/processed``. No network access,
so ``make data`` works from a fresh clone with no live site involved.

Row counts print at every stage and are written to ``reports/ingest_log.json``.
"""

from __future__ import annotations

from ufc_predictor.config import Config, get_config
from ufc_predictor.ingest import normalize as nz
from ufc_predictor.ingest.odds import join_odds_to_fights, load_odds_raw
from ufc_predictor.ingest.stages import StageLog
from ufc_predictor.storage import db


def run(cfg: Config | None = None, *, verbose: bool = True) -> dict:
    """Build every processed table from the committed snapshot.

    Returns:
        Summary dict (also written to ``reports/ingest_log.json``).
    """
    cfg = cfg or get_config()
    log = StageLog(verbose=verbose)
    snap = cfg.paths.ufcstats_snapshot
    ing = cfg.ingest

    if verbose:
        print(f"\nIngest from snapshot: {snap.name}")
        print("  stage                                   rows in      rows out")

    # ---- raw ----
    raw_events = nz.load_raw(snap, "events")
    raw_results = nz.load_raw(snap, "fight_results")
    raw_stats = nz.load_raw(snap, "fight_stats")
    raw_fighters = nz.load_raw(snap, "fighter_tott")

    log.record("raw_events", len(raw_events), len(raw_events))
    log.record("raw_fight_results", len(raw_results), len(raw_results))
    log.record("raw_fight_stats", len(raw_stats), len(raw_stats))
    log.record("raw_fighter_tott", len(raw_fighters), len(raw_fighters))

    # ---- normalize ----
    events = nz.build_events(raw_events, log)
    fighters = nz.build_fighters(raw_fighters, log)
    fights = nz.build_fights(
        raw_results,
        events,
        fighters,
        log,
        max_ambiguous_share=float(ing.get("max_ambiguous_share", 0.02)),
    )
    fight_stats = nz.build_fight_stats(raw_stats, fights, log)

    # ---- odds ----
    odds_raw = load_odds_raw(
        cfg.paths.odds_snapshot,
        log,
        min_overround_exclusive=float(ing.get("min_overround_exclusive", 0.0)),
        max_overround=float(ing.get("max_overround", 0.25)),
    )
    odds = join_odds_to_fights(
        odds_raw,
        fights,
        log,
        aliases_path=cfg.paths.aliases,
        reports_dir=cfg.paths.reports_dir,
        fuzzy_threshold=float(ing.get("fuzzy_threshold", 0.92)),
        ambiguity_margin=float(ing.get("fuzzy_ambiguity_margin", 0.05)),
        max_unmatched_share=float(ing.get("max_unmatched_share", 0.05)),
    )

    # ---- write ----
    processed = cfg.paths.processed_dir
    for name, df in (
        ("events", events),
        ("fighters", fighters),
        ("fights", fights),
        ("fight_stats", fight_stats),
        ("odds", odds),
    ):
        db.write_table(df, processed, name)

    counts = db.table_counts(processed)
    priced = len(odds)
    decisive = int(fights["is_decisive"].sum())

    summary = {
        "snapshot": snap.name,
        "odds_snapshot": cfg.paths.odds_snapshot.name,
        "tables": counts,
        "fights_total": len(fights),
        "fights_decisive": decisive,
        "fights_priced": priced,
        "date_min": str(fights["date"].min()),
        "date_max": str(fights["date"].max()),
    }
    log.write_json(cfg.paths.reports_dir / "ingest_log.json", {"summary": summary})

    if verbose:
        print("\n  processed tables:")
        for k, v in counts.items():
            print(f"    {k:<14} {v:>8,} rows")
        print(
            f"\n  fights {len(fights):,} total | {decisive:,} decisive | {priced:,} priced"
            f" | dates {summary['date_min']} .. {summary['date_max']}"
        )
        print(f"  wrote {cfg.paths.reports_dir / 'ingest_log.json'}\n")

    return summary


if __name__ == "__main__":  # pragma: no cover
    run()
