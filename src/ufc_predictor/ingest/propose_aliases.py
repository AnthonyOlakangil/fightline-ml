"""Propose fighter-name aliases from hard evidence, for human review.

    uv run python -m ufc_predictor.ingest.propose_aliases

The two sources disagree on names for three reasons: legal or married name changes
(UFCStats carries a fighter's current name, the odds data the name used at the time --
``Tecia Torres`` is ``Tecia Pennington``, ``Cris Cyborg`` is ``Cristiane Justino``),
differing word order, and the occasional ring name.

The inference rule used here is deliberately narrow. For an unjoined priced row, look
at bouts on the *same date*; if exactly one of them already matches on one fighter, the
remaining two names must refer to the same person. Date plus one confirmed fighter is
strong evidence, and requiring the partner bout to be unique prevents guessing on cards
where several bouts could fit.

Output is written to ``reports/proposed_aliases.csv`` with the evidence attached. It is
**not** applied automatically: a human reviews it and copies accepted rows into
``src/ufc_predictor/ingest/aliases.csv``. A wrong alias silently mispairs a fight's
closing line with another fighter's result, which no row count would reveal.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import pandas as pd

from ufc_predictor.config import get_config
from ufc_predictor.ingest import normalize as nz
from ufc_predictor.ingest.names import load_aliases, normalize_name
from ufc_predictor.ingest.stages import StageLog


def main() -> None:
    cfg = get_config()
    log = StageLog(verbose=False)
    snap = cfg.paths.ufcstats_snapshot

    events = nz.build_events(nz.load_raw(snap, "events"), log)
    date_by_event = dict(zip(events["event_name"], events["date"], strict=True))

    results = nz.load_raw(snap, "fight_results")
    # date -> list of (name_a, name_b)
    bouts_by_date: dict[object, list[tuple[str, str]]] = defaultdict(list)
    for ev, bout in zip(results["EVENT"], results["BOUT"], strict=True):
        ev = nz._as_text(ev)
        bout = nz._as_text(bout)
        d = date_by_event.get(ev)
        if d is None or " vs. " not in bout:
            continue
        a, b = bout.split(" vs. ", 1)
        bouts_by_date[d].append((normalize_name(a), normalize_name(b)))

    odds = pd.read_parquet(cfg.paths.odds_snapshot / "odds_master.parquet")
    odds["d"] = pd.to_datetime(odds["date"], errors="coerce").dt.date
    existing = load_aliases(cfg.paths.aliases)

    known_pairs = {(d, frozenset(p)) for d, ps in bouts_by_date.items() for p in ps}

    # Every supporting fight is kept, not deduplicated: the count of independent fights
    # backing a mapping is the evidence used to accept or reject it below.
    proposals: list[dict] = []

    for d, rf, bf in zip(odds["d"], odds["R_fighter"], odds["B_fighter"], strict=True):
        if d is None or pd.isna(d):
            continue
        a, b = normalize_name(rf), normalize_name(bf)
        a2, b2 = existing.get(a, a), existing.get(b, b)
        if (d, frozenset((a2, b2))) in known_pairs:
            continue  # already resolves

        same_day = bouts_by_date.get(d, [])
        # Bouts on this date where exactly one side already matches one of our names.
        hits: list[tuple[str, str]] = []  # (odds_name_unmatched, ufcstats_name_unmatched)
        for ca, cb in same_day:
            if a2 == ca and b2 != cb:
                hits.append((b2, cb))
            elif a2 == cb and b2 != ca:
                hits.append((b2, ca))
            elif b2 == ca and a2 != cb:
                hits.append((a2, cb))
            elif b2 == cb and a2 != ca:
                hits.append((a2, ca))

        if len(hits) != 1:
            continue  # zero or ambiguous evidence: do not guess

        odds_name, ufc_name = hits[0]
        if not odds_name or not ufc_name:
            continue
        proposals.append(
            {
                "odds_name": odds_name,
                "ufcstats_name": ufc_name,
                "evidence_date": d,
                "evidence_bout_odds": f"{a} vs {b}",
                "shared_opponent": a2 if odds_name == b2 else b2,
            }
        )

    # Aggregate: how many *independent* fights imply each mapping? A mapping supported
    # by two different cards against two different opponents is not a coincidence, which
    # is a far stronger basis for acceptance than recognising a fighter's nickname.
    agg: dict[tuple[str, str], dict] = {}
    for p in proposals:
        key = (p["odds_name"], p["ufcstats_name"])
        rec = agg.setdefault(
            key,
            {
                "odds_name": p["odds_name"],
                "ufcstats_name": p["ufcstats_name"],
                "evidence_count": 0,
                "opponents": set(),
                "dates": set(),
            },
        )
        rec["evidence_count"] += 1
        rec["opponents"].add(p["shared_opponent"])
        rec["dates"].add(str(p["evidence_date"]))

    # Reject any odds name that maps to more than one UFCStats name: that is a
    # contradiction, and guessing between them risks exactly the mispairing we avoid.
    by_src: dict[str, set[str]] = defaultdict(set)
    for o, u in agg:
        by_src[o].add(u)
    contradictory = {o for o, us in by_src.items() if len(us) > 1}

    rows_out = []
    for rec in agg.values():
        rows_out.append(
            {
                "odds_name": rec["odds_name"],
                "ufcstats_name": rec["ufcstats_name"],
                "evidence_count": rec["evidence_count"],
                "distinct_opponents": len(rec["opponents"]),
                "dates": ";".join(sorted(rec["dates"])[:3]),
                "contradictory": rec["odds_name"] in contradictory,
            }
        )
    rows_out.sort(key=lambda r: (-r["evidence_count"], r["odds_name"]))

    out = cfg.paths.reports_dir / "proposed_aliases.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "odds_name",
                "ufcstats_name",
                "evidence_count",
                "distinct_opponents",
                "dates",
                "contradictory",
            ],
        )
        w.writeheader()
        w.writerows(rows_out)

    multi = [r for r in rows_out if r["evidence_count"] >= 2 and not r["contradictory"]]
    single = [r for r in rows_out if r["evidence_count"] == 1 and not r["contradictory"]]
    bad = [r for r in rows_out if r["contradictory"]]
    print(f"proposed {len(rows_out)} distinct aliases -> {out}")
    print(f"  {len(multi)} corroborated by >=2 independent fights")
    print(f"  {len(single)} supported by a single fight")
    print(
        f"  {len(bad)} contradictory (one odds name -> several UFCStats names); never auto-accept"
    )
    print("\nReview, then copy accepted rows into src/ufc_predictor/ingest/aliases.csv")


if __name__ == "__main__":
    main()
