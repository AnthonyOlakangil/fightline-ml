"""The forward ledger: an append-only record of what was predicted, priced and settled, when.

The forward test is only worth something if each card's predictions demonstrably
existed before the card started. Two records establish that:

* the **git history** of ``reports/forward/<event>/predictions.csv`` -- the commit that
  added it must be at least ``forward.min_lead_hours`` before the scheduled start, and
  the file must never change afterwards (:func:`committed_before`,
  :func:`unchanged_since_added`);
* the **GitHub Actions run** that made the commit, whose id and URL are written into
  the ledger entry, an independent server-side timestamp that a local clock cannot set.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

ACTIONS = ("predict", "close", "settle", "missed")


def load(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else []


def has(entries: list[dict], event_id: str, action: str) -> bool:
    return any(e["event_id"] == event_id and e["action"] == action for e in entries)


def append(path: Path, entry: dict) -> None:
    """Append one entry; an event takes each action at most once.

    Raises:
        ValueError: for an unknown action, or one already recorded for the event.
    """
    if entry.get("action") not in ACTIONS:
        raise ValueError(f"unknown forward action {entry.get('action')!r}")
    entries = load(path)
    if has(entries, entry["event_id"], entry["action"]):
        raise ValueError(f"{entry['action']} is already recorded for event {entry['event_id']}")
    entries.append(entry)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


def first_commit_time(repo: Path, relpath: str) -> dt.datetime | None:
    """Committer time of the commit that added ``relpath``, or None if it never was."""
    out = _git(repo, "log", "--diff-filter=A", "--format=%cI", "--", relpath).split()
    return dt.datetime.fromisoformat(out[-1]).astimezone(dt.UTC) if out else None


def committed_before(repo: Path, relpath: str, deadline: dt.datetime) -> bool:
    added = first_commit_time(repo, relpath)
    return added is not None and added <= deadline


def unchanged_since_added(repo: Path, relpath: str) -> bool:
    """True if no commit after the one that added ``relpath`` changed it, and the
    working copy matches."""
    commits = _git(repo, "log", "--format=%H", "--", relpath).split()
    if len(commits) != 1:
        return False
    return not _git(repo, "status", "--porcelain", "--", relpath).strip()
