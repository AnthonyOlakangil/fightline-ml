"""Which commit produced an artefact, and whether the working tree matched it."""

from __future__ import annotations

import subprocess


def git_state() -> dict:
    """Full HEAD commit, its short form, and whether the working tree matches it.

    Untracked files count as dirty: an uncommitted module is still uncommitted code.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:  # pragma: no cover - git may be absent
        return {"commit": "unknown", "short": "unknown", "clean": False}
    return {"commit": commit, "short": commit[:7], "clean": status.strip() == ""}
