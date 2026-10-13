"""Which commit produced an artefact, and whether the working tree matched it."""

from __future__ import annotations

import subprocess

#: Generated outputs. A step that writes them before they can be committed -- parlay
#: pricing reads a freshly scored run's predictions -- checks the *code* is committed
#: and records digests of its inputs instead.
OUTPUT_DIRS = ("reports/", "app/artifacts/")


def git_state(*, ignore: tuple[str, ...] = ()) -> dict:
    """Full HEAD commit, its short form, and whether the working tree matches it.

    Untracked files count as dirty: an uncommitted module is still uncommitted code.
    Paths under any prefix in ``ignore`` are left out of the cleanliness check.
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
    dirty = [
        line[3:]
        for line in status.splitlines()
        if line.strip() and not line[3:].startswith(ignore or ("\0",))
    ]
    return {"commit": commit, "short": commit[:7], "clean": not dirty}
