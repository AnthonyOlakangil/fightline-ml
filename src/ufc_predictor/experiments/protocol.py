"""The Phase 8 development protocol: which fights may be scored, and the record of every try.

The 2016+ evaluation window is **closed**: it has been scored three times and its
numbers are published, so any further look at it would turn it into a tuning set. Model
improvement therefore happens inside the development window, split in two:

* **selection** (``dev_start`` .. ``selection_end``, 2010-2014): every candidate is
  scored here, walk-forward, and compared by the pre-registered rule;
* **confirmation** (``confirmation_start`` .. ``confirmation_end``, 2015): the frozen
  configuration is scored here exactly once (``experiments/confirm.py``).

Three guards make that structural rather than a matter of care:

* :func:`assert_scorable` refuses any date in the evaluation window (``EvalWindowClosed``),
  and refuses a selection score outside the selection window or a confirmation score
  outside the confirmation year. Every Phase 8 scoring path calls it before returning;
* :func:`assert_after_preregistration` refuses to run an experiment from a commit that
  does not contain ``config/experiments.yaml``, or where it has changed since the commit
  that added it -- the experiments, their order and the decision rule were fixed first;
* :func:`append_experiment` writes every run to an append-only ledger with the commit,
  whether the tree was clean, and digests of the config and the pre-registration.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import yaml

from ufc_predictor.config import DEFAULT_CONFIG_PATH, REPO_ROOT, Config
from ufc_predictor.provenance import OUTPUT_DIRS, git_state

Purpose = Literal["selection", "confirmation", "dev"]

#: The pre-registration, relative to the repository root.
PREREG_RELPATH = "config/experiments.yaml"
PREREG_PATH = REPO_ROOT / PREREG_RELPATH


class EvalWindowClosed(RuntimeError):
    """Raised when a Phase 8 scoring path reaches a date in the closed evaluation window."""


def window_for(purpose: Purpose, cfg: Config) -> tuple[dt.date, dt.date]:
    """The inclusive date range a score for ``purpose`` may cover."""
    w = cfg.windows
    if purpose == "selection":
        return w.dev_start, w.selection_end
    if purpose == "confirmation":
        return w.confirmation_start, w.confirmation_end
    if purpose == "dev":
        return w.dev_start, w.dev_end
    raise ValueError(f"unknown scoring purpose {purpose!r}")


def assert_scorable(dates: Iterable[dt.date], cfg: Config, *, purpose: Purpose) -> None:
    """Raise unless every date may be scored for ``purpose``.

    Raises:
        EvalWindowClosed: if any date is on or after ``eval_start``.
        ValueError: if a date falls outside the purpose's window, or there are no dates.
    """
    dates = list(dates)
    if not dates:
        raise ValueError(f"no dates to score for {purpose}")
    latest = max(dates)
    if latest >= cfg.windows.eval_start:
        raise EvalWindowClosed(
            f"a {purpose} score reaches {latest}, inside the closed evaluation window "
            f"(from {cfg.windows.eval_start}); Phase 8 never scores it"
        )
    start, end = window_for(purpose, cfg)
    earliest = min(dates)
    if earliest < start or latest > end:
        raise ValueError(
            f"a {purpose} score covers {earliest}..{latest}, outside the {purpose} window "
            f"{start}..{end}"
        )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_preregistration(path: Path = PREREG_PATH) -> dict:
    """The pre-registered stages, candidates and decision rule."""
    with Path(path).open("r", encoding="utf-8") as fh:
        pre = yaml.safe_load(fh)
    if not isinstance(pre, dict) or "stages" not in pre:
        raise RuntimeError(f"{path} is not a pre-registration (no 'stages')")
    return pre


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


def preregistration_commit(repo: Path = REPO_ROOT) -> str | None:
    """The commit that added the pre-registration file, if HEAD's history contains one."""
    out = _git(repo, "log", "--diff-filter=A", "--format=%H", "--", PREREG_RELPATH)
    commits = [line for line in out.stdout.split() if line]
    return commits[-1] if commits else None


def assert_after_preregistration(repo: Path = REPO_ROOT) -> str:
    """Raise unless HEAD contains the pre-registration, unchanged since it was committed.

    Returns:
        The pre-registration commit.
    """
    commit = preregistration_commit(repo)
    if commit is None:
        raise RuntimeError(
            f"no pre-registration: {PREREG_RELPATH} is not in this commit's history, so "
            "experiments would run from a commit that predates their registration"
        )
    if _git(repo, "diff", "--quiet", commit, "HEAD", "--", PREREG_RELPATH).returncode != 0:
        raise RuntimeError(
            f"{PREREG_RELPATH} has changed since the pre-registration commit {commit[:7]}; "
            "the experiments and the decision rule are fixed"
        )
    if _git(repo, "diff", "--quiet", "--", PREREG_RELPATH).returncode != 0:
        raise RuntimeError(
            f"{PREREG_RELPATH} has uncommitted edits, so it has changed since the "
            f"pre-registration commit {commit[:7]}"
        )
    return commit


def ledger_path(cfg: Config) -> Path:
    return cfg.paths.reports_dir / "dev" / "experiments.json"


def append_experiment(entry: dict, cfg: Config, *, path: Path | None = None) -> int:
    """Append one run to the experiment ledger and return its run number.

    Raises:
        RuntimeError: if the existing ledger's run numbers are not exactly 1..n, which
            means an entry was removed or reordered.
    """
    path = Path(path) if path is not None else ledger_path(cfg)
    entries: list[dict] = []
    if path.is_file():
        entries = json.loads(path.read_text(encoding="utf-8"))
    runs = [e.get("run") for e in entries]
    if runs != list(range(1, len(entries) + 1)):
        raise RuntimeError(
            f"experiment ledger {path} has run numbers {runs[:10]}..., not 1..{len(entries)}; "
            "it is append-only"
        )
    state = git_state(ignore=OUTPUT_DIRS)
    record = {
        "run": len(entries) + 1,
        "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": state["commit"],
        "worktree_clean": bool(state["clean"]),
        "config_hash": file_sha256(DEFAULT_CONFIG_PATH),
        "preregistration_hash": file_sha256(PREREG_PATH),
        **entry,
    }
    entries.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, default=str) + "\n", encoding="utf-8")
    return record["run"]
