"""Row-count logging for pipeline stages.

CLAUDE.md requires row counts at every pipeline stage and that a row-dropping join
raises rather than continuing with partial data. ``StageLog`` makes both mechanical:
every stage records counts, every drop records a reason, and ``reconcile`` asserts the
arithmetic closes so a quiet row loss cannot slip through.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class PipelineError(RuntimeError):
    """Raised when a pipeline stage loses rows it cannot account for."""


@dataclass
class Stage:
    name: str
    rows_in: int
    rows_out: int
    drops: dict[str, int] = field(default_factory=dict)

    @property
    def dropped(self) -> int:
        return sum(self.drops.values())

    def reconcile(self) -> None:
        """Assert rows_in == rows_out + accounted drops."""
        if self.rows_in != self.rows_out + self.dropped:
            unexplained = self.rows_in - self.rows_out - self.dropped
            raise PipelineError(
                f"stage {self.name!r} lost {unexplained} unaccounted rows: "
                f"in={self.rows_in} out={self.rows_out} "
                f"explained_drops={self.dropped} {self.drops}"
            )


class StageLog:
    """Accumulates per-stage row counts and prints them as the pipeline runs."""

    def __init__(self, *, verbose: bool = True) -> None:
        self.stages: list[Stage] = []
        self.verbose = verbose

    def record(
        self,
        name: str,
        rows_in: int,
        rows_out: int,
        drops: dict[str, int] | None = None,
        *,
        reconcile: bool = True,
    ) -> Stage:
        """Record a stage, print it, and (by default) assert its row arithmetic closes."""
        stage = Stage(name=name, rows_in=rows_in, rows_out=rows_out, drops=dict(drops or {}))
        self.stages.append(stage)
        if self.verbose:
            print(self.format(stage), flush=True)
        if reconcile:
            stage.reconcile()
        return stage

    @staticmethod
    def format(stage: Stage) -> str:
        line = f"  {stage.name:<38} in={stage.rows_in:>7,}  out={stage.rows_out:>7,}"
        if stage.drops:
            detail = ", ".join(f"{k}={v:,}" for k, v in sorted(stage.drops.items()) if v)
            if detail:
                line += f"  dropped: {detail}"
        return line

    def to_dict(self) -> dict:
        return {
            "stages": [
                {
                    "name": s.name,
                    "rows_in": s.rows_in,
                    "rows_out": s.rows_out,
                    "dropped": s.dropped,
                    "drops": s.drops,
                }
                for s in self.stages
            ]
        }

    def write_json(self, path: Path, extra: dict | None = None) -> None:
        payload = self.to_dict()
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
