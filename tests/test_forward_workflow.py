"""The scheduled forward-test workflow keeps the properties the forward test relies on."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "forward.yml"


def load():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_runs_on_a_schedule_and_by_hand():
    on = load()[True]  # PyYAML reads the bare key `on` as True
    assert on["schedule"][0]["cron"] and "workflow_dispatch" in on


def test_can_push_and_open_issues_but_never_runs_twice_at_once():
    wf = load()
    assert wf["permissions"] == {"contents": "write", "issues": "write"}
    assert wf["concurrency"]["cancel-in-progress"] is False


def test_records_are_committed_even_after_a_failure_and_failures_alert():
    steps = load()["jobs"]["run-due"]["steps"]
    checkout = steps[0]
    assert checkout["with"]["fetch-depth"] == 0  # settlement reads first-commit times
    commit = next(s for s in steps if s.get("name", "").startswith("Commit"))
    assert commit["if"] == "always()" and "git push" in commit["run"]
    assert "--force" not in commit["run"]
    refresh = next(s for s in steps if s.get("name", "").startswith("Refresh"))
    assert refresh["if"] == "always()" and "readme_sections" in refresh["run"]
    assert "README.md" in commit["run"]
    alert = next(s for s in steps if "issue" in s.get("name", ""))
    assert alert["if"] == "failure()" and "@AnthonyOlakangil" in alert["run"]
