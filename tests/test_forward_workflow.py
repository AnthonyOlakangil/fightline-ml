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
    due = next(s for s in steps if s.get("id") == "due")
    assert due["env"]["GITHUB_TOKEN"] == "${{ github.token }}"  # settlement queries the API
    refresh = next(s for s in steps if s.get("name", "").startswith("Refresh"))
    assert refresh["if"] == "always()" and "readme_sections" in refresh["run"]
    assert "README.md" in commit["run"]
    alert = next(s for s in steps if "issue" in s.get("name", ""))
    assert alert["if"] == "failure()" and "@AnthonyOlakangil" in alert["run"]


def test_new_predictions_are_sealed_by_an_artifact_before_they_are_committed():
    steps = load()["jobs"]["run-due"]["steps"]
    names = [s.get("name", "") for s in steps]
    seal = next(i for i, s in enumerate(steps) if s.get("id") == "seal")
    upload = next(i for i, s in enumerate(steps) if "upload-artifact" in str(s.get("uses", "")))
    commit = next(i for i, n in enumerate(names) if n.startswith("Commit"))
    assert seal < upload < commit
    assert steps[seal]["if"] == "always()" and "sha256sum" in steps[seal]["run"]
    assert (
        "forward-seal-" in steps[seal]["run"]
        and "git ls-files --error-unmatch" in steps[seal]["run"]
    )
    assert steps[upload]["with"]["name"] == "${{ steps.seal.outputs.name }}"
    assert "steps.seal.outputs.name != ''" in steps[upload]["if"]
