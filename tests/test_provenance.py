"""git_state: which commit produced an artefact, and whether the code matched it."""

from __future__ import annotations

import subprocess

from ufc_predictor.provenance import OUTPUT_DIRS, git_state


def _repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "src" / "m.py").write_text("x = 1\n")
    (tmp_path / "reports" / "out.json").write_text("{}\n")
    (tmp_path / "README.md").write_text("# r\n")
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], check=True)


def test_an_output_only_change_leaves_the_code_tree_clean(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    (tmp_path / "reports" / "out.json").write_text('{"a": 1}\n')
    (tmp_path / "README.md").write_text("# regenerated\n")
    assert git_state()["clean"] is False
    assert git_state(ignore=OUTPUT_DIRS)["clean"] is True


def test_a_code_change_is_dirty_even_when_outputs_are_ignored(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    (tmp_path / "src" / "m.py").write_text("x = 2\n")
    assert git_state(ignore=OUTPUT_DIRS)["clean"] is False
    (tmp_path / "src" / "new.py").write_text("y = 1\n")
    assert git_state(ignore=OUTPUT_DIRS)["clean"] is False
