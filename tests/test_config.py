"""Tests for the configuration loader.

The config module is the project's determinism contract: if the seed can go missing
or a window can silently overlap, every downstream number becomes untrustworthy.
These tests pin that behaviour.
"""

from __future__ import annotations

import datetime as dt
import textwrap

import pytest
import yaml

from ufc_predictor.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def test_seed_is_read_from_yaml_not_hardcoded():
    """The seed must come from config.yaml, proven by editing the file and re-reading."""
    cfg = load_config()
    assert cfg.seed == 42

    on_disk = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert on_disk["seed"] == cfg.seed, "loader returned a seed that is not the one in config.yaml"


def test_seed_tracks_an_edited_config(tmp_path):
    """Change the file, get the changed value -- the loader holds no baked-in default."""
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    base["seed"] = 1234
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")

    assert load_config(p).seed == 1234


def test_windows_parse_to_dates():
    cfg = load_config()
    assert cfg.windows.dev_start == dt.date(2010, 1, 1)
    assert cfg.windows.dev_end == dt.date(2015, 12, 31)
    assert cfg.windows.eval_start == dt.date(2016, 1, 1)
    assert cfg.windows.eval_end == dt.date(2026, 10, 3)


@pytest.mark.leakage
def test_dev_and_eval_windows_are_disjoint():
    """A dev window overlapping the held-out window would leak tuning information.

    Marked `leakage`: this is a point-in-time correctness invariant, not a style
    preference. If the dev window ever reaches into the eval period, every reported
    metric becomes a partly in-sample number.
    """
    cfg = load_config()
    assert cfg.windows.dev_end < cfg.windows.eval_start


def test_paths_resolve_to_absolute_locations():
    cfg = load_config()
    assert cfg.paths.raw_dir.is_absolute()
    assert cfg.paths.processed_dir.is_absolute()
    assert cfg.paths.raw_dir.name == "raw"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_missing_seed_raises(tmp_path):
    """Fail loudly: no silent default seed."""
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del base["seed"]
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ConfigError, match="seed"):
        load_config(p)


@pytest.mark.leakage
def test_overlapping_windows_raise(tmp_path):
    """The headline result is only honest if tuning never saw the eval period.

    Marked `leakage`: an edit to config.yaml must not be able to quietly create an
    overlap. The loader rejects it at load time rather than letting the backtest run.
    """
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    base["windows"]["dev_end"] = "2018-01-01"  # now overlaps eval_start 2016-01-01
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ConfigError, match="leak tuning information"):
        load_config(p)


def test_unparseable_date_raises(tmp_path):
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    base["windows"]["eval_start"] = "not-a-date"
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ConfigError, match="ISO date"):
        load_config(p)


def test_non_integer_seed_raises(tmp_path):
    base = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    base["seed"] = "forty-two"
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ConfigError, match="must be an int"):
        load_config(p)


def test_malformed_yaml_raises(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent("- just\n- a\n- list\n"), encoding="utf-8")

    with pytest.raises(ConfigError, match="top-level mapping"):
        load_config(p)
