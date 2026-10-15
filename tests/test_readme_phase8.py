"""Phase 8 README renderers on constructed inputs (the real ones are checked against the
README in test_readme_sections.py once their JSON exists)."""

from __future__ import annotations

from ufc_predictor.backtest import readme_phase8 as r8

SPEC = {
    "features": ["base", "glicko"],
    "top_k": None,
    "train_window": "all_history",
    "symmetric": True,
    "calibration": "platt",
    "model": "lightgbm",
    "lightgbm_params": None,
}


def test_missing_sources_render_as_not_yet_run():
    assert "Not yet run" in r8.render_stages(None)
    assert "Not yet run" in r8.render_confirmation(None)
    assert "Not yet run" in r8.render_clv(None, None, None)
    assert "No card predicted yet" in r8.render_forward(None, [])


def _candidate(observed, adopted=False, name="platt"):
    return {
        "name": name,
        "log_loss": 0.65 + observed,
        "observed": observed,
        "ci_lower": observed - 0.01,
        "ci_hi": observed + 0.01,
        "p_one_sided": 0.2,
        "p_holm": 0.4,
        "adopted": adopted,
    }


def stages(adopted: str | None):
    inc = {"log_loss": 0.6500, "n_fights": 1693, "n_events": 150}
    return {
        "selection_window": ["2010-01-01", "2014-12-31"],
        "preregistration_commit": "906ac9c" + "0" * 33,
        "git_commit": "abcdef1" + "0" * 33,
        "alpha": 0.1,
        "stages": [
            {
                "name": "stage_0_retune",
                "kind": "retune",
                "n_candidates": 216,
                "validation_rows": 400,
                "validation_span": ["2014-01-01", "2014-12-31"],
                "n_tied": 3,
                "lowest_log_loss": 0.66,
                "selected": {
                    "max_depth": 3,
                    "num_leaves": 7,
                    "min_child_samples": 80,
                    "learning_rate": 0.02,
                    "colsample_bytree": 0.5,
                    "reg_lambda": 5.0,
                },
            },
            {
                "name": "stage_5_calibration",
                "kind": "compare",
                "field": "calibration",
                "incumbent": inc,
                "candidates": [
                    _candidate(-0.004, adopted is not None),
                    _candidate(0.002, name="none"),
                ],
                "adopted": adopted,
            },
            {"name": "final_retune", "kind": "final_retune", "skipped": "model is logistic"},
        ],
        "frozen_spec": SPEC,
        "frozen_selection_log_loss": 0.6460,
        "frozen_selection_market_log_loss": 0.6100,
    }


def test_stages_table_lists_every_candidate_and_the_verdict():
    text = r8.render_stages(stages(None))
    assert text.count("| 5 calibration |") == 3  # incumbent + two candidates
    assert "Nothing was adopted" in text
    assert "0.6460" in text and "0.6100" in text
    adopted = r8.render_stages(stages("platt"))
    assert "**yes**" in adopted and "stage 5 calibration → platt" in adopted


def test_spec_description_is_readable():
    text = r8.describe_spec(SPEC)
    assert (
        text == "run-3 base (117) + Glicko-2, all history, orientation-symmetric, LightGBM, Platt"
    )


def test_forward_table_marks_unverified_cards():
    ledger = [
        {
            "event_id": "1",
            "action": "predict",
            "event_name": "UFC 333",
            "at_utc": "2026-10-22T17:17:00+00:00",
        },
        {"event_id": "1", "action": "settle"},
        {"event_id": "2", "action": "missed"},
    ]
    summary = {
        "events": [
            {
                "event_id": "1",
                "verified": False,
                "sealed": True,
                "n_scored": 11,
                "model_log_loss": 0.66,
                "n_with_kalshi_close": 9,
                "model_log_loss_on_kalshi_close_bouts": 0.65,
                "kalshi_close_log_loss": 0.62,
            }
        ],
        "pooled_verified": None,
    }
    text = r8.render_forward(summary, ledger)
    assert (
        "| UFC 333 | 2026-10-22 17:17 UTC | yes | 11 | 0.6600 | 9 | 0.6500 / 0.6200 | **no** | yes |"
        in text
    )
    assert "| 2 | missed | no |" in text


def _metrics(observed, lo, hi):
    return {
        "results": {
            "eval": {"market": {"n": 4636, "log_loss": 0.6063}, "lightgbm": {"log_loss": 0.6796}}
        },
        "paired_vs_market": {
            "eval": {"lightgbm": {"observed": observed, "ci_lower": lo, "ci_hi": hi}}
        },
        "windows": {"eval": ["2016-01-01", "2026-10-03"]},
    }


def _run(observed):
    return {"paired_vs_market": {"eval": {"lightgbm": {"observed": observed}}}}


PARLAY = {
    "by_source": {
        "lightgbm": {
            "predicted_vs_realized": {
                "all_top_ranked": {"mean_predicted_ev": 6.3207, "realized_roi": -0.5057}
            }
        }
    }
}


def test_summary_states_the_result_the_leak_the_parlays_and_the_forward_test():
    ledger = [{"event_id": "1", "action": "predict"}, {"event_id": "1", "action": "settle"}]
    text = r8.render_summary(
        _metrics(0.0734, 0.0640, 0.0828),
        _run(0.0710),
        _run(0.0853),
        PARLAY,
        stages(None),
        ledger,
        {"events": [{"event_id": "1"}]},
    )
    assert "**It does not:**" in text and "+0.0734 [95% CI +0.0640, +0.0828]" in text
    assert "from +0.0710 to +0.0853" in text
    assert "promised +632% and returned \u221251%" in text
    assert "kept nothing" in text  # stages(None) adopts nothing
    assert "(1 predicted, 1 settled so far)" in text
    assert len(text.split()) < 150


def test_summary_says_so_if_the_model_ever_beats_the_market():
    text = r8.render_summary(
        _metrics(-0.01, -0.02, -0.001), _run(0.0), _run(0.0), PARLAY, None, [], None
    )
    assert "**It does:**" in text and "development is closed" not in text
