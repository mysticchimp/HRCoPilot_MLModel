import pandas as pd

from scripts.calibrate_weights import (
    _credit_count,
    _floor_results,
    _reset_floors,
    _truncate_floor,
)


def test_floor_reset_truncates_new_incumbent_gold_metrics():
    incumbent = {
        "raw_ndcg@5": 0.92789,
        "raw_ndcg@10": 0.93456,
    }

    floors = _reset_floors(incumbent)

    assert _truncate_floor(0.92789) == 0.92
    assert floors["ndcg@5"] == 0.92
    assert floors["ndcg@10"] == 0.93


def test_floor_results_use_unrounded_gold_metrics():
    summary = {
        "seed_found_rate": 1.0,
        "hit@3": 0.5,
        "hit@5": 0.5,
        "hit@10": 0.7,
        "mrr": 0.5,
        "ndcg@5": 0.91,
        "ndcg@10": 0.91,
        "raw_ndcg@5": 0.90999,
        "raw_ndcg@10": 0.91999,
    }
    floors = {
        "seed_found_rate": 1.0,
        "hit@3": 0.42,
        "hit@5": 0.47,
        "hit@10": 0.68,
        "mrr": 0.41,
        "ndcg@5": 0.91,
        "ndcg@10": 0.91,
    }

    result = _floor_results(summary, floors)

    assert result["metrics"]["ndcg@5"]["pass"] is False
    assert result["metrics"]["ndcg@10"]["pass"] is True
    assert result["pass"] is False


def test_structured_credit_count_averages_two_judges():
    frame = pd.DataFrame(
        {
            "left_credited_preferred_signals": [
                '["english", "tagalog_or_filipino"]',
                "[]",
            ],
            "right_credited_preferred_signals": [
                '["english"]',
                '["cipd_or_equivalent"]',
            ],
        }
    )

    language = _credit_count(
        frame,
        ["left", "right"],
        {"english", "arabic", "tagalog_or_filipino"},
    )
    certification = _credit_count(
        frame,
        ["left", "right"],
        {"cipd_or_equivalent"},
    )

    assert language.tolist() == [1.5, 0.0]
    assert certification.tolist() == [0.0, 0.5]