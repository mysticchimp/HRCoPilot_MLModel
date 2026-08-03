import pandas as pd

from core.scoring import calculate_experience_relevance_score


def _score_rows(rows):
    df = pd.DataFrame(rows)
    df = calculate_experience_relevance_score(df, jd=None)  # scorer is JD-independent
    return [round(v, 3) for v in df["experience_relevance_score"].tolist()]


def test_experience_relevance_ratio():
    rows = [
        # samastha-like: all HR -> fully relevant
        {"relevant_months": 55, "adjacent_months": 0, "total_dated_months": 55},
        # amulya-like: 19 relevant of 45 total (HR-Ops vs Process) -> 0.422
        {"relevant_months": 19, "adjacent_months": 0, "total_dated_months": 45},
        # purely adjacent history -> half credit
        {"relevant_months": 0, "adjacent_months": 20, "total_dated_months": 40},
        # mixed relevant + adjacent
        {"relevant_months": 20, "adjacent_months": 20, "total_dated_months": 40},
    ]
    assert _score_rows(rows) == [1.0, 0.422, 0.25, 0.75]


def test_experience_relevance_neutral_when_no_dated_roles():
    rows = [
        {"relevant_months": 0, "adjacent_months": 0, "total_dated_months": 0},
        {"relevant_months": None, "adjacent_months": None, "total_dated_months": None},
    ]
    assert _score_rows(rows) == [0.5, 0.5]


def test_experience_relevance_clamped():
    # defensive: ratio can never exceed 1.0 even if months are inconsistent
    rows = [{"relevant_months": 60, "adjacent_months": 20, "total_dated_months": 50}]
    assert _score_rows(rows) == [1.0]
