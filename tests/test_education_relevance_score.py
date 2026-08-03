import pandas as pd

from core.scoring import calculate_education_relevance_score


def _score_rows(rows):
    df = pd.DataFrame(rows)
    df = calculate_education_relevance_score(df, jd=None)
    return df["education_relevance_score"].tolist()


def test_education_tiers():
    rows = [
        {"fields": ["Human Resources Management and Services"], "certifications": []},  # 1.0
        {"fields": ["Psychology"], "certifications": []},                                # 1.0
        {"fields": ["Legal and Social Sciences"], "certifications": []},                 # 1.0 (law/legal)
        {"fields": ["Business Administration and Management"], "certifications": []},     # 0.75
        {"fields": ["Finance"], "certifications": []},                                    # 0.75
        {"fields": ["Computer Science"], "certifications": []},                           # 0.50 unrelated
    ]
    assert _score_rows(rows) == [1.0, 1.0, 1.0, 0.75, 0.75, 0.5]


def test_education_max_tier_and_certs():
    rows = [
        # best credential wins: unrelated degree + HR cert -> 1.0
        {"fields": ["Computer Science"], "certifications": ["CIPD Level 5 Diploma"]},
        # business + unrelated -> 0.75
        {"fields": ["English Literature", "MBA"], "certifications": []},
        # unrelated + relevant -> 1.0
        {"fields": ["Computer Science", "Psychology"], "certifications": []},
    ]
    assert _score_rows(rows) == [1.0, 0.75, 1.0]


def test_education_neutral_floor_when_absent():
    rows = [
        {"fields": [], "certifications": []},
        {"fields": ["N/A"], "certifications": []},
        {"fields": None, "certifications": None},
    ]
    # never below the 0.5 neutral floor (soft bonus, never a gate)
    assert _score_rows(rows) == [0.5, 0.5, 0.5]
