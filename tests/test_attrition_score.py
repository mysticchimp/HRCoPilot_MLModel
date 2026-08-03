import pandas as pd

from core.scoring import calculate_attrition_score


def _score_rows(rows):
    df = pd.DataFrame(rows)
    df = calculate_attrition_score(df, jd=None)  # scorer is JD-independent
    return df["attrition_score"].tolist()


def test_attrition_bands_and_guards():
    rows = [
        # samastha-like hopper: median completed-perm 13 -> 12-18 band
        {"n_dated_roles": 4, "completed_perm_tenures": [14, 12, 13], "years_experience": 4.6},
        # amulya-like: single 26-mo completed role -> >=24 band
        {"n_dated_roles": 2, "completed_perm_tenures": [26], "years_experience": 3.75},
        # mid band: median 20 -> 18-24
        {"n_dated_roles": 2, "completed_perm_tenures": [20], "years_experience": 5.0},
        # bin-siddique-like chronic hopper: 3 perm roles all < 12 -> 0.30
        {"n_dated_roles": 4, "completed_perm_tenures": [9, 11, 11], "years_experience": 3.6},
        # one-off short stint (not chronic): median 7, only 2 short roles -> 0.50
        {"n_dated_roles": 3, "completed_perm_tenures": [6, 8], "years_experience": 6.0},
    ]
    assert _score_rows(rows) == [0.65, 1.00, 0.85, 0.30, 0.50]


def test_attrition_early_career_floor():
    # junior: one short completed stint would be 0.50, but the early-career floor lifts it
    rows = [{"n_dated_roles": 2, "completed_perm_tenures": [8], "years_experience": 1.5}]
    assert _score_rows(rows) == [0.70]
    # the floor does NOT apply once the history is long enough (n_dated > max_roles)
    rows = [{"n_dated_roles": 4, "completed_perm_tenures": [8, 9, 10], "years_experience": 2.5}]
    assert _score_rows(rows) == [0.30]  # chronic hop, not floored


def test_attrition_neutral_when_cannot_assess():
    rows = [
        {"n_dated_roles": 1, "completed_perm_tenures": [], "years_experience": 1.0},   # < 2 roles
        {"n_dated_roles": 3, "completed_perm_tenures": [], "years_experience": 5.0},   # no completed-perm
        {"n_dated_roles": 0, "completed_perm_tenures": [], "years_experience": None},  # nothing
    ]
    assert _score_rows(rows) == [0.5, 0.5, 0.5]


def test_attrition_short_current_role_not_penalized():
    # a stable history with a short CURRENT role: current is excluded from completed_perm,
    # so the median reflects the stable prior roles, not the short current stint.
    rows = [{"n_dated_roles": 3, "completed_perm_tenures": [36, 30], "current_tenure_months": 3,
             "years_experience": 6.0}]
    assert _score_rows(rows) == [1.00]


def test_attrition_current_tenure_lifts_only():
    # the current role is right-censored: a LONG current tenure can only RAISE the score,
    # a short one is inconclusive and never lowers it.
    rows = [
        # loyal but thin prior history: completed [10] -> 0.50, but 40mo current lifts to 1.00
        {"n_dated_roles": 2, "completed_perm_tenures": [10], "current_tenure_months": 40, "years_experience": 5.0},
        # reformed hopper: chronic [8,9,10] -> 0.30, but 40mo current lifts to 1.00
        {"n_dated_roles": 4, "completed_perm_tenures": [8, 9, 10], "current_tenure_months": 40, "years_experience": 6.0},
        # partial lift: completed [10] -> 0.50, current 20mo -> the 18-24 band (0.85)
        {"n_dated_roles": 2, "completed_perm_tenures": [10], "current_tenure_months": 20, "years_experience": 5.0},
        # short current is inconclusive: stable [36,30] -> 1.00, 2mo current leaves it 1.00
        {"n_dated_roles": 3, "completed_perm_tenures": [36, 30], "current_tenure_months": 2, "years_experience": 6.0},
        # short current never LOWERS a hopper's base either: [8,9,10] chronic 0.30, 4mo current -> 0.30
        {"n_dated_roles": 4, "completed_perm_tenures": [8, 9, 10], "current_tenure_months": 4, "years_experience": 6.0},
    ]
    assert _score_rows(rows) == [1.00, 1.00, 0.85, 1.00, 0.30]
