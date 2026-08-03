import pandas as pd
import pytest

from core.scoring import calculate_experience_score, calculate_total_score
from models.data_models import Company, Experience, ExperienceRange, JobRoleSchema, Skill
from models.enums import ImportanceLevel


def _jd(years_min=None, years_max=None, has_range=True):
    exp = Experience(years_total=ExperienceRange(min=years_min, max=years_max)) if has_range else None
    return JobRoleSchema(
        role="HR Manager",
        company=Company(name="Prime Focus Group"),
        responsibilities=["Lead the HR function"],
        skills=[Skill(skill="HR", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
        experience=exp,
    )


def _df(years):
    return pd.DataFrame({
        "candidate_id": [f"C{i}" for i in range(len(years))],
        "years_experience": years,
    })


def test_within_range_scores_one():
    df = calculate_experience_score(_df([4.0]), _jd(3, 6))
    assert df["experience_score"].iloc[0] == 1.0


def test_under_minimum_penalized_per_year():
    # min 5, candidate 3 -> short 2 years * 0.15 -> 0.70
    df = calculate_experience_score(_df([3.0]), _jd(5, 8))
    assert df["experience_score"].iloc[0] == pytest.approx(0.70)


def test_over_maximum_penalized_more_gently():
    # max 4, candidate 6 -> over 2 years * 0.05 -> 0.90
    df = calculate_experience_score(_df([6.0]), _jd(2, 4))
    assert df["experience_score"].iloc[0] == pytest.approx(0.90)


def test_shortfall_hurts_more_than_overage():
    under = calculate_experience_score(_df([3.0]), _jd(5, 8))["experience_score"].iloc[0]
    over = calculate_experience_score(_df([6.0]), _jd(2, 4))["experience_score"].iloc[0]
    assert under < over


def test_open_ended_minimum_only():
    # min 5, no max -> 10 years is a fit -> 1.0
    df = calculate_experience_score(_df([10.0]), _jd(5, None))
    assert df["experience_score"].iloc[0] == 1.0


def test_missing_years_is_neutral():
    df = calculate_experience_score(_df([None]), _jd(3, 6))
    assert df["experience_score"].iloc[0] == 0.5


def test_no_experience_block_scores_all_neutral():
    df = calculate_experience_score(_df([2.0, 20.0]), _jd(has_range=False))
    assert (df["experience_score"] == 0.5).all()


def test_empty_range_scores_all_neutral():
    # Experience present but both min and max None -> no usable requirement.
    df = calculate_experience_score(_df([2.0, 20.0]), _jd(None, None))
    assert (df["experience_score"] == 0.5).all()


def test_total_score_activates_experience_only_with_range():
    base = pd.DataFrame({
        "candidate_id": ["A", "B"],
        "title_score": [0.5, 0.5],
        "skill_score": [0.5, 0.5],
        "similarity_score": [0.5, 0.5],
        "experience_score": [1.0, 0.0],
    })
    weights = {"title_score": 0.25, "skill_score": 0.25, "similarity_score": 0.45, "experience_score": 0.2}

    no_range = calculate_total_score(base.copy(), _jd(has_range=False), weights=weights)
    assert no_range["total_score"].iloc[0] == no_range["total_score"].iloc[1]

    with_range = calculate_total_score(base.copy(), _jd(2, 4), weights=weights)
    assert with_range["total_score"].iloc[0] > with_range["total_score"].iloc[1]
