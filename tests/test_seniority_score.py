import pandas as pd
import pytest

from core.scoring import calculate_seniority_score, calculate_total_score
from models.data_models import Company, Experience, JobRoleSchema, Skill
from models.enums import ImportanceLevel


def _jd(level=None):
    return JobRoleSchema(
        role="HR Manager",
        company=Company(name="Prime Focus Group"),
        responsibilities=["Lead the HR function"],
        skills=[Skill(skill="HR", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
        experience=Experience(level=level) if level else None,
    )


def _df(seniorities):
    return pd.DataFrame({
        "candidate_id": [f"C{i}" for i in range(len(seniorities))],
        "seniority": seniorities,
    })


def test_exact_level_match_scores_one():
    df = calculate_seniority_score(_df(["senior"]), _jd("senior"))
    assert df["seniority_score"].iloc[0] == 1.0


def test_under_qualification_penalized_more_than_over():
    # JD wants 'senior' (rank 2); 'mid' is one below, 'executive' is one above.
    df = calculate_seniority_score(_df(["mid", "executive"]), _jd("senior"))
    under, over = df["seniority_score"].tolist()
    assert under == pytest.approx(0.6)   # 1 - 0.40 (under penalty)
    assert over == pytest.approx(0.75)   # 1 - 0.25 (over penalty)
    assert under < over


def test_large_gap_clamps_to_zero():
    # entry (0) vs c_level (4): 1 - 0.40 * 4 < 0 -> clamped.
    df = calculate_seniority_score(_df(["entry"]), _jd("c_level"))
    assert df["seniority_score"].iloc[0] == 0.0


def test_missing_candidate_seniority_is_neutral():
    df = calculate_seniority_score(_df([None]), _jd("senior"))
    assert df["seniority_score"].iloc[0] == 0.5


def test_jd_without_level_scores_all_neutral():
    df = calculate_seniority_score(_df(["entry", "c_level"]), _jd(None))
    assert (df["seniority_score"] == 0.5).all()


def test_total_score_activates_seniority_only_when_jd_specifies_level():
    base = pd.DataFrame({
        "candidate_id": ["A", "B"],
        "title_score": [0.5, 0.5],
        "skill_score": [0.5, 0.5],
        "similarity_score": [0.5, 0.5],
        "seniority_score": [1.0, 0.0],
    })
    weights = {"title_score": 0.25, "skill_score": 0.25, "similarity_score": 0.45, "seniority_score": 0.2}

    # No required level -> component excluded -> A and B tie.
    no_level = calculate_total_score(base.copy(), _jd(None), weights=weights)
    assert no_level["total_score"].iloc[0] == no_level["total_score"].iloc[1]

    # Required level -> component active -> A (1.0) outranks B (0.0).
    with_level = calculate_total_score(base.copy(), _jd("senior"), weights=weights)
    assert with_level["total_score"].iloc[0] > with_level["total_score"].iloc[1]


def test_total_score_missing_weight_defaults_to_zero():
    # A custom weights dict that omits seniority_score must not raise, even when
    # the JD specifies a level (component silently gets weight 0).
    base = pd.DataFrame({
        "candidate_id": ["A"],
        "title_score": [0.5],
        "skill_score": [0.5],
        "similarity_score": [0.5],
        "seniority_score": [1.0],
    })
    weights = {"title_score": 0.25, "skill_score": 0.25, "similarity_score": 0.45}
    scored = calculate_total_score(base.copy(), _jd("senior"), weights=weights)
    assert scored["total_score"].iloc[0] == pytest.approx(0.5)
