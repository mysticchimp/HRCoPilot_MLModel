import pytest
from sentence_transformers import SentenceTransformer

from core.adapters.linkedin_adapter import LinkedInAdapter
from core.adapters.resume_adapter import ResumeAdapter
from core.data import profiles_to_dataframe
from core.embedding import embed_profiles
from core.filtering import filter_by_job_title
from core.scoring import (
    calculate_qualification_score,
    calculate_similarity_score,
    calculate_skill_score,
    calculate_total_score,
)
from models.data_models import Company, JobRoleSchema, Skill
from models.enums import ImportanceLevel

RESUME = "./data/resume_data.csv"
LINKEDIN = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"

SCORE_COLS = ["title_score", "skill_score", "qualification_score", "similarity_score", "total_score"]


@pytest.fixture(scope="module")
def model():
    return SentenceTransformer("all-mpnet-base-v2")


def _jd():
    return JobRoleSchema(
        role="HR Assistant",
        company=Company(name="Prime Focus Group"),
        responsibilities=["Support payroll and HR administration", "Coordinate onboarding"],
        skills=[
            Skill(skill="Payroll", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None),
            Skill(skill="Recruitment", priority=ImportanceLevel.IMPORTANT, proficiency_level=None),
        ],
    )


def _score(profiles, model, mode):
    embed_profiles(profiles, model)
    df = profiles_to_dataframe(profiles)
    jd = _jd()
    df = filter_by_job_title(df, jd.role, threshold=0.0, model=(model if mode != "fuzzy" else None), mode=mode, hard=False)
    df = calculate_skill_score(df, jd)
    df = calculate_qualification_score(df, jd)
    df = calculate_similarity_score(df, jd, model)
    df = calculate_total_score(df, jd)
    return df


def test_wiring_resume_end_to_end(model):
    profiles = ResumeAdapter().to_profiles(RESUME)[:25]
    df = _score(profiles, model, "fuzzy")
    assert len(df) == 25
    for col in SCORE_COLS:
        assert df[col].notna().all()
    assert df["title_score"].between(0, 1).all()


def test_wiring_linkedin_end_to_end(model):
    profiles = LinkedInAdapter().to_profiles(LINKEDIN)[:25]
    df = _score(profiles, model, "hybrid")
    assert len(df) == 25  # soft gate keeps everyone
    for col in SCORE_COLS:
        assert df[col].notna().all()
    assert df["title_score"].between(0, 1).all()


def test_hybrid_title_gate_recovers_more_than_fuzzy(model):
    # the hybrid gate must keep at least as many candidates as the brittle fuzzy gate
    profiles = LinkedInAdapter().to_profiles(LINKEDIN)
    embed_profiles(profiles[:1], model)  # warm; not needed for title scoring
    df = profiles_to_dataframe(profiles)
    fuzzy_kept = filter_by_job_title(df, "HR Assistant", threshold=0.4, mode="fuzzy", hard=True)
    hybrid_kept = filter_by_job_title(df, "HR Assistant", threshold=0.4, model=model, mode="hybrid", hard=True)
    assert len(hybrid_kept) >= len(fuzzy_kept)
