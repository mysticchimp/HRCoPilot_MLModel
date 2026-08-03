import pandas as pd
import pytest

from core.industry_normalization import industry_keywords, industry_present, jd_industry_requirements
from core.scoring import calculate_industry_score, calculate_total_score
from models.data_models import Company, Experience, IndustryExperience, JobRoleSchema, Skill
from models.enums import ImportanceLevel


def _jd(industries=None, industry_experience=None):
    exp = None
    if industry_experience is not None:
        exp = Experience(industry_experience=[
            IndustryExperience(industry=i, priority=p) for i, p in industry_experience
        ])
    return JobRoleSchema(
        role="HR Assistant",
        company=Company(name="Prime Focus Group"),
        responsibilities=["Support HR operations"],
        skills=[Skill(skill="HR", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
        industry=industries,
        experience=exp,
    )


def _df(sector_texts):
    return pd.DataFrame({
        "candidate_id": [f"C{i}" for i in range(len(sector_texts))],
        "sector_text": sector_texts,
    })


def test_alias_match():
    assert industry_present("HVAC", "worked at a ductwork and refrigeration firm")
    assert industry_present("manufacturing", "Senior role at ACME Factory production line")
    assert not industry_present("healthcare", "HVAC ductwork installer")


def test_no_substring_false_positive():
    # whole-word matching: "duct" must NOT match inside "product"/"conducted"
    assert not industry_present("HVAC", "conducted product reviews and introductions")
    assert industry_present("HVAC", "installed ducts and ductwork")  # plural + derivative


def test_prefix_word_not_matched():
    # healthcare "hospital" must not leak into "hospitality"
    assert not industry_present("healthcare", "hospitality and hotel operations")


def test_keywords_expand_aliases():
    kws = industry_keywords("HVAC/ductwork manufacturing")
    assert "ductwork" in kws and "refrigeration" in kws   # hvac bucket
    assert "factory" in kws                                 # manufacturing bucket


def test_requirements_dedup_and_priority():
    jd = _jd(industries=["manufacturing"],
             industry_experience=[("HVAC", ImportanceLevel.ESSENTIAL),
                                  ("manufacturing", ImportanceLevel.VALUABLE)])
    reqs = jd_industry_requirements(jd)
    industries = [i.casefold() for i, _ in reqs]
    assert industries.count("manufacturing") == 1          # deduped across both sources
    assert ("HVAC", ImportanceLevel.ESSENTIAL) in reqs


def test_industry_score_weighted_by_priority():
    jd = _jd(industry_experience=[("HVAC", ImportanceLevel.ESSENTIAL),
                                  ("retail", ImportanceLevel.VALUABLE)])
    df = calculate_industry_score(_df(["HVAC ductwork technician at a manufacturer"]), jd)
    # matches essential HVAC (weight 1.0) of total (1.0 + 0.4)
    assert df["industry_score"].iloc[0] == pytest.approx(1.0 / 1.4)


def test_missing_sector_text_is_neutral():
    jd = _jd(industry_experience=[("HVAC", ImportanceLevel.ESSENTIAL)])
    df = calculate_industry_score(_df([""]), jd)
    assert df["industry_score"].iloc[0] == 0.5


def test_no_industry_requirement_scores_all_neutral():
    df = calculate_industry_score(_df(["HVAC ductwork"]), _jd())
    assert (df["industry_score"] == 0.5).all()


def test_total_score_activates_industry_only_with_requirement():
    base = pd.DataFrame({
        "candidate_id": ["A", "B"],
        "title_score": [0.5, 0.5],
        "skill_score": [0.5, 0.5],
        "similarity_score": [0.5, 0.5],
        "industry_score": [1.0, 0.0],
    })
    weights = {"title_score": 0.25, "skill_score": 0.25, "similarity_score": 0.45, "industry_score": 0.2}

    no_ind = calculate_total_score(base.copy(), _jd(), weights=weights)
    assert no_ind["total_score"].iloc[0] == no_ind["total_score"].iloc[1]

    with_ind = calculate_total_score(base.copy(), _jd(industries=["manufacturing"]), weights=weights)
    assert with_ind["total_score"].iloc[0] > with_ind["total_score"].iloc[1]
