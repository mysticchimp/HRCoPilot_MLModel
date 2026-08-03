import pandas as pd
import pytest

from core.language_normalization import (
    jd_language_requirements,
    normalize_candidate_languages,
    normalize_language,
)
from core.scoring import calculate_language_score, calculate_total_score
from models.data_models import Company, JobRoleSchema, LanguageProficiency, Skill
from models.enums import ImportanceLevel


def _jd(languages=None):
    """languages: list of (language, level, priority)."""
    language_proficiency = None
    if languages is not None:
        language_proficiency = [
            LanguageProficiency(language=lang, level=level, priority=priority)
            for lang, level, priority in languages
        ]
    return JobRoleSchema(
        role="HR Assistant",
        company=Company(name="Prime Focus Group"),
        responsibilities=["Support HR operations"],
        skills=[Skill(skill="HR", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
        language_proficiency=language_proficiency,
    )


def _df(language_lists):
    return pd.DataFrame({
        "candidate_id": [f"C{i}" for i in range(len(language_lists))],
        "languages": language_lists,
    })


def test_normalize_aliases_and_qualifiers():
    # Tagalog is an alias of Filipino; qualifiers are stripped
    assert normalize_language("Tagalog") == normalize_language("Filipino") == "filipino"
    assert normalize_language("English (United States)") == "english"
    assert normalize_language("Filipino/Tagalog") == "filipino"
    # mutually-intelligible-but-distinct languages stay separate
    assert normalize_language("Hindi") != normalize_language("Urdu")
    # unknown languages fall back to their own casefolded root (still matchable)
    assert normalize_language("Malayalam") == "malayalam"


def test_normalize_candidate_languages_dedup_and_filter():
    cand = normalize_candidate_languages(["English", "Tagalog", "Filipino", None, 5])
    assert cand == frozenset({"english", "filipino"})
    assert normalize_candidate_languages(None) == frozenset()


def test_requirements_dedup_and_priority():
    jd = _jd([("English", "fluent", ImportanceLevel.ESSENTIAL),
              ("Filipino", "conversational", ImportanceLevel.VALUABLE),
              ("Tagalog", "conversational", ImportanceLevel.IMPORTANT)])  # dupe of Filipino
    reqs = jd_language_requirements(jd)
    langs = [lang for lang, _ in reqs]
    assert langs.count("filipino") == 1                       # Tagalog deduped into Filipino
    assert ("english", ImportanceLevel.ESSENTIAL) in reqs


def test_language_score_weighted_by_priority():
    jd = _jd([("English", "fluent", ImportanceLevel.ESSENTIAL),
              ("Arabic", "professional", ImportanceLevel.VALUABLE)])
    df = calculate_language_score(_df([["English", "Arabic"], ["English"], ["Arabic"]]), jd)
    # essential 1.0 + valuable 0.4 = total 1.4
    assert df["language_score"].iloc[0] == pytest.approx(1.4 / 1.4)   # both
    assert df["language_score"].iloc[1] == pytest.approx(1.0 / 1.4)   # english only
    assert df["language_score"].iloc[2] == pytest.approx(0.4 / 1.4)   # arabic only


def test_alias_match_in_score():
    jd = _jd([("Filipino", "conversational", ImportanceLevel.ESSENTIAL)])
    df = calculate_language_score(_df([["Tagalog"]]), jd)
    assert df["language_score"].iloc[0] == pytest.approx(1.0)


def test_missing_languages_is_neutral():
    jd = _jd([("English", "fluent", ImportanceLevel.ESSENTIAL)])
    df = calculate_language_score(_df([[]]), jd)
    assert df["language_score"].iloc[0] == 0.5


def test_no_language_requirement_scores_all_neutral():
    df = calculate_language_score(_df([["English"], ["Arabic"]]), _jd())
    assert (df["language_score"] == 0.5).all()


def test_total_score_activates_language_only_with_requirement():
    base = pd.DataFrame({
        "candidate_id": ["A", "B"],
        "title_score": [0.5, 0.5],
        "skill_score": [0.5, 0.5],
        "similarity_score": [0.5, 0.5],
        "language_score": [1.0, 0.0],
    })
    weights = {"title_score": 0.25, "skill_score": 0.25, "similarity_score": 0.45, "language_score": 0.2}

    no_lang = calculate_total_score(base.copy(), _jd(), weights=weights)
    assert no_lang["total_score"].iloc[0] == no_lang["total_score"].iloc[1]

    with_lang = calculate_total_score(
        base.copy(), _jd([("English", "fluent", ImportanceLevel.ESSENTIAL)]), weights=weights)
    assert with_lang["total_score"].iloc[0] > with_lang["total_score"].iloc[1]
