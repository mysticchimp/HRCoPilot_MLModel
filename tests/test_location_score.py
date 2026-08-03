import pandas as pd
import pytest

from core.location_normalization import (
    jd_location_requirements,
    normalize_city,
    normalize_country,
    normalize_country_code,
)
from core.scoring import calculate_location_score, calculate_total_score
from models.data_models import Company, JobRoleSchema, Location, Skill
from models.enums import ImportanceLevel


def _jd(cities=None, countries=None):
    location = None
    if cities is not None or countries is not None:
        location = Location(cities=cities, countries=countries)
    return JobRoleSchema(
        role="HR Assistant",
        company=Company(name="Prime Focus Group"),
        responsibilities=["Support HR operations"],
        skills=[Skill(skill="HR", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
        location=location,
    )


def _df(rows):
    """rows: list of (city, country, country_code)."""
    return pd.DataFrame({
        "candidate_id": [f"C{i}" for i in range(len(rows))],
        "location_city": [r[0] for r in rows],
        "location_country": [r[1] for r in rows],
        "location_country_code": [r[2] for r in rows],
    })


def test_normalize_country_and_code_aliases():
    assert normalize_country("UAE") == normalize_country("United Arab Emirates") == "united arab emirates"
    assert normalize_country_code("AE") == "united arab emirates"
    assert normalize_country_code("US") == ""            # unknown code -> empty
    assert normalize_city("AbuDhabi") == normalize_city("Abu Dhabi") == "abu dhabi"


def test_jd_requirements_drop_empty_city():
    cities, countries = jd_location_requirements(_jd(cities=["Dubai", ""], countries=["United Arab Emirates"]))
    assert cities == frozenset({"dubai"})                # blank city dropped
    assert countries == frozenset({"united arab emirates"})


def test_city_match_is_strongest():
    jd = _jd(cities=["Dubai"], countries=["United Arab Emirates"])
    df = calculate_location_score(_df([("Dubai", "United Arab Emirates", "AE")]), jd)
    assert df["location_score"].iloc[0] == pytest.approx(1.0)


def test_in_country_wrong_city_is_partial():
    jd = _jd(cities=["Dubai"], countries=["United Arab Emirates"])
    df = calculate_location_score(_df([("Sharjah", "United Arab Emirates", "AE")]), jd)
    assert df["location_score"].iloc[0] == pytest.approx(0.7)


def test_in_country_absent_city_is_full_credit():
    # right country but no city specified -> NOT penalized (benefit of the doubt,
    # the primary country requirement is met) -> equal to a city match
    jd = _jd(cities=["Dubai"], countries=["United Arab Emirates"])
    df = calculate_location_score(_df([(None, "United Arab Emirates", "AE")]), jd)
    assert df["location_score"].iloc[0] == pytest.approx(1.0)


def test_absent_city_outranks_confirmed_wrong_city():
    # an omitted city (unknown -> 1.0) should score above a confirmed different
    # city (0.7): absence of evidence is not evidence of absence
    jd = _jd(cities=["Dubai"], countries=["United Arab Emirates"])
    df = calculate_location_score(_df([
        (None, "United Arab Emirates", "AE"),        # city unspecified
        ("Sharjah", "United Arab Emirates", "AE"),   # confirmed different city
    ]), jd)
    assert df["location_score"].iloc[0] == pytest.approx(1.0)
    assert df["location_score"].iloc[1] == pytest.approx(0.7)
    assert df["location_score"].iloc[0] > df["location_score"].iloc[1]


def test_country_only_jd_gives_full_credit():
    # extraction missed the city -> country match is full credit
    jd = _jd(countries=["United Arab Emirates"])
    df = calculate_location_score(_df([("Sharjah", "United Arab Emirates", "AE")]), jd)
    assert df["location_score"].iloc[0] == pytest.approx(1.0)


def test_country_match_via_code_only():
    # candidate has no country name, only the ISO code
    jd = _jd(cities=["Dubai"], countries=["United Arab Emirates"])
    df = calculate_location_score(_df([("Sharjah", None, "AE")]), jd)
    assert df["location_score"].iloc[0] == pytest.approx(0.7)


def test_out_of_scope_is_miss():
    jd = _jd(cities=["Dubai"], countries=["United Arab Emirates"])
    df = calculate_location_score(_df([("Mumbai", "India", "IN")]), jd)
    assert df["location_score"].iloc[0] == pytest.approx(0.0)


def test_missing_location_is_neutral():
    jd = _jd(cities=["Dubai"], countries=["United Arab Emirates"])
    df = calculate_location_score(_df([(None, None, None)]), jd)
    assert df["location_score"].iloc[0] == 0.5


def test_no_location_requirement_scores_all_neutral():
    df = calculate_location_score(_df([("Dubai", "United Arab Emirates", "AE")]), _jd())
    assert (df["location_score"] == 0.5).all()


def test_total_score_activates_location_only_with_requirement():
    base = pd.DataFrame({
        "candidate_id": ["A", "B"],
        "title_score": [0.5, 0.5],
        "skill_score": [0.5, 0.5],
        "similarity_score": [0.5, 0.5],
        "location_score": [1.0, 0.0],
    })
    weights = {"title_score": 0.25, "skill_score": 0.25, "similarity_score": 0.45, "location_score": 0.2}

    no_loc = calculate_total_score(base.copy(), _jd(), weights=weights)
    assert no_loc["total_score"].iloc[0] == no_loc["total_score"].iloc[1]

    with_loc = calculate_total_score(
        base.copy(), _jd(cities=["Dubai"], countries=["United Arab Emirates"]), weights=weights)
    assert with_loc["total_score"].iloc[0] > with_loc["total_score"].iloc[1]
