import json

import pytest
from pydantic import ValidationError

from core.swipe import COMPONENT_KEYS, SwipeEvent, append_swipe, build_card, build_matched_signals
from models.candidate import CandidateLanguage, CandidateLocation, CandidatePosition, CandidateProfile
from models.data_models import JobRoleSchema

JD_STORE = "jd/parsed/hr_assistant_prime_ac.json"


@pytest.fixture(scope="module")
def jd():
    with open(JD_STORE) as fh:
        return JobRoleSchema.model_validate(json.load(fh))


def _profile(**kw):
    base = dict(
        candidate_id="test-cand",
        job_title="HR Assistant",
        skills=["Payroll", "Nonexistent Skill XYZ"],
        seniority="entry",
        languages=[CandidateLanguage(language="Tagalog")],
        location=CandidateLocation(text="Dubai, UAE", city="Dubai", country="United Arab Emirates"),
        positions=[CandidatePosition(title="HR Assistant", company="Acme", is_current=True, tenure_months=10)],
        raw={"linkedinUrl": "https://linkedin.com/in/test", "fullName": "Test Candidate"},
    )
    base.update(kw)
    return CandidateProfile(**base)


def _row(**kw):
    row = {"pipeline_rank": 4, "total_score": 0.71, "sector_text": "", "data_completeness_level": "rich"}
    for k in COMPONENT_KEYS:
        row[f"{k}_score"] = 0.5
    row.update(kw)
    return row


def test_build_card_contract(jd):
    card = build_card(_profile(), _row(), jd)
    assert set(card) == {
        "candidate_id", "name", "title", "current_company", "location", "rank",
        "total_score", "component_breakdown", "matched_signals", "flags", "reasoning", "linkedin_url",
    }
    assert card["name"] == "Test Candidate"
    assert card["current_company"] == "Acme"
    assert card["rank"] == 4 and card["total_score"] == 0.71
    assert set(card["component_breakdown"]) == set(COMPONENT_KEYS)
    assert set(card["flags"]) == {"flight_risk", "industrial_sector", "workforce_language", "data_completeness"}
    assert card["linkedin_url"] == "https://linkedin.com/in/test"
    assert isinstance(card["reasoning"], str) and card["reasoning"].endswith(".")


def test_matched_signals_are_grounded(jd):
    signals = build_matched_signals(_profile(), _row(), jd)
    # a bogus candidate skill must NEVER surface (no hallucinated boilerplate)
    assert "Nonexistent Skill XYZ" not in signals
    assert len(signals) <= 8
    # Tagalog is now a JD nice-to-have and the candidate lists it -> grounded language match
    assert any("tagalog" in s.lower() or "filipino" in s.lower() for s in signals)


def test_flags_flight_risk_and_workforce_language(jd):
    hopper = build_card(_profile(), _row(attrition_score=0.30), jd)
    assert hopper["flags"]["flight_risk"] is True
    stable = build_card(_profile(), _row(attrition_score=0.85), jd)
    assert stable["flags"]["flight_risk"] is False
    # workforce_language is grounded in the STRUCTURED language list, not names
    assert build_card(_profile(), _row(), jd)["flags"]["workforce_language"] is True
    no_tagalog = _profile(languages=[CandidateLanguage(language="Hindi")])
    assert build_card(no_tagalog, _row(), jd)["flags"]["workforce_language"] is False


def test_flags_completeness_passthrough(jd):
    card = build_card(_profile(), _row(data_completeness_level="low"), jd)
    assert card["flags"]["data_completeness"] == "low"
    assert "screening" in card["reasoning"].lower()


def test_swipe_event_schema_and_capture(tmp_path):
    ev = SwipeEvent(recruiter_id="r1", candidate_id="test-cand", jd_id="hr_assistant",
                    decision="right", ts="2026-07-29T10:00:00Z", rank_shown=4)
    with pytest.raises(ValidationError):
        SwipeEvent(recruiter_id="r1", candidate_id="c", jd_id="j",
                   decision="sideways", ts="t", rank_shown=1)  # invalid decision
    path = tmp_path / "swipes.jsonl"
    append_swipe(ev, str(path))
    append_swipe(ev, str(path))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["decision"] == "right"
