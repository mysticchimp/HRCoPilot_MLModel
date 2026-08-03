"""API-level unit test for POST /score (JD extraction mocked)."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from models.data_models import Company, JobRoleSchema, Skill
from models.enums import ImportanceLevel

SAMPLE_PROFILE = {
    "publicIdentifier": "sample-hr",
    "linkedinUrl": "https://www.linkedin.com/in/sample-hr",
    "firstName": "Ada",
    "lastName": "HR",
    "fullName": "Ada HR",
    "headline": "HR Assistant",
    "about": "HR assistant with onboarding and payroll experience.",
    "location": {
        "linkedinText": "Dubai, United Arab Emirates",
        "parsed": {"city": "Dubai", "country": "United Arab Emirates", "countryCode": "AE", "text": "Dubai, UAE"},
    },
    "experience": [
        {
            "position": "HR Assistant",
            "companyName": "Prime AC",
            "duration": "2 yrs",
            "employmentType": "Full-time",
            "description": "Onboarding and employee relations.",
            "startDate": {"text": "Jan 2024"},
            "endDate": {"text": "Present"},
        }
    ],
    "education": [
        {"degree": "BBA", "fieldOfStudy": "HR", "schoolName": "UAEU", "endDate": {"year": 2023}}
    ],
    "skills": [{"name": "Onboarding"}, {"name": "Payroll"}, {"name": "HR Policies"}],
    "languages": [{"name": "English", "proficiency": "Fluent"}],
}


def _fake_jd():
    return JobRoleSchema(
        role="HR Assistant",
        company=Company(name="Prime Focus"),
        responsibilities=["Support HR operations", "Onboarding"],
        skills=[
            Skill(skill="Onboarding", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None),
            Skill(skill="Payroll", priority=ImportanceLevel.IMPORTANT, proficiency_level=None),
        ],
    )


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_score_endpoint_returns_swipe_cards():
    client = TestClient(app)
    body = {
        "jd_text": "HR Assistant needed in Dubai. Onboarding and payroll.",
        "candidates": [
            {"candidate_id": "db-1", "raw_profile": SAMPLE_PROFILE},
            {
                "candidate_id": "db-2",
                "raw_profile": {
                    **SAMPLE_PROFILE,
                    "publicIdentifier": "other",
                    "headline": "Software Engineer",
                    "experience": [
                        {
                            "position": "Software Engineer",
                            "companyName": "TechCo",
                            "duration": "3 yrs",
                            "description": "Built APIs.",
                            "startDate": {"text": "Jan 2023"},
                            "endDate": {"text": "Present"},
                        }
                    ],
                    "skills": [{"name": "Python"}],
                },
            },
        ],
    }

    with patch("api.main.process_jd", return_value=_fake_jd()):
        # Keep embeddings cheap/offline by stubbing the heavy pipeline pieces would
        # under-test the adapter wiring; instead run real pipeline but skip network.
        resp = client.post("/score", json=body)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["count"] == 2
    ids = {c["candidate_id"] for c in data["cards"]}
    assert ids == {"db-1", "db-2"}
    for card in data["cards"]:
        assert card["total_score"] is not None
        assert "component_breakdown" in card
        assert "matched_signals" in card
        assert "reasoning" in card
        assert "rank" in card
