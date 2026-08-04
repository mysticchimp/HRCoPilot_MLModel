"""API-level unit test for POST /score (JD extraction mocked)."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api.main as main
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
    with patch.object(main, "_models_ready", True), patch.object(
        main, "_embedding_model", object()
    ), patch.object(main, "_startup_rss_mb", 900.0):
        client = TestClient(app)
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models_ready"] is True
    assert body["startup_process_rss_mb"] == 900.0
    assert "score_max_candidates" in body


def test_score_endpoint_returns_swipe_cards():
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
    fake_cards = [
        {
            "candidate_id": "db-1",
            "rank": 1,
            "total_score": 0.7,
            "component_breakdown": {},
            "matched_signals": [],
            "reasoning": "ok",
        },
        {
            "candidate_id": "db-2",
            "rank": 2,
            "total_score": 0.4,
            "component_breakdown": {},
            "matched_signals": [],
            "reasoning": "ok",
        },
    ]
    with patch.object(main, "_models_ready", True), patch.object(
        main, "_embedding_model", MagicMock()
    ), patch.object(main, "score_candidates", return_value=fake_cards):
        client = TestClient(app)
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


def test_score_rejects_oversized_batch():
    with patch.object(main, "_models_ready", True), patch.object(
        main, "_embedding_model", object()
    ), patch.object(main, "_SCORE_MAX_CANDIDATES", 2):
        client = TestClient(app)
        resp = client.post(
            "/score",
            json={
                "jd_text": "HR role",
                "candidates": [
                    {"candidate_id": f"c{i}", "raw_profile": SAMPLE_PROFILE}
                    for i in range(3)
                ],
            },
        )
    assert resp.status_code == 422
    assert "batch too large" in resp.json()["detail"]
