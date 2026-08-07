"""Tests for /narrate: compression, failure isolation, response contract."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.main import app
from core.narrate import (
    NarrativeResult,
    compress_profile,
    estimate_tokens,
    narrate_candidates,
    narrate_one,
)
from models.data_models import Company, JobRoleSchema, Skill
from models.enums import ImportanceLevel

# Fat Apify-shaped profile (extra noise fields that must be dropped by compression).
FAT_PROFILE = {
    "publicIdentifier": "sample-hr",
    "linkedinUrl": "https://www.linkedin.com/in/sample-hr",
    "firstName": "Ada",
    "lastName": "HR",
    "fullName": "Ada HR",
    "headline": "HR Assistant | UAE Labour Law | WPS",
    "about": ("A" * 800) + " trailing noise that should be truncated.",
    "location": {
        "linkedinText": "Dubai, United Arab Emirates",
        "parsed": {"city": "Dubai", "country": "United Arab Emirates"},
    },
    "experience": [
        {
            "position": f"Role {i}",
            "companyName": f"Co {i}",
            "duration": "1 yr",
            "description": ("Long description " * 40),
            "startDate": {"text": f"Jan {2020 + i}"},
            "endDate": {"text": "Present" if i == 0 else f"Dec {2020 + i}"},
            "extraNoise": {"foo": "bar" * 50},
        }
        for i in range(8)
    ],
    "education": [
        {"degree": "BBA", "fieldOfStudy": "HR", "schoolName": "UAEU", "endDate": {"year": 2019}}
    ],
    "skills": [{"name": f"Skill{i}"} for i in range(60)],
    "languages": [{"name": "English", "proficiency": "Fluent"}],
    # Noise that must never reach Claude:
    "updates": [{"text": "x" * 5000} for _ in range(20)],
    "peopleAlsoViewed": [{"url": f"https://li/{i}"} for i in range(50)],
    "recommendations": [{"text": "y" * 2000} for _ in range(10)],
}


MINIMAL_JD = {
    "role": "HR Assistant",
    "company": {"name": "Prime Focus"},
    "responsibilities": ["Visa coordination", "Payroll support"],
    "skills": [
        {"skill": "WPS", "priority": "essential"},
        {"skill": "UAE Labour Law", "priority": "essential"},
    ],
}


def test_compress_profile_drops_noise_and_caps_fields():
    compressed = compress_profile(FAT_PROFILE)
    assert "updates" not in compressed
    assert "peopleAlsoViewed" not in compressed
    assert "recommendations" not in compressed
    assert "linkedinUrl" not in compressed

    assert compressed["name"] == "Ada HR"
    assert compressed["headline"]
    assert compressed["location"] == "Dubai, United Arab Emirates"
    assert len(compressed["about"]) <= 500
    assert compressed["about"].endswith("…")
    assert len(compressed["experience"]) == 3
    for exp in compressed["experience"]:
        assert len(exp.get("description") or "") <= 220
    assert len(compressed["skills"]) <= 40
    assert compressed["education"]
    assert compressed["languages"]

    # Token estimate must be far below the fat raw profile.
    raw_tokens = estimate_tokens(FAT_PROFILE)
    compact_tokens = estimate_tokens(compressed)
    assert compact_tokens < raw_tokens / 5
    assert compact_tokens < 1500  # comfortably under a full Apify dump


def test_compress_profile_empty_safe():
    assert compress_profile({}) == {}
    assert compress_profile(None) == {}


def test_narrate_one_success_and_uses_cache_flag():
    provider = MagicMock()
    provider.generate_structured.return_value = NarrativeResult(
        summary="Ada is an HR professional based in Dubai.",
        assessment="Strong WPS fit, but limited manufacturing HR evidence.",
    )
    out = narrate_one(
        candidate_id="c1",
        raw_profile=FAT_PROFILE,
        component_breakdown={"skill_score": 0.8, "industry_score": 0.2},
        matched_signals=["WPS", "UAE"],
        cached_system="SYSTEM+JD",
        provider=provider,
    )
    assert out["candidate_id"] == "c1"
    assert "summary" in out and "assessment" in out
    assert "error" not in out
    kwargs = provider.generate_structured.call_args.kwargs
    assert kwargs["cache_system"] is True
    assert kwargs["system"] == "SYSTEM+JD"
    assert "Ada HR" in kwargs["prompt"] or "Ada" in kwargs["prompt"]
    # Compressed prompt must not include Apify noise blobs
    assert "peopleAlsoViewed" not in kwargs["prompt"]
    assert "recommendations" not in kwargs["prompt"]


def test_narrate_one_failure_returns_error_marker():
    provider = MagicMock()
    provider.generate_structured.side_effect = TimeoutError("claude timeout")
    out = narrate_one(
        candidate_id="c-fail",
        raw_profile=FAT_PROFILE,
        component_breakdown={},
        matched_signals=[],
        cached_system="SYS",
        provider=provider,
    )
    assert out == {"candidate_id": "c-fail", "error": "claude timeout"}


def test_narrate_candidates_isolates_failures():
    """One Claude failure must not break the rest of the batch."""
    provider = MagicMock()

    def _side_effect(**kwargs):
        prompt = kwargs.get("prompt") or ""
        if "Bob" in prompt:
            raise RuntimeError("malformed response")
        return NarrativeResult(
            summary="A solid HR operator.",
            assessment="Good skill fit with some industry gaps.",
        )

    provider.generate_structured.side_effect = _side_effect

    rows = narrate_candidates(
        MINIMAL_JD,
        [
            {
                "candidate_id": "ok-1",
                "raw_profile": {**FAT_PROFILE, "fullName": "Ada HR", "firstName": "Ada"},
                "component_breakdown": {"skill_score": 0.9},
                "matched_signals": ["WPS"],
            },
            {
                "candidate_id": "bad-1",
                "raw_profile": {
                    **FAT_PROFILE,
                    "fullName": "Bob Fail",
                    "firstName": "Bob",
                    "lastName": "Fail",
                },
                "component_breakdown": {},
                "matched_signals": [],
            },
            {
                "candidate_id": "ok-2",
                "raw_profile": {
                    **FAT_PROFILE,
                    "fullName": "Cara OK",
                    "firstName": "Cara",
                    "lastName": "OK",
                },
                "component_breakdown": {"title_score": 0.7},
                "matched_signals": ["HR Assistant"],
            },
        ],
        provider=provider,
        concurrency=3,
    )

    assert [r["candidate_id"] for r in rows] == ["ok-1", "bad-1", "ok-2"]
    assert "summary" in rows[0] and "assessment" in rows[0]
    assert rows[1]["error"]
    assert "summary" not in rows[1]
    assert "summary" in rows[2] and "assessment" in rows[2]


def test_narrate_endpoint_response_shape():
    fake_rows = [
        {
            "candidate_id": "db-1",
            "summary": "Ada is an HR assistant in Dubai.",
            "assessment": "Strong WPS match, weak industry overlap.",
        },
        {"candidate_id": "db-2", "error": "timeout"},
    ]
    with patch.object(main, "narrate_candidates", return_value=fake_rows):
        client = TestClient(app)
        resp = client.post(
            "/narrate",
            json={
                "jd_parsed_or_text": MINIMAL_JD,
                "candidates": [
                    {
                        "candidate_id": "db-1",
                        "raw_profile": FAT_PROFILE,
                        "component_breakdown": {"skill_score": 0.8},
                        "matched_signals": ["WPS"],
                    },
                    {
                        "candidate_id": "db-2",
                        "raw_profile": FAT_PROFILE,
                        "component_breakdown": {},
                        "matched_signals": [],
                    },
                ],
            },
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["count"] == 2
    assert len(data["narratives"]) == 2
    first, second = data["narratives"]
    assert first["candidate_id"] == "db-1"
    assert first["summary"]
    assert first["assessment"]
    assert first.get("error") is None
    assert second["candidate_id"] == "db-2"
    assert second["error"] == "timeout"
    assert second.get("summary") is None


def test_narrate_endpoint_accepts_jd_text_string():
    with patch.object(
        main,
        "narrate_candidates",
        return_value=[
            {
                "candidate_id": "c1",
                "summary": "Summary.",
                "assessment": "Assessment.",
            }
        ],
    ) as mocked:
        client = TestClient(app)
        resp = client.post(
            "/narrate",
            json={
                "jd_parsed_or_text": "HR Assistant needed in Dubai with WPS experience.",
                "candidates": [
                    {"candidate_id": "c1", "raw_profile": FAT_PROFILE},
                ],
            },
        )
    assert resp.status_code == 200, resp.text
    assert mocked.call_args.args[0].startswith("HR Assistant")


def test_narrate_endpoint_rejects_empty_candidates():
    client = TestClient(app)
    resp = client.post(
        "/narrate",
        json={"jd_parsed_or_text": MINIMAL_JD, "candidates": []},
    )
    assert resp.status_code == 422


def test_narrate_endpoint_rejects_oversized_batch():
    with patch.object(main, "_NARRATE_MAX_CANDIDATES", 2):
        client = TestClient(app)
        resp = client.post(
            "/narrate",
            json={
                "jd_parsed_or_text": "JD text",
                "candidates": [
                    {"candidate_id": f"c{i}", "raw_profile": {}} for i in range(3)
                ],
            },
        )
    assert resp.status_code == 422
    assert "batch too large" in resp.json()["detail"]


def test_compact_jd_from_schema_object():
    from core.narrate import compact_jd_block

    block = compact_jd_block(MINIMAL_JD)
    assert "HR Assistant" in block
    assert "WPS" in block

    text_block = compact_jd_block("Plain JD text here.")
    assert text_block == "Plain JD text here."
