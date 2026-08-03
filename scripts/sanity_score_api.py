"""Build a small nested-Apify sample payload and POST it to a local /score server.

Usage:
  # terminal 1
  COPILOT_SKIP_CLI_DOWNLOAD=1 uv run uvicorn api.main:app --host 127.0.0.1 --port 8080

  # terminal 2 (uses cached JD when ANTHROPIC_API_KEY is unset — patches process_jd)
  COPILOT_SKIP_CLI_DOWNLOAD=1 uv run python scripts/sanity_score_api.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import urllib.error
import urllib.request

from models.data_models import JobRoleSchema

BASE = os.environ.get("SCORE_API_BASE", "http://127.0.0.1:8080")
JD_STORE = "jd/parsed/hr_assistant_prime_ac.json"
JD_TEXT_PATH = "jd/sample_hr_assistant_jd.txt"


SAMPLES = [
    {
        "candidate_id": "cand-roxanna",
        "raw_profile": {
            "publicIdentifier": "roxanna-ghassemlou-007078191",
            "linkedinUrl": "https://www.linkedin.com/in/roxanna-ghassemlou-007078191",
            "firstName": "Roxanna",
            "lastName": "Ghassemlou",
            "fullName": "Roxanna Ghassemlou",
            "headline": "HR Manager | Distinct Group | Dubai",
            "about": (
                "Motivated HR professional with over 4 years of experience in Human Resources, "
                "specialising in employee relations, onboarding, and HR policies in the UAE."
            ),
            "openToWork": False,
            "verified": True,
            "location": {
                "linkedinText": "Dubai, United Arab Emirates",
                "countryCode": "AE",
                "parsed": {
                    "city": "Dubai",
                    "state": "Dubai",
                    "country": "United Arab Emirates",
                    "countryCode": "AE",
                    "text": "Dubai, United Arab Emirates",
                },
            },
            "experience": [
                {
                    "position": "HR Manager",
                    "companyName": "Distinct Group",
                    "duration": "5 mos",
                    "employmentType": "Full-time",
                    "description": "Led HR operations, onboarding, and employee relations.",
                    "startDate": {"text": "Mar 2026"},
                    "endDate": {"text": "Present"},
                },
                {
                    "position": "HR Assistant",
                    "companyName": "Prime Focus Technologies",
                    "duration": "2 yrs 3 mos",
                    "employmentType": "Full-time",
                    "description": "Supported payroll, onboarding, and manufacturing-site HR admin.",
                    "startDate": {"text": "Jan 2024"},
                    "endDate": {"text": "Mar 2026"},
                },
            ],
            "education": [
                {
                    "degree": "Master of Science - MSc",
                    "fieldOfStudy": "Occupational and Organisational Psychology",
                    "schoolName": "University of Sussex",
                    "endDate": {"year": 2021},
                }
            ],
            "skills": [
                {"name": "Onboarding"},
                {"name": "Payroll"},
                {"name": "HR Policies"},
                {"name": "Employee Relations"},
                {"name": "People Management"},
            ],
            "languages": [
                {"name": "English", "proficiency": "Native or bilingual proficiency"},
                {"name": "Tagalog", "proficiency": "Professional working proficiency"},
            ],
        },
    },
    {
        "candidate_id": "cand-entry-hr",
        "raw_profile": {
            "publicIdentifier": "sample-hr-assistant",
            "linkedinUrl": "https://www.linkedin.com/in/sample-hr-assistant",
            "firstName": "Aisha",
            "lastName": "Khan",
            "fullName": "Aisha Khan",
            "headline": "HR Assistant | Dubai",
            "about": "HR Assistant focused on recruitment coordination and onboarding.",
            "location": {
                "linkedinText": "Dubai, United Arab Emirates",
                "parsed": {
                    "city": "Dubai",
                    "country": "United Arab Emirates",
                    "countryCode": "AE",
                    "text": "Dubai, United Arab Emirates",
                },
            },
            "experience": [
                {
                    "position": "HR Assistant",
                    "companyName": "Gulf Manufacturing LLC",
                    "duration": "1 yr 6 mos",
                    "employmentType": "Full-time",
                    "description": "Coordinated interviews, maintained employee files, supported onboarding.",
                    "startDate": {"text": "Jan 2025"},
                    "endDate": {"text": "Present"},
                }
            ],
            "education": [
                {
                    "degree": "Bachelor of Business Administration",
                    "fieldOfStudy": "Human Resources",
                    "schoolName": "University of Dubai",
                    "endDate": {"year": 2024},
                }
            ],
            "skills": [
                {"name": "Onboarding"},
                {"name": "Recruitment"},
                {"name": "MS Office"},
                {"name": "HR Administration"},
            ],
            "languages": [
                {"name": "English", "proficiency": "Full professional proficiency"},
                {"name": "Arabic", "proficiency": "Native or bilingual proficiency"},
            ],
        },
    },
    {
        "candidate_id": "cand-unrelated",
        "raw_profile": {
            "publicIdentifier": "sample-software-eng",
            "linkedinUrl": "https://www.linkedin.com/in/sample-software-eng",
            "firstName": "Dev",
            "lastName": "Patel",
            "fullName": "Dev Patel",
            "headline": "Software Engineer | Backend",
            "about": "Backend engineer building APIs in Python and Go.",
            "location": {
                "linkedinText": "Bengaluru, India",
                "parsed": {
                    "city": "Bengaluru",
                    "country": "India",
                    "countryCode": "IN",
                    "text": "Bengaluru, India",
                },
            },
            "experience": [
                {
                    "position": "Software Engineer",
                    "companyName": "Acme Tech",
                    "duration": "3 yrs",
                    "employmentType": "Full-time",
                    "description": "Built microservices and data pipelines.",
                    "startDate": {"text": "Jan 2023"},
                    "endDate": {"text": "Present"},
                }
            ],
            "education": [
                {
                    "degree": "B.Tech",
                    "fieldOfStudy": "Computer Science",
                    "schoolName": "IIT",
                    "endDate": {"year": 2022},
                }
            ],
            "skills": [{"name": "Python"}, {"name": "Go"}, {"name": "Kubernetes"}],
            "languages": [{"name": "English", "proficiency": "Full professional proficiency"}],
        },
    },
]


def _load_jd_text() -> str:
    if os.path.exists(JD_TEXT_PATH):
        with open(JD_TEXT_PATH, encoding="utf-8") as fh:
            return fh.read()
    return (
        "HR Assistant — Prime Focus Group (Prime AC), Dubai, UAE.\n"
        "Support HR operations in a manufacturing / HVAC environment: onboarding, "
        "employee relations, payroll coordination, Tagalog preferred."
    )


def main():
    jd_text = _load_jd_text()
    payload = {"jd_text": jd_text, "candidates": SAMPLES}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/score",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"FAILED to reach {BASE}/score: {exc}")
        print("Start the server first: uv run uvicorn api.main:app --host 127.0.0.1 --port 8080")
        sys.exit(1)

    print(json.dumps(body, indent=2)[:4000])
    cards = body.get("cards") or []
    assert body.get("count") == len(cards) >= 1, body
    ids = {c["candidate_id"] for c in cards}
    assert "cand-roxanna" in ids and "cand-entry-hr" in ids, ids
    for c in cards:
        assert "total_score" in c and c["total_score"] is not None, c
        assert "component_breakdown" in c, c
        assert "reasoning" in c, c
        assert "rank" in c, c
    scores = {c["candidate_id"]: c["total_score"] for c in cards}
    print("\nOK — swipe cards returned:")
    for c in sorted(cards, key=lambda x: x["rank"] or 999):
        print(
            f"  #{c['rank']} {c['candidate_id']:<16} score={c['total_score']} "
            f"title={c.get('title')!r} signals={c.get('matched_signals', [])[:4]}"
        )
    # Unrelated SWE should not beat both HR profiles on an HR JD (soft sanity).
    if "cand-unrelated" in scores:
        hr_best = max(scores["cand-roxanna"], scores["cand-entry-hr"])
        assert scores["cand-unrelated"] <= hr_best + 0.15, scores
    # Touch JobRoleSchema import so offline tooling stays honest about the shape.
    if os.path.exists(JD_STORE):
        with open(JD_STORE) as fh:
            JobRoleSchema.model_validate(json.load(fh))
    print("sanity passed")


if __name__ == "__main__":
    main()
