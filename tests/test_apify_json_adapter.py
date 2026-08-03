"""Unit tests for ApifyJsonAdapter (nested LinkedIn Full-mode JSON)."""

from core.adapters.apify_json_adapter import ApifyJsonAdapter

SAMPLE = {
    "publicIdentifier": "roxanna-ghassemlou-007078191",
    "linkedinUrl": "https://www.linkedin.com/in/roxanna-ghassemlou-007078191",
    "firstName": "Roxanna",
    "lastName": "Ghassemlou",
    "fullName": "Roxanna Ghassemlou",
    "headline": "HR Manager | Distinct Group | Dubai",
    "about": "Motivated HR professional with over 4 years of experience.",
    "openToWork": False,
    "verified": True,
    "followerCount": 3554,
    "connectionsCount": 3008,
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
            "description": "Led HR operations and onboarding.",
            "startDate": {"text": "Mar 2026", "month": "Mar", "year": 2026},
            "endDate": {"text": "Present"},
        },
        {
            "position": "HR Assistant",
            "companyName": "Prime Focus",
            "duration": "2 yrs 3 mos",
            "employmentType": "Full-time",
            "description": "Supported payroll and employee relations.",
            "startDate": {"text": "Jan 2024"},
            "endDate": {"text": "Mar 2026"},
        },
    ],
    "education": [
        {
            "degree": "Master of Science - MSc",
            "fieldOfStudy": "Occupational and Organisational Psychology",
            "schoolName": "University of Sussex",
            "endDate": {"year": 2021, "text": "Sep 2021"},
        }
    ],
    "skills": [
        {"name": "Onboarding", "endorsements": "3 endorsements"},
        {"name": "Payroll"},
        {"name": "HR Policies", "endorsements": 1},
    ],
    "topSkills": ["People Management", "Onboarding"],
    "languages": [
        {"name": "English", "proficiency": "Native or bilingual proficiency"},
        {"name": "Arabic", "proficiency": "Professional working proficiency"},
    ],
    "certifications": [{"title": "SHRM-CP"}],
}


def test_apify_json_core_fields():
    p = ApifyJsonAdapter().to_profile(SAMPLE, 0)
    assert p.candidate_id == "roxanna-ghassemlou-007078191"
    assert p.job_title == "HR Manager"
    assert p.source == "apify"
    assert "Onboarding" in p.skills
    assert "People Management" in p.skills  # from topSkills union
    assert p.summary and "Motivated" in p.summary
    assert p.location and p.location.city == "Dubai"
    assert p.location.country_code == "AE"
    assert len(p.education) == 1
    assert p.education[0].school == "University of Sussex"
    assert p.education[0].end_year == 2021
    assert p.years_experience == 2.7  # 5 + 27 months
    assert p.seniority == "senior"
    assert p.employers == ["Distinct Group", "Prime Focus"]
    assert p.languages and p.languages[0].language == "English"
    assert p.certifications == ["SHRM-CP"]
    assert p.endorsed_skills and p.endorsed_skills["Onboarding"] == 3
    assert p.raw.get("linkedinUrl")


def test_apify_json_positions_current_flag():
    p = ApifyJsonAdapter().to_profile(SAMPLE, 0)
    assert len(p.positions) == 2
    assert p.positions[0].is_current is True
    assert p.positions[0].title == "HR Manager"
    assert p.positions[1].is_current is False
    assert p.positions[1].tenure_months == 27


def test_apify_json_caller_candidate_id_override():
    record = {**SAMPLE, "_candidate_id": "db-uuid-123"}
    p = ApifyJsonAdapter().to_profile(record, 0)
    assert p.candidate_id == "db-uuid-123"


def test_apify_json_load_list():
    profiles = ApifyJsonAdapter().to_profiles([SAMPLE, {**SAMPLE, "publicIdentifier": "other"}])
    assert len(profiles) == 2
    assert profiles[1].candidate_id == "other"


def test_apify_json_headline_fallback_title():
    record = {
        "headline": "People Ops Lead | Manufacturing",
        "experience": [],
        "skills": [],
    }
    p = ApifyJsonAdapter().to_profile(record, 5)
    assert p.job_title == "People Ops Lead"
    assert p.candidate_id == "A006"
