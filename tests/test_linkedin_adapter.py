import pytest

from core.adapters.linkedin_adapter import (
    LinkedInAdapter,
    _duration_to_months,
    _infer_seniority,
    _months_between,
    _parse_month_year,
)

RAW = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"


@pytest.fixture(scope="module")
def profiles():
    return LinkedInAdapter().to_profiles(RAW)


def test_linkedin_adapter_loads_all(profiles):
    assert len(profiles) == 145
    assert all(p.source == "linkedin" for p in profiles)


def test_linkedin_core_field_coverage(profiles):
    n = len(profiles)
    assert sum(bool(p.job_title) for p in profiles) == n          # 100%
    assert sum(p.location is not None for p in profiles) == n      # 100%
    assert sum(len(p.skills) > 0 for p in profiles) / n > 0.9
    assert sum(len(p.education) > 0 for p in profiles) / n > 0.9


def test_linkedin_enrichment_present(profiles):
    n = len(profiles)
    assert all(p.open_to_work is not None for p in profiles)       # 100%
    assert all(p.verified is not None for p in profiles)           # 100%
    assert all(p.seniority is not None for p in profiles)          # 100%
    assert sum(p.years_experience is not None for p in profiles) / n > 0.9


def test_linkedin_gold_join_key_present(profiles):
    # every profile must retain linkedinUrl so predictions join to the gold labels
    assert all(p.raw.get("linkedinUrl") for p in profiles)


def test_seniority_no_substring_overmatch(profiles):
    # the 'coo' in 'coordinator' bug produced 15 c_level; an HR-assistant pool
    # should have very few genuine c_level candidates
    c_level = sum(1 for p in profiles if p.seniority == "c_level")
    assert c_level <= 5


def test_seniority_word_boundary():
    assert _infer_seniority("HR Coordinator/Welfare Officer") == "entry"
    assert _infer_seniority("HR Assistant") == "entry"
    assert _infer_seniority("HR Manager") == "senior"
    assert _infer_seniority("Head of People") == "executive"
    assert _infer_seniority("Chief People Officer") == "c_level"
    assert _infer_seniority("International Recruiter") == "mid"  # 'intern' must not match


def test_duration_parsing():
    assert _duration_to_months("3 yrs 10 mos") == 46
    assert _duration_to_months("5 mos") == 5
    assert _duration_to_months("1 yr") == 12
    assert _duration_to_months(None) == 0


def test_parse_month_year():
    assert _parse_month_year("Apr 2025") == (2025, 4)
    assert _parse_month_year("December 2019") == (2019, 12)
    assert _parse_month_year("2021") == (2021, 1)
    assert _parse_month_year("Present") is None
    assert _parse_month_year(None) is None


def test_months_between():
    assert _months_between("Jan 2023", "Jan 2024") == 12
    assert _months_between("Apr 2025", "Jul 2025") == 3
    assert _months_between("Jan 2024", "Jan 2023") is None  # negative -> None
    assert _months_between("Present", "Jan 2024") is None


def test_positions_parsed_from_synthetic_record():
    rec = {
        "experience/0/position": "HR Assistant",
        "experience/0/companyName": "Acme",
        "experience/0/duration": "1 yr 4 mos",
        "experience/0/employmentType": "Full-time",
        "experience/0/startDate/text": "Apr 2025",
        "experience/0/endDate/text": "Present",
        "experience/1/position": "Admin",
        "experience/1/companyName": "Beta",
        "experience/1/duration": None,  # missing duration -> date fallback
        "experience/1/employmentType": "Contract",
        "experience/1/startDate/text": "Jan 2023",
        "experience/1/endDate/text": "Jan 2024",
        "organizations/0/name": "ShouldBeIgnored",  # organizations is not merged
    }
    positions = LinkedInAdapter()._positions(rec)
    assert len(positions) == 2
    cur, prev = positions
    assert (cur.title, cur.company, cur.tenure_months, cur.is_current) == ("HR Assistant", "Acme", 16, True)
    assert cur.employment_type == "Full-time" and cur.start == "Apr 2025" and cur.end == "Present"
    assert (prev.title, prev.tenure_months, prev.is_current) == ("Admin", 12, False)  # 12 mo from date fallback
    assert prev.employment_type == "Contract"
    assert all(p.company != "ShouldBeIgnored" for p in positions)


def test_positions_real_reference_candidates(profiles):
    by_id = {p.candidate_id: p for p in profiles}
    samastha = next(p for cid, p in by_id.items() if "samasthasunoj" in cid)
    assert len(samastha.positions) == 4
    assert all(p.employment_type == "Full-time" for p in samastha.positions)
    assert samastha.positions[0].is_current
    amulya = next(p for cid, p in by_id.items() if cid.startswith("amulya-dattada"))
    assert len(amulya.positions) == 2


def test_positions_coverage(profiles):
    # the raw export carries per-role structure for essentially the whole pool
    assert sum(len(p.positions) > 0 for p in profiles) / len(profiles) > 0.95
