from core.completeness import data_completeness


def test_completeness_rich():
    dc = data_completeness("About me", "Did HR things", n_skills=8, n_dated_roles=3)
    assert dc == {"level": "rich", "missing": []}


def test_completeness_low_thin_skills():
    # Che-like: bullseye title but empty about + <5 skills -> low (routed to screening)
    dc = data_completeness(None, None, n_skills=4, n_dated_roles=4)
    assert dc["level"] == "low"
    assert "about" in dc["missing"] and "responsibilities" in dc["missing"] and "skills_lt_5" in dc["missing"]
    assert "no_dated_roles" not in dc["missing"]  # Che DOES have work history


def test_completeness_low_no_dated_roles():
    dc = data_completeness("About", "Resp", n_skills=10, n_dated_roles=0)
    assert dc["level"] == "low"
    assert dc["missing"] == ["no_dated_roles"]


def test_completeness_partial():
    # only 'about' missing, but responsibilities + >=5 skills + dated roles present -> partial
    dc = data_completeness(None, "Managed onboarding", n_skills=7, n_dated_roles=2)
    assert dc == {"level": "partial", "missing": ["about"]}


def test_completeness_low_requires_both_about_and_responsibilities():
    # missing about alone (with responsibilities present and skills ok) is NOT low
    assert data_completeness(None, "Resp", 6, 2)["level"] == "partial"
    # missing both -> low
    assert data_completeness(None, None, 6, 2)["level"] == "low"
