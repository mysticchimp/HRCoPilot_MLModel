from core.positions import classify_role, is_permanent, relevance_features, tenure_features
from models.candidate import CandidatePosition


def _pos(tenure, is_current=False, emp="Full-time", title=None):
    return CandidatePosition(tenure_months=tenure, is_current=is_current, employment_type=emp, title=title)


def test_is_permanent():
    assert is_permanent("Full-time")
    assert is_permanent("Permanent")
    assert is_permanent("Permanent Full-time")
    assert is_permanent(None)  # unknown -> benefit of the doubt
    assert not is_permanent("Contract")
    assert not is_permanent("Internship")
    assert not is_permanent("Seasonal")
    assert not is_permanent("Freelance")
    assert not is_permanent("Self-employed")


def test_tenure_features_excludes_current_and_contractors():
    positions = [
        _pos(16, is_current=True),          # current -> current_tenure, not completed
        _pos(14),                            # completed permanent
        _pos(6, emp="Contract"),             # contractor -> excluded from hop count
        _pos(12),                            # completed permanent
        _pos(None),                          # undated -> not counted
    ]
    f = tenure_features(positions)
    assert f["current_tenure_months"] == 16
    assert f["completed_perm_tenures"] == [14, 12]
    assert f["n_dated_roles"] == 4  # 16, 14, 6, 12 (the None is excluded)


def test_tenure_features_no_completed_permanent():
    f = tenure_features([_pos(10, is_current=True)])
    assert f["current_tenure_months"] == 10
    assert f["completed_perm_tenures"] == []
    assert f["n_dated_roles"] == 1


def test_tenure_features_empty():
    f = tenure_features([])
    assert f == {"current_tenure_months": None, "completed_perm_tenures": [], "n_dated_roles": 0}


def test_classify_role_precedence():
    assert classify_role("HR Operations Executive") == "relevant"   # hr wins over operations
    assert classify_role("HR Coordinator") == "relevant"           # hr wins over coordinator
    assert classify_role("Payroll Staff") == "relevant"
    assert classify_role("Talent Acquisition Specialist") == "relevant"
    assert classify_role("Process Executive") == "unrelated"       # bare 'executive' is not adjacent
    assert classify_role("Administrative Assistant") == "adjacent"
    assert classify_role("Office Coordinator") == "adjacent"
    assert classify_role("PRO") == "adjacent"                      # UAE public-relations officer
    assert classify_role("Procurement Specialist") == "unrelated"  # \bpro\b must not match mid-word
    assert classify_role("Software Engineer") == "unrelated"
    assert classify_role(None) == "unrelated"


def test_relevance_features():
    positions = [
        _pos(19, title="HR Operations Executive"),  # relevant
        _pos(26, title="Process Executive"),         # unrelated
        _pos(12, title="Office Administrator"),       # adjacent
        _pos(None, title="HR Assistant"),             # undated -> ignored
    ]
    f = relevance_features(positions)
    assert f == {"relevant_months": 19, "adjacent_months": 12, "total_dated_months": 57}
