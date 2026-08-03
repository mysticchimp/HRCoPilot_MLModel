"""Pure per-role work-history features derived from CandidateProfile.positions.

Facts only — the tunable tenure thresholds live in the attrition scorer
(core/scoring.py) / models/mappings.py. Reused by the tenure/attrition component
and (later) the relevant-vs-adjacent experience component and completeness flag.
"""
from __future__ import annotations

import re
from typing import Optional

from models.candidate import CandidatePosition
from models.mappings import (
    adjacent_role_keywords,
    experience_relevance_adjacent_credit,  # noqa: F401  (re-exported for callers/tests)
    relevant_role_keywords,
)

# employment_type substrings that mark a role as NON-permanent, so it is excluded
# from the job-hop count — short contracts/internships are expected, not a
# flight-risk signal. Anything else (including unknown/None) is treated as
# permanent (benefit of the doubt: most roles are permanent).
_NON_PERMANENT_TOKENS = (
    "contract", "temporary", "freelance", "self-employed", "self employed",
    "intern", "apprentice", "seasonal", "volunteer",
)


def is_permanent(employment_type: Optional[str]) -> bool:
    """True unless the raw employment_type names a non-permanent arrangement."""
    if not employment_type:
        return True  # unknown -> benefit of the doubt
    lowered = employment_type.lower()
    return not any(token in lowered for token in _NON_PERMANENT_TOKENS)


def tenure_features(positions: list[CandidatePosition]) -> dict:
    """Raw tenure facts consumed by calculate_attrition_score.

    Returns:
        current_tenure_months: tenure of the current role (if dated), else None.
        completed_perm_tenures: tenures (months) of COMPLETED PERMANENT roles —
            the current role and contractors/interns are excluded, so a short
            *current* stint never looks like flight-risk.
        n_dated_roles: number of roles with a known tenure.
    """
    current: Optional[int] = None
    completed_perm: list[int] = []
    n_dated = 0
    for p in positions:
        if p.tenure_months is not None:
            n_dated += 1
        if p.is_current:
            if p.tenure_months is not None and current is None:
                current = p.tenure_months
            continue
        if is_permanent(p.employment_type) and p.tenure_months is not None:
            completed_perm.append(p.tenure_months)
    return {
        "current_tenure_months": current,
        "completed_perm_tenures": completed_perm,
        "n_dated_roles": n_dated,
    }


def _matches_any(text: str, keywords: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def classify_role(title: Optional[str]) -> str:
    """Classify a role title as 'relevant', 'adjacent', or 'unrelated' (C5 P3).

    Word-boundary matched with precedence RELEVANT > ADJACENT, so 'HR Operations'
    and 'HR Coordinator' land RELEVANT (not ADJACENT via 'operations'/'coordinator').
    """
    if not title:
        return "unrelated"
    t = title.lower()
    if _matches_any(t, relevant_role_keywords):
        return "relevant"
    if _matches_any(t, adjacent_role_keywords):
        return "adjacent"
    return "unrelated"


def relevance_features(positions: list[CandidatePosition]) -> dict:
    """Tenure-weighted relevant/adjacent months for calculate_experience_relevance_score.

    Returns relevant_months, adjacent_months and total_dated_months (roles with a
    known tenure). Undated roles contribute to none.
    """
    relevant = adjacent = total = 0
    for p in positions:
        if p.tenure_months is None:
            continue
        total += p.tenure_months
        cls = classify_role(p.title)
        if cls == "relevant":
            relevant += p.tenure_months
        elif cls == "adjacent":
            adjacent += p.tenure_months
    return {"relevant_months": relevant, "adjacent_months": adjacent, "total_dated_months": total}
