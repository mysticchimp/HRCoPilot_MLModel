"""Data-completeness flag (C5 P6).

A SEPARATE output annotation — deliberately NOT folded into total_score. Low-data
profiles keep their evidence-based rank (they rank low on merit) but are surfaced
for screening rather than silently discarded: a bullseye-title thin profile may be
a real hire, so route it to a human. This does NOT soften the evidence penalty
(that is the rejected "reward emptiness" path); it only annotates what is missing.
"""
from __future__ import annotations

from typing import Optional

from models.mappings import completeness_min_skills


def data_completeness(
    summary: Optional[str],
    responsibilities: Optional[str],
    n_skills: int,
    n_dated_roles: int,
) -> dict:
    """Classify a profile as rich / partial / low with the list of missing signals.

    low  := (no about AND no responsibilities) OR fewer than completeness_min_skills
            skills OR no dated roles  (the spec 8 rule)
    rich := nothing missing
    partial := something missing but not enough to be low
    """
    missing: list[str] = []
    if not summary:
        missing.append("about")
    if not responsibilities:
        missing.append("responsibilities")
    if (n_skills or 0) < completeness_min_skills:
        missing.append("skills_lt_5")
    if (n_dated_roles or 0) == 0:
        missing.append("no_dated_roles")

    low = (
        ("about" in missing and "responsibilities" in missing)
        or "skills_lt_5" in missing
        or "no_dated_roles" in missing
    )
    if low:
        level = "low"
    elif not missing:
        level = "rich"
    else:
        level = "partial"
    return {"level": level, "missing": missing}
