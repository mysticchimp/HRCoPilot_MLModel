"""Swipe-feed backend (C5 P7 / U2) — card data contract + swipe-capture schema.

Backend only (no front-end). Turns a scored candidate row + its profile into the
Tinder-style card the recruiter swipes. Everything on the card is GROUNDED and
DETERMINISTIC — `matched_signals` and `reasoning` are built from the actual matched
skills/industries/languages and the real component scores, NEVER an LLM's
boilerplate (the failure mode documented in the spec 2). Swipes are captured as the
first real human-label signal (the path to closing the T1 gold-set gap).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from core.industry_normalization import industry_present, jd_industry_requirements
from core.language_normalization import jd_language_requirements, normalize_candidate_languages
from models.candidate import CandidateProfile
from models.data_models import JobRoleSchema
from models.mappings import attrition_chronic_hop_score

# every computed component is surfaced on the card (incl. weight-0 ones) for the
# radar/bar chart — transparency; the weighted result is `total_score`.
COMPONENT_KEYS = [
    "title", "skill", "qualification", "seniority", "experience", "industry",
    "language", "location", "attrition", "experience_relevance", "education_relevance", "similarity",
]

_WORKFORCE_LANGUAGE_TOKENS = ("tagalog", "filipino", "pilipino")


def _round(v) -> Optional[float]:
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


def _name(profile: CandidateProfile) -> str:
    raw = profile.raw or {}
    full = raw.get("fullName")
    if full:
        return full
    parts = [raw.get("firstName"), raw.get("lastName")]
    name = " ".join(p for p in parts if p)
    return name or profile.candidate_id


def _current_company(profile: CandidateProfile) -> Optional[str]:
    for pos in (profile.positions or []):
        if pos.is_current and pos.company:
            return pos.company
    if profile.positions and profile.positions[0].company:
        return profile.positions[0].company
    return profile.employers[0] if profile.employers else None


def _has_workforce_language(profile: CandidateProfile) -> bool:
    """True only if the candidate STRUCTURALLY lists Tagalog/Filipino (no name-based inference)."""
    for lang in (profile.languages or []):
        if lang.language and any(t in lang.language.lower() for t in _WORKFORCE_LANGUAGE_TOKENS):
            return True
    return False


def build_matched_signals(profile: CandidateProfile, row: dict, jd: JobRoleSchema) -> list[str]:
    """Grounded match tokens — real skill/industry/language overlaps only, capped at 8."""
    signals: list[str] = []

    cand_skills = [s.lower() for s in (profile.skills or []) if isinstance(s, str)]
    jd_skills = [s.skill for s in jd.skills]
    if jd.technologies:
        jd_skills += [t.technology for t in jd.technologies]
    for js in jd_skills:
        jsl = (js or "").lower().strip()
        if jsl and any(jsl in cs or cs in jsl for cs in cand_skills):
            signals.append(js)

    text = row.get("sector_text") or ""
    for industry, _ in jd_industry_requirements(jd):
        if industry_present(industry, text):
            signals.append(industry)

    cand_langs = set(normalize_candidate_languages(
        [l.language for l in (profile.languages or []) if l.language]))
    for lang, _ in jd_language_requirements(jd):
        if lang in cand_langs:
            signals.append(lang)

    seen, out = set(), []
    for s in signals:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:8]


def _reasoning(profile: CandidateProfile, matched: list[str], flags: dict) -> str:
    bits = [profile.job_title or "Candidate"]
    if profile.seniority:
        bits.append(f"{profile.seniority} level")
    if matched:
        bits.append("matches " + ", ".join(matched[:3]))
    if profile.location and profile.location.text:
        bits.append(profile.location.text)
    if flags.get("flight_risk"):
        bits.append("short tenure (flight-risk)")
    if flags.get("data_completeness") == "low":
        bits.append("thin profile — route to screening")
    return "; ".join(bits) + "."


def build_card(profile: CandidateProfile, row: dict, jd: JobRoleSchema) -> dict:
    """Assemble the per-candidate swipe card (the spec 9.1 contract)."""
    matched = build_matched_signals(profile, row, jd)
    attrition = row.get("attrition_score")
    industry = row.get("industry_score")
    flags = {
        "flight_risk": attrition is not None and float(attrition) <= attrition_chronic_hop_score,
        "industrial_sector": industry is not None and float(industry) >= 0.5,
        "workforce_language": _has_workforce_language(profile),
        "data_completeness": row.get("data_completeness_level"),
    }
    rank = row.get("pipeline_rank")
    return {
        "candidate_id": profile.candidate_id,
        "name": _name(profile),
        "title": profile.job_title,
        "current_company": _current_company(profile),
        "location": profile.location.text if profile.location else None,
        "rank": int(rank) if rank is not None else None,
        "total_score": _round(row.get("total_score")),
        "component_breakdown": {k: _round(row.get(f"{k}_score")) for k in COMPONENT_KEYS},
        "matched_signals": matched,
        "flags": flags,
        "reasoning": _reasoning(profile, matched, flags),
        "linkedin_url": (profile.raw or {}).get("linkedinUrl"),
    }


class SwipeEvent(BaseModel):
    """One recruiter swipe — the first real human-label signal (feeds T1 / learn-to-rank).

    decision: 'right' = shortlist/advance · 'left' = pass · 'up' = send to screening
    (used especially for low-data cards). Mind the selection bias when using swipes
    as labels — only shown candidates get labeled.
    """

    recruiter_id: str
    candidate_id: str
    jd_id: str
    decision: Literal["right", "left", "up"]
    ts: str  # ISO-8601 timestamp
    rank_shown: int


def append_swipe(event: SwipeEvent, path: str) -> None:
    """Persist a swipe as one JSON line (append-only capture log)."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(event.model_dump_json() + "\n")
