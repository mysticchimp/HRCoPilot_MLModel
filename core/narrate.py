"""Candidate Summary + Assessment narration via Claude.

Compresses Apify profiles, builds a cacheable system prefix (instructions + JD),
and fans out per-candidate structured Claude calls with bounded concurrency.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel, Field

from core.llm import get_provider
from core.llm.base import LLMProvider
from models.data_models import JobRoleSchema
from prompts.narration import jd_prefix, system_prompt, user_prompt

logger = logging.getLogger(__name__)

NARRATE_MODEL = os.environ.get("NARRATE_MODEL", "claude-sonnet-4-6")
NARRATE_CONCURRENCY = max(1, int(os.environ.get("NARRATE_CONCURRENCY", "8")))
_ABOUT_CAP = 500
_EXP_CAP = 3
_EXP_DESC_CAP = 220
_SKILL_CAP = 40


class NarrativeResult(BaseModel):
    summary: str = Field(..., description="JD-independent professional summary")
    assessment: str = Field(..., description="JD fit assessment with strengths and gaps")


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _date_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        text = _clean(value.get("text"))
        if text:
            return text
        year = value.get("year")
        month = value.get("month")
        if year is not None and month:
            return f"{month} {int(float(year))}"
        if year is not None:
            return str(int(float(year)))
        return None
    return _clean(value)


def _truncate(text: str | None, cap: int) -> str | None:
    if not text:
        return None
    if len(text) <= cap:
        return text
    return text[: cap - 1].rstrip() + "…"


def compress_profile(raw_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Extract a compact, LLM-ready profile from Apify Full nested JSON.

    Keeps name, headline, current role, location, capped about, ~3 recent
    positions, skills, education, and languages — not the raw scrape blob.
    """
    record = raw_profile or {}

    first = _clean(record.get("firstName"))
    last = _clean(record.get("lastName"))
    full = _clean(record.get("fullName"))
    if not full and (first or last):
        full = " ".join(p for p in (first, last) if p)

    loc = record.get("location")
    location_text = None
    if isinstance(loc, dict):
        location_text = _clean(loc.get("linkedinText")) or _clean(
            (loc.get("parsed") or {}).get("text") if isinstance(loc.get("parsed"), dict) else None
        )
    elif isinstance(loc, str):
        location_text = _clean(loc)

    # Current title/company
    current_title = None
    current_company = None
    cps = _as_list(record.get("currentPosition"))
    for cp in cps:
        if not isinstance(cp, dict):
            continue
        current_title = _clean(cp.get("position") or cp.get("title"))
        current_company = _clean(cp.get("companyName") or cp.get("company"))
        if current_title or current_company:
            break

    experience: list[dict[str, Any]] = []
    for exp in _as_list(record.get("experience")):
        if not isinstance(exp, dict):
            continue
        title = _clean(exp.get("position") or exp.get("title"))
        company = _clean(exp.get("companyName") or exp.get("company"))
        if not title and not company:
            continue
        experience.append(
            {
                "title": title,
                "company": company,
                "start": _date_text(exp.get("startDate")),
                "end": _date_text(exp.get("endDate")),
                "duration": _clean(exp.get("duration")),
                "description": _truncate(_clean(exp.get("description")), _EXP_DESC_CAP),
            }
        )
        if len(experience) >= _EXP_CAP:
            break

    if not current_title and experience:
        current_title = experience[0].get("title")
        current_company = experience[0].get("company")

    skills: list[str] = []
    for item in _as_list(record.get("skills")):
        if isinstance(item, str):
            name = _clean(item)
        elif isinstance(item, dict):
            name = _clean(item.get("name") or item.get("title"))
        else:
            name = None
        if name and name not in skills:
            skills.append(name)
        if len(skills) >= _SKILL_CAP:
            break

    education: list[dict[str, Any]] = []
    for item in _as_list(record.get("education")):
        if not isinstance(item, dict):
            continue
        edu = {
            "degree": _clean(item.get("degree")),
            "field": _clean(item.get("fieldOfStudy") or item.get("field")),
            "school": _clean(item.get("schoolName") or item.get("school")),
        }
        if any(edu.values()):
            education.append(edu)

    languages: list[dict[str, Any]] = []
    for item in _as_list(record.get("languages")):
        if isinstance(item, str):
            name = _clean(item)
            if name:
                languages.append({"language": name})
            continue
        if not isinstance(item, dict):
            continue
        name = _clean(item.get("name") or item.get("language"))
        if not name:
            continue
        languages.append(
            {
                "language": name,
                "proficiency": _clean(item.get("proficiency") or item.get("level")),
            }
        )

    out: dict[str, Any] = {
        "name": full,
        "headline": _clean(record.get("headline")),
        "current_title": current_title,
        "current_company": current_company,
        "location": location_text,
        "about": _truncate(_clean(record.get("about")), _ABOUT_CAP),
        "experience": experience,
        "skills": skills,
        "education": education,
        "languages": languages,
    }
    # Drop empty containers / nulls for fewer tokens
    return {k: v for k, v in out.items() if v not in (None, [], {})}


def estimate_tokens(obj: Any) -> int:
    """Rough token estimate (~4 chars/token) for compression tests."""
    return max(1, len(json.dumps(obj, ensure_ascii=False)) // 4)


def compact_jd_block(jd_parsed_or_text: str | dict[str, Any]) -> str:
    """Render JD content for the shared system prefix (assessment only)."""
    if isinstance(jd_parsed_or_text, str):
        text = jd_parsed_or_text.strip()
        if not text:
            raise ValueError("jd_parsed_or_text string is empty")
        return text

    if not isinstance(jd_parsed_or_text, dict):
        raise ValueError("jd_parsed_or_text must be a string or a JobRoleSchema object")

    jd = JobRoleSchema.model_validate(jd_parsed_or_text)
    # Compact structured dump — keep the fields that drive assessment.
    payload: dict[str, Any] = {
        "role": jd.role,
        "company": jd.company.model_dump(mode="json", exclude_none=True) if jd.company else None,
        "industry": jd.industry,
        "role_objectives": jd.role_objectives,
        "responsibilities": jd.responsibilities,
        "skills": [s.model_dump(mode="json", exclude_none=True) for s in (jd.skills or [])],
        "languages": [
            lp.model_dump(mode="json", exclude_none=True)
            for lp in (jd.language_proficiency or [])
        ]
        or None,
        "experience": jd.experience.model_dump(mode="json", exclude_none=True)
        if jd.experience
        else None,
        "location": jd.location.model_dump(mode="json", exclude_none=True)
        if jd.location
        else None,
        "qualifications": jd.qualifications.model_dump(mode="json", exclude_none=True)
        if jd.qualifications
        else None,
    }
    payload = {k: v for k, v in payload.items() if v not in (None, [], {})}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_cached_system(jd_block: str) -> str:
    """Shared system prefix: narration instructions + JD (prompt-cacheable)."""
    return system_prompt + "\n\n" + jd_prefix.format(jd_block=jd_block)


def build_candidate_user_prompt(
    compressed: dict[str, Any],
    component_breakdown: dict[str, Any] | None,
    matched_signals: list[str] | None,
) -> str:
    return user_prompt.format(
        profile_json=json.dumps(compressed, indent=2, ensure_ascii=False),
        component_breakdown_json=json.dumps(
            component_breakdown or {}, indent=2, ensure_ascii=False
        ),
        matched_signals_json=json.dumps(matched_signals or [], indent=2, ensure_ascii=False),
    )


def narrate_one(
    *,
    candidate_id: str,
    raw_profile: dict[str, Any] | None,
    component_breakdown: dict[str, Any] | None,
    matched_signals: list[str] | None,
    cached_system: str,
    provider: LLMProvider,
    model: str = NARRATE_MODEL,
) -> dict[str, Any]:
    """Run one Claude call. Returns success or error dict (never raises to caller)."""
    try:
        compressed = compress_profile(raw_profile)
        prompt = build_candidate_user_prompt(
            compressed, component_breakdown, matched_signals
        )
        # Prefer Anthropic prompt caching when available.
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "schema": NarrativeResult,
            "system": cached_system,
            "model": model,
            "temperature": 0.3,
        }
        generate = provider.generate_structured
        # AnthropicProvider accepts cache_system=; other providers ignore via ** if not present
        try:
            result = generate(**kwargs, cache_system=True)  # type: ignore[call-arg]
        except TypeError:
            result = generate(**kwargs)
        return {
            "candidate_id": candidate_id,
            "summary": result.summary.strip(),
            "assessment": result.assessment.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "narrate_one failed candidate_id=%s err=%s",
            candidate_id,
            str(exc)[:200],
        )
        return {"candidate_id": candidate_id, "error": str(exc)[:500]}


def narrate_candidates(
    jd_parsed_or_text: str | dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    provider: LLMProvider | None = None,
    concurrency: int | None = None,
    model: str = NARRATE_MODEL,
) -> list[dict[str, Any]]:
    """Narrate a batch with bounded concurrency. Per-candidate failures become error markers."""
    if not candidates:
        raise ValueError("candidates must be a non-empty list")

    provider = provider or get_provider("anthropic")
    workers = concurrency if concurrency is not None else NARRATE_CONCURRENCY
    workers = max(1, min(workers, len(candidates)))

    jd_block = compact_jd_block(jd_parsed_or_text)
    cached_system = build_cached_system(jd_block)

    results_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                narrate_one,
                candidate_id=str(c.get("candidate_id") or ""),
                raw_profile=c.get("raw_profile") or {},
                component_breakdown=c.get("component_breakdown"),
                matched_signals=c.get("matched_signals"),
                cached_system=cached_system,
                provider=provider,
                model=model,
            ): str(c.get("candidate_id") or "")
            for c in candidates
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                results_by_id[cid] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results_by_id[cid] = {
                    "candidate_id": cid,
                    "error": str(exc)[:500],
                }

    # Preserve input order
    ordered: list[dict[str, Any]] = []
    for c in candidates:
        cid = str(c.get("candidate_id") or "")
        ordered.append(
            results_by_id.get(cid) or {"candidate_id": cid, "error": "missing result"}
        )
    return ordered
