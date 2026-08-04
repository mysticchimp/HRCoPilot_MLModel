"""Adapter for Apify LinkedIn Full-mode nested JSON profiles.

Maps the nested shape stored in Sourcing_Apify's ``candidates.raw_profile``
(about, experience[], education[], skills[], languages[], headline, …) into
canonical ``CandidateProfile`` objects. Does not require the flattened CSV
``prefix/i/field`` format used by ``LinkedInAdapter``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Optional

from core.adapters.base import CandidateAdapter
from core.adapters.linkedin_adapter import (
    _duration_to_months,
    _infer_seniority,
    _map_employment,
    _months_between,
)
from models.candidate import (
    CandidateEducation,
    CandidateLanguage,
    CandidateLocation,
    CandidatePosition,
    CandidateProfile,
)


def _clean(value):
    """Normalize a nested-JSON cell to a value or None (drops NaN/empty)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _date_text(value) -> Optional[str]:
    """Normalize Apify date fields: string, or ``{text, year, month}`` dict."""
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
    return _clean(value) if isinstance(value, (str, int, float)) else None


def _end_year(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, dict):
        year = value.get("year")
        if year is None:
            text = _date_text(value)
            if text:
                m = re.search(r"\b(\d{4})\b", text)
                return int(m.group(1)) if m else None
            return None
        try:
            return int(float(year))
        except (TypeError, ValueError):
            return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


class ApifyJsonAdapter(CandidateAdapter):
    """Adapter for Apify Full-mode LinkedIn profile JSON (nested, not flattened CSV)."""

    source_name = "apify"

    def load(self, source) -> list[dict]:
        if isinstance(source, list):
            return list(source)
        if isinstance(source, dict):
            return [source]
        # path to a JSON file (array or single object)
        with open(source, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise TypeError(f"Unsupported Apify JSON source type: {type(data)!r}")

    def _skills(self, record: dict) -> tuple[list[str], Optional[dict]]:
        names: list[str] = []
        endorsed: dict[str, int] = {}
        for item in _as_list(record.get("skills")):
            if isinstance(item, str):
                name = _clean(item)
                if name and name not in names:
                    names.append(name)
                continue
            if not isinstance(item, dict):
                continue
            name = _clean(item.get("name") or item.get("title"))
            if not name:
                continue
            if name not in names:
                names.append(name)
            endorsements = item.get("endorsements")
            if endorsements is not None:
                match = re.search(r"(\d+)", str(endorsements))
                if match:
                    endorsed[name] = int(match.group(1))

        top = record.get("topSkills")
        if isinstance(top, str):
            for part in re.split(r"[•|,;/]+", top):
                name = _clean(part)
                if name and name not in names:
                    names.append(name)
        else:
            for item in _as_list(top):
                name = _clean(item if isinstance(item, str) else (item or {}).get("name"))
                if name and name not in names:
                    names.append(name)
        return names, (endorsed or None)

    def _education(self, record: dict) -> list[CandidateEducation]:
        education: list[CandidateEducation] = []
        for item in _as_list(record.get("education")):
            if not isinstance(item, dict):
                continue
            education.append(
                CandidateEducation(
                    degree=_clean(item.get("degree")),
                    field=_clean(item.get("fieldOfStudy") or item.get("field")),
                    school=_clean(item.get("schoolName") or item.get("school")),
                    end_year=_end_year(item.get("endDate") or item.get("endDate/year")),
                )
            )
        return education

    def _experience_items(self, record: dict) -> list[dict]:
        return [e for e in _as_list(record.get("experience")) if isinstance(e, dict)]

    def _current_position(self, record: dict) -> Optional[dict]:
        cps = _as_list(record.get("currentPosition"))
        for cp in cps:
            if isinstance(cp, dict) and any(_clean(cp.get(k)) for k in ("position", "title", "companyName")):
                return cp
        # Fall back to first experience entry (Apify Full usually orders most-recent first).
        exps = self._experience_items(record)
        return exps[0] if exps else None

    def _responsibilities(self, record: dict) -> Optional[str]:
        # Field contract is current/recent role text — not a career-long dump of every
        # experience description (that blew embedding inputs past 1k tokens on Full scrapes).
        texts: list[str] = []
        for exp in self._experience_items(record)[:3]:
            desc = _clean(exp.get("description"))
            if desc:
                texts.append(desc)
        current = self._current_position(record)
        if current:
            desc = _clean(current.get("description"))
            if desc and desc not in texts:
                texts.insert(0, desc)
        return " ".join(texts) if texts else None

    def _years_experience(self, record: dict) -> Optional[float]:
        total_months = sum(
            _duration_to_months(exp.get("duration")) for exp in self._experience_items(record)
        )
        return round(total_months / 12, 1) if total_months > 0 else None

    def _positions(self, record: dict) -> list[CandidatePosition]:
        positions: list[CandidatePosition] = []
        for exp in self._experience_items(record):
            end = _date_text(exp.get("endDate"))
            is_current = end is None or (isinstance(end, str) and end.strip().lower() in ("", "present"))
            tenure = _duration_to_months(exp.get("duration")) or None
            start = _date_text(exp.get("startDate"))
            if tenure is None and not is_current:
                tenure = _months_between(start, end)
            positions.append(
                CandidatePosition(
                    title=_clean(exp.get("position") or exp.get("title")),
                    company=_clean(exp.get("companyName") or exp.get("company")),
                    start=start,
                    end=end,
                    tenure_months=tenure,
                    is_current=is_current,
                    employment_type=_clean(exp.get("employmentType")),
                )
            )
        return positions

    def _location(self, record: dict) -> Optional[CandidateLocation]:
        loc = record.get("location")
        if isinstance(loc, str):
            text = _clean(loc)
            return CandidateLocation(text=text) if text else None
        if not isinstance(loc, dict):
            return None
        parsed = loc.get("parsed") if isinstance(loc.get("parsed"), dict) else {}
        location = CandidateLocation(
            text=_clean(parsed.get("text")) or _clean(loc.get("linkedinText")) or _clean(loc.get("text")),
            city=_clean(parsed.get("city")),
            state=_clean(parsed.get("state")),
            country=_clean(parsed.get("country")),
            country_code=_clean(parsed.get("countryCode")) or _clean(loc.get("countryCode")),
        )
        return location if any(location.model_dump().values()) else None

    def _languages(self, record: dict) -> Optional[list[CandidateLanguage]]:
        langs: list[CandidateLanguage] = []
        for item in _as_list(record.get("languages")):
            if isinstance(item, str):
                name = _clean(item)
                if name:
                    langs.append(CandidateLanguage(language=name))
                continue
            if not isinstance(item, dict):
                continue
            name = _clean(item.get("name") or item.get("language"))
            if name:
                langs.append(
                    CandidateLanguage(language=name, proficiency=_clean(item.get("proficiency")))
                )
        return langs or None

    def _certifications(self, record: dict) -> Optional[list[str]]:
        certs: list[str] = []
        for item in _as_list(record.get("certifications")):
            if isinstance(item, str):
                name = _clean(item)
            elif isinstance(item, dict):
                name = _clean(item.get("title") or item.get("name"))
            else:
                name = None
            if name:
                certs.append(name)
        return certs or None

    def _employers(self, record: dict) -> Optional[list[str]]:
        names: list[str] = []
        current = self._current_position(record)
        if current:
            company = _clean(current.get("companyName") or current.get("company"))
            if company:
                names.append(company)
        for exp in self._experience_items(record):
            company = _clean(exp.get("companyName") or exp.get("company"))
            if company and company not in names:
                names.append(company)
        return names or None

    @staticmethod
    def _bool(value) -> Optional[bool]:
        cleaned = _clean(value)
        if isinstance(cleaned, bool):
            return cleaned
        if isinstance(cleaned, str):
            return cleaned.lower() == "true"
        return None

    @staticmethod
    def _int(value) -> Optional[int]:
        cleaned = _clean(value)
        if isinstance(cleaned, bool):
            return None
        if isinstance(cleaned, (int, float)):
            return int(cleaned)
        if isinstance(cleaned, str) and cleaned.replace(".", "", 1).isdigit():
            return int(float(cleaned))
        return None

    def _job_title(self, record: dict) -> str:
        current = self._current_position(record)
        if current:
            title = _clean(current.get("position") or current.get("title"))
            if title:
                return title
        headline = _clean(record.get("headline"))
        if headline:
            return headline.split("|")[0].strip()
        return ""

    def to_profile(self, record: dict, index: int) -> CandidateProfile:
        # Caller-supplied id (API) wins over Apify publicIdentifier.
        override = _clean(record.get("_candidate_id")) or _clean(record.get("candidate_id"))
        skills, endorsed = self._skills(record)
        title = self._job_title(record)
        current = self._current_position(record)
        return CandidateProfile(
            candidate_id=override
            or _clean(record.get("publicIdentifier"))
            or f"A{index + 1:03d}",
            job_title=title,
            skills=skills,
            education=self._education(record),
            summary=_clean(record.get("about")),
            responsibilities=self._responsibilities(record),
            location=self._location(record),
            years_experience=self._years_experience(record),
            seniority=_infer_seniority(title),
            employment_type=_map_employment(
                (current or {}).get("employmentType") if current else None
            ),
            positions=self._positions(record),
            employers=self._employers(record),
            languages=self._languages(record),
            certifications=self._certifications(record),
            endorsed_skills=endorsed,
            open_to_work=self._bool(record.get("openToWork")),
            verified=self._bool(record.get("verified")),
            follower_count=self._int(record.get("followerCount")),
            connections_count=self._int(record.get("connectionsCount")),
            source=self.source_name,
            raw=record,
        )
