import math
import re
from typing import Optional

import pandas as pd

from core.adapters.base import CandidateAdapter
from models.candidate import (
    CandidateEducation,
    CandidateLanguage,
    CandidateLocation,
    CandidatePosition,
    CandidateProfile,
)


def _clean(value):
    """Normalize a raw flattened-CSV cell to a value or None (drops NaN/empty)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _duration_to_months(text) -> int:
    """Parse a LinkedIn duration string like '3 yrs 10 mos' into months."""
    if not isinstance(text, str):
        return 0
    years = re.search(r"(\d+)\s*yr", text)
    months = re.search(r"(\d+)\s*mo", text)
    return (int(years.group(1)) * 12 if years else 0) + (int(months.group(1)) if months else 0)


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _parse_month_year(text) -> Optional[tuple[int, int]]:
    """Parse a raw date like 'Apr 2025' into (year, month); year-only -> (year, 1)."""
    if not isinstance(text, str):
        return None
    m = re.search(r"([A-Za-z]{3})[a-z]*\s+(\d{4})", text)
    if m and m.group(1).lower() in _MONTHS:
        return int(m.group(2)), _MONTHS[m.group(1).lower()]
    y = re.search(r"\b(\d{4})\b", text)
    return (int(y.group(1)), 1) if y else None


def _months_between(start, end) -> Optional[int]:
    """Whole months between two raw 'MMM YYYY' dates (fallback when duration is absent)."""
    s, e = _parse_month_year(start), _parse_month_year(end)
    if not s or not e:
        return None
    months = (e[0] - s[0]) * 12 + (e[1] - s[1])
    return months if months >= 0 else None


# ordered keyword -> seniority rules (first match wins)
_SENIORITY_RULES = [
    (("chief", "ceo", "cfo", "cto", "coo", "cmo", "cxo"), "c_level"),
    (("director", "vice president", "vp", "head of"), "executive"),
    (("senior", "lead", "principal", "manager"), "senior"),
    (("assistant", "coordinator", "junior", "intern", "trainee", "entry"), "entry"),
]


def _infer_seniority(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    lowered = title.lower()
    for keywords, level in _SENIORITY_RULES:
        # word-boundary match so short abbreviations (e.g. "coo") don't match
        # inside longer words (e.g. "coordinator")
        if any(re.search(rf"\b{re.escape(k)}\b", lowered) for k in keywords):
            return level
    return "mid"


_EMPLOYMENT_MAP = {
    "full-time": "full_time",
    "part-time": "part_time",
    "contract": "contract",
    "temporary": "contract",
    "freelance": "contract",
    "self-employed": "contract",
    "internship": "internship",
}


def _map_employment(value) -> Optional[str]:
    cleaned = _clean(value)
    if not isinstance(cleaned, str):
        return None
    return _EMPLOYMENT_MAP.get(cleaned.lower())


class LinkedInAdapter(CandidateAdapter):
    """Adapter for the LinkedIn profile-export schema (flattened JSON columns).

    Maps the wide `prefix/<index>/subfield` columns into canonical
    CandidateProfile objects, deriving years_experience/seniority and populating
    LinkedIn-only enrichment (open_to_work, verified, follower/connection counts).
    """

    source_name = "linkedin"

    # per-entity max array lengths observed in the dataset (safe upper bounds)
    _MAX = {"skills": 60, "education": 6, "experience": 15, "languages": 6, "certifications": 25}

    def load(self, source) -> list[dict]:
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            df = pd.read_csv(source, low_memory=False)
        return df.to_dict(orient="records")

    @staticmethod
    def _collect(record: dict, prefix: str, subfields: list[str], max_n: int) -> list[dict]:
        """Collect array items `prefix/i/subfield` into a list of dicts (non-empty only)."""
        items = []
        for i in range(max_n):
            obj = {sf: _clean(record.get(f"{prefix}/{i}/{sf}")) for sf in subfields}
            if any(v is not None for v in obj.values()):
                items.append(obj)
        return items

    def _skills(self, record: dict) -> tuple[list[str], Optional[dict]]:
        objs = self._collect(record, "skills", ["name", "endorsements"], self._MAX["skills"])
        names = [o["name"] for o in objs if o["name"]]
        endorsed = {}
        for o in objs:
            if o["name"] and o["endorsements"]:
                match = re.search(r"(\d+)", str(o["endorsements"]))
                if match:
                    endorsed[o["name"]] = int(match.group(1))
        # union LinkedIn's curated topSkills
        for i in range(5):
            top = _clean(record.get(f"topSkills/{i}"))
            if top and top not in names:
                names.append(top)
        return names, (endorsed or None)

    def _education(self, record: dict) -> list[CandidateEducation]:
        objs = self._collect(
            record, "education", ["degree", "fieldOfStudy", "schoolName", "endDate/year"], self._MAX["education"]
        )
        education = []
        for o in objs:
            raw_year = o["endDate/year"]
            try:
                year = int(float(raw_year)) if raw_year is not None else None
            except (ValueError, TypeError):
                year = None
            education.append(
                CandidateEducation(degree=o["degree"], field=o["fieldOfStudy"], school=o["schoolName"], end_year=year)
            )
        return education

    def _responsibilities(self, record: dict) -> Optional[str]:
        descriptions = self._collect(record, "experience", ["description"], self._MAX["experience"])
        texts = [o["description"] for o in descriptions if o["description"]]
        current = _clean(record.get("currentPosition/0/description"))
        if current and current not in texts:
            texts.insert(0, current)
        return " ".join(texts) if texts else None

    def _years_experience(self, record: dict) -> Optional[float]:
        durations = self._collect(record, "experience", ["duration"], self._MAX["experience"])
        total_months = sum(_duration_to_months(o["duration"]) for o in durations)
        return round(total_months / 12, 1) if total_months > 0 else None

    def _positions(self, record: dict) -> list[CandidatePosition]:
        """Per-role work history from experience/N/* (source order = most-recent-first).

        Keeps the structure `_years_experience` sums away. `organizations/N/*` is
        ignored (empty in this export). Tenure prefers the duration string, falling
        back to the span between raw start/end dates.
        """
        objs = self._collect(
            record, "experience",
            ["position", "companyName", "duration", "employmentType", "startDate/text", "endDate/text"],
            self._MAX["experience"],
        )
        positions = []
        for o in objs:
            end = o["endDate/text"]
            is_current = end is None or (isinstance(end, str) and end.strip().lower() in ("", "present"))
            tenure = _duration_to_months(o["duration"]) or None
            if tenure is None and not is_current:
                tenure = _months_between(o["startDate/text"], end)
            positions.append(CandidatePosition(
                title=o["position"],
                company=o["companyName"],
                start=o["startDate/text"],
                end=end,
                tenure_months=tenure,
                is_current=is_current,
                employment_type=o["employmentType"],
            ))
        return positions

    def _location(self, record: dict) -> Optional[CandidateLocation]:
        location = CandidateLocation(
            text=_clean(record.get("location/parsed/text")) or _clean(record.get("location/linkedinText")),
            city=_clean(record.get("location/parsed/city")),
            state=_clean(record.get("location/parsed/state")),
            country=_clean(record.get("location/parsed/country")),
            country_code=_clean(record.get("location/parsed/countryCode")) or _clean(record.get("location/countryCode")),
        )
        return location if any(location.model_dump().values()) else None

    def _languages(self, record: dict) -> Optional[list[CandidateLanguage]]:
        objs = self._collect(record, "languages", ["name", "proficiency"], self._MAX["languages"])
        langs = [CandidateLanguage(language=o["name"], proficiency=o["proficiency"]) for o in objs if o["name"]]
        return langs or None

    def _certifications(self, record: dict) -> Optional[list[str]]:
        objs = self._collect(record, "certifications", ["title"], self._MAX["certifications"])
        certs = [o["title"] for o in objs if o["title"]]
        return certs or None

    def _employers(self, record: dict) -> Optional[list[str]]:
        """Company names (current + past) — the strongest raw sector signal."""
        names = []
        current = _clean(record.get("currentPosition/0/companyName"))
        if current:
            names.append(current)
        for obj in self._collect(record, "experience", ["companyName"], self._MAX["experience"]):
            name = obj["companyName"]
            if name and name not in names:
                names.append(name)
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
        title = _clean(record.get("currentPosition/0/position"))
        if title:
            return title
        headline = _clean(record.get("headline"))
        if headline:
            return headline.split("|")[0].strip()
        return ""

    def to_profile(self, record: dict, index: int) -> CandidateProfile:
        skills, endorsed = self._skills(record)
        title = self._job_title(record)
        return CandidateProfile(
            candidate_id=_clean(record.get("publicIdentifier")) or f"L{index + 1:03d}",
            job_title=title,
            skills=skills,
            education=self._education(record),
            summary=_clean(record.get("about")),
            responsibilities=self._responsibilities(record),
            location=self._location(record),
            years_experience=self._years_experience(record),
            seniority=_infer_seniority(title),
            employment_type=_map_employment(record.get("currentPosition/0/employmentType")),
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
