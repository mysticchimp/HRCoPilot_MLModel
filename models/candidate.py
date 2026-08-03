from typing import Any, Optional

from pydantic import BaseModel, Field

from models.data_models import EmploymentType, SeniorityLevel


class CandidateLocation(BaseModel):
    text: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None


class CandidateEducation(BaseModel):
    degree: Optional[str] = Field(None, description="Raw degree string, e.g. 'Master of Science - MSc'")
    field: Optional[str] = Field(None, description="Field of study, e.g. 'Computer Science'")
    school: Optional[str] = None
    end_year: Optional[int] = None


class CandidateLanguage(BaseModel):
    language: str
    proficiency: Optional[str] = Field(None, description="Free-text level, e.g. 'Native or bilingual proficiency'")


class CandidatePosition(BaseModel):
    """A single role in a candidate's work history (per-role tenure structure).

    Populated from the raw export's per-role fields (the LinkedIn adapter keeps the
    structure its `years_experience` sum discards). Powers tenure/attrition and
    relevant-vs-adjacent experience scoring. `employment_type` is the RAW source
    string (e.g. 'Full-time', 'Permanent', 'Contract') so the perm-vs-contract
    split lives with the scorer that uses it.
    """

    title: Optional[str] = None
    company: Optional[str] = None
    start: Optional[str] = Field(None, description="Raw start text, e.g. 'Apr 2025'")
    end: Optional[str] = Field(None, description="Raw end text; None/'Present' => current")
    tenure_months: Optional[int] = Field(None, description="Parsed from duration, else end-start")
    is_current: bool = False
    employment_type: Optional[str] = Field(None, description="Raw source value, e.g. 'Full-time'/'Permanent'/'Contract'")


class CandidateProfile(BaseModel):
    """Canonical, dataset-agnostic candidate representation.

    Every adapter (resume CSV, LinkedIn export, ...) maps its source schema into
    this model. The scoring pipeline reads ONLY these fields, never raw columns,
    so adding a data source == adding an adapter (no scorer changes).

    Field tiers:
      * core       -> present in every source; drive the existing scorers.
      * enrichment -> optional signals. A knob keyed on one of these MUST no-op
                      (and renormalize away) when the value is None, so sources
                      that lack the field stay backward compatible.
      * derived    -> computed downstream (embedding input / vector), not by the
                      adapter.
    """

    # --- identity / core (expected from every source) ---
    candidate_id: str
    job_title: str = Field(..., description="Candidate's current / most-recent title")
    skills: list[str] = Field(default_factory=list)
    education: list[CandidateEducation] = Field(default_factory=list)
    summary: Optional[str] = Field(None, description="Career objective / About text")
    responsibilities: Optional[str] = Field(None, description="Current/recent role responsibilities text")
    location: Optional[CandidateLocation] = None

    # --- enrichment (optional; knobs no-op when None) ---
    years_experience: Optional[float] = None
    seniority: Optional[SeniorityLevel] = None
    employment_type: Optional[EmploymentType] = None
    positions: list[CandidatePosition] = Field(
        default_factory=list,
        description="Per-role work history (tenure structure); knobs no-op when empty",
    )
    employers: Optional[list[str]] = Field(None, description="Company names (current + past), for sector inference")
    languages: Optional[list[CandidateLanguage]] = None
    certifications: Optional[list[str]] = None
    endorsed_skills: Optional[dict[str, int]] = Field(None, description="skill name -> endorsement count")
    open_to_work: Optional[bool] = None
    verified: Optional[bool] = None
    follower_count: Optional[int] = None
    connections_count: Optional[int] = None

    # --- derived (filled by the embedding step, not the adapter) ---
    profile_text: Optional[str] = None
    profile_embedding: Optional[list[float]] = None

    # --- provenance / debugging ---
    source: Optional[str] = Field(None, description="Adapter/source name that produced this profile")
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    # Index-aligned convenience views for the current qualification scorer, which
    # consumes parallel {"degrees": [...], "fields": [...]} lists.
    @property
    def degree_names(self) -> list[str]:
        return [edu.degree or "" for edu in self.education]

    @property
    def field_names(self) -> list[str]:
        return [edu.field or "N/A" for edu in self.education]
