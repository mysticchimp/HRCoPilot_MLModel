import pandas as pd

from core.adapters.base import CandidateAdapter
from models.candidate import CandidateEducation, CandidateProfile
from utils.parsing import parse_list_str


class ResumeAdapter(CandidateAdapter):
    """Adapter for the resume_data.csv schema.

    Behavior-preserving refactor of the candidate processing previously in
    core/data.py: same skill union (skills + related + certification),
    same degree/field lists, same career-objective + responsibilities text.
    """

    source_name = "resume"

    def load(self, source) -> list[dict]:
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            df = pd.read_csv(source, encoding="utf-8-sig")
        # job_position_name carries a BOM prefix in the raw CSV header
        df.columns = df.columns.str.replace("\ufeff", "").str.strip()
        return df.to_dict(orient="records")

    @staticmethod
    def _combine_skills(record: dict) -> list[str]:
        skills = parse_list_str(record.get("skills"))
        skills = skills if isinstance(skills, list) else []
        related = parse_list_str(record.get("related_skils_in_job"))
        related = [x for sub in related if isinstance(sub, list) for x in sub] if isinstance(related, list) else []
        certs = parse_list_str(record.get("certification_skills"))
        certs = [x for sub in certs if isinstance(sub, list) for x in sub] if isinstance(certs, list) else []
        return list(set(skills + related + certs))

    @staticmethod
    def _clean_field(value) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return None if cleaned == "" or cleaned.upper() == "N/A" else cleaned

    @staticmethod
    def _education(record: dict) -> list[CandidateEducation]:
        degrees = parse_list_str(record.get("degree_names"))
        fields = parse_list_str(record.get("major_field_of_studies"))
        degrees = degrees if isinstance(degrees, list) else []
        fields = fields if isinstance(fields, list) else []
        n = max(len(degrees), len(fields))
        education = []
        for i in range(n):
            raw_degree = degrees[i] if i < len(degrees) else None
            raw_field = fields[i] if i < len(fields) else None
            education.append(
                CandidateEducation(
                    degree=raw_degree if isinstance(raw_degree, str) and raw_degree.strip() else None,
                    field=ResumeAdapter._clean_field(raw_field),
                )
            )
        return education

    @staticmethod
    def _clean_text(value) -> str | None:
        return value if isinstance(value, str) and value.strip() else None

    def to_profile(self, record: dict, index: int) -> CandidateProfile:
        return CandidateProfile(
            candidate_id=f"C{str(index + 1).zfill(3)}",
            # NOTE: resume_data's job_position_name is the applied-role; the current
            # pipeline treats it as the candidate's title, preserved here for parity.
            job_title=str(record.get("job_position_name") or "").strip(),
            skills=self._combine_skills(record),
            education=self._education(record),
            summary=self._clean_text(record.get("career_objective")),
            responsibilities=self._clean_text(record.get("responsibilities")),
            source=self.source_name,
            raw=record,
        )
