import json

from core.llm import get_provider
from core.llm.base import LLMProvider
from models.candidate import CandidateProfile
from prompts.jd_generation import system_prompt, user_prompt


def generate_jd_from_profile(profile, provider: LLMProvider | None = None) -> str:
    """Generate a synthetic job description tailored to a candidate profile.

    Used by the reverse-matching eval harness: the JD is generated from a seed
    candidate, then the pipeline should rank that candidate highly.
    """
    provider = provider or get_provider()

    if isinstance(profile, CandidateProfile):
        payload = {
            "job_title": profile.job_title,
            "seniority": profile.seniority,
            "years_experience": profile.years_experience,
            "skills": profile.skills,
            "summary": profile.summary,
            "responsibilities": profile.responsibilities,
            "education": [{"degree": e.degree, "field": e.field} for e in profile.education],
        }
        profile_str = json.dumps(payload, indent=2, default=str)
    elif isinstance(profile, dict):
        profile_str = json.dumps(profile, indent=2, default=str)
    else:
        profile_str = str(profile)

    return provider.generate_text(user_prompt.format(profile=profile_str), system=system_prompt)
