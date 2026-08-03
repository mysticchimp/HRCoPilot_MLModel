from core.llm import get_provider
from core.llm.base import LLMProvider
from prompts.email_generation import system_prompt, user_prompt


def generate_email(jd: str, candidate_json_str: str, provider: LLMProvider | None = None) -> str:
    provider = provider or get_provider()
    return provider.generate_text(
        user_prompt.format(job_desc=jd, selected_candidate=candidate_json_str),
        system=system_prompt,
        temperature=0.4,
    )