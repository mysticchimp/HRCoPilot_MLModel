from core.llm import get_provider
from core.llm.base import LLMProvider
from prompts.jd_extraction import system_prompt, user_prompt
from models.data_models import JobRoleSchema
import json
import os


def process_jd(jd: str, provider: LLMProvider | None = None, cache_path: str | None = None) -> JobRoleSchema:
    """Extract a structured JobRoleSchema from raw JD text.

    If `cache_path` is given and the file exists, the parsed JD is loaded from
    disk (no LLM call) — so you can tweak the cached JSON's properties and re-run
    the pipeline without re-extracting. On a cache miss the JD is extracted and
    written to `cache_path` for reuse.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            return JobRoleSchema.model_validate(json.load(fh))

    if not jd.strip():
        raise ValueError("No Job description provided")
    provider = provider or get_provider()
    parsed = provider.generate_structured(
        user_prompt.format(job_desc=jd),
        JobRoleSchema,
        system=system_prompt,
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(parsed.model_dump(mode="json"), fh, indent=2, ensure_ascii=False)
    return parsed


def load_sample_jd(path: str = './jd/sample_jd_01.txt') -> str:
    with open(path, 'r') as file:
        job_desc = file.read()
    return job_desc
