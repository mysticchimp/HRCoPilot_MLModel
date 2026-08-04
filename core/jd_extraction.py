from core.llm import get_provider
from core.llm.base import LLMProvider
from prompts.jd_extraction import system_prompt, user_prompt
from models.data_models import JobRoleSchema
import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default on-disk cache dir (hash-keyed). Override with JD_CACHE_DIR.
# Single-file JD_CACHE_PATH remains supported for local offline eval.
DEFAULT_JD_CACHE_DIR = ".ai-recruiter/jd_cache"


def jd_text_hash(jd: str) -> str:
    """Stable content hash for JD cache keys (invalidates when JD text changes)."""
    return hashlib.sha256((jd or "").strip().encode("utf-8")).hexdigest()


def resolve_jd_cache_dir(cache_dir: str | None = None) -> Path | None:
    """Return the hash-keyed cache directory, or None if disabled."""
    raw = cache_dir if cache_dir is not None else os.environ.get("JD_CACHE_DIR")
    if raw is None:
        # Prefer explicit env; fall back to default dir when unset so production
        # warm re-scores skip Claude. Set JD_CACHE_DIR=0 to disable.
        raw = DEFAULT_JD_CACHE_DIR
    if str(raw).strip().lower() in ("", "0", "false", "off", "none"):
        return None
    return Path(raw)


def _hash_cache_path(cache_dir: Path, jd: str) -> Path:
    return cache_dir / f"{jd_text_hash(jd)}.json"


def process_jd(
    jd: str,
    provider: LLMProvider | None = None,
    cache_path: str | None = None,
    cache_dir: str | None = None,
) -> JobRoleSchema:
    """Extract a structured JobRoleSchema from raw JD text.

    Caching (first hit wins):
    1. ``cache_path`` — legacy single-file path (eval / local offline). If the
       file exists it is loaded with **no** hash check (caller owns validity).
    2. Hash-keyed dir (``cache_dir`` or ``$JD_CACHE_DIR``, default
       ``.ai-recruiter/jd_cache``) — key = sha256(stripped jd_text). Changing the
       JD text automatically misses and re-extracts.

    On a successful LLM extract, the result is written to the hash-keyed path
    (when a cache dir is active) and to ``cache_path`` when given.
    """
    if not (jd or "").strip():
        raise ValueError("No Job description provided")

    # 1) Legacy single-file cache (eval harness / explicit path).
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            logger.info("process_jd cache hit (file)=%s", cache_path)
            return JobRoleSchema.model_validate(json.load(fh))

    # 2) Content-hash cache directory.
    resolved_dir = resolve_jd_cache_dir(cache_dir)
    hash_path = _hash_cache_path(resolved_dir, jd) if resolved_dir is not None else None
    if hash_path is not None and hash_path.exists():
        with open(hash_path) as fh:
            logger.info(
                "process_jd cache hit (hash)=%s… path=%s",
                jd_text_hash(jd)[:12],
                hash_path,
            )
            return JobRoleSchema.model_validate(json.load(fh))

    provider = provider or get_provider()
    logger.info(
        "process_jd cache miss — calling LLM (hash=%s…)",
        jd_text_hash(jd)[:12],
    )
    parsed = provider.generate_structured(
        user_prompt.format(job_desc=jd),
        JobRoleSchema,
        system=system_prompt,
    )
    payload = parsed.model_dump(mode="json")

    if hash_path is not None:
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hash_path, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("process_jd wrote hash cache %s", hash_path)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    return parsed


def load_sample_jd(path: str = './jd/sample_jd_01.txt') -> str:
    with open(path, 'r') as file:
        job_desc = file.read()
    return job_desc
