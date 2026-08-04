"""Process-level embedding model cache for the scoring service.

Loads all-mpnet (title/skill) and the optional isolated Qwen similarity encoder
once per process and reuses them across ``/score`` requests. Avoids the ~19s +
RSS spike from reconstructing SentenceTransformers on every call.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from sentence_transformers import SentenceTransformer

from core.embedding import SimilaritySpec, _resolve_load_dtype, build_similarity_spec
from core.reranking import RerankSpec, build_rerank_model
from models.mappings import base_embedding_dtype as DEFAULT_BASE_DTYPE
from models.mappings import rerank_model_config as CHAMPION_RERANK_CONFIG
from models.mappings import similarity_model_config as CHAMPION_SIM_CONFIG

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_base_models: dict[tuple[str, str], SentenceTransformer] = {}
_sim_specs: dict[tuple, Optional[SimilaritySpec]] = {}
_rerank_specs: dict[tuple, Optional[RerankSpec]] = {}


def resolve_similarity_model_config(
    config: dict | None = CHAMPION_SIM_CONFIG,
) -> dict | None:
    """Apply ``SIMILARITY_MODEL`` env override on top of the champion config.

    - unset / ``qwen`` / ``champion`` → ``config`` (default: Qwen champion dict)
    - ``mpnet-only`` / ``mpnet`` / ``none`` → ``None`` (similarity uses base mpnet)
    """
    mode = (os.environ.get("SIMILARITY_MODEL") or "qwen").strip().lower()
    if mode in ("mpnet-only", "mpnet", "none", "off", "base"):
        return None
    if mode in ("qwen", "champion", "default"):
        return config
    logger.warning("Unknown SIMILARITY_MODEL=%r; using champion config", mode)
    return config


def resolve_base_embedding_dtype(default: str | None = None) -> str:
    """``BASE_EMBEDDING_DTYPE`` env, else mappings default (fp16)."""
    return (os.environ.get("BASE_EMBEDDING_DTYPE") or default or DEFAULT_BASE_DTYPE).strip().lower()


def _sim_cache_key(config: dict | None) -> tuple:
    if not config:
        return ("__none__",)
    return (
        config.get("model_name"),
        config.get("dtype"),
        config.get("device"),
        config.get("max_seq_length"),
        config.get("query_instruction"),
        config.get("doc_instruction"),
        config.get("batch_size"),
    )


def _rerank_cache_key(config: dict | None) -> tuple:
    if not config:
        return ("__none__",)
    return (
        config.get("model_name"),
        config.get("dtype"),
        config.get("device"),
        config.get("max_length"),
        config.get("batch_size"),
        config.get("temperature"),
    )


def get_base_embedding_model(
    name: str = "all-mpnet-base-v2",
    dtype: str | None = None,
) -> SentenceTransformer:
    """Return the shared title/skill encoder (loaded once per process + dtype)."""
    resolved_dtype = resolve_base_embedding_dtype(dtype)
    cache_key = (name, resolved_dtype)
    with _lock:
        model = _base_models.get(cache_key)
        if model is None:
            load_kwargs: dict = {}
            load_dtype = _resolve_load_dtype(resolved_dtype, None)
            if load_dtype is not None:
                load_kwargs["model_kwargs"] = {"torch_dtype": load_dtype}
            logger.info(
                "Loading base embedding model once: %s dtype=%s",
                name,
                resolved_dtype,
            )
            model = SentenceTransformer(name, **load_kwargs)
            _base_models[cache_key] = model
        return model


def get_similarity_spec(
    config: dict | None = CHAMPION_SIM_CONFIG,
    base_model: SentenceTransformer | None = None,
) -> SimilaritySpec | None:
    """Return the shared SimilaritySpec (loaded once per distinct config)."""
    key = _sim_cache_key(config)
    with _lock:
        if key in _sim_specs:
            return _sim_specs[key]
        base = base_model or get_base_embedding_model()
        # Release lock while loading heavy weights? Keep it held so concurrent
        # warmers don't double-load the same key.
        logger.info(
            "Loading similarity model once: %s",
            "(base/mpnet-only)" if not config else (config or {}).get("model_name", "(base)"),
        )
        spec = build_similarity_spec(config, base_model=base)
        _sim_specs[key] = spec
        return spec


def get_rerank_spec(config: dict | None = CHAMPION_RERANK_CONFIG) -> RerankSpec | None:
    """Return the shared rerank spec (usually None / champion OFF)."""
    key = _rerank_cache_key(config)
    with _lock:
        if key in _rerank_specs:
            return _rerank_specs[key]
        spec = build_rerank_model(config)
        _rerank_specs[key] = spec
        return spec


def _prime_encoder(model: SentenceTransformer, texts: list[str], batch_size: int) -> None:
    """Run one throwaway encode so first-request allocator/workspace is paid at warm."""
    model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=max(1, int(batch_size)),
        show_progress_bar=False,
    )


def warm_scoring_models(
    embedding_model_name: str = "all-mpnet-base-v2",
    similarity_model_config: dict | None = CHAMPION_SIM_CONFIG,
    rerank_model_config: dict | None = CHAMPION_RERANK_CONFIG,
    apply_env_overrides: bool = True,
) -> tuple[SentenceTransformer, SimilaritySpec | None, RerankSpec | None]:
    """Eagerly load champion models (call from FastAPI lifespan / startup).

    When ``apply_env_overrides`` is True (default), honors ``SIMILARITY_MODEL`` and
    ``BASE_EMBEDDING_DTYPE``. Pass False to force an explicit config for A/B scripts.
    """
    from models.mappings import encode_batch_size

    if apply_env_overrides:
        similarity_model_config = resolve_similarity_model_config(similarity_model_config)

    base = get_base_embedding_model(embedding_model_name)
    sim = get_similarity_spec(similarity_model_config, base_model=base)
    rerank = get_rerank_spec(rerank_model_config)
    # Prime encoders: weights alone under-report RSS until the first forward
    # allocates native workspace (esp. on CPU). Move that spike off /score.
    _prime_encoder(base, ["HR Assistant", "talent acquisition"], encode_batch_size)
    if sim is not None and sim.model is not base:
        # Two mid-length docs ≈ production micro-batch; pad-ish to exercise L-cap path.
        longish = ("candidate profile experience skills education. " * 40).strip()
        _prime_encoder(sim.model, [longish, longish], sim.batch_size)
    logger.info(
        "Scoring models ready (base=%s dtype=%s, sim=%s, rerank=%s)",
        embedding_model_name,
        resolve_base_embedding_dtype(),
        None if sim is None else sim.model_key,
        None if rerank is None else rerank.model_key,
    )
    return base, sim, rerank
