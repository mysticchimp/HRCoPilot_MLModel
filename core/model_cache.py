"""Process-level embedding model cache for the scoring service.

Loads all-mpnet (title/skill) and the isolated Qwen similarity encoder once per
process and reuses them across ``/score`` requests. Avoids the ~19s + RSS spike
from reconstructing SentenceTransformers on every call.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from sentence_transformers import SentenceTransformer

from core.embedding import SimilaritySpec, build_similarity_spec
from core.reranking import RerankSpec, build_rerank_model
from models.mappings import rerank_model_config as CHAMPION_RERANK_CONFIG
from models.mappings import similarity_model_config as CHAMPION_SIM_CONFIG

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_base_models: dict[str, SentenceTransformer] = {}
_sim_specs: dict[tuple, Optional[SimilaritySpec]] = {}
_rerank_specs: dict[tuple, Optional[RerankSpec]] = {}


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


def get_base_embedding_model(name: str = "all-mpnet-base-v2") -> SentenceTransformer:
    """Return the shared title/skill encoder (loaded once per process)."""
    with _lock:
        model = _base_models.get(name)
        if model is None:
            logger.info("Loading base embedding model once: %s", name)
            model = SentenceTransformer(name)
            _base_models[name] = model
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
            (config or {}).get("model_name", "(base)"),
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


def warm_scoring_models(
    embedding_model_name: str = "all-mpnet-base-v2",
    similarity_model_config: dict | None = CHAMPION_SIM_CONFIG,
    rerank_model_config: dict | None = CHAMPION_RERANK_CONFIG,
) -> tuple[SentenceTransformer, SimilaritySpec | None, RerankSpec | None]:
    """Eagerly load champion models (call from FastAPI lifespan / startup)."""
    base = get_base_embedding_model(embedding_model_name)
    sim = get_similarity_spec(similarity_model_config, base_model=base)
    rerank = get_rerank_spec(rerank_model_config)
    logger.info(
        "Scoring models ready (base=%s, sim=%s, rerank=%s)",
        embedding_model_name,
        None if sim is None else sim.model_key,
        None if rerank is None else rerank.model_key,
    )
    return base, sim, rerank
