"""Cross-encoder reranker (Stage 2).

Stage 1 (the bi-encoder pipeline in core/scoring.py) scores and ranks the whole pool.
This module re-scores only the top-K *Head* of that ranking with a cross-encoder that
attends jointly over the (JD, candidate) pair — higher-fidelity head precision than the
bi-encoder's independent embeddings.

Isolated + config-gated, mirroring core.embedding.build_similarity_spec: when
`rerank_model_config` is None the whole stage no-ops and the ranking is exactly Stage 1.
The CE emits a raw relevance logit; we map it to [0,1] with sigmoid(logit / temperature)
so it can occupy the bi-encoder's slot in the weighted fusion (see core.scoring.apply_rerank).
"""

import hashlib
import logging
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RerankSpec:
    """A loaded cross-encoder plus its knobs. Built by build_rerank_model; None disables reranking."""
    model: CrossEncoder
    model_key: str            # disambiguates the on-disk score cache (model + max_length)
    max_length: int
    batch_size: int = 16
    temperature: float = 1.0  # sigmoid(logit / T); T > 1 de-saturates a congested head


def build_rerank_model(config: dict | None) -> "RerankSpec | None":
    """Construct a RerankSpec from a config dict. Returns None when config is falsy
    (reranking disabled → the pipeline is exactly Stage 1).

    config keys: model_name, max_length, dtype (auto|fp32|fp16|bf16), device
    (None = auto), batch_size, temperature. fp16 is applied post-load via .half()
    (the MPS-friendly path, mirroring build_similarity_spec); some custom encoders NaN
    on Apple MPS → pin device='cpu' (+ dtype fp32).
    """
    if not config:
        return None
    name = config["model_name"]
    max_length = int(config.get("max_length", 1024))
    init_kwargs: dict = {"trust_remote_code": True, "max_length": max_length}
    device = config.get("device")
    if device:
        init_kwargs["device"] = device
    dtype = config.get("dtype", "auto")
    load_dtype = None
    if dtype == "fp32" or (dtype == "auto" and device == "cpu"):
        load_dtype = torch.float32
    elif dtype == "bf16":
        load_dtype = torch.bfloat16
    if load_dtype is not None:
        init_kwargs["model_kwargs"] = {"torch_dtype": load_dtype}
    model = CrossEncoder(name, **init_kwargs)
    if dtype == "fp16":
        model.model = model.model.half()
    return RerankSpec(
        model=model,
        model_key=f"{name}|L={max_length}",
        max_length=max_length,
        batch_size=int(config.get("batch_size", 16)),
        temperature=float(config.get("temperature", 1.0)),
    )


def sigmoid(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Map raw CE logits to (0,1) via sigmoid(x / T). Monotonic, so it preserves the
    reranker's order; T > 1 spreads a saturated head back into sigmoid's steep zone."""
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float) / max(temperature, 1e-6)))


def rerank_scores(
    spec: RerankSpec,
    jd_text: str,
    candidate_texts: list[str],
    cache_path: str | None = None,
) -> np.ndarray:
    """Score each (jd_text, candidate) pair → [0,1] via CE logit → sigmoid(logit / T).

    Raw logits are cached (a dict keyed on model + JD + candidate texts) so re-runs and
    offline sweeps don't re-invoke the model. sigmoid/temperature are applied on read, so
    temperature can be retuned without recomputing. NaN logits raise (MPS/dtype guard).
    """
    if not candidate_texts:
        return np.array([], dtype=float)
    key = hashlib.md5(
        (f"{spec.model_key}\x1f{jd_text}\x1f" + "||".join(candidate_texts)).encode("utf-8")
    ).hexdigest()

    cache: dict = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
    if key in cache:
        raw = cache[key]
    else:
        pairs = [[jd_text, c] for c in candidate_texts]
        raw = spec.model.predict(
            pairs,
            batch_size=spec.batch_size,
            activation_fn=torch.nn.Identity(),  # force raw logits; we sigmoid ourselves
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        raw = np.asarray(raw, dtype=float).reshape(-1)
        if np.isnan(raw).any():
            raise ValueError(
                "rerank_scores: CrossEncoder produced NaN logits — check device/dtype "
                "(some encoders NaN on Apple MPS; retry rerank_model_config with device='cpu', dtype='fp32')."
            )
        if cache_path:
            cache[key] = raw
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(cache, fh)
    return sigmoid(raw, spec.temperature)
