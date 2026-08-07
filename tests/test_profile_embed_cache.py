"""Per-candidate in-memory profile embedding cache."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from core.embedding import (
    clear_profile_embed_cache,
    embed_profiles,
    profile_embed_cache_size,
    _profile_embed_cache_key,
)
from models.candidate import CandidateProfile


@pytest.fixture(autouse=True)
def _isolated_cache():
    clear_profile_embed_cache()
    yield
    clear_profile_embed_cache()


def _profile(cid: str, summary: str) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=cid,
        job_title="HR Assistant",
        summary=summary,
        skills=["recruitment", "payroll"],
        responsibilities="Handled onboarding",
    )


def _mock_model(dim: int = 8):
    model = MagicMock()
    model.max_seq_length = 384
    model.tokenizer = None  # skip truncate/log_truncation tokenizer path

    def _encode(texts, **_kwargs):
        # Distinct vectors per text content so cache correctness is observable.
        out = []
        for t in texts:
            seed = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(seed)
            out.append(rng.standard_normal(dim).astype(np.float32))
        return np.stack(out, axis=0)

    model.encode.side_effect = _encode
    return model


def test_cache_key_uses_sha256_of_model_doc_and_text():
    import hashlib

    key = _profile_embed_cache_key("mpnet", "instr:", "hello")
    expected = hashlib.sha256(b"mpnet|instr:|hello").hexdigest()
    assert key == expected


def test_second_embed_is_full_cache_hit_and_skips_encode():
    model = _mock_model()
    profiles = [_profile("a", "About A"), _profile("b", "About B")]

    embed_profiles(profiles, model, model_key="all-mpnet-base-v2", batch_size=2)
    assert model.encode.call_count == 1
    assert profile_embed_cache_size() == 2
    emb_a = list(profiles[0].profile_embedding)
    emb_b = list(profiles[1].profile_embedding)

    # Fresh profile objects, same content → full hit
    profiles2 = [_profile("a", "About A"), _profile("b", "About B")]
    embed_profiles(profiles2, model, model_key="all-mpnet-base-v2", batch_size=2)
    assert model.encode.call_count == 1  # no second encode
    assert profiles2[0].profile_embedding == emb_a
    assert profiles2[1].profile_embedding == emb_b


def test_partial_miss_encodes_only_new_profiles():
    model = _mock_model()
    embed_profiles(
        [_profile("a", "About A")],
        model,
        model_key="all-mpnet-base-v2",
        batch_size=2,
    )
    assert model.encode.call_count == 1
    first_batch = model.encode.call_args_list[0].args[0]
    assert len(first_batch) == 1

    mixed = [_profile("a", "About A"), _profile("c", "About C")]
    embed_profiles(mixed, model, model_key="all-mpnet-base-v2", batch_size=2)
    assert model.encode.call_count == 2
    second_batch = model.encode.call_args_list[1].args[0]
    assert len(second_batch) == 1  # only the miss
    assert mixed[0].profile_embedding is not None
    assert mixed[1].profile_embedding is not None


def test_model_key_change_is_a_cache_miss():
    model = _mock_model()
    profiles = [_profile("a", "About A")]
    embed_profiles(profiles, model, model_key="model-v1", batch_size=2)
    embed_profiles(profiles, model, model_key="model-v2", batch_size=2)
    assert model.encode.call_count == 2
    assert profile_embed_cache_size() == 2


def test_duplicate_profile_texts_encode_once_per_request():
    model = _mock_model()
    profiles = [
        _profile("a", "Same about"),
        _profile("b", "Same about"),
        _profile("c", "Same about"),
    ]
    embed_profiles(profiles, model, model_key="k", batch_size=2)
    assert model.encode.call_count == 1
    batch = model.encode.call_args_list[0].args[0]
    assert len(batch) == 1  # deduped
    assert profile_embed_cache_size() == 1
    assert profiles[0].profile_embedding == profiles[1].profile_embedding
    assert profiles[1].profile_embedding == profiles[2].profile_embedding


def test_use_memory_cache_false_always_encodes():
    model = _mock_model()
    profiles = [_profile("a", "About A")]
    embed_profiles(
        profiles, model, model_key="k", batch_size=2, use_memory_cache=False
    )
    embed_profiles(
        profiles, model, model_key="k", batch_size=2, use_memory_cache=False
    )
    assert model.encode.call_count == 2
    assert profile_embed_cache_size() == 0
