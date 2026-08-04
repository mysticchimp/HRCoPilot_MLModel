"""Unit tests for process-level model cache + fp16-at-load."""

from unittest.mock import MagicMock, patch

import torch

from core.embedding import _resolve_load_dtype, build_similarity_spec
from core import model_cache


def test_resolve_load_dtype_fp16_at_construction():
    assert _resolve_load_dtype("fp16", None) is torch.float16
    assert _resolve_load_dtype("bf16", None) is torch.bfloat16
    assert _resolve_load_dtype("fp32", None) is torch.float32
    assert _resolve_load_dtype("auto", "cpu") is torch.float32


def test_build_similarity_spec_fp16_uses_torch_dtype_not_half():
    fake = MagicMock()
    with patch("core.embedding.SentenceTransformer", return_value=fake) as ctor:
        spec = build_similarity_spec(
            {
                "model_name": "Qwen/Qwen3-Embedding-0.6B",
                "dtype": "fp16",
                "max_seq_length": 1024,
                "batch_size": 2,
            }
        )
    assert spec is not None
    kwargs = ctor.call_args.kwargs
    assert kwargs["model_kwargs"]["torch_dtype"] is torch.float16
    fake.half.assert_not_called()


def test_truncate_to_max_tokens_respects_model_cap():
    from core.embedding import truncate_to_max_tokens

    class _Tok:
        def encode(self, text, add_special_tokens=True):
            # 1 token per word
            return list(range(len(text.split())))

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"w{i}" for i in ids)

    model = MagicMock()
    model.max_seq_length = 4
    model.tokenizer = _Tok()
    out = truncate_to_max_tokens("a b c d e f g", model)
    assert out == "w0 w1 w2 w3"
    assert truncate_to_max_tokens("a b", model) == "a b"


def test_model_cache_reuses_base_model():
    model_cache._base_models.clear()
    model_cache._sim_specs.clear()
    fake = MagicMock(name="mpnet")
    with patch("core.model_cache.SentenceTransformer", return_value=fake) as ctor:
        a = model_cache.get_base_embedding_model("all-mpnet-base-v2")
        b = model_cache.get_base_embedding_model("all-mpnet-base-v2")
    assert a is b is fake
    assert ctor.call_count == 1
    model_cache._base_models.clear()
