from __future__ import annotations

import builtins
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
from openai import AsyncOpenAI
import pytest

import ragu.embedder.openai_embedder as openai_module
from ragu.embedder.local_embedders import STEmbedder
from ragu.embedder.openai_embedder import OpenAIEmbedder


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.create = AsyncMock()


class _FakeAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.embeddings = _FakeEmbeddingsAPI()
        self.close = AsyncMock()


class _FakeRunner:
    def __init__(self, semaphore, rps_limiter, rpm_limiter, progress_bar):
        self.progress_bar = progress_bar

    async def make_request(self, func, **kwargs):
        try:
            return await func(**kwargs)
        finally:
            self.progress_bar.update(1)


def _build_embedder(monkeypatch, **kwargs) -> OpenAIEmbedder:
    monkeypatch.setattr(openai_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    return OpenAIEmbedder(
        model_name="text-embedding-3-small",
        client=AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="test-token",
        ),
        dim=5,
        **kwargs,
    )


def test_openai_embedder_init_sets_core_fields(monkeypatch):
    embedder = _build_embedder(monkeypatch, concurrency=2, max_requests_per_second=5, max_requests_per_minute=60)

    assert embedder.model_name == "text-embedding-3-small"
    assert embedder.dim == 5
    assert embedder.client is not None
    assert embedder._sem is not None
    assert embedder._rps is not None
    assert embedder._rpm is not None


def test_openai_embedder_init_without_rate_limiters(monkeypatch):
    embedder = _build_embedder(monkeypatch, max_requests_per_second=0, max_requests_per_minute=0)
    assert embedder._rps is None
    assert embedder._rpm is None


@pytest.mark.asyncio
async def test_openai_one_call_returns_embedding(monkeypatch):
    embedder = _build_embedder(monkeypatch)
    embedder.client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])]
    )

    result = await embedder._embed_via_api("hello")

    assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
    embedder.client.embeddings.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_embed_returns_cached_values_without_generation(monkeypatch):
    embedder = _build_embedder(monkeypatch)
    embedder._cache.get = AsyncMock(side_effect=[[1.0] * 5, [2.0] * 5])
    embedder._embed_via_api = AsyncMock()

    result = await embedder.embed(["a", "b"])

    assert result == [[1.0] * 5, [2.0] * 5]
    embedder._embed_via_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_embed_generates_missing_embeddings_preserving_order(monkeypatch):
    monkeypatch.setattr(openai_module, "AsyncRunner", _FakeRunner)
    embedder = _build_embedder(monkeypatch)
    embedder._cache.get = AsyncMock(return_value=None)
    embedder._embed_via_api = AsyncMock(side_effect=[[0.1] * 5, [0.2] * 5])

    result = await embedder.embed(["first", "second"])

    assert result == [[0.1] * 5, [0.2] * 5]
    assert embedder._embed_via_api.await_count == 2


@pytest.mark.asyncio
async def test_openai_embed_sets_none_for_failed_requests(monkeypatch):
    monkeypatch.setattr(openai_module, "AsyncRunner", _FakeRunner)
    embedder = _build_embedder(monkeypatch)
    embedder._cache.get = AsyncMock(return_value=None)
    embedder._embed_via_api = AsyncMock(side_effect=[RuntimeError("api failure"), [0.7] * 5])

    result = await embedder.embed(["bad", "good"])

    assert result == [None, [0.7] * 5]


@pytest.mark.asyncio
async def test_openai_embed_writes_cache_when_enabled(monkeypatch):
    monkeypatch.setattr(openai_module, "AsyncRunner", _FakeRunner)
    embedder = _build_embedder(monkeypatch, use_cache=True)
    embedder._cache.get = AsyncMock(return_value=None)
    embedder._cache.set = AsyncMock()
    embedder._cache.flush_cache = AsyncMock()
    embedder._embed_via_api = AsyncMock(side_effect=[[0.3] * 5, [0.4] * 5])

    result = await embedder.embed(["x", "y"])

    assert result == [[0.3] * 5, [0.4] * 5]
    assert embedder._cache.set.await_count == 2
    assert embedder._cache.flush_cache.await_count == 1


@pytest.mark.asyncio
async def test_openai_embed_accepts_single_string(monkeypatch):
    monkeypatch.setattr(openai_module, "AsyncRunner", _FakeRunner)
    embedder = _build_embedder(monkeypatch)
    embedder._cache.get = AsyncMock(return_value=None)
    embedder._embed_via_api = AsyncMock(return_value=[0.9] * 5)

    result = await embedder.embed_single("single")

    assert result == [[0.9] * 5]


@pytest.mark.asyncio
async def test_openai_embedder_aclose_closes_cache_and_client(monkeypatch):
    embedder = _build_embedder(monkeypatch)
    embedder._cache.close = AsyncMock()
    embedder.client.close = AsyncMock()

    await embedder.aclose()

    embedder._cache.close.assert_awaited_once()
    embedder.client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_embedder_aclose_swallows_exceptions(monkeypatch):
    embedder = _build_embedder(monkeypatch)
    embedder._cache.close = AsyncMock(side_effect=RuntimeError("close failed"))

    await embedder.aclose()

#
# class _FakeSentenceTransformer:
#     def __init__(self, model_name_or_path: str, **kwargs):
#         self.model_name_or_path = model_name_or_path
#         self.kwargs = kwargs
#         self.encode_calls = []
#
#     def get_sentence_embedding_dimension(self) -> int:
#         return 4
#
#     def encode(self, batch, show_progress_bar: bool = False):
#         self.encode_calls.append((list(batch), show_progress_bar))
#         return np.array(
#             [[float(len(text)), float(len(text) + 1), 1.0, 0.0] for text in batch],
#             dtype=float,
#         )
#
#
# def test_st_embedder_raises_import_error_when_sentence_transformers_missing(monkeypatch):
#     original_import = builtins.__import__
#
#     def fake_import(name, *args, **kwargs):
#         if name == "sentence_transformers":
#             raise ImportError("missing package")
#         return original_import(name, *args, **kwargs)
#
#     monkeypatch.setattr(builtins, "__import__", fake_import)
#
#     with pytest.raises(ImportError):
#         STEmbedder(model_name_or_path="fake-model")
#
#
# def test_st_embedder_init_uses_model_dimension_when_dim_not_passed(monkeypatch):
#     fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
#     monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
#
#     embedder = STEmbedder(model_name_or_path="fake-model", device="cpu")
#
#     assert embedder.dim == 4
#     assert embedder.model.model_name_or_path == "fake-model"
#     assert embedder.model.kwargs["device"] == "cpu"
#
#
# def test_st_embedder_init_respects_explicit_dim_override(monkeypatch):
#     fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
#     monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
#
#     embedder = STEmbedder(model_name_or_path="fake-model", dim=128)
#
#     assert embedder.dim == 128
#
#
# @pytest.mark.asyncio
# async def test_st_embedder_embed_single_string(monkeypatch):
#     fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
#     monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
#     embedder = STEmbedder(model_name_or_path="fake-model")
#
#     result = await embedder.embed("hello")
#
#     assert isinstance(result, np.ndarray)
#     assert result.shape == (1, 4)
#     assert embedder.model.encode_calls == [(["hello"], False)]
#
#
# @pytest.mark.asyncio
# async def test_st_embedder_embed_list_uses_batching(monkeypatch):
#     fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
#     monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
#     embedder = STEmbedder(model_name_or_path="fake-model")
#
#     texts = ["a", "bb", "ccc", "dddd", "eeeee"]
#     result = await embedder.embed(texts, batch_size=2)
#
#     assert isinstance(result, np.ndarray)
#     assert result.shape == (5, 4)
#     assert len(embedder.model.encode_calls) == 3
#     assert embedder.model.encode_calls[0] == (["a", "bb"], False)
#     assert embedder.model.encode_calls[1] == (["ccc", "dddd"], False)
#     assert embedder.model.encode_calls[2] == (["eeeee"], False)
#
#
# @pytest.mark.asyncio
# async def test_st_embedder_call_delegates_to_embed(monkeypatch):
#     fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
#     monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
#     embedder = STEmbedder(model_name_or_path="fake-model")
#
#     result = await embedder("call path")
#
#     assert isinstance(result, np.ndarray)
#     assert result.shape == (1, 4)
#
