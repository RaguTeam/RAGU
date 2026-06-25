"""
Tests for per-instance tokenizer/context-length overrides on BaseEngine.

These cover the contract that every search engine inherits from BaseEngine:
constructor parameters ``max_context_length``, ``tokenizer_backend`` and
``tokenizer_model`` take precedence over ``Settings`` when provided, and fall
back to the corresponding ``Settings`` fields when left as ``None``.

A minimal concrete subclass of :class:`BaseEngine` is used so the truncation
wiring can be exercised without constructing a full knowledge graph.
"""
from typing import Any

import pytest

from ragu.common.global_parameters import Settings
from ragu.search_engine.base_engine import BaseEngine, SearchEngineRetrieve


class _DummyEngine(BaseEngine):
    """Minimal concrete engine so BaseEngine.__init__ can be instantiated."""

    async def a_search(self, query, *args: Any, **kwargs: Any) -> SearchEngineRetrieve:
        ...

    async def a_query(self, query: str, *args: Any, **kwargs: Any):
        ...


def test_override_takes_precedence_over_settings():
    engine = _DummyEngine(
        llm=None,
        prompts=[],
        max_context_length=12_345,
        tokenizer_backend="tiktoken",
        tokenizer_model="gpt-4o",
    )

    assert engine.truncation.max_tokens == 12_345
    assert engine.truncation.tokenizer_type == "tiktoken"
    assert engine.truncation.model_id == "gpt-4o"


def test_none_falls_back_to_settings(monkeypatch):
    monkeypatch.setattr(Settings, "tokenizer_llm_name", "gpt-4o")
    monkeypatch.setattr(Settings, "tokenizer_llm_backend", "tiktoken")
    monkeypatch.setattr(Settings, "llm_context_token_limit", 7_777)

    engine = _DummyEngine(llm=None, prompts=[])

    assert engine.truncation.max_tokens == 7_777
    assert engine.truncation.tokenizer_type == "tiktoken"
    assert engine.truncation.model_id == "gpt-4o"


def test_partial_override_mixed_with_settings(monkeypatch):
    monkeypatch.setattr(Settings, "tokenizer_llm_name", "gpt-4o")
    monkeypatch.setattr(Settings, "tokenizer_llm_backend", "tiktoken")
    monkeypatch.setattr(Settings, "llm_context_token_limit", 30_000)

    engine = _DummyEngine(llm=None, prompts=[], max_context_length=999)

    assert engine.truncation.max_tokens == 999
    assert engine.truncation.tokenizer_type == "tiktoken"
    assert engine.truncation.model_id == "gpt-4o"
