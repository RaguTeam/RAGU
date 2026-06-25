"""
Unit tests for the Settings cache_path / debug_errors_path fallback wiring.

These cover the contract that ``CachedAsyncOpenAI``, when constructed without
explicit ``cache`` / ``debug_errors_storage``, falls back to
``Settings.cache_path`` / ``Settings.debug_errors_path``; and that an explicit
value always takes precedence. Both Settings fields default to ``None``
(caching / error capture disabled).

Only the constructor wiring is exercised; no HTTP server is needed.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragu.common.global_parameters import Settings
from ragu.models.openai import CachedAsyncOpenAI


def _dummy_client() -> SimpleNamespace:
    # A truthy placeholder so AsyncOpenAI() is not constructed during the test.
    return SimpleNamespace()


def test_cache_defaults_off_when_settings_none(monkeypatch):
    monkeypatch.setattr(Settings, "_cache_path", None, raising=False)
    monkeypatch.setattr(Settings, "_debug_errors_path", None, raising=False)

    client = CachedAsyncOpenAI(client=_dummy_client())

    assert client.cache is None
    assert client.debug_errors_storage is None


def test_cache_falls_back_to_settings(monkeypatch, tmp_path):
    cache_dir = str(tmp_path / "settings_cache")
    monkeypatch.setattr(Settings, "_cache_path", cache_dir, raising=False)

    client = CachedAsyncOpenAI(client=_dummy_client())

    assert client.cache is not None
    assert Path(client.cache.directory).resolve() == Path(cache_dir).resolve()


def test_explicit_cache_overrides_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, "_cache_path", str(tmp_path / "unused"), raising=False)

    in_memory: dict = {}
    client = CachedAsyncOpenAI(client=_dummy_client(), cache=in_memory)

    assert client.cache is in_memory


def test_debug_errors_falls_back_to_settings(monkeypatch, tmp_path):
    debug_dir = str(tmp_path / "settings_debug")
    monkeypatch.setattr(Settings, "_debug_errors_path", debug_dir, raising=False)

    client = CachedAsyncOpenAI(client=_dummy_client())

    assert client.debug_errors_storage is not None
    assert Path(client.debug_errors_storage.directory).resolve() == Path(debug_dir).resolve()


def test_explicit_debug_errors_overrides_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(Settings, "_debug_errors_path", str(tmp_path / "unused"), raising=False)

    storage: dict = {}
    client = CachedAsyncOpenAI(client=_dummy_client(), debug_errors_storage=storage)

    assert client.debug_errors_storage is storage


def test_save_excludes_cache_and_debug_paths(tmp_path):
    # These are properties (not annotated), so they must never appear in save() output.
    Settings.cache_path = str(tmp_path / "some_cache")
    Settings.debug_errors_path = str(tmp_path / "some_debug")
    try:
        out = tmp_path / "settings.json"
        Settings.save(out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert "cache_path" not in data
        assert "debug_errors_path" not in data
    finally:
        Settings.cache_path = None
        Settings.debug_errors_path = None
