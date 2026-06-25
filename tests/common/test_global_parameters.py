"""
Tests for GlobalSettings: Literal-typed backends and save/load serialization.
"""
import json
from pathlib import Path

import pytest
from typing import Literal, get_args, get_origin, get_type_hints

from ragu.common.global_parameters import GlobalSettings, Settings


@pytest.fixture
def isolated_settings():
    """Snapshot and restore Settings to prevent singleton leakage."""
    snapshot = Settings._serializable_dict()
    working_dir_snapshot = Settings._working_dir
    yield Settings
    for name, value in snapshot.items():
        setattr(Settings, name, value)
    Settings._working_dir = working_dir_snapshot


# --- Literal typing ---------------------------------------------------------


def test_tokenizer_backend_fields_are_literal():
    hints = get_type_hints(GlobalSettings)
    assert get_args(hints["tokenizer_embedder_backend"]) == ("tiktoken", "local")
    assert get_args(hints["tokenizer_llm_backend"]) == ("tiktoken", "local")
    assert get_origin(hints["tokenizer_embedder_backend"]) is Literal


def test_default_backend_values():
    assert Settings.tokenizer_embedder_backend == "tiktoken"
    assert Settings.tokenizer_llm_backend == "tiktoken"


# --- save / load round-trip -------------------------------------------------


def test_save_writes_all_serializable_fields(isolated_settings, tmp_path):
    out = tmp_path / "settings.json"
    Settings.save(out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data.keys()) == {
        "language",
        "tokenizer_embedder_backend",
        "tokenizer_llm_backend",
        "tokenizer_embedder_name",
        "tokenizer_llm_name",
        "embedder_token_limit",
        "llm_context_token_limit",
    }


def test_save_excludes_storage_folder(isolated_settings, tmp_path):
    out = tmp_path / "settings.json"
    Settings.save(out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "storage_folder" not in data
    assert "_working_dir" not in data
    assert "_current_time" not in data


def test_save_captures_instance_overrides(isolated_settings, tmp_path):
    Settings.embedder_token_limit = 123
    Settings.tokenizer_embedder_backend = "local"
    Settings.language = "russian"

    out = tmp_path / "settings.json"
    Settings.save(out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["embedder_token_limit"] == 123
    assert data["tokenizer_embedder_backend"] == "local"
    assert data["language"] == "russian"


def test_round_trip_restores_values(isolated_settings, tmp_path):
    Settings.embedder_token_limit = 555
    Settings.tokenizer_llm_backend = "local"
    Settings.tokenizer_llm_name = "custom-llm"
    Settings.llm_context_token_limit = 12_000

    out = tmp_path / "settings.json"
    Settings.save(out)

    # Mutate to different values, then reload.
    Settings.embedder_token_limit = 1
    Settings.tokenizer_llm_backend = "tiktoken"
    Settings.tokenizer_llm_name = "gpt-4o"
    Settings.llm_context_token_limit = 30_000

    Settings.load(out)

    assert Settings.embedder_token_limit == 555
    assert Settings.tokenizer_llm_backend == "local"
    assert Settings.tokenizer_llm_name == "custom-llm"
    assert Settings.llm_context_token_limit == 12_000


def test_load_does_not_touch_storage_folder(isolated_settings, tmp_path):
    out = tmp_path / "settings.json"
    Settings.save(out)

    original_dir = Settings._working_dir
    Settings.storage_folder = "/some/other/path"

    Settings.load(out)

    assert Settings._working_dir == "/some/other/path"
    # restore for the fixture's own teardown bookkeeping
    Settings._working_dir = original_dir


# --- validation on load -----------------------------------------------------


def test_load_rejects_invalid_literal_backend(isolated_settings, tmp_path):
    out = tmp_path / "bad.json"
    out.write_text(
        json.dumps({"tokenizer_embedder_backend": "fasttoken"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tokenizer_embedder_backend"):
        Settings.load(out)


def test_load_rejects_non_positive_int(isolated_settings, tmp_path):
    out = tmp_path / "bad.json"
    out.write_text(
        json.dumps({"embedder_token_limit": 0}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="positive integer"):
        Settings.load(out)


def test_load_rejects_wrong_int_type(isolated_settings, tmp_path):
    out = tmp_path / "bad.json"
    out.write_text(
        json.dumps({"embedder_token_limit": "8192"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected int"):
        Settings.load(out)


def test_load_rejects_bool_for_int(isolated_settings, tmp_path):
    out = tmp_path / "bad.json"
    out.write_text(
        json.dumps({"embedder_token_limit": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected int"):
        Settings.load(out)


def test_load_rejects_wrong_str_type(isolated_settings, tmp_path):
    out = tmp_path / "bad.json"
    out.write_text(
        json.dumps({"language": 42}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected str"):
        Settings.load(out)


def test_load_rejects_non_object_json(isolated_settings, tmp_path):
    out = tmp_path / "bad.json"
    out.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        Settings.load(out)


def test_load_rejects_invalid_json(isolated_settings, tmp_path):
    out = tmp_path / "bad.json"
    out.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to read"):
        Settings.load(out)


def test_load_rejects_missing_file(isolated_settings, tmp_path):
    out = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="Failed to read"):
        Settings.load(out)


def test_load_ignores_unknown_keys(isolated_settings, tmp_path, caplog):
    out = tmp_path / "extra.json"
    out.write_text(
        json.dumps({
            "embedder_token_limit": 256,
            "mystery_field": "ignored",
        }),
        encoding="utf-8",
    )
    Settings.load(out)
    assert Settings.embedder_token_limit == 256


# --- save filesystem behavior -----------------------------------------------


def test_save_creates_parent_directories(isolated_settings, tmp_path):
    out = tmp_path / "nested" / "deep" / "settings.json"
    Settings.save(out)
    assert out.exists()


def test_save_pathlib_and_str_equivalent(isolated_settings, tmp_path):
    out_str = tmp_path / "as_str.json"
    Settings.save(str(out_str))
    assert out_str.exists()

    out_path = tmp_path / "as_path.json"
    Settings.save(out_path)
    assert out_path.exists()
