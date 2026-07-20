import json
from unittest.mock import patch

import pytest

from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage


@pytest.mark.asyncio
async def test_initializes_empty_file_when_missing(tmp_path):
    storage_file = tmp_path / "kv.json"
    kv = JsonKVStorage(filename=str(storage_file))

    assert storage_file.exists()
    assert await kv.all_keys() == []


@pytest.mark.asyncio
async def test_loads_existing_file_data(tmp_path):
    storage_file = tmp_path / "kv.json"
    initial = {"k1": {"a": 1}, "k2": {"a": 2}}
    storage_file.write_text(json.dumps(initial), encoding="utf-8")

    kv = JsonKVStorage(filename=str(storage_file))

    assert await kv.get_by_id("k1") == {"a": 1}
    assert await kv.get_by_id("k2") == {"a": 2}


@pytest.mark.asyncio
async def test_get_by_ids_with_field_projection(tmp_path):
    storage_file = tmp_path / "kv.json"
    kv = JsonKVStorage(filename=str(storage_file))
    await kv.upsert(
        {
            "k1": {"a": 1, "b": 2},
            "k2": {"a": 10, "c": 30},
        }
    )

    result = await kv.get_by_ids(["k1", "missing", "k2"], fields={"a", "c"})

    assert result[0] == {"a": 1}
    assert result[1] is None
    assert result[2] == {"a": 10, "c": 30}


@pytest.mark.asyncio
async def test_filter_keys_returns_only_missing(tmp_path):
    storage_file = tmp_path / "kv.json"
    kv = JsonKVStorage(filename=str(storage_file))
    await kv.upsert({"k1": {"v": 1}, "k2": {"v": 2}})

    missing = await kv.filter_keys(["k1", "k2", "k3", "k4"])

    assert missing == {"k3", "k4"}


@pytest.mark.asyncio
async def test_upsert_delete_and_drop_work_in_memory(tmp_path):
    storage_file = tmp_path / "kv.json"
    kv = JsonKVStorage(filename=str(storage_file))

    await kv.upsert({"k1": {"v": 1}, "k2": {"v": 2}})
    assert set(await kv.all_keys()) == {"k1", "k2"}

    await kv.delete(["k1", "missing"])
    assert await kv.get_by_id("k1") is None
    assert await kv.get_by_id("k2") == {"v": 2}

    await kv.drop()
    assert await kv.all_keys() == []


@pytest.mark.asyncio
async def test_index_done_callback_persists_data(tmp_path):
    storage_file = tmp_path / "kv.json"
    kv = JsonKVStorage(filename=str(storage_file))
    await kv.upsert({"k1": {"text": "Привет"}})
    await kv.index_done_callback()

    reloaded = JsonKVStorage(filename=str(storage_file))

    assert await reloaded.get_by_id("k1") == {"text": "Привет"}


@pytest.mark.asyncio
async def test_drop_clears_memory_and_is_persisted_by_the_caller(tmp_path):
    """
    drop() stays memory-only, like upsert() and delete(): persistence is the
    caller's job. What must hold is that persisting after a drop actually
    empties the file - that is the path reindex_community relies on.
    """
    storage = JsonKVStorage(storage_folder=str(tmp_path), filename="kv.json")
    await storage.upsert({"com-1": {"summary": "old"}, "com-2": {"summary": "old"}})
    await storage.index_done_callback()

    await storage.drop()
    assert storage.data == {}
    assert json.loads((tmp_path / "kv.json").read_text(encoding="utf-8")) != {}

    await storage.index_done_callback()

    reloaded = JsonKVStorage(storage_folder=str(tmp_path), filename="kv.json")
    assert reloaded.data == {}


@pytest.mark.asyncio
async def test_interrupted_write_leaves_the_previous_file_intact(tmp_path):
    """A crash mid-write must not truncate the store."""
    storage = JsonKVStorage(storage_folder=str(tmp_path), filename="kv.json")
    await storage.upsert({"a": {"v": 1}})
    await storage.index_done_callback()
    path = tmp_path / "kv.json"
    before = path.read_text(encoding="utf-8")

    await storage.upsert({"b": {"v": 2}})
    with patch("json.dump", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            await storage.index_done_callback()

    assert path.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob("*.tmp"))
