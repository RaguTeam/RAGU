"""Adapter-call synchronization preserves adapter delegation and lifecycle."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from typing_extensions import override

from ragu.common.global_parameters import Settings
from ragu.graph._storage_guard import _guard_storage
from ragu.graph.index import Index, StorageArguments
from ragu.graph.types import Community, CommunitySummary, Entity, Relation
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage
from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage


class _DelegatingNetworkX(NetworkXStorage):
    @property
    def connection(self):
        raise ValueError("This connection property must not be evaluated by the guard")

    @override
    async def upsert_nodes(self, nodes):
        await asyncio.gather(self.get_nodes([node.id for node in nodes]))
        await super().upsert_nodes(nodes)


async def test_adapter_write_can_delegate_to_public_reads_in_child_tasks(tmp_path):
    storage = _DelegatingNetworkX(str(tmp_path / "graph.gml"), Entity, Relation)
    _guard_storage(storage)
    node = Entity("A", "PERSON", "Description", ["source"], id="a")

    await asyncio.wait_for(storage.upsert_nodes([node]), 2)
    node.source_chunk_id.append("changed-input")
    snapshot = (await storage.get_nodes([node.id]))[0]
    assert snapshot.source_chunk_id == ["source"]
    snapshot.source_chunk_id.append("changed-snapshot")
    assert (await storage.get_nodes([node.id]))[0].source_chunk_id == ["source"]


async def test_internal_adapter_delegation_keeps_live_values_available():
    class InPlaceAdapter:
        def __init__(self):
            self.data = {"a": {"value": 0}}

        async def get_by_ids(self, ids):
            return [self.data[item_id] for item_id in ids]

        async def upsert(self, values):
            rows = (await asyncio.gather(self.get_by_ids(list(values))))[0]
            rows[0]["value"] = values["a"]["value"]

    storage = InPlaceAdapter()
    _guard_storage(storage)
    await asyncio.wait_for(storage.upsert({"a": {"value": 42}}), 2)

    snapshot = (await storage.get_by_ids(["a"]))[0]
    assert snapshot == {"value": 42}
    snapshot["value"] = -1
    assert storage.data["a"] == {"value": 42}


async def test_completed_adapter_call_does_not_leave_child_tasks_with_a_live_lease(tmp_path):
    late_release = asyncio.Event()
    late_attempted = asyncio.Event()
    read_entered = asyncio.Event()
    writer_entered = asyncio.Event()
    writer_release = asyncio.Event()

    class DeferredNetworkX(NetworkXStorage):
        @override
        async def get_nodes(self, ids):
            read_entered.set()
            return await super().get_nodes(ids)

        @override
        async def upsert_nodes(self, nodes):
            if not hasattr(self, "late_read"):
                async def read_later():
                    await late_release.wait()
                    late_attempted.set()
                    return await self.get_nodes([node.id for node in nodes])

                self.late_read = asyncio.create_task(read_later())
            else:
                writer_entered.set()
                await writer_release.wait()
            await super().upsert_nodes(nodes)

    storage = DeferredNetworkX(str(tmp_path / "graph.gml"), Entity, Relation)
    _guard_storage(storage)
    node = Entity("A", "PERSON", "Original", [], id="a")
    await storage.upsert_nodes([node])
    writing = asyncio.create_task(storage.upsert_nodes([replace(node, description="Updated")]))
    await asyncio.wait_for(writer_entered.wait(), 2)

    try:
        late_release.set()
        await asyncio.wait_for(late_attempted.wait(), 2)
        assert not read_entered.is_set()
        assert not storage.late_read.done()
    finally:
        writer_release.set()
        _, snapshots = await asyncio.wait_for(asyncio.gather(writing, storage.late_read), 2)

    assert snapshots[0].description == "Updated"


class _LifecycleKV(JsonKVStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []
        self.active = False

    @override
    async def index_start_callback(self) -> None:
        if self.active:
            raise ValueError("A write lifecycle is already active")
        self.active = True
        self.events.append("start")

    @override
    async def index_done_callback(self) -> None:
        self.events.append("done")
        self.active = False
        await super().index_done_callback()


@pytest.mark.parametrize("kind", ["community", "summary"])
async def test_update_starts_and_finishes_storage_lifecycle_exactly_once(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    index = Index(StorageArguments(kv_storage_type=_LifecycleKV), embedder=SimpleNamespace(dim=4))
    if kind == "community":
        item = Community(level=0, cluster_id=0, entities=[], relations=[])
        await index.upsert_communities([item])
        storage = index.community_kv_storage
        update = index.update_communities
    else:
        item = CommunitySummary(id="community", summary="Original")
        await index.upsert_summaries([item])
        storage = index.community_summary_kv_storage
        update = index.update_summaries

    storage.events.clear()
    await update([item])

    assert storage.events == ["start", "done"]
    assert not storage.active
