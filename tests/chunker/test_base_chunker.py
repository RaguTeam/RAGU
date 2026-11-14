from inspect import isabstract
from unittest.mock import AsyncMock, Mock

import pytest
from typing_extensions import override

from ragu.chunker.base_chunker import BaseChunker
from ragu.chunker.chunkers import SimpleChunker, SemanticTextChunker, SmartSemanticChunker
from ragu.chunker.types import Chunk, Document, TextChunk
from ragu.common.global_parameters import Settings
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.embedder import Embedder
from ragu.utils.ragu_utils import compute_mdhash_id


@pytest.mark.parametrize("chunker_cls", [SimpleChunker, SemanticTextChunker, SmartSemanticChunker])
def test_synchronous_chunkers_remain_concrete(chunker_cls):
    assert not isabstract(chunker_cls)


@pytest.mark.parametrize("documents", ["", "First. Second.", [], ["First.", "Second."], ("First.", "Second.")])
async def test_string_inputs_preserve_synchronous_batch_behavior(documents):
    chunker = SimpleChunker(max_chunk_size=40)
    expected_input = list(documents) if isinstance(documents, tuple) else documents
    expected = chunker.split(expected_input)
    chunker.split = Mock(wraps=chunker.split)

    assert await chunker.a_split(documents) == expected
    chunker.split.assert_called_once_with(expected_input)


async def test_mixed_sequence_preserves_explicit_document_ids_and_source_order():
    chunker = SimpleChunker(max_chunk_size=40)
    documents = (
        Document("Same content.", id="first-source"),
        "Plain text.",
        Document("Same content.", id="second-source"),
    )

    chunks = await chunker.a_split(documents)

    assert [chunk.content for chunk in chunks] == ["Same content.", "Plain text.", "Same content."]
    assert [chunk.doc_id for chunk in chunks] == [
        "first-source", compute_mdhash_id("Plain text."), "second-source",
    ]
    assert chunks[0].id == chunks[2].id


class _TypedSyncChunker(BaseChunker):
    @override
    def split(self, documents: str | list[str]) -> list[Chunk]:
        texts = [documents] if isinstance(documents, str) else documents
        return [TextChunk(text, 0, "legacy-source", metadata={"section": "Results"}) for text in texts]


async def test_sync_only_subclass_preserves_typed_chunk_identity_for_document():
    document = Document("Text.", id="custom-source")

    chunks = await _TypedSyncChunker().a_split(document)

    assert chunks == [TextChunk("Text.", 0, "custom-source", metadata={"section": "Results"})]
    assert chunks[0].id != TextChunk("Text.", 0, "legacy-source").id


async def test_graph_build_with_synchronous_chunker_and_document(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path))
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 8
    embedder.batch_embed_text.side_effect = lambda texts, **kwargs: [[0.1] * 8 for _ in texts]
    document = Document("First sentence. Second sentence.", id="custom-source")
    graph = KnowledgeGraph(
        llm=None,
        embedder=embedder,
        chunker=SimpleChunker(max_chunk_size=20),
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    try:
        await graph.build_from_docs([document])
        chunk_ids = await graph.index.chunks_kv_storage.all_keys()
        chunks = await graph.get_chunks(chunk_ids)

        assert len(chunks) == 2
        assert all(chunk is not None and chunk.doc_id == document.id for chunk in chunks)
        records = await graph.index.documents_kv_storage.get_by_ids(
            await graph.index.documents_kv_storage.all_keys()
        )
        assert records == [{"chunk_ids": chunk_ids}]
    finally:
        await graph.index.close()
