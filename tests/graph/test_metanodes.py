import re
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.markdown_chunkers import RecursiveMarkdownChunker
from ragu.chunker.types import Chunk, Document, FormulaChunk, TableChunk, TextChunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.default_models import RelationModel, RelationsExtractionModel
from ragu.graph.builder_modules import RemoveIsolatedNodes
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.index import Index, StorageArguments
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.graph.metanodes import NearbyRelationExtractor
from ragu.graph.types import Entity
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.search_engine.naive_search import NaiveSearchEngine, NaiveSearchResult, NaiveSearchRetrieve
from ragu.search_engine.local_search import LocalSearchEngine
from ragu.search_engine.search_functional import _find_most_related_text_unit_from_entities


@pytest.fixture(autouse=True)
def local_token_budget(monkeypatch):
    monkeypatch.setattr(
        'ragu.graph.metanodes.TokenTruncation',
        lambda **kwargs: lambda text: text[:kwargs['max_tokens']],
    )
    monkeypatch.setattr(
        'ragu.search_engine.base_engine.TokenTruncation',
        lambda *args, **kwargs: lambda text: text,
    )


@pytest.fixture
def embedder():
    value = AsyncMock(spec=Embedder)
    value.dim = 8
    value.batch_embed_text.side_effect = lambda texts, **kwargs: [[0.1] * 8 for _ in texts]
    return value


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, 'storage_folder', str(tmp_path / 'storage'))
    monkeypatch.setattr(Settings, 'tokenizer_llm_name', 'gpt-4o')
    monkeypatch.setattr(Settings, 'tokenizer_llm_backend', 'tiktoken')
    return StorageArguments()


def nearby_answer(conversations, **kwargs):
    results = []
    for conversation in conversations:
        prompt = conversation[-1]['content']
        object_name = re.search(r'^Object name: (.+)$', prompt, re.MULTILINE).group(1)
        entity_name = re.search(r'^- Name: (.+)$', prompt, re.MULTILINE).group(1)
        results.append(RelationsExtractionModel(relations=[RelationModel(
            source_entity=object_name, target_entity=entity_name,
            relation_type='DESCRIBES', description='The object describes the nearby subject.',
            relationship_strength=4,
        )]))
    return results


async def test_build_persist_reload_search_and_delete(storage, embedder):
    llm = AsyncMock(spec=LLM)
    llm.batch_chat_completion.side_effect = nearby_answer
    extractor = AsyncMock()

    async def extract(chunks):
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        return [Entity('Battery', 'PRODUCT', 'Battery tests', [chunks[0].id], [chunks[0].doc_id])], []

    extractor.side_effect = extract
    kg = KnowledgeGraph(
        llm=llm, embedder=embedder, chunker=RecursiveMarkdownChunker(), artifact_extractor=extractor,
        builder_settings=BuilderArguments(use_llm_summarization=False, make_community_summary=False),
        storage_settings=storage,
    )
    doc = Document('# Tests\n\nBattery results are shown below.\n\n<table><tr><td>12</td></tr></table>\n\n$$E=mc^2$$')
    await kg.build_from_docs([doc])
    nodes = await kg.index.graph_backend.get_all_nodes()
    assert sorted(node.entity_type for node in nodes) == ['FORMULA', 'PRODUCT', 'TABLE']
    metanodes = [node for node in nodes if node.id.startswith('meta-')]
    assert len({node.id for node in metanodes}) == 2
    chunk_ids = [node.source_chunk_id[0] for node in metanodes]
    chunks = await kg.index.get_chunks(chunk_ids)
    assert {type(chunk) for chunk in chunks} == {TableChunk, FormulaChunk}
    for chunk, chunk_id in zip(chunks, chunk_ids):
        assert chunk.id == chunk_id
        assert any(chunk.to_embed() in call.args[0] for call in embedder.batch_embed_text.call_args_list)
    context = NaiveSearchRetrieve(query='results', result=NaiveSearchResult(chunks=chunks, scores=[1.0, 1.0]), metrics={})
    assert '<table>' in context.to_text() and '$$' in context.to_text()

    node = next(node for node in nodes if node.entity_type == 'PRODUCT')
    edges = (await kg.index.graph_backend.get_all_edges_for_nodes([node.id]))[0]
    related = await _find_most_related_text_unit_from_entities(
        [node], kg, {node.id: edges}, {meta.id: meta.source_chunk_id for meta in metanodes}
    )
    assert set(chunk_ids) <= {chunk.id for chunk in related}

    answer_llm = AsyncMock(spec=LLM)
    answer_llm.batch_chat_completion.return_value = ['An answer']
    for engine_cls in (NaiveSearchEngine, LocalSearchEngine):
        engine = engine_cls(llm=answer_llm, knowledge_graph=kg, embedder=embedder)
        answer = await engine.a_query('Battery results')
        assert answer.response == 'An answer'
        messages = answer_llm.batch_chat_completion.call_args.args[0][0]
        rendered = '\n'.join(str(message['content']) for message in messages)
        assert '<table>' in rendered and '$$' in rendered

    calls = llm.batch_chat_completion.await_count
    await kg.build_from_docs([doc])
    assert llm.batch_chat_completion.await_count == calls
    extractor.assert_awaited_once()
    reloaded = Index(arguments=storage, embedder=embedder)
    restored = await reloaded.get_chunks(chunk_ids)
    assert [chunk.id for chunk in restored] == chunk_ids
    assert {type(chunk) for chunk in restored} == {TableChunk, FormulaChunk}
    assert (await reloaded.check_consistency()).is_consistent
    await reloaded.delete_chunks([chunk_ids[0]])
    assert (await reloaded.get_nodes([metanodes[0].id])) == [None]
    assert (await reloaded.check_consistency()).is_consistent


@pytest.mark.parametrize('kind', [Chunk, TextChunk, TableChunk, FormulaChunk])
def test_chunk_serialization_preserves_id_and_type(kind):
    chunk = kind('content', 3, 'doc', metadata={'section': 'Results'})
    restored = Chunk.from_dict(chunk.to_dict())
    assert type(restored) is kind
    assert restored.id == chunk.id and restored.metadata == chunk.metadata


def test_legacy_chunk_uses_storage_key():
    chunk = Chunk.from_dict({'content': 'text', 'chunk_order_idx': 0, 'doc_id': 'doc'}, 'old-key')
    assert type(chunk) is Chunk and chunk.id == 'old-key'
    with pytest.raises(ValueError, match='Unknown chunk type'):
        Chunk.from_dict({'chunk_type': 'unexpected'})


def test_repeated_formulas_keep_content_based_ids():
    chunks = RecursiveMarkdownChunker().split('$$x$$\n\n$$x$$')
    assert len(chunks) == 2 and chunks[0].id == chunks[1].id
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in RecursiveMarkdownChunker().split('$$x$$\n\n$$x$$')]


@pytest.mark.parametrize('fault', ['unknown_entity', 'unknown_source', 'empty_description', 'self_edge', 'no_object'])
async def test_invalid_nearby_edges_are_rejected(fault):
    chunks = RecursiveMarkdownChunker().split('Text.\n\n$$x$$')
    entity = Entity('Alice', 'PERSON', 'Alice', [chunks[0].id])
    llm = AsyncMock(spec=LLM)

    def answer(conversations, **kwargs):
        result = nearby_answer(conversations, **kwargs)[0]
        edge = result.relations[0]
        if fault == 'unknown_entity':
            edge.target_entity = 'unknown'
        elif fault == 'unknown_source':
            edge.source_entity = 'unknown'
        elif fault == 'empty_description':
            edge.description = ''
        elif fault == 'self_edge':
            edge.target_entity = edge.source_entity
        else:
            edge.source_entity = edge.target_entity
        return [result]

    llm.batch_chat_completion.side_effect = answer
    nodes, edges = await NearbyRelationExtractor(llm).a_extract_nearby(chunks, [entity], [])
    assert len(nodes) == 1 and not edges


async def test_window_does_not_cross_document_or_section():
    chunks = RecursiveMarkdownChunker().split([
        Document('# One\n\nText.\n\n## Two\n\n$$x$$'), Document('# Two\n\nOther text.')
    ])
    entities = [Entity('Alice', 'PERSON', '', [chunk.id]) for chunk in chunks if isinstance(chunk, TextChunk)]
    llm = AsyncMock(spec=LLM)
    nodes, edges = await NearbyRelationExtractor(llm).a_extract_nearby(chunks, entities, [])
    assert len(nodes) == 1 and not edges
    llm.batch_chat_completion.assert_not_awaited()


async def test_unlinked_objects_survive_isolated_node_removal():
    chunks = RecursiveMarkdownChunker().split('$$x$$')
    nodes, edges = await NearbyRelationExtractor(None).a_extract_nearby(chunks, [], [])
    kept, _ = await RemoveIsolatedNodes().run(nodes, edges)
    assert kept == nodes


async def test_document_hash_skips_chunker_before_description(storage, embedder):
    llm = AsyncMock(spec=LLM)
    llm.batch_chat_completion.return_value = ['An equation']
    chunker = RecursiveMarkdownChunker(llm=llm)
    graph = KnowledgeGraph(
        llm=None, embedder=embedder, chunker=chunker, storage_settings=storage,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    await graph.build_from_docs(['$$x$$', '$$x$$'])
    llm.batch_chat_completion.assert_awaited_once()
    chunker.a_split = AsyncMock(side_effect=AssertionError('processed document reached chunking'))
    await graph.build_from_docs(['$$x$$'])
    chunker.a_split.assert_not_awaited()
    reloaded = KnowledgeGraph(
        llm=None, embedder=embedder, chunker=chunker, storage_settings=storage,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    await reloaded.build_from_docs(['$$x$$'])
    chunker.a_split.assert_not_awaited()


async def test_failed_nearby_call_retains_object():
    chunks = RecursiveMarkdownChunker().split('Text.\n\n$$x$$')
    entity = Entity('Alice', 'PERSON', '', [chunks[0].id])
    llm = AsyncMock(spec=LLM)
    llm.batch_chat_completion.return_value = [None]
    nodes, edges = await NearbyRelationExtractor(llm).a_extract_nearby(chunks, [entity], [])
    assert len(nodes) == 1 and edges == []


async def test_oversized_nearby_context_is_skipped(monkeypatch):
    monkeypatch.setattr(Settings, 'llm_context_token_limit', 1)
    chunks = RecursiveMarkdownChunker().split('Text.\n\n$$x$$')
    entity = Entity('Alice', 'PERSON', '', [chunks[0].id])
    llm = AsyncMock(spec=LLM)
    nodes, edges = await NearbyRelationExtractor(llm).a_extract_nearby(chunks, [entity], [])
    assert len(nodes) == 1 and edges == []
    llm.batch_chat_completion.assert_not_awaited()


async def test_object_only_graph_without_llm_or_extractor(storage, embedder):
    kg = KnowledgeGraph(
        llm=None, embedder=embedder, chunker=RecursiveMarkdownChunker(), storage_settings=storage,
        builder_settings=BuilderArguments(use_llm_summarization=False, make_community_summary=False),
    )
    await kg.build_from_docs(['$$x$$'])
    nodes = await kg.index.graph_backend.get_all_nodes()
    assert len(nodes) == 1 and nodes[0].entity_type == 'FORMULA'


@pytest.mark.parametrize('window', [0, -1, True, 1.5])
def test_invalid_context_window(window):
    with pytest.raises(ValueError, match='positive integer'):
        NearbyRelationExtractor(None, window)


async def test_failed_build_does_not_mark_document_processed(storage, embedder):
    graph = KnowledgeGraph(
        llm=None, embedder=embedder, chunker=RecursiveMarkdownChunker(), storage_settings=storage,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    original = graph.pipeline.extract_graph
    graph.pipeline.extract_graph = AsyncMock(side_effect=RuntimeError('extraction failed'))
    with pytest.raises(RuntimeError, match='extraction failed'):
        await graph.build_from_docs(['$$x$$'])
    assert await graph.index.documents_kv_storage.all_keys() == []
    graph.pipeline.extract_graph = original
    await graph.build_from_docs(['$$x$$'])
    assert len(await graph.index.documents_kv_storage.all_keys()) == 1


async def test_deleted_chunks_invalidate_document_hash(storage, embedder):
    graph = KnowledgeGraph(
        llm=None, embedder=embedder, chunker=RecursiveMarkdownChunker(), storage_settings=storage,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    await graph.build_from_docs(['$$x$$'])
    chunk_ids = await graph.index.chunks_kv_storage.all_keys()
    await graph.index.delete_chunks(chunk_ids)
    assert await graph.index.documents_kv_storage.all_keys() == []
    await graph.build_from_docs(['$$x$$'])
    assert await graph.index.chunks_kv_storage.all_keys() == chunk_ids


async def test_document_hash_is_stable_and_tracks_content_and_identity(storage, embedder):
    graph = KnowledgeGraph(llm=None, embedder=embedder, storage_settings=storage,
                          builder_settings=BuilderArguments(build_only_vector_context=True))
    docs = await graph._deduplicate_documents_by_hash(['text', Document('text'), 'changed'])
    assert len(docs) == 2
    assert list(docs) == list(await graph._deduplicate_documents_by_hash(['text', Document('text'), 'changed']))
    assert len(await graph._deduplicate_documents_by_hash([
        Document('text', id='one'), Document('text', id='two'),
    ])) == 2


async def test_read_only_index_does_not_open_document_hash_registry(storage, embedder):
    from pathlib import Path

    index = Index(arguments=storage, embedder=embedder)
    await index.get_chunks([])
    await index.close()
    assert index._documents_kv_storage is None
    assert not (Path(Settings.storage_folder) / 'kv_documents.json').exists()


async def test_only_new_documents_reach_chunker_and_deletion_keeps_other_hashes(storage, embedder):
    chunker = RecursiveMarkdownChunker()
    graph = KnowledgeGraph(
        llm=None, embedder=embedder, chunker=chunker, storage_settings=storage,
        builder_settings=BuilderArguments(build_only_vector_context=True),
    )
    await graph.build_from_docs(['$$x$$', '$$y$$'])
    chunker.a_split = AsyncMock(wraps=chunker.a_split)
    await graph.build_from_docs(['$$x$$', '$$z$$', '$$z$$'])
    chunker.a_split.assert_awaited_once_with(['$$z$$'])
    assert len(await graph.index.documents_kv_storage.all_keys()) == 3
    x_chunks = chunker.split('$$x$$')
    await graph.index.delete_chunks([x_chunks[0].id])
    assert len(await graph.index.documents_kv_storage.all_keys()) == 2
    chunker.a_split.reset_mock()
    await graph.build_from_docs(['$$y$$', '$$z$$'])
    chunker.a_split.assert_not_awaited()


async def test_existing_relation_schema_preserves_strength_and_context():
    chunks = RecursiveMarkdownChunker().split('Before.\n\n$$x$$\n\nAfter.')
    entity = Entity('Alice', 'PERSON', 'Alice', [chunks[0].id])
    llm = AsyncMock(spec=LLM)
    llm.batch_chat_completion.side_effect = nearby_answer
    nodes, edges = await NearbyRelationExtractor(llm).a_extract_nearby(chunks, [entity], [])
    assert llm.batch_chat_completion.call_args.kwargs['output_schema'] is RelationsExtractionModel
    assert len(edges) == 1 and edges[0].relation_strength == 4
    assert edges[0].subject_id == nodes[0].id and edges[0].object_id == entity.id
    assert set(edges[0].source_chunk_id) == {chunk.id for chunk in chunks}


async def test_relation_names_allow_reverse_direction():
    chunks = RecursiveMarkdownChunker().split('Text.\n\n$$x$$')
    entity = Entity('Alice', 'PERSON', 'Alice', [chunks[0].id])
    llm = AsyncMock(spec=LLM)

    def answer(conversations, **kwargs):
        results = nearby_answer(conversations, **kwargs)
        edge = results[0].relations[0]
        edge.source_entity, edge.target_entity = edge.target_entity, edge.source_entity
        return results

    llm.batch_chat_completion.side_effect = answer
    nodes, edges = await NearbyRelationExtractor(llm).a_extract_nearby(chunks, [entity], [])
    assert len(edges) == 1
    assert edges[0].subject_id == entity.id and edges[0].object_id == nodes[0].id


async def test_ambiguous_relation_names_are_rejected():
    chunks = RecursiveMarkdownChunker().split('Text.\n\n$$x$$')
    entities = [Entity('Same name', kind, '', [chunks[0].id]) for kind in ('PERSON', 'ORGANIZATION')]
    llm = AsyncMock(spec=LLM)
    llm.batch_chat_completion.side_effect = nearby_answer
    nodes, edges = await NearbyRelationExtractor(llm).a_extract_nearby(chunks, entities, [])
    assert len(nodes) == 1 and edges == []


@pytest.mark.parametrize(('object_text', 'kind', 'prompt_name'), [
    ('$$x$$', 'FORMULA', 'formula_relations'),
    ('<table><tr><td>1</td></tr></table>', 'TABLE', 'table_relations'),
])
async def test_type_specific_prompts_render_before_and_after_context(object_text, kind, prompt_name):
    before = 'До объекта "quoted" {{ untouched }}.'
    after = 'After the object.'
    chunks = RecursiveMarkdownChunker().split(f'{before}\n\n{object_text}\n\n{after}')
    entity = Entity('Alice', 'PERSON', 'Alice', [chunks[0].id])
    llm = AsyncMock(spec=LLM)
    llm.batch_chat_completion.side_effect = nearby_answer
    extractor = NearbyRelationExtractor(llm)
    _, edges = await extractor.a_extract_nearby(chunks, [entity], [])
    conversation = llm.batch_chat_completion.call_args.args[0][0]
    assert [message['role'] for message in conversation] == ['system', 'user']
    assert conversation[0]['content'].strip() == extractor.get_prompt(prompt_name).messages.messages[0].content.strip()
    prompt = conversation[1]['content']
    assert before in prompt and after in prompt
    assert prompt.index(before) < prompt.index(f'TARGET {kind}') < prompt.index(after)
    assert chunks[0].id in prompt and chunks[2].id in prompt
    assert '"context_before"' not in prompt and '"context_after"' not in prompt
    assert len(edges) == 1


@pytest.mark.parametrize('at_start', [True, False])
async def test_user_prompt_handles_missing_context_side(at_start):
    chunks = RecursiveMarkdownChunker().split('$$x$$\n\nText.' if at_start else 'Text.\n\n$$x$$')
    text = next(chunk for chunk in chunks if isinstance(chunk, TextChunk))
    entity = Entity('Alice', 'PERSON', '', [text.id])
    llm = AsyncMock(spec=LLM)
    llm.batch_chat_completion.side_effect = nearby_answer
    await NearbyRelationExtractor(llm).a_extract_nearby(chunks, [entity], [])
    prompt = llm.batch_chat_completion.call_args.args[0][0][-1]['content']
    assert ('No preceding text chunks.' if at_start else 'No following text chunks.') in prompt
    assert ('No following text chunks.' if at_start else 'No preceding text chunks.') not in prompt
    assert 'Text.' in prompt



async def test_table_and_formula_extraction_use_separate_batches():
    chunks = RecursiveMarkdownChunker().split(
        'Text.\n\n$$x$$\n\nMiddle.\n\n<table><tr><td>1</td></tr></table>\n\n'
        'More text.\n\n$$y$$\n\nAnother text.\n\n<table><tr><td>2</td></tr></table>\n\nEnd.'
    )
    entity = Entity('Alice', 'PERSON', '', [chunk.id for chunk in chunks if isinstance(chunk, TextChunk)])
    llm = AsyncMock(spec=LLM)
    llm.batch_chat_completion.side_effect = nearby_answer
    _, edges = await NearbyRelationExtractor(llm).a_extract_nearby(chunks, [entity], [])
    assert llm.batch_chat_completion.await_count == 2
    seen = set()
    for call in llm.batch_chat_completion.call_args_list:
        conversations = call.args[0]
        assert len(conversations) == 2
        kinds = {'TABLE' if 'TARGET TABLE' in conversation[-1]['content'] else 'FORMULA' for conversation in conversations}
        assert len(kinds) == 1
        kind = kinds.pop()
        seen.add(kind)
        assert all(f'target {kind.lower()}' in conversation[0]['content'] for conversation in conversations)
    assert seen == {'TABLE', 'FORMULA'} and len(edges) == 4
