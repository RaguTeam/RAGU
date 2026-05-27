from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragu.chunker.types import Chunk
from ragu.common.prompts.default_models import ArtifactsModel, EntityModel, RelationModel
from ragu.triplet.llm_artifact_extractor import ArtifactsExtractorLLM


def _make_chunk(text="Hello world"):
    return Chunk(content=text, chunk_order_idx=0, doc_id="doc-1")


def _make_extractor():
    llm = AsyncMock()
    llm.batch_chat_completion = AsyncMock(return_value=[])

    extractor = ArtifactsExtractorLLM(llm=llm, do_validation=False)

    return extractor, llm


def _patch_render(extractor_module):
    patcher = patch.object(extractor_module, 'render_with_few_shots')
    mock_render = patcher.start()

    mock_messages = MagicMock()
    mock_messages.to_openai.return_value = [{'role': 'user', 'content': 'test'}]

    mock_render.return_value = [mock_messages]
    return patcher, mock_render


async def test_extraction_total_failure():
    extractor, llm = _make_extractor()

    import ragu.triplet.llm_artifact_extractor as module
    patcher, mock_render = _patch_render(module)
    try:
        extractor.get_prompt = MagicMock(return_value=SimpleNamespace(
            messages=[MagicMock()],
            pydantic_model=ArtifactsModel,
            few_shot_formatter=None,
        ))

        llm.batch_chat_completion.return_value = [None, None]

        chunks = [_make_chunk("chunk 1"), _make_chunk("chunk 2")]
        entities, relations = await extractor.extract(chunks)

        assert entities == []
        assert relations == []
    finally:
        patcher.stop()


async def test_extraction_success():
    extractor, llm = _make_extractor()

    import ragu.triplet.llm_artifact_extractor as module
    patcher, mock_render = _patch_render(module)
    try:
        extractor.get_prompt = MagicMock(return_value=SimpleNamespace(
            messages=[MagicMock()],
            pydantic_model=ArtifactsModel,
            few_shot_formatter=None,
        ))

        artifacts = ArtifactsModel(
            entities=[EntityModel(entity_name="Alice", entity_type="Person", description="A person")],
            relations=[],
        )
        llm.batch_chat_completion.return_value = [artifacts]

        chunks = [_make_chunk("chunk 1")]
        entities, relations = await extractor.extract(chunks)

        assert len(entities) == 1
        assert entities[0].entity_name == "Alice"
    finally:
        patcher.stop()


async def test_extraction_partial_failure():
    extractor, llm = _make_extractor()

    import ragu.triplet.llm_artifact_extractor as module
    patcher, mock_render = _patch_render(module)
    try:
        extractor.get_prompt = MagicMock(return_value=SimpleNamespace(
            messages=[MagicMock()],
            pydantic_model=ArtifactsModel,
            few_shot_formatter=None,
        ))

        artifacts_ok = ArtifactsModel(
            entities=[EntityModel(entity_name="Alice", entity_type="Person", description="A person")],
            relations=[],
        )
        llm.batch_chat_completion.return_value = [None, artifacts_ok]

        chunks = [_make_chunk("chunk 1"), _make_chunk("chunk 2")]
        entities, relations = await extractor.extract(chunks)

        assert len(entities) == 1
        assert entities[0].entity_name == "Alice"
    finally:
        patcher.stop()


async def test_validation_failure_falls_back_to_unvalidated():
    extractor, llm = _make_extractor()
    extractor.do_validation = True

    import ragu.triplet.llm_artifact_extractor as module
    patcher, mock_render = _patch_render(module)
    try:
        extractor.get_prompt = MagicMock(return_value=SimpleNamespace(
            messages=[MagicMock()],
            pydantic_model=ArtifactsModel,
            few_shot_formatter=None,
        ))

        artifacts = ArtifactsModel(
            entities=[EntityModel(entity_name="Alice", entity_type="Person", description="A person")],
            relations=[],
        )

        call_count = 0
        async def _batch_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [artifacts]
            return [None]

        llm.batch_chat_completion.side_effect = _batch_side_effect

        chunks = [_make_chunk("chunk 1")]
        entities, relations = await extractor.extract(chunks)

        assert len(entities) == 1
        assert entities[0].entity_name == "Alice"
    finally:
        patcher.stop()
