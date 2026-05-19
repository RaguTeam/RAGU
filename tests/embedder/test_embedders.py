from unittest.mock import AsyncMock

import pytest

from ragu.models.embedder import EmbedderOpenAI


class _FakeClient:
    def __init__(self):
        self.embed_text = AsyncMock()
        self.embed_texts = AsyncMock()


def test_embedder_openai_init_sets_fields():
    client = _FakeClient()
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=5,
    )

    assert embedder.client is client
    assert embedder.model_name == "text-embedding-3-small"
    assert embedder.dim == 5


async def test_embedder_openai_forwards_embed_text_call():
    client = _FakeClient()
    client.embed_text.return_value = [0.1, 0.2, 0.3]
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=3,
    )

    result = await embedder.embed_text("hello")

    assert result == [0.1, 0.2, 0.3]
    client.embed_text.assert_awaited_once_with(
        model_name="text-embedding-3-small",
        text="hello",
    )


async def test_embedder_openai_merges_init_kwargs_and_call_kwargs():
    client = _FakeClient()
    client.embed_text.return_value = [0.5, 0.6]
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=2,
        timeout=3,
    )

    result = await embedder.embed_text("hi", user="u-1")

    assert result == [0.5, 0.6]
    client.embed_text.assert_awaited_once_with(
        model_name="text-embedding-3-small",
        text="hi",
        timeout=3,
        user="u-1",
    )


async def test_batch_embed_text_single_batch():
    client = _FakeClient()
    client.embed_texts.return_value = [[0.1], [0.2], [0.3]]
    embedder = EmbedderOpenAI(
        client=client,
        model_name="emb",
        dim=1,
        batch_size=10,
    )

    result = await embedder.batch_embed_text(["a", "b", "c"])

    assert result == [[0.1], [0.2], [0.3]]
    client.embed_texts.assert_awaited_once_with(
        model_name="emb",
        texts=["a", "b", "c"],
    )


async def test_batch_embed_text_splits_into_sub_batches():
    call_log: list[list[str]] = []

    async def _embed_texts_side_effect(**kwargs):
        call_log.append(list(kwargs["texts"]))
        return [[0.0] for _ in kwargs["texts"]]

    client = _FakeClient()
    client.embed_texts.side_effect = _embed_texts_side_effect
    embedder = EmbedderOpenAI(
        client=client,
        model_name="emb",
        dim=1,
        batch_size=2,
        max_concurrent_batches=1,
    )

    result = await embedder.batch_embed_text(["a", "b", "c"])

    assert len(result) == 3
    assert sorted(call_log) == [["a", "b"], ["c"]]


async def test_batch_embed_text_empty_input():
    client = _FakeClient()
    client.embed_texts.return_value = []
    embedder = EmbedderOpenAI(
        client=client,
        model_name="emb",
        dim=1,
    )

    result = await embedder.batch_embed_text([])

    assert result == []
    client.embed_texts.assert_not_awaited()


async def test_batch_embed_text_error_continues_then_raises():
    client = _FakeClient()
    client.embed_texts.side_effect = [
        RuntimeError("boom"),
        [[0.3]],
    ]
    embedder = EmbedderOpenAI(
        client=client,
        model_name="emb",
        dim=1,
        batch_size=1,
        max_concurrent_batches=1,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await embedder.batch_embed_text(["a", "b"])

    assert client.embed_texts.await_count == 2


async def test_batch_embed_text_kwargs_forwarding():
    client = _FakeClient()
    client.embed_texts.return_value = [[0.1]]
    embedder = EmbedderOpenAI(
        client=client,
        model_name="emb",
        dim=1,
        batch_size=10,
        timeout=3,
    )

    await embedder.batch_embed_text(["a"], user="u-1")

    client.embed_texts.assert_awaited_once_with(
        model_name="emb",
        texts=["a"],
        timeout=3,
        user="u-1",
    )

