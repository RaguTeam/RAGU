from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from typing_extensions import override
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from ragu.common.logger import logger
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.ragu_utils import FLOATS


T = TypeVar('T', BaseModel, str)

DEFAULT_EMBED_BATCH_SIZE = 500
DEFAULT_MAX_CONCURRENT_EMBED_BATCHES = 5


class Embedder(ABC):
    """Embedder interface to support various backends (openai, transformers etc.)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""

    @abstractmethod
    async def embed_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        """
        Calculates embedding for a single text.

        :param text: Input text to embed.
        :param kwargs: Backend-specific options.
        :returns: Embedding vector.
        """

    async def batch_embed_text(
        self,
        texts: list[str],
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]] | FLOATS:
        """
        Runs multiple :meth:`embed_text` calls concurrently.

        :param texts: List of input texts.
        :param desc: Optional tqdm progress description.
        :param kwargs: Extra kwargs forwarded to each call.
        :returns: Embeddings in the same order as input texts.
        """
        logger.debug(f'Calling batch_embed_text with size {len(texts)}')
        return await tqdm_asyncio.gather(*[ # type: ignore
            self.embed_text(
                text=text,
                **kwargs,
            )
            for text in texts
        ], desc=desc)


class EmbedderOpenAI(Embedder):
    """
    Adapts :class:`CachedAsyncOpenAI` to the :class:`Embedder` interface.
    
    Supports API-level batching: texts are grouped into sub-batches and
    sent to the ``/embeddings`` endpoint as ``input=[text1, text2, ...]``,
    dramatically reducing the number of HTTP requests compared to one
    request per text.
    
    :param client: OpenAI-compatible backend client.
    :param model_name: Embedding model identifier.
    :param dim: Embedding dimension. If ``None``, it is auto-detected on the
        first call to :meth:`initialize` by sending a probe request.
    :param batch_size: Maximum number of texts per single API call.
        The OpenAI ``/embeddings`` endpoint accepts up to 2048 inputs.
        Defaults to 500.
    :param max_concurrent_batches: Maximum number of batch API calls in
        flight simultaneously.  Controls peak concurrency and prevents
        connection-pool exhaustion.  Defaults to 5.
    :param kwargs: Default kwargs merged into each ``embed_text`` call.
    
    Example:
    ```
    embedder = EmbedderOpenAI(
        client=CachedAsyncOpenAI(),
        model_name='my-embedding-model',
        batch_size=500,
        max_concurrent_batches=5,
    )
    await embedder.initialize()  # auto-detects dim if not provided
    ```
    """
    
    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        dim: int | None = None,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        max_concurrent_batches: int = DEFAULT_MAX_CONCURRENT_EMBED_BATCHES,
        **kwargs: Any,
    ):
        self.client = client
        self.model_name = model_name
        self.kwargs = kwargs
        self._dim = dim
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def initialize(self) -> None:
        """
        Auto-detect embedding dimension if not explicitly provided.

        Sends a single probe request and stores the resulting vector length
        as ``self._dim``.  Safe to call multiple times — subsequent calls are
        no-ops when ``dim`` is already known.
        """
        if self._dim is not None:
            return
        probe = await self.embed_text("probe")
        self._dim = len(probe)
        logger.debug(
            f"Auto-detected embedding dim={self._dim} "
            f"for model '{self.model_name}'"
        )
    
    @override
    async def embed_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        """
        Forwards embedding request.

        :param text: Input text to embed.
        :param kwargs: Per-call kwargs merged with constructor kwargs.
            Per-call values override constructor defaults on conflicts.
        :returns: Embedding vector.
        """
        return await self.client.embed_text(
            model_name=self.model_name,
            text=text,
            **(self.kwargs | kwargs),
        )

    @override
    async def batch_embed_text(
        self,
        texts: list[str],
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]] | FLOATS:
        """
        Computes embeddings for multiple texts using API-level batching.

        Texts are split into sub-batches of ``batch_size`` and sent to
        the ``/embeddings`` endpoint as ``input=[t1, t2, ...]``.  A
        semaphore limits the number of concurrent batch API calls to
        ``max_concurrent_batches``.

        If a sub-batch fails after all retries, its texts are logged and
        the remaining sub-batches continue to be processed.  At the end,
        the first encountered exception is re-raised so that callers are
        aware of partial failure.

        :param texts: List of input texts.
        :param desc: Optional tqdm progress description.
        :param kwargs: Extra kwargs forwarded to each API call.
        :returns: Embeddings in the same order as input texts.
        """
        logger.debug(f'Calling batch_embed_text with size {len(texts)}')
        merged_kwargs = self.kwargs | kwargs

        sub_batches: list[list[str]] = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        first_error: Exception | None = None

        async def _process_batch(batch: list[str]) -> list[list[float] | FLOATS]:
            nonlocal first_error
            async with self._semaphore:
                try:
                    return await self.client.embed_texts(
                        model_name=self.model_name,
                        texts=batch,
                        **merged_kwargs,
                    )
                except Exception as e:
                    logger.warning(
                        f'Embedding sub-batch of {len(batch)} texts failed: '
                        f'{e.__class__.__name__}: {e}'
                    )
                    if first_error is None:
                        first_error = e
                    return [[] for _ in batch]

        batch_results = await tqdm_asyncio.gather(*[
            _process_batch(batch)
            for batch in sub_batches
        ], desc=desc)

        results: list[list[float] | FLOATS] = []
        for batch_result in batch_results:
            results.extend(batch_result)

        if first_error is not None:
            raise first_error

        return results
    
    @property
    @override
    def dim(self) -> int:
        """
        Returns embedding dimension.

        :raises RuntimeError: If ``dim`` was not provided and
            :meth:`initialize` has not been called yet.
        """
        if self._dim is None:
            raise RuntimeError(
                "Embedding dimension is not set. "
                "Either pass dim= to the constructor or call "
                "await embedder.initialize() first."
            )
        return self._dim
