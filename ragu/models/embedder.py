from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypeVar
from typing_extensions import override

from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

from ragu.common.logger import logger
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.ragu_utils import FLOATS


T = TypeVar('T', BaseModel, str)


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
    
    :param client: OpenAI-compatible backend client.
    :param model_name: Embedding model identifier.
    :param dim: Embedding dimension. If ``None``, it is auto-detected on the
        first call to :meth:`initialize` by sending a probe request.
    :param kwargs: Default kwargs merged into each ``embed_text`` call.
    
    Example:
    ```
    embedder = EmbedderOpenAI(
        client=CachedAsyncOpenAI(),
        model_name='my-embedding-model',
    )
    await embedder.initialize()  # auto-detects dim if not provided
    ```
    """
    
    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        dim: int | None = None,
        **kwargs: Any,
    ):
        self.client = client
        self.model_name = model_name
        self.kwargs = kwargs
        self._dim = dim

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
