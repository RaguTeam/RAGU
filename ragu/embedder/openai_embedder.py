import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

from ragu.common.batch_generator import BatchGenerator
from ragu.common.cache import EmbeddingCache, get_cache, make_embedding_cache_key
from ragu.common.logger import logger
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.utils.ragu_utils import FLOATS, AsyncRunner


@dataclass(frozen=True, slots=True)
class PendingEmbeddingRequest:
    """
    Represents an embedding request pending generation (not found in cache).
    """
    index: int
    text: str
    cache_key: str


class OpenAIEmbedder(BaseEmbedder):
    """
    Async embedder for OpenAI-compatible embedding APIs.
    """

    def __init__(
            self,
            model_name: str,
            client: AsyncOpenAI,
            dim: int,
            concurrency: int = 8,
            max_requests_per_second: int = 1,
            max_requests_per_minute: int = 60,
            time_period: int | float = 1,
            cache_dir: str | Path | None = None,
    ):
        """
        Initialize OpenAI embedder client and runtime settings.

        :param model_name: Embedding model name.
        :param client: AsyncOpenAI client.
        :param dim: Embedding dimensionality.
        :param concurrency: Maximum concurrent embedding requests.
        :param max_requests_per_second: Requests-per-second limit.
        :param max_requests_per_minute: Requests-per-minute limit.
        :param time_period: Time period for the RPS limiter.
        :param cache_dir: Directory for caching requests.
        """
        super().__init__(dim=dim)

        self.model_name = model_name
        self.client = client

        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._rpm = AsyncLimiter(max_requests_per_minute, time_period=60) if max_requests_per_minute else None
        self._rps = AsyncLimiter(max_requests_per_second, time_period=time_period) if max_requests_per_second else None
        self._cache = get_cache(cache_dir) if cache_dir else None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _embed_via_api(self, text: str) -> List[float] | None:
        """
        Execute one embedding request with retry policy.

        :param text: Input text.
        :return: Embedding vector or ``None`` when unavailable.
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
            )
            return [item.embedding for item in response.data][0]
        except Exception as e:
            logger.error(f"[OpenAI API Embedder] Exception occurred: {e}")
            raise

    async def _embed_with_caching(self, text: str) -> list[float] | None:
        if self._cache is None:
            if self._rpm: await self._rpm.acquire()
            if self._rps: await self._rps.acquire()
            return await self._embed_via_api(text)
        key = make_embedding_cache_key(text, self.model_name)
        if (embedding := self._cache.get(key, None)) is None:
            print(f'Cache miss for {key}')
            if self._rpm: await self._rpm.acquire()
            if self._rps: await self._rps.acquire()
            embedding = self._cache[key] = await self._embed_via_api(text)
        else:
            print(f'Cache hit for {key}')
        return embedding

    async def embed(
        self,
        texts: list[str],
        progress_bar_desc: str | None = None,
    ) -> list[list[float]] | FLOATS:
        """
        Compute embeddings for a list of texts.

        :param texts: Input text or list of texts.
        :param progress_bar_desc: Optional progress bar description.
        :return: Embeddings aligned with input order.
        """
        async def _bounded_call(text: str) -> list[float] | None:
            async with self._sem:
                return await self._embed_with_caching(text)

        tasks = [_bounded_call(text) for text in texts]

        return await tqdm_asyncio.gather( # type: ignore
            *tasks,
            desc=progress_bar_desc,
            disable=progress_bar_desc is None,
        )

    async def aclose(self):
        """
        Close embedder resources and flush cache.
        """
        try:
            await self.client.close()
        except:
            pass
