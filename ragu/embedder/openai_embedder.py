import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

from ragu.common.batch_generator import BatchGenerator
from ragu.common.cache import EmbeddingCache, make_embedding_cache_key
from ragu.common.logger import logger
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.utils.ragu_utils import AsyncRunner


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
            base_url: str,
            api_token: str,
            dim: int,
            concurrency: int = 8,
            request_timeout: float = 60.0,
            max_requests_per_second: int = 1,
            max_requests_per_minute: int = 60,
            time_period: int | float = 1,
            use_cache: bool = False,
            cache_path: Optional[str | Path] = None,
            cache_flush_every: int=100,
            *args,
            **kwargs
    ):
        """
        Initialize OpenAI embedder client and runtime settings.

        :param model_name: Embedding model name.
        :param base_url: OpenAI-compatible API base URL.
        :param api_token: API key/token.
        :param dim: Embedding dimensionality.
        :param concurrency: Maximum concurrent embedding requests.
        :param request_timeout: HTTP request timeout in seconds.
        :param max_requests_per_second: Requests-per-second limit.
        :param max_requests_per_minute: Requests-per-minute limit.
        :param time_period: Time period for the RPS limiter.
        :param use_cache: Whether to cache embeddings locally.
        :param cache_path: Optional custom cache path.
        :param cache_flush_every: Cache flush frequency in writes.
        """
        super().__init__(dim=dim)

        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=api_token,
            base_url=base_url,
            timeout=request_timeout
        )

        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._rpm = AsyncLimiter(max_requests_per_minute, time_period=60) if max_requests_per_minute else None
        self._rps = AsyncLimiter(max_requests_per_second, time_period=time_period) if max_requests_per_second else None
        self._cache_flush_every = cache_flush_every

        self._use_cache = use_cache
        self._cache = EmbeddingCache(cache_path=cache_path, flush_every_n_writes=cache_flush_every)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _one_call(self, text: str) -> List[float] | None:
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

    async def embed(
            self,
            texts: Union[str, List[str]],
            progress_bar_desc=None
    ) -> List[List[float]]:
        """
        Compute embeddings for one text or a list of texts.

        :param texts: Input text or list of texts.
        :param progress_bar_desc: Optional progress bar description.
        :return: Embeddings aligned with input order.
        """
        if isinstance(texts, str):
            texts = [texts]

        results: List[Optional[List[float]]] = [None] * len(texts)
        pending: List[PendingEmbeddingRequest] = []

        # Check cache for all texts first
        for i, text in enumerate(texts):
            if self._cache is not None:
                cache_key = make_embedding_cache_key(text, self.model_name)
                cached = await self._cache.get(cache_key)
                if cached is not None:
                    results[i] = cached
                else:
                    pending.append(PendingEmbeddingRequest(i, text, cache_key))
            else:
                pending.append(PendingEmbeddingRequest(i, text, ""))

        logger.info(f"[OpenAIEmbedder]: Found {len(texts) - len(pending)}/{len(texts)} embeddings in cache.")

        if not pending:
            return results

        with tqdm_asyncio(total=len(pending), desc=progress_bar_desc) as pbar:
            runner = AsyncRunner(self._sem, self._rps, self._rpm, pbar)

            for batch in BatchGenerator(pending, self._cache_flush_every).get_batches():
                tasks = [runner.make_request(self._one_call, text=req.text) for req in batch]
                generated = await asyncio.gather(*tasks, return_exceptions=True)

                for req, value in zip(batch, generated):
                    if not isinstance(value, Exception) and value is not None:
                        if self._use_cache:
                            await self._cache.set(req.cache_key, value)
                        results[req.index] = value
                    else:
                        results[req.index] = None

                if self._use_cache:
                    await self._cache.flush_cache()

        return results

    async def aclose(self):
        """
        Close embedder resources and flush cache.
        """
        try:
            if self._cache is not None:
                await self._cache.close()
            await self.client.close()
        except Exception as e:
            pass
