import asyncio
import logging
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, TypeVar, cast
import httpx
from pydantic import BaseModel
from typing_extensions import override

from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ParsedChatCompletion,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_chain, wait_fixed, before_sleep_log
from aiolimiter import AsyncLimiter

from ragu.models.caching import ResponseCachingMixin
from ragu.utils.ragu_utils import FLOATS, LoguruAdapter, attach_async_contexts, get_disk_cache, save_args_on_exception
from ragu.common.logger import logger


T = TypeVar('T', BaseModel, str)

DEFAULT_RETRY_TIMES_SEC: Sequence[float] = (2, 4, 8)


def _is_retryable_exception(exc: BaseException) -> bool:
    from openai import APITimeoutError, InternalServerError, RateLimitError, APIConnectionError
    return isinstance(exc, (APITimeoutError, InternalServerError, RateLimitError, APIConnectionError))


@dataclass
class CachedAsyncOpenAI(ResponseCachingMixin):
    """
    OpenAI client able to respond with structured outputs and
    embeddings, with response caching, rate limiting and request retrying.

    If `client` is provided, the arguments `base_url` and `api_key`
    are not used. Otherwise, a new `AsyncOpenAI` client is constructed.

    ### Schema handling

    If `output_schema == str`, runs `client.chat.completions.create`
    and returns the `response.choices[0].message.content`.

    If `output_schema != str`, then an additional parameter `as_tool`
    offers two different ways to handle the `output_schema`. The
    correctness and quality of the responses is model-dependent and
    provider-dependent:
    
    - If `as_tool=True`: calls `client.chat.completions.create` and
      passed `tool_definition` that contain the output format schema.
    - If `as_tool=False`: calls `client.beta.chat.completions.parse` and
      passed the `response_format` argument.

    ### Rate limits and retrying
    
    Rates can be controlled by:
    - `rate_min_delay`: min delay in seconds between requests
    - `rate_max_per_minute`: max requests per minute
    - `rate_max_simultaneous`: max simultaneous requests

    Allows retrying: for example, if `retry_times_sec=(2, 4, 8)`, will
    retry in 2, then 4, then 8 seconds on a retryable exception, and
    finally raise it. Only transient errors (APITimeoutError,
    InternalServerError, RateLimitError, APIConnectionError) are
    retried. Non-retryable errors (ContentFilterFinishReasonError,
    ValueError, BadRequestError, etc.) are raised immediately.
    In rate limiting, each retrying attempt is considered a new request.

    NOTE: This class sets max_retries=0 in AsyncOpenAI, because
    it uses its own `retry_times_sec` mechanism.

    So, these mechanisms are independent: rate limiting delays
    requests, and retrying handles exceptions.

    ### Response caching

    Typically, pass `cache="my_cache_dir/"` to enable caching. For
    details see the base class.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        rate_min_delay: float | None = None,
        rate_max_per_minute: int | None = None,
        rate_max_simultaneous: int | None = None,
        retry_times_sec: Sequence[float] | None = DEFAULT_RETRY_TIMES_SEC,
        max_completion_tokens: int | None = None,
        cache: MutableMapping[str, Any] | str | Path | None = None,
        cache_prefix: str = 'openai',
        debug_errors_storage: MutableMapping[str, Any] | str | Path | None = None,
        embed_timeout: float | None = 60.0,
    ):
        """
        Initializes backend client.

        :param base_url: Base URL for OpenAI-compatible provider.
        :param api_key: API key for provider authentication.
        :param client: Preconfigured AsyncOpenAI client. If provided, ``base_url``
            and ``api_key`` are ignored.
        :param rate_min_delay: Minimum delay in seconds between request starts.
        :param rate_max_per_minute: Maximum number of requests per minute.
        :param rate_max_simultaneous: Maximum number of concurrent requests.
        :param retry_times_sec: Retry wait schedule in seconds, e.g. ``(4, 8)``.
            Defaults to ``(2, 4, 8)`` (3 retries with exponential backoff).
            Set to ``None`` to disable retries. Only retryable exceptions
            (APITimeoutError, InternalServerError, RateLimitError,
            APIConnectionError) are retried.
        :param max_completion_tokens: Maximum number of tokens in the completion.
            Passed to ``max_completion_tokens`` in API calls. Useful when
            structured output parsing fails due to length limits.
        :param cache: Optional cache mapping or path accepted by
            :class:`ResponseCachingMixin`.
        :param cache_prefix: Prefix included in cache keys.
        :param debug_errors_storage: Optional mapping/path to store failing call
            arguments for debugging.
        :param embed_timeout: Per-request timeout in seconds for embedding API
            calls.  Defaults to ``60.0``.  Set to ``None`` to use the client
            default (typically 600 s).
        """
        self.max_completion_tokens = max_completion_tokens
        self.embed_timeout = embed_timeout
        self.client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
        )

        # saving successuful responses
        ResponseCachingMixin.__init__(self, cache=cache, cache_prefix=cache_prefix)

        # storing original unwrapped medthods to be able to call them for debugging
        self._uncached_raw_chat_completion = self._uncached_chat_completion
        self._uncached_raw_embed_text = self._uncached_embed_text
        self._uncached_raw_embed_texts = self._uncached_embed_texts
        self._uncached_raw_score = self._uncached_score

        # saving errors to debug
        self.debug_errors_storage: MutableMapping[str, Any] | None
        match debug_errors_storage:
            case None:
                self.debug_errors_storage = None
            case str() | Path():
                self.debug_errors_storage = get_disk_cache(debug_errors_storage)
            case _:
                self.debug_errors_storage = debug_errors_storage
        if self.debug_errors_storage is not None:
            self._uncached_chat_completion = save_args_on_exception(
                self._uncached_chat_completion, self.debug_errors_storage)
            self._uncached_embed_text = save_args_on_exception(
                self._uncached_embed_text, self.debug_errors_storage)
            self._uncached_embed_texts = save_args_on_exception(
                self._uncached_embed_texts, self.debug_errors_storage)
            self._uncached_score = save_args_on_exception(
                self._uncached_score, self.debug_errors_storage)

        # Handlers/wrappers will be called in this order:
        # 1. Caching
        # 2. Retrying
        # 3. Rate limiting

        # add rate limiter contexts
        contexts: list[AbstractAsyncContextManager[Any]] = []
        if rate_max_per_minute:
            contexts.append(AsyncLimiter(rate_max_per_minute, time_period=60))
        if rate_max_simultaneous:
            contexts.append(asyncio.Semaphore(rate_max_simultaneous))
        if rate_min_delay:
            contexts.append(AsyncLimiter(1, time_period=rate_min_delay))
        if contexts:
            self._uncached_chat_completion = attach_async_contexts(
                self._uncached_chat_completion, *contexts)
            self._uncached_embed_text = attach_async_contexts(
                self._uncached_embed_text, *contexts)
            self._uncached_embed_texts = attach_async_contexts(
                self._uncached_embed_texts, *contexts)
            self._uncached_score = attach_async_contexts(
                self._uncached_score, *contexts)

        # add retrying decorators
        if retry_times_sec:
            retrying_decorator = retry(
                retry=retry_if_exception(_is_retryable_exception),
                stop=stop_after_attempt(len(retry_times_sec) + 1),
                wait=wait_chain(*[wait_fixed(t) for t in retry_times_sec]),
                before_sleep=before_sleep_log(
                    LoguruAdapter('logger'), logging.WARNING
                ),
                reraise=True
            )
            self._uncached_chat_completion = retrying_decorator(self._uncached_chat_completion)
            self._uncached_embed_text = retrying_decorator(self._uncached_embed_text)
            self._uncached_embed_texts = retrying_decorator(self._uncached_embed_texts)
            self._uncached_score = retrying_decorator(self._uncached_score)
    
    async def chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        """
        Returns chat completion result with caching.

        :param model_name: Provider model name.
        :param conversation: OpenAI-format chat messages.
        :param output_schema: ``str`` for text output or ``BaseModel`` subclass
            for structured output.
        :param kwargs: Forwarded generation options (for example temperature).
        :returns: Model response as ``str`` or validated schema instance.
        """
        return await self._cached_chat_completion(
            model_name=model_name,
            conversation=conversation,
            output_schema=output_schema,
            **kwargs
        )
    
    async def embed_text(  # with caching
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        """
        Returns text embedding with caching.

        :param model_name: Provider embedding model name.
        :param text: Input text to embed.
        :param kwargs: Extra backend-specific options.
        :returns: Vector embedding.
        """
        return await self._cached_embed_text(
            model_name=model_name,
            text=text,
            **kwargs,
        )
    
    async def embed_texts(
        self,
        model_name: str,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float] | FLOATS]:
        """
        Returns batch text embeddings with caching.

        Sends multiple texts in a single API call (when cache misses
        occur) for significantly better throughput than calling
        :meth:`embed_text` per text.

        :param model_name: Provider embedding model name.
        :param texts: Input texts to embed.
        :param kwargs: Extra backend-specific options.
        :returns: List of vector embeddings in the same order as input.
        """
        return await self._cached_embed_texts(
            model_name=model_name,
            texts=texts,
            **kwargs,
        )
    
    async def score(
        self,
        model_name: str,
        text_1: str,
        text_2: list[str],
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """
        Returns rerank scores.

        :param model_name: Provider reranker model name.
        :param text_1: Source or query text.
        :param text_2: Candidate texts to score against ``text_1``.
        :param kwargs: Extra backend-specific options.
        :returns: ``(index, score)`` tuples sorted by score descending.
        """
        return await self._cached_score(
            model_name=model_name,
            text_1=text_1,
            text_2=text_2,
            **kwargs,
        )

    @override
    async def _uncached_chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        as_tool: bool = False,
        **kwargs: Any,
    ) -> T:
        """
        Performs uncached chat completion call.

        :param model_name: Provider model name.
        :param conversation: OpenAI-format chat messages.
        :param output_schema: ``str`` for text output or ``BaseModel`` subclass.
        :param as_tool: If ``True``, enforces schema via tool-calling API.
            If ``False``, uses ``beta.chat.completions.parse``.
        :param kwargs: Supported generation options.
        :returns: Raw text or parsed schema instance.
        """
        logger.debug(f'Sending chat_completion API request with schema {output_schema.__name__}...')
        recognized_kwargs = {
            k: kwargs.pop(k, omit)
            for k in ['temperature', 'top_p']
        }
        assert not kwargs, f'Guard triggered: add this to supported kwargs: {kwargs}'
        if self.max_completion_tokens is not None:
            recognized_kwargs['max_completion_tokens'] = self.max_completion_tokens
        if issubclass(output_schema, str):
            response = cast(ChatCompletion, await self.client.chat.completions.create(
                model=model_name,
                messages=conversation,
                **recognized_kwargs,
            ))
            content = response.choices[0].message.content
            return cast(T, content if content is not None else '')

        model_schema = output_schema
        
        if not as_tool:
            # use response_format param
            parsed_completion = cast(
                ParsedChatCompletion[BaseModel],
                await self.client.beta.chat.completions.parse(
                    model=model_name,
                    messages=conversation,
                    response_format=model_schema,
                    **recognized_kwargs,
                )
            )
            
            parsed_result = parsed_completion.choices[0].message.parsed
            
            if parsed_result is None:
                raise ValueError('OpenAI refused to output structured data.')
            return cast(T, parsed_result)

        else:
            # use tool calling to define schema, as in pydantic_ai
            function_name = model_schema.__name__
            tool_definition: ChatCompletionFunctionToolParam = {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": f"Output data in the structure of {function_name}",
                    "parameters": model_schema.model_json_schema(), # type: ignore
                },
            }

            response = cast(ChatCompletion, await self.client.chat.completions.create(
                model=model_name,
                messages=conversation,
                tools=[tool_definition],
                tool_choice={"type": "function", "function": {"name": function_name}},
                **recognized_kwargs,
            ))

            message = response.choices[0].message
            
            if not message.tool_calls:
                raise ValueError('Model did not call the expected tool.')
            
            # Parse the arguments from the tool call back into the Pydantic model
            arguments_json = cast(str, message.tool_calls[0].function.arguments) # type: ignore
            return cast(T, model_schema.model_validate_json(arguments_json))

    @override
    async def _uncached_embed_text(
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        """
        Performs uncached embedding call.

        :param model_name: Provider embedding model name.
        :param text: Input text to embed.
        :param kwargs: Supported backend options.
        :returns: Vector embedding.
        """
        debug_text = text[:20].replace("\n", "\\n")
        logger.debug(f'Sending embed_text API request with text {debug_text}...')
        assert not kwargs, f'Guard triggered: add this to supported kwargs: {kwargs}'
        timeout_kwarg: dict[str, Any] = (
            {'timeout': self.embed_timeout} if self.embed_timeout is not None else {}
        )
        response = await self.client.embeddings.create(
            model=model_name, input=text, **timeout_kwarg,
        )
        return response.data[0].embedding

    @override
    async def _uncached_embed_texts(
        self,
        model_name: str,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float] | FLOATS]:
        """
        Performs uncached batch embedding call.

        Sends all texts in a single ``embeddings.create`` request.  The
        OpenAI-compatible API accepts ``input`` as a list of strings.

        :param model_name: Provider embedding model name.
        :param texts: Input texts to embed.
        :param kwargs: Supported backend options.
        :returns: List of vector embeddings in input order.
        """
        logger.debug(
            f'Sending embed_texts API request with {len(texts)} texts...'
        )
        assert not kwargs, f'Guard triggered: add this to supported kwargs: {kwargs}'
        timeout_kwarg: dict[str, Any] = (
            {'timeout': self.embed_timeout} if self.embed_timeout is not None else {}
        )
        response = await self.client.embeddings.create(
            model=model_name, input=texts, **timeout_kwarg,
        )
        return [item.embedding for item in response.data]

    @override
    async def _uncached_score(
        self,
        model_name: str,
        text_1: str,
        text_2: list[str],
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """
        Performs uncached rerank scoring call.

        :param model_name: Provider reranker model name.
        :param text_1: Source or query text.
        :param text_2: Candidate texts.
        :param kwargs: Supported backend options.
        :returns: ``(index, score)`` tuples sorted by score descending.
        """
        debug_text = text_1[:20].replace("\n", "\\n")
        logger.debug(f'Sending embed_text API request with text {debug_text}...')
        assert not kwargs, f'Guard triggered: add this to supported kwargs: {kwargs}'
        
        headers = {"Content-Type": "application/json"}
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        payload: dict[str, Any] = {
            "model": model_name,
            "text_1": text_1,
            "text_2": text_2,
        }

        async with httpx.AsyncClient(timeout=60) as http_client:   # TODO move to args
            response = await http_client.post(
                f"{self.client.base_url!s}score",
                headers=headers,
                json=payload,
            )
        response.raise_for_status()
        data = response.json()

        results = [(int(item["index"]), float(item["score"])) for item in data["data"]]
        results.sort(key=lambda x: x[1], reverse=True)

        return results
