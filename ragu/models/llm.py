from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from typing_extensions import override

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ragu.common.logger import logger
from ragu.models.openai import CachedAsyncOpenAI


T = TypeVar('T', BaseModel, str)


class LLM(ABC):
    """LLM interface to support various backends (openai, transformers etc.)."""

    @abstractmethod
    async def chat_completion(
        self,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        """
        Returns one chat completion response.

        :param conversation: OpenAI-format conversation messages.
        :param output_schema: ``str`` for plain text or ``BaseModel`` subclass
            for structured output.
        :param kwargs: Backend-specific options (for example temperature).
        :returns: Response value as plain text or validated schema instance.
        """

    async def batch_chat_completion(
        self,
        conversations: list[list[ChatCompletionMessageParam]],
        output_schema: type[T] = str,
        desc: str | None = None,
        continue_on_error: bool = False,
        **kwargs: Any,
    ) -> Sequence[T | None]:
        """
        Runs multiple :meth:`chat_completion` calls concurrently.

        :param conversations: List of conversation message lists.
        :param output_schema: Output schema applied to each call.
        :param desc: Optional tqdm progress description.
        :param continue_on_error: If ``True``, log a warning and return
            ``None`` for failed calls instead of raising.  If ``False``
            (default), raise on the first failure.
        :param kwargs: Extra kwargs forwarded to each call.
        :returns: Responses in the same order as input conversations.
            When *continue_on_error* is ``True``, failed items are
            represented as ``None`` so the caller can distinguish a
            legitimate empty response from an API error.
        """
        logger.debug(f'Calling batch_chat_completion with size {len(conversations)}')

        if not continue_on_error:
            return await tqdm_asyncio.gather(*[
                self.chat_completion(
                    conversation=conversation,
                    output_schema=output_schema,
                    **kwargs,
                )
                for conversation in conversations
            ], desc=desc)

        tasks = [
            asyncio.ensure_future(
                self.chat_completion(
                    conversation=conversation,
                    output_schema=output_schema,
                    **kwargs,
                )
            )
            for conversation in conversations
        ]

        task_to_idx = {id(t): i for i, t in enumerate(tasks)}
        results: list[T | None] = [None] * len(tasks)
        pending: set[asyncio.Future[Any]] = set(tasks)
        pbar = tqdm(total=len(tasks), desc=desc)

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                idx = task_to_idx[id(task)]
                try:
                    results[idx] = task.result()
                except Exception as e:
                    logger.warning(
                        "batch_chat_completion failed for item {}: {}: {}",
                        idx, type(e).__name__, e,
                    )
                    results[idx] = None
                pbar.update(1)

        pbar.close()
        return results
        


class LLMOpenAI(LLM):
    """Adapts :class:`CachedAsyncOpenAI` to the :class:`LLM` interface."""
    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        **kwargs: Any,
    ):
        """
        Adapts :class:`CachedAsyncOpenAI` to the :class:`LLM` interface.

        :param client: OpenAI-compatible backend client.
        :param model_name: Model identifier passed to backend calls.
        :param kwargs: Default kwargs merged into each ``chat_completion`` call.
        
        Example:
        ```
        llm = LLMOpenAI(
            client=CachedAsyncOpenAI(),
            model_name='gpt-5',
        )
        ```
        """
        self.client = client
        self.model_name = model_name
        self.kwargs = kwargs
    
    @override
    async def chat_completion(
        self,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        """
        Forwards chat completion request.

        :param conversation: OpenAI-format conversation messages.
        :param output_schema: ``str`` or ``BaseModel`` subclass.
        :param kwargs: Per-call kwargs merged with constructor kwargs.
            Per-call values override constructor defaults on conflicts.
        :returns: Response value as plain text or validated schema instance.
        """
        return await self.client.chat_completion(
            model_name=self.model_name,
            conversation=conversation,
            output_schema=output_schema,
            **(self.kwargs | kwargs),
        )
