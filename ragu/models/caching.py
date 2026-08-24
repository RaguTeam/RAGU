import json
from collections.abc import Mapping, MutableMapping
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeVar, Union

from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam

from ragu.common.logger import logger
from ragu.utils.ragu_utils import FLOATS, get_disk_cache


#: Schema a chat completion can be asked for: ``str`` for plain text, or a
#: ``BaseModel`` subclass for structured output.
OutputSchema = Union[type[BaseModel], type[str]]

#: Used by overloads to carry the concrete model class through to the return type.
ModelT = TypeVar('ModelT', bound=BaseModel)

_BASE64_MARKER = ';base64,'


def _digest_data_uri(text: str) -> str:
    """
    Replace the payload of an inline data URI with its digest.

    The header is kept, so ``image/png`` and ``image/jpeg`` remain different
    keys — which they must, since the model is told which it is getting.

    :param text: Any string; returned unchanged unless it is a base64 data URI.
    :type text: str
    :returns: The string with its base64 payload replaced by a digest.
    :rtype: str
    """
    if not text.startswith('data:') or _BASE64_MARKER not in text:
        return text

    header, payload = text.split(_BASE64_MARKER, 1)
    return f'{header}{_BASE64_MARKER}sha256:{sha256(payload.encode()).hexdigest()}'


def digest_inline_media(value: Any) -> Any:
    """
    Copy a conversation with every inline media payload reduced to a digest.

    Vision calls carry whole images base64-encoded in the message content. Cache
    keys are JSON strings held in memory and compared in full, so keying on the
    conversation verbatim makes every key as large as the page it asks about —
    and the key is stored a second time alongside the response. A digest keys
    the same image just as precisely for a fixed cost.

    This is the same treatment `_cached_transcribe` gives recordings; it is done
    here rather than in the OCR client so that any vision call benefits, and
    walks the structure generically so audio and other future part types are
    covered too.

    Conversations without inline media come back structurally identical, so
    their keys are unaffected.

    :param value: A conversation, or any part of one.
    :type value: Any
    :returns: A copy with base64 payloads replaced by their digests.
    :rtype: Any
    """
    if isinstance(value, str):
        return _digest_data_uri(value)
    if isinstance(value, Mapping):
        return {key: digest_inline_media(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [digest_inline_media(item) for item in value]
    return value


class ResponseCachingMixin:
    """
    Implements caching wrappers for abstract methods:

    - `_cached_chat_completion` (wrapper)
      and `_uncached_chat_completion` (abstract)
    - `_cached_embed_text` (wrapper)
      and `_uncached_embed_text` (abstract)

    ### How caching works

    This class uses abstract dict (str -> Any) as cache, typically this may
    be a dict() for in-memory caching, or diskcache.Index for disk
    caching.

    Caching key is calculated by combining method arguments
    and `cache_prefix`.

    Optionally subclasses may add more keyword arguments to
    `_cached_chat_completion`, or `_cached_embed_text`, such as `temperature`,
    `tools` etc, they will also be added in the caching key calculation. If
    you have object-level parameters, such as `temperature`, consider moving
    them into `_cached_chat_completion` or `_cached_embed_text` call arguments,
    so that temperature value is cached cofrrectly, or add them as `cache_prefix`.
    """
    def __init__(
        self,
        cache: MutableMapping[str, Any] | str | Path | None = None,
        cache_prefix: str = '',
    ):
        self.cache_prefix = cache_prefix
        match cache:
            case None:
                self.cache = None
            case str() | Path():
                self.cache = get_disk_cache(cache)
            case _:
                self.cache = cache

    async def _cached_chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: OutputSchema = str,
        **kwargs: Any,
    ) -> Any:
        """Caching wrapper for _uncached_chat_completion"""
        if self.cache is None:
            return await self._uncached_chat_completion(
                model_name=model_name,
                conversation=conversation,
                output_schema=output_schema,
                **kwargs,
            )

        args: dict[str, Any] = {
            'cache_prefix': self.cache_prefix,
            'model_name': model_name,
            'method': 'chat_completion',
            'conversation': digest_inline_media(conversation),
            'output_schema': (
                'str' if issubclass(output_schema, str) else output_schema.model_json_schema()
            ),
            'kwargs': kwargs,
        }
        key = json.dumps(args, sort_keys=True)

        if value := self.cache.get(key, None):
            logger.debug(f'Cache hit for {model_name}!')
            cached: str | dict[str, Any]
            _args, cached = value
            if issubclass(output_schema, str):
                return cached
            return output_schema.model_validate(cached)

        response = await self._uncached_chat_completion(
            model_name=model_name,
            conversation=conversation,
            output_schema=output_schema,
            **kwargs,
        )

        cached = response if issubclass(output_schema, str) else response.model_dump()

        self.cache[key] = args, cached

        return response

    async def _uncached_chat_completion(
        self,
        model_name: str,
        conversation: list[ChatCompletionMessageParam],
        output_schema: OutputSchema = str,
        **kwargs: Any,
    ) -> Any:
        """Abstract method to cache.

        kwargs are here to add custom arguments that will also be cached
        """
        raise NotImplementedError

    async def _cached_embed_text(
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        """Caching wrapper for _uncached_embed_text"""
        if self.cache is None:
            return await self._uncached_embed_text(
                model_name=model_name,
                text=text,
                **kwargs,
            )

        args: dict[str, Any] = {
            'cache_prefix': self.cache_prefix,
            'model_name': model_name,
            'method': 'embed_text',
            'text': text,
            'kwargs': kwargs,
        }
        key = json.dumps(args, sort_keys=True)

        if value := self.cache.get(key, None):
            logger.debug(f'Cache hit for {model_name}!')
            cached: list[float] | FLOATS
            _args, cached = value
            return cached

        response = await self._uncached_embed_text(
            model_name=model_name,
            text=text,
            **kwargs,
        )

        self.cache[key] = args, response

        return response

    async def _uncached_embed_text(
        self,
        model_name: str,
        text: str,
        **kwargs: Any,
    ) -> list[float] | FLOATS:
        """
        Abstract method to cache.

        kwargs are here to add custom arguments that will also be cached
        """
        raise NotImplementedError

    async def _cached_embed_texts(
        self,
        model_name: str,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float] | FLOATS]:
        """Batch-aware caching wrapper for _uncached_embed_texts.

        Checks the cache individually for each text, sends only cache
        misses to the API, then stores and returns the full result list
        in the original order.
        """
        if self.cache is None:
            return await self._uncached_embed_texts(
                model_name=model_name, texts=texts, **kwargs
            )

        results: list[list[float] | FLOATS | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_keys: list[str] = []
        for i, text in enumerate(texts):
            args: dict[str, Any] = {
                'cache_prefix': self.cache_prefix,
                'model_name': model_name,
                'method': 'embed_text',
                'text': text,
                'kwargs': kwargs,
            }
            key = json.dumps(args, sort_keys=True)
            if value := self.cache.get(key, None):
                _, cached = value
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_keys.append(key)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            embeddings = await self._uncached_embed_texts(
                model_name=model_name, texts=miss_texts, **kwargs
            )
            for idx, key, embedding in zip(miss_indices, miss_keys, embeddings):
                results[idx] = embedding
                self.cache[key] = args, embedding

        return results  # type: ignore[return-value]

    async def _uncached_embed_texts(
        self,
        model_name: str,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float] | FLOATS]:
        """
        Abstract batch embedding method.

        Subclasses must implement this to send multiple texts in a single
        API call.  kwargs are included for cache-key consistency.
        """
        raise NotImplementedError

    async def _cached_transcribe(
        self,
        model_name: str,
        audio: bytes,
        audio_name: str = 'audio.wav',
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Caching wrapper for _uncached_transcribe.

        :param model_name: Provider transcription model name.
        :param audio: Encoded audio bytes.
        :param audio_name: File name reported to the provider; its suffix
            tells the provider which container to expect.
        :param kwargs: Forwarded transcription options.
        :returns: Provider payload as a mapping.
        """
        if self.cache is None:
            return await self._uncached_transcribe(
                model_name=model_name,
                audio=audio,
                audio_name=audio_name,
                **kwargs,
            )

        args: dict[str, Any] = {
            'cache_prefix': self.cache_prefix,
            'model_name': model_name,
            'method': 'transcribe',
            'audio_sha256': sha256(audio).hexdigest(),
            'audio_suffix': Path(audio_name).suffix.lower(),
            'kwargs': kwargs,
        }
        key = json.dumps(args, sort_keys=True)

        if value := self.cache.get(key, None):
            logger.debug(f'Cache hit for {model_name}!')
            cached: dict[str, Any]
            _args, cached = value
            return cached

        response = await self._uncached_transcribe(
            model_name=model_name,
            audio=audio,
            audio_name=audio_name,
            **kwargs,
        )

        self.cache[key] = args, response

        return response

    async def _uncached_transcribe(
        self,
        model_name: str,
        audio: bytes,
        audio_name: str = 'audio.wav',
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Abstract method to cache.

        kwargs are here to add custom arguments that will also be cached
        """
        raise NotImplementedError

    async def _cached_score(
        self,
        model_name: str,
        text_1: str,
        text_2: list[str],
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Caching wrapper for _uncached__score"""
        args: dict[str, Any] = {
            'cache_prefix': self.cache_prefix,
            'model_name': model_name,
            'method': 'score',
            'text_1': text_1,
            'text_2': text_2,
            'kwargs': kwargs,
        }
        key = json.dumps(args, sort_keys=True)

        if self.cache is not None and (value := self.cache.get(key, None)):
            logger.debug(f'Cache hit for {model_name}!')
            cached: list[tuple[int, float]]
            _args, cached = value
            return cached

        # if self.cache is not None:
        #     logger.debug(f'Cache miss for {model_name}!')

        response = await self._uncached_score(
            model_name=model_name,
            text_1=text_1,
            text_2=text_2,
            **kwargs,
        )

        if self.cache is not None:
            self.cache[key] = args, response

        return response

    async def _uncached_score(
        self,
        model_name: str,
        text_1: str,
        text_2: list[str],
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """
        Abstract method to cache.

        kwargs are here to add custom arguments that will also be cached
        """
        raise NotImplementedError
