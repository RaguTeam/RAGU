"""Reusable HTTP client for Jev decisions on TypeSafe, OpenRouter or a proxy."""

import asyncio
import json
import math
import time
from collections.abc import MutableMapping, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
from aiolimiter import AsyncLimiter
from typing_extensions import override

from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.models.caching import ResponseCachingMixin
from ragu.models.decisions.jev import _parse_jev
from ragu.models.decisions._schemas import _Request, _TypeSafeConfig
from ragu.models.decisions.types import (
    ChoiceQuestion, DecisionQuestion, DecisionResult, DecisionState, ScoreQuestion,
)


_RETRY_STATUS = {429, 500, 502, 503, 504}


def _retry_after(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        delay = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            delay = (date - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'TypeSafe/Jev response.{key}: duplicate JSON field')
        result[key] = value
    return result


class TypeSafeClient(ResponseCachingMixin):
    def __init__(
        self,
        api_key: str, *,
        url: str = 'https://api.typesafe.ai/v1/systemone',
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        cache: MutableMapping[str, Any] | str | Path | None = None,
        cache_prefix: str = 'typesafe',
        rate_min_delay: float | None = None,
        rate_max_per_minute: int | None = None,
        rate_max_simultaneous: int | None = None,
        retry_times_sec: Sequence[float] | None = (2, 4, 8),
    ) -> None:
        """Configure a Jev operation without binding the client to one model.

        :param api_key: Bearer token supplied by the caller.
        :param url: Complete absolute POST URL, including the operation path.
        :param http_client: Optional caller-owned connection pool.
        :param timeout: Timeout in seconds for each network attempt.
        :param cache: Mapping or disk path; defaults to ``Settings.cache_path``.
        :param cache_prefix: Cache namespace, e.g. an experiment identifier.
        :param rate_min_delay: Minimum interval between requests in seconds.
        :param rate_max_per_minute: Maximum request starts per minute.
        :param rate_max_simultaneous: Maximum concurrent attempts.
        :param retry_times_sec: Retry schedule; empty or ``None`` disables retries.
        """
        _TypeSafeConfig.model_validate({
            'api_key': api_key,
            'url': url,
            'timeout': timeout,
            'rate_min_delay': rate_min_delay,
            'rate_max_per_minute': rate_max_per_minute,
            'rate_max_simultaneous': rate_max_simultaneous,
            'retry_times_sec': retry_times_sec,
        })

        super().__init__(cache=cache if cache is not None else Settings.cache_path, cache_prefix=cache_prefix)

        self.url = url
        self.timeout = timeout
        self.retry_times_sec = tuple(retry_times_sec or ())
        self._api_key = api_key
        self._owns_client = http_client is None
        self._closed = False
        self.http_client = http_client if http_client is not None else httpx.AsyncClient()

        self._rate_limit_contexts = []
        if rate_max_per_minute:
            self._rate_limit_contexts.append(AsyncLimiter(rate_max_per_minute, time_period=60))
        if rate_max_simultaneous:
            self._rate_limit_contexts.append(asyncio.Semaphore(rate_max_simultaneous))
        if rate_min_delay:
            self._rate_limit_contexts.append(AsyncLimiter(1, time_period=rate_min_delay))

    async def __aenter__(self) -> 'TypeSafeClient':
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close only an internally created HTTP pool; repeated calls are harmless."""
        if not self._closed:
            if self._owns_client:
                await self.http_client.aclose()
            self._closed = True

    async def decide(
        self, model_name: str, *, state: DecisionState, questions: Sequence[DecisionQuestion],
    ) -> DecisionResult:
        """Evaluate questions about text or JSON through the configured Jev endpoint.

        :param model_name: Model identifier for the configured endpoint.
        :param state: Text, JSON object or array.
        :param questions: Nonempty sequence with unique question names.
        :return: Validated answers in question order with available metadata.
        """
        _Request.model_validate(
            {
                'model_name': model_name,
                'state': state,
                'questions': questions
            },
            context={
                'protocol': 'jev'
            },
        )
        decision_questions = {}
        for question in questions:
            item: dict[str, Any] = {
                'type': 'noul',
                'instructions': question.instructions
            }
            if isinstance(question, ChoiceQuestion):
                item.update(
                    type='choice',
                    criteria={
                        option.value: option.description for option in question.choices
                    }
                )
            elif isinstance(question, ScoreQuestion):
                item.update(
                    type='score',
                    criteria=[f'{level.label} — {level.description}' for level in question.levels]
                )
            decision_questions[question.name] = item
        request = deepcopy({
            'model': model_name,
            'state': state,
            'questions': decision_questions
        })
        questions = deepcopy(questions)
        return await self._cached_decide(
            model_name,
            request=request,
            questions=questions,
            protocol='jev',
            url=self.url,
        )

    @override
    async def _uncached_decide(
        self, model_name: str, *, request: dict[str, Any], questions: Sequence[DecisionQuestion],
    ) -> DecisionResult:
        for attempt in range(len(self.retry_times_sec) + 1):
            delay = 0.0
            start = time.monotonic()
            status: int | str = 'error'
            try:
                async with AsyncExitStack() as stack:
                    for context in self._rate_limit_contexts:
                        await stack.enter_async_context(context)
                    response = await self.http_client.post(
                        self.url,
                        json=request,
                        timeout=self.timeout,
                        headers={'Authorization': f'Bearer {self._api_key}'},
                    )
                    status = response.status_code
                    response.raise_for_status()
                try:
                    raw = json.loads(response.content, object_pairs_hook=_unique_object)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError(f'TypeSafe/Jev response JSON: {exc}') from None
                return _parse_jev(raw, model_name, questions)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRY_STATUS or attempt == len(self.retry_times_sec):
                    raise
                delay = _retry_after(exc.response.headers.get('Retry-After'))
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
                if attempt == len(self.retry_times_sec):
                    raise
            finally:
                logger.debug(
                    'TypeSafe/Jev: model={}, questions={}, status={}, trial={}, time={:.3f} с',
                    model_name, len(questions), status, attempt + 1, time.monotonic() - start,
                )
            # AsyncExitStack has released the concurrency slot before retry backoff.
            await asyncio.sleep(max(delay, self.retry_times_sec[attempt]))
        raise AssertionError('unreachable')
