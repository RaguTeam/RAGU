"""
Conversion from engine results to the wire schema.

The engines are part of this package and are versioned with it, so the
conversion dispatches on the concrete result types instead of probing for
attributes. A result type that changes shape then fails loudly here, rather
than degrading into an empty source list that the service would report as a
missing capability.
"""

from functools import singledispatch
from typing import Any

from ragu.api.models import SourceItem, SubqueryItem
from ragu.search_engine.base_engine import (
    SearchEngineResponse,
    SearchEngineRetrieve,
)
from ragu.search_engine.global_search import GlobalSearchResult
from ragu.search_engine.local_search import LocalSearchResult
from ragu.search_engine.mix_search import MixSearchResult
from ragu.search_engine.naive_search import NaiveSearchResult


def _as_score(value: Any) -> float | None:
    """
    Coerce an engine score to a float, or ``None`` when it is not numeric.

    :param value: Raw score reported by an engine.
    :return: The score as a float, or ``None``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@singledispatch
def _sources_from_result(result: Any) -> list[SourceItem]:
    """
    Flatten an engine-specific retrieval result into flat sources.

    The base implementation covers result types this module does not model;
    :func:`extract_sources` detects that case and falls back to the rendered
    retrieval context instead.

    :param result: Engine-specific retrieval payload.
    :return: Flat sources for the wire schema.
    """
    return []


# Identity of the fallback implementation, used to tell "no registered handler"
# apart from "handler ran and found nothing".
_UNREGISTERED = _sources_from_result.registry[object]


@_sources_from_result.register
def _naive_sources(result: NaiveSearchResult) -> list[SourceItem]:
    scores = result.scores
    return [
        SourceItem(
            id=chunk.id,
            type="chunk",
            content=chunk.content,
            score=_as_score(scores[index]) if index < len(scores) else None,
        )
        for index, chunk in enumerate(result.chunks)
    ]


@_sources_from_result.register
def _local_sources(result: LocalSearchResult) -> list[SourceItem]:
    sources = [
        SourceItem(
            id=entity.id,
            type="entity",
            content=f"{entity.entity_name} — {entity.description}".strip(" —"),
        )
        for entity in result.entities
    ]
    sources += [
        SourceItem(
            id=relation.id,
            type="relation",
            content=(
                f"{relation.subject_name} {relation.relation_type} "
                f"{relation.object_name}: {relation.description}"
            ).strip(),
        )
        for relation in result.relations
    ]
    sources += [
        SourceItem(
            id=summary.id,
            type="community_summary",
            content=summary.summary,
        )
        for summary in result.summaries
    ]
    sources += [
        SourceItem(id=chunk.id, type="chunk", content=chunk.content)
        for chunk in result.chunks
    ]
    return sources


@_sources_from_result.register
def _global_sources(result: GlobalSearchResult) -> list[SourceItem]:
    return [
        SourceItem(
            id=f"insight_{index}",
            type="community_summary",
            content=insight.response,
            score=_as_score(insight.rating),
        )
        for index, insight in enumerate(result.insights, start=1)
    ]


@_sources_from_result.register
def _mix_sources(result: MixSearchResult) -> list[SourceItem]:
    # Entries are child retrieval containers, or full child responses when
    # MixQueryParams.ensemble_responses is set. Children overlap — local and
    # naive both return chunks — so the union is deduplicated.
    sources: list[SourceItem] = []
    for entry in result.results:
        retrieval = entry.retrieval if isinstance(entry, SearchEngineResponse) else entry
        sources.extend(extract_sources(retrieval))
    return _deduplicated(sources)


def extract_sources(retrieval: SearchEngineRetrieve[Any] | None) -> list[SourceItem]:
    """
    Convert a retrieval container into flat sources.

    :param retrieval: Retrieval container returned by an engine, or ``None``
        when the engine produced no retrieval at all.
    :return: Flat sources; a single rendered-context source for result types
        this module does not model.
    """
    if retrieval is None:
        return []

    result = retrieval.result
    if _sources_from_result.dispatch(type(result)) is not _UNREGISTERED:
        return _sources_from_result(result)

    # A result type this module does not model (a new engine, or a custom one):
    # keep the rendered context as one source rather than dropping the evidence.
    text = retrieval.to_text()
    if text and text.strip():
        return [SourceItem(id="retrieval_context", type="context", content=text)]
    return []


def answer_text(response: Any) -> str:
    """
    Render an engine answer, which may be a string or a pydantic model.

    :param response: ``SearchEngineResponse.response`` value.
    :return: The answer as text.
    """
    if response is None:
        return ""
    return str(response) if isinstance(response, str) else _dump(response)


def _dump(response: Any) -> str:
    dump = getattr(response, "model_dump_json", None)
    return dump() if callable(dump) else str(response)


def extract_subqueries(payload: dict[str, Any]) -> list[SubqueryItem]:
    """
    Read the query plan's intermediate answers out of a response payload.

    ``QueryPlanEngine.batch_query`` sets ``payload`` to ``{subquery_id:
    SearchEngineResponse}``; every other engine leaves it empty.

    :param payload: ``SearchEngineResponse.payload``.
    :return: One item per answered subquery, in plan order.
    """
    return [
        SubqueryItem(query=response.query, answer=answer_text(response.response))
        for response in payload.values()
        if isinstance(response, SearchEngineResponse)
    ]


def _subquery_sources(payload: dict[str, Any]) -> list[SourceItem]:
    """
    Collect the sources every subquery retrieved.

    ``QueryPlanEngine`` answers the top-level query from the sink subquery
    alone, so the evidence gathered for the other subqueries would otherwise be
    dropped even though their answers are returned.

    :param payload: ``SearchEngineResponse.payload``.
    :return: Sources from every subquery, in plan order.
    """
    sources: list[SourceItem] = []
    for response in payload.values():
        if isinstance(response, SearchEngineResponse):
            sources.extend(extract_sources(response.retrieval))
    return sources


def _deduplicated(sources: list[SourceItem]) -> list[SourceItem]:
    """
    Drop repeated sources, keeping the first occurrence and its order.

    :param sources: Sources in the order they were collected.
    :return: Sources without duplicates.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[SourceItem] = []
    for source in sources:
        key = (source.id, source.type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def to_outcome(
    response: SearchEngineResponse,
    *,
    used_query_plan: bool,
) -> tuple[str, list[SourceItem], list[SubqueryItem]]:
    """
    Convert an engine response into the parts of the wire response.

    :param response: Response returned by the engine.
    :param used_query_plan: Whether the call went through
        :class:`~ragu.search_engine.query_plan.QueryPlanEngine`. The caller
        knows this, so it is not rediscovered from the payload.
    :return: Answer text, flat sources, and query-plan subqueries.
    """
    sources = extract_sources(response.retrieval)
    subqueries: list[SubqueryItem] = []

    if used_query_plan:
        payload = response.payload
        subqueries = extract_subqueries(payload)
        sources = _deduplicated(sources + _subquery_sources(payload))

    return answer_text(response.response), sources, subqueries
