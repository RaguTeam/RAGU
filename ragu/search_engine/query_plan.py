from collections.abc import AsyncIterator
from typing import Any, Dict, List

from ragu.common.prompts.default_models import SubQuery, QueryPlan, RewriteQuery
from ragu.common.prompts.messages import ChatMessages, render
from ragu.common.prompts.prompt_storage import RAGUInstruction, require_prompt_schema
from ragu.search_engine.base_engine import (
    BaseEngine,
    SearchEngineResponse,
    SearchEngineStreamEvent,
    SearchEngineRetrieve,
    EngineParams,
)
from ragu.search_engine.search_functional import _topological_sort
from typing_extensions import override


class QueryPlanEngine(BaseEngine[EngineParams, SearchEngineRetrieve[Any]]):
    """
    Query planning engine that decomposes complex queries into a DAG of subqueries
    and executes them with an underlying engine.

    Pipeline (batched):
      1. Decompose every input query into a DAG of subqueries in one batched call.
      2. Execute all plans in lockstep by dependency readiness: at each step the
         subqueries that are ready across *all* plans form a single frontier.
      3. The frontier is rewritten (dependency-bearing subqueries only) and
         answered through one batched ``engine.batch_query`` call, so independent
         subqueries — even from different top-level queries — are answered
         together.
      4. Each input query returns the answer of its plan's final (sink) subquery.
    """

    def __init__(self, engine: BaseEngine[Any, Any]):
        """
        Initialize a query planner around an existing search engine.

        :param engine: Engine used to answer each planned subquery.
        """
        _PROMPTS_NAMES = ["query_decomposition", "query_rewrite"]
        super().__init__(engine.llm, prompts=_PROMPTS_NAMES)
        self.engine: BaseEngine[Any, Any] = engine

    async def process_queries(self, queries: List[str]) -> List[List[SubQuery]]:
        """
        Decompose a batch of complex queries into subquery DAGs in one LLM call.

        :param queries: Complex natural-language queries to decompose.
        :return: One list of :class:`SubQuery` (a DAG) per input query, aligned
            with ``queries``.
        """
        if not queries:
            return []

        instruction: RAGUInstruction = self.get_prompt("query_decomposition")
        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=queries,
        )
        plans: List[QueryPlan] = await self.engine.llm.batch_chat_completion(
            [rendered.to_openai() for rendered in rendered_list],
            output_schema=require_prompt_schema(
                instruction, "query_decomposition", QueryPlan
            ),
            desc="QueryPlan decompose",
        )
        return [plan.subqueries for plan in plans]

    async def process_query(self, query: str) -> List[SubQuery]:
        """
        Decompose a single complex query into a subquery DAG.

        Single-query wrapper around :meth:`process_queries`.

        :param query: Complex natural-language query to decompose.
        :return: List of :class:`SubQuery` objects forming a DAG.
        """
        return (await self.process_queries([query]))[0]

    async def _rewrite_subqueries(
        self,
        subqueries: List[SubQuery],
        contexts: List[Dict[str, SearchEngineResponse]],
    ) -> List[SubQuery]:
        """
        Rewrite a batch of subqueries in one LLM call.

        Each subquery is made self-contained by injecting the answers of the
        dependency subqueries listed in its ``depends_on``.

        :param subqueries: Subqueries to rewrite.
        :param contexts: Per-subquery answer maps (subquery id -> response),
            aligned with ``subqueries``.
        :return: Rewritten subqueries aligned with ``subqueries``.
        """
        if not subqueries:
            return []

        instruction: RAGUInstruction = self.get_prompt("query_rewrite")
        dep_contexts = [
            {k: v for k, v in context.items() if k in subquery.depends_on}
            for subquery, context in zip(subqueries, contexts)
        ]
        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            original_query=[subquery.query for subquery in subqueries],
            context=dep_contexts,
        )
        responses = await self.engine.llm.batch_chat_completion(
            [rendered.to_openai() for rendered in rendered_list],
            output_schema=require_prompt_schema(
                instruction, "query_rewrite", RewriteQuery
            ),
            desc="QueryPlan rewrite",
        )

        rewritten: List[SubQuery] = []
        for subquery, response in zip(subqueries, responses):
            text = response.query if isinstance(response, RewriteQuery) else response
            rewritten.append(subquery.model_copy(update={"query": text}))
        return rewritten

    async def _rewrite_subquery(
        self,
        subquery: SubQuery,
        context: Dict[str, SearchEngineResponse],
    ) -> SubQuery:
        """
        Rewrite a single subquery using its dependency answers.

        Single-subquery wrapper around :meth:`_rewrite_subqueries`.

        :param subquery: The subquery to rewrite.
        :param context: Mapping of subquery id to prior ``SearchEngineResponse``.
        :return: Copy of ``subquery`` with a rewritten, self-contained query.
        """
        return (await self._rewrite_subqueries([subquery], [context]))[0]

    async def _rewrite_frontier(
        self,
        subqueries: List[SubQuery],
        contexts: List[Dict[str, SearchEngineResponse]],
    ) -> List[SubQuery]:
        """
        Rewrite the dependency-bearing subqueries of a frontier in one batch.

        Subqueries with no dependencies are returned unchanged (their text is
        already self-contained).

        :param subqueries: Frontier subqueries to (possibly) rewrite.
        :param contexts: Per-subquery answer maps, aligned with ``subqueries``.
        :return: Frontier subqueries with dependency-bearing ones rewritten.
        """
        rewritten = list(subqueries)
        indexed = [
            (i, subquery, context)
            for i, (subquery, context) in enumerate(zip(subqueries, contexts))
            if subquery.depends_on
        ]
        if indexed:
            results = await self._rewrite_subqueries(
                [subquery for _, subquery, _ in indexed],
                [context for _, _, context in indexed],
            )
            for (i, _, _), new_subquery in zip(indexed, results):
                rewritten[i] = new_subquery
        return rewritten

    def _assemble_response(
        self,
        query: str,
        plan: List[SubQuery],
        answers: Dict[str, SearchEngineResponse],
    ) -> SearchEngineResponse:
        """
        Build the final response for one top-level query from its answered plan.

        :param query: The original top-level query.
        :param plan: The query's subquery DAG.
        :param answers: Answers for every subquery in the plan, keyed by id.
        :return: ``SearchEngineResponse`` whose answer/retrieval come from the
            plan's sink subquery and whose ``payload`` holds all subquery answers.
        """
        if not plan:
            return SearchEngineResponse(
                query=query, response="", retrieval=None, payload={}  # type: ignore[arg-type]
            )

        sink = _topological_sort(plan)[-1]
        sink_response = answers[sink.id]
        return SearchEngineResponse(
            query=query,
            response=sink_response.response,
            retrieval=sink_response.retrieval,
            payload=dict(answers),
        )

    @override
    async def query(self, query: str, params: EngineParams | None = None) -> SearchEngineResponse:
        """
        Execute a complex query using the plan-and-execute pipeline.

        Single-query delegate; the implementation lives in :meth:`batch_query`.

        :param query: The complex natural-language query to answer.
        :param params: Query parameters forwarded to the underlying engine.
        :return: ``SearchEngineResponse`` whose ``response`` is the final subquery
            answer, ``retrieval`` is the final subquery retrieval, and ``payload``
            contains all subquery responses by id.
        """
        return (await self.batch_query([query], params))[0]

    @override
    async def batch_query(
        self,
        queries: List[str],
        params: EngineParams | None = None,
    ) -> List[SearchEngineResponse]:
        """
        Execute the plan-and-execute pipeline for a batch of queries.

        All queries are decomposed in one batched call, then every plan is run in
        lockstep: at each step the subqueries that are ready across all plans are
        rewritten and answered through a single batched ``engine.batch_query``
        call. This merges independent subqueries — including those from different
        top-level queries — into the same child batch.

        :param queries: The complex natural-language queries to answer.
        :param params: Query parameters forwarded to the underlying engine.
        :return: ``SearchEngineResponse`` objects aligned with ``queries``.
        :raises ValueError: If a plan's dependencies contain a cycle or reference
            an unknown subquery id.
        """
        if not queries:
            return []

        plans: List[List[SubQuery]] = await self.process_queries(queries)

        answers: List[Dict[str, SearchEngineResponse]] = [{} for _ in plans]
        pending: List[List[SubQuery]] = [list(plan) for plan in plans]

        while any(pending):
            frontier_plan_idx: List[int] = []
            frontier: List[SubQuery] = []
            for plan_idx, plan_pending in enumerate(pending):
                for subquery in plan_pending:
                    if all(dep in answers[plan_idx] for dep in subquery.depends_on):
                        frontier_plan_idx.append(plan_idx)
                        frontier.append(subquery)

            if not frontier:
                raise ValueError("Query plan dependencies contain a cycle or an unknown id")

            rewritten = await self._rewrite_frontier(
                frontier,
                [answers[plan_idx] for plan_idx in frontier_plan_idx],
            )

            responses = await self.engine.batch_query(
                [subquery.query for subquery in rewritten],
                params,
            )

            for plan_idx, subquery, response in zip(frontier_plan_idx, frontier, responses):
                answers[plan_idx][subquery.id] = response

            for plan_idx in set(frontier_plan_idx):
                resolved = answers[plan_idx]
                pending[plan_idx] = [sq for sq in pending[plan_idx] if sq.id not in resolved]

        return [
            self._assemble_response(query, plan, plan_answers)
            for query, plan, plan_answers in zip(queries, plans, answers)
        ]

    @override
    async def stream_query(
        self,
        query: str,
        params: EngineParams | None = None,
    ) -> AsyncIterator[SearchEngineStreamEvent]:
        """
        Execute a planned query and stream the final sink subquery answer.

        Dependency subqueries are executed normally because their complete
        answers are needed to rewrite downstream subqueries. Once the sink
        subquery is ready, it is rewritten and delegated to the wrapped engine's
        ``stream_query`` method.

        :param query: The complex natural-language query to answer.
        :param params: Query parameters forwarded to the underlying engine.
        :returns: Async iterator of final-answer text deltas.
        :raises ValueError: If plan dependencies contain a cycle or reference an
            unknown subquery id.
        """
        plan = await self.process_query(query)
        if not plan:
            return

        sink = _topological_sort(plan)[-1]
        answers: Dict[str, SearchEngineResponse] = {}
        pending: List[SubQuery] = list(plan)

        while pending:
            frontier = [
                subquery
                for subquery in pending
                if all(dep in answers for dep in subquery.depends_on)
            ]
            if not frontier:
                raise ValueError("Query plan dependencies contain a cycle or an unknown id")

            non_sink_frontier = [subquery for subquery in frontier if subquery.id != sink.id]
            if non_sink_frontier:
                rewritten = await self._rewrite_frontier(
                    non_sink_frontier,
                    [answers for _ in non_sink_frontier],
                )
                responses = await self.engine.batch_query(
                    [subquery.query for subquery in rewritten],
                    params,
                )
                for subquery, response in zip(non_sink_frontier, responses):
                    answers[subquery.id] = response

                pending = [subquery for subquery in pending if subquery.id not in answers]
                continue

            rewritten_sink = await self._rewrite_subquery(sink, answers) if sink.depends_on else sink
            payload = {
                "answers": dict(answers),
                "plan": plan,
                "sink_subquery": rewritten_sink,
            }
            async for event in self.engine.stream_query(rewritten_sink.query, params):
                yield SearchEngineStreamEvent(
                    query=query,
                    retrieval=event.retrieval,
                    delta=event.delta,
                    payload=payload | event.payload,
                )
            return

    @override
    async def search(self, query: str, params: EngineParams | None = None) -> SearchEngineRetrieve:
        """
        Delegate search directly to the underlying engine.

        :param query: The search query.
        :param params: Retrieval parameters passed to the underlying engine.
        :return: Retrieval container returned by the underlying engine.
        """
        return await self.engine.search(query, params)

    @override
    async def batch_search(
        self,
        queries: List[str],
        params: EngineParams | None = None,
    ) -> List[SearchEngineRetrieve]:
        """
        Delegate batched search directly to the underlying engine.

        Query planning does not apply to retrieval-only calls, so this forwards
        the whole batch to the child engine's ``batch_search``.

        :param queries: Input query strings.
        :param params: Retrieval parameters passed to the underlying engine.
        :return: Retrieval containers aligned with ``queries``.
        """
        return await self.engine.batch_search(queries, params)
