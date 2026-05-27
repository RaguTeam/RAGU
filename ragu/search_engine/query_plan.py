from typing import List, Dict, Tuple
from typing_extensions import override

from ragu.common.logger import logger
from ragu.common.prompts.default_models import SubQuery, QueryPlan, RewriteQuery
from ragu.search_engine.base_engine import BaseEngine, SearchEngineResponse, SearchEngineRetrieve
from ragu.search_engine.search_functional import _topological_sort

from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


class QueryPlanEngine(BaseEngine):
    """
    Query planning engine that decomposes complex queries into a DAG of subqueries
    and executes them in topological order.

    Pipeline:
      1. Decompose query -> list[SubQuery] (DAG)
      2. Topological sort
      3. For each subquery:
         - rewrite using dependency answers (if needed)
         - execute with underlying engine
         - store answer in context
      4. Return answer of the last subquery
    """

    def __init__(self, engine: BaseEngine, *args, **kwargs):
        """
        Initialize a query planner around an existing search engine.

        :param engine: Engine used to answer each planned subquery.
        """
        _PROMPTS_NAMES = ["query_decomposition", "query_rewrite"]
        super().__init__(llm=engine.llm, prompts=_PROMPTS_NAMES, *args, **kwargs)
        self.engine: BaseEngine = engine

    async def process_query(self, query: str) -> List[SubQuery]:
        """
        Decompose a complex query into atomic subqueries with dependencies.

        Uses an LLM to analyze the input query and break it down into minimal,
        independent subqueries. Each subquery is assigned a unique ID and may
        declare dependencies on other subqueries that must be resolved first.

        :param query: Complex natural-language query to decompose.
        :return: List of SubQuery objects forming a DAG.
        """
        instruction: RAGUInstruction = self.get_prompt("query_decomposition")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=query,
        )
        rendered = rendered_list[0]

        response: QueryPlan = await self.engine.llm.chat_completion(    # type: ignore
            rendered.to_openai(),
            output_schema=instruction.pydantic_model,
        )
        return response.subqueries

    async def _rewrite_subquery(self, subquery: SubQuery, context: Dict[str, SearchEngineResponse]) -> SubQuery:
        """
        Rewrite a subquery by injecting answers from its dependency subqueries.

        Only dependency answers listed in `subquery.depends_on` are provided
        to the rewrite prompt.

        :param subquery: The subquery to rewrite.
        :param context: Mapping of subquery ID to prior ``SearchEngineResponse``.
        :return: Copy of ``subquery`` with a rewritten, self-contained query.
        """
        dep_context = {k: v for k, v in context.items() if k in subquery.depends_on}

        instruction: RAGUInstruction = self.get_prompt("query_rewrite")
        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            original_query=subquery.query,
            context=dep_context,
        )
        rendered = rendered_list[0]

        response: List[RewriteQuery | str] = await self.engine.llm.chat_completion(
            rendered.to_openai(),
            output_schema=instruction.pydantic_model,
        )
        rewritten = response.query if isinstance(response, RewriteQuery) else response

        return subquery.model_copy(update={"query": rewritten})

    async def _answer_subquery(
            self,
            subquery: SubQuery,
            context: Dict[str, SearchEngineResponse]
    ) -> Tuple[SubQuery, SearchEngineResponse]:
        """
        Execute a single subquery, rewriting it first if it has dependencies.

        :param subquery: The subquery to execute.
        :param context: Mapping of {subquery_id -> answer} for dependency injection.
        :return: Tuple containing the possibly rewritten subquery and its full response.
        """
        if subquery.depends_on:
            subquery = await self._rewrite_subquery(subquery, context)

        result = await self.engine.a_query(subquery.query)

        return subquery, result

    @override
    async def a_query(self, query: str, *args, **kwargs) -> SearchEngineResponse:
        """
        Execute a complex query using the plan-and-execute pipeline.

        This method:
        1. Decompose the query into subqueries with dependencies.
        2. Sort subqueries in topological order.
        3. Rewrite subquery based on previous context and answer the query.
        4. Return the final answer from the last subquery

        Dependent subqueries are automatically rewritten to be self-contained
        by injecting answers from their prerequisite subqueries.

        :param query: The complex natural-language query to answer.
        :return: ``SearchEngineResponse`` whose ``response`` is the final
                 subquery answer, ``retrieval`` is the final subquery retrieval,
                 and ``payload`` contains all subquery responses by ID.
        """
        subqueries = await self.process_query(query)

        ordered = _topological_sort(subqueries)

        context: Dict[str,  SearchEngineResponse] = {}
        retrieve: Dict[str, SearchEngineRetrieve] = {}
        for subquery in ordered:
            rewritten_subquery, response = await self._answer_subquery(subquery, context)
            context[subquery.id] = response
            retrieve[subquery.id] = response.retrieval

        if not context:
            return SearchEngineResponse(
                query=query, response="", retrieval=None, payload={}  # type: ignore[arg-type]
            )

        return SearchEngineResponse(
            query=query,
            response=context[ordered[-1].id].response,
            retrieval=retrieve.get(ordered[-1].id),  # type: ignore[arg-type]
            payload=context,
        )

    # TODO: maybe it is good idea to refactor this method so it will returns list of 'subcontexts' for every subquery.
    @override
    async def a_search(self, query, *args, **kwargs):
        """
        Delegate search directly to the underlying engine.

        :param query: The search query.
        :param args: Additional positional arguments passed to the underlying engine.
        :param kwargs: Additional keyword arguments passed to the underlying engine.
        :return: Retrieval container returned by the underlying engine.
        """
        return await self.engine.a_search(query, *args, **kwargs)
