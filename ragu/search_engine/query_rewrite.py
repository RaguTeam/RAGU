from ragu.common.base import RaguGenerativeModule
from ragu.common.global_parameters import Settings
from ragu.common.prompts.default_models import RewriteQuery
from ragu.common.prompts.messages import ChatMessages, render
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.models.llm import LLM


class DialogueQueryRewriter(RaguGenerativeModule):
    """
    Resolves pronouns and other context-dependent expressions in a user query
    using preceding dialogue turns, producing a self-contained query.

    Standalone component: it performs no retrieval and is not wired into any
    search engine or pipeline. Queries with no dialogue history are returned
    unchanged without an LLM call, since there is nothing to resolve.

    :param llm: LLM client used for rewriting.
    :param language: Language of the rewritten query. Defaults to ``Settings.language``.
    """

    def __init__(self, llm: LLM, language: str | None = None) -> None:
        """
        Initialize the dialogue-aware query rewriter.

        :param llm: LLM client used for rewriting.
        :param language: Optional language override.
        """
        super().__init__(prompts=["dialogue_query_rewrite"])

        self.llm = llm
        self.language = language if language else Settings.language

    async def rewrite(self, query: str | ChatMessages) -> str:
        """
        Rewrite a single query into a self-contained form.

        Single-query wrapper around :meth:`batch_rewrite`.

        :param query: A plain query string, or a dialogue (:class:`ChatMessages`)
            ending in the current user query, with the preceding turns as history.
        :return: Self-contained rewritten query, or the original query unchanged
            when no dialogue history is present.
        """
        return (await self.batch_rewrite([query]))[0]

    async def batch_rewrite(self, queries: list[str | ChatMessages]) -> list[str]:
        """
        Rewrite a batch of queries into self-contained form in one LLM call.

        Only queries carrying dialogue history are sent to the LLM; plain
        string queries (no history) are returned unchanged.

        :param queries: Plain query strings and/or dialogues (:class:`ChatMessages`),
            each ending in the current user query, with the preceding turns as
            history.
        :return: Rewritten queries aligned with ``queries``.
        :raises ValueError: If a :class:`ChatMessages` dialogue is empty or does
            not end with a user message.
        """
        if not queries:
            return []

        current_queries: list[str] = []
        histories: list[str] = []
        for dialogue in queries:
            if isinstance(dialogue, ChatMessages):
                if not dialogue.messages:
                    raise ValueError("ChatMessages dialogue must contain at least one message")
                if dialogue.messages[-1].role != "user":
                    raise ValueError("ChatMessages dialogue must end with a user message")
                current_queries.append(dialogue.messages[-1].content)
                histories.append(ChatMessages.from_messages(dialogue.messages[:-1]).to_str())
            else:
                current_queries.append(dialogue)
                histories.append("")

        rewritten = list(current_queries)
        indexed = [
            (i, query, history)
            for i, (query, history) in enumerate(zip(current_queries, histories))
            if history
        ]
        if indexed:
            results = await self._rewrite_with_history(
                [query for _, query, _ in indexed],
                [history for _, _, history in indexed],
            )
            for (i, _, _), rewritten_query in zip(indexed, results):
                rewritten[i] = rewritten_query

        return rewritten

    async def _rewrite_with_history(self, queries: list[str], histories: list[str]) -> list[str]:
        """
        Rewrite a batch of queries that carry dialogue history, in one LLM call.

        :param queries: Current user queries.
        :param histories: Rendered dialogue history, aligned with ``queries``.
        :return: Rewritten queries aligned with ``queries``.
        """
        instruction: RAGUInstruction = self.get_prompt("dialogue_query_rewrite")

        rendered_list: list[ChatMessages] = render(
            instruction.messages,
            dialogue_history=histories,
            query=queries,
            language=self.language,
        )
        responses = await self.llm.batch_chat_completion(
            [rendered.to_openai() for rendered in rendered_list],
            output_schema=instruction.pydantic_model,  # type: ignore
            desc="Dialogue query rewrite",
        )
        return [
            response.query if isinstance(response, RewriteQuery) else response
            for response in responses
        ]
