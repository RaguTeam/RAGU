from textwrap import dedent
from typing import List
from jinja2 import Template

from ragu.common.base import RaguGenerativeModule
from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.common.prompts.default_models import CommunityReportModel
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render
from ragu.graph.types import Community, CommunitySummary
from ragu.models.llm import LLM


class CommunitySummarizer(RaguGenerativeModule):
    """
    Generates textual summaries for detected graph communities using an LLM.

    The summarization process converts a group of entities or
    relations belonging to the same community into a human-readable report.

    :param client: LLM client used for generating community reports.
    :param language: Language of generated summaries. Defaults to ``Settings.language``.
    """

    def __init__(self, llm: LLM, language: str | None = None) -> None:
        """
        Initialize community summarizer.

        :param client: LLM client used for summarization.
        :param language: Optional language override.
        """
        _PROMPTS = ["community_report"]
        super().__init__(prompts=_PROMPTS)

        self.llm = llm
        self.language = language if language else Settings.language

    async def summarize(self, communities: List[Community]) -> List[CommunitySummary]:
        """
        Generate structured summaries for a list of graph communities.

        :param communities: Communities to summarize.
        :return: Community summaries aligned with input communities.
        """
        sorted_communities: list[Community] = []
        for community in communities:
            sorted_communities.append(
                Community(
                    entities=sorted(community.entities, key=lambda e: e.id),
                    relations=sorted(community.relations, key=lambda e: e.id),
                    level=community.level,
                    cluster_id=community.cluster_id,
                )
            )
        instruction: RAGUInstruction = self.get_prompt("community_report")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            community=sorted_communities,
            language=self.language,
        )

        output_schema = instruction.pydantic_model
        assert output_schema is CommunityReportModel
        summaries: List[CommunityReportModel | None] = await self.llm.batch_chat_completion(
            [c.to_openai() for c in rendered_list],
            output_schema=output_schema,
            continue_on_error=True,
            desc="Summarized communities",
        )

        output: List[CommunitySummary] = []
        for community, summary in zip(sorted_communities, summaries):
            if summary is None or getattr(summary, 'title', None) is None:
                logger.warning("Skipping community {}: summarization returned empty result", community.id)
                continue
            output.append(CommunitySummary(
                id=community.id,
                summary=self.combine_report_text(summary),
            ))

        return output

    @staticmethod
    def combine_report_text(report: CommunityReportModel) -> str:
        """
        Merge structured sections of a community report into a readable text block.

        :param report: Structured community report.
        :return: Rendered report text.
        """
        if not report:
            return ""

        return _COMMUNITY_REPORT_TEMPLATE.render(report=report)


_COMMUNITY_REPORT_TEMPLATE = Template(dedent(
    """
    Report title: {{ report.title }}
    Report summary: {{ report.summary }}
    
    {% for finding in report.findings %}
    Finding summary: {{ finding.summary }}
    Finding explanation: {{ finding.explanation }}
    {% endfor %}
    """)
)
