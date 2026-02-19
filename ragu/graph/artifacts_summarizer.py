from dataclasses import asdict
from itertools import chain
from typing import List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from ragu.common.global_parameters import Settings
from ragu.common.base import RaguGenerativeModule
from ragu.common.logger import logger
from ragu.common.prompts.default_models import RelationDescriptionModel, EntityDescriptionModel
from ragu.embedder.base_embedder import BaseEmbedder
from ragu.graph.types import Entity, Relation
from ragu.llm.base_llm import BaseLLM

from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


class EntitySummarizer(RaguGenerativeModule):
    """
    Summarizes and merges textual descriptions of duplicate entities.

    Entities are grouped by ``(entity_name, entity_type)``, merged, and optionally
    summarized with an LLM. When enabled, large description sets can be clustered
    before summarization to reduce prompt size.
    """

    def __init__(
        self,
        client: Optional[BaseLLM] = None,
        use_llm_summarization: bool = True,
        use_clustering: bool = False,
        embedder: Optional[BaseEmbedder] = None,
        cluster_only_if_more_than: int = 128,
        summarize_only_if_more_than: int = 5,
        language: Optional[str] = None,
    ):
        _PROMPTS = ["entity_summarizer", "cluster_summarize"]
        super().__init__(prompts=_PROMPTS)

        self.client = client
        self.language = language if language else Settings.language
        self.use_llm_summarization = use_llm_summarization
        self.summarize_only_if_more_than = summarize_only_if_more_than

        # Clustering parameters
        self.use_clustering = use_clustering
        self.cluster_only_if_more_than = cluster_only_if_more_than
        self.embedder = embedder

        if self.use_llm_summarization and self.client is None:
            raise ValueError(
                "LLM summarization is enabled but no client is provided. Please provide a client."
            )

        if self.use_clustering and not self.use_llm_summarization:
            logger.warning(
                "Clustering is enabled but LLM summarization is disabled. Clustering will be ignored."
            )
            self.use_clustering = False

        if self.use_clustering and not self.embedder:
            raise ValueError(
                "Clustering is enabled but no embedder is provided. Please provide an embedder."
            )

    async def run(self, entities: List[Entity]) -> List[Entity]:
        """
        Execute the full artifact summarization pipeline.

        Steps:
          1. Group duplicated entities by (entity_name, entity_type),
          2. Optionally cluster large description sets and summarize cluster-wise,
          3. Optionally summarize entities with many duplicates via LLM,
          4. Return updated list of Entity objects.

        :param entities: List of extracted entities.
        :return: Summarized/deduplicated entities list.
        """
        if len(entities) == 0:
            logger.warning("Empty list of entities. Seems that something goes wrong.")
            return []

        grouped_entities_df = self.group_entities(entities)

        num_of_duplicated_entities = len(entities) - len(grouped_entities_df)
        logger.info(
            f"Found {num_of_duplicated_entities} duplicated entities. "
            f"Number of unique entities: {len(grouped_entities_df)} "
        )

        entities_to_return = await self.summarize_entities(grouped_entities_df)

        if len(entities_to_return) != len(grouped_entities_df):
            logger.warning(
                f"{len(entities_to_return) - len(grouped_entities_df)} from {len(grouped_entities_df)} entities "
                f"were missed during summarization."
            )

        return entities_to_return

    async def summarize_entities(self, grouped_entities_df: pd.DataFrame) -> List[Entity]:
        """
        Summarize merged entity descriptions.

        Entities with identical ``entity_name`` and ``entity_type`` are grouped
        into a single record. .

        :param grouped_entities_df: DataFrame containing grouped entity data with
                                    a ``duplicate_count`` column.
        :return: A list of summarized :class:`Entity` objects.
        """
        # Convert from list to string and maybe summarize by clusters
        for i, row in grouped_entities_df.iterrows():
            maybe_clustered = await self._summarize_by_cluster_if_needed(row["description"])
            grouped_entities_df.loc[i, "description"] = maybe_clustered

        entity_mask = grouped_entities_df["duplicate_count"].to_numpy() > self.summarize_only_if_more_than

        entity_multi_desc = grouped_entities_df.loc[entity_mask]
        entity_single_desc = grouped_entities_df.loc[~entity_mask]

        entity_multi_desc = entity_multi_desc.drop("duplicate_count", axis=1)
        entity_single_desc = entity_single_desc.drop("duplicate_count", axis=1)

        if entity_multi_desc.empty:
            return [Entity(**row) for _, row in entity_single_desc.iterrows()]

        entities_to_summarize: List[Entity] = []
        if self.use_llm_summarization:
            entities_to_summarize = [Entity(**row) for _, row in entity_multi_desc.iterrows()]

            instruction: RAGUInstruction = self.get_prompt("entity_summarizer")
            rendered_list: List[ChatMessages] = render(
                instruction.messages,
                entity=entities_to_summarize,
                language=self.language,
            )

            response: List[EntityDescriptionModel] = await self.client.generate(  # type: ignore
                conversations=rendered_list,
                response_model=instruction.pydantic_model,
                progress_bar_desc="Entity summarization",
            )

            for i, summary in enumerate(response):
                if summary:
                    entities_to_summarize[i].description = summary.description
        else:
            entities_to_summarize = [Entity(**row) for _, row in entity_multi_desc.iterrows()]

        return [Entity(**row) for _, row in entity_single_desc.iterrows()] + entities_to_summarize

    @staticmethod
    def group_entities(entities: List[Entity]) -> pd.DataFrame:
        """
        Group entities by ``entity_name`` and ``entity_type`` and aggregate their
        fields into combined records.

        :param entities: List of :class:`Entity` objects to group.
        :return: Aggregated entities as a :class:`pandas.DataFrame`.
        """
        entities_df = pd.DataFrame([asdict(entity) for entity in entities])
        grouped_entities = entities_df.groupby(["entity_name", "entity_type"]).agg(
            description=("description", list),
            duplicate_count=("description", "count"),
            source_chunk_id=("source_chunk_id", lambda s: list(set(
                chain.from_iterable(v if isinstance(v, (list, tuple, set)) else [v] for v in s)))
            ),
            documents_id=("documents_id", lambda s: list(set(
                chain.from_iterable(v if isinstance(v, (list, tuple, set)) else [v] for v in s)))
            ),
        ).reset_index()

        return grouped_entities

    async def _summarize_by_cluster_if_needed(self, descriptions: List[str]) -> str:
        """
        Optionally cluster a large set of descriptions and summarize each cluster via LLM.

        If clustering is disabled or there are not enough descriptions, returns the
        concatenation of descriptions.

        :param descriptions: List of raw descriptions for one entity.
        :return: A single merged (and optionally cluster-summarized) description string.
        """
        if len(descriptions) > self.cluster_only_if_more_than and self.use_clustering:
            assert self.embedder
            cluster = DBSCAN(eps=0.5, min_samples=2).fit(
                np.array(await self.embedder.embed(descriptions))
            )
            labels = cluster.labels_

            clusters: dict[int, list[str]] = {}
            for label, text in zip(labels, descriptions):
                clusters.setdefault(int(label), []).append(text)

            result_description: List[str] = []
            for texts in clusters.values():
                instruction: RAGUInstruction = self.get_prompt("cluster_summarize")
                rendered_list: List[ChatMessages] = render(
                    instruction.messages,
                    content=texts,
                    language=self.language,
                )

                result = await self.client.generate(  # type: ignore
                    conversations=rendered_list,
                    response_model=instruction.pydantic_model,
                    progress_bar_desc="Map reduce for clustering",
                )
                result_description.extend([r.content for r in result if r])

            return ". ".join(result_description)

        return ". ".join(descriptions)


class RelationSummarizer(RaguGenerativeModule):
    """
    Summarizes and merges textual descriptions of entities and relations.
    extracted from documents.

    The class groups duplicated entities and relations by their identifiers
    (e.g., ``entity_name``, ``entity_type`` for entities, and
    ``subject_id``, ``object_id`` for relations), merges their attributes,
    and optionally generates concise descriptions through an LLM.

    :param client: LLM client used for summarization. Required if
                   ``use_llm_summarization=True``.
    :param use_llm_summarization: Whether to perform description summarization
                                  with a language model.
    :param language: Target language for summarization (e.g., ``"russian"`` or ``"english"``).
    :raises ValueError: If ``use_llm_summarization=True`` but no client is provided.
    """

    def __init__(
        self,
        client: BaseLLM = None,
        use_llm_summarization: bool = True,
        summarize_only_if_more_than: int = 5,
        language: str = "russian",
    ):
        _PROMPTS = ["relation_summarizer"]
        super().__init__(prompts=_PROMPTS)

        self.client = client
        self.language = language
        self.use_llm_summarization = use_llm_summarization
        self.summarize_only_if_more_than = summarize_only_if_more_than

        if self.use_llm_summarization and self.client is None:
            raise ValueError(
                "LLM summarization is enabled but no client is provided. Please provide a client."
            )

    async def run(self, relations: List[Relation], **kwargs) -> List[Relation]:
        """
        Execute the full artifact summarization pipeline.

        The pipeline performs the following steps:

        1. Group duplicated relations into aggregated dataframes.
        2. Summarize merged entity and relation descriptions if enabled.
        3. Return the updated lists of :class:`Entity` and :class:`Relation` objects.

        :param relations: List of extracted relations to summarize or merge.
        :return: A tuple ``(entities, relations)`` containing updated objects.
        """
        if len(relations) == 0:
            logger.warning("No relations to summarize. Maybe something goes wrong.")
            return []

        grouped_relations_df = self.group_relations(relations)

        num_of_duplicated_relations = len(relations) - len(grouped_relations_df)
        logger.info(
            f"Found {num_of_duplicated_relations} duplicated relations. "
            f"Number of unique relations: {len(grouped_relations_df)} "
        )

        relations_to_return = await self.summarize_relations(grouped_relations_df)

        if len(relations_to_return) != len(grouped_relations_df):
            logger.warning(
                f"{len(relations_to_return) - len(grouped_relations_df)} from {len(grouped_relations_df)} relations "
                f"were missed during summarization."
            )

        return relations_to_return

    async def summarize_relations(self, grouped_relations_df: pd.DataFrame) -> List[Relation]:
        """
        Summarize merged relation descriptions.

        Relations with identical pairs ``(subject_id, object_id)`` are combined
        into a single entry. If duplicates exist and LLM summarization is enabled,
        their descriptions are merged using the ``relation_summarizer`` prompt.

        :param grouped_relations_df: DataFrame containing grouped relation data with
                                     a ``duplicate_count`` column.
        :return: A list of summarized :class:`Relation` objects.
        """
        relation_mask = grouped_relations_df["duplicate_count"].to_numpy() > self.summarize_only_if_more_than

        relation_multi_desc = grouped_relations_df.loc[relation_mask]
        relation_single_desc = grouped_relations_df.loc[~relation_mask]

        relation_multi_desc = relation_multi_desc.drop("duplicate_count", axis=1)
        relation_single_desc = relation_single_desc.drop("duplicate_count", axis=1)

        if relation_multi_desc.empty:
            return [Relation(**row) for _, row in relation_single_desc.iterrows()]

        relations_to_summarize: List[Relation] = []
        if self.use_llm_summarization:
            relations_to_summarize = [Relation(**row) for _, row in relation_multi_desc.iterrows()]

            instruction: RAGUInstruction = self.get_prompt("relation_summarizer")
            rendered_list: List[ChatMessages] = render(
                instruction.messages,
                relation=relations_to_summarize,
                language=self.language,
            )

            response: List[RelationDescriptionModel] = await self.client.generate(  # type: ignore
                conversations=rendered_list,
                response_model=instruction.pydantic_model,
            )

            for i, summary in enumerate(response):
                if summary:
                    relations_to_summarize[i].description = summary.description
        else:
            relations_to_summarize = [Relation(**row) for _, row in relation_multi_desc.iterrows()]

        return [Relation(**row) for _, row in relation_single_desc.iterrows()] + relations_to_summarize

    @staticmethod
    def group_relations(relations: List[Relation]) -> pd.DataFrame:
        """
        Group relations by (subject_id, object_id) and merge their fields.

        :param relations: List of Relation objects.
        :return: Aggregated relations as a pandas DataFrame.
        """
        relations_df = pd.DataFrame([asdict(relation) for relation in relations])
        grouped_relations = relations_df.groupby(["subject_id", "object_id", "relation_type"]).agg(
            subject_name=("subject_name", "first"),
            object_name=("object_name", "first"),
            description=("description", lambda x: "\n".join(x.dropna().drop_duplicates().astype(str))),
            relation_strength=("relation_strength", "mean"),
            source_chunk_id=("source_chunk_id", lambda s: list(set(
                chain.from_iterable(v if isinstance(v, (list, tuple, set)) else [v] for v in s)))
            ),
            duplicate_count=("description", "count"),
        ).reset_index()

        return grouped_relations
