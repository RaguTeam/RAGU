from __future__ import annotations

from typing import Any, List, Tuple, Optional, cast

from ragu.common.logger import logger
from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.default_models import ArtifactsModel
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render_with_few_shots
from ragu.common.prompts.icl_config import ICLConfig
from ragu.common.prompts.icl_manager import InContextLearningManager
from ragu.graph.types import Entity, Relation
from ragu.models.llm import LLM
from ragu.models.embedder import Embedder
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.triplet.types import NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES


class ArtifactsExtractorLLM(BaseArtifactExtractor):
    """
    Extracts entities and relations from text chunks using LLM.

    Pipeline:
      1. Render the `artifact_extraction` instruction in batch mode over chunk texts.
      2. Call the LLM to produce structured artifacts for each chunk.
      3. Optionally render and run `artifact_validation` to refine extracted artifacts.
      4. Convert model outputs into Entity/Relation objects, preserving source chunk ids.

    Supports in-context learning via InContextLearningManager when provided.
    """

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder | None = None,
        icl_config: ICLConfig | None = None,
        do_validation: bool = False,
        language: str | None = None,
        entity_types: Optional[List[str]] = NEREL_ENTITY_TYPES,
        relation_types: Optional[List[str]] = NEREL_RELATION_TYPES,
    ):
        """
        Initialize a new :class:`ArtifactsExtractorLLM`.

        :param llm: Language model for generation and validation.
        :param embedder: Embedder for computing example embeddings (optional).
        :param icl_config: ICL configuration (optional).
        :param do_validation: Whether to perform additional LLM-based validation of artifacts.
        :param language: Output text language.
        :param entity_types: List of entity types to guide extraction prompts.
        :param relation_types: List of relation types to guide extraction prompts.
        """
        _PROMPTS = ["artifact_extraction", "artifact_validation"]
        super().__init__(prompts=_PROMPTS)

        self.llm = llm
        self.embedder = embedder
        self.do_validation = do_validation
        self.language = language if language else Settings.language
        self.entity_types = ", ".join(entity_types) if entity_types else None
        self.relation_types = ", ".join(relation_types) if relation_types else None

        # Initialize ICL manager if config provided
        self.icl_manager: InContextLearningManager | None = None
        if icl_config and icl_config.enabled and embedder:
            self.icl_manager = InContextLearningManager(
                embedder=embedder,
                example_file=f"{icl_config.examples_base_path}/artifact_extraction_examples.json",
                config=icl_config,
                language=icl_config.language
            )

    async def _extract_artifacts(
        self,
        context: List[str],
    ) -> List[ArtifactsModel]:
        """
        Run artifact extraction for a batch of texts.

        :param context: Chunk texts.
        :return: Per-chunk extracted artifacts.
        """
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_manager:
            await self.icl_manager.initialize()
            for chunk_text in context:
                examples = await self.icl_manager.select_examples(
                    query_text=chunk_text,
                    num_examples=self.icl_manager.config.num_examples
                )
                examples_list.append(examples)
        else:
            examples_list = [None] * len(context)

        instruction: RAGUInstruction = self.get_prompt("artifact_extraction")
        assert instruction.pydantic_model is ArtifactsModel
        conversations: List[ChatMessages] = render_with_few_shots(
            instruction.messages,
            examples_list=examples_list,
            few_shot_formatter=instruction.few_shot_formatter,
            context=context,
            language=self.language,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
        )

        result_list = await self.llm.batch_chat_completion(
            [c.to_openai() for c in conversations],
            output_schema=instruction.pydantic_model or str,
            desc="Extracting a knowledge graph from chunks",
        )
        result_list = cast(list[ArtifactsModel], result_list)

        for artifacts in result_list:
            logger.debug(
                f'Got {len(artifacts.entities)} entities'
                f' and {len(artifacts.relations)} relations for chunk'
            )

        return result_list

    async def _validate_artifacts(
        self,
        context: List[str],
        artifacts: List[ArtifactsModel],
    ) -> List[ArtifactsModel]:
        """
        Run artifact validation for a batch of texts and their extracted artifacts.

        :param context: Chunk texts.
        :param artifacts: Per-chunk extracted artifacts from extraction stage.
        :return: Per-chunk validated artifacts.
        """
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_manager:
            for chunk_text in context:
                examples = await self.icl_manager.select_examples(
                    query_text=chunk_text,
                    num_examples=self.icl_manager.config.num_examples
                )
                examples_list.append(examples)
        else:
            examples_list = [None] * len(context)

        instruction: RAGUInstruction = self.get_prompt("artifact_validation")
        assert instruction.pydantic_model is ArtifactsModel

        conversations: List[ChatMessages] = render_with_few_shots(
            instruction.messages,
            examples_list=examples_list,
            few_shot_formatter=instruction.few_shot_formatter,
            artifacts=artifacts,
            context=context,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            language=self.language,
        )

        result_list = await self.llm.batch_chat_completion(
            [c.to_openai() for c in conversations],
            output_schema=instruction.pydantic_model or str,
            desc="Validation of extracted artifacts",
        )
        result_list = cast(list[ArtifactsModel], result_list)

        for artifacts_validated in result_list:
            logger.debug(
                f'After validation got {len(artifacts_validated.entities)} entities'
                f' and {len(artifacts_validated.relations)} relations for chunk'
            )

        return result_list

    async def extract(
        self,
        chunks: List[Chunk],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from a collection of chunks.

        Steps:
          1) Batch-render the extraction prompt with ``context=<chunk_texts>``,
          2) Generate structured artifacts per chunk,
          3) Optionally validate artifacts against the original context,
          4) Convert artifacts into Entity/Relation objects.

        :param chunks: Iterable of Chunk objects.
        :return: (entities, relations) extracted from all chunks.
        """
        if not chunks:
            return [], []

        context: List[str] = [chunk.content for chunk in chunks]

        try:
            result_list = await self._extract_artifacts(context)
        except Exception as e:
            logger.warning(
                "Artifact extraction failed for {} chunks: {}: {}",
                len(context), type(e).__name__, e,
            )
            return [], []

        if self.do_validation:
            try:
                result_list = await self._validate_artifacts(context, result_list)
            except Exception as e:
                logger.warning(
                    "Artifact validation failed: {}: {}. Using unvalidated results.",
                    type(e).__name__, e,
                )

        entities_result: List[Entity] = []
        relations_result: List[Relation] = []

        for artifacts, chunk in zip(result_list, chunks):

            current_chunk_entities: List[Entity] = []

            for entity_model in artifacts.entities:
                entity = Entity(
                    entity_name=entity_model.entity_name,
                    entity_type=entity_model.entity_type or "UNKNOWN",
                    description=entity_model.description,
                    source_chunk_id=[chunk.id],
                    documents_id=[],
                    clusters=[],
                )
                current_chunk_entities.append(entity)

            entities_result.extend(current_chunk_entities)

            for relation in artifacts.relations:
                subject_name = relation.source_entity
                object_name = relation.target_entity
                if not (subject_name and object_name):
                    continue
                subject_entity = next(
                    (e for e in current_chunk_entities if e.entity_name == subject_name),
                    None,
                )
                object_entity = next(
                    (e for e in current_chunk_entities if e.entity_name == object_name),
                    None,
                )

                if subject_entity and object_entity:
                    relation = Relation(
                        subject_name=subject_name,
                        object_name=object_name,
                        subject_id=subject_entity.id,
                        object_id=object_entity.id,
                        relation_type=relation.relation_type or "UNKNOWN",
                        description=relation.description,
                        relation_strength=float(relation.relationship_strength),
                        source_chunk_id=[chunk.id],
                    )
                    relations_result.append(relation)

        return entities_result, relations_result
