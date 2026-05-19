from __future__ import annotations

from typing import Any, List, Optional, Tuple, cast
from typing_extensions import override

from pydantic import BaseModel

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.common.prompts.default_models import (
    EntitiesExtractionModel,
    RelationsExtractionModel,
)
from ragu.common.prompts.messages import ChatMessages, render_with_few_shots
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.icl_config import ICLConfig
from ragu.common.prompts.icl_manager import InContextLearningManager, resolve_example_path
from ragu.graph.types import Entity, Relation
from ragu.models.llm import LLM
from ragu.models.embedder import Embedder
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.triplet.prompts import (
    TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION,
    TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION,
    TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION,
    TWO_STAGE_RELATION_VALIDATION_INSTRUCTION,
)
from ragu.triplet.types import NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES


class TwoStageArtifactsExtractorLLM(BaseArtifactExtractor):
    """
    Two-stage LLM artifact extractor.

    Pipeline:
      1. Extract entities from each chunk.
      2. Optionally validate entities against source chunk text.
      3. Extract relations constrained by validated entities.
      4. Optionally validate relations against source chunk text and entity set.
      5. Convert stage outputs to graph `Entity` and `Relation` objects.

    Supports in-context learning via InContextLearningManager when provided.
    """

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder | None = None,
        icl_config: ICLConfig | None = None,
        do_entity_validation: bool | None = None,
        do_relation_validation: bool | None = None,
        language: str | None = None,
        entity_types: Optional[List[str]] = NEREL_ENTITY_TYPES,
        relation_types: Optional[List[str]] = NEREL_RELATION_TYPES,
    ) -> None:
        """
        Initialize two-stage extractor.

        :param llm: LLM backend used for extraction and validation calls.
        :param embedder: Embedder for computing example embeddings (optional).
        :param icl_config: ICL configuration (optional).
        :param do_entity_validation: If set, overrides entity validation toggle.
        :param do_relation_validation: If set, overrides relation validation toggle.
        :param language: Language hint injected into prompts.
        :param entity_types: Optional allowed entity types for prompts.
        :param relation_types: Optional allowed relation types for prompts.
        """
        prompts = {
            "entity_extraction": TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION,
            "entity_validation": TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION,
            "relation_extraction": TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION,
            "relation_validation": TWO_STAGE_RELATION_VALIDATION_INSTRUCTION,
        }
        super().__init__(prompts=prompts)

        self.llm = llm
        self.embedder = embedder
        self.language = language if language else Settings.language
        self.entity_types = ", ".join(entity_types) if entity_types else None
        self.relation_types = ", ".join(relation_types) if relation_types else None

        self.do_entity_validation = do_entity_validation
        self.do_relation_validation = do_relation_validation

        # Initialize separate ICL managers for each stage
        self.icl_entity: InContextLearningManager | None = None
        self.icl_relation: InContextLearningManager | None = None
        if icl_config and icl_config.enabled and embedder:
            self.icl_entity = InContextLearningManager(
                embedder=embedder,
                example_file=resolve_example_path(
                    icl_config.examples_base_path,
                    "entity_extraction_examples.json",
                ),
                config=icl_config,
                language=icl_config.language
            )
            self.icl_relation = InContextLearningManager(
                embedder=embedder,
                example_file=resolve_example_path(
                    icl_config.examples_base_path,
                    "relation_extraction_examples.json",
                ),
                config=icl_config,
                language=icl_config.language
            )

    @override
    async def extract(
        self,
        chunks: List[Chunk],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from chunks with an explicit two-stage flow.

        :param chunks: List of input chunks.
        :return: Tuple of extracted entities and relations.
        """
        if not chunks:
            return [], []

        entities_result: List[Entity] = []
        relations_result: List[Relation] = []
        context: List[str] = [chunk.content for chunk in chunks]

        try:
            entity_results = await self._extract_entities(context)
        except Exception as e:
            logger.warning(
                "Entity extraction failed for {} chunks: {}: {}",
                len(context), type(e).__name__, e,
            )
            return [], []

        if self.do_entity_validation:
            try:
                entity_results = await self._validate_entities(context, entity_results)
            except Exception as e:
                logger.warning(
                    "Entity validation failed: {}: {}. Using unvalidated entities.",
                    type(e).__name__, e,
                )

        try:
            relation_results = await self._extract_relations(context, entity_results)
        except Exception as e:
            logger.warning(
                "Relation extraction failed for {} chunks: {}: {}",
                len(context), type(e).__name__, e,
            )
            return entities_result, []

        if self.do_relation_validation:
            try:
                relation_results = await self._validate_relations(
                    context=context,
                    entities=entity_results,
                    relations=relation_results,
                )
            except Exception as e:
                logger.warning(
                    "Relation validation failed: {}: {}. Using unvalidated relations.",
                    type(e).__name__, e,
                )

        for entities_model, relations_model, chunk in zip(entity_results, relation_results, chunks):
            current_chunk_entities: List[Entity] = []

            for entity_model in entities_model.entities:
                entity = Entity(
                    entity_name=entity_model.entity_name,
                    entity_type=entity_model.entity_type,
                    description=entity_model.description,
                    source_chunk_id=[chunk.id],
                    documents_id=[],
                    clusters=[],
                )
                current_chunk_entities.append(entity)

            entities_result.extend(current_chunk_entities)
            entity_by_name = {entity.entity_name: entity for entity in current_chunk_entities}

            for relation_model in relations_model.relations:
                subject_entity = entity_by_name.get(relation_model.source_entity)
                object_entity = entity_by_name.get(relation_model.target_entity)
                if not subject_entity or not object_entity:
                    logger.debug(
                        "Skipping relation with unresolved endpoints: "
                        f"{relation_model.source_entity} -> {relation_model.target_entity}"
                    )
                    continue

                relation = Relation(
                    subject_name=subject_entity.entity_name,
                    object_name=object_entity.entity_name,
                    subject_id=subject_entity.id,
                    object_id=object_entity.id,
                    relation_type=relation_model.relation_type,
                    description=relation_model.description,
                    relation_strength=float(relation_model.relationship_strength),
                    source_chunk_id=[chunk.id],
                )
                relations_result.append(relation)

        return entities_result, relations_result

    async def _extract_entities(self, context: List[str]) -> List[EntitiesExtractionModel]:
        """
        Run stage-1 entity extraction for each chunk.

        :param context: Chunk texts.
        :return: Per-chunk extracted entities.
        """

        # Select ICL examples for entity extraction if manager is initialized
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_entity:
            await self.icl_entity.initialize()
            for chunk_text in context:
                examples = await self.icl_entity.select_examples(
                    query_text=chunk_text,
                    num_examples=self.icl_entity.config.num_examples
                )
                examples_list.append(examples)
        else:
            examples_list = [None] * len(context)

        instruction: RAGUInstruction = self.get_prompt("entity_extraction")
        assert instruction.pydantic_model is EntitiesExtractionModel

        conversations: List[ChatMessages] = render_with_few_shots(
            instruction.messages,
            examples_list=examples_list,
            few_shot_formatter=instruction.few_shot_formatter,
            context=context,
            language=self.language,
            entity_types=self.entity_types,
        )

        results = await self.llm.batch_chat_completion(  # type: ignore
            [conversation.to_openai() for conversation in conversations],
            output_schema=instruction.pydantic_model or str,  # type: ignore
            desc="Extracting entities from chunks",
        )
        typed_results = cast(list[EntitiesExtractionModel], results)

        for entities_model in typed_results:
            logger.debug(f"Got {len(entities_model.entities)} entities")

        return typed_results

    async def _validate_entities(
        self,
        context: List[str],
        entities: List[EntitiesExtractionModel],
    ) -> List[EntitiesExtractionModel]:
        """
        Run stage-1 validation for entity outputs.

        :param context: Chunk texts.
        :param entities: Per-chunk entities from extraction stage.
        :return: Validated entities per chunk.
        """

        # Select ICL examples for entity validation if manager is initialized
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_entity:
            for chunk_text in context:
                examples = await self.icl_entity.select_examples(
                    query_text=chunk_text,
                    num_examples=self.icl_entity.config.num_examples
                )
                examples_list.append(examples)
        else:
            examples_list = [None] * len(context)

        instruction: RAGUInstruction = self.get_prompt("entity_validation")
        assert instruction.pydantic_model is EntitiesExtractionModel

        conversations: List[ChatMessages] = render_with_few_shots(
            instruction.messages,
            examples_list=examples_list,
            few_shot_formatter=instruction.few_shot_formatter,
            context=context,
            entities=self._models_to_payload(entities),
            language=self.language,
            entity_types=self.entity_types,
        )

        results = await self.llm.batch_chat_completion(  # type: ignore
            [conversation.to_openai() for conversation in conversations],
            output_schema=instruction.pydantic_model or str,  # type: ignore
            desc="Validating extracted entities",
        )
        typed_results = cast(list[EntitiesExtractionModel], results)

        for entities_model in typed_results:
            logger.debug(f"After validation got {len(entities_model.entities)} entities")

        return typed_results

    async def _extract_relations(
        self,
        context: List[str],
        entities: List[EntitiesExtractionModel],
    ) -> List[RelationsExtractionModel]:
        """
        Run stage-2 relation extraction constrained by extracted entities.

        :param context: Chunk texts.
        :param entities: Per-chunk validated entities.
        :return: Per-chunk extracted relations.
        """

        # Select ICL examples for relation extraction if manager is initialized
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_relation:
            for chunk_text in context:
                examples = await self.icl_relation.select_examples(
                    query_text=chunk_text,
                    num_examples=self.icl_relation.config.num_examples
                )
                examples_list.append(examples)
        else:
            examples_list = [None] * len(context)

        instruction: RAGUInstruction = self.get_prompt("relation_extraction")
        assert instruction.pydantic_model is RelationsExtractionModel

        conversations: List[ChatMessages] = render_with_few_shots(
            instruction.messages,
            examples_list=examples_list,
            few_shot_formatter=instruction.few_shot_formatter,
            context=context,
            entities=self._models_to_payload(entities),
            language=self.language,
            relation_types=self.relation_types,
        )

        results = await self.llm.batch_chat_completion(  # type: ignore
            [conversation.to_openai() for conversation in conversations],
            output_schema=instruction.pydantic_model or str,  # type: ignore
            desc="Extracting relations from chunks",
        )
        typed_results = cast(list[RelationsExtractionModel], results)

        for relations_model in typed_results:
            logger.debug(f"Got {len(relations_model.relations)} relations")

        return typed_results

    async def _validate_relations(
        self,
        context: List[str],
        entities: List[EntitiesExtractionModel],
        relations: List[RelationsExtractionModel],
    ) -> List[RelationsExtractionModel]:
        """
        Run stage-2 validation for relation outputs.

        :param context: Chunk texts.
        :param entities: Per-chunk entity sets.
        :param relations: Per-chunk relation sets.
        :return: Validated relations per chunk.
        """

        # Select ICL examples for relation validation if manager is initialized
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_relation:
            for chunk_text in context:
                examples = await self.icl_relation.select_examples(
                    query_text=chunk_text,
                    num_examples=self.icl_relation.config.num_examples
                )
                examples_list.append(examples)
        else:
            examples_list = [None] * len(context)

        instruction: RAGUInstruction = self.get_prompt("relation_validation")
        assert instruction.pydantic_model is RelationsExtractionModel

        conversations: List[ChatMessages] = render_with_few_shots(
            instruction.messages,
            examples_list=examples_list,
            few_shot_formatter=instruction.few_shot_formatter,
            context=context,
            entities=self._models_to_payload(entities),
            relations=self._models_to_payload(relations),
            language=self.language,
            relation_types=self.relation_types,
        )

        results = await self.llm.batch_chat_completion(  # type: ignore
            [conversation.to_openai() for conversation in conversations],
            output_schema=instruction.pydantic_model or str,  # type: ignore
            desc="Validating extracted relations",
        )
        typed_results = cast(list[RelationsExtractionModel], results)

        for relations_model in typed_results:
            logger.debug(f"After validation got {len(relations_model.relations)} relations")

        return typed_results

    @staticmethod
    def _models_to_payload(models: List[BaseModel]) -> List[List[dict[str, Any]]]:
        """
        Convert stage models to JSON-like payloads expected by Jinja templates.

        :param models: Batch of pydantic models containing list fields.
        :return: List of list dictionaries per chunk.
        """
        payload: List[List[dict[str, Any]]] = []
        for model in models:
            data = model.model_dump()
            first_value = next(iter(data.values()), [])
            payload.append(cast(List[dict[str, Any]], first_value))
        return payload
