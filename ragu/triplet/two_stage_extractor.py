from typing import Any, List, Sequence, Tuple, TypeVar, cast
from typing_extensions import override

from pydantic import BaseModel

from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.common.prompts.default_models import (
    EntitiesExtractionModel,
    EntityModel,
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
from ragu.triplet.ontology import (
    Ontology,
    OntologyValidator,
    ValidationPolicies,
    resolve_ontology,
)

#: Stage model produced by one extraction/validation step.
ModelT = TypeVar("ModelT", bound=BaseModel)


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
        ontology: Ontology | str | None = "nerel",
        validation: ValidationPolicies = ValidationPolicies(),
        show_type_signatures: bool = False,
        prune_relation_types: bool = False,
    ) -> None:
        """
        Initialize two-stage extractor.

        :param llm: LLM backend used for extraction and validation calls.
        :param embedder: Embedder for computing example embeddings (optional).
        :param icl_config: ICL configuration (optional).
        :param do_entity_validation: If set, overrides entity validation toggle.
        :param do_relation_validation: If set, overrides relation validation toggle.
        :param language: Language hint injected into prompts.
        :param ontology: Vocabulary the extraction is restricted to.
        :param validation: What to do about artifacts that violate the ontology.
            Ignored when ``ontology`` is ``None``.
        :param show_type_signatures: Render each predicate in the prompt with its
            ``[DOMAIN -> RANGE]`` signature.
        :param prune_relation_types: Offer only the predicates admissible between the
            entity types actually found in the chunk. Shortens the prompt and removes
            most of the ways to pick an inapplicable predicate.
        """
        prompts = {
            "entity_extraction": TWO_STAGE_ENTITY_EXTRACTION_INSTRUCTION,
            "entity_validation": TWO_STAGE_ENTITY_VALIDATION_INSTRUCTION,
            "relation_extraction": TWO_STAGE_RELATION_EXTRACTION_INSTRUCTION,
            "relation_validation": TWO_STAGE_RELATION_VALIDATION_INSTRUCTION,
        }
        resolved = resolve_ontology(ontology)
        super().__init__(
            prompts=prompts,
            validator=OntologyValidator(resolved, validation) if resolved else None,
        )

        self.llm = llm
        self.embedder = embedder
        self.language = language if language else Settings.language
        self.ontology = resolved
        self.show_type_signatures = show_type_signatures
        self.prune_relation_types = prune_relation_types
        self.entity_types = resolved.render_entity_types() if resolved else None
        self.relation_types = (
            resolved.render_relation_types(with_signatures=show_type_signatures)
            if resolved
            else None
        )

        self.do_entity_validation = do_entity_validation
        self.do_relation_validation = do_relation_validation

        # Initialize separate ICL managers for each stage
        self.icl_manager: InContextLearningManager | None = None
        if icl_config and icl_config.enabled:
            self.icl_manager = InContextLearningManager(
                example_files={
                    "entity_extraction": resolve_example_path(
                        icl_config.examples_base_path,
                        "entity_extraction_examples.json",
                    ),
                    "entity_validation": resolve_example_path(
                        icl_config.examples_base_path,
                        "entity_validation_examples.json",
                    ),
                    "relation_extraction": resolve_example_path(
                        icl_config.examples_base_path,
                        "relation_extraction_examples.json",
                    ),
                    "relation_validation": resolve_example_path(
                        icl_config.examples_base_path,
                        "relation_validation_examples.json",
                    ),
                },
                config=icl_config,
                embedder=embedder,
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

        entity_lists, _ = self._apply_ontology(
            [model.entities if model else [] for model in entity_results],
            [[] for _ in entity_results],
            stage="entities",
        )
        entity_results = [
            EntitiesExtractionModel(entities=entities) for entities in entity_lists
        ]
        entities_payload = self._models_to_payload(entity_results)

        try:
            relation_results = await self._extract_relations(context, entities_payload)
        except Exception as e:
            logger.warning(
                "Relation extraction failed for {} chunks: {}: {}",
                len(context), type(e).__name__, e,
            )
            for chunk_entities, chunk in zip(entity_lists, chunks):
                entities_result.extend(self._to_entities(chunk_entities, chunk))
            return entities_result, []

        if self.do_relation_validation:
            try:
                relation_results = await self._validate_relations(
                    context=context,
                    entities_payload=entities_payload,
                    relations=relation_results,
                )
            except Exception as e:
                logger.warning(
                    "Relation validation failed: {}: {}. Using unvalidated relations.",
                    type(e).__name__, e,
                )

        entity_lists, relation_lists = self._apply_ontology(
            entity_lists,
            [model.relations if model else [] for model in relation_results],
            stage="relations",
        )

        for chunk_entities, chunk_relations, chunk in zip(entity_lists, relation_lists, chunks):
            current_chunk_entities = self._to_entities(chunk_entities, chunk)
            entities_result.extend(current_chunk_entities)

            entity_by_name = {entity.entity_name: entity for entity in current_chunk_entities}

            for relation_model in chunk_relations:
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

    async def _extract_entities(self, context: List[str]) -> List[EntitiesExtractionModel | None]:
        """
        Run stage-1 entity extraction for each chunk.

        :param context: Chunk texts.
        :return: Per-chunk extracted entities.
        """

        # Select ICL examples for entity extraction if manager is initialized
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_manager:
            await self.icl_manager.initialize()
            examples_list = list(await self.icl_manager.batch_select_examples(
                query_texts=context,
                task="entity_extraction",
                num_examples=self.icl_manager.config.num_examples
            ))
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

        results = await self.llm.batch_chat_completion(
            [conversation.to_openai() for conversation in conversations],
            output_schema=instruction.pydantic_model,
            continue_on_error=True,
            desc="Extracting entities from chunks",
        )
        for i, entities_model in enumerate(results):
            if entities_model is not None:
                logger.debug(f"Got {len(entities_model.entities)} entities")
            else:
                logger.warning("LLM call failed for entity extraction chunk at index {}", i)

        return results

    async def _validate_entities(
        self,
        context: List[str],
        entities: List[EntitiesExtractionModel | None],
    ) -> List[EntitiesExtractionModel | None]:
        """
        Run stage-1 validation for entity outputs.

        :param context: Chunk texts.
        :param entities: Per-chunk entities from extraction stage.
        :return: Validated entities per chunk.
        """

        # Select ICL examples for entity validation if manager is initialized
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_manager:
            examples_list = list(await self.icl_manager.batch_select_examples(
                query_texts=context,
                task="entity_validation",
                num_examples=self.icl_manager.config.num_examples
            ))
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

        results = await self.llm.batch_chat_completion(
            [conversation.to_openai() for conversation in conversations],
            output_schema=instruction.pydantic_model,
            continue_on_error=True,
            desc="Validating extracted entities",
        )
        for i, entities_model in enumerate(results):
            if entities_model is not None:
                logger.debug(f"After validation got {len(entities_model.entities)} entities")
            else:
                logger.warning("LLM call failed for entity validation chunk at index {}", i)

        return results

    async def _extract_relations(
        self,
        context: List[str],
        entities_payload: List[List[dict[str, Any]]],
    ) -> List[RelationsExtractionModel | None]:
        """
        Run stage-2 relation extraction constrained by extracted entities.

        :param context: Chunk texts.
        :param entities_payload: Per-chunk entity payloads for prompt rendering.
        :return: Per-chunk extracted relations.
        """

        # Select ICL examples for relation extraction if manager is initialized
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_manager:
            examples_list = list(await self.icl_manager.batch_select_examples(
                query_texts=context,
                task="relation_extraction",
                num_examples=self.icl_manager.config.num_examples
            ))
        else:
            examples_list = [None] * len(context)

        instruction: RAGUInstruction = self.get_prompt("relation_extraction")
        assert instruction.pydantic_model is RelationsExtractionModel

        conversations: List[ChatMessages] = render_with_few_shots(
            instruction.messages,
            examples_list=examples_list,
            few_shot_formatter=instruction.few_shot_formatter,
            context=context,
            entities=entities_payload,
            language=self.language,
            relation_types=self._relation_types_for(entities_payload),
            type_signatures=self.show_type_signatures,
        )

        results = await self.llm.batch_chat_completion(
            [conversation.to_openai() for conversation in conversations],
            output_schema=instruction.pydantic_model,
            continue_on_error=True,
            desc="Extracting relations from chunks",
        )
        for i, relations_model in enumerate(results):
            if relations_model is not None:
                logger.debug(f"Got {len(relations_model.relations)} relations")
            else:
                logger.warning("LLM call failed for relation extraction chunk at index {}", i)

        return results

    async def _validate_relations(
        self,
        context: List[str],
        entities_payload: List[List[dict[str, Any]]],
        relations: List[RelationsExtractionModel | None],
    ) -> List[RelationsExtractionModel | None]:
        """
        Run stage-2 validation for relation outputs.

        :param context: Chunk texts.
        :param entities_payload: Per-chunk entity payloads for prompt rendering.
        :param relations: Per-chunk relation sets.
        :return: Validated relations per chunk.
        """

        # Select ICL examples for relation validation if manager is initialized
        examples_list: List[List[dict[str, Any]] | None] = []
        if self.icl_manager:
            examples_list = list(await self.icl_manager.batch_select_examples(
                query_texts=context,
                task="relation_validation",
                num_examples=self.icl_manager.config.num_examples
            ))
        else:
            examples_list = [None] * len(context)

        instruction: RAGUInstruction = self.get_prompt("relation_validation")
        assert instruction.pydantic_model is RelationsExtractionModel

        conversations: List[ChatMessages] = render_with_few_shots(
            instruction.messages,
            examples_list=examples_list,
            few_shot_formatter=instruction.few_shot_formatter,
            context=context,
            entities=entities_payload,
            relations=self._models_to_payload(relations),
            language=self.language,
            relation_types=self._relation_types_for(entities_payload),
            type_signatures=self.show_type_signatures,
        )

        results = await self.llm.batch_chat_completion(
            [conversation.to_openai() for conversation in conversations],
            output_schema=instruction.pydantic_model,
            continue_on_error=True,
            desc="Validating extracted relations",
        )
        for i, relations_model in enumerate(results):
            if relations_model is not None:
                logger.debug(f"After validation got {len(relations_model.relations)} relations")
            else:
                logger.warning("LLM call failed for relation validation chunk at index {}", i)

        return results

    def _relation_types_for(
        self,
        entities_payload: List[List[dict[str, Any]]],
    ) -> str | List[str] | None:
        """
        Build the predicate list injected into the relation prompts.

        With pruning enabled this becomes a per-chunk value, which
        :func:`~ragu.common.prompts.messages.render` treats as a batch parameter
        aligned with ``context``.

        :param entities_payload: Per-chunk entity payloads of the current batch.
        :return: One string for the whole batch, or one string per chunk.
        """
        if self.ontology is None or not self.prune_relation_types:
            return self.relation_types

        rendered: List[str] = []
        for entities in entities_payload:
            types = {str(entity.get("entity_type", "")) for entity in entities}
            pairs = [(subject, obj) for subject in types for obj in types]
            pruned = self.ontology.render_relation_types(
                with_signatures=self.show_type_signatures,
                for_pairs=pairs,
            )
            rendered.append(pruned or self.relation_types or "")
        return rendered

    @staticmethod
    def _to_entities(entities: Sequence[EntityModel], chunk: Chunk) -> List[Entity]:
        """
        Convert stage-1 entity models of a single chunk into graph entities.

        :param entities: Extracted entities for one chunk.
        :param chunk: Source chunk the entities were extracted from.
        :return: Graph entities referencing the source chunk.
        """
        return [
            Entity(
                entity_name=entity_model.entity_name,
                entity_type=entity_model.entity_type,
                description=entity_model.description,
                source_chunk_id=[chunk.id],
                documents_id=[],
                clusters=[],
            )
            for entity_model in entities
        ]

    @staticmethod
    def _models_to_payload(models: List[ModelT | None]) -> List[List[dict[str, Any]]]:
        """
        Convert stage models to JSON-like payloads expected by Jinja templates.

        :param models: Batch of pydantic models containing list fields.
            ``None`` entries (failed LLM calls) produce empty payloads.
        :return: List of list dictionaries per chunk.
        """
        payload: List[List[dict[str, Any]]] = []
        for model in models:
            if model is None:
                payload.append([])
                continue
            data = model.model_dump()
            first_value: Any = next(iter(data.values()), [])
            payload.append(cast(List[dict[str, Any]], first_value))
        return payload
