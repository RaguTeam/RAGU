from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple, List


from ragu.chunker.types import Chunk
from ragu.common.logger import logger
from ragu.common.prompts.default_models import EntityModel, RelationModel
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.graph.types import Entity, Relation
from ragu.common.base import RaguGenerativeModule
from ragu.triplet.ontology import OntologyValidator


class BaseArtifactExtractor(RaguGenerativeModule, ABC):
    """
    Abstract base class for entity and relation extraction modules.

    This class defines a unified interface for all artifact extraction components
    used in the RAG pipeline. Subclasses must implement the :meth:`extract`
    method to transform raw text chunks into structured graph entities and relations.
    """

    def __init__(
        self,
        prompts: list[str] | dict[str, RAGUInstruction],
        validator: OntologyValidator | None = None,
    ) -> None:
        """
        Initialize a new :class:`BaseArtifactExtractor`.

        :param prompts: One or more prompt templates used for extraction or validation.
        :param validator: Ontology enforcement applied to extracted artifacts, or
            ``None`` to accept whatever the model produced.
        """
        super().__init__(prompts)
        self.validator = validator

    def _apply_ontology(
        self,
        entities_per_chunk: Sequence[Sequence[EntityModel]],
        relations_per_chunk: Sequence[Sequence[RelationModel]],
        stage: str,
    ) -> Tuple[List[List[EntityModel]], List[List[RelationModel]]]:
        """
        Enforce the ontology on a batch of extracted artifacts.

        Runs on stage models rather than on graph objects: ``Entity.id`` hashes the
        entity type, so a type corrected after construction would leave a stale id.
        A no-op when the extractor was built without a validator.

        :param entities_per_chunk: Entities of each chunk.
        :param relations_per_chunk: Relations of each chunk, aligned with the entities.
        :param stage: Pipeline stage name, used in the log record.
        :return: Surviving entities and relations, per chunk.
        :raises ArtifactViolationError: If a failing check has a ``RAISE`` policy.
        """
        if self.validator is None:
            return (
                [list(entities) for entities in entities_per_chunk],
                [list(relations) for relations in relations_per_chunk],
            )

        entities, relations, report = self.validator.validate_batch(
            entities_per_chunk, relations_per_chunk
        )
        if not report.is_clean:
            logger.warning("Ontology validation ({}): {}", stage, report.summary())
            logger.debug(
                "Ontology validation ({}) breakdown:\n{}", stage, report.breakdown()
            )

        return entities, relations

    @abstractmethod
    async def extract(
        self,
        chunks: List[Chunk],
        *args: Any,
        **kwargs: Any
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Abstract method for extracting entities and relations from text chunks.

        Subclasses must implement this method and return all extracted entities
        and relations corresponding to the provided text inputs.

        :param chunks: List of :class:`Chunk` objects containing text content.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: A tuple ``(entities, relations)`` with lists of extracted objects.
        """
        pass

    async def __call__(
        self,
        chunks: List[Chunk],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Execute artifact extraction when the object is called as a coroutine.

        This convenience wrapper calls :meth:`extract` directly, allowing
        the extractor to be used in functional or pipeline-style workflows.

        :param chunks: List of :class:`Chunk` objects to process.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Extracted entities and relations.
        """
        return await self.extract(chunks, *args, **kwargs)
