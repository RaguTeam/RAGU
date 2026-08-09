from abc import ABC, abstractmethod
from typing import Any, Tuple, List


from ragu.chunker.types import Chunk
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.graph.types import Entity, Relation
from ragu.common.base import RaguGenerativeModule


class BaseArtifactExtractor(RaguGenerativeModule, ABC):
    """
    Abstract base class for entity and relation extraction modules.

    This class defines a unified interface for all artifact extraction components
    used in the RAG pipeline. Subclasses must implement the :meth:`extract`
    method to transform raw text chunks into structured graph entities and relations.
    """

    def __init__(self, prompts: list[str] | dict[str, RAGUInstruction]) -> None:
        """
        Initialize a new :class:`BaseArtifactExtractor`.

        :param prompts: One or more prompt templates used for extraction or validation.
        """
        super().__init__(prompts)

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
