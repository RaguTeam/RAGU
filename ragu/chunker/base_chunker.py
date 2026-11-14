from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from typing import List

from ragu.chunker.types import Chunk, Document


class BaseChunker(ABC):
    """
    Abstract base class for text chunking strategies.
    Should be subclassed with specific chunking implementations.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def split(self, documents: str | List[str]) -> List[Chunk]:
        """
        Abstract method for splitting documents into smaller chunks.
        Must be implemented in subclasses.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        pass

    async def a_split(self, documents: str | Document | Sequence[str | Document]) -> list[Chunk]:
        """
        Adapt synchronous splitting for the async pipeline.

        String inputs are passed to :meth:`split` as a batch. Document inputs
        are split individually to preserve their source IDs. Subclasses can
        override this method when splitting requires asynchronous operations.

        :param documents: Input documents.
        :return: Chunks in source order.
        """
        if isinstance(documents, str):
            return self.split(documents)
        if isinstance(documents, Document):
            documents = [documents]
        if all(isinstance(document, str) for document in documents):
            return self.split(list(documents))

        chunks: list[Chunk] = []
        for document in documents:
            if isinstance(document, Document):
                chunks.extend(
                    replace(chunk, doc_id=document.id)
                    for chunk in self.split(document.content)
                )
            else:
                chunks.extend(self.split(document))
        return chunks

    def __call__(self, documents: str | List[str]) -> List[Chunk]:
        """
        Calls the chunker on a given list of documents.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        return self.split(documents)
