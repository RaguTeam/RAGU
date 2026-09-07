from abc import ABC, abstractmethod
from typing import List
from ragu.chunker.types import Chunk


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

    def __call__(self, documents: str | List[str]) -> List[Chunk]:
        """
        Calls the chunker on a given list of documents.

        :param documents: List of input documents.
        :return: List of text chunks.
        """
        return self.split(documents)
