from abc import abstractmethod, ABC
from typing import List


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedders.
    """

    def __init__(self, dim: int | None = None):
        """
        Initialize base embedder.

        :param dim: Embedding dimensionality if known.
        """
        self.dim = dim

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Computes embeddings for a list of text inputs.

        :param texts: Input texts.
        :return: Embedding vectors aligned with input order.
        """
        ...

    async def __call__(self, *args, **kwargs) -> List[List[float]]:
        """
        Call alias for ``embed``.
        """
        return await self.embed(*args, **kwargs)
