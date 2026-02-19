from ragu.embedder.base_embedder import BaseEmbedder
from ragu.common.batch_generator import BatchGenerator

import numpy as np

from ragu.utils.ragu_utils import FLOATS


class STEmbedder(BaseEmbedder):
    """
    Embedder that uses Sentence Transformers to compute text embeddings.

    Warning:
    This embedder currently has limited support and can be unstable. Use OpenAIEmbedder instead.
    """

    def __init__(self, model_name_or_path: str, dim: int=None, *args, **kwargs):
        """
        Initializes the STEmbedder with a specified model.

        :param model_name_or_path: Path or name of the Sentence Transformer model.
        """

        raise ImportError(
            "[STEmbedder] Current support is limited and unstable. "
            "Please, use OpenAIEmbedder for now."
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "RAGU needs SentenceTransformer to use this class. Please install it using `pip install sentence-transformers`."
            )
        super().__init__()
        self.model = SentenceTransformer(model_name_or_path, **kwargs)
        self.dim = dim or self.model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str], batch_size: int=16) -> list[list[float]] | FLOATS:
        """
        Computes embeddings for a list of strings.

        :param texts: Input text(s) to embed.
        :param batch_size: Batch size.
        :return: Embeddings for the input text(s).
        """

        batch_generator = BatchGenerator(texts, batch_size=batch_size)
        embeddings_list = [
            self.model.encode(batch, show_progress_bar=False)
            for batch in batch_generator.get_batches()
        ]

        return np.concatenate(embeddings_list)
