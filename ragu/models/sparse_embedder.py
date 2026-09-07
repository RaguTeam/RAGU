import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Iterable

from fastembed import SparseTextEmbedding
from fastembed.common.types import Device
from fastembed.sparse.bm42 import Bm42 as FastEmbedBM42
from fastembed.sparse.bm25 import Bm25 as FastEmbedBM25

from ragu.common.global_parameters import Settings
from ragu.storage.types import SparseEmbedding
from ragu.utils.text_normalize import BaseNormalizer


class SparseEmbedder(ABC):
    @abstractmethod
    def embed_query(self, texts: List[str]) -> List[SparseEmbedding]:
        ...

    @abstractmethod
    def embed_document(self, texts: List[str]) -> List[SparseEmbedding]:
        ...


class BM25(SparseEmbedder):
    """
    Sparse embedder backed by FastEmbed BM25.
    """

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        cache_dir: str | None = None,
        k: float = 1.2,
        b: float = 0.75,
        avg_len: float = 256.0,
        language: str | None = None,
        token_max_length: int = 40,
        disable_stemmer: bool = False,
        specific_model_path: str | None = None,
        normalizer: BaseNormalizer | None = None,
        **kwargs: Any,
    ) -> None:
        self.specific_model_path = specific_model_path
        self.normalizer = normalizer
        self.kwargs = dict(kwargs)

        if not disable_stemmer and normalizer:
            raise ValueError("You cannot use custom normalizer along with default fastembed stemmer."
                             " Set `disable_stemmer` to True or remove normalizer")

        self._model = FastEmbedBM25(
            model_name=model_name,
            cache_dir=cache_dir,
            k=k,
            b=b,
            avg_len=avg_len,
            language=language if language else Settings.language,
            token_max_length=token_max_length,
            disable_stemmer=disable_stemmer,
            specific_model_path=specific_model_path,
            **kwargs,
        )

    def embed_document(self, texts: List[str]) -> List[SparseEmbedding]:
        if not texts:
            return []
        normalized_texts = self.normalizer.normalize_batch(texts) if self.normalizer else texts
        return [
            SparseEmbedding(
                indices=embedding.indices.astype(int).tolist(),
                values=embedding.values.astype(float).tolist(),
            )
            for embedding in self._model.embed(normalized_texts)
        ]

    def embed_query(self, texts: List[str]) -> List[SparseEmbedding]:
        if not texts:
            return []
        normalized_texts = self.normalizer.normalize_batch(texts) if self.normalizer else texts
        return [
            SparseEmbedding(
                indices=embedding.indices.astype(int).tolist(),
                values=embedding.values.astype(float).tolist(),
            )
            for embedding in self._model.query_embed(normalized_texts)
        ]


class BM42(SparseEmbedder):
    """
    Sparse embedder backed by FastEmbed BM42.

    BM42 is expected to be used with Qdrant sparse vectors configured with
    `modifier=IDF`, just like FastEmbed BM25.
    """

    def __init__(
        self,
        model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
        cache_dir: str | None = None,
        alpha: float = 0.5,
        normalizer: BaseNormalizer | None = None,
        batch_size: int = 32,
        parallel: int | None = None,
        threads: int | None = None,
        providers: list[str | tuple[str, dict[Any, Any]]] | None = None,
        cuda: bool | str = "auto",
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        specific_model_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        if normalizer is not None:
            raise ValueError("BM42 does not support a custom normalizer because FastEmbed BM42 "
                             "already applies its own tokenizer, stopword filtering, and stemming.")

        self.normalizer = normalizer
        self.batch_size = batch_size
        self.parallel = parallel
        self.kwargs = dict(kwargs)

        self._model = FastEmbedBM42(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            alpha=alpha,
            cuda=Device(cuda) if isinstance(cuda, str) else cuda,
            device_ids=device_ids,
            lazy_load=lazy_load,
            specific_model_path=specific_model_path,
            **kwargs,
        )

    def embed_document(self, texts: List[str]) -> List[SparseEmbedding]:
        if not texts:
            return []

        return [
            SparseEmbedding(
                indices=embedding.indices.astype(int).tolist(),
                values=embedding.values.astype(float).tolist(),
            )
            for embedding in self._model.embed(texts, batch_size=self.batch_size, parallel=self.parallel)
        ]

    def embed_query(self, texts: List[str]) -> List[SparseEmbedding]:
        if not texts:
            return []

        return [
            SparseEmbedding(
                indices=embedding.indices.astype(int).tolist(),
                values=embedding.values.astype(float).tolist(),
            )
            for embedding in self._model.query_embed(texts)
        ]


class SPLADE(SparseEmbedder):
    """
    Sparse embedder backed by FastEmbed SPLADE++.

    Notes:
    - Default model is English-only: `prithivida/Splade_PP_en_v1`.
    - Unlike BM25, SPLADE does not use Qdrant IDF modifier.
    - In FastEmbed docs, SPLADE is used via `SparseTextEmbedding`.
    """

    def __init__(
        self,
        model_name_or_path: str = "prithivida/Splade_PP_en_v1",
        cache_dir: str | None = None,
        normalizer: BaseNormalizer | None = None,
        batch_size: int = 32,
        parallel: int = 4,
        **kwargs: Any,
    ) -> None:
        self.normalizer = normalizer
        self.parallel = parallel
        self.batch_size = batch_size
        self.kwargs = dict(kwargs)

        init_signature = inspect.signature(SparseTextEmbedding.__init__)
        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in init_signature.parameters
        }

        self._model = SparseTextEmbedding(
            model_name=model_name_or_path,
            cache_dir=cache_dir,
            **supported_kwargs,
        )

    def embed_document(self, texts: List[str]) -> List[SparseEmbedding]:
        if not texts:
            return []

        normalized_texts = self.normalizer.normalize_batch(texts) if self.normalizer else texts

        return [
            SparseEmbedding(
                indices=embedding.indices.astype(int).tolist(),
                values=embedding.values.astype(float).tolist(),
            ) for embedding in self._model.embed(normalized_texts, batch_size=self.batch_size, parallel=self.parallel)
        ]

    def embed_query(self, texts: List[str]) -> List[SparseEmbedding]:
        if not texts:
            return []

        normalized_texts = self.normalizer.normalize_batch(texts) if self.normalizer else texts

        # Some FastEmbed models/classes expose query_embed, but SPLADE docs
        # consistently show plain `.embed(...)`. So we safely fall back to it.
        query_embed: Callable[[Iterable[str]], Iterable[Any]] = getattr(self._model, "query_embed", self._model.embed)
        return [
            SparseEmbedding(
                indices=embedding.indices.astype(int).tolist(),
                values=embedding.values.astype(float).tolist(),
            ) for embedding in query_embed(normalized_texts)
        ]
