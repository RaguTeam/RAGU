import json

import numpy as np
import pytest

from ragu.models.embedder import Embedder
from ragu.common.prompts.icl_config import ICLConfig
from ragu.common.prompts.icl_manager import Example, InContextLearningManager, resolve_example_path


class DeterministicEmbedder(Embedder):
    """Embedder that produces deterministic vectors based on text hash."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text: str, **kwargs) -> list[float]:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(self._dim).tolist()


class ConstantEmbedder(Embedder):
    """Embedder that returns the same normalized vector for all texts."""

    def __init__(self, dim: int = 64):
        self._dim = dim
        rng = np.random.RandomState(42)
        v = rng.randn(dim)
        v = v / np.linalg.norm(v)
        self._vector = v.tolist()

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text: str, **kwargs) -> list[float]:
        return self._vector


def _make_example_json(examples: list[dict], language: str = "english") -> dict:
    return {
        "version": "1.0",
        "languages": [language],
        "total_examples": len(examples),
        "examples": examples,
    }


def _write_example_file(path: str, examples: list[dict], language: str = "english"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_example_json(examples, language), f)


@pytest.fixture
def embedder():
    return DeterministicEmbedder(dim=64)


@pytest.fixture
def constant_embedder():
    return ConstantEmbedder(dim=64)


@pytest.fixture
def icl_config():
    return ICLConfig(
        enabled=True,
        num_examples=2,
        similarity_threshold=0.1,
        cache_embeddings=True,
        language="english",
    )


@pytest.fixture
def example_file(tmp_path):
    path = str(tmp_path / "test_examples.json")
    examples = [
        {
            "id": "ex-1",
            "input_text": "Apple was founded by Steve Jobs in California.",
            "output": {
                "entities": [
                    {"entity_name": "Apple", "entity_type": "ORGANIZATION"},
                    {"entity_name": "Steve Jobs", "entity_type": "PERSON"},
                ],
                "relations": [
                    {"source_entity": "Steve Jobs", "target_entity": "Apple", "relation_type": "FOUNDED_BY"},
                ],
            },
            "metadata": {"domain": "technology", "language": "english"},
            "quality_rating": 9,
        },
        {
            "id": "ex-2",
            "input_text": "Einstein developed the theory of relativity in Berlin.",
            "output": {
                "entities": [
                    {"entity_name": "Einstein", "entity_type": "PERSON"},
                    {"entity_name": "Berlin", "entity_type": "CITY"},
                ],
                "relations": [],
            },
            "metadata": {"domain": "science", "language": "english"},
            "quality_rating": 8,
        },
        {
            "id": "ex-3",
            "input_text": "Москва — столица России.",
            "output": {
                "entities": [
                    {"entity_name": "Москва", "entity_type": "CITY"},
                ],
                "relations": [],
            },
            "metadata": {"domain": "geography", "language": "russian"},
            "quality_rating": 9,
        },
    ]
    _write_example_file(path, examples)
    return path


class TestExample:
    def test_frozen(self):
        ex = Example(
            id="test", input_text="text", output={},
            metadata={}, language="english", quality_rating=None,
        )
        with pytest.raises(AttributeError):
            ex.id = "changed"

    def test_slots(self):
        ex = Example(
            id="test", input_text="text", output={},
            metadata={}, language="english", quality_rating=None,
        )
        assert hasattr(ex, "__slots__")


class TestInContextLearningManagerInit:
    @pytest.mark.asyncio
    async def test_initialize_loads_examples(self, embedder, icl_config, example_file):
        manager = InContextLearningManager(
            embedder=embedder,
            example_file=example_file,
            config=icl_config,
            language="english",
        )
        await manager.initialize()
        assert len(manager.examples) == 2
        assert all(ex.language == "english" for ex in manager.examples)
        assert manager._embeddings_computed

    @pytest.mark.asyncio
    async def test_initialize_filters_by_language(self, embedder, icl_config, example_file):
        manager = InContextLearningManager(
            embedder=embedder,
            example_file=example_file,
            config=icl_config,
            language="russian",
        )
        await manager.initialize()
        assert len(manager.examples) == 1
        assert manager.examples[0].id == "ex-3"

    @pytest.mark.asyncio
    async def test_initialize_missing_file(self, embedder, icl_config, tmp_path):
        manager = InContextLearningManager(
            embedder=embedder,
            example_file=str(tmp_path / "nonexistent.json"),
            config=icl_config,
        )
        await manager.initialize()
        assert len(manager.examples) == 0

    @pytest.mark.asyncio
    async def test_initialize_no_matching_language(self, embedder, icl_config, example_file):
        manager = InContextLearningManager(
            embedder=embedder,
            example_file=example_file,
            config=icl_config,
            language="french",
        )
        await manager.initialize()
        assert len(manager.examples) == 0


class TestInContextLearningManagerSelection:
    @pytest.mark.asyncio
    async def test_select_examples_returns_correct_count(self, constant_embedder, icl_config, example_file):
        manager = InContextLearningManager(
            embedder=constant_embedder, example_file=example_file, config=icl_config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(["Tech company founded in California"])
        assert len(results[0]) == 2

    @pytest.mark.asyncio
    async def test_select_examples_respects_num_examples_override(self, constant_embedder, icl_config, example_file):
        manager = InContextLearningManager(
            embedder=constant_embedder, example_file=example_file, config=icl_config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(["Tech company founded in California"], num_examples=1)
        assert len(results[0]) == 1

    @pytest.mark.asyncio
    async def test_select_examples_returns_dicts(self, constant_embedder, icl_config, example_file):
        manager = InContextLearningManager(
            embedder=constant_embedder, example_file=example_file, config=icl_config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(["Tech company founded in California"], num_examples=1)
        ex = results[0][0]
        assert "input_text" in ex
        assert "output" in ex
        assert "id" in ex

    @pytest.mark.asyncio
    async def test_select_examples_empty_when_no_examples(self, constant_embedder, icl_config, tmp_path):
        empty_file = str(tmp_path / "empty.json")
        _write_example_file(empty_file, [])
        manager = InContextLearningManager(
            embedder=constant_embedder, example_file=empty_file, config=icl_config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(["some query"])
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_select_examples_empty_without_initialize(self, constant_embedder, icl_config, example_file):
        manager = InContextLearningManager(
            embedder=constant_embedder, example_file=example_file, config=icl_config,
        )
        results = await manager.batch_select_examples(["some query"])
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_select_examples_threshold_filtering(self, embedder, icl_config, example_file):
        high_threshold_config = ICLConfig(
            enabled=True, num_examples=2,
            similarity_threshold=0.99, cache_embeddings=True,
        )
        manager = InContextLearningManager(
            embedder=embedder, example_file=example_file, config=high_threshold_config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(["something completely unrelated xyzzy"])
        assert results == [[]]


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert InContextLearningManager._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert InContextLearningManager._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert InContextLearningManager._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert InContextLearningManager._cosine_similarity(a, b) == 0.0

    def test_known_similarity(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        expected = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert InContextLearningManager._cosine_similarity(a, b) == pytest.approx(expected)


class TestEmbeddingCaching:
    @pytest.mark.asyncio
    async def test_embeddings_stored_when_cache_enabled(self, embedder, example_file):
        config = ICLConfig(cache_embeddings=True, similarity_threshold=0.0)
        manager = InContextLearningManager(
            embedder=embedder, example_file=example_file, config=config,
        )
        await manager.initialize()
        assert len(manager._embeddings) == len(manager.examples)

    @pytest.mark.asyncio
    async def test_embeddings_not_stored_when_cache_disabled(self, embedder, example_file):
        config = ICLConfig(cache_embeddings=False, similarity_threshold=0.0)
        manager = InContextLearningManager(
            embedder=embedder, example_file=example_file, config=config,
        )
        await manager.initialize()
        assert len(manager._embeddings) == 0


class TestResolveExamplePath:
    def test_none_returns_builtin_path(self):
        result = resolve_example_path(None, "artifact_extraction_examples.json")
        assert result.endswith("artifact_extraction_examples.json")
        assert "icl_examples" in result

    def test_none_returns_existing_file(self):
        from pathlib import Path
        result = resolve_example_path(None, "artifact_extraction_examples.json")
        assert Path(result).exists()

    def test_absolute_path_passed_through(self, tmp_path):
        base = str(tmp_path / "my_examples")
        result = resolve_example_path(base, "test.json")
        assert result == str(tmp_path / "my_examples" / "test.json")

    def test_relative_path_resolved_from_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = resolve_example_path("custom_dir", "test.json")
        assert result == str(tmp_path / "custom_dir" / "test.json")


class TestBuiltinExamples:
    @pytest.mark.asyncio
    async def test_load_builtin_artifact_examples(self, embedder):
        config = ICLConfig(enabled=True, similarity_threshold=0.0, language="english")
        path = resolve_example_path(None, "artifact_extraction_examples.json")
        manager = InContextLearningManager(
            embedder=embedder, example_file=path, config=config,
        )
        await manager.initialize()
        assert len(manager.examples) > 0

    @pytest.mark.asyncio
    async def test_load_builtin_entity_examples(self, embedder):
        config = ICLConfig(enabled=True, similarity_threshold=0.0, language="english")
        path = resolve_example_path(None, "entity_extraction_examples.json")
        manager = InContextLearningManager(
            embedder=embedder, example_file=path, config=config,
        )
        await manager.initialize()
        assert len(manager.examples) > 0

    @pytest.mark.asyncio
    async def test_load_builtin_relation_examples(self, embedder):
        config = ICLConfig(enabled=True, similarity_threshold=0.0, language="english")
        path = resolve_example_path(None, "relation_extraction_examples.json")
        manager = InContextLearningManager(
            embedder=embedder, example_file=path, config=config,
        )
        await manager.initialize()
        assert len(manager.examples) > 0
