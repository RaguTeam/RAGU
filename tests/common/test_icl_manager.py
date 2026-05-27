import json

import numpy as np
import pytest

from ragu.models.embedder import Embedder
from ragu.common.prompts.icl_config import ICLConfig
from ragu.common.prompts.icl_manager import Example, InContextLearningManager, resolve_example_path
from ragu.common.global_parameters import Settings
from ragu.common.logger import logger


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
    )


EN_ENTITY_EXAMPLES = [
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
]

RU_EXAMPLE = {
    "id": "ex-3",
    "input_text": "\u041c\u043e\u0441\u043a\u0432\u0430 \u2014 \u0441\u0442\u043e\u043b\u0438\u0446\u0430 \u0420\u043e\u0441\u0441\u0438\u0438.",
    "output": {
        "entities": [
            {"entity_name": "\u041c\u043e\u0441\u043a\u0432\u0430", "entity_type": "CITY"},
        ],
        "relations": [],
    },
    "metadata": {"domain": "geography", "language": "russian"},
    "quality_rating": 9,
}

EN_RELATION_EXAMPLES = [
    {
        "id": "rel-1",
        "input_text": "Microsoft acquired LinkedIn for $26 billion.",
        "output": {
            "relations": [
                {"source_entity": "Microsoft", "target_entity": "LinkedIn", "relation_type": "ACQUIRED"},
            ],
            "entities": [],
        },
        "metadata": {"domain": "business", "language": "english"},
        "quality_rating": 9,
    },
]


@pytest.fixture
def example_file(tmp_path):
    path = str(tmp_path / "test_examples.json")
    _write_example_file(path, EN_ENTITY_EXAMPLES + [RU_EXAMPLE])
    return path


@pytest.fixture
def example_files(tmp_path):
    entity_path = str(tmp_path / "entity_examples.json")
    _write_example_file(entity_path, EN_ENTITY_EXAMPLES + [RU_EXAMPLE])

    relation_path = str(tmp_path / "relation_examples.json")
    _write_example_file(relation_path, EN_RELATION_EXAMPLES)

    return {
        "entity_extraction": entity_path,
        "relation_extraction": relation_path,
    }


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
    async def test_initialize_loads_examples(self, embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files,
            config=icl_config,
            embedder=embedder,
        )
        await manager.initialize()
        assert len(manager.examples) == 3
        assert all(ex.language == "english" for ex in manager.examples)
        assert manager._initialized

    @pytest.mark.asyncio
    async def test_initialize_filters_by_language(self, embedder, icl_config, example_files, monkeypatch):
        monkeypatch.setattr(Settings, "language", "russian")
        manager = InContextLearningManager(
            example_files=example_files,
            config=icl_config,
            embedder=embedder,
        )
        await manager.initialize()
        assert len(manager.examples) == 1
        assert manager.examples[0].id == "ex-3"

    @pytest.mark.asyncio
    async def test_initialize_missing_file(self, icl_config, tmp_path):
        manager = InContextLearningManager(
            example_files={"test": str(tmp_path / "nonexistent.json")},
            config=icl_config,
            embedder=DeterministicEmbedder(dim=64),
        )
        await manager.initialize()
        assert len(manager.examples) == 0

    @pytest.mark.asyncio
    async def test_initialize_no_matching_language(self, embedder, icl_config, example_files, monkeypatch):
        monkeypatch.setattr(Settings, "language", "french")
        manager = InContextLearningManager(
            example_files=example_files,
            config=icl_config,
            embedder=embedder,
        )
        await manager.initialize()
        assert len(manager.examples) == 0

    @pytest.mark.asyncio
    async def test_task_indices_built(self, embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files,
            config=icl_config,
            embedder=embedder,
        )
        await manager.initialize()
        assert "entity_extraction" in manager._task_indices
        assert "relation_extraction" in manager._task_indices
        assert len(manager._task_indices["entity_extraction"]) == 2
        assert len(manager._task_indices["relation_extraction"]) == 1

    @pytest.mark.asyncio
    async def test_examples_tagged_with_task(self, embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files,
            config=icl_config,
            embedder=embedder,
        )
        await manager.initialize()
        entity_examples = [ex for ex in manager.examples if ex.task == "entity_extraction"]
        relation_examples = [ex for ex in manager.examples if ex.task == "relation_extraction"]
        assert len(entity_examples) == 2
        assert len(relation_examples) == 1


class TestSemanticSelection:
    @pytest.mark.asyncio
    async def test_select_examples_returns_correct_count(self, constant_embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files, config=icl_config,
            embedder=constant_embedder,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Tech company founded in California"], task="entity_extraction",
        )
        assert len(results[0]) == 2

    @pytest.mark.asyncio
    async def test_select_examples_respects_num_examples_override(self, constant_embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files, config=icl_config,
            embedder=constant_embedder,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Tech company founded in California"], task="entity_extraction", num_examples=1,
        )
        assert len(results[0]) == 1

    @pytest.mark.asyncio
    async def test_select_examples_returns_dicts(self, constant_embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files, config=icl_config,
            embedder=constant_embedder,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Tech company founded in California"], task="entity_extraction", num_examples=1,
        )
        ex = results[0][0]
        assert "input_text" in ex
        assert "output" in ex
        assert "id" in ex

    @pytest.mark.asyncio
    async def test_select_examples_empty_when_no_examples(self, constant_embedder, icl_config, tmp_path):
        empty_file = str(tmp_path / "empty.json")
        _write_example_file(empty_file, [])
        manager = InContextLearningManager(
            example_files={"test": empty_file}, config=icl_config,
            embedder=constant_embedder,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(["some query"], task="test")
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_select_examples_empty_without_initialize(self, constant_embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files, config=icl_config,
            embedder=constant_embedder,
        )
        results = await manager.batch_select_examples(["some query"], task="entity_extraction")
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_select_examples_filters_by_task(self, constant_embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files, config=icl_config,
            embedder=constant_embedder,
        )
        await manager.initialize()

        entity_results = await manager.batch_select_examples(
            ["Tech company"], task="entity_extraction", num_examples=10,
        )
        relation_results = await manager.batch_select_examples(
            ["Tech company"], task="relation_extraction", num_examples=10,
        )

        assert all(ex["id"].startswith("ex-") for ex in entity_results[0])
        assert all(ex["id"].startswith("rel-") for ex in relation_results[0])
        assert len(relation_results[0]) == 1

    @pytest.mark.asyncio
    async def test_select_examples_unknown_task_returns_empty(self, constant_embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files, config=icl_config,
            embedder=constant_embedder,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["some query"], task="nonexistent_task",
        )
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_select_examples_no_task_uses_all(self, constant_embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files, config=icl_config,
            embedder=constant_embedder,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Tech company"], num_examples=10,
        )
        assert len(results[0]) == 3


class TestBM25Selection:
    @pytest.mark.asyncio
    async def test_bm25_returns_examples_without_embedder(self, example_files):
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="bm25")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()
        assert manager._bm25_embedder is not None
        assert manager._bm25_doc_embeddings is not None
        results = await manager.batch_select_examples(
            ["Tech company founded in California"],
            task="entity_extraction",
        )
        assert len(results[0]) == 2

    @pytest.mark.asyncio
    async def test_bm25_returns_correct_count(self, example_files):
        config = ICLConfig(enabled=True, num_examples=1, selection_strategy="bm25")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Tech company"],
            task="entity_extraction",
        )
        assert len(results[0]) == 1

    @pytest.mark.asyncio
    async def test_bm25_filters_by_task(self, example_files):
        config = ICLConfig(enabled=True, num_examples=10, selection_strategy="bm25")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()

        entity_results = await manager.batch_select_examples(
            ["Tech company"], task="entity_extraction",
        )
        relation_results = await manager.batch_select_examples(
            ["Business acquisition"], task="relation_extraction",
        )

        assert all(ex["id"].startswith("ex-") for ex in entity_results[0])
        assert all(ex["id"].startswith("rel-") for ex in relation_results[0])

    @pytest.mark.asyncio
    async def test_bm25_selects_relevant_examples(self, example_files):
        config = ICLConfig(enabled=True, num_examples=1, selection_strategy="bm25")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()

        results = await manager.batch_select_examples(
            ["Apple and Steve Jobs founded a company"],
            task="entity_extraction",
        )
        assert len(results[0]) == 1
        assert results[0][0]["id"] == "ex-1"

    @pytest.mark.asyncio
    async def test_bm25_with_empty_examples(self, tmp_path):
        empty_file = str(tmp_path / "empty.json")
        _write_example_file(empty_file, [])
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="bm25")
        manager = InContextLearningManager(
            example_files={"test": empty_file},
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(["some query"], task="test")
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_bm25_unknown_task_returns_empty(self, example_files):
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="bm25")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["some query"], task="nonexistent_task",
        )
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_bm25_multiple_queries(self, example_files):
        config = ICLConfig(enabled=True, num_examples=1, selection_strategy="bm25")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Apple was founded by Steve Jobs", "Einstein developed the theory of relativity"],
            task="entity_extraction",
        )
        assert len(results) == 2
        assert results[0][0]["id"] == "ex-1"
        assert results[1][0]["id"] == "ex-2"


class TestHybridSelection:
    @pytest.mark.asyncio
    async def test_hybrid_requires_embedder(self, example_files):
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="hybrid")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        with pytest.raises(ValueError, match="Embedder is required"):
            await manager.initialize()

    @pytest.mark.asyncio
    async def test_hybrid_returns_examples(self, embedder, example_files):
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="hybrid")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
            embedder=embedder,
        )
        await manager.initialize()
        assert manager._example_matrix is not None
        assert manager._bm25_embedder is not None
        assert manager._bm25_doc_embeddings is not None

        results = await manager.batch_select_examples(
            ["Apple was founded by Steve Jobs"],
            task="entity_extraction",
        )
        assert len(results[0]) == 2

    @pytest.mark.asyncio
    async def test_hybrid_filters_by_task(self, embedder, example_files):
        config = ICLConfig(enabled=True, num_examples=10, selection_strategy="hybrid")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
            embedder=embedder,
        )
        await manager.initialize()

        entity_results = await manager.batch_select_examples(
            ["Tech company"], task="entity_extraction",
        )
        relation_results = await manager.batch_select_examples(
            ["Business acquisition"], task="relation_extraction",
        )

        assert all(ex["id"].startswith("ex-") for ex in entity_results[0])
        assert all(ex["id"].startswith("rel-") for ex in relation_results[0])

    @pytest.mark.asyncio
    async def test_hybrid_respects_num_examples(self, constant_embedder, example_files):
        config = ICLConfig(enabled=True, num_examples=1, selection_strategy="hybrid")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
            embedder=constant_embedder,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Tech company"],
            task="entity_extraction",
        )
        assert len(results[0]) == 1

    @pytest.mark.asyncio
    async def test_hybrid_with_empty_examples(self, embedder, tmp_path):
        empty_file = str(tmp_path / "empty.json")
        _write_example_file(empty_file, [])
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="hybrid")
        manager = InContextLearningManager(
            example_files={"test": empty_file},
            config=config,
            embedder=embedder,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(["some query"], task="test")
        assert results == [[]]


class TestRandomSelection:
    @pytest.mark.asyncio
    async def test_random_returns_examples_without_embedder(self, example_files):
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="random")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Tech company"],
            task="entity_extraction",
        )
        assert len(results[0]) == 2

    @pytest.mark.asyncio
    async def test_random_returns_correct_count(self, example_files):
        config = ICLConfig(enabled=True, num_examples=1, selection_strategy="random")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Tech company"],
            task="entity_extraction",
        )
        assert len(results[0]) == 1

    @pytest.mark.asyncio
    async def test_random_filters_by_task(self, example_files):
        config = ICLConfig(enabled=True, num_examples=10, selection_strategy="random")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()

        entity_results = await manager.batch_select_examples(
            ["Tech company"], task="entity_extraction",
        )
        relation_results = await manager.batch_select_examples(
            ["Business"], task="relation_extraction",
        )

        assert all(ex["id"].startswith("ex-") for ex in entity_results[0])
        assert all(ex["id"].startswith("rel-") for ex in relation_results[0])
        assert len(entity_results[0]) == 2
        assert len(relation_results[0]) == 1

    @pytest.mark.asyncio
    async def test_random_returns_dicts(self, example_files):
        config = ICLConfig(enabled=True, num_examples=1, selection_strategy="random")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["Tech company"], task="entity_extraction",
        )
        ex = results[0][0]
        assert "input_text" in ex
        assert "output" in ex
        assert "id" in ex

    @pytest.mark.asyncio
    async def test_random_unknown_task_returns_empty(self, example_files):
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="random")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["some query"], task="nonexistent_task",
        )
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_random_with_empty_examples(self, tmp_path):
        empty_file = str(tmp_path / "empty.json")
        _write_example_file(empty_file, [])
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="random")
        manager = InContextLearningManager(
            example_files={"test": empty_file},
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(["some query"], task="test")
        assert results == [[]]

    @pytest.mark.asyncio
    async def test_random_multiple_queries(self, example_files):
        config = ICLConfig(enabled=True, num_examples=2, selection_strategy="random")
        manager = InContextLearningManager(
            example_files=example_files,
            config=config,
        )
        await manager.initialize()
        results = await manager.batch_select_examples(
            ["query one", "query two", "query three"],
        )
        assert len(results) == 3
        for per_query in results:
            assert len(per_query) == 2


class TestInContextLearningManagerLowMatchWarning:
    @pytest.mark.asyncio
    async def test_low_match_warning_logged_with_empty_corpus(self, embedder, tmp_path):
        empty_file = str(tmp_path / "empty.json")
        _write_example_file(empty_file, [])
        config = ICLConfig(
            enabled=True,
            num_examples=2,
            low_match_warning_threshold=0.01,
        )
        manager = InContextLearningManager(
            example_files={"test": empty_file}, config=config,
            embedder=embedder,
        )
        await manager.initialize()
        captured = []
        handler_id = logger.add(captured.append, level="WARNING", format="{message}")
        try:
            await manager.batch_select_examples(
                ["query one", "query two"],
                task="test",
            )
        finally:
            logger.remove(handler_id)
        assert any("ICL low match rate" in msg for msg in captured)

    @pytest.mark.asyncio
    async def test_no_warning_when_match_rate_ok(self, constant_embedder, icl_config, example_files):
        manager = InContextLearningManager(
            example_files=example_files, config=icl_config,
            embedder=constant_embedder,
        )
        await manager.initialize()
        captured = []
        handler_id = logger.add(captured.append, level="WARNING", format="{message}")
        try:
            await manager.batch_select_examples(
                ["Tech company founded in California"],
                task="entity_extraction",
            )
        finally:
            logger.remove(handler_id)
        assert not any("ICL low match rate" in msg for msg in captured)

    @pytest.mark.asyncio
    async def test_warning_suppressed_when_threshold_zero(self, embedder, tmp_path):
        empty_file = str(tmp_path / "empty.json")
        _write_example_file(empty_file, [])
        config = ICLConfig(
            enabled=True,
            num_examples=2,
            low_match_warning_threshold=0.0,
        )
        manager = InContextLearningManager(
            example_files={"test": empty_file}, config=config,
            embedder=embedder,
        )
        await manager.initialize()
        captured = []
        handler_id = logger.add(captured.append, level="WARNING", format="{message}")
        try:
            await manager.batch_select_examples(
                ["query one"], task="test",
            )
        finally:
            logger.remove(handler_id)
        assert not any("ICL low match rate" in msg for msg in captured)


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
        config = ICLConfig(enabled=True)
        path = resolve_example_path(None, "artifact_extraction_examples.json")
        manager = InContextLearningManager(
            example_files={"artifact_extraction": path}, config=config,
            embedder=embedder,
        )
        await manager.initialize()
        assert len(manager.examples) > 0

    @pytest.mark.asyncio
    async def test_load_builtin_entity_examples(self, embedder):
        config = ICLConfig(enabled=True)
        path = resolve_example_path(None, "entity_extraction_examples.json")
        manager = InContextLearningManager(
            example_files={"entity_extraction": path}, config=config,
            embedder=embedder,
        )
        await manager.initialize()
        assert len(manager.examples) > 0

    @pytest.mark.asyncio
    async def test_load_builtin_relation_examples(self, embedder):
        config = ICLConfig(enabled=True)
        path = resolve_example_path(None, "relation_extraction_examples.json")
        manager = InContextLearningManager(
            example_files={"relation_extraction": path}, config=config,
            embedder=embedder,
        )
        await manager.initialize()
        assert len(manager.examples) > 0

    @pytest.mark.asyncio
    async def test_load_multiple_builtin_examples(self, embedder):
        config = ICLConfig(enabled=True)
        manager = InContextLearningManager(
            example_files={
                "entity_extraction": resolve_example_path(None, "entity_extraction_examples.json"),
                "entity_validation": resolve_example_path(None, "entity_validation_examples.json"),
                "relation_extraction": resolve_example_path(None, "relation_extraction_examples.json"),
                "relation_validation": resolve_example_path(None, "relation_validation_examples.json"),
            },
            config=config,
            embedder=embedder,
        )
        await manager.initialize()
        assert len(manager.examples) > 0
        assert len(manager._task_indices) == 4
