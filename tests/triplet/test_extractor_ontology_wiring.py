from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragu.chunker.types import Chunk
from ragu.common.prompts.default_models import (
    ArtifactsModel,
    EntitiesExtractionModel,
    EntityModel,
    RelationModel,
    RelationsExtractionModel,
)
from ragu.triplet import (
    ArtifactsExtractorLLM,
    ArtifactViolationError,
    Ontology,
    TwoStageArtifactsExtractorLLM,
    ValidationPolicies,
)


def _chunk(text="Hello world"):
    return Chunk(content=text, chunk_order_idx=0, doc_id="doc-1")


def _entity(name, entity_type):
    return EntityModel(entity_name=name, entity_type=entity_type, description="d")


def _relation(source, target, relation_type):
    return RelationModel(
        source_entity=source,
        target_entity=target,
        relation_type=relation_type,
        description="d",
        relationship_strength=4,
    )


def _patch_render(module):
    patcher = patch.object(module, "render_with_few_shots")
    mock_render = patcher.start()

    messages = MagicMock()
    messages.to_openai.return_value = [{"role": "user", "content": "test"}]
    mock_render.return_value = [messages]

    return patcher, mock_render


def _prompt_stub(model):
    return SimpleNamespace(messages=[MagicMock()], pydantic_model=model, few_shot_formatter=None)


class TestSinglePassWiring:
    @staticmethod
    async def _run(artifacts, **extractor_kwargs):
        import ragu.triplet.llm_artifact_extractor as module

        llm = AsyncMock()
        llm.batch_chat_completion = AsyncMock(return_value=[artifacts])
        extractor = ArtifactsExtractorLLM(llm=llm, **extractor_kwargs)
        extractor.get_prompt = MagicMock(return_value=_prompt_stub(ArtifactsModel))

        patcher, _ = _patch_render(module)
        try:
            entities, relations = await extractor.extract([_chunk()])
        finally:
            patcher.stop()

        return extractor, entities, relations

    async def test_entity_type_is_canonicalized(self):
        artifacts = ArtifactsModel(entities=[_entity("Acme", "org")], relations=[])

        _, entities, _ = await self._run(artifacts)

        assert entities[0].entity_type == "ORGANIZATION"

    async def test_entity_outside_the_ontology_is_dropped(self):
        artifacts = ArtifactsModel(
            entities=[_entity("Alice", "PERSON"), _entity("Enterprise", "SPACESHIP")],
            relations=[],
        )

        _, entities, _ = await self._run(artifacts)

        assert [e.entity_name for e in entities] == ["Alice"]

    async def test_reversed_relation_is_repaired(self):
        artifacts = ArtifactsModel(
            entities=[_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")],
            relations=[_relation("Acme", "Alice", "WORKPLACE")],
        )

        _, _, relations = await self._run(artifacts)

        assert relations[0].subject_name == "Alice"
        assert relations[0].object_name == "Acme"

    async def test_ontology_none_disables_every_check(self):
        artifacts = ArtifactsModel(
            entities=[_entity("Enterprise", "SPACESHIP")],
            relations=[],
        )

        extractor, entities, _ = await self._run(artifacts, ontology=None)

        assert extractor.validator is None
        assert entities[0].entity_type == "SPACESHIP"

    async def test_raise_policy_propagates_out_of_extract(self):
        artifacts = ArtifactsModel(entities=[_entity("Enterprise", "SPACESHIP")], relations=[])

        with pytest.raises(ArtifactViolationError):
            await self._run(artifacts, validation=ValidationPolicies.failing())

    async def test_permissive_policies_keep_everything(self):
        artifacts = ArtifactsModel(
            entities=[_entity("Enterprise", "SPACESHIP")],
            relations=[],
        )

        _, entities, _ = await self._run(
            artifacts, validation=ValidationPolicies.permissive()
        )

        assert entities[0].entity_type == "SPACESHIP"


class TestTwoStageWiring:
    @staticmethod
    async def _run(entities_model, relations_model, **extractor_kwargs):
        import ragu.triplet.two_stage_extractor as module

        llm = AsyncMock()
        call_count = 0

        async def _batch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return [entities_model] if call_count == 1 else [relations_model]

        llm.batch_chat_completion = AsyncMock(side_effect=_batch)

        extractor = TwoStageArtifactsExtractorLLM(llm=llm, **extractor_kwargs)
        extractor.get_prompt = MagicMock(
            side_effect=lambda name: _prompt_stub(
                EntitiesExtractionModel if name.startswith("entity") else RelationsExtractionModel
            )
        )

        patcher, _ = _patch_render(module)
        try:
            entities, relations = await extractor.extract([_chunk()])
        finally:
            patcher.stop()

        return extractor, entities, relations

    async def test_entities_are_validated_before_the_relation_stage(self):
        """A dropped entity must not reach the relation prompt payload."""
        import ragu.triplet.two_stage_extractor as module

        entities_model = EntitiesExtractionModel(
            entities=[_entity("Alice", "PERSON"), _entity("Enterprise", "SPACESHIP")]
        )
        relations_model = RelationsExtractionModel(relations=[])

        llm = AsyncMock()
        call_count = 0

        async def _batch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return [entities_model] if call_count == 1 else [relations_model]

        llm.batch_chat_completion = AsyncMock(side_effect=_batch)
        extractor = TwoStageArtifactsExtractorLLM(llm=llm)
        extractor.get_prompt = MagicMock(
            side_effect=lambda name: _prompt_stub(
                EntitiesExtractionModel if name.startswith("entity") else RelationsExtractionModel
            )
        )

        patcher, mock_render = _patch_render(module)
        try:
            await extractor.extract([_chunk()])
            relation_stage_call = mock_render.call_args_list[-1]
        finally:
            patcher.stop()

        payload = relation_stage_call.kwargs["entities"][0]
        assert [item["entity_name"] for item in payload] == ["Alice"]

    async def test_reversed_relation_is_repaired(self):
        entities_model = EntitiesExtractionModel(
            entities=[_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]
        )
        relations_model = RelationsExtractionModel(
            relations=[_relation("Acme", "Alice", "WORKPLACE")]
        )

        _, entities, relations = await self._run(entities_model, relations_model)

        assert len(entities) == 2
        assert relations[0].subject_name == "Alice"
        assert relations[0].object_name == "Acme"

    async def test_relation_outside_the_ontology_is_dropped(self):
        entities_model = EntitiesExtractionModel(
            entities=[_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        )
        relations_model = RelationsExtractionModel(
            relations=[_relation("Alice", "Bob", "COLLABORATES_WITH")]
        )

        _, _, relations = await self._run(entities_model, relations_model)

        assert relations == []

    async def test_ontology_none_disables_every_check(self):
        entities_model = EntitiesExtractionModel(entities=[_entity("Enterprise", "SPACESHIP")])
        relations_model = RelationsExtractionModel(relations=[])

        extractor, entities, _ = await self._run(
            entities_model, relations_model, ontology=None
        )

        assert extractor.validator is None
        assert entities[0].entity_type == "SPACESHIP"


class TestOntologyDrivesThePrompt:
    def test_prompt_types_come_from_the_ontology(self):
        ontology = Ontology.from_type_lists(["LANGUAGE (a tongue)"], ["SPOKEN_IN"])

        extractor = ArtifactsExtractorLLM(llm=AsyncMock(), ontology=ontology)

        assert extractor.entity_types == "LANGUAGE (a tongue)"
        assert extractor.relation_types == "SPOKEN_IN"

    def test_builtin_can_be_named_by_string(self):
        extractor = TwoStageArtifactsExtractorLLM(llm=AsyncMock(), ontology="nerel")

        assert extractor.ontology is not None
        assert extractor.ontology.name == "nerel"
        assert "PERSON" in extractor.entity_types

    def test_ontology_none_leaves_the_prompt_unconstrained(self):
        extractor = ArtifactsExtractorLLM(llm=AsyncMock(), ontology=None)

        assert extractor.entity_types is None
        assert extractor.relation_types is None

    def test_signatures_are_off_by_default(self):
        extractor = TwoStageArtifactsExtractorLLM(llm=AsyncMock())

        assert "[PERSON -> ORGANIZATION" not in extractor.relation_types

    def test_signatures_are_rendered_when_asked(self):
        extractor = TwoStageArtifactsExtractorLLM(llm=AsyncMock(), show_type_signatures=True)

        assert "WORKPLACE [PERSON -> ORGANIZATION|FACILITY]" in extractor.relation_types

    def test_pruning_is_off_by_default(self):
        extractor = TwoStageArtifactsExtractorLLM(llm=AsyncMock())
        payload = [[{"entity_name": "A", "entity_type": "PERSON"}]]

        assert extractor._relation_types_for(payload) == extractor.relation_types

    def test_pruning_yields_one_list_per_chunk(self):
        extractor = TwoStageArtifactsExtractorLLM(llm=AsyncMock(), prune_relation_types=True)
        payload = [
            [{"entity_name": "A", "entity_type": "PERSON"}, {"entity_name": "B", "entity_type": "ORGANIZATION"}],
            [{"entity_name": "C", "entity_type": "PERSON"}, {"entity_name": "D", "entity_type": "DATE"}],
        ]

        rendered = extractor._relation_types_for(payload)

        assert isinstance(rendered, list) and len(rendered) == 2
        assert "WORKPLACE" in rendered[0] and "DATE_OF_BIRTH" not in rendered[0]
        assert "DATE_OF_BIRTH" in rendered[1] and "WORKPLACE" not in rendered[1]
        assert all(len(chunk) < len(extractor.relation_types) for chunk in rendered)

    def test_pruning_falls_back_to_the_full_list_for_an_empty_chunk(self):
        extractor = TwoStageArtifactsExtractorLLM(llm=AsyncMock(), prune_relation_types=True)

        rendered = extractor._relation_types_for([[]])

        assert rendered == [extractor.relation_types]

    async def test_the_signature_legend_flag_reaches_the_prompt(self):
        import ragu.triplet.two_stage_extractor as module

        for signatures in (True, False):
            llm = AsyncMock()
            call_count = 0

            async def _batch(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return [
                    EntitiesExtractionModel(entities=[_entity("Alice", "PERSON")])
                    if call_count == 1
                    else RelationsExtractionModel(relations=[])
                ]

            llm.batch_chat_completion = AsyncMock(side_effect=_batch)
            extractor = TwoStageArtifactsExtractorLLM(llm=llm, show_type_signatures=signatures)
            extractor.get_prompt = MagicMock(
                side_effect=lambda name: _prompt_stub(
                    EntitiesExtractionModel if name.startswith("entity") else RelationsExtractionModel
                )
            )
            patcher, mock_render = _patch_render(module)
            try:
                await extractor.extract([_chunk()])
                relation_stage_call = mock_render.call_args_list[-1]
            finally:
                patcher.stop()

            assert relation_stage_call.kwargs["type_signatures"] is signatures

    async def test_pruned_lists_reach_the_relation_prompt(self):
        import ragu.triplet.two_stage_extractor as module

        entities_model = EntitiesExtractionModel(
            entities=[_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]
        )
        relations_model = RelationsExtractionModel(relations=[])

        llm = AsyncMock()
        call_count = 0

        async def _batch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return [entities_model] if call_count == 1 else [relations_model]

        llm.batch_chat_completion = AsyncMock(side_effect=_batch)
        extractor = TwoStageArtifactsExtractorLLM(
            llm=llm, show_type_signatures=True, prune_relation_types=True
        )
        extractor.get_prompt = MagicMock(
            side_effect=lambda name: _prompt_stub(
                EntitiesExtractionModel if name.startswith("entity") else RelationsExtractionModel
            )
        )

        patcher, mock_render = _patch_render(module)
        try:
            await extractor.extract([_chunk()])
            relation_stage_call = mock_render.call_args_list[-1]
        finally:
            patcher.stop()

        rendered = relation_stage_call.kwargs["relation_types"]
        assert isinstance(rendered, list) and len(rendered) == 1
        assert "WORKPLACE [PERSON -> ORGANIZATION|FACILITY]" in rendered[0]
        assert "DATE_OF_BIRTH" not in rendered[0]

    def test_the_validator_enforces_the_same_ontology_the_prompt_asks_for(self):
        ontology = Ontology.from_type_lists(["LANGUAGE"], ["SPOKEN_IN"])

        extractor = ArtifactsExtractorLLM(llm=AsyncMock(), ontology=ontology)

        assert extractor.validator is not None
        assert extractor.validator.ontology is extractor.ontology
