import pytest

from ragu.graph.types import (
    Entity,
    Relation,
    Community,
    CommunitySummary
)


class TestEntity:
    """Tests for Entity dataclass."""

    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            entity_name="John Doe",
            entity_type="PERSON",
            description="A software engineer",
            source_chunk_id=["chunk-1"],
            documents_id=["doc-1"]
        )

        assert entity.entity_name == "John Doe"
        assert entity.entity_type == "PERSON"
        assert entity.description == "A software engineer"
        assert entity.source_chunk_id == ["chunk-1"]
        assert entity.documents_id == ["doc-1"]

    def test_entity_id_generation(self):
        """Test that entity ID is auto-generated."""
        entity = Entity(
            entity_name="Jane Smith",
            entity_type="PERSON",
            description="Description",
            source_chunk_id=["chunk-1"]
        )

        assert entity.id is not None
        assert isinstance(entity.id, str)
        assert entity.id.startswith("ent-")
        assert len(entity.id) > 4  # "ent-" + hash

    def test_entity_id_deterministic(self):
        entity1 = Entity(
            entity_name="Company X",
            entity_type="ORGANIZATION",
            description="A company",
            source_chunk_id=["chunk-1"]
        )

        entity2 = Entity(
            entity_name="Company X",
            entity_type="ORGANIZATION",
            description="Different description",
            source_chunk_id=["chunk-2"]
        )

        assert entity1.id == entity2.id

    def test_entity_id_unique_per_name_type(self):
        entity1 = Entity(
            entity_name="Name1",
            entity_type="PERSON",
            description="Desc",
            source_chunk_id=["chunk-1"]
        )

        entity2 = Entity(
            entity_name="Name2",
            entity_type="PERSON",
            description="Desc",
            source_chunk_id=["chunk-1"]
        )

        entity3 = Entity(
            entity_name="Name1",
            entity_type="ORGANIZATION",
            description="Desc",
            source_chunk_id=["chunk-1"]
        )

        assert entity1.id != entity2.id  # Different names
        assert entity1.id != entity3.id  # Different types

    def test_entity_default_values(self):
        entity = Entity(
            entity_name="Test",
            entity_type="TYPE",
            description="Desc",
            source_chunk_id=["chunk-1"]
        )

        assert entity.documents_id == []
        assert entity.clusters == []

class TestRelation:
    def test_relation_creation(self):
        relation = Relation(
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Alice knows Bob"
        )

        assert relation.subject_id == "ent-1"
        assert relation.object_id == "ent-2"
        assert relation.subject_name == "Alice"
        assert relation.object_name == "Bob"
        assert relation.relation_type == "KNOWS"
        assert relation.description == "Alice knows Bob"

    def test_relation_id_generation(self):
        relation = Relation(
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="Subject",
            object_name="Object",
            relation_type="RELATED_TO",
            description="Description"
        )

        assert relation.id is not None
        assert isinstance(relation.id, str)
        assert relation.id.startswith("rel-")

    def test_relation_id_deterministic(self):
        relation1 = Relation(
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="A",
            object_name="B",
            relation_type="RELATED_TO",
            description="Desc1"
        )

        relation2 = Relation(
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="A",
            object_name="B",
            relation_type="RELATED_TO",
            description="Desc2"
        )

        assert relation1.id == relation2.id

    def test_relation_id_unique_per_entities(self):
        relation1 = Relation(
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="A",
            object_name="B",
            relation_type="RELATED_TO",
            description="Desc"
        )

        relation2 = Relation(
            subject_id="ent-1",
            object_id="ent-3",
            subject_name="A",
            object_name="C",
            relation_type="RELATED_TO",
            description="Desc"
        )

        assert relation1.id != relation2.id

    def test_relation_default_strength(self):
        relation = Relation(
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="A",
            object_name="B",
            relation_type="RELATED_TO",
            description="Desc"
        )

        assert relation.relation_strength == 1.0

    def test_relation_directional(self):
        relation1 = Relation(
            subject_id="ent-1",
            object_id="ent-2",
            subject_name="A",
            object_name="B",
            relation_type="RELATED_TO",
            description="A -> B"
        )

        relation2 = Relation(
            subject_id="ent-2",
            object_id="ent-1",
            subject_name="B",
            object_name="A",
            relation_type="RELATED_TO",
            description="B -> A"
        )

        # Different directions should have different IDs
        assert relation1.id != relation2.id


class TestCommunity:
    def test_community_creation(self):
        entities = [
            Entity("E1", "TYPE", "Desc", ["chunk-1"]),
            Entity("E2", "TYPE", "Desc", ["chunk-2"])
        ]

        relations = [
            Relation("ent-1", "ent-2", "E1", "E2", "RELATED_TO", "E1->E2")
        ]

        community = Community(
            level=0,
            cluster_id=1,
            entities=entities,
            relations=relations
        )

        assert community.level == 0
        assert community.cluster_id == 1
        assert len(community.entities) == 2
        assert len(community.relations) == 1

    def test_community_id_generation(self):
        community = Community(
            level=1,
            cluster_id=5,
            entities=[],
            relations=[]
        )

        assert community.id is not None
        assert isinstance(community.id, str)
        assert community.id.startswith("com-")

    def test_community_id_deterministic(self):
        community1 = Community(
            level=0,
            cluster_id=1,
            entities=[],
            relations=[]
        )

        community2 = Community(
            level=0,
            cluster_id=1,
            entities=[],
            relations=[]
        )

        assert community1.id == community2.id

    # def test_community_id_unique_per_level_cluster(self):
    #     community1 = Community(level=0, cluster_id=1, entities=[], relations=[])
    #     community2 = Community(level=0, cluster_id=2, entities=[], relations=[])
    #     community3 = Community(level=1, cluster_id=1, entities=[], relations=[])
    #
    #     assert community1.id != community2.id  # Different cluster
    #     assert community1.id != community3.id  # Different level

    def test_community_with_custom_id(self):
        custom_id = "custom-community-id"
        community = Community(
            level=0,
            cluster_id=1,
            entities=[],
            relations=[],
            id=custom_id
        )

        assert community.id == custom_id

    def test_community_empty_entities_relations(self):
        community = Community(
            level=0,
            cluster_id=1,
            entities=[],
            relations=[]
        )

        assert len(community.entities) == 0
        assert len(community.relations) == 0


class TestCommunitySummary:
    def test_community_summary_creation(self):
        summary = CommunitySummary(
            id="com-123",
            summary="This community contains entities related to technology."
        )

        assert summary.id == "com-123"
        assert "technology" in summary.summary

    def test_community_summary_unicode(self):
        summary = CommunitySummary(
            id="com-101",
            summary="Summary with unicode: 你好 Привет"
        )

        assert "你好" in summary.summary
        assert "Привет" in summary.summary


class TestDataclassInteractions:
    def test_entity_in_community(self):
        entity1 = Entity("E1", "PERSON", "Person 1", ["chunk-1"])
        entity2 = Entity("E2", "PERSON", "Person 2", ["chunk-2"])

        community = Community(
            level=0,
            cluster_id=1,
            entities=[entity1, entity2],
            relations=[]
        )

        assert entity1 in community.entities
        assert entity2 in community.entities

    def test_relation_between_entities(self):
        entity1 = Entity("Alice", "PERSON", "Desc", ["chunk-1"])
        entity2 = Entity("Bob", "PERSON", "Desc", ["chunk-2"])

        relation = Relation(
            subject_id=entity1.id,
            object_id=entity2.id,
            subject_name="Alice",
            object_name="Bob",
            relation_type="KNOWS",
            description="Alice knows Bob"
        )

        assert relation.subject_id == entity1.id
        assert relation.object_id == entity2.id


class TestTypeContracts:
    """Equality and hashing contracts for graph types."""

    def _entity(self, name="Alice", description="desc"):
        return Entity(
            entity_name=name,
            entity_type="PERSON",
            description=description,
            source_chunk_id=["chunk-1"],
        )

    def test_entity_eq_against_foreign_type_is_false(self):
        assert self._entity() != "not an entity"
        assert (self._entity() == 42) is False

    def test_entity_is_hashable_and_consistent_with_eq(self):
        e1 = self._entity()
        e2 = self._entity()
        assert e1 == e2
        assert hash(e1) == hash(e2)
        assert len({e1, e2}) == 1

    def test_relation_eq_against_foreign_type_is_false(self):
        e1, e2 = self._entity("A"), self._entity("B")
        relation = Relation(
            subject_id=e1.id,
            object_id=e2.id,
            subject_name="A",
            object_name="B",
            relation_type="KNOWS",
            description="A knows B",
        )
        assert relation != "not a relation"
        assert len({relation, relation}) == 1

    def test_community_is_hashable(self):
        community = Community(level=0, cluster_id=1, entities=[], relations=[])
        same = Community(level=0, cluster_id=1, entities=[], relations=[])
        assert community == same
        assert len({community, same}) == 1
        assert community != "not a community"

    def test_explicit_empty_id_rejected_by_contract(self):
        with pytest.raises(ValueError, match="Entity id"):
            Entity(
                entity_name="X",
                entity_type="PERSON",
                description="d",
                source_chunk_id=[],
                id="",
            )
        with pytest.raises(ValueError, match="Relation id"):
            Relation(
                subject_id="s",
                object_id="o",
                subject_name="S",
                object_name="O",
                relation_type="KNOWS",
                description="d",
                id="",
            )
        with pytest.raises(ValueError, match="Community id"):
            Community(level=0, cluster_id=1, entities=[], relations=[], id="")
        with pytest.raises(ValueError, match="CommunitySummary id"):
            CommunitySummary(id="", summary="s")
