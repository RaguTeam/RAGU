import pytest

from ragu.common.prompts.default_models import EntityModel, RelationModel
from ragu.triplet.ontology import (
    EntityType,
    Ontology,
    RelationType,
    ArtifactViolationError,
    OntologyValidator,
    Policy,
    ValidationPolicies,
    ValidationReport,
    Violation,
)


def _entity(name, entity_type):
    return EntityModel(entity_name=name, entity_type=entity_type, description="d")


def _relation(source, target, relation_type, strength=3):
    return RelationModel(
        source_entity=source,
        target_entity=target,
        relation_type=relation_type,
        description="d",
        relationship_strength=strength,
    )


@pytest.fixture
def nerel():
    return Ontology.builtin("nerel")


@pytest.fixture
def validator(nerel):
    return OntologyValidator(nerel)


class TestValidEntities:
    def test_known_types_pass_through_untouched(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]

        result = validator.validate(entities, [])

        assert [e.entity_name for e in result.entities] == ["Alice", "Acme"]
        assert result.report.is_clean is True

    def test_alias_and_case_are_canonicalized(self, validator):
        entities = [_entity("Acme", "org"), _entity("Mona Lisa", "work of art")]

        result = validator.validate(entities, [])

        assert [e.entity_type for e in result.entities] == ["ORGANIZATION", "WORK_OF_ART"]
        assert result.report.is_clean is True

    def test_canonicalization_is_not_counted_as_a_violation(self, validator):
        result = validator.validate([_entity("Acme", "ORG")], [])

        assert result.report.coerced == {}
        assert result.report.unknown_entity_types == {}


class TestUnknownEntityType:
    def test_coerced_to_the_closest_known_type(self, validator):
        result = validator.validate([_entity("Acme", "ORGANISATION")], [])

        assert result.entities[0].entity_type == "ORGANIZATION"
        assert result.report.coerced[Violation.UNKNOWN_ENTITY_TYPE.value] == 1
        assert result.report.unknown_entity_types["ORGANISATION"] == 1

    def test_dropped_when_nothing_is_close_enough(self, validator):
        result = validator.validate([_entity("Enterprise", "SPACESHIP")], [])

        assert result.entities == []
        assert result.report.dropped[Violation.UNKNOWN_ENTITY_TYPE.value] == 1

    def test_drop_policy_skips_coercion(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(unknown_entity_type=Policy.DROP))

        result = validator.validate([_entity("Acme", "ORGANISATION")], [])

        assert result.entities == []
        assert result.report.dropped[Violation.UNKNOWN_ENTITY_TYPE.value] == 1

    def test_keep_policy_preserves_the_raw_type(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(unknown_entity_type="keep"))

        result = validator.validate([_entity("Enterprise", "SPACESHIP")], [])

        assert result.entities[0].entity_type == "SPACESHIP"
        assert result.report.kept[Violation.UNKNOWN_ENTITY_TYPE.value] == 1

    def test_raise_policy_aborts(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(unknown_entity_type=Policy.RAISE))

        with pytest.raises(ArtifactViolationError, match="not part of the ontology"):
            validator.validate([_entity("Enterprise", "SPACESHIP")], [])

    def test_fuzzy_cutoff_above_one_disables_coercion(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(fuzzy_cutoff=1.5))

        result = validator.validate([_entity("Acme", "ORGANISATION")], [])

        assert result.entities == []


class TestUnknownRelationType:
    def test_coerced_to_the_closest_predicate(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]
        relations = [_relation("Alice", "Acme", "WORKPLACES")]

        result = validator.validate(entities, relations)

        assert result.relations[0].relation_type == "WORKPLACE"
        assert result.report.coerced[Violation.UNKNOWN_RELATION_TYPE.value] == 1

    def test_dropped_when_unrecognizable(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        relations = [_relation("Alice", "Bob", "COLLABORATES_WITH")]

        result = validator.validate(entities, relations)

        assert result.relations == []
        assert result.report.dropped[Violation.UNKNOWN_RELATION_TYPE.value] == 1
        assert result.report.unknown_relation_types["COLLABORATES_WITH"] == 1

    def test_kept_predicate_skips_the_constraint_check(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(unknown_relation_type=Policy.KEEP))
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]
        relations = [_relation("Alice", "Acme", "COLLABORATES_WITH")]

        result = validator.validate(entities, relations)

        assert result.relations[0].relation_type == "COLLABORATES_WITH"
        assert result.report.dropped == {}


class TestInverseAliases:
    def test_endpoints_are_swapped_along_with_the_rename(self, validator):
        entities = [_entity("Poland", "COUNTRY"), _entity("Warsaw", "CITY")]
        relations = [_relation("Poland", "Warsaw", "CONTAINS")]

        result = validator.validate(entities, relations)

        assert result.relations[0].relation_type == "PART_OF"
        assert result.relations[0].source_entity == "Warsaw"
        assert result.relations[0].target_entity == "Poland"
        assert result.report.coerced_details["CONTAINS -> PART_OF inverted"] == 1

    def test_works_where_the_swap_repair_cannot_help(self, validator):
        """PARENT_OF is PERSON -> PERSON, so a flipped CHILD_OF is invisible to domain/range."""
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]

        result = validator.validate(entities, [_relation("Alice", "Bob", "CHILD_OF")])

        assert result.relations[0].relation_type == "PARENT_OF"
        assert result.relations[0].source_entity == "Bob"
        assert result.relations[0].target_entity == "Alice"

    def test_not_applied_under_a_non_coercing_policy(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(unknown_relation_type=Policy.DROP))
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]

        result = validator.validate(entities, [_relation("Alice", "Bob", "CHILD_OF")])

        assert result.relations == []


class TestNameStripping:
    def test_entity_names_and_endpoints_are_stripped(self, validator):
        entities = [_entity(" Alice ", "PERSON"), _entity("Acme", "ORGANIZATION")]
        relations = [_relation("Alice", " Acme", "WORKPLACE")]

        result = validator.validate(entities, relations)

        assert result.entities[0].entity_name == "Alice"
        assert len(result.relations) == 1


class TestConstraints:
    def test_valid_triple_passes(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]
        relations = [_relation("Alice", "Acme", "WORKPLACE")]

        result = validator.validate(entities, relations)

        assert len(result.relations) == 1
        assert result.report.is_clean is True

    def test_reversed_triple_is_repaired_by_swapping(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]
        relations = [_relation("Acme", "Alice", "WORKPLACE")]

        result = validator.validate(entities, relations)

        assert result.relations[0].source_entity == "Alice"
        assert result.relations[0].target_entity == "Acme"
        assert result.report.coerced[Violation.CONSTRAINT.value] == 1

    def test_swap_preserves_the_rest_of_the_relation(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]
        relations = [_relation("Acme", "Alice", "WORKPLACE", strength=5)]

        result = validator.validate(entities, relations)

        assert result.relations[0].relationship_strength == 5
        assert result.relations[0].relation_type == "WORKPLACE"

    def test_retype_rule_repairs_the_predicate(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY")]
        relations = [_relation("Alice", "Kyiv", "LOCATED_IN")]

        result = validator.validate(entities, relations)

        assert result.relations[0].relation_type == "PLACE_RESIDES_IN"
        assert result.relations[0].source_entity == "Alice"
        assert result.report.coerced_details["LOCATED_IN (PERSON -> CITY) -> PLACE_RESIDES_IN"] == 1

    def test_retype_is_preferred_over_swapping(self, validator):
        """LOCATED_IN would also be valid reversed (CITY is a LOCATION), but retyping wins."""
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY")]

        result = validator.validate(entities, [_relation("Alice", "Kyiv", "LOCATED_IN")])

        assert result.relations[0].source_entity == "Alice"
        assert result.relations[0].relation_type == "PLACE_RESIDES_IN"

    def test_retype_can_rename_and_swap_at_once(self, validator):
        entities = [_entity("Solaris", "WORK_OF_ART"), _entity("Lem", "PERSON")]
        relations = [_relation("Solaris", "Lem", "AGENT")]

        result = validator.validate(entities, relations)

        assert result.relations[0].relation_type == "PRODUCES"
        assert result.relations[0].source_entity == "Lem"
        assert result.relations[0].target_entity == "Solaris"
        assert result.report.coerced_details["AGENT (WORK_OF_ART -> PERSON) -> PRODUCES swapped"] == 1

    def test_widened_predicates_no_longer_violate(self, validator):
        entities = [
            _entity("France", "COUNTRY"), _entity("NATO", "ORGANIZATION"),
            _entity("Lem", "PERSON"), _entity("Solaris", "WORK_OF_ART"),
        ]
        relations = [
            _relation("France", "NATO", "MEMBER_OF"),
            _relation("Lem", "Solaris", "PARTICIPANT_IN"),
        ]

        result = validator.validate(entities, relations)

        assert len(result.relations) == 2
        assert result.report.is_clean is True

    def test_retype_is_skipped_under_drop_policy(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(constraint_violation=Policy.DROP))
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY")]

        result = validator.validate(entities, [_relation("Alice", "Kyiv", "LOCATED_IN")])

        assert result.relations == []

    def test_violation_in_both_directions_is_dropped(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY")]
        relations = [_relation("Alice", "Kyiv", "WORKPLACE")]

        result = validator.validate(entities, relations)

        assert result.relations == []
        assert result.report.dropped[Violation.CONSTRAINT.value] == 1

    def test_subtypes_satisfy_a_supertype_slot(self, validator):
        entities = [_entity("Concert", "EVENT"), _entity("Kyiv", "CITY")]
        relations = [_relation("Concert", "Kyiv", "TAKES_PLACE_IN")]

        result = validator.validate(entities, relations)

        assert len(result.relations) == 1

    def test_keep_policy_lets_the_violation_through(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(constraint_violation=Policy.KEEP))
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY")]
        relations = [_relation("Alice", "Kyiv", "WORKPLACE")]

        result = validator.validate(entities, relations)

        assert len(result.relations) == 1
        assert result.report.kept[Violation.CONSTRAINT.value] == 1

    def test_drop_policy_does_not_swap(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(constraint_violation=Policy.DROP))
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]
        relations = [_relation("Acme", "Alice", "WORKPLACE")]

        result = validator.validate(entities, relations)

        assert result.relations == []

    def test_raise_policy_aborts(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(constraint_violation=Policy.RAISE))
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY")]
        relations = [_relation("Alice", "Kyiv", "WORKPLACE")]

        with pytest.raises(ArtifactViolationError, match="does not accept"):
            validator.validate(entities, relations)

    def test_unconstrained_ontology_accepts_any_direction(self):
        flat = Ontology.from_type_lists(["PERSON", "ORGANIZATION"], ["WORKPLACE"])
        validator = OntologyValidator(flat)
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]

        result = validator.validate(entities, [_relation("Acme", "Alice", "WORKPLACE")])

        assert len(result.relations) == 1
        assert result.report.is_clean is True

    def test_endpoint_with_an_unchecked_type_skips_the_constraint_check(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(unknown_entity_type=Policy.KEEP))
        entities = [_entity("Enterprise", "SPACESHIP"), _entity("Acme", "ORGANIZATION")]
        relations = [_relation("Enterprise", "Acme", "WORKPLACE")]

        result = validator.validate(entities, relations)

        assert len(result.relations) == 1
        assert result.report.dropped == {}


class TestStructuralChecks:
    def test_self_loop_is_dropped(self, validator):
        entities = [_entity("Alice", "PERSON")]
        relations = [_relation("Alice", "Alice", "KNOWS")]

        result = validator.validate(entities, relations)

        assert result.relations == []
        assert result.report.dropped[Violation.SELF_LOOP.value] == 1

    def test_self_loop_can_be_kept(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(self_loop=Policy.KEEP))

        result = validator.validate([_entity("Alice", "PERSON")], [_relation("Alice", "Alice", "KNOWS")])

        assert len(result.relations) == 1
        assert result.report.kept[Violation.SELF_LOOP.value] == 1

    def test_dangling_endpoint_is_dropped(self, validator):
        entities = [_entity("Alice", "PERSON")]
        relations = [_relation("Alice", "Nobody", "KNOWS")]

        result = validator.validate(entities, relations)

        assert result.relations == []
        assert result.report.dropped[Violation.DANGLING_ENDPOINT.value] == 1

    def test_relation_of_a_dropped_entity_becomes_dangling(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Enterprise", "SPACESHIP")]
        relations = [_relation("Alice", "Enterprise", "OWNER_OF")]

        result = validator.validate(entities, relations)

        assert [e.entity_name for e in result.entities] == ["Alice"]
        assert result.relations == []
        assert result.report.dropped[Violation.DANGLING_ENDPOINT.value] == 1

    def test_dangling_endpoint_raises_under_raise_policy(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies(dangling_endpoint=Policy.RAISE))

        with pytest.raises(ArtifactViolationError, match="not among the entities"):
            validator.validate([_entity("Alice", "PERSON")], [_relation("Alice", "Nobody", "KNOWS")])


class TestReport:
    def test_counts_inputs_and_outputs(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Enterprise", "SPACESHIP")]
        relations = [
            _relation("Alice", "Enterprise", "KNOWS"),
            _relation("Alice", "Alice", "KNOWS"),
        ]

        report = validator.validate(entities, relations).report

        assert (report.entities_in, report.entities_out) == (2, 1)
        assert (report.relations_in, report.relations_out) == (2, 0)

    def test_summary_names_the_offending_types(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        relations = [_relation("Alice", "Bob", "COLLABORATES_WITH")]

        summary = validator.validate(entities, relations).report.summary()

        assert "relations 1 -> 0" in summary
        assert "unknown_relation_type=1" in summary
        assert "COLLABORATES_WITH=1" in summary

    def test_dropped_details_name_the_offending_triple(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY")]
        relations = [_relation("Alice", "Kyiv", "WORKPLACE")]

        report = validator.validate(entities, relations).report

        assert report.dropped_details["WORKPLACE (PERSON -> CITY)"] == 1

    def test_dropped_details_name_the_unknown_type(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        relations = [_relation("Alice", "Bob", "COLLABORATES_WITH")]

        report = validator.validate(entities, relations).report

        assert report.dropped_details["COLLABORATES_WITH"] == 1

    def test_coerced_details_show_the_mapping(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANISATION")]
        relations = [_relation("Alice", "Acme", "WORKPLACES")]

        report = validator.validate(entities, relations).report

        assert report.coerced_details["ORGANISATION -> ORGANIZATION"] == 1
        assert report.coerced_details["WORKPLACES -> WORKPLACE"] == 1

    def test_coerced_details_mark_a_swap(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]
        relations = [_relation("Acme", "Alice", "WORKPLACE")]

        report = validator.validate(entities, relations).report

        assert report.coerced_details["WORKPLACE (ORGANIZATION -> PERSON) swapped"] == 1

    def test_structural_details_are_labelled(self, validator):
        entities = [_entity("Alice", "PERSON")]
        relations = [_relation("Alice", "Alice", "KNOWS"), _relation("Alice", "Nobody", "KNOWS")]

        report = validator.validate(entities, relations).report

        assert report.dropped_details["KNOWS (self-loop)"] == 1
        assert report.dropped_details["KNOWS (dangling endpoint)"] == 1

    def test_kept_details_are_recorded(self, nerel):
        validator = OntologyValidator(nerel, ValidationPolicies.permissive())
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY")]
        relations = [_relation("Alice", "Kyiv", "WORKPLACE")]

        report = validator.validate(entities, relations).report

        assert report.kept_details["WORKPLACE (PERSON -> CITY)"] == 1

    def test_summary_shows_what_was_dropped_and_coerced(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY"), _entity("Acme", "ORGANIZATION")]
        relations = [
            _relation("Alice", "Kyiv", "WORKPLACE"),
            _relation("Acme", "Alice", "WORKPLACE"),
        ]

        summary = validator.validate(entities, relations).report.summary()

        assert "top dropped: WORKPLACE (PERSON -> CITY)=1" in summary
        assert "top coerced: WORKPLACE (ORGANIZATION -> PERSON) swapped=1" in summary

    def test_summary_marks_a_truncated_detail_list(self, validator):
        entities = [_entity(f"P{i}", "PERSON") for i in range(7)]
        relations = [_relation(f"P{i}", f"P{i}", f"UNKNOWN_{i}") for i in range(7)]

        summary = validator.validate(entities, relations).report.summary(top=2)

        assert summary.endswith(", ...")

    def test_breakdown_lists_every_signature(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Kyiv", "CITY")]
        relations = [
            _relation("Alice", "Kyiv", "WORKPLACE"),
            _relation("Alice", "Nobody", "KNOWS"),
        ]

        breakdown = validator.validate(entities, relations).report.breakdown()

        assert "dropped:" in breakdown
        assert "WORKPLACE (PERSON -> CITY)" in breakdown
        assert "KNOWS (dangling endpoint)" in breakdown

    def test_breakdown_is_empty_for_a_clean_report(self, validator):
        report = validator.validate([_entity("Alice", "PERSON")], []).report

        assert report.breakdown() == ""

    def test_merge_accumulates_details(self):
        first = ValidationReport()
        first.dropped_details["A (X -> Y)"] += 1
        second = ValidationReport()
        second.dropped_details["A (X -> Y)"] += 2
        second.coerced_details["B -> C"] += 1

        first.merge(second)

        assert first.dropped_details["A (X -> Y)"] == 3
        assert first.coerced_details["B -> C"] == 1

    def test_merge_accumulates(self):
        first = ValidationReport(entities_in=2, entities_out=1)
        first.dropped["a"] += 1
        second = ValidationReport(entities_in=3, entities_out=3)
        second.dropped["a"] += 2
        second.coerced["b"] += 1

        first.merge(second)

        assert (first.entities_in, first.entities_out) == (5, 4)
        assert first.dropped["a"] == 3
        assert first.coerced["b"] == 1

    def test_clean_report_on_valid_input(self, validator):
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")]

        report = validator.validate(entities, [_relation("Alice", "Acme", "WORKPLACE")]).report

        assert report.is_clean is True
        assert "dropped" not in report.summary()


class TestBatch:
    def test_validates_each_chunk_independently(self, validator):
        entities_per_chunk = [
            [_entity("Alice", "PERSON"), _entity("Acme", "ORGANIZATION")],
            [_entity("Bob", "PERSON")],
        ]
        relations_per_chunk = [
            [_relation("Acme", "Alice", "WORKPLACE")],
            [_relation("Bob", "Alice", "KNOWS")],
        ]

        entities, relations, report = validator.validate_batch(
            entities_per_chunk, relations_per_chunk
        )

        assert [len(chunk) for chunk in entities] == [2, 1]
        assert relations[0][0].source_entity == "Alice"
        assert relations[1] == []
        assert report.coerced[Violation.CONSTRAINT.value] == 1
        assert report.dropped[Violation.DANGLING_ENDPOINT.value] == 1

    def test_empty_batch(self, validator):
        entities, relations, report = validator.validate_batch([], [])

        assert entities == []
        assert relations == []
        assert report.is_clean is True


class TestCustomOntology:
    def test_policies_apply_to_a_hand_built_ontology(self):
        ontology = Ontology(
            [EntityType(name="PERSON"), EntityType(name="COMPANY")],
            [RelationType(name="WORKS_AT", domain=("PERSON",), range=("COMPANY",))],
            name="tiny",
        )
        validator = OntologyValidator(ontology)
        entities = [_entity("Alice", "PERSON"), _entity("Acme", "COMPANY")]

        result = validator.validate(entities, [_relation("Acme", "Alice", "WORKS_AT")])

        assert result.relations[0].source_entity == "Alice"
        assert result.report.coerced[Violation.CONSTRAINT.value] == 1
