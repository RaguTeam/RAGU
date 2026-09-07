import pytest
import yaml

from ragu.triplet.ontology import (
    EntityType,
    Ontology,
    OntologyError,
    RelationType,
    RetypeRule,
)
from ragu.triplet.types import NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


class TestNormalization:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("person", "PERSON"),
            ("Work of art", "WORK_OF_ART"),
            ("state-or-prov", "STATE_OR_PROV"),
            ("  date_of_birth  ", "DATE_OF_BIRTH"),
            ("PERSON (a human being)", "PERSON"),
            ("", ""),
        ],
    )
    def test_normalize_type_name(self, raw, expected):
        assert Ontology.normalize_type_name(raw) == expected


class TestFromTypeLists:
    def test_parses_legacy_inline_format(self):
        ontology = Ontology.from_type_lists(NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES)

        assert len(ontology.entity_types) == len(NEREL_ENTITY_TYPES)
        assert len(ontology.relation_types) == len(NEREL_RELATION_TYPES)
        assert ontology.entity_types["PERSON"].description.startswith("a human being")

    def test_has_no_constraints(self):
        ontology = Ontology.from_type_lists(NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES)

        assert ontology.has_constraints is False
        assert ontology.allows("WORKPLACE", "ORGANIZATION", "PERSON") is True

    def test_rendering_matches_legacy_joined_string(self):
        ontology = Ontology.from_type_lists(NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES)

        assert ontology.render_entity_types() == ", ".join(NEREL_ENTITY_TYPES)
        assert ontology.render_relation_types() == ", ".join(NEREL_RELATION_TYPES)

    def test_empty_lists_render_as_none(self):
        ontology = Ontology.from_type_lists(None, None)

        assert ontology.render_entity_types() is None
        assert ontology.render_relation_types() is None


class TestBuiltinNerel:
    def test_loads_with_constraints(self):
        ontology = Ontology.builtin("nerel")

        assert len(ontology.entity_types) == len(NEREL_ENTITY_TYPES)
        assert len(ontology.relation_types) == len(NEREL_RELATION_TYPES)
        assert ontology.has_constraints is True

    def test_covers_the_same_type_names_as_the_legacy_lists(self):
        ontology = Ontology.builtin("nerel")
        legacy = Ontology.from_type_lists(NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES)

        assert set(ontology.entity_type_names) == set(legacy.entity_type_names)
        assert set(ontology.relation_type_names) == set(legacy.relation_type_names)

    def test_available_builtins_lists_nerel(self):
        assert "nerel" in Ontology.available_builtins()

    def test_unknown_builtin_reports_alternatives(self):
        with pytest.raises(OntologyError, match="Available: "):
            Ontology.builtin("does-not-exist")


class TestLookup:
    @pytest.fixture
    def ontology(self):
        return Ontology.builtin("nerel")

    @pytest.mark.parametrize("raw", ["ORGANIZATION", "organization", "org", "Company"])
    def test_resolves_names_and_aliases(self, ontology, raw):
        assert ontology.resolve_entity_type(raw) == "ORGANIZATION"

    def test_unknown_type_resolves_to_none(self, ontology):
        assert ontology.resolve_entity_type("SPACESHIP") is None

    def test_subtype_chain(self, ontology):
        assert ontology.ancestors("CITY") == ["CITY", "LOCATION"]
        assert ontology.is_subtype_of("CITY", "LOCATION") is True
        assert ontology.is_subtype_of("LOCATION", "CITY") is False


class TestConstraints:
    @pytest.fixture
    def ontology(self):
        return Ontology.builtin("nerel")

    def test_domain_and_range_are_enforced(self, ontology):
        assert ontology.allows("WORKPLACE", "PERSON", "ORGANIZATION") is True
        assert ontology.allows("WORKPLACE", "ORGANIZATION", "PERSON") is False

    def test_subtypes_satisfy_a_supertype_slot(self, ontology):
        assert ontology.allows("TAKES_PLACE_IN", "EVENT", "CITY") is True
        assert ontology.allows("TAKES_PLACE_IN", "EVENT", "PERSON") is False

    def test_symmetric_predicate_accepts_both_directions(self, ontology):
        assert ontology.allows("SPOUSE", "PERSON", "PERSON") is True
        assert ontology.allows("KNOWS", "PERSON", "ORGANIZATION") is False

    def test_widened_predicates(self, ontology):
        assert ontology.allows("AWARDED_WITH", "WORK_OF_ART", "AWARD") is True
        assert ontology.allows("PRODUCES", "PERSON", "WORK_OF_ART") is True
        assert ontology.allows("MEMBER_OF", "PERSON", "FACILITY") is True

    def test_unconstrained_predicate_accepts_anything(self, ontology):
        assert ontology.allows("PART_OF", "PRODUCT", "MONEY") is True

    def test_unknown_predicate_is_rejected(self, ontology):
        assert ontology.allows("COLLABORATES_WITH", "PERSON", "PERSON") is False

    def test_allowed_relations_prunes_the_vocabulary(self, ontology):
        allowed = ontology.allowed_relations("PERSON", "ORGANIZATION")

        assert "WORKPLACE" in allowed
        assert "DATE_OF_BIRTH" not in allowed
        assert len(allowed) < len(ontology.relation_types)


class TestRetypeRules:
    @pytest.fixture
    def ontology(self):
        return Ontology.builtin("nerel")

    def test_rule_fires_for_the_declared_types(self, ontology):
        assert ontology.retype("LOCATED_IN", "PERSON", "CITY") == ("PLACE_RESIDES_IN", False)
        assert ontology.retype("DATE_OF_CREATION", "PERSON", "DATE") == ("DATE_OF_BIRTH", False)

    def test_swapping_rule_reports_the_swap(self, ontology):
        result = ontology.retype("AGENT", "WORK_OF_ART", "PERSON")

        assert result == ("PRODUCES", True)
        assert result.to == "PRODUCES" and result.swap is True

    def test_swapping_rule_checks_the_target_in_the_swapped_orientation(self):
        """Without the swap the target would reject the triple, so the rule must not fire."""
        ontology = Ontology(
            [EntityType(name="PERSON"), EntityType(name="WORK_OF_ART")],
            [
                RelationType(
                    name="AGENT",
                    domain=("WORK_OF_ART",),
                    retype_when=(RetypeRule(domain=("WORK_OF_ART",), to="MADE", swap=True),),
                ),
                RelationType(name="MADE", domain=("WORK_OF_ART",), range=("PERSON",)),
            ],
        )

        assert ontology.retype("AGENT", "WORK_OF_ART", "PERSON") is None

    def test_rule_does_not_fire_for_other_types(self, ontology):
        assert ontology.retype("LOCATED_IN", "ORGANIZATION", "CITY") is None
        assert ontology.retype("PARENT_OF", "PERSON", "PERSON") is None

    def test_predicate_without_rules(self, ontology):
        assert ontology.retype("WORKPLACE", "ORGANIZATION", "PERSON") is None

    def test_unknown_predicate(self, ontology):
        assert ontology.retype("MADE_UP", "PERSON", "CITY") is None

    def test_rule_is_skipped_when_the_target_would_not_accept_the_triple(self):
        ontology = Ontology(
            [EntityType(name="PERSON"), EntityType(name="CITY")],
            [
                RelationType(
                    name="LIVES",
                    domain=("CITY",),
                    range=("CITY",),
                    retype_when=(RetypeRule(domain=("PERSON",), to="RESIDES"),),
                ),
                RelationType(name="RESIDES", domain=("PERSON",), range=("PERSON",)),
            ],
        )

        assert ontology.retype("LIVES", "PERSON", "CITY") is None

    def test_unknown_target_is_reported(self):
        with pytest.raises(OntologyError, match=r"retype_when\[0\].to: unknown type"):
            Ontology(
                [EntityType(name="PERSON")],
                [RelationType(name="LIVES", retype_when=(RetypeRule(to="NOWHERE"),))],
            )

    def test_self_reference_is_reported(self):
        with pytest.raises(OntologyError, match="into itself"):
            Ontology(
                [EntityType(name="PERSON")],
                [RelationType(name="LIVES", retype_when=(RetypeRule(to="LIVES"),))],
            )

    def test_unknown_filter_type_is_reported(self):
        with pytest.raises(OntologyError, match=r"retype_when\[0\].domain: unknown type"):
            Ontology(
                [EntityType(name="PERSON")],
                [
                    RelationType(name="LIVES", retype_when=(RetypeRule(domain=("ROBOT",), to="RESIDES"),)),
                    RelationType(name="RESIDES"),
                ],
            )


class TestInverseAliases:
    @pytest.fixture
    def ontology(self):
        return Ontology.builtin("nerel")

    @pytest.mark.parametrize(
        "raw, target",
        [("HAS_PART", "PART_OF"), ("CONTAINS", "PART_OF"), ("CHILD_OF", "PARENT_OF"),
         ("WRITTEN_BY", "PRODUCES"), ("OWNED_BY", "OWNER_OF"), ("HAS_MEMBER", "MEMBER_OF")],
    )
    def test_resolves_to_the_inverted_predicate(self, ontology, raw, target):
        assert ontology.resolve_inverse_relation_type(raw) == target

    def test_inverse_aliases_do_not_resolve_as_plain_ones(self, ontology):
        assert ontology.resolve_relation_type("HAS_PART") is None

    def test_plain_aliases_do_not_resolve_as_inverse(self, ontology):
        assert ontology.resolve_inverse_relation_type("SYNONYM") is None

    def test_collision_with_a_name_is_reported(self):
        with pytest.raises(OntologyError, match="already a name or alias"):
            Ontology(
                [EntityType(name="PERSON")],
                [RelationType(name="PART_OF"), RelationType(name="OWNS", inverse_aliases=("PART_OF",))],
            )

    def test_collision_between_two_inverse_aliases_is_reported(self):
        with pytest.raises(OntologyError, match="already inverts"):
            Ontology(
                [EntityType(name="PERSON")],
                [
                    RelationType(name="PART_OF", inverse_aliases=("HAS_PART",)),
                    RelationType(name="MEMBER_OF", inverse_aliases=("HAS_PART",)),
                ],
            )


class TestPromptRendering:
    def test_signatures_are_rendered_for_constrained_predicates(self):
        ontology = Ontology.builtin("nerel")

        rendered = ontology.render_relation_types(with_signatures=True)

        assert "WORKPLACE [PERSON -> ORGANIZATION|FACILITY]" in rendered
        assert "SPOUSE [PERSON <-> PERSON]" in rendered
        assert "PART_OF (links a component" in rendered

    def test_for_pairs_filters_to_applicable_predicates(self):
        ontology = Ontology.builtin("nerel")

        rendered = ontology.render_relation_types(for_pairs=[("PERSON", "ORGANIZATION")])

        assert "WORKPLACE" in rendered
        assert "DATE_OF_BIRTH" not in rendered

    def test_for_pairs_keeps_only_unconstrained_predicates_when_nothing_fits(self):
        ontology = Ontology.builtin("nerel")

        rendered = ontology.render_relation_types(for_pairs=[("MONEY", "PERCENT")])

        assert "PART_OF" in rendered
        assert "WORKPLACE" not in rendered

    def test_empty_ontology_renders_none(self):
        assert Ontology().render_relation_types(for_pairs=[("MONEY", "PERCENT")]) is None


class TestDocumentLoading:
    def test_shorthand_and_full_form_mix(self, tmp_path):
        path = _write(
            tmp_path,
            "mixed.yaml",
            {
                "name": "mixed",
                "entity_types": {
                    "PERSON": "a human being",
                    "ORGANIZATION": {"description": "a company", "aliases": ["ORG"]},
                },
                "relation_types": {
                    "WORKPLACE": {
                        "description": "works for",
                        "domain": ["PERSON"],
                        "range": ["ORGANIZATION"],
                    },
                },
            },
        )

        ontology = Ontology.from_yaml(path)

        assert ontology.entity_types["PERSON"].description == "a human being"
        assert ontology.resolve_entity_type("ORG") == "ORGANIZATION"
        assert ontology.allows("WORKPLACE", "PERSON", "ORGANIZATION") is True

    def test_extends_builtin_and_excludes(self, tmp_path):
        path = _write(
            tmp_path,
            "derived.yaml",
            {
                "name": "derived",
                "extends": "nerel",
                "entity_types": {"SPACECRAFT": "a crewed or robotic spacecraft"},
                "relation_types": {
                    "OPERATES": {
                        "description": "operates a spacecraft",
                        "domain": ["ORGANIZATION"],
                        "range": ["SPACECRAFT"],
                    },
                },
                "exclude": {"entity_types": ["PERCENT"], "relation_types": ["EXPENDITURE"]},
            },
        )

        ontology = Ontology.from_yaml(path)

        assert "SPACECRAFT" in ontology.entity_types
        assert "PERSON" in ontology.entity_types
        assert "PERCENT" not in ontology.entity_types
        assert "EXPENDITURE" not in ontology.relation_types
        assert ontology.allows("OPERATES", "ORGANIZATION", "SPACECRAFT") is True

    def test_local_definition_overrides_inherited_one(self, tmp_path):
        path = _write(
            tmp_path,
            "override.yaml",
            {
                "extends": "nerel",
                "relation_types": {"WORKPLACE": {"description": "loosened", "domain": None}},
            },
        )

        ontology = Ontology.from_yaml(path)

        assert ontology.allows("WORKPLACE", "ORGANIZATION", "ORGANIZATION") is True

    def test_extends_another_file(self, tmp_path):
        _write(tmp_path, "base.yaml", {"entity_types": {"PERSON": "a human"}})
        path = _write(
            tmp_path,
            "child.yaml",
            {"extends": "base.yaml", "entity_types": {"CITY": "a city"}},
        )

        ontology = Ontology.from_yaml(path)

        assert set(ontology.entity_type_names) == {"PERSON", "CITY"}

    def test_inheritance_cycle_is_reported(self, tmp_path):
        _write(tmp_path, "a.yaml", {"extends": "b.yaml", "entity_types": {"PERSON": "x"}})
        _write(tmp_path, "b.yaml", {"extends": "a.yaml", "entity_types": {"CITY": "y"}})

        with pytest.raises(OntologyError, match="cycle"):
            Ontology.from_yaml(tmp_path / "a.yaml")

    def test_missing_file(self, tmp_path):
        with pytest.raises(OntologyError, match="not found"):
            Ontology.from_yaml(tmp_path / "nope.yaml")

    def test_unknown_field_is_rejected(self, tmp_path):
        path = _write(tmp_path, "bad.yaml", {"entity_types": {"PERSON": {"descr": "typo"}}})

        with pytest.raises(OntologyError, match="Malformed ontology document"):
            Ontology.from_yaml(path)


class TestIntegrityChecks:
    def test_unknown_range_reports_suggestion(self):
        with pytest.raises(OntologyError, match=r"unknown type 'ORGANISATION'.*did you mean 'ORGANIZATION'"):
            Ontology(
                [EntityType(name="PERSON"), EntityType(name="ORGANIZATION")],
                [RelationType(name="WORKPLACE", domain=("PERSON",), range=("ORGANISATION",))],
            )

    def test_unknown_parent(self):
        with pytest.raises(OntologyError, match=r"entity_types.CITY.parent: unknown type"):
            Ontology([EntityType(name="CITY", parent="PLACE")], [])

    def test_hierarchy_cycle(self):
        with pytest.raises(OntologyError, match="cycle"):
            Ontology(
                [
                    EntityType(name="A", parent="B"),
                    EntityType(name="B", parent="A"),
                ],
                [],
            )

    def test_symmetric_and_inverse_are_mutually_exclusive(self):
        with pytest.raises(OntologyError, match="mutually exclusive"):
            Ontology(
                [EntityType(name="PERSON")],
                [
                    RelationType(name="SPOUSE", symmetric=True, inverse_of="MARRIED_TO"),
                    RelationType(name="MARRIED_TO"),
                ],
            )

    def test_symmetric_requires_equal_domain_and_range(self):
        with pytest.raises(OntologyError, match="equal domain and range"):
            Ontology(
                [EntityType(name="PERSON"), EntityType(name="ORGANIZATION")],
                [
                    RelationType(
                        name="KNOWS",
                        symmetric=True,
                        domain=("PERSON",),
                        range=("ORGANIZATION",),
                    )
                ],
            )

    def test_inverse_must_point_back(self):
        with pytest.raises(OntologyError, match="declares"):
            Ontology(
                [EntityType(name="PERSON")],
                [
                    RelationType(name="PARENT_OF", inverse_of="CHILD_OF"),
                    RelationType(name="CHILD_OF", inverse_of="KNOWS"),
                    RelationType(name="KNOWS"),
                ],
            )

    def test_inverse_domain_and_range_must_mirror(self):
        with pytest.raises(OntologyError, match="not mirrored"):
            Ontology(
                [EntityType(name="PERSON"), EntityType(name="ORGANIZATION")],
                [
                    RelationType(
                        name="EMPLOYS",
                        domain=("ORGANIZATION",),
                        range=("PERSON",),
                        inverse_of="WORKPLACE",
                    ),
                    RelationType(
                        name="WORKPLACE",
                        domain=("PERSON",),
                        range=("PERSON",),
                        inverse_of="EMPLOYS",
                    ),
                ],
            )

    def test_alias_collision(self):
        with pytest.raises(OntologyError, match="already refers to"):
            Ontology(
                [
                    EntityType(name="ORGANIZATION", aliases=("ORG",)),
                    EntityType(name="ORGAN", aliases=("ORG",)),
                ],
                [],
            )

    def test_all_problems_are_reported_at_once(self):
        with pytest.raises(OntologyError) as error:
            Ontology(
                [EntityType(name="CITY", parent="PLACE")],
                [RelationType(name="LOCATED_IN", domain=("PERSON",), range=("PLACE",))],
            )

        assert error.value.args[0].count("\n  - ") == 3


class TestSerialization:
    def test_roundtrip_preserves_the_ontology(self, tmp_path):
        original = Ontology.builtin("nerel")
        path = tmp_path / "dumped.yaml"
        path.write_text(original.to_yaml(), encoding="utf-8")

        restored = Ontology.from_yaml(path)

        assert restored.entity_types == original.entity_types
        assert restored.relation_types == original.relation_types

    def test_shorthand_is_used_for_plain_types(self):
        ontology = Ontology([EntityType(name="PERSON", description="a human")], [])

        assert ontology.to_mapping()["entity_types"] == {"PERSON": "a human"}
