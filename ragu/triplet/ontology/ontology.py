"""
An ontology describes which entity and relation types an extractor may produce
and, optionally, how they may be combined. The same object covers both levels of
strictness:

* **Type lists only** — every relation type leaves ``domain``/``range`` unset, so
  the only checkable property is "the type belongs to the vocabulary".
* **Full ontology** — relation types declare ``domain``/``range``, entity types
  may declare a ``parent``, predicates may be symmetric or inverse to each other,
  and ``retype_when`` rules rewrite a predicate for specific endpoint types.
"""

import difflib
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ragu.triplet.ontology.types import (
    EntityType,
    OntologyError,
    RelationType,
    RetypeResult,
)

_BUILTIN_ONTOLOGIES_DIR = Path(__file__).parent / "builtin"

_TypeSpec = Union[EntityType, RelationType]

class _ExcludeSection(BaseModel):
    """Types removed from the inherited ontology before local definitions apply."""

    model_config = ConfigDict(extra="forbid")

    entity_types: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()


class _OntologyFile(BaseModel):
    """Schema of an ontology YAML/JSON document."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str = ""
    extends: str | None = None
    entity_types: dict[str, EntityType] = Field(default_factory=dict)
    relation_types: dict[str, RelationType] = Field(default_factory=dict)
    exclude: _ExcludeSection = Field(default_factory=_ExcludeSection)


class Ontology:
    """
    Vocabulary of entity and relation types with optional structural constraints.

    :param entity_types: Entity type specifications; ``name`` must be set.
    :param relation_types: Relation type specifications; ``name`` must be set.
    :param name: Ontology name, used in error messages.
    :raises OntologyError: If the definition is inconsistent (unknown type
        references, cycles in the hierarchy, colliding aliases, and so on).
    """

    @staticmethod
    def normalize_type_name(raw: str) -> str:
        """
        Bring a type name to its canonical form.

        Handles case, separator, and whitespace variance, and tolerates the legacy
        inline format ``"PERSON (a human being)"`` by keeping only the head.

        :param raw: Type name as written by a user or produced by an LLM.
        :return: Canonical ``UPPER_SNAKE_CASE`` name, empty string for blank input.
        """
        head = raw.split(" (", 1)[0]
        return "_".join(head.replace("-", " ").replace("_", " ").upper().split())

    def __init__(
        self,
        entity_types: Iterable[EntityType] = (),
        relation_types: Iterable[RelationType] = (),
        name: str = "",
    ) -> None:
        self.name = name
        self.entity_types: dict[str, EntityType] = {}
        self.relation_types: dict[str, RelationType] = {}

        for entity_spec in entity_types:
            canonical = Ontology.normalize_type_name(entity_spec.name)
            if not canonical:
                raise OntologyError("Entity type with an empty name")
            self.entity_types[canonical] = entity_spec.model_copy(
                update={
                    "name": canonical,
                    "aliases": self._normalize_all(entity_spec.aliases),
                    "parent": self._normalize_optional(entity_spec.parent),
                }
            )

        for relation_spec in relation_types:
            canonical = Ontology.normalize_type_name(relation_spec.name)
            if not canonical:
                raise OntologyError("Relation type with an empty name")
            self.relation_types[canonical] = relation_spec.model_copy(
                update={
                    "name": canonical,
                    "aliases": self._normalize_all(relation_spec.aliases),
                    "inverse_aliases": self._normalize_all(relation_spec.inverse_aliases),
                    "domain": self._normalize_optional_all(relation_spec.domain),
                    "range": self._normalize_optional_all(relation_spec.range),
                    "inverse_of": self._normalize_optional(relation_spec.inverse_of),
                    "retype_when": tuple(
                        rule.model_copy(
                            update={
                                "domain": self._normalize_optional_all(rule.domain),
                                "range": self._normalize_optional_all(rule.range),
                                "to": Ontology.normalize_type_name(rule.to),
                            }
                        )
                        for rule in relation_spec.retype_when
                    ),
                }
            )

        errors: list[str] = []
        self._entity_index = self._build_index(self.entity_types, "entity_types", errors)
        self._relation_index = self._build_index(self.relation_types, "relation_types", errors)
        self._relation_inverse_index = self._build_inverse_index(errors)
        self._check_references(errors)

        if errors:
            scope = f" {self.name!r}" if self.name else ""
            raise OntologyError(
                f"Invalid ontology{scope}:\n  - " + "\n  - ".join(errors)
            )

    @classmethod
    def from_type_lists(
        cls,
        entity_types: Sequence[str] | None,
        relation_types: Sequence[str] | None,
        name: str = "custom",
    ) -> "Ontology":
        """
        Build an unconstrained ontology from plain type lists.

        Accepts the legacy inline format used by :mod:`ragu.triplet.types`,
        where a description follows the name in parentheses.

        :param entity_types: Entity type strings, e.g. ``["PERSON (a human)"]``.
        :param relation_types: Relation type strings.
        :param name: Ontology name.
        :return: Ontology whose predicates carry no ``domain``/``range``.
        """
        entities = [
            EntityType(name=type_name, description=description)
            for type_name, description in map(cls._split_inline, entity_types or ())
        ]
        relations = [
            RelationType(name=type_name, description=description)
            for type_name, description in map(cls._split_inline, relation_types or ())
        ]
        return cls(entities, relations, name=name)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], base_dir: Path | None = None) -> "Ontology":
        """
        Build an ontology from an already-parsed document.

        :param data: Mapping following the ontology file schema.
        :param base_dir: Directory used to resolve a relative ``extends`` path.
        :return: Parsed and validated ontology.
        :raises OntologyError: If the document does not match the schema.
        """
        return cls._from_mapping(data, base_dir=base_dir, seen=frozenset())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Ontology":
        """
        Load an ontology from a YAML (or JSON, which YAML is a superset of) file.

        :param path: Path to the ontology document.
        :return: Parsed and validated ontology.
        :raises OntologyError: If the file is missing or does not match the schema.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise OntologyError(f"Ontology file not found: {file_path}")
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        return cls._from_mapping(
            data or {},
            base_dir=file_path.parent,
            seen=frozenset({file_path.resolve().as_posix()}),
        )

    @classmethod
    def builtin(cls, name: str = "nerel") -> "Ontology":
        """
        Load an ontology shipped with the package.

        :param name: Ontology file name without extension, e.g. ``"nerel"``.
        :return: Parsed and validated ontology.
        :raises OntologyError: If no built-in ontology has that name.
        """
        file_path = _BUILTIN_ONTOLOGIES_DIR / f"{name}.yaml"
        if not file_path.is_file():
            available = ", ".join(cls.available_builtins()) or "none"
            raise OntologyError(
                f"Unknown built-in ontology {name!r}. Available: {available}"
            )
        return cls.from_yaml(file_path)

    @staticmethod
    def available_builtins() -> list[str]:
        """
        List ontologies shipped with the package.

        :return: Sorted names accepted by :meth:`builtin`.
        """
        if not _BUILTIN_ONTOLOGIES_DIR.is_dir():
            return []
        return sorted(path.stem for path in _BUILTIN_ONTOLOGIES_DIR.glob("*.yaml"))

    @property
    def has_constraints(self) -> bool:
        """
        Whether the ontology carries structural constraints beyond type lists.

        :return: ``True`` if any predicate declares ``domain``/``range``, or any
            entity type declares a ``parent``.
        """
        if any(spec.parent for spec in self.entity_types.values()):
            return True
        return any(
            spec.domain is not None or spec.range is not None
            for spec in self.relation_types.values()
        )

    @property
    def entity_type_names(self) -> list[str]:
        """
        Canonical entity type names in declaration order.

        :return: List of names.
        """
        return list(self.entity_types)

    @property
    def relation_type_names(self) -> list[str]:
        """
        Canonical relation type names in declaration order.

        :return: List of names.
        """
        return list(self.relation_types)

    def resolve_entity_type(self, raw: str) -> str | None:
        """
        Map a raw entity type onto a canonical one.

        Resolution is by normalized name first, then by alias.

        :param raw: Type as produced by an LLM, e.g. ``"org"``.
        :return: Canonical name, or ``None`` if the type is not in the ontology.
        """
        return self._entity_index.get(Ontology.normalize_type_name(raw))

    def resolve_relation_type(self, raw: str) -> str | None:
        """
        Map a raw relation type onto a canonical one, keeping its direction.

        :param raw: Predicate as produced by an LLM.
        :return: Canonical name, or ``None`` if the predicate is not in the ontology.
        """
        return self._relation_index.get(Ontology.normalize_type_name(raw))

    def resolve_inverse_relation_type(self, raw: str) -> str | None:
        """
        Map a raw relation type onto the predicate it is the reverse of.

        A caller acting on this must swap the endpoints of the relation.

        :param raw: Predicate as produced by an LLM, e.g. ``"HAS_PART"``.
        :return: Canonical name of the predicate it inverts, or ``None``.
        """
        return self._relation_inverse_index.get(Ontology.normalize_type_name(raw))

    def ancestors(self, entity_type: str) -> list[str]:
        """
        Return the type itself followed by its supertypes, nearest first.

        :param entity_type: Entity type name, canonical or not.
        :return: Chain of canonical names; empty if the type is unknown.
        """
        canonical = self.resolve_entity_type(entity_type)
        chain: list[str] = []
        while canonical is not None and canonical not in chain:
            chain.append(canonical)
            parent = self.entity_types[canonical].parent
            canonical = self._entity_index.get(parent) if parent else None
        return chain

    def is_subtype_of(self, entity_type: str, ancestor: str) -> bool:
        """
        Check whether one entity type is the same as, or a descendant of, another.

        :param entity_type: Candidate subtype.
        :param ancestor: Candidate supertype.
        :return: ``True`` if the hierarchy relation holds.
        """
        canonical_ancestor = self.resolve_entity_type(ancestor)
        if canonical_ancestor is None:
            return False
        return canonical_ancestor in self.ancestors(entity_type)

    def allows(self, relation_type: str, subject_type: str, object_type: str) -> bool:
        """
        Check a candidate triple against ``domain``/``range`` constraints.

        Unconstrained predicates accept any pair. Symmetric predicates also
        accept the reversed pair.

        :param relation_type: Predicate name.
        :param subject_type: Entity type of the subject.
        :param object_type: Entity type of the object.
        :return: ``True`` if the ontology permits the triple.
        """
        canonical = self.resolve_relation_type(relation_type)
        if canonical is None:
            return False

        spec = self.relation_types[canonical]
        if self._fits(subject_type, spec.domain) and self._fits(object_type, spec.range):
            return True
        if spec.symmetric:
            return self._fits(object_type, spec.domain) and self._fits(subject_type, spec.range)
        return False

    def retype(
        self, relation_type: str, subject_type: str, object_type: str
    ) -> RetypeResult | None:
        """
        Find a predicate this triple should be rewritten into.

        Consults the ``retype_when`` rules of the predicate in declaration order and
        returns the first target that both matches the endpoint types and accepts the
        triple — in the swapped orientation when the rule asks for it.

        :param relation_type: Predicate as extracted.
        :param subject_type: Entity type of the subject.
        :param object_type: Entity type of the object.
        :return: Replacement predicate and whether to swap the endpoints, or ``None``.
        """
        canonical = self.resolve_relation_type(relation_type)
        if canonical is None:
            return None

        for rule in self.relation_types[canonical].retype_when:
            if not (self._fits(subject_type, rule.domain) and self._fits(object_type, rule.range)):
                continue
            subject, obj = (object_type, subject_type) if rule.swap else (subject_type, object_type)
            if self.allows(rule.to, subject, obj):
                return RetypeResult(rule.to, rule.swap)
        return None

    def allowed_relations(self, subject_type: str, object_type: str) -> list[str]:
        """
        List predicates admissible between two entity types.

        :param subject_type: Entity type of the subject.
        :param object_type: Entity type of the object.
        :return: Canonical predicate names in declaration order.
        """
        return [
            name
            for name in self.relation_types
            if self.allows(name, subject_type, object_type)
        ]

    def render_entity_types(self) -> str | None:
        """
        Render entity types as a single scalar for prompt templates.

        A scalar is required because :func:`ragu.common.prompts.messages.render`
        treats list values as batch parameters.

        :return: ``"PERSON (a human), CITY (a city)"``, or ``None`` when empty
            so that templates fall back to their unconstrained branch.
        """
        if not self.entity_types:
            return None
        return ", ".join(self._format_type(spec) for spec in self.entity_types.values())

    def render_relation_types(
        self,
        with_signatures: bool = False,
        for_pairs: Iterable[tuple[str, str]] | None = None,
    ) -> str | None:
        """
        Render relation types as a single scalar for prompt templates.

        :param with_signatures: Append ``[DOMAIN -> RANGE]`` to constrained
            predicates so the model sees which pairs they apply to.
        :param for_pairs: Entity type pairs present in the current chunk. When
            given, only predicates admissible for at least one pair are kept,
            which shortens the prompt considerably.
        :return: Rendered string, or ``None`` when nothing is left to render.
        """
        specs = list(self.relation_types.values())

        if for_pairs is not None:
            pairs = list(for_pairs)
            specs = [
                spec
                for spec in specs
                if any(self.allows(spec.name, subject, obj) for subject, obj in pairs)
            ]

        if not specs:
            return None

        return ", ".join(
            self._format_type(spec, signature=self._signature(spec) if with_signatures else "")
            for spec in specs
        )


    def to_mapping(self) -> dict[str, Any]:
        """
        Convert the ontology back into a document mapping.

        Types carrying only a description are emitted in shorthand form.

        :return: Mapping accepted by :meth:`from_mapping`.
        """
        return {
            "version": 1,
            "name": self.name,
            "entity_types": {
                name: self._spec_to_mapping(spec) for name, spec in self.entity_types.items()
            },
            "relation_types": {
                name: self._spec_to_mapping(spec) for name, spec in self.relation_types.items()
            },
        }

    def to_yaml(self) -> str:
        """
        Serialize the ontology as a YAML document.

        :return: YAML text, ready to be written to a file and edited.
        """
        return yaml.safe_dump(
            self.to_mapping(),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    def __repr__(self) -> str:
        return (
            f"Ontology(name={self.name!r}, "
            f"entity_types={len(self.entity_types)}, "
            f"relation_types={len(self.relation_types)}, "
            f"has_constraints={self.has_constraints})"
        )

    @classmethod
    def _from_mapping(
        cls,
        data: Mapping[str, Any],
        base_dir: Path | None,
        seen: frozenset[str],
    ) -> "Ontology":
        """
        Parse a document, resolving ``extends`` and applying ``exclude``.

        :param data: Mapping following the ontology file schema.
        :param base_dir: Directory used to resolve a relative ``extends`` path.
        :param seen: Already-visited sources, used as an inheritance cycle guard.
        :return: Parsed and validated ontology.
        """
        try:
            document = _OntologyFile.model_validate(data)
        except Exception as error:
            raise OntologyError(f"Malformed ontology document: {error}") from error

        entity_types: dict[str, EntityType] = {}
        relation_types: dict[str, RelationType] = {}

        if document.extends:
            base = cls._load_base(document.extends, base_dir, seen)
            entity_types.update(base.entity_types)
            relation_types.update(base.relation_types)

        for excluded in document.exclude.entity_types:
            entity_types.pop(Ontology.normalize_type_name(excluded), None)
        for excluded in document.exclude.relation_types:
            relation_types.pop(Ontology.normalize_type_name(excluded), None)

        for key, entity_spec in document.entity_types.items():
            canonical = Ontology.normalize_type_name(key)
            entity_types[canonical] = entity_spec.model_copy(update={"name": canonical})

        for key, relation_spec in document.relation_types.items():
            canonical = Ontology.normalize_type_name(key)
            relation_types[canonical] = relation_spec.model_copy(update={"name": canonical})

        return cls(
            entity_types.values(),
            relation_types.values(),
            name=document.name or (document.extends or ""),
        )

    @classmethod
    def _load_base(
        cls,
        extends: str,
        base_dir: Path | None,
        seen: frozenset[str],
    ) -> "Ontology":
        """
        Resolve the ``extends`` reference to a built-in name or a file path.

        :param extends: Built-in ontology name or path to another document.
        :param base_dir: Directory used to resolve a relative path.
        :param seen: Already-visited sources, used as a cycle guard.
        :return: Base ontology to inherit from.
        :raises OntologyError: On an inheritance cycle or a missing base.
        """
        if extends.endswith((".yaml", ".yml", ".json")):
            path = Path(extends)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            key = path.resolve().as_posix()
        else:
            path = _BUILTIN_ONTOLOGIES_DIR / f"{extends}.yaml"
            key = extends

        if key in seen:
            raise OntologyError(f"Ontology inheritance cycle at {extends!r}")

        if not path.is_file():
            raise OntologyError(f"Base ontology not found: {extends!r}")

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls._from_mapping(data, base_dir=path.parent, seen=seen | {key})

    @staticmethod
    def _split_inline(raw: str) -> tuple[str, str]:
        """
        Split the legacy ``"NAME (description)"`` format.

        :param raw: Inline type string.
        :return: Pair of name and description; description is empty when absent.
        """
        head, separator, tail = raw.partition(" (")
        if not separator:
            return raw.strip(), ""
        return head.strip(), tail.strip().removesuffix(")").strip()

    @staticmethod
    def _normalize_all(names: tuple[str, ...]) -> tuple[str, ...]:
        """
        Normalize a tuple of type names.

        :param names: Raw names.
        :return: Canonical names, blanks removed.
        """
        return tuple(name for name in map(Ontology.normalize_type_name, names) if name)

    @classmethod
    def _normalize_optional(cls, name: str | None) -> str | None:
        """
        Normalize an optional type reference.

        :param name: Raw name or ``None``.
        :return: Canonical name, or ``None``.
        """
        return Ontology.normalize_type_name(name) or None if name else None

    @classmethod
    def _normalize_optional_all(cls, names: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """
        Normalize an optional constraint list, preserving the "no constraint" state.

        :param names: Raw names, or ``None`` for an unconstrained slot.
        :return: Canonical names, or ``None``.
        """
        return None if names is None else cls._normalize_all(names)

    @staticmethod
    def _build_index(
        specs: Mapping[str, _TypeSpec],
        section: str,
        errors: list[str],
    ) -> dict[str, str]:
        """
        Build a lookup index from canonical names and aliases.

        :param specs: Canonical name to specification.
        :param section: Document section name, used in error messages.
        :param errors: Accumulator for collision messages.
        :return: Mapping from normalized name or alias to canonical name.
        """
        index: dict[str, str] = {}
        for canonical, spec in specs.items():
            index[canonical] = canonical

        for canonical, spec in specs.items():
            for alias in spec.aliases:
                owner = index.get(alias)
                if owner is not None and owner != canonical:
                    errors.append(
                        f"{section}.{canonical}.aliases: {alias!r} already refers to {owner!r}"
                    )
                    continue
                index[alias] = canonical
        return index

    def _build_inverse_index(self, errors: list[str]) -> dict[str, str]:
        """
        Build the lookup for spellings that mean a predicate reversed.

        :param errors: Accumulator for collision messages.
        :return: Mapping from normalized inverse alias to canonical name.
        """
        index: dict[str, str] = {}
        for canonical, spec in self.relation_types.items():
            for alias in spec.inverse_aliases:
                owner = self._relation_index.get(alias)
                if owner is not None:
                    errors.append(
                        f"relation_types.{canonical}.inverse_aliases: {alias!r} is already "
                        f"a name or alias of {owner!r}"
                    )
                    continue
                previous = index.get(alias)
                if previous is not None and previous != canonical:
                    errors.append(
                        f"relation_types.{canonical}.inverse_aliases: {alias!r} already "
                        f"inverts {previous!r}"
                    )
                    continue
                index[alias] = canonical
        return index

    def _check_references(self, errors: list[str]) -> None:
        """
        Verify referential integrity of the ontology.

        :param errors: Accumulator for problem descriptions.
        """
        for name, entity_spec in self.entity_types.items():
            if entity_spec.parent is None:
                continue
            if entity_spec.parent not in self.entity_types:
                errors.append(
                    f"entity_types.{name}.parent: "
                    f"{self._unknown(entity_spec.parent, self.entity_types)}"
                )
            elif self._has_hierarchy_cycle(name):
                errors.append(f"entity_types.{name}.parent: hierarchy cycle detected")

        for name, relation_spec in self.relation_types.items():
            for slot in ("domain", "range"):
                declared: tuple[str, ...] | None = getattr(relation_spec, slot)
                for entity_type in declared or ():
                    if entity_type not in self.entity_types:
                        errors.append(
                            f"relation_types.{name}.{slot}: "
                            f"{self._unknown(entity_type, self.entity_types)}"
                        )

            if relation_spec.symmetric and relation_spec.inverse_of:
                errors.append(
                    f"relation_types.{name}: 'symmetric' and 'inverse_of' are mutually exclusive"
                )

            if (
                relation_spec.symmetric
                and relation_spec.domain is not None
                and relation_spec.range is not None
                and set(relation_spec.domain) != set(relation_spec.range)
            ):
                errors.append(
                    f"relation_types.{name}: symmetric predicate must have equal domain and range"
                )

            self._check_inverse(name, relation_spec, errors)
            self._check_retype_rules(name, relation_spec, errors)

    def _has_hierarchy_cycle(self, entity_type: str) -> bool:
        """
        Detect whether following ``parent`` links leads back to a visited type.

        :param entity_type: Canonical entity type to start from.
        :return: ``True`` if the chain revisits a type.
        """
        visited: set[str] = set()
        current: str | None = entity_type
        while current is not None:
            if current in visited:
                return True
            visited.add(current)
            spec = self.entity_types.get(current)
            parent = spec.parent if spec is not None else None
            current = self._entity_index.get(parent) if parent else None
        return False

    def _check_retype_rules(self, name: str, spec: RelationType, errors: list[str]) -> None:
        """
        Verify that conditional rewrite rules point somewhere sensible.

        :param name: Canonical predicate name.
        :param spec: Predicate specification.
        :param errors: Accumulator for problem descriptions.
        """
        for index, rule in enumerate(spec.retype_when):
            location = f"relation_types.{name}.retype_when[{index}]"

            if rule.to not in self.relation_types:
                errors.append(f"{location}.to: {self._unknown(rule.to, self.relation_types)}")
            elif rule.to == name:
                errors.append(f"{location}.to: a rule cannot rewrite {name!r} into itself")

            for slot in ("domain", "range"):
                for entity_type in getattr(rule, slot) or ():
                    if entity_type not in self.entity_types:
                        errors.append(
                            f"{location}.{slot}: {self._unknown(entity_type, self.entity_types)}"
                        )

    def _check_inverse(self, name: str, spec: RelationType, errors: list[str]) -> None:
        """
        Verify that an ``inverse_of`` declaration is consistent both ways.

        :param name: Canonical predicate name.
        :param spec: Predicate specification.
        :param errors: Accumulator for problem descriptions.
        """
        if spec.inverse_of is None:
            return

        counterpart = self.relation_types.get(spec.inverse_of)
        if counterpart is None:
            errors.append(
                f"relation_types.{name}.inverse_of: "
                f"{self._unknown(spec.inverse_of, self.relation_types)}"
            )
            return

        if counterpart.inverse_of is not None and counterpart.inverse_of != name:
            errors.append(
                f"relation_types.{name}.inverse_of: {counterpart.name!r} declares "
                f"{counterpart.inverse_of!r} as its inverse instead"
            )

        mirrored = (
            spec.domain is not None
            and spec.range is not None
            and counterpart.domain is not None
            and counterpart.range is not None
            and (set(spec.domain) != set(counterpart.range) or set(spec.range) != set(counterpart.domain))
        )
        if mirrored:
            errors.append(
                f"relation_types.{name}.inverse_of: domain/range are not mirrored "
                f"with {counterpart.name!r}"
            )

    @staticmethod
    def _unknown(name: str, known: Mapping[str, _TypeSpec]) -> str:
        """
        Compose an "unknown type" message with a spelling suggestion.

        :param name: Unresolved reference.
        :param known: Available specifications.
        :return: Message body for the error list.
        """
        suggestions = difflib.get_close_matches(name, list(known), n=1)
        hint = f" (did you mean {suggestions[0]!r}?)" if suggestions else ""
        return f"unknown type {name!r}{hint}"

    def _fits(self, entity_type: str, allowed: tuple[str, ...] | None) -> bool:
        """
        Check an entity type against a ``domain``/``range`` slot.

        :param entity_type: Entity type to test.
        :param allowed: Allowed types, or ``None`` for no constraint.
        :return: ``True`` if the slot admits the type.
        """
        if allowed is None:
            return True
        return any(self.is_subtype_of(entity_type, candidate) for candidate in allowed)

    def _signature(self, spec: RelationType) -> str:
        """
        Render the ``[DOMAIN -> RANGE]`` signature of a predicate.

        :param spec: Predicate specification.
        :return: Signature, or an empty string when unconstrained.
        """
        if spec.domain is None and spec.range is None:
            return ""
        domain = "|".join(spec.domain) if spec.domain else "ANY"
        range_ = "|".join(spec.range) if spec.range else "ANY"
        arrow = "<->" if spec.symmetric else "->"
        return f"[{domain} {arrow} {range_}]"

    @staticmethod
    def _format_type(spec: _TypeSpec, signature: str = "") -> str:
        """
        Render a single type for a prompt.

        :param spec: Type specification.
        :param signature: Optional signature inserted before the description.
        :return: ``"NAME [SIGNATURE] (description)"`` with empty parts dropped.
        """
        parts = [spec.name]
        if signature:
            parts.append(signature)
        if spec.description:
            parts.append(f"({spec.description})")
        return " ".join(parts)

    @staticmethod
    def _spec_to_mapping(spec: _TypeSpec) -> Any:
        """
        Convert a specification back to its document form.

        :param spec: Type specification.
        :return: Description string when nothing else is set, else a mapping.
        """
        payload = spec.model_dump(exclude_defaults=True, exclude={"name"})
        if set(payload) <= {"description"}:
            return spec.description
        payload["description"] = spec.description
        return {key: Ontology._jsonify(value) for key, value in payload.items()}

    @staticmethod
    def _jsonify(value: Any) -> Any:
        """
        Turn tuples into lists recursively so the result is YAML-serializable.

        :param value: Dumped field value.
        :return: The same value with every tuple replaced by a list.
        """
        if isinstance(value, (tuple, list)):
            return [Ontology._jsonify(item) for item in value]
        if isinstance(value, dict):
            return {key: Ontology._jsonify(item) for key, item in value.items()}
        return value


@lru_cache(maxsize=None)
def builtin_ontology(name: str) -> Ontology:
    """
    Load a built-in ontology once and reuse it.

    The returned object is shared between callers, so treat it as read-only.

    :param name: Built-in ontology name, e.g. ``"nerel"``.
    :return: Cached ontology.
    :raises OntologyError: If no built-in ontology has that name.
    """
    return Ontology.builtin(name)


def resolve_ontology(ontology: "Ontology | str | None") -> Ontology | None:
    """
    Normalize the many ways an ontology can be supplied to a component.

    :param ontology: An :class:`Ontology`, the name of a built-in one, or ``None``
        to work without a vocabulary at all.
    :return: The ontology, or ``None``.
    :raises OntologyError: If a name is given that no built-in ontology has.
    """
    if ontology is None:
        return None
    if isinstance(ontology, str):
        return builtin_ontology(ontology)
    return ontology
