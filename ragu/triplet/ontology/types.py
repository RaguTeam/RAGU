"""
Building blocks of an ontology: the entity and relation type specifications.

These are pydantic models on purpose -- they double as the schema of an ontology
YAML document, so a malformed file is rejected at load time with a field-level
message instead of failing later during extraction.
"""

from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, model_validator


class OntologyError(ValueError):
    """Raised when an ontology definition is malformed or internally inconsistent."""


class _ShortForm(BaseModel):
    """Base model accepting ``NAME: description`` shorthand instead of a mapping."""

    @model_validator(mode="before")
    @classmethod
    def _expand_shorthand(cls, value: Any) -> Any:
        """
        Expand the scalar shorthand into a full specification mapping.

        :param value: Raw YAML/JSON value for a single type.
        :return: Mapping suitable for field validation.
        """
        if isinstance(value, str):
            return {"description": value}
        if value is None:
            return {}
        return value


class EntityType(_ShortForm):
    """
    Single entity type of an ontology.

    :param name: Canonical type name, e.g. ``"PERSON"``.
    :param description: Human-readable meaning, injected into extraction prompts.
    :param aliases: Alternative spellings coerced to this type, e.g. ``("ORG",)``.
    :param parent: Supertype name; a subtype satisfies any constraint that
        accepts its ancestors.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = ""
    description: str = ""
    aliases: tuple[str, ...] = ()
    parent: str | None = None


class RetypeRule(BaseModel):
    """
    Conditional rewrite of a predicate, keyed on the types of its endpoints.

    Models routinely pick a predicate that is right in spirit but wrong for the
    entities at hand — ``PERSON -LOCATED_IN-> CITY`` when the ontology calls that
    ``PLACE_RESIDES_IN``. An alias cannot express this, because an alias is
    unconditional and would also rewrite the legitimate
    ``ORGANIZATION -LOCATED_IN-> CITY``.

    :param domain: Subject types that trigger the rule; ``None`` matches any.
        Matched against the relation as extracted, before any swap.
    :param range: Object types that trigger the rule; ``None`` matches any.
    :param to: Predicate to rewrite into. The rule only applies if the rewritten
        triple actually satisfies that predicate's constraints.
    :param swap: Whether the endpoints must be exchanged along with the rename, as
        in ``WORK_OF_ART -AGENT-> PERSON`` becoming ``PERSON -PRODUCES-> WORK_OF_ART``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: tuple[str, ...] | None = None
    range: tuple[str, ...] | None = None
    to: str
    swap: bool = False


class RetypeResult(NamedTuple):
    """
    Outcome of :meth:`Ontology.retype`.

    :param to: Canonical name of the replacement predicate.
    :param swap: Whether the endpoints must be exchanged as well.
    """

    to: str
    swap: bool


class RelationType(_ShortForm):
    """
    Single relation type (predicate) of an ontology.

    :param name: Canonical predicate name, e.g. ``"WORKPLACE"``.
    :param description: Human-readable meaning, injected into extraction prompts.
    :param aliases: Alternative spellings coerced to this predicate, keeping the
        direction of the extracted relation.
    :param inverse_aliases: Spellings that mean this predicate *reversed*, such as
        ``HAS_PART`` for ``PART_OF``. Resolving one of these swaps the endpoints as
        well. Needed because a plain alias is direction-blind, and the endpoint-swap
        repair cannot detect a flipped relation when ``domain`` equals ``range`` or
        the predicate is unconstrained.
    :param domain: Entity types allowed as the subject; ``None`` means any type.
    :param range: Entity types allowed as the object; ``None`` means any type.
    :param symmetric: Whether the predicate holds in both directions.
    :param inverse_of: Name of the predicate that expresses the reverse relation.
    :param functional: Whether a subject may have at most one object for this
        predicate (checked after graph merging, not per chunk).
    :param retype_when: Rules rewriting this predicate into another one for
        specific endpoint types, tried in order.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = ""
    description: str = ""
    aliases: tuple[str, ...] = ()
    inverse_aliases: tuple[str, ...] = ()
    domain: tuple[str, ...] | None = None
    range: tuple[str, ...] | None = None
    symmetric: bool = False
    inverse_of: str | None = None
    functional: bool = False
    retype_when: tuple[RetypeRule, ...] = ()


