"""
Ontology-driven validation of extracted artifacts.

Checks performed per chunk:

* entity type outside the vocabulary
* relation type outside the vocabulary
* ``domain``/``range`` violation, repairable by swapping the endpoints
* self-loop
* endpoint that does not resolve to an entity of the same chunk
"""


import difflib
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import List, NamedTuple, Sequence

from ragu.common.logger import logger
from ragu.common.prompts.default_models import EntityModel, RelationModel
from ragu.triplet.ontology.ontology import Ontology, resolve_ontology


class Policy(str, Enum):
    """
    What to do with an artifact that fails a check.

    :cvar COERCE: Repair when possible, drop otherwise.
    :cvar DROP: Discard the artifact.
    :cvar KEEP: Count the violation but let the artifact through.
    :cvar RAISE: Abort extraction with :class:`ArtifactViolationError`.
    """

    COERCE = "coerce"
    DROP = "drop"
    KEEP = "keep"
    RAISE = "raise"


class Violation(str, Enum):
    """
    Checks the validator performs, used as report keys.
    """

    UNKNOWN_ENTITY_TYPE = "unknown_entity_type"
    UNKNOWN_RELATION_TYPE = "unknown_relation_type"
    CONSTRAINT = "constraint_violation"
    SELF_LOOP = "self_loop"
    DANGLING_ENDPOINT = "dangling_endpoint"


class ArtifactViolationError(ValueError):
    """
    Raised when a check fails and its policy is :attr:`Policy.RAISE`.
    """


@dataclass(frozen=True)
class ValidationPolicies:
    """
    What to do about each kind of violation.

    Kept apart from :class:`~ragu.triplet.ontology.Ontology` on purpose: the same
    vocabulary is usually enforced strictly in CI and leniently in production, so
    strictness is a property of the run, not of the domain description.

    :param unknown_entity_type: Entity type outside the vocabulary. Coercion picks
        the closest known name.
    :param unknown_relation_type: Relation type outside the vocabulary. Coercion
        picks the closest known name.
    :param constraint_violation: ``domain``/``range`` violation. Coercion swaps the
        endpoints when the reversed triple is valid.
    :param self_loop: Relation whose endpoints are the same entity. Cannot be
        repaired, so coercion behaves like dropping.
    :param dangling_endpoint: Endpoint that does not resolve to an entity of the
        same chunk. Cannot be repaired either.
    :param fuzzy_cutoff: Similarity required to coerce an unknown type onto a known
        one, between 0 and 1. Raise it to make coercion more conservative; a value
        above 1 disables fuzzy matching entirely.
    """

    unknown_entity_type: Policy = Policy.COERCE
    unknown_relation_type: Policy = Policy.COERCE
    constraint_violation: Policy = Policy.COERCE
    self_loop: Policy = Policy.DROP
    dangling_endpoint: Policy = Policy.DROP
    fuzzy_cutoff: float = 0.85

    def __post_init__(self) -> None:
        """
        Accept plain strings such as ``"keep"`` and turn them into :class:`Policy`.

        Without this a raw string would compare unequal to every policy and the
        artifact would silently take the "drop" path.

        :raises ValueError: If a field does not name a valid policy.
        """
        for name in (
            "unknown_entity_type",
            "unknown_relation_type",
            "constraint_violation",
            "self_loop",
            "dangling_endpoint",
        ):
            object.__setattr__(self, name, Policy(getattr(self, name)))

    @classmethod
    def permissive(cls) -> "ValidationPolicies":
        """
        Count violations without discarding anything.

        Reproduces the behaviour of an extractor that only hints types in the prompt,
        while still reporting how often the model ignored them.

        :return: Policy set where every check is :attr:`Policy.KEEP`.
        """
        return cls(
            unknown_entity_type=Policy.KEEP,
            unknown_relation_type=Policy.KEEP,
            constraint_violation=Policy.KEEP,
            self_loop=Policy.KEEP,
            dangling_endpoint=Policy.KEEP,
        )

    @classmethod
    def strict(cls) -> "ValidationPolicies":
        """
        Discard anything that does not fit the ontology, without guessing.

        :return: Policy set where every check is :attr:`Policy.DROP`.
        """
        return cls(
            unknown_entity_type=Policy.DROP,
            unknown_relation_type=Policy.DROP,
            constraint_violation=Policy.DROP,
        )

    @classmethod
    def failing(cls) -> "ValidationPolicies":
        """
        Turn any violation into an exception, for pipeline tests and CI.

        :return: Policy set where every check is :attr:`Policy.RAISE`.
        """
        return cls(
            unknown_entity_type=Policy.RAISE,
            unknown_relation_type=Policy.RAISE,
            constraint_violation=Policy.RAISE,
            self_loop=Policy.RAISE,
            dangling_endpoint=Policy.RAISE,
        )


@dataclass
class ValidationReport:
    """
    Aggregated outcome of validation.

    Silently discarding artifacts is how a graph ends up sparse for no visible
    reason, so every decision is counted and can be logged or inspected.

    :param entities_in: Entities received.
    :param entities_out: Entities kept.
    :param relations_in: Relations received.
    :param relations_out: Relations kept.
    :param dropped: Number of artifacts dropped, per check.
    :param coerced: Number of artifacts repaired, per check.
    :param kept: Number of violations let through, per check.
    :param dropped_details: What was dropped, aggregated by type signature rather
        than by check, e.g. ``"WORKPLACE (ORGANIZATION -> CITY)"`` or ``"THEME_OF"``.
    :param coerced_details: What was repaired and into what, e.g.
        ``"THEME_OF -> WORKS_AS"`` or ``"AGENT (EVENT -> PERSON) swapped"``.
    :param kept_details: Same signatures, for violations that were let through.
    :param unknown_entity_types: Unresolved entity types and how often they occurred,
        whatever the policy then did with them.
    :param unknown_relation_types: Unresolved relation types and how often they occurred.
    """

    entities_in: int = 0
    entities_out: int = 0
    relations_in: int = 0
    relations_out: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    coerced: Counter[str] = field(default_factory=Counter)
    kept: Counter[str] = field(default_factory=Counter)
    dropped_details: Counter[str] = field(default_factory=Counter)
    coerced_details: Counter[str] = field(default_factory=Counter)
    kept_details: Counter[str] = field(default_factory=Counter)
    unknown_entity_types: Counter[str] = field(default_factory=Counter)
    unknown_relation_types: Counter[str] = field(default_factory=Counter)

    @property
    def is_clean(self) -> bool:
        """
        Whether validation found nothing to report.

        :return: ``True`` if no artifact was dropped, repaired, or flagged.
        """
        return not (self.dropped or self.coerced or self.kept)

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        """
        Accumulate another report into this one.

        :param other: Report to add, typically from the next chunk.
        :return: This report, updated in place.
        """
        self.entities_in += other.entities_in
        self.entities_out += other.entities_out
        self.relations_in += other.relations_in
        self.relations_out += other.relations_out
        self.dropped.update(other.dropped)
        self.coerced.update(other.coerced)
        self.kept.update(other.kept)
        self.dropped_details.update(other.dropped_details)
        self.coerced_details.update(other.coerced_details)
        self.kept_details.update(other.kept_details)
        self.unknown_entity_types.update(other.unknown_entity_types)
        self.unknown_relation_types.update(other.unknown_relation_types)
        return self

    def summary(self, top: int = 5) -> str:
        """
        Render a one-line summary suitable for a log record.

        Counts per check answer "how much was lost"; the detail sections answer
        "what exactly, and what was it turned into".

        :param top: How many distinct signatures to name in each detail section.
        :return: Human-readable summary.
        """
        parts = [
            f"entities {self.entities_in} -> {self.entities_out}",
            f"relations {self.relations_in} -> {self.relations_out}",
        ]
        for label, counter in (
            ("dropped", self.dropped),
            ("coerced", self.coerced),
            ("kept", self.kept),
        ):
            if counter:
                details = ", ".join(f"{key}={value}" for key, value in counter.most_common())
                parts.append(f"{label}: {details}")

        for label, counter in (
            ("top dropped", self.dropped_details),
            ("top coerced", self.coerced_details),
            ("top kept", self.kept_details),
        ):
            if counter:
                details = ", ".join(f"{key}={value}" for key, value in counter.most_common(top))
                suffix = ", ..." if len(counter) > top else ""
                parts.append(f"{label}: {details}{suffix}")

        return "; ".join(parts)

    def breakdown(self) -> str:
        """
        Render every recorded signature, one per line.

        Use when :meth:`summary` is too narrow — for instance when tuning an ontology
        against a corpus and the long tail matters.

        :return: Multi-line report, empty string when nothing was recorded.
        """
        lines: list[str] = []
        for label, counter in (
            ("dropped", self.dropped_details),
            ("coerced", self.coerced_details),
            ("kept", self.kept_details),
        ):
            if not counter:
                continue
            lines.append(f"{label}:")
            lines.extend(
                f"  {count:>4}  {key}" for key, count in counter.most_common()
            )
        return "\n".join(lines)


class ValidationResult(NamedTuple):
    """
    Validated artifacts of a single chunk.

    :param entities: Entities that survived validation.
    :param relations: Relations that survived validation.
    :param report: What happened along the way.
    """

    entities: List[EntityModel]
    relations: List[RelationModel]
    report: ValidationReport


class OntologyValidator:
    """
    Enforce an :class:`Ontology` on extracted artifacts.

    :param ontology: Vocabulary and constraints to enforce, or the name of a
        built-in ontology such as ``"nerel"``.
    :param policies: What to do about each kind of violation.
    """

    def __init__(
        self,
        ontology: Ontology | str,
        policies: ValidationPolicies = ValidationPolicies(),
    ) -> None:
        resolved = resolve_ontology(ontology)
        if resolved is None:
            raise ValueError("OntologyValidator requires an ontology")

        self.ontology = resolved
        self.policies = policies

    def validate(
        self,
        entities: Sequence[EntityModel],
        relations: Sequence[RelationModel],
    ) -> ValidationResult:
        """
        Validate the artifacts of a single chunk.

        Entities are processed first: a relation pointing at an entity that did not
        survive becomes a dangling endpoint, and the constraint check needs the
        final types of both endpoints.

        :param entities: Entities extracted from one chunk.
        :param relations: Relations extracted from the same chunk.
        :return: Surviving artifacts and a report.
        :raises ArtifactViolationError: If a failing check has policy
            :attr:`Policy.RAISE`.
        """
        report = ValidationReport(entities_in=len(entities), relations_in=len(relations))

        kept_entities: List[EntityModel] = []
        for entity in entities:
            checked = self._validate_entity(entity, report)
            if checked is not None:
                kept_entities.append(checked)

        types_by_name = {entity.entity_name: entity.entity_type for entity in kept_entities}

        kept_relations: List[RelationModel] = []
        for relation in relations:
            checked_relation = self._validate_relation(relation, types_by_name, report)
            if checked_relation is not None:
                kept_relations.append(checked_relation)

        report.entities_out = len(kept_entities)
        report.relations_out = len(kept_relations)

        if not report.is_clean:
            logger.debug("Ontology validation: {}", report.summary())

        return ValidationResult(kept_entities, kept_relations, report)

    def validate_batch(
        self,
        entities_per_chunk: Sequence[Sequence[EntityModel]],
        relations_per_chunk: Sequence[Sequence[RelationModel]],
    ) -> tuple[List[List[EntityModel]], List[List[RelationModel]], ValidationReport]:
        """
        Validate a batch of chunks and aggregate their reports.

        :param entities_per_chunk: Entities of each chunk.
        :param relations_per_chunk: Relations of each chunk, aligned with the entities.
        :return: Per-chunk entities, per-chunk relations, and one merged report.
        :raises ArtifactViolationError: If a failing check has policy
            :attr:`Policy.RAISE`.
        """
        all_entities: List[List[EntityModel]] = []
        all_relations: List[List[RelationModel]] = []
        total = ValidationReport()

        for entities, relations in zip(entities_per_chunk, relations_per_chunk):
            result = self.validate(entities, relations)
            all_entities.append(result.entities)
            all_relations.append(result.relations)
            total.merge(result.report)

        return all_entities, all_relations, total

    def _validate_entity(
        self,
        entity: EntityModel,
        report: ValidationReport,
    ) -> EntityModel | None:
        """
        Check one entity against the vocabulary.

        :param entity: Entity to check.
        :param report: Report to update.
        :return: The entity, possibly with a corrected type, or ``None`` to drop it.
        """
        canonical = self.ontology.resolve_entity_type(entity.entity_type)
        if canonical is not None:
            if canonical == entity.entity_type:
                return entity
            return entity.model_copy(update={"entity_type": canonical})

        raw = entity.entity_type or "<empty>"
        report.unknown_entity_types[raw] += 1
        violation = Violation.UNKNOWN_ENTITY_TYPE
        policy = self.policies.unknown_entity_type

        if policy is Policy.RAISE:
            raise ArtifactViolationError(
                f"Entity {entity.entity_name!r} has type {entity.entity_type!r}, "
                f"which is not part of the ontology"
            )
        if policy is Policy.KEEP:
            report.kept[violation.value] += 1
            report.kept_details[raw] += 1
            return entity
        if policy is Policy.COERCE:
            guess = self._closest(entity.entity_type, self.ontology.entity_type_names)
            if guess is not None:
                report.coerced[violation.value] += 1
                report.coerced_details[f"{raw} -> {guess}"] += 1
                return entity.model_copy(update={"entity_type": guess})

        report.dropped[violation.value] += 1
        report.dropped_details[raw] += 1
        return None

    def _validate_relation(
        self,
        relation: RelationModel,
        types_by_name: dict[str, str],
        report: ValidationReport,
    ) -> RelationModel | None:
        """
        Check one relation against the vocabulary and the constraints.

        :param relation: Relation to check.
        :param types_by_name: Entity name to final type, for the surviving entities.
        :param report: Report to update.
        :return: The relation, possibly repaired, or ``None`` to drop it.
        """
        subject, obj = relation.source_entity, relation.target_entity

        if subject not in types_by_name or obj not in types_by_name:
            missing = subject if subject not in types_by_name else obj
            return self._reject(
                relation,
                Violation.DANGLING_ENDPOINT,
                self.policies.dangling_endpoint,
                report,
                f"{relation.relation_type} (dangling endpoint)",
                f"Relation {subject!r} -> {obj!r} has an endpoint ({missing!r}) that is "
                f"not among the entities of this chunk",
            )

        if subject == obj:
            return self._reject(
                relation,
                Violation.SELF_LOOP,
                self.policies.self_loop,
                report,
                f"{relation.relation_type} (self-loop)",
                f"Relation {relation.relation_type!r} is a self-loop on {subject!r}",
            )

        checked = self._check_relation_type(relation, report)
        if checked is None:
            return None

        return self._check_constraints(checked, types_by_name, report)

    def _check_relation_type(
        self,
        relation: RelationModel,
        report: ValidationReport,
    ) -> RelationModel | None:
        """
        Check the predicate against the vocabulary.

        :param relation: Relation to check.
        :param report: Report to update.
        :return: The relation, possibly with a corrected type, or ``None`` to drop it.
        """
        canonical = self.ontology.resolve_relation_type(relation.relation_type)
        if canonical is not None:
            if canonical == relation.relation_type:
                return relation
            return relation.model_copy(update={"relation_type": canonical})

        raw = relation.relation_type or "<empty>"
        report.unknown_relation_types[raw] += 1
        violation = Violation.UNKNOWN_RELATION_TYPE
        policy = self.policies.unknown_relation_type

        # A spelling that means a known predicate reversed, e.g. HAS_PART for PART_OF.
        inverted = self.ontology.resolve_inverse_relation_type(relation.relation_type)
        if inverted is not None and policy is Policy.COERCE:
            report.coerced[violation.value] += 1
            report.coerced_details[f"{raw} -> {inverted} inverted"] += 1
            return relation.model_copy(
                update={
                    "relation_type": inverted,
                    "source_entity": relation.target_entity,
                    "target_entity": relation.source_entity,
                }
            )

        if policy is Policy.RAISE:
            raise ArtifactViolationError(
                f"Relation {relation.source_entity!r} -> {relation.target_entity!r} has type "
                f"{relation.relation_type!r}, which is not part of the ontology"
            )
        if policy is Policy.KEEP:
            report.kept[violation.value] += 1
            report.kept_details[raw] += 1
            return relation
        if policy is Policy.COERCE:
            guess = self._closest(relation.relation_type, self.ontology.relation_type_names)
            if guess is not None:
                report.coerced[violation.value] += 1
                report.coerced_details[f"{raw} -> {guess}"] += 1
                return relation.model_copy(update={"relation_type": guess})

        report.dropped[violation.value] += 1
        report.dropped_details[raw] += 1
        return None

    def _check_constraints(
        self,
        relation: RelationModel,
        types_by_name: dict[str, str],
        report: ValidationReport,
    ) -> RelationModel | None:
        """
        Check the triple against ``domain``/``range``.

        The check is skipped when the predicate or either endpoint type is not part
        of the ontology: those cases were already reported, and rejecting them here
        as well would punish the same artifact twice.

        :param relation: Relation with an already-checked predicate.
        :param types_by_name: Entity name to final type.
        :param report: Report to update.
        :return: The relation, possibly with swapped endpoints, or ``None`` to drop it.
        """
        subject_type = types_by_name[relation.source_entity]
        object_type = types_by_name[relation.target_entity]

        known = (
            self.ontology.resolve_relation_type(relation.relation_type) is not None
            and self.ontology.resolve_entity_type(subject_type) is not None
            and self.ontology.resolve_entity_type(object_type) is not None
        )
        if not known:
            return relation

        if self.ontology.allows(relation.relation_type, subject_type, object_type):
            return relation

        violation = Violation.CONSTRAINT
        policy = self.policies.constraint_violation
        signature = f"{relation.relation_type} ({subject_type} -> {object_type})"
        detail = (
            f"Relation {relation.relation_type!r} does not accept "
            f"{subject_type} -> {object_type}"
        )

        if policy is Policy.RAISE:
            raise ArtifactViolationError(detail)
        if policy is Policy.KEEP:
            report.kept[violation.value] += 1
            report.kept_details[signature] += 1
            return relation

        if policy is Policy.COERCE:
            # A predicate that is right in spirit but wrong for these types.
            retyped = self.ontology.retype(relation.relation_type, subject_type, object_type)
            if retyped is not None:
                update: dict[str, str] = {"relation_type": retyped.to}
                if retyped.swap:
                    update["source_entity"] = relation.target_entity
                    update["target_entity"] = relation.source_entity
                report.coerced[violation.value] += 1
                report.coerced_details[
                    f"{signature} -> {retyped.to}{' swapped' if retyped.swap else ''}"
                ] += 1
                return relation.model_copy(update=update)

            # The endpoints are the wrong way round.
            if self.ontology.allows(relation.relation_type, object_type, subject_type):
                report.coerced[violation.value] += 1
                report.coerced_details[f"{signature} swapped"] += 1
                return relation.model_copy(
                    update={
                        "source_entity": relation.target_entity,
                        "target_entity": relation.source_entity,
                    }
                )

        report.dropped[violation.value] += 1
        report.dropped_details[signature] += 1
        return None

    @staticmethod
    def _reject(
        relation: RelationModel,
        violation: Violation,
        policy: Policy,
        report: ValidationReport,
        signature: str,
        detail: str,
    ) -> RelationModel | None:
        """
        Apply a policy to a violation that cannot be repaired.

        :param relation: Relation under check.
        :param violation: Which check failed.
        :param policy: Policy configured for that check.
        :param report: Report to update.
        :param signature: Aggregation key for the report details.
        :param detail: Message used when raising.
        :return: The relation when kept, ``None`` when dropped.
        :raises ArtifactViolationError: If the policy is :attr:`Policy.RAISE`.
        """
        if policy is Policy.RAISE:
            raise ArtifactViolationError(detail)
        if policy is Policy.KEEP:
            report.kept[violation.value] += 1
            report.kept_details[signature] += 1
            return relation

        report.dropped[violation.value] += 1
        report.dropped_details[signature] += 1
        return None

    def _closest(self, raw: str, candidates: Sequence[str]) -> str | None:
        """
        Find the closest known type name for an unresolved one.

        :param raw: Type as produced by the model.
        :param candidates: Known type names.
        :return: Best match above :attr:`fuzzy_cutoff`, or ``None``.
        """
        if self.policies.fuzzy_cutoff > 1 or not raw:
            return None
        matches = difflib.get_close_matches(
            self.ontology.normalize_type_name(raw),
            list(candidates),
            n=1,
            cutoff=self.policies.fuzzy_cutoff,
        )
        return matches[0] if matches else None
