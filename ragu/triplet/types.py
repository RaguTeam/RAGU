"""
Legacy type-list constants.

``NEREL_ENTITY_TYPES`` and ``NEREL_RELATION_TYPES`` are derived from
``ragu/triplet/ontology/builtin/nerel.yaml``, which is the single source of truth for
the NEREL vocabulary. They are kept in the historical ``"NAME (description)"``
inline format for code that still expects plain lists — new code should take an
:class:`~ragu.triplet.ontology.Ontology` instead, via
``Ontology.builtin("nerel")`` or simply ``ontology="nerel"``.
"""

from ragu.triplet.ontology import builtin_ontology

DEFAULT_ENTITY_TYPES = [
    "PERSON (a human being, including full names and nicknames)",
    "ORGANIZATION (a company, agency, political party, or other organized group)",
    "LOCATION (a geographic location, natural or physical, not otherwise classified)",
    "DATE (a calendar date, absolute or relative)"
]


def _inline(name: str, description: str) -> str:
    """
    Render a type in the legacy inline format.

    :param name: Canonical type name.
    :param description: Type description, may be empty.
    :return: ``"NAME (description)"``, or just the name when there is no description.
    """
    return f"{name} ({description})" if description else name


NEREL_ENTITY_TYPES = [
    _inline(spec.name, spec.description)
    for spec in builtin_ontology("nerel").entity_types.values()
]

NEREL_RELATION_TYPES = [
    _inline(spec.name, spec.description)
    for spec in builtin_ontology("nerel").relation_types.values()
]
