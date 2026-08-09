from typing import Any, List, Tuple

from ragu.common.logger import logger
from ragu.graph.graph_builder_pipeline import GraphBuilderModule
from ragu.graph.types import Entity, Relation


class RemoveIsolatedNodes(GraphBuilderModule):
    """
    Graph-builder module that removes isolated entities.

    Keeps only relations whose endpoints exist in the entity set, then removes
    entities not connected by any remaining relation.
    """

    def __init__(self) -> None:
        super().__init__()

    async def run(
            self,
            entities: List[Entity],
            relations: List[Relation],
            **kwargs: Any
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Remove dangling relations and entities without edges.

        :param entities: Candidate entities.
        :param relations: Candidate relations.
        :return: Filtered entities and relations.
        """

        entity_ids = {e.id for e in entities if e.id}
        relations = [
            r for r in relations
            if r.subject_id in entity_ids and r.object_id in entity_ids
        ]

        connected_ids = {r.subject_id for r in relations} | {r.object_id for r in relations}
        updated_entities = [e for e in entities if e.id in connected_ids]

        if len(entities) != len(updated_entities):
            logger.info(f"Take only {len(updated_entities)}/{len(entities)} entities after removing isolated nodes")

        return updated_entities, relations