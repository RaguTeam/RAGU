from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import asdict
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import override

from ragu.storage.base_storage import BaseGraphStorage, EdgeSpec
from ragu.storage.types import Node, Edge

if TYPE_CHECKING:
    from neo4j import AsyncDriver

NodeT = TypeVar("NodeT", bound=Node)
EdgeT = TypeVar("EdgeT", bound=Edge)

#: Property types Neo4j stores natively; anything else is serialized to JSON.
_PRIMITIVES = (str, int, float, bool)

#: Label applied to every node this adapter writes, used for lookups.
_NODE_LABEL = "NODE"

#: Relationship type applied to every edge this adapter writes.
_EDGE_TYPE = "RELATION"


class Neo4jStorage(BaseGraphStorage[NodeT, EdgeT]):
    """
    Neo4j-backed implementation of :class:`BaseGraphStorage`.

    Nodes are written with a shared ``:NODE`` label plus a per-type label
    derived from the node itself, so Cypher lookups stay uniform while the
    graph remains readable in Neo4j Browser.

    Requires the optional ``neo4j`` driver: ``pip install graph_ragu[neo4j]``.

    :param uri: Bolt URI of the Neo4j server.
    :type uri: str
    :param user: Username for authentication.
    :type user: str
    :param password: Password for authentication.
    :type password: str
    :param node_cls: Dataclass used to materialize nodes.
    :type node_cls: Type[NodeT]
    :param edge_cls: Dataclass used to materialize edges.
    :type edge_cls: Type[EdgeT]
    :param database: Target database name.
    :type database: str
    :raises ImportError: If the ``neo4j`` driver is not installed.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        node_cls: Type[NodeT],
        edge_cls: Type[EdgeT],
        database: str = "neo4j",
        **kwargs: Any,
    ):
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise ImportError(
                "Neo4jStorage requires the optional 'neo4j' driver. "
                "Install it with: pip install graph_ragu[neo4j]"
            ) from exc

        self._driver: "AsyncDriver" = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._node_cls = node_cls
        self._edge_cls = edge_cls
        self._node_json_fields = self._json_fields_for(node_cls)
        self._edge_json_fields = self._json_fields_for(edge_cls)

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """
        Make an arbitrary string safe to interpolate as a Cypher label.

        :param label: Raw label, typically a node type.
        :type label: str
        :returns: Label containing only word characters and not starting with a digit.
        :rtype: str
        """
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", label)
        if not sanitized or sanitized[0].isdigit():
            sanitized = f"_{sanitized}"
        return sanitized

    @staticmethod
    def _label_for(node: Node) -> str:
        """
        Choose the per-type Neo4j label for a node.

        Deriving the label here keeps Neo4j's storage concerns out of the shared
        domain types, which several other backends also persist.

        :param node: Node about to be written.
        :type node: Node
        :returns: Label name, falling back to the generic node label.
        :rtype: str
        """
        return getattr(node, "entity_type", None) or _NODE_LABEL

    @staticmethod
    def _serialize_props(props: dict) -> dict:
        """
        Convert dataclass attributes into Neo4j-storable property values.

        Values Neo4j cannot hold natively (nested lists, dicts) become JSON strings.

        :param props: Attribute mapping.
        :type props: dict
        :returns: Property mapping safe to pass to Cypher.
        :rtype: dict
        """
        result = {}
        for key, value in props.items():
            if isinstance(value, _PRIMITIVES) or value is None:
                result[key] = value
            elif isinstance(value, list) and all(isinstance(x, _PRIMITIVES) for x in value):
                result[key] = value
            else:
                result[key] = json.dumps(value, ensure_ascii=False)
        return result

    @staticmethod
    def _json_fields_for(record_cls: type) -> tuple[str, ...]:
        """
        Determine which fields round-trip through JSON, from their declared types.

        Reading this off the dataclass annotations avoids asking domain types to
        describe how one particular backend stores them: a field is JSON-encoded
        exactly when Neo4j cannot hold its value natively.

        :param record_cls: Node or edge dataclass.
        :type record_cls: type
        :returns: Names of fields stored as JSON strings.
        :rtype: tuple[str, ...]
        """
        if not dataclasses.is_dataclass(record_cls):
            return ()
        try:
            hints = get_type_hints(record_cls)
        except Exception:  # unresolvable forward references
            return ()

        fields: List[str] = []
        for field in dataclasses.fields(record_cls):
            hint = hints.get(field.name)
            origin, args = get_origin(hint), get_args(hint)
            if origin is dict:
                fields.append(field.name)
            elif origin in (list, set, tuple) and args and args[0] not in _PRIMITIVES:
                fields.append(field.name)
        return tuple(fields)

    @staticmethod
    def _deserialize_props(props: dict, json_fields: tuple[str, ...]) -> dict:
        """
        Decode JSON-encoded properties back into Python values.

        :param props: Properties as read from Neo4j.
        :type props: dict
        :param json_fields: Fields written as JSON strings.
        :type json_fields: tuple[str, ...]
        :returns: Properties with structured fields restored.
        :rtype: dict
        """
        for field in json_fields:
            raw = props.get(field)
            if isinstance(raw, str):
                try:
                    props[field] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    pass
        return props

    async def _verify_connectivity(self) -> None:
        async with self._driver.session(database=self._database) as session:
            await session.run("RETURN 1")

    async def index_done_callback(self) -> None:
        pass

    async def query_done_callback(self) -> None:
        pass

    async def index_start_callback(self) -> None:
        await self._verify_connectivity()

    async def get_node_edges(self, source_node_id: str) -> List[EdgeT]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (n:NODE {id: $source_id})-[r]->(m:NODE)
                RETURN n, r, m
                """,
                source_id=source_node_id,
            )
            edges: List[EdgeT] = []
            async for record in result:
                rel = record["r"]
                sn = record["n"]
                mn = record["m"]
                props = dict(rel)
                edge_id = props.pop("id", None) or rel.element_id
                edges.append(
                    self._edge_cls(
                        subject_id=sn.get("id", sn.element_id),
                        object_id=mn.get("id", mn.element_id),
                        id=edge_id,
                        **props,
                    )
                )
            return edges

    @override
    async def edges_degrees(self, edge_specs: List[EdgeSpec]) -> List[int]:
        degrees: List[int] = []
        async with self._driver.session(database=self._database) as session:
            for subject_id, object_id, _edge_id in edge_specs:
                s_result = await session.run(
                    "MATCH (s:NODE {id: $sid})-[r]-() RETURN count(r) AS d",
                    sid=subject_id,
                )
                s_record = await s_result.single()
                o_result = await session.run(
                    "MATCH (o:NODE {id: $oid})-[r2]-() RETURN count(r2) AS d",
                    oid=object_id,
                )
                o_record = await o_result.single()
                total = (s_record["d"] if s_record else 0) + (o_record["d"] if o_record else 0)
                degrees.append(total)
        return degrees

    @override
    async def upsert_nodes(self, nodes: Iterable[NodeT]) -> None:
        async with self._driver.session(database=self._database) as session:
            for node in nodes:
                attrs = asdict(node)
                node_id = attrs.pop("id")
                props = self._serialize_props(attrs)
                label = self._sanitize_label(self._label_for(node))
                await session.run(
                    f"MERGE (n:NODE:{label} {{id: $id}}) SET n += $props",
                    id=node_id,
                    props=props,
                )

    @override
    async def get_nodes(self, node_ids: List[str]) -> List[Optional[NodeT]]:
        results: List[Optional[NodeT]] = []
        async with self._driver.session(database=self._database) as session:
            for node_id in node_ids:
                result = await session.run(
                    "MATCH (n:NODE {id: $id}) RETURN n",
                    id=node_id,
                )
                record = await result.single()
                if record is None:
                    results.append(None)
                else:
                    props = dict(record["n"])
                    node_id_val = props.pop("id")
                    props = self._deserialize_props(props, self._node_json_fields)
                    results.append(self._node_cls(id=node_id_val, **props))
        return results

    @override
    async def delete_nodes(self, node_ids: List[str]) -> None:
        async with self._driver.session(database=self._database) as session:
            for node_id in node_ids:
                await session.run(
                    "MATCH (n:NODE {id: $id}) DETACH DELETE n",
                    id=node_id,
                )

    @override
    async def get_edges(self, edge_specs: List[EdgeSpec]) -> List[Optional[EdgeT]]:
        results: List[Optional[EdgeT]] = []
        async with self._driver.session(database=self._database) as session:
            for subject_id, object_id, edge_id in edge_specs:
                if edge_id is not None:
                    result = await session.run(
                        """
                        MATCH (s:NODE {id: $sid})-[r {id: $eid}]->(t:NODE {id: $oid})
                        RETURN s, r, t
                        """,
                        sid=subject_id,
                        oid=object_id,
                        eid=edge_id,
                    )
                else:
                    result = await session.run(
                        """
                        MATCH (s:NODE {id: $sid})-[r]->(t:NODE {id: $oid})
                        RETURN s, r, t
                        """,
                        sid=subject_id,
                        oid=object_id,
                    )
                record = await result.single()
                if record is None:
                    results.append(None)
                else:
                    rel = record["r"]
                    sn = record["s"]
                    tn = record["t"]
                    props = dict(rel)
                    rid = props.pop("id", None) or rel.element_id
                    results.append(
                        self._edge_cls(
                            subject_id=sn.get("id", sn.element_id),
                            object_id=tn.get("id", tn.element_id),
                            id=rid,
                            **props,
                        )
                    )
        return results

    @override
    async def upsert_edges(self, edges: Iterable[EdgeT]) -> None:
        async with self._driver.session(database=self._database) as session:
            for edge in edges:
                edge_data = asdict(edge)
                subject_id = edge_data.pop("subject_id")
                object_id = edge_data.pop("object_id")
                edge_id = edge_data.pop("id", None)
                props = self._serialize_props(edge_data)
                if edge_id is not None:
                    await session.run(
                        """
                        MATCH (s:NODE {id: $sid})
                        MATCH (t:NODE {id: $oid})
                        MERGE (s)-[r:RELATION {id: $eid}]->(t)
                        SET r += $props
                        """,
                        sid=subject_id,
                        oid=object_id,
                        eid=edge_id,
                        props=props,
                    )
                else:
                    await session.run(
                        """
                        MATCH (s:NODE {id: $sid})
                        MATCH (t:NODE {id: $oid})
                        CREATE (s)-[r:RELATION]->(t)
                        SET r += $props
                        """,
                        sid=subject_id,
                        oid=object_id,
                        props=props,
                    )

    @override
    async def delete_edges(self, edge_specs: List[EdgeSpec]) -> None:
        async with self._driver.session(database=self._database) as session:
            for subject_id, object_id, edge_id in edge_specs:
                if edge_id is not None:
                    await session.run(
                        """
                        MATCH (s:NODE {id: $sid})-[r:RELATION {id: $eid}]->(t:NODE {id: $oid})
                        DELETE r
                        """,
                        sid=subject_id,
                        oid=object_id,
                        eid=edge_id,
                    )
                else:
                    await session.run(
                        """
                        MATCH (s:NODE {id: $sid})-[r:RELATION]->(t:NODE {id: $oid})
                        DELETE r
                        """,
                        sid=subject_id,
                        oid=object_id,
                    )

    @override
    async def get_all_edges_for_nodes(self, node_ids: List[str]) -> List[List[EdgeT]]:
        grouped: List[List[EdgeT]] = []
        async with self._driver.session(database=self._database) as session:
            for node_id in node_ids:
                result = await session.run(
                    """
                    MATCH (n:NODE {id: $id})-[r]-(m:NODE)
                    RETURN n, r, m
                    """,
                    id=node_id,
                )
                edges: List[EdgeT] = []
                async for record in result:
                    rel = record["r"]
                    sn = record["n"]
                    mn = record["m"]
                    props = dict(rel)
                    rid = props.pop("id", None) or rel.element_id
                    edges.append(
                        self._edge_cls(
                            subject_id=sn.get("id", sn.element_id),
                            object_id=mn.get("id", mn.element_id),
                            id=rid,
                            **props,
                        )
                    )
                grouped.append(edges)
        return grouped

    @override
    async def get_all_nodes(self) -> List[NodeT]:
        nodes: List[NodeT] = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run("MATCH (n:NODE) RETURN n")
            async for record in result:
                props = dict(record["n"])
                node_id = props.pop("id")
                props = self._deserialize_props(props, self._node_json_fields)
                nodes.append(self._node_cls(id=node_id, **props))
        return nodes

    @override
    async def get_all_edges(self) -> List[EdgeT]:
        edges: List[EdgeT] = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run("MATCH (s)-[r:RELATION]->(t) RETURN s, r, t")
            async for record in result:
                rel = record["r"]
                sn = record["s"]
                tn = record["t"]
                props = dict(rel)
                rid = props.pop("id", None) or rel.element_id
                edges.append(
                    self._edge_cls(
                        subject_id=sn.get("id", sn.element_id),
                        object_id=tn.get("id", tn.element_id),
                        id=rid,
                        **props,
                    )
                )
        return edges

    async def close(self) -> None:
        await self._driver.close()