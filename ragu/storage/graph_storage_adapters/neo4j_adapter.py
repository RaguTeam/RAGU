from __future__ import annotations

import dataclasses
import json
import re
from collections import defaultdict
from dataclasses import asdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from neo4j import AsyncDriver
from neo4j import AsyncGraphDatabase
from ragu.storage.base_storage import BaseGraphStorage, EdgeSpec
from ragu.storage.types import Node, Edge
from typing_extensions import LiteralString, override

NodeT = TypeVar("NodeT", bound=Node)
EdgeT = TypeVar("EdgeT", bound=Edge)

#: Property types Neo4j stores natively; anything else is serialized to JSON.
_PRIMITIVES = (str, int, float, bool)

#: Label applied to every node this adapter writes, used for lookups.
_NODE_LABEL = "NODE"

#: Relationship type used when an edge declares no grouping field, and the
#: historical type of every edge written before types were derived per edge.
_EDGE_TYPE = "RELATION"

#: Matches the edges this adapter owns. Relationship types vary per edge, so the
#: type cannot be used as a filter; every edge RAGU writes carries an ``id``,
#: relationships created by other tools in the same database do not.
_OURS = "r.id IS NOT NULL"


def _cypher(query: str) -> LiteralString:
    """
    Mark a Cypher statement as safe to execute.

    The driver types ``session.run`` as accepting ``LiteralString`` so that
    queries cannot be assembled from runtime data. Every statement in this
    module is built from string literals and the module constants above, with
    one exception: :meth:`Neo4jStorage.upsert_nodes` interpolates a node label,
    because Cypher has no placeholder for labels. That value passes through
    :meth:`Neo4jStorage._sanitize_label` first, which strips it to word
    characters.

    **Anything routed through this helper is asserted to be injection-free by
    construction.** Never pass it a string containing unsanitized input.

    :param query: Cypher statement built from literals.
    :type query: str
    :returns: The same statement, typed as a literal for the driver.
    :rtype: LiteralString
    """
    return cast(LiteralString, query)


class Neo4jStorage(BaseGraphStorage[NodeT, EdgeT]):
    """
    Neo4j-backed implementation of :class:`BaseGraphStorage`.

    Nodes are written with a shared ``:NODE`` label plus a per-type label
    derived from the node itself, so Cypher lookups stay uniform while the
    graph remains readable in Neo4j Browser.

    Uses the ``neo4j`` driver, which ships as a regular dependency.

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
    :raises ImportError: If the ``neo4j`` driver is not importable.
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
        """
        Open a driver against the given server. No connection is made yet;
        :meth:`index_start_callback` is where connectivity is first checked.

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
        :param kwargs: Ignored; accepted so that the storage can be constructed
            from the same argument bag as the other graph backends, which take
            options this one has no use for.
        :raises ImportError: If the ``neo4j`` driver is not importable.
        """

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

        The node type decides which of its fields groups it, via
        :attr:`~ragu.storage.types.Node.label_field`; this adapter only decides
        what to do with that value. Types without a grouping field get the
        generic label rather than a silently guessed one.

        :param node: Node about to be written.
        :type node: Node
        :returns: Label name, falling back to the generic node label.
        :rtype: str
        """
        return node.get_label() or _NODE_LABEL

    @staticmethod
    def _edge_label_for(edge: Edge) -> str:
        """
        Choose the Neo4j relationship type for an edge.

        Unlike node labels, a relationship carries exactly one type and it
        cannot be changed afterwards, so this value is fixed at creation.

        :param edge: Edge about to be written.
        :type edge: Edge
        :returns: Relationship type, falling back to the generic one.
        :rtype: str
        """
        return edge.get_label() or _EDGE_TYPE

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

    def _edge_from_record(self, record) -> EdgeT:
        """
        Build an edge from a record holding ``s`` (start), ``r`` and ``t`` (end).

        Endpoints are taken from the relationship itself rather than from the
        matched pattern, so an undirected match still reports the stored
        direction instead of the direction it happened to be traversed in.

        :param record: Neo4j record with ``s``, ``r`` and ``t`` keys.
        :returns: Materialized edge.
        :rtype: EdgeT
        """
        relationship = record["r"]
        props = dict(relationship)
        edge_id = props.pop("id", None) or relationship.element_id
        props = self._deserialize_props(props, self._edge_json_fields)

        # Properties the edge class does not declare are dropped rather than
        # passed on: a single relationship carrying an extra property - added by
        # hand in Neo4j Browser, or by another tool sharing the database - would
        # otherwise raise TypeError and take down the whole read.
        known = {field.name for field in dataclasses.fields(self._edge_cls)}
        props = {key: value for key, value in props.items() if key in known}

        return self._edge_cls(
            subject_id=record["s"].get("id", record["s"].element_id),
            object_id=record["t"].get("id", record["t"].element_id),
            id=edge_id,
            **props,
        )

    async def _verify_connectivity(self) -> None:
        """
        Check that the server is reachable and the credentials are accepted.

        :raises neo4j.exceptions.Neo4jError: If the server rejects the query.
        :raises neo4j.exceptions.ServiceUnavailable: If the server is unreachable.
        """
        async with self._driver.session(database=self._database) as session:
            await session.run(_cypher("RETURN 1"))

    async def index_done_callback(self) -> None:
        """
        Post-index hook, intentionally a no-op.

        Unlike file-backed backends there is nothing to flush: writes are
        committed by the server as each statement runs.
        """

    async def query_done_callback(self) -> None:
        """
        Post-query hook, intentionally a no-op. Kept for interface compatibility.
        """

    async def index_start_callback(self) -> None:
        """
        Verify connectivity and ensure the node id constraint and type index exist.

        Without the constraint every ``MATCH (n:NODE {id: ...})`` is a label
        scan, which turns graph building into quadratic work. The constraint
        also backs the uniqueness that :meth:`upsert_nodes` relies on.

        An index is also created on the node type's
        :attr:`~ragu.storage.types.Node.label_field`, if it declares one, so that
        filtering by that property is cheap. Filter on the property rather than
        on the per-type label: labels are additive and may be stale, see
        :meth:`upsert_nodes`.
        """
        await self._verify_connectivity()
        async with self._driver.session(database=self._database) as session:
            await session.run(
                _cypher(
                    f"CREATE CONSTRAINT ragu_node_id IF NOT EXISTS "
                    f"FOR (n:{_NODE_LABEL}) REQUIRE n.id IS UNIQUE"
                ),
            )
            label_field = self._node_cls.label_field
            if label_field:
                await session.run(
                    _cypher(
                        f"CREATE INDEX ragu_node_{label_field} IF NOT EXISTS "
                        f"FOR (n:{_NODE_LABEL}) ON (n.{self._sanitize_label(label_field)})"
                    ),
                )

    @override
    async def edges_degrees(self, edge_specs: List[EdgeSpec]) -> List[int]:
        """
        Return ``degree(subject) + degree(object)`` for each spec.

        Degrees for all distinct endpoints are counted in a single query; the
        previous implementation issued two round trips per spec.

        :param edge_specs: Specs of ``(subject_id, object_id, edge_id)``.
        :type edge_specs: List[EdgeSpec]
        :returns: Degree sums aligned with ``edge_specs``; unknown nodes count as 0.
        :rtype: List[int]
        """
        if not edge_specs:
            return []

        endpoints = list({node_id for spec in edge_specs for node_id in (spec[0], spec[1])})
        degrees: Dict[str, int] = {}
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _cypher(
                    f"""
                    UNWIND $ids AS id
                    MATCH (n:{_NODE_LABEL} {{id: id}})
                    RETURN id AS id, count {{ (n)-[r]-() WHERE r.id IS NOT NULL }} AS degree
                    """,
                ),
                ids=endpoints,
            )
            async for record in result:
                degrees[record["id"]] = record["degree"]

        return [
            degrees.get(subject_id, 0) + degrees.get(object_id, 0)
            for subject_id, object_id, _edge_id in edge_specs
        ]

    @override
    async def upsert_nodes(self, nodes: Iterable[NodeT]) -> None:
        """
        Insert or update nodes, keyed by ``id`` alone.

        The per-type label is applied after the merge rather than being part of
        the match: including it would make a node whose type changed fail to
        match its stored counterpart, silently creating a second node with the
        same ``id``. Type labels therefore accumulate if a node changes type;
        ``:NODE`` plus the ``entity_type`` property remain authoritative.

        .. todo::
           Remove the previous type label instead of accumulating labels, so
           that ``MATCH (n:PERSON)`` stops matching nodes that are no longer of
           that type::

               MATCH (n:NODE {id: row.id})
               WITH n, n.entity_type AS previous
               SET n += row.props
               CALL { WITH n, previous REMOVE n:$(previous) }
               SET n:$(row.label)

           This needs dynamic labels, available only from **Neo4j server 5.24**.
           The ``neo4j`` extra currently pins the *driver* (``>=5.26.0``) and
           says nothing about the server, so adopting this would introduce a
           server requirement that must be declared explicitly first.

        :param nodes: Nodes to insert or update.
        :type nodes: Iterable[NodeT]
        """
        rows_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            attrs = asdict(node)
            node_id = attrs.pop("id")
            label = self._sanitize_label(self._label_for(node))
            rows_by_label[label].append({"id": node_id, "props": self._serialize_props(attrs)})

        if not rows_by_label:
            return

        # One statement per label rather than per node: labels cannot be
        # parameterized before Neo4j 5.24, and the type count is small and
        # bounded (29 NEREL types), while the node count is not.
        async with self._driver.session(database=self._database) as session:
            for label, rows in rows_by_label.items():
                # The label is the one part of the query that is not a literal.
                # It is interpolated rather than passed as a parameter because
                # Cypher has no placeholder for labels, and it is safe to do so
                # only because _sanitize_label() has stripped it to word
                # characters. The cast records that reasoning for the type
                # checker, which cannot see the sanitization.
                await session.run(
                    _cypher(
                        f"UNWIND $rows AS row "
                        f"MERGE (n:{_NODE_LABEL} {{id: row.id}}) "
                        f"SET n += row.props SET n:{label}"
                    ),
                    rows=rows,
                )

    @override
    async def get_nodes(self, node_ids: List[str]) -> List[Optional[NodeT]]:
        """
        Fetch nodes by id in one round trip, preserving input order.

        :param node_ids: Identifiers to fetch.
        :type node_ids: List[str]
        :returns: Nodes aligned with ``node_ids``; missing ids mapped to ``None``.
        :rtype: List[Optional[NodeT]]
        """
        if not node_ids:
            return []

        found: Dict[str, NodeT] = {}
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _cypher(f"UNWIND $ids AS id MATCH (n:{_NODE_LABEL} {{id: id}}) RETURN n"),
                ids=list(dict.fromkeys(node_ids)),
            )
            async for record in result:
                props = dict(record["n"])
                node_id = props.pop("id")
                props = self._deserialize_props(props, self._node_json_fields)
                found[node_id] = self._node_cls(id=node_id, **props)

        return [found.get(node_id) for node_id in node_ids]

    @override
    async def delete_nodes(self, node_ids: List[str]) -> None:
        """
        Delete nodes and their incident edges. Missing ids are tolerated.

        :param node_ids: Identifiers to remove.
        :type node_ids: List[str]
        """
        if not node_ids:
            return

        async with self._driver.session(database=self._database) as session:
            await session.run(
                _cypher(f"UNWIND $ids AS id MATCH (n:{_NODE_LABEL} {{id: id}}) DETACH DELETE n"),
                ids=node_ids,
            )

    @override
    async def get_edges(self, edge_specs: List[EdgeSpec]) -> List[Optional[EdgeT]]:
        """
        Fetch edges by spec in one round trip, preserving input order.

        Each spec is tagged with its position so results can be realigned: a
        batched query returns rows in storage order, and specs may repeat.

        :param edge_specs: Specs of ``(subject_id, object_id, edge_id)``.
        :type edge_specs: List[EdgeSpec]
        :returns: Edges aligned with ``edge_specs``; missing ones mapped to ``None``.
        :rtype: List[Optional[EdgeT]]
        """
        if not edge_specs:
            return []

        rows = [
            {"i": index, "sid": subject_id, "oid": object_id, "eid": edge_id}
            for index, (subject_id, object_id, edge_id) in enumerate(edge_specs)
        ]
        results: List[Optional[EdgeT]] = [None] * len(edge_specs)

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _cypher(
                    f"""
                    UNWIND $rows AS row
                    MATCH (s:{_NODE_LABEL} {{id: row.sid}})-[r]->(t:{_NODE_LABEL} {{id: row.oid}})
                    WHERE {_OURS} AND (row.eid IS NULL OR r.id = row.eid)
                    RETURN row.i AS i, s, r, t
                    """,
                ),
                rows=rows,
            )
            async for record in result:
                results[record["i"]] = self._edge_from_record(record)
        return results

    @override
    async def upsert_edges(self, edges: Iterable[EdgeT]) -> None:
        """
        Insert or update edges, grouped by relationship type. Endpoints must
        already exist.

        Each edge is written under the type its class declares through
        :attr:`~ragu.storage.types.Edge.label_field`, so a graph reads naturally
        in Neo4j Browser. Unlike node labels, a relationship type is fixed at
        creation: re-upserting an edge whose type changed would leave the old
        relationship in place and add a second one. RAGU never does that on its
        own, since ``Relation.id`` is derived from the type among other fields,
        but code assigning ids by hand can.

        :param edges: Edges to insert or update.
        :type edges: Iterable[EdgeT]
        """
        rows_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            edge_data = asdict(edge)
            edge_type = self._sanitize_label(self._edge_label_for(edge))
            rows_by_type[edge_type].append({
                "sid": edge_data.pop("subject_id"),
                "oid": edge_data.pop("object_id"),
                "eid": edge_data.pop("id", None),
                "props": self._serialize_props(edge_data),
            })

        if not rows_by_type:
            return

        # One statement per relationship type, for the same reason as in
        # upsert_nodes: types cannot be parameterized in Cypher.
        async with self._driver.session(database=self._database) as session:
            for edge_type, rows in rows_by_type.items():
                await session.run(
                    _cypher(
                        f"""
                        UNWIND $rows AS row
                        MATCH (s:{_NODE_LABEL} {{id: row.sid}})
                        MATCH (t:{_NODE_LABEL} {{id: row.oid}})
                        MERGE (s)-[r:{edge_type} {{id: row.eid}}]->(t)
                        SET r += row.props
                        """,
                    ),
                    rows=rows,
                )

    @override
    async def delete_edges(self, edge_specs: List[EdgeSpec]) -> None:
        """
        Delete edges by spec in one round trip. Missing edges are tolerated.

        A spec with ``edge_id`` of ``None`` removes every edge between the pair.

        :param edge_specs: Specs of ``(subject_id, object_id, edge_id)``.
        :type edge_specs: List[EdgeSpec]
        """
        if not edge_specs:
            return

        rows = [
            {"sid": subject_id, "oid": object_id, "eid": edge_id}
            for subject_id, object_id, edge_id in edge_specs
        ]
        async with self._driver.session(database=self._database) as session:
            await session.run(
                _cypher(
                    f"""
                    UNWIND $rows AS row
                    MATCH (s:{_NODE_LABEL} {{id: row.sid}})-[r]->(t:{_NODE_LABEL} {{id: row.oid}})
                    WHERE {_OURS} AND (row.eid IS NULL OR r.id = row.eid)
                    DELETE r
                    """,
                ),
                rows=rows,
            )

    @override
    async def get_all_edges_for_nodes(self, node_ids: List[str]) -> List[List[EdgeT]]:
        """
        Retrieve the edges incident to each node, one list per input node.

        Edges are reported with their stored direction regardless of which
        endpoint was queried, and no deduplication happens across nodes: an
        edge between two requested nodes appears in both lists.

        :param node_ids: Identifiers whose edges to fetch.
        :type node_ids: List[str]
        :returns: Edge lists aligned with ``node_ids``; unknown nodes yield ``[]``.
        :rtype: List[List[EdgeT]]
        """
        grouped: List[List[EdgeT]] = []
        async with self._driver.session(database=self._database) as session:
            for node_id in node_ids:
                result = await session.run(
                    _cypher(
                        f"""
                        MATCH (n:{_NODE_LABEL} {{id: $id}})-[r]-(:{_NODE_LABEL})
                        WHERE {_OURS}
                        RETURN startNode(r) AS s, r, endNode(r) AS t
                        """,
                    ),
                    id=node_id,
                )
                grouped.append([self._edge_from_record(record) async for record in result])
        return grouped

    @override
    async def get_all_nodes(self) -> List[NodeT]:
        """
        Retrieve every node this adapter has written.

        Loads the whole graph into memory, so prefer :meth:`get_nodes` or
        :meth:`run_cypher_query` when only part of it is needed.

        :returns: All stored nodes, in no particular order.
        :rtype: List[NodeT]
        """
        nodes: List[NodeT] = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(_cypher(f"MATCH (n:{_NODE_LABEL}) RETURN n"))
            async for record in result:
                props = dict(record["n"])
                node_id = props.pop("id")
                props = self._deserialize_props(props, self._node_json_fields)
                nodes.append(self._node_cls(id=node_id, **props))
        return nodes

    @override
    async def get_all_edges(self) -> List[EdgeT]:
        """
        Retrieve every edge this adapter has written.

        Loads the whole graph into memory, so prefer :meth:`get_edges` or
        :meth:`run_cypher_query` when only part of it is needed.

        :returns: All stored edges, in no particular order.
        :rtype: List[EdgeT]
        """
        edges: List[EdgeT] = []
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                _cypher(
                    f"MATCH (s:{_NODE_LABEL})-[r]->(t:{_NODE_LABEL}) WHERE {_OURS} RETURN s, r, t"
                ),
            )
            edges = [self._edge_from_record(record) async for record in result]
        return edges

    async def run_cypher_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run an arbitrary Cypher statement and return its records as dicts.

        An escape hatch for what :class:`BaseGraphStorage` does not cover:
        filtering nodes by type, aggregations, graph algorithms, ad-hoc
        exploration. Neo4j-specific by definition, so it is absent from the
        base interface and code using it will not run on other backends.

        Values are returned as the driver produces them: nodes and
        relationships arrive as driver objects rather than
        :class:`~ragu.storage.types.Node` / :class:`~ragu.storage.types.Edge`,
        and fields written as JSON (see :meth:`_json_fields_for`) come back as
        strings. Return the properties you need rather than whole nodes if you
        want plain values.

        **Pass data through** ``parameters``, **never by formatting it into**
        ``query``. Cypher has no placeholder for labels and relationship types;
        if you must interpolate one, sanitize it first the way
        :meth:`_sanitize_label` does.

        Filtering by node type is a common case, and it should go through the
        property rather than the label, because labels accumulate::

            rows = await storage.run_cypher_query(
                "MATCH (n:NODE) WHERE n.entity_type IN $types RETURN n.id AS id",
                {"types": ["PERSON", "ORGANIZATION"]},
            )

        :param query: Cypher statement to execute.
        :type query: str
        :param parameters: Query parameters referenced as ``$name`` in the statement.
        :type parameters: Optional[Dict[str, Any]]
        :returns: One dict per record, keyed by the names the query returns.
        :rtype: List[Dict[str, Any]]
        """
        async with self._driver.session(database=self._database) as session:
            result = await session.run(_cypher(query), parameters or {})
            return [dict(record) async for record in result]

    async def close(self) -> None:
        """
        Close the driver and release its connection pool.

        Not part of :class:`BaseGraphStorage`, so nothing in RAGU calls it; a
        long-lived process that creates adapters repeatedly should call it
        itself to avoid leaking connections.
        """
        await self._driver.close()