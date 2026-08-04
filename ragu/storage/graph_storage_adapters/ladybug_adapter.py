import asyncio
import dataclasses
import functools
import inspect
import json
import os
import re
import threading
import weakref
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Iterable, TypeVar, cast

from loguru import logger
from typing_extensions import override

from ragu.common.batch_generator import BatchGenerator
from ragu.storage.base_storage import BaseGraphStorage, EdgeSpec
from ragu.storage.types import Edge, Node
from ragu.utils.ragu_utils import serialize

NodeT = TypeVar("NodeT", bound=Node)
EdgeT = TypeVar("EdgeT", bound=Edge)

_DEFAULT_NODE_TABLE = "RaguNode"
_DEFAULT_EDGE_TABLE = "RaguEdge"

_DEFAULT_BATCH_SIZE = 5000

_WriteMethod = TypeVar("_WriteMethod", bound="Callable[..., Awaitable[Any]]")


def _serialized_write(method: _WriteMethod) -> _WriteMethod:
    """
    Run a mutating method under the adapter's write lock.

    Reads are deliberately not serialized: they are safe to overlap, and they
    rely on running concurrently for throughput.

    :param method: Mutating adapter coroutine method.
    :return: The method, wrapped so that calls on one adapter run one at a time.
    """

    @functools.wraps(method)
    async def wrapper(self: "LadybugGraphStorage", *args: Any, **kwargs: Any) -> Any:
        async with self._write_lock:
            return await method(self, *args, **kwargs)

    return cast(_WriteMethod, wrapper)


class LadybugGraphStorage(BaseGraphStorage[NodeT, EdgeT]):
    """
    LadybugDB-backed graph implementation of :class:`BaseGraphStorage`.

    The adapter owns a small generic schema: one node table and one relationship
    table. Domain-specific dataclass fields are stored as JSON payloads so the
    backend can reconstruct any configured ``node_cls`` and ``edge_cls``.

    :param filename: Ladybug database path.
    :param node_cls: Dataclass used to materialize nodes.
    :param edge_cls: Dataclass used to materialize edges.
    :param graph_name: Reserved for future named subgraph support.
    :param node_table: Managed Ladybug node table name.
    :param edge_table: Managed Ladybug relationship table name.
    :param max_concurrent_queries: Ladybug async connection concurrency.
    :param batch_size: Rows sent per ``UNWIND`` statement. Larger batches mean
        fewer round trips but hold more rows in one query parameter.
    :raises ImportError: If the ``ladybug`` package is not importable.
    :raises ValueError: If table names are unsafe, named subgraphs are
        requested, or ``batch_size`` is not positive.
    """

    _open_databases: ClassVar[dict[str, "weakref.ReferenceType[LadybugGraphStorage]"]] = {}
    _open_databases_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        filename: str,
        node_cls: type[NodeT],
        edge_cls: type[EdgeT],
        graph_name: str | None = None,
        node_table: str = _DEFAULT_NODE_TABLE,
        edge_table: str = _DEFAULT_EDGE_TABLE,
        max_concurrent_queries: int = 4,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ) -> None:
        if graph_name is not None:
            raise ValueError(
                "LadybugGraphStorage graph_name is reserved until Ladybug exposes "
                "idempotent named subgraph setup suitable for adapter initialization."
            )

        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")

        try:
            import ladybug as lb
        except ImportError as exc:
            raise ImportError(
                "LadybugGraphStorage requires the 'ladybug' package. "
                "Install RAGU with a version that includes ladybug support."
            ) from exc

        self._filename = filename
        self._node_cls = node_cls
        self._edge_cls = edge_cls
        self._node_table = self._sanitize_identifier(node_table)
        self._edge_table = self._sanitize_identifier(edge_table)
        self._batch_size = batch_size
        self._write_lock = asyncio.Lock()
        #: Whether the managed tables are known to exist; see :meth:`_ensure_schema`.
        self._schema_ready = False
        #: Keeps concurrent first statements from each issuing the schema DDL.
        self._schema_lock = asyncio.Lock()

        self._path_key: str | None = None
        if filename != ":memory:":
            parent = os.path.dirname(os.path.abspath(filename))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._path_key = os.path.realpath(os.path.abspath(filename))
            self._reserve_path(self._path_key)

        try:
            self._database = lb.Database(filename)
            self._connection = lb.AsyncConnection(
                self._database,
                max_concurrent_queries=max_concurrent_queries,
            )
        except BaseException:
            self._release_path()
            raise

    def _reserve_path(self, path_key: str) -> None:
        """
        Claim a database path for this adapter, for the lifetime of the process.

        :param path_key: Resolved absolute path of the database.
        :raises RuntimeError: If another live adapter already holds the path.
        """
        with LadybugGraphStorage._open_databases_lock:
            existing = LadybugGraphStorage._open_databases.get(path_key)
            if existing is not None and existing() is not None:
                raise RuntimeError(
                    f"Ladybug database {path_key!r} is already open in this process. "
                    "Ladybug does not guard against a second handle inside one "
                    "process, and two adapters on one file corrupt it silently - "
                    "reads diverge and writes interleave into rows that carry "
                    "another record's payload. Reuse the existing storage, or "
                    "close it before opening the file again."
                )
            LadybugGraphStorage._open_databases[path_key] = weakref.ref(self)

    def _release_path(self) -> None:
        """
        Give up this adapter's claim on its database path.

        Only clears the entry if it still belongs to this adapter, so a handle
        closed after its path was reclaimed cannot evict the new owner.
        """
        path_key = getattr(self, "_path_key", None)
        if path_key is None:
            return
        with LadybugGraphStorage._open_databases_lock:
            existing = LadybugGraphStorage._open_databases.get(path_key)
            if existing is None or existing() in (None, self):
                LadybugGraphStorage._open_databases.pop(path_key, None)
        self._path_key = None

    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        """
        Validate an adapter-managed Cypher identifier.

        :param identifier: Requested table or graph identifier.
        :return: Safe identifier.
        :raises ValueError: If the identifier cannot be safely interpolated.
        """
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
            raise ValueError(f"Unsafe Ladybug identifier: {identifier!r}")
        return identifier

    @staticmethod
    def _payload_from_record(record: Node | Edge, excluded: set[str]) -> str:
        """
        Serialize a graph record payload into JSON.

        :param record: Node or edge record.
        :param excluded: Structural fields to omit from the JSON payload.
        :return: JSON string.
        """
        payload = serialize(record)
        if not isinstance(payload, dict):
            raise ValueError(f"Cannot serialize graph record payload: {record!r}")
        for field in excluded:
            payload.pop(field, None)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _filtered_payload(
        payload: str | None,
        record_cls: type,
        excluded: set[str],
    ) -> dict[str, Any]:
        """
        Decode a stored JSON payload into constructor kwargs.

        :param payload: Stored JSON payload.
        :param record_cls: Node or edge dataclass.
        :param excluded: Structural fields stored outside the payload.
        :return: Constructor kwargs.
        :raises ValueError: If the payload is not a JSON object.
        """
        if not payload:
            return {}
        try:
            props = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("Stored Ladybug payload is not valid JSON") from exc
        if not isinstance(props, dict):
            raise ValueError("Stored Ladybug payload must be a JSON object")

        if not dataclasses.is_dataclass(record_cls):
            return props
        known = {field.name for field in dataclasses.fields(record_cls)} - excluded
        if not known:
            return props
        return {key: value for key, value in props.items() if key in known}

    def _node_from_row(self, row: dict[str, Any]) -> NodeT:
        """
        Materialize a node from a Ladybug result row.

        :param row: Result row with ``id`` and ``payload``.
        :return: Node instance.
        """
        props = self._filtered_payload(row.get("payload"), self._node_cls, {"id"})
        return self._node_cls(id=row["id"], **props)

    def _edge_from_row(self, row: dict[str, Any]) -> EdgeT:
        """
        Materialize an edge from a Ladybug result row.

        :param row: Result row with endpoints, edge id, and payload.
        :return: Edge instance.
        """
        props = self._filtered_payload(
            row.get("payload"),
            self._edge_cls,
            {"id", "subject_id", "object_id"},
        )
        return self._edge_cls(
            subject_id=row["subject_id"],
            object_id=row["object_id"],
            id=row["id"],
            **props,
        )

    async def _create_schema(self) -> None:
        """
        Create the managed node and relationship tables if they are absent.

        Uses :meth:`_execute` rather than :meth:`_run`, which would ask for the
        schema this method is creating.
        """
        await self._execute(
            f"""
            CREATE NODE TABLE IF NOT EXISTS {self._node_table}(
                id STRING PRIMARY KEY,
                label STRING,
                payload STRING
            )
            """
        )
        await self._execute(
            f"""
            CREATE REL TABLE IF NOT EXISTS {self._edge_table}(
                FROM {self._node_table} TO {self._node_table},
                id STRING,
                label STRING,
                payload STRING
            )
            """
        )

    async def _ensure_schema(self) -> None:
        """
        Make sure the managed tables exist before a statement touches them.

        Creating the schema is idempotent, so the flag is only an optimisation;
        the lock keeps concurrent first calls from each issuing the DDL.

        :return: Nothing.
        """
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            await self._create_schema()
            self._schema_ready = True

    async def _execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """
        Execute a Cypher statement without ensuring the schema first.

        Only for statements that must not trigger schema creation: the DDL
        itself and the connectivity probe.

        :param query: Cypher statement.
        :param parameters: Query parameters.
        :return: Ladybug query result.
        """
        return await self._connection.execute(query, parameters=parameters or {})

    async def _run(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """
        Execute a Cypher statement, creating the managed schema on first use.

        :param query: Cypher statement.
        :param parameters: Query parameters.
        :return: Ladybug query result.
        """
        await self._ensure_schema()
        return await self._execute(query, parameters)

    async def _query_batches(
        self,
        query: str,
        parameter_sets: list[dict[str, Any]],
        columns: tuple[str, ...],
    ) -> list[list[dict[str, Any]]]:
        """
        Run one read query over several parameter sets concurrently.

        :param query: Cypher statement to run once per parameter set.
        :param parameter_sets: Query parameters, one per chunk.
        :param columns: Positional fallback column names.
        :return: Result rows per parameter set, in the order the sets were given
            regardless of which query finished first.
        """
        if not parameter_sets:
            return []
        if len(parameter_sets) == 1:
            return [await self._query(query, parameter_sets[0], columns)]
        return list(await asyncio.gather(
            *(self._query(query, parameters, columns) for parameters in parameter_sets)
        ))

    async def _query(
        self,
        query: str,
        parameters: dict[str, Any] | None,
        columns: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """
        Execute a read query and convert rows to dictionaries.

        :param query: Cypher statement.
        :param parameters: Query parameters.
        :param columns: Positional fallback column names, used only if the
            driver returns plain tuples rather than mappings.
        :return: Result rows.
        """
        result = await self._run(query, parameters)
        if result is None:
            return []

        if hasattr(result, "rows_as_dict"):
            rows_as_dict = result.rows_as_dict()
            if hasattr(rows_as_dict, "get_all"):
                return [dict(row) for row in rows_as_dict.get_all()]
            return [dict(row) for row in rows_as_dict]

        if hasattr(result, "get_all"):
            rows = result.get_all()
        else:
            rows = list(result)

        converted: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                converted.append(row)
            elif hasattr(row, "keys"):
                converted.append(dict(row))
            else:
                converted.append({column: row[index] for index, column in enumerate(columns)})
        return converted

    @override
    async def index_start_callback(self) -> None:
        """
        Verify that the embedded database accepts queries and set up the schema.

        Calling this is optional - :meth:`_run` creates the schema on first use
        - but it surfaces an unreachable or unusable database at a point where
        the error still points at initialization rather than at whichever query
        happened to run first.
        """
        await self._execute("RETURN 1 AS ok")
        await self._ensure_schema()

    @override
    async def index_done_callback(self) -> None:
        """
        Post-index hook.

        Ladybug persists writes through the database engine, so there is no
        file flush equivalent to NetworkX GML serialization.
        """

    @override
    async def query_done_callback(self) -> None:
        """
        Post-query hook kept for interface compatibility.
        """

    @override
    async def edges_degrees(self, edge_specs: list[EdgeSpec]) -> list[int]:
        """
        Return degree(subject) + degree(object) for each edge spec.

        :param edge_specs: Specs of ``(subject_id, object_id, edge_id)``.
        :return: Degree sums aligned with ``edge_specs``; unknown nodes count as 0.
        """
        if not edge_specs:
            return []

        endpoint_ids = list({
            node_id
            for subject_id, object_id, _ in edge_specs
            for node_id in (subject_id, object_id)
        })
        degrees: dict[str, int] = {}

        results = await self._query_batches(
            f"""
            UNWIND $ids AS wanted
            MATCH (n:{self._node_table})-[r:{self._edge_table}]-(:{self._node_table})
            WHERE n.id = wanted
            RETURN wanted AS id, count(r) AS total
            """,
            [{"ids": chunk}
             for chunk in BatchGenerator(endpoint_ids, self._batch_size).get_batches()],
            ("id", "total"),
        )
        for rows in results:
            for row in rows:
                degrees[row["id"]] = int(row.get("total", 0) or 0)

        return [
            degrees.get(subject_id, 0) + degrees.get(object_id, 0)
            for subject_id, object_id, _edge_id in edge_specs
        ]

    @override
    async def get_nodes(self, node_ids: list[str]) -> list[NodeT | None]:
        """
        Fetch nodes by IDs, preserving input order.

        Duplicate IDs are queried once and mapped back to every position they
        occupy.

        :param node_ids: Node identifiers.
        :return: Nodes aligned with input IDs; missing IDs map to ``None``.
        """
        if not node_ids:
            return []

        found: dict[str, NodeT] = {}
        results = await self._query_batches(
            f"""
            UNWIND $ids AS wanted
            MATCH (n:{self._node_table})
            WHERE n.id = wanted
            RETURN n.id AS id, n.payload AS payload
            """,
            [{"ids": chunk} for chunk in
             BatchGenerator(list(dict.fromkeys(node_ids)), self._batch_size).get_batches()],
            ("id", "payload"),
        )
        for rows in results:
            for row in rows:
                found[row["id"]] = self._node_from_row(row)

        return [found.get(node_id) for node_id in node_ids]

    async def _copy_rows(self, table: str, columns: tuple[str, ...],
                         rows: list[dict[str, Any]]) -> bool:
        """
        Bulk-load rows into a table with ``COPY``, or report that it was not used.


        :param table: Managed table to load into.
        :param columns: Column order matching the table definition.
        :param rows: Rows to load.
        :return: ``True`` if the rows were loaded, ``False`` if the caller must
            write them itself.
        """
        if not rows:
            return True
        try:
            import pandas as pd
        except ImportError:  # declared dependency, but never worth failing over
            return False

        try:
            frame = pd.DataFrame(rows, columns=list(columns))
            await self._run(f"COPY {table} FROM $df", {"df": frame})
            return True
        except Exception as exc:
            logger.debug(
                "Ladybug COPY into {} rejected {} rows ({}); falling back to MERGE",
                table, len(rows), exc,
            )
            return False

    async def _existing_node_ids(self, node_ids: Iterable[str]) -> set[str]:
        """
        Return which of the given node IDs already exist.

        :param node_ids: Candidate identifiers.
        :return: The subset that is already stored.
        """
        wanted = list(dict.fromkeys(node_ids))
        if not wanted:
            return set()

        known: set[str] = set()
        results = await self._query_batches(
            f"""
            UNWIND $ids AS wanted
            MATCH (n:{self._node_table})
            WHERE n.id = wanted
            RETURN n.id AS id
            """,
            [{"ids": chunk} for chunk in
             BatchGenerator(wanted, self._batch_size).get_batches()],
            ("id",),
        )
        for rows in results:
            known.update(row["id"] for row in rows)
        return known

    async def _edge_table_is_empty(self) -> bool:
        """
        Report whether the managed relationship table holds no edges.

        :return: ``True`` when no edge is stored.
        """
        rows = await self._query(
            f"MATCH ()-[r:{self._edge_table}]->() RETURN r.id AS id LIMIT 1", None, ("id",)
        )
        return not rows

    async def _existing_edge_keys(
        self, rows: list[dict[str, Any]]
    ) -> set[tuple[str, str, str]]:
        """
        Return which of the given ``(subject, object, id)`` keys are already stored.

        :param rows: Prepared edge rows.
        :return: Keys that already exist in the relationship table.
        """
        keys = [
            {"subject_id": row["subject_id"], "object_id": row["object_id"], "id": row["id"]}
            for row in rows
        ]
        found: set[tuple[str, str, str]] = set()
        results = await self._query_batches(
            f"""
            UNWIND $rows AS row
            MATCH (s:{self._node_table} {{id: row.subject_id}})
                  -[r:{self._edge_table}]->
                  (t:{self._node_table} {{id: row.object_id}})
            WHERE r.id = row.id
            RETURN row.subject_id AS subject_id, row.object_id AS object_id, r.id AS id
            """,
            [{"rows": chunk} for chunk in BatchGenerator(keys, self._batch_size).get_batches()],
            ("subject_id", "object_id", "id"),
        )
        for batch in results:
            for row in batch:
                found.add((row["subject_id"], row["object_id"], row["id"]))
        return found

    @override
    @_serialized_write
    async def upsert_nodes(self, nodes: Iterable[NodeT]) -> None:
        """
        Insert or update nodes by RAGU node ID.

        :param nodes: Nodes to upsert.
        """
        rows_by_id: dict[str, dict[str, Any]] = {}
        for node in nodes:
            rows_by_id[node.id] = {
                "id": node.id,
                "label": node.get_label(),
                "payload": self._payload_from_record(node, {"id"}),
            }

        if not rows_by_id:
            return

        known = await self._existing_node_ids(rows_by_id)
        fresh = [row for row in rows_by_id.values() if row["id"] not in known]
        pending = [row for row in rows_by_id.values() if row["id"] in known]

        if fresh and not await self._copy_rows(
            self._node_table, ("id", "label", "payload"), fresh
        ):
            pending = list(rows_by_id.values())

        if not pending:
            return

        for chunk in BatchGenerator(pending, self._batch_size).get_batches():
            await self._run(
                f"""
                UNWIND $rows AS row
                MERGE (n:{self._node_table} {{id: row.id}})
                SET n.label = row.label, n.payload = row.payload
                """,
                {"rows": chunk},
            )

    @override
    @_serialized_write
    async def delete_nodes(self, node_ids: list[str]) -> None:
        """
        Delete nodes and incident edges. Missing IDs are tolerated.

        :param node_ids: Node identifiers to remove.
        """
        if not node_ids:
            return

        for chunk in BatchGenerator(list(dict.fromkeys(node_ids)), self._batch_size).get_batches():
            await self._run(
                f"""
                UNWIND $ids AS wanted
                MATCH (n:{self._node_table})
                WHERE n.id = wanted
                DETACH DELETE n
                """,
                {"ids": chunk},
            )

    @override
    async def get_edges(self, edge_specs: list[EdgeSpec]) -> list[list[EdgeT]]:
        """
        Fetch edges by specs, one result list per spec.

        :param edge_specs: Edge specs.
        :return: Edge groups aligned with input specs.
        """
        if not edge_specs:
            return []

        rows = [
            {"i": index, "subject_id": subject_id, "object_id": object_id, "edge_id": edge_id}
            for index, (subject_id, object_id, edge_id) in enumerate(edge_specs)
        ]
        found: dict[int, list[EdgeT]] = defaultdict(list)

        results = await self._query_batches(
            f"""
            UNWIND $rows AS row
            MATCH (s:{self._node_table} {{id: row.subject_id}})
                  -[r:{self._edge_table}]->
                  (t:{self._node_table} {{id: row.object_id}})
            WHERE row.edge_id IS NULL OR r.id = row.edge_id
            RETURN row.i AS i, s.id AS subject_id, t.id AS object_id,
                   r.id AS id, r.payload AS payload
            """,
            [{"rows": chunk} for chunk in BatchGenerator(rows, self._batch_size).get_batches()],
            ("i", "subject_id", "object_id", "id", "payload"),
        )
        for result in results:
            for row in result:
                found[row["i"]].append(self._edge_from_row(row))

        return [found.get(index, []) for index in range(len(edge_specs))]

    @override
    @_serialized_write
    async def upsert_edges(self, edges: Iterable[EdgeT]) -> None:
        """
        Insert or update directed edges by ``(subject_id, object_id, id)``.

        :param edges: Edges to upsert.
        """
        rows_by_key: dict[tuple[str, str, str | None], dict[str, Any]] = {}
        for edge in edges:
            rows_by_key[(edge.subject_id, edge.object_id, edge.id)] = {
                "subject_id": edge.subject_id,
                "object_id": edge.object_id,
                "id": edge.id,
                "label": edge.get_label(),
                "payload": self._payload_from_record(edge, {"id", "subject_id", "object_id"}),
            }

        if not rows_by_key:
            return

        known = await self._existing_node_ids(
            node_id
            for subject_id, object_id, _ in rows_by_key
            for node_id in (subject_id, object_id)
        )
        rows = [
            row for row in rows_by_key.values()
            if row["subject_id"] in known and row["object_id"] in known
        ]

        if not rows:
            return

        # An edge keyed on a missing id cannot be classified, so the whole batch
        # takes the ordinary path rather than risking a duplicate.
        keyed = all(row["id"] is not None for row in rows)
        fresh: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = rows

        if keyed:
            if await self._edge_table_is_empty():
                fresh, pending = rows, []
            else:
                known = await self._existing_edge_keys(rows)
                fresh = [
                    row for row in rows
                    if (row["subject_id"], row["object_id"], row["id"]) not in known
                ]
                pending = [
                    row for row in rows
                    if (row["subject_id"], row["object_id"], row["id"]) in known
                ]

        if fresh and not await self._copy_rows(
            self._edge_table,
            ("subject_id", "object_id", "id", "label", "payload"),
            fresh,
        ):
            pending = rows

        if not pending:
            return

        for chunk in BatchGenerator(pending, self._batch_size).get_batches():
            await self._run(
                f"""
                UNWIND $rows AS row
                MERGE (s:{self._node_table} {{id: row.subject_id}})
                MERGE (t:{self._node_table} {{id: row.object_id}})
                MERGE (s)-[r:{self._edge_table} {{id: row.id}}]->(t)
                SET r.label = row.label, r.payload = row.payload
                """,
                {"rows": chunk},
            )

    @override
    @_serialized_write
    async def delete_edges(self, edge_specs: list[EdgeSpec]) -> None:
        """
        Delete edges by spec. Missing edges are tolerated.

        A spec whose ``edge_id`` is ``None`` removes every edge between the pair.

        Endpoints are bound in the pattern for the same reason as in
        :meth:`get_edges`.

        :param edge_specs: Edge specs to remove.
        """
        if not edge_specs:
            return

        rows = [
            {"subject_id": subject_id, "object_id": object_id, "edge_id": edge_id}
            for subject_id, object_id, edge_id in edge_specs
        ]

        for chunk in BatchGenerator(rows, self._batch_size).get_batches():
            await self._run(
                f"""
                UNWIND $rows AS row
                MATCH (s:{self._node_table} {{id: row.subject_id}})
                      -[r:{self._edge_table}]->
                      (t:{self._node_table} {{id: row.object_id}})
                WHERE row.edge_id IS NULL OR r.id = row.edge_id
                DELETE r
                """,
                {"rows": chunk},
            )

    @override
    async def get_all_edges_for_nodes(self, node_ids: list[str]) -> list[list[EdgeT]]:
        """
        Fetch all incident edges for each node ID.

        :param node_ids: Node identifiers.
        :return: Incident edge groups aligned with input IDs.
        """
        if not node_ids:
            return []

        rows = [{"i": index, "node_id": node_id} for index, node_id in enumerate(node_ids)]
        found: dict[int, list[EdgeT]] = defaultdict(list)
        seen: dict[int, set[tuple[str, str, str]]] = defaultdict(set)

        columns = ("i", "subject_id", "object_id", "id", "payload")
        projection = (
            "RETURN row.i AS i, s.id AS subject_id, t.id AS object_id, "
            "r.id AS id, r.payload AS payload"
        )
        outgoing = f"""
            UNWIND $rows AS row
            MATCH (s:{self._node_table} {{id: row.node_id}})-[r:{self._edge_table}]->(t:{self._node_table})
            {projection}
        """
        incoming = f"""
            UNWIND $rows AS row
            MATCH (s:{self._node_table})-[r:{self._edge_table}]->(t:{self._node_table} {{id: row.node_id}})
            {projection}
        """

        chunk_params = [
            {"rows": chunk} for chunk in BatchGenerator(rows, self._batch_size).get_batches()
        ]
        # One direction at a time, chunks within it run together. Keeping the
        # directions ordered means a self-loop is always first seen as outgoing,
        # so which of the two identical rows survives deduplication does not
        # depend on which query happened to finish first.
        for query in (outgoing, incoming):
            for result in await self._query_batches(query, chunk_params, columns):
                for row in result:
                    key = (row["subject_id"], row["object_id"], row["id"])
                    if key in seen[row["i"]]:
                        continue
                    seen[row["i"]].add(key)
                    found[row["i"]].append(self._edge_from_row(row))

        return [found.get(index, []) for index in range(len(node_ids))]

    @override
    async def get_all_nodes(self) -> list[NodeT]:
        """
        Fetch all nodes stored by the adapter.

        :return: Stored nodes.
        """
        rows = await self._query(
            f"""
            MATCH (n:{self._node_table})
            RETURN n.id AS id, n.payload AS payload
            """,
            None,
            ("id", "payload"),
        )
        return [self._node_from_row(row) for row in rows]

    @override
    async def get_all_edges(self) -> list[EdgeT]:
        """
        Fetch all edges stored by the adapter.

        :return: Stored edges.
        """
        rows = await self._query(
            f"""
            MATCH (s:{self._node_table})-[r:{self._edge_table}]->(t:{self._node_table})
            RETURN s.id AS subject_id, t.id AS object_id, r.id AS id, r.payload AS payload
            """,
            None,
            ("subject_id", "object_id", "id", "payload"),
        )
        return [self._edge_from_row(row) for row in rows]

    async def close(self) -> None:
        """
        Close the async connection and then the database itself.
        """
        close = getattr(self._connection, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

        database_close = getattr(self._database, "close", None)
        if database_close is not None:
            database_close()

        self._release_path()
