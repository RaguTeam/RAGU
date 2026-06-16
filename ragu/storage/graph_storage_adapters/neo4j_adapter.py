from __future__ import annotations

from dataclasses import asdict
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
)

from neo4j import AsyncGraphDatabase, AsyncDriver
from typing_extensions import override

from ragu.storage.base_storage import BaseGraphStorage, EdgeSpec
from ragu.storage.types import Node, Edge

NodeT = TypeVar("NodeT", bound=Node)
EdgeT = TypeVar("EdgeT", bound=Edge)


class Neo4jStorage(BaseGraphStorage[NodeT, EdgeT]):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        node_cls: Type[NodeT],
        edge_cls: Type[EdgeT],
        database: str = "neo4j",
        filename: str | None = None,
        **kwargs: Any,
    ):
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._node_cls = node_cls
        self._edge_cls = edge_cls

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
                await session.run(
                    "MERGE (n:NODE {id: $id}) SET n += $props",
                    id=node_id,
                    props=attrs,
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
                        props=edge_data,
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
                        props=edge_data,
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