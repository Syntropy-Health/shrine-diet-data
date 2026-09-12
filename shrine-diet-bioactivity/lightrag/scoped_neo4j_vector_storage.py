"""Neo4j-native vector storage for LightRAG (per ADR 0001).

The production target for KG vectors is Aura, not local NanoVectorDB. This
class subclasses :class:`lightrag.base.BaseVectorStorage` and stores vectors
as ``embedding`` properties on synthetic nodes labeled ``:VectorEntity``,
``:VectorRelationship``, or ``:VectorChunk`` (one label per LightRAG
namespace), indexed via Neo4j 5.13+ native vector indexes.

Why a synthetic-node design:

- **Separation of concerns:** The KG entity nodes (``:Herb``, ``:Compound``,
  ...) are owned by the graph ETL pipeline. Mutating them with vector
  payloads from a separate process couples vector lifecycle to graph
  lifecycle — exactly what ADR 0001 chose Aura native vectors to avoid.
  Keeping vector nodes separate lets either side rebuild without disturbing
  the other.
- **Per-namespace indexes:** LightRAG has three vector namespaces
  (``entities``, ``relationships``, ``chunks``); each gets its own
  ``CREATE VECTOR INDEX``. Mixing them on shared labels would fight the
  index planner.
- **Scope honoring:** every vector node carries ``scope='shared'`` and is
  filtered through ``scope_context`` like any other workspace node.

Idempotency: ``upsert`` uses ``MERGE`` keyed on ``(id, workspace)`` so
re-running with the same data is a no-op.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Final, final

import numpy as np

from lightrag.base import BaseVectorStorage
from lightrag.utils import logger

try:
    from scope_context import get_scope_filter  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — fallback when imported standalone
    def get_scope_filter() -> tuple[str, ...]:  # type: ignore[no-redef]
        return ("shared",)


_NAMESPACE_TO_LABEL: Final[dict[str, str]] = {
    # LightRAG's actual namespace strings → our Neo4j labels.
    # Empty fallback uses the namespace string verbatim — defensive only;
    # all production namespaces are in this dict.
    "entities": "VectorEntity",
    "relationships": "VectorRelationship",
    "chunks": "VectorChunk",
}


def _safe_label(s: str) -> str:
    """Match LightRAG's Neo4j workspace-label sanitisation rules."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in s)


def _vector_label_for(namespace: str) -> str:
    """Resolve the Neo4j label for a LightRAG vector namespace."""
    if namespace in _NAMESPACE_TO_LABEL:
        return _NAMESPACE_TO_LABEL[namespace]
    # Defensive: unknown namespace → use sanitized form. Won't happen in practice.
    return f"Vector{_safe_label(namespace).capitalize()}"


def _index_name_for(workspace: str, namespace: str) -> str:
    """Stable, unique-per-(workspace, namespace) vector index name."""
    return f"vec_{_safe_label(workspace).lower()}_{_safe_label(namespace).lower()}"


@final
@dataclass
class ScopedNeo4JVectorStorage(BaseVectorStorage):
    """LightRAG vector storage backed by Neo4j Aura native vector indexes.

    Connection parameters (URI, user, password) come from environment
    variables ``NEO4J_URI`` / ``NEO4J_USERNAME`` / ``NEO4J_PASSWORD`` —
    the same convention used by ``scoped_neo4j_storage.py``. The async
    ``neo4j`` driver is imported lazily so unit tests can monkeypatch it.
    """

    def __post_init__(self):
        self._validate_embedding_func()
        self._driver: Any = None
        self._workspace_label = _safe_label(self.workspace) if self.workspace else "unified_diet_kg"
        self._vector_label = _vector_label_for(self.namespace)
        self._index_name = _index_name_for(self.workspace or "unified_diet_kg", self.namespace)
        self._embedding_dim = self.embedding_func.embedding_dim
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

    # ─── Connection management ─────────────────────────────────────────

    def _get_driver(self) -> Any:
        if self._driver is None:
            from neo4j import AsyncGraphDatabase

            uri = os.environ["NEO4J_URI"]
            user = os.environ["NEO4J_USERNAME"]
            pwd = os.environ["NEO4J_PASSWORD"]
            self._driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
        return self._driver

    async def initialize(self) -> None:
        """Create the per-namespace vector index, HEALING a stale wrong-dim index.

        Neo4j's ``CREATE VECTOR INDEX ... IF NOT EXISTS`` is a NO-OP when an index of
        the same name already exists — *even if that index has a different
        dimensionality*. A stale index (e.g. left by a prior ingest at another
        embedding dim; a dropped-nodes shape leaves its index behind) then silently
        shadows the current vectors: inserts succeed but
        ``db.index.vector.queryNodes`` fails with a dim mismatch (measured 2026-09-11:
        a 2048-dim index shadowed 1024-dim gemini vectors). So detect a dim mismatch
        and drop+recreate rather than trusting IF-NOT-EXISTS.
        """
        driver = self._get_driver()
        create_cypher = (
            f"CREATE VECTOR INDEX {self._index_name} IF NOT EXISTS "
            f"FOR (n:`{self._vector_label}`) ON (n.embedding) "
            "OPTIONS {indexConfig: {"
            "`vector.dimensions`: $dim, "
            "`vector.similarity_function`: 'cosine'"
            "}}"
        )
        async with driver.session() as session:
            result = await session.run(
                "SHOW VECTOR INDEXES YIELD name, options "
                "WHERE name = $name RETURN options AS options",
                name=self._index_name,
            )
            rec = await result.single()
            existing_dim = None
            if rec is not None:
                opts = rec["options"] or {}
                existing_dim = (opts.get("indexConfig") or {}).get("vector.dimensions")
            if existing_dim is not None and int(existing_dim) != int(self._embedding_dim):
                logger.warning(
                    f"[{self.workspace}] Vector index {self._index_name} has STALE dim "
                    f"{existing_dim} != embedder dim {self._embedding_dim}; "
                    f"dropping and recreating so queries do not fail on a dim mismatch."
                )
                await session.run(f"DROP INDEX {self._index_name} IF EXISTS")
            await session.run(create_cypher, dim=self._embedding_dim)
        logger.info(
            f"[{self.workspace}] Vector index {self._index_name} ensured "
            f"({self._vector_label}.embedding, dim={self._embedding_dim})"
        )

    # ─── Writes ────────────────────────────────────────────────────────

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Embed and write vectors. Idempotent MERGE on (id, workspace)."""
        if not data:
            return

        ts = int(time.time())
        ids = list(data.keys())
        contents = [v.get("content", "") for v in data.values()]
        meta = [
            {k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}
            for v in data.values()
        ]

        # Batch embeddings, gather concurrently — same pattern as NanoVectorDBStorage.
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embedding_tasks = [self.embedding_func(b) for b in batches]
        embeddings_lists = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_lists)

        if len(embeddings) != len(ids):
            logger.error(
                f"[{self.workspace}] embedding count mismatch: "
                f"{len(embeddings)} vs {len(ids)} — skipping upsert"
            )
            return

        rows = []
        for i, _id in enumerate(ids):
            rows.append({
                "id": _id,
                "embedding": embeddings[i].tolist(),
                "created_at": ts,
                "meta": meta[i],
            })

        cypher = (
            f"UNWIND $rows AS row "
            f"MERGE (n:`{self._workspace_label}`:`{self._vector_label}` "
            "{id: row.id}) "
            f"SET n.embedding = row.embedding, "
            f"    n.created_at = row.created_at, "
            f"    n.scope = 'shared', "
            f"    n.namespace = $namespace, "
            f"    n += row.meta"
        )

        # Write in batches to avoid oversized transactions.
        BATCH = 500
        driver = self._get_driver()
        async with driver.session() as session:
            for i in range(0, len(rows), BATCH):
                await session.run(
                    cypher,
                    rows=rows[i : i + BATCH],
                    namespace=self.namespace,
                )

    # ─── Reads ─────────────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search, scoped to the active scope filter."""
        if query_embedding is None:
            arr = await self.embedding_func([query])
            query_embedding = arr[0].tolist() if hasattr(arr[0], "tolist") else list(arr[0])

        scope_filter = list(get_scope_filter())
        cypher = (
            f"CALL db.index.vector.queryNodes($idx, $k, $emb) YIELD node, score "
            f"WHERE node.scope IN $scope_filter "
            f"  AND '{self._workspace_label}' IN labels(node) "
            f"  AND score >= $threshold "
            f"RETURN node.id AS id, score, properties(node) AS props "
            f"ORDER BY score DESC"
        )
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(
                cypher,
                idx=self._index_name,
                k=top_k,
                emb=query_embedding,
                scope_filter=scope_filter,
                threshold=self.cosine_better_than_threshold,
            )
            records = [r async for r in result]

        out: list[dict[str, Any]] = []
        for r in records:
            props = dict(r["props"])
            # Strip the embedding from results — caller doesn't need it back
            # and it bloats responses by ~4-16 KB per row.
            props.pop("embedding", None)
            out.append({
                "id": r["id"],
                "distance": r["score"],
                **props,
            })
        return out

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        cypher = (
            f"MATCH (n:`{self._workspace_label}`:`{self._vector_label}` {{id: $id}}) "
            f"RETURN properties(n) AS props"
        )
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(cypher, id=id)
            record = await result.single()
        if record is None:
            return None
        props = dict(record["props"])
        props.pop("embedding", None)
        return props

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        cypher = (
            f"UNWIND $ids AS i "
            f"MATCH (n:`{self._workspace_label}`:`{self._vector_label}` {{id: i}}) "
            f"RETURN properties(n) AS props"
        )
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(cypher, ids=ids)
            records = [r async for r in result]
        out = []
        for r in records:
            props = dict(r["props"])
            props.pop("embedding", None)
            out.append(props)
        return out

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        cypher = (
            f"UNWIND $ids AS i "
            f"MATCH (n:`{self._workspace_label}`:`{self._vector_label}` {{id: i}}) "
            f"RETURN n.id AS id, n.embedding AS embedding"
        )
        driver = self._get_driver()
        async with driver.session() as session:
            result = await session.run(cypher, ids=ids)
            records = [r async for r in result]
        return {r["id"]: list(r["embedding"]) for r in records if r["embedding"] is not None}

    # ─── Deletes ───────────────────────────────────────────────────────

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        cypher = (
            f"UNWIND $ids AS i "
            f"MATCH (n:`{self._workspace_label}`:`{self._vector_label}` {{id: i}}) "
            f"DELETE n"
        )
        driver = self._get_driver()
        async with driver.session() as session:
            await session.run(cypher, ids=ids)

    async def delete_entity(self, entity_name: str) -> None:
        """Delete the vector node whose id matches the entity name.

        For the entities namespace this is the natural primary key. For
        other namespaces it's a no-op unless the caller has a deterministic
        id derived from entity_name (LightRAG handles that translation).
        """
        await self.delete([entity_name])

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete relationship-vector nodes whose source or target is the entity.

        Relationship vector ids in LightRAG are typically of the form
        ``<src>--<tgt>`` or similar. We match prefix and suffix patterns
        that include ``entity_name`` as either endpoint.
        """
        if self._vector_label != "VectorRelationship":
            return
        cypher = (
            f"MATCH (n:`{self._workspace_label}`:`{self._vector_label}`) "
            f"WHERE n.id STARTS WITH $sw OR n.id ENDS WITH $ew OR n.id CONTAINS $mid "
            f"DELETE n"
        )
        driver = self._get_driver()
        async with driver.session() as session:
            await session.run(
                cypher,
                sw=f"{entity_name}--",
                ew=f"--{entity_name}",
                mid=f"-{entity_name}-",
            )

    # ─── Lifecycle ─────────────────────────────────────────────────────

    async def index_done_callback(self) -> None:
        """Neo4j auto-persists. No-op (matches base method signature)."""
        return None

    async def drop(self) -> dict[str, str]:
        """Drop all vector data for this namespace + workspace.

        Removes the synthetic vector nodes and the vector index. Re-creating
        them is a single ``initialize()`` call. Used by LightRAG's reset
        flow and by tests; never invoked in normal operation.
        """
        driver = self._get_driver()
        async with driver.session() as session:
            await session.run(
                f"MATCH (n:`{self._workspace_label}`:`{self._vector_label}`) DELETE n"
            )
            await session.run(f"DROP INDEX {self._index_name} IF EXISTS")
        return {"status": "ok", "namespace": self.namespace}

    async def finalize(self) -> None:
        """Close the driver on shutdown."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None


# ---------------------------------------------------------------------------
# Register with upstream LightRAG's storage-compatibility whitelist and the
# STORAGES import-path map.
#
# LightRAG uses two separate dicts:
#   1. ``STORAGE_IMPLEMENTATIONS`` — verify_storage_implementation() check
#   2. ``STORAGES`` — _get_storage_class() dynamic-import resolution
#
# Both must know about our class for LightRAG(vector_storage=
# "ScopedNeo4JVectorStorage") to succeed without touching the submodule.
# The ``scoped_neo4j_vector_storage`` module must be importable from the
# working directory (i.e. sys.path includes lightrag/).
#
# The try/except guards against upstream renaming or shape changes.
# ---------------------------------------------------------------------------
try:
    from lightrag.kg import (  # type: ignore[import]
        STORAGE_IMPLEMENTATIONS as _STORAGE_IMPLEMENTATIONS,
        STORAGES as _STORAGES,
    )

    # 1. Compatibility whitelist
    _vec = _STORAGE_IMPLEMENTATIONS.get("VECTOR_STORAGE", {})
    _impls = _vec.get("implementations")
    if isinstance(_impls, list) and "ScopedNeo4JVectorStorage" not in _impls:
        _impls.append("ScopedNeo4JVectorStorage")

    # 2. Import-path map — absolute module name so lazy_external_import can
    #    resolve it regardless of which package calls _get_storage_class.
    if "ScopedNeo4JVectorStorage" not in _STORAGES:
        _STORAGES["ScopedNeo4JVectorStorage"] = "scoped_neo4j_vector_storage"
except (ImportError, KeyError, AttributeError):
    # Upstream LightRAG not importable or dict shape changed — tolerate silently.
    pass
