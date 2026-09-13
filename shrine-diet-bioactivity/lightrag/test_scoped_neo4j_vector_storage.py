"""Unit tests for ScopedNeo4JVectorStorage.

The Neo4j driver is mocked end-to-end. Live integration is exercised
separately by an opt-in pytest mark; these tests only verify Cypher
construction, scope filtering, idempotency contracts, and the
embedding pipeline plumbing.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from scoped_neo4j_vector_storage import (  # type: ignore[import-not-found]
    ScopedNeo4JVectorStorage,
    _index_name_for,
    _vector_label_for,
)

pytestmark = [pytest.mark.unit]


# ─── Fixtures ─────────────────────────────────────────────────────────────


def _make_embedding_func(dim: int = 4):
    """Build a fake EmbeddingFunc that returns deterministic vectors."""
    fake = MagicMock()
    fake.embedding_dim = dim
    fake.model_name = "fake-embed"

    async def _call(texts):
        return np.array([[float(i) / 10] * dim for i in range(len(texts))])

    fake.side_effect = _call
    fake.__call__ = _call
    # Make `await fake(...)` work — needs to be awaitable
    fake_callable = AsyncMock(side_effect=_call)
    return fake_callable, dim


@pytest.fixture
def storage():
    """A storage instance with embedding_func, mocked driver injected as needed."""
    embed, dim = _make_embedding_func()
    embed.embedding_dim = dim
    embed.model_name = "fake-embed"

    s = ScopedNeo4JVectorStorage(
        namespace="entities",
        workspace="unified_diet_kg",
        embedding_func=embed,
        global_config={
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        meta_fields={"entity_name", "content"},
    )
    return s


def _make_async_session_mock():
    """Build an AsyncGraphDatabase-driver mock that records run() calls."""
    session = MagicMock()
    session.run = AsyncMock()
    session.run.return_value = MagicMock(
        single=AsyncMock(return_value=None),
        __aiter__=lambda s: iter([]),
    )
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    driver.close = AsyncMock()
    return driver, session


# ─── Pure helpers ─────────────────────────────────────────────────────────


def test_vector_label_for_known_namespaces():
    assert _vector_label_for("entities") == "VectorEntity"
    assert _vector_label_for("relationships") == "VectorRelationship"
    assert _vector_label_for("chunks") == "VectorChunk"


def test_vector_label_for_unknown_namespace_falls_back():
    label = _vector_label_for("custom-thing")
    assert label.startswith("Vector")


def test_index_name_is_workspace_namespace_keyed():
    n = _index_name_for("Tenant ACME", "entities")
    assert "tenant_acme" in n
    assert "entities" in n
    assert n.startswith("vec_")


# ─── Class init ────────────────────────────────────────────────────────────


def test_post_init_resolves_label_index_and_dim(storage):
    assert storage._vector_label == "VectorEntity"
    assert storage._index_name == "vec_unified_diet_kg_entities"
    assert storage._embedding_dim == 4
    assert storage._workspace_label == "unified_diet_kg"


def test_post_init_requires_cosine_threshold():
    embed = AsyncMock()
    embed.embedding_dim = 4
    embed.model_name = "x"
    with pytest.raises(ValueError, match="cosine_better_than_threshold"):
        ScopedNeo4JVectorStorage(
            namespace="entities",
            workspace="ws",
            embedding_func=embed,
            global_config={"vector_db_storage_cls_kwargs": {}},
            meta_fields=set(),
        )


# ─── initialize: vector index DDL ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_creates_vector_index_with_dim(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver
    await storage.initialize()
    cypher = session.run.call_args.args[0]
    kwargs = session.run.call_args.kwargs
    assert "CREATE VECTOR INDEX" in cypher
    assert "IF NOT EXISTS" in cypher
    assert "vec_unified_diet_kg_entities" in cypher
    assert ":`VectorEntity`" in cypher
    assert "ON (n.embedding)" in cypher
    assert kwargs["dim"] == 4


@pytest.mark.asyncio
async def test_initialize_recreates_stale_dim_index_when_opted_in(storage, monkeypatch):
    # On the INGEST path (VECTOR_INDEX_ALLOW_RECREATE set), a pre-existing index at a
    # DIFFERENT dim must be dropped + recreated, not left in place by IF NOT EXISTS
    # (the shadow-index bug: a 2048 index silently shadowed 1024 vectors so queryNodes
    # failed on a dim mismatch, measured 2026-09-11).
    monkeypatch.setenv("VECTOR_INDEX_ALLOW_RECREATE", "1")
    driver, session = _make_async_session_mock()
    session.run.return_value.single = AsyncMock(
        return_value={"options": {"indexConfig": {"vector.dimensions": 2048}}}
    )
    storage._driver = driver
    await storage.initialize()
    calls = [c.args[0] for c in session.run.call_args_list]
    assert any("SHOW VECTOR INDEXES" in c for c in calls)
    assert any("DROP INDEX" in c and "vec_unified_diet_kg_entities" in c for c in calls)
    assert any("CREATE VECTOR INDEX" in c for c in calls)


@pytest.mark.asyncio
async def test_initialize_refuses_to_drop_valid_index_on_read_path(storage, monkeypatch):
    # WITHOUT the ingest opt-in (the deployed gateway's read/boot path), a dim mismatch
    # must FAIL LOUD, never DROP — dropping there would destroy a valid production index
    # for correctly-configured readers because the READER is misconfigured (design #1).
    monkeypatch.delenv("VECTOR_INDEX_ALLOW_RECREATE", raising=False)
    driver, session = _make_async_session_mock()
    session.run.return_value.single = AsyncMock(
        return_value={"options": {"indexConfig": {"vector.dimensions": 2048}}}
    )
    storage._driver = driver
    with pytest.raises(RuntimeError, match="Refusing to DROP"):
        await storage.initialize()
    calls = [c.args[0] for c in session.run.call_args_list]
    assert not any("DROP INDEX" in c for c in calls)  # nothing destroyed


@pytest.mark.asyncio
async def test_initialize_keeps_matching_dim_index(storage):
    # An existing index at the SAME dim must NOT be needlessly dropped.
    driver, session = _make_async_session_mock()
    session.run.return_value.single = AsyncMock(
        return_value={"options": {"indexConfig": {"vector.dimensions": 4}}}
    )
    storage._driver = driver
    await storage.initialize()
    calls = [c.args[0] for c in session.run.call_args_list]
    assert not any("DROP INDEX" in c for c in calls)
    assert any("CREATE VECTOR INDEX" in c for c in calls)


# ─── upsert ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_writes_embedding_and_scope(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver

    await storage.upsert({
        "id1": {"content": "hello world", "entity_name": "Hello"},
        "id2": {"content": "another", "entity_name": "Another"},
    })

    # session.run was called once with the bulk UNWIND
    assert session.run.call_count >= 1
    cypher = session.run.call_args_list[0].args[0]
    rows = session.run.call_args_list[0].kwargs["rows"]

    assert "MERGE (n:`unified_diet_kg`:`VectorEntity`" in cypher
    assert "n.embedding = row.embedding" in cypher
    assert "n.scope = 'shared'" in cypher
    assert "n.created_at = row.created_at" in cypher
    # Both rows present, with vector + meta
    assert len(rows) == 2
    assert all("embedding" in r and "id" in r and "meta" in r for r in rows)
    assert rows[0]["meta"]["entity_name"] == "Hello"


@pytest.mark.asyncio
async def test_upsert_empty_data_is_noop(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver
    await storage.upsert({})
    session.run.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_logs_and_returns_on_dim_mismatch(storage, caplog):
    """When embedding count differs from id count, skip rather than corrupt."""
    driver, session = _make_async_session_mock()
    storage._driver = driver

    # Substitute embedding func that returns 1 vector for 2 ids
    async def bad_embed(texts):
        return np.array([[0.1, 0.1, 0.1, 0.1]])

    storage.embedding_func = bad_embed
    storage.embedding_func.embedding_dim = 4

    await storage.upsert({"a": {"content": "x"}, "b": {"content": "y"}})
    # Cypher run should NOT have been called
    session.run.assert_not_called()


# ─── query ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_uses_native_vector_index_call(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver

    rec = {"id": "id1", "score": 0.95, "props": {"id": "id1", "entity_name": "X", "embedding": [0.1, 0.1, 0.1, 0.1]}}

    async def aiter():
        yield rec

    result = MagicMock()
    result.__aiter__ = lambda s: aiter()
    session.run.return_value = result

    out = await storage.query("hello", top_k=5)
    cypher = session.run.call_args.args[0]
    kwargs = session.run.call_args.kwargs

    assert "db.index.vector.queryNodes" in cypher
    assert "node.scope IN $scope_filter" in cypher
    assert "score >= $threshold" in cypher
    assert kwargs["k"] == 5
    assert kwargs["scope_filter"] == ["shared"]
    assert kwargs["threshold"] == 0.2

    # Result strips embedding to keep responses small
    assert len(out) == 1
    assert "embedding" not in out[0]
    assert out[0]["id"] == "id1"
    assert out[0]["distance"] == 0.95


@pytest.mark.asyncio
async def test_query_uses_provided_embedding_when_given(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver

    async def empty_aiter():
        if False:
            yield  # never executes — empty async generator

    class FakeResult:
        def __aiter__(self):
            return empty_aiter()

    session.run.return_value = FakeResult()

    pre = [0.5, 0.5, 0.5, 0.5]
    await storage.query("ignored", top_k=3, query_embedding=pre)
    kwargs = session.run.call_args.kwargs
    # If pre-embedding is honored, embed func should NOT have been called
    # (we'd see it in side_effect tracking — using the real fake from fixture)
    assert kwargs["emb"] == pre


# ─── get_by_id / get_by_ids / get_vectors_by_ids ──────────────────────────


@pytest.mark.asyncio
async def test_get_by_id_strips_embedding(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver
    rec = MagicMock()
    rec.__getitem__ = lambda s, k: {"props": {"id": "x", "embedding": [0.1] * 4, "entity_name": "X"}}[k]
    session.run.return_value.single = AsyncMock(return_value=rec)

    out = await storage.get_by_id("x")
    assert out is not None
    assert "embedding" not in out
    assert out["entity_name"] == "X"


@pytest.mark.asyncio
async def test_get_by_id_returns_none_on_miss(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver
    session.run.return_value.single = AsyncMock(return_value=None)
    out = await storage.get_by_id("missing")
    assert out is None


@pytest.mark.asyncio
async def test_get_vectors_by_ids_returns_id_to_vector_dict(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver

    async def aiter():
        yield {"id": "a", "embedding": [0.1, 0.2, 0.3, 0.4]}
        yield {"id": "b", "embedding": [0.5, 0.6, 0.7, 0.8]}

    class FakeResult:
        def __aiter__(self):
            return aiter()

    session.run.return_value = FakeResult()
    out = await storage.get_vectors_by_ids(["a", "b"])
    assert out == {"a": [0.1, 0.2, 0.3, 0.4], "b": [0.5, 0.6, 0.7, 0.8]}


# ─── delete ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_emits_match_delete(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver
    await storage.delete(["a", "b"])
    cypher = session.run.call_args.args[0]
    assert "DELETE n" in cypher
    assert ":`VectorEntity`" in cypher


@pytest.mark.asyncio
async def test_delete_entity_relation_is_noop_for_entities_namespace(storage):
    """delete_entity_relation only acts on relationship-namespace storages."""
    driver, session = _make_async_session_mock()
    storage._driver = driver
    # storage.namespace == "entities" → label is VectorEntity, not VectorRelationship
    await storage.delete_entity_relation("Ginger")
    session.run.assert_not_called()


@pytest.mark.asyncio
async def test_delete_entity_relation_runs_for_relationships_namespace():
    embed, dim = _make_embedding_func()
    embed.embedding_dim = dim
    embed.model_name = "fake-embed"
    s = ScopedNeo4JVectorStorage(
        namespace="relationships",
        workspace="ws",
        embedding_func=embed,
        global_config={
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        meta_fields=set(),
    )
    driver, session = _make_async_session_mock()
    s._driver = driver
    await s.delete_entity_relation("Ginger")
    cypher = session.run.call_args.args[0]
    kwargs = session.run.call_args.kwargs
    assert ":`VectorRelationship`" in cypher
    assert kwargs["sw"] == "Ginger--"
    assert kwargs["ew"] == "--Ginger"
    assert kwargs["mid"] == "-Ginger-"


# ─── lifecycle ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_index_done_callback_is_noop(storage):
    """Neo4j auto-persists; the callback is a no-op returning None per base contract."""
    assert await storage.index_done_callback() is None


@pytest.mark.asyncio
async def test_drop_deletes_nodes_and_index(storage):
    driver, session = _make_async_session_mock()
    storage._driver = driver
    out = await storage.drop()
    assert out["status"] == "ok"
    # Two cypher calls: DELETE n, then DROP INDEX
    cyphers = [c.args[0] for c in session.run.call_args_list]
    assert any("DELETE n" in c for c in cyphers)
    assert any("DROP INDEX" in c for c in cyphers)


@pytest.mark.asyncio
async def test_finalize_closes_driver(storage):
    driver, _ = _make_async_session_mock()
    storage._driver = driver
    await storage.finalize()
    driver.close.assert_called_once()
    assert storage._driver is None
