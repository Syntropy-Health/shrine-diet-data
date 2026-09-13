"""
Unified Diet Knowledge Graph — LightRAG Ingestion Script.

Extracts entities and relationships from the unified herbal_botanicals.db
SQLite database and loads them into LightRAG via ainsert_custom_kg(),
bypassing LLM extraction entirely (zero API cost for structured data).

Entity types: Herb, Compound, Food, Target, Disease, Symptom
Relationship types: CONTAINS_COMPOUND, FOUND_IN_FOOD, TARGETS_PROTEIN,
                     ASSOCIATED_WITH_DISEASE, TREATS_SYMPTOM

Usage:
    python ingest_unified.py --config local --dry-run
    python ingest_unified.py --config local --batch-size 500
    python ingest_unified.py --config production --batch-size 200
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv

from entity_schema import (
    DESCRIPTION_GENERATORS,
    ENTITY_TYPES,
    QUERY_BUILDERS,
    RELATIONSHIP_TYPES,
    describe_relationship,
    safe_label,
)
from extra_sources import (
    EXTRA_ENTITY_ADAPTERS,
    extract_extra_entities,
    extract_herb2_relationships,
)

# Side-effect import: registers ScopedNeo4JStorage / ScopedNeo4JVectorStorage
# into upstream LightRAG's storage whitelist before any LightRAG() call.
# Without this, LightRAG.__post_init__ rejects the Scoped* storages named in
# config_*.env. See Issue #13.
import lightrag_init  # noqa: F401

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / ".." / "data_local" / "herbal_botanicals.db"


# ---------------------------------------------------------------------------
# Embedding — OpenAI-compatible (OpenRouter) without base64 encoding_format
# ---------------------------------------------------------------------------


# HTTP status / OpenRouter error codes worth retrying — transient upstream
# failures (rate limit, gateway, Cloudflare 5xx). 4xx auth/bad-request codes
# are NOT here: retrying those just wastes time.
_TRANSIENT_HTTP_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504, 520, 522, 524})


async def _openai_compat_embed(
    texts: list[str],
    model: str,
    base_url: str,
    api_key: str | None,
    embedding_dim: int,
    max_retries: int = 5,
):
    """Call an OpenAI-compatible /embeddings endpoint with float encoding.

    lightrag's bundled openai_embed forces ``encoding_format="base64"``,
    which OpenRouter rejects. This calls the endpoint directly, requests
    default (float) encoding, and returns results in input order.

    Transient failures (HTTP 429/5xx, network errors, and OpenRouter's
    200-with-error-body for upstream 5xx) are retried with exponential
    backoff. Without this, one transient blip silently drops a whole
    ingest batch — its entities never get embedded or written.
    """
    import asyncio as _asyncio

    import httpx
    import numpy as np

    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "input": texts, "dimensions": embedding_dim}

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()

            data = body.get("data")
            if isinstance(data, list):
                # A missing `index` must fail loudly — a default sort key
                # would collapse items to 0 and misalign embeddings.
                if any("index" not in d for d in data):
                    raise RuntimeError(f"embeddings response item missing 'index': {data[:2]}")
                ordered = sorted(data, key=lambda d: d["index"])
                return np.array([d["embedding"] for d in ordered], dtype=np.float32)

            # No `data`: OpenRouter returns HTTP 200 with {"message":...,
            # "code":520} when its upstream fails. Retry transient codes.
            err = body.get("error", body)
            code = err.get("code") if isinstance(err, dict) else None
            if code in _TRANSIENT_HTTP_CODES and attempt < max_retries:
                last_err = RuntimeError(f"transient embeddings error: {err}")
            else:
                raise RuntimeError(f"embeddings endpoint returned no data: {err}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _TRANSIENT_HTTP_CODES and attempt < max_retries:
                last_err = exc
            else:
                raise
        except httpx.RequestError as exc:
            # Network-level failure (timeout, connection reset) — transient.
            if attempt < max_retries:
                last_err = exc
            else:
                raise

        await _asyncio.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"embeddings failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the SQLite database."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def fetch_all(conn: sqlite3.Connection, query: str, limit: int | None = None) -> list[dict]:
    """Fetch all rows as dicts, with optional limit."""
    if limit is not None:
        query = f"{query} LIMIT {limit}"
    cursor = conn.execute(query)
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def extract_entities(
    conn: sqlite3.Connection,
    entity_type: str,
    max_count: int | None = None,
) -> list[dict]:
    """Extract entities of a given type from SQLite as LightRAG entity dicts."""
    spec = ENTITY_TYPES[entity_type]
    describe = DESCRIPTION_GENERATORS[entity_type]

    # Tenant-only entity types have no SQLite source
    if spec.get("source_table") is None and "query_builder" not in spec:
        return []

    # Check source table exists (Disease is aggregated, has no single table)
    if spec["source_table"] and not table_exists(conn, spec["source_table"]):
        print(f"  ⚠ Table '{spec['source_table']}' not found, skipping {entity_type}")
        return []

    # Resolve query — may be static string or dynamic builder
    query = spec.get("query")
    if query is None and "query_builder" in spec:
        builder_name = spec["query_builder"]
        query = QUERY_BUILDERS[builder_name](conn)
    if query is None:
        print(f"  ⚠ No query for {entity_type}, skipping")
        return []

    rows = fetch_all(conn, query, limit=max_count)
    entities = []
    seen_names: set[str] = set()

    for row in rows:
        name_field = spec["name_field"]
        entity_name = str(row.get(name_field, "")).strip()
        if not entity_name or entity_name in seen_names:
            continue
        seen_names.add(entity_name)

        description = describe(row)
        entities.append({
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "scope": "shared",
            "source_id": f"sqlite-{entity_type.lower()}",
        })

    return entities


# ---------------------------------------------------------------------------
# Relationship extraction
# ---------------------------------------------------------------------------


def extract_relationships(
    conn: sqlite3.Connection,
    rel_type: str,
    max_count: int | None = None,
) -> list[dict]:
    """Extract relationships of a given type from SQLite as LightRAG edge dicts."""
    spec = RELATIONSHIP_TYPES[rel_type]

    # Tenant-only relationship types have no SQLite source
    if spec.get("source_table") is None:
        return []

    if not table_exists(conn, spec["source_table"]):
        print(f"  ⚠ Table '{spec['source_table']}' not found, skipping {rel_type}")
        return []

    rows = fetch_all(conn, spec["query"], limit=max_count)
    relationships = []

    for row in rows:
        src = str(row.get("src_name", "")).strip()
        tgt = str(row.get("tgt_name", "")).strip()
        if not src or not tgt:
            continue

        description, keywords = describe_relationship(rel_type, row)
        relationships.append({
            "src_id": src,
            "tgt_id": tgt,
            "description": description,
            "keywords": keywords,
            "weight": 1.0,
            "scope": "shared",
            "source_id": f"sqlite-{rel_type.lower()}",
        })

    return relationships


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------


async def ingest_batch(
    rag,
    entities: list[dict],
    relationships: list[dict],
    batch_label: str,
    source_prefix: str = "batch",
) -> None:
    """Insert a batch of entities and relationships into LightRAG.

    ``source_prefix`` is used as the chunk's ``source_id`` prefix so
    that downstream queries can attribute nodes / edges to a specific
    upstream data source (e.g. ``symmap:entities-001``,
    ``herb2:relationships-003``). LightRAG overwrites the per-entity
    source_id with this chunk source_id during ainsert_custom_kg, so
    this is the only surviving attribution channel.
    """
    source_id = f"{source_prefix}:{batch_label}"
    entity_names = [e["entity_name"] for e in entities[:20]]
    chunk_content = (
        f"Batch '{batch_label}' containing {len(entities)} entities "
        f"and {len(relationships)} relationships. "
        f"Sample entities: {', '.join(entity_names)}"
    )

    # Both source_id and file_path are the only attribution channels
    # LightRAG passes through to Neo4j. source_id gets rewritten to the
    # MD5-hashed chunk_id, but file_path is preserved verbatim — we use
    # it to encode the upstream data source (``symmap``, ``herb2``,
    # ``duke``, ``hdi-safe-50``) so downstream queries can attribute.
    custom_kg = {
        "chunks": [
            {
                "content": chunk_content,
                "source_id": source_id,
                "file_path": source_prefix,
            }
        ],
        "entities": [
            {**e, "source_id": source_id, "file_path": source_prefix}
            for e in entities
        ],
        "relationships": [
            {**r, "source_id": source_id, "file_path": source_prefix}
            for r in relationships
        ],
    }

    await rag.ainsert_custom_kg(custom_kg)


def batch_items(items: list, batch_size: int) -> list[list]:
    """Split a list into batches."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest unified diet KG into LightRAG")
    parser.add_argument(
        "--config",
        choices=["local", "production"],
        default="local",
        help="Config profile (default: local)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing")
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="Bypass the additive-only guard (#233b). Only for an intentional "
        "replacing re-ingest — a shrink otherwise kills the six chain tools.",
    )
    parser.add_argument("--max-herbs", type=int, default=None, help="Max herbs to ingest")
    parser.add_argument("--max-compounds", type=int, default=None, help="Max compounds to ingest")
    parser.add_argument("--max-foods", type=int, default=None, help="Max foods to ingest")
    parser.add_argument("--max-relationships", type=int, default=None, help="Max relationships per type")
    parser.add_argument("--batch-size", type=int, default=500, help="Entities per batch (default: 500)")
    parser.add_argument(
        "--herb2-experimental-cap",
        type=int,
        default=None,
        help=(
            "Max experimental-tier HERB 2.0 edges to ingest "
            "(default: 50000; 0 = unlimited; clinical tier always full)"
        ),
    )
    parser.add_argument(
        "--skip-extra-sources",
        action="store_true",
        help="Skip SymMap + HERB 2.0 (legacy Duke-only ingest)",
    )
    parser.add_argument(
        "--max-extra-per-table",
        type=int,
        default=None,
        help="Max rows to extract per SymMap/HERB 2.0 entity table (default: unlimited)",
    )
    parser.add_argument(
        "--only-entities",
        default=None,
        help=(
            "Comma-separated entity type names to include "
            "(e.g. 'BioactivityEvidence' for the Phase 1 bridge layer). "
            "Default: all entity types."
        ),
    )
    parser.add_argument(
        "--only-relationships",
        default=None,
        help=(
            "Comma-separated relationship type names to include "
            "(e.g. 'HAS_EVIDENCE,EVIDENCE_FOR_TARGET'). "
            "Default: all relationship types."
        ),
    )
    args = parser.parse_args()

    only_entities = (
        {s.strip() for s in args.only_entities.split(",") if s.strip()}
        if args.only_entities
        else None
    )
    only_relationships = (
        {s.strip() for s in args.only_relationships.split(",") if s.strip()}
        if args.only_relationships
        else None
    )

    # 1. Load Aura creds from gitignored .env at project root.
    project_env = SCRIPT_DIR.parent / ".env"
    if project_env.exists():
        load_dotenv(project_env)
    # 2. Load profile config but DON'T override (config_local.env has
    #    ``NEO4J_URI=${NEO4J_URI}`` placeholders that would otherwise
    #    stomp the real values from .env).
    config_path = SCRIPT_DIR / f"config_{args.config}.env"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return
    load_dotenv(config_path, override=False)

    # Check DB
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run: cd mcp-herbal-botanicals && make setup")
        return

    conn = sqlite3.connect(str(DB_PATH))
    print(f"Connected to {DB_PATH}")
    print(f"Config: {args.config}")
    print(f"Dry run: {args.dry_run}")
    print()

    # --- Extract entities ---
    limits = {
        "Herb": args.max_herbs,
        "Compound": args.max_compounds,
        "Food": args.max_foods,
        "Target": None,
        "Disease": None,
        "Symptom": None,
    }

    all_entities: dict[str, list[dict]] = {}
    total_entities = 0
    for entity_type in ENTITY_TYPES:
        if only_entities is not None and entity_type not in only_entities:
            continue
        print(f"Extracting {entity_type} entities...")
        entities = extract_entities(conn, entity_type, max_count=limits.get(entity_type))
        all_entities[entity_type] = entities
        total_entities += len(entities)
        print(f"  → {len(entities)} {entity_type} entities")

    # --- Extract extra entities (SymMap v2 + HERB 2.0) ---
    if not args.skip_extra_sources:
        for adapter in EXTRA_ENTITY_ADAPTERS:
            label = f"{adapter['source']}/{adapter['table']}"
            print(f"Extracting {label} ({adapter['entity_type']}) entities...")
            extras = extract_extra_entities(
                conn, adapter, max_count=args.max_extra_per_table
            )
            bucket = all_entities.setdefault(adapter["entity_type"], [])
            # Dedupe across sources by entity_name — first occurrence wins,
            # so Duke entities (loaded first) keep their richer descriptions.
            existing = {e["entity_name"] for e in bucket}
            new_entities = [e for e in extras if e["entity_name"] not in existing]
            bucket.extend(new_entities)
            total_entities += len(new_entities)
            print(
                f"  → {len(extras)} {label} rows, "
                f"{len(new_entities)} new (after dedupe)"
            )

    # --- Extract relationships ---
    all_relationships: dict[str, list[dict]] = {}
    total_relationships = 0
    for rel_type in RELATIONSHIP_TYPES:
        if only_relationships is not None and rel_type not in only_relationships:
            continue
        print(f"Extracting {rel_type} relationships...")
        rels = extract_relationships(conn, rel_type, max_count=args.max_relationships)
        all_relationships[rel_type] = rels
        total_relationships += len(rels)
        print(f"  → {len(rels)} {rel_type} edges")

    # --- Extract HERB 2.0 herb→disease edges ---
    if not args.skip_extra_sources:
        print("Extracting HERB 2.0 herb→disease relationships...")
        herb2_rels = extract_herb2_relationships(
            conn, experimental_cap=args.herb2_experimental_cap
        )
        all_relationships.setdefault("ASSOCIATED_WITH_DISEASE", []).extend(herb2_rels)
        total_relationships += len(herb2_rels)

    conn.close()

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"Total entities:       {total_entities}")
    print(f"Total relationships:  {total_relationships}")
    print(f"Batch size:           {args.batch_size}")
    print(f"{'=' * 50}\n")

    if args.dry_run:
        print("DRY RUN — no data written to LightRAG/Neo4j")
        print("\nEntity breakdown:")
        for et, entities in all_entities.items():
            print(f"  {et:15s}: {len(entities):>8,}")
        print("\nRelationship breakdown:")
        for rt, rels in all_relationships.items():
            print(f"  {rt:30s}: {len(rels):>8,}")
        return

    # --- Initialize LightRAG ---
    print("Initializing LightRAG...")
    from functools import partial
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    embedding_binding = os.getenv("EMBEDDING_BINDING", "ollama")
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    embedding_host = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")

    if embedding_binding == "ollama":
        from lightrag.llm.ollama import ollama_embed, ollama_model_complete
        llm_func = ollama_model_complete
        embed_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=embedding_model,
                host=embedding_host,
            ),
        )
    elif embedding_binding in ("vertex", "aistudio", "local"):
        # T4.0 pluggable embedder arm: bge-m3 (local) vs Gemini (vertex/aistudio),
        # one interface behind embedder_adapters.make_embedder. LLM func stays the
        # shared local/openrouter one — embedder and LLM are independent.
        from embedder_adapters import make_embedder, to_embedding_func

        llm_func = lightrag_init.make_llm_func()
        # Pass the ALREADY-RESOLVED model/dim/host so the adapter uses the same values
        # that stamp WorkspaceMeta via assert_workspace_embedding (no dual source of
        # truth). Build the adapter ONCE (a second make_embedder would construct a
        # second genai.Client just to print its name).
        _adapter = make_embedder(
            embedding_binding,
            model=embedding_model,
            dim=embedding_dim,
            base_url=embedding_host,
        )
        embed_func = to_embedding_func(_adapter)
        print(f"Embedder adapter: {_adapter.name}")
    else:
        # Shared openai-binding LLM func (LLM_MODEL from env) with the
        # json_object -> json_schema response_format shim for local
        # OpenAI-compatible servers (LM Studio). See lightrag_init.make_llm_func.
        llm_func = lightrag_init.make_llm_func()
        # NOTE: lightrag's openai_embed hardcodes `encoding_format="base64"`,
        # which OpenRouter rejects ({"error":...}, no `data`) → downstream
        # 'NoneType' object is not iterable. Use a direct OpenRouter call
        # (float encoding) instead. api_key resolved like scoped_server.
        embed_api_key = (
            os.getenv("EMBEDDING_BINDING_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        embed_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=partial(
                _openai_compat_embed,
                model=embedding_model,
                base_url=embedding_host,
                api_key=embed_api_key,
                embedding_dim=embedding_dim,
            ),
        )

    working_dir = os.getenv("WORKING_DIR", "./rag_storage_local")
    os.makedirs(working_dir, exist_ok=True)

    # Read storage config from env (matches config_local.env / config_production.env)
    graph_storage = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    kv_storage = os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
    vector_storage = os.getenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
    doc_status_storage = os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")
    workspace = os.getenv("WORKSPACE", "unified_diet_kg")

    print(f"  Graph storage: {graph_storage}")
    print(f"  Workspace: {workspace}")

    # Router guard (pre-emptive): refuse a local (bge-m3) embedder against a
    # production workspace BEFORE any WorkspaceMeta is written — the deployed
    # gateway cannot reach a local embedder at query time. Only meaningful for
    # the adapter-owned bindings; ollama/native paths are unaffected.
    if embedding_binding in ("vertex", "aistudio", "local", "openai"):
        from embedder_adapters import assert_binding_allowed_for_workspace

        assert_binding_allowed_for_workspace(embedding_binding, workspace)

    # Workspace <-> embedding-space guard: refuse to ingest vectors of a
    # different embedding model/dim into a Neo4j-backed workspace.
    # Case-insensitive so a storage-class rename (Neo4j vs Neo4J) cannot make
    # the guard silently fail-open.
    neo4j_backed = "neo4j" in vector_storage.lower()
    _neo4j_conn = (
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", ""),
    )
    if neo4j_backed:
        lightrag_init.assert_workspace_embedding(
            model=embedding_model,
            dim=embedding_dim,
            neo4j_uri=_neo4j_conn[0],
            user=_neo4j_conn[1],
            password=_neo4j_conn[2],
            workspace=workspace,
        )

    # Additive-only guard (#233b): snapshot the workspace BEFORE writing so a
    # replacing re-ingest (which would delete chain-tool nodes/edges) fails
    # closed at the end. Only meaningful for a persistent Neo4j workspace.
    additive_before = None
    if neo4j_backed and not args.dry_run:
        from neo4j import GraphDatabase

        import additive_guard

        _guard_driver = GraphDatabase.driver(_neo4j_conn[0], auth=_neo4j_conn[1:])
        with _guard_driver.session() as _s:
            additive_before = additive_guard.snapshot_workspace_counts(_s, workspace)
        _guard_driver.close()

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embed_func,
        graph_storage=graph_storage,
        kv_storage=kv_storage,
        vector_storage=vector_storage,
        doc_status_storage=doc_status_storage,
        workspace=workspace,
    )
    await rag.initialize_storages()
    print("LightRAG initialized\n")

    # --- Ingest entities in batches, grouped by data source ---
    # Source attribution survives via the chunk's source_id (e.g.
    # ``symmap:entities-001``). Group by source_prefix derived from
    # each entity's original source_id (e.g. ``symmap:1`` → ``symmap``).
    start_time = time.time()

    def _source_prefix(item: dict) -> str:
        sid = item.get("source_id", "")
        if ":" in sid:
            return sid.split(":", 1)[0]
        if sid.startswith("sqlite-"):
            return "duke"
        return "unknown"

    # Group entities and relationships by source prefix.
    grouped_entities: dict[str, list[dict]] = {}
    for entities in all_entities.values():
        for ent in entities:
            grouped_entities.setdefault(_source_prefix(ent), []).append(ent)

    grouped_relationships: dict[str, list[dict]] = {}
    for rels in all_relationships.values():
        for rel in rels:
            grouped_relationships.setdefault(_source_prefix(rel), []).append(rel)

    total_entity_batches = 0
    total_rel_batches = 0

    # Ingest entities first (nodes must exist before edges)
    for source, entities in grouped_entities.items():
        batches = batch_items(entities, args.batch_size)
        print(f"\n[{source}] ingesting {len(entities)} entities in {len(batches)} batches")
        for i, batch in enumerate(batches):
            label = f"entities-{i + 1:03d}"
            await ingest_batch(rag, batch, [], label, source_prefix=source)
            total_entity_batches += 1

    # Then ingest relationships
    for source, rels in grouped_relationships.items():
        batches = batch_items(rels, args.batch_size)
        print(f"\n[{source}] ingesting {len(rels)} relationships in {len(batches)} batches")
        for i, batch in enumerate(batches):
            label = f"relationships-{i + 1:03d}"
            await ingest_batch(rag, [], batch, label, source_prefix=source)
            total_rel_batches += 1

    entity_batches_count = total_entity_batches
    rel_batches_count = total_rel_batches

    elapsed = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"Ingestion complete in {elapsed:.1f}s")
    print(f"  Entities:       {total_entities:,}")
    print(f"  Relationships:  {total_relationships:,}")
    print(f"  Entity batches: {entity_batches_count}")
    print(f"  Rel batches:    {rel_batches_count}")
    print(f"{'=' * 50}")

    await rag.finalize_storages()

    # Additive-only guard (#233b): compare post-ingest counts to the pre-ingest
    # snapshot. A DECREASE means the ingest replaced rather than added, which
    # kills the six Layer-B chain tools; fail closed unless --allow-shrink.
    # NOTE: this is a post-commit DETECTOR, not a preventer. The after-snapshot
    # runs after finalize_storages(), so a genuinely replacing ingest has already
    # committed by the time SystemExit fires — the exit is an alarm to trigger a
    # restore, not a rollback. The design bet is that the normal ingest path only
    # MERGEs (LightRAG keys entity nodes by id, so typed labels/edges persist and
    # counts never drop); the guard exists to catch the day that assumption breaks.
    if additive_before is not None:
        from neo4j import GraphDatabase

        import additive_guard

        _guard_driver = GraphDatabase.driver(_neo4j_conn[0], auth=_neo4j_conn[1:])
        try:
            with _guard_driver.session() as _s:
                additive_after = additive_guard.snapshot_workspace_counts(_s, workspace)
        finally:
            _guard_driver.close()
        # evaluate_additive computes the diff once, raises SystemExit on a shrink
        # unless --allow-shrink, and otherwise returns the outcome line.
        print(
            additive_guard.evaluate_additive(
                additive_before, additive_after, allow_shrink=args.allow_shrink
            )
        )

    # Post-ingestion: add entity_type as Neo4j labels for visual exploration
    if graph_storage == "Neo4JStorage":
        print("\nAdding entity type labels to Neo4j nodes...")
        try:
            from neo4j import GraphDatabase as Neo4jGD

            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_pass = os.getenv("NEO4J_PASSWORD", "")
            ws = safe_label(workspace)
            with Neo4jGD.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass)) as driver:
                with driver.session() as neo_session:
                    for etype in ENTITY_TYPES:
                        et = safe_label(etype)
                        result = neo_session.run(
                            f"MATCH (n:`{ws}`) WHERE n.entity_type = $etype "
                            f"SET n:`{et}` RETURN COUNT(n) AS count",
                            etype=etype,
                        ).single()
                        print(f"  :{etype} → {result['count']} nodes")
        except ImportError:
            print("  neo4j package not installed — skipping label step")
        except Exception as e:
            print(f"  Warning: could not add labels — {e}")


if __name__ == "__main__":
    asyncio.run(main())
