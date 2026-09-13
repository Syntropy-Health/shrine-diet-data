# Semantic-vector ingestion — RESUMABLE CHECKPOINT (2026-09-11)

**Authoritative checkpoint = Aura** (durable, machine-independent). The local
`WORKING_DIR` cache is preserved at `data_local/rag_storage_semantic_core/` (112K, chunk KV
only — vectors live in Aura). A future session can resume/extend ingestion from here.

## What is ingested (the "bounded core")
Ingested via `ingest_unified.py --config local --max-compounds 1000 --max-herbs 300
--max-foods 300 --max-relationships 3000 --batch-size 200`, throttled `EMBEDDING_FUNC_MAX_ASYNC=1`.

| entities (16,738 total) | count | | relationships (18,000 total) | count |
|---|---|---|---|---|
| Compound | 1,000 (cap) | | CONTAINS_COMPOUND | 3,000 (cap) |
| Herb | 300 (cap) | | FOUND_IN_FOOD | 3,000 (cap) |
| Food | 300 (cap) | | TARGETS_PROTEIN | 3,000 (cap) |
| Target | 4,352 (all) | | TREATS_SYMPTOM | 3,000 (cap) |
| Symptom | 47 (all) | | HAS_EVIDENCE | 3,000 (cap) |
| BioactivityEvidence | 10,739 (all) | | EVIDENCE_FOR_TARGET | 3,000 (cap) |
| Disease | **0** (extractor returned none — gap) | | | |

## Aura state at checkpoint (query to confirm before resuming)
```
VectorEntity 16,738 · VectorChunk 174 · DIRECTED 17,546 · WorkspaceMeta 1 (gemini-embedding-001/1024)
typed graph intact: HAS_EVIDENCE/EVIDENCE_FOR_TARGET 10,739 · TARGETS_PROTEIN 6,465
totals: 199,840 nodes / 221,009 rels   (baseline pre-semantic was 165,981 / 203,463)
```
kg_query verified live (mode=local): retrieves correct curcumin targets (COX-2, PKC, IKK, NFKB, CYP2C9).

## ⛔ CONSTRAINTS that gate a resume
1. **Aura Free ~200k-node ceiling — we are AT it (199,840).** Only ~160 nodes of headroom on Free
   tier. **A meaningful resume/extend requires a PAID Aura tier first.** The typed graph (166k nodes)
   already consumes most of the Free budget; the semantic layer can only be ~34k nodes on Free.
2. **Vertex embedding quota** — MUST run `EMBEDDING_FUNC_MAX_ASYNC=1` (default 8 → 429 RESOURCE_EXHAUSTED),
   or raise the gemini-embedding quota on `syntropy-passport`. Sustained throttled rate ≈ 14 embeds/sec.
3. **Vector index dim = 1024** — indexes were recreated at 1024 (a stale 2048 index had shadowed them).
   If a resume recreates indexes, ensure `EMBEDDING_DIM=1024`. (Latent bug: `ScopedNeo4JVectorStorage`
   uses `CREATE ... IF NOT EXISTS` and does NOT validate an existing index's dim — flagged.)
4. **custom_kg ingestion writes no doc_status skip-ledger** → a resume MERGE-re-embeds the existing set
   (idempotent but re-spends quota/time). Improving this = add doc_status tracking to the custom_kg path.

## RESUME PROCEDURE (after a paid-tier upgrade, or to top up within headroom)
Run `lightrag/run_semantic_ingest.sh` (see below) with RAISED caps, e.g. drop `--max-compounds` /
`--max-relationships` (uncap) or set higher values. The workspace guard binds to gemini/1024;
`ainsert_custom_kg` MERGEs so re-runs are additive/idempotent. Env (production track):
```
WORKSPACE=unified_diet_kg  LIGHTRAG_GRAPH_STORAGE=Neo4JStorage  LIGHTRAG_VECTOR_STORAGE=ScopedNeo4JVectorStorage
EMBEDDING_BINDING=vertex  EMBEDDING_MODEL=gemini-embedding-001  EMBEDDING_DIM=1024  EMBEDDING_FUNC_MAX_ASYNC=1
GOOGLE_CLOUD_PROJECT=syntropy-passport  GOOGLE_CLOUD_LOCATION=us-central1
LLM_BINDING=openai  LLM_BINDING_HOST=http://localhost:1234/v1  LLM_MODEL=google/gemma-4-12b-qat  (local, merge-summarisation only)
NEO4J_* from Infisical 589d1e3b /research/shrine-diet-bioactivity (USER/DB = b7dbceab)
WORKING_DIR=<repo>/shrine-diet-bioactivity/data_local/rag_storage_semantic_core   (preserved cache)
```
**Before resuming, decide the Disease gap** (0 Disease entities ingested) — the extractor returned none;
if disease-side semantic search is wanted, fix the Disease entity extraction first.
