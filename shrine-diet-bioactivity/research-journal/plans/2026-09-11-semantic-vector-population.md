# Semantic-vector population — bounded proof → full-run decision (2026-09-11)

**Status:** plan-valid, principal-approved (1B1 "Over and out" 2026-09-11). Executed as an
ops task by the shrine-diet-bioactivity lane (additive-guarded, pre-snapshot baseline) — NOT
a `/build`+CTO-land feature, consistent with the prior HAS_EVIDENCE / EVIDENCE_FOR_TARGET populations.

## Goal
Populate the LightRAG **semantic (vector) shape** on Aura so `kg_query` (semantic search) works
alongside the existing typed graph + ChEMBL bioactivity edges. Measured start state (2026-09-11):
`VectorEntity/VectorRelationship/VectorChunk/WorkspaceMeta/DIRECTED = 0` — a **fresh population, not
a re-ingest**. Typed graph intact: HAS_EVIDENCE 10,739 · EVIDENCE_FOR_TARGET 10,739 · TARGETS_PROTEIN
6,465 · totals 165,981 nodes / 203,463 rels on a **1GB Aura Free** instance (`b7dbceab`).

## Decisions (grill, 1B1)
- **Branch 1 — approach:** bounded proof on Aura FIRST, then decide on full. (De-risks unmeasured
  Free-tier headroom; keeps the embedder choice cheaply reversible.)
- **Branch 2 — embedder:** two tracks, **cannot share a workspace** (gemini and bge-m3 occupy
  different, incomparable vector spaces even at equal 1024-dim; `assert_workspace_embedding` enforces
  by `(embedding_model, embedding_dim)`):
  | track | workspace | embedder | where |
  |---|---|---|---|
  | full / production | `unified_diet_kg` | **Vertex `gemini-embedding-001`/1024** (ADC, `syntropy-passport`) | Aura |
  | exploratory / local | `unified_diet_kg_explore` (separate) | bge-m3 (LM Studio) | local |
  The bounded proof is the first step of the production track → uses **Vertex** in `unified_diet_kg`.
- **Branch 3 — vector backend:** `ScopedNeo4JVectorStorage` — vectors stored **in Aura**, prod-reachable
  by the deployed gateway (no external vector DB; Zilliz/Milvus is dead), and lets the proof measure
  true Aura growth for full-run feasibility.

## Track config (production, Aura)
`WORKSPACE=unified_diet_kg` · `LIGHTRAG_GRAPH_STORAGE=Neo4JStorage` ·
`LIGHTRAG_VECTOR_STORAGE=ScopedNeo4JVectorStorage` · `EMBEDDING_BINDING=vertex`
`EMBEDDING_MODEL=gemini-embedding-001` `EMBEDDING_DIM=1024` `GOOGLE_CLOUD_PROJECT=syntropy-passport` ·
`NEO4J_*` from Infisical `589d1e3b` `/research/shrine-diet-bioactivity` (URI/USER=`b7dbceab`, DB=`b7dbceab`).

## Tasks
1. **Router guard** — `lightrag/embedder_adapters.assert_binding_allowed_for_workspace` refuses a local
   (bge-m3) embedder against a production workspace BEFORE `WorkspaceMeta` is written (closes the window
   `assert_workspace_embedding` leaves open). Wired into `ingest_unified`. **DONE** — 10/10 unit tests
   (was 7), py_compile clean.
2. **Bounded proof ingest** — `ingest_unified.py` with the track config + caps
   `--max-compounds 1000 --max-herbs 300 --max-foods 300 --max-relationships 3000`.
   Pre-snapshot typed counts as the restore baseline. **Done-condition:** ingest completes; additive
   guard PASSES (typed counts unchanged); `VectorEntity`/`DIRECTED` > 0; `WorkspaceMeta` =
   `gemini-embedding-001`/1024.
3. **Verify + measure** — `kg_query` returns a relevant bioactivity result; HAS_EVIDENCE+evidence_tier
   still readable (typed untouched); write the Aura node/rel/storage delta → full-run feasibility
   projection vs the 1GB Free cap.
4. **Decision gate (principal)** — full ~120k on Aura / stay bounded / upgrade tier, on Task-3 numbers.

## Failing-test targets / baselines
- Router guard: `local`+`unified_diet_kg` raises `SystemExit` (RED before the guard existed). ✅
- Bounded ingest: pre-ingest `VectorEntity=0`, `DIRECTED=0`, `kg_query` empty; typed edges pinned at the
  counts above and must be unchanged after.

## Results — bounded proof EXECUTED (2026-09-11)
- **Completed rc=0** in 2234s (~37 min): 16,738 entities + 18,000 rels, throttled `EMBEDDING_FUNC_MAX_ASYNC=1`.
- **Semantic shape live on Aura:** VectorEntity 16,738 · VectorChunk 174 · DIRECTED 17,546 · WorkspaceMeta(gemini-embedding-001/1024). Vectors stored 1024-dim (correct).
- **`kg_query` PROVEN:** "targets curcumin acts on" → 38,112 chars, retrieved COX-2/PTGS2, PKC, IKK, NFKB, CYP2C9 — correct.
- **Coexistence PROVEN:** `[additive-guard] OK`; typed graph unchanged (HAS_EVIDENCE/EVIDENCE_FOR_TARGET 10,739, TARGETS_PROTEIN 6,465).
- **Throttled rate:** ~14 embeds/sec under max_async=1 (default max_async=8 → 429 RESOURCE_EXHAUSTED). Full corpus ≈ 2–3 h throttled, or needs a Vertex quota increase for speed.

### ⚠️ Findings for the Task-4 full-run decision
1. **Aura Free-tier ceiling is the hard blocker.** The bounded proof alone took Aura to **199,840 nodes** (+33,859), right at the Free-tier ~200k-node cap. The full corpus (~120k more entities → ~240k+ more nodes) **exceeds Free tier** — needs a paid Aura tier.
2. **Vertex embedding quota** requires throttling (max_async=1) or a quota increase on syntropy-passport.
3. **Bugs fixed en route:** (a) LightRAG 1.5.0 `asdict(self)` deepcopies the embed func → adapter must hand a module-level func, not a bound method (fixed + regression test); (b) a **stale 2048-dim vector index** persisted from a prior ingest and shadowed the 1024 embeddings via `IF NOT EXISTS` — dropped + recreated at 1024. `ScopedNeo4JVectorStorage` should validate an existing index's dim (latent bug, flagged).

## Approval
1B1 plan-validity lock — Over and out — approved by mo (principal), 2026-09-11.
