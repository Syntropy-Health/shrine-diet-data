# shrine-bioactivity — graph schema & data construction

> **Canonical source: Notion** — https://app.notion.com/p/3ce749bd250d8104b4b4d910c9dde733
> ("shrine-bioactivity — graph schema & data construction").
> This local file is a **mirror/reference**. On any schema/guard/pipeline change:
> update the code → update the Notion page → confirm this reference still points there.
> _Last aligned: 2026-09-03._

This document (in Notion) is the design + execution reference for the KG: the graph
**schema**, the **data-construction** pipeline, and the construction **guards**. Summary
of what it covers — read the Notion page for the full, maintained version:

## 1. Graph schema
- **Node types:** Herb · Compound · Food · Target · Disease · Symptom · BioactivityEvidence · Pathway
  (all carry the `unified_diet_kg` workspace label + a `scope` property; `WorkspaceMeta` is
  property-keyed, deliberately un-labelled).
- **Edge types:** CONTAINS_COMPOUND, FOUND_IN_FOOD, TARGETS_PROTEIN, ASSOCIATED_WITH_DISEASE,
  TREATS_SYMPTOM, HAS_EVIDENCE/EVIDENCE_FOR_TARGET (ChEMBL), COMPOUND_IN_PATHWAY, …
- **Edge wire props:** description, keywords, weight, source_id, **evidence_tier**, scope.
- **Two shapes:** typed (`:Compound`/`:TARGETS_PROTEIN`, Aura, chain-tools via `ingest_direct.py`)
  vs LightRAG (`:unified_diet_kg`/`:DIRECTED`+vectors, `kg_query` via `ingest_unified.py`).
  They are not interchangeable; a replacing re-ingest of one kills the other's tools.

## 2. Data-construction pipeline
Sources (Duke, FooDB, CMAUP, TTD, CTD, SymMap, HERB 2.0, ChEMBL) → `data_local/herbal_botanicals.db`
via `download-sources → decompress → build-herbal-db → migrate-kg → migrate-multi-source`, then
`ingest_direct` (typed→Aura) or `ingest_unified` (LightRAG). ChEMBL bridge:
`build_compound_identity` (PubChem) → `build_bioactivity_evidence` (`chembl_extractor`) → HAS_EVIDENCE.

## 3. Construction guards (invariants)
1. **Workspace↔embedding-model** (`lightrag_init.assert_workspace_embedding`) — no mixing embedding spaces.
2. **Additive-only** (`additive_guard.py`, #233b) — fail closed if any label/rel-type count decreases.
3. **evidence_tier** (`entity_schema.evidence_tier_for`, #233b) — assay / TTD-layer-verbatim / annotated / ∅.
4. **ChEMBL specificity** (`chembl_extractor`, #233b spike) — pchembl floor, confidence floor,
   **≥2 distinct doc_ids per (compound,target) pair** (independent-publication corroboration; PAINS guard).

## 4. Plan & execution
- **PR-1 LANDED** (app #96, `7f94d3a`): T1 build path, T2 LM-Studio query path + embedding guard.
- **PR-2 built** (branch `shrine-diet-bioactivity/pr2-t4-ingest-evidence`): T4.0 embedder benchmark,
  T4.1 additive guard, T4.2 evidence_tier, T4.3 ChEMBL ≥2-docs guard — each gated + receipted.
- **ChEMBL population EXECUTED (2026-09-03, credential-free local):** compound_identity 8,325 InChIKeys (PubChem); `bioactivity_evidence` **10,739 rows** (was 0) after the T4.3 guard kept 1,494/7,335 pairs from 18,281 bioactivities; HAS_EVIDENCE edges all `evidence_tier=assay`. Graph write (HAS_EVIDENCE into Aura) is the ingest step (T4.1-guarded).
- **Embeddings (2026-09-05 decision):** local dev = **bge-m3** (LM Studio, credential-free);
  external/prod arm = **Gemini / Vertex AI** embeddings (project `syntropyhealth-shrine`, via ADC /
  gcloud) — **Voyage AI is dismissed entirely** (we already have Google/Vertex access; no free-tier
  rate limit to work around).
- **Aura connection (2026-09-05, RESOLVED):** instance `b7dbceab` "shrine-bioactivity-base". Working Bolt
  creds are **non-default** and must be read from the canonical `687cab01 /research/shrine-diet-bioactivity`:
  `NEO4J_USERNAME=b7dbceab` and `NEO4J_DATABASE=b7dbceab` (NOT the Aura default `neo4j`), password unchanged.
  (Fixed a latent `687cab01` bug: `NEO4J_DATABASE` was `neo4j`, masked by default-home-db routing.)
- **HAS_EVIDENCE population EXECUTED on Aura (2026-09-05):** additive-guarded write of **10,739
  BioactivityEvidence nodes + 10,739 HAS_EVIDENCE edges** (Compound→BioactivityEvidence), all
  `evidence_tier=assay` on the wire. Zero orphans (Compound stayed 104,378); additive guard passed. Deferred:
  **EVIDENCE_FOR_TARGET** (10,739 edges) — 225/735 Target endpoints absent from the graph, needs a create-vs-enrich
  decision. **Gemini/Vertex embedding arm (T4.0)** — separate, not yet run.
