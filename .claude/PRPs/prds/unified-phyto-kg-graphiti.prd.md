# Unified Phytochemical Knowledge Graph with Graphiti Semantic Search

## Problem Statement

LLM dietitian agents answering health queries like "what foods help with chronic inflammation?" currently rely on parametric knowledge or fragmented data sources. The existing mcp-herbal-botanicals server bridges herbs to compounds to foods (Duke + FooDB), but lacks: (1) compound-target-disease relationships needed for mechanistic reasoning, (2) clinical evidence to ground recommendations, (3) ADME/pharmacokinetic properties for bioavailability assessment, and (4) semantic graph search for natural-language traversal across entity types. Without a unified, semantically searchable KG, agents produce shallow answers that can't trace the full pathway from symptom to molecular target to dietary intervention.

## Evidence

- Current MCP server has 11 tools but zero compound-target data (empty tables awaiting CMAUP)
- The `search-by-symptom` tool maps only 47 bioactivity-derived symptoms vs. 2,678 in SymMap
- No chemical-disease associations exist in the current schema — agents can't answer "what foods help with diabetes?" with grounded data
- Graphiti was previously rejected for wrong fit (episodic memory, not static reference) — but user now wants to experimentally validate this decision with a live deployment
- Multiple non-overlapping datasets (CTD, HERB 2.0, TTD, TCMSP) are freely downloadable and would add unique layers (phenotypes, clinical trials, druggability, ADME)

## Proposed Solution

Build a unified data manifest aggregating 8+ non-overlapping datasets into a single ETL pipeline, then index the resulting entity-relationship graph into both: (A) the existing SQLite/MCP server for deterministic tool-call queries, and (B) a Graphiti-powered Neo4j KG for semantic search and LLM agent retrieval. Run a live end-to-end experiment on Railway-hosted Graphiti + Neo4j instances, with local embeddings (LM Studio) to minimize cost. Document the comparative analysis: MCP tool-call queries vs. Graphiti semantic search vs. raw Cypher.

## Key Hypothesis

We believe that indexing structured phytochemical data into a Graphiti temporal knowledge graph will enable richer, more contextual agent responses compared to direct SQL tool calls — particularly for multi-hop queries (symptom -> compound -> target -> disease -> food) that require semantic traversal. We'll know we're right when the Graphiti-backed agent can answer 10 benchmark queries with more complete entity coverage and source attribution than the MCP-only approach.

## What We're NOT Building

- A production-grade deployment — this is an experiment on Railway test instances
- A new MCP server — we extend the existing mcp-herbal-botanicals
- Custom NLP entity extraction — we use Graphiti's built-in LLM extraction on pre-structured JSON
- A web UI — visualization uses Neo4j Browser directly
- Data cleaning/curation of upstream sources — we take datasets as-is with deduplication at join boundaries

## Success Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| Datasets integrated | 6+ non-overlapping sources in manifest | Manifest file audit |
| Entity coverage | 10K+ herbs, 100K+ compounds, 3K+ targets, 5K+ diseases | getStats() counts |
| Graphiti indexing | All core entities + relationships indexed in Neo4j | Neo4j Browser node/edge counts |
| Benchmark queries | 10 multi-hop queries answered by both MCP and Graphiti | Side-by-side comparison doc |
| Local embedding cost | $0 (local LM Studio) | No API billing |

## Open Questions

- [x] Can Graphiti handle 4M+ compound-food edges without OOM during ingestion?
  **RESOLVED**: No. With 8GB memory on both Graphiti and Neo4j, must subsample. Strategy: ingest all herbs (2,376), top 5K compounds, top 10K compound-food pairs, all targets/diseases. Compound-food edges are the OOM risk — hard cap at 10K.
- [ ] What is the embedding dimension of `text-embedding-embeddinggemma-300m-qat`? (Needed for Graphiti config)
  **PENDING**: Need LM Studio running to verify. Defaulting to 768 (GemMA-based models typically use 256 or 768). Run `make embedding-check` to verify.
- [x] Neo4j credentials for Railway instance — need password
  **RESOLVED**: `neo4j`/`demodemo` via TCP proxy `bolt://metro.proxy.rlwy.net:22971`. Verified connection works.
- [x] STITCH 5 is 200GB raw — is pre-filtering to our compound universe feasible in a single session?
  **RESOLVED**: Yes, feasible but not in this phase. Strategy: download only `9606.actions.v5.0.tsv.gz` (human interactions, ~4GB compressed). Pre-filter with a streaming script that checks each line's chemical_id against our compound_name_map (~94K compounds). Expected output: ~50K-200K rows (0.01% of 1.6B). This is a Tier 2 task — worth doing after CMAUP/CTD/TTD are validated, as it adds the largest compound-target interaction network. Implementation: stream with `zlib.createGunzip()` + `readline` (same pattern as load-ctd.ts), filter by CID prefix `CIDm` + normalizeCompoundName on STITCH's flat chemical IDs.
- [x] DisGeNET free tier limits — is the dhimmel/disgenet community snapshot sufficient?
  **RESOLVED**: Yes for experimentation. The [dhimmel/disgenet](https://github.com/dhimmel/disgenet) community snapshot provides ~380K gene-disease associations under ODbL license from the 2015 freeze. This covers the core GDA relationships needed for compound→target→gene→disease traversal. Limitations: (1) data is from 2015 (missing recent GWAS findings), (2) no variant-disease associations, (3) no disease specificity scores that the paid tier provides. For this experiment, the community snapshot is sufficient — it closes the gene→disease gap that CMAUP and TTD partially cover. Full DisGeNET would be needed only for production-grade clinical recommendations.
- [x] BATMAN-TCM has no bulk download — is scraping or author contact required?
  **RESOLVED**: Confirmed no bulk download. GitHub mirror has compound_list.txt (33,053 entries). Live API at `batman2api.cloudna.cn/queryTarget` accepts PubChem CIDs and returns compound-target mappings. A scraper script iterating 33K compounds would work but is rate-limited. For now, CMAUP's 15,204 compound-target associations (already loaded) cover the same space. BATMAN-TCM scraping deferred to Tier 3.

---

## Users & Context

**Primary User**
- **Who**: LLM dietitian agent (the Diet Insight Engine SDO) making tool calls to answer user health queries
- **Current behavior**: Calls 11 MCP tools on a SQLite DB with herb-compound-food data only; no target/disease/clinical layer
- **Trigger**: User asks a multi-hop question like "what foods contain compounds that target COX-2 for inflammation?"
- **Success state**: Agent retrieves a grounded, multi-source answer tracing symptom -> herb -> compound -> target -> food with citations

**Job to Be Done**
When an LLM agent receives a health-related dietary query, it wants to traverse a comprehensive phytochemical knowledge graph, so it can provide evidence-backed dietary recommendations with molecular-level mechanistic reasoning.

**Non-Users**
- End consumers (they interact with the agent, not the KG directly)
- Researchers doing batch analysis (this is optimized for real-time agent queries, not bulk analytics)

---

## Solution Detail

### Core Capabilities (MoSCoW)

| Priority | Capability | Rationale |
|----------|------------|-----------|
| Must | Unified data manifest with 6+ sources | Foundation for all downstream work |
| Must | ETL pipeline populating SQLite with all datasets | Extends existing MCP tools with new data |
| Must | Graphiti submodule + configuration for local embeddings | Required for the semantic search experiment |
| Must | End-to-end Graphiti indexing on Railway Neo4j | Core experiment deliverable |
| Must | Comparative analysis doc (MCP vs Graphiti vs raw Cypher) | Key decision artifact |
| Should | Neo4j visualization guide (local + cloud) | Documentation for team |
| Should | Benchmark query suite (10 queries) | Quantitative comparison |
| Could | Additional MCP tools for new entity types (targets, diseases) | Extends agent capabilities |
| Could | CTD chemical-phenotype integration | Unique phenotype layer |
| Won't | Production Graphiti deployment | Experiment only |
| Won't | Kuzu migration (previously Phase 6) | Deferred pending Graphiti experiment results |

### MVP Scope

1. Data manifest with at least: Duke (existing), FooDB (existing), CMAUP, SymMap, CTD, HERB 2.0, TTD
2. Graphiti cloned as submodule, configured with local embeddings + Railway Neo4j
3. Core entities (herbs, compounds, targets, diseases) indexed into Graphiti
4. 10 benchmark queries run against both MCP tools and Graphiti search
5. Comparison document with quality assessment

### User Flow (Agent Query)

```
User: "What foods help with chronic inflammation?"
  |
  v
Agent decides: MCP tool call OR Graphiti semantic search
  |
  ├── MCP path: search-by-symptom("inflammation") -> structured JSON
  |
  └── Graphiti path: graphiti.search("chronic inflammation foods") 
       -> semantic traversal across symptom/compound/food nodes
       -> returns edges with provenance and temporal validity
  |
  v
Agent synthesizes response with source attribution
```

---

## Technical Approach

**Feasibility**: HIGH — all components exist; this is integration + experiment work

**Architecture**

```
┌─────────────────────────────────────────────────────┐
│                 DATA MANIFEST                        │
│  manifest.yaml — 8 sources, schemas, download URLs  │
└──────────────────────┬──────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────────┐
│              ETL PIPELINE (Python)                   │
│  Download → Normalize → Deduplicate → Load           │
│  Compound name normalization across all sources      │
└──────┬──────────────────────────────────┬───────────┘
       │                                  │
       v                                  v
┌──────────────┐                 ┌──────────────────┐
│   SQLite DB  │                 │  Graphiti → Neo4j │
│ (914MB+)     │                 │  (Railway)        │
│              │                 │                    │
│ herbs        │                 │  EntityNode:Herb   │
│ compounds    │                 │  EntityNode:Cpd    │
│ targets      │                 │  EntityNode:Target │
│ diseases     │                 │  EntityNode:Food   │
│ symptoms     │                 │  EntityNode:Disease│
│ compound_    │                 │  EntityNode:Symptom│
│  foods       │                 │                    │
│ herb_        │                 │  Edges w/ temporal │
│  compounds   │                 │  validity + source │
│ compound_    │                 │  provenance        │
│  targets     │                 │                    │
│ ...          │                 │  Vector indices    │
└──────┬───────┘                 │  (local embeddings)│
       │                         └────────┬───────────┘
       v                                  v
┌──────────────┐                 ┌──────────────────┐
│  MCP Server  │                 │  Graphiti Search  │
│  (11+ tools) │                 │  (Python SDK)     │
│  TypeScript  │                 │  Hybrid retrieval │
│  Deterministic│                │  Semantic + graph │
└──────────────┘                 └──────────────────┘
       │                                  │
       └──────────┬───────────────────────┘
                  v
          ┌──────────────┐
          │  LLM Agent   │
          │  Comparison  │
          │  Benchmark   │
          └──────────────┘
```

**Key Technical Decisions**

| Decision | Choice | Alternatives | Rationale |
|----------|--------|--------------|-----------|
| Graph DB | Neo4j (Railway) | Kuzu (embedded), FalkorDB | User has running Railway instance; Graphiti supports it natively |
| Embeddings | Local LM Studio (embeddinggemma-300m-qat) | OpenAI API, Gemini/Vertex AI | Zero cost; OpenAI-compatible endpoint |
| Graphiti integration | Python submodule | npm package, REST API | Graphiti is Python-native; TypeScript MCP server is separate concern |
| Data manifest format | YAML with JSON Schema | SQL migrations, Python dicts | Human-readable, versionable, tooling-friendly |
| Dedup strategy | Compound name normalization + PubChem CID | InChI key, SMILES | Matches existing normalizeCompoundName() pattern; PubChem CID as fallback |

**Technical Risks**

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Graphiti OOM on 4M+ edges | Medium | Batch ingestion with chunking; ingest compound-food edges last or sample |
| Local embedding dim mismatch | Low | Verify embeddinggemma output dim before configuring Graphiti |
| Neo4j Railway instance too small | Medium | Monitor memory; fall back to local Neo4j Docker if needed |
| BATMAN-TCM no bulk download | High | Skip or scrape subset; prioritize CTD + HERB 2.0 + TTD instead |
| LLM cost for Graphiti entity extraction | Medium | Use local LLM (LM Studio) for extraction too, or pre-structure JSON to minimize extraction calls |

---

## Dataset Manifest (Ranked by Priority)

### Tier 1 — Must Have (Non-overlapping, freely downloadable)

| # | Dataset | Entities Added | Unique Layer | License | Format |
|---|---------|---------------|--------------|---------|--------|
| 1 | Dr. Duke's Phytochemical DB | 2,376 herbs, 94K compounds, 99K links | Herb-compound backbone | CC0 | CSV (existing) |
| 2 | FooDB | 4.1M compound-food pairs | Food-compound bridge | CC BY-NC | CSV (existing) |
| 3 | CMAUP 2024 | 7,865 plants, 60K compounds, 758 targets, 1,399 diseases | Compound-target-disease | Academic | CSV (planned) |
| 4 | SymMap v2 | 1,717 TCM symptoms, 961 MM symptoms, 6,638 herb-symptom links | Symptom-herb mapping | Academic | CSV (planned) |
| 5 | CTD | 17,700 chemicals, 55,400 genes, 7,200 diseases, 6,700 phenotypes | Chemical-disease-phenotype | Academic | CSV/TSV |
| 6 | HERB 2.0 | 7,263 herbs, 49K compounds, 12,933 targets, 8,558 clinical trials | Clinical evidence | Free | CSV |
| 7 | TTD | 3,730 targets with druggability status, 39K drugs | Target druggability | Free | TSV |

### Tier 2 — Should Have (High value, needs filtering)

| # | Dataset | Entities Added | Unique Layer | License | Format |
|---|---------|---------------|--------------|---------|--------|
| 8 | TCMSP | 29,384 compounds with 12 ADME properties | ADME/pharmacokinetics | ODbL | Web/CSV |
| 9 | STITCH 5 | 1.6B interactions (filtered to our compounds) | Massive compound-target | CC BY 4.0 | TSV.gz |

### Tier 3 — Could Have (Access-limited)

| # | Dataset | Entities Added | Unique Layer | License | Format |
|---|---------|---------------|--------------|---------|--------|
| 10 | DisGeNET | 400K gene-disease associations | Gene-disease closure | Limited free | TSV |
| 11 | BATMAN-TCM 2.0 | 2.3M predicted interactions | Predicted TTIs | CC BY-NC | No bulk DL |

### Skip

| Dataset | Reason |
|---------|--------|
| Kaggle TCM | No relevant structured data; image/NLP datasets only |
| HIT 2.0 | Subsumed by BATMAN-TCM + CMAUP |
| Phenol-Explorer | Too narrow; FooDB covers polyphenols |
| USDA Flavonoid DB | Subset of FooDB coverage |
| ChiMed 2.0 | NLP corpus only, no structured relationships |
| IMPPAT 2.0 | Overlaps with Duke + CMAUP |

---

## Approach Comparison: With vs Without Graphiti

### Without Graphiti (Current MCP-Only Approach)

```
Agent → MCP tool call (search-by-symptom, get-herb-compounds, etc.)
     → SQLite query with parameterized SQL
     → Structured JSON response
     → Agent composes answer from multiple tool calls
```

**Strengths:**
- Deterministic, reproducible results
- Sub-200ms query latency on indexed SQLite
- No LLM cost at query time
- Full control over query logic and result ranking
- Works offline (embedded DB)

**Weaknesses:**
- Agent must know which tool to call and chain calls manually
- No semantic understanding — `search-by-symptom("I feel tired")` requires exact symptom name match
- Multi-hop queries require 3-4 sequential tool calls
- No cross-entity similarity search (e.g., "compounds similar to curcumin")
- Adding new query patterns requires new MCP tools

### With Graphiti (Semantic KG Approach)

```
Agent → graphiti.search("natural language query")
     → Hybrid retrieval: semantic embedding + BM25 + graph traversal
     → Returns edges with facts, provenance, temporal validity
     → Agent gets pre-traversed multi-hop results
```

**Strengths:**
- Natural language queries without exact-match requirements
- Automatic multi-hop traversal (symptom -> herb -> compound -> food in one call)
- Semantic similarity enables fuzzy matching ("fatigue" finds "tiredness", "low energy")
- Temporal tracking — facts have valid_at/invalid_at timestamps
- Provenance — every fact traces back to source dataset
- Emergent relationships discovered during LLM-powered ingestion

**Weaknesses:**
- LLM cost during ingestion (entity extraction calls per episode)
- Non-deterministic — different LLM runs may extract slightly different entities
- Requires running Neo4j server (not embedded)
- Higher query latency (embedding + graph traversal + reranking)
- Overkill for simple lookups ("get compounds for herb X")

### Quality Comparison for LLM Agent Use

| Query Type | MCP Quality | Graphiti Quality | Winner |
|------------|------------|-----------------|--------|
| Simple lookup ("compounds in ashwagandha") | Excellent — exact, fast | Good but slower | MCP |
| Symptom search ("I'm tired") | Limited — needs exact symptom name | Excellent — semantic fuzzy match | Graphiti |
| Multi-hop ("foods with anti-inflammatory compounds targeting COX-2") | Requires 3-4 chained tool calls | Single semantic query traverses graph | Graphiti |
| Cross-entity similarity ("compounds like curcumin") | Not supported | Good — vector similarity on compound embeddings | Graphiti |
| Provenance ("where does this fact come from?") | Source column only | Full episode provenance with temporal tracking | Graphiti |
| Batch/analytics ("top 10 herbs by compound count") | Excellent — SQL aggregation | Poor — not designed for analytics | MCP |
| Offline use | Works fully offline | Requires Neo4j server | MCP |

### Recommended Hybrid Strategy

Use **both** in production:
1. **MCP tools** for deterministic, structured queries (lookups, aggregations, pagination)
2. **Graphiti** for natural-language semantic search, multi-hop traversal, and discovery queries
3. Agent routes queries based on intent: structured → MCP, exploratory → Graphiti

---

## Implementation Phases

| # | Phase | Description | Status | Parallel | Depends | PRP Plan |
|---|-------|-------------|--------|----------|---------|----------|
| 1 | Data Manifest & ETL | Create manifest.yaml, download scripts for all Tier 1 datasets | complete | - | - | `.claude/PRPs/plans/completed/unified-phyto-kg-phase1-manifest-etl.plan.md` |
| 2 | SQLite Integration | Populate CMAUP, SymMap, CTD, HERB, TTD into existing schema | complete | - | 1 | `.claude/PRPs/plans/completed/unified-phyto-kg-phase1-manifest-etl.plan.md` |
| 3 | Graphiti Setup | Clone submodule, configure local embeddings + Railway Neo4j | complete | with 2 | 1 | `.claude/PRPs/plans/completed/unified-phyto-kg-phase1-manifest-etl.plan.md` |
| 4 | Graphiti Indexing | Ingest all entities + relationships into Neo4j via Graphiti | pending | - | 2, 3 | - |
| 5 | Benchmark & Comparison | Run 10 queries on both MCP and Graphiti, document results | pending | - | 4 | - |
| 6 | Visualization & Docs | Neo4j Browser guide (local + cloud), architecture docs | pending | with 5 | 4 | - |

### Phase Details

**Phase 1: Data Manifest & ETL**
- **Goal**: Create a single manifest.yaml describing all datasets with schemas, download URLs, normalization rules, and dedup strategy
- **Scope**: manifest.yaml, download scripts for CTD, HERB 2.0, TTD (CMAUP + SymMap already planned)
- **Success signal**: All Tier 1 datasets downloadable via `scripts/download-datasets.sh`

**Phase 2: SQLite Integration**
- **Goal**: Populate all empty Phase 4 tables + new CTD/HERB/TTD tables in SQLite
- **Scope**: Migration scripts, compound name normalization across sources, new MCP tools for targets/diseases
- **Success signal**: `getStats()` shows 10K+ herbs, 100K+ compounds, 3K+ targets, 5K+ diseases

**Phase 3: Graphiti Setup**
- **Goal**: Clone graphiti as git submodule, configure for local embeddings + Railway Neo4j
- **Scope**: `git submodule add`, Python venv, Graphiti config with OpenAIEmbedder pointing to LM Studio, Neo4j connection to Railway instance
- **Success signal**: `graphiti.build_indices()` completes without error on Railway Neo4j
- **Neo4j connection**: `bolt://neo4j-test-2be3.up.railway.app:7687` (or proxy `metro.proxy.rlwy.net:22971`)
- **Embeddings**: `http://127.0.0.1:1234/v1/embeddings` with model `text-embedding-embeddinggemma-300m-qat`

**Phase 4: Graphiti Indexing**
- **Goal**: Ingest all core entities and relationships into the Graphiti KG
- **Scope**: Python ingestion script using `graphiti.add_episode()` with `EpisodeType.json` for structured data; batch processing with progress tracking
- **Success signal**: Neo4j Browser shows all entity types as nodes with labeled edges

**Phase 5: Benchmark & Comparison**
- **Goal**: Quantitative comparison of MCP tool-call vs Graphiti semantic search
- **Scope**: 10 benchmark queries spanning simple lookups, multi-hop traversals, semantic similarity, and natural language symptom queries
- **Success signal**: Comparison document with entity coverage, latency, and quality scores per query

**Phase 6: Visualization & Docs**
- **Goal**: Document how to visualize the Neo4j KG and reproduce the experiment
- **Scope**: 
  - Local: `docker run neo4j` + Graphiti config for localhost
  - Cloud: Neo4j Browser at `https://neo4j-test-2be3.up.railway.app:7474` (HTTP) or Neo4j Desktop connecting to bolt URI
  - Cypher query examples for exploring the graph
  - Architecture decision record: MCP vs Graphiti vs hybrid
- **Success signal**: New team member can visualize the KG in <5 minutes following the guide

### Parallelism Notes

Phases 2 and 3 can run in parallel — Phase 2 works on SQLite schema/data while Phase 3 sets up the Graphiti infrastructure. Both depend on Phase 1 (manifest) being complete. Phases 5 and 6 can also partially overlap since documentation can begin while benchmarking runs.

---

## Decisions Log

| Decision | Choice | Alternatives | Rationale |
|----------|--------|--------------|-----------|
| Graph DB for experiment | Neo4j on Railway | Kuzu (embedded), FalkorDB, local Docker Neo4j | User has running Railway instance; validates cloud deployment path |
| Embeddings | Local LM Studio (embeddinggemma-300m-qat) | OpenAI API, Gemini/Vertex AI | Zero cost; sufficient quality for experiment |
| Graphiti integration | Git submodule | pip install, Docker | Submodule allows local development and version pinning |
| Manifest format | YAML | JSON, TOML, Python config | Human-readable, supports comments, widely tooled |
| Skip BATMAN-TCM | Yes (no bulk download) | Scrape, contact authors | Access risk too high; CTD + HERB 2.0 cover the target-disease gap |
| Skip Kaggle TCM | Yes (no relevant structured data) | N/A | Only image/NLP datasets; no compound-level structured data |
| Dedup strategy | Compound name normalization + PubChem CID | InChI, SMILES fingerprints | Extends existing normalizeCompoundName(); PubChem CID available in SymMap/CMAUP |
| Graphiti entity extraction LLM | Local LLM (LM Studio) | OpenAI GPT-4, Claude | Zero cost; Graphiti supports OpenAI-compatible endpoints for both embeddings and LLM |

---

## Research Summary

**Market Context**
- No existing open-source project combines all 7+ datasets into a single queryable KG for LLM agents
- Graphiti is the leading open-source temporal KG framework but designed for conversational memory, not static reference data — this experiment tests that boundary
- The 2024 Frontiers assessment found only 295 compounds shared across all 8 major TCM databases — confirming low overlap and high additive value from multi-source integration

**Technical Context**
- Existing mcp-herbal-botanicals has clean patterns (Zod schemas, HerbalDBAdapter, vitest) that extend naturally
- Graphiti supports custom OpenAI-compatible embedding endpoints and Neo4j bolt connections — confirmed in source code
- The 4.1M compound-food edges are the largest single relationship set; Graphiti ingestion may need batching/sampling
- Railway Neo4j instance is already running; Graphiti test instance at graphiti-test.up.railway.app is available

---

*Generated: 2026-04-10*
*Status: DRAFT - needs validation of Neo4j credentials and embedding dimension*
