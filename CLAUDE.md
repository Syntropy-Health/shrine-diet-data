# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open Diet Data builds a **unified diet knowledge graph** spanning macronutrients, foods, herbs, phytochemical compounds, molecular targets, and diseases. It aggregates 7+ authoritative datasets into a semantic KG powered by LightRAG (Neo4j graph + vector embeddings), queryable by LLM agents via a FastMCP gateway (`mcp/`) and a scoped FastAPI wrapper.

Data foundation for the [Diet Insight Engine](https://github.com/Syntropy-Health/diet-insight-engine) Symptom-Diet Optimizer (SDO). Part of the [Syntropy Health](https://github.com/Syntropy-Health) ecosystem.

## Repository Layout

- `mcp/` — Active **kg-mcp Python gateway**: FastMCP server exposing 10 tools over streamable-HTTP, ~85% test coverage, PostHog instrumentation. The runtime entrypoint for LLM agents talking to the KG.
- `shrine-diet-bioactivity/` — Nested Python package containing the application code:
  - `eval/` — DietResearchBench evaluation harness
  - `agents/` — AG2 multi-agent panel
  - `lightrag/` — Scoped FastAPI wrapper around upstream LightRAG (`scoped_server.py`, tenant scoping, audit log)
  - `scripts/` — Python + TypeScript data automation
  - `src/` — TypeScript thin-adapter MCP (5 domain-agnostic tools over the scoped wrapper)
  - `data_local/herbal_botanicals.db` — Unified SQLite (herbs, compounds, foods, targets, diseases, symptoms)
  - `Makefile` — Pipeline targets (download, build, migrate, LightRAG ingest, eval)
- `lightrag/` — Git submodule: upstream HKUDS/LightRAG framework (subclassed by the scoped wrapper)
- `research-journal/` — Papers, plans, datasets (`dietresearchbench_v1.json`), results, audits
- `docs/` — Architecture docs, KG comparison, data schematics
- `scripts/` — Top-level operator scripts (bash + Python)
- `.claude/PRPs/` — Historical plans, PRDs, and implementation reports (legacy artifacts; some reference the older `mcp-herbal-botanicals` name from before the unification refactor)

## Build & Setup Commands

```bash
# Initialize submodules (lightrag only — mcp-opennutrition removed)
git submodule update --init --recursive

# Application package: data pipeline + LightRAG ingest
cd shrine-diet-bioactivity && make setup

# Or step-by-step (from shrine-diet-bioactivity/):
make download-all      # Duke + CMAUP + TTD source data
make build             # Build SQLite from CSVs
make migrate           # KG expansion (symptoms, targets)

# LightRAG KG ingestion
make lightrag-setup       # Install Python deps
make lightrag-dry-run     # Preview entity/relationship counts
make lightrag-ingest-local  # Ingest into Neo4j (Ollama embeddings)
make lightrag-benchmark   # Run 10 benchmark queries

# Scoped LightRAG FastAPI wrapper (port 9621)
make lightrag-server

# Tests
cd shrine-diet-bioactivity && python3 -m pytest -m unit  # Python unit tests
cd shrine-diet-bioactivity && npm test                   # vitest (TypeScript adapter)
cd mcp && python3 -m pytest -m unit                      # kg-mcp gateway tests
```

## Architecture

### Unified Data Flow

```
STRUCTURED DATA (zero LLM cost):
  Duke CSV (2.4K herbs, 94K compounds) ─┐
  FooDB CSV (4.1M compound-food pairs) ──┤
  CMAUP TSV (758 targets, 429K links) ──┼──► shrine-diet-bioactivity/data_local/
  CTD CSV.gz (17.7K chemicals) ──────────┤        herbal_botanicals.db
  TTD TSV (3.7K targets) ───────────────┤        (unified SQLite)
  SymMap / HERB 2.0 (TCM symptoms) ─────┘                │
                                                          │
                                                          ├──► LightRAG ainsert_custom_kg()
                                                          ▼
                                          ┌─────────────────────────────┐
                                          │ LightRAG (Neo4j Aura)       │
                                          │ Scoped FastAPI wrapper      │
                                          │ (shrine-diet-bioactivity/   │
                                          │  lightrag/scoped_server.py) │
                                          │ — tenant scoping, audit log │
                                          └──────────────┬──────────────┘
                                                         │
  LLM Agent ──► kg-mcp Python gateway (mcp/) ──► /query, /graphs, custom_kg
       │           (FastMCP, 10 tools, streamable-HTTP)
       └─────► TypeScript thin-adapter MCP ──► scoped FastAPI wrapper
                (shrine-diet-bioactivity/src/, 5 tools)
```

### Key Integration Points

- **MCP Protocol**: `mcp/` exposes the primary kg-mcp gateway (FastMCP, 10 tools, streamable-HTTP, PostHog-instrumented). `shrine-diet-bioactivity/src/` exposes a 5-tool TypeScript thin-adapter using `@modelcontextprotocol/sdk` with Zod schemas.
- **SQLite**: Unified `shrine-diet-bioactivity/data_local/herbal_botanicals.db` with 12 tables (herbs, compounds, compound_foods, targets, diseases, symptoms). OpenNutrition data is vendored as fixtures, not as a live submodule.
- **LightRAG**: Semantic KG indexed in Neo4j Aura with 6 entity types (Herb, Compound, Food, Target, Disease, Symptom) and 5 relationship types. Queries via the scoped FastAPI wrapper (`scoped_server.py`) which subclasses `Neo4JStorage` to enforce tenant scopes and mounts the upstream `/graphs` routes.
- **Dual Config**: `config_local.env` (Ollama, zero cost) for dev/test. `config_production.env` (OpenAI + Jina multilingual reranker) for production with Chinese+English TCM support.

### Technology Stack

| Component | Stack |
|---|---|
| kg-mcp gateway (`mcp/`) | Python 3.10+, FastMCP, streamable-HTTP, PostHog |
| TS thin-adapter MCP | TypeScript, Node.js 22+, better-sqlite3, Zod, vitest |
| Scoped LightRAG wrapper | Python 3.10+, FastAPI, LightRAG, Neo4j 5.26+ |
| Embeddings (local) | Ollama, bge-m3 (1024-dim) |
| Embeddings (prod) | OpenAI text-embedding-3-large (3072-dim) |
| Graph DB | Neo4j Aura (cloud) / Railway (dev) |
| Eval harness | Python (DietResearchBench), AG2 agents |

## Submodules

After cloning, always run: `git submodule update --init --recursive`

| Submodule | Purpose | Upstream |
|---|---|---|
| `lightrag` | LightRAG semantic KG framework (subclassed by the scoped wrapper) | https://github.com/HKUDS/LightRAG |

## Active Branches & Features

- `main` — Stable branch
- `feature/mcp-herbal-botanicals` — Unified diet KG + LightRAG (historical feature branch name; preserved for context)
- `research-journal/shared/plans/` — Active feature plans and implementation reports

## Data Audit Notes

OpenNutrition (vendored fixture, 326,759 foods): Calories 95.5% coverage, protein 74.7%, carbs 90.2%, fat 69.5%, fiber 53.8%. A `data_completeness` score should flag foods with missing core macros. See `docs/data-audit-results.md`.

## Conventions

- Scripts use descriptive docstring headers with usage examples (see `scripts/query-nih-dsld.py`)
- No API keys required for core data sources; only `OPENAI_API_KEY` optional for cloud embeddings
- Python scripts handle missing dependencies gracefully with try/except ImportError
- MCP tools use Zod schemas (TS) or Pydantic (Python) for input validation
- `output/` directory is gitignored — all generated data lives there

## Secrets

**Source of truth: Infisical.** When blocked by a missing credential, check Infisical FIRST — don't ask the user, don't stub, don't hardcode. See the monorepo-level [`SECRETS.md`](../../SECRETS.md) for the full practice (auth, REST fallback, layout conventions, secret-handling hygiene).

Secrets this repo needs and where they live:

| Secret | Used by | Infisical path | GitHub Actions secret? |
|---|---|---|---|
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | `mcp-ci.yml` (aura-data-integrity job); `lightrag/scripts/*` | project `687cab01-ccc1-4789-99a9-1214bd268f2b`, env `prod`, path `/research/shrine-diet-bioactivity` | ✅ mirrored |
| `RAILWAY_TOKEN` | `deploy-mcp.yml` (Railway deploy + post-deploy /health poll) | (set in Railway dashboard; mirror to Infisical when convenient) | ✅ |
| `OPENAI_API_KEY` (optional) | LightRAG production embeddings (only when running `make lightrag-ingest-prod`) | not yet in Infisical — add when needed | ❌ |
| `JINA_API_KEY` (optional) | LightRAG production reranker (Chinese+English TCM) | not yet in Infisical — add when needed | ❌ |
| `KG_MCP_API_KEY` (client) / `MCP_API_KEY` (gateway) | `shrine-diet-bioactivity/eval/mcp_clients.py`; validated by `mcp/src/kg_mcp/auth.py` | **currently** project `589d1e3b-…`, env `prod`, path `/research/shrine-diet-bioactivity` — see note below | ❌ (gateway reads it from Railway env) |

### `KG_MCP_API_KEY` — two names, and a location decision still open

**Two env-var names, one secret value.** The client sends `KG_MCP_API_KEY`; the
gateway validates `MCP_API_KEY` (`mcp/src/kg_mcp/auth.py`). There is no vendor to
revoke at — **rotating it means changing the value the GATEWAY accepts (Railway
env), not the source of truth.** Updating Infisical alone leaves the old value
working, which reports as a completed rotation while the credential stays live.

**Current actual location: project `589d1e3b-…`, `prod`, `/research/shrine-diet-bioactivity`.**
Recorded as measured, not as intended. The folders there had to be created — this
app's other secrets live in `687cab01-…`, and consolidating into it is blocked on
project membership as of 2026-08-28. Until that is resolved, **an agent granted
READ on `687cab01` alone cannot read this key.**

That distinction is the point of this note. An earlier relay inferred this key's
home from the neighbouring `NEO4J_*` row and reported it as documented — the row
existed, but for a different secret. A table that documents four secrets and omits
the fifth gives the next reader no way to tell an omission from a location, so
this row states where the key IS, and flags that where it SHOULD live is undecided.
Do not "correct" this row to `687cab01` until the value has actually moved.

History: the literal was committed to this PUBLIC repo 2026-05-26 and rotated
2026-08-27 after ~3 months of exposure. Verified by pair-test, not by a deploy
badge: old key -> 401 stable across 6 samples, `NONE` control -> 401, new key
authenticates.

**Rotating any of these:** update Infisical first, then `printf '%s' "$VAL" | gh secret set NAME --repo Syntropy-Health/shrine-diet-bioactivity --body -` to re-mirror, then update Railway dashboard. Never let the three stores drift.
