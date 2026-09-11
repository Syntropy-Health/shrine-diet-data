#!/bin/zsh
# Resume/extend the LightRAG SEMANTIC (vector) ingestion on Aura.
# See research-journal/plans/2026-09-11-semantic-core-CHECKPOINT.md for state + constraints.
#
# Usage:   zsh run_semantic_ingest.sh [extra ingest_unified args...]
#   e.g.   zsh run_semantic_ingest.sh --max-compounds 5000 --max-relationships 10000
#   no args -> UNCAPPED (attempts the full corpus).
#
# ⛔ Aura FREE is at its ~200k-node ceiling (199,840). A meaningful extend needs a PAID Aura tier
#    FIRST — on Free this will hit the write limit almost immediately. Vertex quota REQUIRES the
#    EMBEDDING_FUNC_MAX_ASYNC=1 throttle (default 8 -> 429). ainsert_custom_kg MERGEs (idempotent).
set -e
LG="${0:A:h}"                       # this script's dir = the lightrag dir
cd "$LG"
PY="${LRENV_PY:-/tmp/lrenv/bin/python}"   # a venv with lightrag-hku[api]==1.5.0 + neo4j + google-genai + httpx

TOK=$(infisical login --method=universal-auth --client-id="$INFISICAL_UNIVERSAL_AUTH_CLIENT_ID" --client-secret="$INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET" --domain=https://app.infisical.com --plain --silent 2>/dev/null)
APP=589d1e3b-5798-48ea-97c0-2d58086a375b; P=/research/shrine-diet-bioactivity
g(){ infisical secrets get "$1" --projectId="$APP" --env=prod --path="$P" --token="$TOK" --domain=https://app.infisical.com --plain --silent 2>/dev/null; }
export NEO4J_URI="$(g NEO4J_URI)" NEO4J_USERNAME="$(g NEO4J_USERNAME)" NEO4J_PASSWORD="$(g NEO4J_PASSWORD)" NEO4J_DATABASE="$(g NEO4J_DATABASE)"

# --- production semantic track (see CHECKPOINT.md) ---
export WORKSPACE=unified_diet_kg
export LIGHTRAG_GRAPH_STORAGE=Neo4JStorage
export LIGHTRAG_VECTOR_STORAGE=ScopedNeo4JVectorStorage
export EMBEDDING_BINDING=vertex EMBEDDING_MODEL=gemini-embedding-001 EMBEDDING_DIM=1024
export EMBEDDING_FUNC_MAX_ASYNC=1 EMBEDDING_BATCH_NUM=10 MAX_PARALLEL_INSERT=1   # Vertex-quota throttle
export GOOGLE_CLOUD_PROJECT=syntropy-passport GOOGLE_CLOUD_LOCATION=us-central1
export LLM_BINDING=openai LLM_BINDING_HOST=http://localhost:1234/v1 LLM_MODEL=google/gemma-4-12b-qat LLM_BINDING_API_KEY=lm-studio LLM_JSON_SCHEMA_COMPAT=1
export WORKING_DIR="$LG/../data_local/rag_storage_semantic_core"   # preserved chunk cache

echo "=== semantic ingest resume $(date) — workspace=$WORKSPACE embedder=vertex/gemini-1024 ==="
exec "$PY" -u ingest_unified.py --config local --batch-size 200 "$@"
