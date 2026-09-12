# kg-mcp Operational Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the operational gaps surfaced by the 2026-05-25 e2e run against the live kg-mcp gateway — fix three test-suite bugs that mask real KG state (SYN-89), then fix the depth-2 `/traverse` and `/graphs` semantics that leave four MCP tools returning empty results for known-present entities (SYN-88) — and ship a basic integration test that documents the consumer-side call pattern for Syntropy-Journals (replacing the dropped SYN-90 design scope).

**Architecture:** Two-phase fix on a single feature branch off `main`. Phase 1 (SYN-89) is mechanical — extract the two broken helpers from the e2e files into a shared module, fix them to read MCP `structuredContent` and iterate chain `edges`, add unit tests so future regressions surface without a live gateway. Phase 2 (SYN-88) is investigation-then-fix — re-run the suite after Phase 1 to subtract the unmasked failures (some "deployment-state" symptoms will turn out to be test-bug shadows), reproduce the real residual bugs locally with diagnostic Cypher against Aura, then patch `scoped_server.py` (`_build_traverse_cypher` and/or `/graphs`) to fix the entity-resolution semantics. Phase 3 ships a small `mcp/tests/integration/` directory containing one test file that demonstrates the exact consumer-side call pattern for Syntropy-Journals to copy.

**Tech Stack:** Python 3.10+ · pytest · pytest-asyncio · httpx · Neo4j Aura (free tier `b7dbceab`) · FastAPI scoped_server · MCP streamable-HTTP transport · OpenRouter for embed/LLM.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `mcp/tests/e2e/_helpers.py` | **create** | Shared `_payload` and `_extract_source_ids` helpers — single source of truth for MCP-envelope unwrapping and chain-shape parsing |
| `mcp/tests/e2e/test_kg_coverage_probes.py` | modify | Remove inline `_payload`/`_is_error`; import from `._helpers` |
| `mcp/tests/e2e/test_tool_roundtrips.py` | modify | Remove inline `_extract_chains`/`_extract_entity_id`; import `_extract_source_ids` from `._helpers` |
| `mcp/tests/unit/test_e2e_helpers.py` | **create** | Unit tests for the shared helpers — runnable without `KG_MCP_E2E_URL` |
| `shrine-diet-bioactivity/lightrag/scoped_server.py` | modify | `_build_traverse_cypher` (lines 403+) and/or `/graphs` route — exact change depends on Phase 2 investigation |
| `shrine-diet-bioactivity/lightrag/test_scoped_server_typed.py` | modify | Add a regression test that reproduces the live e2e gateway failure locally |
| `mcp/tests/integration/__init__.py` | **create** | Marks the directory as a pytest package |
| `mcp/tests/integration/conftest.py` | **create** | Reusable `mcp_call` fixture — copy-able pattern for Syntropy-Journals to lift |
| `mcp/tests/integration/test_consumer_smoke.py` | **create** | Three-test "basic integration" suite that any consumer can run as a copy-paste reference |
| `mcp/pyproject.toml` | modify | Register new pytest mark `integration` (so the suite gates the same way `e2e` does) |

---

## Pre-flight

- [ ] **Step 0.1: Create the working branch**

Run:
```bash
cd /home/mo/projects/SyntropyHealth/apps/shrine-diet-bioactivity
git fetch origin main
git checkout -B fix/kg-mcp-ops-hardening origin/main
```
Expected: `Switched to a new branch 'fix/kg-mcp-ops-hardening'`.

- [ ] **Step 0.2: Close SYN-90 as won't-do**

Per the 2026-05-26 conversation, the SYN-90 design ticket is dropped in favour of shipping a basic integration test directly. Close it via Linear MCP:
```
mcp__linear__save_issue(id="SYN-90", state="Cancelled")
```
And add a one-line comment via `mcp__linear__save_comment`: "Superseded by 2026-05-26 ops-hardening plan; basic integration test lands in same PR."

---

## Phase 1 — SYN-89: Fix the test-suite helpers (TDD)

### Task 1: Extract helpers to a shared module (no behaviour change)

**Files:**
- Create: `mcp/tests/e2e/_helpers.py`
- Modify: `mcp/tests/e2e/test_kg_coverage_probes.py` (remove inline `_is_error` and `_payload`)
- Modify: `mcp/tests/e2e/test_tool_roundtrips.py` (remove inline `_extract_chains` and `_extract_entity_id`)

- [ ] **Step 1.1: Create the helpers module with the ORIGINAL (buggy) behaviour first**

This is a refactor-then-fix sequence. Move the helpers exactly as they exist now so the e2e tests behave identically. The fix lands in Task 2 (so the diff cleanly separates "extract" from "fix").

Create `mcp/tests/e2e/_helpers.py`:
```python
"""Shared helpers for the e2e probe suite.

Single source of truth for two repeatedly-broken concerns:

* ``_payload`` — unwrap the MCP ``tools/call`` envelope to reach the
  typed tool output. MCP wraps every tool's return in
  ``{"content":[{"type":"text","text":"<json>"}], "structuredContent":{...},
  "isError": bool}``. Probes want ``structuredContent``; the text wrapper
  is for display.

* ``_extract_source_ids`` — pull the ``source_id`` attribution off
  chain edges. Chains from Layer-B traversals are
  ``{"edges":[{"src_id","tgt_id","rel_type","source_id",...}]}`` —
  the source_id lives one level deep, on each edge.

Keep this module dependency-light (no httpx, no pytest fixtures) so the
unit suite in ``mcp/tests/unit/test_e2e_helpers.py`` can import + assert
without booting a live gateway.
"""
from __future__ import annotations

import json
from typing import Any


def _is_error(result: dict) -> bool:
    """True if the JSON-RPC envelope carries an error."""
    return "error" in result and result.get("error") is not None


def _payload(result: dict) -> dict:
    """The tool result payload (the structured tool output).

    NOTE: this function is the SYN-89 fix point. Currently returns the
    outer envelope, which is wrong (the envelope keys are
    ``content``/``structuredContent``/``isError``, not the typed
    payload). Task 2 fixes this — keeping the buggy body here verbatim
    so the extract-then-fix diff is bisectable.
    """
    payload = result.get("result", {})
    return payload if isinstance(payload, dict) else {}


def _extract_chains(result: dict) -> list:
    """Pull a chains list out of an MCP tools/call response.

    Tries (in order):
      1. ``result.chains`` (direct JSON return)
      2. ``result.data.chains`` (envelope variant)
      3. ``result.content[0].text`` parsed as JSON, then ``.chains``

    Returns ``[]`` when no chains can be located.
    """
    payload = result.get("result", {})
    if not isinstance(payload, dict):
        return []
    chains = payload.get("chains")
    if chains:
        return chains
    data = payload.get("data")
    if isinstance(data, dict) and data.get("chains"):
        return data["chains"]
    content_list = payload.get("content")
    if isinstance(content_list, list) and content_list:
        first = content_list[0]
        if isinstance(first, dict):
            text = first.get("text", "")
            if text:
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return []
                if isinstance(parsed, dict):
                    return parsed.get("chains") or parsed.get("data", {}).get("chains") or []
    return []


def _extract_source_ids(chain: Any) -> list[str]:
    """Pull ``source_id`` values from a chain.

    NOTE: this function is the SYN-89 fix point. The current
    ``_extract_entity_id`` (kept for backward compat below) checks the
    chain dict for ``entity_id``/``id``/``source_id`` — but chains are
    ``{"edges":[{...}]}`` and the source_id lives on each EDGE, not on
    the chain. Task 3 fixes this; the body here is the buggy version
    preserved for bisectability.
    """
    # BUG (fixed in Task 3): doesn't descend into edges.
    if isinstance(chain, dict):
        sid = chain.get("entity_id") or chain.get("id") or chain.get("source_id")
        return [sid] if sid else []
    return []
```

- [ ] **Step 1.2: Replace inline helpers in `test_kg_coverage_probes.py`**

Edit `mcp/tests/e2e/test_kg_coverage_probes.py`. Find the existing inline definitions (lines ~47-55):
```python
def _is_error(result: dict) -> bool:
    return "error" in result and result.get("error") is not None


def _payload(result: dict) -> dict:
    payload = result.get("result", {})
    return payload if isinstance(payload, dict) else {}
```
Replace those two function definitions with a single import line directly under the existing imports (right after the `from ._braintrust_logger import bt_span` line):
```python
from ._helpers import _is_error, _payload  # noqa: F401 — re-exported for in-module reads
```

- [ ] **Step 1.3: Replace inline helpers in `test_tool_roundtrips.py`**

Edit `mcp/tests/e2e/test_tool_roundtrips.py`. Remove the inline `_extract_chains` (~lines 288-326) and `_extract_entity_id` (~lines 329-333) function definitions. Add directly under existing imports:
```python
from ._helpers import _extract_chains, _extract_source_ids  # noqa: F401
```

- [ ] **Step 1.4: Update the source_id parametrized test to call `_extract_source_ids` instead of `_extract_entity_id`**

In `mcp/tests/e2e/test_tool_roundtrips.py` (lines ~260-282), replace the manual chain iteration:
```python
entity_ids: list[str] = []
for chain in chains:
    elements = chain if isinstance(chain, list) else [chain]
    for entity in elements:
        eid = _extract_entity_id(entity)
        if eid:
            entity_ids.append(eid)
```
with:
```python
source_ids: list[str] = []
for chain in chains:
    source_ids.extend(_extract_source_ids(chain))
```
And rename `entity_ids` → `source_ids` everywhere it appears in the function body (including the assertions and span.log fields). Source-id-prefix discipline is what's being validated; the rename clarifies that.

- [ ] **Step 1.5: Run the e2e suite to confirm extract-only refactor is behaviour-equivalent**

Run (with the same `KG_MCP_E2E_URL` + `KG_MCP_API_KEY` that the 2026-05-25 run used):
```bash
cd mcp && uv run pytest -m e2e tests/e2e/ -q --no-header 2>&1 | tail -10
```
Expected: identical fail count to the 2026-05-25 baseline (12 failed, 13 passed, 3 skipped). The refactor doesn't change behaviour; the fix lands in Task 2/3.

- [ ] **Step 1.6: Commit the extract**

```bash
git add mcp/tests/e2e/_helpers.py mcp/tests/e2e/test_kg_coverage_probes.py mcp/tests/e2e/test_tool_roundtrips.py
git commit -m "refactor(mcp): extract e2e helpers to shared module (no behaviour change)

Pulls _is_error, _payload, _extract_chains, and _extract_source_ids
(the buggy version) out of the two e2e test files into
mcp/tests/e2e/_helpers.py. Behaviour-preserving — verified by re-running
the e2e suite against the live gateway and getting the same 12 fail / 13
pass / 3 skip count as the 2026-05-25 baseline.

Sets up SYN-89: the next two commits fix the helpers in place,
and a unit test file gives the helpers regression coverage that
doesn't depend on a live gateway."
```

---

### Task 2: Fix `_payload` to descend into `structuredContent` (TDD)

**Files:**
- Create: `mcp/tests/unit/test_e2e_helpers.py`
- Modify: `mcp/tests/e2e/_helpers.py`

- [ ] **Step 2.1: Write the failing unit test for `_payload`**

Create `mcp/tests/unit/test_e2e_helpers.py`:
```python
"""Unit tests for mcp/tests/e2e/_helpers.py.

These run by default (no `e2e` mark, no live-gateway dependency).
Locks in SYN-89's fix — _payload must descend into structuredContent;
_extract_source_ids must iterate chain edges.
"""
from __future__ import annotations

import pytest

# Relative import from sibling test package — mcp/tests has __init__.py
from ..e2e._helpers import (  # type: ignore[import-not-found]
    _extract_chains,
    _extract_source_ids,
    _is_error,
    _payload,
)


pytestmark = [pytest.mark.unit]


# ─── _payload ──────────────────────────────────────────────────────────────


class TestPayload:
    def test_descends_into_structured_content(self):
        envelope = {
            "result": {
                "content": [{"type": "text", "text": "{\"english\": \"X\"}"}],
                "structuredContent": {"english": "Astragalus membranaceus", "confidence": 1.0},
                "isError": False,
            }
        }
        assert _payload(envelope) == {
            "english": "Astragalus membranaceus",
            "confidence": 1.0,
        }

    def test_returns_envelope_when_structured_content_absent(self):
        """Backward-compat: pre-MCP-typed-output gateways returned the
        typed payload at the envelope level."""
        envelope = {"result": {"nodes": [{"id": 1}], "edges": []}}
        assert _payload(envelope) == {"nodes": [{"id": 1}], "edges": []}

    def test_returns_empty_dict_on_no_result_key(self):
        assert _payload({}) == {}

    def test_returns_empty_dict_on_non_dict_result(self):
        assert _payload({"result": None}) == {}
        assert _payload({"result": "string"}) == {}

    def test_isError_envelope_is_still_unwrapped(self):
        """An isError=true response still has structuredContent (with the
        error detail). The probe checks _is_error first; _payload returns
        the structured payload regardless."""
        envelope = {
            "result": {
                "content": [{"type": "text", "text": "{\"detail\": \"...\"}"}],
                "structuredContent": {"detail": "validation failed"},
                "isError": True,
            }
        }
        assert _payload(envelope) == {"detail": "validation failed"}


# ─── _extract_source_ids ──────────────────────────────────────────────────


class TestExtractSourceIds:
    def test_descends_into_chain_edges(self):
        chain = {
            "edges": [
                {"src_id": "Astragalus", "tgt_id": "Aging", "source_id": "duke:treats_symptom"},
                {"src_id": "Astragalus", "tgt_id": "Diabetes", "source_id": "herb2:herb_disease"},
            ]
        }
        assert _extract_source_ids(chain) == [
            "duke:treats_symptom",
            "herb2:herb_disease",
        ]

    def test_returns_empty_for_chain_with_no_edges(self):
        assert _extract_source_ids({"edges": []}) == []
        assert _extract_source_ids({}) == []

    def test_returns_empty_for_non_dict_chain(self):
        assert _extract_source_ids(None) == []
        assert _extract_source_ids("string") == []
        assert _extract_source_ids([]) == []

    def test_handles_flat_entity_shape_backward_compat(self):
        """Earlier chain variants returned flat entity dicts (no edges)
        with the source_id at the top level."""
        chain = {"source_id": "duke:targets_protein", "entity_id": "CURCUMIN"}
        # Edges array absent → backward-compat path
        assert "duke:targets_protein" in _extract_source_ids(chain)

    def test_edge_without_source_id_skipped(self):
        chain = {
            "edges": [
                {"src_id": "A", "tgt_id": "B"},
                {"src_id": "A", "tgt_id": "C", "source_id": "duke:foo"},
            ]
        }
        assert _extract_source_ids(chain) == ["duke:foo"]


# ─── _is_error / _extract_chains (lock current behaviour) ─────────────────


class TestIsError:
    def test_no_error_key(self):
        assert _is_error({"result": {}}) is False

    def test_null_error(self):
        assert _is_error({"error": None}) is False

    def test_real_error(self):
        assert _is_error({"error": {"code": -32603, "message": "internal"}}) is True


class TestExtractChains:
    def test_chains_at_result_level(self):
        envelope = {"result": {"chains": [{"edges": [{"src_id": "A"}]}]}}
        assert _extract_chains(envelope) == [{"edges": [{"src_id": "A"}]}]

    def test_chains_in_content_text(self):
        envelope = {
            "result": {
                "content": [{"type": "text", "text": '{"chains": [{"edges": []}]}'}]
            }
        }
        assert _extract_chains(envelope) == [{"edges": []}]

    def test_no_chains_returns_empty(self):
        assert _extract_chains({"result": {}}) == []
        assert _extract_chains({}) == []
```

- [ ] **Step 2.2: Run the unit test to confirm `_payload` tests fail and `_extract_source_ids` tests fail**

Run:
```bash
cd mcp && uv run pytest tests/unit/test_e2e_helpers.py -v --no-header 2>&1 | tail -25
```
Expected: `TestPayload::test_descends_into_structured_content` FAILS (current `_payload` returns the envelope, not `structuredContent`); `TestPayload::test_returns_envelope_when_structured_content_absent` PASSES; `TestExtractSourceIds::test_descends_into_chain_edges` FAILS (current function doesn't descend into `edges`). All `TestIsError` and `TestExtractChains` tests PASS (those helpers were already correct).

- [ ] **Step 2.3: Fix `_payload` in `mcp/tests/e2e/_helpers.py`**

Replace the `_payload` definition with:
```python
def _payload(result: dict) -> dict:
    """The tool result payload (the structured tool output).

    MCP ``tools/call`` wraps every tool's return in
    ``{"content":[{"type":"text","text":"..."}], "structuredContent":{...},
    "isError": bool}``. The ``structuredContent`` field is the
    Pydantic-validated typed payload — assert against this. The
    ``content[].text`` wrapper is a JSON-string mirror for display.

    Falls back to the envelope itself when ``structuredContent`` is
    absent (pre-typed-output gateway versions).
    """
    envelope = result.get("result", {})
    if not isinstance(envelope, dict):
        return {}
    sc = envelope.get("structuredContent")
    if isinstance(sc, dict):
        return sc
    return envelope
```
Delete the bug-marker comment (`# NOTE: this function is the SYN-89 fix point ...`).

- [ ] **Step 2.4: Run unit tests — `_payload` tests now pass**

Run:
```bash
cd mcp && uv run pytest tests/unit/test_e2e_helpers.py::TestPayload -v --no-header 2>&1 | tail -10
```
Expected: all 5 `TestPayload` tests PASS.

---

### Task 3: Fix `_extract_source_ids` to iterate chain edges

**Files:**
- Modify: `mcp/tests/e2e/_helpers.py`

- [ ] **Step 3.1: Fix `_extract_source_ids`**

Replace the `_extract_source_ids` definition with:
```python
def _extract_source_ids(chain: Any) -> list[str]:
    """Pull ``source_id`` attribution off chain edges.

    Chain shape from Layer-B traversals is
    ``{"edges":[{src_id, tgt_id, rel_type, source_id, ...}]}``;
    the source_id lives one level deep on each edge. Earlier variants
    used flat entity dicts (top-level source_id) — handled as fallback.
    """
    if not isinstance(chain, dict):
        return []
    edges = chain.get("edges")
    if isinstance(edges, list):
        return [
            e["source_id"]
            for e in edges
            if isinstance(e, dict) and e.get("source_id")
        ]
    # Fallback: flat entity-shape chain
    sid = chain.get("source_id") or chain.get("entity_id") or chain.get("id")
    return [sid] if sid else []
```
Delete the `# BUG (fixed in Task 3) ...` marker.

- [ ] **Step 3.2: Run all unit tests**

Run:
```bash
cd mcp && uv run pytest tests/unit/test_e2e_helpers.py -v --no-header 2>&1 | tail -25
```
Expected: all tests PASS (5 TestPayload + 5 TestExtractSourceIds + 3 TestIsError + 3 TestExtractChains = 16 tests pass).

- [ ] **Step 3.3: Re-run e2e suite — expected behaviour changes**

Run:
```bash
cd mcp && uv run pytest -m e2e tests/e2e/ -v --tb=short --no-header 2>&1 | tail -40
```
Expected vs. the 2026-05-25 baseline (12 failed, 13 passed):
- `test_huangqi_trilingual_aliasing` — now PASSES (reads `structuredContent.english`).
- `test_layer_b_source_id_prefixes[kg_compound_to_targets-Curcumin]` — PASSES (extracts `source_id` from edges, validates `duke:` prefix).
- `test_layer_b_source_id_prefixes[kg_herb_to_symptoms-Astragalus membranaceus]` — PASSES.
- `test_curcumin_resolves_to_compound_node` — likely PASSES (was failing on `_payload(...).get("nodes", [])` returning `[]` because the outer envelope has no `nodes` key; now reads from `structuredContent.nodes`).
- `test_herb_node_has_edges` — likely PASSES (same reason).
- `test_t2d_resolves_to_disease_node` — may still FAIL (real KG content gap: "Type 2 diabetes" exact alias absent; node lookup returns 0 even from structuredContent).

Record the actual pass/fail delta — this informs Phase 2's scope.

- [ ] **Step 3.4: Commit SYN-89 fixes**

```bash
git add mcp/tests/e2e/_helpers.py mcp/tests/unit/test_e2e_helpers.py
git commit -m "fix(mcp): SYN-89 — e2e helpers misread MCP envelope and chain shape

Two pre-existing test bugs surfaced by the 2026-05-25 e2e run.

_payload now descends into structuredContent. MCP tools/call wraps
typed tool output in {content:[...], structuredContent:{...}, isError:bool};
probes reading the outer envelope (where the keys are content/
structuredContent/isError, not the typed fields) silently get default
empty values and mis-attribute the cause as KG drift. The huangqi
trilingual aliasing test confirmed this — failure log showed the right
data was in structuredContent and discarded.

_extract_source_ids replaces _extract_entity_id. Chains from /traverse
are {edges:[{src_id, tgt_id, source_id, ...}]}; the documented source-id
prefix attribution lives on edges, not on chain dicts. Old function
returned None for every well-formed chain.

Unit tests in mcp/tests/unit/test_e2e_helpers.py lock both fixes
in place without a live-gateway dependency."
```

---

## Phase 2 — SYN-88: Fix remaining gateway semantics gaps

After Phase 1's `_payload` fix unmasked some e2e failures, what's left in the failed bucket is the *true* SYN-88 scope.

### Task 4: Triage Phase 2 scope from the new e2e baseline

- [ ] **Step 4.1: Re-run the full e2e suite and capture results to a file**

```bash
cd mcp
uv run pytest -m e2e tests/e2e/ -v --tb=line --no-header > /tmp/e2e_post_syn89.txt 2>&1
tail -40 /tmp/e2e_post_syn89.txt
```

- [ ] **Step 4.2: Classify the residual failures**

For each remaining failure, run a Cypher probe directly against Aura `b7dbceab` (via the local scoped_server you started earlier, or a fresh Python `neo4j` session sourcing `shrine-diet-bioactivity/shrine-diet-bioactivity/.env`):
- "0 nodes" or "0 chains" for a seed that DOES exist in Aura → gateway bug (SYN-88 scope).
- "0 nodes" because the named entity isn't in the KG (e.g. `MATCH (n {entity_id: "Type 2 diabetes"}) RETURN n` returns nothing) → KG content gap (NOT SYN-88; document in a separate ticket).

Specifically for the two seeds most likely to be gateway bugs, run:
```bash
cd /home/mo/projects/SyntropyHealth/apps/shrine-diet-bioactivity/shrine-diet-bioactivity/lightrag
python3 -c "
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
load_dotenv(Path('..') / '.env')
with GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD'])) as d:
    with d.session() as s:
        # Q1: Astragalus has ASSOCIATED_WITH_DISEASE edges?
        r = s.run('MATCH (h {entity_id: \$id})-[r:ASSOCIATED_WITH_DISEASE]->(d) RETURN count(r) AS c', id='Astragalus membranaceus').single()
        print('Astragalus → ASSOCIATED_WITH_DISEASE edges:', r['c'])
        # Q2: Does Astragalus have :unified_diet_kg AND :Herb labels?
        r = s.run('MATCH (h {entity_id: \$id}) RETURN labels(h) AS l', id='Astragalus membranaceus').single()
        print('Astragalus labels:', r['l'] if r else None)
        # Q3: Curcumin compound→target→disease depth-2 chain count
        r = s.run('MATCH (c {entity_id: \$id})-[:TARGETS_PROTEIN]->(t)-[:ASSOCIATED_WITH_DISEASE]->(d) RETURN count(*) AS c', id='Curcumin').single()
        print('Curcumin Compound→Target→Disease chains:', r['c'])
        r2 = s.run('MATCH (c {entity_id: \$id})-[:TARGETS_PROTEIN]->(t)-[:ASSOCIATED_WITH_DISEASE]->(d) RETURN count(*) AS c', id='CURCUMIN').single()
        print('CURCUMIN (caps) Compound→Target→Disease chains:', r2['c'])
"
```
Expected outputs let you classify each:
- If labels list **lacks** `Herb` → `_build_traverse_cypher` MATCH on `(start:`unified_diet_kg`:`Herb`)` fails → fix is to add missing labels OR relax the cypher.
- If edges count > 0 but `/traverse` returns 0 → the cypher works but the seed-resolution clause is missing the entity_id case the test sends.
- If both label and edges present and chains>0 → /traverse's depth-2 cypher has a bug; deep-dive into `_build_traverse_cypher` depth=2 branch.

- [ ] **Step 4.3: Commit a triage note**

```bash
git add /dev/null  # placeholder — only commit if you produced an artifact
```
Actually skip — Task 4 produces diagnostic context, not committable artifacts. Move on.

---

### Task 5: Write the failing integration test for SYN-88

**Files:**
- Modify: `shrine-diet-bioactivity/lightrag/test_scoped_server_typed.py`

- [ ] **Step 5.1: Add a regression test mirroring the live failure**

Append to `shrine-diet-bioactivity/lightrag/test_scoped_server_typed.py`:
```python
import pytest


pytestmark_aura_integration = pytest.mark.integration
"""Marker for tests that need a live Aura connection (env: NEO4J_URI/USERNAME/PASSWORD)."""


@pytest.mark.integration
def test_traverse_herb_to_disease_returns_chains_for_known_herb():
    """Reproduces SYN-88 / 2026-05-25 e2e failure.

    `Astragalus membranaceus` has ASSOCIATED_WITH_DISEASE edges in
    Aura `b7dbceab` (verified via direct Cypher). The /traverse
    endpoint with start_label=Herb, depth=1 MUST return >= 1 chain.
    """
    import asyncio
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from httpx import AsyncClient

    load_dotenv(Path(__file__).parent.parent / ".env")
    assert os.environ.get("NEO4J_URI"), "Aura creds required"

    # Spin up scoped_server via TestClient is heavy; this test assumes
    # a local scoped_server on :9621 (started via `make lightrag-server`
    # before running). Skip if not reachable — keeps the test useful
    # both as a local-dev guard and in CI behind LIGHTRAG_RUN_INTEGRATION.
    async def _run():
        async with AsyncClient(timeout=15.0) as c:
            r = await c.post("http://localhost:9621/traverse", json={
                "start_label": "Herb",
                "edge_types": ["ASSOCIATED_WITH_DISEASE"],
                "seed": "Astragalus membranaceus",
                "direction": "outbound",
                "depth": 1,
                "top_k": 5,
                "scope_filter": ["shared"],
            })
            return r

    try:
        resp = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"local scoped_server unreachable: {exc}")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    chains = body.get("chains", [])
    assert len(chains) >= 1, (
        f"Astragalus → ASSOCIATED_WITH_DISEASE returned 0 chains "
        f"(SYN-88). Seeds resolved: {body.get('seeds_resolved')}. "
        f"Raw edge count: {body.get('raw_subgraph_edge_count')}."
    )
```

- [ ] **Step 5.2: Run the test — confirm it FAILS for the right reason**

Ensure the local `scoped_server` is running on `:9621` (started earlier in the session as PID 9059; if dead, restart per the README's "Run locally" section). Then:
```bash
cd shrine-diet-bioactivity/lightrag
LIGHTRAG_RUN_INTEGRATION=true python3 -m pytest test_scoped_server_typed.py::test_traverse_herb_to_disease_returns_chains_for_known_herb -v 2>&1 | tail -10
```
Expected: FAIL with `assert 0 >= 1` and the diagnostic message showing `seeds_resolved=[]` (or similar).

If the test PASSES — the bug doesn't reproduce locally. That means the failure is deployment-state-specific (Railway container has a different code version than `main`). In that case the fix path is to rebuild + redeploy with the latest `scoped_server.py`, not to patch code. Skip to Task 7's verification.

---

### Task 6: Implement the SYN-88 fix in `scoped_server.py`

This task has three concrete branches — pick the one matching Task 4's findings. **Do NOT do all three.** Each branch ends with the same verification (Task 6.X).

#### Branch A — Labels missing on entity nodes

Symptom: `labels(n)` for `Astragalus membranaceus` does NOT include `Herb`.

Root cause: the post-ingest `ALTER ... ADD LABEL` step in `ingest_unified.py` / `ingest_direct.py` either didn't run or was lost. The traverse cypher requires `(start:`workspace`:`start_label`)` so missing labels → 0 nodes match.

- [ ] **Step 6.A.1: Add a one-shot label backfill task to scoped_server bootstrap**

In `shrine-diet-bioactivity/lightrag/bootstrap_scope.py`, after the existing scope tagging block, add:
```python
# Backfill entity_type as a Neo4j label so /traverse multi-label MATCH
# can find seeds. Idempotent — only writes when missing.
async def backfill_entity_type_labels(driver, workspace: str) -> int:
    """Set the entity_type as a Neo4j label on nodes that don't have it.

    Returns the count of nodes updated. Run after ingest as part of
    scope bootstrap so /traverse never finds nodes labeled only with
    entity_id.
    """
    from .scoped_server import _safe_label

    ws = _safe_label(workspace)
    cypher = (
        f"MATCH (n:`{ws}`) "
        f"WHERE n.entity_type IS NOT NULL "
        f"  AND NOT n.entity_type IN labels(n) "
        f"WITH n, n.entity_type AS et "
        f"CALL apoc.create.addLabels(n, [et]) YIELD node "
        f"RETURN count(node) AS c"
    )
    # If APOC isn't installed on the Aura free tier, fall back to the
    # static-CALL-per-type path:
    async with driver.session() as s:
        try:
            r = await s.run(cypher)
            rec = await r.single()
            return rec["c"] if rec else 0
        except Exception:
            # Aura Free doesn't always have apoc; use static labels.
            total = 0
            for et in ("Herb", "Compound", "Food", "Target", "Disease", "Symptom"):
                r = await s.run(
                    f"MATCH (n:`{ws}`) WHERE n.entity_type = $et "
                    f"  AND NOT '{et}' IN labels(n) "
                    f"SET n:`{et}` RETURN count(n) AS c",
                    et=et,
                )
                rec = await r.single()
                total += rec["c"] if rec else 0
            return total
```
Wire this into `bootstrap_scope.py`'s main() after the scope tagging step. Run it once:
```bash
cd shrine-diet-bioactivity/lightrag
python3 bootstrap_scope.py --config local --backfill-labels
```

#### Branch B — Seed-resolution clause too strict

Symptom: labels are correct, edges exist in Cypher, but `/traverse` returns 0 chains for the live seed.

Root cause: `_build_traverse_cypher`'s WHERE clause (lines 437-443) tries `entity_id`, `common_name`, `aliases`, `pubchem_cid`. If the seed is "Astragalus membranaceus" but the stored `entity_id` has trailing whitespace, different unicode normalization, or the seed-as-typed appears only in a Neo4j *label* not in the *property*, the WHERE clause matches zero rows.

- [ ] **Step 6.B.1: Add a label-equality fallback to the WHERE clause**

In `shrine-diet-bioactivity/lightrag/scoped_server.py`, around line 441, extend the WHERE clause:
```python
return (
    f"MATCH (start:`{ws}`:`{sl}`) "
    f"WHERE start.scope IN $scope_filter "
    f"  AND ("
    f"    toLower(start.entity_id) = toLower($seed) "
    f"    OR toLower(coalesce(start.common_name, '')) = toLower($seed) "
    f"    OR any(_a IN coalesce(start.aliases, []) WHERE toLower(_a) = toLower($seed)) "
    f"    OR (start.pubchem_cid IS NOT NULL AND toString(start.pubchem_cid) = $seed) "
    f"    OR any(_lbl IN labels(start) WHERE toLower(_lbl) = toLower($seed)) "
    f"  ) "
    # ... rest unchanged
)
```
The new line lets nodes labeled with the seed (e.g. `:Astragalus membranaceus`) match even when their `entity_id` property has drifted.

#### Branch C — Depth-2 chain Cypher bug

Symptom: depth=1 works; depth=2 (`kg_compound_to_diseases`) returns 0 chains even when Cypher `MATCH (c)-[:TARGETS_PROTEIN]->(t)-[:ASSOCIATED_WITH_DISEASE]->(d) RETURN count(*)` returns > 0.

Root cause: `_build_traverse_cypher`'s depth=2 branch (read at lines 444+; not shown in this plan but accessible via `sed -n '444,510p' shrine-diet-bioactivity/lightrag/scoped_server.py`) likely chains the edge_types in the wrong order, has a wrong direction arrow on the second edge, or omits a scope filter on the intermediate / target node.

- [ ] **Step 6.C.1: Read the depth=2 branch, compare to a reference Cypher**

Run:
```bash
sed -n '444,510p' shrine-diet-bioactivity/lightrag/scoped_server.py
```
Identify the depth=2 cypher template. The reference shape SHOULD be (for `kg_compound_to_diseases` = Compound `-TARGETS_PROTEIN-> Target -ASSOCIATED_WITH_DISEASE-> Disease`):
```
MATCH (start:`unified_diet_kg`:`Compound`)
WHERE <seed match>
  AND start.scope IN $scope_filter
MATCH (start)-[r1:`TARGETS_PROTEIN`]->(mid)
WHERE mid.scope IN $scope_filter
MATCH (mid)-[r2:`ASSOCIATED_WITH_DISEASE`]->(tgt)
WHERE tgt.scope IN $scope_filter
RETURN start.entity_id AS src_id, mid.entity_id AS mid_id, tgt.entity_id AS tgt_id,
       type(r1) AS rel_type_1, r1.description AS description_1, r1.source_id AS source_id_1,
       type(r2) AS rel_type_2, r2.description AS description_2, r2.source_id AS source_id_2
LIMIT $top_k
```

- [ ] **Step 6.C.2: Patch the depth=2 branch**

Replace whatever you find with the reference shape above, ensuring:
- Both intermediate (`mid`) and target (`tgt`) carry the scope filter.
- The two edge types come from `edge_types[0]` (start→mid) and `edge_types[1]` (mid→tgt) respectively.
- The RETURN matches the keys `/traverse` reads at lines 581-606 (`src_id`, `mid_id`, `tgt_id`, `rel_type_1`/`2`, `description_1`/`2`, `source_id_1`/`2`).

---

### Task 6.X: Verify SYN-88 fix (common to all branches)

- [ ] **Step 6.X.1: Restart local scoped_server with the patched code**

If your local scoped_server is still running (from earlier in the session), kill it and restart:
```bash
pkill -f "uvicorn scoped_server" 2>/dev/null
cd shrine-diet-bioactivity/lightrag
set -a; source ../.env; set +a
export LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
nohup uvicorn scoped_server:app --host 0.0.0.0 --port 9621 > /tmp/scoped_server.log 2>&1 &
sleep 5
curl -sS http://localhost:9621/health | head -1
```

- [ ] **Step 6.X.2: Re-run the regression test from Task 5**

```bash
cd shrine-diet-bioactivity/lightrag
LIGHTRAG_RUN_INTEGRATION=true python3 -m pytest test_scoped_server_typed.py::test_traverse_herb_to_disease_returns_chains_for_known_herb -v 2>&1 | tail -5
```
Expected: PASS.

- [ ] **Step 6.X.3: Commit the SYN-88 fix**

```bash
git add shrine-diet-bioactivity/lightrag/scoped_server.py shrine-diet-bioactivity/lightrag/test_scoped_server_typed.py
# If you took Branch A also include bootstrap_scope.py
git commit -m "fix(scoped_server): SYN-88 — <one-line root cause from Task 4 findings>

<one paragraph describing what was wrong and what the patch does.
include the Cypher diff if Branch C, or the WHERE-clause extension
if Branch B, or the label-backfill rationale if Branch A.>

Regression test in test_scoped_server_typed.py reproduces the live
gateway failure for Astragalus membranaceus → ASSOCIATED_WITH_DISEASE
and now passes against the patched scoped_server."
```

---

## Phase 3 — Basic integration test (replaces SYN-90)

A small, copy-paste-able reference suite that demonstrates the exact consumer-side call pattern for Syntropy-Journals. Lives in a new `mcp/tests/integration/` directory so its mark is separate from `e2e` (consumers may want to lift just this).

### Task 7: Scaffold the integration test directory

**Files:**
- Create: `mcp/tests/integration/__init__.py`
- Create: `mcp/tests/integration/conftest.py`
- Create: `mcp/tests/integration/test_consumer_smoke.py`
- Modify: `mcp/pyproject.toml`

- [ ] **Step 7.1: Register the `integration` pytest mark**

Edit `mcp/pyproject.toml`. Find the `[tool.pytest.ini_options]` block. Add to the `markers` list:
```toml
markers = [
    "e2e: live HTTP tests against the deployed MCP gateway (requires KG_MCP_E2E_URL)",
    "integration: consumer-side integration tests — basic smoke that any MCP client (Syntropy-Journals, custom agent) can copy. Same env gating as e2e.",
]
```
And update `addopts` to ALSO exclude `integration` by default:
```toml
addopts = ["-m", "not e2e and not integration"]
```

- [ ] **Step 7.2: Create `__init__.py` and the lifted `conftest.py`**

Create empty `mcp/tests/integration/__init__.py`.

Create `mcp/tests/integration/conftest.py`:
```python
"""Fixtures for the consumer-side basic integration suite.

This conftest is intentionally a copy-paste-friendly snapshot of the
``mcp/tests/e2e/conftest.py`` fixtures. A consumer (e.g. the
Syntropy-Journals backend) can lift this file verbatim into their own
test tree and only re-point the env-var names to whatever they use.

Gated on ``KG_MCP_E2E_URL`` and ``KG_MCP_API_KEY`` — tests that
depend on the ``mcp_call`` fixture are skipped when either env var is
unset (so `pytest -m integration` in CI without those secrets doesn't
fail; it just records skips).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import httpx
import pytest


GATEWAY_URL = os.environ.get("KG_MCP_E2E_URL")
GATEWAY_KEY = os.environ.get("KG_MCP_API_KEY")


def _mcp_headers(token: str | None) -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _parse_sse_or_json(text: str) -> dict:
    """Streamable-HTTP returns SSE or JSON depending on the transport."""
    for line in text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return json.loads(text)


@pytest.fixture
def mcp_call() -> Callable[..., dict[str, Any]]:
    """Three-step MCP streamable-HTTP handshake reduced to one callable.

    Returns ``call(tool_name, args) -> jsonrpc_envelope``. Reuses the
    same shape as ``mcp/tests/e2e/conftest.py::mcp_call`` so probes
    can be copied between dirs.
    """
    if not GATEWAY_URL:
        pytest.skip("KG_MCP_E2E_URL not set")
    if not GATEWAY_KEY:
        pytest.skip("KG_MCP_API_KEY not set")

    def _call(tool_name: str, args: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(
                f"{GATEWAY_URL}/mcp",
                headers=_mcp_headers(GATEWAY_KEY),
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "integration-smoke", "version": "0.1"},
                    },
                },
            )
            assert r.status_code == 200, f"initialize: {r.status_code} {r.text}"
            sid = r.headers.get("mcp-session-id")
            assert sid, "gateway did not return mcp-session-id"
            h2 = {**_mcp_headers(GATEWAY_KEY), "mcp-session-id": sid}
            c.post(
                f"{GATEWAY_URL}/mcp",
                headers=h2,
                json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            )
            r = c.post(
                f"{GATEWAY_URL}/mcp",
                headers=h2,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": args},
                },
            )
            assert r.status_code == 200, f"tools/call {tool_name!r}: {r.status_code} {r.text}"
            return _parse_sse_or_json(r.text)

    return _call
```

- [ ] **Step 7.3: Create the three-test smoke**

Create `mcp/tests/integration/test_consumer_smoke.py`:
```python
"""Basic consumer-side integration smoke.

The minimum a downstream MCP client (Syntropy-Journals chat agent,
custom SDK loop, Claude Desktop) needs to know works before relying
on kg-mcp in a user flow. Three tests:

  1. /health responds 200 without auth — service is up.
  2. tools/list returns the 10 documented MCP tools — contract intact.
  3. One Layer-B traversal returns >= 1 chain with documented
     source-id prefix — the KG actually answers a real query.

Lift this file (and conftest.py) verbatim into a consumer repo and
only re-point the env-var names if you carry a different token type.
"""
from __future__ import annotations

import json
import os
import re

import httpx
import pytest


pytestmark = [pytest.mark.integration]


_SOURCE_PREFIX = re.compile(r"^(cmaup|duke|herb2|symmap|hdi-safe-50|opentcm|food):", re.IGNORECASE)

EXPECTED_TOOLS = {
    "kg_query",
    "kg_diet_to_compounds",
    "kg_compound_to_targets",
    "kg_compound_to_diseases",
    "kg_herb_to_diseases",
    "kg_herb_to_symptoms",
    "kg_compound_to_symptoms",
    "kg_hdi_check",
    "kg_bilingual_term",
    "kg_node_neighborhood",
}


def test_gateway_health_returns_200():
    """Service-up probe — no auth, no MCP. Catches Railway/proxy outages
    before consumer code tries to authenticate."""
    url = os.environ.get("KG_MCP_E2E_URL")
    if not url:
        pytest.skip("KG_MCP_E2E_URL not set")
    r = httpx.get(f"{url}/health", timeout=10.0)
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"


def test_tools_list_contract(mcp_call):
    """The gateway advertises all 10 documented tools — consumer code
    that hardcodes tool names won't silently break on a deploy."""
    # tools/list isn't tools/call; build the handshake inline for clarity.
    url = os.environ["KG_MCP_E2E_URL"]
    key = os.environ["KG_MCP_API_KEY"]
    h = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    with httpx.Client(timeout=30.0) as c:
        r = c.post(
            f"{url}/mcp",
            headers=h,
            json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "consumer-smoke", "version": "0.1"},
                },
            },
        )
        assert r.status_code == 200
        sid = r.headers["mcp-session-id"]
        h2 = {**h, "mcp-session-id": sid}
        c.post(
            f"{url}/mcp", headers=h2,
            json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        )
        r = c.post(
            f"{url}/mcp", headers=h2,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        assert r.status_code == 200
        body = (
            json.loads(r.text.split("data: ", 1)[1])
            if r.text.startswith("event: ") else r.json()
        )
        tools = (body.get("result") or {}).get("tools", [])
        names = {t["name"] for t in tools}
    assert EXPECTED_TOOLS.issubset(names), f"missing tools: {EXPECTED_TOOLS - names}"


def test_herb_to_symptoms_returns_provenance_tagged_chain(mcp_call):
    """End-to-end consumer flow: call a Layer-B tool with a real seed,
    pull a chain out of the typed payload, validate the documented
    source-id prefix. This is what a Syntropy-Journals chat-agent call
    looks like under the hood — if this passes, the agent can ground
    its responses with provenance.
    """
    result = mcp_call(
        "kg_herb_to_symptoms",
        {"seed": "Astragalus membranaceus", "top_k": 3},
    )
    assert "result" in result, f"no result envelope: {result}"
    envelope = result["result"]
    assert not envelope.get("isError"), f"tool returned error: {envelope}"
    payload = envelope.get("structuredContent") or envelope
    chains = payload.get("chains") or []
    assert len(chains) >= 1, f"expected >= 1 chain, got {chains}"
    # Provenance discipline: every edge's source_id matches the
    # documented prefix regex.
    for chain in chains:
        for edge in chain.get("edges", []):
            sid = edge.get("source_id", "")
            assert _SOURCE_PREFIX.match(sid), f"unknown source_id prefix: {sid!r}"
```

- [ ] **Step 7.4: Run the integration suite locally to confirm it passes against the patched gateway**

```bash
cd mcp
export KG_MCP_E2E_URL=https://kg-mcp-test.up.railway.app
export KG_MCP_API_KEY="$KG_MCP_API_KEY"   # ROTATED 2026-08-27 — pull from Infisical, never inline
uv run pytest -m integration tests/integration/ -v --no-header 2>&1 | tail -15
```
Expected: all 3 tests PASS. (The third test relies on SYN-88 being fixed — if Phase 2's fix hasn't deployed to Railway yet, point at a local gateway instead by setting `KG_MCP_E2E_URL=http://localhost:8080` after starting the Python kg_mcp gateway locally per `mcp/README.md` § "Run locally".)

- [ ] **Step 7.5: Commit Phase 3**

```bash
git add mcp/tests/integration/ mcp/pyproject.toml
git commit -m "test(mcp): basic consumer-side integration suite (supersedes SYN-90)

Three tests that any downstream MCP client (Syntropy-Journals chat
agent, custom SDK loop, Claude Desktop) can run against the live
kg-mcp gateway to confirm the consumer-facing contract is intact:

  1. /health responds 200 — service-up probe.
  2. tools/list returns the 10 documented tools — contract intact.
  3. One Layer-B traversal returns >= 1 provenance-tagged chain —
     real KG query lands.

conftest.py is intentionally copy-paste-friendly so consumers can
lift it verbatim into their own test tree.

pyproject.toml registers a new 'integration' mark and excludes it
from default test runs (same gating as 'e2e').

Replaces SYN-90 — the design-then-build cycle would have been
overkill for a suite this small."
```

---

## Phase 4 — Ship

### Task 8: Push, open PR, link Linear

- [ ] **Step 8.1: Push the branch**

```bash
git push -u origin fix/kg-mcp-ops-hardening
```

- [ ] **Step 8.2: Open the PR**

```bash
gh pr create --base main --head fix/kg-mcp-ops-hardening \
  --title "fix(kg-mcp): ops hardening — SYN-88, SYN-89, basic integration smoke" \
  --body "$(cat <<'EOF'
## Summary

Three-phase fix landing the operational hardening from the 2026-05-25 e2e run.

**Phase 1 — SYN-89**: e2e helper bugs. Extracted `_payload`/`_extract_source_ids` to `mcp/tests/e2e/_helpers.py`; fixed `_payload` to descend into `structuredContent` (MCP's typed payload field); fixed `_extract_source_ids` to iterate chain `edges[]` (where the documented `source_id` attribution actually lives). New unit suite in `mcp/tests/unit/test_e2e_helpers.py` locks both fixes without a live-gateway dependency.

**Phase 2 — SYN-88**: gateway entity-resolution semantics. <one paragraph describing which branch (A/B/C) applied + the actual patch> Regression test in `shrine-diet-bioactivity/lightrag/test_scoped_server_typed.py` reproduces the live failure.

**Phase 3 — basic integration test (replaces SYN-90)**: `mcp/tests/integration/` — three-test smoke any downstream MCP client (Syntropy-Journals, custom agent) can copy verbatim. Health, tools/list contract, one provenance-tagged Layer-B chain. Registered as `pytest -m integration`, gated the same way as `e2e`.

## Test plan

- [x] `mcp/tests/unit/test_e2e_helpers.py` — 16 unit tests pass without env gating
- [x] `mcp/tests/e2e/` — re-running against live gateway shows the SYN-89 failures resolve
- [x] `shrine-diet-bioactivity/lightrag/test_scoped_server_typed.py` — SYN-88 regression test fails pre-patch, passes post-patch
- [x] `mcp/tests/integration/` — 3 consumer-smoke tests pass against the patched gateway
- [ ] CI re-runs the full unit + e2e + integration suite (gated)

Linear: **SYN-88**, **SYN-89**, supersedes **SYN-90**.
EOF
)"
```

- [ ] **Step 8.3: Link the PR to both Linear tickets and mark them In Progress**

Use the Linear MCP:
```
mcp__linear__save_issue(
    id="SYN-88", state="In Progress",
    links=[{"url": "<PR URL from gh output>",
            "title": "PR — kg-mcp ops hardening (SYN-88 portion)"}]
)
mcp__linear__save_issue(
    id="SYN-89", state="In Progress",
    links=[{"url": "<PR URL>", "title": "PR — kg-mcp ops hardening (SYN-89 portion)"}]
)
```

---

## Self-review

After writing this plan I checked it against the spec (SYN-88 + SYN-89 + dropped-SYN-90 scope):

- **Coverage:** SYN-89's two bug families both have explicit tasks (Tasks 2, 3) with concrete code + unit tests. SYN-88's investigation has a concrete Cypher probe (Step 4.2) followed by three branched fixes (Tasks 6.A/B/C) with concrete code per branch — engineer picks based on Task 4's findings. The dropped SYN-90 is replaced by Phase 3's three-test smoke.
- **Placeholders:** The only intentionally-deferred content is the PR-body line in Step 8.2 (`<one paragraph describing which branch ...>`) — that's a real commentary slot that requires Phase 2 findings, not a TODO. Branch selection in Task 6 is also intentional — the engineer picks one based on Task 4. No "TBD" / "implement later" / "handle edge cases" patterns.
- **Type consistency:** The helper `_extract_source_ids` is named consistently across creation (Step 1.1), fix (Step 3.1), and consumer (Step 1.4 caller rewrite). The MCP envelope shape (`structuredContent`) is used consistently across `_payload` fix, unit tests, and the consumer smoke at Step 7.3.

No issues found inline. Plan is ready for execution.
