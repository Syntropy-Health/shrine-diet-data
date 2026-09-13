"""Both-arm tests for the additive-only ingest guard (#233b).

Pure-logic + stubbed-session arms are unit (no I/O). The live-Neo4j arm (scratch
workspace) is integration/aura — it runs only when NEO4J_URI is reachable.
"""
from __future__ import annotations

import os

import pytest

from additive_guard import (
    assert_additive,
    diff_additive,
    evaluate_additive,
    snapshot_workspace_counts,
    _safe_label,
)


# ---- pure-logic arms (no I/O) — unit ----

@pytest.mark.unit
def test_additive_passes_when_counts_grow_or_hold():
    before = {"labels": {"Compound": 100, "Target": 10}, "rels": {"TARGETS_PROTEIN": 50}}
    after = {"labels": {"Compound": 100, "Target": 12, "VectorChunk": 5}, "rels": {"TARGETS_PROTEIN": 60}}
    assert diff_additive(before, after) == []
    assert_additive(before, after)  # must not raise


@pytest.mark.unit
def test_replacing_shrink_is_a_violation():
    before = {"labels": {"Compound": 100}, "rels": {"TARGETS_PROTEIN": 50}}
    after = {"labels": {"Compound": 40}, "rels": {"TARGETS_PROTEIN": 50}}
    v = diff_additive(before, after)
    assert len(v) == 1 and "Compound" in v[0] and "100 -> 40" in v[0]
    with pytest.raises(SystemExit):
        assert_additive(before, after)


@pytest.mark.unit
def test_dropped_rel_type_raises():
    # A rel type vanishing entirely (after count 0) is the chain-tool-killer case —
    # prove it both surfaces in the diff AND raises.
    before = {"labels": {"Compound": 1}, "rels": {"TARGETS_PROTEIN": 6465}}
    after = {"labels": {"Compound": 1}, "rels": {}}
    v = diff_additive(before, after)
    assert len(v) == 1 and "TARGETS_PROTEIN" in v[0] and "6465 -> 0" in v[0]
    with pytest.raises(SystemExit):
        assert_additive(before, after)


@pytest.mark.unit
def test_new_labels_and_types_are_not_violations():
    before = {"labels": {"Compound": 1}, "rels": {}}
    after = {"labels": {"Compound": 1, "VectorChunk": 99}, "rels": {"DIRECTED": 10}}
    assert diff_additive(before, after) == []


@pytest.mark.unit
def test_fresh_workspace_is_all_additive():
    before = {"labels": {}, "rels": {}}
    after = {"labels": {"Compound": 500}, "rels": {"TARGETS_PROTEIN": 200}}
    assert diff_additive(before, after) == []
    assert_additive(before, after)  # first ingest never violates


@pytest.mark.unit
def test_multiple_simultaneous_violations_all_named():
    before = {"labels": {"Compound": 100}, "rels": {"TARGETS_PROTEIN": 50}}
    after = {"labels": {"Compound": 90}, "rels": {"TARGETS_PROTEIN": 10}}
    v = diff_additive(before, after)
    assert len(v) == 2
    joined = "\n".join(v)
    assert "Compound" in joined and "TARGETS_PROTEIN" in joined


# ---- evaluate_additive: the wiring's decision, incl. the fail-OPEN override ----

@pytest.mark.unit
def test_evaluate_additive_ok_line_when_additive():
    before = {"labels": {"Compound": 1}, "rels": {}}
    after = {"labels": {"Compound": 2}, "rels": {}}
    out = evaluate_additive(before, after, allow_shrink=False)
    assert out.startswith("[additive-guard] OK")


@pytest.mark.unit
def test_evaluate_additive_fails_closed_on_shrink():
    before = {"labels": {"Compound": 100}, "rels": {}}
    after = {"labels": {"Compound": 40}, "rels": {}}
    with pytest.raises(SystemExit):
        evaluate_additive(before, after, allow_shrink=False)


@pytest.mark.unit
def test_evaluate_additive_allow_shrink_warns_not_raises():
    # The single most dangerous branch: --allow-shrink must NOT raise, but MUST
    # loudly name the violation (never silently swallow a chain-tool-killer).
    before = {"labels": {"Compound": 100}, "rels": {}}
    after = {"labels": {"Compound": 40}, "rels": {}}
    out = evaluate_additive(before, after, allow_shrink=True)  # must not raise
    assert "WARNING" in out and "Compound" in out and "100 -> 40" in out


# ---- stubbed-session snapshot arm (no live Neo4j) — unit ----

class _FakeSession:
    """Returns canned records per Cypher shape (labels vs rels query)."""

    def __init__(self, label_rows, rel_rows):
        self._label_rows = label_rows
        self._rel_rows = rel_rows

    def run(self, query, **_):
        if "UNWIND labels" in query:
            return [{"label": l, "c": c} for l, c in self._label_rows]
        return [{"t": t, "c": c} for t, c in self._rel_rows]


@pytest.mark.unit
def test_snapshot_builds_expected_shape_without_neo4j():
    s = _FakeSession([("unified_diet_kg", 12), ("Compound", 9)], [("TARGETS_PROTEIN", 6)])
    snap = snapshot_workspace_counts(s, "unified_diet_kg")
    assert snap == {"labels": {"unified_diet_kg": 12, "Compound": 9},
                    "rels": {"TARGETS_PROTEIN": 6}}


@pytest.mark.unit
def test_safe_label_doubles_backticks():
    assert _safe_label("ok_ws") == "`ok_ws`"
    assert _safe_label("we`ird") == "`we``ird`"  # injection attempt neutralised


# ---- live-Neo4j arm (scratch workspace) — integration/aura ----

def _driver_or_skip():
    try:
        from neo4j import GraphDatabase
    except ImportError:
        pytest.skip("neo4j driver not installed")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD", "localdevpass")
    try:
        d = GraphDatabase.driver(uri, auth=(user, pw))
        with d.session() as s:
            s.run("RETURN 1").single()
        return d
    except Exception:
        pytest.skip(f"Neo4j not reachable at {uri}")


@pytest.mark.integration
@pytest.mark.aura
def test_live_snapshot_and_red_arm():
    d = _driver_or_skip()
    ws = "additive_guard_test_ws"
    try:
        with d.session() as s:
            s.run(f"MATCH (n:`{ws}`) DETACH DELETE n")
            s.run(
                f"CREATE (c1:`{ws}`:Compound),(c2:`{ws}`:Compound),(c3:`{ws}`:Compound),"
                f"(t1:`{ws}`:Target),(t2:`{ws}`:Target),"
                f"(c1)-[:TARGETS_PROTEIN]->(t1),(c2)-[:TARGETS_PROTEIN]->(t2)"
            )
            before = snapshot_workspace_counts(s, ws)
            assert before["labels"]["Compound"] == 3
            assert before["rels"]["TARGETS_PROTEIN"] == 2

            # GREEN arm: add a Compound + a VectorChunk (new label)
            s.run(f"CREATE (c:`{ws}`:Compound),(:`{ws}`:VectorChunk)")
            after_add = snapshot_workspace_counts(s, ws)
            assert_additive(before, after_add)  # additive -> must not raise
            assert after_add["labels"]["Compound"] == 4

            # RED arm: delete a Target (a replacing operation)
            s.run(f"MATCH (t:`{ws}`:Target) WITH t LIMIT 1 DETACH DELETE t")
            after_del = snapshot_workspace_counts(s, ws)
            with pytest.raises(SystemExit) as ei:
                assert_additive(before, after_del)
            assert "Target" in str(ei.value)
    finally:
        with d.session() as s:
            remaining = s.run(f"MATCH (n:`{ws}`) DETACH DELETE n RETURN count(n) AS c")
            # count() over a delete returns rows matched pre-delete; just drain it
            list(remaining)
        d.close()
