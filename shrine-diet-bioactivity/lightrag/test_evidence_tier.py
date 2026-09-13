"""T4.2 (#233b): evidence_tier populated on chain-tool wire edges.

Unit arms prove the per-rel-type tier derivation. The live arm proves the tier
actually lands on the typed edge property scoped_server /traverse reads
(r.evidence_tier), which 7466 measured empty on the deployed plane.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from entity_schema import evidence_tier_for
from ingest_direct import extract_duke_relationships, upsert_relationships


# ---- unit: tier derivation per rel type ----

@pytest.mark.unit
def test_targets_protein_measured_activity_is_assay():
    assert evidence_tier_for("TARGETS_PROTEIN", {"activity_value": 5.2}) == "assay"
    assert evidence_tier_for("TARGETS_PROTEIN", {"activity_value": 0}) == "assay"  # 0 is a real value


@pytest.mark.unit
def test_targets_protein_no_activity_is_annotated():
    assert evidence_tier_for("TARGETS_PROTEIN", {"activity_value": None}) == "annotated"
    assert evidence_tier_for("TARGETS_PROTEIN", {}) == "annotated"


@pytest.mark.unit
def test_associated_with_disease_passes_layer_through_verbatim():
    # TTD clinical layers are the tier a panel wants to show.
    assert evidence_tier_for("ASSOCIATED_WITH_DISEASE", {"evidence": "Approved"}) == "Approved"
    assert evidence_tier_for("ASSOCIATED_WITH_DISEASE", {"evidence": "Phase 2"}) == "Phase 2"


@pytest.mark.unit
def test_associated_with_disease_empty_layer_is_annotated():
    assert evidence_tier_for("ASSOCIATED_WITH_DISEASE", {"evidence": None}) == "annotated"
    assert evidence_tier_for("ASSOCIATED_WITH_DISEASE", {"evidence": "  "}) == "annotated"


@pytest.mark.unit
def test_has_evidence_pchembl_is_assay():
    assert evidence_tier_for("HAS_EVIDENCE", {"pchembl": 6.1}) == "assay"
    assert evidence_tier_for("HAS_EVIDENCE", {"pchembl": None}) == "annotated"


@pytest.mark.unit
def test_edges_without_structured_evidence_carry_none():
    assert evidence_tier_for("TREATS_SYMPTOM", {}) == ""
    assert evidence_tier_for("CONTAINS_COMPOUND", {}) == ""
    assert evidence_tier_for("FOUND_IN_FOOD", {}) == ""


@pytest.mark.unit
def test_build_relationship_rows_include_evidence_tier():
    # The row dict the ingest writes must carry the key (previously dropped).
    db = Path(__file__).parent / ".." / "data_local" / "herbal_botanicals.db"
    if not db.exists():
        pytest.skip("herbal_botanicals.db not built")
    conn = sqlite3.connect(str(db))
    try:
        rels = extract_duke_relationships(conn, "TARGETS_PROTEIN", max_count=25)
    finally:
        conn.close()
    if not rels:
        pytest.skip("no TARGETS_PROTEIN rows in DB")
    assert all("evidence_tier" in r for r in rels)
    # compound_targets is ~99.8% activity-bearing -> overwhelmingly 'assay'
    assert any(r["evidence_tier"] == "assay" for r in rels)


# ---- live arm: tier lands on the typed edge property (integration/aura) ----

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
def test_evidence_tier_lands_on_typed_edge():
    db = Path(__file__).parent / ".." / "data_local" / "herbal_botanicals.db"
    if not db.exists():
        pytest.skip("herbal_botanicals.db not built")
    d = _driver_or_skip()
    ws = "t42_evidence_test_ws"
    conn = sqlite3.connect(str(db))
    try:
        rels = extract_duke_relationships(conn, "TARGETS_PROTEIN", max_count=30)
        if not rels:
            pytest.skip("no TARGETS_PROTEIN rows")
        with d.session() as s:
            s.run(f"MATCH (n:`{ws}`) DETACH DELETE n")
            # entities the edges need (minimal): src/tgt as workspace nodes
            names = {r["src_id"] for r in rels} | {r["tgt_id"] for r in rels}
            s.run(
                f"UNWIND $names AS n MERGE (x:`{ws}` {{entity_id: n}}) "
                f"SET x.scope='shared'",
                names=list(names),
            )
            upsert_relationships(s, rels, ws, scope="shared")
            # read evidence_tier exactly as scoped_server /traverse does
            got = s.run(
                f"MATCH (a:`{ws}`)-[r:TARGETS_PROTEIN]->(b:`{ws}`) "
                "WHERE r.scope='shared' "
                "RETURN coalesce(r.evidence_tier, '') AS tier, count(*) AS c"
            )
            tiers = {rec["tier"]: rec["c"] for rec in got}
            # the whole point of T4.2: NOT empty on the wire
            assert tiers.get("", 0) == 0, f"empty evidence_tier leaked: {tiers}"
            assert tiers.get("assay", 0) > 0, f"expected assay tier, got {tiers}"
    finally:
        with d.session() as s:
            list(s.run(f"MATCH (n:`{ws}`) DETACH DELETE n"))
        conn.close()
        d.close()


# ---- wiring: the row CARRIES the key, and the edge SET persists it (no real DB) ----

@pytest.mark.unit
def test_extract_emits_evidence_tier_key_inmemory():
    # Guards the row-emission (ingest_direct: "evidence_tier": evidence_tier_for(...))
    # WITHOUT the multi-GB real DB, so deleting that line fails on a clean CI checkout
    # (previously only a skip-prone real-DB test covered it).
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE compounds (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute(
        "CREATE TABLE bioactivity_evidence "
        "(id INTEGER PRIMARY KEY, compound_id INTEGER, pchembl REAL, activity_type TEXT)"
    )
    conn.execute("INSERT INTO compounds VALUES (1, 'Curcumin')")
    conn.execute("INSERT INTO bioactivity_evidence VALUES (1, 1, 6.1, 'IC50')")
    conn.commit()
    rels = extract_duke_relationships(conn, "HAS_EVIDENCE", 10)
    conn.close()
    assert rels, "expected a HAS_EVIDENCE row from the seed"
    assert all("evidence_tier" in r for r in rels)   # the KEY is emitted on the row
    assert rels[0]["evidence_tier"] == "assay"        # pchembl present -> assay


@pytest.mark.unit
def test_upsert_relationships_sets_evidence_tier_on_wire_edge():
    # Guards the SET r.evidence_tier = row.evidence_tier clause — the one that actually
    # lands the tier on the typed edge scoped_server /traverse reads. Mock the sync
    # session so no Neo4j is needed (the aura arm is routinely skipped).
    from unittest.mock import MagicMock
    session = MagicMock()
    rels = [{
        "src_id": "Curcumin", "tgt_id": "1", "rel_type": "HAS_EVIDENCE",
        "description": "d", "keywords": "k", "weight": 1.0,
        "file_path": "f", "source_id": "s", "evidence_tier": "assay",
    }]
    n = upsert_relationships(session, rels, "unified_diet_kg", scope="shared")
    assert n == 1
    cyphers = [c.args[0] for c in session.run.call_args_list]
    assert any("r.evidence_tier = row.evidence_tier" in c for c in cyphers)
    payloads = [c.kwargs.get("rows") for c in session.run.call_args_list if c.kwargs.get("rows")]
    assert any(any(row.get("evidence_tier") == "assay" for row in rp) for rp in payloads)
