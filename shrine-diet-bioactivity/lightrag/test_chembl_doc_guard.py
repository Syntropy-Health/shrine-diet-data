"""T4.3 (#233b spike): the >=2-independent-doc_ids-per-pair specificity guard.

The pchembl and confidence guards already existed; this covers the third — the
one that collapses a promiscuous/PAINS compound's long target list to the pairs
with independent-publication replication. Counts DISTINCT doc_ids, not rows: two
activities from one paper are one piece of evidence.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from chembl_extractor import (
    extract_bioactivities_for_inchikeys,
    filter_min_independent_docs,
)

pytestmark = pytest.mark.unit


def _row(cmp, tgt, doc):
    return {"chembl_compound_id": cmp, "chembl_target_id": tgt, "chembl_doc_id": doc}


def test_pair_with_two_distinct_docs_is_kept():
    rows = [_row("C1", "T1", "D1"), _row("C1", "T1", "D2")]
    kept, stats = filter_min_independent_docs(rows, min_independent_docs=2)
    assert len(kept) == 2
    assert stats["pairs_kept"] == 1 and stats["rows_dropped"] == 0


def test_pair_with_one_doc_is_dropped():
    rows = [_row("C1", "T1", "D1")]
    kept, stats = filter_min_independent_docs(rows, min_independent_docs=2)
    assert kept == [] and stats["pairs_kept"] == 0 and stats["rows_dropped"] == 1


def test_two_rows_same_doc_are_not_independent():
    # Two activities from the SAME publication == one piece of evidence -> dropped.
    rows = [_row("C1", "T1", "D1"), _row("C1", "T1", "D1")]
    kept, stats = filter_min_independent_docs(rows, min_independent_docs=2)
    assert kept == [] and stats["pairs_kept"] == 0


def test_null_and_empty_doc_do_not_count_as_a_source():
    rows = [_row("C1", "T1", "D1"), _row("C1", "T1", None), _row("C1", "T1", "")]
    kept, _ = filter_min_independent_docs(rows, min_independent_docs=2)
    assert kept == []  # only one real doc


def test_promiscuous_compound_collapses_to_replicated_pairs():
    # C1 hits T1 (2 docs, real), T2..T5 (1 doc each, thin) — PAINS shape.
    rows = [_row("C1", "T1", "D1"), _row("C1", "T1", "D2")]
    rows += [_row("C1", f"T{i}", f"D{i}") for i in range(2, 6)]
    kept, stats = filter_min_independent_docs(rows, min_independent_docs=2)
    kept_targets = {r["chembl_target_id"] for r in kept}
    assert kept_targets == {"T1"}
    assert stats["pairs_in"] == 5 and stats["pairs_kept"] == 1


def test_min_one_disables_the_guard():
    rows = [_row("C1", "T1", "D1"), _row("C2", "T2", None)]
    kept, _ = filter_min_independent_docs(rows, min_independent_docs=1)
    # threshold 1: any pair with >=1 distinct real doc kept; the null-only pair drops
    assert {r["chembl_target_id"] for r in kept} == {"T1"}


@pytest.mark.integration
def test_extractor_runs_against_fixture_then_guard_applies():
    fx = Path(__file__).parent / "tests" / "fixtures" / "chembl_subset.sqlite"
    if not fx.exists():
        pytest.skip("chembl fixture missing")
    conn = sqlite3.connect(str(fx))
    try:
        keys = [r[0] for r in conn.execute(
            "SELECT standard_inchi_key FROM compound_structures"
        )]
        rows = extract_bioactivities_for_inchikeys(
            conn, inchikeys=keys, min_pchembl=5.0, min_confidence=5
        )
    finally:
        conn.close()
    # extractor produced rows carrying the guard's inputs
    assert rows and all("chembl_doc_id" in r and "chembl_target_id" in r for r in rows)
    kept, stats = filter_min_independent_docs(rows, min_independent_docs=2)
    # fixture pairs each have a single doc -> the guard correctly drops all of them
    # (proving it is ACTIVE end-to-end; the kept-arm is covered by the unit cases)
    assert stats["rows_in"] == len(rows)
    assert stats["pairs_kept"] == 0
