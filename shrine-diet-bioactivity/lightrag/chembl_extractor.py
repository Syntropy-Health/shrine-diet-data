"""ChEMBL bioactivity extraction — compound-anchored intersect.

For a given list of InChIKeys (the project's resolved compound universe),
return all measured bioactivities meeting min_pchembl and min_confidence
thresholds.

Accepts an open sqlite3 connection so callers can choose:
  - a local ChEMBL SQLite dump (production: chembl-downloader auto-fetched)
  - the test fixture at lightrag/tests/fixtures/chembl_subset.sqlite
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable

_BATCH_SIZE = 1000

_SQL = """
SELECT
  cs.standard_inchi_key       AS inchikey,
  md.chembl_id                AS chembl_compound_id,
  td.chembl_id                AS chembl_target_id,
  td.pref_name                AS target_pref_name,
  td.target_type              AS target_type,
  td.organism                 AS target_organism,
  act.standard_type           AS activity_type,
  act.standard_relation       AS relation,
  act.standard_value          AS value,
  act.standard_units          AS units,
  act.pchembl_value           AS pchembl,
  act.activity_comment        AS activity_comment,
  ass.confidence_score        AS assay_confidence,
  doc.chembl_id               AS chembl_doc_id,
  doc.year                    AS publication_year
FROM compound_structures cs
JOIN molecule_dictionary md  ON md.molregno  = cs.molregno
JOIN activities act          ON act.molregno = cs.molregno
JOIN assays ass              ON ass.assay_id = act.assay_id
JOIN target_dictionary td    ON td.tid       = ass.tid
LEFT JOIN docs doc           ON doc.doc_id   = act.doc_id
WHERE cs.standard_inchi_key IN ({placeholders})
  AND act.standard_value IS NOT NULL
  AND (act.standard_relation IS NULL OR act.standard_relation IN ('=', '<', '<='))
  AND ass.confidence_score >= ?
  AND act.pchembl_value >= ?
"""


def extract_bioactivities_for_inchikeys(
    conn: sqlite3.Connection,
    *,
    inchikeys: Iterable[str],
    min_pchembl: float = 5.0,
    min_confidence: int = 5,
) -> list[dict[str, Any]]:
    """Return bioactivity rows for the given InChIKeys.

    Batches the IN-list to stay under SQLite's parameter limit (default 999).
    Filters out NULL standard_value, low-confidence assays, ratios that aren't
    direct measurements (relation must be '=', '<', or '<='), and rows below
    the pChEMBL threshold (default 5.0 ≈ μM potency).
    """
    keys = [k for k in inchikeys if k]
    if not keys:
        return []
    out: list[dict[str, Any]] = []
    for i in range(0, len(keys), _BATCH_SIZE):
        batch = keys[i : i + _BATCH_SIZE]
        placeholders = ",".join("?" * len(batch))
        sql = _SQL.format(placeholders=placeholders)
        params = (*batch, min_confidence, min_pchembl)
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            out.append(dict(zip(cols, row)))
    return out


def filter_min_independent_docs(
    rows: list[dict[str, Any]], min_independent_docs: int = 2
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Third specificity guard (#233b spike): keep only (compound, target) pairs
    corroborated by >= ``min_independent_docs`` DISTINCT ChEMBL doc_ids.

    Two activities from the same publication are ONE piece of evidence, not two —
    so we count distinct non-null ``chembl_doc_id`` per pair, not row count. This
    is the guard against single-paper artefacts and promiscuous/PAINS compounds
    (a curcumin-style hit against ~467 targets collapses to the handful with
    independent replication). A NULL/empty doc contributes no independent source.

    Returns ``(kept_rows, stats)``; stats carries the audit trail (rows/pairs in
    vs kept) so the ingest can report what the guard removed rather than silently
    dropping it.
    """
    from collections import defaultdict

    def _pair(r: dict[str, Any]) -> tuple[Any, Any]:
        return (r.get("chembl_compound_id"), r.get("chembl_target_id"))

    docs_by_pair: dict[tuple[Any, Any], set] = defaultdict(set)
    for r in rows:
        doc = r.get("chembl_doc_id")
        if doc not in (None, ""):
            docs_by_pair[_pair(r)].add(doc)

    kept_pairs = {
        p for p, docs in docs_by_pair.items() if len(docs) >= min_independent_docs
    }
    kept = [r for r in rows if _pair(r) in kept_pairs]
    stats = {
        "min_independent_docs": min_independent_docs,
        "rows_in": len(rows),
        "rows_kept": len(kept),
        "rows_dropped": len(rows) - len(kept),
        "pairs_in": len({_pair(r) for r in rows}),
        "pairs_kept": len(kept_pairs),
    }
    return kept, stats
