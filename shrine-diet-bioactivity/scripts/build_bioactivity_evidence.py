"""Populate bioactivity_evidence in herbal_botanicals.db from a ChEMBL dump.

Reads InChIKeys from compound_identity, intersects against a local ChEMBL
SQLite dump, applies pChEMBL/confidence filters, writes results.

Usage:
  # Smoke (use fixture)
  python scripts/build_bioactivity_evidence.py \\
      --db data_local/herbal_botanicals.db \\
      --chembl-sqlite lightrag/tests/fixtures/chembl_subset.sqlite \\
      --min-pchembl 5.0 --min-confidence 5

  # Production (chembl-downloader fetches ChEMBL 36 SQLite ~12 GB)
  python scripts/build_bioactivity_evidence.py \\
      --db data_local/herbal_botanicals.db \\
      --chembl-version 36
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lightrag"))

from chembl_extractor import (  # noqa: E402
    extract_bioactivities_for_inchikeys,
    filter_min_independent_docs,
)

INSERT_SQL = """
INSERT INTO bioactivity_evidence (
  compound_id, chembl_compound_id, chembl_target_id,
  target_pref_name, target_type, target_organism,
  activity_type, relation, value, units, pchembl,
  activity_comment, assay_confidence, chembl_doc_id,
  publication_year, ingested_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _open_chembl(args: argparse.Namespace) -> sqlite3.Connection:
    if args.chembl_sqlite:
        return sqlite3.connect(args.chembl_sqlite)
    import chembl_downloader  # type: ignore[import-not-found]

    # download_extract_sqlite returns Path by default; force str() because some
    # versions of the function return a typed wrapper that sqlite3.connect
    # rejects without an explicit cast.
    sqlite_path = chembl_downloader.download_extract_sqlite(
        version=str(args.chembl_version)
    )
    sqlite_path_str = str(sqlite_path)
    print(f"Using ChEMBL {args.chembl_version} at {sqlite_path_str}")
    return sqlite3.connect(sqlite_path_str)


def _build_argparser() -> argparse.ArgumentParser:
    description = (__doc__ or "Build bioactivity_evidence").split("\n\n")[0]
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--db", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--chembl-sqlite",
        type=Path,
        help="path to a local ChEMBL SQLite dump or fixture",
    )
    src.add_argument(
        "--chembl-version",
        type=int,
        help="ChEMBL release to fetch via chembl-downloader (e.g. 36)",
    )
    ap.add_argument("--min-pchembl", type=float, default=5.0)
    ap.add_argument("--min-confidence", type=int, default=5)
    ap.add_argument(
        "--min-independent-docs",
        type=int,
        default=2,
        help="Keep only (compound,target) pairs backed by >= N distinct ChEMBL "
        "doc_ids (independent-publication corroboration; guards PAINS/promiscuous "
        "compounds). Set 1 to require just a single publication — note a pair whose "
        "activities all have a NULL doc_id is dropped even then (no value fully "
        "disables the guard, since a NULL doc is not an independent source).",
    )
    return ap


def main() -> int:
    args = _build_argparser().parse_args()

    if not args.db.exists():
        print(f"ERROR: DB not found: {args.db}", file=sys.stderr)
        return 2

    target_conn = sqlite3.connect(args.db)
    target_conn.row_factory = sqlite3.Row
    cur = target_conn.cursor()

    rows = list(
        cur.execute(
            "SELECT inchikey, compound_id FROM compound_identity "
            "WHERE inchikey IS NOT NULL"
        )
    )
    if not rows:
        print(
            "ERROR: compound_identity has no InChIKey rows yet. "
            "Run build-identity first.",
            file=sys.stderr,
        )
        return 3
    inchikey_to_compound: dict[str, str] = {
        r["inchikey"]: r["compound_id"] for r in rows
    }
    print(f"Querying ChEMBL for {len(inchikey_to_compound)} InChIKeys ...")

    chembl_conn = _open_chembl(args)
    bioactivities = extract_bioactivities_for_inchikeys(
        chembl_conn,
        inchikeys=list(inchikey_to_compound),
        min_pchembl=args.min_pchembl,
        min_confidence=args.min_confidence,
    )
    chembl_conn.close()
    print(f"Got {len(bioactivities)} bioactivity rows passing pchembl/confidence filters")

    # Third specificity guard (#233b): require >= N independent doc_ids per
    # (compound, target) pair. Report what it removes rather than dropping silent.
    bioactivities, doc_stats = filter_min_independent_docs(
        bioactivities, min_independent_docs=args.min_independent_docs
    )
    print(
        f"Independent-docs guard (>={doc_stats['min_independent_docs']} distinct "
        f"doc_ids/pair): kept {doc_stats['rows_kept']}/{doc_stats['rows_in']} rows, "
        f"{doc_stats['pairs_kept']}/{doc_stats['pairs_in']} pairs "
        f"(dropped {doc_stats['rows_dropped']} rows on thin evidence)"
    )

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    inserted = 0
    # Atomic DELETE-then-INSERT: if anything throws mid-loop (OOM, SIGTERM,
    # malformed row), the transaction rolls back and the table keeps its
    # pre-run contents instead of being left empty or partially populated.
    # `with target_conn:` commits on clean exit and rolls back on exception.
    try:
        with target_conn:
            cur.execute("DELETE FROM bioactivity_evidence")
            for r in bioactivities:
                compound_id = inchikey_to_compound.get(r["inchikey"])
                if compound_id is None:
                    continue
                cur.execute(
                    INSERT_SQL,
                    (
                        compound_id,
                        r["chembl_compound_id"],
                        r["chembl_target_id"],
                        r["target_pref_name"],
                        r["target_type"],
                        r["target_organism"],
                        r["activity_type"],
                        r["relation"],
                        r["value"],
                        r["units"],
                        r["pchembl"],
                        r["activity_comment"],
                        r["assay_confidence"],
                        r["chembl_doc_id"],
                        r["publication_year"],
                        now_iso,
                    ),
                )
                inserted += 1
    finally:
        target_conn.close()
    print(f"Inserted {inserted} bioactivity_evidence rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
