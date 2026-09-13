"""Additive-only ingest invariant (#233b).

The six Layer-B chain tools (compound->target, diet->compound, ...) traverse the
TYPED graph shape (entity-type labels like :Compound, typed edges like
:TARGETS_PROTEIN). A re-ingest that REPLACES rather than adds — deleting existing
typed nodes/edges before rewriting — silently kills those tools even when it
"succeeds". Closing the empty-chunk-layer defect by adding LightRAG chunks/vectors
is legitimate and additive; a wholesale replace is not, and the two are
indistinguishable from an ingest's exit code.

This guard snapshots per-label node counts and per-type relationship counts on a
workspace before and after an operation and asserts nothing DECREASED. New labels
/ types (e.g. VectorChunk appearing for the first time) are allowed; any drop in
an existing count fails closed, naming what shrank.

Shape-agnostic by construction: it enumerates whatever labels/types the workspace
actually carries, so it protects the LightRAG shape locally and the typed
chain-tool shape on Aura with the same code.
"""
from __future__ import annotations

from typing import Any


def _safe_label(label: str) -> str:
    """Escape a workspace label for interpolation (labels cannot be bind params)."""
    return "`" + label.replace("`", "``") + "`"


def snapshot_workspace_counts(session: Any, workspace: str) -> dict[str, dict[str, int]]:
    """Return {'labels': {label: n}, 'rels': {type: n}} for the workspace.

    Node counts are per individual label (a node with two labels contributes to
    both), scoped to the workspace label. Relationship counts are per type, for
    edges whose BOTH endpoints are in the workspace.
    """
    ws = _safe_label(workspace)
    labels: dict[str, int] = {}
    for rec in session.run(
        f"MATCH (n:{ws}) UNWIND labels(n) AS l RETURN l AS label, count(*) AS c"
    ):
        labels[rec["label"]] = rec["c"]
    rels: dict[str, int] = {}
    for rec in session.run(
        f"MATCH (n:{ws})-[r]->(m:{ws}) RETURN type(r) AS t, count(r) AS c"
    ):
        rels[rec["t"]] = rec["c"]
    return {"labels": labels, "rels": rels}


def diff_additive(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> list[str]:
    """Return a list of human-readable violations — every existing count that
    DECREASED. Empty list == additive (safe). New keys are never violations.
    """
    violations: list[str] = []
    for kind in ("labels", "rels"):
        b = before.get(kind, {})
        a = after.get(kind, {})
        for key, before_n in b.items():
            after_n = a.get(key, 0)
            if after_n < before_n:
                noun = "label" if kind == "labels" else "rel-type"
                violations.append(
                    f"{noun} '{key}' shrank {before_n} -> {after_n} "
                    f"(dropped {before_n - after_n})"
                )
    return violations


def _violation_message(violations: list[str]) -> str:
    return (
        "[additive-guard] NON-ADDITIVE ingest — the operation removed graph "
        "elements the chain tools traverse:\n  "
        + "\n  ".join(violations)
        + "\nA replacing re-ingest kills the six Layer-B tools (#233b). If this "
        "shrink is intended, re-run with the ingest's explicit shrink override."
    )


def assert_additive(
    before: dict[str, dict[str, int]], after: dict[str, dict[str, int]]
) -> None:
    """Raise SystemExit naming every shrunk count. Fail-closed: any decrease is a
    replacing re-ingest, which kills the chain tools the panel depends on.
    """
    violations = diff_additive(before, after)
    if violations:
        raise SystemExit(_violation_message(violations))


def evaluate_additive(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
    allow_shrink: bool = False,
) -> str:
    """The full decision the ingest wiring makes, in one testable place.

    Returns a human-readable outcome line and NEVER swallows a violation:
    - additive (no decrease)          -> "[additive-guard] OK ..."
    - shrink with allow_shrink=False  -> raises SystemExit (fail closed)
    - shrink with allow_shrink=True   -> returns a "WARNING ..." line naming the
      violations (fail open BY EXPLICIT OPERATOR CHOICE, and loudly)

    Computing the diff exactly once (assert_additive re-diffs internally; the
    wiring must not diff a third time).
    """
    violations = diff_additive(before, after)
    if not violations:
        return "[additive-guard] OK — ingest was additive (no label/rel-type shrank)"
    if not allow_shrink:
        raise SystemExit(_violation_message(violations))
    return (
        "[additive-guard] WARNING: non-additive ingest permitted by "
        "--allow-shrink:\n  " + "\n  ".join(violations)
    )
