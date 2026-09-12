#!/usr/bin/env python3
"""Decide whether a Railway deployment ADVANCED and reached a terminal state.

WHY THIS EXISTS — two months of green deploys that shipped nothing
    (kg-mcp, 2026-06-29 .. 2026-08-27)

    The old gate was:
        railway up --service X --ci --detach   # returns instantly, status never checked
        for i in 1..60: curl -fsS https://$DOMAIN/health && exit 0

    When a new container dies at startup Railway keeps routing to the LAST
    HEALTHY container. That old container answers /health with 200 on the FIRST
    poll, so the step exited 0 and the workflow went GREEN while nothing
    shipped. The rollback was gated on that health check failing, so the one
    safety net was wired to a signal that could not fire.

    The tell was in the log the whole time and read as good news:
        "healthy after 1*5=5s"
    A genuinely new container needs boot time. An INSTANT 200 means the OLD one
    answered. The worse the failure, the better the log looked.

THE TRAP THIS AVOIDS
    "Poll the deployment status instead" is only half a fix. `railway up
    --detach` RETURNS BEFORE THE NEW DEPLOYMENT REGISTERS, so the newest
    deployment can still be the PREVIOUS one carrying its own real SUCCESS —
    the identical stale read wearing a new costume. Identity comes before
    status, always.

WHY THIS IS A PYTHON MODULE AND NOT TWO SHELL HEREDOCS
    The first version was one bash script with an inline python block PER MODE.
    Review found the two blocks had already DIVERGED: `newest-id` did not handle
    a shape the `assert` path did, so it returned "" where the other found an
    id — and an empty prev turns the identity check OFF, producing a live false
    GREEN. The header at the time asserted the two extractions "MUST be
    identical". The invariant was violated in the file that stated it.

    Two heredocs cannot share code. One module can. `_parse` and `_newest` are
    now called by BOTH subcommands, so the property is enforced by construction
    rather than promised by a comment — and `test_both_modes_agree_on_every_shape`
    pins it.

EXIT CODES (assert) — note 1 vs 4, which the workflow depends on:
    0  advanced AND SUCCESS                 the deploy really shipped
    1  advanced AND terminally BAD          a real bad deploy -> SAFE TO ROLL BACK
    2  usage error
    3  NOT YET (unchanged id, or running)   caller should poll again
    4  UNREADABLE / UNKNOWN                 FAIL, but DO NOT roll back

    1 and 4 are deliberately distinct. Collapsing them is how a transient CLI
    blip becomes `railway down` against a healthy container — the gate causing
    the outage it exists to prevent. "Cannot tell" is not "is bad".

EXIT CODES (newest-id):
    0  id printed, or "NONE" for a genuinely empty deployment list
    1  UNREADABLE — never print an empty id and exit 0. The old version did
       (`|| echo ""; exit 0`), which silently degraded the caller to the
       permissive path exactly when something was wrong.
"""

from __future__ import annotations

import json
import sys

TERMINAL_OK = {"SUCCESS"}
TERMINAL_BAD = {"FAILED", "CRASHED", "REMOVED", "REMOVING", "SKIPPED"}
# SLEEPING: Railway app-sleep. Not a failure — the deployment shipped and then
# idled. Classified explicitly because an unknown status routes to exit 4, and
# before the 1/4 split that would have reached `railway down` on a healthy,
# merely-idle service.
IN_PROGRESS = {"BUILDING", "DEPLOYING", "INITIALIZING", "QUEUED", "WAITING",
               "NEEDS_APPROVAL", "SLEEPING"}

EXIT_OK, EXIT_BAD, EXIT_USAGE, EXIT_NOT_YET, EXIT_UNREADABLE = 0, 1, 2, 3, 4


class Unreadable(Exception):
    """Input we cannot trust. Always routes to EXIT_UNREADABLE, never to OK."""


def _parse(raw: str) -> list[dict]:
    """Raw CLI stdout -> list of deployment dicts. SHARED BY BOTH SUBCOMMANDS.

    MEASURED 2026-08-27 against the real kg-mcp service: a TOP-LEVEL BARE LIST,
    newest-first, elements keyed id / status / createdAt / meta. No wrapper, no
    GraphQL edges, no deploymentId/state alternates.

    Honest scope of that claim: the fixture is a hand-recorded capture, so it
    RECORDS the shape rather than detecting drift from it. The other arms below
    are insurance; because both subcommands share this function, drift can no
    longer affect one read and not the other, which was the actual danger.
    """
    if not raw.strip():
        raise Unreadable("empty-input")
    try:
        d = json.loads(raw)
    except Exception:
        raise Unreadable("unparseable-json")

    # An UNRECOGNISED shape is NOT an empty deployment list. Collapsing the two
    # is the fail-open hole in a new place: shape drift would yield "no
    # deployments" -> newest-id returns NONE -> the caller reads that as "no
    # previous deployment" -> the identity check is disabled -> stale SUCCESS
    # passes. Caught by this module's own test on `{"unexpected": 1}`.
    # A container we RECOGNISE and which is genuinely empty is fine (first-ever
    # deploy); a payload where we cannot find a container at all is not.
    if isinstance(d, list):
        items = d
    elif isinstance(d, dict):
        dep = d.get("deployments", d.get("deployment"))
        if isinstance(dep, list):
            items = dep
        elif isinstance(dep, dict) and isinstance(dep.get("edges"), list):
            items = [e.get("node") for e in dep["edges"] if isinstance(e, dict)]
        elif isinstance(dep, dict):
            items = [dep]
        else:
            raise Unreadable("unrecognised-shape")
    else:
        raise Unreadable("unrecognised-shape")
    return [i for i in items if isinstance(i, dict)]


def _created(i: dict) -> str:
    """Timestamp as a string. Coerced: a numeric or null createdAt mixed with
    ISO strings used to raise TypeError inside sort() — which fails closed in
    one mode and open in the other."""
    v = i.get("createdAt") or i.get("created_at") or ""
    return v if isinstance(v, str) else str(v)


def _newest(items: list[dict]) -> dict:
    """Pick the newest deployment. SHARED BY BOTH SUBCOMMANDS.

    Sort by createdAt ONLY when every row has one. A row missing it sorts as ""
    and reverse=True would shove it to the BOTTOM — so a just-created row whose
    timestamp is not yet populated would lose to a stale SUCCESS. When any row
    lacks a timestamp, trust the CLI's documented newest-first order instead.
    Ties break on id so the result is deterministic in either input order.
    """
    if all(_created(i) for i in items):
        items = sorted(items, key=lambda i: (_created(i), str(i.get("id") or "")),
                       reverse=True)
    return items[0]


def _dep_id(row: dict) -> str:
    return str(row.get("id") or row.get("deploymentId") or "").strip()


def newest_id(raw: str) -> str:
    items = _parse(raw)
    if not items:
        return "NONE"  # genuinely empty list: a first-ever deploy is legitimate
    dep_id = _dep_id(_newest(items))
    if not dep_id:
        raise Unreadable("newest-deployment-has-no-id")
    return dep_id


def assert_advanced(raw: str, prev: str) -> tuple[int, str]:
    # Normalise prev the SAME way dep_id is normalised. Review 2026-08-27:
    # dep_id was stripped and prev was not, so a trailing space or CR in prev
    # (trivial to introduce through $GITHUB_OUTPUT) made an UNCHANGED
    # deployment compare as advanced -> exit 0 off a stale SUCCESS.
    prev = (prev or "").strip()
    if prev in ("-", "NONE"):
        prev = ""

    items = _parse(raw)
    if not items:
        raise Unreadable("no-deployments-in-payload")

    row = _newest(items)
    dep_id = _dep_id(row)
    status = str(row.get("status") or row.get("state") or "").strip().upper()

    if not dep_id:
        raise Unreadable("newest-deployment-has-no-id")

    # IDENTITY BEFORE STATUS.
    if prev and dep_id == prev:
        return EXIT_NOT_YET, f"{dep_id} NOT-YET-ADVANCED({status or 'unknown'})"
    if not status:
        raise Unreadable("missing-status")
    if status in TERMINAL_OK:
        return EXIT_OK, f"{dep_id} {status}"
    if status in TERMINAL_BAD:
        return EXIT_BAD, f"{dep_id} {status}"
    if status in IN_PROGRESS:
        return EXIT_NOT_YET, f"{dep_id} {status}"
    raise Unreadable(f"UNKNOWN-STATUS({status})")


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "newest-id":
        if len(argv) != 2:
            print("usage: railway_deployment.py newest-id", file=sys.stderr)
            return EXIT_USAGE
        try:
            print(newest_id(sys.stdin.read()))
            return EXIT_OK
        except Unreadable as e:
            print(f"- {e}")
            return EXIT_UNREADABLE

    if len(argv) == 3 and argv[1] == "assert":
        try:
            code, msg = assert_advanced(sys.stdin.read(), argv[2])
        except Unreadable as e:
            print(f"- {e}")
            return EXIT_UNREADABLE
        print(msg)
        return code

    print("usage: railway_deployment.py newest-id | assert <prev_id>",
          file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main(sys.argv))
