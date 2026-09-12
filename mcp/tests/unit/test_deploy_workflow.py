"""Unit tests for .github/workflows/deploy-mcp.yml CI logic.

Two pieces of behavior locked in here:

1. Env detection — branch → environment mapping. Until the kg-mcp Railway
   service exists in the prod environment, pushes to main MUST deploy to
   test (not prod); otherwise the Deploy step fails on every merge to main.
2. URL resolution — when ``railway status --json`` returns no domain (CLI
   shape drift, auth lag, fresh service), fall back to the known
   ``${SERVICE}-${ENV}.up.railway.app`` pattern. Otherwise a healthy live
   deploy gets reported as failed CI.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/deploy-mcp.yml"
MCP_CI_PATH = REPO_ROOT / ".github/workflows/mcp-ci.yml"
COMPLETENESS_TEST = (
    REPO_ROOT
    / "shrine-diet-bioactivity/lightrag/tests/test_kg_completeness_gates.py"
)
RESOLVE_SCRIPT = REPO_ROOT / "scripts/ci/resolve_railway_domain.sh"
RAILWAY_DEPLOY_PY = REPO_ROOT / "scripts/ci/railway_deployment.py"


def _detect_env_run() -> str:
    """Pull the bash from the detect-environment step's ``run:`` block."""
    data = yaml.safe_load(WORKFLOW_PATH.read_text())
    return data["jobs"]["detect-environment"]["steps"][0]["run"]


def _exec_with_github_output(script: str, env: dict[str, str]) -> str:
    """Run a bash snippet with ``$GITHUB_OUTPUT`` pointing at a temp file.

    Returns the file's contents (the ``key=value`` lines the step writes).
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        out_path = f.name
    try:
        full_env = {**os.environ, **env, "GITHUB_OUTPUT": out_path}
        proc = subprocess.run(
            ["bash", "-c", script],
            env=full_env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return f"::nonzero({proc.returncode})::{proc.stderr}"
        return Path(out_path).read_text()
    finally:
        os.unlink(out_path)


# ─── Env detection ────────────────────────────────────────────────────────


class TestDetectEnv:
    def test_main_branch_routes_to_test(self):
        """Until the prod Railway env has a kg-mcp service, main→test."""
        out = _exec_with_github_output(
            _detect_env_run(),
            {
                "GH_EVENT_NAME": "push",
                "GH_REF": "refs/heads/main",
                "GH_DISPATCH_ENV": "",
            },
        )
        assert "environment=test" in out, f"main should route to test, got: {out!r}"

    def test_test_branch_routes_to_test(self):
        out = _exec_with_github_output(
            _detect_env_run(),
            {
                "GH_EVENT_NAME": "push",
                "GH_REF": "refs/heads/test",
                "GH_DISPATCH_ENV": "",
            },
        )
        assert "environment=test" in out

    def test_dev_branch_routes_to_test(self):
        out = _exec_with_github_output(
            _detect_env_run(),
            {
                "GH_EVENT_NAME": "push",
                "GH_REF": "refs/heads/dev-feature-x",
                "GH_DISPATCH_ENV": "",
            },
        )
        assert "environment=test" in out

    def test_workflow_dispatch_honors_input_prod(self):
        """Manual deploy to prod stays available — only the push-from-main
        default changes."""
        out = _exec_with_github_output(
            _detect_env_run(),
            {
                "GH_EVENT_NAME": "workflow_dispatch",
                "GH_REF": "refs/heads/main",
                "GH_DISPATCH_ENV": "prod",
            },
        )
        assert "environment=prod" in out

    def test_workflow_dispatch_honors_input_test(self):
        out = _exec_with_github_output(
            _detect_env_run(),
            {
                "GH_EVENT_NAME": "workflow_dispatch",
                "GH_REF": "refs/heads/main",
                "GH_DISPATCH_ENV": "test",
            },
        )
        assert "environment=test" in out


# ─── URL resolution ───────────────────────────────────────────────────────


class TestResolveRailwayDomain:
    """``scripts/ci/resolve_railway_domain.sh`` — reads railway status JSON
    on stdin; outputs a domain on stdout. Falls back to
    ``${service}-${env}.up.railway.app`` when JSON has no domain."""

    def _run(
        self,
        json_in: str,
        service: str = "kg-mcp",
        env: str = "test",
    ) -> tuple[int, str]:
        assert RESOLVE_SCRIPT.exists(), f"missing helper: {RESOLVE_SCRIPT}"
        proc = subprocess.run(
            ["bash", str(RESOLVE_SCRIPT), service, env],
            input=json_in,
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout.strip()

    def test_extracts_domain_from_valid_json(self):
        js = '{"service":{"serviceDomains":[{"domain":"kg-mcp-test.up.railway.app"}]}}'
        rc, out = self._run(js)
        assert rc == 0
        assert out == "kg-mcp-test.up.railway.app"

    def test_falls_back_when_json_empty(self):
        rc, out = self._run("{}", service="kg-mcp", env="test")
        assert rc == 0
        assert out == "kg-mcp-test.up.railway.app"

    def test_falls_back_when_serviceDomains_missing(self):
        rc, out = self._run('{"service":{}}', service="kg-mcp", env="test")
        assert rc == 0
        assert out == "kg-mcp-test.up.railway.app"

    def test_falls_back_when_input_is_garbage(self):
        rc, out = self._run("not json at all", service="kg-mcp", env="prod")
        assert rc == 0
        assert out == "kg-mcp-prod.up.railway.app"

    def test_uses_env_in_fallback_pattern(self):
        """Fallback respects the env arg — kg-mcp-prod for prod, etc."""
        rc, out = self._run("{}", service="kg-mcp", env="prod")
        assert rc == 0
        assert out == "kg-mcp-prod.up.railway.app"


# ─── Stale promotion guard (issue #46) ────────────────────────────────────


class TestNoStalePromotionGuard:
    """The test→main promotion guard predates the live workflow: the `test`
    branch is hundreds of commits behind main and effectively dead, so the
    guard fails on every PR for no operational reason. Lock its removal in
    so it can't quietly come back."""

    def _workflow(self) -> dict:
        return yaml.safe_load(WORKFLOW_PATH.read_text())

    def test_pr_promotion_guard_job_is_absent(self):
        jobs = self._workflow()["jobs"]
        assert "pr-promotion-guard" not in jobs, (
            "The pr-promotion-guard job has been removed (see #46) — "
            "do not reintroduce it without also reactivating the test branch."
        )

    def test_no_other_job_depends_on_promotion_guard(self):
        """Sanity: even after removal, ensure nothing in `needs:` still
        names the deleted job (which would silently fail to schedule)."""
        jobs = self._workflow()["jobs"]
        for name, body in jobs.items():
            needs = body.get("needs") or []
            if isinstance(needs, str):
                needs = [needs]
            assert "pr-promotion-guard" not in needs, (
                f"job {name!r} still has pr-promotion-guard in its needs"
            )

    def test_no_step_step_references_promotion_guard(self):
        """Catch leftover step-name strings (e.g., `name: PR Promotion Guard`).
        Stronger than the job-key check because step-name strings are easy
        to copy-paste."""
        text = WORKFLOW_PATH.read_text()
        assert "PR Promotion Guard" not in text
        assert "pr-promotion-guard" not in text


# ─── Aura gate must not silently pass on missing secrets (issue #65) ──────


class TestAuraGateRequiresSecrets:
    """The aura-data-integrity job in mcp-ci.yml previously skipped on
    missing NEO4J_* secrets and still reported SUCCESS — a false-green
    that hides a misconfigured environment. Lock in a guard step that
    fails the job when the event is push/dispatch (i.e., a trusted run
    that should have secrets) and any required secret is empty."""

    def _aura_job_run_text(self) -> str:
        data = yaml.safe_load(MCP_CI_PATH.read_text())
        job = data["jobs"]["aura-data-integrity"]
        # Join all step `run:` blocks so the guard text can live in any of them.
        return "\n".join(
            step.get("run", "")
            for step in job["steps"]
            if isinstance(step, dict)
        )

    def test_aura_job_has_secret_presence_guard(self):
        """The job must error (exit 1) when invoked on a real push/dispatch
        without the secrets being set — not just warn."""
        text = self._aura_job_run_text()
        # The fix must explicitly fail when the event isn't a PR and a
        # NEO4J_* secret is empty. We match on the documented sentinel
        # phrase so the check is robust to bash style.
        assert "exit 1" in text, (
            "aura-data-integrity job has no `exit 1` — likely still "
            "warning-only on missing secrets (see #65)."
        )
        assert "GH_EVENT_NAME" in text or "github.event_name" in text or "EVENT_NAME" in text, (
            "Guard must condition on event type so PRs from forks "
            "still skip cleanly (they have no secrets by design)."
        )


# ─── Completeness gates test must declare a marker (issue #49) ────────────


class TestCompletenessGatesHasMarker:
    """``test_kg_completeness_gates.py`` requires a 5.5 GB local SQLite DB
    that CI doesn't ship, so it must be marked ``integration`` (or
    deselected by default) — otherwise the default pytest run picks it
    up, hits a skip cascade, and inflates the noise floor of test reports.
    """

    def test_file_declares_pytestmark(self):
        text = COMPLETENESS_TEST.read_text()
        # Either ``pytestmark = pytest.mark.X`` or
        # ``pytestmark = [pytest.mark.X, ...]`` — both syntaxes are valid.
        assert "pytestmark" in text, (
            f"{COMPLETENESS_TEST.name} has no pytestmark — add the "
            "`integration` marker so default runs deselect it (see #49)."
        )
        # Must mark with one of the catalogued markers from
        # shrine-diet-bioactivity/pytest.ini. ``integration`` is the
        # appropriate one because the gates need the local KG DB.
        assert "integration" in text, (
            "completeness gates need the local KG DB → mark `integration`."
        )


# ─── deployment-advanced gate ─────────────────────────────────────────────

_spec = importlib.util.spec_from_file_location("railway_deployment", RAILWAY_DEPLOY_PY)
rdep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rdep)

# MEASURED fixture: real `railway deployment list --json` (kg-mcp, 2026-08-27,
# keys only / values redacted). Top-level BARE LIST, newest-first, keyed
# id/status/createdAt/meta. RECORDED, not drift-detecting — no test can tell if
# Railway changes the shape tomorrow; the point is that both subcommands share
# one parser, so drift cannot affect one read and not the other.
REAL_SHAPE = [
    {"id": "dep-newest", "status": "SUCCESS", "createdAt": "2026-08-27T09:00:00.000Z",
     "meta": {"buildLogs": "...", "image": {"digest": "sha256:..."},
              # adversarial decoy: a parser that walks nested dicts would find
              # this and pick a deployment that does not exist.
              "id": "dep-DECOY", "status": "SUCCESS", "createdAt": "2099-01-01T00:00:00Z"}},
    {"id": "dep-older", "status": "FAILED", "createdAt": "2026-08-26T09:00:00.000Z", "meta": {}},
    {"id": "dep-oldest", "status": "REMOVED", "createdAt": "2026-06-04T09:00:00.000Z", "meta": {}},
]

ALL_SHAPES = {
    "bare-list": REAL_SHAPE,
    "deployments-list": {"deployments": REAL_SHAPE},
    "deployments-edges": {"deployments": {"edges": [{"node": r} for r in REAL_SHAPE]}},
    "deployments-single-dict": {"deployments": REAL_SHAPE[0]},
    "deployment-singular": {"deployment": REAL_SHAPE[0]},
}


class TestRailwayDeploymentGate:
    """scripts/ci/railway_deployment.py — the gate that replaced a /health poll
    which could not fail.

    assert exits: 0 advanced+SUCCESS / 1 advanced+terminal-BAD (rollback-safe) /
    2 usage / 3 not-yet / 4 UNREADABLE (fail, do NOT roll back).
    newest-id exits: 0 with id or NONE / 4 unreadable.
    """

    def _assert(self, payload, prev=""):
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        try:
            return rdep.assert_advanced(raw, prev)
        except rdep.Unreadable as e:
            return rdep.EXIT_UNREADABLE, f"- {e}"

    def _newest(self, payload):
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        try:
            return rdep.EXIT_OK, rdep.newest_id(raw)
        except rdep.Unreadable as e:
            return rdep.EXIT_UNREADABLE, f"- {e}"

    # ── THE PROPERTY THAT THE OLD TWO-HEREDOC VERSION VIOLATED ──
    @pytest.mark.parametrize("name", sorted(ALL_SHAPES))
    def test_both_modes_agree_on_every_shape(self, name):
        """The original bug: two hand-copied parsers diverged, so --newest-id
        returned '' where the verdict path found an id. Empty prev disables the
        identity check -> live false GREEN. This property makes the divergence
        impossible to reintroduce silently."""
        payload = ALL_SHAPES[name]
        rc_n, ident = self._newest(payload)
        assert rc_n == rdep.EXIT_OK, f"{name}: newest-id failed: {ident}"
        _, msg = self._assert(payload, prev="some-other-id")
        assert msg.split()[0] == ident, (
            f"{name}: newest-id said {ident!r} but assert classified {msg!r}")

    @pytest.mark.parametrize("name", sorted(ALL_SHAPES))
    def test_roundtrip_never_reports_success_on_an_unchanged_deployment(self, name):
        """Feed the id we just captured back in: must be NOT-YET, never OK.
        This single property kills every mutation that empties newest-id."""
        payload = ALL_SHAPES[name]
        _, prev = self._newest(payload)
        rc, msg = self._assert(payload, prev=prev)
        assert rc == rdep.EXIT_NOT_YET, f"{name}: expected NOT-YET, got {rc}: {msg}"
        assert rc != rdep.EXIT_OK

    # ── newest-id: previously ZERO coverage and fail-OPEN ──
    @pytest.mark.parametrize("payload", ["", "   ", "garbage", "ERROR: unauthorized",
                                         '{"unexpected":1}'])
    def test_newest_id_fails_closed_never_silent_empty(self, payload):
        rc, out = self._newest(payload)
        assert rc == rdep.EXIT_UNREADABLE, f"must fail closed, got {rc}: {out}"

    def test_newest_id_empty_list_is_NONE_not_an_error(self):
        rc, out = self._newest([])
        assert (rc, out) == (rdep.EXIT_OK, "NONE")

    def test_NONE_sentinel_means_no_previous(self):
        rc, _ = self._assert(REAL_SHAPE, prev="NONE")
        assert rc == rdep.EXIT_OK

    # ── the stale-read trap ──
    def test_unchanged_id_with_old_SUCCESS_does_NOT_pass(self):
        rc, msg = self._assert([REAL_SHAPE[0]], prev="dep-newest")
        assert rc == rdep.EXIT_NOT_YET, msg

    def test_prev_is_stripped_like_dep_id(self):
        """dep_id was stripped and prev was not, so a trailing space or CR made
        an UNCHANGED deployment compare as advanced -> exit 0 off a stale row."""
        for prev in ("dep-newest ", " dep-newest", "dep-newest\r", "dep-newest\n"):
            rc, msg = self._assert([REAL_SHAPE[0]], prev=prev)
            assert rc == rdep.EXIT_NOT_YET, f"prev={prev!r} -> {rc}: {msg}"

    # ── UNREADABLE (4) must be distinct from BAD (1): only 1 may roll back ──
    @pytest.mark.parametrize("payload,prev", [
        ("", "dep-old"), ("garbage", "dep-old"), ([], "dep-old"),
        ([{"status": "SUCCESS"}], "dep-old"),
        ([{"id": "dep-new"}], "dep-old"),
        ([{"id": "dep-new", "status": "TELEPORTED"}], "dep-old"),
    ])
    def test_unreadable_is_4_not_1(self, payload, prev):
        # LITERAL 4, not rdep.EXIT_UNREADABLE. Asserting against the symbol is
        # vacuous: collapsing EXIT_UNREADABLE to 1 mutates the constant AND the
        # expectation together, so the test compares the mutant to itself and
        # passes. Found by mutation — the suite stayed green while "cannot tell"
        # became "is bad", which is what arms `railway down` on a blip.
        rc, msg = self._assert(payload, prev)
        assert rc == 4, f"unreadable must be literal 4 (no rollback), got {rc}: {msg}"

    def test_exit_code_literals_are_pinned(self):
        """The workflow dispatches on `case "$RC" in 0|1|3|4)`. These numbers are
        a cross-file contract, so pin the values, not just the names."""
        assert (rdep.EXIT_OK, rdep.EXIT_BAD, rdep.EXIT_USAGE,
                rdep.EXIT_NOT_YET, rdep.EXIT_UNREADABLE) == (0, 1, 2, 3, 4)

    def test_terminal_bad_is_literal_1(self):
        rc, _ = self._assert([{"id": "dep-new", "status": "CRASHED",
                               "createdAt": "2026-08-27T09:00:00Z"}], "dep-old")
        assert rc == 1, "rollback-authorising code must be literal 1"

    def test_unreadable_always_emits_a_reason(self):
        for payload in ("", "garbage", [], [{"id": "x", "status": "??"}]):
            rc, msg = self._assert(payload, "dep-old")
            assert rc == 4
            assert msg.startswith("- ") and len(msg) > 2, f"no reason: {msg!r}"

    def test_terminal_bad_is_1_so_rollback_can_arm(self):
        for st in ("FAILED", "CRASHED", "REMOVED", "REMOVING", "SKIPPED"):
            rc, msg = self._assert([{"id": "dep-new", "status": st,
                                     "createdAt": "2026-08-27T09:00:00Z"}], "dep-old")
            assert rc == rdep.EXIT_BAD, f"{st} -> {rc}: {msg}"

    # ── status vocabulary closure: ONLY SUCCESS may exit 0 ──
    @pytest.mark.parametrize("status", [
        "BUILDING", "DEPLOYING", "INITIALIZING", "QUEUED", "WAITING",
        "NEEDS_APPROVAL", "SLEEPING"])
    def test_in_progress_is_3(self, status):
        rc, msg = self._assert([{"id": "dep-new", "status": status,
                                 "createdAt": "2026-08-27T09:00:00Z"}], "dep-old")
        assert rc == rdep.EXIT_NOT_YET, f"{status} -> {rc}: {msg}"

    def test_only_SUCCESS_can_exit_zero(self):
        """Closure test: moving any other status into TERMINAL_OK is caught."""
        vocab = (rdep.TERMINAL_OK | rdep.TERMINAL_BAD | rdep.IN_PROGRESS |
                 {"TELEPORTED", "PENDING", "ACTIVE", "OK", "DONE", "LIVE"})
        for st in vocab:
            rc, _ = self._assert([{"id": "dep-new", "status": st,
                                   "createdAt": "2026-08-27T09:00:00Z"}], "dep-old")
            assert (rc == rdep.EXIT_OK) == (st == "SUCCESS"), f"{st} exited {rc}"

    def test_status_is_normalised(self):
        for st in ("success", " SUCCESS ", "Success"):
            rc, _ = self._assert([{"id": "dep-new", "status": st,
                                   "createdAt": "2026-08-27T09:00:00Z"}], "dep-old")
            assert rc == rdep.EXIT_OK, st

    # ── ordering / timestamp robustness ──
    def test_picks_newest_by_createdAt_not_list_order(self):
        rc, msg = self._assert(list(reversed(REAL_SHAPE)), prev="dep-older")
        assert rc == rdep.EXIT_OK and "dep-newest" in msg, msg

    def test_row_without_createdAt_is_not_demoted_below_a_stale_row(self):
        """Sorting "" last would shove a just-created row below a stale SUCCESS.
        When any row lacks a timestamp we trust the CLI's newest-first order."""
        payload = [{"id": "dep-NEW", "status": "BUILDING"},
                   {"id": "dep-OLD", "status": "SUCCESS", "createdAt": "2026-06-04T09:00:00Z"}]
        rc, msg = self._assert(payload, prev="dep-OLD")
        assert rc == rdep.EXIT_NOT_YET and "dep-NEW" in msg, msg

    def test_non_string_createdAt_does_not_raise(self):
        payload = [{"id": "a", "status": "SUCCESS", "createdAt": 1756000000},
                   {"id": "b", "status": "SUCCESS", "createdAt": "2026-08-27T09:00:00Z"}]
        rc, msg = self._assert(payload, "dep-old")
        assert rc in (rdep.EXIT_OK, rdep.EXIT_UNREADABLE), msg
        assert msg and not msg.startswith("Traceback")

    def test_createdAt_ties_are_deterministic_in_either_order(self):
        a = {"id": "dep-A", "status": "SUCCESS", "createdAt": "2026-08-27T09:00:00Z"}
        b = {"id": "dep-B", "status": "CRASHED", "createdAt": "2026-08-27T09:00:00Z"}
        first = self._assert([a, b], "dep-old")
        second = self._assert([b, a], "dep-old")
        assert first == second, f"tie resolved differently: {first} vs {second}"

    def test_meta_decoy_is_not_followed(self):
        _, msg = self._assert(REAL_SHAPE, prev="dep-older")
        assert "dep-DECOY" not in msg, f"parser walked into meta: {msg}"

    def test_usage_error_is_2(self):
        proc = subprocess.run([sys.executable, str(RAILWAY_DEPLOY_PY)],
                              input="", capture_output=True, text=True)
        assert proc.returncode == rdep.EXIT_USAGE

    def test_cli_entrypoint_matches_library(self):
        """The workflow invokes the CLI; the tests above call the library. Pin
        that they agree, so the tested path IS the shipped path."""
        proc = subprocess.run(
            [sys.executable, str(RAILWAY_DEPLOY_PY), "assert", "dep-older"],
            input=json.dumps(REAL_SHAPE), capture_output=True, text=True)
        assert proc.returncode == rdep.EXIT_OK
        assert "dep-newest" in proc.stdout


# ─── the wiring: the gate's meaning lives in the YAML, not the helper ─────


class TestDeploymentAdvancedGateIsWired:
    """Deleting the capture step, renaming its id, or typoing the expression
    makes PREV_ID expand to "" — GitHub does not error on an unresolvable step
    output. The two-month bug then returns with every helper test still green.
    """

    @staticmethod
    def _steps():
        wf = yaml.safe_load(WORKFLOW_PATH.read_text())
        job = wf["jobs"]["deploy"]
        return job["steps"]

    def _idx(self, pred):
        for i, st in enumerate(self._steps()):
            if pred(st):
                return i
        return -1

    def test_capture_step_exists_and_uses_newest_id(self):
        i = self._idx(lambda s: s.get("id") == "prev_deploy")
        assert i >= 0, "capture step (id: prev_deploy) is missing"
        assert "railway_deployment.py newest-id" in self._steps()[i]["run"]

    def test_capture_step_fails_closed(self):
        run = self._steps()[self._idx(lambda s: s.get("id") == "prev_deploy")]["run"]
        assert "exit 1" in run, "capture must fail the job when it cannot read a baseline"
        assert "pipefail" in run, "without pipefail the railway failure is masked"

    def test_verdict_step_receives_the_captured_id(self):
        st = self._steps()[self._idx(lambda s: s.get("id") == "deploy_status")]
        assert st["env"]["PREV_ID"] == "${{ steps.prev_deploy.outputs.prev_id }}"
        assert 'railway_deployment.py assert "$PREV_ID"' in st["run"]

    def test_verdict_step_refuses_an_empty_prev_id(self):
        run = self._steps()[self._idx(lambda s: s.get("id") == "deploy_status")]["run"]
        assert '-z "$PREV_ID"' in run, "an unwired PREV_ID must be refused, not judged"

    def test_step_order_capture_deploy_verdict_then_health(self):
        cap = self._idx(lambda s: s.get("id") == "prev_deploy")
        dep = self._idx(lambda s: s.get("name") == "Deploy")
        ver = self._idx(lambda s: s.get("id") == "deploy_status")
        hc = self._idx(lambda s: s.get("id") == "health_check")
        assert cap < dep < ver < hc, f"order wrong: {cap} {dep} {ver} {hc}"

    def test_rollback_gates_on_crashed_only_never_on_unreadable_or_timeout(self):
        """`railway down` deletes the newest deployment. Firing it on an
        unreadable poll or a timeout deletes a HEALTHY container."""
        i = self._idx(lambda s: "railway down" in str(s.get("run", "")))
        assert i >= 0, "rollback step missing"
        cond = self._steps()[i]["if"]
        assert "verdict == 'crashed'" in cond, f"rollback must gate on crashed only: {cond}"
        assert "health_check.outcome" not in cond, "health must not arm the rollback"

    def test_health_probe_is_not_the_verdict(self):
        """It polls the service DOMAIN, which the previous container answers."""
        hc = self._steps()[self._idx(lambda s: s.get("id") == "health_check")]
        assert "secondary" in hc["name"].lower()

    def test_summary_reports_the_authoritative_verdict(self):
        run = str(self._steps()[self._idx(
            lambda s: s.get("name") == "Deployment summary")]["run"])
        assert "DEPLOY_VERDICT" in run, "summary must surface the deploy verdict"
        assert "secondary" in run.lower(), "health must be labelled secondary"

    def test_summary_does_not_interpolate_the_domain_into_bash(self):
        """${{ }} is substituted before bash parses; the domain comes from
        railway status JSON with no charset validation."""
        st = self._steps()[self._idx(lambda s: s.get("name") == "Deployment summary")]
        assert "${{ steps.health_check.outputs.url }}" not in str(st["run"]), \
            "domain must reach bash via env:, not ${{ }} interpolation"
        assert st["env"]["HEALTH_URL"] == "${{ steps.health_check.outputs.url }}"
