"""
DriftGuard FastAPI service.

Endpoint map:

  POST /api/agent/message        talk to the agent (capability token enforced)
  POST /api/agent/version        swap the live build -- the "inject drift" lever
  GET  /api/agent/info           which build is live, provider, cap
  GET  /api/agent/prompts        certified vs live system prompt, with the edits
  POST /api/agent/compare        same message to certified and live build, side by side

  POST /api/probes/run           trigger a shadow-test run
  GET  /api/probes/suite         the probe suite itself
  POST /api/probes/certify       (re)certify the baseline

  GET  /api/certification/status current scores + token status (dashboard poll)
  GET  /api/certification/runs   run history for the timeline chart
  GET  /api/certification/runs/{id}  per-probe detail for one run
  GET  /api/certification/runs/{id}/story  plain-English account of why it drifted

  GET  /api/token                current capability status
  GET  /api/token/history        every token ever issued
  POST /api/token/reinstate      issue a fresh token (after remediation)

  GET  /api/approvals            human reviewer queue
  POST /api/approvals/{id}/decide  human approves or rejects a refund

  GET  /api/audit/log            paginated audit trail
  GET  /api/audit/verify         hash-chain integrity check

  POST /api/demo/reset           wipe and re-certify -- for repeat demos
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import approvals, config, explain
from .agent import refund_agent
from .agent.prompts import AGENT_VERSIONS
from .audit import get_log, log_event, verify_chain
from .db import get_conn, init_db, reset_all
from .probes.run_probes import baseline_status, certify_baseline, run_probe_suite
from .probes.suite import CATEGORIES, get_suite, suite_summary
from .scheduler import start_scheduler, stop_scheduler
from .scoring import drift as drift_mod
from .scoring import embeddings
from .tokens import (current_capability_status, ensure_token, issue_token,
                     list_tokens, revoke_token)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("driftguard.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bring the system up in a certified, token-holding state."""
    init_db()

    # Say plainly how this instance is secured, rather than letting a demo
    # deployment quietly inherit weak defaults.
    if config.JWT_SECRET_IS_GENERATED:
        logger.warning(
            "JWT_SECRET was not set; using a generated key persisted at "
            "%s. Fine for a local demo. Set JWT_SECRET explicitly for any "
            "shared or long-lived deployment.", config.DATA_DIR / ".jwt_secret",
        )
    logger.info("API has no authentication; CORS origins limited to %s. "
                "Do not expose this port to an untrusted network.",
                config.CORS_ALLOW_ORIGINS)

    log_event("system_started", {
        "provider": config.LLM_PROVIDER,
        "cap": config.REFUND_AUTO_APPROVE_CAP,
        "scheduler_enabled": config.ENABLE_SCHEDULER,
    })
    ensure_token()
    if not baseline_status()["certified"]:
        logger.info("No certified baseline found -- certifying now...")
        certify_baseline()
        logger.info("Baseline certified.")
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(
    title="DriftGuard",
    version="0.1.0",
    description="Continuous behavioural re-certification and capability-token "
                "revocation for LLM agents.",
    lifespan=lifespan,
)

# Only browser clients need this: the Streamlit dashboard calls the API
# server-side. Origins are an explicit allowlist, never "*", because the API
# is unauthenticated -- see README, Security posture.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ==========================================================================
# Schemas
# ==========================================================================
class MessageRequest(BaseModel):
    message: str = Field(..., description="The customer's message to the agent.")
    version: Optional[str] = Field(None, description="Override the live build (testing only).")


class VersionRequest(BaseModel):
    version: str = Field(..., description="'baseline' or 'drifted'.")


class ProbeRunRequest(BaseModel):
    target_version: Optional[str] = Field(None, description="Defaults to the live build.")
    category: Optional[str] = Field(None, description="Restrict to one category.")
    auto_revoke: bool = Field(True, description="Set false to score without enforcing.")
    trigger: str = Field("manual", description="Label recorded in the audit log.")


class CertifyRequest(BaseModel):
    force: bool = False


class ApprovalDecision(BaseModel):
    decision: str = Field(..., description="'approved' or 'rejected'.")
    reviewer: str = Field("human-reviewer", description="Who signed off.")
    note: str = Field("", description="Optional reviewer note, kept in the audit log.")


# ==========================================================================
# Agent
# ==========================================================================
@app.post("/api/agent/message")
def agent_message(req: MessageRequest) -> Dict[str, Any]:
    """
    Send a message to the agent. The capability token is checked here: once
    revoked, an 'approve' is downgraded to 'escalate' before it reaches you.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    result = refund_agent.handle_message(req.message, version=req.version, enforce_token=True)
    log_event("agent_message", {
        "message": req.message[:500],
        "agent_version": result["agent_version"],
        "decision": result["decision"],
        "amount": result["amount"],
        "downgraded": result["downgraded"],
        "capability_valid": result["capability_valid"],
    })
    return result


@app.post("/api/agent/version")
def set_agent_version(req: VersionRequest) -> Dict[str, Any]:
    """
    Swap the live agent build. This is the demo's drift injection: it stands in
    for someone shipping a prompt-template change to production.
    """
    if req.version not in AGENT_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown version '{req.version}'. Expected one of {list(AGENT_VERSIONS)}.",
        )
    out = refund_agent.set_active_version(req.version)
    log_event("agent_version_changed", out)
    return out


@app.get("/api/agent/info")
def agent_info() -> Dict[str, Any]:
    return refund_agent.agent_info()


@app.get("/api/agent/prompts")
def agent_prompts(version: Optional[str] = Query(None)) -> Dict[str, Any]:
    """The certified system prompt next to the live one, and what was edited."""
    version = version or refund_agent.get_active_version()
    if version not in AGENT_VERSIONS:
        raise HTTPException(status_code=400, detail=f"Unknown version '{version}'.")
    return explain.prompt_comparison(version)


@app.post("/api/agent/compare")
def agent_compare(req: MessageRequest) -> Dict[str, Any]:
    """
    Ask the certified build and the live build the same question.

    Both calls use the shadow path (no token enforcement, no approval queue),
    so this has no side effects -- it only shows how the two builds differ.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    live_version = req.version or refund_agent.get_active_version()
    baseline = refund_agent.handle_message(req.message, version="baseline", enforce_token=False)
    live = refund_agent.handle_message(req.message, version=live_version, enforce_token=False)
    return {
        "message": req.message,
        "baseline": baseline,
        "live": live,
        "same_decision": baseline["decision"] == live["decision"],
        "differences": explain.compare_decisions(baseline, live),
    }


# ==========================================================================
# Probes
# ==========================================================================
@app.get("/api/probes/suite")
def probes_suite(category: Optional[str] = Query(None)) -> Dict[str, Any]:
    if category and category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown category '{category}'.")
    return {"summary": suite_summary(), "probes": get_suite(category)}


@app.post("/api/probes/run")
def probes_run(req: ProbeRunRequest) -> Dict[str, Any]:
    """Trigger one shadow-test run. Returns the full scored result."""
    if req.target_version and req.target_version not in AGENT_VERSIONS:
        raise HTTPException(status_code=400, detail=f"Unknown version '{req.target_version}'.")
    if req.category and req.category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Unknown category '{req.category}'.")
    try:
        return run_probe_suite(
            target_version=req.target_version,
            trigger=req.trigger,
            category=req.category,
            auto_revoke=req.auto_revoke,
        )
    except Exception as exc:
        logger.exception("Probe run failed")
        raise HTTPException(status_code=500, detail=f"Probe run failed: {exc}") from exc


@app.post("/api/probes/certify")
def probes_certify(req: CertifyRequest) -> Dict[str, Any]:
    """(Re)establish the certified baseline."""
    return certify_baseline(force=req.force)


# ==========================================================================
# Certification status
# ==========================================================================
def _latest_run() -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM probe_runs WHERE finished_at IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["category_scores"] = json.loads(d["category_scores"] or "{}")
    d["breached"] = json.loads(d["breached"] or "[]")
    d["cusum_scores"] = json.loads(d["cusum_scores"] or "{}")
    d["cusum_alarming"] = json.loads(d["cusum_alarming"] or "[]")
    return d


@app.get("/api/certification/status")
def certification_status() -> Dict[str, Any]:
    """Everything the dashboard needs for its health panel, in one call."""
    latest = _latest_run()
    token = current_capability_status()

    if latest:
        scores = latest["category_scores"]
        statuses = {c: drift_mod.category_status(c, scores.get(c, 0.0)) for c in CATEGORIES}
        overall_status = drift_mod.overall_status(latest["overall_score"] or 0.0)
    else:
        scores = {c: 0.0 for c in CATEGORIES}
        statuses = {c: "unknown" for c in CATEGORIES}
        overall_status = "unknown"

    return {
        "baseline": baseline_status(),
        "agent": refund_agent.agent_info(),
        "capability": token,
        "latest_run": {
            "run_id": latest["id"] if latest else None,
            "started_at": latest["started_at"] if latest else None,
            "finished_at": latest["finished_at"] if latest else None,
            "agent_version": latest["agent_version"] if latest else None,
            "trigger": latest["trigger"] if latest else None,
            "overall_score": latest["overall_score"] if latest else None,
            "category_scores": scores,
            "breached": latest["breached"] if latest else [],
            "revoked": bool(latest["revoked"]) if latest else False,
            "detector": latest.get("detector") if latest else None,
        },
        # Sequential detection: reads the run history rather than one run, so
        # it catches gradual drift that never trips a single-run threshold.
        "sequential": {
            "cusum_scores": latest["cusum_scores"] if latest else {},
            "cusum_alarming": latest["cusum_alarming"] if latest else [],
            "cusum_threshold": config.CUSUM_THRESHOLD,
            "cusum_slack": config.CUSUM_SLACK,
            "cusum_enforces": config.CUSUM_ENFORCES,
            "cusum_min_runs": config.CUSUM_MIN_RUNS,
            "psi_score": latest["psi_score"] if latest else None,
        },
        "statuses": statuses,
        "overall_status": overall_status,
        "thresholds": dict(config.CATEGORY_THRESHOLDS),
        "overall_threshold": config.OVERALL_THRESHOLD,
        "revocation_categories": config.REVOCATION_CATEGORIES,
        "approvals": approvals.summary(),
        "embedding_backend": embeddings.backend_name(),
        "scheduler": {
            "enabled": config.ENABLE_SCHEDULER,
            "interval_minutes": config.PROBE_INTERVAL_MINUTES,
        },
    }


@app.get("/api/certification/runs")
def certification_runs(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    """Run history, oldest first -- feeds the drift timeline chart."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM probe_runs WHERE finished_at IS NOT NULL "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    runs = []
    for r in reversed(rows):
        d = dict(r)
        d["category_scores"] = json.loads(d["category_scores"] or "{}")
        d["breached"] = json.loads(d["breached"] or "[]")
        d["cusum_scores"] = json.loads(d["cusum_scores"] or "{}")
        d["cusum_alarming"] = json.loads(d["cusum_alarming"] or "[]")
        d["revoked"] = bool(d["revoked"])
        runs.append(d)
    return {"runs": runs, "count": len(runs)}


@app.get("/api/certification/runs/{run_id}")
def certification_run_detail(run_id: int) -> Dict[str, Any]:
    """Per-probe evidence for one run: the drill-down behind a score."""
    with get_conn() as conn:
        run = conn.execute("SELECT * FROM probe_runs WHERE id=?", (run_id,)).fetchone()
        if run is None:
            raise HTTPException(status_code=404, detail=f"No run with id {run_id}")
        rows = conn.execute(
            "SELECT * FROM probe_results WHERE run_id=? ORDER BY id ASC", (run_id,)
        ).fetchall()

    run_d = dict(run)
    run_d["category_scores"] = json.loads(run_d["category_scores"] or "{}")
    run_d["breached"] = json.loads(run_d["breached"] or "[]")
    run_d["cusum_scores"] = json.loads(run_d["cusum_scores"] or "{}")
    run_d["cusum_alarming"] = json.loads(run_d["cusum_alarming"] or "[]")
    run_d["revoked"] = bool(run_d["revoked"])

    results = []
    for r in rows:
        d = dict(r)
        d["violations"] = json.loads(d["violations"] or "[]")
        results.append(d)

    return {"run": run_d, "results": results}


@app.get("/api/certification/runs/{run_id}/story")
def certification_run_story(run_id: int) -> Dict[str, Any]:
    """Plain-English explanation of one run: what went wrong and the likely cause."""
    detail = certification_run_detail(run_id)
    with get_conn() as conn:
        base = {r["probe_id"]: r["decision"] for r in
                conn.execute("SELECT probe_id, decision FROM baseline_responses").fetchall()}
    expected = {p["id"]: p["expected_decision"] for p in get_suite()}
    for r in detail["results"]:
        r["baseline_decision"] = base.get(r["probe_id"])
        r["expected_decision"] = expected.get(r["probe_id"])
    story = explain.drift_story(detail["results"], detail["run"]["agent_version"])
    story["run"] = detail["run"]
    return story


# ==========================================================================
# Capability token
# ==========================================================================
@app.get("/api/token")
def token_status() -> Dict[str, Any]:
    return current_capability_status()


@app.get("/api/token/history")
def token_history() -> Dict[str, Any]:
    return {"tokens": list_tokens()}


@app.post("/api/token/reinstate")
def token_reinstate() -> Dict[str, Any]:
    """
    Issue a fresh capability token.

    In a real deployment this is the post-incident step: the agent is rolled
    back or fixed, re-certified, and only then re-granted its capability.
    """
    return issue_token(reason="manual reinstatement after remediation")


@app.post("/api/token/revoke")
def token_revoke_manual(reason: str = Query("manual revocation via API")) -> Dict[str, Any]:
    return revoke_token(reason=reason)


# ==========================================================================
# Human-in-the-loop approvals
# ==========================================================================
@app.get("/api/approvals")
def approvals_list(
    status: Optional[str] = Query(None, description="pending | approved | rejected"),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    """The human reviewer's work queue -- refunds the agent may no longer make."""
    if status and status not in ("pending", "approved", "rejected"):
        raise HTTPException(status_code=400, detail=f"Unknown status '{status}'.")
    return {"summary": approvals.summary(),
            "requests": approvals.list_requests(status=status, limit=limit)}


@app.post("/api/approvals/{request_id}/decide")
def approvals_decide(request_id: int, req: ApprovalDecision) -> Dict[str, Any]:
    """
    A human signs off (or rejects) a refund the agent was not allowed to make.

    Recorded in the audit log with the reviewer's name, so the trail shows both
    that the agent was stopped and who authorised the refund instead.
    """
    try:
        result = approvals.decide(
            request_id, req.decision, reviewer=req.reviewer, note=req.note
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result["ok"]:
        raise HTTPException(status_code=409, detail=result["reason"])
    return result


# ==========================================================================
# Audit
# ==========================================================================
@app.get("/api/audit/log")
def audit_log(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    event_type: Optional[str] = Query(None),
) -> Dict[str, Any]:
    entries, total = get_log(limit=limit, offset=offset, event_type=event_type)
    return {"entries": entries, "total": total, "limit": limit, "offset": offset}


@app.get("/api/audit/verify")
def audit_verify() -> Dict[str, Any]:
    """Recompute the hash chain from genesis and report tampering."""
    return verify_chain()


# ==========================================================================
# Demo helpers
# ==========================================================================
@app.post("/api/demo/reset")
def demo_reset() -> Dict[str, Any]:
    """
    Return the system to a clean, certified, token-holding baseline state.

    Lets the demo be run repeatedly (e.g. once per panel member) without
    restarting containers.
    """
    reset_all()
    init_db()
    log_event("demo_reset", {})
    refund_agent.set_active_version("baseline")
    cert = certify_baseline(force=True)
    token = ensure_token()
    return {
        "reset": True,
        "baseline": cert,
        "token": {"jti": token.get("jti"), "status": "active"},
        "active_version": "baseline",
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "provider": config.LLM_PROVIDER, "version": app.version}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "DriftGuard",
        "version": app.version,
        "docs": "/docs",
        "description": "Continuous behavioural re-certification and "
                       "capability-token revocation for LLM agents.",
    }
