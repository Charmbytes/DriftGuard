"""
Human-in-the-loop approval queue.

When DriftGuard revokes the agent's capability token, the agent does not stop
working -- it stops *acting alone*. Every refund it would have auto-approved
becomes a pending approval request that a human reviewer must sign off.

This is the part that makes revocation safe to automate. Without it, pulling
the token would simply break the refund desk, and no business would ever switch
the feature on. With it, the failure mode is "slower", not "broken":

    token active   -> agent approves            (seconds, no human)
    token revoked  -> human approves            (minutes, one human)

Every state change is written to the hash-chained audit log, so the record
shows both that the agent was stopped and who authorised each refund instead.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .audit import log_event
from .db import get_conn

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_request(
    customer_message: str,
    amount: float,
    agent_version: str,
    agent_decision: str,
    reason: str,
) -> Dict[str, Any]:
    """
    Queue a refund for human review.

    Called from the enforcement point when the agent wanted to approve but was
    not allowed to. `reason` records *why* it needed a human -- revoked token,
    or an amount beyond the token's scope.
    """
    created_at = _now()
    with get_conn(write=True) as conn:
        cur = conn.execute(
            "INSERT INTO approval_requests"
            "(created_at, customer_message, amount, agent_version, agent_decision, "
            " reason, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (created_at, customer_message, amount, agent_version, agent_decision,
             reason, PENDING),
        )
        request_id = cur.lastrowid

    log_event("approval_requested", {
        "request_id": request_id,
        "amount": amount,
        "agent_version": agent_version,
        "agent_decision": agent_decision,
        "reason": reason,
        "customer_message": customer_message[:300],
    })

    return {
        "id": request_id,
        "status": PENDING,
        "created_at": created_at,
        "amount": amount,
        "reason": reason,
        # Shown to the customer so they have something to quote when chasing.
        "customer_notice": (
            f"Your refund request has been sent to a human reviewer "
            f"(reference #HR-{request_id:04d}). You'll hear back shortly."
        ),
    }


def decide(
    request_id: int,
    decision: str,
    reviewer: str = "human-reviewer",
    note: str = "",
) -> Dict[str, Any]:
    """
    Record a human's approve/reject decision.

    Only pending requests can be decided, so a refund cannot be double-approved
    by two reviewers racing each other.
    """
    if decision not in (APPROVED, REJECTED):
        raise ValueError(f"decision must be '{APPROVED}' or '{REJECTED}', got '{decision}'")

    decided_at = _now()
    with get_conn(write=True) as conn:
        row = conn.execute(
            "SELECT * FROM approval_requests WHERE id=?", (request_id,)
        ).fetchone()
        if row is None:
            return {"ok": False, "reason": f"no approval request with id {request_id}"}
        if row["status"] != PENDING:
            return {
                "ok": False,
                "reason": f"request #{request_id} was already {row['status']}",
                "request": dict(row),
            }
        conn.execute(
            "UPDATE approval_requests SET status=?, reviewer=?, decided_at=?, note=? "
            "WHERE id=? AND status=?",
            (decision, reviewer, decided_at, note, request_id, PENDING),
        )
        updated = conn.execute(
            "SELECT * FROM approval_requests WHERE id=?", (request_id,)
        ).fetchone()

    log_event("approval_decided", {
        "request_id": request_id,
        "decision": decision,
        "reviewer": reviewer,
        "note": note,
        "amount": updated["amount"],
        "decided_at": decided_at,
    })

    return {"ok": True, "request": dict(updated)}


def list_requests(status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Newest first. Pass status='pending' for the reviewer's work queue."""
    with get_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM approval_requests WHERE status=? "
                "ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM approval_requests ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def summary() -> Dict[str, Any]:
    """Counts per status, for the dashboard header."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS c, COALESCE(SUM(amount), 0) AS total "
            "FROM approval_requests GROUP BY status"
        ).fetchall()

    counts = {PENDING: 0, APPROVED: 0, REJECTED: 0}
    pending_value = 0.0
    for r in rows:
        counts[r["status"]] = r["c"]
        if r["status"] == PENDING:
            pending_value = r["total"]

    return {
        "counts": counts,
        "pending_value": pending_value,
        "total": sum(counts.values()),
    }
