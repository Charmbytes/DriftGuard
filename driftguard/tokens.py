"""
Capability-token broker.

The agent holds a short-lived JWT that encodes *what it is allowed to do*:

    {"sub": "refund-agent-01", "scope": "refund:approve:max_5000", "jti": "..."}

Validity is a two-part check, and both parts must pass:

  1. Cryptographic  -- signature verifies and the token has not expired.
  2. Revocation     -- the token's `jti` is still marked 'active' in the store.

Part 2 is what makes revocation *immediate*. A JWT cannot be un-signed, so the
store is the authority on whether a structurally valid token is still honoured.
This is the standard denylist pattern; for the demo the store is a SQLite
table, which keeps the whole trust decision inspectable in one place.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt  # PyJWT

from . import config
from .audit import log_event
from .db import get_conn


def _now() -> datetime:
    return datetime.now(timezone.utc)


def issue_token(
    agent_id: Optional[str] = None,
    scope: Optional[str] = None,
    reason: str = "startup",
) -> Dict[str, Any]:
    """
    Mint a new capability token and record it as active.

    Any previously active token for the same agent is superseded, so the agent
    always has exactly one live capability at a time.
    """
    agent_id = agent_id or config.AGENT_ID
    scope = scope or config.CAPABILITY_SCOPE

    issued_at = _now()
    expires_at = issued_at + timedelta(hours=config.TOKEN_TTL_HOURS)
    jti = str(uuid.uuid4())

    claims = {
        "sub": agent_id,
        "scope": scope,
        "jti": jti,
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
        "iss": "driftguard",
        "max_amount": config.REFUND_AUTO_APPROVE_CAP,
    }
    token = jwt.encode(claims, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)

    with get_conn(write=True) as conn:
        # Supersede older active tokens for this agent.
        conn.execute(
            "UPDATE capability_tokens SET status='revoked', revoked_at=?, "
            "revoked_reason='superseded by new token' "
            "WHERE agent_id=? AND status='active'",
            (issued_at.isoformat(), agent_id),
        )
        conn.execute(
            "INSERT INTO capability_tokens"
            "(jti, agent_id, scope, token, issued_at, expires_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'active')",
            (jti, agent_id, scope, token, issued_at.isoformat(), expires_at.isoformat()),
        )

    log_event(
        "token_issued",
        {"jti": jti, "agent_id": agent_id, "scope": scope,
         "expires_at": expires_at.isoformat(), "reason": reason},
    )
    return {
        "jti": jti,
        "token": token,
        "agent_id": agent_id,
        "scope": scope,
        "issued_at": issued_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "status": "active",
    }


def get_active_token(agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the current active token row for an agent, if any."""
    agent_id = agent_id or config.AGENT_ID
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM capability_tokens WHERE agent_id=? AND status='active' "
            "ORDER BY issued_at DESC LIMIT 1",
            (agent_id,),
        ).fetchone()
    return dict(row) if row else None


def get_latest_token(agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Most recently issued token *regardless of status*.

    Needed so the dashboard can distinguish "never had a capability" from
    "had one and DriftGuard pulled it" -- and show the revocation reason.
    """
    agent_id = agent_id or config.AGENT_ID
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM capability_tokens WHERE agent_id=? "
            "ORDER BY issued_at DESC, rowid DESC LIMIT 1",
            (agent_id,),
        ).fetchone()
    return dict(row) if row else None


def validate_token(token: str) -> Dict[str, Any]:
    """
    Full validity check used by the agent before it auto-approves anything.

    Returns {"valid": bool, "reason": str, "claims": dict|None}.
    """
    try:
        claims = jwt.decode(
            token,
            config.JWT_SECRET,
            algorithms=[config.JWT_ALGORITHM],
            issuer="driftguard",
        )
    except jwt.ExpiredSignatureError:
        return {"valid": False, "reason": "token expired", "claims": None}
    except jwt.InvalidTokenError as exc:
        return {"valid": False, "reason": f"invalid signature/claims: {exc}", "claims": None}

    # Cryptographically fine -- now ask the revocation store.
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status, revoked_reason FROM capability_tokens WHERE jti=?",
            (claims.get("jti"),),
        ).fetchone()

    if row is None:
        return {"valid": False, "reason": "token not found in capability store", "claims": claims}
    if row["status"] != "active":
        return {
            "valid": False,
            "reason": f"token revoked: {row['revoked_reason'] or 'no reason recorded'}",
            "claims": claims,
        }
    return {"valid": True, "reason": "active", "claims": claims}


def current_capability_status(agent_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Whether the agent may currently auto-approve, and why not if it may not.

    Looks at the latest token whatever its status, so a revoked capability
    reports as 'revoked' (with the reason and timestamp) rather than
    disappearing into 'no_token'.
    """
    agent_id = agent_id or config.AGENT_ID
    latest = get_latest_token(agent_id)
    if not latest:
        return {
            "agent_id": agent_id,
            "has_token": False,
            "can_auto_approve": False,
            "status": "no_token",
            "reason": "No capability token has ever been issued.",
        }

    check = validate_token(latest["token"])
    if check["valid"]:
        status = "active"
    elif latest["status"] == "revoked":
        status = "revoked"
    else:
        status = "expired"

    return {
        "agent_id": agent_id,
        "has_token": True,
        "can_auto_approve": check["valid"],
        "status": status,
        "reason": check["reason"],
        "jti": latest["jti"],
        "scope": latest["scope"],
        "issued_at": latest["issued_at"],
        "expires_at": latest["expires_at"],
        "revoked_at": latest["revoked_at"],
        "revoked_reason": latest["revoked_reason"],
    }


def revoke_token(
    jti: Optional[str] = None,
    agent_id: Optional[str] = None,
    reason: str = "manual revocation",
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Revoke a capability. This is the enforcement action that distinguishes
    DriftGuard from a monitoring-and-alerting tool.
    """
    agent_id = agent_id or config.AGENT_ID
    revoked_at = _now().isoformat()

    with get_conn(write=True) as conn:
        if jti:
            cur = conn.execute(
                "UPDATE capability_tokens SET status='revoked', revoked_at=?, revoked_reason=? "
                "WHERE jti=? AND status='active'",
                (revoked_at, reason, jti),
            )
        else:
            row = conn.execute(
                "SELECT jti FROM capability_tokens WHERE agent_id=? AND status='active' "
                "ORDER BY issued_at DESC LIMIT 1",
                (agent_id,),
            ).fetchone()
            if row is None:
                return {"revoked": False, "reason": "no active token to revoke"}
            jti = row["jti"]
            cur = conn.execute(
                "UPDATE capability_tokens SET status='revoked', revoked_at=?, revoked_reason=? "
                "WHERE jti=?",
                (revoked_at, reason, jti),
            )
        changed = cur.rowcount

    if changed == 0:
        return {"revoked": False, "jti": jti, "reason": "token was not active"}

    log_event(
        "token_revoked",
        {"jti": jti, "agent_id": agent_id, "reason": reason,
         "revoked_at": revoked_at, "evidence": evidence or {}},
    )
    return {"revoked": True, "jti": jti, "agent_id": agent_id,
            "reason": reason, "revoked_at": revoked_at}


def list_tokens(agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Full token history -- useful for the panel to see issue/revoke cycles."""
    with get_conn() as conn:
        if agent_id:
            rows = conn.execute(
                "SELECT jti, agent_id, scope, issued_at, expires_at, status, "
                "revoked_at, revoked_reason FROM capability_tokens "
                "WHERE agent_id=? ORDER BY issued_at DESC",
                (agent_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT jti, agent_id, scope, issued_at, expires_at, status, "
                "revoked_at, revoked_reason FROM capability_tokens ORDER BY issued_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def ensure_token(agent_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Issue a token at startup if the agent does not already hold a usable one.

    "Usable" means it still passes validation, not merely that a row is marked
    active. If the signing key was rotated, or the token expired while the
    service was down, the stored row is stale: its signature no longer
    verifies, so the agent would hold a capability it can never exercise and
    there would be no path back to a working state. In that case the stale row
    is retired and a fresh token issued.

    A row that is marked revoked is left alone -- revocation is a decision, not
    a fault, and must not be undone by a restart.
    """
    agent_id = agent_id or config.AGENT_ID
    latest = get_latest_token(agent_id)

    # First boot for this agent: grant the initial capability.
    if latest is None:
        return issue_token(agent_id=agent_id, reason="startup")

    # Revoked stays revoked. Restarting the service must never re-arm an agent
    # DriftGuard disabled -- otherwise enforcement is one `docker restart` deep,
    # and the entire guarantee is worthless. Re-granting is a deliberate act:
    # POST /api/token/reinstate, after the agent has been fixed and re-certified.
    if latest["status"] == "revoked":
        return dict(latest)

    check = validate_token(latest["token"])
    if check["valid"]:
        return dict(latest)

    # Marked active but no longer verifiable: the signing key was rotated, or
    # it expired while the service was down. Retire it and issue a fresh one,
    # so a key rotation does not strand the agent with an unusable capability.
    with get_conn(write=True) as conn:
        conn.execute(
            "UPDATE capability_tokens SET status='revoked', revoked_at=?, "
            "revoked_reason=? WHERE jti=? AND status='active'",
            (_now().isoformat(),
             f"retired at startup: {check['reason']}", latest["jti"]),
        )
    log_event("token_retired", {
        "jti": latest["jti"], "agent_id": agent_id, "reason": check["reason"],
    })
    return issue_token(agent_id=agent_id, reason="reissued after stale token")
