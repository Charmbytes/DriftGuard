"""
The agent under certification: a customer-support refund assistant.

It decides approve / escalate / deny for a refund request, and is authorised to
auto-approve up to INR 5,000 -- but only while it holds a valid capability
token.

Two call paths, and the difference matters for the demo:

  * handle_message(enforce_token=True)   -- the *runtime* path used by
    POST /api/agent/message. The capability token is checked before an approval
    is allowed to stand. Once DriftGuard revokes the token, an "approve" is
    downgraded to "escalate" here. The agent keeps chatting; it just loses the
    authority to act.

  * handle_message(enforce_token=False)  -- the *shadow-test* path used by the
    probe harness. We want to measure what the model actually decided, not what
    the enforcement layer allowed. If probes ran with enforcement on, a revoked
    token would mask the very drift we are trying to measure.
"""

import time
from typing import Any, Dict, Optional

from .. import config
from ..db import get_state, set_state
from . import llm
from .prompts import AGENT_VERSIONS, get_system_prompt

_VERSION_KEY = "agent_version"


def get_active_version() -> str:
    """Which agent build is currently live. Defaults to the certified baseline."""
    return get_state(_VERSION_KEY, "baseline")


def set_active_version(version: str) -> Dict[str, Any]:
    """
    Swap the live agent build. This is the "inject drift" lever the dashboard
    pulls -- it stands in for a prompt-template deploy in a real system.
    """
    if version not in AGENT_VERSIONS:
        raise ValueError(f"Unknown agent version '{version}'.")
    previous = get_active_version()
    set_state(_VERSION_KEY, version)
    return {"previous_version": previous, "active_version": version,
            **{k: v for k, v in AGENT_VERSIONS[version].items() if k != "system_prompt"}}


def handle_message(
    message: str,
    version: Optional[str] = None,
    enforce_token: bool = True,
) -> Dict[str, Any]:
    """
    Run one customer message through the agent.

    Returns the parsed decision plus enforcement metadata.
    """
    # Local imports avoid an import cycle (tokens/approvals -> audit -> db).
    from ..approvals import create_request
    from ..tokens import current_capability_status

    version = version or get_active_version()
    system_prompt = get_system_prompt(version)

    started = time.perf_counter()
    try:
        raw = llm.complete(system_prompt, message, version=version)
        error = None
    except Exception as exc:  # provider outage, bad key, timeout
        raw = ""
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000

    parsed = llm.parse_agent_json(raw)
    result: Dict[str, Any] = {
        "agent_version": version,
        "decision": parsed["decision"],
        "amount": parsed["amount"],
        "reply": parsed["reply"],
        "reason": parsed["reason"],
        "raw_response": raw,
        "parse_ok": parsed["_parse_ok"],
        "latency_ms": round(latency_ms, 1),
        "error": error,
        "capability_enforced": enforce_token,
        "capability_valid": None,
        "downgraded": False,
    }

    if not enforce_token:
        return result

    # ---- Enforcement point -------------------------------------------------
    status = current_capability_status()
    result["capability_valid"] = status["can_auto_approve"]
    result["capability_status"] = status["status"]

    if parsed["decision"] == "approve":
        if not status["can_auto_approve"]:
            # Token revoked or missing: the agent may not act on its own.
            result["decision"] = "escalate"
            result["downgraded"] = True
            result["downgrade_reason"] = (
                f"Capability token not valid ({status['reason']}). "
                "Auto-approval withheld; routed to a human reviewer."
            )
            result["reply"] = (
                "Thanks for your patience. I'm passing this to a human colleague "
                "for approval, and they'll follow up with you shortly."
            )
        elif parsed["amount"] > config.REFUND_AUTO_APPROVE_CAP:
            # Belt-and-braces: the token's scope caps the amount even if the
            # model tried to exceed it. Behavioural drift is still recorded by
            # the probe harness -- this only stops the action.
            result["decision"] = "escalate"
            result["downgraded"] = True
            result["downgrade_reason"] = (
                f"Requested amount INR {parsed['amount']:,.0f} exceeds token scope "
                f"'{status.get('scope')}'. Auto-approval withheld."
            )
            result["reply"] = (
                "I'm not able to approve a refund of this size on my own. "
                "I've routed it to a colleague who can authorise it."
            )

    # The work does not vanish -- it moves to a human. Queue the refund the
    # agent wanted to make so a reviewer can sign it off, and give the customer
    # a reference number. This is why revoking a capability is safe to
    # automate: the failure mode is "slower", not "broken".
    if result["downgraded"]:
        request = create_request(
            customer_message=message,
            amount=parsed["amount"],
            agent_version=version,
            agent_decision=parsed["decision"],
            reason=result["downgrade_reason"],
        )
        result["approval_request"] = request
        result["reply"] = f"{result['reply']} {request['customer_notice']}"

    return result


def agent_info() -> Dict[str, Any]:
    """Metadata for the dashboard: which build is live and what builds exist."""
    active = get_active_version()
    return {
        "active_version": active,
        "provider": config.LLM_PROVIDER,
        "model": {
            "mock": "deterministic-mock",
            "groq": config.GROQ_MODEL,
            "openai": config.OPENAI_MODEL,
        }.get(config.LLM_PROVIDER, "unknown"),
        "auto_approve_cap": config.REFUND_AUTO_APPROVE_CAP,
        "versions": {
            k: {kk: vv for kk, vv in v.items() if kk != "system_prompt"}
            for k, v in AGENT_VERSIONS.items()
        },
    }
