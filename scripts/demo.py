"""
Scripted end-to-end demo -- the same sequence the dashboard performs, from a
terminal. Useful as a dry run before the viva, and as a fallback if the
projector setup makes the dashboard awkward.

    python scripts/demo.py                     # against localhost:8000
    python scripts/demo.py --api http://...    # against another host

Sequence:
    1. Reset to a clean, certified, token-holding state.
    2. Ask the agent for an in-cap refund      -> APPROVED.
    3. Baseline probe run                      -> all green.
    4. Inject the drifted build.
    5. Drifted probe run                       -> safety + leakage breach,
                                                  capability token REVOKED.
    6. Ask for the exact same refund           -> ESCALATED, not approved.
    7. A human reviewer approves the queued refund.
    8. Verify the audit hash chain.
"""

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional

import requests

TIMEOUT = 300
CUSTOMER_MESSAGE = "Please refund order #A1042, I paid Rs 3000."


def hr(title: str = "") -> None:
    print("\n" + "=" * 72)
    if title:
        print(f"  {title}")
        print("=" * 72)


def post(api: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.post(f"{api}{path}", json=payload or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get(api: str, path: str) -> Dict[str, Any]:
    r = requests.get(f"{api}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def show_run(run: Dict[str, Any]) -> None:
    print(f"  run #{run['run_id']}  build={run['agent_version']}  "
          f"probes={run['n_probes']}  ({run['duration_seconds']}s)")
    print("  " + "-" * 60)
    for category in ("accuracy", "safety", "leakage", "tone"):
        score = run["category_scores"][category]
        threshold = run["thresholds"][category]
        state = run["statuses"][category].upper()
        flag = "  <<< BREACH" if category in run["breached"] else ""
        print(f"    {category:<9} {score:>6.3f}  (threshold {threshold:.2f})  {state:<7}{flag}")
    print("  " + "-" * 60)
    print(f"    overall   {run['overall_score']:>6.3f}  "
          f"(threshold {run['overall_threshold']:.2f})")


def show_reply(reply: Dict[str, Any]) -> None:
    print(f"    decision   : {reply['decision'].upper()}")
    print(f"    amount     : INR {reply['amount']:,.0f}")
    print(f"    capability : {'VALID' if reply['capability_valid'] else 'INVALID'}")
    if reply["downgraded"]:
        print(f"    ENFORCED   : {reply['downgrade_reason']}")
    print(f"    reply      : {reply['reply'][:150]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="DriftGuard end-to-end demo")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="Seconds to pause between steps (use ~2 when presenting).")
    args = parser.parse_args()
    api = args.api.rstrip("/")

    def beat() -> None:
        if args.pause:
            time.sleep(args.pause)

    try:
        get(api, "/api/health")
    except requests.RequestException as exc:
        print(f"Cannot reach the DriftGuard API at {api}: {exc}")
        print("Start it with:  docker compose up   (or see README, Option B)")
        return 1

    hr("STEP 1  Reset to a clean, certified baseline")
    reset = post(api, "/api/demo/reset")
    print(f"  baseline certified : {reset['baseline']['n_probes']} probes")
    print(f"  capability token   : issued ({reset['token']['jti'][:8]}...)")
    print(f"  live build         : {reset['active_version']}")
    beat()

    hr("STEP 2  Customer asks for an in-cap refund (certified agent)")
    print(f'  customer: "{CUSTOMER_MESSAGE}"')
    show_reply(post(api, "/api/agent/message", {"message": CUSTOMER_MESSAGE}))
    beat()

    hr("STEP 3  Scheduled re-certification against the certified build")
    show_run(post(api, "/api/probes/run", {"trigger": "demo-script"}))
    print("\n  Verdict: within tolerance. Capability retained.")
    beat()

    hr("STEP 4  A prompt-template change ships to production")
    swap = post(api, "/api/agent/version", {"version": "drifted"})
    print(f"  live build: {swap['previous_version']} -> {swap['active_version']}")
    print(f"  {swap['description']}")
    print("\n  Note: the agent still holds its capability token at this point.")
    beat()

    hr("STEP 5  Next re-certification cycle runs")
    run = post(api, "/api/probes/run", {"trigger": "demo-script"})
    show_run(run)
    print(f"\n  {run['explanation']['summary']}")
    if run["revocation"].get("revoked"):
        print(f"\n  *** CAPABILITY TOKEN REVOKED ***")
        print(f"  jti: {run['revocation']['jti']}")
        print("\n  Probes that triggered it:")
        for o in run["explanation"]["top_offenders"][:4]:
            print(f"    - {o['probe_id']:<9} [{o['category']:<8}] "
                  f"{', '.join(o['codes'])}")
            print(f"      {o['detail'][:100]}")
    beat()

    hr("STEP 6  The SAME customer request, after revocation")
    print(f'  customer: "{CUSTOMER_MESSAGE}"')
    show_reply(post(api, "/api/agent/message", {"message": CUSTOMER_MESSAGE}))
    print("\n  The drifted agent still wanted to approve. DriftGuard withheld the action.")
    beat()

    hr("STEP 7  The refund is not lost -- a human takes over")
    queue = get(api, "/api/approvals?status=pending")
    pending = queue["requests"]
    print(f"  refunds awaiting a human: {queue['summary']['counts'].get('pending', 0)}"
          f"  (INR {queue['summary']['pending_value']:,.0f} on hold)")
    if pending:
        r = pending[0]
        print(f"\n  #HR-{r['id']:04d}  INR {r['amount']:,.0f}")
        print(f'    customer: "{r["customer_message"]}"')
        print(f"    why a human: {r['reason'][:90]}...")

        decided = post(api, f"/api/approvals/{r['id']}/decide",
                       {"decision": "approved", "reviewer": "support-lead",
                        "note": "verified order, genuine damage claim"})
        req = decided["request"]
        print(f"\n  Human reviewer '{req['reviewer']}' APPROVED it at {req['decided_at'][:19]}")
        print("  The customer still gets their refund -- a person authorised it, not the agent.")
    beat()

    hr("STEP 8  Audit trail integrity")
    verify = get(api, "/api/audit/verify")
    print(f"  chain valid  : {verify['valid']}")
    print(f"  rows checked : {verify['rows_checked']}")
    if verify["valid"]:
        print(f"  head hash    : {verify['head_hash']}")
    else:
        print(f"  BROKEN AT    : row {verify.get('broken_at_id')} - {verify['reason']}")

    log = get(api, "/api/audit/log")
    print("\n  Recent events (newest first):")
    for e in log["entries"][:8]:
        print(f"    #{e['id']:<3} {e['event_type']}")

    hr("Demo complete")
    print("  Same message, same agent, same token -- different outcome, because")
    print("  DriftGuard re-tested the agent and pulled its capability.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
