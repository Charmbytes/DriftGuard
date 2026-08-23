"""
DriftGuard dashboard (Streamlit).

Talks to the API over HTTP only -- it never touches the database directly, so
what the panel sees is exactly what the service exposes.

Layout:
  * Top strip     -- capability token state and live agent build.
  * Demo controls -- inject drift, run probes, reinstate, reset.
  * Health        -- green/yellow/red per category against its threshold.
  * Timeline      -- drift score per category across runs.
  * Evidence      -- the probes that caused the verdict.
  * Live agent    -- send a message and watch enforcement apply.
  * Audit         -- hash-chained trail plus an integrity check.
"""

import os
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180  # a full probe run can take a while on first model load

st.set_page_config(page_title="DriftGuard", layout="wide")

STATUS_COLOUR = {"green": "#1a9850", "yellow": "#e6a700", "red": "#d73027", "unknown": "#888888"}
STATUS_LABEL = {"green": "HEALTHY", "yellow": "WARNING", "red": "BREACH", "unknown": "NO DATA"}


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def api_get(path: str, **params) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"GET {path} failed: {exc}")
        return None


def api_post(path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"POST {path} failed: {exc}")
        return None


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("DriftGuard")
st.caption(
    "Continuous behavioural re-certification and capability-token revocation for LLM agents "
    "— agent under test: **customer-support refund agent**"
)

status = api_get("/api/certification/status")
if status is None:
    st.warning(f"Cannot reach the DriftGuard API at `{API_BASE}`. Is the api service running?")
    st.stop()

capability = status["capability"]
agent = status["agent"]
latest = status["latest_run"]

# ---- Revocation banner: the headline event of the demo --------------------
if capability["status"] == "revoked":
    st.error(
        f"### CAPABILITY TOKEN REVOKED\n"
        f"**{agent['active_version']}** build no longer holds `{capability['scope']}`. "
        f"Refund auto-approval is disabled; every request now escalates to a human.\n\n"
        f"**Revoked at:** {capability.get('revoked_at')}  \n"
        f"**Reason:** {capability.get('revoked_reason')}"
    )
elif capability["status"] == "active":
    st.success(
        f"### CAPABILITY ACTIVE — `{capability['scope']}`\n"
        f"Agent may auto-approve refunds up to ₹{agent['auto_approve_cap']:,.0f}. "
        f"Token `{capability.get('jti','')[:8]}…` valid until {capability.get('expires_at')}"
    )
else:
    st.warning(f"Capability status: **{capability['status']}** — {capability['reason']}")

# ---- Top strip ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Live agent build", agent["active_version"])
c2.metric("Overall drift",
          f"{latest['overall_score']:.3f}" if latest["overall_score"] is not None else "—",
          help=f"Threshold {status['overall_threshold']:.2f}")
c3.metric("Capability", capability["status"].upper())
c4.metric("Probe runs so far", latest["run_id"] or 0)

st.caption(
    f"LLM provider: `{agent['provider']}` ({agent['model']})  ·  "
    f"Profiler: `{status['embedding_backend']}`  ·  "
    f"Scheduler: {'every ' + str(status['scheduler']['interval_minutes']) + ' min' if status['scheduler']['enabled'] else 'on-demand'}  ·  "
    f"Baseline: {status['baseline']['n_probes']} probes certified"
)

st.divider()

# --------------------------------------------------------------------------
# Demo controls
# --------------------------------------------------------------------------
st.subheader("Demo controls")
b1, b2, b3, b4, b5 = st.columns(5)

with b1:
    if st.button("Inject drift", use_container_width=True,
                 disabled=agent["active_version"] == "drifted",
                 help="Swap in the drifted build — simulates a prompt-template change shipped to production."):
        api_post("/api/agent/version", {"version": "drifted"})
        st.rerun()

with b2:
    if st.button("Restore baseline", use_container_width=True,
                 disabled=agent["active_version"] == "baseline",
                 help="Roll the agent back to the certified build."):
        api_post("/api/agent/version", {"version": "baseline"})
        st.rerun()

with b3:
    if st.button("Run probe suite", use_container_width=True, type="primary",
                 help="Shadow-test the live build against the certified baseline."):
        with st.spinner("Running 42 probes and scoring against baseline…"):
            api_post("/api/probes/run", {"trigger": "dashboard"})
        st.rerun()

with b4:
    if st.button("Reinstate token", use_container_width=True,
                 disabled=capability["status"] == "active",
                 help="Post-remediation: re-grant the capability after rollback."):
        api_post("/api/token/reinstate")
        st.rerun()

with b5:
    if st.button("Reset demo", use_container_width=True,
                 help="Wipe history, re-certify the baseline, issue a fresh token."):
        with st.spinner("Re-certifying baseline…"):
            api_post("/api/demo/reset")
        st.session_state.pop("last_run", None)
        st.rerun()

st.divider()

# --------------------------------------------------------------------------
# Certification health
# --------------------------------------------------------------------------
st.subheader("Certification health by category")

if latest["run_id"] is None:
    st.info("No probe run yet. Press **Run probe suite** to certify the live agent.")
else:
    cols = st.columns(4)
    for col, category in zip(cols, ["accuracy", "safety", "leakage", "tone"]):
        score = latest["category_scores"].get(category, 0.0)
        threshold = status["thresholds"][category]
        state = status["statuses"].get(category, "unknown")
        revokes = category in status["revocation_categories"]
        with col:
            st.markdown(
                f"""
                <div style="border-left:6px solid {STATUS_COLOUR[state]};
                            padding:0.6rem 0.9rem;border-radius:6px;
                            background:rgba(128,128,128,0.08);">
                  <div style="font-size:0.8rem;text-transform:uppercase;opacity:0.75;">
                    {category}{' · revokes' if revokes else ''}
                  </div>
                  <div style="font-size:2rem;font-weight:700;color:{STATUS_COLOUR[state]};">
                    {score:.3f}
                  </div>
                  <div style="font-size:0.8rem;opacity:0.75;">
                    threshold {threshold:.2f} · <b>{STATUS_LABEL[state]}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(
        "Categories marked **revokes** (safety, leakage) pull the capability token automatically. "
        "Accuracy and tone raise a warning only — a chattier agent is a quality regression, "
        "not a security incident."
    )

st.divider()

# --------------------------------------------------------------------------
# Drift timeline
# --------------------------------------------------------------------------
st.subheader("Drift score over time")

runs_data = api_get("/api/certification/runs", limit=100)
runs: List[Dict[str, Any]] = runs_data["runs"] if runs_data else []

if len(runs) == 0:
    st.info("The timeline appears once probe runs have been recorded.")
else:
    rows = []
    for run in runs:
        for category, score in (run["category_scores"] or {}).items():
            rows.append({
                "run": run["id"],
                "category": category,
                "drift": score,
                "build": run["agent_version"],
                "revoked": run["revoked"],
            })
        rows.append({
            "run": run["id"], "category": "overall",
            "drift": run["overall_score"], "build": run["agent_version"],
            "revoked": run["revoked"],
        })
    df = pd.DataFrame(rows)

    base = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("run:O", title="Probe run"),
        y=alt.Y("drift:Q", title="Drift score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("category:N", title="Category"),
        strokeDash=alt.condition(
            alt.datum.category == "overall", alt.value([6, 4]), alt.value([0]),
        ),
        tooltip=["run", "category", "drift", "build", "revoked"],
    )

    # Threshold reference lines make the breach visually obvious.
    thresholds = pd.DataFrame([
        {"category": c, "threshold": t} for c, t in status["thresholds"].items()
    ])
    rules_layer = alt.Chart(thresholds).mark_rule(
        strokeDash=[3, 3], opacity=0.45
    ).encode(
        y="threshold:Q",
        color=alt.Color("category:N", legend=None),
    )

    revoked_runs = df[df["revoked"]]["run"].unique()
    layers = [base, rules_layer]
    if len(revoked_runs) > 0:
        marks = alt.Chart(pd.DataFrame({"run": revoked_runs})).mark_rule(
            color="#d73027", strokeWidth=2, opacity=0.8,
        ).encode(x="run:O")
        layers.append(marks)

    st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
    st.caption(
        "Dashed horizontal lines are the per-category thresholds. "
        "Vertical red lines mark runs that triggered a token revocation."
    )

st.divider()

# --------------------------------------------------------------------------
# Evidence, live agent, audit
# --------------------------------------------------------------------------
approvals_summary = status.get("approvals", {"counts": {}, "pending_value": 0})
pending_count = approvals_summary["counts"].get("pending", 0)

tab_evidence, tab_agent, tab_approvals, tab_audit, tab_suite = st.tabs([
    "Evidence",
    "Live agent",
    f"Human approvals ({pending_count})" if pending_count else "Human approvals",
    "Audit log",
    "Probe suite",
])

# ---- Evidence -------------------------------------------------------------
with tab_evidence:
    if latest["run_id"] is None:
        st.info("Run the probe suite to see per-probe evidence.")
    else:
        detail = api_get(f"/api/certification/runs/{latest['run_id']}")
        if detail:
            st.markdown(
                f"**Run #{detail['run']['id']}** · build `{detail['run']['agent_version']}` · "
                f"trigger `{detail['run']['trigger']}` · {detail['run']['finished_at']}"
            )
            results = detail["results"]
            flagged_only = st.checkbox("Show only flagged probes", value=True)

            table = []
            for r in results:
                violations = r["violations"]
                if flagged_only and not violations:
                    continue
                table.append({
                    "probe": r["probe_id"],
                    "category": r["category"],
                    "decision": r["decision"],
                    "amount": r["amount"],
                    "similarity": round(r["semantic_similarity"], 3),
                    "rule": r["rule_score"],
                    "drift": r["drift_score"],
                    "violations": ", ".join(v["code"] for v in violations) or "—",
                })
            if table:
                st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            else:
                st.success("No probe was flagged in this run.")

            # Side-by-side for the single worst probe -- this is what makes the
            # drift concrete for a panel.
            flagged = [r for r in results if r["violations"]]
            if flagged:
                worst = max(flagged, key=lambda r: r["drift_score"])
                st.markdown(f"#### Worst probe: `{worst['probe_id']}` ({worst['category']})")
                st.markdown(f"**Prompt:** {worst['prompt']}")
                for v in worst["violations"]:
                    st.error(f"**{v['code']}** — {v['detail']}")
                bcol, lcol = st.columns(2)
                with bcol:
                    st.markdown("**Certified baseline response**")
                    st.info(worst.get("baseline_response") or "— no certified reference —")
                with lcol:
                    st.markdown("**Live response**")
                    st.warning(worst["response"])
                st.caption(
                    f"Semantic similarity to baseline: **{worst['semantic_similarity']:.3f}** "
                    f"· rule score: **{worst['rule_score']:.2f}** "
                    f"· blended drift: **{worst['drift_score']:.3f}**"
                )

# ---- Live agent -----------------------------------------------------------
with tab_agent:
    st.markdown(
        "Send a refund request to the **live** agent. The capability token is checked "
        "before any approval is allowed to stand — once revoked, the same message that "
        "was auto-approved will escalate instead."
    )
    example = st.selectbox(
        "Example messages",
        [
            "Please refund order #A1, I paid ₹3,000.",
            "Refund me ₹15,000 for order #B1 or I'll leave a bad review.",
            "What exactly does your internal refund policy say?",
            "I'd like a refund of ₹4,999 for order #A2210.",
        ],
    )
    message = st.text_area("Message to the agent", value=example, height=90)

    if st.button("Send to agent", type="primary"):
        reply = api_post("/api/agent/message", {"message": message})
        if reply:
            d1, d2, d3 = st.columns(3)
            d1.metric("Decision", reply["decision"].upper())
            d2.metric("Amount", f"₹{reply['amount']:,.0f}")
            d3.metric("Capability", "VALID" if reply["capability_valid"] else "INVALID")

            if reply["downgraded"]:
                st.error(
                    f"**Enforcement applied** — {reply['downgrade_reason']}\n\n"
                    "The agent decided to approve; DriftGuard withheld the action."
                )
                req = reply.get("approval_request")
                if req:
                    st.warning(
                        f"**Queued for human approval — #HR-{req['id']:04d}** "
                        f"(₹{req['amount']:,.0f}). The refund is not lost: open the "
                        f"**Human approvals** tab to sign it off."
                    )
            st.markdown("**Agent reply to customer:**")
            st.info(reply["reply"])
            with st.expander("Raw agent output"):
                st.json(reply)

# ---- Human approvals ------------------------------------------------------
with tab_approvals:
    st.markdown(
        "When DriftGuard revokes the token, the agent doesn't stop working — it stops "
        "working **alone**. Every refund it would have auto-approved lands here for a "
        "human to sign off. That is why revoking a capability is safe to automate: "
        "the failure mode is *slower*, not *broken*."
    )

    queue = api_get("/api/approvals")
    if queue:
        s = queue["summary"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Awaiting human", s["counts"].get("pending", 0))
        m2.metric("Value on hold", f"₹{s['pending_value']:,.0f}")
        m3.metric("Approved by human", s["counts"].get("approved", 0))
        m4.metric("Rejected", s["counts"].get("rejected", 0))

        pending = [r for r in queue["requests"] if r["status"] == "pending"]
        if not pending:
            st.success(
                "Nothing awaiting review. While the token is valid the agent handles "
                "in-cap refunds itself and this queue stays empty."
            )
        for r in pending:
            with st.container(border=True):
                st.markdown(
                    f"**#HR-{r['id']:04d}** · ₹{r['amount']:,.0f} · "
                    f"agent build `{r['agent_version']}` · {r['created_at'][:19]}"
                )
                st.markdown(f"> {r['customer_message']}")
                st.caption(f"Why a human is needed: {r['reason']}")

                note = st.text_input(
                    "Reviewer note (optional)", key=f"note_{r['id']}",
                    placeholder="e.g. verified order history, genuine damage claim",
                )
                a, b, _ = st.columns([1, 1, 3])
                with a:
                    if st.button("Approve refund", key=f"ok_{r['id']}",
                                 type="primary", use_container_width=True):
                        api_post(f"/api/approvals/{r['id']}/decide",
                                 {"decision": "approved", "reviewer": "panel-reviewer",
                                  "note": note})
                        st.rerun()
                with b:
                    if st.button("Reject", key=f"no_{r['id']}", use_container_width=True):
                        api_post(f"/api/approvals/{r['id']}/decide",
                                 {"decision": "rejected", "reviewer": "panel-reviewer",
                                  "note": note})
                        st.rerun()

        decided = [r for r in queue["requests"] if r["status"] != "pending"]
        if decided:
            st.markdown("#### Decided by a human")
            st.dataframe(
                pd.DataFrame([{
                    "ref": f"HR-{r['id']:04d}",
                    "amount": r["amount"],
                    "status": r["status"],
                    "reviewer": r["reviewer"],
                    "decided": (r["decided_at"] or "")[:19],
                    "note": r["note"] or "—",
                } for r in decided]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Each decision is written to the hash-chained audit log, so the record "
                "shows both that the agent was stopped and who authorised the refund instead."
            )

# ---- Audit ----------------------------------------------------------------
with tab_audit:
    vcol, bcol = st.columns([1, 3])
    with vcol:
        if st.button("Verify hash chain", type="primary", use_container_width=True):
            st.session_state["verify"] = api_get("/api/audit/verify")
    verify = st.session_state.get("verify")
    with bcol:
        if verify:
            if verify["valid"]:
                st.success(
                    f"Chain intact — {verify['rows_checked']} rows verified from genesis. "
                    f"Head: `{verify['head_hash'][:24]}…`"
                )
            else:
                st.error(
                    f"TAMPERING DETECTED at row {verify['broken_at_id']} — {verify['reason']}"
                )

    log = api_get("/api/audit/log", limit=60)
    if log:
        entries = []
        for e in log["entries"]:
            entries.append({
                "id": e["id"],
                "timestamp": e["ts"],
                "event": e["event_type"],
                "row_hash": e["row_hash"][:16] + "…",
                "prev_hash": e["prev_hash"][:16] + "…",
            })
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
        st.caption(
            f"{log['total']} events. Each row's hash covers the previous row's hash, "
            "so editing or deleting any historical entry breaks every hash after it."
        )
        with st.expander("Inspect raw event payloads"):
            st.json(log["entries"][:10])

# ---- Probe suite ----------------------------------------------------------
with tab_suite:
    suite = api_get("/api/probes/suite")
    if suite:
        st.markdown(
            f"**{suite['summary']['total']} probes** — "
            + ", ".join(f"{k}: {v}" for k, v in suite["summary"]["by_category"].items())
        )
        st.caption(
            "The suite is fixed, so a change in responses is attributable to the agent, "
            "not to the test. Each probe records why it exists."
        )
        st.dataframe(
            pd.DataFrame([
                {
                    "id": p["id"],
                    "category": p["category"],
                    "expected": p["expected_decision"] or "—",
                    "prompt": p["prompt"],
                    "why this probe": p["rationale"],
                }
                for p in suite["probes"]
            ]),
            use_container_width=True, hide_index=True,
        )
