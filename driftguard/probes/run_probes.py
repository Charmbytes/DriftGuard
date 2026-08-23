"""
Shadow-testing harness.

Two operations:

  certify_baseline()  -- run the suite against the certified agent build and
                         store each response (plus its embedding) as the
                         reference. This is the "Certified Baseline" box in the
                         proposal's flow diagram. Done once at deploy time.

  run_probe_suite()   -- run the same fixed suite against whatever build is
                         live, score every response against the baseline, and
                         act on the verdict. This is the continuous loop.

The harness calls the agent with enforce_token=False on purpose: it measures
what the model *decided*, not what the enforcement layer let through. Measuring
post-enforcement behaviour would hide drift the moment the token was pulled.

Runnable directly:
    python -m driftguard.probes.run_probes --certify
    python -m driftguard.probes.run_probes --target drifted
"""

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .. import config
from ..agent import refund_agent
from ..audit import log_event
from ..db import get_conn, init_db
from ..scoring import drift as drift_mod
from ..scoring import embeddings, rules
from ..tokens import ensure_token, revoke_token
from .suite import CATEGORIES, get_suite


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ==========================================================================
# Baseline certification
# ==========================================================================
def certify_baseline(force: bool = False) -> Dict[str, Any]:
    """
    Establish the certified reference behaviour.

    Runs every probe against the 'baseline' build and stores the response and
    its embedding. Subsequent live runs are scored against these rows.
    """
    init_db()
    suite = get_suite()

    with get_conn() as conn:
        existing = conn.execute("SELECT COUNT(*) AS c FROM baseline_responses").fetchone()["c"]
    if existing and not force:
        return {"certified": False, "reason": "baseline already certified",
                "n_probes": existing}

    started = time.time()
    responses: List[Dict[str, Any]] = []
    for probe in suite:
        result = refund_agent.handle_message(
            probe["prompt"], version="baseline", enforce_token=False
        )
        responses.append({"probe": probe, "result": result})

    # Batch-embed for speed; None when running on the lexical fallback.
    texts = [r["result"]["reply"] for r in responses]
    vectors = embeddings.embed_many(texts)

    certified_at = _now()
    with get_conn(write=True) as conn:
        conn.execute("DELETE FROM baseline_responses")
        for i, item in enumerate(responses):
            probe, result = item["probe"], item["result"]
            vec = json.dumps(vectors[i]) if vectors else None
            conn.execute(
                "INSERT INTO baseline_responses"
                "(probe_id, category, prompt, response, decision, amount, embedding, certified_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (probe["id"], probe["category"], probe["prompt"], result["reply"],
                 result["decision"], result["amount"], vec, certified_at),
            )

    # Sanity check: the certified baseline should itself pass its own suite.
    self_check = _score_against_baseline(
        [{"probe": i["probe"], "result": i["result"]} for i in responses]
    )
    aggregated = drift_mod.aggregate(self_check)

    log_event("baseline_certified", {
        "n_probes": len(suite),
        "certified_at": certified_at,
        "provider": config.LLM_PROVIDER,
        "embedding_backend": embeddings.backend_name(),
        "self_check_overall": aggregated["overall_score"],
        "self_check_categories": aggregated["category_scores"],
    })

    return {
        "certified": True,
        "n_probes": len(suite),
        "certified_at": certified_at,
        "duration_seconds": round(time.time() - started, 2),
        "embedding_backend": embeddings.backend_name(),
        "self_check": {
            "overall_score": aggregated["overall_score"],
            "category_scores": aggregated["category_scores"],
            "note": "Baseline scored against itself; rule violations here would "
                    "mean the certified build already fails its own suite.",
        },
    }


def baseline_status() -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c, MAX(certified_at) AS at FROM baseline_responses"
        ).fetchone()
    return {"certified": row["c"] > 0, "n_probes": row["c"], "certified_at": row["at"]}


def _load_baseline() -> Dict[str, Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM baseline_responses").fetchall()
    out = {}
    for r in rows:
        d = dict(r)
        d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else None
        out[d["probe_id"]] = d
    return out


# ==========================================================================
# Scoring
# ==========================================================================
def _score_against_baseline(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Score a list of {probe, result} against the stored baseline.

    Each probe gets: semantic similarity, rule violations, blended drift score.
    """
    baseline = _load_baseline()

    # Batch-embed the live replies once.
    live_texts = [i["result"]["reply"] for i in items]
    live_vectors = embeddings.embed_many(live_texts)

    scored: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        probe, result = item["probe"], item["result"]
        base = baseline.get(probe["id"])

        if base is None:
            # No certified reference (probe added after certification):
            # score on rules only, and say so.
            similarity, backend = 1.0, "no-baseline"
            sem_drift = 0.0
        else:
            similarity, backend = embeddings.similarity(
                result["reply"],
                base["response"],
                live_vec=live_vectors[idx] if live_vectors else None,
                baseline_vec=base["embedding"],
            )
            sem_drift = embeddings.semantic_drift(similarity)

        rule_out = rules.check_probe(probe, result)
        score = drift_mod.probe_drift_score(sem_drift, rule_out["rule_score"])

        scored.append({
            "probe_id": probe["id"],
            "category": probe["category"],
            "prompt": probe["prompt"],
            "expected_decision": probe["expected_decision"],
            "response": result["reply"],
            "decision": result["decision"],
            "amount": result["amount"],
            "baseline_response": base["response"] if base else None,
            "baseline_decision": base["decision"] if base else None,
            "semantic_similarity": round(similarity, 4),
            "semantic_drift": round(sem_drift, 4),
            "rule_score": rule_out["rule_score"],
            "violations": rule_out["violations"],
            "drift_score": score,
            "latency_ms": result.get("latency_ms"),
            "embedding_backend": backend,
        })
    return scored


# ==========================================================================
# Full run
# ==========================================================================
def run_probe_suite(
    target_version: Optional[str] = None,
    trigger: str = "manual",
    category: Optional[str] = None,
    auto_revoke: bool = True,
) -> Dict[str, Any]:
    """
    Run the suite against the live (or a named) agent build and act on the result.

    Returns the full run record, including the revocation outcome.
    """
    init_db()

    base_status = baseline_status()
    if not base_status["certified"]:
        certify_baseline()

    target_version = target_version or refund_agent.get_active_version()
    suite = get_suite(category)
    started_at = _now()
    t0 = time.time()

    with get_conn(write=True) as conn:
        cur = conn.execute(
            "INSERT INTO probe_runs(started_at, agent_version, trigger, n_probes) "
            "VALUES (?, ?, ?, ?)",
            (started_at, target_version, trigger, len(suite)),
        )
        run_id = cur.lastrowid

    log_event("probe_run_started", {
        "run_id": run_id, "agent_version": target_version,
        "trigger": trigger, "n_probes": len(suite),
        "category_filter": category,
    })

    # --- execute the suite ------------------------------------------------
    items: List[Dict[str, Any]] = []
    for probe in suite:
        result = refund_agent.handle_message(
            probe["prompt"], version=target_version, enforce_token=False
        )
        items.append({"probe": probe, "result": result})

    # --- score ------------------------------------------------------------
    scored = _score_against_baseline(items)
    aggregated = drift_mod.aggregate(scored)
    explanation = drift_mod.explain(aggregated, scored)
    finished_at = _now()

    # --- persist ----------------------------------------------------------
    with get_conn(write=True) as conn:
        for s in scored:
            conn.execute(
                "INSERT INTO probe_results"
                "(run_id, probe_id, category, prompt, response, baseline_response, "
                " decision, amount, semantic_similarity, semantic_drift, rule_score, "
                " violations, drift_score, latency_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, s["probe_id"], s["category"], s["prompt"], s["response"],
                 s["baseline_response"], s["decision"], s["amount"],
                 s["semantic_similarity"], s["semantic_drift"], s["rule_score"],
                 json.dumps(s["violations"]), s["drift_score"], s["latency_ms"]),
            )
        conn.execute(
            "UPDATE probe_runs SET finished_at=?, overall_score=?, category_scores=?, "
            "breached=? WHERE id=?",
            (finished_at, aggregated["overall_score"],
             json.dumps(aggregated["category_scores"]),
             json.dumps(aggregated["breached"]), run_id),
        )

    log_event("drift_scored", {
        "run_id": run_id,
        "agent_version": target_version,
        "overall_score": aggregated["overall_score"],
        "category_scores": aggregated["category_scores"],
        "thresholds": aggregated["thresholds"],
        "breached": aggregated["breached"],
        "should_revoke": aggregated["should_revoke"],
        "summary": explanation["summary"],
    })

    # --- enforce ----------------------------------------------------------
    revocation: Dict[str, Any] = {"revoked": False}
    if aggregated["should_revoke"] and auto_revoke:
        reason = (
            f"Automatic revocation after probe run #{run_id}: "
            + "; ".join(explanation["reasons"])
        )
        revocation = revoke_token(
            reason=reason,
            evidence={
                "run_id": run_id,
                "agent_version": target_version,
                "category_scores": aggregated["category_scores"],
                "breached": aggregated["breached"],
                "top_offenders": explanation["top_offenders"],
            },
        )
        if revocation.get("revoked"):
            with get_conn(write=True) as conn:
                conn.execute("UPDATE probe_runs SET revoked=1 WHERE id=?", (run_id,))

    return {
        "run_id": run_id,
        "agent_version": target_version,
        "trigger": trigger,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(time.time() - t0, 2),
        "n_probes": len(suite),
        **aggregated,
        "explanation": explanation,
        "revocation": revocation,
        "results": scored,
    }


# ==========================================================================
# CLI
# ==========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="DriftGuard shadow-test harness")
    parser.add_argument("--certify", action="store_true",
                        help="Certify the baseline build (run this first).")
    parser.add_argument("--force", action="store_true",
                        help="Re-certify even if a baseline already exists.")
    parser.add_argument("--target", default=None, choices=["baseline", "drifted"],
                        help="Which build to test. Defaults to the live build.")
    parser.add_argument("--category", default=None, choices=CATEGORIES,
                        help="Run only one category.")
    parser.add_argument("--no-revoke", action="store_true",
                        help="Score only; do not pull the capability token.")
    args = parser.parse_args()

    init_db()
    ensure_token()

    if args.certify:
        out = certify_baseline(force=args.force)
        print(json.dumps(out, indent=2))
        return

    run = run_probe_suite(
        target_version=args.target,
        trigger="cli",
        category=args.category,
        auto_revoke=not args.no_revoke,
    )
    # Compact console summary -- the full record is in the DB.
    print(f"\nRun #{run['run_id']}  target={run['agent_version']}  "
          f"probes={run['n_probes']}  ({run['duration_seconds']}s)")
    print("-" * 62)
    for cat in CATEGORIES:
        score = run["category_scores"][cat]
        thr = run["thresholds"][cat]
        status = run["statuses"][cat].upper()
        flag = "BREACH" if cat in run["breached"] else "ok"
        print(f"  {cat:<9} {score:>6.3f}  (threshold {thr:.2f})  {status:<6} {flag}")
    print("-" * 62)
    print(f"  overall   {run['overall_score']:>6.3f}  (threshold {run['overall_threshold']:.2f})")
    print(f"\n  {run['explanation']['summary']}")
    if run["revocation"].get("revoked"):
        print(f"\n  ** CAPABILITY TOKEN REVOKED ** jti={run['revocation']['jti']}")
    print()


if __name__ == "__main__":
    main()
