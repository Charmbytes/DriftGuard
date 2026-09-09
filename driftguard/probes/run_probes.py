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
from ..scoring import embeddings, rules, sequential
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


def _run_history() -> List[Dict[str, Any]]:
    """
    Completed runs, oldest first -- the input CUSUM accumulates over.

    Scoped to the *current capability period*: only runs since the active token
    was issued. This matters. CUSUM is a sequential test over one process, and
    a reinstatement means the agent was remediated and re-certified -- it is
    a different process from here on. Carrying the old accumulator across a
    reinstatement would re-revoke a freshly fixed agent within a run or two,
    for drift it no longer has.

    Only build and category scores are needed, so this stays cheap.
    """
    from ..tokens import get_latest_token  # local import avoids a cycle

    token = get_latest_token()
    since = token["issued_at"] if token else None

    with get_conn() as conn:
        if since:
            rows = conn.execute(
                "SELECT agent_version, category_scores FROM probe_runs "
                "WHERE finished_at IS NOT NULL AND started_at >= ? ORDER BY id ASC",
                (since,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT agent_version, category_scores FROM probe_runs "
                "WHERE finished_at IS NOT NULL ORDER BY id ASC"
            ).fetchall()
    history = []
    for r in rows:
        history.append({
            "agent_version": r["agent_version"],
            "category_scores": json.loads(r["category_scores"] or "{}"),
        })
    return history


def _baseline_probe_scores() -> List[float]:
    """
    Per-probe drift scores from the most recent run of the certified build.

    This is PSI's "expected" distribution. Falls back to an all-zero
    distribution matching the suite size when no baseline run exists yet,
    which is what a perfectly-certified agent would produce anyway.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM probe_runs WHERE agent_version='baseline' "
            "AND finished_at IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return [0.0] * len(get_suite())
        scores = conn.execute(
            "SELECT drift_score FROM probe_results WHERE run_id=?", (row["id"],)
        ).fetchall()
    return [s["drift_score"] for s in scores] or [0.0] * len(get_suite())


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

    # --- sequential detection over the run history ------------------------
    # Fixed thresholds judge this run alone. CUSUM reads every run before it,
    # which is the only way to catch an agent that drifts a little each cycle
    # without ever tripping a single-run threshold.
    history = _run_history()
    history.append({
        "agent_version": target_version,
        "category_scores": aggregated["category_scores"],
    })
    cusum = sequential.evaluate(history, CATEGORIES)

    # CUSUM needs a few runs before it is allowed to act, so one noisy early
    # run cannot pull a capability before any baseline history exists.
    cusum_may_enforce = (
        config.CUSUM_ENFORCES and len(history) >= config.CUSUM_MIN_RUNS
    )

    psi_result = sequential.psi(
        expected=_baseline_probe_scores(),
        actual=[s["drift_score"] for s in scored],
    )

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
            "breached=?, cusum_scores=?, cusum_alarming=?, psi_score=? WHERE id=?",
            (finished_at, aggregated["overall_score"],
             json.dumps(aggregated["category_scores"]),
             json.dumps(aggregated["breached"]),
             json.dumps({c: v["value"] for c, v in cusum["per_category"].items()}),
             json.dumps(cusum["alarming"]),
             psi_result["psi"], run_id),
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
    # Two independent detectors can pull the token. The threshold detector
    # catches a sudden collapse; CUSUM catches a slow slide that no single run
    # would flag. Whichever fires, the reason recorded says which one it was.
    cusum_triggers = cusum["revoke_categories"] if cusum_may_enforce else []
    should_revoke = aggregated["should_revoke"] or bool(cusum_triggers)

    if aggregated["should_revoke"]:
        detector = "threshold"
        reason_detail = "; ".join(explanation["reasons"])
    elif cusum_triggers:
        detector = "cusum"
        reason_detail = "; ".join(
            f"{c}: CUSUM {cusum['per_category'][c]['value']:.2f} exceeds "
            f"{cusum['per_category'][c]['threshold']:.2f} accumulated over "
            f"{cusum['per_category'][c]['n_runs']} runs "
            f"(no single run breached its {config.CATEGORY_THRESHOLDS[c]:.2f} threshold)"
            for c in cusum_triggers
        )
    else:
        detector = None
        reason_detail = ""

    revocation: Dict[str, Any] = {"revoked": False}
    if should_revoke and auto_revoke:
        reason = f"Automatic revocation after probe run #{run_id} [{detector}]: {reason_detail}"
        revocation = revoke_token(
            reason=reason,
            evidence={
                "run_id": run_id,
                "agent_version": target_version,
                "detector": detector,
                "category_scores": aggregated["category_scores"],
                "breached": aggregated["breached"],
                "cusum": {c: v["value"] for c, v in cusum["per_category"].items()},
                "cusum_alarming": cusum["alarming"],
                "psi": psi_result["psi"],
                "top_offenders": explanation["top_offenders"],
            },
        )
        if revocation.get("revoked"):
            with get_conn(write=True) as conn:
                conn.execute("UPDATE probe_runs SET revoked=1, detector=? WHERE id=?",
                             (detector, run_id))

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
        "cusum": cusum,
        "cusum_enforcing": cusum_may_enforce,
        "psi": psi_result,
        "detector": detector,
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
