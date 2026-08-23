"""
Tamper-evident audit log.

Every meaningful event (probe run, score computed, token issued, token
revoked, agent version swapped) is appended to `audit_log` as a link in a
SHA-256 hash chain:

    row_hash[i] = sha256( prev_hash[i]  ||  canonical_json(row_data[i]) )
    prev_hash[i] = row_hash[i-1]        (genesis prev_hash = 64 zeroes)

Because each hash covers the previous one, editing or deleting any historical
row invalidates every hash after it. `verify_chain()` walks the whole table and
reports the first index where the recomputed hash stops matching.

Note (see README "Known limitations"): this detects tampering by anyone who
edits rows without recomputing the chain. A determined attacker with write
access to the whole DB could recompute the entire chain. Production hardening
would anchor the head hash externally (e.g. periodic notarisation) or sign each
row with a key the DB host does not hold.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .db import get_conn

GENESIS_HASH = "0" * 64


def _canonical(payload: Dict[str, Any]) -> str:
    """Deterministic JSON so the same data always hashes identically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_row_hash(prev_hash: str, ts: str, event_type: str, payload_json: str) -> str:
    """The chain link function. Kept tiny and pure so it is easy to audit."""
    material = f"{prev_hash}|{ts}|{event_type}|{payload_json}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def log_event(event_type: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Append one event to the chain. Returns the stored row."""
    payload = payload or {}
    ts = datetime.now(timezone.utc).isoformat()
    payload_json = _canonical(payload)

    with get_conn(write=True) as conn:
        row = conn.execute("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = row["row_hash"] if row else GENESIS_HASH
        row_hash = compute_row_hash(prev_hash, ts, event_type, payload_json)
        cur = conn.execute(
            "INSERT INTO audit_log(ts, event_type, payload, prev_hash, row_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, event_type, payload_json, prev_hash, row_hash),
        )
        new_id = cur.lastrowid

    return {
        "id": new_id,
        "ts": ts,
        "event_type": event_type,
        "payload": payload,
        "prev_hash": prev_hash,
        "row_hash": row_hash,
    }


def verify_chain() -> Dict[str, Any]:
    """
    Recompute every hash from the genesis link and report integrity.

    Returns a dict with `valid`, `rows_checked`, and -- when broken -- the id
    of the first bad row plus the expected/actual hashes, so the dashboard can
    show precisely where the log was altered.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, ts, event_type, payload, prev_hash, row_hash FROM audit_log ORDER BY id ASC"
        ).fetchall()

    prev_hash = GENESIS_HASH
    for row in rows:
        # Link check: does this row point at the previous row's hash?
        if row["prev_hash"] != prev_hash:
            return {
                "valid": False,
                "rows_checked": len(rows),
                "broken_at_id": row["id"],
                "reason": "prev_hash does not match the preceding row_hash "
                          "(a row was deleted, reordered, or inserted)",
                "expected_prev_hash": prev_hash,
                "actual_prev_hash": row["prev_hash"],
            }
        # Content check: does the stored hash match the stored data?
        expected = compute_row_hash(prev_hash, row["ts"], row["event_type"], row["payload"])
        if expected != row["row_hash"]:
            return {
                "valid": False,
                "rows_checked": len(rows),
                "broken_at_id": row["id"],
                "reason": "row_hash does not match row contents (this row was edited)",
                "expected_row_hash": expected,
                "actual_row_hash": row["row_hash"],
            }
        prev_hash = row["row_hash"]

    return {
        "valid": True,
        "rows_checked": len(rows),
        "head_hash": prev_hash,
        "reason": "All rows hash-verified from genesis.",
    }


def get_log(limit: int = 50, offset: int = 0, event_type: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Paginated read of the audit trail, newest first."""
    where, params = "", []
    if event_type:
        where = "WHERE event_type = ?"
        params.append(event_type)

    with get_conn() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS c FROM audit_log {where}", params).fetchone()["c"]
        rows = conn.execute(
            f"SELECT id, ts, event_type, payload, prev_hash, row_hash FROM audit_log {where} "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()

    entries = []
    for r in rows:
        d = dict(r)
        try:
            d["payload"] = json.loads(d["payload"])
        except json.JSONDecodeError:
            pass
        entries.append(d)
    return entries, total
