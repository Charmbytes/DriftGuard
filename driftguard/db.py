"""
SQLite persistence layer.

Deliberately plain `sqlite3` rather than an ORM: the schema is small and the
panel can read exactly what is stored. WAL mode is enabled so the API thread
and the APScheduler thread can write concurrently without locking each other.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from . import config

# One lock guards writes. SQLite handles concurrent readers fine under WAL,
# but serialising writers keeps the audit hash-chain strictly ordered.
_write_lock = threading.Lock()

SCHEMA = """
-- Every certification run (one full sweep of the probe suite).
CREATE TABLE IF NOT EXISTS probe_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    agent_version   TEXT    NOT NULL,   -- 'baseline' | 'drifted'
    trigger         TEXT    NOT NULL,   -- 'manual' | 'scheduled' | 'baseline-certification'
    overall_score   REAL,
    category_scores TEXT,               -- JSON {category: score}
    breached        TEXT,               -- JSON [category, ...]
    revoked         INTEGER DEFAULT 0,
    n_probes        INTEGER DEFAULT 0
);

-- One row per probe per run: the raw evidence behind a score.
CREATE TABLE IF NOT EXISTS probe_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            INTEGER NOT NULL,
    probe_id          TEXT    NOT NULL,
    category          TEXT    NOT NULL,
    prompt            TEXT    NOT NULL,
    response          TEXT    NOT NULL,
    baseline_response TEXT,               -- the certified reference, for side-by-side
    decision          TEXT,
    amount            REAL,
    semantic_similarity REAL,
    semantic_drift    REAL,
    rule_score        REAL,
    violations        TEXT,             -- JSON [ {code, detail}, ... ]
    drift_score       REAL,
    latency_ms        REAL,
    FOREIGN KEY (run_id) REFERENCES probe_runs(id)
);

-- The certified baseline: the approved reference response for each probe.
CREATE TABLE IF NOT EXISTS baseline_responses (
    probe_id    TEXT PRIMARY KEY,
    category    TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    response    TEXT NOT NULL,
    decision    TEXT,
    amount      REAL,
    embedding   TEXT,                   -- JSON list[float], cached
    certified_at TEXT NOT NULL
);

-- Capability tokens issued to the agent.
CREATE TABLE IF NOT EXISTS capability_tokens (
    jti            TEXT PRIMARY KEY,
    agent_id       TEXT NOT NULL,
    scope          TEXT NOT NULL,
    token          TEXT NOT NULL,
    issued_at      TEXT NOT NULL,
    expires_at     TEXT NOT NULL,
    status         TEXT NOT NULL,       -- 'active' | 'revoked'
    revoked_at     TEXT,
    revoked_reason TEXT
);

-- Tamper-evident audit trail (hash chain).
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload    TEXT NOT NULL,           -- canonical JSON
    prev_hash  TEXT NOT NULL,
    row_hash   TEXT NOT NULL
);

-- Refunds the agent wanted to approve but was not allowed to, awaiting a
-- human reviewer. This is what the capability revocation redirects work into.
CREATE TABLE IF NOT EXISTS approval_requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL,
    customer_message TEXT    NOT NULL,
    amount           REAL    NOT NULL,
    agent_version    TEXT    NOT NULL,
    agent_decision   TEXT    NOT NULL,   -- what the agent wanted to do
    reason           TEXT    NOT NULL,   -- why a human is needed
    status           TEXT    NOT NULL,   -- 'pending' | 'approved' | 'rejected'
    reviewer         TEXT,
    decided_at       TEXT,
    note             TEXT
);

-- Small key/value store for runtime state (e.g. which agent version is live).
CREATE TABLE IF NOT EXISTS runtime_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_run ON probe_results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON probe_runs(started_at);
"""


def connect() -> sqlite3.Connection:
    """Open a connection with row-dict access and WAL enabled."""
    conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_conn(write: bool = False) -> Iterator[sqlite3.Connection]:
    """Context manager yielding a connection; commits on clean exit."""
    if write:
        _write_lock.acquire()
    conn = connect()
    try:
        yield conn
        if write:
            conn.commit()
    except Exception:
        if write:
            conn.rollback()
        raise
    finally:
        conn.close()
        if write:
            _write_lock.release()


def init_db() -> None:
    """Create tables if they do not exist. Safe to call repeatedly."""
    with get_conn(write=True) as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# runtime_state helpers
# --------------------------------------------------------------------------
def set_state(key: str, value: Any) -> None:
    with get_conn(write=True) as conn:
        conn.execute(
            "INSERT INTO runtime_state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )


def get_state(key: str, default: Any = None) -> Any:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM runtime_state WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def reset_all() -> None:
    """Wipe every table -- used by the demo reset endpoint."""
    with get_conn(write=True) as conn:
        for table in (
            "probe_results",
            "probe_runs",
            "baseline_responses",
            "capability_tokens",
            "approval_requests",
            "audit_log",
            "runtime_state",
        ):
            conn.execute(f"DELETE FROM {table}")
        # sqlite_sequence is created lazily on the first AUTOINCREMENT insert,
        # so it may not exist yet on a database that has never been written to.
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone()
        if exists:
            conn.execute("DELETE FROM sqlite_sequence")
