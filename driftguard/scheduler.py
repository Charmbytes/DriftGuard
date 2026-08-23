"""
Periodic re-certification loop.

An APScheduler background job re-runs the probe suite every
PROBE_INTERVAL_MINUTES. This is the "continuous" in continuous
re-certification: nobody has to remember to press a button.

For the live demo the scheduler is off by default (ENABLE_SCHEDULER=false) and
runs are triggered from the dashboard, because a 15-minute tick is not
watchable in a viva. Set ENABLE_SCHEDULER=true and
PROBE_INTERVAL_MINUTES=1 to show the unattended loop working.
"""

import logging
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

from . import config
from .audit import log_event

logger = logging.getLogger("driftguard.scheduler")

_scheduler: Optional[BackgroundScheduler] = None


def _scheduled_run() -> None:
    """One unattended re-certification cycle."""
    from .probes.run_probes import run_probe_suite  # local import avoids a cycle

    try:
        run = run_probe_suite(trigger="scheduled")
        logger.info(
            "Scheduled probe run #%s: overall=%.3f breached=%s revoked=%s",
            run["run_id"], run["overall_score"], run["breached"],
            run["revocation"].get("revoked", False),
        )
    except Exception as exc:
        logger.exception("Scheduled probe run failed")
        log_event("scheduler_error", {"error": f"{type(exc).__name__}: {exc}"})


def start_scheduler() -> Optional[BackgroundScheduler]:
    """Start the loop if enabled. Idempotent."""
    global _scheduler
    if not config.ENABLE_SCHEDULER:
        logger.info("Scheduler disabled (ENABLE_SCHEDULER=false); runs are on-demand.")
        return None
    if _scheduler is not None:
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _scheduled_run,
        "interval",
        minutes=config.PROBE_INTERVAL_MINUTES,
        id="recertification",
        max_instances=1,       # never let two runs overlap
        coalesce=True,         # if we fell behind, run once, not N times
    )
    _scheduler.start()
    logger.info("Scheduler started: probe suite every %s minute(s).",
                config.PROBE_INTERVAL_MINUTES)
    log_event("scheduler_started", {"interval_minutes": config.PROBE_INTERVAL_MINUTES})
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        log_event("scheduler_stopped", {})
